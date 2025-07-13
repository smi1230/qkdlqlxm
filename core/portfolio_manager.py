# /core/portfolio_manager.py

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

# 시스템의 다른 핵심 모듈을 임포트합니다.
try:
    from core.bybit_client import BybitClient
    from configs import settings
except ImportError:
    print("[CRITICAL ERROR] Failed to import modules in portfolio_manager.py.")
    raise SystemExit("Module loading failed.")


logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    [ v5.1 - 실시간 자산 연동 강화 ]
    포지션 상태와 자본금의 유일한 진실 공급원(Single Source of Truth).
    시스템의 모든 자산과 포지션을 중앙에서 추적, 관리하고 리스크 정책을 집행합니다.
    """

    def __init__(self, bybit_client: BybitClient):
        self.client = bybit_client
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.user_capital: Optional[float] = None
        self.daily_realised_pnl: float = 0.0
        self.last_pnl_reset_date: Optional[datetime.date] = None

        logger.info("PortfolioManager initialized. Ready to manage assets and positions.")
        self.update_positions()

    # =================================================================
    # ✨ 자본금 및 자산 조회 관리 (Capital & Asset Inquiry Management)
    # =================================================================
    
    def set_user_capital(self, capital: float) -> bool:
        """사용자가 지정한 운용 자본금을 설정합니다."""
        if capital < 0:
            logger.error(f"Invalid capital amount: {capital}. Must be non-negative.")
            return False
        self.user_capital = float(capital)
        logger.info(f"User-defined operating capital set to: ${self.user_capital:,.2f} USDT")
        return True

    def get_wallet_balance(self) -> Optional[Dict[str, Any]]:
        """ ✨ Bybit에서 직접 지갑 잔고 정보를 가져옵니다. """
        try:
            # accountType을 'UNIFIED'로 명시하여 통합거래계좌 잔고를 조회합니다.
            response = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if response and response.get('retCode') == 0 and response['result']['list']:
                return response['result']['list'][0]
            logger.warning(f"Could not retrieve wallet balance from API. Response: {response}")
            return None
        except Exception as e:
            logger.error(f"Exception while fetching wallet balance: {e}", exc_info=True)
            return None

    def get_total_capital(self) -> float:
        """설정된 총 운용 자본금을 반환합니다."""
        if self.user_capital is None:
            return 0.0
        return self.user_capital

    def get_available_capital(self) -> float:
        """
        현재 사용 가능한 자본금을 반환합니다 (총 자본금 - 현재 포지션에 사용된 자본금).
        """
        total_capital = self.get_total_capital()
        if total_capital <= 0: return 0.0
            
        used_capital = sum(pos.get('positionValue', 0) for pos in self.positions.values())
        available = total_capital - used_capital
        
        logger.debug(f"Capital Check - Total Defined: ${total_capital:,.2f}, Used: ${used_capital:,.2f}, Available: ${available:,.2f}")
        return max(0.0, available)

    # =================================================================
    # 포지션 관리 (Position Management)
    # =================================================================

    def update_positions(self) -> bool:
        """
        [핵심 동기화] Bybit API로부터 최신 포지션 정보를 가져와 내부 상태를 완벽하게 동기화합니다.
        """
        logger.debug("Synchronizing positions with the exchange...")
        try:
            response = self.client.get_positions(category="linear", settleCoin="USDT")
            if not response or response.get('retCode') != 0:
                logger.error(f"Failed to fetch positions from API. Response: {response}")
                return False

            live_positions_on_exchange = {
                item['symbol']: self._parse_position_data(item)
                for item in response['result']['list']
                if float(item.get('size', 0)) > 0
            }
            
            live_symbols = set(live_positions_on_exchange.keys())
            local_symbols = set(self.positions.keys())

            for symbol in live_symbols - local_symbols:
                self.positions[symbol] = live_positions_on_exchange[symbol]
                self.positions[symbol]['createdTime'] = datetime.now(timezone.utc)
                logger.info(f"✅ New position detected: {symbol} | Side: {self.positions[symbol]['side']} | Size: {self.positions[symbol]['size']}")

            for symbol in local_symbols.intersection(live_symbols):
                created_time = self.positions[symbol].get('createdTime')
                self.positions[symbol] = live_positions_on_exchange[symbol]
                self.positions[symbol]['createdTime'] = created_time

            for symbol in local_symbols - live_symbols:
                closed_pnl = self.positions[symbol].get('realisedPnl', 0)
                self.update_daily_pnl(closed_pnl)
                logger.info(f"❌ Position for {symbol} has been closed. Realised PnL for this trade: ${closed_pnl:,.4f}")
                del self.positions[symbol]

            logger.debug(f"Position sync complete. Current open positions: {len(self.positions)}")
            return True

        except Exception as e:
            logger.error(f"An exception occurred during position update: {e}", exc_info=True)
            return False

    def _parse_position_data(self, pos_data: Dict) -> Dict:
        """API 응답을 내부 포맷으로 변환하고, 필요한 값을 미리 계산합니다."""
        try:
            size_val = float(pos_data.get('size', 0))
            avg_price_val = float(pos_data.get('avgPrice', 0))
            return {
                'symbol': pos_data.get('symbol'),
                'side': pos_data.get('side'),
                'size': size_val,
                'avgPrice': avg_price_val,
                'positionValue': float(pos_data.get('positionValue', 0)),
                'unrealisedPnl': float(pos_data.get('unrealisedPnl', 0)),
                'realisedPnl': float(pos_data.get('realisedPnl', 0)),
                'markPrice': float(pos_data.get('markPrice', 0)),
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing position data for {pos_data.get('symbol', 'N/A')}: {e}")
            return {}
            
    def get_all_positions(self) -> Dict[str, Dict]:
        """현재 관리 중인 모든 포지션 정보를 반환합니다."""
        return self.positions

    # ... (기존 should_rebalance_position 및 리스크 정책 집행 함수들)
    def should_rebalance_position(self, symbol: str, target_side: str, target_size: float, threshold: float) -> bool:
        if symbol not in self.positions: return True
        current_pos = self.positions[symbol]
        if current_pos['side'].upper() != target_side.upper(): return True
        try:
            size_change_pct = abs(current_pos['size'] - target_size) / current_pos['size']
            return size_change_pct > threshold
        except ZeroDivisionError:
             return True

    def get_positions_to_liquidate_by_time(self) -> List[Dict]:
        positions_to_close = []
        now = datetime.now(timezone.utc)
        max_hold_delta = timedelta(minutes=settings.POSITION_MAX_HOLD_MINUTES)
        for symbol, pos_details in self.positions.items():
            created_time = pos_details.get('createdTime')
            if created_time and (now - created_time > max_hold_delta):
                logger.warning(f"🚨 FORCED EXIT TRIGGERED: {symbol} has exceeded max holding time.")
                positions_to_close.append({'symbol': symbol, 'reason': 'max_hold_time_exceeded'})
        return positions_to_close

    def update_daily_pnl(self, realised_pnl_from_trade: float):
        today = datetime.now(timezone.utc).date()
        if self.last_pnl_reset_date != today:
            logger.info(f"New day. Resetting daily PnL from {self.daily_realised_pnl:,.4f} to 0.")
            self.daily_realised_pnl = 0
            self.last_pnl_reset_date = today
        self.daily_realised_pnl += realised_pnl_from_trade
        logger.info(f"Daily realised PnL updated to: ${self.daily_realised_pnl:,.4f}")

    def has_breached_daily_loss_limit(self) -> bool:
        total_capital = self.get_total_capital()
        if total_capital <= 0: return False
        current_unrealised_pnl = sum(p.get('unrealisedPnl', 0) for p in self.positions.values())
        total_daily_pnl = self.daily_realised_pnl + current_unrealised_pnl
        loss_percentage = (total_daily_pnl / total_capital) * 100 if total_capital > 0 else 0
        if loss_percentage < settings.DAILY_LOSS_LIMIT_PERCENT:
            logger.critical(f"🚨 DAILY LOSS LIMIT BREACHED! Current Loss: {loss_percentage:.2f}%")
            return True
        return False

    def print_portfolio_status(self):
        """ ✨ 개선된 포트폴리오 상태를 콘솔에 출력합니다. """
        print("\n" + "---" * 15)
        print(f"PORTFOLIO STATUS @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("---" * 15)
        
        # 실제 계좌 잔고를 직접 조회
        wallet_balance = self.get_wallet_balance()
        if wallet_balance:
            total_equity = float(wallet_balance.get('totalEquity', 0))
            available_to_withdraw = float(wallet_balance.get('totalAvailableBalance', 0))
            total_unrealised_pnl = float(wallet_balance.get('totalUnrealisedPnl', 0))
            
            print(f"💰 Account Balance (from API):")
            print(f"  - Total Equity: ${total_equity:,.4f} USDT")
            print(f"  - Available Balance: ${available_to_withdraw:,.4f} USDT")
            print(f"  - Total Unrealised PnL: ${total_unrealised_pnl:,.4f} USDT")
        else:
            print("Could not fetch live account balance from API.")

        # 시스템이 사용하는 운용 자본금 정보
        defined_capital = self.get_total_capital()
        available_for_trades = self.get_available_capital()
        used_capital = defined_capital - available_for_trades
        
        print(f"\n📈 Trading Capital (System):")
        print(f"  - Defined for Trading: ${defined_capital:,.2f}")
        print(f"  - Used in Positions: ${used_capital:,.2f}")
        print(f"  - Available for New Trades: ${available_for_trades:,.2f}")
        
        if not self.positions:
            print("\n🛡️ No open positions.")
        else:
            print(f"\n🛡️ Open Positions ({len(self.positions)}):")
            for symbol, details in self.positions.items():
                pnl = details.get('unrealisedPnl', 0)
                pnl_color = "\033[92m" if pnl >= 0 else "\033[91m"
                reset_color = "\033[0m"
                
                print(f"  ▶ {symbol} ({details.get('side', 'N/A')}) | Size: {details.get('size', 0)} "
                      f"| Value: ${details.get('positionValue', 0):,.2f}")
                print(f"    Entry: ${details.get('avgPrice', 0):,.4f} | Mark: ${details.get('markPrice', 0):,.4f} "
                      f"| PnL: {pnl_color}${pnl:,.4f}{reset_color}")
        print("---" * 15 + "\n")
