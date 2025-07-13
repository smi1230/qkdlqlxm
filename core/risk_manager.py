# /core/risk_manager.py 테스트넷용

import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# 시스템의 다른 핵심 모듈을 임포트합니다.
try:
    from core.bybit_client import BybitClient
    from core.portfolio_manager import PortfolioManager
    from configs import settings
except ImportError:
    print("[CRITICAL ERROR] Failed to import modules in risk_manager.py.")
    raise SystemExit("Module loading failed.")

logger = logging.getLogger(__name__)

class RiskManager:
    """
    [ v5.3 - 가격 결정 로직 안정화 ]
    - 호가 정보가 없는 경우, lastPrice를 사용하여 계산을 계속 진행하도록 수정.
    - 자본 배분 로직의 가독성 개선.
    """

    def __init__(self, client: BybitClient):
        self.client = client
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.max_allocation = settings.MAX_SINGLE_ALLOCATION_RATIO
        self.min_allocation = settings.MIN_SINGLE_ALLOCATION_RATIO
        self.volatility_factor = settings.VOLATILITY_ADJUSTMENT_FACTOR
        self.alignment_bonus = settings.SIGNAL_ALIGNMENT_BONUS
        self.sl_atr_multiplier = settings.SL_ATR_MULTIPLIER
        self.tp1_ratio = settings.TP1_REWARD_RATIO
        self.min_order_value_buffer = 1.02 # 최소 주문 금액에 2% 버퍼 추가
        logger.info("RiskManager initialized with v5.3 robust price selection logic.")

    def set_portfolio_manager(self, portfolio_manager: PortfolioManager):
        self.portfolio_manager = portfolio_manager
        logger.info("PortfolioManager injected into RiskManager for dynamic capital access.")

    def get_current_capital(self) -> float:
        if self.portfolio_manager is None:
            logger.error("PortfolioManager is not set. Cannot retrieve capital. Returning 0.")
            return 0.0
        return self.portfolio_manager.get_available_capital()

    def calculate_dynamic_position_sizes(self, top_opportunities: List[Dict]) -> Dict[str, Dict]:
        """
        [✨ 수정] 호가 정보가 없는 경우를 처리하는 안정적인 로직으로 TP/SL 및 포지션 사이즈를 계산합니다.
        """
        final_positions = {}
        if not top_opportunities:
            return final_positions

        total_capital_for_cycle = self.get_current_capital()
        if total_capital_for_cycle <= 0:
            return final_positions

        logger.info(f"Calculating dynamic position sizes for {len(top_opportunities)} opportunities...")

        total_abs_score = sum(abs(opp['final_composite_score']) for opp in top_opportunities)
        if total_abs_score == 0:
            total_abs_score = len(top_opportunities) if top_opportunities else 1
            base_weight_func = lambda score: 1.0
        else:
            base_weight_func = lambda score: abs(score)

        for opp in top_opportunities:
            base_weight = base_weight_func(opp['final_composite_score']) / total_abs_score
            dl_details = opp['dl_score_details']
            predicted_volatility = dl_details.get('volatility', 0.5)
            volatility_adjustment = 1.0 - (predicted_volatility * self.volatility_factor)
            alignment_adjustment = self.alignment_bonus if np.sign(dl_details['risk_adjusted_score']) == np.sign(opp['technical_score']) else 1.0
            opp['adjusted_weight'] = base_weight * volatility_adjustment * alignment_adjustment

        total_adjusted_weight = sum(opp['adjusted_weight'] for opp in top_opportunities)
        if total_adjusted_weight == 0:
            return final_positions

        for opp in top_opportunities:
            symbol = opp['symbol']
            
            ticker_info_res = self.client.get_tickers(category="linear", symbol=symbol)
            if not ticker_info_res or ticker_info_res.get('retCode') != 0 or not ticker_info_res['result']['list']:
                logger.error(f"[{symbol}] Could not get ticker info for risk calculation. Skipping.")
                continue
            
            ticker_data = ticker_info_res['result']['list'][0]
            try:
                last_price = float(ticker_data['lastPrice'])
                bid_price = float(ticker_data.get('bid1Price', 0.0)) # .get()으로 안전하게 접근
                ask_price = float(ticker_data.get('ask1Price', 0.0)) # .get()으로 안전하게 접근
                if last_price <= 0:
                    raise ValueError("Invalid lastPrice data")
            except (KeyError, ValueError) as e:
                logger.error(f"[{symbol}] Invalid ticker data for risk calculation: {e}. Skipping.")
                continue

            # ✨ [핵심 수정] 자본 배분 비율을 5% ~ 15% 사이로 제한하는 명확한 로직
            normalized_allocation = opp['adjusted_weight'] / total_adjusted_weight
            clamped_allocation = min(max(normalized_allocation, self.min_allocation), self.max_allocation)
            
            capital_for_position = total_capital_for_cycle * clamped_allocation
            
            min_order_value = 5.0
            if capital_for_position < (min_order_value * self.min_order_value_buffer):
                capital_for_position = min_order_value * self.min_order_value_buffer
                logger.debug(f"[{symbol}] Capital allocation adjusted to ${capital_for_position:,.2f} to meet min order value.")

            quantity = capital_for_position / last_price
            atr = opp['latest_features'].get(f'atr_{settings.ATR_PERIOD}', last_price * 0.01)

            # ✨ [핵심 수정] 호가 정보가 유효하면 사용하고, 아니면 lastPrice로 대체
            if bid_price > 0 and ask_price > 0:
                pessimistic_entry_price = ask_price if opp['side'].upper() == 'BUY' else bid_price
            else:
                logger.warning(f"[{symbol}] Zero bid/ask price detected. Falling back to lastPrice for TP/SL calculation.")
                pessimistic_entry_price = last_price
            
            stop_loss_price, take_profit_price = self.calculate_atr_based_sl_tp(
                entry_price=pessimistic_entry_price,
                atr=atr,
                side=opp['side']
            )

            final_positions[symbol] = {
                'side': opp['side'],
                'size_in_usd': round(capital_for_position, 2),
                'quantity': quantity,
                'entry_price': last_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
            }
            logger.info(f"[{symbol}] Position Sized: {opp['side']} | Capital: ${capital_for_position:,.2f} | "
                        f"Qty: {quantity:.6f} | SL: ${stop_loss_price:,.4f} | TP: ${take_profit_price:,.4f}")

        return final_positions
    
    def _calculate_stop_loss(self, side: str, entry_price: float, atr: float) -> float:
        sl_distance = atr * self.sl_atr_multiplier
        return entry_price - sl_distance if side.upper() == 'BUY' else entry_price + sl_distance

    def calculate_atr_based_sl_tp(self, entry_price: float, atr: float, side: str) -> Tuple[float, float]:
        """ATR 기반으로 손절(SL) 및 익절(TP) 가격을 계산하여 튜플로 반환합니다."""
        try:
            stop_loss = self._calculate_stop_loss(side, entry_price, atr)
            sl_distance = abs(entry_price - stop_loss)
            
            if side.upper() == 'BUY':
                take_profit = entry_price + (sl_distance * self.tp1_ratio)
            else: # 'Sell'
                take_profit = entry_price - (sl_distance * self.tp1_ratio)
            
            return stop_loss, take_profit
        except Exception as e:
            logger.error(f"Error calculating ATR-based SL/TP: {e}")
            return entry_price, entry_price

    def validate_position(self, symbol: str, quantity: float, capital_allocated: float) -> bool:
        """계산된 포지션이 진입하기에 합리적인지 최종 검증합니다."""
        try:
            if quantity <= 0:
                logger.warning(f"[{symbol}] Invalid quantity for validation: {quantity}")
                return False
            
            if self.portfolio_manager is None:
                logger.error("PortfolioManager not set in RiskManager, cannot validate position.")
                return False
                
            total_capital = self.portfolio_manager.get_total_capital()
            if total_capital > 0 and (capital_allocated / total_capital) > (self.max_allocation * 1.5):
                logger.warning(f"[{symbol}] Allocated capital (${capital_allocated:,.2f}) "
                               f"exceeds safety limit. Position rejected.")
                return False
            
            return True
        except Exception as e:
            logger.error(f"[{symbol}] Error validating position: {e}")
            return False
