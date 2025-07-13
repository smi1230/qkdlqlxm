# /strategy/main_strategy.py

import logging
from typing import List, Dict, Optional
import numpy as np

# 시스템의 다른 핵심 모듈을 임포트합니다.
try:
    from core.portfolio_manager import PortfolioManager
    from strategy.opportunity_ranker import OpportunityRanker
    from core.risk_manager import RiskManager
    from configs import settings
except ImportError:
    print("[CRITICAL ERROR] Failed to import modules in main_strategy.py.")
    raise SystemExit("Module loading failed.")


logger = logging.getLogger(__name__)

class MainStrategy:
    """
    [ v5.1 - 지능형 손절 및 출구 전략 통합 ]
    시스템의 거래 결정을 총괄하는 최상위 전략 클래스.
    """

    def __init__(self,
                 opportunity_ranker: OpportunityRanker,
                 portfolio_manager: PortfolioManager,
                 risk_manager: RiskManager):
        self.opportunity_ranker = opportunity_ranker
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.strategy_mode = settings.STRATEGY_MODE
        self.rebalance_threshold = settings.POSITION_CHANGE_THRESHOLD
        self.force_rebalance_cycles = settings.FORCED_REBALANCE_CYCLES
        self.cycle_count = 0
        logger.info(f"MainStrategy initialized. Mode: '{self.strategy_mode}', Intelligent Loss Cut: {settings.USE_INTELLIGENT_LOSS_CUT}")

    def generate_signals(self, tradeable_symbols: list[str]) -> Optional[List[Dict]]:
        if self.strategy_mode == 'AI_HYBRID_REBALANCE':
            return self._generate_smart_rebalance_signals(tradeable_symbols)
        else:
            logger.warning(f"Strategy mode '{self.strategy_mode}' is not implemented.")
            return None

    def _generate_smart_rebalance_signals(self, tradeable_symbols: list[str]) -> Optional[List[Dict]]:
        """
        [✨ 수정] 지능형 손절 로직을 포함하여 거래 신호 목록을 생성합니다.
        """
        self.cycle_count += 1
        logger.info(f"===== Cycle #{self.cycle_count}: Starting Smart Rebalancing Logic =====")

        # 모든 거래 가능 종목에 대한 최신 예측을 한 번에 가져옴
        all_opportunities = self.opportunity_ranker.get_all_opportunities(tradeable_symbols)
        if not all_opportunities:
            logger.warning("No new opportunities found. Checking for forced exits only.")
            return self._check_and_generate_forced_exits()

        # 점수 기준으로 상위 N개 필터링
        top_opportunities = sorted(all_opportunities, key=lambda x: abs(x['final_composite_score']), reverse=True)[:settings.TOP_N_SYMBOLS]
        
        target_symbols_set = {opp['symbol'] for opp in top_opportunities}
        logger.info(f"Top {len(top_opportunities)} target symbols for this cycle: {list(target_symbols_set)}")

        target_positions = self.risk_manager.calculate_dynamic_position_sizes(top_opportunities)
        if not target_positions:
            logger.error("Failed to calculate target position sizes. Aborting rebalance.")
            return self._check_and_generate_forced_exits()

        is_forced_rebalance_cycle = (self.cycle_count % self.force_rebalance_cycles == 0)
        if is_forced_rebalance_cycle:
            logger.warning(f"FORCE REBALANCE cycle #{self.cycle_count}. Rebalancing all targets.")

        signals = []
        current_positions = self.portfolio_manager.get_all_positions()
        processed_for_close = set()

        # ✨ [핵심 수정] 1. 지능형 손절 로직 우선 실행
        if settings.USE_INTELLIGENT_LOSS_CUT:
            all_opportunities_map = {opp['symbol']: opp for opp in all_opportunities}
            for symbol, pos_details in current_positions.items():
                if pos_details.get('unrealisedPnl', 0) < 0:
                    latest_opp = all_opportunities_map.get(symbol)
                    if latest_opp:
                        current_side_is_buy = pos_details['side'].upper() == 'BUY'
                        new_signal_is_sell = latest_opp['side'].upper() == 'SELL'
                        
                        if (current_side_is_buy and new_signal_is_sell) or \
                           (not current_side_is_buy and not new_signal_is_sell):
                            
                            reason = f"Intelligent Loss Cut (AI signal reversed on losing position)"
                            signals.append(self._create_close_signal(symbol, pos_details, reason))
                            processed_for_close.add(symbol)

        # 2. 일반 리밸런싱 로직 실행
        for symbol, pos_details in current_positions.items():
            if symbol in processed_for_close: continue
            if symbol not in target_symbols_set:
                signals.append(self._create_close_signal(symbol, pos_details, "Not a top opportunity anymore"))

        for symbol, target_details in target_positions.items():
            if symbol in processed_for_close: continue
            
            current_pos = current_positions.get(symbol)
            
            if current_pos is None:
                signals.append(self._create_open_signal(symbol, target_details, "New entry"))
            else:
                needs_rebalance = is_forced_rebalance_cycle or \
                                  self.portfolio_manager.should_rebalance_position(
                                      symbol, target_details['side'], target_details['quantity'], self.rebalance_threshold)
                
                if needs_rebalance:
                    reason = "Forced Rebalance" if is_forced_rebalance_cycle else "Rebalancing (Threshold Exceeded)"
                    signals.append(self._create_close_signal(symbol, current_pos, reason))
                    signals.append(self._create_open_signal(symbol, target_details, reason))
                else:
                    logger.debug(f"Keeping {symbol} position unchanged (change is within threshold).")

        # 3. 시간 초과 등 기타 강제 청산 신호 추가
        forced_exits = self._check_and_generate_forced_exits()
        if forced_exits:
            existing_close_symbols = {s['symbol'] for s in signals if s['action'] == 'CLOSE'}
            for exit_signal in forced_exits:
                if exit_signal['symbol'] not in existing_close_symbols:
                    signals.append(exit_signal)

        signals.sort(key=lambda x: 0 if x['action'] == 'CLOSE' else 1)
        
        close_count = len([s for s in signals if s['action'] == 'CLOSE'])
        open_count = len([s for s in signals if s['action'] == 'OPEN'])
        logger.info(f"Generated {len(signals)} signals ({close_count} CLOSE, {open_count} OPEN).")
        
        return signals if signals else None

    def _create_close_signal(self, symbol: str, pos_details: Dict, reason: str) -> Dict:
        close_side = 'Buy' if pos_details['side'].upper() == 'SELL' else 'Sell'
        logger.info(f"Signal: CLOSE {pos_details['side']} on {symbol} by placing a {close_side} order | Size: {pos_details['size']} | Reason: {reason}")
        return {'action': 'CLOSE', 'symbol': symbol, 'side': close_side, 'size': pos_details['size'], 'reason': reason}

    def _create_open_signal(self, symbol: str, target_details: Dict, reason: str) -> Dict:
        """ ✨ TP/SL 가격을 신호에 포함하여 반환 """
        logger.info(f"Signal: OPEN {target_details['side']} on {symbol} | Value: ${target_details['size_in_usd']:,.2f} | Reason: {reason}")
        return {
            'action': 'OPEN',
            'symbol': symbol,
            'side': target_details['side'],
            'size_in_usd': target_details['size_in_usd'],
            'quantity': target_details['quantity'],
            'stop_loss': target_details['stop_loss_price'],
            'take_profit': target_details['take_profit_price'], # ✨ TP 가격 추가
            'reason': reason
        }

    def _check_and_generate_forced_exits(self) -> List[Dict]:
        exit_signals = []
        positions_to_exit = self.portfolio_manager.get_positions_to_liquidate_by_time()
        for pos_info in positions_to_exit:
            current_pos = self.portfolio_manager.get_all_positions().get(pos_info['symbol'])
            if current_pos:
                exit_signals.append(self._create_close_signal(pos_info['symbol'], current_pos, pos_info['reason']))
        return exit_signals

    def should_pause_trading(self) -> bool:
        if self.portfolio_manager.has_breached_daily_loss_limit():
            logger.critical("TRADING PAUSED: Daily loss limit has been breached.")
            return True
        total_capital = self.portfolio_manager.get_total_capital()
        if total_capital > 0:
            available_capital = self.portfolio_manager.get_available_capital()
            if (available_capital / total_capital) < 0.05:
                logger.warning("Trading paused: Low available capital (< 5%). Waiting for positions to close.")
                return True
        return False
