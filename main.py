# main.py

import logging
import time
import sys
import os
from datetime import datetime

# 시스템 경로 설정 및 로깅 초기화
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from configs.logging_config import setup_logging
setup_logging()

# 시스템 구성 요소 임포트
from configs import settings
from core.bybit_client import BybitClient
from core.market_scanner import MarketScanner
from core.portfolio_manager import PortfolioManager
from core.data_handler import DataHandler
from core.order_manager import OrderManager
from core.risk_manager import RiskManager
from ml.predictor import DeepLearningPredictor
from ml.enhanced_feature_engineer import EnhancedFeatureEngineer
from strategy.opportunity_ranker import OpportunityRanker
from strategy.main_strategy import MainStrategy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class TradingSystem:
    """
    [ v5.2 - 동적 주문 전략 적용 ]
    AI 자동매매 시스템의 모든 구성 요소를 초기화, 조립하고,
    메인 실행 루프를 명확하고 안정적으로 관장하는 컨트롤 타워 클래스.
    """
    def __init__(self):
        logger.info(">>> Initializing Trading System Components (v5.2)...")
        
        self.client = BybitClient(api_key=settings.API_KEY, api_secret=settings.API_SECRET, testnet=settings.TESTNET)
        self.order_manager = OrderManager(self.client)
        self.portfolio_manager = PortfolioManager(self.client)
        self.market_scanner = MarketScanner(self.client)
        self.feature_engineer = EnhancedFeatureEngineer()
        self.risk_manager = RiskManager(self.client)
        
        self.predictor = DeepLearningPredictor(model_dir=settings.MODEL_DIR)
        try:
            self.predictor.load_model_and_artifacts()
        except Exception as e:
            raise SystemExit("CRITICAL: AI model artifacts are missing or corrupted. Cannot start.")

        self.data_handler = None
        self.opportunity_ranker = OpportunityRanker(self.predictor, self.data_handler, self.feature_engineer)
        self.risk_manager.set_portfolio_manager(self.portfolio_manager)
        self.strategy = MainStrategy(self.opportunity_ranker, self.portfolio_manager, self.risk_manager)
        
        self.is_running = True
        logger.info("✅ Trading System initialized successfully.")

    def _recovery_check(self):
        """시스템 시작 시, 이전 실행에서 남은 미체결 주문을 정리합니다."""
        logger.info("Performing recovery check: Clearing any dangling open orders...")
        try:
            all_symbols_info = self.client.get_instruments_info(category="linear")
            if not all_symbols_info or all_symbols_info.get('retCode') != 0:
                logger.error("Could not fetch symbols for recovery check.")
                return
            
            symbols_to_check = [item['symbol'] for item in all_symbols_info['result']['list'] if item['symbol'].endswith('USDT')]

            for symbol in symbols_to_check:
                open_orders = self.client.get_open_orders(category="linear", symbol=symbol)
                if open_orders and open_orders.get('retCode') == 0 and open_orders['result']['list']:
                    logger.warning(f"Found dangling open orders for {symbol}. Cancelling all...")
                    self.client.cancel_all_orders(category="linear", symbol=symbol)
                    time.sleep(0.2)
            logger.info("Recovery check complete.")
        except Exception as e:
            logger.error(f"Error during recovery check: {e}")

    def _setup_initial_state(self) -> bool:
        """거래 시작 전, 사용자 자본금 설정, 복구 확인, 초기 데이터 로딩을 수행합니다."""
        self._recovery_check()

        try:
            # --- 자본금 설정 ---
            wallet_info = self.portfolio_manager.get_wallet_balance()
            if not wallet_info:
                logger.error("Failed to fetch wallet balance. Cannot proceed.")
                return False
            
            actual_balance = float(wallet_info.get('totalAvailableBalance', 0.0))
            logger.info(f"✅ Actual available balance in your account: ${actual_balance:,.2f} USDT")

            capital_input_str = input(f"Enter the amount of capital (USDT) to be used (Press Enter to use max): $")
            
            if not capital_input_str.strip():
                capital_to_set = actual_balance
                logger.info(f"Using maximum available balance: ${capital_to_set:,.2f}")
            else:
                user_capital = float(capital_input_str.replace(",", "").replace("$", ""))
                capital_to_set = min(user_capital, actual_balance)
                if capital_to_set < user_capital:
                    logger.warning(f"Input capital exceeds available balance. Automatically adjusted to ${capital_to_set:,.2f}")

            if not self.portfolio_manager.set_user_capital(capital_to_set): return False

            # ✨ 시장가 강제 여부 질문 제거. 시스템이 자동으로 판단.

        except ValueError:
            logger.error("Invalid capital input. Please enter a number.")
            return False
        except Exception as e:
            logger.error(f"An error occurred during initial setup: {e}", exc_info=True)
            return False

        tradeable_symbols = self.market_scanner.get_tradeable_symbols()
        if not tradeable_symbols:
            logger.error("No tradeable symbols found. Exiting system.")
            return False

        self.data_handler = DataHandler(self.client, tradeable_symbols, settings.INTERVAL, settings.MIN_KLINE_DATA_SIZE)
        self.opportunity_ranker.data_handler = self.data_handler
        self.data_handler.fetch_initial_data_for_all()
        return True

    def run(self):
        """자동매매 시스템의 메인 루프를 실행합니다."""
        if not self._setup_initial_state():
            logger.error("System setup failed. Exiting.")
            return

        logger.info(">>> Starting Trading System Main Loop <<<")
        try:
            while self.is_running:
                cycle_start_time = time.time()
                logger.info(f"\n{'='*30} New Cycle Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*30}")
                
                self.portfolio_manager.update_positions()
                self.portfolio_manager.print_portfolio_status()
                
                if self.strategy.should_pause_trading():
                    logger.warning("Trading is paused based on risk conditions (e.g., daily loss limit).")
                    time.sleep(300)
                    continue

                self.data_handler.update_all_data()
                signals = self.strategy.generate_signals(self.data_handler.symbols)
                self.execute_trade_signals(signals)

                logger.info("--- Cycle End Portfolio Sync ---")
                time.sleep(5) 
                self.portfolio_manager.update_positions()
                self.portfolio_manager.print_portfolio_status()

                work_duration = time.time() - cycle_start_time
                sleep_time = max(0, int(settings.INTERVAL) * 60 - work_duration)
                logger.info(f"Cycle finished in {work_duration:.2f}s. Sleeping for {sleep_time:.2f}s.")
                logger.info("="*88)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.warning("Shutdown initiated by user (Ctrl+C).")
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred in the main loop: {e}", exc_info=True)
        finally:
            self.shutdown()

    def execute_trade_signals(self, signals: list):
        """OPEN 신호에 포함된 TP/SL 가격을 OrderManager에 전달합니다."""
        if not signals:
            logger.info("No actionable signals generated in this cycle.")
            return

        close_signals = [s for s in signals if s['action'] == 'CLOSE']
        if close_signals:
            logger.info(f"--- Executing {len(close_signals)} CLOSE signals first ---")
            for signal in close_signals:
                self.order_manager.create_order(
                    symbol=signal['symbol'], side=signal['side'],
                    qty=signal['size'], order_type="Market", reduce_only=True
                )
            time.sleep(3)

        open_signals = [s for s in signals if s['action'] == 'OPEN']
        if open_signals:
            self.portfolio_manager.update_positions() 
            logger.info(f"--- Executing {len(open_signals)} OPEN signals ---")
            for signal in open_signals:
                if self.risk_manager.validate_position(signal['symbol'], signal['quantity'], signal['size_in_usd']):
                    self.order_manager.create_scalping_order(
                        symbol=signal['symbol'], 
                        side=signal['side'], 
                        capital_allocation=signal['size_in_usd'],
                        take_profit=signal.get('take_profit'),
                        stop_loss=signal.get('stop_loss')
                    )
                else:
                    logger.warning(f"[{signal['symbol']}] Position failed final validation. Skipping trade.")

    def shutdown(self):
        """[우아한 종료] 시스템을 안전하게 종료합니다."""
        logger.warning("<<<<< Initiating System Shutdown... >>>>>")
        self.is_running = False
        
        self.portfolio_manager.update_positions()
        open_positions = self.portfolio_manager.get_all_positions()
        
        if not open_positions:
            logger.info("No open positions. System shut down cleanly.")
            return
            
        logger.warning(f"There are {len(open_positions)} open positions:")
        for symbol, details in open_positions.items():
            pnl = details.get('unrealisedPnl', 0)
            color = "\033[92m" if pnl >= 0 else "\033[91m"
            logger.warning(f"  - {symbol} ({details['side']}): PnL {color}${pnl:,.4f}\033[0m")

        try:
            close_all = input("Do you want to MARKET CLOSE all open positions? (y/n): ").lower()
            if close_all == 'y':
                logger.info("Closing all open positions...")
                for symbol, details in open_positions.items():
                    close_side = 'Buy' if details['side'].upper() == 'SELL' else 'Sell'
                    self.order_manager.create_order(
                        symbol=symbol, side=close_side, qty=details['size'],
                        order_type="Market", reduce_only=True
                    )
                logger.info("All positions have been instructed to close.")
            else:
                logger.info("Leaving positions open. Please manage them manually.")
        except Exception as e:
            logger.error(f"Error during shutdown interaction: {e}. Please check positions manually.")
            
        logger.info("<<<<< System Shutdown Complete. >>>>>")

def main():
    """메인 실행 함수"""
    try:
        if "YOUR" in settings.API_KEY or not settings.API_KEY:
            print("CRITICAL: API keys are not configured correctly in .env or configs/settings.py.")
            return

        print("\n" + "="*60)
        print("          AI-Powered Trading System v5.2 (Dynamic Order Strategy)")
        print("="*60 + "\n")
        
        trading_system = TradingSystem()
        trading_system.run()
        
    except Exception as e:
        logger.critical(f"Failed to initialize or run TradingSystem: {e}", exc_info=True)

if __name__ == '__main__':
    main()
