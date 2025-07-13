# core

# /core/bybit_client.py

import logging
import time
from typing import Optional, Dict, Any, Callable

# pybit 라이브러리를 직접 임포트합니다.
from pybit.unified_trading import HTTP

# 시스템의 중앙 설정 파일을 임포트합니다.
try:
    from configs import settings
except ImportError:
    print("[치명적 오류] bybit_client.py에서 설정 파일을 임포트하는데 실패했습니다.")
    raise SystemExit("모듈 로딩 실패.")


logger = logging.getLogger(__name__)

class BybitClient:
    """
    [ v5.0 서킷 브레이커 통합 완료 ]
    모든 Bybit API 통신을 위한 중앙 관문(Gateway) 클래스.
    내장된 서킷 브레이커 패턴을 통해 API 서버의 지속적인 오류로부터 시스템을 보호합니다.
    """
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.session: Optional[HTTP] = None
        
        # --- 서킷 브레이커(Circuit Breaker) 상태 변수 ---
        self.state = "CLOSED"  # 상태: CLOSED (정상), OPEN (차단), HALF_OPEN (복구 시도)
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.failure_threshold = settings.CB_FAILURE_THRESHOLD
        self.recovery_timeout = settings.CB_RECOVERY_TIMEOUT
        
        try:
            # 1. 원본 코드와 동일하게 HTTP 세션을 먼저 생성합니다.
            # settings.py에서 TRADING_MODE에 따라 testnet 변수가 올바르게 설정됩니다.
            self.session = HTTP(
                testnet=testnet, api_key=api_key, api_secret=api_secret, recv_window=20000
            )
            
            # 2. ✨ [핵심 수정] 생성된 세션의 endpoint 속성을 사용자의 설정("d", "t", "m")에 맞게 덮어씁니다.
            if settings.TRADING_MODE == "d":
                self.session.endpoint = "https://api-demo.bybit.com"
                logger.warning(f"바이비트 데모 트레이딩 서버에 접속합니다 ({self.session.endpoint})")
            elif settings.TRADING_MODE == "t":
                logger.warning(f"바이비트 테스트넷 서버에 접속합니다 ({self.session.endpoint})")
            else: # "m" 또는 기타
                 logger.info(f"바이비트 메인넷 서버에 접속합니다 ({self.session.endpoint})")

            # 3. 초기 연결 상태 검증 (원본과 동일)
            server_time_res = self.get_server_time()
            if server_time_res and server_time_res.get('retCode') == 0:
                 logger.info("BybitClient 초기화 및 연결 검증 완료.")
            else:
                 raise ConnectionError(f"바이비트 API에 연결할 수 없습니다. 응답: {server_time_res}")

        except Exception as e:
            logger.critical(f"Bybit HTTP 세션 초기화 실패: {e}", exc_info=True)
            raise

    # --- 서킷 브레이커 로직 ---

    def _is_circuit_open(self) -> bool:
        """서킷이 OPEN 상태인지, 그리고 복구 시간이 지났는지 확인합니다."""
        if self.state == "OPEN":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.warning("서킷 브레이커 상태: HALF_OPEN. 테스트 호출을 시도합니다.")
                return False
            else:
                return True
        return False

    def _handle_success(self):
        """요청 성공 시 서킷 브레이커 상태를 리셋합니다."""
        if self.state == "HALF_OPEN":
            logger.info("서킷 브레이커 상태: CLOSED. 시스템 통신이 정상화되었습니다.")
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None

    def _handle_failure(self):
        """요청 실패 시 실패 횟수를 기록하고, 필요 시 서킷을 OPEN합니다."""
        self.failure_count += 1
        logger.warning(f"API 실패 횟수 증가: {self.failure_count}")
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_failure_time = time.time()
            logger.critical(f"🚨 서킷 브레이커 상태: OPEN. {self.failure_count}회 연속 실패로 {self.recovery_timeout}초 동안 API 호출을 차단합니다.")

    # --- 핵심 요청 처리 메서드 ---

    def _request_with_retry(self, api_call: Callable, **kwargs) -> Optional[Dict[str, Any]]:
        """
        [핵심 안정성] 서킷 브레이커와 재시도 로직을 결합한 모든 API 요청의 관문.
        """
        if self._is_circuit_open():
            logger.error(f"API 호출 '{api_call.__name__}' 차단됨. 서킷 브레이커가 OPEN 상태입니다.")
            return None

        retries = settings.API_REQUEST_RETRY_COUNT
        delay = settings.API_REQUEST_RETRY_DELAY

        last_response = None
        for i in range(retries):
            try:
                response = api_call(**kwargs)
                last_response = response
                if response and response.get('retCode') == 0:
                    self._handle_success()
                    return response

                if response and response.get('retCode') in [110007, 10005, 10004, 10003]:
                    logger.error(f"재시도 불가능한 API 오류. 코드: {response.get('retCode')}, 메시지: {response.get('retMsg')}")
                    return response

                logger.warning(f"API 호출 '{api_call.__name__}' 실패 (시도 {i+1}/{retries}). 응답: {response}")

            except Exception as e:
                logger.error(f"API 호출 '{api_call.__name__}' 중 예외 발생 (시도 {i+1}/{retries}): {e}")
                last_response = None # 예외 발생 시 응답 없음

            if i < retries - 1:
                time.sleep(delay)
        
        self._handle_failure()
        return last_response

    # =================================================================
    # [ 공식 API 래퍼 메서드 ]
    # 시스템의 다른 모든 모듈은 아래의 메서드만을 통해 API와 통신해야 합니다.
    # =================================================================

    def get_server_time(self) -> Optional[Dict[str, Any]]:
        """서버 시간을 조회합니다."""
        try:
            return self.session.get_server_time()
        except Exception as e:
            logger.error(f"서버 시간 조회 실패: {e}")
            return None

    def get_instruments_info(self, **kwargs) -> Optional[Dict[str, Any]]:
        """거래 상품 정보를 조회합니다."""
        return self._request_with_retry(self.session.get_instruments_info, **kwargs)

    def get_kline(self, **kwargs) -> Optional[Dict[str, Any]]:
        """K-line(캔들) 데이터를 조회합니다."""
        return self._request_with_retry(self.session.get_kline, **kwargs)

    def get_tickers(self, **kwargs) -> Optional[Dict[str, Any]]:
        """티커 정보를 조회합니다."""
        return self._request_with_retry(self.session.get_tickers, **kwargs)

    def get_positions(self, **kwargs) -> Optional[Dict[str, Any]]:
        """현재 보유 포지션 정보를 조회합니다."""
        return self._request_with_retry(self.session.get_positions, **kwargs)

    def get_wallet_balance(self, **kwargs) -> Optional[Dict[str, Any]]:
        """지갑 잔고를 조회합니다."""
        return self._request_with_retry(self.session.get_wallet_balance, **kwargs)

    def place_order(self, **kwargs) -> Optional[Dict[str, Any]]:
        """주문을 실행합니다."""
        return self._request_with_retry(self.session.place_order, **kwargs)

    def get_open_orders(self, **kwargs) -> Optional[Dict[str, Any]]:
        """미체결 주문 목록을 조회합니다."""
        return self._request_with_retry(self.session.get_open_orders, **kwargs)

    def cancel_order(self, **kwargs) -> Optional[Dict[str, Any]]:
        """단일 주문을 취소합니다."""
        # 주문 취소는 Race Condition 가능성 때문에 재시도 로직을 통과하되,
        # 실패 응답(예: 주문 없음)을 상위 로직에서 잘 처리하는 것이 중요합니다.
        return self._request_with_retry(self.session.cancel_order, **kwargs)

    def cancel_all_orders(self, **kwargs) -> Optional[Dict[str, Any]]:
        """특정 종목의 모든 미체결 주문을 취소합니다."""
        return self._request_with_retry(self.session.cancel_all_orders, **kwargs)

    # =================================================================
    # [ 중앙화된 유틸리티 메서드 ]
    # =================================================================

    def get_current_price(self, symbol: str) -> Optional[float]:
        """단일 종목의 현재가를 조회합니다."""
        try:
            ticker_info = self.get_tickers(category="linear", symbol=symbol)
            if ticker_info and ticker_info.get('retCode') == 0 and ticker_info.get('result', {}).get('list'):
                return float(ticker_info['result']['list'][0]['lastPrice'])
            else:
                 logger.warning(f"[{symbol}] 현재가 조회 실패. API 응답: {ticker_info}")
        except (IndexError, KeyError, ValueError) as e:
            logger.warning(f"[{symbol}] 티커 정보에서 현재가를 파싱할 수 없습니다: {e}")
        except Exception as e:
            logger.error(f"[{symbol}] 현재가 조회 중 예상치 못한 오류 발생: {e}", exc_info=False)
        return None



# /core/data_handler.py

import logging
import pandas as pd
from collections import deque
import time
import math
from typing import Deque, Dict, List, Optional

# 시스템의 다른 핵심 모듈을 임포트합니다.
try:
    from core.bybit_client import BybitClient
    from configs import settings
except ImportError as e:
    print(f"[CRITICAL ERROR] Failed to import modules in data_handler.py: {e}")
    print("Please ensure you are running the script from the project's root directory.")
    raise SystemExit("Core module loading failed.")

logger = logging.getLogger(__name__)

class DataHandler:
    """
    [ v5.0 인덱스 조작 기능 제거됨 ]
    Bybit API와의 통신을 통해 시계열 데이터를 관리하는 핵심 기능에 집중합니다.
    - timestamp 컬럼을 인덱스로 변경하는 월권 행위를 제거하여 다른 모듈과의 호환성을 보장합니다.
    """

    def __init__(self,
                 bybit_client: BybitClient,
                 symbols: List[str],
                 interval: str = settings.INTERVAL,
                 initial_klines: int = settings.MIN_KLINE_DATA_SIZE):
        self.client = bybit_client
        self.symbols = symbols
        self.interval = interval
        self.initial_klines = initial_klines
        self.api_kline_limit = settings.API_KLINE_LIMIT
        self.kline_data: Dict[str, Deque[Dict]] = {
            symbol: deque(maxlen=self.initial_klines + 20) for symbol in self.symbols
        }
        logger.info(f"DataHandler initialized for {len(self.symbols)} symbols with interval '{self.interval}'.")

    def _fetch_data_for_symbol(self, symbol: str) -> bool:
        """단일 종목의 초기 데이터를 페이지네이션을 통해 수집합니다."""
        logger.debug(f"[{symbol}] Fetching initial {self.initial_klines} k-lines...")
        all_klines = []
        needed_requests = math.ceil(self.initial_klines / self.api_kline_limit)
        end_time = None

        for i in range(needed_requests):
            try:
                params = {'category': 'linear', 'symbol': symbol, 'interval': self.interval, 'limit': self.api_kline_limit}
                if end_time: params['end'] = end_time

                response = self.client.get_kline(**params)
                if response and response.get('retCode') == 0 and response['result']['list']:
                    klines = response['result']['list']
                    all_klines.extend(klines)
                    end_time = int(klines[-1][0])
                    time.sleep(0.1)
                else:
                    logger.warning(f"[{symbol}] Could not fetch further kline data. Response: {response.get('retMsg', 'N/A')}")
                    break
            except Exception as e:
                logger.error(f"[{symbol}] Exception during paginated data fetch: {e}")
                return False

        if not all_klines:
            logger.error(f"[{symbol}] Failed to fetch any initial kline data.")
            return False

        # --- 데이터 정제 파이프라인 ---
        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        cols_to_numeric = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)

        if df.empty:
            logger.error(f"[{symbol}] All data was removed after type conversion. Please check API response.")
            return False

        # [수정] timestamp를 datetime 객체로 변환하되, 인덱스로 설정하지 않음
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 중복 제거 및 시간순 정렬 (timestamp 컬럼 기준)
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.sort_values(by='timestamp', inplace=True)
        
        final_klines_df = df.tail(self.initial_klines)

        self.kline_data[symbol].clear()
        
        # DataFrame을 다시 dictionary 리스트로 변환하여 저장
        for _, row in final_klines_df.iterrows():
            self.kline_data[symbol].append({
                # FeatureEngineer가 ms 단위 정수를 기대하므로 다시 변환
                'timestamp': int(row['timestamp'].timestamp() * 1000), 
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': float(row['volume'])
            })

        logger.debug(f"[{symbol}] Successfully fetched and stored {len(self.kline_data[symbol])} initial k-lines.")
        return True

    def fetch_initial_data_for_all(self):
        """모든 관리 대상 종목의 초기 데이터를 로드합니다."""
        logger.info(f"Fetching initial data for all {len(self.symbols)} symbols...")
        for symbol in self.symbols:
            if not self._fetch_data_for_symbol(symbol):
                logger.error(f"Failed to fetch initial data for {symbol}. It will be excluded from this run.")
            time.sleep(0.2)
        logger.info("Initial data fetch process completed for all symbols.")

    def _update_data_for_symbol(self, symbol: str):
        """단일 종목의 데이터를 최신 캔들 1~2개로 업데이트합니다."""
        if not self.kline_data[symbol]:
            self._fetch_data_for_symbol(symbol)
            return

        try:
            response = self.client.get_kline(category='linear', symbol=symbol, interval=self.interval, limit=2)
            if response and response.get('retCode') == 0 and len(response['result']['list']) >= 1:
                latest_kline_data = response['result']['list'][0]
                new_kline = {
                    'timestamp': int(latest_kline_data[0]),
                    'open': float(latest_kline_data[1]), 'high': float(latest_kline_data[2]),
                    'low': float(latest_kline_data[3]), 'close': float(latest_kline_data[4]),
                    'volume': float(latest_kline_data[5])
                }
                last_local_ts = self.kline_data[symbol][-1]['timestamp']

                if new_kline['timestamp'] > last_local_ts:
                    self.kline_data[symbol].append(new_kline)
                elif new_kline['timestamp'] == last_local_ts:
                    self.kline_data[symbol][-1] = new_kline
            else:
                logger.warning(f"[{symbol}] Could not fetch latest kline for update. Response: {response.get('retMsg', 'N/A')}")
        except Exception as e:
            logger.error(f"[{symbol}] Exception during data update: {e}")

    def update_all_data(self):
        """모든 관리 대상 종목의 데이터를 최신으로 업데이트합니다."""
        logger.info("Updating data for all symbols...")
        for symbol in self.symbols:
            self._update_data_for_symbol(symbol)
            time.sleep(0.1)
        logger.info("Data update process for all symbols completed.")

    def get_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        특정 종목의 현재 데이터를 pandas DataFrame으로 변환하여 반환합니다.
        [수정] timestamp를 일반 컬럼으로 유지한 채로 DataFrame을 반환합니다.
        """
        if symbol not in self.kline_data or not self.kline_data[symbol]:
            logger.warning(f"No kline data available for symbol: {symbol}")
            return None
            
        df = pd.DataFrame(list(self.kline_data[symbol]))
        # FeatureEngineer가 ms 단위 정수 timestamp를 기대하므로, 그대로 반환합니다.
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # 이 변환은 FeatureEngineer에서 수행
        return df



# /core/market_scanner.py

import logging
import pandas as pd
from typing import List
import time

from core.bybit_client import BybitClient
from configs import settings

logger = logging.getLogger(__name__)

class MarketScanner:
    """
    [v5.1 - 최근 활동성 필터 추가]
    거래 가능한 암호화폐 종목을 스캔하고, 유동성 필터(24시간 기준)와
    최근 활동성 필터(1시간 기준)를 순차적으로 적용하여 최종 종목을 선정합니다.
    """
    def __init__(self, bybit_client: BybitClient):
        self.client = bybit_client
        logger.info("MarketScanner initialized.")

    def _get_trimmed_mean(self, data: pd.Series, trim_count: int) -> float:
        """상위/하위 극단값을 제외한 평균(절삭 평균)을 계산합니다."""
        if len(data) <= trim_count * 2:
            return data.mean() if not data.empty else 0
        
        sorted_data = sorted(data)
        trimmed_data = sorted_data[trim_count:-trim_count]
        
        return sum(trimmed_data) / len(trimmed_data) if trimmed_data else 0

    def get_tradeable_symbols(self) -> List[str]:
        """
        [✨ 수정] 2단계 필터링 로직을 적용하여 최종 거래 가능 종목 목록을 반환합니다.
        1. 24시간 거래대금/거래량 기준 1차 필터링
        2. 최근 1시간 거래대금 기준 2차 필터링
        """
        logger.info("Starting to scan for tradeable symbols with 2-step filtering...")
        
        # 1. 24시간 기준 1차 필터링 (기존 로직)
        instr_res = self.client.get_instruments_info(category="linear")
        tickers_res = self.client.get_tickers(category="linear")

        if not (instr_res and tickers_res and instr_res.get('retCode') == 0 and tickers_res.get('retCode') == 0):
            logger.error("Failed to get initial data from Bybit (instruments or tickers).")
            return []

        df_instr = pd.DataFrame(instr_res['result']['list'])
        df_tickers = pd.DataFrame(tickers_res['result']['list'])
        
        if df_instr.empty or df_tickers.empty:
            logger.error("Instrument or Ticker data from API is empty.")
            return []

        df = pd.merge(df_instr, df_tickers, on='symbol')

        df_filtered = df[
            (df['status'] == 'Trading') &
            (df['symbol'].str.endswith('USDT')) &
            (~df['symbol'].isin(settings.SYMBOL_BLACKLIST))
        ].copy()

        numeric_cols = ['turnover24h', 'lastPrice']
        for col in numeric_cols:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        df_filtered.dropna(subset=numeric_cols, inplace=True)
        
        initial_count = len(df_filtered)
        df_filtered = df_filtered[df_filtered['turnover24h'] > 0].copy()
        if initial_count > len(df_filtered):
            logger.info(f"Removed {initial_count - len(df_filtered)} symbols with zero turnover.")
        
        if df_filtered.empty:
            logger.warning("No symbols with active trading volume found.")
            return []

        df_filtered['volume24h_in_coins'] = df_filtered['turnover24h'] / df_filtered['lastPrice']

        trim_count = settings.FILTER_TRIM_COUNT
        avg_turnover = self._get_trimmed_mean(df_filtered['turnover24h'], trim_count)
        avg_volume_in_coins = self._get_trimmed_mean(df_filtered['volume24h_in_coins'], trim_count)

        first_pass_df = df_filtered[
            (df_filtered['turnover24h'] >= avg_turnover)]# &
          #  (df_filtered['volume24h_in_coins'] >= avg_volume_in_coins)
        #]
        
        first_pass_symbols = first_pass_df['symbol'].tolist()
        logger.info(f"1st Pass Filter (24h liquidity): {len(first_pass_symbols)} symbols selected.")

        # 2. ✨ 최근 활동성 기준 2차 필터링 (신규 로직)
        if not settings.USE_RECENT_ACTIVITY_FILTER:
            logger.info("Recent activity filter is disabled. Skipping 2nd pass.")
            return first_pass_df.sort_values('turnover24h', ascending=False)['symbol'].tolist()

        logger.info(f"Starting 2nd Pass Filter (Recent Activity, last {settings.RECENT_ACTIVITY_TIMEFRAME_MINUTES} mins)...")
        final_tradeable_symbols = []
        
        # 5분봉 기준, 1시간 = 12개 캔들. 여유있게 15개 요청
        kline_count = int(settings.RECENT_ACTIVITY_TIMEFRAME_MINUTES / int(settings.INTERVAL)) + 3

        for symbol in first_pass_symbols:
            try:
                kline_res = self.client.get_kline(
                    category="linear", symbol=symbol, interval=settings.INTERVAL, limit=kline_count
                )
                if kline_res and kline_res.get('retCode') == 0 and kline_res['result']['list']:
                    klines = kline_res['result']['list']
                    # API 응답은 ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'] 순서
                    recent_turnover = sum(float(k[6]) for k in klines if k[6])
                    
                    if recent_turnover >= settings.RECENT_ACTIVITY_MIN_TURNOVER_USD:
                        final_tradeable_symbols.append(symbol)
                    else:
                        logger.debug(f"[{symbol}] Filtered out. Recent turnover ${recent_turnover:,.0f} < "
                                     f"min required ${settings.RECENT_ACTIVITY_MIN_TURNOVER_USD:,.0f}")
                else:
                    logger.warning(f"[{symbol}] Could not fetch k-lines for activity check.")
                
                time.sleep(0.1) # API 요청 간격
            except Exception as e:
                logger.error(f"[{symbol}] Error during recent activity check: {e}")

        # 유동성이 가장 높은 순서로 정렬
        final_df = first_pass_df[first_pass_df['symbol'].isin(final_tradeable_symbols)]
        final_sorted_symbols = final_df.sort_values('turnover24h', ascending=False)['symbol'].tolist()

        logger.info(
            f"Final tradeable symbols after 2nd Pass Filter: {len(final_sorted_symbols)}"
        )
        logger.debug(f"Final tradeable symbol list: {final_sorted_symbols[:10]}...")
        
        return final_sorted_symbols



# /core/order_manager.py

import logging
import uuid
import math
from time import sleep
from typing import Optional, Dict, Any, Tuple, Union

# 시스템의 다른 핵심 모듈을 임포트합니다.
try:
    from core.bybit_client import BybitClient
    from configs import settings
except ImportError:
    print("[CRITICAL ERROR] Failed to import modules in order_manager.py.")
    raise SystemExit("Module loading failed.")


logger = logging.getLogger(__name__)

class OrderManager:
    """
    [ v5.6 - 동적 주문 전략 적용 ]
    주문 시점의 호가 스프레드를 분석하여 지정가(Maker)와 시장가(Taker) 중
    가장 비용 효율적인 주문 방식을 자동으로 선택합니다.
    """

    def __init__(self, bybit_client: BybitClient):
        self.client = bybit_client
        self.instrument_info_cache: Dict[str, Dict] = {}
        logger.info(f"OrderManager initialized with Dynamic Order Strategy: {settings.DYNAMIC_ORDER_STRATEGY}")

    # ... (기존 _get_instrument_info, _adjust_qty_to_step 함수는 변경 없음)
    def _get_instrument_info(self, symbol: str) -> Optional[Dict]:
        if symbol in self.instrument_info_cache: return self.instrument_info_cache[symbol]
        try:
            response = self.client.get_instruments_info(category="linear", symbol=symbol)
            if response and response.get('retCode') == 0 and response['result']['list']:
                info = response['result']['list'][0]
                self.instrument_info_cache[symbol] = info
                return info
        except Exception as e: logger.error(f"[{symbol}] Exception fetching instrument info: {e}")
        return None

    def _adjust_qty_to_step(self, symbol: str, quantity: float) -> Optional[str]:
        info = self._get_instrument_info(symbol)
        if not info: return None
        lot_size_filter = info.get('lotSizeFilter', {})
        qty_step_str = lot_size_filter.get('qtyStep')
        if not qty_step_str: return str(quantity)
        try:
            qty_step = float(qty_step_str)
            if qty_step <= 0: return str(quantity)
            adjusted_qty = math.floor(quantity / qty_step) * qty_step
            decimal_places = len(qty_step_str.split('.')[1]) if '.' in qty_step_str else 0
            return f"{adjusted_qty:.{decimal_places}f}"
        except (ValueError, TypeError) as e:
            logger.error(f"[{symbol}] Error adjusting quantity '{quantity}': {e}")
            return None
            
    def create_order(self, symbol: str, side: str, qty: Union[float, str], order_type: str, price: Optional[float] = None, reduce_only: bool = False, **kwargs) -> Optional[Dict]:
        """
        [✨ 핵심 수정] 모든 주문 요청을 받아, 동적 주문 전략에 따라 최적의 방식으로 실행합니다.
        """
        qty_str = str(qty)

        # 1. 동적 주문 전략 활성화 시
        if settings.DYNAMIC_ORDER_STRATEGY and order_type == "Market":
            ticker_info = self.client.get_tickers(category="linear", symbol=symbol)
            if not ticker_info or ticker_info.get('retCode') != 0 or not ticker_info['result']['list']:
                logger.error(f"[{symbol}] Could not get ticker for dynamic strategy. Falling back to Market order.")
                return self._execute_standard_order(symbol, side, "Market", qty_str, None, reduce_only, **kwargs)

            ticker_data = ticker_info['result']['list'][0]
            try:
                bid_price = float(ticker_data['bid1Price'])
                ask_price = float(ticker_data['ask1Price'])
                
                if bid_price <= 0 or ask_price <= 0: raise ValueError("Invalid bid/ask price")

                spread_pct = ((ask_price - bid_price) / bid_price) * 100
                logger.debug(f"[{symbol}] Spread: {spread_pct:.4f}%. Threshold: {settings.DYNAMIC_ORDER_SPREAD_THRESHOLD_PCT:.4f}%")

                # 2. 스프레드 기반 의사결정
                # 스프레드가 임계값보다 넓으면 -> 지정가 시도 (비용 절약)
                if spread_pct > settings.DYNAMIC_ORDER_SPREAD_THRESHOLD_PCT:
                    logger.info(f"[{symbol}] Wide spread detected. Attempting MAKER order to save cost.")
                    return self._try_maker_order(symbol, side, qty_str, reduce_only, ticker_data, **kwargs)
                # 스프레드가 좁으면 -> 시장가 즉시 실행 (기회비용 절약)
                else:
                    logger.info(f"[{symbol}] Tight spread detected. Executing TAKER order for speed.")
                    return self._execute_standard_order(symbol, side, "Market", qty_str, None, reduce_only, **kwargs)

            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"[{symbol}] Error processing ticker for dynamic strategy: {e}. Falling back to Market order.")
                return self._execute_standard_order(symbol, side, "Market", qty_str, None, reduce_only, **kwargs)

        # 3. 기존 지정가 우선 방식 (DYNAMIC_ORDER_STRATEGY = False일 때)
        elif settings.PREFER_MAKER_ORDERS and order_type == "Market":
            ticker_data = self.client.get_tickers(category="linear", symbol=symbol)['result']['list'][0]
            logger.info(f"[{symbol}] Using legacy maker-first strategy...")
            maker_result = self._try_maker_order(symbol, side, qty_str, reduce_only, ticker_data, **kwargs)
            if maker_result:
                return maker_result
            logger.info(f"[{symbol}] Maker order failed or timed out. Falling back to market taker order.")
        
        # 4. 모든 조건에 해당하지 않으면 표준 주문 실행
        return self._execute_standard_order(symbol, side, order_type, qty_str, price, reduce_only, **kwargs)

    def _try_maker_order(self, symbol: str, side: str, qty_str: str, reduce_only: bool, ticker_data: Dict, **kwargs) -> Optional[Dict]:
        """Best Bid/Ask를 사용하여 체결 확률이 높은 지정가 주문을 시도합니다."""
        try:
            if side.upper() == 'BUY':
                maker_price = float(ticker_data['bid1Price'])
            else: # SELL
                maker_price = float(ticker_data['ask1Price'])

            if maker_price <= 0:
                logger.error(f"[{symbol}] Invalid maker price ({maker_price}). Aborting maker order.")
                return None

            response = self._execute_standard_order(symbol, side, "Limit", qty_str, maker_price, reduce_only, timeInForce="PostOnly", **kwargs)
            
            if not response or response.get('retCode') != 0: return None
            order_id = response['result'].get('orderId')
            if not order_id: return None
            
            logger.info(f"[{symbol}] Maker order placed (ID: {order_id}) at price ${maker_price}. Waiting up to {settings.MAKER_ORDER_TIMEOUT}s for fill...")
            
            if self._wait_for_order_fill(symbol, order_id):
                logger.info(f"[{symbol}] Maker order (ID: {order_id}) confirmed as FILLED!")
                return response
            
            logger.info(f"[{symbol}] Maker order not filled. Attempting to cancel...")
            cancelled, err_code = self._cancel_order(symbol, order_id)
            
            if cancelled: return None
            if err_code == 110001:
                logger.warning(f"[{symbol}] Cancel failed (Code 110001). Assuming it was filled. SUCCESS.")
                return response
            
            logger.error(f"[{symbol}] CRITICAL: Failed to cancel untracked maker order (ID: {order_id}). Code: {err_code}. Aborting.")
            return None

        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"[{symbol}] Failed to parse ticker data for maker price: {e}")
            return None
        except Exception as e:
            logger.error(f"[{symbol}] Exception in _try_maker_order: {e}", exc_info=True)
            return None

    def _execute_standard_order(self, symbol: str, side: str, order_type: str, qty_str: str, price: Optional[float] = None, reduce_only: bool = False, **kwargs) -> Optional[Dict]:
        """API에 직접 주문 요청을 보내는 최종 실행 함수."""
        try:
            params = {"orderLinkId": f"auto_{symbol.lower()}_{uuid.uuid4().hex[:10]}"}
            if order_type.upper() == 'LIMIT' and price: params['price'] = str(price)
            if reduce_only: params['reduceOnly'] = True
            
            if settings.SET_TPSL_ON_ENTRY and not reduce_only:
                if kwargs.get('take_profit'): params['takeProfit'] = str(kwargs.get('take_profit'))
                if kwargs.get('stop_loss'): params['stopLoss'] = str(kwargs.get('stop_loss'))

            kwargs.pop('take_profit', None)
            kwargs.pop('stop_loss', None)
            params.update(kwargs)

            logger.info(f"==> EXECUTING ORDER: [{symbol}] {side} {qty_str} @ {order_type} | Params: {params}")
            
            return self.client.place_order(
                category="linear", symbol=symbol, side=side, orderType=order_type, qty=qty_str, **params
            )
        except Exception as e:
            logger.error(f"[{symbol}] Exception during standard order execution: {e}", exc_info=True)
            return None

    # ... (이하 _wait_for_order_fill, _get_order_status, _cancel_order, create_scalping_order 함수는 변경 없음)
    def create_scalping_order(self, symbol: str, side: str, capital_allocation: float, **kwargs) -> Optional[Dict]:
        instrument_info = self._get_instrument_info(symbol)
        if not instrument_info:
            logger.error(f"[{symbol}] Could not get instrument info. Skipping order.")
            return None
        lot_size_filter = instrument_info.get('lotSizeFilter', {})
        min_order_amt_str = lot_size_filter.get('minOrderAmt')
        if min_order_amt_str:
            try:
                min_order_amt = float(min_order_amt_str)
                if capital_allocation < min_order_amt:
                    logger.warning(f"[{symbol}] Order REJECTED. Allocated capital ${capital_allocation:,.2f} is below min value of ${min_order_amt:,.2f} USDT.")
                    return None
            except ValueError:
                logger.error(f"[{symbol}] Could not parse minOrderAmt '{min_order_amt_str}'.")
        try:
            current_price = self.client.get_current_price(symbol)
            if current_price is None or current_price <= 0: return None
            raw_quantity = capital_allocation / current_price
            adjusted_qty_str = self._adjust_qty_to_step(symbol, raw_quantity)
            if adjusted_qty_str is None or float(adjusted_qty_str) <= 0:
                logger.error(f"[{symbol}] Adjusted quantity '{adjusted_qty_str}' is invalid for capital ${capital_allocation:,.2f}.")
                return None
            logger.info(f"[{symbol}] Scalping order quantity adjusted: Raw {raw_quantity:.8f} -> Final {adjusted_qty_str}")
            return self.create_order(symbol=symbol, side=side, qty=adjusted_qty_str, order_type="Market", **kwargs)
        except Exception as e:
            logger.error(f"[{symbol}] Error in create_scalping_order: {e}", exc_info=True)
            return None

    def _wait_for_order_fill(self, symbol: str, order_id: str) -> bool:
        for _ in range(settings.MAKER_ORDER_TIMEOUT):
            sleep(1)
            status = self._get_order_status(symbol, order_id)
            if status == "Filled": return True
            if status is None: return True
            if status in ["Cancelled", "Rejected", "Expired"]: return False
        return False

    def _get_order_status(self, symbol: str, order_id: str) -> Optional[str]:
        try:
            response = self.client.get_open_orders(category="linear", symbol=symbol, orderId=order_id)
            if response and response.get('retCode') == 0 and response['result']['list']:
                return response['result']['list'][0].get('orderStatus')
        except Exception: pass
        return None

    def _cancel_order(self, symbol: str, order_id: str) -> Tuple[bool, Optional[int]]:
        try:
            response = self.client.cancel_order(category="linear", symbol=symbol, orderId=order_id)
            if response and response.get('retCode') == 0:
                logger.info(f"[{symbol}] Order {order_id} cancelled successfully.")
                return True, 0
            err_code = response.get('retCode') if response else -1
            logger.warning(f"[{symbol}] Failed to cancel order {order_id}. Response: {response}")
            return False, err_code
        except Exception as e:
            logger.error(f"[{symbol}] Exception cancelling order {order_id}: {e}")
            return False, -1



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
