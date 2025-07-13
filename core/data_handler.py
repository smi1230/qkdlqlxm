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
