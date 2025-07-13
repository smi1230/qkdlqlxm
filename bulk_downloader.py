### /bulk_downloader.py

import os
import logging
import pandas as pd
import time
from datetime import datetime, timezone

# 프로젝트의 다른 모듈을 임포트합니다.
# 이 스크립트는 trading_system 폴더 외부에서 실행될 수도 있으므로,
# 경로 문제를 방지하기 위해 시스템 경로를 추가하는 로직이 포함될 수 있습니다.
# (예: import sys; sys.path.append('.'))
try:
    from core.bybit_client import BybitClient
    from configs.settings import (
        API_KEY, API_SECRET, TESTNET, SYMBOL_BLACKLIST,
        DATA_DIR, TRAINING_DATA_START_DATE, INTERVAL
    )
except ImportError as e:
    print(f"[ERROR] 모듈 임포트 실패: {e}")
    print("이 스크립트는 프로젝트 루트 디렉토리에서 실행되어야 합니다. (예: python bulk_downloader.py)")
    exit()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BulkDownloader:
    """
    모델 훈련을 위한 대량의 과거 K-line 데이터를 Bybit에서 다운로드하는 클래스.
    API의 1000개 제한을 극복하기 위한 페이지네이션 로직을 포함합니다.
    """

    def __init__(self, bybit_client: BybitClient, data_dir: str, start_date_str: str, interval: str):
        """
        BulkDownloader를 초기화합니다.

        Args:
            bybit_client (BybitClient): 통신을 위한 BybitClient 인스턴스.
            data_dir (str): 다운로드한 데이터를 저장할 디렉토리 경로.
            start_date_str (str): 데이터 수집 시작일 ("YYYY-MM-DD" 형식).
            interval (str): 다운로드할 캔들 데이터의 시간 주기.
        """
        self.client = bybit_client
        self.data_dir = data_dir
        self.interval = interval
        # 시작 날짜 문자열을 UTC 타임스탬프(밀리초)로 변환합니다.
        self.start_timestamp_ms = int(datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        # 데이터 저장 디렉토리가 없으면 생성합니다.
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"BulkDownloader initialized. Data will be saved to '{self.data_dir}'.")

    def _fetch_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        """
        단일 종목에 대해 지정된 시작일부터 현재까지의 모든 과거 데이터를 다운로드합니다.

        Args:
            symbol (str): 데이터를 다운로드할 종목.

        Returns:
            pd.DataFrame: 해당 종목의 전체 기간 데이터가 포함된 데이터프레임.
        """
        all_data = []
        limit = 1000  # Bybit API의 최대 요청 한도
        end_timestamp_ms = None  # 가장 최근 데이터부터 시작하기 위해 None으로 설정

        logger.info(f"[{symbol}] Starting download from {datetime.fromtimestamp(self.start_timestamp_ms / 1000)}...")

        while True:
            try:
                response = self.client.get_kline(
                    symbol=symbol,
                    interval=self.interval,
                    limit=limit,
                    end=end_timestamp_ms
                )

                if response and response.get('retCode') == 0:
                    klines = response['result']['list']
                    if not klines:
                        logger.info(f"[{symbol}] No more data returned from API. Fetching complete.")
                        break
                    
                    all_data.extend(klines)
                    oldest_ts_in_batch = int(klines[-1][0])

                    # 다음 요청을 위해 end_timestamp를 업데이트합니다.
                    end_timestamp_ms = oldest_ts_in_batch

                    logger.debug(f"[{symbol}] Fetched {len(klines)} k-lines, ending at {datetime.fromtimestamp(oldest_ts_in_batch / 1000)}")

                    if oldest_ts_in_batch < self.start_timestamp_ms:
                        logger.info(f"[{symbol}] Reached the target start date.")
                        break
                    
                    time.sleep(0.2)  # API 속도 제한을 준수하기 위해 잠시 대기
                else:
                    logger.error(f"[{symbol}] Failed to fetch data: {response.get('retMsg', 'Unknown error')}")
                    return pd.DataFrame() # 실패 시 빈 데이터프레임 반환

            except Exception as e:
                logger.error(f"[{symbol}] An exception occurred during data fetch: {e}")
                time.sleep(5) # 예외 발생 시 잠시 후 재시도
                continue
        
        if not all_data:
            return pd.DataFrame()

        # 데이터프레임 생성 및 전처리
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df = df.sort_values('timestamp', ascending=True).drop_duplicates(subset='timestamp')
        df = df[df['timestamp'] >= self.start_timestamp_ms]

        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.set_index('timestamp', inplace=True)
        return df

    def download_all_symbols(self, symbols_to_download: list):
        """
        지정된 모든 종목의 데이터를 다운로드하여 CSV 파일로 저장합니다.

        Args:
            symbols_to_download (list): 다운로드할 종목 문자열의 리스트.
        """
        total_symbols = len(symbols_to_download)
        logger.info(f"Starting bulk download for {total_symbols} symbols.")
        
        for i, symbol in enumerate(symbols_to_download):
            logger.info(f"--- Processing {i + 1}/{total_symbols}: {symbol} ---")
            
            file_path = os.path.join(self.data_dir, f"{symbol}_{self.interval}.csv")
            if os.path.exists(file_path):
                logger.info(f"[{symbol}] Data file already exists. Skipping download.")
                continue

            df = self._fetch_data_for_symbol(symbol)
            if not df.empty:
                df.to_csv(file_path)
                logger.info(f"[{symbol}] Successfully downloaded {len(df)} records and saved to {file_path}")
            else:
                logger.warning(f"[{symbol}] No data was downloaded. File not created.")
        
        logger.info("--- Bulk download process finished. ---")


if __name__ == '__main__':
    """
    이 스크립트를 직접 실행하면 모델 훈련에 필요한 전체 과거 데이터를 다운로드합니다.
    """
    logger.info("=============================================")
    logger.info("===    BYBIT BULK DATA DOWNLOADER    ===")
    logger.info("=============================================")

    # 1. Bybit 클라이언트 초기화
    client = BybitClient(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET)
    
    # 2. 거래 가능한 USDT 무기한 계약 종목 목록 가져오기
    logger.info("Fetching list of all available USDT perpetual symbols...")
    instruments_info = client.get_instruments_info(category="linear")
    
    if not (instruments_info and instruments_info.get('retCode') == 0):
        logger.error("Could not fetch instrument info. Please check API keys and network.")
        exit()
        
    all_symbols = [
        item['symbol'] for item in instruments_info['result']['list']
        if 'USDT' in item['symbol'] and item.get('status') == 'Trading'
    ]
    
    # 3. 블랙리스트에 포함된 종목 제외
    symbols_to_process = [s for s in all_symbols if s not in SYMBOL_BLACKLIST]
    
    logger.info(f"Found {len(all_symbols)} total symbols. After applying blacklist, {len(symbols_to_process)} symbols will be processed.")

    # 4. 다운로더 인스턴스 생성 및 실행
    downloader = BulkDownloader(
        bybit_client=client,
        data_dir=DATA_DIR,
        start_date_str=TRAINING_DATA_START_DATE,
        interval=INTERVAL
    )
    downloader.download_all_symbols(symbols_to_process)