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
