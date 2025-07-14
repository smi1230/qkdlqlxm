import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple

try:
    from configs import settings
except ImportError:
    print("[CRITICAL ERROR] Failed to import settings file in enhanced_feature_engineer.")
    raise SystemExit()

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """
    [ 1단계 수정 완료 (TBM) ]
    - 이 클래스는 원본 데이터(가격, 거래량)로부터 AI 모델이 학습할 수 있는
      다양한 피쳐(Feature, 특성)와 타겟(Target, 정답)을 생성하는 모든 책임을 집니다.
    - create_multitask_targets 함수가 삼중 장벽 기법(TBM)을 사용하여
      실제 P&L 기반의 타겟을 생성하도록 수정되었습니다.
    """

    def __init__(self, volume_threshold: float = 1e-8):
        self.volume_threshold = volume_threshold
        self.epsilon = 1e-9
        # 이제 target_period는 TBM의 최대 보유 기간으로 대체됩니다.
        all_periods = settings.MA_PERIODS_FOR_PRICE_POS + settings.VWAP_PERIODS + \
                      settings.OFI_PERIODS + settings.VRSI_PERIODS + settings.RSI_PERIODS + settings.LEGACY_RSI_PERIODS + \
                      settings.VOLUME_ROC_PERIODS + [settings.BB_PERIOD, settings.MACD_SLOW, settings.ATR_PERIOD, 50]
        self.longest_lookback = max(all_periods) + 50

    # ==============================================================================
    # 1. 공개 인터페이스 (Public Interface)
    # ==============================================================================
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 피쳐 생성 과정을 총괄하는 메인 메서드입니다. (이 함수는 변경되지 않았습니다)"""
        if df.empty or len(df) < self.longest_lookback:
            return pd.DataFrame()
        
        try:
            features_df = pd.DataFrame(index=df.index)
            features_df = self._create_time_features(df, features_df)
            safe_volume = self._handle_zero_volume(df['volume'])
            
            features_df = self._create_base_features(df, features_df, safe_volume)
            features_df = self._create_paper_features(df, features_df, safe_volume)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Feature creation failed during create_features: {e}", exc_info=True)
            return pd.DataFrame()

    def create_multitask_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        [핵심 수정] 삼중 장벽 기법(TBM)을 사용하여 실제 P&L 기반의 타겟을 생성합니다.
        - 레이블 0: 손절 (하단 장벽 도달)
        - 레이블 1: 시간만료 (수직 장벽 도달)
        - 레이블 2: 수익 (상단 장벽 도달)
        """
        try:
            # 1. 동적 변동성(ATR) 및 장벽 계산
            atr = ta.atr(df['high'], df['low'], df['close'], length=settings.TBM_ATR_PERIOD).fillna(0)
            atr_scaled = atr * df['close'] * 0.01 # ATR을 가격 비율로 변환 (선택적)
            
            upper_barrier_mult = settings.TBM_PROFIT_TAKE_MULT
            lower_barrier_mult = settings.TBM_STOP_LOSS_MULT
            
            # 각 시점의 종가를 기준으로 상단/하단 장벽 가격을 계산합니다.
            upper_barrier = df['close'] + (atr * upper_barrier_mult)
            lower_barrier = df['close'] - (atr * lower_barrier_mult)
            
            max_hold_periods = settings.TBM_MAX_HOLD_PERIODS
            labels = pd.Series(np.nan, index=df.index) # 결과를 저장할 빈 시리즈

            # 2. 미래 가격 경로 탐색 및 레이블링
            # 벡터화된 연산을 위해 미래 가격 데이터를 미리 준비합니다.
            future_highs = [df['high'].shift(-i) for i in range(1, max_hold_periods + 1)]
            future_lows = [df['low'].shift(-i) for i in range(1, max_hold_periods + 1)]

            # 각 시점부터 미래 N개 캔들 동안의 최고가/최저가를 계산합니다.
            path_high = pd.concat(future_highs, axis=1).max(axis=1)
            path_low = pd.concat(future_lows, axis=1).min(axis=1)

            # 상단/하단 장벽 도달 여부를 확인합니다.
            hit_upper = path_high >= upper_barrier
            hit_lower = path_low <= lower_barrier

            # 레이블링: 수익(2), 손절(0), 시간만료(1)
            labels[hit_upper] = 2
            labels[hit_lower & ~hit_upper] = 0 # 상단과 하단을 동시에 넘는 경우, 수익을 우선
            labels.fillna(1, inplace=True) # 어느 장벽에도 닿지 않으면 시간만료(1)로 처리

            # 멀티태스크 학습을 위해 다른 타겟들도 계산합니다 (기존 로직 유지 또는 개선 가능).
            # 여기서는 price_direction만 TBM으로 교체하고 나머지는 비활성화합니다.
            future_volatility_raw = (path_high - path_low) / (df['close'] + self.epsilon)
            
            return {
                'price_direction': labels.astype(int),
                'volatility': np.log1p(future_volatility_raw),
                # 'volume' 타겟은 필요 시 추가 구현
            }
        except Exception as e:
            logger.error(f"TBM Target creation failed: {e}", exc_info=True)
            return {}

    # ==============================================================================
    # 2. 내부 헬퍼 함수 (Private Helper Functions) - 이 부분은 변경되지 않았습니다.
    # ==============================================================================
    def _create_time_features(self, original_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(original_df['timestamp'], unit='ms', errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(dt):
            features_df['month'] = dt.dt.month.astype(np.int8)
            features_df['day'] = dt.dt.day.astype(np.int8)
            features_df['hour'] = dt.dt.hour.astype(np.int8)
            features_df['minute'] = dt.dt.minute.astype(np.int8)
            features_df['day_of_week'] = dt.dt.dayofweek.astype(np.int8)
        return features_df

    def _handle_zero_volume(self, volume_series: pd.Series) -> pd.Series:
        volume = volume_series.copy()
        volume_no_tiny = np.where(volume < self.volume_threshold, 0, volume)
        volume_no_zero = pd.Series(volume_no_tiny).replace(0, np.nan)
        volume_interpolated = volume_no_zero.interpolate(method='linear', limit_direction='both')
        return volume_interpolated.fillna(self.epsilon)

    def _create_base_features(self, original_df: pd.DataFrame, features_df: pd.DataFrame, safe_volume: pd.Series) -> pd.DataFrame:
        close, open_, high, low = original_df['close'], original_df['open'], original_df['high'], original_df['low']
        
        features_df['o_to_c_pct'] = ((open_ / (close + self.epsilon)) - 1) * 100
        features_df['h_to_c_pct'] = ((high / (close + self.epsilon)) - 1) * 100
        features_df['l_to_c_pct'] = ((low / (close + self.epsilon)) - 1) * 100
        features_df['body_to_range_ratio'] = (abs(open_ - close)) / ((high - low) + self.epsilon)
        features_df['log_volume'] = np.log1p(safe_volume)
        features_df['atr_pct'] = ta.atr(high, low, close, length=settings.ATR_PERIOD) / (close + self.epsilon) * 100

        features_df['ema_short'] = ta.ema(close, length=settings.EMA_SHORT_PERIOD)
        features_df['ema_mid'] = ta.ema(close, length=settings.EMA_MID_PERIOD)
        ema_ribbon_dfs = [ta.ema(close, length=p) for p in settings.EMA_RIBBON_PERIODS]
        if ema_ribbon_dfs:
            ema_ribbon = pd.concat(ema_ribbon_dfs, axis=1)
            features_df['ema_ribbon_spread'] = (ema_ribbon.max(axis=1) - ema_ribbon.min(axis=1)) / (close + self.epsilon) * 100
        
        for p in settings.MA_PERIODS_FOR_PRICE_POS:
            features_df[f'close_to_ma_{p}_pct'] = (close / (ta.sma(close, length=p) + self.epsilon) - 1) * 100

        bb_df = ta.bbands(close, length=settings.BB_PERIOD, std=settings.BB_STD_DEV)
        if bb_df is not None and not bb_df.empty:
            features_df['bb_position'] = (close - bb_df[f'BBL_{settings.BB_PERIOD}_{settings.BB_STD_DEV:.1f}']) / (bb_df[f'BBU_{settings.BB_PERIOD}_{settings.BB_STD_DEV:.1f}'] - bb_df[f'BBL_{settings.BB_PERIOD}_{settings.BB_STD_DEV:.1f}'] + self.epsilon)
            features_df['bb_width_pct'] = (bb_df[f'BBU_{settings.BB_PERIOD}_{settings.BB_STD_DEV:.1f}'] - bb_df[f'BBL_{settings.BB_PERIOD}_{settings.BB_STD_DEV:.1f}']) / (close + self.epsilon) * 100

        for p in settings.RSI_PERIODS + settings.LEGACY_RSI_PERIODS:
            features_df[f'rsi_{p}'] = ta.rsi(close, length=p)
            
        macd_df = ta.macd(close, fast=settings.MACD_FAST, slow=settings.MACD_SLOW, signal=settings.MACD_SIGNAL)
        if macd_df is not None and not macd_df.empty:
            features_df['macd_hist_pct'] = macd_df[f'MACDh_{settings.MACD_FAST}_{settings.MACD_SLOW}_{settings.MACD_SIGNAL}'] / (close + self.epsilon) * 100
        
        stoch_fast_df = ta.stoch(high, low, close, k=settings.STOCH_K_FAST, d=settings.STOCH_D, smooth_k=settings.STOCH_SMOOTH)
        if stoch_fast_df is not None and not stoch_fast_df.empty:
            features_df['stoch_k_fast'] = stoch_fast_df[f'STOCHk_{settings.STOCH_K_FAST}_{settings.STOCH_D}_{settings.STOCH_SMOOTH}']
            features_df['stoch_d_fast'] = stoch_fast_df[f'STOCHd_{settings.STOCH_K_FAST}_{settings.STOCH_D}_{settings.STOCH_SMOOTH}']
        
        stoch_slow_df = ta.stoch(high, low, close, k=settings.STOCH_K_SLOW, d=settings.STOCH_D, smooth_k=settings.STOCH_SMOOTH)
        if stoch_slow_df is not None and not stoch_slow_df.empty:
            features_df['stoch_k_slow'] = stoch_slow_df[f'STOCHk_{settings.STOCH_K_SLOW}_{settings.STOCH_D}_{settings.STOCH_SMOOTH}']
            features_df['stoch_d_slow'] = stoch_slow_df[f'STOCHd_{settings.STOCH_K_SLOW}_{settings.STOCH_D}_{settings.STOCH_SMOOTH}']
        
        mfi_series = ta.mfi(high, low, close, safe_volume, length=settings.MFI_PERIOD)
        if mfi_series is not None:
            features_df[f'mfi_{settings.MFI_PERIOD}'] = mfi_series
            
        return features_df

    def _create_paper_features(self, original_df: pd.DataFrame, features_df: pd.DataFrame, safe_volume: pd.Series) -> pd.DataFrame:
        close, high, low = original_df['close'], original_df['high'], original_df['low']
        tp = (high + low + close) / 3
        for p in settings.VWAP_PERIODS:
            vwap = (tp * safe_volume).rolling(p).sum() / (safe_volume.rolling(p).sum() + self.epsilon)
            price_diff_sq = ((tp - vwap) ** 2) * safe_volume
            cum_price_diff_sq_sum = price_diff_sq.rolling(p).sum()
            cum_volume_sum = safe_volume.rolling(p).sum()
            vwap_vol = np.sqrt(cum_price_diff_sq_sum / (cum_volume_sum + self.epsilon))
            features_df[f'paper_vwap_deviation_{p}_pct'] = ((close - vwap) / (close + self.epsilon)) * 100
            features_df[f'paper_vwap_volatility_{p}'] = vwap_vol
            features_df[f'paper_vwap_slope_{p}'] = vwap.diff(3)

        price_change = close.diff()
        tick_dir = np.sign(price_change).replace(0, np.nan).ffill().fillna(0)
        signed_vol = tick_dir * safe_volume
        dollar_volume = close * safe_volume
        
        for p in settings.VRSI_PERIODS:
            gains = np.where(price_change > 0, price_change * safe_volume, 0)
            losses = np.where(price_change < 0, -price_change * safe_volume, 0)
            rs = pd.Series(gains).rolling(p).mean() / (pd.Series(losses).rolling(p).mean() + self.epsilon)
            features_df[f'paper_vrsi_{p}'] = 100 - (100 / (1 + rs))

        for w in settings.OFI_PERIODS:
            ofi_ratio = signed_vol.rolling(w).sum() / (safe_volume.rolling(w).sum() + self.epsilon)
            features_df[f'paper_ofi_ratio_{w}'] = ofi_ratio
            features_df[f'paper_ofi_zscore_{w}'] = (ofi_ratio - ofi_ratio.rolling(50).mean()) / (ofi_ratio.rolling(50).std() + self.epsilon)
        
        for w in settings.AMIHUD_PERIODS:
            amihud_series = (close.pct_change().abs() / (dollar_volume + self.epsilon))
            features_df[f'paper_amihud_{w}'] = amihud_series.rolling(w).mean()

        for p in settings.VOLUME_ROC_PERIODS:
            features_df[f'volume_roc_{p}'] = safe_volume.pct_change(p) * 100
            
        features_df['volume_expansion'] = (safe_volume.rolling(settings.VOLUME_MA_SHORT_PERIOD).mean() / (safe_volume.rolling(settings.VOLUME_MA_LONG_PERIOD).mean() + self.epsilon) - 1) * 100
        features_df['volume_percentile_50'] = safe_volume.rolling(settings.VOLUME_PERCENTILE_WINDOW).rank(pct=True) * 100
        
        vwema_fast = self._volume_weighted_ema(close, safe_volume, settings.MACD_FAST)
        vwema_slow = self._volume_weighted_ema(close, safe_volume, settings.MACD_SLOW)
        vwmacd_line = vwema_fast - vwema_slow
        vwmacd_signal = self._volume_weighted_ema(vwmacd_line, safe_volume, settings.MACD_SIGNAL)
        features_df['paper_vwmacd_hist'] = vwmacd_line - vwmacd_signal
        
        return features_df

    def _volume_weighted_ema(self, prices: pd.Series, volumes: pd.Series, period: int) -> pd.Series:
        alpha = 2 / (period + 1)
        vwema = np.zeros_like(prices.values, dtype=float)
        
        if prices.empty: return pd.Series(vwema, index=prices.index)
            
        vwema[0] = prices.iloc[0]
        
        vol_rolling_mean = volumes.rolling(window=period, min_periods=1).mean().fillna(volumes.expanding().mean()).values
        
        for i in range(1, len(prices)):
            vol_mean_safe = vol_rolling_mean[i] if vol_rolling_mean[i] > 0 else self.epsilon
            vol_weight = volumes.iloc[i] / vol_mean_safe
            effective_alpha = alpha * min(vol_weight, 2.0)
            vwema[i] = effective_alpha * prices.iloc[i] + (1 - effective_alpha) * vwema[i-1]
            
        return pd.Series(vwema, index=prices.index)
