# /strategy/opportunity_ranker.py

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any

try:
    from ml.predictor import DeepLearningPredictor
    from ml.enhanced_feature_engineer import EnhancedFeatureEngineer
    from core.data_handler import DataHandler
    from configs import settings
except ImportError:
    print("[CRITICAL ERROR] Failed to import modules in opportunity_ranker.py.")
    raise SystemExit("Module loading failed.")


logger = logging.getLogger(__name__)

class OpportunityRanker:
    """
    [ v5.2 - 오류 수정 ]
    - AI의 멀티태스크 예측과 기술적 지표를 결합하여 모든 거래 가능 종목의 기회를 분석.
    - get_all_opportunities 함수가 모든 분석 결과를 반환하도록 수정.
    """

    def __init__(self,
                 model_predictor: DeepLearningPredictor,
                 data_handler: DataHandler,
                 feature_engineer: EnhancedFeatureEngineer):
        self.predictor = model_predictor
        self.data_handler = data_handler
        self.feature_engineer = feature_engineer
        self.sequence_length = settings.HYBRID_MODEL_CONFIG['sequence_length']
        self.dl_confidence_weight = settings.DL_CONFIDENCE_WEIGHT
        self.composite_dl_weight = settings.COMPOSITE_DL_WEIGHT
        self.composite_technical_weight = settings.COMPOSITE_TECHNICAL_WEIGHT
        self.min_dl_confidence = settings.MIN_DL_CONFIDENCE_THRESHOLD
        self.min_composite_score = settings.MIN_COMPOSITE_SCORE_THRESHOLD
        self.epsilon = 1e-9
        logger.info("OpportunityRanker initialized.")

    def get_all_opportunities(self, tradeable_symbols: List[str]) -> List[Dict]:
        """
        [✨ 핵심 수정] 모든 거래 가능 종목에 대해 기회 분석을 수행하고, 필터링되지 않은 전체 결과를 반환합니다.
        """
        all_opportunities = []
        for symbol in tradeable_symbols:
            try:
                opportunity = self._analyze_single_symbol(symbol)
                if opportunity:
                    all_opportunities.append(opportunity)
            except Exception as e:
                logger.error(f"[{symbol}] Error during opportunity analysis: {e}", exc_info=False)

        if not all_opportunities:
            logger.info("No actionable opportunities found in this cycle.")
        
        # 정렬이나 필터링 없이 모든 분석 결과를 반환
        return all_opportunities

    def _analyze_single_symbol(self, symbol: str) -> Optional[Dict]:
        """단일 종목에 대한 종합적인 기회 분석 (데이터 조회 -> 피처 생성 -> AI 예측 -> 점수 계산)"""
        df = self.data_handler.get_dataframe(symbol)
        if df is None or len(df) < self.feature_engineer.longest_lookback: return None

        features_df = self.feature_engineer.create_features(df)
        if features_df.empty: return None

        latest_sequence_df = features_df.iloc[-self.sequence_length:]
        if len(latest_sequence_df) < self.sequence_length: return None
        
        # feature_names가 로드되었는지 확인
        if self.predictor.feature_names is None:
            logger.error("Feature names are not loaded in the predictor. Cannot proceed.")
            return None
            
        sequence_to_predict = latest_sequence_df[self.predictor.feature_names]
        sequence_3d = np.expand_dims(sequence_to_predict.values, axis=0)
        
        try:
            predictions = self.predictor.predict_multitask(sequence_3d)
            if not predictions: return None
        except Exception as e:
            logger.warning(f"[{symbol}] AI multitask prediction failed: {e}"); return None
            
        if predictions.get('confidence', 0.0) < self.min_dl_confidence: return None

        dl_score_details = self._calculate_dl_score(predictions)
        latest_features = features_df.iloc[-1]
        technical_score = self._calculate_technical_score(latest_features)
        final_composite_score = self._calculate_composite_score(dl_score_details['risk_adjusted_score'], technical_score)

        if abs(final_composite_score) < self.min_composite_score: return None

        return {
            "symbol": symbol,
            "side": "Buy" if final_composite_score > 0 else "Sell",
            "final_composite_score": final_composite_score,
            "dl_score_details": dl_score_details,
            "technical_score": technical_score,
            "latest_features": latest_features.to_dict()
        }

    def _calculate_dl_score(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """AI의 멀티태스크 예측 결과를 하나의 리스크 조정 점수로 변환 (누락된 예측값 안전 처리)"""
        price_direction_probs = predictions.get('price_direction')
        confidence = predictions.get('confidence', 0.0)
        volatility = predictions.get('volatility', 0.0)
        volume_pred = predictions.get('volume', 0.0)

        if price_direction_probs is None:
            logger.warning("'_calculate_dl_score' received predictions without 'price_direction'. Returning neutral score.")
            return {
                "direction_score": 0.0, "confidence": 0.0, "volatility": 0.0,
                "volume_pred_log": 0.0, "risk_adjusted_score": 0.0
            }

        weights = np.array([-2, -1, 0, 1, 2])
        direction_score = np.sum(price_direction_probs * weights)
        risk_adjusted_score = direction_score * (1 + confidence * self.dl_confidence_weight) / (1 + volatility * settings.VOLATILITY_ADJUSTMENT_FACTOR)
        
        return {
            "direction_score": float(direction_score),
            "confidence": float(confidence),
            "volatility": float(volatility),
            "volume_pred_log": float(volume_pred),
            "risk_adjusted_score": float(risk_adjusted_score)
        }

    def _calculate_technical_score(self, features: pd.Series) -> float:
        """다변량 기술 지표를 분석하여 단일 점수를 계산."""
        try:
            rsi = features.get(f'rsi_{settings.RSI_PERIODS[0]}', 50.0)
            rsi_score = -((rsi - 50.0) / 25.0)

            macd_hist = features.get('macd_hist_pct', 0.0)
            macd_score = np.clip(macd_hist, -2.0, 2.0)

            ofi_zscore = features.get(f'paper_ofi_zscore_{settings.OFI_PERIODS[-1]}', 0.0)
            ofi_score = np.clip(ofi_zscore, -2.0, 2.0)

            total_ta_score = (rsi_score * 0.4) + (macd_score * 0.3) + (ofi_score * 0.3)
            
            return np.clip(total_ta_score, -1.5, 1.5)
        
        except Exception as e:
            logger.warning(f"Failed to calculate technical score: {e}")
            return 0.0

    def _calculate_composite_score(self, dl_score: float, technical_score: float) -> float:
        """DL 점수와 TA 점수를 최종 가중 합산하여 반환"""
        final_score = (dl_score * self.composite_dl_weight) + (technical_score * self.composite_technical_weight)

        if np.sign(dl_score) == np.sign(technical_score) and dl_score != 0:
            final_score *= settings.SIGNAL_ALIGNMENT_BONUS

        return final_score
