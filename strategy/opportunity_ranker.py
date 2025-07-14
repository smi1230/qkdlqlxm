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
    [ 1단계 수정 완료 (TBM) ]
    - AI 예측 점수 계산 시, TBM 기반의 3-클래스({0:손절, 1:중립, 2:수익})
      출력에 맞게 가중치를 [-1, 0, 1]로 수정했습니다.
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
        
        return all_opportunities

    def _analyze_single_symbol(self, symbol: str) -> Optional[Dict]:
        df = self.data_handler.get_dataframe(symbol)
        if df is None or len(df) < self.feature_engineer.longest_lookback: return None

        features_df = self.feature_engineer.create_features(df)
        if features_df.empty: return None

        latest_sequence_df = features_df.iloc[-self.sequence_length:]
        if len(latest_sequence_df) < self.sequence_length: return None
        
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
        price_direction_probs = predictions.get('price_direction')
        confidence = predictions.get('confidence', 0.0)
        volatility = predictions.get('volatility', 0.0)
        
        if price_direction_probs is None:
            logger.warning("'_calculate_dl_score' received predictions without 'price_direction'. Returning neutral score.")
            return {"direction_score": 0.0, "confidence": 0.0, "volatility": 0.0, "risk_adjusted_score": 0.0}

        # [핵심 수정] 가중치를 3-클래스({0:손절, 1:중립, 2:수익})에 맞게 변경
        weights = np.array([-1, 0, 1])
        direction_score = np.sum(price_direction_probs * weights)
        risk_adjusted_score = direction_score * (1 + confidence * self.dl_confidence_weight) / (1 + volatility * settings.VOLATILITY_ADJUSTMENT_FACTOR)
        
        return {
            "direction_score": float(direction_score),
            "confidence": float(confidence),
            "volatility": float(volatility),
            "volume_pred_log": float(predictions.get('volume', 0.0)),
            "risk_adjusted_score": float(risk_adjusted_score)
        }

    def _calculate_technical_score(self, features: pd.Series) -> float:
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
        final_score = (dl_score * self.composite_dl_weight) + (technical_score * self.composite_technical_weight)

        if np.sign(dl_score) == np.sign(technical_score) and dl_score != 0:
            final_score *= settings.SIGNAL_ALIGNMENT_BONUS

        return final_score
