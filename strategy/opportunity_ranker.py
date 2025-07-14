import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any

try:
    # 시스템의 다른 핵심 모듈들을 임포트합니다.
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
    [ 1.4단계 수정 완료 (원본 유지 및 일관성 최종 수정) ]
    - AI 예측 점수 계산 시, 'price_direction' 대신 명확한 의미의
      'trade_outcome'을 참조하도록 최종 수정했습니다.
    """

    def __init__(self,
                 model_predictor: DeepLearningPredictor,
                 data_handler: DataHandler,
                 feature_engineer: EnhancedFeatureEngineer):
        # 외부에서 생성된 핵심 모듈 인스턴스들을 주입받습니다.
        self.predictor = model_predictor
        self.data_handler = data_handler
        self.feature_engineer = feature_engineer
        # settings.py에서 각종 가중치 및 임계값을 불러와 멤버 변수로 저장합니다.
        self.sequence_length = settings.HYBRID_MODEL_CONFIG['sequence_length']
        self.dl_confidence_weight = settings.DL_CONFIDENCE_WEIGHT
        self.composite_dl_weight = settings.COMPOSITE_DL_WEIGHT
        self.composite_technical_weight = settings.COMPOSITE_TECHNICAL_WEIGHT
        self.min_dl_confidence = settings.MIN_DL_CONFIDENCE_THRESHOLD
        self.min_composite_score = settings.MIN_COMPOSITE_SCORE_THRESHOLD
        self.epsilon = 1e-9 # 0으로 나누는 것을 방지하기 위한 매우 작은 값
        logger.info("OpportunityRanker initialized.")

    def get_all_opportunities(self, tradeable_symbols: List[str]) -> List[Dict]:
        """
        [핵심 기능] 거래 가능한 모든 종목에 대해 기회 분석을 수행하고,
        필터링되지 않은 전체 분석 결과를 리스트 형태로 반환합니다.
        """
        all_opportunities = []
        for symbol in tradeable_symbols:
            try:
                # 각 종목에 대해 개별 분석을 수행합니다.
                opportunity = self._analyze_single_symbol(symbol)
                if opportunity:
                    all_opportunities.append(opportunity)
            except Exception as e:
                logger.error(f"[{symbol}] Error during opportunity analysis: {e}", exc_info=False)

        if not all_opportunities:
            logger.info("No actionable opportunities found in this cycle.")
        
        # 정렬이나 추가 필터링 없이, 분석된 모든 기회 정보를 그대로 반환합니다.
        return all_opportunities

    def _analyze_single_symbol(self, symbol: str) -> Optional[Dict]:
        """단일 종목에 대한 종합적인 기회 분석을 수행하는 내부 메서드입니다."""
        # 1. 데이터 조회: DataHandler로부터 해당 종목의 최신 데이터를 가져옵니다.
        df = self.data_handler.get_dataframe(symbol)
        if df is None or len(df) < self.feature_engineer.longest_lookback: return None

        # 2. 피쳐 생성: 가져온 데이터로 AI 모델이 사용할 피쳐(특성)들을 생성합니다.
        features_df = self.feature_engineer.create_features(df)
        if features_df.empty: return None

        # 3. AI 예측: 최신 데이터 시퀀스를 모델에 입력하여 미래를 예측합니다.
        latest_sequence_df = features_df.iloc[-self.sequence_length:]
        if len(latest_sequence_df) < self.sequence_length: return None
        
        if self.predictor.feature_names is None:
            logger.error("Feature names are not loaded in the predictor. Cannot proceed.")
            return None
            
        sequence_to_predict = latest_sequence_df[self.predictor.feature_names]
        sequence_3d = np.expand_dims(sequence_to_predict.values, axis=0)
        
        try:
            predictions = self.predictor.predict(sequence_3d)
            if not predictions: return None
        except Exception as e:
            logger.warning(f"[{symbol}] AI prediction failed: {e}"); return None
            
        # 4. 점수 계산: 예측 결과와 기술적 지표를 종합하여 최종 점수를 계산합니다.
        # AI 예측의 신뢰도가 너무 낮으면 더 이상 분석하지 않고 기회를 무시합니다.
        if predictions.get('confidence', 0.0) < self.min_dl_confidence: return None

        dl_score_details = self._calculate_dl_score(predictions)
        latest_features = features_df.iloc[-1]
        technical_score = self._calculate_technical_score(latest_features)
        final_composite_score = self._calculate_composite_score(dl_score_details['risk_adjusted_score'], technical_score)

        # 최종 종합 점수가 너무 낮으면 거래 가치가 없는 것으로 판단하고 기회를 무시합니다.
        if abs(final_composite_score) < self.min_composite_score: return None

        # 5. 최종 결과 반환: 모든 분석 결과를 담은 딕셔너리를 반환합니다.
        return {
            "symbol": symbol,
            "side": "Buy" if final_composite_score > 0 else "Sell",
            "final_composite_score": final_composite_score,
            "dl_score_details": dl_score_details,
            "technical_score": technical_score,
            "latest_features": latest_features.to_dict()
        }

    def _calculate_dl_score(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """AI의 예측 결과를 하나의 리스크 조정 점수(risk_adjusted_score)로 변환합니다."""
        # [핵심 수정] 'price_direction' 대신 'trade_outcome'을 참조합니다.
        outcome_probs = predictions.get('trade_outcome')
        confidence = predictions.get('confidence', 0.0)
        
        if outcome_probs is None:
            logger.warning("'_calculate_dl_score' received predictions without 'trade_outcome'. Returning neutral score.")
            return {"direction_score": 0.0, "confidence": 0.0, "risk_adjusted_score": 0.0}

        # 가중치: [손절(-1), 시간만료(0), 수익(1)] - 클래스 0, 1, 2 순서에 대응
        weights = np.array([-1, 0, 1])
        direction_score = np.sum(outcome_probs * weights)
        
        # 리스크 조정 점수 (변동성 예측이 없으므로 단순화)
        risk_adjusted_score = direction_score * (1 + confidence * self.dl_confidence_weight)
        
        return {
            "direction_score": float(direction_score),
            "confidence": float(confidence),
            "risk_adjusted_score": float(risk_adjusted_score)
        }

    def _calculate_technical_score(self, features: pd.Series) -> float:
        """RSI, MACD, OFI 등 여러 기술적 지표를 종합하여 단일 기술 분석 점수를 계산합니다."""
        try:
            # RSI 점수: 50을 기준으로 과매수/과매도 상태를 점수화합니다.
            rsi = features.get(f'rsi_{settings.RSI_PERIODS[0]}', 50.0)
            rsi_score = -((rsi - 50.0) / 25.0) # 75이면 -1점, 25이면 +1점

            # MACD 점수: MACD 히스토그램 값으로 추세의 강도를 점수화합니다.
            macd_hist = features.get('macd_hist_pct', 0.0)
            macd_score = np.clip(macd_hist, -2.0, 2.0)

            # OFI 점수: 주문 흐름 불균형의 Z-score로 매수/매도 압력을 점수화합니다.
            ofi_zscore = features.get(f'paper_ofi_zscore_{settings.OFI_PERIODS[-1]}', 0.0)
            ofi_score = np.clip(ofi_zscore, -2.0, 2.0)

            # 각 지표의 점수를 설정된 가중치에 따라 합산합니다.
            total_ta_score = (rsi_score * 0.4) + (macd_score * 0.3) + (ofi_score * 0.3)
            
            return np.clip(total_ta_score, -1.5, 1.5)
        
        except Exception as e:
            logger.warning(f"Failed to calculate technical score: {e}")
            return 0.0

    def _calculate_composite_score(self, dl_score: float, technical_score: float) -> float:
        """AI 점수와 기술 분석 점수를 최종적으로 가중 합산하여 종합 점수를 계산합니다."""
        final_score = (dl_score * self.composite_dl_weight) + (technical_score * self.composite_technical_weight)

        # AI 예측 방향과 기술적 분석 방향이 일치하면 보너스 점수를 부여합니다.
        if np.sign(dl_score) == np.sign(technical_score) and dl_score != 0:
            final_score *= settings.SIGNAL_ALIGNMENT_BONUS

        return final_score
