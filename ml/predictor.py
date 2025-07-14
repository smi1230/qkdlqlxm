import logging
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D, LSTM,
                                     MaxPooling1D, Add, Flatten,
                                     LayerNormalization, MultiHeadAttention)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from typing import Dict, Any, Optional, List

try:
    from configs import settings
except ImportError:
    print("[CRITICAL ERROR] Failed to import modules in predictor.py.")
    raise SystemExit("Module loading failed.")

logger = logging.getLogger(__name__)

class DeepLearningPredictor:
    """
    [ 1.4단계 수정 완료 (원본 유지 및 일관성 최종 수정) ]
    - AI 모델의 생성, 컴파일, 훈련, 예측, 저장/로드를 총괄하는 클래스입니다.
    - 모델의 입출력 및 모든 관련 로직에서 'price_direction' 대신
      명확한 의미를 가진 'trade_outcome'을 사용하도록 최종 수정했습니다.
    - 단일 출력 모델에 맞게 컴파일 로직을 간소화했습니다.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir # 모델 관련 파일이 저장될 디렉토리
        self.model: Optional[Model] = None # Keras 모델 객체
        self.feature_names: Optional[List[str]] = None # 모델이 학습한 피쳐 이름 목록
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _create_optimizer(self, hyperparameters: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
        """
        하이퍼파라미터를 기반으로 학습률 스케줄러와 Adam 옵티마이저를 생성합니다.
        CosineDecayRestarts 스케줄러를 사용하여 훈련 중에 학습률을 동적으로 조절합니다.
        """
        lr_config = hyperparameters.get("learning_rate_scheduler", {})
        logger.info(f"Initializing CosineDecayRestarts learning rate scheduler with config: {lr_config}")

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr_config.get("initial_learning_rate", 0.001),
            first_decay_steps=lr_config.get("first_decay_steps", 350),
            t_mul=lr_config.get("t_mul", 1.01),
            m_mul=lr_config.get("m_mul", 0.998),
            alpha=lr_config.get("alpha", 1e-7)
        )

        optimizer = Adam(learning_rate=lr_schedule)

        # 혼합 정밀도 사용 시, LossScaleOptimizer로 옵티마이저를 감싸줍니다.
        if settings.MIXED_PRECISION:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        return optimizer

    def _build_model(self, sequence_length: int, num_features: int, config: Dict[str, Any], hyperparameters: Dict[str, Any]) -> Model:
        """
        [핵심 아키텍처] CNN-LSTM-Transformer 하이브리드 모델의 구조를 정의하고 생성합니다.
        """
        # 입력층: (시퀀스 길이, 피쳐 개수) 형태의 3D 텐서
        inputs = Input(shape=(sequence_length, num_features), name='input_sequence')

        # 1. CNN 블록: 데이터의 지역적인 패턴(local pattern)을 추출합니다.
        x = Conv1D(filters=config['cnn_filters'][0], kernel_size=3, padding='same', activation='relu')(inputs)
        x = Conv1D(filters=config['cnn_filters'][1], kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(config['dropout_rate'])(x)
        
        # 2. LSTM 블록: 시계열 데이터의 시간적 순서(temporal dependency)를 학습합니다.
        x = LSTM(units=config['lstm_units'][0], return_sequences=True)(x)
        lstm_output = Dropout(config['dropout_rate'])(x)
        
        # 3. Transformer (Self-Attention) 블록: 시퀀스 내의 중요한 부분에 더 집중(attention)합니다.
        norm_lstm_output = LayerNormalization(epsilon=1e-6)(lstm_output)
        attn_output = MultiHeadAttention(
            num_heads=config['transformer_heads'],
            key_dim=config['key_dim'],
            dropout=config['dropout_rate']
        )(norm_lstm_output, norm_lstm_output)
        
        # 4. 잔차 연결(Residual Connection) 및 정규화: LSTM 결과와 Attention 결과를 합쳐서 안정적인 학습을 돕습니다.
        x = Add()([lstm_output, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Flatten()(x)
        
        # 5. MLP (Fusion) 블록: 모든 추출된 정보를 융합하여 최종 예측을 준비합니다.
        x = Dense(units=config['fusion_units'], activation='relu')(x)
        mlp_output = Dropout(config['dropout_rate'])(x)

        # 6. 출력층 (Single-task): [핵심 수정] 이제 모델은 오직 'trade_outcome' 하나만 예측합니다.
        output_name = settings.ACTIVE_OUTPUTS[0]
        outputs = Dense(3, activation='softmax', name=output_name)(mlp_output)

        model = Model(inputs=inputs, outputs=outputs, name="TBM_Outcome_Predictor_v1")
        optimizer = self._create_optimizer(hyperparameters)

        # 모델을 컴파일합니다.
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        logger.info(f"Model '{model.name}' built for single output: '{output_name}'")
        model.summary(print_fn=logger.info)
        return model

    def train(self, train_generator, val_generator, hyperparameters: Dict[str, Any]):
        """모델 훈련을 시작합니다."""
        if self.model is None:
            # 미리 로드된 모델이 없으면, 새로운 모델을 생성합니다.
            logger.info("No pre-loaded model found. Building a new model from scratch...")
            try:
                self.feature_names = joblib.load(settings.FEATURES_PATH)
                logger.info(f"Loaded {len(self.feature_names)} feature names for model building from {settings.FEATURES_PATH}.")
            except FileNotFoundError:
                logger.error(f"{settings.FEATURES_PATH} not found! Cannot build model. Run preprocessing first.")
                return

            num_features = len(self.feature_names)
            sequence_length = settings.HYBRID_MODEL_CONFIG['sequence_length']
            self.model = self._build_model(sequence_length, num_features, settings.HYBRID_MODEL_CONFIG, hyperparameters)
        else:
            # 미리 로드된 모델이 있으면, 해당 모델로 훈련을 계속합니다.
            logger.info("Pre-loaded model found. Proceeding with training.")

        # 콜백(Callback) 설정: 훈련 중 특정 조건에서 특정 동작을 수행하도록 합니다.
        model_path = settings.model_path
        callbacks = [
            # ModelCheckpoint: 검증 손실(val_loss)이 가장 낮은 최적의 모델을 자동으로 저장합니다.
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
            # EarlyStopping: 검증 손실이 일정 기간(patience) 개선되지 않으면 훈련을 조기 종료하여 시간을 절약합니다.
            EarlyStopping(monitor='val_loss', patience=hyperparameters.get('early_stopping_patience', 10),
                          verbose=1, mode='min', restore_best_weights=False)
        ]

        logger.info("Starting model training...")
        self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=hyperparameters.get('epochs', 50),
            callbacks=callbacks
        )
        logger.info("Training finished.")

    def recompile_loaded_model(self, hyperparameters: Dict[str, Any]):
        """
        로드된 모델의 옵티마이저와 학습률을 새로 설정하여 다시 컴파일합니다.
        훈련을 이어갈 때 학습률을 초기화하고 싶을 때 사용됩니다.
        """
        if self.model is None:
            logger.error("No model is loaded. Cannot recompile. Load a model first.")
            return

        logger.info("Re-compiling the loaded model with a new optimizer and learning rate schedule...")
        new_optimizer = self._create_optimizer(hyperparameters)

        self.model.compile(optimizer=new_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logger.info(f"Model has been successfully re-compiled.")


    def predict(self, sequence_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """주어진 시퀀스 데이터에 대해 예측을 수행합니다."""
        if self.model is None or self.feature_names is None:
            logger.error("Model or feature names not loaded. Cannot perform prediction.")
            raise RuntimeError("Model or feature names not loaded.")

        if sequence_data.ndim == 2:
            sequence_data = np.expand_dims(sequence_data, axis=0)

        if sequence_data.shape[2] != len(self.feature_names):
                raise ValueError(f"Input feature count mismatch. Model expects {len(self.feature_names)}, but got {sequence_data.shape[2]}")

        predictions = self.model.predict(sequence_data)
        
        # 예측 결과를 처리하기 쉬운 딕셔너리 형태로 가공합니다.
        output_name = settings.ACTIVE_OUTPUTS[0]
        softmax_values = predictions[0]
        confidence_score = np.max(softmax_values).item()
        
        return {
            output_name: softmax_values,
            'confidence': confidence_score
        }


    def load_model_and_artifacts(self):
        """
        [핵심] settings.py에 정의된 경로를 사용하여 훈련된 모델(.keras)과
        피쳐 이름 목록(.joblib)을 불러옵니다.
        """
        try:
            model_path = settings.model_path
            features_path = settings.FEATURES_PATH

            self.model = load_model(model_path)
            self.feature_names = joblib.load(features_path)

            logger.info(f"Model '{os.path.basename(model_path)}' loaded from {model_path}")
            logger.info(f"{len(self.feature_names)} feature names loaded from {features_path}")
            logger.info(f"Loaded model outputs: {self.model.output_names}")
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}", exc_info=True)
            self.model = None
            self.feature_names = None
            raise
