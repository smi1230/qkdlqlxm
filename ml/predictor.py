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
    [ v5.13 - 경로 설정 중앙 관리 적용 ]
    - load_model_and_artifacts 함수에서 하드코딩된 파일명 인자를 '제거'.
    - 모든 모델 및 아티팩트 경로는 settings.py의 설정값(settings.model_path, settings.FEATURES_PATH)을
      직접 사용하도록 '수정'하여 설정 중앙 관리 원칙을 준수.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir # model_dir은 여전히 로그나 임시 파일 저장 등을 위해 유지될 수 있음
        self.model: Optional[Model] = None
        self.feature_names: Optional[List[str]] = None
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _create_optimizer(self, hyperparameters: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
        """
        하이퍼파라미터를 기반으로 학습률 스케줄러와 Adam 옵티마이저를 생성합니다.
        """
        lr_config = hyperparameters.get("learning_rate_scheduler", {})
        logger.info(f"Initializing CosineDecayRestarts learning rate scheduler with config: {lr_config}")

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr_config.get("initial_learning_rate", 0.001),
            first_decay_steps=lr_config.get("first_decay_steps", 1000),
            t_mul=lr_config.get("t_mul", 2.0),
            m_mul=lr_config.get("m_mul", 1.0),
            alpha=lr_config.get("alpha", 0.0)
        )

        optimizer = Adam(learning_rate=lr_schedule)

        if settings.MIXED_PRECISION:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        return optimizer

    def _build_model(self, sequence_length: int, num_features: int, config: Dict[str, Any], hyperparameters: Dict[str, Any]) -> Model:
        """
        설정 파일 기반으로 모델 아키텍처를 구성.
        """
        inputs = Input(shape=(sequence_length, num_features), name='input_sequence')

        x = Conv1D(filters=config['cnn_filters'][0], kernel_size=3, padding='same', activation='relu')(inputs)
        x = Conv1D(filters=config['cnn_filters'][1], kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(config['dropout_rate'])(x)
        x = LSTM(units=config['lstm_units'][0], return_sequences=True)(x)
        lstm_output = Dropout(config['dropout_rate'])(x)
        norm_lstm_output = LayerNormalization(epsilon=1e-6)(lstm_output)
        attn_output = MultiHeadAttention(
            num_heads=config['transformer_heads'],
            key_dim=config['key_dim'],
            dropout=config['dropout_rate']
        )(norm_lstm_output, norm_lstm_output)
        x = Add()([lstm_output, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Flatten()(x)
        x = Dense(units=config['fusion_units'], activation='relu')(x)
        mlp_output = Dropout(config['dropout_rate'])(x)

        outputs = {}
        loss_dict = {}
        metrics_dict = {}
        active_loss_weights = {}

        if 'price_direction' in settings.ACTIVE_OUTPUTS:
            outputs['price_direction'] = Dense(5, activation='softmax', name='price_direction')(mlp_output)
            loss_dict['price_direction'] = tf.keras.losses.SparseCategoricalCrossentropy()
            metrics_dict['price_direction'] = 'accuracy'
            active_loss_weights['price_direction'] = settings.MULTITASK_LOSS_WEIGHTS.get('price_direction', 1.0)

        if 'volatility' in settings.ACTIVE_OUTPUTS:
            outputs['volatility'] = Dense(1, activation='linear', name='volatility')(mlp_output)
            loss_dict['volatility'] = tf.keras.losses.MeanSquaredError()
            metrics_dict['volatility'] = 'mae'
            active_loss_weights['volatility'] = settings.MULTITASK_LOSS_WEIGHTS.get('volatility', 0.2)

        if 'volume' in settings.ACTIVE_OUTPUTS:
            outputs['volume'] = Dense(1, activation='linear', name='volume')(mlp_output)
            loss_dict['volume'] = tf.keras.losses.MeanSquaredError()
            metrics_dict['volume'] = 'mae'
            active_loss_weights['volume'] = settings.MULTITASK_LOSS_WEIGHTS.get('volume', 0.1)
        
        if not outputs:
            raise ValueError("No active outputs defined in settings.py. 'ACTIVE_OUTPUTS' cannot be empty.")

        if len(outputs) == 1:
            model_outputs = list(outputs.values())[0]
            model_name = f"Single_Task_Model_v5_13_{list(outputs.keys())[0]}"
        else:
            model_outputs = outputs
            model_name = "Hybrid_CNN_LSTM_Transformer_v5_13"

        model = Model(inputs=inputs, outputs=model_outputs, name=model_name)
        optimizer = self._create_optimizer(hyperparameters)

        if len(outputs) == 1:
            single_loss = list(loss_dict.values())[0]
            single_metrics = list(metrics_dict.values())
            model.compile(optimizer=optimizer, loss=single_loss, metrics=single_metrics)
        else:
            model.compile(optimizer=optimizer, loss=loss_dict, loss_weights=active_loss_weights, metrics=metrics_dict)

        logger.info(f"Model '{model.name}' built with active outputs: {list(outputs.keys())}")
        model.summary(print_fn=logger.info)
        return model

    def train(self, train_generator, val_generator, hyperparameters: Dict[str, Any]):
        if self.model is None:
            logger.info("No pre-loaded model found. Building a new model from scratch...")
            try:
                # [수정] settings.py에 정의된 경로를 사용
                self.feature_names = joblib.load(settings.FEATURES_PATH)
                logger.info(f"Loaded {len(self.feature_names)} feature names for model building from {settings.FEATURES_PATH}.")
            except FileNotFoundError:
                logger.error(f"{settings.FEATURES_PATH} not found! Cannot build model. Run preprocessing first.")
                return

            num_features = len(self.feature_names)
            sequence_length = settings.HYBRID_MODEL_CONFIG['sequence_length']
            self.model = self._build_model(sequence_length, num_features, settings.HYBRID_MODEL_CONFIG, hyperparameters)
        else:
            logger.info("Pre-loaded model found. Proceeding with training.")
            try:
                current_step = self.model.optimizer.iterations.numpy()
                logger.info(f"Optimizer state loaded. Resuming from training step: {current_step}")
                if hasattr(self.model.optimizer.learning_rate, 'initial_learning_rate'):
                     current_lr = self.model.optimizer.learning_rate(current_step).numpy()
                     logger.info(f"Current learning rate in schedule: {current_lr:.8f}")
            except Exception:
                 logger.info("Optimizer state seems to be fresh or could not be read. Starting with initial LR of the schedule.")

        # [수정] settings.py에 정의된 경로를 사용
        model_path = settings.model_path
        callbacks = [
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
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
        if self.model is None:
            logger.error("No model is loaded. Cannot recompile. Load a model first.")
            return

        logger.info("Re-compiling the loaded model with a new optimizer and learning rate schedule...")
        new_optimizer = self._create_optimizer(hyperparameters)

        active_outputs = settings.ACTIVE_OUTPUTS
        loss_dict = {}
        metrics_dict = {}
        active_loss_weights = {}

        if 'price_direction' in active_outputs:
            loss_dict['price_direction'] = tf.keras.losses.SparseCategoricalCrossentropy()
            metrics_dict['price_direction'] = 'accuracy'
            active_loss_weights['price_direction'] = settings.MULTITASK_LOSS_WEIGHTS.get('price_direction', 1.0)
        if 'volatility' in active_outputs:
            loss_dict['volatility'] = tf.keras.losses.MeanSquaredError()
            metrics_dict['volatility'] = 'mae'
            active_loss_weights['volatility'] = settings.MULTITASK_LOSS_WEIGHTS.get('volatility', 0.2)
        if 'volume' in active_outputs:
            loss_dict['volume'] = tf.keras.losses.MeanSquaredError()
            metrics_dict['volume'] = 'mae'
            active_loss_weights['volume'] = settings.MULTITASK_LOSS_WEIGHTS.get('volume', 0.1)

        if len(active_outputs) == 1:
            single_loss = list(loss_dict.values())[0]
            single_metrics = list(metrics_dict.values())
            self.model.compile(optimizer=new_optimizer, loss=single_loss, metrics=single_metrics)
        else:
            self.model.compile(optimizer=new_optimizer, loss=loss_dict, loss_weights=active_loss_weights, metrics=metrics_dict)
        
        logger.info(f"Model has been successfully re-compiled with active outputs: {active_outputs}")


    def predict_multitask(self, sequence_data: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.model is None or self.feature_names is None:
            logger.error("Model or feature names not loaded. Cannot perform prediction.")
            raise RuntimeError("Model or feature names not loaded.")

        if sequence_data.ndim == 2:
            sequence_data = np.expand_dims(sequence_data, axis=0)

        if sequence_data.shape[2] != len(self.feature_names):
                raise ValueError(f"Input feature count mismatch. Model expects {len(self.feature_names)}, but got {sequence_data.shape[2]}")

        predictions = self.model.predict(sequence_data)
        
        processed_result = {}

        if isinstance(predictions, dict):
            for key, value in predictions.items():
                if key == 'price_direction':
                    processed_result[key] = value[0]
                else:
                    processed_result[key] = value[0].item()
        elif isinstance(predictions, np.ndarray):
            output_name = settings.ACTIVE_OUTPUTS[0]
            processed_result[output_name] = predictions[0]
        else:
            logger.error(f"Unexpected prediction output type: {type(predictions)}")
            return None

        if 'price_direction' in processed_result:
            softmax_values = processed_result['price_direction']
            confidence_score = np.max(softmax_values).item()
            processed_result['confidence'] = confidence_score
            logger.debug(f"Derived confidence score: {confidence_score:.4f}")

        return processed_result


    def load_model_and_artifacts(self):
        """
        [핵심 수정] settings.py에 정의된 경로를 사용하여 모델과 아티팩트를 로드합니다.
        더 이상 파일명을 인자로 받지 않습니다.
        """
        try:
            # settings.py에서 직접 경로를 가져옴
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
