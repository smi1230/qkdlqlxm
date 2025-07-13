# /train.py

import os
import sys
import logging
import warnings
import tensorflow as tf

# ==============================================================================
# [ 1. 시스템 경로 설정 및 전역 설정 ]
# ==============================================================================
warnings.simplefilter(action='ignore', category=FutureWarning)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# ==============================================================================
# [ 2. 모듈 임포트 ]
# ==============================================================================
try:
    from configs.logging_config import setup_logging
    setup_logging()

    from configs import settings
    from ml.gpu_optimizer import GpuOptimizer
    from ml.model_trainer import ModelTrainer
    from ml.predictor import DeepLearningPredictor
except ImportError as e:
    print(f"[CRITICAL ERROR] Failed to import essential modules: {e}")
    print("Please ensure the script is run from the project root or the path is correctly set.")
    sys.exit(1)

logger = logging.getLogger(__name__)

def main():
    """
    [ v5.7 - 훈련 제어 로직 개선 ]
    AI 모델 훈련 파이프라인을 실행하는 메인 함수.
    - settings.py의 RESUME_TRAINING, REPLACE_OPTIMIZER_ON_RESUME 플래그를 통해
      훈련 이어하기 및 옵티마이저 교체 여부를 명확하게 제어합니다.
    """
    logger.info("====================================================================================")
    logger.info("========== Starting AI Model Training Pipeline (v5.7 - Resumable Control) ==========")
    logger.info("====================================================================================")

    try:
        # --- 단계 1: 훈련 환경 설정 ---
        logger.info("[Step 1] Setting up training environment...")
        GpuOptimizer.setup_gpu()

        # --- 단계 2: 핵심 구성 요소 초기화 및 모델 로드 확인 ---
        logger.info("[Step 2] Initializing components and checking for existing model...")
        
        predictor = DeepLearningPredictor(model_dir=settings.MODEL_DIR)
        model_path = settings.model_path
        #os.path.join(settings.MODEL_DIR, 'best_hybrid_model2.keras')

        # [핵심 수정 v5.7] settings.py 플래그에 따라 이어하기 및 옵티마이저 교체 로직 제어
        if settings.RESUME_TRAINING and os.path.exists(model_path):
            try:
                logger.warning(f"Found existing model at '{model_path}'. RESUME_TRAINING is True.")
                logger.warning("Attempting to load model to resume training...")
                predictor.load_model_and_artifacts()
                logger.info("Model loaded successfully.")

                # [핵심 수정 v5.7] 옵티마이저 교체 여부를 명시적으로 확인
                if settings.REPLACE_OPTIMIZER_ON_RESUME:
                    logger.warning("REPLACE_OPTIMIZER_ON_RESUME is True. Replacing the optimizer...")
                    new_hyperparams = settings.TRAIN_HYPERPARAMETERS
                    predictor.recompile_loaded_model(new_hyperparams)
                    logger.info("Optimizer has been replaced successfully.")
                else:
                    logger.info("REPLACE_OPTIMIZER_ON_RESUME is False. Using the optimizer state from the loaded model.")

            except Exception as e:
                logger.error(f"Failed to load existing model, starting a fresh training. Error: {e}", exc_info=True)
                # 실패 시 predictor의 모델을 None으로 초기화하여 새 모델을 만들도록 보장
                predictor.model = None
        else:
            if not settings.RESUME_TRAINING:
                logger.info("RESUME_TRAINING is False. Starting a fresh training session.")
            else:
                logger.info(f"No existing model found at '{model_path}'. Starting a fresh training session.")
            
            if not os.path.exists(settings.MODEL_DIR):
                os.makedirs(settings.MODEL_DIR)
                logger.info(f"Model directory created at '{settings.MODEL_DIR}'.")

        trainer = ModelTrainer(
            predictor=predictor,
            data_dir=settings.DATA_DIR,
            model_dir=settings.MODEL_DIR
        )
        logger.info("All components initialized successfully.")

        # --- 단계 3: 하이퍼파라미터 로드 및 훈련 실행 ---
        hyperparameters = settings.TRAIN_HYPERPARAMETERS
        logger.info(f"[Step 3] Loading hyperparameters and starting training...")
        logger.info(f"Using pre-processed data from '{os.path.join(settings.DATA_DIR, 'preprocessed')}'")
        logger.info(f"Epochs: {hyperparameters.get('epochs')}, Batch Size: {hyperparameters.get('batch_size')}")
        
        # ModelTrainer는 predictor 내부에 모델이 로드되었는지 확인하고
        # 그에 맞춰 새 모델을 만들거나, 있는 모델을 그대로 사용합니다.
        trainer.run_training_pipeline(hyperparameters=hyperparameters)

        logger.info("===============================================================")
        logger.info("===== AI Model Training Pipeline Completed Successfully! =====")
        logger.info(f"Trained model and artifacts are saved in: {settings.MODEL_DIR}")
        logger.info("===============================================================")

    except FileNotFoundError as e:
        logger.error(f"**************************** TRAINING FAILED ****************************")
        logger.error(f"[CRITICAL ERROR] Preprocessed data not found. Please run 'preprocess_and_save_features.py' first.")
        logger.error(f"Details: {e}")
        preprocessed_dir = os.path.join(settings.DATA_DIR, 'preprocessed')
        logger.error(f"Looked for .hdf5 files in directory: {preprocessed_dir}")
        logger.error(f"*************************************************************************")
    except Exception as e:
        logger.critical(f"**************************** TRAINING FAILED ****************************")
        logger.critical(f"[CRITICAL ERROR] An unexpected error occurred during the training process.", exc_info=True)
        logger.critical(f"*************************************************************************")

if __name__ == '__main__':
    main()
