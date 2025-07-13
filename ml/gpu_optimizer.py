# /ml/gpu_optimizer.py

import logging
import tensorflow as tf

# 시스템의 중앙 설정 파일을 임포트합니다.
try:
    from configs import settings
except ImportError:
    print("[CRITICAL WARNING] Could not import settings.py. GpuOptimizer will use default values.")
    # settings.py 임포트 실패 시 사용할 기본값 (안전 모드)
    class MockSettings:
        GPU_MEMORY_LIMIT = 4096
        MIXED_PRECISION = False
    settings = MockSettings()

logger = logging.getLogger(__name__)

class GpuOptimizer:
    """
    [ v5.0 간단 명확 버전 ]
    TensorFlow GPU 사용을 최적화하기 위한 유틸리티 클래스.
    복잡성을 줄이고 안정적인 GPU 환경 설정에만 집중합니다.
    """

    @staticmethod
    def setup_gpu():
        """
        settings.py의 설정을 기반으로 TensorFlow의 GPU 환경을 안전하게 구성합니다.

        1. 물리적 GPU 장치를 찾습니다.
        2. [중요] '메모리 동적 증가'를 먼저 활성화하여, set_virtual_device_configuration과의 충돌을 방지합니다.
        3. 설정된 메모리 상한(GPU_MEMORY_LIMIT)에 따라 가상 디바이스를 설정합니다.
        4. 설정된 혼합 정밀도(MIXED_PRECISION) 정책을 적용합니다.
        """
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logger.warning("No GPU found. TensorFlow will run on CPU.")
                return

            # [핵심 수정] 가상 장치를 설정하기 '전에' 반드시 메모리 동적 증가를 활성화해야 합니다.
            # 이것이 "Cannot set memory growth on device when virtual devices configured" 오류를 방지합니다.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 설정 파일에 지정된 메모리 상한이 0보다 클 경우, 가상 장치를 설정합니다.
            if settings.GPU_MEMORY_LIMIT > 0:
                logger.info(f"Setting virtual device memory limit to {settings.GPU_MEMORY_LIMIT}MB for {gpus[0].name}.")
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=settings.GPU_MEMORY_LIMIT)]
                )

            # 혼합 정밀도(Mixed Precision) 설정
            if settings.MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision ('mixed_float16') policy enabled globally.")

            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"✅ GPU setup successful. Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")

        except RuntimeError as e:
            # "Virtual devices cannot be modified after being initialized"와 같은 오류 처리
            logger.error(f"GPU setup failed, possibly because it was already initialized. Error: {e}", exc_info=False)
        except Exception as e:
            logger.error(f"An unexpected error occurred during GPU setup: {e}", exc_info=True)


# ==============================================================================
# [ 독립 실행 테스트 ]
# ==============================================================================
# 이 파일을 직접 실행 (`python ml/gpu_optimizer.py`)하여 GPU 설정이 올바르게 동작하는지 테스트합니다.
if __name__ == '__main__':
    import os
    import sys
    PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(PROJECT_ROOT_DIR)
    from configs.logging_config import setup_logging
    setup_logging()

    print("\n--- Testing GPU Optimizer (Simple & Clear Version) ---")
    GpuOptimizer.setup_gpu()

    print("\n--- Verifying TensorFlow GPU Access ---")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # 간단한 텐서 연산을 통해 GPU가 실제로 사용되는지 확인
            print("Performing a test calculation on GPU...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            print("Test calculation successful. GPU is accessible.")
        else:
            print("No physical GPU detected by TensorFlow.")
    except Exception as e:
        print(f"An error occurred during GPU verification test: {e}")