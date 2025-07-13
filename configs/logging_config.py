# /configs/logging_config.py

import logging
import logging.config
import os
from logging.handlers import RotatingFileHandler

# ==============================================================================
# [ 1. 의존성 주입: 경로 설정 안정화 ]
# ==============================================================================
# settings.py에 완벽하게 정의된 절대 경로 LOG_DIR 변수를 직접 임포트합니다.
# 이로써 이 파일은 더 이상 경로 계산에 대해 고민할 필요가 없으며,
# 어떤 환경에서 실행되더라도 일관된 로그 파일 경로를 보장합니다.
# ------------------------------------------------------------------------------
try:
    from .settings import LOG_DIR
except ImportError:
    # 이 파일이 단독으로 테스트되거나, settings.py에 문제가 있을 경우를 대비한 최종 예외 처리.
    # 정상적인 프로젝트 구조에서는 발생하지 않아야 합니다.
    print("[CRITICAL WARNING] Could not import LOG_DIR from settings.py. Logging will be unreliable.")
    print("Using a temporary default path './logs'.")
    # 임시로 현재 작업 디렉토리 기준의 상대 경로를 사용합니다.
    LOG_DIR = os.path.join(os.getcwd(), 'logs')

# ==============================================================================
# [ 2. 로깅 시스템 설정 함수 ]
# ==============================================================================
def setup_logging():
    """
    프로젝트 전반에 사용될 표준 로깅 설정을 구성합니다.

    - 콘솔 핸들러: INFO 레벨 이상만 출력하여 실시간 모니터링에 용이하게 합니다.
    - 파일 핸들러: DEBUG 레벨 이상의 모든 로그를 파일에 기록하여 상세한 분석 및 디버깅을 지원합니다.
    """
    # 1. 로그 디렉토리 존재 여부 확인 및 생성
    # LOG_DIR은 이제 settings.py에서 보장된 절대 경로입니다.
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
            print(f"Log directory created at: {LOG_DIR}")
        except OSError as e:
            # 동시에 여러 프로세스가 디렉토리를 만들려고 할 때 발생할 수 있는 경쟁 상태(race condition) 방지
            if not os.path.isdir(LOG_DIR):
                print(f"Error creating log directory: {e}")
                # 로깅 설정이 실패하면 시스템을 계속 진행하는 것이 위험할 수 있으므로, 예외를 발생시킵니다.
                raise

    # 2. 로그 파일의 전체 경로 생성
    log_file_path = os.path.join(LOG_DIR, "trading_system.log")

    # 3. 로깅 설정 딕셔너리
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False, # 다른 라이브러리(예: requests, tensorflow)의 로거를 비활성화하지 않음
        'formatters': {
            # 콘솔 출력용 포맷: 시간 - 로거이름 - 로그레벨 - 메시지
            'console_format': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            # 파일 기록용 상세 포맷: 시간 - 로거이름:파일:줄번호 - 로그레벨 - 메시지 (디버깅에 용이)
            'file_format': {
                'format': '%(asctime)s - %(name)s:%(filename)s:%(lineno)d - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            # 콘솔(화면)에 로그를 출력하는 핸들러
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',  # [핵심] INFO, WARNING, ERROR, CRITICAL 레벨의 로그만 화면에 표시
                'formatter': 'console_format',
                'stream': 'ext://sys.stdout', # 표준 출력 사용
            },
            # 파일에 로그를 기록하는 핸들러
            'file_rotating': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG', # [핵심] DEBUG 레벨 이상의 모든 로그를 파일에 기록 (가장 상세)
                'formatter': 'file_format',
                'filename': log_file_path,
                'maxBytes': 1024 * 1024 * 10,  # 로그 파일 최대 크기 (10 MB)
                'backupCount': 5,            # 최대 5개의 백업 파일 유지 (trading_system.log.1, ... .5)
                'encoding': 'utf-8',
            },
        },
        # 루트 로거 설정: 모든 로거의 기본이 되는 설정
        'root': {
            'level': 'DEBUG', # 처리할 로그의 최소 레벨 (DEBUG 이상 모두 처리)
            'handlers': ['console', 'file_rotating'] # 로그를 'console'과 'file' 핸들러 모두에게 전달
        }
    }

    # 4. 위에서 정의한 딕셔너리 설정을 로깅 시스템에 적용
    logging.config.dictConfig(LOGGING_CONFIG)

    # 5. 설정 완료 후, 성공 메시지를 로깅하여 기록합니다.
    # 이 메시지는 setup_logging()이 호출되는 즉시 출력되어, 로깅 시스템이 정상 작동함을 증명합니다.
    logging.info(f"Logging configured successfully. Log files will be saved to '{log_file_path}'")

# ==============================================================================
# [ 3. 독립 실행 테스트 ]
# ==============================================================================
# 이 파일(logging_config.py)이 직접 실행될 때 로깅 설정을 테스트하는 코드
if __name__ == '__main__':
    print("--- Testing Logging Configuration (v5.0) ---")
    setup_logging()

    # 다양한 레벨의 테스트 로그 생성
    # __name__ 대신 'TestLogger'라는 명시적인 이름을 사용하여 출력을 더 명확하게 함
    test_logger = logging.getLogger("TestLogger")

    print("\n[Step 1] Generating test logs at various levels...")
    test_logger.debug("This is a DEBUG message. It should ONLY appear in the log file.")
    test_logger.info("This is an INFO message. It should appear on the CONSOLE and in the log file.")
    test_logger.warning("This is a WARNING message. Also appears on both.")
    test_logger.error("This is an ERROR message.")
    test_logger.critical("This is a CRITICAL message, the highest level.")

    try:
        1 / 0
    except ZeroDivisionError as e:
        # exc_info=True는 예외 traceback 정보를 로그에 함께 기록하게 합니다. 디버깅에 매우 유용합니다.
        test_logger.error(f"An exception occurred: {e}", exc_info=True)


    print("\n[Step 2] Logging test finished.")
    print("----------------------------------------")
    print("\n[Verification Steps]")
    print("1. CONSOLE: Check if only INFO, WARNING, ERROR, CRITICAL messages are displayed above.")
    print(f"2. LOG FILE: Open the file '{os.path.join(LOG_DIR, 'trading_system.log')}' and verify:")
    print("   - All messages (including DEBUG and the exception traceback) are recorded.")
    print("   - The format is more detailed (includes filename and line number).")