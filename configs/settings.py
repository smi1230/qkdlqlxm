import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수(API 키 등 민감 정보)를 로드합니다.
# 이를 통해 코드에 직접 키를 하드코딩하는 것을 방지합니다.
load_dotenv()

# ==============================================================================
# [ 1. 프로젝트 경로 설정 ]
# 이 섹션에서는 프로젝트의 주요 디렉토리 경로를 절대 경로로 정의하여,
# 어떤 환경에서 스크립트를 실행하더라도 파일 경로가 일관되게 유지되도록 합니다.
# ==============================================================================
# 현재 파일(settings.py)의 상위 폴더의 상위 폴더, 즉 프로젝트의 최상위 루트 디렉토리입니다.
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 로그 파일이 저장될 디렉토리 경로입니다.
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, 'logs')
# 훈련 데이터(CSV) 및 전처리된 데이터가 저장될 디렉토리 경로입니다.
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')
# 훈련된 AI 모델 파일(.keras)과 관련 아티팩트(feature_names.joblib)가 저장될 디렉토리 경로입니다.
MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
# 실제 사용할 훈련된 모델 파일의 전체 경로입니다.
model_path = os.path.join(MODEL_DIR, 'best_tbm_model.keras')
# 모델이 학습한 피쳐(특성)의 이름 목록 파일 경로입니다. 예측 시 입력 데이터의 순서를 맞추기 위해 필수적입니다.
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.joblib')

# ==============================================================================
# [ 2. API 및 운영 모드 설정 ]
# 시스템이 어떤 거래 환경에서 동작할지 결정하고, 해당 환경에 맞는 API 키를 설정합니다.
# ==============================================================================
# 모드 선택: "m"(MAINNET, 실거래), "t"(TESTNET, 테스트넷), "d"(DEMO_MAINNET, 데모) 중 하나를 선택합니다.
TRADING_MODE = "d" 

# 위에서 설정한 TRADING_MODE에 따라 적절한 API 키와 시크릿을 .env 파일에서 불러옵니다.
if TRADING_MODE == "m":
    API_KEY = os.getenv("BYBIT_MAINNET_API_KEY")
    API_SECRET = os.getenv("BYBIT_MAINNET_API_SECRET")
    TESTNET = False # 실거래 모드
elif TRADING_MODE == "t":
    API_KEY = os.getenv("BYBIT_TESTNET_API_KEY")
    API_SECRET = os.getenv("BYBIT_TESTNET_API_SECRET")
    TESTNET = True # 테스트넷 모드
elif TRADING_MODE == "d":
    API_KEY = os.getenv("BYBIT_DEMO_MAINNET_API_KEY")
    API_SECRET = os.getenv("BYBIT_DEMO_MAINNET_API_SECRET")
    TESTNET = False # 데모 모드는 실거래 서버 기반이므로 False
else:
    raise ValueError(f"Invalid TRADING_MODE: {TRADING_MODE}")

# API 키가 올바르게 설정되었는지 확인합니다.
if not API_KEY or not API_SECRET or "YOUR" in API_KEY:
    raise ValueError("[CRITICAL ERROR] API Keys are not set correctly in .env file.")

# ==============================================================================
# [ 3. 시스템 아키텍처 및 성능 설정 ]
# AI 모델 훈련 및 실행 시 하드웨어(특히 GPU) 자원을 어떻게 사용할지 설정합니다.
# ==============================================================================
GPU_MEMORY_LIMIT = 0
MIXED_PRECISION = True

# ==============================================================================
# [ 4. 데이터 수집 및 처리 설정 ]
# 거래 데이터(K-line)를 어떤 주기로, 얼마나 많이 수집할지 정의합니다.
# ==============================================================================
INTERVAL = '5'                          # K-line(캔들) 데이터의 시간 주기 (예: '5'는 5분봉)
MIN_KLINE_DATA_SIZE = 500               # 실시간 거래 시 피쳐 계산을 위해 필요한 최소 캔들 수
API_KLINE_LIMIT = 1000                  # Bybit API에서 한 번에 요청할 수 있는 최대 캔들 수
TRAINING_DATA_START_DATE = "2025-06-15" # bulk_downloader.py 실행 시 데이터 수집을 시작할 날짜 ("YYYY-MM-DD" 형식)

# ==============================================================================
# [ 5. 피쳐 엔지니어링 설정 (5분봉 최적화 버전) ]
# AI 모델이 학습할 데이터를 가공(피쳐 생성)할 때 사용되는 다양한 기술적 지표의 파라미터입니다.
# ==============================================================================
TARGET_PERIOD = 6 # 예측할 미래 시점 (1은 다음 캔들을 의미)
# 최대 포지션 보유 기간 (캔들 수)로 변경됨 아래 TBM_MAX_HOLD_PERIODS 에 할당됨. 
# tbm 최초 구현 시에는 12, 즉 1시간의 값을 대상으로 했으나 15분과 1시간 사이의 값이 적절한 것으로 예상됨(폭락과 폭등은 보통 그 사이에 가장 많음)
# 30분으로 해야지.

# --- 5-1. 이동평균 (Moving Averages) ---
EMA_SHORT_PERIOD = 9
EMA_MID_PERIOD = 21
EMA_RIBBON_PERIODS = [5, 8, 13]
SMA_LONG_PERIOD = 50
SMA_VERY_LONG_PERIOD = 200
MA_PERIODS_FOR_PRICE_POS = [20, 50, 200]

# --- 5-2. 오실레이터 (Oscillators) ---
RSI_PERIODS = [7, 14]
LEGACY_RSI_PERIODS = [21, 30]
STOCH_K_FAST = 5
STOCH_K_SLOW = 14
STOCH_D = 3
STOCH_SMOOTH = 3
MFI_PERIOD = 14

# --- 5-3. 변동성 및 모멘텀 (Volatility & Momentum) ---
BB_PERIOD = 20
BB_STD_DEV = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

# --- 5-4. 논문 기반 고급 피쳐 (Market Microstructure & Volume) ---
VWAP_PERIODS = [20, 50]
OFI_PERIODS = [10, 20]
AMIHUD_PERIODS = [10, 20]
VRSI_PERIODS = [7, 14, 21]
VOLUME_ROC_PERIODS = [5, 10, 20]
VOLUME_MA_SHORT_PERIOD = 5
VOLUME_MA_LONG_PERIOD = 20
VOLUME_PERCENTILE_WINDOW = 50

# ==============================================================================
# [ 5-X. 신규: 삼중 장벽 기법(TBM) 타겟 설정 ]
# ==============================================================================
TBM_ATR_PERIOD = 12                 # 동적 변동성 계산을 위한 ATR 주기 (본래 20이었으나 1시간을 기준으로 하려 12로 설정)
TBM_PROFIT_TAKE_MULT = 2.0          # 이익 실현 장벽 (ATR의 2.0배)
TBM_STOP_LOSS_MULT = 1.0            # 손절 장벽 (ATR의 1.0배)
TBM_MAX_HOLD_PERIODS = TARGET_PERIOD # 최대 포지션 보유 기간 (캔들 수)

# ==============================================================================
# [ 6. AI 모델 아키텍처 설정 (CNN-LSTM-Transformer) ]
# AI 모델의 내부 구조(레이어, 유닛 수 등)를 정의합니다.
# ==============================================================================
HYBRID_MODEL_CONFIG = {
    'sequence_length': 12,
    'cnn_filters': [64, 128],
    'lstm_units': [128, 64],
    'transformer_heads': 8,
    'key_dim': 128,
    'transformer_layers': 4,
    'fusion_units': 256,
    'dropout_rate': 0.3
}

# 이제 모델은 단 하나의 타겟('trade_outcome')에만 집중합니다.
ACTIVE_OUTPUTS = ['trade_outcome']

# ==============================================================================
# [ 7. AI 모델 훈련 설정 ]
# 모델을 어떻게 훈련시킬지에 대한 상세 파라미터를 정의합니다.
# ==============================================================================
RESUME_TRAINING = True
REPLACE_OPTIMIZER_ON_RESUME = True

TRAIN_HYPERPARAMETERS = {
    "epochs": 3600,
    "batch_size": 50000,
    "early_stopping_patience": 150,
    "learning_rate_scheduler": {
        "initial_learning_rate": 0.001,
        "first_decay_steps": 350,
        "t_mul": 1.01,
        "m_mul": 0.998,
        "alpha": 1e-7
    }
}

# ==============================================================================
# [ 8. 거래 전략 및 기회 탐색 설정 ]
# AI 예측 결과를 어떻게 해석하고 거래에 활용할지 정의합니다.
# ==============================================================================
STRATEGY_MODE = 'AI_HYBRID_REBALANCE'
TOP_N_SYMBOLS = 10
POSITION_CHANGE_THRESHOLD = 0.20
FORCED_REBALANCE_CYCLES = int(60 / int(INTERVAL))

DL_CONFIDENCE_WEIGHT = 0.4
COMPOSITE_DL_WEIGHT = 0.70
COMPOSITE_TECHNICAL_WEIGHT = 0.30
MIN_DL_CONFIDENCE_THRESHOLD = 0.6
MIN_COMPOSITE_SCORE_THRESHOLD = 0.5
SIGNAL_ALIGNMENT_BONUS = 1.2

# ==============================================================================
# [ 9. 리스크 및 자금 관리 설정 ]
# 거래 시 자본을 어떻게 배분하고 리스크를 관리할지 정의합니다.
# ==============================================================================
MAX_SINGLE_ALLOCATION_RATIO = 0.15
MIN_SINGLE_ALLOCATION_RATIO = 0.05
VOLATILITY_ADJUSTMENT_FACTOR = 0.3
SL_ATR_MULTIPLIER = 1.5
TP1_REWARD_RATIO = 1.5
DAILY_LOSS_LIMIT_PERCENT = -3.0
POSITION_MAX_HOLD_MINUTES = 60 * 4

# ==============================================================================
# [ 10. 주문 실행 및 안정성 설정 ]
# 주문 방식과 API 통신 안정성에 대한 설정을 정의합니다.
# ==============================================================================
PREFER_MAKER_ORDERS = True
MAKER_ORDER_TIMEOUT = 5
MAX_SLIPPAGE_PERCENT = 0.0005

API_REQUEST_RETRY_COUNT = 3
API_REQUEST_RETRY_DELAY = 3

CB_FAILURE_THRESHOLD = 5
CB_RECOVERY_TIMEOUT = 60

DYNAMIC_ORDER_STRATEGY = True
DYNAMIC_ORDER_SPREAD_THRESHOLD_PCT = 0.04
TAKER_FEE_RATE = 0.055 / 100
MAKER_FEE_RATE = 0.02 / 100

# ==============================================================================
# [ 11. 필터링 및 블랙리스트 설정 ]
# 거래 대상에서 제외할 종목 등을 정의합니다.
# ==============================================================================
FILTER_TRIM_COUNT = 10
SYMBOL_BLACKLIST = ["BTCUSDT-11JUL25", "BTCUSDT-18JUL25","BTCUSDT-26JUN26", 'DMCUSDT', 'ETHUSDT-11JUL25', 'ETHUSDT-18JUL25', 'ETHUSDT-26JUN26', 'FRAGUSDT', 'HUSDT', 'ICNTUSDT','MUSDT','NEWTUSDT','SAHARAUSDT','SOLUSDT-11JUL25','SOLUSDT-18JUL25','SOLUSDT-29AUG25','SOSOUSDT', 'SPKUSDT']

USE_INTELLIGENT_LOSS_CUT = True
SET_TPSL_ON_ENTRY = True

USE_RECENT_ACTIVITY_FILTER = True
RECENT_ACTIVITY_TIMEFRAME_MINUTES = 60
RECENT_ACTIVITY_MIN_TURNOVER_USD = 5000

# --- 초기화 완료 로그 ---
print("✅ All settings loaded successfully. Running in TESTNET mode." if TESTNET else
      "--- !!! WARNING: Live Trading Mode (MAINNET) !!! ---")
