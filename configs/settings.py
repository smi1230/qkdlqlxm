import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (API 키 등 민감 정보 관리)
load_dotenv()

# ==============================================================================
# [ 1. 프로젝트 경로 설정 ]
# ==============================================================================
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
model_path = os.path.join(MODEL_DIR, 'best_hybrid_model_remodeling_newdata_01.keras')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.joblib')

# ==============================================================================
# [ 2. API 및 운영 모드 설정 ]
# ==============================================================================
# 모드 선택: "MAINNET", "TESTNET", "DEMO_MAINNET" 중 하나 선택
# d,t,m
TRADING_MODE = "d" 

# 각 모드에 맞는 API 키와 시크릿 설정
if TRADING_MODE == "m":
    API_KEY = os.getenv("BYBIT_MAINNET_API_KEY")
    API_SECRET = os.getenv("BYBIT_MAINNET_API_SECRET")
    TESTNET = False
elif TRADING_MODE == "t":
    API_KEY = os.getenv("BYBIT_TESTNET_API_KEY")
    API_SECRET = os.getenv("BYBIT_TESTNET_API_SECRET")
    TESTNET = True
elif TRADING_MODE == "d":
    API_KEY = os.getenv("BYBIT_DEMO_MAINNET_API_KEY") # .env 파일에 추가 필요
    API_SECRET = os.getenv("BYBIT_DEMO_MAINNET_API_SECRET") # .env 파일에 추가 필요
    TESTNET = False # 데모 모드는 메인넷 기반이므로 False
else:
    raise ValueError(f"Invalid TRADING_MODE: {TRADING_MODE}")


if not API_KEY or not API_SECRET or "YOUR" in API_KEY:
    raise ValueError("[CRITICAL ERROR] API Keys are not set correctly in .env file.")

# ==============================================================================
# [ 3. 시스템 아키텍처 및 성능 설정 ]
# ==============================================================================
GPU_MEMORY_LIMIT = 0  # 10GB #숫자 지정하면 그만큼 그냥 먹음. 0으로 해두면 필요한 만큼만 먹음.
MIXED_PRECISION = True

# ==============================================================================
# [ 4. 데이터 수집 및 처리 설정 ]
# ==============================================================================
INTERVAL = '5'                          # 5분봉
MIN_KLINE_DATA_SIZE = 500               # 피쳐 계산을 위한 최소 캔들 수
API_KLINE_LIMIT = 1000
TRAINING_DATA_START_DATE = "2025-06-15"

# ==============================================================================
# [ 5. 피쳐 엔지니어링 설정 (5분봉 최적화 버전) ]
# ==============================================================================
TARGET_PERIOD = 1

# --- 5-1. 이동평균 (Moving Averages) ---
EMA_SHORT_PERIOD = 9
EMA_MID_PERIOD = 21
EMA_RIBBON_PERIODS = [5, 8, 13]
SMA_LONG_PERIOD = 50
SMA_VERY_LONG_PERIOD = 200
MA_PERIODS_FOR_PRICE_POS = [20, 50, 200]

# --- 5-2. 오실레이터 (Oscillators) ---
RSI_PERIODS = [7, 14]
LEGACY_RSI_PERIODS = [21, 30] # 기존 시스템과의 호환성 또는 추가 분석용
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
AMIHUD_PERIODS = [10, 20]             # [신규 추가] Amihud 비유동성 지표 주기
VRSI_PERIODS = [7, 14, 21]
VOLUME_ROC_PERIODS = [5, 10, 20]
VOLUME_MA_SHORT_PERIOD = 5            # [신규 추가] 거래량 확장/압축 지표용 단기 MA
VOLUME_MA_LONG_PERIOD = 20            # [신규 추가] 거래량 확장/압축 지표용 장기 MA
VOLUME_PERCENTILE_WINDOW = 50         # [신규 추가] 거래량 백분위 순위 계산 기간

# ==============================================================================
# [ 6. AI 모델 아키텍처 설정 (CNN-LSTM-Transformer) ]
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

# [수정] 활성화할 '학습 타겟'을 리스트로 정의. 'confidence'는 이제 학습 타겟이 아님.
# 'confidence'는 'price_direction'의 예측 결과(softmax)로부터 파생되는 값으로 자동 계산됨.
ACTIVE_OUTPUTS = ['price_direction']#, 'volatility']


# [수정] 멀티태스크 학습 시 각 손실(loss)에 대한 가중치. 'confidence' 항목 제거.
MULTITASK_LOSS_WEIGHTS = {
    'price_direction': 0.8,
    'volatility': 0.2,
    'volume': 0.1, # ACTIVE_OUTPUTS에 없으므로 현재는 사용되지 않음
}

# ==============================================================================
# [ 7. AI 모델 훈련 설정 ]
# ==============================================================================
# [ v5.7 수정 ] 훈련 이어하기 및 옵티마이저 교체 제어 플래그 추가
RESUME_TRAINING = True
REPLACE_OPTIMIZER_ON_RESUME = True

TRAIN_HYPERPARAMETERS = {
    "epochs": 3600,
    "batch_size": 50000,
    "early_stopping_patience": 150,
    "learning_rate_scheduler": {
        "initial_learning_rate": 0.001,
        # 배치 사이즈가 작아졌으므로, 1 에포크에 필요한 스텝 수가 늘어남.
        # 따라서 첫 주기가 끝나는 스텝 수도 비례하여 늘려주는 것이 논리적으로 타당함.
        # (예: 10 에포크에 해당하는 스텝 수로 설정)
        "first_decay_steps": 350, # [논리 수정] 사용자의 지적에 따라 다시 늘림
        "t_mul": 1.01,
        "m_mul": 0.998,
        "alpha": 1e-7
    }
}

# ==============================================================================
# [ 8. 거래 전략 및 기회 탐색 설정 ]
# ==============================================================================
STRATEGY_MODE = 'AI_HYBRID_REBALANCE'
TOP_N_SYMBOLS = 10
POSITION_CHANGE_THRESHOLD = 0.20
FORCED_REBALANCE_CYCLES = int(60 / int(INTERVAL))

DL_DIRECTION_WEIGHT = 0.6
DL_CONFIDENCE_WEIGHT = 0.4

COMPOSITE_DL_WEIGHT = 0.70
COMPOSITE_TECHNICAL_WEIGHT = 0.30
MIN_DL_CONFIDENCE_THRESHOLD = 0.6
MIN_COMPOSITE_SCORE_THRESHOLD = 0.5
SIGNAL_ALIGNMENT_BONUS = 1.2

# ==============================================================================
# [ 9. 리스크 및 자금 관리 설정 ]
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
# ==============================================================================
PREFER_MAKER_ORDERS = True
MAKER_ORDER_TIMEOUT = 5
MAX_SLIPPAGE_PERCENT = 0.0005

API_REQUEST_RETRY_COUNT = 3
API_REQUEST_RETRY_DELAY = 3

CB_FAILURE_THRESHOLD = 5
CB_RECOVERY_TIMEOUT = 60

# ✨ 동적 주문 전략 설정
DYNAMIC_ORDER_STRATEGY = True  # True: 스프레드 기반 동적 주문, False: 기존 지정가 우선 방식
# 스프레드가 이 비율(%)보다 작거나 같으면 시장가 주문, 크면 지정가 주문 시도
DYNAMIC_ORDER_SPREAD_THRESHOLD_PCT = 0.04 
# Bybit 선물 기본 수수료 (필요 시 실제 계정 수수료율로 수정)
TAKER_FEE_RATE = 0.055 / 100 # 0.055%
MAKER_FEE_RATE = 0.02 / 100 # 0.02%

# ==============================================================================
# [ 11. 필터링 및 블랙리스트 설정 ]
# ==============================================================================
FILTER_TRIM_COUNT = 10
SYMBOL_BLACKLIST = ["BTCUSDT-11JUL25", "BTCUSDT-18JUL25","BTCUSDT-26JUN26", 'DMCUSDT', 'ETHUSDT-11JUL25', 'ETHUSDT-18JUL25', 'ETHUSDT-26JUN26', 'FRAGUSDT', 'HUSDT', 'ICNTUSDT','MUSDT','NEWTUSDT','SAHARAUSDT','SOLUSDT-11JUL25','SOLUSDT-18JUL25','SOLUSDT-29AUG25','SOSOUSDT', 'SPKUSDT']
# [
#     "USDCUSDT", "USDTARS", "USDTBIDR", "USDTBRL", "USDTVND", "USDTUAH",
#     "BTCUSDC", "ETHUSDC", "SOLUSDC", "USDDUSDT", "WAVESUSDT", "LUNA2USDT"
# ]


# ✨ [핵심 추가] 지능형 손절 및 TP/SL 동시 주문 설정
USE_INTELLIGENT_LOSS_CUT = True  # AI 예측 기반의 동적 손절 기능 활성화 여부
SET_TPSL_ON_ENTRY = True         # 진입 시 TP/SL 동시 주문 기능 활성화 여부

# ✨ [핵심 추가] 최근 활동성 필터 설정
USE_RECENT_ACTIVITY_FILTER = True  # True로 설정 시, 아래 조건으로 추가 필터링
# 확인할 시간(분). 5분봉 기준 12개 캔들
RECENT_ACTIVITY_TIMEFRAME_MINUTES = 60
# 위 시간 동안 최소 거래대금 (USD)
RECENT_ACTIVITY_MIN_TURNOVER_USD = 5000


# --- 초기화 완료 로그 ---
print("✅ All settings loaded successfully. Running in TESTNET mode." if TESTNET else
      "--- !!! WARNING: Live Trading Mode (MAINNET) !!! ---")
