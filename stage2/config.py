"""
Stage 2 (Whale Classification) Global Configuration.
"""
# =====================================================================================
# ---- GLOBAL CONFIGURATION (Defaults)
# =====================================================================================
# --- Data/Column Names ---
TARGET_COL = "PAY_AMT_SUM" # LTV(Whale) 분류의 원본 목표 변수 (실제 결제 금액)
ID_COL = "PLAYERID" # 사용자 ID 컬럼
PRED_PAYER_COL = "pred_is_payer" # Stage 1에서 예측한 과금자 여부 컬럼 (1인 데이터만 Stage 2에 사용)
STAGE1_PROBA_COL = "stage1_proba" # Stage 1에서 예측한 과금 확률 컬럼 (Stage 2의 피처로 사용될 수 있음)

# --- Hardware/Environment ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # PyTorch 및 CatBoost의 실행 장치 설정
#DEVICE = "cpu"

CAT_TASK_PARAMS = {"task_type": "GPU"} if DEVICE == "cuda" else {} # CatBoost 학습에 사용되는 추가 파라미터 딕셔너리 (GPU 사용 여부 결정)
#CAT_TASK_PARAMS = {"task_type": "CPU"}


# --- Global Placeholders for the Run ---
SEED = 2025 # 단일 시드 훈련 (`run_stage2_train_core`)에 사용되는 기본 난수 시드
OPTUNA_SEED = 2025 # Optuna 하이퍼파라미터 튜닝에 사용되는 난수 시드
DEFAULT_SEEDS_STR = "2021,2022,2023,2024,2025,2026,2027,2028,2029,2030" # 멀티 시드 훈련 (`run_stage2_multi_core`)에 사용되는 기본 시드 범위 (parse_seeds 함수로 파싱됨)
N_JOBS = 12 # 병렬 작업자 수 (LightGBM 등에서 n_jobs로 사용됨)

# --- Data/Path Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent         # = ROOT/stage2
DATA_DIR = PROJECT_ROOT.parent / "Data"                # = ROOT/Data
DEFAULT_INPUT_DATA_PATH = DATA_DIR / "stage1_data.parquet"
# 결과물은 stage2 아래에 떨어지게 하고 싶다면:
DEF_OUTPUT_DIR = PROJECT_ROOT / "stage2_results"

# --- Modeling/Metric Constants ---
DEFAULT_TRIALS = 30 # Optuna를 사용한 하이퍼파라미터 튜닝 시 시도할 횟수
DEFAULT_TEST_SIZE = 0.2 # 데이터 분할 시 검증(Validation) 세트의 비율 (0.2 = 20%)
BASE_SPLIT_SEED = 2025 # Imputer 생성 시 사용되는 고정 난수 시드

# --- F-Beta Score Constants (컷오프 튜닝 및 최종 평가에 사용) ---
F_BETA = 2.0 # F-beta 점수의 'Beta' 값 (1.0이면 F1, 2.0이면 F2)
MIN_PREC_AT_CUT = 0.55 # 컷오프를 결정할 때 충족해야 하는 최소 정밀도(Precision) 제약 조건 (0.0은 제약 없음)
CUT_STEP = 0.01 # 컷오프(Threshold) 튜닝 시 임계값을 검색할 간격 (예: 0.50, 0.51, 0.52...)
DELTA_AROUND = 0.3 # 컷오프 튜닝 시 중심값(center) 주변으로 탐색할 범위 (예: center=0.5, delta=0.15면 [0.35, 0.65] 탐색)
REWEIGHT_BY_FBETA = True # F-Beta 최적화에 맞춰 클래스 가중치를 조정할지 여부

# --- Whale Definition Constants ---
DEFAULT_WHALE_PCT = 0.05 # LTV 상위 p%를 Whale로 정의 (0.05 = 상위 5%)
WHALE_Q = 1.0 - DEFAULT_WHALE_PCT # Whale Cutoff을 계산하기 위한 Quantile (0.95)
WHALE_CUT = 0.0 # Whale cutoff value (실제 LTV 금액, 데이터 로드 후 BASE_SPLIT_SEED 기준으로 계산되어 업데이트됨)

# --- Feature/Model Flags ---
USE_STAGE1_FEATURES = False # Stage 1 예측 확률(STAGE1_PROBA_COL)을 Stage 2 모델의 피처로 포함할지 여부
EXCLUDE_PROBA_FEATURES = True  # 'proba'가 포함된 컬럼 자동 제외 여부
NO_CATBOOST = False # CatBoost 모델 학습을 건너뛸지 여부
NO_LGBM = False # LightGBM 모델 학습을 건너뛸지 여부
NO_TABPFN = False # TabPFN 모델 학습을 건너뛸지 여부

# --- TabPFN Specific ---
# 선택지: "auto", "cpu", "cuda"
TABPFN_DEVICE = "auto" # TabPFN 모델의 실행 장치
TABPFN_CONFIGS = 32 # TabPFN 모델의 앙상블 설정(N_ensemble_configurations) 수

# --- Ensemble Configuration (Multi-seed) ---
ENSEMBLE_MODE = "weighted_by_ap" # 여러 시드의 예측 확률을 결합하는 방식
SKIP_IF_EXISTS = False # per-seed 예측 결과 파일이 이미 존재할 경우, 해당 시드의 훈련/예측을 건너뛸지 여부
