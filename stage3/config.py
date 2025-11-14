# config_stage3.py

"""
Stage 3 (Final LTV Regression) Global Configuration.
"""
import torch
from pathlib import Path
from typing import List

# =====================================================================================
# ---- GLOBAL CONFIGURATION (Defaults) ---
# =====================================================================================
# --- Data/Column Names ---
TARGET_COL = "PAY_AMT_SUM" # LTV 예측의 원본 목표 변수 (실제 결제 금액)
ID_COL = "PLAYERID" # 사용자 ID 컬럼
PRED_PAYER_COL = "pred_is_payer" # Stage 1 예측 결과 (과금자 여부)
PRED_WHALE_COL = "pred_is_high_payer" # Stage 2 예측 결과 (Whale 여부)
STAGE1_PROBA_COL = "stage1_proba" # Stage 1 예측 확률
STAGE2_PROBA_COL = "stage2_proba" # Stage 2 예측 확률

# --- Hardware/Environment ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # PyTorch 및 CatBoost의 실행 장치 설정 ("cuda", "cpu")
CAT_TASK_PARAMS = {"task_type": "GPU"} if DEVICE == "cuda" else {} # CatBoost 학습에 사용되는 추가 파라미터 딕셔너리 (GPU 사용 여부 결정)

# --- Global Placeholders for the Run ---
SEED = 2025 # 단일 시드 훈련에 사용되는 기본 난수 시드 (int)
OPTUNA_SEED = 2025 # Optuna 하이퍼파라미터 튜닝에 사용되는 난수 시드 (int)
DEFAULT_SEEDS_STR = "2024..2025" # 멀티 시드 훈련에 사용되는 기본 시드 범위 (예: "2024,2025" 또는 "2024..2025")
N_JOBS = 12 # 병렬 작업자 수 (LightGBM 등에서 n_jobs로 사용됨)

# --- Data/Path Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent 
DATA_DIR = PROJECT_ROOT.parent / "data"
# Stage 2의 최종 결과물(전체 데이터의 Stage 2 예측 결과 포함)을 Stage 3의 입력으로 사용
DEFAULT_INPUT_DATA_PATH = PROJECT_ROOT / "data" / "stage2_data.parquet"
DEF_OUTPUT_DIR = PROJECT_ROOT / "stage3_results"

# --- Modeling/Metric Constants ---
DEFAULT_TRIALS = 4 # Optuna 튜닝 시 시도할 횟수 (int)
DEFAULT_TEST_SIZE = 0.2 # 데이터 분할 시 검증(Validation) 세트의 비율 (float)
BASE_SPLIT_SEED = 2021 # Imputer 생성 시 사용되는 고정 난수 시드 (int)

# Stage 3 회귀 목표 변수 (로그 변환 사용 여부)
USE_LOG_TRANSFORM = True # 목표 변수 PAY_AMT_SUM에 로그 변환(np.log1p)을 적용할지 여부 (bool)
LOG_EPSILON = 1e-6 # 예측값 역변환 시 np.expm1(p) + log_epsilon
SMAPE_EPSILON = 0.1 # sMAPE 계산 시 분모의 안정화를 위한 엡실론 값 추가

# --- Model Specific Flags ---
USE_STAGE1_FEATURES = False # Stage 1 예측 확률을 피처로 사용할지
USE_STAGE2_FEATURES = False # Stage 2 예측 확률을 피처로 사용할지
NO_CATBOOST = False # CatBoost 모델 학습을 건너뛸지 여부 (bool)
NO_LGBM = False # LightGBM 모델 학습을 건너뛸지 여부 (bool)
NO_TABPFN = False # TabPFN 모델 학습을 건너뛸지 여부 (bool)

# --- TabPFN Specific ---
TABPFN_DEVICE = "auto" # TabPFN 모델의 실행 장치 ("auto", "cpu", "cuda")
TABPFN_CONFIGS = 32 # TabPFN 모델의 앙상블 설정(N_ensemble_configurations) 수

# --- Regression Loss (회귀 손실 함수) ---
LGBM_LOSS = "rmse" # LightGBM에서 사용할 손실 함수 ("rmse", "mae", "huber" 등)
CAT_LOSS = "RMSE" # CatBoost에서 사용할 손실 함수 ("RMSE", "MAE", "Huber" 등)


# --- 튜닝 범위: Non-Whale Payer (LTV가 낮을 것으로 예상되는 그룹) ---
# 비교적 모델 복잡도를 낮게 설정하여 과적합 방지
TUNE_NON_WHALE = {
    "lgbm": {
        "n_estimators": (400, 700), "learning_rate": (0.01, 0.1), "num_leaves": (20, 60), 
        "max_depth": (-1, 7), "min_child_samples": (15, 150), 
        "reg_alpha": (0.0, 5.0), "reg_lambda": (0.0, 10.0),
    },
    "cat": {
        "iterations": (500, 700), "depth": (4, 7), "learning_rate": (0.02, 0.15),
        "l2_leaf_reg": (1.0, 8.0), "bagging_temperature": (0.0, 2.0),
    }
}

# --- 튜닝 범위: Whale Payer (LTV가 높을 것으로 예상되는 그룹) ---
# 모델 복잡도를 높여 높은 분산을 학습할 수 있도록 설정
TUNE_WHALE = {
    "lgbm": {
        "n_estimators": (600, 1000), "learning_rate": (0.01, 0.15), "num_leaves": (50, 127), 
        "max_depth": (-1, 10), "min_child_samples": (5, 50), 
        "reg_alpha": (0.0, 8.0), "reg_lambda": (0.0, 15.0),
    },
    "cat": {
        "iterations": (700, 1000), "depth": (6, 10), "learning_rate": (0.01, 0.1),
        "l2_leaf_reg": (2.0, 12.0), "bagging_temperature": (0.0, 3.0),
    }
}


# --- Ensemble Configuration (Multi-seed) ---
ENSEMBLE_MODE = "mean" # 여러 시드의 예측 확률을 결합하는 방식 ("mean", "weighted_by_mae" 등)
SKIP_IF_EXISTS = False # per-seed 예측 결과 파일이 이미 존재할 경우, 해당 시드의 훈련/예측을 건너뛸지 여부 (bool)