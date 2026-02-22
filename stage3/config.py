
TARGET_COL = "PAY_AMT_SUM"
ID_COL = "PLAYERID"
PRED_PAYER_COL = "pred_is_payer" 
PRED_WHALE_COL = "pred_is_high_payer"
STAGE1_PROBA_COL = "stage1_proba"
STAGE2_PROBA_COL = "stage2_proba"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
CAT_TASK_PARAMS = {"task_type": "GPU"} if DEVICE == "cuda" else {}

SEED = 2025 
OPTUNA_SEED = 2025 
DEFAULT_SEEDS_STR = "2024..2025"
N_JOBS = 12 

PROJECT_ROOT = Path(__file__).resolve().parent 
DATA_DIR = PROJECT_ROOT.parent / "data"
DEFAULT_INPUT_DATA_PATH = PROJECT_ROOT / "data" / "stage2_data.parquet"
DEF_OUTPUT_DIR = PROJECT_ROOT / "stage3_results"

DEFAULT_TRIALS = 4 
DEFAULT_TEST_SIZE = 0.2 
BASE_SPLIT_SEED = 2021

USE_LOG_TRANSFORM = True
LOG_EPSILON = 1e-6
SMAPE_EPSILON = 0.1

USE_STAGE1_FEATURES = False
USE_STAGE2_FEATURES = False
NO_CATBOOST = False 
NO_LGBM = False
NO_TABPFN = False 

TABPFN_DEVICE = "auto" 
TABPFN_CONFIGS = 32 

LGBM_LOSS = "rmse"
CAT_LOSS = "RMSE" 

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

ENSEMBLE_MODE = "mean"
SKIP_IF_EXISTS = False
