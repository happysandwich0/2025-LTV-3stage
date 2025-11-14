#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTV Prediction Pipeline - Stage 1 Final Training & Prediction (PATCHED)

- Phase 1: Train final Stage 1 models on train+val with fixed hyperparams.
  * Save artifacts: encoder, imputer, feat_order, cat_cols_idx, cutoffs, models (joblib + native)
- Phase 2: Predict payer for (train+val+test), apply fixed cutoffs + hard voting.
  * Save columns: original + pred_is_payer + per-model proba + stage1_proba(mean)

Key patches:
  1) Preserve feature order between train and predict via manifest feat_order.
  2) Save native model files (lgbm.txt, xgb.json, cat.cbm) for portability.
  3) Load manifest in prediction and reindex columns accordingly.
"""

import os
import sys
import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
import joblib
import logging
from datetime import datetime
import uuid
import traceback
from time import perf_counter

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb

import torch

from sklearn.metrics import precision_score  # (미사용 가능) 컷오프 고정이라 남겨둠

# =====================================================================================
# ---- 1. CONFIGURATION & PATHS
# =====================================================================================

SCRIPT_DIR = Path.cwd()  # __file__ 미보장 환경 대응
DATA_DIR = SCRIPT_DIR.parent / "Data"

ARTIFACTS_DIR = SCRIPT_DIR / "final_stage1_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR = SCRIPT_DIR / "final_predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

DATA_PATHS = {
    "train": str(DATA_DIR / "train_df_5days.parquet"),
    "val":   str(DATA_DIR / "val_df_5days.parquet"),
    "test":  str(DATA_DIR / "test_df_5days.parquet"),
    "train_robust": str(DATA_DIR / "train_df_5days_robust.parquet"),
    "val_robust":   str(DATA_DIR / "val_df_5days_robust.parquet"),
    "test_robust":  str(DATA_DIR / "test_df_5days_robust.parquet"),
}

TARGET_COL = "PAY_AMT_SUM"
ID_COL = "PLAYERID"
FINAL_SEED = 2021  # 대표 시드

# --- Data Loading ---
action_trash_list = ['길드_하우스 대여', '캐시 상점_아이템 삭제', '길드_가입 신청', '계정_로그인', '클래스_잠금',
                     '길드_설정 변경', '성장_레벨 다운', '성장_스킬 습득', '그로아_소환 확정 대기 변경', '아이템 컬렉션_추가',
                     '그로아_소환', '탈것_스킬 설정', '퀘스트_보상 미리보기 삭제', '캐시 상점_아이템 추가', '길드_생성', '제작_제작',
                     '클래스_소환 확정 대기 생성', '계정_로그아웃', '길드_적대 등록 취소', '길드_등급', '길드_동맹 신청 취소', '보스전_필드 보스',
                     '길드_동맹 신청', '탈것_추가', '탈것_소환 확정 대기 변경', '퀘스트_포기', '그로아_소환 확정 대기 생성', '성장_레벨 업',
                     '캐시 상점_월드 추가', '사망 불이익_경험치', '캐시 상점_캐시 상점에서 재화로 구매', '퀘스트_보상 미리보기', '캐릭터_생성',
                     '클래스_소환 확정 대기 변경', '길드_적대 등록', '던젼_충전', '스탯_설정', '기믹_등짐', '클래스_소환 확정 대기 삭제', '그로아_소환 확정 대기 삭제',
                     '성장_상태 변화 습득', '성장_죽음', '제작_추가', '퀘스트_의뢰 갱신', '길드_지원자 제거', '캐시 상점_캐릭터 추가', '길드_동맹 파기', '워프_갱신',
                     '워프_삭제', '클래스_추가', '길드_가입', '길드_동맹 신청 확인', '보스전_월드 보스', '퀘스트_완료', '길드_해체', '탈것_잠금', '캐시 상점_계정 추가',
                     '워프_생성', '워프_순간이동 사용', '성장_경험치 손실', '퀘스트_의뢰', '퀘스트_수락', '탈것_등록', '퀘스트_수행', '길드_경험치 획득', '그로아_잠금',
                     '캐시 상점_구매 나이 변경', '길드_동맹 신청 거절', '탈것_소환 확정 대기 생성', '클래스_변경', '탈것_소환 확정 대기 삭제', '길드_탈퇴', '사망 불이익_아이템',
                     '길드_출석', '그로아_추가']

action_list = ['PLAYERID','계정', '그로아', '기믹', '길드', '던젼', '보스전',
               '사망 불이익', '성장', '스탯', '아이템 컬렉션', '워프', '제작',
               '캐릭터', '캐시 상점', '퀘스트', '클래스', '탈것']

# --- Fixed Cutoffs ---
FINAL_CUTOFFS = {"cat": 0.2, "lgbm": 0.2, "xgb": 0.5}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT_TASK_PARAMS = {"task_type": "GPU"} if DEVICE == "cuda" else {}

# --- Fixed Hyperparameters ---
RUN_LR = 0.05
STAGE1_FIXED = {
    "lgbm": dict(
        objective="binary",
        n_estimators=358,
        learning_rate=RUN_LR,
        max_depth=8,
        min_child_samples=20,
        subsample=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbosity=-1,
    ),
    "xgb": dict(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=133,
        learning_rate=RUN_LR,
        max_depth=9,
        subsample=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        max_bin=256,
    ),
    "cat": dict(
        loss_function="Logloss",
        eval_metric="F1",
        iterations=347,
        depth=9,
        learning_rate=RUN_LR,
        verbose=0,
    )
}

# =====================================================================================
# ---- 2. UTILS
# =====================================================================================

class SectionTimer:
    def __init__(self, msg: str = ""):
        self.msg = msg
        self.t0 = None
    def __enter__(self):
        logging.info(f"⏱️  {self.msg} ...")
        self.t0 = perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = perf_counter() - self.t0 if self.t0 else 0.0
        logging.info(f"✅ {self.msg} done in {dt:.2f}s")

class OrdinalCategoryEncoder:
    def __init__(self):
        self.maps: Dict[str, Dict] = {}
        self.cols: List[str] = []
    def fit(self, df: pd.DataFrame, cat_cols: List[str]):
        self.cols = list(cat_cols)
        for c in self.cols:
            if c not in df.columns:
                logging.warning(f"[Encoder] Column '{c}' not found in df during fit. Skipping.")
                continue
            unique_vals = df[c].dropna().unique()
            try:
                sorted_cats = sorted(unique_vals, key=float)
            except (ValueError, TypeError):
                sorted_cats = sorted(map(str, unique_vals))
            self.maps[c] = {cat: i for i, cat in enumerate(sorted_cats)}
        logging.info(f"[Encoder] Fitted on {len(self.cols)} cols.")
        return self
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.cols:
            if c in out.columns:
                mapping = self.maps[c]
                s = out[c].astype(object)
                out[c] = s.map(mapping).fillna(-1).astype(np.int32)
        return out

def fit_imputer(train_df: pd.DataFrame, num_cols: List[str]):
    valid_num_cols = [c for c in num_cols if c in train_df.columns]
    if not valid_num_cols:
        logging.warning("[Imputer] No valid numeric columns found to fit.")
        return [], pd.Series(dtype=float)
    med = train_df[valid_num_cols].median(numeric_only=True).fillna(0)
    logging.info(f"[Imputer] Fitted median for {len(valid_num_cols)} numeric cols.")
    return valid_num_cols, med

def apply_imputer(df: pd.DataFrame, num_cols: List[str], med: pd.Series):
    df = df.copy()
    cols_to_impute = [col for col in num_cols if col in df.columns]
    med_filtered = med.reindex(cols_to_impute).fillna(0)
    df[cols_to_impute] = df[cols_to_impute].fillna(med_filtered)
    df = df.fillna(0)
    return df

def _sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.replace(r"\s+", "_", regex=True)
    out.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in out.columns]
    return out

def build_features(df: pd.DataFrame, target_col: str, drop_cols: List[str]):
    cols = [c for c in df.columns if c not in drop_cols and c != target_col]
    cat_cols = [c for c in cols if df[c].dtype == 'object' or str(df[c].dtype).startswith('category')]
    valid_cols = [c for c in cols if c in df.columns]
    valid_cat_cols = [c for c in cat_cols if c in df.columns]
    logging.info(f"Selected {len(valid_cols)} features. Identified {len(valid_cat_cols)} categorical.")
    return df[valid_cols].copy(), valid_cols, valid_cat_cols

def hard_vote(preds: Dict[str, np.ndarray], cutoffs: Dict[str, float]) -> np.ndarray:
    votes = []
    for k in ["cat", "lgbm", "xgb"]:
        if k in preds:
            t = cutoffs[k]
            p = preds[k]
            votes.append((p >= t).astype(int))
        else:
            logging.warning(f"[hard_vote] Missing model: {k}")
    if not votes:
        logging.error("No predictions for voting.")
        return np.array([])
    votes = np.column_stack(votes)
    threshold = int(math.ceil(votes.shape[1] / 2))
    return (votes.sum(axis=1) >= threshold).astype(int)

# --- Model Wrappers ---
class XGBCompat:
    def __init__(self, **params):
        self.params = params.copy()
        self.booster_ = None
        self.best_ntree_limit_ = None
        self._num_boost_round = int(self.params.pop("n_estimators", 100))
        self.n_estimators_ = self._num_boost_round
        self.best_iteration_ = None
    def _to_train_params(self):
        p = self.params.copy()
        if "random_state" in p and "seed" not in p:
            p["seed"] = p.pop("random_state")
        if "n_jobs" in p and "nthread" not in p:
            p["nthread"] = p.pop("n_jobs")
        p.setdefault("objective", "binary:logistic")
        p.setdefault("eval_metric", "auc")
        p.setdefault("max_bin", 256)
        if torch.cuda.is_available():
            p["tree_method"] = "hist"
            p["device"] = "cuda"
            p.pop("predictor", None)
            p.pop("gpu_id", None)
        else:
            p["tree_method"] = "hist"
            p["predictor"] = "auto"
            p.pop("device", None)
        return p
    def fit(self, X_tr, y_tr, X_va=None, y_va=None, early_stopping_rounds=None, verbose_eval=False):
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        train_params = self._to_train_params()
        evals = [(dtr, "train")]
        if X_va is not None and y_va is not None:
            dva = xgb.DMatrix(X_va, label=y_va)
            evals.append((dva, "valid"))
        def _train(params):
            kwargs = dict(
                params=params, dtrain=dtr, num_boost_round=self._num_boost_round,
                evals=evals, verbose_eval=verbose_eval
            )
            if isinstance(early_stopping_rounds, int) and early_stopping_rounds > 0 and len(evals) > 1:
                kwargs["early_stopping_rounds"] = early_stopping_rounds
            return xgb.train(**kwargs)
        try:
            self.booster_ = _train(train_params)
        except xgb.core.XGBoostError as e:
            msg = str(e).lower()
            if any(k in msg for k in ["gpu", "cuda", "device"]):
                cpu_params = train_params.copy()
                cpu_params.pop("device", None)
                cpu_params["tree_method"] = "hist"
                cpu_params["predictor"] = "auto"
                logging.warning("XGBoost GPU training failed, falling back to CPU.")
                self.booster_ = _train(cpu_params)
            else:
                raise
        self.best_iteration_ = getattr(self.booster_, "best_iteration", self._num_boost_round - 1)
        self.best_ntree_limit_ = getattr(self.booster_, "best_ntree_limit", None)
        return self
    def predict_proba(self, X):
        d = xgb.DMatrix(X)
        iteration_limit = self.best_iteration_ + 1 if self.best_iteration_ is not None else None
        try:
            p1 = self.booster_.predict(d, iteration_range=(0, iteration_limit))
        except (TypeError, xgb.core.XGBoostError):
            ntree_limit = self.best_ntree_limit_ or iteration_limit
            p1 = self.booster_.predict(d, ntree_limit=ntree_limit)
        p1 = np.asarray(p1, dtype=float).reshape(-1)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

class LGBBoosterWrapper:
    def __init__(self, booster, params):
        self.booster_ = booster
        self.params_  = params.copy()
        self.best_iteration_ = getattr(self.booster_, "best_iteration", None)
    def predict_proba(self, X):
        num_iter = self.best_iteration_ if self.best_iteration_ is not None else None
        p1 = self.booster_.predict(X, num_iteration=num_iter)
        p1 = np.asarray(p1, dtype=float).reshape(-1)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
    def get_params(self, deep=False):
        return self.params_.copy()

# =====================================================================================
# ---- 3. PHASE 1: TRAIN + SAVE
# =====================================================================================

def train_and_save_final_stage1_model(seed=FINAL_SEED):
    global SEED
    SEED = int(seed)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    logging.info(f"=== PHASE 1: Training Final Stage 1 Models (Seed: {SEED}) ===")

    # 1) Load Train+Val
    with SectionTimer("Loading train and val data"):
        df_train_base = pd.read_parquet(DATA_PATHS["train"], engine="pyarrow")
        df_val_base   = pd.read_parquet(DATA_PATHS["val"],   engine="pyarrow")
        df_train_rb   = pd.read_parquet(DATA_PATHS["train_robust"], engine="pyarrow")
        df_val_rb     = pd.read_parquet(DATA_PATHS["val_robust"],   engine="pyarrow")

        df_train = df_train_base.drop(columns=action_trash_list, errors="ignore").merge(df_train_rb[action_list], on=ID_COL, how="left")
        df_val   = df_val_base.drop(columns=action_trash_list, errors="ignore").merge(df_val_rb[action_list], on=ID_COL, how="left")

        # Optional: NAT_CD category alignment (encoder가 -1 처리하므로 필수는 아님)
        if "NAT_CD" in df_train.columns:
            df_train["NAT_CD"] = df_train["NAT_CD"].astype("category")
            train_cats = df_train["NAT_CD"].cat.categories
            if "NAT_CD" in df_val.columns:
                df_val["NAT_CD"] = pd.Categorical(df_val["NAT_CD"], categories=train_cats)

        full_train_df = pd.concat([df_train, df_val], ignore_index=True)
        logging.info(f"Combined train+val data shape: {full_train_df.shape}")

    # 2) Preprocess Fit & Transform
    with SectionTimer("Fitting preprocessors on full train data"):
        y_full_train = (full_train_df[TARGET_COL] > 0).astype(int)
        X_full_raw, feat_cols, cat_cols = build_features(full_train_df, TARGET_COL, [ID_COL])

        # Identify numeric cols BEFORE encoding
        num_cols = [c for c in feat_cols if c not in cat_cols]

        # Fit imputer and encoder
        num_cols_fitted, imputer_medians = fit_imputer(X_full_raw, num_cols)
        encoder = OrdinalCategoryEncoder().fit(X_full_raw, cat_cols)

        # Transform: encoder -> imputer -> sanitize
        X_full_encoded = encoder.transform(X_full_raw)
        X_full_imputed = apply_imputer(X_full_encoded, num_cols_fitted, imputer_medians)
        X_full_train = _sanitize_cols(X_full_imputed)

        # CatBoost categorical idx on the FINAL matrix
        cat_cols_idx = [X_full_train.columns.get_loc(c) for c in cat_cols if c in X_full_train.columns]

        # Save preprocessors
        joblib.dump(encoder, ARTIFACTS_DIR / "stage1_encoder.joblib")
        joblib.dump((num_cols_fitted, imputer_medians), ARTIFACTS_DIR / "stage1_imputer.joblib")
        logging.info(f"Saved encoder/imputer to {ARTIFACTS_DIR}")

    # 3) Train Models (no early stopping)
    models = {}

    # LightGBM
    with SectionTimer("Training final LGBM model"):
        lgb_params = STAGE1_FIXED["lgbm"].copy()
        lgb_params["random_state"] = SEED
        lgb_params["n_jobs"] = max(1, (os.cpu_count() or 8)//2)
        lgb_params["force_row_wise"] = True

        params_for_train = lgb_params.copy()
        num_boost_round = int(params_for_train.pop("n_estimators"))
        params_for_train.pop("metric", None)

        lgb_train_data = lgb.Dataset(X_full_train, label=y_full_train)

        booster = lgb.train(
            params_for_train, lgb_train_data,
            num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(period=100)]
        )
        booster.best_iteration = num_boost_round
        models["lgbm"] = LGBBoosterWrapper(booster, lgb_params)
        joblib.dump(models["lgbm"], ARTIFACTS_DIR / "stage1_lgbm_final.joblib")
        booster.save_model(str(ARTIFACTS_DIR / "stage1_lgbm_final.txt"))

    # XGBoost
    with SectionTimer("Training final XGBoost model"):
        xgb_params = STAGE1_FIXED["xgb"].copy()
        xgb_params["random_state"] = SEED
        xgb_params["n_jobs"] = max(1, (os.cpu_count() or 8)//2)
        n_neg = (y_full_train == 0).sum()
        n_pos = (y_full_train == 1).sum()
        xgb_params["scale_pos_weight"] = n_neg / max(n_pos, 1) if n_pos > 0 else 1.0

        models["xgb"] = XGBCompat(**xgb_params)
        models["xgb"].fit(X_full_train.values, y_full_train.values, X_va=None, y_va=None, early_stopping_rounds=None, verbose_eval=100)
        joblib.dump(models["xgb"], ARTIFACTS_DIR / "stage1_xgb_final.joblib")
        models["xgb"].booster_.save_model(str(ARTIFACTS_DIR / "stage1_xgb_final.json"))

    # CatBoost
    with SectionTimer("Training final CatBoost model"):
        cat_params = STAGE1_FIXED["cat"].copy()
        cat_params["random_seed"] = SEED
        models["cat"] = CatBoostClassifier(**cat_params, **CAT_TASK_PARAMS)
        pool_full_train = Pool(X_full_train, y_full_train, cat_features=cat_cols_idx or None)
        models["cat"].fit(pool_full_train, verbose=100)
        joblib.dump(models["cat"], ARTIFACTS_DIR / "stage1_cat_final.joblib")
        models["cat"].save_model(str(ARTIFACTS_DIR / "stage1_cat_final.cbm"))

    # 4) Save manifest (feat_order, cat_idx, cutoffs, params)
    feat_order = X_full_train.columns.tolist()
    manifest = {
        "feat_order": feat_order,
        "cat_cols_idx": cat_cols_idx,
        "cutoffs": FINAL_CUTOFFS,
        "params": STAGE1_FIXED,
        "seed": int(SEED),
        "library_versions": {
            "catboost": getattr(__import__("catboost"), "__version__", "unknown"),
            "lightgbm": getattr(__import__("lightgbm"), "__version__", "unknown"),
            "xgboost":  getattr(__import__("xgboost"), "__version__", "unknown"),
        }
    }
    import json
    json.dump(manifest, open(ARTIFACTS_DIR / "stage1_manifest.json","w"), ensure_ascii=False, indent=2)

    logging.info(f"Saved final models and manifest to {ARTIFACTS_DIR}")
    return ARTIFACTS_DIR

# =====================================================================================
# ---- 4. PHASE 2: PREDICT ALL (train+val+test)
# =====================================================================================

def predict_stage1_on_all_data(artifacts_dir: Path, cutoffs: Dict):
    logging.info("=== PHASE 2: Predicting on All Data (Train+Val+Test) ===")

    # 1) Load data
    with SectionTimer("Loading all data (train+val+test)"):
        df_train_base = pd.read_parquet(DATA_PATHS["train"], engine="pyarrow")
        df_val_base   = pd.read_parquet(DATA_PATHS["val"],   engine="pyarrow")
        df_test_base  = pd.read_parquet(DATA_PATHS["test"],  engine="pyarrow")
        df_train_rb   = pd.read_parquet(DATA_PATHS["train_robust"], engine="pyarrow")
        df_val_rb     = pd.read_parquet(DATA_PATHS["val_robust"],   engine="pyarrow")
        df_test_rb    = pd.read_parquet(DATA_PATHS["test_robust"],  engine="pyarrow")

        df_train = df_train_base.merge(df_train_rb[action_list], on=ID_COL, how="left")
        df_val   = df_val_base.merge(df_val_rb[action_list], on=ID_COL, how="left")
        df_test  = df_test_base.merge(df_test_rb[action_list], on=ID_COL, how="left")

        original_columns = df_train.columns.tolist()  # 기준

        all_data_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
        cols_to_drop = [col for col in action_trash_list if col in all_data_df.columns]
        all_data_df = all_data_df.drop(columns=cols_to_drop, errors='ignore')

        logging.info(f"Combined all data shape: {all_data_df.shape}")
        final_original_columns = [col for col in original_columns if col not in cols_to_drop]

    # 2) Load artifacts
    with SectionTimer("Loading models & preprocessors & manifest"):
        lgbm_final = joblib.load(artifacts_dir / "stage1_lgbm_final.joblib")
        xgb_final  = joblib.load(artifacts_dir / "stage1_xgb_final.joblib")
        cat_final  = joblib.load(artifacts_dir / "stage1_cat_final.joblib")
        encoder    = joblib.load(artifacts_dir / "stage1_encoder.joblib")
        num_cols_fitted, imputer_medians = joblib.load(artifacts_dir / "stage1_imputer.joblib")

        import json
        manifest = json.load(open(artifacts_dir / "stage1_manifest.json"))
        feat_order = manifest["feat_order"]
        cat_cols_idx_from_manifest = manifest["cat_cols_idx"]
        logging.info("Artifacts loaded.")

    # 3) Preprocess (transform only) + REINDEX by feat_order
    with SectionTimer("Preprocessing all data using loaded artifacts"):
        cat_cols_from_encoder = encoder.cols
        # 순서가 중요한 지점: set 금지!
        feat_cols = list(cat_cols_from_encoder) + [c for c in num_cols_fitted if c not in cat_cols_from_encoder]
        valid_feat_cols = [c for c in feat_cols if c in all_data_df.columns]
        X_all_raw = all_data_df[valid_feat_cols].copy()

        X_all_encoded = encoder.transform(X_all_raw)
        X_all_imputed = apply_imputer(X_all_encoded, num_cols_fitted, imputer_medians)
        X_all_processed = _sanitize_cols(X_all_imputed)

        # 🔴 가장 중요: 학습 시 피처 순서로 강제 정렬
        X_all_processed = X_all_processed.reindex(columns=feat_order, fill_value=0)

        # CatBoost에 넘길 cat_features index (feat_order 기준)
        final_cat_cols_idx = cat_cols_idx_from_manifest

    # 4) Predict proba
    with SectionTimer("Predicting probabilities on all data"):
        proba = {}
        proba["lgbm"] = lgbm_final.predict_proba(X_all_processed)[:, 1]
        proba["xgb"]  = xgb_final.predict_proba(X_all_processed.values)[:, 1]
        pool_all = Pool(X_all_processed, cat_features=final_cat_cols_idx or None)
        proba["cat"]  = cat_final.predict_proba(pool_all)[:, 1]

    # 5) Hard vote with cutoffs
    with SectionTimer("Applying hard vote with fixed cutoffs"):
        y_hat_all = hard_vote(proba, cutoffs)

    # 6) Merge & Save
    with SectionTimer("Merging predictions with original data and saving"):
        # 평균 확률(ensemble soft ref): Stage2 라우팅 지표로 활용
        stage1_proba = (proba["lgbm"] + proba["xgb"] + proba["cat"]) / 3.0

        pred_df = pd.DataFrame({
            "pred_is_payer": y_hat_all,
            "proba_lgbm": proba["lgbm"],
            "proba_xgb":  proba["xgb"],
            "proba_cat":  proba["cat"],
            "stage1_proba": stage1_proba
        }, index=all_data_df.index)

        cols_to_keep = [col for col in final_original_columns if col in all_data_df.columns]
        final_output_df = pd.concat([all_data_df[cols_to_keep], pred_df], axis=1)

        output_filename = PREDICTIONS_DIR / "stage1_final_predictions_all_data.parquet"
        final_output_df.to_parquet(output_filename, index=False, engine='pyarrow')
        logging.info(f"Final predictions saved to: {output_filename}")
        logging.info(f"Output shape: {final_output_df.shape}")

    return output_filename

# =====================================================================================
# ---- 5. MAIN
# =====================================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f'final_pipeline_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)]
    )
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback); return
        logging.error("💥 Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    logging.info("=" * 80)
    logging.info("🚀 Starting Stage 1 Final Training & Prediction Pipeline")
    logging.info(f"🔹 Data Directory: {DATA_DIR.resolve()}")
    logging.info(f"🔹 Artifacts Directory: {ARTIFACTS_DIR.resolve()}")
    logging.info(f"🔹 Predictions Directory: {PREDICTIONS_DIR.resolve()}")
    logging.info(f"🔹 Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.info(f"🔹 Fixed Cutoffs: {FINAL_CUTOFFS}")
    logging.info("=" * 80)

    if not Path(DATA_PATHS["train"]).exists():
        logging.error(f"❌ FATAL: Train data not found at {DATA_PATHS['train']}")
        return

    pd.options.display.float_format = '{:,.2f}'.format
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    CPU = os.cpu_count() or 2
    os.environ["OMP_NUM_THREADS"] = str(CPU)

    try:
        artifacts_path = train_and_save_final_stage1_model(seed=FINAL_SEED)
        prediction_file = predict_stage1_on_all_data(artifacts_dir=artifacts_path, cutoffs=FINAL_CUTOFFS)
        logging.info("\n✅ Pipeline Finished Successfully!")
        logging.info(f"Final predictions are in: {prediction_file}")
    except Exception:
        logging.error("☠️ A critical error occurred during pipeline execution.")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()