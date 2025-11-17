"""
Stage 2 Unified Pipeline Core Logic (with single ensemble() utility).
- Config-driven (no CLI parsing).
- One `ensemble()` function handles both soft/hard voting for:
  * per-seed (model ensemble)
  * multi-seed (seed ensemble) 
"""

import json
import sys
import os
import random
import logging
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    fbeta_score, f1_score, precision_recall_fscore_support,
    precision_score, recall_score, average_precision_score,
    precision_recall_curve
)

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

def feature_drop_cols():
    """
    모델 입력 생성 시 항상 제거할 컬럼(타깃/ID/Stage1 결과 등)만 명시.
    TVT 컬럼(stage1_tvt, stage2_tvt)은 build_features 호출 '직전'에
    errors='ignore'로 미리 드롭하고, 여기에는 넣지 않는다.
    """
    return [ID_COL, TARGET_COL, PRED_PAYER_COL]

import config
from config import (
    TARGET_COL, ID_COL, PRED_PAYER_COL,
    DEF_OUTPUT_DIR, BASE_SPLIT_SEED, CAT_TASK_PARAMS,
    WHALE_CUT, F_BETA, WHALE_Q, SEED, N_JOBS,
    DEFAULT_TRIALS, MIN_PREC_AT_CUT, DEFAULT_TEST_SIZE,
    TABPFN_DEVICE, TABPFN_CONFIGS, OPTUNA_SEED,
    USE_STAGE1_FEATURES, DEFAULT_INPUT_DATA_PATH,
    NO_CATBOOST, NO_LGBM, NO_TABPFN,
    ENSEMBLE_MODE, SKIP_IF_EXISTS, DEFAULT_SEEDS_STR
)

from utils import (
    SectionTimer, _sanitize_cols, tune_cutoff,
    make_tabpfn_classifier, parse_seeds
)

from data_preprocessing import (
    OrdinalCategoryEncoder, build_features, fit_imputer, apply_imputer, _HAS_TABPFN
)

from models import (
    tune_lgbm_cls, tune_cat_cls, plot_lgbm_error_trajectory, plot_cat_error_trajectory
)

# -----------------------------
# Single ensemble utility
# -----------------------------
def ensemble(
    scores,                     # dict[str, np.ndarray] or 2D np.ndarray (n, k)
    mode: str = "soft",         # "soft" or "hard"
    *, 
    cutoffs=None,               # hard: dict[str,float] or scalar; soft: ignored
    weights=None,               # dict[str,float] or 1D array length k
    threshold: float | None = None  # soft: optional final threshold to directly get labels
  ):
    if isinstance(scores, dict):
        keys = list(scores.keys())
        M = _np.column_stack([_np.asarray(scores[k]).ravel() for k in keys])
    else:
        keys = None
        M = _np.asarray(scores)
        if M.ndim != 2:
            raise ValueError("scores must be 2D array or dict of 1D arrays")

    n, k = M.shape

    # weights
    if weights is None:
        w = _np.ones(k, dtype=float) / max(k, 1)
    elif isinstance(weights, dict):
        if keys is None:
            raise ValueError("weights as dict requires dict scores for key alignment")
        arr = [float(weights.get(k_i, 0.0)) for k_i in keys]
        s = sum(arr)
        w = _np.asarray(arr, dtype=float) / (s if s > 0 else 1.0)
    else:
        w = _np.asarray(weights, dtype=float)
        s = float(w.sum())
        w = w / (s if s > 0 else 1.0)
        if w.shape[0] != k:
            raise ValueError("weights length mismatch")

    if mode == "soft":
        proba = (M * w).sum(axis=1)
        if threshold is None:
            return proba, None
        yhat = (proba >= float(threshold)).astype(int)
        return proba, yhat

    if mode == "hard":
        if cutoffs is None:
            cuts = _np.full(k, 0.5, dtype=float)
        elif isinstance(cutoffs, dict):
            if keys is None:
                raise ValueError("cutoffs as dict requires dict scores for key alignment")
            cuts = _np.asarray([float(cutoffs.get(k_i, 0.5)) for k_i in keys], dtype=float)
        else:
            cuts = _np.full(k, float(cutoffs), dtype=float)

        votes = (M >= cuts).astype(int)
        tally = (votes * w).sum(axis=1)
        yhat = (tally >= 0.5).astype(int)

        proba_ref = M.mean(axis=1) if k > 0 else _np.zeros(n, dtype=float)
        return proba_ref, yhat

    raise ValueError("mode must be 'soft' or 'hard'")

# -----------------------------
# Global artifact paths
# -----------------------------
OUTPUT_DIR = Path(DEF_OUTPUT_DIR)
ARTIFACTS_PATH = OUTPUT_DIR / "global_artifacts"
ENCODER_PATH = ARTIFACTS_PATH / "stage2_encoder.joblib"
IMPUTER_PATH = ARTIFACTS_PATH / "stage2_imputer.joblib"
WHALE_CUT_PATH = ARTIFACTS_PATH / "stage2_whale_cut.json"
MODELS_PATH = OUTPUT_DIR / "models"
LOGS_PATH = OUTPUT_DIR / "logs"

# === Stage2 splitting policy ===
USE_STAGE2_STRATIFIED_SPLIT = True
STAGE2_SPLIT_RATIOS = (0.60, 0.20, 0.20)

BASE_DROP_COLS = [ID_COL, TARGET_COL, PRED_PAYER_COL, "stage1_tvt", "stage2_tvt"]

# -----------------------------
# Initial data prep
# -----------------------------
def create_stage1_data_parquet():
    data_dir = DEFAULT_INPUT_DATA_PATH.parent
    final_output_path = DEFAULT_INPUT_DATA_PATH

    if final_output_path.exists():
        logging.info(f"✅ Initial Data Prep: {final_output_path.name} already exists. Skipping data creation.")
        return

    logging.info("▶ START: Creating initial stage1_data.parquet")
    try:
        base_data_path = data_dir / "stage1_final_predictions_all_data.parquet"
        if not base_data_path.exists():
            raise FileNotFoundError(f"Base data not found: {base_data_path}")
        df_base = pd.read_parquet(base_data_path)

        tvt_maps = []
        tvt_files = {
            "train": data_dir / "train_df_5days.parquet",
            "val":   data_dir / "val_df_5days.parquet",
            "test":  data_dir / "test_df_5days.parquet",
        }
        for tvt_label, file_path in tvt_files.items():
            if file_path.exists():
                df_tvt = pd.read_parquet(file_path, columns=[ID_COL])
                df_tvt["stage1_tvt"] = tvt_label
                tvt_maps.append(df_tvt[[ID_COL, "stage1_tvt"]])
            else:
                logging.warning(f"⚠️ TVT file missing: {file_path}. Skipping {tvt_label} mapping.")
        if not tvt_maps:
            raise ValueError("No TVT map files were found. Cannot create stage1_tvt column.")

        df_tvt_map = pd.concat(tvt_maps, ignore_index=True)
        if df_tvt_map[ID_COL].duplicated().any():
            logging.warning("⚠️ Duplicate PLAYERID found in TVT map files. Using first occurrence.")
            df_tvt_map = df_tvt_map.drop_duplicates(subset=[ID_COL], keep='first')

        df_final = pd.merge(df_base, df_tvt_map, on=ID_COL, how="left")
        df_final['stage1_tvt'] = df_final['stage1_tvt'].fillna('unknown')

        df_final.to_parquet(final_output_path, index=False)
        logging.info(f"✅ Initial Data Prep: Successfully created {final_output_path.name} ({df_final.shape}).")

    except Exception as e:
        logging.error(f"❌ Initial Data Prep failed: {e}", exc_info=True)
        raise RuntimeError(f"Data preparation failed: {e}")

# -----------------------------
# Data loading / splitting
# -----------------------------
def load_and_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_full = pd.read_parquet(DEFAULT_INPUT_DATA_PATH)
    if ID_COL in df_full.columns:
        df_full = df_full.set_index(ID_COL, drop=False)
    
    # Stage 1 예측 과금자만 사용
    df_filtered = df_full[df_full[PRED_PAYER_COL] == 1].copy()
    if len(df_filtered) == 0:
        raise ValueError("❌ No predicted payers in the dataset (df_filtered).")
    logging.info(f"✅ Filtered Data (Total): {df_filtered.shape}")

    if USE_STAGE2_STRATIFIED_SPLIT:
        payers = df_filtered[df_filtered[TARGET_COL] > 0][TARGET_COL]
        if len(payers) == 0:
            raise ValueError("❌ No true payers in filtered data; cannot compute provisional whale cut.")
        whale_cut_prov = float(np.quantile(payers, WHALE_Q))
        y_whale_all = (df_filtered[TARGET_COL] >= whale_cut_prov).astype(int)

        tr_ratio, va_ratio, te_ratio = STAGE2_SPLIT_RATIOS
        hold_ratio = va_ratio + te_ratio

        df_tr, df_hold = train_test_split(
            df_filtered,
            test_size=hold_ratio,
            random_state=BASE_SPLIT_SEED,
            stratify=y_whale_all
        )

        y_hold = (df_hold[TARGET_COL] >= whale_cut_prov).astype(int)
        val_ratio_in_hold = va_ratio / (va_ratio + te_ratio)
        df_va, df_te = train_test_split(
            df_hold,
            test_size=(1.0 - val_ratio_in_hold),
            random_state=BASE_SPLIT_SEED,
            stratify=y_hold
        )

        df_tr = df_tr.copy(); df_tr["stage2_tvt"] = "train"
        df_va = df_va.copy(); df_va["stage2_tvt"] = "val"
        df_te = df_te.copy(); df_te["stage2_tvt"] = "test"

        df_train_val = pd.concat([df_tr, df_va], axis=0)
        df_test = df_te.copy()
        df_all = pd.concat([df_tr, df_va, df_te], axis=0)

        logging.info(f"✅ Stage2 stratified split | "
                     f"train={df_tr.shape} val={df_va.shape} test={df_te.shape} "
                     f"(BASE_SPLIT_SEED={BASE_SPLIT_SEED})")

    else:
        if "stage1_tvt" not in df_filtered.columns:
            raise KeyError("❌ 'stage1_tvt' column not found. Run initial data prep or enable Stage2 split.")
        df_train_val = df_filtered[df_filtered["stage1_tvt"].isin(["train", "val"])].copy()
        df_test = df_filtered[df_filtered["stage1_tvt"] == "test"].copy()
        df_all = df_filtered.copy()
        logging.info(f"✅ Stage1 TVT split | train+val={df_train_val.shape} test={df_test.shape}")

    if len(df_train_val) == 0:
        raise ValueError("❌ Stage 2 Training/Validation set is empty.")
    if len(df_test) == 0:
        logging.warning("⚠️ Stage 2 Test set is empty. Final test metrics will be skipped.")

    return df_train_val, df_test, df_all

# -----------------------------
# Single-seed train
# -----------------------------
def run_stage2_train_core(train_seed: int, df_train_val: pd.DataFrame):
    np.random.seed(train_seed)
    random.seed(train_seed)

    output_dir = OUTPUT_DIR
    trials = DEFAULT_TRIALS

    ARTIFACTS_PATH.mkdir(exist_ok=True)
    logging.info(f"--- Stage 2 Training (Seed: {train_seed}) ---")

    Xtr = None; y_tr = None
    Xva = None; y_va = None
    cat_cols_idx = None
    pos_prior = 0.0

    whale_cut = 0.0
    enc = None
    num_cols = None
    med = None

    best = {"score": -1.0, "strategy": None, "models": {}, "preds": {}, "cutoffs": {}}
    best_lgb_params = {}; best_cat_params = {}
    critical_error = None

    with SectionTimer("Data Loading and Preparation"):
        try:
            df_train_val_ = df_train_val.copy()
            df_train_val_.drop(columns=["stage2_tvt", "stage1_tvt"], errors="ignore", inplace=True)
            _, _, cat_cols_global = build_features(df_train_val_, TARGET_COL, feature_drop_cols())

            if not ENCODER_PATH.exists():
                enc = OrdinalCategoryEncoder().fit(df_train_val, cat_cols_global)
                joblib.dump(enc, ENCODER_PATH)
                logging.info(f"✅ Created and saved global encoder at {ENCODER_PATH}")
            else:
                enc = joblib.load(ENCODER_PATH)
                logging.info(f"✅ Loaded global encoder from {ENCODER_PATH}")

            if not IMPUTER_PATH.exists() or not WHALE_CUT_PATH.exists() or config.WHALE_CUT == 0.0:
                df_non_test = df_train_val.copy()
                df_payers_non_test = df_non_test[df_non_test[TARGET_COL] > 0]
                if len(df_payers_non_test) == 0:
                    raise ValueError("❌ No true payers in non-test pool for whale_cut.")
                whale_cut = float(np.quantile(df_payers_non_test[TARGET_COL], WHALE_Q))
                json.dump({"whale_cut": whale_cut}, open(WHALE_CUT_PATH, "w"))
                config.WHALE_CUT = whale_cut
                logging.info(f"✅ Global whale_cut(non-test payers, q={WHALE_Q:.2f}): {whale_cut:.2f}")

                df_non_test.drop(columns=["stage2_tvt", "stage1_tvt"], errors="ignore", inplace=True)
                X_non_test_raw, _, _ = build_features(df_non_test, TARGET_COL, feature_drop_cols())
                X_non_test_enc = enc.transform(X_non_test_raw)
                num_cols, med = fit_imputer(X_non_test_enc)
                joblib.dump((num_cols, med), IMPUTER_PATH)
                logging.info(f"✅ Global imputer(median) saved: {IMPUTER_PATH}")
            else:
                num_cols, med = joblib.load(IMPUTER_PATH)
                whale_cut = json.load(open(WHALE_CUT_PATH))["whale_cut"]
                config.WHALE_CUT = whale_cut
                logging.info(f"✅ Loaded global artifacts: whale_cut={whale_cut:.2f}")

            tvt_col = "stage2_tvt" if "stage2_tvt" in df_train_val.columns else "stage1_tvt"
            if tvt_col not in df_train_val.columns:
                raise KeyError("❌ Neither 'stage2_tvt' nor 'stage1_tvt' found in df_train_val.")

            df_tr = df_train_val[df_train_val[tvt_col] == "train"].copy()
            df_va = df_train_val[df_train_val[tvt_col] == "val"].copy()
            if len(df_tr) == 0 or len(df_va) == 0:
                raise ValueError("❌ Invalid Stage2 split: empty train/val.")

            y_tr = (df_tr[TARGET_COL] >= config.WHALE_CUT).astype(int)
            y_va = (df_va[TARGET_COL] >= config.WHALE_CUT).astype(int)

            # TVT 컬럼은 build_features 전에 안전 제거(중복 드롭로 인한 KeyError 방지)
            df_tr.drop(columns=["stage2_tvt", "stage1_tvt"], errors="ignore", inplace=True)
            df_va.drop(columns=["stage2_tvt", "stage1_tvt"], errors="ignore", inplace=True)

            drop_cols_for_feature = feature_drop_cols()
            Xtr_raw, feat_cols, cat_cols = build_features(df_tr, TARGET_COL, drop_cols_for_feature)
            Xva_raw = df_va[feat_cols].copy()

            Xtr = enc.transform(Xtr_raw)
            Xva = enc.transform(Xva_raw)

            cat_cols_idx = [Xtr.columns.get_loc(c) for c in cat_cols if c in Xtr.columns]

            Xtr = apply_imputer(Xtr, num_cols, med)
            Xva = apply_imputer(Xva, num_cols, med)

            Xtr = _sanitize_cols(Xtr)
            Xva = _sanitize_cols(Xva)

            Xtr.index = df_tr.index
            Xva.index = df_va.index

            pos_prior = float(y_tr.mean())
            logging.info(f"    - Train set shape: {Xtr.shape}")
            logging.info(f"    - Validation set shape: {Xva.shape}")
            logging.info(f"    - Train Pos Prior (Whale ratio): {pos_prior:.4f}")

        except Exception as e:
            logging.error(f"❌ Data Preparation failed: {e}")
            critical_error = e

    if not critical_error:
        try:
            with SectionTimer("Model Training and Ensemble"):
                best_temp = {"score": -1.0, "strategy": None}

                for strat in ["prior", "reweight"]:
                    models = {}; preds = {}; cutoffs = {}

                    if not NO_CATBOOST:
                        cat_params = tune_cat_cls(Xtr, y_tr, Xva, y_va, cat_cols_idx, "stage2", strat, pos_prior, DEFAULT_TRIALS, train_seed)
                        best_cat_params = cat_params
                        cat2 = CatBoostClassifier(**cat_params, **CAT_TASK_PARAMS, od_type="Iter", od_wait=200)
                        pool_tr = Pool(Xtr, y_tr, cat_features=cat_cols_idx or None)
                        pool_va = Pool(Xva, y_va, cat_features=cat_cols_idx or None)
                        cat2.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
                        p_cat = cat2.predict_proba(pool_va)[:, 1]
                        t_cat, _ = tune_cutoff(y_va, p_cat, strategy=strat, train_pos_prior=pos_prior, metric="fbeta", beta=config.F_BETA)
                        models["cat"], preds["cat"], cutoffs["cat"] = cat2, p_cat, t_cat

                    if not NO_LGBM:
                        lgb_params = tune_lgbm_cls(Xtr, y_tr, Xva, y_va, "stage2", strat, pos_prior, DEFAULT_TRIALS, train_seed)
                        best_lgb_params = lgb_params
                        lgbm2 = lgb.LGBMClassifier(**lgb_params)
                        lgbm2.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], eval_metric="auc",
                                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
                        p_lgb = lgbm2.predict_proba(Xva)[:, 1]
                        t_lgb, _ = tune_cutoff(y_va, p_lgb, strategy=strat, train_pos_prior=pos_prior, metric="fbeta", beta=config.F_BETA)
                        models["lgbm"], preds["lgbm"], cutoffs["lgbm"] = lgbm2, p_lgb, t_lgb

                    if _HAS_TABPFN and not NO_TABPFN:
                        tab2 = make_tabpfn_classifier(device=TABPFN_DEVICE, seed=train_seed, n_ens=TABPFN_CONFIGS)
                        if tab2 is not None:
                            tab2.fit(Xtr.values, y_tr.values)
                            p_tab = tab2.predict_proba(Xva.values)[:, 1]
                            t_tab, _ = tune_cutoff(y_va, p_tab, strategy=strat, train_pos_prior=pos_prior, metric="fbeta", beta=config.F_BETA)
                            models["tab"], preds["tab"], cutoffs["tab"] = tab2, p_tab, t_tab

                    if not preds:
                        raise ValueError("❌ No models were trained.")

                    # unified ensemble (hard-by-model) on VAL
                    proba_ref, yhat_final = ensemble(preds, mode="hard", cutoffs=cutoffs, weights=None)
                    ap_val_score = average_precision_score(y_va, proba_ref)

                    if ap_val_score > best_temp["score"]:
                        best_temp.update(score=ap_val_score, strategy=strat, models=models, preds=preds, cutoffs=cutoffs)

                best = best_temp

                proba_ref, yhat_final = ensemble(best["preds"], mode="hard", cutoffs=best["cutoffs"], weights=None)
                f1_val = f1_score(y_va, yhat_final, zero_division=0)
                pr_val = precision_score(y_va, yhat_final, zero_division=0)
                rc_val = recall_score(y_va, yhat_final, zero_division=0)
                ap_val = average_precision_score(y_va, proba_ref)
                fbeta_final = fbeta_score(y_va, yhat_final, beta=config.F_BETA, zero_division=0)
                logging.info(f"[VAL] AP={ap_val:.4f} F1={f1_val:.4f} F{config.F_BETA}={fbeta_final:.4f} | Strategy={best['strategy']}")

                models_list = ["cat", "lgbm", "tab"]
                metrics = {"model": [m for m in models_list if m in best["models"]] + ["ensemble"], "f1": [], "precision": [], "recall": [], "ap": []}
                for m in models_list:
                    if m in best["models"]:
                        p = best["preds"].get(m)
                        t = best["cutoffs"].get(m)
                        y_hat = (p >= t).astype(int)
                        metrics["f1"].append(f1_score(y_va, y_hat, zero_division=0))
                        metrics["precision"].append(precision_score(y_va, y_hat, zero_division=0))
                        metrics["recall"].append(recall_score(y_va, y_hat, zero_division=0))
                        metrics["ap"].append(average_precision_score(y_va, p))

                metrics["f1"].append(f1_val); metrics["precision"].append(pr_val); metrics["recall"].append(rc_val); metrics["ap"].append(ap_val)
                metrics_df = pd.DataFrame(metrics)

                CUTOFFS_FILE = ARTIFACTS_PATH / f"stage2_cutoffs_{train_seed}.json"
                save_payload = {
                    "strategy": best["strategy"],
                    "whale_cut": float(config.WHALE_CUT),
                    "per_model": {k: float(v) for k, v in best["cutoffs"].items()},
                    "ensemble": 0.5
                }
                with open(CUTOFFS_FILE, "w") as f:
                    json.dump(save_payload, f)
                logging.info(f"✅ Saved VAL thresholds to {CUTOFFS_FILE}")

        except Exception as e:
            logging.error(f"❌ Model Training or Ensemble failed: {e}")
            critical_error = e

        try:
            if not NO_LGBM and best_lgb_params and best["models"].get("lgbm"):
                plot_lgbm_error_trajectory(Xtr, y_tr, Xva, y_va, best_lgb_params, output_dir, train_seed)
            if not NO_CATBOOST and best_cat_params and best["models"].get("cat"):
                plot_cat_error_trajectory(Xtr, y_tr, Xva, y_va, cat_cols_idx, best_cat_params, output_dir, train_seed)
        except Exception as e:
            logging.error(f"⚠️ Plot generation failed (non-critical): {e}", exc_info=True)

    if best["models"]:
        MODELS_PATH.mkdir(exist_ok=True)
        model_file = MODELS_PATH / f"stage2_models_{train_seed}.joblib"
        try:
            joblib.dump(best["models"], model_file)
        except Exception as e:
            logging.error(f"❌ Failed to save Stage 2 models to joblib: {e}")

    if Xva is not None and y_va is not None and best["score"] != -1.0:
        pred_file = output_dir / f"stage2_predictions_val_{train_seed}.csv"
        metrics_file = output_dir / f"stage2_metrics_{train_seed}.csv"
        try:
            proba_ref, yhat_final = ensemble(best["preds"], mode="hard", cutoffs=best["cutoffs"], weights=None)
            pred_data = {
                ID_COL: Xva.index.values, "seed": train_seed, "strategy": best["strategy"], "whale_cutoff": config.WHALE_CUT,
                "p_cat": best["preds"].get("cat"), "p_lgbm": best["preds"].get("lgbm"), "p_tab": best["preds"].get("tab"),
                "proba_ensemble": proba_ref, "yhat_stage2": yhat_final, "IS_WHALE_true": y_va.values,
            }
            pred_df = pd.DataFrame({k: v for k, v in pred_data.items() if v is not None and v is not False})
            pred_df.to_csv(pred_file, index=False)
            metrics_df.to_csv(metrics_file, index=False)
        except Exception as e:
            logging.error(f"❌ Failed to save CSV files: {e}", exc_info=True)

    if critical_error:
        logging.error(f"☠️ Pipeline terminated due to critical error: {critical_error}")
        raise critical_error

# -----------------------------
# Predict ALL
# -----------------------------
def _predict_with_model(model_key: str, model: Any, X: pd.DataFrame, cat_cols_idx: list) -> np.ndarray:
    if model_key == "cat":
        pool = Pool(X, cat_features=cat_cols_idx or None)
        proba = model.predict(pool, prediction_type="Probability")
        return np.asarray(proba)[:, 1] if proba.ndim == 2 else np.asarray(proba)
    elif model_key == "lgbm":
        return model.predict_proba(X)[:, 1]
    elif model_key == "tab":
        return model.predict_proba(X.values)[:, 1]
    return np.zeros(len(X))

def run_stage2_predict_all_core(seed: int, df_all: pd.DataFrame):
    logging.info(f"--- Starting Stage 2 Prediction for ALL Data (Seed: {seed}) ---")
    output_dir = OUTPUT_DIR
    model_file = MODELS_PATH / f"stage2_models_{seed}.joblib"
    predict_output_path = output_dir / f"stage2_predictions_all_{seed}.csv"

    if not model_file.exists():
        logging.error(f"❌ Model file not found for seed {seed}: {model_file}. Skipping prediction.")
        return
    if predict_output_path.exists() and SKIP_IF_EXISTS:
        logging.info(f"✅ Prediction file for seed {seed} already exists. Skipping prediction.")
        return

    try:
        with SectionTimer("Loading Global Artifacts and Model"):
            enc = joblib.load(ENCODER_PATH)
            num_cols, med = joblib.load(IMPUTER_PATH)
            whale_cut = json.load(open(WHALE_CUT_PATH))["whale_cut"]
            models: Dict[str, Any] = joblib.load(model_file)

            preds_file_val = output_dir / f"stage2_predictions_val_{seed}.csv"
            if preds_file_val.exists():
                df_val_preds = pd.read_csv(preds_file_val)
                ensemble_strategy = df_val_preds["strategy"].iloc[0] if "strategy" in df_val_preds.columns else "prior"
            else:
                ensemble_strategy = "prior"
            logging.info(f"Loaded ensemble strategy: {ensemble_strategy}")

        with SectionTimer("Data Transformation"):
            y_true_all = (df_all[TARGET_COL] >= whale_cut).astype(int)
            df_all_ = df_all.copy()
            df_all_.drop(columns=["stage2_tvt", "stage1_tvt"], errors="ignore", inplace=True)
            drop_cols_for_feature = feature_drop_cols()
            X_all_raw, feat_cols, cat_cols = build_features(df_all_, TARGET_COL, drop_cols_for_feature)
            X_all = enc.transform(X_all_raw)
            cat_cols_idx = [X_all.columns.get_loc(c) for c in cat_cols if c in X_all.columns]
            X_all = apply_imputer(X_all, num_cols, med)
            X_all = _sanitize_cols(X_all)
            X_all.index = df_all.index

        with SectionTimer("Generating Predictions and Ensemble"):
            tvt_col = "stage2_tvt" if "stage2_tvt" in df_all.columns else ("stage1_tvt" if "stage1_tvt" in df_all.columns else None)
            pred_data = {
                ID_COL: X_all.index.values, "seed": seed, "strategy": ensemble_strategy, "whale_cutoff": whale_cut,
            }
            if tvt_col: pred_data[tvt_col] = df_all[tvt_col]

            all_preds = {}
            for model_key, model in models.items():
                try:
                    p = _predict_with_model(model_key, model, X_all, cat_cols_idx)
                    all_preds[model_key] = p
                    pred_data[f"p_{model_key}"] = p
                except Exception as e:
                    logging.warning(f"⚠️ Prediction failed for model {model_key}: {e}")

            if not all_preds:
                raise RuntimeError("❌ No models successfully generated predictions.")

            cut_file = ARTIFACTS_PATH / f"stage2_cutoffs_{seed}.json"
            per_model_cuts = {}
            if cut_file.exists():
                saved = json.load(open(cut_file))
                per_model_cuts = saved.get("per_model", {})

            proba_ref, yhat_final = ensemble(all_preds, mode="hard", cutoffs=per_model_cuts, weights=None)
            pred_data["proba_ensemble"] = proba_ref
            pred_data["IS_WHALE_true"] = y_true_all.values
            pred_data["yhat_stage2"] = yhat_final

            pred_df = pd.DataFrame({k: v for k, v in pred_data.items() if v is not None and v is not False})
            pred_df.to_csv(predict_output_path, index=False)
            logging.info(f"✅ Saved all data predictions for seed {seed} at {predict_output_path}")

    except Exception as e:
        logging.error(f"❌ Critical error in predict_all for seed {seed}: {e}", exc_info=True)

# -----------------------------
# Predict TEST
# -----------------------------
def run_stage2_predict_test_core(seed: int, df_test: pd.DataFrame):
    logging.info(f"--- Starting Stage 2 Prediction for TEST Data (Seed: {seed}) ---")

    output_dir = OUTPUT_DIR
    predict_output_path_test = output_dir / f"stage2_predictions_test_{seed}.csv"
    model_file = MODELS_PATH / f"stage2_models_{seed}.joblib"
    metrics_test_file = output_dir / f"stage2_metrics_test_{seed}.csv"
    all_preds_file = output_dir / f"stage2_predictions_all_{seed}.csv"

    if len(df_test) == 0:
        logging.warning("⚠️ Skipping TEST prediction as the test set is empty.")
        return
    if not model_file.exists():
        logging.error(f"❌ Model file not found for seed {seed}: {model_file}. Skipping TEST prediction.")
        return
    if predict_output_path_test.exists() and SKIP_IF_EXISTS:
        logging.info(f"✅ TEST Prediction file for seed {seed} already exists. Skipping prediction.")
        return

    try:
        with SectionTimer("Loading Artifacts for Test Prediction"):
            enc = joblib.load(ENCODER_PATH)
            num_cols, med = joblib.load(IMPUTER_PATH)
            whale_cut = json.load(open(WHALE_CUT_PATH))["whale_cut"]
            models: Dict[str, Any] = joblib.load(model_file)

            preds_file_val = output_dir / f"stage2_predictions_val_{seed}.csv"
            ensemble_strategy = "prior"
            if preds_file_val.exists():
                ensemble_strategy = pd.read_csv(preds_file_val)["strategy"].iloc[0]

        with SectionTimer("Test Data Transformation"):
            y_true_test = (df_test[TARGET_COL] >= whale_cut).astype(int)
            df_test_ = df_test.copy()
            df_test_.drop(columns=["stage2_tvt", "stage1_tvt"], errors="ignore", inplace=True)
            drop_cols_for_feature = feature_drop_cols()
            X_test_raw, feat_cols, cat_cols = build_features(df_test_, TARGET_COL, drop_cols_for_feature)
            X_test = enc.transform(X_test_raw)
            cat_cols_idx = [X_test.columns.get_loc(c) for c in cat_cols if c in X_test.columns]
            X_test = apply_imputer(X_test, num_cols, med)
            X_test = _sanitize_cols(X_test)
            X_test.index = df_test.index

        with SectionTimer("Generating Test Predictions"):
            tvt_col = "stage2_tvt" if "stage2_tvt" in df_test.columns else ("stage1_tvt" if "stage1_tvt" in df_test.columns else None)
            pred_data = {
                ID_COL: X_test.index.values, "seed": seed, "strategy": ensemble_strategy, "whale_cutoff": whale_cut,
            }
            if tvt_col: pred_data[tvt_col] = df_test[tvt_col]

            all_preds = {}
            for model_key, model in models.items():
                try:
                    p = _predict_with_model(model_key, model, X_test, cat_cols_idx)
                    all_preds[model_key] = p
                    pred_data[f"p_{model_key}"] = p
                except Exception as e:
                    logging.warning(f"⚠️ TEST Prediction failed for model {model_key}: {e}")

            if not all_preds:
                raise RuntimeError("❌ No models successfully generated TEST predictions.")

            cut_file = ARTIFACTS_PATH / f"stage2_cutoffs_{seed}.json"
            per_model_cuts = {}
            if cut_file.exists():
                saved = json.load(open(cut_file))
                per_model_cuts = saved.get("per_model", {})

            proba_ref, yhat_final = ensemble(all_preds, mode="hard", cutoffs=per_model_cuts, weights=None)
            pred_data["proba_ensemble"] = proba_ref
            pred_data["IS_WHALE_true"] = y_true_test.values
            pred_data["yhat_stage2"] = yhat_final

            prec_test = precision_score(y_true_test, yhat_final, zero_division=0)
            rec_test = recall_score(y_true_test, yhat_final, zero_division=0)
            f1_test = f1_score(y_true_test, yhat_final, zero_division=0)
            fbeta_test = fbeta_score(y_true_test, yhat_final, beta=F_BETA, zero_division=0)
            ap_test = average_precision_score(y_true_test, proba_ref)

            metrics_test_df = pd.DataFrame({
                "model": ["ensemble"],
                "precision": [prec_test],
                "recall": [rec_test],
                "f1": [f1_test],
                f"f{F_BETA}": [fbeta_test],
                "ap": [ap_test],
                "threshold": [0.5],
            })
            metrics_test_df.to_csv(metrics_test_file, index=False)

            pred_df = pd.DataFrame({k: v for k, v in pred_data.items() if v is not None and v is not False})
            pred_df.to_csv(predict_output_path_test, index=False)
            logging.info(f"✅ Saved TEST data predictions for seed {seed} at {predict_output_path_test}")
            logging.info(f"✅ Saved TEST metrics for seed {seed} at {metrics_test_file}")

    except Exception as e:
        logging.error(f"❌ Critical error in predict_test for seed {seed}: {e}", exc_info=True)

# -----------------------------
# Multi-seed ensemble
# -----------------------------
def load_seed_preds(output_dir, seed, scope="all"):
    p = output_dir / f"stage2_predictions_{scope}_{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing predictions for seed={seed}, scope={scope}: {p}")
    # yhat_stage2 컬럼을 추가로 로드
    df = pd.read_csv(p)
    df["seed"] = seed
    return df

def load_seed_ap_weight(output_dir, seed):
    m = output_dir / f"stage2_metrics_{seed}.csv"
    if not m.exists():
        return 1.0
    df = pd.read_csv(m)
    if "model" in df.columns and "ap" in df.columns:
        if (df["model"] == "ensemble").any():
            return float(df.loc[df["model"] == "ensemble", "ap"].values[0])
        else:
            return float(df["ap"].mean())
    return 1.0

def calculate_ensemble_metrics(agg_df: pd.DataFrame, scope: str):
    y_true = agg_df["IS_WHALE_true"].values
    proba = agg_df["proba_seed_ensemble"].values
    thr = 0.5

    if y_true is not None and y_true.sum() > 0:
        p_grid, r_grid, thr_grid = precision_recall_curve(y_true, proba)
        best_fbeta = -1.0; best_thr = 0.5
        b2 = F_BETA * F_BETA
        full_thr_grid = np.unique(np.concatenate(([0.0], thr_grid, [1.0])))

        for t in full_thr_grid:
            y_hat = (proba >= t).astype(int)
            prec = precision_score(y_true, y_hat, zero_division=0)
            rec = recall_score(y_true, y_hat, zero_division=0)
            if prec < MIN_PREC_AT_CUT: 
                continue
            f_b = 0.0 if (prec == 0 and rec == 0) else (1 + b2) * (prec * rec) / (b2 * prec + rec + 1e-12)
            if f_b > best_fbeta:
                best_fbeta, best_thr = f_b, float(t)

        thr = best_thr
        yhat = (proba >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary")
        fbeta = fbeta_score(y_true, yhat, beta=F_BETA)
        ap = average_precision_score(y_true, proba)
        logging.info(f"[{scope.upper()} ENSEMBLE] P={prec:.4f} R={rec:.4f} F{F_BETA}={fbeta:.4f} AP={ap:.4f} thr={thr:.4f}")
        report = {
            "scope": scope, "beta": F_BETA, "min_precision": MIN_PREC_AT_CUT,
            "threshold": float(thr), "precision": float(prec), "recall": float(rec),
            "f1": float(f1), f"f{F_BETA}": float(fbeta), "ap": float(ap),
        }
    else:
        yhat = (proba >= thr).astype(int)
        report = {"scope": scope, "beta": F_BETA, "min_precision": MIN_PREC_AT_CUT,
                  "threshold": float(thr), "precision": 0.0, "recall": 0.0, "f1": 0.0, f"f{F_BETA}": 0.0, "ap": 0.0}

    agg_df["pred_seed_ensemble"] = yhat
    return report, thr

def run_stage2_multi_core():
    seeds = parse_seeds(DEFAULT_SEEDS_STR)
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    try:
        df_train_val, df_test, df_all = load_and_split_data()
    except Exception as e:
        logging.error(f"❌ Initial Data Loading failed: {e}")
        sys.exit(1)

    ensemble_mode = ENSEMBLE_MODE      # "weighted_by_ap" | "mean" | "hard"
    skip_if_exists = SKIP_IF_EXISTS

    FINAL_PARQUET_PATH = output_dir / "stage2_final_predictions_all_data.parquet"
    if skip_if_exists and FINAL_PARQUET_PATH.exists():
        logging.info(f"✅ Final ensemble prediction file exists at {FINAL_PARQUET_PATH}. Skipping entire stage.")
        sys.exit(0)

    if not skip_if_exists or any(not (output_dir / f"stage2_predictions_all_{s}.csv").exists() for s in seeds):
        for s in seeds:
            pred_file_all = output_dir / f"stage2_predictions_all_{s}.csv"
            pred_file_test = output_dir / f"stage2_predictions_test_{s}.csv"
            if skip_if_exists and pred_file_all.exists() and pred_file_test.exists():
                logging.info(f"[SKIP] Seed {s} ALL and TEST predictions exist.")
                continue
            try:
                run_stage2_train_core(s, df_train_val)
                run_stage2_predict_all_core(s, df_all)
                run_stage2_predict_test_core(s, df_test)
            except Exception as e:
                logging.warning(f"Seed {s} run failed: {e}")

    # Load per-seed ALL predictions + optional AP weights
    with SectionTimer("Loading Predictions and Calculating Weights"):
        all_pred_dfs = []
        weights = []
        seeds_used = []
        for s in seeds:
            try:
                # yhat_stage2 (개별 시드의 최종 예측 레이블)을 포함하여 로드
                df = load_seed_preds(output_dir, s, scope="all")
                all_pred_dfs.append(df)
                if ensemble_mode in ("weighted_by_ap", "soft"):
                    w = load_seed_ap_weight(output_dir, s)
                else:
                    w = 1.0
                weights.append(w)
                seeds_used.append(s)
            except FileNotFoundError:
                logging.warning(f"Skipping seed {s} due to missing ALL prediction file (CSV).")
                continue

        if not all_pred_dfs:
            logging.error("❌ No valid ALL prediction files found for ensembling.")
            sys.exit(1)

        import numpy as _np
        weights = _np.array(weights, dtype=float)
        if weights.sum() <= 0: weights = _np.ones_like(weights)
        weights = weights / weights.sum()
       
        yhat_seed_dfs = []
        for df, s in zip(all_pred_dfs, seeds_used):
            df_yhat = df[[ID_COL, "yhat_stage2"]].copy()
            df_yhat = df_yhat.rename(columns={"yhat_stage2": f"pred_is_high_payer_{s}"})
            df_yhat_seed_indexed = df_yhat.set_index(ID_COL)
            yhat_seed_dfs.append(df_yhat_seed_indexed)

    # Aggregate across seeds
    with SectionTimer("Aggregating Ensemble Probabilities (ALL Data)"):
        # 최종 앙상블 확률 계산을 위해 proba_ensemble 컬럼만 사용
        all_df_proba = pd.concat([df[[ID_COL, "proba_ensemble", "seed", "IS_WHALE_true", "stage2_tvt", "stage1_tvt"]].copy() 
                                  for df in all_pred_dfs], ignore_index=True)

        id_col_local = ID_COL
        seed_map = {int(s): w for s, w in zip(seeds_used, weights)}
        tvt_col_any = "stage2_tvt" if "stage2_tvt" in all_df_proba.columns else ("stage1_tvt" if "stage1_tvt" in all_df_proba.columns else None)

        def agg_soft(g):
            probs = g["proba_ensemble"].values
            seeds_order = g["seed"].values
            w = np.array([seed_map.get(int(sd), 0.0) for sd in seeds_order], dtype=float)
            p = float(np.dot(probs, w / (w.sum() if w.sum() > 0 else 1.0)))
            y = int(g["IS_WHALE_true"].iloc[0]) if "IS_WHALE_true" in g.columns else 0
            tvt_value = g[tvt_col_any].iloc[0] if (tvt_col_any and tvt_col_any in g.columns) else "unknown"
            return pd.Series({"proba_seed_ensemble": p, "IS_WHALE_true": y, "tvt": tvt_value})

        def agg_hard(g):
            probs = g["proba_ensemble"].values
            seeds_order = g["seed"].values
            cuts = []
            for sd in seeds_order:
                cut_file = ARTIFACTS_PATH / f"stage2_cutoffs_{int(sd)}.json"
                if cut_file.exists():
                    saved = json.load(open(cut_file))
                    cuts.append(float(saved.get("ensemble", 0.5)))
                else:
                    cuts.append(0.5)
            w = np.array([seed_map.get(int(sd), 0.0) for sd in seeds_order], dtype=float)
            # Hard voting at seed level
            proba_ref, yhat = ensemble(probs.reshape(1, -1), mode="hard", cutoffs=np.asarray(cuts), weights=w)
            p_mean = float(np.mean(probs))
            y = int(g["IS_WHALE_true"].iloc[0]) if "IS_WHALE_true" in g.columns else 0
            tvt_value = g[tvt_col_any].iloc[0] if (tvt_col_any and tvt_col_any in g.columns) else "unknown"
            return pd.Series({"proba_seed_ensemble": p_mean, "IS_WHALE_true": y, "tvt": tvt_value, "yhat_seed_vote": int(yhat[0])})

        if ensemble_mode in ("weighted_by_ap", "mean", "soft"):
            agg_all = all_df_proba.groupby(id_col_local, as_index=False).apply(agg_soft).reset_index(drop=True)
        else:
            agg_all = all_df_proba.groupby(id_col_local, as_index=False).apply(agg_hard).reset_index(drop=True)
        
        agg_all = agg_all.set_index(ID_COL)
        for df_yhat_seed in yhat_seed_dfs:
             # ID_COL을 인덱스로 사용하여 join
             agg_all = agg_all.join(df_yhat_seed, how="left")
        agg_all = agg_all.reset_index()

 
    with SectionTimer("Final Threshold Tuning (ALL Data)"):
        if "yhat_seed_vote" in agg_all.columns:
            y_true = agg_all["IS_WHALE_true"].values
            yhat = agg_all["yhat_seed_vote"].values
            proba = agg_all["proba_seed_ensemble"].values
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary")
            fbeta = fbeta_score(y_true, yhat, beta=F_BETA)
            ap = average_precision_score(y_true, proba)
            logging.info(f"[ALL ENSEMBLE (HARD)] P={prec:.4f} R={rec:.4f} F{F_BETA}={fbeta:.4f} AP={ap:.4f}")
            report_all = {
                "scope": "ALL", "beta": F_BETA, "mode": "hard",
                "threshold": None, "precision": float(prec), "recall": float(rec),
                "f1": float(f1), f"f{F_BETA}": float(fbeta), "ap": float(ap),
            }
            final_thr_all = None
            agg_all["pred_seed_ensemble"] = yhat
        else:
            report_all, final_thr_all = calculate_ensemble_metrics(agg_all, "ALL")

        outp_all = output_dir / "stage2_seed_ensemble_all.csv"
        agg_all.to_csv(outp_all, index=False)
        logging.info(f"[OK] Saved seed-ensemble ALL predictions (CSV): {outp_all}")

    with SectionTimer("Final Threshold Tuning (TEST Data Only)"):
        agg_test = agg_all[agg_all["tvt"] == "test"].copy()
        if len(agg_test) == 0:
            logging.warning("⚠️ Skipping TEST Ensemble Metrics: Test set is empty.")
            report_test = {"scope": "TEST", "fbeta": 0.0, "ap": 0.0}
        else:
            y_true_test = agg_test["IS_WHALE_true"].values
            proba_test = agg_test["proba_seed_ensemble"].values
            if "yhat_seed_vote" in agg_test.columns:
                yhat_test = agg_test["yhat_seed_vote"].values
            else:
                thr = final_thr_all if final_thr_all is not None else 0.5
                yhat_test = (proba_test >= thr).astype(int)

            prec_test = precision_score(y_true_test, yhat_test, zero_division=0)
            rec_test = recall_score(y_true_test, yhat_test, zero_division=0)
            fbeta_test = fbeta_score(y_true_test, yhat_test, beta=F_BETA, zero_division=0)
            ap_test = average_precision_score(y_true_test, proba_test)
            report_test = {
                "scope": "TEST", "beta": F_BETA, "min_precision": MIN_PREC_AT_CUT,
                "threshold": float(final_thr_all) if final_thr_all is not None else None,
                "precision": float(prec_test), "recall": float(rec_test),
                "f1": float(f1_score(y_true_test, yhat_test, zero_division=0)),
                f"f{F_BETA}": float(fbeta_test), "ap": float(ap_test),
            }

        with SectionTimer("Saving Final Ensemble Parquet File"):
            parquet_cols = ["proba_seed_ensemble", "pred_seed_ensemble"]
            for s in seeds_used:
                parquet_cols.append(f"pred_is_high_payer_{s}")
            
            agg_all_indexed = agg_all.set_index(ID_COL).rename(columns={
                "proba_seed_ensemble": "stage2_proba",
                "pred_seed_ensemble": "pred_is_high_payer"
            })[parquet_cols]
            
            rename_map = {
                "stage2_proba": "stage2_proba",
                "pred_is_high_payer": "pred_is_high_payer",
            }
            for s in seeds_used:
                rename_map[f"pred_is_high_payer_{s}"] = f"stage2_is_whale_{s}"
            
            agg_all_indexed = agg_all_indexed.rename(columns=rename_map)

            data_dir = DEFAULT_INPUT_DATA_PATH.parent
            base_data_path = data_dir / "stage1_final_predictions_all_data.parquet"
            df_base_full = pd.read_parquet(base_data_path)
            if ID_COL in df_base_full.columns:
                df_base_full = df_base_full.set_index(ID_COL, drop=False)

            final_output_df = df_base_full.copy()
            final_output_df = final_output_df.join(agg_all_indexed, how='left')

            FINAL_PARQUET_PATH = output_dir / "stage2_final_predictions_all_data.parquet"
            final_output_df.to_parquet(FINAL_PARQUET_PATH, index=False, engine='pyarrow')
            logging.info(f"✅ Saved Final Ensemble Predictions to: {FINAL_PARQUET_PATH}")

        final_report_path = output_dir / "stage2_seed_ensemble_report.json"
        report_final = report_all
        report_final["test_metrics"] = report_test

        with open(final_report_path, "w", encoding="utf-8") as f:
            json.dump(report_final, f, ensure_ascii=False, indent=2)
        logging.info(f"[OK] Saved seed-ensemble report (json): {final_report_path}")

    logging.info("✅ Multi-Seed Ensemble Pipeline Complete.")
    sys.exit(0)

# -----------------------------
# Main entry
# -----------------------------
def main_cli_entry():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_PATH / f"ltv_pipeline_{timestamp}_overall.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    )
    try:
        create_stage1_data_parquet()
        run_stage2_multi_core()
    except SystemExit:
        pass
    except Exception as e:
        logging.error(f"☠️ A critical error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main_cli_entry()
