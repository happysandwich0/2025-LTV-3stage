"""
Stage 2 Model Tuning, Training, and Diagnostic Plotting.
"""
import logging
import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import average_precision_score, precision_recall_curve

# Import necessary configuration variables and utilities
from config import SEED, OPTUNA_SEED, CAT_TASK_PARAMS, F_BETA, REWEIGHT_BY_FBETA
from utils import score_stage2_objective, optuna_progress_cb_strict


# =============================================================================
# LightGBM Tuning
# =============================================================================
def tune_lgbm_cls(
    X_tr,
    y_tr,
    X_va,
    y_va,
    stage_key,
    strategy,
    train_pos_prior,
    trials,
    current_seed
    ):
    """
    LightGBM Binary Classifier Tuning for Stage 2.
    """

    bounds = dict(
        n_estimators=(300, 900),
        learning_rate=(0.01, 0.08),
        num_leaves=(16, 96),
        max_depth=(3, 9),
        min_child_samples=(20, 200),
        subsample=(0.6, 1.0),
        colsample_bytree=(0.6, 1.0),
        reg_alpha=(0.0, 10.0),
        reg_lambda=(0.0, 20.0),
    )

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", *bounds["n_estimators"]),
            learning_rate=trial.suggest_float(
                "learning_rate", *bounds["learning_rate"], log=True
            ),
            num_leaves=trial.suggest_int("num_leaves", *bounds["num_leaves"]),
            max_depth=trial.suggest_int("max_depth", *bounds["max_depth"]),
            min_child_samples=trial.suggest_int(
                "min_child_samples", *bounds["min_child_samples"]
            ),
            subsample=trial.suggest_float(
                "subsample", *bounds["subsample"]
            ),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", *bounds["colsample_bytree"]
            ),
            reg_alpha=trial.suggest_float(
                "reg_alpha", *bounds["reg_alpha"]
            ),
            reg_lambda=trial.suggest_float(
                "reg_lambda", *bounds["reg_lambda"]
            ),
            objective="binary",
            random_state=current_seed,
            n_jobs=max(1, (os.cpu_count() or 8) // 4),
            verbosity=-1,
            force_row_wise=True,
        )

        if strategy == "reweight":
            w_neg = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
            if REWEIGHT_BY_FBETA:
                w_neg *= (F_BETA ** 2)
            params["class_weight"] = {0: 1.0, 1: w_neg}

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                LightGBMPruningCallback(trial, "average_precision"),
            ],
        )

        proba = model.predict_proba(X_va)[:, 1]
        return score_stage2_objective(y_va, proba)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=current_seed), 
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study.optimize(
        objective,
        n_trials=trials,
        callbacks=[optuna_progress_cb_strict(stage_key)],
    )

    best = study.best_params
    best.update(
        dict(
            objective="binary",
            random_state=current_seed,
            n_jobs=max(1, (os.cpu_count() or 8) // 4),
            verbosity=-1,
            force_row_wise=True,
        )
    )
    if strategy == "reweight":
        w_pos = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        if REWEIGHT_BY_FBETA:
            w_pos *= (F_BETA ** 2)
        w_pos = float(min(w_pos, 20.0))
        best["class_weight"] = {0: 1.0, 1: w_pos}

    logging.info(f"[LGBM][{strategy}] Best params: {best}")
    return best

def tune_cat_cls(
    X_tr,
    y_tr,
    X_va,
    y_va,
    cat_cols_idx,
    stage_key,
    strategy,
    train_pos_prior,
    trials: int,
    current_seed,
):
    """
    CatBoost Tuning for Stage 2.

    - iterations: 300–900
    - depth: 4–8
    - learning_rate: 0.02–0.12
    """

    bounds = dict(
        iterations=(300, 900),
        depth=(4, 8),
    )

    def objective(trial):
        params = dict(
            depth=trial.suggest_int("depth", *bounds["depth"]),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.02, 0.12, log=True
            ),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 2.0, 12.0),
            bagging_temperature=trial.suggest_float(
                "bagging_temperature", 0.0, 3.0
            ),
            random_strength=trial.suggest_float(
                "random_strength", 0.0, 2.0
            ),
            iterations=trial.suggest_int(
                "iterations", *bounds["iterations"]
            ),
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=current_seed, 
            verbose=0,
        )

        if strategy == "reweight":
            w_neg = 1.0
            w_pos = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
            if REWEIGHT_BY_FBETA:
                w_pos *= (F_BETA ** 2)
            params["class_weights"] = [w_neg, w_pos]

        model = CatBoostClassifier(
            **params,
            **CAT_TASK_PARAMS,
            od_type="Iter",
            od_wait=200,
        )

        pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols_idx or None)
        pool_va = Pool(X_va, y_va, cat_features=cat_cols_idx or None)

        model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
        proba = model.predict_proba(pool_va)[:, 1]
        return score_stage2_objective(y_va, proba)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=current_seed), 
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=trials)

    best = study.best_params
    best.update(
        dict(
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=current_seed,
            verbose=0,
        )
    )

    if strategy == "reweight":
        w_neg = 1.0
        w_pos = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        if REWEIGHT_BY_FBETA:
            w_pos *= (F_BETA ** 2)
        best["class_weights"] = [w_neg, w_pos]

    logging.info(f"[CAT][{strategy}] Best params: {best}")
    return best


def plot_lgbm_error_trajectory(
    Xtr,
    y_tr,
    Xva,
    y_va,
    best_lgb_params,
    output_dir,
    current_seed,
):
    if Xva.shape[1] == 0:
        logging.warning("[PLOT] Xva is empty, skipping LGBM plot.")
        return

    params = best_lgb_params.copy()
    n_estimators_big = params.pop("n_estimators", 900)
    params["learning_rate"] = float(params.get("learning_rate", 0.05))
    params["random_state"] = current_seed 

    model = lgb.LGBMClassifier(**params, n_estimators=n_estimators_big)

    model.fit(
        Xtr,
        y_tr,
        eval_set=[(Xva, y_va)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(0)],
    )

    iters = np.arange(1, n_estimators_big + 1)
    ap_list = []

    if hasattr(model, "booster_") and model.booster_ is not None:
        num_trees = model.booster_.num_trees()
        for i in iters:
            if i > num_trees:
                break
            try:
                proba = model.predict_proba(Xva, num_iteration=i)[:, 1]
                ap_list.append(average_precision_score(y_va, proba))
            except Exception as e:
                logging.warning(
                    f"[PLOT] LGBM prediction failed at iter {i}: {e}. "
                    "Stopping trajectory collection."
                )
                break
    else:
        logging.warning(
            "[PLOT] LGBM Booster not found after training, skipping trajectory."
        )

    ap_arr = np.array(ap_list, dtype=float)
    if len(ap_arr) == 0:
        logging.warning(
            "[PLOT] LGBM: No AP values collected. Skipping plot save."
        )
        return

    val_error = 1.0 - ap_arr

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(iters[: len(val_error)], val_error, linewidth=1.5)

    best_idx = int(np.argmin(val_error))
    ax.axvline(
        x=best_idx + 1,
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"Min Error ({best_idx + 1})",
    )
    logging.info(
        f" [LGBM Plot] Min Error: iter={best_idx + 1:,} | "
        f"AP={ap_arr[best_idx]:.6f}"
    )

    title_str = (
        f"LGBM PR-AUC Error (Seed {current_seed}) | "
        f"MaxD={params.get('max_depth', 'NA')}, "
        f"LR={params['learning_rate']:.3f}"
    )
    ax.set_title(title_str)
    ax.set_xlabel("n_estimators (boosting rounds)")
    ax.set_ylabel("Validation Error (1 - AP)")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / f"plots/lgbm_error_trajectory_{current_seed}.png"
    plot_path.parent.mkdir(exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def plot_cat_error_trajectory(
    Xtr: pd.DataFrame,
    y_tr: pd.Series,
    Xva: pd.DataFrame,
    y_va: pd.Series,
    cat_cols_idx: List[int],
    best_cat_params: dict,
    output_dir: Path,
    current_seed: int,
):
    if Xva.shape[1] == 0:
        logging.warning("[PLOT] Xva is empty, skipping CatBoost plot.")
        return

    pool_tr = Pool(Xtr, y_tr, cat_features=cat_cols_idx or None)
    pool_va = Pool(Xva, y_va, cat_features=cat_cols_idx or None)

    params = best_cat_params.copy()
    n_estimators_big = params.pop("iterations", 900)
    params.pop("verbose", None)
    params["learning_rate"] = float(params.get("learning_rate", 0.05))
    params["random_seed"] = current_seed 

    model = CatBoostClassifier(
        **params,
        iterations=n_estimators_big,
        **CAT_TASK_PARAMS,
    )

    model.fit(pool_tr, eval_set=pool_va, use_best_model=False, verbose=False)

    iters = np.arange(1, n_estimators_big + 1)
    ap_list = []

    num_trees_learned = model.tree_count_
    for i in iters:
        if i > num_trees_learned:
            break
        try:
            proba = model.predict(
                pool_va,
                prediction_type="Probability",
                ntree_end=i,
            )
            proba = np.asarray(proba)

            if proba.ndim == 2 and proba.shape[1] == 2:
                proba = proba[:, 1]
            elif proba.ndim == 1:
                pass
            else:
                continue

            if proba.shape[0] != len(y_va):
                continue

            ap_list.append(average_precision_score(y_va, proba))
        except Exception as e:
            logging.warning(
                f"[PLOT] CatBoost prediction failed at iter {i}: {e}. "
                "Stopping trajectory collection."
            )
            break

    ap_arr = np.array(ap_list, dtype=float)
    if len(ap_arr) == 0:
        logging.warning(
            "[PLOT] CatBoost: No AP values collected. Skipping plot save."
        )
        return

    val_error = 1.0 - ap_arr

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(iters[: len(val_error)], val_error, linewidth=1.5)

    best_idx = int(np.argmin(val_error))
    ax.axvline(
        x=best_idx + 1,
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"Min Error ({best_idx + 1})",
    )
    logging.info(
        f" [CAT Plot] Min Error: iter={best_idx + 1:,} | "
        f"AP={ap_arr[best_idx]:.6f}"
    )

    title_str = (
        f"CatBoost PR-AUC Error (Seed {current_seed}) | "
        f"Depth={params.get('depth', 'NA')}, "
        f"LR={params['learning_rate']:.3f}"
    )
    ax.set_title(title_str)
    ax.set_xlabel("iterations")
    ax.set_ylabel("Validation Error (1 - AP)")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / f"plots/cat_error_trajectory_{current_seed}.png"
    plot_path.parent.mkdir(exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)

def best_threshold_for_fbeta(y_true, proba, beta: float = 1.0):
    p, r, t = precision_recall_curve(y_true, proba)
    f = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    i = int(np.nanargmax(f))
    best_thr = float(t[max(i-1, 0)])
    return best_thr, float(f[i])

def fit_and_get_threshold_lgbm(X_tr, y_tr, X_va, y_va, best_params: dict, beta: float):
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr)
    proba_va = model.predict_proba(X_va)[:, 1]
    thr, fbest = best_threshold_for_fbeta(y_va, proba_va, beta)
    return model, thr, fbest, proba_va

def fit_and_get_threshold_cat(X_tr, y_tr, X_va, y_va, best_params: dict,
                              cat_cols_idx, beta: float):
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols_idx or None)
    pool_va = Pool(X_va, y_va, cat_features=cat_cols_idx or None)
    model = CatBoostClassifier(**best_params, **CAT_TASK_PARAMS)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
    proba_va = model.predict_proba(pool_va)[:, 1]
    thr, fbest = best_threshold_for_fbeta(y_va, proba_va, beta)
    return model, thr, fbest, proba_va
