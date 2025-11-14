# models_stage3.py

"""
Stage 3 Model Tuning, Training, and Diagnostic Plotting for Regression.
"""
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
import random 

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import necessary configuration variables and utilities
from config import (
    CAT_TASK_PARAMS, LGBM_LOSS, CAT_LOSS,
    TUNE_NON_WHALE, TUNE_WHALE, USE_LOG_TRANSFORM, LOG_EPSILON, SMAPE_EPSILON, 
    NO_LGBM, NO_CATBOOST, NO_TABPFN
)
from utils import optuna_progress_cb_strict

# === Metric Helper Functions ===

def inverse_transform(p):
    """로그 변환된 예측값을 원래 스케일로 되돌립니다."""
    if USE_LOG_TRANSFORM:
        return np.maximum(0.0, np.expm1(p) + LOG_EPSILON) 
    return np.maximum(0.0, p)

def score_stage3_objective(y_true_raw, proba_log):
    """Stage 3 튜닝 목적 함수: 역변환된 예측값으로 MAE(Mean Absolute Error)를 최소화합니다."""
    try:
        y_pred_raw = inverse_transform(proba_log)
        return -float(mean_absolute_error(y_true_raw, y_pred_raw)) 
    except ValueError:
        return 0.0 

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (sMAPE)를 계산합니다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + SMAPE_EPSILON
    
    return np.mean(numerator / denominator) * 100.0


# === Tuning Functions ===

def _get_tuning_bounds(group_key):
    """그룹 키에 따라 튜닝 범위를 반환합니다."""
    return TUNE_WHALE if group_key == "whale" else TUNE_NON_WHALE

def tune_lgbm_reg(X_tr, y_tr_log, X_va, y_va_log, y_va_raw, 
                  group_key, trials, seed, optuna_seed): # <-- seed, optuna_seed 인자 추가
    
    # 🚨 수정: np.random.seed() 호출만 유지.
    np.random.seed(seed) 

    LGBM_MAE_METRIC = "l1"
    
    bounds = _get_tuning_bounds(group_key)["lgbm"]
    
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", *bounds["n_estimators"]),
            learning_rate=trial.suggest_float("learning_rate", *bounds["learning_rate"], log=True),
            num_leaves=trial.suggest_int("num_leaves", *bounds["num_leaves"]),
            max_depth=trial.suggest_int("max_depth", *bounds["max_depth"]),
            min_child_samples=trial.suggest_int("min_child_samples", *bounds["min_child_samples"]),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", *bounds["reg_alpha"]),
            reg_lambda=trial.suggest_float("reg_lambda", *bounds["reg_lambda"]),
            objective=LGBM_LOSS, 
            metric=LGBM_MAE_METRIC,
            # 🚨 수정: n_jobs=1로 설정하고 seed 인자 사용
            random_state=seed, n_jobs=1, verbosity=-1, force_row_wise=True,
        )
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], eval_metric=LGBM_MAE_METRIC,
                  callbacks=[lgb.early_stopping(100, verbose=False), LightGBMPruningCallback(trial, LGBM_MAE_METRIC)]) 
        
        proba_log = model.predict(X_va)
        
        y_pred_raw = inverse_transform(proba_log)
        return float(mean_absolute_error(y_va_raw.values, y_pred_raw)) # Minimize MAE

    study = optuna.create_study(direction="minimize", 
                                # 🚨 수정: optuna_seed 인자 사용
                                sampler=TPESampler(seed=optuna_seed),
                                pruner=MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=trials, callbacks=[optuna_progress_cb_strict(f"s3-lgbm-reg-{group_key}")])
    
    # 🚨 수정: best_params에도 n_jobs=1 및 seed 반영
    best = study.best_params
    best.update(dict(objective=LGBM_LOSS, random_state=seed, n_jobs=1, verbosity=-1, force_row_wise=True))
    return best

def tune_cat_reg(X_tr, y_tr_log, X_va, y_va_log, y_va_raw, 
                 cat_cols_idx, group_key, trials, seed, optuna_seed): # <-- seed, optuna_seed 인자 추가
    
    # 🚨 수정: np.random.seed() 호출만 유지
    np.random.seed(seed)
    
    bounds = _get_tuning_bounds(group_key)["cat"]

    def objective(trial):
        params = dict(
            depth=trial.suggest_int("depth", *bounds["depth"]),
            learning_rate=trial.suggest_float("learning_rate", *bounds["learning_rate"], log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", *bounds["l2_leaf_reg"]),
            bagging_temperature=trial.suggest_float("bagging_temperature", *bounds["bagging_temperature"]),
            random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
            iterations=trial.suggest_int("iterations", *bounds["iterations"]),
            # 🚨 수정: seed 인자 사용
            loss_function=CAT_LOSS, eval_metric=CAT_LOSS, random_seed=seed, verbose=0,
        )

        model = CatBoostRegressor(**params, **CAT_TASK_PARAMS, od_type="Iter", od_wait=100)
        pool_tr = Pool(X_tr, y_tr_log, cat_features=cat_cols_idx or None)
        pool_va = Pool(X_va, y_va_log, cat_features=cat_cols_idx or None)

        model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
        proba_log = model.predict(pool_va)
        return score_stage3_objective(y_va_raw.values, proba_log) 

    study = optuna.create_study(direction="maximize", 
                                # 🚨 수정: optuna_seed 인자 사용
                                sampler=TPESampler(seed=optuna_seed),
                                pruner=MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=trials)
    
    # 🚨 수정: best_params에도 seed 반영
    best = study.best_params
    best.update(dict(loss_function=CAT_LOSS, eval_metric=CAT_LOSS, random_seed=seed, verbose=0))
        
    return best

# seed 인자 추가
def make_tabpfn_regressor(device, seed, n_ens, input_size = 100):
    """TabPFNClassifier 대신 TabPFNRegressor를 생성합니다."""
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        logging.warning("⚠️ TabPFNRegressor import failed. TabPFN model will be skipped.")
        return None
    try:
        from utils import construct_tabpfn
    except ImportError:
        def construct_tabpfn(cls, device, seed, n_ens):
            kw = {}
            if device in ["cuda", "auto"]: kw["device"] = "cuda" if device == "auto" else device
            kw["N_ensemble_configurations"] = n_ens
            kw["seed"] = seed
            return cls(**kw)

    return construct_tabpfn(TabPFNRegressor, device=device, seed=seed, n_ens=n_ens)

# seed 인자 추가
def train_and_ensemble_reg(X_tr, y_tr_log, X_va, y_va_log, y_va_raw, 
                           cat_cols_idx, group_key, trials, has_tabpfn, seed): # <-- seed 인자 추가
    """단일 시드, 단일 그룹(Whale/Non-Whale)에 대한 회귀 모델 튜닝 및 학습을 수행합니다."""
    
    models = {}; best_params = {}; preds_log = {}

    LGBM_MAE_METRIC = "l1" 
    
    optuna_seed = seed

    # 1. LightGBM
    if not NO_LGBM:
        # seed, optuna_seed 인자 전달
        lgb_params = tune_lgbm_reg(X_tr, y_tr_log, X_va, y_va_log, y_va_raw, group_key, trials, seed, optuna_seed)
        best_params["lgbm"] = lgb_params 
        
        lgbm_reg = lgb.LGBMRegressor(**lgb_params)
        lgbm_reg.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], eval_metric=LGBM_MAE_METRIC,
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        models["lgbm"] = lgbm_reg
        preds_log["lgbm"] = lgbm_reg.predict(X_va)

    # 2. CatBoost
    if not NO_CATBOOST:
        # seed, optuna_seed 인자 전달
        cat_params = tune_cat_reg(X_tr, y_tr_log, X_va, y_va_log, y_va_raw, cat_cols_idx, group_key, trials, seed, optuna_seed)
        best_params["cat"] = cat_params
        cat_reg = CatBoostRegressor(**cat_params, **CAT_TASK_PARAMS, od_type="Iter", od_wait=100)
        pool_tr = Pool(X_tr, y_tr_log, cat_features=cat_cols_idx or None)
        cat_reg.fit(pool_tr, eval_set=Pool(X_va, y_va_log, cat_features=cat_cols_idx or None), 
                    use_best_model=True, verbose=False)
        models["cat"] = cat_reg
        preds_log["cat"] = cat_reg.predict(Pool(X_va, cat_features=cat_cols_idx or None))

    # 3. TabPFN (No tuning)
    if has_tabpfn and not NO_TABPFN:
        from config import TABPFN_DEVICE, TABPFN_CONFIGS
        # seed 인자 사용
        tab_reg = make_tabpfn_regressor(device=TABPFN_DEVICE, seed=seed, n_ens=TABPFN_CONFIGS, input_size=X_tr.shape[1])
        if tab_reg is not None:
            tab_reg.fit(X_tr.values, y_tr_log.values) 
            models["tab"] = tab_reg
            preds_log["tab"] = tab_reg.predict(X_va.values)
            
    if not models:
        raise ValueError(f"❌ No models were trained for group {group_key}.")

    # 앙상블 (단순 평균)
    ensemble_log_proba = np.mean(np.column_stack(list(preds_log.values())), axis=1)
    
    # 앙상블 성능 평가 (MAE)
    ensemble_raw_proba = inverse_transform(ensemble_log_proba)
    mae_val = mean_absolute_error(y_va_raw.values, ensemble_raw_proba)
    
    logging.info(f"[VAL|{group_key.upper()}] Ensemble MAE={mae_val:.2f} | Models: {list(models.keys())}")
    
    return models, best_params, mae_val

# === Prediction Functions ===

def predict_reg_model(model_key, model, X, cat_cols_idx):
    """단일 회귀 모델로 예측값(로그 스케일)을 계산합니다."""
    
    if model_key == "cat":
        pool = Pool(X, cat_features=cat_cols_idx or None)
        return model.predict(pool)
        
    elif model_key == "lgbm":
        return model.predict(X)
        
    elif model_key == "tab":
        return model.predict(X.values)
        
    return np.zeros(len(X))