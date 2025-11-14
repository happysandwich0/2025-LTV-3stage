# main_stage3.py

"""
Stage 3 Hybrid Pipeline Core Logic: Final LTV Regression.
- Uses Stage 1 (Payer/Non-Payer) and Stage 2 (Whale/Non-Whale) predictions to segment the data.
- Trains separate Regressors for Non-Whale Payers and Whale Payers.
"""
import json
import sys
import os
import random
import logging
import joblib 
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config as config
from config import (
    TARGET_COL, ID_COL, PRED_PAYER_COL, PRED_WHALE_COL, 
    DEF_OUTPUT_DIR, BASE_SPLIT_SEED, 
    DEFAULT_TRIALS, DEFAULT_TEST_SIZE,
    USE_LOG_TRANSFORM, LOG_EPSILON,
    NO_CATBOOST, NO_LGBM, NO_TABPFN,
    ENSEMBLE_MODE, SKIP_IF_EXISTS, DEFAULT_SEEDS_STR, DEFAULT_INPUT_DATA_PATH
)
from config import SEED, OPTUNA_SEED # 기본값 로드를 위해 유지

from utils import SectionTimer, _sanitize_cols, parse_seeds 
from data_preprocessing import (
    OrdinalCategoryEncoder, build_features, fit_imputer, apply_imputer, _HAS_TABPFN 
)
from models import (
    train_and_ensemble_reg, predict_reg_model, inverse_transform, smape
)

# --- Global Artifact Paths (전역 경로 상수) ---
OUTPUT_DIR = Path(DEF_OUTPUT_DIR)
ARTIFACTS_PATH = OUTPUT_DIR / "global_artifacts"
ENCODER_PATH = ARTIFACTS_PATH / "stage3_encoder.joblib" # Stage 3 전용 인코더 경로
IMPUTER_PATH = ARTIFACTS_PATH / "stage3_imputer.joblib" # Stage 3 전용 Imputer 경로
FEAT_COLS_PATH = ARTIFACTS_PATH / "stage3_feat_cols.joblib" # 피처 목록 저장 경로
MODELS_PATH = OUTPUT_DIR / "models"
LOGS_PATH = OUTPUT_DIR / "logs"

# =====================================================================================
# ---- DATA LOADING CORE
# =====================================================================================

def create_stage2_data_parquet():
    """
    Stage 2 최종 예측 파일에 stage1_tvt 컬럼을 추가하고 새로운 parquet 파일을 생성합니다.
    (새로운 파일이 존재하지 않을 때만 실행)
    """
    # Stage 3의 최종 입력 파일 경로
    data_dir = DEFAULT_INPUT_DATA_PATH.parent 
    final_output_path = DEFAULT_INPUT_DATA_PATH 
    
    if final_output_path.exists():
        logging.info(f"✅ Initial Data Prep: {final_output_path.name} already exists. Skipping data creation.")
        return

    logging.info(f"▶ START: Creating initial {final_output_path.name}")
    
    try:
        # 1. Stage 2 최종 예측 파일 로드
        base_data_path = data_dir / "stage2_final_predictions_all_data.parquet"
        if not base_data_path.exists():
            raise FileNotFoundError(f"Base data not found: {base_data_path}")
            
        df_base = pd.read_parquet(base_data_path)
        
        # 2. TVT 정보 로드 및 병합 (로직 재사용)
        tvt_maps = []
        tvt_files = {
            "train": data_dir / "train_df_5days.parquet",
            "val": data_dir / "val_df_5days.parquet",
            "test": data_dir / "test_df_5days.parquet",
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
             
        # 3. 데이터 병합 (Stage 2 예측 파일 + TVT 정보)
        # ID_COL (PLAYERID)을 기준으로 병합
        df_final = pd.merge(df_base, df_tvt_map, on=ID_COL, how="left")
        
        # TVT 정보가 없는 행은 'unknown' 처리
        unknown_count = df_final['stage1_tvt'].isnull().sum() # <- unknown 개수 카운트
        df_final['stage1_tvt'] = df_final['stage1_tvt'].fillna('unknown')
        
        # 4. 저장
        df_final.to_parquet(final_output_path, index=False)
        logging.info(f"✅ Initial Data Prep: Successfully created {final_output_path.name} ({df_final.shape}). Unknown count: {unknown_count}.")
        
    except Exception as e:
        logging.error(f"❌ Initial Data Prep: Failed to create {final_output_path.name}. Error: {e}", exc_info=True)
        # 실패 시 에러를 다시 발생시켜 main_cli_entry에서 처리하도록 함 (파이프라인 진행 방지)
        raise RuntimeError(f"Data preparation failed: {e}")

def load_data_and_prepare_for_regression():
    """
    Stage 2 최종 예측 파일을 로드하고 회귀를 위한 데이터셋을 준비합니다.
    - BASE_SPLIT_SEED로 고정 분할된 데이터를 사용합니다.
    """
    logging.info(f"▶ START: Loading Stage 2 final predictions from {config.DEFAULT_INPUT_DATA_PATH.name}")
    
    try:
        df_full = pd.read_parquet(config.DEFAULT_INPUT_DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"❌ Stage 2 final predictions file not found: {config.DEFAULT_INPUT_DATA_PATH}. "
            f"Ensure Stage 2 was run successfully."
        )

    if ID_COL in df_full.columns:
        df_full = df_full.set_index(ID_COL, drop=False)

    if PRED_PAYER_COL not in df_full.columns or PRED_WHALE_COL not in df_full.columns:
        raise KeyError("❌ Missing required prediction columns (pred_is_payer or pred_is_high_payer).")

    if "stage1_tvt" not in df_full.columns:
        raise KeyError("❌ 'stage1_tvt' column not found.")
        
    # 1. 데이터 분할: Stage 1/2 예측을 기준으로 3개 그룹으로 분할
    df_non_payer = df_full[df_full[PRED_PAYER_COL] == 0].copy()
    
    # Stage 3 훈련 대상 데이터 (Stage 1 TVT 'train' + 'val' 그룹)
    df_train_val_scope = df_full[df_full["stage1_tvt"].isin(["train", "val"])].copy()
    
    # Non-Whale Payer 그룹 (Stage 3 Train/Val 범위)
    df_nw_tr_val_scope = df_train_val_scope[(df_train_val_scope[PRED_PAYER_COL] == 1) & (df_train_val_scope[PRED_WHALE_COL] == 0)].copy()
    # Whale Payer 그룹 (Stage 3 Train/Val 범위)
    df_w_tr_val_scope = df_train_val_scope[(df_train_val_scope[PRED_PAYER_COL] == 1) & (df_train_val_scope[PRED_WHALE_COL] == 1)].copy()

    # Stage 3 Test 세트 (Stage 1 TVT 'test' 그룹)
    df_test_scope = df_full[df_full["stage1_tvt"] == "test"].copy()
    
    # 2. Non-Whale Payer 그룹 (회귀 1): BASE_SPLIT_SEED로 Train/Val 분할
    tr_nw_full, va_nw_full = train_test_split(
        df_nw_tr_val_scope, test_size=DEFAULT_TEST_SIZE, random_state=BASE_SPLIT_SEED, shuffle=True,
        stratify=df_nw_tr_val_scope[PRED_WHALE_COL] if PRED_WHALE_COL in df_nw_tr_val_scope else None
    )
    test_nw = df_test_scope[(df_test_scope[PRED_PAYER_COL] == 1) & (df_test_scope[PRED_WHALE_COL] == 0)].copy()

    # 3. Whale Payer 그룹 (회귀 2): BASE_SPLIT_SEED로 Train/Val 분할
    tr_whale_full, va_whale_full = train_test_split(
        df_w_tr_val_scope, test_size=DEFAULT_TEST_SIZE, random_state=BASE_SPLIT_SEED, shuffle=True,
        stratify=df_w_tr_val_scope[PRED_WHALE_COL] if PRED_WHALE_COL in df_w_tr_val_scope else None
    )
    test_whale = df_test_scope[(df_test_scope[PRED_PAYER_COL] == 1) & (df_test_scope[PRED_WHALE_COL] == 1)].copy()
    
    # Stage 3 분할 정보 컬럼 추가
    tr_nw_full["stage3_split"] = "train"
    va_nw_full["stage3_split"] = "val"
    test_nw["stage3_split"] = "test"
    
    tr_whale_full["stage3_split"] = "train"
    va_whale_full["stage3_split"] = "val"
    test_whale["stage3_split"] = "test"


    logging.info(f"✅ Total data: {df_full.shape[0]} | Predicted Payer (Train/Val/Test scope): {df_full[df_full[PRED_PAYER_COL]==1].shape[0]}")
    logging.info(f"    - Non-Whale Payer (Stage 3 Train/Val/Test): {len(tr_nw_full)}/{len(va_nw_full)}/{len(test_nw)}")
    logging.info(f"    - Whale Payer (Stage 3 Train/Val/Test): {len(tr_whale_full)}/{len(va_whale_full)}/{len(test_whale)}")
    
    # 예측 대상인 두 그룹의 데이터프레임 딕셔너리 반환
    return {
        "non_whale": {"tr": tr_nw_full, "va": va_nw_full, "test": test_nw},
        "whale": {"tr": tr_whale_full, "va": va_whale_full, "test": test_whale},
        "all": df_full # 최종 평가를 위한 전체 데이터셋
    }

# =====================================================================================
# ---- CORE LOGIC: STAGE 3 TRAIN (Single Seed)
# =====================================================================================

def _prepare_features_and_artifacts(data_sets, group_key, train_seed):
    """
    Stage 3 회귀 모델 학습을 위한 인코더, 임퓨터, 피처를 준비합니다.
    - BASE_SPLIT_SEED로 고정된 Train/Val 데이터를 사용합니다.
    """
    artifacts_path = ARTIFACTS_PATH
    artifacts_path.mkdir(exist_ok=True)
    
    # BASE_SPLIT_SEED로 고정된 Stage 3 Train/Val 데이터를 사용
    df_tr = data_sets[group_key]["tr"]
    df_va = data_sets[group_key]["va"]
    
    # Payer 전체를 기준으로 Global Artifacts를 한 번만 생성
    df_tr_all = pd.concat([data_sets["non_whale"]["tr"], data_sets["whale"]["tr"]], ignore_index=False)
    df_va_all = pd.concat([data_sets["non_whale"]["va"], data_sets["whale"]["va"]], ignore_index=False)
    df_train_val_all = pd.concat([df_tr_all, df_va_all], ignore_index=False)
    
    if len(df_tr_all) == 0 or len(df_va_all) == 0:
        raise ValueError(f"❌ Invalid Stage 3 split for all_payers: train={len(df_tr_all)}, val={len(df_va_all)}.")
    
    drop_cols_for_feature = [ID_COL, TARGET_COL, PRED_PAYER_COL, PRED_WHALE_COL, "stage1_tvt", "stage3_split"]

    # 1) 글로벌 인코더: Payer 전체 train+val 기준 (seed와 무관)
    Xtr_all_raw, feat_cols_global, cat_cols_global = build_features(df_train_val_all, TARGET_COL, drop_cols_for_feature)
    
    if not ENCODER_PATH.exists():
        enc = OrdinalCategoryEncoder().fit(df_train_val_all, cat_cols_global)
        joblib.dump(enc, ENCODER_PATH)
        logging.info(f"✅ Created and saved global encoder at {ENCODER_PATH}")
    else:
        enc = joblib.load(ENCODER_PATH)
        logging.info(f"✅ Loaded global encoder from {ENCODER_PATH}")

    # 2) Imputer: BASE_SPLIT_SEED 기준 (한 번만)
    if not IMPUTER_PATH.exists(): 
        # df_tr_all을 사용하여 Imputer 학습
        Xtr_base_raw, _, _ = build_features(df_tr_all, TARGET_COL, drop_cols_for_feature)
        Xtr_base_encoded = enc.transform(Xtr_base_raw)
        num_cols, med = fit_imputer(Xtr_base_encoded)
        joblib.dump((num_cols, med), IMPUTER_PATH)
        logging.info(f"✅ Created and saved global imputer (median) at {IMPUTER_PATH}")
    
    else:
        num_cols, med = joblib.load(IMPUTER_PATH)
        logging.info(f"✅ Loaded global imputer.")

    # 3) Group별 Train/Val에 인코딩/임퓨팅 적용
    
    # Train/Val 피처 구성: feat_cols_global을 사용하여 컬럼 필터링 및 순서 보장
    Xtr_raw = df_tr.reindex(columns=feat_cols_global, fill_value=0)
    Xva_raw = df_va.reindex(columns=feat_cols_global, fill_value=0)

    # 인코딩 & 임퓨팅 & 정리
    Xtr = enc.transform(Xtr_raw)
    Xva = enc.transform(Xva_raw)

    cat_cols_idx = [Xtr.columns.get_loc(c) for c in cat_cols_global if c in Xtr.columns]

    Xtr = apply_imputer(Xtr, num_cols, med)
    Xva = apply_imputer(Xva, num_cols, med)

    Xtr = _sanitize_cols(Xtr)
    Xva = _sanitize_cols(Xva)

    Xtr.index = df_tr.index
    Xva.index = df_va.index

    # 타겟 변수 준비 (로그 변환 선택적 적용)
    y_tr_raw = df_tr[TARGET_COL]
    y_va_raw = df_va[TARGET_COL]
    
    if config.USE_LOG_TRANSFORM:
        y_tr_log = np.log1p(y_tr_raw)
        y_va_log = np.log1p(y_va_raw)
    else:
        y_tr_log = y_tr_raw
        y_va_log = y_va_raw

    # feat_cols_global를 반환 목록에 추가합니다.
    return Xtr, y_tr_log, y_tr_raw, Xva, y_va_log, y_va_raw, cat_cols_idx, num_cols, med, enc, feat_cols_global


def run_stage3_train_core(train_seed, data_sets):
    """
    단일 시드 학습을 위한 핵심 함수. 두 개의 회귀 모델 (Non-Whale, Whale)을 학습합니다.
    """
    
    # 전역 SEED 변경 없이 로컬 난수 상태만 설정
    np.random.seed(train_seed)
    random.seed(train_seed)
    
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    MODELS_PATH.mkdir(exist_ok=True)
    
    logging.info(f"--- Stage 3 Training (Seed: {train_seed}) ---")
    
    critical_error = None
    all_models = {}
    
    try:
        # 1. Non-Whale Payer 회귀 모델 학습
        with SectionTimer("Non-Whale Payer Regression Training"):
            # feat_cols_global을 받아옵니다.
            Xtr_nw, ytr_log_nw, ytr_raw_nw, Xva_nw, yva_log_nw, yva_raw_nw, cat_cols_idx, _, _, _, feat_cols_global = _prepare_features_and_artifacts(
                data_sets, "non_whale", train_seed
            )
            
            # 피처 목록 저장 (예측 단계에서 사용)
            if not FEAT_COLS_PATH.exists():
                 joblib.dump(feat_cols_global, FEAT_COLS_PATH)
                 logging.info(f"✅ Saved feature columns list at {FEAT_COLS_PATH}")


            # train_seed를 명시적으로 전달
            models_nw, _, _ = train_and_ensemble_reg(
                Xtr_nw, ytr_log_nw, Xva_nw, yva_log_nw, yva_raw_nw, cat_cols_idx, "non_whale", 
                config.DEFAULT_TRIALS, _HAS_TABPFN, train_seed
            )
            all_models.update({f"nw_{k}": v for k, v in models_nw.items()})

        # 2. Whale Payer 회귀 모델 학습
        with SectionTimer("Whale Payer Regression Training"):
            # Whale 그룹은 feat_cols_global을 반환하지 않으므로 무시
            Xtr_w, ytr_log_w, ytr_raw_w, Xva_w, yva_log_w, yva_raw_w, cat_cols_idx, _, _, _, _ = _prepare_features_and_artifacts(
                data_sets, "whale", train_seed
            )
            
            # train_seed를 명시적으로 전달
            models_w, _, _ = train_and_ensemble_reg(
                Xtr_w, ytr_log_w, Xva_w, yva_log_w, yva_raw_w, cat_cols_idx, "whale", 
                config.DEFAULT_TRIALS, _HAS_TABPFN, train_seed
            )
            all_models.update({f"w_{k}": v for k, v in models_w.items()})

        # 3. 모델 객체 저장 
        model_file = MODELS_PATH / f"stage3_models_{train_seed}.joblib"
        joblib.dump(all_models, model_file)
        logging.info(f"✅ Saved 6 regression models for seed {train_seed} at {model_file}")

    except Exception as e:
        logging.error(f"❌ Model Training or Ensemble failed: {e}")
        critical_error = e 
        
    if critical_error:
        logging.error(f"☠️ Pipeline terminated due to critical error: {critical_error}")
        raise critical_error


# =====================================================================================
# ---- CORE LOGIC: STAGE 3 PREDICT ALL (Train+Val+Test)
# =====================================================================================

def run_stage3_predict_all_core(seed, data_sets):
    """
    단일 시드 훈련 후, 해당 시드의 모델을 사용하여 전체 데이터셋 (Train+Val+Test)에 대한 예측을 수행하고 저장합니다.
    """
    logging.info(f"--- Starting Stage 3 Prediction for ALL Data (Seed: {seed}) ---")

    output_dir = OUTPUT_DIR
    model_file = MODELS_PATH / f"stage3_models_{seed}.joblib"
    predict_output_path = output_dir / f"stage3_predictions_all_{seed}.csv"
    
    df_all = data_sets["all"]

    if not model_file.exists():
        logging.error(f"❌ Model file not found for seed {seed}: {model_file}. Skipping prediction.")
        return

    if predict_output_path.exists() and SKIP_IF_EXISTS:
        logging.info(f"✅ Prediction file for seed {seed} already exists. Skipping prediction.")
        return
        
    try:
        # --- 1. 아티팩트 및 모델 로드 ---
        with SectionTimer("Loading Global Artifacts and Model"):
            enc = joblib.load(ENCODER_PATH)
            num_cols, med = joblib.load(IMPUTER_PATH)
            models = joblib.load(model_file)
            logging.info(f"Loaded {len(models)} regression models.")

            # 추가: 저장된 피처 목록 로드
            feat_cols = joblib.load(FEAT_COLS_PATH)
            logging.info(f"Loaded feature columns: {len(feat_cols)} columns.")
            
        # --- 2. 전체 데이터셋 전처리 ---
        with SectionTimer("Data Transformation (All Data)"):
            y_true_all_raw = df_all[TARGET_COL]
            
            drop_cols_for_feature = [ID_COL, TARGET_COL, PRED_PAYER_COL, PRED_WHALE_COL, "stage1_tvt", "stage3_split"]
            
            # 훈련 시 사용된 피처 목록(feat_cols)을 사용하여 피처 구성 및 정제
            X_all_raw_base, all_data_cols, cat_cols_raw = build_features(df_all, TARGET_COL, drop_cols_for_feature)
            
            # 핵심 수정: X_all_raw_base를 훈련 시 사용된 컬럼으로만 필터링 (순서 보장, 누락 시 0 채움)
            X_all_raw = X_all_raw_base.reindex(columns=feat_cols, fill_value=0)

            # 훈련 시 사용된 범주형 컬럼만 추출
            cat_cols = [c for c in feat_cols if c in cat_cols_raw]

            X_all = enc.transform(X_all_raw)
            # 인코딩된 데이터에서 범주형 컬럼의 인덱스를 찾습니다.
            cat_cols_idx = [X_all.columns.get_loc(c) for c in cat_cols if c in X_all.columns]
            
            X_all = apply_imputer(X_all, num_cols, med)
            X_all = _sanitize_cols(X_all) 
            X_all.index = df_all.index

        # --- 3. 그룹별 예측 및 최종 결합 ---
        with SectionTimer("Generating Group Predictions and Final Ensemble"):
            
            # Non-Payer 그룹: 최종 예측 금액 0
            idx_non_payer = df_all[df_all[PRED_PAYER_COL] == 0].index
            
            # Non-Whale Payer 그룹
            idx_nw = df_all[(df_all[PRED_PAYER_COL] == 1) & (df_all[PRED_WHALE_COL] == 0)].index
            # Whale Payer 그룹
            idx_w = df_all[(df_all[PRED_PAYER_COL] == 1) & (df_all[PRED_WHALE_COL] == 1)].index
            
            final_pred_map = pd.Series(np.zeros(len(df_all)), index=df_all.index)
            
            # Non-Whale Payer 예측 (로그 스케일)
            if len(idx_nw) > 0 and models.get("nw_lgbm"): # 최소 하나의 모델 존재 확인
                X_nw = X_all.loc[idx_nw]
                preds_log = []
                for k in ["lgbm", "cat", "tab"]:
                    model = models.get(f"nw_{k}")
                    if model is not None:
                        p_log = predict_reg_model(k, model, X_nw, cat_cols_idx)
                        preds_log.append(p_log)
                
                if preds_log:
                    # Non-Whale Payer 앙상블 예측 (로그 스케일)
                    p_log_ensemble_nw = np.mean(np.column_stack(preds_log), axis=1)
                    # 역변환하여 최종 금액 예측
                    p_raw_ensemble_nw = inverse_transform(p_log_ensemble_nw)
                    final_pred_map.loc[idx_nw] = p_raw_ensemble_nw

            # Whale Payer 예측 (로그 스케일)
            if len(idx_w) > 0 and models.get("w_lgbm"): # 최소 하나의 모델 존재 확인
                X_w = X_all.loc[idx_w]
                preds_log = []
                for k in ["lgbm", "cat", "tab"]:
                    model = models.get(f"w_{k}")
                    if model is not None:
                        p_log = predict_reg_model(k, model, X_w, cat_cols_idx)
                        preds_log.append(p_log)
                        
                if preds_log:
                    # Whale Payer 앙상블 예측 (로그 스케일)
                    p_log_ensemble_w = np.mean(np.column_stack(preds_log), axis=1)
                    # 역변환하여 최종 금액 예측
                    p_raw_ensemble_w = inverse_transform(p_log_ensemble_w)
                    final_pred_map.loc[idx_w] = p_raw_ensemble_w
                    
            # 최종 예측 결과 결합
            df_all["pred_ltv_final"] = final_pred_map
            
            # --- 최종 지표 계산 ---
            mae_all = mean_absolute_error(y_true_all_raw.values, df_all["pred_ltv_final"].values)
            rmse_all = np.sqrt(mean_squared_error(y_true_all_raw.values, df_all["pred_ltv_final"].values))
            
            logging.info(f"[ALL PRED] Seed {seed} Final MAE={mae_all:.2f} | RMSE={rmse_all:.2f}")

            # --- 4. 결과 저장 ---
            pred_data = df_all[[ID_COL, TARGET_COL, "stage1_tvt", PRED_PAYER_COL, PRED_WHALE_COL, "pred_ltv_final"]].copy()
            pred_data["seed"] = seed
            
            pred_data.to_csv(predict_output_path, index=False)
            logging.info(f"✅ Saved all data predictions for seed {seed} at {predict_output_path}")

    except Exception as e:
        logging.error(f"❌ Critical error in predict_all for seed {seed}: {e}", exc_info=True)


# =====================================================================================
# ---- CORE LOGIC: STAGE 3 MULTI (Multi Seed Ensemble)
# =====================================================================================

def load_seed_preds(output_dir, seed, scope="all"):
    """scope에 따라 'all' 예측 파일을 로드합니다."""
    p = output_dir / f"stage3_predictions_{scope}_{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing predictions for seed={seed}, scope={scope}: {p}")
    df = pd.read_csv(p)
    df["seed"] = seed
    return df

def calculate_detailed_metrics(df_agg, group_name):
    """지정된 그룹에 대해 MAE, 예측값 평균, 실제값 평균, sMAPE를 계산합니다."""
    
    y_true = df_agg[TARGET_COL].values
    
    # 'ZERO_PREDICTION' 그룹에 대한 특별 처리: 예측값을 0으로 설정
    if "ZERO_PREDICTION" in group_name:
        y_pred = np.zeros_like(y_true, dtype=float)
        # Avg.Pred는 0
        avg_pred = 0.0
    else:
        y_pred = df_agg["pred_ltv_ensemble"].values
        avg_pred = y_pred.mean()
    
    if len(y_true) == 0:
        return {"scope": group_name, "mae": 0.0, "avg_pred": 0.0, "avg_true": 0.0, "smape": 0.0, "count": 0}

    mae = mean_absolute_error(y_true, y_pred)
    avg_true = y_true.mean()
    smape_val = smape(y_true, y_pred)
    
    logging.info(f"[{group_name.upper()} METRICS] MAE={mae:.2f} | Avg.Pred={avg_pred:.2f} | Avg.True={avg_true:.2f} | sMAPE={smape_val:.2f}% | Count={len(y_true)}")

    return {
        "scope": group_name, 
        "mae": float(mae), 
        "avg_pred": float(avg_pred), 
        "avg_true": float(avg_true), 
        "smape": float(smape_val),
        "count": len(y_true)
    }

def run_stage3_multi_core():
    """멀티 시드 학습 및 앙상블을 위한 핵심 함수."""
    
    seeds = parse_seeds(config.DEFAULT_SEEDS_STR)
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    # 데이터 로드 및 분할
    try:
        data_sets = load_data_and_prepare_for_regression()
    except Exception as e:
        logging.error(f"❌ Initial Data Loading failed: {e}")
        sys.exit(1)
    
    ensemble_mode = config.ENSEMBLE_MODE
    skip_if_exists = config.SKIP_IF_EXISTS

    FINAL_PARQUET_PATH = output_dir / "stage3_final_predictions_all_data.parquet"
    
    if skip_if_exists and FINAL_PARQUET_PATH.exists():
        logging.info(f"✅ Final ensemble prediction file exists at {FINAL_PARQUET_PATH}. Skipping entire stage.")
        sys.exit(0)
    
    # 1) Train per seed (if not skipping)
    for s in seeds:
        pred_file_all = output_dir / f"stage3_predictions_all_{s}.csv"
        
        if skip_if_exists and pred_file_all.exists():
            logging.info(f"[SKIP] Seed {s} ALL predictions exist.")
            continue
        
        try:
            # 훈련 실행 (Non-Whale/Whale 그룹에 대한 회귀 모델 6개 학습 및 저장)
            run_stage3_train_core(s, data_sets)
            # 전체 데이터셋에 대한 예측 실행
            run_stage3_predict_all_core(s, data_sets)
        except Exception as e:
            logging.warning(f"Seed {s} run failed: {e}")
            
    # 2) Load predictions and aggregate
    with SectionTimer("Aggregating Ensemble Predictions (ALL Data)"):
        all_pred_dfs = []
        seeds_used = []
        weights = None 
        
        for s in seeds:
            try:
                df = load_seed_preds(output_dir, s, scope="all")
                all_pred_dfs.append(df)
                seeds_used.append(s)
            except FileNotFoundError:
                logging.warning(f"Skipping seed {s} due to missing ALL prediction file (CSV).")
                continue
        
        if not all_pred_dfs:
            logging.error("❌ No valid ALL prediction files found for ensembling.")
            sys.exit(1)
            
        all_df = pd.concat(all_pred_dfs, ignore_index=True)
        id_col_local = ID_COL

        # 시드별 예측값의 평균을 계산
        def agg_fn_mean(g):
            p = g["pred_ltv_final"].mean()
            y = g[TARGET_COL].iloc[0]
            tvt = g["stage1_tvt"].iloc[0]
            pred_w = g[PRED_WHALE_COL].iloc[0]
            return pd.Series({
                "pred_ltv_ensemble": p, 
                TARGET_COL: y, 
                "stage1_tvt": tvt, 
                PRED_WHALE_COL: pred_w
            })

        agg_all = all_df.groupby(id_col_local, as_index=False).apply(agg_fn_mean).reset_index(drop=True)
        
        # 실제 Payer 정의: TARGET_COL > 0
        agg_all["is_real_payer"] = (agg_all[TARGET_COL] > 0).astype(int)
        
        # 실제 Whale 정의: 실제 결제 금액이 Payer 중 상위 5% 이상인 경우 (임시 컷오프)
        real_payer_df = agg_all[agg_all["is_real_payer"] == 1]
        real_whale_threshold = real_payer_df[TARGET_COL].quantile(0.95) if not real_payer_df.empty else 0
        agg_all["is_real_whale"] = ((agg_all[TARGET_COL] > real_whale_threshold) & (agg_all["is_real_payer"] == 1)).astype(int)
        
    # 3) Final Metrics (Detailed)
    with SectionTimer("Final Detailed Metrics Calculation (TEST Set)"):
        all_metrics = {}
        df_test = agg_all[agg_all["stage1_tvt"] == "test"].copy()
        
        # 그룹 정의
        df_real_payer = df_test[df_test["is_real_payer"] == 1].copy()
        df_real_non_payer = df_test[df_test["is_real_payer"] == 0].copy()
        df_real_whale = df_test[df_test["is_real_whale"] == 1].copy()
        df_real_non_whale_payer = df_test[(df_test["is_real_whale"] == 0) & (df_test["is_real_payer"] == 1)].copy()
        
        # --- 추가: Zero Prediction 지표 계산 ---
        # Zero Prediction을 위한 임시 DataFrame 생성
        df_test_zero = df_test.copy()
        df_test_zero["pred_ltv_ensemble"] = 0.0
        
        # Zero Prediction 지표 계산
        all_metrics["zero_prediction_all"] = calculate_detailed_metrics(df_test_zero, "ZERO_PREDICTION_ALL_TEST")


        # 1. Real Whale
        all_metrics["real_whale"] = calculate_detailed_metrics(df_real_whale, "REAL_WHALE_TEST")
        # 2. Real Non-Whale Payer
        all_metrics["real_non_whale_payer"] = calculate_detailed_metrics(df_real_non_whale_payer, "REAL_NON_WHALE_PAYER_TEST")
        # 3. Real Non-Payer
        all_metrics["real_non_payer"] = calculate_detailed_metrics(df_real_non_payer, "REAL_NON_PAYER_TEST")
        
        # 4. Real Payer: 전체
        all_metrics["real_payer_overall"] = calculate_detailed_metrics(df_real_payer, "REAL_PAYER_OVERALL_TEST")
        
        # 4. Real Payer: 20% 구간별 (Quantile)
        if len(df_real_payer) > 0:
            # qcut의 labels=False는 0부터 시작하는 정수 레이블을 반환
            df_real_payer['quantile_group'] = pd.qcut(df_real_payer[TARGET_COL], q=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=False, duplicates='drop')
            for i in sorted(df_real_payer['quantile_group'].unique()):
                group_name = f"REAL_PAYER_Q{i*20+1}-{(i+1)*20}_TEST"
                all_metrics[f"real_payer_q{i}"] = calculate_detailed_metrics(df_real_payer[df_real_payer['quantile_group'] == i], group_name)
        
        # 4. Real Payer: Top 1, 3, 5, 10% (Percentile)
        for pct in [1, 3, 5, 10]:
            if len(df_real_payer) > 0:
                # 1 - (pct / 100)로 상위 %의 임계값을 구함 (예: Top 5% -> 0.95 quantile)
                top_threshold = df_real_payer[TARGET_COL].quantile(1 - (pct / 100))
                df_top_pct = df_real_payer[df_real_payer[TARGET_COL] >= top_threshold].copy()
                if not df_top_pct.empty:
                    group_name = f"REAL_PAYER_TOP_{pct}PCT_TEST"
                    all_metrics[f"real_payer_top_{pct}pct"] = calculate_detailed_metrics(df_top_pct, group_name)
                    
        # 5. All
        all_metrics["all"] = calculate_detailed_metrics(df_test, "ALL_TEST")

        # --- 6. 지표를 DataFrame으로 변환하고 CSV로 저장 ---
        df_metrics = pd.DataFrame.from_dict(all_metrics, orient='index')
        
        # timestamp 추가 (시분까지)
        timestamp_metrics = datetime.now().strftime("%Y%m%d_%H%M")
        csv_report_path = output_dir / f"stage3_final_metrics_{timestamp_metrics}.csv"
        
        # 'scope' 컬럼을 인덱스로 설정
        df_metrics = df_metrics.set_index('scope')
        
        # CSV 파일 저장
        df_metrics.to_csv(csv_report_path, float_format='%.2f')
        logging.info(f"[OK] Saved detailed metrics report (CSV): {csv_report_path}")
        
        
        # 4) Save outputs (ALL & TEST)
        outp_all = output_dir / "stage3_seed_ensemble_all.csv"
        agg_all.to_csv(outp_all, index=False)
        logging.info(f"[OK] Saved seed-ensemble ALL predictions (CSV): {outp_all}")
        
        outp_test = output_dir / "stage3_seed_ensemble_test.csv"
        df_test[[ID_COL, TARGET_COL, "stage1_tvt", PRED_WHALE_COL, "pred_ltv_ensemble"]].to_csv(outp_test, index=False)
        logging.info(f"[OK] Saved seed-ensemble TEST predictions (CSV): {outp_test}")

        # 5) Save Final Report (Detailed JSON)
        final_report_path = output_dir / "stage3_seed_ensemble_report.json"
        
        with open(final_report_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2) 
        
        logging.info(f"[OK] Saved seed-ensemble detailed report (json): {final_report_path}")

    # 6) Final Parquet File 생성
    with SectionTimer("Saving Final Ensemble Parquet File"):
        agg_all_indexed = agg_all.set_index(ID_COL).rename(columns={
        "pred_ltv_ensemble": "final_ltv_pred"
        })[["final_ltv_pred"]]
        
        df_base_full = data_sets["all"].copy()
        if ID_COL in df_base_full.columns:
            df_base_full = df_base_full.set_index(ID_COL, drop=False)
        
        final_output_df = df_base_full.copy()
        final_output_df = final_output_df.join(agg_all_indexed, how='left')

        # 기존 예측이 없던 Non-Payer 예측값 (0)을 채워 넣음
        final_output_df["final_ltv_pred"] = final_output_df["final_ltv_pred"].fillna(0.0) 

        final_output_df.to_parquet(FINAL_PARQUET_PATH, index=False, engine='pyarrow')
        logging.info(f"✅ Saved Final Ensemble Predictions to: {FINAL_PARQUET_PATH}")
            
    # 7) Exit
    logging.info("✅ Multi-Seed Ensemble Regression Pipeline Complete.")
    sys.exit(0) 

# =====================================================================================
# ---- MAIN EXECUTION ENTRY POINT
# =====================================================================================

def main_cli_entry():
    """메인 진입점."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_PATH / f"ltv_pipeline_stage3_{timestamp}_overall.log"

    # --- 로깅 초기화 (콘솔 출력 + 파일 저장) ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    )
    
    try:
        # --- 0. 최초 실행 시 데이터 전처리 (stage2_data.parquet 생성) ---
        create_stage2_data_parquet()
        # --- 1. 멀티 시드 앙상블 실행 ---
        run_stage3_multi_core()
    except SystemExit: 
        pass
    except Exception as e:
        logging.error(f"☠️ A critical error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main_cli_entry()