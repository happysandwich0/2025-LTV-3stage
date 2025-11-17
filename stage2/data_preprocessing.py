"""
Stage 2 Data Preparation: Feature Engineering, Encoding, Imputation.
Also handles global TabPFN import check.
"""

import logging
import numpy as np
import pandas as pd
import warnings

from config import (
    ID_COL, 
    PRED_PAYER_COL, 
    STAGE1_PROBA_COL, 
    DEVICE, 
    USE_STAGE1_FEATURES,
    EXCLUDE_PROBA_FEATURES
)

# Global flag for TabPFN status
_HAS_TABPFN = False
try:
    from tabpfn import TabPFNClassifier
    _HAS_TABPFN = True
except Exception:
    _HAS_TABPFN = False
    warnings.warn("TabPFN import failed. Stage2 will skip TabPFN.")

# Update utils global status
import utils
utils._HAS_TABPFN = _HAS_TABPFN


class OrdinalCategoryEncoder:
    def __init__(self):
        self.maps = {}
        self.cols = []

    def fit(self, df, cat_cols):
        self.cols = list(cat_cols)
        for c in self.cols:
            cats = pd.Series(df[c].astype("category").cat.categories)
            self.maps[c] = {cat: i for i, cat in enumerate(cats)}
        return self

    def transform(self, df):
        out = df.copy()
        for c in self.cols:
            if c in out.columns:
                s = out[c].astype(object)
                out[c] = s.apply(lambda v: self.maps[c].get(v, -1)).astype(np.int32)
        return out


def build_features(df, target_col, drop_cols):
    """
    Stage 2 Feature Builder
    - drop_cols에 포함된 변수 제거
    - config.USE_STAGE1_FEATURES / EXCLUDE_PROBA_FEATURES 옵션 반영
    """
    drop_cols_no_id = [c for c in drop_cols if c != ID_COL]

    additional_drop = [c for c in [PRED_PAYER_COL, STAGE1_PROBA_COL] if c in df.columns]
    cols = [c for c in df.columns if c not in drop_cols_no_id and c not in additional_drop]
    X = df[cols].copy()

    # --- ① proba 컬럼 제거 ---
    if EXCLUDE_PROBA_FEATURES:
        proba_cols = [c for c in X.columns if "proba" in c.lower()]
        if proba_cols:
            logging.info(f"Excluding proba features: {proba_cols}")
            X = X.drop(columns=proba_cols, errors="ignore")

    # --- ② Stage1 확률 feature 추가 (조건부) ---
    if USE_STAGE1_FEATURES and STAGE1_PROBA_COL in df.columns:
        logging.info(f"Adding Stage 1 proba feature: {STAGE1_PROBA_COL}")
        X[STAGE1_PROBA_COL] = df[STAGE1_PROBA_COL]

    # --- ③ 최종 피처 목록 및 범주형 컬럼 추출 ---
    final_cols = X.columns.tolist()
    cat_cols = [
        c for c in final_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    return X, final_cols, cat_cols


def fit_imputer(train_df):
    num_cols = [
        c for c in train_df.columns
        if train_df[c].dtype != 'object'
        and not str(train_df[c].dtype).startswith('category')
    ]
    med = train_df[num_cols].median(numeric_only=True)
    return num_cols, med


def apply_imputer(df, num_cols, med):
    """결측치 보정"""
    df = df.copy()
    df[num_cols] = df[num_cols].fillna(med)
    df = df.fillna(0)
    return df
