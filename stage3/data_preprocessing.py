# data_preprocessing.py

"""
Data Preparation: Feature Engineering, Encoding, Imputation.
Also handles global TabPFN import check.
"""

import logging
import numpy as np
import pandas as pd
import warnings

# config_stage3에서 상수 직접 로드
from config import (
    ID_COL, PRED_PAYER_COL, STAGE1_PROBA_COL, STAGE2_PROBA_COL, 
    DEVICE, USE_STAGE1_FEATURES, USE_STAGE2_FEATURES
)

# Global flag for TabPFN status
_HAS_TABPFN = False
try:
    from tabpfn import TabPFNRegressor
    _HAS_TABPFN = True
except Exception:
    _HAS_TABPFN = False
    warnings.warn("TabPFN import failed. Stage 3 might skip TabPFN.")
    
# Update stage3_utils global status
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
                # 맵에 없는 값은 -1로 인코딩
                out[c] = s.apply(lambda v: self.maps[c].get(v, -1)).astype(np.int32) 
        return out

def build_features(df, target_col, drop_cols):
    """
    데이터프레임에서 피처를 구성하고 Stage 1/2 예측 확률을 조건부로 추가합니다.
    """
    drop_cols_no_id = [c for c in drop_cols if c != ID_COL]

    # Stage 1/2 예측 확률 컬럼이 drop_cols에 포함되어 있다면 제외하고, DataFrame에 포함된 것만 추가 드롭
    additional_drop = [
        c for c in [PRED_PAYER_COL, STAGE1_PROBA_COL, STAGE2_PROBA_COL, "pred_is_high_payer"] 
        if c in df.columns and c not in drop_cols
    ]
    
    cols = [c for c in df.columns if c not in drop_cols_no_id and c not in additional_drop]
    
    X = df[cols].copy()
    
    # config.USE_STAGE1_FEATURES 직접 사용
    if USE_STAGE1_FEATURES and STAGE1_PROBA_COL in df.columns:
        logging.info(f"Adding Stage 1 proba feature: {STAGE1_PROBA_COL}")
        X[STAGE1_PROBA_COL] = df[STAGE1_PROBA_COL]

    # config.USE_STAGE2_FEATURES 직접 사용
    if USE_STAGE2_FEATURES and STAGE2_PROBA_COL in df.columns:
        logging.info(f"Adding Stage 2 proba feature: {STAGE2_PROBA_COL}")
        X[STAGE2_PROBA_COL] = df[STAGE2_PROBA_COL]

    final_cols = X.columns.tolist() 
    cat_cols = [c for c in final_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    return X, final_cols, cat_cols


def fit_imputer(train_df):
    # 'object'와 'category'를 제외한 모든 것을 숫자형으로 간주
    num_cols = [c for c in train_df.columns if train_df[c].dtype != 'object' and not str(train_df[c].dtype).startswith('category')]
    # 숫자형 컬럼에 대해서만 중앙값 계산 (numeric_only=True 사용)
    med = train_df[num_cols].median(numeric_only=True) 
    return num_cols, med


def apply_imputer(df, num_cols, med):
    df = df.copy()
    # 숫자형 컬럼은 중앙값으로 채우기
    df[num_cols] = df[num_cols].fillna(med)
    # 인코딩되지 않은 범주형(object, category)이나 기타 남은 NaN은 0으로 채우기
    df = df.fillna(0) 
    return df