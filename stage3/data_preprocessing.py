
import logging
import numpy as np
import pandas as pd
import warnings

from config import (
    ID_COL, PRED_PAYER_COL, STAGE1_PROBA_COL, STAGE2_PROBA_COL, 
    DEVICE, USE_STAGE1_FEATURES, USE_STAGE2_FEATURES
)

_HAS_TABPFN = False
try:
    from tabpfn import TabPFNRegressor
    _HAS_TABPFN = True
except Exception:
    _HAS_TABPFN = False
    warnings.warn("TabPFN import failed. Stage 3 might skip TabPFN.")
    
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
    drop_cols_no_id = [c for c in drop_cols if c != ID_COL]
    additional_drop = [
        c for c in [PRED_PAYER_COL, STAGE1_PROBA_COL, STAGE2_PROBA_COL, "pred_is_high_payer"] 
        if c in df.columns and c not in drop_cols
    ]
    
    cols = [c for c in df.columns if c not in drop_cols_no_id and c not in additional_drop]
    
    X = df[cols].copy()
    
    if USE_STAGE1_FEATURES and STAGE1_PROBA_COL in df.columns:
        logging.info(f"Adding Stage 1 proba feature: {STAGE1_PROBA_COL}")
        X[STAGE1_PROBA_COL] = df[STAGE1_PROBA_COL]

    if USE_STAGE2_FEATURES and STAGE2_PROBA_COL in df.columns:
        logging.info(f"Adding Stage 2 proba feature: {STAGE2_PROBA_COL}")
        X[STAGE2_PROBA_COL] = df[STAGE2_PROBA_COL]

    final_cols = X.columns.tolist() 
    cat_cols = [c for c in final_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    return X, final_cols, cat_cols


def fit_imputer(train_df):
    num_cols = [c for c in train_df.columns if train_df[c].dtype != 'object' and not str(train_df[c].dtype).startswith('category')]
    med = train_df[num_cols].median(numeric_only=True) 
    return num_cols, med


def apply_imputer(df, num_cols, med):
    df = df.copy()
    df[num_cols] = df[num_cols].fillna(med)
    df = df.fillna(0) 
    return df
