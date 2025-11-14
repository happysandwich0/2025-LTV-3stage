# utils.py

"""
Stage 2/3 Utility Classes and Functions (Timer, Scoring, Cutoff Tuning, CLI Parsing).
Stage 3에서는 주로 SectionTimer와 TabPFN 유틸리티를 사용합니다.
"""
import logging
from time import perf_counter
import math
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, fbeta_score, f1_score, precision_score, recall_score
)
import optuna
from optuna.trial import TrialState
import inspect
import warnings

# config_stage3에서 상수 로드
from config import DEVICE, SEED

# Global TabPFN status (managed in data_preprocessing.py for the class)
_HAS_TABPFN = False

class SectionTimer:
    """코드 섹션 실행 시간을 측정하고 로깅하는 컨텍스트 관리자."""
    def __init__(self, name):
        self.name = name
        self.t0 = None
    
    def __enter__(self):
        self.t0 = perf_counter()
        logging.info(f"▶ START: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc, tb):
        dt = perf_counter() - self.t0
        if exc:
            logging.error(f"■ FAIL:  {self.name} ({dt:.1f}s)", exc_info=True)
            return False 
        else:
            logging.info(f"■ DONE:  {self.name} ({dt:.1f}s)")
            return True 
        
def _sanitize_cols(df):
    """컬럼 이름의 공백을 밑줄로 바꾸어 LightGBM/CatBoost 호환성을 확보합니다."""
    out = df.copy()
    out.columns = out.columns.astype(str).str.replace(r"\s+", "_", regex=True)
    return out

def optuna_progress_cb_strict(tag):
    """Optuna 튜닝 진행 상황을 로깅하는 콜백 함수."""
    def _cb(study, trial):
        val = trial.value if trial.value is not None else float('nan')
        complete_vals = [t.value for t in study.trials
                         if t.state == TrialState.COMPLETE and t.value is not None]
        running_best = max(complete_vals) if complete_vals else float('nan')
        logging.info(f"[Optuna|{tag}] trial#{trial.number:03d} value={val:.6f} best_now={running_best:.6f} ...")
    return _cb

def construct_tabpfn(cls, device, seed, n_ens):
    """TabPFNClassifier 또는 TabPFNRegressor 인스턴스를 생성합니다."""
    global _HAS_TABPFN
    if not _HAS_TABPFN: return None 
    import torch # TabPFNClassifier/Regressor 인자 구성에 필요
    device_to_use = DEVICE if device == "auto" else device
    
    # 클래스 생성자의 시그니처를 검사하여 파라미터를 동적으로 구성
    try:
        sig = inspect.signature(cls.__init__)
        kw = {}
        if "device" in sig.parameters: kw["device"] = device_to_use
        if "N_ensemble_configurations" in sig.parameters: kw["N_ensemble_configurations"] = n_ens
        elif "n_estimators" in sig.parameters: kw["n_estimators"] = n_ens # CatBoost/LGBM과의 호환성 (TabPFN에서는 N_ensemble_configurations 사용)
        if "seed" in sig.parameters: kw["seed"] = seed
        elif "random_state" in sig.parameters: kw["random_state"] = seed
        return cls(**kw) if kw else cls()
    except Exception as e:
        logging.warning(f"Failed to inspect or construct TabPFN class {cls.__name__}: {e}. Trying default construction.")
        return cls()
        
def make_tabpfn_classifier(device, seed, n_ens):
    """TabPFNClassifier 인스턴스를 생성합니다 (Stage 2 유틸리티, Stage 3에서는 Regressor를 사용해야 함)."""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return None
        
    return construct_tabpfn(TabPFNClassifier, device=device, seed=seed, n_ens=n_ens)

def parse_seeds(text):
    """시드 문자열(예: '2024..2025' 또는 '1,2,3')을 파싱하여 정수 리스트로 반환합니다."""
    text = str(text).strip()
    if ".." in text:
        a, b = text.split("..", 1)
        a, b = int(a), int(b)
        return list(range(min(a, b), max(a, b) + 1))
    if "," in text:
        return [int(s.strip()) for s in text.split(",") if s.strip()]
    return [int(text)]
