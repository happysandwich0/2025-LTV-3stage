
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

from config import DEVICE, SEED

_HAS_TABPFN = False

class SectionTimer:
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
    out = df.copy()
    out.columns = out.columns.astype(str).str.replace(r"\s+", "_", regex=True)
    return out

def optuna_progress_cb_strict(tag):
    def _cb(study, trial):
        val = trial.value if trial.value is not None else float('nan')
        complete_vals = [t.value for t in study.trials
                         if t.state == TrialState.COMPLETE and t.value is not None]
        running_best = max(complete_vals) if complete_vals else float('nan')
        logging.info(f"[Optuna|{tag}] trial#{trial.number:03d} value={val:.6f} best_now={running_best:.6f} ...")
    return _cb

def construct_tabpfn(cls, device, seed, n_ens):
    device_to_use = DEVICE if device == "auto" else device
    
    try:
        sig = inspect.signature(cls.__init__)
        kw = {}
        if "device" in sig.parameters: kw["device"] = device_to_use
        if "N_ensemble_configurations" in sig.parameters: kw["N_ensemble_configurations"] = n_ens
        elif "n_estimators" in sig.parameters: kw["n_estimators"] = n_ens 
        if "seed" in sig.parameters: kw["seed"] = seed
        elif "random_state" in sig.parameters: kw["random_state"] = seed
        return cls(**kw) if kw else cls()
    except Exception as e:
        logging.warning(f"Failed to inspect or construct TabPFN class {cls.__name__}: {e}. Trying default construction.")
        return cls()
        
def make_tabpfn_classifier(device, seed, n_ens):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return None
        
    return construct_tabpfn(TabPFNClassifier, device=device, seed=seed, n_ens=n_ens)

def parse_seeds(text):
    text = str(text).strip()
    if ".." in text:
        a, b = text.split("..", 1)
        a, b = int(a), int(b)
        return list(range(min(a, b), max(a, b) + 1))
    if "," in text:
        return [int(s.strip()) for s in text.split(",") if s.strip()]
    return [int(text)]
