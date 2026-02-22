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

from config import F_BETA, DELTA_AROUND, CUT_STEP, MIN_PREC_AT_CUT, DEVICE

_HAS_TABPFN = False

class SectionTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
    def __enter__(self):
        self.t0 = perf_counter()
        logging.info(f" START: {self.name}")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = perf_counter() - self.t0
        if exc:
            logging.error(f" FAIL:  {self.name} ({dt:.1f}s)", exc_info=True)
            return False 
        else:
            logging.info(f" DONE:  {self.name} ({dt:.1f}s)")
            return True 

def _sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.replace(r"\s+", "_", regex=True)
    return out

def hard_vote(preds: Dict[str, np.ndarray], cutoffs: Dict[str, float]) -> np.ndarray:
    votes = []
    for k, p in preds.items():
        t = cutoffs[k]
        votes.append((p >= t).astype(int))
    votes = np.column_stack(votes)
    return (votes.sum(axis=1) >= int(math.ceil(votes.shape[1]/2))).astype(int)

def _search_cutoff_grid(y_true, proba, center: float, delta: float, step: float, metric: str, beta: float, min_prec: float):
    lo = max(0.0, center - delta); hi = min(1.0, center + delta)
    grid = np.arange(lo, hi + 1e-9, step)
    y_true = np.asarray(y_true).astype(int)
    best_t, best_s = 0.5, -1.0

    for t in grid:
        y_hat = (proba >= t).astype(int)
        if y_hat.sum() == 0 or y_true.sum() == 0: s = 0.0
        else:
            if min_prec is not None and min_prec > 0:
                prec = precision_score(y_true, y_hat, zero_division=0)
                if prec < min_prec: continue
            
            s = fbeta_score(y_true, y_hat, beta=beta, zero_division=0) if metric == "fbeta" else f1_score(y_true, y_hat)
        
        if s > best_s: best_t, best_s = float(t), float(s)

    return best_t, {"f_beta": best_s, "threshold": best_t}

def tune_cutoff(y_true, proba, strategy: str, train_pos_prior: float, metric: str = "fbeta", beta: float = F_BETA):
    center = float(np.clip(train_pos_prior, 0.05, 0.95)) if strategy == "prior" else 0.5
    return _search_cutoff_grid(y_true, proba, center=center, delta=DELTA_AROUND, step=CUT_STEP, metric=metric, beta=beta, min_prec=MIN_PREC_AT_CUT)

def score_stage2_objective(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, proba))
    except ValueError:
        return 0.0 

def optuna_progress_cb_strict(tag: str):
    def _cb(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        val = trial.value if trial.value is not None else float('nan')
        complete_vals = [t.value for t in study.trials
                         if t.state == TrialState.COMPLETE and t.value is not None]
        running_best = max(complete_vals) if complete_vals else float('nan')
        logging.info(f"[Optuna|{tag}] trial#{trial.number:03d} value={val:.6f} best_now={running_best:.6f} ...")
    return _cb

def construct_tabpfn(cls, device: str, seed: int, n_ens: int):
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
    except Exception:
        return cls()
        
def make_tabpfn_classifier(device: str, seed: int, n_ens: int):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return None
    return construct_tabpfn(TabPFNClassifier, device=device, seed=seed, n_ens=n_ens)

def parse_seeds(text: str) -> List[int]:
    text = str(text).strip()
    if ".." in text:
        a, b = text.split("..", 1)
        a, b = int(a), int(b)
        return list(range(min(a, b), max(a, b) + 1))
    if "," in text:
        return [int(s.strip()) for s in text.split(",") if s.strip()]
    return [int(text)]
