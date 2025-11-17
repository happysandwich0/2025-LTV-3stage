import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
from functools import partial
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import contextlib
import random
import traceback
import torch.nn.functional as F
from datetime import datetime
from typing import List, Tuple, Dict

from utils import (
    setup_logger,
    get_device,
    set_seed,
    load_seq_parquet,
    build_event_vocab,
    save_ckpt,
    load_ckpt,
    evaluate_reg,
    transform_target,
    inverse_transform_target,
    DEVICE,
    DEVICE_TYPE,
    PIN_MEMORY,
)

from datasets import (
    SeqDataset,
    SeqDatasetInfer,
    collate_batch,
    collate_infer,
    make_length_sorted_loader,
)
from models import LongformerRegressor

from config import HP

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(os.path.dirname(__file__), f"logs/training_{timestamp}.log")
logger = setup_logger("main_logger", log_file_path)
output_dir = os.path.join("outputs", timestamp)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

class OrdinalCategoryEncoder:
    def __init__(self):
        self.maps: Dict[str, Dict] = {}
        self.cols: List[str] = []

    def fit(self, df: pd.DataFrame, cat_cols: List[str]):
        self.cols = list(cat_cols)
        for c in self.cols:
            if c in df.columns:
                cats = pd.Series(df[c].astype("category").cat.categories)
                self.maps[c] = {cat: i for i, cat in enumerate(cats)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.cols:
            if c in out.columns:
                mapping = self.maps[c]
                s = out[c].astype(object)
                out[c] = s.apply(lambda v: mapping.get(v, -1)).astype(np.int32)
        return out

def train_one(train_df, valid_df, stoi, max_len=None, batch_size=512, epochs=12, patience=3,
              d_model=96, nhead=4, nlayers=2, lr=3e-4, wd=1e-4, y_col='PAY_AMT',
              base_rate=0.03675, verbose=True, save_dir="checkpoints/seq_cls", run_name="default",
              resume=False, global_tokens=[], tabular_data_train=None, tabular_data_valid=None, **kwargs):
    
    transformation_mode = kwargs.get('transformation_mode', 'log1p')
    loss_mode = kwargs.get('loss_mode', 'mae')
    huber_delta = kwargs.get('huber_delta', 1.0)
    num_workers = kwargs.get('num_workers', 0)
    regression_model_type = kwargs.get('regression_model_type', 'mlp')

    vocab_size = len(stoi)
    if verbose: logger.info(f"[INFO] Vocab size: {vocab_size}")

    ckpt_last = f"{save_dir}/{run_name}_last.pt"
    ckpt_best = f"{save_dir}/{run_name}_best.pt"

    tabular_input_dim = tabular_data_train.drop(columns=['PLAYERID'], errors='ignore').shape[1] if tabular_data_train is not None and not tabular_data_train.empty else 0
    
    tr_ds = SeqDataset(train_df, stoi, y_col=y_col, max_len=max_len, global_tokens=global_tokens, transformation_mode=transformation_mode, tabular_data=tabular_data_train)
    va_ds = SeqDataset(valid_df, stoi, y_col=y_col, max_len=max_len, global_tokens=global_tokens, transformation_mode=transformation_mode, tabular_data=tabular_data_valid)
    
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                                        collate_fn=partial(collate_batch, max_len=max_len), num_workers=num_workers, pin_memory=PIN_MEMORY)
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                                        collate_fn=partial(collate_infer, max_len=max_len), num_workers=num_workers, pin_memory=PIN_MEMORY)
    logger.info("Train/Validation loaders created. Starting training.")

    model = LongformerRegressor(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead, nlayers=nlayers,
        p=0.1, base_rate=base_rate, max_len=max_len, tabular_input_dim=tabular_input_dim, regression_model_type=regression_model_type
    ).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    if DEVICE_TYPE == "cuda":
        scaler = None
        amp_ctx = contextlib.nullcontext()
        
    elif DEVICE_TYPE == "mps":
        scaler = None
        amp_ctx = contextlib.nullcontext()
    else:
        scaler = None
        amp_ctx = contextlib.nullcontext()

    start_epoch = 1
    best_metric = float('inf')
    transform_params = tr_ds.transform_params
    
    if resume and os.path.exists(ckpt_last):
        try:
            ckpt = load_ckpt(ckpt_last, model=model, opt=opt, sched=sched)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = ckpt.get("best_metric", float('inf'))
            transform_params = ckpt.get("transform_params")
            if verbose:
                logger.info(f"[resume] Loaded {ckpt_last} @epoch {start_epoch-1} | best RMSE={best_metric:.4f}")
        except Exception:
            logger.exception(f"Failed to load checkpoint from {ckpt_last}. Starting from scratch.")
            
    wait = 0
    
    if loss_mode == 'mae':
        loss_fn = F.l1_loss
    elif loss_mode == 'huber':
        loss_fn = partial(F.huber_loss, delta=huber_delta)
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    
    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_loss, n_samples = 0.0, 0
        
        if verbose:
            t_bar = tqdm(tr_ld, desc=f"Epoch {ep}/{epochs}")
        else:
            t_bar = tr_ld

        for batch in t_bar:
            try:
                ev = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                global_mask = batch.get('global_attention_mask', None)
                if global_mask is not None:
                    global_mask = global_mask.to(DEVICE)
                y = batch['labels'].float().to(DEVICE)
                tabular_features = batch['tabular_features'].to(DEVICE)

                opt.zero_grad(set_to_none=True)
                with amp_ctx:
                    position_ids = torch.arange(ev.shape[1], dtype=torch.long, device=DEVICE)
                    position_ids = position_ids.unsqueeze(0).expand_as(ev)
                    logits = model(ev, attention_mask=mask, global_attention_mask=global_mask, tabular_features=tabular_features, position_ids=position_ids)
                    
                    loss = loss_fn(logits, y, reduction='mean')

                if scaler is not None:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total_loss += loss.item() * ev.size(0)
                n_samples += ev.size(0)

                if verbose:
                    t_bar.set_postfix(loss=total_loss / max(n_samples, 1))

            except Exception:
                logger.exception(f"Error during training epoch {ep}, batch {n_samples}:")
                raise

        tr_loss = total_loss / max(n_samples, 1)
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for batch in va_ld:
                ev = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                global_mask = batch.get('global_attention_mask', None)
                if global_mask is not None:
                    global_mask = global_mask.to(DEVICE)
                y = batch['labels'].to(DEVICE)
                tabular_features = batch['tabular_features'].to(DEVICE)

                out = model(ev, attention_mask=mask, global_attention_mask=global_mask, tabular_features=tabular_features)
                all_p.append(out.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
        
        y_true_valid = np.concatenate(all_y)
        p_hat_valid = np.concatenate(all_p)
        
        metrics_valid = evaluate_reg(y_true_valid, p_hat_valid)
        score = metrics_valid['RMSE']

        if verbose:
            logger.info(f"[{ep:02d}] loss {tr_loss:.4f} | RMSE {metrics_valid['RMSE']:.4f} | MAE {metrics_valid['MAE']:.4f} | R2 {metrics_valid['R2']:.4f}")

        sched.step()
        
        save_ckpt(ckpt_last, model, opt, sched, ep, best_metric, stoi, HP, transform_params)
        if score < best_metric - 1e-4:
            best_metric = score
            save_ckpt(ckpt_best, model, opt, sched, ep, best_metric, stoi, HP, transform_params)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose: logger.info(f"Early stopping at epoch {ep} (best RMSE={best_metric:.4f})")
                break

    @torch.no_grad()
    def collect_probs(df, tabular_data=None):
        ds = SeqDataset(df, stoi, y_col=y_col, max_len=max_len, global_tokens=global_tokens, transformation_mode=transformation_mode, tabular_data=tabular_data)
        ld = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_batch, max_len=max_len), num_workers=num_workers, pin_memory=PIN_MEMORY)
        model.eval()
        all_pid, all_p, all_y = [], [], []
        for batch in ld:
            ev = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            global_mask = batch.get('global_attention_mask', None)
            if global_mask is not None:
                global_mask = global_mask.to(DEVICE)
            tabular_features = batch['tabular_features'].to(DEVICE)
            
            position_ids = torch.arange(ev.shape[1], dtype=torch.long, device=DEVICE)
            position_ids = position_ids.unsqueeze(0).expand_as(ev)
            
            p = model(ev, attention_mask=mask, global_attention_mask=global_mask, tabular_features=tabular_features, position_ids=position_ids).cpu().numpy()

            all_pid.extend(batch['ids'])
            all_p.append(p)
            all_y.append(batch['labels'].numpy())
        return np.array(all_pid), np.concatenate(all_p), np.concatenate(all_y)

    pid_v, p1_v, y_v = collect_probs(valid_df, tabular_data=tabular_data_valid)
    
    pred_valid = pd.DataFrame({
        "PLAYERID": pid_v,
        "payer_pred_valid": inverse_transform_target(p1_v, transformation_mode, transform_params),
        "true_value": inverse_transform_target(y_v, transformation_mode, transform_params),
    })

    return model, stoi, pred_valid, best_metric


if __name__ == '__main__':
    TEST_MODE = False
    
    try:
        tabular_train_df_raw = pd.read_parquet('/root/sblm/3stage/data/train_df_5days.parquet')
        tabular_val_df_raw = pd.read_parquet('/root/sblm/3stage/data/val_df_5days.parquet')
        tabular_test_df_raw = pd.read_parquet('/root/sblm/3stage/data/test_df_5days.parquet')
        
        TABULAR_Y_COL = 'PAY_AMT_SUM'
        
        all_tabular_df = pd.concat([tabular_train_df_raw, tabular_val_df_raw, tabular_test_df_raw], ignore_index=True)
        
        logger.info(f"All tabular data loaded and merged. Shape: {all_tabular_df.shape}")
        
        train_df = load_seq_parquet('./seq/train_df_5days_seq.parquet')
        val_df = load_seq_parquet('./seq/val_df_5days_seq.parquet')
        test_df = load_seq_parquet('./seq/test_df_5days_seq.parquet')

        logger.info("Sequence data loaded.")
        
        logger.info(f"Columns in all_tabular_df: {list(all_tabular_df.columns)}")
        logger.info(f"Columns in train_df (sequence): {list(train_df.columns)}")

        all_tabular_df_indexed = all_tabular_df.set_index('PLAYERID')
        
        tabular_train_df_for_model = all_tabular_df_indexed.loc[train_df['PLAYERID']].reset_index()
        tabular_val_df_for_model = all_tabular_df_indexed.loc[val_df['PLAYERID']].reset_index()
        tabular_test_df_for_model = all_tabular_df_indexed.loc[test_df['PLAYERID']].reset_index()

        logger.info(f"Tabular data partitioned based on sequence data's PLAYERIDs.")

        cat_cols = [col for col in tabular_train_df_for_model.columns 
                    if tabular_train_df_for_model[col].dtype == 'object']
        
        if 'NAT_CD' in cat_cols:
            logger.info("NAT_CD column found and will be handled as a categorical feature.")
        else:
            logger.warning("NAT_CD column not found in tabular data. It may have been excluded or renamed.")

        enc = OrdinalCategoryEncoder().fit(tabular_train_df_for_model, cat_cols)
        tabular_train_df_for_model = enc.transform(tabular_train_df_for_model)
        tabular_val_df_for_model = enc.transform(tabular_val_df_for_model)
        tabular_test_df_for_model = enc.transform(tabular_test_df_for_model)

        tabular_cols_to_use = [col for col in tabular_train_df_for_model.columns 
                               if col not in ['PLAYERID', TABULAR_Y_COL]]
        
        tabular_train_df_for_model = tabular_train_df_for_model[['PLAYERID'] + tabular_cols_to_use]
        tabular_val_df_for_model = tabular_val_df_for_model[['PLAYERID'] + tabular_cols_to_use]
        tabular_test_df_for_model = tabular_test_df_for_model[['PLAYERID'] + tabular_cols_to_use]
        
        logger.info(f"[DEBUG] tabular_train_df_for_model shape: {tabular_train_df_for_model.shape}")
        logger.info(f"[DEBUG] tabular_val_df_for_model shape: {tabular_val_df_for_model.shape}")
        logger.info(f"[DEBUG] tabular_test_df_for_model shape: {tabular_test_df_for_model.shape}")
        logger.info(f"[DEBUG] Final tabular columns for model: {list(tabular_train_df_for_model.columns)}")
        
        train_full = pd.concat([train_df, val_df], ignore_index=True)
        del train_df, val_df
        logger.info("Data loaded and preprocessed successfully.")
        
        if TEST_MODE:
            logger.info("\n===== TEST MODE ENABLED: Sampling 1000 items =====")
            if len(train_full) > 1000:
                train_full = train_full.sample(n=1000, random_state=2025).reset_index(drop=True)
            logger.info(f"Sampled train_full shape: {train_full.shape}")
    
    except FileNotFoundError:
        logger.error("Parquet 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected error occurred during data loading.")
        sys.exit(1)

    set_seed(2025)

    try:
        stoi, _ = build_event_vocab(train_full, min_freq=HP.get('min_freq', 3), top_k=HP.get('top_k_vocab', None))
        vocab_size = len(stoi)
        logger.info(f"[INFO] Global Vocab size: {vocab_size}")

        logger.info(f"[DEBUG] HP.max_len = {HP['max_len']}, HP.min_freq = {HP['min_freq']}, HP.top_k_vocab = {HP['top_k_vocab']}")
        logger.info(f"[DEBUG] stoi sample: {dict(list(stoi.items())[:10])}")

        logger.info("\n===== Final Model Training =====")
        final_model, _, _, final_metric = train_one(
            train_full, train_full, stoi, tabular_data_train=tabular_train_df_for_model, tabular_data_valid=tabular_train_df_for_model, **HP, verbose=True
        )
        
        logger.info(f"Final model trained. Best validation RMSE: {final_metric:.4f}")
        
        logger.info("\n===== Test Data Inference =====")
        
        test_ds = SeqDatasetInfer(test_df, stoi, max_len=HP['max_len'], tabular_data=tabular_test_df_for_model)
        test_ld = torch.utils.data.DataLoader(
            test_ds, batch_size=HP['batch_size'], shuffle=False,
            collate_fn=partial(collate_infer, max_len=HP['max_len']),
            num_workers=HP.get('num_workers', 0), pin_memory=PIN_MEMORY, persistent_workers=(HP.get('num_workers', 0) > 0)
        )
        
        final_model.eval()
        test_pids, test_preds = [], []
        
        ckpt_path = f"checkpoints/seq_cls/default_best.pt"
        ckpt = load_ckpt(ckpt_path, map_location=DEVICE)
        transform_params = ckpt.get("transform_params")
        
        with torch.no_grad():
            for batch in test_ld:
                ev = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                global_mask = batch.get('global_attention_mask', None)
                if global_mask is not None:
                    global_mask = global_mask.to(DEVICE)
                tabular_features = batch['tabular_features'].to(DEVICE)
                
                out = final_model(ev, attention_mask=mask, global_attention_mask=global_mask, tabular_features=tabular_features)
                test_pids.extend(batch['ids'])
                test_preds.append(out.detach().cpu().numpy())
        
        test_preds = np.concatenate(test_preds)
        
        predictions_df = pd.DataFrame({
            "PLAYERID": test_pids,
            "payer_prediction": inverse_transform_target(test_preds, HP['transformation_mode'], transform_params),
        })
        
        os.makedirs(output_dir, exist_ok=True)
        predictions_df.to_csv(os.path.join(output_dir, "test_predictions_reg.csv"), index=False)
        logger.info(f"[saved] {os.path.join(output_dir, 'test_predictions_reg.csv')}")

    except Exception:
        logger.exception("An error occurred during training or inference.")
        sys.exit(1)
