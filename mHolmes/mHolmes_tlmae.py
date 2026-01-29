import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from pathlib import Path  # ADD
from src.utils.common import seed_everything
from transformers import Trainer, TrainingArguments, PatchTSTConfig, PatchTSTForPrediction, EarlyStoppingCallback

CONTEXT_LEN = 7  
FUTURE_STEPS = 4 
TOTAL_LEN = CONTEXT_LEN + FUTURE_STEPS
SEED = 42
seed_everything(SEED)

# ADD: absolute paths based on this script location (mHolmes/ and corpse_data/ are siblings)
BASE_DIR = Path(__file__).resolve().parent          # .../my_project/mHolmes
PROJECT_DIR = BASE_DIR.parent                       # .../my_project
DATA_DIR = PROJECT_DIR / "corpse_data"              # .../my_project/corpse_data
RESULT_DIR = PROJECT_DIR / "result" / "MAE"         # .../my_project/result/MAE

def clean_xy(X, y):
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]

def compute_day_metrics(y_true, y_pred, bin_to_int=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    day = np.rint(y_true).astype(int) if bin_to_int else y_true
    mask = np.isfinite(day) & np.isfinite(y_true) & np.isfinite(y_pred)
    df_tmp = pd.DataFrame({"day": day[mask], "y_true": y_true[mask], "y_pred": y_pred[mask]})
    # group by day
    g = df_tmp.groupby("day")
    mae = g.apply(lambda d: mean_absolute_error(d["y_true"], d["y_pred"])).to_dict()
    return mae

def build_windows(df, context_len=7, future_steps=4):
    """
    Return:
      past_values: (N, context_len, F)
      future_values: (N, future_steps, F)
      past_days: (N, context_len)
      future_days: (N, future_steps)
      feature_names: list[str]
    """
    assert "ID" in df.columns and "day" in df.columns
    feats = [c for c in df.columns if c not in ("ID", "day")]
    groups = df.groupby("ID")
    past_vals, fut_vals, past_days, fut_days = [], [], [], []
    for _, g in groups:
        g = g.sort_values("day")
        Xg = g[feats].to_numpy(dtype=float)
        days = g["day"].to_numpy(dtype=float)
        if len(g) < context_len + future_steps:
            continue
        for start in range(0, 16-(context_len + future_steps)+1):  # days 1 to 15-(7+4)+1=5
            pv = Xg[start : start + context_len]
            fv = Xg[start + context_len : start + context_len + future_steps]
            pdays = days[start : start + context_len]
            fdays = days[start + context_len : start + context_len + future_steps]
            # sanity: finite
            if not (np.isfinite(pv).all() and np.isfinite(fv).all() and np.isfinite(pdays).all() and np.isfinite(fdays).all()):
                continue
            past_vals.append(pv)
            fut_vals.append(fv)
            past_days.append(pdays)
            fut_days.append(fdays)
    if not past_vals:
        raise ValueError("No valid windows constructed. Check data coverage and NaNs.")
    return (
        np.stack(past_vals, axis=0),
        np.stack(fut_vals, axis=0),
        np.stack(past_days, axis=0),
        np.stack(fut_days, axis=0),
        feats
    )

class PatchTSTWithSoftmax(PatchTSTForPrediction):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, past_values=None,
                past_observed_mask=None,
                future_values=None,
                **kwargs):
        # only pass inputs needed by parent forward
        outputs = super().forward(past_values=past_values,
                                  past_observed_mask=past_observed_mask,
                                  **{k: v for k, v in kwargs.items() if k != 'future_values'})
        # if no predictions, return as-is
        if not (hasattr(outputs, 'prediction_outputs') and outputs.prediction_outputs is not None):
            return outputs

        # normalize to simplex (abundances often compositional)
        softmax_predictions = F.softmax(outputs.prediction_outputs, dim=-1)
        loss = None
        if future_values is not None:
            loss = F.mse_loss(softmax_predictions, future_values)
        out = {'prediction_outputs': softmax_predictions}
        if hasattr(outputs, 'loc'):
            out['loc'] = outputs.loc
        if hasattr(outputs, 'scale'):
            out['scale'] = outputs.scale
        if loss is not None:
            out['loss'] = loss
        return out

class TSDataset(torch.utils.data.Dataset):
    def __init__(self, past_values, future_values):
        """
        past_values: (N, L, F)
        future_values: (N, H, F)
        """
        self.past = torch.tensor(past_values, dtype=torch.float32)
        self.future = torch.tensor(future_values, dtype=torch.float32)
        self.mask = torch.ones_like(self.past)

    def __len__(self):
        return self.past.shape[0]

    def __getitem__(self, idx):
        return {
            "past_values": self.past[idx],                  # (L, F)
            "past_observed_mask": self.mask[idx],          # (L, F)
            "future_values": self.future[idx],             # (H, F)
        }

def train_patchtst(past_values, future_values, num_features, context_len, future_steps, device="cuda" if torch.cuda.is_available() else "cpu",  pretrained_model=None, freeze_layers=False):
    ds = TSDataset(past_values, future_values)
    # split 80/20 on windows for training PatchTST
    n = len(ds)
    n_train = max(1, int(0.8 * n))
    n_val = n - n_train
    train_set, val_set = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    if pretrained_model is None:
        config = PatchTSTConfig(
            num_input_channels=num_features,
            context_length=context_len,
            do_mask_input=True,
            num_hidden_layers=3,
            d_model=64,
            patch_length=1,
            patch_stride=1,
            nhead=4,
            ffn_dim=64*4,
            scaling=False,
            prediction_length=future_steps,
            dropout=0.2,
        )
        model = PatchTSTWithSoftmax(config)
    else:
        model = pretrained_model
    model.to(device)

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        for param in model.head.projection.parameters():
            param.requires_grad = True

    args = TrainingArguments(
        do_train=True,
        do_eval=True,
        disable_tqdm=False,
        lr_scheduler_type='linear',
        per_device_train_batch_size=8,
        num_train_epochs=1000,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        label_names=['past_values'],
        logging_steps=1,
        output_dir='model/',
        logging_dir='transformers_logs',
        load_best_model_at_end=True,
        weight_decay=0.05,
        remove_unused_columns=False
    )
    callback = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        callbacks=[callback],
    )
    trainer.train()
    return model

@torch.no_grad()
def patchtst_predict(model, past_values):
    """
    past_values: (N, L, F) numpy -> returns pred_future: (N, H, F) numpy
    """
    device = next(model.parameters()).device
    pv = torch.tensor(past_values, dtype=torch.float32, device=device)
    mask = torch.ones_like(pv)
    out = model(past_values=pv, past_observed_mask=mask)
    preds = out["prediction_outputs"] if isinstance(out, dict) else getattr(out, "prediction_outputs", out)
    # ensure non-negative (optional; model already softmaxes)
    preds = torch.clip(preds, min=0.0)
    return preds.detach().cpu().numpy()

folds_mae_past = []
folds_mae_pastgen = []

# CHANGED: output dir creation (absolute)
# os.makedirs('result/MAE', exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# CHANGED: input csv paths (absolute)
source_df = pd.read_csv(DATA_DIR / "source_hip.csv")
target_df = pd.read_csv(DATA_DIR / "target_face.csv")

features = [c for c in source_df.columns if c not in ("ID", "day")]
subjects = target_df["ID"].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)


for fold_idx, (train_idx, val_idx) in enumerate(kf.split(subjects), start=1):
    train_subj = subjects[train_idx]
    val_subj = subjects[val_idx]

    X_train = source_df[features].to_numpy(dtype=float)
    y_train = source_df["day"].to_numpy(dtype=float)
    X_train, y_train = clean_xy(X_train, y_train)

    src_past, src_future, _, _, _ = build_windows(source_df, CONTEXT_LEN, FUTURE_STEPS)

    model = train_patchtst(src_past, src_future, num_features=len(features), context_len=CONTEXT_LEN, future_steps=FUTURE_STEPS, pretrained_model=None,
        freeze_layers=False)

    target_train_df = target_df[target_df["ID"].isin(train_subj)].reset_index(drop=True)
    target_val_df   = target_df[target_df["ID"].isin(val_subj)].reset_index(drop=True)

    if len(target_train_df) >= CONTEXT_LEN + FUTURE_STEPS and len(target_val_df) >= CONTEXT_LEN + FUTURE_STEPS:
        Xh_train = target_train_df[features].to_numpy(dtype=float)
        yh_train = target_train_df["day"].to_numpy(dtype=float)
        Xh_train, yh_train = clean_xy(Xh_train, yh_train)
        rf_target = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=SEED,
            n_jobs=-1
        )
        rf_target.fit(Xh_train, yh_train)

        tgt_past, tgt_future, _, _, _ = build_windows(target_train_df, CONTEXT_LEN, FUTURE_STEPS)
        tgt_val_past, tgt_val_future, tgt_val_past_days, tgt_val_future_days, _ = build_windows(target_val_df, CONTEXT_LEN, FUTURE_STEPS)

        finetune_ratio = 0.6
        num_ft = max(1, int(len(tgt_past) * finetune_ratio))
        rng = np.random.default_rng(SEED)
        sel_idx = rng.choice(len(tgt_past), size=num_ft, replace=False)
        tgt_ft_past = tgt_past[sel_idx]
        tgt_ft_future = tgt_future[sel_idx]

        transfer_model = train_patchtst(
            tgt_ft_past, tgt_ft_future,
            num_features=len(features),
            context_len=CONTEXT_LEN,
            future_steps=FUTURE_STEPS,
            pretrained_model=model,
            freeze_layers=True
        )

        pred_future_target = patchtst_predict(transfer_model, tgt_val_past)  # (N, FUTURE_STEPS, F)

        y_true_past_target = tgt_val_past_days.reshape(-1)
        X_past_target = tgt_val_past.reshape(-1, len(features))
        y_pred_past_target = rf_target.predict(X_past_target)

        full_feat_target = np.concatenate([tgt_val_past, pred_future_target], axis=1)  # (N, 11, F)
        full_days_target = np.concatenate([tgt_val_past_days, tgt_val_future_days], axis=1)
        y_true_full_target = full_days_target.reshape(-1)
        X_full_target = full_feat_target.reshape(-1, len(features))
        y_pred_full_target = rf_target.predict(X_full_target)

        mae_past_target = compute_day_metrics(y_true_past_target, y_pred_past_target, bin_to_int=True)
        mae_full_target = compute_day_metrics(y_true_full_target, y_pred_full_target, bin_to_int=True)

        def dict_to_df(d, scenario):
            return pd.DataFrame(
                [{"day": int(day), "mae": float(v), "scenario": scenario, "fold": fold_idx}
                for day, v in d.items()]
            )
    
        df_out = []
        df_out.append(dict_to_df(mae_past_target, "past-only"))
        df_out.append(dict_to_df(mae_full_target, "past+generated"))
        fold_metrics = pd.concat(df_out, axis=0)
        
        # CHANGED: output csv path (absolute)
        fold_metrics.to_csv(RESULT_DIR / f"transfer_{fold_idx}_metrics.csv", index=False)
