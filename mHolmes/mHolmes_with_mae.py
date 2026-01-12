import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, PatchTSTConfig, PatchTSTForPrediction, EarlyStoppingCallback

# -------------------------
# Config
# -------------------------
CSV_PATH = "/home/hanjin/projects/my_project/Human cadaver tansfer learning/reverse_hip_keytaxa.csv"
FT_CSV_PATH = "/home/hanjin/projects/my_project/Human cadaver tansfer learning/reverse_face_keytaxa.csv"
OUTPUT_DIR = "/home/hanjin/projects/my_project/Human cadaver tansfer learning/tf_rf_patchtst_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CONTEXT_LEN = 7   # 前7天
FUTURE_STEPS = 4  # 预测后4天
TOTAL_LEN = CONTEXT_LEN + FUTURE_STEPS

# -------------------------
# Utils
# -------------------------
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
    mse = g.apply(lambda d: mean_squared_error(d["y_true"], d["y_pred"])).to_dict()
    mae = g.apply(lambda d: mean_absolute_error(d["y_true"], d["y_pred"])).to_dict()
    return mse, mae

# -------------------------
# Sliding windows
# -------------------------
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
        feats,
    )

# -------------------------
# PatchTST model wrapper
# -------------------------
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
    train_set, val_set = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
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
        # 冻结整个 encoder（包括 positional encoder 和所有 self‐attn / FFN 层）
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        # 冻结输出投影层
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

# -------------------------
# Main pipeline
# -------------------------
df = pd.read_csv(CSV_PATH)
ft_df = pd.read_csv(FT_CSV_PATH)
features = [c for c in df.columns if c not in ("ID", "day")]
subjects = ft_df["ID"].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# For boxplots accumulation
folds_mse_past = []
folds_mae_past = []
folds_mse_pastgen = []
folds_mae_pastgen = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(subjects), start=1):
    fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_subj = subjects[train_idx]
    val_subj = subjects[val_idx]
    # train_df = df[df["ID"].isin(train_subj)].reset_index(drop=True)
    # val_df = df[df["ID"].isin(val_subj)].reset_index(drop=True)

    # Train RF on raw training rows (single time-point samples)
    X_train = df[features].to_numpy(dtype=float)
    y_train = df["day"].to_numpy(dtype=float)
    X_train, y_train = clean_xy(X_train, y_train)

    # rf = RandomForestRegressor(
    #     n_estimators=300,
    #     max_depth=None,
    #     random_state=SEED,
    #     n_jobs=-1
    # )
    # rf.fit(X_train, y_train)

    # Build sliding windows for PatchTST training (from training set) and for validation prediction (from validation set)
    tr_past, tr_future, _, _, _ = build_windows(df, CONTEXT_LEN, FUTURE_STEPS)
    # va_past, va_future, va_past_days, va_future_days, _ = build_windows(val_df, CONTEXT_LEN, FUTURE_STEPS)

    # Train PatchTST on training windows
    model = train_patchtst(tr_past, tr_future, num_features=len(features), context_len=CONTEXT_LEN, future_steps=FUTURE_STEPS, pretrained_model=None,
        freeze_layers=False)

    # # Predict future 4 days on validation windows
    # pred_future = patchtst_predict(model, va_past)  # (N, 4, F)

    # # Scenario A: past-only (first 7 days real)
    # y_true_past = va_past_days.reshape(-1)                 # (N*7,)
    # X_past_flat = va_past.reshape(-1, len(features))       # (N*7, F)
    # y_pred_past = rf.predict(X_past_flat)

    # # Scenario B: past + generated (first 7 real + 4 predicted)
    # full_feat = np.concatenate([va_past, pred_future], axis=1)  # (N, 11, F)
    # full_days = np.concatenate([va_past_days, va_future_days], axis=1)  # (N, 11)
    # y_true_full = full_days.reshape(-1)
    # X_full_flat = full_feat.reshape(-1, len(features))
    # y_pred_full = rf.predict(X_full_flat)

    # # Compute per-day MSE/MAE
    # mse_past, mae_past = compute_day_metrics(y_true_past, y_pred_past, bin_to_int=True)
    # mse_full, mae_full = compute_day_metrics(y_true_full, y_pred_full, bin_to_int=True)

    # folds_mse_past.append(mse_past)
    # folds_mae_past.append(mae_past)
    # folds_mse_pastgen.append(mse_full)
    # folds_mae_pastgen.append(mae_full)

    # # Save fold-level CSVs
    # def dict_to_df(d, metric_name, scenario):
    #     return pd.DataFrame([{"day": int(k), metric_name: v, "scenario": scenario, "fold": fold_idx} for k, v in d.items()])

    # df_out = []
    # df_out.append(dict_to_df(mse_past, "mse", "past-only"))
    # df_out.append(dict_to_df(mae_past, "mae", "past-only"))
    # df_out.append(dict_to_df(mse_full, "mse", "past+generated"))
    # df_out.append(dict_to_df(mae_full, "mae", "past+generated"))
    # fold_metrics = pd.concat(df_out, axis=0)
    # fold_metrics.to_csv(os.path.join(fold_dir, "per_day_metrics.csv"), index=False)
    # print(f"[FOLD {fold_idx}] Saved per-day metrics to {os.path.join(fold_dir, 'per_day_metrics.csv')}")

    hip_train_df = ft_df[ft_df["ID"].isin(train_subj)].reset_index(drop=True)
    hip_val_df   = ft_df[ft_df["ID"].isin(val_subj)].reset_index(drop=True)

    if len(hip_train_df) >= CONTEXT_LEN + FUTURE_STEPS and len(hip_val_df) >= CONTEXT_LEN + FUTURE_STEPS:
        # 2) 训练 hip 侧 RF（用于天数回归）
        Xh_train = hip_train_df[features].to_numpy(dtype=float)
        yh_train = hip_train_df["day"].to_numpy(dtype=float)
        Xh_train, yh_train = clean_xy(Xh_train, yh_train)
        rf_hip = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=SEED,
            n_jobs=-1
        )
        rf_hip.fit(Xh_train, yh_train)

        # 3) 构造滑动窗口（臀部）
        hip_tr_past, hip_tr_future, _, _, _ = build_windows(hip_train_df, CONTEXT_LEN, FUTURE_STEPS)
        hip_va_past, hip_va_future, hip_va_past_days, hip_va_future_days, _ = build_windows(hip_val_df, CONTEXT_LEN, FUTURE_STEPS)
       
        #仅取臀部训练窗口的 60% 用于微调
        finetune_ratio = 0.6
        num_ft = max(1, int(len(hip_tr_past) * finetune_ratio))
        rng = np.random.default_rng(SEED)
        sel_idx = rng.choice(len(hip_tr_past), size=num_ft, replace=False)
        hip_ft_past = hip_tr_past[sel_idx]
        hip_ft_future = hip_tr_future[sel_idx]

        # 4) 使用已在 face 上训练的 model 作为预训练权重进行微调（冻结主体参数）
        model_face_to_hip = train_patchtst(
            hip_ft_past, hip_ft_future,
            num_features=len(features),
            context_len=CONTEXT_LEN,
            future_steps=FUTURE_STEPS,
            pretrained_model=model,
            freeze_layers=True
        )

        # 5) 用微调后的模型预测臀部验证窗口未来步
        pred_future_hip = patchtst_predict(model_face_to_hip, hip_va_past)  # (N, FUTURE_STEPS, F)

        # 6) 两种场景：仅过去7天 vs 过去7天+生成4天
        y_true_past_hip = hip_va_past_days.reshape(-1)
        X_past_hip = hip_va_past.reshape(-1, len(features))
        y_pred_past_hip = rf_hip.predict(X_past_hip)

        full_feat_hip = np.concatenate([hip_va_past, pred_future_hip], axis=1)  # (N, 11, F)
        full_days_hip = np.concatenate([hip_va_past_days, hip_va_future_days], axis=1)
        y_true_full_hip = full_days_hip.reshape(-1)
        X_full_hip = full_feat_hip.reshape(-1, len(features))
        y_pred_full_hip = rf_hip.predict(X_full_hip)

        # 7) 逐天指标
        mse_past_hip, mae_past_hip = compute_day_metrics(y_true_past_hip, y_pred_past_hip, bin_to_int=True)
        mse_full_hip, mae_full_hip = compute_day_metrics(y_true_full_hip, y_pred_full_hip, bin_to_int=True)

        # 8) 保存迁移结果
        def dict_to_df_hip(d, metric_name, scenario):
            return pd.DataFrame([{"day": int(k), "tfdir": "hip-to-face",metric_name: v, "scenario": scenario, "fold": fold_idx} for k, v in d.items()])

        hip_out = []
        # hip_out.append(dict_to_df_hip(mse_past_hip, "mse", "past-only"))
        hip_out.append(dict_to_df_hip(mae_past_hip, "mae", "past-only"))
        # hip_out.append(dict_to_df_hip(mse_full_hip, "mse", "past+generated"))
        hip_out.append(dict_to_df_hip(mae_full_hip, "mae", "past+generated"))
        hip_metrics = pd.concat(hip_out, axis=0)
        hip_path = os.path.join(fold_dir, f"per_day_metrics_hip-to-face_keytaxa_{fold_idx}.csv")
        hip_metrics.to_csv(hip_path, index=False)
        print(f"[FOLD {fold_idx}] Saved transfer per-day metrics to {hip_path}")
# # Boxplots across folds
# plot_boxplot_from_folds(
#     folds_mse_past,
#     title="Per-day MSE (past-only, first 7 days)",
#     ylabel="MSE",
#     save_path=os.path.join(OUTPUT_DIR, "boxplot_face_keytaxa_mse_past_only.pdf")
# )
# plot_boxplot_from_folds(
#     folds_mae_past,
#     title="Per-day MAE (past-only, first 7 days)",
#     ylabel="MAE",
#     save_path=os.path.join(OUTPUT_DIR, "boxplot_face_keytaxa_mae_past_only.pdf")
# )
# plot_boxplot_from_folds(
#     folds_mse_pastgen,
#     title="Per-day MSE (past+generated, 7+4 days)",
#     ylabel="MSE",
#     save_path=os.path.join(OUTPUT_DIR, "boxplot_face_keytaxa_mse_past_plus_generated.pdf")
# )
# plot_boxplot_from_folds(
#     folds_mae_pastgen,
#     title="Per-day MAE (past+generated, 7+4 days)",
#     ylabel="MAE",
#     save_path=os.path.join(OUTPUT_DIR, "boxplot_face_keytaxa_mae_past_plus_generated.pdf")
# )
