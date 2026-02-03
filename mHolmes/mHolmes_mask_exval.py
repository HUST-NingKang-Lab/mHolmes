import pandas as pd
import numpy as np
import os
from pathlib import Path  # ADD
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from src.data.TimeSeries import MicroTSDataset
from src.utils.common import seed_everything
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction
)

seed_everything(42)
PATCHTST_FUTURE_STEPS = 3

# ADD: robust absolute paths based on script location
BASE_DIR = Path(__file__).resolve().parent          # .../my_project/mHolmes
PROJECT_DIR = BASE_DIR.parent                       # .../my_project
DATA_DIR = PROJECT_DIR / "corpse_data"              # .../my_project/corpse_data
RESULT_DIR = PROJECT_DIR / "result" / "mask_exval"  # .../my_project/result/mask_exval
MODEL_DIR = PROJECT_DIR / "model" / "mask_exval"    # .../my_project/model/mask_exval
LOG_DIR = PROJECT_DIR / "transformers_logs" / "mask_exval"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def freeze_patchtst_layers(model):
    # 冻结除最后一层外的所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 冻结整个 encoder（包括 positional encoder 和所有 self‐attn / FFN 层）
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    # 冻结输出投影层
    for param in model.head.projection.parameters():
        param.requires_grad = True

class PatchTSTWithSoftmax(PatchTSTForPrediction):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, past_values=None,
                past_observed_mask=None,
                future_values=None,
                **kwargs):
        # 调用父类forward，但不计算损失
        kwargs_no_future = dict(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            **{k: v for k, v in kwargs.items() if k != 'future_values'}
        )
        outputs = super().forward(**kwargs_no_future)
        
        # If no predictions, return as-is
        if not (hasattr(outputs, 'prediction_outputs') and outputs.prediction_outputs is not None):
            return outputs
        
        # 应用softmax
        softmax_predictions = F.softmax(outputs.prediction_outputs, dim=-1)
        
        # Compute loss in softmax space if labels provided
        loss = None
        if future_values is not None:
            loss = F.mse_loss(softmax_predictions, future_values)
        out = {
            'prediction_outputs': softmax_predictions,
        }
        if hasattr(outputs, 'loc'):
            out['loc'] = outputs.loc
        if hasattr(outputs, 'scale'):
            out['scale'] = outputs.scale
        if loss is not None:
            out['loss'] = loss

        return out
    

class MaskedDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset that applies random masking to the past_values of a source dataset.
    Masking is only applied during training.
    """
    def __init__(self, source_dataset, mask_rate, is_train=True, base_seed=42):
        self.source_dataset = source_dataset
        self.mask_rate = mask_rate
        self.base_seed = base_seed
        self.is_train = is_train

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        sample = self.source_dataset[idx]
        if not self.is_train or self.mask_rate == 0:
            return sample

        past_values = sample['past_values'].clone()
        past_observed_mask = sample['past_observed_mask'].clone()
        
        L, C = past_values.shape # Length, Channels

        
        # Scheme 1: Randomly mask individual timesteps across all features.
        num_mask = int(self.mask_rate * L)
        rng = np.random.default_rng(self.base_seed + idx)  # 关键：跟 idx 绑定
        mask_indices = rng.choice(L, num_mask, replace=False)
        past_values[mask_indices, :] = 0
        past_observed_mask[mask_indices, :] = 0

        new_sample = sample.copy()
        new_sample['past_values'] = past_values
        new_sample['past_observed_mask'] = past_observed_mask
        return new_sample

def train_patchtst_on_dataset(train_set, val_set, test_set, num_input_channels, context_length, future_steps, model_save_path, freeze_layers=False, pretrained_model=None):  # 新增参数
    config = {
        'num_input_channels': num_input_channels,
        'context_length': context_length,
        'do_mask_input': True,
        'num_hidden_layers': 3,
        'd_model': 64,
        'patch_length': 1,
        'patch_stride': 1,
        'nhead': 4,
        'ffn_dim': 64*4,
        'scaling': False,
        'prediction_length': future_steps,
        'dropout': 0.2,
    }
    if pretrained_model is not None:
        model = pretrained_model
    else:
        model = PatchTSTWithSoftmax(PatchTSTConfig(**config))
        # for name, _ in model.named_modules():
        #     print(name)
    if freeze_layers:
        freeze_patchtst_layers(model)
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        disable_tqdm=False,
        lr_scheduler_type='linear',
        per_device_train_batch_size=8,
        # per_device_train_batch_size=2,
        num_train_epochs=1000,
        eval_strategy='epoch',
        save_strategy='epoch',
        label_names=['past_values'],
        logging_steps=1,
        output_dir=str(MODEL_DIR),      # CHANGED: was 'model/'
        logging_dir=str(LOG_DIR),       # CHANGED: was 'transformers_logs'
        load_best_model_at_end=True,
        weight_decay=0.05,
        remove_unused_columns = False    # 关键：保留 future_values
    )
    callback_es = EarlyStoppingCallback(early_stopping_patience=3)
    callbacks = [callback_es]
    # if log_path is not None:
    #     callbacks.append(ConvergenceLogger(log_path, fold_id=fold_id))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        callbacks=callbacks,
    )
    trainer.train()
    if model_save_path:
        trainer.save_model(model_save_path)
    predictions = trainer.predict(test_set)
    pred = predictions.predictions[0]
    pred = F.relu(torch.tensor(pred)).numpy()
    labels = torch.stack([batch['future_values'] for batch in test_set]).numpy()

    # 取对应样本的 past_values，并与未来拼接
    past_values = torch.stack([batch['past_values'] for batch in test_set]).numpy()      # (N, context_length, F)

    # 全序列：pred 用 past 直接拷贝，future 用模型预测；labels 用真实的 past 与 future
    pred_full = np.concatenate([past_values, pred], axis=1)    # (N, context_length+future_steps, F)
    labels_full = np.concatenate([past_values, labels], axis=1)
    return pred_full, labels_full, model

def calculate_metrics(pred, labels):
    mse = np.mean((pred - labels) ** 2)
    r2 = r2_score(labels.flatten(), pred.flatten())
    corr = np.corrcoef(pred.flatten(), labels.flatten())[0, 1]
    return mse, r2, corr

def load_data_from_df(df):
    subject_ids = df['ID']
    timepoints = df['day']
    abu = df[df.columns.difference(['ID', 'day'])]
    return abu, subject_ids, timepoints

def create_dataset(data, subject_ids, timepoints):
    # timeline_length = len(np.unique(timepoints))
    dataset = MicroTSDataset(data, subject_ids, timepoints, forecast = True, future_steps = PATCHTST_FUTURE_STEPS)
    return dataset

def eval_on_dataset(model, dataset, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    preds, labs = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            pv = sample['past_values'].unsqueeze(0).to(device)
            mask = sample['past_observed_mask'].unsqueeze(0).to(device)
            out = model(pv, mask)
            pred = out['prediction_outputs'] if isinstance(out, dict) else getattr(out, 'prediction_outputs', out)
            pred = torch.relu(pred).squeeze(0).detach().cpu().numpy()  # (future_steps, n_features)
            lab = sample['future_values'].detach().cpu().numpy()       # (future_steps, n_features)
            preds.append(pred)
            labs.append(lab)
    pred = np.stack(preds, axis=0)      # (N, future_steps, n_features)
    labels = np.stack(labs, axis=0)     # (N, future_steps, n_features)
    return calculate_metrics(pred, labels)

def slice_by_window(df, start_day, window_len):
    end_day = start_day + window_len - 1
    return df[(df['day'] >= start_day) & (df['day'] <= end_day)].copy(), end_day

ext_results = [] 

# CHANGED: input csv paths
source_full_df = pd.read_csv(DATA_DIR / 'source_hip_keytaxa.csv')
target_full_df = pd.read_csv(DATA_DIR / 'target_face_keytaxa.csv')
val_head_df = pd.read_csv(DATA_DIR / 'val_head.csv')

subjects = target_full_df['ID'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

val_head_abu, val_head_ids, val_head_days = load_data_from_df(val_head_df)
val_head_dataset = create_dataset(val_head_abu, val_head_ids, val_head_days)

ft_rate = 0.6

WINDOW_LEN = 8
START_DAYS = range(1, 15)  # 1-14
MAX_DAY = 21

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
    for start_day in START_DAYS:
        if start_day + WINDOW_LEN - 1 > MAX_DAY:
            continue
        target_df = slice_by_window(target_full_df, start_day, WINDOW_LEN)[0]
        source_df = slice_by_window(source_full_df, start_day, WINDOW_LEN)[0]
        test_subjects = subjects[test_idx]
        train_subjects = subjects[train_idx]

        train_df = target_df[target_df['ID'].isin(train_subjects)].reset_index(drop=True)
        test_df = target_df[target_df['ID'].isin(test_subjects)].reset_index(drop=True)

        n_ft = max(1, int(len(train_subjects) * ft_rate))
        sampled_ft = np.random.choice(train_subjects, n_ft, replace=False)
        ft_train_df = train_df[train_df['ID'].isin(sampled_ft)].reset_index(drop=True)
        # 数据集构建
        ft_abu, ft_ids, ft_days = load_data_from_df(ft_train_df)
        test_abu, test_ids, test_days = load_data_from_df(test_df)
        source_abu, source_ids, source_days = load_data_from_df(source_df)
        
        ft_dataset = create_dataset(ft_abu, ft_ids, ft_days)
        test_dataset = create_dataset(test_abu, test_ids, test_days)
        source_dataset = create_dataset(source_abu, source_ids, source_days)

        train_set, val_set = random_split(source_dataset, [0.8, 0.2])
        mask_train_set = MaskedDataset(train_set, mask_rate=0.15, is_train=True)
        _, _, pretrain_model = train_patchtst_on_dataset(
            mask_train_set, val_set, test_dataset,
            source_dataset.features.shape[0],
            source_dataset.timeline.shape[0] - source_dataset.future_steps,
            source_dataset.future_steps,
            None
        )

        train_set, val_set = random_split(ft_dataset, [0.8, 0.2])
        # Apply timestep masking to the fine-tuning data
        masked_ft_train_set = MaskedDataset(train_set, mask_rate=0.25, is_train=True)
        # print(len(train_set), len(val_set))
        pred, labels, trained_transfer_model = train_patchtst_on_dataset(
            masked_ft_train_set, val_set, test_dataset,
            ft_dataset.features.shape[0],
            ft_dataset.timeline.shape[0] - ft_dataset.future_steps,
            ft_dataset.future_steps,
            None,
            freeze_layers=True,
            pretrained_model=pretrain_model
        )

        mse_ext, r2_ext, corr_ext = eval_on_dataset(trained_transfer_model, val_head_dataset)
        ext_results.append({'fold': fold_idx, 'MSE': mse_ext, 'R2': r2_ext, 'Corr': corr_ext})

ext_df = pd.DataFrame(ext_results)
ext_summary = ext_df.groupby('fold', as_index=False).agg({'MSE': 'mean', 'R2': 'mean', 'Corr': 'mean'})

# CHANGED: output csv path
ext_summary.to_csv(RESULT_DIR / 'mask_val_head.csv', index=False)
