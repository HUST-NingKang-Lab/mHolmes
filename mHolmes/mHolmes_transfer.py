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
PATCHTST_FUTURE_STEPS = 7

# ADD: absolute paths based on this script location (mHolmes/ and corpse_data/ are siblings)
BASE_DIR = Path(__file__).resolve().parent          # .../my_project/mHolmes
PROJECT_DIR = BASE_DIR.parent                       # .../my_project
DATA_DIR = PROJECT_DIR / "corpse_data"              # .../my_project/corpse_data
RESULT_DIR = PROJECT_DIR / "result" / "transfer_model"

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
        self._debug_step = 0
        self.debug_print_batches = 3  # 只打印前3个batch，避免刷屏

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
    if freeze_layers:
        freeze_patchtst_layers(model)
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        disable_tqdm=False,
        lr_scheduler_type='linear',
        per_device_train_batch_size=8,
        num_train_epochs=1000,
        eval_strategy='epoch',
        save_strategy='epoch',
        label_names=['past_values'],
        logging_steps=1,
        output_dir='model/',
        logging_dir='transformers_logs',
        load_best_model_at_end=True,
        weight_decay=0.05,
        remove_unused_columns = False    # 关键：保留 future_values
    )
    callback_es = EarlyStoppingCallback(early_stopping_patience=3)
    callbacks = [callback_es]

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

    return pred, labels, model

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

def save_pred_labels(pred, labels, features, samples, timeline, future_steps, out_prefix):
    pred = pred.reshape(-1, features.shape[0])
    labels = labels.reshape(-1, features.shape[0])
    pred_df = pd.DataFrame(pred, columns=features)
    labels_df = pd.DataFrame(labels, columns=features)
    pred_df.insert(0, 'subject_id', np.repeat(samples, future_steps))
    labels_df.insert(0, 'subject_id', np.repeat(samples, future_steps))
    pred_df.insert(1, 'time', np.tile(timeline[-future_steps:], len(samples)))
    labels_df.insert(1, 'time', np.tile(timeline[-future_steps:], len(samples)))
    pred_df.to_csv(f'{out_prefix}_pred.csv', index=False)
    labels_df.to_csv(f'{out_prefix}_labels.csv', index=False)

# CHANGED: input csv paths (absolute)
source_df = pd.read_csv(DATA_DIR / "source_hip.csv")
target_df = pd.read_csv(DATA_DIR / "target_face.csv")
ft_rate = 0.6

subjects = target_df['ID'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

# CHANGED: output dir creation (absolute)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
    test_subjects = subjects[test_idx]
    train_subjects = subjects[train_idx]

    train_df = target_df[target_df['ID'].isin(train_subjects)].reset_index(drop=True)
    test_df = target_df[target_df['ID'].isin(test_subjects)].reset_index(drop=True)

    n_ft = max(1, int(len(train_subjects) * ft_rate))
    sampled_ft = np.random.choice(train_subjects, n_ft, replace=False)
    ft_train_df = train_df[train_df['ID'].isin(sampled_ft)].reset_index(drop=True)

    ft_abu, ft_ids, ft_days = load_data_from_df(ft_train_df)
    test_abu, test_ids, test_days = load_data_from_df(test_df)
    source_abu, source_ids, source_days = load_data_from_df(source_df)

    ft_dataset = create_dataset(ft_abu, ft_ids, ft_days)
    test_dataset = create_dataset(test_abu, test_ids, test_days)
    source_dataset = create_dataset(source_abu, source_ids, source_days)

    # 先在source_dataset预训练，再在ft_dataset微调
    train_set, val_set = random_split(source_dataset, [0.8, 0.2])
    _, _, pretrain_model = train_patchtst_on_dataset(
        train_set, val_set, test_dataset,
        source_dataset.features.shape[0],
        source_dataset.timeline.shape[0] - source_dataset.future_steps,
        source_dataset.future_steps,
        None
    )
    train_set, val_set = random_split(ft_dataset, [0.8, 0.2])

    pred, labels, trained_transfer_model = train_patchtst_on_dataset(
        train_set, val_set, test_dataset,
        ft_dataset.features.shape[0],
        ft_dataset.timeline.shape[0] - ft_dataset.future_steps,
        ft_dataset.future_steps,
        None,
        freeze_layers=True,
        pretrained_model=pretrain_model
    )
    mse, r2, corr = calculate_metrics(pred, labels)
    save_pred_labels(
        pred, labels,
        test_dataset.features, test_dataset.samples, test_dataset.timeline, test_dataset.future_steps,
        str(RESULT_DIR / f"fold_{fold_idx}")  # CHANGED
    )
    all_results.append({'fold_num': fold_idx, 'MSE': mse, 'R2': r2, 'Corr': corr})

results_df = pd.DataFrame(all_results)
# CHANGED: output csv path (absolute)
results_df.to_csv(RESULT_DIR / "results.csv", index=False)
