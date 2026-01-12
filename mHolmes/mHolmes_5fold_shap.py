import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.spatial.distance import braycurtis
import torch
import torch.nn.functional as F
import shap
import torch.nn as nn
from torch.utils.data import random_split
from src.TimeSeries import MicroTSDataset
from src.utils import seed_everything
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    TrainerCallback,        # 新增
)

seed_everything(42)
PATCHTST_FUTURE_STEPS = 3

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

# 新增：收集每个 epoch 的验证 MSE (eval_loss)
class ConvergenceLogger(TrainerCallback):
    def __init__(self, path, fold_id=None):
        self.path = path
        self.fold_id = fold_id
        self.records = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if 'eval_loss' in metrics:
            self.records.append({
                'epoch': metrics.get('epoch', state.epoch),
                'val_mse': metrics['eval_loss'],
                'fold': self.fold_id
            })

    def on_train_end(self, args, state, control, **kwargs):
        import pandas as pd, os
        df_new = pd.DataFrame(self.records)
        if os.path.exists(self.path):
            try:
                df_old = pd.read_csv(self.path)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            except Exception:
                df_all = df_new
        else:
            df_all = df_new
        df_all.to_csv(self.path, index=False)

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

def train_patchtst_on_dataset(train_set, val_set, test_set, num_input_channels, context_length, future_steps, model_save_path, freeze_layers=False, pretrained_model=None, log_path=None, fold_id=None):  # 新增参数
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
        evaluation_strategy='epoch',
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

def analyze_feature_importance(model, test_set, dataset, device='cuda', target_timestep=7, background_size=50, explain_size=100):
    """使用 SHAP 计算 PatchTST 模型的特征重要性，并输出表格"""
    model.to(device)
    model.eval()

# 收集输入张量：past_values 和 past_observed_mask
    past_values = []
    past_observed_mask = []
    for i in range(len(test_set)):
        sample = test_set[i]
        past_values.append(sample['past_values'].unsqueeze(0))
        past_observed_mask.append(sample['past_observed_mask'].unsqueeze(0))

    past_values = torch.cat(past_values, dim=0).to(device) #（N,L,C）
    past_observed_mask = torch.cat(past_observed_mask, dim=0).to(device) #（N,L,C）
     # 固定 mask 为样本均值（1 x L x C），在 wrapper 中 expand 到 batch
    fixed_mask = past_observed_mask.mean(dim=0, keepdim=True)  # (1, L, C)

    # DEBUG: 打印关键维度与特征列表（便于定位为什么只看到 6 个特征）
    print("DEBUG dataset.features len:", len(getattr(dataset, 'features', [])))
    print("DEBUG past_values.shape:", tuple(past_values.shape))            # (N, L, C)
    print("DEBUG past_observed_mask.shape:", tuple(past_observed_mask.shape))  # (N, L, C)
    print("DEBUG full_input.shape:", tuple(fixed_mask.shape))             
    # optional: 检查模型输出维度
    with torch.no_grad():
        out = model(past_values[:1].to(next(model.parameters()).device), past_observed_mask[:1].to(next(model.parameters()).device))
        print("DEBUG model output type:", type(out))
        try:
            preds = out.get('prediction_outputs', None) if isinstance(out, dict) else getattr(out, 'prediction_outputs', None)
        except Exception:
            preds = out
        if preds is not None:
            print("DEBUG prediction_outputs.shape:", tuple(preds.shape))  # (batch, future_steps, n_features)

    # Wrapper 接受一个输入张量（concat 后），并返回对每个输出特征的标量/向量
    class WrapperModelLocal(nn.Module):
        def __init__(self, base_model, fixed_mask, target_timestep):
            super().__init__()
            self.model = base_model
            self.target_timestep = target_timestep
            self.fixed_mask = fixed_mask

        def forward(self, pv):
            # x shape: (batch, C*2, L); 将其拆回 past_values 和 mask
            mask = self.fixed_mask.expand(pv.size(0), -1, -1).to(pv.device)
            # base model 返回 prediction_outputs: (batch, future_steps, n_features)
            out = self.model(pv, mask)
            # 兼容 dict 和对象两种返回形式
            if isinstance(out, dict):
                preds = out.get('prediction_outputs', None)
            else:
                preds = getattr(out, 'prediction_outputs', None) if hasattr(out, '__dict__') else out

            if preds is None:
                preds = out  # fallback
            # preds expected (batch, future_steps, n_features)
            if torch.is_tensor(preds) and preds.dim() == 3:
                if 0 <= self.target_timestep < preds.shape[1]:
                    return preds[:, self.target_timestep, :]   # (batch, n_features)
                else:
                    return preds.mean(dim=1)
            return preds

    wrapper = WrapperModelLocal(model, fixed_mask, target_timestep=target_timestep).to(device).eval()
     # 采样 background / to_explain（只使用 past_values）
    n = past_values.shape[0]
    bg_n = min(background_size, n)
    expl_n = min(explain_size, n)
    background = past_values[:bg_n].clone().detach()
    to_explain = past_values[:expl_n].clone().detach()

    # 调用 SHAP（保留 try/except 兼容不同 shap 版本）
    try:
        explainer = shap.GradientExplainer((wrapper,), background)  # tuple wrapper for compatibility
        shap_out = explainer.shap_values(to_explain)  # for multi-output returns list length = n_outputs
    except Exception:
        # 备用：尝试直接不使用 tuple
        explainer = shap.GradientExplainer(wrapper, background)
        shap_out = explainer.shap_values(to_explain)

    # shap_out: 若多输出，通常为 list of arrays, 每个 array shape (expl_n, input_channels, seq_len)
    # 将 list 转为 numpy array: (n_outputs, expl_n, input_channels, seq_len)
    if isinstance(shap_out, list):
        shap_arr = np.stack([np.asarray(x) for x in shap_out], axis=0)
        print("DEBUG shap_arr.shape (multi-output):", shap_arr.shape)
    else:
        shap_arr = np.asarray(shap_out)
        # 如果返回形状是 (expl_n, input_channels, seq_len) -> 转为 (1, expl_n, ...)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[np.newaxis, ...]

    # 计算每个输入通道（包括 mask）的全局平均 SHAP 值：平均 over outputs, samples, time
    feat_abundance = (
        past_values.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    )  # 形状 (C,)
    eps = 1e-8
    norm = feat_abundance[np.newaxis, np.newaxis, :, np.newaxis] + eps  # 形状对齐到 (1,1,C,1)
    shap_arr = shap_arr / norm
    mean_over = np.asarray(shap_arr).mean(axis=(0, 3)) #mean_over shape: (n,L,F,m)（样本数，天数，特征，shap值）
    print("DEBUG shap_arr.shape (multi-output):", shap_arr.shape)
    # 只保留前半部分（对应原始 features，而非 observed_mask）
    # input_channels_total = mean_over.shape[0]
    # n_features = input_channels_total
    # feature_importance = mean_over[:n_features]
    input_channels_total = mean_over.shape[1]
    print("DEBUG input_channels_total:", input_channels_total)
    feature_importance = np.abs(mean_over[:14]).mean(axis=0)  # 取绝对值后平均，对前14天进行计算

    # 构建 DataFrame 并排序
    feature_names = list(dataset.features)
    # if len(feature_names) != n_features:
    #     # 如果长度不匹配，尝试截断或补齐名称
    #     feature_names = feature_names[:n_features] + [f'feat_{i}' for i in range(n_features - len(feature_names))]

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Importance': feature_importance
    }).sort_values(by='SHAP Importance', ascending=False).reset_index(drop=True)

    return shap_df

# os.makedirs('5_fold_selected_results', exist_ok=True)
# OUTPUT_DIR = '/home/hanjin/projects/my_project/Human cadaver tansfer learning/5_fold_new_transfer_results/hip_to_face_keytaxa'  # 新的结果目录
full_hip_df = pd.read_csv('/home/hanjin/projects/my_project/Human cadaver tansfer learning/reverse_hip_keytaxa.csv')
full_face_df = pd.read_csv('/home/hanjin/projects/my_project/Human cadaver tansfer learning/reverse_face_keytaxa.csv')
val_body_df = pd.read_csv('/home/hanjin/projects/my_project/Human cadaver tansfer learning/validation_cohort_data/class_abu_body_reverse.csv')
val_head_df = pd.read_csv('/home/hanjin/projects/my_project/Human cadaver tansfer learning/validation_cohort_data/class_abu_head_reverse.csv')


subjects = full_face_df['ID'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

val_head_abu, val_head_ids, val_head_days = load_data_from_df(val_head_df)
val_body_abu, val_body_ids, val_body_days = load_data_from_df(val_body_df)
val_head_dataset = create_dataset(val_head_abu, val_head_ids, val_head_days)
val_body_dataset = create_dataset(val_body_abu, val_body_ids, val_body_days)

ft_rate = 0.6

# 在五折中加入内循环：对 face_df 和 hip_df 做时间窗口（长度8，起点1-14，对 day 列）
WINDOW_LEN = 8
START_DAYS = range(1, 15)  # 1-14
MAX_DAY = 21

def slice_by_window(df, start_day, window_len=WINDOW_LEN):
    end_day = start_day + window_len - 1
    return df[(df['day'] >= start_day) & (df['day'] <= end_day)].copy(), end_day

os.makedirs('5_fold_new_transfer_results/mask_external_val', exist_ok=True)
ext_results = []  # 收集每个时间窗口下5折在外部验证集上的指标

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
    for start_day in START_DAYS:
        if start_day + WINDOW_LEN - 1 > MAX_DAY:
            continue
        face_df = slice_by_window(full_face_df, start_day, WINDOW_LEN)[0]
        hip_df = slice_by_window(full_hip_df, start_day, WINDOW_LEN)[0]
        test_subjects = subjects[test_idx]
        train_subjects = subjects[train_idx]

        train_df = face_df[face_df['ID'].isin(train_subjects)].reset_index(drop=True)
        test_df = face_df[face_df['ID'].isin(test_subjects)].reset_index(drop=True)

        n_ft = max(1, int(len(train_subjects) * ft_rate))
        sampled_ft = np.random.choice(train_subjects, n_ft, replace=False)
        ft_train_df = train_df[train_df['ID'].isin(sampled_ft)].reset_index(drop=True)
        # 数据集构建
        ft_abu, ft_ids, ft_days = load_data_from_df(ft_train_df)
        test_abu, test_ids, test_days = load_data_from_df(test_df)
        hip_abu, hip_ids, hip_days = load_data_from_df(hip_df)
        # face_abu, face_ids, face_days = load_data_from_df(train_df)
        # face_abu, face_ids, face_days = load_data_from_df(face_df)
        # face_dataset = create_dataset(ft_abu, ft_ids, ft_days)
        ft_dataset = create_dataset(ft_abu, ft_ids, ft_days)
        test_dataset = create_dataset(test_abu, test_ids, test_days)
        hip_dataset = create_dataset(hip_abu, hip_ids, hip_days)
        # face_dataset = create_dataset(face_abu, face_ids, face_days)
        # # 方法一：hip model
        # train_set, val_set = random_split(hip_dataset, [0.8, 0.2])
        # pred, labels, foundational_model = train_patchtst_on_dataset(
        #     train_set, val_set, test_dataset,
        #     hip_dataset.features.shape[0],
        #     hip_dataset.timeline.shape[0] - hip_dataset.future_steps,
        #     hip_dataset.future_steps,
        #     None
        # )
        # # #将每一折结果保存
        # # hip_to_hip_preds.append(pred)
        # # hip_to_hip_labels.append(labels)
        # # hip_to_hip_samples.extend(list(test_dataset.samples))

        # try:
        #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #     # 选择要解释的目标步，取平均时间步
        #     target_timestep = -1
        #     shap_df = analyze_feature_importance(foundational_model, test_dataset, hip_dataset, device=device, target_timestep=target_timestep)
        #     out_path = f'5_fold_new_transfer_results/shap_result/hip_to_hip_{fold_idx}_shap.csv'
        #     shap_df.to_csv(out_path, index=False)
        # except Exception as e:
        #     print(f"No transfer SHAP computation failed for fold {fold_idx} ft_rate {ft_rate}: {e}")
        # mse, r2, corr = calculate_metrics(pred, labels)
        # all_results.append({'test_num': fold_idx,  'method': 'hip-to-hip', 'time':f"{12-start_day}-{22-start_day}", 'mse': mse, 'r2': r2, 'corr': corr})
        # # bc = bray_curtis_sum(pred, labels)
        # save_pred_labels(pred, labels, face_dataset.features, test_dataset.samples, test_dataset.timeline, test_dataset.future_steps,
        #                     f'5_fold_new_transfer_results/face_to_hip_keytaxa/hip-to-hip_model_{fold_idx}')
        # all_results.append({'test_num': fold_idx, 'ft_rate': ft_rate, 'method': 'hip-to-hip_model', 'mse': mse, 'r2': r2, 'corr': corr})
        # # 方法二：mix model
        # mix_df = pd.concat([face_df, ft_train_df], ignore_index=True)
        # mix_abu, mix_ids, mix_days = load_data_from_df(mix_df)
        # mix_dataset = create_dataset(mix_abu, mix_ids, mix_days)
        # train_set, val_set = random_split(mix_dataset, [0.8, 0.2])
        # pred, labels, _ = train_patchtst_on_dataset(
        #     train_set, val_set, test_dataset,
        #     mix_dataset.features.shape[0],
        #     mix_dataset.timeline.shape[0] - mix_dataset.future_steps,
        #     mix_dataset.future_steps,
        #     None
        # )
        # mse, r2, corr = calculate_metrics(pred, labels)
        # # bc = bray_curtis_sum(pred, labels)
        # save_pred_labels(pred, labels, mix_dataset.features, test_dataset.samples, test_dataset.timeline, test_dataset.future_steps,
        #                     f'5_fold_new_transfer_results/face_to_hip_results/mix-to-hip_model_{fold_idx}_{ft_rate}')
        # all_results.append({'test_num': fold_idx, 'ft_rate': ft_rate, 'method': 'mix-to-hip_model', 'mse': mse, 'r2': r2, 'corr': corr})
        # # 方法三：face model
        # train_set, val_set = random_split(face_dataset, [0.8, 0.2])
        # print(len(train_set), len(val_set))
        # pred, labels, foundational_model = train_patchtst_on_dataset(
        #     train_set, val_set, test_dataset,
        #     face_dataset.features.shape[0],
        #     face_dataset.timeline.shape[0] - face_dataset.future_steps,
        #     face_dataset.future_steps,
        #     None
        # )
        # face_to_face_preds.append(pred)
        # face_to_face_labels.append(labels)
        # face_to_face_samples.extend(list(test_dataset.samples))
        # try:
        #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #     # 选择要解释的目标步，取平均时间步
        #     target_timestep = -1
        #     shap_df = analyze_feature_importance(foundational_model, test_dataset, face_dataset, device=device, target_timestep=target_timestep)
        #     out_path = f'5_fold_new_transfer_results/shap_result/face_to_face_{fold_idx}_shap.csv'
        #     shap_df.to_csv(out_path, index=False)
        # except Exception as e:
        #     print(f"No transfer SHAP computation failed for fold {fold_idx} ft_rate {ft_rate}: {e}")
        # mse, r2, corr = calculate_metrics(pred, labels)
        # # # bc = bray_curtis_sum(pred, labels)
        # save_pred_labels(pred, labels, face_dataset.features, test_dataset.samples, test_dataset.timeline, test_dataset.future_steps,
        #                     f'5_fold_new_transfer_results/hip_to_face_keytaxa/face-to-face_model_{fold_idx}')
        # all_results.append({'test_num': fold_idx,  'method': 'face-to-face', 'time':f"{12-start_day}-{22-start_day}", 'mse': mse, 'r2': r2, 'corr': corr})
        # all_results.append({'test_num': fold_idx, 'method': 'face-to-face_model', 'mse': mse, 'r2': r2, 'corr': corr})
        # 方法四：transfer model
        # 先在hip_dataset预训练，再在train_dataset微调
        train_set, val_set = random_split(hip_dataset, [0.8, 0.2])
        mask_train_set = MaskedDataset(train_set, mask_rate=0.15, is_train=True)
        _, _, pretrain_model = train_patchtst_on_dataset(
            mask_train_set, val_set, test_dataset,
            hip_dataset.features.shape[0],
            hip_dataset.timeline.shape[0] - hip_dataset.future_steps,
            hip_dataset.future_steps,
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
            pretrained_model=pretrain_model,
            log_path="5_fold_new_transfer_results/epoch-loss_results/hip-to-face/mHolmes_convergence_log.csv",   # 新增：保存验证 MSE
            fold_id=fold_idx                  # 新增：折编号
        )
        # # 计算并保存 SHAP 特征重要性（对迁移后的模型）
        # try:
        #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #     # 选择要解释的目标步，取平均时间步
        #     target_timestep = -1
        #     shap_df = analyze_feature_importance(trained_transfer_model, test_dataset, face_dataset, device=device, target_timestep=target_timestep)
        #     out_path = f'5_fold_new_transfer_results/shap_result/hip_to_face_{fold_idx}_shap.csv'
        #     shap_df.to_csv(out_path, index=False)
        # except Exception as e:
        #     print(f"Transfer SHAP computation failed for fold {fold_idx} ft_rate {ft_rate}: {e}")
        # mse, r2, corr = calculate_metrics(pred, labels)
        # # # bc = bray_curtis_sum(pred, labels)
        # save_pred_labels(pred, labels, test_dataset.features, test_dataset.samples, test_dataset.timeline, test_dataset.future_steps,
        #                     f'5_fold_new_transfer_results/face_to_hip_keytaxa/transfer_model_{fold_idx}')
        # all_results.append({'test_num': fold_idx, 'mask_rate':mask_rate, 'method': 'transfer_model', 'mse': mse, 'r2': r2, 'corr': corr})
        # 外部验证：直接用 val_head_dataset（不做窗口划分）
        mse_ext, r2_ext, corr_ext = eval_on_dataset(trained_transfer_model, val_head_dataset)
        ext_results.append({'fold': fold_idx, 'start_day': start_day, 'mse': mse_ext, 'r2': r2_ext, 'corr': corr_ext})
    # all_results.append({'test_num': fold_idx,  'method': 'transfer_model', 'mse': mse, 'r2': r2, 'corr': corr})

# 按窗口聚合外部验证集的5折平均指标并保存
ext_df = pd.DataFrame(ext_results)
ext_summary = ext_df.groupby('fold', as_index=False).agg({'mse': 'mean', 'r2': 'mean', 'corr': 'mean'})
ext_summary.to_csv('5_fold_new_transfer_results/mask_external_val/mHolmes_mask_val_head.csv', index=False)

# # 统一按窗口保存
# save_pred_labels(
#     hip_to_hip_preds, hip_to_hip_labels,
#     hip_dataset.features, np.array(hip_to_hip_samples),
#     test_dataset.timeline, PATCHTST_FUTURE_STEPS,
#     f'{OUTPUT_DIR}/hip-to-hip_{start_day}'
# )

# save_pred_labels(
#     face_to_face_preds, face_to_face_labels,
#     face_dataset.features, np.array(face_to_face_samples),
#     test_dataset.timeline, PATCHTST_FUTURE_STEPS,
#     f'{OUTPUT_DIR}/face-to-face_{start_day}'
# )
