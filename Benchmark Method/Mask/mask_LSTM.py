import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)


def relu_l1_normalize_torch(t: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    t = torch.clamp(t, min=0)
    denom = t.sum(dim=dim, keepdim=True).clamp_min(eps)
    return t / denom

def load_train_test_data(train_df, test_df, known_steps=14):
    def process_file(data):
        microbe_columns = data.columns[2:]  # 从第三列开始是微生物数据
        input_size = len(microbe_columns)
        subjects = data['ID'].unique()

        input_data = []
        target_data = []
        subject_ids = []
        time_stamps = []

        for subject in subjects:
            patient_df = data[data['ID'] == subject]
            patient_values = patient_df.iloc[:, 2:].values  # 提取微生物列
            patient_time = patient_df['day'].values  # 提取时间列

            if patient_values.shape[0] > known_steps:
                input_data.append(patient_values[:known_steps, :])
                target_data.append(patient_values[known_steps:, :])
                subject_ids.append([subject] * (patient_values.shape[0] - known_steps))
                time_stamps.append(patient_time[known_steps:])

        return input_data, target_data, subject_ids, time_stamps, input_size, microbe_columns

    train_result = process_file(train_df)
    test_result = process_file(test_df)

    # 确保 train 和 test 的微生物列和大小一致
    assert train_result[4] == test_result[4]
    assert train_result[5].equals(test_result[5])

    print("加载完成：")
    print(f"训练集样本数: {len(train_result[0])}, 测试集样本数: {len(test_result[0])}")

    return train_result[:4], test_result[:4], train_result[4], train_result[5]

class InterpolationMaskedDataset(Dataset):
    """
    The masking logic is identical to mHolmes, but instead of zeroing out,
    it interpolates the masked values.
    """
    def __init__(self, source_dataset, mask_rate, is_train=True, base_seed=42):
        self.source_dataset = source_dataset
        self.mask_rate = mask_rate
        self.is_train = is_train
        self.base_seed = base_seed

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        inputs, targets = self.source_dataset[idx]

        if not self.is_train or self.mask_rate == 0:
            return inputs, targets

        inputs_clone = inputs.clone()
        L, C = inputs_clone.shape  # Length, Channels

        # 1. IDENTIFY: Use the exact same logic as mHolmes to find mask indices
        num_mask = int(self.mask_rate * L)
        if num_mask == 0:
            return inputs_clone, targets
        
        rng = np.random.default_rng(self.base_seed + idx)
        mask_indices = rng.choice(L, num_mask, replace=False)
        
        # 2. INTERPOLATE/FILL: Handle masked points, including endpoints.
        mask_set = set(mask_indices)
        for i in sorted(mask_indices):
            # Case 1: The first point is masked. Use backward fill.
            if i == 0:
                next_idx = i + 1
                while next_idx in mask_set and next_idx < L:
                    next_idx += 1
                # If the whole sequence is masked, do nothing (leave as original). Otherwise, fill.
                if next_idx < L:
                    inputs_clone[i] = inputs_clone[next_idx]
            
            # Case 2: The last point is masked. Use forward fill.
            elif i == L - 1:
                prev_idx = i - 1
                while prev_idx in mask_set and prev_idx >= 0:
                    prev_idx -= 1
                if prev_idx >= 0:
                    inputs_clone[i] = inputs_clone[prev_idx]

            # Case 3: An internal point is masked. Use linear interpolation.
            else:
                prev_idx, next_idx = i - 1, i + 1
                while prev_idx in mask_set and prev_idx > 0:
                    prev_idx -= 1
                while next_idx in mask_set and next_idx < L - 1:
                    next_idx += 1

                prev_val = inputs_clone[prev_idx]
                next_val = inputs_clone[next_idx]
                
                # Avoid division by zero if a block is masked up to an edge
                if next_idx == prev_idx:
                    continue

                alpha = (i - prev_idx) / (next_idx - prev_idx)
                interpolated_val = prev_val + alpha * (next_val - prev_val)
                inputs_clone[i] = interpolated_val

        # 3. TRAIN: Return the interpolated tensor
        return inputs_clone, targets

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, steps_to_predict):
        outputs = []
        h, c = None, None 
        
        for _ in range(steps_to_predict):
            out, (h, c) = self.lstm(x if len(outputs) == 0 else outputs[-1].unsqueeze(1), (h, c) if h is not None else None)
            out = self.fc(out[:, -1, :]) 
            out = self.relu(out)
            # 新增：输出归一化到相对丰度空间（非负、每步按物种维度求和为1）
            out = torch.softmax(out, dim=-1)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)  

def evaluate_mse(model, val_loader):
    model.eval()
    mse_list = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs, steps_to_predict=targets.size(1))
            mse_list.append(nn.MSELoss()(outputs, targets).item())
    return float(np.mean(mse_list)) if mse_list else float("nan")

def train_lstm_transfer(model,
                        source_loader,
                        target_loader,
                        optimizer,
                        pretrain_epochs,
                        finetune_epochs,
                        val_loader=None,
                        log_path="5_fold_new_transfer_results/epoch-loss_results/face-to-hip/lstm_convergence_log.csv",
                        fold_num=None,
                        patience=3):
    """
    迁移学习两步走：预训练 + 微调
    在每个微调 epoch 结束后，计算验证 MSE 并追加到 lstm_convergence_log.csv
    """
    criterion = nn.MSELoss()
    # 预训练
    model.train()
    for epoch in range(pretrain_epochs):
        total_loss = 0.0
        for inputs, targets in source_loader:
            targets = relu_l1_normalize_torch(targets, dim=-1)
            optimizer.zero_grad()
            outputs = model(inputs, steps_to_predict=targets.size(1))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Pretrain] Epoch {epoch+1}/{pretrain_epochs}, loss={total_loss/len(source_loader):.4f}")

   # 微调并记录验证 MSE + 早停
    logs = []
    best_val = float("inf")
    wait = 0

    for epoch in range(finetune_epochs):
        total_loss = 0.0
        model.train()
        for inputs, targets in target_loader:
            targets = relu_l1_normalize_torch(targets, dim=-1)
            optimizer.zero_grad()
            outputs = model(inputs, steps_to_predict=targets.size(1))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss/len(target_loader)
        # 验证 MSE（若提供 val_loader）
        val_mse = evaluate_mse(model, val_loader) if val_loader is not None else float("nan")
        print(f"[Finetune] Epoch {epoch+1}/{finetune_epochs}, train_loss={avg_train:.4f}, val_mse={val_mse:.4f}")
        logs.append({"fold": fold_num, "epoch": epoch+1, "val_mse": val_mse})

        # Early stopping on validation MSE
        if np.isfinite(val_mse):
            if val_mse + 1e-8 < best_val:
                best_val = val_mse
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"[EarlyStop] No improvement for {patience} epochs. Stopping at epoch {epoch+1}.")
                    break

    # if logs:
    #     os.makedirs(os.path.dirname(log_path), exist_ok=True)
    #     # 指定列顺序
    #     df_new = pd.DataFrame(logs)[["epoch", "val_mse", "fold"]]
        
    #     # 判断是否需要写表头（文件不存在或为空时写表头）
    #     write_header = (not os.path.exists(log_path)) or (os.path.getsize(log_path) == 0)
        
    #     # 使用追加模式 'a' 写入
    #     df_new.to_csv(log_path, mode='a', header=write_header, index=False)
    
    return model

class MicrobiomeDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = [torch.tensor(item, dtype=torch.float32) for item in inputs]
        self.targets = [torch.tensor(item, dtype=torch.float32) for item in targets]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_model(model, train_loader, optimizer, fold_num, epochs):
    model.train()
    losses = [] 
    for epoch in range(epochs):
        total_loss = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, steps_to_predict=targets.size(1))
            loss = nn.MSELoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


def predict_and_save(model, test_loader, microbe_columns, test_subject_ids, test_times, save_path, fold_num, results):
    model.eval()
    predictions = []
    actuals = []
    subject_ids = []
    times = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            # 预测结果
            outputs = model(inputs, steps_to_predict=targets.size(1))
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())

            # 加载实际的 subject_id 和 time 信息
            subject_ids.extend(test_subject_ids[i])
            times.extend(test_times[i])

    # 转换为 2D 数组
    pred_array = np.concatenate([pred.reshape(-1, pred.shape[-1]) for pred in predictions], axis=0)
    actual_array = np.concatenate([act.reshape(-1, act.shape[-1]) for act in actuals], axis=0)
    # # 创建 DataFrame
    # pred_df = pd.DataFrame(pred_array, columns=microbe_columns)
    # actual_df = pd.DataFrame(actual_array, columns=microbe_columns)
    # pred_df.insert(0, "time", times)
    # pred_df.insert(0, "subject_id", subject_ids)
    # actual_df.insert(0, "time", times)
    # actual_df.insert(0, "subject_id", subject_ids)

    # # 保存到文件
    # pred_df.to_csv(f"comparison_model/hip_to_face/hip-to-face_lstm_{fold_num}_pred.csv", index=False)
    # actual_df.to_csv(f"comparison_model/hip_to_face/hip-to-face_lstm_{fold_num}_labels.csv", index=False)

    # mse = mean_squared_error(actual_array, pred_array)
    # r2 = r2_score(actual_array.flatten(), pred_array.flatten())
    #     # 计算平均pearson相关性
    # corr = np.corrcoef(actual_array.flatten(), pred_array.flatten())[0, 1]
    # results.append({
    #     "fold_num": fold_num,
    #     "method": "lstm",
    #     "mse": mse,
    #     "r2": r2,
    #     "corr": corr
    # })
    # return results
WINDOW_LEN = 8
START_DAYS = range(1, 15)  # 1-14
MAX_DAY = 21
def slice_by_window(df, start_day, window_len=WINDOW_LEN):
    end_day = start_day + window_len - 1
    return df[(df['day'] >= start_day) & (df['day'] <= end_day)].copy(), end_day

# 外部验证数据（对齐特征列，去掉不存在的列）
ext_results = []
val_body_df = pd.read_csv('/home/hanjin/projects/my_project/Human cadaver tansfer learning/validation_cohort_data/class_abu_body_reverse.csv')
val_head_df = pd.read_csv('/home/hanjin/projects/my_project/Human cadaver tansfer learning/validation_cohort_data/class_abu_head_reverse.csv')
# 构建外部验证 DataLoader（仅作为“测试”，known_steps与训练一致）
(X_ext, Y_ext, ext_subject_ids, ext_times), _, _, _ = load_train_test_data(
    val_body_df, val_body_df, known_steps= 5
)
ext_dataset = MicrobiomeDataset(X_ext, Y_ext)
ext_loader = DataLoader(ext_dataset, batch_size=1, shuffle=False)

hidden_size = 32
batch_size = 1
learning_rate = 0.001
known_steps = 5  
ft_rate = 0.6
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
full_source_df = pd.read_csv("reverse_face_keytaxa.csv")
full_target_df = pd.read_csv("reverse_hip_keytaxa.csv")
subjects = full_target_df['ID'].unique()




for fold_num, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
    for start_day in START_DAYS:
        if start_day + WINDOW_LEN - 1 > MAX_DAY:
            continue
        source_df = slice_by_window(full_source_df, start_day, WINDOW_LEN)[0]
        target_df = slice_by_window(full_target_df, start_day, WINDOW_LEN)[0]

        #预训练
        src_train, _, input_size, microbe_cols = load_train_test_data(source_df, source_df, known_steps)
        src_train_dataset = MicrobiomeDataset(src_train[0], src_train[1])
        interpolated_src_train = InterpolationMaskedDataset(
            src_train_dataset, 
            mask_rate=0.15,  # Ensure this matches mHolmes
            is_train=True,
            base_seed=42     # Ensure this matches mHolmes
    )
        source_loader = DataLoader(
            interpolated_src_train,
            batch_size=batch_size, shuffle=True
        )

        base_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=input_size)
        base_opt   = optim.Adam(base_model.parameters(), lr=learning_rate)
        train_lstm_transfer(base_model, source_loader, source_loader, base_opt, pretrain_epochs=50, finetune_epochs=0)
        pretrained_weights = {k: v.cpu() for k, v in base_model.state_dict().items()}
        
        #五折划分和微调数据集划分
        train_subjects = subjects[train_idx]
        test_subjects = subjects[test_idx]
        train_data = target_df[target_df['ID'].isin(train_subjects)]
        test_data = target_df[target_df['ID'].isin(test_subjects)]
        n_ft = max(1, int(len(train_subjects) * ft_rate))
        sampled_ft = np.random.choice(train_subjects, n_ft, replace = False)
        ft_train_df = train_data[train_data['ID'].isin(sampled_ft)].reset_index(drop=True)

        ids_all = ft_train_df['ID'].unique()
        rng = np.random.RandomState(42)
        ids_all = rng.permutation(ids_all)
        split_n = int(0.8 * len(ids_all))
        if split_n == len(ids_all) and len(ids_all) > 1:
            split_n -= 1
        train_ids = ids_all[:split_n]
        val_ids = ids_all[split_n:]

        tgt_train = ft_train_df[ft_train_df['ID'].isin(train_ids)].reset_index(drop=True)
        tgt_val   = ft_train_df[ft_train_df['ID'].isin(val_ids)].reset_index(drop=True)

        val_tuple, _, _, _ = load_train_test_data(tgt_val, tgt_val, known_steps)
        val_loader = DataLoader(MicrobiomeDataset(val_tuple[0], val_tuple[1]), batch_size=1, shuffle=False)

        tgt_train, _, _, _ = load_train_test_data(tgt_train, tgt_train, known_steps)
        tgt_train_dataset = MicrobiomeDataset(tgt_train[0], tgt_train[1])
        interpolated_tgt_train = InterpolationMaskedDataset(
            tgt_train_dataset, 
            mask_rate=0.25,  # Ensure this matches mHolmes
            is_train=True,
            base_seed=42     # Ensure this matches mHolmes
    )
        target_loader = DataLoader(
            interpolated_tgt_train,
            batch_size=batch_size, shuffle=True
        )


        model = LSTMModel(input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=input_size)
        model.load_state_dict(pretrained_weights, strict=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model = train_lstm_transfer(
            model,
            source_loader,
            target_loader,
            optimizer,
            pretrain_epochs=0,
            finetune_epochs=150,
            val_loader=val_loader,
            log_path="5_fold_new_transfer_results/epoch-loss_results/face-to-hip/lstm_convergence_log.csv",
            fold_num=fold_num,
            patience=3
        )

        # 外部验证：只计算 pred 与 labels 的 MSE/R2/Corr（未来窗口）
        model.eval()
        preds_ext, labels_ext = [], []
        with torch.no_grad():
            for inputs, targets in ext_loader:
                outputs = model(inputs, steps_to_predict=targets.size(1))
                preds_ext.append(outputs.numpy())
                labels_ext.append(targets.numpy())
        pred_arr = np.concatenate([p.reshape(-1, p.shape[-1]) for p in preds_ext], axis=0)
        lab_arr  = np.concatenate([l.reshape(-1, l.shape[-1]) for l in labels_ext], axis=0)
        mse = mean_squared_error(lab_arr, pred_arr)
        r2  = r2_score(lab_arr.flatten(), pred_arr.flatten())
        corr = np.corrcoef(lab_arr.flatten(), pred_arr.flatten())[0, 1]
        ext_results.append({"fold": fold_num, 'start_day': start_day, "mse": mse, "r2": r2, "corr": corr})

# 五折平均并保存
ext_df = pd.DataFrame(ext_results)
ext_summary = ext_df.groupby('fold', as_index=False).agg({'mse': 'mean', 'r2': 'mean', 'corr': 'mean'})
ext_summary.to_csv('5_fold_new_transfer_results/mask_external_val/lstm_val_body.csv', index=False)