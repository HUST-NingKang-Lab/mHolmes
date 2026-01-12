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
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance

# base_dir = "/home/hanjin/projects/my_project/Human cadaver tansfer learning"
# os.chdir(base_dir)
# data_name = "reverse_hip_selected.csv"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)


# 新增：ReLU + L1 归一化（最后一维）
def relu_l1_normalize(t: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    t = t.clamp_min(0)
    denom = t.sum(dim=dim, keepdim=True).clamp_min(eps)
    return t / denom

def load_train_test_data(train_df, test_df, known_steps=14):
    def process_file(data):
        microbe_columns = data.columns[2:]  
        input_size = len(microbe_columns)
        subjects = data['ID'].unique()

        input_data = []
        target_data = []
        subject_ids = []
        time_stamps = []

        for subject in subjects:
            patient_df = data[data['ID'] == subject]
            patient_values = patient_df.iloc[:, 2:].values  
            patient_time = patient_df['day'].values  

            if patient_values.shape[0] > known_steps:
                input_data.append(patient_values[:known_steps, :])
                target_data.append(patient_values[known_steps:, :])
                subject_ids.append([subject] * (patient_values.shape[0] - known_steps))
                time_stamps.append(patient_time[known_steps:])

        return input_data, target_data, subject_ids, time_stamps, input_size, microbe_columns

    train_result = process_file(train_df)
    test_result = process_file(test_df)

   
    assert train_result[4] == test_result[4]
    assert train_result[5].equals(test_result[5])
    return train_result[:4], test_result[:4], train_result[4], train_result[5]


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
            lstm_in = x if len(outputs) == 0 else outputs[-1].unsqueeze(1)  # 自回归
            out, (h, c) = self.lstm(lstm_in, (h, c) if h is not None else None)
            out = self.fc(out[:, -1, :])
            # ReLU + L1 归一化到物种维度
            out = self.relu(out)
            out = torch.softmax(out, dim=-1)
            outputs.append(out)
        return torch.stack(outputs, dim=1) 

def calculate_glv_residual(abundances, interactions, ri, time_increment):
    if isinstance(interactions, np.ndarray):
        interactions = torch.tensor(interactions, dtype=torch.float32, device=abundances.device)
    if isinstance(ri, np.ndarray):
        ri = torch.tensor(ri, dtype=torch.float32, device=abundances.device)
    
    batch_size, time_steps, num_species = abundances.shape
    residuals = torch.zeros_like(abundances)
    
    for t in range(time_steps):
        for i in range(num_species):
            interaction_row = interactions[i].unsqueeze(0)
            change_per_capita = (ri[i] + torch.sum(abundances[:, t, :] * interaction_row, dim=1)) * time_increment
            change_per_capita = change_per_capita.view(-1)
            predicted_change = abundances[:, t, i] * change_per_capita
            residuals[:, t, i] = predicted_change

    return residuals


def composite_loss(outputs, targets, abundances, interactions, ri, time_increment, data_weight=1.0, physics_weight=1.0):
    data_loss = nn.MSELoss()(outputs, targets)
    glv_residuals = calculate_glv_residual(abundances, interactions, ri, time_increment)
    physics_loss = torch.mean(glv_residuals ** 2)
    total_loss = data_weight * data_loss + physics_weight * physics_loss
    return total_loss


class MicrobiomeDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = [torch.tensor(item, dtype=torch.float32) for item in inputs]
        self.targets = [torch.tensor(item, dtype=torch.float32) for item in targets]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# def train_model(model, train_loader, optimizer, interactions, ri, time_increment, microbe_columns, epochs=80, data_weight=1.0, physics_weight=1.0):
#     model.train()
#     losses = []
#     for epoch in range(epochs):
#         total_loss = 0
#         predictions = []
#         actuals = []
        
#         for inputs, targets in train_loader:
#             # 新增：标签与模型输出空间对齐（非负 + 按物种维度 L1 归一化）
#             targets = relu_l1_normalize(targets, dim=-1)
            
#             optimizer.zero_grad()
#             outputs = model(inputs, steps_to_predict=targets.size(1))
#             loss = composite_loss(outputs, targets, inputs, interactions, ri, time_increment, data_weight, physics_weight)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
            
#             predictions.append(outputs.detach().numpy())
#             actuals.append(targets.numpy())
        
#         avg_loss = total_loss / len(train_loader)
#         losses.append(avg_loss)
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        
        # for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        #     pred_2d = pred.reshape(-1, pred.shape[-1])
        #     actual_2d = actual.reshape(-1, actual.shape[-1])
            
        #     pred_df = pd.DataFrame(pred_2d, columns=microbe_columns)
        #     actual_df = pd.DataFrame(actual_2d, columns=microbe_columns)
            
        #     pred_file = f'result/abundance/train/patient_{i+1}_epoch_{epoch+1}_predicted.csv'
        #     actual_file = f'result/abundance/train/patient_{i+1}_epoch_{epoch+1}_actual.csv'
        #     pred_df.to_csv(pred_file, index=False)
        #     actual_df.to_csv(actual_file, index=False)
    
    # plt.figure()
    # plt.plot(range(1, epochs + 1), losses, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')
    # plt.legend()
    # plt.savefig("result/loss.png")
    # print("Loss curve saved to: result/loss.png")

def predict_and_save(model, test_loader, microbe_columns, test_subject_ids, test_times, save_path, fold_num, results):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    predictions = []
    actuals = []
    subject_ids = []
    times = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            outputs = model(inputs, steps_to_predict=targets.size(1))
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
            subject_ids.extend(test_subject_ids[i])
            times.extend(test_times[i])

    pred_array = np.concatenate([pred.reshape(-1, pred.shape[-1]) for pred in predictions], axis=0)
    actual_array = np.concatenate([act.reshape(-1, act.shape[-1]) for act in actuals], axis=0)
    # pred_array = apply_relu_l1_normalization(pred_array)
    pred_df = pd.DataFrame(pred_array, columns=microbe_columns)
    actual_df = pd.DataFrame(actual_array, columns=microbe_columns)
    pred_df.insert(0, "time", times)
    pred_df.insert(0, "subject_id", subject_ids)
    actual_df.insert(0, "time", times)
    actual_df.insert(0, "subject_id", subject_ids)

    pred_df.to_csv(f"comparison_model/hip_to_face/hip-to-face_pinn_{fold_num}_pred.csv", index=False)
    actual_df.to_csv(f"comparison_model/hip_to_face/hip-to-face_pinn_{fold_num}_labels.csv", index=False)
    # print(f"Prediction and true abundance saved to '{data_name}_pinn_{fold_num}_pred.csv' and '{data_name}_pinn_{fold_num}_labels.csv'.")
    mse = mean_squared_error(actual_array, pred_array)
    r2 = r2_score(actual_array.flatten(), pred_array.flatten())
        # 计算平均pearson相关性
    corr = np.corrcoef(actual_array.flatten(), pred_array.flatten())[0, 1]

    results.append({
        "fold_num": fold_num,
        "method": "pinn",
        "mse": mse,
        "r2": r2,
        "corr": corr
    })
    return results

def evaluate_mse(model: nn.Module, val_loader: DataLoader) -> float:
    model.eval()
    mse_list = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            # 与训练一致：目标做非负+L1归一化
            targets = relu_l1_normalize(targets, dim=-1)
            outputs = model(inputs, steps_to_predict=targets.size(1))
            mse_list.append(nn.MSELoss()(outputs, targets).item())
    return float(np.mean(mse_list)) if mse_list else float("nan")

def train_pinn_transfer(model,
                        source_loader,
                        target_loader,
                        optimizer,
                        interactions,
                        ri,
                        time_increment,
                        microbe_columns,
                        pretrain_epochs=50,
                        finetune_epochs=50,
                        data_weight=1.0,
                        physics_weight=1.0,
                        val_loader=None,
                        fold_num=None,
                        log_path="5_fold_new_transfer_results/epoch-loss_results/face-to-hip/pinn_convergence_log.csv",
                        patience=3):
    """
    迁移学习：先在 source_loader 上预训练，再在 target_loader 上微调
    """
    model.train()
    # 1) 源域预训练
    for epoch in range(pretrain_epochs):
        total_loss = 0
        for inputs, targets in source_loader:
            targets = relu_l1_normalize(targets, dim=-1)
            optimizer.zero_grad()
            outputs = model(inputs, steps_to_predict=targets.size(1))
            loss = composite_loss(outputs, targets, inputs,
                                  interactions, ri, time_increment,
                                  data_weight, physics_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Pretrain] Epoch {epoch+1}/{pretrain_epochs}, loss={total_loss/len(source_loader):.4f}")

    # 2) 目标域微调
    logs = []
    best_val = float("inf")
    wait = 0

    for epoch in range(finetune_epochs):
        total_loss = 0
        model.train()
        for inputs, targets in target_loader:
            targets = relu_l1_normalize(targets, dim=-1)
            optimizer.zero_grad()
            outputs = model(inputs, steps_to_predict=targets.size(1))
            loss = composite_loss(outputs, targets, inputs,
                                  interactions, ri, time_increment,
                                  data_weight, physics_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / max(1, len(target_loader))

        val_mse = evaluate_mse(model, val_loader) if val_loader is not None else float("nan")
        print(f"[Finetune] Epoch {epoch+1}/{finetune_epochs}, train_loss={avg_train:.4f}, val_mse={val_mse:.4f}")
        
        logs.append({
            "epoch": epoch + 1,
            "val_mse": val_mse,
            "fold": fold_num
        })

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

# def save_growth_and_interactions(ri, interactions, microbe_columns):
#     ri_df = pd.DataFrame(ri, index=microbe_columns, columns=['Intrinsic Growth Rate'])
#     ri_file = 'result/glv/intrinsic_growth_rates.csv'
#     ri_df.to_csv(ri_file)
#     print("Intrinsic Growth Rates saved to:", ri_file)
    
#     interactions = (interactions + interactions.T) / 2
#     np.fill_diagonal(interactions, 0)
 
#     interactions_df = pd.DataFrame(interactions, index=microbe_columns, columns=microbe_columns)
#     interactions_file = 'result/glv/species_interactions.csv'
#     interactions_df.to_csv(interactions_file)
#     print("Species Interactions Matrix saved to:", interactions_file)
# 在五折中加入内循环：对 face_df 和 hip_df 做时间窗口（长度8，起点1-14，对 day 列）
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
batch_size = 32
learning_rate = 0.001
epochs = 700
time_increment = 0.1  
known_steps = 5  
ft_rate = 0.6
num_species = 7
interaction_strength = 0.1
growth_rate_range = (-1, 1)

interactions = np.random.uniform(-interaction_strength, interaction_strength, (num_species, num_species))
np.fill_diagonal(interactions, 0)
ri = np.random.uniform(growth_rate_range[0], growth_rate_range[1], num_species)

# full_data = pd.read_csv(data_name)
# subjects = full_data['ID'].unique()
full_source_df = pd.read_csv("reverse_face_keytaxa.csv")
full_target_df = pd.read_csv("reverse_hip_keytaxa.csv")
subjects = full_target_df['ID'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []


for fold_num, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
    for start_day in START_DAYS:
        if start_day + WINDOW_LEN - 1 > MAX_DAY:
            continue
        source_df = slice_by_window(full_source_df, start_day, WINDOW_LEN)[0]
        target_df = slice_by_window(full_target_df, start_day, WINDOW_LEN)[0]

        (X_src, Y_src, _, _), _, input_size, microbe_columns = load_train_test_data(
        source_df, source_df, known_steps=known_steps)
        source_loader = DataLoader(MicrobiomeDataset(X_src, Y_src), batch_size=batch_size, shuffle=True)

        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=input_size)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #pretrain
        model = train_pinn_transfer(
            model,
            source_loader,
            source_loader,  # 占位
            optimizer,
            interactions,
            ri,
            time_increment,
            microbe_columns,
            pretrain_epochs=200,
            finetune_epochs=0
        )
        # 缓存预训练权重
        pretrained_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        train_subjects = subjects[train_idx]
        test_subjects = subjects[test_idx]
        train_data = target_df[target_df['ID'].isin(train_subjects)]
        test_data = target_df[target_df['ID'].isin(test_subjects)]
        n_ft = max(1, int(len(train_subjects) * ft_rate))
        sampled_ft = np.random.choice(train_subjects, n_ft, replace = False)
        ft_train_df = train_data[train_data['ID'].isin(sampled_ft)].reset_index(drop=True)
        
        #划分微调训练集和验证集
        ids_all = ft_train_df['ID'].unique()
        rng = np.random.RandomState(42)
        ids_all = rng.permutation(ids_all)
        split_n = int(0.8 * len(ids_all))
        if split_n == len(ids_all) and len(ids_all) > 1:
            split_n -= 1
        train_ids = ids_all[:split_n]
        val_ids = ids_all[split_n:]
        
        tgt_train_df = ft_train_df[ft_train_df['ID'].isin(train_ids)].reset_index(drop=True)
        tgt_val_df   = ft_train_df[ft_train_df['ID'].isin(val_ids)].reset_index(drop=True)


        train_data_tuple, _, _, microbe_columns = load_train_test_data(
            tgt_train_df, test_data, known_steps=known_steps)

        X_train, Y_train, train_subject_ids, train_times = train_data_tuple
        
        train_dataset = MicrobiomeDataset(X_train, Y_train)
        target_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        # 加载验证集
        val_data_tuple, _, _, _ = load_train_test_data(
            tgt_val_df, tgt_val_df, known_steps=known_steps)
        X_val, Y_val, _, _ = val_data_tuple
        val_dataset = MicrobiomeDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # 3. 加载测试集 (保持不变)
        _, test_data_tuple, _, _ = load_train_test_data(
            test_data, test_data, known_steps=known_steps)
        X_test, Y_test, test_subject_ids, test_times = test_data_tuple
        test_dataset = MicrobiomeDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # model_save_path = f"comparison_model/model/transfer_pinn_model.pth"
        # 新建模型并加载预训练权重；新建优化器（可选择较小 LR 微调）
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=input_size)
        model.load_state_dict(pretrained_weights, strict=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 可调小如 lr=1e-4

        # train_model(model, train_loader, optimizer, interactions, ri, time_increment, microbe_columns, epochs=epochs)
        model = train_pinn_transfer(
            model,
            source_loader,#占位，不使用
            target_loader,
            optimizer,
            interactions,
            ri,
            time_increment,
            microbe_columns,
            pretrain_epochs=0,
            fold_num = fold_num,
            finetune_epochs=150,
            val_loader=val_loader,
            log_path="5_fold_new_transfer_results/epoch-loss_results/face-to-hip/pinn_convergence_log.csv",  # 统一日志路径
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
ext_summary.to_csv('5_fold_new_transfer_results/external_val/pinn_val_body.csv', index=False)


