import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
  

def set_seed(seed=42):
    np.random.seed(seed)

set_seed(42)

def apply_mask_and_interpolation(X, known_steps, num_features, mask_rate, base_seed=42):
    """
    Args:
        X (np.array): 输入数据，形状为 (num_samples, known_steps * num_features)。
        known_steps (int): 每个样本的时间步长。
        num_features (int): 每个时间步的特征数量。
        mask_rate (float): 遮盖比例。
        base_seed (int): 随机种子。
    """
    if mask_rate == 0:
        return X

    X_interpolated = np.copy(X)
    num_samples = X.shape[0]

    for idx in range(num_samples):
        # 1. 将展平的数据恢复为时序形状 (L, C)
        sample_flat = X[idx]
        sample_2d = sample_flat.reshape(known_steps, num_features)
        
        L, C = sample_2d.shape
        
        # 2. 应用与之前完全相同的遮盖和插值逻辑
        num_mask = int(mask_rate * L)
        if num_mask == 0:
            continue
            
        rng = np.random.default_rng(base_seed + idx)
        mask_indices = rng.choice(L, num_mask, replace=False)
        
        sample_clone = np.copy(sample_2d)
        mask_set = set(mask_indices)

        for i in sorted(mask_indices):
            if i == 0:
                next_idx = i + 1
                while next_idx in mask_set and next_idx < L:
                    next_idx += 1
                if next_idx < L:
                    sample_clone[i] = sample_clone[next_idx]
            elif i == L - 1:
                prev_idx = i - 1
                while prev_idx in mask_set and prev_idx >= 0:
                    prev_idx -= 1
                if prev_idx >= 0:
                    sample_clone[i] = sample_clone[prev_idx]
            else:
                prev_idx, next_idx = i - 1, i + 1
                while prev_idx in mask_set and prev_idx > 0:
                    prev_idx -= 1
                while next_idx in mask_set and next_idx < L - 1:
                    next_idx += 1
                
                if next_idx == prev_idx: continue
                
                prev_val = sample_clone[prev_idx]
                next_val = sample_clone[next_idx]
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                interpolated_val = prev_val + alpha * (next_val - prev_val)
                sample_clone[i] = interpolated_val
        
        # 3. 将处理后的数据重新展平并放回结果数组
        X_interpolated[idx] = sample_clone.flatten()
        
    return X_interpolated

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
            corpse_df = data[data['ID'] == subject]
            corpse_values = corpse_df.iloc[:, 2:].values 
            corpse_time = corpse_df['day'].values  

            if corpse_values.shape[0] > known_steps:
                input_data.append(corpse_values[:known_steps, :].flatten())  
                target_data.append(corpse_values[known_steps:, :].flatten())  
                subject_ids.append([subject] * (corpse_values.shape[0] - known_steps))
                time_stamps.append(corpse_time[known_steps:])

        return input_data, target_data, subject_ids, time_stamps, input_size, microbe_columns

    train_result = process_file(train_df)
    test_result = process_file(test_df)

    
    assert train_result[4] == test_result[4]
    assert train_result[5].equals(test_result[5])
    return train_result[:4], test_result[:4], train_result[4], train_result[5]



def train_elasticnet_transfer(X_source, Y_source,
                              X_target, Y_target,
                              model_save_path,
                              alpha=0.5, l1_ratio=0.2,
                              max_iter=30000, tol = 1e-2):
    scaler_X = StandardScaler()
    # 全量数据一起 fit scaler 保证同一标准
    scaler_X.fit(np.vstack([X_source, X_target]))
    Xs, Xt = scaler_X.transform(X_source), scaler_X.transform(X_target)
    scaler_y = StandardScaler()
    scaler_y.fit(np.vstack([Y_source, Y_target]))
    Y_source, Y_target = scaler_y.transform(Y_source), scaler_y.transform(Y_target)
    # warm_start=True 可保留上次 fit 的系数
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                       random_state=42, warm_start=True,
                       max_iter=max_iter, tol=tol)
    # 先在源域训练
    model.fit(Xs, Y_source)
    # 再在目标域微调
    model.fit(Xt, Y_target)

    joblib.dump(model, model_save_path)
    joblib.dump(scaler_X, model_save_path.replace('.joblib', '_scaler.joblib'))
    joblib.dump(scaler_y, model_save_path.replace('.joblib', '_scaler_y.joblib'))
    return model, scaler_X, scaler_y
# def train_elasticnet(X_train, Y_train, model_save_path, alpha=1.0, l1_ratio=0.5):

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)

#     model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
#     model.fit(X_train, Y_train)
#     joblib.dump(model, model_save_path)
#     joblib.dump(scaler, model_save_path.replace('.joblib', '_scaler.joblib'))

#     return model, scaler
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
    val_head_df, val_head_df, known_steps= 5
)


known_steps = 5 
ft_rate = 0.6
# full_data = pd.read_csv(data_name)
full_source_df = pd.read_csv("reverse_hip_keytaxa.csv")
full_target_df = pd.read_csv("reverse_face_keytaxa.csv")
subjects = full_target_df['ID'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold_num, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
    for start_day in START_DAYS:
        if start_day + WINDOW_LEN - 1 > MAX_DAY:
            continue
        source_df = slice_by_window(full_source_df, start_day, WINDOW_LEN)[0]
        target_df = slice_by_window(full_target_df, start_day, WINDOW_LEN)[0]

        src_train, _, _, microbe_cols = load_train_test_data(source_df, source_df, known_steps)
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
        #源域预训练数据
        src_tuple, _, _, _ = load_train_test_data(source_df, source_df, known_steps)
        #目标域微调数据
        tgt_tuple, _, _, _ = load_train_test_data(tgt_train, tgt_train, known_steps)
        #留出测试集
        _,(X_test, Y_test, test_subject_ids, test_times), input_size, microbe_columns = load_train_test_data(test_data, test_data, known_steps)
        X_src, Y_src, _, _, = src_tuple
        X_tgt, Y_tgt, _, _ = tgt_tuple
        #转为numpy数组
        X_src,  Y_src  = map(np.array, (X_src,  Y_src))
        X_tgt,   Y_tgt   = map(np.array, (X_tgt,   Y_tgt))
        X_test, Y_test = map(np.array, (X_test, Y_test))
        X_ext, Y_ext = map(np.array, (X_ext, Y_ext))

        # 对源域数据和目标域数据应用数据增强
        num_features = len(microbe_cols)
        X_src_masked = apply_mask_and_interpolation(X_src, known_steps, num_features, mask_rate=0.15)
        X_tgt_masked = apply_mask_and_interpolation(X_tgt, known_steps, num_features, mask_rate=0.25)

        model_save_path = f"comparison_model/model/trained_transfer_elasticnet.joblib"
        # model, scaler = train_elasticnet(X_train, Y_train, model_save_path, alpha=0.1, l1_ratio=0.5)  # alpha 和 l1_ratio 可以根据需要调整
        model, scaler_X, scaler_y = train_elasticnet_transfer(
        X_src_masked, Y_src,
        X_tgt_masked, Y_tgt,
        model_save_path,
        alpha=0.1, l1_ratio=0.5, max_iter = 100000, tol = 1e-3
        )

        # 预测与保存
        X_test_scaled = scaler_X.transform(X_test)
        predictions = model.predict(X_test_scaled)
        predictions = scaler_y.inverse_transform(predictions)
        predictions = predictions.reshape(-1, len(microbe_columns))
        Y_test = Y_test.reshape(-1, len(microbe_columns))
        test_times_flat = np.concatenate(test_times) if isinstance(test_times, list) else test_times
        test_subject_ids_flat = np.concatenate(test_subject_ids) if isinstance(test_subject_ids, list) else test_subject_ids

        # pred_df = pd.DataFrame(predictions, columns=microbe_columns)
        # actual_df = pd.DataFrame(Y_test, columns=microbe_columns)
        # pred_df.insert(0, "time", test_times_flat)
        # pred_df.insert(0, "subject_id", test_subject_ids_flat)
        # actual_df.insert(0, "time", test_times_flat)
        # actual_df.insert(0, "subject_id", test_subject_ids_flat)

        # pred_df.to_csv(f"comparison_model/face_to_hip/face-to-hip_elasticnet_{fold_num}_pred.csv", index=False)
        # actual_df.to_csv(f"comparison_model/face_to_hip/face-to-hip_elasticnet_{fold_num}_labels.csv", index=False)

        # mse = mean_squared_error(Y_test, predictions)
        # r2 = r2_score(Y_test.flatten(), predictions.flatten())
        # # 计算平均pearson相关性
        # corr = np.corrcoef(Y_test.flatten(), predictions.flatten())[0, 1]

        # results.append({
        #     "fold_num": fold_num,
        #     "method": "elasticnet",
        #     "mse": mse,
        #     "r2": r2,
        #     "corr": corr
        # })
        # 外部验证预测
        X_ext_scaled = scaler_X.transform(X_ext)
        pred_ext = model.predict(X_ext_scaled)
        pred_ext = scaler_y.inverse_transform(pred_ext)

        mse = mean_squared_error(Y_ext, pred_ext)
        r2  = r2_score(Y_ext.flatten(), pred_ext.flatten())
        corr = np.corrcoef(Y_ext.flatten(), pred_ext.flatten())[0, 1]
        ext_results.append({"fold": fold_num, 'start_day': start_day, "mse": mse, "r2": r2, "corr": corr})

# 五折平均并保存
ext_df = pd.DataFrame(ext_results)
ext_summary = ext_df.groupby('fold', as_index=False).agg({'mse': 'mean', 'r2': 'mean', 'corr': 'mean'})
ext_summary.to_csv('5_fold_new_transfer_results/mask_external_val/ElasticNet_val_head.csv', index=False)


