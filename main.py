import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

# 多层感知机模型
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super(MLPModel, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self(X).numpy()

# LSTM模型，适合时序数据
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # 将输入重塑为(batch, seq_len=1, features)用于LSTM
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out.squeeze(1))
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self(X).numpy()

# 简单线性模型
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self(X).numpy()

# 集成模型
class EnsembleModel:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
                model.eval()
                with torch.no_grad():
                    pred = model(X_tensor).numpy()
            else:
                pred = model.predict(X)
            predictions.append(pred)
        return np.mean(predictions, axis=0)

# 自定义损失函数 - 更接近比赛评分标准
class RelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(RelativeErrorLoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        # 确保预测值和真实值是正数
        y_pred = torch.clamp(y_pred, min=self.epsilon)
        y_true = torch.clamp(y_true, min=self.epsilon)
        
        # 计算相对误差 |pred - true| / true
        relative_error = torch.abs(y_pred - y_true) / (y_true + self.epsilon)
        
        # 使用1-相对误差作为优化目标，与评分标准一致
        return torch.mean(relative_error)  # 注意：优化器会最小化，所以直接返回相对误差

# 添加组合损失函数，平衡绝对误差和相对误差
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, epsilon=1e-8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # 相对误差权重
        self.epsilon = epsilon
        self.mse = nn.MSELoss()
        self.rel_error = RelativeErrorLoss(epsilon)
        
    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        rel_loss = self.rel_error(y_pred, y_true)
        return self.alpha * mse_loss + self.beta * rel_loss

# 专门处理风电场和太阳能电场的损失函数
class RenewablePowerLoss(nn.Module):
    def __init__(self, is_wind_farm=True, epsilon=1e-8):
        super(RenewablePowerLoss, self).__init__()
        self.is_wind_farm = is_wind_farm
        self.epsilon = epsilon
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # 先计算基本的MSE损失
        mse_loss = self.mse(y_pred, y_true)
        
        # 确保预测值和真实值是正数
        y_pred = torch.clamp(y_pred, min=self.epsilon)
        y_true = torch.clamp(y_true, min=self.epsilon)
        
        # 计算相对误差
        relative_error = torch.abs(y_pred - y_true) / (y_true + self.epsilon)
        
        # 根据场站类型和值大小进行权重调整
        if self.is_wind_farm:
            # 风电场：对非常小的值给予更少的权重，以避免它们过度影响损失
            weight = torch.clamp(y_true, min=0.1)  # 小值有较小权重
            weighted_rel_error = relative_error * weight
            rel_loss = torch.sum(weighted_rel_error) / torch.sum(weight)
            return 0.5 * mse_loss + 0.5 * rel_loss
        else:
            # 太阳能电场：夜间为0的时段应该有不同处理
            # 判断哪些是夜间（真实值接近0的）
            night_mask = y_true < 0.05
            if torch.any(night_mask):
                # 夜间：主要用MSE
                day_mask = ~night_mask
                if torch.any(day_mask):
                    # 白天：结合MSE和相对误差
                    day_rel_error = torch.mean(relative_error[day_mask])
                    return 0.3 * mse_loss + 0.7 * day_rel_error
                return mse_loss
            # 全部是白天数据
            return 0.3 * mse_loss + 0.7 * torch.mean(relative_error)

nwps = ['NWP_1', 'NWP_2', 'NWP_3']
fact_path = 'training/middle_school/TRAIN/fact_data'

def data_preprocess(x_df, y_df):
    x_df = x_df.dropna()
    y_df = y_df.dropna()
    ind = [i for i in y_df.index if i in x_df.index]
    x_df = x_df.loc[ind]
    y_df = y_df.loc[ind]
    return x_df, y_df

def add_time_features(df):
    """添加增强的时间特征"""
    # 小时特征
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # 日期特征（年中的天）
    df['dayofyear'] = df.index.dayofyear
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # 月份特征
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    return df

def extract_features(nwp_data, is_wind_farm=True, num_days=365):
    """提取特定类型场站的相关特征"""
    features_dict = {}
    channels = list(nwp_data.channel.values)
    
    # 提取基本风速特征（风电场和太阳能电场都需要）
    u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                    channel=['u100']).data.values.reshape(num_days * 24, 9)
    v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                    channel=['v100']).data.values.reshape(num_days * 24, 9)
    
    # 计算风速和风向
    ws = np.sqrt(u**2 + v**2)
    wd = np.arctan2(v, u) * 180 / np.pi
    
    features_dict['u'] = u
    features_dict['v'] = v
    features_dict['ws'] = ws
    features_dict['wd'] = wd
    
    # 提取风电场特有特征
    if is_wind_farm:
        # 计算风速方差（作为湍流强度代理）
        features_dict['ws_var'] = np.var(ws, axis=1).reshape(-1, 1)
        
        # 添加风速平方项（功率与风速的平方成正比）
        features_dict['ws_squared'] = ws**2
        
        # 添加风向正弦余弦分量
        features_dict['wd_sin'] = np.sin(wd * np.pi / 180)
        features_dict['wd_cos'] = np.cos(wd * np.pi / 180)
        
        # 添加温度如果有（影响空气密度）
        if 't2m' in channels:
            t2m = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                            channel=['t2m']).data.values.reshape(num_days * 24, 9)
            features_dict['t2m'] = t2m
    
    # 提取太阳能电场特有特征
    else:
        # 添加辐射相关特征
        if 'ghi' in channels:
            ghi = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                             channel=['ghi']).data.values.reshape(num_days * 24, 9)
            features_dict['ghi'] = ghi
            
            # 辐射方差表示云层不均匀性
            features_dict['ghi_var'] = np.var(ghi, axis=1).reshape(-1, 1)
        
        if 'poai' in channels:
            poai = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                               channel=['poai']).data.values.reshape(num_days * 24, 9)
            features_dict['poai'] = poai
        
        # 云量对太阳能发电影响很大
        if 'tcc' in channels:
            tcc = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                              channel=['tcc']).data.values.reshape(num_days * 24, 9)
            features_dict['tcc'] = tcc
            
            # 云量平方项，强调大云量的影响
            features_dict['tcc_squared'] = tcc**2
        
        # 温度影响太阳能电池板效率
        if 't2m' in channels:
            t2m = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                              channel=['t2m']).data.values.reshape(num_days * 24, 9)
            features_dict['t2m'] = t2m
            
            # 高温会降低太阳能面板效率
            features_dict['t2m_sq'] = t2m**2
    
    return features_dict

def create_feature_dataframe(features_dict, nwp):
    """创建特征DataFrame"""
    dataframes = []
    
    for feature_name, feature_array in features_dict.items():
        if len(feature_array.shape) == 1 or feature_array.shape[1] == 1:
            df = pd.DataFrame(feature_array, columns=[f"{nwp}_{feature_name}"])
        else:
            df = pd.DataFrame(feature_array, 
                             columns=[f"{nwp}_{feature_name}_{i}" for i in range(feature_array.shape[1])])
        dataframes.append(df)
    
    return pd.concat(dataframes, axis=1)

def calculate_accuracy(y_true, y_pred):
    """计算竞赛评分标准下的准确率"""
    # 避免除以0，同时保持非常小的值
    y_true_safe = np.maximum(y_true, 1e-8)
    y_pred_safe = np.clip(y_pred, 0, 1) 
    
    # 单个样本的准确率
    sample_acc = 1 - np.abs(y_pred_safe - y_true_safe) / y_true_safe
    
    # 过滤掉极端异常值（相对误差超过10倍的视为异常）
    abnormal = np.abs(y_pred_safe - y_true_safe) / y_true_safe > 10
    
    # 如果全部是异常值，则返回最糟糕的准确率
    if np.all(abnormal):
        return -9.0  # 一个合理的负值作为最糟糕情况
    
    # 只计算非异常值的平均准确率
    accuracy = np.mean(sample_acc[~abnormal])
    
    return accuracy

def train_model(model, X_train, y_train, X_val, y_val, is_wind_farm=True, num_epochs=200, batch_size=128, lr=0.001):
    """训练单个PyTorch模型"""
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 使用专门的损失函数
    criterion = RenewablePowerLoss(is_wind_farm=is_wind_farm)
    
    # 学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练循环
    best_val_acc = float('-inf')  # 改为跟踪最佳准确率
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 使用专门的损失函数
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            
            val_preds.extend(val_outputs.cpu().numpy().flatten())
            val_targets.extend(y_val.cpu().numpy().flatten())
        
        # 计算竞赛评分标准下的准确率
        val_acc = calculate_accuracy(np.array(val_targets), np.array(val_preds))
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # 使用验证集准确率来调整学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型（基于准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience or optimizer.param_groups[0]['lr'] < 1e-6:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 计算最终验证集准确率
    model.eval()
    with torch.no_grad():
        final_val_preds = model(X_val).cpu().numpy().flatten()
    final_val_acc = calculate_accuracy(y_val.cpu().numpy().flatten(), final_val_preds)
    
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    
    return model, final_val_acc

def train(farm_id):
    # 确定风电场或太阳能电场
    is_wind_farm = farm_id <= 5
    print(f"Training {'wind' if is_wind_farm else 'solar'} farm {farm_id}")
    
    # 处理训练数据
    x_df = pd.DataFrame()
    nwp_train_path = f'training/middle_school/TRAIN/nwp_data_train/{farm_id}'
    
    for nwp in nwps:
        nwp_path = os.path.join(nwp_train_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        
        # 提取特征
        features_dict = extract_features(nwp_data, is_wind_farm, num_days=365)
        nwp_df = create_feature_dataframe(features_dict, nwp)
        x_df = pd.concat([x_df, nwp_df], axis=1)
    
    x_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')
    
    # 添加时间特征
    x_df = add_time_features(x_df)
    
    # 加载目标数据
    y_df = pd.read_csv(os.path.join(fact_path, f'{farm_id}_normalization_train.csv'), index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']
    
    # 数据预处理
    x_processed, y_processed = data_preprocess(x_df, y_df)
    y_processed[y_processed < 0] = 0
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x_processed.values)
    
    # 划分训练集和验证集（按时间顺序）
    train_size = int(0.8 * len(X_scaled))
    X_train = torch.tensor(X_scaled[:train_size], dtype=torch.float32)
    y_train = torch.tensor(y_processed.values[:train_size], dtype=torch.float32)
    X_val = torch.tensor(X_scaled[train_size:], dtype=torch.float32)
    y_val = torch.tensor(y_processed.values[train_size:], dtype=torch.float32)
    
    # 训练不同模型
    input_dim = X_train.shape[1]
    model_results = {}
    
    # 1. 线性模型
    print("Training Linear model...")
    linear_model = LinearModel(input_dim)
    linear_model, linear_acc = train_model(linear_model, X_train, y_train, X_val, y_val, is_wind_farm=is_wind_farm, num_epochs=100, batch_size=128, lr=0.01)
    model_results['linear'] = (linear_model, linear_acc)
    print(f"Linear model validation accuracy: {linear_acc:.4f}")
    
    # 2. 小型MLP
    print("Training small MLP...")
    mlp_small = MLPModel(input_dim, [64, 32])
    mlp_small, mlp_small_acc = train_model(mlp_small, X_train, y_train, X_val, y_val, is_wind_farm=is_wind_farm)
    model_results['mlp_small'] = (mlp_small, mlp_small_acc)
    print(f"Small MLP validation accuracy: {mlp_small_acc:.4f}")
    
    # 3. 中型MLP
    print("Training medium MLP...")
    mlp_medium = MLPModel(input_dim, [128, 64])
    mlp_medium, mlp_medium_acc = train_model(mlp_medium, X_train, y_train, X_val, y_val, is_wind_farm=is_wind_farm)
    model_results['mlp_medium'] = (mlp_medium, mlp_medium_acc)
    print(f"Medium MLP validation accuracy: {mlp_medium_acc:.4f}")
    
    # 4. LSTM模型（如果数据量足够）
    if len(X_train) > 1000:
        print("Training LSTM model...")
        lstm_model = LSTMModel(input_dim, hidden_dim=128)
        lstm_model, lstm_acc = train_model(lstm_model, X_train, y_train, X_val, y_val, is_wind_farm=is_wind_farm)
        model_results['lstm'] = (lstm_model, lstm_acc)
        print(f"LSTM model validation accuracy: {lstm_acc:.4f}")
    
    # 选择最佳单一模型
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k][1])
    best_single_model, best_acc = model_results[best_model_name]
    print(f"Best single model: {best_model_name} with accuracy {best_acc:.4f}")
    
    # 创建集成模型（选择表现前两名的模型）
    sorted_models = sorted(model_results.items(), key=lambda x: x[1][1], reverse=True)
    top_models = [model for name, (model, acc) in sorted_models[:2]]
    
    ensemble = EnsembleModel(top_models)
    
    # 存储元数据
    ensemble.scaler = scaler
    ensemble.feature_columns = list(x_processed.columns)
    ensemble.is_wind_farm = is_wind_farm
    ensemble.model_names = [name for name, _ in sorted_models[:2]]
    
    return ensemble

def predict(model, farm_id):
    # 确定风电场或太阳能电场
    is_wind_farm = hasattr(model, 'is_wind_farm') and model.is_wind_farm
    
    # 处理测试数据
    x_df = pd.DataFrame()
    nwp_test_path = f'training/middle_school/TEST/nwp_data_test/{farm_id}'
    
    for nwp in nwps:
        nwp_path = os.path.join(nwp_test_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        
        # 提取特征
        features_dict = extract_features(nwp_data, is_wind_farm, num_days=31)
        nwp_df = create_feature_dataframe(features_dict, nwp)
        x_df = pd.concat([x_df, nwp_df], axis=1)
    
    x_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')
    
    # 添加时间特征
    x_df = add_time_features(x_df)
    
    # 对齐特征列
    if hasattr(model, 'feature_columns'):
        common_columns = set(x_df.columns) & set(model.feature_columns)
        missing_columns = set(model.feature_columns) - common_columns
        
        # 添加缺失的列
        for col in missing_columns:
            x_df[col] = 0
        
        # 选择模型需要的列并按顺序排列
        x_df = x_df[model.feature_columns]
    
    # 标准化特征
    X_scaled = model.scaler.transform(x_df.values) if hasattr(model, 'scaler') else x_df.values
    
    # 使用模型进行预测
    if hasattr(model, 'predict'):
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_pw = model.predict(X_scaled).flatten()
    else:
        # 集成模型处理
        pred_pw = model.predict(X_scaled).flatten()
    
    # 确保预测值在有效范围内
    pred_pw = np.clip(pred_pw, 0, 1)
    
    # 创建小时级别的预测序列
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0], periods=len(pred_pw), freq='h'))
    
    # 插值到15分钟间隔
    res = pred.resample('15min').interpolate(method='linear')
    
    # 应用后处理规则 - 太阳能电场
    if not is_wind_farm:
        # 获取日期和小时
        hours = res.index.hour
        
        # 夜间值设为0（以当地日出日落时间为准）
        night_mask = (hours <= 5) | (hours >= 19)
        res.loc[night_mask] = 0
        
        # 日出时段（5:00-8:00）应该有平滑的增长
        dawn_hours = (hours >= 5) & (hours < 8)
        dawn_df = res[dawn_hours].copy()
        
        if not dawn_df.empty:
            # 对每天分别处理
            for day in set(dawn_df.index.date):
                day_mask = dawn_df.index.date == day
                day_values = dawn_df[day_mask]
                
                if not day_values.empty:
                    # 创建平滑的增长曲线
                    start_idx = day_values.index[0]
                    end_idx = day_values.index[-1]
                    start_val = max(0, day_values.iloc[0])
                    
                    # 查找8点的值，如果没有则使用下一个可用值
                    next_hour_idx = res.index.searchsorted(pd.Timestamp(day).replace(hour=8))
                    if next_hour_idx < len(res):
                        end_val = res.iloc[next_hour_idx]
                    else:
                        end_val = max(0.1, day_values.iloc[-1])
                    
                    # 创建线性增长曲线
                    t = np.linspace(0, 1, len(day_values))
                    smooth_curve = start_val + t * (end_val - start_val)
                    
                    # 应用到原始数据 - 确保数据类型兼容
                    res.loc[day_values.index] = smooth_curve.astype(res.dtype)
        
        # 日落时段（17:00-19:00）应该有平滑的下降
        dusk_hours = (hours >= 17) & (hours < 19)
        dusk_df = res[dusk_hours].copy()
        
        if not dusk_df.empty:
            # 对每天分别处理
            for day in set(dusk_df.index.date):
                day_mask = dusk_df.index.date == day
                day_values = dusk_df[day_mask]
                
                if not day_values.empty:
                    # 创建平滑的下降曲线
                    start_idx = day_values.index[0]
                    end_idx = day_values.index[-1]
                    
                    # 查找17点的值
                    prev_hour_idx = res.index.searchsorted(pd.Timestamp(day).replace(hour=17)) - 1
                    if prev_hour_idx >= 0:
                        start_val = res.iloc[prev_hour_idx]
                    else:
                        start_val = max(0.1, day_values.iloc[0])
                    
                    # 确保平滑下降到0
                    t = np.linspace(0, 1, len(day_values))
                    smooth_curve = start_val * (1 - t)
                    
                    # 应用到原始数据 - 确保数据类型兼容
                    res.loc[day_values.index] = smooth_curve.astype(res.dtype)
    
    # 应用后处理规则 - 风电场
    else:
        # 风电场的输出不应该有明显的零值
        # 查找潜在的零值和异常低值
        zero_mask = res < 0.01
        
        if np.any(zero_mask):
            # 找到序列中最小的非零值作为替代值
            min_nonzero = 0.01  # 默认最小值
            nonzero_values = res[~zero_mask]
            
            if len(nonzero_values) > 0:
                # 使用非零值的第5百分位数
                percentile_5 = np.percentile(nonzero_values, 5)
                min_nonzero = max(0.01, percentile_5 * 0.5)  # 确保至少为0.01
            
            # 将零值替换为最小非零值
            if isinstance(res, pd.Series):
                # 直接获取res的dtype
                res_dtype = res.dtype
                # 使用numpy的类型转换
                min_nonzero_typed = np.array(min_nonzero).astype(res_dtype)
                # 将标量值赋给Series
                res.loc[zero_mask] = min_nonzero_typed.item()
        
        # 平滑处理 - 消除异常峰值和谷值
        window_size = 4  # 1小时
        res_smoothed = res.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # 找出异常波动（与平滑值差距过大的点）
        anomaly_mask = np.abs(res - res_smoothed) > 0.2 * res_smoothed
        
        # 仅替换异常点，保留正常波动，确保数据类型兼容
        if np.any(anomaly_mask):
            # 确保类型兼容
            replacement_values = res_smoothed.loc[anomaly_mask].astype(res.dtype)
            res.loc[anomaly_mask] = replacement_values
    
    # 最终确保所有预测在合理范围内
    res = np.clip(res, 0, 1)
    
    return res

# 主程序执行
farms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for farm_id in farms:
    try:
        print(f"\n==== Processing farm {farm_id} ====")
        model = train(farm_id)
        
        # 保存模型
        model_path = f'models/{farm_id}'
        os.makedirs(model_path, exist_ok=True)
        model_name = 'baseline_middle_school.pkl'  # 与原始代码保持一致的名称
        
        print(f"Saving ensemble model with {model.model_names} for farm {farm_id}")
        with open(os.path.join(model_path, model_name), "wb") as f:
            pickle.dump(model, f)
        
        # 生成预测
        print(f"Generating predictions for farm {farm_id}")
        pred = predict(model, farm_id)
        
        # 保存预测结果
        result_path = f'result/output'
        os.makedirs(result_path, exist_ok=True)
        pred.to_csv(os.path.join(result_path, f'output{farm_id}.csv'))
        print(f'Successfully processed farm {farm_id}')
        
    except Exception as e:
        print(f"Error processing farm {farm_id}: {str(e)}")
        
        # 回退到简单线性模型
        try:
            print(f"Using fallback linear model for farm {farm_id}")
            is_wind_farm = farm_id <= 5
            
            # 处理训练数据
            x_df = pd.DataFrame()
            nwp_train_path = f'training/middle_school/TRAIN/nwp_data_train/{farm_id}'
            
            for nwp in nwps:
                nwp_path = os.path.join(nwp_train_path, nwp)
                nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
                u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['u100']).data.values.reshape(365 * 24, 9)
                v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['v100']).data.values.reshape(365 * 24, 9)
                u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
                v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
                ws = np.sqrt(u**2 + v**2)
                ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
                nwp_df = pd.concat([u_df, v_df, ws_df], axis=1)
                x_df = pd.concat([x_df, nwp_df], axis=1)
            
            x_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')
            x_df = add_time_features(x_df)
            
            y_df = pd.read_csv(os.path.join(fact_path, f'{farm_id}_normalization_train.csv'), index_col=0)
            y_df.index = pd.to_datetime(y_df.index)
            y_df.columns = ['power']
            
            x_processed, y_processed = data_preprocess(x_df, y_df)
            y_processed[y_processed < 0] = 0
            
            # 使用简单线性PyTorch模型
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(x_processed.values)
            X = torch.tensor(X_scaled, dtype=torch.float32)
            y = torch.tensor(y_processed.values, dtype=torch.float32)
            
            model = LinearModel(X.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            # 简单训练
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            # 保存模型
            model.scaler = scaler
            model.feature_columns = list(x_processed.columns)
            model.is_wind_farm = is_wind_farm
            
    model_path = f'models/{farm_id}' 
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, 'baseline_middle_school.pkl'), "wb") as f:
        pickle.dump(model, f)
            
            # 预测
            pred = predict(model, farm_id)
            
            result_path = f'result/output'
            os.makedirs(result_path, exist_ok=True)
            pred.to_csv(os.path.join(result_path, f'output{farm_id}.csv'))
            print(f'Successfully processed farm {farm_id} with fallback model')
            
        except Exception as e2:
            print(f"Fallback model also failed for farm {farm_id}: {str(e2)}")
            
            # 最后的补救措施 - 生成简单模式
            try:
                print(f"Generating simple pattern for farm {farm_id}")
                is_wind_farm = farm_id <= 5
                dates = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23, 45), freq='15min')
                
                if is_wind_farm:
                    # 风电场：创建基本的周期性模式
                    hours = np.array([d.hour for d in dates])
                    base = 0.3 + 0.2 * np.sin(hours * np.pi / 12)
                    random = np.random.rand(len(dates)) * 0.2
                    values = base + random
                    values = np.clip(values, 0.05, 0.95)  # 确保值在合理范围内且不为零
                else:
                    # 太阳能电场：创建日间钟形曲线
                    hours = np.array([d.hour for d in dates])
                    minutes = np.array([d.minute for d in dates])
                    total_minutes = hours * 60 + minutes
                    values = np.zeros(len(dates))
                    
                    # 6:00到18:00之间是有效发电时间
                    day_minutes = (total_minutes >= 6*60) & (total_minutes <= 18*60)
                    mid_day = 12*60  # 正午时刻
                    
                    # 创建钟形曲线
                    values[day_minutes] = 0.8 * np.exp(-((total_minutes[day_minutes] - mid_day) ** 2) / (2 * (3*60) ** 2))
                
                # 保存结果
                res = pd.Series(values, index=dates)
    result_path = f'result/output'
                os.makedirs(result_path, exist_ok=True)
                res.to_csv(os.path.join(result_path, f'output{farm_id}.csv'))
                print(f"Created simple pattern for farm {farm_id}")
                
            except Exception as e3:
                print(f"All methods failed for farm {farm_id}: {str(e3)}")

print('All farms processed')
