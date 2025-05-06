import os
import pickle
import warnings
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.interpolate import CubicSpline

# 添加PyTorch相关依赖
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# 定义风电场和光伏场站ID
WIND_FARMS = [1, 2, 3, 4, 5]
SOLAR_FARMS = [6, 7, 8, 9, 10]

warnings.filterwarnings('ignore')

nwps = ['NWP_1','NWP_2','NWP_3']
fact_path = 'training/middle_school/TRAIN/fact_data'

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置随机种子以确保结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 创建自定义Dataset
class PowerGenerationDataset(Dataset):
    def __init__(self, features, targets=None):
        # 检查是否是pandas对象，如果是则获取numpy数组
        if hasattr(features, 'values'):
            self.features = torch.tensor(features.values, dtype=torch.float32)
        else:
            self.features = torch.tensor(features, dtype=torch.float32)
            
        if targets is not None:
            if hasattr(targets, 'values'):
                self.targets = torch.tensor(targets.values, dtype=torch.float32)
            else:
                self.targets = torch.tensor(targets, dtype=torch.float32)
            self.has_targets = True
        else:
            self.has_targets = False
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.has_targets:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # 调整输入形状为 [batch_size, seq_len=1, input_size]
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out.squeeze(1))
        return output

# 定义混合模型 (MLP + LSTM)
class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_dims, lstm_hidden_size, num_layers, dropout=0.3):
        super(HybridModel, self).__init__()
        
        # 特征提取器 - 改进的MLP部分
        layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 改进的LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dims[-1],
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=4)
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden_size, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = 1  # 假设是单时间步，如果是序列数据需要调整
        
        # 特征提取
        x = self.feature_extractor(x)
        
        # 重塑以适应LSTM (batch_size, seq_len, features)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 应用注意力
        attn_out, _ = self.attention(lstm_out.transpose(0, 1), lstm_out.transpose(0, 1), lstm_out.transpose(0, 1))
        attn_out = attn_out.transpose(0, 1)
        
        # 输出预测
        out = self.fc(attn_out[:, -1, :])
        return out

def add_time_features(df):
    """添加时间特征"""
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    df_copy['day'] = df_copy.index.day
    df_copy['month'] = df_copy.index.month
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['sin_hour'] = np.sin(2 * np.pi * df_copy.index.hour / 24)
    df_copy['cos_hour'] = np.cos(2 * np.pi * df_copy.index.hour / 24)
    df_copy['sin_month'] = np.sin(2 * np.pi * df_copy.index.month / 12)
    df_copy['cos_month'] = np.cos(2 * np.pi * df_copy.index.month / 12)
    return df_copy

def add_weather_derivatives(df, is_wind_farm=True):
    """添加气象衍生特征，区分风电场和光伏场站"""
    df_copy = df.copy()
    
    # 计算风速的一些统计特征
    ws_cols = [col for col in df_copy.columns if '_ws_' in col]
    if ws_cols:
        df_copy['ws_mean'] = df_copy[ws_cols].mean(axis=1)
        df_copy['ws_std'] = df_copy[ws_cols].std(axis=1)
        df_copy['ws_min'] = df_copy[ws_cols].min(axis=1)
        df_copy['ws_max'] = df_copy[ws_cols].max(axis=1)
        df_copy['ws_range'] = df_copy['ws_max'] - df_copy['ws_min']
        
    # 计算风向特征
    u_cols = [col for col in df_copy.columns if '_u_' in col]
    v_cols = [col for col in df_copy.columns if '_v_' in col]
    
    if u_cols and v_cols:
        # 计算平均风向
        df_copy['mean_u'] = df_copy[u_cols].mean(axis=1)
        df_copy['mean_v'] = df_copy[v_cols].mean(axis=1)
        df_copy['wind_direction'] = np.arctan2(df_copy['mean_v'], df_copy['mean_u']) * 180 / np.pi
        df_copy['wind_direction'] = df_copy['wind_direction'].apply(lambda x: x + 360 if x < 0 else x)
        
        # 添加风向的正弦和余弦分量以避免周期性问题
        df_copy['sin_wind_dir'] = np.sin(np.radians(df_copy['wind_direction']))
        df_copy['cos_wind_dir'] = np.cos(np.radians(df_copy['wind_direction']))
    
    # 风电场特定特征
    if is_wind_farm:
        # 风能密度估算 (ρ * v^3 / 2，其中ρ为空气密度，假设为常数)
        df_copy['wind_energy_density'] = 0.5 * 1.225 * df_copy['ws_mean'] ** 3
        
        # 风切变指数估算（简化）
        if 'NWP_1_ws_0' in df_copy.columns and 'NWP_1_ws_1' in df_copy.columns:
            df_copy['wind_shear_index'] = np.log(df_copy['NWP_1_ws_0'] / df_copy['NWP_1_ws_1']) / np.log(100/10)
            df_copy['wind_shear_index'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_copy['wind_shear_index'].fillna(0.143, inplace=True)  # 默认风切变指数
        
        # 风电曲线特征（简化模拟，实际应根据风机参数调整）
        # 定义切入风速、额定风速和切出风速
        cut_in = 3.0
        rated = 12.0
        cut_out = 25.0
        
        def simplified_power_curve(ws):
            if ws < cut_in or ws > cut_out:
                return 0
            elif ws < rated:
                return (ws**3 - cut_in**3) / (rated**3 - cut_in**3)
            else:
                return 1
        
        df_copy['theoretical_power'] = df_copy['ws_mean'].apply(simplified_power_curve)
        
        # 湍流强度估计（风速标准差/平均风速）
        df_copy['turbulence_intensity'] = df_copy['ws_std'] / df_copy['ws_mean']
        df_copy['turbulence_intensity'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_copy['turbulence_intensity'].fillna(0.1, inplace=True)  # 默认湍流强度
    
    # 光伏场站特定特征
    else:
        # 提取光伏相关特征
        poai_cols = [col for col in df_copy.columns if 'poai' in col.lower()]
        ghi_cols = [col for col in df_copy.columns if 'ghi' in col.lower()]
        
        if poai_cols:
            df_copy['poai_mean'] = df_copy[poai_cols].mean(axis=1)
            df_copy['poai_max'] = df_copy[poai_cols].max(axis=1)
        
        if ghi_cols:
            df_copy['ghi_mean'] = df_copy[ghi_cols].mean(axis=1)
            df_copy['ghi_max'] = df_copy[ghi_cols].max(axis=1)
        
        # 云量特征
        tcc_cols = [col for col in df_copy.columns if 'tcc' in col.lower()]
        if tcc_cols:
            df_copy['cloud_coverage'] = df_copy[tcc_cols].mean(axis=1)
            df_copy['clear_sky_index'] = 1 - df_copy['cloud_coverage']
        
        # 温度特征（对光伏效率有影响）
        t2m_cols = [col for col in df_copy.columns if 't2m' in col.lower()]
        if t2m_cols:
            df_copy['temp_mean'] = df_copy[t2m_cols].mean(axis=1)
            df_copy['temp_max'] = df_copy[t2m_cols].max(axis=1)
            
            # 模拟温度对光伏效率的影响
            # 假设25摄氏度为参考温度，每升高1度效率下降0.4%
            reference_temp = 273.15 + 25  # 开尔文温度
            df_copy['temp_efficiency'] = 1 - 0.004 * (df_copy['temp_mean'] - reference_temp)
            df_copy['temp_efficiency'] = df_copy['temp_efficiency'].clip(0.7, 1.0)
        
        # 理论光伏发电能力估算
        if 'poai_mean' in df_copy.columns and 'is_daylight' in df_copy.columns:
            df_copy['theoretical_pv_power'] = df_copy['poai_mean'] * df_copy['is_daylight']
            if 'temp_efficiency' in df_copy.columns and 'clear_sky_index' in df_copy.columns:
                df_copy['adjusted_pv_power'] = (df_copy['theoretical_pv_power'] * 
                                               df_copy['temp_efficiency'] * 
                                               df_copy['clear_sky_index'])
    
    # 计算一些交叉特征
    for nwp in nwps:
        ws_cols = [col for col in df_copy.columns if f'{nwp}_ws_' in col]
        if ws_cols:
            df_copy[f'{nwp}_ws_var'] = df_copy[ws_cols].var(axis=1)
            df_copy[f'{nwp}_ws_kurt'] = df_copy[ws_cols].kurtosis(axis=1)
    
    return df_copy

def add_lag_features(df, lag_hours=[1, 2, 3, 6, 12, 24]):
    """添加滞后特征"""
    df_copy = df.copy()
    for col in df_copy.columns:
        for lag in lag_hours:
            df_copy[f'{col}_lag{lag}'] = df_copy[col].shift(lag)
    return df_copy

def data_preprocess(x_df, y_df=None, is_train=True):
    """改进的数据预处理"""
    x_df = x_df.copy()
    
    # 添加时间特征
    x_df = add_time_features(x_df)
    
    # 添加气象衍生特征
    x_df = add_weather_derivatives(x_df)
    
    if is_train and y_df is not None:
        y_df = y_df.copy()
        # 清理数据
        x_df = x_df.dropna()
        y_df = y_df.dropna()
        
        # 数据对扣
        ind = [i for i in y_df.index if i in x_df.index]
        x_df = x_df.loc[ind]
        y_df = y_df.loc[ind]
        
        # 处理异常值
        y_df[y_df < 0] = 0
        y_df[y_df > 1] = 1
        
        # 标准化数据
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(x_df)
        y_scaled = scaler_y.fit_transform(y_df.values.reshape(-1, 1)).flatten()
        
        # 分割训练集和验证集
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        X_val = X_scaled[split_idx:]
        y_train = y_scaled[:split_idx]
        y_val = y_scaled[split_idx:]
        
        return X_train, X_val, y_train, y_val, scaler_X, scaler_y
    else:
        # 测试数据处理
        # 仅填充滞后特征的缺失值
        lag_cols = [col for col in x_df.columns if 'lag' in col]
        for col in lag_cols:
            x_df[col].fillna(x_df[col].mean(), inplace=True)
        return x_df

def train(farm_id):
    """改进的基于深度学习的训练函数"""
    print(f"开始训练发电站 {farm_id} 的深度学习模型...")
    
    # 创建模型保存路径
    model_path = f'models/{farm_id}'
    os.makedirs(model_path, exist_ok=True)
    
    # 读取和准备数据
    x_df = pd.DataFrame()
    nwp_train_path = f'training/middle_school/TRAIN/nwp_data_train/{farm_id}'
    
    # 处理NWP数据
    for nwp in nwps:
        nwp_path = os.path.join(nwp_train_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        
        # 提取更大范围的空间网格以捕获更多信息
        u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                         channel=['u100']).data.values.reshape(365 * 24, 9)
        v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                     channel=['v100']).data.values.reshape(365 * 24, 9)
        
        # 创建基本特征
        u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
        v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
        
        # 添加风向角度特征
        wd = np.arctan2(v, u) * 180 / np.pi
        wd = np.where(wd < 0, wd + 360, wd)
        wd_df = pd.DataFrame(wd, columns=[f"{nwp}_wd_{i}" for i in range(wd.shape[1])])
        
        nwp_df = pd.concat([u_df, v_df, ws_df, wd_df], axis=1)
        x_df = pd.concat([x_df, nwp_df], axis=1)
    
    x_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')
    
    # 读取目标变量
    y_df = pd.read_csv(os.path.join(fact_path,f'{farm_id}_normalization_train.csv'), index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']
    
    # 添加滞后特征
    x_df = add_lag_features(x_df)
    
    # 特征选择 - 增加相关性分析
    corr_matrix = pd.concat([x_df, y_df], axis=1).corr()
    relevant_features = corr_matrix['power'].abs().sort_values(ascending=False)
    print(f"Top 20 most relevant features: {relevant_features.head(20).index.tolist()}")
    
    # 数据预处理
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = data_preprocess(x_df, y_df)
    
    # 创建PyTorch数据集和数据加载器
    train_dataset = PowerGenerationDataset(X_train, y_train)
    val_dataset = PowerGenerationDataset(X_val, y_val)
    
    batch_size = 128  # 增加批量大小以提高稳定性
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 确定模型是风电场还是光伏场站
    is_wind_farm = farm_id in WIND_FARMS
    
    # 初始化改进后的模型
    input_size = X_train.shape[1]
    model = HybridModel(
        input_size=input_size,
        hidden_dims=[512, 256, 128],  # 更复杂的网络架构
        lstm_hidden_size=128,
        num_layers=3,
        dropout=0.4  # 增加dropout以减轻过拟合
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # 使用AdamW并添加权重衰减
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 早停机制
    early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.0001)
    
    # 训练循环
    num_epochs = 200  # 增加最大轮次，配合早停使用
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        predictions = []
        actual_values = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                # 收集预测和实际值用于计算额外的指标
                predictions.extend(outputs.cpu().numpy())
                actual_values.extend(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 计算额外的评估指标
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 打印当前轮次的结果
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"MAE: {mae:.4f} | "
              f"R²: {r2:.4f}")
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("早停激活! 停止训练。")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'checkpoint_{farm_id}.pt'))
    
    # 保存模型和缩放器
    model_package = {
        'model': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'input_size': input_size,
        'farm_id': farm_id
    }
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {farm_id}')
    plt.legend()
    plt.savefig(f'loss_curve_{farm_id}.png')
    
    torch.save(model_package, os.path.join(model_path, f"{farm_id}_model.pth"))
    return model_package

def predict(model_package, farm_id):
    """基于深度学习的预测函数"""
    # 解包模型组件
    model = model_package['model'].to(device)
    scaler = model_package['scaler']
    model.eval()  # 设置为评估模式
    
    # 读取测试数据
    x_df = pd.DataFrame()
    nwp_test_path = f'training/middle_school/TEST/nwp_data_test/{farm_id}'
    
    # 处理NWP数据
    for nwp in nwps:
        nwp_path = os.path.join(nwp_test_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        
        # 提取相同的特征
        u = nwp_data.sel(lat=range(4, 7), lon=range(4, 7), lead_time=range(24),
                         channel=['u100']).data.values.reshape(31 * 24, 9)
        v = nwp_data.sel(lat=range(4, 7), lon=range(4, 7), lead_time=range(24),
                         channel=['v100']).data.values.reshape(31 * 24, 9)
        u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
        v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
        
        # 添加风向角度特征
        wd = np.arctan2(v, u) * 180 / np.pi
        wd = np.where(wd < 0, wd + 360, wd)
        wd_df = pd.DataFrame(wd, columns=[f"{nwp}_wd_{i}" for i in range(wd.shape[1])])
        
        nwp_df = pd.concat([u_df, v_df, ws_df, wd_df], axis=1)
        x_df = pd.concat([x_df, nwp_df], axis=1)
    
    x_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')
    
    # 添加滞后特征
    x_df = add_lag_features(x_df)
    
    # 预处理测试数据
    x_test = data_preprocess(x_df, is_train=False)
    
    # 应用标准化
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index
    )
    
    # 创建PyTorch测试数据集和数据加载器
    test_dataset = PowerGenerationDataset(x_test_scaled)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 预测
    pred_pw = []
    with torch.no_grad():
        for batch_X in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            pred_pw.extend(outputs.cpu().numpy())
    
    # 创建预测序列
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0], periods=len(pred_pw), freq='h'))
    
    # 将预测重采样为15分钟
    res = pred.resample('15min').interpolate(method='cubic')
    
    # 修正预测值范围
    res[res < 0] = 0
    res[res > 1] = 1
    
    return res

def main():
    """主程序执行函数，处理所有场站的训练与预测"""
    print("===== 新能源功率预报系统启动 - 深度学习版 =====")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # 处理所有场站
    farms = WIND_FARMS + SOLAR_FARMS
    print(f"将处理 {len(farms)} 个发电站: {farms}")
    
    # 存储准确率信息
    accuracies = {}
    
    # 循环处理每个场站
    for farm_id in farms:
        try:
            print(f"\n===== 开始处理发电站 {farm_id} =====")
            model_path = f'models/{farm_id}'
            os.makedirs(model_path, exist_ok=True)
            model_file = os.path.join(model_path, 'deep_learning_model.pkl')
            
            # 判断是否已有训练好的模型
            if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
                print(f"发现已有训练模型，加载模型: {model_file}")
                with open(model_file, "rb") as f:
                    model_package = pickle.load(f)
            else:
                print(f"未发现已有模型，开始训练新模型")
                # 训练模型
                model_package = train(farm_id)
                # 保存模型
                with open(model_file, "wb") as f:
                    pickle.dump(model_package, f)
                print(f"模型已保存至: {model_file}")
            
            # 生成预测
            print(f"开始生成发电站 {farm_id} 的预测结果...")
            pred = predict(model_package, farm_id)
            
            # 记录模型精度
            if 'accuracy' in model_package:
                accuracies[farm_id] = model_package['accuracy']
            
            # 保存预测结果
            output_file = os.path.join('output', f'output{farm_id}.csv')
            pred.to_csv(output_file)
            print(f"预测结果已保存至: {output_file}")
            print(f"发电站 {farm_id} 处理完成")
            
        except Exception as e:
            print(f"处理发电站 {farm_id} 时发生错误: {e}")
            import traceback
            print(traceback.format_exc())
    
    # 输出整体模型准确率信息
    if accuracies:
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        print(f"\n===== 模型准确率总结 =====")
        for farm_id, acc in accuracies.items():
            print(f"发电站 {farm_id}: {acc:.4f}")
        print(f"平均准确率: {avg_accuracy:.4f}")
    
    # 打包所有输出结果
    try:
        import zipfile
        
        output_zip_path = 'output.zip'
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            for root, dirs, files in os.walk('output'):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, os.path.dirname('output'))
                    zipf.write(file_path, arcname=rel_path)
        
        print(f"整个output目录已打包至: {output_zip_path}")
    except Exception as e:
        print(f"打包输出结果时发生错误: {e}")
    
    print("\n===== 新能源功率预报系统运行完成 =====")

if __name__ == "__main__":
    main()