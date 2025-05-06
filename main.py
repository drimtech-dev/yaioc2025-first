import os
import pickle
import warnings
from datetime import datetime
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
        self.features = torch.tensor(features.values, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets.values, dtype=torch.float32)
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
    def __init__(self, input_size, hidden_dims=[256, 128], lstm_hidden_size=64, num_layers=2, dropout=0.3):
        super(HybridModel, self).__init__()
        
        # MLP部分
        mlp_layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        self.mlp = nn.Sequential(*mlp_layers)
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 合并层
        self.fc1 = nn.Linear(hidden_dims[-1] + lstm_hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # MLP路径
        mlp_out = self.mlp(x)
        
        # LSTM路径
        lstm_in = x.unsqueeze(1)  # 添加序列维度
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out.squeeze(1)  # 移除序列维度
        
        # 合并MLP和LSTM输出
        combined = torch.cat([mlp_out, lstm_out], dim=1)
        x = F.relu(self.fc1(combined))
        output = self.fc2(x)
        
        return output.squeeze(-1)  # 确保输出维度是 [batch_size]

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
        
        return x_df, y_df
    else:
        # 测试数据处理
        # 仅填充滞后特征的缺失值
        lag_cols = [col for col in x_df.columns if 'lag' in col]
        for col in lag_cols:
            x_df[col].fillna(x_df[col].mean(), inplace=True)
        return x_df

def train(farm_id):
    """基于深度学习的训练函数"""
    print(f"开始训练发电站 {farm_id} 的深度学习模型...")
    
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
    
    # 预处理数据
    x_processed, y_processed = data_preprocess(x_df, y_df, is_train=True)
    
    # 特征标准化
    scaler = StandardScaler()
    x_processed_scaled = pd.DataFrame(
        scaler.fit_transform(x_processed), 
        columns=x_processed.columns,
        index=x_processed.index
    )
    
    # 划分训练集和验证集 (使用时间序列分割)
    tscv = TimeSeriesSplit(n_splits=5)
    # 取最后一个分割作为训练/验证集
    for train_index, val_index in tscv.split(x_processed_scaled):
        pass  # 我们只使用最后一个分割
    
    X_train = x_processed_scaled.iloc[train_index]
    y_train = y_processed.iloc[train_index]
    X_val = x_processed_scaled.iloc[val_index]
    y_val = y_processed.iloc[val_index]
    
    # 创建PyTorch数据集和数据加载器
    train_dataset = PowerGenerationDataset(X_train, y_train)
    val_dataset = PowerGenerationDataset(X_val, y_val)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 确定模型是风电场还是光伏场站
    is_wind_farm = farm_id in WIND_FARMS
    
    # 初始化模型
    input_size = X_train.shape[1]
    model = HybridModel(
        input_size=input_size,
        hidden_dims=[256, 128],
        lstm_hidden_size=64,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # 训练模型
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))
        val_r2 = r2_score(val_true, val_preds)
        
        # 打印训练进度
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.6f}, Val R2: {val_r2:.4f}')
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
        
        # 早停
        if counter >= patience:
            print(f'早停：连续 {patience} 个轮次验证损失未改善')
            break
    
    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    
    # 在整个训练集上重新训练一次
    print("在完整训练集上进行最终训练...")
    full_dataset = PowerGenerationDataset(x_processed_scaled, y_processed)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    final_model = HybridModel(
        input_size=input_size,
        hidden_dims=[256, 128],
        lstm_hidden_size=64,
        num_layers=2,
        dropout=0.3
    ).to(device)
    final_model.load_state_dict(best_model_state)
    
    # 使用较小的学习率进行微调
    final_optimizer = optim.Adam(final_model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    final_epochs = 10
    for epoch in range(final_epochs):
        final_model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in full_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            final_optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            final_optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 2 == 0:
            print(f'最终训练 Epoch {epoch}/{final_epochs}, Loss: {epoch_loss/len(full_loader):.6f}')
    
    # 评估最终模型
    final_model.eval()
    with torch.no_grad():
        all_X = torch.tensor(x_processed_scaled.values, dtype=torch.float32).to(device)
        all_preds = final_model(all_X).cpu().numpy()
    
    train_rmse = np.sqrt(mean_squared_error(y_processed, all_preds))
    train_r2 = r2_score(y_processed, all_preds)
    
    print(f"发电站 {farm_id} 训练评估: RMSE = {train_rmse:.4f}, R² = {train_r2:.4f}")
    
    # 返回所有需要保存的组件
    model_package = {
        'model': final_model.to('cpu'),  # 保存到CPU避免GPU内存问题
        'scaler': scaler,
        'input_size': input_size,
        'accuracy': train_r2,
        'is_wind_farm': is_wind_farm
    }
    
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