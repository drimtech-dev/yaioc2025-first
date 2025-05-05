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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.interpolate import CubicSpline

# 导入PyTorch相关库
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F

# 定义风电场和光伏场站ID
WIND_FARMS = [1, 2, 3, 4, 5]
SOLAR_FARMS = [6, 7, 8, 9, 10]

warnings.filterwarnings('ignore')

nwps = ['NWP_1','NWP_2','NWP_3']
fact_path = 'training/middle_school/TRAIN/fact_data'

# 设置随机种子，确保结果可重现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 检查是否可用CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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

# PyTorch模型定义
class LSTMModel(nn.Module):
    """
    基于LSTM的深度学习模型
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, output_dim=1):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim // 2)  # *2是因为双向LSTM
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        # LSTM层的输出
        lstm_out, _ = self.lstm(x)
        
        # 只取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 全连接层
        out = F.relu(self.fc1(lstm_out))
        out = self.dropout1(out)
        out = self.fc2(out)
        
        return out

class CNNLSTMModel(nn.Module):
    """
    CNN+LSTM混合模型，CNN用于提取空间特征，LSTM用于时序特征
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.3, output_dim=1):
        super(CNNLSTMModel, self).__init__()
        
        # CNN层
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout)
        
        # 计算CNN输出后的序列长度
        self.lstm_input_dim = 128
        self.lstm_seq_len = input_dim // 4  # 因为有两次池化层，每次长度减半
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        # 调整输入维度顺序以适配CNN (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN层
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 调整维度以适配LSTM (batch, features, seq_len) -> (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 只取最后一个时间步
        lstm_out = lstm_out[:, -1, :]
        
        # 全连接层
        out = F.relu(self.fc1(lstm_out))
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out

def create_sequences(data, target=None, seq_length=24):
    """
    创建用于时间序列深度学习模型的序列数据
    """
    X = []
    y = []
    
    for i in range(seq_length, len(data)):
        X.append(data.iloc[i-seq_length:i].values)
        if target is not None:
            y.append(target.iloc[i])
    
    if target is not None:
        return np.array(X), np.array(y)
    else:
        return np.array(X)

def train_pytorch_model(model, train_loader, val_loader, num_epochs=100, patience=10):
    """训练PyTorch模型并进行早停"""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=0.0001
    )
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping!')
                break
    
    # 恢复最佳模型状态
    model.load_state_dict(best_model_state)
    return model, best_val_loss

def data_preprocess(x_df, y_df=None, is_train=True, is_dl_model=False, seq_length=24):
    """改进的数据预处理，支持深度学习模型的数据准备"""
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
        
        # 为深度学习模型创建序列数据
        if is_dl_model:
            # 确保索引连续并按时间排序
            x_df = x_df.sort_index()
            y_df = y_df.sort_index()
            
            # 创建序列数据
            X_seq, y_seq = create_sequences(x_df, y_df, seq_length=seq_length)
            return X_seq, y_seq
        else:
            return x_df, y_df
    else:
        # 测试数据处理
        # 仅填充滞后特征的缺失值
        lag_cols = [col for col in x_df.columns if 'lag' in col]
        for col in lag_cols:
            x_df[col].fillna(x_df[col].mean(), inplace=True)
            
        if is_dl_model:
            # 创建序列数据，不需要目标变量
            X_seq = create_sequences(x_df, seq_length=seq_length)
            return X_seq
        else:
            return x_df

def train(farm_id):
    """增强版训练函数，包含PyTorch深度学习模型"""
    print(f"开始训练发电站 {farm_id} 的模型...")
    
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
    
    # 特征选择
    selector = SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
        threshold='median'
    )
    selector.fit(x_processed_scaled, y_processed)
    selected_features = x_processed_scaled.columns[selector.get_support()]
    x_processed_selected = x_processed_scaled[selected_features]
    
    print(f"选择了 {len(selected_features)} 个特征，从总共 {x_processed_scaled.shape[1]} 个")
    
    # 创建树模型
    model_lgb = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED
    )
    model_xgb = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED
    )
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_SEED
    )
    
    # 训练树模型
    model_lgb.fit(x_processed_selected, y_processed)
    model_xgb.fit(x_processed_selected, y_processed)
    model_rf.fit(x_processed_selected, y_processed)
    
    # 预处理深度学习模型的数据
    seq_length = 24  # 使用24小时的历史数据
    X_seq, y_seq = data_preprocess(x_df, y_df, is_train=True, is_dl_model=True, seq_length=seq_length)
    
    # 特征缩放
    X_seq_scaled = np.zeros_like(X_seq)
    for i in range(X_seq.shape[0]):
        X_seq_scaled[i] = scaler.transform(X_seq[i])
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_seq_scaled)
    y_tensor = torch.FloatTensor(y_seq.reshape(-1, 1))
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 创建深度学习模型
    input_dim = X_seq_scaled.shape[2]  # 特征维度
    
    # 训练LSTM模型
    print("训练LSTM模型...")
    lstm_model = LSTMModel(input_dim=input_dim)
    lstm_model, lstm_val_loss = train_pytorch_model(
        lstm_model, train_loader, val_loader, num_epochs=100, patience=10
    )
    
    # 训练CNN-LSTM模型
    print("训练CNN-LSTM模型...")
    cnn_lstm_model = CNNLSTMModel(input_dim=input_dim)
    cnn_lstm_model, cnn_lstm_val_loss = train_pytorch_model(
        cnn_lstm_model, train_loader, val_loader, num_epochs=100, patience=10
    )
    
    # 评估各个模型
    # 树模型评估
    lgb_pred = model_lgb.predict(x_processed_selected)
    xgb_pred = model_xgb.predict(x_processed_selected)
    rf_pred = model_rf.predict(x_processed_selected)
    
    # 深度学习模型评估
    lstm_model.eval()
    cnn_lstm_model.eval()
    
    with torch.no_grad():
        X_tensor_gpu = X_tensor.to(device)  # 将输入数据移到GPU
        lstm_pred = lstm_model(X_tensor_gpu).cpu().numpy().flatten()
        cnn_lstm_pred = cnn_lstm_model(X_tensor_gpu).cpu().numpy().flatten()
    
    # 集成预测
    tree_preds = np.column_stack([lgb_pred, xgb_pred, rf_pred])
    tree_weights = np.array([0.35, 0.35, 0.3])
    tree_ensemble_pred = np.sum(tree_preds * tree_weights.reshape(1, -1), axis=1)
    
    # 计算各模型的RMSE
    tree_rmse = np.sqrt(mean_squared_error(y_processed, tree_ensemble_pred))
    lstm_rmse = np.sqrt(mean_squared_error(y_seq, lstm_pred))
    cnn_lstm_rmse = np.sqrt(mean_squared_error(y_seq, cnn_lstm_pred))
    
    print(f"树模型集成RMSE: {tree_rmse:.4f}")
    print(f"LSTM模型RMSE: {lstm_rmse:.4f}")
    print(f"CNN-LSTM模型RMSE: {cnn_lstm_rmse:.4f}")
    
    # 确定最终权重（基于RMSE的倒数）
    total_inv_rmse = 1/tree_rmse + 1/lstm_rmse + 1/cnn_lstm_rmse
    tree_weight = (1/tree_rmse) / total_inv_rmse
    lstm_weight = (1/lstm_rmse) / total_inv_rmse
    cnn_lstm_weight = (1/cnn_lstm_rmse) / total_inv_rmse
    
    print(f"最终集成权重 - 树模型: {tree_weight:.2f}, LSTM: {lstm_weight:.2f}, CNN-LSTM: {cnn_lstm_weight:.2f}")
    
    # 返回所有需要保存的组件
    model_package = {
        'tree_models': {
            'lgb': model_lgb,
            'xgb': model_xgb,
            'rf': model_rf,
            'tree_weights': tree_weights
        },
        'dl_models': {
            'lstm': lstm_model,
            'cnn_lstm': cnn_lstm_model
        },
        'ensemble_weights': {
            'tree': tree_weight,
            'lstm': lstm_weight,
            'cnn_lstm': cnn_lstm_weight
        },
        'scaler': scaler,
        'feature_selector': selector,
        'selected_features': selected_features,
        'seq_length': seq_length,
        'accuracy': 1 - (tree_rmse + lstm_rmse + cnn_lstm_rmse) / 3  # 简单的准确率估计
    }
    
    return model_package

def predict(model_package, farm_id):
    """增强版预测函数，集成树模型和PyTorch深度学习模型"""
    # 解包模型组件
    tree_models = model_package['tree_models']
    dl_models = model_package['dl_models']
    ensemble_weights = model_package['ensemble_weights']
    scaler = model_package['scaler']
    selector = model_package['selector']
    selected_features = model_package['selected_features']
    seq_length = model_package['seq_length']
    
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
    
    # 树模型预测准备
    x_test = data_preprocess(x_df, is_train=False)
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index
    )
    
    x_test_selected = x_test_scaled[selected_features]
    
    # 树模型预测
    lgb_pred = tree_models['lgb'].predict(x_test_selected)
    xgb_pred = tree_models['xgb'].predict(x_test_selected)
    rf_pred = tree_models['rf'].predict(x_test_selected)
    
    tree_preds = np.column_stack([lgb_pred, xgb_pred, rf_pred])
    tree_ensemble_pred = np.sum(tree_preds * tree_models['tree_weights'].reshape(1, -1), axis=1)
    
    # 深度学习模型预测准备
    X_seq_test = data_preprocess(x_df, is_train=False, is_dl_model=True, seq_length=seq_length)
    
    X_seq_test_scaled = np.zeros_like(X_seq_test)
    for i in range(X_seq_test.shape[0]):
        X_seq_test_scaled[i] = scaler.transform(X_seq_test[i])
    
    X_tensor_test = torch.FloatTensor(X_seq_test_scaled).to(device)
    
    # 深度学习模型预测
    dl_models['lstm'].eval()
    dl_models['cnn_lstm'].eval()
    
    with torch.no_grad():
        lstm_pred = dl_models['lstm'](X_tensor_test).cpu().numpy().flatten()
        cnn_lstm_pred = dl_models['cnn_lstm'](X_tensor_test).cpu().numpy().flatten()
    
    # 集成预测
    final_pred = (tree_ensemble_pred * ensemble_weights['tree'] +
                  lstm_pred * ensemble_weights['lstm'] +
                  cnn_lstm_pred * ensemble_weights['cnn_lstm'])
    
    return final_pred

def main():
    """主函数，训练和预测所有风电场和光伏场站"""
    # 确保存储模型和结果的目录存在
    os.makedirs('models', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    
    model_packages = {}
    
    # 训练所有风电场和光伏场站的模型
    for farm_id in WIND_FARMS + SOLAR_FARMS:
        model_package = train(farm_id)
        model_packages[farm_id] = model_package
        
        # 保存模型包
        with open(f'models/model_package_{farm_id}.pkl', 'wb') as f:
            pickle.dump(model_package, f)
    
    # 预测所有风电场和光伏场站的功率
    predictions = {}
    for farm_id in WIND_FARMS + SOLAR_FARMS:
        with open(f'models/model_package_{farm_id}.pkl', 'rb') as f:
            model_package = pickle.load(f)
        
        pred = predict(model_package, farm_id)
        predictions[farm_id] = pred
        
        # 保存预测结果
        pred_df = pd.DataFrame(pred, columns=['predicted_power'])
        pred_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')
        pred_df.to_csv(f'result/prediction_{farm_id}.csv')
    
    # 打包所有预测结果
    try:
        import zipfile
        with zipfile.ZipFile('result/output.zip', 'w') as zipf:
            for farm_id in WIND_FARMS + SOLAR_FARMS:
                zipf.write(f'result/prediction_{farm_id}.csv')
        
        print(f"所有预测结果已打包至: result/output.zip")
    except Exception as e:
        print(f"打包输出结果时发生错误: {e}")
    
    print("\n===== 新能源功率预报系统运行完成 =====")

if __name__ == "__main__":
    main()