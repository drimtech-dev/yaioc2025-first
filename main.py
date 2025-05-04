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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# 设置随机种子以保证可重复性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 定义风电场和光伏场站ID
WIND_FARMS = [1, 2, 3, 4, 5]
SOLAR_FARMS = [6, 7, 8, 9, 10]

warnings.filterwarnings('ignore')

nwps = ['NWP_1','NWP_2','NWP_3']
fact_path = 'training/middle_school/TRAIN/fact_data'

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 增强CUDA相关设置
try:
    # 尝试获取CUDA设备信息
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # 加速卷积运算
        torch.backends.cudnn.deterministic = False  # 提高性能
        print(f"使用CUDA加速: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        device = torch.device('cpu')
        print("CUDA不可用，使用CPU")
except Exception as e:
    device = torch.device('cpu')
    print(f"设置CUDA时出错: {e}，使用CPU替代")

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
        
        # 添加风电场特有的功率密度相关特征
        df_copy['power_density'] = 0.5 * 1.225 * df_copy['ws_mean']**3
        
        # 添加风速的高阶多项式特征（捕捉非线性关系）
        df_copy['ws_mean_squared'] = df_copy['ws_mean']**2
        df_copy['ws_mean_cubed'] = df_copy['ws_mean']**3
        df_copy['ws_exp_term'] = np.exp(-df_copy['ws_mean']/10)
    
    # 光伏场站特定特征
    else:
        # 提取光伏相关特征
        poai_cols = [col for col in df_copy.columns if 'poai' in col.lower()]
        ghi_cols = [col for col in df_copy.columns if 'ghi' in col.lower()]
        
        if poai_cols:
            df_copy['poai_mean'] = df_copy[poai_cols].mean(axis=1)
            df_copy['poai_max'] = df_copy[poai_cols].max(axis=1)
            df_copy['poai_min'] = df_copy[poai_cols].min(axis=1)
            df_copy['poai_std'] = df_copy[poai_cols].std(axis=1)
            # 添加太阳能相关非线性特征
            df_copy['poai_log'] = np.log1p(df_copy['poai_mean'].clip(min=0))
            df_copy['poai_sqrt'] = np.sqrt(df_copy['poai_mean'].clip(min=0))
        
        if ghi_cols:
            df_copy['ghi_mean'] = df_copy[ghi_cols].mean(axis=1)
            df_copy['ghi_max'] = df_copy[ghi_cols].max(axis=1)
            df_copy['ghi_min'] = df_copy[ghi_cols].min(axis=1)
            df_copy['ghi_std'] = df_copy[ghi_cols].std(axis=1)
            # 添加太阳能相关非线性特征
            df_copy['ghi_log'] = np.log1p(df_copy['ghi_mean'].clip(min=0))
            df_copy['ghi_sqrt'] = np.sqrt(df_copy['ghi_mean'].clip(min=0))
            
        # 转换温度从开尔文到摄氏度，便于模型理解
        t2m_cols = [col for col in df_copy.columns if 't2m' in col.lower()]
        if t2m_cols:
            for col in t2m_cols:
                df_copy[f'{col}_celsius'] = df_copy[col] - 273.15
                
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

def data_preprocess(x_df, y_df=None, is_train=True, is_wind_farm=True):
    """改进的数据预处理"""
    x_df = x_df.copy()
    
    # 添加时间特征
    x_df = add_time_features(x_df)
    
    # 添加气象衍生特征
    x_df = add_weather_derivatives(x_df, is_wind_farm)  # 正确传递is_wind_farm参数
    
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
    """增强版训练函数"""
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
    
    is_wind_farm = farm_id in WIND_FARMS
    # 预处理数据
    x_processed, y_processed = data_preprocess(x_df, y_df, is_train=True, is_wind_farm=is_wind_farm)
    
    # 特征标准化
    scaler = StandardScaler()
    x_processed_scaled = pd.DataFrame(
        scaler.fit_transform(x_processed), 
        columns=x_processed.columns,
        index=x_processed.index
    )
    
    # 特征选择
    selector = SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=42),
        threshold='median'
    )
    selector.fit(x_processed_scaled, y_processed)
    selected_features = x_processed_scaled.columns[selector.get_support()]
    x_processed_selected = x_processed_scaled[selected_features]
    
    print(f"选择了 {len(selected_features)} 个特征，从总共 {x_processed_scaled.shape[1]} 个")
    
    # 创建模型
    model_lgb = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_xgb = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # 集成模型
    final_model = VotingRegressor(
        estimators=[
            ('lgb', model_lgb),
            ('xgb', model_xgb),
            ('rf', model_rf)
        ],
        weights=[0.4, 0.4, 0.2]
    )
    
    # 训练模型
    final_model.fit(x_processed_selected, y_processed)
    
    # 评估模型
    y_pred = final_model.predict(x_processed_selected)
    train_rmse = np.sqrt(mean_squared_error(y_processed, y_pred))
    train_r2 = r2_score(y_processed, y_pred)
    
    print(f"发电站 {farm_id} 训练评估: RMSE = {train_rmse:.4f}, R² = {train_r2:.4f}")
    
    # 返回所有需要保存的组件
    model_package = {
        'model': final_model,
        'scaler': scaler,
        'feature_selector': selector,
        'selected_features': selected_features
    }
    
    return model_package

def predict(model_package, farm_id):
    """增强版预测函数"""
    # 解包模型组件
    final_model = model_package['model']
    scaler = model_package['scaler']
    selector = model_package['feature_selector']
    selected_features = model_package['selected_features']
    
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
    
    is_wind_farm = farm_id in WIND_FARMS
    # 预处理测试数据
    x_test = data_preprocess(x_df, is_train=False, is_wind_farm=is_wind_farm)
    
    # 确保只使用训练时的特征
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = scaler.feature_names_in_
        
        # 1. 删除测试集中训练时不存在的特征
        extra_features = [col for col in x_test.columns if col not in scaler_features]
        if extra_features:
            print(f"移除预测时多出的 {len(extra_features)} 个特征")
            x_test = x_test.drop(columns=extra_features)
        
        # 2. 添加测试集中缺失但训练时存在的特征(用0填充)
        missing_features = [col for col in scaler_features if col not in x_test.columns]
        if missing_features:
            print(f"添加预测时缺失的 {len(missing_features)} 个特征")
            for col in missing_features:
                x_test[col] = 0
        
        # 3. 确保特征顺序一致
        x_test = x_test[scaler_features]
    
    # 应用标准化
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index
    )
    
    # 确保使用与训练完全相同的特征集和顺序
    x_test_selected = pd.DataFrame(index=x_test_scaled.index)
    for feature in selected_features:
        if feature in x_test_scaled.columns:
            x_test_selected[feature] = x_test_scaled[feature]
        else:
            x_test_selected[feature] = 0
    
    # 预测
    pred_pw = final_model.predict(x_test_selected).flatten()
    
    # 创建预测序列
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0], periods=len(pred_pw), freq='h'))
    
    # 将预测重采样为15分钟
    res = pred.resample('15min').interpolate(method='cubic')
    
    # 修正预测值范围
    res[res < 0] = 0
    res[res > 1] = 1
    
    return res

# 深度学习模型类
class TimeSeriesCNN(nn.Module):
    """一维CNN用于提取时间序列的特征"""
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x形状: [batch, seq_len, input_dim] -> 转换为 [batch, input_dim, seq_len]
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        # 转回 [batch, seq_len, hidden_dim]
        return x.permute(0, 2, 1)

class LSTMAttention(nn.Module):
    """带注意力机制的LSTM"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=bidirectional, dropout=0.2)
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x形状: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_dim*num_directions]
        
        # 计算注意力权重
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 应用注意力权重
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden_dim*num_directions]
        return context

class HybridWindPowerModel(nn.Module):
    """风电场混合深度学习模型，结合CNN和LSTM with Attention"""
    def __init__(self, input_dim, seq_len, hidden_dim=64, fc_dim=32):
        super().__init__()
        self.cnn = TimeSeriesCNN(input_dim, hidden_dim)
        self.lstm_attn = LSTMAttention(hidden_dim, hidden_dim)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x形状: [batch, seq_len, input_dim]
        cnn_out = self.cnn(x)
        lstm_out = self.lstm_attn(cnn_out)
        
        x = self.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)

class HybridSolarPowerModel(nn.Module):
    """光伏场站混合深度学习模型，注重日照和时间特性"""
    def __init__(self, input_dim, seq_len, hidden_dim=64, fc_dim=32):
        super().__init__()
        self.cnn = TimeSeriesCNN(input_dim, hidden_dim)
        
        # 添加Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)
        self.dropout = nn.Dropout(0.3)  # 光伏模型使用更高的dropout防止过拟合
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x形状: [batch, seq_len, input_dim]
        cnn_out = self.cnn(x)
        
        # 使用Transformer处理序列
        transformer_out = self.transformer(cnn_out)
        
        # 取序列中最后一个时间步的输出
        last_hidden = transformer_out[:, -1, :]
        
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)

class PhysicsGuidedLinearLayer(nn.Module):
    """物理引导的线性层，将已知的物理关系融入模型"""
    def __init__(self, input_dim, output_dim, physics_guides):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        
        # 使用物理知识初始化权重
        with torch.no_grad():
            for i, j, value in physics_guides:
                if i < input_dim and j < output_dim:
                    self.linear.weight[j, i] = value
    
    def forward(self, x):
        return self.linear(x)

class EnsembleModel(nn.Module):
    """集成多个基础模型的输出"""
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        
        # 应用权重
        weights = torch.softmax(self.weights, dim=0).unsqueeze(-1).unsqueeze(-1)
        weighted_output = torch.sum(outputs * weights, dim=0)
        
        return weighted_output.squeeze()

def prepare_dl_dataset(x, y=None, seq_length=24, stride=1, is_train=True):
    """准备深度学习模型的输入数据，将数据组织成时序窗口的形式"""
    # 确保索引是连续的时间
    x = x.copy()
    
    n_samples, n_features = x.shape
    
    if n_samples < seq_length:
        raise ValueError(f"输入数据的样本数({n_samples})小于序列长度({seq_length})")
    
    # 创建时间窗口序列
    x_windows = []
    y_values = []
    
    for i in range(0, n_samples - seq_length + 1, stride):
        x_win = x.iloc[i:i+seq_length].values
        x_windows.append(x_win)
        
        if is_train and y is not None:
            # 使用窗口结束时刻的值作为目标
            y_values.append(y.iloc[i+seq_length-1])
    
    x_tensor = torch.FloatTensor(np.array(x_windows))
    
    if is_train and y is not None:
        y_tensor = torch.FloatTensor(np.array(y_values))
        return x_tensor, y_tensor
    else:
        return x_tensor

def train_dl_model(model, x_train, y_train, epochs=100, batch_size=32, lr=0.001, patience=10, val_ratio=0.2):
    """训练深度学习模型，优化CUDA内存使用"""
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 将模型移至设备
    model.to(device)
    
    # 划分训练集和验证集
    val_size = int(len(x_train) * val_ratio)
    train_size = len(x_train) - val_size
    
    indices = list(range(len(x_train)))
    # 时序数据，按时间顺序划分，而不是随机划分
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 使用try-except确保张量正确地移至GPU
    try:
        train_x = x_train[train_indices].to(device)
        train_y = y_train[train_indices].to(device)
        val_x = x_train[val_indices].to(device)
        val_y = y_train[val_indices].to(device)
    except RuntimeError as e:
        print(f"将数据移至GPU时出错: {e}")
        print("尝试减小批量大小或使用CPU")
        # 如果GPU内存不足，尝试在CPU上运行
        device_fallback = torch.device('cpu')
        model.to(device_fallback)
        train_x = x_train[train_indices]
        train_y = y_train[train_indices]
        val_x = x_train[val_indices]
        val_y = y_train[val_indices]
    
    # 创建DataLoader，增加num_workers以加速数据加载
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    # 使用tqdm显示进度
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        # 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_x, batch_y in progress_bar:
            # 确保数据在正确的设备上
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y).item()
            
            # 计算验证集指标
            val_mae = torch.mean(torch.abs(val_outputs - val_y)).item()
            val_rmse = torch.sqrt(torch.mean((val_outputs - val_y)**2)).item()
            
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss

def predict_with_dl_model(model, x_test, scaler=None):
    """使用深度学习模型进行预测"""
    model.to(device)
    model.eval()
    
    x_test = x_test.to(device)
    
    with torch.no_grad():
        predictions = model(x_test)
    
    return predictions.cpu().numpy()

def create_dl_data_pipeline(x_df, y_df=None, seq_length=24, is_train=True, is_wind_farm=True):
    """创建深度学习数据管道，包括特征处理和序列化"""
    # 特征工程和预处理
    if is_train:
        x_processed, y_processed = data_preprocess(x_df, y_df, is_train=True)
        
        # 特征缩放
        scaler = MinMaxScaler()
        x_scaled = pd.DataFrame(
            scaler.fit_transform(x_processed),
            columns=x_processed.columns,
            index=x_processed.index
        )
        
        # 准备序列数据
        x_tensor, y_tensor = prepare_dl_dataset(
            x_scaled, y_processed, seq_length=seq_length, is_train=True
        )
        
        return x_tensor, y_tensor, scaler, x_processed.columns
    else:
        x_processed = data_preprocess(x_df, is_train=False)
        
        # 使用训练时的scaler进行变换
        if 'scaler' in globals() and globals()['scaler'] is not None:
            scaler = globals()['scaler']
            x_scaled = pd.DataFrame(
                scaler.transform(x_processed),
                columns=x_processed.columns,
                index=x_processed.index
            )
        else:
            # 如果没有保存scaler，创建一个新的
            scaler = MinMaxScaler()
            x_scaled = pd.DataFrame(
                scaler.fit_transform(x_processed),
                columns=x_processed.columns,
                index=x_processed.index
            )
        
        # 准备序列数据
        x_tensor = prepare_dl_dataset(
            x_scaled, seq_length=seq_length, is_train=False
        )
        
        return x_tensor, scaler

def resample_predictions(hourly_preds, original_index, target_freq='15min', method='cubic_spline'):
    """将小时级预测重采样为15分钟级，并使用更复杂的插值方法"""
    hourly_series = pd.Series(hourly_preds, index=original_index)
    
    if method == 'cubic_spline':
        # 创建目标15分钟间隔的时间索引
        target_index = pd.date_range(
            start=hourly_series.index[0], 
            end=hourly_series.index[-1] + pd.Timedelta(hours=1) - pd.Timedelta(minutes=15), 
            freq='15min'
        )
        
        # 使用三次样条插值
        x_original = np.arange(len(hourly_series))
        y_original = hourly_series.values
        x_target = np.linspace(0, len(hourly_series) - 1, len(target_index))
        
        cs = CubicSpline(x_original, y_original)
        interpolated_values = cs(x_target)
        
        result = pd.Series(interpolated_values, index=target_index)
    else:
        # 使用pandas的resample和interpolate
        result = hourly_series.resample(target_freq).interpolate(method='cubic')
    
    # 确保值在有效范围内
    result = np.clip(result, 0, 1)
    
    return result

def train_deep_learning(farm_id):
    """训练深度学习模型的函数"""
    print(f"开始训练发电站 {farm_id} 的深度学习模型...")
    
    # 判断是风电场还是光伏场站
    is_wind_farm = farm_id in WIND_FARMS
    farm_type = "风电场" if is_wind_farm else "光伏场站"
    print(f"场站类型: {farm_type}")
    
    # 读取和准备数据
    x_df = pd.DataFrame()
    nwp_train_path = f'training/middle_school/TRAIN/nwp_data_train/{farm_id}'
    
    # 处理NWP数据，提取更多空间格点信息
    for nwp in nwps:
        nwp_path = os.path.join(nwp_train_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        
        # 扩大提取范围以捕获更多信息
        u = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                         channel=['u100']).data.values.reshape(365 * 24, 25)
        v = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                     channel=['v100']).data.values.reshape(365 * 24, 25)
        
        # 创建基本特征
        u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
        v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
        
        # 添加风向角度特征
        wd = np.arctan2(v, u) * 180 / np.pi
        wd = np.where(wd < 0, wd + 360, wd)
        wd_df = pd.DataFrame(wd, columns=[f"{nwp}_wd_{i}" for i in range(wd.shape[1])])
        
        # 提取其他气象变量（根据场站类型）
        additional_features = []
        
        if is_wind_farm:
            # 风电场额外特征
            try:
                t2m = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                              channel=['t2m']).data.values.reshape(365 * 24, 25)
                t2m_df = pd.DataFrame(t2m, columns=[f"{nwp}_t2m_{i}" for i in range(t2m.shape[1])])
                additional_features.append(t2m_df)
                
                # 添加气压特征
                if 'sp' in nwp_data.channel.values:
                    sp = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                                  channel=['sp']).data.values.reshape(365 * 24, 25)
                    sp_df = pd.DataFrame(sp, columns=[f"{nwp}_sp_{i}" for i in range(sp.shape[1])])
                    additional_features.append(sp_df)
                elif 'msl' in nwp_data.channel.values:
                    msl = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                                  channel=['msl']).data.values.reshape(365 * 24, 25)
                    msl_df = pd.DataFrame(msl, columns=[f"{nwp}_msl_{i}" for i in range(msl.shape[1])])
                    additional_features.append(msl_df)
            except Exception as e:
                print(f"提取风电场附加特征时出错: {e}")
        else:
            # 光伏场站额外特征
            try:
                # 光伏场站需要辐照度和云量特征
                for var in ['poai', 'ghi', 'tcc', 't2m']:
                    if var in nwp_data.channel.values:
                        feat = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                                   channel=[var]).data.values.reshape(365 * 24, 25)
                        feat_df = pd.DataFrame(feat, columns=[f"{nwp}_{var}_{i}" for i in range(feat.shape[1])])
                        additional_features.append(feat_df)
            except Exception as e:
                print(f"提取光伏场站附加特征时出错: {e}")
        
        # 合并所有特征
        nwp_dfs = [u_df, v_df, ws_df, wd_df] + additional_features
        nwp_df = pd.concat(nwp_dfs, axis=1)
        x_df = pd.concat([x_df, nwp_df], axis=1)
    
    x_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')
    
    # 读取目标变量
    y_df = pd.read_csv(os.path.join(fact_path,f'{farm_id}_normalization_train.csv'), index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']
    
    # 添加滞后特征
    x_df = add_lag_features(x_df)
    
    # 创建深度学习数据管道
    seq_length = 24  # 使用24小时的时间窗口
    x_tensor, y_tensor, scaler, feature_names = create_dl_data_pipeline(
        x_df, y_df, seq_length=seq_length, is_train=True, is_wind_farm=is_wind_farm
    )
    
    # 创建模型
    input_dim = x_tensor.shape[2]
    if is_wind_farm:
        model = HybridWindPowerModel(input_dim=input_dim, seq_len=seq_length)
    else:
        model = HybridSolarPowerModel(input_dim=input_dim, seq_len=seq_length)
    
    # 训练模型
    trained_model, val_loss = train_dl_model(
        model, x_tensor, y_tensor, 
        epochs=100, batch_size=64, lr=0.001, patience=10
    )
    
    # 计算评估指标
    model.eval()
    x_tensor_device = x_tensor.to(device)
    y_tensor_device = y_tensor.to(device)
    
    with torch.no_grad():
        y_pred = model(x_tensor_device).cpu().numpy()
        y_true = y_tensor.numpy()
        
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"发电站 {farm_id} 深度学习模型评估结果:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 返回模型包
    model_package = {
        'dl_model': trained_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'seq_length': seq_length,
        'is_wind_farm': is_wind_farm,
        'evaluation': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'val_loss': val_loss
        }
    }
    
    return model_package

def predict_with_deep_learning(model_package, farm_id):
    """使用深度学习模型进行预测，优化CUDA使用"""
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 解包模型包
    dl_model = model_package['dl_model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    seq_length = model_package['seq_length']
    is_wind_farm = model_package['is_wind_farm']
    
    # 将模型设置为评估模式并移至设备
    dl_model.to(device)
    dl_model.eval()
    
    # 读取测试数据
    x_df = pd.DataFrame()
    nwp_test_path = f'training/middle_school/TEST/nwp_data_test/{farm_id}'
    
    # 处理NWP数据，提取与训练时相同的空间格点
    for nwp in nwps:
        nwp_path = os.path.join(nwp_test_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        
        # 提取相同的特征和空间范围
        u = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                         channel=['u100']).data.values.reshape(31 * 24, 25)
        v = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                         channel=['v100']).data.values.reshape(31 * 24, 25)
        
        u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
        v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
        
        # 添加风向角度特征
        wd = np.arctan2(v, u) * 180 / np.pi
        wd = np.where(wd < 0, wd + 360, wd)
        wd_df = pd.DataFrame(wd, columns=[f"{nwp}_wd_{i}" for i in range(wd.shape[1])])
        
        # 提取其他气象变量（根据场站类型）
        additional_features = []
        
        if is_wind_farm:
            # 风电场额外特征
            try:
                t2m = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                              channel=['t2m']).data.values.reshape(31 * 24, 25)
                t2m_df = pd.DataFrame(t2m, columns=[f"{nwp}_t2m_{i}" for i in range(t2m.shape[1])])
                additional_features.append(t2m_df)
                
                # 添加气压特征
                if 'sp' in nwp_data.channel.values:
                    sp = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                                  channel=['sp']).data.values.reshape(31 * 24, 25)
                    sp_df = pd.DataFrame(sp, columns=[f"{nwp}_sp_{i}" for i in range(sp.shape[1])])
                    additional_features.append(sp_df)
                elif 'msl' in nwp_data.channel.values:
                    msl = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                                  channel=['msl']).data.values.reshape(31 * 24, 25)
                    msl_df = pd.DataFrame(msl, columns=[f"{nwp}_msl_{i}" for i in range(msl.shape[1])])
                    additional_features.append(msl_df)
            except Exception as e:
                print(f"提取风电场附加特征时出错: {e}")
        else:
            # 光伏场站额外特征
            try:
                # 光伏场站需要辐照度和云量特征
                for var in ['poai', 'ghi', 'tcc', 't2m']:
                    if var in nwp_data.channel.values:
                        feat = nwp_data.sel(lat=range(3,8), lon=range(3,8), lead_time=range(24),
                                   channel=[var]).data.values.reshape(31 * 24, 25)
                        feat_df = pd.DataFrame(feat, columns=[f"{nwp}_{var}_{i}" for i in range(feat.shape[1])])
                        additional_features.append(feat_df)
            except Exception as e:
                print(f"提取光伏场站附加特征时出错: {e}")
        
        # 合并所有特征
        nwp_dfs = [u_df, v_df, ws_df, wd_df] + additional_features
        nwp_df = pd.concat(nwp_dfs, axis=1)
        x_df = pd.concat([x_df, nwp_df], axis=1)
    
    # 设置时间索引
    x_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')
    
    # 添加滞后特征
    x_df = add_lag_features(x_df)
    
    # 数据预处理
    x_processed = data_preprocess(x_df, is_train=False)
    
    # 确保所有特征名称匹配
    missing_features = set(feature_names) - set(x_processed.columns)
    if missing_features:
        for feature in missing_features:
            x_processed[feature] = 0  # 用0填充缺失特征
    
    extra_features = set(x_processed.columns) - set(feature_names)
    if extra_features:
        x_processed = x_processed.drop(columns=extra_features)
    
    # 确保特征顺序一致
    x_processed = x_processed[feature_names]
    
    # 特征缩放
    x_scaled = pd.DataFrame(
        scaler.transform(x_processed),
        columns=x_processed.columns,
        index=x_processed.index
    )
    
    # 准备序列数据
    x_windows = []
    for i in range(len(x_scaled) - seq_length + 1):
        x_win = x_scaled.iloc[i:i+seq_length].values
        x_windows.append(x_win)
    
    x_tensor = torch.FloatTensor(np.array(x_windows)).to(device)
    
    # 预测
    dl_model.to(device)
    dl_model.eval()
    
    # 处理大型输入数据，分批次预测以避免内存溢出
    batch_size = 512  # 可根据GPU内存调整
    num_batches = (len(x_tensor) + batch_size - 1) // batch_size
    predictions = []
    
    with torch.no_grad():  # 确保不计算梯度
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(x_tensor))
            batch_input = x_tensor[start_idx:end_idx]
            
            # 预测并将结果移回CPU
            batch_output = dl_model(batch_input).cpu().numpy()
            predictions.append(batch_output)
    
    # 合并所有批次的预测结果
    predictions = np.concatenate(predictions) if len(predictions) > 1 else predictions[0]
    
    # 创建预测序列
    pred_index = x_df.index[seq_length-1:]
    if len(pred_index) != len(predictions):
        # 处理长度不匹配的情况
        min_len = min(len(pred_index), len(predictions))
        pred_index = pred_index[:min_len]
        predictions = predictions[:min_len]
    
    pred = pd.Series(predictions, index=pred_index)
    
    # 将预测重采样为15分钟
    res = resample_predictions(pred, pred_index)
    
    # 修正预测值范围
    res = np.clip(res, 0, 1)
    
    # 确保完成预测后释放GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return res

def main():
    """主程序执行函数，处理所有场站的训练与预测"""
    print("===== 新能源功率预报系统启动 =====")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示CUDA信息
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('result/output', exist_ok=True)
    
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
            ml_model_file = os.path.join(model_path, 'enhanced_model.pkl')
            dl_model_file = os.path.join(model_path, 'deep_learning_model.pkl')
            
            # 训练传统机器学习模型
            ml_model_package = None
            if os.path.exists(ml_model_file) and os.path.getsize(ml_model_file) > 0:
                print(f"加载已有机器学习模型: {ml_model_file}")
                with open(ml_model_file, "rb") as f:
                    ml_model_package = pickle.load(f)
            else:
                print(f"训练新的机器学习模型")
                ml_model_package = train(farm_id)
                with open(ml_model_file, "wb") as f:
                    pickle.dump(ml_model_package, f)
            
            # 训练深度学习模型
            dl_model_package = None
            if os.path.exists(dl_model_file) and os.path.getsize(dl_model_file) > 0:
                print(f"加载已有深度学习模型: {dl_model_file}")
                with open(dl_model_file, "rb") as f:
                    dl_model_package = pickle.load(f)
                    # 确保模型在正确设备上
                    if 'dl_model' in dl_model_package:
                        dl_model_package['dl_model'].to(device)
            else:
                print(f"训练新的深度学习模型")
                dl_model_package = train_deep_learning(farm_id)
                # 保存模型前将其移至CPU，确保可在不同设备加载
                dl_model_package['dl_model'].to('cpu')
                with open(dl_model_file, "wb") as f:
                    pickle.dump(dl_model_package, f)
                # 用完后再移回设备
                dl_model_package['dl_model'].to(device)
            
            # 使用两种模型分别进行预测
            print(f"使用机器学习模型生成预测...")
            ml_pred = predict(ml_model_package, farm_id)
            
            print(f"使用深度学习模型生成预测...")
            dl_pred = predict_with_deep_learning(dl_model_package, farm_id)
            
            # 集成预测结果（加权平均）
            # 根据两种模型在验证集上的表现分配权重
            if 'evaluation' in dl_model_package and 'r2' in dl_model_package['evaluation']:
                dl_r2 = dl_model_package['evaluation']['r2']
                # 根据R2分配权重，但确保即使R2很低也给予一定权重
                dl_weight = max(0.3, min(0.7, 0.3 + dl_r2 * 0.5))
                ml_weight = 1 - dl_weight
            else:
                # 默认权重
                dl_weight = 0.5
                ml_weight = 0.5
            
            # 清理每个站点处理完后的CUDA内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 确保两个预测具有相同的索引
            common_index = ml_pred.index.intersection(dl_pred.index)
            ml_pred = ml_pred.loc[common_index]
            dl_pred = dl_pred.loc[common_index]
            
            # 加权组合预测
            final_pred = ml_pred * ml_weight + dl_pred * dl_weight
            
            # 确保值在有效范围内
            final_pred = np.clip(final_pred, 0, 1)
            
            # 保存最终预测结果
            result_file = os.path.join('result/output', f'output{farm_id}.csv')
            final_pred.to_csv(result_file)
            print(f"预测结果已保存至: {result_file}")
            
            print(f"发电站 {farm_id} 处理完成")
            
        except Exception as e:
            print(f"处理发电站 {farm_id} 时发生错误: {e}")
            import traceback
            print(traceback.format_exc())
            # 出错时也清理CUDA内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 打包所有输出结果
    try:
        import zipfile
        output_files = [f'output{i}.csv' for i in farms]
        with zipfile.ZipFile('result/output.zip', 'w') as zipf:
            for file in output_files:
                file_path = os.path.join('result/output', file)
                if os.path.exists(file_path):
                    zipf.write(file_path, arcname=file)
        print(f"所有预测结果已打包至: result/output.zip")
    except Exception as e:
        print(f"打包输出结果时发生错误: {e}")
    
    print("\n===== 新能源功率预报系统运行完成 =====")

if __name__ == "__main__":
    main()