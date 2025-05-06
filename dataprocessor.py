import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, RobustScaler
from datetime import datetime, timedelta

class PowerGenerationDataset(Dataset):
    """发电量预测数据集类"""
    def __init__(self, features, targets=None, seq_length=24):
        """
        初始化数据集
        
        Args:
            features: 特征数据帧
            targets: 目标变量数据帧
            seq_length: 序列长度，用于序列模型
        """
        self.seq_length = seq_length
        
        if seq_length > 1:
            # 准备序列数据
            self.features, self.targets = self._prepare_sequences(features, targets)
        else:
            # 标准模式（非序列）
            self.features = torch.tensor(features.values, dtype=torch.float32)
            if targets is not None:
                self.targets = torch.tensor(targets.values, dtype=torch.float32)
                self.has_targets = True
            else:
                self.has_targets = False
                
    def _prepare_sequences(self, features, targets):
        """准备序列数据"""
        x_seqs = []
        y_seqs = []
        
        # 假设数据是按时间排序的
        for i in range(len(features) - self.seq_length + 1):
            x_seq = features.iloc[i:i+self.seq_length].values
            x_seqs.append(x_seq)
            
            if targets is not None:
                # 预测序列最后一个时间点的值
                y_seq = targets.iloc[i+self.seq_length-1]
                y_seqs.append(y_seq)
        
        x_tensor = torch.tensor(np.array(x_seqs), dtype=torch.float32)
        
        if targets is not None:
            y_tensor = torch.tensor(np.array(y_seqs), dtype=torch.float32)
            return x_tensor, y_tensor
        else:
            return x_tensor, None
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.has_targets:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]

def smooth_power_values(y_series, window_length=11, polyorder=3):
    """使用Savitzky-Golay滤波平滑功率值"""
    if len(y_series) > window_length:
        try:
            y_smooth = savgol_filter(y_series, window_length, polyorder)
            return pd.Series(y_smooth, index=y_series.index)
        except:
            return y_series
    return y_series

def add_weather_channels(x_df, farm_id, is_wind_farm=True):
    """添加更多气象变量"""
    # 获取nwp数据的其他通道
    if is_wind_farm:
        channels = ['t2m', 'sp'] if farm_id <= 3 else ['t2m', 'sp', 'tp']
    else:
        channels = ['t2m', 'poai', 'ghi', 'tcc']
        
    nwp_train_path = f'training/middle_school/TRAIN/nwp_data_train/{farm_id}'
    
    # 尝试添加更多气象通道
    try:
        for nwp in ['NWP_1', 'NWP_2', 'NWP_3']:
            for channel in channels:
                try:
                    nwp_data = xr.open_dataset(f"{nwp_train_path}/{nwp}/{channel}_0.nc")
                    values = nwp_data.sel(
                        lat=range(5, 6), 
                        lon=range(5, 6), 
                        lead_time=range(24)
                    ).data.values.flatten()
                    
                    x_df[f'{nwp}_{channel}'] = values
                except:
                    continue
    except:
        pass
        
    return x_df

def create_time_features(df):
    """创建高级时间特征"""
    df_copy = df.copy()
    
    # 基本时间特征
    df_copy['hour'] = df_copy.index.hour
    df_copy['day'] = df_copy.index.day
    df_copy['month'] = df_copy.index.month
    df_copy['dayofweek'] = df_copy.index.dayofweek
    
    # 周期性时间特征
    df_copy['sin_hour'] = np.sin(2 * np.pi * df_copy.index.hour / 24)
    df_copy['cos_hour'] = np.cos(2 * np.pi * df_copy.index.hour / 24)
    df_copy['sin_month'] = np.sin(2 * np.pi * df_copy.index.month / 12)
    df_copy['cos_month'] = np.cos(2 * np.pi * df_copy.index.month / 12)
    
    # 白天/黑夜指示器（估算）
    df_copy['is_daylight'] = ((df_copy.index.hour >= 6) & (df_copy.index.hour <= 18)).astype(int)
    
    # 季节性特征
    df_copy['season'] = (df_copy.index.month % 12 + 3) // 3
    df_copy['sin_season'] = np.sin(2 * np.pi * df_copy['season'] / 4)
    df_copy['cos_season'] = np.cos(2 * np.pi * df_copy['season'] / 4)
    
    return df_copy

def preprocess_for_deep_learning(x_df, y_df=None, is_train=True, is_wind_farm=True, seq_length=1):
    """为深度学习模型准备数据"""
    # 增强时间特征
    x_df = create_time_features(x_df)
    
    # 添加特定场站类型的特征
    if is_wind_farm:
        # 风电场特有特征
        pass
    else:
        # 光伏场站特有特征
        pass
    
    # 处理缺失值
    x_df = x_df.fillna(method='ffill')
    x_df = x_df.fillna(method='bfill')
    
    # 标准化
    scaler = RobustScaler()  # 使用稳健缩放器以处理异常值
    
    if is_train and y_df is not None:
        # 训练数据
        x_scaled = pd.DataFrame(
            scaler.fit_transform(x_df), 
            columns=x_df.columns,
            index=x_df.index
        )
        
        # 平滑目标值
        y_smooth = smooth_power_values(y_df.values.flatten())
        y_smooth_series = pd.Series(y_smooth, index=y_df.index)
        
        # 创建数据集
        dataset = PowerGenerationDataset(x_scaled, y_smooth_series, seq_length=seq_length)
        
        return dataset, scaler
    else:
        # 测试数据
        x_scaled = pd.DataFrame(
            scaler.transform(x_df),
            columns=x_df.columns,
            index=x_df.index
        )
        
        # 创建数据集
        dataset = PowerGenerationDataset(x_scaled, seq_length=seq_length)
        
        return dataset
