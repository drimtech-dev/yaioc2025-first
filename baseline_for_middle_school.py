import os
import pickle
import warnings
from datetime import datetime, timedelta
import math

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_regression
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.interpolate import CubicSpline

warnings.filterwarnings('ignore')

nwps = ['NWP_1','NWP_2','NWP_3']
fact_path = 'training/middle_school/TRAIN/fact_data'

# 区分风电场和光伏场站
WIND_FARMS = [1, 2, 3, 4, 5]
SOLAR_FARMS = [6, 7, 8, 9, 10]

def add_time_features(df):
    """增强版时间特征"""
    df_copy = df.copy()
    
    # 基础时间特征
    df_copy['hour'] = df_copy.index.hour
    df_copy['day'] = df_copy.index.day
    df_copy['month'] = df_copy.index.month
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['dayofyear'] = df_copy.index.dayofyear
    df_copy['is_weekend'] = df_copy['dayofweek'] >= 5
    
    # 周期性时间特征
    df_copy['sin_hour'] = np.sin(2 * np.pi * df_copy.index.hour / 24)
    df_copy['cos_hour'] = np.cos(2 * np.pi * df_copy.index.hour / 24)
    df_copy['sin_month'] = np.sin(2 * np.pi * df_copy.index.month / 12)
    df_copy['cos_month'] = np.cos(2 * np.pi * df_copy.index.month / 12)
    df_copy['sin_day'] = np.sin(2 * np.pi * df_copy.index.day / 31)
    df_copy['cos_day'] = np.cos(2 * np.pi * df_copy.index.day / 31)
    
    # 光照相关特征（对光伏发电特别重要）
    # 简化计算日照时长和太阳高度角，实际应用中可以使用更精确的天文算法
    latitude = 35.0  # 假设场站在北纬35度左右，根据实际情况调整
    
    # 计算太阳高度角（简化版本）
    df_copy['day_length'] = df_copy.apply(
        lambda x: estimate_day_length(x['dayofyear'], latitude), axis=1
    )
    df_copy['solar_zenith'] = df_copy.apply(
        lambda x: calculate_solar_zenith(x['dayofyear'], x['hour'], latitude), axis=1
    )
    df_copy['solar_elevation'] = 90 - df_copy['solar_zenith']
    
    # 日出日落相关特征（简化）
    df_copy['hours_since_sunrise'] = df_copy.apply(
        lambda x: (x['hour'] - (12 - x['day_length']/2)) % 24, axis=1
    )
    df_copy['hours_until_sunset'] = df_copy.apply(
        lambda x: ((12 + x['day_length']/2) - x['hour']) % 24, axis=1
    )
    
    # 是否白天特征
    df_copy['is_daylight'] = df_copy.apply(
        lambda x: is_daylight(x['hour'], x['day_length']), axis=1
    )
    
    return df_copy

def estimate_day_length(dayofyear, latitude):
    """估算日长，单位：小时"""
    # 简化的太阳赤纬角计算
    declination = 23.45 * np.sin(np.radians(360/365 * (dayofyear - 81)))
    
    # 日出日落时间计算
    cos_hour_angle = -np.tan(np.radians(latitude)) * np.tan(np.radians(declination))
    
    # 确保值在有效范围内
    cos_hour_angle = np.clip(cos_hour_angle, -1.0, 1.0)
    
    # 计算日出日落时角
    hour_angle = np.degrees(np.arccos(cos_hour_angle))
    
    # 日长（小时）
    day_length = 2 * hour_angle / 15.0
    
    return day_length

def calculate_solar_zenith(dayofyear, hour, latitude):
    """计算太阳天顶角"""
    # 太阳赤纬角
    declination = 23.45 * np.sin(np.radians(360/365 * (dayofyear - 81)))
    
    # 时角（每小时15度）
    hour_angle = 15 * (hour - 12)
    
    # 计算太阳天顶角的余弦值
    cos_zenith = (np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) + 
                  np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * 
                  np.cos(np.radians(hour_angle)))
    
    # 确保值在有效范围内
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    
    # 计算天顶角（度）
    zenith = np.degrees(np.arccos(cos_zenith))
    
    return zenith

def is_daylight(hour, day_length):
    """判断是否在日照时段"""
    sunrise = 12 - day_length/2
    sunset = 12 + day_length/2
    return sunrise <= hour <= sunset

def add_weather_derivatives(df, is_wind_farm=True):
    """增强版气象衍生特征，区分风电场和光伏场站"""
    df_copy = df.copy()
    
    # 共有特征
    # 计算风速的统计特征
    ws_cols = [col for col in df_copy.columns if '_ws_' in col]
    if ws_cols:
        df_copy['ws_mean'] = df_copy[ws_cols].mean(axis=1)
        df_copy['ws_std'] = df_copy[ws_cols].std(axis=1)
        df_copy['ws_min'] = df_copy[ws_cols].min(axis=1)
        df_copy['ws_max'] = df_copy[ws_cols].max(axis=1)
        df_copy['ws_range'] = df_copy['ws_max'] - df_copy['ws_min']
        df_copy['ws_median'] = df_copy[ws_cols].median(axis=1)
        df_copy['ws_q25'] = df_copy[ws_cols].quantile(0.25, axis=1)
        df_copy['ws_q75'] = df_copy[ws_cols].quantile(0.75, axis=1)
        df_copy['ws_iqr'] = df_copy['ws_q75'] - df_copy['ws_q25']
    
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
        
        # 计算风向的一致性（越接近1表示风向越一致）
        df_copy['wind_consistency'] = np.sqrt(
            np.square(df_copy[u_cols].mean(axis=1)) + 
            np.square(df_copy[v_cols].mean(axis=1))
        ) / df_copy[ws_cols].mean(axis=1)
        df_copy['wind_consistency'].fillna(0, inplace=True)
    
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
    
    # 计算各NWP来源之间的差异（模型不确定性指标）
    for nwp_pair in [('NWP_1', 'NWP_2'), ('NWP_1', 'NWP_3'), ('NWP_2', 'NWP_3')]:
        nwp1, nwp2 = nwp_pair
        
        # 计算风速差异
        ws1_cols = [col for col in df_copy.columns if f'{nwp1}_ws_' in col]
        ws2_cols = [col for col in df_copy.columns if f'{nwp2}_ws_' in col]
        
        if ws1_cols and ws2_cols and len(ws1_cols) == len(ws2_cols):
            df_copy[f'{nwp1}_{nwp2}_ws_diff'] = np.mean([
                df_copy[ws1_cols[i]] - df_copy[ws2_cols[i]] 
                for i in range(min(len(ws1_cols), len(ws2_cols)))
            ], axis=0)
    
    # 计算NWP不同来源的权重特征
    for var in ['ws', 'u', 'v']:
        for i in range(min(9, len([col for col in df_copy.columns if f'__{var}__' in col]))):
            nwp1_val = df_copy.get(f'NWP_1_{var}_{i}', 0)
            nwp2_val = df_copy.get(f'NWP_2_{var}_{i}', 0)
            nwp3_val = df_copy.get(f'NWP_3_{var}_{i}', 0)
            
            if isinstance(nwp1_val, pd.Series) and isinstance(nwp2_val, pd.Series) and isinstance(nwp3_val, pd.Series):
                df_copy[f'nwp_spread_{var}_{i}'] = np.std([nwp1_val, nwp2_val, nwp3_val], axis=0)
    
    return df_copy

def add_lag_features(df, lag_hours=[1, 2, 3, 6, 12, 24]):
    """增强版滞后特征"""
    df_copy = df.copy()
    
    # 基本滞后特征
    for col in df_copy.columns:
        for lag in lag_hours:
            df_copy[f'{col}_lag{lag}'] = df_copy[col].shift(lag)
    
    # 添加滚动统计特征
    important_cols = [col for col in df_copy.columns if any(x in col for x in ['ws', 'u', 'v', 'poai', 'ghi', 'tcc'])]
    windows = [3, 6, 12, 24]
    
    for col in important_cols[:20]:  # 限制列数以避免特征爆炸
        for window in windows:
            # 移动平均
            df_copy[f'{col}_rolling_mean_{window}'] = df_copy[col].rolling(window=window, min_periods=1).mean()
            # 移动标准差
            df_copy[f'{col}_rolling_std_{window}'] = df_copy[col].rolling(window=window, min_periods=1).std()
    
    # 添加差分特征
    for col in important_cols[:10]:
        df_copy[f'{col}_diff1'] = df_copy[col].diff(1)
        df_copy[f'{col}_diff24'] = df_copy[col].diff(24)
    
    return df_copy

def data_preprocess(x_df, y_df=None, is_train=True, is_wind_farm=True):
    """改进的数据预处理，区分风电场和光伏场站"""
    x_df = x_df.copy()
    
    # 添加时间特征
    x_df = add_time_features(x_df)
    
    # 添加气象衍生特征，区分风电场和光伏场站
    x_df = add_weather_derivatives(x_df, is_wind_farm=is_wind_farm)
    
    if is_train and y_df is not None:
        y_df = y_df.copy()
        
        # 清理数据前记录原始索引，用于后续分析
        original_indices = y_df.index
        
        # 清理数据
        x_df = x_df.dropna()
        y_df = y_df.dropna()
        
        # 记录删除的索引，可用于异常值分析
        removed_indices = [idx for idx in original_indices if idx not in y_df.index]
        if len(removed_indices) > 0:
            print(f"删除了 {len(removed_indices)} 条缺失数据记录")
        
        # 数据对扣
        ind = [i for i in y_df.index if i in x_df.index]
        x_df = x_df.loc[ind]
        y_df = y_df.loc[ind]
        
        # 处理异常值
        # 使用更严格的异常值检测
        if is_wind_farm:
            # 风电场的值应该在0-1之间，但也检测统计异常
            lower_bound = y_df['power'].quantile(0.001)
            upper_bound = min(1.0, y_df['power'].quantile(0.999))
        else:
            # 光伏场站白天和夜晚处理不同
            day_mask = x_df['is_daylight'] == 1
            if day_mask.sum() > 0:
                # 白天数据
                day_lower = y_df.loc[day_mask, 'power'].quantile(0.001)
                day_upper = min(1.0, y_df.loc[day_mask, 'power'].quantile(0.999))
                # 夜晚数据应该接近零
                night_upper = 0.1
                
                # 应用不同的约束
                y_df.loc[day_mask & (y_df['power'] < day_lower), 'power'] = day_lower
                y_df.loc[day_mask & (y_df['power'] > day_upper), 'power'] = day_upper
                y_df.loc[(~day_mask) & (y_df['power'] > night_upper), 'power'] = night_upper
            else:
                # 如果没有白天数据，使用通用约束
                lower_bound = 0
                upper_bound = min(1.0, y_df['power'].quantile(0.999))
                y_df[y_df < lower_bound] = lower_bound
                y_df[y_df > upper_bound] = upper_bound
        
        return x_df, y_df
    else:
        # 测试数据处理
        # 使用更健壮的缺失值填充策略
        # 首先尝试前向填充，然后后向填充
        x_df = x_df.fillna(method='ffill')
        x_df = x_df.fillna(method='bfill')
        
        # 对于仍然存在的缺失值，使用列均值填充
        for col in x_df.columns:
            if x_df[col].isna().any():
                x_df[col].fillna(x_df[col].mean(), inplace=True)
        
        return x_df

def create_model(is_wind_farm=True):
    """创建专门的风电或光伏预测模型"""
    # 为不同类型的场站设置不同的模型参数
    if is_wind_farm:
        lgb_params = {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 8,
            'num_leaves': 40,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'min_child_samples': 20
        }
        
        xgb_params = {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'min_child_weight': 3
        }
        
        rf_params = {
            'n_estimators': 150,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        gbr_params = {
            'n_estimators': 200,
            'learning_rate': 0.04,
            'max_depth': 6,
            'min_samples_split': 5,
            'random_state': 42
        }
    else:
        # 光伏模型参数
        lgb_params = {
            'n_estimators': 250,
            'learning_rate': 0.025,
            'max_depth': 7,
            'num_leaves': 35,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'random_state': 42,
            'min_child_samples': 15
        }
        
        xgb_params = {
            'n_estimators': 250,
            'learning_rate': 0.025,
            'max_depth': 7,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'random_state': 42,
            'min_child_weight': 2
        }
        
        rf_params = {
            'n_estimators': 120,
            'max_depth': 10,
            'min_samples_split': 4,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        gbr_params = {
            'n_estimators': 180,
            'learning_rate': 0.03,
            'max_depth': 5,
            'min_samples_split': 4,
            'random_state': 42
        }
    
    # 创建基础模型
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_rf = RandomForestRegressor(**rf_params)
    model_gbr = GradientBoostingRegressor(**gbr_params)
    
    # 创建更先进的集成模型：堆栈集成而非简单投票
    base_models = [
        ('lgb', model_lgb),
        ('xgb', model_xgb),
        ('rf', model_rf),
        ('gbr', model_gbr)
    ]
    
    # 堆栈集成的元模型
    meta_model = Ridge(alpha=0.5)
    
    # 创建堆栈集成
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=TimeSeriesSplit(n_splits=3),  # 时间序列交叉验证
        n_jobs=-1
    )
    
    return stacking_model

def train(farm_id):
    """增强版训练函数，针对风电场和光伏场站区分处理"""
    print(f"开始训练发电站 {farm_id} 的模型...")
    
    # 判断是风电场还是光伏场站
    is_wind_farm = farm_id in WIND_FARMS
    farm_type = "风电场" if is_wind_farm else "光伏场站"
    print(f"场站类型: {farm_type}")
    
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
        
        # 提取其他气象变量（根据场站类型）
        if is_wind_farm:
            # 风电场重点关注风资源
            other_vars = ['t2m', 'sp']
        else:
            # 光伏场站重点关注辐照和云量
            other_vars = ['t2m', 'ghi', 'poai', 'tcc']
        
        other_dfs = []
        for var in other_vars:
            try:
                var_data = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                        channel=[var]).data.values.reshape(365 * 24, 9)
                var_df = pd.DataFrame(var_data, columns=[f"{nwp}_{var}_{i}" for i in range(var_data.shape[1])])
                other_dfs.append(var_df)
            except (KeyError, ValueError) as e:
                print(f"无法提取变量 {var} 从 {nwp}: {e}")
        
        # 合并所有特征
        nwp_df_parts = [u_df, v_df, ws_df, wd_df] + other_dfs
        nwp_df = pd.concat(nwp_df_parts, axis=1)
        x_df = pd.concat([x_df, nwp_df], axis=1)
    
    x_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')
    
    # 读取目标变量
    y_df = pd.read_csv(os.path.join(fact_path,f'{farm_id}_normalization_train.csv'), index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']
    
    # 添加滞后特征
    x_df = add_lag_features(x_df)
    
    # 预处理数据，区分风电场和光伏场站
    x_processed, y_processed = data_preprocess(x_df, y_df, is_train=True, is_wind_farm=is_wind_farm)
    
    # 特征标准化（使用RobustScaler更健壮地处理异常值）
    scaler = RobustScaler()
    x_processed_scaled = pd.DataFrame(
        scaler.fit_transform(x_processed), 
        columns=x_processed.columns,
        index=x_processed.index
    )
    
    # 特征选择
    # 使用基于特定场站类型的特征选择策略
    if is_wind_farm:
        # 风电场特征选择
        selector = SelectFromModel(
            RandomForestRegressor(n_estimators=150, random_state=42),
            threshold='1.5*mean'  # 调整阈值以保留更多相关特征
        )
    else:
        # 光伏场站特征选择
        selector = SelectFromModel(
            GradientBoostingRegressor(n_estimators=150, random_state=42),
            threshold='1.25*mean'  # 光伏可能更依赖某些特定特征
        )
    
    selector.fit(x_processed_scaled, y_processed)
    selected_features = x_processed_scaled.columns[selector.get_support()]
    x_processed_selected = x_processed_scaled[selected_features]
    
    print(f"选择了 {len(selected_features)} 个特征，从总共 {x_processed_scaled.shape[1]} 个")
    
    # 创建专门的模型
    final_model = create_model(is_wind_farm=is_wind_farm)
    
    # 训练模型
    final_model.fit(x_processed_selected, y_processed)
    
    # 评估模型
    y_pred = final_model.predict(x_processed_selected)
    train_rmse = np.sqrt(mean_squared_error(y_processed, y_pred))
    train_mae = mean_absolute_error(y_processed, y_pred)
    train_r2 = r2_score(y_processed, y_pred)
    
    # 计算模型在该场站的训练准确率
    accuracy = 1 - np.sum(np.abs(y_processed - y_pred)) / len(y_processed)
    
    print(f"场站 {farm_id} 训练评估: RMSE = {train_rmse:.4f}, MAE = {train_mae:.4f}, R² = {train_r2:.4f}, 准确率 = {accuracy:.4f}")
    
    # 返回所有需要保存的组件
    model_package = {
        'model': final_model,
        'scaler': scaler,
        'feature_selector': selector,
        'selected_features': selected_features,
        'is_wind_farm': is_wind_farm
    }
    
    return model_package

def predict(model_package, farm_id):
    """增强版预测函数，针对风电场和光伏场站区分处理"""
    # 解包模型组件
    final_model = model_package['model']
    scaler = model_package['scaler']
    selector = model_package['feature_selector']
    selected_features = model_package['selected_features']
    is_wind_farm = model_package['is_wind_farm']

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
        
        # 提取其他气象变量（根据场站类型）
        if is_wind_farm:
            other_vars = ['t2m', 'sp']
        else:
            other_vars = ['t2m', 'ghi', 'poai', 'tcc']
        
        other_dfs = []
        for var in other_vars:
            try:
                var_data = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                        channel=[var]).data.values.reshape(31 * 24, 9)
                var_df = pd.DataFrame(var_data, columns=[f"{nwp}_{var}_{i}" for i in range(var_data.shape[1])])
                other_dfs.append(var_df)
            except (KeyError, ValueError) as e:
                print(f"无法提取变量 {var} 从 {nwp}: {e}")

        # 合并所有特征
        nwp_df_parts = [u_df, v_df, ws_df, wd_df] + other_dfs
        nwp_df = pd.concat(nwp_df_parts, axis=1)
        x_df = pd.concat([x_df, nwp_df], axis=1)

    x_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')

    # 添加滞后特征
    x_df = add_lag_features(x_df)

    # 预处理测试数据
    x_test = data_preprocess(x_df, is_train=False, is_wind_farm=is_wind_farm)

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
            # 如果缺少某特征，用0填充
            x_test_selected[feature] = 0

    # 预测
    pred_pw = final_model.predict(x_test_selected).flatten()

    # 创建预测序列
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0], periods=len(pred_pw), freq='h'))

    # 将预测重采样为15分钟，根据不同场站类型使用不同的插值方法
    if is_wind_farm:
        # 风电场使用三次样条插值，保持平滑过渡
        res = interpolate_to_15min(pred, method='cubic')
    else:
        # 光伏场站使用日照感知型插值
        res = solar_aware_interpolation(pred, x_test)

    # 修正预测值范围
    res[res < 0] = 0
    res[res > 1] = 1

    return res

def interpolate_to_15min(hourly_series, method='cubic'):
    """高级插值到15分钟分辨率"""
    # 创建目标15分钟间隔的时间索引
    start_time = hourly_series.index[0]
    end_time = hourly_series.index[-1] + pd.Timedelta(hours=1) - pd.Timedelta(minutes=15)
    target_index = pd.date_range(start=start_time, end=end_time, freq='15min')
    
    # 转换原始小时数据为数值数组
    x_original = np.arange(len(hourly_series))
    y_original = hourly_series.values
    
    # 目标15分钟的x坐标
    x_target = np.linspace(0, len(hourly_series) - 1, len(target_index))
    
    # 使用三次样条插值
    cs = CubicSpline(x_original, y_original)
    interpolated_values = cs(x_target)
    
    # 创建15分钟分辨率的结果Series
    result = pd.Series(interpolated_values, index=target_index)
    
    return result

def solar_aware_interpolation(hourly_series, features_df):
    """针对光伏场站的日照感知型插值"""
    # 创建目标15分钟间隔的时间索引
    start_time = hourly_series.index[0]
    end_time = hourly_series.index[-1] + pd.Timedelta(hours=1) - pd.Timedelta(minutes=15)