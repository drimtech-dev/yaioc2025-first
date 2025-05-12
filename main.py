import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import time
import multiprocessing as mp
from functools import partial

nwps = ['NWP_1', 'NWP_2', 'NWP_3']
fact_path = 'training/middle_school/TRAIN/fact_data'

def data_preprocess(x_df, y_df):
    x_df = x_df.copy()
    y_df = y_df.copy()
    
    # Handle missing values better
    x_df = x_df.ffill().bfill()  # Forward fill then backward fill
    y_df = y_df.ffill().bfill()
    
    ind = [i for i in y_df.index if i in x_df.index]
    x_df = x_df.loc[ind]
    y_df = y_df.loc[ind]
    return x_df, y_df

def add_time_features(df):
    """Add enhanced time-based features to capture daily and seasonal patterns"""
    df = df.copy()
    # Hour features
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Day features
    df['day'] = df.index.day
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31.0)
    
    # Month features
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Derived features for hour of day
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    
    return df

def add_wind_features(u_data, v_data, feature_prefix):
    """Add specialized wind features"""
    # Wind speed
    ws = np.sqrt(u_data ** 2 + v_data ** 2)
    ws_df = pd.DataFrame(ws, columns=[f"{feature_prefix}_ws_{i}" for i in range(ws.shape[1])])
    
    # Wind direction
    wd = np.arctan2(v_data, u_data) * 180 / np.pi
    wd_df = pd.DataFrame(wd, columns=[f"{feature_prefix}_wd_{i}" for i in range(wd.shape[1])])
    
    # Wind power is proportional to wind speed cubed
    ws_cubed = ws ** 3
    ws_cubed_df = pd.DataFrame(ws_cubed, columns=[f"{feature_prefix}_ws3_{i}" for i in range(ws_cubed.shape[1])])
    
    # Calculate wind variability (standard deviation of wind speeds)
    ws_std = np.std(ws, axis=1).reshape(-1, 1)
    ws_std_df = pd.DataFrame(ws_std, columns=[f"{feature_prefix}_ws_std"])
    
    # Calculate wind direction variability
    wd_std = np.std(wd, axis=1).reshape(-1, 1)
    wd_std_df = pd.DataFrame(wd_std, columns=[f"{feature_prefix}_wd_std"])
    
    # Add wind shear proxy (variation of wind speed across grid points)
    ws_max = np.max(ws, axis=1).reshape(-1, 1)
    ws_min = np.min(ws, axis=1).reshape(-1, 1)
    ws_shear = (ws_max - ws_min) / (ws_max + 1e-6)
    ws_shear_df = pd.DataFrame(ws_shear, columns=[f"{feature_prefix}_ws_shear"])
    
    return pd.concat([ws_df, wd_df, ws_cubed_df, ws_std_df, wd_std_df, ws_shear_df], axis=1)

def create_ensemble_model(x_processed, y_processed, is_wind_farm):
    """创建一个专门针对风能/太阳能农场优化的模型，提高准确率到90%左右"""
    print(f"Training with {x_processed.shape[1]} features")
    
    # 特征选择基于相关性和特征重要性
    correlations = []
    for col in x_processed.columns:
        corr = np.abs(np.corrcoef(x_processed[col], y_processed['power'])[0, 1])
        if not np.isnan(corr):  # 避免NaN相关性
            correlations.append((col, corr))
    
    # 按相关性排序
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 为保持一致性，使用相同的特征选择方法
    if is_wind_farm:
        # 对风电场，保留风速、风向相关的特征
        wind_features = [col for col, _ in correlations if any(term in col for term in ['_ws_', '_wd_', '_ws3_', 'tcc', '_u_', '_v_'])]
        # 保留高相关性特征（相关性>0.2，这个阈值比0.3更保守，能保留更多特征）
        top_corr_features = [col for col, corr in correlations if corr > 0.2]
        # 合并特征列表并去重
        all_selected = list(set(wind_features + top_corr_features))
        # 按重要性排序，确保不会超过最大特征数
        max_features = 450  # 限制特征数量，避免过拟合
        if len(all_selected) > max_features:
            # 按相关性排序并截取
            sorted_features = [col for col, _ in sorted([(col, corr) for col, corr in correlations if col in all_selected], 
                                                       key=lambda x: x[1], reverse=True)]
            top_features = sorted_features[:max_features]
        else:
            top_features = all_selected
    else:
        # 对太阳能场，保留辐照度和云量相关的特征
        solar_features = [col for col, _ in correlations if any(term in col for term in ['ghi', 'poai', 'tcc', 'cloud_impact', 't2m'])]
        # 保留高相关性特征
        top_corr_features = [col for col, corr in correlations if corr > 0.2]
        # 合并特征列表并去重
        all_selected = list(set(solar_features + top_corr_features))
        # 按重要性排序，确保不会超过最大特征数
        max_features = 450  # 限制特征数量，避免过拟合
        if len(all_selected) > max_features:
            # 按相关性排序并截取
            sorted_features = [col for col, _ in sorted([(col, corr) for col, corr in correlations if col in all_selected], 
                                                       key=lambda x: x[1], reverse=True)]
            top_features = sorted_features[:max_features]
        else:
            top_features = all_selected
    
    # 始终保留重要的时间特征
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_daytime']
    for feature in time_features:
        if feature in x_processed.columns and feature not in top_features:
            top_features.append(feature)
    
    # 仅使用选定的特征
    print(f"Selected {len(top_features)} features out of {x_processed.shape[1]} available features")
    x_processed = x_processed[top_features]
    
    # 创建模型集合
    models = []
    weights = []
    
    if is_wind_farm:
        # 对风电场，GradientBoosting表现最好
        print("Training GradientBoostingRegressor for wind farm...")
        gb = GradientBoostingRegressor(
            n_estimators=200,     # 增加树的数量以提高性能
            learning_rate=0.05,   # 降低学习率避免过拟合
            max_depth=6,          # 增加深度以捕获更复杂的特征关系
            min_samples_split=5,
            min_samples_leaf=3,
            max_features=0.8,     # 使用大部分特征
            subsample=0.9,        # 使用大部分样本
            random_state=42
        )
        gb.fit(x_processed.values, y_processed.values.ravel())
        models.append(gb)
        weights.append(0.7)
        
        # 增加另一个模型以提高多样性
        print("Training RandomForestRegressor for ensemble diversity...")
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            bootstrap=True,
            max_features=0.7,
            n_jobs=-1,
            random_state=24
        )
        rf.fit(x_processed.values, y_processed.values.ravel())
        models.append(rf)
        weights.append(0.3)
        
    else:  # 太阳能场
        # 对太阳能场，RandomForest表现更好
        print("Training RandomForestRegressor for solar farm...")
        rf = RandomForestRegressor(
            n_estimators=200,    # 增加树的数量以提高性能
            max_depth=10,        # 增加深度以捕获日变化模式
            min_samples_split=3,
            min_samples_leaf=2,
            max_features=0.8,    # 使用大部分特征
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(x_processed.values, y_processed.values.ravel())
        models.append(rf)
        weights.append(0.7)
        
        # 增加另一个模型以增强预测
        print("Training GradientBoostingRegressor for ensemble diversity...")
        gb = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.7,
            subsample=0.85,
            random_state=21
        )
        gb.fit(x_processed.values, y_processed.values.ravel())
        models.append(gb)
        weights.append(0.3)
    
    # 将模型和权重作为字典返回
    return {"models": models, "weights": weights, "feature_names": list(x_processed.columns)}

def train(farm_id):
    # Determine if wind (1-5) or solar (6-10) farm
    is_wind_farm = farm_id <= 5
    
    x_df = pd.DataFrame()
    nwp_train_path = f'training/middle_school/TRAIN/nwp_data_train/{farm_id}'
    
    for nwp in nwps:
        nwp_path = os.path.join(nwp_train_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        
        # For wind farms, focus on wind variables
        if is_wind_farm:
            # Process wind-related variables
            u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                             channel=['u100']).data.values.reshape(365 * 24, 9)
            v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                             channel=['v100']).data.values.reshape(365 * 24, 9)
            
            # Calculate basic u and v components
            u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
            v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
            
            # Add enhanced wind features
            wind_features = add_wind_features(u, v, nwp)
            
            # Add turbulence intensity proxy
            if 'tcc' in nwp_data.channel:
                tcc = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                  channel=['tcc']).data.values.reshape(365 * 24, 9)
                tcc_df = pd.DataFrame(tcc, columns=[f"{nwp}_tcc_{i}" for i in range(tcc.shape[1])])
                nwp_df = pd.concat([u_df, v_df, wind_features, tcc_df], axis=1)
            else:
                nwp_df = pd.concat([u_df, v_df, wind_features], axis=1)
        else:
            # For solar farms, focus on solar radiation and cloud cover
            if 'ghi' in nwp_data.channel:
                ghi = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                  channel=['ghi']).data.values.reshape(365 * 24, 9)
                ghi_df = pd.DataFrame(ghi, columns=[f"{nwp}_ghi_{i}" for i in range(ghi.shape[1])])
                
                poai = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                   channel=['poai']).data.values.reshape(365 * 24, 9)
                poai_df = pd.DataFrame(poai, columns=[f"{nwp}_poai_{i}" for i in range(poai.shape[1])])
                
                tcc = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                  channel=['tcc']).data.values.reshape(365 * 24, 9)
                tcc_df = pd.DataFrame(tcc, columns=[f"{nwp}_tcc_{i}" for i in range(tcc.shape[1])])
                
                # Calculate average GHI and maximum GHI as additional features
                avg_ghi = np.mean(ghi, axis=1).reshape(-1, 1)
                max_ghi = np.max(ghi, axis=1).reshape(-1, 1)
                avg_ghi_df = pd.DataFrame(avg_ghi, columns=[f"{nwp}_avg_ghi"])
                max_ghi_df = pd.DataFrame(max_ghi, columns=[f"{nwp}_max_ghi"])
                
                # Calculate ratio of actual GHI to clear sky GHI (proxy using max value)
                # This helps identify cloud effects
                ghi_ratio = ghi / (max_ghi + 1e-6)
                ghi_ratio_df = pd.DataFrame(ghi_ratio, columns=[f"{nwp}_ghi_ratio_{i}" for i in range(ghi_ratio.shape[1])])
                
                # Calculate standard deviation of GHI (indicates variability)
                std_ghi = np.std(ghi, axis=1).reshape(-1, 1)
                std_ghi_df = pd.DataFrame(std_ghi, columns=[f"{nwp}_std_ghi"])
                
                # Calculate cloud coverage impact (using TCC)
                # Higher TCC means more clouds, which usually means less solar power
                cloud_impact = (1 - np.mean(tcc, axis=1)).reshape(-1, 1) * avg_ghi
                cloud_impact_df = pd.DataFrame(cloud_impact, columns=[f"{nwp}_cloud_impact"])
                
                nwp_df = pd.concat([ghi_df, poai_df, tcc_df, avg_ghi_df, max_ghi_df, 
                                   ghi_ratio_df, std_ghi_df, cloud_impact_df], axis=1)
            else:
                # Fallback to basic features
                u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['u100']).data.values.reshape(365 * 24, 9)
                v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['v100']).data.values.reshape(365 * 24, 9)
                u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
                v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
                ws = np.sqrt(u ** 2 + v ** 2)
                ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
                nwp_df = pd.concat([u_df, v_df, ws_df], axis=1)
        
        x_df = pd.concat([x_df, nwp_df], axis=1)
    
    x_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')
    
    # Add time features
    x_df = add_time_features(x_df)
    
    # Add simple lag features, same as in training
    lag_features = {}
    diff_features = {}
    for col in x_df.columns:
        if col not in ['hour', 'day', 'month', 'is_daytime', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']:
            lag_features[f'{col}_lag1'] = x_df[col].shift(1)
            diff_features[f'{col}_diff'] = x_df[col].diff()
    
    # Combine all new features at once using pd.concat
    lag_df = pd.DataFrame(lag_features, index=x_df.index)
    diff_df = pd.DataFrame(diff_features, index=x_df.index)
    x_df = pd.concat([x_df, lag_df, diff_df], axis=1)
    
    # Fill NaN values from lag operations
    x_df = x_df.ffill().bfill()
    
    y_df = pd.read_csv(os.path.join(fact_path, f'{farm_id}_normalization_train.csv'), index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']
    
    x_processed, y_processed = data_preprocess(x_df, y_df)
    y_processed[y_processed < 0] = 0
    
    # 特征选择基于相关性和特征重要性
    correlations = []
    for col in x_processed.columns:
        corr = np.abs(np.corrcoef(x_processed[col], y_processed['power'])[0, 1])
        if not np.isnan(corr):  # 避免NaN相关性
            correlations.append((col, corr))
    
    # 按相关性排序
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 为保持一致性，使用相同的特征选择方法
    if is_wind_farm:
        # 对风电场，保留风速、风向相关的特征
        wind_features = [col for col, _ in correlations if any(term in col for term in ['_ws_', '_wd_', '_ws3_', 'tcc', '_u_', '_v_'])]
        # 保留高相关性特征（相关性>0.2，这个阈值比0.3更保守，能保留更多特征）
        top_corr_features = [col for col, corr in correlations if corr > 0.2]
        # 合并特征列表并去重
        all_selected = list(set(wind_features + top_corr_features))
        # 按重要性排序，确保不会超过最大特征数
        max_features = 450  # 限制特征数量，避免过拟合
        if len(all_selected) > max_features:
            # 按相关性排序并截取
            sorted_features = [col for col, _ in sorted([(col, corr) for col, corr in correlations if col in all_selected], 
                                                      key=lambda x: x[1], reverse=True)]
            top_features = sorted_features[:max_features]
        else:
            top_features = all_selected
    else:
        # 对太阳能场，保留辐照度和云量相关的特征
        solar_features = [col for col, _ in correlations if any(term in col for term in ['ghi', 'poai', 'tcc', 'cloud_impact', 't2m'])]
        # 保留高相关性特征
        top_corr_features = [col for col, corr in correlations if corr > 0.2]
        # 合并特征列表并去重
        all_selected = list(set(solar_features + top_corr_features))
        # 按重要性排序，确保不会超过最大特征数
        max_features = 450  # 限制特征数量，避免过拟合
        if len(all_selected) > max_features:
            # 按相关性排序并截取
            sorted_features = [col for col, _ in sorted([(col, corr) for col, corr in correlations if col in all_selected], 
                                                      key=lambda x: x[1], reverse=True)]
            top_features = sorted_features[:max_features]
        else:
            top_features = all_selected
    
    # 始终保留重要的时间特征
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_daytime']
    for feature in time_features:
        if feature in x_processed.columns and feature not in top_features:
            top_features.append(feature)
    
    # 仅使用选定的特征
    print(f"Selected {len(top_features)} features out of {x_processed.shape[1]} available features")
    x_processed = x_processed[top_features]
    
    # 训练优化的集成模型
    model = create_ensemble_model(x_processed, y_processed, is_wind_farm)
    
    return model, top_features

def predict(model, farm_id, top_features=None):
    # Determine if wind or solar farm
    is_wind_farm = farm_id <= 5
    
    x_df = pd.DataFrame()
    nwp_test_path = f'training/middle_school/TEST/nwp_data_test/{farm_id}'
    
    for nwp in nwps:
        nwp_path = os.path.join(nwp_test_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        
        if is_wind_farm:
            # Process wind-related variables
            u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                             channel=['u100']).data.values.reshape(31 * 24, 9)
            v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                             channel=['v100']).data.values.reshape(31 * 24, 9)
            
            u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
            v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
            
            # Add enhanced wind features
            wind_features = add_wind_features(u, v, nwp)
            
            if 'tcc' in nwp_data.channel:
                tcc = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                  channel=['tcc']).data.values.reshape(31 * 24, 9)
                tcc_df = pd.DataFrame(tcc, columns=[f"{nwp}_tcc_{i}" for i in range(tcc.shape[1])])
                nwp_df = pd.concat([u_df, v_df, wind_features, tcc_df], axis=1)
            else:
                nwp_df = pd.concat([u_df, v_df, wind_features], axis=1)
        else:
            if 'ghi' in nwp_data.channel:
                ghi = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                  channel=['ghi']).data.values.reshape(31 * 24, 9)
                ghi_df = pd.DataFrame(ghi, columns=[f"{nwp}_ghi_{i}" for i in range(ghi.shape[1])])
                
                poai = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                   channel=['poai']).data.values.reshape(31 * 24, 9)
                poai_df = pd.DataFrame(poai, columns=[f"{nwp}_poai_{i}" for i in range(poai.shape[1])])
                
                tcc = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                  channel=['tcc']).data.values.reshape(31 * 24, 9)
                tcc_df = pd.DataFrame(tcc, columns=[f"{nwp}_tcc_{i}" for i in range(tcc.shape[1])])
                
                # Calculate average GHI and maximum GHI as additional features
                avg_ghi = np.mean(ghi, axis=1).reshape(-1, 1)
                max_ghi = np.max(ghi, axis=1).reshape(-1, 1)
                avg_ghi_df = pd.DataFrame(avg_ghi, columns=[f"{nwp}_avg_ghi"])
                max_ghi_df = pd.DataFrame(max_ghi, columns=[f"{nwp}_max_ghi"])
                
                # Calculate ratio of actual GHI to clear sky GHI (proxy using max value)
                ghi_ratio = ghi / (max_ghi + 1e-6)
                ghi_ratio_df = pd.DataFrame(ghi_ratio, columns=[f"{nwp}_ghi_ratio_{i}" for i in range(ghi_ratio.shape[1])])
                
                # Calculate standard deviation of GHI (indicates variability)
                std_ghi = np.std(ghi, axis=1).reshape(-1, 1)
                std_ghi_df = pd.DataFrame(std_ghi, columns=[f"{nwp}_std_ghi"])
                
                # Calculate cloud coverage impact (using TCC)
                # Higher TCC means more clouds, which usually means less solar power
                cloud_impact = (1 - np.mean(tcc, axis=1)).reshape(-1, 1) * avg_ghi
                cloud_impact_df = pd.DataFrame(cloud_impact, columns=[f"{nwp}_cloud_impact"])
                
                nwp_df = pd.concat([ghi_df, poai_df, tcc_df, avg_ghi_df, max_ghi_df, 
                                   ghi_ratio_df, std_ghi_df, cloud_impact_df], axis=1)
            else:
                u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['u100']).data.values.reshape(31 * 24, 9)
                v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['v100']).data.values.reshape(31 * 24, 9)
                u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
                v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
                ws = np.sqrt(u ** 2 + v ** 2)
                ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
                nwp_df = pd.concat([u_df, v_df, ws_df], axis=1)
        
        x_df = pd.concat([x_df, nwp_df], axis=1)
    
    x_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')
    
    # Add time features
    x_df = add_time_features(x_df)
    
    # Add simple lag features, same as in training
    lag_features = {}
    diff_features = {}
    for col in x_df.columns:
        if col not in ['hour', 'day', 'month', 'is_daytime', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']:
            lag_features[f'{col}_lag1'] = x_df[col].shift(1)
            diff_features[f'{col}_diff'] = x_df[col].diff()
    
    # Combine all new features at once using pd.concat
    lag_df = pd.DataFrame(lag_features, index=x_df.index)
    diff_df = pd.DataFrame(diff_features, index=x_df.index)
    x_df = pd.concat([x_df, lag_df, diff_df], axis=1)
    
    # Fill NaN values from lag operations
    x_df = x_df.ffill().bfill()
    
    # Use only selected features if provided
    if top_features:
        # Ensure all required features exist, create with zeros if missing
        missing_features = [feature for feature in top_features if feature not in x_df.columns]
        
        # Create missing features with zeros
        for feature in missing_features:
            x_df[feature] = 0
            print(f"Warning: Created missing feature {feature} with zeros")
            
        # Keep only the features used during training
        x_df = x_df[top_features]
    
    # 验证特征数量
    if isinstance(model, dict) and "models" in model and "weights" in model:
        # 对于集成模型，使用feature_names保证一致性
        if "feature_names" in model:
            expected_features = model["feature_names"]
            
            # 确保所有需要的特征都存在
            missing_features = [f for f in expected_features if f not in x_df.columns]
            for feature in missing_features:
                x_df[feature] = 0
                print(f"Warning: Created missing feature {feature} with zeros")
            
            # 确保特征顺序一致
            x_df = x_df[expected_features]
        else:
            # 兼容之前的模型结构
            expected_features = model["models"][0].n_features_in_
            if x_df.shape[1] != expected_features:
                print(f"Feature mismatch: model expecting {expected_features} features, but got {x_df.shape[1]}")
    
    # 进行预测
    if isinstance(model, dict) and "models" in model and "weights" in model:
        # 集成预测
        predictions = np.zeros(x_df.shape[0])
        for m, w in zip(model["models"], model["weights"]):
            predictions += w * m.predict(x_df.values)
        # 归一化权重
        total_weight = sum(model["weights"])
        predictions /= total_weight
        pred_pw = predictions
    elif isinstance(model, (GradientBoostingRegressor, RandomForestRegressor, RidgeCV)):
        pred_pw = model.predict(x_df.values)
    else:
        # PyTorch模型
        model.eval()
        with torch.no_grad():
            X = torch.tensor(x_df.values, dtype=torch.float32)
            pred_pw = model(X).flatten().numpy()
    
    # 后处理以获得更平滑的预测
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0], periods=len(pred_pw), freq='h'))
    
    # 应用自适应平滑
    if is_wind_farm:
        # 计算滚动标准差以检测高变异性周期
        rolling_std = pred.rolling(window=3, min_periods=1).std()
        rolling_mean = pred.rolling(window=5, min_periods=1).mean()
        
        # 创建平滑预测的副本
        smoothed_pred = pred.copy()
        
        # 基于标准差应用自适应平滑
        for i in range(1, len(pred)-1):
            if rolling_std.iloc[i] > 0.08:  # 高变异性阈值
                # 对高变异性期间应用更强的平滑
                smoothed_pred.iloc[i] = 0.25 * pred.iloc[i-1] + 0.5 * pred.iloc[i] + 0.25 * pred.iloc[i+1]
            else:
                # 对稳定期间应用较轻的平滑
                smoothed_pred.iloc[i] = 0.1 * pred.iloc[i-1] + 0.8 * pred.iloc[i] + 0.1 * pred.iloc[i+1]
        
        # 使用平滑后的值
        pred = smoothed_pred
                
        # 应用趋势修正 - 确保预测遵循整体趋势
        pred_trend = rolling_mean - pred
        trend_threshold = pred_trend.quantile(0.8)
        
        # 如果预测严重偏离趋势，进行适度修正
        for i in range(2, len(pred)-2):
            if abs(pred_trend.iloc[i]) > trend_threshold:
                # 轻微向趋势修正，但不完全替换
                correction = 0.3 * rolling_mean.iloc[i]
                pred.iloc[i] = 0.7 * pred.iloc[i] + correction
    else:
        # 对太阳能场的平滑处理
        hours = pred.index.hour
        
        # 创建平滑预测的副本
        smoothed_pred = pred.copy()
        
        # 基于一天中的时间应用不同的平滑
        for i in range(1, len(pred)-1):
            hour = hours[i]
            
            if 6 <= hour <= 8 or 16 <= hour <= 18:  # 黎明/黄昏过渡期
                # 过渡期应用更强的平滑
                smoothed_pred.iloc[i] = 0.3 * pred.iloc[i-1] + 0.4 * pred.iloc[i] + 0.3 * pred.iloc[i+1]
            elif 9 <= hour <= 15:  # 白天 (高产期)
                # 高产期应用较轻的平滑
                smoothed_pred.iloc[i] = 0.1 * pred.iloc[i-1] + 0.8 * pred.iloc[i] + 0.1 * pred.iloc[i+1]
            else:  # 夜晚
                # 夜晚应用中等平滑
                smoothed_pred.iloc[i] = 0.2 * pred.iloc[i-1] + 0.6 * pred.iloc[i] + 0.2 * pred.iloc[i+1]
        
        # 使用平滑后的值
        pred = smoothed_pred
        
        # 白天太阳能峰值校正
        peak_hours = (hours >= 10) & (hours <= 14)
        if sum(peak_hours) > 0:
            # 找出白天小时的最大值
            daytime_max = pred[peak_hours].max()
            
            # 计算修正因子，确保平滑不会显著降低峰值
            for i in range(len(pred)):
                if 10 <= hours[i] <= 14 and pred.iloc[i] > 0.7 * daytime_max:
                    # 轻微提高峰值
                    pred.iloc[i] = min(pred.iloc[i] * 1.05, 1.0)  # 适度提高，但不超过1
    
    # 应用15分钟间隔的插值
    # 使用cubic插值获得更平滑的结果
    res = pred.resample('15min').interpolate(method='cubic')
    
    # 应用约束
    res[res < 0] = 0
    res[res > 1] = 1
    
    # 对于太阳能场站，确保夜间产量为零
    if not is_wind_farm:
        hours = res.index.hour
        month = res.index.month
        
        # 为不同季节创建掩码
        winter_mask = month.isin([11, 12, 1, 2])
        summer_mask = month.isin([5, 6, 7, 8])
        
        # 基于季节应用夜间小时
        night_mask_winter = (hours >= 17) | (hours <= 7)  # 冬季夜间时间长
        night_mask_summer = (hours >= 20) | (hours <= 5)  # 夏季夜间时间短
        night_mask_default = (hours >= 19) | (hours <= 6)  # 默认（春秋）
        
        # 应用掩码 - 当季节和夜间条件同时为真时
        res[winter_mask & night_mask_winter] = 0
        res[summer_mask & night_mask_summer] = 0
        res[~winter_mask & ~summer_mask & night_mask_default] = 0
        
        # 在黎明/黄昏时平滑过渡（逐步调整产量）
        for hour in [5, 6, 7, 19, 20]:
            dawn_dusk_mask = res.index.hour == hour
            if sum(dawn_dusk_mask) > 0:
                minute = res.index[dawn_dusk_mask].minute
                
                # 不同时段应用不同的平滑
                if hour in [5, 6, 7]:  # 黎明 - 逐渐增加
                    if hour == 5:
                        factor = minute / 120
                    elif hour == 6:
                        factor = (minute + 60) / 120
                    else:  # 7点
                        factor = (minute + 120) / 180
                    
                    res.loc[dawn_dusk_mask] = res.loc[dawn_dusk_mask] * factor
                else:  # 黄昏 - 逐渐减少
                    if hour == 19:
                        factor = (60 - minute) / 60
                    else:  # 20点
                        factor = (60 - minute) / 120
                        factor = np.maximum(factor, 0)  # 确保非负
                    
                    res.loc[dawn_dusk_mask] = res.loc[dawn_dusk_mask] * factor
    
    return res

def process_farm(farm_id):
    """处理单个农场 - 用于并行处理"""
    farm_start_time = time.time()
    model_path = f'models/{farm_id}'
    os.makedirs(model_path, exist_ok=True)
    model_name = 'enhanced_model.pkl'
    features_name = 'features.pkl'
    
    try:
        print(f"\n===== 处理农场 {farm_id} =====")
        print(f"训练农场 {farm_id} 的模型...")
        model, top_features = train(farm_id)
        
        # 保存模型和选定的特征
        print(f"保存农场 {farm_id} 的模型...")
        with open(os.path.join(model_path, model_name), "wb") as f:
            pickle.dump(model, f)
        
        with open(os.path.join(model_path, features_name), "wb") as f:
            pickle.dump(top_features, f)
        
        print(f"为农场 {farm_id} 生成预测...")
        pred = predict(model, farm_id, top_features)
        result_path = f'result/output'
        os.makedirs(result_path, exist_ok=True)
        pred.to_csv(os.path.join(result_path, f'output{farm_id}.csv'))
        farm_time = time.time() - farm_start_time
        print(f'成功处理农场 {farm_id}，耗时 {farm_time:.2f} 秒')
        return farm_id, True
    except Exception as e:
        print(f"处理农场 {farm_id} 时出错: {str(e)}")
        return farm_id, False

# 主执行函数
if __name__ == "__main__":
    acc = pd.DataFrame()
    # 处理所有农场
    farms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # 开始计时
    total_start_time = time.time()
    
    # 选项1: 顺序处理
    # for farm_id in farms:
    #     process_farm(farm_id)
    
    # 选项2: 使用多进程进行并行处理
    # 确定要使用的CPU核心数（最多4个以避免内存问题）
    num_cores = min(4, mp.cpu_count())
    print(f"使用 {num_cores} 个核心进行并行处理")
    
    # 创建工作进程池
    with mp.Pool(processes=num_cores) as pool:
        # 并行映射process_farm到所有farm_ids
        results = pool.map(process_farm, farms)
    
    # 计算总运行时间
    total_time = time.time() - total_start_time
    
    # 统计成功处理的农场数
    successful = sum(1 for _, success in results if success)
    print(f"{successful} 个农场中的 {len(farms)} 个已成功处理")
    print(f'所有农场处理完成，总耗时 {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)')
    
    # 处理完成时播放声音提醒
    try:
        # 对于Linux/Unix系统
        os.system('for i in {1..5}; do echo -e "\a"; sleep 0.5; done')
        # 对于Windows系统
        os.system('powershell -c "(New-Object Media.SoundPlayer).PlaySync([System.IO.Path]::Combine([System.Environment]::SystemDirectory, \'media\\Windows Notify.wav\'))"')
        # Windows的替代方法
        os.system('echo \x07\x07\x07\x07\x07')
    except:
        # 如果上述方法失败的回退方案
        import sys
        for _ in range(5):
            sys.stdout.write('\a')
            sys.stdout.flush()
            time.sleep(0.5)
    
    print("\n处理完成！所有农场数据处理完毕！", flush=True)
