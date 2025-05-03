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

# 定义风电场和光伏场站ID
WIND_FARMS = [1, 2, 3, 4, 5]
SOLAR_FARMS = [6, 7, 8, 9, 10]

warnings.filterwarnings('ignore')

nwps = ['NWP_1','NWP_2','NWP_3']
fact_path = 'training/middle_school/TRAIN/fact_data'

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
    
    # 预处理测试数据
    x_test = data_preprocess(x_df, is_train=False)
    
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

def main():
    """主程序执行函数，处理所有场站的训练与预测"""
    print("===== 新能源功率预报系统启动 =====")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
            model_file = os.path.join(model_path, 'enhanced_model.pkl')
            
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
            result_file = os.path.join('result/output', f'output{farm_id}.csv')
            pred.to_csv(result_file)
            print(f"预测结果已保存至: {result_file}")
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