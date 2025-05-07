import os
import pickle
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import lr_scheduler

# 设置随机种子以确保结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义风电场和光伏场站ID
WIND_FARMS = [1, 2, 3, 4, 5]
SOLAR_FARMS = [6, 7, 8, 9, 10]

warnings.filterwarnings('ignore')
nwps = ['NWP_1', 'NWP_2', 'NWP_3']
fact_path = 'training/middle_school/TRAIN/fact_data'

# 数据标准化类 - 替代sklearn的StandardScaler/RobustScaler
class TorchScaler:
    def __init__(self, method='standard'):
        self.method = method
        self.mean = None
        self.std = None
        self.median = None
        self.q1 = None
        self.q3 = None
    
    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = torch.tensor(data.values).float()
            
        if self.method == 'standard':
            self.mean = torch.mean(data, dim=0)
            self.std = torch.std(data, dim=0)
        elif self.method == 'robust':
            self.median = torch.median(data, dim=0).values
            self.q1 = torch.quantile(data, 0.25, dim=0)
            self.q3 = torch.quantile(data, 0.75, dim=0)
        
        return self
    
    def transform(self, data):
        columns = None  # 初始化这些变量，避免未定义错误
        index = None
        
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            if isinstance(data, pd.DataFrame):
                columns = data.columns
                index = data.index
            else:
                columns = None
                index = data.index
            data = torch.tensor(data.values).float()
        
        if self.method == 'standard':
            normalized = (data - self.mean) / (self.std + 1e-8)
        elif self.method == 'robust':
            iqr = self.q3 - self.q1
            normalized = (data - self.median) / (iqr + 1e-8)
            
        if isinstance(normalized, torch.Tensor):
            normalized = normalized.numpy()
            
        if columns is not None:
            return pd.DataFrame(normalized, columns=columns, index=index)
        elif index is not None:
            return pd.Series(normalized.flatten(), index=index)
        return normalized
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
        
    def inverse_transform(self, data):
        columns = None  # 初始化这些变量，避免未定义错误
        index = None
        
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            if isinstance(data, pd.DataFrame):
                columns = data.columns
                index = data.index
            else:
                columns = None
                index = data.index
            data = torch.tensor(data.values).float()
            
        if self.method == 'standard':
            original = data * self.std + self.mean
        elif self.method == 'robust':
            iqr = self.q3 - self.q1
            original = data * (iqr + 1e-8) + self.median
            
        if isinstance(original, torch.Tensor):
            original = original.numpy()
            
        if columns is not None:
            return pd.DataFrame(original, columns=columns, index=index)
        elif index is not None:
            return pd.Series(original.flatten(), index=index)
        return original

# 自定义Dataset类
class PowerGenerationDataset(Dataset):
    def __init__(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.X = torch.tensor(X.values, dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
            
        self.y = None
        if y is not None:
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                self.y = torch.tensor(y.values, dtype=torch.float32)
            else:
                self.y = torch.tensor(y, dtype=torch.float32)
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# 早停类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.counter = 0
        
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
            print(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 传统机器学习模型的PyTorch实现
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super(MLPRegressor, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(layer_sizes[-1], 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).view(-1)

# 替代RandomForestRegressor和GradientBoostingRegressor的集成模型
class EnsembleModel(nn.Module):
    def __init__(self, input_size, n_models=10, hidden_sizes=[32, 16]):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([
            MLPRegressor(input_size, hidden_sizes) for _ in range(n_models)
        ])
        
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# 改进的混合模型
class ImprovedHybridModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        super(ImprovedHybridModel, self).__init__()
        
        # 线性分支
        self.linear = nn.Linear(input_size, 1)
        
        # MLP分支
        self.mlp_layers = []
        layer_sizes = [input_size] + hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            self.mlp_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            
        self.mlp_layers.append(nn.Linear(layer_sizes[-1], 1))
        self.mlp = nn.Sequential(*self.mlp_layers)
        
        # 特征权重学习
        self.feature_weights = nn.Parameter(torch.ones(input_size))
        
    def forward(self, x):
        # 加权特征
        weighted_x = x * F.softmax(self.feature_weights, dim=0)
        
        # 线性预测
        linear_out = self.linear(weighted_x).view(-1)
        
        # MLP预测
        mlp_out = self.mlp(weighted_x).view(-1)
        
        # 组合输出 (可学习的权重)
        return 0.2 * linear_out + 0.8 * mlp_out

# 添加时间特征
def add_time_features(df):
    """添加时间特征"""
    # 已有实现，保持不变
    hour = df.index.hour
    df['hour'] = hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    month = df.index.month
    df['month'] = month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    day = df.index.day
    df['day'] = day
    df['day_sin'] = np.sin(2 * np.pi * day / 31)
    df['day_cos'] = np.cos(2 * np.pi * day / 31)
    
    return df

# 添加滞后特征
def add_lag_features(df, lag_hours=[1, 2, 3, 6, 12, 24]):
    """添加滞后特征"""
    # 已有实现，保持不变
    temp_df = df.copy()
    feature_columns = temp_df.columns
    
    for feature in feature_columns:
        for lag in lag_hours:
            temp_df[f"{feature}_lag{lag}"] = temp_df[feature].shift(lag)
            
    # 填充NaN值
    temp_df = temp_df.fillna(method='bfill').fillna(method='ffill')
    
    return temp_df

# 数据预处理函数
def data_preprocess(x_df, y_df=None, is_train=True):
    """PyTorch风格的数据预处理"""
    if is_train and y_df is not None:
        # 处理离群值
        if isinstance(y_df, pd.DataFrame):
            q1 = y_df.quantile(0.25)[0]
            q3 = y_df.quantile(0.75)[0]
        else:
            q1 = y_df.quantile(0.25)
            q3 = y_df.quantile(0.75)
            
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = ((y_df < lower_bound) | (y_df > upper_bound)).sum()
        if hasattr(outliers, '__iter__'):
            outliers = outliers[0]
            
        if outliers > 0:
            print(f"发现 {outliers} 个离群值，进行处理...")
            
        # 范围约束，避免极端值
        if isinstance(y_df, pd.DataFrame):
            y_df = y_df.copy()
            y_df[y_df < 0] = 0
            y_df[y_df > 1] = 1
        else:
            y_df = y_df.copy()
            y_df[y_df < 0] = 0
            y_df[y_df > 1] = 1
        
        # 使用自定义的标准化器
        scaler_X = TorchScaler(method='robust')
        scaler_y = TorchScaler(method='standard')
        
        X_scaled = scaler_X.fit_transform(x_df)
        
        if isinstance(y_df, pd.DataFrame):
            y_scaled = scaler_y.fit_transform(y_df.values).flatten()
        else:
            y_scaled = scaler_y.fit_transform(y_df.values.reshape(-1, 1)).flatten()
        
        # 分割训练集和验证集 - 时间序列分割
        train_size = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:train_size]
        X_val = X_scaled[train_size:]
        y_train = y_scaled[:train_size]
        y_val = y_scaled[train_size:]
        
        return X_train, X_val, y_train, y_val, scaler_X, scaler_y
    else:
        # 测试数据处理
        return x_df

# 训练函数
def train(farm_id):
    """基于PyTorch的传统机器学习训练函数"""
    print(f"开始训练发电站 {farm_id} 的 PyTorch 模型...")
    
    # 创建模型保存路径
    model_path = f'models/{farm_id}'
    os.makedirs(model_path, exist_ok=True)
    checkpoint_path = os.path.join(model_path, f'checkpoint_{farm_id}.pt')
    
    # 读取和准备数据
    x_df = pd.DataFrame()
    nwp_train_path = f'training/middle_school/TRAIN/nwp_data_train/{farm_id}'
    
    # 处理NWP数据
    for nwp in nwps:
        nwp_path = os.path.join(nwp_train_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        
        # 提取特征
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
    
    # 特征选择 - 只保留相关性最强的特征
    corr_matrix = pd.concat([x_df, y_df], axis=1).corr()
    relevant_features = corr_matrix['power'].abs().sort_values(ascending=False)
    print(f"Top 20 most relevant features: {relevant_features.head(20).index.tolist()}")
    
    # 只保留前100个最相关的特征
    top_features = relevant_features.iloc[1:101].index.tolist()  # 跳过第一个(power自身)
    x_df_selected = x_df[top_features]
    
    # 数据预处理
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = data_preprocess(x_df_selected, y_df)
    
    # 创建PyTorch数据集和数据加载器
    train_dataset = PowerGenerationDataset(X_train, y_train)
    val_dataset = PowerGenerationDataset(X_val, y_val)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    input_size = X_train.shape[1]
    model = ImprovedHybridModel(input_size=input_size).to(device)
    
    # 结合MSE和MAE的损失函数
    def combined_loss(pred, target, alpha=0.7):
        mae_loss = F.l1_loss(pred, target)
        mse_loss = F.mse_loss(pred, target)
        return alpha * mse_loss + (1 - alpha) * mae_loss
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 添加学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 早停机制
    early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.0001, path=checkpoint_path)
    
    # 训练循环
    num_epochs = 200
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
            loss = combined_loss(outputs, targets)
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
                loss = combined_loss(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                predictions.extend(outputs.cpu().numpy())
                actual_values.extend(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印进度
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 检查是否早停
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(checkpoint_path))
    
    # 计算并打印验证集的评估指标
    model.eval()
    with torch.no_grad():
        val_dataset = PowerGenerationDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        inputs, targets = next(iter(val_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs).cpu().numpy()
        
        # 反标准化结果
        if hasattr(scaler_y, 'inverse_transform'):
            predictions = scaler_y.inverse_transform(predictions)
            actuals = scaler_y.inverse_transform(targets.cpu().numpy())
        else:
            predictions = predictions * scaler_y.std + scaler_y.mean
            actuals = targets.cpu().numpy() * scaler_y.std + scaler_y.mean
        
        mse = F.mse_loss(torch.tensor(predictions), torch.tensor(actuals)).item()
        mae = F.l1_loss(torch.tensor(predictions), torch.tensor(actuals)).item()
        print(f'验证集 RMSE: {np.sqrt(mse):.6f}, MAE: {mae:.6f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for Farm {farm_id}')
    plt.legend()
    plt.savefig(os.path.join(model_path, f'loss_curve_{farm_id}.png'))
    
    # 保存模型和标准化器
    model_package = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': top_features,
    }
    
    with open(os.path.join(model_path, f'torch_model_{farm_id}.pkl'), 'wb') as f:
        pickle.dump(model_package, f)
    
    return model_package

# 预测函数
def predict(model_package, farm_id):
    """基于PyTorch的预测函数"""
    # 解包模型组件
    model = model_package['model'].to(device)
    scaler_X = model_package['scaler_X']
    scaler_y = model_package['scaler_y']
    feature_list = model_package['features']
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
    
    # 只选择训练中使用的特征
    common_features = [col for col in feature_list if col in x_df.columns]
    x_test = x_df[common_features]
    
    # 标准化数据
    x_test_scaled = scaler_X.transform(x_test)
    
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
    
    # 反标准化预测
    if hasattr(scaler_y, 'inverse_transform'):
        pred_pw = scaler_y.inverse_transform(np.array(pred_pw).reshape(-1, 1)).flatten()
    else:
        pred_pw = np.array(pred_pw) * scaler_y.std + scaler_y.mean
    
    # 创建预测序列
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0], periods=len(pred_pw), freq='h'))
    
    # 将预测重采样为15分钟
    res = pred.resample('15min').interpolate(method='cubic')
    
    # 修正预测值范围
    res[res < 0] = 0
    res[res > 1] = 1
    
    return res

# 主函数
def main():
    """主程序执行函数，处理所有场站的训练与预测"""
    # 创建模型存储目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # 训练所有风电场模型
    for farm_id in WIND_FARMS:
        print(f"\n开始处理风电场 {farm_id}")
        model_package = train(farm_id)
        
        # 预测并保存结果
        predictions = predict(model_package, farm_id)
        predictions.to_csv(f'output/output{farm_id}.csv')
        print(f"风电场 {farm_id} 的预测结果已保存")
    
    # 训练所有光伏电站模型
    for farm_id in SOLAR_FARMS:
        print(f"\n开始处理光伏电站 {farm_id}")
        model_package = train(farm_id)
        
        # 预测并保存结果
        predictions = predict(model_package, farm_id)
        predictions.to_csv(f'output/output{farm_id}.csv')
        print(f"光伏电站 {farm_id} 的预测结果已保存")
    
    print("\n所有预测完成！")

if __name__ == "__main__":
    main()
