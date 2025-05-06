import numpy as np
import pandas as pd
import torch
import os
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class EarlyStopping:
    """早停机制，监控验证损失，当指定轮次内未改善时停止训练"""
    
    def __init__(self, patience=15, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
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
        """保存表现最好的模型"""
        if self.verbose:
            print(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f})。保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class PowerGenerationDataset(torch.utils.data.Dataset):
    """用于风电/光伏发电预测的数据集类"""
    
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def enhance_features(df):
    """增强特征工程，创建更多有意义的特征"""
    # 时间特征
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df.index.dayofweek >= 5
    
    # 周期性特征（时间的正弦和余弦变换）
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # 对数值型特征，计算统计特征
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # 添加滚动统计特征（过去24小时）
        if len(df) > 24:  # 确保有足够的数据
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24).mean()
            df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24).std()
            df[f'{col}_rolling_min_24h'] = df[col].rolling(window=24).min()
            df[f'{col}_rolling_max_24h'] = df[col].rolling(window=24).max()
    
    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def data_preprocess_enhanced(x_df, y_df=None, is_train=True, test_size=0.2):
    """增强的数据预处理函数"""
    # 增强特征工程
    x_df = enhance_features(x_df)
    
    # 移除低方差特征
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    
    # 处理缺失值
    x_df = x_df.fillna(method='ffill').fillna(method='bfill')
    
    if is_train and y_df is not None:
        # 处理异常值
        y_df = y_df.clip(lower=y_df.quantile(0.001), upper=y_df.quantile(0.999))
        
        # 时间序列分割（考虑数据的时间依赖性）
        split_idx = int(len(x_df) * (1 - test_size))
        X_train, X_val = x_df.iloc[:split_idx].values, x_df.iloc[split_idx:].values
        y_train, y_val = y_df.iloc[:split_idx].values, y_df.iloc[split_idx:].values
        
        # 特征缩放
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        
        return X_train, X_val, y_train, y_val, scaler_X, scaler_y
    else:
        # 预测模式
        X_test = x_df.values
        return X_test

def cross_validate(x_df, y_df, model_class, model_params, n_splits=5, verbose=True):
    """使用时间序列交叉验证评估模型性能"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(x_df)):
        if verbose:
            print(f"Fold {fold+1}/{n_splits}")
        
        # 分割数据
        X_train, X_val = x_df.iloc[train_idx].values, x_df.iloc[val_idx].values
        y_train, y_val = y_df.iloc[train_idx].values, y_df.iloc[val_idx].values
        
        # 特征缩放
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        
        # 创建数据加载器
        train_dataset = PowerGenerationDataset(X_train, y_train)
        val_dataset = PowerGenerationDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
        
        # 初始化模型
        model = model_class(**model_params).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # 训练模型
        for epoch in range(50):  # 简短训练，仅用于交叉验证
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # 评估模型
        model.eval()
        val_pred = []
        val_true = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_pred.extend(outputs.cpu().numpy())
                val_true.extend(targets.numpy())
        
        # 计算指标
        val_pred = scaler_y.inverse_transform(np.array(val_pred))
        val_true = scaler_y.inverse_transform(np.array(val_true))
        
        mae = mean_absolute_error(val_true, val_pred)
        fold_scores.append(mae)
        
        if verbose:
            print(f"Fold {fold+1} MAE: {mae:.4f}")
    
    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    if verbose:
        print(f"Cross-validation MAE: {avg_score:.4f} ± {std_score:.4f}")
    
    return avg_score, std_score
