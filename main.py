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

# Define PyTorch linear model to exactly match sklearn's LinearRegression
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
    
    def forward(self, x):
        return self.linear(x)
    
    # Add methods to mimic sklearn's LinearRegression interface
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self(X).numpy()

nwps = ['NWP_1','NWP_2','NWP_3']
fact_path = 'training/middle_school/TRAIN/fact_data'

def data_preprocess(x_df, y_df):
    x_df = x_df.dropna()
    y_df = y_df.dropna()
    # 数据对扣
    ind = [i for i in y_df.index if i in x_df.index]
    x_df = x_df.loc[ind]
    y_df = y_df.loc[ind]
    return x_df,y_df

def train(farm_id):
    x_df = pd.DataFrame()
    nwp_train_path = f'training/middle_school/TRAIN/nwp_data_train/{farm_id}'
    for nwp in nwps:
        nwp_path = os.path.join(nwp_train_path,nwp,)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        u = nwp_data.sel(lat=range(4,7),lon=range(4,7),lead_time=range(24),
                        channel=['u100']).data.values.reshape(365 * 24, 9)
        v = nwp_data.sel(lat=range(4,7), lon=range(4,7),lead_time=range(24),
                    channel=['v100']).data.values.reshape(365 * 24, 9)
        u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
        v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
        nwp_df = pd.concat([u_df,v_df,ws_df],axis=1)
        x_df = pd.concat([x_df,nwp_df],axis=1)
    x_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')
    y_df = pd.read_csv(os.path.join(fact_path,f'{farm_id}_normalization_train.csv'),index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']
    x_processed,y_processed = data_preprocess(x_df,y_df)
    y_processed[y_processed < 0] = 0
    
    # Convert to torch tensors
    X = torch.tensor(x_processed.values, dtype=torch.float32)
    y = torch.tensor(y_processed.values, dtype=torch.float32)
    
    # Solve analytically using normal equation to match sklearn's LinearRegression
    # X^T * X * w = X^T * y
    # w = (X^T * X)^(-1) * X^T * y
    X_t = X.T
    X_t_X = X_t @ X
    X_t_y = X_t @ y
    
    # Add small regularization for numerical stability
    reg = 1e-10 * torch.eye(X_t_X.shape[0])
    weights = torch.linalg.solve(X_t_X + reg, X_t_y)
    
    # Create model and set weights directly
    input_dim = X.shape[1]
    model = LinearModel(input_dim)
    with torch.no_grad():
        model.linear.weight.copy_(weights.T)
        # Calculate bias (intercept)
        model.linear.bias.zero_()
    
    return model

def predict(model, farm_id):
    x_df = pd.DataFrame()
    nwp_test_path = f'training/middle_school/TEST/nwp_data_test/{farm_id}'
    for nwp in nwps:
        nwp_path = os.path.join(nwp_test_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
        u = nwp_data.sel(lat=range(4,7),lon=range(4,7), lead_time=range(24),
                         channel=['u100']).data.values.reshape(31 * 24, 9)
        v = nwp_data.sel(lat=range(4,7), lon=range(4,7),lead_time=range(24),
                     channel=['v100']).data.values.reshape(31 * 24, 9)
        u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
        v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
        nwp_df = pd.concat([u_df,v_df,ws_df],axis=1)
        x_df = pd.concat([x_df,nwp_df],axis=1)
    x_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        X = torch.tensor(x_df.values, dtype=torch.float32)
        pred_pw = model(X).flatten().numpy()
    
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0],periods=len(pred_pw), freq='h'))
    res = pred.resample('15min').interpolate(method='linear')
    res[res<0] = 0
    res[res>1] = 1
    return res

acc = pd.DataFrame()
farms = [1,2,3,4,5,6,7,8,9,10]
for farm_id in farms:
    model_path = f'models/{farm_id}' 
    os.makedirs(model_path,exist_ok=True)
    model_name = 'baseline_middle_school.pkl'  # Use the same filename as original
    model = train(farm_id)
    
    # Save the model
    with open(os.path.join(model_path, model_name), "wb") as f:
        pickle.dump(model, f)
    
    pred = predict(model, farm_id)
    result_path = f'result/output'
    os.makedirs(result_path,exist_ok=True)
    pred.to_csv(os.path.join(result_path,f'output{farm_id}.csv'))
print('ok')
