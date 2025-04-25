
import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from sklearn.linear_model import LinearRegression


nwps = ['NWP_1','NWP_2','NWP_3']
fact_path = '数据集/middle_school/TRAIN/fact_data'

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
    nwp_train_path = f'数据集/middle_school/TRAIN/nwp_data_train/{farm_id}'
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
    model = LinearRegression()
    model.fit(x_processed,y_processed)
    return model

def predict(model,farm_id):
    x_df = pd.DataFrame()
    nwp_test_path = f'数据集/middle_school/TEST/nwp_data_test/{farm_id}'
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
    pred_pw = model.predict(x_df).flatten()
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
    model_name = 'baseline_middle_school.pkl'
    model = train(farm_id)
    with open(
            os.path.join(model_path, model_name),
            "wb") as f:
        pickle.dump(model, f)
    pred = predict(model,farm_id)
    result_path = f'result/output'
    os.makedirs(result_path,exist_ok=True)
    pred.to_csv(os.path.join(result_path,f'output{farm_id}.csv'))
print('ok')
