import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from sklearn.linear_model import LinearRegression

nwps = ['NWP_1','NWP_2','NWP_3']  # 定义数值天气预报模型列表
fact_path = '数据集/middle_school/TRAIN/fact_data'  # 定义实际发电数据的路径

def data_preprocess(x_df, y_df):  # 定义数据预处理函数
    x_df = x_df.dropna()  # 删除特征数据中的NaN值
    y_df = y_df.dropna()  # 删除标签数据中的NaN值
    # 数据对扣
    ind = [i for i in y_df.index if i in x_df.index]  # 获取x_df和y_df共有的索引
    x_df = x_df.loc[ind]  # 筛选共有索引的特征数据
    y_df = y_df.loc[ind]  # 筛选共有索引的标签数据
    return x_df,y_df  # 返回预处理后的数据

def train(farm_id):  # 定义训练函数，接收风电场ID参数
    x_df = pd.DataFrame()  # 创建空的特征DataFrame
    nwp_train_path = f'数据集/middle_school/TRAIN/nwp_data_train/{farm_id}'  # 设置训练数据路径
    for nwp in nwps:  # 遍历每个数值天气预报模型
        nwp_path = os.path.join(nwp_train_path,nwp,)  # 获取当前NWP模型的数据路径
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")  # 读取NetCDF格式的NWP数据
        u = nwp_data.sel(lat=range(4,7),lon=range(4,7),lead_time=range(24),
                         channel=['u100']).data.values.reshape(365 * 24, 9)  # 提取100米高度的东西风速分量数据
        v = nwp_data.sel(lat=range(4,7), lon=range(4,7),lead_time=range(24),
                     channel=['v100']).data.values.reshape(365 * 24, 9)  # 提取100米高度的南北风速分量数据
        u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])  # 将u风速分量转为DataFrame
        v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])  # 将v风速分量转为DataFrame
        ws = np.sqrt(u ** 2 + v ** 2)  # 计算合成风速
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])  # 将合成风速转为DataFrame
        nwp_df = pd.concat([u_df,v_df,ws_df],axis=1)  # 横向合并u、v和ws特征
        x_df = pd.concat([x_df,nwp_df],axis=1)  # 将当前NWP特征加入总特征集
    x_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')  # 设置特征数据的时间索引
    y_df = pd.read_csv(os.path.join(fact_path,f'{farm_id}_normalization_train.csv'),index_col=0)  # 读取归一化后的发电量数据
    y_df.index = pd.to_datetime(y_df.index)  # 将索引转换为日期时间类型
    y_df.columns = ['power']  # 设置列名为power
    x_processed,y_processed = data_preprocess(x_df,y_df)  # 对特征和标签数据进行预处理
    y_processed[y_processed < 0] = 0  # 将负值发电量设为0（因为实际发电量不可能为负）
    model = LinearRegression()  # 创建线性回归模型
    model.fit(x_processed,y_processed)  # 训练模型
    return model  # 返回训练好的模型

def predict(model,farm_id):  # 定义预测函数
    x_df = pd.DataFrame()  # 创建空的特征DataFrame
    nwp_test_path = f'数据集/middle_school/TEST/nwp_data_test/{farm_id}'  # 设置测试数据路径
    for nwp in nwps:  # 遍历每个数值天气预报模型
        nwp_path = os.path.join(nwp_test_path, nwp)  # 获取当前NWP模型的数据路径
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")  # 读取NetCDF格式的NWP数据
        u = nwp_data.sel(lat=range(4,7),lon=range(4,7), lead_time=range(24),
                         channel=['u100']).data.values.reshape(31 * 24, 9)  # 提取100米高度的东西风速分量数据
        v = nwp_data.sel(lat=range(4,7), lon=range(4,7),lead_time=range(24),
                     channel=['v100']).data.values.reshape(31 * 24, 9)  # 提取100米高度的南北风速分量数据
        u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])  # 将u风速分量转为DataFrame
        v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])  # 将v风速分量转为DataFrame
        ws = np.sqrt(u ** 2 + v ** 2)  # 计算合成风速
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])  # 将合成风速转为DataFrame
        nwp_df = pd.concat([u_df,v_df,ws_df],axis=1)  # 横向合并u、v和ws特征
        x_df = pd.concat([x_df,nwp_df],axis=1)  # 将当前NWP特征加入总特征集
    x_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')  # 设置特征数据的时间索引
    pred_pw = model.predict(x_df).flatten()  # 使用模型预测发电量并展平结果
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0],periods=len(pred_pw), freq='h'))  # 将预测结果转为Series并设置时间索引
    res = pred.resample('15min').interpolate(method='linear')  # 将小时预测结果重采样为15分钟，使用线性插值
    res[res<0] = 0  # 将负值预测结果设为0
    res[res>1] = 1  # 将大于1的预测结果设为1（归一化后的功率范围为0-1）
    return res  # 返回预测结果

acc = pd.DataFrame()  # 创建空的统计DataFrame
farms = [1,2,3,4,5,6,7,8,9,10]  # 定义风电场ID列表
for farm_id in farms:  # 遍历每个风电场
    model_path = f'models/{farm_id}'  # 设置模型保存路径
    os.makedirs(model_path,exist_ok=True)  # 创建模型保存目录（如果不存在）
    model_name = 'baseline_middle_school.pkl'  # 设置模型文件名
    model = train(farm_id)  # 训练当前风电场的模型
    with open(
            os.path.join(model_path, model_name),
            "wb") as f:  # 打开文件用于写入模型
        pickle.dump(model, f)  # 将模型序列化保存到文件
    pred = predict(model,farm_id)  # 使用模型进行预测
    result_path = f'result/output'  # 设置结果保存路径
    os.makedirs(result_path,exist_ok=True)  # 创建结果保存目录（如果不存在）
    pred.to_csv(os.path.join(result_path,f'output{farm_id}.csv'))  # 将预测结果保存为CSV文件
print('ok')  # 打印完成信息
