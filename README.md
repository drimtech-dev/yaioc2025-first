# 新能源发电功率预测系统

该项目为第五届长三角青少年人工智能奥林匹克挑战赛 - "AI算法争霸"赛道（第三届世界科学智能大赛，中学-高中组）**初赛**的解决方案，聚焦于新能源发电功率预测任务。

## 项目概述

本项目旨在根据历史发电功率数据和对应时段多类别气象预测数据，实现次日零时起到未来24小时逐15分钟级新能源场站发电功率预测。通过机器学习和深度学习模型，为风电场和光伏电站提供准确的发电功率预测。

## 数据说明

项目使用的数据包括：

1. **气象数据**：来自三个不同的气象预报源（NWP_1, NWP_2, NWP_3），包含如下变量：
   - 风力相关：`u100`（100米高度纬向风）、`v100`（100米高度经向风）
   - 温度相关：`t2m`（2米气温）
   - 降水相关：`tp`（总降水量）
   - 云量相关：`tcc`（总云量）
   - 气压相关：`sp`（地面气压）、`msl`（海平面气压）
   - 辐照度相关：`poai`（光伏面板辐照度）、`ghi`（水平面总辐照度）

2. **场站实发功率**：来自10个新能源场站的归一化处理后的实发功率数据
   - 场站1-5：风电场
   - 场站6-10：光伏电场
   - 数据时间为北京时间，时间间隔为15分钟

## 项目结构

```
.
├── main.py               # 主程序代码，包含模型训练和预测功能
├── models/               # 保存训练好的模型
│   ├── 1/                # 场站1的模型
│   ├── 2/                # 场站2的模型
│   └── ...
├── training/             # 训练数据目录
│   └── middle_school/
│       ├── TRAIN/        # 训练集
│       └── TEST/         # 测试集
├── result/               # 预测结果输出目录
├── explanation.md        # 比赛说明文档
└── README.md             # 项目说明文档（当前文件）
```

## 模型设计

项目采用了多种模型，根据不同类型的发电场站（风电/光伏）进行差异化处理：

1. **增强型神经网络模型**（EnhancedModel）：
   - 包含残差连接的深度学习模型
   - 批量归一化和Dropout防止过拟合
   - 自定义损失函数（PowerForecastLoss）更好地匹配竞赛评估指标

2. **梯度提升回归树模型**（GradientBoostingRegressor）

3. **随机森林回归模型**（RandomForestRegressor）

## 特征工程

项目实现了丰富的特征工程：

1. **时间特征**：
   - 小时特征（正弦和余弦变换）
   - 日特征（正弦和余弦变换）
   - 月特征（正弦和余弦变换）
   - 昼夜指标

2. **专业特征**：
   - **风电场**：风速、风向、风力立方（与风能成正比）等特征
   - **光伏场**：辐照度、云量、日照角度等特征

3. **滞后特征**：包含前一时间步特征值
4. **差分特征**：特征值的变化率

## 使用方法

1. **训练模型**：
   ```bash
   python main.py --train
   ```

2. **生成预测**：
   ```bash
   python main.py --predict
   ```

## 评估指标

项目使用与比赛一致的评估指标：

1. **每日预测准确率**（CR）：
   ```
   CR = 1 - (∑|PM,i - PP,i| / ∑PM,i)
   ```
   其中，PM,i为实际功率，PP,i为预测功率

2. **场站预测准确率**（Cf）：所有预测日的精度平均值

3. **最终准确率**（C）：所有场站精度的平均值

## 特别说明

- 模型在不同类型的发电场站（风电/光伏）上采用了不同的特征工程和参数设置
- 项目严格遵循比赛规则，不使用外部数据或预训练模型 