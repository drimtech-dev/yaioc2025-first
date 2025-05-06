import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesTransformer(nn.Module):
    """使用Transformer架构的时间序列模型"""
    def __init__(self, feature_size, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.feature_size = feature_size
        self.d_model = d_model
        
        # 特征投影层
        self.input_projection = nn.Linear(feature_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # 调整输入形状为 [batch_size, seq_len=1, feature_size]
        x = x.unsqueeze(1)
        
        # 特征投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        # 输出层
        output = self.output_projection(x).squeeze(-1)
        
        return output.squeeze(1)

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class WindPowerModel(nn.Module):
    """专为风电场设计的模型"""
    def __init__(self, input_size, hidden_dims=[256, 128]):
        super(WindPowerModel, self).__init__()
        
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
        
        # 添加特殊的风速处理层
        self.wind_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dims[0]),
            nn.Sigmoid()  # 风速影响因子
        )
        
    def forward(self, x):
        # 主要特征路径
        main_path = self.model(x)
        
        # 风速特殊处理路径
        wind_factor = self.wind_layer(x)
        
        # 结合两个路径（残差连接）
        output = main_path * wind_factor.mean(dim=1, keepdim=True)
        
        return output.squeeze(-1)

class SolarPowerModel(nn.Module):
    """专为光伏场站设计的模型"""
    def __init__(self, input_size, hidden_dims=[256, 128]):
        super(SolarPowerModel, self).__init__()
        
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
        
        # 添加特殊的日照/温度处理层
        self.solar_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dims[0]),
            nn.Sigmoid()  # 日照影响因子
        )
        
        # 添加日夜分类器（增加零值预测的准确性）
        self.day_night_classifier = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 主要特征路径
        main_path = self.model(x)
        
        # 日照特殊处理路径
        solar_factor = self.solar_layer(x)
        
        # 日夜分类（接近0表示夜间，接近1表示白天）
        day_night = self.day_night_classifier(x)
        
        # 结合（夜间输出接近0）
        output = main_path * solar_factor.mean(dim=1, keepdim=True) * day_night
        
        return output.squeeze(-1)

def create_model_for_farm(farm_id, input_size, is_wind_farm=True):
    """根据场站类型创建适合的模型"""
    if is_wind_farm:
        return WindPowerModel(input_size, hidden_dims=[256, 128])
    else:
        return SolarPowerModel(input_size, hidden_dims=[256, 128])
