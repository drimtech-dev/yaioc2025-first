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

# Define better PyTorch model with residual connections
class EnhancedModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super(EnhancedModel, self).__init__()
        layers = []
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.bn_input = nn.BatchNorm1d(hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Input layer
        out = self.activation(self.bn_input(self.input_layer(x)))
        out = self.dropout(out)
        
        # Hidden layers with residual connections where possible
        prev_out = out
        for i, (layer, bn) in enumerate(zip(self.hidden_layers, self.bn_layers)):
            out = layer(out)
            out = bn(out)
            out = self.activation(out)
            
            # Add residual connection if dimensions match
            if prev_out.shape == out.shape:
                out = out + prev_out
            
            out = self.dropout(out)
            prev_out = out
        
        # Output layer
        return self.output_layer(out)
    
    # Compatibility with sklearn-like interface
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self(X).numpy()

# Custom loss function that better matches the competition metric
class PowerForecastLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(PowerForecastLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-6
        
        # MSE component
        mse_loss = self.mse(pred, target)
        
        # Competition metric component: 1 - mean(|pred - actual|/actual)
        # We convert it to a loss by taking 1 - metric
        mask = target > epsilon  # To avoid division by zero
        if torch.sum(mask) > 0:
            pred_filtered = pred[mask]
            target_filtered = target[mask]
            relative_error = torch.abs(pred_filtered - target_filtered) / (target_filtered + epsilon)
            metric_loss = torch.mean(relative_error)
        else:
            metric_loss = 0.0
        
        # Combined loss
        return self.alpha * mse_loss + (1 - self.alpha) * metric_loss

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
    
    # More refined time periods
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    df['is_peak_sun'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)
    df['is_transition'] = (((df['hour'] >= 5) & (df['hour'] < 7)) | 
                           ((df['hour'] >= 17) & (df['hour'] < 19))).astype(int)
    
    # Season features (astronomical seasons)
    df['season'] = df['month'].apply(lambda m: 0 if m in [12, 1, 2] else  # Winter
                                         1 if m in [3, 4, 5] else  # Spring
                                         2 if m in [6, 7, 8] else  # Summer
                                         3)  # Fall
    df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4.0)
    df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4.0)
    
    # Day of week
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Interaction features
    df['day_month_sin'] = np.sin(2 * np.pi * (df['day'] + 31 * (df['month'] - 1)) / 365.0)
    df['day_month_cos'] = np.cos(2 * np.pi * (df['day'] + 31 * (df['month'] - 1)) / 365.0)
    
    return df

def add_wind_features(u_data, v_data, feature_prefix):
    """Add specialized wind features with physics-based derivations"""
    # Wind speed
    ws = np.sqrt(u_data ** 2 + v_data ** 2)
    ws_df = pd.DataFrame(ws, columns=[f"{feature_prefix}_ws_{i}" for i in range(ws.shape[1])])
    
    # Wind direction in degrees (0-360)
    wd = (np.arctan2(v_data, u_data) * 180 / np.pi) % 360
    wd_df = pd.DataFrame(wd, columns=[f"{feature_prefix}_wd_{i}" for i in range(wd.shape[1])])
    
    # Wind power is proportional to wind speed cubed (fundamental physics relationship)
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
    """Create a simplified model for better prediction and avoid feature mismatch"""
    print(f"Training with {x_processed.shape[1]} features")
    
    if is_wind_farm:
        # For wind farms, use GradientBoostingRegressor
        print("Training GradientBoostingRegressor for wind farm...")
        model = GradientBoostingRegressor(
            n_estimators=150,     # More trees for better accuracy
            learning_rate=0.08,   # Slightly lower learning rate for stability
            max_depth=5,          # Moderate depth to avoid overfitting
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.8,     # Use most features
            subsample=0.9,        # Use most of the data
            random_state=42
        )
        model.fit(x_processed.values, y_processed.values.ravel())
        
    else:  # Solar farm
        # For solar farms, use RandomForestRegressor
        print("Training RandomForestRegressor for solar farm...")
        model = RandomForestRegressor(
            n_estimators=150,    # More trees for better accuracy
            max_depth=8,         # Moderate depth to avoid overfitting
            min_samples_split=3,
            min_samples_leaf=2,
            max_features=0.8,    # Use most features
            n_jobs=-1,           # Use all cores
            random_state=42
        )
        model.fit(x_processed.values, y_processed.values.ravel())
    
    return model

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
                
                # Get temperature if available (affects panel efficiency)
                if 't2m' in nwp_data.channel:
                    t2m = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                      channel=['t2m']).data.values.reshape(365 * 24, 9)
                    t2m_df = pd.DataFrame(t2m, columns=[f"{nwp}_t2m_{i}" for i in range(t2m.shape[1])])
                    
                    # Convert to Celsius if in Kelvin (assuming t2m is in Kelvin)
                    if t2m.mean() > 100:  # Simple check if likely in Kelvin
                        for col in t2m_df.columns:
                            t2m_df[col] = t2m_df[col] - 273.15
                    
                    # Panel temperature estimation (simplified model)
                    # Panel temp is typically 20-30°C higher than ambient when under full sun
                    panel_temp_cols = []
                    for i in range(t2m.shape[1]):
                        col_name = f"{nwp}_panel_temp_{i}"
                        # Higher GHI = higher panel temp above ambient
                        ghi_col = f"{nwp}_ghi_{i}"
                        t2m_col = f"{nwp}_t2m_{i}"
                        panel_temp = t2m_df[t2m_col] + 25 * (ghi_df[ghi_col] / (ghi_df[ghi_col].max() + 1e-6))
                        t2m_df[col_name] = panel_temp
                        panel_temp_cols.append(col_name)
                    
                    # Panel efficiency factor (efficiency decreases with temperature)
                    # Typical temp coefficient is -0.4% per °C above 25°C
                    for col in panel_temp_cols:
                        eff_col = f"{col}_eff_factor"
                        t2m_df[eff_col] = 1.0 - 0.004 * np.maximum(0, t2m_df[col] - 25)
                
                # GHI features
                avg_ghi = np.mean(ghi, axis=1).reshape(-1, 1)
                max_ghi = np.max(ghi, axis=1).reshape(-1, 1)
                std_ghi = np.std(ghi, axis=1).reshape(-1, 1)  # Measure of spatial variability
                avg_ghi_df = pd.DataFrame(avg_ghi, columns=[f"{nwp}_avg_ghi"])
                max_ghi_df = pd.DataFrame(max_ghi, columns=[f"{nwp}_max_ghi"])
                std_ghi_df = pd.DataFrame(std_ghi, columns=[f"{nwp}_std_ghi"])
                
                # Cloud cover impact
                avg_tcc = np.mean(tcc, axis=1).reshape(-1, 1)
                tcc_impact = 1.0 - 0.7 * avg_tcc  # Simple model: clear sky (0) = 100%, overcast (1) = 30%
                tcc_impact_df = pd.DataFrame(tcc_impact, columns=[f"{nwp}_tcc_impact"])
                
                # Clear sky index (GHI ratio)
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
    
    # Feature selection based on correlation and importance
    correlations = []
    for col in x_processed.columns:
        corr = np.abs(np.corrcoef(x_processed[col], y_processed['power'])[0, 1])
        if not np.isnan(corr):  # Avoid NaN correlations
            correlations.append((col, corr))
    
    # Sort by correlation and keep top features
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Select features more intelligently based on farm type
    if is_wind_farm:
        # For wind farms, we want more features related to wind speed and direction
        wind_features = [col for col, _ in correlations if any(term in col for term in ['_ws_', '_wd_', '_ws3_', 'tcc'])]
        top_corr_features = [col for col, corr in correlations if corr > 0.3]  # Use correlation threshold instead of percentage
        top_features = list(set(wind_features + top_corr_features))
    else:
        # For solar farms, we want more features related to solar radiation and cloud cover
        solar_features = [col for col, _ in correlations if any(term in col for term in ['ghi', 'poai', 'tcc', 'cloud_impact'])]
        top_corr_features = [col for col, corr in correlations if corr > 0.3]  # Use correlation threshold instead of percentage
        top_features = list(set(solar_features + top_corr_features))
    
    # Always keep important time features
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_daytime']
    for feature in time_features:
        if feature in x_processed.columns and feature not in top_features:
            top_features.append(feature)
    
    # Use only selected features
    print(f"Selected {len(top_features)} features out of {x_processed.shape[1]} available features")
    x_processed = x_processed[top_features]
    
    # Create a train/validation split for model evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        x_processed.values, y_processed.values.ravel(), 
        test_size=0.15, random_state=42
    )
    
    # Simple ensemble approach - train multiple models with different settings
    if is_wind_farm:
        # For wind farms, use ensemble model
        ensemble = create_ensemble_model(x_processed, y_processed, is_wind_farm=True)
        return ensemble, top_features
    else:
        # For solar farms, use ensemble model
        ensemble = create_ensemble_model(x_processed, y_processed, is_wind_farm=False)
        return ensemble, top_features
    
    # Neural network approach (as fallback)
    # Convert to PyTorch tensors
    X = torch.tensor(x_processed.values, dtype=torch.float32)
    y = torch.tensor(y_processed.values, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    input_dim = X.shape[1]
    
    # Use different architectures based on farm type
    if is_wind_farm:
        model = EnhancedModel(input_dim, hidden_dims=[256, 128, 64])
    else:
        model = EnhancedModel(input_dim, hidden_dims=[192, 96, 64])
    
    # Use custom loss function
    criterion = PowerForecastLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler with warm-up
    def lr_lambda(epoch):
        if epoch < 5:  # Warm-up phase
            return 0.2 + 0.8 * epoch / 5
        else:  # Decay phase
            return 1.0 * (0.95 ** (epoch - 5))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    num_epochs = 200
    best_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Model 2: GradientBoostingRegressor as complementary model
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features=0.7,
            subsample=0.9,
            random_state=43  # Different seed
        )
        gb_model.fit(X_train, y_train)
        models.append(gb_model)
        
        # Calculate weights based on validation performance
        weights = []
        for model in models:
            preds = model.predict(X_val)
            mask = y_val > 1e-6
            if np.sum(mask) > 0:
                rel_error = np.mean(np.abs(preds[mask] - y_val[mask]) / y_val[mask])
                weights.append(max(0, 1 - rel_error))
            else:
                weights.append(1.0)
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        else:
            weights = [1.0/len(models)] * len(models)
        
        # Build weighted ensemble
        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def predict(self, X):
                preds = np.zeros((X.shape[0],))
                for model, weight in zip(self.models, self.weights):
                    preds += weight * model.predict(X)
                return preds
        
        ensemble_model = WeightedEnsemble(models, weights)
        
        # Verify ensemble performance
        ensemble_preds = ensemble_model.predict(X_val)
        best_individual_score = max(weights)
        
        mask = y_val > 1e-6
        if np.sum(mask) > 0:
            ensemble_error = np.mean(np.abs(ensemble_preds[mask] - y_val[mask]) / y_val[mask])
            ensemble_score = 1 - ensemble_error
        else:
            ensemble_score = 1.0
        
        if ensemble_score < best_individual_score - 0.02:
            best_model_idx = weights.index(max(weights))
            return models[best_model_idx], top_features
        
        return ensemble_model, top_features

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
    
    # Verify feature count
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
        if x_df.shape[1] != expected_features:
            print(f"Feature mismatch: model expecting {expected_features} features, but got {x_df.shape[1]}")
    
    # Make predictions
    if isinstance(model, (GradientBoostingRegressor, RandomForestRegressor, RidgeCV)):
        pred_pw = model.predict(x_df.values)
    else:
        # PyTorch model
        model.eval()
        with torch.no_grad():
            X = torch.tensor(x_df.values, dtype=torch.float32)
            pred_pw = model(X).flatten().numpy()
    
    # Post-processing for smoother predictions
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0], periods=len(pred_pw), freq='h'))
    
    # Apply adaptive smoothing based on farm type
    if is_wind_farm:
        # For wind farms, use adaptive window based on variability
        # Calculate rolling standard deviation to detect high variability periods
        rolling_std = pred.rolling(window=3, min_periods=1).std()
        
        # Apply stronger smoothing to high-variability periods
        for i in range(1, len(pred)-1):
            if rolling_std.iloc[i] > 0.1:  # High variability threshold
                # Apply stronger smoothing for highly variable periods
                pred.iloc[i] = 0.2 * pred.iloc[i-1] + 0.6 * pred.iloc[i] + 0.2 * pred.iloc[i+1]
            else:
                # Apply lighter smoothing for stable periods
                pred.iloc[i] = 0.1 * pred.iloc[i-1] + 0.8 * pred.iloc[i] + 0.1 * pred.iloc[i+1]
    else:
        # For solar farms, preserve the daily pattern better
        for i in range(1, len(pred)-1):
            hour = pred.index[i].hour
            
            # Apply different smoothing based on time of day
            if 5 <= hour <= 8 or 16 <= hour <= 19:  # Dawn/dusk transition periods
                # Stronger smoothing during transition periods
                pred.iloc[i] = 0.25 * pred.iloc[i-1] + 0.5 * pred.iloc[i] + 0.25 * pred.iloc[i+1]
            elif 9 <= hour <= 15:  # Midday (peak production)
                # Lighter smoothing during peak production
                pred.iloc[i] = 0.1 * pred.iloc[i-1] + 0.8 * pred.iloc[i] + 0.1 * pred.iloc[i+1]
            else:  # Night
                # Medium smoothing at night
                pred.iloc[i] = 0.2 * pred.iloc[i-1] + 0.6 * pred.iloc[i] + 0.2 * pred.iloc[i+1]
    
    # Apply smoother interpolation for 15min intervals
    res = pred.resample('15min').interpolate(method='cubic')
    
    # Apply constraints
    res[res < 0] = 0
    res[res > 1] = 1
    
    # For solar farms, ensure zero production at night
    if not is_wind_farm:
        hours = res.index.hour
        month = res.index.month
        
        # Create more precise seasonal boundaries for dusk/dawn times
        winter_mask = month.isin([11, 12, 1, 2])
        spring_mask = month.isin([3, 4])
        summer_mask = month.isin([5, 6, 7, 8])
        fall_mask = month.isin([9, 10])
        
        # Apply night hours based on season for each timestamp
        night_mask_winter = (hours >= 17) | (hours <= 7)  # Updated winter hours
        night_mask_summer = (hours >= 20) | (hours <= 5)  # Updated summer hours
        night_mask_default = (hours >= 19) | (hours <= 6)  # Updated default hours
        
        # Apply masks with bitwise operations
        res[winter_mask & night_mask_winter] = 0
        res[spring_mask & night_mask_spring] = 0
        res[summer_mask & night_mask_summer] = 0
        res[fall_mask & night_mask_fall] = 0
        
        # Smoother transitions at dawn/dusk (adjust production gradually)
        for hour, is_winter, is_summer in [(5, True, False), (6, True, False), (7, True, False), 
                                          (8, True, False), (17, True, False), (18, True, False),
                                          (5, False, True), (6, False, True), 
                                          (19, False, True), (20, False, True)]:
            
            dawn_dusk_mask = res.index.hour == hour
            if sum(dawn_dusk_mask) > 0:
                minute = res.index[dawn_dusk_mask].minute
                season_mask = winter_mask if is_winter else (summer_mask if is_summer else ~winter_mask & ~summer_mask)
                combined_mask = dawn_dusk_mask & season_mask
                
                if combined_mask.sum() > 0:
                    if hour in [5, 6, 7, 8]:  # Dawn - gradually increase
                        if hour == 5:
                            factor = minute / 120
                        elif hour == 6:
                            factor = (minute + 60) / 120
                        elif hour == 7:
                            factor = (minute + 120) / 180
                        else:  # hour == 8
                            factor = (minute + 180) / 240
                        res[combined_mask] = res[combined_mask] * factor
                    else:  # Dusk - gradually decrease
                        if hour == 17:
                            factor = (120 - minute) / 120
                        elif hour == 18:
                            factor = (60 - minute) / 120
                        elif hour == 19:
                            factor = (60 - minute) / 60
                        else:  # hour == 20
                            factor = (30 - minute) / 60
                            factor = np.maximum(factor, 0)  # Ensure non-negative
                        res[combined_mask] = res[combined_mask] * factor
        
        # Add cap to maximum production based on time of day
        midday_mask = (hours >= 10) & (hours <= 14)
        morning_mask = (hours >= 7) & (hours < 10)
        afternoon_mask = (hours > 14) & (hours <= 17)
        
        # Reduce production during non-peak hours
        res[morning_mask] = res[morning_mask] * 0.9
        res[afternoon_mask] = res[afternoon_mask] * 0.85
    
    return res

def process_farm(farm_id):
    """Process a single farm - for parallel processing"""
    farm_start_time = time.time()
    model_path = f'models/{farm_id}'
    os.makedirs(model_path, exist_ok=True)
    model_name = 'enhanced_model.pkl'
    features_name = 'features.pkl'
    
    try:
        print(f"\n===== Processing farm {farm_id} =====")
        print(f"Training model for farm {farm_id}...")
        model, top_features = train(farm_id)
        
        # Save the model and selected features
        print(f"Saving model for farm {farm_id}...")
        with open(os.path.join(model_path, model_name), "wb") as f:
            pickle.dump(model, f)
        
        with open(os.path.join(model_path, features_name), "wb") as f:
            pickle.dump(top_features, f)
        
        print(f"Making predictions for farm {farm_id}...")
        pred = predict(model, farm_id, top_features)
        result_path = f'result/output'
        os.makedirs(result_path, exist_ok=True)
        pred.to_csv(os.path.join(result_path, f'output{farm_id}.csv'))
        farm_time = time.time() - farm_start_time
        print(f'Successfully processed farm {farm_id} in {farm_time:.2f} seconds')
        return farm_id, True
    except Exception as e:
        print(f"Error processing farm {farm_id}: {str(e)}")
        return farm_id, False

# Main execution
if __name__ == "__main__":
    acc = pd.DataFrame()
    # Process all farms now that the approach is verified
    farms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Start timing
    total_start_time = time.time()
    
    # Option 1: Sequential processing
    # for farm_id in farms:
    #     process_farm(farm_id)
    
    # Option 2: Parallel processing with multiprocessing
    # Determine number of cores to use (max 4 to avoid memory issues)
    num_cores = min(4, mp.cpu_count())
    print(f"Using {num_cores} cores for parallel processing")
    
    # Create a pool of workers
    with mp.Pool(processes=num_cores) as pool:
        # Map process_farm to all farm_ids in parallel
        results = pool.map(process_farm, farms)
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Count successful farms
    successful = sum(1 for _, success in results if success)
    print(f"{successful} out of {len(farms)} farms successfully processed")
    print(f'All farms processed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
    
    # Play sound alert when finished (try multiple methods for different systems)
    try:
        # For Linux/Unix systems
        os.system('for i in {1..5}; do echo -e "\a"; sleep 0.5; done')
        # For Windows systems
        os.system('powershell -c "(New-Object Media.SoundPlayer).PlaySync([System.IO.Path]::Combine([System.Environment]::SystemDirectory, \'media\\Windows Notify.wav\'))"')
        # Alternative method for Windows
        os.system('echo \x07\x07\x07\x07\x07')
    except:
        # Fallback if above methods fail
        import sys
        for _ in range(5):
            sys.stdout.write('\a')
            sys.stdout.flush()
            time.sleep(0.5)
    
    print("\n处理完成！所有农场数据处理完毕！", flush=True)
