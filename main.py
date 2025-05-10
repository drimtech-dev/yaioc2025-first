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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
    
    # Add wind speed squared (related to dynamic pressure)
    ws_squared = ws ** 2
    ws_squared_df = pd.DataFrame(ws_squared, columns=[f"{feature_prefix}_ws2_{i}" for i in range(ws_squared.shape[1])])
    
    # Wind speed variability (std of 9 grid points, proxy for turbulence)
    ws_std = np.std(ws, axis=1).reshape(-1, 1)
    ws_std_df = pd.DataFrame(ws_std, columns=[f"{feature_prefix}_ws_std"])
    
    # Wind direction variability (circular standard deviation, proxy for directional stability)
    # Using approximation via resultant vector length
    sin_wd = np.sin(np.radians(wd))
    cos_wd = np.cos(np.radians(wd))
    mean_sin = np.mean(sin_wd, axis=1).reshape(-1, 1)
    mean_cos = np.mean(cos_wd, axis=1).reshape(-1, 1)
    resultant_length = np.sqrt(mean_sin**2 + mean_cos**2)
    dir_stability = resultant_length  # 1 = perfectly aligned, 0 = completely scattered
    dir_stability_df = pd.DataFrame(dir_stability, columns=[f"{feature_prefix}_dir_stability"])
    
    # Wind shear approximation (difference between max and min wind speed in grid)
    wind_shear = (np.max(ws, axis=1) - np.min(ws, axis=1)).reshape(-1, 1)
    wind_shear_df = pd.DataFrame(wind_shear, columns=[f"{feature_prefix}_wind_shear"])
    
    # Directionally weighted wind speed (project wind to main farm axis, typically along prevailing wind direction)
    # Assuming cardinal directions as examples (can be refined with actual farm data)
    ns_component = np.abs(v_data).mean(axis=1).reshape(-1, 1)  # North-South component
    ew_component = np.abs(u_data).mean(axis=1).reshape(-1, 1)  # East-West component
    ns_df = pd.DataFrame(ns_component, columns=[f"{feature_prefix}_ns_wind"])
    ew_df = pd.DataFrame(ew_component, columns=[f"{feature_prefix}_ew_wind"])
    
    return pd.concat([ws_df, wd_df, ws_squared_df, ws_cubed_df, ws_std_df, 
                      dir_stability_df, wind_shear_df, ns_df, ew_df], axis=1)

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
                
                # Combine all features
                if 't2m' in nwp_data.channel:
                    nwp_df = pd.concat([ghi_df, poai_df, tcc_df, avg_ghi_df, max_ghi_df, std_ghi_df,
                                      tcc_impact_df, ghi_ratio_df, t2m_df], axis=1)
                else:
                    nwp_df = pd.concat([ghi_df, poai_df, tcc_df, avg_ghi_df, max_ghi_df, std_ghi_df,
                                      tcc_impact_df, ghi_ratio_df], axis=1)
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
    
    # Feature selection based on correlation with target (simple but effective)
    correlations = []
    for col in x_processed.columns:
        corr = np.abs(np.corrcoef(x_processed[col], y_processed['power'])[0, 1])
        if not np.isnan(corr):  # Avoid NaN correlations
            correlations.append((col, corr))
    
    # Sort by correlation and keep top features (70% for wind, 60% for solar)
    correlations.sort(key=lambda x: x[1], reverse=True)
    if is_wind_farm:
        top_features = [col for col, _ in correlations[:int(len(correlations)*0.7)]]
    else:
        top_features = [col for col, _ in correlations[:int(len(correlations)*0.6)]]
    
    # Keep important time features regardless of correlation
    for col in x_processed.columns:
        if any(time_feat in col for time_feat in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_daytime']):
            if col not in top_features:
                top_features.append(col)
    
    # Use only selected features if provided
    if top_features:
        # Ensure all required features exist, create with zeros if missing
        missing_features = [feature for feature in top_features if feature not in x_df.columns]
        if missing_features:
            # Create a DataFrame with zeros for all missing features at once
            missing_df = pd.DataFrame(0, index=x_df.index, columns=missing_features)
            # Concatenate with original DataFrame
            x_df = pd.concat([x_df, missing_df], axis=1)
        x_df = x_df[top_features]
    
    # Create a train/validation split for model evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        x_processed.values, y_processed.values.ravel(), 
        test_size=0.15, random_state=42
    )
    
    # Simple ensemble approach - train multiple models with different settings
    if is_wind_farm:
        # For wind farms, use both Gradient Boosting and Random Forest
        models = []
        
        # Model 1: GradientBoostingRegressor with optimized parameters
        gb_model = GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=7,
            min_samples_split=6,
            min_samples_leaf=4,
            max_features=0.8,
            subsample=0.85,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        models.append(gb_model)
        
        # Model 2: RandomForestRegressor as a complementary model
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features=0.7,
            bootstrap=True,
            random_state=43  # Different seed
        )
        rf_model.fit(X_train, y_train)
        models.append(rf_model)
        
        # Calculate weights based on validation performance
        # Using 1 - abs_rel_error as the metric (higher is better)
        weights = []
        for model in models:
            preds = model.predict(X_val)
            # Calculate rel error where possible (avoid div by zero)
            mask = y_val > 1e-6
            if np.sum(mask) > 0:
                rel_error = np.mean(np.abs(preds[mask] - y_val[mask]) / y_val[mask])
                weights.append(max(0, 1 - rel_error))  # Higher weight for better models
            else:
                weights.append(1.0)  # Default weight
        
        # Normalize weights to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        else:
            weights = [1.0/len(models)] * len(models)
        
        # Build a simple weighted ensemble class
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
        
        # Verify ensemble performance is better than individual models
        ensemble_preds = ensemble_model.predict(X_val)
        best_individual_score = max(weights)
        
        # Calculate ensemble score
        mask = y_val > 1e-6
        if np.sum(mask) > 0:
            ensemble_error = np.mean(np.abs(ensemble_preds[mask] - y_val[mask]) / y_val[mask])
            ensemble_score = 1 - ensemble_error
        else:
            ensemble_score = 1.0
        
        # If ensemble is worse than best model, use best model instead
        if ensemble_score < best_individual_score - 0.02:  # Only switch if significantly worse
            best_model_idx = weights.index(max(weights))
            return models[best_model_idx], top_features
        
        return ensemble_model, top_features
    else:
        # For solar farms, similar ensemble approach
        models = []
        
        # Model 1: RandomForestRegressor with optimized parameters
        rf_model = RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features=0.75,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        models.append(rf_model)
        
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
                
                nwp_df = pd.concat([ghi_df, poai_df, tcc_df, avg_ghi_df, max_ghi_df, ghi_ratio_df], axis=1)
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
        if missing_features:
            # Create a DataFrame with zeros for all missing features at once
            missing_df = pd.DataFrame(0, index=x_df.index, columns=missing_features)
            # Concatenate with original DataFrame
            x_df = pd.concat([x_df, missing_df], axis=1)
        x_df = x_df[top_features]
    
    # Make predictions
    if isinstance(model, (GradientBoostingRegressor, RandomForestRegressor)):
        pred_pw = model.predict(x_df.values)
    else:
        # PyTorch model
        model.eval()
        with torch.no_grad():
            X = torch.tensor(x_df.values, dtype=torch.float32)
            pred_pw = model(X).flatten().numpy()
    
    # Post-processing for smoother predictions
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0], periods=len(pred_pw), freq='h'))
    
    # Apply smoothing with weighted moving average before resampling
    # This helps reduce noise in predictions while preserving important patterns
    if is_wind_farm:
        # Wind power can fluctuate rapidly, use 3-hour window with more weight on central value
        window_size = 3
        weights = np.array([0.2, 0.6, 0.2]) # More weight on current hour
        
        # Apply smoothing only if we have enough data points
        if len(pred) >= window_size:
            smoothed_values = np.convolve(pred.values, weights, mode='same')
            # Fix the edges (first and last points have no full window)
            smoothed_values[0] = pred.values[0] * 0.8 + pred.values[1] * 0.2
            smoothed_values[-1] = pred.values[-2] * 0.2 + pred.values[-1] * 0.8
            pred = pd.Series(smoothed_values, index=pred.index)
    else:
        # Solar power has more regular daily patterns, use slightly larger window
        window_size = 5
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1]) # More weight on central value
        
        # Apply smoothing only if we have enough data points
        if len(pred) >= window_size:
            smoothed_values = np.convolve(pred.values, weights, mode='same')
            # Fix the edges
            smoothed_values[0] = pred.values[0] * 0.7 + pred.values[1] * 0.3
            smoothed_values[1] = pred.values[0] * 0.2 + pred.values[1] * 0.5 + pred.values[2] * 0.3
            smoothed_values[-2] = pred.values[-3] * 0.3 + pred.values[-2] * 0.5 + pred.values[-1] * 0.2
            smoothed_values[-1] = pred.values[-2] * 0.3 + pred.values[-1] * 0.7
            pred = pd.Series(smoothed_values, index=pred.index)
    
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
        
        # More precise night hours based on season
        night_mask_winter = (hours >= 17) | (hours <= 7)  # Shorter days in winter
        night_mask_spring = (hours >= 19) | (hours <= 6)  # Transition season
        night_mask_summer = (hours >= 21) | (hours <= 5)  # Longer days in summer
        night_mask_fall = (hours >= 18) | (hours <= 6)    # Transition season
        
        # Apply masks with bitwise operations
        res[winter_mask & night_mask_winter] = 0
        res[spring_mask & night_mask_spring] = 0
        res[summer_mask & night_mask_summer] = 0
        res[fall_mask & night_mask_fall] = 0
        
        # Create more realistic dawn/dusk transitions based on season
        # Winter dawn (gradual 2-hour transition)
        for hour in range(6, 9):
            dawn_mask = winter_mask & (res.index.hour == hour)
            if sum(dawn_mask) > 0:
                minute = res.index[dawn_mask].minute
                # Gradual ramp up over 2 hours
                factor = ((hour - 6) * 60 + minute) / 120
                factor = np.minimum(1.0, factor)  # Cap at 1.0
                res[dawn_mask] = res[dawn_mask] * factor
        
        # Winter dusk (gradual transition)
        for hour in range(15, 18):
            dusk_mask = winter_mask & (res.index.hour == hour)
            if sum(dusk_mask) > 0:
                minute = res.index[dusk_mask].minute
                # Gradual ramp down
                factor = (17 * 60 + 60 - (hour * 60 + minute)) / 120
                factor = np.maximum(0.0, factor)  # Floor at 0.0
                res[dusk_mask] = res[dusk_mask] * factor
        
        # Summer dawn (gradual transition)
        for hour in range(4, 7):
            dawn_mask = summer_mask & (res.index.hour == hour)
            if sum(dawn_mask) > 0:
                minute = res.index[dawn_mask].minute
                factor = ((hour - 4) * 60 + minute) / 120
                factor = np.minimum(1.0, factor)
                res[dawn_mask] = res[dawn_mask] * factor
        
        # Summer dusk (gradual transition)
        for hour in range(19, 22):
            dusk_mask = summer_mask & (res.index.hour == hour)
            if sum(dusk_mask) > 0:
                minute = res.index[dusk_mask].minute
                factor = (21 * 60 + 60 - (hour * 60 + minute)) / 120
                factor = np.maximum(0.0, factor)
                res[dusk_mask] = res[dusk_mask] * factor
        
        # Spring and Fall (intermediate transitions)
        # Dawn
        for hour in range(5, 8):
            dawn_mask = (spring_mask | fall_mask) & (res.index.hour == hour)
            if sum(dawn_mask) > 0:
                minute = res.index[dawn_mask].minute
                factor = ((hour - 5) * 60 + minute) / 120
                factor = np.minimum(1.0, factor)
                res[dawn_mask] = res[dawn_mask] * factor
        
        # Dusk
        for hour in range(17, 20):
            dusk_mask = (spring_mask | fall_mask) & (res.index.hour == hour)
            if sum(dusk_mask) > 0:
                minute = res.index[dusk_mask].minute
                factor = (19 * 60 + 60 - (hour * 60 + minute)) / 120
                factor = np.maximum(0.0, factor)
                res[dusk_mask] = res[dusk_mask] * factor
    
    return res

acc = pd.DataFrame()
farms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for farm_id in farms:
    model_path = f'models/{farm_id}'
    os.makedirs(model_path, exist_ok=True)
    model_name = 'enhanced_model.pkl'
    features_name = 'features.pkl'
    
    try:
        model, top_features = train(farm_id)
        
        # Save the model and selected features
        with open(os.path.join(model_path, model_name), "wb") as f:
            pickle.dump(model, f)
        
        with open(os.path.join(model_path, features_name), "wb") as f:
            pickle.dump(top_features, f)
        
        pred = predict(model, farm_id, top_features)
        result_path = f'result/output'
        os.makedirs(result_path, exist_ok=True)
        pred.to_csv(os.path.join(result_path, f'output{farm_id}.csv'))
        print(f'Successfully processed farm {farm_id}')
    except Exception as e:
        print(f"Error processing farm {farm_id}: {str(e)}")
        # Fallback to simple linear model if enhanced model fails
        # This ensures we always have a prediction
        print(f"Falling back to linear model for farm {farm_id}")
        
        # Load data again for fallback model
        is_wind_farm = farm_id <= 5
        
        # Simplified data processing for fallback
        x_df = pd.DataFrame()
        nwp_test_path = f'training/middle_school/TEST/nwp_data_test/{farm_id}'
        
        for nwp in nwps:
            nwp_path = os.path.join(nwp_test_path, nwp)
            nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
            
            if is_wind_farm:
                u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['u100']).data.values.reshape(31 * 24, 9)
                v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['v100']).data.values.reshape(31 * 24, 9)
                
                ws = np.sqrt(u ** 2 + v ** 2)
                ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
                ws_mean = np.mean(ws, axis=1).reshape(-1, 1)
                ws_mean_df = pd.DataFrame(ws_mean, columns=[f"{nwp}_ws_mean"])
                
                nwp_df = pd.concat([ws_df, ws_mean_df], axis=1)
            else:
                if 'ghi' in nwp_data.channel:
                    ghi = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                       channel=['ghi']).data.values.reshape(31 * 24, 9)
                    ghi_df = pd.DataFrame(ghi, columns=[f"{nwp}_ghi_{i}" for i in range(ghi.shape[1])])
                    ghi_mean = np.mean(ghi, axis=1).reshape(-1, 1)
                    ghi_mean_df = pd.DataFrame(ghi_mean, columns=[f"{nwp}_ghi_mean"])
                    
                    nwp_df = pd.concat([ghi_df, ghi_mean_df], axis=1)
                else:
                    # If no GHI, use wind as proxy (less ideal but better than nothing)
                    u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                    channel=['u100']).data.values.reshape(31 * 24, 9)
                    v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                    channel=['v100']).data.values.reshape(31 * 24, 9)
                    ws = np.sqrt(u ** 2 + v ** 2)
                    ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
                    nwp_df = pd.concat([ws_df], axis=1)
            
            x_df = pd.concat([x_df, nwp_df], axis=1)
        
        x_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')
        
        # Add basic time features
        x_df['hour'] = x_df.index.hour
        x_df['hour_sin'] = np.sin(2 * np.pi * x_df['hour'] / 24.0)
        x_df['hour_cos'] = np.cos(2 * np.pi * x_df['hour'] / 24.0)
        x_df['month'] = x_df.index.month
        x_df['day'] = x_df.index.day
        x_df['is_daytime'] = ((x_df['hour'] >= 6) & (x_df['hour'] <= 18)).astype(int)
        
        # Load training data for the linear model
        nwp_train_path = f'training/middle_school/TRAIN/nwp_data_train/{farm_id}'
        x_train_df = pd.DataFrame()
        
        for nwp in nwps:
            nwp_path = os.path.join(nwp_train_path, nwp)
            nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")
            
            if is_wind_farm:
                u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['u100']).data.values.reshape(365 * 24, 9)
                v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                channel=['v100']).data.values.reshape(365 * 24, 9)
                
                ws = np.sqrt(u ** 2 + v ** 2)
                ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
                ws_mean = np.mean(ws, axis=1).reshape(-1, 1)
                ws_mean_df = pd.DataFrame(ws_mean, columns=[f"{nwp}_ws_mean"])
                
                nwp_df = pd.concat([ws_df, ws_mean_df], axis=1)
            else:
                if 'ghi' in nwp_data.channel:
                    ghi = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                       channel=['ghi']).data.values.reshape(365 * 24, 9)
                    ghi_df = pd.DataFrame(ghi, columns=[f"{nwp}_ghi_{i}" for i in range(ghi.shape[1])])
                    ghi_mean = np.mean(ghi, axis=1).reshape(-1, 1)
                    ghi_mean_df = pd.DataFrame(ghi_mean, columns=[f"{nwp}_ghi_mean"])
                    
                    nwp_df = pd.concat([ghi_df, ghi_mean_df], axis=1)
                else:
                    u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                    channel=['u100']).data.values.reshape(365 * 24, 9)
                    v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=range(24),
                                    channel=['v100']).data.values.reshape(365 * 24, 9)
                    ws = np.sqrt(u ** 2 + v ** 2)
                    ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
                    nwp_df = pd.concat([ws_df], axis=1)
            
            x_train_df = pd.concat([x_train_df, nwp_df], axis=1)
        
        x_train_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')
        
        # Add basic time features
        x_train_df['hour'] = x_train_df.index.hour
        x_train_df['hour_sin'] = np.sin(2 * np.pi * x_train_df['hour'] / 24.0)
        x_train_df['hour_cos'] = np.cos(2 * np.pi * x_train_df['hour'] / 24.0)
        x_train_df['month'] = x_train_df.index.month
        x_train_df['day'] = x_train_df.index.day
        x_train_df['is_daytime'] = ((x_train_df['hour'] >= 6) & (x_train_df['hour'] <= 18)).astype(int)
        
        # Get target data
        y_train_df = pd.read_csv(os.path.join(fact_path, f'{farm_id}_normalization_train.csv'), index_col=0)
        y_train_df.index = pd.to_datetime(y_train_df.index)
        y_train_df.columns = ['power']
        
        # Align data
        x_train_df, y_train_df = data_preprocess(x_train_df, y_train_df)
        y_train_df[y_train_df < 0] = 0
        
        # Train simple linear model
        fallback_model = LinearRegression()
        fallback_model.fit(x_train_df.values, y_train_df.values)
        
        # Make predictions
        fallback_pred = fallback_model.predict(x_df.values)
        fallback_pred[fallback_pred < 0] = 0
        fallback_pred[fallback_pred > 1] = 1
        
        # Format predictions
        fallback_series = pd.Series(fallback_pred.flatten(), 
                                   index=pd.date_range(x_df.index[0], periods=len(fallback_pred), freq='h'))
        
        # Resample to 15-min intervals
        fallback_res = fallback_series.resample('15min').interpolate(method='linear')
        
        # For solar farms, zero out nighttime values
        if not is_wind_farm:
            hours = fallback_res.index.hour
            fallback_res[((hours >= 20) | (hours <= 5))] = 0
        
        # Save fallback predictions
        fallback_res.to_csv(os.path.join(result_path, f'output{farm_id}.csv'))
        print(f'Fallback model completed for farm {farm_id}')
        
print('All farms processed')
