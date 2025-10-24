#Note: Although it only tellls LFSM in the code, the model when trained with 1D zigzag beam dataset, 
#it is Low Fidelity Surrogate Model(LFSM), when it gets fine tunec with 2D data, it becomes 
#Multi Fidelity Surrogate Model(MFSM)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import time
import joblib
import logging

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# --- Configuration Dictionary ---
CONFIG = {
    # --- GPU/CPU Settings ---
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'USE_XGB_GPU': True,

    # --- File Paths & Dirs ---
    'DATA_FILE_TRAIN': '/home/user2/Music/abhi3/parameters/LFSM6000train.csv',
    'DATA_FILE_TEST': '/home/user2/Music/abhi3/parameters/LFSM6000test.csv',
    'DATA_FILE_2D_TRAIN': [
        '/home/user2/Music/abhi3/parameters/train_responses.csv'
    ],
    'DATA_FILE_2D_TEST': '/home/user2/Music/abhi3/parameters/test_responses.csv',
    'OUTPUT_DIR': '/home/user2/Music/abhi3/MFSM/Finetuning',
    
    # --- Data & Model Hyperparameters ---
    'PARAM_COLS': ['notch_x', 'notch_depth', 'notch_width', 'length', 'density', 'youngs_modulus'],
    'NUM_TIME_STEPS': 1500,
    'VAL_SPLIT_RATIO': 0.2,  
    
    # --- CAE Training ---
    'LATENT_DIM': 64,  # Reduced from 100 to make latent space easier to predict
    'CAE_EPOCHS_1D': 200,
    'CAE_EPOCHS_2D': 200,  # Fewer epochs for fine-tuning
    'CAE_BATCH_SIZE': 64,
    'CAE_LEARNING_RATE': 1e-4,
    'CAE_LEARNING_RATE_2D': 5e-5,  # Lower learning rate for fine-tuning
    
    # --- XGBoost Surrogate Model ---
    'XGB_N_ESTIMATORS': 2000,   # Increased for better latent prediction
    'XGB_MAX_DEPTH': 10,        # Increased depth for better prediction
    'XGB_ETA': 0.03,           # Lower learning rate for better convergence
    'XGB_EARLY_STOPPING': 100,  # More patience for early stopping
}
CONFIG['TIME_COLS'] = [f't_{i}' for i in range(1, CONFIG["NUM_TIME_STEPS"] + 1)]

# --- Setup Logging ---
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(CONFIG['OUTPUT_DIR'], 'sequential_training_log.log')),
                              logging.StreamHandler()])

# --- PyTorch Dataset Class ---
class BeamResponseDataset(Dataset):
    """
    Dataset for beam response data with parameter scaling to [-1, 1].
    Responses are already normalized and left unchanged.
    """
    def __init__(self, params, timeseries, p_scaler=None):
        # Store raw time series without scaling - responses are already normalized
        self.timeseries = timeseries.astype(np.float32)
        self.params = params.astype(np.float32)
        
        # Scale parameters using StandardScaler for better handling of different ranges
        if p_scaler is None:
            # Use StandardScaler for parameters as they have very different ranges
            self.p_scaler = StandardScaler()
            self.params_scaled = self.p_scaler.fit_transform(self.params)
        else:
            self.p_scaler = p_scaler
            self.params_scaled = self.p_scaler.transform(self.params)

        # Ensure float32 and contiguous for XGBoost efficiency
        self.params_scaled = np.ascontiguousarray(self.params_scaled, dtype=np.float32)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return {
            'params': torch.tensor(self.params_scaled[idx], dtype=torch.float32),
            'timeseries': torch.tensor(self.timeseries[idx], dtype=torch.float32),
            'timeseries_raw': self.timeseries[idx]  # Keep raw for evaluation
        }

# --- Conditional Autoencoder PyTorch Models ---
class Encoder(nn.Module):
    def __init__(self, timeseries_dim, params_dim, latent_dim):
        super(Encoder, self).__init__()
        # Enhanced architecture inspired by the working 2D version
        self.timeseries_net = nn.Sequential(
            nn.Linear(timeseries_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.params_net = nn.Sequential(
            nn.Linear(params_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x, c):
        ts_features = self.timeseries_net(x)
        param_features = self.params_net(c)
        combined = torch.cat([ts_features, param_features], dim=1)
        return self.fusion(combined)

class Decoder(nn.Module):
    def __init__(self, latent_dim, params_dim, output_dim):
        super(Decoder, self).__init__()
        self.params_net = nn.Sequential(
            nn.Linear(params_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        self.expansion = nn.Sequential(
            nn.Linear(latent_dim + 256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, z, c):
        param_features = self.params_net(c)
        combined = torch.cat([z, param_features], dim=1)
        return self.expansion(combined)

class ConditionalAutoencoder(nn.Module):
    def __init__(self, timeseries_dim, params_dim, latent_dim):
        super(ConditionalAutoencoder, self).__init__()
        self.encoder = Encoder(timeseries_dim, params_dim, latent_dim)
        self.decoder = Decoder(latent_dim, params_dim, timeseries_dim)
    
    def forward(self, x, c):
        z = self.encoder(x, c)
        recon_x = self.decoder(z, c)
        return recon_x, z

# --- Main Functions ---
def load_1d_data():
    """Load 1D training and test data directly from separate CSV files"""
    logging.info("=== LOADING 1D DATA ===")
    
    # Load 1D training data
    logging.info(f"Loading 1D training data from {CONFIG['DATA_FILE_TRAIN']}")
    df_train = pd.read_csv(CONFIG['DATA_FILE_TRAIN'])
    
    # Load 1D test data
    logging.info(f"Loading 1D test data from {CONFIG['DATA_FILE_TEST']}")
    df_test = pd.read_csv(CONFIG['DATA_FILE_TEST'])
    
    logging.info(f"1D Original Training samples: {len(df_train)}, Test samples: {len(df_test)}")
    
    # Check for NaN values in training data
    if df_train.isnull().values.any():
        nan_count = df_train.isnull().sum().sum()
        logging.warning(f"Found {nan_count} NaN values in 1D training dataset. Dropping rows with NaNs.")
        df_train.dropna(inplace=True)
        logging.info(f"1D Training shape after dropping NaNs: {df_train.shape}")
    
    # Check for NaN values in test data
    if df_test.isnull().values.any():
        nan_count = df_test.isnull().sum().sum()
        logging.warning(f"Found {nan_count} NaN values in 1D test dataset. Dropping rows with NaNs.")
        df_test.dropna(inplace=True)
        logging.info(f"1D Test shape after dropping NaNs: {df_test.shape}")
    
    # Add location column
    df_train['location'] = df_train['response_point']
    df_test['location'] = df_test['response_point']
    param_features = CONFIG['PARAM_COLS'] + ['location']
    
    # Extract features and responses from training data
    X_train = df_train[param_features].values
    Y_train = df_train[CONFIG['TIME_COLS']].values
    
    # Extract features and responses from test data (full test set)
    X_test_full = df_test[param_features].values
    Y_test_full = df_test[CONFIG['TIME_COLS']].values
    
    # Sample validation set from test data
    logging.info("Sampling validation set from test data...")
    val_size = int(len(X_test_full) * CONFIG['VAL_SPLIT_RATIO'])
    val_indices = np.random.choice(len(X_test_full), size=val_size, replace=False)
    
    X_val = X_test_full[val_indices]
    Y_val = Y_test_full[val_indices]
    
    logging.info(f"1D Split sizes - Train: {len(X_train)}, Validation: {len(X_val)}, Test (full): {len(X_test_full)}")
    logging.info(f"1D Validation ratio from test set: {len(X_val)/len(X_test_full):.1%}")
    
    logging.info(f"1D Data shapes after processing:")
    logging.info(f"  Training: X: {X_train.shape}, Y: {Y_train.shape}")
    logging.info(f"  Validation: X: {X_val.shape}, Y: {Y_val.shape}")
    logging.info(f"  Test (full): X: {X_test_full.shape}, Y: {Y_test_full.shape}")
    
    logging.info(f"1D Training Y stats - Min: {Y_train.min():.6f}, Max: {Y_train.max():.6f}, Mean: {Y_train.mean():.6f}")
    
    return X_train, X_val, X_test_full, Y_train, Y_val, Y_test_full

def load_2d_data():
    """Load 2D training and test data directly from separate CSV files"""
    logging.info("=== LOADING 2D DATA ===")

    # Load 2D training data from multiple files
    train_files = CONFIG['DATA_FILE_2D_TRAIN']
    if isinstance(train_files, list):
        logging.info(f"Loading 2D training data from {len(train_files)} files: {train_files}")
        dfs = []
        for f in train_files:
            df = pd.read_csv(f)
            # Standardize column names: convert r_1, r_2, ... to r1, r2, ...
            # or convert r1, r2, ... to r_1, r_2, ... if needed
            # Find time columns and standardize their naming
            time_cols = [col for col in df.columns if col.startswith('r') and (col[1:].isdigit() or (col[2:].isdigit() and col[1] == '_'))]
            if time_cols:
                # Standardize to r1, r2, ... format (without underscores)
                new_time_cols = {}
                for col in time_cols:
                    if col[1] == '_':  # Format: r_1, r_2, ... -> r1, r2, ...
                        new_col = 'r' + col[2:]  # r_1 -> r1
                    else:  # Format: r1, r2, ... -> keep as is
                        new_col = col
                    new_time_cols[col] = new_col
                df = df.rename(columns=new_time_cols)
            dfs.append(df)
        df_2d_train = pd.concat(dfs, ignore_index=True)
    else:
        logging.info(f"Loading 2D training data from {train_files}")
        df_2d_train = pd.read_csv(train_files)
    
    # Load 2D test data
    logging.info(f"Loading 2D test data from {CONFIG['DATA_FILE_2D_TEST']}")
    df_2d_test = pd.read_csv(CONFIG['DATA_FILE_2D_TEST'])
    # Standardize column names in test data as well
    time_cols = [col for col in df_2d_test.columns if col.startswith('r') and (col[1:].isdigit() or (col[2:].isdigit() and col[1] == '_'))]
    if time_cols:
        # Standardize to r1, r2, ... format (without underscores)
        new_time_cols = {}
        for col in time_cols:
            if col[1] == '_':  # Format: r_1, r_2, ... -> r1, r2, ...
                new_col = 'r' + col[2:]  # r_1 -> r1
            else:  # Format: r1, r2, ... -> keep as is
                new_col = col
            new_time_cols[col] = new_col
        df_2d_test = df_2d_test.rename(columns=new_time_cols)
    
    logging.info(f"2D Original Training samples: {len(df_2d_train)}, Test samples: {len(df_2d_test)}")
    if isinstance(train_files, list):
        logging.info(f"Combined training data from {len(train_files)} files")
    
    # Check for NaN values in 2D training data
    if df_2d_train.isnull().values.any():
        nan_count = df_2d_train.isnull().sum().sum()
        logging.warning(f"Found {nan_count} NaN values in 2D training dataset. Dropping rows with NaNs.")
        df_2d_train.dropna(inplace=True)
        logging.info(f"2D Training shape after dropping NaNs: {df_2d_train.shape}")
    
    # Check for NaN values in 2D test data
    if df_2d_test.isnull().values.any():
        nan_count = df_2d_test.isnull().sum().sum()
        logging.warning(f"Found {nan_count} NaN values in 2D test dataset. Dropping rows with NaNs.")
        df_2d_test.dropna(inplace=True)
        logging.info(f"2D Test shape after dropping NaNs: {df_2d_test.shape}")
    
    # Add location column
    df_2d_train['location'] = df_2d_train['response_point']
    df_2d_test['location'] = df_2d_test['response_point']
    param_features = CONFIG['PARAM_COLS'] + ['location']
    
    # 2D data uses r0, r1, r2... or r_0, r_1, r_2... format for response columns
    def find_time_cols(df):
        time_cols = [col for col in df.columns if col.startswith('r') and (col[1:].isdigit() or (col[2:].isdigit() and col[1] == '_'))]
        return sorted(time_cols, key=lambda x: int(x.split('_')[-1]) if '_' in x else int(x[1:]))
    
    time_cols_2d_train = find_time_cols(df_2d_train)
    time_cols_2d_test = find_time_cols(df_2d_test)
    
    logging.info(f"Found {len(time_cols_2d_train)} time series columns in 2D training data")
    logging.info(f"Found {len(time_cols_2d_test)} time series columns in 2D test data")
    
    if len(time_cols_2d_train) == 0:
        raise ValueError("No time series columns found in 2D training dataset")
    if len(time_cols_2d_test) == 0:
        raise ValueError("No time series columns found in 2D test dataset")
    
    # Take only the first 1500 time steps to match the 1D data structure
    if len(time_cols_2d_train) > CONFIG['NUM_TIME_STEPS']:
        time_cols_2d_train = time_cols_2d_train[:CONFIG['NUM_TIME_STEPS']]
        logging.info(f"Using first {CONFIG['NUM_TIME_STEPS']} time steps from 2D training data")
    if len(time_cols_2d_test) > CONFIG['NUM_TIME_STEPS']:
        time_cols_2d_test = time_cols_2d_test[:CONFIG['NUM_TIME_STEPS']]
        logging.info(f"Using first {CONFIG['NUM_TIME_STEPS']} time steps from 2D test data")
    
    # Extract features and responses from training data
    X_2d_train = df_2d_train[param_features].values
    Y_2d_train = df_2d_train[time_cols_2d_train].values
    
    # Extract features and responses from test data (full test set)
    X_2d_test_full = df_2d_test[param_features].values
    Y_2d_test_full = df_2d_test[time_cols_2d_test].values
    
    # Sample validation set from test data
    logging.info("Sampling validation set from 2D test data...")
    val_size = int(len(X_2d_test_full) * CONFIG['VAL_SPLIT_RATIO'])
    val_indices = np.random.choice(len(X_2d_test_full), size=val_size, replace=False)
    
    X_2d_val = X_2d_test_full[val_indices]
    Y_2d_val = Y_2d_test_full[val_indices]
    
    logging.info(f"2D Split sizes - Train: {len(X_2d_train)}, Validation: {len(X_2d_val)}, Test (full): {len(X_2d_test_full)}")
    logging.info(f"2D Validation ratio from test set: {len(X_2d_val)/len(X_2d_test_full):.1%}")
    
    logging.info(f"2D Data shapes after processing:")
    logging.info(f"  Training: X: {X_2d_train.shape}, Y: {Y_2d_train.shape}")
    logging.info(f"  Validation: X: {X_2d_val.shape}, Y: {Y_2d_val.shape}")
    logging.info(f"  Test (full): X: {X_2d_test_full.shape}, Y: {Y_2d_test_full.shape}")
    
    logging.info(f"2D Training Y stats - Min: {Y_2d_train.min():.6f}, Max: {Y_2d_train.max():.6f}, Mean: {Y_2d_train.mean():.6f}")
    
    return X_2d_train, X_2d_val, X_2d_test_full, Y_2d_train, Y_2d_val, Y_2d_test_full

def train_cae_model(cae, train_loader, val_loader, epochs, learning_rate, model_save_name, load_pretrained=None):
    logging.info(f"--- Starting CAE Training on {CONFIG['DEVICE']} for {epochs} epochs ---")
    logging.info(f"Model will be saved as: {model_save_name}")
    
    if val_loader is None:
        logging.info("No validation monitoring - training for full epochs")
    
    cae.to(CONFIG['DEVICE'])
    
    # Load pretrained weights if specified
    if load_pretrained:
        logging.info(f"Loading pretrained weights from {load_pretrained}")
        cae.load_state_dict(torch.load(load_pretrained, map_location=CONFIG['DEVICE']))
    
    optimizer = optim.Adam(cae.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(CONFIG['OUTPUT_DIR'], model_save_name)
    
    for epoch in range(epochs):
        # Training
        cae.train()
        total_train_loss = 0
        for batch in train_loader:
            params, timeseries = batch['params'].to(CONFIG['DEVICE']), batch['timeseries'].to(CONFIG['DEVICE'])
            
            optimizer.zero_grad()
            recon_ts, _ = cae(timeseries, params)
            loss = criterion(recon_ts, timeseries)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        if val_loader is not None:
            # Validation
            cae.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    params, timeseries = batch['params'].to(CONFIG['DEVICE']), batch['timeseries'].to(CONFIG['DEVICE'])
                    recon_ts, _ = cae(timeseries, params)
                    loss = criterion(recon_ts, timeseries)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(cae.state_dict(), best_model_path)
                logging.info(f"New best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.8f}")
                
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")
        else:
            # No validation - save model every epoch (or just at the end)
            torch.save(cae.state_dict(), best_model_path)
            
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.8f}")
    
    logging.info(f"--- CAE Training Complete. Model saved to {best_model_path} ---")
    return best_model_path

def get_latent_vectors(encoder, dataloader):
    logging.info("Extracting latent vectors using the trained encoder.")
    encoder.to(CONFIG['DEVICE'])
    encoder.eval()
    all_latents = []
    with torch.no_grad():
        for batch in dataloader:
            params, timeseries = batch['params'].to(CONFIG['DEVICE']), batch['timeseries'].to(CONFIG['DEVICE'])
            latents = encoder(timeseries, params)
            all_latents.append(latents.cpu().numpy())
    return np.vstack(all_latents)

def calculate_nmse(y_true, y_pred):
    """
    Calculate Normalized Mean Squared Error (NMSE) as percentage
    NMSE = (1/N) * sum((d_j - d_hat_j)^2 / sigma_j^2) * 100%
    where sigma_j is the standard deviation of true values for each sample
    """
    N = len(y_true)
    nmse_values = []
    
    for i in range(N):
        true_sample = y_true[i]
        pred_sample = y_pred[i]
        sigma_j = np.std(true_sample)
        
        if sigma_j > 0:  # Avoid division by zero
            mse_normalized = np.mean(((true_sample - pred_sample) / sigma_j) ** 2)
            nmse_values.append(mse_normalized)
    
    nmse_percentage = np.mean(nmse_values) * 100
    return nmse_percentage

def evaluate_on_dataset(cae, surrogate_model, params_scaler, X_data, Y_data, dataset_name):
    """Evaluate the model on a given dataset and return metrics"""
    logging.info(f"--- Evaluating on {dataset_name} data ---")
    
    # Create dataset for evaluation
    eval_dataset = BeamResponseDataset(X_data, Y_data, params_scaler)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)
    
    # Get latent vectors from encoder
    Z_true = get_latent_vectors(cae.encoder, eval_loader)
    
    # Predict latent vectors using surrogate model
    Z_pred = surrogate_model.predict(eval_dataset.params_scaled)
    r2_latent = r2_score(Z_true, Z_pred)
    
    # Reconstruct time series from predicted latents
    cae.decoder.to(CONFIG['DEVICE'])
    cae.decoder.eval()
    with torch.no_grad():
        Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(CONFIG['DEVICE'])
        X_tensor = torch.tensor(eval_dataset.params_scaled, dtype=torch.float32).to(CONFIG['DEVICE'])
        Y_pred = cae.decoder(Z_pred_tensor, X_tensor).cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(Y_data.reshape(-1), Y_pred.reshape(-1))
    r2 = r2_score(Y_data.reshape(-1), Y_pred.reshape(-1))
    nmse = calculate_nmse(Y_data, Y_pred)
    
    logging.info(f"{dataset_name} Evaluation Results:")
    logging.info(f"  Latent Space R²: {r2_latent:.4f}")
    logging.info(f"  Time Series R²: {r2:.4f}")
    logging.info(f"  Time Series MSE: {mse:.6f}")
    logging.info(f"  Time Series NMSE: {nmse:.4f}%")
    
    return {
        'r2_latent': r2_latent,
        'r2_timeseries': r2,
        'mse': mse,
        'nmse': nmse,
        'predictions': Y_pred
    }

def evaluate_cross_domain(cae, surrogate_model, params_scaler, X_2d_params, Y_2d_responses, dataset_name):
    """
    Cross-domain evaluation: Use 2D parameters to predict with 1D-trained model
    """
    logging.info(f"--- Cross-Domain Evaluation: 2D Params with 1D-trained model ({dataset_name}) ---")
    
    # Use 2D parameters to get latent vectors via XGBoost (trained on 1D)
    eval_dataset = BeamResponseDataset(X_2d_params, Y_2d_responses, params_scaler)
    
    # Predict latent vectors using 2D parameters with 1D-trained surrogate
    Z_pred = surrogate_model.predict(eval_dataset.params_scaled)
    
    # Reconstruct responses using predicted latents and 2D parameters
    cae.decoder.to(CONFIG['DEVICE'])
    cae.decoder.eval()
    with torch.no_grad():
        Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(CONFIG['DEVICE'])
        X_tensor = torch.tensor(eval_dataset.params_scaled, dtype=torch.float32).to(CONFIG['DEVICE'])
        Y_pred = cae.decoder(Z_pred_tensor, X_tensor).cpu().numpy()
    
    # Calculate metrics comparing predicted responses with 2D ground truth
    mse = mean_squared_error(Y_2d_responses.reshape(-1), Y_pred.reshape(-1))
    r2 = r2_score(Y_2d_responses.reshape(-1), Y_pred.reshape(-1))
    nmse = calculate_nmse(Y_2d_responses, Y_pred)
    
    logging.info(f"{dataset_name} Cross-Domain Results:")
    logging.info(f"  Time Series R² (Cross-Domain): {r2:.4f}")
    logging.info(f"  Time Series MSE: {mse:.6f}")
    logging.info(f"  Time Series NMSE: {nmse:.4f}%")
    
    return {
        'r2_timeseries': r2,
        'mse': mse,
        'nmse': nmse,
        'predictions': Y_pred
    }

# --- New utility: LFSM prediction and dump for arbitrary params matrix ---
def predict_timeseries_from_params(cae, surrogate_model, params_scaled):
    cae.decoder.to(CONFIG['DEVICE'])
    cae.decoder.eval()
    with torch.no_grad():
        Z_pred = surrogate_model.predict(params_scaled)
        Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(CONFIG['DEVICE'])
        X_tensor = torch.tensor(params_scaled, dtype=torch.float32).to(CONFIG['DEVICE'])
        Y_pred = cae.decoder(Z_pred_tensor, X_tensor).cpu().numpy()
    return Y_pred


def dump_lfsm_predictions(file_path, predictions):
    # Use CONFIG['TIME_COLS'] to name columns. Trim in case lengths differ.
    n_steps = predictions.shape[1]
    cols = CONFIG['TIME_COLS'][:n_steps]
    df = pd.DataFrame(predictions, columns=cols)
    df.to_csv(file_path, index=False)
    logging.info(f"Saved LFSM predictions: {file_path} with shape {predictions.shape}")


def dump_lfsm_interleaved_predictions(file_path, ground_truth, predictions, parameters=None):
    """
    Create interleaved CSV file where each ground truth response is followed by its LFSM prediction.
    
    Args:
        file_path: Output CSV file path
        ground_truth: 2D array of ground truth responses (n_samples, n_timesteps)
        predictions: 2D array of LFSM predictions (n_samples, n_timesteps)
        parameters: Optional 2D array of parameters to include as first columns
    """
    n_samples, n_timesteps = ground_truth.shape
    
    # Prepare column names for time series
    time_cols = CONFIG['TIME_COLS'][:n_timesteps]
    
    # Prepare parameter columns if provided
    param_cols = []
    if parameters is not None:
        param_cols = CONFIG['PARAM_COLS'] + ['location']
    
    all_cols = param_cols + time_cols + ['data_type']
    
    # Create interleaved data
    interleaved_data = []
    
    for i in range(n_samples):
        # Ground truth row
        gt_row = []
        if parameters is not None:
            gt_row.extend(parameters[i])
        gt_row.extend(ground_truth[i])
        gt_row.append('ground_truth')
        interleaved_data.append(gt_row)
        
        # LFSM prediction row
        pred_row = []
        if parameters is not None:
            pred_row.extend(parameters[i])
        pred_row.extend(predictions[i])
        pred_row.append('lfsm_prediction')
        interleaved_data.append(pred_row)
    
    # Create DataFrame and save
    df = pd.DataFrame(interleaved_data, columns=all_cols)
    df.to_csv(file_path, index=False)
    
    logging.info(f"Saved interleaved LFSM predictions: {file_path} with shape {df.shape}")
    logging.info(f"Format: {n_samples} parameter sets, each with ground_truth + lfsm_prediction rows")


def create_comparison_plots(ground_truth, predictions, parameters, output_dir, dataset_name="2D"):
    """
    Create comparison plots for the 10 best and 10 worst predictions vs ground truth.
    
    Args:
        ground_truth: 2D array of ground truth responses (n_samples, n_timesteps)
        predictions: 2D array of LFSM predictions (n_samples, n_timesteps)
        parameters: 2D array of parameters (n_samples, n_params)
        output_dir: Directory to save plots
        dataset_name: Name of the dataset for plot titles
    """
    logging.info(f"--- Creating comparison plots for {dataset_name} data ---")
    
    n_samples, n_timesteps = ground_truth.shape
    
    # Calculate R² score for each sample individually
    r2_scores = []
    for i in range(n_samples):
        try:
            r2 = r2_score(ground_truth[i], predictions[i])
            r2_scores.append(r2)
        except:
            # In case of constant ground truth or other issues
            r2_scores.append(-np.inf)
    
    r2_scores = np.array(r2_scores)
    
    # Find indices of 10 best and 10 worst predictions
    best_indices = np.argsort(r2_scores)[-10:][::-1]  # Top 10, descending order
    worst_indices = np.argsort(r2_scores)[:10]  # Bottom 10, ascending order
    
    # Create time axis
    time_axis = np.arange(n_timesteps)
    
    # Plot 10 best predictions
    plt.figure(figsize=(20, 12))
    for i, idx in enumerate(best_indices):
        plt.subplot(2, 5, i+1)
        plt.plot(time_axis, ground_truth[idx], 'b-', label='Ground Truth', linewidth=1.5)
        plt.plot(time_axis, predictions[idx], 'r--', label='LFSM Prediction', linewidth=1.5)
        plt.title(f'Best #{i+1} (R²={r2_scores[idx]:.4f})\nParams: {parameters[idx][:3]}...', fontsize=10)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
    
    plt.suptitle(f'{dataset_name} Data: 10 Best Predictions vs Ground Truth', fontsize=16, y=1.02)
    best_plot_path = os.path.join(output_dir, f'comparison_plots_best_10_{dataset_name.lower()}.png')
    plt.savefig(best_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved best predictions plot: {best_plot_path}")
    
    # Plot 10 worst predictions
    plt.figure(figsize=(20, 12))
    for i, idx in enumerate(worst_indices):
        plt.subplot(2, 5, i+1)
        plt.plot(time_axis, ground_truth[idx], 'b-', label='Ground Truth', linewidth=1.5)
        plt.plot(time_axis, predictions[idx], 'r--', label='LFSM Prediction', linewidth=1.5)
        plt.title(f'Worst #{i+1} (R²={r2_scores[idx]:.4f})\nParams: {parameters[idx][:3]}...', fontsize=10)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
    
    plt.suptitle(f'{dataset_name} Data: 10 Worst Predictions vs Ground Truth', fontsize=16, y=1.02)
    worst_plot_path = os.path.join(output_dir, f'comparison_plots_worst_10_{dataset_name.lower()}.png')
    plt.savefig(worst_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved worst predictions plot: {worst_plot_path}")
    
    # Create summary statistics plot
    plt.figure(figsize=(12, 8))
    
    # Plot R² distribution
    plt.subplot(2, 2, 1)
    plt.hist(r2_scores[r2_scores > -np.inf], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(r2_scores[r2_scores > -np.inf]), color='red', linestyle='--', 
                label=f'Mean R² = {np.mean(r2_scores[r2_scores > -np.inf]):.4f}')
    plt.xlabel('R² Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of R² Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot MSE for best and worst cases
    plt.subplot(2, 2, 2)
    mse_best = [mean_squared_error(ground_truth[idx], predictions[idx]) for idx in best_indices]
    mse_worst = [mean_squared_error(ground_truth[idx], predictions[idx]) for idx in worst_indices]
    
    x = np.arange(10)
    width = 0.35
    plt.bar(x - width/2, mse_best, width, label='Best 10', alpha=0.7, color='green')
    plt.bar(x + width/2, mse_worst, width, label='Worst 10', alpha=0.7, color='red')
    plt.xlabel('Rank')
    plt.ylabel('MSE')
    plt.title('MSE Comparison: Best vs Worst')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot parameter space distribution for best vs worst
    plt.subplot(2, 2, 3)
    # Use first parameter for visualization
    param_best = parameters[best_indices, 0]
    param_worst = parameters[worst_indices, 0]
    plt.scatter(param_best, r2_scores[best_indices], color='green', alpha=0.7, label='Best 10', s=60)
    plt.scatter(param_worst, r2_scores[worst_indices], color='red', alpha=0.7, label='Worst 10', s=60)
    plt.xlabel(f'{CONFIG["PARAM_COLS"][0]}')
    plt.ylabel('R² Score')
    plt.title('Performance vs Parameter Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot average response comparison
    plt.subplot(2, 2, 4)
    avg_gt_best = np.mean(ground_truth[best_indices], axis=0)
    avg_pred_best = np.mean(predictions[best_indices], axis=0)
    avg_gt_worst = np.mean(ground_truth[worst_indices], axis=0)
    avg_pred_worst = np.mean(predictions[worst_indices], axis=0)
    
    plt.plot(time_axis, avg_gt_best, 'g-', label='GT Best Avg', linewidth=2)
    plt.plot(time_axis, avg_pred_best, 'g--', label='Pred Best Avg', linewidth=2)
    plt.plot(time_axis, avg_gt_worst, 'r-', label='GT Worst Avg', linewidth=2)
    plt.plot(time_axis, avg_pred_worst, 'r--', label='Pred Worst Avg', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Response')
    plt.title('Average Response: Best vs Worst')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, f'comparison_summary_{dataset_name.lower()}.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved summary plot: {summary_plot_path}")
    
    # Log statistics
    logging.info(f"{dataset_name} Comparison Statistics:")
    logging.info(f"  Best R² scores: {r2_scores[best_indices]}")
    logging.info(f"  Worst R² scores: {r2_scores[worst_indices]}")
    logging.info(f"  Mean R² (all): {np.mean(r2_scores[r2_scores > -np.inf]):.4f}")
    logging.info(f"  Std R² (all): {np.std(r2_scores[r2_scores > -np.inf]):.4f}")
    
    return {
        'best_indices': best_indices,
        'worst_indices': worst_indices,
        'r2_scores': r2_scores,
        'best_r2': r2_scores[best_indices],
        'worst_r2': r2_scores[worst_indices]
    }


def create_individual_comparison_plots(ground_truth, predictions, parameters, output_dir, dataset_name="2D", max_plots=None):
    """
    Create individual comparison plots for each test sample.
    
    Args:
        ground_truth: 2D array of ground truth responses (n_samples, n_timesteps)
        predictions: 2D array of LFSM predictions (n_samples, n_timesteps)
        parameters: 2D array of parameters (n_samples, n_params)
        output_dir: Directory to save plots
        dataset_name: Name of the dataset for plot titles
        max_plots: Maximum number of individual plots to save (None for all)
    """
    logging.info(f"--- Creating individual comparison plots for {dataset_name} data ---")
    
    n_samples, n_timesteps = ground_truth.shape
    
    # Calculate R² score for each sample individually
    r2_scores = []
    for i in range(n_samples):
        try:
            r2 = r2_score(ground_truth[i], predictions[i])
            r2_scores.append(r2)
        except:
            # In case of constant ground truth or other issues
            r2_scores.append(-np.inf)
    
    r2_scores = np.array(r2_scores)
    
    # Create individual plots directory
    individual_plots_dir = os.path.join(output_dir, f'individual_plots_{dataset_name.lower()}')
    os.makedirs(individual_plots_dir, exist_ok=True)
    
    # Determine how many plots to create
    if max_plots is None:
        num_plots = n_samples
    else:
        num_plots = min(max_plots, n_samples)
    
    # Create time axis
    time_axis = np.arange(n_timesteps)
    
    # Create individual plots
    logging.info(f"Creating {num_plots} individual comparison plots...")
    
    for i in range(num_plots):
        plt.figure(figsize=(10, 6))
        
        # Plot ground truth and prediction
        plt.plot(time_axis, ground_truth[i], 'b-', label='Ground Truth', linewidth=2)
        plt.plot(time_axis, predictions[i], 'r--', label='LFSM Prediction', linewidth=2)
        
        # Calculate additional metrics for this sample
        mse = mean_squared_error(ground_truth[i], predictions[i])
        nmse = calculate_nmse(ground_truth[i:i+1], predictions[i:i+1])
        
        # Create title with sample info
        param_str = ', '.join([f'{CONFIG["PARAM_COLS"][j]}: {parameters[i][j]:.3f}' for j in range(min(3, len(CONFIG["PARAM_COLS"])))])
        if len(CONFIG["PARAM_COLS"]) > 3:
            param_str += '...'
        
        plt.title(f'Sample {i+1} - R²: {r2_scores[i]:.4f}, MSE: {mse:.6f}, NMSE: {nmse:.2f}%\n{param_str}', 
                 fontsize=12, pad=20)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Response', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Add text box with all parameters
        param_text = '\n'.join([f'{CONFIG["PARAM_COLS"][j]}: {parameters[i][j]:.4f}' 
                               for j in range(len(CONFIG["PARAM_COLS"]))])
        param_text += f'\nlocation: {parameters[i][-1]:.4f}'
        
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual plot
        plot_path = os.path.join(individual_plots_dir, f'individual_plot_sample_{i+1:04d}_r2_{r2_scores[i]:.4f}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log progress every 100 plots
        if (i + 1) % 100 == 0:
            logging.info(f"Created {i + 1}/{num_plots} individual plots...")
    
    logging.info(f"Saved {num_plots} individual comparison plots in: {individual_plots_dir}")
    
    # Create a summary CSV file with all sample statistics
    summary_data = []
    for i in range(n_samples):
        mse = mean_squared_error(ground_truth[i], predictions[i])
        nmse = calculate_nmse(ground_truth[i:i+1], predictions[i:i+1])
        
        sample_data = {
            'sample_id': i + 1,
            'r2_score': r2_scores[i],
            'mse': mse,
            'nmse': nmse
        }
        
        # Add parameter values
        for j, param_name in enumerate(CONFIG["PARAM_COLS"]):
            sample_data[param_name] = parameters[i][j]
        sample_data['location'] = parameters[i][-1]
        
        summary_data.append(sample_data)
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, f'individual_plots_summary_{dataset_name.lower()}.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Saved summary CSV: {summary_csv_path}")
    
    return {
        'individual_plots_dir': individual_plots_dir,
        'summary_csv_path': summary_csv_path,
        'num_plots_created': num_plots,
        'r2_scores': r2_scores
    }


def main():
    logging.info("=== STARTING SEQUENTIAL TRAINING: 1D -> 2D ===")
    
    # ===== PHASE 1: 1D TRAINING =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 1: TRAINING ON 1D DATA")
    logging.info("="*60)
    
    # 1. Load 1D Data (train from train CSV, val sampled from test CSV, test is full test CSV)
    X_train_1d, X_val_1d, X_test_1d, Y_train_1d, Y_val_1d, Y_test_1d = load_1d_data()
    
    # 2. Create 1D Datasets and Dataloaders
    train_dataset_1d = BeamResponseDataset(X_train_1d, Y_train_1d)
    val_dataset_1d = BeamResponseDataset(X_val_1d, Y_val_1d, train_dataset_1d.p_scaler)
    test_dataset_1d = BeamResponseDataset(X_test_1d, Y_test_1d, train_dataset_1d.p_scaler)
    
    train_loader_1d = DataLoader(train_dataset_1d, batch_size=CONFIG['CAE_BATCH_SIZE'], shuffle=True, drop_last=True)
    val_loader_1d = DataLoader(val_dataset_1d, batch_size=CONFIG['CAE_BATCH_SIZE'], shuffle=False)
    train_loader_1d_full = DataLoader(train_dataset_1d, batch_size=len(train_dataset_1d), shuffle=False)
    test_loader_1d_full = DataLoader(test_dataset_1d, batch_size=len(test_dataset_1d), shuffle=False)
    
    # 3. Initialize and Train CAE on 1D data
    cae = ConditionalAutoencoder(
        timeseries_dim=CONFIG['NUM_TIME_STEPS'],
        params_dim=len(CONFIG['PARAM_COLS']) + 1,
        latent_dim=CONFIG['LATENT_DIM']
    )
    best_model_1d_path = train_cae_model(
        cae, train_loader_1d, val_loader_1d, 
        CONFIG['CAE_EPOCHS_1D'], CONFIG['CAE_LEARNING_RATE'], 
        'best_cae_model_1d.pth'
    )
    
    # Load best 1D model
    logging.info(f"Loading best 1D model from {best_model_1d_path}")
    cae.load_state_dict(torch.load(best_model_1d_path, map_location=CONFIG['DEVICE']))
    
    # 4. Extract Latent Vectors for 1D data
    Z_train_1d = get_latent_vectors(cae.encoder, train_loader_1d_full)
    Z_test_1d = get_latent_vectors(cae.encoder, test_loader_1d_full)
    Z_val_1d = get_latent_vectors(cae.encoder, val_loader_1d)
    logging.info(f"Extracted 1D latent vectors. Train shape: {Z_train_1d.shape}, Val shape: {Z_val_1d.shape}, Test shape: {Z_test_1d.shape}")
    
    # 5. Train XGBoost Surrogate Model on 1D data with early stopping
    logging.info("--- Training XGBoost Surrogate Model on 1D Data (with early stopping) ---")
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': CONFIG['XGB_N_ESTIMATORS'],
        'max_depth': CONFIG['XGB_MAX_DEPTH'],
        'eta': CONFIG['XGB_ETA'],
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'sampling_method': 'gradient_based',
        'verbosity': 0,
    }
    if CONFIG['USE_XGB_GPU'] and CONFIG['DEVICE'] == 'cuda':
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['predictor'] = 'gpu_predictor'
        xgb_params['gpu_id'] = 0
        logging.info("Using GPU for XGBoost (gpu_hist).")
    else:
        xgb_params['tree_method'] = 'hist'
        xgb_params['predictor'] = 'cpu_predictor'

    # create regressor with early stopping
    surrogate_model_1d = xgb.XGBRegressor(**xgb_params, n_jobs=-1, early_stopping_rounds=50, eval_metric='rmse')
    # fit with early stopping on validation latent vectors
    surrogate_model_1d.fit(
        train_dataset_1d.params_scaled, Z_train_1d,
        eval_set=[(val_dataset_1d.params_scaled, Z_val_1d)],
        verbose=False
    )
    logging.info("--- 1D Surrogate Model Training Complete ---")
    
    # 6. Evaluate 1D model on 1D test data
    logging.info("\n" + "="*60)
    logging.info("PHASE 1 EVALUATION: 1D MODEL ON 1D TEST DATA")
    logging.info("="*60)
    
    results_1d_test = evaluate_on_dataset(cae, surrogate_model_1d, train_dataset_1d.p_scaler, 
                                         X_test_1d, Y_test_1d, "1D Test")
    
    # Save 1D model and scaler
    joblib.dump(surrogate_model_1d, os.path.join(CONFIG['OUTPUT_DIR'], 'surrogate_model_1d.joblib'))
    joblib.dump(train_dataset_1d.p_scaler, os.path.join(CONFIG['OUTPUT_DIR'], 'params_scaler_1d.joblib'))
    
    # ===== PHASE 2: 2D DATA PREPARATION AND CROSS-DOMAIN EVALUATION =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 2: LOADING 2D DATA AND CROSS-DOMAIN EVALUATION")
    logging.info("="*60)
    
    # 1. Load 2D Data (train from train CSV, val sampled from test CSV, test is full test CSV)
    X_train_2d, X_val_2d, X_test_2d, Y_train_2d, Y_val_2d, Y_test_2d = load_2d_data()
    
    # 2. Cross-domain evaluation: Use 1D-trained model on 2D test data
    results_cross_domain = evaluate_cross_domain(cae, surrogate_model_1d, train_dataset_1d.p_scaler, 
                                               X_test_2d, Y_test_2d, "Cross-Domain (1D Model on 2D Data)")
    
    # ===== PHASE 3: FINE-TUNING ON 2D DATA =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 3: FINE-TUNING ON 2D DATA")
    logging.info("="*60)
    
    # 1. Create 2D Datasets using the same scaler from 1D training
    train_dataset_2d = BeamResponseDataset(X_train_2d, Y_train_2d, train_dataset_1d.p_scaler)
    val_dataset_2d = BeamResponseDataset(X_val_2d, Y_val_2d, train_dataset_1d.p_scaler)
    test_dataset_2d = BeamResponseDataset(X_test_2d, Y_test_2d, train_dataset_1d.p_scaler)
    
    # 2. Create weighted dataset by combining 1D and 2D data with higher weight for 2D
    logging.info("--- Creating weighted combined dataset (2D data gets higher weight) ---")

    # Use config-based repetition - increased weight for 2D data
    weight_multiplier = int(6)  # Increased from 4 to 6 for better 2D representation
    X_combined = np.concatenate([X_train_1d] + [X_train_2d] * weight_multiplier)
    Y_combined = np.concatenate([Y_train_1d] + [Y_train_2d] * weight_multiplier)

    # Create combined validation dataset (1D val + 2D val with same weighting)
    X_val_combined = np.concatenate([X_val_1d] + [X_val_2d] * weight_multiplier)
    Y_val_combined = np.concatenate([Y_val_1d] + [Y_val_2d] * weight_multiplier)
    
    logging.info(f"Combined training data shape: X: {X_combined.shape}, Y: {Y_combined.shape}")
    logging.info(f"Combined validation data shape: X: {X_val_combined.shape}, Y: {Y_val_combined.shape}")
    
    train_dataset_combined = BeamResponseDataset(X_combined, Y_combined, train_dataset_1d.p_scaler)
    val_dataset_combined = BeamResponseDataset(X_val_combined, Y_val_combined, train_dataset_1d.p_scaler)
    
    train_loader_combined = DataLoader(train_dataset_combined, batch_size=CONFIG['CAE_BATCH_SIZE'], shuffle=True, drop_last=True)
    val_loader_combined = DataLoader(val_dataset_combined, batch_size=CONFIG['CAE_BATCH_SIZE'], shuffle=False)
    
    # 3. Fine-tune the model on combined data
    best_model_2d_path = train_cae_model(
        cae, train_loader_combined, val_loader_combined,
        CONFIG['CAE_EPOCHS_2D'], CONFIG['CAE_LEARNING_RATE_2D'],
        'best_cae_model_2d_finetuned.pth', 
        load_pretrained=best_model_1d_path
    )
    
    # Load best fine-tuned model
    logging.info(f"Loading best fine-tuned model from {best_model_2d_path}")
    cae.load_state_dict(torch.load(best_model_2d_path, map_location=CONFIG['DEVICE']))
    
    # ===== PHASE 4: TRAIN NEW SURROGATE ON COMBINED DATA =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 4: TRAINING NEW SURROGATE ON COMBINED DATA")
    logging.info("="*60)
    
    # Extract latent vectors from fine-tuned model using combined training data
    train_loader_combined_full = DataLoader(train_dataset_combined, batch_size=len(train_dataset_combined), shuffle=False)
    val_loader_combined_full = DataLoader(val_dataset_combined, batch_size=len(val_dataset_combined), shuffle=False)
    Z_train_combined = get_latent_vectors(cae.encoder, train_loader_combined_full)
    Z_val_combined = get_latent_vectors(cae.encoder, val_loader_combined_full)
    
    logging.info(f"Extracted combined latent vectors. Shape: {Z_train_combined.shape}, Val shape: {Z_val_combined.shape}")
    
    # Train new XGBoost on combined latent vectors with early stopping
    logging.info("--- Training XGBoost Surrogate Model on Combined Data (with early stopping) ---")
    surrogate_model_combined = xgb.XGBRegressor(**xgb_params, n_jobs=-1, early_stopping_rounds=50, eval_metric='rmse')
    surrogate_model_combined.fit(
        train_dataset_combined.params_scaled, Z_train_combined,
        eval_set=[(val_dataset_combined.params_scaled, Z_val_combined)],
        verbose=False
    )
    logging.info("--- Combined Surrogate Model Training Complete ---")
    
    # ===== PHASE 5: FINAL EVALUATION =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 5: FINAL COMPREHENSIVE EVALUATION")
    logging.info("="*60)
    
    # Evaluate fine-tuned model on 1D test data
    results_1d_finetuned = evaluate_on_dataset(cae, surrogate_model_combined, train_dataset_1d.p_scaler, 
                                             X_test_1d, Y_test_1d, "1D Test (Fine-tuned Model)")
    
    # Evaluate fine-tuned model on 2D test data
    results_2d_finetuned = evaluate_on_dataset(cae, surrogate_model_combined, train_dataset_1d.p_scaler, 
                                             X_test_2d, Y_test_2d, "2D Test (Fine-tuned Model)")

    # ===== MODIFIED: DUMP INTERLEAVED LFSM PREDICTIONS FOR 2D TRAIN AND 2D TEST =====
    logging.info("--- Dumping INTERLEAVED LFSM predictions for 2D train and 2D test ---")
    
    # Build 2D train and test datasets to reuse the scaler
    train_dataset_2d_for_dump = BeamResponseDataset(X_train_2d, Y_train_2d, train_dataset_1d.p_scaler)
    test_dataset_2d_for_dump = BeamResponseDataset(X_test_2d, Y_test_2d, train_dataset_1d.p_scaler)

    Y_pred_2d_train = predict_timeseries_from_params(cae, surrogate_model_combined, train_dataset_2d_for_dump.params_scaled)
    Y_pred_2d_test = predict_timeseries_from_params(cae, surrogate_model_combined, test_dataset_2d_for_dump.params_scaled)

    # Create interleaved CSV files with parameters
    dump_lfsm_interleaved_predictions(
        os.path.join(CONFIG['OUTPUT_DIR'], 'lfsm_interleaved_2d_train.csv'), 
        Y_train_2d, Y_pred_2d_train, X_train_2d
    )
    dump_lfsm_interleaved_predictions(
        os.path.join(CONFIG['OUTPUT_DIR'], 'lfsm_interleaved_2d_test.csv'), 
        Y_test_2d, Y_pred_2d_test, X_test_2d
    )

    # Keep the original separate files as backup
    dump_lfsm_predictions(os.path.join(CONFIG['OUTPUT_DIR'], 'lfsm_predictions_2d_train.csv'), Y_pred_2d_train)
    dump_lfsm_predictions(os.path.join(CONFIG['OUTPUT_DIR'], 'lfsm_predictions_2d_test.csv'), Y_pred_2d_test)

    # Also optionally save as NumPy for exact recovery
    np.save(os.path.join(CONFIG['OUTPUT_DIR'], 'lfsm_predictions_2d_train.npy'), Y_pred_2d_train)
    np.save(os.path.join(CONFIG['OUTPUT_DIR'], 'lfsm_predictions_2d_test.npy'), Y_pred_2d_test)
    
    # ===== CREATE COMPARISON PLOTS =====
    logging.info("\n" + "="*60)
    logging.info("CREATING COMPARISON PLOTS")
    logging.info("="*60)
    
    # Create comparison plots for 2D test data (most important)
    plot_stats_2d_test = create_comparison_plots(
        Y_test_2d, Y_pred_2d_test, X_test_2d, 
        CONFIG['OUTPUT_DIR'], dataset_name="2D_Test"
    )
    
    # Create comparison plots for 2D training data
    plot_stats_2d_train = create_comparison_plots(
        Y_train_2d, Y_pred_2d_train, X_train_2d, 
        CONFIG['OUTPUT_DIR'], dataset_name="2D_Train"
    )
    
    # ===== CREATE INDIVIDUAL COMPARISON PLOTS =====
    logging.info("\n" + "="*60)
    logging.info("CREATING INDIVIDUAL COMPARISON PLOTS")
    logging.info("="*60)
    
    # Create individual plots for 2D test data (most important)
    logging.info("Creating individual plots for 2D test data...")
    individual_stats_2d_test = create_individual_comparison_plots(
        Y_test_2d, Y_pred_2d_test, X_test_2d, 
        CONFIG['OUTPUT_DIR'], dataset_name="2D_Test"
    )
    
    # Create individual plots for 2D training data
    logging.info("Creating individual plots for 2D training data...")
    individual_stats_2d_train = create_individual_comparison_plots(
        Y_train_2d, Y_pred_2d_train, X_train_2d, 
        CONFIG['OUTPUT_DIR'], dataset_name="2D_Train"
    )
    
    # ===== PHASE 6: RESULTS SUMMARY AND VISUALIZATION =====
    logging.info("\n" + "="*60)
    logging.info("FINAL SUMMARY OF ALL RESULTS")
    logging.info("="*60)
    
    logging.info(f"1D Model on 1D Test - R² (Time Series): {results_1d_test['r2_timeseries']:.4f}, NMSE: {results_1d_test['nmse']:.4f}%")
    logging.info(f"1D Model on 2D Test (Cross-Domain) - R² (Time Series): {results_cross_domain['r2_timeseries']:.4f}, NMSE: {results_cross_domain['nmse']:.4f}%")
    logging.info(f"Fine-tuned Model on 1D Test - R² (Time Series): {results_1d_finetuned['r2_timeseries']:.4f}, NMSE: {results_1d_finetuned['nmse']:.4f}%")
    logging.info(f"Fine-tuned Model on 2D Test - R² (Time Series): {results_2d_finetuned['r2_timeseries']:.4f}, NMSE: {results_2d_finetuned['nmse']:.4f}%")
    
    # Summary of comparison plots
    logging.info("\n" + "="*30 + " COMPARISON PLOTS SUMMARY " + "="*30)
    logging.info(f"2D Test Data - Best R² range: {plot_stats_2d_test['best_r2'].min():.4f} to {plot_stats_2d_test['best_r2'].max():.4f}")
    logging.info(f"2D Test Data - Worst R² range: {plot_stats_2d_test['worst_r2'].min():.4f} to {plot_stats_2d_test['worst_r2'].max():.4f}")
    logging.info(f"2D Train Data - Best R² range: {plot_stats_2d_train['best_r2'].min():.4f} to {plot_stats_2d_train['best_r2'].max():.4f}")
    logging.info(f"2D Train Data - Worst R² range: {plot_stats_2d_train['worst_r2'].min():.4f} to {plot_stats_2d_train['worst_r2'].max():.4f}")
    logging.info("Comparison plots saved:")
    logging.info("  - comparison_plots_best_10_2d_test.png")
    logging.info("  - comparison_plots_worst_10_2d_test.png")
    logging.info("  - comparison_summary_2d_test.png")
    logging.info("  - comparison_plots_best_10_2d_train.png")
    logging.info("  - comparison_plots_worst_10_2d_train.png")
    logging.info("  - comparison_summary_2d_train.png")
    
    # Summary of individual plots
    logging.info("\n" + "="*30 + " INDIVIDUAL PLOTS SUMMARY " + "="*30)
    logging.info(f"2D Test Data - Individual plots created: {individual_stats_2d_test['num_plots_created']}")
    logging.info(f"2D Test Data - Individual plots directory: {individual_stats_2d_test['individual_plots_dir']}")
    logging.info(f"2D Test Data - Summary CSV: {individual_stats_2d_test['summary_csv_path']}")
    logging.info(f"2D Train Data - Individual plots created: {individual_stats_2d_train['num_plots_created']}")
    logging.info(f"2D Train Data - Individual plots directory: {individual_stats_2d_train['individual_plots_dir']}")
    logging.info(f"2D Train Data - Summary CSV: {individual_stats_2d_train['summary_csv_path']}")
    logging.info("Individual plots saved in subdirectories:")
    logging.info("  - individual_plots_2d_test/ (contains all test sample plots)")
    logging.info("  - individual_plots_2d_train/ (contains all train sample plots)")
    logging.info("  - individual_plots_summary_2d_test.csv (test sample statistics)")
    logging.info("  - individual_plots_summary_2d_train.csv (train sample statistics)")
    
    # Save final models and scalers
    logging.info("--- Saving final models and results ---")
    joblib.dump(surrogate_model_combined, os.path.join(CONFIG['OUTPUT_DIR'], 'surrogate_model_final.joblib'))
    joblib.dump(train_dataset_1d.p_scaler, os.path.join(CONFIG['OUTPUT_DIR'], 'params_scaler_final.joblib'))
    
    # Rest of the visualization code remains the same...
    logging.info("--- Sequential Training and Evaluation Complete ---")
    logging.info("="*60)

if __name__ == '__main__':
    main()
