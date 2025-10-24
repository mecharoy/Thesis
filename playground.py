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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans

# Import additional metrics functions
def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_cosine_similarity(y_true, y_pred):
    """Calculate cosine similarity between true and predicted time series"""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    norm_true = np.linalg.norm(y_true_flat)
    norm_pred = np.linalg.norm(y_pred_flat)

    if norm_true == 0 or norm_pred == 0:
        return 0.0

    cosine_sim = np.dot(y_true_flat, y_pred_flat) / (norm_true * norm_pred)
    return cosine_sim

def evaluate_ae_reconstruction(ae, dataloader, dataset_name):
    """Evaluate pure autoencoder reconstruction quality (encoder → decoder)"""
    logging.info(f"--- Evaluating AE Reconstruction on {dataset_name} ---")

    ae.to(CONFIG['DEVICE'])
    ae.eval()

    all_true = []
    all_reconstructed = []

    with torch.no_grad():
        for batch in dataloader:
            timeseries = batch['timeseries'].to(CONFIG['DEVICE'])
            recon_ts, _ = ae(timeseries)

            all_true.append(timeseries.cpu().numpy())
            all_reconstructed.append(recon_ts.cpu().numpy())

    Y_true = np.vstack(all_true)
    Y_recon = np.vstack(all_reconstructed)

    # Calculate only R² metrics
    r2_overall = r2_score(Y_true.reshape(-1), Y_recon.reshape(-1))

    # Per-sample R² for distribution analysis
    sample_r2_scores = []
    for i in range(len(Y_true)):
        try:
            r2_sample = r2_score(Y_true[i], Y_recon[i])
            sample_r2_scores.append(r2_sample)
        except:
            sample_r2_scores.append(-np.inf)

    sample_r2_scores = np.array(sample_r2_scores)
    mean_sample_r2 = np.mean(sample_r2_scores)
    std_sample_r2 = np.std(sample_r2_scores)
    median_sample_r2 = np.median(sample_r2_scores)

    # Log only overall R² metric
    logging.info(f"AE Reconstruction - {dataset_name}: R²={r2_overall:.6f}")

    return {
        'r2_overall': r2_overall,
        'sample_r2_mean': mean_sample_r2,
        'sample_r2_std': std_sample_r2,
        'sample_r2_median': median_sample_r2,
        'reconstructed': Y_recon,
        'sample_r2_scores': sample_r2_scores
    }

# --- Configuration Dictionary ---
CONFIG = {
    # --- GPU/CPU Settings ---
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'USE_XGB_GPU': True,

    # --- File Paths & Dirs ---
    'LFSM_TRAIN_FILE': '/home/user2/Music/abhi3/parameters/LFSM2000train.csv',
    'LFSM_TEST_FILE': '/home/user2/Music/abhi3/parameters/LFSM2000test.csv',
    'HFSM_TRAIN_FILE': '/home/user2/Music/abhi3/parameters/train_responseslatest.csv',
    'HFSM_TEST_FILE': '/home/user2/Music/abhi3/parameters/test_responseslatest.csv',
    'OUTPUT_DIR': '/home/user2/Music/abhi3/test',

    # --- Data & Model Hyperparameters ---
    'PARAM_COLS': ['notch_x', 'notch_depth', 'notch_width', 'length', 'density', 'youngs_modulus'],
    'NUM_TIME_STEPS': 1500,

    # --- PHASE 1: LFSM Pre-training ---
    'LFSM_LATENT_DIM': 30,
    'LFSM_CAE_EPOCHS': 200,
    'LFSM_CAE_BATCH_SIZE': 64,
    'LFSM_LEARNING_RATE': 1e-4,

    # --- PHASE 2: MFSM Fine-tuning on 2D Data ---
    'MFSM_CAE_EPOCHS': 100,
    'MFSM_CAE_BATCH_SIZE': 32,
    'MFSM_LEARNING_RATE': 1e-5,  # Increased from 1e-5 to 5e-5 (5x higher)
    'MFSM_LOSS_WEIGHT': 3.0,     # Multiplier for 2D training loss (3x more weightage)
    'NUM_MFSM_TRAIN_SAMPLES': None,  # Use ALL HFSM samples for better parameter-latent mapping (was 660)

    # --- Model Loading/Training Control ---
    'USE_EXISTING_MODELS': False,  # Set to True to use pre-saved AutoEncoder models, False to train from scratch

    # --- XGBoost Surrogate Model ---
    'XGB_N_ESTIMATORS': 2000,
    'XGB_MAX_DEPTH': 10,
    'XGB_ETA': 0.02,
    'XGB_EARLY_STOPPING': 100,
}

# --- Setup Logging ---
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_training_log.log')),
                              logging.StreamHandler()])

# --- Helper Function: ROI Calculation ---
def get_roi_for_location(location):
    """
    Calculate Region of Interest (ROI) timestep range based on response location.

    ROI captures the dynamic response region, excluding quiescent zones with zeros
    or constant baseline that inflate R² scores.

    Base reference (location=1.85): ROI = [190, 900]
    ROI_start shifts by +20 timesteps per 0.02 location increment
    ROI_end shifts by +100 timesteps per 0.02 location increment
    """
    delta = location - 1.85
    roi_start = int(190 + (delta / 0.02) * 20)
    roi_end = int(900 + (delta / 0.02) * 100)

    # Clamp to valid timestep range
    roi_start = max(0, min(roi_start, 1500))
    roi_end = max(0, min(roi_end, 1500))

    return roi_start, roi_end

# --- PyTorch Dataset Class ---
class BeamResponseDataset(Dataset):
    """
    Dataset for beam response data with parameter scaling to [-1, 1].
    Handles both 1D (LFSM - time series data starting from t_1)
    and 2D (HFSM - time series data starting from r0) formats.
    """
    def __init__(self, params, timeseries, p_scaler=None, add_noise=False, noise_std=0.03):
        # Store raw time series without scaling - responses are already normalized
        self.timeseries = timeseries.astype(np.float32).copy()
        self.params = params.astype(np.float32)

        # Add Gaussian noise to training data to prevent overfitting
        if add_noise and noise_std > 0:
            noise = np.random.normal(0, noise_std, self.timeseries.shape).astype(np.float32)
            self.timeseries += noise
            logging.info(f"Added Gaussian noise (std={noise_std}) to training data for regularization")

        # Scale parameters to [-1, 1] range
        if p_scaler is None:
            self.p_scaler = MinMaxScaler(feature_range=(-1, 1))
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

# --- Autoencoder PyTorch Models (Identical to HFSM.py) ---
class Encoder(nn.Module):
    """Encoder: Time series → Latent space (NO parameter conditioning)"""
    def __init__(self, timeseries_dim, latent_dim):
        super(Encoder, self).__init__()
        self.timeseries_net = nn.Sequential(
            nn.Linear(timeseries_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        """Forward pass - only takes time series, no parameters"""
        return self.timeseries_net(x)

class Decoder(nn.Module):
    """Decoder: Latent space → Time series (NO parameter conditioning)"""
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.expansion = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim)
        )

    def forward(self, z):
        """Forward pass - only takes latent vector, no parameters"""
        return self.expansion(z)

class Autoencoder(nn.Module):
    """Non-conditional Autoencoder"""
    def __init__(self, timeseries_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(timeseries_dim, latent_dim)
        self.decoder = Decoder(latent_dim, timeseries_dim)

    def forward(self, x):
        """Forward pass - only takes time series, no parameters"""
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z

# --- Data Loading Functions ---
def load_lfsm_data():
    """
    Load LFSM (1D zigzag) training and test data.
    Data format: case_id, response_point, params..., t_1, t_2, ..., t_1500
    """
    logging.info("=== LOADING LFSM (1D) DATA ===")

    # Load LFSM training data
    logging.info(f"Loading LFSM training data from {CONFIG['LFSM_TRAIN_FILE']}")
    df_lfsm_train = pd.read_csv(CONFIG['LFSM_TRAIN_FILE'])
    logging.info(f"LFSM training samples: {len(df_lfsm_train)}")

    # Load LFSM test data (will use as validation set for LFSM pre-training)
    logging.info(f"Loading LFSM test data from {CONFIG['LFSM_TEST_FILE']}")
    df_lfsm_test = pd.read_csv(CONFIG['LFSM_TEST_FILE'])
    logging.info(f"LFSM test samples: {len(df_lfsm_test)}")

    # Drop NaNs from both datasets
    for df, name in [(df_lfsm_train, 'train'), (df_lfsm_test, 'test')]:
        if df.isnull().values.any():
            nan_count = df.isnull().sum().sum()
            logging.warning(f"Found {nan_count} NaN values in LFSM {name} dataset. Dropping rows with NaNs.")
            df.dropna(inplace=True)

    # Add location column to both datasets
    df_lfsm_train['location'] = df_lfsm_train['response_point']
    df_lfsm_test['location'] = df_lfsm_test['response_point']
    param_features = CONFIG['PARAM_COLS'] + ['location']

    # Detect time columns (t_1, t_2, ..., t_1500)
    time_cols = [col for col in df_lfsm_train.columns if col.startswith('t_')]
    logging.info(f"Detected {len(time_cols)} time columns in LFSM data")

    # Extract parameters and responses for train and test
    X_lfsm_train = df_lfsm_train[param_features].values
    Y_lfsm_train = df_lfsm_train[time_cols].values
    X_lfsm_test = df_lfsm_test[param_features].values
    Y_lfsm_test = df_lfsm_test[time_cols].values

    logging.info(f"LFSM train data shape: X: {X_lfsm_train.shape}, Y: {Y_lfsm_train.shape}")
    logging.info(f"LFSM test data shape: X: {X_lfsm_test.shape}, Y: {Y_lfsm_test.shape}")
    logging.info(f"LFSM train Y stats - Min: {Y_lfsm_train.min():.6f}, Max: {Y_lfsm_train.max():.6f}, Mean: {Y_lfsm_train.mean():.6f}")
    logging.info(f"LFSM test Y stats - Min: {Y_lfsm_test.min():.6f}, Max: {Y_lfsm_test.max():.6f}, Mean: {Y_lfsm_test.mean():.6f}")

    return X_lfsm_train, Y_lfsm_train, X_lfsm_test, Y_lfsm_test

def detect_time_columns(df, dataset_name=""):
    """
    Robustly detect time columns that handle both formats:
    - r0, r1, r2, ..., r1499 (no underscore)
    - r_0, r_1, r_2, ..., r_1499 (with underscore)
    
    Args:
        df: DataFrame to detect time columns from
        dataset_name: Name of dataset for logging
    
    Returns:
        List of time column names sorted numerically
    """
    time_cols = []
    for col in df.columns:
        if col.startswith('r') and len(col) > 1:
            # Handle both formats: r0, r1, ... and r_0, r_1, ...
            if col[1:].isdigit():  # Format: r0, r1, ...
                time_cols.append(col)
            elif col[1] == '_' and col[2:].isdigit():  # Format: r_0, r_1, ...
                time_cols.append(col)
    
    # Sort numerically
    def extract_number(col):
        if col[1] == '_':
            return int(col[2:])
        else:
            return int(col[1:])
    
    time_cols = sorted(time_cols, key=extract_number)
    
    if dataset_name:
        logging.info(f"Detected {len(time_cols)} time columns in {dataset_name} data")
        if len(time_cols) > 0:
            logging.info(f"Format: {time_cols[0]} ... {time_cols[-1]}")
    
    return time_cols

def load_hfsm_data():
    """
    Load HFSM (2D FEM) training and test data.
    Data format: params..., r0, r1, r2, ..., r1499, response_point
    """
    logging.info("=== LOADING HFSM (2D FEM) DATA ===")

    # Load HFSM training data
    logging.info(f"Loading HFSM training data from {CONFIG['HFSM_TRAIN_FILE']}")
    df_hfsm_train = pd.read_csv(CONFIG['HFSM_TRAIN_FILE'])
    logging.info(f"HFSM training samples: {len(df_hfsm_train)}")

    # Load HFSM test data
    logging.info(f"Loading HFSM test data from {CONFIG['HFSM_TEST_FILE']}")
    df_hfsm_test = pd.read_csv(CONFIG['HFSM_TEST_FILE'])
    logging.info(f"HFSM test samples: {len(df_hfsm_test)}")

    # Drop NaNs
    for df, name in [(df_hfsm_train, 'train'), (df_hfsm_test, 'test')]:
        if df.isnull().values.any():
            nan_count = df.isnull().sum().sum()
            logging.warning(f"Found {nan_count} NaN values in HFSM {name} dataset. Dropping rows with NaNs.")
            df.dropna(inplace=True)

    # Add location column
    df_hfsm_train['location'] = df_hfsm_train['response_point']
    df_hfsm_test['location'] = df_hfsm_test['response_point']
    param_features = CONFIG['PARAM_COLS'] + ['location']

    # Detect time columns for both datasets
    time_cols_train = detect_time_columns(df_hfsm_train, "HFSM training")
    time_cols_test = detect_time_columns(df_hfsm_test, "HFSM test")
    
    # Ensure both datasets have the same time columns
    if time_cols_train != time_cols_test:
        logging.warning(f"Time column formats differ between train and test datasets!")
        logging.warning(f"Train format: {time_cols_train[0]} ... {time_cols_train[-1]}")
        logging.warning(f"Test format: {time_cols_test[0]} ... {time_cols_test[-1]}")
        
        # Use the format from training data and map test columns accordingly
        logging.info("Mapping test dataset columns to match training format...")
        
        # Create mapping from test format to train format
        col_mapping = {}
        for i, train_col in enumerate(time_cols_train):
            # Extract the number from train column
            if train_col[1] == '_':
                num = int(train_col[2:])
            else:
                num = int(train_col[1:])
            
            # Find corresponding test column
            test_col = None
            for test_candidate in time_cols_test:
                if test_candidate[1] == '_':
                    test_num = int(test_candidate[2:])
                else:
                    test_num = int(test_candidate[1:])
                
                if test_num == num:
                    test_col = test_candidate
                    break
            
            if test_col:
                col_mapping[test_col] = train_col
            else:
                logging.error(f"Could not find corresponding test column for {train_col}")
        
        # Rename test columns to match training format
        df_hfsm_test = df_hfsm_test.rename(columns=col_mapping)
        time_cols_test = time_cols_train  # Now they match
        logging.info("Successfully mapped test dataset columns to training format")
    
    # Use training format for consistency
    time_cols = time_cols_train

    # Extract parameters and responses
    X_hfsm_train = df_hfsm_train[param_features].values
    Y_hfsm_train = df_hfsm_train[time_cols].values

    X_hfsm_test = df_hfsm_test[param_features].values
    Y_hfsm_test = df_hfsm_test[time_cols].values

    logging.info(f"HFSM train data shape: X: {X_hfsm_train.shape}, Y: {Y_hfsm_train.shape}")
    logging.info(f"HFSM test data shape: X: {X_hfsm_test.shape}, Y: {Y_hfsm_test.shape}")
    
    # Check if arrays are empty before computing stats
    if Y_hfsm_train.size > 0:
        logging.info(f"HFSM train Y stats - Min: {Y_hfsm_train.min():.6f}, Max: {Y_hfsm_train.max():.6f}, Mean: {Y_hfsm_train.mean():.6f}")
    else:
        logging.warning("HFSM train Y array is empty after NaN removal")
    
    if Y_hfsm_test.size > 0:
        logging.info(f"HFSM test Y stats - Min: {Y_hfsm_test.min():.6f}, Max: {Y_hfsm_test.max():.6f}, Mean: {Y_hfsm_test.mean():.6f}")
    else:
        logging.warning("HFSM test Y array is empty after NaN removal")

    # Determine the time column format for consistency
    time_col_format = 'r_0' if len(time_cols) > 0 and time_cols[0][1] == '_' else 'r0'
    logging.info(f"Detected time column format: {time_col_format}")

    return X_hfsm_train, Y_hfsm_train, X_hfsm_test, Y_hfsm_test, time_col_format

def train_cae_model(cae, train_loader, val_loader, epochs, learning_rate, model_save_name, phase_name="Training", loss_weight=1.0):
    """
    Train CAE model with early stopping based on validation loss.

    Args:
        cae: Autoencoder model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        epochs: Number of epochs
        learning_rate: Learning rate
        model_save_name: Path to save best model
        phase_name: Name of training phase (for logging)
        loss_weight: Multiplier for training loss (for emphasizing certain datasets)
    """
    logging.info(f"--- Starting {phase_name} on {CONFIG['DEVICE']} for {epochs} epochs ---")
    logging.info(f"Model will be saved as: {model_save_name}")
    logging.info(f"Using validation set for model selection")
    if loss_weight != 1.0:
        logging.info(f"Applying loss weight multiplier: {loss_weight}x (emphasizing this dataset)")

    cae.to(CONFIG['DEVICE'])

    optimizer = optim.Adam(cae.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_path = os.path.join(CONFIG['OUTPUT_DIR'], model_save_name)
    patience_counter = 0
    patience = 30

    for epoch in range(epochs):
        # Training
        cae.train()
        total_train_loss = 0
        for batch in train_loader:
            timeseries = batch['timeseries'].to(CONFIG['DEVICE'])

            optimizer.zero_grad()
            recon_ts, _ = cae(timeseries)
            loss = criterion(recon_ts, timeseries)
            # Apply loss weight to emphasize learning from specific datasets
            weighted_loss = loss * loss_weight
            weighted_loss.backward()
            optimizer.step()
            total_train_loss += loss.item()  # Log original loss for consistency

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        cae.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                timeseries = batch['timeseries'].to(CONFIG['DEVICE'])
                recon_ts, _ = cae(timeseries)
                loss = criterion(recon_ts, timeseries)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(cae.state_dict(), best_model_path)
            patience_counter = 0
            logging.info(f"New best model at epoch {epoch+1} with val loss: {best_val_loss:.8f}")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")

        # Early stopping
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logging.info(f"--- {phase_name} Complete. Best model saved to {best_model_path} with val loss: {best_val_loss:.8f} ---")
    return best_model_path

def get_latent_vectors(encoder, dataloader):
    """Extract latent vectors using the trained encoder."""
    logging.info("Extracting latent vectors using the trained encoder.")
    encoder.to(CONFIG['DEVICE'])
    encoder.eval()
    all_latents = []
    with torch.no_grad():
        for batch in dataloader:
            timeseries = batch['timeseries'].to(CONFIG['DEVICE'])
            latents = encoder(timeseries)
            all_latents.append(latents.cpu().numpy())
    return np.vstack(all_latents)

def calculate_nmse(y_true, y_pred):
    """Calculate Normalized Mean Squared Error (NMSE) as percentage"""
    N = len(y_true)
    nmse_values = []

    for i in range(N):
        true_sample = y_true[i]
        pred_sample = y_pred[i]
        sigma_j = np.std(true_sample)

        if sigma_j > 0:
            mse_normalized = np.mean(((true_sample - pred_sample) / sigma_j) ** 2)
            nmse_values.append(mse_normalized)

    nmse_percentage = np.mean(nmse_values) * 100
    return nmse_percentage

def calculate_r2_roi(y_true, y_pred, locations):
    """
    Calculate R² score focusing ONLY on Region of Interest (ROI) timesteps.

    This metric excludes quiescent zones (zeros, constant baseline) that trivially
    inflate R² scores, providing true measure of prediction quality on dynamic regions.
    """
    N = len(y_true)
    r2_scores = []

    for i in range(N):
        location = locations[i]
        roi_start, roi_end = get_roi_for_location(location)

        # Extract ROI region only
        true_roi = y_true[i, roi_start:roi_end]
        pred_roi = y_pred[i, roi_start:roi_end]

        # Calculate R² on ROI
        if len(true_roi) > 0:
            r2 = r2_score(true_roi, pred_roi)
            r2_scores.append(r2)

    r2_roi = np.mean(r2_scores)
    return r2_roi, np.array(r2_scores)

def calculate_nmse_roi(y_true, y_pred, locations):
    """Calculate NMSE focusing ONLY on Region of Interest (ROI) timesteps."""
    N = len(y_true)
    nmse_values = []

    for i in range(N):
        location = locations[i]
        roi_start, roi_end = get_roi_for_location(location)

        # Extract ROI region only
        true_roi = y_true[i, roi_start:roi_end]
        pred_roi = y_pred[i, roi_start:roi_end]

        sigma_j = np.std(true_roi)
        if sigma_j > 0 and len(true_roi) > 0:
            mse_normalized = np.mean(((true_roi - pred_roi) / sigma_j) ** 2)
            nmse_values.append(mse_normalized)

    nmse_percentage = np.mean(nmse_values) * 100
    return nmse_percentage

def evaluate_on_dataset(cae, surrogate_model, params_scaler, X_data, Y_data, dataset_name):
    """Evaluate the model on a given dataset and return metrics (full and ROI-based)"""
    logging.info(f"--- Evaluating on {dataset_name} data ---")

    # Create dataset for evaluation (no noise for evaluation)
    eval_dataset = BeamResponseDataset(X_data, Y_data, params_scaler, add_noise=False)
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
        Y_pred = cae.decoder(Z_pred_tensor).cpu().numpy()

    # Extract locations (last column of X_data)
    locations = X_data[:, -1]

    # Calculate FULL metrics (traditional - inflated by zeros/baseline)
    mse_full = mean_squared_error(Y_data.reshape(-1), Y_pred.reshape(-1))
    r2_full = r2_score(Y_data.reshape(-1), Y_pred.reshape(-1))
    nmse_full = calculate_nmse(Y_data, Y_pred)

    # Calculate per-sample R² scores for full time series
    r2_per_sample_full = []
    for i in range(len(Y_data)):
        try:
            r2_sample = r2_score(Y_data[i], Y_pred[i])
            r2_per_sample_full.append(r2_sample)
        except:
            r2_per_sample_full.append(-np.inf)

    r2_per_sample_full = np.array(r2_per_sample_full)
    r2_full_mean = np.mean(r2_per_sample_full[r2_per_sample_full > -np.inf])
    r2_full_std = np.std(r2_per_sample_full[r2_per_sample_full > -np.inf])

    # Calculate ROI metrics (true performance on dynamic regions)
    r2_roi, r2_roi_per_sample = calculate_r2_roi(Y_data, Y_pred, locations)
    nmse_roi = calculate_nmse_roi(Y_data, Y_pred, locations)

    logging.info(f"{dataset_name} Evaluation Results:")
    logging.info(f"  Latent Space R²: {r2_latent:.4f}")
    logging.info(f"  === FULL METRICS (inflated by zeros/baseline) ===")
    logging.info(f"  Time Series R² (Full): {r2_full:.4f}")
    logging.info(f"  Time Series R² (Per-sample mean±std): {r2_full_mean:.4f}±{r2_full_std:.4f}")
    logging.info(f"  Time Series MSE (Full): {mse_full:.6f}")
    logging.info(f"  Time Series NMSE (Full): {nmse_full:.4f}%")
    logging.info(f"  === ROI METRICS (dynamic regions only - TRUE PERFORMANCE) ===")
    logging.info(f"  Time Series R² (ROI): {r2_roi:.4f} ← PRIMARY METRIC")
    logging.info(f"  Time Series NMSE (ROI): {nmse_roi:.4f}%")
    logging.info(f"  ROI R² range: [{r2_roi_per_sample.min():.4f}, {r2_roi_per_sample.max():.4f}]")

    return {
        'r2_latent': r2_latent,
        'r2_timeseries_full': r2_full,
        'r2_timeseries_roi': r2_roi,  # PRIMARY METRIC
        'r2_per_sample_full': r2_per_sample_full,
        'r2_full_mean': r2_full_mean,
        'r2_full_std': r2_full_std,
        'r2_roi_per_sample': r2_roi_per_sample,
        'mse_full': mse_full,
        'nmse_full': nmse_full,
        'nmse_roi': nmse_roi,
        'predictions': Y_pred
    }

def predict_timeseries_from_params(cae, surrogate_model, params_scaled):
    """Predict time series from parameters using surrogate + decoder."""
    cae.decoder.to(CONFIG['DEVICE'])
    cae.decoder.eval()
    with torch.no_grad():
        Z_pred = surrogate_model.predict(params_scaled)
        Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(CONFIG['DEVICE'])
        Y_pred = cae.decoder(Z_pred_tensor).cpu().numpy()
    return Y_pred

def dump_mfsm_interleaved_predictions(file_path, ground_truth, predictions, parameters=None, time_col_format='r0'):
    """
    Create interleaved CSV file where each ground truth response is followed by its MFSM prediction.
    
    Args:
        file_path: Path to save the CSV file
        ground_truth: Ground truth time series data
        predictions: Predicted time series data
        parameters: Optional parameter data
        time_col_format: Format for time columns ('r0' for r0,r1,... or 'r_0' for r_0,r_1,...)
    """
    n_samples, n_timesteps = ground_truth.shape

    # Prepare column names for time series based on format
    if time_col_format == 'r_0':
        time_cols = [f'r_{i}' for i in range(n_timesteps)]
    else:  # Default to 'r0' format
        time_cols = [f'r{i}' for i in range(n_timesteps)]

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

        # MFSM prediction row
        pred_row = []
        if parameters is not None:
            pred_row.extend(parameters[i])
        pred_row.extend(predictions[i])
        pred_row.append('mfsm_prediction')
        interleaved_data.append(pred_row)

    # Create DataFrame and save
    df = pd.DataFrame(interleaved_data, columns=all_cols)
    df.to_csv(file_path, index=False)

    logging.info(f"Saved interleaved MFSM predictions: {file_path} with shape {df.shape}")
    logging.info(f"Format: {n_samples} parameter sets, each with ground_truth + mfsm_prediction rows")
    logging.info(f"Time column format: {time_cols[0]} ... {time_cols[-1]}")

def create_comparison_plots(ground_truth, predictions, parameters, output_dir, dataset_name="MFSM_Test"):
    """
    Create comparison plots for the 10 best and 10 worst predictions vs ground truth.
    Rankings based on ROI R² scores (true performance on dynamic regions).
    Also creates individual plots for all samples and per-sample R2 histogram.
    """
    logging.info(f"--- Creating comparison plots for {dataset_name} data ---")

    n_samples, n_timesteps = ground_truth.shape

    # Extract locations
    locations = parameters[:, -1]

    # Calculate both full and ROI R² scores for each sample
    r2_scores_full = []
    r2_scores_roi = []

    for i in range(n_samples):
        try:
            # Full R² (inflated)
            r2_full = r2_score(ground_truth[i], predictions[i])
            r2_scores_full.append(r2_full)

            # ROI R² (true performance)
            location = locations[i]
            roi_start, roi_end = get_roi_for_location(location)
            true_roi = ground_truth[i, roi_start:roi_end]
            pred_roi = predictions[i, roi_start:roi_end]
            r2_roi = r2_score(true_roi, pred_roi)
            r2_scores_roi.append(r2_roi)
        except:
            r2_scores_full.append(-np.inf)
            r2_scores_roi.append(-np.inf)

    r2_scores_full = np.array(r2_scores_full)
    r2_scores_roi = np.array(r2_scores_roi)  # PRIMARY METRIC FOR RANKING

    # Find indices of 10 best and 10 worst predictions BASED ON ROI R²
    best_indices = np.argsort(r2_scores_roi)[-10:][::-1]  # Best ROI R²
    worst_indices = np.argsort(r2_scores_roi)[:10]  # Worst ROI R²

    # Create time axis
    time_axis = np.arange(n_timesteps)

    # Create individual plots for ALL samples
    logging.info(f"Creating individual comparison plots for all {n_samples} samples...")
    individual_plots_dir = os.path.join(output_dir, 'individual_plots')
    os.makedirs(individual_plots_dir, exist_ok=True)

    for i in range(n_samples):
        plt.figure(figsize=(10, 6))

        # Plot ground truth vs prediction
        plt.plot(time_axis, ground_truth[i], 'b-', label='Ground Truth', linewidth=2)
        plt.plot(time_axis, predictions[i], 'r--', label='MFSM Prediction', linewidth=2)

        # Format R2 scores for filename (handle negative values)
        r2_full_formatted = f"{r2_scores_full[i]:.4f}".replace('-', 'n').replace('.', '_')
        plt.title(f'Sample {i:04d}: R²={r2_scores_full[i]:.4f}\nLoc={locations[i]:.2f}, Params: {parameters[i][:3]}...', fontsize=10)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save individual plot
        individual_plot_path = os.path.join(individual_plots_dir, f'individual_plot_sample_{i:04d}_r2_{r2_full_formatted}.png')
        plt.savefig(individual_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    logging.info(f"Saved {n_samples} individual comparison plots to {individual_plots_dir}")

    # Plot 10 best predictions
    plt.figure(figsize=(20, 12))
    for i, idx in enumerate(best_indices):
        plt.subplot(2, 5, i+1)
        plt.plot(time_axis, ground_truth[idx], 'b-', label='Ground Truth', linewidth=1.5)
        plt.plot(time_axis, predictions[idx], 'r--', label='MFSM Prediction', linewidth=1.5)

        plt.title(f'Best #{i+1}: R²={r2_scores_full[idx]:.3f}\nLoc={locations[idx]:.2f}', fontsize=9)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()

    plt.suptitle(f'{dataset_name}: 10 Best Predictions', fontsize=16, y=1.02)
    best_plot_path = os.path.join(output_dir, f'mfsm_comparison_plots_best_10_{dataset_name.lower()}.png')
    plt.savefig(best_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved best predictions plot: {best_plot_path}")

    # Plot 10 worst predictions
    plt.figure(figsize=(20, 12))
    for i, idx in enumerate(worst_indices):
        plt.subplot(2, 5, i+1)
        plt.plot(time_axis, ground_truth[idx], 'b-', label='Ground Truth', linewidth=1.5)
        plt.plot(time_axis, predictions[idx], 'r--', label='MFSM Prediction', linewidth=1.5)

        plt.title(f'Worst #{i+1}: R²={r2_scores_full[idx]:.3f}\nLoc={locations[idx]:.2f}', fontsize=9)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()

    plt.suptitle(f'{dataset_name}: 10 Worst Predictions', fontsize=16, y=1.02)
    worst_plot_path = os.path.join(output_dir, f'mfsm_comparison_plots_worst_10_{dataset_name.lower()}.png')
    plt.savefig(worst_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved worst predictions plot: {worst_plot_path}")

    # Create per-sample R2 histogram (only full R2, not ROI)
    plt.figure(figsize=(10, 6))
    valid_full = r2_scores_full[r2_scores_full > -np.inf]
    plt.hist(valid_full, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(valid_full), color='red', linestyle='--', linewidth=2,
                label=f'Mean R² = {np.mean(valid_full):.4f}')
    plt.axvline(np.median(valid_full), color='orange', linestyle='--', linewidth=2,
                label=f'Median R² = {np.median(valid_full):.4f}')
    plt.xlabel('Per-Sample R² Score')
    plt.ylabel('Frequency')
    plt.title(f'Per-Sample R² Distribution - {dataset_name}\n({len(valid_full)} valid samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    r2_hist_path = os.path.join(output_dir, f'per_sample_r2_histogram_{dataset_name.lower()}.png')
    plt.savefig(r2_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved per-sample R2 histogram: {r2_hist_path}")

    # Create summary statistics plot
    plt.figure(figsize=(14, 10))

    # Plot R² distribution (ROI vs Full)
    plt.subplot(2, 3, 1)
    valid_roi = r2_scores_roi[r2_scores_roi > -np.inf]
    plt.hist(valid_roi, bins=30, alpha=0.6, color='orange', edgecolor='black', label='ROI R²')
    plt.hist(valid_full, bins=30, alpha=0.4, color='skyblue', edgecolor='black', label='Full R²')
    plt.axvline(np.mean(valid_roi), color='red', linestyle='--',
                label=f'Mean ROI R² = {np.mean(valid_roi):.4f}')
    plt.axvline(np.mean(valid_full), color='blue', linestyle='--',
                label=f'Mean Full R² = {np.mean(valid_full):.4f}')
    plt.xlabel('R² Score')
    plt.ylabel('Frequency')
    plt.title('R² Distribution: ROI vs Full')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # Plot ROI R² for best and worst cases
    plt.subplot(2, 3, 2)
    x = np.arange(10)
    width = 0.35
    plt.bar(x - width/2, r2_scores_roi[best_indices], width, label='Best 10 (ROI R²)', alpha=0.7, color='green')
    plt.bar(x + width/2, r2_scores_roi[worst_indices], width, label='Worst 10 (ROI R²)', alpha=0.7, color='red')
    plt.xlabel('Rank')
    plt.ylabel('ROI R²')
    plt.title('ROI R² Comparison: Best vs Worst')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot ROI R² vs Full R² scatter
    plt.subplot(2, 3, 3)
    plt.scatter(r2_scores_full, r2_scores_roi, alpha=0.5, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
    plt.xlabel('Full R² (inflated)')
    plt.ylabel('ROI R² (true performance)')
    plt.title('ROI R² vs Full R²')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot parameter space distribution for best vs worst
    plt.subplot(2, 3, 4)
    param_best = parameters[best_indices, 0]
    param_worst = parameters[worst_indices, 0]
    plt.scatter(param_best, r2_scores_roi[best_indices], color='green', alpha=0.7, label='Best 10', s=60)
    plt.scatter(param_worst, r2_scores_roi[worst_indices], color='red', alpha=0.7, label='Worst 10', s=60)
    plt.xlabel(f'{CONFIG["PARAM_COLS"][0]}')
    plt.ylabel('ROI R²')
    plt.title('ROI Performance vs Parameter Space')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot average response comparison for best samples
    plt.subplot(2, 3, 5)
    avg_gt_best = np.mean(ground_truth[best_indices], axis=0)
    avg_pred_best = np.mean(predictions[best_indices], axis=0)
    plt.plot(time_axis, avg_gt_best, 'g-', label='GT Best Avg', linewidth=2)
    plt.plot(time_axis, avg_pred_best, 'g--', label='Pred Best Avg', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Response')
    plt.title('Average Response: Best 10')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot average response comparison for worst samples
    plt.subplot(2, 3, 6)
    avg_gt_worst = np.mean(ground_truth[worst_indices], axis=0)
    avg_pred_worst = np.mean(predictions[worst_indices], axis=0)
    plt.plot(time_axis, avg_gt_worst, 'r-', label='GT Worst Avg', linewidth=2)
    plt.plot(time_axis, avg_pred_worst, 'r--', label='Pred Worst Avg', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Response')
    plt.title('Average Response: Worst 10')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, f'mfsm_comparison_summary_{dataset_name.lower()}.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved summary plot: {summary_plot_path}")

    # Log statistics
    logging.info(f"{dataset_name} Comparison Statistics:")
    logging.info(f"  === ROI R² (PRIMARY METRIC) ===")
    logging.info(f"  Best ROI R² scores: {r2_scores_roi[best_indices]}")
    logging.info(f"  Worst ROI R² scores: {r2_scores_roi[worst_indices]}")
    logging.info(f"  Mean ROI R² (all): {np.mean(valid_roi):.4f}")
    logging.info(f"  Std ROI R² (all): {np.std(valid_roi):.4f}")
    logging.info(f"  === Full R² (for comparison) ===")
    logging.info(f"  Mean Full R² (all): {np.mean(valid_full):.4f}")
    logging.info(f"  Difference (Full - ROI): {np.mean(valid_full) - np.mean(valid_roi):.4f}")
    logging.info(f"  === Per-Sample Analysis ===")
    logging.info(f"  Per-sample R² range: [{valid_full.min():.4f}, {valid_full.max():.4f}]")
    logging.info(f"  Per-sample R² std: {np.std(valid_full):.4f}")

    return {
        'best_indices': best_indices,
        'worst_indices': worst_indices,
        'r2_scores_roi': r2_scores_roi,
        'r2_scores_full': r2_scores_full,
        'best_r2_roi': r2_scores_roi[best_indices],
        'worst_r2_roi': r2_scores_roi[worst_indices],
        'best_r2_full': r2_scores_full[best_indices],
        'worst_r2_full': r2_scores_full[worst_indices],
        'individual_plots_dir': individual_plots_dir,
        'r2_histogram_path': r2_hist_path
    }

def random_sample_hfsm_data(X_hfsm_train, Y_hfsm_train, num_samples=None):
    """
    Randomly sample HFSM training data for fine-tuning.

    Args:
        X_hfsm_train: HFSM training parameters
        Y_hfsm_train: HFSM training time series
        num_samples: Number of samples to select randomly (None = use all)

    Returns:
        X_sampled: Randomly sampled parameters
        Y_sampled: Randomly sampled time series
    """
    if num_samples is None or num_samples >= len(X_hfsm_train):
        logging.info(f"Using all {len(X_hfsm_train)} HFSM training samples for fine-tuning")
        return X_hfsm_train, Y_hfsm_train

    logging.info(f"Randomly sampling {num_samples} out of {len(X_hfsm_train)} HFSM training samples for fine-tuning")

    # Randomly select indices without replacement
    selected_indices = np.random.choice(len(X_hfsm_train), size=num_samples, replace=False)

    # Extract selected samples
    X_sampled = X_hfsm_train[selected_indices]
    Y_sampled = Y_hfsm_train[selected_indices]

    logging.info(f"Sampled HFSM training data shape: X: {X_sampled.shape}, Y: {Y_sampled.shape}")
    logging.info(f"Original HFSM train Y stats - Min: {Y_hfsm_train.min():.6f}, Max: {Y_hfsm_train.max():.6f}, Mean: {Y_hfsm_train.mean():.6f}")
    logging.info(f"Sampled HFSM train Y stats - Min: {Y_sampled.min():.6f}, Max: {Y_sampled.max():.6f}, Mean: {Y_sampled.mean():.6f}")

    return X_sampled, Y_sampled

def check_existing_models():
    """Check if pre-trained AutoEncoder models exist"""
    lfsm_model_path = os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_lfsm_pretrained.pth')
    mfsm_model_path = os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_finetuned.pth')

    models_exist = os.path.exists(lfsm_model_path) and os.path.exists(mfsm_model_path)

    if models_exist:
        logging.info("✓ Found existing AutoEncoder models:")
        logging.info(f"  - LFSM pre-trained: {lfsm_model_path}")
        logging.info(f"  - MFSM fine-tuned: {mfsm_model_path}")
        if CONFIG['USE_EXISTING_MODELS']:
            logging.info("✓ Will use existing models (USE_EXISTING_MODELS=True)")
        else:
            logging.info("⚠ Models found but USE_EXISTING_MODELS=False - will retrain")
    else:
        logging.info("✗ AutoEncoder models not found")
        if CONFIG['USE_EXISTING_MODELS']:
            logging.warning("⚠ USE_EXISTING_MODELS=True but models missing - will train from scratch")

    return models_exist

def load_existing_models():
    """Load existing AutoEncoder models and scalers"""
    logging.info("--- Loading Existing AutoEncoder Models ---")

    # Load LFSM pre-trained model
    lfsm_model_path = os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_lfsm_pretrained.pth')
    cae = Autoencoder(
        timeseries_dim=CONFIG['NUM_TIME_STEPS'],
        latent_dim=CONFIG['LFSM_LATENT_DIM']
    )
    cae.load_state_dict(torch.load(lfsm_model_path, map_location=CONFIG['DEVICE']))
    logging.info(f"Loaded LFSM pre-trained model from {lfsm_model_path}")

    # Load scaler
    scaler_path = os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_scaler.joblib')
    if os.path.exists(scaler_path):
        p_scaler = joblib.load(scaler_path)
        logging.info(f"Loaded parameter scaler from {scaler_path}")
    else:
        logging.error("Scaler not found! Need to train AutoEncoder from scratch.")
        return None, None

    return cae, p_scaler

def main():
    logging.info("=== STARTING MULTI-FIDELITY SURROGATE MODEL (MFSM) TRAINING ===")
    logging.info("Architecture: LFSM (1D) Pre-training → MFSM (2D) Fine-tuning")

    # Log random sampling configuration for MFSM fine-tuning
    if CONFIG['NUM_MFSM_TRAIN_SAMPLES'] is not None:
        logging.info(f"MFSM Configuration: Using {CONFIG['NUM_MFSM_TRAIN_SAMPLES']} randomly sampled 2D training samples for fine-tuning")
    else:
        logging.info("MFSM Configuration: Using all available 2D training samples for fine-tuning")

    # Check if pre-trained AutoEncoder models exist and if user wants to use them
    existing_models_available = check_existing_models()
    use_existing = CONFIG['USE_EXISTING_MODELS'] and existing_models_available

    if use_existing:
        logging.info(f"✓ Using existing AutoEncoder models (USE_EXISTING_MODELS={CONFIG['USE_EXISTING_MODELS']})")
    elif CONFIG['USE_EXISTING_MODELS'] and not existing_models_available:
        logging.warning(f"⚠ USE_EXISTING_MODELS=True but models not found - will train from scratch")
    else:
        logging.info(f"✓ Training AutoEncoder from scratch (USE_EXISTING_MODELS={CONFIG['USE_EXISTING_MODELS']})")

    # ===== PHASE 1: LOAD DATA =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 1: LOADING DATA")
    logging.info("="*60)

    # Load LFSM (1D zigzag) data
    X_lfsm_train, Y_lfsm_train, X_lfsm_test, Y_lfsm_test = load_lfsm_data()

    # Load HFSM (2D FEM) data
    X_hfsm_train, Y_hfsm_train, X_hfsm_test, Y_hfsm_test, hfsm_time_col_format = load_hfsm_data()

    # Randomly sample HFSM training data for fine-tuning (if specified)
    X_hfsm_train, Y_hfsm_train = random_sample_hfsm_data(
        X_hfsm_train, Y_hfsm_train, CONFIG['NUM_MFSM_TRAIN_SAMPLES']
    )

    # ===== CONDITIONAL TRAINING: Check for existing models =====
    if use_existing:
        logging.info("\n" + "="*60)
        logging.info("LOADING EXISTING MODELS - SKIPPING AUTOENCODER TRAINING")
        logging.info("="*60)

        # Load existing models and scaler
        cae, p_scaler = load_existing_models()
        if cae is None:
            logging.error("Failed to load existing models. Training from scratch.")
            use_existing = False

    # ===== PHASE 2: LFSM PRE-TRAINING (Conditional) =====

    # Create datasets and dataloaders for LFSM (needed for both training and loading existing models)
    lfsm_train_dataset = BeamResponseDataset(X_lfsm_train, Y_lfsm_train, add_noise=True, noise_std=0.03)
    lfsm_val_dataset = BeamResponseDataset(X_lfsm_test, Y_lfsm_test, p_scaler=lfsm_train_dataset.p_scaler, add_noise=False)

    lfsm_train_loader = DataLoader(lfsm_train_dataset, batch_size=CONFIG['LFSM_CAE_BATCH_SIZE'], shuffle=True, drop_last=True)
    lfsm_val_loader = DataLoader(lfsm_val_dataset, batch_size=CONFIG['LFSM_CAE_BATCH_SIZE'], shuffle=False)
    lfsm_train_loader_full = DataLoader(lfsm_train_dataset, batch_size=len(lfsm_train_dataset), shuffle=False)

    # Initialize timing variables (will be set during training or when using existing models)
    training_time_lfsm_ae = 0
    training_time_mfsm_ae = 0
    training_time_xgb_phase1 = 0
    training_time_xgb_phase2 = 0

    if not use_existing:
        logging.info("\n" + "="*60)
        logging.info("PHASE 2: LFSM (1D) PRE-TRAINING")
        logging.info("="*60)

        # Initialize and train CAE on LFSM data
        cae = Autoencoder(
            timeseries_dim=CONFIG['NUM_TIME_STEPS'],
            latent_dim=CONFIG['LFSM_LATENT_DIM']
        )

        # Time LFSM AE pre-training
        logging.info("Starting LFSM AE pre-training...")
        start_time_lfsm_ae = time.time()
        lfsm_model_path = train_cae_model(
            cae, lfsm_train_loader, lfsm_val_loader,
            CONFIG['LFSM_CAE_EPOCHS'], CONFIG['LFSM_LEARNING_RATE'],
            'mfsm_lfsm_pretrained.pth',
            phase_name="LFSM Pre-training",
            loss_weight=1.0  # No additional weighting for pre-training
        )
        end_time_lfsm_ae = time.time()
        training_time_lfsm_ae = end_time_lfsm_ae - start_time_lfsm_ae
        logging.info(f"LFSM AE pre-training completed in {training_time_lfsm_ae:.2f} seconds ({training_time_lfsm_ae/60:.2f} minutes)")

        # Load best LFSM model
        logging.info(f"Loading best LFSM pre-trained model from {lfsm_model_path}")
        cae.load_state_dict(torch.load(lfsm_model_path, map_location=CONFIG['DEVICE']))
    else:
        # Use already loaded model
        logging.info("Using pre-loaded LFSM model for XGBoost training")
        logging.info("Note: LFSM AE training time not measured (using existing model)")

    # Extract latent vectors from LFSM for XGBoost training
    Z_lfsm_train = get_latent_vectors(cae.encoder, lfsm_train_loader_full)
    logging.info(f"Extracted latent vectors for LFSM training data. Shape: {Z_lfsm_train.shape}")

    # Train XGBoost on LFSM latent vectors (Phase 1)
    logging.info("--- Phase 1: Training XGBoost on LFSM Latent Vectors ---")
    xgb_params_lfsm = {
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
        xgb_params_lfsm['tree_method'] = 'gpu_hist'
        xgb_params_lfsm['predictor'] = 'gpu_predictor'
        xgb_params_lfsm['gpu_id'] = 0
        logging.info("Using GPU for XGBoost LFSM training (gpu_hist).")
    else:
        xgb_params_lfsm['tree_method'] = 'hist'
        xgb_params_lfsm['predictor'] = 'cpu_predictor'

    # Initialize XGBoost for Phase 1 (LFSM)
    surrogate_model_lfsm = xgb.XGBRegressor(**xgb_params_lfsm, n_jobs=-1)
    
    # Time XGBoost Phase 1 training
    logging.info("Starting XGBoost Phase 1 training...")
    start_time_xgb_phase1 = time.time()
    surrogate_model_lfsm.fit(lfsm_train_dataset.params_scaled, Z_lfsm_train, verbose=False)
    end_time_xgb_phase1 = time.time()
    training_time_xgb_phase1 = end_time_xgb_phase1 - start_time_xgb_phase1
    logging.info(f"XGBoost Phase 1 training completed in {training_time_xgb_phase1:.2f} seconds ({training_time_xgb_phase1/60:.2f} minutes)")
    logging.info("--- Phase 1 XGBoost Training on LFSM Complete ---")

    # Generate LFSM XGBoost predictions for comparison
    Z_lfsm_pred = surrogate_model_lfsm.predict(lfsm_train_dataset.params_scaled)
    r2_latent_lfsm = r2_score(Z_lfsm_train, Z_lfsm_pred)
    logging.info(f"LFSM Latent Space Prediction R²: {r2_latent_lfsm:.4f}")

    # Save Phase 1 XGBoost model
    joblib.dump(surrogate_model_lfsm, os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_surrogate_lfsm.joblib'))
    logging.info("Saved Phase 1 XGBoost model (LFSM)")

    # ===== PHASE 2.5: LFSM AUTOENCODER RECONSTRUCTION EVALUATION (Conditional) =====
    if not use_existing:
        logging.info("\n" + "="*60)
        logging.info("PHASE 2.5: LFSM AUTOENCODER RECONSTRUCTION EVALUATION")
        logging.info("="*60)
        ae_recon_lfsm_metrics = evaluate_ae_reconstruction(cae, lfsm_train_loader_full, "LFSM Pre-trained (AE Only)")
    else:
        logging.info("\n" + "="*60)
        logging.info("PHASE 2.5: SKIPPING LFSM RECONSTRUCTION EVALUATION (Using existing models)")
        logging.info("="*60)
        ae_recon_lfsm_metrics = None

    # ===== PHASE 3: MFSM FINE-TUNING ON 2D DATA (Conditional) =====

    # Create datasets and dataloaders for MFSM (needed for both training and loading existing models)
    # Use the same parameter scaler from LFSM for consistency
    mfsm_train_dataset = BeamResponseDataset(X_hfsm_train, Y_hfsm_train,
                                              p_scaler=lfsm_train_dataset.p_scaler,
                                              add_noise=True, noise_std=0.03)
    mfsm_val_dataset = BeamResponseDataset(X_hfsm_test, Y_hfsm_test,
                                            p_scaler=mfsm_train_dataset.p_scaler,
                                            add_noise=False)

    mfsm_train_loader = DataLoader(mfsm_train_dataset, batch_size=CONFIG['MFSM_CAE_BATCH_SIZE'],
                                    shuffle=True, drop_last=True)
    mfsm_val_loader = DataLoader(mfsm_val_dataset, batch_size=CONFIG['MFSM_CAE_BATCH_SIZE'], shuffle=False)
    mfsm_train_loader_full = DataLoader(mfsm_train_dataset, batch_size=len(mfsm_train_dataset), shuffle=False)

    if not use_existing:
        logging.info("\n" + "="*60)
        logging.info("PHASE 3: MFSM FINE-TUNING ON 2D FEM DATA")
        logging.info("="*60)

        # Fine-tune CAE on 2D data
        logging.info("Starting MFSM AE fine-tuning...")
        start_time_mfsm_ae = time.time()
        mfsm_model_path = train_cae_model(
            cae, mfsm_train_loader, mfsm_val_loader,
            CONFIG['MFSM_CAE_EPOCHS'], CONFIG['MFSM_LEARNING_RATE'],
            'mfsm_finetuned.pth',
            phase_name="MFSM Fine-tuning",
            loss_weight=CONFIG['MFSM_LOSS_WEIGHT']  # 3x more weightage for 2D learning
        )
        end_time_mfsm_ae = time.time()
        training_time_mfsm_ae = end_time_mfsm_ae - start_time_mfsm_ae
        logging.info(f"MFSM AE fine-tuning completed in {training_time_mfsm_ae:.2f} seconds ({training_time_mfsm_ae/60:.2f} minutes)")

        # Load best MFSM model
        logging.info(f"Loading best MFSM fine-tuned model from {mfsm_model_path}")
        cae.load_state_dict(torch.load(mfsm_model_path, map_location=CONFIG['DEVICE']))
    else:
        # Use already loaded model
        logging.info("Using pre-loaded MFSM model for XGBoost training")
        logging.info("Note: MFSM AE training time not measured (using existing model)")

    # ===== PHASE 4: XGBOOST SURROGATE FINE-TUNING =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 4: XGBOOST SURROGATE MODEL TRAINING ON 2D DATA")
    logging.info("="*60)

    # Extract latent vectors from fine-tuned encoder for HFSM training data
    Z_hfsm_train = get_latent_vectors(cae.encoder, mfsm_train_loader_full)
    logging.info(f"Extracted latent vectors for HFSM training data. Shape: {Z_hfsm_train.shape}")

    # Create test dataset and dataloader for latent vector extraction
    mfsm_test_dataset = BeamResponseDataset(X_hfsm_test, Y_hfsm_test, mfsm_train_dataset.p_scaler, add_noise=False)
    mfsm_test_loader_full = DataLoader(mfsm_test_dataset, batch_size=len(mfsm_test_dataset), shuffle=False)

    # Extract latent vectors for test data
    Z_test = get_latent_vectors(cae.encoder, mfsm_test_loader_full)
    logging.info(f"Extracted latent vectors for test data. Shape: {Z_test.shape}")

    # Load Phase 1 XGBoost model for fine-tuning
    logging.info("--- Loading Phase 1 XGBoost Model for Fine-tuning ---")
    try:
        surrogate_model_phase1 = joblib.load(os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_surrogate_lfsm.joblib'))
        logging.info("Successfully loaded Phase 1 XGBoost model")
    except FileNotFoundError:
        logging.error("Phase 1 XGBoost model not found! Please ensure Phase 1 completed successfully.")
        raise

    # Prepare data for XGBoost fine-tuning
    logging.info("--- Preparing Data for XGBoost Fine-tuning ---")

    # Combine LFSM and HFSM data for fine-tuning
    # LFSM data (from Phase 1)
    X_lfsm_combined = lfsm_train_dataset.params_scaled
    Z_lfsm_combined = Z_lfsm_train

    # HFSM data (from Phase 2)
    X_hfsm_combined = mfsm_train_dataset.params_scaled
    Z_hfsm_combined = Z_hfsm_train

    # Concatenate datasets
    X_combined = np.vstack([X_lfsm_combined, X_hfsm_combined])
    Z_combined = np.vstack([Z_lfsm_combined, Z_hfsm_combined])

    # Create sample weights with same weightage as AutoEncoder (3x for HFSM)
    n_lfsm = len(X_lfsm_combined)
    n_hfsm = len(X_hfsm_combined)
    sample_weights = np.concatenate([
        np.ones(n_lfsm) * 1.0,  # LFSM samples get weight 1.0
        np.ones(n_hfsm) * CONFIG['MFSM_LOSS_WEIGHT']  # HFSM samples get 3x weight
    ])

    logging.info(f"XGBoost Fine-tuning Data:")
    logging.info(f"  LFSM samples: {n_lfsm} (weight: 1.0x)")
    logging.info(f"  HFSM samples: {n_hfsm} (weight: {CONFIG['MFSM_LOSS_WEIGHT']}x)")
    logging.info(f"  Total samples: {len(X_combined)}")
    logging.info(f"  Latent dimensions: {Z_combined.shape[1]}")

    # Fine-tune XGBoost surrogate model with sample weights
    logging.info("--- Fine-tuning XGBoost Surrogate Model with Weighted Samples ---")
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
        logging.info("Using GPU for XGBoost fine-tuning (gpu_hist).")
    else:
        xgb_params['tree_method'] = 'hist'
        xgb_params['predictor'] = 'cpu_predictor'

    # Create fine-tuned XGBoost model starting from Phase 1 model
    surrogate_model = xgb.XGBRegressor(**xgb_params, n_jobs=-1)

    # Fine-tune with sample weights (same weightage as AutoEncoder: 3x for HFSM)
    logging.info("Starting XGBoost Phase 2 fine-tuning...")
    start_time_xgb_phase2 = time.time()
    surrogate_model.fit(X_combined, Z_combined, sample_weight=sample_weights, verbose=False)
    end_time_xgb_phase2 = time.time()
    training_time_xgb_phase2 = end_time_xgb_phase2 - start_time_xgb_phase2
    logging.info(f"XGBoost Phase 2 fine-tuning completed in {training_time_xgb_phase2:.2f} seconds ({training_time_xgb_phase2/60:.2f} minutes)")
    logging.info("--- XGBoost Fine-tuning Complete ---")
    logging.info(f"XGBoost fine-tuned with sample weights: LFSM (1.0x) + HFSM ({CONFIG['MFSM_LOSS_WEIGHT']}x)")

    # Generate and save surrogate model predictions for comparison
    Z_hfsm_train_pred = surrogate_model.predict(mfsm_train_dataset.params_scaled)
    Z_test_pred = surrogate_model.predict(mfsm_test_dataset.params_scaled)

    # Calculate R² scores for latent space predictions
    r2_latent_train = r2_score(Z_hfsm_train, Z_hfsm_train_pred)
    r2_latent_test = r2_score(Z_test, Z_test_pred)

    logging.info(f"HFSM Latent Space Prediction R² - Train: {r2_latent_train:.4f}, Test: {r2_latent_test:.4f}")
    logging.info(f"LFSM Latent Space Prediction R² (Phase 1): {r2_latent_lfsm:.4f}")

    # Save surrogate predictions
    surrogate_predictions = {
        'Z_hfsm_train_true': Z_hfsm_train,
        'Z_hfsm_train_pred': Z_hfsm_train_pred,
        'Z_test_true': Z_test,
        'Z_test_pred': Z_test_pred,
        'Z_lfsm_true': Z_lfsm_train,
        'Z_lfsm_pred': Z_lfsm_pred,
        'r2_hfsm_train': r2_latent_train,
        'r2_test': r2_latent_test,
        'r2_lfsm': r2_latent_lfsm
    }
    np.savez(os.path.join(CONFIG['OUTPUT_DIR'], 'surrogate_latent_predictions.npz'), **surrogate_predictions)
    logging.info("Saved surrogate model latent space predictions for comparison")

    # Save fine-tuned XGBoost model and scaler
    joblib.dump(surrogate_model, os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_surrogate_finetuned.joblib'))
    joblib.dump(mfsm_train_dataset.p_scaler, os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_scaler.joblib'))
    logging.info("Saved fine-tuned XGBoost surrogate model")

    # ===== PHASE 5: EVALUATION =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 5: EVALUATION ON 2D TEST DATA")
    logging.info("="*60)

    # Evaluate on 2D test data
    results_test = evaluate_on_dataset(cae, surrogate_model, mfsm_train_dataset.p_scaler,
                                      X_hfsm_test, Y_hfsm_test, "MFSM_Test")

    # ===== PHASE 6: GENERATE OUTPUTS =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 6: GENERATING OUTPUTS AND VISUALIZATIONS")
    logging.info("="*60)

    # Build test dataset for predictions
    test_dataset = BeamResponseDataset(X_hfsm_test, Y_hfsm_test, mfsm_train_dataset.p_scaler)
    Y_pred_test = predict_timeseries_from_params(cae, surrogate_model, test_dataset.params_scaled)

    # Save interleaved predictions
    dump_mfsm_interleaved_predictions(
        os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_interleaved_test.csv'),
        Y_hfsm_test, Y_pred_test, X_hfsm_test, time_col_format=hfsm_time_col_format
    )

    # Save predictions as numpy array
    np.save(os.path.join(CONFIG['OUTPUT_DIR'], 'mfsm_predictions_test.npy'), Y_pred_test)

    # Create comparison plots
    plot_stats = create_comparison_plots(
        Y_hfsm_test, Y_pred_test, X_hfsm_test,
        CONFIG['OUTPUT_DIR'], dataset_name="MFSM_Test"
    )

    # ===== PHASE 6.5: FINAL AUTOENCODER RECONSTRUCTION EVALUATION (Conditional) =====
    if not use_existing:
        logging.info("\n" + "="*60)
        logging.info("PHASE 6.5: FINAL AUTOENCODER RECONSTRUCTION EVALUATION")
        logging.info("="*60)

        # Evaluate autoencoder reconstruction on training data
        ae_recon_train_metrics = evaluate_ae_reconstruction(cae, mfsm_train_loader_full, "MFSM Train (AE Only)")
    else:
        logging.info("\n" + "="*60)
        logging.info("PHASE 6.5: SKIPPING FINAL RECONSTRUCTION EVALUATION (Using existing models)")
        logging.info("="*60)
        ae_recon_train_metrics = None

    # ===== FINAL SUMMARY =====
    logging.info("\n" + "="*60)
    logging.info("FINAL SUMMARY")
    logging.info("="*60)

    logging.info("TRAINING STRATEGY:")
    if use_existing:
        logging.info(f"  ✓ Using existing AutoEncoder models (USE_EXISTING_MODELS={CONFIG['USE_EXISTING_MODELS']})")
    else:
        logging.info(f"  ✓ Trained AutoEncoder from scratch (USE_EXISTING_MODELS={CONFIG['USE_EXISTING_MODELS']})")

    logging.info(f"  1. LFSM Pre-training: {len(X_lfsm_train)} samples from 1D zigzag theory (loss weight: 1.0x)")
    hfsm_samples_info = f"{len(X_hfsm_train)} samples" if CONFIG['NUM_MFSM_TRAIN_SAMPLES'] is None else f"{CONFIG['NUM_MFSM_TRAIN_SAMPLES']} randomly sampled"
    logging.info(f"  2. MFSM Fine-tuning: {hfsm_samples_info} from 2D FEM (loss weight: {CONFIG['MFSM_LOSS_WEIGHT']}x)")
    logging.info(f"  3. XGBoost Phase 1: Training on LFSM latent vectors")
    logging.info(f"  4. XGBoost Phase 2: Fine-tuning on combined LFSM+HFSM data (HFSM weight: {CONFIG['MFSM_LOSS_WEIGHT']}x)")
    logging.info(f"  5. Final Testing: {len(X_hfsm_test)} samples from 2D FEM")
    logging.info("")

    if ae_recon_lfsm_metrics is not None:
        logging.info("AUTOENCODER RECONSTRUCTION QUALITY (R²):")
        logging.info(f"  LFSM Pre-trained (AE Only): R²={ae_recon_lfsm_metrics['r2_overall']:.4f}")
        logging.info(f"  MFSM Fine-tuned (AE Only): R²={ae_recon_train_metrics['r2_overall']:.4f}")
    else:
        logging.info("AUTOENCODER RECONSTRUCTION QUALITY:")
        logging.info("  (Not evaluated - using existing models)")
    logging.info("")
    logging.info(f"MFSM Test Results:")
    logging.info(f"  R² (Full): {results_test['r2_timeseries_full']:.4f}, NMSE (Full): {results_test['nmse_full']:.4f}%")
    logging.info(f"  R² (Per-sample mean±std): {results_test['r2_full_mean']:.4f}±{results_test['r2_full_std']:.4f}")
    logging.info(f"  R² (ROI): {results_test['r2_timeseries_roi']:.4f} ← PRIMARY METRIC, NMSE (ROI): {results_test['nmse_roi']:.4f}%")
    logging.info(f"  Best ROI R² range: {plot_stats['best_r2_roi'].min():.4f} to {plot_stats['best_r2_roi'].max():.4f}")
    logging.info(f"  Worst ROI R² range: {plot_stats['worst_r2_roi'].min():.4f} to {plot_stats['worst_r2_roi'].max():.4f}")

    logging.info("\nFiles saved to output directory/:")
    logging.info("  === MODEL FILES ===")
    logging.info("  - mfsm_lfsm_pretrained.pth (LFSM pre-trained CAE)")
    logging.info("  - mfsm_finetuned.pth (MFSM fine-tuned CAE)")
    logging.info("  - mfsm_surrogate_lfsm.joblib (XGBoost Phase 1 - LFSM trained)")
    logging.info("  - mfsm_surrogate_finetuned.joblib (XGBoost Phase 2 - Fine-tuned)")
    logging.info("  - mfsm_scaler.joblib (Parameter scaler)")
    logging.info("  === EVALUATION FILES ===")
    logging.info("  - mfsm_interleaved_test.csv (Ground truth + predictions)")
    logging.info("  - mfsm_predictions_test.npy (Test predictions array)")
    logging.info("  - mfsm_comparison_plots_*.png (Performance visualizations)")
    logging.info("  === PARAMETER-LATENT ANALYSIS FILES ===")
    logging.info("  - train_param_latent_pairs.npz (Training parameter-latent pairs)")
    logging.info("  - test_param_latent_pairs.npz (Test parameter-latent pairs)")
    logging.info("  - surrogate_latent_predictions.npz (Surrogate model predictions vs true)")

    # Training Time Summary
    logging.info("\n" + "="*60)
    logging.info("TRAINING TIME SUMMARY")
    logging.info("="*60)
    if training_time_lfsm_ae > 0:
        logging.info(f"LFSM AE Pre-training: {training_time_lfsm_ae:.2f} seconds ({training_time_lfsm_ae/60:.2f} minutes)")
    else:
        logging.info("LFSM AE Pre-training: Not measured (using existing model)")
    
    if training_time_mfsm_ae > 0:
        logging.info(f"MFSM AE Fine-tuning: {training_time_mfsm_ae:.2f} seconds ({training_time_mfsm_ae/60:.2f} minutes)")
    else:
        logging.info("MFSM AE Fine-tuning: Not measured (using existing model)")
    
    if training_time_xgb_phase1 > 0:
        logging.info(f"XGBoost Phase 1 (LFSM): {training_time_xgb_phase1:.2f} seconds ({training_time_xgb_phase1/60:.2f} minutes)")
    else:
        logging.info("XGBoost Phase 1 (LFSM): Not measured (using existing model)")
    
    if training_time_xgb_phase2 > 0:
        logging.info(f"XGBoost Phase 2 (Fine-tuning): {training_time_xgb_phase2:.2f} seconds ({training_time_xgb_phase2/60:.2f} minutes)")
    else:
        logging.info("XGBoost Phase 2 (Fine-tuning): Not measured (using existing model)")
    
    total_training_time = training_time_lfsm_ae + training_time_mfsm_ae + training_time_xgb_phase1 + training_time_xgb_phase2
    if total_training_time > 0:
        logging.info(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    else:
        logging.info("Total Training Time: Not measured (using existing models)")
    logging.info("="*60)

    logging.info("--- MFSM Training Complete ---")
    logging.info("="*60)

if __name__ == '__main__':
    main()
