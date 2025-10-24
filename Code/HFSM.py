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

# Import additional metrics functions (similar to LFSMIII.py)
def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_cosine_similarity(y_true, y_pred):
    """Calculate cosine similarity between true and predicted time series"""
    # Flatten the time series for cosine similarity calculation
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # Normalize vectors
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
    'DATA_FILE_TRAIN': '/home/user2/Music/abhi3/parameters/train_responseslatest.csv',
    'DATA_FILE_TEST': '/home/user2/Music/abhi3/parameters/test_responseslatest.csv',
    'OUTPUT_DIR': '/home/user2/Music/abhi3/HFSM',

    # --- Data & Model Hyperparameters ---
    'PARAM_COLS': ['notch_x', 'notch_depth', 'notch_width', 'length', 'density', 'youngs_modulus'],
    'NUM_TIME_STEPS': 1500,
    'NUM_TRAIN_SAMPLES': 1375,  
    'NUM_CLUSTERS': 10,  # K-means clusters for aggressive sampling

    # --- CAE Training ---
    'LATENT_DIM': 30,
    'CAE_EPOCHS': 100,
    'CAE_BATCH_SIZE': 32, #reduced for smaller dataset
    'CAE_LEARNING_RATE': 1e-5,

    # --- XGBoost Surrogate Model ---
    'XGB_N_ESTIMATORS': 2000,
    'XGB_MAX_DEPTH': 10,
    'XGB_ETA': 0.02,
    'XGB_EARLY_STOPPING': 10,
}
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

# Initialize TIME_COLS as None - will be set dynamically based on data format
CONFIG['TIME_COLS'] = None

# --- Setup Logging ---
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(CONFIG['OUTPUT_DIR'], 'hfsm_training_log.log')),
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
    Responses are already normalized. Optional noise augmentation for training.
    """
    def __init__(self, params, timeseries, p_scaler=None, add_noise=False, noise_std=0.03):
        # Store raw time series without scaling - responses are already normalized
        self.timeseries = timeseries.astype(np.float32).copy()
        self.params = params.astype(np.float32)

        # Add Gaussian noise to training data to prevent overfitting
        # Reflects measurement uncertainty in experimental validation
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

# --- Autoencoder PyTorch Models (Identical to LFSMIII.py) ---
class Encoder(nn.Module):
    """Encoder: Time series � Latent space (NO parameter conditioning)"""
    def __init__(self, timeseries_dim, latent_dim):
        super(Encoder, self).__init__()
        self.timeseries_net = nn.Sequential(
            nn.Linear(timeseries_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        """Forward pass - only takes time series, no parameters"""
        return self.timeseries_net(x)

class Decoder(nn.Module):
    """Decoder: Latent space � Time series (NO parameter conditioning)"""
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.expansion = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
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

# --- Main Functions ---
def load_and_cluster_sample_data():
    """
    Load 2D training data and apply clustered sampling to select 200 samples.
    Uses K-means clustering to create parameter space clusters.
    """
    logging.info("=== LOADING 2D TRAINING DATA WITH CLUSTERED SAMPLING ===")

    # Load training data
    logging.info(f"Loading training data from {CONFIG['DATA_FILE_TRAIN']}")
    df_train = pd.read_csv(CONFIG['DATA_FILE_TRAIN'])
    logging.info(f"Original training samples: {len(df_train)}")

    # Drop NaNs
    if df_train.isnull().values.any():
        nan_count = df_train.isnull().sum().sum()
        logging.warning(f"Found {nan_count} NaN values in training dataset. Dropping rows with NaNs.")
        df_train.dropna(inplace=True)

    # Add location column
    df_train['location'] = df_train['response_point']
    param_features = CONFIG['PARAM_COLS'] + ['location']

    # Detect time columns dynamically
    time_cols = detect_time_columns(df_train, "training")
    CONFIG['TIME_COLS'] = time_cols  # Set globally for consistency
    
    # Extract parameters for clustering
    X_full = df_train[param_features].values
    Y_full = df_train[time_cols].values

    logging.info(f"Full training data shape: X: {X_full.shape}, Y: {Y_full.shape}")

    # Apply K-means clustering for sample selection
    logging.info(f"Applying K-means clustering with {CONFIG['NUM_CLUSTERS']} clusters...")
    kmeans = KMeans(n_clusters=CONFIG['NUM_CLUSTERS'], random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_full)

    # Imbalanced cluster sampling: reflect industrial bias toward "safe" configurations
    # For k=2: 70 samples from cluster 0, 30 from cluster 1
    if CONFIG['NUM_CLUSTERS'] == 2:
        samples_distribution = [70, 30]
    elif CONFIG['NUM_CLUSTERS'] == 3:
        samples_distribution = [50, 30, 20]
    else:
        # Fallback to equal distribution
        samples_per_cluster = CONFIG['NUM_TRAIN_SAMPLES'] // CONFIG['NUM_CLUSTERS']
        samples_distribution = [samples_per_cluster] * CONFIG['NUM_CLUSTERS']

    logging.info(f"Imbalanced sampling distribution: {samples_distribution}")
    selected_indices = []

    for cluster_id in range(CONFIG['NUM_CLUSTERS']):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        n_samples = samples_distribution[cluster_id]
        logging.info(f"Cluster {cluster_id}: {len(cluster_indices)} available, selecting {n_samples} samples")

        # Randomly select n_samples from this cluster
        if len(cluster_indices) >= n_samples:
            selected = np.random.choice(cluster_indices, size=n_samples, replace=False)
        else:
            selected = cluster_indices
            logging.warning(f"Cluster {cluster_id} has fewer samples than requested")

        selected_indices.extend(selected)

    # Ensure exactly NUM_TRAIN_SAMPLES
    remaining = CONFIG['NUM_TRAIN_SAMPLES'] - len(selected_indices)
    if remaining > 0:
        available = list(set(range(len(X_full))) - set(selected_indices))
        additional = np.random.choice(available, size=remaining, replace=False)
        selected_indices.extend(additional)
        logging.info(f"Added {remaining} additional samples to reach target")

    selected_indices = np.array(selected_indices[:CONFIG['NUM_TRAIN_SAMPLES']])

    logging.info(f"Selected {len(selected_indices)} samples using clustered sampling")

    # Extract selected samples
    X_train = X_full[selected_indices]
    Y_train = Y_full[selected_indices]

    logging.info(f"HFSM Training data shape: X: {X_train.shape}, Y: {Y_train.shape}")
    logging.info(f"Training Y stats - Min: {Y_train.min():.6f}, Max: {Y_train.max():.6f}, Mean: {Y_train.mean():.6f}")

    return X_train, Y_train

def load_test_data():
    """Load full 2D test data"""
    logging.info("=== LOADING 2D TEST DATA ===")

    logging.info(f"Loading test data from {CONFIG['DATA_FILE_TEST']}")
    df_test = pd.read_csv(CONFIG['DATA_FILE_TEST'])
    logging.info(f"Original test samples: {len(df_test)}")

    # Drop NaNs
    if df_test.isnull().values.any():
        nan_count = df_test.isnull().sum().sum()
        logging.warning(f"Found {nan_count} NaN values in test dataset. Dropping rows with NaNs.")
        df_test.dropna(inplace=True)

    # Add location column
    df_test['location'] = df_test['response_point']
    param_features = CONFIG['PARAM_COLS'] + ['location']

    # Detect time columns dynamically
    time_cols_test = detect_time_columns(df_test, "test")
    
    # Ensure test data uses same format as training data
    if CONFIG['TIME_COLS'] is not None and time_cols_test != CONFIG['TIME_COLS']:
        logging.warning(f"Time column formats differ between train and test datasets!")
        logging.warning(f"Train format: {CONFIG['TIME_COLS'][0]} ... {CONFIG['TIME_COLS'][-1]}")
        logging.warning(f"Test format: {time_cols_test[0]} ... {time_cols_test[-1]}")
        
        # Use the format from training data and map test columns accordingly
        logging.info("Mapping test dataset columns to match training format...")
        
        # Create mapping from test format to train format
        col_mapping = {}
        for train_col in CONFIG['TIME_COLS']:
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
        df_test = df_test.rename(columns=col_mapping)
        time_cols_test = CONFIG['TIME_COLS']  # Now they match
        logging.info("Successfully mapped test dataset columns to training format")
    elif CONFIG['TIME_COLS'] is None:
        # If training data hasn't been loaded yet, use test format
        CONFIG['TIME_COLS'] = time_cols_test
        logging.info("Using test data format for time columns")

    # Extract features and responses
    X_test = df_test[param_features].values
    Y_test = df_test[CONFIG['TIME_COLS']].values

    logging.info(f"Test data shape: X: {X_test.shape}, Y: {Y_test.shape}")
    logging.info(f"Test Y stats - Min: {Y_test.min():.6f}, Max: {Y_test.max():.6f}, Mean: {Y_test.mean():.6f}")

    return X_test, Y_test

def train_cae_model(cae, train_loader, epochs, learning_rate, model_save_name):
    logging.info(f"--- Starting CAE Training on {CONFIG['DEVICE']} for {epochs} epochs ---")
    logging.info(f"Model will be saved as: {model_save_name}")

    cae.to(CONFIG['DEVICE'])

    optimizer = optim.Adam(cae.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_train_loss = float('inf')
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
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Save best model based on training loss (no validation set)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(cae.state_dict(), best_model_path)
            patience_counter = 0
            if (epoch + 1) % 10 == 0:
                logging.info(f"New best model at epoch {epoch+1} with train loss: {best_train_loss:.8f}")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            # Evaluate autoencoder reconstruction quality every 10 epochs
            ae_recon_metrics = evaluate_ae_reconstruction(cae, train_loader, "Training")
            logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.8f}, AE R²={ae_recon_metrics['r2_overall']:.6f}")

        # Early stopping
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logging.info(f"--- CAE Training Complete. Model saved to {best_model_path} ---")
    return best_model_path

def get_latent_vectors(encoder, dataloader):
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
    """
    Calculate Normalized Mean Squared Error (NMSE) as percentage
    """
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

    Args:
        y_true: Ground truth responses (n_samples, n_timesteps)
        y_pred: Predicted responses (n_samples, n_timesteps)
        locations: Response point locations (n_samples,)

    Returns:
        r2_roi: R² score computed only on ROI regions
        r2_roi_per_sample: Per-sample R² scores for analysis
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
    """
    Calculate NMSE focusing ONLY on Region of Interest (ROI) timesteps.
    """
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

    # Calculate ROI metrics (true performance on dynamic regions)
    r2_roi, r2_roi_per_sample = calculate_r2_roi(Y_data, Y_pred, locations)
    nmse_roi = calculate_nmse_roi(Y_data, Y_pred, locations)

    logging.info(f"{dataset_name} Evaluation Results:")
    logging.info(f"  Latent Space R²: {r2_latent:.4f}")
    logging.info(f"  === FULL METRICS (inflated by zeros/baseline) ===")
    logging.info(f"  Time Series R² (Full): {r2_full:.4f}")
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
        'r2_roi_per_sample': r2_roi_per_sample,
        'mse_full': mse_full,
        'nmse_full': nmse_full,
        'nmse_roi': nmse_roi,
        'predictions': Y_pred
    }

def predict_timeseries_from_params(cae, surrogate_model, params_scaled):
    cae.decoder.to(CONFIG['DEVICE'])
    cae.decoder.eval()
    with torch.no_grad():
        Z_pred = surrogate_model.predict(params_scaled)
        Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(CONFIG['DEVICE'])
        Y_pred = cae.decoder(Z_pred_tensor).cpu().numpy()
    return Y_pred

def dump_lfsm_interleaved_predictions(file_path, ground_truth, predictions, parameters=None):
    """
    Create interleaved CSV file where each ground truth response is followed by its HFSM prediction.
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

        # HFSM prediction row
        pred_row = []
        if parameters is not None:
            pred_row.extend(parameters[i])
        pred_row.extend(predictions[i])
        pred_row.append('hfsm_prediction')
        interleaved_data.append(pred_row)

    # Create DataFrame and save
    df = pd.DataFrame(interleaved_data, columns=all_cols)
    df.to_csv(file_path, index=False)

    logging.info(f"Saved interleaved HFSM predictions: {file_path} with shape {df.shape}")
    logging.info(f"Format: {n_samples} parameter sets, each with ground_truth + hfsm_prediction rows")

def create_comparison_plots(ground_truth, predictions, parameters, output_dir, dataset_name="HFSM_Test"):
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
        plt.plot(time_axis, predictions[i], 'r--', label='HFSM Prediction', linewidth=2)

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
        plt.plot(time_axis, predictions[idx], 'r--', label='HFSM Prediction', linewidth=1.5)

        # Highlight ROI region
        location = locations[idx]
        roi_start, roi_end = get_roi_for_location(location)
        plt.axvspan(roi_start, roi_end, alpha=0.1, color='green', label='ROI')

        plt.title(f'Best #{i+1}: R²_ROI={r2_scores_roi[idx]:.3f}, R²_full={r2_scores_full[idx]:.3f}\nLoc={location:.2f}, Params: {parameters[idx][:3]}...', fontsize=9)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()

    plt.suptitle(f'{dataset_name}: 10 Best Predictions (ranked by ROI R²)', fontsize=16, y=1.02)
    best_plot_path = os.path.join(output_dir, f'hfsm_comparison_plots_best_10_{dataset_name.lower()}.png')
    plt.savefig(best_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved best predictions plot: {best_plot_path}")

    # Plot 10 worst predictions
    plt.figure(figsize=(20, 12))
    for i, idx in enumerate(worst_indices):
        plt.subplot(2, 5, i+1)
        plt.plot(time_axis, ground_truth[idx], 'b-', label='Ground Truth', linewidth=1.5)
        plt.plot(time_axis, predictions[idx], 'r--', label='HFSM Prediction', linewidth=1.5)

        # Highlight ROI region
        location = locations[idx]
        roi_start, roi_end = get_roi_for_location(location)
        plt.axvspan(roi_start, roi_end, alpha=0.1, color='red', label='ROI')

        plt.title(f'Worst #{i+1}: R²_ROI={r2_scores_roi[idx]:.3f}, R²_full={r2_scores_full[idx]:.3f}\nLoc={location:.2f}, Params: {parameters[idx][:3]}...', fontsize=9)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()

    plt.suptitle(f'{dataset_name}: 10 Worst Predictions (ranked by ROI R²)', fontsize=16, y=1.02)
    worst_plot_path = os.path.join(output_dir, f'hfsm_comparison_plots_worst_10_{dataset_name.lower()}.png')
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
    summary_plot_path = os.path.join(output_dir, f'hfsm_comparison_summary_{dataset_name.lower()}.png')
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

def main():
    logging.info("=== STARTING HFSM TRAINING WITH CLUSTERED SAMPLING ===")
    logging.info(f"Training on {CONFIG['NUM_TRAIN_SAMPLES']} samples using {CONFIG['NUM_CLUSTERS']}-cluster sampling")

    # ===== PHASE 1: DATA LOADING WITH CLUSTERED SAMPLING =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 1: LOADING DATA WITH CLUSTERED SAMPLING")
    logging.info("="*60)

    # Load and cluster-sample training data
    X_train, Y_train = load_and_cluster_sample_data()

    # Load full test data
    X_test, Y_test = load_test_data()

    # ===== PHASE 2: CAE TRAINING =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 2: CAE TRAINING ON CLUSTERED HFSM DATA")
    logging.info("="*60)

    # Create datasets and dataloaders with training noise for regularization
    train_dataset = BeamResponseDataset(X_train, Y_train, add_noise=True, noise_std=0.03)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['CAE_BATCH_SIZE'], shuffle=True, drop_last=True)
    train_loader_full = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    # Initialize and train CAE
    cae = Autoencoder(
        timeseries_dim=CONFIG['NUM_TIME_STEPS'],
        latent_dim=CONFIG['LATENT_DIM']
    )

    best_model_path = train_cae_model(
        cae, train_loader,
        CONFIG['CAE_EPOCHS'], CONFIG['CAE_LEARNING_RATE'],
        'hfsm_cae_model.pth'
    )

    # Load best model
    logging.info(f"Loading best CAE model from {best_model_path}")
    cae.load_state_dict(torch.load(best_model_path, map_location=CONFIG['DEVICE']))

    # ===== PHASE 3: XGBOOST SURROGATE TRAINING =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 3: XGBOOST SURROGATE MODEL TRAINING")
    logging.info("="*60)

    # Extract latent vectors
    Z_train = get_latent_vectors(cae.encoder, train_loader_full)
    logging.info(f"Extracted latent vectors. Train shape: {Z_train.shape}")

    # Train XGBoost surrogate model
    logging.info("--- Training XGBoost Surrogate Model ---")
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

    surrogate_model = xgb.XGBRegressor(**xgb_params, n_jobs=-1)
    surrogate_model.fit(train_dataset.params_scaled, Z_train, verbose=False)
    logging.info("--- HFSM Surrogate Model Training Complete ---")

    # Save models and scaler
    joblib.dump(surrogate_model, os.path.join(CONFIG['OUTPUT_DIR'], 'hfsm_surrogate.joblib'))
    joblib.dump(train_dataset.p_scaler, os.path.join(CONFIG['OUTPUT_DIR'], 'hfsm_scaler.joblib'))

    # ===== PHASE 4: EVALUATION =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 4: EVALUATION ON TEST DATA")
    logging.info("="*60)

    # Evaluate on test data
    results_test = evaluate_on_dataset(cae, surrogate_model, train_dataset.p_scaler,
                                      X_test, Y_test, "HFSM_Test")

    # ===== PHASE 5: GENERATE OUTPUTS =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 5: GENERATING OUTPUTS AND VISUALIZATIONS")
    logging.info("="*60)

    # Build test dataset for predictions
    test_dataset = BeamResponseDataset(X_test, Y_test, train_dataset.p_scaler)
    Y_pred_test = predict_timeseries_from_params(cae, surrogate_model, test_dataset.params_scaled)

    # Save interleaved predictions
    dump_lfsm_interleaved_predictions(
        os.path.join(CONFIG['OUTPUT_DIR'], 'hfsm_interleaved_test.csv'),
        Y_test, Y_pred_test, X_test
    )

    # Save predictions as numpy array
    np.save(os.path.join(CONFIG['OUTPUT_DIR'], 'hfsm_predictions_test.npy'), Y_pred_test)

    # Create comparison plots
    plot_stats = create_comparison_plots(
        Y_test, Y_pred_test, X_test,
        CONFIG['OUTPUT_DIR'], dataset_name="HFSM_Test"
    )

    # ===== PHASE 3.5: FINAL AUTOENCODER RECONSTRUCTION EVALUATION =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 3.5: FINAL AUTOENCODER RECONSTRUCTION EVALUATION")
    logging.info("="*60)

    # Evaluate autoencoder reconstruction on training data
    ae_recon_train_metrics = evaluate_ae_reconstruction(cae, train_loader_full, "HFSM Train (AE Only)")

    # ===== FINAL SUMMARY =====
    logging.info("\n" + "="*60)
    logging.info("FINAL SUMMARY")
    logging.info("="*60)

    logging.info(f"HFSM Test Results:")
    logging.info(f"  R² (Full): {results_test['r2_timeseries_full']:.4f}, NMSE (Full): {results_test['nmse_full']:.4f}%")
    logging.info(f"  R² (ROI): {results_test['r2_timeseries_roi']:.4f} ← PRIMARY METRIC, NMSE (ROI): {results_test['nmse_roi']:.4f}%")
    logging.info(f"Best ROI R² range: {plot_stats['best_r2_roi'].min():.4f} to {plot_stats['best_r2_roi'].max():.4f}")
    logging.info(f"Worst ROI R² range: {plot_stats['worst_r2_roi'].min():.4f} to {plot_stats['worst_r2_roi'].max():.4f}")
    logging.info("")
    logging.info("AUTOENCODER RECONSTRUCTION QUALITY (R²):")
    logging.info(f"HFSM Train (AE Only): R²={ae_recon_train_metrics['r2_overall']:.4f}")

    logging.info("\nFiles saved to Claude_res/:")
    logging.info("  - hfsm_cae_model.pth")
    logging.info("  - hfsm_surrogate.joblib")
    logging.info("  - hfsm_scaler.joblib")
    logging.info("  - hfsm_interleaved_test.csv")
    logging.info("  - hfsm_predictions_test.npy")
    logging.info("  - hfsm_comparison_plots_*.png")

    logging.info("--- HFSM Training Complete ---")
    logging.info("="*60)

if __name__ == '__main__':
    main()
