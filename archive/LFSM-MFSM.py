# Note: This implements LFSM (Low Fidelity Surrogate Model) when trained with 1D data,
# and MFSM (Multi-Fidelity Surrogate Model) when fine-tuned with 2D data.
# The autoencoder is NON-CONDITIONAL (no parameters injected), while XGBoost learns parameter�latent mapping.

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
    'DATA_FILE_TRAIN': '/home/user2/Music/abhi3/parameters/LFSM1000train.csv',
    'DATA_FILE_TEST': '/home/user2/Music/abhi3/parameters/LFSM1000test.csv',
    'DATA_FILE_2D_TRAIN': [
        '/home/user2/Music/abhi3/parameters/train_responses.csv'
    ],
    'DATA_FILE_2D_TEST': '/home/user2/Music/abhi3/parameters/test_responses.csv',
    'OUTPUT_DIR': '/home/user2/Music/abhi3/MFSM/Finetuning',

    # --- Data & Model Hyperparameters ---
    'PARAM_COLS': ['notch_x', 'notch_depth', 'notch_width', 'youngs_modulus', 'density'],
    'NUM_TIME_STEPS': 1500,
    'VAL_SPLIT_RATIO': 0.2,

    # --- CAE Training ---
    'LATENT_DIM': 50,
    'CAE_EPOCHS_1D': 100,
    'CAE_EPOCHS_2D': 200,
    'CAE_BATCH_SIZE': 64,
    'CAE_LEARNING_RATE': 1e-4,
    'CAE_LEARNING_RATE_2D': 5e-5,

    # --- XGBoost Surrogate Model ---
    'XGB_N_ESTIMATORS': 2000,
    'XGB_MAX_DEPTH': 10,
    'XGB_ETA': 0.03,
    'XGB_EARLY_STOPPING': 100,
}
CONFIG['TIME_COLS'] = [f't_{i}' for i in range(1, CONFIG["NUM_TIME_STEPS"] + 1)]

# --- Setup Logging ---
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(CONFIG['OUTPUT_DIR'], 'training_log.log')),
                              logging.StreamHandler()])

# --- PyTorch Dataset Class ---
class BeamResponseDataset(Dataset):
    """
    Dataset for beam response data with parameter scaling.
    Parameters are stored and scaled for XGBoost, but NOT passed to autoencoder.
    Responses are already normalized and left unchanged.
    """
    def __init__(self, params, timeseries, p_scaler=None):
        # Store raw time series without scaling - responses are already normalized
        self.timeseries = timeseries.astype(np.float32)
        self.params = params.astype(np.float32)

        # Scale parameters using StandardScaler for XGBoost
        if p_scaler is None:
            self.p_scaler = StandardScaler()
            self.params_scaled = self.p_scaler.fit_transform(self.params)
        else:
            self.p_scaler = p_scaler
            self.params_scaled = self.p_scaler.transform(self.params)

        # Ensure float32 and contiguous for efficiency
        self.params_scaled = np.ascontiguousarray(self.params_scaled, dtype=np.float32)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return {
            'params': torch.tensor(self.params_scaled[idx], dtype=torch.float32),
            'timeseries': torch.tensor(self.timeseries[idx], dtype=torch.float32),
            'timeseries_raw': self.timeseries[idx]
        }

# --- Non-Conditional Autoencoder PyTorch Models ---
class Encoder(nn.Module):
    """Encoder: Time series � Latent space (NO parameter conditioning)"""
    def __init__(self, timeseries_dim, latent_dim):
        super(Encoder, self).__init__()
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
            nn.Dropout(0.1),
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
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
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

# --- Main Functions ---
def load_1d_data():
    """Load 1D training and test data with parameters from separate CSV files"""
    logging.info("=== LOADING 1D DATA ===")

    df_train = pd.read_csv(CONFIG['DATA_FILE_TRAIN'])
    df_test = pd.read_csv(CONFIG['DATA_FILE_TEST'])

    logging.info(f"1D data loaded: {len(df_train)} train, {len(df_test)} test samples")

    # Check for NaN values
    if df_train.isnull().values.any():
        nan_count = df_train.isnull().sum().sum()
        logging.warning(f"Dropping {nan_count} NaN values from 1D training data")
        df_train.dropna(inplace=True)

    if df_test.isnull().values.any():
        nan_count = df_test.isnull().sum().sum()
        logging.warning(f"Dropping {nan_count} NaN values from 1D test data")
        df_test.dropna(inplace=True)

    # Add location column
    df_train['location'] = df_train['response_point']
    df_test['location'] = df_test['response_point']
    param_features = CONFIG['PARAM_COLS'] + ['location']

    # Extract parameters and responses
    X_train = df_train[param_features].values
    Y_train = df_train[CONFIG['TIME_COLS']].values
    X_test_full = df_test[param_features].values
    Y_test_full = df_test[CONFIG['TIME_COLS']].values

    # Sample validation set from test data
    val_size = int(len(X_test_full) * CONFIG['VAL_SPLIT_RATIO'])
    np.random.seed(42)
    val_indices = np.random.choice(len(X_test_full), size=val_size, replace=False)

    X_val = X_test_full[val_indices]
    Y_val = Y_test_full[val_indices]

    logging.info(f"1D split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test_full)}")

    return X_train, X_val, X_test_full, Y_train, Y_val, Y_test_full

def load_2d_data():
    """Load 2D training and test data with parameters from separate CSV files"""
    logging.info("=== LOADING 2D DATA ===")

    # Load 2D training data
    train_files = CONFIG['DATA_FILE_2D_TRAIN']
    if isinstance(train_files, list):
        dfs = []
        for f in train_files:
            df = pd.read_csv(f)
            # Standardize column names
            time_cols = [col for col in df.columns if col.startswith('r') and (col[1:].isdigit() or (col[2:].isdigit() and col[1] == '_'))]
            if time_cols:
                new_time_cols = {}
                for col in time_cols:
                    if col[1] == '_':
                        new_col = 'r' + col[2:]
                    else:
                        new_col = col
                    new_time_cols[col] = new_col
                df = df.rename(columns=new_time_cols)
            dfs.append(df)
        df_2d_train = pd.concat(dfs, ignore_index=True)
    else:
        df_2d_train = pd.read_csv(train_files)

    # Load 2D test data
    df_2d_test = pd.read_csv(CONFIG['DATA_FILE_2D_TEST'])
    time_cols = [col for col in df_2d_test.columns if col.startswith('r') and (col[1:].isdigit() or (col[2:].isdigit() and col[1] == '_'))]
    if time_cols:
        new_time_cols = {}
        for col in time_cols:
            if col[1] == '_':
                new_col = 'r' + col[2:]
            else:
                new_col = col
            new_time_cols[col] = new_col
        df_2d_test = df_2d_test.rename(columns=new_time_cols)

    logging.info(f"2D data loaded: {len(df_2d_train)} train, {len(df_2d_test)} test samples")

    # Check for NaN values
    if df_2d_train.isnull().values.any():
        nan_count = df_2d_train.isnull().sum().sum()
        logging.warning(f"Dropping {nan_count} NaN values from 2D training data")
        df_2d_train.dropna(inplace=True)

    if df_2d_test.isnull().values.any():
        nan_count = df_2d_test.isnull().sum().sum()
        logging.warning(f"Dropping {nan_count} NaN values from 2D test data")
        df_2d_test.dropna(inplace=True)

    # Add location column
    df_2d_train['location'] = df_2d_train['response_point']
    df_2d_test['location'] = df_2d_test['response_point']
    param_features = CONFIG['PARAM_COLS'] + ['location']

    # Find time columns
    def find_time_cols(df):
        time_cols = [col for col in df.columns if col.startswith('r') and (col[1:].isdigit() or (col[2:].isdigit() and col[1] == '_'))]
        return sorted(time_cols, key=lambda x: int(x.split('_')[-1]) if '_' in x else int(x[1:]))

    time_cols_2d_train = find_time_cols(df_2d_train)
    time_cols_2d_test = find_time_cols(df_2d_test)

    if len(time_cols_2d_train) == 0:
        raise ValueError("No time series columns found in 2D training dataset")
    if len(time_cols_2d_test) == 0:
        raise ValueError("No time series columns found in 2D test dataset")

    # Take only first 1500 time steps
    if len(time_cols_2d_train) > CONFIG['NUM_TIME_STEPS']:
        time_cols_2d_train = time_cols_2d_train[:CONFIG['NUM_TIME_STEPS']]
    if len(time_cols_2d_test) > CONFIG['NUM_TIME_STEPS']:
        time_cols_2d_test = time_cols_2d_test[:CONFIG['NUM_TIME_STEPS']]

    # Extract parameters and responses
    X_2d_train = df_2d_train[param_features].values
    Y_2d_train = df_2d_train[time_cols_2d_train].values
    X_2d_test_full = df_2d_test[param_features].values
    Y_2d_test_full = df_2d_test[time_cols_2d_test].values

    # Sample validation set
    val_size = int(len(X_2d_test_full) * CONFIG['VAL_SPLIT_RATIO'])
    np.random.seed(42)
    val_indices = np.random.choice(len(X_2d_test_full), size=val_size, replace=False)

    X_2d_val = X_2d_test_full[val_indices]
    Y_2d_val = Y_2d_test_full[val_indices]

    logging.info(f"2D split: Train={len(X_2d_train)}, Val={len(X_2d_val)}, Test={len(X_2d_test_full)}")

    return X_2d_train, X_2d_val, X_2d_test_full, Y_2d_train, Y_2d_val, Y_2d_test_full

def train_ae_model(ae, train_loader, val_loader, epochs, learning_rate, model_save_name, load_pretrained=None):
    """Train autoencoder (parameters extracted from batch but NOT passed to AE)"""
    logging.info(f"--- Training Autoencoder on {CONFIG['DEVICE']} for {epochs} epochs ---")

    ae.to(CONFIG['DEVICE'])

    if load_pretrained:
        logging.info(f"Loading pretrained weights from {load_pretrained}")
        ae.load_state_dict(torch.load(load_pretrained, map_location=CONFIG['DEVICE']))

    optimizer = optim.Adam(ae.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_path = os.path.join(CONFIG['OUTPUT_DIR'], model_save_name)

    for epoch in range(epochs):
        # Training
        ae.train()
        total_train_loss = 0
        for batch in train_loader:
            # Extract both params and timeseries from batch
            params = batch['params'].to(CONFIG['DEVICE'])
            timeseries = batch['timeseries'].to(CONFIG['DEVICE'])

            # BUT only pass timeseries to autoencoder (no parameters)
            optimizer.zero_grad()
            recon_ts, _ = ae(timeseries)
            loss = criterion(recon_ts, timeseries)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        if val_loader is not None:
            # Validation
            ae.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    timeseries = batch['timeseries'].to(CONFIG['DEVICE'])
                    recon_ts, _ = ae(timeseries)
                    loss = criterion(recon_ts, timeseries)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(ae.state_dict(), best_model_path)

            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}] Train={avg_train_loss:.8f}, Val={avg_val_loss:.8f}")
        else:
            torch.save(ae.state_dict(), best_model_path)
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}] Train={avg_train_loss:.8f}")

    logging.info(f"Training complete. Model saved to {best_model_path}")
    return best_model_path

def get_latent_vectors(encoder, dataloader):
    """Extract latent vectors (parameters extracted but NOT passed to encoder)"""
    encoder.to(CONFIG['DEVICE'])
    encoder.eval()
    all_latents = []
    with torch.no_grad():
        for batch in dataloader:
            # Extract both but only use timeseries
            params = batch['params'].to(CONFIG['DEVICE'])
            timeseries = batch['timeseries'].to(CONFIG['DEVICE'])

            # Only pass timeseries to encoder
            latents = encoder(timeseries)
            all_latents.append(latents.cpu().numpy())
    return np.vstack(all_latents)

def calculate_nmse(y_true, y_pred):
    """Calculate Normalized Mean Squared Error as percentage"""
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

def evaluate_on_dataset(ae, surrogate_model, params_scaler, X_data, Y_data, dataset_name):
    """Evaluate model: XGBoost predicts latent, decoder reconstructs"""
    logging.info(f"--- Evaluating on {dataset_name} ---")

    eval_dataset = BeamResponseDataset(X_data, Y_data, params_scaler)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

    # Get true latent vectors from encoder
    Z_true = get_latent_vectors(ae.encoder, eval_loader)

    # Predict latent vectors using XGBoost
    Z_pred = surrogate_model.predict(eval_dataset.params_scaled)
    r2_latent = r2_score(Z_true, Z_pred)

    # Reconstruct time series from predicted latents (no parameters)
    ae.decoder.to(CONFIG['DEVICE'])
    ae.decoder.eval()
    with torch.no_grad():
        Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(CONFIG['DEVICE'])
        Y_pred = ae.decoder(Z_pred_tensor).cpu().numpy()

    # Calculate metrics
    mse = mean_squared_error(Y_data.reshape(-1), Y_pred.reshape(-1))
    r2 = r2_score(Y_data.reshape(-1), Y_pred.reshape(-1))
    nmse = calculate_nmse(Y_data, Y_pred)

    logging.info(f"{dataset_name}: Latent R�={r2_latent:.4f}, TS R�={r2:.4f}, MSE={mse:.6f}, NMSE={nmse:.4f}%")

    return {
        'r2_latent': r2_latent,
        'r2_timeseries': r2,
        'mse': mse,
        'nmse': nmse,
        'predictions': Y_pred
    }

def evaluate_cross_domain(ae, surrogate_model, params_scaler, X_2d_params, Y_2d_responses, dataset_name):
    """Cross-domain evaluation: 2D params with 1D-trained model"""
    logging.info(f"--- Cross-Domain: {dataset_name} ---")

    eval_dataset = BeamResponseDataset(X_2d_params, Y_2d_responses, params_scaler)

    # Predict latent vectors using XGBoost
    Z_pred = surrogate_model.predict(eval_dataset.params_scaled)

    # Reconstruct responses (no parameters)
    ae.decoder.to(CONFIG['DEVICE'])
    ae.decoder.eval()
    with torch.no_grad():
        Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(CONFIG['DEVICE'])
        Y_pred = ae.decoder(Z_pred_tensor).cpu().numpy()

    mse = mean_squared_error(Y_2d_responses.reshape(-1), Y_pred.reshape(-1))
    r2 = r2_score(Y_2d_responses.reshape(-1), Y_pred.reshape(-1))
    nmse = calculate_nmse(Y_2d_responses, Y_pred)

    logging.info(f"{dataset_name}: R�={r2:.4f}, MSE={mse:.6f}, NMSE={nmse:.4f}%")

    return {
        'r2_timeseries': r2,
        'mse': mse,
        'nmse': nmse,
        'predictions': Y_pred
    }

def predict_timeseries_from_params(ae, surrogate_model, params_scaled):
    """Predict time series from parameters via XGBoost�Decoder"""
    ae.decoder.to(CONFIG['DEVICE'])
    ae.decoder.eval()
    with torch.no_grad():
        Z_pred = surrogate_model.predict(params_scaled)
        Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(CONFIG['DEVICE'])
        Y_pred = ae.decoder(Z_pred_tensor).cpu().numpy()
    return Y_pred

def dump_predictions(file_path, predictions):
    """Save predictions to CSV"""
    n_steps = predictions.shape[1]
    cols = CONFIG['TIME_COLS'][:n_steps]
    df = pd.DataFrame(predictions, columns=cols)
    df.to_csv(file_path, index=False)
    logging.info(f"Saved predictions: {file_path} {predictions.shape}")

def dump_interleaved_predictions(file_path, ground_truth, predictions, parameters=None):
    """Save interleaved ground truth and predictions"""
    n_samples, n_timesteps = ground_truth.shape
    time_cols = CONFIG['TIME_COLS'][:n_timesteps]

    param_cols = []
    if parameters is not None:
        param_cols = CONFIG['PARAM_COLS'] + ['location']

    all_cols = param_cols + time_cols + ['data_type']
    interleaved_data = []

    for i in range(n_samples):
        # Ground truth row
        gt_row = []
        if parameters is not None:
            gt_row.extend(parameters[i])
        gt_row.extend(ground_truth[i])
        gt_row.append('ground_truth')
        interleaved_data.append(gt_row)

        # Prediction row
        pred_row = []
        if parameters is not None:
            pred_row.extend(parameters[i])
        pred_row.extend(predictions[i])
        pred_row.append('prediction')
        interleaved_data.append(pred_row)

    df = pd.DataFrame(interleaved_data, columns=all_cols)
    df.to_csv(file_path, index=False)
    logging.info(f"Saved interleaved: {file_path} {df.shape}")

def create_comparison_plots(ground_truth, predictions, parameters, output_dir, dataset_name="2D"):
    """Create comparison plots for best/worst 10 predictions"""
    logging.info(f"Creating plots for {dataset_name}...")

    n_samples, n_timesteps = ground_truth.shape

    # Calculate R� for each sample
    r2_scores = []
    for i in range(n_samples):
        try:
            r2 = r2_score(ground_truth[i], predictions[i])
            r2_scores.append(r2)
        except:
            r2_scores.append(-np.inf)

    r2_scores = np.array(r2_scores)

    # Find best and worst indices
    best_indices = np.argsort(r2_scores)[-10:][::-1]
    worst_indices = np.argsort(r2_scores)[:10]

    time_axis = np.arange(n_timesteps)

    # Plot 10 best
    plt.figure(figsize=(20, 12))
    for i, idx in enumerate(best_indices):
        plt.subplot(2, 5, i+1)
        plt.plot(time_axis, ground_truth[idx], 'b-', label='Ground Truth', linewidth=1.5)
        plt.plot(time_axis, predictions[idx], 'r--', label='Prediction', linewidth=1.5)
        plt.title(f'Best #{i+1} (R�={r2_scores[idx]:.4f})', fontsize=10)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

    plt.suptitle(f'{dataset_name}: 10 Best Predictions', fontsize=16, y=1.02)
    plt.savefig(os.path.join(output_dir, f'best_10_{dataset_name.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 10 worst
    plt.figure(figsize=(20, 12))
    for i, idx in enumerate(worst_indices):
        plt.subplot(2, 5, i+1)
        plt.plot(time_axis, ground_truth[idx], 'b-', label='Ground Truth', linewidth=1.5)
        plt.plot(time_axis, predictions[idx], 'r--', label='Prediction', linewidth=1.5)
        plt.title(f'Worst #{i+1} (R�={r2_scores[idx]:.4f})', fontsize=10)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

    plt.suptitle(f'{dataset_name}: 10 Worst Predictions', fontsize=16, y=1.02)
    plt.savefig(os.path.join(output_dir, f'worst_10_{dataset_name.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Plots saved for {dataset_name}")

    return {'best_r2': r2_scores[best_indices], 'worst_r2': r2_scores[worst_indices]}

def main():
    logging.info("=== STARTING LFSM�MFSM TRAINING ===")

    # ===== PHASE 1: 1D TRAINING (LFSM) =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 1: LFSM TRAINING ON 1D DATA")
    logging.info("="*60)

    # Load 1D data
    X_train_1d, X_val_1d, X_test_1d, Y_train_1d, Y_val_1d, Y_test_1d = load_1d_data()

    # Create datasets
    train_dataset_1d = BeamResponseDataset(X_train_1d, Y_train_1d)
    val_dataset_1d = BeamResponseDataset(X_val_1d, Y_val_1d, train_dataset_1d.p_scaler)
    test_dataset_1d = BeamResponseDataset(X_test_1d, Y_test_1d, train_dataset_1d.p_scaler)

    train_loader_1d = DataLoader(train_dataset_1d, batch_size=CONFIG['CAE_BATCH_SIZE'], shuffle=True, drop_last=True)
    val_loader_1d = DataLoader(val_dataset_1d, batch_size=CONFIG['CAE_BATCH_SIZE'], shuffle=False)
    train_loader_1d_full = DataLoader(train_dataset_1d, batch_size=len(train_dataset_1d), shuffle=False)

    # Train autoencoder
    ae = Autoencoder(
        timeseries_dim=CONFIG['NUM_TIME_STEPS'],
        latent_dim=CONFIG['LATENT_DIM']
    )
    best_model_1d_path = train_ae_model(
        ae, train_loader_1d, val_loader_1d,
        CONFIG['CAE_EPOCHS_1D'], CONFIG['CAE_LEARNING_RATE'],
        'best_ae_model_1d.pth'
    )

    ae.load_state_dict(torch.load(best_model_1d_path, map_location=CONFIG['DEVICE']))

    # Extract latent vectors
    Z_train_1d = get_latent_vectors(ae.encoder, train_loader_1d_full)
    Z_val_1d = get_latent_vectors(ae.encoder, DataLoader(val_dataset_1d, batch_size=len(val_dataset_1d), shuffle=False))

    logging.info(f"Latent vectors: Train={Z_train_1d.shape}, Val={Z_val_1d.shape}")

    # Train XGBoost
    logging.info("Training XGBoost surrogate...")
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': CONFIG['XGB_N_ESTIMATORS'],
        'max_depth': CONFIG['XGB_MAX_DEPTH'],
        'eta': CONFIG['XGB_ETA'],
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0,
    }
    if CONFIG['USE_XGB_GPU'] and CONFIG['DEVICE'] == 'cuda':
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['predictor'] = 'gpu_predictor'
    else:
        xgb_params['tree_method'] = 'hist'

    surrogate_model_1d = xgb.XGBRegressor(**xgb_params, n_jobs=-1, early_stopping_rounds=50, eval_metric='rmse')
    surrogate_model_1d.fit(
        train_dataset_1d.params_scaled, Z_train_1d,
        eval_set=[(val_dataset_1d.params_scaled, Z_val_1d)],
        verbose=False
    )

    # Evaluate
    results_1d_test = evaluate_on_dataset(ae, surrogate_model_1d, train_dataset_1d.p_scaler, X_test_1d, Y_test_1d, "1D Test")

    # Save models
    joblib.dump(surrogate_model_1d, os.path.join(CONFIG['OUTPUT_DIR'], 'surrogate_model_1d.joblib'))
    joblib.dump(train_dataset_1d.p_scaler, os.path.join(CONFIG['OUTPUT_DIR'], 'params_scaler.joblib'))
    torch.save(ae.state_dict(), os.path.join(CONFIG['OUTPUT_DIR'], 'ae_model_1d.pth'))

    # ===== PHASE 2: CROSS-DOMAIN EVALUATION =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 2: CROSS-DOMAIN EVALUATION")
    logging.info("="*60)

    # Load 2D data
    X_train_2d, X_val_2d, X_test_2d, Y_train_2d, Y_val_2d, Y_test_2d = load_2d_data()

    # Cross-domain: 1D model on 2D data
    results_cross = evaluate_cross_domain(ae, surrogate_model_1d, train_dataset_1d.p_scaler, X_test_2d, Y_test_2d, "1D�2D Cross-Domain")

    # ===== PHASE 3: FINE-TUNING (MFSM) =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 3: MFSM FINE-TUNING ON COMBINED DATA")
    logging.info("="*60)

    # Create weighted combined dataset
    weight_multiplier = 6
    X_combined = np.concatenate([X_train_1d] + [X_train_2d] * weight_multiplier)
    Y_combined = np.concatenate([Y_train_1d] + [Y_train_2d] * weight_multiplier)
    X_val_combined = np.concatenate([X_val_1d] + [X_val_2d] * weight_multiplier)
    Y_val_combined = np.concatenate([Y_val_1d] + [Y_val_2d] * weight_multiplier)

    logging.info(f"Combined: Train={X_combined.shape}, Val={X_val_combined.shape}")

    train_dataset_combined = BeamResponseDataset(X_combined, Y_combined, train_dataset_1d.p_scaler)
    val_dataset_combined = BeamResponseDataset(X_val_combined, Y_val_combined, train_dataset_1d.p_scaler)

    train_loader_combined = DataLoader(train_dataset_combined, batch_size=CONFIG['CAE_BATCH_SIZE'], shuffle=True, drop_last=True)
    val_loader_combined = DataLoader(val_dataset_combined, batch_size=CONFIG['CAE_BATCH_SIZE'], shuffle=False)

    # Fine-tune
    best_model_2d_path = train_ae_model(
        ae, train_loader_combined, val_loader_combined,
        CONFIG['CAE_EPOCHS_2D'], CONFIG['CAE_LEARNING_RATE_2D'],
        'best_ae_model_2d.pth',
        load_pretrained=best_model_1d_path
    )

    ae.load_state_dict(torch.load(best_model_2d_path, map_location=CONFIG['DEVICE']))

    # Re-train XGBoost on combined latent space
    train_loader_combined_full = DataLoader(train_dataset_combined, batch_size=len(train_dataset_combined), shuffle=False)
    val_loader_combined_full = DataLoader(val_dataset_combined, batch_size=len(val_dataset_combined), shuffle=False)
    Z_train_combined = get_latent_vectors(ae.encoder, train_loader_combined_full)
    Z_val_combined = get_latent_vectors(ae.encoder, val_loader_combined_full)

    logging.info(f"Combined latent vectors: {Z_train_combined.shape}")

    surrogate_model_combined = xgb.XGBRegressor(**xgb_params, n_jobs=-1, early_stopping_rounds=50, eval_metric='rmse')
    surrogate_model_combined.fit(
        train_dataset_combined.params_scaled, Z_train_combined,
        eval_set=[(val_dataset_combined.params_scaled, Z_val_combined)],
        verbose=False
    )

    # ===== PHASE 4: FINAL EVALUATION =====
    logging.info("\n" + "="*60)
    logging.info("PHASE 4: FINAL EVALUATION")
    logging.info("="*60)

    results_1d_final = evaluate_on_dataset(ae, surrogate_model_combined, train_dataset_1d.p_scaler, X_test_1d, Y_test_1d, "1D Test (MFSM)")
    results_2d_final = evaluate_on_dataset(ae, surrogate_model_combined, train_dataset_1d.p_scaler, X_test_2d, Y_test_2d, "2D Test (MFSM)")

    # Generate predictions
    test_dataset_2d = BeamResponseDataset(X_test_2d, Y_test_2d, train_dataset_1d.p_scaler)
    train_dataset_2d_dump = BeamResponseDataset(X_train_2d, Y_train_2d, train_dataset_1d.p_scaler)

    Y_pred_2d_test = predict_timeseries_from_params(ae, surrogate_model_combined, test_dataset_2d.params_scaled)
    Y_pred_2d_train = predict_timeseries_from_params(ae, surrogate_model_combined, train_dataset_2d_dump.params_scaled)

    # Save predictions
    dump_interleaved_predictions(os.path.join(CONFIG['OUTPUT_DIR'], 'predictions_2d_test_interleaved.csv'), Y_test_2d, Y_pred_2d_test, X_test_2d)
    dump_interleaved_predictions(os.path.join(CONFIG['OUTPUT_DIR'], 'predictions_2d_train_interleaved.csv'), Y_train_2d, Y_pred_2d_train, X_train_2d)
    dump_predictions(os.path.join(CONFIG['OUTPUT_DIR'], 'predictions_2d_test.csv'), Y_pred_2d_test)
    dump_predictions(os.path.join(CONFIG['OUTPUT_DIR'], 'predictions_2d_train.csv'), Y_pred_2d_train)

    # Create plots
    create_comparison_plots(Y_test_2d, Y_pred_2d_test, X_test_2d, CONFIG['OUTPUT_DIR'], "2D_Test")
    create_comparison_plots(Y_train_2d, Y_pred_2d_train, X_train_2d, CONFIG['OUTPUT_DIR'], "2D_Train")

    # Save final models
    joblib.dump(surrogate_model_combined, os.path.join(CONFIG['OUTPUT_DIR'], 'surrogate_model_final.joblib'))
    torch.save(ae.state_dict(), os.path.join(CONFIG['OUTPUT_DIR'], 'ae_model_final.pth'))

    # Summary
    logging.info("\n" + "="*60)
    logging.info("FINAL SUMMARY")
    logging.info("="*60)
    logging.info(f"LFSM on 1D Test: R�={results_1d_test['r2_timeseries']:.4f}, NMSE={results_1d_test['nmse']:.4f}%")
    logging.info(f"LFSM Cross-Domain: R�={results_cross['r2_timeseries']:.4f}, NMSE={results_cross['nmse']:.4f}%")
    logging.info(f"MFSM on 1D Test: R�={results_1d_final['r2_timeseries']:.4f}, NMSE={results_1d_final['nmse']:.4f}%")
    logging.info(f"MFSM on 2D Test: R�={results_2d_final['r2_timeseries']:.4f}, NMSE={results_2d_final['nmse']:.4f}%")
    logging.info("="*60)
    logging.info("Training complete!")

if __name__ == '__main__':
    main()
