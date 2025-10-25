"""
Modified MFSM-Based Inverse Problem Solver for Notch Location and Severity

This module provides a modified inverse problem solver that predicts:
1. notch_x (location parameter)
2. notch_severity (depth × width combined parameter)

Key Features:
- Severity classification (Mild, Moderate, Severe)
- Uncertainty quantification for both parameters
- Success criteria based on correct severity categorization
- Uses MFSM (Multi-Fidelity Surrogate Model) forward model

Usage Examples:
1. Run with default settings:
   python modified_inverse_problem.py

2. Run with custom number of test cases:
   from modified_inverse_problem import run_inverse_problem
   run_inverse_problem(num_cases=5)

Severity Classification:
- Mild: 0.00000001 to 0.0000004 (33% of range, minimal impact)
- Moderate: 0.0000004 to 0.0000008 (33% of range, noticeable impact)
- Severe: 0.0000008 to 0.0000012 (33% of range, significant concern)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import os
import joblib
import logging
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# --- Modified Configuration ---
CONFIG = {
    # --- Pre-trained MFSM Model Paths ---
    'CAE_MODEL_PATH': '/home/user2/Music/abhi3/test/mfsm_finetuned.pth',
    'SURROGATE_MODEL_PATH': '/home/user2/Music/abhi3/test/mfsm_surrogate_finetuned.joblib',
    'PARAMS_SCALER_PATH': '/home/user2/Music/abhi3/test/mfsm_scaler.joblib',

    # --- 2D Response Data Source ---
    'CSV_PATH': '/home/user2/Music/abhi3/parameters/test_responseslatest.csv',
    'USE_2D_RESPONSES': True,  # Use real 2D responses as ground truth instead of MFSM-generated

    
    
    # --- Modified Problem Parameters (3D independent parameter estimation) ---
    'PARAM_COLS': ['notch_x', 'notch_depth', 'notch_width', 'length', 'density', 'youngs_modulus'],
    'NOTCH_PARAMS': ['notch_x', 'notch_depth', 'notch_width'],  # Independent estimation of all 3 parameters
    'FIXED_PARAMS': {  # Fixed parameters for the beam
        'length': 3.0,
        'density': 2700.0,
        'youngs_modulus': 7e10,
        'location': 1.9  # Response point location
    },
    
    # --- MFSM Model Parameters ---
    'NUM_TIME_STEPS': 1500,
    'LATENT_DIM': 30,  # MFSM uses 30-dimensional latent space 
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # --- Independent Parameter Ranges ---
    'PARAM_BOUNDS': {
        'notch_x': (1.6545, 1.8355),  # Location bounds
        'notch_depth': (0.0001, 0.001),  # Depth bounds (lowered minimum for better log-space)
        'notch_width': (0.0001, 0.0012)  # Width bounds (lowered minimum for better log-space)
    },
    
    # --- Severity Classification Thresholds ---
    'SEVERITY_THRESHOLDS': {
        'mild': {'min': 0.00000001, 'max': 0.0000004, 'label': 'Mild'},
        'moderate': {'min': 0.0000004, 'max': 0.0000008, 'label': 'Moderate'}, 
        'severe': {'min': 0.0000008, 'max': 0.0000012, 'label': 'Severe'}
    },
    
    # --- Optimization Settings ---
    'OPT_METHOD': 'differential_evolution',  # or 'minimize'
    'OPT_MAXITER': 100,  # Increased for better convergence
    'OPT_POPSIZE': 50,    # Increased population for better exploration
    'OPT_ATOL': 1e-5,    # Much tighter tolerance to prevent premature convergence
    'OPT_TOL': 1e-3,     # Relative tolerance (LOWERED from 0.001 to prevent plateau)
    'OPT_STRATEGY': 'best2bin',  # Mutation strategy (best1bin, randtobest1bin, best2bin, rand2bin, rand1bin)
    'OPT_MUTATION': (0.5, 1.8),  # Mutation factor range (INCREASED upper bound for more exploration)
    'OPT_RECOMBINATION': 0.7,    # Crossover probability
    
    # --- Early Stopping Settings ---
    'ENABLE_EARLY_STOPPING': True,  # Enable smart early stopping
    'EARLY_STOP_PATIENCE': 15,  # Stop if no improvement for this many generations
    'EARLY_STOP_MIN_DELTA': 1e-6,  # Minimum improvement to reset patience counter
    'EARLY_STOP_GOOD_ENOUGH': 1e-6,  # Stop if loss is below this threshold (good enough)
    
    # --- Loss / Search Settings ---
    'LOSS_TYPE': 'composite',  # 'mse' or 'composite'
    'LOSS_WEIGHTS': {'mse': 1.0, 'corr': 0.0, 'fft': 0.0},  # Increased correlation weight, reduced FFT
    'USE_LOG_SPACE': True,  # optimize selected params in log10-space
    'LOG_SPACE_PARAMS': ['notch_depth', 'notch_width'],  # depth and width benefit from log-space optimization
    
    # --- Multi-stage optimization for better parameter estimation ---
    'USE_MULTI_STAGE_OPTIMIZATION': False,  # Enable multi-stage refinement (ENABLED to break plateaus)
    'STAGE1_MAXITER': 800,   # Stage 1: Global search with relaxed tolerance
    'STAGE1_ATOL': 1e-5,     # Relaxed tolerance for stage 1
    'STAGE2_MAXITER': 700,   # Stage 2: Fine-tuning with tight tolerance  
    'STAGE2_ATOL': 1e-7,     # Tight tolerance for stage 2
    
    # --- Test Case Settings ---
    'NUM_TEST_CASES': 40,  # Number of inverse problem test cases to run (randomly selected from available cases)
    'RANDOM_SEED': None,   # Set to integer for reproducible results, None for random sampling
    'VERBOSE_LOGGING': True,  # Enable detailed logging for each test case
    'FAVOR_SEVERE_CASES': True,  # Use weighted sampling to favor severe cases over mild ones
    'SAVE_ANIMATIONS': True,  # Enable GIF animations for all test cases
    'SAVE_ALL_CASE_ANIMATIONS': True,  # Save animations for all individual cases
    'ANIMATION_FOLDER': '/home/user2/Music/optimization_animations/2',  # Folder to save animations
    'ANIMATION_FPS': 2,  # Frames per second for animations (slower = more detailed viewing)
    'ANIMATION_FRAMES': 50,  # Number of frames in animation
    
    
    # --- Robust Optimization Settings ---
    'USE_ROBUST_LOSS': False,  # Use Huber loss instead of MSE for model-reality mismatch
    'MISMATCH_TOLERANCE': 0.1,  # Tolerance factor for Huber loss
    'HUBER_DELTA_FACTOR': 0.1,  # Factor to multiply with response std for Huber delta
    
    # --- Diagnostic Settings ---
    'ENABLE_PARAMETER_DIAGNOSTICS': True,  # Log parameter evolution during optimization
    'DIAGNOSTIC_INTERVAL': 100,  # Log diagnostics every N function evaluations
    
    
    # --- Time-Weighted Loss Settings ---
    'USE_TIME_WEIGHTED_LOSS': True,  # Enable time-weighted loss for important wavemodes
    'TIME_WEIGHTS': {
        'symmetric_antisymmetric': {'ranges': [(230, 470), (560, 900)], 'weight': 1.0},  # High priority: symmetric/anti-symmetric wavemodes
        'intermediate': {'ranges': [(470, 560)], 'weight': 1.0},  # Medium priority: intermediate region
        'baseline': {'weight': 1.0}  # Low priority: rest of the time series
    },
    
    # --- Performance Settings ---
    'ENABLE_TF32': False,  # TF32 optional for speed on Ampere+ (left off to preserve exact numerics)
    'COMPILE_DECODER': True,  # Compile decoder for faster inference when PyTorch>=2.0
}

# Set device to GPU if available, otherwise CPU
print(f"Using device: {CONFIG['DEVICE']}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

# --- Performance toggles (safe by default) ---
if CONFIG['DEVICE'] == 'cuda':
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    try:
        if CONFIG.get('ENABLE_TF32', False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
    except Exception:
        pass

# --- MFSM Architecture Classes (Non-conditional Autoencoder) ---
class BeamResponseDataset(Dataset):
    """
    Dataset for beam response data with parameter scaling to [-1, 1].
    Used for MFSM model which is non-conditional (doesn't use parameters in encoder/decoder).
    """
    def __init__(self, params, timeseries, p_scaler=None):
        self.timeseries = timeseries.astype(np.float32)
        self.params = params.astype(np.float32)
        
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
            'timeseries_raw': self.timeseries[idx]
        }

class Encoder(nn.Module):
    """Encoder: Time series → Latent space (NO parameter conditioning)"""
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
    """Decoder: Latent space → Time series (NO parameter conditioning)"""
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
    """Non-conditional Autoencoder for MFSM"""
    def __init__(self, timeseries_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(timeseries_dim, latent_dim)
        self.decoder = Decoder(latent_dim, timeseries_dim)
    
    def forward(self, x):
        """Forward pass - only takes time series, no parameters"""
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z

# --- Severity Conversion Functions ---
def create_parameter_vector(notch_x, notch_depth, notch_width, fixed_params):
    """
    Create parameter vector from individual notch parameters and fixed parameters
    
    Args:
        notch_x: Notch location parameter
        notch_depth: Notch depth parameter
        notch_width: Notch width parameter
        fixed_params: Fixed beam parameters
    
    Returns:
        np.array: Parameter vector compatible with MFSM model
    """
    params = np.array([
        notch_x,
        notch_depth,
        notch_width,
        fixed_params['length'],
        fixed_params['density'], 
        fixed_params['youngs_modulus'],
        fixed_params['location']
    ], dtype=np.float32).reshape(1, -1)
    
    return params

def depth_width_to_severity(notch_depth, notch_width):
    """
    Convert individual depth and width parameters to severity
    
    Args:
        notch_depth: Notch depth parameter
        notch_width: Notch width parameter
    
    Returns:
        float: Combined severity (depth × width)
    """
    return notch_depth * notch_width

def classify_severity(notch_severity):
    """
    Classify notch severity into categories
    
    Args:
        notch_severity: Combined severity parameter
    
    Returns:
        dict: Classification results with category and confidence
    """
    thresholds = CONFIG['SEVERITY_THRESHOLDS']
    
    if thresholds['mild']['min'] <= notch_severity <= thresholds['mild']['max']:
        return {
            'category': 'mild',
            'label': thresholds['mild']['label'],
            'confidence': 'high',
            'numeric_category': 0
        }
    elif thresholds['moderate']['min'] < notch_severity <= thresholds['moderate']['max']:
        return {
            'category': 'moderate', 
            'label': thresholds['moderate']['label'],
            'confidence': 'high',
            'numeric_category': 1
        }
    elif thresholds['severe']['min'] < notch_severity <= thresholds['severe']['max']:
        return {
            'category': 'severe',
            'label': thresholds['severe']['label'], 
            'confidence': 'high',
            'numeric_category': 2
        }
    else:
        # Outside defined ranges - determine closest category
        if notch_severity < thresholds['mild']['min']:
            return {
                'category': 'mild',
                'label': f"Below {thresholds['mild']['label']}",
                'confidence': 'low',
                'numeric_category': 0
            }
        else:  # above severe max
            return {
                'category': 'severe', 
                'label': f"Above {thresholds['severe']['label']}",
                'confidence': 'low',
                'numeric_category': 2
            }

def load_mfsm_model():
    """Load pre-trained MFSM model components (non-conditional autoencoder)"""
    logging.info("=" * 50)
    logging.info("LOADING PRE-TRAINED MFSM MODEL COMPONENTS")
    logging.info("=" * 50)
    
    logging.info(f"Model paths:")
    logging.info(f"  CAE Model: {CONFIG['CAE_MODEL_PATH']}")
    logging.info(f"  Surrogate Model: {CONFIG['SURROGATE_MODEL_PATH']}")
    logging.info(f"  Parameter Scaler: {CONFIG['PARAMS_SCALER_PATH']}")
    
    # Initialize non-conditional autoencoder
    logging.info(f"Initializing MFSM Autoencoder (non-conditional)...")
    logging.info(f"  Time series dimension: {CONFIG['NUM_TIME_STEPS']}")
    logging.info(f"  Latent dimension: {CONFIG['LATENT_DIM']}")
    
    cae = Autoencoder(
        timeseries_dim=CONFIG['NUM_TIME_STEPS'],
        latent_dim=CONFIG['LATENT_DIM']
    )
    
    # Load pre-trained weights
    try:
        logging.info("Loading MFSM pre-trained weights...")
        cae.load_state_dict(torch.load(CONFIG['CAE_MODEL_PATH'], map_location=CONFIG['DEVICE']))
        cae.to(CONFIG['DEVICE'])
        cae.eval()
        # Optionally compile decoder for faster inference without changing numerics
        if CONFIG.get('COMPILE_DECODER', True) and hasattr(torch, 'compile'):
            try:
                cae.decoder = torch.compile(cae.decoder, mode="reduce-overhead", fullgraph=False)
                logging.info("  Decoder compiled with torch.compile")
            except Exception as e:
                logging.info(f"  torch.compile unavailable or failed: {e}")
        logging.info(f"✓ Successfully loaded MFSM model from {CONFIG['CAE_MODEL_PATH']}")
        
        # Count parameters
        total_params = sum(p.numel() for p in cae.parameters())
        trainable_params = sum(p.numel() for p in cae.parameters() if p.requires_grad)
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,}")
        
    except FileNotFoundError:
        logging.error(f"✗ MFSM model file not found: {CONFIG['CAE_MODEL_PATH']}")
        raise
    except Exception as e:
        logging.error(f"✗ Error loading MFSM model: {e}")
        raise
    
    # Load XGBoost surrogate model
    try:
        logging.info("Loading XGBoost surrogate model...")
        surrogate_model = joblib.load(CONFIG['SURROGATE_MODEL_PATH'])
        logging.info(f"✓ Successfully loaded surrogate model from {CONFIG['SURROGATE_MODEL_PATH']}")
        logging.info(f"  Model type: {type(surrogate_model).__name__}")
        if hasattr(surrogate_model, 'n_estimators'):
            logging.info(f"  Number of estimators: {surrogate_model.n_estimators}")
        
    except FileNotFoundError:
        logging.error(f"✗ Surrogate model file not found: {CONFIG['SURROGATE_MODEL_PATH']}")
        raise
    except Exception as e:
        logging.error(f"✗ Error loading surrogate model: {e}")
        raise
    
    # Load parameter scaler
    try:
        logging.info("Loading parameter scaler...")
        params_scaler = joblib.load(CONFIG['PARAMS_SCALER_PATH'])
        logging.info(f"✓ Successfully loaded parameter scaler from {CONFIG['PARAMS_SCALER_PATH']}")
        logging.info(f"  Scaler type: {type(params_scaler).__name__}")
        
        # Handle different scaler types
        if hasattr(params_scaler, 'feature_range'):
            logging.info(f"  Feature range: {params_scaler.feature_range}")
        elif hasattr(params_scaler, 'mean_'):
            logging.info(f"  Mean: {params_scaler.mean_}")
            logging.info(f"  Scale: {params_scaler.scale_}")
        
        logging.info(f"  Number of features: {len(params_scaler.feature_names_in_) if hasattr(params_scaler, 'feature_names_in_') else 'N/A'}")
        
    except FileNotFoundError:
        logging.error(f"✗ Parameter scaler file not found: {CONFIG['PARAMS_SCALER_PATH']}")
        raise
    except Exception as e:
        logging.error(f"✗ Error loading parameter scaler: {e}")
        raise
    
    logging.info("✓ All MFSM model components loaded successfully!")
    return cae, surrogate_model, params_scaler






def mfsm_forward_model(notch_params_3d, cae, surrogate_model, params_scaler, fixed_params, verbose=False):
    """
    MFSM forward model: Given notch_x, notch_depth, and notch_width, predict time series response
    
    Args:
        notch_params_3d: array of [notch_x, notch_depth, notch_width] 
        cae: Pre-trained non-conditional autoencoder
        surrogate_model: Pre-trained XGBoost model
        params_scaler: Parameter scaler
        fixed_params: Dict of fixed beam parameters
        verbose: Whether to log detailed information
    
    Returns:
        predicted_response: 1D array of time series response (as trained in MFSM)
    """
    if verbose:
        logging.debug(f"MFSM Forward Model Input:")
        logging.debug(f"  Notch parameters [x, depth, width]: {notch_params_3d}")
        logging.debug(f"  Fixed parameters: {fixed_params}")
    
    # Extract parameters
    notch_x = notch_params_3d[0]
    notch_depth = notch_params_3d[1]
    notch_width = notch_params_3d[2]
    
    # Create full parameter vector directly (no conversion needed)
    params = create_parameter_vector(notch_x, notch_depth, notch_width, fixed_params)
    
    if verbose:
        logging.debug(f"  Full parameter vector: {params.flatten()}")
        severity = depth_width_to_severity(notch_depth, notch_width)
        logging.debug(f"  Calculated severity: {severity:.12f}")
    
    # Scale parameters
    params_scaled = params_scaler.transform(params)
    
    if verbose:
        logging.debug(f"  Scaled parameters: {params_scaled.flatten()}")
    
    # Predict latent vector using surrogate model
    z_pred = surrogate_model.predict(params_scaled)

    # Ensure the latent vector has the correct shape
    if z_pred.ndim == 1:
        z_pred = z_pred.reshape(1, -1)
    elif z_pred.shape[0] != 1:
        # If multiple predictions, take the first one
        z_pred = z_pred[0:1]

    if verbose:
        logging.debug(f"  Predicted latent vector shape: {z_pred.shape}")
        logging.debug(f"  Latent vector stats: min={z_pred.min():.4f}, max={z_pred.max():.4f}, mean={z_pred.mean():.4f}")

    # Validate latent vector shape matches expected dimension
    expected_latent_dim = CONFIG['LATENT_DIM']
    if z_pred.shape[1] != expected_latent_dim:
        logging.warning(f"Latent vector dimension mismatch: expected {expected_latent_dim}, got {z_pred.shape[1]}")
        # Try to handle the mismatch - this might indicate the wrong model is being used
        if z_pred.shape[1] == CONFIG['NUM_TIME_STEPS']:
            logging.error("Surrogate model appears to be predicting time series instead of latent vectors!")
            logging.error("This suggests the wrong surrogate model file is being used.")
            raise ValueError(f"Surrogate model predicts {z_pred.shape[1]} dimensions (time series) instead of {expected_latent_dim} (latent vectors)")
    
    # Reconstruct time series using decoder (non-conditional - no params needed)
    with torch.inference_mode():
        z_tensor = torch.tensor(z_pred, dtype=torch.float32).to(CONFIG['DEVICE'])
        response_raw = cae.decoder(z_tensor).cpu().numpy().squeeze()
    
    if verbose:
        logging.debug(f"  Generated response shape: {response_raw.shape}")
        logging.debug(f"  Generated response stats: min={response_raw.min():.4f}, max={response_raw.max():.4f}, mean={response_raw.mean():.4f}")
    
    return response_raw

# --- Loss Functions (adapted for severity-based approach) ---
def create_time_weights(length=1500):
    """
    Create time-weighted array for loss calculation based on wavemode importance
    
    Args:
        length: Length of time series (default 1500)
        
    Returns:
        np.array: Weight array with shape (length,)
    """
    if not CONFIG.get('USE_TIME_WEIGHTED_LOSS', False):
        return np.ones(length, dtype=np.float32)
    
    weights = np.full(length, CONFIG['TIME_WEIGHTS']['baseline']['weight'], dtype=np.float32)
    
    # Apply high priority weights (symmetric/anti-symmetric wavemodes)
    high_priority = CONFIG['TIME_WEIGHTS']['symmetric_antisymmetric']
    high_weight_points = 0
    for start, end in high_priority['ranges']:
        start_idx = max(0, min(start, length-1))
        end_idx = max(0, min(end, length))
        if start_idx < end_idx:
            weights[start_idx:end_idx] = high_priority['weight']
            high_weight_points += end_idx - start_idx
    
    # Apply medium priority weights (intermediate region)
    med_priority = CONFIG['TIME_WEIGHTS']['intermediate']
    med_weight_points = 0
    for start, end in med_priority['ranges']:
        start_idx = max(0, min(start, length-1))
        end_idx = max(0, min(end, length))
        if start_idx < end_idx:
            weights[start_idx:end_idx] = med_priority['weight']
            med_weight_points += end_idx - start_idx
    
    # Log weighting info on first call (static variable simulation)
    if not hasattr(create_time_weights, '_logged'):
        create_time_weights._logged = True
        baseline_points = length - high_weight_points - med_weight_points
        logging.info(f"🎯 TIME-WEIGHTED LOSS ENABLED:")
        logging.info(f"  High priority regions (weight={high_priority['weight']}): {high_weight_points} points ({high_weight_points/length*100:.1f}%)")
        logging.info(f"  Medium priority regions (weight={med_priority['weight']}): {med_weight_points} points ({med_weight_points/length*100:.1f}%)")
        logging.info(f"  Baseline regions (weight={CONFIG['TIME_WEIGHTS']['baseline']['weight']}): {baseline_points} points ({baseline_points/length*100:.1f}%)")
        
        # Show the ranges
        logging.info(f"  High priority ranges: {high_priority['ranges']}")
        logging.info(f"  Medium priority ranges: {med_priority['ranges']}")
    
    return weights

def compute_inverse_loss(predicted, target):
    """Composite loss: amplitude-aligned MSE + (1 - corr) + spectral MSE.
    Falls back to plain MSE if CONFIG['LOSS_TYPE'] == 'mse'.
    Now supports time-weighted loss for important wavemode regions.
    """
    eps = 1e-12
    
    # Get time weights for important regions
    time_weights = create_time_weights(len(target))
    
    if CONFIG.get('LOSS_TYPE', 'mse') == 'mse':
        # Simple weighted MSE
        weighted_mse = np.mean(time_weights * (predicted - target) ** 2)
        return float(weighted_mse)

    weights = CONFIG.get('LOSS_WEIGHTS', {'mse': 0.6, 'corr': 0.4, 'fft': 0.2})

    # Amplitude alignment (closed-form scaling)
    dot = float(np.dot(predicted, target))
    denom = float(np.dot(predicted, predicted)) + eps
    a = dot / denom
    pred_adj = a * predicted

    # Time-domain weighted MSE (prioritizing important wavemode regions)
    time_mse = np.mean(time_weights * (pred_adj - target) ** 2)

    # Shape correlation term (1 - Pearson correlation) - also weighted
    pred_z = (pred_adj - pred_adj.mean()) / (pred_adj.std() + eps)
    targ_z = (target - target.mean()) / (target.std() + eps)
    # Weight the correlation calculation to focus on important regions
    weighted_corr = np.sum(time_weights * pred_z * targ_z) / np.sum(time_weights)
    corr_loss = 1.0 - np.clip(weighted_corr, -1.0, 1.0)

    # Light frequency-domain magnitude MSE (first K rFFT bins)
    Pm = np.abs(np.fft.rfft(pred_adj))
    Tm = np.abs(np.fft.rfft(target))
    Pm = Pm / (Pm.sum() + eps)
    Tm = Tm / (Tm.sum() + eps)
    K = min(256, Pm.shape[0])
    fft_mse = np.mean((Pm[:K] - Tm[:K]) ** 2)

    return float(weights.get('mse', 0.0) * time_mse +
                 weights.get('corr', 0.0) * corr_loss +
                 weights.get('fft', 0.0) * fft_mse)

def precompute_target_metrics(target):
    eps = 1e-12
    target = target.astype(np.float64, copy=False)
    target_mean = float(target.mean())
    target_std = float(target.std()) + eps
    targ_z = (target - target_mean) / target_std
    Tm = np.abs(np.fft.rfft(target))
    Tm = Tm / (Tm.sum() + eps)
    K = int(min(256, Tm.shape[0]))
    
    # Precompute time weights for important wavemode regions
    time_weights = create_time_weights(len(target))
    
    return {
        'target': target,
        'targ_z': targ_z,
        'Tm': Tm,
        'K': K,
        'eps': eps,
        'time_weights': time_weights,
    }

def compute_inverse_loss_fast(predicted, target_stats):
    weights = CONFIG.get('LOSS_WEIGHTS', {'mse': 0.6, 'corr': 0.4, 'fft': 0.2})
    eps = target_stats['eps']
    target = target_stats['target']
    targ_z = target_stats['targ_z']
    Tm = target_stats['Tm']
    K = target_stats['K']
    time_weights = target_stats['time_weights']

    predicted = predicted.astype(np.float64, copy=False)

    dot = float(np.dot(predicted, target))
    denom = float(np.dot(predicted, predicted)) + eps
    a = dot / denom
    pred_adj = a * predicted

    # Time-domain weighted MSE (prioritizing important wavemode regions)
    time_mse = np.mean(time_weights * (pred_adj - target) ** 2)

    # Shape correlation term - also weighted
    pred_z = (pred_adj - pred_adj.mean()) / (pred_adj.std() + eps)
    weighted_corr = np.sum(time_weights * pred_z * targ_z) / np.sum(time_weights)
    corr_loss = 1.0 - np.clip(weighted_corr, -1.0, 1.0)

    Pm = np.abs(np.fft.rfft(pred_adj))
    Pm = Pm / (Pm.sum() + eps)
    fft_mse = np.mean((Pm[:K] - Tm[:K]) ** 2)

    return float(weights.get('mse', 0.0) * time_mse +
                 weights.get('corr', 0.0) * corr_loss +
                 weights.get('fft', 0.0) * fft_mse)

def compute_huber_loss(predicted, target, delta=None):
    """
    Compute Huber loss - robust to outliers, less sensitive to model-reality mismatch
    Now supports time-weighted loss for important wavemode regions.
    
    Args:
        predicted: Predicted response
        target: Target response (2D ground truth)
        delta: Huber loss threshold. If None, computed as fraction of target std
    
    Returns:
        float: Huber loss value
    """
    if delta is None:
        delta = CONFIG.get('HUBER_DELTA_FACTOR', 0.1) * np.std(target)
    
    # Get time weights for important regions
    time_weights = create_time_weights(len(target))
    
    residuals = predicted - target
    abs_residuals = np.abs(residuals)
    
    # Huber loss: quadratic for small errors, linear for large errors
    huber_loss = np.where(
        abs_residuals <= delta,
        0.5 * residuals**2,
        delta * (abs_residuals - 0.5 * delta)
    )
    
    # Apply time weighting to prioritize important wavemode regions
    weighted_huber_loss = time_weights * huber_loss
    
    return float(np.mean(weighted_huber_loss))

# --- Optimization and Parameter Estimation (modified for 2-parameter case) ---
def robust_objective_function(notch_params_3d, target_2d_response, cae, surrogate_model,
                             params_scaler, fixed_params, use_robust_loss=True):
    """
    Robust objective function for 3-parameter optimization
    
    Args:
        notch_params_3d: Parameters to optimize [notch_x, notch_depth, notch_width]
        target_2d_response: Ground truth 2D response
        cae: Conditional autoencoder model
        surrogate_model: XGBoost surrogate model
        params_scaler: Parameter scaler
        fixed_params: Fixed beam parameters
        use_robust_loss: Whether to use Huber loss (robust) or MSE (standard)
    
    Returns:
        float: Loss value
    """
    try:
        # Generate MFSM prediction using 3-parameter forward model
        mfsm_response = mfsm_forward_model(notch_params_3d, cae, surrogate_model, params_scaler, fixed_params)
        
        if use_robust_loss and CONFIG.get('USE_ROBUST_LOSS', True):
            # Use Huber loss for robustness to model-reality mismatch
            return compute_huber_loss(mfsm_response, target_2d_response)
        else:
            # Use standard composite loss or MSE
            return compute_inverse_loss(mfsm_response, target_2d_response)
            
    except Exception as e:
        logging.warning(f"Error in robust objective function: {e}")
        return 1e6  # Return high penalty for invalid parameters

def generate_random_notch_parameters(seed=None):
    """
    Generate random notch parameters within specified bounds for 3-parameter approach
    
    Args:
        seed: Random seed for reproducible results (optional)
    
    Returns:
        np.array: Random notch parameters [notch_x, notch_depth, notch_width]
    """
    if seed is not None:
        # Create a local random state for this generation
        rng = np.random.RandomState(seed)
        notch_x = rng.uniform(*CONFIG['PARAM_BOUNDS']['notch_x'])
        notch_depth = rng.uniform(*CONFIG['PARAM_BOUNDS']['notch_depth'])
        notch_width = rng.uniform(*CONFIG['PARAM_BOUNDS']['notch_width'])
    else:
        notch_x = np.random.uniform(*CONFIG['PARAM_BOUNDS']['notch_x'])
        notch_depth = np.random.uniform(*CONFIG['PARAM_BOUNDS']['notch_depth'])
        notch_width = np.random.uniform(*CONFIG['PARAM_BOUNDS']['notch_width'])
    
    return np.array([notch_x, notch_depth, notch_width])

# --- Modified Parameter Estimation Function ---
def estimate_notch_parameters(target_response, cae, surrogate_model, params_scaler, fixed_params,
                             method='differential_evolution', save_animations=False, case_id=0,
                             database_df=None, response_cols=None):
    """
    Estimate notch parameters using optimization to match target response (3-parameter approach)
    Enhanced with database integration for smart population initialization

    Args:
        target_response: Target time series response to match
        cae: Pre-trained conditional autoencoder
        surrogate_model: Pre-trained XGBoost model
        params_scaler: Parameter scaler
        fixed_params: Dict of fixed beam parameters
        method: Optimization method ('differential_evolution' or 'minimize')
        save_animations: Whether to save GIF animations of the optimization process
        case_id: Test case identifier for naming animation files
        database_df: Database DataFrame for similarity analysis (optional)
        response_cols: List of response column names in database (optional)

    Returns:
        result: Optimization result with estimated parameters (in physical space)
        optimization_trajectory: Dictionary containing actual optimization trajectory data (if save_animations=True)
    """
    
    logging.info("🎯 STARTING 3-PARAMETER ESTIMATION")
    logging.info(f"  Optimization method: {method}")
    logging.info(f"  Target response shape: {target_response.shape}")
    logging.info(f"  Target response stats: min={target_response.min():.12f}, max={target_response.max():.12f}, mean={target_response.mean():.12f}")

    # Initialize population equidistant over the whole parameter space
    logging.info("🔄 EQUIDISTANT POPULATION INITIALIZATION")

    # Precompute target invariants for faster objective evaluation
    target_stats = precompute_target_metrics(target_response)
    
    # Base physical bounds for 3-parameter case
    param_names = CONFIG['NOTCH_PARAMS']
    physical_bounds = [
        CONFIG['PARAM_BOUNDS']['notch_x'],
        CONFIG['PARAM_BOUNDS']['notch_depth'],
        CONFIG['PARAM_BOUNDS']['notch_width']
    ]

    # Optional log10-space for selected params
    use_log = CONFIG.get('USE_LOG_SPACE', False)
    log_set = set(CONFIG.get('LOG_SPACE_PARAMS', []))
    log_idx = [i for i, n in enumerate(param_names) if n in log_set] if use_log else []

    def to_opt_space(x_phys):
        x = np.array(x_phys, dtype=np.float64)
        for i in log_idx:
            x[i] = np.log10(x[i])
        return x

    def to_physical_space(x_opt):
        x = np.array(x_opt, dtype=np.float64)
        for i in log_idx:
            x[i] = 10.0 ** x[i]
        return x

    # Build optimizer bounds
    if log_idx:
        bounds_opt = []
        for i, (lo, hi) in enumerate(physical_bounds):
            if i in log_idx:
                bounds_opt.append((np.log10(lo), np.log10(hi)))
            else:
                bounds_opt.append((lo, hi))
        shown_bounds = physical_bounds
    else:
        bounds_opt = physical_bounds
        shown_bounds = physical_bounds

    logging.info("  Parameter bounds for estimation (physical space):")
    for (name, (lo, hi)) in zip(param_names, shown_bounds):
        logging.info(f"    {name}: [{lo:.12f}, {hi:.12f}]")
    
    if method == 'differential_evolution':
        logging.info(f"  Differential Evolution settings:")
        logging.info(f"    Max iterations: {CONFIG['OPT_MAXITER']}")
        logging.info(f"    Population size: {CONFIG['OPT_POPSIZE']}")
        logging.info(f"    Absolute tolerance: {CONFIG['OPT_ATOL']}")
        logging.info(f"    Relative tolerance: {CONFIG['OPT_TOL']}")
        logging.info(f"    Strategy: {CONFIG['OPT_STRATEGY']}")
        logging.info(f"    Mutation: {CONFIG['OPT_MUTATION']}")
        logging.info(f"    Recombination: {CONFIG['OPT_RECOMBINATION']}")

        # Initialize trajectory tracking for animations
        optimization_trajectory = {
            'parameters': [],
            'losses': [],
            'responses': [],
            'iterations': []
        } if save_animations else None
        
        generation_counter = [0]  # Track generation number
        
        def wrapped_objective_with_logging(x_opt, target_stats, cae, surrogate_model, params_scaler, fixed_params):
            x_phys = to_physical_space(x_opt) if log_idx else x_opt
            try:
                predicted_response = mfsm_forward_model(x_phys, cae, surrogate_model, params_scaler, fixed_params)
                loss = compute_inverse_loss_fast(predicted_response, target_stats) if CONFIG.get('LOSS_TYPE', 'mse') != 'mse' else float(np.mean((predicted_response - target_stats['target']) ** 2))
                return loss
            except Exception as e:
                logging.warning(f"Error in objective function: {e}")
                return 1e6
        
        # Early stopping state
        early_stop_state = {
            'best_loss': float('inf'),
            'patience_counter': 0,
            'best_params': None,
            'stopped_early': False,
            'stop_reason': None
        }
        
        # Callback function to track only the best solution per generation and implement early stopping
        def track_best_callback(xk, convergence):
            """Callback called after each generation with the best solution"""
            # Track trajectory for animation if enabled
            if save_animations and optimization_trajectory is not None:
                x_phys = to_physical_space(xk) if log_idx else xk
                try:
                    predicted_response = mfsm_forward_model(x_phys, cae, surrogate_model, params_scaler, fixed_params)
                    loss = compute_inverse_loss_fast(predicted_response, target_stats) if CONFIG.get('LOSS_TYPE', 'mse') != 'mse' else float(np.mean((predicted_response - target_stats['target']) ** 2))
                    
                    optimization_trajectory['parameters'].append(x_phys.copy())
                    optimization_trajectory['losses'].append(loss)
                    optimization_trajectory['responses'].append(predicted_response.copy())
                    optimization_trajectory['iterations'].append(generation_counter[0])
                    generation_counter[0] += 1
                except Exception as e:
                    logging.warning(f"Error in callback: {e}")
            
            # Early stopping logic
            if CONFIG.get('ENABLE_EARLY_STOPPING', False):
                x_phys = to_physical_space(xk) if log_idx else xk
                try:
                    # Calculate current loss
                    predicted_response = mfsm_forward_model(x_phys, cae, surrogate_model, params_scaler, fixed_params)
                    current_loss = compute_inverse_loss_fast(predicted_response, target_stats) if CONFIG.get('LOSS_TYPE', 'mse') != 'mse' else float(np.mean((predicted_response - target_stats['target']) ** 2))
                    
                    # Check if loss is good enough
                    if current_loss < CONFIG.get('EARLY_STOP_GOOD_ENOUGH', 1e-6):
                        early_stop_state['stopped_early'] = True
                        early_stop_state['stop_reason'] = f'Loss {current_loss:.8e} below threshold {CONFIG.get("EARLY_STOP_GOOD_ENOUGH", 1e-6):.8e}'
                        logging.info(f"✓ Early stopping: {early_stop_state['stop_reason']} at generation {generation_counter[0]}")
                        return True  # Stop optimization
                    
                    # Check for improvement
                    min_delta = CONFIG.get('EARLY_STOP_MIN_DELTA', 1e-6)
                    improvement = early_stop_state['best_loss'] - current_loss
                    
                    if improvement > min_delta:
                        # Significant improvement found
                        early_stop_state['best_loss'] = current_loss
                        early_stop_state['best_params'] = x_phys.copy()
                        early_stop_state['patience_counter'] = 0
                    else:
                        # No significant improvement
                        early_stop_state['patience_counter'] += 1
                        
                        # Check if patience exhausted
                        patience = CONFIG.get('EARLY_STOP_PATIENCE', 15)
                        if early_stop_state['patience_counter'] >= patience:
                            early_stop_state['stopped_early'] = True
                            early_stop_state['stop_reason'] = f'No improvement (>{min_delta:.8e}) for {patience} generations (best loss: {early_stop_state["best_loss"]:.8e})'
                            logging.info(f"✓ Early stopping: {early_stop_state['stop_reason']} at generation {generation_counter[0]}")
                            return True  # Stop optimization
                    
                except Exception as e:
                    logging.warning(f"Error in early stopping check: {e}")
            
            return False  # Continue optimization

        # Create equidistant population over the entire parameter space
        logging.info("🧬 CREATING EQUIDISTANT POPULATION")

        # Create equidistant population in optimization space
        n_params = len(bounds_opt)
        popsize = CONFIG['OPT_POPSIZE']

        # Create equidistant grid points across all dimensions
        # For 3D space, use Latin Hypercube Sampling for better coverage
        if n_params == 3:
            # Create a 3D grid that's evenly spaced
            grid_size = int(np.ceil(popsize ** (1/3)))

            # Create coordinate arrays for each dimension
            coords = []
            for i in range(n_params):
                low, high = bounds_opt[i]
                coord = np.linspace(low, high, grid_size)
                coords.append(coord)

            # Create meshgrid and flatten to get all combinations
            meshes = np.meshgrid(*coords, indexing='ij')
            population_list = []

            for i in range(n_params):
                population_list.append(meshes[i].flatten())

            initial_population = np.column_stack(population_list)

            # If we have more points than needed, randomly sample
            if len(initial_population) > popsize:
                indices = np.random.choice(len(initial_population), size=popsize, replace=False)
                initial_population = initial_population[indices]
            elif len(initial_population) < popsize:
                # If we have fewer points, add random points to fill
                n_missing = popsize - len(initial_population)
                random_points = np.random.uniform(
                    [b[0] for b in bounds_opt],
                    [b[1] for b in bounds_opt],
                    size=(n_missing, n_params)
                )
                initial_population = np.vstack([initial_population, random_points])
        else:
            # Fallback for other dimensionalities
            initial_population = np.random.uniform(
                [b[0] for b in bounds_opt],
                [b[1] for b in bounds_opt],
                size=(popsize, n_params)
            )

        logging.info(f"Created equidistant population: {initial_population.shape}")

        # Validate population shape and bounds
        if initial_population.shape != (CONFIG['OPT_POPSIZE'], len(param_names)):
            raise ValueError(f"Initial population shape {initial_population.shape} invalid")

        # Check that all individuals are within bounds
        for i in range(len(param_names)):
            low, high = bounds_opt[i]
            if not np.all((initial_population[:, i] >= low) & (initial_population[:, i] <= high)):
                logging.warning(f"Some individuals in parameter {i} outside bounds [{low}, {high}]")

        # Multi-stage optimization for better parameter estimation
        if CONFIG.get('USE_MULTI_STAGE_OPTIMIZATION', False):
            logging.info("🔄 MULTI-STAGE OPTIMIZATION")
            
            # Stage 1: Global exploration with relaxed tolerance
            logging.info(f"Stage 1: Global exploration (maxiter={CONFIG['STAGE1_MAXITER']}, atol={CONFIG['STAGE1_ATOL']})")
            stage1_result = differential_evolution(
                wrapped_objective_with_logging,
                bounds_opt,
                args=(target_stats, cae, surrogate_model, params_scaler, fixed_params),
                maxiter=CONFIG['STAGE1_MAXITER'],
                popsize=CONFIG['OPT_POPSIZE'],
                atol=CONFIG['STAGE1_ATOL'],
                tol=CONFIG['OPT_TOL'],
                strategy=CONFIG['OPT_STRATEGY'],
                mutation=CONFIG['OPT_MUTATION'],
                recombination=CONFIG['OPT_RECOMBINATION'],
                seed=None,
                disp=False,
                polish=False,
                init=initial_population,
                callback=track_best_callback if save_animations else None
            )
            
            # Stage 2: Fine-tuning around best solution
            logging.info(f"Stage 2: Fine-tuning (maxiter={CONFIG['STAGE2_MAXITER']}, atol={CONFIG['STAGE2_ATOL']})")
            
            # Create focused population around stage 1 result
            best_x = stage1_result.x
            refined_population = []
            
            # Add best solution
            refined_population.append(best_x)
            
            # Add perturbed versions of best solution
            for _ in range(CONFIG['OPT_POPSIZE'] - 1):
                perturbed = best_x + np.random.normal(0, 0.1 * (np.array([b[1] - b[0] for b in bounds_opt])), len(best_x))
                # Clip to bounds
                for j, (low, high) in enumerate(bounds_opt):
                    perturbed[j] = np.clip(perturbed[j], low, high)
                refined_population.append(perturbed)
            
            refined_population = np.array(refined_population)
            
            result = differential_evolution(
                wrapped_objective_with_logging,
                bounds_opt,
                args=(target_stats, cae, surrogate_model, params_scaler, fixed_params),
                maxiter=CONFIG['STAGE2_MAXITER'],
                popsize=CONFIG['OPT_POPSIZE'],
                atol=CONFIG['STAGE2_ATOL'],
                tol=CONFIG['OPT_TOL'],
                strategy=CONFIG['OPT_STRATEGY'],
                mutation=CONFIG['OPT_MUTATION'],
                recombination=CONFIG['OPT_RECOMBINATION'],
                seed=None,
                disp=True,
                polish=True,
                init=refined_population,
                callback=track_best_callback if save_animations else None
            )
            
            logging.info(f"Multi-stage optimization: Stage 1 loss={stage1_result.fun:.8f} -> Stage 2 loss={result.fun:.8f}")
            
        else:
            # Single-stage optimization (original)
            result = differential_evolution(
                wrapped_objective_with_logging,
                bounds_opt,
                args=(target_stats, cae, surrogate_model, params_scaler, fixed_params),
                maxiter=CONFIG['OPT_MAXITER'],
                popsize=CONFIG['OPT_POPSIZE'],
                atol=CONFIG['OPT_ATOL'],
                tol=CONFIG['OPT_TOL'],
                strategy=CONFIG['OPT_STRATEGY'],
                mutation=CONFIG['OPT_MUTATION'],
                recombination=CONFIG['OPT_RECOMBINATION'],
                seed=None,
                disp=True,
                polish=True,
                init=initial_population,
                callback=track_best_callback if save_animations else None
            )
        
        # Convert the reported solution to physical space so downstream code remains unchanged
        result.x = to_physical_space(result.x) if log_idx else result.x
        
    else:
        # Use initial guess from center of bounds
        initial_guess_phys = np.array([np.mean(b) for b in physical_bounds], dtype=np.float64)
        initial_guess = to_opt_space(initial_guess_phys) if log_idx else initial_guess_phys
        logging.info(f"  L-BFGS-B settings:")
        logging.info(f"    Initial guess (physical): {dict(zip(param_names, initial_guess_phys))}")
        logging.info(f"    Max iterations: {CONFIG['OPT_MAXITER']}")

        def wrapped_objective(x_opt, target_stats, cae, surrogate_model, params_scaler, fixed_params):
            x_phys = to_physical_space(x_opt) if log_idx else x_opt
            try:
                predicted_response = mfsm_forward_model(x_phys, cae, surrogate_model, params_scaler, fixed_params)
                return compute_inverse_loss_fast(predicted_response, target_stats) if CONFIG.get('LOSS_TYPE', 'mse') != 'mse' else float(np.mean((predicted_response - target_stats['target']) ** 2))
            except Exception as e:
                logging.warning(f"Error in objective function: {e}")
                return 1e6

        result = minimize(
            wrapped_objective,
            initial_guess,
            args=(target_stats, cae, surrogate_model, params_scaler, fixed_params),
            bounds=bounds_opt,
            method='L-BFGS-B',
            options={'maxiter': CONFIG['OPT_MAXITER'], 'disp': True}
        )
        result.x = to_physical_space(result.x) if log_idx else result.x

    logging.info("✓ 3-PARAMETER ESTIMATION COMPLETE")
    logging.info(f"  Success: {result.success}")
    logging.info(f"  Message: {result.message}")
    logging.info(f"  Final objective value: {result.fun:.8f}")
    logging.info(f"  Function evaluations: {result.nfev}")
    if hasattr(result, 'nit'):
        logging.info(f"  Iterations: {result.nit}")
    
    # Log early stopping information if applicable
    if method == 'differential_evolution' and 'early_stop_state' in locals():
        if early_stop_state['stopped_early']:
            logging.info(f"  ⚡ Early stopping activated: {early_stop_state['stop_reason']}")
        else:
            logging.info(f"  ⏱️ Ran full {CONFIG['OPT_MAXITER']} iterations (no early stopping)")

    # Return trajectory data if animations were requested
    if save_animations and optimization_trajectory is not None:
        logging.info(f"  Captured {len(optimization_trajectory['parameters'])} trajectory points for animation")
        return result, optimization_trajectory
    else:
        return result

def create_optimization_animation(case_data, cae, surrogate_model, params_scaler, case_id, animation_folder, optimization_trajectory=None):
    """
    Create and save separate parameter and response optimization animations for a test case
    Creates two GIFs: optimization_parameters_case_X.gif and optimization_response_case_X.gif
    
    Args:
        case_data: Test case data dictionary
        cae: Conditional autoencoder model
        surrogate_model: XGBoost surrogate model
        params_scaler: Parameter scaler
        case_id: Case identifier for naming
        animation_folder: Folder to save animations
        optimization_trajectory: Dictionary with actual optimization trajectory data (parameters, losses, responses, iterations)
    """
    try:
        import os
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Ellipse
        
        # Check if pillow is available for GIF creation
        try:
            from PIL import Image
        except ImportError:
            logging.error("PIL/Pillow not available - cannot create GIF animations")
            return
        
        # Create animation folder if it doesn't exist
        os.makedirs(animation_folder, exist_ok=True)
        
        logging.info(f"Creating optimization animations for case {case_id}...")
        
        true_params_3d = case_data['true_params_3d']
        target_2d_response = case_data['2d_response']
        fixed_params = case_data['fixed_params']
        
        # Use real trajectory data if available, otherwise fallback to simulation
        param_bounds = CONFIG['PARAM_BOUNDS']
        
        if optimization_trajectory is not None and len(optimization_trajectory['parameters']) > 0:
            logging.info(f"  Using real optimization trajectory with {len(optimization_trajectory['parameters'])} points")
            trajectory_params = np.array(optimization_trajectory['parameters'])
            trajectory_losses = np.array(optimization_trajectory['losses'])
            trajectory_responses = optimization_trajectory['responses']
            n_points = len(trajectory_params)
            
            # Subsample if too many points for smooth animation
            max_frames = CONFIG.get('ANIMATION_FRAMES', 100)
            if n_points > max_frames:
                indices = np.linspace(0, n_points-1, max_frames, dtype=int)
                trajectory_params = trajectory_params[indices]
                trajectory_losses = trajectory_losses[indices]
                trajectory_responses = [trajectory_responses[i] for i in indices]
                n_points = max_frames
                logging.info(f"  Subsampled to {n_points} frames for animation")
        else:
            logging.warning("  No trajectory data available, using simulated linear progression")
            n_points = CONFIG['ANIMATION_FRAMES']
            trajectory_params = None
            trajectory_losses = None
            trajectory_responses = None
        
        # Create parameter grid around true values for visualization
        x_range = np.linspace(param_bounds['notch_x'][0], param_bounds['notch_x'][1], int(np.sqrt(n_points)))
        depth_range = np.linspace(param_bounds['notch_depth'][0], param_bounds['notch_depth'][1], int(np.sqrt(n_points)))
        
        # Calculate loss surface for a subset of parameter combinations
        X, Y = np.meshgrid(x_range[:8], depth_range[:8])  # Reduce for performance
        Z = np.zeros_like(X)
        
        fixed_width = true_params_3d[2]  # Use true width for 2D visualization
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                test_params = np.array([X[i, j], Y[i, j], fixed_width])
                try:
                    mfsm_response = mfsm_forward_model(test_params, cae, surrogate_model, params_scaler, fixed_params)
                    Z[i, j] = compute_inverse_loss(mfsm_response, target_2d_response)
                except:
                    Z[i, j] = 1e6

        # ==================== CREATE PARAMETERS GIF ====================
        logging.info(f"  Creating parameters GIF...")
        
        # Create parameters figure
        fig_params, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig_params.suptitle(f'Parameter Optimization - Case {case_id}\n'
                           f'True: x={true_params_3d[0]:.4f}, depth={true_params_3d[1]:.6f}, width={true_params_3d[2]:.6f}', 
                           fontsize=14)
        
        def animate_params(frame):
            progress = frame / n_points
            
            # Use real trajectory data if available, otherwise simulate
            if trajectory_params is not None:
                current_params = trajectory_params[frame]
                current_x, current_depth, current_width = current_params[0], current_params[1], current_params[2]
            else:
                # Fallback to linear interpolation (old behavior)
                current_x = param_bounds['notch_x'][0] + progress * (true_params_3d[0] - param_bounds['notch_x'][0])
                current_depth = param_bounds['notch_depth'][0] + progress * (true_params_3d[1] - param_bounds['notch_depth'][0])
                current_width = param_bounds['notch_width'][0] + progress * (true_params_3d[2] - param_bounds['notch_width'][0])
            
            # Parameter space plot (notch_x vs notch_depth)
            ax1.clear()
            contour = ax1.contour(X, Y, Z, levels=20, alpha=0.7, cmap='viridis')
            ax1.scatter(true_params_3d[0], true_params_3d[1], color='red', s=150, marker='*', label='True Parameters', edgecolor='black', linewidth=2)
            ax1.scatter(current_x, current_depth, color='lime', s=100, marker='o', label='Current Estimate', edgecolor='black', linewidth=1)
            ax1.set_xlabel('Notch X Position')
            ax1.set_ylabel('Notch Depth')
            ax1.set_title('Parameter Space (notch_x vs notch_depth)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Parameter evolution over iterations
            ax2.clear()
            
            if trajectory_params is not None:
                # Use real trajectory data
                iterations = np.arange(frame + 1)
                current_trajectory = trajectory_params[:frame + 1]
                
                x_evolution = current_trajectory[:, 0]
                depth_evolution = current_trajectory[:, 1]
                width_evolution = current_trajectory[:, 2]
                
                # Normalize for plotting
                x_norm = (x_evolution - param_bounds['notch_x'][0]) / (param_bounds['notch_x'][1] - param_bounds['notch_x'][0])
                depth_norm = (depth_evolution - param_bounds['notch_depth'][0]) / (param_bounds['notch_depth'][1] - param_bounds['notch_depth'][0])
                width_norm = (width_evolution - param_bounds['notch_width'][0]) / (param_bounds['notch_width'][1] - param_bounds['notch_width'][0])
                
                ax2.plot(iterations, x_norm, 'b-', linewidth=2, label='notch_x (normalized)', marker='o', markersize=4)
                ax2.plot(iterations, depth_norm, 'r-', linewidth=2, label='notch_depth (normalized)', marker='s', markersize=4)
                ax2.plot(iterations, width_norm, 'g-', linewidth=2, label='notch_width (normalized)', marker='^', markersize=4)
            else:
                # Fallback to linear interpolation
                iterations = np.arange(frame + 1)
                x_evolution = np.linspace(param_bounds['notch_x'][0], current_x, frame + 1)
                depth_evolution = np.linspace(param_bounds['notch_depth'][0], current_depth, frame + 1)
                width_evolution = np.linspace(param_bounds['notch_width'][0], current_width, frame + 1)
                
                ax2.plot(iterations, (x_evolution - param_bounds['notch_x'][0]) / (param_bounds['notch_x'][1] - param_bounds['notch_x'][0]), 
                        'b-', linewidth=2, label='notch_x (normalized)', marker='o', markersize=4)
                ax2.plot(iterations, (depth_evolution - param_bounds['notch_depth'][0]) / (param_bounds['notch_depth'][1] - param_bounds['notch_depth'][0]), 
                        'r-', linewidth=2, label='notch_depth (normalized)', marker='s', markersize=4)
                ax2.plot(iterations, (width_evolution - param_bounds['notch_width'][0]) / (param_bounds['notch_width'][1] - param_bounds['notch_width'][0]), 
                        'g-', linewidth=2, label='notch_width (normalized)', marker='^', markersize=4)
            
            # True parameter lines
            ax2.axhline((true_params_3d[0] - param_bounds['notch_x'][0]) / (param_bounds['notch_x'][1] - param_bounds['notch_x'][0]), 
                       color='b', linestyle='--', alpha=0.7, linewidth=2, label='True x')
            ax2.axhline((true_params_3d[1] - param_bounds['notch_depth'][0]) / (param_bounds['notch_depth'][1] - param_bounds['notch_depth'][0]), 
                       color='r', linestyle='--', alpha=0.7, linewidth=2, label='True depth')
            ax2.axhline((true_params_3d[2] - param_bounds['notch_width'][0]) / (param_bounds['notch_width'][1] - param_bounds['notch_width'][0]), 
                       color='g', linestyle='--', alpha=0.7, linewidth=2, label='True width')
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Parameter Value (Normalized)')
            ax2.set_title('Parameter Evolution During Optimization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.1, 1.1)
            
            # Current parameter values
            ax3.clear()
            param_names = ['notch_x', 'notch_depth', 'notch_width']
            current_values = [current_x, current_depth, current_width]
            true_values = [true_params_3d[0], true_params_3d[1], true_params_3d[2]]
            
            x_pos = np.arange(len(param_names))
            ax3.bar(x_pos - 0.2, current_values, 0.4, label='Current', alpha=0.7, color='lightblue')
            ax3.bar(x_pos + 0.2, true_values, 0.4, label='True', alpha=0.7, color='orange')
            ax3.set_xlabel('Parameters')
            ax3.set_ylabel('Parameter Values')
            ax3.set_title('Current vs True Parameter Values')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(param_names)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Severity classification
            severity_info = case_data.get('true_severity_classification', classify_severity(depth_width_to_severity(true_params_3d[1], true_params_3d[2])))
            current_severity = depth_width_to_severity(current_depth, current_width)
            current_severity_info = classify_severity(current_severity)
            
            ax4.clear()
            ax4.text(0.5, 0.8, f"True Severity: {severity_info['label']}", transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
            ax4.text(0.5, 0.6, f"Current Severity: {current_severity_info['label']}", transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
            ax4.text(0.5, 0.4, f"Progress: {progress*100:.1f}%", transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
            ax4.text(0.5, 0.2, f"Iteration: {frame+1}/{n_points}", transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('Optimization Progress')
            ax4.axis('off')
            
            return []
        
        # Create parameters animation
        anim_params = animation.FuncAnimation(fig_params, animate_params, frames=n_points, 
                                            interval=1000/CONFIG['ANIMATION_FPS'], blit=False)
        
        # Save parameters animation
        params_filename = os.path.join(animation_folder, f'optimization_parameters_case_{case_id}.gif')
        logging.info(f"  Saving parameters animation to: {params_filename}")
        os.makedirs(os.path.dirname(params_filename), exist_ok=True)
        anim_params.save(params_filename, writer='pillow', fps=CONFIG['ANIMATION_FPS'], dpi=80)
        plt.close(fig_params)

        # ==================== CREATE RESPONSE GIF ====================
        logging.info(f"  Creating response GIF...")
        
        # Create response figure
        fig_response, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig_response.suptitle(f'Response Optimization - Case {case_id}', fontsize=14)
        
        time_steps = np.arange(len(target_2d_response))
        
        def animate_response(frame):
            progress = frame / n_points
            
            # Use real trajectory data if available, otherwise simulate
            if trajectory_params is not None and trajectory_responses is not None:
                current_params = trajectory_params[frame]
                current_response = trajectory_responses[frame]
                current_loss = trajectory_losses[frame] if trajectory_losses is not None else None
            else:
                # Fallback to linear interpolation (old behavior)
                current_x = param_bounds['notch_x'][0] + progress * (true_params_3d[0] - param_bounds['notch_x'][0])
                current_depth = param_bounds['notch_depth'][0] + progress * (true_params_3d[1] - param_bounds['notch_depth'][0])
                current_width = param_bounds['notch_width'][0] + progress * (true_params_3d[2] - param_bounds['notch_width'][0])
                current_params = np.array([current_x, current_depth, current_width])
                current_response = None
                current_loss = None
            
            # Response comparison
            ax1.clear()
            ax1.plot(time_steps, target_2d_response, 'b-', linewidth=2, label='Target 2D Response', alpha=0.8)
            
            try:
                if current_response is None:
                    current_response = mfsm_forward_model(current_params, cae, surrogate_model, params_scaler, fixed_params)
                
                ax1.plot(time_steps, current_response, 'r--', linewidth=2, label='MFSM Prediction', alpha=0.8)
                
                # Calculate current error
                if current_loss is not None:
                    # Use actual loss from optimization
                    if CONFIG.get('LOSS_TYPE', 'mse') == 'mse':
                        mse = current_loss
                    else:
                        # For composite loss, calculate MSE separately for display
                        mse = np.mean((target_2d_response - current_response) ** 2)
                else:
                    mse = np.mean((target_2d_response - current_response) ** 2)
                
                r2 = r2_score(target_2d_response, current_response)
                
                # Add trajectory info if available
                trajectory_info = ""
                if trajectory_params is not None:
                    trajectory_info = f"\nActual Optimization Step: {frame+1}"
                
                ax1.text(0.02, 0.98, f'MSE: {mse:.6f}\nR²: {r2:.4f}\nProgress: {progress*100:.1f}%{trajectory_info}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            except:
                ax1.plot(time_steps, target_2d_response * 0, 'r--', linewidth=2, label='LFSM Prediction (Error)', alpha=0.8)
                ax1.text(0.02, 0.98, f'Error in prediction\nProgress: {progress*100:.1f}%', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3))
            
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Response Amplitude')
            ax1.set_title('Response Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Error evolution
            ax2.clear()
            if frame > 0:
                if trajectory_losses is not None:
                    # Use real trajectory loss data
                    current_losses = trajectory_losses[:frame + 1]
                    
                    # Convert composite loss to MSE for display if needed
                    if CONFIG.get('LOSS_TYPE', 'mse') != 'mse':
                        error_evolution = []
                        for f in range(frame + 1):
                            try:
                                response = trajectory_responses[f]
                                mse_error = np.mean((target_2d_response - response) ** 2)
                                error_evolution.append(mse_error)
                            except:
                                error_evolution.append(current_losses[f])
                    else:
                        error_evolution = current_losses.tolist()
                    
                    iterations = np.arange(len(error_evolution))
                    ax2.plot(iterations, error_evolution, 'g-', linewidth=2, marker='o', markersize=4, label='Actual MSE')
                    ax2.set_title('Actual Error Evolution During Optimization')
                else:
                    # Fallback to simulated error evolution
                    error_evolution = []
                    for f in range(frame + 1):
                        prog = f / n_points
                        sim_x = param_bounds['notch_x'][0] + prog * (true_params_3d[0] - param_bounds['notch_x'][0])
                        sim_depth = param_bounds['notch_depth'][0] + prog * (true_params_3d[1] - param_bounds['notch_depth'][0])
                        sim_width = param_bounds['notch_width'][0] + prog * (true_params_3d[2] - param_bounds['notch_width'][0])
                        sim_params = np.array([sim_x, sim_depth, sim_width])
                        
                        try:
                            sim_response = mfsm_forward_model(sim_params, cae, surrogate_model, params_scaler, fixed_params)
                            error = np.mean((target_2d_response - sim_response) ** 2)
                            error_evolution.append(error)
                        except:
                            error_evolution.append(1.0)
                    
                    iterations = np.arange(len(error_evolution))
                    ax2.plot(iterations, error_evolution, 'g-', linewidth=2, marker='o', markersize=4, label='Simulated MSE')
                    ax2.set_title('Simulated Error Evolution (Linear Interpolation)')
                
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Mean Squared Error')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')
            
            return []
        
        # Create response animation
        anim_response = animation.FuncAnimation(fig_response, animate_response, frames=n_points, 
                                              interval=1000/CONFIG['ANIMATION_FPS'], blit=False)
        
        # Save response animation
        response_filename = os.path.join(animation_folder, f'optimization_response_case_{case_id}.gif')
        logging.info(f"  Saving response animation to: {response_filename}")
        os.makedirs(os.path.dirname(response_filename), exist_ok=True)
        anim_response.save(response_filename, writer='pillow', fps=CONFIG['ANIMATION_FPS'], dpi=80)
        plt.close(fig_response)
        
        # Verify both files were created
        files_created = []
        for filename, name in [(params_filename, "parameters"), (response_filename, "response")]:
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                logging.info(f"✓ {name.capitalize()} animation saved: {filename} ({file_size} bytes)")
                files_created.append(filename)
            else:
                logging.error(f"✗ {name.capitalize()} animation file was not created: {filename}")
        
        if len(files_created) == 2:
            logging.info(f"✅ Both animations created successfully for case {case_id}")
        else:
            logging.warning(f"⚠️ Only {len(files_created)}/2 animations created for case {case_id}")
        
    except Exception as e:
        logging.warning(f"Failed to create animation for case {case_id}: {e}")

# --- Evaluation Functions (modified for severity-based approach) ---
def evaluate_parameter_estimation(true_params_3d, estimated_params_3d):
    """
    Evaluate parameter estimation performance for 3-parameter approach
    
    Args:
        true_params_3d: Ground truth parameters [notch_x, notch_depth, notch_width] 
        estimated_params_3d: Estimated parameters [notch_x, notch_depth, notch_width]
    
    Returns:
        dict: Dictionary containing various error metrics and severity classification results
    """
    
    # Standard error metrics
    absolute_errors = np.abs(estimated_params_3d - true_params_3d)
    relative_errors = np.abs((estimated_params_3d - true_params_3d) / true_params_3d) * 100
    
    # Calculate overall metrics
    mae = np.mean(absolute_errors)
    mape = np.mean(relative_errors)
    rmse = np.sqrt(np.mean((estimated_params_3d - true_params_3d) ** 2))
    
    param_names = CONFIG['NOTCH_PARAMS']
    
    # Calculate severities for classification comparison
    true_depth, true_width = true_params_3d[1], true_params_3d[2]
    estimated_depth, estimated_width = estimated_params_3d[1], estimated_params_3d[2]
    
    # Use both regular and weighted severity
    true_severity = depth_width_to_severity(true_depth, true_width)
    estimated_severity = depth_width_to_severity(estimated_depth, estimated_width)
    
    # Classification based on severity
    true_classification = classify_severity(true_severity)
    estimated_classification = classify_severity(estimated_severity)
    
    # Classification success
    classification_success = (true_classification['category'] == estimated_classification['category'])
    
    results = {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'absolute_errors': {param_names[i]: absolute_errors[i] for i in range(len(param_names))},
        'relative_errors': {param_names[i]: relative_errors[i] for i in range(len(param_names))},
        'true_params': {param_names[i]: true_params_3d[i] for i in range(len(param_names))},
        'estimated_params': {param_names[i]: estimated_params_3d[i] for i in range(len(param_names))},
        
        # Severity-specific results
        'true_severity': true_severity,
        'estimated_severity': estimated_severity,
        'true_severity_classification': true_classification,
        'estimated_severity_classification': estimated_classification,
        'classification_success': classification_success,
        'severity_absolute_error': abs(estimated_severity - true_severity),
        'severity_relative_error': abs(estimated_severity - true_severity) / true_severity * 100 if true_severity != 0 else 0
    }
    
    return results


def load_2d_responses_from_csv(csv_path, n_cases=10, seed=None, youngs_modulus=None, density=None, favor_severe=True):
    """
    Load 2D responses from CSV for 3-parameter approach with severity-biased sampling
    Material properties (youngs_modulus, density) are now inputs instead of fixed filters

    Args:
        csv_path: Path to the CSV file containing 2D responses
        n_cases: Number of cases to randomly sample
        seed: Random seed for reproducible sampling
        youngs_modulus: Optional filter for Young's modulus (if None, use all values)
        density: Optional filter for density (if None, use all values)
        favor_severe: Whether to use weighted sampling that favors severe cases

    Returns:
        List of dictionaries with case_id, true_params_3d, 2d_response, fixed_params
    """
    logging.info("=" * 60)
    logging.info("LOADING 2D RESPONSES FROM CSV (3-PARAMETER)")
    logging.info("=" * 60)
    logging.info(f"CSV file path: {csv_path}")
    logging.info(f"Requested cases: {n_cases}")
    logging.info(f"Random seed: {seed}")
    logging.info(f"Severity-biased sampling: {'Enabled' if favor_severe else 'Disabled'}")

    try:
        # Load CSV file
        logging.info("Reading CSV file...")
        df = pd.read_csv(csv_path)
        logging.info(f"Total rows in CSV: {len(df)}")
        logging.info(f"CSV columns: {list(df.columns)}")

        # Apply filtering criteria
        logging.info("Applying filtering criteria...")

        # Start with base filter (response_point)
        mask = (df['response_point'] == 1.9)
        logging.info(f"  response_point == 1.9")

        # Optional youngs_modulus filter
        if youngs_modulus is not None:
            mask = mask & (df['youngs_modulus'] == youngs_modulus)
            logging.info(f"  youngs_modulus == {youngs_modulus}")
        else:
            logging.info(f"  youngs_modulus: ALL VALUES (no filter)")

        # Optional density filter
        if density is not None:
            mask = mask & (df['density'] == density)
            logging.info(f"  density == {density}")
        else:
            logging.info(f"  density: ALL VALUES (no filter)")

        filtered_df = df[mask].copy()
        logging.info(f"Rows after filtering: {len(filtered_df)}")

        if len(filtered_df) == 0:
            raise ValueError("No cases found matching the filtering criteria")

        # Calculate severity for each case and implement weighted sampling if enabled
        if favor_severe and len(filtered_df) > 1:
            logging.info("Calculating severity for weighted sampling...")

            # Calculate severity values for all filtered cases
            severity_values = []
            for idx, (_, row) in enumerate(filtered_df.iterrows()):
                notch_depth = float(row['notch_depth'])
                notch_width = float(row['notch_width'])
                severity_value = depth_width_to_severity(notch_depth, notch_width)
                severity_values.append(severity_value)

            # Assign weights based on severity categories
            weights = []
            for severity in severity_values:
                severity_info = classify_severity(severity)
                category = severity_info['category']

                # Assign higher weights to severe cases
                if category == 'severe':
                    weight = 4.0  # Highest weight for severe cases
                elif category == 'moderate':
                    weight = 2.0  # Medium weight for moderate cases
                else:  # mild
                    weight = 1.0  # Lowest weight for mild cases

                weights.append(weight)

            weights = np.array(weights)

            # Log severity distribution before weighted sampling
            original_severity_counts = {}
            for severity in severity_values:
                category = classify_severity(severity)['category']
                original_severity_counts[category] = original_severity_counts.get(category, 0) + 1

            logging.info(f"Original severity distribution before weighted sampling:")
            for category in ['mild', 'moderate', 'severe']:
                count = original_severity_counts.get(category, 0)
                percentage = count / len(filtered_df) * 100
                logging.info(f"  {category.capitalize()}: {count} cases ({percentage:.1f}%)")

            # Normalize weights to probabilities
            weights = weights / weights.sum()

            # Use weighted random sampling
            if seed is not None:
                np.random.seed(seed)
                logging.info(f"Using random seed: {seed}")

            selected_indices = np.random.choice(
                filtered_df.index,
                size=min(n_cases, len(filtered_df)),
                replace=False,
                p=weights
            ).tolist()

            logging.info(f"Successfully selected {len(selected_indices)} cases using severity-biased sampling")

        else:
            # Use simple random sampling (original behavior)
            logging.info(f"Using simple random sampling (favor_severe={favor_severe})")

            if seed is not None:
                np.random.seed(seed)
                logging.info(f"Using random seed: {seed}")

            # Randomly select indices without replacement
            if len(filtered_df) >= n_cases:
                selected_indices = np.random.choice(filtered_df.index, size=n_cases, replace=False).tolist()
                logging.info(f"Successfully selected {len(selected_indices)} random cases")
            else:
                selected_indices = filtered_df.index.tolist()
                logging.warning(f"Only {len(selected_indices)} cases available, using all of them")

        # Ensure we don't exceed available cases
        if len(selected_indices) > len(filtered_df):
            selected_indices = selected_indices[:len(filtered_df)]

        if len(selected_indices) < n_cases:
            logging.warning(f"Only {len(selected_indices)} cases available, reducing n_cases from {n_cases}")
            n_cases = len(selected_indices)

        sampled_df = filtered_df.loc[selected_indices].copy()
        logging.info(f"Selected {len(sampled_df)} cases for analysis")
        
        # Extract response columns (r_0 to r_1499)
        response_cols = [f'r_{i}' for i in range(1500)]
        missing_cols = [col for col in response_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"Missing response columns: {missing_cols[:5]}..." if len(missing_cols) > 5 else str(missing_cols))
            # Use only available response columns
            response_cols = [col for col in response_cols if col in df.columns]
        
        logging.info(f"Using {len(response_cols)} response time steps")
        
        # Build case list with 3-parameter approach
        cases = []
        for idx, (_, row) in enumerate(sampled_df.iterrows()):
            # Extract 3 independent parameters directly
            notch_x = float(row['notch_x'])
            notch_depth = float(row['notch_depth'])
            notch_width = float(row['notch_width'])
            
            # Calculate severity for classification
            notch_severity = depth_width_to_severity(notch_depth, notch_width)
            severity_info = classify_severity(notch_severity)
            
            case_data = {
                'case_id': int(row['case_id']),
                'original_index': int(row.name),
                'true_params_3d': np.array([
                    notch_x,
                    notch_depth, 
                    notch_width
                ], dtype=np.float32),
                '2d_response': row[response_cols].values.astype(np.float32),
                'fixed_params': {
                    'length': float(row['length']),
                    'density': float(row['density']),
                    'youngs_modulus': float(row['youngs_modulus']),
                    'location': float(row['response_point'])
                },
                'true_severity_classification': severity_info
            }
            cases.append(case_data)
            
            if idx == 0:  # Log details for first case
                logging.info(f"Sample case details (case_id={case_data['case_id']}):") 
                logging.info(f"  True parameters [x, depth, width]: {case_data['true_params_3d']}")
                logging.info(f"  Severity classification: {severity_info['label']} ({severity_info['category']})")
                logging.info(f"  Fixed parameters: {case_data['fixed_params']}")
                logging.info(f"  2D response shape: {case_data['2d_response'].shape}")
                logging.info(f"  2D response stats: min={case_data['2d_response'].min():.12f}, max={case_data['2d_response'].max():.12f}")
        
        # Log final severity distribution after sampling
        severity_counts = {}
        for case in cases:
            category = case['true_severity_classification']['category']
            severity_counts[category] = severity_counts.get(category, 0) + 1

        logging.info(f"Final severity distribution after sampling:")
        for category, count in severity_counts.items():
            percentage = count / len(cases) * 100
            logging.info(f"  {category.capitalize()}: {count} cases ({percentage:.1f}%)")

        if favor_severe and len(filtered_df) > 1:
            # Calculate improvement in severe case selection
            original_severe_pct = original_severity_counts.get('severe', 0) / len(filtered_df) * 100
            final_severe_pct = severity_counts.get('severe', 0) / len(cases) * 100
            improvement = final_severe_pct - original_severe_pct
            logging.info(f"🎯 Severity-biased sampling improvement: Severe cases increased by {improvement:.1f}% (from {original_severe_pct:.1f}% to {final_severe_pct:.1f}%)")
        
        logging.info("✓ Successfully loaded 2D responses from CSV (3-parameter)")
        return cases
        
    except FileNotFoundError:
        logging.error(f"✗ CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logging.error(f"✗ Error loading 2D responses: {e}")
        raise


# --- Main Test Case Execution Functions ---
def run_3parameter_test_case(case_data, cae, surrogate_model, params_scaler, verbose=None,
                             database_df=None, response_cols=None):
    """
    Run a single 3-parameter inverse problem test case using real 2D response data
    Enhanced with database integration for smart optimization and data registration
    """
    if verbose is None:
        verbose = CONFIG.get('VERBOSE_LOGGING', True)

    case_id = case_data['case_id']
    true_params_3d = case_data['true_params_3d']            # [notch_x, depth, width]
    target_2d_response = case_data['2d_response']
    fixed_params = case_data['fixed_params']
    true_severity_classification = case_data['true_severity_classification']

    # Always log case start for visibility
    logging.info("\n" + "🧪" + "="*60)
    logging.info(f"3-PARAMETER INVERSE PROBLEM TEST CASE {case_id}")
    logging.info("="*62)
    logging.info("Using real 2D response as ground truth with independent parameter estimation")
    
    if verbose:
        pass  # Additional verbose logging can go here if needed

    # Run single optimization with database integration
    start_time = time.time()
    
    # Enable animations if we're going to create them later
    save_animations_for_optimization = CONFIG.get('SAVE_ALL_CASE_ANIMATIONS', False)
    
    optimization_result = estimate_notch_parameters(
        target_2d_response, cae, surrogate_model, params_scaler, fixed_params,
        method=CONFIG['OPT_METHOD'], save_animations=save_animations_for_optimization, case_id=case_id,
        database_df=database_df, response_cols=response_cols
    )
    estimation_time = time.time() - start_time
    
    # Handle the new return format
    if save_animations_for_optimization and isinstance(optimization_result, tuple):
        result, trajectory_data = optimization_result
    else:
        result = optimization_result
        trajectory_data = None

    if not result.success:
        logging.error("❌ Optimization failed!")
        logging.error(f"   Optimization message: {result.message}")
        logging.error(f"   Function evaluations: {result.nfev}")
        logging.error(f"   Final objective value: {result.fun}")
        if hasattr(result, 'x'):
            logging.error(f"   Best parameters found: {result.x}")
        
        # Still show basic parameter info even if optimization failed
        if hasattr(result, 'x') and result.x is not None:
            try:
                best_params_3d = result.x
                est_notch_x = float(best_params_3d[0])
                est_depth = float(best_params_3d[1])
                est_width = float(best_params_3d[2])
                
                # Get true parameters
                true_notch_x = float(true_params_3d[0])
                true_depth = float(true_params_3d[1])
                true_width = float(true_params_3d[2])
                
                # Show detailed parameter comparison for failed optimization
                logging.info("\n" + "="*70)
                logging.info(f"❌ CASE {case_id} RESULTS - OPTIMIZATION FAILED")
                logging.info("="*70)

                logging.info("\n📊 ESTIMATED vs TRUE PARAMETERS (Best Found):")
                logging.info(f"    notch_x:     True: {true_notch_x:.6f}, Estimated: {est_notch_x:.6f}")
                logging.info(f"    notch_depth: True: {true_depth:.6f}, Estimated: {est_depth:.6f}")
                logging.info(f"    notch_width: True: {true_width:.6f}, Estimated: {est_width:.6f}")

                # Show severity classification
                evaluation_results = evaluate_parameter_estimation(true_params_3d, best_params_3d)
                estimated_classification = evaluation_results['estimated_severity_classification']
                classification_success = evaluation_results['classification_success']

                logging.info("\n🎯 SEVERITY CLASSIFICATION:")
                logging.info(f"  True Severity:      {true_severity_classification['label']} ({true_severity_classification['category']})")
                logging.info(f"  Estimated Severity: {estimated_classification['label']} ({estimated_classification['category']})")
                logging.info(f"  Classification:     {'✅ CORRECT' if classification_success else '❌ INCORRECT'}")

                logging.info("="*70)
                logging.info(f"❌ CASE {case_id} COMPLETED (FAILED)")
                logging.info("="*70)
                
            except Exception as e:
                logging.error(f"   Error showing failed optimization results: {e}")
        
        return {
            'case_id': case_id,
            'success': False,
            'error': 'Optimization failed',
            'estimation_time': estimation_time,
            'classification_success': False
        }

    best_params_3d = result.x  # [notch_x, notch_depth, notch_width]

    # Safety check for when optimization completely fails
    if best_params_3d is None or len(best_params_3d) != 3:
        logging.error(f"❌ CRITICAL FAILURE: No valid parameters found for case {case_id}")
        logging.error(f"   Optimization result: {result}")
        return {
            'case_id': case_id,
            'success': False,
            'error': 'No valid parameters found',
            'estimation_time': estimation_time,
            'classification_success': False,
            'true_params_3d': true_params_3d,
            'target_2d_response': target_2d_response,
            'fixed_params': fixed_params
        }

    # Extract estimated parameters directly
    est_notch_x = float(best_params_3d[0])
    est_depth = float(best_params_3d[1])
    est_width = float(best_params_3d[2])

    # Extract true parameters
    true_notch_x = float(true_params_3d[0])
    true_depth = float(true_params_3d[1])
    true_width = float(true_params_3d[2])

    # Build arrays for simple error metrics across the three physical params
    true_vec = np.array([true_notch_x, true_depth, true_width], dtype=np.float64)
    est_vec = np.array([est_notch_x, est_depth, est_width], dtype=np.float64)
    abs_errors = np.abs(est_vec - true_vec)
    rel_errors = np.where(true_vec != 0.0, abs_errors / np.abs(true_vec) * 100.0, np.nan)

    mae = float(np.mean(abs_errors))
    mape = float(np.nanmean(rel_errors))
    rmse = float(np.sqrt(np.mean((est_vec - true_vec) ** 2)))

    # Generate MFSM response using best estimated parameters for comparison
    mfsm_response = mfsm_forward_model(best_params_3d, cae, surrogate_model, params_scaler, fixed_params)
    response_mse = np.mean((target_2d_response - mfsm_response) ** 2)
    response_r2 = r2_score(target_2d_response, mfsm_response)
    huber_loss = compute_huber_loss(mfsm_response, target_2d_response)

    # Severity classification evaluation
    evaluation_results = evaluate_parameter_estimation(true_params_3d, best_params_3d)
    classification_success = evaluation_results['classification_success']
    estimated_classification = evaluation_results['estimated_severity_classification']

    # Print the formatted per-case results prominently after each case
    logging.info("\n" + "="*70)
    logging.info(f"🎯 CASE {case_id} RESULTS - PARAMETER COMPARISON")
    logging.info("="*70)

    logging.info("\n📊 ESTIMATED vs TRUE PARAMETERS:")
    # notch_x
    logging.info(f"    notch_x:")
    logging.info(f"      True:      {true_notch_x:.6f}")
    logging.info(f"      Estimated: {est_notch_x:.6f}")
    logging.info(f"      Abs Error: {abs_errors[0]:.6f}")
    logging.info(f"      Rel Error: {rel_errors[0]:.2f}%")
    # notch_depth
    logging.info(f"    notch_depth:")
    logging.info(f"      True:      {true_depth:.6f}")
    logging.info(f"      Estimated: {est_depth:.6f}")
    logging.info(f"      Abs Error: {abs_errors[1]:.6f}")
    logging.info(f"      Rel Error: {rel_errors[1]:.2f}%")
    # notch_width
    logging.info(f"    notch_width:")
    logging.info(f"      True:      {true_width:.6f}")
    logging.info(f"      Estimated: {est_width:.6f}")
    logging.info(f"      Abs Error: {abs_errors[2]:.6f}")
    logging.info(f"      Rel Error: {rel_errors[2]:.2f}%")

    logging.info("\n📊 PERFORMANCE EVALUATION:")
    logging.info(f"  📈 Overall Error Metrics:")
    logging.info(f"    MAE:  {mae:.6f}")
    logging.info(f"    MAPE: {mape:.2f}%")
    logging.info(f"    RMSE: {rmse:.6f}")

    logging.info("\n🎯 SEVERITY CLASSIFICATION:")
    logging.info(f"  True Severity:      {true_severity_classification['label']} ({true_severity_classification['category']})")
    logging.info(f"  Estimated Severity: {estimated_classification['label']} ({estimated_classification['category']})")
    logging.info(f"  Classification:     {'✅ CORRECT' if classification_success else '❌ INCORRECT'}")

    logging.info("\n📈 RESPONSE QUALITY METRICS:")
    logging.info(f"  MSE: {response_mse:.8f}")
    logging.info(f"  R²: {response_r2:.12f}")
    logging.info(f"  Huber Loss: {huber_loss:.8f}")

    logging.info("="*70)
    logging.info(f"✅ CASE {case_id} COMPLETED")
    logging.info("="*70)


    # Summary logging (existing)
    if verbose:
        logging.info("\n🔬 STEP 4: Response Verification")
        logging.info(f"  MSE: {response_mse:.8f}")
        logging.info(f"  R²: {response_r2:.12f}")
        logging.info(f"  Huber Loss: {huber_loss:.8f}")
        logging.info(f"  Max abs diff: {np.max(np.abs(target_2d_response - mfsm_response)):.8f}")

        logging.info("\n🎯 STEP 5: Severity Classification Evaluation")
        logging.info(f"  True Category: {true_severity_classification['label']} ({true_severity_classification['category']})")
        logging.info(f"  Estimated Category: {estimated_classification['label']} ({estimated_classification['category']})")
        logging.info(f"  Classification Success: {'✅ PASSED' if classification_success else '❌ FAILED'}")
        logging.info(f"  Severity Error (abs): {evaluation_results['severity_absolute_error']:.8f}")
        logging.info(f"  Severity Relative Error: {evaluation_results['severity_relative_error']:.2f}%")


    test_case_results = {
        'case_id': case_id,
        'success': result.success and (response_r2 > 0.0),  # keep previous flow (main will decide)
        'true_params_3d': true_params_3d,
        'target_2d_response': target_2d_response,
        'mfsm_response': mfsm_response,
        'fixed_params': fixed_params,
        'estimation_time': estimation_time,
        'optimization_result': result,
        'best_params_3d': best_params_3d,
        'best_fitness': result.fun,
        'response_mse': response_mse,
        'response_r2': response_r2,
        'huber_loss': huber_loss,
        'evaluation_results': evaluation_results,
        'classification_success': classification_success,
        'true_severity_classification': true_severity_classification,
        'estimated_severity_classification': estimated_classification,
        'optimization_trajectory': trajectory_data  # Add trajectory data for animations
    }

    return test_case_results


def main():
    """Main function to run the 3-parameter inverse problem solver (updated to use actual loaded cases)."""
    logging.info("=" * 60)
    logging.info("3-PARAMETER INVERSE PROBLEM SOLVER")
    logging.info("=" * 60)

    logging.info("🎯 Using real 2D simulation responses as ground truth")
    logging.info("🔍 MFSM used for independent 3-parameter estimation [notch_x, notch_depth, notch_width]")
    logging.info("🏷️ Success criteria: Correct severity categorization (Mild/Moderate/Severe)")
    logging.info(f"⚖️ Case selection: {'Severity-biased sampling (more severe cases)' if CONFIG['FAVOR_SEVERE_CASES'] else 'Random sampling'}")

    # Set random seed if specified for reproducibility
    if CONFIG['RANDOM_SEED'] is not None:
        np.random.seed(CONFIG['RANDOM_SEED'])
        logging.info(f"Random seed set to: {CONFIG['RANDOM_SEED']}")
    else:
        logging.info("Using random sampling (no seed set)")

    logging.info(f"Requested number of test cases: {CONFIG['NUM_TEST_CASES']}")
    logging.info(f"Verbose logging: {'Enabled' if CONFIG['VERBOSE_LOGGING'] else 'Disabled'}")
    logging.info("=" * 60)

    # 1. Load pre-trained MFSM model components
    try:
        cae, surrogate_model, params_scaler = load_mfsm_model()
    except Exception as e:
        logging.error(f"Failed to load MFSM model: {e}")
        return

    # 2. Initialize database variables for inverse problem solving
    database_df, response_cols = None, None

    # 3. Load 2D response data from CSV (3-parameter format) with severity-biased sampling
    try:
        case_data_list = load_2d_responses_from_csv(
            CONFIG['CSV_PATH'],
            n_cases=CONFIG['NUM_TEST_CASES'],
            seed=CONFIG['RANDOM_SEED'],
            favor_severe=CONFIG['FAVOR_SEVERE_CASES']  # Use configuration setting for severity-biased sampling
        )
        bias_type = "severity-biased" if CONFIG['FAVOR_SEVERE_CASES'] else "random"
        logging.info(f"✓ Loaded {len(case_data_list)} 2D response cases (3-parameter) with {bias_type} sampling")
    except Exception as e:
        logging.error(f"Failed to load 2D response data: {e}")
        return

    # Calculate baseline R2 scores for MFSM predictions before optimization
    logging.info("🔬 CALCULATING BASELINE MFSM PERFORMANCE ON TEST SET")
    baseline_r2_scores = []

    for case_idx, case_data in enumerate(case_data_list):
        try:
            # Generate MFSM prediction using true parameters
            true_params_3d = case_data['true_params_3d']
            fixed_params = case_data['fixed_params']
            target_response = case_data['2d_response']

            # Generate prediction using LFSM forward model
            baseline_prediction = mfsm_forward_model(
                true_params_3d, cae, surrogate_model, params_scaler, fixed_params, verbose=False
            )

            # Calculate R2 score
            baseline_r2 = r2_score(target_response, baseline_prediction)
            baseline_r2_scores.append(baseline_r2)

            if case_idx == 0:  # Log details for first case only
                logging.info(f"  Case {case_data['case_id']}: R² = {baseline_r2:.6f}")
                logging.info(f"    True params: x={true_params_3d[0]:.6f}, depth={true_params_3d[1]:.6f}, width={true_params_3d[2]:.6f}")

        except Exception as e:
            logging.warning(f"Error calculating baseline R2 for case {case_data['case_id']}: {e}")
            continue

    # Calculate R2 on the entire concatenated test set (all responses combined)
    whole_test_r2 = None
    if len(case_data_list) > 1:
        try:
            # Concatenate all target responses and predictions
            all_targets = []
            all_predictions = []

            for case_idx, case_data in enumerate(case_data_list):
                try:
                    true_params_3d = case_data['true_params_3d']
                    fixed_params = case_data['fixed_params']
                    target_response = case_data['2d_response']

                    prediction = mfsm_forward_model(
                        true_params_3d, cae, surrogate_model, params_scaler, fixed_params, verbose=False
                    )

                    all_targets.append(target_response)
                    all_predictions.append(prediction)

                except Exception as e:
                    logging.warning(f"Error processing case {case_data['case_id']} for whole test R2: {e}")
                    continue

            if all_targets and all_predictions:
                # Concatenate all responses
                concatenated_targets = np.concatenate(all_targets)
                concatenated_predictions = np.concatenate(all_predictions)

                # Calculate R2 on the entire concatenated dataset
                whole_test_r2 = r2_score(concatenated_targets, concatenated_predictions)

                logging.info(f"  Whole Test Set R² (concatenated): {whole_test_r2:.6f}")
                logging.info(f"    Total time steps: {len(concatenated_targets)}")
                logging.info(f"    Average per-case time steps: {len(concatenated_targets)/len(all_targets):.0f}")

        except Exception as e:
            logging.warning(f"Error calculating whole test set R2: {e}")

    # Log comprehensive baseline statistics
    if baseline_r2_scores:
        r2_array = np.array(baseline_r2_scores)
        baseline_avg_r2 = np.mean(r2_array)
        baseline_std_r2 = np.std(r2_array)
        baseline_min_r2 = np.min(r2_array)
        baseline_max_r2 = np.max(r2_array)
        baseline_median_r2 = np.median(r2_array)

        logging.info(f"✓ BASELINE MFSM PERFORMANCE SUMMARY:")
        logging.info(f"  Test cases processed: {len(baseline_r2_scores)}/{len(case_data_list)}")

        # Show both individual case statistics and whole test statistics
        logging.info(f"  📊 INDIVIDUAL CASE STATISTICS:")
        logging.info(f"    Average R²: {baseline_avg_r2:.6f} ± {baseline_std_r2:.6f}")
        logging.info(f"    R² Range: [{baseline_min_r2:.6f}, {baseline_max_r2:.6f}]")
        logging.info(f"    R² Median: {baseline_median_r2:.6f}")
        logging.info(f"    R² Quartiles: Q25={np.percentile(r2_array, 25):.6f}, Q75={np.percentile(r2_array, 75):.6f}")

        if whole_test_r2 is not None:
            logging.info(f"  📊 WHOLE TEST SET STATISTICS:")
            logging.info(f"    Overall R² (concatenated): {whole_test_r2:.6f}")

        # Distribution analysis
        excellent_count = np.sum(r2_array >= 0.95)
        good_count = np.sum((r2_array >= 0.80) & (r2_array < 0.95))
        fair_count = np.sum((r2_array >= 0.50) & (r2_array < 0.80))
        poor_count = np.sum(r2_array < 0.50)

        logging.info(f"  Performance Distribution:")
        logging.info(f"    Excellent (R² ≥ 0.95): {excellent_count} cases ({excellent_count/len(r2_array)*100:.1f}%)")
        logging.info(f"    Good (0.80 ≤ R² < 0.95): {good_count} cases ({good_count/len(r2_array)*100:.1f}%)")
        logging.info(f"    Fair (0.50 ≤ R² < 0.80): {fair_count} cases ({fair_count/len(r2_array)*100:.1f}%)")
        logging.info(f"    Poor (R² < 0.50): {poor_count} cases ({poor_count/len(r2_array)*100:.1f}%)")
    else:
        logging.warning("⚠️  Could not calculate baseline R2 scores for any test cases")

    # Use actual number of loaded cases and update CONFIG so logs match reality
    num_cases_loaded = len(case_data_list)
    if num_cases_loaded < CONFIG['NUM_TEST_CASES']:
        logging.warning(f"Requested {CONFIG['NUM_TEST_CASES']} cases but only {num_cases_loaded} were loaded. Adjusting to {num_cases_loaded}.")
    CONFIG['NUM_TEST_CASES'] = num_cases_loaded

    # Log case distribution by severity for information
    severe_count = sum(1 for case in case_data_list if case['true_severity_classification']['category'] == 'severe')
    moderate_count = sum(1 for case in case_data_list if case['true_severity_classification']['category'] == 'moderate')
    mild_count = sum(1 for case in case_data_list if case['true_severity_classification']['category'] == 'mild')
    logging.info(f"Case distribution by severity: Severe={severe_count}, Moderate={moderate_count}, Mild={mild_count}")

    # 3. Setup animation folder
    animation_folder = None
    if CONFIG.get('SAVE_ANIMATIONS', False) and CONFIG.get('SAVE_ALL_CASE_ANIMATIONS', False):
        animation_folder = CONFIG.get('ANIMATION_FOLDER', '/home/user2/Music/optimization_animations')
        # Use absolute path to ensure it's created in the right location
        animation_folder = os.path.abspath(animation_folder)
        try:
            os.makedirs(animation_folder, exist_ok=True)
            logging.info(f"✓ Animation folder ready: {animation_folder}")
            # Test write permissions
            test_file = os.path.join(animation_folder, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logging.info(f"✓ Animation folder write permissions verified")
        except Exception as e:
            logging.error(f"✗ Failed to setup animation folder: {e}")
            animation_folder = None

    # 4. Run multiple test cases
    all_results = []
    successful_cases = 0
    classification_success_count = 0

    total_r2 = 0.0
    total_objective = 0.0

    severity_confusion = {'mild': {'mild': 0, 'moderate': 0, 'severe': 0},
                         'moderate': {'mild': 0, 'moderate': 0, 'severe': 0},
                         'severe': {'mild': 0, 'moderate': 0, 'severe': 0}}

    logging.info(f"\nRunning {CONFIG['NUM_TEST_CASES']} 3-parameter inverse problem test cases...")

    for case_idx in range(CONFIG['NUM_TEST_CASES']):
        try:
            case_data = case_data_list[case_idx]
            case_results = run_3parameter_test_case(
                case_data, cae, surrogate_model, params_scaler,
                verbose=CONFIG['VERBOSE_LOGGING'],
                database_df=database_df, response_cols=response_cols
            )

            if case_results.get('success'):
                successful_cases += 1
                total_r2 += case_results['response_r2']
                total_objective += case_results['best_fitness']

            # Track classification performance
            if case_results.get('classification_success'):
                classification_success_count += 1

            # Update confusion matrix if categories available
            true_cat = case_results['true_severity_classification']['category']
            est_cat = case_results['estimated_severity_classification']['category']
            severity_confusion[true_cat][est_cat] += 1

            # Show quick summary after each case
            case_status = "✅ SUCCESS" if case_results.get('success') else "❌ FAILED"
            classification_status = "✅ CORRECT" if case_results.get('classification_success') else "❌ INCORRECT"
            logging.info(f"\n📋 CASE {case_results['case_id']} SUMMARY: {case_status} | Classification: {classification_status}")
            if case_results.get('success'):
                logging.info(f"   R²: {case_results.get('response_r2', 0):.4f} | Best Fitness: {case_results.get('best_fitness', 0):.6f}")
            logging.info(f"   True: {case_results['true_severity_classification']['label']} | Estimated: {case_results['estimated_severity_classification']['label']}")
            logging.info("")

            all_results.append(case_results)

            # Create individual case animation if enabled - do this immediately after each case
            if animation_folder and CONFIG.get('SAVE_ALL_CASE_ANIMATIONS', False):
                try:
                    logging.info(f"🎬 Creating animation for case {case_results['case_id']}...")
                    create_optimization_animation(
                        case_data, cae, surrogate_model, params_scaler, 
                        case_results['case_id'], animation_folder,
                        optimization_trajectory=case_results.get('optimization_trajectory')
                    )
                    logging.info(f"✅ Animation completed for case {case_results['case_id']}")
                except Exception as e:
                    logging.error(f"❌ Failed to create animation for case {case_results['case_id']}: {e}")
                    import traceback
                    logging.error(f"Full traceback: {traceback.format_exc()}")
                    
            # Force immediate flush of logs to see animation progress
            import sys
            sys.stdout.flush()

        except Exception as e:
            logging.error(f"Error in test case {case_idx + 1}: {e}")

    # 4. Calculate overall statistics and save results (if any successful)
    if successful_cases > 0:
        logging.info("\n" + "=" * 60)
        logging.info("3-PARAMETER OVERALL RESULTS SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Successful cases: {successful_cases}/{CONFIG['NUM_TEST_CASES']}")
        logging.info(f"Success rate: {successful_cases/CONFIG['NUM_TEST_CASES']*100:.1f}%")

        classification_success_rate = classification_success_count / CONFIG['NUM_TEST_CASES'] * 100
        logging.info(f"🎯 SEVERITY CLASSIFICATION PERFORMANCE:")
        logging.info(f"  Classification success rate: {classification_success_rate:.1f}% ({classification_success_count}/{CONFIG['NUM_TEST_CASES']})")

        logging.info("  📊 Confusion Matrix:")
        logging.info("      Predicted:  Mild  Moderate  Severe")
        for true_cat in ['mild', 'moderate', 'severe']:
            counts = severity_confusion[true_cat]
            logging.info(f"  True {true_cat.capitalize():>8}: {counts['mild']:>5} {counts['moderate']:>9} {counts['severe']:>7}")

        avg_r2 = total_r2 / successful_cases
        avg_objective = total_objective / successful_cases

        logging.info(f"\n📊 PARAMETER ESTIMATION STATISTICS:")
        logging.info(f"  Average response R²: {avg_r2:.4f}")
        logging.info(f"  Average objective value: {avg_objective:.8f}")

        # Save detailed results to file
        results_file = f"severity_based_inverse_problem_results.txt"
        with open(results_file, 'w') as f:
            f.write("SEVERITY-BASED Inverse Problem Results (3-Parameter with Individual Depth/Width)\n")
            f.write("="*70 + "\n\n")
            f.write(f"Configuration:\n{CONFIG}\n\n")
            f.write(f"Overall Statistics:\n")
            f.write(f"Successful cases: {successful_cases}/{CONFIG['NUM_TEST_CASES']}\n")
            f.write(f"Success rate: {successful_cases/CONFIG['NUM_TEST_CASES']*100:.1f}%\n")
            f.write(f"Classification success rate: {classification_success_rate:.1f}%\n")
            f.write(f"Average response R²: {avg_r2:.4f}\n")
            f.write(f"Average objective value: {avg_objective:.8f}\n")
            f.write("\n")
            f.write("Confusion Matrix:\n")
            f.write("      Predicted:  Mild  Moderate  Severe\n")
            for true_cat in ['mild', 'moderate', 'severe']:
                counts = severity_confusion[true_cat]
                f.write(f"True {true_cat.capitalize():>8}: {counts['mild']:>5} {counts['moderate']:>9} {counts['severe']:>7}\n")
            f.write("\n")
            for i, result in enumerate(all_results):
                if result.get('success', False):
                    # Extract individual parameters
                    true_x, true_depth, true_width = result['true_params_3d']
                    best_x, best_depth, best_width = result['best_params_3d']
                    
                    # Calculate severity values
                    true_severity_value = depth_width_to_severity(true_depth, true_width)
                    best_severity_value = depth_width_to_severity(best_depth, best_width)
                    
                    f.write(f"Test Case {result['case_id']}:\n")
                    f.write(f"  Success: {result['success']}\n")
                    f.write(f"  True params [x, depth, width]: {result['true_params_3d'].tolist()}\n")
                    f.write(f"  Best params [x, depth, width]: {result['best_params_3d'].tolist()}\n")
                    f.write(f"  True severity value: {true_severity_value:.12f}\n")
                    f.write(f"  Best severity value: {best_severity_value:.12f}\n")
                    f.write(f"  True severity: {result['true_severity_classification']['label']}\n")
                    f.write(f"  Estimated severity: {result['estimated_severity_classification']['label']}\n")
                    f.write(f"  Classification success: {result['classification_success']}\n")
                    f.write(f"  Response R²: {result['response_r2']:.4f}\n")
                    f.write(f"  Objective value: {result['best_fitness']:.8f}\n")
                    f.write("\n")

        logging.info(f"Detailed results saved to: {results_file}")

        # Final pass/fail
        overall_pass = classification_success_rate >= 70.0
        final_emoji = "✅ PASSED" if overall_pass else "❌ FAILED"
        logging.info("\n" + "🏆" + "="*60)
        logging.info(f"FINAL RESULT: {final_emoji}")
        logging.info(f"Classification Success Rate: {classification_success_rate:.1f}%")
        logging.info(f"Threshold for passing: 70%")
        if overall_pass:
            logging.info("🎉 The 3-parameter inverse problem solver successfully")
            logging.info("   categorizes notch severity with acceptable accuracy!")
        else:
            logging.info("⚠️  The 3-parameter approach needs improvement.")
            logging.info("   Consider adjusting optimization parameters or thresholds.")
        logging.info("=" * 62)

    else:
        logging.error("No successful test cases!")

    logging.info("\n" + "=" * 60)
    logging.info("3-PARAMETER INVERSE PROBLEM SOLVER COMPLETE")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()