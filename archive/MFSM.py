import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import time
import logging
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

# For Hilbert envelope computation
from scipy.signal import hilbert, find_peaks
from scipy import signal

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# --- Configuration Dictionary ---
CONFIG = {
    # --- GPU/CPU Settings ---
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',

    # --- File Paths & Dirs ---
    'PRETRAIN_TRAIN_FILE': '/home/mecharoy/Thesis/parameters/LFSM6000train.csv',  # 1D LFSM training data
    'PRETRAIN_VAL_FILE': '/home/mecharoy/Thesis/parameters/LFSM6000test.csv',    # 1D LFSM validation data
    'TRAIN_FILE': '/home/user2/Music/abhi3/LFSM/lfsm_interleaved_2d_train.csv',
    'TEST_FILE': '/home/user2/Music/abhi3/LFSM/lfsm_interleaved_2d_test.csv',
    'OUTPUT_DIR': '/home/user2/Music/abhi3/MFSM',

    # --- Data Parameters ---
    'NUM_TIME_STEPS': 1500,
    'VAL_SPLIT_RATIO': 0.2,
    'TRAIN_NUM_PAIRS': 550,   # Number of training pairs to randomly sample from train set

    # --- Model Hyperparameters ---
    'LATENT_DIM': 100,  # Smaller latent space for response-to-response mapping
    'PRETRAIN_EPOCHS': 100,  # Epochs for pre-training on 1D LFSM data
    'FINETUNE_EPOCHS': 200,  # Epochs for fine-tuning on 2D data
    'EPOCHS': 200,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-4,
    'WEIGHT_DECAY': 1e-5,
    'EARLY_STOPPING_PATIENCE': 30,

    # --- Training Parameters ---
    'DROPOUT': 0.2,
    'CLIP_GRAD_NORM': 1.0,  # Gradient clipping
    'L2_REGULARIZATION': 1e-4,  # Additional L2 regularization for overfitting prevention
    # --- Performance/Training Tweaks ---
    'USE_AMP': True,
    'RESIDUAL_OUTPUT': True,

    # --- Region-Focused Training ---
    'REGION_WEIGHT': 1.5,  # Weight multiplier for the focus region identified by Hilbert envelope

    # --- Two-Stage Training ---
    'FINETUNE_LOSS_WEIGHT': 3.0,  # Higher weight for 2D fine-tuning loss
}

# --- Setup Logging ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG['OUTPUT_DIR'], f'response_cae_training_{timestamp}.log')),
        logging.StreamHandler()
    ]
)

def compute_hilbert_envelope_and_region(time_series, region_weight=10.0):
    """
    Compute Hilbert envelope of ground truth response and identify focus region.

    Args:
        time_series: 1D numpy array of ground truth response
        region_weight: Weight multiplier for the focus region (default: 10.0)

    Returns:
        envelope: Hilbert envelope of the time series
        peaks: Indices of the two main peaks
        region_mask: Boolean mask for the focus region between X and Y points
        region_weights: Weight array for loss function (1.0 everywhere, region_weight in focus region)
    """
    # Compute Hilbert envelope
    analytic_signal = hilbert(time_series)
    envelope = np.abs(analytic_signal)

    # Find peaks in the envelope (find all significant peaks)
    # Use a lower prominence threshold to detect more peaks
    max_env = np.max(envelope)
    if max_env > 0:
        prominence_threshold = max(0.05 * max_env, 0.01)  # More lenient threshold
    else:
        prominence_threshold = 0.01

    peaks, properties = find_peaks(envelope, prominence=prominence_threshold, distance=50)

    if len(peaks) < 2:
        logging.warning(f"Found only {len(peaks)} peaks, need at least 2. Using fallback strategy.")
        # Fallback: use approximate peak locations if not enough peaks found
        if len(peaks) == 0:
            peaks = np.array([len(time_series)//4, 3*len(time_series)//4])
        elif len(peaks) == 1:
            peaks = np.array([peaks[0], min(peaks[0] + 500, len(time_series)-1)])

    # Sort peaks by position (left to right)
    peaks = np.sort(peaks)

    # Select the first peak (leftmost) and the highest remaining peak
    if len(peaks) >= 2:
        # Always take the leftmost peak as the first peak
        left_peak_idx = peaks[0]

        # Find the highest peak among the remaining peaks
        remaining_peaks = peaks[1:]  # All peaks except the leftmost
        if len(remaining_peaks) > 0:
            # Find the peak with maximum envelope value among remaining peaks
            remaining_peak_values = envelope[remaining_peaks]
            highest_remaining_idx = remaining_peaks[np.argmax(remaining_peak_values)]
            right_peak_idx = highest_remaining_idx
        else:
            # Fallback if somehow no remaining peaks
            right_peak_idx = min(left_peak_idx + 500, len(time_series) - 1)

        logging.info(f"Selected peaks: left={left_peak_idx}, right={right_peak_idx} (highest among {len(remaining_peaks)} remaining peaks)")
    else:
        # Fallback case (should not happen after above logic)
        left_peak_idx = peaks[0] if len(peaks) > 0 else len(time_series) // 4
        right_peak_idx = peaks[1] if len(peaks) > 1 else min(left_peak_idx + 500, len(time_series) - 1)

    # Get peak heights
    left_peak_height = envelope[left_peak_idx]
    right_peak_height = envelope[right_peak_idx]

    # Find points X and Y at 0.3 * peak height
    # For left peak: go right until we hit 0.3 * height
    left_threshold = 0.3 * left_peak_height
    x_idx = left_peak_idx
    while x_idx < len(time_series) - 1 and envelope[x_idx] > left_threshold:
        x_idx += 1

    # For right peak: go left until we hit 0.3 * height
    right_threshold = 0.3 * right_peak_height
    y_idx = right_peak_idx
    while y_idx > 0 and envelope[y_idx] > right_threshold:
        y_idx -= 1

    # Create region mask and weights
    region_mask = np.zeros(len(time_series), dtype=bool)
    region_mask[x_idx:y_idx+1] = True

    region_weights = np.ones(len(time_series))
    region_weights[region_mask] = region_weight

    return envelope, (left_peak_idx, right_peak_idx), region_mask, region_weights

def create_hybrid_loss(criterion, region_weights, device, overall_weight=0.7, region_weight=0.3):
    """
    Create a hybrid loss function that balances overall performance with region-specific focus.

    Args:
        criterion: Base loss function (e.g., MSELoss)
        region_weights: Weight array for each time step
        device: Device to place the weights tensor
        overall_weight: Weight for overall loss (default: 0.7)
        region_weight: Weight for region-focused loss (default: 0.3)

    Returns:
        Hybrid loss function
    """
    def hybrid_loss(predictions, targets):
        # Standard loss (overall performance)
        overall_loss = criterion(predictions, targets)

        # Region-focused loss
        weights_tensor = torch.tensor(region_weights, dtype=torch.float32, device=device)
        weights_tensor = weights_tensor.view(1, -1)  # [1, time_steps] for broadcasting

        weighted_preds = predictions * weights_tensor
        weighted_targets = targets * weights_tensor
        region_loss = criterion(weighted_preds, weighted_targets)

        # Combine losses with weights
        total_loss = overall_weight * overall_loss + region_weight * region_loss

        return total_loss

    return hybrid_loss

# CUDA/Performance switches
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logging.info("Enabled TF32 and cuDNN benchmark for CUDA")

class LFSMPretrainDataset(Dataset):
    """
    Dataset for pre-training on 1D LFSM data.
    Uses LFSM predictions as both input and target for self-supervised learning.
    """
    def __init__(self, df, p_scaler=None, ts_scaler=None, is_training=True):
        # Extract parameter columns (no 'location' in LFSM files, use response_point instead)
        param_cols = ['notch_x', 'notch_depth', 'notch_width', 'length', 'density', 'youngs_modulus', 'response_point']

        # Extract time series columns (t_1 to t_1500)
        time_cols = [col for col in df.columns if col.startswith('t_')]
        time_cols = sorted(time_cols, key=lambda x: int(x.split('_')[1]))[:CONFIG['NUM_TIME_STEPS']]

        # Get parameter and response data
        self.params = df[param_cols].values.astype(np.float32)
        self.responses = df[time_cols].values.astype(np.float32)

        # For pre-training: use responses as both input and target (self-supervised)
        # Add small noise to input for regularization
        self.inputs = self.responses + np.random.normal(0, 0.001, self.responses.shape).astype(np.float32)
        self.targets = self.responses

        # Compute region information for each sample
        self.region_weights = []
        self.region_masks = []
        self.envelopes = []

        region_weight = CONFIG.get('REGION_WEIGHT', 10.0)

        for i, sample in enumerate(self.targets):
            envelope, peaks, region_mask, weights = compute_hilbert_envelope_and_region(sample, region_weight)
            self.envelopes.append(envelope)
            self.region_masks.append(region_mask)
            self.region_weights.append(weights)

            if i == 0:
                logging.info(f"Pre-training Sample {i}: Found peaks at {peaks}, focus region from {np.where(region_mask)[0][0]} to {np.where(region_mask)[0][-1]}")

        # Scale parameters to [-1, 1] range
        if p_scaler is None and is_training:
            self.p_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.params_scaled = self.p_scaler.fit_transform(self.params)
        else:
            self.p_scaler = p_scaler
            self.params_scaled = self.p_scaler.transform(self.params) if self.p_scaler is not None else self.params

        # Scale time series data
        if ts_scaler is None and is_training:
            all_data = np.vstack([self.inputs, self.targets])
            self.ts_scaler = StandardScaler()
            self.ts_scaler.fit(all_data)
        else:
            self.ts_scaler = ts_scaler

        if self.ts_scaler is not None:
            self.inputs_scaled = self.ts_scaler.transform(self.inputs)
            self.targets_scaled = self.ts_scaler.transform(self.targets)
        else:
            self.inputs_scaled = self.inputs
            self.targets_scaled = self.targets

        # Ensure float32 and contiguous
        self.params_scaled = np.ascontiguousarray(self.params_scaled, dtype=np.float32)

        logging.info(f"Pre-training dataset created with {len(self.targets)} samples")
        logging.info(f"Parameters shape: {self.params.shape}")
        logging.info(f"Response range: [{self.responses.min():.6f}, {self.responses.max():.6f}]")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'params': torch.tensor(self.params_scaled[idx], dtype=torch.float32),
            'input': torch.tensor(self.inputs_scaled[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets_scaled[idx], dtype=torch.float32),
            'input_raw': self.inputs[idx],
            'target_raw': self.targets[idx],
            'params_raw': self.params[idx],
            'region_weights': torch.tensor(self.region_weights[idx], dtype=torch.float32),
            'region_mask': torch.tensor(self.region_masks[idx], dtype=torch.bool),
            'envelope': torch.tensor(self.envelopes[idx], dtype=torch.float32)
        }


class ResponsePairDataset(Dataset):
    """
    Dataset for response-to-response mapping with parameters.
    Maps LFSM predictions (even rows) to ground truth (odd rows) conditioned on parameters.
    """
    def __init__(self, df, p_scaler=None, ts_scaler=None, is_training=True):
        # Extract parameter columns
        param_cols = ['notch_x', 'notch_depth', 'notch_width', 'length', 'density', 'youngs_modulus', 'location']

        # Extract time series columns (t_1 to t_1500)
        time_cols = [col for col in df.columns if col.startswith('t_')]
        time_cols = sorted(time_cols, key=lambda x: int(x.split('_')[1]))[:CONFIG['NUM_TIME_STEPS']]

        # Get parameter and response data
        params = df[param_cols].values.astype(np.float32)
        responses = df[time_cols].values.astype(np.float32)

        # Separate odd and even rows (excluding header)
        # In the CSV: row 1 = ground truth, row 2 = LFSM prediction, row 3 = ground truth, row 4 = LFSM prediction, etc.
        ground_truth_responses = responses[::2]  # Odd indices (0, 2, 4, ...) = rows 1, 3, 5, ...
        lfsm_responses = responses[1::2]  # Even indices (1, 3, 5, ...) = rows 2, 4, 6, ...

        # Parameters are the same for both rows in each pair, so take from odd rows
        params_pairs = params[::2]

        # Ensure we have matching pairs
        min_pairs = min(len(ground_truth_responses), len(lfsm_responses), len(params_pairs))
        self.ground_truth = ground_truth_responses[:min_pairs]
        self.lfsm_predictions = lfsm_responses[:min_pairs]
        self.params = params_pairs[:min_pairs]

        # Compute region information for each ground truth sample
        self.region_weights = []
        self.region_masks = []
        self.envelopes = []

        region_weight = CONFIG.get('REGION_WEIGHT', 10.0)

        for i, gt_sample in enumerate(self.ground_truth):
            envelope, peaks, region_mask, weights = compute_hilbert_envelope_and_region(gt_sample, region_weight)
            self.envelopes.append(envelope)
            self.region_masks.append(region_mask)
            self.region_weights.append(weights)

            if i == 0:  # Log info for first sample only
                logging.info(f"Sample {i}: Found peaks at {peaks}, focus region from {np.where(region_mask)[0][0]} to {np.where(region_mask)[0][-1]}")
                logging.info(f"Region weight multiplier: {region_weight}")

        # Scale parameters to [-1, 1] range like in LFSMIII
        if p_scaler is None and is_training:
            self.p_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.params_scaled = self.p_scaler.fit_transform(self.params)
        else:
            self.p_scaler = p_scaler
            self.params_scaled = self.p_scaler.transform(self.params) if self.p_scaler is not None else self.params

        # Scale time series data
        if ts_scaler is None and is_training:
            # Fit scaler on training data (both input and output responses)
            all_data = np.vstack([self.ground_truth, self.lfsm_predictions])
            self.ts_scaler = StandardScaler()
            self.ts_scaler.fit(all_data)
        else:
            self.ts_scaler = ts_scaler

        if self.ts_scaler is not None:
            self.ground_truth_scaled = self.ts_scaler.transform(self.ground_truth)
            self.lfsm_scaled = self.ts_scaler.transform(self.lfsm_predictions)
        else:
            self.ground_truth_scaled = self.ground_truth
            self.lfsm_scaled = self.lfsm_predictions

        # Ensure float32 and contiguous for efficiency
        self.params_scaled = np.ascontiguousarray(self.params_scaled, dtype=np.float32)

        logging.info(f"Dataset created with {len(self.ground_truth)} response pairs")
        logging.info(f"Parameters shape: {self.params.shape}")
        logging.info(f"Ground truth range: [{self.ground_truth.min():.6f}, {self.ground_truth.max():.6f}]")
        logging.info(f"LFSM predictions range: [{self.lfsm_predictions.min():.6f}, {self.lfsm_predictions.max():.6f}]")

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        return {
            'params': torch.tensor(self.params_scaled[idx], dtype=torch.float32),  # Parameters
            'input': torch.tensor(self.lfsm_scaled[idx], dtype=torch.float32),  # LFSM prediction
            'target': torch.tensor(self.ground_truth_scaled[idx], dtype=torch.float32),  # Ground truth
            'input_raw': self.lfsm_predictions[idx],  # For evaluation
            'target_raw': self.ground_truth[idx],  # For evaluation
            'params_raw': self.params[idx],  # For evaluation
            'region_weights': torch.tensor(self.region_weights[idx], dtype=torch.float32),  # Region weights for loss
            'region_mask': torch.tensor(self.region_masks[idx], dtype=torch.bool),  # Region mask
            'envelope': torch.tensor(self.envelopes[idx], dtype=torch.float32)  # Hilbert envelope
        }

# --- U-Net with Attention and FiLM Conditioning ---

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies affine transformation conditioned on parameters: γ(params) ⊙ features + β(params)

    Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
    """
    def __init__(self, num_channels, params_dim):
        super(FiLMLayer, self).__init__()
        # Parameter network generates scaling (gamma) and shifting (beta) factors
        self.param_net = nn.Sequential(
            nn.Linear(params_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2 * num_channels)  # 2x for gamma and beta
        )

    def forward(self, x, params):
        """
        Args:
            x: Feature tensor (B, C, T) where B=batch, C=channels, T=time_steps
            params: Parameter tensor (B, P) where P=parameter dimension
        Returns:
            Modulated features (B, C, T)
        """
        # Generate gamma and beta from parameters
        film_params = self.param_net(params)  # (B, 2*C)
        gamma, beta = torch.chunk(film_params, 2, dim=1)  # Each (B, C)

        # Reshape for broadcasting: (B, C, 1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        # Apply affine transformation
        return gamma * x + beta


class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention for 1D temporal sequences.
    Captures long-range dependencies with O(1) sequential operations.

    Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
    """
    def __init__(self, channels, num_heads=4):
        super(SelfAttention1D, self).__init__()
        self.num_heads = num_heads
        self.channels = channels

        assert channels % num_heads == 0, "Channels must be divisible by num_heads"

        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

        # Layer normalization
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, T)
        Returns:
            Attention output (B, C, T)
        """
        B, C, T = x.shape

        # Residual connection
        residual = x

        # LayerNorm (need to transpose for LayerNorm)
        x_norm = x.transpose(1, 2)  # (B, T, C)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.transpose(1, 2)  # (B, C, T)

        # Compute Q, K, V
        qkv = self.qkv(x_norm)  # (B, 3*C, T)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, T)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each (B, num_heads, head_dim, T)

        # Transpose for attention computation
        q = q.transpose(-2, -1)  # (B, num_heads, T, head_dim)
        k = k.transpose(-2, -1)  # (B, num_heads, T, head_dim)
        v = v.transpose(-2, -1)  # (B, num_heads, T, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attn_out = torch.matmul(attn_probs, v)  # (B, num_heads, T, head_dim)

        # Reshape and project
        attn_out = attn_out.transpose(-2, -1).reshape(B, C, T)  # (B, C, T)
        attn_out = self.proj_out(attn_out)

        # Residual connection
        return residual + attn_out


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections in U-Net.
    Learns to focus on salient features by suppressing irrelevant regions.

    Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018
    """
    def __init__(self, skip_channels, decoder_channels, params_dim):
        super(AttentionGate, self).__init__()
        self.inter_channels = skip_channels // 2

        # Transform skip connection features
        self.W_skip = nn.Conv1d(skip_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Transform decoder features
        self.W_decoder = nn.Conv1d(decoder_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Parameter influence on attention
        self.W_params = nn.Linear(params_dim, self.inter_channels)

        # Final attention coefficients
        self.psi = nn.Sequential(
            nn.Conv1d(self.inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip, decoder, params):
        """
        Args:
            skip: Skip connection features (B, C_skip, T)
            decoder: Decoder features (B, C_decoder, T)
            params: Physical parameters (B, P)
        Returns:
            Gated skip features (B, C_skip, T)
        """
        # Transform inputs
        skip_transformed = self.W_skip(skip)
        decoder_transformed = self.W_decoder(decoder)

        # Parameter influence
        param_influence = self.W_params(params).unsqueeze(-1)  # (B, inter_channels, 1)

        # Combine: skip + decoder + params
        combined = self.relu(skip_transformed + decoder_transformed + param_influence)

        # Compute attention coefficients
        attention_coeffs = self.psi(combined)  # (B, 1, T)

        # Apply gating
        return skip * attention_coeffs


class EncoderBlock(nn.Module):
    """
    U-Net encoder block with FiLM conditioning.
    Double convolution pattern with parameter-based modulation.
    """
    def __init__(self, in_channels, out_channels, params_dim, kernel_size=3, padding=1):
        super(EncoderBlock, self).__init__()

        # First convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.film1 = FiLMLayer(out_channels, params_dim)
        self.norm1 = nn.LayerNorm(out_channels)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(CONFIG['DROPOUT'])

        # Second convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.film2 = FiLMLayer(out_channels, params_dim)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(CONFIG['DROPOUT'])

        # Residual connection if dimensions match
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, params):
        """
        Args:
            x: Input features (B, C_in, T)
            params: Physical parameters (B, P)
        Returns:
            Output features (B, C_out, T)
        """
        residual = self.residual(x)

        # First conv + FiLM
        out = self.conv1(x)
        out = self.film1(out, params)
        out = out.transpose(1, 2)  # (B, T, C) for LayerNorm
        out = self.norm1(out)
        out = out.transpose(1, 2)  # (B, C, T)
        out = self.act1(out)
        out = self.dropout1(out)

        # Second conv + FiLM
        out = self.conv2(out)
        out = self.film2(out, params)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.act2(out)
        out = self.dropout2(out)

        # Residual connection
        return out + residual


class DecoderBlock(nn.Module):
    """
    U-Net decoder block with attention-gated skip connections and FiLM conditioning.
    """
    def __init__(self, in_channels, skip_channels, out_channels, params_dim, kernel_size=3, padding=1):
        super(DecoderBlock, self).__init__()

        # Upsampling
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)

        # Attention gate for skip connection
        self.attention_gate = AttentionGate(skip_channels, in_channels, params_dim)

        # Convolutions after concatenation
        self.conv1 = nn.Conv1d(in_channels + skip_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.film1 = FiLMLayer(out_channels, params_dim)
        self.norm1 = nn.LayerNorm(out_channels)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(CONFIG['DROPOUT'])

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.film2 = FiLMLayer(out_channels, params_dim)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(CONFIG['DROPOUT'])

    def forward(self, x, skip, params):
        """
        Args:
            x: Decoder features (B, C_in, T)
            skip: Skip connection from encoder (B, C_skip, T*2)
            params: Physical parameters (B, P)
        Returns:
            Output features (B, C_out, T*2)
        """
        # Upsample
        x_up = self.upsample(x)

        # Handle potential size mismatch due to odd dimensions
        # Can occur in either direction depending on pooling/upsampling rounding
        if x_up.shape[-1] != skip.shape[-1]:
            if x_up.shape[-1] < skip.shape[-1]:
                # x_up is smaller: pad it to match skip
                diff = skip.shape[-1] - x_up.shape[-1]
                x_up = torch.nn.functional.pad(x_up, (diff // 2, diff - diff // 2))
            else:
                # x_up is larger: center-crop it to match skip
                diff = x_up.shape[-1] - skip.shape[-1]
                start = diff // 2
                end = x_up.shape[-1] - (diff - diff // 2)
                x_up = x_up[:, :, start:end]

        # Apply attention gate to skip connection
        skip_gated = self.attention_gate(skip, x_up, params)

        # Concatenate
        out = torch.cat([x_up, skip_gated], dim=1)

        # First conv + FiLM
        out = self.conv1(out)
        out = self.film1(out, params)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.act1(out)
        out = self.dropout1(out)

        # Second conv + FiLM
        out = self.conv2(out)
        out = self.film2(out, params)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.act2(out)
        out = self.dropout2(out)

        return out


class UNet1D(nn.Module):
    """
    1D U-Net with FiLM conditioning, self-attention, and attention gates.

    Architecture:
    - 4 encoder levels with progressive downsampling
    - Bottleneck with self-attention for long-range dependencies
    - 4 decoder levels with attention-gated skip connections
    - FiLM conditioning at every level for parameter influence
    - Residual learning: predicts corrections to input

    References:
    - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
    - Oktay et al., "Attention U-Net", MIDL 2018
    - Perez et al., "FiLM", AAAI 2018
    """
    def __init__(self, input_channels=1, params_dim=7, base_channels=64):
        super(UNet1D, self).__init__()

        self.use_residual = CONFIG.get('RESIDUAL_OUTPUT', True)

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)

        # Encoder path
        self.enc1 = EncoderBlock(base_channels, base_channels, params_dim)
        self.pool1 = nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)

        self.enc2 = EncoderBlock(base_channels, base_channels * 2, params_dim)
        self.pool2 = nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, params_dim)
        self.pool3 = nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8, params_dim)
        self.pool4 = nn.Conv1d(base_channels * 8, base_channels * 8, kernel_size=3, stride=2, padding=1)

        # Bottleneck with self-attention
        self.bottleneck_conv = EncoderBlock(base_channels * 8, base_channels * 16, params_dim)
        self.bottleneck_attn = SelfAttention1D(base_channels * 16, num_heads=8)

        # Decoder path
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8, params_dim)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, params_dim)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, params_dim)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels, params_dim)

        # Final output projection
        self.output_conv = nn.Conv1d(base_channels, input_channels, kernel_size=1)

    def forward(self, x, params):
        """
        Args:
            x: Input time series (B, T) or (B, 1, T)
            params: Physical parameters (B, P)
        Returns:
            Corrected output (B, T), latent representation (B, C, T_bottleneck)
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, T) -> (B, 1, T)

        original_input = x

        # Initial projection
        x0 = self.input_proj(x)

        # Encoder path with skip connections
        e1 = self.enc1(x0, params)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1, params)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2, params)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3, params)
        p4 = self.pool4(e4)

        # Bottleneck with self-attention
        bottleneck = self.bottleneck_conv(p4, params)
        bottleneck = self.bottleneck_attn(bottleneck)

        # Decoder path with attention-gated skip connections
        d4 = self.dec4(bottleneck, e4, params)
        d3 = self.dec3(d4, e3, params)
        d2 = self.dec2(d3, e2, params)
        d1 = self.dec1(d2, e1, params)

        # Final output
        correction = self.output_conv(d1)

        # Residual learning
        if self.use_residual:
            output = original_input + correction
        else:
            output = correction

        # Remove channel dimension for compatibility
        output = output.squeeze(1)  # (B, 1, T) -> (B, T)

        return output, bottleneck

    def apply_xavier_init(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

def load_and_split_data():
    """Load data; use full train for training and sample validation from test set."""
    logging.info("=== LOADING INTERLEAVED DATA ===")

    # Load training data
    logging.info(f"Loading training data from {CONFIG['TRAIN_FILE']}")
    df_train = pd.read_csv(CONFIG['TRAIN_FILE'])

    # Load test data
    logging.info(f"Loading test data from {CONFIG['TEST_FILE']}")
    df_test = pd.read_csv(CONFIG['TEST_FILE'])

    logging.info(f"Original training rows: {len(df_train)}, test rows: {len(df_test)}")

    # Check for NaN values
    if df_train.isnull().values.any():
        nan_count = df_train.isnull().sum().sum()
        logging.warning(f"Found {nan_count} NaN values in training data. Dropping rows with NaNs.")
        df_train.dropna(inplace=True)

    if df_test.isnull().values.any():
        nan_count = df_test.isnull().sum().sum()
        logging.warning(f"Found {nan_count} NaN values in test data. Dropping rows with NaNs.")
        df_test.dropna(inplace=True)

    # Subsample training data in PAIRS (odd: GT, even: input) before fitting scalers
    num_train_pairs = len(df_train) // 2
    desired_pairs = min(CONFIG['TRAIN_NUM_PAIRS'], num_train_pairs)
    rng = np.random.RandomState(42)
    all_train_pair_indices = np.arange(num_train_pairs)
    selected_train_pair_indices = rng.choice(all_train_pair_indices, size=desired_pairs, replace=False)

    def pair_indices_to_rows(pair_indices_array):
        rows = []
        for pair_idx in pair_indices_array:
            start = int(pair_idx) * 2
            rows.extend([start, start + 1])
        return rows

    train_rows = pair_indices_to_rows(selected_train_pair_indices)
    df_train_sub = df_train.iloc[train_rows].reset_index(drop=True)

    logging.info(f"Sampling {desired_pairs} training pairs out of {num_train_pairs} available pairs")

    # Create training dataset (fit scalers on the sampled train subset)
    train_dataset = ResponsePairDataset(df_train_sub, p_scaler=None, ts_scaler=None, is_training=True)

    # Build validation and test datasets from TEST set, preserving pair alignment
    num_test_pairs = len(df_test) // 2
    val_pairs = int(num_test_pairs * CONFIG['VAL_SPLIT_RATIO'])

    rng = np.random.RandomState(42)
    all_indices = np.arange(num_test_pairs)
    val_pair_indices = rng.choice(all_indices, size=val_pairs, replace=False)
    test_pair_indices = np.setdiff1d(all_indices, val_pair_indices, assume_unique=False)

    # Expand pair indices to row indices (odd/even rows per pair)
    val_rows = pair_indices_to_rows(val_pair_indices)
    test_rows = pair_indices_to_rows(test_pair_indices)

    val_df = df_test.iloc[val_rows].reset_index(drop=True)
    test_df = df_test.iloc[test_rows].reset_index(drop=True)

    # Use the same scalers learned from training for val/test
    val_dataset = ResponsePairDataset(val_df, p_scaler=train_dataset.p_scaler, ts_scaler=train_dataset.ts_scaler, is_training=False)
    test_dataset = ResponsePairDataset(test_df, p_scaler=train_dataset.p_scaler, ts_scaler=train_dataset.ts_scaler, is_training=False)

    logging.info(f"Validation sampled from test set: {CONFIG['VAL_SPLIT_RATIO']*100:.1f}% of {num_test_pairs} pairs")
    logging.info(f"Data split - Train (pairs): {len(train_dataset)}, Val (pairs): {len(val_dataset)}, Test (pairs): {len(test_dataset)}")

    # Return ts_scaler (time-series scaler) as 'scaler' expected by caller
    return train_dataset, val_dataset, test_dataset, train_dataset.ts_scaler

def calculate_nmse(y_true, y_pred):
    """Calculate Normalized Mean Squared Error (NMSE) as percentage."""
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

def train_model(model, train_loader, val_loader, device, loss_weight=1.0, stage_name="Training"):
    """Train the CAE model with region-focused loss.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        loss_weight: Weight multiplier for the loss (for fine-tuning)
        stage_name: Name of the training stage (for logging)
    """
    logging.info(f"=== STARTING {stage_name.upper()} ON {device} (Loss Weight: {loss_weight}) ===")

    model.to(device)

    # Optimizer and loss function
    try:
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'], fused=True)
        logging.info("Using fused AdamW optimizer")
    except Exception:
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
        logging.info("Using standard AdamW optimizer")

    # Base criterion for hybrid loss
    base_criterion = nn.MSELoss()

    # Get region weights from first batch for creating hybrid loss
    first_batch = next(iter(train_loader))
    sample_region_weights = first_batch['region_weights'][0].cpu().numpy()

    # Create hybrid loss function (70% overall + 30% region-focused)
    hybrid_criterion = create_hybrid_loss(base_criterion, sample_region_weights, device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    use_amp = CONFIG.get('USE_AMP', True) and device == 'cuda'
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    # Track best validation loss for checkpointing and early stopping
    best_val_loss = float('inf')
    best_model_path = os.path.join(CONFIG['OUTPUT_DIR'], f'best_response_cae_{timestamp}.pth')

    # Early stopping
    patience_counter = 0
    min_delta = 1e-4  # Minimum change to be considered as improvement

    # Training history
    train_losses = []
    val_losses = []

    for epoch in range(CONFIG['EPOCHS']):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            params = batch['params'].to(device)
            region_weights = batch['region_weights'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device, enabled=use_amp):
                outputs, latent = model(inputs, params)
                # Use hybrid loss (70% overall + 30% region-focused) with loss weight
                loss = loss_weight * hybrid_criterion(outputs, targets)

            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['CLIP_GRAD_NORM'])

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                params = batch['params'].to(device)
                region_weights = batch['region_weights'].to(device)

                outputs, latent = model(inputs, params)
                # Use hybrid loss for validation too (with loss weight)
                loss = loss_weight * hybrid_criterion(outputs, targets)

                total_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            logging.info(f"New best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.8f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
                logging.info(f"Early stopping triggered at epoch {epoch+1}. No improvement for {patience_counter} epochs.")
                break

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{CONFIG['EPOCHS']}] - "
                        f"Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}, "
                        f"Patience: {patience_counter}/{CONFIG['EARLY_STOPPING_PATIENCE']}")


    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    logging.info(f"Training complete. Best model loaded from {best_model_path}")

    return model, train_losses, val_losses, best_model_path

def evaluate_model(model, dataloader, scaler, device, dataset_name="Test"):
    """Evaluate the model on a dataset."""
    logging.info(f"=== EVALUATING ON {dataset_name.upper()} DATA ===")

    model.eval()
    model.to(device)

    all_targets_raw = []
    all_outputs_raw = []
    all_inputs_raw = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            params = batch['params'].to(device)

            outputs, latent = model(inputs, params)

            # Convert back to original scale
            targets_raw = scaler.inverse_transform(targets.cpu().numpy())
            outputs_raw = scaler.inverse_transform(outputs.cpu().numpy())
            inputs_raw = batch['input_raw'].numpy()

            all_targets_raw.append(targets_raw)
            all_outputs_raw.append(outputs_raw)
            all_inputs_raw.append(inputs_raw)

    # Concatenate all results
    all_targets_raw = np.vstack(all_targets_raw)
    all_outputs_raw = np.vstack(all_outputs_raw)
    all_inputs_raw = np.vstack(all_inputs_raw)

    # Calculate metrics
    mse = mean_squared_error(all_targets_raw.flatten(), all_outputs_raw.flatten())
    r2 = r2_score(all_targets_raw.flatten(), all_outputs_raw.flatten())
    nmse = calculate_nmse(all_targets_raw, all_outputs_raw)

    # Calculate per-sample R² scores
    r2_scores = []
    for i in range(len(all_targets_raw)):
        try:
            r2_sample = r2_score(all_targets_raw[i], all_outputs_raw[i])
            r2_scores.append(r2_sample)
        except:
            r2_scores.append(-np.inf)

    r2_scores = np.array(r2_scores)
    valid_r2 = r2_scores[r2_scores > -np.inf]

    logging.info(f"{dataset_name} Evaluation Results:")
    logging.info(f"  MSE: {mse:.8f}")
    logging.info(f"  R² (overall): {r2:.6f}")
    logging.info(f"  R² (mean per-sample): {np.mean(valid_r2):.6f} ± {np.std(valid_r2):.6f}")
    logging.info(f"  NMSE: {nmse:.4f}%")

    return {
        'mse': mse,
        'r2_overall': r2,
        'r2_per_sample': r2_scores,
        'nmse': nmse,
        'targets': all_targets_raw,
        'predictions': all_outputs_raw,
        'inputs': all_inputs_raw
    }

def create_visualizations(train_results, val_results, test_results, train_losses, val_losses):
    """Create comprehensive visualizations of the results."""
    logging.info("=== CREATING VISUALIZATIONS ===")

    # 1. Region-focused analysis - Hilbert envelope and focus regions
    if test_results is not None:
        plt.figure(figsize=(15, 10))

        # Get a sample ground truth and its region information
        sample_idx = len(test_results['targets']) // 2  # Middle sample
        sample_gt = test_results['targets'][sample_idx]
        sample_pred = test_results['predictions'][sample_idx]
        sample_input = test_results['inputs'][sample_idx]

        # Compute envelope and region for this sample
        envelope, peaks, region_mask, region_weights = compute_hilbert_envelope_and_region(sample_gt)

        plt.subplot(2, 3, 1)
        plt.plot(sample_gt, 'b-', alpha=0.7, label='Ground Truth')
        plt.plot(envelope, 'r-', alpha=0.8, label='Hilbert Envelope')
        plt.axvline(peaks[0], color='g', linestyle='--', alpha=0.7, label='Left Peak')
        plt.axvline(peaks[1], color='g', linestyle='--', alpha=0.7, label='Right Peak')
        plt.fill_between(range(len(sample_gt)), 0, np.max(sample_gt),
                        where=region_mask, alpha=0.3, color='orange', label='Focus Region')
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.title('Hilbert Envelope Analysis (Sample)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Region weights visualization
        plt.subplot(2, 3, 2)
        plt.plot(region_weights, 'g-', linewidth=2)
        plt.axhline(1.0, color='r', linestyle='--', alpha=0.7, label='Normal Weight (1.0)')
        plt.axhline(CONFIG['REGION_WEIGHT'], color='b', linestyle='--', alpha=0.7, label=f'Focus Weight ({CONFIG["REGION_WEIGHT"]})')
        plt.xlabel('Time Step')
        plt.ylabel('Weight Multiplier')
        plt.title('Region Weight Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Focus region prediction accuracy
        plt.subplot(2, 3, 3)
        focus_region_gt = sample_gt[region_mask]
        focus_region_pred = sample_pred[region_mask]
        focus_region_input = sample_input[region_mask]

        plt.plot(focus_region_gt, 'b-', label='Ground Truth (Focus)')
        plt.plot(focus_region_pred, 'r-', label='CAE Prediction (Focus)')
        plt.plot(focus_region_input, 'g--', label='LFSM Input (Focus)')
        plt.xlabel('Time Step (Focus Region)')
        plt.ylabel('Response')
        plt.title('Focus Region: Detailed Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Error analysis by region
        plt.subplot(2, 3, 4)
        all_region_errors = []
        all_nonregion_errors = []

        for i in range(min(50, len(test_results['targets']))):  # Analyze first 50 samples
            gt = test_results['targets'][i]
            pred = test_results['predictions'][i]

            _, _, region_mask, _ = compute_hilbert_envelope_and_region(gt)

            region_errors = np.abs(gt[region_mask] - pred[region_mask])
            nonregion_errors = np.abs(gt[~region_mask] - pred[~region_mask])

            all_region_errors.extend(region_errors)
            all_nonregion_errors.extend(nonregion_errors)

        plt.hist(all_region_errors, bins=30, alpha=0.7, label='Focus Region Errors', density=True)
        plt.hist(all_nonregion_errors, bins=30, alpha=0.7, label='Other Region Errors', density=True)
        plt.xlabel('Absolute Error')
        plt.ylabel('Density')
        plt.title('Error Distribution: Focus vs Other Regions')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Focus region improvement analysis
        plt.subplot(2, 3, 5)
        focus_improvements = []
        for i in range(min(50, len(test_results['targets']))):
            gt = test_results['targets'][i]
            lfsm_pred = test_results['inputs'][i]
            cae_pred = test_results['predictions'][i]

            _, _, region_mask, _ = compute_hilbert_envelope_and_region(gt)

            lfsm_focus_error = np.mean(np.abs(gt[region_mask] - lfsm_pred[region_mask]))
            cae_focus_error = np.mean(np.abs(gt[region_mask] - cae_pred[region_mask]))

            if lfsm_focus_error > 0:
                improvement = (lfsm_focus_error - cae_focus_error) / lfsm_focus_error
                focus_improvements.append(improvement)

        plt.hist(focus_improvements, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(np.mean(focus_improvements), color='red', linestyle='--', linewidth=2,
                   label=f'Mean Improvement: {np.mean(focus_improvements):.4f}')
        plt.xlabel('Error Reduction Ratio (Focus Region)')
        plt.ylabel('Frequency')
        plt.title('Focus Region: CAE Improvement over LFSM')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Overall vs Focus region R² comparison
        plt.subplot(2, 3, 6)
        focus_r2_scores = []
        overall_r2_scores = []

        for i in range(min(100, len(test_results['targets']))):
            gt = test_results['targets'][i]
            cae_pred = test_results['predictions'][i]

            try:
                overall_r2 = r2_score(gt, cae_pred)
                _, _, region_mask, _ = compute_hilbert_envelope_and_region(gt)
                focus_r2 = r2_score(gt[region_mask], cae_pred[region_mask])

                focus_r2_scores.append(focus_r2)
                overall_r2_scores.append(overall_r2)
            except:
                pass

        plt.scatter(overall_r2_scores, focus_r2_scores, alpha=0.6, s=20)
        plt.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Equal Performance')
        plt.xlabel('Overall R² Score')
        plt.ylabel('Focus Region R² Score')
        plt.title('Focus Region vs Overall Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], f'region_focus_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', alpha=0.7)
    plt.plot(epochs, val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], f'training_curves_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Performance comparison across datasets
    datasets = ['Test']
    results = [test_results]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # R² distribution
    axes[0, 0].hist(test_results['r2_per_sample'][test_results['r2_per_sample'] > -np.inf],
                    bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(test_results['r2_per_sample'][test_results['r2_per_sample'] > -np.inf]),
                       color='red', linestyle='--', linewidth=2,
                       label=f'Mean R² = {np.mean(test_results["r2_per_sample"][test_results["r2_per_sample"] > -np.inf]):.4f}')
    axes[0, 0].set_xlabel('R² Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Per-Sample R² Scores (Test)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Best vs worst predictions
    r2_scores = test_results['r2_per_sample']
    valid_indices = r2_scores > -np.inf
    valid_r2 = r2_scores[valid_indices]

    best_indices = np.argsort(valid_r2)[-10:]  # Top 10
    worst_indices = np.argsort(valid_r2)[:10]  # Bottom 10

    # Map back to original indices
    original_indices = np.where(valid_indices)[0]
    best_original = original_indices[best_indices]
    worst_original = original_indices[worst_indices]

    time_axis = np.arange(CONFIG['NUM_TIME_STEPS'])

    # Plot best predictions
    axes[0, 1].clear()
    for i, idx in enumerate(best_original[-3:]):  # Show top 3
        axes[0, 1].plot(time_axis, test_results['targets'][idx], 'b-', alpha=0.7, linewidth=1.5, label='Ground Truth' if i == 0 else "")
        axes[0, 1].plot(time_axis, test_results['predictions'][idx], 'r--', alpha=0.7, linewidth=1.5, label='CAE Prediction' if i == 0 else "")
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Response')
    axes[0, 1].set_title('Best 3 Predictions (Test)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot worst predictions
    axes[1, 0].clear()
    for i, idx in enumerate(worst_original[:3]):  # Show worst 3
        axes[1, 0].plot(time_axis, test_results['targets'][idx], 'b-', alpha=0.7, linewidth=1.5, label='Ground Truth' if i == 0 else "")
        axes[1, 0].plot(time_axis, test_results['predictions'][idx], 'r--', alpha=0.7, linewidth=1.5, label='CAE Prediction' if i == 0 else "")
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Response')
    axes[1, 0].set_title('Worst 3 Predictions (Test)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Overall correlation plot
    all_targets_flat = test_results['targets'].flatten()
    all_predictions_flat = test_results['predictions'].flatten()

    # Sample for plotting (too many points otherwise)
    sample_size = min(50000, len(all_targets_flat))
    sample_indices = np.random.choice(len(all_targets_flat), sample_size, replace=False)

    axes[1, 1].scatter(all_targets_flat[sample_indices], all_predictions_flat[sample_indices],
                       alpha=0.3, s=1, c='blue')

    # Perfect prediction line
    min_val = min(all_targets_flat.min(), all_predictions_flat.min())
    max_val = max(all_targets_flat.max(), all_predictions_flat.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    axes[1, 1].set_xlabel('Ground Truth Response')
    axes[1, 1].set_ylabel('CAE Predicted Response')
    axes[1, 1].set_title(f'Prediction vs Ground Truth (R² = {test_results["r2_overall"]:.4f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], f'evaluation_summary_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Improvement analysis: LFSM vs CAE-corrected vs Ground Truth
    plt.figure(figsize=(15, 10))

    # Calculate metrics for LFSM predictions vs ground truth
    lfsm_mse = mean_squared_error(test_results['targets'].flatten(), test_results['inputs'].flatten())
    lfsm_r2 = r2_score(test_results['targets'].flatten(), test_results['inputs'].flatten())

    # Metrics comparison
    plt.subplot(2, 3, 1)
    metrics = ['R²', 'MSE', 'NMSE']
    lfsm_values = [lfsm_r2, lfsm_mse, calculate_nmse(test_results['targets'], test_results['inputs'])]
    cae_values = [test_results['r2_overall'], test_results['mse'], test_results['nmse']]

    x = np.arange(len(metrics))
    width = 0.35

    # Normalize for better visualization
    plt.bar(x - width/2, [lfsm_values[0], lfsm_values[1]/max(lfsm_values[1], cae_values[1]), lfsm_values[2]/100],
            width, label='LFSM Original', alpha=0.7, color='orange')
    plt.bar(x + width/2, [cae_values[0], cae_values[1]/max(lfsm_values[1], cae_values[1]), cae_values[2]/100],
            width, label='CAE Corrected', alpha=0.7, color='blue')

    plt.xlabel('Metrics')
    plt.ylabel('Normalized Values')
    plt.title('Performance Comparison: LFSM vs CAE-Corrected')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sample comparison plots
    for i, (idx, title) in enumerate([(best_original[-1], 'Best Case'),
                                     (len(test_results['targets'])//2, 'Average Case'),
                                     (worst_original[0], 'Worst Case')]):
        plt.subplot(2, 3, i+2)
        plt.plot(time_axis, test_results['targets'][idx], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time_axis, test_results['inputs'][idx], 'r--', linewidth=1.5, alpha=0.7, label='LFSM Original')
        plt.plot(time_axis, test_results['predictions'][idx], 'b:', linewidth=2, label='CAE Corrected')
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.title(f'{title} (R² = {test_results["r2_per_sample"][idx]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Error reduction analysis
    plt.subplot(2, 3, 5)
    lfsm_errors = np.abs(test_results['targets'] - test_results['inputs'])
    cae_errors = np.abs(test_results['targets'] - test_results['predictions'])

    error_reduction = (lfsm_errors - cae_errors) / (lfsm_errors + 1e-10)  # Avoid division by zero
    error_reduction_mean = np.mean(error_reduction, axis=1)

    plt.hist(error_reduction_mean, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(np.mean(error_reduction_mean), color='red', linestyle='--', linewidth=2,
                label=f'Mean Reduction = {np.mean(error_reduction_mean):.4f}')
    plt.xlabel('Error Reduction Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Error Reduction (CAE vs LFSM)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Improvement scatter
    plt.subplot(2, 3, 6)
    lfsm_r2_per_sample = []
    for i in range(len(test_results['targets'])):
        try:
            r2_lfsm = r2_score(test_results['targets'][i], test_results['inputs'][i])
            lfsm_r2_per_sample.append(r2_lfsm)
        except:
            lfsm_r2_per_sample.append(-np.inf)

    lfsm_r2_per_sample = np.array(lfsm_r2_per_sample)
    valid_mask = (lfsm_r2_per_sample > -np.inf) & (test_results['r2_per_sample'] > -np.inf)

    plt.scatter(lfsm_r2_per_sample[valid_mask], test_results['r2_per_sample'][valid_mask],
                alpha=0.6, s=20, c='blue')
    plt.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='No Improvement Line')
    plt.xlabel('LFSM R² Score')
    plt.ylabel('CAE R² Score')
    plt.title('Per-Sample R² Improvement')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], f'improvement_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("Visualizations saved successfully!")

def save_results(test_results, model_path, scaler):
    """Save results and model artifacts."""
    # Save predictions
    results_dict = {
        'ground_truth': test_results['targets'],
        'cae_predictions': test_results['predictions'],
        'lfsm_inputs': test_results['inputs'],
        'r2_per_sample': test_results['r2_per_sample'],
        'config': CONFIG
    }

    np.savez(os.path.join(CONFIG['OUTPUT_DIR'], f'response_cae_results_{timestamp}.npz'), **results_dict)

    # Save model and scaler
    import joblib
    joblib.dump(scaler, os.path.join(CONFIG['OUTPUT_DIR'], f'response_scaler_{timestamp}.joblib'))

    # Save configuration
    import json
    with open(os.path.join(CONFIG['OUTPUT_DIR'], f'config_{timestamp}.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)

    logging.info("Results and artifacts saved successfully!")

def main():
    """Main execution function with two-stage training."""
    logging.info("=== TWO-STAGE RESPONSE-TO-RESPONSE CAE TRAINING ===")
    logging.info(f"Device: {CONFIG['DEVICE']}")
    logging.info(f"Configuration: {CONFIG}")

    # ========== STAGE 1: PRE-TRAINING ON 1D LFSM DATA ==========
    logging.info("\n" + "="*80)
    logging.info("STAGE 1: PRE-TRAINING ON 1D LFSM DATA")
    logging.info("="*80)

    # Load pre-training training data
    logging.info(f"Loading pre-training training data from {CONFIG['PRETRAIN_TRAIN_FILE']}")
    pretrain_train_df = pd.read_csv(CONFIG['PRETRAIN_TRAIN_FILE'])
    logging.info(f"Pre-training training data loaded: {len(pretrain_train_df)} samples")

    # Load pre-training validation data
    logging.info(f"Loading pre-training validation data from {CONFIG['PRETRAIN_VAL_FILE']}")
    pretrain_val_df = pd.read_csv(CONFIG['PRETRAIN_VAL_FILE'])
    logging.info(f"Pre-training validation data loaded: {len(pretrain_val_df)} samples")

    # Create pre-training datasets
    pretrain_train_dataset = LFSMPretrainDataset(pretrain_train_df, p_scaler=None, ts_scaler=None, is_training=True)
    pretrain_val_dataset = LFSMPretrainDataset(pretrain_val_df,
                                                p_scaler=pretrain_train_dataset.p_scaler,
                                                ts_scaler=pretrain_train_dataset.ts_scaler,
                                                is_training=False)

    # Create pre-training data loaders
    pretrain_train_loader = DataLoader(
        pretrain_train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        drop_last=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )
    pretrain_val_loader = DataLoader(
        pretrain_val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )

    # Initialize model
    params_dim = 7  # Same for both datasets
    model = UNet1D(input_channels=1, params_dim=params_dim, base_channels=64)
    model.apply_xavier_init()

    logging.info(f"Model architecture: U-Net1D with Attention and FiLM")
    logging.info(f"  Input dimension: {CONFIG['NUM_TIME_STEPS']}")
    logging.info(f"  Base channels: 64")
    logging.info(f"  Bottleneck channels: 1024")
    logging.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Pre-train model on 1D data
    original_epochs = CONFIG['EPOCHS']
    CONFIG['EPOCHS'] = CONFIG['PRETRAIN_EPOCHS']
    model, pretrain_losses, pretrain_val_losses, pretrain_model_path = train_model(
        model, pretrain_train_loader, pretrain_val_loader, CONFIG['DEVICE'],
        loss_weight=1.0, stage_name="Pre-training on 1D LFSM"
    )
    CONFIG['EPOCHS'] = original_epochs

    logging.info(f"Pre-training complete. Model saved at {pretrain_model_path}")

    # ========== STAGE 2: FINE-TUNING ON 2D DATA ==========
    logging.info("\n" + "="*80)
    logging.info("STAGE 2: FINE-TUNING ON 2D DATA WITH HIGHER WEIGHT")
    logging.info("="*80)

    # Load and split 2D data
    train_dataset, val_dataset, test_dataset, scaler = load_and_split_data()

    # Create 2D data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        drop_last=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )

    # Fine-tune model on 2D data with higher loss weight
    CONFIG['EPOCHS'] = CONFIG['FINETUNE_EPOCHS']
    model, finetune_losses, finetune_val_losses, model_path = train_model(
        model, train_loader, val_loader, CONFIG['DEVICE'],
        loss_weight=CONFIG['FINETUNE_LOSS_WEIGHT'], stage_name="Fine-tuning on 2D data"
    )

    logging.info(f"Fine-tuning complete. Final model saved at {model_path}")

    # ========== EVALUATION ==========
    logging.info("\n" + "="*80)
    logging.info("FINAL EVALUATION")
    logging.info("="*80)

    # Evaluate model
    test_results = evaluate_model(model, test_loader, scaler, CONFIG['DEVICE'], "Test")
    val_results = evaluate_model(model, val_loader, scaler, CONFIG['DEVICE'], "Validation")

    # Combine training histories
    all_train_losses = pretrain_losses + finetune_losses
    all_val_losses = pretrain_val_losses + finetune_val_losses

    # Create visualizations
    create_visualizations(None, val_results, test_results, all_train_losses, all_val_losses)

    # Save results
    save_results(test_results, model_path, scaler)

    logging.info("=== TWO-STAGE TRAINING AND EVALUATION COMPLETE ===")
    logging.info(f"Pre-training epochs: {CONFIG['PRETRAIN_EPOCHS']}")
    logging.info(f"Fine-tuning epochs: {CONFIG['FINETUNE_EPOCHS']}")
    logging.info(f"Fine-tuning loss weight: {CONFIG['FINETUNE_LOSS_WEIGHT']}")
    logging.info(f"Final Test Results:")
    logging.info(f"  R² Score: {test_results['r2_overall']:.6f}")
    logging.info(f"  MSE: {test_results['mse']:.8f}")
    logging.info(f"  NMSE: {test_results['nmse']:.4f}%")

def evaluate_r2_between_csvs(lfsm_file, test_responses_file, output_dir='/home/user2/Music/abhi3/comparison_plots'):
    """
    Evaluate R2 between LFSM predictions and 2D test responses, and create comparison plots.

    Args:
        lfsm_file: Path to LFSM predictions CSV
        test_responses_file: Path to 2D test responses CSV
        output_dir: Directory to save comparison plots
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading LFSM predictions from: {lfsm_file}")
    lfsm_df = pd.read_csv(lfsm_file)

    print(f"Loading test responses from: {test_responses_file}")
    test_df = pd.read_csv(test_responses_file)

    # Extract time series columns
    lfsm_time_cols = [col for col in lfsm_df.columns if col.startswith('t_')]
    test_time_cols = [col for col in test_df.columns if col.startswith('r')]

    print(f"LFSM time columns: {len(lfsm_time_cols)}")
    print(f"Test time columns: {len(test_time_cols)}")

    # Align the time series - LFSM has t_1 to t_1500, test has r0 to r1500
    # We'll use r0 to r1499 from test to match t_1 to t_1500 from LFSM
    test_time_cols = test_time_cols[1:]  # Remove r0, use r1 to r1500

    print(f"Adjusted test time columns: {len(test_time_cols)}")

    # Get parameter columns for grouping
    param_cols = ['case_id', 'response_point', 'notch_x', 'notch_depth', 'notch_width', 'length', 'density', 'youngs_modulus']

    # Group by parameters to ensure proper alignment
    lfsm_responses = lfsm_df[lfsm_time_cols].values
    test_responses = test_df[test_time_cols].values

    print(f"LFSM responses shape: {lfsm_responses.shape}")
    print(f"Test responses shape: {test_responses.shape}")

    if lfsm_responses.shape != test_responses.shape:
        print(f"Warning: Shape mismatch - LFSM: {lfsm_responses.shape}, Test: {test_responses.shape}")
        min_samples = min(lfsm_responses.shape[0], test_responses.shape[0])
        lfsm_responses = lfsm_responses[:min_samples]
        test_responses = test_responses[:min_samples]
        print(f"Using first {min_samples} samples for comparison")

    # Calculate R2 scores for each sample
    r2_scores = []
    for i in range(len(lfsm_responses)):
        try:
            r2 = r2_score(test_responses[i], lfsm_responses[i])
            r2_scores.append(r2)
        except Exception as e:
            print(f"Error calculating R2 for sample {i}: {e}")
            r2_scores.append(np.nan)

    r2_scores = np.array(r2_scores)
    valid_r2 = r2_scores[~np.isnan(r2_scores)]

    print("=== R2 EVALUATION RESULTS ===")
    print(f"Overall R2: {r2_score(test_responses, lfsm_responses):.6f}")
    print(f"Mean R2 per sample: {np.mean(valid_r2):.6f} ± {np.std(valid_r2):.6f}")
    print(f"Best R2: {np.max(valid_r2):.6f}")
    print(f"Worst R2: {np.min(valid_r2):.6f}")
    print(f"Number of valid samples: {len(valid_r2)}")

    # Create comparison plots for all samples
    print(f"\nCreating comparison plots in: {output_dir}")

    time_axis = np.arange(len(lfsm_time_cols))

    for i in range(len(lfsm_responses)):  # Process all samples
        plt.figure(figsize=(12, 8))

        # Plot both responses
        plt.subplot(2, 2, 1)
        plt.plot(time_axis, test_responses[i], 'b-', linewidth=1.5, label='2D Test Response', alpha=0.8)
        plt.plot(time_axis, lfsm_responses[i], 'r--', linewidth=1.5, label='LFSM Prediction', alpha=0.8)
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.title(f'Sample {i+1}: Response Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot difference
        plt.subplot(2, 2, 2)
        difference = test_responses[i] - lfsm_responses[i]
        plt.plot(time_axis, difference, 'g-', linewidth=1.5)
        plt.axhline(0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Difference (2D - LFSM)')
        plt.title(f'Sample {i+1}: Prediction Error')
        plt.grid(True, alpha=0.3)

        # Plot absolute error distribution
        plt.subplot(2, 2, 3)
        abs_error = np.abs(difference)
        plt.hist(abs_error, bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(np.mean(abs_error), color='red', linestyle='--', linewidth=2,
                   label=f'Mean Error: {np.mean(abs_error):.4f}')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title(f'Sample {i+1}: Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot R2 context
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f'Sample {i+1}', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.6, f'R² Score: {r2_scores[i]:.4f}', fontsize=12)
        plt.text(0.1, 0.4, f'MSE: {np.mean(difference**2):.6f}', fontsize=12)
        plt.text(0.1, 0.2, f'MAE: {np.mean(abs_error):.6f}', fontsize=12)

        # Add parameters if available
        try:
            sample_params = test_df.iloc[i][param_cols]
            param_text = '\n'.join([f'{col}: {sample_params[col]:.4f}' for col in param_cols[2:]])
            plt.text(0.5, 0.5, param_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        except:
            pass

        plt.axis('off')
        plt.title(f'Sample {i+1}: Statistics & Parameters')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_sample_{(i+1):03d}_r2_{r2_scores[i]:.4f}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        if (i + 1) % 10 == 0:
            print(f"Created {i + 1} comparison plots...")

    print(f"Completed creating {len(lfsm_responses)} comparison plots")

    # Create summary statistics plot
    plt.figure(figsize=(15, 10))

    # R2 distribution
    plt.subplot(2, 3, 1)
    plt.hist(valid_r2, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(valid_r2), color='red', linestyle='--', linewidth=2,
               label=f'Mean R²: {np.mean(valid_r2):.4f}')
    plt.xlabel('R² Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of R² Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Best and worst samples
    best_indices = np.argsort(r2_scores)[-3:]  # Top 3
    worst_indices = np.argsort(r2_scores)[:3]  # Bottom 3

    for j, (indices, title) in enumerate([(best_indices, 'Best'), (worst_indices, 'Worst')]):
        plt.subplot(2, 3, j+2)
        for k, idx in enumerate(indices):
            plt.plot(time_axis, test_responses[idx], 'b-', alpha=0.7, linewidth=1.5,
                    label=f'2D Response (R²={r2_scores[idx]:.3f})' if k == 0 else "")
            plt.plot(time_axis, lfsm_responses[idx], 'r--', alpha=0.7, linewidth=1.5,
                    label=f'LFSM Prediction' if k == 0 else "")
        plt.xlabel('Time Step')
        plt.ylabel('Response')
        plt.title(f'{title} 3 Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Overall correlation
    plt.subplot(2, 3, 5)
    # Sample for plotting (too many points otherwise)
    sample_size = min(50000, len(test_responses.flatten()))
    sample_indices = np.random.choice(len(test_responses.flatten()), sample_size, replace=False)

    plt.scatter(test_responses.flatten()[sample_indices],
               lfsm_responses.flatten()[sample_indices],
               alpha=0.3, s=1, c='blue')
    plt.xlabel('2D Test Response')
    plt.ylabel('LFSM Prediction')
    plt.title(f'Overall Correlation (R² = {r2_score(test_responses, lfsm_responses):.4f}')
    plt.grid(True, alpha=0.3)

    # Add perfect prediction line
    min_val = min(test_responses.min(), lfsm_responses.min())
    max_val = max(test_responses.max(), lfsm_responses.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.legend()

    # Summary statistics
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, 'EVALUATION SUMMARY', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f'Total Samples: {len(r2_scores)}', fontsize=12)
    plt.text(0.1, 0.6, f'Valid Samples: {len(valid_r2)}', fontsize=12)
    plt.text(0.1, 0.5, f'Overall R²: {r2_score(test_responses, lfsm_responses):.4f}', fontsize=12)
    plt.text(0.1, 0.4, f'Mean R²: {np.mean(valid_r2):.4f}', fontsize=12)
    plt.text(0.1, 0.3, f'Std R²: {np.std(valid_r2):.4f}', fontsize=12)
    plt.text(0.1, 0.2, f'Best R²: {np.max(valid_r2):.4f}', fontsize=12)
    plt.text(0.1, 0.1, f'Worst R²: {np.min(valid_r2):.4f}', fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_evaluation_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Summary plot saved!")
    return {
        'overall_r2': r2_score(test_responses, lfsm_responses),
        'mean_r2': np.mean(valid_r2),
        'std_r2': np.std(valid_r2),
        'best_r2': np.max(valid_r2),
        'worst_r2': np.min(valid_r2),
        'num_samples': len(valid_r2)
    }

if __name__ == '__main__':
    # Run the original main function or the new evaluation function
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        # Run R2 evaluation between CSVs
        lfsm_file = '/home/user2/Music/abhi3/parameters/LFSM2000testonHFSM.csv'
        test_responses_file = '/home/user2/Music/abhi3/parameters/test_responses.csv'
        results = evaluate_r2_between_csvs(lfsm_file, test_responses_file)
        print("\n=== EVALUATION COMPLETE ===")
        print(f"Results: {results}")
    else:
        # Run original training
        main()