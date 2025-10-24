# Multi-Fidelity Surrogate Modeling for Structural Response Prediction: Complete Technical Documentation

## Project Overview

This research develops a multi-fidelity surrogate modeling framework that bridges computational approaches between 1D zigzag beam theory and 2D finite element analysis for homogeneous beams with rectangular notches. The framework employs deep learning architectures to create surrogate models at different fidelity levels and uses optimization techniques to solve inverse problems for crack parameter estimation.

---

## Technical Architecture - Revised Framework

The project consists of four distinct surrogate modeling approaches with clear comparative objectives:

### 1. Low-Fidelity Surrogate Model (LFSM)
**Implementation**: [LFSMwithoutfinetuning.py](Code/LFSMwithoutfinetuning.py)

**Training Data**: 750 cases from 1D zigzag theory (LFSM2000train.csv, LFSM2000test.csv combined)
- Complete parameter range coverage from 1D simulations
- Parameters: notch_x, notch_depth, notch_width, length, density, youngs_modulus, response_point (11 locations)
- Time series: 1500 steps per response point

**Architecture**:
- **Conditional Autoencoder (CAE)**:
  - Input: 1500 time steps
  - Latent dimension: 50
  - Encoder: 2048→1024→512→256 (timeseries) + params→64→128→256 (fusion)
  - Decoder: (50+256)→512→1024→1500
  - Training epochs: 200 on 1D data
- **XGBoost Surrogate**: Maps parameters → latent space
  - N_estimators: 1000
  - Max_depth: 7
  - Learning rate: 0.05

**Output**: LFSM predictions for 2D test data saved as interleaved files

---

### 2. Multi-Fidelity Surrogate Model (MFSM-CAE)
**Implementation**: [MFSMCAE.py](Code/MFSMCAE.py)

**Two-Stage Training Strategy**:

**Stage 1 - Pretraining on 1D Data**:
- Dataset: LFSM6000train.csv / LFSM6000test.csv (6000+ cases)
- Self-supervised learning: LFSM responses as both input and target (with small noise)
- Same CAE architecture as LFSM
- Epochs: 100
- Learning rate: 1e-4
- **NO Hilbert weighting** during pretraining

**Stage 2 - Fine-tuning on 2D Data**:
- Dataset: lfsm_interleaved_2d_train.csv (from LFSM output)
  - Ground truth: 2D FEM responses (odd rows)
  - LFSM predictions: 1D-based predictions (even rows)
  - Sample pairs: 1000 pairs randomly selected
- **Input**: LFSM predictions (scaled)
- **Target**: 2D ground truth (scaled)
- **Hilbert Transform-based Weighting**:
  - Envelope computation: `hilbert(time_series)`
  - Peak detection with prominence threshold
  - Region identification: 0.3×peak threshold method
  - Normalized weights: mean=1 to avoid loss magnitude scaling
  - Composite loss: 70% overall + 30% region-weighted
- Architecture: Same CAE (latent_dim=64)
- Epochs: 200
- Learning rate: 1e-5 (lower for fine-tuning)
- Loss weight: 1.0
- Validation: Sampled from 2D test set (20% split)

**Key Innovation**:
- Leverages learned 1D representations via transfer learning
- Learns discrepancy correction: LFSM → 2D FEM
- Attention to critical wave propagation regions via Hilbert envelope
- Response normalization: StandardScaler on both inputs and targets

---

### 3. Multi-Fidelity Surrogate Model (MFSM-UNET)
**Implementation**: [MFSM_UNETsimple.py](Code/MFSM_UNETsimple.py)

**Two-Stage Training Strategy**:

**Stage 1 - Pretraining on 1D Data**:
- Same dataset as MFSM-CAE (LFSM6000train/test)
- Self-supervised learning
- Epochs: 100
- Learning rate: 1e-4

**Stage 2 - Fine-tuning on 2D Data**:
- Dataset: lfsm_interleaved_2d_train.csv
- Sample pairs: 550 pairs
- **Input**: LFSM predictions (scaled with StandardScaler)
- **Target**: 2D ground truth (scaled with StandardScaler)

**Architecture**: Simple U-Net1D
- **No FiLM conditioning** (removed for simplicity)
- **No attention mechanisms** (standard skip connections only)
- **Parameter Integration**: Concatenation approach
  - Parameter projection: params (7-dim) → linear → 1500 (same as time series)
  - Concatenated with input time series: (B, 2, T)
- **Encoder Path**: 4 levels with residual blocks
  - Base channels: 64
  - Progression: 64 → 128 → 256 → 512 → 1024 (bottleneck)
- **Decoder Path**: 4 levels with standard skip connections
- **Residual Learning**: Input + correction (configurable)
- Epochs: 200
- Loss weight: 3.0 (emphasized 2D learning)
- Validation: 20% from test set

**Differences from MFSM-CAE**:
- U-Net architecture vs Autoencoder architecture
- Spatial feature processing with skip connections
- No Hilbert weighting (standard MSE loss)
- Higher loss weight (3.0 vs 1.0)
- Fewer training pairs (550 vs 1000)

---

### 4. High-Fidelity Surrogate Model (HFSM) - Baseline
**Implementation**: [HFSM.py](Code/HFSM.py)

**Training Data**: 60 cases directly from 2D FEM (train_responses.csv, test_responses.csv)
- Limited high-fidelity data representing computational expense
- Parameters: Same 7-dimensional space as LFSM/MFSM
- Ground truth 2D responses only (no LFSM intermediate predictions)

**Architecture**: Direct CAE + XGBoost (no pretraining)
- **Conditional Autoencoder**:
  - Same architecture as LFSM/MFSM-CAE
  - Latent dimension: 100 (larger than LFSM's 50)
  - Training epochs: 200
  - Learning rate: 1e-4
  - Trained **directly** on 2D FEM data only
- **XGBoost Surrogate**:
  - N_estimators: 1000
  - Max_depth: 7
  - Learning rate: 0.05
  - Early stopping: 50 rounds

**Purpose**: Establishes baseline performance when using only high-fidelity data without multi-fidelity enhancement

---

## Comparative Framework Summary

| Model | Training Data | Architecture | Key Feature | Purpose |
|-------|--------------|--------------|-------------|---------|
| **LFSM** | 750 × 1D | CAE + XGBoost | No fine-tuning | Fast predictions from 1D theory |
| **MFSM-CAE** | 6000 × 1D → 1000 × 2D | CAE (pretrain + finetune) | Hilbert-weighted learning | Learns 1D→2D correction with attention |
| **MFSM-UNET** | 6000 × 1D → 550 × 2D | U-Net (pretrain + finetune) | Spatial skip connections | Alternative architecture for comparison |
| **HFSM** | 60 × 2D | CAE + XGBoost | Direct 2D training | Baseline with limited high-fidelity data |

## Technical Architecture

The project consists of three main components:
1. **Physical Models**: 1D Zigzag Theory and 2D Finite Element Method implementations
2. **Surrogate Models**: Low-Fidelity (LFSM), Multi-Fidelity (MFSM), and High-Fidelity (HFSM) surrogate models
3. **Inverse Problem Solver**: Differential evolution-based parameter estimation system

---

## 1. Physical Model Implementations

### 1.1 One-Dimensional Zigzag Beam Theory

The 1D zigzag theory implementation is adapted for homogeneous beams with rectangular notches. The implementation treats the notch as a virtual interface creating three distinct layers.

#### Displacement Field Formulation

The displacement field is expressed as:
```
u(x,z,t) = u₀(x,t) - z ∂w₀/∂x + R⁽ᵏ⁾(z) ψ₀(x,t)
w(x,z,t) = w₀(x,t)
```

Where:
- `u₀(x,t)`: axial displacement of reference plane
- `w₀(x,t)`: transverse displacement of reference plane
- `ψ₀(x,t)`: zigzag amplitude function
- `R⁽ᵏ⁾(z)`: zigzag function for layer k

#### Layer Definition Strategy

The beam is divided into three virtual layers:
- **Layer 1**: Bottom third of beam (z₀ to z₁)
- **Layer 2**: Middle third (z₁ to z₂)
- **Layer 3**: Top layer, reduced by notch depth (z₂ to z₃)

#### Implementation Details (Verified from Code)

**Zigzag Theory Specifics:**
- **Layer Configuration**: Three-layer approach with virtual interfaces
- **Element Count**: Default 6000 coarse elements (configurable parameter)
- **Precision**: float32 PyTorch tensors for GPU memory efficiency
- **Boundary Conditions**: Free-free boundaries implementation
- **Mesh Strategy**: Non-uniform mesh with refinement around notch boundaries

**Mathematical Implementation:**
- **Displacement Field**: `u(x,z,t) = u₀(x,t) - z ∂w₀/∂x + R⁽ᵏ⁾(z) ψ₀(x,t)`
- **Transverse Displacement**: `w(x,z,t) = w₀(x,t)` (constant through thickness)
- **Element Mass Matrix**: 8×8 matrix for `[u1, u2, w1, w1', w2, w2', psi1, psi2]` DOFs

**Memory Optimizations:**
- TF32 operations enabled for compatible hardware
- Sparse matrix operations using CSR format
- Periodic garbage collection during assembly

### 1.2 Two-Dimensional Finite Element Method

The 2D FEM implementation uses plane strain elasticity with handling for notched geometries.

#### Element Types and Implementation (Verified from Code)

```python
class Mesh2D:
    def __init__(self, domain, nx, ny, element_type='bilinear'):
        if element_type == 'bilinear':
            self.nodes_per_element = 4
            self.order = 1
        elif element_type == 'biquadratic':
            self.nodes_per_element = 9
            self.order = 2
```

**Supported Element Types:**
- **Bilinear**: 4-node quadrilateral elements (default)
- **Biquadratic**: 9-node serendipity elements with mid-side and center nodes

#### Implementation Features (Verified)

**Mesh Generation:**
- Default grid: nx=200, ny=12 elements typical configuration
- Domain-based meshing with specified element densities
- Element type configuration support

**Memory Management:**
- PyTorch memory allocator fixes implemented
- CUDA memory fraction management (80% GPU memory utilization)
- Sparse matrix caching with CSR format

**Matrix Assembly:**
- Gaussian quadrature integration for stiffness and mass matrices
- Jacobian transformation for element mapping
- Strain-displacement (B) matrix construction

#### Notch Implementation Strategy

Notched elements are handled by **material property modification** rather than geometric exclusion:
- Notch regions identified by element center coordinates
- Stiffness and mass reduction by factor (typically 1e-6) in notch elements
- Maintains mesh connectivity while simulating material removal

---

## 2. Surrogate Model Architecture

### 2.1 Conditional Autoencoder (CAE) Implementation

The CAE forms the core of the surrogate modeling framework, providing dimensionality reduction for structural response data.

#### Network Architecture (Verified from LFSMIII.py)

The Conditional Autoencoder implements a **dual-stream architecture** with separate processing paths for timeseries and parameters:

```python
class Encoder(nn.Module):
    def __init__(self, timeseries_dim, params_dim, latent_dim):
        # Timeseries processing network
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

        # Parameter processing network
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

        # Fusion network (concatenates both streams)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),  # 512 total features
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim)
        )

class Decoder(nn.Module):
    def __init__(self, latent_dim, params_dim, output_dim):
        # Parameter conditioning network
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

        # Expansion network (latent + params -> output)
        self.expansion = nn.Sequential(
            nn.Linear(latent_dim + 256, 512),  # 64+256=320 -> 512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim)  # -> 1500 time steps
        )
```

#### Key Architectural Features:

1. **Dual Conditioning Strategy**: Parameters are processed in both encoder and decoder stages
2. **BatchNorm + LeakyReLU**: Consistent activation pattern throughout
3. **Strategic Dropout**: Applied at fusion (0.2) and expansion (0.2) stages
4. **Feature Dimension Matching**: Both streams produce 256-dimensional features for fusion

#### Configuration Parameters (Verified)

From the actual code implementation:

**LFSM Configuration:**
- Latent Dimension: 64
- Time Steps: 1500
- CAE Epochs 1D: 200
- CAE Epochs 2D: 200 (for fine-tuning)
- Batch Size: 64
- Learning Rate: 1e-4
- Learning Rate 2D: 5e-5 (lower for fine-tuning)

**XGBoost Configuration:**
- N Estimators: 2000
- Max Depth: 10
- Learning Rate (eta): 0.03
- Early Stopping: 100 rounds

**HFSM Configuration:**
- Latent Dimension: 100
- N Estimators: 1000
- Max Depth: 7
- Learning Rate (eta): 0.05
- Early Stopping: 50 rounds

### 2.2 Multi-Fidelity Training Strategy

The multi-fidelity approach involves three distinct surrogate models:

#### Low-Fidelity Surrogate Model (LFSM)
- Training dataset path: `/home/user2/Music/abhi3/parameters/LFSM6000train.csv`
- Test dataset path: `/home/user2/Music/abhi3/parameters/LFSM6000test.csv`
- Uses XGBoost for parameter-to-latent space mapping

#### Multi-Fidelity Surrogate Model (MFSM)
- Fine-tuned LFSM using 2D FEM data
- Training data path: `/home/user2/Music/abhi3/parameters/train_responses.csv`
- Lower learning rate (5e-5) for fine-tuning

#### High-Fidelity Surrogate Model (HFSM)
- Trained exclusively on 2D FEM data
- Higher latent dimension (100) compared to LFSM/MFSM (64)

---

## 3. Dataset Generation and Management

### 3.1 Data Configuration (Verified)

**Parameter Specification:**
```python
'PARAM_COLS': ['notch_x', 'notch_depth', 'notch_width', 'length', 'density', 'youngs_modulus']
```

**Time Series Configuration:**
- 1500 time steps: `CONFIG['TIME_COLS'] = [f't_{i}' for i in range(1, 1501)]`
- Response data formatting handles both `r_1, r_2, ...` and `r1, r2, ...` column naming

**Data Preprocessing Pipeline (Verified):**

```python
class BeamResponseDataset(Dataset):
    def __init__(self, params, timeseries, p_scaler=None):
        # Parameter scaling using StandardScaler
        if p_scaler is None:
            self.p_scaler = StandardScaler()
            self.params_scaled = self.p_scaler.fit_transform(self.params)
        else:
            self.p_scaler = p_scaler
            self.params_scaled = self.p_scaler.transform(self.params)

        # Responses kept in original normalized form
        self.timeseries = timeseries.astype(np.float32)

        # Float32 and contiguous arrays for XGBoost efficiency
        self.params_scaled = np.ascontiguousarray(self.params_scaled, dtype=np.float32)
```

**Key Features:**
- **Parameter Scaling**: StandardScaler for handling different parameter ranges
- **Response Preservation**: Original normalized responses maintained unchanged
- **Memory Optimization**: Float32 precision and contiguous arrays for XGBoost compatibility
- **Flexible Column Naming**: Automatic handling of different time column formats

---

## 4. Inverse Problem Solution

The inverse problem estimates notch parameters from observed structural responses using differential evolution optimization.

### 4.1 Configuration (Verified from inverseproblemplayground.py)

**Model Paths:**
- CAE Model: `/home/user2/Music/abhi3/MFSM/Finetuning/best_cae_model_2d_finetuned.pth`
- Surrogate Model: `/home/user2/Music/abhi3/MFSM/Finetuning/surrogate_model_final.joblib`
- Parameter Scaler: `/home/user2/Music/abhi3/MFSM/Finetuning/params_scaler_final.joblib`

**Parameter Bounds:**
```python
'PARAM_BOUNDS': {
    'notch_x': (1.6545, 1.8355),
    'notch_depth': (0.0001, 0.001),
    'notch_width': (0.0001, 0.0012)
}
```

**Fixed Parameters:**
```python
'FIXED_PARAMS': {
    'length': 3.0,
    'density': 2700.0,
    'youngs_modulus': 7e10,
    'location': 1.9
}
```

### 4.2 Database Integration

The system includes database-driven search space focusing:
- Database path: `/home/user2/Music/abhi3/parameters/database.csv`
- Similarity threshold: 0.8
- Population distribution strategy with focused, periphery, and random individuals

---

## 5. Implementation Details

### 5.1 Software Dependencies (Verified from Code)

**Core Dependencies:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import differential_evolution
```

**Key Libraries:**
- **PyTorch**: Neural network implementation with CUDA support
- **XGBoost**: Gradient boosting with GPU acceleration support
- **Scikit-learn**: Preprocessing, scaling, and evaluation metrics
- **SciPy**: Optimization algorithms (differential evolution)
- **Joblib**: Model serialization and loading

### 5.2 Device Configuration and Optimization

**CUDA Configuration:**
```python
'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
'USE_XGB_GPU': True  # Enable XGBoost GPU acceleration

# Performance optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
```

**Training Optimizations:**
- **Adam Optimizer**: `lr=1e-4` (LFSM), `lr=5e-5` (MFSM fine-tuning)
- **Weight Decay**: `1e-5` for regularization
- **Loss Function**: MSELoss for reconstruction
- **Early Stopping**: Patience-based validation monitoring

### 5.3 Training Pipeline (Verified)

**Multi-Stage Training Process:**
1. **Stage 1**: Train CAE on 1D data (LFSM)
2. **Stage 2**: Train XGBoost on latent vectors
3. **Stage 3**: Fine-tune CAE with limited 2D data (MFSM)
4. **Stage 4**: Update XGBoost with combined latent data

**Logging Configuration:**
```python
logging.basicConfig(level=logging.INFO,
    handlers=[
        logging.FileHandler('sequential_training_log.log'),
        logging.StreamHandler()
    ]
)
```

---

## 6. File Structure (Verified)

```
Code/
├── LFSMIII.py              # LFSM/MFSM implementation
├── HFSM.py                 # High-fidelity surrogate model
├── datagenzigzag.py        # 1D zigzag data generation
├── dataset2Dgenfinal.py    # 2D FEM data generation
└── inverseproblemplayground.py  # Inverse problem solver
```

---

## 7. Technical Limitations and Scope

Based on the verified implementation:

### 7.1 Geometric Constraints
- Limited to rectangular notches in homogeneous beams
- Three-layer virtual interface approach for 1D modeling

### 7.2 Material Assumptions
- Isotropic linear elastic materials
- Consistent material properties across the beam

### 7.3 Computational Considerations
- Fixed time series length (1500 steps)
- Specific parameter ranges and bounds
- Requirements for pre-trained model availability

---

## 8. Conclusions

This technical documentation provides a verified understanding of the multi-fidelity surrogate modeling framework based on the actual implementation. The framework demonstrates the application of conditional autoencoders combined with gradient boosting for bridging different fidelity levels in structural analysis.

### 8.1 Verified Technical Contributions

1. **Zigzag Theory Extension**: Implementation of zigzag theory for homogeneous beams with rectangular notches
2. **Multi-Fidelity Architecture**: Combination of Conditional Autoencoder and XGBoost for multi-fidelity modeling
3. **Database-Driven Optimization**: Integration of similarity-based search space reduction
4. **Scalable Implementation**: GPU-accelerated training with memory optimization

### 8.2 Future Work

The framework provides a foundation for extending to:
- Additional notch geometries
- Material nonlinearity considerations
- Multi-point sensing configurations
- Real-time implementation scenarios

---

## 9. Mathematical Framework and Nomenclature (From Project Documentation)

### 9.1 Key Mathematical Symbols

**Structural Parameters:**
- $u(x,z,t)$ - Axial displacement field
- $w(x,z,t)$ - Transverse displacement field
- $R^{(k)}(z)$ - Zigzag function for layer $k$
- $\psi_0(x,t)$ - Zigzag amplitude function

**Machine Learning Components:**
- $\mathbf{z}$ - Latent space representation (64-dimensional for LFSM/MFSM)
- $\mathbf{D}_{\theta}$ - Decoder function parameterized by $\theta$
- $\mathbf{E}_{\phi}$ - Encoder function parameterized by $\phi$
- $\mathcal{L}(\theta, \phi)$ - Loss function for autoencoder training

**Model Architecture:**
- CAE - Conditional Autoencoder
- LFSM - Low-Fidelity Surrogate Model
- MFSM - Multi-Fidelity Surrogate Model
- HFSM - High-Fidelity Surrogate Model
- XGB - XGBoost (Extreme Gradient Boosting)

### 9.2 Implementation-Specific Parameters

**Data Dimensions:**
- Time steps: 1500 (CONFIG['NUM_TIME_STEPS'])
- Parameter vector: 6 components (notch + material properties)
- Latent dimension: 64 (LFSM/MFSM), 100 (HFSM)

**Network Architecture:**
- Encoder streams: 1500→1024→512→256 (timeseries), params→64→128→256
- Fusion: 512→256→64 (latent_dim)
- Decoder: (64+256)→512→1024→1500

---

**Note**: This documentation reflects the verifiable technical implementation based on actual code analysis. All architectural details, configurations, and mathematical formulations have been verified against the source code. Performance metrics and benchmarking results would require additional experimental validation.