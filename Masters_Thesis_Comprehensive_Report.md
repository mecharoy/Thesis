# Masters Thesis Comprehensive Report
## Surrogate Modeling and Inverse Problem Solving for Damage Detection in Composite Beams

**Date:** October 26, 2025
**Author:** Abhijit
**Field:** Computational Mechanics / Structural Health Monitoring

---

## Executive Summary

This Masters thesis develops a sophisticated multi-fidelity machine learning framework for detecting and characterizing damage (notches) in homogeneous beams using wave propagation analysis. The work combines finite element modeling (FEM), deep learning (Autoencoders), gradient boosting (XGBoost), and optimization techniques to solve both forward and inverse problems in structural health monitoring.

**Key Achievements:**
- Developed High-Fidelity Surrogate Model (HFSM) using 2D FEM data
- Created Multi-Fidelity Surrogate Model (MFSM) by fine-tuning with low-fidelity 1D data
- Implemented inverse problem solver to predict notch location and severity from response data
- Achieved high prediction accuracy (R² > 0.9) on test datasets
- Integrated database-driven optimization for real-time damage detection

---

## 1. Problem Statement

### 1.1 Background
Composite laminated beams are widely used in aerospace, automotive, and civil engineering applications. Damage detection through non-destructive testing is critical for safety and maintenance. Traditional analytical methods struggle with:
- Complex wave propagation in multi-layered materials
- Computational cost of high-fidelity simulations
- Real-time damage characterization requirements

This thesis adapts the zigzag beam theory—a refined technique originally developed for composite laminates—to homogeneous beams by dividing them into fictitious layers. This approach enables accurate capture of shear deformation effects while maintaining computational efficiency compared to full 2D/3D simulations.

### 1.2 Research Objectives
1. **Forward Problem:** Develop fast, accurate surrogate models to predict wave propagation responses given beam parameters and damage characteristics
2. **Inverse Problem:** Predict damage location and severity from measured wave responses
3. **Multi-Fidelity Integration:** Combine low-cost 1D models with high-fidelity 2D models for optimal performance

### 1.3 Damage Characterization
The thesis focuses on rectangular notch damage defined by three parameters:
- **notch_x:** Location along beam length (1.6545m - 1.8355m)
- **notch_depth:** Vertical depth of damage (0.0001m - 0.001m)
- **notch_width:** Horizontal width of damage (0.0001m - 0.0012m)


---

## 2. Methodology

### 2.1 Overall Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA GENERATION LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  1D Zigzag Theory (LFSM)  →  1000 elements, fast generation │
│  2D FEM Analysis (HFSM)    →  10000×10 mesh, high accuracy   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   SURROGATE MODEL LAYER                      │
├─────────────────────────────────────────────────────────────┤
│  Autoencoder (AE):                             │
│    • Encoder: Time series → Latent space (30D)              │
│    • Decoder: Latent → Reconstructed response  │
│  XGBoost Surrogate:                                         │
│    • Maps Parameters → Latent vectors                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 INVERSE PROBLEM SOLVER                       │
├─────────────────────────────────────────────────────────────┤
│  Optimization (Differential Evolution):                      │
│    • Input: Measured response (1500 time steps)             │
│    • Output: Predicted notch parameters                     │
│  Database Integration:                                       │
│    • Similar case retrieval via correlation                 │
│    • Focused search bounds for faster convergence           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Physics-Based Data Generation

#### 2.2.1 One-Dimensional Zigzag Theory Model (LFSM)
**File:** `datagenzigzag.py`

**Key Features:**
- Implementation of refined zigzag beam theory (originally developed for composite laminates) applied to a homogeneous beam with three fictitious layers
- 3-layer homogeneous beam model with layer-wise transverse shear deformation
- Wave propagation using central difference time integration
- Parametric notch modeling with property degradation

**Governing Equations:**
- Displacement field: `u(x,z,t) = u₀(x,t) - z·w₀,ₓ(x,t) + R^k(z)·ψ₀(x,t)`
- Zigzag function R^k(z) accounts for discontinuous shear strains
- Free-free boundary conditions (no PML damping)

**Computational Efficiency:**
- 1000 elements along beam length
- Non-uniform mesh refinement around notch
- Adaptive time stepping with CFL condition: `dt = 0.7 × Δx_min / c_wave`
- GPU acceleration with PyTorch tensors
- Memory-efficient sparse matrix storage

**Data Generated:**
- Training: 750 cases × 11 sensor points = 8250 responses
- Test: 100 cases
- Time series: 1500 points per response (300μs duration)
- Normalized displacement responses (-1 to 1)

#### 2.2.2 Two-Dimensional FEM Model (HFSM)
**File:** `dataset2Dgenfinal.py`

**Key Features:**
- Full 2D plane stress finite element analysis
- Biquadratic elements (9 nodes) for improved accuracy
- Gauss quadrature integration (reduced for efficiency)
- Material properties: Isotropic with Young's modulus E, Poisson's ratio ν=0.33

**Mesh Characteristics:**
- 10000 × 10 elements (x × y directions)
- Refined mesh around notch region
- Total DOFs: ~900,000 (2 DOF per node)
- Free-free boundary conditions

**Notch Modeling:**
- Geometric removal: Elements in notch region have reduced stiffness (10⁻⁶ factor)
- Mass matrix modification for consistency
- Avoids complete element removal to prevent mesh discontinuities

**Time Integration:**
- Central difference method with lumped mass matrix
- Stable time step: `dt = 0.8 × 2/ω_max`
- Memory-efficient storage (only response DOFs saved)
- 150,000 time steps total

**Computational Optimizations:**
- Matrix caching system (saves K, M for reuse)
- Hash-based cache validation
- Sparse matrix format (CSR)
- PyTorch GPU acceleration for time integration

### 2.3 Surrogate Model Architecture

#### 2.3.1  Autoencoder (AE)
**Files:** `HFSM.py`, `LFSMIII.py`

**Architecture Design:**

**Encoder Network:**
```
Time Series Path (1500 → 256):
  Linear(1500, 1024) → BatchNorm → LeakyReLU → Dropout(0.1)
  Linear(1024, 512)  → BatchNorm → LeakyReLU → Dropout(0.1)
  Linear(512, 256)   → BatchNorm → LeakyReLU

Parameter Path (7 → 256):
  Linear(7, 64)   → BatchNorm → LeakyReLU → Dropout(0.1)
  Linear(64, 128) → BatchNorm → LeakyReLU → Dropout(0.1)
  Linear(128, 256) → BatchNorm → LeakyReLU

Fusion Network (512 → 64):
  Concatenate [Time Features, Param Features]
  Linear(512, 512) → BatchNorm → LeakyReLU → Dropout(0.2)
  Linear(512, 256) → BatchNorm → LeakyReLU
  Linear(256, 30)  # Latent space
```

**Decoder Network:**
```


Expansion Network (30 → 1500):
  
  Linear(30, 512)  → BatchNorm → LeakyReLU → Dropout(0.2)
  Linear(512, 1024) → BatchNorm → LeakyReLU → Dropout(0.2)
  Linear(1024, 1500) # Reconstructed time series
```

**Training Strategy:**
- Loss function: MSE between input and reconstructed time series
- Optimizer: Adam with learning rate 1e-4, weight decay 1e-5
- Validation-based early stopping
- Best model selection based on validation loss

**Key Design Choices:**
1. **Architecture:** Parameters influence both encoding and decoding
2. **Latent dimension 30:** Balances compression and reconstruction quality
3. **BatchNorm + Dropout:** Prevents overfitting, improves generalization
4. **LeakyReLU:** Handles negative gradients better than ReLU

#### 2.3.2 XGBoost Surrogate Model
**Purpose:** Map beam parameters directly to latent vectors

**Configuration:**
```python
{
    'objective': 'reg:squarederror',
    'n_estimators': 2000,  # Increased for better latent prediction
    'max_depth': 10,
    'eta': 0.03,           # Learning rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist',  # GPU acceleration
    'early_stopping_rounds': 100
}
```

**Training Process:**
1. Extract latent vectors from trained encoder: `Z = Encoder(Y, P)`
2. Train XGBoost: `Z_pred = XGBoost(P_scaled)`
3. Validate on held-out test set
4. Evaluate R² in latent space

**Why XGBoost?**
- Excellent for tabular data (parameter vectors)
- Non-linear relationships between parameters and latent space
- Fast inference for inverse problem optimization
- GPU support for large datasets

#### 2.3.3 Multi-Fidelity Training Strategy
**File:** `LFSMIII.py`

**Phase 1: Low-Fidelity Pre-training**
```
Step 1: Train CAE on 1D zigzag data
  - 750 training cases
  - 200 epochs
  - Learning rate: 1e-4

Step 2: Train XGBoost on 1D latent vectors
  - Parameters → Latent mapping
  - Validation R²: ~0.95
```

**Phase 2: High-Fidelity Fine-tuning**
```
Step 3: Load 2D FEM data
  - Higher quality, more expensive data
  - Smaller dataset (~100 cases)

Step 4: Create weighted combined dataset
  - 1D data × 1 (base representation)
  - 2D data × 3 (higher weight for quality)
  - Total: 750 + 3×100 = 9,300 effective samples

Step 5: Fine-tune CAE on combined data
  - Initialize from 1D weights
  - Lower learning rate: 1e-5
  - 100 epochs
  - Preserves 1D knowledge while adapting to 2D

Step 6: Retrain XGBoost on combined latent vectors
  - New parameter-to-latent mapping
  - Captures both data sources
```

**Benefits of Multi-Fidelity Approach:**
1. **Data efficiency:** Leverages abundant low-cost 1D data
2. **Performance:** Achieves high-fidelity accuracy with limited 2D data
3. **Robustness:** Model learns from multiple physical representations
4. **Cost reduction:** Fewer expensive 2D simulations required

### 2.4 Inverse Problem Formulation

#### 2.4.1 Problem Definition
**File:** `inverseproblemplayground.py`

**Given:** Measured wave response Y_measured (1500 time points)

**Find:** Optimal damage parameters θ = [notch_x, notch_depth, notch_width]

**Such that:** `Y_surrogate(θ) ≈ Y_measured`

**Mathematical Formulation:**
```
minimize: L(θ) = Loss(Y_surrogate(θ), Y_measured)
subject to: θ_min ≤ θ ≤ θ_max
```

#### 2.4.2 Loss Function Design

**Composite Loss (Primary):**
```python
L_composite = w₁·L_MSE + w₂·L_correlation + w₃·L_FFT

where:
  L_MSE = MSE(Y_pred, Y_true)
  L_correlation = 1 - Pearson_correlation(Y_pred, Y_true)
  L_FFT = MSE(FFT(Y_pred), FFT(Y_true))

  Weights: w₁=0.4, w₂=0.3, w₃=0.3
```


**Robust Loss (Huber):**
```python
# Handle model-reality mismatch
δ = 0.1 × std(Y_true)

L_huber(r) = {
    0.5 × r²           if |r| ≤ δ
    δ × (|r| - 0.5δ)   otherwise
}
```

#### 2.4.3 Optimization Algorithm

**Differential Evolution (Primary Method):**
```python
{
    'strategy': 'best1bin',
    'maxiter': 100,
    'popsize': 50,
    'atol': 1e-6,
    'mutation': (0.5, 1.0),
    'recombination': 0.7,
    'workers': 1  # Sequential for stability
}
```

**Search Space:**
- notch_x ∈ [1.6545, 1.8355] meters
- notch_depth ∈ [0.0001, 0.001] meters (log-space)
- notch_width ∈ [0.0001, 0.0012] meters (log-space)

**Log-Space Optimization:**
For depth and width parameters, optimize in log₁₀ space:
```
θ_log = log₁₀(θ)
θ = 10^(θ_log)
```
Benefits: Better exploration of small values, uniform sampling in logarithmic scale

#### 2.4.4 Database-Driven Search Enhancement

**Similar Case Retrieval:**
```python
1. Compute correlation with all database responses
   corr_i = Pearson(Y_measured, Y_database[i])

2. Select top-K similar cases (K=10)
   similar_cases = argsort(corr)[-K:]
   threshold = 0.8

3. Extract parameter ranges from similar cases
   bounds_focused = {
       'notch_x': [min(notch_x_similar), max(notch_x_similar)],
       'notch_depth': [min(depth_similar), max(depth_similar)],
       'notch_width': [min(width_similar), max(width_similar)]
   }

4. Generate focused initial population
   - 50% within focused bounds
   - 33% in periphery region
   - 17% random exploration
```

**Benefits:**
- Faster convergence (fewer iterations)
- Higher success rate in complex search spaces
- Leverages historical simulation data
- Adaptive to measurement quality

---

