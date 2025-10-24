# Detailed Code Description: datagenzigzag.py

## Executive Summary

`datagenzigzag.py` is a comprehensive finite element analysis code that implements a **1D Zigzag Theory** for wave propagation analysis in **homogeneous notched beams**. The code generates Low Fidelity Surrogate Model (LFSM) training data by simulating ultrasonic wave propagation through beams with rectangular notches using a specialized three-layer zigzag displacement field formulation.

---

## 1. Code Overview and Purpose

### 1.1 Primary Objectives

The code performs the following key functions:

1. **Implements 1D Zigzag Theory** for a homogeneous beam represented as three fictitious layers
2. **Simulates wave propagation** in notched beams using free-free boundary conditions
3. **Generates time-domain response data** at multiple sensor locations along the beam
4. **Processes batch datasets** for machine learning training (LFSM development)
5. **Produces normalized displacement responses** for deep learning model training

### 1.2 Non-Standard Application

⚠️ **Important Note**: This code applies zigzag theory (typically used for composite/layered materials) to a **homogeneous beam** by artificially dividing it into three equal layers. This is a non-standard application designed to capture through-thickness effects near the notch.

---

## 2. Theoretical Foundation

### 2.1 Zigzag Theory Implementation

The code implements a refined 1D zigzag theory with the following displacement field:

```
u(x, z, t) = u₀(x, t) - z · w₀,ₓ(x, t) + R^k(z) · ψ₀(x, t)
```

Where:
- `u₀`: Mid-plane axial displacement
- `w₀`: Mid-plane transverse displacement
- `w₀,ₓ`: Rotation (slope of deflection)
- `ψ₀`: Zigzag rotation variable
- `R^k(z)`: Piecewise zigzag function through thickness
- `z`: Through-thickness coordinate

### 2.2 Beam Configuration

**Geometric Parameters:**
- Total beam length: `L = 3.0 m`
- Beam height: `h = 1.5 mm` (0.0015 m)
- Beam width: `b = 1.0 m` (unit width)
- Three fictitious layers of equal thickness (h/3 each)

**Material Properties:**
- Young's modulus: `E ≈ 70 GPa` (Aluminum)
- Density: `ρ ≈ 2700 kg/m³`
- Poisson's ratio: `ν = 0.33`
- Reduced stiffness: `Q₁₁ = E/(1-ν²)`, `Q₅₅ = 0.9·G` (shear correction)

**Notch Parameters (Variable):**
- Location: `notch_x` ∈ [1.65, 1.84] m
- Depth: `notch_depth` ∈ [0.1, 1.0] mm
- Width: `notch_width` ∈ [0.1, 1.2] mm
- Shape: Rectangular, symmetric from top surface

---

## 3. Numerical Implementation

### 3.1 Finite Element Discretization

**Element Type:**
- 2-node beam element with 4 DOFs per node: `[u₀, w₀, w₀,ₓ, ψ₀]`
- Total 8 DOFs per element

**Mesh Strategy:**
- Non-uniform mesh with refinement near notch boundaries
- Base coarse elements: 6000 elements (adjustable)
- Automatic node insertion at notch start/end
- Node merging for elements closer than 10% of coarse element size
- Optional removal of smallest element for stability

**Shape Functions:**
- Linear interpolation for axial displacement `u₀`
- Hermite cubic interpolation for transverse displacement `w₀`
- Linear interpolation for zigzag rotation `ψ₀`

### 3.2 Matrix Formulation

**Element Stiffness Matrix (8×8):**

The element stiffness is constructed from stiffness coefficients:

```
K_e = [K₁₁  K₁₂  K₁₃]
      [K₁₂ᵀ K₂₂  K₂₃]
      [K₁₃ᵀ K₂₃ᵀ K₃₃]
```

Where:
- `K₁₁ = A₁₁·C₅` (2×2, axial stiffness)
- `K₁₂ = -A₁₂·C₆` (2×4, axial-bending coupling)
- `K₁₃ = A₁₃·C₅` (2×2, axial-zigzag coupling)
- `K₂₂ = A₂₂·C₇` (4×4, bending stiffness)
- `K₂₃ = -A₂₃·C₆ᵀ` (4×2, bending-zigzag coupling)
- `K₃₃ = A₃₃·C₅ + A₃₃̄·C₁` (2×2, zigzag stiffness)

**A-Matrices (Stiffness Integrals):**

Computed via Gaussian quadrature (6 points per layer) through beam thickness:

```python
A₁₁ = b·∫∫∫ Q₁₁ dz                    # Extensional stiffness
A₁₂ = b·∫∫∫ Q₁₁·z dz                  # Coupling stiffness
A₁₃ = b·∫∫∫ Q₁₁·R^k(z) dz             # Zigzag coupling
A₂₂ = b·∫∫∫ Q₁₁·z² dz                 # Bending stiffness
A₂₃ = b·∫∫∫ Q₁₁·z·R^k(z) dz           # Bending-zigzag
A₃₃ = b·∫∫∫ Q₁₁·[R^k(z)]² dz          # Zigzag stiffness
A₃₃̄ = b·∫∫∫ Q₅₅·[dR^k/dz]² dz        # Shear zigzag
```

**Element Mass Matrix (8×8):**

Constructed from inertia integrals:

```python
I₀₀ = b·∫∫∫ ρ dz                      # Mass
I₀₁ = b·∫∫∫ ρ·z dz                    # Coupling
I₀₂ = b·∫∫∫ ρ·R^k(z) dz               # Zigzag coupling
I₁₁ = b·∫∫∫ ρ·z² dz                   # Rotary inertia
I₁₂ = b·∫∫∫ ρ·z·R^k(z) dz             # Mixed inertia
I₂₂ = b·∫∫∫ ρ·[R^k(z)]² dz            # Zigzag inertia
```

### 3.3 Zigzag Function Computation

The zigzag function `R^k(z)` is computed following exact theory:

**Step 1: Layer integrals**
```python
C₁^k = Q₅₅·(z_{k+1} - z_k)
C₂^k = 0.5·Q₅₅·(z_{k+1}² - z_k²)
```

**Step 2: Global coefficients**
```python
Δ = 4·z₀²·C₁^L - 8·z₀·C₂^L
R₃ = 4·C₂^L / Δ
R₄ = -4·C₁^L / (3·Δ)
```

**Step 3: Layer coefficients**
```python
a₁^k = 2·(C₁^k/Q₅₅ - z_k)
a₂^k = 3·(2·C₂^k/Q₅₅ - z_k²)
R₂^k = a₁^k·R₃ + a₂^k·R₄
```

**Step 4: Normalization**
- Reference layer: k₀ = 1 (middle layer)
- All coefficients normalized by |R₂^{k₀}|

**Final zigzag function:**
```python
R^k(z) = R̂₁^k + z·R̂₂^k + z²·R̂₃ + z³·R̂₄
```

---

## 4. Wave Propagation Analysis

### 4.1 Excitation Configuration

**Excitation Type:** Pair of forces creating pure moment

**Location:**
- Force 1 position: `x = 1.6535 m`
- Force 2 position: `x = 1.6465 m`
- Separation: 7 mm (creating moment arm)

**Signal Characteristics:**
- Frequency: `f = 100 kHz` (ultrasonic)
- Active duration: `50 μs`
- Total simulation time: `300 μs`
- Window: Hanning-windowed sinusoid
- Amplitude: Normalized

**Force Application:**
```python
F[node1] = +F₀·sin(2πft)·hanning(t)
F[node1+2] = -F₀·sin(2πft)·hanning(t)·(h/2)  # Moment contribution

F[node2] = -F₀·sin(2πft)·hanning(t)
F[node2+2] = +F₀·sin(2πft)·hanning(t)·(h/2)  # Moment contribution
```

### 4.2 Time Integration Scheme

**Method:** Newmark-β implicit time integration

**Parameters:**
- β = 0.3025 (numerical damping)
- γ = 0.6 (numerical damping)
- Stability: Unconditionally stable

**Time Step Calculation:**
```python
c_wave = √(E/ρ)              # Longitudinal wave speed (~5100 m/s)
CFL = 0.7                    # Courant number
dt = CFL·min_dx/c_wave       # Adaptive time step
```

**Algorithm:**

```
For each time step:
  1. Compute effective stiffness: K_eff = K + a₀·M + a₁·C
  2. LU factorization: K_eff = L·U (done once)
  3. Compute effective force: F_eff = F + M·(a₀u + a₂v + a₃a) + C·(a₁u + a₄v + a₅a)
  4. Solve: u_{n+1} = K_eff⁻¹·F_eff
  5. Update: a_{n+1} = a₀(u_{n+1} - u_n) - a₂v_n - a₃a_n
  6. Update: v_{n+1} = v_n + dt[(1-γ)a_n + γa_{n+1}]
```

### 4.3 Response Extraction

**Sensor Locations (11 points):**
```python
response_points = [1.85, 1.87, 1.9, 1.92, 1.95, 1.97,
                   2.0, 2.02, 2.05, 2.07, 2.1] m
```

**Measurement Depth:** `z = 0.75 mm` (top surface)

**Response Calculation:**
For each sensor location, displacement is computed as:

```python
u(x, z, t) = u₀(x,t) - z·w₀,ₓ(x,t) + R^k(z)·ψ₀(x,t)
```

Where `R^k(z)` is evaluated at the measurement depth using the local zigzag function.

**Post-Processing:**
1. Interpolation to uniform time grid (1500 points)
2. Normalization by maximum absolute value per sensor
3. Rounding to 5 decimal places
4. Storage in row format with time series as columns

---

## 5. Computational Optimization

### 5.1 Memory Management

**Sparse Matrix Operations:**
- Uses `scipy.sparse.lil_matrix` for assembly
- Converts to `csr_matrix` for efficient operations
- Reduces memory footprint by ~90%

**GPU Acceleration:**
- Automatic CUDA detection and utilization
- `torch.float32` for memory efficiency
- TF32 enabled for Ampere GPUs
- Batch operations on GPU

**Memory Cleanup:**
- Explicit garbage collection after major operations
- Strategic `torch.cuda.empty_cache()` calls
- Deletion of intermediate tensors

### 5.2 Performance Enhancements

**Reduced Integration:**
- 6-point Gaussian quadrature (reduced from standard 10)
- Balanced accuracy vs. speed

**Save Interval:**
- Saves every 500th time step during integration
- Linear interpolation for full time series reconstruction
- Reduces storage by 99.8%

**Vectorization:**
- NumPy/PyTorch vectorized operations
- Pre-allocated tensors for time integration
- Batched matrix-vector products

**Print Reduction:**
- Progress updates every 10,000 steps (reduced frequency)

---

## 6. Dataset Structure

### 6.1 Input Datasets

**File Locations:**
- Training: `/parameters/lfsm_train_dataset.csv`
- Testing: `/parameters/lfsm_test_dataset.csv`

**Input Dataset Schema:**

| Column | Description | Units | Range/Values |
|--------|-------------|-------|--------------|
| `case_id` | Unique case identifier | - | 0001-0150 |
| `damage_present` | Notch presence flag | - | 'y' or 'n' |
| `notch_x` | Notch center location | m | [1.65, 1.84] or 0.0 |
| `notch_depth` | Notch depth | m | [0.0001, 0.001] or 0.0 |
| `notch_width` | Notch width | m | [0.0001, 0.0012] or 0.0 |
| `length` | Beam length | m | 3.0 (fixed) |
| `density` | Material density | kg/m³ | ~2700 (±5) |
| `youngs_modulus` | Young's modulus | Pa | ~70e9 (±1%) |

**Dataset Characteristics:**

**Training Dataset** (`lfsm_train_dataset.csv`):
- Total cases: 150
- Damaged cases: 143 (95.3%)
- Undamaged cases: 7 (4.7%)
- Purpose: Generate LFSM training data (750 cases for autoencoder)

**Test Dataset** (`lfsm_test_dataset.csv`):
- Total cases: ~30-50 (proportional to training)
- Similar damage/undamaged ratio
- Purpose: Generate LFSM validation data

**Notch Parameter Sampling:**
- **Location (`notch_x`)**: Uniform sampling in [1.65, 1.84] m (right half of beam)
- **Depth (`notch_depth`)**: Levels at [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] mm
- **Width (`notch_width`)**: Levels at [0.1, 0.257, 0.414, 0.571, 0.729, 0.886, 1.043, 1.2] mm
- **Material perturbation**: ±5% variation in E and ρ for some cases

**Undamaged Cases:**
- All notch parameters = 0.0
- Nominal material properties (some with variation)
- Used as baseline/reference signals

### 6.2 Output Datasets

**File Locations (Generated):**
- Training: `LFSM6000train.csv` (in working directory or parameters/)
- Testing: `LFSM6000test.csv`

**Output Dataset Schema:**

| Column Group | Description | Count | Format |
|--------------|-------------|-------|--------|
| `case_id` | Original case identifier | 1 | Integer |
| `response_point` | Sensor x-location | 1 | Float (m) |
| `notch_x` | Notch center | 1 | Float (m) |
| `notch_depth` | Notch depth | 1 | Float (m) |
| `notch_width` | Notch width | 1 | Float (m) |
| `length` | Beam length | 1 | Float (m) |
| `density` | Material density | 1 | Float (kg/m³) |
| `youngs_modulus` | Young's modulus | 1 | Float (Pa) |
| `t_1` to `t_1500` | Time series data | 1500 | Float (normalized) |

**Output Dataset Characteristics:**

**Data Expansion:**
- Input: 1 case → Output: 11 rows (one per sensor)
- Training: 150 cases → 1650 rows (150 × 11)
- Each row contains complete time history at one sensor location

**Time Series Properties:**
- **Length**: 1500 time points
- **Temporal spacing**: Uniform, 300 μs / 1500 = 0.2 μs per point
- **Time column naming**: `t_1`, `t_2`, ..., `t_1500`
- **Physical meaning**: Normalized displacement at sensor location

**Normalization:**
```python
For each sensor location:
  max_val = max(|displacement_series|)
  normalized = displacement_series / max_val
  rounded = round(normalized, 5)
```

**Storage Format:**
- CSV file with header
- Float format: 5 decimal precision
- File size: ~85-100 MB for training set (6000 elements)

**Naming Convention:**
- `LFSM` = Low Fidelity Surrogate Model
- `6000` = Number of coarse elements used
- `train`/`test` = Dataset split

### 6.3 Multi-Resolution Outputs

The code can generate datasets with different mesh refinements:

| File | Elements | Output Size | Computation Time |
|------|----------|-------------|------------------|
| `LFSM1000train.csv` | 1000 | ~102 MB | ~2-3 min/case |
| `LFSM2000train.csv` | 2000 | ~96 MB | ~3-5 min/case |
| `LFSM6000train.csv` | 6000 | ~85 MB | ~5-8 min/case |

**Note:** Higher element counts yield more accurate results but similar file sizes due to fixed 1500-point output.

### 6.4 Additional Dataset Files

**High Fidelity Reference:**
- `hfsm_2d_train_dataset.csv`: 2D FEM training parameters
- `hfsm_2d_test_dataset.csv`: 2D FEM test parameters
- Used for multi-fidelity model comparison

**Training on HF Grid:**
- `LFSM2000trainonHFSM.csv`: 1D zigzag responses for 2D FEM parameter set
- `LFSM2000testonHFSM.csv`: For direct comparison with 2D results

---

## 7. Code Structure and Flow

### 7.1 Main Function Hierarchy

```
main()
  └─ run_batch_wave_propagation_analysis()
      └─ process_dataset()
          ├─ Read CSV input
          └─ For each case:
              ├─ run_wave_propagation_analysis_free_free()
              │   ├─ create_non_uniform_mesh()
              │   ├─ assemble_global_stiffness_matrix()
              │   │   └─ For each element:
              │   │       ├─ get_beam_properties_at_x()
              │   │       ├─ compute_zigzag_functions()
              │   │       ├─ compute_C_matrices()
              │   │       ├─ compute_A_matrices()
              │   │       └─ compute_element_stiffness()
              │   ├─ assemble_global_mass_matrix()
              │   │   └─ For each element:
              │   │       ├─ compute_inertia_integrals()
              │   │       └─ compute_element_mass_matrix()
              │   ├─ memory_efficient_time_integration_with_damping()
              │   └─ get_response_at_specific_point()
              └─ Save to output CSV
```

### 7.2 Key Functions

#### 7.2.1 `get_beam_properties_at_x(x, h, notch_center, notch_width, notch_depth)`

**Purpose:** Returns layer interface coordinates at position x

**Logic:**
```python
if x in notch region:
    z₀ = -h/2
    z₁ = -h/2 + h/3
    z₂ = h/2 - notch_depth  # Adjusted for notch
    z₃ = h/2 - notch_depth  # Top surface reduced
else:
    z₀ = -h/2
    z₁ = -h/2 + h/3
    z₂ = -h/2 + 2h/3
    z₃ = h/2
```

#### 7.2.2 `compute_zigzag_functions(z0, z1, z2, z3, Q55)`

**Purpose:** Computes normalized zigzag function coefficients

**Returns:**
- `R_k(z)`: Function to evaluate zigzag at any z
- `dR_k_dz(z)`: Function to evaluate zigzag derivative

#### 7.2.3 `create_non_uniform_mesh(L, notch_center, notch_width, coarse_elements_on_L, remove_smallest_element)`

**Purpose:** Generates adaptive mesh with notch refinement

**Algorithm:**
1. Create uniform coarse mesh (N elements)
2. Insert nodes at notch boundaries
3. Merge nodes closer than 10% of coarse spacing
4. Optionally remove smallest element
5. Return node coordinates and element lengths

**Output:**
- `x_coords`: Node x-coordinates
- `element_lengths`: Length of each element
- `min_dx`: Minimum element size (for time step)

#### 7.2.4 `memory_efficient_time_integration_with_damping(...)`

**Purpose:** Solves equation of motion: M·ü + C·u̇ + K·u = F(t)

**Features:**
- LU decomposition of effective stiffness (once)
- Pre-allocated temporary vectors
- Selective saving (every Nth step)
- Progress monitoring

**Returns:**
- `u_saved`: Displacement at saved time steps
- `save_indices`: Indices of saved steps

#### 7.2.5 `get_response_at_specific_point(x_coords, u_full, target_x, target_z, params)`

**Purpose:** Extract displacement at specific (x, z) location

**Algorithm:**
1. Find nearest node to target_x
2. Extract DOFs: u₀, w₀,ₓ, ψ₀
3. Get local zigzag function
4. Compute: u = u₀ - z·w₀,ₓ + R^k(z)·ψ₀

**Returns:**
- Displacement time series
- Actual x-coordinate of nearest node

---

## 8. Practical Usage

### 8.1 Running the Code

**Basic execution:**
```bash
cd /home/mecharoy/Thesis/Code
source ../venv/bin/activate
python datagenzigzag.py
```

**Expected behavior:**
1. Reads input CSVs from hardcoded paths
2. Processes each case sequentially
3. Prints progress for each case
4. Generates output CSVs: `LFSM6000train.csv` and `LFSM6000test.csv`
5. Displays sample plots (optional, last function)

### 8.2 Customization

**Adjusting mesh refinement:**
```python
# Line 888 in run_batch_wave_propagation_analysis()
coarse_elements_on_L=6000  # Change to 1000, 2000, etc.
```

**Modifying output time points:**
```python
# Line 848
TARGET_NUM_POINTS = 1500  # Change to desired number
```

**Changing sensor locations:**
```python
# Line 847
response_points = [1.85, 1.87, ...]  # Modify list
```

**Input file paths:**
```python
# Lines 844-845
train_file = "your/path/to/train.csv"
test_file = "your/path/to/test.csv"
```

### 8.3 Computational Requirements

**Estimated Resource Needs (per case, 6000 elements):**

| Resource | Requirement | Notes |
|----------|-------------|-------|
| RAM | 8-16 GB | Depends on mesh size |
| GPU VRAM | 4-8 GB | Optional, for acceleration |
| CPU cores | 1-8 | Single-threaded FEM |
| Time/case | 5-8 min | GPU: 3-5 min |
| Disk space | ~100 MB | Per output CSV |

**Total Dataset Generation:**
- Training (150 cases): ~12-20 hours
- Testing (30 cases): ~2-4 hours

**Optimization recommendations:**
1. Use GPU if available (5× speedup)
2. Reduce `coarse_elements_on_L` for prototyping
3. Decrease `TARGET_NUM_POINTS` if time resolution not critical
4. Run in background with `nohup` or `screen`

### 8.4 Output Verification

**Quality checks:**

1. **File size consistency:**
   ```bash
   ls -lh LFSM6000train.csv
   # Should be ~80-100 MB for 150 cases × 11 sensors
   ```

2. **Row count verification:**
   ```bash
   wc -l LFSM6000train.csv
   # Should be 1651 (1650 data rows + 1 header)
   ```

3. **Data integrity:**
   ```python
   df = pd.read_csv('LFSM6000train.csv')
   print(df.shape)  # (1650, 1508) expected
   print(df['case_id'].nunique())  # 150 unique cases
   print(df['response_point'].nunique())  # 11 unique sensors
   ```

4. **Normalization check:**
   ```python
   time_cols = [f't_{i}' for i in range(1, 1501)]
   for col in time_cols:
       assert df[col].abs().max() <= 1.0  # All normalized
   ```

---

## 9. Integration with Research Pipeline

### 9.1 Role in Multi-Fidelity Approach

This code is the **LFSM data generator** in the multi-fidelity framework:

```
┌─────────────────────────────────────────────────┐
│    Input Parameter Space                        │
│    (notch_x, notch_depth, notch_width)          │
└───────────────┬─────────────────────────────────┘
                │
                ├──────────────────┬──────────────────┐
                ▼                  ▼                  ▼
         ┌──────────────┐   ┌──────────────┐  ┌──────────────┐
         │  1D Zigzag   │   │  2D FEM      │  │  Experiment  │
         │  (This Code) │   │  (HFSM)      │  │  (Real Data) │
         │  750 cases   │   │  50 cases    │  │  Limited     │
         └──────┬───────┘   └──────┬───────┘  └──────┬───────┘
                │                  │                  │
                ├──────────────────┴──────────────────┘
                ▼
         ┌──────────────────────────────┐
         │  Autoencoder (LFSM)          │
         │  Encoder-Latent-Decoder      │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │  XGBoost Latent Predictor    │
         │  Parameters → Latent Space   │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │  Fine-tuned MFSM             │
         │  (Corrected with 2D data)    │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │  Inverse Problem Solver      │
         │  (Differential Evolution)    │
         └──────────────────────────────┘
```

### 9.2 Data Flow

**Forward path (Training):**
```
lfsm_train_dataset.csv (150 cases)
    ↓ [datagenzigzag.py]
LFSM6000train.csv (1650 rows)
    ↓ [Split by sensor]
750 training pairs: (params, response) × 11 sensors
    ↓ [Autoencoder training]
Latent space representation (e.g., 32-dim)
    ↓ [XGBoost training]
Parameter → Latent mapping
    ↓ [2D FEM fine-tuning]
Multi-Fidelity Surrogate Model (MFSM)
```

**Inverse path (Inference):**
```
Measured response (11 sensors × 1500 points)
    ↓ [Differential Evolution]
Initial population (30 candidates)
    ↓ [MFSM forward solver]
Objective: min ||response_pred - response_meas||²
    ↓ [Iterative refinement]
Identified parameters: (notch_x, notch_depth, notch_width)
```

### 9.3 Expected Dataset Statistics

**Training Set Output:**
- Rows: 1650 (150 cases × 11 sensors)
- Columns: 8 parameters + 1500 time points = 1508 total
- File size: ~85 MB
- Damaged cases: 1573 rows (143 cases × 11)
- Undamaged cases: 77 rows (7 cases × 11)

**Test Set Output:**
- Rows: ~330-550 (30-50 cases × 11 sensors)
- Similar structure to training set

**Multi-Sensor Data:**
Each case produces 11 response curves, capturing:
- Wave reflection from notch
- Mode conversion at notch edges
- Through-thickness effects via zigzag theory
- Spatial variation of wave field

---

## 10. Known Limitations and Assumptions

### 10.1 Physical Assumptions

1. **Linear elasticity:** Small deformations, Hooke's law valid
2. **Isotropic homogeneous material:** No actual layering (artificial division)
3. **No material damping:** Free-free boundaries, zero C matrix
4. **Plane stress:** Width >> other dimensions
5. **Perfect excitation:** Idealized moment loading
6. **Rectangular notch:** Sharp corners (no rounding)

### 10.2 Numerical Limitations

1. **Element type:** C0 continuity for zigzag variable (may need C1)
2. **Reduced integration:** 6 Gauss points (affects accuracy near notch)
3. **Time step:** CFL-limited (very small for fine mesh)
4. **Normalization artifacts:** Different cases have different scaling
5. **Mesh adaptation:** Simple refinement strategy

### 10.3 Computational Constraints

1. **Memory:** Large cases (>10000 elements) may exceed RAM
2. **Time:** Serial processing of cases (no parallelization)
3. **Storage:** Output files can be large (>1 GB total)
4. **GPU dependency:** CUDA required for acceleration

### 10.4 Zigzag Theory Applicability

⚠️ **Critical Note:**
Zigzag theory is designed for **composite laminates** with:
- Distinct material layers
- Stiffness discontinuities at interfaces
- Interlaminar shear effects

Applying it to a **homogeneous beam** is unconventional:
- Artificial layer division (three equal layers)
- No physical interfaces
- Justification: Captures through-thickness gradients near notch
- Validation required against 2D FEM/experiments

---

## 11. Future Enhancement Opportunities

### 11.1 Code Improvements

1. **Parallel processing:** Multi-process case execution
2. **Adaptive meshing:** Error-driven refinement
3. **Higher-order elements:** Quadratic/cubic shape functions
4. **Contact modeling:** Notch face closure under compression
5. **Material damping:** Rayleigh damping (α·M + β·K)

### 11.2 Physical Extensions

1. **Variable cross-section:** Non-constant h along length
2. **Multiple notches:** Array of defects
3. **Different notch shapes:** V-notch, U-notch, crack
4. **Thermal effects:** Thermo-elastic coupling
5. **Nonlinear material:** Plasticity at notch tip

### 11.3 Analysis Features

1. **Modal analysis:** Natural frequencies and mode shapes
2. **Frequency domain:** FFT of responses, dispersion curves
3. **Damage indices:** Statistical features for ML
4. **Uncertainty quantification:** Monte Carlo on parameters
5. **Sensitivity analysis:** Parameter influence study

---

## 12. Troubleshooting Guide

### 12.1 Common Errors

**Error: "File not found"**
- **Cause:** Hardcoded paths in lines 844-845
- **Fix:** Update `train_file` and `test_file` paths

**Error: "CUDA out of memory"**
- **Cause:** Insufficient GPU VRAM
- **Fix 1:** Reduce `coarse_elements_on_L`
- **Fix 2:** Use CPU (slower): `device = 'cpu'`
- **Fix 3:** Clear GPU: `torch.cuda.empty_cache()`

**Error: "Mesh generation resulted in zero-length element"**
- **Cause:** Notch boundaries too close to mesh nodes
- **Fix:** Adjust `merge_tolerance` in `create_non_uniform_mesh()`

**Warning: "LU decomposition failed"**
- **Cause:** Singular stiffness matrix (free-free motion)
- **Impact:** Uses direct solve (slower but stable)
- **Fix:** Normal behavior, no action needed

### 12.2 Performance Issues

**Slow execution (>15 min/case):**
- Check GPU utilization: `nvidia-smi`
- Verify TF32 enabled: `torch.backends.cuda.matmul.allow_tf32`
- Reduce `coarse_elements_on_L` for testing

**Large output files (>200 MB):**
- Reduce `TARGET_NUM_POINTS` if acceptable
- Check for duplicate entries in output CSV
- Verify rounding to 5 decimals (line 915)

**Memory leaks:**
- Ensure `gc.collect()` after each case
- Clear GPU cache between cases
- Monitor with `htop` or `nvidia-smi`

### 12.3 Validation Checks

**Sanity tests:**

1. **Undamaged case:** Response should be symmetric
2. **Deep notch:** Significant reflection/mode conversion
3. **Wide notch:** Lower frequency content
4. **Near-notch sensors:** Largest amplitudes

**Comparison with 2D FEM:**
- Expect 5-15% difference in amplitudes
- Time-of-flight should match within 1%
- Mode shapes qualitatively similar

---

## 13. Mathematical Formulation Summary

### 13.1 Governing Equations

**Strong form (3D elasticity):**
```
∇·σ + ρf = ρü        (Equation of motion)
σ = C:ε              (Constitutive law)
ε = ∇ˢu              (Strain-displacement)
```

**Weak form (Principle of Virtual Work):**
```
∫∫∫ ρü·δu dV + ∫∫∫ σ:δε dV = ∫∫ t·δu dS + ∫∫∫ ρf·δu dV
```

**Discretized (FEM):**
```
M·ü + C·u̇ + K·u = F(t)
```

Where:
- M: Global mass matrix (N×N sparse)
- C: Global damping matrix (N×N, zero for free-free)
- K: Global stiffness matrix (N×N sparse)
- u: Displacement vector (N×1)
- F: Force vector (N×1)
- N: Total DOFs = 4 × (num_nodes)

### 13.2 Element Formulation

**Displacement interpolation:**
```
u₀(x) = N_u(ξ) · u₀^e
w₀(x) = N_w(ξ) · w₀^e
ψ₀(x) = N_ψ(ξ) · ψ₀^e
```

Shape functions:
- `N_u`: Linear [1-ξ, ξ]
- `N_w`: Hermite cubic
- `N_ψ`: Linear [1-ξ, ξ]

**Stiffness matrix:**
```
K_e = ∫∫∫ B^T · D · B dV
```

**Mass matrix:**
```
M_e = ∫∫∫ ρ · N^T · N dV
```

### 13.3 Zigzag Enrichment

**Through-thickness displacement:**
```
u(x,z,t) = u₀(x,t) - z·w₀,ₓ(x,t) + R^k(z)·ψ₀(x,t)
          └─────┘   └───────────┘   └─────────────┘
          Axial     Bending         Zigzag
```

**Axial strain:**
```
εₓₓ = ∂u/∂x = u₀,ₓ - z·w₀,ₓₓ + R^k(z)·ψ₀,ₓ
```

**Shear strain:**
```
γₓz = ∂u/∂z + ∂w/∂x = [dR^k/dz - w₀,ₓ]·ψ₀ + w₀,ₓ
```

---

## 14. References and Theoretical Background

### 14.1 Zigzag Theory Foundation

The zigzag theory implementation follows:

1. **Tessler, A., Di Sciuva, M., & Gherlone, M. (2009)**
   "Refined zigzag theory for homogeneous, laminated composite, and sandwich beams"
   NASA Technical Publication TP-2009-215561

2. **Di Sciuva, M. (1984)**
   "A refined transverse shear deformation theory for multilayered anisotropic plates"
   Atti Accademia delle Scienze di Torino

3. **Tessler, A. (2015)**
   "Refined zigzag theory for homogeneous, laminated composite, and sandwich plates: A homogeneous limit methodology for zigzag function selection"
   NASA Technical Memorandum TM-2015-218853

### 14.2 FEM and Wave Propagation

1. **Bathe, K.J. (1996)**
   "Finite Element Procedures"
   Prentice Hall

2. **Newmark, N.M. (1959)**
   "A method of computation for structural dynamics"
   Journal of Engineering Mechanics Division, ASCE

3. **Hughes, T.J.R. (2000)**
   "The Finite Element Method: Linear Static and Dynamic Finite Element Analysis"
   Dover Publications

### 14.3 Structural Health Monitoring

1. **Giurgiutiu, V. (2008)**
   "Structural Health Monitoring with Piezoelectric Wafer Active Sensors"
   Academic Press

2. **Rose, J.L. (2014)**
   "Ultrasonic Guided Waves in Solid Media"
   Cambridge University Press

---

## 15. Code Quality and Best Practices

### 15.1 Strengths

✅ **Well-documented functions:** Clear docstrings
✅ **Modular design:** Reusable functions
✅ **Error handling:** Try-except blocks for batch processing
✅ **Memory efficiency:** Sparse matrices, GPU acceleration
✅ **Progress monitoring:** Real-time feedback
✅ **Type consistency:** Uses `numpy.float32` throughout

### 15.2 Areas for Improvement

⚠️ **Hardcoded paths:** Input file paths not configurable
⚠️ **Magic numbers:** Some constants not well-explained
⚠️ **Limited input validation:** Assumes correct CSV format
⚠️ **No unit tests:** Functions not individually tested
⚠️ **Minimal comments:** Complex sections lack explanation
⚠️ **No logging:** Uses print instead of logging module

### 15.3 Recommended Practices

For future development:
1. Use `argparse` for command-line configuration
2. Implement `logging` instead of `print`
3. Add input validation with informative errors
4. Create unit tests for critical functions
5. Use configuration files (YAML/JSON) for parameters
6. Add type hints for better code clarity
7. Implement checkpointing for long runs

---

## 16. Conclusion

`datagenzigzag.py` is a sophisticated scientific computing code that bridges structural mechanics, finite element analysis, and machine learning data generation. Its primary contribution is generating **low-fidelity surrogate data** for training autoencoders in a multi-fidelity framework for structural health monitoring.

**Key Achievements:**
- ✅ Implements 1D zigzag theory with proper mathematical formulation
- ✅ Handles notched beams with adaptive meshing
- ✅ Generates clean, normalized datasets for deep learning
- ✅ Optimized for GPU acceleration and memory efficiency
- ✅ Produces reproducible, well-structured outputs

**Primary Use Case:**
Training a non-conditional autoencoder (LFSM) that learns to compress wave propagation responses, with XGBoost handling the parameter-to-latent mapping. This is later enhanced with 2D FEM data to create a multi-fidelity surrogate model (MFSM) for inverse parameter identification.

---

**Document Version:** 1.0
**Last Updated:** January 2025
**Code Version:** datagenzigzag.py (as of latest commit)
**Author:** Generated for thesis documentation
