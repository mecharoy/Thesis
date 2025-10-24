# Critical Questions for Data Collection - Forward MFSM Report

## Purpose
This document provides a structured questionnaire to guide data collection and analysis. Answer these questions by running your trained models and analyzing the results. Organized by priority level.

---

## PRIORITY 1: CRITICAL (Must have to write Results section)

### Q1.1: Test Set Performance Metrics
**Question**: What are the test set performance metrics for each model?

**How to obtain**:
```python
# After loading trained model and test data
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# For each model (LFSM, MFSM-CAE, MFSM-UNET, HFSM):
y_pred = model.predict(X_test)  # Shape: (n_test, 11, T)
y_true = y_test  # Ground truth 2D responses

# Global metrics
r2 = r2_score(y_true.flatten(), y_pred.flatten())
mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
nmse = mse / np.var(y_true.flatten())

print(f"Model: {model_name}")
print(f"  Test R²: {r2:.4f}")
print(f"  Test MSE: {mse:.6e}")
print(f"  Test NMSE: {nmse:.4f}")
```

**Fill in this table**:
```
Model         | Test R²  | Test MSE | Test NMSE | Train R² | Train MSE
--------------|----------|----------|-----------|----------|----------
LFSMon2Ddata  |  0.8662  |  0.0069  |   12.98   |   0.98   |     -
MFSM-CAE      |  0.9341  |  0.0036  |   06.58   |    -     |     -
MFCAEw/oparam |  0.9249  |  0.0039  |   07.02   |    -     |     -
MFSM-UNET     |  0.9280  |  0.0039  |   07.19   |    -     |     -
HFSM          |  0.8102  |  0.0101  |   19.16   |    -     |     -
```

---

### Q1.2: Test Set Specifications
**Question**: What are the exact specifications of your test set?

**Answer format**:
```
Total test cases: [NUMBER]
Damaged cases: [NUMBER]
Pristine cases: [NUMBER]
Test set split ratio: [e.g., 20% of total data]
Sampling method: [Random / Stratified / Specific selection]
```

**Provide test set case IDs**:
```
Test case IDs: [List, e.g., 0501-0600 or specific IDs]
```

---

### Q1.3: Best and Worst Performing Cases
**Question**: Which test cases show best and worst performance?

**How to obtain**:
```python
# For each test case
case_errors = []
for i in range(n_test):
    case_mse = mean_squared_error(y_true[i].flatten(), y_pred[i].flatten())
    case_errors.append((i, case_mse))

# Sort by error
case_errors.sort(key=lambda x: x[1])
best_case_idx = case_errors[0][0]
worst_case_idx = case_errors[-1][0]

print(f"Best case: {test_case_ids[best_case_idx]}, MSE: {case_errors[0][1]:.6e}")
print(f"Worst case: {test_case_ids[worst_case_idx]}, MSE: {case_errors[-1][1]:.6e}")

# Get damage parameters for these cases
print(f"Best case parameters: {test_params[best_case_idx]}")
print(f"Worst case parameters: {test_params[worst_case_idx]}")
```

**Fill in**:
```
Best Performing Case:
  Case ID: [ID]
  Damage present: [y/n]
  If yes:
    x_notch: [VALUE] m
    d_notch: [VALUE] m
    w_notch: [VALUE] m
  MSE (LFSM): [VALUE]
  MSE (MFSM-CAE): [VALUE]
  MSE (MFSM-UNET): [VALUE]
  MSE (HFSM): [VALUE]

Worst Performing Case:
  Case ID: [ID]
  Damage present: [y/n]
  If yes:
    x_notch: [VALUE] m
    d_notch: [VALUE] m
    w_notch: [VALUE] m
  MSE (LFSM): [VALUE]
  MSE (MFSM-CAE): [VALUE]
  MSE (MFSM-UNET): [VALUE]
  MSE (HFSM): [VALUE]
```

---

### Q1.4: Visualization File Paths
**Question**: Where are the saved visualization files?

**Provide paths**:
```
Training curves:
  LFSM: [PATH to loss vs epoch plot]
  MFSM-CAE Stage 1: [PATH]
  MFSM-CAE Stage 2: [PATH]
  MFSM-UNET Stage 1: [PATH]
  MFSM-UNET Stage 2: [PATH]
  HFSM: [PATH]

Prediction plots:
  Best case (all models): [PATH]
  Worst case (all models): [PATH]
  Scatter plots (pred vs actual): [PATH]
  Error distributions: [PATH]

Model checkpoints:
  LFSM: [PATH to .pth or .pkl file]
  MFSM-CAE: [PATH]
  MFSM-UNET: [PATH]
  HFSM: [PATH]
```

---

### Q1.5: Time Series Length
**Question**: How many time steps in your response data?

**Answer**:
```
Number of time steps T: [VALUE]
Time step size Δt: [VALUE] seconds
Total duration: [VALUE] seconds
Sampling frequency: [VALUE] Hz
```

---

## PRIORITY 2: HIGH (Needed for Discussion section)

### Q2.1: Per-Sensor Performance
**Question**: How does performance vary across the 11 sensors?

**How to obtain**:
```python
# For each sensor
for sensor_idx in range(11):
    y_true_sensor = y_true[:, sensor_idx, :]  # Shape: (n_test, T)
    y_pred_sensor = y_pred[:, sensor_idx, :]

    r2_sensor = r2_score(y_true_sensor.flatten(), y_pred_sensor.flatten())
    mse_sensor = mean_squared_error(y_true_sensor.flatten(), y_pred_sensor.flatten())

    print(f"Sensor {sensor_idx+1}: R² = {r2_sensor:.4f}, MSE = {mse_sensor:.6e}")
```

**Fill in table**:
```
Sensor | Position (m) | LFSM R² | MFSM-CAE R² | MFSM-UNET R² | HFSM R² | Observations
-------|-------------|---------|-------------|--------------|---------|-------------
1      |             |         |             |              |         |
2      |             |         |             |              |         |
...    |             |         |             |              |         |
11     |             |         |             |              |         |
```

---

### Q2.2: Error Distribution Characteristics
**Question**: What are the statistical properties of prediction errors?

**How to obtain**:
```python
from scipy.stats import skew, kurtosis

errors = y_pred.flatten() - y_true.flatten()

print(f"Mean error: {np.mean(errors):.6e}")
print(f"Std error: {np.std(errors):.6e}")
print(f"Skewness: {skew(errors):.4f}")
print(f"Kurtosis: {kurtosis(errors):.4f}")
print(f"Max absolute error: {np.max(np.abs(errors)):.6e}")
```

**Fill in**:
```
Model      | Mean Error | Std Error | Skewness | Kurtosis | Max |Error|
-----------|------------|-----------|----------|----------|-------------
LFSM       |            |           |          |          |
MFSM-CAE   |            |           |          |          |
MFSM-UNET  |            |           |          |          |
HFSM       |            |           |          |          |
```

---

### Q2.3: Performance by Damage Configuration
**Question**: How does performance vary with damage parameters?

**How to obtain**:
```python
# Bin test cases by damage characteristics
damaged_cases = test_params[test_params['damage_present'] == 'y']

# Analyze by notch location (e.g., left/center/right third)
left_third = damaged_cases[damaged_cases['notch_x'] < L/3]
center_third = damaged_cases[(damaged_cases['notch_x'] >= L/3) &
                              (damaged_cases['notch_x'] < 2*L/3)]
right_third = damaged_cases[damaged_cases['notch_x'] >= 2*L/3]

# Compute R² for each region
# Repeat for depth and width bins
```

**Fill in**:
```
Notch Location:
  Left third (x < 1.0 m):   LFSM R²=___, MFSM-CAE R²=___, MFSM-UNET R²=___, HFSM R²=___
  Center third (1-2 m):     LFSM R²=___, MFSM-CAE R²=___, MFSM-UNET R²=___, HFSM R²=___
  Right third (x > 2.0 m):  LFSM R²=___, MFSM-CAE R²=___, MFSM-UNET R²=___, HFSM R²=___

Notch Depth:
  Shallow (d < [THRESHOLD]): LFSM R²=___, MFSM-CAE R²=___, etc.
  Medium:                    ...
  Deep:                      ...

Notch Width:
  Narrow:                    ...
  Medium:                    ...
  Wide:                      ...

Pristine cases:              LFSM R²=___, MFSM-CAE R²=___, etc.
```

---

### Q2.4: Training Time Breakdown
**Question**: How long did each training stage take?

**Check your training logs or time outputs**:
```
LFSM:
  CAE training (200 epochs): [TIME] hours/minutes
  XGBoost training: [TIME] minutes
  Total: [TIME]

MFSM-CAE:
  Stage 1 pretrain (100 epochs): [TIME]
  Stage 2 finetune (200 epochs): [TIME]
  Total: [TIME]

MFSM-UNET:
  Stage 1 pretrain (100 epochs): [TIME]
  Stage 2 finetune (200 epochs): [TIME]
  Total: [TIME]

HFSM:
  CAE training (200 epochs): [TIME]
  XGBoost training: [TIME]
  Total: [TIME]
```

---

### Q2.5: Convergence Analysis
**Question**: At what epoch did each model converge?

**How to determine**:
```python
# Define convergence as: loss change < 0.1% for 5 consecutive epochs
def find_convergence_epoch(loss_history, patience=5, threshold=0.001):
    for i in range(len(loss_history) - patience):
        recent_losses = loss_history[i:i+patience]
        if max(recent_losses) - min(recent_losses) < threshold * recent_losses[0]:
            return i
    return len(loss_history)

convergence_epoch = find_convergence_epoch(train_losses)
print(f"Converged at epoch: {convergence_epoch}")
```

**Fill in**:
```
Model         | Stage | Convergence Epoch | Final Loss | Best Validation Loss (if applicable)
--------------|-------|-------------------|------------|-------------------------------------
LFSM          | 1     |                   |            |
MFSM-CAE      | 1     |                   |            |
MFSM-CAE      | 2     |                   |            |
MFSM-UNET     | 1     |                   |            |
MFSM-UNET     | 2     |                   |            |
HFSM          | 1     |                   |            |
```

---

## PRIORITY 3: MEDIUM (Valuable for comprehensive analysis)

### Q3.1: Ablation Study - Hilbert Weighting
**Question**: What is the effect of Hilbert weighting in MFSM-CAE?

**Experiment** (if not already done):
```python
# Train MFSM-CAE without Hilbert weighting (use standard MSE)
# Keep all other hyperparameters identical
# Compare:
#   - MFSM-CAE with Hilbert (current)
#   - MFSM-CAE without Hilbert (ablation)
```

**If experiment performed, fill in**:
```
Configuration:
  MFSM-CAE with Hilbert:    Test R² = [VALUE], MSE = [VALUE]
  MFSM-CAE without Hilbert: Test R² = [VALUE], MSE = [VALUE]
  Improvement: [%]

Per-sensor analysis (are improvements localized?):
  Sensor 1: ΔR² = [VALUE]
  Sensor 2: ΔR² = [VALUE]
  ...

Critical region performance (near peaks):
  With Hilbert:    R² = [VALUE]
  Without Hilbert: R² = [VALUE]
```

**If NOT performed**:
```
Status: Ablation study not conducted
Reason: [Time constraints / Computational cost / Other]
Estimated importance for paper: [High / Medium / Low]
```

---

### Q3.2: Ablation Study - Pretraining Effect
**Question**: Does pretraining on 1D data help?

**Experiment** (if not already done):
```python
# Train MFSM-CAE (or UNET) without pretraining
# Start from random initialization
# Train directly on 2D finetuning data for same 200 epochs
# Compare to MFSM with pretraining
```

**If experiment performed**:
```
Configuration:
  MFSM with pretrain (current):    Test R² = [VALUE]
  MFSM without pretrain (ablation): Test R² = [VALUE]
  Benefit of pretraining: [%] improvement

Convergence speed:
  With pretrain: converged at epoch [EPOCH]
  Without pretrain: converged at epoch [EPOCH]
  (both in stage 2 / finetuning phase)
```

---

### Q3.3: Data Efficiency Curve
**Question**: How does MFSM performance scale with amount of 2D finetuning data?

**Experiment** (if not already done):
```python
# Train MFSM-CAE with varying amounts of 2D data:
# - 100 pairs
# - 250 pairs
# - 500 pairs
# - 1000 pairs (current)

# Plot: Test R² vs Number of 2D training pairs
```

**If experiment performed**:
```
2D Training Pairs | MFSM-CAE Test R² | MFSM-UNET Test R² (if tested)
------------------|------------------|-------------------------------
100               |                  |
250               |                  |
500               |                  |
1000              |                  |

Observations:
  - Diminishing returns after [N] pairs?
  - Minimum data needed for >90% of max performance: [N] pairs
```

---

### Q3.4: Worst-Case Analysis
**Question**: What characteristics make cases difficult to predict?

**Analysis**:
```python
# Identify bottom 10% of cases by R²
worst_10pct = test_cases[test_cases['r2'] < np.percentile(r2_scores, 10)]

# Analyze common characteristics:
# - Notch location distribution
# - Depth/width ranges
# - Are they near training data boundaries?
# - Are they pristine or damaged?
```

**Fill in**:
```
Worst 10% of test cases (N = [COUNT]):
  Damage type: [% damaged, % pristine]
  Notch location: Mean = [VALUE], Std = [VALUE]
  Notch depth: Mean = [VALUE], Std = [VALUE]
  Notch width: Mean = [VALUE], Std = [VALUE]

Common failure modes:
  1. [DESCRIPTION, e.g., "Underestimate peak amplitudes"]
  2. [DESCRIPTION]
  3. [DESCRIPTION]

Comparison to training distribution:
  [Are worst cases out-of-distribution? Near boundaries? Interpolation/extrapolation?]
```

---

### Q3.5: Feature Importance (if using XGBoost)
**Question**: Which parameters are most important for latent space prediction?

**How to obtain**:
```python
import xgboost as xgb

# After training XGBoost model
importance = xgb_model.get_score(importance_type='gain')
# Or: xgb_model.feature_importances_ if using sklearn wrapper

print("Feature importances:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {score:.4f}")
```

**Fill in**:
```
LFSM XGBoost Feature Importance:
  1. [PARAMETER NAME]: [SCORE]
  2. [PARAMETER NAME]: [SCORE]
  ...
  7. [PARAMETER NAME]: [SCORE]

HFSM XGBoost Feature Importance:
  1. [PARAMETER NAME]: [SCORE]
  ...

Interpretation:
  [Are damage parameters (x_notch, d_notch, w_notch) most important?]
  [Do fixed parameters (L, ρ, E) contribute despite small variation?]
```

---

## PRIORITY 4: LOW (Nice to have for completeness)

### Q4.1: Inference Time
**Question**: How fast are predictions at inference time?

**How to measure**:
```python
import time

# Warm-up
_ = model.predict(X_test[0:1])

# Time inference
n_repeats = 100
start = time.time()
for _ in range(n_repeats):
    _ = model.predict(X_test[0:1])
end = time.time()

time_per_sample_ms = (end - start) / n_repeats * 1000
print(f"Inference time: {time_per_sample_ms:.2f} ms/sample")

# Throughput
samples_per_sec = 1000 / time_per_sample_ms
print(f"Throughput: {samples_per_sec:.1f} samples/sec")
```

**Fill in**:
```
Model      | CPU Time (ms) | GPU Time (ms) | Throughput (samples/sec)
-----------|---------------|---------------|-------------------------
LFSM       |               |               |
MFSM-CAE   |               |               |
MFSM-UNET  |               |               |
HFSM       |               |               |

Reference:
  1D Zigzag simulation: [TIME] ms/sample
  2D FEM simulation: [TIME] ms/sample

Speedup factors:
  MFSM vs 2D FEM: [FACTOR]×
  MFSM vs 1D Zigzag: [FACTOR]×
```

---

### Q4.2: GPU Memory Usage
**Question**: How much GPU memory does each model require?

**How to measure** (PyTorch):
```python
import torch

torch.cuda.reset_peak_memory_stats()
_ = model.predict(X_test)  # Or model(batch) if using PyTorch directly
peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak GPU memory: {peak_memory_mb:.1f} MB")
```

**Fill in**:
```
Model      | Training Peak (MB) | Inference Peak (MB) | Param Count
-----------|-------------------|---------------------|------------
LFSM       |                   |                     |
MFSM-CAE   |                   |                     |
MFSM-UNET  |                   |                     |
HFSM       |                   |                     |
```

---

### Q4.3: Parameter Space Coverage Visualization
**Question**: How well do training/test sets cover parameter space?

**Analysis**:
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D scatter plot: x_notch, d_notch, w_notch
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

train_damaged = train_params[train_params['damage_present'] == 'y']
test_damaged = test_params[test_params['damage_present'] == 'y']

ax.scatter(train_damaged['notch_x'], train_damaged['notch_depth'],
           train_damaged['notch_width'], c='blue', label='Train', alpha=0.6)
ax.scatter(test_damaged['notch_x'], test_damaged['notch_depth'],
           test_damaged['notch_width'], c='red', label='Test', alpha=0.8)

ax.set_xlabel('Notch Location (m)')
ax.set_ylabel('Notch Depth (m)')
ax.set_zlabel('Notch Width (m)')
plt.legend()
plt.savefig('parameter_space_coverage.png')
```

**Provide**:
- Path to visualization: [PATH]
- Observation: [Do test cases fill gaps in training data? Are there clusters? Uniform coverage?]

---

### Q4.4: Cross-Validation Results (if performed)
**Question**: If k-fold cross-validation was used, what were the results?

**Fill in**:
```
K-fold CV (k = [K]):
  LFSM:      Mean R² = [VALUE] ± [STD]
  MFSM-CAE:  Mean R² = [VALUE] ± [STD]
  MFSM-UNET: Mean R² = [VALUE] ± [STD]
  HFSM:      Mean R² = [VALUE] ± [STD]

OR:

Cross-validation not performed: [REASON]
```

---

### Q4.5: Hyperparameter Sensitivity (if explored)
**Question**: Were different hyperparameters tested?

**If yes**:
```
Latent dimension sweep (LFSM/MFSM):
  d=25:  R² = [VALUE]
  d=50:  R² = [VALUE] (current)
  d=100: R² = [VALUE]

Learning rate sweep (finetuning):
  lr=1e-6: R² = [VALUE]
  lr=1e-5: R² = [VALUE] (current)
  lr=1e-4: R² = [VALUE]

Epoch count sweep (finetuning):
  100 epochs:  R² = [VALUE]
  200 epochs:  R² = [VALUE] (current)
  300 epochs:  R² = [VALUE]
```

**If no**:
```
Hyperparameter exploration not conducted: [REASON]
Rationale for chosen values: [e.g., "Based on preliminary tests" / "Literature values" / "Default values"]
```

---

## ADDITIONAL INFORMATION NEEDED

### Data Augmentation Details
**Question**: How were 750 1D cases augmented to 6000+?

**Provide**:
```
Augmentation strategy:
  Method 1: [e.g., "Gaussian noise injection, σ = [VALUE]"]
  Method 2: [e.g., "Parameter perturbation within ±[X]%"]
  Method 3: [e.g., "Time series interpolation"]
  Other: [DESCRIPTION]

Augmentation factor: [N]× (750 → [FINAL COUNT])

Validation of augmentation:
  [Did you verify augmented data are realistic?]
  [How did you ensure no information leakage to test set?]
```

---

### Parameter Bounds
**Question**: What are the exact bounds for damage parameters?

**From code analysis, you have**:
```
Length: L = 3.0 m (fixed)
Density: ρ = 2700 kg/m³ (±0.15% variation in some cases)
Young's modulus: E = 70 GPa (±0.15% variation in some cases)
```

**Still needed**:
```
Notch location range: [MIN] ≤ x_notch ≤ [MAX] meters
Notch depth range: [MIN] ≤ d_notch ≤ [MAX] meters
Notch width range: [MIN] ≤ w_notch ≤ [MAX] meters

Physical constraints:
  - Minimum distance from beam ends: [VALUE] m
  - Maximum notch depth (as % of beam height): [VALUE]%
  - Notch aspect ratio constraints: [if any]
```

---

### Computational Environment
**Question**: What hardware/software was used?

**Fill in**:
```
Hardware:
  GPU: [Model, e.g., "NVIDIA RTX 3090, 24GB VRAM"]
  CPU: [Model and core count, e.g., "Intel i9-12900K, 16 cores"]
  RAM: [Amount, e.g., "64 GB DDR4"]
  Storage: [Type, e.g., "NVMe SSD"]

Software:
  OS: [e.g., "Ubuntu 20.04 LTS"]
  Python version: [e.g., "3.8.10"]
  PyTorch version: [e.g., "1.12.1"]
  CUDA version: [e.g., "11.6"]
  XGBoost version: [e.g., "1.6.2"]
  NumPy version: [e.g., "1.21.2"]
  SciPy version: [e.g., "1.7.1"]
  Scikit-learn version: [e.g., "1.0.2"]
```

---

## HOW TO PROVIDE THIS INFORMATION

### Option 1: Structured Response
Copy this document and fill in all [VALUE] / [PATH] placeholders directly.

### Option 2: Data Files
Provide CSV files with:
- `test_results.csv`: columns [case_id, model, r2, mse, nmse]
- `per_sensor_results.csv`: columns [sensor_id, model, r2, mse]
- `timing_results.csv`: columns [model, stage, time_hours, gpu_memory_mb]

### Option 3: Jupyter Notebook
Create a notebook that:
1. Loads all trained models
2. Computes all metrics in this document
3. Generates all required visualizations
4. Exports summary tables as CSVs

### Option 4: Combination
- Provide critical metrics (Priority 1) in text response
- Upload visualization PNGs to Claude_res/
- Share paths to model checkpoints and data files

---

## TIMELINE SUGGESTION

**Day 1 (2-3 hours)**:
- Run all four trained models on test set
- Compute Priority 1 metrics (Q1.1-Q1.5)
- Generate best/worst case visualizations
- Provide results → enables Results section writing

**Day 2 (2-3 hours)**:
- Compute Priority 2 metrics (Q2.1-Q2.5)
- Analyze per-sensor performance
- Extract timing data from logs
- Provide results → enables Discussion section writing

**Day 3 (optional, 1-2 hours)**:
- Compute Priority 3 metrics if time allows
- Run ablation studies if not yet done
- Provide results → enhances paper depth

**Day 4 (optional)**:
- Gather Priority 4 and additional info
- Final polishing of visualizations
- Provide results → completes all sections

---

## END OF QUESTIONNAIRE

Once you provide the critical data (Priority 1-2), we can begin drafting the report sections using the outline and templates provided in the companion documents.
