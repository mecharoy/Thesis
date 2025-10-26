# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a computational mechanics research project focused on **Multi-Fidelity Surrogate Modeling (MFSM)** for beam response analysis. The system combines Low-Fidelity Surrogate Models (LFSM) based on 1D zigzag theory with High-Fidelity data from 2D Finite Element Method (FEM) simulations to create efficient predictive models.

## Project Architecture

### Core Methodology
The project implements a sophisticated multi-phase training pipeline:

1. **Phase 1**: LFSM Pre-training on 1D zigzag theory data
2. **Phase 2**: XGBoost surrogate training on LFSM latent vectors
3. **Phase 3**: MFSM fine-tuning on 2D FEM data
4. **Phase 4**: XGBoost fine-tuning on combined LFSM+HFSM data

### Key Components

**Autoencoder Architecture** (`LFSM-MFSM.py`)
- Encoder: Compresses 1500-timestep time series into 30-dimensional latent space
- Decoder: Reconstructs time series from latent representations
- Uses batch normalization, LeakyReLU activations, and dropout regularization

**Surrogate Model**
- XGBoost regression for parameter-to-latent mapping
- GPU-accelerated training when available
- Weighted sample training to emphasize high-fidelity data

**Data Processing Pipeline**
- Parameter scaling to [-1, 1] range using MinMaxScaler
- Time series response data (1500 timesteps per sample)
- Region of Interest (ROI) evaluation for true performance metrics

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (required for all Python commands)
source venv/bin/activate

# Install dependencies (if needed)
pip install pandas numpy torch scikit-learn xgboost matplotlib joblib
```

### Running Main Training Pipeline
```bash
# Main MFSM training (single-file implementation)
python playground.py

# Alternative LFSM-MFSM implementation
python Code/LFSM-MFSM.py
```

### Data Generation and Processing
```bash
# Generate 2D FEM datasets
python Code/dataset2Dgenfinal.py

# Generate zigzag theory datasets
python Code/datagenzigzag.py

# Run inverse problem examples
python Code/inverseproblemplayground.py
```

### Comparison and Analysis
```bash
# Compare zigzag vs Timoshenko theories
python Code/comparison_zigzag_timoshenko.py

# Pure HFSM modeling
python Code/HFSM.py
```

## Data Structure and File Organization

### Input Data Format
**Parameters** (7 columns): `notch_x`, `notch_depth`, `notch_width`, `length`, `density`, `youngs_modulus`, `location`
**Responses**: 1500 timestep time series data (`r0` to `r1499` or `r_0` to `r_1499`)

### Key Directories
- `parameters/`: CSV training and test datasets for both LFSM and HFSM
- `Code/`: Main Python implementations and documentation
- `Results/`: Training outputs, model files, and evaluation results
- `Claude_res/`: Output directory for Claude-generated results (required by user instructions)
- `Illustration/`: Mathematical explanations and visualizations

### Configuration
All model hyperparameters and file paths are centralized in a `CONFIG` dictionary at the top of each main script. Key settings include:
- GPU/CPU selection and XGBoost GPU acceleration
- File paths for training/test data
- Model architecture parameters (latent dimensions, batch sizes, learning rates)
- Multi-fidelity training weights and sampling strategies

## Model Training and Evaluation

### Training Strategy
The system uses a sophisticated transfer learning approach:
1. **Pre-training** on abundant 1D data (LFSM) to learn basic physics
2. **Fine-tuning** on limited 2D data (HFSM) to capture high-fidelity behavior
3. **Weighted loss functions** emphasizing high-fidelity data (typically 3x weight)
4. **Early stopping** based on validation loss with patience mechanisms

### Key Evaluation Metrics
- **ROI R²**: Primary metric focusing on dynamic response regions only
- **Full R²**: Secondary metric (inflated by quiescent zones)
- **NMSE**: Normalized Mean Squared Error in percentage
- **Latent Space R²**: Parameter-to-latent mapping quality

### Model Persistence
- Autoencoder models saved as `.pth` files (PyTorch)
- XGBoost models saved as `.joblib` files
- Parameter scalers saved for consistent preprocessing
- Support for loading existing models to skip retraining

## Important Implementation Notes

### Numerical Stability
- Uses log-sum-exp and other numerical stability techniques
- Implements gradient clipping for training stability
- Handles edge cases in R² calculation and ROI determination

### Performance Optimization
- GPU acceleration for both neural networks and XGBoost
- Efficient data loading with PyTorch DataLoader
- Vectorized operations using NumPy
- Memory-efficient handling of large datasets

### Scientific Best Practices
- Random seed setting for reproducibility
- Comprehensive logging of training progress
- Region-specific evaluation focusing on dynamic response
- Proper train/validation/test splits with stratification

## User Preferences and Guidelines

From the user's CLAUDE.md instructions:
- **Single-file preference**: Default to single-file implementations unless complexity demands modularization
- **Educational approach**: Explain code logic clearly as if user is learning
- **Validation emphasis**: Double-check implementations and provide validation steps
- **Output directory**: Always produce results in `Claude_res/` directory
- **Virtual environment**: Use `venv` for running Python codes
- **Question-asking**: Always ask important questions before making major decisions

## Common Workflows

### Training New Models
1. Update `CONFIG` dictionary with desired file paths and hyperparameters
2. Set `USE_EXISTING_MODELS = False` to train from scratch
3. Run `python playground.py` - training progress will be logged
4. Models automatically saved to `OUTPUT_DIR` with descriptive names

### Using Pre-trained Models
1. Set `USE_EXISTING_MODELS = True` in configuration
2. Ensure model files exist in output directory
3. System will load models and skip training phases
4. Proceed directly to evaluation and prediction

### Generating Predictions
1. Use `predict_timeseries_from_params()` function with trained models
2. Call `create_comparison_plots()` for comprehensive visualizations
3. Use `dump_mfsm_interleaved_predictions()` for CSV output format

### Evaluating Model Performance
1. Run `evaluate_on_dataset()` for comprehensive metrics
2. Pay attention to ROI R² (primary performance indicator)
3. Review individual sample plots for failure analysis
4. Check training logs for convergence patterns