#!/bin/bash
# System Information Collection Script
# For Thesis Experimental Setup Section
# Run with: bash gather_system_info.sh > system_info_output.txt

echo "======================================"
echo "SYSTEM INFORMATION COLLECTION"
echo "Date: $(date)"
echo "======================================"
echo ""

echo "--- CPU Information ---"
lscpu | grep -E "Model name|Architecture|CPU\(s\)|Thread|Core|MHz|Socket"
echo ""

echo "--- GPU Information ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader
    echo ""
    echo "CUDA Runtime Version:"
    nvcc --version 2>/dev/null || echo "nvcc not in PATH"
else
    echo "No NVIDIA GPU detected or nvidia-smi not available"
fi
echo ""

echo "--- Memory Information ---"
free -h | grep Mem
cat /proc/meminfo | grep MemTotal
echo ""

echo "--- Storage Information ---"
echo "Thesis directory storage:"
df -h /home/mecharoy/Thesis
echo ""
echo "Block devices:"
lsblk -d -o name,type,size,rota | grep -v loop
echo ""

echo "--- Operating System ---"
if [ -f /etc/os-release ]; then
    cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_ID"
else
    echo "OS release file not found"
fi
echo ""
echo "Kernel:"
uname -a
echo ""

echo "--- Python & Library Versions ---"
cd /home/mecharoy/Thesis

# Check if venv exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated: venv"
    echo ""

    echo "Python version:"
    python --version
    echo ""

    echo "Installed packages:"
    python << 'PYEOF'
import sys

def safe_import(module_name, attr='__version__'):
    try:
        mod = __import__(module_name)
        version = getattr(mod, attr, 'Unknown')
        return f"{module_name}: {version}"
    except ImportError:
        return f"{module_name}: NOT INSTALLED"
    except Exception as e:
        return f"{module_name}: ERROR - {str(e)}"

libraries = [
    ('torch', '__version__'),
    ('xgboost', '__version__'),
    ('sklearn', '__version__'),
    ('numpy', '__version__'),
    ('pandas', '__version__'),
    ('scipy', '__version__'),
    ('matplotlib', '__version__'),
    ('joblib', '__version__'),
]

for lib, attr in libraries:
    print(safe_import(lib, attr))

# Special PyTorch CUDA checks
print("\n--- PyTorch CUDA Configuration ---")
try:
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
except Exception as e:
    print(f"Error checking PyTorch CUDA: {str(e)}")

# Check for XGBoost GPU support
print("\n--- XGBoost GPU Support ---")
try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
    print(f"GPU support built: {xgb.build_info()['USE_CUDA']}")
except Exception as e:
    print(f"Cannot check XGBoost GPU support: {str(e)}")
PYEOF

else
    echo "Virtual environment 'venv' not found!"
    echo "Using system Python:"
    python3 --version
fi
echo ""

echo "--- Dataset Information ---"
if [ -d "/home/mecharoy/Thesis/parameters" ]; then
    cd /home/mecharoy/Thesis/parameters
    echo "Dataset sample counts (excluding header):"
    for file in LFSM*.csv train_responses*.csv test_responses*.csv database.csv; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file")
            samples=$((lines - 1))
            printf "%-40s: %6d samples\n" "$file" "$samples"
        fi
    done
else
    echo "Parameters directory not found!"
fi
echo ""

echo "--- Code Files ---"
if [ -d "/home/mecharoy/Thesis/Code" ]; then
    cd /home/mecharoy/Thesis/Code
    echo "Python scripts in Code directory:"
    ls -lh *.py 2>/dev/null || echo "No Python files found"
else
    echo "Code directory not found!"
fi
echo ""

echo "======================================"
echo "COLLECTION COMPLETE"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Review the output above"
echo "2. Copy relevant information to experimental_setup_template.md"
echo "3. Fill in remaining details (training times, hyperparameter justifications)"
