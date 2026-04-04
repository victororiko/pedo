#!/bin/bash
# =============================================================================
# Elumina — RunPod Environment Setup
# Run this script after SSH-ing into your RunPod instance.
# Recommended template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1
# Recommended GPU: A100 80GB or H100 for 27B model
# =============================================================================

set -e

echo "=========================================="
echo "  Elumina — RunPod Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
apt-get update -qq && apt-get install -y -qq git-lfs htop tmux > /dev/null 2>&1

# Set up git LFS (for HuggingFace model downloads)
git lfs install

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Login to HuggingFace (Gemma requires access agreement)
echo ""
echo "=========================================="
echo "  HuggingFace Login Required"
echo "=========================================="
echo "Gemma 4 requires accepting Google's license on HuggingFace."
echo "1. Go to https://huggingface.co/google/gemma-4-27b-it"
echo "2. Accept the license agreement"
echo "3. Create a token at https://huggingface.co/settings/tokens"
echo ""
read -p "Enter your HuggingFace token (or press Enter to skip): " HF_TOKEN

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
    echo "HuggingFace login successful."
else
    echo "Skipped HuggingFace login. Run 'huggingface-cli login' later."
fi

# Optional: WandB setup
echo ""
read -p "Set up Weights & Biases tracking? (y/n): " SETUP_WANDB
if [ "$SETUP_WANDB" = "y" ]; then
    wandb login
else
    echo "Skipping WandB. Set report_to: 'none' in config/training_config.yaml"
fi

# Create output directories
mkdir -p output data/processed data/raw data/cultural

# Verify GPU
echo ""
echo "=========================================="
echo "  GPU Information"
echo "=========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  GPU {i}: {name} ({mem:.1f} GB)')
"

# Flash Attention check
python -c "
try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('Flash Attention: NOT INSTALLED (training will be slower)')
    print('  Install with: pip install flash-attn --no-build-isolation')
"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Add training data to data/raw/ and data/cultural/"
echo "  2. Run: python scripts/prepare_data.py"
echo "  3. Run: python scripts/train.py"
echo "  4. Run: python scripts/merge_adapter.py"
echo "  5. Run: python scripts/inference.py"
echo ""
echo "Tip: Use tmux to keep training running if SSH disconnects:"
echo "  tmux new -s train"
echo "  python scripts/train.py"
echo "  (Ctrl+B, D to detach; tmux attach -t train to reattach)"
