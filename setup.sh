#!/bin/bash
# setup.sh — run once per vast.ai instance before the main script
# Usage: bash setup.sh
# Takes ~5 min on a fresh PyTorch image

set -e  # stop on any error

echo "=== [1/4] System packages ==="
apt-get update -qq
apt-get install -y -qq tmux htop

echo ""
echo "=== [2/4] Python packages ==="

# torch alone first (~2 GB) — separate so a timeout only retries torch, not everything
pip install --quiet --timeout=120 --retries=5 torch

# smaller packages
pip install --quiet --timeout=120 --retries=5 \
    transformers \
    datasets \
    scipy \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn \
    tqdm

# textattack last — heaviest dependency tree
pip install --quiet --timeout=120 --retries=5 textattack

echo ""
echo "=== [3/4] NLTK data ==="
python3 -c "
import nltk
for pkg in [
    'averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger',
    'punkt', 'punkt_tab', 'stopwords', 'omw-1.4', 'wordnet', 'universal_tagset',
]:
    nltk.download(pkg, quiet=True)
print('NLTK ready.')
"

echo ""
echo "=== [4/4] Verify GPU + PyTorch ==="
python3 -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.version.cuda}')
print(f'GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')
print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB' if torch.cuda.is_available() else '')

# Check vmap availability
try:
    from torch.func import vjp, vmap
    print('vmap    : available (batched sigma_low enabled)')
except ImportError:
    print('vmap    : NOT available — upgrade PyTorch to 2.0+')
    print('          pip install torch>=2.0 --upgrade')
"

echo ""
echo "=== Setup complete ==="
echo "Run the experiment with:"
echo "  tmux new -s tf"
echo "  ATTACK_NAME=TextFooler python3 scm_v3_run.py 2>&1 | tee logs_TextFooler.txt"
