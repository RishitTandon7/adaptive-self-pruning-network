"""Quick validation that all modules import and work correctly."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

from models.prunable_layer import PrunableLinear
from models.network import SelfPruningNetwork
from training.train import train_one_epoch, evaluate
from utils.data import get_cifar10_loaders
from utils.sparsity import compute_sparsity_stats

# Test PrunableLinear
layer = PrunableLinear(100, 50)
x = torch.randn(4, 100)
out = layer(x)
assert out.shape == (4, 50), f"Expected (4, 50), got {out.shape}"
print(f"[OK] PrunableLinear: input (4,100) -> output {out.shape}")
print(f"     Initial sparsity: {layer.get_sparsity()*100:.1f}%")

# Test SelfPruningNetwork
model = SelfPruningNetwork()
img = torch.randn(2, 3, 32, 32)
logits = model(img)
assert logits.shape == (2, 10), f"Expected (2, 10), got {logits.shape}"
print(f"[OK] SelfPruningNetwork: input (2,3,32,32) -> output {logits.shape}")
print(f"     Overall sparsity: {model.get_overall_sparsity()*100:.1f}%")
print(f"     Sparsity loss: {model.compute_sparsity_loss().item():.2f}")
print(f"     Parameters: {model.count_parameters()}")

# Test layer sparsities
layer_sp = model.get_layer_sparsities()
for name, info in layer_sp.items():
    print(f"     {name}: {info['sparsity']*100:.1f}% sparse ({info['pruned_weights']}/{info['total_weights']})")

print("\n[OK] ALL VALIDATIONS PASSED - Ready to train!")
