"""
Sparsity tracking utilities.

Provides functions to compute and log sparsity metrics across
the network and per-layer.
"""

import torch
from models.network import SelfPruningNetwork


def compute_sparsity_stats(model: SelfPruningNetwork, threshold: float = 0.01) -> dict:
    """Compute comprehensive sparsity statistics for the model.

    Args:
        model: The self-pruning network.
        threshold: Gate values below this are considered pruned.

    Returns:
        Dictionary containing:
            - overall_sparsity: Network-wide sparsity ratio
            - total_weights: Total number of gated weights
            - pruned_weights: Number of weights below threshold
            - layer_stats: Per-layer breakdown
    """
    total_weights = 0
    pruned_weights = 0
    layer_stats = {}

    for name, module in model.named_modules():
        from models.prunable_layer import PrunableLinear
        if isinstance(module, PrunableLinear):
            gates = module.get_gate_values()
            total = gates.numel()
            pruned = (gates < threshold).sum().item()
            total_weights += total
            pruned_weights += pruned

            layer_stats[name] = {
                "sparsity": pruned / total,
                "total": total,
                "pruned": pruned,
                "mean_gate": gates.mean().item(),
                "min_gate": gates.min().item(),
                "max_gate": gates.max().item(),
            }

    overall_sparsity = pruned_weights / total_weights if total_weights > 0 else 0.0

    return {
        "overall_sparsity": overall_sparsity,
        "total_weights": total_weights,
        "pruned_weights": pruned_weights,
        "layer_stats": layer_stats,
    }


def log_sparsity(model: SelfPruningNetwork, epoch: int, threshold: float = 0.01):
    """Print a formatted sparsity report to stdout.

    Args:
        model: The self-pruning network.
        epoch: Current epoch number (for display).
        threshold: Gate threshold for pruning.
    """
    stats = compute_sparsity_stats(model, threshold)
    print(f"\n--- Sparsity Report (Epoch {epoch}) ---")
    print(f"  Overall: {stats['overall_sparsity']:.2%}  "
          f"({stats['pruned_weights']:,}/{stats['total_weights']:,} weights pruned)")

    for name, ls in stats["layer_stats"].items():
        print(f"  {name:>8s}: {ls['sparsity']:>6.2%}  "
              f"(mean gate={ls['mean_gate']:.4f}, "
              f"range=[{ls['min_gate']:.4f}, {ls['max_gate']:.4f}])")
    print()
