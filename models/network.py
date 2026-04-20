"""
SelfPruningNetwork: A feedforward classifier built with PrunableLinear layers.

Architecture (per task.md):
    Input:  3072  (CIFAR-10: 3×32×32 flattened)
    Layer1: 512   + ReLU
    Layer2: 256   + ReLU
    Output: 10
"""

import torch
import torch.nn as nn

from .prunable_layer import PrunableLinear


class SelfPruningNetwork(nn.Module):
    """Feedforward network with prunable layers for CIFAR-10 classification.

    The network uses PrunableLinear layers so that gate-based sparsity
    regularization can drive automatic weight pruning during training.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # Prunable hidden layers
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)

        # Output layer (prunable too — the network decides what to keep)
        self.fc3 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: flatten → fc1 → relu → fc2 → relu → fc3."""
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_prunable_layers(self) -> list:
        """Return a list of all PrunableLinear layers in the network."""
        return [module for module in self.modules() if isinstance(module, PrunableLinear)]

    def compute_sparsity_loss(self) -> torch.Tensor:
        """Compute the sparsity regularization term.

        SparsityLoss = sum of all sigmoid(gate_scores) across all prunable layers.
        Minimizing this encourages gate values to approach 0, pruning weights.
        """
        sparsity_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.get_prunable_layers():
            sparsity_loss = sparsity_loss + torch.sigmoid(layer.gate_scores).sum()
        return sparsity_loss

    def get_overall_sparsity(self, threshold: float = 0.01) -> float:
        """Compute network-wide sparsity (fraction of gates below threshold)."""
        total_weights = 0
        pruned_weights = 0
        for layer in self.get_prunable_layers():
            gates = layer.get_gate_values()
            total_weights += gates.numel()
            pruned_weights += (gates < threshold).sum().item()
        return pruned_weights / total_weights if total_weights > 0 else 0.0

    def get_layer_sparsities(self, threshold: float = 0.01) -> dict:
        """Return per-layer sparsity information.

        Returns:
            Dict mapping layer name → {sparsity, total_weights, pruned_weights}
        """
        sparsities = {}
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gate_values()
                total = gates.numel()
                pruned = (gates < threshold).sum().item()
                sparsities[name] = {
                    "sparsity": pruned / total,
                    "total_weights": total,
                    "pruned_weights": pruned,
                }
        return sparsities

    def hard_prune_all(self, threshold: float = 0.01) -> dict:
        """Apply hard pruning to all prunable layers.

        Returns:
            Dict mapping layer name → number of weights pruned.
        """
        results = {}
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                results[name] = module.hard_prune(threshold)
        return results

    def count_parameters(self) -> dict:
        """Count total and effective (non-zero) parameters."""
        total = sum(p.numel() for p in self.parameters())
        nonzero = sum((p != 0).sum().item() for p in self.parameters())
        return {"total": total, "nonzero": nonzero, "compression": total / max(nonzero, 1)}
