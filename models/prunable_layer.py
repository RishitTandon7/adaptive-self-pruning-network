"""
PrunableLinear: A custom linear layer with learnable gate scores.

Each weight has an associated gate score. During the forward pass, gates are
computed as sigmoid(gate_scores) and element-wise multiplied with the weights.
A sparsity penalty on the gates drives the network to learn which weights
are unnecessary, effectively self-pruning during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """Linear layer with learnable per-weight gate scores for self-pruning.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.

    Attributes:
        weight (nn.Parameter): Learnable weight matrix of shape (out_features, in_features).
        bias (nn.Parameter): Learnable bias vector of shape (out_features,).
        gate_scores (nn.Parameter): Learnable gate scores of shape (out_features, in_features).
            Initialized to +5.0 so that sigmoid(gate_scores) ≈ 1.0 at the start,
            meaning all weights are fully active before any pruning pressure is applied.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — initialized to 3.0 so sigmoid ≈ 0.95 (near full capacity).
        # This allows the network to start with most weights active,
        # letting the sparsity regularizer prune them gracefully over time.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 3.0))

        # Kaiming uniform initialization for weights (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated weights.

        1. Compute gates = sigmoid(gate_scores)  — values in [0, 1]
        2. Compute pruned_weights = weight * gates
        3. Return F.linear(x, pruned_weights, bias)
        """
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gate_values(self) -> torch.Tensor:
        """Return current gate activation values (sigmoid of gate_scores)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def get_sparsity(self, threshold: float = 0.01) -> float:
        """Compute the fraction of gates below threshold (i.e., effectively pruned).

        Args:
            threshold: Gate values below this are considered pruned.

        Returns:
            Sparsity ratio in [0, 1].
        """
        gates = self.get_gate_values()
        pruned = (gates < threshold).sum().item()
        total = gates.numel()
        return pruned / total

    def hard_prune(self, threshold: float = 0.01) -> int:
        """Zero out weights whose gate values fall below the threshold.

        This permanently removes pruned parameters by setting both the weight
        and gate_score to zero, making the pruning irreversible.

        Args:
            threshold: Gate values below this trigger pruning.

        Returns:
            Number of weights pruned.
        """
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            mask = gates < threshold
            self.weight[mask] = 0.0
            # Set gate_scores to a large negative value so sigmoid → 0
            self.gate_scores[mask] = -10.0
            return mask.sum().item()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"sparsity={self.get_sparsity():.1%}"
        )
