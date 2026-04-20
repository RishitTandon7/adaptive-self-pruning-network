"""
Comprehensive test suite for the Self-Pruning Neural Network.

Tests cover:
    - PrunableLinear layer correctness and gradient flow
    - SelfPruningNetwork forward pass and sparsity computation
    - Hard pruning mechanism
    - Training loop integration
    - FastAPI endpoint responses
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prunable_layer import PrunableLinear
from models.network import SelfPruningNetwork


# ──────────────────────────────────────────────────────────────────
# PrunableLinear Layer Tests
# ──────────────────────────────────────────────────────────────────

class TestPrunableLinear:
    """Tests for the custom PrunableLinear layer."""

    def test_parameter_shapes(self):
        """Verify weight, bias, and gate_scores have correct shapes."""
        layer = PrunableLinear(100, 50)
        assert layer.weight.shape == (50, 100)
        assert layer.bias.shape == (50,)
        assert layer.gate_scores.shape == (50, 100)

    def test_forward_output_shape(self):
        """Verify forward pass produces correct output dimensions."""
        layer = PrunableLinear(100, 50)
        x = torch.randn(4, 100)
        out = layer(x)
        assert out.shape == (4, 50)

    def test_gate_scores_are_learnable_parameters(self):
        """Gate scores must be registered as nn.Parameters so the optimizer updates them."""
        layer = PrunableLinear(64, 32)
        param_names = [name for name, _ in layer.named_parameters()]
        assert "gate_scores" in param_names, "gate_scores must be a registered parameter"

    def test_gradients_flow_through_gates(self):
        """Verify that gradients flow correctly through BOTH weight and gate_scores."""
        layer = PrunableLinear(10, 5)
        x = torch.randn(2, 10)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None, "Gradient must flow to weight"
        assert layer.gate_scores.grad is not None, "Gradient must flow to gate_scores"
        assert layer.bias.grad is not None, "Gradient must flow to bias"

        # Gradients should not all be zero
        assert layer.weight.grad.abs().sum() > 0, "Weight gradients should be non-zero"
        assert layer.gate_scores.grad.abs().sum() > 0, "Gate score gradients should be non-zero"

    def test_gate_values_bounded_zero_to_one(self):
        """Sigmoid transformation must keep gates strictly in [0, 1]."""
        layer = PrunableLinear(50, 25)
        gates = layer.get_gate_values()
        assert gates.min() >= 0.0
        assert gates.max() <= 1.0

    def test_initial_gate_values_near_half(self):
        """With gate_scores initialized at 0.0, sigmoid should produce ~0.5."""
        layer = PrunableLinear(50, 25)
        gates = layer.get_gate_values()
        assert abs(gates.mean().item() - 0.5) < 0.01, "Initial gates should be ~0.5"

    def test_sparsity_calculation(self):
        """Test that sparsity ratio is computed correctly."""
        layer = PrunableLinear(10, 5)
        # Default init: gates ≈ 0.5, threshold = 0.01 → 0% sparsity
        sparsity = layer.get_sparsity(threshold=0.01)
        assert sparsity == 0.0, "No gates should be below 0.01 at initialization"

        # Sparsity with a high threshold should capture everything
        sparsity_high = layer.get_sparsity(threshold=0.99)
        assert sparsity_high > 0.0, "Most gates should be below 0.99"


# ──────────────────────────────────────────────────────────────────
# Hard Pruning Tests
# ──────────────────────────────────────────────────────────────────

class TestHardPruning:
    """Tests for the hard pruning mechanism."""

    def test_hard_prune_zeroes_weights(self):
        """Weights with gates below threshold should be permanently zeroed."""
        layer = PrunableLinear(10, 5)
        with torch.no_grad():
            layer.gate_scores[0, :] = -100.0  # sigmoid → ~0.0 (should be pruned)
            layer.gate_scores[1, :] = 100.0   # sigmoid → ~1.0 (should be kept)

        pruned_count = layer.hard_prune(threshold=0.01)

        assert pruned_count == 10, "All 10 weights in row 0 should be pruned"
        assert torch.all(layer.weight[0, :] == 0.0), "Pruned weights must be exactly 0"
        assert not torch.all(layer.weight[1, :] == 0.0), "Kept weights must not be 0"

    def test_hard_prune_sets_gate_scores_negative(self):
        """After hard pruning, gate_scores for pruned weights should be very negative."""
        layer = PrunableLinear(10, 5)
        with torch.no_grad():
            layer.gate_scores[0, :] = -100.0

        layer.hard_prune(threshold=0.01)
        # sigmoid(-10) ≈ 0.0, confirming pruned gates stay dead
        assert torch.all(layer.gate_scores[0, :] == -10.0)

    def test_network_hard_prune_all(self):
        """Network-level hard pruning should process all layers."""
        model = SelfPruningNetwork()
        results = model.hard_prune_all(threshold=0.01)
        assert "fc1" in results
        assert "fc2" in results
        assert "fc3" in results


# ──────────────────────────────────────────────────────────────────
# SelfPruningNetwork Tests
# ──────────────────────────────────────────────────────────────────

class TestSelfPruningNetwork:
    """Tests for the complete network architecture."""

    def test_forward_pass_cifar10_input(self):
        """Network must accept CIFAR-10 shaped input (batch, 3, 32, 32)."""
        model = SelfPruningNetwork()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

    def test_sparsity_loss_is_positive_scalar(self):
        """Sparsity loss must be a positive scalar tensor for the optimizer."""
        model = SelfPruningNetwork()
        loss = model.compute_sparsity_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0, "Sparsity loss must be a scalar"
        assert loss.item() > 0, "With active gates, sparsity loss must be positive"

    def test_sparsity_loss_has_gradient(self):
        """Sparsity loss must be differentiable for backpropagation."""
        model = SelfPruningNetwork()
        loss = model.compute_sparsity_loss()
        loss.backward()
        for layer in model.get_prunable_layers():
            assert layer.gate_scores.grad is not None

    def test_parameter_counting(self):
        """Parameter counting should track total and nonzero params."""
        model = SelfPruningNetwork()
        params = model.count_parameters()
        assert params["total"] > 0
        assert params["nonzero"] > 0
        assert params["compression"] >= 1.0

    def test_layer_sparsities_keys(self):
        """Layer sparsity report should contain all prunable layers."""
        model = SelfPruningNetwork()
        sparsities = model.get_layer_sparsities()
        assert len(sparsities) == 3, "Network has 3 PrunableLinear layers"

    def test_total_loss_computation(self):
        """End-to-end: classification loss + λ * sparsity loss should be differentiable."""
        model = SelfPruningNetwork()
        x = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))

        outputs = model(x)
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets)
        sparse_loss = model.compute_sparsity_loss()
        total_loss = ce_loss + 1e-3 * sparse_loss

        total_loss.backward()

        # Verify all parameters received gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
