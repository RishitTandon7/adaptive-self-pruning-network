import torch
import pytest
from models.prunable_layer import PrunableLinear
from models.network import SelfPruningNetwork

def test_prunable_linear_shapes():
    layer = PrunableLinear(100, 50)
    assert layer.weight.shape == (50, 100)
    assert layer.bias.shape == (50,)
    assert layer.gate_scores.shape == (50, 100)

def test_prunable_linear_forward():
    layer = PrunableLinear(100, 50)
    x = torch.randn(4, 100)
    out = layer(x)
    assert out.shape == (4, 50)

def test_self_pruning_network_forward():
    model = SelfPruningNetwork()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

def test_sparsity_loss_computation():
    model = SelfPruningNetwork()
    loss = model.compute_sparsity_loss()
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

def test_parameter_counting():
    model = SelfPruningNetwork()
    params = model.count_parameters()
    assert "total" in params
    assert "nonzero" in params
    assert "compression" in params
    assert params["total"] > 0
    assert params["total"] == params["nonzero"]  # Initially no exact zeros

def test_hard_pruning():
    layer = PrunableLinear(10, 5)
    # Force some gates to be very negative so sigmoid goes to 0
    with torch.no_grad():
        layer.gate_scores[0, :] = -100.0  # Should be pruned
        layer.gate_scores[1, :] = 100.0   # Should be kept
        
    pruned_count = layer.hard_prune(threshold=0.01)
    
    assert pruned_count > 0
    # Weights for the pruned gates should be exactly 0
    assert torch.all(layer.weight[0, :] == 0.0)
    assert not torch.all(layer.weight[1, :] == 0.0)
