"""
Training engine for the self-pruning neural network.

Implements the training loop with:
    - Cross-entropy classification loss
    - Sparsity regularization loss (λ × sum of sigmoid(gate_scores))
    - Combined loss backpropagation
    - Per-epoch evaluation and sparsity tracking
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.network import SelfPruningNetwork
from utils.sparsity import compute_sparsity_stats
from utils.logger import setup_logger

logger = setup_logger("training")


def train_one_epoch(
    model: SelfPruningNetwork,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_sparse: float,
    device: torch.device,
) -> dict:
    """Train the model for one epoch.

    Args:
        model: Self-pruning network.
        train_loader: Training data loader.
        optimizer: Optimizer instance.
        lambda_sparse: Sparsity regularization coefficient (λ).
        device: Device to train on.

    Returns:
        Dict with avg_loss, avg_ce_loss, avg_sparse_loss, correct, total.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_ce_loss = 0.0
    running_sparse_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Classification loss
        ce_loss = criterion(outputs, targets)

        # Sparsity regularization loss
        sparse_loss = model.compute_sparsity_loss()

        # Combined loss
        loss = ce_loss + lambda_sparse * sparse_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_sparse_loss += sparse_loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    num_batches = len(train_loader)
    return {
        "avg_loss": running_loss / num_batches,
        "avg_ce_loss": running_ce_loss / num_batches,
        "avg_sparse_loss": running_sparse_loss / num_batches,
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
    }


@torch.no_grad()
def evaluate(
    model: SelfPruningNetwork,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate the model on the test set.

    Args:
        model: Self-pruning network.
        test_loader: Test data loader.
        device: Device to evaluate on.

    Returns:
        Dict with accuracy, correct, total, avg_loss.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
        "avg_loss": running_loss / len(test_loader),
    }


def train_model(
    model: SelfPruningNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    lambda_sparse: float = 0.0,
    lambda_schedule: str = "constant",
    lambda_max: float = 1e-2,
    device: torch.device = None,
    verbose: bool = True,
    resume_from: str = None,
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """Full training loop.

    Args:
        model: Self-pruning network.
        train_loader: Training data loader.
        test_loader: Test data loader.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        lambda_sparse: Base sparsity coefficient (λ).
        lambda_schedule: "constant" or "dynamic" (linear ramp-up).
        lambda_max: Maximum λ value for dynamic schedule.
        device: Device to train on (auto-detected if None).
        verbose: Whether to print epoch summaries.

    Returns:
        Dict with history (per-epoch metrics) and final_metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    if resume_from and os.path.exists(resume_from):
        if verbose:
            logger.info(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "sparsity": [],
        "lambda_value": [],
        "ce_loss": [],
        "sparse_loss": [],
    }

    for epoch in range(start_epoch, epochs + 1):
        # Compute current lambda
        if lambda_schedule == "dynamic":
            # Linear ramp-up from 0 to lambda_max
            current_lambda = lambda_max * (epoch / epochs)
        else:
            current_lambda = lambda_sparse

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, current_lambda, device)

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)

        # Sparsity
        sparsity_stats = compute_sparsity_stats(model)
        sparsity = sparsity_stats["overall_sparsity"]

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["avg_loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["test_accuracy"].append(test_metrics["accuracy"])
        history["sparsity"].append(sparsity)
        history["lambda_value"].append(current_lambda)
        history["ce_loss"].append(train_metrics["avg_ce_loss"])
        history["sparse_loss"].append(train_metrics["avg_sparse_loss"])

        if verbose:
            logger.info(
                f"Epoch {epoch:>3d}/{epochs} | "
                f"Loss: {train_metrics['avg_loss']:.4f} "
                f"(CE: {train_metrics['avg_ce_loss']:.4f}, "
                f"Sparse: {current_lambda:.1e}x{train_metrics['avg_sparse_loss']:.1f}) | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Test Acc: {test_metrics['accuracy']:.2f}% | "
                f"Sparsity: {sparsity:.2%}"
            )
            
        # Model Checkpointing
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lambda_value': current_lambda,
            }, checkpoint_path)

    final_metrics = {
        "final_test_accuracy": test_metrics["accuracy"],
        "final_sparsity": sparsity,
        "final_train_loss": train_metrics["avg_loss"],
        "lambda_value": current_lambda,
        "lambda_schedule": lambda_schedule,
    }

    return {"history": history, "final_metrics": final_metrics}
