"""
Experiment runner: executes multiple training runs across different λ values
and stores results in a CSV file.
"""

import os
import csv
import torch
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network import SelfPruningNetwork
from training.train import train_model, evaluate
from utils.data import get_cifar10_loaders
from utils.sparsity import compute_sparsity_stats, log_sparsity
from utils.logger import setup_logger

logger = setup_logger("experiment_runner")


def run_experiment(
    experiment_name: str,
    lambda_sparse: float,
    train_loader,
    test_loader,
    lambda_schedule: str = "constant",
    lambda_max: float = 1e-2,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = None,
    prune_threshold: float = 0.01,
) -> dict:
    """Run a single training experiment.

    Args:
        experiment_name: Human-readable name for this experiment.
        lambda_sparse: Sparsity coefficient λ.
        train_loader: Pre-loaded training data loader.
        test_loader: Pre-loaded test data loader.
        lambda_schedule: "constant" or "dynamic".
        lambda_max: Max λ for dynamic schedule.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Compute device.
        prune_threshold: Gate threshold for sparsity/pruning.

    Returns:
        Dict with experiment results including before/after pruning metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"\n{'='*70}")
    logger.info(f"  EXPERIMENT: {experiment_name}")
    logger.info(f"  lambda={lambda_sparse}, schedule={lambda_schedule}, epochs={epochs}")
    logger.info(f"  Device: {device}")
    logger.info(f"{'='*70}\n")

    # Model
    model = SelfPruningNetwork()

    # Train
    results = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        lambda_sparse=lambda_sparse,
        lambda_schedule=lambda_schedule,
        lambda_max=lambda_max,
        device=device,
    )

    # Pre-pruning metrics
    pre_prune_acc = results["final_metrics"]["final_test_accuracy"]
    pre_prune_sparsity = results["final_metrics"]["final_sparsity"]
    pre_prune_params = model.count_parameters()

    log_sparsity(model, epochs)

    # Hard pruning
    logger.info("Applying hard pruning...")
    prune_results = model.hard_prune_all(threshold=prune_threshold)
    for layer_name, count in prune_results.items():
        logger.info(f"  {layer_name}: {count:,} weights pruned")

    # Post-pruning evaluation
    post_prune_eval = evaluate(model, test_loader, device)
    post_prune_acc = post_prune_eval["accuracy"]
    post_prune_params = model.count_parameters()

    logger.info(f"\n--- Pruning Summary ---")
    logger.info(f"  Accuracy: {pre_prune_acc:.2f}% -> {post_prune_acc:.2f}% "
          f"(delta={post_prune_acc - pre_prune_acc:+.2f}%)")
    logger.info(f"  Params:   {pre_prune_params['total']:,} total, "
          f"{post_prune_params['nonzero']:,} nonzero "
          f"({post_prune_params['compression']:.2f}x compression)")

    experiment_result = {
        "name": experiment_name,
        "lambda": lambda_sparse,
        "lambda_schedule": lambda_schedule,
        "epochs": epochs,
        "pre_prune_accuracy": pre_prune_acc,
        "pre_prune_sparsity": pre_prune_sparsity,
        "post_prune_accuracy": post_prune_acc,
        "total_params": pre_prune_params["total"],
        "nonzero_params": post_prune_params["nonzero"],
        "compression_ratio": post_prune_params["compression"],
        "history": results["history"],
        "model": model,
    }

    return experiment_result


def run_all_experiments(
    output_dir: str = "experiments",
    epochs: int = 15,
    device: torch.device = None,
) -> list:
    """Run the full suite of experiments defined in task.md.

    Experiments:
        1. λ = 0       (baseline — no pruning)
        2. λ = 1e-4    (light pruning)
        3. λ = 1e-3    (moderate pruning)
        4. λ = 1e-2    (aggressive pruning)
        5. Dynamic λ   (linearly increasing from 0 to 1e-2)

    Args:
        output_dir: Directory to save CSV results.
        epochs: Number of training epochs per experiment.
        device: Compute device.

    Returns:
        List of experiment result dicts.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data once and share across all experiments
    logger.info("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    logger.info("Dataset loaded.\n")

    experiments = [
        {"name": "Baseline (L=0)",      "lambda_sparse": 0.0,   "lambda_schedule": "constant"},
        {"name": "Light (L=1e-4)",      "lambda_sparse": 1e-4,  "lambda_schedule": "constant"},
        {"name": "Moderate (L=1e-3)",   "lambda_sparse": 1e-3,  "lambda_schedule": "constant"},
        {"name": "Aggressive (L=1e-2)", "lambda_sparse": 1e-2,  "lambda_schedule": "constant"},
        {"name": "Dynamic L",           "lambda_sparse": 0.0,   "lambda_schedule": "dynamic", "lambda_max": 1e-2},
    ]

    all_results = []
    for exp_config in experiments:
        result = run_experiment(
            experiment_name=exp_config["name"],
            lambda_sparse=exp_config["lambda_sparse"],
            train_loader=train_loader,
            test_loader=test_loader,
            lambda_schedule=exp_config.get("lambda_schedule", "constant"),
            lambda_max=exp_config.get("lambda_max", 1e-2),
            epochs=epochs,
            device=device,
        )
        all_results.append(result)

    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment", "Lambda", "Schedule", "Epochs",
            "Pre-Prune Accuracy (%)", "Sparsity (%)",
            "Post-Prune Accuracy (%)", "Total Params",
            "Nonzero Params", "Compression Ratio",
        ])
        for r in all_results:
            writer.writerow([
                r["name"],
                r["lambda"],
                r["lambda_schedule"],
                r["epochs"],
                f"{r['pre_prune_accuracy']:.2f}",
                f"{r['pre_prune_sparsity']*100:.2f}",
                f"{r['post_prune_accuracy']:.2f}",
                r["total_params"],
                r["nonzero_params"],
                f"{r['compression_ratio']:.2f}",
            ])

    logger.info(f"\n[DONE] Results saved to {csv_path}")
    return all_results


if __name__ == "__main__":
    results = run_all_experiments(epochs=10)
