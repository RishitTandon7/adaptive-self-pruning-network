"""
Main entry point for the Self-Pruning Neural Network project.

Orchestrates:
    1. Data loading (CIFAR-10)
    2. Running all experiments (5 λ configurations)
    3. Generating visualizations
    4. Printing final summary table
"""

import os
import sys
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import run_all_experiments
from utils.visualize import generate_all_plots
from utils.logger import setup_logger
import yaml

logger = setup_logger("main")


def print_summary_table(results: list):
    """Print a formatted results table to the console."""
    logger.info("\n" + "=" * 95)
    logger.info("  EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 95)
    header = (
        f"{'Experiment':<25} | {'Lambda':>8} | {'Test Acc':>8} | {'Sparsity':>8} | "
        f"{'Post-Prune':>10} | {'Compression':>11}"
    )
    logger.info(header)
    logger.info("-" * 95)

    for r in results:
        lam = f"{r['lambda']:.0e}" if r['lambda'] > 0 else "0"
        if r['lambda_schedule'] == 'dynamic':
            lam = "dynamic"
        logger.info(
            f"{r['name']:<25} | {lam:>8} | "
            f"{r['pre_prune_accuracy']:>7.2f}% | "
            f"{r['pre_prune_sparsity']*100:>7.2f}% | "
            f"{r['post_prune_accuracy']:>9.2f}% | "
            f"{r['compression_ratio']:>10.2f}x"
        )

    logger.info("=" * 95)


def main():
    """Run the full self-pruning network pipeline."""
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    epochs = config['training'].get('epochs', 15)

    logger.info(">> Self-Pruning Neural Network - Full Pipeline")
    logger.info(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"   PyTorch: {torch.__version__}")

    # Run all experiments (Steps 5-9)
    results = run_all_experiments(
        output_dir="experiments",
        epochs=epochs,
    )

    # Generate all visualizations (Step 10)
    generate_all_plots(results, save_dir="plots")

    # Print summary table
    print_summary_table(results)

    logger.info("\n[DONE] Pipeline complete! Check:")
    logger.info("   > experiments/results.csv  - raw results")
    logger.info("   > plots/                   - all visualizations")


if __name__ == "__main__":
    main()
