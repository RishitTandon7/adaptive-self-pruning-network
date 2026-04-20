"""
Visualization utilities for the self-pruning neural network.

Generates all plots required by task.md:
    - Accuracy vs Sparsity
    - Lambda vs Accuracy
    - Lambda vs Sparsity
    - Gate value histogram
    - Layer-wise sparsity chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

# Style configuration
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.3,
    "figure.figsize": (10, 6),
    "font.size": 12,
})


def plot_accuracy_vs_sparsity(results: list, save_dir: str = "plots"):
    """Plot test accuracy vs sparsity for all experiments.

    Args:
        results: List of experiment result dicts.
        save_dir: Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [r["name"] for r in results]
    sparsities = [r["pre_prune_sparsity"] * 100 for r in results]
    accuracies = [r["pre_prune_accuracy"] for r in results]

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(results)))

    scatter = ax.scatter(sparsities, accuracies, c=colors, s=200, zorder=5,
                         edgecolors="white", linewidths=1.5)

    for i, name in enumerate(names):
        ax.annotate(name, (sparsities[i], accuracies[i]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9, color=colors[i], fontweight="bold")

    ax.set_xlabel("Sparsity (%)", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax.set_title("Accuracy vs Sparsity Trade-off", fontsize=16, fontweight="bold", color="#e94560")
    ax.grid(True, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_sparsity.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/accuracy_vs_sparsity.png")


def plot_lambda_vs_metrics(results: list, save_dir: str = "plots"):
    """Plot lambda vs accuracy and lambda vs sparsity (dual-axis).

    Args:
        results: List of experiment result dicts.
        save_dir: Directory to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Filter only constant-lambda experiments for a clean plot
    constant_exps = [r for r in results if r["lambda_schedule"] == "constant"]

    lambdas = [r["lambda"] for r in constant_exps]
    accuracies = [r["pre_prune_accuracy"] for r in constant_exps]
    sparsities = [r["pre_prune_sparsity"] * 100 for r in constant_exps]

    # Lambda labels (handle lambda=0 for log scale)
    lambda_labels = [f"{l:.0e}" if l > 0 else "0" for l in lambdas]

    # --- Lambda vs Accuracy ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(lambdas)), accuracies, "o-", color="#e94560",
            linewidth=2.5, markersize=10, markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels(lambda_labels)
    ax.set_xlabel("Lambda (λ)", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax.set_title("Lambda vs Accuracy", fontsize=16, fontweight="bold", color="#e94560")
    ax.grid(True, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lambda_vs_accuracy.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/lambda_vs_accuracy.png")

    # --- Lambda vs Sparsity ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(lambdas)), sparsities, "s-", color="#0f3460",
            linewidth=2.5, markersize=10, markeredgecolor="white", markeredgewidth=1.5)
    ax.fill_between(range(len(lambdas)), sparsities, alpha=0.3, color="#0f3460")
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels(lambda_labels)
    ax.set_xlabel("Lambda (λ)", fontsize=14)
    ax.set_ylabel("Sparsity (%)", fontsize=14)
    ax.set_title("Lambda vs Sparsity", fontsize=16, fontweight="bold", color="#e94560")
    ax.grid(True, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lambda_vs_sparsity.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/lambda_vs_sparsity.png")


def plot_gate_histogram(model, save_dir: str = "plots"):
    """Plot histogram of gate values across all prunable layers.

    Args:
        model: Trained SelfPruningNetwork.
        save_dir: Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    all_gates = []
    colors_list = ["#e94560", "#0f3460", "#533483"]

    for i, (name, module) in enumerate(model.named_modules()):
        from models.prunable_layer import PrunableLinear
        if isinstance(module, PrunableLinear):
            gates = module.get_gate_values().cpu().numpy().flatten()
            all_gates.append(gates)
            color = colors_list[i % len(colors_list)]
            ax.hist(gates, bins=50, alpha=0.6, label=name, color=color, edgecolor="none")

    ax.axvline(x=0.01, color="#e94560", linestyle="--", linewidth=2, label="Prune threshold (0.01)")
    ax.set_xlabel("Gate Value", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Distribution of Gate Values", fontsize=16, fontweight="bold", color="#e94560")
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#555")
    ax.grid(True, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gate_histogram.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/gate_histogram.png")


def plot_layer_sparsity(model, save_dir: str = "plots"):
    """Plot per-layer sparsity as a horizontal bar chart.

    Args:
        model: Trained SelfPruningNetwork.
        save_dir: Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    layer_names = []
    sparsities = []
    from models.prunable_layer import PrunableLinear
    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            layer_names.append(name)
            sparsities.append(module.get_sparsity() * 100)

    colors = plt.cm.plasma(np.linspace(0.3, 0.8, len(layer_names)))
    bars = ax.barh(layer_names, sparsities, color=colors, edgecolor="white", linewidth=0.8)

    for bar, sparsity in zip(bars, sparsities):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{sparsity:.1f}%", va="center", fontsize=11, fontweight="bold", color="#eee")

    ax.set_xlabel("Sparsity (%)", fontsize=14)
    ax.set_title("Layer-wise Sparsity", fontsize=16, fontweight="bold", color="#e94560")
    max_sp = max(sparsities) if sparsities else 0
    ax.set_xlim(0, max(max_sp * 1.2, 10))  # At least 10% range to avoid singular xlim
    ax.grid(True, linestyle="--", axis="x")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "layer_sparsity.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/layer_sparsity.png")


def plot_training_curves(results: list, save_dir: str = "plots"):
    """Plot training loss and test accuracy curves for all experiments.

    Args:
        results: List of experiment result dicts (each must contain 'history').
        save_dir: Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(results)))

    for i, r in enumerate(results):
        h = r["history"]
        ax1.plot(h["epoch"], h["train_loss"], "-", color=colors[i],
                 linewidth=2, label=r["name"])
        ax2.plot(h["epoch"], h["test_accuracy"], "-", color=colors[i],
                 linewidth=2, label=r["name"])

    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Training Loss", fontsize=13)
    ax1.set_title("Training Loss Curves", fontsize=15, fontweight="bold", color="#e94560")
    ax1.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#555")
    ax1.grid(True, linestyle="--")

    ax2.set_xlabel("Epoch", fontsize=13)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax2.set_title("Test Accuracy Curves", fontsize=15, fontweight="bold", color="#e94560")
    ax2.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#555")
    ax2.grid(True, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/training_curves.png")


def generate_all_plots(results: list, save_dir: str = "plots"):
    """Generate all visualizations.

    Args:
        results: List of experiment result dicts.
        save_dir: Directory to save all plots.
    """
    print("\n[PLOT] Generating plots...")
    plot_accuracy_vs_sparsity(results, save_dir)
    plot_lambda_vs_metrics(results, save_dir)
    plot_training_curves(results, save_dir)

    # Use the last experiment's model (Dynamic λ) for gate/sparsity plots
    # since it typically shows the most interesting pruning behavior
    best_sparse_model = max(results, key=lambda r: r["pre_prune_sparsity"])
    plot_gate_histogram(best_sparse_model["model"], save_dir)
    plot_layer_sparsity(best_sparse_model["model"], save_dir)

    print(f"\n[DONE] All plots saved to {save_dir}/")
