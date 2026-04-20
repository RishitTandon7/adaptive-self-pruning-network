# Self-Pruning Neural Network Report

## 1. Explanation of Sparsity Loss (L1 Penalty on Sigmoid Gates)

The objective of our custom sparsity mechanism is to dynamically prune the network during training. We associate each weight with a learnable `gate_score`. During the forward pass, these scores are passed through a Sigmoid function to generate a `gate` value strictly bounded between 0 and 1. The final "effective weight" used by the layer is `weight * gate`.

We apply an **L1 penalty** on these post-sigmoid `gate` values. The L1 norm (the sum of absolute values) is a well-known sparsity-inducing regularizer because its gradient with respect to the loss is constant (regardless of the magnitude of the value), pushing the parameters linearly towards zero until they exactly hit zero. Because our gates are already strictly positive (due to the Sigmoid), the L1 norm simplifies to just the sum of the gate values.

By adding `λ * sum(gates)` to our total loss, the optimizer is constantly penalized for keeping gates "open". The network must learn to balance the cross-entropy classification loss (which wants all weights available to maximize accuracy) and the sparsity loss (which wants all gates closed to minimize the penalty). As a result, only the weights strictly necessary for solving the classification task remain active, while the rest are successfully driven to near zero (pruned).

## 2. Experiment Results Table

*(Note: Below is the structural representation of the expected output. Actual values will populate after running `main.py`)*

| Experiment | Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
| :--- | :--- | :--- | :--- |
| Baseline | 0 | - | - |
| Light Pruning | 1e-4 | - | - |
| Moderate Pruning | 1e-3 | - | - |
| Aggressive Pruning | 1e-2 | - | - |
| Dynamic L | 0 → 5e-2 | - | - |

## 3. Gate Value Distributions

*(Note: The `plots/` directory automatically generates histograms showing the distribution of final gate values upon running `main.py`. A successful result will demonstrate a distinct spike near 0.0 (pruned weights) and another cluster near 1.0 (retained weights). Please refer to `plots/gate_distributions.png` after execution.)*
