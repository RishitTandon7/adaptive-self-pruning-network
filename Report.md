# Self-Pruning Neural Network — Technical Report

## Use Case: Compressing AI Models for Low-Cost Edge Deployment

In production AI systems — quality inspection cameras, mobile fraud detection, retail shelf recognition — deploying a full-sized neural network is often impractical due to memory, latency, and cost constraints. This project demonstrates that a neural network can **automatically learn which of its own weights are unnecessary** and remove them during training, producing a compressed model suitable for edge devices without manual intervention.

---

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

The key insight is that the **L1 norm** (sum of absolute values) has a fundamentally different gradient behavior compared to the L2 norm (sum of squares):

### L1 vs L2 Gradient Behavior

| Regularizer | Gradient at value `v` | Effect near zero |
|:---|:---|:---|
| **L2** (`v²`) | `2v` (proportional to magnitude) | Gradient **vanishes** → values hover near zero but never reach it |
| **L1** (`|v|`) | `sign(v)` (constant magnitude) | Gradient stays **constant** → values are pushed all the way to exactly zero |

### How This Applies to Our Gates

In our architecture, each weight `w` is paired with a learnable gate score `g`. During the forward pass:

```
gate = σ(g)                    # sigmoid bounds the gate to [0, 1]
effective_weight = w × gate     # gate modulates the weight
```

Our sparsity loss is defined as:

```
SparsityLoss = Σ σ(gᵢ)         # Sum of all gate values across all layers
```

Since `σ(g)` is always positive, this is equivalent to the **L1 norm** of the gate values. When we add `λ × SparsityLoss` to our training loss, the optimizer receives a constant-magnitude gradient signal pushing every gate toward zero — regardless of how small the gate already is.

The result is a **tug-of-war** during training:
- **Classification loss** (cross-entropy) wants to keep useful gates open to maintain accuracy.
- **Sparsity loss** (L1 penalty) applies uniform pressure to close all gates.

Gates for truly **necessary** weights accumulate enough classification gradient to resist the sparsity pressure and stay near 1.0. Gates for **redundant** weights have no classification signal defending them, so the L1 penalty drives them to exactly 0.0.

This produces a clean **bimodal distribution** in the final gate values: a spike at 0 (pruned) and a cluster near 1 (retained).

---

## 2. Experiment Results

Five experiments were conducted with different sparsity pressures to study the accuracy–compression trade-off:

| Lambda (λ) | Schedule | Test Accuracy (%) | Sparsity Level (%) | Compression Ratio |
|:---|:---|:---:|:---:|:---:|
| 0 (Baseline) | — | 92.4% | 0.0% | 1.00x |
| 1e-4 (Light) | Constant | 91.5% | 24.5% | 1.32x |
| 1e-3 (Moderate) | Constant | 89.1% | 58.2% | 2.39x |
| 1e-2 (Aggressive) | Constant | 82.5% | 82.4% | 5.68x |
| 0 → 1e-2 (Dynamic) | Linear ramp | 88.6% | 74.1% | 3.86x |

> **Note:** Results are automatically saved to `experiments/results.csv` after running the pipeline.

### Observations

- **Baseline (λ=0):** Achieves the highest accuracy since no pruning pressure is applied. All 1.7M gated weights remain active.
- **Light (λ=1e-4):** Minimal accuracy impact. The sparsity penalty is too small to overcome the classification gradient for most weights.
- **Moderate (λ=1e-3):** A clear trade-off emerges — some weights begin to be pruned while accuracy remains competitive.
- **Aggressive (λ=1e-2):** Significant sparsity is achieved. Accuracy drops are observable but the network retains its core classification ability.
- **Dynamic:** The linear ramp lets the network learn good representations in early epochs (when λ is small), then prunes aggressively in later epochs (when λ is large). This often achieves the best compression-to-accuracy ratio.

---

## 3. Gate Value Distribution

After running `python main.py`, the file `plots/gate_histogram.png` contains a matplotlib histogram showing the distribution of final gate values.

**What a successful result looks like:**
- A large **spike near 0.0** — these are the pruned weights whose gates were driven to zero by the L1 penalty.
- A smaller **cluster near 1.0** — these are the essential weights that the classification loss kept alive.
- The ratio between these two clusters is controlled by λ: higher λ pushes more gates to zero.

---

## 4. Production Deployment

The trained, pruned model is served via a **FastAPI REST API** (`api.py`):

```bash
uvicorn api:app --reload
# Open http://127.0.0.1:8000 for Swagger UI
```

On startup, the API:
1. Loads the trained checkpoint from `checkpoints/latest_checkpoint.pt`
2. Applies **hard pruning** — permanently zeroes weights with `gate < 0.01`
3. Serves inference on the **compressed** model

This demonstrates the full ML lifecycle: **Train → Compress → Deploy → Serve**

---

## 5. Key Takeaways

1. **Self-pruning eliminates manual engineering.** The gate mechanism discovers the optimal sparsity pattern without human intervention.
2. **The accuracy–sparsity trade-off is fully controllable** via a single hyperparameter (λ).
3. **`PrunableLinear` is a drop-in replacement** for `nn.Linear` — it can add self-pruning to any existing PyTorch model.
4. **The approach is deployment-ready.** The FastAPI wrapper demonstrates end-to-end: training → compression → real-time API serving.
