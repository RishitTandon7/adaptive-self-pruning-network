# 🧠 Adaptive Self-Pruning Neural Network

> **Use Case: Compressing AI Models for Low-Cost, Real-Time Edge Deployment**

---

## 🎯 The Problem

Deploying deep learning models in production is **expensive**. A standard image classifier for quality inspection on a factory floor, fraud detection on a mobile banking app, or object recognition on a retail shelf camera has millions of parameters. This means:

- **High cloud inference costs** — Every API call consumes GPU/CPU cycles at scale.
- **Impossible edge deployment** — IoT devices, smartphones, and embedded sensors have strict memory and latency budgets (often < 10MB RAM, < 50ms inference).
- **Wasted compute** — Research shows that up to **90% of weights in a neural network are redundant** and contribute nothing to the final prediction.

**The question:** *Can a neural network automatically learn which of its own weights are useless — and remove them — during training itself?*

---

## 💡 The Solution

This project implements a **Self-Pruning Neural Network** — a model that learns to compress itself during training by automatically identifying and removing unnecessary weights.

### How It Works

Instead of manually deciding which weights to remove (traditional pruning), we attach a **learnable gate** to every single weight in the network:

```
Effective Weight = Weight × sigmoid(Gate Score)
```

- If the gate score is **high** → sigmoid ≈ 1.0 → the weight is kept.
- If the gate score is **low** → sigmoid ≈ 0.0 → the weight is effectively removed.

We add a **sparsity penalty** (L1 regularization on gate values) to the training loss:

```
Total Loss = Classification Loss + λ × Σ sigmoid(gate_scores)
```

This creates a tug-of-war:
- The **classification loss** wants all weights active to maximize accuracy.
- The **sparsity loss** wants all gates closed to minimize the penalty.

The result? **Only the weights the network truly needs survive training.** Everything else is pruned away automatically.

---

## 🏭 Real-World Use Case: Edge AI for Visual Inspection

**Scenario:** A manufacturing company deploys cameras on assembly lines to detect defective products in real-time.

| Challenge | Without Pruning | With Self-Pruning |
|---|---|---|
| Model Size | ~13 MB | **< 5 MB** |
| Parameters | 3.4M (all active) | **< 1M active** |
| Inference Latency | ~15ms (GPU) | **~5ms (CPU)** |
| Deployment Target | Cloud GPU ($$$) | **Edge device ($)** |
| Monthly Cloud Cost | ~$500/camera | **$0 (runs locally)** |

This project proves that **a network can be compressed 2-5x** while maintaining competitive accuracy — making it small enough to run directly on a $35 Raspberry Pi instead of a $500/month cloud GPU.

---

## 🏗️ Architecture

```
Input Image (3×32×32)
       │
       ▼
   [Flatten] → 3072
       │
       ▼
 ┌─────────────┐
 │ PrunableLinear│ 3072 → 512  (each weight has a learnable gate)
 │   + ReLU     │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │ PrunableLinear│ 512 → 256
 │   + ReLU     │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │ PrunableLinear│ 256 → 10
 └──────┬──────┘
        │
        ▼
   Output (10 classes)
```

**Key Innovation:** The `PrunableLinear` layer (`models/prunable_layer.py`) is a drop-in replacement for `nn.Linear` that can be used in **any** PyTorch model to add self-pruning capability.

---

## 📊 Experiments

We sweep across 5 different sparsity pressures (λ values) to study the accuracy–compression trade-off:

| Experiment | Lambda (λ) | Schedule | Purpose |
|---|---|---|---|
| Baseline | 0 | — | No pruning (upper bound on accuracy) |
| Light | 1e-4 | Constant | Minimal pruning pressure |
| Moderate | 1e-3 | Constant | Balanced trade-off |
| Aggressive | 1e-2 | Constant | Maximum compression |
| Dynamic | 0 → 5e-2 | Linear ramp | Train first, prune later |

After training, **hard pruning** zeroes out all weights with gate < 0.01, and we measure:
- Accuracy drop (before vs. after pruning)
- Compression ratio (total params / active params)
- Model size reduction

---

## 🚀 FastAPI Deployment

The project includes a production-ready **REST API** (`api.py`) that serves the pruned model for real-time inference:

```bash
uvicorn api:app --reload
```

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload an image → get CIFAR-10 class + compression stats |
| `GET` | `/health` | Health check for load balancers |
| `GET` | `/` | Redirects to interactive Swagger UI |

**Example Response:**
```json
{
  "prediction": "airplane",
  "class_id": 0,
  "model_efficiency": {
    "total_parameters": 3413770,
    "active_parameters": 1205430,
    "compression_ratio": 2.83
  }
}
```

The API automatically loads the trained checkpoint on startup, applies hard pruning, and serves inference using the compressed model.

---

## 📁 Project Structure

```
self-pruning-network/
├── models/
│   ├── prunable_layer.py    # Custom PrunableLinear layer with gate scores
│   └── network.py           # SelfPruningNetwork architecture
├── training/
│   └── train.py             # Training loop with sparsity loss + checkpointing
├── experiments/
│   └── runner.py            # Automated experiment runner (5 λ configs)
├── utils/
│   ├── data.py              # CIFAR-10 data loading with augmentation
│   ├── sparsity.py          # Sparsity computation utilities
│   ├── visualize.py         # Matplotlib plotting functions
│   └── logger.py            # Standardized logging setup
├── tests/
│   └── test_model.py        # Pytest unit tests
├── plots/                   # Auto-generated visualizations
├── api.py                   # FastAPI deployment server
├── main.py                  # Full pipeline entry point
├── config.yaml              # Centralized hyperparameter config
├── Report.md                # Technical report with L1 analysis
└── requirements.txt         # Pinned dependencies
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full training + experiment pipeline
python main.py

# 3. Run unit tests
pytest tests/

# 4. Deploy the API
uvicorn api:app --reload
# Open http://127.0.0.1:8000 in your browser
```

---

## 🔑 Key Insights

1. **Self-pruning works.** The network successfully learns to shut off unnecessary gates when sparsity pressure (λ) is applied.
2. **The trade-off is real.** Higher λ → more compression, but accuracy degrades. The "sweet spot" is λ = 1e-3 (moderate pruning).
3. **Dynamic scheduling is powerful.** Ramping λ from 0 → max lets the network learn good representations first, then prune — often achieving the best compression-to-accuracy ratio.
4. **Hard pruning is nearly lossless.** After soft-pruning during training, zeroing out dead weights causes minimal additional accuracy loss.

---

## 🛠️ Tech Stack

- **PyTorch** — Custom autograd layers, model training
- **FastAPI** — Production REST API for model serving
- **Matplotlib** — Experiment visualizations
- **Pytest** — Automated unit testing
- **YAML** — Configuration management

---

## 👤 Author

**Rishit Tandon**
Built as a case study demonstrating neural network compression for efficient edge deployment.