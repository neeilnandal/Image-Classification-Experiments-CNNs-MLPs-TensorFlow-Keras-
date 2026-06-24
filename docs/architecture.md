
# Architecture Documentation

This document describes the model architectures used across both tasks in detail — useful as a reference independent of the code.

---

## Task 1: Image Classification (MLP & CNN)

### MLP — Fashion-MNIST

Input images (28×28 grayscale) are flattened into a 784-dim vector before entering the network.

```
Input (784)
   │
   ▼
Dense(128) ── ReLU
   │
   ▼
Dense(64) ── ReLU
   │
   ▼
Dense(10) ── Softmax
```

| Layer | Units | Activation |
|---|---:|---|
| Input | 784 | — |
| Dense | 128 | ReLU |
| Dense | 64 | ReLU |
| Dense (output) | 10 | Softmax |

No spatial structure is preserved — every pixel is treated as an independent feature.

### CNN — Fashion-MNIST

```
Input (28×28×1)
   │
   ▼
Conv2D(32, 3×3) ── ReLU
   │
   ▼
MaxPool2D(2×2)
   │
   ▼
Conv2D(64, 3×3) ── ReLU
   │
   ▼
MaxPool2D(2×2)
   │
   ▼
Flatten
   │
   ▼
Dense(64) ── ReLU
   │
   ▼
Dense(10) ── Softmax
```

### CNN — CIFAR-10

Deeper than the Fashion-MNIST CNN to handle 32×32 RGB inputs and 3 color channels.

```
Input (32×32×3)
   │
   ▼
Conv2D(32, 3×3) ── ReLU
   │
   ▼
MaxPool2D(2×2)
   │
   ▼
Conv2D(64, 3×3) ── ReLU
   │
   ▼
MaxPool2D(2×2)
   │
   ▼
Conv2D(128, 3×3) ── ReLU
   │
   ▼
Flatten
   │
   ▼
Dense(128) ── ReLU
   │
   ▼
Dense(10) ── Softmax
```

**Design rationale:** Convolution + pooling layers extract local, translation-invariant spatial features before any fully-connected layer compresses them into class scores — the key structural advantage CNNs have over MLPs for image data.

---

## Task 2: Clock Time Prediction (Shared CNN Backbone)

All five label-encoding variants share the same convolutional feature extractor at 75×75 resolution; only the output head(s) and loss differ.

### Shared Backbone (75×75)

```
Input (75×75×1)
   │
   ▼
[Conv2D(32, 3×3, He-init) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)]
   │
   ▼
[Conv2D(64, 3×3, He-init) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)]
   │
   ▼
[Conv2D(64, 3×3, He-init) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)]
   │
   ▼
Conv2D(128, 3×3, He-init) → BatchNorm → ReLU
   │
   ▼
GlobalAveragePooling2D
   │
   ▼
Dense(128) ── ReLU
   │
   ▼
Dropout(0.5)
   │
   ▼
  [Head — varies by encoding, see below]
```

Filter progression: `base → 2×base → 2×base → 4×base`, with `base = 32`.

### Output Heads by Encoding

**A. Multi-Head (Unscaled)**
```
shared features ──┬── Dense(12, Softmax)  → hour
                   └── Dense(1, Linear)    → minute (raw 0–59)
```
Loss: `sparse_categorical_crossentropy` (hour) + `MSE` (minute), weights `{hour: 1.0, minute: 0.2}`

**B. Multi-Head (Scaled)** — best performer
```
shared features ──┬── Dense(12, Softmax) → hour
                   └── Dense(1, Sigmoid)  → minute (scaled to [0,1], ×60 at inference)
```

**C. Sine–Cosine**
```
shared features ── Dense(4, Tanh) → [cos_hour, sin_hour, cos_minute, sin_minute]
```
Decoded via `atan2` back to hour/minute angles, then rounded/clipped.

**D. Bucketed Classification (30 min / 15 min)**
```
shared features ── Dense(K, Softmax)   # K = 24 (30min) or 48 (15min)
```
Predicted bucket's center time is used as the decoded output.

### 150×150 Scaled-Up Model (Multi-Head Scaled only)

```
Input (150×150×1)
   │
   ▼
GaussianNoise(0.02)
   │
   ▼
RandomContrast(0.1)
   │
   ▼
[Conv2D(64, 3×3, He-init) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.30)]
   │
   ▼
[Conv2D(128, 3×3, He-init) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.30)]
   │
   ▼
[Conv2D(128, 3×3, He-init) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.30)]
   │
   ▼
Conv2D(256, 3×3, He-init) → BatchNorm → ReLU
   │
   ▼
GlobalAveragePooling2D
   │
   ▼
Dense(256) ── ReLU
   │
   ▼
Dropout(0.30)
   │
   ▼
  ┌── Dense(12, Softmax) → hour
  └── Dense(1, Sigmoid)  → minute
```

Optimizer: Adam(5e-4), equal loss weights `{hour: 1.0, minute: 1.0}`, with `ReduceLROnPlateau` and early stopping.

**Note:** This deeper model regressed to near-random (half-cycle) errors despite higher capacity — see `task2_clock_time_prediction/README.md` for the suspected causes (loss-weight change, augmentation-at-inference, or data/label mismatch).

---

## Cross-Task Design Notes

- **He initialization** is used throughout Task 2's backbone since all conv layers use ReLU activations, for which He init is the standard choice to avoid vanishing/exploding gradients early in training.
- **BatchNorm before ReLU** is consistently applied in Task 2 but absent in Task 1 — a difference worth noting if extending Task 1's CNNs further.
- **Global Average Pooling** (Task 2) vs. **Flatten** (Task 1) before the dense head: GAP is more parameter-efficient and slightly more robust to spatial translation, at the cost of discarding fine positional detail — a reasonable tradeoff for the clock-reading task where global features (hand angles) matter more than precise pixel position.
