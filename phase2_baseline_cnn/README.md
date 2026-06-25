# Phase 2 — Baseline CNN

← [Phase 1](../phase1_naive/README.md) | [Back](../README.md) | [Phase 3 →](../phase3_unet/README.md)

First trained model. All 128 SPC frames are stacked into 384 input channels and passed
through a flat 8-layer CNN at full resolution. No downsampling, no skip connections.
This establishes a learned baseline for Phase 3 to improve on.

---

## Input Representation

Rather than summarizing frames (as Phase 1 did by summing), we preserve all 128 and let
the network learn how to combine them. Each frame contributes 3 channels (RGB), giving
128 × 3 = 384 input channels.

```
.npy (1024, 800, 100, 3)
  → slice last 128 frames    (128, 800, 100, 3)
  → unpackbits axis=2        (128, 800, 800, 3)
  → reshape + transpose      (384, 800, 800)     ← CNN input
```

---

## Architecture — `SimCNN`

8 convolutional layers, all 3×3 with padding=1 (spatial resolution preserved throughout).

```
(384, 800, 800) → Conv+ReLU ×7 → Conv+Sigmoid → (3, 800, 800)
Channel progression: 384→128→256→256→128→64→32→16→3
```

~1.72M parameters.

---

## Loss & Training

```
Loss = 0.5 × L1  +  0.5 × (1 − SSIM)
```

L1 alone produces blurry outputs — SSIM adds structural supervision. Adam, lr=1e-4,
50 epochs, sample-by-sample (batch size 1). 45 train / 5 test samples.

---

## Code

**`cnn_reconstruction.py`**

| Function | What it does |
|----------|-------------|
| `load_input(npy_path)` | Loads SPC data, unpacks bits, reshapes to (384, 800, 800), normalizes |
| `load_target(png_path)` | Loads ground truth PNG as (3, 800, 800) float in [0, 1] |
| `SimCNN` | The model — 8 conv layers, expand then contract, Sigmoid output |
| `train(...)` | Training loop — combined loss, backprop, epoch loss history |
| `evaluate(...)` | Inference on test set, scikit-image PSNR/SSIM, saves comparison figures |
| `save_loss_curve(epoch_losses)` | Plots loss vs epoch |

---

## Running

```bash
pip install torch torchvision torchmetrics scikit-image matplotlib Pillow
python cnn_reconstruction.py
```

Update `DATA_ROOT` at the top. Outputs saved to `results/`.

---

## Results

| Sample | Scene | PSNR ↑ | SSIM ↑ |
|:------:|-------|:------:|:------:|
| 0 | sunny-room | 26.70 dB | 0.8951 |
| 1 | tv-couch | 31.98 dB | 0.8848 |
| 2 | ultramodern | 26.47 dB | 0.8207 |
| 3 | white-room | 30.53 dB | 0.9120 |
| 4 | wooden-staircase | 28.81 dB | 0.8382 |
| **Avg** | | **28.90 dB** | **0.8701** |

**vs Phase 1:** +7.35 dB PSNR, +0.42 SSIM

---

## Visual Results

![Loss curve](results/loss_curve.png)

| Scene | Comparison |
|-------|-----------|
| sunny-room | ![](results/comparison_sample0_sunny-room.png) |
| tv-couch | ![](results/comparison_sample1_tv-couch.png) |
| ultramodern | ![](results/comparison_sample2_ultramodern.png) |
| white-room | ![](results/comparison_sample3_white-room.png) |
| wooden-staircase | ![](results/comparison_sample4_wooden-staircase.png) |

---

## Observations

The jump from Phase 1 is large (+7.35 dB) — preserving all 128 frames as separate channels
and learning to combine them far outperforms any fixed summation rule.

The loss curve is still declining at epoch 50, meaning the model hasn't fully converged.
These numbers are a lower bound on what this architecture can achieve.

PSNR varies 5.5 dB across scenes (26.47 to 31.98) while SSIM stays above 0.82 everywhere.
The network recovers structure well but struggles with precise per-pixel intensity on
complex scenes — exactly what UNet's skip connections are designed to fix.

---

← [Phase 1](../phase1_naive/README.md) | [Back](../README.md) | [Phase 3 →](../phase3_unet/README.md)
