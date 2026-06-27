# Phase 2 - Baseline CNN

← [Phase 1](../phase1_naive/README.md) | [Back](../README.md) | [Phase 3 ->](../phase3_unet/README.md)

> **What this phase shows:** Translating domain knowledge into a model design decision -
> instead of summarizing frames analytically, preserve all 128 as separate input channels
> and let the network learn how to combine them. First end-to-end trained model in the pipeline.

All 128 SPC frames are stacked into 384 input channels and passed through a flat 8-layer
CNN at full 800×800 resolution. No downsampling, no skip connections. The result is a
**+13.15 dB PSNR jump over Phase 1**, confirming that a learned prior dramatically
outperforms any fixed accumulation rule.

---

## Input Representation

Rather than summarizing frames (as Phase 1 did by summing), all 128 are preserved and
concatenated. Each frame contributes 3 channels (RGB), giving 128 × 3 = 384 input
channels. The network learns directly which temporal and spatial patterns correspond to
real scene content versus photon noise.

```
.npy (1024, 800, 100, 3)
  -> slice last 128 frames    (128, 800, 100, 3)
  -> unpackbits axis=2        (128, 800, 800, 3)
  -> reshape + transpose      (384, 800, 800)     ← CNN input
```

---

## Architecture - `SimCNN`

8 convolutional layers, all 3×3 with padding=1 (spatial resolution preserved throughout).
The channel count expands to build features, then contracts back to the 3-channel output.

```
(384, 800, 800) -> Conv+ReLU ×7 -> Conv+Sigmoid -> (3, 800, 800)
Channel progression: 384->128->256->256->128->64->32->16->3
```

~1.72M parameters. Sigmoid output constrains predictions to [0, 1] without clamping.

---

## Loss & Training

```
Loss = 0.5 × L1  +  0.5 × (1 − SSIM)
```

L1 alone produces blurry outputs. Adding SSIM supervision directly optimizes for structural
similarity, encouraging the model to preserve edges and object boundaries rather than
just minimizing average pixel error. Adam optimizer, lr=1e-4, 50 epochs, sample-by-sample
(batch size 1). 45 train / 5 test samples.

---

## Code

**`cnn_reconstruction.py`**

| Function / Class | What it does |
|-----------------|-------------|
| `unpack_last_frame(npy_path)` | Loads SPC data, unpacks bits, reshapes to (800, 800, 384), normalizes |
| `SimCNN` | The model - 8 conv layers, expand then contract, Sigmoid output |
| `save_comparison(...)` | 3-panel figure: Input \| Model Output \| Ground Truth |
| `save_loss_curve(epoch_losses)` | Plots loss vs epoch |
| `print_summary(results)` | Prints per-sample and average PSNR/SSIM to terminal |

---

## Running

```bash
pip install torch torchvision torchmetrics scikit-image matplotlib Pillow
python cnn_reconstruction.py
```

Update `DATA_ROOT` at the top. Model saved to `results/cnn_model.pth`.

---

## Results

### Common evaluation scenes (000015, 000023, 000030)

| Scene | PSNR ↑ | SSIM ↑ |
|:-----:|:------:|:------:|
| 000015 | 20.92 dB | 0.5964 |
| 000023 | 27.59 dB | 0.8974 |
| 000030 | 30.91 dB | 0.8962 |
| **Avg** | **26.47 dB** | **0.7967** |

**vs Phase 1:** +13.15 dB PSNR, +0.518 SSIM

| 000015 | 000023 | 000030 |
|:------:|:------:|:------:|
| ![](results/new_samples/cnn_000015.png) | ![](results/new_samples/cnn_000023.png) | ![](results/new_samples/cnn_000030.png) |

---

### Original test scenes (sunny-room, tv-couch, ultramodern, white-room, wooden-staircase)

| Sample | Scene | PSNR ↑ | SSIM ↑ |
|:------:|-------|:------:|:------:|
| 0 | sunny-room | 27.21 dB | 0.8922 |
| 1 | tv-couch | 32.09 dB | 0.8851 |
| 2 | ultramodern | 27.57 dB | 0.8202 |
| 3 | white-room | 31.20 dB | 0.9102 |
| 4 | wooden-staircase | 29.60 dB | 0.8442 |
| **Avg** | | **29.54 dB** | **0.8704** |

![Loss curve](results/loss_curve.png)

| sunny-room | tv-couch | ultramodern |
|:----------:|:--------:|:-----------:|
| ![](results/comparison_sample0_sunny-room.png) | ![](results/comparison_sample1_tv-couch.png) | ![](results/comparison_sample2_ultramodern.png) |

| white-room | wooden-staircase |
|:----------:|:----------------:|
| ![](results/comparison_sample3_white-room.png) | ![](results/comparison_sample4_wooden-staircase.png) |

---

## Key Findings

**Learning to combine frames dramatically outperforms any fixed rule.** The +13.15 dB
gain over Phase 1 is the largest single jump in the project, showing that the core design
decision - treating all 128 frames as separate input channels - is correct.

**The model hasn't fully converged.** The loss curve is still declining at epoch 50,
meaning these numbers are a lower bound on what this architecture can actually achieve
with more training time.

**Scene complexity exposes the architecture's limits.** PSNR ranges from 20.92 dB
(000015, a dark challenging scene) to 30.91 dB (000030, a well-lit structured scene).
A flat CNN applies the same spatial operation everywhere with no mechanism for multi-scale
reasoning - it cannot adapt to local complexity. This is precisely the problem that
UNet's encoder-decoder structure and skip connections address in Phase 3.

---

← [Phase 1](../phase1_naive/README.md) | [Back](../README.md) | [Phase 3 ->](../phase3_unet/README.md)