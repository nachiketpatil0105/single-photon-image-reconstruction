# Phase 3 — UNet

> Encoder-decoder with skip connections: recovering spatial detail that a flat CNN cannot.

← [Phase 2](../phase2_baseline_cnn/README.md) | [Back to main repo](../README.md) | [Phase 4 →](../phase4_resunet_attention/README.md)

---

## What This Phase Is About

Phase 2 showed that a flat CNN learns scene structure well — SSIM climbed to 0.87 — but
PSNR varied significantly across scenes (26.5 to 32.0 dB). The core limitation is that a
flat network processes everything at a single spatial resolution. To reconstruct a pixel at
position (x, y), it can only look at nearby pixels in the input. There is no mechanism to
reason about large-scale structure and fine texture simultaneously.

This phase introduces **UNet**: an encoder-decoder architecture where the encoder
progressively compresses spatial resolution to build a global scene understanding, and the
decoder reconstructs the image at full resolution. The critical addition is **skip
connections** — direct paths from each encoder level to the corresponding decoder level.
These allow the decoder to combine the bottleneck's global context with the encoder's
preserved spatial detail at every scale, which is exactly what single-photon reconstruction
needs to recover sharp edges and fine texture.

Everything else is also upgraded: the loss function, training schedule, data augmentation,
and evaluation protocol are all significantly improved over Phase 2.

---

## What Changed from Phase 2

| Component | Phase 2 | Phase 3 |
|-----------|---------|---------|
| Architecture | Flat CNN, 8 layers, single scale | UNet encoder-decoder, 3 scales + bottleneck |
| Skip connections | None | Yes — encoder features concatenated to decoder |
| Parameters | ~1.72M | ~31.2M |
| Loss | L1 + SSIM | Charbonnier + MS-SSIM + VGG Perceptual |
| Training schedule | Fixed LR 1e-4 | Cosine annealing (1e-4 → 1e-6) |
| Precision | FP32 | Mixed precision (FP16 autocast + GradScaler) |
| Gradient clipping | None | Max norm = 1.0 |
| Augmentation | None | Random flips, 90° rotations, 512×512 crop |
| Validation split | None (train/test only) | 80 / 10 / 10 train/val/test |
| Checkpointing | Last epoch | Best val SSIM |
| Epochs | 50 | 150 |

---

## Architecture

`UNetBasic` follows the classic encoder-decoder structure with three downsampling levels
and a bottleneck. Each encoder block is two 3×3 convolutions followed by MaxPool2d.
The decoder mirrors this, using learned ConvTranspose2d for upsampling, then
concatenating the corresponding encoder feature map before the conv block.

```
Input (384, 800, 800)
  ┌─────────────────────────────────────────────────────────────┐
  │ Encoder                                                     │
  │   enc1: 384→128  (2× Conv3×3 + ReLU)  → e1 (128,800,800)  │
  │   MaxPool2d ↓2                                              │
  │   enc2: 128→256  (2× Conv3×3 + ReLU)  → e2 (256,400,400)  │
  │   MaxPool2d ↓2                                              │
  │   enc3: 256→512  (2× Conv3×3 + ReLU)  → e3 (512,200,200)  │
  │   MaxPool2d ↓2                                              │
  └─────────────────────────────────────────────────────────────┘
  Bottleneck: 512→1024  (2× Conv3×3 + ReLU)    (1024,100,100)
  ┌─────────────────────────────────────────────────────────────┐
  │ Decoder  (skip connections shown as →→)                     │
  │   ConvTranspose2d ↑2  + cat(e3 →→) → dec1: 1024→512       │
  │   ConvTranspose2d ↑2  + cat(e2 →→) → dec2:  512→256       │
  │   ConvTranspose2d ↑2  + cat(e1 →→) → dec3:  256→128       │
  └─────────────────────────────────────────────────────────────┘
  Output: Conv1×1  128→3  →  (3, 800, 800)
```

Total trainable parameters: **~31.2 million**

---

## Loss Function

Phase 3 uses a three-component loss:

```
Loss = 0.25 × Charbonnier  +  0.50 × (1 − MS-SSIM)  +  0.25 × VGG Perceptual
```

**Charbonnier** replaces L1. It computes `sqrt((pred − target)² + ε²)` — smooth and
differentiable everywhere including at zero, which stabilises gradients at late training
when errors are very small.

**MS-SSIM** replaces single-scale SSIM. Evaluates structural similarity at multiple
scales simultaneously, which is more robust for indoor scenes that have both large planar
surfaces and fine texture detail at different frequencies.

**VGG Perceptual Loss** compares intermediate feature maps from a frozen pretrained
VGG16 at three layers: relu1_2 (fine textures), relu2_2 (mid-level patterns), relu3_3
(higher-level structures), weighted 0.2 / 0.3 / 0.5 so structural features contribute
most. This pushes the model toward perceptually sharp outputs rather than just
numerically close ones.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 → 1e-6 (cosine annealing) |
| Epochs | 150 |
| Batch Size | 1 |
| Loss | 0.25 × Charbonnier + 0.50 × (1 − MS-SSIM) + 0.25 × VGG |
| Gradient Clipping | Max norm = 1.0 |
| Mixed Precision | FP16 autocast + GradScaler |
| Train / Val / Test | 80% / 10% / 10% |
| Augmentation | H/V flip, 90° rotation, random 512×512 crop |
| Checkpointing | Best val SSIM (epoch 139) |

---

## Code Walkthrough

**`unet_reconstruction.py`** runs the complete pipeline — data loading, training,
validation, test evaluation, and saving all results — in one script.

**`unpack_spc(npy_path)`**
Loads the last 128 SPC frames, unpacks the bit-packed temporal bins, reshapes into a
flat 384-channel array, and normalises. A small epsilon in the denominator prevents
division by zero on completely dark frames — a robustness fix over Phase 2.

**`SPCDataset`**
A proper PyTorch Dataset replacing the manual loop from Phase 2. Training samples go
through `_augment` (random horizontal flip, vertical flip, and 90° rotation — applied
identically to input and target to preserve alignment) and `_crop` (random 512×512
patch). Val and test samples are served at full 800×800 with no augmentation so that
metrics are consistent across runs.

**`CharbonnierLoss`**
Smooth L1 replacement. The epsilon inside the square root keeps the function
differentiable at zero, which matters at late training when per-pixel errors become
very small and plain L1 gradients become noisy.

**`VGGPerceptualLoss`**
Builds three sequential slices from a frozen pretrained VGG16 and computes L1 distance
between prediction and target feature maps at three depths. VGG weights are entirely
frozen — this module acts as a fixed perceptual feature extractor, not a trainable
component.

**`UNetBasic`**
The model. The `_block` helper creates the repeated two-Conv2d + ReLU pattern used
throughout encoder and decoder. In `forward`, encoder outputs `e1`, `e2`, `e3` are
saved and concatenated with the upsampled decoder tensors via `torch.cat([u, e], dim=1)`
— this is the skip connection in practice, and why each decoder conv block takes
`in_ch + skip_ch` as its input channel count.

**`train_one_epoch(...)`**
One full pass over the training set under mixed precision. The `autocast` context reduces
memory usage and speeds up training on modern GPUs. `scaler.unscale_` is called before
gradient clipping so the clip operates on true gradient magnitudes, not the scaled ones.

**`validate(...)`**
Same structure as training but under `torch.no_grad()` with no backward pass. Returns
loss, SSIM, MS-SSIM, and PSNR averaged over all validation samples each epoch.

**`evaluate_test(...)`**
Loads the best checkpoint and runs inference on the test set. Uses scikit-image's PSNR
and SSIM for final reporting — the same convention as Phase 2 so numbers are directly
comparable across phases.

**`save_training_curves(log_csv)`**
Reads the per-epoch CSV log and produces a 3-panel figure: Loss, SSIM, and PSNR for
both train and val across all 150 epochs. The train/val gap in these curves reveals
whether augmentation was sufficient to prevent overfitting.

---

## Running It

```bash
pip install torch torchvision torchmetrics scikit-image matplotlib Pillow
python unet_reconstruction.py
```

Update `DATA_ROOT` at the top. On Kaggle, use `run_and_collect_results.ipynb`.

---

## Output Files

| File | What it shows |
|------|--------------|
| `results/training_curves.png` | Loss, SSIM, PSNR across 150 epochs — train and val |
| `results/training_log.csv` | Full per-epoch metrics |
| `results/comparison_sample{i}_{scene}.png` | SPC input vs UNet prediction vs Ground Truth |
| `results/metrics.json` | PSNR and SSIM per test sample |
| `checkpoints/best_model.pth` | Best model by val SSIM (epoch 139) |

---

## Results

| Sample | Scene | PSNR ↑ | SSIM ↑ |
|:------:|-------|:------:|:------:|
| 0 | sunny-room       | 30.67 dB | 0.9313 |
| 1 | tv-couch         | 34.76 dB | 0.9091 |
| 2 | ultramodern      | 31.23 dB | 0.8596 |
| 3 | white-room       | 34.66 dB | 0.9480 |
| 4 | wooden-staircase | 31.77 dB | 0.7616 |
| **Average** | | **32.62 dB** | **0.8819** |

### Phase 2 → Phase 3 Comparison

| Scene | Phase 2 PSNR | Phase 3 PSNR | Δ |
|-------|:------------:|:------------:|:-:|
| sunny-room       | 26.70 dB | 30.67 dB | **+3.97 dB** |
| tv-couch         | 31.98 dB | 34.76 dB | **+2.78 dB** |
| ultramodern      | 26.47 dB | 31.23 dB | **+4.76 dB** |
| white-room       | 30.53 dB | 34.66 dB | **+4.13 dB** |
| wooden-staircase | 28.81 dB | 31.77 dB | **+2.96 dB** |
| **Average**      | **28.90 dB** | **32.62 dB** | **+3.72 dB** |

| Metric | Phase 1 (Naive) | Phase 2 (CNN) | Phase 3 (UNet) |
|--------|:---------------:|:-------------:|:--------------:|
| Avg PSNR | 21.55 dB | 28.90 dB | **32.62 dB** |
| Avg SSIM | 0.4547 | 0.8701 | **0.8819** |

---

## Visual Results

### Training Curves

![Training curves](results/training_curves.png)

### Test Set Comparisons

| Scene | Comparison |
|-------|-----------|
| sunny-room       | ![](results/comparison_sample0_sunny-room.png) |
| tv-couch         | ![](results/comparison_sample1_tv-couch.png) |
| ultramodern      | ![](results/comparison_sample2_ultramodern.png) |
| white-room       | ![](results/comparison_sample3_white-room.png) |
| wooden-staircase | ![](results/comparison_sample4_wooden-staircase.png) |

---

## Observations

**Skip connections delivered the largest per-scene gains where they were needed most.**
`ultramodern` was the hardest scene in Phase 2 at 26.47 dB — the scene with the most
complex geometry and fine texture. Phase 3 brings it to 31.23 dB, a +4.76 dB gain,
the largest improvement across all scenes. `white-room` gained +4.13 dB. These are
exactly the scenes where multi-scale feature reuse matters most: the encoder builds
a global understanding of the room layout while the skip connections feed fine edge
and texture detail directly to the decoder, bypassing the bottleneck compression.

**PSNR variance across scenes tightened significantly.**
Phase 2 ranged from 26.47 to 31.98 dB — a 5.5 dB spread. Phase 3 ranges from
30.67 to 34.76 dB — a 4.1 dB spread. The UNet's ability to reason at multiple
scales raised the floor for harder scenes more than the ceiling for easier ones,
which is the expected behaviour of multi-scale architectures.

**The training curves tell a healthy story.**
Loss, SSIM, and PSNR all converge cleanly over 150 epochs. The train and val curves
stay close throughout — the final train/val SSIM gap is only 0.0002, meaning
augmentation (flips, rotations, random crops) was effective at preventing overfitting
despite the model having 31M parameters. The best checkpoint was saved at epoch 139,
close to the end, suggesting the cosine annealing schedule was well-calibrated.

**Val PSNR exceeded train PSNR at the end (33.50 vs 30.81 dB).**
This is a counterintuitive but explainable result. Training uses random 512×512 crops,
which are harder to reconstruct than full images because the network sees less context
per sample. Validation uses full 800×800 images, where the model has the complete scene
available. The larger input effectively gives the network more context per inference,
leading to higher measured quality on val despite the model being the same.

**`wooden-staircase` has the lowest SSIM at 0.7616.**
This is likely the scene with the most repetitive fine texture — wood grain and stair
steps are structurally complex at a local level, making SSIM harder to maximise. PSNR
is 31.77 dB (reasonable), but SSIM penalises the loss of repeated local patterns more
than it penalises diffuse errors. This is the clearest motivation for Phase 4: attention
gates can learn to focus reconstruction effort on these high-frequency, structurally
demanding regions specifically.

---

← [Phase 2](../phase2_baseline_cnn/README.md) | [Back to main repo](../README.md) | [Phase 4 →](../phase4_resunet_attention/README.md)
