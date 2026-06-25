# Phase 4 — ResUNet with Guided Attention Gates

> The final and most capable model: residual blocks, a 5-level encoder, a double
> bottleneck, and guided attention gates on every skip connection.

← [Phase 3](../phase3_unet/README.md) | [Back to main repo](../README.md)

---

## What This Phase Is About

Phase 3 established that a UNet with skip connections significantly outperforms a flat CNN
— especially on hard scenes like `ultramodern` (+4.76 dB). But it still has three
structural limitations. First, the encoder only has three levels, so the bottleneck sees
the scene at 100×100 resolution — not enough context for global reasoning. Second, the
conv blocks are plain stacked convolutions with no residual connections, making training
harder as depth increases. Third, the skip connections pass raw encoder features to the
decoder without filtering: every pixel in the encoder map is treated as equally relevant,
even noisy or uninformative regions.

This phase addresses all three. The result is a significantly more powerful architecture
trained on a much larger dataset on a dedicated server with a high-RAM GPU — a setup that
was necessary given the model's 237M parameters.

---

## What Changed from Phase 3

| Component | Phase 3 (UNet) | Phase 4 (ResUNet + Attention) |
|-----------|:--------------|:-----------------------------|
| Conv blocks | Plain Conv + ReLU | Residual blocks with GroupNorm + SiLU |
| Encoder depth | 3 levels | 5 levels |
| Bottleneck | Single ConvBlock, 1024ch | Double ConvBlock, 1024→2048→2048ch |
| Skip connections | Raw concatenation | Guided attention gates (Oktay et al., 2018) |
| Deep supervision | None | Auxiliary output head at dec2, annealed to 0 |
| Loss | Charb + MS-SSIM + VGG | Charb + MS-SSIM + VGG + **Edge** |
| Input frames | 128 frames | 96 frames (preprocessed .pt tensors) |
| Augmentation | CPU, in Dataset | GPU, in training loop |
| Precision | FP16 | **BF16** (more stable range for large models) |
| LR schedule | Cosine annealing | Cosine annealing **with warm restarts** (T₀=5) |
| Dataset size | ~45 samples | **1850 samples** |
| Batch size | 1 | 2 |
| Parameters | ~31.2M | **~236.9M** |
| Training hardware | Kaggle GPU | Dedicated server, cuda:3 |
| Epochs | 150 | 70 |

---

## Architecture

`ResUNetAttention` has a 5-level encoder-decoder with guided attention at every skip.

```
Input (96, 512, 512)
  input_proj: 96→128  (1×1 Conv)

  ┌──────────────────────────────────────────────────────────────┐
  │ Encoder (ResConvBlocks + MaxPool)                           │
  │  enc1: 128→128  (512×512)  → e1                            │
  │  enc2: 128→256  (256×256)  → e2                            │
  │  enc3: 256→512  (128×128)  → e3                            │
  │  enc4: 512→1024  (64×64)   → e4                            │
  │  enc5: 1024→1024 (32×32)   → e5                            │
  └──────────────────────────────────────────────────────────────┘
  Bottleneck: 1024→2048→2048  (32×32, double ConvBlock)

  ┌──────────────────────────────────────────────────────────────┐
  │ Decoder (ConvTranspose ↑2 + GuidedAttentionGate + concat)  │
  │  dec0: 2048→1024   (64×64)   attn(e5)                      │
  │  dec1: 1024→512   (128×128)  attn(e4)                      │
  │  dec2:  512→256   (256×256)  attn(e3)  ← aux head here     │
  │  dec3:  256→128   (512×512)  attn(e2)                      │
  │  dec4:  128→128   (512×512)  attn(e1)                      │
  └──────────────────────────────────────────────────────────────┘
  Output: Conv1×1  128→3  →  (3, 512, 512)
```

Total trainable parameters: **~236.9 million**

---

## Key Design Decisions Explained

**Residual ConvBlocks (`ConvBlock`)**
Each block computes a correction over its input rather than a full transformation:
`output = act(block(x) + residual_proj(x))`. When `in_ch == out_ch` the projection is an
identity. GroupNorm with 8 groups replaces BatchNorm because batch size 2 makes BatchNorm
statistics unreliable. SiLU replaces ReLU for smoother gradients near zero.

**5-level encoder + double bottleneck**
Adding `enc5` halves the spatial resolution one more time to 32×32, giving the bottleneck
a much larger effective receptive field. The double bottleneck (1024→2048→2048) maximises
reasoning capacity at this scale where computation is cheap but semantic content is richest.

**Guided Attention Gate (`GuidedAttentionGate`)**
The standard Phase 3 skip — `torch.cat([upsampled_decoder, encoder_skip], dim=1)` — passes
all encoder features equally. The attention gate improves this by computing a scalar
attention map α ∈ [0,1] per spatial position. It does so using BOTH the encoder skip
(`skip`) and the upsampled decoder feature (`gate`): the decoder says what to look for,
the encoder says where it is. Positions with low α are suppressed before concatenation.

**Deep supervision + annealing**
An auxiliary 1×1 conv at dec2 (256×256) projects to 3 channels, upsampled to the output
size and compared against the target using Charbonnier loss. Its weight starts at 0.4 and
decays linearly to 0 by epoch 70. This forces the intermediate feature maps to stay
semantically meaningful early in training, acting as a regulariser when there is no
held-out validation set. By the end, the aux head contributes nothing to the loss.

**Edge Loss**
New in Phase 4. Computes finite-difference gradients horizontally (dx) and vertically (dy)
for both prediction and target, then penalises L1 distance between them. This directly
encourages preservation of sharp boundaries — the most visually obvious failure mode of
photon-counting reconstruction.

**GPU augmentation**
Augmentation (random flips, 90° rotations) is applied after moving tensors to GPU rather
than in the Dataset. At batch size 2 with large tensors, this avoids CPU→GPU transfer
of augmented data and is noticeably faster.

**BF16 mixed precision**
bfloat16 instead of float16 for autocast. BF16 has the same range as FP32 (8 exponent
bits vs FP16's 5) so it is far less likely to overflow or underflow in the large
bottleneck layers (2048 channels). At 237M parameters this matters.

**CosineAnnealingWarmRestarts (T₀=5)**
LR restarts every 5 epochs, cycling between `LEARNING_RATE` (1e-4) and `ETA_MIN` (5e-5).
Warm restarts help escape local minima by periodically increasing the LR, which works
well when training on a large dataset (1850 samples) for fewer total epochs.

---

## Loss Function

```
Loss = 0.5 × Charbonnier
     + 0.3 × (1 − MS-SSIM)
     + 0.1 × VGG Perceptual
     + 0.1 × Edge
     + deep_sup_w × Charbonnier(aux, target)
```

Edge loss is the new addition vs Phase 3. The Charbonnier weight increases to 0.5
(from 0.25) and MS-SSIM reduces to 0.3 (from 0.50), shifting the balance slightly
toward pixel-level accuracy now that the edge loss handles structural sharpness.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 (warm restarts, T₀=5, η_min=5e-5) |
| Epochs | 70 |
| Batch Size | 2 |
| Precision | BF16 autocast + GradScaler |
| Gradient Clipping | Max norm = 1.0 |
| Augmentation | GPU flips + 90° rotations (prob 0.2) |
| Dataset | 1850 training samples |
| Checkpoint | Best train SSIM |
| Training hardware | Dedicated server, NVIDIA GPU (cuda:3) |

---

## Code Walkthrough

**`resunet_reconstruction.py`** contains the full model, losses, dataset, and training loop.

**`collect_paths(npy_root, img_root)`**
Walks paired `train_*` folder trees and returns matched .pt file paths. The .pt format is
a key difference from earlier phases — data was preprocessed offline into PyTorch tensors,
making each load a single `torch.load` instead of the NumPy unpack pipeline from Phases 2/3.

**`SPCDataset`**
Minimal dataset — just loads two .pt files and slices the last 96 input channels. All
augmentation happens in `gpu_augment` after moving to the device, so no augmentation
logic lives here.

**`gpu_augment(x, y)`**
Random horizontal flip, vertical flip, and 90° rotation applied to both input and target
simultaneously on GPU. Applied in-batch before the forward pass. Faster than CPU
augmentation for large tensors.

**`CharbonnierLoss`, `VGGPerceptualLoss`, `EdgeLoss`**
Same Charbonnier and VGG as Phase 3. `EdgeLoss` is new: it computes dx/dy finite
differences for prediction and target and penalises their L1 distance, directly
encouraging sharp edge preservation.

**`ConvBlock`**
The fundamental building unit. Two 3×3 convs with GroupNorm and SiLU, plus a residual
path (1×1 conv if channel counts differ, else identity). Every encoder and decoder block
in the model is built from this.

**`GuidedAttentionGate`**
The most architecturally significant addition over Phase 3. Projects encoder skip and
decoder gate to a shared intermediate space, adds them, applies a 1×1 conv + Sigmoid to
produce a spatial attention map, and multiplies it back onto the skip. The decoder's
context guides which encoder positions are passed forward.

**`Down`** and **`Up`**
Thin wrappers. `Down` = MaxPool2d + ConvBlock. `Up` = ConvTranspose2d + GuidedAttentionGate
+ ConvBlock (with concatenated attended skip). These keep the main model's `forward`
readable.

**`ResUNetAttention`**
The model. `forward` during training returns `(main_out, aux_out)` — a tuple. During
inference (`model.eval()`) it returns only `main_out`. The caller must handle this
difference, which is why the training loop unpacks two values and the inference script
calls `model.eval()` first.

**`train(...)`**
The training loop. Deep supervision weight `deep_sup_w` is recalculated each epoch as
`W_DEEP_SUP * max(0, 1 - epoch/NUM_EPOCHS)`. BF16 autocast wraps both the forward pass
and loss computation. Gradient clipping is applied after `scaler.unscale_()` so it
operates on true gradient magnitudes. Checkpoints are saved both when a new best SSIM is
reached and every 10 epochs.

---

## Running It

This code requires a high-RAM GPU server. It was trained with:
- ~40GB GPU VRAM (237M parameters + batch 2 at 512×512)
- 12 CPU workers for data loading
- PyTorch 2.x with `torch.compile`

```bash
pip install torch torchvision torchmetrics tensorboard
python resunet_reconstruction.py
```

Update `BASE_DIR`, `DATA_DIR_IMG`, and `DATA_DIR_NPY` at the top of the script.

---

## Output Files

| File | What it shows |
|------|--------------|
| `results/training_curves.png` | Train SSIM and PSNR across 70 epochs |
| `results/comparison_{id}.png` | Model prediction vs Ground Truth per sample |
| `output/training_log.csv` | Per-epoch metrics including all loss components |
| `output/training.log` | Full timestamped training log |
| `checkpoints/best_model.pth` | Best model by train SSIM (epoch 70) |

---

## Results

Evaluated on 3 held-out samples using the checkpoint from epoch 70
(best train SSIM = 0.9145).

| Sample | PSNR ↑ | SSIM ↑ |
|:------:|:------:|:------:|
| 000015 | 28.21 dB | 0.7561 |
| 000023 | 36.35 dB | 0.9442 |
| 000030 | 37.43 dB | 0.9496 |
| **Average** | **34.00 dB** | **0.8833** |

### Full Phase Progression

| Phase | Model | Avg PSNR | Avg SSIM |
|:-----:|-------|:--------:|:--------:|
| 1 | Naive Summation | 21.55 dB | 0.4547 |
| 2 | Baseline CNN | 28.90 dB | 0.8701 |
| 3 | UNet | 32.62 dB | 0.8819 |
| **4** | **ResUNet + Attention** | **34.00 dB** | **0.8833** |

**Total improvement across all phases: +12.45 dB PSNR, +0.43 SSIM**

---

## Visual Results

### Training Curves

![Training curves](results/training_curves.png)

### Test Comparisons (Prediction vs Ground Truth)

| Sample 000015 | Sample 000023 | Sample 000030 |
|:-------------:|:-------------:|:-------------:|
| ![](results/comparison_000015.png) | ![](results/comparison_000023.png) | ![](results/comparison_000030.png) |

---

## Observations

**The jump from Phase 3 to Phase 4 is real but context-dependent.**
Average PSNR improved from 32.62 to 34.00 dB (+1.38 dB). That is a smaller absolute gain
than Phase 2→3 (+3.72 dB) or Phase 1→2 (+7.35 dB), which is expected: each phase is
operating on harder residual errors left by the previous one. The jump from 0.88 to 0.88
in SSIM looks small numerically but SSIM is highly non-linear at high values — moving
from 0.94 to 0.95 requires far more of the model than moving from 0.45 to 0.55.

**The per-sample spread reveals what attention gates do best.**
Sample 000015 sits at only 28.21 dB PSNR / 0.7561 SSIM — considerably below the other
two. This is likely a scene with complex, high-frequency texture or challenging lighting
where even 237M parameters and guided attention cannot fully close the gap. Samples
000023 and 000030 both exceed 36 dB PSNR and 0.94 SSIM, showing the model is highly
capable on well-lit, structurally cleaner scenes. Attention gates help most on scenes
where informative features are spatially concentrated — that is where selectively
amplifying encoder positions pays off most.

**Training on 1850 samples vs ~45 is the biggest single factor.**
The jump in dataset scale from Phases 2/3 to Phase 4 is 40×. Even with a much larger
model, this change alone accounts for much of the gain. The model sees far more scene
diversity, lighting conditions, and texture types, making it less likely to overfit to
the training distribution. This is why the deep supervision anneal and GPU augmentation
matter: with 1850 samples you have enough signal to train a 237M parameter model well,
but you still need regularisation to generalise cleanly.

**The training curve shows the warm restart fingerprint clearly.**
SSIM oscillates with a period of 5 epochs throughout training — rising sharply as LR
decays within each cycle, then dipping slightly when LR resets. This is the expected
behaviour of CosineAnnealingWarmRestarts and confirms the schedule is working correctly.
The overall trend is consistently upward from 0.74 at epoch 1 to 0.91 at epoch 70,
with no signs of plateauing — training for longer would likely yield further gains.

**Deep supervision annealing works as designed.**
The auxiliary loss weight starts at 0.4 (epoch 1) and reaches exactly 0.0 at epoch 70.
In the training log, the aux Charbonnier loss drops from 0.0556 at epoch 1 to 0.0201 at
epoch 70 — it improved substantially over training even though its weight was declining,
meaning the intermediate features at dec2 became genuinely more informative rather than
being forced by a strong loss signal. By epoch 70 the aux head contributes nothing to
the gradient, so the final model is purely determined by the main output head.

---

← [Phase 3](../phase3_unet/README.md) | [Back to main repo](../README.md)
