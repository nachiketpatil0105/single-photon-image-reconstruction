# Phase 4 — ResUNet with Guided Attention Gates

← [Phase 3](../phase3_unet/README.md) | [Back](../README.md)

The final model. Four improvements over Phase 3's UNet: residual conv blocks, a 5-level
encoder with double bottleneck, guided attention gates on every skip connection, and deep
supervision. Trained on 1850 samples on a dedicated GPU server.

---

## What Changed from Phase 3

| | Phase 3 | Phase 4 |
|--|---------|---------|
| Conv blocks | Plain Conv + ReLU | Residual (GroupNorm + SiLU + residual path) |
| Encoder depth | 3 levels | 5 levels |
| Bottleneck | 1024ch, single block | 1024→2048→2048, double block |
| Skip connections | Raw concatenation | Guided attention gates |
| Deep supervision | None | Aux head at dec2, annealed 0.4→0 |
| Loss | Charb + MS-SSIM + VGG | + **Edge loss** |
| Precision | FP16 | BF16 |
| LR schedule | Cosine annealing | Cosine annealing **with warm restarts** (T₀=5) |
| Dataset | ~45 samples | **1850 samples** |
| Parameters | ~31M | **~237M** |
| Training hardware | Kaggle GPU | Dedicated server |

---

## Architecture — `ResUNetAttention`

```
Input (96, 512, 512)
  input_proj: 96→128

  Encoder (ResConvBlock + MaxPool)
    enc1: 128→128   (512×512) → e1
    enc2: 128→256   (256×256) → e2
    enc3: 256→512   (128×128) → e3
    enc4: 512→1024  ( 64×64)  → e4
    enc5: 1024→1024 ( 32×32)  → e5

  Bottleneck: 1024→2048→2048  (32×32)

  Decoder (ConvTranspose ↑2 + GuidedAttentionGate + concat)
    dec0: 2048→1024  ( 64×64)   attn(e5)
    dec1: 1024→512   (128×128)  attn(e4)
    dec2:  512→256   (256×256)  attn(e3)  ← aux head
    dec3:  256→128   (512×512)  attn(e2)
    dec4:  128→128   (512×512)  attn(e1)

  Output: Conv1×1  128→3
```

~237M parameters.

---

## Key Design Decisions

**`ConvBlock` — Residual**
Each block computes `act(conv(x) + proj(x))`. GroupNorm (8 groups) replaces BatchNorm
because batch size 2 makes BN statistics unreliable. SiLU replaces ReLU for smoother
gradients.

**`GuidedAttentionGate`**
Standard Phase 3 skips pass raw encoder features to the decoder. The attention gate
computes a scalar map α ∈ [0,1] per spatial position using both the encoder skip
(where) and the upsampled decoder feature (what to look for). Low-α positions are
suppressed before concatenation.

**Deep supervision + annealing**
Auxiliary 1×1 conv at dec2 projected to 3 channels. Weight decays linearly from 0.4
to 0 over 70 epochs — forces intermediate features to be meaningful early, contributes
nothing to the gradient by the end.

**Edge Loss**
New in Phase 4. Computes finite-difference dx/dy gradients for prediction and target,
penalizes L1 distance between them. Directly encourages sharp boundary preservation.

**BF16 precision**
BF16 has the same exponent range as FP32 (vs FP16's narrower range). At 237M parameters
and 2048-channel bottleneck, this prevents overflow/underflow in the large layers.

**GPU augmentation**
Flips and rotations applied after moving tensors to device — faster than CPU augmentation
for large tensors at batch size 2.

---

## Loss Function

```
Loss = 0.5  × Charbonnier
     + 0.3  × (1 − MS-SSIM)
     + 0.1  × VGG Perceptual
     + 0.1  × Edge
     + deep_sup_w × Charbonnier(aux, target)
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4, cosine warm restarts (T₀=5, η_min=5e-5) |
| Epochs | 70 |
| Batch Size | 2 |
| Precision | BF16 autocast + GradScaler |
| Gradient Clipping | max norm 1.0 |
| Augmentation | GPU flips + rotations (p=0.2) |
| Dataset | 1850 samples |
| Checkpoint | Best train SSIM |
| Hardware | Dedicated server, NVIDIA GPU |

---

## Code

**`resunet_reconstruction.py`**

| Function / Class | What it does |
|-----------------|-------------|
| `collect_paths(npy_root, img_root)` | Walks paired `train_*` folders, returns matched .pt file lists |
| `SPCDataset` | Loads preprocessed .pt tensors, slices last 96 input channels |
| `gpu_augment(x, y)` | Random flips and rotations on GPU after device transfer |
| `CharbonnierLoss` | `sqrt((pred - target)² + ε²)` — smooth, stable L1 replacement |
| `VGGPerceptualLoss` | Frozen VGG16, L1 at relu1_2 / relu2_2 / relu3_3 |
| `EdgeLoss` | Finite-difference gradient loss — penalizes edge blurring |
| `ConvBlock` | Two Conv3×3 + GroupNorm + SiLU with residual projection |
| `GuidedAttentionGate` | Computes spatial attention from skip + decoder gate, filters skip |
| `Down` / `Up` | MaxPool+ConvBlock / ConvTranspose+AttentionGate+ConvBlock |
| `ResUNetAttention` | Full model — returns `(main, aux)` during training, `main` during eval |
| `train(...)` | Full training loop — BF16, deep supervision annealing, CSV logging, TensorBoard |

---

## Running

> Requires a high-VRAM GPU server (~40GB for 237M params + batch 2 at 512×512).

```bash
pip install torch torchvision torchmetrics tensorboard
python resunet_reconstruction.py
```

Update `BASE_DIR`, `DATA_DIR_IMG`, and `DATA_DIR_NPY` at the top of the script.
Training resumes automatically if `checkpoints/best_model.pth` exists.

---

## Results

Evaluated on 3 held-out samples using the epoch 70 checkpoint (best train SSIM = 0.9145).

| Sample | PSNR ↑ | SSIM ↑ |
|:------:|:------:|:------:|
| 000015 | 28.21 dB | 0.7561 |
| 000023 | 36.35 dB | 0.9442 |
| 000030 | 37.43 dB | 0.9496 |
| **Avg** | **34.00 dB** | **0.8833** |

**vs Phase 3:** +1.38 dB PSNR, +0.001 SSIM

### Full Progression

| Phase | Model | Avg PSNR | Avg SSIM |
|:-----:|-------|:--------:|:--------:|
| 1 | Naive | 21.55 dB | 0.4547 |
| 2 | CNN | 28.90 dB | 0.8701 |
| 3 | UNet | 32.62 dB | 0.8819 |
| **4** | **ResUNet + Attention** | **34.00 dB** | **0.8833** |

---

## Visual Results

![Training curves](results/training_curves.png)

| 000015 | 000023 | 000030 |
|:------:|:------:|:------:|
| ![](results/comparison_000015.png) | ![](results/comparison_000023.png) | ![](results/comparison_000030.png) |

---

## Observations

The P3→P4 gain (+1.38 dB) is smaller than P2→P3 (+3.72 dB), which is expected — each
phase targets harder residual errors left by the previous one.

Per-sample spread is large (28.21 to 37.43 dB). Sample 000015 scores significantly lower,
likely a scene with complex lighting or high-frequency texture that even guided attention
can't fully recover. Samples 000023 and 000030 exceed 36 dB, showing the model is
highly capable on structured scenes.

The training curves show the warm restart fingerprint — SSIM oscillates with period 5,
rising within each cosine cycle then dipping slightly on restart. The overall trend is
cleanly upward (0.74 → 0.91) with no plateau at epoch 70, suggesting further gains are
possible with more training.

Deep supervision worked as designed. The aux Charbonnier loss improved from 0.0556 at
epoch 1 to 0.0201 at epoch 70 despite its weight decaying to zero — intermediate features
became genuinely more informative rather than being forced by a strong signal.

---

← [Phase 3](../phase3_unet/README.md) | [Back](../README.md)
