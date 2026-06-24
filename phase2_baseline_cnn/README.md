# Phase 2 — Baseline CNN

> First learned model: a flat convolutional network trained to reconstruct clean images
> directly from stacked single-photon frames.

← [Phase 1](../phase1_naive/README.md) | [Back to main repo](../README.md) | [Phase 3 →](../phase3_unet/README.md)

---

## What This Phase Is About

Phase 1 showed that naive summation hits a hard ceiling — it recovers rough scene structure
but loses sharpness, fine texture, and precise intensity. The fundamental problem is that
summation treats every frame and every pixel equally. It has no way to learn which patterns
matter and which are noise.

This phase introduces the first **trained model**. The idea is straightforward: give the
network all 128 SPC frames at once as input channels, and let it learn — through
backpropagation — what to emphasize and what to suppress. No handcrafted rules. The network
figures out the mapping from noisy photon data to clean image entirely from examples.

The architecture is intentionally kept simple: no skip connections, no downsampling, no
attention. Just eight convolutional layers operating at full resolution. This is a baseline —
something clean and honest that Phase 3 and 4 can genuinely improve on.

---

## Input Representation

The way SPC data is fed into the network is worth understanding because it is different from
a typical image input.

The raw file has shape `(1024, H, W, 100, 3)` — 1024 binary frames, each with 100 packed
temporal bins per pixel across 3 channels. We take the last 128 frames, unpack the bits to
recover the spatial data, and reshape so that each pixel has `128 × 3 = 384` values — one
per frame per channel. This flattened stack becomes the 384 input channels to the CNN.

The key idea: rather than summarizing the frames (as Phase 1 did by summing), we preserve
all of them and let the network decide how to combine them. Every frame gets its own
dedicated input channels.

```
Raw .npy  (1024, 800, 100, 3)
    ↓  slice last 128 frames
         (128, 800, 100, 3)
    ↓  unpack bits
         (128, 800, 800, 3)
    ↓  reshape + transpose
         (384, 800, 800)        ← 384 channels into the CNN
```

---

## Architecture

`SimCNN` is a flat, fully-convolutional network. All eight layers operate at the same
spatial resolution (800 × 800) — no pooling, no downsampling, no upsampling. Padding of 1
on every 3×3 conv preserves spatial dimensions throughout.

```
Input  (384, 800, 800)
  Conv2d 384 → 128  + ReLU      compress 384 channels into a richer feature space
  Conv2d 128 → 256  + ReLU      expand: build more complex feature combinations
  Conv2d 256 → 256  + ReLU      refine at peak capacity
  Conv2d 256 → 128  + ReLU      contract: start reducing toward image channels
  Conv2d 128 →  64  + ReLU
  Conv2d  64 →  32  + ReLU
  Conv2d  32 →  16  + ReLU
  Conv2d  16 →   3  + Sigmoid   output RGB image in [0, 1]
Output   (3, 800, 800)
```

Total trainable parameters: **~1.72 million**

The channel count first expands (384 → 256) to build a richer intermediate representation,
then contracts back (256 → 3) to produce the final RGB output. Sigmoid at the end ensures
values stay in [0, 1] without explicit clipping.

---

## Loss Function

Training uses a **combined L1 + SSIM loss**:

```
Loss = 0.5 × L1(pred, target) + 0.5 × (1 − SSIM(pred, target))
```

L1 alone encourages the network to minimize pixel-level error, which often produces blurry
outputs because averaging pixel values lowers the loss even when texture is destroyed. SSIM
penalizes the loss of structural information — edges, local contrast, luminance — which is
exactly what single-photon reconstruction needs to recover. The 50/50 combination balances
pixel accuracy with perceptual quality.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Epochs | 50 |
| Batch Size | 1 (sample-by-sample) |
| Loss | 0.5 × L1 + 0.5 × (1 − SSIM) |
| Train / Test Split | 45 / 5 samples |
| Input Frames | Last 128 SPC frames |

---

## Code Walkthrough

**`cnn_reconstruction.py`** runs the full pipeline — data loading, training, evaluation,
and saving results — in a single script.

**`load_input(npy_path)`**
The most important function in this phase. Loads the SPC data, takes the last 128 frames,
unpacks the bit-packed bins, and reshapes the frame and channel dimensions into a flat
384-channel array. Normalizes and returns it in channel-first format `(384, 800, 800)`
ready for PyTorch. This is where raw photon data becomes a neural network input.

**`load_target(png_path)`**
Loads the ground truth PNG, normalizes pixel values from [0, 255] to [0, 1], and
transposes to channel-first format `(3, 800, 800)`.

**`to_tensor(arr, device)`**
A small utility that wraps a NumPy array into a PyTorch tensor, adds a batch dimension,
and sends it to the correct device. Called for both input and target in every training
and evaluation step.

**`SimCNN`**
The model class. The entire architecture lives inside `self.net` as one `nn.Sequential`
block. The `forward` method just passes input through it — no branching, no skip
connections. This simplicity is intentional: Phase 3 adds the skip connections, so the
gap in performance will be directly attributable to that architectural change.

**`train(...)`**
Loops over epochs and samples. For each sample it loads input and target, runs a forward
pass, computes the combined loss, backpropagates, and updates weights. Prints loss per
epoch and returns the full history for the loss curve.

**`evaluate(...)`**
Runs inference on the 5 test samples under `torch.no_grad()`. Uses scikit-image's PSNR
and SSIM on NumPy arrays — the standard for reporting image reconstruction metrics. Saves
a 3-panel comparison figure per sample and collects all results for JSON export.

**`save_comparison(...)`**
3-panel figure: SPC input (first 3 channels as a proxy visual) | CNN prediction | Ground
Truth. PSNR and SSIM are printed as a subtitle under the prediction panel so numbers are
visible directly on the figure.

**`save_loss_curve(epoch_losses)`**
Plots training loss vs epoch. Useful for checking if the model converged cleanly or is
still declining at epoch 50 — which would suggest more training is warranted.

---

## Running It

```bash
pip install torch torchvision torchmetrics scikit-image matplotlib Pillow
python cnn_reconstruction.py
```

Update `DATA_ROOT` at the top of the script to your local dataset path. On Kaggle, use
`run_and_collect_results.ipynb` — the same logic formatted as a notebook with inline plots.

---

## Output Files

| File | What it shows |
|------|--------------|
| `results/loss_curve.png` | Training loss across all 50 epochs |
| `results/comparison_sample{i}_{scene}.png` | SPC input vs CNN prediction vs Ground Truth |
| `results/metrics.json` | PSNR and SSIM per test sample + full epoch loss history |
| `results/cnn_model.pth` | Saved model weights |

---

## Results

| Sample | Scene | PSNR ↑ | SSIM ↑ |
|:------:|-------|:------:|:------:|
| 0 | sunny-room       | 26.70 dB | 0.8951 |
| 1 | tv-couch         | 31.98 dB | 0.8848 |
| 2 | ultramodern      | 26.47 dB | 0.8207 |
| 3 | white-room       | 30.53 dB | 0.9120 |
| 4 | wooden-staircase | 28.81 dB | 0.8382 |
| **Average** | | **28.90 dB** | **0.8701** |

**Phase 1 → Phase 2 improvement: +7.35 dB PSNR, +0.42 SSIM**

---

## Visual Results

### Training Loss Curve

![Loss curve](results/loss_curve.png)

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

**The jump from Phase 1 is large and immediate.**
Average PSNR went from 21.55 dB (naive, best scene) to 28.90 dB — a gain of +7.35 dB.
SSIM went from 0.45 to 0.87. This confirms the core hypothesis: preserving all 128 frames
as separate input channels and letting the network learn how to combine them is far more
powerful than any fixed summation rule.

**The loss curve shows healthy but incomplete convergence.**
Loss drops sharply in the first 10 epochs (0.318 → 0.142), then continues to decline
steadily all the way to epoch 50 (0.096). Crucially, it is still descending at the end —
it has not plateaued. This means the model has not fully converged, and more epochs would
likely push the metrics higher. The current numbers are a conservative lower bound.

**Scene difficulty varies significantly.**
`tv-couch` achieves 31.98 dB PSNR — likely because it has large flat regions (sofa,
walls) where the signal-to-noise ratio is naturally higher and easier to reconstruct.
`ultramodern` and `sunny-room` are the hardest at ~26.5 dB, probably due to complex
lighting, fine furniture detail, or high-frequency texture that a flat CNN without
multi-scale features cannot fully recover.

**SSIM is uniformly high, PSNR varies more.**
All five scenes score above 0.82 SSIM, meaning structural similarity is well preserved
across the board. But PSNR spreads from 26.47 to 31.98 dB — a 5.5 dB range. This
suggests the network has learned scene structure reliably but struggles with precise
per-pixel intensity in complex scenes. That intensity precision is what a UNet's skip
connections are specifically designed to recover, motivating Phase 3.

---

← [Phase 1](../phase1_naive/README.md) | [Back to main repo](../README.md) | [Phase 3 →](../phase3_unet/README.md)
