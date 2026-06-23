# Phase 1 — Naive Summation Baseline

> No learning. No weights. Just physics — sum enough photon frames and the scene emerges.

← [Back to main repo](../README.md) | [Phase 2 →](../phase2_baseline_cnn/README.md)

---

## What This Phase Is About

Before training any model, we need to know what we're trying to beat. This phase builds a
parameter-free baseline using the simplest possible idea: if you stack enough binary
single-photon frames on top of each other and sum them, the random noise cancels out and
the actual scene starts to appear.

There is no training, no loss function, no optimization. The only question this phase
answers is — **how good can you get with just summation, and how many frames do you need?**
That number becomes the floor every model in Phase 2 onwards has to clear.

---

## How It Works

A single-photon camera doesn't capture a normal image. Each frame is binary — every pixel
is either 0 (no photon detected) or 1 (photon detected). A single frame looks like random
noise. But photons arrive more often at brighter parts of the scene, so if you sum thousands
of frames, the bright regions accumulate higher counts and the image gradually reveals itself.

The data is stored as bit-packed `.npy` files with shape `(1024, H, W, 100, 3)`:
- `1024` — total photon frames available
- `H, W` — spatial resolution (800 × 800)
- `100` — packed temporal bins per pixel
- `3` — RGB channels

The reconstruction pipeline is four steps:

```
Load .npy with memory-map   →   Slice last B frames   →   Unpack bits   →   Sum + Normalize
```

We experiment with different values of B (called batch size) from 16 to 1024 to see how
quality scales with the number of frames used.

---

## Code Walkthrough

**`naive_reconstruction.py`** runs the entire experiment end-to-end. Here is what each
part does:

**`rmse`, `psnr`, `ssim_score`**
Three standard image quality metrics. RMSE measures raw pixel error. PSNR converts that
into decibels — a more interpretable scale where higher means better. SSIM measures
perceptual similarity (structure, contrast, luminance) rather than just per-pixel difference.
All three are computed against the ground truth PNG for each reconstruction.

**`reconstruct(npy_path, batch_size)`**
The core function. It memory-maps the `.npy` file so the full gigabyte-scale data never
loads into RAM. It then slices the last `batch_size` frames, calls `np.unpackbits` to
reverse the bit-packing, sums the photon counts across all frames for each pixel, and
normalizes the result to `[0, 1]`. The output is a float32 image ready for metric
computation or visualization.

**`evaluate_scene(npy_path, gt_path)`**
Loops over every batch size in `BATCH_SIZES`, calls `reconstruct` for each, computes all
three metrics against the ground truth, and stores everything in a dictionary. Also prints
a live progress line per batch so you can see results as they come in.

**`save_comparison(metrics, recons, gt, scene_name)`**
Picks the worst reconstruction (highest RMSE) and the best (highest SSIM) and plots them
side by side with the ground truth in a 3-panel figure. Saves it as a PNG to `results/`.
This is the main visual for the README — it makes it immediately obvious how much quality
difference there is between low and high frame counts.

**`save_metric_curves(all_metrics)`**
Plots PSNR and SSIM against batch size for all three scenes on a log-scale x-axis. This
chart tells the story of where diminishing returns kick in — it's the most important
analytical output of this phase.

**`print_summary(all_metrics)`**
Prints a clean table to the terminal with all metrics across all scenes and batch sizes.
Also saves everything to `results/metrics.json` for reference in later phases.

---

## Running It

```bash
pip install numpy matplotlib imageio scikit-image
python naive_reconstruction.py
```

Before running, update the `SCENES` dictionary at the top of the script to point to your
local dataset paths. Everything else runs automatically.

---

## Output Files

After running, the `results/` folder will contain:

| File | What it shows |
|------|--------------|
| `comparison_bathroom1.png` | Ground Truth vs Worst vs Best — bathroom scene |
| `comparison_attic.png` | Ground Truth vs Worst vs Best — attic scene |
| `comparison_bedroom1.png` | Ground Truth vs Worst vs Best — bedroom scene |
| `metric_curves.png` | PSNR and SSIM vs batch size across all three scenes |
| `metrics.json` | All numeric results in JSON format |

---

## Results

### bathroom1

| Batch Size | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|:----------:|:------:|:------:|:------:|
| 16  | 0.4957 |  6.10 dB | 0.0459 |
| 32  | 0.4379 |  7.17 dB | 0.0641 |
| 64  | 0.3714 |  8.60 dB | 0.0905 |
| 128 | 0.3137 | 10.07 dB | 0.1339 |
| 256 | 0.2591 | 11.73 dB | 0.1975 |
| 512 | 0.2181 | 13.23 dB | 0.2904 |
| **1024** | **0.1723** | **15.27 dB** | **0.3910** |

### attic

| Batch Size | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|:----------:|:------:|:------:|:------:|
| 16  | 0.0982 | 20.16 dB | 0.2977 |
| 32  | 0.0923 | 20.69 dB | 0.3243 |
| 64  | 0.0856 | 21.35 dB | 0.3519 |
| 128 | 0.0847 | 21.44 dB | 0.3815 |
| **256** | **0.0836** | **21.55 dB** | 0.4142 |
| 512 | 0.0880 | 21.11 dB | 0.4333 |
| 1024 | 0.0922 | 20.71 dB | **0.4547** |

### bedroom1

| Batch Size | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|:----------:|:------:|:------:|:------:|
| 16  | 0.3485 |  9.16 dB | 0.0770 |
| 32  | 0.3112 | 10.14 dB | 0.1049 |
| 64  | 0.2861 | 10.87 dB | 0.1373 |
| 128 | 0.2713 | 11.33 dB | 0.1763 |
| **256** | **0.2511** | **12.00 dB** | 0.2287 |
| 512 | 0.2517 | 11.98 dB | 0.2868 |
| 1024 | 0.2678 | 11.44 dB | **0.3627** |

---

## Visual Results

![Metric curves](results/metric_curves.png)

| bathroom1 | attic | bedroom1 |
|:---------:|:-----:|:--------:|
| ![](results/comparison_bathroom1.png) | ![](results/comparison_attic.png) | ![](results/comparison_bedroom1.png) |

---

## Observations

**More frames always helps structurally, but not always metrically.**
SSIM climbs consistently across all three scenes as batch size increases — the structure
and contrast of the image genuinely improve with more frames. But PSNR tells a different
story depending on the scene.

**Scene brightness matters a lot.**
`bathroom1` is a relatively dark indoor scene — PSNR improves steadily all the way to
batch=1024, because it needs many frames just to get enough signal above the noise floor.
`attic` and `bedroom1` both peak in PSNR around batch=256, then actually decline slightly
at 512 and 1024. In brighter or more uniformly lit scenes, accumulating too many frames
causes bright regions to dominate the normalization, which can hurt per-pixel accuracy
even as the overall structure looks better to the eye.

**The gap between PSNR and SSIM at high batch sizes is meaningful.**
At batch=1024, SSIM is still climbing while PSNR has levelled off or dropped. This tells
us that the reconstructions are becoming more structurally similar to the ground truth
(SSIM), but the pixel-level intensity values are drifting (PSNR). Summation recovers
scene structure well but cannot reproduce precise intensities — which is exactly the
limitation a learned model in Phase 2 is designed to fix.

**The ceiling is low.**
Even at best, PSNR stays under 21.6 dB and SSIM under 0.46 across all scenes. Fine edges,
textures, and low-frequency colour gradients remain blurry regardless of how many frames
are used. This is the fundamental limit of summation without any learned prior — and it
sets a clear, honest target for the CNN in Phase 2.

---

← [Back to main repo](../README.md) | [Phase 2 →](../phase2_baseline_cnn/README.md)
