# Phase 1 — Naive Summation Baseline

← [Back](../README.md) | [Phase 2 →](../phase2_baseline_cnn/README.md)

No learning, no parameters. Binary photon frames are summed and normalized to recover
approximate scene intensity. We sweep batch sizes from 16 to 1024 to find where
diminishing returns kick in. This sets the performance floor for all subsequent models.

---

## How It Works

SPC files are bit-packed with shape `(1024, H, W, 100, 3)`. We take the last B frames,
unpack the bits (`np.unpackbits` on axis=2, giving 100×8=800 spatial bins), sum across
frames, and normalize.

```
.npy (1024, 800, 100, 3)
  → slice last B frames    (B, 800, 100, 3)
  → unpackbits axis=2      (B, 800, 800, 3)
  → sum across frames      (800, 800, 3)
  → normalize to [0, 1]
```

---

## Code

**`naive_reconstruction.py`**

| Function | What it does |
|----------|-------------|
| `reconstruct(npy_path, batch_size)` | Loads, unpacks, sums, and normalizes one scene at a given batch size |
| `evaluate_scene(npy_path, gt_path)` | Runs reconstruction for all batch sizes and computes RMSE, PSNR, SSIM |
| `save_comparison(...)` | 3-panel figure: Ground Truth \| Worst \| Best reconstruction |
| `save_metric_curves(all_metrics)` | PSNR and SSIM vs batch size across all scenes |
| `print_summary(all_metrics)` | Prints a metric table to terminal and saves `metrics.json` |

---

## Running

```bash
pip install numpy matplotlib imageio scikit-image
python naive_reconstruction.py
```

Update the `SCENES` dict at the top of the script with your dataset paths before running.

**Output** → `results/comparison_{scene}.png`, `results/metric_curves.png`, `results/metrics.json`

---

## Results

### bathroom1

| Batch | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|:-----:|:------:|:------:|:------:|
| 16 | 0.4957 | 6.10 dB | 0.0459 |
| 64 | 0.3714 | 8.60 dB | 0.0905 |
| 256 | 0.2591 | 11.73 dB | 0.1975 |
| 512 | 0.2181 | 13.23 dB | 0.2904 |
| **1024** | **0.1723** | **15.27 dB** | **0.3910** |

### attic

| Batch | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|:-----:|:------:|:------:|:------:|
| 16 | 0.0982 | 20.16 dB | 0.2977 |
| 64 | 0.0856 | 21.35 dB | 0.3519 |
| **256** | **0.0836** | **21.55 dB** | 0.4142 |
| 512 | 0.0880 | 21.11 dB | 0.4333 |
| 1024 | 0.0922 | 20.71 dB | **0.4547** |

### bedroom1

| Batch | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|:-----:|:------:|:------:|:------:|
| 16 | 0.3485 | 9.16 dB | 0.0770 |
| 64 | 0.2861 | 10.87 dB | 0.1373 |
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

**More frames helps structurally but not always metrically.** SSIM climbs consistently
across all scenes. PSNR tells a different story — `bathroom1` (a dark scene) improves all
the way to batch=1024, while `attic` and `bedroom1` peak around batch=256 and then
decline slightly. In brighter scenes, accumulating too many frames causes bright regions
to dominate normalization, hurting per-pixel accuracy even as structure improves.

**The ceiling is low.** Best PSNR stays under 21.6 dB, SSIM under 0.46. Edges, textures,
and fine detail remain blurry regardless of frame count. This is the fundamental limit of
summation without a learned prior, and the target Phase 2 onwards must beat.

---

← [Back](../README.md) | [Phase 2 →](../phase2_baseline_cnn/README.md)

---

## Results on Common Evaluation Scenes

These 3 scenes are used across all phases for direct comparison.

| Scene | Best Batch | PSNR ↑ | SSIM ↑ |
|:-----:|:----------:|:------:|:------:|
| 000015 | 128 | 10.67 dB | 0.1409 |
| 000023 | 512 | 12.58 dB | 0.3174 |
| 000030 | 1024 | 16.71 dB | 0.3765 |
| **Avg** | | **13.32 dB** | **0.2783** |

![Metric curves — new scenes](results/metric_curves_new_samples.png)

| 000015 | 000023 | 000030 |
|:------:|:------:|:------:|
| ![](results/comparison_000015.png) | ![](results/comparison_000023.png) | ![](results/comparison_000030.png) |
