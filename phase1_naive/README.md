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
| `print_summary(all_metrics)` | Prints metric table to terminal and saves `metrics.json` |

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

### Common evaluation scenes (000015, 000023, 000030)

Used across all phases for direct comparison. Numbers shown at best batch size per scene.

| Scene | Best Batch | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|:-----:|:----------:|:------:|:------:|:------:|
| 000015 | 128  | 0.2929 | 10.67 dB | 0.1409 |
| 000023 | 512  | 0.2349 | 12.58 dB | 0.3174 |
| 000030 | 1024 | 0.1460 | 16.71 dB | 0.3765 |
| **Avg** | | | **13.32 dB** | **0.2783** |

![Metric curves — new scenes](results/metric_curves_new_samples.png)

| 000015 | 000023 | 000030 |
|:------:|:------:|:------:|
| ![](results/comparison_000015.png) | ![](results/comparison_000023.png) | ![](results/comparison_000030.png) |

---

### Additional scenes (bathroom1, attic, bedroom1)

| Scene | Best Batch | PSNR ↑ | SSIM ↑ |
|:-----:|:----------:|:------:|:------:|
| bathroom1 | 1024 | 15.27 dB | 0.3910 |
| attic     | 256  | 21.55 dB | 0.4142 |
| bedroom1  | 256  | 12.00 dB | 0.2287 |

![Metric curves — original scenes](results/metric_curves.png)

| bathroom1 | attic | bedroom1 |
|:---------:|:-----:|:--------:|
| ![](results/comparison_bathroom1.png) | ![](results/comparison_attic.png) | ![](results/comparison_bedroom1.png) |

---

## Observations

**More frames helps structurally but not always metrically.** SSIM climbs consistently
across all scenes. PSNR tells a different story — dark scenes (bathroom1, bedroom1)
improve steadily with more frames, while scenes with large bright regions (attic, 000023)
peak around batch=256–512 then decline slightly. Too many frames causes bright areas to
dominate normalization, hurting per-pixel accuracy even as structure improves.

**The ceiling is low.** Best PSNR stays under 21.6 dB on original scenes and under
16.8 dB on the harder new scenes. Edges, textures, and fine detail remain blurry
regardless of frame count — the fundamental limit of summation without a learned prior.

---

← [Back](../README.md) | [Phase 2 →](../phase2_baseline_cnn/README.md)
