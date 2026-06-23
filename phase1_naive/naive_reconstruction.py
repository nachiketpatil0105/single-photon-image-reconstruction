import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import imageio.v3 as iio
from skimage.metrics import structural_similarity as ssim


# Scenes to evaluate — update paths if your dataset is in a different location
SCENES = {
    "bathroom1": {
        "npy": "/kaggle/input/spc-data/train/bathroom1/000015.npy",
        "gt":  "/kaggle/input/spc-data/train/bathroom1/000015.png",
    },
    "attic": {
        "npy": "/kaggle/input/spc-data/train/attic/000015.npy",
        "gt":  "/kaggle/input/spc-data/train/attic/000015.png",
    },
    "bedroom1": {
        "npy": "/kaggle/input/spc-data/train/bedroom1/000015.npy",
        "gt":  "/kaggle/input/spc-data/train/bedroom1/000015.png",
    },
}

# Batch sizes to sweep — how many photon frames to sum
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]

os.makedirs("results", exist_ok=True)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

def psnr(pred, gt):
    return 20 * np.log10(1.0 / (rmse(pred, gt) + 1e-8))

def ssim_score(pred, gt):
    return ssim(pred, gt, channel_axis=2, data_range=1.0)


# ------------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------------

def reconstruct(npy_path, batch_size):
    """
    Load SPC data and reconstruct an image by summing photon frames.

    The .npy file stores bit-packed photon counts with shape (1024, H, W, 100, 3).
    We memory-map it to avoid loading the full file into RAM, slice the last
    `batch_size` frames, unpack the bits, sum across frames, and normalize.
    """
    data = np.load(npy_path, mmap_mode="r")
    data = data[-batch_size:]                   # take the last B frames
    data = np.unpackbits(data, axis=2)          # unpack bit-packed bins
    img  = data.sum(axis=0).astype(np.float32)  # sum photon counts per pixel
    img /= img.max()                             # normalize to [0, 1]
    return img


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_scene(npy_path, gt_path):
    gt = iio.imread(gt_path).astype(np.float32) / 255.0

    metrics = {}
    recons  = {}

    for b in BATCH_SIZES:
        img = reconstruct(npy_path, b)
        recons[b] = img
        metrics[b] = {
            "RMSE": float(rmse(img, gt)),
            "PSNR": float(psnr(img, gt)),
            "SSIM": float(ssim_score(img, gt)),
        }
        print(f"  batch={b:>4d}  "
              f"RMSE={metrics[b]['RMSE']:.4f}  "
              f"PSNR={metrics[b]['PSNR']:.2f} dB  "
              f"SSIM={metrics[b]['SSIM']:.4f}")

    return metrics, recons, gt


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def save_comparison(metrics, recons, gt, scene_name):
    """
    3-panel figure: Ground Truth | Worst | Best reconstruction.
    'Worst' = highest RMSE, 'Best' = highest SSIM.
    """
    worst_b = max(metrics, key=lambda b: metrics[b]["RMSE"])
    best_b  = max(metrics, key=lambda b: metrics[b]["SSIM"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Naive Summation — {scene_name}", fontsize=13, fontweight="bold")

    panels = [
        (gt,              "Ground Truth",       ""),
        (recons[worst_b], f"Worst  (batch={worst_b})",
         f"RMSE={metrics[worst_b]['RMSE']:.4f}  "
         f"PSNR={metrics[worst_b]['PSNR']:.2f} dB  "
         f"SSIM={metrics[worst_b]['SSIM']:.4f}"),
        (recons[best_b],  f"Best   (batch={best_b})",
         f"RMSE={metrics[best_b]['RMSE']:.4f}  "
         f"PSNR={metrics[best_b]['PSNR']:.2f} dB  "
         f"SSIM={metrics[best_b]['SSIM']:.4f}"),
    ]

    for ax, (img, title, subtitle) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    path = f"results/comparison_{scene_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


def save_metric_curves(all_metrics):
    """
    PSNR and SSIM vs batch size — one curve per scene.
    Useful for seeing where diminishing returns kick in.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Naive Summation: Metric vs Batch Size", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10.colors

    for i, (scene, metrics) in enumerate(all_metrics.items()):
        psnr_vals = [metrics[b]["PSNR"] for b in BATCH_SIZES]
        ssim_vals = [metrics[b]["SSIM"] for b in BATCH_SIZES]
        c = colors[i % len(colors)]
        ax1.plot(BATCH_SIZES, psnr_vals, marker="o", label=scene, color=c)
        ax2.plot(BATCH_SIZES, ssim_vals, marker="o", label=scene, color=c)

    for ax, ylabel in [(ax1, "PSNR (dB)"), (ax2, "SSIM")]:
        ax.set_xlabel("Batch Size (frames)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/metric_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


def print_summary(all_metrics):
    print(f"\n{'Scene':<12} {'Batch':>6} | {'RMSE':>7} | {'PSNR (dB)':>9} | {'SSIM':>7}")
    print("─" * 52)
    for scene, metrics in all_metrics.items():
        for b in BATCH_SIZES:
            m = metrics[b]
            print(f"{scene:<12} {b:>6} | "
                  f"{m['RMSE']:>7.4f} | "
                  f"{m['PSNR']:>9.2f} | "
                  f"{m['SSIM']:>7.4f}")
        print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    all_metrics = {}

    for scene_name, paths in SCENES.items():
        print(f"\n{'─' * 52}")
        print(f"Scene: {scene_name}")
        print(f"{'─' * 52}")

        metrics, recons, gt = evaluate_scene(paths["npy"], paths["gt"])
        all_metrics[scene_name] = metrics

        save_comparison(metrics, recons, gt, scene_name)

    save_metric_curves(all_metrics)
    print_summary(all_metrics)

    with open("results/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Metrics saved → results/metrics.json")
