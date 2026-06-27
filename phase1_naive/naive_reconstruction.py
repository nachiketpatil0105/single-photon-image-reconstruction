import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import imageio.v3 as iio
from skimage.metrics import structural_similarity as ssim

# Update paths to match your dataset location
SCENES = {
    "bathroom1": {
        "npy": "/kaggle/input/spc-data/train/bathroom1/000015.npy",
        "gt": "/kaggle/input/spc-data/train/bathroom1/000015.png",
    },
    "attic": {
        "npy": "/kaggle/input/spc-data/train/attic/000015.npy",
        "gt": "/kaggle/input/spc-data/train/attic/000015.png",
    },
    "bedroom1": {
        "npy": "/kaggle/input/spc-data/train/bedroom1/000015.npy",
        "gt": "/kaggle/input/spc-data/train/bedroom1/000015.png",
    },
}

BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]

os.makedirs("results", exist_ok=True)


# Root Mean Squared Error
def rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))


# Peak Signal-to-Noise Ratio
def psnr(pred, gt):
    return 20 * np.log10(1.0 / (rmse(pred, gt) + 1e-8))


# Structural Similarity Index
def ssim_metric(pred, gt):
    return ssim(pred, gt, channel_axis=2, data_range=1.0)


def load_and_unpack(npy_path, batch_size):
    # Memory-map to avoid loading the full file into RAM
    pc = np.load(npy_path, mmap_mode="r")
    pc = pc[-batch_size:]
    pc = np.unpackbits(pc, axis=2)
    img = pc.sum(axis=0)
    img = img / img.max()
    return img.astype(np.float32)


def evaluate_batch_sizes(npy_path, gt_path, batch_sizes):
    gt = iio.imread(gt_path).astype(np.float32) / 255.0
    results = {}
    recon_images = {}
    for b in batch_sizes:
        img = load_and_unpack(npy_path, b)
        recon_images[b] = img
        results[b] = {
            "RMSE": rmse(img, gt),
            "PSNR": psnr(img, gt),
            "SSIM": ssim_metric(img, gt),
        }
    return results, recon_images, gt


def show_comparisons(results, recon_images, gt, scene_name, save_path=None):
    # Worst by highest RMSE, best by highest SSIM
    worst_b = max(results, key=lambda b: results[b]["RMSE"])
    best_b = max(results, key=lambda b: results[b]["SSIM"])

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Naive Summation — {scene_name}", fontsize=13, fontweight="bold")

    plt.subplot(1, 3, 1)
    plt.imshow(gt)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(recon_images[worst_b])
    plt.title(f"Worst (batch={worst_b})\nRMSE={results[worst_b]['RMSE']:.3f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(recon_images[best_b])
    plt.title(f"Best (batch={best_b})\nSSIM={results[best_b]['SSIM']:.3f}")
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def save_metric_curves(all_metrics):
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
    print(f"Saved → {path}")
    plt.close()


def print_metrics(results):
    print("\nBatch Size | RMSE ↓ | PSNR ↑ | SSIM ↑")
    print("-" * 40)
    for b, m in results.items():
        print(f"{b:10d} | {m['RMSE']:.4f} | {m['PSNR']:.2f} | {m['SSIM']:.4f}")


if __name__ == "__main__":
    all_metrics = {}

    for scene_name, paths in SCENES.items():
        print(f"\n{'─' * 52}")
        print(f"Scene: {scene_name}")
        print(f"{'─' * 52}")

        results, recon_images, gt = evaluate_batch_sizes(
            paths["npy"], paths["gt"], BATCH_SIZES
        )
        all_metrics[scene_name] = results

        print_metrics(results)
        show_comparisons(
            results, recon_images, gt, scene_name,
            save_path=f"results/comparison_{scene_name}.png"
        )

    save_metric_curves(all_metrics)

    with open("results/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Metrics saved → results/metrics.json")