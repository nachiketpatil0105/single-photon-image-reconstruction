import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

os.makedirs("results", exist_ok=True)


# Dataset paths — update this to match your environment
DATA_ROOT = "/kaggle/input/datasets/nachiketpatil0105/spc-data/train/"

# Training config
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 50
NUM_FRAMES    = 128   # how many SPC frames to use per sample


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def collect_paths(root):
    path = Path(root)
    npy_files = sorted(path.rglob("*.npy"))
    png_files = sorted(path.rglob("*.png"))
    return npy_files, png_files


def load_input(npy_path):
    """
    Load the last NUM_FRAMES binary photon frames from an SPC .npy file
    and reshape them into a (384, 800, 800) tensor ready for the CNN.

    The raw file has shape (1024, H, W, 100, 3) — 1024 frames of bit-packed
    spatial-temporal data across 3 channels. We take the last 128 frames,
    unpack the bits to recover the temporal bins, then flatten the frame and
    channel dimensions together so each pixel has 128 × 3 = 384 input features.
    PyTorch expects channels first, so the final shape is (384, 800, 800).
    """
    data = np.load(npy_path, mmap_mode="r")[-NUM_FRAMES:]  # (128, 800, 100, 3)
    data = np.unpackbits(data, axis=2)                      # (128, 800, 800, 3)
    data = data.reshape(NUM_FRAMES, 800, 800, 3)
    data = np.transpose(data, (1, 2, 0, 3))                 # (800, 800, 128, 3)
    data = data.reshape(800, 800, NUM_FRAMES * 3)           # (800, 800, 384)
    data = data / data.max()
    data = data.transpose(2, 0, 1)                          # (384, 800, 800)
    return data.astype(np.float32)


def load_target(png_path):
    img = np.array(Image.open(png_path)).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)   # (3, 800, 800)


def to_tensor(arr, device):
    return torch.from_numpy(arr).float().unsqueeze(0).to(device)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class SimCNN(nn.Module):
    """
    A simple, flat convolutional network for single-photon image reconstruction.

    The network takes 384 input channels (128 SPC frames × 3 RGB channels) and
    progressively compresses them down to 3 output channels (RGB image) using
    eight 3×3 conv layers with padding=1 to preserve spatial resolution throughout.

    The channel progression (384 → 128 → 256 → 256 → 128 → 64 → 32 → 16 → 3)
    first expands to build a richer feature representation, then contracts back
    to produce the final RGB reconstruction. Sigmoid at the end keeps output in [0, 1].
    """
    def __init__(self):
        super(SimCNN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128,  64, 3, padding=1), nn.ReLU(),
            nn.Conv2d( 64,  32, 3, padding=1), nn.ReLU(),
            nn.Conv2d( 32,  16, 3, padding=1), nn.ReLU(),
            nn.Conv2d( 16,   3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(model, train_X, train_y, optimizer, ssim_fn, l1_fn, device):
    model.train()
    epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        for npy_path, png_path in zip(train_X, train_y):
            x      = to_tensor(load_input(npy_path), device)
            target = to_tensor(load_target(png_path), device)

            pred = model(x)

            # Combined loss: L1 keeps pixel accuracy, SSIM preserves structure
            ssim_loss = 1 - ssim_fn(pred, target)
            l1        = l1_fn(pred, target)
            loss      = 0.5 * ssim_loss + 0.5 * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_X)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Loss: {avg_loss:.4f}")

    return epoch_losses


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(model, test_X, test_y, device):
    """
    Run inference on the test set and compute PSNR and SSIM for each sample.
    Also saves a 3-panel comparison figure (Input | Prediction | Ground Truth)
    for each test image, plus a JSON file with all numeric results.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for i, (npy_path, png_path) in enumerate(zip(test_X, test_y)):
            x      = to_tensor(load_input(npy_path), device)
            target = to_tensor(load_target(png_path), device)

            pred = model(x)

            # Move to numpy for metric computation and plotting
            pred_np   = np.clip(pred.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)
            target_np = target.squeeze().cpu().numpy().transpose(1, 2, 0)
            input_np  = x.squeeze().cpu().numpy()[:3].transpose(1, 2, 0)  # show first 3 ch

            p = psnr_metric(target_np, pred_np, data_range=1.0)
            s = ssim_metric(target_np, pred_np, channel_axis=2, data_range=1.0)

            results.append({
                "sample": i,
                "scene":  Path(png_path).parent.name,
                "PSNR":   float(p),
                "SSIM":   float(s),
            })

            print(f"  Sample {i}  ({Path(png_path).parent.name})  "
                  f"PSNR={p:.2f} dB  SSIM={s:.4f}")

            save_comparison(input_np, pred_np, target_np, i,
                            Path(png_path).parent.name, p, s)

    return results


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def save_comparison(input_img, pred_img, target_img, idx, scene_name, p, s):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"CNN Reconstruction — {scene_name}", fontsize=13, fontweight="bold")

    panels = [
        (input_img,  "SPC Input (first 3 ch)", ""),
        (pred_img,   "CNN Prediction",          f"PSNR={p:.2f} dB  SSIM={s:.4f}"),
        (target_img, "Ground Truth",            ""),
    ]

    for ax, (img, title, subtitle) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    path = f"results/comparison_sample{idx}_{scene_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


def save_loss_curve(epoch_losses):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (0.5 × L1 + 0.5 × (1 - SSIM))")
    ax.set_title("Training Loss Curve — Baseline CNN", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "results/loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


def print_summary(results):
    print(f"\n{'Sample':<8} {'Scene':<14} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("─" * 45)
    for r in results:
        print(f"{r['sample']:<8} {r['scene']:<14} {r['PSNR']:>10.2f} {r['SSIM']:>8.4f}")

    avg_psnr = sum(r["PSNR"] for r in results) / len(results)
    avg_ssim = sum(r["SSIM"] for r in results) / len(results)
    print("─" * 45)
    print(f"{'Average':<8} {'':<14} {avg_psnr:>10.2f} {avg_ssim:>8.4f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    npy_files, png_files = collect_paths(DATA_ROOT)

    train_X, train_y = npy_files[:45], png_files[:45]
    test_X,  test_y  = npy_files[-5:], png_files[-5:]

    print(f"Train: {len(train_X)} samples  |  Test: {len(test_X)} samples\n")

    model     = SimCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ssim_fn   = SSIM(data_range=1.0).to(device)
    l1_fn     = nn.L1Loss()

    print("Training...")
    epoch_losses = train(model, train_X, train_y, optimizer, ssim_fn, l1_fn, device)
    save_loss_curve(epoch_losses)

    torch.save(model.state_dict(), "results/cnn_model.pth")
    print("\nModel saved → results/cnn_model.pth")

    print("\nEvaluating on test set...")
    results = evaluate(model, test_X, test_y, device)
    print_summary(results)

    with open("results/metrics.json", "w") as f:
        json.dump({
            "test_results": results,
            "epoch_losses": epoch_losses,
            "avg_psnr": sum(r["PSNR"] for r in results) / len(results),
            "avg_ssim": sum(r["SSIM"] for r in results) / len(results),
        }, f, indent=2)
    print("\nMetrics saved → results/metrics.json")
