import numpy as np
import csv
import random
import logging
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from torch.utils.tensorboard import SummaryWriter


# ── Paths ─────────────────────────────────────────────────────────────────────
# Update these to your environment
BASE_DIR     = Path("/home/user/spc")
DATA_DIR_IMG = BASE_DIR / "Data_PT"          # ground truth .pt tensors
DATA_DIR_NPY = BASE_DIR / "Preprocessed_PT" # SPC input .pt tensors
OUTPUT_DIR   = BASE_DIR / "output"
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
LOG_DIR      = OUTPUT_DIR / "tensorboard"
LOG_CSV      = OUTPUT_DIR / "training_log.csv"
LOG_FILE     = OUTPUT_DIR / "training.log"

for d in [CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
log.info(f"Using device: {DEVICE}")


# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED           = 42
PATCH_SIZE     = 512
AUG_PROB       = 0.2
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 70
BATCH_SIZE     = 2
NUM_WORKERS    = 12
GRAD_CLIP_NORM = 1.0

# Loss weights
W_CHARB    = 0.5
W_MSSSIM   = 0.3
W_VGG      = 0.1
W_EDGE     = 0.1

# Deep supervision weight — starts at 0.4, annealed to 0 across training.
# Prevents the aux head from competing with the main head late in training.
W_DEEP_SUP = 0.4

# Cosine annealing with warm restarts: restart every T_0 epochs
COSINE_T0  = 5
ETA_MIN    = 5e-5

random.seed(SEED)


# ── Data ──────────────────────────────────────────────────────────────────────

def collect_paths(npy_root, img_root):
    """
    Walk paired npy/img folder trees and collect matching .pt file paths.
    Folders named train_* are expected in both roots. Any folder with
    no .pt files inside is skipped silently.
    """
    npy_paths, img_paths = [], []
    for npy_folder, img_folder in zip(
        sorted(npy_root.glob("train_*")),
        sorted(img_root.glob("train_*")),
    ):
        npys = sorted(npy_folder.rglob("*.pt"))
        imgs = sorted(img_folder.rglob("*.pt"))
        if len(npys) == 0:
            continue
        npy_paths.extend(npys)
        img_paths.extend(imgs)
    return npy_paths, img_paths


class SPCDataset(Dataset):
    """
    Dataset for SPC reconstruction using preprocessed .pt tensors.

    Unlike Phases 2 and 3, data here has been preprocessed into .pt tensors
    offline so loading is fast. Each .pt file is a ready-to-use tensor
    (already reshaped and normalised). We slice the last 96 channels at
    load time to use 96 frames rather than the full set.

    Augmentation is done on GPU in the training loop (gpu_augment) rather
    than here, since large tensors augment faster on GPU.
    """

    def __init__(self, npy_paths, img_paths):
        self.npy_paths = npy_paths
        self.img_paths = img_paths

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        x      = torch.load(self.npy_paths[idx], weights_only=True)
        target = torch.load(self.img_paths[idx], weights_only=True)
        x = x[-96:, :, :]   # use last 96 frames
        return x, target


def gpu_augment(x, y):
    """
    Apply random flips and 90° rotations on GPU.
    Done after moving to device to avoid CPU→GPU overhead per augmented sample.
    Both x and y receive identical transforms to preserve alignment.
    """
    if torch.rand(1).item() < AUG_PROB:
        x = torch.flip(x, dims=[3]); y = torch.flip(y, dims=[3])
    if torch.rand(1).item() < AUG_PROB:
        x = torch.flip(x, dims=[2]); y = torch.flip(y, dims=[2])
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        x = torch.rot90(x, k, dims=[2, 3])
        y = torch.rot90(y, k, dims=[2, 3])
    return x, y


# ── Loss functions ────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    """
    Smooth L1 replacement. sqrt((pred - target)^2 + eps^2) is differentiable
    everywhere — including at zero — which stabilises gradients at late training
    when per-pixel errors are very small and plain L1 gradients become noisy.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using feature maps from a frozen pretrained VGG16.

    Rather than comparing pixels directly, we compare how prediction and
    ground truth look to a pretrained feature extractor at three depths:
      relu1_2 (fine textures)    weight 0.2
      relu2_2 (mid-level detail) weight 0.3
      relu3_3 (structural)       weight 0.5

    VGG weights are frozen — this is not trained.
    """

    LAYER_WEIGHTS = {"relu1_2": 0.2, "relu2_2": 0.3, "relu3_3": 0.5}

    def __init__(self):
        super().__init__()
        vgg      = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())
        self.slice1 = nn.Sequential(*features[:5]).eval()
        self.slice2 = nn.Sequential(*features[5:10]).eval()
        self.slice3 = nn.Sequential(*features[10:17]).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.l1 = nn.L1Loss()
        self.w  = self.LAYER_WEIGHTS

    def forward(self, pred, target):
        f1_p = self.slice1(pred);   f1_t = self.slice1(target)
        f2_p = self.slice2(f1_p);   f2_t = self.slice2(f1_t)
        f3_p = self.slice3(f2_p);   f3_t = self.slice3(f2_t)
        return (self.w["relu1_2"] * self.l1(f1_p, f1_t) +
                self.w["relu2_2"] * self.l1(f2_p, f2_t) +
                self.w["relu3_3"] * self.l1(f3_p, f3_t))


class EdgeLoss(nn.Module):
    """
    Penalises errors in horizontal and vertical image gradients.

    Single-photon reconstruction often loses sharp edges — this loss
    directly encourages the model to preserve them by comparing finite
    differences between adjacent pixels in both directions.
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        def gradient(x):
            dx = x[:, :, :, :-1] - x[:, :, :, 1:]
            dy = x[:, :, :-1, :] - x[:, :, 1:, :]
            return dx, dy
        pred_dx, pred_dy = gradient(pred)
        tgt_dx,  tgt_dy  = gradient(target)
        return self.l1(pred_dx, tgt_dx) + self.l1(pred_dy, tgt_dy)


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Two conv layers with GroupNorm + SiLU activation and a residual projection.

    The residual path (1×1 conv when in_ch != out_ch, else identity) adds the
    input directly to the block output. This makes each block learn a residual
    correction rather than a full transformation, which helps gradient flow
    and speeds up convergence in deep networks.

    GroupNorm with 8 groups is used instead of BatchNorm because batch size 2
    makes BatchNorm statistics unreliable. SiLU (Sigmoid Linear Unit) replaces
    ReLU for smoother gradients.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.act = nn.SiLU(inplace=True)
        self.res = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.block(x) + self.res(x))


class GuidedAttentionGate(nn.Module):
    """
    Attention gate using both the encoder skip and the decoder feature (Oktay et al., 2018).

    The key difference from simple self-gating (skip * sigmoid(conv(skip))) is that
    the gate is informed by TWO signals:
      - skip: the encoder feature at this level (WHERE useful information is)
      - gate: the upsampled decoder feature (WHAT to look for, based on global context)

    Both are projected to an intermediate space, added, activated, then a 1×1 conv
    produces a scalar attention map alpha in [0,1] for each spatial position.
    The skip is multiplied by alpha — positions with low attention are suppressed,
    high-attention positions pass through unchanged.
    """

    def __init__(self, skip_ch, gate_ch, inter_ch=None):
        super().__init__()
        inter_ch    = inter_ch or max(skip_ch // 2, 8)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.psi    = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=True), nn.Sigmoid())
        self.norm   = nn.GroupNorm(1, inter_ch)
        self.act    = nn.ReLU(inplace=True)

    def forward(self, skip, gate):
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        combined = self.act(self.norm(self.W_skip(skip) + self.W_gate(gate)))
        alpha    = self.psi(combined)   # (B, 1, H, W) attention map
        return skip * alpha             # broadcast over channel dim


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Decoder block: upsample → attend skip → concatenate → ConvBlock.

    The attention gate filters the encoder skip before concatenation,
    so the decoder only receives spatially relevant encoder features
    rather than the full unfiltered skip connection.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, skip_ch, 2, stride=2)
        self.attn = GuidedAttentionGate(skip_ch=skip_ch, gate_ch=skip_ch)
        self.conv = ConvBlock(skip_ch * 2, out_ch)

    def forward(self, x, skip):
        x_up       = self.up(x)
        skip_gated = self.attn(skip, gate=x_up)
        if x_up.shape[2:] != skip_gated.shape[2:]:
            x_up = F.interpolate(x_up, size=skip_gated.shape[2:],
                                 mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x_up, skip_gated], dim=1))


class ResUNetAttention(nn.Module):
    """
    ResUNet with Guided Attention Gates for single-photon image reconstruction.

    Built on top of the UNet from Phase 3, with four key improvements:

    1. Residual ConvBlocks — every conv block learns a correction over its
       input rather than a full transformation (GroupNorm + SiLU + residual path).

    2. 5-level encoder/decoder — one extra downsampling level gives the bottleneck
       access to a much larger receptive field (32×32 at 512 input) which is
       critical for understanding global scene layout.

    3. Double bottleneck — two stacked ConvBlocks at the smallest spatial scale
       (1024→2048→2048) maximises reasoning capacity where spatial resolution
       is cheapest but semantic content is richest.

    4. Guided attention gates on every skip connection — the decoder tells each
       gate what to look for; the encoder tells it where. This replaces the raw
       concatenation from Phase 3.

    5. Deep supervision head — a lightweight auxiliary output at the dec2 level
       (256×256) forces mid-level features to stay semantically meaningful.
       Annealed to zero weight by the end of training.

    Channel / resolution progression:
        Input proj : 96  → 128
        enc1       : 128 → 128   (512×512)
        enc2 ↓2    : 128 → 256   (256×256)
        enc3 ↓2    : 256 → 512   (128×128)
        enc4 ↓2    : 512 → 1024  ( 64×64)
        enc5 ↓2    : 1024→ 1024  ( 32×32)
        bottleneck : 1024→2048→2048
        dec0 ↑2    : 2048→ 1024  ( 64×64)
        dec1 ↑2    : 1024→  512  (128×128)
        dec2 ↑2    : 512 →  256  (256×256)  ← aux head here
        dec3 ↑2    : 256 →  128  (512×512)
        dec4 ↑2    : 128 →  128  (512×512)
        out        : 128 →    3
    """

    def __init__(self):
        super().__init__()

        self.input_proj = nn.Conv2d(96, 128, kernel_size=1)

        # Encoder
        self.enc1 = ConvBlock(128,  128)
        self.enc2 = Down(128,  256)
        self.enc3 = Down(256,  512)
        self.enc4 = Down(512,  1024)
        self.enc5 = Down(1024, 1024)

        # Double bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(1024, 2048),
            ConvBlock(2048, 2048),
        )

        # Decoder with guided attention at every skip
        self.dec0 = Up(2048, 1024, 1024)
        self.dec1 = Up(1024, 1024,  512)
        self.dec2 = Up(512,   512,  256)
        self.dec3 = Up(256,   256,  128)
        self.dec4 = Up(128,   128,  128)

        # Main output
        self.out = nn.Conv2d(128, 3, kernel_size=1)

        # Auxiliary deep supervision head (dec2, 256ch at 256×256)
        # Only active during training — annealed to zero weight by end.
        self.aux_head = nn.Conv2d(256, 3, kernel_size=1)

    def forward(self, x):
        x  = self.input_proj(x)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)

        d0 = self.dec0(b,  e5)
        d1 = self.dec1(d0, e4)
        d2 = self.dec2(d1, e3)
        d3 = self.dec3(d2, e2)
        d4 = self.dec4(d3, e1)

        main_out = self.out(d4)

        if self.training:
            # Upsample aux output from 256×256 to match target size
            aux_out = F.interpolate(
                self.aux_head(d2),
                size=main_out.shape[2:],
                mode="bilinear", align_corners=False,
            )
            return main_out, aux_out

        return main_out


# ── Metric ────────────────────────────────────────────────────────────────────

def compute_psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * np.log10((data_range ** 2) / mse)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, loader, optimizer, scheduler, scaler,
          charb_fn, msssim_fn, vgg_fn, edge_fn, ssim_fn):

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr", "deep_sup_w",
            "train_loss", "train_charb", "train_msssim_loss",
            "train_vgg", "train_edge", "train_aux",
            "train_ssim", "train_msssim", "train_psnr",
        ])

    writer          = SummaryWriter(log_dir=str(LOG_DIR))
    best_train_ssim = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Deep supervision weight: linear anneal from W_DEEP_SUP → 0
        deep_sup_w = W_DEEP_SUP * max(0.0, 1.0 - epoch / NUM_EPOCHS)

        model.train()
        t_loss = t_charb = t_msssim_l = t_vgg = t_edge = t_aux = 0.0
        t_ssim = t_msssim = t_psnr = 0.0

        for x, target in loader:
            x, target = x.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            x, target = gpu_augment(x, target)

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction, aux_pred = model(x)
                pred_c    = torch.clamp(prediction, 0.0, 1.0)
                aux_pred_c = torch.clamp(aux_pred,  0.0, 1.0)

                charb        = charb_fn(prediction, target)
                msssim_score = msssim_fn(pred_c, target)
                msssim_l     = 1.0 - msssim_score
                vgg          = vgg_fn(pred_c, target)
                edge         = edge_fn(pred_c, target)
                aux_loss     = charb_fn(aux_pred, target)

                loss = (W_CHARB  * charb    +
                        W_MSSSIM * msssim_l +
                        W_VGG    * vgg      +
                        W_EDGE   * edge     +
                        deep_sup_w * aux_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                t_ssim   += ssim_fn(pred_c.float(), target.float()).item()
                t_psnr   += compute_psnr(pred_c.float(), target.float())
                t_msssim += msssim_score.item()

            t_loss     += loss.item()
            t_charb    += charb.item()
            t_msssim_l += msssim_l.item()
            t_vgg      += vgg.item()
            t_edge     += edge.item()
            t_aux      += aux_loss.item()

        scheduler.step(epoch)
        lr = scheduler.get_last_lr()[0]
        n  = len(loader)

        for attr in ["t_loss","t_charb","t_msssim_l","t_vgg","t_edge","t_aux","t_ssim","t_msssim","t_psnr"]:
            exec(f"{attr} /= n")  # noqa

        t_loss/=n; t_charb/=n; t_msssim_l/=n; t_vgg/=n
        t_edge/=n; t_aux/=n;   t_ssim/=n;     t_msssim/=n; t_psnr/=n

        log.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  LR: {lr:.2e}  DS_w: {deep_sup_w:.3f}  "
            f"| Loss: {t_loss:.4f}  Aux: {t_aux:.4f}  "
            f"SSIM: {t_ssim:.4f}  MS-SSIM: {t_msssim:.4f}  PSNR: {t_psnr:.2f} dB"
        )

        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.2e}", f"{deep_sup_w:.4f}",
                f"{t_loss:.4f}", f"{t_charb:.4f}", f"{t_msssim_l:.4f}",
                f"{t_vgg:.4f}",  f"{t_edge:.4f}",  f"{t_aux:.4f}",
                f"{t_ssim:.4f}", f"{t_msssim:.4f}", f"{t_psnr:.2f}",
            ])

        writer.add_scalars("Loss/total",        {"train": t_loss},     epoch)
        writer.add_scalars("Metric/SSIM",       {"train": t_ssim},     epoch)
        writer.add_scalars("Metric/PSNR_dB",    {"train": t_psnr},     epoch)
        writer.add_scalar("Train/lr",            lr,                    epoch)
        writer.add_scalar("Train/deep_sup_w",    deep_sup_w,            epoch)

        if t_ssim > best_train_ssim:
            best_train_ssim = t_ssim
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "best_train_ssim":      best_train_ssim,
            }, CKPT_DIR / "best_model.pth")
            log.info(f"  ✓ New best train SSIM: {best_train_ssim:.4f} — saved best_model.pth")

        if epoch % 10 == 0:
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "best_train_ssim":  best_train_ssim,
            }, CKPT_DIR / f"ckpt_epoch{epoch:03d}.pth")

    writer.close()
    log.info(f"Training complete. Best train SSIM: {best_train_ssim:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_npy, train_img = collect_paths(DATA_DIR_NPY, DATA_DIR_IMG)
    log.info(f"Total training samples: {len(train_npy)}")

    loader = DataLoader(
        SPCDataset(train_npy, train_img),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    log.info(f"Train loader: {len(loader)} batches")

    model     = ResUNetAttention().to(DEVICE)
    model     = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=1, eta_min=ETA_MIN
    )
    scaler = GradScaler("cuda")

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {total_params:,}")

    # Resume from checkpoint if one exists
    ckpt_path = CKPT_DIR / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        log.info(f"Resumed from epoch {ckpt['epoch']}  best SSIM: {ckpt['best_train_ssim']:.4f}")

    charb_fn  = CharbonnierLoss(eps=1e-3)
    vgg_fn    = VGGPerceptualLoss().to(DEVICE)
    edge_fn   = EdgeLoss()
    ssim_fn   = SSIM(data_range=1.0).to(DEVICE)
    msssim_fn = MSSSIM(data_range=1.0, kernel_size=11).to(DEVICE)

    train(model, loader, optimizer, scheduler, scaler,
          charb_fn, msssim_fn, vgg_fn, edge_fn, ssim_fn)
