# utils/visualize.py
"""
Visualization utilities:
  - EDA plots (sample grid, resolution, mask coverage, boundary band)
  - Training curves
  - Ablation tables (styled matplotlib)
  - Qualitative comparison grids (L0 vs L4)
  - Sensitivity ablation plots
  - Cross-domain results
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from PIL import Image

# Consistent color palette
COLORS = {
    "L0": "#e74c3c",
    "L1": "#e67e22",
    "L2": "#f1c40f",
    "L3": "#27ae60",
    "L4": "#2980b9",
    "LaMa": "#8e44ad",
    "EdgeConnect": "#16a085",
}


def save_fig(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── EDA ───────────────────────────────────────────────────────────────────────

def plot_sample_grid(image_paths: List[str], results_dir: Path, n: int = 18):
    """Random sample grid of training images."""
    rng = np.random.RandomState(42)
    idxs = rng.choice(len(image_paths), min(n, len(image_paths)), replace=False)

    cols = 6
    rows = max(1, len(idxs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
    fig.suptitle("Sample Training Images", fontsize=14, fontweight="bold")

    for i, ax in enumerate(np.array(axes).flat):
        if i < len(idxs):
            img = Image.open(image_paths[idxs[i]]).convert("RGB").resize((256, 256))
            ax.imshow(np.array(img))
        ax.axis("off")

    save_fig(fig, results_dir / "eda_sample_grid.png")


def plot_mask_examples(results_dir: Path, img_size: int = 256):
    """Example masks at each difficulty level."""
    from data.dataset import generate_freeform_mask, get_boundary_band

    mask_bins = {
        "easy": (0.10, 0.20), "mid": (0.20, 0.35),
        "hard": (0.35, 0.50), "xhard": (0.50, 0.60),
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Masks by Difficulty Level", fontsize=14, fontweight="bold")

    rng = np.random.RandomState(100)
    for col, (name, (lo, hi)) in enumerate(mask_bins.items()):
        mask = None
        for _ in range(100):
            m = generate_freeform_mask(img_size, img_size, (lo, hi), rng=rng)
            if lo <= m.mean() <= hi:
                mask = m
                break
        if mask is None:
            mask = m

        band = get_boundary_band(mask)
        cov = mask.mean()

        axes[0, col].imshow(mask, cmap="gray")
        axes[0, col].set_title(f"{name}\n{cov*100:.1f}% coverage")
        axes[0, col].axis("off")

        axes[1, col].imshow(band, cmap="hot")
        axes[1, col].set_title(f"Boundary band")
        axes[1, col].axis("off")

        white = np.ones((img_size, img_size, 3))
        white[mask > 0.5] = [1, 0.3, 0.3]
        white[band > 0.5] = [0.3, 0.6, 1.0]
        axes[2, col].imshow(white)
        axes[2, col].set_title("Mask + Band overlay")
        axes[2, col].axis("off")

    save_fig(fig, results_dir / "eda_masks.png")


def plot_mask_coverage(results_dir: Path, n: int = 300, img_size: int = 256):
    """Histogram of mask coverage distribution."""
    from data.dataset import generate_freeform_mask

    rng = np.random.RandomState(42)
    coverages = [
        generate_freeform_mask(img_size, img_size, rng=rng).mean() * 100
        for _ in range(n)
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(coverages, bins=40, color="#8e44ad", edgecolor="white", alpha=0.8)

    bin_colors = ["#3498db", "#27ae60", "#e67e22", "#e74c3c"]
    bins = [("easy", 10, 20), ("mid", 20, 35), ("hard", 35, 50), ("xhard", 50, 60)]
    for (name, lo, hi), color in zip(bins, bin_colors):
        ax.axvspan(lo, hi, alpha=0.12, color=color, label=f"{name} ({lo}–{hi}%)")

    ax.set_xlabel("Mask Coverage (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Mask Coverage Distribution (n={n})", fontweight="bold")
    ax.legend(fontsize=9)
    save_fig(fig, results_dir / "eda_coverage.png")


# ── Training Curves ────────────────────────────────────────────────────────────

def plot_training_curves(
    histories: Dict[str, dict],
    results_dir: Path,
    title: str = "Training Curves",
):
    """Plot train/val loss curves for multiple experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    has_data = False
    for exp_name, h in histories.items():
        color = COLORS.get(exp_name.split("_")[-1].upper(), None)
        if h.get("train_loss"):
            axes[0].plot(h["train_loss"], label=exp_name, color=color, alpha=0.85)
            has_data = True
        if h.get("val_loss"):
            axes[1].plot(h["val_loss"], label=exp_name, color=color, alpha=0.85)

    for ax, title_str in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title_str)
        ax.grid(alpha=0.3)
        if has_data:
            ax.legend(fontsize=8)

    if not has_data:
        for ax in axes:
            ax.text(0.5, 0.5, "No training history\n(loaded from checkpoints)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11)

    save_fig(fig, results_dir / "training_curves.png")


# ── Ablation Table ─────────────────────────────────────────────────────────────

def plot_ablation_table(
    phase4_eval: Dict[str, dict],
    results_dir: Path,
    title: str = "Phase 4 — Loss Ablation (Test Set)",
):
    """Styled matplotlib table with best-value highlighting."""
    metrics = ["psnr", "ssim", "hole_mae", "boundary_mae", "spectral_coherence"]
    display = ["PSNR ↑", "SSIM ↑", "Hole-MAE ↓", "Boundary-MAE ↓", "Spectral ↑"]

    rows = []
    exp_names = list(phase4_eval.keys())
    for exp in exp_names:
        g = phase4_eval[exp].get("global", {})
        rows.append([exp] + [f"{g.get(m, float('nan')):.4f}" for m in metrics])

    if not rows:
        print("  No Phase 4 results to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.8 + 1)))
    ax.axis("off")

    headers = ["Experiment"] + display
    table = ax.table(
        cellText=rows, colLabels=headers, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Highlight best per column
    for col_idx, m in enumerate(metrics, start=1):
        vals = []
        for exp in exp_names:
            g = phase4_eval[exp].get("global", {})
            vals.append(g.get(m, float("nan")))
        if all(np.isnan(v) for v in vals):
            continue
        best_idx = int(np.nanargmin(vals) if "mae" in m else np.nanargmax(vals))
        table[best_idx + 1, col_idx].set_facecolor("#d5f5e3")
        table[best_idx + 1, col_idx].set_text_props(fontweight="bold")

    ax.set_title(title, fontweight="bold", fontsize=13, pad=20)
    save_fig(fig, results_dir / "ablation_table.png")


# ── Sensitivity Plots ──────────────────────────────────────────────────────────

def plot_sensitivity(sensitivity_results: dict, results_dir: Path):
    """Plot sensitivity of metrics to K, patch_size, and dilation r."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Hyperparameter Sensitivity Analysis", fontsize=14, fontweight="bold")

    # K ablation
    if sensitivity_results.get("k_ablation"):
        data = sensitivity_results["k_ablation"]
        ks = [d["k"] for d in data]
        bmaes = [d["boundary_mae"] for d in data]
        axes[0].plot(ks, bmaes, "o-", color="#2980b9", linewidth=2, markersize=8)
        axes[0].set_xlabel("K (spectral patches per image)")
        axes[0].set_ylabel("Boundary-MAE ↓")
        axes[0].set_title("Sensitivity to K")
        axes[0].set_xticks(ks)
        axes[0].grid(alpha=0.3)

    # Patch size ablation
    if sensitivity_results.get("patch_ablation"):
        data = sensitivity_results["patch_ablation"]
        pss = [d["patch_size"] for d in data]
        scs = [d["spectral_coherence"] for d in data]
        axes[1].plot(pss, scs, "o-", color="#27ae60", linewidth=2, markersize=8)
        axes[1].set_xlabel("Patch Size (pixels)")
        axes[1].set_ylabel("Spectral Coherence ↑")
        axes[1].set_title("Sensitivity to Patch Size")
        axes[1].set_xticks(pss)
        axes[1].grid(alpha=0.3)

    # Dilation ablation
    if sensitivity_results.get("dilation_ablation"):
        data = sensitivity_results["dilation_ablation"]
        rs = [d["dilation"] for d in data]
        bmaes = [d["boundary_mae"] for d in data]
        axes[2].plot(rs, bmaes, "o-", color="#e74c3c", linewidth=2, markersize=8)
        axes[2].set_xlabel("Dilation Radius r (pixels)")
        axes[2].set_ylabel("Boundary-MAE ↓")
        axes[2].set_title("Sensitivity to Boundary Band Width r")
        axes[2].set_xticks(rs)
        axes[2].grid(alpha=0.3)

    save_fig(fig, results_dir / "sensitivity_analysis.png")


# ── Qualitative Comparison Grid ────────────────────────────────────────────────

def plot_qualitative(
    model_l0,
    model_l4,
    test_loader,
    device,
    results_dir: Path,
    n_show: int = 4,
    use_amp: bool = True,
):
    """Visual comparison: original / masked / L0 output / L4 output / error maps."""
    import torch

    def _autocast(use_amp):
        if use_amp and torch.cuda.is_available():
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return torch.amp.autocast("cpu", dtype=torch.float32, enabled=False)

    model_l0.eval()
    model_l4.eval()

    images, masks, boundaries = next(iter(test_loader))
    images = images[:n_show].to(device)
    masks = masks[:n_show].to(device)
    boundaries = boundaries[:n_show].to(device)
    masked_input = images * (1.0 - masks)

    with torch.no_grad(), _autocast(use_amp):
        pred_l0 = model_l0(masked_input, masks, boundaries)["output"].float()
        pred_l4 = model_l4(masked_input, masks, boundaries)["output"].float()

    comp_l0 = images * (1.0 - masks) + pred_l0 * masks
    comp_l4 = images * (1.0 - masks) + pred_l4 * masks
    err_l0 = torch.abs(images - comp_l0) * boundaries
    err_l4 = torch.abs(images - comp_l4) * boundaries

    col_titles = [
        "Original", "Masked Input",
        "L0 (Base)", "L4 (Full Method)",
        "Boundary Error L0", "Boundary Error L4",
    ]

    fig, axes = plt.subplots(n_show, 6, figsize=(24, n_show * 4))
    for row in range(n_show):
        items = [
            images[row].cpu(), masked_input[row].cpu(),
            comp_l0[row].cpu(), comp_l4[row].cpu(),
            err_l0[row].cpu(), err_l4[row].cpu(),
        ]
        for col, img_t in enumerate(items):
            arr = img_t.permute(1, 2, 0).numpy().clip(0, 1)
            if col >= 4:
                axes[row, col].imshow(arr.mean(-1), cmap="hot", vmin=0, vmax=0.25)
            else:
                axes[row, col].imshow(arr)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10)

    fig.suptitle("Qualitative: L0 (Base) vs L4 (Full Method)", fontsize=13, fontweight="bold")
    save_fig(fig, results_dir / "qualitative_comparison.png")


# ── Baseline Comparison Bar Chart ─────────────────────────────────────────────

def plot_baseline_comparison(comparison: Dict[str, Dict], results_dir: Path):
    """Bar chart comparing our method against baselines on key metrics."""
    if not comparison:
        return

    key_metrics = ["psnr", "ssim", "boundary_mae", "spectral_coherence"]
    display = ["PSNR ↑", "SSIM ↑", "Boundary-MAE ↓", "Spectral Coh. ↑"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Method Comparison on Test Set", fontsize=14, fontweight="bold")

    methods = list(comparison.keys())
    x = np.arange(len(methods))
    bar_colors = [COLORS.get(m.split()[0], "#7f8c8d") for m in methods]

    for ax, metric, disp in zip(axes, key_metrics, display):
        vals = [comparison[m].get(metric, 0) for m in methods]
        bars = ax.bar(x, vals, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        ax.set_title(disp)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9
            )

    save_fig(fig, results_dir / "baseline_comparison.png")
