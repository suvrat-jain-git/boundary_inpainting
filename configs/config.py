# configs/config.py
"""
Central configuration — ALL settings in one place.

HOW TO CONTROL SPEED:
  - test_fraction: fraction of dataset used in ALL phases (0.001 = ~25 imgs, 0.01 = ~255 imgs)
  - epochs_per_phase: epochs for phases 1-3 and phase 4 ablation runs
  - epochs_full: epochs for the final L4 full model
  - val_batches: how many validation batches to run (None = all, 50 = fast)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    img_size: int = 256
    batch_size: int = 4
    num_workers: int = 0          # keep 0 on Windows to avoid errors
    pin_memory: bool = True

    # ── YOUR DATASET PATHS ────────────────────────────────────────────────────
    places365_dir: str = "./datasets/places365"
    celeba_dir: str = "./datasets/celeba_hq"

    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # Mask difficulty bins (coverage fraction)
    mask_bins: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "easy":  (0.10, 0.20),
        "mid":   (0.20, 0.35),
        "hard":  (0.35, 0.50),
        "xhard": (0.50, 0.60),
    })

    # Cached masks for reproducible eval
    n_val_masks: int = 50
    n_test_masks: int = 50


@dataclass
class SpeedConfig:
    """
    ── SPEED CONTROL — change these to go faster or slower ──────────────────

    test_fraction:
        Fraction of full dataset used in ALL phases for training AND validation.
        0.001 = ~25 images  → very fast test (~1 hr CPU)
        0.01  = ~255 images → quick check  (~3 hrs CPU)
        0.1   = ~2550 imgs  → proper run   (use GPU/Kaggle)
        1.0   = full dataset → final paper run (GPU only)

    epochs_per_phase:
        Epochs for phases 1, 2, 3 and phase 4 ablation (L0-L3).
        2 is fine for sanity check. Use 15+ for real results.

    epochs_full:
        Epochs for the final L4 full method only.
        Use 2 for testing, 50 for final paper results.

    val_batches:
        How many val batches to run per epoch.
        None = run all (slow). 30 = fast check.
    """

    test_fraction: float = 0.001   # ← CHANGE THIS to control speed

    epochs_per_phase: int = 2      # ← phases 1-3 and L0-L3
    epochs_full: int = 2           # ← L4 final model

    val_batches: Optional[int] = 30  # ← None for full val, 30 for fast


@dataclass
class ModelConfig:
    backbone: str = "resnet34"
    use_attention_skip: bool = True
    use_gated_conv: bool = True
    multiscale_output: bool = True
    use_transformer_bottleneck: bool = True
    transformer_heads: int = 8
    transformer_layers: int = 2
    use_bilinear_upsample: bool = True
    use_boundary_conditioning: bool = True


@dataclass
class LossConfig:
    hole_weight: float = 6.0
    valid_weight: float = 1.0
    perceptual_weight: float = 0.1
    boundary_weight: float = 3.0
    boundary_dilation: int = 5
    spectral_weight: float = 1.0
    spectral_patches_k: int = 8
    spectral_patch_size: int = 16
    spectral_use_log: bool = False
    tv_weight: float = 0.01
    ms_weight: float = 0.5
    loss_config: str = "full"


@dataclass
class TrainConfig:
    seed: int = 42
    lr_peak: float = 3e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 1
    patience: int = 2
    freeze_encoder_epochs: int = 1
    use_amp: bool = True


@dataclass
class EvalConfig:
    k_values: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    patch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    dilation_values: List[int] = field(default_factory=lambda: [3, 5, 7, 10])
    lama_checkpoint: Optional[str] = None
    edge_connect_checkpoint: Optional[str] = None


@dataclass
class PathConfig:
    results_dir: str = "./results"
    checkpoints_dir: str = "./checkpoints"
    splits_dir: str = "./splits"
    cached_masks_dir: str = "./cached_masks"
    logs_dir: str = "./logs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def make_dirs(self):
        for d in [
            self.paths.results_dir, self.paths.checkpoints_dir,
            self.paths.splits_dir, self.paths.cached_masks_dir,
            self.paths.logs_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    return Config()
