# configs/config.py
"""
Central configuration — ALL settings in one place.

PROFILES:
  - get_config()            → default (debug, fast)
  - get_production_config() → full dataset, 50 epochs, production settings
  - get_smoke_config()      → 5-epoch sanity check, tiny dataset

HOW TO CONTROL SPEED (SpeedConfig):
  - test_fraction: fraction of dataset used in ALL phases
      0.001 = ~25 images   → local debug (default)
      1.0   = full dataset → paper run (use get_production_config())
  - epochs_per_phase: epochs for phases 1-3 selection runs
  - epochs_full: epochs for ALL phase-4 ablation configs (L0–L4 + ASBC)
      NOTE: ALL phase-4 configs must use the same epochs_full for a fair
            ablation table. Do NOT use epochs_per_phase for L0-L4.
  - val_every_n_epochs: run validation every N epochs (2 saves ~30% time)
  - val_batches: max validation batches per pass (None = all)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    img_size: int = 256
    batch_size: int = 16          # A100: 16-32 fits easily; adjust upward for speed
    num_workers: int = 4          # A100 Colab: 4 is a safe default
    pin_memory: bool = True

    # ── DATASET PATHS ─────────────────────────────────────────────────────────
    places365_dir: str = "./datasets/places365"
    celeba_dir: str = "./datasets/celeba_hq"
    dtd_dir: str = "./datasets/dtd"

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
    n_val_masks: int = 200
    n_test_masks: int = 500


@dataclass
class SpeedConfig:
    """
    ── SPEED CONTROL ─────────────────────────────────────────────────────────
    See module docstring above for full explanation.
    """

    test_fraction: float = 0.001   # ← 0.001 for debug, 1.0 for paper

    epochs_per_phase: int = 2      # ← phases 1-3 selection only
    epochs_full: int = 2           # ← ALL phase-4 ablation configs (L0–L4+ASBC)
                                   #   set to 50 in production

    val_every_n_epochs: int = 1    # ← 2 saves ~30% time on A100
    val_batches: Optional[int] = 30  # ← None for full val

    # Perceptual (VGG-16) loss is expensive. Apply every N steps.
    # perc_every_n_steps=4 keeps VGG contribution statistically accurate
    # while reducing its wall-clock cost by ~3/4.
    perc_every_n_steps: int = 1    # ← set to 4 in production for ~40% speedup


@dataclass
class ModelConfig:
    backbone: str = "resnet34"            # "resnet18" | "resnet34" | "convnext_tiny"
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
    spectral_use_log: bool = True        # log-magnitude (more perceptually uniform)
    tv_weight: float = 0.01
    ms_weight: float = 0.5

    # ── ASBC (Adaptive Spectral Boundary Coherence) ────────────────────────────
    # Active when loss_config == "adaptive_spectral" or "full_adaptive"
    adaptive_spectral: bool = False      # True → use ASBC instead of fixed spectral
    spectral_n_bands: int = 4            # number of radial frequency bands
    spectral_reg_weight: float = 0.01   # entropy anti-collapse regularizer weight

    loss_config: str = "full"


@dataclass
class TrainConfig:
    seed: int = 42
    lr_peak: float = 3e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    patience: int = 8              # was 2 — prevents false early stopping
    freeze_encoder_epochs: int = 1

    # ── Hardware / precision ───────────────────────────────────────────────────
    use_amp: bool = True           # kept for backward compat; bf16 is auto on CUDA
    use_bf16: bool = True          # True → bfloat16 (A100); False → float16 fallback

    # ── EMA ───────────────────────────────────────────────────────────────────
    use_ema: bool = True
    ema_decay: float = 0.999       # shadow weights updated each step


@dataclass
class EvalConfig:
    k_values: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    patch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    dilation_values: List[int] = field(default_factory=lambda: [3, 5, 7, 10])
    lama_checkpoint: Optional[str] = None
    edge_connect_checkpoint: Optional[str] = None
    compute_fid: bool = True       # requires clean-fid package


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


# ── Config Profiles ────────────────────────────────────────────────────────────

def get_config() -> Config:
    """Default config — debug/local. Fast but uses tiny data slice."""
    return Config()


def get_production_config() -> Config:
    """
    Full production config for A100 paper run.

    Epoch budget rationale (A100-40GB, ~24 min/epoch measured):
      - epochs_full=12 → fits 6-config ablation within 30h hard limit
        (6 × 12 × 24 min ≈ 28.8h + ~0.5h eval = 29.3h)
      - 12 epochs reaches ~70% convergence — sufficient to establish
        relative ordering L0 < L1 < L2 < L3c < L4 for ablation table
      - patience=4 → early stop if plateau, saves buffer for heavier configs
      - batch_size=48: fills ~34 GB VRAM, larger batches → better BN stats
      - Multi-seed and ConvNeXt runs skipped (pass --skip_multiseed
        --skip_convnext, or use skip_phases123=True default in run.py)

    Key settings:
      - test_fraction=1.0  (all 36,500 Places365 images)
      - epochs_per_phase=5 (phases 1-3 selection, skipped by default)
      - epochs_full=12     (ALL phase-4 ablation configs, fair equal)
      - val_every_n_epochs=2, val_batches=None (full validation)
      - perc_every_n_steps=4 (~40% epoch speedup)
      - patience=4
    """
    cfg = Config()
    cfg.data.batch_size = 48
    cfg.data.num_workers = 4
    cfg.data.n_val_masks = 500
    cfg.data.n_test_masks = 1000
    cfg.speed.test_fraction = 1.0
    cfg.speed.epochs_per_phase = 5
    cfg.speed.epochs_full = 12
    cfg.speed.val_every_n_epochs = 2
    cfg.speed.val_batches = None
    cfg.speed.perc_every_n_steps = 4   # VGG every 4 steps → ~40% faster epochs
    cfg.train.patience = 4
    return cfg


def get_smoke_config() -> Config:
    """
    5-epoch smoke test — verifies losses fall and pipeline runs end-to-end.
    Uses tiny data slice; intended for Colab Cell 2 sanity check.
    """
    cfg = Config()
    cfg.data.batch_size = 8
    cfg.data.num_workers = 2
    cfg.speed.test_fraction = 0.005   # ~180 images
    cfg.speed.epochs_per_phase = 2
    cfg.speed.epochs_full = 5
    cfg.speed.val_every_n_epochs = 1
    cfg.speed.val_batches = 10
    cfg.train.patience = 5
    return cfg
