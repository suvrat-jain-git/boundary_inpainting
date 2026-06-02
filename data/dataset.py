# data/dataset.py
"""
Dataset, mask generation, and DataLoader utilities.
All loaders respect speed.test_fraction so phases 1-4
all use the same small subset during testing.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import torchvision.transforms as T

from configs.config import Config


# ── Mask Generation ────────────────────────────────────────────────────────────

def generate_freeform_mask(
    h: int,
    w: int,
    coverage_range: Tuple[float, float] = (0.10, 0.60),
    max_strokes: int = 15,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState()

    lo, hi = coverage_range
    mask = np.zeros((h, w), dtype=np.float32)
    target = rng.uniform(lo, hi)

    for _ in range(max_strokes):
        if mask.mean() >= target:
            break
        n_pts = rng.randint(3, 10)
        pts_x = rng.randint(0, w, size=n_pts)
        pts_y = rng.randint(0, h, size=n_pts)
        thickness = rng.randint(8, max(9, min(h, w) // 4))

        for i in range(len(pts_x) - 1):
            n = max(abs(pts_x[i+1] - pts_x[i]), abs(pts_y[i+1] - pts_y[i])) + 1
            xs = np.linspace(pts_x[i], pts_x[i+1], n).astype(int)
            ys = np.linspace(pts_y[i], pts_y[i+1], n).astype(int)
            for x, y in zip(xs, ys):
                cy, cx = np.ogrid[-y:h-y, -x:w-x]
                mask[cx*cx + cy*cy <= (thickness // 2) ** 2] = 1.0

    attempts = 0
    while mask.mean() < lo and attempts < 50:
        rx = rng.randint(0, max(1, w // 2))
        ry = rng.randint(0, max(1, h // 2))
        rw = rng.randint(w // 8, max(w // 8 + 1, w // 3))
        rh = rng.randint(h // 8, max(h // 8 + 1, h // 3))
        mask[ry:ry+rh, rx:rx+rw] = 1.0
        attempts += 1

    erode_iters = 0
    while mask.mean() > hi and erode_iters < 20:
        mask = binary_erosion(mask > 0.5, iterations=1).astype(np.float32)
        erode_iters += 1

    if mask.mean() < lo:
        pad = int(min(h, w) * (lo - mask.mean()) ** 0.5 + 1)
        cy, cx = h // 2, w // 2
        mask[cy-pad:cy+pad, cx-pad:cx+pad] = 1.0

    return mask


def get_boundary_band(mask: np.ndarray, dilation: int = 5) -> np.ndarray:
    mask_bool = mask > 0.5
    hole_dilated = binary_dilation(mask_bool, iterations=dilation)
    valid_dilated = binary_dilation(~mask_bool, iterations=dilation)
    band = hole_dilated & valid_dilated
    return band.astype(np.float32)


def generate_cached_masks(
    num_masks: int,
    h: int,
    w: int,
    cache_dir: Path,
    prefix: str = "mask",
    seed: int = 42,
) -> List[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_dir.glob(f"{prefix}_*.npy"))

    if len(existing) >= num_masks:
        return existing[:num_masks]

    print(f"  Generating {num_masks} cached masks (prefix='{prefix}') ...")
    rng = np.random.RandomState(seed + 999)
    paths = []

    for i in tqdm(range(num_masks), desc=f"  Caching {prefix} masks"):
        mp = cache_dir / f"{prefix}_{i:04d}.npy"
        if not mp.exists():
            np.save(mp, generate_freeform_mask(h, w, rng=rng))
        paths.append(mp)

    return paths


# ── Dataset ────────────────────────────────────────────────────────────────────

class InpaintingDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        fixed_mask_paths: Optional[List[Path]] = None,
        img_size: int = 256,
        coverage_range: Tuple[float, float] = (0.10, 0.60),
        boundary_dilation: int = 5,
        augment: bool = False,
    ):
        if len(image_paths) == 0:
            raise ValueError("image_paths is empty.")

        self.image_paths = image_paths
        self.fixed_mask_paths = fixed_mask_paths
        self.img_size = img_size
        self.coverage_range = coverage_range
        self.boundary_dilation = boundary_dilation

        base = [T.Resize((img_size, img_size)), T.ToTensor()]
        if augment:
            base = [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
            ]
        self.transform = T.Compose(base)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        if self.fixed_mask_paths is not None:
            mask_np = np.load(self.fixed_mask_paths[idx % len(self.fixed_mask_paths)])
        else:
            mask_np = generate_freeform_mask(self.img_size, self.img_size, self.coverage_range)

        band_np = get_boundary_band(mask_np, self.boundary_dilation)
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()
        boundary = torch.from_numpy(band_np).unsqueeze(0).float()

        return img, mask, boundary


# ── Splits ─────────────────────────────────────────────────────────────────────

def create_splits(
    img_dir: Path,
    splits_file: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    if splits_file.exists():
        with open(splits_file) as f:
            splits = json.load(f)
        sample = splits["train"][:5]
        if sample and all(Path(p).exists() for p in sample):
            print(f"Loaded splits: train={len(splits['train'])}, "
                  f"val={len(splits['val'])}, test={len(splits['test'])}")
            return splits
        print("Splits stale — regenerating.")

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    all_imgs = sorted([
        str(p) for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    if not all_imgs:
        raise RuntimeError(f"No images found in {img_dir}")

    rng = np.random.RandomState(seed)
    rng.shuffle(all_imgs)

    n = len(all_imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_imgs[:n_train],
        "val":   all_imgs[n_train:n_train + n_val],
        "test":  all_imgs[n_train + n_val:],
    }

    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Created splits → train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


# ── DataLoader Factory ─────────────────────────────────────────────────────────

def make_loader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )


# ── Setup ──────────────────────────────────────────────────────────────────────

def setup_data(cfg: Config):
    """
    Creates all datasets and loaders.

    KEY: fast_train_ds and fast_val_ds are subsets controlled by
    speed.test_fraction. ALL four phases use these small subsets
    during testing so the pipeline runs quickly end-to-end.

    To use the full dataset, set speed.test_fraction = 1.0
    """
    paths = cfg.paths
    data  = cfg.data
    speed = cfg.speed

    # Splits
    places_splits = create_splits(
        Path(data.places365_dir),
        Path(paths.splits_dir) / "places365_splits.json",
        train_ratio=data.train_ratio,
        val_ratio=data.val_ratio,
        seed=cfg.train.seed,
    )

    celeba_available = Path(data.celeba_dir).exists()
    if celeba_available:
        celeba_splits = create_splits(
            Path(data.celeba_dir),
            Path(paths.splits_dir) / "celeba_splits.json",
            seed=cfg.train.seed,
        )
    else:
        celeba_splits = {"train": [], "val": [], "test": []}
        print("CelebA-HQ not found — cross-domain eval will be skipped.")

    # Cached masks
    mask_dir = Path(paths.cached_masks_dir)
    val_masks  = generate_cached_masks(data.n_val_masks,  data.img_size,
                                       data.img_size, mask_dir, "val",  cfg.train.seed)
    test_masks = generate_cached_masks(data.n_test_masks, data.img_size,
                                       data.img_size, mask_dir, "test", cfg.train.seed + 1)

    # Full datasets
    train_ds = InpaintingDataset(
        places_splits["train"],
        img_size=data.img_size,
        boundary_dilation=cfg.loss.boundary_dilation,
        augment=True,
    )
    val_ds = InpaintingDataset(
        places_splits["val"],
        fixed_mask_paths=val_masks,
        img_size=data.img_size,
        boundary_dilation=cfg.loss.boundary_dilation,
    )
    test_ds = InpaintingDataset(
        places_splits["test"],
        fixed_mask_paths=test_masks,
        img_size=data.img_size,
        boundary_dilation=cfg.loss.boundary_dilation,
    )

    # ── Fast subsets — controlled by speed.test_fraction ──────────────────────
    n_fast_train = max(2, int(len(train_ds) * speed.test_fraction))
    n_fast_val   = max(2, int(len(val_ds)   * speed.test_fraction))
    n_fast_test  = max(2, int(len(test_ds)  * speed.test_fraction))

    fast_train_ds = Subset(train_ds, list(range(n_fast_train)))
    fast_val_ds   = Subset(val_ds,   list(range(n_fast_val)))
    fast_test_ds  = Subset(test_ds,  list(range(n_fast_test)))

    print(f"\nDataset sizes (test_fraction={speed.test_fraction}):")
    print(f"  fast train : {n_fast_train} images")
    print(f"  fast val   : {n_fast_val}   images")
    print(f"  fast test  : {n_fast_test}  images")
    print(f"  full train : {len(train_ds)} images")
    print(f"  full val   : {len(val_ds)}   images")
    print(f"  full test  : {len(test_ds)}  images")

    nw = data.num_workers
    bs = data.batch_size
    pm = data.pin_memory

    # CelebA
    celeba_test_ds = None
    if celeba_available and celeba_splits["test"]:
        celeba_test_ds = InpaintingDataset(
            celeba_splits["test"],
            fixed_mask_paths=test_masks,
            img_size=data.img_size,
            boundary_dilation=cfg.loss.boundary_dilation,
        )

    return {
        # Full loaders (for final paper runs)
        "train_loader":      make_loader(train_ds,  bs, True,  nw, pm, True),
        "val_loader":        make_loader(val_ds,    bs, False, nw, pm),
        "test_loader":       make_loader(test_ds,   bs, False, nw, pm),

        # Fast loaders (used in ALL phases during testing)
        "fast_train_loader": make_loader(fast_train_ds, bs, True,  nw, pm, True),
        "fast_val_loader":   make_loader(fast_val_ds,   bs, False, nw, pm),
        "fast_test_loader":  make_loader(fast_test_ds,  bs, False, nw, pm),

        # CelebA
        "celeba_loader": (
            make_loader(celeba_test_ds, bs, False, nw, pm)
            if celeba_test_ds else None
        ),

        "places_splits":   places_splits,
        "celeba_available": celeba_available,
    }
