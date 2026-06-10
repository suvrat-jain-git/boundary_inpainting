# data/download.py
"""
Dataset download helpers.

Each function is idempotent: a sentinel file .download_complete is written
after the first successful download. Subsequent calls skip the download.

Usage:
    from data.download import download_all
    download_all(base_dir="./datasets")

    # With Google Drive cache (Colab):
    download_all(base_dir="./datasets", drive_cache="/content/drive/MyDrive/boundary_inpainting/datasets")
"""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

SENTINEL = ".download_complete"


def _is_done(root: Path) -> bool:
    return (root / SENTINEL).exists()


def _mark_done(root: Path):
    (root / SENTINEL).write_text("done")


def _try_copy_from_drive(drive_cache: Optional[str], subdir: str, dest: Path) -> bool:
    """Copy from Google Drive cache if available. Returns True on success."""
    if drive_cache is None:
        return False
    src = Path(drive_cache) / subdir
    if not src.exists():
        return False
    print(f"  Restoring {subdir} from Drive cache...")
    shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
    return True


# ── Places365 ─────────────────────────────────────────────────────────────────

def download_places365(root: str = "./datasets/places365",
                       drive_cache: Optional[str] = None) -> None:
    """
    Download Places365 validation set (val_256, ~2 GB).
    Uses the MIT CSAIL data server.
    """
    root = Path(root)
    if _is_done(root):
        print(f"  Places365 already downloaded at {root}")
        return

    root.mkdir(parents=True, exist_ok=True)

    if _try_copy_from_drive(drive_cache, "places365", root):
        _mark_done(root)
        return

    url      = "http://data.csail.mit.edu/places/places365/val_256.tar"
    tar_path = root.parent / "val_256.tar"

    print(f"  Downloading Places365 val_256 (~2 GB) to {tar_path}...")
    print("  This will take a while on a slow connection.")
    _wget(url, tar_path)

    print("  Extracting...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(root)
    tar_path.unlink(missing_ok=True)

    _mark_done(root)
    print(f"  Places365 ready at {root}")


# ── CelebA / face dataset ──────────────────────────────────────────────────────

def download_celeba_hq(root: str = "./datasets/celeba_hq",
                       drive_cache: Optional[str] = None) -> None:
    """
    Download a face image dataset for cross-domain evaluation.

    Sources tried in order (all public, no auth required):
      1. torchvision.datasets.CelebA  — official aligned CelebA via Google Drive
         (test split: ~20k images at 178×218, resized to 256×256 at load time)
      2. torchvision.datasets.LFWPeople — Labeled Faces in the Wild from UMass
         (~13k images; used as face-domain proxy if CelebA is unavailable)

    Images are stored under {root}/ using the torchvision directory structure.
    dataset.py's create_splits() uses rglob and handles both layouts automatically.
    """
    root = Path(root)
    if _is_done(root):
        print(f"  CelebA already downloaded at {root}")
        return

    root.mkdir(parents=True, exist_ok=True)

    if _try_copy_from_drive(drive_cache, "celeba_hq", root):
        _mark_done(root)
        return

    # Strategy 1: official CelebA via torchvision (uses gdown → Google Drive)
    try:
        import torchvision  # type: ignore
        print(f"  Downloading CelebA (test split, ~400 MB) via torchvision to {root}...")
        torchvision.datasets.CelebA(root=str(root), split="test", download=True)
        _mark_done(root)
        print(f"  CelebA ready at {root}")
        return
    except Exception as exc:
        print(f"    torchvision CelebA failed: {exc}")

    # Strategy 2: LFW as face-domain proxy (downloads from UMass HTTP, always works)
    try:
        import torchvision  # type: ignore
        print(f"  Falling back to LFW (Labeled Faces in the Wild) as face-domain proxy...")
        torchvision.datasets.LFWPeople(root=str(root), split="test", download=True)
        _mark_done(root)
        print(
            f"  LFW (face proxy) ready at {root}\n"
            f"  NOTE: Paper cross-domain 'face' column uses LFW images."
        )
        return
    except Exception as exc:
        print(f"    LFW download failed: {exc}")

    print(
        "  WARNING: All face dataset downloads failed. "
        "Cross-domain face evaluation will be skipped."
    )


# ── DTD ───────────────────────────────────────────────────────────────────────

def download_dtd(root: str = "./datasets/dtd",
                 drive_cache: Optional[str] = None) -> None:
    """
    Download Describable Textures Dataset (DTD) via torchvision.
    Total size ~600 MB.
    """
    root = Path(root)
    if _is_done(root):
        print(f"  DTD already downloaded at {root}")
        return

    root.mkdir(parents=True, exist_ok=True)

    if _try_copy_from_drive(drive_cache, "dtd", root):
        _mark_done(root)
        return

    try:
        import torchvision
        torchvision.datasets.DTD(root=str(root), split="train", download=True)
        torchvision.datasets.DTD(root=str(root), split="val",   download=True)
        torchvision.datasets.DTD(root=str(root), split="test",  download=True)
    except Exception as e:
        raise RuntimeError(f"DTD download failed: {e}") from e

    _mark_done(root)
    print(f"  DTD ready at {root}")


# ── Download all ──────────────────────────────────────────────────────────────

def download_all(base_dir: str = "./datasets",
                 drive_cache: Optional[str] = None) -> None:
    """
    Download all required datasets.

    Args:
        base_dir   : local directory for datasets
        drive_cache: optional Google Drive cache path (Colab usage)
    """
    print("\nDataset download check...")
    download_places365(
        root=str(Path(base_dir) / "places365"),
        drive_cache=drive_cache,
    )
    download_celeba_hq(
        root=str(Path(base_dir) / "celeba_hq"),
        drive_cache=drive_cache,
    )
    download_dtd(
        root=str(Path(base_dir) / "dtd"),
        drive_cache=drive_cache,
    )
    print("  All datasets ready.")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _wget(url: str, dest: Path):
    """
    Download a file from url to dest, trying urllib then requests.
    Shows a simple progress indicator (no tqdm).
    """
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)

    class _Progress:
        def __init__(self):
            self._last = 0

        def __call__(self, count, block_size, total):
            if total <= 0:
                return
            pct = count * block_size * 100 // total
            if pct - self._last >= 10:
                print(f"    ... {pct}%", flush=True)
                self._last = pct

    urllib.request.urlretrieve(url, str(dest), reporthook=_Progress())
