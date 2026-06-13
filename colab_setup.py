# colab_setup.py
"""
Google Colab persistence for boundary-aware inpainting.

Drive layout — upload the full repo FLAT to MyDrive/inpainting/:
    MyDrive/inpainting/
        main.py, run.py, colab_setup.py, requirements.txt, ...
        configs/, data/, models/, training/, losses/, evaluation/, utils/
        checkpoints/                  <- managed by this module
            manifest.json             <- tracks what's been saved to Drive
            ckpt_YYYYMMDD_HHMMSS.tar.gz
        results/                      <- managed by this module
            manifest.json
            results_YYYYMMDD_HHMMSS.tar.gz
        datasets/                     <- optional dataset cache on Drive

IO design — ALL Drive writes are single tarball uploads (never individual files).
Idle auto-save ticks are purely local (manifest read only, no Drive touch).

Bootstrap pattern (Cell 1 — before repo is local):
    import shutil, importlib.util, sys
    from google.colab import drive; drive.mount("/content/drive")
    shutil.copy("/content/drive/MyDrive/inpainting/colab_setup.py",
                "/tmp/_cs_boot.py")
    spec = importlib.util.spec_from_file_location("cs", "/tmp/_cs_boot.py")
    cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
    cs.init()   # copies full repo, restores checkpoints + results

Usage (Cell 2 — repo is now local):
    from colab_setup import init, auto_save, require_data, save, status
    init()           # idempotent — skips already-restored tarballs
    auto_save(20)    # background incremental saves every 20 min
    require_data()   # fail-fast if Places365 missing
    save("final")    # force-save everything at end of run
"""

import json
import shutil
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ── Path constants (exported; used by notebook cells too) ─────────────────────

DRIVE_ROOT = Path("/content/drive/MyDrive/inpainting")
LOCAL_ROOT = Path("/content/inpainting")

DRIVE_CKPT = DRIVE_ROOT / "checkpoints"
DRIVE_RES  = DRIVE_ROOT / "results"
DRIVE_DATA = DRIVE_ROOT / "datasets"   # optional dataset cache on Drive

LOCAL_CKPT = LOCAL_ROOT / "checkpoints"
LOCAL_RES  = LOCAL_ROOT / "results"
LOCAL_DATA = LOCAL_ROOT / "datasets"

# Local manifest file (fast local read; one small copy pushed to Drive after save)
_MANIFEST_LOCAL = LOCAL_ROOT / ".colab_manifest.json"

# Subdirectories inside the Drive repo root to skip when copying code → local
_CODE_SKIP = frozenset(
    {"checkpoints", "results", "datasets", "__pycache__", ".git", ".ipynb_checkpoints"}
)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mount_drive() -> bool:
    try:
        from google.colab import drive as _gd  # type: ignore
        if not Path("/content/drive/MyDrive").exists():
            _gd.mount("/content/drive")
            print("  Google Drive mounted.")
        else:
            print("  Google Drive already mounted.")
        return True
    except ImportError:
        print("  Not in Colab — Drive mount skipped.")
        return False


# ── Manifest ───────────────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    if _MANIFEST_LOCAL.exists():
        try:
            return json.loads(_MANIFEST_LOCAL.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "checkpoints":               {},  # fname  -> {size, mtime, tarball}
        "results":                   {},  # run_name -> {tarball, saved_at}
        "_restored_ckpt_tarballs":   [],
        "_restored_res_tarballs":    [],
    }


def _save_manifest(manifest: dict) -> None:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    _MANIFEST_LOCAL.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Push one tiny file to Drive so restarts can recover the manifest
    DRIVE_CKPT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(_MANIFEST_LOCAL), str(DRIVE_CKPT / "manifest.json"))


def _restore_manifest() -> dict:
    """Pull manifest from Drive if local copy is absent (e.g. fresh session)."""
    if not _MANIFEST_LOCAL.exists():
        src = DRIVE_CKPT / "manifest.json"
        if src.exists():
            _MANIFEST_LOCAL.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(_MANIFEST_LOCAL))
            print("  Manifest restored from Drive.")
    return _load_manifest()


# ── Code copy: Drive → Local ───────────────────────────────────────────────────

def _copy_code_from_drive() -> None:
    """
    Mirror code files MyDrive/inpainting/ → /content/inpainting/.
    Skips checkpoints/, results/, datasets/ (those come from tarballs).
    Uses mtime comparison so re-runs are fast (only newer files copied).
    All in one sequential pass — no per-file Drive reads during training.
    """
    if not DRIVE_ROOT.exists():
        print("  WARNING: MyDrive/inpainting/ not found. Did you upload the repo?")
        return
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for src in DRIVE_ROOT.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(DRIVE_ROOT)
        if any(part in _CODE_SKIP for part in rel.parts):
            continue
        dest = LOCAL_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(str(src), str(dest))
            copied += 1
        else:
            skipped += 1
    print(f"  Code: {copied} file(s) copied, {skipped} already current.")


# ── Tarball extraction ─────────────────────────────────────────────────────────

def _extract_tarballs(
    drive_dir: Path,
    local_dir: Path,
    pattern: str,
    already_restored: Set[str],
    label: str,
) -> Tuple[int, Set[str]]:
    """
    Extract tarballs in drive_dir matching glob pattern that aren't in
    already_restored.  Returns (files_extracted, updated_restored_set).
    """
    if not drive_dir.exists():
        print(f"  {label}: no Drive folder yet.")
        return 0, already_restored

    local_dir.mkdir(parents=True, exist_ok=True)
    new_tarballs = sorted(
        p for p in drive_dir.glob(pattern)
        if p.name not in already_restored
    )
    if not new_tarballs:
        print(f"  {label}: nothing new to restore.")
        return 0, already_restored

    extracted = 0
    restored = set(already_restored)
    for tb in new_tarballs:
        try:
            with tarfile.open(str(tb), "r:gz") as tf:
                tf.extractall(str(local_dir))
                extracted += len(tf.getnames())
            restored.add(tb.name)
        except Exception as e:
            print(f"  WARNING: failed to extract {tb.name}: {e}")

    print(f"  {label}: {extracted} file(s) from {len(new_tarballs)} tarball(s).")
    return extracted, restored


# ── Completed run detection ────────────────────────────────────────────────────

def _completed_run_names() -> List[str]:
    """
    A run is 'complete' only when BOTH its *_curves.json AND its checkpoint
    (.pt file) exist locally.  In-progress runs (curves file but no checkpoint
    yet, or vice versa) are deliberately excluded so we never tarball partial
    state that will change on the next auto-save tick.
    """
    if not LOCAL_RES.exists() or not LOCAL_CKPT.exists():
        return []
    complete = []
    for cf in LOCAL_RES.glob("*_curves.json"):
        run_name = cf.stem[: -len("_curves")]
        ckpt_ok = (
            (LOCAL_CKPT / f"{run_name}_ema.pt").exists()
            or (LOCAL_CKPT / f"{run_name}.pt").exists()
        )
        if ckpt_ok:
            complete.append(run_name)
    return sorted(complete)


# ── Incremental checkpoint save ────────────────────────────────────────────────

def _save_checkpoints_incremental(manifest: dict, force_all: bool = False) -> bool:
    """
    Bundle new/changed .pt files into one tarball and push to Drive.
    Manifest tracks {fname: {size, mtime, tarball}} so runs that haven't
    changed are never re-uploaded.

    Uses a .tmp_ prefix while building the tarball so a mid-write crash
    leaves no corrupt file on Drive.
    """
    if not LOCAL_CKPT.exists():
        return False

    ckpt_manifest: Dict[str, dict] = manifest.get("checkpoints", {})
    to_bundle: List[Path] = []

    for pt in sorted(LOCAL_CKPT.glob("*.pt")):
        stat = pt.stat()
        prev = ckpt_manifest.get(pt.name)
        if (
            force_all
            or prev is None
            or stat.st_size != prev.get("size", -1)
            or stat.st_mtime > prev.get("mtime", 0)
        ):
            to_bundle.append(pt)

    if not to_bundle:
        print("  Checkpoints: nothing new.")
        return False

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_name = f"ckpt_{ts}.tar.gz"
    tb_tmp  = LOCAL_ROOT / f".tmp_{tb_name}"   # build in fast /content/ space
    tb_dst  = DRIVE_CKPT / tb_name

    DRIVE_CKPT.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(str(tb_tmp), "w:gz") as tf:
            for pt in to_bundle:
                tf.add(str(pt), arcname=pt.name)
        shutil.move(str(tb_tmp), str(tb_dst))  # single Drive write
        for pt in to_bundle:
            s = pt.stat()
            ckpt_manifest[pt.name] = {
                "size":    s.st_size,
                "mtime":   s.st_mtime,
                "tarball": tb_name,
            }
        manifest["checkpoints"] = ckpt_manifest
        print(f"  Checkpoints: {len(to_bundle)} file(s) → {tb_name}")
        return True
    except Exception as e:
        print(f"  WARNING: checkpoint save failed: {e}")
        if tb_tmp.exists():
            tb_tmp.unlink()
        return False


# ── Incremental results save ───────────────────────────────────────────────────

def _save_results_incremental(manifest: dict, force_all: bool = False) -> bool:
    """
    Bundle results for newly completed runs into one tarball → push to Drive.

    Rules:
    - A run must be COMPLETE (curves JSON + checkpoint both present) before
      it can be included.  Partially-trained runs are never tarballed.
    - Runs already in manifest["results"] are skipped (immutable once complete).
    - If a run completed between two auto-save ticks without being saved, it
      will be caught by the next tick (not in manifest → included).
    - pipeline_state.json and all_results.json are ALWAYS included (critical
      for pipeline resume after restart).
    """
    if not LOCAL_RES.exists():
        return False

    res_manifest: Dict[str, dict] = manifest.get("results", {})
    completed = _completed_run_names()

    to_save = completed if force_all else [r for r in completed if r not in res_manifest]

    # Collect per-run files
    files_to_bundle: List[Path] = []
    for run_name in to_save:
        for f in LOCAL_RES.iterdir():
            if f.is_file() and f.stem.startswith(run_name):
                files_to_bundle.append(f)

    # Always include state/summary files (small but critical for resume)
    for fname in ("all_results.json", "pipeline_state.json"):
        p = LOCAL_RES / fname
        if p.exists() and p not in files_to_bundle:
            files_to_bundle.append(p)

    if not files_to_bundle:
        print("  Results: nothing new to save.")
        return False

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_name = f"results_{ts}.tar.gz"
    tb_tmp  = LOCAL_ROOT / f".tmp_{tb_name}"
    tb_dst  = DRIVE_RES / tb_name

    DRIVE_RES.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(str(tb_tmp), "w:gz") as tf:
            for f in files_to_bundle:
                tf.add(str(f), arcname=f.name)
        shutil.move(str(tb_tmp), str(tb_dst))  # single Drive write
        for run_name in to_save:
            res_manifest[run_name] = {"tarball": tb_name, "saved_at": ts}
        manifest["results"] = res_manifest
        print(
            f"  Results: {len(to_save)} run(s), "
            f"{len(files_to_bundle)} file(s) → {tb_name}"
        )
        return True
    except Exception as e:
        print(f"  WARNING: results save failed: {e}")
        if tb_tmp.exists():
            tb_tmp.unlink()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def init() -> None:
    """
    Full init sequence (idempotent — safe to call multiple times):
      1. Mount Google Drive
      2. Copy code: MyDrive/inpainting/** → /content/inpainting/ (mtime-based)
      3. Restore manifest from Drive if local copy is missing
      4. Extract new checkpoint tarballs from Drive (skips already-extracted)
      5. Extract new results tarballs from Drive (skips already-extracted)
    """
    print("Initialising Colab environment...")
    _mount_drive()
    _copy_code_from_drive()
    LOCAL_CKPT.mkdir(parents=True, exist_ok=True)
    LOCAL_RES.mkdir(parents=True, exist_ok=True)

    manifest = _restore_manifest()

    # On a fresh runtime the checkpoints directory is empty even though the
    # manifest lists all previous tarballs as "already restored".  Reset the
    # restored-tarball tracking so every tarball is re-extracted.
    if not any(LOCAL_CKPT.glob("*.pt")):
        manifest["_restored_ckpt_tarballs"] = []
    if not any(LOCAL_RES.glob("*.json")):
        manifest["_restored_res_tarballs"] = []

    _, restored_ckpt = _extract_tarballs(
        DRIVE_CKPT, LOCAL_CKPT, "ckpt_*.tar.gz",
        set(manifest.get("_restored_ckpt_tarballs", [])),
        "Checkpoints",
    )
    _, restored_res = _extract_tarballs(
        DRIVE_RES, LOCAL_RES, "results_*.tar.gz",
        set(manifest.get("_restored_res_tarballs", [])),
        "Results",
    )
    manifest["_restored_ckpt_tarballs"] = sorted(restored_ckpt)
    manifest["_restored_res_tarballs"]  = sorted(restored_res)
    _save_manifest(manifest)
    print("Init done.\n")


def save(tag: str = "auto", force_all: bool = False) -> None:
    """
    Incremental tarball save to Drive (checkpoints + results).

    Args:
        tag:       Label for log output (e.g. 'epoch10', 'final').
        force_all: Re-save all files regardless of manifest state.
                   Use for final save at end of pipeline.
    """
    print(f"[{tag}] Saving to Drive...")
    manifest = _load_manifest()
    did_ckpt = _save_checkpoints_incremental(manifest, force_all=force_all)
    did_res  = _save_results_incremental(manifest, force_all=force_all)
    if did_ckpt or did_res:
        _save_manifest(manifest)
    else:
        print("  Nothing new to save.")
    print(f"[{tag}] Done.")


def incremental_save() -> None:
    """One-shot incremental save — called by the auto_save background thread."""
    manifest = _load_manifest()
    did_ckpt = _save_checkpoints_incremental(manifest)
    did_res  = _save_results_incremental(manifest)
    if did_ckpt or did_res:
        _save_manifest(manifest)


def auto_save(interval_min: float = 20) -> None:
    """
    Start a background daemon thread for incremental Drive saves.

    Drive I/O only happens when new completed runs or changed checkpoints
    are found.  Idle ticks are instant — local manifest read only, no
    Drive touch.

    Args:
        interval_min: Save interval in minutes (default 20).
    """
    def _loop():
        n = 0
        while True:
            time.sleep(interval_min * 60)
            n += 1
            ts = datetime.now().strftime("%H:%M")
            print(f"\n[auto_save tick {n} @ {ts}]", flush=True)
            try:
                incremental_save()
            except Exception as e:
                print(f"  auto_save warning: {e}", flush=True)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print(f"Auto-save started (every {interval_min:.0f} min, background daemon).")


def status() -> None:
    """Print a summary of local state vs what has been saved to Drive."""
    manifest  = _load_manifest()
    local_pts = sorted(LOCAL_CKPT.glob("*.pt")) if LOCAL_CKPT.exists() else []
    saved_pts = set(manifest.get("checkpoints", {}).keys())
    unsaved_pts = [p.name for p in local_pts if p.name not in saved_pts]

    completed    = _completed_run_names()
    saved_runs   = set(manifest.get("results", {}).keys())
    unsaved_runs = [r for r in completed if r not in saved_runs]

    ckpt_tbs = len({v["tarball"] for v in manifest.get("checkpoints", {}).values()})
    res_tbs  = len({v["tarball"] for v in manifest.get("results", {}).values()})

    print("── colab_setup status ────────────────────────────────────────")
    print(f"  Local checkpoints : {len(local_pts)} .pt file(s)")
    print(f"  Not yet on Drive  : {len(unsaved_pts)}  ({', '.join(unsaved_pts) or 'none'})")
    print(f"  Completed runs    : {len(completed)}")
    print(f"  Not yet on Drive  : {len(unsaved_runs)}  ({', '.join(unsaved_runs) or 'none'})")
    print(f"  Drive tarballs    : {ckpt_tbs} checkpoint, {res_tbs} results")
    print("─────────────────────────────────────────────────────────────")


def require_data() -> None:
    """Raise RuntimeError if Places365 sentinel is not present."""
    sentinel = LOCAL_DATA / "places365" / ".download_complete"
    if not sentinel.exists():
        raise RuntimeError(
            f"Places365 not found at {LOCAL_DATA / 'places365'}.\n"
            "Run Cell 1 first:\n"
            "  from data.download import download_all\n"
            f"  download_all(base_dir='{LOCAL_DATA}')"
        )
    print("  Data check passed.")
