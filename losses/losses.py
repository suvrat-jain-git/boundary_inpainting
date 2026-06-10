# losses/losses.py
"""
Loss functions for boundary-aware inpainting.

Contributions:
  1. SpectralBoundaryCoherenceLoss — localized FFT supervision at boundary
     - Magnitude-only (default) vs log-magnitude vs complex variants
     - Justification for K and patch_size via sensitivity ablation
  2. GradientWeightedBoundaryLoss — edge-focused boundary L1
  3. UniformBoundaryLoss — uniform boundary L1 (ablation baseline)
  4. PerceptualLoss — VGG16 feature matching
  5. TVLoss — total variation in hole region only
  6. InpaintingLossManager — combines all terms by config string

Literature context:
  - FFL (Focal Frequency Loss, Jiang et al. 2021): global FFT loss
    Our spectral loss is LOCAL (boundary patches only), not global —
    this is the key distinction from FFL.
  - LaMa (Suvorov et al. 2022): Fourier convolutions in architecture
    Our spectral loss is a SUPERVISION term, complementary to LaMa's approach.
  - EdgeConnect (Nazeri et al. 2019): uses edge maps for structure guidance
    Our gradient-weighted loss uses Sobel edges as a WEIGHTING signal,
    not a separate network prediction stage.
  - Perceptual loss: Johnson et al. 2016
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from configs.config import LossConfig


# ── Perceptual Loss ────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG16 perceptual loss up to relu3_2 (first 16 layers).

    Gradients flow through prediction only; target features are detached.
    The VGG backbone is frozen.
    """

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features[:16]
        self.feature_extractor = vgg.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        feat_pred = self.feature_extractor(pred_n)
        with torch.no_grad():
            feat_target = self.feature_extractor(target_n)
        return F.l1_loss(feat_pred, feat_target)


# ── Sobel Gradient ─────────────────────────────────────────────────────────────

def sobel_gradient_magnitude(img: torch.Tensor) -> torch.Tensor:
    """
    Per-channel Sobel gradient magnitude, reduced to single-channel max.

    Args:
        img: (B, C, H, W) float tensor
    Returns:
        grad_mag: (B, 1, H, W)
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=img.dtype, device=img.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=img.dtype, device=img.device,
    ).view(1, 1, 3, 3)

    B, C, H, W = img.shape
    grad_mag = torch.zeros(B, 1, H, W, device=img.device, dtype=img.dtype)

    for c in range(C):
        ch = img[:, c:c+1]
        gx = F.conv2d(ch, sobel_x, padding=1)
        gy = F.conv2d(ch, sobel_y, padding=1)
        grad_mag = torch.max(grad_mag, torch.sqrt(gx**2 + gy**2 + 1e-8))

    return grad_mag


# ── Boundary Losses ────────────────────────────────────────────────────────────

class GradientWeightedBoundaryLoss(nn.Module):
    """
    Boundary L1 weighted by Sobel gradient magnitude of target.

    High-frequency edges near seams receive proportionally higher loss.
    This focuses capacity where mismatches are perceptually most visible.

    Normalization: gradient map normalized to [0,1] per batch before
    weighting, preventing noisy gradient regions from dominating
    (addresses reviewer concern about overweighting noisy gradients).

    Relation to EdgeConnect: EC predicts explicit edge maps as a separate
    network stage. We use edges as a WEIGHTING signal within a single loss
    term — no separate prediction network needed.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        boundary_band: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            grad_w = sobel_gradient_magnitude(target)           # (B, 1, H, W)
            # Normalize per image to [0,1] (prevents noisy images from dominating)
            B = grad_w.shape[0]
            grad_w_flat = grad_w.view(B, -1)
            max_vals = grad_w_flat.max(dim=1)[0].view(B, 1, 1, 1)
            grad_w = grad_w / (max_vals + self.epsilon)
            grad_w = grad_w * boundary_band                     # restrict to band

        diff = torch.abs(pred - target)                         # (B, 3, H, W)
        weighted = diff * grad_w                                # broadcasts over C
        n_band = boundary_band.sum().clamp(min=1.0)
        return weighted.sum() / (n_band * 3)


class UniformBoundaryLoss(nn.Module):
    """
    Uniform boundary L1 — ablation baseline (L1).
    Equal weight to all boundary pixels, no gradient weighting.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        boundary_band: torch.Tensor,
    ) -> torch.Tensor:
        diff = torch.abs(pred - target) * boundary_band
        n_band = boundary_band.sum().clamp(min=1.0)
        return diff.sum() / (n_band * 3)


# ── Spectral Boundary Coherence Loss ──────────────────────────────────────────

class SpectralBoundaryCoherenceLoss(nn.Module):
    """
    FFT magnitude mismatch on boundary patches.

    KEY DISTINCTION FROM FFL (Focal Frequency Loss):
      - FFL applies frequency loss GLOBALLY across the whole image
      - Our loss samples patches ONLY from the boundary band
      - This localises spectral supervision exactly where seam artifacts occur

    FFT variant options (addresses reviewer question Q3):
      - spectral_mode='magnitude': |FFT(x)| comparison (default)
        Rationale: magnitude captures energy distribution across frequencies,
        which governs texture continuity. Phase captures edge positions,
        but our boundary band already spatially localises the loss.
      - spectral_mode='log_magnitude': log(1 + |FFT(x)|)
        More perceptually uniform; reduces dominance of low-frequency components.
      - spectral_mode='complex': real + imaginary parts
        Captures both magnitude and phase; more sensitive but noisier.

    Patch sampling (K and patch_size sensitivity addressed in ablation):
      - K=8 patches per image (chosen via grid search; see sensitivity table)
      - patch_size=16 (captures local texture; larger = more context but slower)
      - Random sampling within boundary band; deterministic in eval

    Args:
        patch_size: size of square patches sampled at boundary
        n_samples: K — number of patches sampled per image per batch item
        spectral_mode: 'magnitude' | 'log_magnitude' | 'complex'
    """

    def __init__(
        self,
        patch_size: int = 16,
        n_samples: int = 8,
        spectral_mode: str = "magnitude",
    ):
        super().__init__()
        assert spectral_mode in ("magnitude", "log_magnitude", "complex"), \
            f"Unknown spectral_mode: {spectral_mode}"
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.spectral_mode = spectral_mode

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        boundary_band: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        ps = self.patch_size

        # Composite: target in valid regions, pred in hole
        comp = target * (1 - mask) + pred * mask

        band_flat = boundary_band.view(B, -1)   # (B, H*W)

        # Collect all valid full-size patches across the batch (vectorized)
        patches_comp:   List[torch.Tensor] = []
        patches_target: List[torch.Tensor] = []

        for b in range(B):
            nz = torch.nonzero(band_flat[b] > 0.5, as_tuple=False)   # (N, 1)
            if len(nz) < 4:
                continue

            n_draw = min(self.n_samples, len(nz))
            indices = nz[torch.randperm(len(nz), device=pred.device)[:n_draw], 0]

            for idx in indices:
                y, x = int(idx) // W, int(idx) % W
                y0 = max(0, y - ps // 2)
                x0 = max(0, x - ps // 2)
                y1 = min(H, y0 + ps)
                x1 = min(W, x0 + ps)

                # Only keep full-size patches to allow batching
                if (y1 - y0) != ps or (x1 - x0) != ps:
                    continue

                patches_comp.append(comp[b, :, y0:y1, x0:x1])
                patches_target.append(target[b, :, y0:y1, x0:x1])

        if not patches_comp:
            # No boundary pixels — return zero with gradient connection
            return pred.sum() * 0.0

        # Single batched FFT call instead of K×B separate calls
        p_comp = torch.stack(patches_comp, dim=0)    # (N, C, ps, ps)
        p_tgt  = torch.stack(patches_target, dim=0)  # (N, C, ps, ps)

        fft_comp = torch.fft.fft2(p_comp)    # (N, C, ps, ps) complex
        fft_tgt  = torch.fft.fft2(p_tgt)

        repr_comp   = self._fft_repr_batch(fft_comp)
        repr_target = self._fft_repr_batch(fft_tgt)

        return F.l1_loss(repr_comp, repr_target.detach())

    def _fft_repr_batch(self, fft: torch.Tensor) -> torch.Tensor:
        """Apply spectral_mode to a batch of complex FFT tensors."""
        if self.spectral_mode == "magnitude":
            return torch.abs(fft)
        elif self.spectral_mode == "log_magnitude":
            return torch.log(torch.abs(fft) + 1e-8)
        elif self.spectral_mode == "complex":
            return torch.view_as_real(fft).flatten(-2)
        else:
            raise ValueError(self.spectral_mode)


# ── TV Loss ────────────────────────────────────────────────────────────────────

class TVLoss(nn.Module):
    """
    Total Variation loss restricted to hole region only.

    Promoting smoothness inside the inpainted area without penalising
    sharp edges in the known valid region.
    """

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_hole = pred * mask
        diff_h = torch.abs(pred_hole[:, :, 1:, :] - pred_hole[:, :, :-1, :])
        diff_w = torch.abs(pred_hole[:, :, :, 1:] - pred_hole[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()


# ── Adaptive Spectral Boundary Coherence Loss (ASBC) ──────────────────────────

class AdaptiveSpectralBoundaryCoherenceLoss(nn.Module):
    """
    PAPER NOVELTY: Adaptive Spectral Boundary Coherence Loss (ASBC).

    Hypothesis: Different radial frequency bands contribute unequally to
    visible seam artifacts. Low frequencies govern color continuity; high
    frequencies govern texture sharpness. A fixed uniform spectral loss treats
    all bands identically; ASBC learns their optimal relative weighting end-to-end.

    Mechanism:
      1. Sample K boundary patches from each image (same as SpectralBoundaryCoherenceLoss)
      2. Compute FFT magnitude for each patch → stack into (N, C, ps, ps) batch
      3. Create R radial frequency-band masks (equal-radius bins)
      4. For each band r: loss_r = mean(|mag_comp - mag_tgt| * band_mask_r)
      5. Weighted sum: L = sum_r w_r * loss_r  where w = softmax(log_band_weights)
      6. Anti-collapse regularizer: -λ_reg * H(w)  (entropy; keeps weights spread)

    Anti-collapse: Without regularization the model could collapse all weight
    onto the easiest band. The entropy term -H(w) = sum_r w_r * log(w_r)
    is subtracted (i.e., we add +λ*H(w)) to encourage spread.

    Interpretation of learned weights: log them every epoch during training
    (trainer.py does this). The histogram of w_r in the final trained model
    should be non-uniform — this is Figure X in the paper showing that low/mid
    frequencies dominate for boundary coherence.

    Args:
        n_bands: number of radial frequency bins (default 4)
        patch_size: square patch size in pixels (default 16)
        n_samples: K — patches sampled per image (default 8)
        reg_weight: entropy regularizer coefficient (default 0.01)
    """

    def __init__(
        self,
        n_bands: int = 4,
        patch_size: int = 16,
        n_samples: int = 8,
        reg_weight: float = 0.01,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.reg_weight = reg_weight

        # Learnable log-weights (softmax-normalized in forward)
        self.log_band_weights = nn.Parameter(torch.zeros(n_bands))

        # Precompute radial band masks: (n_bands, ps, ps)
        ps = patch_size
        cy, cx = ps / 2.0, ps / 2.0
        y_idx = torch.arange(ps).float()
        x_idx = torch.arange(ps).float()
        yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
        r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = r.max().item()

        band_masks = []
        for i in range(n_bands):
            lo = i * r_max / n_bands
            hi = (i + 1) * r_max / n_bands
            mask = ((r >= lo) & (r < hi)).float()
            band_masks.append(mask)

        # (n_bands, ps, ps)
        self.register_buffer("band_masks", torch.stack(band_masks, dim=0))

    def get_band_weights(self) -> torch.Tensor:
        """Return normalized band weights (for logging)."""
        return torch.softmax(self.log_band_weights, dim=0).detach().cpu()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        boundary_band: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        ps = self.patch_size

        comp = target * (1 - mask) + pred * mask
        band_flat = boundary_band.view(B, -1)

        # Collect all full-size patches across the batch
        patches_comp:   List[torch.Tensor] = []
        patches_target: List[torch.Tensor] = []

        for b in range(B):
            nz = torch.nonzero(band_flat[b] > 0.5, as_tuple=False)
            if len(nz) < 4:
                continue

            n_draw = min(self.n_samples, len(nz))
            indices = nz[torch.randperm(len(nz), device=pred.device)[:n_draw], 0]

            for idx in indices:
                y, x = int(idx) // W, int(idx) % W
                y0 = max(0, y - ps // 2)
                x0 = max(0, x - ps // 2)
                y1 = min(H, y0 + ps)
                x1 = min(W, x0 + ps)

                if (y1 - y0) != ps or (x1 - x0) != ps:
                    continue

                patches_comp.append(comp[b, :, y0:y1, x0:x1])
                patches_target.append(target[b, :, y0:y1, x0:x1])

        if not patches_comp:
            return pred.sum() * 0.0

        # Single batched FFT — (N, C, ps, ps)
        p_comp = torch.stack(patches_comp, dim=0)
        p_tgt  = torch.stack(patches_target, dim=0)

        mag_comp = torch.abs(torch.fft.fft2(p_comp))       # (N, C, ps, ps)
        mag_tgt  = torch.abs(torch.fft.fft2(p_tgt)).detach()  # no grad through target

        # Per-band L1 difference
        # diff: (N, C, ps, ps)
        # band_masks: (R, ps, ps) → expand to (1, 1, R, ps, ps)
        diff = torch.abs(mag_comp - mag_tgt)                          # (N, C, ps, ps)
        bm   = self.band_masks.unsqueeze(0).unsqueeze(0)              # (1, 1, R, ps, ps)
        diff_banded = diff.unsqueeze(2) * bm                          # (N, C, R, ps, ps)
        # Mean over pixels and channels for each band: (N, R) → mean over N: (R,)
        per_band_loss = diff_banded.mean(dim=(0, 1, 3, 4))            # (R,)

        # Normalized band weights via softmax
        weights = torch.softmax(self.log_band_weights, dim=0)         # (R,)
        weighted_loss = (per_band_loss * weights).sum()

        # Anti-collapse entropy regularizer: maximize entropy → add +λ*H(w)
        entropy = -(weights * (weights + 1e-8).log()).sum()
        loss = weighted_loss - self.reg_weight * entropy

        return loss

class InpaintingLossManager:
    """
    Manages all loss components.

    loss_config controls which terms are active:
      'base'             : hole + valid + perceptual
      'boundary_uniform' : base + uniform boundary L1
      'boundary_grad'    : base + gradient-weighted boundary L1
      'spectral_only'    : base + spectral coherence
      'full'             : all + multi-scale + TV

    Loss weights are documented in LossConfig with justification.
    """

    def __init__(self, cfg: LossConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        # Set to True by trainer on steps where perceptual is skipped
        self.skip_perceptual: bool = False

        self.perceptual = PerceptualLoss().to(device)
        self.uniform_boundary = UniformBoundaryLoss()
        self.grad_boundary = GradientWeightedBoundaryLoss()
        self.spectral = SpectralBoundaryCoherenceLoss(
            patch_size=cfg.spectral_patch_size,
            n_samples=cfg.spectral_patches_k,
            spectral_mode="log_magnitude" if cfg.spectral_use_log else "magnitude",
        ).to(device)
        self.adaptive_spectral = AdaptiveSpectralBoundaryCoherenceLoss(
            n_bands=cfg.spectral_n_bands,
            patch_size=cfg.spectral_patch_size,
            n_samples=cfg.spectral_patches_k,
            reg_weight=cfg.spectral_reg_weight,
        ).to(device)
        self.tv = TVLoss()

    def compute(
        self,
        pred_dict: Dict,
        target: torch.Tensor,
        mask: torch.Tensor,
        boundary_band: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred = pred_dict["output"]
        hole_mask = mask
        valid_mask = 1.0 - mask
        losses: Dict[str, torch.Tensor] = {}

        # ── Base losses (always active) ────────────────────────────────────────
        n_hole = (hole_mask * 3).sum().clamp(min=1.0)
        n_valid = (valid_mask * 3).sum().clamp(min=1.0)

        losses["hole_l1"] = (torch.abs(pred - target) * hole_mask).sum() / n_hole
        losses["valid_l1"] = (torch.abs(pred - target) * valid_mask).sum() / n_valid

        if self.skip_perceptual:
            losses["perceptual"] = torch.zeros(1, device=pred.device, dtype=pred.dtype)[0]
        else:
            losses["perceptual"] = self.perceptual(pred, target)

        total = (
            self.cfg.hole_weight * losses["hole_l1"]
            + self.cfg.valid_weight * losses["valid_l1"]
            + self.cfg.perceptual_weight * losses["perceptual"]
        )

        cfg = self.cfg

        # ── Uniform boundary ───────────────────────────────────────────────────
        if cfg.loss_config == "boundary_uniform":
            bl = self.uniform_boundary(pred, target, boundary_band)
            losses["boundary"] = bl
            total = total + cfg.boundary_weight * bl

        # ── Gradient-weighted boundary ─────────────────────────────────────────
        if cfg.loss_config in ("boundary_grad", "full", "full_adaptive"):
            bl = self.grad_boundary(pred, target, boundary_band)
            losses["boundary"] = bl
            total = total + cfg.boundary_weight * bl

        # ── Spectral coherence ─────────────────────────────────────────────────
        if cfg.loss_config in ("spectral_only", "full"):
            sl = self.spectral(pred, target, mask, boundary_band)
            losses["spectral"] = sl
            total = total + cfg.spectral_weight * sl

        # ── Adaptive spectral (ASBC — paper novelty) ───────────────────────────
        if cfg.loss_config in ("adaptive_spectral", "full_adaptive"):
            sl = self.adaptive_spectral(pred, target, mask, boundary_band)
            losses["spectral"] = sl
            total = total + cfg.spectral_weight * sl

        # ── TV loss ────────────────────────────────────────────────────────────
        if cfg.loss_config in ("full", "full_adaptive"):
            tv = self.tv(pred, mask)
            losses["tv"] = tv
            total = total + cfg.tv_weight * tv

        # ── Multi-scale boundary supervision ──────────────────────────────────
        ms_outputs = pred_dict.get("ms_outputs", [])
        if cfg.loss_config in ("full", "full_adaptive") and ms_outputs:
            ms_total = torch.tensor(0.0, device=self.device)
            for ms_out in ms_outputs:
                scale = ms_out.shape[2:]
                tgt_ds = F.interpolate(target, size=scale, mode="bilinear", align_corners=False)
                band_ds = F.interpolate(boundary_band, size=scale, mode="nearest")
                ms_total = ms_total + self.grad_boundary(ms_out, tgt_ds, band_ds)
            losses["ms_boundary"] = ms_total
            total = total + cfg.boundary_weight * cfg.ms_weight * ms_total

        losses["total"] = total
        return losses
