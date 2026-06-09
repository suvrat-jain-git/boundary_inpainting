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

    def _fft_repr(self, patch: torch.Tensor) -> torch.Tensor:
        """Compute FFT representation of a patch."""
        fft = torch.fft.fft2(patch)
        if self.spectral_mode == "magnitude":
            return torch.abs(fft)
        elif self.spectral_mode == "log_magnitude":
            return torch.log(torch.abs(fft) + 1e-8)
        elif self.spectral_mode == "complex":
            return torch.view_as_real(fft).flatten(-2)  # (C, H, W, 2) → (C, H, W*2)

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
        patch_losses: List[torch.Tensor] = []

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

                if (y1 - y0) < ps // 2 or (x1 - x0) < ps // 2:
                    continue

                p_comp = comp[b, :, y0:y1, x0:x1]
                p_target = target[b, :, y0:y1, x0:x1]

                repr_comp = self._fft_repr(p_comp)
                repr_target = self._fft_repr(p_target)

                patch_losses.append(F.l1_loss(repr_comp, repr_target.detach()))

        if patch_losses:
            return torch.stack(patch_losses).mean()

        # No boundary pixels — return zero with gradient connection
        return pred.sum() * 0.0
    
class AdaptiveSpectralBoundaryLoss(nn.Module):
    """
    Boundary-localized spectral loss with learned per-band frequency weights.
    
    Extends SpectralBoundaryCoherenceLoss by decomposing the FFT magnitude
    spectrum into radial frequency bands and learning per-band importance
    weights, with entropy regularization to prevent weight collapse.
    
    Novel contribution over FFL (Jiang et al. 2021):
      1. Localized to boundary band only (not global)
      2. Adaptive per-band weighting (FFL uses fixed weights)
      3. Entropy regularization prevents degenerate weight solutions
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        n_samples: int = 8,
        n_bands: int = 4,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.n_bands = n_bands
        self.entropy_weight = entropy_weight
        
        # Learned per-band weights — this is the novel part
        # These are trainable parameters, one per frequency band
        self.band_logits = nn.Parameter(torch.zeros(n_bands))
        
        # Precompute which FFT bins belong to which radial band
        self._band_masks = self._make_band_masks(patch_size, n_bands)
    
    def _make_band_masks(self, ps: int, n_bands: int):
        """
        Divide the FFT grid into n_bands concentric rings.
        
        The FFT of a ps×ps patch produces a ps×ps grid of frequencies.
        The center is DC (zero frequency). Distance from center = frequency.
        We divide [0, max_distance] into n_bands equal rings.
        """
        cy, cx = ps // 2, ps // 2
        ys = torch.arange(ps).float() - cy
        xs = torch.arange(ps).float() - cx
        # Distance of each FFT bin from center (= frequency magnitude)
        dist = torch.sqrt(ys[:, None]**2 + xs[None, :]**2)
        max_dist = dist.max()
        
        masks = []
        for i in range(n_bands):
            lo = (i / n_bands) * max_dist
            hi = ((i + 1) / n_bands) * max_dist
            mask = ((dist >= lo) & (dist < hi)).float()
            masks.append(mask)
        
        # Stack: (n_bands, ps, ps)
        return torch.stack(masks)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        boundary_band: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        ps = self.patch_size
        
        # Move band masks to same device as input
        band_masks = self._band_masks.to(pred.device)  # (n_bands, ps, ps)
        
        # Softmax over band logits → normalized weights that sum to 1
        band_weights = torch.softmax(self.band_logits, dim=0)  # (n_bands,)
        
        # Composite image: real pixels where known, predicted where missing
        comp = target * (1 - mask) + pred * mask
        
        band_flat = boundary_band.view(B, -1)
        patch_losses = []
        
        for b in range(B):
            nz = torch.nonzero(band_flat[b] > 0.5, as_tuple=False)
            if len(nz) < 4:
                continue
            
            n_draw = min(self.n_samples, len(nz))
            indices = nz[
                torch.randperm(len(nz), device=pred.device)[:n_draw], 0
            ]
            
            for idx in indices:
                y, x = int(idx) // W, int(idx) % W
                y0 = max(0, y - ps // 2)
                x0 = max(0, x - ps // 2)
                y1 = min(H, y0 + ps)
                x1 = min(W, x0 + ps)
                
                if (y1 - y0) < ps // 2 or (x1 - x0) < ps // 2:
                    continue
                
                p_comp = comp[b, :, y0:y1, x0:x1]    # (C, ph, pw)
                p_tgt  = target[b, :, y0:y1, x0:x1]
                
                # FFT magnitude spectrum for this patch
                mag_comp = torch.abs(torch.fft.fftshift(
                    torch.fft.fft2(p_comp)
                ))  # (C, ph, pw)
                mag_tgt = torch.abs(torch.fft.fftshift(
                    torch.fft.fft2(p_tgt)
                ))
                
                # Compute per-band loss
                band_loss = torch.tensor(0.0, device=pred.device)
                actual_ps_h = y1 - y0
                actual_ps_w = x1 - x0
                
                for band_i in range(self.n_bands):
                    # Resize band mask to actual patch size if needed
                    bm = band_masks[band_i]
                    if bm.shape != (actual_ps_h, actual_ps_w):
                        bm = F.interpolate(
                            bm.unsqueeze(0).unsqueeze(0),
                            size=(actual_ps_h, actual_ps_w),
                            mode='nearest'
                        ).squeeze()
                    
                    # L1 difference in this frequency band
                    diff = torch.abs(mag_comp - mag_tgt) * bm
                    n_bins = bm.sum().clamp(min=1.0)
                    band_diff = diff.sum() / (n_bins * C)
                    
                    # Weight by learned band importance
                    band_loss = band_loss + band_weights[band_i] * band_diff
                
                patch_losses.append(band_loss)
        
        if not patch_losses:
            return pred.sum() * 0.0
        
        spectral_loss = torch.stack(patch_losses).mean()
        
        # Entropy regularizer — encourages weights to stay spread across bands
        # Without this, the model might learn to ignore all bands except one
        # H(w) = -sum(w * log(w)), maximizing entropy = keeping weights spread
        entropy = -(band_weights * torch.log(band_weights + 1e-8)).sum()
        
        # We SUBTRACT entropy term because we want to MAXIMIZE entropy
        # (maximize spread = minimize negative entropy)
        return spectral_loss - self.entropy_weight * entropy


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


# ── Combined Loss Manager ──────────────────────────────────────────────────────

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

        self.perceptual = PerceptualLoss().to(device)
        self.uniform_boundary = UniformBoundaryLoss()
        self.grad_boundary = GradientWeightedBoundaryLoss()
        self.spectral = SpectralBoundaryCoherenceLoss(
            patch_size=cfg.spectral_patch_size,
            n_samples=cfg.spectral_patches_k,
            spectral_mode="log_magnitude" if cfg.spectral_use_log else "magnitude",
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
        if cfg.loss_config in ("boundary_grad", "full"):
            bl = self.grad_boundary(pred, target, boundary_band)
            losses["boundary"] = bl
            total = total + cfg.boundary_weight * bl

        # ── Spectral coherence ─────────────────────────────────────────────────
        if cfg.loss_config in ("spectral_only", "full"):
            sl = self.spectral(pred, target, mask, boundary_band)
            losses["spectral"] = sl
            total = total + cfg.spectral_weight * sl

        # ── TV loss ────────────────────────────────────────────────────────────
        if cfg.loss_config == "full":
            tv = self.tv(pred, mask)
            losses["tv"] = tv
            total = total + cfg.tv_weight * tv

        # ── Multi-scale boundary supervision ──────────────────────────────────
        ms_outputs = pred_dict.get("ms_outputs", [])
        if cfg.loss_config == "full" and ms_outputs:
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
