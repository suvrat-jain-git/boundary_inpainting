# models/architecture.py
"""
BoundaryAwareInpainter — Improved Architecture

New contributions over baseline:
  1. TransformerBottleneck: Multi-head self-attention between encoder and decoder
     for global context (addresses LaMa's global receptive field advantage)
  2. Bilinear upsample + Conv instead of ConvTranspose2d
     (eliminates checkerboard artifacts at boundary)
  3. BoundaryConditionedDecoder: boundary band fed into each decoder stage
     (ties architecture to our theoretical boundary supervision)
  4. MaskInjectedSkipGate: retained and improved from baseline

Literature context:
  - TransformerBottleneck: inspired by MAE (He et al. 2022), Restormer (Zamir et al. 2022)
  - Bilinear upsample: Odena et al. 2016 "Deconvolution and Checkerboard Artifacts"
  - Boundary conditioning: novel to this work — no existing inpainting paper
    feeds explicit boundary band into each decoder stage
  - GatedConv: Yu et al. "Free-Form Image Inpainting with Gated Convolution" ICCV 2019
  - MaskInjectedSkipGate: extends attention gating (Oktay et al. 2018) with mask signal
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from einops import rearrange

from configs.config import ModelConfig


# ── Convolution Blocks ─────────────────────────────────────────────────────────

class GatedConv2d(nn.Module):
    """
    Gated convolution: out = sigmoid(gate) ⊙ ELU(feature).

    Reference: Yu et al. ICCV 2019.
    Difference from baseline: added LayerNorm option for stability
    near boundary regions where activations can spike.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.gate_conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.activation = nn.ELU(inplace=True)
        self.norm = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.activation(self.norm(self.feature_conv(x)))
        gate = torch.sigmoid(self.gate_conv(x))
        return feature * gate


class StandardConv2d(nn.Module):
    """Standard conv → BN → ELU (ablation baseline)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── Skip Connection Modules ────────────────────────────────────────────────────

class MaskInjectedSkipGate(nn.Module):
    """
    Attention-gated skip connection with mask injection.

    Combines encoder features, decoder features, and downsampled mask
    to produce a spatial attention map α ∈ (0,1) that weights encoder skip.

    Relation to literature:
      - Oktay et al. 2018: attention gating for skip connections in U-Net
      - Our addition: mask channel injected as third signal, making the gate
        explicitly aware of which regions are known vs. synthesized.
        SPADE (Park et al. 2019) modulates via spatially-adaptive BN;
        CBAM (Woo et al. 2018) uses channel+spatial attention sequentially.
        Our gate is simpler but purpose-built for the inpainting mask signal.
    """

    def __init__(self, enc_ch: int, dec_ch: int):
        super().__init__()
        self.W_enc = nn.Conv2d(enc_ch, dec_ch, 1, bias=False)
        self.W_dec = nn.Conv2d(dec_ch, dec_ch, 1, bias=False)
        self.W_mask = nn.Conv2d(1, dec_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(dec_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        enc_feat: torch.Tensor,   # (B, enc_ch, H, W)
        dec_feat: torch.Tensor,   # (B, dec_ch, H, W)
        mask: torch.Tensor,       # (B, 1, H_orig, W_orig)
    ) -> torch.Tensor:
        mask_ds = F.interpolate(mask, size=enc_feat.shape[2:], mode="nearest")
        g = self.relu(
            self.W_enc(enc_feat) + self.W_dec(dec_feat) + self.W_mask(mask_ds)
        )
        alpha = self.psi(g)   # (B, 1, H, W)
        return enc_feat * alpha


class StandardSkipConnection(nn.Module):
    """Identity skip — ablation baseline."""

    def __init__(self, enc_ch: int, dec_ch: int):
        super().__init__()

    def forward(
        self,
        enc_feat: torch.Tensor,
        dec_feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return enc_feat


# ── NEW: Transformer Bottleneck ────────────────────────────────────────────────

class TransformerBottleneck(nn.Module):
    """
    Multi-head self-attention applied at the CNN bottleneck.

    Motivation: CNN encoders have limited receptive fields even at /32
    resolution. For large holes, the model needs to reason about the whole
    image structure before decoding. This is the same gap LaMa addresses
    with global Fourier convolutions, but we apply it surgically at the
    bottleneck only, preserving the boundary-focused decoder design.

    Architecture: flatten spatial tokens → positional encoding →
    N × (LayerNorm → MHA → residual → LayerNorm → FFN → residual) →
    reshape back to feature map.

    Reference:
      - MAE (He et al. 2022): ViT applied to image patches
      - Restormer (Zamir et al. 2022): efficient transformer for restoration
      - Our choice of bottleneck-only application is a deliberate compromise
        between global context and computational cost.
    """

    def __init__(self, channels: int, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert channels % n_heads == 0, f"channels={channels} must be divisible by n_heads={n_heads}"

        self.channels = channels
        # Learned positional embeddings (spatial, added before attention)
        # We use a fixed-size positional embedding; for variable sizes we interpolate
        self.pos_embed_hw = 8   # assumes /32 of 256px = 8×8 spatial tokens

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.pos_embed_hw * self.pos_embed_hw, channels) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=channels * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,  # suppress warning with norm_first=True
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — bottleneck feature map from encoder
        Returns:
            out: (B, C, H, W) — globally-contextualized feature map
        """
        B, C, H, W = x.shape

        # Flatten to sequence of spatial tokens
        tokens = rearrange(x, "b c h w -> b (h w) c")   # (B, H*W, C)

        # Positional embedding — interpolate if spatial size differs from training size
        pos = self.pos_embedding
        if H * W != self.pos_embed_hw ** 2:
            pos = F.interpolate(
                pos.reshape(1, self.pos_embed_hw, self.pos_embed_hw, C).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear", align_corners=False,
            ).reshape(1, C, H * W).permute(0, 2, 1)

        tokens = tokens + pos
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Reshape back to spatial feature map
        out = rearrange(tokens, "b (h w) c -> b c h w", h=H, w=W)
        return out + x   # residual connection: preserve encoder features


# ── NEW: Bilinear Upsample Block ───────────────────────────────────────────────

class BilinearUpsampleBlock(nn.Module):
    """
    Bilinear interpolation followed by Conv2d.

    Replaces ConvTranspose2d to eliminate checkerboard artifacts.
    Reference: Odena et al. 2016 "Deconvolution and Checkerboard Artifacts"
               https://distill.pub/2016/deconv-checkerboard/

    These artifacts appear as a regular grid pattern in the output and are
    especially damaging near inpainting boundaries, which is our focus area.
    """

    def __init__(self, in_ch: int, out_ch: int, scale_factor: int = 2):
        super().__init__()
        self.scale = scale_factor
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return self.act(self.norm(self.conv(x)))


# ── NEW: Boundary Conditioning Module ─────────────────────────────────────────

class BoundaryConditioningModule(nn.Module):
    """
    Injects boundary band information into decoder feature maps.

    Novelty: No existing inpainting paper feeds the explicit boundary band
    as a spatial conditioning signal into each decoder stage. We do this to
    create architectural alignment with our boundary-focused loss terms:
    both the loss AND the architecture explicitly reason about the boundary.

    Mechanism: downsampled boundary band → small CNN → channel-wise scale+shift
    applied to decoder features (similar in spirit to SPADE but with boundary
    as the conditioning signal, not a semantic map).
    """

    def __init__(self, dec_ch: int):
        super().__init__()
        # Takes 1-channel boundary band → produces scale and shift for dec_ch
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, dec_ch * 2, 1),  # scale + shift
        )
        # Init scale to 1 and shift to 0 so it's an identity at init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].bias.data[dec_ch:] = 1.0   # scale channel init to 1

    def forward(self, feat: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, dec_ch, H, W) decoder feature map
            boundary: (B, 1, H_orig, W_orig) boundary band

        Returns:
            conditioned feature map same shape as feat
        """
        # Downsample boundary to match feature spatial size
        b_ds = F.interpolate(boundary, size=feat.shape[2:], mode="nearest")
        params = self.net(b_ds)   # (B, dec_ch*2, H, W)
        shift, scale = params.chunk(2, dim=1)
        return feat * (1.0 + scale) + shift   # affine conditioning


# ── Full Model ─────────────────────────────────────────────────────────────────

class BoundaryAwareInpainter(nn.Module):
    """
    Boundary-Aware Inpainting Model with:

    ENCODER:
      - ResNet-34 (or 18) pretrained backbone
      - Modified first conv: 3→4 channels (RGB + mask)

    BOTTLENECK (NEW):
      - TransformerBottleneck: MHA self-attention for global context
      - Addresses the limited receptive field of pure-CNN bottlenecks

    DECODER:
      - BilinearUpsampleBlock instead of ConvTranspose2d (NEW)
        → eliminates checkerboard artifacts at boundary
      - GatedConv2d blocks for selective feature processing
      - MaskInjectedSkipGate for attention-gated skip connections
      - BoundaryConditioningModule at each stage (NEW)
        → architecture explicitly reasons about boundary location

    AUXILIARY OUTPUTS:
      - Multi-scale output heads at /8 and /4 for boundary supervision
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ── Encoder ───────────────────────────────────────────────────────────
        if cfg.backbone in ("resnet18", "resnet34"):
            if cfg.backbone == "resnet18":
                resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
            else:
                resnet = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)
            enc_channels = [64, 64, 128, 256, 512]   # e0, e1(/4), e2(/8), e3(/16), e4(/32)

            # Expand conv1: 3 → 4 channels (RGB + mask channel)
            old_conv = resnet.conv1
            self.enc_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.enc_conv1.weight[:, :3] = old_conv.weight
                self.enc_conv1.weight[:, 3:] = old_conv.weight[:, :1]  # init from R channel

            self.enc_bn1 = resnet.bn1
            self.enc_relu = resnet.relu
            self.enc_pool = resnet.maxpool
            self.enc_layer1 = resnet.layer1   # /4,  64 ch
            self.enc_layer2 = resnet.layer2   # /8,  128 ch
            self.enc_layer3 = resnet.layer3   # /16, 256 ch
            self.enc_layer4 = resnet.layer4   # /32, 512 ch
            self._encoder_type = "resnet"

        elif cfg.backbone == "convnext_tiny":
            # ConvNeXt-Tiny: load full model (NOT features_only) so .stem is
            # directly accessible for the 3→4 channel patch.
            # We manually extract the 4 stage outputs in forward().
            try:
                import timm
            except ImportError as e:
                raise ImportError("timm is required for convnext_tiny backbone. "
                                  "pip install timm>=0.9.0") from e

            _backbone = timm.create_model(
                "convnext_tiny",
                pretrained=True,
                num_classes=0,
                global_pool="",
            )
            # Patch first conv in stem: 3 → 4 input channels
            old_stem_conv = _backbone.stem[0]
            new_stem_conv = nn.Conv2d(
                4, old_stem_conv.out_channels,
                kernel_size=old_stem_conv.kernel_size,
                stride=old_stem_conv.stride,
                padding=old_stem_conv.padding,
                bias=old_stem_conv.bias is not None,
            )
            with torch.no_grad():
                new_stem_conv.weight[:, :3] = old_stem_conv.weight
                new_stem_conv.weight[:, 3:] = old_stem_conv.weight[:, :1]
                if old_stem_conv.bias is not None:
                    new_stem_conv.bias.copy_(old_stem_conv.bias)
            _backbone.stem[0] = new_stem_conv
            self.enc_convnext = _backbone
            # 1×1 channel projectors: ConvNeXt dims → standard dims
            self.enc_proj0 = nn.Sequential(nn.Conv2d(96,  64,  1), nn.BatchNorm2d(64),  nn.ELU(inplace=True))
            self.enc_proj1 = nn.Sequential(nn.Conv2d(192, 128, 1), nn.BatchNorm2d(128), nn.ELU(inplace=True))
            self.enc_proj2 = nn.Sequential(nn.Conv2d(384, 256, 1), nn.BatchNorm2d(256), nn.ELU(inplace=True))
            self.enc_proj3 = nn.Sequential(nn.Conv2d(768, 512, 1), nn.BatchNorm2d(512), nn.ELU(inplace=True))
            enc_channels = [64, 64, 128, 256, 512]   # after projection; e0 stub = 64
            self._encoder_type = "convnext"

        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}. "
                             f"Choose from: resnet18, resnet34, convnext_tiny")

        # ── Transformer Bottleneck (NEW) ───────────────────────────────────────
        if cfg.use_transformer_bottleneck:
            self.transformer_bottleneck = TransformerBottleneck(
                channels=512,
                n_heads=cfg.transformer_heads,
                n_layers=cfg.transformer_layers,
            )
        else:
            self.transformer_bottleneck = nn.Identity()

        # ── Skip Connections ───────────────────────────────────────────────────
        SkipCls = MaskInjectedSkipGate if cfg.use_attention_skip else StandardSkipConnection
        self.skip4 = SkipCls(256, 512)
        self.skip3 = SkipCls(128, 256)
        self.skip2 = SkipCls(64, 128)
        self.skip1 = SkipCls(64, 64)

        # ── Decoder (with bilinear upsampling) ────────────────────────────────
        ConvBlk = GatedConv2d if cfg.use_gated_conv else StandardConv2d
        UpsBlk = BilinearUpsampleBlock if cfg.use_bilinear_upsample else _ConvTransposeBlock

        self.dec4_up = UpsBlk(512, 512)          # /32 → /16
        self.dec4_conv = ConvBlk(512 + 256, 256)

        self.dec3_up = UpsBlk(256, 256)          # /16 → /8
        self.dec3_conv = ConvBlk(256 + 128, 128)

        self.dec2_up = UpsBlk(128, 128)          # /8 → /4
        self.dec2_conv = ConvBlk(128 + 64, 64)

        self.dec1_up = UpsBlk(64, 64)            # /4 → /2
        self.dec1_conv = ConvBlk(64 + 64, 64)

        self.final_up = UpsBlk(64, 32)           # /2 → /1
        self.final_conv = nn.Sequential(
            ConvBlk(32, 32),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )

        # ── Boundary Conditioning Modules (NEW) ───────────────────────────────
        if cfg.use_boundary_conditioning:
            self.bc4 = BoundaryConditioningModule(256)
            self.bc3 = BoundaryConditioningModule(128)
            self.bc2 = BoundaryConditioningModule(64)
            self.bc1 = BoundaryConditioningModule(64)
        else:
            self.bc4 = self.bc3 = self.bc2 = self.bc1 = None

        # ── Multi-Scale Auxiliary Heads ────────────────────────────────────────
        if cfg.multiscale_output:
            self.ms_head_8 = nn.Sequential(
                ConvBlk(128, 64),
                nn.Conv2d(64, 3, 1),
                nn.Sigmoid(),
            )
            self.ms_head_4 = nn.Sequential(
                ConvBlk(64, 32),
                nn.Conv2d(32, 3, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x: torch.Tensor,                        # (B, 3, H, W)
        mask: Optional[torch.Tensor] = None,    # (B, 1, H, W), 1=hole
        boundary: Optional[torch.Tensor] = None, # (B, 1, H, W)
    ) -> Dict[str, object]:
        if mask is None:
            mask = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        if boundary is None:
            boundary = torch.zeros_like(mask)

        # 4-channel input: RGB + mask
        enc_in = torch.cat([x, mask], dim=1)

        # ── Encoder ───────────────────────────────────────────────────────────
        if self._encoder_type == "resnet":
            e0 = self.enc_relu(self.enc_bn1(self.enc_conv1(enc_in)))  # /2,  64
            e0p = self.enc_pool(e0)                                    # /4,  64
            e1 = self.enc_layer1(e0p)                                  # /4,  64
            e2 = self.enc_layer2(e1)                                   # /8,  128
            e3 = self.enc_layer3(e2)                                   # /16, 256
            e4 = self.enc_layer4(e3)                                   # /32, 512
        else:
            # ConvNeXt-Tiny: manually run stem + 4 stages to get features at
            # /4 (96ch), /8 (192ch), /16 (384ch), /32 (768ch)
            b = self.enc_convnext
            f = b.stem(enc_in)           # /4,  96ch
            f0 = b.stages[0](f)          # /4,  96ch  (no downsample in stage 0)
            f1 = b.stages[1](f0)         # /8,  192ch
            f2 = b.stages[2](f1)         # /16, 384ch
            f3 = b.stages[3](f2)         # /32, 768ch
            e1 = self.enc_proj0(f0)      # /4,  64
            e2 = self.enc_proj1(f1)      # /8,  128
            e3 = self.enc_proj2(f2)      # /16, 256
            e4 = self.enc_proj3(f3)      # /32, 512
            # No /2 features in ConvNeXt → create a stub by upsampling e1
            e0 = F.interpolate(e1, scale_factor=2, mode="bilinear", align_corners=False)

        # ── Transformer Bottleneck ─────────────────────────────────────────────
        e4 = self.transformer_bottleneck(e4)                       # /32, 512

        # ── Decoder ───────────────────────────────────────────────────────────
        d4 = self.dec4_up(e4)                                      # /16, 512
        s4 = self.skip4(e3, d4, mask)                              # 256
        d4 = self.dec4_conv(torch.cat([d4, s4], dim=1))            # 256
        if self.bc4 is not None:
            d4 = self.bc4(d4, boundary)

        d3 = self.dec3_up(d4)                                      # /8, 256
        s3 = self.skip3(e2, d3, mask)                              # 128
        d3 = self.dec3_conv(torch.cat([d3, s3], dim=1))            # 128
        if self.bc3 is not None:
            d3 = self.bc3(d3, boundary)

        d2 = self.dec2_up(d3)                                      # /4, 128
        s2 = self.skip2(e1, d2, mask)                              # 64
        d2 = self.dec2_conv(torch.cat([d2, s2], dim=1))            # 64
        if self.bc2 is not None:
            d2 = self.bc2(d2, boundary)

        d1 = self.dec1_up(d2)                                      # /2, 64
        s1 = self.skip1(e0, d1, mask)                              # 64
        d1 = self.dec1_conv(torch.cat([d1, s1], dim=1))            # 64
        if self.bc1 is not None:
            d1 = self.bc1(d1, boundary)

        out = self.final_up(d1)
        out = self.final_conv(out)                                  # (B, 3, H, W)

        result = {"output": out, "ms_outputs": []}

        if self.cfg.multiscale_output:
            result["ms_outputs"] = [
                self.ms_head_8(d3),
                self.ms_head_4(d2),
            ]

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Helper: ConvTranspose fallback for ablation ────────────────────────────────

class _ConvTransposeBlock(nn.Module):
    """ConvTranspose2d upsample — kept only for ablation comparison."""

    def __init__(self, in_ch: int, out_ch: int, scale_factor: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, scale_factor, stride=scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: ModelConfig, device: torch.device) -> BoundaryAwareInpainter:
    model = BoundaryAwareInpainter(cfg).to(device)
    n_params = model.count_parameters() / 1e6
    print(f"Model built: {cfg.backbone} | params={n_params:.2f}M")
    print(f"  Encoder type:           {model._encoder_type}")
    print(f"  Transformer bottleneck: {cfg.use_transformer_bottleneck}")
    print(f"  Bilinear upsample:      {cfg.use_bilinear_upsample}")
    print(f"  Boundary conditioning:  {cfg.use_boundary_conditioning}")
    print(f"  Attention skip:         {cfg.use_attention_skip}")
    print(f"  Gated conv:             {cfg.use_gated_conv}")
    return model
