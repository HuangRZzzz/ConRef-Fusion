import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels, max_groups=8):
    for groups in (max_groups, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _spatial_gradient(x):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_x = kernel_x.expand(x.shape[1], 1, 3, 3).contiguous()
    kernel_y = kernel_y.expand(x.shape[1], 1, 3, 3).contiguous()
    grad_x = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
    grad_y = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
    return torch.abs(grad_x) + torch.abs(grad_y)


def _normalize_spatial_map(x, eps=1e-6):
    min_value = x.amin(dim=(-2, -1), keepdim=True)
    max_value = x.amax(dim=(-2, -1), keepdim=True)
    return (x - min_value) / (max_value - min_value + eps)


def _local_mean(x, kernel_size=9):
    pad = kernel_size // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size, stride=1)


def _zero_param_reliability_prior(vis, ir, target_size=None, eps=1e-6):
    vis_gray = vis[:, :1]
    ir_gray = ir[:, :1]

    ir_intensity = _normalize_spatial_map(ir_gray)
    ir_contrast = _normalize_spatial_map(torch.abs(ir_gray - _local_mean(ir_gray)))
    ir_saliency = _normalize_spatial_map(ir_intensity + ir_contrast)

    vis_texture = _normalize_spatial_map(_spatial_gradient(vis_gray))
    ir_weight = (ir_saliency + eps) / (ir_saliency + vis_texture + 2.0 * eps)
    vis_weight = 1.0 - ir_weight

    if target_size is not None and vis_weight.shape[-2:] != target_size:
        vis_weight = F.interpolate(vis_weight, size=target_size, mode="bilinear", align_corners=False)
        ir_weight = F.interpolate(ir_weight, size=target_size, mode="bilinear", align_corners=False)
    return vis_weight, ir_weight


class ConvResidualBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None, conv_mode="standard"):
        super().__init__()
        hidden_channels = channels if hidden_channels is None else hidden_channels
        self.net = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            conv3x3(channels, hidden_channels, mode=conv_mode),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            conv3x3(hidden_channels, channels, mode=conv_mode),
        )

    def forward(self, x):
        return x + self.net(x)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def conv3x3(in_channels, out_channels, mode="standard"):
    if mode == "standard":
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
    if mode == "depthwise":
        return DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
    raise ValueError(f"Unsupported conv mode: {mode}")


class LatentEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=32, stem_channels=32, num_blocks=2, block_conv_mode="standard"):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(stem_channels, latent_channels, kernel_size=1, bias=True),
        )
        self.blocks = nn.Sequential(*[ConvResidualBlock(latent_channels, conv_mode=block_conv_mode) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(self.project(x))


class LatentDecoder(nn.Module):
    def __init__(
        self,
        latent_channels=32,
        out_channels=1,
        hidden_channels=48,
        num_blocks=2,
        in_proj_conv_mode="standard",
        block_conv_mode="standard",
    ):
        super().__init__()
        self.in_proj = conv3x3(latent_channels, hidden_channels, mode=in_proj_conv_mode)
        self.blocks = nn.Sequential(*[ConvResidualBlock(hidden_channels, conv_mode=block_conv_mode) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, z):
        return torch.sigmoid(self.out_proj(self.blocks(self.in_proj(z))))


class LatentAdapter(nn.Module):
    def __init__(self, latent_channels=32, hidden_channels=16, use_difference=True):
        super().__init__()
        hidden_channels = max(8, int(hidden_channels))
        in_channels = latent_channels * (2 if use_difference else 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=1, bias=True),
        )

    def forward(self, z_base, z_vis, z_ir):
        if self.net[0].in_channels == z_base.shape[1]:
            features = z_base
        else:
            features = torch.cat([z_base, torch.abs(z_vis - z_ir)], dim=1)
        return self.net(features)


class GatedLatentResidualAdapter(nn.Module):
    def __init__(self, latent_channels=32, hidden_channels=16):
        super().__init__()
        hidden_channels = max(8, int(hidden_channels))
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels * 3, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=1, bias=True),
        )

    def forward(self, z_base, z_vis, z_ir):
        difference = z_vis - z_ir
        features = torch.cat([z_base, difference, torch.abs(difference)], dim=1)
        gate = torch.sigmoid(self.net(features))
        residual = (gate - 0.5) * difference
        return residual, gate


class ConsensusResidualAdapter(nn.Module):
    def __init__(self, latent_channels=32, hidden_channels=16):
        super().__init__()
        hidden_channels = max(8, int(hidden_channels))
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels * 3, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=1, bias=True),
        )

    def forward(self, z_base, z_vis, z_ir):
        difference = z_vis - z_ir
        features = torch.cat([z_base, difference, torch.abs(difference)], dim=1)
        return torch.tanh(self.net(features))


class FusionImageLoss(nn.Module):
    def __init__(self, gradient_weight=10.0):
        super().__init__()
        self.gradient_weight = gradient_weight

    def forward(self, fused, vis, ir):
        target_intensity = torch.maximum(vis[:, :1], ir[:, :1])
        intensity_loss = F.l1_loss(fused, target_intensity)
        target_grad = torch.maximum(_spatial_gradient(vis[:, :1]), _spatial_gradient(ir[:, :1]))
        fused_grad = _spatial_gradient(fused)
        gradient_loss = F.l1_loss(fused_grad, target_grad)
        return intensity_loss + self.gradient_weight * gradient_loss, intensity_loss, gradient_loss


class ManifoldCycleFusionNet(nn.Module):
    """Light-only infrared-visible fusion network.

    Kept path:
        VIS/IR -> separate latent encoders -> latent mixing
        z_fused -> z_fused-only decoder -> fused image

    Only this path is implemented in the final light code.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        latent_channels=32,
        encoder_stem_channels=32,
        lambda_fusion=1.0,
        lambda_latent_grad=0.03,
        lambda_edge_smooth=0.05,
        edge_smooth_alpha=10.0,
        fusion_gradient_weight=10.0,
        encoder_block_conv="standard",
        decoder_in_proj_conv="standard",
        decoder_block_conv="standard",
        mixing_mode="equal",
        latent_adapter_mode="none",
        latent_adapter_hidden_channels=16,
        latent_adapter_beta_max=0.1,
        **_unused_kwargs,
    ):
        super().__init__()
        if mixing_mode not in {"equal", "channel", "region_prior"}:
            raise ValueError(f"mixing_mode must be 'equal', 'channel', or 'region_prior', got {mixing_mode!r}.")
        if latent_adapter_mode not in {
            "none",
            "residual",
            "residual_clamp",
            "residual_no_diff",
            "gated_residual",
            "consensus_residual",
        }:
            raise ValueError(
                "latent_adapter_mode must be 'none', 'residual', 'residual_clamp', or "
                f"'residual_no_diff', 'gated_residual', or 'consensus_residual', got {latent_adapter_mode!r}."
            )
        for name, value in {
            "encoder_block_conv": encoder_block_conv,
            "decoder_in_proj_conv": decoder_in_proj_conv,
            "decoder_block_conv": decoder_block_conv,
        }.items():
            if value not in {"standard", "depthwise"}:
                raise ValueError(f"{name} must be 'standard' or 'depthwise', got {value!r}.")
        self.latent_channels = latent_channels
        self.lambda_fusion = lambda_fusion
        self.lambda_latent_grad = lambda_latent_grad
        self.lambda_edge_smooth = lambda_edge_smooth
        self.edge_smooth_alpha = edge_smooth_alpha
        self.encoder_block_conv = encoder_block_conv
        self.decoder_in_proj_conv = decoder_in_proj_conv
        self.decoder_block_conv = decoder_block_conv
        self.mixing_mode = mixing_mode
        self.latent_adapter_mode = latent_adapter_mode
        self.latent_adapter_hidden_channels = latent_adapter_hidden_channels
        self.latent_adapter_beta_max = float(latent_adapter_beta_max)

        self.vis_encoder = LatentEncoder(
            in_channels,
            latent_channels,
            stem_channels=encoder_stem_channels,
            block_conv_mode=encoder_block_conv,
        )
        self.ir_encoder = LatentEncoder(
            in_channels,
            latent_channels,
            stem_channels=encoder_stem_channels,
            block_conv_mode=encoder_block_conv,
        )
        self.fusion_decoder = LatentDecoder(
            latent_channels,
            out_channels,
            in_proj_conv_mode=decoder_in_proj_conv,
            block_conv_mode=decoder_block_conv,
        )
        self.fusion_loss = FusionImageLoss(gradient_weight=fusion_gradient_weight)
        if self.mixing_mode == "channel":
            self.mix_logit = nn.Parameter(torch.zeros(1, latent_channels, 1, 1))
        else:
            self.register_parameter("mix_logit", None)
        if self.latent_adapter_mode == "none":
            self.latent_adapter = None
            self.register_parameter("latent_adapter_beta_raw", None)
        elif self.latent_adapter_mode == "gated_residual":
            self.latent_adapter = GatedLatentResidualAdapter(
                latent_channels=latent_channels,
                hidden_channels=latent_adapter_hidden_channels,
            )
            self.latent_adapter_beta_raw = nn.Parameter(torch.zeros(()))
        elif self.latent_adapter_mode == "consensus_residual":
            self.latent_adapter = ConsensusResidualAdapter(
                latent_channels=latent_channels,
                hidden_channels=latent_adapter_hidden_channels,
            )
            self.latent_adapter_beta_raw = nn.Parameter(torch.zeros(()))
        else:
            self.latent_adapter = LatentAdapter(
                latent_channels=latent_channels,
                hidden_channels=latent_adapter_hidden_channels,
                use_difference=latent_adapter_mode != "residual_no_diff",
            )
            self.latent_adapter_beta_raw = nn.Parameter(torch.zeros(()))

    def encode_modalities(self, vis, ir):
        return self.vis_encoder(vis), self.ir_encoder(ir)

    def mixing_alpha(self, reference, vis=None, ir=None):
        if self.mixing_mode == "region_prior":
            if vis is None or ir is None:
                raise ValueError("vis and ir are required when mixing_mode='region_prior'.")
            vis_weight, _ = _zero_param_reliability_prior(vis, ir, target_size=reference.shape[-2:])
            return vis_weight.to(dtype=reference.dtype, device=reference.device)
        if self.mix_logit is None:
            return reference.new_full((1, reference.shape[1], 1, 1), 0.5)
        return torch.sigmoid(self.mix_logit).to(dtype=reference.dtype, device=reference.device)

    def mix_latents(self, z_vis, z_ir, vis=None, ir=None):
        alpha = self.mixing_alpha(z_vis, vis=vis, ir=ir)
        return alpha * z_vis + (1.0 - alpha) * z_ir, alpha

    def effective_adapter_beta(self, reference):
        if self.latent_adapter_beta_raw is None:
            return reference.new_zeros(())
        beta = self.latent_adapter_beta_raw.to(dtype=reference.dtype, device=reference.device)
        if self.latent_adapter_mode in {"residual_clamp", "gated_residual", "consensus_residual"}:
            beta = self.latent_adapter_beta_max * torch.tanh(beta)
        return beta

    def apply_latent_adapter(self, z_base, z_vis, z_ir):
        if self.latent_adapter is None:
            zero = z_base.new_zeros(())
            return z_base, zero, zero, zero, zero, zero
        if self.latent_adapter_mode == "gated_residual":
            residual, gate = self.latent_adapter(z_base, z_vis, z_ir)
            gate_mean = gate.mean()
            gate_shift_l1 = torch.abs(gate - 0.5).mean()
        else:
            residual = self.latent_adapter(z_base, z_vis, z_ir)
            gate_mean = z_base.new_zeros(())
            gate_shift_l1 = z_base.new_zeros(())
        beta = self.effective_adapter_beta(z_base)
        delta = beta * residual
        return z_base + delta, beta, residual.abs().mean(), delta.abs().mean(), gate_mean, gate_shift_l1

    def latent_gradient_loss(self, z_fused, z_vis, z_ir):
        fused_grad = _spatial_gradient(z_fused)
        target_grad = torch.maximum(_spatial_gradient(z_vis), _spatial_gradient(z_ir))
        return F.l1_loss(fused_grad, target_grad)

    def edge_aware_smooth_loss(self, fused, vis, ir):
        fused_dx = torch.abs(fused[:, :, :, 1:] - fused[:, :, :, :-1])
        fused_dy = torch.abs(fused[:, :, 1:, :] - fused[:, :, :-1, :])
        vis_dx = torch.abs(vis[:, :1, :, 1:] - vis[:, :1, :, :-1])
        vis_dy = torch.abs(vis[:, :1, 1:, :] - vis[:, :1, :-1, :])
        ir_dx = torch.abs(ir[:, :1, :, 1:] - ir[:, :1, :, :-1])
        ir_dy = torch.abs(ir[:, :1, 1:, :] - ir[:, :1, :-1, :])
        edge_dx = torch.maximum(vis_dx, ir_dx).detach()
        edge_dy = torch.maximum(vis_dy, ir_dy).detach()
        weight_dx = torch.exp(-self.edge_smooth_alpha * edge_dx)
        weight_dy = torch.exp(-self.edge_smooth_alpha * edge_dy)
        return (fused_dx * weight_dx).mean() + (fused_dy * weight_dy).mean()

    def cycle_loss(self, vis, ir, return_images=True, **_unused_kwargs):
        z_vis, z_ir = self.encode_modalities(vis, ir)
        z_base, mix_alpha = self.mix_latents(z_vis, z_ir, vis=vis, ir=ir)
        (
            z_fused,
            adapter_beta,
            adapter_residual_l1,
            adapter_delta_l1,
            adapter_gate_mean,
            adapter_gate_shift_l1,
        ) = self.apply_latent_adapter(z_base, z_vis, z_ir)
        fused = self.fusion_decoder(z_fused)

        loss_fusion, loss_intensity, loss_gradient = self.fusion_loss(fused, vis, ir)
        loss_latent_grad = self.latent_gradient_loss(z_fused, z_vis, z_ir)
        loss_edge_smooth = self.edge_aware_smooth_loss(fused, vis, ir)
        loss = (
            self.lambda_fusion * loss_fusion
            + self.lambda_latent_grad * loss_latent_grad
            + self.lambda_edge_smooth * loss_edge_smooth
        )

        outputs = {
            "loss": loss,
            "loss_fusion": loss_fusion,
            "loss_intensity": loss_intensity,
            "loss_gradient": loss_gradient,
            "loss_latent_grad": loss_latent_grad,
            "loss_edge_smooth": loss_edge_smooth,
            "adapter_beta": adapter_beta,
            "adapter_residual_l1": adapter_residual_l1,
            "adapter_delta_l1": adapter_delta_l1,
            "adapter_gate_mean": adapter_gate_mean,
            "adapter_gate_shift_l1": adapter_gate_shift_l1,
        }
        if return_images:
            outputs.update({"fused": fused, "z_fused": z_fused, "z_base": z_base, "mix_alpha": mix_alpha})
        return outputs

    @torch.no_grad()
    def generate(self, vis, ir, **_unused_kwargs):
        z_vis, z_ir = self.encode_modalities(vis, ir)
        z_base, mix_alpha = self.mix_latents(z_vis, z_ir, vis=vis, ir=ir)
        (
            z_fused,
            adapter_beta,
            adapter_residual_l1,
            adapter_delta_l1,
            adapter_gate_mean,
            adapter_gate_shift_l1,
        ) = self.apply_latent_adapter(z_base, z_vis, z_ir)
        fused = self.fusion_decoder(z_fused)
        return {
            "fused": fused,
            "z_fused": z_fused,
            "z_base": z_base,
            "mix_alpha": mix_alpha,
            "adapter_beta": adapter_beta,
            "adapter_residual_l1": adapter_residual_l1,
            "adapter_delta_l1": adapter_delta_l1,
            "adapter_gate_mean": adapter_gate_mean,
            "adapter_gate_shift_l1": adapter_gate_shift_l1,
        }

    def forward(self, vis, ir):
        return self.generate(vis, ir)["fused"]


def build_manifold_cycle_model(**kwargs):
    return ManifoldCycleFusionNet(**kwargs)
