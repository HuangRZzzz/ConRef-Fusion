import math

import torch
import torch.nn.functional as F


NEW_METRICS = ["MI", "Qabf", "QCB", "QCV", "QCC"]
METRIC_DIRECTIONS = {
    "MI": "higher",
    "Qabf": "higher",
    "QCB": "higher",
    "QCV": "lower",
    "QCC": "higher",
}


def _to_2d_tensor(image, device):
    return torch.as_tensor(image, dtype=torch.float32, device=device)


def _to_conv_tensor(image):
    return image.unsqueeze(0).unsqueeze(0)


def _normalize_to_255(image):
    data_min = torch.amin(image)
    data_max = torch.amax(image)
    if torch.abs(data_max - data_min) < 1e-12:
        return torch.zeros_like(image)
    return torch.round((image - data_min) / (data_max - data_min) * 255.0)


def _freqspace_mesh(height, width, device):
    if width % 2 == 0:
        u = torch.linspace(-1.0, 1.0 - 2.0 / width, width, device=device)
    else:
        u = torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, device=device)
    if height % 2 == 0:
        v = torch.linspace(-1.0, 1.0 - 2.0 / height, height, device=device)
    else:
        v = torch.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height, device=device)
    return torch.meshgrid(v, u, indexing="ij")


def _fft_filter(image, response):
    spectrum = torch.fft.fftshift(torch.fft.fft2(image))
    filtered = torch.fft.ifft2(torch.fft.ifftshift(spectrum * response))
    return torch.real(filtered)


def _conv2d_same(image, kernel):
    kernel = kernel.to(device=image.device, dtype=image.dtype)
    padding = (kernel.shape[-1] // 2, kernel.shape[-2] // 2)
    return F.conv2d(_to_conv_tensor(image), kernel.unsqueeze(0).unsqueeze(0), padding=padding).squeeze(0).squeeze(0)


def _gaussian2d_31(sigma, device):
    coords = torch.arange(-15, 16, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return torch.exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / (2.0 * math.pi * sigma * sigma)


def _contrast_map(g1, g2, image):
    numerator = _conv2d_same(image, g1)
    denominator = _conv2d_same(image, g2)
    return numerator / (denominator + 1e-12) - 1.0


def _block_reduce(image, block_size, reduce):
    height, width = image.shape
    block_h = height // block_size
    block_w = width // block_size
    if block_h <= 0 or block_w <= 0:
        return reduce(image.reshape(1, 1, -1), dim=2)
    cropped = image[: block_h * block_size, : block_w * block_size]
    blocks = cropped.reshape(block_h, block_size, block_w, block_size).permute(0, 2, 1, 3)
    return reduce(blocks.reshape(block_h, block_w, block_size * block_size), dim=2)


def _mutual_information(image_x, image_y):
    x = torch.round(image_x).clamp(0, 255).to(torch.long).reshape(-1)
    y = torch.round(image_y).clamp(0, 255).to(torch.long).reshape(-1)
    joint = x * 256 + y
    hist_2d = torch.bincount(joint, minlength=256 * 256).reshape(256, 256).float()
    pxy = hist_2d / torch.clamp(torch.sum(hist_2d), min=1.0)
    px = torch.sum(pxy, dim=1, keepdim=True)
    py = torch.sum(pxy, dim=0, keepdim=True)
    px_py = px @ py
    nz = pxy > 0
    return torch.sum(pxy[nz] * torch.log2(pxy[nz] / (px_py[nz] + 1e-12) + 1e-12))


def _qabf_arrays(image):
    h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=image.device)
    h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device)
    sx = _conv2d_same(image, h3)
    sy = _conv2d_same(image, h1)
    gradient = torch.sqrt(sx * sx + sy * sy)
    angle = torch.where(torch.abs(sx) < 1e-12, torch.full_like(sx, math.pi / 2.0), torch.atan(sy / (sx + 1e-12)))
    return gradient, angle


def _qabf_quality(source_angle, source_gradient, fused_angle, fused_gradient):
    tg = 0.9994
    kg = -15.0
    dg = 0.5
    ta = 0.9879
    ka = -22.0
    da = 0.8
    gaf = torch.where(
        source_gradient > fused_gradient,
        fused_gradient / (source_gradient + 1e-12),
        source_gradient / (fused_gradient + 1e-12),
    )
    gaf = torch.where(torch.abs(source_gradient - fused_gradient) < 1e-12, fused_gradient, gaf)
    aaf = 1.0 - torch.abs(source_angle - fused_angle) / (math.pi / 2.0)
    qgaf = tg / (1.0 + torch.exp(kg * (gaf - dg)))
    qaaf = ta / (1.0 + torch.exp(ka * (aaf - da)))
    return qgaf * qaaf


def _mi(fused, ir, vi):
    return _mutual_information(fused, ir) + _mutual_information(fused, vi)


def _qabf(fused, ir, vi):
    ga, aa = _qabf_arrays(ir)
    gb, ab = _qabf_arrays(vi)
    gf, af = _qabf_arrays(fused)
    qaf = _qabf_quality(aa, ga, af, gf)
    qbf = _qabf_quality(ab, gb, af, gf)
    return torch.sum(qaf * ga + qbf * gb) / (torch.sum(ga + gb) + 1e-12)


def _qcb(fused, ir, vi):
    im1 = _normalize_to_255(ir)
    im2 = _normalize_to_255(vi)
    fused = _normalize_to_255(fused)
    height, width = im1.shape
    device = im1.device

    y, x = _freqspace_mesh(height, width, device)
    radius = torch.sqrt((width / 30.0 * x) ** 2 + (height / 30.0 * y) ** 2)
    response = torch.exp(-((radius / 15.3870) ** 2)) - 0.7622 * torch.exp(-((radius / 1.3456) ** 2))

    im1_filtered = _fft_filter(im1, response)
    im2_filtered = _fft_filter(im2, response)
    fused_filtered = _fft_filter(fused, response)

    g1 = _gaussian2d_31(2.0, device)
    g2 = _gaussian2d_31(4.0, device)
    c1 = torch.abs(_contrast_map(g1, g2, im1_filtered))
    c2 = torch.abs(_contrast_map(g1, g2, im2_filtered))
    cf = torch.abs(_contrast_map(g1, g2, fused_filtered))

    c1p = torch.pow(c1, 3.0) / (torch.pow(c1, 2.0) + 0.0001)
    c2p = torch.pow(c2, 3.0) / (torch.pow(c2, 2.0) + 0.0001)
    cfp = torch.pow(cf, 3.0) / (torch.pow(cf, 2.0) + 0.0001)

    q1f = torch.where(c1p < cfp, c1p / (cfp + 1e-12), cfp / (c1p + 1e-12))
    q2f = torch.where(c2p < cfp, c2p / (cfp + 1e-12), cfp / (c2p + 1e-12))
    energy = c1p * c1p + c2p * c2p + 1e-12
    quality = (c1p * c1p / energy) * q1f + (c2p * c2p / energy) * q2f
    return torch.nan_to_num(torch.mean(quality), nan=0.0, posinf=0.0, neginf=0.0)


def _qcv(fused, ir, vi):
    im1 = _normalize_to_255(ir)
    im2 = _normalize_to_255(vi)
    fused = _normalize_to_255(fused)
    height, width = im1.shape
    device = im1.device

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    im1_g = torch.sqrt(_conv2d_same(im1, sobel_x).pow(2) + _conv2d_same(im1, sobel_y).pow(2))
    im2_g = torch.sqrt(_conv2d_same(im2, sobel_x).pow(2) + _conv2d_same(im2, sobel_y).pow(2))
    lambda1 = _block_reduce(torch.pow(im1_g, 5.0), 16, torch.sum)
    lambda2 = _block_reduce(torch.pow(im2_g, 5.0), 16, torch.sum)

    y, x = _freqspace_mesh(height, width, device)
    radius = torch.sqrt((width / 8.0 * x) ** 2 + (height / 8.0 * y) ** 2)
    theta_m = 2.6 * (0.0192 + 0.144 * radius) * torch.exp(-torch.pow(0.144 * radius, 1.1))
    df1 = _fft_filter(im1 - fused, theta_m)
    df2 = _fft_filter(im2 - fused, theta_m)
    d1 = _block_reduce(df1.pow(2), 16, torch.mean)
    d2 = _block_reduce(df2.pow(2), 16, torch.mean)
    value = torch.sum(lambda1 * d1 + lambda2 * d2) / (torch.sum(lambda1 + lambda2) + 1e-12)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _qcc(fused, ir, vi):
    def corr(source):
        source_centered = source - torch.mean(source)
        fused_centered = fused - torch.mean(fused)
        return torch.sum(source_centered * fused_centered) / (
            torch.sqrt(torch.sum(source_centered.pow(2)) * torch.sum(fused_centered.pow(2))) + 1e-12
        )

    return 0.5 * (corr(ir) + corr(vi))


@torch.no_grad()
def compute_new_metrics(fused, ir, vi, device):
    fused_tensor = _to_2d_tensor(fused, device)
    ir_tensor = _to_2d_tensor(ir, device)
    vi_tensor = _to_2d_tensor(vi, device)
    metrics = {
        "MI": _mi(fused_tensor, ir_tensor, vi_tensor),
        "Qabf": _qabf(fused_tensor, ir_tensor, vi_tensor),
        "QCB": _qcb(fused_tensor, ir_tensor, vi_tensor),
        "QCV": _qcv(fused_tensor, ir_tensor, vi_tensor),
        "QCC": _qcc(fused_tensor, ir_tensor, vi_tensor),
    }
    return {name: float(value.detach().cpu()) for name, value in metrics.items()}
