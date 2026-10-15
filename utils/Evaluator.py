import math

import numpy as np
import torch
import torch.nn.functional as F


def _as_tensor(image):
    return torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def _conv2d(image, kernel, mode):
    tensor = _as_tensor(image)
    kernel_tensor = torch.tensor(np.array(kernel, copy=True), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel_h, kernel_w = kernel_tensor.shape[-2:]

    if mode == "same":
        padding = (kernel_w // 2, kernel_h // 2)
        out = F.conv2d(tensor, kernel_tensor, padding=padding)
    elif mode == "valid":
        out = F.conv2d(tensor, kernel_tensor, padding=0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return out.squeeze(0).squeeze(0).cpu().numpy()


def _gaussian_kernel(size, sigma):
    center = (size - 1) / 2.0
    y, x = np.ogrid[:size, :size]
    kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2.0 * sigma * sigma))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel = kernel / kernel_sum
    return kernel.astype(np.float32)


def _structural_similarity(image_x, image_y, window_size=11, sigma=1.5):
    x = _as_tensor(image_x)
    y = _as_tensor(image_y)
    window = torch.tensor(_gaussian_kernel(window_size, sigma), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding)
    mu_y = F.conv2d(y, window, padding=padding)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-8)
    return float(ssim_map.mean().item())


def _mutual_information(image_x, image_y):
    x = np.uint8(np.round(image_x)).ravel()
    y = np.uint8(np.round(image_y)).ravel()
    hist_2d, _, _ = np.histogram2d(x, y, bins=256, range=[[0, 255], [0, 255]])
    pxy = hist_2d / np.maximum(hist_2d.sum(), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    px_py = px @ py
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2((pxy[nz] / px_py[nz]) + 1e-12)))


def _normalize_to_255(image):
    data = np.asarray(image, dtype=np.float64)
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == 0 and data_min == 0:
        return data.copy()
    if abs(data_max - data_min) < 1e-12:
        return np.zeros_like(data)
    return np.round((data - data_min) / (data_max - data_min) * 255.0)


def _freqspace_mesh(height, width):
    if width % 2 == 0:
        u = np.linspace(-1.0, 1.0 - 2.0 / width, width)
    else:
        u = np.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width)
    if height % 2 == 0:
        v = np.linspace(-1.0, 1.0 - 2.0 / height, height)
    else:
        v = np.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height)
    return np.meshgrid(u, v)


def _fft_filter(image, response):
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * response))
    return np.real(filtered)


def _gaussian2d_31(sigma):
    coords = np.arange(-15, 16, dtype=np.float64)
    x, y = np.meshgrid(coords, coords)
    return np.exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / (2.0 * math.pi * sigma * sigma)


def _contrast_map(g1, g2, image):
    numerator = _conv2d(np.asarray(image, dtype=np.float32), g1.astype(np.float32), mode="same")
    denominator = _conv2d(np.asarray(image, dtype=np.float32), g2.astype(np.float32), mode="same")
    return numerator / (denominator + 1e-12) - 1.0


def _block_reduce(image, block_size, reducer):
    height, width = image.shape
    block_h = height // block_size
    block_w = width // block_size
    if block_h <= 0 or block_w <= 0:
        return np.asarray([[reducer(image)]], dtype=np.float64)
    cropped = image[: block_h * block_size, : block_w * block_size]
    blocks = cropped.reshape(block_h, block_size, block_w, block_size).transpose(0, 2, 1, 3)
    return reducer(blocks.reshape(block_h, block_w, block_size * block_size), axis=2)


class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a, minlength=256) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return float(np.std(img))

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return float(np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2)))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        gx, gy = np.zeros_like(img), np.zeros_like(img)

        gx[:, 0] = img[:, 1] - img[:, 0]
        gx[:, -1] = img[:, -1] - img[:, -2]
        gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        gy[0, :] = img[1, :] - img[0, :]
        gy[-1, :] = img[-1, :] - img[-2, :]
        gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2)))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return _mutual_information(image_F, image_A) + _mutual_information(image_F, image_B)

    @classmethod
    def MSE(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return float((np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2)

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)) + 1e-12)
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)) + 1e-12)
        return float((rAF + rBF) / 2)

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return float(10 * np.log10(np.max(image_F) ** 2 / (cls.MSE(image_F, image_A, image_B) + 1e-12)))

    @classmethod
    def SCD(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)) + 1e-12)
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)) + 1e-12)
        return float(corr1 + corr2)

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F) + cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls, ref, dist):
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            win = _gaussian_kernel(N, sd)

            if scale > 1:
                ref = _conv2d(ref, np.rot90(win, 2), mode='valid')[::2, ::2]
                dist = _conv2d(dist, np.rot90(win, 2), mode='valid')[::2, ::2]

            mu1 = _conv2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = _conv2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = _conv2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = _conv2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = _conv2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / (den + eps)
        return 1.0 if np.isnan(vifp) else float(vifp)

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return float(nume / (deno + 1e-12))

    @classmethod
    def Qabf_getArray(cls, img):
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

        SAx = _conv2d(img, h3, mode='same')
        SAy = _conv2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0] = np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls, aA, gA, aF, gF):
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF = np.zeros_like(aA)
        GAF[gA > gF] = gF[gA > gF] / (gA[gA > gF] + 1e-12)
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA < gF] = gA[gA < gF] / (gF[gA < gF] + 1e-12)
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    @classmethod
    def QCB(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        im1 = _normalize_to_255(image_A)
        im2 = _normalize_to_255(image_B)
        fused = _normalize_to_255(image_F)

        height, width = im1.shape
        f0 = 15.3870
        f1 = 1.3456
        a = 0.7622
        k = 1.0
        h = 1.0
        p = 3.0
        q = 2.0
        z = 0.0001

        u, v = _freqspace_mesh(height, width)
        r = np.sqrt((width / 30.0 * u) ** 2 + (height / 30.0 * v) ** 2)
        sd = np.exp(-((r / f0) ** 2)) - a * np.exp(-((r / f1) ** 2))

        im1_filtered = _fft_filter(im1, sd)
        im2_filtered = _fft_filter(im2, sd)
        fused_filtered = _fft_filter(fused, sd)

        g1 = _gaussian2d_31(2.0)
        g2 = _gaussian2d_31(4.0)
        c1 = np.abs(_contrast_map(g1, g2, im1_filtered))
        c2 = np.abs(_contrast_map(g1, g2, im2_filtered))
        cf = np.abs(_contrast_map(g1, g2, fused_filtered))

        c1p = (k * np.power(c1, p)) / (h * np.power(c1, q) + z)
        c2p = (k * np.power(c2, p)) / (h * np.power(c2, q) + z)
        cfp = (k * np.power(cf, p)) / (h * np.power(cf, q) + z)

        q1f = np.where(c1p < cfp, c1p / (cfp + 1e-12), cfp / (c1p + 1e-12))
        q2f = np.where(c2p < cfp, c2p / (cfp + 1e-12), cfp / (c2p + 1e-12))

        energy = c1p * c1p + c2p * c2p + 1e-12
        lambda1 = (c1p * c1p) / energy
        lambda2 = (c2p * c2p) / energy
        quality = lambda1 * q1f + lambda2 * q2f
        return float(np.nan_to_num(np.mean(quality), nan=0.0, posinf=0.0, neginf=0.0))

    @classmethod
    def QCV(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        im1 = _normalize_to_255(image_A)
        im2 = _normalize_to_255(image_B)
        fused = _normalize_to_255(image_F)

        height, width = im1.shape
        block_size = 16
        alpha = 5.0

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        im1_x = _conv2d(im1.astype(np.float32), sobel_x, mode="same")
        im1_y = _conv2d(im1.astype(np.float32), sobel_y, mode="same")
        im2_x = _conv2d(im2.astype(np.float32), sobel_x, mode="same")
        im2_y = _conv2d(im2.astype(np.float32), sobel_y, mode="same")
        im1_g = np.sqrt(im1_x * im1_x + im1_y * im1_y)
        im2_g = np.sqrt(im2_x * im2_x + im2_y * im2_y)

        lambda1 = _block_reduce(np.power(im1_g, alpha), block_size, np.sum)
        lambda2 = _block_reduce(np.power(im2_g, alpha), block_size, np.sum)

        f1 = im1 - fused
        f2 = im2 - fused
        u, v = _freqspace_mesh(height, width)
        r = np.sqrt((width / 8.0 * u) ** 2 + (height / 8.0 * v) ** 2)
        theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-np.power(0.144 * r, 1.1))

        df1 = _fft_filter(f1, theta_m)
        df2 = _fft_filter(f2, theta_m)
        d1 = _block_reduce(np.square(df1), block_size, np.mean)
        d2 = _block_reduce(np.square(df2), block_size, np.mean)

        numerator = np.sum(lambda1 * d1 + lambda2 * d2)
        denominator = np.sum(lambda1 + lambda2) + 1e-12
        return float(np.nan_to_num(numerator / denominator, nan=0.0, posinf=0.0, neginf=0.0))

    @classmethod
    def QCC(cls, image_F, image_A, image_B):
        return cls.CC(image_F, image_A, image_B)

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return _structural_similarity(image_F, image_A) + _structural_similarity(image_F, image_B)
