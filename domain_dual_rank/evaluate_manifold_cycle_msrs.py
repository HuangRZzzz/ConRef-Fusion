import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from manifold_cycle_net import build_manifold_cycle_model


MMIF_ROOT = Path("/mnt/e/ubuntu/MMIF-CDDFuse")
if MMIF_ROOT.exists():
    sys.path.insert(0, str(MMIF_ROOT))

from utils.Evaluator import Evaluator  # noqa: E402
from utils.img_read_save import img_save  # noqa: E402


DEFAULT_METRICS = ["EN", "SD", "SF", "MI", "SCD", "VIFF", "Qabf", "SSIM"]
DEFAULT_SCORE_METRICS = ["SF", "MI", "SCD", "VIFF", "Qabf", "SSIM"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the light latent-mixing IVIF model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="/mnt/e/ubuntu/MSRS_dataset")
    parser.add_argument("--datasets", nargs="+", default=["eval"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None, help="Ignored; kept only for old command compatibility.")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def normalize_fused_tensor(tensor):
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    return (tensor - min_value) / (max_value - min_value + 1e-8)


def fusion_quality_score(result):
    return sum(result[name] for name in DEFAULT_SCORE_METRICS)


def build_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    model = build_manifold_cycle_model(
        latent_channels=int(args.get("latent_channels", 32)),
        encoder_stem_channels=int(args.get("encoder_stem_channels", 32)),
        lambda_fusion=float(args.get("lambda_fusion", 1.0)),
        lambda_latent_grad=float(args.get("lambda_latent_grad", 0.03)),
        lambda_edge_smooth=float(args.get("lambda_edge_smooth", 0.05)),
        edge_smooth_alpha=float(args.get("edge_smooth_alpha", 10.0)),
        fusion_gradient_weight=float(args.get("fusion_gradient_weight", 10.0)),
        encoder_block_conv=args.get("encoder_block_conv", "standard"),
        decoder_in_proj_conv=args.get("decoder_in_proj_conv", "standard"),
        decoder_block_conv=args.get("decoder_block_conv", "standard"),
        mixing_mode=args.get("mixing_mode", "equal"),
    )
    load_info = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return checkpoint, model, load_info


@torch.no_grad()
def infer_single(model, vis_image, ir_image, device, steps=None):
    del steps
    vis = torch.from_numpy(vis_image / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
    ir = torch.from_numpy(ir_image / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
    fused = model.generate(vis, ir)["fused"]
    fused = normalize_fused_tensor(fused)
    return np.squeeze((fused * 255.0).cpu().numpy())


def evaluate_dataset(dataset_name, dataset_root, output_root, model, device, steps, save_images):
    dataset_dir = Path(dataset_root) / dataset_name
    ir_dir = dataset_dir / "ir"
    vi_dir = dataset_dir / "vi"
    image_names = sorted(path.name for path in ir_dir.iterdir() if path.is_file())
    metrics_sum = np.zeros(len(DEFAULT_METRICS), dtype=np.float64)
    image_count = 0

    if save_images:
        image_output_dir = output_root / dataset_name / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)

    for image_name in image_names:
        ir = read_gray(ir_dir / image_name)
        vi = read_gray(vi_dir / image_name)
        fused = infer_single(model, vi, ir, device, steps)
        fused_gray = np.round(fused).astype(np.float32)

        if save_images:
            img_save(fused_gray, Path(image_name).stem, str(image_output_dir))

        metrics_sum += np.array(
            [
                Evaluator.EN(fused_gray),
                Evaluator.SD(fused_gray),
                Evaluator.SF(fused_gray),
                Evaluator.MI(fused_gray, ir, vi),
                Evaluator.SCD(fused_gray, ir, vi),
                Evaluator.VIFF(fused_gray, ir, vi),
                Evaluator.Qabf(fused_gray, ir, vi),
                Evaluator.SSIM(fused_gray, ir, vi),
            ]
        )
        image_count += 1

    metrics_mean = metrics_sum / max(image_count, 1)
    result = {name: float(value) for name, value in zip(DEFAULT_METRICS, metrics_mean)}
    result["Score"] = fusion_quality_score(result)
    return result


def write_outputs(output_root, checkpoint_path, checkpoint, load_info, results):
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "model_variant": "light_latent_mixing",
                "architecture": "encoder + latent mixing + z_fused-only decoder",
                "checkpoint_epoch": checkpoint.get("epoch"),
                "score_formula": "SF + MI + SCD + VIFF + Qabf + SSIM",
                "score_metrics": DEFAULT_SCORE_METRICS,
                "load_missing_keys": list(load_info.missing_keys),
                "load_unexpected_keys": list(load_info.unexpected_keys),
                "results": results,
            },
            handle,
            indent=2,
        )

    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", *DEFAULT_METRICS, "Score"])
        for dataset_name, result in results.items():
            writer.writerow([dataset_name, *[result[name] for name in DEFAULT_METRICS], result["Score"]])


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint, model, load_info = build_from_checkpoint(checkpoint_path, device)
    results = {}
    for dataset_name in args.datasets:
        result = evaluate_dataset(
            dataset_name,
            args.dataset_root,
            output_root,
            model,
            device,
            args.steps,
            args.save_images,
        )
        results[dataset_name] = result
        print("=" * 80)
        print(f"The test result of {dataset_name}:")
        print("\t\t EN\t SD\t SF\t MI\t SCD\t VIF\t Qabf\t SSIM\t Score")
        print("checkpoint\t" + "\t".join(str(np.round(result[name], 4)) for name in [*DEFAULT_METRICS, "Score"]))
        print("=" * 80)

    write_outputs(output_root, checkpoint_path, checkpoint, load_info, results)

    if args.summary_dir:
        summary_dir = Path(args.summary_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        name = "eval_best" if checkpoint_path.name == "best.pth" else "eval_final"
        shutil.copy2(output_root / "metrics.json", summary_dir / f"{name}_metrics.json")
        shutil.copy2(output_root / "metrics.csv", summary_dir / f"{name}_metrics.csv")


if __name__ == "__main__":
    main()
