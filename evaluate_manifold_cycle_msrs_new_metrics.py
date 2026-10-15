import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from evaluate_manifold_cycle_msrs import (
    build_from_checkpoint,
    infer_single,
    read_gray,
    resolve_device,
)
from utils.img_read_save import img_save
from utils.new_fusion_metrics_torch import METRIC_DIRECTIONS, NEW_METRICS, compute_new_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the light latent-mixing model with extended metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="/mnt/e/ubuntu/MSRS_dataset")
    parser.add_argument("--datasets", nargs="+", default=["eval"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None, help="Ignored; kept only for old command compatibility.")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def evaluate_dataset(dataset_name, dataset_root, output_root, model, device, steps, save_images):
    dataset_dir = Path(dataset_root) / dataset_name
    ir_dir = dataset_dir / "ir"
    vi_dir = dataset_dir / "vi"
    image_names = sorted(path.name for path in ir_dir.iterdir() if path.is_file())
    metrics_sum = np.zeros(len(NEW_METRICS), dtype=np.float64)
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

        result = compute_new_metrics(fused_gray, ir, vi, device)
        metrics_sum += np.array([result[name] for name in NEW_METRICS], dtype=np.float64)
        image_count += 1

    metrics_mean = metrics_sum / max(image_count, 1)
    return {name: float(value) for name, value in zip(NEW_METRICS, metrics_mean)}


def write_outputs(output_root, checkpoint_path, checkpoint, load_info, results):
    with (output_root / "new_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "model_variant": "light_latent_mixing",
                "architecture": "encoder + latent mixing + z_fused-only decoder",
                "checkpoint_epoch": checkpoint.get("epoch"),
                "metrics": NEW_METRICS,
                "metric_directions": METRIC_DIRECTIONS,
                "load_missing_keys": list(load_info.missing_keys),
                "load_unexpected_keys": list(load_info.unexpected_keys),
                "results": results,
            },
            handle,
            indent=2,
        )

    with (output_root / "new_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", *NEW_METRICS])
        for dataset_name, result in results.items():
            writer.writerow([dataset_name, *[result[name] for name in NEW_METRICS]])


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
        print(f"The new metric result of {dataset_name}:")
        print("\t\t " + "\t ".join(NEW_METRICS))
        print("checkpoint\t" + "\t".join(str(np.round(result[name], 4)) for name in NEW_METRICS))
        print("=" * 80)

    write_outputs(output_root, checkpoint_path, checkpoint, load_info, results)
    if args.summary_dir:
        summary_dir = Path(args.summary_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        prefix = "new_eval_best" if checkpoint_path.name == "best.pth" else "new_eval_final"
        shutil.copy2(output_root / "new_metrics.json", summary_dir / f"{prefix}_metrics.json")
        shutil.copy2(output_root / "new_metrics.csv", summary_dir / f"{prefix}_metrics.csv")


if __name__ == "__main__":
    main()
