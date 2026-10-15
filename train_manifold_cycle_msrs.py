import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from manifold_cycle_net import build_manifold_cycle_model


DEFAULT_DATA_PATHS = (
    "/mnt/e/ubuntu/MMIF-CDDFuse/data/MSRS_dataset_train1_imgsize_128_stride_200.h5",
    "E:/ubuntu/MMIF-CDDFuse/data/MSRS_dataset_train1_imgsize_128_stride_200.h5",
)
DEFAULT_RUN_NAME = "msrs_light_latent_mixing_h5_128"
LOSS_KEYS = (
    "loss",
    "loss_fusion",
    "loss_intensity",
    "loss_gradient",
    "loss_latent_grad",
    "loss_edge_smooth",
)


class MSRSH5PatchDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as h5_file:
            for group in ("vis_patchs", "ir_patchs"):
                if group not in h5_file:
                    raise KeyError(f"Missing group '{group}' in {self.h5_path}")
            self.keys = list(h5_file["ir_patchs"].keys())

        if not self.keys:
            raise ValueError(f"No patches found in {self.h5_path}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        with h5py.File(self.h5_path, "r") as h5_file:
            vis = np.asarray(h5_file["vis_patchs"][key], dtype=np.float32)
            ir = np.asarray(h5_file["ir_patchs"][key], dtype=np.float32)
        return self._to_chw_tensor(vis), self._to_chw_tensor(ir)

    @staticmethod
    def _to_chw_tensor(array):
        if array.ndim == 2:
            array = array[None, :, :]
        elif array.ndim == 3 and array.shape[0] not in (1, 3):
            array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        return tensor[:1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the light latent-mixing IVIF model on H5 patches.")
    parser.add_argument("--data-path", default=None, help="Path to an H5 patch file.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", default=None, help="Run output directory. Defaults to <code>/runs/<run-name>.")
    parser.add_argument("--summary-dir", default="/mnt/e/ubuntu/summary", help="Summary root directory.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--encoder-stem-channels", type=int, default=32)
    parser.add_argument("--lambda-fusion", type=float, default=1.0)
    parser.add_argument("--lambda-latent-grad", type=float, default=0.03)
    parser.add_argument("--lambda-edge-smooth", type=float, default=0.05)
    parser.add_argument("--edge-smooth-alpha", type=float, default=10.0)
    parser.add_argument("--fusion-gradient-weight", type=float, default=10.0)
    parser.add_argument("--encoder-block-conv", choices=("standard", "depthwise"), default="standard")
    parser.add_argument("--decoder-in-proj-conv", choices=("standard", "depthwise"), default="standard")
    parser.add_argument("--decoder-block-conv", choices=("standard", "depthwise"), default="standard")
    parser.add_argument("--mixing-mode", choices=("equal", "channel", "region_prior"), default="equal")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    return parser.parse_args()


def resolve_existing_path(path_or_none, candidates):
    if path_or_none:
        path = Path(path_or_none)
        if path.is_file():
            return path
        raise FileNotFoundError(f"File does not exist: {path}")
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find an H5 file. Pass --data-path explicitly.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(device_arg):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def to_float_dict(outputs):
    values = {}
    for key in LOSS_KEYS:
        value = outputs[key]
        values[key] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    return values


def average_meter_init():
    return {key: 0.0 for key in LOSS_KEYS}, 0


def average_meter_update(sums, count, values, batch_size):
    for key, value in values.items():
        sums[key] += value * batch_size
    return count + batch_size


def average_meter_compute(sums, count):
    if count <= 0:
        return {key: 0.0 for key in LOSS_KEYS}
    return {key: sums[key] / count for key in LOSS_KEYS}


def save_checkpoint(path, model, optimizer, scaler, epoch, best_loss, args, dataset_size):
    checkpoint = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "dataset_size": dataset_size,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)


def copy_to_summary(output_dir, summary_dir):
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best.pth", "final.pth", "latest.pth", "config.json", "status.json", "train_log.csv"):
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, summary_dir / name)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    data_path = resolve_existing_path(args.data_path, DEFAULT_DATA_PATHS)
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "runs" / args.run_name
    summary_root = Path(args.summary_dir)
    summary_dir = summary_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    device = make_device(args.device)
    dataset = MSRSH5PatchDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = build_manifold_cycle_model(
        latent_channels=args.latent_channels,
        encoder_stem_channels=args.encoder_stem_channels,
        lambda_fusion=args.lambda_fusion,
        lambda_latent_grad=args.lambda_latent_grad,
        lambda_edge_smooth=args.lambda_edge_smooth,
        edge_smooth_alpha=args.edge_smooth_alpha,
        fusion_gradient_weight=args.fusion_gradient_weight,
        encoder_block_conv=args.encoder_block_conv,
        decoder_in_proj_conv=args.decoder_in_proj_conv,
        decoder_block_conv=args.decoder_block_conv,
        mixing_mode=args.mixing_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (KeyError, ValueError):
            print("Optimizer state is incompatible with the light-only model; continuing with a new optimizer.")
        if "scaler_state_dict" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except ValueError:
                print("AMP scaler state is incompatible; continuing with a new scaler.")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    config = {
        **vars(args),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "summary_dir": str(summary_dir),
        "dataset_size": len(dataset),
        "device": str(device),
        "model": "LightLatentMixingFusionNet",
        "architecture": "encoder + latent mixing + z_fused-only decoder",
        "mixing_mode": args.mixing_mode,
        "conv_variant": {
            "encoder_block_conv": args.encoder_block_conv,
            "decoder_in_proj_conv": args.decoder_in_proj_conv,
            "decoder_block_conv": args.decoder_block_conv,
        },
    }
    write_json(output_dir / "config.json", config)
    write_json(summary_dir / "config.json", config)

    log_path = output_dir / "train_log.csv"
    log_mode = "a" if args.resume else "w"
    log_exists = log_path.exists() and args.resume
    with log_path.open(log_mode, newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=("epoch", "seconds", "lr", *LOSS_KEYS))
        if not log_exists:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_start = time.time()
            sums, seen = average_meter_init()

            for step, batch in enumerate(loader, start=1):
                vis, ir = batch[:2]
                vis = vis.to(device, non_blocking=True)
                ir = ir.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    outputs = model.cycle_loss(vis, ir, return_images=False)
                    loss = outputs["loss"]

                scaler.scale(loss).backward()
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                values = to_float_dict(outputs)
                seen = average_meter_update(sums, seen, values, vis.shape[0])

                if args.log_interval > 0 and step % args.log_interval == 0:
                    avg = average_meter_compute(sums, seen)
                    print(
                        f"epoch {epoch:03d}/{args.epochs:03d} "
                        f"step {step:04d}/{len(loader):04d} "
                        f"loss={avg['loss']:.6f} fusion={avg['loss_fusion']:.6f} "
                        f"latent_grad={avg['loss_latent_grad']:.6f} "
                        f"edge_smooth={avg['loss_edge_smooth']:.6f}",
                        flush=True,
                    )

                if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                    break

            metrics = average_meter_compute(sums, seen)
            elapsed = time.time() - epoch_start
            row = {"epoch": epoch, "seconds": round(elapsed, 3), "lr": optimizer.param_groups[0]["lr"], **metrics}
            writer.writerow(row)
            log_file.flush()

            is_best = metrics["loss"] < best_loss
            if is_best:
                best_loss = metrics["loss"]
                save_checkpoint(output_dir / "best.pth", model, optimizer, scaler, epoch, best_loss, args, len(dataset))

            save_checkpoint(output_dir / "latest.pth", model, optimizer, scaler, epoch, best_loss, args, len(dataset))
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scaler, epoch, best_loss, args, len(dataset))

            status = {
                "run_name": args.run_name,
                "state": "running",
                "epoch": epoch,
                "epochs": args.epochs,
                "best_loss": best_loss,
                "last_metrics": metrics,
                "data_path": str(data_path),
                "output_dir": str(output_dir),
                "summary_dir": str(summary_dir),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_json(output_dir / "status.json", status)
            copy_to_summary(output_dir, summary_dir)
            print(
                f"epoch {epoch:03d} done in {elapsed:.1f}s "
                f"loss={metrics['loss']:.6f} best={best_loss:.6f}",
                flush=True,
            )

    save_checkpoint(output_dir / "final.pth", model, optimizer, scaler, args.epochs, best_loss, args, len(dataset))
    final_status = {
        "run_name": args.run_name,
        "state": "completed",
        "epoch": args.epochs,
        "epochs": args.epochs,
        "best_loss": best_loss,
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "summary_dir": str(summary_dir),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "status.json", final_status)
    copy_to_summary(output_dir, summary_dir)
    print(f"Training completed. Outputs: {output_dir}")
    print(f"Summary copy: {summary_dir}")


if __name__ == "__main__":
    main()
