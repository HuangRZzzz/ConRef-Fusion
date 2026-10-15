# ConRef-Fusion

Official PyTorch implementation of *ConRef-Fusion*: consensus anchoring and
bounded refinement for infrared and visible image fusion.

## Structure

- `manifold_cycle_net.py` — network definition (encoder, equal-mixing consensus
  anchor, decoder-side gated refinement)
- `train_manifold_cycle_msrs.py` — two-stage training script
- `evaluate_manifold_cycle_msrs.py` — evaluation script (basic metrics)
- `evaluate_manifold_cycle_msrs_new_metrics.py` — evaluation script (extended metrics)
- `utils/` — dataset loading, losses, and evaluation metrics

## Requirements

- Python >= 3.9
- PyTorch (CUDA recommended), h5py, numpy, Pillow

## Data

Training consumes an H5 file of cropped patch pairs (IR / VIS), e.g.
128x128 patches. Pass your own H5 file via `--data-path`.

## Training

```bash
python train_manifold_cycle_msrs.py \
    --data-path /path/to/train_imgsize_128_stride_200.h5 \
    --run-name my_run
```

Defaults follow the paper: 120 epochs, batch size 16, AdamW with lr 1e-4,
seed 20260520.

## Evaluation

```bash
python evaluate_manifold_cycle_msrs_new_metrics.py \
    --checkpoint /path/to/checkpoint.pth \
    --dataset-root /path/to/dataset \
    --output-root ./results
```

## Citation

If you find this code useful, please cite our paper (details to be added upon
publication).
