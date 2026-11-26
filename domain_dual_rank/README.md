# Domain-shared / Domain-specific Dual-Rank Adapter (experimental)

Experimental variant of ConRef-Fusion where the bounded residual adapter (CRA)
is replaced by a **dual-rank domain adapter**:

- **Domain-shared branch**: low-rank bypass (LoRA-style 1x1 down/up projections).
- **Domain-specific branch**: high-rank (full 3x3) convolution over
  `[z_base, z_vis - z_ir, |z_vis - z_ir|]`.
- **Uncertainty allocation**: a small head maps per-sample/per-channel
  spatial mean/std of `z_base` to a gate `u`; the correction is
  `(1-u) * delta_shared + u * delta_specific` (per channel).

Everything else (encoders, decoder, equal-weight anchor `0.5/0.5`,
losses, training loop) is unchanged from the released code.

## Quick sanity test (CPU)

```bash
python smoke_test.py
```

Expected: three modes (`none`, `consensus_residual`, `domain_dual_rank`) run
forward/backward; `domain_dual_rank` uses ~0.205M params at the default
settings; freeze test shows encoders frozen while decoder/adapter still train.

## Two-stage training

Stage I (consensus anchor only, adapter bypassed):

```bash
python train_manifold_cycle_msrs.py \
    --data-path /path/to/train_imgsize_128_stride_200.h5 \
    --latent-adapter-mode none \
    --run-name stage1
```

Stage II (frozen encoders + domain adapter):

```bash
python train_manifold_cycle_msrs.py \
    --data-path /path/to/train_imgsize_128_stride_200.h5 \
    --resume runs/stage1/best.pth \
    --latent-adapter-mode domain_dual_rank \
    --domain-rank 4 \
    --adapter-beta-max 0.1 \
    --freeze-encoders \
    --run-name stage2
```

New flags: `--latent-adapter-mode {none, consensus_residual, domain_dual_rank}`,
`--latent-adapter-hidden-channels`, `--adapter-beta-max`, `--domain-rank`,
`--freeze-encoders`.

## Notes

- The adapter output is scaled by `beta = beta_max * tanh(beta_raw)`; `beta_raw`
  starts at 0, so training begins from the pure anchor solution and the adapter
  gradually takes effect.
- Pseudo-domain clustering / adversarial domain alignment (GRL) from the
  original idea are **not** implemented yet — this version covers the
  lightweight shared/specific decomposition only.
- CPU-only machines with `torch_npu` installed may hit an `aclInit` error in
  `AdamW.step()`; pass `foreach=False` to the optimizer (environment quirk only;
  unaffected on GPU).
