"""CPU 冒烟测试：验证 domain_dual_rank 变体的前向/反向/参数/冻结逻辑。"""
import torch

from manifold_cycle_net import build_manifold_cycle_model


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def run(mode):
    torch.manual_seed(0)
    model = build_manifold_cycle_model(
        latent_channels=32,
        latent_adapter_mode=mode,
        domain_rank=4,
        latent_adapter_beta_max=0.1,
    )
    vis = torch.rand(2, 1, 64, 64)
    ir = torch.rand(2, 1, 64, 64)
    outputs = model.cycle_loss(vis, ir)
    loss = outputs["loss"]
    loss.backward()

    grads = {}
    if mode == "domain_dual_rank":
        for name, parameter in model.latent_adapter.named_parameters():
            grads[name] = None if parameter.grad is None else float(parameter.grad.abs().mean())
    print(f"[{mode}] loss={loss.item():.4f} beta={float(outputs['adapter_beta']):.4f} "
          f"delta_l1={float(outputs['adapter_delta_l1']):.5f} "
          f"params(trainable)={count_params(model)/1e6:.3f}M")
    if grads:
        print("   adapter grads:", {k: (None if v is None else round(v, 6)) for k, v in list(grads.items())[:4]})
    return model


if __name__ == "__main__":
    for mode in ("none", "consensus_residual", "domain_dual_rank"):
        run(mode)

    # 阶段二冻结逻辑：编码器 requires_grad=False，适配器/解码器仍可训
    model = build_manifold_cycle_model(latent_channels=32, latent_adapter_mode="domain_dual_rank", domain_rank=4)
    for p in model.vis_encoder.parameters():
        p.requires_grad = False
    for p in model.ir_encoder.parameters():
        p.requires_grad = False
    vis = torch.rand(2, 1, 64, 64)
    ir = torch.rand(2, 1, 64, 64)
    outputs = model.cycle_loss(vis, ir)
    outputs["loss"].backward()
    enc_grad = any(p.grad is not None for p in model.vis_encoder.parameters())
    dec_grad = any(p.grad is not None for p in model.fusion_decoder.parameters())
    adp_grad = any(p.grad is not None for p in model.latent_adapter.parameters())
    print(f"[freeze test] encoder_grad={enc_grad} decoder_grad={dec_grad} adapter_grad={adp_grad}")

    # 推理接口
    with torch.no_grad():
        out = model.generate(vis, ir)
    print("generate fused shape:", tuple(out["fused"].shape))
    print("ALL OK")
