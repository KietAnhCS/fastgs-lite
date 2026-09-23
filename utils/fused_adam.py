"""Fused-CUDA Adam optimizer, always used by GaussianModel.training_setup
(Nhóm C2 của kế hoạch tích hợp Faster-GS).

Drop-in replacement for `torch.optim.Adam` that calls a single fused CUDA
kernel (`diff_gaussian_rasterization_fastgs._C.fused_adam_step`, see
`submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/adam_fused.cu`)
per parameter instead of doing the elementwise update through several
separate torch ops/kernel launches.

It subclasses `torch.optim.Optimizer` (not a bespoke class) precisely so
that `scene/gaussian_model.py`'s existing optimizer-surgery helpers
(`_prune_optimizer`, `cat_tensors_to_optimizer`, `replace_tensor_to_optimizer`,
state resets, `state_dict()/load_state_dict()`) keep working unmodified —
those only assume the standard `Optimizer.state[param] = {"exp_avg": ..,
"exp_avg_sq": ..}` / `param_groups` structure, which this class provides
identically to `torch.optim.Adam`.

Always used by `GaussianModel.training_setup` (no toggle) as of the
FastGS + Faster-GS integration.
"""

import torch


class FusedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-15):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        from diff_gaussian_rasterization_fastgs import _C

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("FusedAdam does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                state["step"] += 1

                # The fused kernel writes in place into param_flat, so it must alias
                # param.data's storage. `.view(-1)` would raise on a non-contiguous
                # tensor; `.reshape(-1)` would silently return a *copy* in that case
                # and the kernel's update would be lost. Force contiguity in place
                # first so the flat view is always a true alias.
                if not param.data.is_contiguous():
                    param.data = param.data.contiguous()
                if not state["exp_avg"].is_contiguous():
                    state["exp_avg"] = state["exp_avg"].contiguous()
                if not state["exp_avg_sq"].is_contiguous():
                    state["exp_avg_sq"] = state["exp_avg_sq"].contiguous()
                param_flat = param.data.view(-1)
                grad_flat = grad.contiguous().view(-1)
                exp_avg_flat = state["exp_avg"].view(-1)
                exp_avg_sq_flat = state["exp_avg_sq"].view(-1)

                _C.fused_adam_step(
                    param_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat,
                    float(lr), float(beta1), float(beta2), float(eps), int(state["step"]),
                )

        return loss
