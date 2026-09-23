/*
 * Fused CUDA Adam step, added while integrating ideas from
 * Faster-GS (Hahlbohm et al., CVPR 2026) into fastgs-lite. Always used by
 * GaussianModel.training_setup (no toggle).
 *
 * This is NOT a fusion of backward+optimizer into one kernel (that would
 * require rewriting the differentiable rasterizer's gradient bookkeeping,
 * which was judged too risky to write correctly without a build/test
 * environment). It is a standalone elementwise Adam kernel: one CUDA
 * launch that reads (param, grad, exp_avg, exp_avg_sq) and writes the
 * updated (param, exp_avg, exp_avg_sq) in place, following the exact
 * same math as torch.optim.Adam (with bias correction). The benefit vs.
 * torch.optim.Adam is fewer separate CUDA kernel launches / tensor temp
 * allocations for the same update, not a different algorithm.
 *
 * It only touches this file + adam_fused.cu + one pybind entry point in
 * ext.cpp + one line in setup.py's sources list. It does not modify any
 * existing rasterizer kernel (forward.cu / backward.cu / rasterizer_impl.cu
 * are untouched).
 */

#pragma once
#include <torch/extension.h>

// In-place fused Adam step for a single flat parameter tensor.
// param, grad, exp_avg, exp_avg_sq must all be contiguous CUDA float32
// tensors of identical shape. Updates param, exp_avg, exp_avg_sq in place.
void FusedAdamStepCUDA(
    torch::Tensor& param,
    const torch::Tensor& grad,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const int step);
