/*
 * See adam_fused.h for scope/rationale.
 */

#include "adam_fused.h"
#include <cuda.h>
#include "cuda_runtime.h"

namespace {

__global__ void fused_adam_step_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const int64_t n,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1,
    const float bias_correction2_sqrt)
{
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float g = grad[i];
    const float m = beta1 * exp_avg[i] + (1.0f - beta1) * g;
    const float v = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
    exp_avg[i] = m;
    exp_avg_sq[i] = v;

    // Standard torch.optim.Adam update (matches eps-inside-sqrt-denominator
    // convention, i.e. eps=1e-15 used by GaussianModel's optimizers):
    // step_size = lr / bias_correction1
    // denom = sqrt(v) / bias_correction2_sqrt + eps
    // param -= step_size * m / denom
    const float step_size = lr / bias_correction1;
    const float denom = sqrtf(v) / bias_correction2_sqrt + eps;
    param[i] -= step_size * m / denom;
}

} // namespace

void FusedAdamStepCUDA(
    torch::Tensor& param,
    const torch::Tensor& grad,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const int step)
{
    TORCH_CHECK(param.is_cuda() && grad.is_cuda() && exp_avg.is_cuda() && exp_avg_sq.is_cuda(),
                "FusedAdamStepCUDA: all tensors must be CUDA tensors");
    TORCH_CHECK(param.scalar_type() == torch::kFloat32 && grad.scalar_type() == torch::kFloat32 &&
                exp_avg.scalar_type() == torch::kFloat32 && exp_avg_sq.scalar_type() == torch::kFloat32,
                "FusedAdamStepCUDA: all tensors must be float32");
    TORCH_CHECK(param.is_contiguous() && grad.is_contiguous() && exp_avg.is_contiguous() && exp_avg_sq.is_contiguous(),
                "FusedAdamStepCUDA: all tensors must be contiguous");
    TORCH_CHECK(param.numel() == grad.numel() && param.numel() == exp_avg.numel() && param.numel() == exp_avg_sq.numel(),
                "FusedAdamStepCUDA: all tensors must have the same number of elements");

    const int64_t n = param.numel();
    if (n == 0) return;

    const float bias_correction1 = 1.0f - powf(beta1, (float)step);
    const float bias_correction2_sqrt = sqrtf(1.0f - powf(beta2, (float)step));

    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    fused_adam_step_kernel<<<(unsigned int)blocks, threads>>>(
        param.data_ptr<float>(),
        grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        n, lr, beta1, beta2, eps, bias_correction1, bias_correction2_sqrt);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "FusedAdamStepCUDA: kernel launch failed: ", cudaGetErrorString(err));
}
