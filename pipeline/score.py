"""Ba metrics của cuộc thi và công thức điểm tổng hợp.

Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)
"""

import torch

W_LPIPS, W_SSIM, W_PSNR = 0.4, 0.3, 0.3


def composite_score(psnr_val, ssim_val, lpips_val, psnr_max=30.0):
    """Trả về (score, psnr_norm) đúng công thức ban tổ chức."""
    psnr_norm = torch.clamp(torch.as_tensor(float(psnr_val)) / float(psnr_max), 0.0, 1.0)
    ssim_t = torch.as_tensor(float(ssim_val))
    lpips_t = torch.as_tensor(float(lpips_val))
    score = W_LPIPS * (1.0 - lpips_t) + W_SSIM * ssim_t + W_PSNR * psnr_norm
    return float(score), float(psnr_norm)


@torch.no_grad()
def evaluate_cameras(gaussians, cams, pipe, background, mult,
                     max_views=None, psnr_max=30.0, lpips_net="alex"):
    """Render một tập camera có ground-truth rồi tính PSNR/SSIM/LPIPS/Score trung bình."""
    from gaussian_renderer import render_fastgs
    from utils.image_utils import psnr as psnr_fn
    from fused_ssim import fused_ssim as fast_ssim
    from lpipsPyTorch import lpips as lpips_fn

    cams = list(cams)
    if max_views:
        cams = cams[:max_views]
    if not cams:
        return None

    psnr_sum = ssim_sum = lpips_sum = 0.0
    for cam in cams:
        rendered = torch.clamp(render_fastgs(cam, gaussians, pipe, background, mult)["render"], 0.0, 1.0)
        gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        psnr_sum += psnr_fn(rendered, gt).mean().item()
        ssim_sum += fast_ssim(rendered.unsqueeze(0), gt.unsqueeze(0)).item()
        lpips_sum += lpips_fn(rendered, gt, net_type=lpips_net).mean().item()
        del rendered, gt

    n = len(cams)
    psnr_val, ssim_val, lpips_val = psnr_sum / n, ssim_sum / n, lpips_sum / n
    score, psnr_norm = composite_score(psnr_val, ssim_val, lpips_val, psnr_max)
    torch.cuda.empty_cache()
    return dict(psnr=psnr_val, ssim=ssim_val, lpips=lpips_val,
                psnr_norm=psnr_norm, score=score, n_views=n)
