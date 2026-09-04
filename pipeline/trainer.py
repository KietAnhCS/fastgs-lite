"""Vòng lặp huấn luyện FastGS + móc theo dõi Score ngay trong lúc train."""

import gc
import os
import time
from argparse import ArgumentParser, Namespace
from random import randint

from pipeline import data as data_mod
from pipeline.env import mem, show_mem
from pipeline.score import evaluate_cameras


def build_args(cfg, scene, model_path=None, iterations=None, resolution=None, extra=(),
               write_cfg=True):
    """Dựng Namespace tham số của 3DGS cho một scene rồi ghi `cfg_args` để render lại."""
    from arguments import ModelParams, PipelineParams, OptimizationParams

    model_path = model_path or cfg.model_path(scene)
    parser = ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    argv = ["-s", data_mod.scene_path(cfg, scene), "-m", model_path,
            "-i", cfg.images_dir,
            "-r", str(cfg.resolution if resolution is None else resolution), "--eval",
            "--iterations", str(iterations or cfg.iterations),
            "--position_lr_max_steps", str(iterations or cfg.iterations),
            "--mult", str(cfg.mult),
            "--llffhold", str(cfg.llffhold)]
    if cfg.white_background:
        argv.append("-w")
    argv += list(cfg.train_extra_args) + list(extra)
    args = parser.parse_args(argv)

    os.makedirs(model_path, exist_ok=True)
    if write_cfg:
        with open(os.path.join(model_path, "cfg_args"), "w") as handle:
            handle.write(str(Namespace(**vars(args))))
    return args, lp.extract(args), op.extract(args), pp.extract(args)


def _save_checkpoint(scene_obj, iteration, previous=None, drop_previous=True):
    """Lưu `.ply` rồi xoá checkpoint giữa chừng trước đó (đĩa Colab không rộng)."""
    import shutil

    scene_obj.save(iteration)
    if drop_previous and previous is not None and previous != iteration:
        stale = os.path.join(scene_obj.model_path, "point_cloud", f"iteration_{previous}")
        shutil.rmtree(stale, ignore_errors=True)
    return iteration


def train_scene(cfg, scene, iterations=None, tag=None, keep_model=False, quiet_eval=False):
    """Huấn luyện một scene.

    Trả về (result, gaussians, scene_obj); gaussians/scene_obj = None khi `keep_model=False`
    (mặc định — giải phóng RAM/VRAM ngay để scene kế tiếp có chỗ chạy).
    """
    import torch
    from tqdm.auto import tqdm

    from scene import Scene, GaussianModel
    from utils.loss_utils import l1_loss
    from utils.general_utils import safe_state
    from fused_ssim import fused_ssim as fast_ssim
    from gaussian_renderer import render_fastgs
    from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras

    iterations = int(iterations or cfg.iterations)
    tag = tag or scene
    safe_state(True)

    args, dataset, opt, pipe = build_args(cfg, scene, iterations=iterations)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene_obj = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_cams = scene_obj.getTestCameras()
    holdout = list(test_cams) if len(test_cams) else scene_obj.getTrainCameras()[::8]
    holdout_kind = "test" if len(test_cams) else "train_subset"

    history, prev = [], None
    stack, indices = [], []
    ema_loss = 0.0
    saved_at = None
    last = dict(score=float("nan"), psnr=float("nan"))
    start = time.time()
    torch.cuda.reset_peak_memory_stats()

    score_every = max(1, min(cfg.score_every, iterations))
    pbar = tqdm(range(1, iterations + 1), desc=f"train[{tag}]", dynamic_ncols=True)
    for iteration in pbar:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        train_cams = scene_obj.getTrainCameras()
        if not stack:
            stack = train_cams.copy()
            indices = list(range(len(stack)))
        pick = randint(0, len(indices) - 1)
        cam = stack.pop(pick)
        indices.pop(pick)

        bg = torch.rand(3, device="cuda") if opt.random_background else background
        pkg = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
        image, viewspace, visibility, radii = (pkg["render"], pkg["viewspace_points"],
                                               pkg["visibility_filter"], pkg["radii"])

        gt = cam.original_image.cuda()
        ll1 = l1_loss(image, gt)
        ssim_value = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

            # --- densify + prune đa góc nhìn của FastGS ---
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility] = torch.max(gaussians.max_radii2D[visibility],
                                                              radii[visibility])
                gaussians.add_densification_stats(viewspace, visibility)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    camlist = sampling_cameras(train_cams.copy())
                    importance, pruning = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=size_threshold, min_opacity=0.005,
                        extent=scene_obj.cameras_extent, radii=radii, args=opt,
                        importance_score=importance, pruning_score=pruning)
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 3000 == 0 and 15_000 < iteration < 30_000:
                camlist = sampling_cameras(scene_obj.getTrainCameras().copy())
                _, pruning = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning)

            if iteration < iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                else:
                    gaussians.optimizer.step(radii > 0, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)

            # --- % tiến trình + điểm số hiện ngay trên thanh tqdm ---
            if iteration % 10 == 0:
                pbar.set_postfix_str(
                    f"loss {ema_loss:.4f} | G {gaussians._xyz.shape[0]:,} | "
                    f"Score {last['score']:.4f} | PSNR {last['psnr']:.2f}", refresh=False)

            if iteration % score_every == 0 or iteration == iterations:
                torch.cuda.empty_cache()
                evaluation = evaluate_cameras(gaussians, holdout, pipe, background, opt.mult,
                                              cfg.eval_views, cfg.psnr_max, cfg.lpips_net_live)
                usage = mem()
                row = dict(scene=scene, iter=iteration, pct=100.0 * iteration / iterations,
                           n_gauss=int(gaussians._xyz.shape[0]), ema_loss=ema_loss,
                           elapsed_s=time.time() - start,
                           ram_gb=usage["ram_used_gb"], vram_gb=usage.get("vram_alloc_gb", 0.0),
                           **(evaluation or {}))
                if evaluation is not None:
                    if prev is not None:
                        for key in ("score", "psnr", "ssim", "lpips"):
                            row[f"d_{key}"] = row[key] - prev[key]
                    prev, last = row, row
                    if not quiet_eval:
                        delta = row.get("d_score")
                        delta_txt = f"{delta:+.4f}" if delta is not None else "  n/a "
                        tqdm.write(
                            f"[{tag}] {row['pct']:5.1f}% iter {iteration:6d}"
                            f" | Score {row['score']:.4f} ({delta_txt})"
                            f" | PSNR {row['psnr']:5.2f} (norm {row['psnr_norm']:.3f})"
                            f" SSIM {row['ssim']:.4f} LPIPS {row['lpips']:.4f}"
                            f" | G {row['n_gauss']:>9,} | loss {ema_loss:.4f}"
                            f" | RAM {row['ram_gb']:.1f} VRAM {row['vram_gb']:.1f} GB")
                history.append(row)
                if usage["ram_used_gb"] > cfg.ram_soft_limit_gb:
                    gc.collect()
                    torch.cuda.empty_cache()

            # --- checkpoint định kỳ: mất session Colab thì vẫn còn .ply gần nhất ---
            if cfg.save_every and iteration % cfg.save_every == 0 and iteration < iterations:
                saved_at = _save_checkpoint(scene_obj, iteration, saved_at,
                                            cfg.keep_last_checkpoint)
                tqdm.write(f"[{tag}] checkpoint: point_cloud/iteration_{iteration}")

            del pkg, image, gt, viewspace, visibility, radii, loss, ll1, ssim_value

    pbar.close()
    _save_checkpoint(scene_obj, iterations, saved_at, cfg.keep_last_checkpoint)
    total_time = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    final = history[-1] if history else {}
    result = dict(scene=scene, tag=tag, model_path=scene_obj.model_path, iterations=iterations,
                  holdout_kind=holdout_kind, total_time_s=total_time, peak_vram_gb=peak_vram,
                  n_gauss=int(gaussians._xyz.shape[0]), history=history)
    for key in ("score", "psnr", "ssim", "lpips", "psnr_norm"):
        if key in final:
            result[key] = final[key]
    print(f"[{tag}] xong: {result['n_gauss']:,} gaussians | {total_time:.1f}s"
          f" | Score {result.get('score', float('nan')):.4f} | peak VRAM {peak_vram:.2f} GB")

    if not keep_model:
        del gaussians, scene_obj
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        show_mem(f"sau train {tag}")
        return result, None, None
    return result, gaussians, scene_obj
