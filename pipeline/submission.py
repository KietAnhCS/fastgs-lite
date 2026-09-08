"""Render toàn bộ test pose và đóng gói `submission.zip` đúng định dạng ban tổ chức.

submission.zip
├── <scene>/0001.png, 0002.png, ...
└── ...
"""

import gc
import json
import os
import zipfile

from pipeline import testposes
from pipeline.score import composite_score
from pipeline.trainer import build_args


def _image_name(index, cfg):
    return f"{index:0{cfg.submission_digits}d}{cfg.submission_ext}"


def render_scene(cfg, scene, iterations=None, score=True):
    """Render mọi test camera của một scene ra `submission/<scene>/0001.png`.

    Trả về dict thông tin + metrics (nếu ảnh GT có sẵn và `score=True`).
    """
    import torch
    import torchvision
    from tqdm.auto import tqdm

    from scene import Scene, GaussianModel
    from gaussian_renderer import render_fastgs
    from utils.image_utils import psnr as psnr_fn
    from utils.loss_utils import ssim as ssim_fn
    from lpipsPyTorch import lpips as lpips_fn

    iterations = int(iterations or cfg.iterations)
    _, dataset, opt, pipe = build_args(cfg, scene, iterations=iterations,
                                       resolution=cfg.submission_resolution, write_cfg=False)

    gaussians = GaussianModel(dataset.sh_degree)
    scene_obj = None
    cams, pose_info = testposes.load_test_cameras(cfg, scene)

    if cams is not None:
        # Có test_poses.csv -> đây mới là bộ camera ban tổ chức yêu cầu.
        # load_ply tự đặt active_sh_degree nên không cần dựng Scene, tiết kiệm cả
        # việc đọc COLMAP lẫn việc nạp ảnh train vào VRAM.
        ply = os.path.join(dataset.model_path, "point_cloud",
                           f"iteration_{iterations}", "point_cloud.ply")
        if not os.path.exists(ply):
            raise FileNotFoundError(f"chưa có model để render: {ply}")
        gaussians.load_ply(ply)
    else:
        # Không phải layout cuộc thi -> giữ hành vi cũ (llffhold tách từ tập train).
        # Chỉ nạp test camera: ảnh train chiếm VRAM mà lượt render này không dùng tới.
        print(f"  [{scene}] không có test_poses.csv -> dùng holdout llffhold={cfg.llffhold}")
        scene_obj = Scene(dataset, gaussians, load_iteration=iterations, shuffle=False,
                          camera_sets=("test",))
        cams = sorted(scene_obj.getTestCameras(), key=lambda c: c.image_name)

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    out_dir = cfg.submission_scene_dir(scene)
    os.makedirs(out_dir, exist_ok=True)
    if pose_info is not None and not pose_info["has_gt"]:
        score = False                                  # ảnh GT là tensor 0, chấm sẽ vô nghĩa

    psnrs, ssims, lpipss, files = [], [], [], []
    with torch.no_grad():
        for index, cam in enumerate(tqdm(cams, desc=f"render[{scene}]", dynamic_ncols=True), start=1):
            rendered = torch.clamp(render_fastgs(cam, gaussians, pipe, background, cfg.mult)["render"],
                                   0.0, 1.0)
            name = _image_name(index, cfg)
            torchvision.utils.save_image(rendered, os.path.join(out_dir, name))
            files.append(dict(file=name, source=cam.image_name,
                              width=cam.image_width, height=cam.image_height))
            if score:
                gt = torch.clamp(cam.original_image.to("cuda")[:3], 0.0, 1.0)
                psnrs.append(psnr_fn(rendered, gt).mean().item())
                ssims.append(ssim_fn(rendered.unsqueeze(0), gt.unsqueeze(0)).item())
                lpipss.append(lpips_fn(rendered, gt, net_type=cfg.lpips_net_report).mean().item())
                del gt
            del rendered
            cam.original_image = None                  # trả VRAM ngay, ảnh này xong việc

    info = dict(scene=scene, images=len(files), out_dir=out_dir,
                width=files[0]["width"] if files else None,
                height=files[0]["height"] if files else None,
                camera_source="test_poses.csv" if pose_info else f"llffhold={cfg.llffhold}",
                poses=pose_info, files=files)
    if psnrs:
        psnr_val = sum(psnrs) / len(psnrs)
        ssim_val = sum(ssims) / len(ssims)
        lpips_val = sum(lpipss) / len(lpipss)
        score_val, psnr_norm = composite_score(psnr_val, ssim_val, lpips_val, cfg.psnr_max)
        info.update(psnr=psnr_val, ssim=ssim_val, lpips=lpips_val,
                    psnr_norm=psnr_norm, score=score_val)
        print(f"[{scene}] {len(files)} ảnh | PSNR {psnr_val:.2f} SSIM {ssim_val:.4f}"
              f" LPIPS {lpips_val:.4f} -> Score {score_val:.4f}")
    else:
        print(f"[{scene}] {len(files)} ảnh (không có ground-truth để chấm)")

    with open(os.path.join(out_dir, "_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=1)

    del gaussians, scene_obj
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return info


def render_all(cfg, scenes, iterations=None, score=True):
    return [render_scene(cfg, scene, iterations, score) for scene in scenes]


def build_zip(cfg, scenes=None):
    """Nén thư mục submission thành ZIP (ZIP_STORED: PNG đã nén sẵn, gần như không tốn RAM)."""
    scenes = scenes or sorted(d for d in os.listdir(cfg.submission_dir)
                              if os.path.isdir(os.path.join(cfg.submission_dir, d)))
    if os.path.exists(cfg.submission_zip):
        os.remove(cfg.submission_zip)

    total = 0
    with zipfile.ZipFile(cfg.submission_zip, "w", zipfile.ZIP_STORED) as zf:
        for scene in scenes:
            folder = cfg.submission_scene_dir(scene)
            for name in sorted(os.listdir(folder)):
                if not name.endswith(cfg.submission_ext):
                    continue
                zf.write(os.path.join(folder, name), f"{scene}/{name}")
                total += 1
    size_mb = os.path.getsize(cfg.submission_zip) / 1024 ** 2
    print(f"{cfg.submission_zip} | {len(scenes)} scene | {total} ảnh | {size_mb:.1f} MB")
    return cfg.submission_zip


def verify(cfg, expected=None):
    """Kiểm tra ZIP: tên scene, số ảnh mỗi scene, tên file liên tục 0001.png, kích thước ảnh."""
    import io

    from PIL import Image

    rows, problems = [], []
    with zipfile.ZipFile(cfg.submission_zip) as zf:
        names = [n for n in zf.namelist() if n.endswith(cfg.submission_ext)]
        by_scene = {}
        for name in names:
            scene, _, file = name.partition("/")
            if not file:
                problems.append(f"ảnh nằm ngoài thư mục scene: {name}")
                continue
            by_scene.setdefault(scene, []).append(file)

        for scene in sorted(by_scene):
            files = sorted(by_scene[scene])
            if not files:
                problems.append(f"{scene}: không có ảnh nào")
                continue
            wanted = [_image_name(i, cfg) for i in range(1, len(files) + 1)]
            if files != wanted:
                problems.append(f"{scene}: tên file không liên tục ({files[:3]} ...)")
            with zf.open(f"{scene}/{files[0]}") as handle:
                width, height = Image.open(io.BytesIO(handle.read())).size
            row = dict(scene=scene, images=len(files), width=width, height=height)

            # đối chiếu thẳng với test_poses.csv: số ảnh và kích thước render
            csv_path = testposes.poses_csv(cfg, scene)
            if csv_path:
                want = testposes.read_rows(csv_path)
                row["expected"] = len(want)
                if len(files) != len(want):
                    problems.append(f"{scene}: {len(files)} ảnh nhưng CSV có {len(want)} pose")
                want_w, want_h = int(float(want[0]["width"])), int(float(want[0]["height"]))
                if (width, height) != (want_w, want_h):
                    problems.append(f"{scene}: ảnh {width}x{height} nhưng CSV yêu cầu "
                                    f"{want_w}x{want_h}")
            rows.append(row)

    if expected:
        missing = [s for s in expected if s not in by_scene]
        extra = [s for s in by_scene if s not in expected]
        if missing:
            problems.append(f"thiếu scene: {missing}")
        if extra:
            problems.append(f"thừa scene: {extra}")

    import pandas as pd

    frame = pd.DataFrame(rows)
    print("OK: submission hợp lệ" if not problems else "CẢNH BÁO:\n- " + "\n- ".join(problems))
    return frame, problems
