"""Ghép mọi bước lại: kiểm tra máy -> dữ liệu -> train -> submission -> báo cáo -> tải về."""

import dataclasses
import gc
import json
import os

from pipeline import data as data_mod
from pipeline import deliver, report, submission
from pipeline.env import check_gpu, free_memory, install_dependencies, show_mem
from pipeline.trainer import train_scene


def setup(cfg, install=True, require_gpu=True):
    """Cài phụ thuộc (lần đầu ~3-5 phút), gắn Drive, rồi in cấu hình máy.

    Gắn Drive ở đây chứ không đợi tới lúc cần: hộp thoại cấp quyền của Colab
    chặn ô đang chạy, nên Run All chỉ phải dừng đúng một lần ngay từ đầu.
    """
    if install:
        install_dependencies()
    if getattr(cfg, "drive_mount", False) or getattr(cfg, "autosave_to_drive", False):
        data_mod.mount_drive(cfg)
    info = check_gpu(require=require_gpu)
    os.makedirs(cfg.output_root, exist_ok=True)
    os.makedirs(cfg.submission_dir, exist_ok=True)
    return info


def load_data(cfg, profile=True, force=False):
    """Tải dữ liệu nếu cần, liệt kê scene, kiểm tra đủ ảnh, in hồ sơ dữ liệu."""
    os.makedirs(cfg.output_root, exist_ok=True)
    os.makedirs(cfg.submission_dir, exist_ok=True)
    data_mod.download_dataset(cfg, force=force)
    data_mod.resolve_subdir(cfg)
    scenes = data_mod.find_scenes(cfg)
    if not scenes:
        raise RuntimeError(f"không thấy scene nào trong {cfg.resolved_scene_root()}")
    print("scene:", scenes)
    for scene in scenes:
        data_mod.verify_scene(cfg, scene)
    frame = data_mod.profile_scenes(cfg, scenes) if profile else None
    return scenes, frame


def smoke_test(cfg, scene, iterations=None):
    """Chạy thử vài trăm vòng trên một scene để chắc chắn toàn bộ đường ống chạy được."""
    if not getattr(cfg, "run_smoke", True):
        print("cfg.run_smoke = False -> bỏ qua bước chạy thử")
        return None
    trial = dataclasses.replace(cfg,
                                output_root=os.path.join(cfg.output_root, "_smoke"),
                                score_every=max(50, (iterations or cfg.smoke_iterations) // 2),
                                eval_views=min(cfg.eval_views, 3))
    trial._scene_paths = getattr(cfg, "_scene_paths", {})
    result, _, _ = train_scene(trial, scene, iterations or cfg.smoke_iterations, tag="smoke")
    gc.collect()
    print(f"đường ống OK: Score thử nghiệm {result.get('score', float('nan')):.4f}"
          f" sau {result['iterations']} vòng")
    return result


def run_all(cfg, scenes, iterations=None, score_submission=True):
    """Train từng scene rồi render ngay test pose của scene đó, giải phóng bộ nhớ giữa các scene."""
    results, submissions = [], []
    for index, scene in enumerate(scenes, start=1):
        print(f"\n===== [{index}/{len(scenes)}] {scene} =====")
        show_mem(f"trước {scene}")
        result, _, _ = train_scene(cfg, scene, iterations)
        results.append(result)
        deliver.autosave_scene(cfg, scene)             # .ply lên Drive trước khi làm gì khác
        free_memory(tag=f"sau train {scene}")
        submissions.append(submission.render_scene(cfg, scene, result["iterations"],
                                                   score=score_submission))
        free_memory(tag=f"sau render {scene}")
        # ghi kết quả sau MỖI scene: mất phiên giữa chừng vẫn còn phần đã chạy
        _dump_results(cfg, results, submissions)

    return results, submissions


def _dump_results(cfg, results, submissions):
    with open(os.path.join(cfg.output_root, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(dict(config=cfg.as_dict(), results=results, submissions=submissions),
                  handle, indent=1, default=str)


def analytics(cfg, results, submissions):
    """Bảng + biểu đồ so sánh, lưu kèm vào `output_root`."""
    history = report.history_frame(results, save_to=os.path.join(cfg.output_root, "history.csv"))
    board = report.leaderboard(results, submissions,
                               save_to=os.path.join(cfg.output_root, "leaderboard.csv"))
    report.plot_training(history, save_to=os.path.join(cfg.output_root, "training.png"))
    report.plot_leaderboard(board, save_to=os.path.join(cfg.output_root, "leaderboard.png"))
    return history, board


def finish(cfg, scenes):
    """Đóng gói submission.zip (+ mô hình) rồi tải về máy."""
    zip_path = submission.build_zip(cfg, scenes)
    frame, problems = submission.verify(cfg, expected=scenes)
    targets = [zip_path] if cfg.download_submission else []
    if cfg.download_model:
        targets.append(deliver.pack_models(cfg, scenes, extra_files=[
            os.path.join(cfg.output_root, "leaderboard.csv"),
            os.path.join(cfg.output_root, "history.csv"),
            os.path.join(cfg.output_root, "training.png"),
            os.path.join(cfg.output_root, "leaderboard.png"),
        ]))
    deliver.download(cfg, targets)
    return frame, problems
