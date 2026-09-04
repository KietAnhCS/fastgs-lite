"""Đóng gói mô hình + báo cáo và tải kết quả về máy."""

import glob
import os
import shutil
import zipfile


def pack_models(cfg, scenes, extra_files=()):
    """Gói `.ply` + cấu hình + báo cáo của mọi scene vào một ZIP để render lại sau này."""
    if os.path.exists(cfg.model_zip):
        os.remove(cfg.model_zip)

    with zipfile.ZipFile(cfg.model_zip, "w", zipfile.ZIP_STORED) as zf:
        for scene in scenes:
            model_path = cfg.model_path(scene)
            plys = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*",
                                          "point_cloud.ply"))
            plys.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[-1]))
            for path in plys[-1:]:                     # chỉ giữ mốc mới nhất
                relative = os.path.relpath(path, model_path).replace(os.sep, "/")
                zf.write(path, f"{scene}/{relative}")
            for relative in ("cfg_args", "cameras.json"):
                path = os.path.join(model_path, relative)
                if os.path.exists(path):
                    zf.write(path, f"{scene}/{relative}")
        for path in extra_files:
            if os.path.exists(path):
                zf.write(path, os.path.basename(path))

    size_mb = os.path.getsize(cfg.model_zip) / 1024 ** 2
    print(f"{cfg.model_zip} | {len(scenes)} scene | {size_mb:.1f} MB")
    return cfg.model_zip


def copy_to_drive(cfg, paths):
    """Chép file sang Google Drive (mount trước nếu chưa)."""
    if not os.path.isdir(cfg.drive_dir):
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as error:                     # noqa: BLE001 - ngoài Colab thì bỏ qua
            print("không mount được Drive:", error)
            return []
    os.makedirs(cfg.drive_dir, exist_ok=True)
    copied = []
    for path in paths:
        if os.path.exists(path):
            target = os.path.join(cfg.drive_dir, os.path.basename(path))
            shutil.copy2(path, target)
            copied.append(target)
            print("đã chép:", target)
    return copied


def download(cfg, paths):
    """Tự động tải file về máy (trình duyệt Colab); ngoài Colab chỉ in đường dẫn."""
    paths = [p for p in paths if p and os.path.exists(p)]
    if cfg.save_to_drive:
        copy_to_drive(cfg, paths)
    try:
        from google.colab import files
    except ImportError:
        for path in paths:
            print("file sẵn sàng:", path, f"({os.path.getsize(path) / 1024 ** 2:.1f} MB)")
        return paths
    for path in paths:
        print("đang tải về:", path, f"({os.path.getsize(path) / 1024 ** 2:.1f} MB)")
        files.download(path)
    return paths
