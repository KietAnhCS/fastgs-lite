"""Tải dữ liệu, tìm scene và lập hồ sơ dữ liệu."""

import os
import subprocess
import zipfile

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")


def _is_scene(path):
    return (os.path.isdir(os.path.join(path, "sparse"))
            or os.path.exists(os.path.join(path, "transforms_train.json")))


def download_dataset(cfg, force=False):
    """Tải + giải nén archive dữ liệu vào `cfg.data_root` (bỏ qua nếu đã có scene)."""
    os.makedirs(cfg.data_root, exist_ok=True)
    if not force and find_scenes(cfg):
        print("dữ liệu đã có sẵn trong", cfg.data_root)
        return cfg.data_root
    if not cfg.dataset_url:
        print("cfg.dataset_url = None -> tự đặt dữ liệu vào", cfg.data_root)
        return cfg.data_root

    archive = os.path.join(cfg.data_root, cfg.dataset_archive)
    if not os.path.exists(archive):
        print("tải:", cfg.dataset_url)
        try:
            code = subprocess.run(["wget", "-q", "--show-progress", "-O", archive, cfg.dataset_url],
                                  check=False).returncode
        except FileNotFoundError:                     # máy không có wget
            import urllib.request

            urllib.request.urlretrieve(cfg.dataset_url, archive)
            code = 0
        if code != 0:
            raise RuntimeError(f"tải dữ liệu thất bại ({cfg.dataset_url})")
    print("giải nén:", archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(cfg.data_root)
    os.remove(archive)
    return cfg.data_root


def find_scenes(cfg, max_depth=3):
    """Tìm mọi thư mục scene (COLMAP `sparse/` hoặc Blender `transforms_train.json`)."""
    root = cfg.resolved_scene_root()
    found = {}
    if not os.path.isdir(root):
        return []
    if _is_scene(root):
        found[os.path.basename(root.rstrip("/\\"))] = root
    for current, dirs, _ in os.walk(root):
        depth = current[len(root):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
            continue
        for name in list(dirs):
            path = os.path.join(current, name)
            if _is_scene(path):
                found.setdefault(name, path)
                dirs.remove(name)

    names = sorted(found)
    if cfg.scenes:
        names = [n for n in cfg.scenes if n in found]
        missing = [n for n in cfg.scenes if n not in found]
        if missing:
            print("không tìm thấy scene:", missing)
    if cfg.max_scenes:
        names = names[:cfg.max_scenes]
    cfg._scene_paths = found
    return names


def scene_path(cfg, scene):
    return getattr(cfg, "_scene_paths", {}).get(scene, cfg.scene_dir(scene))


def _images_root(cfg, path):
    candidate = os.path.join(path, cfg.images_dir)
    return candidate if os.path.isdir(candidate) else path


def profile_scenes(cfg, scenes):
    """Bảng hồ sơ dữ liệu: số ảnh, kích thước, tách train/test, dung lượng."""
    import pandas as pd
    from PIL import Image

    rows = []
    for scene in scenes:
        path = scene_path(cfg, scene)
        image_root = _images_root(cfg, path)
        files = sorted(f for f in os.listdir(image_root) if f.endswith(IMAGE_EXT))
        if not files:
            rows.append(dict(scene=scene, images=0, note="không thấy ảnh"))
            continue
        with Image.open(os.path.join(image_root, files[0])) as im:
            width, height = im.size
        n_bytes = sum(os.path.getsize(os.path.join(image_root, f)) for f in files)
        n_test = len(files) // cfg.llffhold if cfg.llffhold else 0
        rows.append(dict(
            scene=scene,
            images=len(files),
            width=width,
            height=height,
            train=len(files) - n_test,
            test=n_test,
            train_px=f"{width // cfg.resolution}x{height // cfg.resolution}",
            size_mb=round(n_bytes / 1024 ** 2, 1),
            colmap=os.path.isdir(os.path.join(path, "sparse")),
            path=path,
        ))
    frame = pd.DataFrame(rows)
    print(f"{len(scenes)} scene | tổng {frame.get('images', pd.Series(dtype=int)).sum()} ảnh")
    return frame
