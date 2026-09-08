"""Tải dữ liệu, tìm scene và lập hồ sơ dữ liệu."""

import os
import re
import subprocess
import sys
import zipfile

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")
DRIVE_HOSTS = ("drive.google.com", "docs.google.com")


def _is_scene(path):
    return (os.path.isdir(os.path.join(path, "sparse"))
            or os.path.exists(os.path.join(path, "transforms_train.json")))


def _is_competition_scene(path):
    """Layout của ban tổ chức: `<scene>/train/{images,sparse}` + `<scene>/test/test_poses.csv`.

    Kiểm tra cả hai nhánh để không nhầm với scene tên "train" của Tanks&Temples.
    """
    return (_is_scene(os.path.join(path, "train"))
            and os.path.exists(os.path.join(path, "test", "test_poses.csv")))


# --- Google Drive ------------------------------------------------------


def _is_drive_url(url):
    return bool(url) and any(host in url for host in DRIVE_HOSTS)


def _ensure_gdown():
    try:
        import gdown                                  # noqa: F401
        return True
    except ImportError:
        print("cài gdown ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "gdown"],
                       check=False)
    try:
        import gdown                                  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive(cfg=None):
    """Gắn Google Drive trong Colab. Không tải gì cả — đọc thẳng từ Drive.

    Gọi được trước khi có `cfg` (ô đầu của notebook) để hộp thoại cấp quyền —
    thứ duy nhất chặn Run All — xảy ra ngay từ đầu thay vì giữa chừng.
    """
    mount_point = getattr(cfg, "drive_mount_point", None) or "/content/drive"
    if os.path.isdir(os.path.join(mount_point, "MyDrive")):
        print("Drive đã gắn sẵn:", mount_point)
        return True
    try:
        from google.colab import drive as colab_drive
    except ImportError:
        print("không chạy trong Colab -> bỏ qua mount Drive")
        return False
    colab_drive.mount(mount_point)
    return True


def download_drive_folder(cfg, force=False):
    """Tải một thư mục Google Drive vào `cfg.data_root` bằng gdown."""
    os.makedirs(cfg.data_root, exist_ok=True)
    if not force and find_scenes(cfg):
        print("dữ liệu đã có sẵn trong", cfg.data_root)
        return cfg.data_root
    if not _ensure_gdown():
        raise RuntimeError("không cài được gdown — hãy chạy `pip install gdown` rồi thử lại")

    import gdown

    print("tải thư mục Drive:", cfg.drive_folder_url)
    kwargs = dict(url=cfg.drive_folder_url, output=cfg.data_root,
                  quiet=False, use_cookies=False)
    try:
        gdown.download_folder(remaining_ok=True, **kwargs)
    except TypeError:                                 # gdown cũ không có remaining_ok
        gdown.download_folder(**kwargs)
    return cfg.data_root


def resolve_subdir(cfg):
    """Trỏ `cfg.scene_root` vào thư mục con `cfg.drive_subdir` (vd "HCM0539") nếu tìm thấy."""
    name = getattr(cfg, "drive_subdir", None)
    if not name:
        return cfg.resolved_scene_root()
    root = cfg.resolved_scene_root()
    for current, dirs, _ in os.walk(root):
        if name in dirs:
            cfg.scene_root = os.path.join(current, name)
            print("scene_root ->", cfg.scene_root)
            return cfg.scene_root
    print(f"cảnh báo: không thấy thư mục con {name!r} trong {root}")
    return root


def verify_scene(cfg, scene):
    """Đối chiếu số ảnh thực tế với README.txt của ban tổ chức.

    gdown giới hạn 50 file mỗi thư mục ở một số phiên bản, nên một lần tải thiếu
    có thể im lặng đi qua. README ghi rõ "Train images: 240" nên kiểm được.
    """
    path = scene_path(cfg, scene)
    root = os.path.dirname(path) if os.path.basename(path) == "train" else path
    readme = os.path.join(root, "README.txt")
    if not os.path.exists(readme):
        return None

    with open(readme, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    expected = {key: int(value) for key, value in
                re.findall(r"(Train|Test) images:\s*(\d+)", text)}
    if not expected:
        return None

    actual = {}
    for split, key in (("train", "Train"), ("test", "Test")):
        images = os.path.join(root, split, cfg.images_dir)
        actual[key] = (len([f for f in os.listdir(images) if f.endswith(IMAGE_EXT)])
                       if os.path.isdir(images) else 0)

    ok = True
    for key, want in expected.items():
        got = actual.get(key, 0)
        mark = "OK" if got == want else "THIẾU"
        if got != want:
            ok = False
        print(f"  [{mark}] {key} images: {got}/{want}")
    if not ok:
        print("  -> tải thiếu ảnh. Dùng cách gắn Drive (drive_mount=True) "
              "hoặc tải lại với `run.load_data(cfg, force=True)`.")
    return ok


def download_dataset(cfg, force=False):
    """Tải + giải nén archive dữ liệu vào `cfg.data_root` (bỏ qua nếu đã có scene)."""
    if getattr(cfg, "drive_mount", False):
        mount_drive(cfg)
        return cfg.resolved_scene_root()
    if getattr(cfg, "drive_folder_url", None):
        return download_drive_folder(cfg, force=force)

    os.makedirs(cfg.data_root, exist_ok=True)
    if not force and find_scenes(cfg):
        print("dữ liệu đã có sẵn trong", cfg.data_root)
        return cfg.data_root
    if _is_drive_url(cfg.dataset_url):
        raise RuntimeError(
            "cfg.dataset_url trỏ tới Google Drive — dùng cfg.drive_folder_url "
            "(link thư mục) hoặc cfg.drive_mount=True thay vì dataset_url")
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
    if _is_competition_scene(root):                   # root CHÍNH LÀ scene -> không duyệt sâu
        found[os.path.basename(root.rstrip("/\\"))] = os.path.join(root, "train")
        return _select(cfg, found)
    if _is_scene(root):
        found[os.path.basename(root.rstrip("/\\"))] = root
    for current, dirs, _ in os.walk(root):
        depth = current[len(root):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
            continue
        for name in list(dirs):
            path = os.path.join(current, name)
            if _is_competition_scene(path):           # <scene>/train + <scene>/test
                found.setdefault(name, os.path.join(path, "train"))
                dirs.remove(name)
            elif _is_scene(path):
                found.setdefault(name, path)
                dirs.remove(name)

    return _select(cfg, found)


def _select(cfg, found):
    """Lọc theo `cfg.scenes` / `cfg.max_scenes` rồi ghi lại đường dẫn vào cfg."""
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

        # scene cuộc thi có test set riêng; chỉ khi không có mới suy ra từ llffhold
        from pipeline import testposes

        csv_path = testposes.poses_csv(cfg, scene)
        if csv_path:
            n_test = len(testposes.read_rows(csv_path))
            n_train, split = len(files), "test_poses.csv"
        else:
            n_test = len(files) // cfg.llffhold if cfg.llffhold else 0
            n_train, split = len(files) - n_test, f"llffhold={cfg.llffhold}"

        rows.append(dict(
            scene=scene,
            images=len(files),
            width=width,
            height=height,
            train=n_train,
            test=n_test,
            split=split,
            train_px=f"{width // cfg.resolution}x{height // cfg.resolution}",
            size_mb=round(n_bytes / 1024 ** 2, 1),
            colmap=os.path.isdir(os.path.join(path, "sparse")),
            path=path,
        ))
    frame = pd.DataFrame(rows)
    print(f"{len(scenes)} scene | tổng {frame.get('images', pd.Series(dtype=int)).sum()} ảnh")
    return frame
