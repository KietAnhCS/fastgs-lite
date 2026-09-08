"""Đọc `test/test_poses.csv` của ban tổ chức thành các `Camera` để render.

Vì sao cần module này: `Scene` chỉ biết tách test set bằng `llffhold` từ chính tập
train (cứ 8 ảnh lấy 1). Bộ test thật của cuộc thi nằm ở thư mục `test/` riêng, với
pose cho sẵn trong CSV chứ không có trong `sparse/`. Không đọc file này thì
submission render **sai bộ camera**.

Cột CSV: image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
Quy ước giống COLMAP: quaternion + translation là world-to-camera.
"""

import csv
import os

TEST_DIR = "test"
POSES_CSV = "test_poses.csv"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG")
REQUIRED = ("image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
            "fx", "fy", "cx", "cy", "width", "height")


def test_root(cfg, scene):
    """`<...>/HCM0539/test` — suy ra từ đường dẫn scene (vốn trỏ vào `.../train`)."""
    from pipeline import data as data_mod

    path = data_mod.scene_path(cfg, scene)
    parent = os.path.dirname(path) if os.path.basename(path) == "train" else path
    return os.path.join(parent, TEST_DIR)


def poses_csv(cfg, scene):
    """Đường dẫn tới test_poses.csv, hoặc None nếu scene không theo layout cuộc thi."""
    candidate = os.path.join(test_root(cfg, scene), POSES_CSV)
    return candidate if os.path.exists(candidate) else None


def _find_image(folder, name):
    """Tìm ảnh GT theo `image_name`, chấp nhận có hoặc không có phần mở rộng."""
    direct = os.path.join(folder, name)
    if os.path.exists(direct):
        return direct
    stem = os.path.splitext(name)[0]
    for ext in IMAGE_EXT:
        candidate = os.path.join(folder, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def read_rows(path):
    """Đọc CSV, giữ nguyên thứ tự dòng. Raise nếu thiếu cột."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{path} rỗng")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise RuntimeError(f"{path} thiếu cột: {missing}")
    return rows


def load_test_cameras(cfg, scene, data_device="cuda"):
    """Dựng list[Camera] từ test_poses.csv.

    Trả về (cams, info). `info["has_gt"]` cho biết có chấm điểm được không.
    """
    import numpy as np
    import torch
    from PIL import Image

    from scene.cameras import Camera
    from scene.colmap_loader import qvec2rotmat
    from utils.general_utils import PILtoTorch
    from utils.graphics_utils import focal2fov

    path = poses_csv(cfg, scene)
    if path is None:
        return None, None

    rows = read_rows(path)
    images_dir = os.path.join(test_root(cfg, scene), cfg.images_dir)
    order = getattr(cfg, "submission_order", "csv")
    if order == "name":
        rows = sorted(rows, key=lambda r: r["image_name"])

    cams, missing_gt, offsets = [], [], []
    for index, row in enumerate(rows):
        width, height = int(float(row["width"])), int(float(row["height"]))
        qvec = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
        R = np.transpose(qvec2rotmat(qvec))            # giống readColmapCameras
        T = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
        fov_x = focal2fov(float(row["fx"]), width)
        fov_y = focal2fov(float(row["fy"]), height)

        # 3DGS chỉ dựng ma trận chiếu từ FoV, không có chỗ cho principal point lệch
        offsets.append((abs(float(row["cx"]) - width / 2.0),
                        abs(float(row["cy"]) - height / 2.0)))

        name = row["image_name"]
        gt_path = _find_image(images_dir, name)
        if gt_path is None:
            missing_gt.append(name)
            image = torch.zeros((3, height, width))
        else:
            with Image.open(gt_path) as handle:
                image = PILtoTorch(handle.convert("RGB"), (width, height))

        cams.append(Camera(colmap_id=index, R=R, T=T, FoVx=fov_x, FoVy=fov_y,
                           image=image, gt_alpha_mask=None,
                           image_name=os.path.splitext(name)[0], uid=index,
                           data_device=data_device))

    max_dx = max(o[0] for o in offsets)
    max_dy = max(o[1] for o in offsets)
    if max_dx > 1.0 or max_dy > 1.0:
        print(f"  ⚠ principal point lệch tâm tối đa ({max_dx:.1f}, {max_dy:.1f}) px — "
              "3DGS dựng ma trận chiếu chỉ từ FoV nên ảnh render sẽ bị dịch đúng lượng đó")
    if missing_gt:
        print(f"  {len(missing_gt)}/{len(rows)} ảnh GT không tìm thấy -> render nhưng không chấm điểm")

    info = dict(source=path, count=len(cams), order=order,
                has_gt=not missing_gt, missing_gt=missing_gt,
                max_cx_offset=max_dx, max_cy_offset=max_dy,
                width=int(float(rows[0]["width"])), height=int(float(rows[0]["height"])))
    print(f"  test_poses.csv: {len(cams)} camera | thứ tự '{order}' | "
          f"{info['width']}x{info['height']}")
    return cams, info
