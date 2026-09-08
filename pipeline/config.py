"""Mọi tham số của pipeline nằm ở một chỗ duy nhất."""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Sequence

DEMO_DATASET_URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"


@dataclass
class Config:
    # --- mã nguồn -------------------------------------------------------
    repo_url: str = "https://github.com/KietAnhCS/fastgs-lite.git"
    repo_dir: str = "/content/fastgs-lite"

    # --- dữ liệu --------------------------------------------------------
    data_root: str = "/content/data"
    dataset_url: Optional[str] = DEMO_DATASET_URL
    dataset_archive: str = "dataset.zip"

    # --- dữ liệu trên Google Drive --------------------------------------
    # Ưu tiên: drive_mount > drive_folder_url > dataset_url
    drive_mount: bool = False                 # True -> gắn Drive, đọc tại chỗ, không tải
    drive_mount_point: str = "/content/drive"
    drive_folder_url: Optional[str] = None    # link THƯ MỤC Drive -> tải bằng gdown
    drive_subdir: Optional[str] = None        # chỉ dùng thư mục con này, vd "HCM0539"

    scene_root: Optional[str] = None          # None -> tự tìm trong data_root
    scenes: Sequence[str] = ()                # () -> lấy hết scene tìm được
    max_scenes: Optional[int] = None
    images_dir: str = "images"
    resolution: int = 2                       # -r khi train
    llffhold: int = 8                         # cứ 8 ảnh lấy 1 ảnh hold-out
    white_background: bool = False

    # --- huấn luyện -----------------------------------------------------
    output_root: str = "/content/output"
    iterations: int = 7000
    smoke_iterations: int = 300
    score_every: int = 1000
    save_every: int = 2000                    # lưu .ply định kỳ; 0 = chỉ lưu ở vòng cuối
    keep_last_checkpoint: bool = True         # xoá checkpoint giữa chừng cũ, chỉ giữ cái mới nhất
    eval_views: int = 6
    mult: float = 0.5
    psnr_max: float = 30.0
    lpips_net_live: str = "alex"              # nhanh, dùng cho theo dõi
    lpips_net_report: str = "vgg"             # dùng cho số liệu báo cáo
    ram_soft_limit_gb: float = 10.5
    train_extra_args: Sequence[str] = field(default_factory=lambda: [
        "--densification_interval", "500",
        "--lambda_dssim", "0.25",
        "--highfeature_lr", "0.02",
        "--loss_thresh", "0.07",
        "--grad_abs_thresh", "0.0012",
    ])

    # --- submission -----------------------------------------------------
    submission_dir: str = "/content/submission"
    submission_zip: str = "/content/submission.zip"
    submission_ext: str = ".png"
    submission_digits: int = 4                # 0001.png
    submission_resolution: int = 1            # 1 = render đúng kích thước ảnh gốc
    # thứ tự đánh số 0001.png khi có test_poses.csv:
    #   "csv"  = giữ nguyên thứ tự dòng trong file ban tổ chức (mặc định)
    #   "name" = sắp theo image_name
    submission_order: str = "csv"

    # --- lấy kết quả về máy ---------------------------------------------
    download_submission: bool = True
    download_model: bool = True
    save_to_drive: bool = False
    drive_dir: str = "/content/drive/MyDrive"
    model_zip: str = "/content/fastgs_models.zip"
    # chép .ply sang Drive NGAY sau mỗi scene: /content bị xoá khi phiên Colab kết thúc
    autosave_to_drive: bool = True
    drive_run_dir: str = "fastgs_runs"
    run_smoke: bool = True                    # False -> Run All bỏ qua bước chạy thử

    # ------------------------------------------------------------------
    def model_path(self, scene: str) -> str:
        return os.path.join(self.output_root, scene)

    def scene_dir(self, scene: str) -> str:
        return os.path.join(self.resolved_scene_root(), scene)

    def resolved_scene_root(self) -> str:
        return self.scene_root or self.data_root

    def submission_scene_dir(self, scene: str) -> str:
        return os.path.join(self.submission_dir, scene)

    def as_dict(self) -> dict:
        return asdict(self)

    def show(self) -> "Config":
        keys = ("repo_dir", "data_root", "scene_root", "scenes", "resolution",
                "output_root", "iterations", "smoke_iterations", "score_every", "save_every",
                "eval_views", "psnr_max", "mult", "submission_dir", "submission_zip",
                "submission_resolution", "download_submission", "download_model")
        for key in keys:
            print(f"{key:22s} = {getattr(self, key)}")
        return self

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2, default=str)
        return path
