# Gói `pipeline/` và hợp đồng `submission.zip`

> Tài liệu tham chiếu cho gói `pipeline/*.py` — logic đứng sau notebook
> [`fastgs-acceleration-method.ipynb`](../fastgs-acceleration-method.ipynb) (nay chỉ còn 9 code cell, tiếng Anh, thuần glue-code).
> Cơ chế tăng tốc FastGS: [fastgs-acceleration-method.md](fastgs-acceleration-method.md).
> Vận hành thật trên Colab T4 (ràng buộc RAM/VRAM, preset, roadmap): [colab-t4-guide.md](colab-t4-guide.md).
> Mọi số liệu (tên hàm, mặc định, chữ ký) trong tài liệu này lấy trực tiếp từ mã nguồn trong `pipeline/`.

---

## 1. Notebook cell ↔ hàm pipeline

Notebook chỉ ghép các bước lại; toàn bộ logic nằm trong `pipeline/*.py`.

| Cell | Việc | Gọi tới |
|---|---|---|
| 0 | Clone repo (bỏ qua nếu đang chạy sẵn trong repo) | thao tác `git clone` trực tiếp trong cell, không qua `pipeline.env.clone_repo` |
| 1 | Cài phụ thuộc + kiểm tra GPU | `pipeline.env.install_dependencies()`, `pipeline.env.check_gpu(require=True)` |
| 2 | Khai báo cấu hình | `pipeline.Config(...)`, `cfg.show()` |
| 3 | Tải/liệt kê dữ liệu, in hồ sơ | `pipeline.run.load_data(cfg)` → `data.download_dataset`, `data.find_scenes`, `data.profile_scenes` |
| 4 | Chạy thử vài trăm vòng | `pipeline.run.smoke_test(cfg, scenes[0])` → `trainer.train_scene(..., tag="smoke")` |
| 5 | Train toàn bộ scene | `pipeline.run.run_all(cfg, scenes)` → `trainer.train_scene` + `submission.render_scene` mỗi scene |
| 6 | Bảng + biểu đồ so sánh | `pipeline.run.analytics(cfg, results, submissions)` → `report.history_frame`, `report.leaderboard`, `report.plot_training`, `report.plot_leaderboard`; cell 7 gọi thêm `report.show_samples` |
| 7 | Đóng gói + tải về | `pipeline.run.finish(cfg, scenes)` → `submission.build_zip`, `submission.verify`, `deliver.pack_models`, `deliver.download` |

Ghi chú: `pipeline/__init__.py` hiện chỉ re-export `Config` và các hàm của `pipeline.env` (`check_gpu`, `free_memory`, `install_dependencies`, `mem`, `show_mem`); notebook import `run` và `report` trực tiếp từ `pipeline` (`from pipeline import Config, run, report`) vì đó là submodule, không cần khai trong `__all__`.

### Module → nội dung

| Module | Trách nhiệm | Hàm chính |
|---|---|---|
| `pipeline/config.py` | Một `dataclass Config` gom mọi tham số | `model_path`, `scene_dir`, `resolved_scene_root`, `submission_scene_dir`, `as_dict`, `show`, `save` |
| `pipeline/env.py` | Môi trường máy | `check_gpu`, `install_dependencies`, `mem`, `show_mem`, `free_memory`, `clone_repo` |
| `pipeline/data.py` | Dữ liệu vào | `download_dataset`, `find_scenes`, `scene_path`, `profile_scenes` |
| `pipeline/score.py` | Công thức điểm chính thức | `composite_score`, `evaluate_cameras` |
| `pipeline/trainer.py` | Vòng lặp train FastGS + theo dõi | `build_args`, `train_scene` |
| `pipeline/submission.py` | Render test pose + đóng gói/kiểm tra zip | `render_scene`, `render_all`, `build_zip`, `verify` |
| `pipeline/report.py` | Bảng + biểu đồ | `history_frame`, `leaderboard`, `plot_training`, `plot_leaderboard`, `show_samples` |
| `pipeline/deliver.py` | Đóng gói model + tải về máy | `pack_models`, `copy_to_drive`, `download` |
| `pipeline/run.py` | Điều phối toàn bộ | `setup`, `load_data`, `smoke_test`, `run_all`, `analytics`, `finish` |

---

## 2. `Config` — các trường quan trọng

Đọc trực tiếp từ `pipeline/config.py`. Mặc định là giá trị dùng khi chạy demo trên bộ Tanks&Temples/DB (`DEMO_DATASET_URL`).

| Nhóm | Trường | Mặc định | Ý nghĩa |
|---|---|---|---|
| mã nguồn | `repo_url` | `https://github.com/KietAnhCS/fastgs-lite.git` | repo clone khi chạy trên Colab sạch |
| | `repo_dir` | `/content/fastgs-lite` | thư mục làm việc |
| dữ liệu | `data_root` | `/content/data` | nơi giải nén / đặt sẵn dữ liệu |
| | `dataset_url` | link demo `tandt_db.zip` | `None` = dữ liệu đã có sẵn trong `data_root`, không tải |
| | `dataset_archive` | `"dataset.zip"` | tên file archive tạm khi tải về |
| | `scene_root` | `None` | `None` → dùng `data_root` (qua `resolved_scene_root()`) |
| | `scenes` | `()` (rỗng) | rỗng = tự tìm hết scene có trong `scene_root` |
| | `max_scenes` | `None` | giới hạn số scene lấy ra, hữu ích khi thử nhanh |
| | `images_dir` | `"images"` | thư mục ảnh con trong mỗi scene |
| | `resolution` | `2` | `-r` khi train (downscale 2x cho vừa T4) |
| | `llffhold` | `8` | cứ 8 ảnh giữ 1 ảnh làm hold-out/test; `build_args` truyền xuống `Scene` qua cờ `--llffhold` |
| | `white_background` | `False` | nền trắng khi render (Blender-style datasets) |
| huấn luyện | `output_root` | `/content/output` | nơi lưu model mỗi scene |
| | `iterations` | `7000` | số vòng train mặc định (đặt `30000` cho chất lượng đầy đủ) |
| | `smoke_iterations` | `300` | số vòng của `run.smoke_test` |
| | `score_every` | `1000` | tần suất chấm điểm + in log trong lúc train |
| | `save_every` | `2000` | tần suất ghi checkpoint `.ply` giữa chừng; `0` = chỉ lưu ở vòng cuối |
| | `keep_last_checkpoint` | `True` | xoá checkpoint trung gian cũ, đĩa chỉ giữ bản mới nhất |
| | `eval_views` | `6` | số camera hold-out dùng để chấm điểm sống |
| | `mult` | `0.5` | hệ số compact-box của FastGS renderer |
| | `psnr_max` | `30.0` | mốc chuẩn hoá PSNR — đúng giá trị ban tổ chức dùng |
| | `lpips_net_live` | `"alex"` | mạng LPIPS dùng khi theo dõi trong lúc train (nhanh) |
| | `lpips_net_report` | `"vgg"` | mạng LPIPS dùng khi chấm điểm báo cáo lúc render submission |
| | `ram_soft_limit_gb` | `10.5` | ngưỡng RAM để chủ động `gc.collect()` |
| | `train_extra_args` | `["--densification_interval","500", "--lambda_dssim","0.25", "--highfeature_lr","0.02", "--loss_thresh","0.07", "--grad_abs_thresh","0.0012"]` | cờ CLI FastGS bổ sung, nối thẳng vào `build_args` |
| submission | `submission_dir` | `/content/submission` | thư mục gốc chứa `<scene>/000N.png` |
| | `submission_zip` | `/content/submission.zip` | đường dẫn zip cuối cùng |
| | `submission_ext` | `.png` | đuôi file ảnh nộp |
| | `submission_digits` | `4` | số chữ số đánh index → `0001.png` |
| | `submission_resolution` | `1` | `-r 1` = render đúng kích thước ảnh gốc (không downscale) |
| tải về | `download_submission` | `True` | có tải `submission.zip` khi `run.finish` |
| | `download_model` | `True` | có tải kèm `.ply` + báo cáo |
| | `save_to_drive` | `False` | chép thêm sang Google Drive |
| | `drive_dir` | `/content/drive/MyDrive` | đích khi `save_to_drive=True` |
| | `model_zip` | `/content/fastgs_models.zip` | tên zip chứa model |

`cfg.show()` chỉ in một tập con các trường trên (xem danh sách `keys` trong `config.py`), không phải toàn bộ; dùng `cfg.as_dict()` hoặc `cfg.save(path)` để lấy/ghi đầy đủ.

### Checkpoint `.ply` trong lúc train

`train_scene` gọi `_save_checkpoint` (`pipeline/trainer.py:39-47`) ở hai thời điểm:

| Khi nào | Ghi ra | Dọn dẹp |
|---|---|---|
| Mỗi `save_every` vòng (mặc định 2000; `0` = tắt) | `output/<scene>/point_cloud/iteration_<n>/point_cloud.ply` | xoá mốc trung gian trước đó nếu `keep_last_checkpoint=True` |
| Vòng cuối cùng | `.../iteration_<iterations>/` | xoá nốt mốc trung gian còn lại |

Mục đích là **chống mất phiên Colab**: đứt giữa chừng thì vẫn còn `.ply` gần nhất, render được ngay bằng
`submission.render_scene(cfg, scene, iterations=<n>)`. Đây **không** phải checkpoint để train tiếp — chỉ tham số
Gaussian được ghi, trạng thái Adam và bộ đếm densify thì không, nên chạy lại vẫn là train từ đầu.
Đặt `keep_last_checkpoint=False` nếu muốn giữ mọi mốc để so chất lượng theo iteration (tốn đĩa hơn).

---

## 3. Công thức điểm — ba metric của cuộc thi

Ban tổ chức chấm bằng ba chỉ số:

- **LPIPS** (Learned Perceptual Image Patch Similarity) — Zhang et al., CVPR 2018, [arXiv:1801.03924](https://arxiv.org/abs/1801.03924). Càng **thấp** càng tốt.
- **SSIM** và **PSNR** — Wang et al., *IEEE Transactions on Image Processing* 13(4):600–612, 2004, [doi:10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861). Càng **cao** càng tốt.

PSNR được chuẩn hoá trước khi cộng vào điểm tổng:

```python
psnr_norm = torch.clamp(psnr_val / psnr_max, 0.0, 1.0)
```

Điểm tổng hợp của một scene:

```
Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * psnr_norm
```

Giá trị leaderboard = **trung bình cộng `Score` trên mọi scene**. Đây đúng là công thức `pipeline/report.leaderboard` dùng: nó thêm một hàng `MEAN` (trung bình mọi cột số, gồm cả `score`) vào cuối bảng, và in ra `Điểm leaderboard (trung bình các scene): <MEAN score>`.

### Nơi triển khai

| Mục đích | File | Chi tiết |
|---|---|---|
| Điểm sống trong lúc train (theo dõi tiến trình, chọn siêu tham số) | `pipeline/score.py` — `composite_score`, `evaluate_cameras` | Dùng `lpips_net_live` (mặc định `"alex"`, nhanh); gọi mỗi `score_every` vòng trên `eval_views` camera hold-out, tqdm và log in cả `Score`, `ΔScore`, `PSNR`, `psnr_norm`, `SSIM`, `LPIPS`, RAM, VRAM |
| Điểm báo cáo khi render submission (gần với điểm thi thật hơn) | `pipeline/submission.py` — `render_scene` | Dùng `lpips_net_report` (mặc định `"vgg"` — đúng mạng gốc bài báo LPIPS hay dùng để đánh giá), tính trên **toàn bộ** test camera của scene, không chỉ mẫu con |
| Trung bình toàn bộ scene → số leaderboard | `pipeline/report.py` — `leaderboard` | Hàng `MEAN` trên mọi scene |

Cả hai nơi gọi chung `composite_score` nên công thức không lệch nhau — chỉ khác **mạng LPIPS** và **số lượng/tập camera** dùng để ước lượng (mẫu nhanh lúc train, đầy đủ lúc render submission).

---

## 4. Hợp đồng `submission.zip`

Theo luật cuộc thi: nộp ảnh RGB cho **mọi** test pose được cấp, đúng hình học/vị trí vật thể, hình ảnh thực tế và nhất quán; thiếu/thừa scene so với ground-truth thì **không được chấm điểm**.

### Cây thư mục

```
submission.zip
├── <scene_1>/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── <scene_2>/
│   └── ...
└── ...
```

- Tên thư mục = đúng tên scene (khớp với `find_scenes` phát hiện, hoặc `cfg.scenes` nếu chỉ định thủ công).
- Tên file = index liên tục bắt đầu từ 1, đệm số 0, đuôi lấy từ `cfg.submission_ext`: `f"{index:0{cfg.submission_digits}d}{cfg.submission_ext}"` → mặc định `0001.png`, `0002.png`, ... (`submission_digits=4`).
- Số ảnh mỗi scene = số test camera của scene đó (`Scene.getTestCameras()`, sắp theo `image_name`).
- Kích thước ảnh: `submission_resolution=1` nghĩa là render đúng độ phân giải ảnh gốc của scene (không downscale như lúc train).
- `render_scene` còn ghi kèm `_manifest.json` trong mỗi thư mục scene (không nằm trong zip — `build_zip` chỉ nhặt file có đuôi `submission_ext`) — chứa danh sách file, `source` (tên ảnh gốc), `width`/`height`, và metric nếu có ground-truth.
- Zip dùng `ZIP_STORED` (không nén thêm — PNG đã nén sẵn), nên gần như không tốn RAM khi đóng gói.

### `submission.verify()` kiểm tra gì

`pipeline/submission.py::verify`:

1. Mọi ảnh phải nằm trong một thư mục con (`<scene>/<file>`) — ảnh nằm ngoài scene bị báo lỗi.
2. Với mỗi scene: tên file phải **liên tục** đúng mẫu `0001.png, 0002.png, ...` không thiếu/nhảy số (so với `_image_name(i, cfg)` sinh ra).
3. Lấy kích thước (width, height) từ ảnh đầu tiên của mỗi scene để hiển thị trong bảng kết quả.
4. Nếu truyền `expected` (danh sách scene mong đợi, `run.finish` truyền vào là `scenes` đã train) — báo **thiếu scene** hoặc **thừa scene** so với `expected`.
5. Trả về `(frame, problems)`: `frame` là bảng `pandas` (scene, số ảnh, width, height); in `"OK: submission hợp lệ"` nếu `problems` rỗng, ngược lại in danh sách cảnh báo.

**Giới hạn cần biết:** `verify()` chỉ kiểm tra được tính **nội bộ nhất quán** của file zip (tên liên tục, không có scene thừa/thiếu so với danh sách `expected` do chính người dùng cung cấp) và **không** so khớp số lượng ảnh hay kích thước với ground-truth thật của ban tổ chức (vì local không có ground-truth đó) — nó không thể tự phát hiện việc thiếu ảnh nếu số test-pose cục bộ vốn đã khác với bộ chấm thật. Việc "hình học đúng, vật thể đúng vị trí, ảnh thực tế và nhất quán" cũng không được kiểm tra tự động ở đây — chỉ có thể soi bằng mắt qua `report.show_samples` (so ảnh render với ảnh ground-truth cạnh nhau).

---

## 5. Chạy trên dữ liệu của riêng bạn

Trong cell 2 của notebook (hoặc gọi trực tiếp `pipeline.Config`):

```python
from pipeline import Config

cfg = Config(
    data_root="/content/data_cua_toi",   # thư mục chứa scene (COLMAP sparse/ hoặc transforms_train.json)
    dataset_url=None,                    # None = dữ liệu đã đặt sẵn trong data_root, không tải/giải nén
    scenes=(),                           # () = tự tìm hết scene trong data_root (qua find_scenes)
    iterations=30000,                    # chất lượng đầy đủ — đừng hạ dưới 20000 (xem colab-t4-guide.md §2.1)
)
```

- `find_scenes` (trong `pipeline/data.py`) tự nhận diện một thư mục là scene nếu có `sparse/` (COLMAP) hoặc `transforms_train.json` (kiểu Blender), quét tối đa `max_depth=3` cấp thư mục con.
- Muốn chỉ train một tập con: đặt `scenes=("scene_a", "scene_b")` — scene không tìm thấy sẽ được in cảnh báo và bỏ qua.
- `iterations=30000` là ngân sách đầy đủ mà lịch optimizer/densify của FastGS được thiết kế theo (xem `colab-t4-guide.md` mục 2.1) — dùng giá trị nhỏ hơn (mặc định demo `7000`, hoặc test nhanh bằng `smoke_iterations=300`) chỉ để thử nghiệm, không phải để nộp bài.
- Phần còn lại của quy trình (`run.load_data` → `run.smoke_test` → `run.run_all` → `run.analytics` → `run.finish`) giữ nguyên, không cần sửa gì thêm.
