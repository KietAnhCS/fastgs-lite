# DOCS — mục lục

Tài liệu của fork này. `docs/` và `docs2/` đã được gộp làm một thư mục `DOCS/`.

## Đang dùng — bám sát mã nguồn hiện tại (FastGS)

| Tài liệu | Nội dung | Đi kèm |
|---|---|---|
| [gaussian-splatting-math.md](gaussian-splatting-math.md) | Nền tảng toán học 3DGS: chiếu covariance, alpha blending, loss | — |
| [fastgs-acceleration-method.md](fastgs-acceleration-method.md) | Cơ chế tăng tốc của FastGS: điểm số nhất quán đa góc nhìn, densify điều kiện kép, compact box `--mult`, tách lr SH | `fastgs-acceleration-method.ipynb` Phần 1–6 |
| [colab-t4-guide.md](colab-t4-guide.md) | Huấn luyện thật trên Colab free T4: preset A, điểm Score theo dõi, chống tràn RAM, ba nâng cấp thuần Python, roadmap tầng CUDA | `fastgs-acceleration-method.ipynb` Phần 7–9 |

Notebook nằm ở gốc repo: [`fastgs-acceleration-method.ipynb`](../fastgs-acceleration-method.ipynb).
Phần 1–6 chạy được không cần CUDA (mô phỏng thu nhỏ bằng NumPy); Phần 7–9 cần GPU.

## Lưu trữ — mô tả codebase DroneSplat cũ, **đã lỗi thời**

| Tài liệu | Nội dung |
|---|---|
| [DIGITAL-TWIN-GS-PIPELINE-1.md](DIGITAL-TWIN-GS-PIPELINE-1.md) | Giải phẫu một phiên train: khởi tạo `Scene`, `GaussianModel`, tư thế camera |
| [DIGITAL-TWIN-GS-PIPELINE-2.md](DIGITAL-TWIN-GS-PIPELINE-2.md) | Thân vòng lặp train, mặt nạ, backward, densify, lưu |
| [DIGITAL-TWIN-GS-PIPELINE-3.md](DIGITAL-TWIN-GS-PIPELINE-3.md) | Lưu, render, chấm điểm, đối chiếu output |

> ⚠️ **Ba tài liệu này mô tả một `train.py` khác.** Chúng mổ xẻ lệnh
> `python train.py -s data/HCM0539 --scene HCM0539 --iter 7000 --use_masks --schedule_densify_grad_threshold`,
> nhưng `train.py` hiện tại **không còn** các cờ `--scene`, `--iter`, `--use_masks`,
> `--schedule_densify_grad_threshold`, cũng như luồng mặt nạ SAM2 / tinh chỉnh tư thế DUSt3R.
>
> Giữ lại vì phần **giải phẫu chung của 3DGS** (đọc COLMAP, dựng camera, backward qua rasterizer,
> ghi `.ply`, chấm PSNR/SSIM/LPIPS) vẫn đúng và rất chi tiết. Nhưng **mọi tên cờ, đường dẫn và
> chữ ký hàm phải đối chiếu lại với mã nguồn** trước khi tin.
>
> Muốn hiểu luồng train hiện tại: đọc `fastgs-acceleration-method.md` + `colab-t4-guide.md`.

## Bản đồ tài liệu ↔ mã nguồn

| Chủ đề | File mã nguồn |
|---|---|
| Điểm số nhất quán đa góc nhìn | `utils/fast_utils.py` (`sampling_cameras:10`, `compute_gaussian_score_fastgs:45`) |
| Densify điều kiện kép, ba tầng prune | `scene/gaussian_model.py` (`densify_and_prune_fastgs:468`, `metric_mask:494`, `final_prune_fastgs:533`) |
| Lịch optimizer đóng đinh theo 30k vòng | `scene/gaussian_model.py:225–244` |
| Tách lr SH bậc thấp/cao | `scene/gaussian_model.py:198–205` (lưu ý `highfeature_lr / 20.0`) |
| Compact box `--mult`, cấu hình rasterizer | `gaussian_renderer/__init__.py:18–58` |
| Tham số CLI + mặc định | `arguments/__init__.py` |
| Vòng train gốc | `train.py:37–177` |
| Render + chấm điểm | `render.py`, `metrics.py` |
| Nạp camera đa độ phân giải | `scene/__init__.py:25–83`, `utils/camera_utils.py:19–60` |
