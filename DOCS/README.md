# DOCS — mục lục

Tài liệu của fork này. `docs/` và `docs2/` đã được gộp làm một thư mục `DOCS/`.

## Đang dùng — bám sát mã nguồn hiện tại (fastgs-lite)

| Tài liệu | Nội dung | Đi kèm |
|---|---|---|
| [fastgs-acceleration-method.md](fastgs-acceleration-method.md) | **Tài liệu chính, tự chứa.** Nền tảng toán 3DGS (chiếu covariance, alpha blending, loss) → mô hình chi phí một vòng lặp → ba đòn bẩy tăng tốc (điểm số nhất quán đa góc nhìn, compact box `--mult`, nhịp Adam) → **phép đo fastgs-lite vs 3DGS** | `demos/fastgs_mechanisms.py`, `demos/fastgs_cost_model.py` |
| [colab-t4-guide.md](colab-t4-guide.md) | Huấn luyện thật trên Colab free T4: preset A, điểm Score theo dõi, chống tràn RAM, roadmap tầng CUDA | `fastgs-acceleration-method.ipynb`, `pipeline/` |
| [pipeline-and-submission.md](pipeline-and-submission.md) | Tham chiếu gói `pipeline/`: bảng cell↔hàm, các trường `Config`, công thức điểm (LPIPS/SSIM/PSNR) và nơi triển khai, hợp đồng `submission.zip`, cách chạy trên dữ liệu riêng | `pipeline/`, `fastgs-acceleration-method.ipynb` |
| [history-train.md](history-train.md) | Nhật ký các phiên train thật: cấu hình, bảng điểm từng cảnh, diễn biến theo vòng lặp, tài nguyên đo được, việc cần làm cho phiên sau | `fastgs-acceleration-method.ipynb` |

### Cấu trúc repo hiện tại

- `pipeline/` — toàn bộ logic Python thuần (không phụ thuộc Colab để import): `config.py` (tham số), `env.py` (máy/GPU/RAM), `data.py` (tải & liệt kê scene), `score.py` (công thức điểm chính thức), `trainer.py` (vòng train fastgs-lite), `submission.py` (render test pose + đóng gói/kiểm tra zip), `report.py` (bảng/biểu đồ), `deliver.py` (đóng gói model + tải về), `run.py` (điều phối toàn bộ). Chi tiết: [pipeline-and-submission.md](pipeline-and-submission.md).
- `fastgs-acceleration-method.ipynb` (gốc repo) — chỉ còn **glue code tiếng Anh**, 9 code cell (clone → GPU → config → data → smoke test → train → analytics → submission+download); mọi logic nằm trong `pipeline/*.py`; cần GPU.
- `demos/fastgs_mechanisms.py` — các mô phỏng thu nhỏ bằng NumPy của cơ chế fastgs-lite (không cần CUDA), tách ra khỏi notebook: `python demos/fastgs_mechanisms.py`.
- `demos/fastgs_cost_model.py` — mô hình chi phí một vòng lặp và **phép đo fastgs-lite vs 3DGS** (số tile mỗi splat, số lần Adam step, quỹ đạo số Gaussian): `python demos/fastgs_cost_model.py`. Xem [fastgs-acceleration-method.md](fastgs-acceleration-method.md) Phần VI.

## Tham chiếu sâu — giải phẫu toàn bộ một phiên train

Bộ ba tài liệu chi tiết nhất repo: mỗi file, mỗi hàm, mỗi hằng số, mỗi nhánh `if` mà một phiên
train chạm tới, theo đúng thứ tự thực thi. Đánh số mục §1–§48 chạy liên tục qua cả ba tài liệu.
Đã viết lại hoàn toàn theo mã nguồn fastgs-lite hiện tại (bản cũ mô tả codebase DroneSplat với
`--use_masks`, SAM2, DUSt3R, `pose_optimizer` — những thứ **không còn tồn tại**).

| Tài liệu | Mục | Nội dung |
|---|---|---|
| [DIGITAL-TWIN-GS-PIPELINE-1.md](DIGITAL-TWIN-GS-PIPELINE-1.md) | §1–§18 | Từ câu lệnh tới dữ liệu sẵn sàng: hai đường chạy (CLI vs `pipeline/`), bản đồ hệ thống, toàn bộ cờ CLI, dựng `Scene` từ COLMAP, `GaussianModel.create_from_pcd` + `training_setup` |
| [DIGITAL-TWIN-GS-PIPELINE-2.md](DIGITAL-TWIN-GS-PIPELINE-2.md) | §19–§33 | Một vòng lặp train: `render_fastgs` + compact box, rasterizer CUDA, loss, `backward`, điểm số đa góc nhìn, densify điều kiện kép, ba tầng prune, phẫu thuật trạng thái Adam |
| [DIGITAL-TWIN-GS-PIPELINE-3.md](DIGITAL-TWIN-GS-PIPELINE-3.md) | §34–§48 | Lưu `.ply`, render, chấm điểm, `submission.zip`, bảng hằng số toàn hệ thống, bảng tra nhanh, chẩn đoán sự cố, thuật ngữ |

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
