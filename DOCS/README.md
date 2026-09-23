# DOCS/ — Bản đồ tài liệu digital-twin-gs (fastgs-lite)

README gốc ở thư mục cha trỏ vào vài đường dẫn (`DOCS/fastgs-acceleration-method.md`,
`DOCS/colab-t4-guide.md`, `DOCS/history-train.md`, `DOCS/DIGITAL-TWIN-GS-PIPELINE-{1,2,3}.md`)
không còn tồn tại dưới tên đó — toàn bộ nội dung đã được **hợp nhất vào `DOCS/BOOK/`**
(17 chương, một mạch đọc liền, không lược bỏ gì). Trang này là bản đồ điều hướng: đi từ
tên file cũ (hoặc từ nhu cầu đọc) tới đúng chương trong `BOOK/`, cộng thêm mục lục
`Report/` và `Slides67/`.

## Kết quả thực nghiệm mới nhất

Chạy thật trên Colab T4 (không mô phỏng), scene `HCM0539` (dữ liệu cuộc thi
`VAI_NVS_DATA_ROUND2`, 240 ảnh train / 60 ảnh test), 30 000 iterations, toàn bộ cơ chế
Faster-GS đã merge (Fused Adam, 3D anti-aliasing filter, Morton reordering — xem
[00-muc-luc.md](BOOK/00-muc-luc.md#cập-nhật-tích-hợp-cơ-chế-từ-faster-gs)) đang **bật sẵn**
trong lần chạy này:

| Score | PSNR | SSIM | LPIPS | Gaussians cuối | Thời gian train | VRAM đỉnh |
|---|---|---|---|---|---|---|
| 0.8579 | 25.27 dB | 0.8659 | 0.1364 | 344 484 | 1776 s (~29.6 phút) | 1.92 GB / 14.56 GB (T4) |

Đây chính là số liệu mà `BOOK/00-muc-luc.md` còn ghi là "chưa đo được" cho các cơ chế
Faster-GS mới merge — lần chạy này lấp khoảng trống đó. Chi tiết đầy đủ (cấu hình,
`leaderboard.csv`, `history.csv`) nằm trong nhật ký ở
[14-trien-khai-colab-nhat-ky-train.md](BOOK/14-trien-khai-colab-nhat-ky-train.md), và phần
so sánh chất lượng với 3DGS gốc / FasterGS nằm ở
[17-so-sanh-3dgs-fastgs-fastergs.md](BOOK/17-so-sanh-3dgs-fastgs-fastergs.md).

## Từ tên file cũ tới chương mới

| Tên cũ (không còn tồn tại) | Nằm ở đâu trong `BOOK/` bây giờ |
|---|---|
| `DOCS/fastgs-acceleration-method.md` | [13-tong-hop-chi-phi-fastgs.md](BOOK/13-tong-hop-chi-phi-fastgs.md) — nguyên văn Phần I–IX: nền toán 3DGS → mô hình chi phí per-iteration → ba đòn bẩy tăng tốc → so sánh FastGS-lite vs 3DGS |
| `DOCS/colab-t4-guide.md` | [14-trien-khai-colab-nhat-ky-train.md](BOOK/14-trien-khai-colab-nhat-ky-train.md) — preset T4, quy tắc chống OOM, nhật ký các phiên train thật |
| `DOCS/history-train.md` | cũng trong [14-trien-khai-colab-nhat-ky-train.md](BOOK/14-trien-khai-colab-nhat-ky-train.md) (phần nhật ký) |
| `DOCS/DIGITAL-TWIN-GS-PIPELINE-1.md` | [02-kien-truc-pipeline-phan-1.md](BOOK/02-kien-truc-pipeline-phan-1.md) |
| `DOCS/DIGITAL-TWIN-GS-PIPELINE-2.md` | [03-vong-lap-huan-luyen-phan-2.md](BOOK/03-vong-lap-huan-luyen-phan-2.md) |
| `DOCS/DIGITAL-TWIN-GS-PIPELINE-3.md` | [04-luu-render-cham-diem-phan-3.md](BOOK/04-luu-render-cham-diem-phan-3.md) |

## Mục lục `BOOK/` (17 chương)

Xem đầy đủ tại [BOOK/00-muc-luc.md](BOOK/00-muc-luc.md); tóm tắt nhanh:

1. [Giới thiệu & Tổng quan dự án](BOOK/01-gioi-thieu-tong-quan.md)
2. [Kiến trúc Pipeline — Phần 1](BOOK/02-kien-truc-pipeline-phan-1.md) — CLI/notebook, `Scene`, `GaussianModel`
3. [Vòng lặp huấn luyện — Phần 2](BOOK/03-vong-lap-huan-luyen-phan-2.md) — render, rasterizer, loss, backward, densify/prune
4. [Lưu, Render, Chấm điểm & Vận hành — Phần 3](BOOK/04-luu-render-cham-diem-phan-3.md) — `.ply`, `submission.zip`, chẩn đoán sự cố
5. [Ký hiệu & Nền tảng toán học chung](BOOK/05-ky-hieu-nen-tang-toan-hoc.md)
6. [Initialization](BOOK/06-initialization.md) — SfM points → Gaussian khởi tạo
7. [Biểu diễn 3D Gaussians](BOOK/07-3d-gaussians.md)
8. [Projection & Compact Box](BOOK/08-projection-compact-box.md) — đòn bẩy giảm K
9. [Differentiable Tile Rasterizer](BOOK/09-differentiable-tile-rasterizer.md)
10. [Từ Ảnh đến Loss & Metrics](BOOK/10-anh-den-loss-metrics.md)
11. [Gradient Flow & Backpropagation](BOOK/11-gradient-flow-backprop.md)
12. [Adaptive Density Control](BOOK/12-adaptive-density-control.md) — đòn bẩy giảm N, đóng góp chính của FastGS
13. [Tổng hợp Mô hình Chi phí & Phương pháp Tăng tốc FastGS](BOOK/13-tong-hop-chi-phi-fastgs.md)
14. [Triển khai Thực tế: Colab T4 & Nhật ký Huấn luyện](BOOK/14-trien-khai-colab-nhat-ky-train.md)
15. [Phụ lục: Kiểm định số chéo, Dữ liệu & Tài nguyên](BOOK/15-phu-luc-kiem-dinh-tai-nguyen.md)
16. [Bài toán lớn: Từ COLMAP đến `.ply` render 3D](BOOK/16-bai-toan-lon-de-bai.md) + [lời giải 8 phần](BOOK/16-loi-giai/00-muc-luc-loi-giai.md)
17. [So sánh chất lượng: 3DGS gốc vs FastGS-lite vs FasterGS](BOOK/17-so-sanh-3dgs-fastgs-fastergs.md)

## `Report/`

Báo cáo minh hoạ bằng hình (`Report/assets/`, `Report/test/figures/` — covariance
ellipsoid, SH sphere, EWA projection, alpha blending, Adam trajectory, cost model, v.v.).
Hiện thư mục này **chỉ có ảnh, chưa có file `.md`/`.tex` thuyết minh riêng** — narrative
tương ứng với từng ảnh đã được viết trực tiếp trong các chương `BOOK/06`–`BOOK/13` (mỗi
chương đó dẫn ảnh từ đúng thư mục `Report/assets/chN_*.png` này).

## `Slides67/`

Bài thuyết trình LaTeX (`main.tex` + các `partN.tex`, `partADC_*.tex`, `partSH_*.tex`,
`part_fastergs_*.tex`) — build ra `main.pdf`. Nội dung bám theo `BOOK/`: `part1`–`part7`
tương ứng chương 1–13, `partADC_*` đi sâu chương 12 (Adaptive Density Control),
`partSH_*` đi sâu Spherical Harmonics (chương 7/`BOOK/SH.md`), `part_fastergs_*` là phần
merge Faster-GS (đối chiếu chương 17 mới).
