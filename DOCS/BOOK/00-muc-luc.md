# Digital Twin GS — FastGS-lite: Sách hợp nhất toàn tập

## Lời nói đầu

Cuốn sách này là bản **hợp nhất toàn bộ** nội dung của thư mục `DOCS/` trong repo `digital-twin-gs` (fastgs-lite) — trước đây gồm khoảng 30 file Markdown rời rạc nằm ở nhiều cấp thư mục (`DOCS/`, `DOCS/Report/`, `DOCS/Report/test/`). Mục tiêu là gộp lại thành một mạch đọc liền lạc, chia thành **15 chương**, mỗi chương một file, để người đọc — kể cả người chưa quen với repo — có thể đi từ tổng quan dự án, xuống tới từng dòng code, rồi tới nền tảng toán học 3D Gaussian Splatting và các đòn bẩy tăng tốc FastGS-lite, mà không phải tự lần theo hàng chục link chéo.

**Không nội dung nào bị lược bỏ.** Mỗi chương ghi rõ ở đầu file nó lấy nguyên văn từ (những) file nào trong `DOCS/` gốc — dùng để truy vết ngược khi cần đối chiếu với bản gốc hoặc với mã nguồn. Khi hai nguồn có nội dung trùng lặp (ví dụ phần lý thuyết trong `Report/` và tài liệu gốc `fastgs-acceleration-method.md`), sách giữ **cả hai**, chỉ ghi chú để người đọc hiểu vì sao thấy lặp.

Công thức toán (LaTeX), bảng, code block, sơ đồ Mermaid và ảnh minh hoạ đều được giữ nguyên; chỉ có đường dẫn ảnh tương đối được sửa lại cho khớp vị trí file mới trong `DOCS/BOOK/`.

## Mục lục

| # | Chương | Nội dung chính |
|---|---|---|
| 1 | [Giới thiệu & Tổng quan dự án](01-gioi-thieu-tong-quan.md) | Bản đồ tài liệu, gói `pipeline/`, `Config`, công thức điểm thi, hợp đồng `submission.zip` |
| 2 | [Kiến trúc Pipeline toàn hệ thống — Phần 1 (§1–§18)](02-kien-truc-pipeline-phan-1.md) | Hai đường chạy CLI/notebook, bản đồ hệ thống, toàn bộ cờ CLI, dựng `Scene` từ COLMAP, khởi tạo `GaussianModel` |
| 3 | [Vòng lặp huấn luyện chi tiết — Phần 2 (§19–§33)](03-vong-lap-huan-luyen-phan-2.md) | `render_fastgs` + compact box, rasterizer CUDA, loss, backward, điểm số đa góc nhìn, densify/prune, Adam |
| 4 | [Lưu, Render, Chấm điểm & Vận hành — Phần 3 (§34–§48)](04-luu-render-cham-diem-phan-3.md) | Lưu `.ply`, render, chấm điểm, `submission.zip`, bảng hằng số, chẩn đoán sự cố, thuật ngữ |
| 5 | [Ký hiệu & Nền tảng toán học chung](05-ky-hieu-nen-tang-toan-hoc.md) | Sơ đồ pipeline 3DGS, bảng ký hiệu $\theta_i,\mu_i,\Sigma_i,\ldots$, hằng số hệ thống |
| 6 | [Initialization — Từ SfM Points đến Gaussian khởi tạo](06-initialization.md) | `create_from_pcd`, `extent`, cảnh đồ chơi dùng chung, kiểm định số chương 1 |
| 7 | [Biểu diễn 3D Gaussians](07-3d-gaussians.md) | $\Sigma=RSS^\top R^\top$, $G(x)$, SH color, kiểm định số chương 2 |
| 8 | [Projection & Compact Box (đòn bẩy giảm K)](08-projection-compact-box.md) | EWA splatting, compact box, `box.md`/`box2.md`, kiểm định số chương 3 |
| 9 | [Differentiable Tile Rasterizer](09-differentiable-tile-rasterizer.md) | Alpha-blending, transmittance, sort theo khoá 64-bit, kiểm định số chương 4 |
| 10 | [Từ Ảnh đến Loss & Metrics](10-anh-den-loss-metrics.md) | L1, SSIM, PSNR, Score tổng hợp, kiểm định số chương 5 |
| 11 | [Gradient Flow & Backpropagation](11-gradient-flow-backprop.md) | $\partial C/\partial\alpha_n$, gradient tuyệt đối, Adam, lr schedule, kiểm định số chương 6 |
| 12 | [Adaptive Density Control (đòn bẩy giảm N)](12-adaptive-density-control.md) | Importance/Pruning score, densify AND, prune multinomial, ví dụ số end-to-end |
| 13 | [Tổng hợp Mô hình Chi phí & Phương pháp Tăng tốc FastGS](13-tong-hop-chi-phi-fastgs.md) | Mô hình chi phí $T_{\text{iter}}$, ba đòn bẩy nhân nhau, toàn văn `fastgs-acceleration-method.md` (Phần I–IX) |
| 14 | [Triển khai Thực tế: Colab T4 & Nhật ký Huấn luyện](14-trien-khai-colab-nhat-ky-train.md) | Preset A trên T4, chống tràn RAM, roadmap CUDA, nhật ký các phiên train thật |
| 15 | [Phụ lục: Kiểm định số chéo, Dữ liệu & Tài nguyên](15-phu-luc-kiem-dinh-tai-nguyen.md) | Tổng hợp 8 bài test số, `history.csv`/`leaderboard.csv`, công cụ `test.html`, danh mục script |

## Nguồn tài liệu gốc

Toàn bộ nội dung trên được hợp nhất từ các file sau trong `DOCS/` (đường dẫn tính từ gốc `digital-twin-gs/`):

- `DOCS/README.md`
- `DOCS/pipeline-and-submission.md`
- `DOCS/DIGITAL-TWIN-GS-PIPELINE-1.md`
- `DOCS/DIGITAL-TWIN-GS-PIPELINE-2.md`
- `DOCS/DIGITAL-TWIN-GS-PIPELINE-3.md`
- `DOCS/fastgs-acceleration-method.md`
- `DOCS/3.md`
- `DOCS/box.md`
- `DOCS/box2.md`
- `DOCS/colab-t4-guide.md`
- `DOCS/history-train.md`
- `DOCS/test.html`
- `DOCS/assets/history.csv`
- `DOCS/assets/leaderboard.csv`
- `DOCS/Report/README.md`
- `DOCS/Report/00-ky-hieu.md`
- `DOCS/Report/01-initialization.md`
- `DOCS/Report/02-3d-gaussians.md`
- `DOCS/Report/03-projection.md`
- `DOCS/Report/04-rasterizer.md`
- `DOCS/Report/05-image-loss.md`
- `DOCS/Report/06-gradient-flow.md`
- `DOCS/Report/07-adaptive-density-control.md`
- `DOCS/Report/08-tong-hop.md`
- `DOCS/Report/test/README.md`
- `DOCS/Report/test/00-scene.md`
- `DOCS/Report/test/01-test.md` … `08-test.md` (8 file)
- `DOCS/Report/test/scripts/ch01_test.py` … `ch08_test.py`, `ch01_plot.py` … `ch08_plot.py` (16 file, liệt kê ở Chương 15)

Ảnh minh hoạ giữ nguyên từ `DOCS/assets/`, `DOCS/Report/assets/`, `DOCS/Report/test/figures/`.

---

[Bắt đầu đọc — Chương 1 →](01-gioi-thieu-tong-quan.md)
