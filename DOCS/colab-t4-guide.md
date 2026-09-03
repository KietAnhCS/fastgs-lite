# Huấn luyện FastGS trên Google Colab free (T4)

> Tài liệu đồng hành của notebook [`fastgs-acceleration-method.ipynb`](../fastgs-acceleration-method.ipynb), **Phần 7–9**.
> Phần 1–6 của notebook là mô phỏng thu nhỏ (không cần CUDA) — xem [fastgs-acceleration-method.md](fastgs-acceleration-method.md).
> Nền tảng toán học 3DGS: [gaussian-splatting-math.md](gaussian-splatting-math.md).

Mục tiêu: **chất lượng cao nhất trên mỗi phút GPU** với ràng buộc của Colab free, và train xong vẫn **lấy được mô hình về máy**.

---

## 1. Ràng buộc thật của Colab free

| Tài nguyên | Hạn mức | Rủi ro chính |
|---|---|---|
| GPU T4 | ~15 GB VRAM | OOM khi số Gaussian phình hoặc ảnh độ phân giải cao |
| RAM hệ thống | ~12.7 GB | **train xong RAM còn kẹt → `zip` + `files.download` OOM → mất mô hình** |
| Đĩa | ~78–110 GB | thường không phải nút thắt |
| Phiên | có thể bị ngắt | phải lưu `.ply` ra đĩa **sớm**, không đợi tới cuối |

So sánh: README gốc khuyến nghị **24 GB VRAM / RTX 4090** cho kết quả đúng như paper. T4 vẫn chạy được, nhưng phải chọn độ phân giải và ngân sách primitive cho phù hợp.

---

## 2. Ba phát hiện khi đọc mã nguồn

Ba điều này quyết định mọi khuyến nghị bên dưới. Chúng **không có trong README**.

### 2.1 — Lịch optimizer bị đóng đinh theo ngân sách 30.000 vòng

`scene/gaussian_model.py:225` — `optimizer_step(iteration)`:

| Khoảng vòng lặp | Hành vi |
|---|---|
| `iteration <= 15000` (dòng 227) | bước Adam **mỗi vòng**; optimizer SH bước mỗi 16 vòng |
| `15000 < iteration <= 20000` (dòng 233) | bước Adam **mỗi 32 vòng** |
| `iteration > 20000` | bước Adam **mỗi 64 vòng** |

Cộng thêm `densify_until_iter = 15000` và `position_lr_max_steps = 30000`, hệ quả:

- **Toàn bộ chi phí nằm ở 0–15k** (densify + Adam đầy đủ + số Gaussian đang tăng).
- **15k–30k gần như miễn phí** nhưng vẫn chạy `final_prune_fastgs` (mỗi 3000 vòng) → ít Gaussian hơn, LPIPS tốt hơn, file `.ply` nhỏ hơn.

> **Kết luận: đừng hạ `--iterations` xuống 15k.** Tiết kiệm rất ít thời gian mà mất luôn phần đánh bóng + tỉa gần-như-cho-không. Nếu buộc phải rút ngắn, sàn là **20000**, và phải đồng bộ `--position_lr_max_steps = N`, `--densify_until_iter ≈ N/2`, đồng thời sửa hai ngưỡng cứng ở dòng 227/233 thành `int(0.5*N)` / `int(0.67*N)`.

### 2.2 — `--antialiasing` là cờ chết trong fork này

`arguments/__init__.py:70` có `self.antialiasing = False`, nhưng:

- `gaussian_renderer/__init__.py` dựng `GaussianRasterizationSettings(...)` **không có** trường `antialiasing`;
- `submodules/diff-gaussian-rasterization_fastgs/` **không tham chiếu** chữ `antialiasing` ở bất kỳ đâu.

Truyền cờ này chỉ bị bỏ qua âm thầm — không lỗi, không tác dụng. Đây là di sản từ code INRIA gốc. Muốn có chống răng cưa thật phải vá kernel CUDA (mục 6.1).

### 2.3 — Bước chấm điểm đa góc nhìn đắt hơn vẻ ngoài

`utils/fast_utils.py:45` `compute_gaussian_score_fastgs` lấy mẫu **10 camera** (`num_cams = 10`, dòng 13) và render **hai lượt** mỗi camera (một lượt thường, một lượt `get_flag=True` với `metric_map`). Nó chạy mỗi `densification_interval` vòng trong khoảng 500–15000.

| `densification_interval` | Số lần gọi trong 0–15k |
|---|---|
| `100` (mặc định của `arguments/__init__.py`) | ~145 lần |
| `500` (giá trị `train_base.sh` thật sự dùng) | ~29 lần |

⇒ Đây là lever thời gian lớn thứ hai sau độ phân giải, và nó là **setting chính chủ**, không phải thoả hiệp.

---

## 3. Điểm số theo dõi tiến trình

Một số duy nhất trong $[0,1]$, càng cao càng tốt:

$$\mathrm{Score}=0.4\,(1-\mathrm{LPIPS})+0.3\,\mathrm{SSIM}+0.3\,\widehat{\mathrm{PSNR}},
\qquad
\widehat{\mathrm{PSNR}}=\operatorname{clamp}\!\Big(\tfrac{\mathrm{PSNR}}{\mathrm{PSNR}_{\max}},\,0,\,1\Big)$$

| Ký hiệu | Ý nghĩa | Miền | Đóng góp |
|---|---|---|---|
| $\mathrm{LPIPS}$ | khoảng cách tri giác (VGG); thấp = giống | $[0,1]$ | $0.4\,(1-\mathrm{LPIPS})$ |
| $\mathrm{SSIM}$ | tương đồng cấu trúc; cao = giống | $[0,1]$ | $0.3\,\mathrm{SSIM}$ |
| $\widehat{\mathrm{PSNR}}$ | `torch.clamp(psnr_val / psnr_max, 0.0, 1.0)` | $[0,1]$ | $0.3\,\widehat{\mathrm{PSNR}}$ |
| $\mathrm{PSNR}_{\max}$ | mốc bão hoà, mặc định **30 dB** | hằng | — |

**Vì sao chuẩn hoá PSNR.** LPIPS và SSIM đã nằm sẵn trong $[0,1]$; PSNR thì không có trần. Để nguyên, một cảnh dễ (PSNR 35) sẽ áp đảo tổng và Score mất khả năng so sánh giữa các cảnh. `clamp` kéo về cùng thang **và triệt tiêu phần thưởng cho việc vượt xa ngưỡng "đủ tốt"**.

### Biên lợi ích — tối ưu cái gì

$$\frac{\partial \text{Score}}{\partial \text{LPIPS}}=-0.4,\qquad
\frac{\partial \text{Score}}{\partial \text{SSIM}}=+0.3,\qquad
\frac{\partial \text{Score}}{\partial \text{PSNR}}=\frac{0.3}{30}=+0.01\ \text{/dB}\ \ (\text{chỉ khi } \mathrm{PSNR}<30)$$

Để **+0.01 Score** cần một trong: **LPIPS −0.025**, hoặc **SSIM +0.033**, hoặc **PSNR +1 dB**.

| Hệ quả | Ý nghĩa |
|---|---|
| LPIPS trọng số lớn nhất | ưu tiên chất lượng tri giác |
| PSNR bão hoà ở 30 dB | cảnh trong nhà (test PSNR ~29–32) đã gần đầy ⇒ **đừng train thêm để lấy PSNR** |
| Cảnh ngoài trời (~25–27 dB) | PSNR vẫn còn tác dụng |

> Giữ `PSNR_MAX` **cố định** giữa các lần chạy. Hạ nó xuống để "làm đẹp số" là tự lừa.

### Đo mỗi 1000 vòng

Notebook in $\Delta_{1k}\mathrm{Score}=\mathrm{Score}_t-\mathrm{Score}_{t-1000}$ cùng $\Delta_{1k}$ của từng thành phần. Chuỗi $\Delta$ dương và co dần về 0 = hội tụ lành mạnh; $\Delta$ âm kéo dài = overfit hoặc prune quá tay.

**Lưu ý nhất quán:** LPIPS "live" trong lúc train dùng `net_type="alex"` (nhanh ~3×, nạp ~30 MB thay vì ~500 MB). **Con số báo cáo** luôn lấy từ `metrics.py` (`vgg`). Hai giá trị lệch nhẹ là bình thường — live chỉ để xem xu hướng.

---

## 4. Preset A — cân bằng

```bash
-r 2 --eval --iterations 30000 \
  --densification_interval 500 \
  --lambda_dssim 0.25 \
  --highfeature_lr 0.02 \
  --loss_thresh 0.07 \
  --grad_abs_thresh 0.0012
```

| Tham số | Mặc định → dùng | Lý do |
|---|---|---|
| `-r` | `-1` → **`2`** (trong nhà) / **`4`** (ngoài trời) | đây là **độ phân giải eval chuẩn** của MipNeRF360 (`images_2`/`images_4`), không phải hy sinh. Lever thời gian lớn nhất: ×2–3 |
| `--densification_interval` | `100` → **`500`** | mục 2.3; giá trị `train_base.sh` thật sự dùng |
| `--iterations` | giữ **`30000`** | mục 2.1 |
| `--lambda_dssim` | `0.2` → **`0.25`** | nâng trọng số $(1-\mathrm{SSIM})$ trong loss ⇒ SSIM và thường cả LPIPS |
| `--highfeature_lr` | `0.005` → **`0.02`** | mọi cảnh indoor trong `train_base.sh` dùng 0.02; SH bậc cao hội tụ nhanh hơn. Lưu ý `gaussian_model.py:205` chia thêm cho 20 ⇒ lr hiệu dụng `0.02/20 = 0.001` |
| `--loss_thresh` | `0.1` → **`0.07`** | đánh dấu nhiều pixel lỗi hơn ⇒ giữ chi tiết (cảnh `garden` dùng 0.06) |
| `--grad_abs_thresh` | `0.0012` | núm chính theo cảnh: thấp hơn (0.0008–0.0010) ⇒ nét hơn nhưng nhiều Gaussian; cao hơn (0.0015–0.002) cho cảnh ngoài trời/thưa |
| `--dense` | `0.001` (indoor) → **0.004–0.01** cho ngoài trời | tỉ lệ clone/split đúng loại cảnh |
| ~~`--antialiasing`~~ | **không dùng** | mục 2.2 |

`--mult` giữ mặc định `0.5`; dùng `0.7` cho cảnh nền phức tạp (Tanks&Temples, Deep Blending) và **phải truyền y hệt cho `render.py`**.

> Cách chọn `grad_abs_thresh` / `dense`: mở `train_base.sh`, tìm dòng của cảnh giống dataset của bạn nhất.

---

## 5. Chống tràn RAM trên Colab — bảy quy tắc

Bẫy kinh điển: train xong, kernel vẫn giữ `gaussians` (tham số + trạng thái Adam), `scene` (toàn bộ ảnh GT), cache CUDA — RAM ~10–12 GB. Gọi `shutil.make_archive` / `files.download` lúc đó → thêm một bản copy trong RAM → **OOM, kernel chết, mất mô hình dù `.ply` đã nằm trên đĩa**.

| Khi nào | Việc làm | Ô notebook |
|---|---|---|
| Trong lúc train | `scene.save()` ngay tại `save_iterations` → `.ply` an toàn **trước khi** RAM căng | 7.4 |
| Mỗi 1000 vòng | `torch.cuda.empty_cache()`; RAM > `RAM_SOFT_LIMIT_GB` ⇒ `gc.collect()` | 7.4 |
| Mỗi vòng lặp | `del` mọi tensor trung gian | 7.4 |
| Ngay sau train | `del gaussians, scene`; `gc.collect()`; `empty_cache()`; `ipc_collect()` | 7.5 |
| Render + metrics | chạy `!python render.py` / `!python metrics.py` — **tiến trình con**, RAM/VRAM được HĐH thu hồi khi thoát | 7.6 |
| Đóng gói | `zipfile` chế độ `ZIP_STORED` ghi **ra đĩa**; không `BytesIO` / `make_archive` vào RAM | 7.9 |
| File > ~200 MB | chép sang Google Drive thay cho `files.download` | 7.9 |

Quy tắc bao trùm: **kernel không bao giờ giữ đồng thời "mô hình trong RAM" và "bản nén trong RAM".**

`metrics.py` nạp VGG cho LPIPS (~500 MB) — đó là lý do nó phải chạy ở tiến trình con, không gọi hàm trong kernel.

---

## 6. Ba nâng cấp thuần Python (notebook Phần 8)

Cài sẵn trong `train_fastgs`, **mặc định tắt**, bật bằng cờ. Không cần biên dịch lại CUDA.

### 6.1 — Coarse-to-fine (ý tưởng DashGaussian)

Dùng `Scene(resolution_scales=[...])` có sẵn của repo, đổi scale theo lịch:

$$r(t)=\begin{cases}
4.0 & t < 0.10\,T\\
2.0 & 0.10\,T \le t < 0.27\,T\\
1.0 & t \ge 0.27\,T
\end{cases}$$

Với `-r 2`, scale `4.0` cho ảnh bằng **1/8** cạnh gốc ⇒ mỗi vòng còn ~1/64 số pixel. Đoạn đắt nhất (0–15k) được rút gọn mạnh; về full-res tại `0.27·T` vẫn còn ~7k vòng densify ở độ phân giải thật.

Ba chi tiết đúng đắn phải xử lý (đã làm trong `train_fastgs`):

- `max_screen_size = 20 px` **chỉ áp dụng khi `scale == 1.0`** — ngưỡng theo pixel vô nghĩa ở ảnh nhỏ;
- `max_radii2D` được **reset** mỗi lần đổi scale, tránh trộn bán kính giữa hai độ phân giải;
- hold-out tính Score **luôn ở scale 1.0** để đường cong so sánh được xuyên suốt.

Đánh đổi: nạp 3 bộ camera ⇒ thêm ~30% bộ nhớ ảnh GT. VRAM căng thì dùng `resolution_scales=(2.0, 1.0)`.

### 6.2 — Ngân sách primitive (Taming-3DGS / 3DGS-MCMC)

FastGS chặn tăng trưởng bằng ngưỡng cứng `importance_score > 5` (`gaussian_model.py:494`). Thêm một **trần tuyệt đối** có lịch:

$$B(t) = N_0 + (N_{\max}-N_0)\Big(\tfrac{t}{0.5\,T}\Big)^{p}$$

Sau mỗi lần densify, nếu $n > B(t)$ thì cắt $n-B(t)$ Gaussian có opacity thấp nhất (`enforce_budget`). Số điểm bị chặn trần ⇒ thời gian mỗi vòng và VRAM đều bị chặn ⇒ **không OOM giữa chừng trên T4**. Gợi ý `gaussian_budget = 1_200_000` với `-r 2`.

### 6.3 — SGLD noise + phạt L1 (3DGS-MCMC)

$$\mathbf{x}_i \leftarrow \mathbf{x}_i + \underbrace{\sigma\big(-k(o_i-\tau)\big)}_{\text{cổng: chỉ đá Gaussian mờ}} \cdot\ \mathbf{s}_i \odot \boldsymbol{\epsilon}\ \cdot \lambda\,\eta_t,
\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(0, I)$$

cộng hai số hạng phạt vào loss: $\lambda_o\,\overline{o} + \lambda_s\,\overline{s}$. Nhiễu tỉ lệ với **scale riêng** của từng Gaussian và lr vị trí hiện tại $\eta_t$; chỉ bật khi `iteration < densify_until_iter` (giai đoạn optimizer còn bước mỗi vòng — xem mục 2.1).

### Ablation

Notebook chạy 3 cấu hình cùng dataset, cùng preset A, chỉ khác cờ:

| Cấu hình | Bật |
|---|---|
| `presetA` | không gì (đối chứng) |
| `coarse2fine` | 6.1 |
| `c2f_budget_mcmc` | 6.1 + 6.2 + 6.3 |

Xuất `ablation.csv` và biểu đồ **Score theo phút** — đường cong trả lời trực tiếp "chất lượng trên mỗi phút GPU".

---

## 7. Chưa wire — cần đụng CUDA

| Hạng mục | Công thức / việc phải làm | Ưu tiên |
|---|---|---|
| **Mip-Splatting** | $\alpha_i = o_i\sqrt{\frac{\lvert\Sigma'_i\rvert}{\lvert\Sigma'_i+sI\rvert}}\exp(-\tfrac12\Delta^\top(\Sigma'_i+sI)^{-1}\Delta)$, $s\approx0.1$ px², thay cho $\Sigma'\leftarrow\Sigma'+0.3I$; kèm lọc 3D $\Sigma\leftarrow\Sigma+\hat s^{-2}I$ với $\hat s=\max_k f_k/d_k$. Sửa `forward.cu`/`backward.cu`, thêm trường vào `GaussianRasterizationSettings`, truyền `pipe.antialiasing` từ `render_fastgs`, cài lại submodule | **1** — nhất là cho ảnh drone (tỉ lệ biến thiên lớn trong 1 khung hình) |
| **StopThePop** | khoá sắp xếp = độ sâu tại điểm đóng góp cực đại dọc tia: $t^\*=\arg\max_t \mathcal{G}(\mathbf{o}+t\mathbf{d})$. Thay submodule rasterizer, giữ nguyên Python | 2 |
| **2DGS / GOF** | đổi primitive sang đĩa phẳng, giao tia chính xác: $\mathbf{x}(u,v)=\mathbf{p}_c+s_u u\mathbf{t}_u+s_v v\mathbf{t}_v$, $\mathcal{G}=\exp(-\tfrac{u^2+v^2}{2})$. Cần `diff-surfel-rasterization` + 2 regularizer | cao, nếu mục tiêu là **mesh/bề mặt** |
| **3DGS-MCMC đầy đủ** | thiếu mảnh di dời bảo toàn ảnh: $o_{\text{new}}=1-(1-o_{\text{old}})^{1/N}$ + hiệu chỉnh $\Sigma$ khớp mô-men bậc 2. Đường ngắn hơn: chuyển backend sang `gsplat` (có sẵn strategy MCMC + antialiasing) | 3 |
| **Khởi tạo MASt3R/DUSt3R** | viết `sceneLoadTypeCallbacks["Dust3r"]` trả `BasicPointCloud` từ pointmap dày $X\in\mathbb{R}^{H\times W\times3}$ | cao, nếu ảnh trên không / ít chồng lấp |

> 3DGS và FastGS tối ưu cho **ảnh mới**, không cho **hình học**. Digital twin cần mesh thì nhánh đúng là 2DGS/GOF.

---

## 8. Thứ tự chạy notebook

| Ô | Việc |
|---|---|
| 7.1 | kiểm tra GPU + helper đo RAM/VRAM; clone repo, build submodule CUDA |
| 7.2 | trỏ `SOURCE_PATH`, lập hồ sơ dữ liệu (số ảnh, phân giải, tách train/test, điểm COLMAP) |
| 7.3 | định nghĩa `composite_score`, vẽ phương trình 2D/3D |
| 7.4 | `evaluate_on_holdout` → helper args → `train_fastgs` → chạy preset A |
| 7.5 | dọn RAM |
| 7.6 | `render.py` + `metrics.py` ở tiến trình con |
| 7.7 | biểu đồ: đường cong Score, cột $\Delta_{1k}$, số Gaussian + bộ nhớ, ảnh GT/render/bản đồ lỗi |
| 7.8 | thống kê per-view, xuất `stats.json` / `per_view_metrics.csv` / `REPORT.md` |
| 7.9 | đóng gói + tải mô hình |
| 8 | ablation 3 cấu hình |

**Ba chỗ phải sửa trước khi chạy:** `REPO_URL` (7.1), `SOURCE_PATH` (7.2), `RESOLUTION` (2 trong nhà / 4 ngoài trời).

Quy ước code trong notebook: **toàn bộ code là tiếng Anh, không comment**; giải thích nằm ở ô markdown.

---

## 9. Điều chưa kiểm chứng

Notebook được viết dựa trên việc đọc mã nguồn (`train.py`, `render.py`, `metrics.py`, `scene/gaussian_model.py`, `scene/__init__.py`, `utils/camera_utils.py`, `utils/fast_utils.py`, `gaussian_renderer/__init__.py`), **chưa chạy thử trên GPU**.

Trước khi dựa vào số liệu: chạy Phần 8 với `ABLATION_ITERS = 4000` một lượt để chắc cả ba nhánh không lỗi, rồi mới nâng lên 7000 và chạy preset A đầy đủ 30k.

Các con số trong tài liệu này là **suy luận từ mã nguồn và tham số của `train_base.sh`**, không phải kết quả đo trên máy bạn. Bảng ablation của chính bạn mới là bằng chứng.
