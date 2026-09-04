# Huấn luyện FastGS trên Google Colab free (T4)

> Tài liệu đồng hành của notebook [`fastgs-acceleration-method.ipynb`](../fastgs-acceleration-method.ipynb) và gói [`pipeline/`](../pipeline).
> Notebook chỉ gọi hàm trong `pipeline/`; toàn bộ code trong notebook và trong `pipeline/*.py` là **tiếng Anh** (tài liệu này viết bằng tiếng Việt).
> Mô phỏng thu nhỏ các cơ chế FastGS (không cần CUDA): [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) — xem [fastgs-acceleration-method.md](fastgs-acceleration-method.md).
> Nền tảng toán học 3DGS: [gaussian-splatting-math.md](gaussian-splatting-math.md).

Mục tiêu: **chất lượng cao nhất trên mỗi phút GPU** với ràng buộc của Colab free, và train xong vẫn **lấy được mô hình về máy**.

---

## 1. Ràng buộc thật của Colab free

| Tài nguyên | Hạn mức | Rủi ro chính |
|---|---|---|
| GPU T4 | ~15 GB VRAM | OOM khi số Gaussian phình hoặc ảnh độ phân giải cao |
| RAM hệ thống | ~12.7 GB | **train + render xong nhưng RAM còn kẹt → nén/tải về OOM → mất mô hình** |
| Đĩa | ~78–110 GB | thường không phải nút thắt |
| Phiên | có thể bị ngắt | giảm nhẹ bằng checkpoint định kỳ mỗi `Config.save_every` vòng (mặc định 2000) — xem mục 5 |

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

Cộng thêm `densify_until_iter = 15000` và `position_lr_max_steps = 30000` (`arguments/__init__.py:79,90`), hệ quả:

- **Toàn bộ chi phí nằm ở 0–15k** (densify + Adam đầy đủ + số Gaussian đang tăng).
- **15k–30k gần như miễn phí** nhưng vẫn chạy `final_prune_fastgs` (mỗi 3000 vòng, xem `pipeline/trainer.py:125-128`) → ít Gaussian hơn, LPIPS tốt hơn, file `.ply` nhỏ hơn.

> **Kết luận: đừng hạ `Config.iterations` xuống 15k khi muốn kết quả đầy đủ.** Tiết kiệm rất ít thời gian mà mất luôn phần đánh bóng + tỉa gần-như-cho-không. `Config.iterations` mặc định là **7000** (đủ cho Run all nhanh, xem mục 8) — nâng lên **30000** để lấy đúng phần thưởng của lịch optimizer. Nếu rút ngắn, sàn hợp lý là **20000**; ngưỡng 15000/20000 trong `optimizer_step` là hằng số cứng trong mã nguồn (không phải tham số của `pipeline/`), muốn đồng bộ với một ngân sách khác thì phải sửa trực tiếp `scene/gaussian_model.py`.

### 2.2 — `--antialiasing` là cờ chết trong fork này

`arguments/__init__.py:70` có `self.antialiasing = False`, nhưng:

- `gaussian_renderer/__init__.py` dựng `GaussianRasterizationSettings(...)` **không có** trường `antialiasing`;
- `submodules/diff-gaussian-rasterization_fastgs/` **không tham chiếu** chữ `antialiasing` ở bất kỳ đâu.

Truyền cờ này chỉ bị bỏ qua âm thầm — không lỗi, không tác dụng. Đây là di sản từ code INRIA gốc. `pipeline/trainer.py::build_args` không thêm cờ này. Muốn có chống răng cưa thật phải vá kernel CUDA (mục 7).

### 2.3 — Bước chấm điểm đa góc nhìn đắt hơn vẻ ngoài

`utils/fast_utils.py:45` `compute_gaussian_score_fastgs` lấy mẫu **10 camera** (`num_cams = 10`, dòng 13) và render **hai lượt** mỗi camera (một lượt thường, một lượt `get_flag=True` với `metric_map`). Nó chạy mỗi `densification_interval` vòng trong khoảng 500–15000 (gọi từ `pipeline/trainer.py:114-116`).

| `densification_interval` | Số lần gọi trong 0–15k |
|---|---|
| `100` (mặc định của `arguments/__init__.py:87`) | ~145 lần |
| `500` (giá trị `pipeline/config.py` đặt sẵn trong `train_extra_args`, khớp `train_base.sh`) | ~29 lần |

⇒ Đây là lever thời gian lớn thứ hai sau độ phân giải, và nó là **setting chính chủ**, không phải thoả hiệp. `Config()` mặc định đã dùng `500`, không cần tự thêm.

---

## 3. Điểm số theo dõi tiến trình

Công thức được cài trong [`pipeline/score.py`](../pipeline/score.py) (`composite_score`, `evaluate_cameras`) — một số duy nhất trong $[0,1]$, càng cao càng tốt:

$$\mathrm{Score}=0.4\,(1-\mathrm{LPIPS})+0.3\,\mathrm{SSIM}+0.3\,\widehat{\mathrm{PSNR}},
\qquad
\widehat{\mathrm{PSNR}}=\operatorname{clamp}\!\Big(\tfrac{\mathrm{PSNR}}{\mathrm{PSNR}_{\max}},\,0,\,1\Big)$$

| Ký hiệu | Ý nghĩa | Miền | Đóng góp |
|---|---|---|---|
| $\mathrm{LPIPS}$ | khoảng cách tri giác (VGG hoặc AlexNet); thấp = giống | $[0,1]$ | $0.4\,(1-\mathrm{LPIPS})$ |
| $\mathrm{SSIM}$ | tương đồng cấu trúc; cao = giống | $[0,1]$ | $0.3\,\mathrm{SSIM}$ |
| $\widehat{\mathrm{PSNR}}$ | `torch.clamp(psnr_val / psnr_max, 0.0, 1.0)` (`pipeline/score.py:14`) | $[0,1]$ | $0.3\,\widehat{\mathrm{PSNR}}$ |
| $\mathrm{PSNR}_{\max}$ | mốc bão hoà, `Config.psnr_max`, mặc định **30 dB** | hằng | — |

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

> Giữ `Config.psnr_max` **cố định** giữa các lần chạy. Hạ nó xuống để "làm đẹp số" là tự lừa.

### Đo trong lúc train, và đo lại khi nộp bài

Hai lần tính Score dùng **hai cấu hình LPIPS khác nhau**, đều nằm trong `Config`:

| Giai đoạn | Hàm | `net_type` | Vì sao |
|---|---|---|---|
| Trong lúc train, mỗi `Config.score_every` vòng (mặc định 1000) | `pipeline.score.evaluate_cameras` (gọi từ `pipeline/trainer.py:145`) | `Config.lpips_net_live = "alex"` | nhanh hơn ~3×, nạp ~30 MB thay vì ~500 MB — chỉ để xem xu hướng |
| Lúc render submission | `pipeline.submission.render_scene` (`pipeline/submission.py:62`) | `Config.lpips_net_report = "vgg"` | số liệu chính thức, chậm hơn nhưng khớp cách ban tổ chức chấm |

Thanh tiến trình (`tqdm`) hiện `loss`, số Gaussian và Score/PSNR mới nhất mỗi 10 vòng; mỗi mốc `score_every`, `train_scene` in một dòng đầy đủ gồm $\Delta$Score so với mốc trước (`pipeline/trainer.py:158-167`) và ghi vào `history` (đổ ra `history.csv` qua `pipeline.report.history_frame`). Chuỗi $\Delta$ dương và co dần về 0 = hội tụ lành mạnh; $\Delta$ âm kéo dài = overfit hoặc prune quá tay. Hai giá trị Score (live vs. report) lệch nhẹ là bình thường.

---

## 4. Preset A — cân bằng (đã là mặc định của `Config`)

`pipeline/config.py::Config.train_extra_args` đã đặt sẵn:

```python
[
    "--densification_interval", "500",
    "--lambda_dssim", "0.25",
    "--highfeature_lr", "0.02",
    "--loss_thresh", "0.07",
    "--grad_abs_thresh", "0.0012",
]
```

`pipeline/trainer.py::build_args` ghép thêm `-r <Config.resolution>`, `--eval`, `--iterations <Config.iterations>`, `--mult <Config.mult>` và (nếu `Config.white_background`) `-w`, rồi gọi `ArgumentParser` gốc của 3DGS (`ModelParams`/`OptimizationParams`/`PipelineParams`).

| Tham số | Mặc định gốc → dùng trong `Config` | Lý do |
|---|---|---|
| `resolution` (`-r`) | `-1` → **`2`** (trong nhà) / **`4`** (ngoài trời, tự sửa trong ô cấu hình) | đây là **độ phân giải eval chuẩn** của MipNeRF360 (`images_2`/`images_4`), không phải hy sinh. Lever thời gian lớn nhất: ×2–3 |
| `--densification_interval` | `100` → **`500`** | mục 2.3; giá trị `train_base.sh` thật sự dùng |
| `iterations` | — | giữ **`30000`** cho kết quả đầy đủ (mục 2.1); `Config` mặc định `7000` để Run-all nhanh (mục 8) |
| `--lambda_dssim` | `0.2` → **`0.25`** | nâng trọng số $(1-\mathrm{SSIM})$ trong loss ⇒ SSIM và thường cả LPIPS |
| `--highfeature_lr` | `0.005` → **`0.02`** | mọi cảnh indoor trong `train_base.sh` dùng 0.02; SH bậc cao hội tụ nhanh hơn. Lưu ý `gaussian_model.py:205` chia thêm cho 20 ⇒ lr hiệu dụng `0.02/20 = 0.001` |
| `--loss_thresh` | `0.1` → **`0.07`** | đánh dấu nhiều pixel lỗi hơn ⇒ giữ chi tiết (cảnh `garden` dùng 0.06) |
| `--grad_abs_thresh` | `0.0012` | núm chính theo cảnh: thấp hơn (0.0008–0.0010) ⇒ nét hơn nhưng nhiều Gaussian; cao hơn (0.0015–0.002) cho cảnh ngoài trời/thưa |
| `--dense` | `0.001` (indoor) → **0.004–0.01** cho ngoài trời | tỉ lệ clone/split đúng loại cảnh — không có trong `train_extra_args` mặc định, tự thêm vào `Config.train_extra_args` nếu cần |
| ~~`--antialiasing`~~ | **không dùng** | mục 2.2 |

`Config.mult` giữ mặc định `0.5`; dùng `0.7` cho cảnh nền phức tạp (Tanks&Temples, Deep Blending). `build_args` dùng **cùng một `cfg.mult`** cho cả train (`pipeline/trainer.py:94`) và render submission (`pipeline/submission.py:52`), nên không có nguy cơ lệch giữa hai bước như khi gọi `train.py`/`render.py` tách rời bằng tay.

> Cách chọn `grad_abs_thresh` / `dense` theo cảnh: mở `train_base.sh` ở gốc repo, tìm dòng của cảnh giống dataset của bạn nhất, rồi thêm các cờ đó vào `Config.train_extra_args`.

---

## 5. Chống tràn RAM trên Colab

Bẫy kinh điển: train + render xong, kernel vẫn giữ tensor lớn, cache CUDA — RAM ~10–12 GB. Nén/tải về lúc đó → thêm bản copy trong RAM → **OOM, kernel chết**. `pipeline/` xử lý việc này ở nhiều lớp, tất cả nằm sẵn trong hàm — không cần tự viết code dọn RAM trong notebook nữa.

| Khi nào | Việc làm | Ở đâu |
|---|---|---|
| Mỗi `Config.save_every` vòng (mặc định 2000, đặt `0` để tắt) | `_save_checkpoint` ghi `.ply` ra `point_cloud/iteration_<n>/` rồi **xoá checkpoint giữa chừng trước đó** nếu `Config.keep_last_checkpoint = True` — đĩa chỉ giữ một bản trung gian | `pipeline/trainer.py:39-47,186-189` |
| Cuối vòng lặp train của một scene | `_save_checkpoint(scene_obj, iterations, ...)` ghi bản cuối và dọn nốt checkpoint trung gian | `pipeline/trainer.py:194` |
| Mỗi vòng lặp | `del pkg, image, gt, viewspace, visibility, radii, loss, ll1, ssim_value` | `pipeline/trainer.py:191` |
| Mỗi `Config.score_every` vòng | `torch.cuda.empty_cache()`; nếu RAM > `Config.ram_soft_limit_gb` (mặc định 10.5 GB) thì thêm `gc.collect()` | `pipeline/trainer.py:156,181-183` |
| Ngay sau `train_scene` (mặc định `keep_model=False`) | `del gaussians, scene_obj; gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()` | `pipeline/trainer.py:207-212` |
| Sau khi render xong mỗi scene, trước khi qua scene kế | `pipeline.env.free_memory()` — cùng bốn bước dọn dẹp trên | `pipeline/run.py:61` |
| Render + chấm điểm submission | `pipeline.submission.render_scene` chạy **trong cùng tiến trình kernel** (không còn gọi `render.py`/`metrics.py` như tiến trình con); VGG-LPIPS (~500 MB) được nạp vào kernel lúc này | `pipeline/submission.py:21-89` |
| Đóng gói | `zipfile.ZIP_STORED` ghi **thẳng ra đĩa**, không qua `BytesIO`/`shutil.make_archive` trong RAM | `pipeline/submission.py::build_zip`, `pipeline/deliver.py::pack_models` |
| Tải về | `pipeline.deliver.download` gọi `google.colab.files.download`; đặt `Config.save_to_drive = True` để copy sang Drive trước khi tải (hữu ích khi file lớn) | `pipeline/deliver.py:36-71` |

Quy tắc bao trùm: **`.ply` được ghi ra đĩa trước khi kernel dọn `gaussians`/`scene`, và mọi bước nén/tải luôn ghi ra đĩa chứ không giữ bản nén trong RAM.**

**Khác với thiết kế cũ:** trước đây `render.py`/`metrics.py` được gọi bằng `!python ...` (tiến trình con) để RAM/VRAM được hệ điều hành thu hồi ngay khi thoát. Bản `pipeline/` hiện tại **render trong cùng kernel** (gọi thẳng `render_fastgs`, `GaussianModel`, `Scene`) để tái dùng code và log Score thống nhất — đổi lại, VGG-LPIPS nạp vào RAM/VRAM của kernel trong lúc render mỗi scene, rồi được giải phóng bởi `free_memory()` ngay sau đó (dòng trên).

**Mất phiên giữa chừng thì sao?** `run_all` (`pipeline/run.py:51-66`) train-rồi-render-rồi-giải phóng **từng scene một**, nên mọi scene đã xong đều an toàn trên đĩa. Với scene đang train dở, `Config.save_every` (mặc định 2000 vòng) để lại checkpoint gần nhất trong `output/<scene>/point_cloud/iteration_<n>/`. Muốn dùng nó ngay mà không train lại:

```python
sub = submission.render_scene(cfg, "<scene>", iterations=<n>)   # <n> = mốc checkpoint còn lại
```

Điểm sẽ thấp hơn bản train đủ vòng, nhưng vẫn có ảnh hợp lệ cho scene đó. Lưu ý `save_every` **không** phải checkpoint để train tiếp: chỉ `.ply` được ghi, trạng thái Adam và các bộ đếm densify thì không — chạy lại là train từ đầu. Đặt `save_every=0` nếu muốn tắt hẳn (chỉ lưu ở vòng cuối), hoặc `keep_last_checkpoint=False` nếu muốn giữ mọi mốc để so sánh chất lượng theo iteration.

---

## 6. Chưa wire — cần đụng CUDA (không nằm trong `pipeline/`)

Các ý tưởng dưới đây **không có trong mã nguồn hiện tại** (không trong `pipeline/`, không trong `scene/gaussian_model.py`, không trong `arguments/__init__.py`) — liệt kê để tham khảo hướng phát triển tiếp theo, không phải mô tả tính năng đang chạy.

| Hạng mục | Công thức / việc phải làm | Cần sửa CUDA? | Ưu tiên |
|---|---|---|---|
| **Coarse-to-fine theo lịch** (ý tưởng DashGaussian) | đổi `resolution_scales` của `Scene` theo lịch $r(t)$ giảm dần từ 4.0 → 1.0 khi $t$ tăng, reset `max_radii2D` mỗi lần đổi scale | không — `Scene(resolution_scales=[...])` đã có sẵn trong repo, chỉ cần code điều phối trong `pipeline/trainer.py` | trung bình |
| **Ngân sách primitive** (Taming-3DGS / 3DGS-MCMC) | thêm trần tuyệt đối $B(t)$ theo lịch, sau mỗi lần densify cắt bớt Gaussian opacity thấp nếu $n > B(t)$ | không — có thể viết thuần Python đè lên `densify_and_prune_fastgs` | trung bình, giúp tránh OOM có kiểm soát trên T4 |
| **SGLD noise + phạt L1** (3DGS-MCMC) | thêm nhiễu tỉ lệ theo scale mỗi Gaussian vào vị trí, cộng phạt $\lambda_o\overline{o}+\lambda_s\overline{s}$ vào loss | không — thuần Python trong vòng lặp train | thấp |
| **Mip-Splatting** | $\alpha_i = o_i\sqrt{\frac{\lvert\Sigma'_i\rvert}{\lvert\Sigma'_i+sI\rvert}}\exp(-\tfrac12\Delta^\top(\Sigma'_i+sI)^{-1}\Delta)$, $s\approx0.1$ px², thay cho $\Sigma'\leftarrow\Sigma'+0.3I$; kèm lọc 3D $\Sigma\leftarrow\Sigma+\hat s^{-2}I$ với $\hat s=\max_k f_k/d_k$ | **có** — sửa `forward.cu`/`backward.cu`, thêm trường vào `GaussianRasterizationSettings`, cài lại submodule | 1 — nhất là cho ảnh drone (tỉ lệ biến thiên lớn trong 1 khung hình) |
| **StopThePop** | khoá sắp xếp = độ sâu tại điểm đóng góp cực đại dọc tia: $t^\*=\arg\max_t \mathcal{G}(\mathbf{o}+t\mathbf{d})$ | **có** — thay submodule rasterizer, giữ nguyên Python | 2 |
| **2DGS / GOF** | đổi primitive sang đĩa phẳng, giao tia chính xác: $\mathbf{x}(u,v)=\mathbf{p}_c+s_u u\mathbf{t}_u+s_v v\mathbf{t}_v$, $\mathcal{G}=\exp(-\tfrac{u^2+v^2}{2})$ | **có** — cần `diff-surfel-rasterization` + 2 regularizer | cao, nếu mục tiêu là **mesh/bề mặt** |
| **3DGS-MCMC đầy đủ** | thiếu mảnh di dời bảo toàn ảnh: $o_{\text{new}}=1-(1-o_{\text{old}})^{1/N}$ + hiệu chỉnh $\Sigma$ khớp mô-men bậc 2 | tuỳ — đường ngắn hơn là chuyển backend sang `gsplat` (có sẵn strategy MCMC + antialiasing) | 3 |
| **Khởi tạo MASt3R/DUSt3R** | viết `sceneLoadTypeCallbacks["Dust3r"]` trả `BasicPointCloud` từ pointmap dày $X\in\mathbb{R}^{H\times W\times3}$ | không (Python + model có sẵn) | cao, nếu ảnh trên không / ít chồng lấp |

> 3DGS và FastGS tối ưu cho **ảnh mới**, không cho **hình học**. Digital twin cần mesh thì nhánh đúng là 2DGS/GOF.

**Đã gỡ bỏ khỏi notebook:** bản trước đây có một ô chạy "ablation" so sánh 3 cấu hình (`presetA` / `coarse2fine` / `c2f_budget_mcmc`) xuất `ablation.csv` — ô này **không còn tồn tại**; các cờ `coarse2fine`, `gaussian_budget`, MCMC noise ở trên chưa được cài vào `pipeline/trainer.py`. Notebook hiện tại (9 ô, mục 8) không có công tắc `RUN_ABLATION`.

---

## 7. Thứ tự chạy notebook

Notebook chạy được bằng **Runtime → Run all** trên T4 free mà không cần sửa gì (dùng bộ dữ liệu demo `tandt/truck`). Notebook có **9 ô code, đánh số 0–7** (mục 6 gồm 2 ô code):

| Ô code | Mục markdown | Việc | Hàm/module gọi |
|---|---|---|---|
| 0 | 0 — Get the code | clone repo (nếu chưa có), `%cd`, thêm vào `sys.path` | — |
| 1 | 1 — Check the GPU | cài phụ thuộc (pip + 3 submodule CUDA, bỏ qua nếu đã cài) rồi in GPU/VRAM/torch/CUDA/RAM | `pipeline.env.install_dependencies`, `pipeline.env.check_gpu` |
| 2 | 2 — Configuration | **ô duy nhất cần sửa**: đường dẫn, độ phân giải, số vòng, các mốc | tạo `pipeline.Config(...)`, `cfg.show()` |
| 3 | 3 — Load the dataset and profile it | tải dữ liệu nếu cần, liệt kê scene, in bảng hồ sơ (số ảnh, kích thước, train/test split) | `pipeline.run.load_data` |
| 4 | 4 — Quick test | train thử vài trăm vòng (`Config.smoke_iterations`, mặc định 300) trên 1 scene để chắc cả pipeline chạy được trước khi tốn hàng giờ | `pipeline.run.smoke_test` |
| 5 | 5 — Train | train + render lần lượt từng scene, giải phóng bộ nhớ giữa các scene | `pipeline.run.run_all` |
| 6 | 6 — Data analytics | bảng leaderboard + 2 biểu đồ (`training.png`, `leaderboard.png`), rồi ảnh mẫu render-vs-GT | `pipeline.run.analytics`, `pipeline.report.show_samples` |
| 7 | 7 — Build `submission.zip` | nén + kiểm tra định dạng (tên scene, tên file liên tục `0001.png`...) + tải mô hình/submission về máy | `pipeline.run.finish` |

**Không cần sửa gì để chạy thử lần đầu.** Khi dùng dữ liệu của mình, sửa trong ô 2: đặt `dataset_url=None` (nếu tự upload ảnh vào `data_root`) hoặc trỏ `dataset_url` tới ZIP của bạn, và `resolution` (`2` trong nhà / `4` ngoài trời). `scenes=()` để pipeline tự tìm mọi scene có `sparse/` (COLMAP) hoặc `transforms_train.json` (Blender) trong `data_root`.

Quy ước code trong notebook: **toàn bộ code là tiếng Anh**; giải thích nằm ở ô markdown (cũng tiếng Anh) phía trên mỗi ô code.

**Đã gỡ bỏ khỏi notebook so với bản trước:** không còn ô dọn RAM thủ công (nay nằm trong `pipeline/env.py::free_memory`, gọi tự động — mục 5), không còn ô mô phỏng NumPy thuần Python cho từng cơ chế (chuyển sang [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py), chạy độc lập bằng `python demos/fastgs_mechanisms.py`), và không còn ô roadmap CUDA (nội dung đó chuyển hẳn vào tài liệu này, mục 6).

---

## 8. Điều chưa kiểm chứng

Tài liệu này được viết dựa trên việc đọc mã nguồn (`pipeline/*.py`, `train.py`, `render.py`, `metrics.py`, `scene/gaussian_model.py`, `scene/__init__.py`, `utils/camera_utils.py`, `utils/fast_utils.py`, `gaussian_renderer/__init__.py`), **chưa chạy thử toàn bộ notebook trên GPU thật**.

Trước khi dựa vào số liệu: chạy ô 4 (Quick test) trước để chắc dữ liệu + CUDA + chấm điểm đều chạy được, rồi mới nâng `Config.iterations` lên 30000 và chạy ô 5 (Train) đầy đủ.

Các con số trong tài liệu này là **suy luận từ mã nguồn và tham số của `train_base.sh`**, không phải kết quả đo trên máy bạn. Bảng `leaderboard.csv`/`history.csv` của chính bạn (mục 7, ô 6) mới là bằng chứng.
