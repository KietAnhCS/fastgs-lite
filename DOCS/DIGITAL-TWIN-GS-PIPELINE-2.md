# DIGITAL TWIN GS PIPELINE (2/3) — Một vòng lặp train: render, loss, backward, densify

Tài liệu này mổ xẻ đúng một vòng lặp `for iteration in range(...)` của fastgs-lite: từ lúc bốc camera, gọi rasterizer CUDA, tính loss, `loss.backward()`, đến các bước điều khiển mật độ Gaussian (densify/prune/reset) và bước `optimizer.step()`. Toàn bộ nội dung được đối chiếu trực tiếp với `train.py` (đường dòng lệnh) và `pipeline/trainer.py::train_scene()` (đường notebook), không mang lại bất kỳ khái niệm nào từ bản tài liệu cũ (mask SAM2, DUSt3R, `pose_optimizer`, `compute_combined_loss`... — những thứ đó **không tồn tại** trong mã nguồn hiện tại).

## MỤC LỤC

**PHẦN IV — MỘT VÒNG LẶP TRAIN**
- §19. Khung một vòng lặp — hai bản cài đặt
- §20. `update_learning_rate` và `oneupSHdegree`
- §21. Bốc camera khỏi `viewpoint_stack`
- §22. `render_fastgs` — cấu hình rasterizer và compact box `--mult`
- §23. Gọi rasterizer CUDA
- §24. `l1_loss` và `fused_ssim`
- §25. Loss tổng và `lambda_dssim`

**PHẦN V — BACKWARD VÀ KIỂM SOÁT MẬT ĐỘ** *(agent khác viết, chỉ liệt kê ở đây)*
- §26. `loss.backward()` — gradient chảy đi đâu
- §27. `add_densification_stats`
- §28. `sampling_cameras` + `compute_gaussian_score_fastgs`
- §29. `densify_and_prune_fastgs` — điều kiện kép, clone vs split
- §30. Ba tầng pruning + `final_prune_fastgs`
- §31. Phẫu thuật trạng thái Adam
- §32. `reset_opacity`
- §33. `optimizer_step`, lịch learning rate và lưu `.ply`

---

# PHẦN IV — MỘT VÒNG LẶP TRAIN

## 19. Khung một vòng lặp — hai bản cài đặt

Cùng một thuật toán tồn tại ở hai nơi: `train.py::training()` (chạy từ CLI, dòng 34) và `pipeline/trainer.py::train_scene()` (dòng 46, dùng trong notebook Colab). `train_scene` **gọi cùng các hàm lõi** (`render_fastgs`, `l1_loss`, `fast_ssim`, `add_densification_stats`, `compute_gaussian_score_fastgs`, `densify_and_prune_fastgs`, `final_prune_fastgs`, `optimizer_step`) theo đúng thứ tự và đúng ngưỡng số như `train.py`, chỉ khác phần vỏ bọc theo dõi/lưu.

Bảng đối chiếu từng bước trong một iteration (file:line là vị trí bắt đầu của khối lệnh tương ứng):

| Bước trong 1 iteration | `train.py::training()` | `pipeline/trainer.py::train_scene()` |
|---|---|---|
| Cầu nối viewer qua websocket (nếu bật `--websockets`) | `train.py:67-73` | **không có** |
| `update_learning_rate(iteration)` | `train.py:78` | `pipeline/trainer.py:91` |
| `oneupSHdegree()` mỗi 1000 iter | `train.py:81-82` | `pipeline/trainer.py:92-93` |
| Bốc camera khỏi `viewpoint_stack`/`viewpoint_indices` | `train.py:86-91` | `pipeline/trainer.py:95-100` |
| Bật `pipe.debug` tại `--debug_from` | `train.py:94-95` | **không có** (không nhận `debug_from`) |
| `render_fastgs(...)` | `train.py:97-98` | `pipeline/trainer.py:104` |
| `l1_loss` + `fast_ssim` + loss tổng | `train.py:101-103` | `pipeline/trainer.py:108-110` |
| `loss.backward()` | `train.py:104` | `pipeline/trainer.py:111` |
| Cập nhật thanh tiến trình (EMA loss) | `train.py:110-114` (tqdm, mỗi 10 iter) | `pipeline/trainer.py:114`, `146-149` (tqdm + hậu tố Score/PSNR mỗi 10 iter) |
| Lưu checkpoint theo lịch | `train.py:120-122` (`saving_iterations`, gọi `scene.save`) | `pipeline/trainer.py:154-156` (`cfg.save_every`, gọi `_save_checkpoint` — lưu `.ply` rồi xoá checkpoint cũ) |
| Densify: `max_radii2D` + `add_densification_stats` | `train.py:129-131` | `pipeline/trainer.py:118-120` |
| `densify_and_prune_fastgs` mỗi `densification_interval` | `train.py:133-142` | `pipeline/trainer.py:121-128` |
| `reset_opacity` mỗi `opacity_reset_interval` | `train.py:144-145` | `pipeline/trainer.py:129-130` |
| Pruning đa góc nhìn 3 tầng (`final_prune_fastgs`, 15k–30k, mỗi 3000 iter) | `train.py:148-156` | `pipeline/trainer.py:132-135` |
| `optimizer_step` | `train.py:159-160` | `pipeline/trainer.py:137-141` |
| Đo thời gian bằng CUDA event | `train.py:53-54, 108, 165-169` | **không có** — dùng `time.time()` tổng thể (`pipeline/trainer.py:80, 165`) |

Những gì **`train_scene` thêm** so với `train.py`:
- Chấm điểm trực tuyến định kỳ mỗi `score_every` iteration qua `evaluate_cameras` (đọc PSNR/SSIM/LPIPS/Score chuẩn hoá trên tập `holdout`), có in delta so với lần chấm trước (`d_score`, `d_psnr`, …) — `pipeline/trainer.py:146-168`.
- Hậu tố tiến trình tqdm hiển thị % hoàn thành, số Gaussian, loss, Score, PSNR ngay trên thanh (`pipeline/trainer.py:146-149`).
- Checkpoint định kỳ theo `cfg.save_every`, tự xoá checkpoint giữa chừng trước đó để tiết kiệm đĩa Colab (`_save_checkpoint`, `pipeline/trainer.py:41-47`).
- Giới hạn RAM mềm: nếu `usage["ram_used_gb"] > cfg.ram_soft_limit_gb` thì gọi `gc.collect()` + `torch.cuda.empty_cache()` (`pipeline/trainer.py:169-171`).
- Gọi `safe_state(True)` — tức luôn chạy silent/quiet (`pipeline/trainer.py:59`), khác `train.py` truyền `args.quiet` từ CLI (`train.py:266`).
- Dọn bộ nhớ tường minh mỗi iteration (`del pkg, image, gt, viewspace, visibility, radii, loss, ll1, ssim_value` — `pipeline/trainer.py:172`) và dọn toàn bộ model sau khi train nếu `keep_model=False` (`pipeline/trainer.py:182-188`).

Những gì **`train_scene` bỏ** so với `train.py` (đã kiểm tra từng cái trong `train.py`):
- Cầu nối viewer qua websocket: `train.py` import `network_gui_ws` và có khối `if websockets: ...` gửi ảnh render trực tiếp cho client xem trực tiếp (`train.py:16, 67-73, 261-263, 273`). `train_scene` không import `gaussian_renderer.network_gui_ws`, không có tham số nào tương đương — xác nhận đã bỏ.
- Tensorboard: `train.py` có `TENSORBOARD_FOUND`, `SummaryWriter`, hàm `training_report()` ghi scalar/ảnh lên TB (`train.py:28-31, 200, 209-243`) — nhưng lưu ý dòng gọi thực tế `training_report(...)` trong vòng lặp chính đã bị **comment out** (`train.py:118`), nên ngay trong `train.py` nó cũng không chạy khi train thường. `train_scene` không có bất kỳ đoạn nào liên quan TensorBoard.
- `--checkpoint_iterations` / `checkpoint` / `--start_checkpoint`: `train.py` nhận `checkpoint_iterations` và `checkpoint` làm tham số của `training()` nhưng bên trong hàm chỉ dùng `checkpoint` để `torch.load` model phục hồi (`train.py:36-39`) — **không có đoạn nào trong `training()` lưu theo `checkpoint_iterations`** (chỉ có `saving_iterations` được dùng để `scene.save`). `train_scene` không nhận và không dùng khái niệm `checkpoint_iterations`/`start_checkpoint` nào cả — nó tự quản lý qua `cfg.save_every`.
- `--debug_from`: `train.py` bật `pipe.debug = True` khi `(iteration - 1) == debug_from` (`train.py:94-95`), dùng để dump snapshot lỗi CUDA. `train_scene` không nhận `debug_from`, `pipe.debug` giữ nguyên giá trị mặc định `False` suốt quá trình train qua notebook.
- `torch.autograd.set_detect_anomaly(args.detect_anomaly)`: chỉ có ở khối `if __name__ == "__main__"` của `train.py:269`, không có trong `pipeline/trainer.py`.

## 20. `update_learning_rate` và `oneupSHdegree`

`GaussianModel.training_setup(training_args)` (`scene/gaussian_model.py:162-180`) tạo 5 param-group cho optimizer chính (`self.optimizer`), mỗi nhóm có `lr` riêng cố định trừ `xyz`:

| Param group | `lr` khởi tạo | Có được `update_learning_rate` cập nhật mỗi iteration không |
|---|---|---|
| `xyz` | `position_lr_init * spatial_lr_scale` | **Có** — theo lịch exponential (`get_expon_lr_func`) |
| `f_dc` | `lowfeature_lr` | Không — cố định suốt training |
| `opacity` | `opacity_lr` | Không |
| `scaling` | `scaling_lr` | Không |
| `rotation` | `rotation_lr` | Không |
| `f_rest` (trong `shoptimizer` riêng) | `highfeature_lr / 20.0` | Không |

`self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)` truyền `lr=0.0` làm giá trị mặc định của `Adam`, nhưng từng phần tử trong `l` đã tự mang khoá `'lr'` riêng nên override giá trị này — `lr=0.0` thực chất không bao giờ được dùng.

`update_learning_rate(self, iteration)` (`scene/gaussian_model.py:182-188`):
```python
for param_group in self.optimizer.param_groups:
    if param_group["name"] == "xyz":
        lr = self.xyz_scheduler_args(iteration)
        param_group['lr'] = lr
        return lr
```
Nó chỉ tìm đúng group tên `"xyz"`, gán `lr` mới rồi `return lr` ngay lập tức (bỏ qua các group còn lại trong vòng `for` — không cần `break` vì đã `return`). Giá trị trả về không được `train.py`/`train_scene` sử dụng (lời gọi `gaussians.update_learning_rate(iteration)` không gán biến nào) — hàm được gọi vì tác dụng phụ (side-effect) lên `param_group['lr']`.

`xyz_scheduler_args` được tạo bởi `get_expon_lr_func(lr_init=position_lr_init*spatial_lr_scale, lr_final=position_lr_final*spatial_lr_scale, max_steps=position_lr_max_steps)` (`scene/gaussian_model.py:178-180`) — lịch suy giảm mũ (log-linear interpolation giữa `lr_init` và `lr_final` theo `iteration/max_steps`, có delay mult ở giai đoạn đầu).

`oneupSHdegree()` (`scene/gaussian_model.py:133-135`):
```python
def oneupSHdegree(self):
    if self.active_sh_degree < self.max_sh_degree:
        self.active_sh_degree += 1
```
Cả hai vòng lặp gọi hàm này mỗi 1000 iteration (`if iteration % 1000 == 0`) — mỗi lần tăng bậc SH (spherical harmonics) lên 1, cho tới khi chạm `max_sh_degree` (= `dataset.sh_degree`, mặc định 3). `active_sh_degree` là bậc SH thực sự được rasterizer dùng (`raster_settings.sh_degree=pc.active_sh_degree`, §22) — tăng dần độ chi tiết màu theo góc nhìn khi training tiến triển, tránh học nhiễu bậc cao ngay từ đầu khi hình học còn chưa ổn định.

## 21. Bốc camera khỏi `viewpoint_stack`

Cả hai bản đều dùng cùng một mẫu lấy-không-hoàn-lại (sampling without replacement) mỗi iteration, thay vì đơn giản `random.choice`:

```python
if not viewpoint_stack:
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
rand_idx = randint(0, len(viewpoint_indices) - 1)
viewpoint_cam = viewpoint_stack.pop(rand_idx)
_ = viewpoint_indices.pop(rand_idx)
```
(`train.py:85-91`; tương đương ở `pipeline/trainer.py:95-100` với tên biến `stack`/`indices`/`pick`/`cam`).

`viewpoint_stack` là danh sách camera train, bị `pop` dần mỗi iteration cho tới khi rỗng thì nạp lại toàn bộ (`scene.getTrainCameras().copy()`) — đảm bảo mỗi camera được thấy đúng một lần mỗi "epoch" (một vòng qua hết N camera) trước khi lặp lại, thay vì có thể bốc trùng liên tục.

`viewpoint_indices` là một danh sách chỉ số song song (`0..N-1`), bị `pop` cùng vị trí `rand_idx` với `viewpoint_stack`. Bản thân giá trị `_ = viewpoint_indices.pop(rand_idx)` không được dùng ở đâu khác trong vòng lặp — nó tồn tại chỉ để giữ độ dài `len(viewpoint_indices)` luôn đồng bộ với `len(viewpoint_stack)`, làm biên trên cho `randint(0, len(viewpoint_indices) - 1)`. Về mặt logic, `randint(0, len(viewpoint_stack) - 1)` sẽ cho kết quả y hệt — việc giữ danh sách song song là di sản viết code (có thể để tiện debug chỉ số gốc), không mang thêm thông tin nào khác trong bản hiện tại.

## 22. `render_fastgs` — cấu hình rasterizer và compact box `--mult`

Đọc `gaussian_renderer/__init__.py:18-110` theo đúng thứ tự:

**Chữ ký hàm** (dòng 18):
```python
def render_fastgs(viewpoint_camera, pc: GaussianModel, pipe, bg_color, mult,
                   scaling_modifier=1.0, override_color=None, get_flag=None, metric_map=None)
```
Vòng lặp train luôn gọi `render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)` — không truyền `get_flag`/`metric_map` (dùng giá trị mặc định `None`); hai tham số này chỉ được truyền tường minh bởi `compute_gaussian_score_fastgs` khi cần thu thập điểm số fastgs-lite (xem §28).

**`screenspace_points`** (dòng 27):
```python
screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
```
Đây là tensor "giả" toàn số 0, shape `(N, 4)` (chú thích ở dòng 26 còn ghi shape cũ `(N,3)` — dấu vết code cũ, không khớp dòng thực thi ngay dưới nó dùng shape `(N,4)`). Vì `requires_grad=True` và được truyền làm `means2D` vào rasterizer, PyTorch autograd sẽ gán vào `.grad` của nó đúng gradient của loss theo toạ độ màn hình 2D (screen-space) của từng Gaussian — cách "mượn" gradient này (thay vì `pc.get_xyz` không có gradient màn-hình trực tiếp) là kỹ thuật chuẩn của 3DGS để lấy `viewspace_points.grad` phục vụ `add_densification_stats` (§27). Vì `screenspace_points` không phải leaf-tensor được optimizer theo dõi (không nằm trong `training_setup`), phải gọi `retain_grad()` (dòng 29, bọc trong `try/except` phòng trường hợp không cần gradient — ví dụ khi `torch.no_grad()`).

**`tanfovx`/`tanfovy`** (dòng 34-35): `tan(FoVx/2)`, `tan(FoVy/2)` — nửa góc nhìn ngang/dọc của camera, dùng để rasterizer chuyển toạ độ camera-space sang screen-space (tham số chuẩn của phép chiếu perspective).

**`metric_map`** (dòng 37-38): nếu không truyền, khởi tạo tensor 0 kiểu `int` dài `H*W` trên CUDA — buffer để rasterizer cộng dồn số liệu (đếm) phục vụ chấm điểm fastgs-lite khi `get_flag=True` (xem §28); trong render bình thường của vòng lặp train nó chỉ là buffer rỗng không dùng tới.

**`GaussianRasterizationSettings`** (dòng 40-56) — mọi field và nguồn gốc:

| Field | Giá trị | Ý nghĩa |
|---|---|---|
| `image_height`, `image_width` | từ `viewpoint_camera` | kích thước ảnh cần render |
| `tanfovx`, `tanfovy` | tính ở trên | góc nhìn |
| `bg` | `bg_color` (tham số hàm) | màu nền (đen hoặc trắng, hoặc ngẫu nhiên nếu `random_background`) |
| `scale_modifier` | `scaling_modifier` (mặc định 1.0) | hệ số nhân thêm lên scale Gaussian |
| `viewmatrix` | `viewpoint_camera.world_view_transform` | ma trận world→camera |
| `projmatrix` | `viewpoint_camera.full_proj_transform` | ma trận chiếu đầy đủ |
| `sh_degree` | `pc.active_sh_degree` | bậc SH hiện tại (xem §20) |
| `campos` | `viewpoint_camera.camera_center` | vị trí camera trong world-space |
| `mult` | tham số hàm `mult` (= `opt.mult`) | hệ số compact-box, xem dưới |
| `prefiltered` | `False` | luôn tắt (không lọc trước Gaussian ngoài frustum ở phía Python) |
| `debug` | `pipe.debug` | bật dump `snapshot_fw.dump`/`snapshot_bw.dump` khi lỗi CUDA (xem §19, `--debug_from`) |
| `get_flag` | tham số hàm (mặc định `None`) | cờ bật thu thập `metric_map` trong kernel CUDA |
| `metric_map` | tính ở trên hoặc truyền vào | buffer đếm cho chấm điểm fastgs-lite |

**Compact box — `mult`**: giá trị này đi thẳng vào `GaussianRasterizationSettings.mult` rồi xuống kernel CUDA `duplicateToTilesTouched` (`submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/auxiliary.h:318-358`). Kernel tính ngưỡng cắt hộp bao (bounding box) mỗi splat theo kiểu SNUGBOX: `t = 2*log(opacity*255)`, sau đó `t = mult * t` (dòng 337-338, biến `t` được chú thích là "beta in Compact Box"). `t` càng nhỏ (mult càng nhỏ) → hộp bao ellipse càng hẹp → splat chạm ít tile 16×16 hơn → ít công việc rasterize hơn (nhanh hơn) nhưng có nguy cơ cắt mất phần đuôi mờ của Gaussian nếu `mult` quá nhỏ. Giá trị mặc định `opt.mult = 0.5` (`arguments/__init__.py:96`), các preset lớn (`train_big.sh`, README) dùng `--mult 0.7` cho scene lớn/nhiều chi tiết. **`--mult` phải khớp giữa `train.py` và `render.py`** khi render lại sau train (ghi rõ trong `README.md:137`) vì nó ảnh hưởng trực tiếp đến hình dạng splat được rasterize.

**`compute_cov3D_python` / `convert_SHs_python`** (dòng 70-88, cả hai mặc định `False` trong `PipelineParams`, `arguments/__init__.py:67-68`):
- Nếu `pipe.compute_cov3D_python=True`: `cov3D_precomp = pc.get_covariance(scaling_modifier)` — hiệp phương sai 3D được tính sẵn ở phía Python (chậm hơn, dùng để debug/so sánh), rasterizer CUDA sẽ nhận `cov3D_precomp` thay vì tự tính từ `scales`/`rotations`.
- Nếu `False` (mặc định): `scales = pc.get_scaling`, `rotations = pc.get_rotation` được truyền thẳng, rasterizer CUDA tự dựng ma trận hiệp phương sai — nhanh hơn, đường đi mặc định của fastgs-lite.
- Nếu `pipe.convert_SHs_python=True`: SH được eval thành RGB ngay ở Python qua `eval_sh(...)` (`utils/sh_utils.py`), kết quả clamp `max(sh2rgb+0.5, 0)` rồi truyền làm `colors_precomp` — rasterizer không cần làm việc với hệ số SH nữa.
- Nếu `False` (mặc định): `dc, shs = pc.get_features_dc, pc.get_features_rest` được truyền thẳng, kernel CUDA tự eval SH→RGB theo từng tia nhìn — đây là đường mặc định, tách riêng `dc` (bậc 0, màu nền) khỏi `shs` (các bậc còn lại) vì fastgs-lite xử lý learning-rate hai nhóm này riêng (`lowfeature_lr` cho `f_dc`, `highfeature_lr` cho `f_rest`, xem §20).

## 23. Gọi rasterizer CUDA

`rasterizer(means3D=..., means2D=..., dc=..., shs=..., colors_precomp=..., opacities=..., scales=..., rotations=..., cov3D_precomp=...)` (dòng 93-102) trả về đúng 3 giá trị:
```python
rendered_image, radii, accum_metric_counts = rasterizer(...)
```
Xác nhận bằng chữ ký `forward` của `_RasterizeGaussians` trong submodule (`submodules/diff-gaussian-rasterization_fastgs/diff_gaussian_rasterization_fastgs/__init__.py:110`: `return color, radii, accum_metric_counts`). Khác với 3DGS gốc (chỉ trả `color, radii`), fastgs-lite **thêm** `accum_metric_counts` — buffer đếm tích luỹ theo `metric_map`, dùng cho chấm điểm/pruning đa góc nhìn (§28); trong render bình thường của vòng lặp train, `render_fastgs` trả nó ra trong dict (`"accum_metric_counts"`, dòng 109) nhưng `train.py`/`train_scene` **không đọc** giá trị này ở đường train chính — chỉ `compute_gaussian_score_fastgs` mới dùng.

`render_fastgs` build dict trả về (dòng 106-110):
```python
return {"render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "accum_metric_counts": accum_metric_counts}
```
Lưu ý: `visibility_filter` **không phải** boolean mask thuần như bản 3DGS gốc thường thấy — nó là `(radii > 0).nonzero()`, tức tensor chỉ số (index tensor, shape `(K, 1)`) của các Gaussian có `radii > 0`. Việc dùng để index (`gaussians.max_radii2D[visibility_filter]`, `add_densification_stats(..., visibility_filter)`) vẫn hoạt động đúng vì PyTorch fancy-indexing chấp nhận cả hai dạng cho các phép này, nhưng đây là điểm khác biệt cần lưu ý khi đọc code so với tài liệu 3DGS gốc.

`radii > 0` là điều kiện một Gaussian có "bán kính màn hình" dương — tức nó chạm ít nhất một pixel sau khi chiếu và cắt bởi compact box (§22); Gaussian bị frustum-cull hoặc quá nhỏ/mờ sẽ có `radii = 0` và bị loại khỏi các thống kê densify/prune.

**Forward pass tile-based (tóm tắt cơ chế CUDA, không phải mã Python)**: rasterizer chia ảnh thành các tile `16×16` pixel (`#define BLOCK_X 16`, `#define BLOCK_Y 16`, `cuda_rasterizer/config.h:16-17`). Với mỗi tile, các Gaussian chạm tile đó (xác định bởi `duplicateToTilesTouched`, §22) được sắp xếp theo độ sâu (depth, front-to-back) rồi blend tuần tự theo công thức alpha-blending chuẩn (`cuda_rasterizer/forward.cu`, vòng lặp quanh dòng 333-413):
```
alpha = min(0.99, opacity * exp(power))      # power: hàm mũ Gaussian 2D tại pixel
C    += color * alpha * T                     # cộng dồn màu, trọng số bởi độ trong suốt còn lại T
T    *= (1 - alpha)                           # cập nhật độ trong suốt còn lại cho lớp sau
```
Vòng lặp dừng sớm khi `T < 0.0001` (tile coi như đã bão hoà, các Gaussian xa hơn không còn đóng góp đáng kể) — đây là early-termination chuẩn của 3DGS, không phải cơ chế riêng của fastgs-lite.

## 24. `l1_loss` và `fused_ssim`

`utils/loss_utils.py:20-21`:
```python
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()
```
Trả về **một scalar** — trung bình trị tuyệt đối sai khác trên toàn bộ tensor (mọi pixel, mọi kênh màu), không phải per-pixel hay per-channel.

File này còn định nghĩa `ssim(img1, img2, window_size=11, size_average=True)` (dòng 36-44) — SSIM cổ điển cài bằng `conv2d` với cửa sổ Gaussian 11×11 tự viết bằng PyTorch thuần (không CUDA). Với `size_average=True` (mặc định) nó trả về **một scalar** (`ssim_map.mean()`); nếu `False` trả về vector theo batch.

Vòng lặp train (cả `train.py` và `train_scene`) **không dùng** `ssim` này — nó import `fused_ssim as fast_ssim` từ package `fused_ssim` (một CUDA submodule ngoài, biên dịch riêng), gọi `fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))`. Lý do: `fused_ssim` chạy trên GPU bằng kernel CUDA fused (gộp nhiều phép convolution/elementwise vào một kernel) nên nhanh hơn đáng kể so với `ssim` cài bằng `F.conv2d` tuần tự — quan trọng vì SSIM được tính lại mỗi iteration training.

Ngược lại, `metrics.py` (script đánh giá sau khi train xong, chạy một lần trên toàn bộ test set chứ không phải mỗi iteration) dùng `from utils.loss_utils import ssim` (`metrics.py:17`, gọi ở dòng 72) — bản Python thuần, không cần biên dịch CUDA riêng, đơn giản và dễ tái lập cho việc đánh giá cuối cùng, tốc độ không phải ưu tiên ở đây.

Tóm lại:

| Nơi dùng | Hàm | Cài đặt |
|---|---|---|
| Vòng lặp train (`train.py`, `pipeline/trainer.py`) | `fast_ssim` (alias của `fused_ssim`) | CUDA fused kernel (submodule ngoài) |
| Đánh giá cuối (`metrics.py`) | `ssim` | PyTorch thuần, `utils/loss_utils.py` |

## 25. Loss tổng và `lambda_dssim`

Công thức loss, giống hệt ở cả hai vòng lặp (`train.py:103`, `pipeline/trainer.py:110`):
```python
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
```
tức `(1 - λ)·L1 + λ·(1 - SSIM)`, với `λ = opt.lambda_dssim`.

Giá trị mặc định: `self.lambda_dssim = 0.2` (`arguments/__init__.py:82`, trong `OptimizationParams`). `README.md:173` xác nhận cùng giá trị mặc định `0.2`.

Preset notebook ghi đè: `pipeline/config.py:44` truyền `"--lambda_dssim", "0.25"` vào `build_args` — tức đường train qua notebook luôn train với `λ = 0.25`, coi trọng SSIM (cấu trúc ảnh) hơn một chút so với mặc định CLI `0.2`. Đây là ví dụ minh hoạ số cụ thể của cấu hình mặc định notebook, không phải quy tắc cố định — người dùng CLI hoàn toàn có thể tự truyền `--lambda_dssim` khác.

**Không có mask loss, không có pose loss trong codebase hiện tại.** Toàn bộ `loss` chỉ gồm hai số hạng L1 và DSSIM ở trên — không có `compute_combined_loss`, không có `compute_instance_losses`, không có nhánh cộng thêm theo mask SAM2 hay theo sai số pose như tài liệu cũ mô tả; những hàm/khái niệm đó không tồn tại trong `train.py`, `pipeline/trainer.py`, `utils/loss_utils.py`, hay `utils/fast_utils.py` của codebase hiện tại (đã kiểm tra: hàm `get_loss`/`compute_photometric_loss` trong `utils/fast_utils.py` chỉ được dùng nội bộ trong luồng chấm điểm fastgs-lite ở PHẦN V, không nằm trong công thức loss chính của vòng lặp train).

---

# PHẦN V — BACKWARD VÀ KIỂM SOÁT MẬT ĐỘ

## 26. `loss.backward()` — gradient chảy đi đâu

Cả hai vòng lặp gọi đúng một dòng: `loss.backward()` (`train.py:104`, `pipeline/trainer.py:114`). `loss` là scalar `(1-λ)·L1 + λ·(1-SSIM)` dựng trên `image` — ảnh render trả về từ `render_fastgs`. Autograd đi ngược qua `_RasterizeGaussians.apply(...)` (`submodules/diff-gaussian-rasterization_fastgs/diff_gaussian_rasterization_fastgs/__init__.py:33-44`), tức là qua đúng một `torch.autograd.Function` mà `forward`/`backward` được cài bằng CUDA (`_C.rasterize_gaussians` / `_C.rasterize_gaussians_backward`). `_RasterizeGaussians.backward` (dòng 112-172) nhận `grad_out_color` (dL/d ảnh) và trả về gradient cho đúng 9 input đã truyền vào forward, theo thứ tự khai báo:

```
grads = (grad_means3D, grad_means2D, grad_dc, grad_sh, grad_colors_precomp,
          grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, None)
```

Trong `render_fastgs` (`gaussian_renderer/__init__.py:60-102`) các input này được gán từ property của `GaussianModel`:

| Input rasterizer | Nguồn (`GaussianModel`) | Activation nằm giữa gradient và tham số thô |
|---|---|---|
| `means3D` | `pc.get_xyz` → `self._xyz` | không có activation, `get_xyz` trả thẳng `_xyz` (`scene/gaussian_model.py:109-111`) |
| `opacities` | `pc.get_opacity` → `sigmoid(self._opacity)` | `torch.sigmoid` (`scene/gaussian_model.py:37, 157-158`) |
| `scales` | `pc.get_scaling` → `exp(self._scaling)` | `torch.exp` (`scene/gaussian_model.py:33`) |
| `rotations` | `pc.get_rotation` → `normalize(self._rotation)` | `torch.nn.functional.normalize` (`scene/gaussian_model.py:40, 135-137`) |
| `dc`, `shs` | `pc.get_features_dc`, `pc.get_features_rest` | không có activation (chỉ transpose) |
| `means2D` | `screenspace_points` (tensor phụ, xem bên dưới) | không có activation |

Vì `get_xyz`, `get_opacity`, `get_scaling`, `get_rotation` đều là `@property` áp activation lên `nn.Parameter` gốc, gradient của rasterizer đối với *giá trị đã activate* sẽ tự động được nhân với đạo hàm của activation đó (chain rule của autograd) khi lan tới `self._scaling`, `self._opacity`, `self._rotation`. Đây là lý do các tham số thô có thể mang giá trị âm/không giới hạn (`_scaling` là log-scale, `_opacity` là logit) trong khi giá trị dùng để render luôn hợp lệ (dương, trong [0,1], đơn vị norm).

**Trick `screenspace_points` / `viewspace_point_tensor`.** `means2D` không phải toạ độ 2D thật của Gaussian trên màn hình — nó là một tensor phụ khởi tạo bằng 0, không liên quan gì tới phép chiếu:

```python
screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype,
                                  requires_grad=True, device="cuda") + 0
screenspace_points.retain_grad()
```
(`gaussian_renderer/__init__.py:26-31`)

`means3D` (toạ độ 3D thật) mới là thứ được rasterizer chiếu sang màn hình để tính `dL/d(mean2D)` nội bộ. Nhưng CUDA kernel *cộng dồn* gradient màn hình đó vào input `means2D` mà nó nhận được — vì `means2D = screenspace_points` (một tensor zero, không đóng góp giá trị gì vào forward, nên phần "gradient của loss theo giá trị của means2D" chính là gradient màn hình thô, không bị pha trộn với gradient của `means3D`). Nói cách khác, `screenspace_points` được dùng như một "cổng dò" (probe) — trị số của nó luôn là 0 nên nó không ảnh hưởng đến ảnh render, nhưng backward vẫn phải tính `dL/d(means2D)` để lan ra `means3D` theo chuỗi phép chiếu, và giá trị trung gian đó được ghi thẳng vào `.grad` của `screenspace_points` nhờ `retain_grad()` (nếu không gọi `retain_grad()`, PyTorch sẽ giải phóng `.grad` của mọi tensor không phải lá sau `backward()`). Do đó sau `loss.backward()`, `render_pkg["viewspace_points"].grad` (đặt tên `viewspace_point_tensor` ở `train.py:97` / `viewspace` ở `pipeline/trainer.py:107`) chính là gradient 2D-màn hình mà `add_densification_stats` cần, không phải suy ra lại từ `means3D.grad`.

Điểm khác lạ so với 3DGS gốc: `screenspace_points` ở đây có **4 cột**, không phải 3 (dòng 27, dòng comment 26 còn giữ lại bản cũ 3 cột đã bị comment-out). Xem §27 để biết ý nghĩa 2 cột sau.

## 27. `add_densification_stats` — điều gì được cộng dồn

```python
def add_densification_stats(self, viewspace_point_tensor, update_filter):
    self.xyz_gradient_accum[update_filter]     += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
    self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
    self.denom[update_filter] += 1
```
(`scene/gaussian_model.py:493-496`)

- `update_filter` chính là `visibility_filter` — trong cả hai vòng lặp nó được tính là `(radii > 0).nonzero()` (`gaussian_renderer/__init__.py:108`), tức **chỉ số** (không phải mask bool) của các Gaussian có bán kính màn hình > 0 ở lượt render đó. Do đó chỉ Gaussian *nhìn thấy được ở camera vừa render* mới được cộng dồn — Gaussian bị frustum-cull hoặc có `radii == 0` không được cập nhật ở bước này.
- Cột `[:, :2]` của `viewspace_point_tensor.grad` là gradient 2D màn hình "ký hiệu" thông thường — norm Euclid của nó được cộng vào `xyz_gradient_accum`.
- Cột `[:, 2:]` (cột 2 và 3) là gradient **giá trị tuyệt đối** tích luỹ trên GPU: kernel CUDA cộng dồn `fabs(dL/dx)` và `fabs(dL/dy)` theo từng pixel thay vì cộng có dấu (`submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/backward.cu:592-596,607-610` — biến `Register_dL_dmean2D_z/_w` dùng `fabs(tmp_x)`, `fabs(tmp_y)`). Đây là kỹ thuật kiểu AbsGS/Pixel-GS: với một Gaussian lớn phủ nhiều pixel có gradient trái dấu (một phần ảnh cần nó dịch trái, phần khác cần dịch phải), gradient có-dấu bị triệt tiêu gần 0 dù Gaussian đó thực sự "kém", còn gradient trị tuyệt đối thì không bị triệt tiêu. Norm của cặp này được cộng vào `xyz_gradient_accum_abs`.
- `denom[update_filter] += 1` đếm số lần Gaussian đó được nhìn thấy (qua các lượt render khác nhau, tức các iteration khác nhau) — dùng làm mẫu số khi lấy trung bình ở `densify_and_prune_fastgs` (§29): `grad_vars = xyz_gradient_accum / denom`.

`max_radii2D` **không** được cập nhật trong hàm này — nó được duy trì ngay tại lời gọi, ở caller:
```python
gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
```
(`train.py:129`, tương đương `pipeline/trainer.py:121-122`) — chạy **trước** `add_densification_stats` trong cả hai vòng lặp, dùng `torch.max` theo từng phần tử để giữ lại bán kính lớn nhất từng quan sát được của mỗi Gaussian (dùng làm điều kiện prune "quá to trên màn hình" ở §30).

Tất cả các bộ đếm này (`xyz_gradient_accum`, `xyz_gradient_accum_abs`, `denom`, `max_radii2D`) bị **reset về 0** mỗi khi có Gaussian mới sinh ra hoặc bị xoá, trong `densification_postfix` (dòng 426-429) và `prune_points` (dòng 375-379) — chúng chỉ tích luỹ *giữa hai lần densify liên tiếp*, không phải trong suốt quá trình train.

## 28. `sampling_cameras` + `compute_gaussian_score_fastgs` — trái tim của fastgs-lite

Toàn bộ logic nằm trong `utils/fast_utils.py` (106 dòng). Đây là cơ chế "đa góc nhìn" thay thế cho gradient thuần của 3DGS gốc.

**Bước 0 — lấy mẫu camera.** `sampling_cameras(my_viewpoint_stack)` (`utils/fast_utils.py:10-19`):
```python
num_cams = 10
camlist = []
for _ in range(num_cams):
    loc = random.randint(0, len(my_viewpoint_stack) - 1)
    camlist.append(my_viewpoint_stack.pop(loc))
return camlist
```
Luôn lấy **đúng 10 camera** (hằng số cứng `num_cams = 10`, không phải tham số cấu hình), lấy ngẫu nhiên đều và `pop` khỏi list nên không trùng lặp trong một lần gọi (`my_viewpoint_stack` là bản copy toàn bộ tập train camera, được tạo mới ở caller mỗi lần: `scene.getTrainCameras().copy()`, `train.py:134` / `pipeline/trainer.py:126,138`). Nếu tập train có ít hơn 10 camera, `randint(0, len-1)` vẫn hợp lệ nhưng vòng lặp sẽ pop tới khi rỗng rồi lỗi — đọc mã cho thấy hàm không tự giới hạn `num_cams` theo độ dài stack.

**Bước 1 — với mỗi camera trong `camlist`, hai lượt render.** Trong `compute_gaussian_score_fastgs` (`utils/fast_utils.py:33-93`), vòng `for view in range(len(camlist))`:

1. *Lượt render thường* (dòng 75): `render_fastgs(cam, gaussians, pipe, bg, args.mult)["render"]` → `render_image`.
2. *Loss ảnh chuẩn* (dòng 76): `compute_photometric_loss` — L1+SSIM giống loss huấn luyện chính, nhưng dùng trọng số cố định `0.2` (không phải `opt.lambda_dssim`):
   ```python
   loss = 0.8 * Ll1 + 0.2 * (1 - fast_ssim(image, gt_image))
   ```
   (`utils/fast_utils.py:27-31`) — đây là scalar `photometric_loss` cho *cả ảnh*, dùng làm trọng số ở bước gộp.
3. *Bản đồ lỗi min-max chuẩn hoá* (`get_loss`, dòng 21-25):
   ```python
   l1_loss = mean(|render - gt|, dim=0).detach()          # (H, W), trung bình 3 kênh màu
   l1_loss_norm = (l1_loss - min(l1_loss)) / (max(l1_loss) - min(l1_loss))
   ```
   Đây là **min-max chuẩn hoá trên toàn ảnh** của riêng camera đó — không so sánh giữa các camera.
4. *Mặt nạ nhị phân* (dòng 82): `metric_map = (l1_loss_norm > args.loss_thresh).int()`, với `args.loss_thresh = 0.1` mặc định (`arguments/__init__.py:90`).

   ⚠ **Không có bước flatten nào ở đây.** `l1_loss_norm` có shape `(H, W)` và `metric_map` giữ nguyên shape đó — `fast_utils.py:70` chỉ so ngưỡng rồi `.int()`. Chỗ duy nhất tạo tensor 1D là **nhánh mặc định** trong `render_fastgs` khi caller *không* truyền `metric_map`:

   ```python
   # gaussian_renderer/__init__.py:37-38
   if metric_map==None:
       metric_map=torch.zeros(int(...image_height)*int(...image_width), dtype=torch.int, device='cuda')
   ```

   Hai đường đi vào kernel vì thế có **shape khác nhau** (`(H,W)` vs `(H*W,)`) nhưng cùng số phần tử và cùng layout bộ nhớ liên tục, nên phía CUDA đọc như nhau. Đây là điểm lỏng lẻo của code chứ không phải thiết kế — đừng mô tả nó thành "được phẳng hoá".
5. *Lượt render thứ hai, có cờ đếm* (dòng 84): `render_fastgs(cam, gaussians, pipe, bg, args.mult, get_flag=True, metric_map=metric_map)`. Lượt này rasterizer (phía CUDA) duyệt lại từng pixel bị đánh dấu lỗi trong `metric_map`, và với mỗi pixel đó cộng `+1` vào bộ đếm của **mọi Gaussian có đóng góp (alpha-blend) vào pixel đó** — trả về qua `render_pkg["accum_metric_counts"]`, một vector độ dài = số Gaussian hiện tại, kiểu đếm nguyên. Đây chính là "đổ ngược mặt nạ lỗi 2D về không gian Gaussian".
6. *Cộng dồn qua các view* (dòng 88-97):
   ```python
   if DENSIFY:
       full_metric_counts = accum_loss_counts  nếu view đầu, ngược lại += accum_loss_counts
   full_metric_score = photometric_loss * accum_loss_counts  nếu view đầu, ngược lại += ...
   ```
   `full_metric_counts` (chỉ tích khi `DENSIFY=True`) là **tổng số view đã đánh dấu Gaussian đó là lỗi** (cộng dồn số nguyên qua 10 camera). `full_metric_score` là **tổng có trọng số photometric-loss** của số lần bị đánh dấu — Gaussian bị đếm nhiều lần ở các view có loss ảnh cao sẽ có `full_metric_score` lớn hơn.

**Bước 2 — hai công thức gộp** (dòng 99-105):
```python
pruning_score = (full_metric_score - min(full_metric_score)) / (max(full_metric_score) - min(full_metric_score))
if DENSIFY:
    importance_score = floor(full_metric_counts / len(camlist))     # rounding_mode='floor'
else:
    importance_score = None
```
- `pruning_score` = min-max chuẩn hoá của `full_metric_score` trên **toàn bộ Gaussian** (không giới hạn theo view) → nằm trong [0,1], càng gần 1 càng "đóng góp nhiều vào các vùng lỗi nặng ở nhiều view" → ứng viên prune (khi hội tụ, Gaussian còn gây lỗi lớn dai dẳng có nghĩa nó đặt sai chỗ).

- `importance_score` = **số pixel-lỗi trung bình mỗi góc nhìn**, làm tròn xuống.

### ⚠ Đơn vị của `importance_score`: pixel, KHÔNG phải "phiếu bầu"

Đây là chỗ dễ hiểu sai nhất trong cả tài liệu, và bản trước của mục này đã hiểu sai. Phải bám vào **đơn vị** của `accum_metric_counts`:

| Đại lượng | Đơn vị | Miền giá trị |
|---|---|---|
| `accum_metric_counts` (một view) | **số pixel** mà Gaussian $i$ đóng góp *và* pixel đó bị `metric_map` gắn cờ lỗi | $0 \dots$ (số pixel trong footprint của splat) — có thể hàng trăm |
| `full_metric_counts` (cộng 10 view) | **số pixel**, cộng dồn | $0 \dots$ hàng nghìn |
| `importance_score` | **số pixel trung bình mỗi view** | $0 \dots$ hàng trăm |

Kernel CUDA cộng `+1` cho **mỗi pixel lỗi**, không phải `+1` cho mỗi view. Vì thế:

$$\text{importance}_i=\left\lfloor \frac{1}{V}\sum_{v=1}^{V}\underbrace{\text{counts}^{(v)}_i}_{\text{số PIXEL}} \right\rfloor,\qquad V=10$$

**Ngưỡng `importance_score > 5` nghĩa là:** "Gaussian này phủ trung bình **hơn 5 pixel bị đánh dấu lỗi** trên mỗi góc nhìn lấy mẫu." Một splat cỡ trung bình phủ vài chục đến vài trăm pixel, nên ngưỡng này hoàn toàn đạt được — nó lọc ra khoảng vài phần trăm quần thể chứ không phải gần như không ai qua nổi.

> **Vì sao cách hiểu "phiếu bầu" là bất khả thi về mặt logic.** Nếu `counts` là cờ 0/1 mỗi view thì $\sum_v \text{counts} \le 10$, nên $\lfloor \sum/10 \rfloor \in \{0, 1\}$ — và điều kiện `> 5` sẽ **không bao giờ** thoả với bất kỳ Gaussian nào. Densify sẽ chết hoàn toàn, `all_clones`/`all_splits` luôn AND với một mask toàn `False`, số Gaussian đứng yên từ đầu tới cuối. Nhưng `DOCS/assets/history.csv` ghi lại số Gaussian của `drjohnson` tăng từ 87.375 (vòng 1000) lên 174.003 (vòng 7000) — bằng chứng thực nghiệm trực tiếp rằng densify có chạy, tức cách hiểu đó sai.
>
> **Mẹo tự bắt lỗi:** khi một diễn giải làm cho một nhánh code trở nên *chết hoàn toàn*, gần như chắc chắn diễn giải đó sai chứ không phải code sai. Hãy kiểm bằng một số đo thực tế trước khi viết.

**Vai trò của phép `floor`.** Vì lấy trung bình rồi mới làm tròn xuống, một Gaussian chỉ bị **một** camera duy nhất tố sai (dù tố rất nặng) sẽ bị chia cho 10 và tụt xuống gần 0. Đây chính là ý nghĩa "multi-view consistent": muốn điểm cao thì phải sai **một cách nhất quán qua nhiều góc nhìn**. Phép floor còn là bộ lọc nhiễu miễn phí — mọi Gaussian có tổng counts $< 10$ đều nhận điểm 0.

### Ví dụ số (dùng $V=3$ cho gọn; repo thật dùng $V=10$)

Mọi con số trong cột giữa là **số pixel lỗi**, không phải cờ 0/1:

| | cam 1 ($\mathcal{L}_{\text{photo}}=0.20$) | cam 2 ($0.05$) | cam 3 ($0.10$) | `full_metric_counts` | `importance_score` |
|---|---|---|---|---|---|
| $G_A$ | 8 px | 6 px | 7 px | 21 | $\lfloor 21/3\rfloor=\mathbf{7}$ |
| $G_B$ | 12 px | 0 | 0 | 12 | $\lfloor 12/3\rfloor=\mathbf{4}$ |
| $G_C$ | 1 px | 0 | 1 px | 2 | $\lfloor 2/3\rfloor=\mathbf{0}$ |

- $G_A$ sai **nhất quán ở cả 3 góc nhìn** → điểm 7, vượt ngưỡng 5 → **ứng viên densify**.
- $G_B$ sai *nặng hơn* $G_A$ ở cam 1 (12 > 8) nhưng **chỉ ở một góc nhìn** → bị phép chia cho $V$ dìm xuống 4 → **không densify**. Đây đúng là hành vi mong muốn: không nhồi Gaussian để chữa một artefact chỉ nhìn thấy từ một phía (floater, phản chiếu).
- $G_C$ nhiễu lẻ tẻ → floor triệt về 0.

`full_metric_score` (cho pruning) thì **nhân trọng số** bằng loss toàn khung hình:

- $s_A=8(0.20)+6(0.05)+7(0.10)=1.6+0.3+0.7=2.60$
- $s_B=12(0.20)=2.40$
- $s_C=1(0.20)+1(0.10)=0.30$

Sau min-max trên toàn bộ tập Gaussian: $\text{Pruning}_A=1.000$, $\text{Pruning}_B=\frac{2.40-0.30}{2.60-0.30}=0.913$, $\text{Pruning}_C=0.000$.

> **Không mâu thuẫn khi $G_A$ vừa là ứng viên densify vừa là ứng viên prune.** Hai điểm số được tiêu thụ ở **hai giai đoạn khác nhau**: `importance_score` dùng ở vòng $< 15000$ để *thử thêm chi tiết*; `pruning_score` dùng ở `final_prune_fastgs` (vòng $> 15000$) để *dọn những gì đã thử mà không giúp được*.

Chạy `python demos/fastgs_mechanisms.py` (mục "2.") in ra đúng ba con số 7 / 4 / 0 và 1.000 / 0.913 / 0.000 này.

**Mô phỏng chạy được:** `demos/fastgs_mechanisms.py` tái hiện thu nhỏ đúng ba bước trên bằng NumPy (không cần CUDA):
- `error_mask(height, width, loss_thresh)` (dòng 18-23) — mô phỏng bước 3+4 (chuẩn hoá min-max rồi ngưỡng).
- `accum_metric_counts(mask, footprint_map)` (dòng 26-31) — mô phỏng bước 5 (đổ ngược mặt nạ pixel-lỗi về từng Gaussian qua "vùng phủ" `footprint_map` giả lập cho vai trò của rasterizer CUDA).
- `merge_views(counts, photometric_loss)` (dòng 35-39) — mô phỏng đúng công thức bước 2 ở trên (`importance = floor(counts.mean(axis=1))`, `pruning = minmax(counts · photometric_loss theo view rồi sum)`).
Chạy `python demos/fastgs_mechanisms.py` in ra ví dụ số cụ thể cho cả ba bước.

## 29. `densify_and_prune_fastgs` — điều kiện kép

Chữ ký: `densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score=None, pruning_score=None)` (`scene/gaussian_model.py:433`).

**Bước A — điều kiện gradient** (dòng 477-487):
```python
grad_vars = xyz_gradient_accum / denom;      grad_vars[isnan] = 0
grads_abs = xyz_gradient_accum_abs / denom;  grads_abs[isnan] = 0

grad_qualifiers      = norm(grad_vars, dim=-1) >= args.grad_thresh       # 0.0002
grad_qualifiers_abs  = norm(grads_abs, dim=-1) >= args.grad_abs_thresh   # 0.0012

clone_qualifiers = max(get_scaling, dim=1) <= args.dense * extent   # args.dense = 0.001
split_qualifiers = max(get_scaling, dim=1)  > args.dense * extent

all_clones = clone_qualifiers AND grad_qualifiers          # Gaussian nhỏ + gradient thường vượt ngưỡng
all_splits = split_qualifiers AND grad_qualifiers_abs       # Gaussian lớn + gradient tuyệt đối vượt ngưỡng
```
`extent = scene.cameras_extent` truyền từ caller — kích thước cảnh (bán kính bao camera), nên `args.dense * extent` là ngưỡng kích thước tương đối theo scale thực của scene, không phải hằng số tuyệt đối.

**Bước B — điều kiện kép với `importance_score`** (dòng 492-497): đây là đóng góp chính của fastgs-lite so với 3DGS gốc.
```python
metric_mask = importance_score > 5
densify_and_clone_fastgs(metric_mask, all_clones)
densify_and_split_fastgs(metric_mask, all_splits)
```
Cả clone lẫn split chỉ áp dụng cho Gaussian thoả **đồng thời**:

1. điều kiện gradient truyền thống của 3DGS (`all_clones` / `all_splits`), **VÀ**
2. `metric_mask`, tức phủ **trung bình hơn 5 pixel bị đánh dấu lỗi trên mỗi góc nhìn** trong 10 camera lấy mẫu.

⚠ **Đọc điều kiện 2 cho đúng đơn vị.** `importance_score` đếm **pixel**, không đếm "phiếu bầu của camera" — xem bảng đơn vị ở §28. Đọc `> 5` thành "hơn 5 trên 10 camera đồng ý" là sai, và nếu đúng như vậy thì nhánh densify sẽ không bao giờ kích hoạt (§28 giải thích vì sao).

Đây là phép AND giữa hai luồng tín hiệu **hoàn toàn độc lập**: gradient tích luỹ qua toàn bộ iteration kể từ lần densify trước, so với lỗi ảnh đo *tức thời* trên 10 camera mẫu tại đúng thời điểm này. Comment trong code gọi đây là "multi-view consistent metric... similar to taming 3dgs" (dòng 492-493).

**Nhánh clone** (`densify_and_clone_fastgs`, dòng 455-466): với Gaussian nhỏ đạt điều kiện, sao chép y hệt mọi thuộc tính (`_xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation, tmp_radii`) sang bản mới, nối vào cuối qua `densification_postfix`. Không dịch chuyển vị trí — bản sao trùng vị trí bản gốc (khác 3DGS gốc chỗ này? Không — 3DGS gốc cũng clone y hệt, gradient khác nhau sẽ tự tách chúng ra ở các bước sau).

**Nhánh split** (`densify_and_split_fastgs`, dòng 431-453), cho Gaussian lớn: lấy `N=2` mẫu mỗi Gaussian được chọn (`selected_pts_mask`):
```python
stds = get_scaling[selected].repeat(N, 1)
samples = torch.normal(mean=0, std=stds)                       # lấy mẫu N điểm quanh tâm theo phân phối scaling hiện tại
new_xyz = rotate(samples) + get_xyz[selected].repeat(N, 1)      # xoay theo rotation hiện tại rồi cộng vào tâm cũ
new_scaling = scaling_inverse_activation(get_scaling[selected].repeat(N,1) / (0.8*N))   # co scale lại
```
Với `N=2` mặc định (tham số hàm, không phải từ `args`), scale mới = scale cũ / 1.6 (tức `0.8*N = 1.6`) — giống hệt công thức chia scale của 3DGS gốc (`0.8*N`, không đổi). Sau khi tạo `N` bản mới, Gaussian gốc bị xoá: `prune_filter` đánh dấu `selected_pts_mask` (phần gốc) là `True` rồi gọi `prune_points(prune_filter)` — split luôn thay thế 1 Gaussian bằng N Gaussian con nhỏ hơn, không giữ bản gốc.

**Lịch gọi.** Cả hai vòng lặp gọi `densify_and_prune_fastgs` từ cùng một điều kiện:
```python
if iteration < opt.densify_until_iter:              # 15_000
    ... 
    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:   # 500, 100
        ... gọi compute_gaussian_score_fastgs rồi densify_and_prune_fastgs
```
(`train.py:127,132-145`; `pipeline/trainer.py:120,124-132`) — bắt đầu sau iteration 500 (`densify_from_iter = 500`) và dừng hẳn khi `iteration >= densify_until_iter = 15_000`. Nhịp densify khác nhau giữa hai đường chạy:

| Đường chạy | `densification_interval` | Nguồn |
|---|---|---|
| `train.py` không truyền cờ | **100** | mặc định `arguments/__init__.py:83` |
| `train_base.sh` / `train_big.sh` | **500** | truyền `--densification_interval 500` |
| Notebook / `pipeline/` | **500** | `Config.train_extra_args`, `pipeline/config.py:43` |

Với cấu hình notebook mặc định `iterations = 7000` (`pipeline/config.py:31`), điều kiện `iteration < 15_000` luôn đúng suốt toàn bộ quá trình train — densify chạy mỗi 500 iteration từ iteration 1000 đến 6500 mà không bao giờ chạm mốc dừng 15k.

## 30. Ba tầng pruning + `final_prune_fastgs`

| Tầng | Khi nào chạy | Điều kiện xoá | Hằng số | File:line |
|---|---|---|---|---|
| 1. Prune trong mỗi lần densify | Mỗi 100 iter, từ iter 500 đến <15000 (`densify_and_prune_fastgs`, cuối hàm) | `opacity < min_opacity` HOẶC (nếu `max_screen_size` được truyền) `max_radii2D > max_screen_size` HOẶC `scale.max > 0.1*extent` | `min_opacity=0.005` (truyền cứng ở caller, không phải `opt`); `max_screen_size = 20` chỉ khi `iteration > opacity_reset_interval` (3000), ngược lại `None` (tắt 2 điều kiện screen/world-size); `0.1*extent` | `scene/gaussian_model.py:464-468`; gọi ở `train.py:133`/`pipeline/trainer.py:125` |
| 2. Lấy mẫu ngân sách xoá (không phải một tầng độc lập, mà là cách *thực thi* tầng 1) | Cùng lúc với tầng 1, ngay sau khi tính `prune_mask` | Trong số các điểm đã bị `prune_mask` đánh dấu, chỉ xoá `remove_budget = floor(0.5 * số điểm bị đánh dấu)` điểm, lấy mẫu có trọng số `1/(pruning_score_inverse)` bằng `torch.multinomial` không hoàn lại | `remove_budget = int(0.5 * to_remove)`; trọng số = `1/(1e-6 + (1 - pruning_score))` | `scene/gaussian_model.py:470-483` |
| 3. `final_prune_fastgs` (hậu kỳ) | Mỗi 3000 iter, chỉ khi `15_000 < iteration < 30_000` | `opacity < min_opacity` HOẶC `pruning_score > 0.9` | `min_opacity = 0.1` (khác hẳn 0.005 của tầng 1); ngưỡng `pruning_score` cố định `0.9` | `scene/gaussian_model.py:498-505`; gọi ở `train.py:153-158`/`pipeline/trainer.py:137-140` |

Ghi chú quan trọng: **tầng 2 không phải là một cơ chế "prune điểm-mờ theo mẫu" tách biệt** — comment trong code còn nói thẳng "The budget is not necessary for our method" (`scene/gaussian_model.py:474`), tức nhóm tác giả tự nhận đây là phần thừa kế từ code cũ (kiểu Taming-3DGS) mà không có tác dụng bắt buộc gì trong fastgs-lite. Nó vẫn **chạy thật** mỗi lần densify: giới hạn số điểm bị prune ở tầng 1 xuống còn phân nửa, ưu tiên xoá trước những điểm có `pruning_score` thấp (vì trọng số tỉ lệ nghịch với `pruning_score`, `scores = 1 - pruning_score`, nên điểm có `pruning_score` càng nhỏ, `scores` càng lớn, trọng số lấy mẫu `1/(1e-6+scores)` càng nhỏ — nghĩa là thực ra trọng số lấy mẫu tỉ lệ **nghịch** với `1 - pruning_score`, tức các điểm **pruning_score cao** (được multi-view đánh giá là tệ) mới có xác suất bị chọn xoá cao hơn — khớp với vai trò của `pruning_score` là "điểm càng cao càng nên xoá").

Đúng như phần task đề cập, cần xác minh xem có tầng "prune ngẫu nhiên theo opacity thấp" độc lập nào khác không — không có; toàn bộ logic pruning nằm trong hai hàm `densify_and_prune_fastgs` (đuôi hàm) và `final_prune_fastgs`, không có hàm riêng biệt nào khác gọi `prune_points` trong `train.py`/`pipeline/trainer.py`.

**Tầng 3 dưới cấu hình mặc định notebook (`iterations=7000`) không bao giờ chạy.** Điều kiện `iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000` đòi hỏi `iteration > 15000`, nhưng vòng lặp chỉ chạy tới 7000 — `final_prune_fastgs` không được gọi lần nào trong một lần train mặc định của `pipeline/trainer.py`. Nó chỉ có ý nghĩa với cấu hình `--iterations` lớn hơn 15000 (ví dụ preset "big" nếu có, hoặc chạy `train.py` thủ công với iterations mặc định 30000 của `OptimizationParams`).

## 31. 🔒 Phẫu thuật trạng thái Adam

Vì số Gaussian thay đổi theo từng lần densify/prune, không thể để nguyên các tensor tham số của Adam (`nn.Parameter` có kích thước cố định) — mọi thao tác thêm/bớt điểm đều phải đồng bộ ba thứ cùng lúc: tensor tham số, `optimizer.state[param]["exp_avg"]`, `optimizer.state[param]["exp_avg_sq"]`.

**Xoá điểm — `_prune_optimizer(mask)`** (`scene/gaussian_model.py:307-327`), với `mask` ở đây là **valid mask** (`~mask` xoá, gọi từ `prune_points`, dòng 364-366):
```python
for opt in [self.optimizer, self.shoptimizer nếu có]:
    for group in opt.param_groups:
        stored_state = opt.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"]    = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
            del opt.state[group['params'][0]]
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            opt.state[group['params'][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
```
Chìa khoá: `nn.Parameter` mới được tạo ra (không sửa in-place tensor cũ), nên khoá cũ trong `opt.state` (dict khoá theo `id(tensor)`) sẽ trỏ tới tham số đã bị thay — bắt buộc phải `del opt.state[group['params'][0]]` (khoá **cũ**, đọc trước khi gán) rồi gán lại `opt.state[group['params'][0]]` (khoá **mới**) = `stored_state` đã được index theo cùng `mask`. Nếu quên bước `del`/gán lại này, Adam sẽ dùng state cũ (kích thước không khớp tensor mới) → lỗi shape, hoặc tệ hơn là state "ma" không bao giờ bị garbage-collect (rò rỉ bộ nhớ GPU) vì `opt.state` vẫn giữ tham chiếu tới `nn.Parameter` cũ. Kết quả: các điểm **giữ lại** mang đúng `exp_avg`/`exp_avg_sq` cũ của chúng (đã tích luỹ qua các bước Adam trước đó, chỉ lọc theo mask), các điểm bị xoá mất theo.

**Thêm điểm (clone/split) — `cat_tensors_to_optimizer(tensors_dict)`** (dòng 383-407), gọi từ `densification_postfix`:
```python
stored_state["exp_avg"]    = cat([stored_state["exp_avg"],    zeros_like(extension_tensor)], dim=0)
stored_state["exp_avg_sq"] = cat([stored_state["exp_avg_sq"], zeros_like(extension_tensor)], dim=0)
del opt.state[group['params'][0]]
group["params"][0] = nn.Parameter(cat([group["params"][0], extension_tensor], dim=0).requires_grad_(True))
opt.state[group['params'][0]] = stored_state
```
Cả điểm **clone** lẫn điểm **split** đều nhận `exp_avg`/`exp_avg_sq` **khởi tạo lại bằng 0** — không kế thừa moment bậc 1/2 từ Gaussian cha, dù về mặt giá trị (vị trí, màu, opacity) chúng có thể gần giống hệt cha (clone) hoặc suy ra trực tiếp từ cha (split). Đây là một điểm cần lưu ý: Gaussian mới sinh sẽ có bước Adam đầu tiên gần như "mù" về động lượng (giống lúc train mới bắt đầu với momentum=0), có thể khiến bước đầu tiên nhảy hơi giật nếu learning rate lớn — nhưng vì `xyz_gradient_accum`/`denom`/`max_radii2D` cũng bị reset toàn bộ về 0 trong cùng hàm (dòng 426-429) nên các bộ đếm densify liên quan cũng đồng bộ về "chưa quan sát".

**Điều gì hỏng nếu làm sai:** nếu index-theo-mask không nhất quán giữa param và state (ví dụ prune tensor tham số nhưng quên prune state, hoặc dùng nhầm `mask` thay vì `~mask`), Adam sẽ áp `exp_avg` của điểm A lên điểm B → gradient/momentum bị gán nhầm chủ, mô hình phân kỳ hoặc render sai màu ở đúng những điểm vừa densify/prune. Việc `del` + gán lại key theo tensor mới (thay vì sửa item tại chỗ) là bắt buộc vì `torch.optim.Optimizer.state` là `defaultdict` khoá theo **object identity** (`id()`) của `nn.Parameter`, và `group["params"][0] = nn.Parameter(...)` tạo object mới mỗi lần.

**Hai optimizer, một đường.** `training_setup` (dòng 176-177) luôn dựng đúng hai `torch.optim.Adam`:
```python
self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)     # tối ưu riêng cho f_rest (SH bậc cao)
```
Fork từng có thêm nhánh `--optimizer_type sparse_adam` dựng một `SparseGaussianAdam` gộp `l + sh_l` (kernel `adam.cu`, chỉ cập nhật Gaussian visible mỗi iteration). Nhánh đó không bao giờ chạy được — `SparseGaussianAdam` được import từ gói vanilla `diff_gaussian_rasterization` trong `try/except: pass` — nên toàn bộ nhánh, cờ `--optimizer_type` và `adam.cu` đã bị **xóa**. `_prune_optimizer`/`cat_tensors_to_optimizer` vì thế luôn duyệt đúng `[self.optimizer, self.shoptimizer]`.

## 32. `reset_opacity`

```python
def reset_opacity(self):
    opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
    optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    self._opacity = optimizable_tensors["opacity"]
```
(`scene/gaussian_model.py:244-247`) — với mỗi Gaussian, opacity sau activation bị **kẹp trần ở 0.01**: `opacity_new = min(opacity_hiện_tại, 0.01)`, sau đó chuyển ngược qua `inverse_sigmoid` để lưu vào `_opacity` thô. Nghĩa là Gaussian nào đang có opacity ≤ 0.01 giữ nguyên, còn Gaussian có opacity lớn hơn bị ép xuống 0.01 — không đặt cứng toàn bộ về một giá trị, chỉ *hạ trần*.

`replace_tensor_to_optimizer` (dòng 327-340) thực hiện đúng kiểu phẫu thuật Adam như §31 nhưng cho trường hợp "thay tensor mà không đổi số lượng điểm": state Adam của riêng nhóm `"opacity"` bị **reset về 0** (`exp_avg = exp_avg_sq = zeros_like(tensor)`), các nhóm tham số khác (xyz, scaling, rotation, features) không bị đụng tới.

**Lịch gọi** (`train.py:147-148`, `pipeline/trainer.py:133-135`):
```python
if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
    gaussians.reset_opacity()
```
`opacity_reset_interval = 3000` mặc định. Trường hợp đặc biệt nền trắng: nếu `dataset.white_background` bật, còn có một lần reset sớm đúng tại `iteration == densify_from_iter` (500) — vì trên nền trắng, Gaussian có thể "trốn" trong opacity cao ngay từ đầu để giả làm nền, reset sớm buộc chúng phải học lại độ mờ dựa trên tín hiệu thật thay vì lợi dụng nền.

**Lý do cần bước này:** đây là kỹ thuật chuẩn của 3DGS — theo thời gian, một số Gaussian tích luỹ opacity cao chỉ vì nằm ở vị trí "ăn theo" nền hoặc bị chồng lấp bởi các Gaussian khác che khuất phần render sai của chúng, khiến gradient prune (điều kiện `opacity < min_opacity` ở §30) không bao giờ đụng tới chúng. Định kỳ ép trần opacity buộc mọi Gaussian phải "chứng minh lại" độ cần thiết của mình qua vài trăm iteration tiếp theo — Gaussian nào không đóng góp thật sẽ tụt trở lại dưới `min_opacity` và bị tầng pruning kế tiếp dọn đi.

## 33. `optimizer_step`, lịch learning rate và lưu `.ply`

**`optimizer_step(iteration)`** (`scene/gaussian_model.py:190-209`, gọi từ `train.py:162`/`pipeline/trainer.py:143`):
```python
if iteration <= 15000:
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if iteration % 16 == 0:
        shoptimizer.step(); shoptimizer.zero_grad(set_to_none=True)
elif iteration <= 20000:
    if iteration % 32 == 0:
        optimizer.step(); ...; shoptimizer.step(); ...
else:
    if iteration % 64 == 0:
        optimizer.step(); ...; shoptimizer.step(); ...
```
Khác với một `optimizer.step()` trơn: (1) `self.optimizer` (xyz, opacity, scaling, rotation, f_dc) cập nhật **mỗi iteration** chỉ trong giai đoạn ≤15000, sau đó tần suất cập nhật giảm dần — mỗi 32 iteration (15000-20000) rồi mỗi 64 iteration (>20000); (2) `self.shoptimizer` (riêng `f_rest`, tức SH bậc cao) luôn cập nhật **thưa hơn** — mỗi 16 iteration trong giai đoạn đầu, rồi đồng bộ tần suất với `optimizer` ở hai giai đoạn sau. Đây là cách xấp xỉ ý tưởng "sparse Adam" của Taming-3DGS mà không cần CUDA kernel riêng: gradient vẫn được cộng dồn qua backward mỗi iteration bình thường (vì `zero_grad` chỉ được gọi khi thực sự `step()`), chỉ có tần suất *áp dụng* Adam là giảm đi để tiết kiệm compute — comment trong code xác nhận đúng ý này ("An optimization scheduler. The goal is similar to the sparse Adam of taming 3dgs.", dòng 226).

**Lịch learning rate ghim theo `position_lr_max_steps`.** `training_setup` (dòng 212-215):
```python
self.xyz_scheduler_args = get_expon_lr_func(
    lr_init=training_args.position_lr_init * spatial_lr_scale,
    lr_final=training_args.position_lr_final * spatial_lr_scale,
    max_steps=training_args.position_lr_max_steps)
```
`position_lr_max_steps` mặc định **30 000** (`arguments/__init__.py:78`). Trường này không tự đọc `--iterations`; đường notebook bù lại bằng cách cho `build_args` truyền `--position_lr_max_steps` bằng đúng số vòng train, còn đường CLI thuần thì người dùng phải tự truyền. `update_learning_rate(iteration)` (dòng 182-188) chỉ chỉnh lr của nhóm `"xyz"` theo hàm suy giảm mũ này, tính theo `iteration` tuyệt đối truyền vào — **không** tính theo tỉ lệ % tiến trình so với tổng số iteration thực tế của lần train đó.

Hệ quả khi train ít hơn 30 000 iteration (ví dụ mặc định notebook `iterations=7000`, `pipeline/config.py:31`): lr của `xyz` mới suy giảm được một đoạn nhỏ đầu của lịch trình mũ 30k bước, dừng lại ở một giá trị còn khá cao so với `position_lr_final` — nó **không bao giờ đi hết lịch trình** để chạm tới `lr_final = 0.0000016`. Đường notebook đã đóng khoảng cách này: `pipeline/trainer.py::build_args` truyền `--position_lr_max_steps` bằng đúng số vòng train, nên lịch mũ co lại vừa khít và lr vẫn chạm `lr_final` khi train kết thúc. Đường CLI thuần (`train.py`, `train_base.sh`, `train_big.sh`) **không** làm việc đó — chạy ngắn hơn 30k mà không tự truyền cờ này thì vẫn kết thúc với lr `xyz` cao hơn thiết kế gốc.

**Lưu `.ply` — hai cơ chế khác nhau ở hai vòng lặp:**
- `train.py`: `--save_iterations` (mặc định `[30_000]`, dòng 255) cộng thêm `args.iterations` vào cuối danh sách (dòng 262: `args.save_iterations.append(args.iterations)`), rồi mỗi khi `iteration in saving_iterations` gọi `scene.save(iteration)` (dòng 120-122). Không có xoá checkpoint cũ — mỗi mốc lưu là một thư mục `point_cloud/iteration_<n>/` riêng, giữ lại toàn bộ.
- `pipeline/trainer.py`: `_save_checkpoint(scene_obj, iteration, previous, drop_previous)` (dòng 39-47) gọi `scene_obj.save(iteration)` rồi, nếu `drop_previous=True` và có `previous`, xoá thư mục `point_cloud/iteration_<previous>/` bằng `shutil.rmtree`. Vòng lặp gọi hàm này mỗi `cfg.save_every` iteration (mặc định **2000**, `pipeline/config.py:34`) khi `iteration < iterations` (dòng 186-189), truyền `cfg.keep_last_checkpoint` (mặc định **True**, dòng 35) làm `drop_previous` — nghĩa là mặc định chỉ giữ **một** checkpoint trung gian tại một thời điểm (checkpoint mới ghi đè bằng cách xoá cái cũ ngay sau khi ghi cái mới), để tiết kiệm dung lượng đĩa Colab. Sau khi vòng lặp kết thúc, còn một lần lưu cuối cùng bắt buộc: `_save_checkpoint(scene_obj, iterations, saved_at, cfg.keep_last_checkpoint)` (dòng 194) — đảm bảo luôn có checkpoint tại đúng iteration cuối cùng dù `save_every` có chia hết cho `iterations` hay không.

**`.ply` không phải checkpoint có thể resume.** `save_ply` (dòng 260-277) chỉ ghi `xyz, f_dc, f_rest, opacity, scale, rotation` dưới dạng thuộc tính PLY thuần — không có `optimizer.state_dict()`, không có `xyz_gradient_accum`/`xyz_gradient_accum_abs`/`denom`/`max_radii2D`, không có `active_sh_degree` hay `spatial_lr_scale` (những thứ mà `capture()`/`restore()` mới lưu đủ, dòng 73-131, dùng cho checkpoint `.pth`). `train.py` vẫn có **nửa** cơ chế này: nếu truyền `--start_checkpoint`, nó `torch.load(checkpoint)` rồi `gaussians.restore(model_params, opt)` (dòng 43-45) để khôi phục đầy đủ trạng thái Adam/densify từ một file `.pth`. Nhưng **nửa còn lại — ghi file đó — không tồn tại**: tham số `--checkpoint_iterations` được khai báo (`train.py:252`, mặc định `[30_000]`) và truyền vào `training(...)` nhưng không hề được dùng bên trong hàm để gọi `torch.save((gaussians.capture(...), iteration), ...)` ở bất kỳ đâu — đây là phần vestigial còn sót lại từ mã 3DGS gốc, không hoạt động trong repo hiện tại. `pipeline/trainer.py` không có cơ chế `--start_checkpoint`/`.pth` nào cả, chỉ dùng `.ply` qua `_save_checkpoint`. Kết luận chung: trong pipeline hiện tại (cả `train.py` chạy mặc định lẫn `pipeline/trainer.py`), việc dừng giữa chừng rồi "tiếp tục" chỉ có thể khôi phục lại đám mây điểm từ `.ply`, không khôi phục được trạng thái Adam hay các bộ đếm densify — train tiếp từ một `.ply` tương đương khởi động lại optimizer/densify từ đầu trên một point cloud đã qua huấn luyện, chứ không phải "tiếp tục đúng như đang dở".
