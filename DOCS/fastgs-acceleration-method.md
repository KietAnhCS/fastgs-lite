# Cơ chế tăng tốc của FastGS

> Tài liệu này giải thích **FastGS làm gì để huấn luyện 3D Gaussian Splatting trong ~100 giây**.
> Cách trình bày giống hệt [gaussian-splatting-math.md](gaussian-splatting-math.md): mỗi cơ chế đi kèm công thức, bảng ký hiệu, ví dụ số và phần "vì sao".
> Tên hàm/tham số trong bài trỏ thẳng tới mã nguồn: `utils/fast_utils.py`, `scene/gaussian_model.py`, `gaussian_renderer/__init__.py`, `train.py`.
> Mỗi cơ chế còn có một mô phỏng NumPy chạy được (không cần CUDA) trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) — tên hàm tương ứng được ghi kèm ngay đầu mỗi phần.

---

## Phần 0: Thời gian huấn luyện 3DGS đi đâu?

3DGS gốc chậm vì ba nguyên nhân cộng dồn qua 30.000 vòng lặp:

| Nguồn tốn thời gian | Bản chất |
|---|---|
| **Số lượng Gaussian phình to** | Adaptive Density Control nhân bản Gaussian ở *mọi* vùng có gradient lớn, kể cả vùng đã hội tụ → tới hàng triệu điểm, mỗi vòng lặp phải rasterize và backward hết. |
| **Rasterization theo tile** | Mỗi splat 2D được gán cho tất cả các tile 16×16 mà bounding box của nó chạm tới. Bounding box lỏng → 1 splat đè lên rất nhiều tile → nhiều cặp (tile, Gaussian) phải sort và blend. |
| **Backward của Spherical Harmonics** | Màu phụ thuộc góc nhìn được mã hoá bằng 48 hệ số SH bậc 3; hội tụ chậm nếu mọi hệ số dùng chung một learning rate nhỏ. |

FastGS tấn công cả ba bằng ba đòn bẩy tương ứng:

1. **Điểm số nhất quán đa góc nhìn (multi-view consistency score)** — chỉ nhân bản Gaussian ở nơi *nhiều camera cùng thấy sai*, và cắt bỏ mạnh tay Gaussian vô ích. → **Phần 1–3**
2. **Compact box (`--mult`)** — thu nhỏ bounding box của mỗi splat để giảm số tile. → **Phần 4**
3. **Tách learning rate SH bậc thấp / bậc cao** (`--lowfeature_lr` / `--highfeature_lr`). → **Phần 5**

Kết quả: cùng số vòng lặp nhưng mỗi vòng rẻ hơn nhiều, và số Gaussian được giữ ở mức tối thiểu cần thiết.

---

## Phần 1: Bản đồ lỗi nhị phân trên mỗi góc nhìn

> Mô phỏng chạy được: `normalize_minmax()`, `error_mask()` và `accum_metric_counts()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "1." khi chạy `python demos/fastgs_mechanisms.py`).

Hàm `compute_gaussian_score_fastgs` (`utils/fast_utils.py:45`) là đóng góp chính. Nó chạy trước mỗi lần densify, lấy mẫu **10 camera ngẫu nhiên** (`sampling_cameras`, `fast_utils.py:10`) và với từng camera làm 3 bước.

### Bước 1.1 — Chuẩn hoá bản đồ lỗi L1 về [0, 1]

$$e(u,v)=\frac{1}{3}\sum_{k\in\{R,G,B\}}\bigl|\,I_{\text{render}}(u,v,k)-I_{\text{gt}}(u,v,k)\,\bigr|$$

$$\hat{e}(u,v)=\frac{e(u,v)-\min_{u,v} e}{\max_{u,v} e-\min_{u,v} e}$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $I_{\text{render}}, I_{\text{gt}}$ | Ảnh render và ảnh thật của camera đang xét |
| $e(u,v)$ | Sai số L1 trung bình 3 kênh màu tại pixel $(u,v)$ — hàm `get_loss` |
| $\hat{e}(u,v)$ | Sai số đã chuẩn hoá min–max về đoạn $[0,1]$ trên toàn ảnh |

Chuẩn hoá min–max khiến ngưỡng ở bước sau **không phụ thuộc độ sáng tuyệt đối của cảnh** — cảnh tối và cảnh sáng dùng chung một `loss_thresh`.

### Bước 1.2 — Nhị phân hoá bằng ngưỡng `loss_thresh`

$$m(u,v)=\begin{cases}1 & \hat{e}(u,v) > \tau_{\text{loss}}\\[4pt] 0 & \text{ngược lại}\end{cases}
\qquad \tau_{\text{loss}}=\texttt{loss\_thresh}\ (\text{mặc định }0.1)$$

`m` là `metric_map` trong code (`fast_utils.py:82`): một mặt nạ đánh dấu **những pixel mà mô hình hiện tại đang tái tạo tệ nhất**. `loss_thresh` thấp → nhiều pixel bị đánh dấu → nhiều Gaussian được coi là "cần sửa" → giữ lại / sinh thêm nhiều Gaussian hơn.

### Bước 1.3 — Đổ ngược mặt nạ pixel về từng Gaussian

Rasterizer được gọi lại với `get_flag=True` và `metric_map=m` (`fast_utils.py:84`). Trong lượt này, mỗi Gaussian $i$ nhận về một bộ đếm:

$$\text{counts}^{(v)}_i=\#\bigl\{\text{pixel }(u,v)\ \text{mà Gaussian }i\ \text{đóng góp và}\ m(u,v)=1\bigr\}$$

trả về ở `render_pkg["accum_metric_counts"]`. Trực giác: **Gaussian $i$ "chịu trách nhiệm" cho bao nhiêu pixel lỗi** ở góc nhìn $v$ này.

---

## Phần 2: Gộp nhiều góc nhìn thành hai điểm số

> Mô phỏng chạy được: `merge_views()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "2.").

Sau khi lặp qua cả 10 camera, hàm gộp lại thành hai đại lượng per-Gaussian.

### Công thức (1) — Importance score (điều khiển việc *sinh thêm*)

$$\text{Importance}_i=\left\lfloor \frac{1}{V}\sum_{v=1}^{V}\text{counts}^{(v)}_i \right\rfloor
\qquad V=10$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $V$ | Số camera lấy mẫu (`num_cams = 10`) |
| $\text{counts}^{(v)}_i$ | Số pixel lỗi mà Gaussian $i$ phủ ở góc nhìn $v$ |
| $\lfloor\cdot\rfloor$ | Làm tròn xuống (`rounding_mode='floor'`, `fast_utils.py:102`) |

Đây là **số pixel-lỗi trung bình mỗi góc nhìn** mà một Gaussian gây ra. Vì lấy trung bình rồi làm tròn xuống, một Gaussian chỉ bị "một camera duy nhất" tố sai sẽ có điểm gần 0 — **phải sai một cách nhất quán trên nhiều góc nhìn** mới được điểm cao. Đó là ý nghĩa của "multi-view consistent".

### Công thức (2) — Pruning score (điều khiển việc *cắt bỏ*)

$$s_i=\sum_{v=1}^{V}\mathcal{L}^{(v)}_{\text{photo}}\cdot\text{counts}^{(v)}_i$$

$$\text{Pruning}_i=\frac{s_i-\min_j s_j}{\max_j s_j-\min_j s_j}\in[0,1]$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathcal{L}^{(v)}_{\text{photo}}$ | Loss ảnh của cả khung hình $v$: $(1-\lambda)\mathcal{L}_1+\lambda(1-\text{SSIM})$ với $\lambda=0.2$ (`compute_photometric_loss`) |
| $s_i$ | Điểm thô: số pixel-lỗi của Gaussian $i$, **nhân trọng số** bằng độ tệ toàn cục của góc nhìn đó |
| $\text{Pruning}_i$ | Điểm đã chuẩn hoá về $[0,1]$; càng gần 1 = Gaussian càng "vô dụng / gây hại" |

Khác biệt then chốt so với Importance: pruning score **nhân thêm $\mathcal{L}^{(v)}_{\text{photo}}$**. Một Gaussian phủ nhiều pixel lỗi trong một khung hình vốn đã render rất tệ sẽ bị phạt nặng hơn.

### Ví dụ số

3 Gaussian, $V=3$ camera. Bảng `counts` và loss khung hình:

| | cam 1 ($\mathcal{L}_{\text{photo}}=0.20$) | cam 2 ($0.05$) | cam 3 ($0.10$) |
|---|---|---|---|
| $G_A$ | 8 | 6 | 7 |
| $G_B$ | 12 | 0 | 0 |
| $G_C$ | 1 | 0 | 1 |

**Importance** (trung bình rồi floor):
- $G_A=\lfloor(8+6+7)/3\rfloor=\lfloor7.0\rfloor=7$ → **vượt ngưỡng 5** → ứng viên densify.
- $G_B=\lfloor12/3\rfloor=4$ → dưới ngưỡng: sai nhiều nhưng **chỉ ở 1 góc nhìn** → không densify (tránh nhồi Gaussian cho artefact cục bộ).
- $G_C=\lfloor2/3\rfloor=0$.

**Pruning** (điểm thô $s_i$):
- $s_A=0.20\cdot8+0.05\cdot6+0.10\cdot7=1.60+0.30+0.70=2.60$
- $s_B=0.20\cdot12=2.40$
- $s_C=0.20\cdot1+0.10\cdot1=0.30$

Chuẩn hoá: $\text{Pruning}_A=1.0$, $\text{Pruning}_B=\dfrac{2.40-0.30}{2.60-0.30}=0.913$, $\text{Pruning}_C=0.0$.

Ở lần "final prune" ($\tau=0.9$, Phần 3.3), **cả $G_A$ và $G_B$ đều bị xoá** — chúng liên tục nằm dưới các pixel sai. $G_C$ giữ lại.

> Nghịch lý biểu kiến: $G_A$ vừa là ứng viên **densify** (Importance = 7) vừa là ứng viên **prune** (Pruning ≈ 1). Không mâu thuẫn: hai điểm số dùng ở **hai giai đoạn khác nhau** — densify chạy ở vòng < 15k để *thử thêm chi tiết*, final-prune chạy ở vòng > 15k để *dọn những gì không giúp được*.

---

## Phần 3: Densification có điều kiện kép

> Mô phỏng chạy được: `densify_masks()` (điều kiện AND khi densify) và `final_prune_fastgs()` (tầng pruning cuối) trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "3.").

Hàm `densify_and_prune_fastgs` (`scene/gaussian_model.py:468`). Một Gaussian chỉ được nhân bản khi **thoả đồng thời hai điều kiện độc lập**.

### 3.1 — Điều kiện gradient (chọn *ở đâu* cần thêm chi tiết)

$$\text{clone}_i:\ \lVert \bar{g}_i\rVert \ge \tau_{\text{grad}}\ \ \wedge\ \ \max(\text{scale}_i)\le \delta\cdot\text{extent}$$

$$\text{split}_i:\ \lVert \bar{g}^{\text{abs}}_i\rVert \ge \tau_{\text{grad}}^{\text{abs}}\ \ \wedge\ \ \max(\text{scale}_i) > \delta\cdot\text{extent}$$

| Ký hiệu | Ý nghĩa | Tham số CLI (mặc định) |
|---|---|---|
| $\bar{g}_i$ | Gradient vị trí 2D tích luỹ / số lần quan sát (`xyz_gradient_accum / denom`) | `--grad_thresh` (0.0002) |
| $\bar{g}^{\text{abs}}_i$ | Gradient **trị tuyệt đối** tích luỹ (kiểu Abs-GS) — bắt được vùng có gradient dao động đổi dấu mà tổng gần 0 | `--grad_abs_thresh` (0.0012) |
| $\delta\cdot\text{extent}$ | Ngưỡng kích thước: Gaussian nhỏ thì **clone** (thiếu mật độ), Gaussian to thì **split** (thiếu độ mịn) | `--dense` (0.001) |

Bước này giống 3DGS/Abs-GS: tách "clone vs split" theo gradient và kích thước.

### 3.2 — Điều kiện nhất quán đa góc nhìn (lọc *cái nào* thực sự đáng thêm)

$$\text{metric\_mask}_i = \bigl[\ \text{Importance}_i > 5\ \bigr]$$

```python
# scene/gaussian_model.py:494
metric_mask = importance_score > 5
self.densify_and_clone_fastgs(metric_mask, all_clones)   # AND theo từng phần tử
self.densify_and_split_fastgs(metric_mask, all_splits)
```

**Đây là điểm khác biệt cốt lõi với 3DGS gốc.** 3DGS densify mọi Gaussian có gradient lớn. FastGS thêm phép **AND**: gradient lớn *và* nhiều góc nhìn cùng thấy sai. Hệ quả:

- Gaussian ở vùng đã hội tụ (gradient còn dư nhưng render đã đúng) → Importance thấp → **không nhân bản** → số điểm không phình.
- Gaussian ở artefact chỉ thấy từ 1–2 góc (ví dụ floater phản chiếu) → Importance thấp → **không được củng cố**.

### 3.3 — Ba tầng pruning

| Khi nào | Điều kiện xoá | Vị trí trong code |
|---|---|---|
| Mỗi lần densify | `opacity < 0.005`, hoặc bán kính màn hình > 20 px, hoặc scale > `0.1·extent` | `gaussian_model.py:499–503` |
| Mỗi lần densify (tuỳ chọn) | Lấy mẫu `0.5 × (số điểm opacity thấp)` để xoá, xác suất tỉ lệ $1/(1-\text{Pruning}_i)$ | `gaussian_model.py:505–518` |
| **Mỗi 3.000 vòng, sau vòng 15k** | `opacity < 0.1` **hoặc** `Pruning_i > 0.9` — `final_prune_fastgs` | `train.py:153`, `gaussian_model.py:533` |

Tầng thứ ba là "dọn dẹp mạnh tay" ở giai đoạn cuối: sau ~15k vòng mô hình đã cơ bản hội tụ, nên có thể cắt aggressively những Gaussian điểm-pruning cao mà **không giảm chất lượng** (xem thí nghiệm 20k vòng trong bản arXiv).

---

## Phần 4: Compact box — giảm số tile mỗi splat (`--mult`)

> Mô phỏng chạy được: `tiles_touched()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "4.").

3DGS gán mỗi splat 2D cho **mọi tile 16×16 mà bounding box hình chữ nhật của nó chạm tới**. Bounding box tính theo $3\sigma$ của ellipse là **lỏng** — nhiều tile ở góc hộp gần như không nhận đóng góp nào nhưng vẫn phải vào bước sort + blend.

FastGS nhân bán trục của hộp với hệ số `mult`:

$$\text{half-extent}_{x} = \texttt{mult}\cdot 3\sqrt{\Sigma'_{11}},\qquad
\text{half-extent}_{y} = \texttt{mult}\cdot 3\sqrt{\Sigma'_{22}}$$

| `mult` | Hệ quả |
|---|---|
| `0.5` (mặc định) | Hộp co còn nửa → số cặp (tile, Gaussian) giảm mạnh → sort & rasterize nhanh hơn |
| `0.7` (dùng cho Tanks&Temples, Deep Blending trong cả `train_base.sh` và `train_big.sh`) | Cân bằng an toàn hơn khi splat lớn/nền phức tạp |
| → 1.0 | Quay về hành vi 3DGS gốc |

Vì đuôi Gaussian ở ngoài ~$2\sigma$ đóng góp $\alpha$ rất nhỏ (công thức (4) trong tài liệu toán), cắt bớt phần rìa hộp gần như **không đổi ảnh** nhưng bỏ được nhiều phép tính. `mult` phải truyền **nhất quán cho cả `train.py` và `render.py`** (thấy rõ trong `train_base.sh`).

---

## Phần 5: Tách learning rate cho Spherical Harmonics

> Mô phỏng chạy được: `steps_to_converge()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "5.").

3DGS dùng một `feature_lr = 0.0025` cho toàn bộ hệ số SH. FastGS chia đôi:

| Tham số | Điều khiển | Mặc định | Vai trò |
|---|---|---|---|
| `--lowfeature_lr` | `features_dc` — SH bậc 0, tức **màu cơ bản** không phụ thuộc góc nhìn | 0.0025 | Giữ nguyên nhịp cũ |
| `--highfeature_lr` | `features_rest` — SH bậc 1–3, tức **phần màu thay đổi theo góc nhìn** (specular, ánh kim) | 0.005 (tới 0.02–0.04 cho cảnh indoor/outdoor nhiều phản xạ) | Tăng gấp 2–8× để hội tụ nhanh |

Lý do tách: thành phần bậc thấp mang phần lớn năng lượng màu, dễ bất ổn nếu lr cao; thành phần bậc cao nhỏ và cần nhiều bước để "nở" ra. Cho phần bậc cao một lr lớn hơn giúp màu phụ thuộc góc nhìn **đạt được trong ít vòng lặp hơn** — quan trọng khi tổng ngân sách chỉ ~vài nghìn vòng hiệu dụng.

Xem `train_base.sh`: các cảnh `garden`, `room`, `counter`, `kitchen`, `bonsai` đều đặt `--highfeature_lr 0.02`; Tanks&Temples đặt `0.04` (`truck`) / `0.042` (`train`); riêng Deep Blending (`playroom`, `drjohnson`) lại đặt *thấp* hơn mặc định (`0.0015`–`0.0025`) vì cảnh trong nhà ít phản xạ specular.

---

## Phần 6: Vì sao cộng lại thành "100 giây"?

> Mô phỏng chạy được: `gaussian_trajectory()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "6.") — vẽ quỹ đạo số Gaussian 3DGS so với FastGS qua 30.000 vòng, minh hoạ hiệu ứng cộng dồn của các đòn bẩy dưới đây.

| Cơ chế | Cắt giảm cái gì | Đòn bẩy |
|---|---|---|
| Điều kiện AND khi densify (Phần 3.2) | Số Gaussian sinh ra ở vùng đã tốt / artefact cục bộ | Ít điểm hơn ⇒ forward + backward + optimizer mỗi vòng đều rẻ hơn |
| Final-prune theo pruning score (Phần 3.3) | Đuôi dài các Gaussian vô ích ở nửa sau huấn luyện | Giảm điểm ⇒ giảm bộ nhớ VRAM và thời gian mỗi vòng |
| Compact box `--mult` (Phần 4) | Số cặp (tile, Gaussian) phải sort/blend | Kernel rasterization nhanh hơn ở **mọi** vòng, cả train lẫn render |
| Tách lr SH (Phần 5) | Số vòng cần để màu specular hội tụ | Về đích với ít vòng lặp hiệu dụng hơn |
| Lấy mẫu 10 camera cho scoring (Phần 1) | Chi phí của chính bước scoring | Ước lượng đủ tốt mà không phải render toàn bộ tập train |

**So sánh tổng quan** (số liệu từ README):

| | 3DGS gốc | FastGS |
|---|---|---|
| Thời gian train | 5–30 phút | **~100 giây** |
| Tiêu chí densify | Chỉ gradient | Gradient **AND** nhất quán đa góc nhìn |
| Kiểm soát số Gaussian | Phình tự do | Chặn ở cả hai đầu (sinh có điều kiện + prune theo score) |
| Tăng tốc so với 3DGS | — | 3.32× so với DashGaussian (Mip-NeRF 360); 15.45× so với 3DGS (Deep Blending) |
| Chất lượng render | chuẩn | ngang ngửa SOTA |

**Kết luận:** ý tưởng trung tâm của FastGS là thay câu hỏi *"Gaussian này có gradient lớn không?"* bằng *"Nhiều camera có cùng đồng ý rằng chỗ này đang sai không?"*. Tín hiệu nhất quán đa góc nhìn vừa **rẻ để tính** (10 lần render phụ), vừa **chọn lọc hơn nhiều** — nên FastGS thêm Gaussian đúng chỗ, bỏ Gaussian đúng lúc, và giữ mỗi bước rasterization gọn nhẹ.

---

## Phần 7: Ba điều mã nguồn nói mà README không nói

Sáu phần trên mô tả **thiết kế**. Ba mục dưới đây là kết quả đọc mã nguồn, và chúng quyết định cách cấu hình thực tế.

### 7.1 — Lịch optimizer bị đóng đinh theo ngân sách 30.000 vòng

`scene/gaussian_model.py:225` — `optimizer_step(iteration)` không bước Adam mỗi vòng:

| Khoảng vòng lặp | Hành vi |
|---|---|
| `iteration <= 15000` (dòng 227) | bước Adam **mỗi vòng**; optimizer SH mỗi 16 vòng |
| `15000 < iteration <= 20000` (dòng 233) | bước Adam **mỗi 32 vòng** |
| `iteration > 20000` | bước Adam **mỗi 64 vòng** |

Cùng với `densify_until_iter = 15000` và `position_lr_max_steps = 30000`: **toàn bộ chi phí nằm ở 0–15k**, còn **15k–30k gần như miễn phí** nhưng vẫn chạy `final_prune_fastgs`.

⇒ Hạ `--iterations` xuống 15k là lỗ: tiết kiệm rất ít thời gian mà mất phần tỉa cuối. Sàn hợp lý là **20000**, và khi rút ngắn phải đồng bộ `--position_lr_max_steps`, `--densify_until_iter`, cùng hai ngưỡng cứng ở dòng 227/233.

### 7.2 — `--antialiasing` là cờ chết trong fork này

`arguments/__init__.py:70` khai báo `self.antialiasing = False`, nhưng `gaussian_renderer/__init__.py` dựng `GaussianRasterizationSettings(...)` **không có** trường đó, và `submodules/diff-gaussian-rasterization_fastgs/` không tham chiếu chữ `antialiasing` ở đâu. Truyền cờ chỉ bị bỏ qua âm thầm. Muốn chống răng cưa thật phải vá công thức Mip-Splatting vào kernel CUDA.

### 7.3 — `densification_interval` là lever thời gian bị bỏ quên

Mặc định trong `arguments/__init__.py` là `100`, nhưng `train_base.sh` thật sự dùng `500`. Vì `compute_gaussian_score_fastgs` render 10 camera × 2 lượt mỗi lần gọi, khác biệt là **~145 lần gọi so với ~29 lần** trong khoảng 500–15000.

---

## Đọc tiếp

| Tài liệu | Nội dung |
|---|---|
| [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) | Mô phỏng NumPy chạy được cho từng cơ chế ở trên (`python demos/fastgs_mechanisms.py`, không cần GPU) — hàm tương ứng được ghi ngay đầu mỗi Phần |
| [`fastgs-acceleration-method.ipynb`](../fastgs-acceleration-method.ipynb) | Notebook chỉ còn phần glue tiếng Anh: dựng pipeline `pipeline/` thật trên Colab T4 (check GPU → config → load dataset → smoke test → train → analytics → submission.zip) — cần GPU |
| [colab-t4-guide.md](colab-t4-guide.md) | Hướng dẫn vận hành pipeline Colab T4 ở trên: preset, theo dõi tiến độ, xử lý sự cố |
| [README.md](README.md) | Tổng quan repo và cách bắt đầu |
| [gaussian-splatting-math.md](gaussian-splatting-math.md) | Nền tảng toán học 3DGS mà tài liệu này giả định người đọc đã biết |

Mã nguồn tương ứng: `utils/fast_utils.py`, `scene/gaussian_model.py:468–540`, `train.py:126–158`, `gaussian_renderer/__init__.py`.
