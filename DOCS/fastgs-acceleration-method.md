# Cơ chế tăng tốc của fastgs-lite

> Tài liệu này giải thích **những gì mã nguồn trong repo này thật sự làm để tăng tốc huấn luyện 3D Gaussian Splatting**, và ở đâu nó khác 3DGS gốc.
> Mọi công thức, hằng số và hành vi dưới đây đều được đối chiếu với code trong repo — không lấy con số nào từ paper upstream. Chỗ nào là suy luận chưa đo thì có ghi rõ.
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

fastgs-lite tấn công cả ba bằng ba đòn bẩy tương ứng:

1. **Điểm số nhất quán đa góc nhìn (multi-view consistency score)** — chỉ nhân bản Gaussian ở nơi *nhiều camera cùng thấy sai*, và cắt bỏ mạnh tay Gaussian vô ích. → **Phần 1–3**
2. **Compact box (`--mult`)** — thu nhỏ bounding box của mỗi splat để giảm số tile. → **Phần 4**
3. **Tách learning rate SH bậc thấp / bậc cao** (`--lowfeature_lr` / `--highfeature_lr`). → **Phần 5**

Ý đồ thiết kế: cùng số vòng lặp nhưng mỗi vòng rẻ hơn, và số Gaussian được giữ ở mức tối thiểu cần thiết.

---

## Phần 1: Bản đồ lỗi nhị phân trên mỗi góc nhìn

> Mô phỏng chạy được: `normalize_minmax()`, `error_mask()` và `accum_metric_counts()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "1." khi chạy `python demos/fastgs_mechanisms.py`).

Hàm `compute_gaussian_score_fastgs` (`utils/fast_utils.py:45`) là cơ chế trung tâm. Nó chạy trước mỗi lần densify, lấy mẫu **10 camera ngẫu nhiên** (`sampling_cameras`, `fast_utils.py:10`) và với từng camera làm 3 bước.

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
| $\mathcal{L}^{(v)}_{\text{photo}}$ | Loss ảnh của cả khung hình $v$: $(1-\lambda)\mathcal{L}_1+\lambda(1-\text{SSIM})$ với $\lambda=0.2$ **cứng trong code** (`compute_photometric_loss`, `fast_utils.py:30`) |
| $s_i$ | Điểm thô: số pixel-lỗi của Gaussian $i$, **nhân trọng số** bằng độ tệ toàn cục của góc nhìn đó |
| $\text{Pruning}_i$ | Điểm đã chuẩn hoá về $[0,1]$; càng gần 1 = Gaussian càng "vô dụng / gây hại" |

Khác biệt then chốt so với Importance: pruning score **nhân thêm $\mathcal{L}^{(v)}_{\text{photo}}$**. Một Gaussian phủ nhiều pixel lỗi trong một khung hình vốn đã render rất tệ sẽ bị phạt nặng hơn.

> **Bẫy:** $\lambda=0.2$ ở đây là hằng số viết thẳng trong `fast_utils.py:30`, **không** đọc `--lambda_dssim`. Preset của repo dùng `--lambda_dssim 0.25` cho loss train, nên loss dùng để *tối ưu* và loss dùng để *chấm điểm pruning* đang lệch nhau. Đổi `--lambda_dssim` không kéo theo chỗ này.

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

**Đây là điểm khác biệt cốt lõi với 3DGS gốc.** 3DGS densify mọi Gaussian có gradient lớn. fastgs-lite thêm phép **AND**: gradient lớn *và* nhiều góc nhìn cùng thấy sai. Hệ quả:

- Gaussian ở vùng đã hội tụ (gradient còn dư nhưng render đã đúng) → Importance thấp → **không nhân bản** → số điểm không phình.
- Gaussian ở artefact chỉ thấy từ 1–2 góc (ví dụ floater phản chiếu) → Importance thấp → **không được củng cố**.

### 3.3 — Hai đường xoá Gaussian (không phải ba tầng độc lập)

`densify_and_prune_fastgs` có một khối lọc trông như hai tầng, nhưng **chỉ gọi `prune_points` đúng một lần**:

```python
# scene/gaussian_model.py:499-518 (rút gọn)
prune_mask = (self.get_opacity < min_opacity).squeeze()          # tập ỨNG VIÊN
if max_screen_size:
    prune_mask |= (self.max_radii2D > max_screen_size)
    prune_mask |= (self.get_scaling.max(dim=1).values > 0.1 * extent)

scores = 1 - pruning_score
remove_budget = int(0.5 * torch.sum(prune_mask))                  # trần: một nửa ứng viên
if remove_budget:
    padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
    sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
    selected_pts_mask[sampled_indices] = True
    self.prune_points(torch.logical_and(prune_mask, selected_pts_mask))   # lệnh xoá DUY NHẤT
```

Nên đọc như sau:

| | Vai trò | Chi tiết |
|---|---|---|
| `prune_mask` | **Không xoá gì cả** — chỉ là tập ứng viên | `opacity < 0.005`; cộng thêm `max_radii2D > 20 px` và `scale > 0.1·extent` **chỉ khi** `max_screen_size` khác `None`, tức chỉ từ vòng > `opacity_reset_interval` (3000) trở đi — `train.py:133`, `pipeline/trainer.py:126` |
| Lấy mẫu multinomial | Quyết định **ai** trong tập ứng viên thật sự bị xoá | Trần cứng bằng một nửa số ứng viên; trọng số $1/(10^{-6}+1-\text{Pruning}_i)$ nên Gaussian có `pruning_score` gần 1 gần như chắc chắn bị chọn. Gaussian vừa sinh ở lần densify này có trọng số 0 (nằm ngoài `scores.shape[0]`) nên miễn nhiễm |
| **`final_prune_fastgs`** | Đường xoá thứ hai, tách rời hẳn | `opacity < 0.1` **hoặc** `Pruning_i > 0.9`, xoá thẳng không lấy mẫu — `gaussian_model.py:533` |

Hai điều dễ hiểu sai:

- **Mỗi lần densify, tối đa một nửa số Gaussian "đáng xoá" bị xoá.** Nếu `remove_budget == 0` (dưới 2 ứng viên) thì không xoá gì. `if remove_budget:` là chốt chặn chia-0, **không** phải công tắc bật/tắt — comment trong code (`"The budget is not necessary for our method"`, dòng 509) cho thấy nhóm tác giả coi cơ chế trần này là phần thừa kế, nhưng nó **vẫn chạy thật** mỗi lần densify.
- **`final_prune_fastgs` chỉ chạy trong khoảng `15_000 < iteration < 30_000`, mỗi 3000 vòng** (`train.py:153`, `pipeline/trainer.py:139`). Ở đúng vòng 30000 nó **không** chạy. Ngân sách dưới 15k — kể cả mặc định `7000` của `pipeline/config.py` — **không bao giờ chạm tới đường xoá này**, nên model giao ra là model chưa tỉa cuối.

Tầng cuối là "dọn dẹp mạnh tay" ở giai đoạn hội tụ: sau ~15k vòng mô hình đã cơ bản ổn định, nên cắt aggressively những Gaussian pruning-score cao được coi là an toàn (comment ở `train.py:150-152`).

---

## Phần 4: Compact box — giảm số tile mỗi splat (`--mult`)

> Mô phỏng chạy được: `tiles_touched()` (compact box) và `tiles_touched_3dgs()` (hộp $3\sigma$ của 3DGS) trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "4.").
> Mã nguồn: `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/auxiliary.h:318-345` (`duplicateToTilesTouched`), comment trong file ghi rõ *"This is built upon Speedy-Splat"*.

3DGS gán mỗi splat 2D cho **mọi tile 16×16 mà hộp chữ nhật $3\sigma$ của nó chạm tới**. Hộp đó **lỏng** theo hai nghĩa: nó là hình chữ nhật bao quanh một ellipse nghiêng (bốn góc gần như không nhận đóng góp nào), và nó **không nhìn tới opacity** — splat mờ tịt vẫn chiếm đúng ngần ấy tile.

fastgs-lite không co hộp $3\sigma$ lại. Nó thay hẳn cách dựng hộp, kế thừa **SnugBox của Speedy-Splat**, rồi thêm một hệ số `mult` vào **ngưỡng level-set**.

### 4.1 — Hộp được dựng từ conic và opacity, không từ $3\sigma$

Rasterizer làm việc trên **conic** $M=\Sigma'^{-1}=\begin{pmatrix}A&B\B&C\end{pmatrix}$ (`con_o.x`, `con_o.y`, `con_o.z`) và opacity $o$ (`con_o.w`). Vùng cần rasterize là mặt cắt nơi splat còn đóng góp quá 1/255 mức xám:

$$o\cdot\exp\!\Big(-\tfrac12\Delta^\top M\,\Delta\Big)\ \ge\ \frac{1}{255}
\qquad\Longleftrightarrow\qquad
\Delta^\top M\,\Delta\ \le\ t,\quad t=2\ln(255\,o)$$

```c
// auxiliary.h:336-338
float t = 2.0f * log(con_o.w * 255.0f);   // level-set o·G = 1/255
t = mult * t;                             // beta trong Compact Box
```

| Ký hiệu | Ý nghĩa | Nguồn |
|---|---|---|
| $M=\Sigma'^{-1}$ | conic 2D — nghịch đảo covariance đã chiếu (công thức (5) của [tài liệu toán](gaussian-splatting-math.md)) | `con_o.x/.y/.z` |
| $o$ | opacity của Gaussian, sau sigmoid | `con_o.w` |
| $t$ | **ngưỡng bình phương khoảng cách Mahalanobis** của ellipse cần bao | `auxiliary.h:337` |
| `mult` | hệ số nhân vào $t$ — đây là toàn bộ tác dụng của `--mult` | `auxiliary.h:338` |
| $\text{disc}=B^2-AC$ | phải $<0$, nếu không ellipse suy biến và hàm trả 0 tile | `auxiliary.h:327-333` |

Hộp bao trục-song-song của ellipse $\Delta^\top M\Delta\le t$ có nửa cạnh:

$$\text{half-extent}_x=\sqrt{t\,\Sigma'_{11}}=\sqrt{t}\cdot\sigma'_x,
\qquad
\text{half-extent}_y=\sqrt{t\,\Sigma'_{22}}=\sqrt{t}\cdot\sigma'_y$$

(Trong code, `x_term`/`y_term` ở dòng 340-343 **không phải** nửa cạnh — đó là toạ độ điểm tiếp tuyến $h=\sqrt{-B^2t/(\text{disc}\cdot C)}$, dùng làm đầu vào cho `computeEllipseIntersection` để lấy ra biên chính xác.)

### 4.2 — Ba hệ quả của việc `mult` nhân vào $t$ chứ không vào cạnh

**(a) Cạnh co theo $\sqrt{\texttt{mult}}$, số tile co theo $\texttt{mult}$.**
Vì $\text{half-extent}\propto\sqrt{t}$ và diện tích $\propto t$:

| `mult` | $t$ (khi $o=1$) | Bán kính hiệu dụng | Diện tích hộp so với `mult=1` |
|---|---|---|---|
| `1.0` | $2\ln 255=11.08$ | $3.33\,\sigma'$ | 100% |
| `0.7` (Tanks&Temples, Deep Blending trong cả `train_base.sh` lẫn `train_big.sh`) | 7.76 | $2.79\,\sigma'$ | 70% |
| `0.5` (mặc định, `arguments/__init__.py:101`) | 5.54 | $2.35\,\sigma'$ | 50% |

Nên `mult=0.5` **giảm nửa** diện tích hộp, không phải giảm ba phần tư như cách hiểu "nhân thẳng vào cạnh". So với hộp $3\sigma$ của 3DGS (cùng $o=1$), `mult=0.5` cho $2.35\sigma'$ — hẹp hơn khoảng 22% mỗi chiều.

**(b) Hộp phụ thuộc opacity.** $t=2\ln(255\,o)$ nên splat càng mờ hộp càng nhỏ — điều hộp $3\sigma$ của 3DGS không làm được:

| $o$ | $t$ tại `mult=0.5` | Bán kính hiệu dụng |
|---|---|---|
| 1.0 | 5.54 | $2.35\,\sigma'$ |
| 0.5 | 4.85 | $2.20\,\sigma'$ |
| 0.1 | 3.24 | $1.80\,\sigma'$ |
| 0.01 | 0.94 | $0.97\,\sigma'$ |

> Suy ra từ công thức, chưa đo: khi $o<1/255$ thì $t<0$ và các `sqrt` ở dòng 340-343 nhận đối số âm. Ngưỡng prune `min_opacity = 0.005` (`train.py:141`) nằm ngay trên $1/255=0.0039$, nên vùng này gần như không chạm tới trong thực tế — nhưng nó không được chặn tường minh bởi kiểm tra ellipse suy biến ở dòng 330.

**(c) `mult = 1.0` KHÔNG quay về hành vi 3DGS gốc.** Nó quay về SnugBox nguyên bản của Speedy-Splat. Đáng chú ý: xét **riêng hộp bao**, ở `mult=1` và $o=1$ SnugBox cho $3.33\sigma'$, tức **rộng hơn** hộp $3\sigma$ của 3DGS — nó bao đúng tới mức 1/255 chứ không cắt ở $3\sigma$. Cái làm nên tiết kiệm không phải kích thước hộp mà là **(b)** (hộp co nhanh theo opacity) và **4.3** (lọc tile theo ellipse chính xác). Trong codebase này **không có** đường quay lại cách dựng hộp của 3DGS.

### 4.3 — Tập tile chạm không phải hình chữ nhật

Sau khi có hộp, `processTiles` (`auxiliary.h:199-315`, phần *AccuTile* của Speedy-Splat) duyệt từng **lát tile** theo trục ngắn hơn (`isY = y_span < x_span`) và với mỗi lát gọi `computeEllipseIntersection` để lấy **giao chính xác của ellipse với lát đó**. Tile nào nằm trong hộp nhưng ngoài ellipse thì bị loại luôn, không vào danh sách sort. Nên tiết kiệm thực tế lớn hơn tỉ lệ diện tích hộp ở bảng (a).

`mult` phải truyền **nhất quán cho cả train và render** — thấy rõ trong `train_base.sh` (dòng 10-13 train và 24-27 render đều `--mult 0.7`). Đường notebook không có rủi ro này: `pipeline/trainer.py:108` và `pipeline/submission.py:52` dùng chung một `cfg.mult`.

---

## Phần 5: Tách learning rate cho Spherical Harmonics

> Mô phỏng chạy được: `steps_to_converge()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "5.").
> Mã nguồn: `scene/gaussian_model.py:198-210` (`training_setup`) và `:225-244` (`optimizer_step`).

3DGS gốc cũng đã tách SH bậc thấp / bậc cao, nhưng bằng **một** tham số: `feature_lr` cho `features_dc` và `feature_lr / 20` cho `features_rest`. fastgs-lite giữ nguyên phép chia 20 đó và thêm hai thứ: **hai flag độc lập** và **một optimizer riêng có nhịp cập nhật riêng**.

### 5.1 — Hai flag, và phép chia 20 dễ bị bỏ sót

```python
# scene/gaussian_model.py:198-205
l = [ ...
      {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
      ... ]
sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]
```

| Tham số | Điều khiển | Mặc định (`arguments/__init__.py:97-98`) | **LR thực sự nạp vào Adam** |
|---|---|---|---|
| `--lowfeature_lr` | `features_dc` — SH bậc 0, màu cơ bản không phụ thuộc góc nhìn | `0.0025` | `0.0025` |
| `--highfeature_lr` | `features_rest` — SH bậc 1–3, phần màu đổi theo góc nhìn (specular, ánh kim) | `0.005` | **`0.005 / 20 = 0.00025`** |

⚠️ **Đọc kỹ cột cuối.** Con số bạn truyền vào `--highfeature_lr` bị chia 20 trước khi tới optimizer. Ở mặc định, SH bậc cao học với lr **nhỏ hơn 10 lần** SH bậc thấp. Ngay cả preset `0.02` của `train_base.sh`/`pipeline/config.py` cũng chỉ ra lr hiệu dụng `0.001`, vẫn **thấp hơn** `lowfeature_lr = 0.0025`.

Nói cách khác: trong codebase này, thành phần phụ thuộc góc nhìn **luôn được học chậm hơn** màu cơ bản. Tăng `--highfeature_lr` là để **thu hẹp** khoảng cách đó, không phải để vượt lên trên.

### 5.2 — `features_rest` còn có optimizer riêng, bước thưa hơn

`training_setup` tạo **hai** Adam: `self.optimizer` (xyz, f_dc, opacity, scaling, rotation) và `self.shoptimizer` (chỉ `f_rest`). `optimizer_step` cho chúng hai nhịp khác nhau:

| Khoảng vòng lặp | `self.optimizer` | `self.shoptimizer` (`f_rest`) |
|---|---|---|
| `iteration <= 15000` | mỗi vòng | **mỗi 16 vòng** |
| `15000 < iteration <= 20000` | mỗi 32 vòng | mỗi 32 vòng |
| `iteration > 20000` | mỗi 64 vòng | mỗi 64 vòng |

Trong giai đoạn 0–15k — nơi gần như toàn bộ việc học diễn ra — `f_rest` chỉ nhận **1/16 số bước cập nhật** so với `f_dc`. Gradient vẫn cộng dồn bình thường mỗi vòng (vì `zero_grad` chỉ gọi khi thực sự `step`), nên mỗi bước hiếm hoi đó áp một gradient đã tích luỹ; nhưng số lần đi qua đường Adam thì đúng là ít hơn 16 lần.

Hai yếu tố cộng lại — lr chia 20 và nhịp 1/16 — là lý do vệt specular hội tụ chậm, quan sát được trực tiếp trong [history-train.md](history-train.md) §1.7 (vệt sáng trên mặt tủ gỗ của `drjohnson` gần như biến mất sau 7000 vòng).

### 5.3 — Vì sao lại thiết kế như vậy

SH bậc 0 mang phần lớn năng lượng màu: lr cao ở đó làm cả ảnh dao động. SH bậc 1–3 là số hiệu chỉnh nhỏ quanh màu nền, và có 45 hệ số trên mỗi Gaussian (so với 3) — cho chúng lr đầy đủ ở nhịp đầy đủ vừa tốn thời gian optimizer vừa dễ overfit vào từng góc nhìn. Việc tách thành flag riêng cho phép **chỉnh riêng theo cảnh** thay vì bị buộc chặt vào `feature_lr` như 3DGS.

Giá trị theo cảnh trong `train_base.sh`:

| Cảnh | `--highfeature_lr` | LR hiệu dụng |
|---|---|---|
| `garden`, `room`, `counter`, `kitchen`, `bonsai` (MipNeRF360 nhiều phản xạ) | `0.02` | `0.001` |
| `truck` / `train` (Tanks&Temples) | `0.04` / `0.042` | `0.002` / `0.0021` |
| `drjohnson` / `playroom` (Deep Blending, trong nhà ít specular) | `0.0025` / `0.0015` | `0.000125` / `0.000075` |
| `bicycle`, `flowers`, `stump`, `treehill` (không truyền cờ) | mặc định `0.005` | `0.00025` |

Ngay cả `train` — giá trị cao nhất cả file — vẫn cho lr hiệu dụng `0.0021`, chỉ suýt soát `lowfeature_lr`.

> **Lưu ý về `--dense`.** Cách chia "trong nhà `0.001` / ngoài trời `0.004–0.01`" hay được nhắc kèm ở đây **không khớp `train_base.sh`**: `drjohnson` (trong nhà) dùng `--dense 0.013`, cao nhất cả file, còn `playroom` (cũng trong nhà) dùng `0.003`. Đừng chọn `--dense` theo nhãn indoor/outdoor — copy thẳng dòng của cảnh giống dataset của bạn nhất.

---

## Phần 6: Bốn đòn bẩy cộng lại thành cái gì

> Mô phỏng chạy được: `gaussian_trajectory()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "6.") — vẽ quỹ đạo số Gaussian của 3DGS so với fastgs-lite qua 30.000 vòng. Đây là **mô hình định tính bằng NumPy**, không phải số đo.

| Cơ chế | Cắt giảm cái gì | Đòn bẩy |
|---|---|---|
| Điều kiện AND khi densify (Phần 3.2) | Số Gaussian sinh ra ở vùng đã tốt / artefact cục bộ | Ít điểm hơn ⇒ forward + backward + optimizer mỗi vòng đều rẻ hơn |
| Trần xoá + `final_prune_fastgs` (Phần 3.3) | Đuôi dài các Gaussian vô ích ở nửa sau huấn luyện | Giảm điểm ⇒ giảm VRAM và thời gian mỗi vòng — **chỉ có tác dụng khi ngân sách > 15k** |
| Compact box `--mult` (Phần 4) | Số cặp (tile, Gaussian) phải sort/blend | Kernel rasterization nhanh hơn ở **mọi** vòng, cả train lẫn render |
| Nhịp optimizer thưa dần (Phần 7.1) | Số lần thật sự chạy Adam sau vòng 15k | 15k–30k gần như miễn phí về thời gian |
| Lấy mẫu 10 camera cho scoring (Phần 1) | Chi phí của chính bước scoring | Ước lượng đủ tốt mà không phải render toàn bộ tập train |

Lưu ý phép tách lr SH (Phần 5) **không** nằm trong danh sách này: như 5.1–5.2 cho thấy, nó không tăng tốc hội tụ trong cấu hình mặc định.

### So sánh: fastgs-lite vs 3DGS gốc, theo mã nguồn

Bảng dưới chỉ liệt kê những khác biệt **đọc được trực tiếp từ code trong repo này**:

| | 3DGS gốc | fastgs-lite |
|---|---|---|
| Tiêu chí densify | Chỉ gradient vị trí | Gradient **AND** `importance_score > 5` (`gaussian_model.py:494`) |
| Tín hiệu split | `densify_grad_threshold` | `grad_abs_thresh` trên gradient trị tuyệt đối, kiểu Abs-GS (`add_densification_stats`, `:527-530`) |
| Xoá Gaussian | Prune theo opacity/kích thước | Thêm lấy mẫu theo `pruning_score`, cộng `final_prune_fastgs` sau vòng 15k |
| Hộp bao khi rasterize | Chữ nhật $3\sigma$ | Compact box: ngưỡng level-set $t=\texttt{mult}\cdot 2\ln(255\,o)$ + giao ellipse–tile chính xác (Phần 4) |
| LR cho SH | Một `feature_lr` (và `/20` cho bậc cao) | Hai flag `lowfeature_lr` / `highfeature_lr` (vẫn `/20`), optimizer riêng nhịp 1/16 |
| Nhịp Adam | Mỗi vòng, suốt 30k | Mỗi vòng tới 15k, rồi 1/32, rồi 1/64 (`optimizer_step`) |

### Số đo thật có trong repo

Repo này **không** có phép đo A/B nào giữa fastgs-lite và 3DGS. Thứ duy nhất đo được là một lần chạy fastgs-lite trên Colab free T4, ghi trong [`DOCS/assets/leaderboard.csv`](assets/leaderboard.csv) và thuật lại ở [history-train.md](history-train.md):

| | Giá trị |
|---|---|
| Cấu hình | `iterations=7000`, `resolution=2`, `densification_interval=500`, T4 14.56 GB |
| Thời gian train | 327.6 s cho 4 cảnh (67–95 s/cảnh, ~85 it/s) |
| Score trung bình (LPIPS-VGG, full-res) | 0.7643 |
| Số Gaussian | 129k–201k tuỳ cảnh |
| Peak VRAM | 0.68–1.16 GB |

Đây là ngân sách 7000 vòng, tức **chưa từng chạy `final_prune_fastgs`** (Phần 3.3) — không so sánh được với bất kỳ con số published nào, vốn dùng ngân sách 30k. Muốn có số so sánh với 3DGS thì phải tự chạy cả hai cùng cấu hình trên cùng máy; repo chưa làm việc đó.

**Kết luận:** ý tưởng trung tâm là thay câu hỏi *"Gaussian này có gradient lớn không?"* bằng *"Nhiều camera có cùng đồng ý rằng chỗ này đang sai không?"*. Tín hiệu nhất quán đa góc nhìn vừa **rẻ để tính** (10 lần render phụ mỗi `densification_interval` vòng), vừa **chọn lọc hơn** — nên thêm Gaussian đúng chỗ, bỏ Gaussian đúng lúc, và compact box giữ mỗi bước rasterization gọn nhẹ.

---

## Phần 7: Ba điều mã nguồn nói mà README upstream không nói

Sáu phần trên mô tả **thiết kế**. Ba mục dưới đây là kết quả đọc mã nguồn, và chúng quyết định cách cấu hình thực tế.

### 7.1 — Lịch optimizer bị đóng đinh theo ngân sách 30.000 vòng

`scene/gaussian_model.py:225` — `optimizer_step(iteration)` không bước Adam mỗi vòng:

| Khoảng vòng lặp | Hành vi |
|---|---|
| `iteration <= 15000` (dòng 227) | bước Adam **mỗi vòng**; optimizer SH mỗi 16 vòng |
| `15000 < iteration <= 20000` (dòng 233) | cả `optimizer` lẫn `shoptimizer` bước **mỗi 32 vòng** |
| `iteration > 20000` (dòng 239) | cả hai bước **mỗi 64 vòng** |

Cùng với `densify_until_iter = 15000` và `position_lr_max_steps = 30000`: **toàn bộ chi phí nằm ở 0–15k**, còn **15k–30k gần như miễn phí** nhưng vẫn chạy `final_prune_fastgs`.

⇒ Hạ `--iterations` xuống 15k là lỗ: tiết kiệm rất ít thời gian mà mất phần tỉa cuối. Sàn hợp lý là **20000**, và khi rút ngắn phải đồng bộ `--position_lr_max_steps`, `--densify_until_iter`, cùng hai ngưỡng cứng ở dòng 227/233.

### 7.2 — `--antialiasing` là cờ chết trong fork này

`arguments/__init__.py:71` khai báo `self.antialiasing = False`, nhưng `gaussian_renderer/__init__.py` dựng `GaussianRasterizationSettings(...)` **không có** trường đó, và `submodules/diff-gaussian-rasterization_fastgs/` không tham chiếu chữ `antialiasing` ở đâu. Truyền cờ chỉ bị bỏ qua âm thầm. Muốn chống răng cưa thật phải vá công thức Mip-Splatting vào kernel CUDA.

### 7.3 — `densification_interval` là lever thời gian bị bỏ quên

Mặc định trong `arguments/__init__.py` là `100`, nhưng `train_base.sh` thật sự dùng `500`. Vì `compute_gaussian_score_fastgs` render 10 camera × 2 lượt mỗi lần gọi, khác biệt là **~145 lần gọi so với ~29 lần** trong khoảng 500–15000.

---

## Đọc tiếp

| Tài liệu | Nội dung |
|---|---|
| [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) | Mô phỏng NumPy chạy được cho từng cơ chế ở trên (`python demos/fastgs_mechanisms.py`, không cần GPU) — hàm tương ứng được ghi ngay đầu mỗi Phần |
| [`fastgs-acceleration-method.ipynb`](../fastgs-acceleration-method.ipynb) | Notebook chỉ còn phần glue tiếng Anh: dựng pipeline `pipeline/` thật trên Colab T4 (check GPU → config → load dataset → smoke test → train → analytics → submission.zip) — cần GPU |
| [colab-t4-guide.md](colab-t4-guide.md) | Hướng dẫn vận hành pipeline Colab T4 ở trên: preset, theo dõi tiến độ, xử lý sự cố |
| [history-train.md](history-train.md) | Số đo thật của các phiên train đã chạy — nguồn của mọi con số ở Phần 6 |
| [README.md](README.md) | Tổng quan repo và cách bắt đầu |
| [gaussian-splatting-math.md](gaussian-splatting-math.md) | Nền tảng toán học 3DGS mà tài liệu này giả định người đọc đã biết |

Mã nguồn tương ứng: `utils/fast_utils.py`, `scene/gaussian_model.py:198–244` và `:468–540`, `train.py:126–158`, `gaussian_renderer/__init__.py`, `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/auxiliary.h:175–345`.
