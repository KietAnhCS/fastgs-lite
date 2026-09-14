[← Mục lục](00-muc-luc.md) · Chương 12/15

# Chương 12 — Adaptive Density Control (đòn bẩy giảm N)

> Nguồn: `DOCS/Report/07-adaptive-density-control.md`, `DOCS/Report/test/07-test.md`, `DOCS/3.md`

## 12.1 Lý thuyết

# Chương 7 — Adaptive Density Control

> Khối quay ngược về 3D Gaussians: thêm Gaussian ở chỗ thiếu chi tiết, xoá Gaussian vô dụng. Đây là **đòn bẩy 1 của FastGS-lite: giảm $N$** — và cũng là đóng góp chính so với 3DGS.
> Code: `utils/fast_utils.py:33-90` (`compute_gaussian_score_fastgs`), `scene/gaussian_model.py:433-486` (`densify_and_prune_fastgs`), `:498-504` (`final_prune_fastgs`), `train.py:127-158`.
> Ví dụ số end-to-end (ảnh $4\times4$, 5 Gaussian, 3 view): [`3.md`](../3.md).

## 7.1 — Lịch chạy

| Việc | Khi nào | Điều kiện trong `train.py` |
|---|---|---|
| Tích luỹ thống kê gradient | mỗi vòng, $t<15000$ | `iteration < densify_until_iter` |
| Densify + prune | mỗi `densification_interval` (100; preset 500), $500<t<15000$ | `:132` |
| Opacity reset | mỗi 3000 vòng | `:147` |
| `final_prune_fastgs` | mỗi 3000 vòng, $15000<t<30000$ | `:153-158` |

Vòng đúng 30 000 **không** chạy final prune. Ngân sách dưới 15k không bao giờ chạm tới đường xoá này.

## 7.2 — Hai điểm số multi-view (mới so với 3DGS)

Trước mỗi lần densify/prune, `compute_gaussian_score_fastgs` lấy mẫu $V=10$ camera (`sampling_cameras`), render từng view và làm bảy bước:

### ① Sai số màu từng pixel

$$
e_v(x)=\frac13\sum_{\text{ch}\in RGB}\bigl|I^{(v)}_{\text{rend},\text{ch}}(x)-I^{(v)}_{\text{gt},\text{ch}}(x)\bigr|
$$

### ② Chuẩn hoá min–max theo ảnh

$$
\hat e_v(x)=\frac{e_v(x)-\min_xe_v}{\max_xe_v-\min_xe_v}
$$

Ngưỡng tự co theo ảnh: view "tốt" và view "tệ" đều cho ra số pixel lỗi tương đương.

### ③ Mặt nạ nhị phân

$$
m_v(x)=\mathbb 1\bigl[\hat e_v(x)>\tau_{\text{loss}}\bigr],\qquad\tau_{\text{loss}}=0.1
$$

### ④ Đổ mặt nạ về từng Gaussian

Với $\Omega^{(v)}_i$ là footprint **hữu hình** của Gaussian $i$ ở view $v$ (đã qua cull tile, $\alpha\ge1/255$, $T\ge10^{-4}$ — không phải ellipse hình học):

$$
\text{counts}^{(v)}_i=\sum_{x\in\Omega^{(v)}_i}m_v(x)
$$

Kernel làm việc này trong forward pass với cờ `DENSIFY=True`. Một pixel lỗi "tố cáo" **mọi** Gaussian đóng góp vào nó.

### ⑤ Importance score

$$
\boxed{\ \text{Importance}_i=\Bigl\lfloor\frac1V\sum_{v=1}^{V}\text{counts}^{(v)}_i\Bigr\rfloor\ }
$$

Phép chia $V$ là bản chất của "multi-view consistency": Gaussian bị tố nhiều ở một view nhưng bị che ở các view khác sẽ bị dìm xuống.

### ⑥ Photometric loss toàn ảnh

$$
E^{(v)}_{\text{photo}}=(1-\lambda)\mathcal L^{(v)}_1+\lambda\bigl(1-\text{SSIM}^{(v)}\bigr),\qquad\lambda=0.2
$$

### ⑦ Pruning score

$$
\boxed{\ \text{Pruning}_i=\text{minmax}_i\Bigl(\sum_{v=1}^{V}\text{counts}^{(v)}_i\cdot E^{(v)}_{\text{photo}}\Bigr)\ }
$$

![Importance và Pruning cho 5 Gaussian mẫu](../Report/assets/ch7_importance_pruning.png)

*Hai biểu đồ cột cho G1..G5. Trục x là tên Gaussian; trái là $\text{Importance}_i$ với ngưỡng đỏ tại 5 (chỉ G3 vượt); phải là $\text{Pruning}_i\in[0,1]$ với ngưỡng đen tại 0.9 (G3 chạm đúng 1.0) — Importance cao (đáng nhân bản) và Pruning cao (đáng xoá) không mâu thuẫn: đặt cược ở giai đoạn đầu, thanh lý nếu không cứu được ở giai đoạn cuối.*

min–max ở đây chạy **qua $N$ Gaussian** (khác ② chạy qua pixel). $\text{Pruning}_i$ cao nghĩa là **xoá**, không phải giữ.

### Ví dụ (từ `3.md`)

| | $\sum_v\text{counts}$ | $\text{Importance}$ | $\text{Pruning}$ |
|---|---|---|---|
| $G_1$ | 0 | 0 | 0.000 |
| $G_2$ | 1 | 0 | 0.057 |
| $G_3$ | 18 | **6** | **1.000** |
| $G_4$ | 11 | 3 | 0.610 |
| $G_5$ | 5 (chỉ ở 1 view) | 1 | 0.367 |

$G_5$: tệ nặng ở một view, bị che ở hai view kia → Importance $=1$, không đáng nhân bản.

## 7.3 — Densify: phép AND

Với $\bar g_i=\text{accum}_i/\text{denom}_i$, $\bar g^{\text{abs}}_i=\text{accum}^{\text{abs}}_i/\text{denom}_i$ (chương 6.3):

**3DGS gốc:**

$$
\text{clone}_i=\bigl[\lVert\bar g_i\rVert\ge\tau_{\text{grad}}\bigr]\wedge\bigl[\max s_i\le\delta\,\text{extent}\bigr]
$$

**FastGS-lite** (`gaussian_model.py:449-462`):

$$
\boxed{\ \text{clone}_i=\underbrace{\bigl[\lVert\bar g_i\rVert\ge\tau_{\text{grad}}\bigr]\wedge\bigl[\max s_i\le\delta\,\text{extent}\bigr]}_{\text{y hệt 3DGS}}\ \wedge\ \underbrace{\bigl[\text{Importance}_i>5\bigr]}_{\text{mới}}\ }
$$

$$
\boxed{\ \text{split}_i=\bigl[\lVert\bar g^{\text{abs}}_i\rVert\ge\tau^{\text{abs}}_{\text{grad}}\bigr]\wedge\bigl[\max s_i>\delta\,\text{extent}\bigr]\wedge\bigl[\text{Importance}_i>5\bigr]\ }
$$

![Vùng thoả điều kiện AND của clone](../Report/assets/ch7_clone_region.png)

*Trục x là $\lVert\bar g_i\rVert$ (chuẩn gradient tích luỹ), trục y là $\max s_i$ (scale lớn nhất). Vùng xanh lá là nơi hai điều kiện đầu của clone cùng thoả — góc dưới-phải (gradient đủ lớn, scale còn nhỏ). Điều kiện thứ ba (Importance>5) là một chiều độc lập không vẽ được trên mặt phẳng này, chỉ lọc thêm.*

| Ký hiệu | Giá trị | CLI |
|---|---|---|
| $\tau_{\text{grad}}$ | $2\times10^{-4}$ | `--grad_thresh` |
| $\tau^{\text{abs}}_{\text{grad}}$ | $1.2\times10^{-3}$ | `--grad_abs_thresh` |
| $\delta$ | $0.001$ | `--dense` |

Split dùng gradient **trị tuyệt đối** (cột 2–3, chương 6.1) — chỗ tổng có dấu bị triệt tiêu ở biên vật thể.

**Clone:** $\theta_{\text{new}}=\theta_i$ ($1\to2$).

**Split** ($j=1,2$; bản gốc bị xoá ngay sau, nên $1\to2$ chứ không phải $1\to3$):

$$
\epsilon^{(j)}\sim\mathcal N\bigl(0,\operatorname{diag}(s_{i,1}^2,s_{i,2}^2,s_{i,3}^2)\bigr),\qquad
\mu^{(j)}=\mu_i+R(q_i)\,\epsilon^{(j)},\qquad
\tilde s^{(j)}=\log\frac{s_i}{0.8\cdot2}=\log\frac{s_i}{1.6}
$$

![Lấy mẫu Gaussian con khi split](../Report/assets/ch7_split_sampling.png)

*Trục x, y là world space. Ellipse nét đứt là $1\sigma$ của Gaussian gốc; các điểm xanh nhạt là mẫu $\epsilon^{(j)}$ xoay theo $R(q_i)$; hai điểm đỏ là 2 Gaussian con $\mu^{(j)}$ thực tế sau split.*

**Vì sao phép AND là đóng góp chính:** 3DGS densify mọi Gaussian có gradient lớn. Gaussian ở vùng đã hội tụ (gradient còn dư nhưng render đã đúng) hay ở artefact chỉ thấy từ 1–2 góc đều có Importance thấp → không nhân bản. Vì densify lặp ~28–145 lần, tác động lên $N$ là luỹ thừa:

$$
N_{\text{cuối}}\approx N_0\bigl[(1+r_{\text{spawn}})(1-r_{\text{prune}})\bigr]^{n_{\text{densify}}}
$$

![Tăng trưởng N theo số lần densify](../Report/assets/ch7_n_growth.png)

*Trục x là số lần densify $n_{\text{densify}}$, trục y là $N$. Ba đường ứng với ba cặp $(r_{spawn}, r_{prune})$ — tăng/giảm luỹ thừa (đường cong lồi/lõm trên trục tuyến tính), không phải tuyến tính, vì hệ số nhân lặp lại 28–145 lần.*

## 7.4 — Prune: lấy mẫu có trọng số (không xoá thẳng)

**3DGS gốc:** xoá mọi Gaussian có $\alpha<0.005$.

**FastGS-lite** (`gaussian_model.py:464-483`):

Tập ứng viên:

$$
\mathcal C=\bigl\{i:\ \alpha_i<0.005\ \vee\ r^{2D}_i>20\text{ px}\ \vee\ \max s_i>0.1\,\text{extent}\bigr\}
$$

(hai điều kiện sau chỉ bật khi $t>3000$.)

Trọng số và lấy mẫu **trên toàn quần thể**:

$$
w_i=\frac{1}{10^{-6}+(1-\text{Pruning}_i)},\qquad
\mathcal S\sim\text{Multinomial}\bigl(w,\ \lfloor0.5\,|\mathcal C|\rfloor,\ \text{không hoàn lại}\bigr)
$$

$$
\boxed{\ \text{xoá}=\mathcal C\cap\mathcal S\ }
$$

Ba điều dễ hiểu sai:

1. $\mathcal C$ **không xoá gì cả** — chỉ là ứng viên.
2. Trọng số phân kỳ khi $\text{Pruning}_i\to1$ ($w=10^6$ so với $w=2$ ở $0.5$) → về thực chất là sắp xếp giảm dần theo Pruning rồi lấy phần đầu.
3. Lấy mẫu trên **cả $N$**, rồi mới giao với $\mathcal C$ → số xoá thật thấp hơn nhiều `remove_budget`. Gaussian vừa sinh ở lần densify này có trọng số 0 (ngoài `scores.shape[0]`) nên miễn nhiễm.

## 7.5 — Ép opacity: hai đường

Sau **mỗi** lần densify (`:485-486`), và reset cổ điển mỗi 3000 vòng (`reset_opacity`):

$$
\text{(1)}\quad\tilde\alpha_i\leftarrow\sigma^{-1}\bigl(\min(\alpha_i,\,0.8)\bigr)
\qquad\qquad
\text{(2)}\quad\tilde\alpha_i\leftarrow\sigma^{-1}\bigl(\min(\alpha_i,\,0.01)\bigr)
$$

Cả hai đều xoá trạng thái Adam của nhóm `opacity`: $(m,v)\leftarrow(0,0)$ nhưng bộ đếm `step` giữ nguyên. Ngưỡng $0.01$ ở (2) vẫn trên ngưỡng prune $0.005$, nên Gaussian sống sót một nhịp và phải tự học lại opacity.

## 7.6 — Tỉa cuối (mới so với 3DGS)

`final_prune_fastgs`, chỉ chạy trong $15000<t<30000$ mỗi 3000 vòng, xoá thẳng không lấy mẫu:

$$
\boxed{\ \text{final\_prune}_i=\bigl[\alpha_i<0.1\bigr]\ \vee\ \bigl[\text{Pruning}_i>0.9\bigr]\ }
$$

**Nghịch lý biểu kiến** (ví dụ `3.md`): $G_3$ vừa là Gaussian duy nhất được clone ở giai đoạn đầu ($\text{Importance}=6$), vừa bị xoá ở giai đoạn cuối ($\text{Pruning}=1.0$). Không mâu thuẫn — giai đoạn đầu **đặt cược** vào vùng sai nhất quán, giai đoạn cuối **thanh lý** nếu thêm chi tiết vẫn không cứu được.

## 7.7 — Đối chiếu tổng

| | 3DGS | FastGS-lite |
|---|---|---|
| Tín hiệu densify | gradient có dấu | gradient có dấu (clone) + trị tuyệt đối (split) **AND** Importance $>5$ |
| Tín hiệu prune | $\alpha<0.005$ | ứng viên $\cap$ multinomial theo Pruning |
| Tỉa cuối | không có | $\alpha<0.1\ \vee\ \text{Pruning}>0.9$ |
| Khái niệm mới | — | $\text{Importance}_i$, $\text{Pruning}_i$ |

## 7.8 — Đầu ra của khối

$$
N\leftarrow N+|\text{clone}|+|\text{split}|-|\text{xoá}|
$$

Các tensor tham số và trạng thái Adam được cắt/nối tương ứng (`cat_tensors_to_optimizer`, `_prune_optimizer`). Mũi tên quay về **3D Gaussians** (chương 2), vòng lặp tiếp tục.

## 12.2 Kiểm định số — Chương 7, ADC

# Test số Chương 7 — Adaptive Density Control

> Cảnh: [`00-scene.md`](00-scene.md) (4 điểm SfM, 3 camera, ảnh $48\times32$, $f_x=f_y=40$).
> Script: [`scripts/ch07_test.py`](scripts/ch07_test.py) (numpy + scipy, seed 0). Mọi số dưới đây là **output thật** của script.
> Lý thuyết: [`../07-adaptive-density-control.md`](../07-adaptive-density-control.md); ví dụ tay: [`../../3.md`](../../3.md).
>
> **Khác với code thật:** $V=3$ camera (cảnh chỉ có 3) thay vì $V=10$ của `sampling_cameras` (`fast_utils.py:13`). Mọi công thức giữ nguyên.

## Sơ đồ khối với số của cảnh đồ chơi (V = 3 view)

```mermaid
flowchart TD
  V["3 view: render (α = 0.1) vs GT (α = 0.9)"] --> E["① e_v(x) = ⅓ Σ_ch ∣I_rend − I_gt∣"]
  E --> N["② minmax theo ảnh → ê_v(x) ∈ [0, 1]"]
  N --> M["③ mask m_v(x) = 𝟙[ê trên 0.1]"]
  M --> C["④ counts_i = Σ m_v(x) trên footprint hữu hình Ω_i<br/>Σ_v counts: G1 3132 · G2 3053 · G3 3200 · G4 2923"]
  C --> I["⑤ Importance = ⌊mean_v counts⌋<br/>1044 / 1017 / 1066 / 974 → cả 4 trên ngưỡng 5"]
  V --> P["⑥ E_photo^(v) = 0.8·L1 + 0.2·(1 − SSIM) mỗi view"]
  C --> PR["⑦ Pruning = minmax_i(Σ_v counts · E_photo)<br/>G1 0.765 · G2 0.451 · G3 1.000 · G4 0.000"]
  P --> PR
  I --> D{"densify AND<br/>ḡ ≥ 2e−4 ✓ · Importance trên 5 ✓<br/>max s = 0.85…1.12 trên δ·extent = 0.00169 → nhánh split"}
  D --> SP["split ×4 (seed 0): ε ~ N(0, s²), μ_con = μ + R·ε<br/>s̃_con = log(s / 1.6) → s = 0.53 / 0.59 / 0.70 / 0.56<br/>8 con, 4 bản gốc xoá"]
  PR --> W["w_i = 1 / (1e−6 + 1 − Pruning)<br/>G3 ≈ 1e6 · G1 4.26 · G2 1.82 · G4 1.00"]
  W --> MN["multinomial không hoàn lại, budget ⌊0.5·∣𝒞∣⌋ = 2<br/>rút {G1, G3} → xoá = 𝒞 ∩ 𝒮"]
  SP --> OUT["N = 4 + 0 + 4 − 2 = 6 (công thức 7.8)<br/>theo thứ tự thật của code: split rồi prune → 4"]
  MN --> OUT
  PR --> FP["final_prune (15k … 30k, mỗi 3000):<br/>α dưới 0.1 ∨ Pruning trên 0.9 → xoá G3"]
  OP["ép opacity sau densify: α̃ ← σ⁻¹(min(α, 0.8))<br/>α = 0.9 → 0.8 (logit 2.197 → 1.386)<br/>reset 3000 vòng: σ⁻¹(0.01) = −4.595"] -.-> OUT
```

## Đầu vào của khối

Lấy từ chương 1–5 (tính lại trong script; gradient đối chiếu với `06-test.md` ở mục 7.3):

| $i$ | $\mu_i$ | $s_i$ (đẳng hướng) | $\tilde s_i=\log s_i$ | $\alpha_i$ mô hình | $\alpha_i$ GT | $c_i$ |
|---|---|---|---|---|---|---|
| 1 | $(0,0,0)$ | 0.8505 | $-0.1619$ | 0.1 | 0.9 | (0.8,0.2,0.2) |
| 2 | $(0.5,0.3,0.5)$ | 0.9434 | $-0.0583$ | 0.1 | 0.9 | (0.2,0.7,0.3) |
| 3 | $(-0.4,-0.2,1.0)$ | 1.1150 | $+0.1089$ | 0.1 | 0.9 | (0.1,0.3,0.9) |
| 4 | $(0.3,-0.5,0.2)$ | 0.8888 | $-0.1179$ | 0.1 | 0.9 | (0.5,0.5,0.5) |

$R_i=I$, $\Sigma_i=s_i^2I$. Renderer numpy tối giản (chương 3–5): $t=\mu-c_v$, $\mu'=$`ndc2Pix`, $J$ clamp $1.3\tan$, $\Sigma'=J\Sigma J^\top+0.3I$, conic, blend theo depth với 3 cửa loại (cull tile bằng compact box `mult`$=0.5$, $\alpha\ge1/255$, $T\ge10^{-4}$), nền trắng. Renderer trả thêm **footprint hữu hình** $\Omega_i^{(v)}$ = tập pixel Gaussian $i$ thực sự đóng góp (đúng chỗ `forward.cu:406-408` gọi `atomicAdd(metricCount)` — sau cả 3 cửa).

Projection mô hình ($\alpha=0.1$) — kết quả thật:

| view | $i$ | $\mu'$ (px) | $\Sigma'_{11},\Sigma'_{12},\Sigma'_{22}$ | depth | $r^{2D}$ | tile chạm | $\lvert\Omega_i^{(v)}\rvert$ |
|---|---|---|---|---|---|---|---|
| 1 | 1 | (23.50, 15.50) | 72.63, 0.00, 72.63 | 4.0 | 26 | 6/6 | 1248 |
| 1 | 2 | (27.94, 18.17) | 71.49, 0.52, 70.93 | 4.5 | 26 | 6/6 | 1197 |
| 1 | 3 | (20.30, 13.90) | 80.38, 0.25, 80.00 | 5.0 | 27 | 6/6 | 1301 |
| 1 | 4 | (26.36, 10.74) | 72.32, −0.61, 72.97 | 4.2 | 26 | 6/6 | 1193 |
| 2 | 1 | (8.50, 15.50) | 82.81, 0.00, 72.63 | 4.0 | 28 | 4/6 | 954 |
| 2 | 2 | (14.61, 18.17) | 74.09, −1.04, 70.93 | 4.5 | 26 | 4/6 | 1005 |
| 2 | 3 | (8.30, 13.90) | 91.36, 1.21, 80.00 | 5.0 | 29 | 4/6 | 978 |
| 2 | 4 | (12.07, 10.74) | 77.80, 2.44, 72.97 | 4.2 | 27 | 4/6 | 974 |
| 3 | 1 | (38.50, 10.50) | 82.81, −3.39, 73.76 | 4.0 | 28 | 4/6 | 930 |
| 3 | 2 | (41.28, 13.72) | 84.51, −1.39, 70.76 | 4.5 | 28 | 4/6 | 869 |
| 3 | 3 | (32.30, 9.90) | 83.72, −2.45, 81.43 | 5.0 | 28 | 4/6 | 990 |
| 3 | 4 | (40.64, 5.98) | 85.12, −7.31, 76.02 | 4.2 | 29 | 4/6 | 770 |

$T_{\text{final}}$ nhỏ nhất $\approx0.686$ — với $\alpha=0.1$ không pixel nào bị dừng sớm; cửa loại thực sự cắt footprint là compact box (level-set $\Delta^\top M\Delta\le t_i$, $t_i=0.5\cdot2\ln(25.5)=3.24$) và $\alpha\ge1/255$.

## 7.1 — Lịch chạy

Không có số để tính; giả định thời điểm $t$ của bài test: **$500\lt t<15000$, $t>3000$** (để cả ba điều kiện của tập ứng viên prune đều bật, `size_threshold = 20`). Mục 7.6 giả định thêm một lần gọi ở $15000\lt t<30000$.

## 7.2 — Hai điểm số multi-view

### ① Sai số màu từng pixel

$$e_v(x)=\tfrac13\sum_{\text{ch}}|I^{(v)}_{\text{rend}}(x)-I^{(v)}_{\text{gt}}(x)|$$

| view | $\min_x e_v$ | $\max_x e_v$ | $\text{mean}_x e_v$ |
|---|---|---|---|
| 1 | 0.006232 | 0.438925 | 0.259278 |
| 2 | 0.000000 | 0.441639 | 0.222367 |
| 3 | 0.000000 | 0.440789 | 0.204733 |

Mô hình ($\alpha=0.1$) mờ hơn GT ($\alpha=0.9$) rất nhiều nên sai số lớn gần như toàn ảnh.

### ② Chuẩn hoá min–max theo ảnh

$$\hat e_v(x)=\frac{e_v(x)-\min_xe_v}{\max_xe_v-\min_xe_v}$$

| view | mẫu số | ngưỡng $e$ tương đương $\hat e>0.1$ |
|---|---|---|
| 1 | 0.432693 | $e>0.006232+0.1\cdot0.432693=0.049502$ |
| 2 | 0.441639 | $e>0.044164$ |
| 3 | 0.440789 | $e>0.044079$ |

### ③ Mặt nạ nhị phân $m_v(x)=\mathbb 1[\hat e_v(x)>0.1]$

![Sai số màu, chuẩn hoá min-max và mặt nạ nhị phân](../Report/test/figures/ch07_error_maps.png)

*Hình: 3 view × (I_rend, I_gt, e_v(x), mặt nạ ê_v > 0.1) — mỗi view cho một tập pixel "lỗi" khác nhau dù cùng ngưỡng 0.1 nhờ chuẩn hoá theo ảnh.*


| view | số pixel bật / 1536 |
|---|---|
| 1 | **1409** |
| 2 | **1077** |
| 3 | **1042** |

(Bản đồ mask từng view in trong output script: vùng bật là hợp của các splat, phần nền trắng ở góc ảnh tắt.)

### ④ Đổ mặt nạ về từng Gaussian

$$\text{counts}^{(v)}_i=\sum_{x\in\Omega^{(v)}_i}m_v(x)$$

| | view 1 | view 2 | view 3 | $\sum_v$ | $\lvert\Omega\rvert$ (v1, v2, v3) |
|---|---|---|---|---|---|
| $G_1$ | 1248 | 954 | 930 | **3132** | 1248, 954, 930 |
| $G_2$ | 1181 | 1003 | 869 | **3053** | 1197, 1005, 869 |
| $G_3$ | 1245 | 978 | 977 | **3200** | 1301, 978, 990 |
| $G_4$ | 1180 | 973 | 770 | **2923** | 1193, 974, 770 |

Kiểm tra chéo view 1: $\sum_i\text{counts}=4854$ trong khi chỉ 1409 pixel bật — một pixel lỗi "tố cáo" mọi Gaussian đóng góp vào nó (giống nhận xét ở `3.md`).

### ⑤ Importance

![Footprint hữu hình, counts và Importance](../Report/test/figures/ch07_footprint_counts.png)

*Hình: footprint Ω_i^(1) của từng Gaussian chồng lên mặt nạ lỗi; bảng counts 4×3 theo view; Importance = ⌊mean_v counts⌋ cho cả 4 Gaussian đều vượt xa ngưỡng 5.*


$$\text{Importance}_i=\Bigl\lfloor\tfrac13\sum_v\text{counts}^{(v)}_i\Bigr\rfloor$$

| | $\sum_v$ | $/3$ | floor | $>5$? |
|---|---|---|---|---|
| $G_1$ | 3132 | 1044.00 | **1044** | ✅ |
| $G_2$ | 3053 | 1017.67 | **1017** | ✅ |
| $G_3$ | 3200 | 1066.67 | **1066** | ✅ |
| $G_4$ | 2923 | 974.33 | **974** | ✅ |

Ở cảnh này mọi Gaussian phủ gần cả ảnh và cả ảnh đều sai, nên Importance rất lớn — ngưỡng 5 không chặn ai.

### ⑥ Photometric loss toàn ảnh

$$E^{(v)}_{\text{photo}}=0.8\,\mathcal L_1^{(v)}+0.2\,(1-\text{SSIM}^{(v)})$$

SSIM cài như `utils/loss_utils.py` (cửa sổ gaussian $11\times11$, $\sigma=1.5$, zero-pad, $C_1=10^{-4}$, $C_2=9\times10^{-4}$, `scipy.ndimage.convolve`).

| view | $\mathcal L_1$ | SSIM | $0.8\mathcal L_1$ | $0.2(1-\text{SSIM})$ | $E_{\text{photo}}$ |
|---|---|---|---|---|---|
| 1 | 0.259278 | 0.647773 | 0.207422 | 0.070445 | **0.277867** |
| 2 | 0.222367 | 0.702127 | 0.177893 | 0.059575 | **0.237468** |
| 3 | 0.204733 | 0.704586 | 0.163786 | 0.059083 | **0.222869** |

### ⑦ Pruning

$$\text{Pruning}_i=\text{minmax}_i\Bigl(\sum_v\text{counts}^{(v)}_i\,E^{(v)}_{\text{photo}}\Bigr)$$

| | $\text{cnt}\cdot E_1$ | $\text{cnt}\cdot E_2$ | $\text{cnt}\cdot E_3$ | thô | Pruning |
|---|---|---|---|---|---|
| $G_1$ | 346.778 | 226.545 | 207.268 | 780.591 | $\frac{780.591-730.549}{795.932-730.549}=$ **0.7654** |
| $G_2$ | 328.161 | 238.181 | 193.673 | 760.015 | **0.4507** |
| $G_3$ | 345.945 | 232.244 | 217.743 | 795.932 | **1.0000** |
| $G_4$ | 327.883 | 231.056 | 171.609 | 730.549 | **0.0000** |

min–max chạy **qua 4 Gaussian** (khác ② chạy qua pixel).

### Bảng tổng (như ví dụ `3.md`)

![Photometric loss từng view và điểm Pruning](../Report/test/figures/ch07_scores.png)

*Hình: E_photo mỗi view; Σ_v counts·E_photo trước minmax và Pruning sau minmax (G3 đạt đúng 1.000 — ứng viên xoá mạnh nhất); trọng số w_i = 1/(1e−6+1−Pruning) trên thang log cho thấy G3 áp đảo hoàn toàn.*


| | $\sum_v\text{counts}$ | Importance | Pruning |
|---|---|---|---|
| $G_1$ | 3132 | 1044 | 0.7654 |
| $G_2$ | 3053 | 1017 | 0.4507 |
| $G_3$ | 3200 | 1066 | **1.0000** |
| $G_4$ | 2923 | 974 | 0.0000 |

## 7.3 — Densify

### Gradient $\bar g_i$, $\bar g^{\text{abs}}_i$ (tự tính bằng sai phân hữu hạn; đối chiếu với `06-test.md` ở cuối mục)

Sai phân hữu hạn trên loss $\mathcal L_1$ (chỉ L1, không có SSIM, để bài test độc lập với chương 6): dịch $\mu'$ của **một** Gaussian $\pm0.5$ px theo từng trục, render lại. Với bản đồ loss từng pixel $\ell_x=\frac{1}{HW}\text{mean}_{\text{ch}}|C(x)-I_{\text{gt}}(x)|$:

- cột có dấu: $g=\sum_x\frac{\partial\ell_x}{\partial\mu'}$ (như 3DGS);
- cột abs: $g^{\text{abs}}=\sum_x\bigl|\frac{\partial\ell_x}{\partial\mu'}\bigr|$ (trị tuyệt đối **từng pixel** rồi mới cộng — `backward.cu:589-597`).

Đổi sang đơn vị NDC như code: nhân $W/2=24$ cho trục $u$, $H/2=16$ cho trục $v$. `add_densification_stats`: $\text{accum}\mathrel{+}=\lVert g\rVert$, $\text{accum}^{\text{abs}}\mathrel{+}=\lVert g^{\text{abs}}\rVert$, $\text{denom}\mathrel{+}=1$ mỗi view (3 view ≡ 3 iteration).

| view | $i$ | $g$ (NDC) | $\lVert g\rVert$ | $g^{\text{abs}}$ | $\lVert g^{\text{abs}}\rVert$ |
|---|---|---|---|---|---|
| 1 | 1 | (+4.44e−4, −2.07e−4) | 4.90e−4 | (3.40e−2, 1.99e−2) | 3.94e−2 |
| 1 | 2 | (−5.26e−4, +1.63e−3) | 1.72e−3 | (3.22e−2, 1.92e−2) | 3.75e−2 |
| 1 | 3 | (+8.74e−4, −1.51e−3) | 1.74e−3 | (3.21e−2, 1.88e−2) | 3.73e−2 |
| 1 | 4 | (−4.41e−4, −3.04e−3) | 3.07e−3 | (2.63e−2, 1.48e−2) | 3.02e−2 |
| 2 | 1 | (−9.87e−3, −2.10e−4) | 9.87e−3 | (2.28e−2, 1.75e−2) | 2.87e−2 |
| 2 | 2 | (−2.62e−3, +1.62e−3) | 3.08e−3 | (2.69e−2, 1.83e−2) | 3.26e−2 |
| 2 | 3 | (−9.88e−3, −1.25e−3) | 9.96e−3 | (2.10e−2, 1.62e−2) | 2.65e−2 |
| 2 | 4 | (−4.32e−3, −2.84e−3) | 5.17e−3 | (2.04e−2, 1.39e−2) | 2.47e−2 |
| 3 | 1 | (+9.90e−3, −4.53e−3) | 1.09e−2 | (2.21e−2, 1.63e−2) | 2.74e−2 |
| 3 | 2 | (+1.28e−2, −2.22e−3) | 1.30e−2 | (1.95e−2, 1.58e−2) | 2.51e−2 |
| 3 | 3 | (+2.84e−3, −5.56e−3) | 6.24e−3 | (2.43e−2, 1.62e−2) | 2.92e−2 |
| 3 | 4 | (+8.21e−3, −5.48e−3) | 9.87e−3 | (1.40e−2, 9.71e−3) | 1.71e−2 |

Ví dụ view 1, $G_1$: gradient có dấu chỉ $4.9\times10^{-4}$ nhưng trị tuyệt đối $3.9\times10^{-2}$ — gấp 80 lần: Gaussian nằm giữa ảnh, nửa trái bị kéo sang trái, nửa phải sang phải, tổng có dấu triệt tiêu. Đây là lý do của cột abs.

| | accum | accum$^{\text{abs}}$ | denom | $\bar g_i$ | $\bar g^{\text{abs}}_i$ | $\ge\tau_{\text{grad}}=2\text{e−4}$ | $\ge\tau^{\text{abs}}_{\text{grad}}=1.2\text{e−3}$ |
|---|---|---|---|---|---|---|---|
| $G_1$ | 2.125e−2 | 9.550e−2 | 3 | **7.082e−3** | **3.183e−2** | ✅ | ✅ |
| $G_2$ | 1.779e−2 | 9.509e−2 | 3 | **5.931e−3** | **3.170e−2** | ✅ | ✅ |
| $G_3$ | 1.794e−2 | 9.295e−2 | 3 | **5.980e−3** | **3.098e−2** | ✅ | ✅ |
| $G_4$ | 1.810e−2 | 7.198e−2 | 3 | **6.035e−3** | **2.399e−2** | ✅ | ✅ |

**Đối chiếu với [`06-test.md`](06-test.md)** (bot 06 tính giải tích theo `backward.cu`, cùng 3 camera, cùng L1):

| | $\bar g_i$ (06, giải tích) | $\bar g_i$ (07, FD ±0.5 px) | $\bar g^{\text{abs}}_i$ (06) | $\bar g^{\text{abs}}_i$ (07) |
|---|---|---|---|---|
| $G_1$ | 6.430e−3 | 7.082e−3 | 3.037e−2 | 3.183e−2 |
| $G_2$ | 5.988e−3 | 5.931e−3 | 3.113e−2 | 3.170e−2 |
| $G_3$ | 6.233e−3 | 5.980e−3 | 3.081e−2 | 3.098e−2 |
| $G_4$ | 5.605e−3 | 6.035e−3 | 2.304e−2 | 2.399e−2 |

Chênh 1–10 % do bước sai phân 0.5 px là thô (L1 không trơn tại $|\cdot|=0$); cả hai đều cho **cùng kết luận**: 4/4 Gaussian vượt cả $\tau_{\text{grad}}$ lẫn $\tau^{\text{abs}}_{\text{grad}}$, và tỉ số abs/có dấu ≈ 5–6 sau khi trung bình 3 view (riêng view 1 của $G_1$ là ≈ 80).

### extent và nhánh clone / split

![Sơ đồ quyết định densify: gradient vs kích thước](../Report/test/figures/ch07_densify_decision.png)

*Hình: trục ḡ (và ḡ_abs) so với hai ngưỡng τ_grad/τ_abs, trục max s_i so với δ·extent — cả 4 Gaussian rơi vào góc split (gradient đủ lớn + đã to hơn ngưỡng) và đều có Importance > 5 nên điều kiện AND được thoả.*


$$\bar c=\tfrac13\sum_v c_v=(0,\ 0.1667,\ -4),\qquad \lVert c_v-\bar c\rVert=(0.1667,\ 1.5092,\ 1.5366)$$
$$\text{extent}=1.1\cdot1.5366=1.690250,\qquad \delta\,\text{extent}=0.001690,\qquad 0.1\,\text{extent}=0.169025$$

| | $\max s_i$ | $\le\delta\,\text{extent}$? | nhánh | gradient đạt | 3DGS gốc | Importance $>5$ | **FastGS-lite** |
|---|---|---|---|---|---|---|---|
| $G_1$ | 0.8505 | ❌ | split | ✅ $3.18\text{e−2}\ge1.2\text{e−3}$ | split | ✅ 1044 | **SPLIT** |
| $G_2$ | 0.9434 | ❌ | split | ✅ | split | ✅ 1017 | **SPLIT** |
| $G_3$ | 1.1150 | ❌ | split | ✅ | split | ✅ 1066 | **SPLIT** |
| $G_4$ | 0.8888 | ❌ | split | ✅ | split | ✅ 974 | **SPLIT** |

$\delta\,\text{extent}=0.0017$ nhỏ hơn mọi scale khởi tạo ($\approx0.85$–$1.1$, vì 4 điểm cách nhau xa) → không ai vào nhánh clone. Với cảnh này phép AND không chặn gì (Importance đều $\gg5$); 3DGS gốc và FastGS-lite cho cùng quyết định: **clone 0, split 4**. Phép AND chỉ khác biệt khi có Gaussian ở vùng đã đúng hoặc chỉ sai ở 1–2 view (như $G_1,G_5$ ở `3.md`).

### Split cụ thể (seed 0)

![Split: 4 Gaussian gốc → 8 con, xoá 2](../Report/test/figures/ch07_split.png)

*Hình: 4 Gaussian gốc (vòng tròn bán kính s) và 8 con sau split (bán kính s/1.6, lệch theo ε ~ N(0, s²) seed 0); hai con {G1c1, G3c1} bị multinomial rút bỏ (đánh dấu X), N: 4 → 8 → 6.*


$$\epsilon^{(j)}\sim\mathcal N(0,\operatorname{diag}(s_i^2)),\quad \mu^{(j)}=\mu_i+R\epsilon^{(j)}\ (R=I),\quad \tilde s^{(j)}=\log\frac{s_i}{0.8\cdot2}=\log\frac{s_i}{1.6}$$

| gốc | $\epsilon^{(1)}$ | $\mu^{(1)}$ | $\epsilon^{(2)}$ | $\mu^{(2)}$ | $\tilde s^{(j)}$ | $s^{(j)}$ |
|---|---|---|---|---|---|---|
| $G_1$ | (0.1069, −0.1124, 0.5447) | (0.1069, −0.1124, 0.5447) | (0.0892, −0.4556, 0.3075) | (0.0892, −0.4556, 0.3075) | $\log(0.8505/1.6)=-0.6319$ | 0.5316 |
| $G_2$ | (1.2302, 0.8935, −0.6639) | (1.7302, 1.1935, −0.1639) | (−1.1938, −0.5880, 0.0390) | (−0.6938, −0.2880, 0.5390) | $-0.5283$ | 0.5896 |
| $G_3$ | (−2.5925, −0.2440, −1.3893) | (−2.9925, −0.4440, −0.3893) | (−0.8165, −0.6069, −0.3527) | (−1.2165, −0.8069, 0.6473) | $-0.3611$ | 0.6969 |
| $G_4$ | (0.3659, 0.9266, −0.1142) | (0.6659, 0.4266, 0.0858) | (1.2145, −0.5912, 0.3124) | (1.5145, −1.0912, 0.5124) | $-0.5879$ | 0.5555 |

Bản gốc bị xoá ngay sau split → mỗi split ròng $+1$. Clone (không có ở đây): $\theta_{\text{new}}=\theta_i$.

### Luỹ thừa $N$ (cuối 7.3)

$$N_{\text{cuối}}\approx N_0\bigl[(1+r_{\text{spawn}})(1-r_{\text{prune}})\bigr]^{n},\qquad (1.15)(0.95)=1.0925$$

| $N_0$ | $n$ | $1.0925^n$ | $N_{\text{cuối}}$ |
|---|---|---|---|
| 4 | 28 | 11.91 | **47.6** |
| 4 | 145 | $3.72\times10^5$ | $1.49\times10^6$ |
| 100 000 | 28 | 11.91 | $1.19\times10^6$ |
| 100 000 | 145 | $3.72\times10^5$ | $3.72\times10^{10}$ |

Cùng $r_{\text{spawn}},r_{\text{prune}}$, chỉ khác số lần densify (interval 500 → 28 lần, interval 100 → 145 lần) mà $N$ lệch nhau 4 bậc — vì vậy giảm $r_{\text{spawn}}$ qua phép AND có tác động luỹ thừa.

## 7.4 — Prune có trọng số

### Tập ứng viên (theo công thức chương 7, tính trên 4 Gaussian gốc)

$$\mathcal C=\{i:\ \alpha_i<0.005\ \vee\ r^{2D}_i>20\ \vee\ \max s_i>0.1\,\text{extent}\}$$

$r^{2D}_i=\max_v\lceil3\sqrt{\lambda_{\max}(\Sigma'^{(v)}_i)}\rceil$ (= `max_radii2D`):

| | $\alpha$ | $<0.005$ | $r^{2D}$ | $>20$ | $\max s$ | $>0.169$ | $\in\mathcal C$ |
|---|---|---|---|---|---|---|---|
| $G_1$ | 0.1 | ❌ | 28 | ✅ | 0.8505 | ✅ | ✅ |
| $G_2$ | 0.1 | ❌ | 28 | ✅ | 0.9434 | ✅ | ✅ |
| $G_3$ | 0.1 | ❌ | 29 | ✅ | 1.1150 | ✅ | ✅ |
| $G_4$ | 0.1 | ❌ | 29 | ✅ | 0.8888 | ✅ | ✅ |

- $t\le3000$ (`size_threshold=None`): $\mathcal C=\{\alpha<0.005\}=\varnothing$ → `remove_budget = 0`, không xoá.
- $t>3000$: $\mathcal C=\{1,2,3,4\}$, $\text{remove\_budget}=\lfloor0.5\cdot4\rfloor=2$.

$$w_i=\frac{1}{10^{-6}+(1-\text{Pruning}_i)}=(4.2621,\ 1.8204,\ 10^6,\ 1.0000),\qquad p_i=w_i/\textstyle\sum w=(4\text{e−6},\ 2\text{e−6},\ 0.999993,\ 1\text{e−6})$$

Multinomial không hoàn lại (rút lần lượt theo $w/\sum w$ trên phần còn lại), seed 0: $\mathcal S=\{1,3\}$ → **xoá $=\mathcal C\cap\mathcal S=\{G_1,G_3\}$**.

1000 lần rút (seed 0): tần suất bị xoá $=(0.618,\ 0.249,\ \mathbf{1.000},\ 0.133)$; số xoá trung bình $2.000$ = budget (vì $\mathcal C$ là toàn bộ $N$). $G_3$ ($\text{Pruning}=1$, $w=10^6$) chắc chắn bị rút ở lượt đầu; lượt hai chia theo $4.26:1.82:1.00$.

### Thứ tự thật trong `densify_and_prune_fastgs` (đối chiếu code)

Code split **trước**, rồi mới tính `prune_mask` trên quần thể mới, và `padded_importance[:scores.shape[0]]` gán trọng số cũ **theo chỉ số** không đánh lại (`gaussian_model.py:473-476`). Với cảnh này cả 4 gốc đều split nên quần thể mới là 8 con (thứ tự `repeat(2,1)`: $G_1c_1,G_2c_1,G_3c_1,G_4c_1,G_1c_2,\dots$), `max_radii2D` của con $=0$, $\max s=s/1.6\in[0.53,0.70]>0.169$ → $|\mathcal C|=8$, budget $=4$, trọng số $=(4.26,1.82,10^6,1.0,0,0,0,0)$ → $\mathcal S=\{1,2,3,4\}$ (đúng 4 phần tử có trọng số dương) → xoá $G_1c_1,G_2c_1,G_3c_1,G_4c_1$ → $N=8-4=4$. Đây là hệ quả của việc điểm số không được đánh chỉ số lại sau densify; trong huấn luyện thật tỉ lệ split nhỏ nên lệch ít.

### Kịch bản GIẢ ĐỊNH $N=10$

![Prune multinomial và luỹ thừa số Gaussian](../Report/test/figures/ch07_prune_multinomial.png)

*Hình: kịch bản N=10 minh hoạ trọng số w_i lệch mạnh theo Pruning, tần suất bị xoá qua 1000 lần lấy mẫu (budget 2, xoá thật trung bình 1.879); subplot N_cuối ≈ N0[(1+r_spawn)(1−r_prune)]^n tăng theo cấp số nhân với n, đánh dấu n=28 và n=145.*


$\text{Pruning}=(0,0.05,0.2,0.4,0.6,0.8,0.9,0.95,0.99,1.0)$, $\mathcal C=\{3,6,8,9,10\}$, budget $=\lfloor0.5\cdot5\rfloor=2$.

| $i$ | Pruning | $w_i$ | $p_i$ | $\in\mathcal C$ | tần suất $\in\mathcal S$ | **tần suất bị xoá** |
|---|---|---|---|---|---|---|
| 1 | 0.00 | 1.0000 | 1e−6 | | 0.008 | 0 |
| 2 | 0.05 | 1.0526 | 1e−6 | | 0.013 | 0 |
| 3 | 0.20 | 1.2500 | 1e−6 | ✅ | 0.010 | 0.010 |
| 4 | 0.40 | 1.6667 | 2e−6 | | 0.018 | 0 |
| 5 | 0.60 | 2.5000 | 2e−6 | | 0.014 | 0 |
| 6 | 0.80 | 5.0000 | 5e−6 | ✅ | 0.035 | 0.035 |
| 7 | 0.90 | 9.9999 | 1e−5 | | 0.068 | 0 |
| 8 | 0.95 | 19.9996 | 2e−5 | ✅ | 0.136 | 0.136 |
| 9 | 0.99 | 99.9900 | 1e−4 | ✅ | 0.698 | 0.698 |
| 10 | 1.00 | $10^6$ | 0.999858 | ✅ | 1.000 | 1.000 |

1000 lần seed 0: số xoá thật trung bình **1.879** so với budget 2 (min 1, max 2). Gaussian 10 luôn bị rút; lượt hai rơi vào 9 (70%), 8 (14%), 7 (7% — nhưng 7 $\notin\mathcal C$ nên không xoá), … Minh hoạ đúng ba ý của 7.4: $\mathcal C$ không xoá gì; $w$ phân kỳ khi $\text{Pruning}\to1$ (thực chất là sắp xếp giảm dần); lấy mẫu trên cả $N$ rồi mới giao với $\mathcal C$ nên số xoá thật $<$ budget.

## 7.5 — Ép opacity

| $\alpha$ | $\tilde\alpha$ trước $=\sigma^{-1}(\alpha)$ | (1) $\sigma^{-1}(\min(\alpha,0.8))$ | (2) $\sigma^{-1}(\min(\alpha,0.01))$ |
|---|---|---|---|
| 0.1 | $-2.197225$ | $\min=0.1\to-2.197225$ (không đổi) | $\min=0.01\to-4.595120$ |
| 0.9 | $+2.197225$ | $\min=0.8\to+1.386294$ | $\min=0.01\to-4.595120$ |

Với mô hình ($\alpha=0.1$) đường (1) sau mỗi densify không làm gì; đường (2) `reset_opacity` kéo mọi Gaussian về $0.01$ — vẫn trên ngưỡng prune $0.005$.

## 7.6 — Tỉa cuối

$$\text{final\_prune}_i=[\alpha_i<0.1]\ \vee\ [\text{Pruning}_i>0.9]$$

| | $\alpha$ | $<0.1$ | Pruning | $>0.9$ | kết quả |
|---|---|---|---|---|---|
| $G_1$ | 0.1 | ❌ (đúng biên, so sánh chặt) | 0.7654 | ❌ | giữ |
| $G_2$ | 0.1 | ❌ | 0.4507 | ❌ | giữ |
| $G_3$ | 0.1 | ❌ | **1.0000** | ✅ | **XOÁ** |
| $G_4$ | 0.1 | ❌ | 0.0000 | ❌ | giữ |

$\alpha=0.1$ nằm **đúng trên biên**: `get_opacity < 0.1` là so sánh chặt nên không xoá; chỉ cần opacity học xuống $0.0999$ là cả 4 bị xoá.

Kịch bản $N=10$ (giả định $\alpha=0.5$ cho mọi Gaussian): xoá $\{8,9,10\}$ (Pruning $0.95,0.99,1.0$); Gaussian 7 có Pruning $=0.9$ **không** $>0.9$ → giữ.

Nghịch lý biểu kiến như `3.md`: $G_3$ vừa được split ở 7.3 (Importance 1066, cao nhất), vừa là ứng viên xoá chắc chắn ở 7.4 ($w=10^6$) và bị tỉa thẳng ở 7.6 ($\text{Pruning}=1$).

## 7.7 — Đối chiếu

| | 3DGS | FastGS-lite (cảnh này) |
|---|---|---|
| Densify | split 4 | split 4 (AND không chặn vì Importance $\gg5$) |
| Prune ($t>3000$) | $\alpha<0.005$ → $\varnothing$ | $\mathcal C=\{1,2,3,4\}$, budget 2, xoá $\{G_1,G_3\}$ (seed 0) |
| Tỉa cuối | không có | xoá $G_3$ |

## 7.8 — Đầu ra của khối

$$N\leftarrow N+|\text{clone}|+|\text{split}|-|\text{xoá}|=4+0+4-2=\mathbf 6$$

(Theo thứ tự thực thi thật của code — split rồi prune trên quần thể mới với trọng số theo chỉ số cũ — kết quả là $8-4=4$; xem 7.4.)

| Đại lượng | Giá trị |
|---|---|
| $N$ mới (công thức 7.8) | **6** |
| clone | $\varnothing$ |
| split | $G_1,G_2,G_3,G_4$ → 8 con (bảng 7.3) |
| xoá (multinomial, seed 0) | $G_1,G_3$ |
| Opacity sau densify | $\tilde\alpha=\sigma^{-1}(\min(0.1,0.8))=-2.1972$ (không đổi) |

Danh sách Gaussian sau ADC theo công thức 7.8 (con của $G_1,G_3$ vẫn tồn tại vì xoá áp lên chỉ số gốc, bản gốc đã bị thay bởi con khi split; 6 = 8 con − 2 chỉ số bị rút):

| Gaussian | $\mu$ | $\tilde s$ | $s$ | $\tilde\alpha$ |
|---|---|---|---|---|
| $G_2c_1$ | (1.7302, 1.1935, −0.1639) | −0.5283 | 0.5896 | −2.1972 |
| $G_2c_2$ | (−0.6938, −0.2880, 0.5390) | −0.5283 | 0.5896 | −2.1972 |
| $G_4c_1$ | (0.6659, 0.4266, 0.0858) | −0.5879 | 0.5555 | −2.1972 |
| $G_4c_2$ | (1.5145, −1.0912, 0.5124) | −0.5879 | 0.5555 | −2.1972 |
| $G_1c_2$ | (0.0892, −0.4556, 0.3075) | −0.6319 | 0.5316 | −2.1972 |
| $G_3c_2$ | (−1.2165, −0.8069, 0.6473) | −0.3611 | 0.6969 | −2.1972 |

(Quy ước gộp: hai chỉ số bị rút $\{1,3\}$ ứng với $G_1c_1,G_3c_1$ trong thứ tự `repeat(2,1)` — trùng với kết quả 7.4b ở hai phần tử đó; 7.4b xoá thêm $G_2c_1,G_4c_1$ vì budget tăng lên 4.) Mũi tên quay về **3D Gaussians** (chương 2); chương 8 nhận $N=6$.

## 12.3 Ví dụ số end-to-end (3.md)

# Phần III — Bài toán end-to-end: từ pixel thô đến quyết định densify/prune

> Tài liệu này là ví dụ số **chạy xuyên suốt** cho §3.1–§3.4 của [`fastgs-acceleration-method.md`](fastgs-acceleration-method.md).
> Mọi con số dưới đây tự nhất quán và kiểm tra tay được.

**Quy ước:** ảnh $4\times4$ (16 pixel, $P_1..P_{16}$ theo hàng), $P=5$ Gaussian, $K=3$ góc nhìn (code thật dùng $K=10$ — `fast_utils.py:13`; lấy 3 để tính tay được, mọi công thức giữ nguyên). Tham số đúng như repo: $\tau_{\text{loss}}=0.1$ (`arguments/__init__.py:90`), $\lambda=0.2$ (`fast_utils.py:30`), $\tau_d=5$ (`gaussian_model.py:459`), ngưỡng prune $0.9$ (`gaussian_model.py:503`).

---

## INPUT

### A. Ảnh view 1 — RGB thực, thang $[0,1]$

| Pixel | render $r^1$ | ground-truth $g^1$ | | Pixel | render $r^1$ | ground-truth $g^1$ |
|---|---|---|---|---|---|---|
| $P_1$ | 0.48, 0.51, 0.47 | 0.50, 0.50, 0.50 | | $P_9$ | 0.32, 0.30, 0.29 | 0.30, 0.31, 0.29 |
| $P_2$ | 0.43, 0.38, 0.44 | 0.42, 0.40, 0.44 | | $P_{10}$ | 0.42, 0.44, 0.49 | 0.45, 0.44, 0.46 |
| $P_3$ | 0.57, 0.60, 0.61 | 0.60, 0.58, 0.62 | | $P_{11}$ | 0.39, 0.41, 0.40 | 0.72, 0.68, 0.64 |
| $P_4$ | 0.37, 0.34, 0.36 | 0.35, 0.36, 0.34 | | $P_{12}$ | 0.39, 0.40, 0.40 | 0.75, 0.70, 0.64 |
| $P_5$ | 0.54, 0.54, 0.56 | 0.55, 0.53, 0.57 | | $P_{13}$ | 0.39, 0.35, 0.39 | 0.38, 0.37, 0.39 |
| $P_6$ | 0.48, 0.50, 0.49 | 0.48, 0.47, 0.49 | | $P_{14}$ | 0.43, 0.41, 0.43 | 0.41, 0.42, 0.40 |
| $P_7$ | 0.40, 0.42, 0.41 | 0.70, 0.66, 0.62 | | $P_{15}$ | 0.45, 0.44, 0.43 | 0.66, 0.62, 0.58 |
| $P_8$ | 0.40, 0.40, 0.40 | 0.68, 0.60, 0.58 | | $P_{16}$ | 0.35, 0.35, 0.34 | 0.80, 0.74, 0.70 |

### B. View 2 và view 3 — cho sẵn ở dạng $e_{u,v}$ (đã qua ① để bảng gọn)

| | $P_1$ | $P_2$ | $P_3$ | $P_4$ | $P_5$ | $P_6$ | $P_7$ | $P_8$ | $P_9$ | $P_{10}$ | $P_{11}$ | $P_{12}$ | $P_{13}$ | $P_{14}$ | $P_{15}$ | $P_{16}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $e^2$ | .01 | .02 | .01 | .02 | .02 | .01 | **.13** | **.11** | .02 | .01 | **.15** | **.14** | .01 | .02 | **.12** | **.16** |
| $e^3$ | .02 | .01 | .02 | .03 | .01 | .02 | **.19** | **.17** | .02 | **.12** | **.21** | **.20** | .01 | .02 | **.16** | **.24** |

### C. Footprint hữu hình $\Omega_i^j$

Đây là $\Omega$ **theo nghĩa của code** (§3.1④): đã qua cull tile, có $\alpha\ge1/255$, và pixel chưa bão hoà $T\ge10^{-4}$ — không phải ellipse hình học.

| | view 1 | view 2 | view 3 |
|---|---|---|---|
| $G_1$ | $\{1,2,5,6\}$ | $\{1,2,5,6\}$ | $\{1,2,5,6\}$ |
| $G_2$ | $\{9,10,13,14\}$ | $\{9,10,13,14\}$ | $\{9,10,13,14\}$ |
| $G_3$ | $\{7,8,11,12,15,16\}$ | $\{7,8,11,12,15,16\}$ | $\{7,8,11,12,15,16\}$ |
| $G_4$ | $\{10,11,12,14,15,16\}$ | $\{11,12,15,16\}$ | $\{11,12,16\}$ |
| $G_5$ | $\{3,4,7,8,12,15,16\}$ | $\varnothing$ **(bị che, $T<10^{-4}$)** | $\varnothing$ **(bị che)** |

### D. SSIM mỗi view (input của ⑥) và thuộc tính Gaussian (input của §3.3–3.4)

$$\mathcal{L}^1_{SSIM}=0.72,\qquad \mathcal{L}^2_{SSIM}=0.88,\qquad \mathcal{L}^3_{SSIM}=0.80$$

| | $\lVert\bar g_i\rVert$ | $\lVert\bar g^{abs}_i\rVert$ | $\max(\text{scale})$ vs $\delta\cdot\text{extent}$ | opacity |
|---|---|---|---|---|
| $G_1$ | 0.00025 | 0.00040 | nhỏ hơn | 0.85 |
| $G_2$ | 0.00008 | 0.00030 | nhỏ hơn | 0.62 |
| $G_3$ | 0.00031 | 0.00055 | nhỏ hơn | 0.40 |
| $G_4$ | 0.00011 | 0.00150 | **lớn hơn** | **0.07** |
| $G_5$ | 0.00042 | 0.00070 | nhỏ hơn | 0.55 |

---

## ① Sai số màu từng pixel — công thức (6)

$$e^j_{u,v}=\frac{1}{3}\sum_{c'=1}^{3}\bigl|\,r^{\,j,c'}_{u,v}-g^{\,j,c'}_{u,v}\,\bigr|$$

$$e^1_{P_7}=\tfrac13(0.30+0.24+0.21)=\tfrac{0.75}{3}=0.25,
\qquad
e^1_{P_{16}}=\tfrac13(0.45+0.39+0.36)=\tfrac{1.20}{3}=0.40$$

| | $P_1$ | $P_2$ | $P_3$ | $P_4$ | $P_5$ | $P_6$ | $P_7$ | $P_8$ | $P_9$ | $P_{10}$ | $P_{11}$ | $P_{12}$ | $P_{13}$ | $P_{14}$ | $P_{15}$ | $P_{16}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $e^1$ | .02 | .01 | .02 | .02 | .01 | .01 | .25 | .22 | .01 | .02 | .28 | .30 | .01 | .02 | .18 | **.40** |

## ② Chuẩn hoá min–max — công thức (7)

$$\mathcal{N}(e_{u,v})=\frac{e_{u,v}-e_{\min}}{e_{\max}-e_{\min}}\qquad(\text{min/max lấy trên toàn ảnh của riêng view } j)$$

| view | $e_{\min}$ | $e_{\max}$ | mẫu số |
|---|---|---|---|
| 1 | 0.01 | 0.40 | 0.39 |
| 2 | 0.01 | 0.16 | 0.15 |
| 3 | 0.01 | 0.24 | 0.23 |

View 1: $\mathcal{M}^1_{P_7}=\frac{0.24}{0.39}=0.6154$, $\mathcal{M}^1_{P_{11}}=0.6923$, $\mathcal{M}^1_{P_{12}}=0.7436$, $\mathcal{M}^1_{P_{15}}=0.4359$, $\mathcal{M}^1_{P_{16}}=1.0$; mọi pixel nền $\le 0.0256$.

## ③ Ngưỡng hoá nhị phân — công thức (8)

$$\mathcal{M}^j_{mask}=\mathbb{I}\bigl(\mathcal{M}^j>0.1\bigr)$$

Ngưỡng $0.1$ trên $\mathcal{M}$ tương đương ngưỡng tuyệt đối $e > e_{\min}+0.1\,(e_{\max}-e_{\min})$:

| view | ngưỡng $e$ tương đương | pixel bị đánh dấu | số lượng |
|---|---|---|---|
| 1 | $>0.049$ | $P_7,P_8,P_{11},P_{12},P_{15},P_{16}$ | 6 |
| 2 | $>0.025$ | $P_7,P_8,P_{11},P_{12},P_{15},P_{16}$ | 6 |
| 3 | $>0.033$ | $P_7,P_8,P_{10},P_{11},P_{12},P_{15},P_{16}$ | 7 |

Đây là chỗ min–max phát huy tác dụng: view 2 "tốt hơn hẳn" ($e_{\max}=0.16$ so với $0.40$) nhưng vẫn cho ra đúng 6 pixel lỗi — ngưỡng tự co theo ảnh.

## ④ Đổ mặt nạ về từng Gaussian — phần trong công thức (9)

$$\text{counts}^{\,j}_i=\sum_{p\in\Omega_i}\mathbb{I}\bigl(\mathcal{M}^j_{mask}(p)=1\bigr)$$

Ma trận `counts` kích thước $5\times3$ — **đây là output của §3.1**:

| | view 1 | view 2 | view 3 | $\sum_j$ |
|---|---|---|---|---|
| $G_1$ | 0 | 0 | 0 | 0 |
| $G_2$ | 0 | 0 | 1 | 1 |
| $G_3$ | **6** | **6** | **6** | **18** |
| $G_4$ | 4 | 4 | 3 | 11 |
| $G_5$ | 5 | 0 | 0 | 5 |

Kiểm tra chéo view 1: $0+0+6+4+5=15$ trong khi chỉ có 6 pixel lỗi. Không mâu thuẫn — $P_{16}$ nằm dưới cả $G_3,G_4,G_5$ nên "tố cáo" cả ba.

## ⑤ Importance score — công thức (9), phần ngoài

$$s^i_d=\Bigl\lfloor\ \frac{1}{K}\sum_{j=1}^{K}\text{counts}^{\,j}_i\ \Bigr\rfloor,\qquad K=3$$

| | $\sum_j$ | $\sum_j/3$ | $s_d$ (floor) | $>5$? |
|---|---|---|---|---|
| $G_1$ | 0 | 0 | **0** | ❌ |
| $G_2$ | 1 | 0.33 | **0** | ❌ |
| $G_3$ | 18 | 6.00 | **6** | ✅ |
| $G_4$ | 11 | 3.67 | **3** | ❌ |
| $G_5$ | 5 | 1.67 | **1** | ❌ |

$G_5$ là ca kinh điển: ở view 1 nó bị tố 5 pixel — **nhiều hơn $G_4$** — nhưng hai view kia nó bị che nên $\Omega_5=\varnothing$. Phép chia $K$ dìm nó xuống 1. Đó đúng là ý nghĩa "multi-view consistency".

## ⑥ Photometric loss toàn ảnh — công thức (10)

$$E^j_{photo}=(1-\lambda)\,\mathcal{L}^j_1+\lambda\bigl(1-\mathcal{L}^j_{SSIM}\bigr),\qquad \lambda=0.2$$

$\mathcal{L}^j_1$ là trung bình $e^j$ trên 16 pixel:

| view | $\sum e^j$ | $\mathcal{L}_1=\sum/16$ | $0.8\,\mathcal{L}_1$ | $0.2(1-\text{SSIM})$ | $E_{photo}$ |
|---|---|---|---|---|---|
| 1 | 1.78 | 0.111250 | 0.089000 | $0.2(0.28)=0.056$ | **0.145000** |
| 2 | 0.96 | 0.060000 | 0.048000 | $0.2(0.12)=0.024$ | **0.072000** |
| 3 | 1.45 | 0.090625 | 0.072500 | $0.2(0.20)=0.040$ | **0.112500** |

## ⑦ Pruning score — công thức (11)

$$s^i_p=\mathcal{N}\left(\sum_{j=1}^{K}\text{counts}^{\,j}_i\cdot E^j_{photo}\right)$$

| | $\text{cnt}\times0.145$ | $\text{cnt}\times0.072$ | $\text{cnt}\times0.1125$ | thô | $s_p=\dfrac{\text{thô}-0}{1.977}$ |
|---|---|---|---|---|---|
| $G_1$ | 0 | 0 | 0 | **0.0000** | **0.0000** |
| $G_2$ | 0 | 0 | 0.1125 | **0.1125** | **0.0569** |
| $G_3$ | 0.8700 | 0.4320 | 0.6750 | **1.9770** | **1.0000** |
| $G_4$ | 0.5800 | 0.2880 | 0.3375 | **1.2055** | **0.6098** |
| $G_5$ | 0.7250 | 0 | 0 | **0.7250** | **0.3667** |

min–max ở đây chạy **qua 5 Gaussian** (khác ② chạy qua pixel trong một ảnh): $s_{\min}=0$, $s_{\max}=1.977$.

---

## OUTPUT của Phần III

$$s_d=[\,0,\;0,\;\mathbf{6},\;3,\;1\,],\qquad s_p=[\,0.000,\;0.057,\;\mathbf{1.000},\;0.610,\;0.367\,]$$

### §3.3 — Densify (giai đoạn $500<i<15000$): AND hai điều kiện

| | gradient đạt? | loại | $s_d>5$? | **Kết quả** |
|---|---|---|---|---|
| $G_1$ | ✅ $0.00025\ge0.0002$ | clone | ❌ 0 | **chặn** — 3DGS gốc *sẽ* clone nó, fastgs thì không |
| $G_2$ | ❌ $0.00008$ | — | ❌ 0 | không |
| $G_3$ | ✅ $0.00031\ge0.0002$, scale nhỏ | clone | ✅ 6 | ✅ **CLONE, $1\to2$** |
| $G_4$ | ✅ $0.00150\ge0.0012$, scale lớn | split | ❌ 3 | **chặn** |
| $G_5$ | ✅ $0.00042$ | clone | ❌ 1 | **chặn** — artefact chỉ thấy ở 1 view |

$$N:\ 5 \longrightarrow 6$$

Ba Gaussian có gradient hợp lệ bị phép AND chặn lại — đó chính là cơ chế giữ cho $N$ không phình.

Nhánh prune trong cùng hàm `densify_and_prune_fastgs`: tập ứng viên $=\{\text{opacity}<0.005\}=\varnothing$ ⟹ `remove_budget = 0` ⟹ không xoá ai.

### §3.4 — `final_prune_fastgs` (giai đoạn $15000<i<30000$): opacity $<0.1$ **HOẶC** $s_p>0.9$

| | opacity $<0.1$ | $s_p>0.9$ | **Kết quả** |
|---|---|---|---|
| $G_1$ | ❌ 0.85 | ❌ 0.000 | giữ |
| $G_2$ | ❌ 0.62 | ❌ 0.057 | giữ |
| $G_3$ | ❌ 0.40 | ✅ **1.000** | 🗑️ **XOÁ** |
| $G_4$ | ✅ **0.07** | ❌ 0.610 | 🗑️ **XOÁ** |
| $G_5$ | ❌ 0.55 | ❌ 0.367 | giữ |

> **Nghịch lý biểu kiến, đúng như §3.2 mô tả.** $G_3$ vừa là Gaussian *duy nhất* được nhân bản ở giai đoạn đầu ($s_d=6$), vừa là Gaussian bị xoá ở giai đoạn cuối ($s_p=1.0$). Không mâu thuẫn — giai đoạn đầu **đặt cược** vào vùng sai nhất quán, giai đoạn sau **thanh lý** khoản cược nếu thêm chi tiết vẫn không cứu được.
>
> $G_5$ thì ngược lại: tệ rất nặng ở một view nên không đáng nhân bản ($s_d=1$), nhưng $s_p=0.367$ vẫn chưa đủ cao để bị xoá ở lượt này.

---

## Bảng tổng kết luồng dữ liệu

| # | Công thức | Vào | Ra | Trục gộp |
|---|---|---|---|---|
| ① | $e=\frac13\sum_{RGB}\lvert r-g\rvert$ | 2 ảnh $3\times4\times4$ | bản đồ $4\times4$ | kênh màu |
| ② | $\mathcal{M}=\frac{e-e_{\min}}{e_{\max}-e_{\min}}$ | bản đồ $4\times4$ | bản đồ $\in[0,1]$ | — (chuẩn hoá theo ảnh) |
| ③ | $\mathcal{M}_{mask}=\mathbb{I}(\mathcal{M}>0.1)$ | bản đồ $[0,1]$ | mặt nạ $\{0,1\}$ | — |
| ④ | $\text{counts}^j_i=\sum_{p\in\Omega_i}\mathcal{M}_{mask}(p)$ | mặt nạ + hình học | vector 5 số nguyên | **pixel → Gaussian** |
| ⑤ | $s_d=\lfloor\frac1K\sum_j\text{counts}\rfloor$ | ma trận $5\times3$ | vector 5 | góc nhìn (trung bình) |
| ⑥ | $E_{photo}=0.8\mathcal{L}_1+0.2(1-\text{SSIM})$ | 2 ảnh | **1 số** | toàn ảnh |
| ⑦ | $s_p=\mathcal{N}(\sum_j\text{counts}\cdot E_{photo})$ | ma trận $5\times3$ + 3 số | vector 5 $\in[0,1]$ | góc nhìn (tổng có trọng số) |

## Bài tập (Exercise)

**Bài tập 12.1.** Mục 7.2 định nghĩa Importance$_i=\lfloor\frac1V\sum_v\text{counts}^{(v)}_i\rfloor$ và giải thích phép chia $V$ là "bản chất của multi-view consistency". Dùng ví dụ $G_5$ ở bảng mục "Ví dụ (từ `3.md`)" ($\sum_v\text{counts}=5$, chỉ tệ ở 1 view, Importance$=1$): nếu không chia cho $V$ mà chỉ lấy $\max_v\text{counts}^{(v)}_i$, $G_5$ sẽ có điểm cao hay thấp? Giải thích vì sao cách tính $\max$ sẽ khiến FastGS-lite densify nhầm các Gaussian chỉ có artefact ở một góc nhìn (ví dụ phản chiếu hoặc che khuất tạm thời).

**Bài tập 12.2.** Trace code: đọc mục 7.3, công thức clone và split đều có điều kiện chung `Importance_i > 5`. Với cảnh đồ chơi (4 điểm SfM, 3 camera) ở bảng "extent và nhánh clone/split" của phần kiểm định số, Importance của cả 4 Gaussian đều nằm trong khoảng 974–1066, vượt xa ngưỡng 5. Giải thích tại sao ngưỡng 5 "không chặn ai" trong cảnh đồ chơi này (gợi ý: liên hệ tới việc $\alpha$ mô hình $=0.1$ khác rất xa $\alpha$ GT $=0.9$ trên toàn ảnh). Điều kiện Importance$>5$ sẽ phát huy tác dụng rõ nhất trong tình huống huấn luyện thật nào?

**Bài tập 12.3.** Tính tay $\delta\cdot\text{extent}$ và $0.1\cdot\text{extent}$ cho cảnh đồ chơi, biết $\text{extent}=1.690250$, $\delta=0.001$ (bảng mục 7.3). So sánh với $\max s_i\in\{0.8505,0.9434,1.1150,0.8888\}$ (bảng "Đầu vào của khối") để xác nhận vì sao cả 4 Gaussian đều rơi vào nhánh **split** chứ không phải **clone**. Sau đó giải thích bằng lời: nếu SfM khởi tạo với 4 điểm gần nhau hơn nhiều (ví dụ scale ban đầu $s_i\approx0.001$), nhánh nào sẽ được kích hoạt thay vào đó, và vì sao.

**Bài tập 12.4.** Mục 7.4 nêu ba điều "dễ hiểu sai" về prune có trọng số. Dùng bảng $w_i=(4.2621,\,1.8204,\,10^6,\,1.0000)$ (tương ứng $G_1,G_2,G_3,G_4$, cảnh đồ chơi $N=4$, $t>3000$): (a) tính $p_i=w_i/\sum w_i$ và xác nhận các giá trị $(4\times10^{-6},\,2\times10^{-6},\,0.999993,\,1\times10^{-6})$ đã nêu trong văn bản; (b) giải thích vì sao $\text{Pruning}_i\to1$ khiến $w_i$ "phân kỳ" và vì sao tác giả nói cơ chế multinomial "về thực chất là sắp xếp giảm dần theo Pruning rồi lấy phần đầu"; (c) với $\mathcal C=\{1,2,3,4\}$ và budget $=\lfloor0.5\times4\rfloor=2$, giải thích tại sao kết quả rút mẫu $\mathcal S=\{1,3\}$ (seed 0) không mâu thuẫn với việc $G_4$ có Pruning $=0$ (không bao giờ bị rút) mà vẫn nằm trong $\mathcal C$.

**Bài tập 12.5.** So sánh công thức lý thuyết ở mục 7.8, $N\leftarrow N+|\text{clone}|+|\text{split}|-|\text{xoá}|=4+0+4-2=6$, với kết quả thật theo đúng thứ tự thực thi của code (split trước, prune sau trên quần thể mới), cho ra $N=8-4=4$ (mục "Thứ tự thật trong `densify_and_prune_fastgs`"). Giải thích cơ chế gây ra sự khác biệt này: vì sao `padded_importance[:scores.shape[0]]` (`gaussian_model.py:473-476`) gán trọng số Pruning **cũ** cho các Gaussian **con** theo đúng chỉ số, và hệ quả là 4 Gaussian con đầu tiên ($G_1c_1,G_2c_1,G_3c_1,G_4c_1$) đều có trọng số dương trong khi 4 con còn lại có trọng số 0. Tại sao văn bản nói "trong huấn luyện thật tỉ lệ split nhỏ nên lệch ít"?

**Bài tập 12.6.** Mục 7.6 định nghĩa $\text{final\_prune}_i=[\alpha_i<0.1]\vee[\text{Pruning}_i>0.9]$, chỉ chạy trong $15000<t<30000$. Dùng bảng kiểm định số của mục này ($\alpha=0.1$ cho cả 4 Gaussian, so sánh chặt "$<0.1$" nên không ai bị xoá theo điều kiện thứ nhất; Pruning $=(0.7654,\,0.4507,\,1.0000,\,0.0000)$): xác nhận chỉ $G_3$ bị xoá. Giải thích "nghịch lý biểu kiến" — vì sao $G_3$ vừa là Gaussian có Importance cao nhất (1066, đáng nhân bản nhất ở mục 7.3) vừa là Gaussian duy nhất bị tỉa cuối. Đây có phải lỗi thiết kế không? Biện luận theo đúng lý lẽ "đặt cược giai đoạn đầu, thanh lý giai đoạn cuối" trong văn bản.

**Bài tập 12.7.** Bảng 7.7 (Đối chiếu tổng) so sánh 3DGS gốc và FastGS-lite trên ba trục: tín hiệu densify, tín hiệu prune, và tỉa cuối. Với cảnh đồ chơi cụ thể ở chương này, hãy chỉ ra: (a) một chỗ mà FastGS-lite ra quyết định **giống hệt** 3DGS gốc (gợi ý: nhánh split ở mục 7.3, vì Importance không chặn); (b) một chỗ mà FastGS-lite ra quyết định **khác hẳn** 3DGS gốc (gợi ý: mục 7.4, 3DGS gốc không xoá gì vì $\alpha=0.1\not<0.005$, còn FastGS-lite xoá 2 Gaussian). Từ hai quan sát này, giải thích luận điểm cuối chương: vì densify lặp lại 28–145 lần trong một lần huấn luyện, một khác biệt nhỏ ở mỗi lần gọi ($r_{\text{spawn}}$, $r_{\text{prune}}$) tích luỹ thành khác biệt luỹ thừa ở $N$ cuối cùng — minh hoạ bằng công thức $N_{\text{cuối}}\approx N_0[(1+r_{\text{spawn}})(1-r_{\text{prune}})]^{n}$ và bảng số ($N_0=4$, $n=28$ cho $N_{\text{cuối}}\approx47.6$; $n=145$ cho $N_{\text{cuối}}\approx1.49\times10^6$).

---

[← Chương 11](11-gradient-flow-backprop.md) | [Mục lục](00-muc-luc.md) | [Chương 13 →](13-tong-hop-chi-phi-fastgs.md)
