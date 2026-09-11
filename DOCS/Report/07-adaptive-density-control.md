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

**Vì sao phép AND là đóng góp chính:** 3DGS densify mọi Gaussian có gradient lớn. Gaussian ở vùng đã hội tụ (gradient còn dư nhưng render đã đúng) hay ở artefact chỉ thấy từ 1–2 góc đều có Importance thấp → không nhân bản. Vì densify lặp ~28–145 lần, tác động lên $N$ là luỹ thừa:

$$
N_{\text{cuối}}\approx N_0\bigl[(1+r_{\text{spawn}})(1-r_{\text{prune}})\bigr]^{n_{\text{densify}}}
$$

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
