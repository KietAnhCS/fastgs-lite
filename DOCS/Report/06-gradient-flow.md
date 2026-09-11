# Chương 6 — Gradient Flow

> Mũi tên xanh trong sơ đồ: từ Image ngược qua Rasterizer, Projection, về 3D Gaussians. Gồm ba việc: (1) backward qua blend, (2) backward qua projection về 59 tham số, (3) cập nhật tham số bằng Adam.
> **FastGS-lite thay đổi ở (1)** — thêm gradient trị tuyệt đối — **và ở (3)** — đòn bẩy 3: Adam thưa dần, lr SH riêng.
> Code: `backward.cu`, `scene/gaussian_model.py:190-209` (`optimizer_step`), `:167-181` (`training_setup`), `:494-495` (`add_densification_stats`).

## 6.1 — Backward qua blend

Fork này (kế thừa Taming-3DGS) duyệt **theo chiều forward**, mỗi warp một cửa sổ 32 Gaussian, dùng checkpoint `sampled_T`/`sampled_ar` để khôi phục trạng thái. Nên công thức viết bằng **hiệu hai tiền tố** thay vì tổng đuôi.

Đặt $C^{\le n}=\sum_{k\le n}c_k\alpha_kT_k$, $C^{\text{tot}}=C^{\le\text{end}}$:

$$
\boxed{\ \frac{\partial C_{\text{ch}}}{\partial\alpha_n}
=c_{n,\text{ch}}T_n-\frac{C^{\text{tot}}_{\text{ch}}-C^{\le n}_{\text{ch}}}{1-\alpha_n}-\frac{T_{\text{final}}\,C_{\text{bg},\text{ch}}}{1-\alpha_n}\ }
$$

Số hạng thứ ba hay bị quên: background cũng phụ thuộc $\alpha_n$ qua $T_{\text{final}}$.

Chuỗi còn lại:

$$
\frac{\partial\mathcal L}{\partial c_n}=\alpha_nT_n\frac{\partial\mathcal L}{\partial C},\qquad
\frac{\partial\mathcal L}{\partial\alpha_n}=G_n\frac{\partial\mathcal L}{\partial\alpha_n(x)},\qquad
\frac{\partial\mathcal L}{\partial G_n}=\alpha_n\frac{\partial\mathcal L}{\partial\alpha_n(x)}
$$

$$
\frac{\partial G}{\partial\Delta_u}=-G\,(A\Delta_u+B\Delta_v),\qquad
\frac{\partial G}{\partial\Delta_v}=-G\,(C\Delta_v+B\Delta_u)
$$

$$
\frac{\partial\mathcal L}{\partial\mu'_u}=\frac{\partial\mathcal L}{\partial G}\frac{\partial G}{\partial\Delta_u}\cdot\frac W2,\qquad
\frac{\partial\mathcal L}{\partial\mu'_v}=\frac{\partial\mathcal L}{\partial G}\frac{\partial G}{\partial\Delta_v}\cdot\frac H2
$$

$$
\frac{\partial\mathcal L}{\partial A}=-\tfrac12G\Delta_u^2\frac{\partial\mathcal L}{\partial G},\quad
\frac{\partial\mathcal L}{\partial B}=-\tfrac12G\Delta_u\Delta_v\frac{\partial\mathcal L}{\partial G},\quad
\frac{\partial\mathcal L}{\partial C}=-\tfrac12G\Delta_v^2\frac{\partial\mathcal L}{\partial G}
$$

### FastGS: gradient 4 cột (`backward.cu:589-597`)

Cộng dồn trong register qua toàn bộ pixel $x$ của tile, rồi `atomicAdd` một lần:

$$
g_n=\sum_x\frac{\partial\mathcal L}{\partial\mu'_n}\Big|_x\qquad(\text{cột 0–1, có dấu — như 3DGS})
$$

$$
\boxed{\ g^{\text{abs}}_n=\sum_x\Bigl|\frac{\partial\mathcal L}{\partial\mu'_n}\Big|_x\Bigr|\ }\qquad(\text{cột 2–3, trị tuyệt đối — mới})
$$

Vì sao cần: Gaussian phủ biên vật thể nhận gradient đẩy sang trái ở nửa trái footprint và sang phải ở nửa phải — **trong cùng một ảnh**. Tổng có dấu triệt tiêu về $\approx0$, 3DGS gốc không thấy nó cần split. Bất đẳng thức, đúng tầng pixel:

$$
\Bigl\lVert\sum_xg_x\Bigr\rVert\le\sum_x\lVert g_x\rVert
$$

Đây là lý do `screenspace_points` có 4 cột thay vì 3.

## 6.2 — Backward qua Projection về 59 tham số

Chuỗi $\partial\mathcal L/\partial M\to\partial\mathcal L/\partial\Sigma'\to\partial\mathcal L/\partial\Sigma\to(\partial\mathcal L/\partial q,\ \partial\mathcal L/\partial s)$; đồng thời $\partial\mathcal L/\partial\Sigma'\to\partial\mathcal L/\partial J\to\partial\mathcal L/\partial t$ (có cổng clamp: gradient bằng 0 nếu $t_x/t_z$ đã bị clip); và $\partial\mathcal L/\partial c\to\partial\mathcal L/\partial k_{lm}$ (chặn ở kênh bị `clamped`), $\to\partial\mathcal L/\partial\vec d\to\partial\mathcal L/\partial\mu$.

$\mu$ nhận gradient từ **ba** nguồn: qua $\mu'$ (chiếu tâm), qua $J$ (chiếu covariance), qua $\vec d$ (SH). Chuẩn hoá quaternion do autograd PyTorch lo.

## 6.3 — Tích luỹ thống kê cho Adaptive Density Control

`add_densification_stats`, chỉ với Gaussian nhìn thấy ($\text{radii}>0$):

$$
\text{accum}_i\mathrel{+}=\lVert g_i\rVert_2,\qquad
\text{accum}^{\text{abs}}_i\mathrel{+}=\lVert g^{\text{abs}}_i\rVert_2,\qquad
\text{denom}_i\mathrel{+}=1
$$

Lấy `norm` **trước** khi cộng qua iteration — sau đó mọi số đều không âm, không còn gì triệt tiêu. Chương 7 dùng $\bar g_i=\text{accum}_i/\text{denom}_i$.

## 6.4 — Cập nhật tham số: Adam — **FastGS thay nhịp gọi**

### Update rule (giữ nguyên)

$$
m\leftarrow\beta_1m+(1-\beta_1)g,\qquad v\leftarrow\beta_2v+(1-\beta_2)g^2
$$

$$
\theta\leftarrow\theta-\eta_\theta\frac{\hat m}{\sqrt{\hat v}+\epsilon},\qquad\epsilon=10^{-15}
$$

### Hai optimizer, sáu nhóm learning rate

| Nhóm | Optimizer | $\eta$ mặc định | Không gian |
|---|---|---|---|
| `xyz` | `optimizer` | $1.6\times10^{-4}\cdot\text{extent}$, decay xuống $1.6\times10^{-6}\cdot\text{extent}$ | world |
| `f_dc` | `optimizer` | `lowfeature_lr` $=0.0025$ | SH bậc 0 |
| `opacity` | `optimizer` | $0.025$ | logit |
| `scaling` | `optimizer` | $0.005$ | log |
| `rotation` | `optimizer` | $0.001$ | quaternion thô |
| `f_rest` | `shoptimizer` | $\boxed{\texttt{highfeature\_lr}/20}=0.005/20=0.00025$ | SH bậc 1–3 |

⚠️ Giá trị `--highfeature_lr` bị **chia 20** trước khi tới Adam. Ở mặc định, SH bậc cao học với lr nhỏ hơn 10× SH bậc thấp. Preset 0.02 cũng chỉ ra 0.001.

Decay của `xyz`:

$$
\eta_{xyz}(t)=\text{extent}\cdot\exp\Bigl((1-\tfrac tT)\ln\eta_{\text{init}}+\tfrac tT\ln\eta_{\text{final}}\Bigr),\qquad T=\texttt{position\_lr\_max\_steps}=30000
$$

### Lịch step thưa dần (`optimizer_step`)

$$
\mathbb 1_{\text{main}}(t)=\begin{cases}
1&t\le15000\\
[t\bmod32=0]&15000<t\le20000\\
[t\bmod64=0]&t>20000
\end{cases}
\qquad
\mathbb 1_{\text{SH}}(t)=\begin{cases}
[t\bmod16=0]&t\le15000\\
\mathbb 1_{\text{main}}(t)&t>15000
\end{cases}
$$

Đếm chính xác trên 30 000 vòng (vòng cuối không step vì `if iteration < opt.iterations`):

| | `optimizer` | `shoptimizer` | tổng |
|---|---|---|---|
| 3DGS gốc | 29 999 | 29 999 | 59 998 |
| FastGS-lite | 15 313 | 1 250 | 16 563 |
| tỉ số | 0.510 | 0.042 | **0.276** |

(Khoảng $(15000,20000]$ có **157** bội của 32 vì $20000=32\times625$ — đếm nhẩm dễ hụt đúng bước này.)

### Gradient tích luỹ giữa hai lần step

`zero_grad` chỉ gọi khi step, nên gradient đưa vào Adam là

$$
g^{(t)}_{\text{eff}}=\sum_{t'=t_{\text{prev}}+1}^{t}\nabla\mathcal L^{(t')}
$$

Nhưng Adam chuẩn hoá theo $\sqrt{\hat v}$: cộng 64 gradient rồi step một lần cho bước đi cỡ $\approx\eta$, **không** bằng 64 bước nhỏ. Đó là lý do giai đoạn 15k–30k gần như miễn phí về thời gian và cũng học được rất ít. Hạ `--iterations` xuống 15k tiết kiệm rất ít nhưng mất `final_prune_fastgs` (chương 7); sàn hợp lý là 20k, và phải đồng bộ `position_lr_max_steps`, `densify_until_iter` cùng hai ngưỡng cứng 15000/20000 trong `optimizer_step`.

### Tách lr SH ≠ đòn bẩy tốc độ

Lr là số nhân vô hướng, không đổi số phép tính. Thứ tăng tốc là nhịp $1/16$. Cộng cả hai, trong 0–15k `f_rest` nhận lr nhỏ hơn 10× và số bước ít hơn 16× so với `f_dc` — đó là lý do vệt specular hội tụ chậm.

## 6.5 — Đầu ra của khối

Mũi tên xanh kết thúc ở **3D Gaussians**: $\theta_i\leftarrow\theta_i-\Delta\theta_i$ khi $\mathbb 1(t)=1$, và ba mảng thống kê $(\text{accum},\text{accum}^{\text{abs}},\text{denom})$ sẵn sàng cho **Adaptive Density Control** (chương 7).
