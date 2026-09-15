[← Mục lục](00-muc-luc.md) · Chương 11/15

# Chương 11 — Gradient Flow & Backpropagation

> Nguồn: `DOCS/Report/06-gradient-flow.md`, `DOCS/Report/test/06-test.md`

## 11.1 Lý thuyết

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

![Gradient tại biên vật thể](../Report/assets/ch6_edge_gradient.png)

*Trục là toạ độ pixel u, v. Đường thẳng đứng đen là biên vật thể; mũi tên đỏ là $\partial\mathcal L/\partial\mu'$ tại các Gaussian hai bên biên — nửa trái đẩy trái, nửa phải đẩy phải. Cộng có dấu qua toàn ảnh gần như triệt tiêu; cộng trị tuyệt đối (độ dài mũi tên) thì không, nên $g^{abs}$ giữ được tín hiệu "cần split" ở biên.*

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

![Quỹ đạo Adam trên mặt loss đồ chơi](../Report/assets/ch6_adam_trajectory.png)

*Trục x, y là hai chiều của tham số đồ chơi $\theta_1,\theta_2$. Đường viền xám là các mức đường đồng mức của hàm loss; đường đỏ là quỹ đạo $\theta$ qua từng bước Adam, từ chấm xanh $\theta_0$ hội tụ về sao đen — quỹ đạo uốn cong theo trục dốc hơn, đặc trưng của chuẩn hoá theo $\sqrt{\hat v}$.*

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

![Suy giảm learning rate vị trí](../Report/assets/ch6_lr_decay.png)

*Trục x là vòng lặp $t$ (0→30000), trục y là $\eta_{xyz}(t)$ vẽ theo thang log. Đường cong gần như tuyến tính trên thang log — đúng bản chất nội suy log-linear — giảm khoảng 100× từ đầu tới cuối.*

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

![Lịch step thưa dần: main vs SH](../Report/assets/ch6_step_schedule.png)

*Trên: trục x là $t$, trục y là số bước step luỹ kế — đường xanh (main) gãy khúc thoải dần tại $t=15000,20000$; đường đỏ (SH) tăng chậm hơn rồi nhập cùng lịch main sau 15000. Dưới: mỗi vạch là một thời điểm step thực sự trong 0–20000 — hàng SH thưa hơn hẳn hàng main, đúng tỉ lệ 1/16 so với 1/1.*

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

## 11.2 Kiểm định số — Chương 6, Gradient

# Test số chương 6 — Gradient Flow

> Tính trên cảnh đồ chơi ở `00-scene.md` (4 điểm SfM, 3 camera, ảnh $48\times32$, $f_x=f_y=40$).
> GT $=$ render với $\alpha=0.9$; mô hình đang học có $\alpha=0.1$. Nền trắng.
> Script sinh số: `scripts/ch06_test.py` (chỉ `numpy`, không torch). Mọi số dưới đây là output thật của script,
> làm tròn 4 chữ số có nghĩa; script giữ full precision.
> **Mọi đạo hàm giải tích đều được kiểm chứng bằng sai phân hữu hạn trung tâm** (FD, $h=10^{-6}$),
> in cả hai cột và sai số tương đối $\lvert a-b\rvert/\max(\lvert a\rvert,\lvert b\rvert)$.
>
> Công thức chép từ code: `backward.cu:549-597` (`renderCUDA` backward: 3 số hạng `dL_dalpha`, `dG_ddel`, 4 cột `dL_dmean2D`),
> `backward.cu:212-214` (hệ số 2 cho `dL_dconic.y`), `forward.cu:429-430` (`pixel_colors` **không** chứa nền),
> `scene/gaussian_model.py:167-181` (`training_setup`), `:190-209` (`optimizer_step`), `:494-497` (`add_densification_stats`),
> `train.py:161-162` (`if iteration < opt.iterations`), `utils/general_utils.py:29-50` (`get_expon_lr_func`), `arguments/__init__.py:76-93` (lr mặc định).
>
> Renderer numpy tối giản trong script **lấy từ chương 3–4**: $\mu=p$, $s=\sqrt{\text{mean } d^2 \text{ tới 3 láng giềng}}$, $R=I$, $\Sigma=s^2I$, màu $=c_k$;
> $t=\mu-c_v$, $\mu'$ qua `ndc2Pix`, $J$ với clamp $1.3\tan$, $\Sigma'=J\Sigma J^\top+0.3I$, conic $=\Sigma'^{-1}$;
> render mọi Gaussian theo depth với 3 cửa loại ($\text{power}>0$, $\alpha<1/255$, $T<10^{-4}$). Không cull tile (4 Gaussian phủ cả ảnh, `radius` $=26..29$ px).

## Sơ đồ khối với số của cảnh đồ chơi

```mermaid
flowchart TD
  L["`**∂𝓛/∂C(x)**
  từ L1 (chương 5)`"]

  A["`**Alpha blending backward**
  ∂C/∂αₙ = cₙTₙ − (C_tot − C_≤ₙ)/(1−αₙ) − T_final·C_bg/(1−αₙ)
  *pixel (23,15), 4 Gaussian*
  khớp sai phân hữu hạn ≤ **1.8e−9**`"]

  G["`**Gaussian 2D backward**
  ∂G/∂Δ = −G·(AΔu + BΔv, CΔv + BΔu)
  ∂𝓛/∂A, ∂𝓛/∂C: khớp FD ≤ **3e−7**
  ∂𝓛/∂B: code lưu ½ rồi ×2 ở computeCov2D backward`"]

  MU["`**Screen-space gradient**
  ∂𝓛/∂μ' × (W/2, H/2) → 4 cột
  cộng dồn **1536 pixel** của camera 1`"]

  S["`**Cột 0–1 (có dấu)**
  G1: g = (4.37e−4, −2.06e−4)
  ‖g‖ = **4.83e−4**`"]

  AB["`**Cột 2–3 (trị tuyệt đối)**
  G1: g_abs = (3.17e−2, 1.92e−2)
  ‖g_abs‖ = **3.71e−2**
  gấp **77×** — nửa trái/phải footprint triệt tiêu nhau`"]

  ACC["`**Gộp 3 camera = 3 iteration**
  accum += ‖g‖, denom += 1
  ḡ = 6.43e−3 ≥ τ_grad = 2e−4 ✓
  ḡ_abs = 3.04e−2 ≥ τ_abs = 1.2e−3 ✓ (cả 4 Gaussian)`"]

  ADC["`→ **ADC** (chương 7)`"]

  ADAM["`**Adam update**
  m̂ / (√v̂ + ε) = 1.000 với g lặp lại
  ⇒ Δθ = η bất kể độ lớn g
  xyz 2.7e−4 · f_dc 2.5e−3 · opacity 2.5e−2
  scaling 5e−3 · rotation 1e−3 · f_rest 0.005/20 = 2.5e−4`"]

  SCH["`**Lịch step** (t = 1 … 29999)
  optimizer 15000+157+156 = 15313
  shoptimizer 937+157+156 = 1250
  tổng 16563 / 59998 = **0.276**`"]

  GE["`**g_eff**
  cộng 64 gradient rồi 1 step ⇒ Δθ = 1.0·η
  64 step nhỏ ⇒ 63.6·η (khác 63.6×)`"]

  L --> A --> G --> MU
  MU --> S --> ACC
  MU --> AB --> ACC
  ACC --> ADC
  MU --> ADAM --> SCH --> GE
```

## 6.0 — Đầu vào (từ chương 1–4)

| $i$ | $\mu_i$ | $s_i$ | $c_i$ | camera 1: $t_i$ | $\mu'_i$ (px) | $\Sigma'_i$ $(a,b,c)$ | conic $(A,B,C)$ |
|---|---|---|---|---|---|---|---|
| 1 | $(0,0,0)$ | 0.8505 | (0.8,0.2,0.2) | $(0,0,4)$ | (23.50, 15.50) | (72.63, 0, 72.63) | (0.01377, 0, 0.01377) |
| 2 | $(0.5,0.3,0.5)$ | 0.9434 | (0.2,0.7,0.3) | $(0.5,0.3,4.5)$ | (27.94, 18.17) | (71.49, 0.5209, 70.93) | (0.01399, $-1.027\times10^{-4}$, 0.01410) |
| 3 | $(-0.4,-0.2,1)$ | 1.115 | (0.1,0.3,0.9) | $(-0.4,-0.2,5)$ | (20.30, 13.90) | (80.38, 0.2546, 80.00) | (0.01244, $-3.960\times10^{-5}$, 0.01250) |
| 4 | $(0.3,-0.5,0.2)$ | 0.8888 | (0.5,0.5,0.5) | $(0.3,-0.5,4.2)$ | (26.36, 10.74) | (72.32, $-0.6093$, 72.97) | (0.01383, $1.155\times10^{-4}$, 0.01371) |

$\text{extent}=1.1\cdot\max_v\lVert c_v-\bar c\rVert=1.1\times1.537=1.690$ (khớp chương 1).
$\mathcal L=\text{L1}$ trung bình: $\mathcal L^{(1)}=0.2593$ (camera 1); $\partial\mathcal L/\partial C_{\text{ch}}(x)=\operatorname{sign}(I_{\text{rend}}-I_{\text{gt}})/(3HW)$, $3HW=4608$ → $\pm2.170\times10^{-4}$.

## 6.1 — Backward qua blend

### (a) Tại một pixel có 4 Gaussian đóng góp

![Ba số hạng của ∂C/∂α_n](../Report/test/figures/ch06_dC_dalpha.png)

*Hình: ba số hạng của công thức hiệu hai tiền tố tại pixel (23,15), kênh R, cho 4 Gaussian — giá trị giải tích trùng khít sai phân hữu hạn.*


Pixel chọn tự động: nhiều contributor nhất và gần tâm Gaussian 1 → $x=(23,15)$, **4 contributor** (thứ tự depth: G1, G4, G2, G3).

$I_{\text{rend}}(x)=(0.8262,0.8134,0.8307)$, $I_{\text{gt}}(x)=(0.7612,0.2340,0.2304)$ → $\partial\mathcal L/\partial C=(+,+,+)\cdot2.170\times10^{-4}$ (render sáng hơn GT ở cả 3 kênh — mô hình quá mờ nên lộ nền trắng).

| $n$ | Gaussian | $d=\mu'-x$ | $G_n$ | $\alpha_n(x)$ | $T_n$ | $C^{\le n}$ |
|---|---|---|---|---|---|---|
| 1 | G1 | (0.5, 0.5) | 0.9966 | 0.09966 | 1.0000 | (0.07973, 0.01993, 0.01993) |
| 2 | G4 | (3.357, $-4.262$) | 0.8181 | 0.08181 | 0.9003 | (0.1166, 0.05676, 0.05676) |
| 3 | G2 | (4.944, 3.167) | 0.7866 | 0.07866 | 0.8267 | (0.1296, 0.1023, 0.07627) |
| 4 | G3 | ($-2.7$, $-1.1$) | 0.9486 | 0.09486 | 0.7617 | (0.1368, 0.1240, 0.1413) |

$C^{\text{tot}}=(0.1368,0.1240,0.1413)$ (không nền), $T_{\text{final}}=0.6894$, $C(x)=C^{\text{tot}}+T_{\text{final}}\cdot(1,1,1)=(0.8262,0.8134,0.8307)$ ✓ khớp $I_{\text{rend}}(x)$.

**Công thức** (hiệu hai tiền tố, `backward.cu:559-568`; `ar` khởi tạo $=$ `sampled_ar` $-$ `pixel_colors`, mà `pixel_colors` không chứa nền — `forward.cu:429`):

$$\frac{\partial C_{\text{ch}}}{\partial\alpha_n}=\underbrace{c_{n,\text{ch}}T_n}_{\text{(1)}}\ \underbrace{-\frac{C^{\text{tot}}_{\text{ch}}-C^{\le n}_{\text{ch}}}{1-\alpha_n}}_{\text{(2)}}\ \underbrace{-\frac{T_{\text{final}}C_{\text{bg,ch}}}{1-\alpha_n}}_{\text{(3)}}$$

**Thay số** cho $n=1$ (G1): (1) $=(0.8,0.2,0.2)\cdot1$; (2) $=-(0.1368-0.07973,\ldots)/(1-0.09966)=(-0.06338,-0.1155,-0.1348)$; (3) $=-0.6894\cdot1/0.9003=(-0.7657,-0.7657,-0.7657)$.

**Kết quả** — analytic vs FD (nhiễu $\alpha_n(x)\pm h$ rồi blend lại pixel):

| $n$ | (1) $c_nT_n$ | (2) | (3) nền | analytic $\partial C/\partial\alpha_n$ | FD | sai số tđ |
|---|---|---|---|---|---|---|
| 1 (G1) | (0.8, 0.2, 0.2) | ($-0.06338$, $-0.1155$, $-0.1348$) | $-0.7657$ | ($-0.02909$, $-0.6813$, $-0.7005$) | ($-0.02909$, $-0.6813$, $-0.7005$) | $1.8\times10^{-9}$ |
| 2 (G4) | 0.4502 (×3) | ($-0.02203$, $-0.07318$, $-0.09206$) | $-0.7508$ | ($-0.3227$, $-0.3738$, $-0.3927$) | ($-0.3227$, $-0.3738$, $-0.3927$) | $1.5\times10^{-10}$ |
| 3 (G2) | (0.1653, 0.5787, 0.2480) | ($-0.007842$, $-0.02353$, $-0.07058$) | $-0.7483$ | ($-0.5908$, $-0.1931$, $-0.5708$) | ($-0.5908$, $-0.1931$, $-0.5708$) | $8.2\times10^{-11}$ |
| 4 (G3) | (0.07617, 0.2285, 0.6855) | (0, 0, 0) — cuối cùng | $-0.7617$ | ($-0.6855$, $-0.5332$, $-0.07617$) | ($-0.6855$, $-0.5332$, $-0.07617$) | $9.1\times10^{-10}$ |

Nhận xét: số hạng nền (3) **lớn nhất** ở mọi $n$ (vì $T_{\text{final}}=0.69$, cảnh còn rất trong suốt); bỏ nó thì $\partial C_R/\partial\alpha_1$ đổi dấu ($+0.737$ thay vì $-0.029$). Với G3 (contributor cuối) số hạng (2) $=0$ đúng như kỳ vọng.

$\partial\mathcal L/\partial\alpha_n(x)=\sum_{\text{ch}}\partial\mathcal L/\partial C_{\text{ch}}\cdot\partial C_{\text{ch}}/\partial\alpha_n$: $-3.062\times10^{-4}$ (G1), $-2.364\times10^{-4}$ (G4), $-2.940\times10^{-4}$ (G2), $-2.810\times10^{-4}$ (G3) — đều âm: tăng $\alpha$ giảm loss (mô hình mờ hơn GT).

### (b) Chuỗi $G\to\Delta$, conic tại pixel đó

Code dùng $d=\mu'-x=-\Delta$ (`backward.cu:552`), $\text{power}=-\tfrac12(Ad_u^2+Cd_v^2)-Bd_ud_v$:

$$\frac{\partial G}{\partial d_u}=-G(Ad_u+Bd_v),\quad\frac{\partial G}{\partial d_v}=-G(Cd_v+Bd_u),\quad\frac{\partial\mathcal L}{\partial G}=\alpha_n\frac{\partial\mathcal L}{\partial\alpha_n(x)}$$

$$\frac{\partial\mathcal L}{\partial A}=-\tfrac12Gd_u^2\frac{\partial\mathcal L}{\partial G},\quad\frac{\partial\mathcal L}{\partial B}\Big|_{\text{code}}=-\tfrac12Gd_ud_v\frac{\partial\mathcal L}{\partial G},\quad\frac{\partial\mathcal L}{\partial C}=-\tfrac12Gd_v^2\frac{\partial\mathcal L}{\partial G}$$

**Thay số** $n=1$ (G1): $\partial\mathcal L/\partial G=0.1\times(-3.062\times10^{-4})=-3.062\times10^{-5}$; $\partial G/\partial d_u=-0.9966(0.01377\cdot0.5+0)=-6.860\times10^{-3}$.

**Kết quả** (FD trên $G$ cho $\partial G/\partial d$; FD trên loss-pixel $\mathcal L_x=\sum_{\text{ch}}\lvert C_{\text{ch}}-I_{\text{gt,ch}}\rvert/(3HW)$ cho các đại lượng còn lại):

| $n$ | đại lượng | analytic | FD | sai số tđ |
|---|---|---|---|---|
| 1 (G1) | $\partial G/\partial d_u$ | $-6.860\times10^{-3}$ | $-6.860\times10^{-3}$ | $6.9\times10^{-9}$ |
| | $\partial G/\partial d_v$ | $-6.860\times10^{-3}$ | $-6.860\times10^{-3}$ | $6.9\times10^{-9}$ |
| | $\partial\mathcal L_x/\partial d_u$ | $2.100\times10^{-7}$ | $2.100\times10^{-7}$ | $2.6\times10^{-7}$ |
| | $\partial\mathcal L_x/\partial A$ | $3.814\times10^{-6}$ | $3.814\times10^{-6}$ | $2.0\times10^{-8}$ |
| | $\partial\mathcal L_x/\partial B$ (code, $\tfrac12$) | $3.814\times10^{-6}$ | $7.628\times10^{-6}$ | **0.50** |
| | $\partial\mathcal L_x/\partial B$ thực $=2\times$code | $7.628\times10^{-6}$ | $7.628\times10^{-6}$ | $1.2\times10^{-9}$ |
| | $\partial\mathcal L_x/\partial C$ | $3.814\times10^{-6}$ | $3.814\times10^{-6}$ | $2.0\times10^{-8}$ |
| 2 (G4) | $\partial G/\partial d_u$ | $-3.758\times10^{-2}$ | $-3.758\times10^{-2}$ | $2.0\times10^{-10}$ |
| | $\partial G/\partial d_v$ | $4.747\times10^{-2}$ | $4.747\times10^{-2}$ | $9.3\times10^{-11}$ |
| | $\partial\mathcal L_x/\partial A$ | $1.090\times10^{-4}$ | $1.090\times10^{-4}$ | $1.8\times10^{-10}$ |
| | $\partial\mathcal L_x/\partial B$ (code) / thực | $-1.384\times10^{-4}$ / $-2.767\times10^{-4}$ | $-2.767\times10^{-4}$ | 0.50 / $5.7\times10^{-11}$ |
| | $\partial\mathcal L_x/\partial C$ | $1.756\times10^{-4}$ | $1.756\times10^{-4}$ | $9.1\times10^{-11}$ |
| 3 (G2) | $\partial G/\partial d_u$, $\partial G/\partial d_v$ | $-5.415\times10^{-2}$, $-3.472\times10^{-2}$ | như analytic | $\le2\times10^{-10}$ |
| | $\partial\mathcal L_x/\partial A,\ B_{\text{thực}},\ C$ | $2.827\times10^{-4}$, $3.621\times10^{-4}$, $1.159\times10^{-4}$ | như analytic | $\le2.3\times10^{-11}$ |
| 4 (G3) | $\partial G/\partial d_u$, $\partial G/\partial d_v$ | $3.182\times10^{-2}$, $1.294\times10^{-2}$ | như analytic | $\le2.7\times10^{-9}$ |
| | $\partial\mathcal L_x/\partial A,\ B_{\text{thực}},\ C$ | $9.716\times10^{-5}$, $7.916\times10^{-5}$, $1.613\times10^{-5}$ | như analytic | $\le3.9\times10^{-9}$ |

**Phát hiện khi đối chiếu code:** FD cho $\partial\mathcal L/\partial B$ đúng **gấp 2** giá trị `-0.5f * gdx * d.y * dL_dG` của `backward.cu:594`. Không phải lỗi: vì $\text{power}$ chứa $2B\Delta_u\Delta_v$ nên đạo hàm thực là $-G\Delta_u\Delta_v\,\partial\mathcal L/\partial G$; code lưu **một nửa** vào `dL_dconic2D.y` rồi `computeCov2DCUDA` backward nhân lại $2\cdot$`dL_dconic.y` ở `backward.cu:212-214`. Công thức $-\tfrac12G\Delta_u\Delta_v$ trong `06-gradient-flow.md` là **quy ước của code**, không phải đạo hàm toán học của $B$.

Gradient tâm tại pixel (nhân $W/2=24$, $H/2=16$ như `ddelx_dx`): G1: $(5.041\times10^{-6},\ 3.361\times10^{-6})$; G4: $(2.132\times10^{-5},\ -1.795\times10^{-5})$; G2: $(3.821\times10^{-5},\ 1.633\times10^{-5})$; G3: $(-2.146\times10^{-5},\ -5.818\times10^{-6})$.

### (c) FastGS: gradient 4 cột (`backward.cu:583-597`), cộng dồn qua **toàn bộ** 1536 pixel, camera 1

![Trường gradient per-pixel của μ'_1 (G1)](../Report/test/figures/ch06_grad_field.png)

*Hình: quiver cho thấy nửa trái footprint của G1 kéo gradient sang trái, nửa phải kéo sang phải — tổng có dấu gần triệt tiêu (‖g‖=4.83e−4) trong khi tổng trị tuyệt đối vẫn lớn (‖g_abs‖=3.71e−2, gấp 77 lần); đây là lý do FastGS cần cột "abs".*

![Gradient 4 cột: có dấu vs trị tuyệt đối](../Report/test/figures/ch06_4cols.png)

*Hình: so sánh ‖g‖ và ‖g_abs‖ cho cả 4 Gaussian (trục log) với hai ngưỡng τ_grad và τ_abs; cả 4 Gaussian đều vượt cả hai ngưỡng sau khi tích luỹ qua 3 camera.*


$$g_n=\sum_x\frac{\partial\mathcal L}{\partial\mu'_n}\Big|_x\ (\text{cột 0–1}),\qquad g^{\text{abs}}_n=\sum_x\Bigl|\frac{\partial\mathcal L}{\partial\mu'_n}\Big|_x\Bigr|\ (\text{cột 2–3})$$

| $i$ | $g_u$ | $g_v$ | $g^{\text{abs}}_u$ | $g^{\text{abs}}_v$ | $\lVert g\rVert$ | $\lVert g^{\text{abs}}\rVert$ | $\lVert g^{\text{abs}}\rVert/\lVert g\rVert$ |
|---|---|---|---|---|---|---|---|
| 1 | $4.367\times10^{-4}$ | $-2.064\times10^{-4}$ | $3.168\times10^{-2}$ | $1.921\times10^{-2}$ | $4.830\times10^{-4}$ | $3.705\times10^{-2}$ | 76.7 |
| 2 | $-1.122\times10^{-3}$ | $1.357\times10^{-3}$ | $3.057\times10^{-2}$ | $1.854\times10^{-2}$ | $1.760\times10^{-3}$ | $3.575\times10^{-2}$ | 20.3 |
| 3 | $1.466\times10^{-3}$ | $-1.370\times10^{-3}$ | $3.065\times10^{-2}$ | $1.832\times10^{-2}$ | $2.007\times10^{-3}$ | $3.570\times10^{-2}$ | 17.8 |
| 4 | $-7.168\times10^{-4}$ | $-2.570\times10^{-3}$ | $2.474\times10^{-2}$ | $1.411\times10^{-2}$ | $2.669\times10^{-3}$ | $2.848\times10^{-2}$ | 10.7 |

$\bigl\lVert\sum_xg_x\bigr\rVert\le\sum_x\lVert g_x\rVert$ đúng bằng số ở cả 4 hàng; ở Gaussian 1 (tâm gần giữa ảnh, footprint đối xứng) tổng có dấu **triệt tiêu 77×** so với tổng trị tuyệt đối — đúng hiện tượng "kéo trái ở nửa trái, kéo phải ở nửa phải" mà chương 6 mô tả.

**Kiểm chứng $g$ có dấu bằng FD toàn ảnh**: dịch toạ độ NDC của tâm Gaussian $i$ đi $\pm h$ (vì `dL_dmean2D` là gradient theo NDC, `ddelx_dx = 0.5*W`), render lại 1536 pixel, tính L1:

| $i$ | trục | analytic $g$ | FD $(\mathcal L^+-\mathcal L^-)/2h$ | sai số tđ |
|---|---|---|---|---|
| 1 | $u$ | $4.3672\times10^{-4}$ | $4.3672\times10^{-4}$ | $6.8\times10^{-9}$ |
| 1 | $v$ | $-2.0642\times10^{-4}$ | $-2.0642\times10^{-4}$ | $9.2\times10^{-8}$ |
| 2 | $u$ | $-1.1218\times10^{-3}$ | $-1.1218\times10^{-3}$ | $1.4\times10^{-9}$ |
| 2 | $v$ | $1.3566\times10^{-3}$ | $1.3566\times10^{-3}$ | $8.4\times10^{-9}$ |
| 3 | $u$ | $1.4662\times10^{-3}$ | $1.4662\times10^{-3}$ | $2.8\times10^{-9}$ |
| 3 | $v$ | $-1.3705\times10^{-3}$ | $-1.3705\times10^{-3}$ | $2.0\times10^{-8}$ |
| 4 | $u$ | $-7.1682\times10^{-4}$ | $-7.1682\times10^{-4}$ | $4.0\times10^{-8}$ |
| 4 | $v$ | $-2.5704\times10^{-3}$ | $-2.5704\times10^{-3}$ | $4.6\times10^{-9}$ |

Toàn bộ chuỗi backward (3 số hạng $\alpha$ → $G$ → $d$ → $\mu'_{\text{ndc}}$, cộng dồn qua pixel với 3 cửa loại) khớp FD tới $\sim10^{-8}$.

## 6.2 — Backward qua projection

![Kiểm chứng giải tích vs sai phân hữu hạn](../Report/test/figures/ch06_fd_check.png)

*Hình: mọi đại lượng đã kiểm chứng (∂C/∂α, ∂G/∂Δ, ∂L/∂A,B,C, 8 thành phần gradient) nằm khít trên đường y=x — sai số tương đối tối đa dưới 1e−7.*


Không tính ở đây (chuỗi $\Sigma'\to\Sigma\to(q,s)$, $J\to t$, SH$\to\mu$ do autograd/`computeCov2DCUDA` lo). Ghi nhận duy nhất từ 6.1(b): hệ số $\tfrac12$ của $B$ được bù bằng $2\cdot$`dL_dconic.y` ở `backward.cu:212-214`.

## 6.3 — Tích luỹ thống kê cho ADC (`add_densification_stats`, `gaussian_model.py:494-497`)

Giả sử 3 iteration liên tiếp dùng camera 1, 2, 3 (mỗi iteration render + backward toàn ảnh, GT riêng của camera đó). `radii` $>0$ cho cả 4 Gaussian ở cả 3 camera → `update_filter` toàn True.

| cam | $\mathcal L^{(v)}$ | $i$ | $g_u$ | $g_v$ | $g^{\text{abs}}_u$ | $g^{\text{abs}}_v$ | $\lVert g\rVert_2$ | $\lVert g^{\text{abs}}\rVert_2$ |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2593 | 1 | $4.367\times10^{-4}$ | $-2.064\times10^{-4}$ | $3.168\times10^{-2}$ | $1.921\times10^{-2}$ | $4.830\times10^{-4}$ | $3.705\times10^{-2}$ |
| | | 2 | $-1.122\times10^{-3}$ | $1.357\times10^{-3}$ | $3.057\times10^{-2}$ | $1.854\times10^{-2}$ | $1.760\times10^{-3}$ | $3.575\times10^{-2}$ |
| | | 3 | $1.466\times10^{-3}$ | $-1.370\times10^{-3}$ | $3.065\times10^{-2}$ | $1.832\times10^{-2}$ | $2.007\times10^{-3}$ | $3.570\times10^{-2}$ |
| | | 4 | $-7.168\times10^{-4}$ | $-2.570\times10^{-3}$ | $2.474\times10^{-2}$ | $1.411\times10^{-2}$ | $2.669\times10^{-3}$ | $2.848\times10^{-2}$ |
| 2 | 0.2237 | 1 | $-8.933\times10^{-3}$ | $-2.093\times10^{-4}$ | $2.183\times10^{-2}$ | $1.721\times10^{-2}$ | $8.935\times10^{-3}$ | $2.780\times10^{-2}$ |
| | | 2 | $-4.109\times10^{-3}$ | $1.364\times10^{-3}$ | $2.823\times10^{-2}$ | $1.833\times10^{-2}$ | $4.330\times10^{-3}$ | $3.366\times10^{-2}$ |
| | | 3 | $-9.362\times10^{-3}$ | $-1.183\times10^{-3}$ | $2.050\times10^{-2}$ | $1.588\times10^{-2}$ | $9.437\times10^{-3}$ | $2.593\times10^{-2}$ |
| | | 4 | $-4.630\times10^{-3}$ | $-2.489\times10^{-3}$ | $2.059\times10^{-2}$ | $1.360\times10^{-2}$ | $5.257\times10^{-3}$ | $2.468\times10^{-2}$ |
| 3 | 0.2055 | 1 | $8.957\times10^{-3}$ | $-4.146\times10^{-3}$ | $2.106\times10^{-2}$ | $1.568\times10^{-2}$ | $9.870\times10^{-3}$ | $2.626\times10^{-2}$ |
| | | 2 | $1.168\times10^{-2}$ | $-2.144\times10^{-3}$ | $1.836\times10^{-2}$ | $1.541\times10^{-2}$ | $1.187\times10^{-2}$ | $2.397\times10^{-2}$ |
| | | 3 | $5.022\times10^{-3}$ | $-5.236\times10^{-3}$ | $2.616\times10^{-2}$ | $1.624\times10^{-2}$ | $7.255\times10^{-3}$ | $3.079\times10^{-2}$ |
| | | 4 | $7.400\times10^{-3}$ | $-4.927\times10^{-3}$ | $1.311\times10^{-2}$ | $9.124\times10^{-3}$ | $8.890\times10^{-3}$ | $1.598\times10^{-2}$ |

Nhận xét: camera 2 (lệch phải) đẩy mọi tâm sang trái ($g_u<0$), camera 3 (lệch trái) đẩy sang phải ($g_u>0$) — nếu cộng **vector** qua iteration thì $g_u$ của G1 triệt tiêu ($-8.93+8.96\approx0.02\times10^{-3}$); code lấy `norm` **trước** khi cộng nên không triệt tiêu.

**Công thức**: $\text{accum}_i\mathrel{+}=\lVert g_i\rVert_2$, $\text{accum}^{\text{abs}}_i\mathrel{+}=\lVert g^{\text{abs}}_i\rVert_2$, $\text{denom}_i\mathrel{+}=1$; $\bar g_i=\text{accum}_i/\text{denom}_i$.

**Thay số** G1: $\text{accum}_1=4.830\times10^{-4}+8.935\times10^{-3}+9.870\times10^{-3}=1.929\times10^{-2}$; $\bar g_1=1.929\times10^{-2}/3=6.430\times10^{-3}$.

| $i$ | accum | accum$^{\text{abs}}$ | denom | $\bar g_i$ | $\bar g^{\text{abs}}_i$ | $\bar g\ge\tau_{\text{grad}}=2\times10^{-4}$ | $\bar g^{\text{abs}}\ge\tau^{\text{abs}}=1.2\times10^{-3}$ |
|---|---|---|---|---|---|---|---|
| 1 | $1.929\times10^{-2}$ | $9.111\times10^{-2}$ | 3 | $6.430\times10^{-3}$ | $3.037\times10^{-2}$ | ✓ (32×) | ✓ (25×) |
| 2 | $1.796\times10^{-2}$ | $9.338\times10^{-2}$ | 3 | $5.988\times10^{-3}$ | $3.113\times10^{-2}$ | ✓ | ✓ |
| 3 | $1.870\times10^{-2}$ | $9.242\times10^{-2}$ | 3 | $6.233\times10^{-3}$ | $3.081\times10^{-2}$ | ✓ | ✓ |
| 4 | $1.682\times10^{-2}$ | $6.913\times10^{-2}$ | 3 | $5.605\times10^{-3}$ | $2.304\times10^{-2}$ | ✓ | ✓ |

**Cả 4 Gaussian vượt cả hai ngưỡng** (cảnh đồ chơi: 4 Gaussian to phủ cả ảnh 48×32 và sai lệch opacity 0.1 vs 0.9 rất lớn, nên gradient lớn hơn cảnh thật vài bậc). Đây là đầu vào chương 7.

## 6.4 — Adam và lịch step

### (a) Adam bằng số cho 1 tham số

Lấy $g=$ trung bình $g_u$ của G1 qua 3 camera $=(4.367\times10^{-4}-8.933\times10^{-3}+8.957\times10^{-3})/3=1.536\times10^{-4}$, lặp 3 bước, $\beta=(0.9,0.999)$, $\epsilon=10^{-15}$ (`gaussian_model.py:177-178`).

| $k$ | $m$ | $v$ | $\hat m$ | $\hat v$ | $\hat m/(\sqrt{\hat v}+\epsilon)$ |
|---|---|---|---|---|---|
| 1 | $1.536\times10^{-5}$ | $2.360\times10^{-11}$ | $1.536\times10^{-4}$ | $2.360\times10^{-8}$ | 1.0000000000 |
| 2 | $2.919\times10^{-5}$ | $4.718\times10^{-11}$ | $1.536\times10^{-4}$ | $2.360\times10^{-8}$ | 1.0000000000 |
| 3 | $4.163\times10^{-5}$ | $7.074\times10^{-11}$ | $1.536\times10^{-4}$ | $2.360\times10^{-8}$ | 1.0000000000 |

Với gradient lặp lại, $\hat m=g$, $\hat v=g^2$ nên tỉ số $=g/(\lvert g\rvert+\epsilon)=\operatorname{sign}(g)$. $\Delta\theta=\eta\cdot\hat m/(\sqrt{\hat v}+\epsilon)$ cho 6 nhóm (`training_setup`, `arguments/__init__.py`):

| nhóm | $\eta$ | $\Delta\theta$ ($k=1,2,3$ giống nhau) | $\lvert\Delta\theta\rvert/\eta$ |
|---|---|---|---|
| `xyz` ($t=0$): $1.6\times10^{-4}\cdot1.690$ | $2.704\times10^{-4}$ | $2.704\times10^{-4}$ | 1.000000 |
| `f_dc` | $2.5\times10^{-3}$ | $2.5\times10^{-3}$ | 1.000000 |
| `opacity` | $2.5\times10^{-2}$ | $2.5\times10^{-2}$ | 1.000000 |
| `scaling` | $5\times10^{-3}$ | $5\times10^{-3}$ | 1.000000 |
| `rotation` | $1\times10^{-3}$ | $1\times10^{-3}$ | 1.000000 |
| `f_rest` $=0.005/20$ | $2.5\times10^{-4}$ | $2.5\times10^{-4}$ | 1.000000 |

Thử $g'=g/1000=1.536\times10^{-7}$: tỉ số $=0.9999999935$ → **bước đi $\approx\eta$ bất kể độ lớn $g$** (chỉ khi $\lvert g\rvert\sim\epsilon=10^{-15}$ mới nhỏ đi).

### (b) Decay $\eta_{xyz}(t)$ (`get_expon_lr_func`, `utils/general_utils.py:29-50`)

$$\eta_{xyz}(t)=\exp\bigl((1-\tfrac tT)\ln\eta_{\text{init}}+\tfrac tT\ln\eta_{\text{final}}\bigr),\quad\eta_{\text{init}}=1.6\times10^{-4}\cdot1.690=2.704\times10^{-4},\ \eta_{\text{final}}=2.704\times10^{-6},\ T=30000$$

| $t$ | $t/T$ | $\eta_{xyz}(t)$ | $\eta/\eta_{\text{init}}$ |
|---|---|---|---|
| 0 | 0 | $2.704\times10^{-4}$ | 1 |
| 7500 | 0.25 | $8.552\times10^{-5}$ | $10^{-0.5}=0.3162$ |
| 15000 | 0.5 | $2.704\times10^{-5}$ | 0.1 |
| 30000 | 1 | $2.704\times10^{-6}$ | 0.01 |

### (c) Lịch step thưa dần — đếm bằng vòng lặp thật

Điều kiện chép đúng `optimizer_step` (`gaussian_model.py:190-209`): $t\le15000$: main mỗi vòng, SH khi $t\bmod16=0$; $15000\lt t\le20000$: cả hai khi $t\bmod32=0$; $t>20000$: cả hai khi $t\bmod64=0$. Vòng lặp `train.py:66` chạy $t=1..30000$ nhưng `train.py:161` chỉ gọi khi `iteration < opt.iterations` → step cho $t=1..29999$.

| | $\le15000$ | $(15000,20000]$ | $(20000,29999]$ | tổng |
|---|---|---|---|---|
| `optimizer` | 15000 | **157** | 156 | **15313** |
| `shoptimizer` | 937 | 157 | 156 | **1250** |
| cả hai | | | | **16563** |

3DGS gốc: $29999+29999=59998$ → tỉ số $16563/59998=0.2761$. Số bội của 32 trong $(15000,20000]$ là 157 ($20000=32\cdot625$, $15000/32=468.75$ → $625-468=157$). Kết quả code **khớp hoàn toàn** với bảng trong `06-gradient-flow.md` (15313 / 1250 / 16563 / 157).

### (d) $g_{\text{eff}}$: cộng 64 gradient rồi 1 step vs 64 step nhỏ

![Adam bằng số và decay learning rate](../Report/test/figures/ch06_adam.png)

*Hình: (a) sau 3 bước Adam với gradient lặp lại, m̂/(√v̂+ε) → 1.000 nên Δθ ≈ η cho cả 6 nhóm lr; (b) decay η_xyz(t) trên thang log; (c) cộng 64 gradient rồi step 1 lần cho bước đi nhỏ hơn nhiều so với 64 step nhỏ liên tiếp.*


64 gradient cùng dấu, $g_k=g(1+0.3\xi_k)$, $\xi\sim\mathcal N(0,1)$ (seed 0), $\lvert g\rvert$ trung bình $1.567\times10^{-4}$, $\sum g=1.003\times10^{-2}$; $\eta=\eta_{xyz}(0)=2.704\times10^{-4}$; $m=v=0$ ban đầu.

| cách | $\Delta\theta$ | $/\eta$ |
|---|---|---|
| 1 step với $g_{\text{eff}}=\sum_{64}g$ (giai đoạn $>20000$) | $-2.704\times10^{-4}$ | $-1.0000$ |
| 64 step nhỏ (giai đoạn $\le15000$) | $-1.720\times10^{-2}$ | $-63.60$ |

Tỉ số $63.6\times$: gộp 64 gradient rồi step một lần chỉ đi được **một** bước cỡ $\eta$ (Adam chuẩn hoá theo $\sqrt{\hat v}$, độ lớn $\sum g$ bị triệt), không phải 64 bước. Đó là lý do giai đoạn 15k–30k gần như không học thêm về vị trí.

## 6.5 — Đầu ra của khối

![Lịch step thưa dần của optimizer_step](../Report/test/figures/ch06_schedule.png)

*Hình: tần suất step của optimizer (xyz, f_dc, opacity, scaling, rotation) và shoptimizer (f_rest) theo t = 1..29999 — tổng 16563 so với 59998 của 3DGS gốc, tỉ số R_adam = 0.276.*


**Đầu vào của khối**: 4 Gaussian ($\mu,s,c$ ở bảng 6.0, $\alpha=0.1$), 3 camera, GT $=$ render $\alpha=0.9$, $\partial\mathcal L/\partial C=\operatorname{sign}/(3HW)$.

**Đầu ra của khối** (cho chương 7 — Adaptive Density Control), sau 3 iteration với camera 1, 2, 3:

| $i$ | $\text{accum}_i$ | $\text{accum}^{\text{abs}}_i$ | $\text{denom}_i$ | $\bar g_i$ | $\bar g^{\text{abs}}_i$ | $\bar g_i\ge\tau_{\text{grad}}$ | $\bar g^{\text{abs}}_i\ge\tau^{\text{abs}}_{\text{grad}}$ |
|---|---|---|---|---|---|---|---|
| 1 | $1.928867\times10^{-2}$ | $9.110511\times10^{-2}$ | 3 | $6.429555\times10^{-3}$ | $3.036837\times10^{-2}$ | True | True |
| 2 | $1.796432\times10^{-2}$ | $9.338040\times10^{-2}$ | 3 | $5.988106\times10^{-3}$ | $3.112680\times10^{-2}$ | True | True |
| 3 | $1.869827\times10^{-2}$ | $9.242443\times10^{-2}$ | 3 | $6.232756\times10^{-3}$ | $3.080814\times10^{-2}$ | True | True |
| 4 | $1.681544\times10^{-2}$ | $6.913455\times10^{-2}$ | 3 | $5.605145\times10^{-3}$ | $2.304485\times10^{-2}$ | True | True |

Kèm: $\eta_{xyz}(t)$ bảng 6.4(b), `extent` $=1.690$, và bước Adam $\lvert\Delta\theta\rvert\approx\eta$ cho mọi nhóm. Với chương 7: cả 4 Gaussian là ứng viên densify; theo chương 1, $\max s_i\in[0.8505,1.115]\gt \delta\cdot\text{extent}=0.00169$ nên đều đi nhánh **split**.

## Bài tập (Exercise)

**Bài tập 11.1.** Công thức $\partial C_{\text{ch}}/\partial\alpha_n$ ở mục 6.1 có ba số hạng: $c_{n,\text{ch}}T_n$, $-(C^{\text{tot}}_{\text{ch}}-C^{\le n}_{\text{ch}})/(1-\alpha_n)$, và $-T_{\text{final}}C_{\text{bg,ch}}/(1-\alpha_n)$. Giải thích bằng lời tại sao số hạng thứ ba tồn tại — nghĩa là vì sao nền (background) lại phụ thuộc vào $\alpha_n$ của một Gaussian $n$ bất kỳ nằm trước nó trong thứ tự depth. Dựa vào bảng ở mục 6.1(a) của phần kiểm định số (pixel $(23,15)$, $T_{\text{final}}=0.6894$), hãy nêu vì sao số hạng này "lớn nhất ở mọi $n$" trong ví dụ đó, và điều gì sẽ xảy ra với cảnh có $T_{\text{final}}\to0$ (ví dụ nhiều Gaussian mờ đục hơn).

**Bài tập 11.2.** Dùng đúng bảng ở mục 6.1(a) (4 Gaussian tại pixel $(23,15)$, thứ tự depth G1, G4, G2, G3), tính tay $\partial C_R/\partial\alpha_1$ (kênh R, Gaussian G1) từ ba số hạng đã cho: $(1)=(0.8,0.2,0.2)$, $(2)=(-0.06338,-0.1155,-0.1348)$, $(3)=-0.7657$. So sánh với giá trị $-0.02909$ trong bảng kết quả. Sau đó bỏ số hạng $(3)$ đi và tính lại — xác nhận kết quả đổi dấu thành $+0.737$ như văn bản đã nêu, và giải thích ý nghĩa vật lý của sự đổi dấu này (mô hình sẽ học sai hướng nếu quên số hạng nền).

**Bài tập 11.3.** Mục 6.1 giải thích vì sao `screenspace_points` cần 4 cột thay vì 3 (cột có dấu và cột trị tuyệt đối). Dựa trên bất đẳng thức $\lVert\sum_x g_x\rVert\le\sum_x\lVert g_x\rVert$ và số liệu Gaussian 1 ở mục 6.1(c) ($\lVert g\rVert=4.830\times10^{-4}$, $\lVert g^{\text{abs}}\rVert=3.705\times10^{-2}$, tỉ số $76.7\times$): giải thích tại sao Gaussian nằm gần tâm ảnh (footprint đối xứng qua biên vật thể) có tỉ số $\lVert g^{\text{abs}}\rVert/\lVert g\rVert$ lớn hơn nhiều so với Gaussian 4 (tỉ số chỉ $10.7$). Điều này gợi ý gì về việc dùng $g$ có dấu làm tiêu chí densify duy nhất (như 3DGS gốc)?

**Bài tập 11.4.** Trace code: đọc đoạn văn bản ở mục "Phát hiện khi đối chiếu code" (cuối mục 6.1(b)). Giải thích vì sao FD của $\partial\mathcal L/\partial B$ lớn gấp đúng 2 lần giá trị lưu trong `backward.cu:594`, và vì sao đây **không phải là lỗi**, bằng cách nêu vai trò của `backward.cu:212-214` trong `computeCov2DCUDA`. Nếu một sinh viên chỉ đọc công thức $-\tfrac12G\Delta_u\Delta_v$ trong `06-gradient-flow.md` mà không đọc code, họ có thể mắc lỗi gì khi tự cài đặt lại kernel backward?

**Bài tập 11.5.** Chương 6.4(c) so sánh "cộng 64 gradient rồi step 1 lần" với "64 step nhỏ liên tiếp", cho kết quả $\Delta\theta=-2.704\times10^{-4}$ (tức $-1.000\eta$) so với $-1.720\times10^{-2}$ (tức $-63.60\eta$). Giải thích bằng công thức Adam ($m\leftarrow\beta_1m+(1-\beta_1)g$, $v\leftarrow\beta_2v+(1-\beta_2)g^2$, $\theta\leftarrow\theta-\eta\hat m/(\sqrt{\hat v}+\epsilon)$) tại sao gộp gradient rồi step một lần luôn cho bước đi cỡ $\approx\eta$ bất kể độ lớn $\sum g$ — miễn $\lvert g\rvert$ đồng dấu và đủ lớn so với $\epsilon=10^{-15}$. Từ đó suy luận: nếu FastGS-lite hạ `--iterations` xuống 15000 (bỏ hẳn giai đoạn thưa dần 15k–30k), điều gì sẽ mất đi ngoài thời gian huấn luyện (gợi ý: đọc lại đoạn về `final_prune_fastgs` ở cuối mục 6.4).

**Bài tập 11.6.** Đếm chính xác số lần step trong bảng mục 6.4 (`optimizer_step`, `gaussian_model.py:190-209`): với lịch $\mathbb 1_{\text{main}}(t)$ và $\mathbb 1_{\text{SH}}(t)$ đã cho, hãy tự đếm số bội của 64 trong khoảng $(20000,29999]$ và xác nhận ra con số 156 đã nêu trong bảng. Sau đó giải thích bằng lời tại sao khoảng $(15000,20000]$ có đúng 157 bội của 32 (không phải 156 hay 158) — chú ý biên $20000=32\times625$ và $15000/32=468.75$.

**Bài tập 11.7.** So sánh sáu nhóm learning rate ở bảng mục 6.4: `xyz`, `f_dc`, `opacity`, `scaling`, `rotation` dùng `optimizer`, còn `f_rest` dùng `shoptimizer` riêng với lr $=\texttt{highfeature\_lr}/20=0.00025$. Dựa vào mục "Tách lr SH ≠ đòn bẩy tốc độ", giải thích vì sao bản thân giá trị lr nhỏ hơn **không** làm giảm chi phí tính toán, trong khi nhịp step thưa ($1/16$ trong giai đoạn $t\le15000$) mới là thứ giảm chi phí. Kết hợp cả hai yếu tố (lr nhỏ hơn 10× và số bước ít hơn 16×), hãy giải thích hiện tượng "vệt specular hội tụ chậm" được nhắc ở cuối mục 6.4.

---

[← Chương 10](10-anh-den-loss-metrics.md) | [Mục lục](00-muc-luc.md) | [Chương 12 →](12-adaptive-density-control.md)
