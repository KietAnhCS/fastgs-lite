# Chương 1 — SfM Points → Initialization

> Khối đầu tiên của sơ đồ. **FastGS-lite giữ nguyên hoàn toàn** so với 3DGS gốc.
> Code: `scene/gaussian_model.py:137-160` (`create_from_pcd`), `scene/dataset_readers.py:45-66` (`getNerfppNorm`).

## 1.1 — Đầu vào

COLMAP cho ra:

- Point cloud thưa $\{(p_k,\ c_k)\}_{k=1}^{N_0}$, với $p_k\in\mathbb R^3$ là toạ độ world và $c_k\in[0,1]^3$ là màu RGB.
- Tập camera $\{(W_v,\ \text{fov}_{x,v},\ \text{fov}_{y,v},\ I^{(v)}_{\text{gt}})\}_{v=1}^{V}$ — ma trận view (world→camera), góc nhìn, ảnh ground-truth.

## 1.2 — Mỗi điểm SfM sinh một Gaussian

Mỗi $p_k$ trở thành một Gaussian với 59 tham số (đếm ở chương 2). Giá trị khởi tạo:

$$
\mu_i = p_i
$$

$$
\tilde s_i=\log\sqrt{\max\bigl(d^2_{\text{knn3}}(p_i),\ 10^{-7}\bigr)}\cdot\mathbf 1_3,
\qquad d^2_{\text{knn3}}(p_i)=\frac13\sum_{j\in\text{3-NN}(i)}\lVert p_i-p_j\rVert^2
$$

$$
\tilde q_i=(1,0,0,0),\qquad \tilde\alpha_i=\sigma^{-1}(0.1)=\log\frac{0.1}{0.9}
$$

$$
k_{i,00}=\text{RGB2SH}(c_i)=\frac{c_i-0.5}{C_0},\qquad C_0=\frac{1}{2\sqrt\pi}\approx0.28209,
\qquad k_{i,lm}=0\ \ (l\ge1)
$$

Ba điều đáng nhớ:

1. Gaussian ban đầu **đẳng hướng** (ba scale bằng nhau) và bằng khoảng cách tới láng giềng — đủ để phủ kín khe hở giữa các điểm SfM.
2. Opacity khởi tạo $0.1$, tức mờ. Phần lớn "hình" phải học từ gradient.
3. Chỉ SH bậc 0 có giá trị; 45 hệ số bậc cao bắt đầu từ 0 và chỉ được kích hoạt dần theo lịch $D(t)$ ở chương 2.

## 1.3 — `extent`: bán kính cảnh

Đại lượng này chưa được định nghĩa trong paper nhưng xuất hiện ở bốn ngưỡng của chương 7. Với $c_v=(W_v)^{-1}[{:}3,3]$ là vị trí camera $v$ trong world space:

$$
\bar c=\frac1V\sum_{v=1}^{V}c_v,
\qquad
\boxed{\ \text{extent}=1.1\cdot\max_v\lVert c_v-\bar c\rVert_2\ }
$$

Hằng $1.1$ là biên an toàn 10%. `extent` được dùng làm:

| Nơi dùng | Công thức |
|---|---|
| ngưỡng clone / split | $\delta\cdot\text{extent}$ (chương 7) |
| ngưỡng "scale quá lớn" khi prune | $0.1\cdot\text{extent}$ (chương 7) |
| `spatial_lr_scale` | $\eta_{xyz}\leftarrow\eta_{xyz}\cdot\text{extent}$ (chương 6) |

## 1.4 — Trạng thái optimizer đi kèm

Ngay khi khởi tạo, `training_setup` tạo hai Adam (chương 6): `optimizer` cho $(\mu,\tilde q,\tilde s,\tilde\alpha,k_{00})$ và `shoptimizer` riêng cho $k_{lm},\ l\ge1$. Mỗi tham số mang thêm hai moment $(m,v)$, nên bộ nhớ trạng thái là $\approx 3\times59=177$ float/Gaussian. Vì thế $N$ vừa quyết định thời gian, vừa quyết định VRAM.

## 1.5 — Đầu ra của khối

$$
\mathcal G_0=\{\theta_i\}_{i=1}^{N_0},\qquad N_0=\text{số điểm SfM}
$$

đi vào khối **3D Gaussians** (chương 2).
