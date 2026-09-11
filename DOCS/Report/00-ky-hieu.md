# Chương 0 — Ký hiệu dùng chung

| Ký hiệu | Ý nghĩa | Nơi xuất hiện trong code |
|---|---|---|
| $N$ | số Gaussian hiện có | `gaussians.get_xyz.shape[0]` |
| $i$ | chỉ số Gaussian, $i=1..N$ | |
| $v$ | chỉ số camera / góc nhìn | `viewpoint_cam` |
| $\theta_i=(\mu_i,\tilde q_i,\tilde s_i,\tilde\alpha_i,k_i)$ | 59 tham số học được của Gaussian $i$ | `_xyz, _rotation, _scaling, _opacity, _features_dc/_rest` |
| $\mu_i\in\mathbb R^3$ | tâm (world space) | `_xyz` |
| $q_i,\ R_i=R(q_i)$ | quaternion đơn vị và ma trận xoay | `get_rotation` |
| $s_i\in\mathbb R^3_{>0},\ S_i=\operatorname{diag}(s_i)$ | scale | `get_scaling` |
| $\alpha_i\in(0,1)$ | opacity (sau sigmoid); code gọi là $o$ | `get_opacity` |
| $k_{i,lm}\in\mathbb R^3$ | hệ số SH, $l=0..3$, 16 hệ số/kênh | `get_features` |
| $\Sigma_i$ | covariance 3D | `computeCov3D` |
| $\Sigma'_i$ | covariance 2D sau chiếu | `computeCov2D` |
| $M_i=\Sigma_i'^{-1}=\begin{pmatrix}A&B\B&C\end{pmatrix}$ | conic | `con_o.x, .y, .z` |
| $\mu'_i\in\mathbb R^2$ | tâm trên ảnh (pixel) | `points_xy_image` |
| $\Delta=x-\mu'_i$ | độ lệch pixel–tâm | `d` trong `renderCUDA` |
| $t_i$ | ngưỡng level-set của compact box | `auxiliary.h:312-314` |
| $\mathcal K_i,\ K_i=\lvert\mathcal K_i\rvert$ | tập tile Gaussian $i$ chạm, và số tile | `tiles_touched[i]` |
| $K$ | số tile trung bình mỗi splat | |
| $P=\sum_iK_i\approx NK$ | tổng số cặp (tile, Gaussian) | `num_rendered` |
| $T_n$ | transmittance tích luỹ trước Gaussian thứ $n$ | `T` |
| $C(x)$ | màu pixel $x$ | `out_color` |
| $\mathcal L$ | loss huấn luyện | `loss` |
| $\text{extent}$ | bán kính cảnh ước lượng từ camera | `scene.cameras_extent` |
| $\text{Importance}_i,\ \text{Pruning}_i$ | hai điểm số multi-view của FastGS | `importance_score, pruning_score` |
| $\mathbb 1[\cdot]$ | hàm chỉ báo | |
| $\sigma(\cdot),\ \sigma^{-1}(\cdot)$ | sigmoid và nghịch đảo | `torch.sigmoid, inverse_sigmoid` |

Hằng số cứng hay gặp (đều nằm trong kernel hoặc `arguments/__init__.py`):

| Hằng | Giá trị | Ý nghĩa |
|---|---|---|
| tile | $16\times16$ px | `BLOCK_X, BLOCK_Y` |
| low-pass | $0.3$ | cộng vào đường chéo $\Sigma'$ |
| clamp Jacobian | $1.3\tan(\text{fov}/2)$ | |
| ngưỡng alpha | $1/255$ | bỏ splat mờ hơn 1 bước 8-bit |
| dừng sớm | $10^{-4}$ | dừng blend khi $T<10^{-4}$ |
| `mult` | $0.5$ | hệ số compact box |
| $\tau_{\text{grad}}$ | $2\times10^{-4}$ | `densify_grad_threshold` |
| $\tau^{\text{abs}}_{\text{grad}}$ | $1.2\times10^{-3}$ | `grad_abs_thresh` |
| $\tau_{\text{loss}}$ | $0.1$ | `loss_thresh` |
| $\delta$ | $0.001$ | `--dense` — ngưỡng clone/split theo scale |
| $\lambda$ | $0.2$ (preset notebook $0.25$) | `lambda_dssim` |
