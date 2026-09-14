# Chương 3 — Camera + Projection

> Khối **Projection** nhận 3D Gaussians và Camera, cho ra các splat 2D cùng tập tile mà mỗi splat chạm.
> Đây là **đòn bẩy 2 của FastGS-lite: giảm $K$** (số tile mỗi splat) bằng compact box và lọc tile theo ellipse.
> Code: `forward.cu` (`preprocessCUDA`, `computeCov2D`), `auxiliary.h:159-321` (`computeEllipseIntersection`, `processTiles`, `duplicateToTilesTouched`).
> Bài tập số đầy đủ: [`box.md`](../box.md); ba công thức đếm tile: [`box2.md`](../box2.md).

## 3.1 — Đầu vào từ Camera

$$
W_v\in\mathbb R^{4\times4}\ (\text{world}\to\text{camera}),\qquad
f_x=\frac{W}{2\tan(\text{fov}_x/2)},\qquad f_y=\frac{H}{2\tan(\text{fov}_y/2)}
$$

Ma trận chiếu (`utils/graphics_utils.py:51-71`), $z_n=0.01$, $z_f=100$:

$$
P=\begin{pmatrix}
\frac{1}{\tan(\text{fov}_x/2)}&0&0&0\\
0&\frac{1}{\tan(\text{fov}_y/2)}&0&0\\
0&0&\frac{z_f}{z_f-z_n}&\frac{-z_fz_n}{z_f-z_n}\\
0&0&1&0\end{pmatrix},
\qquad P_{\text{full}}=W_v\cdot P
$$

`--resolution` tác động vào đây: đổi $W,H$ → đổi $f_x,f_y$ → đổi $J$ → đổi $\Sigma'$ → đổi $K$.

## 3.2 — Bước 0: cull theo frustum

$$
t_i=W_v\begin{pmatrix}\mu_i\\1\end{pmatrix}=(t_x,t_y,t_z),\qquad
\text{loại nếu } t_z\le0.2
$$

Không chỉ để tiết kiệm: $J$ chứa $1/t_z$ và $1/t_z^2$, và khoá sort ở chương 4 `bit_cast` float→uint32 chỉ giữ đúng thứ tự khi depth **dương**. Cull là điều kiện đúng đắn.

## 3.3 — Bước 1: chiếu tâm

$$
p^{\text{hom}}=P_{\text{full}}\begin{pmatrix}\mu_i\\1\end{pmatrix},\qquad
p^{\text{proj}}=\frac{p^{\text{hom}}_{xyz}}{p^{\text{hom}}_w+10^{-7}}
$$

$$
\mu'_i=\Bigl(\tfrac{(p^{\text{proj}}_x+1)W-1}{2},\ \tfrac{(p^{\text{proj}}_y+1)H-1}{2}\Bigr)\quad[\text{pixel}]
$$

![Pinhole projection](assets/ch3_pinhole_projection.png)

*Trục x, y, z là toạ độ camera/world (z là chiều sâu $t_z$). Camera tại gốc toạ độ; điểm đỏ $\mu_i$ là tâm Gaussian 3D; mặt phẳng xanh là "mặt phẳng ảnh"; điểm xanh $\mu'_i$ là giao điểm tia chiếu với mặt phẳng đó — sau quy đổi pixel chính là tâm splat 2D.*

## 3.4 — Bước 2: chiếu covariance (EWA splatting)

**Clamp trước khi tuyến tính hoá** — chi tiết không có trong paper:

$$
\hat t_x=\operatorname{clip}\Bigl(\frac{t_x}{t_z},\ \pm1.3\tan\tfrac{\text{fov}_x}{2}\Bigr)\cdot t_z,\qquad \hat t_y\ \text{tương tự}
$$

Jacobian của phép chiếu phối cảnh, tuyến tính hoá tại tâm Gaussian:

$$
J=\begin{pmatrix}
\dfrac{f_x}{t_z}&0&-\dfrac{f_x\hat t_x}{t_z^2}\\[8pt]
0&\dfrac{f_y}{t_z}&-\dfrac{f_y\hat t_y}{t_z^2}\\[6pt]
0&0&0\end{pmatrix}
$$

$$
\Sigma'_{3\times3}=J\,W_v\,\Sigma_i\,W_v^\top J^\top
\quad\xrightarrow{\ \text{khối }2\times2\ }\quad
\boxed{\ \Sigma'_i=\begin{pmatrix}\Sigma'_{11}+0.3&\Sigma'_{12}\\ \Sigma'_{12}&\Sigma'_{22}+0.3\end{pmatrix}\ }
$$

![EWA splatting: 3D ellipsoid chiếu xuống 2D ellipse](assets/ch3_ewa_projection.png)

*Trái: mặt cắt ellipse của $\Sigma_i$ 3D, trục x/y world space. Phải: ellipse $\Sigma'_i$ sau khi chiếu 2D, trục u/v pixel — đã "phồng" thêm bởi low-pass $0.3I$ nên không bao giờ suy biến thành đường thẳng dù nhìn nghiêng.*

Hằng $0.3$ là **low-pass filter**: ép mọi splat rộng ít nhất ~1 pixel, tránh aliasing và gradient bằng 0 cho Gaussian nhỏ hơn pixel. Hệ quả toán học: $\det\Sigma'\ge0.09>0$, phép nghịch đảo ở bước sau luôn hợp lệ.

**Vì sao phải chiếu xuống 2D:** khoảng cách Mahalanobis trong 3D không tương đương tuyến tính với khoảng cách trên ảnh (foreshortening theo góc nhìn, phối cảnh theo khoảng cách). Chiếu một lần mỗi Gaussian mỗi khung hình rồi rasterize rẻ hơn nhiều so với ray-cast từng pixel.

## 3.5 — Bước 3: conic

$$
\det=\Sigma'_{11}\Sigma'_{22}-\Sigma'^2_{12},\qquad
M_i=\Sigma_i'^{-1}=\frac1{\det}\begin{pmatrix}\Sigma'_{22}&-\Sigma'_{12}\\-\Sigma'_{12}&\Sigma'_{11}\end{pmatrix}
=\begin{pmatrix}A&B\\B&C\end{pmatrix}
$$

Kernel đóng gói $(A,B,C,\alpha_i)$ vào một `float4` `con_o`. $\Sigma'$ không được lưu. Kiểm tra rẻ nhất: $\text{disc}=B^2-AC=-1/\det\Sigma'<0$.

## 3.6 — Bước 4: bounding box — **FastGS thay đổi ở đây**

### 3DGS gốc: hộp vuông $3\sigma$

$$
\lambda_{\max,\min}=\frac{\Sigma'_{11}+\Sigma'_{22}}{2}\pm\sqrt{\Bigl(\frac{\Sigma'_{11}-\Sigma'_{22}}{2}\Bigr)^2+\Sigma'^2_{12}},
\qquad r_i=\bigl\lceil3\sqrt{\lambda_{\max}}\bigr\rceil
$$

$$
\text{rect}_{\min}=\Bigl\lfloor\frac{\mu'-r}{16}\Bigr\rfloor,\quad
\text{rect}_{\max}=\Bigl\lfloor\frac{\mu'+r+15}{16}\Bigr\rfloor,\quad
K^{\text{3dgs}}_i=(\text{rect}_{\max}-\text{rect}_{\min})_x\cdot(\text{rect}_{\max}-\text{rect}_{\min})_y
$$

Với splat dẹt tỉ lệ trục $\rho=\sigma_{\max}/\sigma_{\min}$, hộp vuông lãng phí so với ellipse $3\sigma$:

$$
\frac{S_{\text{vuông}}}{S_{\text{ellipse}}}=\frac{36\sigma_{\max}^2}{9\pi\sigma_{\max}\sigma_{\min}}=\frac4\pi\rho
$$

![Tỉ lệ diện tích hộp vuông/ellipse theo độ dẹt ρ](assets/ch3_area_ratio.png)

*Trục x là độ dẹt $\rho=\sigma_{max}/\sigma_{min}$, trục y là tỉ lệ diện tích. Đường thẳng tăng tuyến tính: Gaussian càng dẹt (bề mặt nhìn nghiêng) thì phần diện tích thừa mà hộp vuông phải rasterize so với ellipse thật càng lớn.*

($\rho=4$ → lãng phí $5.09\times$.) Hộp $3\sigma$ cũng **không nhìn đến opacity**: splat mờ trả giá như splat đục.

### FastGS-lite: compact box theo tiêu chí "còn nhìn thấy ở 8-bit"

Pixel chỉ đáng rasterize nếu đóng góp $\ge$ một bước lượng tử 8-bit:

$$
\alpha_i\,e^{-\frac12\Delta^\top M_i\Delta}\ \ge\ \frac1{255}
\iff
\Delta^\top M_i\Delta\ \le\ 2\ln(255\,\alpha_i)
$$

Code nhân thêm `mult` vào ngưỡng (`auxiliary.h:312-314`):

$$
\boxed{\ t_i=\texttt{mult}\cdot2\ln(255\,\alpha_i)\ },\qquad\texttt{mult}=0.5
$$

Hộp bao **theo từng trục** của ellipse $\Delta^\top M\Delta=t$:

$$
\boxed{\ \text{half}_{x,i}=\sqrt{t_i\,\Sigma'_{11}},\qquad \text{half}_{y,i}=\sqrt{t_i\,\Sigma'_{22}}\ }
$$

$$
\text{Box}_i=[\mu'_x-\text{half}_x,\ \mu'_x+\text{half}_x]\times[\mu'_y-\text{half}_y,\ \mu'_y+\text{half}_y]
$$

![Compact box vs hộp 3σ vs ellipse, trên lưới tile](assets/ch3_compact_box.png)

*Trục u, v là pixel, lưới mảnh là biên tile 16×16. Hình vuông nét chấm xám là hộp $3\sigma$ của 3DGS gốc (không phụ thuộc $\alpha$); hình chữ nhật tím là compact box của FastGS (co theo $\alpha$ và mult); đường cong đỏ là biên ellipse thật $\Delta^\top M\Delta=t$. Hộp vuông lãng phí diện tích nhiều hơn hẳn so với ellipse thật, nhất là khi ellipse dẹt.*

Ba khác biệt bản chất:

| | 3DGS | FastGS |
|---|---|---|
| Hình dạng | vuông, cạnh $2r$ | chữ nhật $2\text{half}_x\times2\text{half}_y$ — hệ số $\tfrac4\pi\rho$ biến mất |
| Phụ thuộc $\alpha_i$ | không | có — splat mờ → $t$ nhỏ → hộp nhỏ |
| `mult` | — | nhân vào $t$ (dưới căn): **diện tích** co theo `mult`, **cạnh** co theo $\sqrt{\texttt{mult}}$ |

Bảng $t$ theo cấu hình (từ `box.md` câu 3):

| `mult` | $\alpha$ | $t$ | bán kính hiệu dụng $\sqrt t\,\sigma'$ |
|---|---|---|---|
| 1.0 | 1.0 | 11.08 | $3.33\sigma'$ |
| 0.5 | 1.0 | 5.54 | $2.35\sigma'$ |
| 0.5 | 0.1 | 3.24 | $1.80\sigma'$ |
| 0.5 | 0.02 | 1.63 | $1.28\sigma'$ |

Lưu ý: `mult = 1.0` **không** đưa về 3DGS gốc mà về SnugBox của Speedy-Splat ($\text{half}_x=3.33\sigma$ vẫn rộng hơn $3\sigma$). `getRect` đã bị xoá khỏi `auxiliary.h`; `my_radius` chỉ còn ghi vào `radii[]` cho ngưỡng prune 20 px.

## 3.7 — Bước 5: lọc tile theo ellipse (AccuTile)

3DGS lấy mọi tile giao hộp. FastGS chỉ lấy tile thực sự giao **ellipse**:

$$
\boxed{\ T\in\mathcal K_i\iff T\subset\text{Box}_i\ \wedge\ \min_{\Delta\in T}\Delta^\top M_i\Delta\le t_i\ }
$$

`processTiles` thực hiện bằng cách quét theo trục có span ngắn hơn (`isY = y_span < x_span`), mỗi lát giải giao ellipse–đường thẳng:

$$
v_\pm(u)=\frac{-B\,h\pm\sqrt{\text{disc}\cdot h^2+t\,k}}{k}+\mu'_v,\qquad h=u-\mu'_u,\quad k=\begin{cases}C&\text{cắt theo }x\\A&\text{cắt theo }y\end{cases}
$$

Với mỗi lát: nếu tiếp điểm cực trị ($x_{\text{term}}=-\tfrac BA\,\text{half}_y$, $y_{\text{term}}=-\tfrac BC\,\text{half}_x$) rơi vào trong lát thì lấy thẳng biên hộp, nếu không thì lấy $\min/\max$ tại hai biên lát; làm tròn **ra ngoài** tới biên tile.

Tỉ lệ diện tích ellipse / hộp bao (hiệu chỉnh cho ellipse nghiêng, `box.md` câu 9):

$$
\frac{S_{\text{ellipse}}}{S_{\text{box}}}=\frac{\pi t\sqrt{\det\Sigma'}}{4t\sqrt{\Sigma'_{11}\Sigma'_{22}}}=\frac\pi4\sqrt{1-\rho_{xy}^2},
\qquad\rho_{xy}=\frac{\Sigma'_{12}}{\sqrt{\Sigma'_{11}\Sigma'_{22}}}
$$

## 3.8 — Ví dụ số (từ `box.md`)

$\Sigma'=\begin{pmatrix}117&54\\54&36\end{pmatrix}$, $\mu'=(120,88)$, $\alpha=1$, `mult` $=0.5$:

| Bước | Kết quả |
|---|---|
| $\lambda_{\max},\lambda_{\min}$ | $144,\ 9$ → $\rho=4$ |
| $M=(A,B,C)$ | $(1/36,\ -1/24,\ 13/144)$ |
| $t$ | $\ln255=5.541$ |
| $\text{half}_x,\text{half}_y$ | $25.46,\ 14.12$ px |
| $K^{\text{3dgs}}$ (hộp vuông $r=36$) | $5\times5=\mathbf{25}$ |
| $K^{\text{box}}$ (chỉ compact box) | $5\times3=\mathbf{15}$ |
| $K^{\text{fastgs}}$ (sau lọc ellipse) | $3+3+3=\mathbf 9$ |
| $R_{\text{tile}}=K^{\text{fastgs}}/K^{\text{3dgs}}$ | $\mathbf{0.360}$ |

Chuỗi $25\to15\to9$: hộp theo trục $-40\%$, lọc ellipse $-40\%$ nữa. Tiệm cận diện tích cho $0.121$ nhưng đo được $0.360$ — chênh lệch do **lượng tử hoá lưới tile** (số hạng "+1" không co theo splat). Kernel trả tiền theo tile, nên $0.360$ mới là số đi vào mô hình chi phí.

## 3.9 — Bước 6: màu từ SH

Chỉ đánh giá khi $K_i>0$ (nếu không chạm tile nào thì `return` trước bước này). Công thức ở chương 2.4.

## 3.10 — Đầu ra của khối

$$
\bigl(\mu'_i,\ M_i,\ \alpha_i,\ c_i,\ \text{depth}_i=t_z,\ \mathcal K_i\bigr)_{i:\ K_i>0}
$$

đi vào **Differentiable Tile Rasterizer** (chương 4). $P=\sum_iK_i$ là kích thước của mọi buffer phía sau — compact box tiết kiệm cả VRAM, không chỉ thời gian.
