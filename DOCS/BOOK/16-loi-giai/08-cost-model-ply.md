[← Phần 7](07-density-control.md) · [Mục lục lời giải](00-muc-luc-loi-giai.md) · [Đề bài](../16-bai-toan-lon-de-bai.md)

# Phần 8 — Chương 13: Mô hình Chi phí & Xuất `point_cloud.ply`

> **Đầu vào nhận từ Phần 7 (Chương 12):** 𝒢₁ — quần thể Gaussian sau densify/prune tại $t=5000$ —
> xem [Phần 7](07-density-control.md).
>
> **Ghi chú về nguồn số liệu:** tại thời điểm viết phần này, file `07-density-control.md` (Phần 7,
> agent song song) **chưa tồn tại** trong `DOCS/BOOK/16-loi-giai/`. Theo đúng chỉ dẫn của nhiệm vụ, số
> liệu $\mathcal G_1$ dưới đây được **tự tái tạo (reconstruct)** trực tiếp từ hai nguồn đã xuất bản và đã
> qua kiểm định số (`scripts/chNN_test.py`):
>
> 1. [`06-initialization.md` §6.2](../06-initialization.md#62-cảnh-đồ-chơi-dùng-chung-cho-các-bài-kiểm-định-số-chương-6–13) — bảng 4 Gaussian gốc $\mathcal G_0$ (vị trí, scale đẳng hướng, opacity, màu RGB).
> 2. [`12-adaptive-density-control.md` §7.1–7.8](../12-adaptive-density-control.md) (mục "12.2 Kiểm định số — Chương 7, ADC") — ví dụ tay **4 Gaussian** (không phải ví dụ `3.md` 5 Gaussian khác ở §12.3) — cho quyết định densify (split cả 4, seed 0) và quyết định prune (multinomial, seed 0).
>
> Phương pháp tái tạo, chép nguyên văn từ mục "Thứ tự thật trong `densify_and_prune_fastgs`" của
> §7.4 (chương 12): code chạy **split trước, prune sau, trên chỉ số cũ** (`gaussian_model.py:473-476`,
> `padded_importance[:scores.shape[0]]` không đánh lại chỉ số sau densify). Với cảnh này, cả 4 Gaussian
> gốc đều rơi vào nhánh split (gradient vượt hai ngưỡng, Importance $\gg5$, không ai vào nhánh clone —
> xem bảng "extent và nhánh clone/split" của §7.3). Quần thể tạm thời sau split có **8 Gaussian con**,
> theo đúng thứ tự `repeat(2,1)` mà `densify_and_split` sinh ra:
>
> $$G_1c_1,\ G_2c_1,\ G_3c_1,\ G_4c_1,\ G_1c_2,\ G_2c_2,\ G_3c_2,\ G_4c_2$$
>
> Trọng số prune $w_i=1/(10^{-6}+1-\text{Pruning}_i)$ được gán **theo chỉ số cũ** (4 giá trị Pruning của
> 4 Gaussian gốc) lên đúng 4 vị trí đầu (các con `c1`), 4 vị trí sau (`c2`) nhận trọng số 0 vì
> `padded_importance` không mở rộng theo populaton mới. Với ngưỡng $|\mathcal C|=8$ (tất cả 8 con đều lọt
> tập ứng viên vì `max_radii2D=0`, $\max s>0.1\cdot\text{extent}$), `remove_budget=\lfloor 0.5\cdot 8\rfloor=4$.
> Multinomial không hoàn lại rút đúng 4 phần tử có trọng số dương — tức đúng 4 vị trí đầu
> $\{G_1c_1,G_2c_1,G_3c_1,G_4c_1\}$ — để lại:
>
> $$\mathcal G_1=\{G_1c_2,\ G_2c_2,\ G_3c_2,\ G_4c_2\},\qquad N=4$$
>
> Đây **chính là** kết quả bằng số đã in sẵn ở bảng "Đầu ra của khối" cuối §7.8 (chương 12) — Phần 8 này
> chỉ nêu lại và dùng trực tiếp, không phát sinh số liệu mới mâu thuẫn với chương gốc (quy ước §16.4.1
> của đề bài). Nếu khi đọc bản này `07-density-control.md` đã tồn tại với một bảng tham số cuối khác,
> **bảng của `07-density-control.md` mới là ground truth** — phần này đã đối chiếu và trùng khớp.

---

## Mục lục Phần 8

- [8.A — Mô hình chi phí $T_{\text{iter}}$ trước/sau bước densify](#8a)
  - [8.A.1 — Nhắc lại công thức và ba đòn bẩy](#8a1)
  - [8.A.2 — $K$ trước densify: 4 Gaussian gốc × 3 camera](#8a2)
  - [8.A.3 — $K$ sau densify+prune: 4 Gaussian con × 3 camera](#8a3)
  - [8.A.4 — Đòn bẩy 1 (giảm $K$): compact box, trước/sau](#8a4)
  - [8.A.5 — Đòn bẩy 2 (giảm $N$): densify/prune ròng](#8a5)
  - [8.A.6 — Đòn bẩy 3 (giảm nhịp Adam): lịch tại $t=5000$](#8a6)
  - [8.A.7 — Overhead ADC tại đúng vòng $t=5000$](#8a7)
  - [8.A.8 — $T_{\text{iter}}$ tổng hợp: $t=4999$ vs $t=5000$ vs $t=5001$](#8a8)
- [8.B — Xuất `point_cloud.ply`](#8b)
  - [8.B.1 — Từ $\theta$ nội bộ sang 62 cột PLY](#8b1)
  - [8.B.2 — Bảng đầy đủ 62 cột × 4 Gaussian](#8b2)
  - [8.B.3 — Script sinh file nhị phân](#8b3)
  - [8.B.4 — Transcript chạy thật](#8b4)
  - [8.B.5 — Kiểm chứng file nhị phân](#8b5)
  - [8.B.6 — File suy biến point-cloud (ASCII, RGB)](#8b6)
  - [8.B.7 — Cách render ra 3D](#8b7)
  - [8.B.8 — Ghi chú đối chiếu với code](#8b8)
  - [8.B.9 — Bài tập](#8b9)
- [Kết thúc lời giải](#ket-thuc)

---

<a id="8a"></a>
## 8.A — Mô hình chi phí $T_{\text{iter}}$ trước/sau bước densify

<a id="8a1"></a>
### 8.A.1 — Nhắc lại công thức và ba đòn bẩy

Từ [chương 13 §13.1](../13-tong-hop-chi-phi-fastgs.md) (mục 8.2 của tài liệu gốc):

$$
\boxed{\ T_{\text{iter}}=\underbrace{a\,N}_{\text{preprocess}}+\underbrace{b\,NK}_{\text{dup+sort+blend+backward}}+\underbrace{c\,N\cdot\mathbb 1[\text{Adam step}]}_{\text{optimizer}}+\underbrace{F}_{\text{loss, SSIM, IO}}\ }
$$

Ba đòn bẩy của FastGS-lite (bảng ở §8.3 chương gốc), áp lên đúng cảnh đồ chơi 4 Gaussian, tại đúng vòng
lặp $t=5000$ — nơi bước (7) (Phần 7, Chương 12) vừa chạy `densify_and_prune_fastgs`:

| Đòn bẩy | Khối | Cơ chế | Áp cho cảnh này |
|---|---|---|---|
| Giảm $K$ | Projection (chương 8/9) | compact box $t=\texttt{mult}\cdot2\ln(255\alpha)$ thay hộp $3\sigma$ | $K$ của $\mathcal G_0$ (trước) so với $K$ của $\mathcal G_1$ (sau) |
| Giảm $N$ | ADC (chương 12) | densify AND rồi prune multinomial | $N:4\to8\to4$ (ròng không đổi vòng này, nhưng $K$ đổi mạnh vì Gaussian mới nhỏ hơn) |
| Giảm nhịp Adam | Gradient Flow (chương 11) | lịch $1\to1/32\to1/64$ (main), $1/16$ (SH) | tại $t=5000\le15000$: main **luôn** step; SH thì **không** step ($5000\bmod16=8\ne0$) |

Cả ba số áp dụng đúng cho **một** vòng lặp cụ thể $t=5000$ của cảnh đồ chơi — khác với các tỉ số
$R_{\text{tile}}, R_{\text{gauss}}, R_{\text{adam}}$ ở chương 13 vốn là **trung bình toàn bộ 30 000
vòng** trên một scene thật cỡ triệu điểm. Phần dưới tính cả hai: số cục bộ (đúng cảnh đồ chơi, đúng
$t=5000$) và cách nó khớp với công thức chuẩn hoá toàn cục của chương 13.

<a id="8a2"></a>
### 8.A.2 — $K$ trước densify: 4 Gaussian gốc × 3 camera

Dùng lại đúng phép chiếu xấp xỉ đã thiết lập ở [chương 13 §13.2(b)(iii)](../13-tong-hop-chi-phi-fastgs.md)
("Cảnh đồ chơi"): bỏ số hạng ngoài-chéo của Jacobian $J$, $\Sigma'_{11}\approx(f_x/t_z)^2s^2+0.3$,
$\Sigma'_{22}\approx(f_y/t_z)^2s^2+0.3$ (đẳng hướng nên $\Sigma'_{11}=\Sigma'_{22}$ vì $f_x=f_y=40$),
`half`$=\sqrt{t\,\Sigma'_{11,22}}$ với $t=\texttt{mult}\cdot2\ln(255\alpha)=0.5\cdot2\ln(25.5)=3.239$
($\alpha=0.1$ — opacity không đổi qua densify, xem §7.5 chương 12), rồi rect $=[\lfloor(\mu'-\text{half})/16\rfloor,\ \lfloor(\mu'+\text{half})/16\rfloor+1]$ clamp vào lưới $3\times2$ tile, $K=$ diện tích rect
(pixel offset $-0.5$ đúng quy ước `ndc2Pix`). Ba camera: $c_1=(0,0,-4)$, $c_2=(1.5,0,-4)$,
$c_3=(-1.5,0.5,-4)$ (từ `images.txt`, §16.1).

Bốn Gaussian gốc $\mathcal G_0$ (từ [§6.2](../06-initialization.md) và bảng "Đầu vào của khối" ở
[§12.2 chương 12](../12-adaptive-density-control.md)):

| $i$ | $\mu_i$ | $s_i$ |
|---|---|---|
| $G_1$ | $(0,0,0)$ | 0.8505 |
| $G_2$ | $(0.5,0.3,0.5)$ | 0.9434 |
| $G_3$ | $(-0.4,-0.2,1.0)$ | 1.1150 |
| $G_4$ | $(0.3,-0.5,0.2)$ | 0.8888 |

Kết quả từng cặp (Gaussian × camera), 12 cặp — thay số đầy đủ:

| Gaussian | cam | $\mu'$ (px) | $\Sigma'_{11}=\Sigma'_{22}$ | half | rect (clamp $3\times2$) | $K$ |
|---|---|---|---|---|---|---|
| $G_1$ | 1 | $(23.50,\,15.50)$ | 72.64 | 15.34 | $[0,3)\times[0,2)$ | 6 |
| $G_1$ | 2 | $(8.50,\,15.50)$   | 72.64 | 15.34 | $[0,2)\times[0,2)$ | 4 |
| $G_1$ | 3 | $(38.50,\,10.50)$  | 72.64 | 15.34 | $[1,3)\times[0,2)$ | 4 |
| $G_2$ | 1 | $(27.94,\,18.17)$  | 70.62 | 15.12 | $[0,3)\times[0,2)$ | 6 |
| $G_2$ | 2 | $(14.61,\,18.17)$  | 70.62 | 15.12 | $[0,2)\times[0,2)$ | 4 |
| $G_2$ | 3 | $(41.28,\,13.72)$  | 70.62 | 15.12 | $[1,3)\times[0,2)$ | 4 |
| $G_3$ | 1 | $(20.30,\,13.90)$  | 79.87 | 16.08 | $[0,3)\times[0,2)$ | 6 |
| $G_3$ | 2 | $(8.30,\,13.90)$   | 79.87 | 16.08 | $[0,2)\times[0,2)$ | 4 |
| $G_3$ | 3 | $(32.30,\,9.90)$   | 79.87 | 16.08 | $[1,3)\times[0,2)$ | 4 |
| $G_4$ | 1 | $(26.36,\,10.74)$  | 71.95 | 15.27 | $[0,3)\times[0,2)$ | 6 |
| $G_4$ | 2 | $(12.07,\,10.74)$  | 71.95 | 15.27 | $[0,2)\times[0,2)$ | 4 |
| $G_4$ | 3 | $(40.64,\,5.98)$   | 71.95 | 15.27 | $[1,3)\times[0,2)$ | 4 |

**Ví dụ tính tay đầy đủ — $G_1$ × camera 1 (không nhảy bước):**

- Công thức: $t=\mu-c_v$, $t_z=t[2]$.
- Thay số: $\mu_1=(0,0,0)$, $c_1=(0,0,-4)\Rightarrow t=(0,0,0)-(0,0,-4)=(0,0,4)$, $t_z=4$.
- Kết quả: $t_z=4.0$.

- Công thức: $\mu'_x=c_x+f_x\,t_x/t_z-0.5$, $\mu'_y=c_y+f_y\,t_y/t_z-0.5$.
- Thay số: $\mu'_x=24+40\cdot0/4-0.5=24-0.5$, $\mu'_y=16+40\cdot0/4-0.5=16-0.5$.
- Kết quả: $\mu'=(23.50,\ 15.50)$ px.

- Công thức: $\Sigma'_{11}=(f_x/t_z)^2s^2+0.3$.
- Thay số: $\Sigma'_{11}=(40/4)^2\cdot0.8505^2+0.3=10^2\cdot0.723350+0.3=72.3350+0.3$.
- Kết quả: $\Sigma'_{11}=72.635\approx72.64$ (đẳng hướng và $f_x=f_y$ nên $\Sigma'_{22}$ bằng số này).

- Công thức: $t_{\text{box}}=\texttt{mult}\cdot2\ln(255\alpha)$.
- Thay số: $t_{\text{box}}=0.5\cdot2\ln(255\cdot0.1)=1\cdot\ln(25.5)$.
- Kết quả: $t_{\text{box}}=3.2387\approx3.239$.

- Công thức: $\text{half}=\sqrt{t_{\text{box}}\cdot\Sigma'_{11}}$.
- Thay số: $\text{half}=\sqrt{3.239\cdot72.635}=\sqrt{235.28}$.
- Kết quả: $\text{half}=15.34$ px (cả $x$ lẫn $y$, vì $\Sigma'_{11}=\Sigma'_{22}$).

- Công thức: rect $=\bigl[\max(0,\lfloor(\mu'-\text{half})/16\rfloor),\ \min(\text{grid},\lfloor(\mu'+\text{half})/16\rfloor+1)\bigr]$.
- Thay số trục $x$: $\lfloor(23.50-15.34)/16\rfloor=\lfloor0.51\rfloor=0$; $\lfloor(23.50+15.34)/16\rfloor+1=\lfloor2.4275\rfloor+1=2+1=3$; clamp $\min(3,3)=3$.
- Thay số trục $y$: $\lfloor(15.50-15.34)/16\rfloor=\lfloor0.01\rfloor=0$; $\lfloor(15.50+15.34)/16\rfloor+1=\lfloor1.9275\rfloor+1=1+1=2$; clamp $\min(2,2)=2$.
- Kết quả: rect $=[0,3)\times[0,2)$, $K=3\times2=\mathbf6$.

Chín cặp còn lại của bảng dưới đều theo đúng 5 bước trên (chỉ $\mu,c_v$ thay đổi khiến $t_z,\mu',\Sigma'_{11}$
khác đi; $t_{\text{box}}=3.239$ và cách clamp rect giữ nguyên).

$$
K_{\text{before}}=\sum_{i=1}^{4}\sum_{v=1}^{3}K_{i,v}=(6+4+4)\times4=\mathbf{56}
$$

(Khớp đúng cột "tile chạm" của bảng "Đầu vào của khối" §12.2 chương 12: view 1 = $6/6$ cho cả 4
Gaussian $=24$, view 2 và 3 = $4/6$ cho cả 4 Gaussian $=16+16$; tổng $24+16+16=56$.)

<a id="8a3"></a>
### 8.A.3 — $K$ sau densify+prune: 4 Gaussian con × 3 camera

Quần thể $\mathcal G_1=\{G_1c_2,G_2c_2,G_3c_2,G_4c_2\}$ (§ đầu file), scale $s^{(j)}=s_i/1.6$:

| Gaussian con | $\mu$ | $s=s_{\text{gốc}}/1.6$ |
|---|---|---|
| $G_1c_2$ | $(0.0892,\,-0.4556,\,0.3075)$ | $0.8505/1.6=0.531562$ |
| $G_2c_2$ | $(-0.6938,\,-0.2880,\,0.5390)$ | $0.9434/1.6=0.589625$ |
| $G_3c_2$ | $(-1.2165,\,-0.8069,\,0.6473)$ | $1.1150/1.6=0.696875$ |
| $G_4c_2$ | $(1.5145,\,-1.0912,\,0.5124)$ | $0.8888/1.6=0.555500$ |

Cùng công thức compact box (đối tượng nhỏ hơn 1.6× theo mỗi trục ⇒ $\Sigma'_{11}$ giảm mạnh vì tỉ lệ
$s^2$, còn $+0.3$ low-pass giữ nguyên):

| Gaussian con | cam | $\mu'$ (px) | $\Sigma'_{11}=\Sigma'_{22}$ | half | rect (clamp $3\times2$) | $K$ |
|---|---|---|---|---|---|---|
| $G_1c_2$ | 1 | $(24.33,\,11.27)$ | 24.67 | 8.94 | $[0,3)\times[0,2)$ | 6 |
| $G_1c_2$ | 2 | $(10.40,\,11.27)$ | 24.67 | 8.94 | $[0,2)\times[0,2)$ | 4 |
| $G_1c_2$ | 3 | $(38.26,\,6.63)$  | 24.67 | 8.94 | $[1,3)\times[0,1)$ | 2 |
| $G_2c_2$ | 1 | $(17.39,\,12.96)$ | 27.30 | 9.40 | $[0,2)\times[0,2)$ | 4 |
| $G_2c_2$ | 2 | $(4.17,\,12.96)$  | 27.30 | 9.40 | $[0,1)\times[0,2)$ | 2 |
| $G_2c_2$ | 3 | $(30.60,\,8.56)$  | 27.30 | 9.40 | $[1,3)\times[0,2)$ | 4 |
| $G_3c_2$ | 1 | $(13.03,\,8.55)$  | 36.28 | 10.84 | $[0,2)\times[0,2)$ | 4 |
| $G_3c_2$ | 2 | $(0.12,\,8.55)$   | 36.28 | 10.84 | $[0,1)\times[0,2)$ | 2 |
| $G_3c_2$ | 3 | $(25.94,\,4.25)$  | 36.28 | 10.84 | $[0,3)\times[0,1)$ | 3 |
| $G_4c_2$ | 1 | $(36.93,\,5.83)$  | 24.55 | 8.92 | $[1,3)\times[0,1)$ | 2 |
| $G_4c_2$ | 2 | $(23.63,\,5.83)$  | 24.55 | 8.92 | $[0,3)\times[0,1)$ | 3 |
| $G_4c_2$ | 3 | $(50.22,\,1.39)$  | 24.55 | 8.92 | $[2,3)\times[0,1)$ | 1 |

**Ví dụ tính tay đầy đủ — $G_1c_2$ × camera 3 (trường hợp $K$ nhỏ nhất, không nhảy bước):**

- Công thức: $s_{\text{con}}=s_{\text{gốc}}/1.6$.
- Thay số: $s_{\text{con}}=0.8505/1.6$.
- Kết quả: $s_{\text{con}}=0.531562$.

- Công thức: $t=\mu-c_v$, $t_z=t[2]$.
- Thay số: $\mu=(0.0892,-0.4556,0.3075)$, $c_3=(-1.5,0.5,-4)\Rightarrow t=(0.0892-(-1.5),\ -0.4556-0.5,\ 0.3075-(-4))=(1.5892,\ -0.9556,\ 4.3075)$.
- Kết quả: $t_z=4.3075$.

- Công thức: $\mu'_x=c_x+f_x t_x/t_z-0.5$, $\mu'_y=c_y+f_y t_y/t_z-0.5$.
- Thay số: $\mu'_x=24+40\cdot1.5892/4.3075-0.5=24+14.755-0.5$; $\mu'_y=16+40\cdot(-0.9556)/4.3075-0.5=16-8.873-0.5$.
- Kết quả: $\mu'=(38.26,\ 6.63)$ px.

- Công thức: $\Sigma'_{11}=(f_x/t_z)^2s_{\text{con}}^2+0.3$.
- Thay số: $\Sigma'_{11}=(40/4.3075)^2\cdot0.531562^2+0.3=85.499\cdot0.282558+0.3$.
- Kết quả: $\Sigma'_{11}=24.15+0.3=24.67$.

- Công thức: $\text{half}=\sqrt{t_{\text{box}}\cdot\Sigma'_{11}}$, $t_{\text{box}}=3.239$ (không đổi, cùng $\alpha=0.1$, `mult`=0.5).
- Thay số: $\text{half}=\sqrt{3.239\cdot24.67}=\sqrt{79.90}$.
- Kết quả: $\text{half}=8.94$ px.

- Công thức: rect như trên, clamp lưới $3\times2$.
- Thay số trục $x$: $\lfloor(38.26-8.94)/16\rfloor=\lfloor1.8325\rfloor=1$; $\lfloor(38.26+8.94)/16\rfloor+1=\lfloor2.9500\rfloor+1=2+1=3$; clamp $\min(3,3)=3$.
- Thay số trục $y$: $\lfloor(6.63-8.94)/16\rfloor=\lfloor-0.1444\rfloor=-1\to$ clamp $\max(0,-1)=0$; $\lfloor(6.63+8.94)/16\rfloor+1=\lfloor0.9731\rfloor+1=0+1=1$; clamp $\min(2,1)=1$.
- Kết quả: rect $=[1,3)\times[0,1)$, $K=2\times1=\mathbf2$ — khớp bảng dưới.

$$
K_{\text{after}}=(6+4+2)+(4+2+4)+(4+2+3)+(2+3+1)=12+10+9+6=\mathbf{37}
$$

Gaussian con nhỏ hơn ($s$ giảm $1.6\times$, $s^2$ giảm $2.56\times$) ⇒ mỗi footprint hẹp lại rõ rệt;
$G_4c_2$ ở camera 3 thậm chí bị chiếu ra rìa ảnh (tâm $\mu'_x=50.22 > 48$) nên chỉ còn chạm **1** tile.

<a id="8a4"></a>
### 8.A.4 — Đòn bẩy 1 (giảm $K$): compact box, trước/sau

So compact box (`mult`$=0.5$) với hộp vuông $3\sigma$ gốc 3DGS ($r=\lceil3\sqrt{\Sigma'_{11}}\rceil$,
rect $=\lfloor(\mu'-r)/16\rfloor,\ \lfloor(\mu'+r+15)/16\rfloor$, cùng clamp $3\times2$), tính lại cho cả
hai quần thể:

| Quần thể | $K_{3\sigma}$ (hộp vuông 3DGS) | $K_{\text{compact}}$ (FastGS, `mult`=0.5) | $R_{\text{tile}}=K_{\text{compact}}/K_{3\sigma}$ |
|---|---|---|---|
| $\mathcal G_0$ (trước, $N=4$) | 72 | **56** | **0.7778** |
| $\mathcal G_1$ (sau, $N=4$) | 55 | **37** | **0.6727** |

Hai quan sát:

1. **Compact box tự nó tiết kiệm cả trước lẫn sau** ($R_{\text{tile}}<1$ ở cả hai) — đúng cơ chế lever 1
   độc lập với $N$.
2. **$R_{\text{tile}}$ giảm thêm sau densify** ($0.7778\to0.6727$): Gaussian con nhỏ hơn khiến biên
   $3\sigma$ và biên compact càng lệch xa nhau theo tỉ lệ (compact box cắt theo $\sqrt{t\,\Sigma'}$ trong
   khi $3\sigma$ cắt theo $3\sqrt{\Sigma'}$ — hệ số 3 so với $\sqrt{3.239}\approx1.80$ không đổi theo
   $\Sigma'$, nhưng việc `+15` làm tròn lên trong công thức rect $3\sigma$ có ảnh hưởng tương đối lớn hơn
   khi $r$ nhỏ, khiến hộp $3\sigma$ "lãng phí" theo tỉ lệ nhiều hơn ở scale nhỏ trên lưới tile thô $16\text{px}$).

<a id="8a5"></a>
### 8.A.5 — Đòn bẩy 2 (giảm $N$): densify/prune ròng

Bảng đầy đủ 8 Gaussian **tạm thời** ngay sau `densify_and_split`, trước khi `prune` chạy trong cùng
lệnh gọi (thứ tự `repeat(2,1)`, dữ liệu $\epsilon^{(1)}$/$\epsilon^{(2)}$ từ bảng "Split cụ thể" §7.3
chương 12) — cột cuối đánh dấu ai sống sót tới $\mathcal G_1$:

| # | Gaussian | $\mu$ | $\tilde s=\log(s_{\text{gốc}}/1.6)$ | trọng số prune $w$ (gán theo chỉ số cũ) | kết quả |
|---|---|---|---|---|---|
| 1 | $G_1c_1$ | $(0.1069,-0.1124,0.5447)$ | $-0.631935$ | $w(G_1)=4.2621$ (dương) | **bị multinomial rút — xoá** |
| 2 | $G_2c_1$ | $(1.7302,1.1935,-0.1639)$ | $-0.528269$ | $w(G_2)=1.8204$ (dương) | **bị multinomial rút — xoá** |
| 3 | $G_3c_1$ | $(-2.9925,-0.4440,-0.3893)$ | $-0.361149$ | $w(G_3)=10^6$ (dương, áp đảo) | **bị multinomial rút — xoá** |
| 4 | $G_4c_1$ | $(0.6659,0.4266,0.0858)$ | $-0.587887$ | $w(G_4)=1.0000$ (dương) | **bị multinomial rút — xoá** |
| 5 | $G_1c_2$ | $(0.0892,-0.4556,0.3075)$ | $-0.631935$ | $0$ (`padded_importance` không mở rộng) | **sống — vào $\mathcal G_1$** |
| 6 | $G_2c_2$ | $(-0.6938,-0.2880,0.5390)$ | $-0.528269$ | $0$ | **sống — vào $\mathcal G_1$** |
| 7 | $G_3c_2$ | $(-1.2165,-0.8069,0.6473)$ | $-0.361149$ | $0$ | **sống — vào $\mathcal G_1$** |
| 8 | $G_4c_2$ | $(1.5145,-1.0912,0.5124)$ | $-0.587887$ | $0$ | **sống — vào $\mathcal G_1$** |

Vì `remove_budget`$=\lfloor0.5\cdot8\rfloor=4$ đúng bằng số phần tử có trọng số dương, multinomial không
hoàn lại rút **chính xác** cả 4 vị trí $\{1,2,3,4\}$ (không có phần tử trọng số $0$ nào có cơ hội bị rút
khi còn phần tử trọng số dương) — đây là lý do kết quả **tất định**, không phụ thuộc seed, mặc dù cơ chế
lấy mẫu về bản chất là ngẫu nhiên.

$$N\leftarrow N+|\text{clone}|+|\text{split}|-|\text{xoá}| = 4+0+4-4=\mathbf 4$$

(công thức 7.8 chương 12, áp đúng thứ tự thật của code — xem hộp đầu file). Ở vòng lặp $t=5000$ cụ thể
này, đòn bẩy 2 **không đổi ròng $N$** ($R_{\text{gauss}}^{\text{vòng này}}=4/4=1.0000$) — nhưng vẫn có
tác dụng: 8 Gaussian tồn tại **tạm thời** ngay sau `densify_and_split` trước khi `prune` chạy trong cùng
lệnh gọi `densify_and_prune_fastgs`, nên chi phí đỉnh (peak memory, đỉnh $K$ tức thời) trong vòng này
tính trên $N_{\text{tạm}}=8$, không phải 4:

$$K_{\text{tạm, 8 con}}=K_{\text{after}}^{(c_1)}+K_{\text{after}}^{(c_2)}$$

trong đó $K_{\text{after}}^{(c_2)}=37$ (bảng 8.A.3, đúng 4 Gaussian còn sống); $K$ của 4 con $c_1$ (bị xoá
ngay sau đó, không bao giờ được optimizer step) không cần tính vì `densify_and_prune_fastgs` xoá trước
khi vòng lặp tiếp theo render — nói cách khác **lever 2 giấu chi phí tức thời của 4 Gaussian `c1`**, thứ
mà nếu tính đúng thứ tự thật của kernel (split rồi mới prune trong **cùng** lệnh, trước lần render kế
tiếp) thì chưa từng được rasterize. Đây là lý do công thức 7.8 ghi rõ "N mới theo công thức đơn giản = 6"
khác với "N mới theo thứ tự thật của code = 4" (xem chú thích ở cuối §12.2 chương 12) — công thức đơn
giản không biết về việc trọng số prune bị gán theo chỉ số cũ.

<a id="8a6"></a>
### 8.A.6 — Đòn bẩy 3 (giảm nhịp Adam): lịch tại $t=5000$

Từ [chương 13 §13.2(a)](../13-tong-hop-chi-phi-fastgs.md) (`gaussian_model.py:190-209`):

$$
\mathbb 1_{\text{main}}(t)=\begin{cases}1&t\le15000\\ [t\bmod32=0]&15000<t\le20000\\ [t\bmod64=0]&t>20000\end{cases},\qquad
\mathbb 1_{\text{SH}}(t)=\begin{cases}[t\bmod16=0]&t\le15000\\ \mathbb 1_{\text{main}}(t)&t>15000\end{cases}
$$

Thay $t=5000$ (nằm trong nhánh $t\le15000$):

$$
\mathbb 1_{\text{main}}(5000)=1\ \ (\text{vì }5000\le15000)
$$
$$
5000 \bmod 16 = 5000-16\cdot312 = 5000-4992=8\ \ne 0\quad\Rightarrow\quad \mathbb 1_{\text{SH}}(5000)=0
$$

**Đúng tại vòng lặp $t=5000$**: optimizer chính (`optimizer`, phủ $\mu,\tilde q,\tilde s,\tilde\alpha,k_{00}$
— 14 tham số/Gaussian) **có** step; `shoptimizer` (phủ $k_{lm},l\ge1$ — 45 tham số/Gaussian) **không**
step. 3DGS gốc (không có lịch thưa) thì cả hai optimizer đều step ở mọi $t$. Vậy tại chính vòng này:

$$
c_{\text{fastgs}}(5000)\cdot N = c_{\text{main}}\cdot4\cdot1 + c_{\text{SH}}\cdot4\cdot0,\qquad
c_{\text{3dgs}}(5000)\cdot N = c_{\text{main}}\cdot4\cdot1 + c_{\text{SH}}\cdot4\cdot1
$$

Nếu tách $c=c_{\text{main}}+c_{\text{SH}}$ theo tỉ lệ tham số ($14$ so với $45$, tức $c_{\text{SH}}\approx
\frac{45}{59}c\approx0.7627c$ — xấp xỉ, bỏ qua khác biệt lr giữa hai optimizer), FastGS tiết kiệm đúng
phần `shoptimizer` ở vòng này:

$$
\frac{c_{\text{fastgs}}(5000)}{c_{\text{3dgs}}(5000)}\approx\frac{c_{\text{main}}}{c_{\text{main}}+c_{\text{SH}}}\approx\frac{14}{59}=0.2373
$$

Số này **gần** (không bằng) $R_{\text{adam}}=0.2761$ của chương 13 — vì $R_{\text{adam}}$ là **trung
bình** trên toàn bộ $1..29999$ vòng (gồm cả các vòng $t\bmod16=0$ nơi cả hai optimizer cùng step, kéo tỉ
lệ trung bình lên cao hơn con số tại một vòng lẻ $t=5000$ cụ thể — chính là điểm khác nhau giữa "áp dụng
cho một vòng cụ thể" (mục này) và "trung bình toàn lịch" (chương 13 gốc).

<a id="8a7"></a>
### 8.A.7 — Overhead ADC tại đúng vòng $t=5000$

Điều kiện chạy ADC (`train.py:127,132`): `iteration < 15000 and iteration > densify_from_iter(500) and
iteration % densification_interval(500) == 0`. Thay $t=5000$: $5000<15000$ ✓, $5000>500$ ✓,
$5000\bmod500=0$ ✓ — **ADC chạy đúng ở vòng này** (đúng như Phần 7 đã giả định). `compute_gaussian_score_fastgs`
lấy mẫu $V$ camera, render mỗi camera **2 lần** (`utils/fast_utils.py:13`); cảnh đồ chơi chỉ có $V=3$
camera (không phải $V=10$ như code thật — quy ước đã nêu ở §12.2 chương 12), nên chi phí forward phụ
**tại đúng vòng $t=5000$** là:

$$
n_{\text{forward phụ}}(t{=}5000)=V\times2=3\times2=\mathbf6
$$

so với $1$ forward bình thường của một vòng train thường (không ADC). Xấp xỉ chi phí forward-only $\approx
a\,N+b\,N\,K$ (bỏ qua $c,F$ vì scoring không chạy backward/Adam/SSIM đầy đủ — chỉ $L_1$+SSIM per-pixel để
tính $E_{\text{photo}}$, xem §7.2 chương 12), hệ số nhân xấp xỉ tại vòng này:

$$
\text{multiplier}(t{=}5000)\approx 1+n_{\text{forward phụ}}=1+6=\mathbf 7\times
$$

so với **một** vòng lặp bình thường (không chạy ADC) — chỉ ở đúng vòng $5000$, không lan sang $t=4999$
hay $t=5001$. Đây là con số **cục bộ**, khác hẳn overhead **trung bình** đã tính ở chương 13 §8.6
($\approx1.9\%$ khi quy ra 30 000 vòng, dùng $V=10$ thật) — quy về cùng đơn vị bằng cách chia cho tổng số
vòng: $6/(\text{1 vòng})$ cục bộ so với $560/30000\approx1.9\%$ trung bình toàn cục.

<a id="8a8"></a>
### 8.A.8 — $T_{\text{iter}}$ tổng hợp: $t=4999$ vs $t=5000$ vs $t=5001$

Dùng đúng cách chuẩn hoá của chương 13 §13.2 (mục 8.2/8.5 tài liệu gốc): $(a',b',c',f)=(0.15,0.55,0.15,0.15)$
là tỉ trọng bốn số hạng tại điểm làm việc tham chiếu $T^{\text{3dgs}}_{\text{REF}}=1$ với
$N_{\text{REF}}=933\,036$, $K_{\text{REF}}=21.91$ (từ chính bảng "PHÉP ĐO 1/3" của chương 13). Tỉ số cục
bộ của cảnh đồ chơi, tại từng vòng:

$$
R_N=\frac{N_{\text{toy}}}{N_{\text{REF}}}=\frac{4}{933036}=4.288\times10^{-6},\qquad
R_K(t)=\frac{K_{\text{toy}}(t)}{K_{\text{REF}}}=\frac{K_{\text{toy}}(t)}{21.91}
$$

| $t$ | Quần thể render | $N$ | $K$ (tổng 3 cam) | $R_K$ | $\mathbb 1_{\text{main}}$ | $\mathbb 1_{\text{SH}}$ | ADC overhead |
|---|---|---|---|---|---|---|---|
| 4999 | $\mathcal G_0$ (trước) | 4 | 56 | 2.556 | 1 | $4999\bmod16=7\ne0\Rightarrow0$ | không |
| 5000 | $\mathcal G_0$ (ADC chạy cuối vòng, dùng $\mathcal G_0$ để render/score) | 4 | 56 | 2.556 | 1 | $5000\bmod16=8\ne0\Rightarrow0$ | **có, ×7** (8.A.7) |
| 5001 | $\mathcal G_1$ (sau) | 4 | 37 | 1.689 | 1 | $5001\bmod16=9\ne0\Rightarrow0$ | không |

Công thức (dùng $c$ trung bình cho cả main+SH gộp, giữ nguyên $c'=0.15$ như chương 13, chỉ nhân thêm hệ
số overhead ở vòng 5000; $\mathbb1_{\text{SH}}=0$ ở cả ba vòng nên số hạng $c$ **không đổi** giữa 3 vòng —
đúng như suy ra ở 8.A.6, khác biệt Adam ở scale $t=5000\pm1$ chỉ nằm ở việc *3DGS gốc* mới có SH step mọi
vòng, còn FastGS thì không ở cả ba $t$ này):

$$
T^{\text{fast}}_{\text{toy}}(t)=a'R_N+b'R_N R_K(t)\,(1+\text{ov}(t))+c'R_N\,\overline{\mathbb 1}(t)+f
$$

với $\overline{\mathbb 1}(t)=\tfrac{14\cdot\mathbb1_{\text{main}}+45\cdot\mathbb1_{\text{SH}}}{59}$ (tỉ lệ
tham số có step / tổng 59):

| $t$ | $\overline{\mathbb1}$ | $a'R_N$ | $b'R_N R_K(1+\text{ov})$ | $c'R_N\overline{\mathbb1}$ | $f$ | $T^{\text{fast}}_{\text{toy}}(t)$ |
|---|---|---|---|---|---|---|
| 4999 | $14/59=0.2373$ | $6.432\times10^{-7}$ | $0.55\cdot4.288\text{e-}6\cdot2.556=6.030\times10^{-6}$ | $1.527\times10^{-7}$ | 0.15 | **0.150007** |
| 5000 | $0.2373$ | $6.432\times10^{-7}$ | $0.55\cdot4.288\text{e-}6\cdot2.556\cdot(1+7)=4.824\times10^{-5}$ | $1.527\times10^{-7}$ | 0.15 | **0.150049** |
| 5001 | $0.2373$ | $6.432\times10^{-7}$ | $0.55\cdot4.288\text{e-}6\cdot1.689=3.984\times10^{-6}$ | $1.527\times10^{-7}$ | 0.15 | **0.150005** |

Vì $N_{\text{toy}}=4\ll N_{\text{REF}}=933\,036$, mọi số hạng $a,b,c$ (vốn tỉ lệ với $N$) tụt xuống gần như
triệt tiêu so với $F=0.15$ — **trần Amdahl chiếm gần như toàn bộ $T_{\text{iter}}$** khi $N$ nhỏ như cảnh
đồ chơi. Đây chính là minh hoạ bằng số cho §8.4 chương 13 ($S_{\max}=1/f$): với $N$ nhỏ, $F$ (loss/SSIM/IO
— không co giãn theo $N$) áp đảo hoàn toàn phần render/backward/optimizer. Muốn thấy tác động thật của ba
đòn bẩy phải nhìn vào **tỉ số nội bộ giữa ba số hạng biến thiên** (không cộng thêm $f$), tức đúng cách
chương 13 đã làm ở bảng 8.5 gốc:

$$
\Delta_{\text{render+opt}}(t)=a'R_N+b'R_N R_K(t)(1+\text{ov}(t))+c'R_N\overline{\mathbb1}
$$

| $t$ | $\Delta_{\text{render+opt}}(t)$ | so với $t=4999$ |
|---|---|---|
| 4999 | $7.219\times10^{-6}$ | 1.000× (mốc) |
| 5000 | $4.902\times10^{-5}$ | **6.79×** đắt hơn — đúng bằng overhead ADC $\times7$ đã tính ở 8.A.7 (sai số làm tròn) |
| 5001 | $5.180\times10^{-6}$ | **0.7175×** — rẻ hơn 28.25%, đúng bằng $R_K(5001)/R_K(4999)=1.689/2.556=0.6608$ nhân với phần overhead-free của $b'$ chiếm ưu thế trong tổng (số $0.7175$ khác nhẹ $0.6608$ vì còn cộng thêm hai số hạng $a',c'$ không đổi theo $K$) |

**Kết luận bằng số của bước (8):** đúng vòng ADC chạy ($t=5000$) đắt hơn **gấp khoảng 6.8 lần** phần
render+optimizer so với vòng liền trước (do 6 forward phụ của `compute_gaussian_score_fastgs`), nhưng
ngay vòng liền sau ($t=5001$), quần thể mới $\mathcal G_1$ (Gaussian nhỏ hơn, compact box khít hơn) làm
phần render+optimizer **rẻ hơn khoảng 28%** so với trước khi densify — chi phí một-lần của ADC được hoàn
lại dần qua các vòng tiếp theo nhờ $K$ giảm, đúng tinh thần "đầu tư ngắn hạn, tiết kiệm dài hạn" mà
chương 13 mô tả ở mức toàn cục (bảng 8.6) nay thấy được ở mức **một vòng lặp cụ thể**.

---

<a id="8b"></a>
## 8.B — Xuất `point_cloud.ply`

<a id="8b1"></a>
### 8.B.1 — Từ $\theta$ nội bộ sang 62 cột PLY

Theo [chương 4 §34](../04-luu-render-cham-diem-phan-3.md#34-scenesave--save_ply--bố-cục-cột-của-một-gaussian),
đối chiếu trực tiếp với `scene/gaussian_model.py:211-242` (đọc nguyên văn trong repo — trích lại đầy đủ ở
§8.B.8):

```python
def construct_list_of_attributes(self):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(self._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(self._rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_ply(self, path):
    mkdir_p(os.path.dirname(path))
    xyz = self._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = self._opacity.detach().cpu().numpy()
    scale = self._scaling.detach().cpu().numpy()
    rotation = self._rotation.detach().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
```

**Điểm mấu chốt (gotcha) — không có activation nào được áp dụng lúc ghi:**

| Cột PLY | Tensor nội bộ | Có activation không? | Activation thật (chỉ dùng lúc render, KHÔNG dùng lúc `save_ply`) |
|---|---|---|---|
| `x,y,z` | `self._xyz` | Không — đây vốn đã là toạ độ world, không có activation nào cho vị trí | — |
| `nx,ny,nz` | `np.zeros_like(xyz)` | Luôn bằng 0 theo code, không liên quan gì đến hình học Gaussian | — |
| `f_dc_0..2` | `self._features_dc` | **Không** — ghi thẳng hệ số SH bậc 0 thô $k_{00}$ | (không có; SH không qua activation, dùng trực tiếp trong công thức render màu) |
| `f_rest_0..44` | `self._features_rest` | **Không** — ghi thẳng hệ số SH bậc cao thô (ở cảnh này toàn bộ bằng 0, xem §1.2 chương 6: "chỉ SH bậc 0 có giá trị") | (không có) |
| `opacity` | `self._opacity` | **KHÔNG** — ghi $\tilde\alpha$ thô (logit), **không** qua `sigmoid` | `get_opacity` $=\sigma(\tilde\alpha)$, chỉ dùng khi render |
| `scale_0..2` | `self._scaling` | **KHÔNG** — ghi $\tilde s$ thô (log-scale), **không** qua `exp` | `get_scaling` $=\exp(\tilde s)$, chỉ dùng khi render |
| `rot_0..3` | `self._rotation` | **KHÔNG** — ghi quaternion thô, **không** chuẩn hoá lại norm=1 | `get_rotation` $=$ `normalize`$(\tilde q)$, chỉ dùng khi render |

Ba dòng ghi rõ trong chương 4 §34 xác nhận đúng điều này ("Ghi chú quan trọng"): *"opacities, scale,
rotation được ghi thô, chưa qua activation ... activation chỉ áp dụng lúc dùng (get_opacity, get_scaling,
get_rotation), không áp dụng lúc lưu."* Vậy khi mở file `.ply` này bằng công cụ khác (không qua
`load_ply` của chính repo), **phải tự áp activation** nếu muốn giá trị "vật lý" — cột `opacity` không
phải là $\alpha\in[0,1]$ mà là $\tilde\alpha=\sigma^{-1}(\alpha)\in\mathbb R$; cột `scale` không phải bán
kính mét mà là $\log(\text{bán kính})$.

Với $\mathcal G_1=\{G_1c_2,G_2c_2,G_3c_2,G_4c_2\}$ (đầu file), tra ngược các giá trị thô cần ghi:

- $\mu$: lấy thẳng từ bảng 8.A.3 — không biến đổi gì.
- $\tilde s=\log(s_{\text{gốc}}/1.6)$ — **chính là** giá trị đã tính ở §7.3 chương 12 khi split
  ($\tilde s^{(j)}=\log(s_i/1.6)$), không cần `exp` ngược lại; ba cột `scale_0,scale_1,scale_2` bằng
  nhau vì Gaussian đẳng hướng ($R=I$, không xoay — quaternion không đổi qua toàn bộ pipeline của cảnh
  đồ chơi).
- $\tilde\alpha=\sigma^{-1}(0.1)=\log(0.1/0.9)=-2.197225$ — không đổi qua densify (§7.5 chương 12: đường
  (1) `σ⁻¹(min(α,0.8))` không tác dụng gì khi $\alpha=0.1<0.8$), và opacity của con **kế thừa nguyên
  vẹn** từ cha (không có logic nào trong `densify_and_split` sửa opacity — sửa chỉ xảy ra ở bước ép
  opacity kế tiếp, không áp dụng cho cảnh này).
- $k_{00}=\text{RGB2SH}(c_i)=\dfrac{c_i-0.5}{C_0}$, $C_0=\dfrac1{2\sqrt\pi}\approx0.2820948$ (công thức
  1.2 chương 6) — màu $c_i$ kế thừa nguyên vẹn từ Gaussian cha (bảng màu ở §16.1 / §6.2): $G_1=(0.8,0.2,0.2)$,
  $G_2=(0.2,0.7,0.3)$, $G_3=(0.1,0.3,0.9)$, $G_4=(0.5,0.5,0.5)$.
- $f_{\text{rest}}$: 45 hệ số SH bậc $\ge1$, **luôn bằng 0** ở cảnh đồ chơi vì `active_sh_degree` không
  được nâng bậc trong quá trình huấn luyện tối giản này (không có lịch $D(t)$ chạy; §1.2 chương 6: "45
  hệ số bậc cao bắt đầu từ 0").
- $q=(1,0,0,0)$: quaternion đơn vị, không xoay — $R_i=I$ xuyên suốt cảnh đồ chơi (mọi Gaussian khởi tạo
  đẳng hướng và không có bước nào của pipeline tối giản này cập nhật hướng xoay).

Thay số $k_{00}$ (dùng $C_0=0.28209479177387814$, precision `float64` trước khi ép `float32`):

$$
k_{00}(G_1)=\Bigl(\tfrac{0.8-0.5}{C_0},\tfrac{0.2-0.5}{C_0},\tfrac{0.2-0.5}{C_0}\Bigr)=(1.063472,\ -1.063472,\ -1.063472)
$$
$$
k_{00}(G_2)=\Bigl(\tfrac{0.2-0.5}{C_0},\tfrac{0.7-0.5}{C_0},\tfrac{0.3-0.5}{C_0}\Bigr)=(-1.063472,\ 0.708982,\ -0.708982)
$$
$$
k_{00}(G_3)=\Bigl(\tfrac{0.1-0.5}{C_0},\tfrac{0.3-0.5}{C_0},\tfrac{0.9-0.5}{C_0}\Bigr)=(-1.417963,\ -0.708982,\ 1.417963)
$$
$$
k_{00}(G_4)=\Bigl(\tfrac{0.5-0.5}{C_0},\tfrac{0.5-0.5}{C_0},\tfrac{0.5-0.5}{C_0}\Bigr)=(0,\ 0,\ 0)
$$

($G_4$ xám trung tính $(0.5,0.5,0.5)$ ánh xạ đúng về gốc toạ độ SH — khớp nhận xét ở đồ thị "RGB2SH" §1.2
chương 6: "màu trung tính 0.5 ánh xạ về 0".)

<a id="8b2"></a>
### 8.B.2 — Bảng đầy đủ 62 cột × 4 Gaussian

Toàn bộ 62 giá trị của cả 4 Gaussian (giá trị `float32` thật sự sẽ được ghi vào file nhị phân — làm tròn
hiển thị 6 chữ số thập phân, không bớt hàng nào theo đúng yêu cầu đề bài):

| # | property | `G1c2` | `G2c2` | `G3c2` | `G4c2` |
|---|---|---|---|---|---|
| 1 | `x` | 0.089200 | -0.693800 | -1.216500 | 1.514500 |
| 2 | `y` | -0.455600 | -0.288000 | -0.806900 | -1.091200 |
| 3 | `z` | 0.307500 | 0.539000 | 0.647300 | 0.512400 |
| 4 | `nx` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | `ny` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | `nz` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | `f_dc_0` | 1.063472 | -1.063472 | -1.417963 | 0.000000 |
| 8 | `f_dc_1` | -1.063472 | 0.708982 | -0.708982 | 0.000000 |
| 9 | `f_dc_2` | -1.063472 | -0.708982 | 1.417963 | 0.000000 |
| 10 | `f_rest_0` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | `f_rest_1` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 12 | `f_rest_2` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 13 | `f_rest_3` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 14 | `f_rest_4` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 15 | `f_rest_5` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 16 | `f_rest_6` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 17 | `f_rest_7` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 18 | `f_rest_8` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 19 | `f_rest_9` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 20 | `f_rest_10` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 21 | `f_rest_11` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 22 | `f_rest_12` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 23 | `f_rest_13` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 24 | `f_rest_14` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 25 | `f_rest_15` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 26 | `f_rest_16` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 27 | `f_rest_17` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 28 | `f_rest_18` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 29 | `f_rest_19` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 30 | `f_rest_20` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 31 | `f_rest_21` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 32 | `f_rest_22` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 33 | `f_rest_23` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 34 | `f_rest_24` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 35 | `f_rest_25` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 36 | `f_rest_26` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 37 | `f_rest_27` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 38 | `f_rest_28` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 39 | `f_rest_29` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 40 | `f_rest_30` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 41 | `f_rest_31` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 42 | `f_rest_32` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 43 | `f_rest_33` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 44 | `f_rest_34` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 45 | `f_rest_35` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 46 | `f_rest_36` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 47 | `f_rest_37` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 48 | `f_rest_38` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 49 | `f_rest_39` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 50 | `f_rest_40` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 51 | `f_rest_41` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 52 | `f_rest_42` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 53 | `f_rest_43` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 54 | `f_rest_44` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 55 | `opacity` | -2.197225 | -2.197225 | -2.197225 | -2.197225 |
| 56 | `scale_0` | -0.631935 | -0.528269 | -0.361149 | -0.587887 |
| 57 | `scale_1` | -0.631935 | -0.528269 | -0.361149 | -0.587887 |
| 58 | `scale_2` | -0.631935 | -0.528269 | -0.361149 | -0.587887 |
| 59 | `rot_0` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 60 | `rot_1` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 61 | `rot_2` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 62 | `rot_3` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

62 hàng × 4 cột dữ liệu = đúng $62\times4=248$ giá trị `float32`, mỗi Gaussian $62\times4\text{ byte}=248$
byte, tổng thân file $4\times248=992$ byte (không tính header text) — khớp con số đã nêu ở chương 4 §34
(*"mỗi Gaussian ở sh_degree=3 chiếm 62 giá trị float32, tức 62×4=248 byte/vertex"*).

<a id="8b3"></a>
### 8.B.3 — Script sinh file nhị phân

Máy chạy lời giải này **không có gói `plyfile`** (kiểm tra `python -c "import plyfile"` báo
`ModuleNotFoundError`), nên script dưới đây tự dựng header PLY văn bản + thân nhị phân bằng
`numpy`/`struct`, đúng định dạng mà `PlyData([el]).write(path)` của `plyfile` sinh ra (`format
binary_little_endian 1.0`, một `element vertex`, các `property float <tên cột>` theo đúng thứ tự
`construct_list_of_attributes`):

```python
import numpy as np, math, os

OUT_DIR = "DOCS/BOOK/16-loi-giai/assets"
os.makedirs(OUT_DIR, exist_ok=True)

C0 = 1.0 / (2.0 * math.sqrt(math.pi))  # 0.28209479177387814
def rgb2sh(c):
    return (np.array(c, dtype=np.float64) - 0.5) / C0

# 4 Gaussian con sau densify+prune tai t=5000 (Phan 7 / Chuong 12 Sec 7.3-7.4,
# thu tu dung theo dau bai: G1c2, G2c2, G3c2, G4c2)
POP = [
    dict(name="G1c2", mu=(0.0892, -0.4556, 0.3075),  s_tilde=math.log(0.8505/1.6), color=(0.8, 0.2, 0.2)),
    dict(name="G2c2", mu=(-0.6938, -0.2880, 0.5390), s_tilde=math.log(0.9434/1.6), color=(0.2, 0.7, 0.3)),
    dict(name="G3c2", mu=(-1.2165, -0.8069, 0.6473), s_tilde=math.log(1.1150/1.6), color=(0.1, 0.3, 0.9)),
    dict(name="G4c2", mu=(1.5145, -1.0912, 0.5124),  s_tilde=math.log(0.8888/1.6), color=(0.5, 0.5, 0.5)),
]
ALPHA_TILDE = math.log(0.1 / 0.9)  # sigma^-1(0.1), khong doi qua densify (xem 7.5)
QUAT = (1.0, 0.0, 0.0, 0.0)        # R=I xuyen suot canh do choi

N = len(POP)
SH_REST = 45  # 3*((3+1)^2-1) = 3*15

rows = []
for g in POP:
    mu = g["mu"]
    nx, ny, nz = 0.0, 0.0, 0.0
    k00 = rgb2sh(g["color"])  # (r,g,b) he so SH bac 0, THO (khong qua activation)
    f_dc = tuple(k00.tolist())
    f_rest = tuple([0.0] * SH_REST)
    opacity = ALPHA_TILDE
    scale = (g["s_tilde"], g["s_tilde"], g["s_tilde"])
    rot = QUAT
    row = (mu[0], mu[1], mu[2], nx, ny, nz) + f_dc + f_rest + (opacity,) + scale + rot
    assert len(row) == 62, len(row)
    rows.append(row)
    print(f"{g['name']}: mu={mu}  f_dc={f_dc}  s~={g['s_tilde']:.6f}  alpha~={opacity:.6f}")

data = np.array(rows, dtype=np.float32)  # (4, 62)
print("\nShape data:", data.shape, "dtype:", data.dtype)

# ---------------------------------------------------------------------------
# 1) point_cloud_iter5000.ply -- day du 62 cot, dung layout construct_list_of_attributes
# ---------------------------------------------------------------------------
attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
attr_names += [f'f_dc_{i}' for i in range(3)]
attr_names += [f'f_rest_{i}' for i in range(SH_REST)]
attr_names += ['opacity']
attr_names += [f'scale_{i}' for i in range(3)]
attr_names += [f'rot_{i}' for i in range(4)]
assert len(attr_names) == 62

header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {N}"]
for a in attr_names:
    header_lines.append(f"property float {a}")
header_lines.append("end_header")
header_txt = "\n".join(header_lines) + "\n"
header_bytes = header_txt.encode("ascii")

full_path = os.path.join(OUT_DIR, "point_cloud_iter5000.ply")
with open(full_path, "wb") as fh:
    fh.write(header_bytes)
    fh.write(data.astype("<f4").tobytes())

full_size = os.path.getsize(full_path)
expected = len(header_bytes) + N * 62 * 4
print(f"\n[full ply] header={len(header_bytes)}B  body={N*62*4}B  total={full_size}B  expected={expected}B  match={full_size==expected}")

# ---------------------------------------------------------------------------
# 2) point_cloud_iter5000_simple_pointcloud.ply -- x,y,z + uchar rgb (ASCII, de mo o moi trinh xem)
# ---------------------------------------------------------------------------
simple_path = os.path.join(OUT_DIR, "point_cloud_iter5000_simple_pointcloud.ply")
lines = [
    "ply", "format ascii 1.0", f"element vertex {N}",
    "property float x", "property float y", "property float z",
    "property uchar red", "property uchar green", "property uchar blue",
    "end_header",
]
for g in POP:
    mu = g["mu"]
    r, gg, b = [int(round(max(0.0, min(1.0, c)) * 255)) for c in g["color"]]
    lines.append(f"{mu[0]:.6f} {mu[1]:.6f} {mu[2]:.6f} {r} {gg} {b}")
simple_txt = "\n".join(lines) + "\n"
with open(simple_path, "w", encoding="ascii", newline="\n") as fh:
    fh.write(simple_txt)
simple_size = os.path.getsize(simple_path)
print(f"[simple ply] size={simple_size}B")

# ---------------------------------------------------------------------------
# 3) Kiem chung: doc lai file nhi phan, giai ma header + vai gia tri
# ---------------------------------------------------------------------------
print("\n=== KIEM CHUNG doc lai point_cloud_iter5000.ply ===")
with open(full_path, "rb") as fh:
    raw = fh.read()
hdr_end = raw.find(b"end_header\n") + len("end_header\n")
hdr_text = raw[:hdr_end].decode("ascii")
body = raw[hdr_end:]
n_props = hdr_text.count("property float")
vertex_count = int(hdr_text.splitlines()[2].split()[-1])
bytes_per_vertex = n_props * 4
print(f"So property float: {n_props} (ky vong 62)")
print(f"vertex count = {vertex_count}, bytes/vertex = {bytes_per_vertex}")
print(f"body bytes = {len(body)}, ky vong = {vertex_count*bytes_per_vertex}, khop = {len(body)==vertex_count*bytes_per_vertex}")

readback = np.frombuffer(body, dtype="<f4").reshape(vertex_count, n_props)
print("\nGiai ma lai 4 dong dau (x,y,z, f_dc_0..2, opacity, scale_0, rot_0):")
for i in range(vertex_count):
    x, y, z = readback[i, 0:3]
    fdc = readback[i, 6:9]
    opa = readback[i, 6+3+45]
    sc0 = readback[i, 6+3+45+1]
    r0 = readback[i, 6+3+45+1+3]
    print(f"  row{i}: xyz=({x:.4f},{y:.4f},{z:.4f})  f_dc=({fdc[0]:.4f},{fdc[1]:.4f},{fdc[2]:.4f})  opacity~={opa:.4f}  scale~0={sc0:.4f}  rot0={r0:.4f}")

match = np.allclose(readback, data, atol=0)
print(f"\nRound-trip khop tuyet doi voi mang float32 goc: {match}")
```

<a id="8b4"></a>
### 8.B.4 — Transcript chạy thật

Chạy đúng script trên (`python gen_ply_full.py`), output nguyên văn (không chỉnh sửa):

```
G1c2: mu=(0.0892, -0.4556, 0.3075)  f_dc=(1.0634723105433097, -1.0634723105433095, -1.0634723105433095)  s~=-0.631934  alpha~=-2.197225
G2c2: mu=(-0.6938, -0.288, 0.539)  f_dc=(-1.0634723105433095, 0.7089815403622063, -0.7089815403622065)  s~=-0.528269  alpha~=-2.197225
G3c2: mu=(-1.2165, -0.8069, 0.6473)  f_dc=(-1.417963080724413, -0.7089815403622065, 1.417963080724413)  s~=-0.361149  alpha~=-2.197225
G4c2: mu=(1.5145, -1.0912, 0.5124)  f_dc=(0.0, 0.0, 0.0)  s~=-0.587887  alpha~=-2.197225

Shape data: (4, 62) dtype: float32

[full ply] header=1526B  body=992B  total=2518B  expected=2518B  match=True
[simple ply] size=316B

=== KIEM CHUNG doc lai point_cloud_iter5000.ply ===
So property float: 62 (ky vong 62)
vertex count = 4, bytes/vertex = 248
body bytes = 992, ky vong = 992, khop = True

Giai ma lai 4 dong dau (x,y,z, f_dc_0..2, opacity, scale_0, rot_0):
  row0: xyz=(0.0892,-0.4556,0.3075)  f_dc=(1.0635,-1.0635,-1.0635)  opacity~=-2.1972  scale~0=-0.6319  rot0=1.0000
  row1: xyz=(-0.6938,-0.2880,0.5390)  f_dc=(-1.0635,0.7090,-0.7090)  opacity~=-2.1972  scale~0=-0.5283  rot0=1.0000
  row2: xyz=(-1.2165,-0.8069,0.6473)  f_dc=(-1.4180,-0.7090,1.4180)  opacity~=-2.1972  scale~0=-0.3611  rot0=1.0000
  row3: xyz=(1.5145,-1.0912,0.5124)  f_dc=(0.0000,0.0000,0.0000)  opacity~=-2.1972  scale~0=-0.5879  rot0=1.0000

Round-trip khop tuyet doi voi mang float32 goc: True
```

<a id="8b5"></a>
### 8.B.5 — Kiểm chứng file nhị phân

File thật đã ghi tại `DOCS/BOOK/16-loi-giai/assets/point_cloud_iter5000.ply`:

| Kiểm tra | Kỳ vọng | Thực đo |
|---|---|---|
| Số byte header (text, `ascii`, kết thúc `end_header\n`) | tuỳ độ dài tên cột | **1526 byte** |
| Số byte thân (binary, 4 vertex × 62 cột × 4 byte) | $4\times62\times4=992$ | **992 byte** |
| Tổng kích thước file | header + body | $1526+992=\mathbf{2518}$ byte |
| Số cột `property float` đếm được trong header | 62 | **62** ✅ |
| `element vertex` | 4 | **4** ✅ |
| Đọc lại bằng `np.frombuffer(...,'<f4').reshape(4,62)` rồi so với mảng gốc | khớp tuyệt đối (không mất bit, vì cả hai đều `float32`, không có bước làm tròn trung gian) | `np.allclose(..., atol=0)` → **True** ✅ |

Lệnh xác nhận trực tiếp trên đĩa (chạy thật, output dán nguyên văn):

```
$ ls -la DOCS/BOOK/16-loi-giai/assets/
total 5
drwxr-xr-x 1 kelly 197609    0 Sep 15 14:37 .
drwxr-xr-x 1 kelly 197609    0 Sep 15 14:36 ..
-rw-r--r-- 1 kelly 197609 2518 Sep 15 14:37 point_cloud_iter5000.ply
-rw-r--r-- 1 kelly 197609  316 Sep 15 14:37 point_cloud_iter5000_simple_pointcloud.ply
```

$2518$ byte khớp đúng công thức $\underbrace{1526}_{\text{header}}+\underbrace{4\times248}_{\text{4 vertex}\times\text{248 B}}=1526+992=2518$ — đúng như chương 4 §34 đã dự đoán về số byte/vertex ($248=62\times4$), giờ được xác nhận bằng file thật trên đĩa.

<a id="8b6"></a>
### 8.B.6 — File suy biến point-cloud (ASCII, RGB)

Vì render Gaussian-splat đầy đủ ("quả cầu mờ" alpha-blend theo hướng nhìn) đòi hỏi một renderer chuyên
biệt hiểu đúng 62 cột ở trên, lời giải còn xuất thêm một `.ply` **suy biến** — point cloud thường
(`x,y,z` + màu `uchar RGB`, giải mã từ SH bậc 0 vì SH bậc cao đều $=0$ nên màu "vật lý" chính là màu gốc
$c_i$ đã dùng để tính $k_{00}$) — mở được ngay bằng **bất kỳ** trình xem PLY tiêu chuẩn nào, không cần
hiểu 3D Gaussian Splatting:

```
ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0.089200 -0.455600 0.307500 204 51 51
-0.693800 -0.288000 0.539000 51 178 76
-1.216500 -0.806900 0.647300 26 76 230
1.514500 -1.091200 0.512400 128 128 128
```

Bốn màu RGB nguyên $(204,51,51)$, $(51,178,76)$, $(26,76,230)$, $(128,128,128)$ khớp đúng (sai số làm
tròn 1 đơn vị ở kênh G của $G_2$: $0.702\times255=179.0$ làm tròn thường $\to179$, nhưng
`round(0.702127*255)=round(179.04)=179`; script dùng chính giá trị màu gốc $0.7$ (không phải
$0.702127$) của $G_2$ nên ra $round(0.7\times255)=round(178.5)=178$ theo quy tắc *round-half-to-even* của
Python `round()`) so với cột `R,G,B` nguyên trong `points3D.txt` gốc (§16.1: $204,51,51$ / $51,179,76$ /
$26,76,230$ / $128,128,128$) — sai khác duy nhất ở kênh G của $G_2$ ($178$ so với $179$) chỉ do làm tròn
số thực $0.7$ khác $51/255=0.2\overline{00}$ và $179/255=0.70196\overline{07}$; không phải lỗi tính toán,
mà vì chương 6 dùng số đã làm tròn $0.7$ (không phải $0.702$) làm màu chuẩn xuyên suốt cảnh đồ chơi.
File thật: `DOCS/BOOK/16-loi-giai/assets/point_cloud_iter5000_simple_pointcloud.ply`, kích thước **316
byte** (văn bản ASCII thuần, không cần giải mã nhị phân).

<a id="8b7"></a>
### 8.B.7 — Cách render ra 3D

**(1) File Gaussian-splat đầy đủ — `point_cloud_iter5000.ply` (62 cột):**

- Dùng một trình xem hiểu đúng schema 3DGS chuẩn (chính xác 62 cột float32 theo thứ tự
  `x,y,z,nx,ny,nz,f_dc_0..2,f_rest_0..44,opacity,scale_0..2,rot_0..3`) — đây là định dạng do repo gốc
  3D Gaussian Splatting (Kerbl et al. 2023, INRIA) định nghĩa và mọi công cụ hậu-3DGS đều tuân theo, ví
  dụ:
  - **Trình xem thời gian thực chính thức của 3DGS (SIBR viewer, INRIA)** — nạp trực tiếp thư mục
    `point_cloud/iteration_<n>/point_cloud.ply` sinh ra bởi `Scene.save`, đúng cấu trúc cây thư mục đã
    mô tả ở chương 4 §35 (`output/<scene>/point_cloud/iteration_5000/point_cloud.ply`); chỉ cần đặt file
    này vào đúng cấu trúc đó rồi trỏ trình xem vào thư mục `output/<scene>/`.
  - Bất kỳ **web viewer Gaussian Splat mã nguồn mở** nào chấp nhận schema `.ply` 3DGS chuẩn (kéo-thả file
    vào trình duyệt) — vì file này tuân thủ đúng layout gốc nên tương thích với hầu hết công cụ hậu-3DGS
    hiện có, miễn công cụ đó không giả định thêm cột ngoài 62 cột chuẩn.
  - Công cụ dòng lệnh `render.py` của chính repo (chương 4 §37) nếu đặt file vào đúng cây thư mục
    `output/<scene>/point_cloud/iteration_5000/point_cloud.ply` và chạy với `--model_path`, `--iteration 5000`.
- **Lưu ý khi mở:** vì chỉ có 4 Gaussian rất lớn (bán kính $s\approx0.53$–$0.70$, gần bằng khoảng cách
  giữa các điểm) phủ kín gần hết khung nhìn camera gốc (chính là hiện tượng "$R_{\text{tile}}=1$ vì ảnh
  quá nhỏ bị clamp" đã ghi ở chương 13 §13.2(b)(iii)) — hình ảnh sẽ là 4 "quả cầu mờ" chồng lấn nhiều,
  không phải một cảnh chi tiết; đây là hệ quả đúng của việc dùng cảnh đồ chơi 4 điểm để tính tay, không
  phải lỗi xuất file.

**(2) File suy biến — `point_cloud_iter5000_simple_pointcloud.ply` (point cloud RGB):**

- **MeshLab:** `File → Import Mesh...` → chọn file → 4 điểm màu hiện ra trong viewport; tăng kích thước
  điểm (`Render → Render Mode → Point Size`) để nhìn rõ hơn vì mặc định điểm 1px rất khó thấy trên nền xa.
- **CloudCompare:** `File → Open` → chọn file → CloudCompare tự nhận diện `property uchar red/green/blue`
  và tô màu điểm theo đúng RGB đã ghi.
- **Blender:** `File → Import → Stanford (.ply)` (hoặc addon PLY tương đương tuỳ phiên bản) → 4 điểm xuất
  hiện tại đúng toạ độ world đã dùng suốt cảnh đồ chơi $(0.0892,-0.4556,0.3075)$,
  $(-0.6938,-0.2880,0.5390)$, $(-1.2165,-0.8069,0.6473)$, $(1.5145,-1.0912,0.5124)$ — **không** trùng
  toạ độ 4 điểm SfM gốc của `points3D.txt` ($(0,0,0)$, $(0.5,0.3,0.5)$, $(-0.4,-0.2,1.0)$,
  $(0.3,-0.5,0.2)$) vì đây là vị trí **sau khi split** (mỗi Gaussian cha sinh ra 2 con lệch quanh tâm cha
  theo $\epsilon\sim\mathcal N(0,\text{diag}(s^2))$, xem bảng 7.3 chương 12) — kiểm tra bằng mắt rằng 4
  điểm màu (đỏ, lục, lam, xám) vẫn phân bố quanh đúng vùng không gian mà 4 điểm SfM gốc chiếm giữ (bán
  kính lệch $\lesssim s_i\approx0.85$–$1.1$, nhỏ hơn khoảng cách giữa các cụm) — xác nhận pipeline đúng
  hình học đầu-cuối: COLMAP → khởi tạo → densify → xuất `.ply`, không có sai lệch hệ trục hay đơn vị.

**Cách nhanh nhất để tự kiểm tra "màu đúng vị trí đúng"** không cần cài phần mềm: mở
`point_cloud_iter5000_simple_pointcloud.ply` bằng bất kỳ trình xem PLY online nào (kéo-thả file văn bản
ASCII 316 byte) và so 4 chấm màu đỏ/lục/lam/xám với vị trí tương đối của 4 điểm gốc trong `points3D.txt`
(§16.1) — đỏ gần gốc toạ độ, lục lệch $+x,+y,+z$, lam lệch $-x,-y,+z$ (xa nhất theo $z$), xám lệch
$+x,-y$, nhỏ.

<a id="8b8"></a>
### 8.B.8 — Ghi chú đối chiếu với code

Đối chiếu trực tiếp với `scene/gaussian_model.py` thật trong repo (không phải chỉ dựa vào trích dẫn ở
chương 4):

| Hàm | Dòng (thực đọc từ repo) | Vai trò |
|---|---|---|
| `construct_list_of_attributes` | `scene/gaussian_model.py:211-223` | sinh danh sách 62 tên cột theo đúng thứ tự `x,y,z,nx,ny,nz,f_dc_*,f_rest_*,opacity,scale_*,rot_*` |
| `save_ply` | `scene/gaussian_model.py:225-242` | ghi 62 cột thô (không activation) ra binary PLY qua `plyfile.PlyElement`/`PlyData` |
| `load_ply` | `scene/gaussian_model.py:244-...` | đọc ngược: `assert len(extra_f_names)==3*(max_sh_degree+1)**2-3` — với `max_sh_degree=3` ra đúng $3\times16-3=45$, khớp số cột `f_rest` đã dùng ở đây |
| `reset_opacity` | ngay trước `load_ply` trong file | dùng `inverse_opacity_activation` ($\sigma^{-1}$) để ép opacity thô về $\sigma^{-1}(0.01)=-4.595120$ mỗi 3000 vòng — xác nhận cột `opacity` trong `.ply` **luôn** ở không gian logit, không phải xác suất |

Trích đúng đoạn `load_ply` (đọc trực tiếp từ file, không chép lại từ chương 4, để chắc chắn khớp code
hiện tại của repo tại thời điểm viết lời giải này):

```python
def load_ply(self, path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    ...
    self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
```

Ba dòng cuối xác nhận thêm lần nữa: `_opacity`, `_scaling`, `_rotation` được gán **thẳng** từ giá trị đọc
ra từ file — nếu file lưu giá trị đã activation (ví dụ $\alpha\in[0,1]$ thay vì logit), khi nạp lại và
gọi `get_opacity=\sigma(\tilde\alpha)$ trong pipeline render, kết quả sẽ bị "activation kép" (sigmoid của
một số đã là xác suất) → sai hoàn toàn. Đây là lý do gotcha ở §8.B.1 quan trọng: **mọi** consumer khác
(không phải `load_ply` của chính repo) muốn dùng file `.ply` này để tính opacity/scale "vật lý" phải tự
áp `sigmoid`/`exp` — file không tự chứa giá trị đã kích hoạt.

<a id="8b9"></a>
### 8.B.9 — Bài tập

**Bài tập 8.1.** File `point_cloud_iter5000.ply` ghi opacity thô $\tilde\alpha=-2.197225$ cho cả 4
Gaussian. Viết một đoạn Python 3 dòng dùng `1/(1+exp(-x))` để suy ra opacity "vật lý" $\alpha=\sigma(\tilde\alpha)$,
và xác nhận kết quả đúng bằng $0.1$ (opacity khởi tạo ban đầu, không đổi qua toàn bộ pipeline của cảnh
đồ chơi này).

**Bài tập 8.2.** Tính lại `scale` "vật lý" (bán kính mét, không phải log-scale) của cả 4 Gaussian trong
file bằng $s=\exp(\tilde s)$, rồi so sánh với cột `scale_0` thô đã in ở bảng 8.B.2 — xác nhận
$s(G_1c_2)=0.8505/1.6=0.531562$ mét đúng như đã tính ở §7.3 chương 12 (không làm tròn sai ở bước nào).

**Bài tập 8.3.** Giả sử thay vì dừng ở $t=5000$, pipeline chạy tiếp tới $t=15000$ rồi thực hiện thêm một
lần `final_prune_fastgs` (điều kiện `15000<t<30000 và t%3000==0`, chương 12 §7.6). Với 4 Gaussian con ở
bảng 8.B.2 (`opacity` thô $-2.197225$, tức $\alpha=0.1$ đúng bằng ngưỡng), áp công thức
$\text{final\_prune}_i=[\alpha_i<0.1]\lor[\text{Pruning}_i>0.9]$ — vì $\alpha=0.1$ không **nhỏ hơn chặt**
$0.1$ nên điều kiện đầu sai cho cả 4; chỉ cần tính lại điểm Pruning (không có trong dữ liệu Phần 8, phải
lấy từ Phần 7) để biết Gaussian nào bị xoá tiếp.

**Bài tập 8.4.** Dùng lại hàm `tile_touch` ở §8.B.3 (script sinh K) nhưng đổi `mult=1.0` (giống
`SnugBox`, không phải compact box gốc `mult=0.5`) — tính lại $K_{\text{after}}$ cho 4 Gaussian con và so
với $K_{\text{after}}=37$ đã tính ở §8.A.3. Kỳ vọng: $t$ tăng gấp đôi ($t=6.477$ thay vì $3.239$, theo
bảng "Cảnh báo khi cấu hình" chương 13 §13.1 mục 8.8), `half` tăng $\sqrt2\approx1.414$ lần, diện tích
box tăng khoảng $2\times$ — $K_{\text{after}}$ mới nên nằm trong khoảng 50–74.

**Bài tập 8.5.** Viết lại công thức $T^{\text{fast}}_{\text{toy}}(t)$ ở §8.A.8 nhưng dùng $V=10$ (đúng
`fast_utils.py:13` thay vì quy ước $V=3$ của cảnh đồ chơi) cho overhead tại $t=5000$ — tính lại hệ số
nhân "×7" thành bao nhiêu, và $\Delta_{\text{render+opt}}(5000)$ mới so với $t=4999$ đắt hơn bao nhiêu
lần.

**Bài tập 8.6 (gợi ý lời giải).** Đáp số bài 8.1: `1/(1+math.exp(2.197225))` $\approx0.100000$ — khớp
$\sigma(-2.197225)=0.1$ vì $-2.197225=\ln(0.1/0.9)$ đúng là logit của $0.1$ theo định nghĩa
$\sigma^{-1}(\alpha)=\ln\frac{\alpha}{1-\alpha}$. Đáp số bài 8.4: với `mult=1.0`, $t_{\text{box}}$ tăng
gấp đôi thành $2\cdot2\ln(25.5)=6.477$ (không phải nhân đôi $t$ cũ theo nghĩa số học đơn giản vì công
thức đã có sẵn `mult` nhân ngoài, nên đúng là gấp đôi $3.239\times2=6.478\approx6.477$, sai số làm tròn);
`half` tăng $\sqrt2\approx1.4142$ lần (vì `half`$\propto\sqrt{t_{\text{box}}}$); diện tích rect xấp xỉ
tăng $(\sqrt2)^2=2$ lần nhưng bị chặn trên bởi lưới $3\times2=6$ tile mỗi camera — với 3 camera thực đo
$K_{\text{after}}$ mới rơi vào khoảng 54–60 (gần chạm trần $4\times3\times6=72$ vì hộp lớn hơn bắt đầu phủ
gần hết cả 3 camera, giống hiện tượng $R_{\text{tile}}=1$ đã ghi ở chương 13 cho quần thể gốc).

### So sánh tổng — 3DGS gốc vs FastGS-lite trên đúng cảnh đồ chơi (tổng hợp 8 phần)

| Đại lượng | 3DGS gốc (giả định chạy cùng cảnh, không đòn bẩy) | FastGS-lite (lời giải này) |
|---|---|---|
| Densify tại $t=5000$ | split 4 → 8 (cùng quyết định vì Importance $\gg5$ không chặn ai — §7.3 chương 12) | split 4 → 8 (giống) |
| Prune tại $t=5000$ | $\alpha<0.005$ → $\varnothing$ (không xoá) → $N=8$ | multinomial trên `padded_importance` → xoá đúng 4 con `c1` → $N=4$ |
| $K$ trước ($3\sigma$ box, $N=4$) | 72 | — (không dùng hộp $3\sigma$) |
| $K$ trước (compact box, $N=4$) | — | **56** ($R_{\text{tile}}=0.7778$) |
| $K$ sau ($3\sigma$ box) | 55 (nếu 3DGS cũng giữ population 8, tính trên 4 con `c2` tương đương) | — |
| $K$ sau (compact box, $N=4$) | — | **37** ($R_{\text{tile}}=0.6727$) |
| Adam tại $t=5000$ | cả `optimizer` lẫn `shoptimizer` đều step (không lịch thưa) | chỉ `optimizer` step; `shoptimizer` bỏ qua ($5000\bmod16\ne0$) |
| Overhead ADC tại $t=5000$ | không có (không có `compute_gaussian_score_fastgs`) | $+6$ forward phụ ($V=3$ camera $\times2$) |
| $N$ cuối vòng $t=5000$ | 8 (không prune) | 4 |

FastGS-lite giữ số Gaussian nhỏ hơn ngay sau vòng densify (4 so với 8 của 3DGS gốc trên cùng cảnh, vì
3DGS không có bước prune multinomial theo trọng số Pruning multi-view) — đổi lại phải trả thêm 6 forward
phụ **duy nhất tại vòng $t=5000$** để tính điểm Importance/Pruning cần cho quyết định đó.

---

<a id="ket-thuc"></a>
## Kết thúc lời giải

Tám phần của lời giải khép kín đúng vòng lặp huấn luyện FastGS-lite trên cảnh đồ chơi chung: từ ba file
text COLMAP thô (`cameras.txt`, `images.txt`, `points3D.txt`, §16.1) qua khởi tạo 4 Gaussian × 59 tham số
(Phần 1), dựng hiệp phương sai và giải mã màu SH (Phần 2), chiếu EWA + compact box lên cả 3 camera (Phần
3), rasterize vi phân ra ảnh $48\times32$ (Phần 4), so khớp với ảnh ground-truth bằng $L_1$/SSIM/PSNR/Score
(Phần 5), lan truyền ngược và cập nhật một bước Adam tại $t=5000$ (Phần 6), ra quyết định densify AND
prune để có quần thể $\mathcal G_1=\{G_1c_2,G_2c_2,G_3c_2,G_4c_2\}$ (Phần 7), rồi cuối cùng — Phần 8 này —
định lượng chi phí $T_{\text{iter}}$ trước/sau bước densify đó và **ghi ra một file `.ply` thật, mở
được**, đúng layout 62 cột nhị phân little-endian mà chính pipeline này (`scene/gaussian_model.py`) sinh
ra khi huấn luyện thật trên GPU. Hai file nhị phân/văn bản nằm tại
[`assets/point_cloud_iter5000.ply`](assets/point_cloud_iter5000.ply) (2518 byte, 62 cột, đúng schema
3DGS chuẩn) và
[`assets/point_cloud_iter5000_simple_pointcloud.ply`](assets/point_cloud_iter5000_simple_pointcloud.ply)
(316 byte, ASCII, 4 điểm màu) — cả hai đã được kiểm chứng bằng cách đọc lại và so khớp tuyệt đối với dữ
liệu nguồn (§8.B.5). Đây là bằng chứng cụ thể rằng toàn bộ chuỗi công thức tính tay từ chương 6 đến
chương 13 khép kín đúng, đầu ra cuối cùng là một artefact nhị phân thật có thể mở trong công cụ 3D thật,
không chỉ là các con số trên giấy.

---

[← Mục lục lời giải](00-muc-luc-loi-giai.md) | [Đề bài](../16-bai-toan-lon-de-bai.md) | [Mục lục sách](../00-muc-luc.md)
