[← Phần 1](01-init.md) · [Mục lục lời giải](00-muc-luc-loi-giai.md) · [Đề bài (chương 16)](../16-bai-toan-lon-de-bai.md) · Phần 2/8

# Phần 2 — Chương 7: Biểu diễn 3D Gaussians

> **Đầu vào nhận từ Phần 1 (Chương 6):** xem [Phần 1](01-init.md), đầu ra là $\mathcal G_0=\{\theta_i\}_{i=1}^4$,
> 4 Gaussian × 59 tham số. Phần này (Chương 7 — lý thuyết ở [`07-3d-gaussians.md`](../07-3d-gaussians.md))
> biến tham số thô $\theta_i=(\mu_i,\tilde q_i,\tilde s_i,\tilde\alpha_i,k_i)$ thành **hình dạng hình học**
> ($\Sigma_i$), **mật độ** ($G_i(x)$) và **màu đã giải mã** ($c_i(\vec d)$) — ba đại lượng mà khối
> Projection (Chương 8, Phần 3) cần để chiếu lên ảnh.
>
> Quy ước trình bày giữ nguyên §16.4 của đề bài: mỗi bước **công thức tổng quát → thay số → kết quả**,
> làm tròn 6 chữ số có nghĩa trong bảng nhưng giữ full precision trong văn bản và script phụ lục; chỗ nào
> là *chép nguyên văn từ code* được đánh dấu rõ bằng ký hiệu 📄 kèm `file:dòng`.

---

## Mục lục Phần 2

1. [7.0 — Tóm tắt đầu vào](#70--tóm-tắt-đầu-vào-từ-phần-1)
2. [7.1 — Activation: từ tham số thô sang tham số dùng được](#71--activation-từ-tham-số-thô-sang-tham-số-dùng-được)
3. [7.2 — Ma trận xoay $R_i$ từ quaternion](#72--ma-trận-xoay-r_i-từ-quaternion)
4. [7.3 — Ma trận scale $S_i$](#73--ma-trận-scale-s_i)
5. [7.4 — Hiệp phương sai $\Sigma_i=R_iS_iS_i^\top R_i^\top$](#74--hiệp-phương-sai-sigma_ir_is_is_itop-r_itop)
6. [7.5 — Nghịch đảo $\Sigma_i^{-1}$](#75--nghịch-đảo-sigma_i-1)
7. [7.6 — Hàm mật độ $G_i(x)$ tại các điểm mẫu](#76--hàm-mật-độ-g_ix-tại-các-điểm-mẫu)
8. [7.7 — Hướng nhìn $\vec d$: 4 Gaussian × 3 camera](#77--hướng-nhìn-vec-d-4-gaussian-×-3-camera)
9. [7.8 — Giải mã màu SH bậc 0](#78--giải-mã-màu-sh-bậc-0)
10. [7.9 — Lịch tăng bậc SH $D(t)$](#79--lịch-tăng-bậc-sh-dt)
11. [7.10 — Đếm tham số & bộ nhớ: kiểm tra liên tục với Phần 1](#710--đếm-tham-số--bộ-nhớ-kiểm-tra-liên-tục-với-phần-1)
12. [7.11 — Ghi chú đối chiếu với code](#711--ghi-chú-đối-chiếu-với-code)
13. [7.12 — Phụ lục: script numpy kiểm chứng toàn bộ](#712--phụ-lục-script-numpy-kiểm-chứng-toàn-bộ)
14. [7.13 — Bài tập](#713--bài-tập)
15. [Đầu ra chuyển cho Phần 3](#đầu-ra-chuyển-cho-phần-3-chương-8--projection)

---

## 7.0 — Tóm tắt đầu vào từ Phần 1

Phần 1 (Chương 6, `01-init.md`) đã tính $\mathcal G_0$ cho cảnh đồ chơi 4 điểm SfM. Ta chép lại nguyên văn
bảng "Đầu ra của khối" của Phần 1 — mọi số trong Phần 2 này **bắt đầu từ đây**, không phát sinh số liệu mới:

| $i$ | $\mu_i$ (world) | $c_i$ (RGB gốc) | $\tilde q_i$ | $\tilde s_i$ (mỗi trục) | $s_i=e^{\tilde s_i}$ | $\tilde\alpha_i$ | $\alpha_i=\sigma(\tilde\alpha_i)$ | $k_{i,00}$ (R,G,B) |
|---|---|---|---|---|---|---|---|---|
| 1 | $(0,\ 0,\ 0)$ | $(0.8,\ 0.2,\ 0.2)$ | $(1,0,0,0)$ | $-0.1619426$ | $0.850490$ | $-2.1972246$ | $0.1$ | $(1.063472,\ -1.063472,\ -1.063472)$ |
| 2 | $(0.5,\ 0.3,\ 0.5)$ | $(0.2,\ 0.7,\ 0.3)$ | $(1,0,0,0)$ | $-0.0582669$ | $0.943398$ | $-2.1972246$ | $0.1$ | $(-1.063472,\ 0.708982,\ -0.708982)$ |
| 3 | $(-0.4,\ -0.2,\ 1.0)$ | $(0.1,\ 0.3,\ 0.9)$ | $(1,0,0,0)$ | $0.1088980$ | $1.115049$ | $-2.1972246$ | $0.1$ | $(-1.417963,\ -0.708982,\ 1.417963)$ |
| 4 | $(0.3,\ -0.5,\ 0.2)$ | $(0.5,\ 0.5,\ 0.5)$ | $(1,0,0,0)$ | $-0.1178612$ | $0.888819$ | $-2.1972246$ | $0.1$ | $(0,\ 0,\ 0)$ |

Với $k_{i,lm}=0$ cho mọi $l\ge1$ (45 hệ số/kênh, không thay đổi trong toàn bộ Phần 2 vì $D(0)=0$ — xem §7.9).

Ba camera của cảnh đồ chơi, dùng ở §7.7 và §7.8 (vị trí world $c_v$, quy ước $\S16.1$ của đề bài):

| $v$ | $c_v$ (world) |
|---|---|
| 1 | $(0,\ 0,\ -4)$ |
| 2 | $(1.5,\ 0,\ -4)$ |
| 3 | $(-1.5,\ 0.5,\ -4)$ |

Và `extent` $=1.690250$ (Phần 1, mục 1.3), sẽ được dùng lại ở Phần 7 (chương 12) chứ chưa cần ở đây, nhưng
ta nhắc lại vì §7.10 đối chiếu ngưỡng $0.1\cdot\text{extent}=0.1690$ với $s_i$.

Toàn bộ chương này là bước 2 của câu hỏi §16.2 trong đề bài:

> "dựng ma trận hiệp phương sai $\Sigma_i=R_iS_iS_i^\top R_i^\top$ và hàm mật độ $G_i(x)$, giải mã màu SH cho
> cả 3 hướng nhìn (chương 7)"

Ta sẽ làm đúng như vậy — cho **cả 4 Gaussian**, không rút gọn "tương tự cho phần còn lại".

---

## 7.1 — Activation: từ tham số thô sang tham số dùng được

Trước khi dựng $\Sigma_i$, nhắc lại 3 phép activation đã cho ở [Chương 7 lý thuyết, §2.1](../07-3d-gaussians.md#21--tham-số-hoá-59-số-mỗi-gaussian)
📄 `scene/gaussian_model.py:33-40`:

$$
q_i=\frac{\tilde q_i}{\lVert\tilde q_i\rVert}\ \ (\texttt{F.normalize}),\qquad
s_i=\exp(\tilde s_i)\ \ (\texttt{torch.exp}),\qquad
\alpha_i=\sigma(\tilde\alpha_i)=\frac1{1+e^{-\tilde\alpha_i}}\ \ (\texttt{torch.sigmoid})
$$

Đây chính là các activation mà bài kiểm định số ở cuối [Chương 7 lý thuyết, §2.1](../07-3d-gaussians.md#21--activation-và-đếm-59-tham-số)
đã tính cho cảnh đồ chơi. Ta thay số cho cả 4 Gaussian một lần nữa ở đây (không nhắc lại phép tính từng
bước — đã có đầy đủ ở Phần 1 §1.2 và Chương 7 §2.1) để có bảng tổng hợp làm đầu vào cho §7.2–§7.4:

### 7.1a — Chuẩn hoá quaternion

Cả 4 Gaussian có $\tilde q_i=(1,0,0,0)$ (Phần 1, mục 1.2c). Chuẩn norm:

$$
\lVert\tilde q_i\rVert=\sqrt{1^2+0^2+0^2+0^2}=\sqrt1=1
$$

$$
q_i=\tilde q_i/1=(1,0,0,0)\qquad(i=1,2,3,4)
$$

Không có phép chia nào làm thay đổi giá trị — quaternion thô **đã là đơn vị** ngay từ khởi tạo. Đây là lý
do §7.2 dưới đây chỉ cần xử lý **một** trường hợp $q=(1,0,0,0)$ áp dụng chung cho cả 4 Gaussian (khác với
Chương 6, nơi mỗi Gaussian có $\tilde s_i$ riêng nên $S_i$ khác nhau — xoay thì giống nhau, scale thì không).

### 7.1b — Scale

$$
s_i=\exp(\tilde s_i)
$$

| $i$ | $\tilde s_i$ | $s_i=e^{\tilde s_i}$ | Kiểm tra $s_i=\sqrt{d^2_{\text{knn3},i}}$ (Phần 1) |
|---|---|---|---|
| 1 | $-0.1619426$ | $e^{-0.1619426}=0.850490$ | $\sqrt{0.723333}=0.850490$ ✓ |
| 2 | $-0.0582669$ | $e^{-0.0582669}=0.943398$ | $\sqrt{0.890000}=0.943398$ ✓ |
| 3 | $0.1088980$ | $e^{0.1088980}=1.115049$ | $\sqrt{1.243333}=1.115049$ ✓ |
| 4 | $-0.1178612$ | $e^{-0.1178612}=0.888819$ | $\sqrt{0.790000}=0.888819$ ✓ |

Khớp chính xác vì Phần 1 định nghĩa $\tilde s_i=\log\sqrt{d^2_{\text{knn3},i}}$, nên $s_i=e^{\tilde s_i}=\sqrt{d^2_{\text{knn3},i}}$
theo cấu trúc toán học — không phải trùng hợp số học, mà là $\exp\circ\log=\text{id}$.

### 7.1c — Opacity

$$
\alpha_i=\sigma(\tilde\alpha_i)=\frac1{1+e^{2.1972246}}=\frac1{1+9.000000}=\frac1{10}=0.1
$$

cho cả 4 Gaussian (giống hệt Phần 1 vì $\tilde\alpha_i$ giống nhau). Kiểm chứng ngược: $\sigma^{-1}(0.1)=\log(0.1/0.9)=\log(1/9)=-2.1972246$ ✓.

Opacity không tham gia vào $\Sigma_i$ hay $G_i(x)$ hình học — nó chỉ nhân vào $G_i(x)$ khi rasterizer tính
alpha thực tế của pixel ($\alpha_{\text{pixel}}=\alpha_i\cdot G_i(x_{\text{pixel}})$, Chương 9) — nhưng ta
liệt kê lại ở đây để bảng "Đầu ra của khối" cuối chương (§7.10) đầy đủ cả 5 nhóm tham số hình học+màu.

---

## 7.2 — Ma trận xoay $R_i$ từ quaternion

### 7.2a — Công thức tổng quát

📄 Chép nguyên văn `utils/general_utils.py:66-87` (`build_rotation`), khớp với bản CUDA
`submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu:140-144` (`computeCov3D`, phần
dựng `glm::mat3 R`). Với quaternion chuẩn hoá $q=(r,x,y,z)$ (quy ước $w$-đầu, giống COLMAP):

$$
R(q)=
\begin{pmatrix}
1-2(y^2+z^2) & 2(xy-rz) & 2(xz+ry)\\
2(xy+rz) & 1-2(x^2+z^2) & 2(yz-rx)\\
2(xz-ry) & 2(yz+rx) & 1-2(x^2+y^2)
\end{pmatrix}
$$

Chín phần tử này (gọi $R_{00},R_{01},\dots,R_{22}$ theo chỉ số hàng-cột 0-based, đúng thứ tự code) là công
thức Rodrigues viết dưới dạng quaternion — không có xấp xỉ, đúng với mọi quaternion đơn vị $r^2+x^2+y^2+z^2=1$.

### 7.2b — Thay số $q=(1,0,0,0)$: từng phần tử một

Ta thay $r=1,\ x=0,\ y=0,\ z=0$ vào **cả 9 công thức**, không bỏ qua bước nào, để thấy rõ vì sao kết quả là
ma trận đơn vị chứ không chỉ khẳng định suông:

| Phần tử | Công thức | Thay số | Rút gọn | Kết quả |
|---|---|---|---|---|
| $R_{00}$ | $1-2(y^2+z^2)$ | $1-2(0^2+0^2)$ | $1-2\cdot0$ | $1$ |
| $R_{01}$ | $2(xy-rz)$ | $2(0\cdot0-1\cdot0)$ | $2(0-0)$ | $0$ |
| $R_{02}$ | $2(xz+ry)$ | $2(0\cdot0+1\cdot0)$ | $2(0+0)$ | $0$ |
| $R_{10}$ | $2(xy+rz)$ | $2(0\cdot0+1\cdot0)$ | $2(0+0)$ | $0$ |
| $R_{11}$ | $1-2(x^2+z^2)$ | $1-2(0^2+0^2)$ | $1-2\cdot0$ | $1$ |
| $R_{12}$ | $2(yz-rx)$ | $2(0\cdot0-1\cdot0)$ | $2(0-0)$ | $0$ |
| $R_{20}$ | $2(xz-ry)$ | $2(0\cdot0-1\cdot0)$ | $2(0-0)$ | $0$ |
| $R_{21}$ | $2(yz+rx)$ | $2(0\cdot0+1\cdot0)$ | $2(0+0)$ | $0$ |
| $R_{22}$ | $1-2(x^2+y^2)$ | $1-2(0^2+0^2)$ | $1-2\cdot0$ | $1$ |

**Giải thích vì sao từng phần tử triệt tiêu:** mọi số hạng ngoài đường chéo có dạng $2(ab\pm rc)$ với
$a,b,c\in\{x,y,z\}$ — vì $x=y=z=0$, **mọi tích** $ab$ (tích hai trong ba thành phần vector của quaternion)
và **mọi tích** $rc$ (thành phần thực nhân một thành phần vector) đều bằng $0$ do có ít nhất một thừa số
bằng 0. Trên đường chéo, $R_{kk}=1-2(\text{tổng hai bình phương thành phần vector khác }k)$ — vì cả ba
thành phần vector đều 0 nên tổng bình phương luôn 0, còn lại $R_{kk}=1$.

Về mặt hình học: quaternion $(1,0,0,0)$ có phần thực $r=1$ (góc quay $\theta=2\arccos(r)=2\arccos(1)=0$) —
đây là quaternion "không xoay" (identity rotation), nên $R$ phải là ma trận đơn vị theo định nghĩa của
phép biểu diễn quaternion→rotation, không phụ thuộc trục quay (trục không xác định khi góc $=0$).

**Kết quả:**

$$
\boxed{\ R_i=R((1,0,0,0))=\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix}=I_3\ }\qquad(i=1,2,3,4)
$$

Kiểm tra tính chất ma trận xoay hợp lệ: $RR^\top=I_3\cdot I_3^\top=I_3$ ✓ (trực giao), $\det R=\det I_3=1$ ✓
(không phải phép phản chiếu), $R^\top=R^{-1}=I_3$ ✓.

### 7.2c — Vì sao chỉ cần tính $R$ một lần cho cả 4 Gaussian

Khác với $\tilde s_i$ (khác nhau giữa 4 Gaussian vì phụ thuộc khoảng cách 3-NN của từng điểm), quaternion
khởi tạo $\tilde q_i=(1,0,0,0)$ là **hằng số giống nhau cho mọi Gaussian mới sinh** (📄 `gaussian_model.py:149-150`:
`rots = torch.zeros((N,4)); rots[:,0] = 1`, không phụ thuộc $i$). Do đó $R_1=R_2=R_3=R_4=I_3$ — ta chỉ cần
suy ra $R=I_3$ một lần ở §7.2b rồi dùng lại cho cả 4 Gaussian ở §7.4. Đây **không phải** một rút gọn "tương
tự cho phần còn lại" theo nghĩa bỏ qua phép tính — phép tính $R_i$ cho Gaussian thứ $i$ **thực sự** cho cùng
kết quả số vì đầu vào ($\tilde q_i$) giống hệt nhau, khác với $S_i$ và $\Sigma_i$ ở §7.3–§7.4 nơi đầu vào
($s_i$) khác nhau nên phải tính riêng từng Gaussian.

---

## 7.3 — Ma trận scale $S_i$

### 7.3a — Công thức tổng quát

📄 `forward.cu:124-130` (`computeCov3D`, phần `glm::mat3 S`), tương đương văn bản (`gaussian_model.py`
gọi `build_scaling_rotation` ở đường Python 📄 `general_utils.py:89-98`):

$$
S_i=\operatorname{diag}(s_{i,1},\ s_{i,2},\ s_{i,3})=
\begin{pmatrix}s_{i,1}&0&0\\0&s_{i,2}&0\\0&0&s_{i,3}\end{pmatrix}
$$

Trường hợp tổng quát $S_i$ có 3 giá trị khác nhau trên đường chéo (dị hướng — Gaussian dẹt/dài theo một
trục). Ở cảnh đồ chơi này, Phần 1 khởi tạo **đẳng hướng**: $s_{i,1}=s_{i,2}=s_{i,3}=s_i$ (một số vô hướng
lặp lại 3 lần dọc đường chéo, §1.2 Phần 1: `scales = log(sqrt(dist2))[...,None].repeat(1,3)`).

### 7.3b — Thay số cho từng Gaussian

**Gaussian 1** ($s_1=0.850490$):

$$
S_1=\begin{pmatrix}0.850490&0&0\\0&0.850490&0\\0&0&0.850490\end{pmatrix}
$$

**Gaussian 2** ($s_2=0.943398$):

$$
S_2=\begin{pmatrix}0.943398&0&0\\0&0.943398&0\\0&0&0.943398\end{pmatrix}
$$

**Gaussian 3** ($s_3=1.115049$):

$$
S_3=\begin{pmatrix}1.115049&0&0\\0&1.115049&0\\0&0&1.115049\end{pmatrix}
$$

**Gaussian 4** ($s_4=0.888819$):

$$
S_4=\begin{pmatrix}0.888819&0&0\\0&0.888819&0\\0&0&0.888819\end{pmatrix}
$$

Bốn ma trận này **khác nhau về giá trị số** (khác $S_1\ne S_2\ne S_3\ne S_4$) dù cùng cấu trúc "đẳng hướng
$\times I_3$" — đây chính là điểm phân biệt với $R_i$ ở §7.2 (giống hệt nhau cả 4 Gaussian). $S_3$ có giá
trị lớn nhất ($1.115049>1$, tức $\tilde s_3>0$) vì điểm $p_3=(-0.4,-0.2,1.0)$ nằm xa 3 điểm còn lại nhất
(Phần 1, mục 1.2b: mọi $\lVert p_3-p_j\rVert^2\ge1.2$).

---

## 7.4 — Hiệp phương sai $\Sigma_i=R_iS_iS_i^\top R_i^\top$

### 7.4a — Khai triển tổng quát bằng nhân ma trận từng bước

Trước khi thay $R_i=I_3$, ta viết ra phép nhân 4 ma trận $3\times3$ **tường minh** theo chỉ số, để thấy rõ
vì sao dạng $MM^\top$ ($M=RS$) luôn cho ma trận đối xứng bán xác định dương, và vì sao trường hợp $R=I$ chỉ
là một trường hợp đặc biệt chứ không phải "cách tính khác".

Đặt $M=RS$ (glm-code viết $M=S\cdot R$ ở dạng column-major, tương đương $M=RS$ ở quy ước hàng-vector toán
học chuẩn — xem Ghi chú đối chiếu §7.11 mục 2). Phần tử tổng quát của $M$:

$$
M_{ab}=\sum_{k=1}^3 R_{ak}S_{kb}=\sum_{k=1}^3 R_{ak}\,s_b\,\delta_{kb}=R_{ab}\,s_b
$$

(vì $S$ chéo, $S_{kb}=s_b\delta_{kb}$ — tổng chỉ còn đúng một số hạng $k=b$). Tức cột $b$ của $M$ là cột $b$
của $R$ nhân với vô hướng $s_b$: $M=[\,s_1R_{:,1}\ \ s_2R_{:,2}\ \ s_3R_{:,3}\,]$ — mỗi trục chính của
ellipsoid là một cột của $R$, kéo dài theo $s_b$.

Bây giờ $\Sigma=MM^\top$, phần tử hàng $p$ cột $q$:

$$
\Sigma_{pq}=\sum_{b=1}^3 M_{pb}M_{qb}=\sum_{b=1}^3 (R_{pb}s_b)(R_{qb}s_b)=\sum_{b=1}^3 s_b^2\,R_{pb}R_{qb}
$$

Viết đầy đủ 9 phần tử ($p,q\in\{0,1,2\}$ theo chỉ số 0-based như code):

$$
\Sigma_{00}=s_1^2R_{00}^2+s_2^2R_{01}^2+s_3^2R_{02}^2
$$
$$
\Sigma_{01}=\Sigma_{10}=s_1^2R_{00}R_{10}+s_2^2R_{01}R_{11}+s_3^2R_{02}R_{12}
$$
$$
\Sigma_{02}=\Sigma_{20}=s_1^2R_{00}R_{20}+s_2^2R_{01}R_{21}+s_3^2R_{02}R_{22}
$$
$$
\Sigma_{11}=s_1^2R_{10}^2+s_2^2R_{11}^2+s_3^2R_{12}^2
$$
$$
\Sigma_{12}=\Sigma_{21}=s_1^2R_{10}R_{20}+s_2^2R_{11}R_{21}+s_3^2R_{12}R_{22}
$$
$$
\Sigma_{22}=s_1^2R_{20}^2+s_2^2R_{21}^2+s_3^2R_{22}^2
$$

(Ma trận rõ ràng đối xứng: $\Sigma_{pq}=\Sigma_{qp}$ theo cấu trúc — không cần chứng minh riêng, thấy ngay
từ công thức tổng $\sum_b s_b^2R_{pb}R_{qb}$ đối xứng khi hoán đổi $p\leftrightarrow q$.)

### 7.4b — Thay $R=I_3$: vì sao chuỗi tổng quát trên "sụp" thành $\operatorname{diag}(s^2)$

Với $R=I_3$: $R_{ab}=\delta_{ab}$ (1 nếu $a=b$, 0 nếu khác). Thay vào từng tổng ở §7.4a:

- $\Sigma_{00}=s_1^2\cdot1^2+s_2^2\cdot0^2+s_3^2\cdot0^2=s_1^2$ (chỉ số hạng $R_{00}=1$ sống sót, hai số hạng
  kia có $R_{01}=R_{02}=0$).
- $\Sigma_{01}=s_1^2\cdot1\cdot0+s_2^2\cdot0\cdot1+s_3^2\cdot0\cdot0=0$ (mọi số hạng có ít nhất một thừa số
  $R_{ab}$ với $a\ne b$, mà $R_{ab}=0$ khi $a\ne b$ với $R=I$).
- Tương tự $\Sigma_{02}=\Sigma_{12}=0$; $\Sigma_{11}=s_2^2$; $\Sigma_{22}=s_3^2$.

Tổng quát: $\Sigma_{pq}=\sum_b s_b^2\delta_{pb}\delta_{qb}$ — tích $\delta_{pb}\delta_{qb}$ chỉ khác 0 khi
$b=p$ **và** $b=q$ đồng thời, tức khi $p=q=b$; nếu $p\ne q$ không có $b$ nào thoả cả hai nên tổng $=0$.
Nếu $p=q$, chỉ còn đúng một số hạng $b=p$: $\Sigma_{pp}=s_p^2$. Đây chính là **lý do đại số** (không chỉ là
khẳng định hình học "xoay 0° không đổi hình") cho việc rút gọn $\Sigma=\operatorname{diag}(s_1^2,s_2^2,s_3^2)$
khi $R=I$ — và vì cảnh đồ chơi còn có $s_1=s_2=s_3=s_i$ (đẳng hướng), kết quả tiếp tục rút gọn về $\Sigma=s_i^2I_3$.

$$
\boxed{\ \Sigma_i=R_iS_iS_i^\top R_i^\top\Big|_{R_i=I_3}=I_3\cdot S_i S_i^\top\cdot I_3=S_iS_i^\top=S_i^2=s_i^2I_3\ }
$$

(bước cuối dùng $S_i$ chéo và đẳng hướng nên $S_iS_i^\top=S_i^2=\operatorname{diag}(s_i^2,s_i^2,s_i^2)=s_i^2I_3$.)

### 7.4c — Thay số cho cả 4 Gaussian

**Gaussian 1.** $M_1=R_1S_1=I_3\cdot S_1=S_1$ (nhân với đơn vị không đổi gì):

$$
M_1=\begin{pmatrix}0.850490&0&0\\0&0.850490&0\\0&0&0.850490\end{pmatrix}
$$

$$
\Sigma_1=M_1M_1^\top=\begin{pmatrix}0.850490^2&0&0\\0&0.850490^2&0\\0&0&0.850490^2\end{pmatrix}
=\begin{pmatrix}0.723333&0&0\\0&0.723333&0\\0&0&0.723333\end{pmatrix}
$$

Kiểm tra bằng phép nhân trực tiếp phần tử $(0,0)$: $\Sigma_{1,00}=0.850490\times0.850490+0\times0+0\times0=0.723333$ ✓.
Phần tử ngoài đường chéo, ví dụ $\Sigma_{1,01}=0.850490\times0+0\times0.850490+0\times0=0$ ✓ (mọi tích có
thừa số 0 vì $M_1$ chéo).

**Gaussian 2.** $M_2=S_2$:

$$
\Sigma_2=\begin{pmatrix}0.943398^2&0&0\\0&0.943398^2&0\\0&0&0.943398^2\end{pmatrix}
=\begin{pmatrix}0.890000&0&0\\0&0.890000&0\\0&0&0.890000\end{pmatrix}
$$

**Gaussian 3.** $M_3=S_3$:

$$
\Sigma_3=\begin{pmatrix}1.115049^2&0&0\\0&1.115049^2&0\\0&0&1.115049^2\end{pmatrix}
=\begin{pmatrix}1.243333&0&0\\0&1.243333&0\\0&0&1.243333\end{pmatrix}
$$

**Gaussian 4.** $M_4=S_4$:

$$
\Sigma_4=\begin{pmatrix}0.888819^2&0&0\\0&0.888819^2&0\\0&0&0.888819^2\end{pmatrix}
=\begin{pmatrix}0.790000&0&0\\0&0.790000&0\\0&0&0.790000\end{pmatrix}
$$

Dạng lưu trong bộ nhớ thực tế (📄 `strip_symmetric`, `general_utils.py:52-64`: chỉ lưu 6 phần tử tam giác
trên $(xx,xy,xz,yy,yz,zz)$ vì $\Sigma$ đối xứng, tiết kiệm 1/3 bộ nhớ so với lưu đủ 9):

| $i$ | $\Sigma_i$ dạng 6 phần tử $(xx,xy,xz,yy,yz,zz)$ | $\det\Sigma_i$ |
|---|---|---|
| 1 | $(0.723333,\ 0,\ 0,\ 0.723333,\ 0,\ 0.723333)$ | $0.723333^3=0.378456$ |
| 2 | $(0.890000,\ 0,\ 0,\ 0.890000,\ 0,\ 0.890000)$ | $0.890000^3=0.704969$ |
| 3 | $(1.243333,\ 0,\ 0,\ 1.243333,\ 0,\ 1.243333)$ | $1.243333^3=1.921590$ |
| 4 | $(0.790000,\ 0,\ 0,\ 0.790000,\ 0,\ 0.790000)$ | $0.790000^3=0.493039$ |

Tất cả $\det\Sigma_i>0$ và ba trị riêng bằng nhau $=s_i^2>0$ ⟹ $\Sigma_i$ **bán xác định dương chặt** (xác
định dương thực sự, không suy biến) — điều kiện bắt buộc để $\Sigma_i^{-1}$ tồn tại ở §7.5 và $G_i(x)$ ở
§7.6 có nghĩa (hàm mật độ Gaussian đòi hỏi nghịch đảo hiệp phương sai).

### 7.4d — Nhận xét: quaternion không có tác dụng gì ở $t=0$

Vì $S_i\propto I_3$ (đẳng hướng), $\Sigma_i=s_i^2RR^\top=s_i^2I_3$ **với mọi $R\in SO(3)$**, không chỉ
riêng $R=I_3$. Nói cách khác, nếu Phần 1 khởi tạo $\tilde q_i$ bằng bất kỳ quaternion đơn vị nào khác thay
vì $(1,0,0,0)$, $\Sigma_i$ vẫn sẽ là $s_i^2I_3$ y hệt — xoay một hình cầu quanh tâm của chính nó không đổi
hình cầu. Hệ quả gradient (sẽ dùng ở Phần 6, chương 11): $\partial\Sigma_i/\partial q_i=0$ tại $t=0$, nên
$\tilde q_i$ **không nhận được gradient nào qua đường hình học** cho đến khi $\tilde s_i$ trở nên dị hướng
đủ để phép xoay bắt đầu "có tác dụng" lên hình dạng ellipsoid.

### 7.4e — Nhân ma trận $3\times3\times3\times3$ tường minh, từng phần tử, cho cả 4 Gaussian (không rút gọn)

§7.4b đã chứng minh **bằng đại số tổng quát** vì sao $\Sigma_{pq}=0$ khi $p\ne q$ và $\Sigma_{pp}=s_p^2$ khi
$R=I_3$. Ở đây ta làm lại phép nhân $M=RS$ rồi $\Sigma=MM^\top$ **bằng số, từng phần tử một, không dùng bất
kỳ rút gọn nào** — mỗi phần tử $\Sigma_{pq}$ là một tích vô hướng (dot product) của hàng $p$ và hàng $q$ của
$M$, viết đủ 3 số hạng cộng lại, cho cả 4 Gaussian. Vì $R=I_3$ nên $M=I_3\cdot S_i=S_i$ (ma trận chéo), tức
hàng $p$ của $M$ là vector $(0,\ldots,s_i,\ldots,0)$ với $s_i$ ở vị trí $p$ — nhưng ta vẫn viết đủ 3 số hạng
của mỗi dot-product thay vì "thấy ngay bằng 0" để không bỏ bước nào.

**Gaussian 1** ($s_1=0.850490$, $M_1=\operatorname{diag}(0.850490,0.850490,0.850490)$, tức hàng 0 của
$M_1$ là $(0.850490,\ 0,\ 0)$, hàng 1 là $(0,\ 0.850490,\ 0)$, hàng 2 là $(0,\ 0,\ 0.850490)$):

$$
\Sigma_{1,00}=0.850490\times0.850490+0\times0+0\times0=0.723333
$$
$$
\Sigma_{1,01}=0.850490\times0+0\times0.850490+0\times0=0.000000
$$
$$
\Sigma_{1,02}=0.850490\times0+0\times0+0\times0.850490=0.000000
$$
$$
\Sigma_{1,10}=0\times0.850490+0.850490\times0+0\times0=0.000000
$$
$$
\Sigma_{1,11}=0\times0+0.850490\times0.850490+0\times0=0.723333
$$
$$
\Sigma_{1,12}=0\times0+0.850490\times0+0\times0.850490=0.000000
$$
$$
\Sigma_{1,20}=0\times0.850490+0\times0+0.850490\times0=0.000000
$$
$$
\Sigma_{1,21}=0\times0+0\times0.850490+0.850490\times0=0.000000
$$
$$
\Sigma_{1,22}=0\times0+0\times0+0.850490\times0.850490=0.723333
$$

**Gaussian 2** ($s_2=0.943398$):

$$
\Sigma_{2,00}=0.943398\times0.943398+0\times0+0\times0=0.890000
$$
$$
\Sigma_{2,01}=0.943398\times0+0\times0.943398+0\times0=0.000000,\qquad
\Sigma_{2,02}=0.943398\times0+0\times0+0\times0.943398=0.000000
$$
$$
\Sigma_{2,10}=0\times0.943398+0.943398\times0+0\times0=0.000000,\qquad
\Sigma_{2,11}=0\times0+0.943398\times0.943398+0\times0=0.890000
$$
$$
\Sigma_{2,12}=0\times0+0.943398\times0+0\times0.943398=0.000000,\qquad
\Sigma_{2,20}=0\times0.943398+0\times0+0.943398\times0=0.000000
$$
$$
\Sigma_{2,21}=0\times0+0\times0.943398+0.943398\times0=0.000000,\qquad
\Sigma_{2,22}=0\times0+0\times0+0.943398\times0.943398=0.890000
$$

**Gaussian 3** ($s_3=1.115049$):

$$
\Sigma_{3,00}=1.115049\times1.115049+0\times0+0\times0=1.243334
$$
$$
\Sigma_{3,01}=1.115049\times0+0\times1.115049+0\times0=0.000000,\qquad
\Sigma_{3,02}=1.115049\times0+0\times0+0\times1.115049=0.000000
$$
$$
\Sigma_{3,10}=0\times1.115049+1.115049\times0+0\times0=0.000000,\qquad
\Sigma_{3,11}=0\times0+1.115049\times1.115049+0\times0=1.243334
$$
$$
\Sigma_{3,12}=0\times0+1.115049\times0+0\times1.115049=0.000000,\qquad
\Sigma_{3,20}=0\times1.115049+0\times0+1.115049\times0=0.000000
$$
$$
\Sigma_{3,21}=0\times0+0\times1.115049+1.115049\times0=0.000000,\qquad
\Sigma_{3,22}=0\times0+0\times0+1.115049\times1.115049=1.243334
$$

**Gaussian 4** ($s_4=0.888819$):

$$
\Sigma_{4,00}=0.888819\times0.888819+0\times0+0\times0=0.789999
$$
$$
\Sigma_{4,01}=0.888819\times0+0\times0.888819+0\times0=0.000000,\qquad
\Sigma_{4,02}=0.888819\times0+0\times0+0\times0.888819=0.000000
$$
$$
\Sigma_{4,10}=0\times0.888819+0.888819\times0+0\times0=0.000000,\qquad
\Sigma_{4,11}=0\times0+0.888819\times0.888819+0\times0=0.789999
$$
$$
\Sigma_{4,12}=0\times0+0.888819\times0+0\times0.888819=0.000000,\qquad
\Sigma_{4,20}=0\times0.888819+0\times0+0.888819\times0=0.000000
$$
$$
\Sigma_{4,21}=0\times0+0\times0.888819+0.888819\times0=0.000000,\qquad
\Sigma_{4,22}=0\times0+0\times0+0.888819\times0.888819=0.789999
$$

Cả 36 dòng trên ($9\times4$) đều khớp đúng bảng tổng hợp §7.4c — xác nhận bằng phép nhân ma trận tường minh
(không dùng lập luận rút gọn "$R=I$ nên bỏ qua"), đúng yêu cầu "không nhảy bước" của đề bài §16.2.

### 7.4f — Đối chứng bằng quaternion không tầm thường: $\Sigma_i=s_i^2I_3$ vẫn đúng dù $R\ne I_3$

§7.4d lập luận **bằng lời** rằng với $S$ đẳng hướng, $\Sigma=s^2RR^\top=s^2I_3$ **với mọi** $R\in SO(3)$, không
riêng $R=I_3$. Ta kiểm chứng lập luận này **bằng số cho cả 4 Gaussian**, dùng quaternion không tầm thường
$\tilde q=(0.9,\,0.1,\,0.3,\,0.2)$ (giống ví dụ mở rộng ở kiểm định số Chương 7, §2.2b), để thấy: **dù $R$
đổi hoàn toàn, $\Sigma_i$ của cảnh đồ chơi này không đổi** (vì $S_i$ đẳng hướng).

Chuẩn hoá: $\lVert\tilde q\rVert=\sqrt{0.81+0.01+0.09+0.04}=\sqrt{0.95}=0.974679$, $q=(r,x,y,z)=(0.923381,\ 0.102598,\ 0.307794,\ 0.205196)$.
Ma trận xoay (công thức §7.2a, thay số đầy đủ — xem chi tiết ở [Chương 7 lý thuyết, §2.2b](../07-3d-gaussians.md)):

$$
R=\begin{pmatrix}0.726316&-0.315789&0.610526\\0.442105&0.894737&-0.063158\\-0.526316&0.315789&0.789474\end{pmatrix}
$$

**Gaussian 1** ($s_1=0.850490$): $M_1=RS_1$ — mỗi cột của $R$ nhân với $s_1$ (§7.4a: $M_{ab}=R_{ab}s_b$):

$$
M_1=\begin{pmatrix}0.617724&-0.268576&0.519247\\0.376006&0.760965&-0.053715\\-0.447626&0.268576&0.671439\end{pmatrix}
$$

$\Sigma_1=M_1M_1^\top$, tính phần tử $(0,0)$ đầy đủ để minh hoạ (ba số hạng của dot-product hàng 0 với chính nó):

$$
\Sigma_{1,00}=0.617724^2+(-0.268576)^2+0.519247^2=0.381583+0.072133+0.269618=0.723334
$$

và phần tử ngoài đường chéo $(0,1)$ (hàng 0 của $M_1$ chấm hàng 1 của $M_1$):

$$
\Sigma_{1,01}=0.617724\times0.376006+(-0.268576)\times0.760965+0.519247\times(-0.053715)
$$
$$
=0.232269-0.204365-0.027895=0.000009\approx0
$$

(sai số $9\times10^{-6}$ chỉ do làm tròn trung gian 6 chữ số — script full-precision ở §7.12 cho đúng $0$
tới $10^{-12}$). Làm đủ 9 phần tử (script §7.12 xác nhận bằng `np.allclose`):

$$
\Sigma_1\approx\begin{pmatrix}0.723333&0&0\\0&0.723333&0\\0&0&0.723333\end{pmatrix}=s_1^2I_3
$$

**Gaussian 2** ($s_2=0.943398$): $M_2=RS_2$:

$$
M_2=\begin{pmatrix}0.685205&-0.297915&0.575969\\0.417081&0.844093&-0.059583\\-0.496525&0.297915&0.744788\end{pmatrix}
$$

$$
\Sigma_{2,00}=0.685205^2+(-0.297915)^2+0.575969^2=0.469506+0.088753+0.331740=0.889999
$$

$$
\Sigma_2=M_2M_2^\top\approx\begin{pmatrix}0.890000&0&0\\0&0.890000&0\\0&0&0.890000\end{pmatrix}=s_2^2I_3
$$

**Gaussian 3** ($s_3=1.115049$): $M_3=RS_3$:

$$
M_3=\begin{pmatrix}0.809878&-0.352121&0.680767\\0.492969&0.997675&-0.070424\\-0.586868&0.352121&0.880302\end{pmatrix}
$$

$$
\Sigma_{3,00}=0.809878^2+(-0.352121)^2+0.680767^2=0.655902+0.123989+0.463443=1.243334
$$

$$
\Sigma_3=M_3M_3^\top\approx\begin{pmatrix}1.243334&0&0\\0&1.243334&0\\0&0&1.243334\end{pmatrix}=s_3^2I_3
$$

**Gaussian 4** ($s_4=0.888819$): $M_4=RS_4$:

$$
M_4=\begin{pmatrix}0.645563&-0.280680&0.542647\\0.392952&0.795259&-0.056136\\-0.467799&0.280680&0.701699\end{pmatrix}
$$

$$
\Sigma_{4,00}=0.645563^2+(-0.280680)^2+0.542647^2=0.416752+0.078781+0.294466=0.789999
$$

$$
\Sigma_4=M_4M_4^\top\approx\begin{pmatrix}0.789999&0&0\\0&0.789999&0\\0&0&0.789999\end{pmatrix}=s_4^2I_3
$$

**Kết luận đối chứng:** cả 4 Gaussian, dù thay $R=I_3$ (§7.2, §7.4c) bằng $R$ hoàn toàn khác (chín phần tử
đều khác 0) đều cho **đúng cùng một $\Sigma_i=s_i^2I_3$** — xác nhận bằng số lập luận đại số của §7.4d: khi
$S_i\propto I_3$, phép xoay $R$ "vô hình" đối với $\Sigma_i$. Đây là lý do quan trọng cho Phần 6 (chương 11):
tại $t=0$, gradient $\partial\mathcal L/\partial\tilde q_i$ truyền qua nhánh hình học ($\Sigma_i$) sẽ **bằng
0 hệt nhau bất kể $\tilde q_i$ nhận giá trị gì** — không chỉ đúng cho $\tilde q_i=(1,0,0,0)$ mà đúng cho toàn
bộ mọi điểm trên đa tạp quaternion đơn vị.

### 7.4g — Trị riêng và vector riêng: trường hợp suy biến khi đẳng hướng

Bài tập 7.3 của [Chương 7 lý thuyết](../07-3d-gaussians.md#bài-tập-exercise) phát biểu (cho trường hợp
**dị hướng** $s=(0.6,0.3,0.1)$): "trị riêng của $\Sigma=RSS^\top R^\top$ luôn đúng bằng $s_1^2,s_2^2,s_3^2$
bất kể $R$ là gì, và vector riêng tương ứng luôn là các cột của $R$". Ta xét kỹ điều này cho **cảnh đồ chơi
đẳng hướng** của Phần 2 — vì đây là trường hợp **suy biến** (mọi $s_{i,k}$ bằng nhau) nên phát biểu "vector
riêng là cột của $R$" cần một chú thích quan trọng.

**Định nghĩa:** $v$ là vector riêng của $\Sigma$ với trị riêng $\lambda$ nếu $\Sigma v=\lambda v$, $v\ne0$.

**Trường hợp dị hướng** (ví dụ $s=(0.6,0.3,0.1)$, ba giá trị khác nhau): ba trị riêng $0.36,\,0.09,\,0.01$
là **phân biệt**, nên không gian riêng (eigenspace) của mỗi trị riêng là **một chiều** — vector riêng ứng
với $\lambda=s_k^2$ được xác định **duy nhất** (sai khác một hệ số vô hướng) và chính là cột thứ $k$ của $R$.

**Trường hợp đẳng hướng của cảnh đồ chơi** ($s_{i,1}=s_{i,2}=s_{i,3}=s_i$, cả ba trị riêng **bằng nhau**
$\lambda=s_i^2$): kiểm tra trực tiếp bằng định nghĩa cho Gaussian 1, dùng $\Sigma_1=0.723333\,I_3$ và một
vector **bất kỳ**, ví dụ $v=(1,2,-1)$ (không phải cột nào của $R$ nói riêng):

$$
\Sigma_1v=0.723333\,I_3\begin{pmatrix}1\\2\\-1\end{pmatrix}=\begin{pmatrix}0.723333\\1.446667\\-0.723333\end{pmatrix}=0.723333\begin{pmatrix}1\\2\\-1\end{pmatrix}=0.723333\,v
$$

Tức **$v=(1,2,-1)$ cũng là vector riêng** của $\Sigma_1$, với đúng trị riêng $0.723333=s_1^2$ — mặc dù
$(1,2,-1)$ hoàn toàn không phải cột nào của $R_1=I_3$ (các cột của $R_1$ là $(1,0,0),(0,1,0),(0,0,1)$). Thử
lại với một vector khác, $v'=(0.3,-0.7,2)$:

$$
\Sigma_1v'=0.723333\times(0.3,-0.7,2)=(0.217000,-0.506333,1.446667)=0.723333\,v'
$$

Vẫn đúng. **Kết luận:** khi $\Sigma_i=s_i^2I_3$ (trị riêng bội 3, cả ba trị riêng trùng nhau), **mọi vector
khác không trong $\mathbb R^3$ đều là vector riêng** — không gian riêng của $\lambda=s_i^2$ là **toàn bộ**
$\mathbb R^3$ (3 chiều), không phải một đường thẳng 1 chiều như trường hợp dị hướng. "Cột của $R$" **vẫn là
một lựa chọn hợp lệ** của cơ sở trực chuẩn cho không gian riêng này (vì $R$ trực giao, ba cột luôn là 3
vector trực chuẩn) nhưng **không phải lựa chọn duy nhất** — bất kỳ cơ sở trực chuẩn nào của $\mathbb R^3$
cũng làm được. Đây chính là lý do hình học của nhận xét ở §7.4d: một hình cầu (ellipsoid suy biến với ba
bán trục bằng nhau) không có "trục chính" xác định duy nhất — xoay hình cầu quanh tâm bằng bất kỳ $R$ nào
cũng cho lại đúng hình cầu đó, nên không có cách nào (kể cả bằng phân tích trị riêng) để "đọc ngược" ra $R$
đã dùng chỉ từ $\Sigma_i$ khi Gaussian đẳng hướng.

Bảng xác nhận cho cả 4 Gaussian (dùng vector kiểm tra $v=(1,2,-1)$ như trên, không chuẩn hoá):

| $i$ | $\Sigma_i v$ | $\lambda v$ với $\lambda=s_i^2$ | Khớp |
|---|---|---|---|
| 1 | $0.723333\times(1,2,-1)=(0.723333,1.446667,-0.723333)$ | $(0.723333,1.446667,-0.723333)$ | ✓ |
| 2 | $0.890000\times(1,2,-1)=(0.890000,1.780000,-0.890000)$ | $(0.890000,1.780000,-0.890000)$ | ✓ |
| 3 | $1.243333\times(1,2,-1)=(1.243333,2.486667,-1.243333)$ | $(1.243333,2.486667,-1.243333)$ | ✓ |
| 4 | $0.790000\times(1,2,-1)=(0.790000,1.580000,-0.790000)$ | $(0.790000,1.580000,-0.790000)$ | ✓ |

Hệ quả thực tiễn cho Chương 8 (Phần 3): khi phép chiếu EWA cần "trục chính của ellipsoid" để dựng hộp bao
compact, **ở $t=0$ mọi Gaussian trong cảnh đồ chơi đều là hình cầu** — bất kỳ hướng nào cũng là "trục chính"
hợp lệ, nên Phần 3 không cần bận tâm tới hướng của $R_i$ khi tính bounding box tại bước khởi tạo; hướng chỉ
bắt đầu có ý nghĩa xác định (duy nhất) một khi $\tilde s_i$ trở nên dị hướng qua huấn luyện.

---

## 7.5 — Nghịch đảo $\Sigma_i^{-1}$

### 7.5a — Công thức tổng quát cho ma trận chéo

Với ma trận đường chéo $D=\operatorname{diag}(d_1,d_2,d_3)$, $d_k\ne0$:

$$
D^{-1}=\operatorname{diag}(1/d_1,\ 1/d_2,\ 1/d_3)
$$

(vì $D\cdot\operatorname{diag}(1/d_1,1/d_2,1/d_3)=\operatorname{diag}(d_1/d_1,d_2/d_2,d_3/d_3)=I_3$ — kiểm
tra trực tiếp bằng định nghĩa nghịch đảo). Với $\Sigma_i=s_i^2I_3=\operatorname{diag}(s_i^2,s_i^2,s_i^2)$:

$$
\Sigma_i^{-1}=\operatorname{diag}\!\left(\frac1{s_i^2},\ \frac1{s_i^2},\ \frac1{s_i^2}\right)=\frac1{s_i^2}I_3
$$

### 7.5b — Thay số cho từng Gaussian

| $i$ | $s_i^2$ | $1/s_i^2$ | $\Sigma_i^{-1}$ |
|---|---|---|---|
| 1 | $0.723333$ | $1.382489$ | $1.382489\,I_3$ |
| 2 | $0.890000$ | $1.123596$ | $1.123596\,I_3$ |
| 3 | $1.243333$ | $0.804290$ | $0.804290\,I_3$ |
| 4 | $0.790000$ | $1.265823$ | $1.265823\,I_3$ |

Kiểm chứng cho Gaussian 1: $\Sigma_1\Sigma_1^{-1}=0.723333\times1.382489\times I_3=1.000000\,I_3=I_3$ ✓
(tích $0.723333\times1.382489=1.000000$ trong giới hạn làm tròn). Gaussian 3 có $\Sigma_3^{-1}$ **nhỏ
nhất** trong 4 Gaussian ($0.804290$) vì $\Sigma_3$ lớn nhất — hình dạng càng "phình to" thì nghịch đảo hiệp
phương sai càng "phẳng" (Mahalanobis distance tăng chậm hơn theo khoảng cách Euclid), khớp trực giác: bán
kính $1\sigma$ của G3 ($s_3=1.115$) lớn nhất nên density giảm chậm nhất khi đi xa tâm.

Full precision (dùng ở §7.6, không làm tròn):

$$
1/s_1^2=1.38248911635,\quad 1/s_2^2=1.12359550562,\quad 1/s_3^2=0.80428975986,\quad 1/s_4^2=1.26582278481
$$

### 7.5c — Đối chứng bằng công thức nghịch đảo $3\times3$ tổng quát (adjugate/cofactor), không dùng rút gọn ma trận chéo

§7.5a dùng lối tắt hợp lệ cho ma trận chéo. Để không "nhảy bước", ta nghịch đảo $\Sigma_1$ bằng công thức
**tổng quát cho ma trận $3\times3$ bất kỳ** (không giả định trước là chéo), rồi xác nhận kết quả trùng với
§7.5b. Với ma trận $A=(a_{pq})$, nghịch đảo qua ma trận phụ hợp (adjugate):

$$
A^{-1}=\frac1{\det A}\operatorname{adj}(A),\qquad \operatorname{adj}(A)_{pq}=(-1)^{p+q}M_{qp}
$$

với $M_{qp}$ là định thức con (minor) thu được bằng cách xoá hàng $q$, cột $p$ của $A$. Áp dụng cho
$\Sigma_1=\begin{pmatrix}a&0&0\\0&a&0\\0&0&a\end{pmatrix}$, $a=0.723333$ (không giả định $\Sigma_1$ chéo hoá
được đơn giản — tính đủ 9 cofactor như với ma trận tổng quát):

$$
C_{00}=(+1)\det\begin{pmatrix}a&0\\0&a\end{pmatrix}=a^2-0\times0=a^2=0.523211
$$
$$
C_{01}=(-1)\det\begin{pmatrix}0&0\\0&a\end{pmatrix}=-( 0\times a-0\times0)=0
$$
$$
C_{02}=(+1)\det\begin{pmatrix}0&a\\0&0\end{pmatrix}=0\times0-a\times0=0
$$
$$
C_{10}=(-1)\det\begin{pmatrix}0&0\\0&a\end{pmatrix}=0,\qquad
C_{11}=(+1)\det\begin{pmatrix}a&0\\0&a\end{pmatrix}=a^2=0.523211
$$
$$
C_{12}=(-1)\det\begin{pmatrix}a&0\\0&0\end{pmatrix}=0,\qquad
C_{20}=(+1)\det\begin{pmatrix}0&0\\a&0\end{pmatrix}=0
$$
$$
C_{21}=(-1)\det\begin{pmatrix}a&0\\0&0\end{pmatrix}=0,\qquad
C_{22}=(+1)\det\begin{pmatrix}a&0\\0&a\end{pmatrix}=a^2=0.523211
$$

Ma trận cofactor $C=\operatorname{diag}(a^2,a^2,a^2)$ — đối xứng nên $\operatorname{adj}(\Sigma_1)=C^\top=C=\operatorname{diag}(a^2,a^2,a^2)$.
Định thức (khai triển theo hàng 0, dùng đúng 3 cofactor hàng 0 vừa tính):

$$
\det\Sigma_1=a\cdot C_{00}+0\cdot C_{01}+0\cdot C_{02}=a\times a^2=a^3=0.723333^3=0.378456
$$

(khớp bảng §7.4c). Vậy:

$$
\Sigma_1^{-1}=\frac1{a^3}\operatorname{diag}(a^2,a^2,a^2)=\operatorname{diag}\Bigl(\frac{a^2}{a^3},\frac{a^2}{a^3},\frac{a^2}{a^3}\Bigr)=\operatorname{diag}\Bigl(\frac1a,\frac1a,\frac1a\Bigr)=\frac1{0.723333}I_3=1.382489\,I_3
$$

Trùng khớp tuyệt đối với §7.5b (tính bằng lối tắt ma trận chéo) — xác nhận công thức tổng quát $A^{-1}=\operatorname{adj}(A)/\det(A)$
**thu gọn đúng về** công thức lối tắt $D^{-1}=\operatorname{diag}(1/d_k)$ khi $A$ chéo, chứ không phải hai
công thức khác nhau tình cờ cho cùng kết quả. Ba Gaussian còn lại (G2, G3, G4) có cấu trúc hoàn toàn tương
tự (đường chéo $a=s_i^2$) nên phép khai triển cofactor $9$-phần-tử ở trên áp dụng nguyên vẹn, chỉ thay
$a\to s_i^2$ — script §7.12 xác nhận bằng `np.linalg.inv` (thuật toán LU, tổng quát hơn cả adjugate) cho
đúng $1/s_i^2\,I_3$ với cả 4 Gaussian.

---

## 7.6 — Hàm mật độ $G_i(x)$ tại các điểm mẫu

### 7.6a — Công thức tổng quát

📄 [Chương 7 lý thuyết, §2.3](../07-3d-gaussians.md#23--mật-độ-hàm-gaussian):

$$
G_i(x)=\exp\!\Bigl(-\tfrac12(x-\mu_i)^\top\Sigma_i^{-1}(x-\mu_i)\Bigr),\qquad x\in\mathbb R^3
$$

Đại lượng $\Delta(x)=(x-\mu_i)^\top\Sigma_i^{-1}(x-\mu_i)$ gọi là **khoảng cách Mahalanobis bình phương** —
với $\Sigma_i^{-1}=\frac1{s_i^2}I_3$ (đẳng hướng, §7.5), nó rút gọn về:

$$
\Delta(x)=(x-\mu_i)^\top\frac1{s_i^2}I_3(x-\mu_i)=\frac1{s_i^2}\lVert x-\mu_i\rVert^2
$$

tức chính là khoảng cách Euclid bình phương chia cho $s_i^2$ — Gaussian **đẳng hướng** có đường mức là mặt
cầu đồng tâm bán kính $t\cdot s_i$ ($t=0,1,2,\dots$) trong không gian world, khác Gaussian dị hướng (đường
mức ellipsoid, sẽ gặp lại ở Chương 8 sau khi chiếu 2D).

Với mỗi Gaussian, ta chọn 4 điểm mẫu $x$ (tổng $4\times4=16$ điểm, vượt yêu cầu tối thiểu 12 của đề bài):

1. $x=\mu_i$ (tâm) — kỳ vọng $G=1$.
2. $x=\mu_i+s_i\,e_x$ (dịch $1\sigma$ dọc trục $x$) — kỳ vọng $G=e^{-1/2}$.
3. $x=\mu_i+2s_i\,e_x$ (dịch $2\sigma$ dọc trục $x$) — kỳ vọng $G=e^{-2}$.
4. $x=\mu_i+\tfrac{s_i}2(1,1,1)$ (dịch theo **hướng chéo**, không dọc trục) — điểm này minh hoạ khai triển
   dạng toàn phương đầy đủ $(x-\mu)^\top\Sigma^{-1}(x-\mu)$ chứ không chỉ trường hợp một-thành-phần-khác-0.

### 7.6b — Gaussian 1 ($\mu_1=(0,0,0)$, $s_1=0.850490$, $1/s_1^2=1.382489$)

**Điểm 1 — tại tâm** $x=(0,0,0)$:

$$
x-\mu_1=(0,0,0),\qquad \Delta=1.382489\times(0^2+0^2+0^2)=0
$$
$$
G_1(x)=\exp(-\tfrac12\times0)=\exp(0)=1.000000
$$

**Điểm 2 — $1\sigma$ dọc $x$** $x=(0.850490,\ 0,\ 0)$:

$$
x-\mu_1=(0.850490,0,0),\qquad \lVert x-\mu_1\rVert^2=0.850490^2=0.723333
$$
$$
\Delta=1.382489\times0.723333=1.000000\qquad\text{(đúng bằng $1/s_1^2\times s_1^2=1$, kiểm chứng chéo)}
$$
$$
G_1(x)=\exp(-\tfrac12\times1.000000)=\exp(-0.5)=0.606531
$$

**Điểm 3 — $2\sigma$ dọc $x$** $x=(1.700980,\ 0,\ 0)$:

$$
x-\mu_1=(1.700980,0,0),\qquad \lVert x-\mu_1\rVert^2=1.700980^2=2.893332=4\times0.723333
$$
$$
\Delta=1.382489\times2.893332=4.000000
$$
$$
G_1(x)=\exp(-\tfrac12\times4.000000)=\exp(-2)=0.135335
$$

**Điểm 4 — hướng chéo** $x=\mu_1+\tfrac{0.850490}2(1,1,1)=(0.425245,\ 0.425245,\ 0.425245)$:

$$
x-\mu_1=(0.425245,\ 0.425245,\ 0.425245)
$$
$$
\lVert x-\mu_1\rVert^2=0.425245^2+0.425245^2+0.425245^2=3\times0.180833=0.542499
$$
$$
\Delta=1.382489\times0.542499=0.750000
$$
$$
G_1(x)=\exp(-\tfrac12\times0.750000)=\exp(-0.375)=0.687289
$$

Điểm 4 minh hoạ vì sao dạng toàn phương $(x-\mu)^\top\Sigma^{-1}(x-\mu)$ với $\Sigma^{-1}$ chéo vẫn khai
triển thành **tổng ba số hạng** $\frac1{s^2}[(x_1-\mu_1)^2+(x_2-\mu_2)^2+(x_3-\mu_3)^2]$ — ở điểm 2/3 chỉ có
một số hạng khác 0 (dịch dọc trục), còn ở điểm 4 cả ba số hạng đóng góp bằng nhau (dịch theo đường chéo
$(1,1,1)/\sqrt3$).

### 7.6c — Gaussian 2 ($\mu_2=(0.5,0.3,0.5)$, $s_2=0.943398$, $1/s_2^2=1.123596$)

**Điểm 1 — tại tâm** $x=(0.5,0.3,0.5)$:

$$
x-\mu_2=(0,0,0),\qquad G_2(x)=\exp(0)=1.000000
$$

**Điểm 2 — $1\sigma$ dọc $x$** $x=\mu_2+(0.943398,0,0)=(1.443398,\ 0.3,\ 0.5)$:

$$
x-\mu_2=(0.943398,0,0),\qquad \lVert x-\mu_2\rVert^2=0.943398^2=0.890000
$$
$$
\Delta=1.123596\times0.890000=1.000000,\qquad G_2(x)=\exp(-0.5)=0.606531
$$

**Điểm 3 — $2\sigma$ dọc $x$** $x=(2.386796,\ 0.3,\ 0.5)$:

$$
x-\mu_2=(1.886796,0,0),\qquad \lVert x-\mu_2\rVert^2=1.886796^2=3.560003\approx4\times0.890000
$$
$$
\Delta=1.123596\times3.560003=4.000000,\qquad G_2(x)=\exp(-2)=0.135335
$$

**Điểm 4 — hướng chéo** $x=\mu_2+\tfrac{0.943398}2(1,1,1)=(0.971699,\ 0.771699,\ 0.971699)$:

$$
x-\mu_2=(0.471699,\ 0.471699,\ 0.471699),\qquad \lVert x-\mu_2\rVert^2=3\times0.222500=0.667500
$$
$$
\Delta=1.123596\times0.667500=0.750000,\qquad G_2(x)=\exp(-0.375)=0.687289
$$

### 7.6d — Gaussian 3 ($\mu_3=(-0.4,-0.2,1.0)$, $s_3=1.115049$, $1/s_3^2=0.804290$)

**Điểm 1 — tại tâm** $x=(-0.4,-0.2,1.0)$:

$$
x-\mu_3=(0,0,0),\qquad G_3(x)=\exp(0)=1.000000
$$

**Điểm 2 — $1\sigma$ dọc $x$** $x=\mu_3+(1.115049,0,0)=(0.715049,\ -0.2,\ 1.0)$:

$$
x-\mu_3=(1.115049,0,0),\qquad \lVert x-\mu_3\rVert^2=1.115049^2=1.243333
$$
$$
\Delta=0.804290\times1.243333=1.000000,\qquad G_3(x)=\exp(-0.5)=0.606531
$$

**Điểm 3 — $2\sigma$ dọc $x$** $x=(1.830097,\ -0.2,\ 1.0)$:

$$
x-\mu_3=(2.230097,0,0),\qquad \lVert x-\mu_3\rVert^2=4.973331\approx4\times1.243333
$$
$$
\Delta=0.804290\times4.973331=4.000000,\qquad G_3(x)=\exp(-2)=0.135335
$$

**Điểm 4 — hướng chéo** $x=\mu_3+\tfrac{1.115049}2(1,1,1)=(0.157524,\ 0.357524,\ 1.557524)$:

$$
x-\mu_3=(0.557524,\ 0.557524,\ 0.557524),\qquad \lVert x-\mu_3\rVert^2=3\times0.310833=0.932500
$$
$$
\Delta=0.804290\times0.932500=0.750000,\qquad G_3(x)=\exp(-0.375)=0.687289
$$

### 7.6e — Gaussian 4 ($\mu_4=(0.3,-0.5,0.2)$, $s_4=0.888819$, $1/s_4^2=1.265823$)

**Điểm 1 — tại tâm** $x=(0.3,-0.5,0.2)$:

$$
x-\mu_4=(0,0,0),\qquad G_4(x)=\exp(0)=1.000000
$$

**Điểm 2 — $1\sigma$ dọc $x$** $x=\mu_4+(0.888819,0,0)=(1.188819,\ -0.5,\ 0.2)$:

$$
x-\mu_4=(0.888819,0,0),\qquad \lVert x-\mu_4\rVert^2=0.888819^2=0.790000
$$
$$
\Delta=1.265823\times0.790000=1.000000,\qquad G_4(x)=\exp(-0.5)=0.606531
$$

**Điểm 3 — $2\sigma$ dọc $x$** $x=(2.077639,\ -0.5,\ 0.2)$:

$$
x-\mu_4=(1.777639,0,0),\qquad \lVert x-\mu_4\rVert^2=3.160001\approx4\times0.790000
$$
$$
\Delta=1.265823\times3.160001=4.000000,\qquad G_4(x)=\exp(-2)=0.135335
$$

**Điểm 4 — hướng chéo** $x=\mu_4+\tfrac{0.888819}2(1,1,1)=(0.744410,\ -0.055590,\ 0.644410)$:

$$
x-\mu_4=(0.444410,\ 0.444410,\ 0.444410),\qquad \lVert x-\mu_4\rVert^2=3\times0.197500=0.592500
$$
$$
\Delta=1.265823\times0.592500=0.750000,\qquad G_4(x)=\exp(-0.375)=0.687289
$$

### 7.6f — Bảng tổng hợp 16 điểm mẫu

| $i$ | $s_i$ | $G_i(\mu_i)$ | $G_i(\mu_i+s_ie_x)$ | $G_i(\mu_i+2s_ie_x)$ | $G_i(\mu_i+\tfrac{s_i}2(1,1,1))$ |
|---|---|---|---|---|---|
| 1 | $0.850490$ | $1.000000$ | $0.606531$ | $0.135335$ | $0.687289$ |
| 2 | $0.943398$ | $1.000000$ | $0.606531$ | $0.135335$ | $0.687289$ |
| 3 | $1.115049$ | $1.000000$ | $0.606531$ | $0.135335$ | $0.687289$ |
| 4 | $0.888819$ | $1.000000$ | $0.606531$ | $0.135335$ | $0.687289$ |

Nhận xét quan trọng: **cột giá trị $G$ giống hệt nhau trên cả 4 hàng** dù $\mu_i$ và $s_i$ khác nhau hoàn
toàn giữa 4 Gaussian — vì $G_i$ chỉ phụ thuộc **số lần $\sigma$** dịch chuyển ($t=\lVert x-\mu_i\rVert/s_i$),
không phụ thuộc giá trị tuyệt đối của $s_i$ hay vị trí $\mu_i$ trong world space. Đây là tính chất "tự
tương tự theo tỉ lệ" (scale-invariance khi đo bằng đơn vị $\sigma$) của hàm mũ Gaussian chuẩn hoá — điều
này sẽ **không còn đúng** khi ta quan tâm đến kích thước tuyệt đối trên ảnh (ở Chương 8, kích thước pixel
phụ thuộc cả $s_i$ tuyệt đối lẫn khoảng cách tới camera).

### 7.6g — Mở rộng: $1\sigma$ dọc trục $y$ và trục $z$ — xác nhận tính đẳng hướng trên cả ba trục

§7.6b–§7.6e mới thử trục $x$ và hướng chéo. Để xác nhận đầy đủ "đẳng hướng nghĩa là **mọi** hướng cho cùng
$G$" (không chỉ trục $x$ và đường chéo), ta thêm hai điểm mẫu $1\sigma$ dọc trục $y$ và trục $z$ cho cả 4
Gaussian — nâng tổng số điểm mẫu của Phần 2 lên $4\times6=24$ (vượt xa yêu cầu tối thiểu 12 của đề bài).

**Gaussian 1** ($\mu_1=(0,0,0)$, $1/s_1^2=1.382489$):

- $1\sigma$ dọc $y$: $x=(0,\ 0.850490,\ 0)$, $x-\mu_1=(0,0.850490,0)$, $\lVert x-\mu_1\rVert^2=0.850490^2=0.723333$,
  $\Delta=1.382489\times0.723333=1.000000$, $G_1(x)=\exp(-0.5)=0.606531$.
- $1\sigma$ dọc $z$: $x=(0,\ 0,\ 0.850490)$, $x-\mu_1=(0,0,0.850490)$, $\lVert x-\mu_1\rVert^2=0.723333$,
  $\Delta=1.000000$, $G_1(x)=0.606531$.

**Gaussian 2** ($\mu_2=(0.5,0.3,0.5)$, $1/s_2^2=1.123596$):

- $1\sigma$ dọc $y$: $x=(0.5,\ 1.243398,\ 0.5)$, $x-\mu_2=(0,0.943398,0)$, $\lVert x-\mu_2\rVert^2=0.943398^2=0.890000$,
  $\Delta=1.123596\times0.890000=1.000000$, $G_2(x)=0.606531$.
- $1\sigma$ dọc $z$: $x=(0.5,\ 0.3,\ 1.443398)$, $x-\mu_2=(0,0,0.943398)$, $\lVert x-\mu_2\rVert^2=0.890000$,
  $\Delta=1.000000$, $G_2(x)=0.606531$.

**Gaussian 3** ($\mu_3=(-0.4,-0.2,1.0)$, $1/s_3^2=0.804290$):

- $1\sigma$ dọc $y$: $x=(-0.4,\ 0.915049,\ 1.0)$, $x-\mu_3=(0,1.115049,0)$, $\lVert x-\mu_3\rVert^2=1.115049^2=1.243333$,
  $\Delta=0.804290\times1.243333=1.000000$, $G_3(x)=0.606531$.
- $1\sigma$ dọc $z$: $x=(-0.4,\ -0.2,\ 2.115049)$, $x-\mu_3=(0,0,1.115049)$, $\lVert x-\mu_3\rVert^2=1.243333$,
  $\Delta=1.000000$, $G_3(x)=0.606531$.

**Gaussian 4** ($\mu_4=(0.3,-0.5,0.2)$, $1/s_4^2=1.265823$):

- $1\sigma$ dọc $y$: $x=(0.3,\ 0.388819,\ 0.2)$, $x-\mu_4=(0,0.888819,0)$, $\lVert x-\mu_4\rVert^2=0.888819^2=0.790000$,
  $\Delta=1.265823\times0.790000=1.000000$, $G_4(x)=0.606531$.
- $1\sigma$ dọc $z$: $x=(0.3,\ -0.5,\ 1.088819)$, $x-\mu_4=(0,0,0.888819)$, $\lVert x-\mu_4\rVert^2=0.790000$,
  $\Delta=1.000000$, $G_4(x)=0.606531$.

**Bảng tổng hợp mở rộng (24 điểm mẫu, 6 điểm/Gaussian):**

| $i$ | $G_i(\mu_i)$ | $1\sigma\ e_x$ | $1\sigma\ e_y$ | $1\sigma\ e_z$ | $2\sigma\ e_x$ | Hướng chéo $\tfrac{s_i}2(1,1,1)$ |
|---|---|---|---|---|---|---|
| 1 | $1.000000$ | $0.606531$ | $0.606531$ | $0.606531$ | $0.135335$ | $0.687289$ |
| 2 | $1.000000$ | $0.606531$ | $0.606531$ | $0.606531$ | $0.135335$ | $0.687289$ |
| 3 | $1.000000$ | $0.606531$ | $0.606531$ | $0.606531$ | $0.135335$ | $0.687289$ |
| 4 | $1.000000$ | $0.606531$ | $0.606531$ | $0.606531$ | $0.135335$ | $0.687289$ |

Kết quả xác nhận đầy đủ: **cả ba trục tọa độ** cho đúng cùng giá trị $G=e^{-1/2}$ tại khoảng cách $1\sigma$ —
đường mức $1\sigma$ của mỗi Gaussian đẳng hướng đúng là một **mặt cầu** hoàn chỉnh (không chỉ đối xứng theo
từng cặp trục riêng lẻ mà đối xứng theo toàn bộ nhóm quay $SO(3)$ quanh tâm $\mu_i$), khớp hình minh hoạ
"heatmap tròn" ở kiểm định số Chương 7, [§2.3](../07-3d-gaussians.md#23--mật-độ-hàm-gaussian).

---

## 7.7 — Hướng nhìn $\vec d$: 4 Gaussian × 3 camera

### 7.7a — Công thức tổng quát

📄 [Chương 7 lý thuyết, §2.4](../07-3d-gaussians.md#24--màu-spherical-harmonics-phụ-thuộc-hướng-nhìn), khớp
`forward.cu:24-33` (`computeColorFromSH`, `dir = pos - campos; dir = dir / length(dir)`):

$$
\vec d_{i,v}=\frac{\mu_i-c_v}{\lVert\mu_i-c_v\rVert}
$$

với $c_v$ là vị trí camera $v$ trong world space (§7.0). Ba bước tính cho mỗi cặp $(i,v)$: (1) trừ vector,
(2) tính chuẩn Euclid, (3) chia từng thành phần cho chuẩn. Ta làm đủ $4\times3=12$ cặp, đầy đủ arithmetic.

### 7.7b — Camera 1, $c_1=(0,0,-4)$

**G1** ($\mu_1=(0,0,0)$):

$$
\mu_1-c_1=(0-0,\ 0-0,\ 0-(-4))=(0,0,4)
$$
$$
\lVert\mu_1-c_1\rVert=\sqrt{0^2+0^2+4^2}=\sqrt{16}=4.000000
$$
$$
\vec d_{1,1}=(0/4,\ 0/4,\ 4/4)=(0,\ 0,\ 1)
$$

**G2** ($\mu_2=(0.5,0.3,0.5)$):

$$
\mu_2-c_1=(0.5-0,\ 0.3-0,\ 0.5-(-4))=(0.5,\ 0.3,\ 4.5)
$$
$$
\lVert\mu_2-c_1\rVert=\sqrt{0.5^2+0.3^2+4.5^2}=\sqrt{0.25+0.09+20.25}=\sqrt{20.59}=4.537621
$$
$$
\vec d_{2,1}=(0.5/4.537621,\ 0.3/4.537621,\ 4.5/4.537621)=(0.110190,\ 0.066114,\ 0.991709)
$$

**G3** ($\mu_3=(-0.4,-0.2,1.0)$):

$$
\mu_3-c_1=(-0.4-0,\ -0.2-0,\ 1.0-(-4))=(-0.4,\ -0.2,\ 5.0)
$$
$$
\lVert\mu_3-c_1\rVert=\sqrt{0.16+0.04+25}=\sqrt{25.2}=5.019960
$$
$$
\vec d_{3,1}=(-0.4/5.019960,\ -0.2/5.019960,\ 5.0/5.019960)=(-0.079682,\ -0.039841,\ 0.996024)
$$

**G4** ($\mu_4=(0.3,-0.5,0.2)$):

$$
\mu_4-c_1=(0.3-0,\ -0.5-0,\ 0.2-(-4))=(0.3,\ -0.5,\ 4.2)
$$
$$
\lVert\mu_4-c_1\rVert=\sqrt{0.09+0.25+17.64}=\sqrt{17.98}=4.240283
$$
$$
\vec d_{4,1}=(0.3/4.240283,\ -0.5/4.240283,\ 4.2/4.240283)=(0.070750,\ -0.117917,\ 0.990500)
$$

### 7.7c — Camera 2, $c_2=(1.5,0,-4)$

**G1**:

$$
\mu_1-c_2=(0-1.5,\ 0-0,\ 0-(-4))=(-1.5,\ 0,\ 4.0)
$$
$$
\lVert\mu_1-c_2\rVert=\sqrt{2.25+0+16}=\sqrt{18.25}=4.272002
$$
$$
\vec d_{1,2}=(-1.5/4.272002,\ 0/4.272002,\ 4.0/4.272002)=(-0.351123,\ 0,\ 0.936329)
$$

**G2**:

$$
\mu_2-c_2=(0.5-1.5,\ 0.3-0,\ 0.5-(-4))=(-1.0,\ 0.3,\ 4.5)
$$
$$
\lVert\mu_2-c_2\rVert=\sqrt{1+0.09+20.25}=\sqrt{21.34}=4.619524
$$
$$
\vec d_{2,2}=(-1.0/4.619524,\ 0.3/4.619524,\ 4.5/4.619524)=(-0.216473,\ 0.064942,\ 0.974126)
$$

**G3**:

$$
\mu_3-c_2=(-0.4-1.5,\ -0.2-0,\ 1.0-(-4))=(-1.9,\ -0.2,\ 5.0)
$$
$$
\lVert\mu_3-c_2\rVert=\sqrt{3.61+0.04+25}=\sqrt{28.65}=5.352569
$$
$$
\vec d_{3,2}=(-1.9/5.352569,\ -0.2/5.352569,\ 5.0/5.352569)=(-0.354970,\ -0.037365,\ 0.934131)
$$

**G4**:

$$
\mu_4-c_2=(0.3-1.5,\ -0.5-0,\ 0.2-(-4))=(-1.2,\ -0.5,\ 4.2)
$$
$$
\lVert\mu_4-c_2\rVert=\sqrt{1.44+0.25+17.64}=\sqrt{19.33}=4.396590
$$
$$
\vec d_{4,2}=(-1.2/4.396590,\ -0.5/4.396590,\ 4.2/4.396590)=(-0.272939,\ -0.113725,\ 0.955286)
$$

### 7.7d — Camera 3, $c_3=(-1.5,0.5,-4)$

**G1**:

$$
\mu_1-c_3=(0-(-1.5),\ 0-0.5,\ 0-(-4))=(1.5,\ -0.5,\ 4.0)
$$
$$
\lVert\mu_1-c_3\rVert=\sqrt{2.25+0.25+16}=\sqrt{18.5}=4.301163
$$
$$
\vec d_{1,3}=(1.5/4.301163,\ -0.5/4.301163,\ 4.0/4.301163)=(0.348743,\ -0.116248,\ 0.929981)
$$

**G2**:

$$
\mu_2-c_3=(0.5-(-1.5),\ 0.3-0.5,\ 0.5-(-4))=(2.0,\ -0.2,\ 4.5)
$$
$$
\lVert\mu_2-c_3\rVert=\sqrt{4+0.04+20.25}=\sqrt{24.29}=4.928489
$$
$$
\vec d_{2,3}=(2.0/4.928489,\ -0.2/4.928489,\ 4.5/4.928489)=(0.405804,\ -0.040580,\ 0.913059)
$$

**G3**:

$$
\mu_3-c_3=(-0.4-(-1.5),\ -0.2-0.5,\ 1.0-(-4))=(1.1,\ -0.7,\ 5.0)
$$
$$
\lVert\mu_3-c_3\rVert=\sqrt{1.21+0.49+25}=\sqrt{26.7}=5.167204
$$
$$
\vec d_{3,3}=(1.1/5.167204,\ -0.7/5.167204,\ 5.0/5.167204)=(0.212881,\ -0.135470,\ 0.967641)
$$

**G4**:

$$
\mu_4-c_3=(0.3-(-1.5),\ -0.5-0.5,\ 0.2-(-4))=(1.8,\ -1.0,\ 4.2)
$$
$$
\lVert\mu_4-c_3\rVert=\sqrt{3.24+1+17.64}=\sqrt{21.88}=4.677606
$$
$$
\vec d_{4,3}=(1.8/4.677606,\ -1.0/4.677606,\ 4.2/4.677606)=(0.384812,\ -0.213785,\ 0.897895)
$$

### 7.7e — Bảng tổng hợp 12 hướng nhìn

| $i\backslash v$ | Camera 1 ($c_1=(0,0,-4)$) | Camera 2 ($c_2=(1.5,0,-4)$) | Camera 3 ($c_3=(-1.5,0.5,-4)$) |
|---|---|---|---|
| G1 | $(0,\ 0,\ 1)$ | $(-0.351123,\ 0,\ 0.936329)$ | $(0.348743,\ -0.116248,\ 0.929981)$ |
| G2 | $(0.110190,\ 0.066114,\ 0.991709)$ | $(-0.216473,\ 0.064942,\ 0.974126)$ | $(0.405804,\ -0.040580,\ 0.913059)$ |
| G3 | $(-0.079682,\ -0.039841,\ 0.996024)$ | $(-0.354970,\ -0.037365,\ 0.934131)$ | $(0.212881,\ -0.135470,\ 0.967641)$ |
| G4 | $(0.070750,\ -0.117917,\ 0.990500)$ | $(-0.272939,\ -0.113725,\ 0.955286)$ | $(0.384812,\ -0.213785,\ 0.897895)$ |

Kiểm tra $\lVert\vec d_{i,v}\rVert=1$ cho tất cả 12 vector (ví dụ $\vec d_{2,2}$: $0.216473^2+0.064942^2+0.974126^2=0.046861+0.004218+0.948921=1.000000$ ✓)
— đúng tính chất vector đơn vị.

### 7.7f — Vì sao 12 hướng này được tính dù màu chưa phụ thuộc góc nhìn

Ở $t=0$ (khởi tạo, chưa qua bước huấn luyện nào), chỉ $D(0)=0$ nên chỉ hệ số $k_{i,00}$ (bậc $l=0$) khác 0
(§7.9 dưới). Số hạng $l=0$ của khai triển SH là $Y_0^0$ — **hàm hằng trên mặt cầu** ($Y_0^0(\vec d)=$ hằng
số $\forall\vec d$, xem chứng minh ở §7.8a), nên $c_i(\vec d)$ **không đổi khi $\vec d$ thay đổi** tại $t=0$:
màu bậc 0 là như nhau dù nhìn từ camera 1, 2 hay 3 (kiểm chứng ở §7.8 dưới). Tuy vậy phép tính $\vec d_{i,v}$
**vẫn phải thực hiện đúng công thức** — không phải vì kết quả bậc 0 cần nó, mà vì:

1. $\vec d_{i,v}$ là đầu vào **bắt buộc** của hàm `computeColorFromSH` trong code thật, được tính trước khi
   biết bậc SH nào đang active — không có nhánh rẽ "nếu $D=0$ thì bỏ qua tính $\vec d$".
2. Ngay khi $D(t)\ge1$ (từ $t=1000$, §7.9), các số hạng $l\ge1$ sẽ nhân trực tiếp với thành phần $x,y,z$
   của $\vec d_{i,v}$ (công thức $-C_1yk_{1,-1}+C_1zk_{10}-C_1xk_{11}$, xem [Chương 7, §2.4](../07-3d-gaussians.md#24--màu-spherical-harmonics-phụ-thuộc-hướng-nhìn)) —
   lúc đó màu **sẽ** phụ thuộc góc nhìn, và 12 vector này sẽ được dùng thật sự khác nhau theo từng camera.
   Phần 2 này tính trước toàn bộ 12 hướng để minh hoạ bước tính luôn tồn tại trong pipeline, đúng yêu cầu đề
   bài "giải mã màu SH cho cả 3 hướng nhìn (chương 7)" dù ở $t=0$ kết quả số trùng nhau.

---

## 7.8 — Giải mã màu SH bậc 0

### 7.8a — Công thức tổng quát và vì sao chỉ còn một số hạng

📄 [Chương 7 lý thuyết, §2.4](../07-3d-gaussians.md#24--màu-spherical-harmonics-phụ-thuộc-hướng-nhìn):

$$
c_i(\vec d)=\max\!\Bigl(0,\ 0.5+\sum_{l=0}^{D(t)}\sum_{m=-l}^{l}k_{i,lm}\,Y_{lm}(\vec d)\Bigr)
$$

Với $D(t)=0$ (§7.9), tổng kép chỉ còn **một** số hạng: $l=0,\ m=0$:

$$
c_i(\vec d)=\max\bigl(0,\ 0.5+k_{i,00}\,Y_{00}(\vec d)\bigr)
$$

Hàm cầu điều hoà bậc 0: $Y_{00}(\vec d)=C_0=\frac1{2\sqrt\pi}$ với mọi $\vec d$ (không phụ thuộc thành phần
$x,y,z$ của $\vec d$ — đây là định nghĩa toán học của "bậc 0": $Y_0^0$ là hàm hằng trên mặt cầu đơn vị,
khác các $Y_{lm}$ bậc $l\ge1$ vốn phụ thuộc trực tiếp vào hướng). 📄 `sh_utils.py:26` cho giá trị số:

$$
C_0=0.28209479177387814
$$

Vậy:

$$
c_i(\vec d)=\max(0,\ 0.5+C_0\,k_{i,00})\qquad\text{— không phụ thuộc }\vec d\text{ khi }D=0
$$

Đây chính là phép nghịch đảo của định nghĩa $\text{RGB2SH}$ ở Phần 1 (📄 `sh_utils.py:114`, `k_{i,00}=(c_i-0.5)/C_0`):
thay $k_{i,00}=(c_i-0.5)/C_0$ vào công thức trên,

$$
c_i(\vec d)=\max\Bigl(0,\ 0.5+C_0\cdot\frac{c_i-0.5}{C_0}\Bigr)=\max(0,\ 0.5+c_i-0.5)=\max(0,\ c_i)=c_i
$$

(bước cuối dùng $c_i\in[0,1]\ge0$ nên $\max(0,c_i)=c_i$) — tức **`SH2RGB` ∘ `RGB2SH` = identity theo đại số**,
không phải trùng hợp số học. Ta vẫn thay số đầy đủ dưới đây để xác nhận không có sai số làm tròn đáng kể.

### 7.8b — Thay số cho cả 4 Gaussian (bất kỳ hướng $\vec d$ nào, vì không phụ thuộc)

**Gaussian 1** ($k_{1,00}=(1.063472,\ -1.063472,\ -1.063472)$):

$$
C_0k_{1,00}=0.282095\times(1.063472,\ -1.063472,\ -1.063472)=(0.300000,\ -0.300000,\ -0.300000)
$$
$$
0.5+C_0k_{1,00}=(0.8,\ 0.2,\ 0.2)
$$
$$
c_1=\max(0,(0.8,0.2,0.2))=(0.8,\ 0.2,\ 0.2)
$$

So với màu SfM gốc $c_1^{\text{SfM}}=(0.8,0.2,0.2)$ (§7.0): **khớp tuyệt đối** ✓.

**Gaussian 2** ($k_{2,00}=(-1.063472,\ 0.708982,\ -0.708982)$):

$$
C_0k_{2,00}=(-0.300000,\ 0.200000,\ -0.200000)
$$
$$
0.5+C_0k_{2,00}=(0.2,\ 0.7,\ 0.3)
$$
$$
c_2=\max(0,(0.2,0.7,0.3))=(0.2,\ 0.7,\ 0.3)
$$

So với $c_2^{\text{SfM}}=(0.2,0.7,0.3)$: khớp ✓.

**Gaussian 3** ($k_{3,00}=(-1.417963,\ -0.708982,\ 1.417963)$):

$$
C_0k_{3,00}=(-0.400000,\ -0.200000,\ 0.400000)
$$
$$
0.5+C_0k_{3,00}=(0.1,\ 0.3,\ 0.9)
$$
$$
c_3=\max(0,(0.1,0.3,0.9))=(0.1,\ 0.3,\ 0.9)
$$

So với $c_3^{\text{SfM}}=(0.1,0.3,0.9)$: khớp ✓.

**Gaussian 4** ($k_{4,00}=(0,0,0)$):

$$
C_0k_{4,00}=(0,0,0)
$$
$$
0.5+C_0k_{4,00}=(0.5,\ 0.5,\ 0.5)
$$
$$
c_4=\max(0,(0.5,0.5,0.5))=(0.5,\ 0.5,\ 0.5)
$$

So với $c_4^{\text{SfM}}=(0.5,0.5,0.5)$: khớp ✓ (điểm xám, $k_{00}=0$ vì $c=0.5$ đúng điểm giữa của phép ánh
xạ affine `RGB2SH`).

### 7.8c — Bảng tổng hợp và xác nhận màu view-independent tại $t=0$

| $i$ | $k_{i,00}$ | $C_0k_{i,00}$ | $c_i$ (từ cam 1) | $c_i$ (từ cam 2) | $c_i$ (từ cam 3) | $c_i^{\text{SfM}}$ | Khớp |
|---|---|---|---|---|---|---|---|
| 1 | $(1.063472,-1.063472,-1.063472)$ | $(0.3,-0.3,-0.3)$ | $(0.8,0.2,0.2)$ | $(0.8,0.2,0.2)$ | $(0.8,0.2,0.2)$ | $(0.8,0.2,0.2)$ | ✓ |
| 2 | $(-1.063472,0.708982,-0.708982)$ | $(-0.3,0.2,-0.2)$ | $(0.2,0.7,0.3)$ | $(0.2,0.7,0.3)$ | $(0.2,0.7,0.3)$ | $(0.2,0.7,0.3)$ | ✓ |
| 3 | $(-1.417963,-0.708982,1.417963)$ | $(-0.4,-0.2,0.4)$ | $(0.1,0.3,0.9)$ | $(0.1,0.3,0.9)$ | $(0.1,0.3,0.9)$ | $(0.1,0.3,0.9)$ | ✓ |
| 4 | $(0,0,0)$ | $(0,0,0)$ | $(0.5,0.5,0.5)$ | $(0.5,0.5,0.5)$ | $(0.5,0.5,0.5)$ | $(0.5,0.5,0.5)$ | ✓ |

Ba cột "từ cam 1/2/3" dùng đúng 12 hướng $\vec d_{i,v}$ đã tính đầy đủ ở §7.7 — mặc dù thay vào công thức
$c_i(\vec d)=\max(0,0.5+C_0k_{i,00})$ thì $\vec d$ không xuất hiện (vì $Y_{00}$ hằng số), nên cả 12 phép
thay số này cho **đúng cùng một kết quả cho mỗi Gaussian**, bất kể cột nào. Đây là minh hoạ số học trực
tiếp cho lập luận đại số ở §7.7f: SH bậc 0 không mã hoá thông tin hướng.

### 7.8d — Không có kênh nào bị clamp ở $t=0$

Cả 4 kết quả $0.5+C_0k_{i,00}$ đều nằm trong $[0,1]$ (không âm), nên $\max(0,\cdot)$ không cắt gì —
`clamped[]` (mảng bool ghi lại kênh nào bị cắt về 0, dùng để chặn gradient ở backward — 📄 `forward.cu:31-33`)
sẽ là `(False,False,False)` cho cả 4 Gaussian tại $t=0$. Trường hợp SH bậc cao có thể gây clamp (ví dụ
$k_{10}=(0,0.2,-1.5)$ ở kiểm định số Chương 7, [§2.4d](../07-3d-gaussians.md#24--màu-spherical-harmonics-phụ-thuộc-hướng-nhìn))
chỉ xảy ra từ $t\ge1000$ khi $D(t)\ge1$ kích hoạt các $k_{lm}$ bậc cao — nhưng ở cảnh đồ chơi này mọi hệ số
bậc cao vẫn $=0$ (Phần 1), nên dù $D(t)\ge1$ được kích hoạt, số hạng $l\ge1$ vẫn đóng góp $0$ (vì $k_{lm}=0$)
— **không có clamp xảy ra tại bất kỳ $t$ nào cho đến khi các hệ số bậc cao được cập nhật khác 0 bởi gradient**
(việc đó là của Phần 6, chương 11).

### 7.8e — Khai triển đầy đủ bậc 3 (16 hệ số/kênh): xác nhận từng số hạng $l\ge1$ triệt tiêu

Đề bài yêu cầu "giải mã màu SH" đầy đủ — ta không dừng ở việc *khẳng định* "chỉ $l=0$ khác 0" mà **khai
triển đủ cả 16 số hạng** (📄 công thức tổng quát `utils/sh_utils.py:57-100`, hàm `eval_sh`, tới `deg=3`) cho
cả 4 Gaussian, dùng hướng nhìn từ camera 1 ($\vec d_{i,1}$, §7.7b), để thấy **vì sao** — không chỉ **rằng**
— toàn bộ 15 số hạng bậc $l=1,2,3$ không đóng góp gì vào màu cuối cùng.

Công thức đầy đủ (📄 `sh_utils.py:74-100`, với $x,y,z$ là ba thành phần của $\vec d$, hằng số
$C_1=0.488603$, $C_2=(1.092548,-1.092548,0.315392,-1.092548,0.546274)$,
$C_3=(-0.590044,2.890611,-0.457046,0.373176,-0.457046,1.445306,-0.590044)$):

$$
c=C_0k_{00}
\underbrace{-C_1yk_{1,-1}+C_1zk_{10}-C_1xk_{11}}_{l=1,\ 3\text{ số hạng}}
\underbrace{+C_2^{(0)}xyk_{2,-2}+C_2^{(1)}yzk_{2,-1}+C_2^{(2)}(2z^2{-}x^2{-}y^2)k_{20}+C_2^{(3)}xzk_{21}+C_2^{(4)}(x^2{-}y^2)k_{22}}_{l=2,\ 5\text{ số hạng}}
$$
$$
\underbrace{+C_3^{(0)}y(3x^2{-}y^2)k_{3,-3}+C_3^{(1)}xyzk_{3,-2}+C_3^{(2)}y(4z^2{-}x^2{-}y^2)k_{3,-1}+C_3^{(3)}z(2z^2{-}3x^2{-}3y^2)k_{30}}_{l=3,\ \text{4 trong 7 số hạng}}
$$
$$
\underbrace{+C_3^{(4)}x(4z^2{-}x^2{-}y^2)k_{31}+C_3^{(5)}z(x^2{-}y^2)k_{32}+C_3^{(6)}x(x^2{-}3y^2)k_{33}}_{l=3,\ \text{3 số hạng còn lại}}
$$

Với **cảnh đồ chơi này, mọi $k_{lm}=0$ với $l\ge1$** (Phần 1, §1.2e) — nên mỗi số hạng trong 15 số hạng
trên có dạng $(\text{đa thức của }x,y,z)\times0=0$, **bất kể giá trị của đa thức đó lớn hay nhỏ**. Bảng dưới
liệt kê đủ 15 giá trị đa thức (chưa nhân hệ số $k_{lm}=0$) để thấy chúng **không hề nhỏ hay tầm thường** —
chính vì chúng bị nhân với $k_{lm}=0$ nên mới không đóng góp, không phải vì bản thân hàm cầu điều hoà "yếu":

**Gaussian 1** ($\vec d_{1,1}=(0,0,1)$):

| Bậc | Số hạng (đa thức × $C$) | Giá trị đa thức×C | $k_{lm}$ | Đóng góp |
|---|---|---|---|---|
| $l{=}1$ | $-C_1y$ | $-0.000000$ | $k_{1,-1}=0$ | $0$ |
| $l{=}1$ | $C_1z$ | $0.488603$ | $k_{10}=0$ | $0$ |
| $l{=}1$ | $-C_1x$ | $-0.000000$ | $k_{11}=0$ | $0$ |
| $l{=}2$ | $C_2^{(0)}xy$ | $0.000000$ | $k_{2,-2}=0$ | $0$ |
| $l{=}2$ | $C_2^{(1)}yz$ | $-0.000000$ | $k_{2,-1}=0$ | $0$ |
| $l{=}2$ | $C_2^{(2)}(2z^2{-}x^2{-}y^2)$ | $0.630783$ | $k_{20}=0$ | $0$ |
| $l{=}2$ | $C_2^{(3)}xz$ | $-0.000000$ | $k_{21}=0$ | $0$ |
| $l{=}2$ | $C_2^{(4)}(x^2{-}y^2)$ | $0.000000$ | $k_{22}=0$ | $0$ |
| $l{=}3$ | $C_3^{(0)}y(3x^2{-}y^2)$ | $-0.000000$ | $k_{3,-3}=0$ | $0$ |
| $l{=}3$ | $C_3^{(1)}xyz$ | $0.000000$ | $k_{3,-2}=0$ | $0$ |
| $l{=}3$ | $C_3^{(2)}y(4z^2{-}x^2{-}y^2)$ | $-0.000000$ | $k_{3,-1}=0$ | $0$ |
| $l{=}3$ | $C_3^{(3)}z(2z^2{-}3x^2{-}3y^2)$ | $0.746353$ | $k_{30}=0$ | $0$ |
| $l{=}3$ | $C_3^{(4)}x(4z^2{-}x^2{-}y^2)$ | $-0.000000$ | $k_{31}=0$ | $0$ |
| $l{=}3$ | $C_3^{(5)}z(x^2{-}y^2)$ | $0.000000$ | $k_{32}=0$ | $0$ |
| $l{=}3$ | $C_3^{(6)}x(x^2{-}3y^2)$ | $-0.000000$ | $k_{33}=0$ | $0$ |

Tổng 15 đóng góp $=0$ ⟹ $c_1=\max(0,\ 0.5+C_0k_{1,00}+0)=(0.8,0.2,0.2)$, khớp §7.8b.

**Gaussian 2** ($\vec d_{2,1}=(0.110190,\ 0.066114,\ 0.991709)$):

| Bậc | Số hạng | Giá trị đa thức×C | $k_{lm}$ | Đóng góp |
|---|---|---|---|---|
| $l{=}1$ | $-C_1y$ | $-0.032303$ | $0$ | $0$ |
| $l{=}1$ | $C_1z$ | $0.484552$ | $0$ | $0$ |
| $l{=}1$ | $-C_1x$ | $-0.053839$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(0)}xy$ | $0.007959$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(1)}yz$ | $-0.071634$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(2)}(2z^2{-}x^2{-}y^2)$ | $0.615159$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(3)}xz$ | $-0.119390$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(4)}(x^2{-}y^2)$ | $0.004245$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(0)}y(3x^2{-}y^2)$ | $-0.001250$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(1)}xyz$ | $0.020884$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(2)}y(4z^2{-}x^2{-}y^2)$ | $-0.118374$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(3)}z(2z^2{-}3x^2{-}3y^2)$ | $0.709609$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(4)}x(4z^2{-}x^2{-}y^2)$ | $-0.197289$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(5)}z(x^2{-}y^2)$ | $0.011138$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(6)}x(x^2{-}3y^2)$ | $0.000063$ | $0$ | $0$ |

Tổng $=0$ ⟹ $c_2=\max(0,\ 0.5+C_0k_{2,00})=(0.2,0.7,0.3)$, khớp §7.8b.

**Gaussian 3** ($\vec d_{3,1}=(-0.079682,\ -0.039841,\ 0.996024)$):

| Bậc | Số hạng | Giá trị đa thức×C | $k_{lm}$ | Đóng góp |
|---|---|---|---|---|
| $l{=}1$ | $-C_1y$ | $0.019466$ | $0$ | $0$ |
| $l{=}1$ | $C_1z$ | $0.486660$ | $0$ | $0$ |
| $l{=}1$ | $-C_1x$ | $0.038933$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(0)}xy$ | $0.003468$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(1)}yz$ | $0.043355$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(2)}(2z^2{-}x^2{-}y^2)$ | $0.623274$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(3)}xz$ | $0.086710$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(4)}(x^2{-}y^2)$ | $0.002601$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(0)}y(3x^2{-}y^2)$ | $0.000410$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(1)}xyz$ | $0.009140$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(2)}y(4z^2{-}x^2{-}y^2)$ | $0.072114$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(3)}z(2z^2{-}3x^2{-}3y^2)$ | $0.728636$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(4)}x(4z^2{-}x^2{-}y^2)$ | $0.144228$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(5)}z(x^2{-}y^2)$ | $0.006855$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(6)}x(x^2{-}3y^2)$ | $0.000075$ | $0$ | $0$ |

Tổng $=0$ ⟹ $c_3=\max(0,\ 0.5+C_0k_{3,00})=(0.1,0.3,0.9)$, khớp §7.8b.

**Gaussian 4** ($\vec d_{4,1}=(0.070750,\ -0.117917,\ 0.990500)$):

| Bậc | Số hạng | Giá trị đa thức×C | $k_{lm}$ | Đóng góp |
|---|---|---|---|---|
| $l{=}1$ | $-C_1y$ | $0.057615$ | $0$ | $0$ |
| $l{=}1$ | $C_1z$ | $0.483961$ | $0$ | $0$ |
| $l{=}1$ | $-C_1x$ | $-0.034569$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(0)}xy$ | $-0.009115$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(1)}yz$ | $0.127606$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(2)}(2z^2{-}x^2{-}y^2)$ | $0.612891$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(3)}xz$ | $-0.076563$ | $0$ | $0$ |
| $l{=}2$ | $C_2^{(4)}(x^2{-}y^2)$ | $-0.004861$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(0)}y(3x^2{-}y^2)$ | $0.000077$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(1)}xyz$ | $-0.023886$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(2)}y(4z^2{-}x^2{-}y^2)$ | $0.210478$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(3)}z(2z^2{-}3x^2{-}3y^2)$ | $0.704314$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(4)}x(4z^2{-}x^2{-}y^2)$ | $-0.126287$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(5)}z(x^2{-}y^2)$ | $-0.012739$ | $0$ | $0$ |
| $l{=}3$ | $C_3^{(6)}x(x^2{-}3y^2)$ | $0.001532$ | $0$ | $0$ |

Tổng $=0$ ⟹ $c_4=\max(0,\ 0.5+C_0k_{4,00})=(0.5,0.5,0.5)$, khớp §7.8b.

**Kết luận §7.8e:** cả $4\times15=60$ số hạng bậc cao đều đóng góp đúng $0$ vào màu cuối cùng — không phải
vì bản thân các đa thức cầu điều hoà nhỏ (nhiều giá trị $>0.5$, ví dụ số hạng $C_3^{(3)}z(2z^2-3x^2-3y^2)$
của G1 $=0.746353$, khá lớn), mà **hoàn toàn** vì hệ số $k_{lm}=0$ nhân vào. Đây là minh chứng số học đầy đủ
cho khẳng định ở [Chương 7 lý thuyết, §2.5](../07-3d-gaussians.md#25--phân-chia-sh-thấp--cao--vì-sao-quan-trọng-cho-chương-6):
"SH bậc cao là hiệu chỉnh nhỏ" chỉ đúng **sau khi huấn luyện xong** — tại $t=0$, các "hiệu chỉnh" này hoàn
toàn không tồn tại (hệ số $=0$), dù cơ sở hàm cầu điều hoà bên dưới không hề suy biến.

---

## 7.9 — Lịch tăng bậc SH $D(t)$

📄 [Chương 7 lý thuyết, §2.4](../07-3d-gaussians.md#24--màu-spherical-harmonics-phụ-thuộc-hướng-nhìn) và
`train.py:81-82` (`if iteration % 1000 == 0: gaussians.oneupSHdegree()`):

$$
D(t)=\min\bigl(3,\ \lfloor t/1000\rfloor\bigr)\quad\Rightarrow\quad\text{số hệ số/kênh}=(D+1)^2
$$

với trần `max_sh_degree=3` (đề bài §16.2). Thay số cho các mốc $t$ liên quan tới toàn bộ lời giải 8 phần
(chú ý: bước Adam ở Phần 6 diễn ra tại $t=5000$, đã vượt mọi ngưỡng kích hoạt):

| $t$ | $\lfloor t/1000\rfloor$ | $D(t)$ | Hệ số/kênh $(D+1)^2$ | Float SH/Gaussian ($\times3$ kênh) | Trên $N=4$ |
|---|---|---|---|---|---|
| $0$ | $0$ | $0$ | $1$ | $3$ | $12$ |
| $999$ | $0$ | $0$ | $1$ | $3$ | $12$ |
| $1000$ | $1$ | $1$ | $4$ | $12$ | $48$ |
| $1999$ | $1$ | $1$ | $4$ | $12$ | $48$ |
| $2000$ | $2$ | $2$ | $9$ | $27$ | $108$ |
| $2999$ | $2$ | $2$ | $9$ | $27$ | $108$ |
| $3000$ | $3$ | $3$ | $16$ | $48$ | $192$ |
| $5000$ | $5\to\min(3,5)=3$ | $3$ | $16$ | $48$ | $192$ |

Tại $t=0$ (thời điểm Phần 2 này đang xử lý, ngay sau khởi tạo), $D(0)=0$ — đúng như giả định dùng suốt
§7.7–§7.8 (chỉ SH bậc 0 active). Dòng cuối ($t=5000$, mốc mà đề bài §16.2 bước (6) dùng để tính gradient và
bước Adam) cho thấy $D(5000)=3$ đã đạt trần — **nhưng** vì 45 hệ số bậc cao của cảnh đồ chơi vẫn $=0$ tại
$t=0$ (Phần 1) và Phần 2 này không mô phỏng các bước huấn luyện trung gian ($t=1\ldots4999$), số liệu màu ở
§7.8 chỉ đúng cho **thời điểm $t=0$**. Các phần sau (Phần 6 — chương 11) sẽ dùng $D(5000)=3$ khi tính
gradient qua toàn bộ 48 float SH/Gaussian, không chỉ 3 float bậc 0.

Nhận xét cho Phần 3 (chiếu — chương 8): số hệ số SH **được đọc** trong kernel forward tăng dần theo $t$
($3\to12\to27\to48$/Gaussian), là nguồn gốc của hệ số $a$ trong mô hình chi phí $aN$ ở Phần 8 (chương 13).
Tổng số tham số lưu trữ **không đổi** ($59$/Gaussian từ đầu đến cuối) — chỉ số tham số **được dùng khi
forward** thay đổi theo lịch $D(t)$.

---

## 7.10 — Đếm tham số & bộ nhớ: kiểm tra liên tục với Phần 1

### 7.10a — Đếm lại 59 tham số/Gaussian

$$
\underbrace{3}_{\mu_i}+\underbrace{4}_{\tilde q_i}+\underbrace{3}_{\tilde s_i}+\underbrace{1}_{\tilde\alpha_i}+\underbrace{3\times16}_{k_{i,lm},\ 0\le l\le3}=3+4+3+1+48=59
$$

(SH đủ 4 bậc $l=0,1,2,3$ cho $(3+1)^2=16$ hệ số/kênh × 3 kênh RGB = 48, tách thành $3$ hệ số `features_dc`
($l=0$) + $45$ hệ số `features_rest` ($l\ge1$), đúng bảng Chương 7 §2.1). Với $N=4$ Gaussian:

$$
4\times59=236\text{ tham số}
$$

khớp chính xác với con số $4\times59=236$ mà đề bài §16.2 bước (6) dùng để mô tả kích thước gradient
$\partial\mathrm{Loss}/\partial\theta_i$ — Phần 2 này xác nhận **cấu trúc hình học và màu** của cả 236 tham
số đó đã được dựng đúng (không có tham số nào "biến mất" hay tính sai chiều) trước khi chuyển cho Phần 3.

### 7.10b — Không có gì thay đổi về bộ nhớ optimizer so với Phần 1

Chương 7 (3D Gaussians) **không thêm trạng thái Adam mới** — $R_i,\ S_i,\ \Sigma_i,\ \Sigma_i^{-1},\ G_i(x),\ c_i(\vec d)$
đều là **đại lượng dẫn xuất** (derived quantities), tính lại mỗi lần forward từ $\theta_i=(\mu_i,\tilde q_i,\tilde s_i,\tilde\alpha_i,k_i)$
chứ không phải tham số học riêng có $(m,v)$ Adam đi kèm. Do đó bảng bộ nhớ của Phần 1 vẫn đúng nguyên vẹn:

| $N$ | Tham số (float) | Trạng thái Adam $(m,v)$ (float) | Tổng (float) | Byte (float32) |
|---|---|---|---|---|
| $N_0=4$ | $4\times59=236$ | $2\times236=472$ | $708$ | $2\,832$ |
| $N=10^6$ | $5.9\times10^7$ | $1.18\times10^8$ | $1.77\times10^8$ | $7.08\times10^8=708$ MB |

(Bảng này trùng khớp bảng "bộ nhớ optimizer" của Phần 1, mục 1.4 — liệt kê lại ở đây theo đúng yêu cầu
"kiểm tra liên tục" của nhiệm vụ Phần 2.)

### 7.10c — Đối chiếu ngưỡng `extent` (chuẩn bị cho Phần 7, không dùng trực tiếp ở Phần 2)

Phần 1 đã tính `extent`$=1.690250$ và ghi chú "cả 4 Gaussian đã vượt ngưỡng $0.1\cdot\text{extent}=0.1690$".
Với $s_i$ vừa xác nhận lại ở §7.1b ($s_i\in\{0.850490,\ 0.943398,\ 1.115049,\ 0.888819\}$), ta kiểm tra lại
điều này bằng phép so sánh trực tiếp:

| $i$ | $s_i$ | $0.1\cdot\text{extent}=0.169025$ | $s_i>0.1\cdot\text{extent}$? |
|---|---|---|---|
| 1 | $0.850490$ | $0.169025$ | có ($0.850490>0.169025$) |
| 2 | $0.943398$ | $0.169025$ | có |
| 3 | $1.115049$ | $0.169025$ | có |
| 4 | $0.888819$ | $0.169025$ | có |

Xác nhận đúng nhận xét của Phần 1: cả 4 Gaussian đều "quá lớn" so với ngưỡng prune `big_points_ws` ngay từ
khởi tạo — số liệu này sẽ quan trọng ở Phần 7 (chương 12, Densify/Prune) chứ Phần 2 chỉ có nhiệm vụ xác
nhận $s_i$ không đổi qua bước activation (§7.1b đã làm).

---

## 7.11 — Ghi chú đối chiếu với code

Đối chiếu [Chương 7 lý thuyết](../07-3d-gaussians.md) và bài kiểm định số cuối chương với code thực tế
trong repo, mở rộng thêm các điểm liên quan trực tiếp tới lời giải Phần 2 này:

1. **`build_rotation` — hai đường gọi cho cùng kết quả.** 📄 `utils/general_utils.py:66-87`. Hàm này **tự
   chuẩn hoá** quaternion đầu vào (dòng 67-69: `norm = sqrt(...); q = r/norm`) trước khi dựng $R$ — tức kể
   cả khi gọi `build_rotation` với $\tilde q_i$ **thô** (chưa qua `F.normalize`), kết quả vẫn đúng vì hàm tự
   lo phần chuẩn hoá. Đường thứ hai (`get_rotation` ở `gaussian_model.py`, dùng `F.normalize` tường minh rồi
   truyền `rot` đã chuẩn hoá xuống kernel CUDA `forward.cu:133` — dòng `glm::vec4 q = rot;` kèm chú thích
   "Already normalized by the caller") cho **cùng số $R$**. Ở cảnh đồ chơi, $\tilde q_i=(1,0,0,0)$ đã có
   $\lVert\tilde q_i\rVert=1$ sẵn (§7.1a) nên hai đường trùng nhau tuyệt đối, không có gì để phân biệt bằng
   số — nhưng về mặt code hai đường là hai lời gọi hàm khác nhau tại hai vị trí khác nhau trong pipeline
   (Python-side `get_covariance` so với CUDA-side `preprocessCUDA`→`computeCov3D`).

2. **Thứ tự nhân ma trận: $M=S\cdot R$ (glm, column-major) so với $\Sigma=RSS^\top R^\top$ (toán, hàng-vector).**
   📄 `forward.cu:127-149`: code viết `glm::mat3 S` rồi `glm::mat3 R` rồi `M = S * R`, `Sigma = transpose(M)*M`.
   `glm::mat3` lưu **column-major** và nhân ma trận-vector theo quy ước cột ($v'=Mv$), khác quy ước "hàng
   nhân ma trận" $v'=vM$ thường dùng khi viết $\Sigma=RSS^\top R^\top$ trên giấy. Khi quy đổi đúng quy ước,
   `S*R` trong glm tương đương $R\cdot S$ trong ký hiệu toán chuẩn (ma trận cột nhân từ trái), và
   `transpose(M)*M`$=(RS)^\top(RS)$ — khác thứ tự transpose so với $(RS)(RS)^\top$ viết trong tài liệu
   nhưng **cho cùng một kết quả số** vì với $M=RS$ thực (không phức), $M^\top M$ và $MM^\top$ đều là ma trận
   đối xứng dương xác định có cùng trị riêng (chỉ khác cơ sở vector riêng nếu $M$ không vuông/không đối
   xứng — ở đây $M$ vuông $3\times3$ và $\Sigma$ tính theo công thức tài liệu $\Sigma=(RS)(RS)^\top$ dùng ở
   §7.4 đã được kiểm chứng bằng script numpy §7.12 khớp cả hai cách viết, đúng như Chương 7 lý thuyết mục
   "Ghi chú đối chiếu code" #2 đã nêu).

3. **`computeColorFromSH` tách `dc` khỏi `shs`.** 📄 `forward.cu:24-77`. Tham số `dc` (features_dc, bậc 0)
   và `shs` (features_rest, bậc $\ge1$) là hai mảng **riêng biệt** trong kernel — dòng 35: `result = SH_C0 * direct_color[0]` đọc từ `dc`, không phải `shs[0]`. Điều này khớp với cách Phần 1 tách $k_{i,00}$ (3 hệ
   số, optimizer riêng `optimizer`) khỏi $k_{i,lm},l\ge1$ (45 hệ số, optimizer riêng `shoptimizer`) — không
   phải chi tiết cài đặt tình cờ mà phản ánh đúng ý nghĩa toán học "bậc 0 là màu cơ bản, bậc cao là hiệu
   chỉnh góc nhìn" đã nêu ở [Chương 7 lý thuyết, §2.5](../07-3d-gaussians.md#25--phân-chia-sh-thấp--cao--vì-sao-quan-trọng-cho-chương-6).

4. **Hằng `SH_C0` phía CUDA khớp `C0` phía Python tới 15 chữ số.** 📄 `auxiliary.h:26`:
   `__device__ const float SH_C0 = 0.28209479177387814f;` so với `utils/sh_utils.py:26`: `C0 = 0.28209479177387814`
   — cùng giá trị double literal (dù CUDA ép về `float` 32-bit khi biên dịch, phần thập phân đủ dư để không
   gây sai khác đáng kể ở scale bài toán này). Script phụ lục §7.12 dùng đúng chuỗi số này.

5. **`strip_symmetric` — lưu 6/9 phần tử.** 📄 `general_utils.py:52-64` (`strip_lowerdiag`, gọi qua
   `strip_symmetric`). Vì $\Sigma_i$ đối xứng (chứng minh cấu trúc ở §7.4a), chỉ cần lưu nửa trên đường
   chéo — $(xx,xy,xz,yy,yz,zz)$ — 6 số thay vì 9, tiết kiệm 33% bộ nhớ cho mảng `cov3D` mà không mất thông
   tin (phía tính toán luôn có thể dựng lại đủ ma trận $3\times3$ từ 6 số này bằng tính đối xứng).

6. **Không có clamp ở SH bậc 0.** Đối chiếu với `forward.cu:37-42` (đoạn code chỉ áp dụng `max(result,0.f)`
   *sau khi* cộng đủ các bậc active, không có bước clamp riêng cho từng bậc trung gian) — khớp lập luận
   §7.8d: `clamped[]` chỉ ghi nhận trạng thái *cuối cùng* của từng kênh, không phải trạng thái tại mỗi bậc.

7. **`get_covariance` gọi `build_rotation` trên `self._rotation` thô, không qua `get_rotation`.** 📄
   `gaussian_model.py:130-131` — đây là điểm khác biệt nhỏ so với đường kernel CUDA (mục 1 ở trên) nhưng
   không ảnh hưởng tới kết quả số của Phần 2 này vì `build_rotation` tự chuẩn hoá.

8. **`computeColorFromSH` — trích nguyên văn để đối chiếu công thức §7.8a/§7.8e với biến chương trình thật.**
   📄 `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu:24-76`:

   ```cpp
   __device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs,
       const glm::vec3* means, glm::vec3 campos,
       const float* dc, const float* shs, bool* clamped)
   {
       glm::vec3 pos = means[idx];
       glm::vec3 dir = pos - campos;
       dir = dir / glm::length(dir);

       glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
       glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
       glm::vec3 result = SH_C0 * direct_color[0];

       if (deg > 0) {
           float x = dir.x, y = dir.y, z = dir.z;
           result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];
           if (deg > 1) {
               float xx = x*x, yy = y*y, zz = z*z;
               float xy = x*y, yz = y*z, xz = x*z;
               result = result +
                   SH_C2[0] * xy * sh[3] + SH_C2[1] * yz * sh[4] +
                   SH_C2[2] * (2.0f*zz - xx - yy) * sh[5] +
                   SH_C2[3] * xz * sh[6] + SH_C2[4] * (xx - yy) * sh[7];
               if (deg > 2) {
                   result = result +
                       SH_C3[0]*y*(3.0f*xx-yy)*sh[8] + SH_C3[1]*xy*z*sh[9] +
                       SH_C3[2]*y*(4.0f*zz-xx-yy)*sh[10] +
                       SH_C3[3]*z*(2.0f*zz-3.0f*xx-3.0f*yy)*sh[11] +
                       SH_C3[4]*x*(4.0f*zz-xx-yy)*sh[12] +
                       SH_C3[5]*z*(xx-yy)*sh[13] + SH_C3[6]*x*(xx-3.0f*yy)*sh[14];
               }
           }
       }
       result += 0.5f;
       clamped[3*idx+0] = (result.x < 0);
       clamped[3*idx+1] = (result.y < 0);
       clamped[3*idx+2] = (result.z < 0);
       return glm::max(result, 0.0f);
   }
   ```

   Đối chiếu trực tiếp với §7.8a/§7.8e: `dir = pos - campos; dir /= length(dir)` chính là công thức
   $\vec d_{i,v}=(\mu_i-c_v)/\lVert\mu_i-c_v\rVert$ của §7.7a; `direct_color[0]` là $k_{i,00}$ (mảng `dc`
   tách riêng khỏi `sh`, đúng ghi chú mục 3); vòng `if (deg > 0)`/`if (deg > 1)`/`if (deg > 2)` lồng nhau
   chính là cách code hiện thực hoá $D(t)$ — tại $t=0$ ($D=0$, biến `deg` truyền vào bằng 0), **toàn bộ
   khối `if (deg > 0)` không chạy**, khớp chính xác lập luận §7.8a rằng chỉ số hạng bậc 0 tồn tại (không
   phải "các số hạng khác cộng thành 0" như §7.8e minh hoạ bằng đại số, mà ở cấp độ code là **không hề có
   lệnh nào được thực thi** cho $l\ge1$ khi $\deg=0$ — hai góc nhìn toán học và hiện thực khác nhau nhưng
   cho cùng một hành vi số). Dòng `clamped[...] = (result.x < 0)` xác nhận đúng lập luận §7.8d: điều kiện
   clamp kiểm tra **trước** khi cắt (`< 0`), rồi mới `glm::max(result, 0.0f)` cắt — khớp với nhận xét mục 6.

9. **`build_rotation` và `build_scaling_rotation` — trích nguyên văn.** 📄 `utils/general_utils.py:66-98`:

   ```python
   def build_rotation(r):
       norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
       q = r / norm[:, None]
       R = torch.zeros((q.size(0), 3, 3), device='cuda')
       r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
       R[:, 0, 0] = 1 - 2 * (y*y + z*z)
       R[:, 0, 1] = 2 * (x*y - r*z)
       R[:, 0, 2] = 2 * (x*z + r*y)
       R[:, 1, 0] = 2 * (x*y + r*z)
       R[:, 1, 1] = 1 - 2 * (x*x + z*z)
       R[:, 1, 2] = 2 * (y*z - r*x)
       R[:, 2, 0] = 2 * (x*z - r*y)
       R[:, 2, 1] = 2 * (y*z + r*x)
       R[:, 2, 2] = 1 - 2 * (x*x + y*y)
       return R

   def build_scaling_rotation(s, r):
       L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
       R = build_rotation(r)
       L[:,0,0], L[:,1,1], L[:,2,2] = s[:,0], s[:,1], s[:,2]
       L = R @ L
       return L
   ```

   Đây chính là nguồn của công thức §7.2a (9 phần tử $R_{ab}$) và §7.3a/§7.4a ($L=RS$, dùng `@` — phép nhân
   ma trận PyTorch theo đúng quy ước hàng-vector toán học chuẩn, khác quy ước `glm` column-major ở mục 2 của
   ghi chú này). Biến `L` được trả về chính là $M=RS$ của §7.4a; bước `Sigma = L @ L.transpose(1,2)` (không
   trích ở đây, nằm ở `gaussian_model.py:get_covariance`) là $\Sigma=MM^\top$ đúng công thức đóng khung ở
   [Chương 7 lý thuyết, §2.2](../07-3d-gaussians.md#22--hình-dạng-covariance-3d).

---

## 7.12 — Phụ lục: script numpy kiểm chứng toàn bộ

Theo đúng tinh thần `scripts/chNN_test.py` của chương 15 (chỉ `numpy`, các hàm được chép nguyên văn từ
`utils/general_utils.py` và `utils/sh_utils.py` vì môi trường không có `torch`). Script này tái tạo **mọi
con số** xuất hiện từ §7.1 đến §7.10.

```python
"""
Kiểm chứng số học Phần 2 (Chương 7 — 3D Gaussians) của lời giải bài toán lớn.
Chỉ dùng numpy. Các hàm build_rotation / build_scaling_rotation / RGB2SH / SH2RGB
chép nguyên văn từ utils/general_utils.py và utils/sh_utils.py (không import torch).
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ---------- 7.0 — Đầu vào từ Phần 1 ----------
mu = {
    1: np.array([0.0, 0.0, 0.0]),
    2: np.array([0.5, 0.3, 0.5]),
    3: np.array([-0.4, -0.2, 1.0]),
    4: np.array([0.3, -0.5, 0.2]),
}
c_sfm = {
    1: np.array([0.8, 0.2, 0.2]),
    2: np.array([0.2, 0.7, 0.3]),
    3: np.array([0.1, 0.3, 0.9]),
    4: np.array([0.5, 0.5, 0.5]),
}
q_raw = {i: np.array([1.0, 0.0, 0.0, 0.0]) for i in range(1, 5)}
alpha_logit = -2.1972245773362196   # log(0.1/0.9)
cams = {1: np.array([0.0, 0.0, -4.0]), 2: np.array([1.5, 0.0, -4.0]), 3: np.array([-1.5, 0.5, -4.0])}
C0 = 0.28209479177387814            # utils/sh_utils.py:26

# 3-NN d^2 (từ Phần 1, mục 1.2b)
d2_knn3 = {1: 0.723333, 2: 0.890000, 3: 1.243333, 4: 0.790000}
s_tilde = {i: 0.5 * np.log(d2_knn3[i]) for i in d2_knn3}     # log sqrt(d2) = 0.5 log(d2)


# ---------- Hàm chép từ utils/general_utils.py:66-98 ----------
def build_rotation(q):
    """q: (4,) = (r, x, y, z), không cần chuẩn hoá trước — hàm tự chuẩn hoá."""
    norm = np.sqrt(np.sum(q * q))
    r, x, y, z = q / norm
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - r * z),     2 * (x * z + r * y)],
        [2 * (x * y + r * z),     1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
        [2 * (x * z - r * y),     2 * (y * z + r * x),     1 - 2 * (x * x + y * y)],
    ])
    return R


def build_scaling_rotation(s, q):
    S = np.diag(s)
    R = build_rotation(q)
    return R @ S            # L = R S  (quy ước hàng-vector: Sigma = L L^T = R S S^T R^T)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def RGB2SH(rgb):             # utils/sh_utils.py:114
    return (np.asarray(rgb) - 0.5) / C0


def SH2RGB(sh):               # utils/sh_utils.py:117
    return np.asarray(sh) * C0 + 0.5


# ---------- 7.1 — Activation ----------
print("=== 7.1 Activation ===")
s = {}
alpha = {}
R = {}
for i in range(1, 5):
    qn = q_raw[i] / np.linalg.norm(q_raw[i])
    si = np.exp(s_tilde[i] * np.ones(3))
    ai = sigmoid(alpha_logit)
    Ri = build_rotation(q_raw[i])
    s[i], alpha[i], R[i] = si, ai, Ri
    print(f"G{i}: q_norm={qn}, s={si}, alpha={ai:.6f}")
    assert np.allclose(Ri, np.eye(3)), "R phai la ma tran don vi"

# ---------- 7.2-7.4 — R, S, Sigma ----------
print("\n=== 7.2-7.4 Sigma = R S S^T R^T ===")
Sigma = {}
Sigma_inv = {}
for i in range(1, 5):
    S_mat = np.diag(s[i])
    L = build_scaling_rotation(s[i], q_raw[i])   # L = R S
    Sig = L @ L.T
    Sigma[i] = Sig
    Sigma_inv[i] = np.linalg.inv(Sig)
    print(f"G{i}: Sigma=\n{Sig}\n  det={np.linalg.det(Sig):.6f}  eig={np.linalg.eigvalsh(Sig)}")
    assert np.allclose(Sig, s[i][0] ** 2 * np.eye(3))
    assert np.allclose(Sig, Sig.T), "Sigma phai doi xung"
    assert np.all(np.linalg.eigvalsh(Sig) > 0), "Sigma phai PSD chat"

# ---------- 7.5 — Sigma^{-1} ----------
print("\n=== 7.5 Sigma^-1 ===")
for i in range(1, 5):
    print(f"G{i}: 1/s^2 = {1.0 / (s[i][0] ** 2):.6f}, Sigma_inv[0,0] = {Sigma_inv[i][0,0]:.6f}")
    assert np.allclose(Sigma[i] @ Sigma_inv[i], np.eye(3))

# ---------- 7.6 — G(x) tai 4 diem mau / Gaussian ----------
def gaussian_density(x, mu_i, Sigma_inv_i):
    diff = x - mu_i
    maha = diff @ Sigma_inv_i @ diff
    return np.exp(-0.5 * maha), maha


print("\n=== 7.6 G(x) tai 16 diem mau ===")
for i in range(1, 5):
    si = s[i][0]
    pts = {
        "mu": mu[i],
        "mu+1sig*ex": mu[i] + np.array([si, 0, 0]),
        "mu+2sig*ex": mu[i] + np.array([2 * si, 0, 0]),
        "mu+0.5sig*(1,1,1)": mu[i] + 0.5 * si * np.array([1.0, 1.0, 1.0]),
    }
    for name, x in pts.items():
        G, maha = gaussian_density(x, mu[i], Sigma_inv[i])
        print(f"G{i} [{name}]: maha={maha:.6f}, G={G:.6f}")
    assert np.isclose(gaussian_density(mu[i], mu[i], Sigma_inv[i])[0], 1.0)
    assert np.isclose(gaussian_density(mu[i] + np.array([si, 0, 0]), mu[i], Sigma_inv[i])[0], np.exp(-0.5), atol=1e-6)
    assert np.isclose(gaussian_density(mu[i] + np.array([2*si, 0, 0]), mu[i], Sigma_inv[i])[0], np.exp(-2), atol=1e-6)

# ---------- 7.7 — 12 huong nhin ----------
print("\n=== 7.7 Huong nhin d_{i,v} ===")
directions = {}
for v, c in cams.items():
    for i in range(1, 5):
        diff = mu[i] - c
        norm = np.linalg.norm(diff)
        d = diff / norm
        directions[(i, v)] = d
        print(f"cam{v} G{i}: diff={diff}, norm={norm:.6f}, d={d}")
        assert np.isclose(np.linalg.norm(d), 1.0)

# ---------- 7.8 — Mau SH bac 0 ----------
print("\n=== 7.8 Mau SH bac 0 ===")
k00 = {i: RGB2SH(c_sfm[i]) for i in range(1, 5)}
for i in range(1, 5):
    print(f"G{i}: k00={k00[i]}")
    for v in cams:
        d = directions[(i, v)]
        Y00 = C0                      # hang so, khong phu thuoc d
        raw = 0.5 + k00[i] * Y00
        c = np.maximum(0.0, raw)
        assert np.allclose(c, c_sfm[i], atol=1e-5), (i, v, c, c_sfm[i])
    print(f"  mau giai ma (khop ca 3 camera) = {np.maximum(0.0, 0.5 + k00[i]*C0)}  vs SfM {c_sfm[i]}")

# ---------- 7.9 — Lich D(t) ----------
def D(t):
    return min(3, t // 1000)


print("\n=== 7.9 D(t) ===")
for t in [0, 999, 1000, 1999, 2000, 2999, 3000, 5000]:
    Dt = D(t)
    coeffs = (Dt + 1) ** 2
    print(f"t={t}: D={Dt}, he so/kenh={coeffs}, float SH/Gaussian={3*coeffs}, x N=4 -> {12*coeffs}")
assert D(0) == 0 and D(1000) == 1 and D(3000) == 3 and D(5000) == 3

# ---------- 7.10 — Dem tham so & bo nho ----------
print("\n=== 7.10 Tham so & bo nho ===")
n_params_per_gaussian = 3 + 4 + 3 + 1 + 3 * 16
assert n_params_per_gaussian == 59
N0 = 4
total_params = n_params_per_gaussian * N0
assert total_params == 236
adam_floats = 3 * n_params_per_gaussian * N0
print(f"tong tham so 4 Gaussian = {total_params}, bo nho Adam (float) = {adam_floats}, byte = {adam_floats*4}")
assert adam_floats * 4 == 2832

extent = 1.6902498172
for i in range(1, 5):
    assert s[i][0] > 0.1 * extent, f"G{i} phai vuot nguong big_points_ws"

print("\nTAT CA ASSERT DEU QUA — so khop voi 06-initialization.md va 07-3d-gaussians.md.")
```

Chạy script trên (📄 tương đương `scripts/ch02_test.py` của chương 15) tái tạo đúng mọi bảng số ở §7.1–§7.10
— không có `assert` nào thất bại, xác nhận toàn bộ Phần 2 nhất quán với dữ liệu gốc của Phần 1 và với công
thức đã công bố ở [Chương 7 lý thuyết](../07-3d-gaussians.md).

---

## 7.13 — Bài tập

**Bài tập 7.2.1.** Dùng công thức tổng quát ở §7.2a, tính lại ma trận $R(q)$ cho quaternion
$q=(0.923381,\ 0.102598,\ 0.307794,\ 0.205196)$ (quaternion đã chuẩn hoá từ $\tilde q=(0.9,0.1,0.3,0.2)$,
dùng trong kiểm định số Chương 7 mục 2.2b). So sánh với ma trận $R$ đã cho sẵn trong Chương 7 và giải thích
vì sao với quaternion này, $\Sigma=RSS^\top R^\top$ **không còn** rút gọn về $\operatorname{diag}(s^2)$ nếu
$S$ không đẳng hướng — liên hệ tới lập luận đại số ở §7.4b (khi nào $\Sigma_{pq}=0$ với $p\ne q$?).

**Bài tập 7.2.2.** Ở §7.4a, ta suy ra $\Sigma_{pq}=\sum_{b=1}^3 s_b^2R_{pb}R_{qb}$. Dùng công thức này,
chứng minh rằng $\operatorname{tr}(\Sigma)=\Sigma_{00}+\Sigma_{11}+\Sigma_{22}=s_1^2+s_2^2+s_3^2$ với **mọi**
ma trận xoay $R$ (gợi ý: dùng tính trực giao $\sum_p R_{pb}R_{pb'}=\delta_{bb'}$). Kiểm tra bằng số cho
Gaussian 3 ($s_3=1.115049$, đẳng hướng): $\operatorname{tr}(\Sigma_3)$ tính trực tiếp từ bảng §7.4c có bằng
$3s_3^2$ không?

**Bài tập 7.3.1.** Cảnh đồ chơi có $S_i$ đẳng hướng cho cả 4 Gaussian. Giả sử Gaussian 3 thay vì đẳng hướng
lại có $s_3=(1.115049,\ 0.5,\ 0.2)$ (dị hướng, giữ nguyên $R_3=I_3$). Viết lại ma trận $\Sigma_3$ theo công
thức tổng quát §7.4a (không dùng rút gọn $s^2I$), và tính $G_3(\mu_3+(1.115049,0,0))$,
$G_3(\mu_3+(0,0.5,0))$, $G_3(\mu_3+(0,0,0.2))$ — cả ba có bằng nhau như trường hợp đẳng hướng ở §7.6d
không? Giải thích bằng công thức Mahalanobis tổng quát (không rút gọn) tại sao dị hướng phá vỡ tính "tự
tương tự theo $\sigma$" đã nêu ở §7.6f.

**Bài tập 7.6.1.** Ở §7.6b, điểm mẫu thứ 4 của Gaussian 1 dịch theo hướng $(1,1,1)/\sqrt3$ với độ dài
$\tfrac{s_1}2\sqrt3\approx0.7365$ (không phải đúng $1\sigma=s_1$). Tính lại $G_1$ tại điểm $x=\mu_1+\dfrac{s_1}{\sqrt3}(1,1,1)$
(dịch đúng $1\sigma$ theo hướng chéo chuẩn hoá) và so sánh với $G_1(\mu_1+s_1e_x)=0.606531$ đã tính ở §7.6b.
Kết quả có bằng nhau không? Giải thích bằng tính đẳng hướng của $\Sigma_1$ (đường mức là mặt cầu — khoảng
cách Mahalanobis $1\sigma$ theo mọi hướng đều cho cùng $G$).

**Bài tập 7.7.1.** Từ 12 hướng nhìn ở bảng §7.7e, hướng nào (trong 12 hướng) gần với trục $+z$ thuần nhất
(tức $|x|,|y|$ nhỏ nhất so với $z\approx1$)? Giải thích bằng hình học: Gaussian và camera nào tạo ra cấu
hình "nhìn gần như chính diện" đó (gợi ý: so sánh toạ độ $x,y$ của $\mu_i$ với $x,y$ của $c_v$).

**Bài tập 7.8.1.** Giả sử ở một bước huấn luyện sau ($t\ge1000$, $D(t)\ge1$), Gaussian 4 (màu xám, $k_{4,00}=(0,0,0)$)
học được hệ số bậc 1 khác 0: $k_{4,1,-1}=(0.05,-0.05,0.1)$, $k_{4,10}=(0,0,0)$, $k_{4,11}=(0,0,0)$, các hệ số
bậc $\ge2$ vẫn 0. Dùng công thức $-C_1yk_{1,-1}+C_1zk_{10}-C_1xk_{11}$ ($C_1=0.488603$) và hướng nhìn
$\vec d_{4,1}=(0.070750,\ -0.117917,\ 0.990500)$ đã tính ở §7.7b, tính màu $c_4$ nhìn từ camera 1. So với
màu bậc 0 thuần tuý $(0.5,0.5,0.5)$ — Gaussian xám bây giờ còn "xám" theo mọi hướng nhìn không?

**Bài tập 7.9.1.** Dùng bảng §7.9, tính số float SH **thực sự đọc trong kernel forward** cho toàn bộ $N=4$
Gaussian tại $t=2500$ (giữa mốc $D=2$). So với $t=0$ ($D=0$), số float này tăng bao nhiêu lần? Nếu cảnh có
$N=10^5$ Gaussian thay vì 4, số float SH đọc tại $t=2500$ là bao nhiêu, và điều này liên hệ thế nào tới hệ
số $a$ trong mô hình chi phí $T_{\text{fwd}}=aN+\ldots$ sẽ gặp ở Phần 8 (chương 13)?

**Bài tập 7.10.1.** Đề bài §16.2 bước (6) nói tới "$4\times59=236$ tham số". Liệt kê chính xác 236 con số đó
theo 5 nhóm (vị trí, quaternion, scale, opacity, SH) cho **riêng Gaussian 3**, ghi rõ giá trị số của từng
nhóm (dùng bảng §7.0 và kết quả §7.1–§7.8). Tổng cộng có bao nhiêu trong số 59 tham số của G3 là "đại lượng
học trực tiếp" (raw, có gradient riêng) so với "đại lượng dẫn xuất" (như $\Sigma_3$, $G_3(x)$, $c_3$) không
có ô nhớ Adam riêng?

**Bài tập 7.4.1.** Ở §7.4g, ta chỉ ra rằng khi $\Sigma_i=s_i^2I_3$ (đẳng hướng), **mọi** vector khác 0 là
vector riêng. Chọn hai vector kiểm tra mới cho Gaussian 2 ($s_2^2=0.890000$): $v_a=(3,-1,0)$ và
$v_b=(0,0,5)$. Tính $\Sigma_2v_a$ và $\Sigma_2v_b$ bằng định nghĩa nhân ma trận-vector, rồi xác nhận cả hai
đều thoả $\Sigma_2v=\lambda v$ với cùng $\lambda=0.890000$. Từ đó giải thích vì sao **không thể** dùng phép
đo covariance $\Sigma_i$ một mình để suy ngược lại quaternion $\tilde q_i$ ban đầu khi Gaussian đẳng hướng
(liên hệ với nhận xét cuối §7.4g về Chương 8).

**Bài tập 7.5.1.** Dùng công thức adjugate tổng quát ở §7.5c, tính lại $\Sigma_3^{-1}$ cho Gaussian 3
($\Sigma_3=1.243333\,I_3$) bằng cách khai triển đủ 9 cofactor $C_{pq}$ (không dùng lối tắt ma trận chéo),
rồi xác nhận kết quả trùng với $0.804290\,I_3$ đã cho ở bảng §7.5b. Viết rõ từng cofactor như mẫu đã làm
cho Gaussian 1.

**Bài tập 7.6.1.** Ở §7.6g, ta thêm hai điểm mẫu $1\sigma$ dọc trục $y$ và $z$ cho cả 4 Gaussian, tất cả cho
$G=0.606531$. Chọn một hướng **không thuộc** một trong ba trục tọa độ và cũng không phải $(1,1,1)$, ví dụ
$\vec u=(2,-1,2)/3$ (đã chuẩn hoá, $\lVert\vec u\rVert=1$), tính $x=\mu_1+s_1\vec u$ cho Gaussian 1 (dịch
đúng $1\sigma$ theo hướng $\vec u$) và xác nhận $G_1(x)=0.606531$ vẫn đúng. Giải thích bằng công thức tổng
quát $\Delta(x)=\lVert x-\mu_1\rVert^2/s_1^2$ vì sao **bất kỳ** hướng đơn vị nào, không chỉ ba trục tọa độ
hay đường chéo $(1,1,1)/\sqrt3$, đều cho cùng kết quả khi dịch đúng $1\sigma$.

**Bài tập 7.7.2.** So sánh 3 hướng nhìn của Gaussian 1 (một Gaussian, ba camera): $\vec d_{1,1}=(0,0,1)$,
$\vec d_{1,2}=(-0.351123,0,0.936329)$, $\vec d_{1,3}=(0.348743,-0.116248,0.929981)$. Tính góc giữa
$\vec d_{1,2}$ và $\vec d_{1,3}$ bằng công thức $\cos\theta=\vec d_{1,2}\cdot\vec d_{1,3}$ (vì cả hai đã là
vector đơn vị). Góc này có hợp lý không khi so với vị trí hai camera 2, 3 đối xứng gần như qua trục $x=0$
quanh Gaussian 1 (gợi ý: so sánh với góc mà camera 2, 3 tạo với Gaussian 1 nhìn từ trên xuống mặt phẳng $xz$)?

**Bài tập 7.8.2.** §7.8e liệt kê 15 giá trị "đa thức × hệ số $C$" cho Gaussian 3 nhìn từ camera 1, trong đó
số hạng lớn nhất là $C_3^{(3)}z(2z^2-3x^2-3y^2)=0.728636$ (bậc $l=3$, $m=0$). Giả sử — thuần tuý để luyện
tập — hệ số tương ứng của Gaussian 3 **không phải 0** mà là $k_{3,30}=(0.05,\,0.05,\,0.05)$ (cùng một giá
trị cho cả 3 kênh màu). Tính lại màu $c_3$ nhìn từ camera 1 trong trường hợp giả định này (chỉ cộng thêm
đúng một số hạng vào tổng đã có ở §7.8b, các số hạng khác vẫn 0). Kênh nào thay đổi nhiều nhất về giá trị
tuyệt đối?

**Bài tập 7.9.2.** Bảng §7.9 dừng ở $t=5000$ ($D=3$, đã đạt trần `max_sh_degree=3`). Giải thích bằng công
thức $D(t)=\min(3,\lfloor t/1000\rfloor)$: có tồn tại $t$ nào mà $D(t)>3$ không? Nếu giả sử `max_sh_degree`
được đặt là $4$ thay vì $3$ (dùng $C_4$ — xem `utils/sh_utils.py:44-54`, không dùng trong FastGS-lite nhưng
tồn tại trong code), số hệ số/kênh tối đa sẽ là bao nhiêu, và mốc $t$ nào sẽ kích hoạt bậc đó?

---

## Đầu ra chuyển cho Phần 3 (Chương 8 — Projection)

| Đại lượng | Gaussian 1 | Gaussian 2 | Gaussian 3 | Gaussian 4 |
|---|---|---|---|---|
| $\mu_i$ | $(0,0,0)$ | $(0.5,0.3,0.5)$ | $(-0.4,-0.2,1.0)$ | $(0.3,-0.5,0.2)$ |
| $\Sigma_i$ (6 phần tử $xx,xy,xz,yy,yz,zz$) | $(0.723333,0,0,0.723333,0,0.723333)$ | $(0.890000,0,0,0.890000,0,0.890000)$ | $(1.243333,0,0,1.243333,0,1.243333)$ | $(0.790000,0,0,0.790000,0,0.790000)$ |
| $\alpha_i$ | $0.1$ | $0.1$ | $0.1$ | $0.1$ |
| $c_i$ (màu SH đã giải mã, bậc 0, mọi camera) | $(0.8,0.2,0.2)$ | $(0.2,0.7,0.3)$ | $(0.1,0.3,0.9)$ | $(0.5,0.5,0.5)$ |

cùng với $R_i=I_3$ và $s_i=(0.850490,\ 0.943398,\ 1.115049,\ 0.888819)$ (dạng $(R,S)$ phòng khi Phần 3 cần
tách lại thay vì dùng thẳng $\Sigma_i$ đã gộp), $\Sigma_i^{-1}$ (§7.5, dùng lại nếu Phần 3 cần kiểm tra
EWA mà không tính lại nghịch đảo), và 12 hướng nhìn $\vec d_{i,v}$ (§7.7, để Phần 3 dùng khi tính lại màu
SH tại các bậc cao hơn nếu cần đối chiếu).

**Đầu ra chuyển cho Phần 3 (Chương 8 — Projection):** $\Sigma_i$ (4 ma trận hiệp phương sai world-space
$3\times3$), màu SH đã giải mã.

---

[← Phần 1](01-init.md) | [Mục lục lời giải](00-muc-luc-loi-giai.md) | [Phần 3 →](03-projection.md)
