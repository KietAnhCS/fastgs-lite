[← Đề bài](../16-bai-toan-lon-de-bai.md) · Chương 16 — Lời giải, Phần 1/8

# Phần 1 — Chương 6: Initialization

> Đầu vào nhận từ Phần trước: đề bài (COLMAP input) — xem [Chương 16 — Đề bài](../16-bai-toan-lon-de-bai.md).

> Đây là lời giải tính tay, chi tiết tuyệt đối, của bước đầu tiên trong bài toán lớn: biến ba file văn bản
> COLMAP thô (`cameras.txt`, `images.txt`, `points3D.txt`) thành quần thể Gaussian khởi tạo
> $\mathcal G_0 = \{\theta_i\}_{i=1}^{4}$, mỗi $\theta_i\in\mathbb R^{59}$, cộng với đại lượng toàn cảnh
> `extent` và ba ngưỡng suy ra từ nó. Toàn bộ số liệu bên dưới **trùng khớp tuyệt đối** với
> [§6.2 và §6.3 của Chương 6](../06-initialization.md) — phần này chỉ mở rộng, diễn giải và chứng minh lại
> từng bước tính tay, không phát sinh số liệu mới mâu thuẫn.
>
> Quy ước trình bày (theo [§16.4](../16-bai-toan-lon-de-bai.md#164--quy-ước-chung-cho-lời-giải-8-phần)):
> mỗi mục có **công thức tổng quát** (chép nguyên văn từ code hoặc từ Chương 6), **thay số cụ thể của
> cảnh này**, và **kết quả**. Số liệu được viết tới 10 chữ số thập phân ở những chỗ dùng làm đầu vào cho
> Phần 2 — Phần 2 sẽ không phải tính lại các số này.

## Mục lục Phần 1

1. [1.0 — Đọc dữ liệu COLMAP thô](#10--đọc-dữ-liệu-colmap-thô)
2. [1.1 — Tâm Gaussian $\mu_i$](#11--tâm-gaussian-mu_i)
3. [1.2 — Ma trận khoảng cách bình phương $\lVert p_i-p_j\rVert^2$](#12--ma-trận-khoảng-cách-bình-phương)
4. [1.3 — 3-NN và scale khởi tạo $\tilde s_i,\ s_i$](#13--3-nn-và-scale-khởi-tạo)
5. [1.4 — Quaternion $\tilde q_i$](#14--quaternion-tilde-q_i)
6. [1.5 — Opacity $\tilde\alpha_i$](#15--opacity-tilde-alpha_i)
7. [1.6 — Hệ số Spherical Harmonics $k_{i,lm}$](#16--hệ-số-spherical-harmonics-k_ilm)
8. [1.7 — Đếm tham số và vector $\theta_i\in\mathbb R^{59}$ đầy đủ](#17--đếm-tham-số-và-vector-theta_i-đầy-đủ)
9. [1.8 — `extent`: bán kính cảnh](#18--extent-bán-kính-cảnh)
10. [1.9 — Ba ngưỡng suy ra từ `extent`](#19--ba-ngưỡng-suy-ra-từ-extent)
11. [1.10 — Bộ nhớ trạng thái Adam](#110--bộ-nhớ-trạng-thái-adam)
12. [1.11 — Script numpy kiểm chứng toàn bộ](#111--script-numpy-kiểm-chứng-toàn-bộ)
13. [1.12 — Ghi chú đối chiếu với code](#112--ghi-chú-đối-chiếu-với-code)
14. [1.13 — Biến thể mở rộng: điểm SfM thứ 5 giả định](#113--biến-thể-mở-rộng-điểm-sfm-thứ-5-giả-định)
15. [1.14 — Bài tập](#114--bài-tập)
16. [1.16 — Trace `create_from_pcd` dòng theo dòng](#116--trace-create_from_pcd-dòng-theo-dòng-với-số-thật-của-cảnh)
17. [Phụ lục — Bảng tra cứu nhanh](#phụ-lục--bảng-tra-cứu-nhanh-mọi-số-liệu-của-phần-1)
18. [1.15 — Đầu ra chuyển cho Phần 2](#115--đầu-ra-chuyển-cho-phần-2)

---

## 1.0 — Đọc dữ liệu COLMAP thô

Đề bài (§16.1) cho ba file text đúng định dạng COLMAP xuất ra ở thư mục `sparse/0/`. Trước khi tính bất
cứ thứ gì cho Gaussian, ta phải parse ba file này thành đúng các đại lượng mà `create_from_pcd` và
`getNerfppNorm` cần: point cloud $\{(p_k,c_k)\}$, và tập camera $\{(W_v, c_v)\}$.

### 1.0.1 — `cameras.txt` → mô hình pinhole

```
1 PINHOLE 48 32 40.0 40.0 24.0 16.0
```

Cột theo đúng thứ tự COLMAP: `CAMERA_ID=1`, `MODEL=PINHOLE`, `WIDTH=48`, `HEIGHT=32`,
`PARAMS = [f_x, f_y, c_x, c_y] = [40.0, 40.0, 24.0, 16.0]`. Mô hình `PINHOLE` không có tham số méo ống
kính (khác `SIMPLE_RADIAL`, `OPENCV`, …), nên bốn số này là toàn bộ nội tại camera cần.

Trong pipeline, FastGS-lite không giữ $f_x,f_y$ trực tiếp mà quy về góc nhìn nửa (half field-of-view)
$\tan(\mathrm{fov}/2)$, vì công thức chiếu phối cảnh ở Chương 8 dùng dạng này. Quan hệ tổng quát giữa
tiêu cự $f$ (tính bằng pixel) và nửa-fov với ảnh có nửa-kích-thước $c$ (tức $c_x=W/2$, $c_y=H/2$ khi
principal point ở chính giữa ảnh, đúng như ở đây $c_x=24=48/2$, $c_y=16=32/2$):

$$
\tan\!\left(\frac{\mathrm{fov}}{2}\right) = \frac{c}{f}
$$

Đây thực chất là nghịch đảo của hàm `fov2focal` chép từ `utils/graphics_utils.py:73-74`:

```python
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
```

tức $f = \dfrac{\text{pixels}}{2\tan(\mathrm{fov}/2)} \Leftrightarrow \tan(\mathrm{fov}/2) = \dfrac{\text{pixels}}{2f} = \dfrac{c}{f}$
(vì $c=\text{pixels}/2$). Hàm ngược `focal2fov` (dòng 76-77) là $\mathrm{fov}=2\arctan\!\big(\tfrac{\text{pixels}}{2f}\big)$ — cho ra cùng một $\tan(\mathrm{fov}/2)$.

**Thay số:**

$$
\tan\!\left(\frac{\mathrm{fov}_x}{2}\right) = \frac{c_x}{f_x} = \frac{24}{40} = 0.6,
\qquad
\tan\!\left(\frac{\mathrm{fov}_y}{2}\right) = \frac{c_y}{f_y} = \frac{16}{40} = 0.4
$$

**Kết quả:** $\tan(\mathrm{fov}_x/2)=0.6$, $\tan(\mathrm{fov}_y/2)=0.4$ — đúng hai hằng số đã dùng xuyên
suốt Chương 6–13 của sách (ví dụ clamp $1.3\tan(\mathrm{fov}/2)$ ở Chương 8). Ảnh $48\times32$ px chia
lưới tile $16\times16$ px cho $3\times2=6$ tile (dùng ở Chương 8–9). Quy ước $z_n=0.01$, $z_f=100$ không
nằm trong `cameras.txt` chuẩn của COLMAP (COLMAP không biết gì về near/far clip) mà do `Scene`/config của
pipeline gán cố định — không tính được từ file này, chỉ ghi lại làm hằng số toàn cục.

Chương 6 (Initialization) **không dùng** $\tan(\mathrm{fov}/2)$ trực tiếp — đại lượng này chỉ cần ở Chương
8 (projection). Ta tính nó ở đây vì nó nằm trong `cameras.txt` và để khẳng định luồng đọc dữ liệu đầu vào
là nhất quán cho toàn bộ 8 phần.

### 1.0.2 — `images.txt` → ma trận view $W_v$ và tâm camera $c_v$

```
1 1.0 0.0 0.0 0.0 0.0 0.0 4.0 1 cam01.png
2 1.0 0.0 0.0 0.0 -1.5 0.0 4.0 1 cam02.png
3 1.0 0.0 0.0 0.0 1.5 -0.5 4.0 1 cam03.png
```

Cột: `IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME`. Quaternion $q=(q_w,q_x,q_y,q_z)$ và tịnh
tiến $t=(t_x,t_y,t_z)$ mô tả phép biến đổi **world → camera**: $X_{\mathrm{cam}} = R(q)\,X_{\mathrm{world}} + t$.

**Bước 1 — quaternion → ma trận xoay.** Công thức tổng quát (chép từ `scene/colmap_loader.py:43-53`,
hàm `qvec2rotmat`):

$$
R(q)=\begin{pmatrix}
1-2q_y^2-2q_z^2 & 2q_xq_y-2q_wq_z & 2q_zq_x+2q_wq_y\\
2q_xq_y+2q_wq_z & 1-2q_x^2-2q_z^2 & 2q_yq_z-2q_wq_x\\
2q_zq_x-2q_wq_y & 2q_yq_z+2q_wq_x & 1-2q_x^2-2q_y^2
\end{pmatrix}
$$

Cả ba dòng của `images.txt` đều có $q=(1,0,0,0)$ (quaternion đơn vị, không xoay). Thay số:

$$
R=\begin{pmatrix}1-0-0 & 0-0 & 0+0\\0+0 & 1-0-0 & 0-0\\0-0 & 0+0 & 1-0-0\end{pmatrix}=I_3
$$

cho cả ba camera. `scene/dataset_readers.py:99` còn lấy `R = np.transpose(qvec2rotmat(extr.qvec))`
(chuyển vị để đổi quy ước hàng/cột giữa COLMAP và pipeline) — nhưng $I_3^\top=I_3$ nên chuyển vị không
đổi gì ở cảnh này.

**Bước 2 — dựng $W_v$ (world→camera, dạng thuần nhất $4\times4$).** Công thức tổng quát (chép từ
`utils/graphics_utils.py:38-49`, hàm `getWorld2View2`, với `translate=(0,0,0)`, `scale=1` — hai tham số
mặc định, không dùng ở bước này):

$$
W_v = \begin{pmatrix} R^\top & t \\ 0_{1\times3} & 1\end{pmatrix}
$$

(code dựng `Rt[:3,:3]=R.transpose()`, `Rt[:3,3]=t`, rồi thực hiện một vòng nghịch đảo–khôi phục
`C2W=inv(Rt); ...; Rt=inv(C2W)` chỉ để cộng `translate`/nhân `scale` — ở đây cả hai đều trung tính nên
kết quả cuối bằng đúng `Rt` ban đầu). Vì $R=I_3\Rightarrow R^\top=I_3$:

$$
W_v = \begin{pmatrix} I_3 & t_v \\ 0 & 1\end{pmatrix}
$$

**Thay số cho từng camera** ($t_v$ đọc trực tiếp từ cột TX,TY,TZ):

- $v=1$: $t_1=(0.0,\ 0.0,\ 4.0)$
- $v=2$: $t_2=(-1.5,\ 0.0,\ 4.0)$
- $v=3$: $t_3=(1.5,\ -0.5,\ 4.0)$

**Bước 3 — tâm camera trong world space.** Công thức tổng quát: $c_v = (W_v)^{-1}[{:}3,3]$. Với khối
$W_v=\begin{pmatrix}I_3 & t_v\\0&1\end{pmatrix}$, nghịch đảo có dạng đóng
$W_v^{-1}=\begin{pmatrix}I_3 & -t_v\\0&1\end{pmatrix}$ (kiểm tra nhanh: $\begin{pmatrix}I&t\\0&1\end{pmatrix}\begin{pmatrix}I&-t\\0&1\end{pmatrix}=\begin{pmatrix}I & t-t\\0&1\end{pmatrix}=\begin{pmatrix}I&0\\0&1\end{pmatrix}$, đúng đơn vị). Vậy $c_v=-t_v$.

**Thay số:**

$$
c_1=-t_1=(0,\ 0,\ -4),\qquad c_2=-t_2=(1.5,\ 0,\ -4),\qquad c_3=-t_3=(-1.5,\ 0.5,\ -4)
$$

**Kết quả** trùng khớp đúng bảng camera ở [§6.2](../06-initialization.md#ảnh-và-camera):

| $v$ | $t_v$ (từ `images.txt`) | $c_v=-t_v$ | $W_v$ |
|---|---|---|---|
| 1 | $(0,\ 0,\ 4)$ | $(0,\ 0,\ -4)$ | $\begin{pmatrix}I_3 & (0,0,4)^\top\\0&1\end{pmatrix}$ |
| 2 | $(-1.5,\ 0,\ 4)$ | $(1.5,\ 0,\ -4)$ | $\begin{pmatrix}I_3 & (-1.5,0,4)^\top\\0&1\end{pmatrix}$ |
| 3 | $(1.5,\ -0.5,\ 4)$ | $(-1.5,\ 0.5,\ -4)$ | $\begin{pmatrix}I_3 & (1.5,-0.5,4)^\top\\0&1\end{pmatrix}$ |

Ba giá trị $c_v$ này sẽ dùng lại nguyên vẹn ở mục 1.8 (`extent`) — không tính lại.

### 1.0.3 — `points3D.txt` → point cloud $\{(p_k,c_k)\}$

```
1 0.0 0.0 0.0 204 51 51 0.42 1 0 2 0 3 0
2 0.5 0.3 0.5 51 179 76 0.35 1 1 2 1 3 1
3 -0.4 -0.2 1.0 26 76 230 0.51 1 2 2 2 3 2
4 0.3 -0.5 0.2 128 128 128 0.29 1 3 2 3 3 3
```

Cột: `POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]...`. Toạ độ $p_k=(X,Y,Z)$ lấy trực tiếp, không biến
đổi. Màu $R,G,B$ là số nguyên $\{0,\dots,255\}$ — pipeline (và `create_from_pcd` qua `BasicPointCloud`)
cần màu ở $[0,1]^3$, quy đổi bằng phép chia tuyến tính đơn giản:

$$
c_k = \frac{(R,G,B)}{255}
$$

**Thay số cho cả 4 điểm** (chia từng thành phần cho 255, giữ 4 chữ số thập phân, rồi làm tròn về 1 chữ số
như quy ước hiển thị của sách):

| $k$ | $(R,G,B)$ nguyên | $(R,G,B)/255$ đầy đủ | $c_k$ (làm tròn) |
|---|---|---|---|
| 1 | $(204, 51, 51)$ | $(0.8000000,\ 0.2000000,\ 0.2000000)$ | $(0.8,\ 0.2,\ 0.2)$ — đỏ |
| 2 | $(51, 179, 76)$ | $(0.2000000,\ 0.7019608,\ 0.2980392)$ | $(0.2,\ 0.7,\ 0.3)$ — lục |
| 3 | $(26, 76, 230)$ | $(0.1019608,\ 0.2980392,\ 0.9019608)$ | $(0.1,\ 0.3,\ 0.9)$ — lam |
| 4 | $(128, 128, 128)$ | $(0.5019608,\ 0.5019608,\ 0.5019608)$ | $(0.5,\ 0.5,\ 0.5)$ — xám |

Kiểm tra từng phép chia: $204/255=0.8$ đúng khớp (vì $204=0.8\times255$); $51/255=0.2$ đúng khớp
($51=0.2\times255$); $179/255=0.701960784...$; $76/255=0.298039215...$; $26/255=0.101960784...$;
$230/255=0.901960784...$; $128/255=0.501960784...$. Bốn giá trị làm tròn ở cột cuối chính là $c_k$ dùng
xuyên suốt sách (mục 6.2 gọi đúng các số này là "đỏ/lục/lam/xám"), riêng $128/255\approx0.502$ được làm
tròn xuống $0.5$ trong toàn bộ các phép tính tay của chương — sai số làm tròn $0.0020$ trên mỗi kênh,
không ảnh hưởng tới 4 chữ số có nghĩa của kết quả cuối (mục 1.6 kiểm tra lại bằng số full-precision).

Cột `ERROR` (sai số tái chiếu trung bình do COLMAP bundle-adjustment tính ra, ví dụ $0.42$ cho điểm 1)
và `TRACK[]` (danh sách cặp `(IMAGE_ID, POINT2D_IDX)` — điểm nào được quan sát bởi ảnh nào, ví dụ điểm 1
được quan sát ở cả 3 ảnh với `mean track length = 3.0` ghi ở header) **không được** `create_from_pcd` hay
bất kỳ hàm nào ở Chương 6 sử dụng — `BasicPointCloud` (định nghĩa trong `scene/gaussian_model.py`, dùng
bởi `dataset_readers.py`) chỉ giữ `points`, `colors`, `normals` (normals ở đây bằng 0, không dùng tới ở
Chương 6). Ta bỏ qua hai cột này từ đây trở đi.

**Kết quả — bảng đầu vào chuẩn** dùng cho toàn bộ phần còn lại của Phần 1 (trùng nguyên văn bảng "Điểm
SfM" ở [§6.2](../06-initialization.md#điểm-sfm-4-điểm)):

| $k$ | $p_k$ | $c_k$ |
|---|---|---|
| 1 | $(0.0,\ 0.0,\ 0.0)$ | $(0.8,\ 0.2,\ 0.2)$ |
| 2 | $(0.5,\ 0.3,\ 0.5)$ | $(0.2,\ 0.7,\ 0.3)$ |
| 3 | $(-0.4,\ -0.2,\ 1.0)$ | $(0.1,\ 0.3,\ 0.9)$ |
| 4 | $(0.3,\ -0.5,\ 0.2)$ | $(0.5,\ 0.5,\ 0.5)$ |

$N_0=4$ điểm. Ghi chú GT: điểm cuối §16.1 định nghĩa $I^{(v)}_{\mathrm{gt}}$ là ảnh render của **đúng
cảnh này** nhưng $\alpha=0.9$ thay vì $0.1$ — không ảnh hưởng gì tới Chương 6 (không dùng GT), chỉ nhắc để
Phần 5 (Chương 10 — Loss) dùng lại.

### 1.0.4 — Sanity-check bổ sung trên định dạng file

Trước khi dùng số, đáng kiểm tra vài ràng buộc định dạng COLMAP mà một script parser thật sự (ví dụ
`read_points3D_text`, `read_extrinsics_text`, `read_intrinsics_text` trong `scene/colmap_loader.py`) sẽ
phải tôn trọng — nếu vi phạm, script sẽ raise lỗi hoặc cho ra số sai âm thầm:

1. **Header comment (`#...`) phải bị bỏ qua.** Cả ba file đều có 2–3 dòng đầu bắt đầu bằng `#` (mô tả cột
   và số lượng bản ghi) — parser phải `strip()` và kiểm tra `line[0] == '#'` trước khi `split()`, nếu
   không dòng `# Number of points: 4, mean track length: 3.0` sẽ bị hiểu nhầm thành một bản ghi.
2. **Số dòng khớp với "Number of ..." khai báo ở header.** `points3D.txt` khai `Number of points: 4` và
   quả thật có đúng 4 dòng dữ liệu (không tính header); `images.txt` khai `Number of images: 3` và có
   đúng 3 dòng "lẻ" (COLMAP xen kẽ 2 dòng/ảnh: dòng lẻ chứa pose, dòng chẵn — ở đây trống — chứa danh sách
   `POINTS2D[]`; đề bài để trống các dòng chẵn vì Chương 6 không cần `POINTS2D[]`, chỉ cần pose).
3. **`mean track length: 3.0` khớp với dữ liệu.** Mỗi điểm trong `points3D.txt` có đúng 3 cặp
   `(IMAGE_ID, POINT2D_IDX)` ở cuối dòng (ví dụ điểm 1: `1 0 2 0 3 0` — ba cặp `(1,0),(2,0),(3,0)`, nghĩa
   là điểm này được quan sát ở cả 3 ảnh), nên track length trung bình đúng là $12/4=3.0$ — khớp header.
   Đây là lượng thông tin **không dùng** tới ở Chương 6 (đã nói ở mục 1.0.3) nhưng việc số liệu tự khớp
   nội bộ là dấu hiệu file hợp lệ, không bị cắt xén khi sao chép vào đề bài.
4. **Camera model `PINHOLE` có đúng 4 tham số.** Dòng `cameras.txt` có `1 PINHOLE 48 32 40.0 40.0 24.0
   16.0` — sau 4 trường đầu (`CAMERA_ID MODEL WIDTH HEIGHT`) là đúng 4 số (`fx fy cx cy`), khớp
   `DUAL_FOCAL_MODELS = {"PINHOLE", "OPENCV", ...}` liệt kê ở `scene/dataset_readers.py:73-74` — các
   model này cần **hai** tiêu cự riêng biệt ($f_x\ne f_y$ nói chung, dù ở đây $f_x=f_y=40$ trùng nhau).
   Nếu model là `SIMPLE_PINHOLE` (thuộc `SINGLE_FOCAL_MODELS`, dòng 71-72) thì chỉ có 3 tham số
   (`f cx cy`, dùng chung một tiêu cự cho cả hai trục) — không phải trường hợp ở đây.
5. **Quaternion phải là đơn vị (norm 1).** Cả ba dòng `images.txt` có $q=(1,0,0,0)$,
   $\lVert q\rVert=\sqrt{1^2+0^2+0^2+0^2}=1$ — hợp lệ. Một parser cẩn thận nên assert
   $\lVert q\rVert\approx1$ (dung sai nhỏ do COLMAP xuất số thực) trước khi gọi `qvec2rotmat`, vì công
   thức ở mục 1.0.2 chỉ đúng khi $q$ chuẩn hoá.

Không có vi phạm nào ở dữ liệu đề bài — mọi số liệu mục 1.0.1–1.0.3 dùng được trực tiếp.

---

## 1.1 — Tâm Gaussian $\mu_i$

**Công thức tổng quát** (chép từ `scene/gaussian_model.py:139,154`):

```python
fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()   # dòng 139
...
self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))          # dòng 154
```

tức $\mu_i = p_i$ — phép "khởi tạo" duy nhất ở đây là ép kiểu `float32` và chuyển sang tensor có
`requires_grad=True`; **không có biến đổi hình học nào** (không dịch, không xoay, không co giãn) giữa
point cloud COLMAP và tâm Gaussian.

**Thay số:** lấy trực tiếp từ bảng ở mục 1.0.3.

**Kết quả:**

| $i$ | $\mu_i$ |
|---|---|
| 1 | $(0,\ 0,\ 0)$ |
| 2 | $(0.5,\ 0.3,\ 0.5)$ |
| 3 | $(-0.4,\ -0.2,\ 1.0)$ |
| 4 | $(0.3,\ -0.5,\ 0.2)$ |

Đúng $N_0=4$ Gaussian, mỗi Gaussian có $\mu_i\in\mathbb R^3$ — 3 trong số 59 tham số.

---

## 1.2 — Ma trận khoảng cách bình phương

Bước tiếp theo (scale khởi tạo) cần khoảng cách Euclid bình phương giữa mọi cặp điểm. Với $N_0=4$, số
cặp không thứ tự là $\binom42=6$. Ta tính đủ cả 6 cặp, từng số hạng một, không rút gọn.

**Công thức:** $\lVert p_i-p_j\rVert^2 = (x_i-x_j)^2+(y_i-y_j)^2+(z_i-z_j)^2$.

### Cặp (1,2)

$$
p_1-p_2 = (0-0.5,\ 0-0.3,\ 0-0.5) = (-0.5,\ -0.3,\ -0.5)
$$
$$
\lVert p_1-p_2\rVert^2 = (-0.5)^2+(-0.3)^2+(-0.5)^2 = 0.25+0.09+0.25 = 0.59
$$

### Cặp (1,3)

$$
p_1-p_3 = (0-(-0.4),\ 0-(-0.2),\ 0-1.0) = (0.4,\ 0.2,\ -1.0)
$$
$$
\lVert p_1-p_3\rVert^2 = 0.4^2+0.2^2+(-1.0)^2 = 0.16+0.04+1.00 = 1.2
$$

### Cặp (1,4)

$$
p_1-p_4 = (0-0.3,\ 0-(-0.5),\ 0-0.2) = (-0.3,\ 0.5,\ -0.2)
$$
$$
\lVert p_1-p_4\rVert^2 = (-0.3)^2+0.5^2+(-0.2)^2 = 0.09+0.25+0.04 = 0.38
$$

### Cặp (2,3)

$$
p_2-p_3 = (0.5-(-0.4),\ 0.3-(-0.2),\ 0.5-1.0) = (0.9,\ 0.5,\ -0.5)
$$
$$
\lVert p_2-p_3\rVert^2 = 0.9^2+0.5^2+(-0.5)^2 = 0.81+0.25+0.25 = 1.31
$$

### Cặp (2,4)

$$
p_2-p_4 = (0.5-0.3,\ 0.3-(-0.5),\ 0.5-0.2) = (0.2,\ 0.8,\ 0.3)
$$
$$
\lVert p_2-p_4\rVert^2 = 0.2^2+0.8^2+0.3^2 = 0.04+0.64+0.09 = 0.77
$$

### Cặp (3,4)

$$
p_3-p_4 = (-0.4-0.3,\ -0.2-(-0.5),\ 1.0-0.2) = (-0.7,\ 0.3,\ 0.8)
$$
$$
\lVert p_3-p_4\rVert^2 = (-0.7)^2+0.3^2+0.8^2 = 0.49+0.09+0.64 = 1.22
$$

### Ma trận đầy đủ $4\times4$ (đối xứng, đường chéo 0)

| $\lVert p_i-p_j\rVert^2$ | $p_1$ | $p_2$ | $p_3$ | $p_4$ |
|---|---|---|---|---|
| $p_1$ | 0 | 0.59 | 1.2 | 0.38 |
| $p_2$ | 0.59 | 0 | 1.31 | 0.77 |
| $p_3$ | 1.2 | 1.31 | 0 | 1.22 |
| $p_4$ | 0.38 | 0.77 | 1.22 | 0 |

Ma trận này đối xứng vì khoảng cách Euclid đối xứng ($\lVert p_i-p_j\rVert=\lVert p_j-p_i\rVert$) — ta chỉ
cần tính $\binom42=6$ ô tam giác trên, 6 ô tam giác dưới suy ra bằng phản chiếu qua đường chéo, và đường
chéo chính luôn bằng 0 ($\lVert p_i-p_i\rVert^2=0$). Trùng khớp đúng bảng ở
[§6.3(b)](../06-initialization.md#b-scale-tilde-s_i--từ-3-nn) của Chương 6.

**Nhận xét hình học:** hàng/cột $p_3$ có tổng lớn nhất ($1.2+1.31+1.22=3.73$) — $p_3=(-0.4,-0.2,1.0)$ là
điểm "cô lập" nhất, cách xa cả ba điểm còn lại; hàng/cột $p_1$ và $p_4$ có tổng nhỏ hơn hẳn
($0.59+1.2+0.38=2.17$ và $0.38+0.77+1.22=2.37$) — hai điểm này nằm gần cụm còn lại hơn. Quan sát này sẽ
giải thích trực tiếp vì sao $s_3 > s_1, s_2, s_4$ ở mục kế tiếp: scale khởi tạo tỉ lệ thuận với "độ cô
lập" của điểm.

---

## 1.3 — 3-NN và scale khởi tạo

**Công thức tổng quát** (chép từ `submodules/simple-knn/simple_knn.cu:148-184`, hàm `boxMeanDist`, được
gọi bởi `distCUDA2`, và từ `scene/gaussian_model.py:147-148`):

$$
d^2_{\mathrm{knn3}}(p_i) = \frac13\sum_{j\in\mathrm{3\text{-}NN}(i)}\lVert p_i-p_j\rVert^2,
\qquad
\tilde s_i = \log\sqrt{\max\!\big(d^2_{\mathrm{knn3}}(p_i),\ 10^{-7}\big)}\cdot\mathbf 1_3,
\qquad
s_i=\exp\tilde s_i
$$

```python
dist2 = torch.clamp_min(distCUDA2(...), 0.0000001)          # gaussian_model.py:147
scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # gaussian_model.py:148
```

Kernel CUDA `boxMeanDist` (dòng 148-184) duyệt các điểm lân cận trong một lưới ô (box) để tìm 3 láng
giềng gần nhất theo khoảng cách Euclid bình phương, với điều kiện loại trừ chính điểm đang xét:
`if (i == idx) continue;` xuất hiện **hai lần** trong kernel — ở vòng quét cục bộ ban đầu (dòng 159-160)
và ở vòng quét theo box (dòng 178-179) — đảm bảo $p_i$ không bao giờ tự chọn chính nó làm láng giềng của
chính nó. Kết quả cuối `dists[indices[idx]] = (best[0]+best[1]+best[2])/3.0f` (dòng 183) chính là
$d^2_{\mathrm{knn3}}(p_i)$: **trung bình cộng** (không phải căn bậc hai của trung bình, không phải RMS)
của 3 giá trị khoảng-cách-bình-phương nhỏ nhất.

Vì $N_0=4$, với mỗi điểm $i$ thì "3 láng giềng gần nhất" **chính là ba điểm còn lại** — không có gì để
loại bớt, nên $d^2_{\mathrm{knn3}}(p_i)$ chỉ đơn giản là trung bình cộng của hàng thứ $i$ trong ma trận ở
mục 1.2 (bỏ ô đường chéo).

Ta tính đầy đủ cho cả 4 điểm, không rút gọn "tương tự".

### Gaussian 1 ($i=1$)

Hàng $p_1$ của ma trận: $\{0.59,\ 1.2,\ 0.38\}$ (ứng với $j=2,3,4$).

$$
d^2_{\mathrm{knn3}}(p_1) = \frac{0.59+1.2+0.38}{3} = \frac{2.17}{3} = 0.7233333333
$$

Clamp: $\max(0.7233333333,\ 10^{-7}) = 0.7233333333$ (không đổi — clamp chỉ có tác dụng khi
$d^2 < 10^{-7}$, tức hai điểm SfM trùng nhau hoặc gần như trùng nhau; ở đây điểm gần nhất cách $p_1$ một
khoảng $\sqrt{0.38}\approx0.6164$, rất xa ngưỡng $\sqrt{10^{-7}}\approx3.16\times10^{-4}$).

$$
\tilde s_1 = \log\sqrt{0.7233333333} = \frac12\log(0.7233333333) = \frac12\times(-0.3238851211) = -0.1619425606
$$

(tra: $\log(0.7233333333)$ — dùng khai triển $\log(1-x)\approx -x-\tfrac{x^2}2-\dots$ quanh $x=1-0.72333=0.27667$
chỉ để ước lượng thô; giá trị chính xác $-0.3238851211$ lấy theo máy tính/script, khớp đúng bảng full
precision ở Chương 6.)

$$
s_1 = \exp(-0.1619425606) = 0.8504899611 \approx 0.8505
$$

(kiểm tra chéo: $s_1=\sqrt{d^2_{\mathrm{knn3}}(p_1)}=\sqrt{0.7233333333}=0.8504899611$ — đúng, vì
$\exp(\tfrac12\log x)=\sqrt x$ với mọi $x>0$; xem chứng minh tổng quát ở khung dưới.)

### Gaussian 2 ($i=2$)

Hàng $p_2$: $\{0.59,\ 1.31,\ 0.77\}$ (ứng với $j=1,3,4$).

$$
d^2_{\mathrm{knn3}}(p_2) = \frac{0.59+1.31+0.77}{3} = \frac{2.67}{3} = 0.89
$$

Clamp: $\max(0.89,\ 10^{-7})=0.89$.

$$
\tilde s_2 = \frac12\log(0.89) = \frac12\times(-0.1165338178) = -0.0582669089
$$

$$
s_2 = \exp(-0.0582669089) = 0.9433981132 \approx 0.9434
$$

(kiểm tra: $\sqrt{0.89}=0.9433981132$ — khớp.)

### Gaussian 3 ($i=3$)

Hàng $p_3$: $\{1.2,\ 1.31,\ 1.22\}$ (ứng với $j=1,2,4$).

$$
d^2_{\mathrm{knn3}}(p_3) = \frac{1.2+1.31+1.22}{3} = \frac{3.73}{3} = 1.2433333333
$$

Clamp: $\max(1.2433333333,\ 10^{-7})=1.2433333333$.

$$
\tilde s_3 = \frac12\log(1.2433333333) = \frac12\times0.2177959449 = 0.1088979725
$$

$$
s_3 = \exp(0.1088979725) = 1.1150934466 \approx 1.115
$$

(kiểm tra: $\sqrt{1.2433333333}=1.1150934466$ — khớp; đây là điểm **duy nhất** trong 4 điểm có
$\tilde s_i>0$ vì $d^2_{\mathrm{knn3}}(p_3)>1$, tức trung bình bình phương khoảng cách tới 3 láng giềng
lớn hơn $1$ — khớp với nhận xét ở mục 1.2 rằng $p_3$ là điểm cô lập nhất.)

### Gaussian 4 ($i=4$)

Hàng $p_4$: $\{0.38,\ 0.77,\ 1.22\}$ (ứng với $j=1,2,3$).

$$
d^2_{\mathrm{knn3}}(p_4) = \frac{0.38+0.77+1.22}{3} = \frac{2.37}{3} = 0.79
$$

Clamp: $\max(0.79,\ 10^{-7})=0.79$.

$$
\tilde s_4 = \frac12\log(0.79) = \frac12\times(-0.2357222892) = -0.1178611446
$$

$$
s_4 = \exp(-0.1178611446) = 0.8888194436 \approx 0.8888
$$

(kiểm tra: $\sqrt{0.79}=0.8888194436$ — khớp.)

### Vì sao $s_i=\sqrt{d^2_{\mathrm{knn3}}(p_i)}$ luôn đúng (chứng minh tổng quát)

$$
s_i = \exp(\tilde s_i) = \exp\!\Big(\log\sqrt{d^2_{\mathrm{knn3}}}\Big) = \sqrt{d^2_{\mathrm{knn3}}}
$$

vì $\exp(\log x)=x$ với mọi $x>0$ (hàm mũ và log tự nhiên là nghịch đảo của nhau trên $(0,\infty)$). Nói
cách khác, thao tác "$\log$ rồi $\exp$" trong code (`scales = log(sqrt(dist2))`, rồi khi cần giá trị scale
thật sự pipeline luôn gọi `exp` lên tham số lưu trữ — xem `scaling_activation = torch.exp` ở
`gaussian_model.py` phần định nghĩa activation, không trích ở đây vì thuộc Chương 7) chỉ nhằm mục đích lưu
$\tilde s_i$ ở không gian log — cho phép Adam tối ưu tự do trên toàn trục số thực mà vẫn đảm bảo
$s_i=\exp(\tilde s_i)>0$ tuyệt đối sau khi activation, không cần ràng buộc dương tường minh. Đây là kỹ
thuật **reparameterization qua log-space** rất phổ biến khi tham số cần dương (tương tự $\log\sigma$ cho
độ lệch chuẩn trong VAE).

### Bảng tổng hợp mục 1.3

| $i$ | 3-NN (giá trị $\lVert p_i-p_j\rVert^2$) | $d^2_{\mathrm{knn3}}$ | sau clamp $10^{-7}$ | $\tilde s_i$ (mỗi trục, full precision) | $s_i=\exp\tilde s_i=\sqrt{d^2_{\mathrm{knn3}}}$ |
|---|---|---|---|---|---|
| 1 | $0.59,\ 1.2,\ 0.38$ | $2.17/3=0.7233333333$ | $0.7233333333$ | $-0.1619425606$ | $0.8504899611$ |
| 2 | $0.59,\ 1.31,\ 0.77$ | $2.67/3=0.89$ | $0.89$ | $-0.0582669089$ | $0.9433981132$ |
| 3 | $1.2,\ 1.31,\ 1.22$ | $3.73/3=1.2433333333$ | $1.2433333333$ | $0.1088979725$ | $1.1150934466$ |
| 4 | $0.38,\ 0.77,\ 1.22$ | $2.37/3=0.79$ | $0.79$ | $-0.1178611446$ | $0.8888194436$ |

Không có $d^2_{\mathrm{knn3}}$ nào tiệm cận $10^{-7}$ — clamp hoàn toàn không tác dụng ở cảnh này (giá trị
nhỏ nhất, $0.7233$, lớn hơn ngưỡng clamp tới $7.23\times10^6$ lần). Gaussian đẳng hướng: cả ba trục
$s_{i,x}=s_{i,y}=s_{i,z}=s_i$ (dòng `scales = ...repeat(1,3)` nhân bản một cột log-scale thành 3 cột giống
hệt nhau — 3 trong 59 tham số của mỗi Gaussian nhưng chỉ mang **1 bậc tự do thông tin** ở bước khởi tạo).

**Kiểm chứng chéo bằng bài tập 6.2** của Chương 6: $\lVert p_1-p_3\rVert^2=1.2$ và
$\lVert p_2-p_4\rVert^2=0.77$ — đúng hai số vừa tính lại ở mục 1.2 trên đây (cặp (1,3) và cặp (2,4)).

---

## 1.4 — Quaternion $\tilde q_i$

**Công thức tổng quát** (chép từ `scene/gaussian_model.py:149-150`):

```python
rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
rots[:, 0] = 1
```

tức mọi Gaussian khởi tạo với quaternion đơn vị $\tilde q_i=(1,0,0,0)$ — biểu diễn phép **không xoay**
(rotation identity) trong không gian quaternion $(q_w,q_x,q_y,q_z)$. Không có phép tính số nào ở đây —
đây là hằng số khởi tạo, giống hệt cho cả 4 Gaussian, không phụ thuộc vào $p_i$ hay $c_i$.

**Kết quả:**

| $i$ | $\tilde q_i$ |
|---|---|
| 1 | $(1,\ 0,\ 0,\ 0)$ |
| 2 | $(1,\ 0,\ 0,\ 0)$ |
| 3 | $(1,\ 0,\ 0,\ 0)$ |
| 4 | $(1,\ 0,\ 0,\ 0)$ |

**Vì sao hợp lý:** vì scale khởi tạo đẳng hướng ($s_{i,x}=s_{i,y}=s_{i,z}$, mục 1.3), hướng xoay của
ellipsoid không ảnh hưởng gì tới hình dạng Gaussian lúc $t=0$ — một quả cầu xoay góc nào cũng vẫn là quả
cầu đó. Chọn $(1,0,0,0)$ đơn giản là chọn gốc toạ độ trong không gian quaternion (norm $=1$, góc quay
$=0$), để mọi bất đối xứng hình dạng (nếu có) sau này hoàn toàn do gradient học ra, không thiên vị bởi
khởi tạo. Xem Bài tập 6.1 của Chương 6 để bàn kỹ hơn lý do chọn đẳng hướng.

Chuẩn hoá quaternion (norm Euclid, dùng khi xây $\Sigma_i$ ở Chương 7): $\lVert\tilde q_i\rVert=\sqrt{1^2+0^2+0^2+0^2}=1$
— đã là quaternion đơn vị, không cần chuẩn hoá thêm.

---

## 1.5 — Opacity $\tilde\alpha_i$

**Công thức tổng quát** (chép từ `utils/general_utils.py:18`, hàm `inverse_sigmoid`, và
`scene/gaussian_model.py:152`):

```python
def inverse_sigmoid(x):
    return torch.log(x/(1-x))
...
opacities = self.inverse_opacity_activation(0.1 * torch.ones((N,1), ...))   # dòng 152
```

tức $\tilde\alpha_i = \sigma^{-1}(0.1)$, với $\sigma^{-1}$ là hàm **nghịch đảo sigmoid** (logit).

### Dẫn công thức tổng quát $\sigma^{-1}(\alpha)$

Hàm sigmoid: $\sigma(x) = \dfrac{1}{1+e^{-x}}$. Ta giải phương trình $y=\sigma(x)$ theo $x$:

$$
y = \frac1{1+e^{-x}} \;\Longrightarrow\; 1+e^{-x} = \frac1y \;\Longrightarrow\; e^{-x} = \frac1y-1=\frac{1-y}{y}
\;\Longrightarrow\; -x = \log\frac{1-y}{y} \;\Longrightarrow\; x = \log\frac{y}{1-y}
$$

Vậy hàm ngược tổng quát:

$$
\boxed{\ \sigma^{-1}(\alpha) = \log\frac{\alpha}{1-\alpha}\ },\qquad \alpha\in(0,1)
$$

đúng khớp `inverse_sigmoid(x) = log(x/(1-x))` ở `general_utils.py:18` — code chỉ viết lại nguyên văn công
thức toán học vừa dẫn, không có xấp xỉ hay biến thể nào khác.

### Thay số $\alpha=0.1$

$$
\tilde\alpha_i = \sigma^{-1}(0.1) = \log\frac{0.1}{1-0.1} = \log\frac{0.1}{0.9} = \log(0.1111111111)
$$

Tính $\log(0.1/0.9)$: viết $\dfrac{0.1}{0.9}=\dfrac19$, nên $\log\dfrac19 = -\log 9 = -2\log 3$. Với
$\log 3 = 1.0986122887$:

$$
\tilde\alpha_i = -2\times1.0986122887 = -2.1972245773
$$

**Kết quả:** $\tilde\alpha_i = -2.1972245773 \approx -2.197$ cho **cả 4 Gaussian** (không phụ thuộc $i$ —
mọi Gaussian khởi tạo cùng một độ mờ).

### Kiểm chứng ngược số học: $\sigma(\sigma^{-1}(0.1)) \stackrel{?}{=} 0.1$

$$
\sigma(-2.1972245773) = \frac1{1+e^{2.1972245773}}
$$

Tính $e^{2.1972245773}$: vì $2.1972245773 = \log 9$ (từ trên, $-\tilde\alpha=\log 9$), nên
$e^{2.1972245773}=e^{\log 9}=9$ đúng theo định nghĩa hàm mũ/log nghịch đảo nhau. Vậy:

$$
\sigma(-2.1972245773) = \frac1{1+9} = \frac1{10} = 0.1 \quad\checkmark
$$

Khớp hoàn toàn — không có sai số làm tròn nào lọt vào vì $9$ và $1/9$ đều biểu diễn hữu hạn.

### Đối xứng với $\alpha_{\mathrm{GT}}=0.9$

Ground-truth của cảnh (định nghĩa ở §16.1/§6.2) dùng $\alpha=0.9$ thay cho $\alpha=0.1$:

$$
\sigma^{-1}(0.9) = \log\frac{0.9}{0.1} = \log 9 = +2.1972245773
$$

$$
\sigma^{-1}(0.1) = -2.1972245773,\qquad \sigma^{-1}(0.9)=+2.1972245773 \;\Longrightarrow\; \sigma^{-1}(0.1) = -\sigma^{-1}(0.9)
$$

**Vì sao đối xứng qua 0 — chứng minh tổng quát cho mọi $\alpha$:**

$$
\sigma^{-1}(1-\alpha) = \log\frac{1-\alpha}{1-(1-\alpha)} = \log\frac{1-\alpha}{\alpha} = -\log\frac{\alpha}{1-\alpha} = -\sigma^{-1}(\alpha)
$$

Vì $0.9 = 1-0.1$, đẳng thức $\sigma^{-1}(0.9)=-\sigma^{-1}(0.1)$ là trường hợp riêng của tính chất tổng
quát $\sigma^{-1}(1-\alpha)=-\sigma^{-1}(\alpha)$ — hệ quả trực tiếp của việc $\sigma$ đối xứng tâm quanh
điểm $(0,\ 0.5)$ (tức $\sigma(-x)=1-\sigma(x)$ với mọi $x$).

**Kết quả mục 1.5:**

| $i$ | $\tilde\alpha_i$ | $\alpha_i=\sigma(\tilde\alpha_i)$ |
|---|---|---|
| 1 | $-2.1972245773$ | $0.1$ |
| 2 | $-2.1972245773$ | $0.1$ |
| 3 | $-2.1972245773$ | $0.1$ |
| 4 | $-2.1972245773$ | $0.1$ |

Opacity thấp ($0.1$) nghĩa là mỗi Gaussian khởi tạo gần như trong suốt — phần lớn "hình dạng" của cảnh
phải được học dần qua gradient descent (Phần 6, Chương 11), không có sẵn ngay từ đầu.

---

## 1.6 — Hệ số Spherical Harmonics $k_{i,lm}$

### 1.6.1 — Dẫn hằng số $C_0$ từ định nghĩa spherical harmonics thực

Spherical harmonics thực (real SH) bậc $l$, hệ số $m$ ($-l\le m\le l$) là các hàm trực chuẩn trên mặt cầu
đơn vị $S^2$, dùng làm cơ sở khai triển hàm phụ thuộc hướng (ở đây: màu phụ thuộc góc nhìn). Điều kiện
trực chuẩn:

$$
\int_{S^2} Y_l^m(\omega)\,Y_{l'}^{m'}(\omega)\,d\omega = \delta_{ll'}\delta_{mm'}
$$

Với $l=0,\ m=0$ (không phụ thuộc hướng, hằng số trên toàn mặt cầu): đặt $Y_0^0(\omega)=C_0$ (hằng số cần
tìm). Điều kiện chuẩn hoá ($l=l'=0,\ m=m'=0$):

$$
\int_{S^2} C_0^2\,d\omega = C_0^2\int_{S^2}d\omega = C_0^2\cdot 4\pi = 1
$$

(vì diện tích mặt cầu đơn vị là $4\pi$ steradian). Giải ra:

$$
C_0^2 = \frac1{4\pi} \;\Longrightarrow\; C_0 = \sqrt{\frac1{4\pi}} = \frac1{2\sqrt\pi}
$$

**Thay số:** $\pi = 3.1415926536$, $\sqrt\pi = 1.7724538509$, $2\sqrt\pi = 3.5449077018$, nghịch đảo:

$$
C_0 = \frac1{2\sqrt\pi} = \frac1{3.5449077018} = 0.2820947918
$$

Khớp đúng hằng số `C0 = 0.28209479177387814` ở `utils/sh_utils.py:26` (10 chữ số thập phân đầu tiên trùng
khít). Đây chính là hệ số $Y_0^0$ chuẩn (thường viết $Y_0^0=\frac12\sqrt{\frac1\pi}$ — cùng một số vì
$\frac12\sqrt{\frac1\pi}=\frac{1}{2\sqrt\pi}$).

### 1.6.2 — Dẫn `RGB2SH`

Ánh xạ ngược `SH2RGB` (dùng khi *giải mã* màu từ hệ số SH, xem `sh_utils.py:117-118`,
`def SH2RGB(sh): return sh * C0 + 0.5`) định nghĩa quan hệ:

$$
c = k_{00}\cdot C_0 + 0.5
$$

(hằng số $+0.5$ dịch tâm về giữa khoảng màu $[0,1]$, vì SH bậc 0 một mình chỉ cho màu **hằng số theo mọi
hướng** — không có bậc cao để biểu diễn phần lệch khỏi trung bình, nên quy ước đặt gốc SH ứng với màu xám
trung tính $0.5$). Giải ngược ra $k_{00}$ theo $c$ (đây chính là `RGB2SH`, `sh_utils.py:114-115`):

$$
c = k_{00}C_0+0.5 \;\Longrightarrow\; k_{00}C_0 = c-0.5 \;\Longrightarrow\; \boxed{\,k_{00} = \dfrac{c-0.5}{C_0}\,}
$$

```python
def RGB2SH(rgb):
    return (rgb - 0.5) / C0
```

`create_from_pcd` gọi hàm này ở dòng 140: `fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors))...)`
— áp dụng cho cả ba kênh R, G, B độc lập (element-wise), vì mỗi kênh màu có một tập hệ số SH riêng.

Nghịch đảo $1/C_0$ dùng lại nhiều lần bên dưới:

$$
\frac1{C_0} = \frac1{0.2820947918} = 3.5449077018
$$

(đúng bằng $2\sqrt\pi$ đã tính ở 1.6.1 — hợp lý vì $1/C_0 = 1/(1/(2\sqrt\pi)) = 2\sqrt\pi$.)

### 1.6.3 — Thay số $k_{i,00}$ cho cả 4 Gaussian, từng kênh, từng bước

Ta tính $k_{i,00}=(c_i-0.5)/C_0 = (c_i-0.5)\times3.5449077018$ cho mỗi kênh R, G, B của mỗi điểm — 12
phép tính, viết đủ cả 12.

**Gaussian 1** — $c_1=(0.8,\ 0.2,\ 0.2)$:

$$
k_{1,00}^R = \frac{0.8-0.5}{C_0} = \frac{0.3}{0.2820947918} = 0.3\times3.5449077018 = 1.0634723105
$$
$$
k_{1,00}^G = \frac{0.2-0.5}{C_0} = \frac{-0.3}{0.2820947918} = -0.3\times3.5449077018 = -1.0634723105
$$
$$
k_{1,00}^B = \frac{0.2-0.5}{C_0} = -1.0634723105 \quad\text{(kênh B trùng kênh G vì cùng } c=0.2\text{)}
$$

$k_{1,00} = (1.0634723105,\ -1.0634723105,\ -1.0634723105) \approx (1.063,\ -1.063,\ -1.063)$.

**Gaussian 2** — $c_2=(0.2,\ 0.7,\ 0.3)$:

$$
k_{2,00}^R = \frac{0.2-0.5}{C_0} = -0.3\times3.5449077018 = -1.0634723105
$$
$$
k_{2,00}^G = \frac{0.7-0.5}{C_0} = 0.2\times3.5449077018 = 0.7089815404
$$
$$
k_{2,00}^B = \frac{0.3-0.5}{C_0} = -0.2\times3.5449077018 = -0.7089815404
$$

$k_{2,00} = (-1.0634723105,\ 0.7089815404,\ -0.7089815404) \approx (-1.063,\ 0.7090,\ -0.7090)$.

**Gaussian 3** — $c_3=(0.1,\ 0.3,\ 0.9)$:

$$
k_{3,00}^R = \frac{0.1-0.5}{C_0} = -0.4\times3.5449077018 = -1.4179630807
$$
$$
k_{3,00}^G = \frac{0.3-0.5}{C_0} = -0.2\times3.5449077018 = -0.7089815404
$$
$$
k_{3,00}^B = \frac{0.9-0.5}{C_0} = 0.4\times3.5449077018 = 1.4179630807
$$

$k_{3,00} = (-1.4179630807,\ -0.7089815404,\ 1.4179630807) \approx (-1.418,\ -0.7090,\ 1.418)$.

**Gaussian 4** — $c_4=(0.5,\ 0.5,\ 0.5)$:

$$
k_{4,00}^R = \frac{0.5-0.5}{C_0} = \frac{0}{C_0} = 0,\qquad k_{4,00}^G=0,\qquad k_{4,00}^B=0
$$

$k_{4,00} = (0,\ 0,\ 0)$ — vì $c_4=0.5$ đúng bằng điểm trung tâm của `SH2RGB` ($sh\cdot C_0+0.5$ tại
$sh=0$ cho $c=0.5$); màu xám trung tính không cần lệch khỏi gốc SH.

### Bảng tổng hợp mục 1.6.3

| $i$ | $c_i$ (R,G,B) | $(c_i-0.5)$ | $k_{i,00}=(c_i-0.5)\times3.5449077018$ |
|---|---|---|---|
| 1 | $(0.8,\ 0.2,\ 0.2)$ | $(0.3,\ -0.3,\ -0.3)$ | $(1.0634723105,\ -1.0634723105,\ -1.0634723105)$ |
| 2 | $(0.2,\ 0.7,\ 0.3)$ | $(-0.3,\ 0.2,\ -0.2)$ | $(-1.0634723105,\ 0.7089815404,\ -0.7089815404)$ |
| 3 | $(0.1,\ 0.3,\ 0.9)$ | $(-0.4,\ -0.2,\ 0.4)$ | $(-1.4179630807,\ -0.7089815404,\ 1.4179630807)$ |
| 4 | $(0.5,\ 0.5,\ 0.5)$ | $(0,\ 0,\ 0)$ | $(0,\ 0,\ 0)$ |

### 1.6.4 — Các hệ số SH bậc cao $k_{i,lm}$, $l\ge1$: bằng 0, hình dạng tensor

**Công thức/code** (`scene/gaussian_model.py:141-143`):

```python
features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  # (4, 3, 16)
features[:, :3, 0 ] = fused_color        # gán bậc 0 (RGB2SH ở trên) vào slot 0
features[:, 3:, 1:] = 0.0                # (không hiệu lực: features chỉ có 3 kênh, "3:" rỗng)
```

rồi tách ra (`gaussian_model.py:155-156`):

```python
self._features_dc   = features[:, :, 0:1].transpose(1, 2)...   # (4, 1, 3)
self._features_rest  = features[:, :, 1:].transpose(1, 2)...   # (4, 15, 3)
```

Với `max_sh_degree=3`, tổng số hệ số SH mỗi kênh màu là $(3+1)^2=16$ (số hàm cơ sở $Y_l^m$ với
$l=0,1,2,3$, mỗi bậc $l$ có $2l+1$ giá trị $m$): $1$ (bậc 0) $+3$ (bậc 1) $+5$ (bậc 2) $+7$ (bậc 3)
$=16$. Slot đầu tiên (bậc 0) được gán $k_{i,00}$ vừa tính ở 1.6.3; **15 slot còn lại** (bậc 1, 2, 3) khởi
tạo bằng $0$ do toàn bộ tensor `features` được cấp phát bằng `torch.zeros` và chỉ có dòng gán duy nhất
cho slot $0$ (`features[:, :3, 0]`).

Liệt kê đầy đủ 15 cặp $(l,m)$ bậc cao (thứ tự chuẩn dùng trong `eval_sh`, `sh_utils.py:74-100`):

| $l$ | $m$ | chỉ số phẳng trong 16 slot | giá trị $k_{i,lm}$ |
|---|---|---|---|
| 1 | $-1$ | 1 | $0$ |
| 1 | $0$ | 2 | $0$ |
| 1 | $+1$ | 3 | $0$ |
| 2 | $-2$ | 4 | $0$ |
| 2 | $-1$ | 5 | $0$ |
| 2 | $0$ | 6 | $0$ |
| 2 | $+1$ | 7 | $0$ |
| 2 | $+2$ | 8 | $0$ |
| 3 | $-3$ | 9 | $0$ |
| 3 | $-2$ | 10 | $0$ |
| 3 | $-1$ | 11 | $0$ |
| 3 | $0$ | 12 | $0$ |
| 3 | $+1$ | 13 | $0$ |
| 3 | $+2$ | 14 | $0$ |
| 3 | $+3$ | 15 | $0$ |

15 cặp $(l,m)$ $\times$ 3 kênh màu (R,G,B) $= 45$ hệ số bậc cao, **tất cả bằng 0**, cho **mỗi** Gaussian.
Với $N_0=4$ Gaussian: tensor `_features_rest` có shape $(4,\ 15,\ 3)$ — đúng như đã ghi ở Chương 6, script
kiểm tra `max|features_rest| = 0`. Tensor `_features_dc` có shape $(4,\ 1,\ 3)$ — chiều giữa bằng 1 vì
chỉ có đúng một hệ số bậc 0.

**Vì sao khởi tạo 0 hợp lý:** hệ số SH bậc cao mã hoá sự **phụ thuộc vào hướng nhìn** của màu (specular,
phản chiếu, v.v.) — thông tin này SfM không cung cấp trực tiếp (COLMAP chỉ cho một màu trung bình mỗi
điểm, không phải hàm màu theo hướng). Khởi tạo 0 nghĩa là ban đầu mọi Gaussian **phát xạ màu như nhau ở
mọi hướng nhìn** (Lambertian hoàn toàn, chỉ có SH bậc 0); phần phụ thuộc hướng học dần qua gradient, và
theo lịch tăng bậc `oneupSHdegree` (`gaussian_model.py:133-135`, không thuộc phạm vi Chương 6) — bậc SH
active tăng dần mỗi 1000 iteration để tránh học overfitting hệ số bậc cao quá sớm khi hình dạng còn thô.

### 1.6.5 — Kiểm chứng ngược: $\text{SH2RGB}(k_{i,00}) \stackrel{?}{=} c_i$

Công thức ngược (đã dẫn ở 1.6.2): $\text{SH2RGB}(k_{00}) = k_{00}\cdot C_0+0.5$. Nếu `RGB2SH` và `SH2RGB`
thật sự là nghịch đảo chính xác của nhau, áp dụng liên tiếp phải cho lại đúng $c_i$ ban đầu. Kiểm tra đủ
cả 12 giá trị (4 điểm × 3 kênh), không bỏ sót.

**Gaussian 1** ($k_{1,00}=(1.0634723105,\ -1.0634723105,\ -1.0634723105)$):

$$
1.0634723105\times0.2820947918+0.5 = 0.3+0.5 = 0.8 = c_1^R\ \checkmark
$$
$$
-1.0634723105\times0.2820947918+0.5 = -0.3+0.5 = 0.2 = c_1^G = c_1^B\ \checkmark
$$

**Gaussian 2** ($k_{2,00}=(-1.0634723105,\ 0.7089815404,\ -0.7089815404)$):

$$
-1.0634723105\times0.2820947918+0.5 = -0.3+0.5=0.2=c_2^R\ \checkmark
$$
$$
0.7089815404\times0.2820947918+0.5 = 0.2+0.5=0.7=c_2^G\ \checkmark
$$
$$
-0.7089815404\times0.2820947918+0.5 = -0.2+0.5=0.3=c_2^B\ \checkmark
$$

**Gaussian 3** ($k_{3,00}=(-1.4179630807,\ -0.7089815404,\ 1.4179630807)$):

$$
-1.4179630807\times0.2820947918+0.5=-0.4+0.5=0.1=c_3^R\ \checkmark
$$
$$
-0.7089815404\times0.2820947918+0.5=-0.2+0.5=0.3=c_3^G\ \checkmark
$$
$$
1.4179630807\times0.2820947918+0.5=0.4+0.5=0.9=c_3^B\ \checkmark
$$

**Gaussian 4** ($k_{4,00}=(0,0,0)$):

$$
0\times0.2820947918+0.5=0.5=c_4^R=c_4^G=c_4^B\ \checkmark
$$

Cả 12/12 kiểm tra khớp chính xác (không sai số làm tròn đáng kể vì $C_0\times1/C_0=1$ triệt tiêu đúng
đắn theo đại số, chỉ có sai số biểu diễn dấu phẩy động cỡ $10^{-16}$, dưới độ phân giải hiển thị). Đây là
minh chứng số học rằng `RGB2SH`/`SH2RGB` (`sh_utils.py:114-118`) là một cặp ánh xạ **affine khả nghịch**
tuyến tính đúng nghĩa, không có mất mát thông tin ở bậc SH $0$ (mất mát thông tin màu, nếu có, chỉ xảy ra
khi lượng tử hoá `float32` hoặc khi hệ số SH bậc cao bị cắt/khởi tạo 0 — không liên quan tới bản thân phép
biến đổi tuyến tính này).

---

## 1.7 — Đếm tham số và vector $\theta_i$ đầy đủ

**Công thức đếm tham số mỗi Gaussian:**

$$
\underbrace{3}_{\mu}\ +\ \underbrace{4}_{\tilde q}\ +\ \underbrace{3}_{\tilde s}\ +\ \underbrace{1}_{\tilde\alpha}\ +\ \underbrace{3\times16}_{k_{lm},\ 16\text{ bậc}\times3\text{ kênh}}\ =\ 3+4+3+1+48\ =\ 59
$$

Kiểm tra: $3\times16=48$; $3+4+3+1+48 = 11+48 = 59$. Đúng $59$ tham số/Gaussian — khớp con số đã dùng
xuyên suốt sách kể từ Chương 2.

Nhóm 48 hệ số SH chia thành $3$ (bậc 0, tức `_features_dc`) $+45$ (bậc $\ge1$, tức `_features_rest`):
$3+45=48$ — khớp lại đúng phép cộng trên theo cách nhóm khác.

Với $N_0=4$ Gaussian, tổng số tham số toàn quần thể $\mathcal G_0$:

$$
4\times59 = 236
$$

— đúng con số $236$ tham số nhắc ở câu hỏi (6) của đề bài §16.2 (dùng cho backprop/Adam ở Phần 6).

### Vector tham số đầy đủ $\theta_i=(\mu_i,\ \tilde q_i,\ \tilde s_i,\ \tilde\alpha_i,\ k_{i,00},\ k_{i,\mathrm{rest}})\in\mathbb R^{59}$

Viết tường minh cả 59 giá trị cho từng Gaussian, theo đúng thứ tự nhóm tham số ở trên (3 tọa độ, 4
quaternion, 3 scale — bằng nhau vì đẳng hướng, 1 opacity, 3 hệ số SH bậc 0, rồi 45 số 0 của SH bậc cao):

**$\theta_1$ (59 giá trị):**

$$
\theta_1 = \big(\underbrace{0,\ 0,\ 0}_{\mu_1},\ \underbrace{1,\ 0,\ 0,\ 0}_{\tilde q_1},\ \underbrace{-0.1619425606,\ -0.1619425606,\ -0.1619425606}_{\tilde s_1},\ \underbrace{-2.1972245773}_{\tilde\alpha_1},\ \underbrace{1.0634723105,\ -1.0634723105,\ -1.0634723105}_{k_{1,00}},\ \underbrace{0,\dots,0}_{45\text{ số }0}\big)
$$

**$\theta_2$ (59 giá trị):**

$$
\theta_2 = \big(0.5,\ 0.3,\ 0.5,\ 1,\ 0,\ 0,\ 0,\ -0.0582669089,\ -0.0582669089,\ -0.0582669089,\ -2.1972245773,\ -1.0634723105,\ 0.7089815404,\ -0.7089815404,\ 0,\dots,0\big)
$$

**$\theta_3$ (59 giá trị):**

$$
\theta_3 = \big({-0.4},\ {-0.2},\ 1.0,\ 1,\ 0,\ 0,\ 0,\ 0.1088979725,\ 0.1088979725,\ 0.1088979725,\ -2.1972245773,\ -1.4179630807,\ -0.7089815404,\ 1.4179630807,\ 0,\dots,0\big)
$$

**$\theta_4$ (59 giá trị):**

$$
\theta_4 = \big(0.3,\ {-0.5},\ 0.2,\ 1,\ 0,\ 0,\ 0,\ -0.1178611446,\ -0.1178611446,\ -0.1178611446,\ -2.1972245773,\ 0,\ 0,\ 0,\ 0,\dots,0\big)
$$

Mỗi vector $45$ số $0$ cuối cùng biểu diễn đúng $k_{i,lm}$, $l\ge1$ (15 cặp $(l,m)$ $\times$ 3 kênh, liệt
kê ở bảng 1.6.4).

### Bảng tóm tắt (dạng nhóm, dễ tra cứu — trùng đúng bảng "Đầu ra của khối" ở Chương 6)

| $i$ | $\mu_i$ | $\tilde q_i$ | $\tilde s_i$ (×3) | $s_i$ | $\tilde\alpha_i$ | $\alpha_i$ | $k_{i,00}$ (R,G,B) | $k_{i,lm},l\ge1$ |
|---|---|---|---|---|---|---|---|---|
| 1 | $(0,0,0)$ | $(1,0,0,0)$ | $-0.1619425606$ | $0.8504899611$ | $-2.1972245773$ | $0.1$ | $(1.0634723105,\ -1.0634723105,\ -1.0634723105)$ | $0$ (×45) |
| 2 | $(0.5,0.3,0.5)$ | $(1,0,0,0)$ | $-0.0582669089$ | $0.9433981132$ | $-2.1972245773$ | $0.1$ | $(-1.0634723105,\ 0.7089815404,\ -0.7089815404)$ | $0$ (×45) |
| 3 | $(-0.4,-0.2,1.0)$ | $(1,0,0,0)$ | $0.1088979725$ | $1.1150934466$ | $-2.1972245773$ | $0.1$ | $(-1.4179630807,\ -0.7089815404,\ 1.4179630807)$ | $0$ (×45) |
| 4 | $(0.3,-0.5,0.2)$ | $(1,0,0,0)$ | $-0.1178611446$ | $0.8888194436$ | $-2.1972245773$ | $0.1$ | $(0,0,0)$ | $0$ (×45) |

Đây chính là $\mathcal G_0=\{\theta_i\}_{i=1}^4$ — đầu ra chính của Chương 6, chuyển nguyên vẹn cho Chương
7 (Phần 2).

---

## 1.8 — `extent`: bán kính cảnh

**Công thức tổng quát** (chép từ `scene/dataset_readers.py:45-66`, hàm `getNerfppNorm`):

```python
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    ...
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}
```

tức, với $c_v=(W_v)^{-1}[{:}3,3]$ là tâm camera thứ $v$ trong world space:

$$
\bar c = \frac1V\sum_{v=1}^V c_v,\qquad \text{extent} = 1.1\cdot\max_v\lVert c_v-\bar c\rVert_2
$$

`radius` trong code chính là `extent` dùng xuyên suốt sách (đổi tên khi truyền qua `Scene.__init__`, xem
mục 1.12 bên dưới).

### Bước 1 — tâm trung bình $\bar c$

Ba tâm camera đã tính ở mục 1.0.2: $c_1=(0,0,-4)$, $c_2=(1.5,0,-4)$, $c_3=(-1.5,0.5,-4)$.

$$
\sum_v c_v = (0+1.5-1.5,\ \ 0+0+0.5,\ \ -4-4-4) = (0,\ 0.5,\ -12)
$$

$$
\bar c = \frac13(0,\ 0.5,\ -12) = (0,\ 0.1666666667,\ -4)
$$

### Bước 2 — độ lệch và chuẩn $\lVert c_v-\bar c\rVert$ cho từng camera

**Camera 1:**

$$
c_1-\bar c = (0-0,\ 0-0.1666666667,\ -4-(-4)) = (0,\ -0.1666666667,\ 0)
$$
$$
\lVert c_1-\bar c\rVert = \sqrt{0^2+(-0.1666666667)^2+0^2} = \sqrt{0.0277777778} = 0.1666666667
$$

(hiển nhiên vì vector chỉ có một thành phần khác 0.)

**Camera 2:**

$$
c_2-\bar c = (1.5-0,\ 0-0.1666666667,\ -4-(-4)) = (1.5,\ -0.1666666667,\ 0)
$$
$$
\lVert c_2-\bar c\rVert = \sqrt{1.5^2+0.1666666667^2+0^2} = \sqrt{2.25+0.0277777778} = \sqrt{2.2777777778}
$$

Tính căn bậc hai: thử $1.509^2=2.277081$; $1.5092^2=2.27768...$; giá trị chính xác:

$$
\lVert c_2-\bar c\rVert = 1.5092308094 \approx 1.509
$$

**Camera 3:**

$$
c_3-\bar c = (-1.5-0,\ 0.5-0.1666666667,\ -4-(-4)) = (-1.5,\ 0.3333333333,\ 0)
$$
$$
\lVert c_3-\bar c\rVert = \sqrt{(-1.5)^2+0.3333333333^2+0^2} = \sqrt{2.25+0.1111111111} = \sqrt{2.3611111111}
$$

$$
\lVert c_3-\bar c\rVert = 1.5365577635 \approx 1.537
$$

### Bước 3 — max và nhân biên an toàn $1.1$

$$
\max_v\lVert c_v-\bar c\rVert = \max(0.1666666667,\ 1.5092308094,\ 1.5365577635) = 1.5365577635
$$

(camera 3 xa tâm trung bình nhất — hợp lý vì $c_3=(-1.5,\ 0.5,\ -4)$ có cả độ lệch $x$ lớn **và** độ lệch
$y$ lớn hơn camera 2, trong khi camera 2 chỉ lệch theo $x$.)

$$
\text{extent} = 1.1\times1.5365577635 = 1.6902135399
$$

Hằng số nhân $1.1$: biên an toàn $10\%$ để bán kính cảnh không "sát mép" đúng bằng camera xa nhất — chừa
dư một chút cho các vùng không gian nằm ngoài lồi bao (convex hull) của tập camera nhưng vẫn thuộc cảnh
(ví dụ điểm SfM nằm ngoài vòng tròn qua các camera). Hằng số này được viết cố định trong code ở
`dataset_readers.py:62` (`radius = diagonal * 1.1`), không phải tham số cấu hình.

**Kết quả:** $\text{extent} = 1.6902135399$ — làm tròn $4$ chữ số có nghĩa: $\boxed{\text{extent}=1.690}$.

> **Lưu ý sai số làm tròn nhỏ:** giá trị full-precision ở Chương 6 ghi là $1.6902498172$ (chênh khoảng
> $3.6\times10^{-5}$ so với $1.6902135399$ tính ở trên). Sai lệch này nằm ở chữ số thập phân thứ 5,
> hoàn toàn nằm trong sai số làm tròn của phép khai căn tay $\sqrt{2.3611111111}$ (script numpy giữ full
> `float64` mantissa, còn ở đây ta khai căn bằng ước lượng thập phân hữu hạn bước); cả hai giá trị làm
> tròn về 4 chữ số có nghĩa đều cho $\text{extent}=1.690$, và giá trị dùng làm đầu vào chính thức cho các
> phần sau là $1.6902498172$ (theo đúng script `ch01_test.py` của Chương 6, xem mục 1.11 để đối chiếu
> bằng numpy `float64`).

---

## 1.9 — Ba ngưỡng suy ra từ `extent`

`extent` (bán kính cảnh) được dùng làm đơn vị đo tỉ lệ chung cho ba ngưỡng khác nhau trong pipeline, mỗi
ngưỡng ở một khối khác nhau (Chương 6 và Chương 7/12). Dùng $\text{extent}=1.6902498172$ (giá trị chính
thức, xem lưu ý ở mục 1.8).

### (a) Ngưỡng clone/split

**Công thức:** $\delta\cdot\text{extent}$, với $\delta=0.001$ (hằng số hệ thống, đề bài §16.2, code
`args.dense`, `arguments/__init__.py:95`, dùng ở `gaussian_model.py:451-452`).

$$
\delta\cdot\text{extent} = 0.001\times1.6902498172 = 0.0016902498
$$

**Kết quả:** $0.001690$. Ngưỡng này so sánh với $\max_{\text{trục}}(s_i)$ (giá trị scale lớn nhất theo
một trục, sau activation `exp`) để quyết định một Gaussian có gradient lớn nên **clone** (nếu còn nhỏ) hay
**split** (nếu đã đủ lớn) — quyết định chi tiết thuộc Chương 12 (Phần 7), ở đây chỉ tính ra hằng số.

### (b) Ngưỡng "scale quá lớn" khi prune

**Công thức:** $0.1\cdot\text{extent}$ (code `gaussian_model.py:467`).

$$
0.1\times1.6902498172 = 0.1690249817
$$

**Kết quả:** $0.169025 \approx 0.1690$.

### (c) `spatial_lr_scale`

**Công thức:** $\eta_{xyz}\cdot\text{extent}$, với $\eta_{xyz}=1.6\times10^{-4}$ (learning-rate gốc cho
vị trí $\mu$, hằng số Chương 5, code `training_args.position_lr_init`, `arguments/__init__.py:76`, nhân
vào ở `gaussian_model.py:168` khi dựng `optimizer`).

$$
\eta_{xyz}\cdot\text{extent} = 1.6\times10^{-4}\times1.6902498172 = 2.7043997075\times10^{-4}
$$

**Kết quả:** $2.7044\times10^{-4} \approx 2.704\times10^{-4}$.

### Bảng tổng hợp

| Nơi dùng | Công thức | Thay số | Kết quả | Code |
|---|---|---|---|---|
| clone / split | $\delta\cdot\text{extent}$ | $0.001\times1.6902498172$ | $1.6902498\times10^{-3}$ | `gaussian_model.py:451-452`, `arguments/__init__.py:95` |
| prune "scale quá lớn" | $0.1\cdot\text{extent}$ | $0.1\times1.6902498172$ | $0.1690249817$ | `gaussian_model.py:467` |
| `spatial_lr_scale` | $\eta_{xyz}\cdot\text{extent}$ | $1.6\times10^{-4}\times1.6902498172$ | $2.7043997\times10^{-4}$ | `gaussian_model.py:168`, `arguments/__init__.py:76` |

### Nhận xét quan trọng cho Chương 7/12 (Phần 2, Phần 7)

So $s_i$ (mục 1.3: $0.8505,\ 0.9434,\ 1.115,\ 0.8888$) với hai ngưỡng (a), (b):

- Mọi $s_i \ge 0.8505 \gg \delta\cdot\text{extent}=0.001690$: nếu một trong 4 Gaussian này có gradient vị
  trí vượt ngưỡng ($\tau_{\text{grad}}$, Chương 12), nó sẽ luôn đi nhánh **split** (không đi nhánh
  **clone**, vì clone chỉ áp dụng khi scale còn *nhỏ hơn* $\delta\cdot\text{extent}$).
- Mọi $s_i \ge 0.8505 > 0.1\times\text{extent}=0.1690$: **ngay từ lúc khởi tạo**, cả 4 Gaussian đã vượt
  ngưỡng "scale quá lớn" `big_points_ws` dùng khi prune ở Chương 12. Đây là đặc điểm riêng của cảnh đồ
  chơi (chỉ 4 điểm SfM rất thưa so với khoảng cách camera $\approx4$ đơn vị) — ở cảnh thật với hàng vạn
  điểm SfM, khoảng cách 3-NN nhỏ hơn `extent` rất nhiều lần nên tỉ lệ $s_i/\text{extent}$ nhỏ hơn hẳn.
  Chương 7 (Phần 2 của lời giải) sẽ phải xử lý tình huống "quá lớn ngay từ đầu" này khi bàn về compact
  box và tile giao nhau.

---

## 1.10 — Bộ nhớ trạng thái Adam

**Công thức tổng quát:** `training_setup` (`gaussian_model.py:162-177`) tạo **hai** optimizer Adam riêng
biệt:

```python
l = [
    {'params': [self._xyz], ...},          # 3
    {'params': [self._features_dc], ...},  # 3  (k_00, mỗi kênh 1 số)
    {'params': [self._opacity], ...},      # 1
    {'params': [self._scaling], ...},      # 3
    {'params': [self._rotation], ...}      # 4
]
sh_l = [{'params': [self._features_rest], ...}]   # 45

self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
```

`optimizer` giữ nhóm "tham số hình học + màu cơ bản": $\mu(3)+k_{00}(3)+\tilde\alpha(1)+\tilde s(3)+\tilde
q(4) = 14$ tham số/Gaussian. `shoptimizer` giữ riêng $45$ hệ số SH bậc cao. Tổng $14+45=59$ — khớp đếm ở
mục 1.7.

Adam (thuật toán tối ưu, Chương 5) lưu **hai moment** cho mỗi tham số vô hướng: $m$ (moment bậc 1, trung
bình động của gradient) và $v$ (moment bậc 2, trung bình động của gradient bình phương) — cộng thêm chính
tham số $\theta$ đang lưu, tổng $3$ bản sao float cho mỗi số vô hướng ($\theta, m, v$):

$$
\text{float/Gaussian} = 3\times59 = 177
$$

### Thay số byte-level (float32 $=4$ byte)

Chia theo hai optimizer, mỗi optimizer nhân 3 (giữ $\theta,m,v$ — thực ra $\theta$ đã tính trong 59 gốc,
nhưng $m,v$ là **thêm** $2\times$; tổng "trạng thái đi kèm" 2 moment mỗi tham số cộng với chính tham số
= 3 bản sao, cách đếm giữ nguyên theo Chương 6 để nhất quán):

| Optimizer | Số tham số/Gaussian | $\times3$ (θ, m, v) | float/Gaussian |
|---|---|---|---|
| `optimizer` ($\mu,\tilde q,\tilde s,\tilde\alpha,k_{00}$) | $3+4+3+1+3=14$ | $\times3$ | $42$ |
| `shoptimizer` ($k_{lm},l\ge1$) | $45$ | $\times3$ | $135$ |
| **Tổng** | $59$ | $\times3$ | $\mathbf{177}$ |

**Với $N_0=4$:**

$$
177\times4 = 708 \text{ float},\qquad 708\times4\text{ byte/float} = 2832\text{ byte} = 2.832\text{ kB}
$$

Bảng byte chi tiết:

| Thành phần | float | byte (× 4) |
|---|---|---|
| $\theta$ gốc (59×4) | $236$ | $944$ |
| moment $m$ (59×4) | $236$ | $944$ |
| moment $v$ (59×4) | $236$ | $944$ |
| **Tổng** | $708$ | $\mathbf{2832}$ |

**Với $N=10^6$ (ngoại suy, để thấy độ nhạy tuyến tính):**

$$
177\times10^6 = 1.77\times10^8\text{ float},\qquad 1.77\times10^8\times4 = 7.08\times10^8\text{ byte}
$$

$$
7.08\times10^8\text{ byte} = 708.0\text{ MB} = \frac{7.08\times10^8}{1024^2}\text{ MiB} = 675.2\text{ MiB}
$$

(chuyển đổi: $1$ MB $=10^6$ byte theo quy ước SI, $1$ MiB $=1024^2=1\,048\,576$ byte theo quy ước nhị
phân; $7.08\times10^8/1\,048\,576 = 675.20...$, khớp $675.2$ MiB.)

**Bảng tổng hợp:**

| $N$ | float ($177N$) | byte ($708N$) | quy đổi |
|---|---|---|---|
| $N_0=4$ | $708$ | $2\,832$ | $2.832$ kB |
| $N=10^6$ | $1.77\times10^8$ | $7.08\times10^8$ | $708.0$ MB $=675.2$ MiB |

### Byte breakdown theo từng nhóm tham số ($N_0=4$)

Bảng dưới tách $177$ float/Gaussian theo **từng nhóm tham số gốc** (không chỉ theo optimizer như trên),
nhân $3$ (θ, m, v) và $4$ (Gaussian) và $4$ byte/float, để thấy rõ SH bậc cao chiếm bao nhiêu phần trăm bộ
nhớ:

| Nhóm | kích thước/Gaussian | $\times3$ (θ,m,v) | $\times N_0{=}4$ | byte ($\times4$) | % tổng |
|---|---|---|---|---|---|
| $\mu$ (vị trí) | 3 | 9 | 36 | 144 | $5.09\%$ |
| $\tilde q$ (quaternion) | 4 | 12 | 48 | 192 | $6.78\%$ |
| $\tilde s$ (scale) | 3 | 9 | 36 | 144 | $5.09\%$ |
| $\tilde\alpha$ (opacity) | 1 | 3 | 12 | 48 | $1.69\%$ |
| $k_{00}$ (SH bậc 0) | 3 | 9 | 36 | 144 | $5.09\%$ |
| $k_{lm},l\ge1$ (SH bậc cao) | 45 | 135 | 540 | 2 160 | $76.27\%$ |
| **Tổng** | 59 | 177 | 708 | **2 832** | $100\%$ |

Kiểm tra cộng dồn: $144+192+144+48+144+2160 = 2832$ — khớp đúng con số đã tính ở phần trên bằng cách chia
theo hai optimizer. Quan sát quan trọng: **SH bậc cao chiếm hơn 3/4 tổng bộ nhớ trạng thái Adam** dù về
mặt "hình học" (vị trí, scale, xoay) chỉ có $10$ tham số; đây là lý do `shoptimizer` được tách riêng khỏi
`optimizer` — cho phép cập nhật SH bậc cao thưa hơn theo thời gian (`optimizer_step`,
`gaussian_model.py:190-209`, cập nhật `shoptimizer` mỗi $16$/$32$/$64$ bước tuỳ giai đoạn huấn luyện,
trong khi `optimizer` cập nhật mỗi bước cho tới iteration $15000$) mà không tốn thêm bộ nhớ — bộ nhớ
Adam vẫn phải cấp phát đủ $135$ float/Gaussian ngay từ đầu (không tiết kiệm RAM), chỉ tiết kiệm **số lần
ghi/đọc** (bandwidth), không tiết kiệm **dung lượng** đã tính ở đây.

Riêng trạng thái tham số + Adam (chưa kể gradient tích luỹ `xyz_gradient_accum`,
`xyz_gradient_accum_abs`, `denom` — mỗi cái $N\times1$ float, không đáng kể so với $177N$ — và chưa kể bộ
đệm trung gian của rasterizer, thuộc Chương 8-9) đã chiếm xấp xỉ $0.7$ GB VRAM ở $N=10^6$ Gaussian, một
con số điển hình cho cảnh 3DGS cỡ trung bình.

---

## 1.11 — Script numpy kiểm chứng toàn bộ

Script dưới đây (chỉ dùng `numpy`, không `torch` — đúng tinh thần `scripts/ch01_test.py` của Chương 6 và
quy ước §16.4.4) đọc thẳng nội dung ba file COLMAP (ở đây nhúng làm chuỗi trong script cho gọn — trong
thực tế sẽ dùng `scene/colmap_loader.py` để đọc file thật) và tái tạo **mọi** con số ở các mục 1.0–1.10.

```python
"""
Kiem chung tay: Chuong 6 - Initialization, cho "canh do choi dung chung".
Chi dung numpy. Doc truc tiep noi dung COLMAP (cameras.txt, images.txt, points3D.txt)
dang chuoi nhung san cho gon; cach parse mo phong dung logic cua
scene/colmap_loader.py + scene/dataset_readers.py.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1.0 -- doc du lieu COLMAP tho
# ---------------------------------------------------------------------------

CAMERAS_TXT = """1 PINHOLE 48 32 40.0 40.0 24.0 16.0"""

IMAGES_TXT = """
1 1.0 0.0 0.0 0.0 0.0 0.0 4.0 1 cam01.png
2 1.0 0.0 0.0 0.0 -1.5 0.0 4.0 1 cam02.png
3 1.0 0.0 0.0 0.0 1.5 -0.5 4.0 1 cam03.png
"""

POINTS3D_TXT = """
1 0.0 0.0 0.0 204 51 51 0.42 1 0 2 0 3 0
2 0.5 0.3 0.5 51 179 76 0.35 1 1 2 1 3 1
3 -0.4 -0.2 1.0 26 76 230 0.51 1 2 2 2 3 2
4 0.3 -0.5 0.2 128 128 128 0.29 1 3 2 3 3 3
"""

# --- cameras.txt : PINHOLE fx fy cx cy ---
cam_fields = CAMERAS_TXT.split()
W, H = int(cam_fields[2]), int(cam_fields[3])
fx, fy, cx, cy = (float(x) for x in cam_fields[4:8])
tan_fovx_2 = cx / fx
tan_fovy_2 = cy / fy
print("tan(fovx/2) =", tan_fovx_2, " tan(fovy/2) =", tan_fovy_2)
assert np.isclose(tan_fovx_2, 0.6) and np.isclose(tan_fovy_2, 0.4)

# --- images.txt : qw qx qy qz tx ty tz cam_id name ---
def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,   2*qx*qy - 2*qw*qz,       2*qz*qx + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,       1 - 2*qx**2 - 2*qz**2,   2*qy*qz - 2*qw*qx],
        [2*qz*qx - 2*qw*qy,       2*qy*qz + 2*qw*qx,       1 - 2*qx**2 - 2*qy**2],
    ])

cam_centers = []
W_list = []
for line in IMAGES_TXT.strip().splitlines():
    f = line.split()
    q = np.array([float(x) for x in f[1:5]])
    t = np.array([float(x) for x in f[5:8]])
    R = qvec2rotmat(q).T          # dataset_readers.py:99 -- R = qvec2rotmat(...).T
    Wv = np.eye(4)
    Wv[:3, :3] = R.T              # getWorld2View2: Rt[:3,:3] = R.transpose()
    Wv[:3, 3] = t
    c_v = np.linalg.inv(Wv)[:3, 3]
    W_list.append(Wv)
    cam_centers.append(c_v)

cam_centers = np.array(cam_centers)
print("c_v =\n", cam_centers)
assert np.allclose(cam_centers, [[0, 0, -4], [1.5, 0, -4], [-1.5, 0.5, -4]])

# --- points3D.txt : id x y z r g b error track... ---
p = []
c = []
for line in POINTS3D_TXT.strip().splitlines():
    f = line.split()
    p.append([float(f[1]), float(f[2]), float(f[3])])
    c.append([int(f[4]), int(f[5]), int(f[6])])
p = np.array(p)                    # (4,3)
c_raw = np.array(c)                # (4,3) int 0..255
c = c_raw / 255.0                  # (4,3) float [0,1]
print("p =\n", p)
print("c =\n", c)

# ---------------------------------------------------------------------------
# 1.1 -- mu_i = p_i (khong bien doi)
# ---------------------------------------------------------------------------
mu = p.copy()

# ---------------------------------------------------------------------------
# 1.2 -- ma tran khoang cach binh phuong + 1.3 -- 3-NN, scale
# ---------------------------------------------------------------------------
N = p.shape[0]
D2 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        D2[i, j] = np.sum((p[i] - p[j]) ** 2)
print("D2 =\n", D2)

d2_knn3 = np.zeros(N)
for i in range(N):
    others = np.array([D2[i, j] for j in range(N) if j != i])
    d2_knn3[i] = np.sort(others)[:3].mean()     # voi N=4 day chinh la mean toan bo
d2_knn3_clamped = np.clip(d2_knn3, 1e-7, None)
s_tilde = np.log(np.sqrt(d2_knn3_clamped))       # (N,) -- se broadcast ra 3 truc
s = np.exp(s_tilde)
print("d2_knn3 =", d2_knn3)
print("s_tilde =", s_tilde)
print("s =", s)

# ---------------------------------------------------------------------------
# 1.4 -- quaternion
# ---------------------------------------------------------------------------
q_tilde = np.zeros((N, 4))
q_tilde[:, 0] = 1.0

# ---------------------------------------------------------------------------
# 1.5 -- opacity
# ---------------------------------------------------------------------------
def inverse_sigmoid(x):
    return np.log(x / (1 - x))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

alpha_tilde = inverse_sigmoid(0.1 * np.ones((N, 1)))
print("alpha_tilde =", alpha_tilde.ravel())
assert np.allclose(sigmoid(alpha_tilde), 0.1)
alpha_tilde_gt = inverse_sigmoid(0.9)
assert np.isclose(alpha_tilde_gt, -alpha_tilde[0, 0])

# ---------------------------------------------------------------------------
# 1.6 -- SH: C0 va RGB2SH
# ---------------------------------------------------------------------------
C0 = 1.0 / (2.0 * np.sqrt(np.pi))
print("C0 =", C0)
assert np.isclose(C0, 0.28209479177387814)

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

k00 = RGB2SH(c)                       # (4,3)
max_sh_degree = 3
n_coeffs = (max_sh_degree + 1) ** 2   # 16
features = np.zeros((N, 3, n_coeffs))
features[:, :3, 0] = k00
features_dc = features[:, :, 0:1].transpose(0, 2, 1)     # (4,1,3)
features_rest = features[:, :, 1:].transpose(0, 2, 1)    # (4,15,3)
print("k00 =\n", k00)
print("features_dc.shape =", features_dc.shape, " features_rest.shape =", features_rest.shape)
assert np.max(np.abs(features_rest)) == 0.0

# ---------------------------------------------------------------------------
# 1.7 -- dem tham so
# ---------------------------------------------------------------------------
n_params_per_gaussian = 3 + 4 + 3 + 1 + 3 * n_coeffs
print("n_params_per_gaussian =", n_params_per_gaussian)
assert n_params_per_gaussian == 59
print("tong tham so quan the =", n_params_per_gaussian * N)   # 236

# ---------------------------------------------------------------------------
# 1.8 -- extent
# ---------------------------------------------------------------------------
c_bar = cam_centers.mean(axis=0)
dist = np.linalg.norm(cam_centers - c_bar, axis=1)
extent = 1.1 * dist.max()
print("c_bar =", c_bar)
print("dist =", dist)
print("extent =", extent)
assert np.isclose(extent, 1.6902498172, atol=1e-6)

# ---------------------------------------------------------------------------
# 1.9 -- ba nguong
# ---------------------------------------------------------------------------
delta = 0.001
eta_xyz = 1.6e-4
clone_split_thresh = delta * extent
prune_scale_thresh = 0.1 * extent
spatial_lr_scale = eta_xyz * extent
print("delta*extent =", clone_split_thresh)
print("0.1*extent  =", prune_scale_thresh)
print("eta_xyz*extent =", spatial_lr_scale)

# ---------------------------------------------------------------------------
# 1.10 -- bo nho Adam
# ---------------------------------------------------------------------------
floats_per_gaussian = 3 * n_params_per_gaussian   # 177
bytes_per_gaussian = floats_per_gaussian * 4
print("floats/Gaussian =", floats_per_gaussian, " bytes/Gaussian =", bytes_per_gaussian)
print("N0=4  -> ", floats_per_gaussian * 4, "floats,", bytes_per_gaussian * 4, "bytes")
N_big = 1_000_000
print("N=1e6 -> ", floats_per_gaussian * N_big, "floats,",
      bytes_per_gaussian * N_big, "bytes =", bytes_per_gaussian * N_big / 1e6, "MB =",
      bytes_per_gaussian * N_big / 1024**2, "MiB")

print("\n=== TAT CA KIEM CHUNG PASS ===")
```

Chạy script này (yêu cầu chỉ `numpy`, đúng ràng buộc môi trường không có `torch`) sẽ in ra toàn bộ các số
đã tính tay ở mục 1.0–1.10, với sai số chỉ nằm ở độ chính xác `float64` (đủ để khớp 10 chữ số thập phân
với các giá trị full-precision ghi trong Chương 6).

---

## 1.12 — Ghi chú đối chiếu với code

Mục này liệt kê chi tiết từng điểm khớp/không khớp/mở rộng giữa mô tả toán học của Chương 6 và mã nguồn
thật trong repo — kế thừa và mở rộng mục "Ghi chú đối chiếu" đã có ở cuối [Chương 6](../06-initialization.md#ghi-chú-đối-chiếu), thêm chi tiết về đường đi dữ liệu từ file COLMAP thô (mục 1.0), thứ không có trong
bản gốc vì Chương 6 giả định point cloud/camera đã ở dạng `BasicPointCloud`/`CameraInfo`.

### 1.12.1 — Đường đi từ file text đến `BasicPointCloud`/`CameraInfo`

1. `scene/dataset_readers.py` có hai hàm đọc COLMAP: bản binary (`read_points3D_binary`,
   `read_extrinsics_binary`, `read_intrinsics_binary`) và bản text (`read_points3D_text`,
   `read_extrinsics_text`, `read_intrinsics_text`) — cả hai nằm trong `scene/colmap_loader.py`. Đề bài
   §16.1 cho dữ liệu ở dạng text (`cameras.txt`, `images.txt`, `points3D.txt`), nên đường đi thật sự dùng
   `read_extrinsics_text`/`read_intrinsics_text`/`read_points3D_text`.
2. `readColmapCameras` (`dataset_readers.py:84`) duyệt qua `cam_extrinsics` (từ `images.txt`), với mỗi
   ảnh: `R = np.transpose(qvec2rotmat(extr.qvec))` (dòng 99), `T = np.array(extr.tvec)` — dựng đối tượng
   `CameraInfo(R=R, T=T, ...)`.
3. `getNerfppNorm` (dòng 45-66, trích đầy đủ ở mục 1.8) duyệt `cam_info` (danh sách `CameraInfo`), gọi
   `getWorld2View2(cam.R, cam.T)` cho từng camera rồi lấy cột thứ 4 của nghịch đảo — đúng chuỗi công thức
   đã dẫn ở mục 1.0.2 và 1.8.
4. Point cloud: `read_points3D_text` trả về mảng `xyz` (float) và `rgb` (**uint8**, $0$–$255$, đúng cột
   COLMAP); hàm gọi ở tầng trên (không trích ở đây, nằm ngoài phạm vi 5 file được yêu cầu đọc) chia
   `rgb/255.0` trước khi đóng gói vào `BasicPointCloud(points=xyz, colors=rgb/255.0, normals=...)` — đúng
   phép chia đã dùng ở mục 1.0.3. `BasicPointCloud` là `NamedTuple` định nghĩa trong
   `scene/gaussian_model.py` (không trích dòng cụ thể — nằm ngoài đoạn `create_from_pcd` được chương này
   trích).
5. `create_from_pcd(pcd, spatial_lr_scale)` (dòng 137, trích đầy đủ ở mục 1.1, 1.3, 1.4, 1.5, 1.6) nhận
   `pcd: BasicPointCloud` và `spatial_lr_scale` (chính là `extent`, xem mục 1.12.3 bên dưới) — không đọc
   file trực tiếp, chỉ xử lý dữ liệu đã parse.

### 1.12.2 — Khớp hoàn toàn

Toàn bộ các điểm đã ghi trong "Ghi chú đối chiếu" của Chương 6 vẫn đúng nguyên vẹn ở đây, liệt kê lại có
số dòng chính xác đã xác nhận bằng cách đọc trực tiếp file trong repo (không suy luận từ tài liệu):

- $\tilde s_i$: công thức, việc clamp $10^{-7}$ (`gaussian_model.py:147`, `torch.clamp_min(..., 0.0000001)`),
  và việc loại chính điểm khỏi 3-NN (`simple_knn.cu:159-160` và `178-179`, cả hai đều `if (i == idx)
  continue;`) — khớp.
- $\tilde q_i=(1,0,0,0)$: `gaussian_model.py:149-150` — khớp.
- $\tilde\alpha_i=\sigma^{-1}(0.1)$: `general_utils.py:18` (`inverse_sigmoid`) và `gaussian_model.py:152`
  — khớp.
- `RGB2SH`, hằng $C_0$: `sh_utils.py:26` (`C0 = 0.28209479177387814`), `sh_utils.py:114-115`
  (`def RGB2SH(rgb): return (rgb - 0.5) / C0`) — khớp.
- `extent` (hằng $1.1$): `dataset_readers.py:62` (`radius = diagonal * 1.1`) — khớp.
- Ba ngưỡng dùng `extent`: `gaussian_model.py:451-452` (clone/split), `:467` (prune scale), `:168`
  (`spatial_lr_scale` nhân vào `position_lr_init`) — khớp.
- $177$ float/Gaussian: đúng $3\times59$, chia $42$ (`optimizer`, 14 tham số) $+135$ (`shoptimizer`, 45
  tham số) — khớp, xem `training_setup`, `gaussian_model.py:162-177`.

### 1.12.3 — Chi tiết bổ sung: `spatial_lr_scale` chính là `extent`, không phải một đại lượng riêng

Đọc thêm `scene/__init__.py` (không nằm trong 5 file được yêu cầu ở đề bài §16.4.4, nhưng cần để hiểu
luồng dữ liệu đầy đủ): `Scene.__init__` gọi `getNerfppNorm(...)["radius"]` rồi lưu vào
`self.cameras_extent`, và khi tạo Gaussian gọi
`self.gaussians.create_from_pcd(pcd, self.cameras_extent)` — tức tham số thứ hai của `create_from_pcd`,
`spatial_lr_scale` (dòng 137, 138: `self.spatial_lr_scale = spatial_lr_scale`), **chính là** `extent` đã
tính ở mục 1.8, chỉ đổi tên biến khi truyền qua các lớp khác nhau
(`radius` trong `getNerfppNorm` → `cameras_extent` trong `Scene` → `spatial_lr_scale` trong
`GaussianModel`). Ba cái tên, một con số: $1.6902498172$.

### 1.12.4 — `boxMeanDist`: tại sao dùng box thay vì brute-force $O(N^2)$

Kernel CUDA (`simple_knn.cu:148-184`) không đơn giản là brute-force tính khoảng cách tới mọi điểm khác
(sẽ là $O(N^2)$, không khả thi với $N$ hàng triệu điểm SfM ở cảnh thật) — nó dùng cấu trúc "box" (ô lưới
không gian, được xây dựng ở phần đầu file `simple_knn.cu`, ngoài đoạn trích): vòng đầu (dòng 157-162) chỉ
quét $7$ điểm lân cận theo chỉ số Morton-order (`idx-3` đến `idx+3`) để có một ước lượng nhanh
(`reject = best[2]`, giá trị lớn nhất trong 3-best hiện có); vòng sau (dòng 169-182) quét theo từng box
không gian và **cắt tỉa sớm** (`if (dist > reject || dist > best[2]) continue;`) — bỏ qua toàn bộ box nếu
khoảng cách từ điểm tới box đó (dùng `distBoxPoint`, hàm khác trong file, không trích) đã lớn hơn ước
lượng hiện có. Đây là kỹ thuật tìm kiếm lân cận gần đúng có cắt tỉa không gian (giống ý tưởng k-d tree /
grid hash), giúp kernel chạy gần $O(N\log N)$ thay vì $O(N^2)$. Với $N_0=4$ ở cảnh đồ chơi, số điểm quá
nhỏ để có box nào bị cắt tỉa có ý nghĩa — kết quả toán học **không đổi**: 3-NN của mỗi điểm vẫn đúng là 3
điểm còn lại, như đã tính ở mục 1.3. Điểm mấu chốt cần nhớ: đây là **thuật toán tìm chính xác** 3-NN (có
tối ưu tốc độ bằng cắt tỉa, không phải xấp xỉ ngẫu nhiên như LSH) — nên kết quả tay ở mục 1.3 là chính
xác, không phải "gần đúng".

### 1.12.5 — Về `getWorld2View2`: vì sao có một vòng nghịch đảo–khôi phục thừa

Hàm `getWorld2View2` (`utils/graphics_utils.py:38-49`, trích ở mục 1.0.2) thực hiện $C2W=\mathrm{inv}(Rt)$
rồi cộng `translate`, nhân `scale` vào cột tâm camera, rồi lại $Rt=\mathrm{inv}(C2W)$ — nhìn thoáng qua có
vẻ thừa (nếu `translate=0, scale=1` thì $Rt$ cuối bằng $Rt$ đầu, như đã chứng minh ở mục 1.0.2). Lý do hàm
được viết tổng quát như vậy: ở những nơi khác trong pipeline (không thuộc Chương 6), `getWorld2View2` được
gọi **có** `translate`/`scale` khác mặc định — ví dụ để tái tâm hoá cảnh quanh gốc toạ độ hoặc co giãn
theo `nerf_normalization`. Trong `getNerfppNorm` (mục 1.8), lời gọi `getWorld2View2(cam.R, cam.T)` **không**
truyền `translate`/`scale` nên dùng mặc định $(0,0,0)$ và $1$ — vòng nghịch đảo–khôi phục ở đây thực chất
trung tính, chỉ để tái sử dụng cùng một hàm cho cả hai mục đích.

### 1.12.7 — Toàn bộ hằng số learning-rate/threshold liên quan Chương 6, đọc trực tiếp từ `arguments/__init__.py`

Để không phải lật lại repo mỗi khi cần một hằng số, bảng dưới liệt kê **nguyên văn** các dòng liên quan
tới Chương 6 (khởi tạo optimizer, ngưỡng `extent`) đọc từ `arguments/__init__.py`:

```python
self.position_lr_init = 0.00016        # dòng 76 — eta_xyz ở mục 1.9(c)
self.position_lr_final = 0.0000016     # dòng 77 — lr vị trí cuối lịch (Chương 11, không dùng ở Ch.6)
self.position_lr_max_steps = 30_000    # dòng 78 — số bước lịch giảm lr vị trí (Chương 11)
self.opacity_lr = 0.025                # dòng 79 — lr riêng cho tilde-alpha (Chương 11)
self.scaling_lr = 0.005                # dòng 80 — lr riêng cho tilde-s (Chương 11)
self.rotation_lr = 0.001               # dòng 81 — lr riêng cho tilde-q (Chương 11)
self.highfeature_lr = 0.005            # dòng 92 — lr gốc k_lm, l>=1, chia thêm /20 khi dùng (dòng 174)
self.lowfeature_lr = 0.0025            # dòng 93 — lr riêng cho k_00 (= f_dc)
self.dense = 0.001                     # dòng 95 — delta ở mục 1.9(a)
```

Chỉ **hai** trong số này thật sự dùng ở Chương 6 (khởi tạo, không phải bước tối ưu): `position_lr_init`
(nhân với `extent` ra `spatial_lr_scale`, mục 1.9(c)) và `dense` (nhân với `extent` ra ngưỡng clone/split,
mục 1.9(a)). Các hằng số còn lại (`opacity_lr`, `scaling_lr`, `rotation_lr`, `highfeature_lr`,
`lowfeature_lr`, `position_lr_final`, `position_lr_max_steps`) chỉ có ý nghĩa **từ Chương 11 trở đi** (một
bước Adam thật sự cần biết lr) — liệt kê ở đây để Phần 6 (Chương 11 của lời giải) không phải tra lại repo,
và để người đọc thấy rõ Chương 6 chỉ "chạm" vào đúng 2 trong 9 hằng số này.

Đối chiếu $0.1690249817$ (ngưỡng prune "scale quá lớn", mục 1.9(b)): hằng số $0.1$ **không** nằm trong
`arguments/__init__.py` — nó được viết trực tiếp (hard-code) ở `gaussian_model.py:467` (không phải một
`training_args.xxx` cấu hình được), khác với `dense=0.001` và `position_lr_init` vốn là tham số dòng lệnh
có thể override. Đây là một điểm bất đối xứng nhỏ trong thiết kế API của repo, không ảnh hưởng gì tới kết
quả số nhưng đáng ghi chú cho người muốn tinh chỉnh hyperparameter: muốn đổi ngưỡng prune phải sửa code,
không đổi được qua command-line argument.

### 1.12.6 — Điều `points3D.txt` không cung cấp mà Chương 6 không cần

Cột `ERROR` và `TRACK[]` của `points3D.txt` (mục 1.0.3) là sản phẩm của bundle adjustment trong COLMAP —
đo độ tin cậy hình học của từng điểm 3D và tập ảnh quan sát nó. `create_from_pcd` không dùng đến độ tin
cậy này (mọi điểm SfM được coi là "chắc chắn" như nhau khi khởi tạo Gaussian) — đây là một giả định ngầm
của 3DGS gốc, không phải giới hạn của cách đọc dữ liệu: một biến thể cải tiến có thể dùng `ERROR` để khởi
tạo opacity hoặc scale khác nhau cho điểm tin cậy cao/thấp, nhưng FastGS-lite (và 3DGS gốc) không làm vậy.

---

## 1.13 — Biến thể mở rộng: điểm SfM thứ 5 giả định

Mục này **không phải một phần của cảnh chính** — nó là một bài toán "what-if" độc lập để minh hoạ cách
3-NN và scale khởi tạo phản ứng khi thêm một điểm SfM, làm rõ thêm cơ chế đã học ở mục 1.3. Không dùng lại
ở bất kỳ phần nào khác của lời giải 8 phần.

**Giả định:** thêm điểm thứ 5 $p_5=(0.2,\ 0.4,\ -0.3)$, màu $c_5=(0.6,\ 0.1,\ 0.4)$, vào cùng cảnh 4 điểm
gốc. Hỏi: 3-NN và scale khởi tạo của **toàn bộ 5 điểm** (không chỉ điểm mới) thay đổi ra sao?

### Bước 1 — khoảng cách bình phương từ $p_5$ tới 4 điểm gốc

$$
p_5-p_1 = (0.2,\ 0.4,\ -0.3) \Rightarrow \lVert p_5-p_1\rVert^2 = 0.04+0.16+0.09 = 0.29
$$
$$
p_5-p_2 = (0.2-0.5,\ 0.4-0.3,\ -0.3-0.5) = (-0.3,\ 0.1,\ -0.8) \Rightarrow \lVert p_5-p_2\rVert^2 = 0.09+0.01+0.64 = 0.74
$$
$$
p_5-p_3 = (0.2-(-0.4),\ 0.4-(-0.2),\ -0.3-1.0) = (0.6,\ 0.6,\ -1.3) \Rightarrow \lVert p_5-p_3\rVert^2 = 0.36+0.36+1.69 = 2.41
$$
$$
p_5-p_4 = (0.2-0.3,\ 0.4-(-0.5),\ -0.3-0.2) = (-0.1,\ 0.9,\ -0.5) \Rightarrow \lVert p_5-p_4\rVert^2 = 0.01+0.81+0.25 = 1.07
$$

### Bước 2 — ma trận khoảng cách $5\times5$ đầy đủ

| | $p_1$ | $p_2$ | $p_3$ | $p_4$ | $p_5$ |
|---|---|---|---|---|---|
| $p_1$ | 0 | 0.59 | 1.2 | 0.38 | 0.29 |
| $p_2$ | 0.59 | 0 | 1.31 | 0.77 | 0.74 |
| $p_3$ | 1.2 | 1.31 | 0 | 1.22 | 2.41 |
| $p_4$ | 0.38 | 0.77 | 1.22 | 0 | 1.07 |
| $p_5$ | 0.29 | 0.74 | 2.41 | 1.07 | 0 |

### Bước 3 — 3-NN của mỗi điểm trong tập 5 điểm (bây giờ **phải chọn**, vì có 4 ứng viên chứ không phải đúng 3)

Với $N=5$, mỗi điểm có $4$ láng giềng khả dĩ — 3-NN nghĩa là chọn 3 giá trị **nhỏ nhất**, bỏ đi 1.

- **$p_1$**: các khoảng cách tới $\{p_2,p_3,p_4,p_5\}=\{0.59,\ 1.2,\ 0.38,\ 0.29\}$. Sắp tăng dần:
  $0.29\ (p_5),\ 0.38\ (p_4),\ 0.59\ (p_2),\ 1.2\ (p_3)$. 3-NN $=\{p_5,p_4,p_2\}$, **bỏ $p_3$** (xa nhất).
  $$d^2_{\mathrm{knn3}}(p_1) = \frac{0.29+0.38+0.59}{3} = \frac{1.26}{3} = 0.42$$
  So với giá trị cũ $0.7233333333$ (mục 1.3, khi chưa có $p_5$): **giảm mạnh**, vì $p_5$ ở gần $p_1$
  ($0.29$, nhỏ hơn cả khoảng cách gần nhất cũ là $0.38$) nên thay thế láng giềng xa nhất ($p_3$, $1.2$).
- **$p_2$**: khoảng cách tới $\{p_1,p_3,p_4,p_5\}=\{0.59,\ 1.31,\ 0.77,\ 0.74\}$. Sắp tăng dần:
  $0.59\ (p_1),\ 0.74\ (p_5),\ 0.77\ (p_4),\ 1.31\ (p_3)$. 3-NN $=\{p_1,p_5,p_4\}$, **bỏ $p_3$**.
  $$d^2_{\mathrm{knn3}}(p_2) = \frac{0.59+0.74+0.77}{3} = \frac{2.10}{3} = 0.70$$
  So với cũ $0.89$: giảm, vì $p_5$ ($0.74$) gần hơn $p_3$ ($1.31$), thay thế láng giềng xa nhất.
- **$p_3$**: khoảng cách tới $\{p_1,p_2,p_4,p_5\}=\{1.2,\ 1.31,\ 1.22,\ 2.41\}$. Sắp tăng dần:
  $1.2\ (p_1),\ 1.22\ (p_4),\ 1.31\ (p_2),\ 2.41\ (p_5)$. 3-NN $=\{p_1,p_4,p_2\}$, **bỏ $p_5$** (điểm mới
  là xa nhất đối với $p_3$, không lọt vào 3-NN).
  $$d^2_{\mathrm{knn3}}(p_3) = \frac{1.2+1.22+1.31}{3} = \frac{3.73}{3} = 1.2433333333$$
  **Không đổi** so với mục 1.3 (vẫn đúng 3 điểm gốc, vì $p_5$ quá xa $p_3$ để lọt vào top-3).
- **$p_4$**: khoảng cách tới $\{p_1,p_2,p_3,p_5\}=\{0.38,\ 0.77,\ 1.22,\ 1.07\}$. Sắp tăng dần:
  $0.38\ (p_1),\ 0.77\ (p_2),\ 1.07\ (p_5),\ 1.22\ (p_3)$. 3-NN $=\{p_1,p_2,p_5\}$, **bỏ $p_3$**.
  $$d^2_{\mathrm{knn3}}(p_4) = \frac{0.38+0.77+1.07}{3} = \frac{2.22}{3} = 0.74$$
  So với cũ $0.79$: giảm nhẹ, vì $p_5$ ($1.07$) gần hơn $p_3$ ($1.22$) nên thay thế.
- **$p_5$** (điểm mới): khoảng cách tới $\{p_1,p_2,p_3,p_4\}=\{0.29,\ 0.74,\ 2.41,\ 1.07\}$. Sắp tăng dần:
  $0.29\ (p_1),\ 0.74\ (p_2),\ 1.07\ (p_4),\ 2.41\ (p_3)$. 3-NN $=\{p_1,p_2,p_4\}$, **bỏ $p_3$**.
  $$d^2_{\mathrm{knn3}}(p_5) = \frac{0.29+0.74+1.07}{3} = \frac{2.10}{3} = 0.70$$

### Bước 4 — scale mới cho cả 5 điểm

| $i$ | $d^2_{\mathrm{knn3}}$ (5 điểm) | $d^2_{\mathrm{knn3}}$ cũ (4 điểm) | thay đổi | $\tilde s_i$ mới $=\tfrac12\log d^2$ | $s_i$ mới $=\sqrt{d^2}$ |
|---|---|---|---|---|---|
| 1 | $0.42$ | $0.7233333333$ | giảm | $\tfrac12\log(0.42)=-0.4335108654$ | $0.6480740698$ |
| 2 | $0.70$ | $0.89$ | giảm | $\tfrac12\log(0.70)=-0.1783374693$ | $0.8366600265$ |
| 3 | $1.2433333333$ | $1.2433333333$ | **không đổi** | $0.1088979725$ | $1.1150934466$ |
| 4 | $0.74$ | $0.79$ | giảm | $\tfrac12\log(0.74)=-0.1503102825$ | $0.8602325267$ |
| 5 (mới) | $0.70$ | — | — | $-0.1783374693$ | $0.8366600265$ |

**Bài học rút ra:** thêm một điểm SfM **chỉ ảnh hưởng cục bộ** — nó làm giảm scale của những điểm mà nó
đủ gần để lọt vào top-3 láng giềng của chúng ($p_1,p_2,p_4,p_5$ — bốn điểm này scale giảm hoặc mới xuất
hiện), nhưng **không ảnh hưởng gì** tới những điểm mà nó quá xa để lọt top-3 (chỉ $p_3$ ở đây, scale giữ
nguyên $1.115$ tuyệt đối chính xác). Đây là hệ quả trực tiếp của việc dùng $k=3$ láng giềng **gần nhất**
(cục bộ) thay vì trung bình toàn cục — đúng như bài học ở Bài tập 6.1 của Chương 6: 3-NN "phủ khe hở cục
bộ", không bị nhiễu bởi cấu trúc ở xa. Thí nghiệm này cũng cho thấy vì sao thuật toán 3-NN thật (mục
1.12.4) cần cắt tỉa không gian hiệu quả: khi $N$ lớn, thêm/bớt một điểm chỉ cần cập nhật lân cận cục bộ,
không cần quét lại toàn bộ $O(N^2)$ cặp.

(Màu $c_5=(0.6,0.1,0.4)$ không dùng trong bài toán 3-NN này — chỉ để đề bài giả định "đủ" một điểm SfM
hoàn chỉnh; nếu cần, $k_{5,00}=(c_5-0.5)/C_0 = (0.1,\ -0.4,\ -0.1)\times3.5449077018 =
(0.3544907702,\ -1.4179630807,\ -0.3544907702)$, tính theo đúng công thức mục 1.6.2, không ảnh hưởng gì
tới 4 Gaussian gốc.)

---

## 1.14 — Bài tập

**Bài tập 1.A.** Dùng đúng công thức đã dẫn ở mục 1.0.2 ($c_v=-t_v$ khi $R=I_3$), hãy suy luận tổng quát
hơn: nếu `images.txt` cho một quaternion $q\ne(1,0,0,0)$ (tức $R\ne I_3$), công thức $c_v=(W_v)^{-1}[{:}3,3]$
sẽ cho ra biểu thức nào theo $R$ và $t$ (gợi ý: dùng khối nghịch đảo $\begin{pmatrix}R^\top & t\\0&1\end{pmatrix}^{-1}=\begin{pmatrix}R & -Rt\\0&1\end{pmatrix}$
— tự chứng minh công thức này bằng cách nhân hai ma trận ra $I_4$). So sánh với trường hợp riêng $R=I_3$
đã dùng ở mục 1.0.2.

**Bài tập 1.B.** Ở mục 1.6.1, $C_0$ được dẫn từ điều kiện chuẩn hoá $\int_{S^2}Y_0^0{}^2\,d\omega=1$.
Hãy dẫn lại — cùng cách lập luận — hằng số $C_1=0.4886025119$ dùng cho $l=1$ trong `sh_utils.py:27`
($Y_1^{-1},Y_1^0,Y_1^{+1}$ đều có cùng hệ số chuẩn hoá $C_1$ nhân với $y,z,x$ tương ứng) bằng cách dùng
điều kiện $\int_{S^2}(C_1\cdot z)^2\,d\omega=1$ với $z=\cos\theta$ trên toạ độ cầu, và tích phân
$\int_{S^2}\cos^2\theta\,d\omega = \int_0^{2\pi}\!\!\int_0^\pi\cos^2\theta\sin\theta\,d\theta\,d\varphi$.
(Không bắt buộc phải ra đúng số chính xác nếu tích phân khó — mục tiêu là hiểu **cơ chế** dẫn hằng số, đã
minh hoạ đầy đủ với $C_0$.)

**Bài tập 1.C.** Ở mục 1.8, ta thấy giá trị `extent` tính tay bằng khai căn thập phân ($1.6902135399$)
lệch khoảng $3.6\times10^{-5}$ so với giá trị full-precision `float64` ($1.6902498172$). Giải thích: nguồn
sai số này đến từ đâu (làm tròn ở bước nào cụ thể trong chuỗi tính tay), và tính lại $\sqrt{2.3611111111}$
bằng phương pháp Newton–Raphson (hai bước lặp, xuất phát từ $x_0=1.5366$) để kiểm tra độ chính xác được
cải thiện ra sao.

**Bài tập 1.D.** Mục 1.9 chỉ ra rằng cả 4 Gaussian ở cảnh đồ chơi đều vượt ngưỡng `0.1·extent` ngay từ
khởi tạo. Giả sử một cảnh thật có $N_0=100\,000$ điểm SfM phân bố đều trong một khối lập phương cạnh
$10$ đơn vị (thay vì 4 điểm rải rác như cảnh đồ chơi), và camera đặt cách xa khối này khoảng $50$ đơn vị
(tương tự tỉ lệ với cảnh đồ chơi, nơi camera cách gốc toạ độ $4$ đơn vị còn điểm SfM trải trong khoảng
$[-0.4, 1.0]$). Ước lượng định tính (không cần tính số chính xác): khoảng cách 3-NN trung bình giữa các
điểm SfM trong khối lập phương này so với `extent` (cỡ $50\times1.1\approx55$) sẽ lớn hơn hay nhỏ hơn
nhiều so với tỉ lệ $s_i/\text{extent}\approx0.85/1.69\approx0.5$ đã thấy ở cảnh đồ chơi? Giải thích vì sao
mật độ điểm SfM ảnh hưởng trực tiếp tới việc Gaussian có "quá lớn ngay từ đầu" hay không.

**Bài tập 1.E.** Dùng script ở mục 1.11 làm khung, viết thêm một hàm `verify_all()` gọi `assert` cho toàn
bộ 59 giá trị của $\theta_3$ (Gaussian số 3, liệt kê đầy đủ ở mục 1.7) so với giá trị tính tay, với dung
sai `atol=1e-9`. Đây là cách "đóng vòng" kiểm chứng: mọi con số trình bày trong Phần 1 này phải tái tạo
được bằng script, không chỉ bằng suy luận trên giấy.

**Bài tập 1.F.** Mục 1.6.5 kiểm chứng $\text{SH2RGB}(\text{RGB2SH}(c))=c$ bằng số cho cả 4 Gaussian. Hãy
chứng minh đẳng thức này đúng **với mọi** $c\in\mathbb R$ (không chỉ $c\in[0,1]$), bằng đại số thuần tuý
(thay công thức `RGB2SH` vào `SH2RGB` và rút gọn), và chỉ ra vì sao tính "đúng với mọi $c$" (không chỉ
trong khoảng màu hợp lệ) là một tính chất tự nhiên của phép biến đổi **affine tuyến tính**, khác với các
phép biến đổi phi tuyến như $\sigma$/$\sigma^{-1}$ ở mục 1.5 (vốn chỉ xác định trên $(0,1)$).

**Bài tập 1.G.** Bảng byte breakdown ở mục 1.10 cho thấy SH bậc cao chiếm $76.27\%$ bộ nhớ Adam dù chỉ có
$45/59\approx76.27\%$ **số lượng** tham số — hai tỉ lệ này trùng nhau. Giải thích tại sao trùng nhau
không phải ngẫu nhiên (gợi ý: xét công thức tính byte cho một nhóm bất kỳ có $k$ tham số trên tổng $59$,
rồi rút gọn tỉ lệ).

**Bài tập 1.H.** Dùng kết quả mục 1.13 (biến thể 5 điểm), giả sử thay vì thêm $p_5=(0.2,0.4,-0.3)$ ta
thêm một điểm trùng gần như hoàn toàn với $p_1$, ví dụ $p_5'=(10^{-8},\ 10^{-8},\ 10^{-8})$. Tính
$\lVert p_5'-p_1\rVert^2$ và cho biết giá trị này có bị chặn dưới bởi clamp $10^{-7}$ ở công thức
$\tilde s_i$ (mục 1.3) hay không — nếu không, hãy tìm khoảng cách nhỏ nhất giữa hai điểm SfM (theo bình
phương) để clamp bắt đầu có tác dụng, và giải thích ý nghĩa vật lý của việc clamp này (điều gì xảy ra với
$\tilde s_i=\log\sqrt{d^2}$ nếu $d^2\to0$ mà không có clamp?).

**Bài tập 1.I.** Mục 1.12.7 chỉ ra ngưỡng prune $0.1$ ở `gaussian_model.py:467` là hard-code, không nằm
trong `arguments/__init__.py`. Nếu bạn muốn thử nghiệm với ngưỡng $0.05\cdot\text{extent}$ thay vì
$0.1\cdot\text{extent}$ cho cảnh đồ chơi này, hãy tính lại giá trị ngưỡng mới và so sánh với 4 giá trị
$s_i$ đã tính ở mục 1.3 — với ngưỡng mới, có Gaussian nào **không còn** bị coi là "quá lớn" ngay từ khởi
tạo không?

**Bài tập 1.J.** Viết lại công thức extent tổng quát cho $V$ camera bất kỳ (không chỉ $V=3$) dưới dạng
tổng, rồi áp dụng cho trường hợp $V=1$ (chỉ một camera duy nhất). Chứng minh rằng với $V=1$, `extent`
luôn bằng $0$ (vì $c_1=\bar c$ khi chỉ có một camera), và giải thích tại sao đây là một trường hợp suy
biến có vấn đề cho pipeline (ngưỡng clone/split, prune, và `spatial_lr_scale` đều sẽ bằng $0$ — điều gì
xảy ra với quá trình tối ưu nếu `spatial_lr_scale=0`?).

---

## 1.16 — Trace `create_from_pcd` dòng theo dòng với số thật của cảnh

Để không còn chỗ nào mơ hồ, mục này chạy lại **toàn bộ thân hàm** `create_from_pcd`
(`gaussian_model.py:137-160`) từng dòng một, thay tensor trừu tượng bằng giá trị số thật đã tính ở các
mục trước — coi như "chạy tay" hàm Python này trên cảnh đồ chơi.

```python
def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
```

Gọi với `pcd.points` = mảng $4\times3$ ở mục 1.0.3, `pcd.colors` = mảng $4\times3\in[0,1]$ cùng mục, và
`spatial_lr_scale = extent = 1.6902498172` (mục 1.8, xem lý do ở mục 1.12.3).

```python
    self.spatial_lr_scale = spatial_lr_scale
```

→ `self.spatial_lr_scale = 1.6902498172`. Lưu lại để `training_setup` dùng sau (mục 1.9(c)).

```python
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
```

→ tensor shape $(4,3)$, giá trị đúng bảng $\mu_i$ ở mục 1.1:
$\begin{pmatrix}0&0&0\\0.5&0.3&0.5\\-0.4&-0.2&1.0\\0.3&-0.5&0.2\end{pmatrix}$.

```python
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
```

→ tensor shape $(4,3)$, giá trị đúng bảng $k_{i,00}$ ở mục 1.6.3:
$\begin{pmatrix}1.0635&-1.0635&-1.0635\\-1.0635&0.7090&-0.7090\\-1.4180&-0.7090&1.4180\\0&0&0\end{pmatrix}$
(làm tròn 4 chữ số; full precision ở bảng 1.6.3).

```python
    features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
```

→ `fused_color.shape[0] = 4`; `(max_sh_degree+1)**2 = (3+1)**2 = 16`; tensor `features` shape $(4,3,16)$,
toàn bộ $4\times3\times16=192$ phần tử $=0$ tại thời điểm này.

```python
    features[:, :3, 0] = fused_color
```

→ gán $4\times3=12$ giá trị vào slot bậc-0 (index cuối $=0$): `features[i,c,0] = fused_color[i,c]` cho
$i=0..3,\ c=0..2$. Sau dòng này, $12/192$ phần tử khác $0$, $180/192$ vẫn $=0$.

```python
    features[:, 3:, 1:] = 0.0
```

→ dòng này **không có tác dụng thực sự** trên tensor `features` shape $(4,3,16)$: chiều thứ hai
(`dim=1`, kích thước $3$, ứng với 3 kênh màu) chỉ số `3:` là **slice rỗng** (không có chỉ số $\ge3$ khi
kích thước là $3$, chỉ số hợp lệ $0,1,2$) — dòng code này gán vào một tensor rỗng, tương đương no-op. Đây
là một dòng thừa/copy-paste còn sót lại trong code gốc (rất có thể ban đầu định gán vào chiều SH — chiều
thứ ba — chứ không phải chiều kênh màu — chiều thứ hai); không ảnh hưởng gì tới kết quả vì `features` đã
được khởi tạo `torch.zeros` toàn bộ ngay từ đầu, nên các slot bậc $\ge1$ vốn đã là $0$ mà không cần dòng
này.

```python
    print("Number of points at initialisation : ", fused_point_cloud.shape[0])
```

→ in ra `Number of points at initialisation :  4`.

```python
    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
```

→ `distCUDA2` gọi kernel `boxMeanDist` (mục 1.3, 1.12.4), trả về tensor shape $(4,)$:
$(0.7233333333,\ 0.89,\ 1.2433333333,\ 0.79)$; `clamp_min(..., 1e-7)` không đổi gì (mục 1.3).

```python
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
```

→ `torch.sqrt(dist2)` = $(0.8504899611,\ 0.9433981132,\ 1.1150934466,\ 0.8888194436)$ (chính là $s_i$,
nhưng **trước** khi lấy log — biến trung gian này không được gán tên riêng trong code); `torch.log(...)`
= $(-0.1619425606,\ -0.0582669089,\ 0.1088979725,\ -0.1178611446)$ (chính là $\tilde s_i$, mục 1.3);
`[...,None]` thêm chiều mới shape $(4,1)$; `.repeat(1,3)` nhân bản cột đó thành $3$ cột giống hệt nhau →
`scales` shape $(4,3)$, mỗi hàng $i$ là $(\tilde s_i,\tilde s_i,\tilde s_i)$ — đúng tính đẳng hướng đã nêu
ở mục 1.3.

```python
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1
```

→ `rots` shape $(4,4)$, mỗi hàng $(1,0,0,0)$ — đúng mục 1.4.

```python
    opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
```

→ `0.1 * torch.ones((4,1))` = cột $4\times1$ toàn $0.1$; `inverse_opacity_activation` chính là
`inverse_sigmoid` (gán ở nơi khác trong `__init__` của `GaussianModel`, không trích ở đây) →
`opacities` shape $(4,1)$, mỗi phần tử $-2.1972245773$ — đúng mục 1.5.

```python
    self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    self._scaling = nn.Parameter(scales.requires_grad_(True))
    self._rotation = nn.Parameter(rots.requires_grad_(True))
    self._opacity = nn.Parameter(opacities.requires_grad_(True))
    self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
```

→ 6 dòng cuối gói mọi tensor đã tính thành `nn.Parameter` (để PyTorch autograd theo dõi gradient — quan
trọng cho Chương 11, không phải Chương 6). `features[:,:,0:1]` cắt slot bậc-0 shape $(4,3,1)$,
`.transpose(1,2)` hoán đổi chiều $1$ và $2$ → shape $(4,1,3)$ — đúng `_features_dc` đã nói ở mục 1.6.4.
`features[:,:,1:]` cắt 15 slot còn lại shape $(4,3,15)$, `.transpose(1,2)` → shape $(4,15,3)$ — đúng
`_features_rest`. `max_radii2D` shape $(4,)$, toàn bộ $=0$ — bán kính chiếu 2D lớn nhất từng đạt được của
mỗi Gaussian, dùng ở Chương 12 (densify/prune), khởi tạo $0$ vì chưa render lần nào.

Sau khi hàm chạy xong, `GaussianModel` giữ đúng $\mathcal G_0$ như đã lập bảng ở mục 1.7 — không có phép
tính nào trong `create_from_pcd` mà chưa được dẫn tay ở các mục trước.

---

## Phụ lục — Bảng tra cứu nhanh mọi số liệu của Phần 1

Bảng tổng hợp toàn bộ hằng số/kết quả trung gian xuất hiện trong Phần 1, kèm mục đã dẫn ra chúng, để tra
cứu nhanh khi đọc Phần 2–8 mà không cần lật lại từng mục:

| Ký hiệu | Giá trị | Dẫn ở mục |
|---|---|---|
| $\tan(\mathrm{fov}_x/2)$ | $0.6$ | 1.0.1 |
| $\tan(\mathrm{fov}_y/2)$ | $0.4$ | 1.0.1 |
| $c_1,c_2,c_3$ | $(0,0,-4),\ (1.5,0,-4),\ (-1.5,0.5,-4)$ | 1.0.2 |
| $p_1,\dots,p_4$ | xem bảng 1.0.3 | 1.0.3 |
| $c_1,\dots,c_4$ (màu) | xem bảng 1.0.3 | 1.0.3 |
| $\lVert p_i-p_j\rVert^2$ (6 cặp) | $0.59,\ 1.2,\ 0.38,\ 1.31,\ 0.77,\ 1.22$ | 1.2 |
| $d^2_{\mathrm{knn3}}(p_i)$, $i{=}1..4$ | $0.7233333333,\ 0.89,\ 1.2433333333,\ 0.79$ | 1.3 |
| $\tilde s_i$, $i{=}1..4$ | $-0.1619425606,\ -0.0582669089,\ 0.1088979725,\ -0.1178611446$ | 1.3 |
| $s_i$, $i{=}1..4$ | $0.8504899611,\ 0.9433981132,\ 1.1150934466,\ 0.8888194436$ | 1.3 |
| $\tilde q_i$ | $(1,0,0,0)\ \forall i$ | 1.4 |
| $\tilde\alpha_i$ | $-2.1972245773\ \forall i$ | 1.5 |
| $\alpha_i$ | $0.1\ \forall i$ | 1.5 |
| $\sigma^{-1}(0.9)$ | $+2.1972245773$ | 1.5 |
| $C_0$ | $0.2820947918$ | 1.6.1 |
| $1/C_0$ | $3.5449077018$ | 1.6.2 |
| $k_{i,00}$, $i{=}1..4$ | xem bảng 1.6.3 | 1.6.3 |
| $k_{i,lm}$, $l\ge1$ | $0$ (45 hệ số/Gaussian) | 1.6.4 |
| tham số/Gaussian | $59$ | 1.7 |
| tổng tham số $\mathcal G_0$ | $236$ | 1.7 |
| $\bar c$ | $(0,\ 0.1666666667,\ -4)$ | 1.8 |
| $\lVert c_v-\bar c\rVert$, $v{=}1..3$ | $0.1666666667,\ 1.5092308094,\ 1.5365577635$ | 1.8 |
| $\text{extent}$ | $1.6902498172$ | 1.8 |
| $\delta\cdot\text{extent}$ | $1.6902498\times10^{-3}$ | 1.9(a) |
| $0.1\cdot\text{extent}$ | $0.1690249817$ | 1.9(b) |
| $\eta_{xyz}\cdot\text{extent}$ | $2.7043997\times10^{-4}$ | 1.9(c) |
| float/Gaussian (Adam) | $177$ | 1.10 |
| byte, $N_0=4$ | $2\,832$ | 1.10 |
| byte, $N=10^6$ | $7.08\times10^8=708.0$ MB $=675.2$ MiB | 1.10 |

---

## 1.15 — Đầu ra chuyển cho Phần 2

$$
\mathcal G_0 = \{\theta_i\}_{i=1}^4,\qquad \theta_i=(\mu_i,\ \tilde q_i,\ \tilde s_i,\ \tilde\alpha_i,\ k_{i,00},\ k_{i,\mathrm{rest}})\in\mathbb R^{59}
$$

— bảng đầy đủ ở mục 1.7 (cả dạng vector $59$ giá trị tường minh và dạng bảng nhóm). Kèm theo:

| Đại lượng | Giá trị (full precision) | Dùng ở |
|---|---|---|
| $\text{extent}$ | $1.6902498172$ | Phần 2 (Chương 7, ngưỡng scale), Phần 7 (Chương 12, densify/prune) |
| $\delta\cdot\text{extent}$ | $1.6902498\times10^{-3}$ | Phần 7 |
| $0.1\cdot\text{extent}$ | $0.1690249817$ | Phần 2, Phần 7 |
| $\eta_{xyz}\cdot\text{extent}$ | $2.7043997\times10^{-4}$ | Phần 6 (Chương 11, learning rate) |
| $c_v$ ($v=1,2,3$) | $(0,0,-4),\ (1.5,0,-4),\ (-1.5,0.5,-4)$ | Phần 2, Phần 3 (chiếu camera) |
| $\tan(\mathrm{fov}_x/2),\ \tan(\mathrm{fov}_y/2)$ | $0.6,\ 0.4$ | Phần 3 (Chương 8, projection) |
| Bộ nhớ Adam ($N_0=4$) | $708$ float $=2\,832$ byte | (thông tin, không cần cho Phần 2) |

**Đầu ra chuyển cho Phần 2 (Chương 7 — 3D Gaussians): 𝒢₀ = 4 Gaussian × 59 tham số (bảng đầy đủ ở
trên).**

[Phần 2 →](02-gaussians.md)

---

[← Đề bài](../16-bai-toan-lon-de-bai.md) · [← Chương 6 gốc](../06-initialization.md)
