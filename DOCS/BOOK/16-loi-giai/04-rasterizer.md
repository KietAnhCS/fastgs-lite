[← Phần 3](03-projection.md) · [Mục lục lời giải](00-muc-luc-loi-giai.md) · Phần 4/8

# Phần 4 — Chương 9: Differentiable Tile Rasterizer

> Đầu vào nhận từ Phần 3 (Chương 8): $\Sigma'_{i,v}$, compact box, tile list, thứ tự độ sâu — xem [Phần 3](03-projection.md).

> Nguồn lý thuyết: [Chương 9 — Differentiable Tile Rasterizer](../09-differentiable-tile-rasterizer.md).
> Cảnh dùng chung: [§6.2](../06-initialization.md). Script kiểm chứng đầy đủ nằm ở phụ lục cuối file này
> (đồng thời là script đã chạy để sinh **mọi** con số trong file — không có số nào chép tay).

Quy ước ký hiệu và hằng số hệ thống theo [chương 5](../05-ky-hieu-nen-tang-toan-hoc.md): tile $16\times16$
px, low-pass $0.3$, clamp $1.3\tan(\mathrm{fov}/2)$, ngưỡng alpha $\tfrac1{255}$, dừng sớm $T(1-\alpha)<10^{-4}$,
`mult`$=0.5$, nền $C_{bg}=(1,1,1)$ (trắng), clamp $\alpha_n(x)\le0.99$.

---

## 4.0 — Tóm tắt đầu vào từ Phần 3 (cả 3 camera)

Bốn Gaussian $i=1..4$ với màu SH bậc 0 (không đổi theo hướng nhìn):
$c_1=(0.8,0.2,0.2)$ (đỏ), $c_2=(0.2,0.7,0.3)$ (lục), $c_3=(0.1,0.3,0.9)$ (lam), $c_4=(0.5,0.5,0.5)$ (xám);
$\alpha_i=0.1$ cho cả 4. Camera: $c_1=(0,0,-4)$, $c_2=(1.5,0,-4)$, $c_3=(-1.5,0.5,-4)$, $R=I$, $f_x=f_y=40$,
ảnh $48\times32=$ lưới tile $3\times2$ (6 tile, đánh số $T=t_y\cdot3+t_x$, $T=0..5$).

Depth $t_z=\mu_z-c_{v,z}$ **không phụ thuộc camera** (camera chỉ tịnh tiến theo $x,y$, không theo $z$),
nên thứ tự độ sâu giống hệt nhau ở cả 3 view:

$$\text{depth}_1=4.0<\text{depth}_4=4.2<\text{depth}_2=4.5<\text{depth}_3=5.0$$

$$\boxed{\ \text{thứ tự front-to-back (mọi tile, mọi camera nếu Gaussian đó chạm tile)}: \ G_1\to G_4\to G_2\to G_3\ }$$

### Camera 1

| $i$ | $\mu'_i$ (px) | $\Sigma'_{11},\Sigma'_{12},\Sigma'_{22}$ | $(A,B,C)$ | depth | half$_x$,half$_y$ | $\mathcal K_i$ (tile) | $K_i$ |
|---|---|---|---|---|---|---|---|
| 1 | (23.5000, 15.5000) | (72.6333, 0.0000, 72.6333) | (0.01376778, -0.00000000, 0.01376778) | 4.0000 | (15.3374, 15.3374) | $\{0, 1, 2, 3, 4, 5\}$ | 6 |
| 2 | (27.9444, 18.1667) | (71.4891, 0.5209, 70.9335) | (0.01398889, -0.00010273, 0.01409846) | 4.5000 | (15.2161, 15.1569) | $\{0, 1, 2, 3, 4, 5\}$ | 6 |
| 3 | (20.3000, 13.9000) | (80.3826, 0.2546, 80.0007) | (0.01244063, -0.00003960, 0.01250002) | 5.0000 | (16.1349, 16.0965) | $\{0, 1, 2, 3, 4, 5\}$ | 6 |
| 4 | (26.3571, 10.7381) | (72.3209, -0.6093, 72.9709) | (0.01382823, 0.00011547, 0.01370507) | 4.2000 | (15.3044, 15.3730) | $\{0, 1, 2, 3, 4, 5\}$ | 6 |

$P^{(cam1)}=\sum_i K_i=24$.

### Camera 2

| $i$ | $\mu'_i$ (px) | $\Sigma'_{11},\Sigma'_{12},\Sigma'_{22}$ | $(A,B,C)$ | depth | half$_x$,half$_y$ | $\mathcal K_i$ (tile) | $K_i$ |
|---|---|---|---|---|---|---|---|
| 1 | (8.5000, 15.5000) | (82.8052, -0.0000, 72.6333) | (0.01207654, 0.00000000, 0.01376778) | 4.0000 | (16.3762, 15.3374) | $\{0, 1, 3, 4\}$ | 4 |
| 2 | (14.6111, 18.1667) | (74.0936, -1.0418, 70.9335) | (0.01349922, 0.00019826, 0.01410062) | 4.5000 | (15.4908, 15.1569) | $\{0, 1, 3, 4\}$ | 4 |
| 3 | (8.3000, 13.9000) | (91.3637, 1.2095, 80.0007) | (0.01094745, -0.00016551, 0.01250240) | 5.0000 | (17.2017, 16.0965) | $\{0, 1, 3, 4\}$ | 4 |
| 4 | (12.0714, 10.7381) | (77.8047, 2.4373, 72.9709) | (0.01286615, -0.00042973, 0.01371846) | 4.2000 | (15.8740, 15.3730) | $\{0, 1, 3, 4\}$ | 4 |

$P^{(cam2)}=\sum_i K_i=16$.

### Camera 3

| $i$ | $\mu'_i$ (px) | $\Sigma'_{11},\Sigma'_{12},\Sigma'_{22}$ | $(A,B,C)$ | depth | half$_x$,half$_y$ | $\mathcal K_i$ (tile) | $K_i$ |
|---|---|---|---|---|---|---|---|
| 1 | (38.5000, 10.5000) | (82.8052, -3.3906, 73.7635) | (0.01209931, 0.00055616, 0.01358240) | 4.0000 | (16.3762, 15.4563) | $\{1, 2, 4, 5\}$ | 4 |
| 2 | (41.2778, 13.7222) | (84.5116, -1.3891, 70.7599) | (0.01183652, 0.00023236, 0.01413686) | 4.5000 | (16.5441, 15.1383) | $\{1, 2, 4, 5\}$ | 4 |
| 3 | (32.3000, 9.9000) | (83.7247, -2.4509, 81.4330) | (0.01195444, 0.00035979, 0.01229087) | 5.0000 | (16.4669, 16.2399) | $\{0, 1, 2, 4, 5\}$ | 5 |
| 4 | (40.6429, 5.9762) | (85.1165, -7.3118, 76.0174) | (0.01184648, 0.00113946, 0.01326448) | 4.2000 | (16.6032, 15.6906) | $\{1, 2, 4, 5\}$ | 4 |

$P^{(cam3)}=\sum_i K_i=17$.

Ba bảng trên là **chép nguyên văn** kết quả Phần 3 (không tính lại) — tuân thủ quy ước §16.4.1: lời giải
không phát sinh số liệu mới mâu thuẫn với chương gốc; các giá trị $\mu', \Sigma', A, B, C$ khớp đến sai số
làm tròn với bảng mục 8.2 (camera 1) và bảng "Camera 2, 3 (gọn)" của Phần 3.

---

## 4.1 — Sinh khoá 64-bit và radix sort cho cả 3 camera

Công thức (chép nguyên từ chương 9, `auxiliary.h:278-280`):

$$\boxed{\ \text{key}(T,i)=\bigl(\text{id}(T)\ll32\bigr)\ \big|\ \text{bit\_cast}_{u32}(\text{depth}_i)\ }$$

32 bit cao là id tile $T$ (gom mọi entry cùng tile lại với nhau sau khi sort); 32 bit thấp là
`bit_cast` của `depth` (float32) sang `uint32` — với `depth`$>0$ (đảm bảo bởi cull $t_z>0.2$, chương 8 §3.2),
thứ tự uint32 trùng thứ tự float, nên sort tăng dần trên khoá 64-bit đồng thời sort theo tile rồi theo
độ sâu tăng dần **trong** mỗi tile — đúng thứ tự front-to-back mà alpha-blending cần.

Mỗi Gaussian $i$ ghi $K_i$ bản sao của khoá — một bản mỗi tile trong $\mathcal K_i$ — vào buffer có độ dài
$P=\sum_iK_i$, tại vị trí $\text{offset}_i+k$ ($\text{offset}_i=\sum_{j<i}K_j$ từ `InclusiveSum`, $k$ là chỉ số
tile trong $\mathcal K_i$). Vì thứ tự tile trong $\mathcal K_i$ đã tăng dần (script/`processTiles` quét tile
theo id tăng), thứ tự ghi vào buffer là: mọi tile của $G_1$, rồi mọi tile của $G_2$, ... theo đúng thứ tự
$i=1,2,3,4$ (không phải thứ tự depth) — đây chính là thứ tự **trước khi sort**, bảng dưới liệt kê nguyên văn.

### Camera 1 — buffer trước sort ($P=24$)

| vị trí | tile $T$ | $i$ | depth (float32) | bit_cast (hex) | key (hex 64-bit) |
|---|---|---|---|---|---|
| 0 | 0 | 1 | 4 | 0x40800000 | 0x0000000040800000 |
| 1 | 1 | 1 | 4 | 0x40800000 | 0x0000000140800000 |
| 2 | 2 | 1 | 4 | 0x40800000 | 0x0000000240800000 |
| 3 | 3 | 1 | 4 | 0x40800000 | 0x0000000340800000 |
| 4 | 4 | 1 | 4 | 0x40800000 | 0x0000000440800000 |
| 5 | 5 | 1 | 4 | 0x40800000 | 0x0000000540800000 |
| 6 | 0 | 2 | 4.5 | 0x40900000 | 0x0000000040900000 |
| 7 | 1 | 2 | 4.5 | 0x40900000 | 0x0000000140900000 |
| 8 | 2 | 2 | 4.5 | 0x40900000 | 0x0000000240900000 |
| 9 | 3 | 2 | 4.5 | 0x40900000 | 0x0000000340900000 |
| 10 | 4 | 2 | 4.5 | 0x40900000 | 0x0000000440900000 |
| 11 | 5 | 2 | 4.5 | 0x40900000 | 0x0000000540900000 |
| 12 | 0 | 3 | 5 | 0x40A00000 | 0x0000000040A00000 |
| 13 | 1 | 3 | 5 | 0x40A00000 | 0x0000000140A00000 |
| 14 | 2 | 3 | 5 | 0x40A00000 | 0x0000000240A00000 |
| 15 | 3 | 3 | 5 | 0x40A00000 | 0x0000000340A00000 |
| 16 | 4 | 3 | 5 | 0x40A00000 | 0x0000000440A00000 |
| 17 | 5 | 3 | 5 | 0x40A00000 | 0x0000000540A00000 |
| 18 | 0 | 4 | 4.2 | 0x40866666 | 0x0000000040866666 |
| 19 | 1 | 4 | 4.2 | 0x40866666 | 0x0000000140866666 |
| 20 | 2 | 4 | 4.2 | 0x40866666 | 0x0000000240866666 |
| 21 | 3 | 4 | 4.2 | 0x40866666 | 0x0000000340866666 |
| 22 | 4 | 4 | 4.2 | 0x40866666 | 0x0000000440866666 |
| 23 | 5 | 4 | 4.2 | 0x40866666 | 0x0000000540866666 |

### Camera 2 — buffer trước sort ($P=16$)

| vị trí | tile $T$ | $i$ | depth (float32) | bit_cast (hex) | key (hex 64-bit) |
|---|---|---|---|---|---|
| 0 | 0 | 1 | 4 | 0x40800000 | 0x0000000040800000 |
| 1 | 1 | 1 | 4 | 0x40800000 | 0x0000000140800000 |
| 2 | 3 | 1 | 4 | 0x40800000 | 0x0000000340800000 |
| 3 | 4 | 1 | 4 | 0x40800000 | 0x0000000440800000 |
| 4 | 0 | 2 | 4.5 | 0x40900000 | 0x0000000040900000 |
| 5 | 1 | 2 | 4.5 | 0x40900000 | 0x0000000140900000 |
| 6 | 3 | 2 | 4.5 | 0x40900000 | 0x0000000340900000 |
| 7 | 4 | 2 | 4.5 | 0x40900000 | 0x0000000440900000 |
| 8 | 0 | 3 | 5 | 0x40A00000 | 0x0000000040A00000 |
| 9 | 1 | 3 | 5 | 0x40A00000 | 0x0000000140A00000 |
| 10 | 3 | 3 | 5 | 0x40A00000 | 0x0000000340A00000 |
| 11 | 4 | 3 | 5 | 0x40A00000 | 0x0000000440A00000 |
| 12 | 0 | 4 | 4.2 | 0x40866666 | 0x0000000040866666 |
| 13 | 1 | 4 | 4.2 | 0x40866666 | 0x0000000140866666 |
| 14 | 3 | 4 | 4.2 | 0x40866666 | 0x0000000340866666 |
| 15 | 4 | 4 | 4.2 | 0x40866666 | 0x0000000440866666 |

### Camera 3 — buffer trước sort ($P=17$)

| vị trí | tile $T$ | $i$ | depth (float32) | bit_cast (hex) | key (hex 64-bit) |
|---|---|---|---|---|---|
| 0 | 1 | 1 | 4 | 0x40800000 | 0x0000000140800000 |
| 1 | 2 | 1 | 4 | 0x40800000 | 0x0000000240800000 |
| 2 | 4 | 1 | 4 | 0x40800000 | 0x0000000440800000 |
| 3 | 5 | 1 | 4 | 0x40800000 | 0x0000000540800000 |
| 4 | 1 | 2 | 4.5 | 0x40900000 | 0x0000000140900000 |
| 5 | 2 | 2 | 4.5 | 0x40900000 | 0x0000000240900000 |
| 6 | 4 | 2 | 4.5 | 0x40900000 | 0x0000000440900000 |
| 7 | 5 | 2 | 4.5 | 0x40900000 | 0x0000000540900000 |
| 8 | 0 | 3 | 5 | 0x40A00000 | 0x0000000040A00000 |
| 9 | 1 | 3 | 5 | 0x40A00000 | 0x0000000140A00000 |
| 10 | 2 | 3 | 5 | 0x40A00000 | 0x0000000240A00000 |
| 11 | 4 | 3 | 5 | 0x40A00000 | 0x0000000440A00000 |
| 12 | 5 | 3 | 5 | 0x40A00000 | 0x0000000540A00000 |
| 13 | 1 | 4 | 4.2 | 0x40866666 | 0x0000000140866666 |
| 14 | 2 | 4 | 4.2 | 0x40866666 | 0x0000000240866666 |
| 15 | 4 | 4 | 4.2 | 0x40866666 | 0x0000000440866666 |
| 16 | 5 | 4 | 4.2 | 0x40866666 | 0x0000000540866666 |

**Kiểm tra thay số** (camera 1, dòng đầu): $\text{depth}_1=4.0$ → IEEE-754 float32: dấu $0$, exponent $=129$
($10000001_2$), mantissa $=0$ → bit pattern `0x40800000`. Tile $T=0$ → `key`$=(0\ll32)\,|\,$`0x40800000`
$=$ `0x0000000040800000`. Với tile $T=1$: `key`$=(1\ll32)\,|\,$`0x40800000`$=$`0x0000000140800000` — khớp bảng.

### Camera 1 — sau radix sort (`point_list_keys`, `point_list`)

Số bit cần sort: $32+\lceil\log_2 6\rceil=32+3=35$ (`getHigherMsb(6)=3`, không đổi theo camera vì lưới
tile luôn $3\times2=6$). `np.argsort(kind='stable')` trên `uint64` cho cùng kết quả radix sort ổn định.

| idx | key (hex) | tile$=$key$\gg32$ | $i=$point_list[idx] | depth |
|---|---|---|---|---|
| 0 | 0x0000000040800000 | 0 | 1 | 4 |
| 1 | 0x0000000040866666 | 0 | 4 | 4.2 |
| 2 | 0x0000000040900000 | 0 | 2 | 4.5 |
| 3 | 0x0000000040A00000 | 0 | 3 | 5 |
| 4 | 0x0000000140800000 | 1 | 1 | 4 |
| 5 | 0x0000000140866666 | 1 | 4 | 4.2 |
| 6 | 0x0000000140900000 | 1 | 2 | 4.5 |
| 7 | 0x0000000140A00000 | 1 | 3 | 5 |
| 8 | 0x0000000240800000 | 2 | 1 | 4 |
| 9 | 0x0000000240866666 | 2 | 4 | 4.2 |
| 10 | 0x0000000240900000 | 2 | 2 | 4.5 |
| 11 | 0x0000000240A00000 | 2 | 3 | 5 |
| 12 | 0x0000000340800000 | 3 | 1 | 4 |
| 13 | 0x0000000340866666 | 3 | 4 | 4.2 |
| 14 | 0x0000000340900000 | 3 | 2 | 4.5 |
| 15 | 0x0000000340A00000 | 3 | 3 | 5 |
| 16 | 0x0000000440800000 | 4 | 1 | 4 |
| 17 | 0x0000000440866666 | 4 | 4 | 4.2 |
| 18 | 0x0000000440900000 | 4 | 2 | 4.5 |
| 19 | 0x0000000440A00000 | 4 | 3 | 5 |
| 20 | 0x0000000540800000 | 5 | 1 | 4 |
| 21 | 0x0000000540866666 | 5 | 4 | 4.2 |
| 22 | 0x0000000540900000 | 5 | 2 | 4.5 |
| 23 | 0x0000000540A00000 | 5 | 3 | 5 |

`identifyTileRanges`: tại idx mà tile đổi so với idx−1 → đóng `ranges[prev].y=idx` và mở `ranges[curr].x=idx`;
idx$=0$ → `ranges[tile(0)].x=0`; idx$=L-1$ → `ranges[tile(L-1)].y=L`. **Tile không có Gaussian nào** (ví dụ
tile 2, 5 ở camera 2) không xuất hiện trong buffer nên `ranges[T]=[0,0]` (rỗng) — mọi pixel trong tile đó
là nền theo định nghĩa, không cần duyệt vòng lặp Gaussian nào.

| tile $T$ | $[\text{start},\text{end})$ | $|\mathcal G_T|$ | $\mathcal G_T$ theo depth tăng dần |
|---|---|---|---|
| 0 | [0, 4) | 4 | [1, 4, 2, 3] |
| 1 | [4, 8) | 4 | [1, 4, 2, 3] |
| 2 | [8, 12) | 4 | [1, 4, 2, 3] |
| 3 | [12, 16) | 4 | [1, 4, 2, 3] |
| 4 | [16, 20) | 4 | [1, 4, 2, 3] |
| 5 | [20, 24) | 4 | [1, 4, 2, 3] |

### Camera 2 — sau radix sort (`point_list_keys`, `point_list`)

Số bit cần sort: $32+\lceil\log_2 6\rceil=32+3=35$ (`getHigherMsb(6)=3`, không đổi theo camera vì lưới
tile luôn $3\times2=6$). `np.argsort(kind='stable')` trên `uint64` cho cùng kết quả radix sort ổn định.

| idx | key (hex) | tile$=$key$\gg32$ | $i=$point_list[idx] | depth |
|---|---|---|---|---|
| 0 | 0x0000000040800000 | 0 | 1 | 4 |
| 1 | 0x0000000040866666 | 0 | 4 | 4.2 |
| 2 | 0x0000000040900000 | 0 | 2 | 4.5 |
| 3 | 0x0000000040A00000 | 0 | 3 | 5 |
| 4 | 0x0000000140800000 | 1 | 1 | 4 |
| 5 | 0x0000000140866666 | 1 | 4 | 4.2 |
| 6 | 0x0000000140900000 | 1 | 2 | 4.5 |
| 7 | 0x0000000140A00000 | 1 | 3 | 5 |
| 8 | 0x0000000340800000 | 3 | 1 | 4 |
| 9 | 0x0000000340866666 | 3 | 4 | 4.2 |
| 10 | 0x0000000340900000 | 3 | 2 | 4.5 |
| 11 | 0x0000000340A00000 | 3 | 3 | 5 |
| 12 | 0x0000000440800000 | 4 | 1 | 4 |
| 13 | 0x0000000440866666 | 4 | 4 | 4.2 |
| 14 | 0x0000000440900000 | 4 | 2 | 4.5 |
| 15 | 0x0000000440A00000 | 4 | 3 | 5 |

`identifyTileRanges`: tại idx mà tile đổi so với idx−1 → đóng `ranges[prev].y=idx` và mở `ranges[curr].x=idx`;
idx$=0$ → `ranges[tile(0)].x=0`; idx$=L-1$ → `ranges[tile(L-1)].y=L`. **Tile không có Gaussian nào** (ví dụ
tile 2, 5 ở camera 2) không xuất hiện trong buffer nên `ranges[T]=[0,0]` (rỗng) — mọi pixel trong tile đó
là nền theo định nghĩa, không cần duyệt vòng lặp Gaussian nào.

| tile $T$ | $[\text{start},\text{end})$ | $|\mathcal G_T|$ | $\mathcal G_T$ theo depth tăng dần |
|---|---|---|---|
| 0 | [0, 4) | 4 | [1, 4, 2, 3] |
| 1 | [4, 8) | 4 | [1, 4, 2, 3] |
| 2 | [·, ·) — rỗng | 0 | [∅ (toàn tile là nền)] |
| 3 | [8, 12) | 4 | [1, 4, 2, 3] |
| 4 | [12, 16) | 4 | [1, 4, 2, 3] |
| 5 | [·, ·) — rỗng | 0 | [∅ (toàn tile là nền)] |

### Camera 3 — sau radix sort (`point_list_keys`, `point_list`)

Số bit cần sort: $32+\lceil\log_2 6\rceil=32+3=35$ (`getHigherMsb(6)=3`, không đổi theo camera vì lưới
tile luôn $3\times2=6$). `np.argsort(kind='stable')` trên `uint64` cho cùng kết quả radix sort ổn định.

| idx | key (hex) | tile$=$key$\gg32$ | $i=$point_list[idx] | depth |
|---|---|---|---|---|
| 0 | 0x0000000040A00000 | 0 | 3 | 5 |
| 1 | 0x0000000140800000 | 1 | 1 | 4 |
| 2 | 0x0000000140866666 | 1 | 4 | 4.2 |
| 3 | 0x0000000140900000 | 1 | 2 | 4.5 |
| 4 | 0x0000000140A00000 | 1 | 3 | 5 |
| 5 | 0x0000000240800000 | 2 | 1 | 4 |
| 6 | 0x0000000240866666 | 2 | 4 | 4.2 |
| 7 | 0x0000000240900000 | 2 | 2 | 4.5 |
| 8 | 0x0000000240A00000 | 2 | 3 | 5 |
| 9 | 0x0000000440800000 | 4 | 1 | 4 |
| 10 | 0x0000000440866666 | 4 | 4 | 4.2 |
| 11 | 0x0000000440900000 | 4 | 2 | 4.5 |
| 12 | 0x0000000440A00000 | 4 | 3 | 5 |
| 13 | 0x0000000540800000 | 5 | 1 | 4 |
| 14 | 0x0000000540866666 | 5 | 4 | 4.2 |
| 15 | 0x0000000540900000 | 5 | 2 | 4.5 |
| 16 | 0x0000000540A00000 | 5 | 3 | 5 |

`identifyTileRanges`: tại idx mà tile đổi so với idx−1 → đóng `ranges[prev].y=idx` và mở `ranges[curr].x=idx`;
idx$=0$ → `ranges[tile(0)].x=0`; idx$=L-1$ → `ranges[tile(L-1)].y=L`. **Tile không có Gaussian nào** (ví dụ
tile 2, 5 ở camera 2) không xuất hiện trong buffer nên `ranges[T]=[0,0]` (rỗng) — mọi pixel trong tile đó
là nền theo định nghĩa, không cần duyệt vòng lặp Gaussian nào.

| tile $T$ | $[\text{start},\text{end})$ | $|\mathcal G_T|$ | $\mathcal G_T$ theo depth tăng dần |
|---|---|---|---|
| 0 | [0, 1) | 1 | [3] |
| 1 | [1, 5) | 4 | [1, 4, 2, 3] |
| 2 | [5, 9) | 4 | [1, 4, 2, 3] |
| 3 | [·, ·) — rỗng | 0 | [∅ (toàn tile là nền)] |
| 4 | [9, 13) | 4 | [1, 4, 2, 3] |
| 5 | [13, 17) | 4 | [1, 4, 2, 3] |

---

## 4.2 — Công thức alpha tại pixel và alpha-blending front-to-back

Với pixel nguyên $x=(u,v)$ thuộc tile $T=\lfloor v/16\rfloor\cdot3+\lfloor u/16\rfloor$, duyệt
$n\in\mathcal G_T$ theo `point_list[ranges[T].x .. ranges[T].y)` (đã sort theo depth tăng dần ở §4.1).
Code tính $d=\mu'_n-\text{pixf}$ (tức $d=-\Delta$ nếu định nghĩa $\Delta=x-\mu'_n$); vì dạng toàn phương
đối xứng, $\Delta^\top M_n\Delta=d^\top M_n d$ nên công thức tương đương nhau, chọn $\Delta=x-\mu'_n$
(quy ước sách) và $d_x=\mu'_{n,x}-u,\ d_y=\mu'_{n,y}-v$ khi lập trình đều cho cùng kết quả:

$$\text{power}_n(x)=-\tfrac12\bigl(A_n\Delta_u^2+2B_n\Delta_u\Delta_v+C_n\Delta_v^2\bigr)=-\tfrac12\Delta^\top M_n\Delta$$

$$G_n(x)=e^{\text{power}_n(x)},\qquad\boxed{\ \alpha_n(x)=\min\bigl(0.99,\ \alpha_i\,G_n(x)\bigr)\ }$$

Ba cửa loại, đúng thứ tự trong `renderCUDA`:

| # | Điều kiện | Hành động | Ý nghĩa |
|---|---|---|---|
| 1 | $\text{power}_n>0$ | `continue` (bỏ splat) | $M_n$ không PSD tại pixel này (sai số/ngoài ellipse) |
| 2 | $\alpha_n(x)<1/255$ | `continue` (bỏ splat) | dưới bước lượng tử 8-bit — cùng ngưỡng với compact box |
| 3 | $T_n(1-\alpha_n(x))<10^{-4}$ | `break` (dừng cả pixel) | early termination |

Alpha-blending front-to-back:

$$T_1=1,\qquad T_{n+1}=T_n\bigl(1-\alpha_n(x)\bigr)$$

$$\boxed{\ C(x)=\sum_{n\in\mathcal G_T}c_n\,\alpha_n(x)\,T_n\;+\;T_{\text{final}}\cdot C_{\text{bg}}\ },
\qquad C_{\text{bg}}=(1,1,1)$$

Dưới đây áp dụng **đúng chuỗi phép toán này, từng bước một** cho 144 pixel đại diện — 8 pixel
(tâm tile, 4 góc, 3 điểm giữa cạnh) cho mỗi tile trong số 6 tile, lặp lại cho cả 3 camera
($8\times6\times3=144$, vượt yêu cầu tối thiểu $2\times6\times3=36$). Cách chọn này đảm bảo **mọi**
tile ở **mọi** camera đều có nhiều pixel được tính tay đầy đủ, kể cả tile rỗng (chỉ có nền).

---

## 4.3 — 144 pixel tính tay đầy đủ (8 pixel × 6 tile × 3 camera)

Ngoài yêu cầu tối thiểu 2 pixel/tile/camera, mỗi tile được minh hoạ bằng **8** vị trí đặc trưng để lộ rõ
cách $\alpha_n(x)$ thay đổi theo khoảng cách tới từng tâm splat: tâm hình học của tile, bốn góc (cách
biên tile 1–2 px để tránh trùng đúng biên) và ba điểm giữa cạnh (trên, dưới, trái) — đủ để thấy splat
nào "vừa chạm" hay "vừa mất tác dụng" khi đi từ trong tile ra ngoài theo từng hướng.

### Camera 1

#### Tile 0, pixel $x=(8,8)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (15.5000, 7.5000) | -2.041074 | 0.129889 | 0.012989 | 1.000000 | cộng | (0.010391, 0.002598, 0.002598) | 0.987011 |
| 2 | 4 | (0.5, 0.5, 0.5) | (18.3571, 2.7381) | -2.387129 | 0.091893 | 0.009189 | 0.987011 | cộng | (0.004535, 0.004535, 0.004535) | 0.977941 |
| 3 | 2 | (0.2, 0.7, 0.3) | (19.9444, 10.1667) | -3.490042 | 0.030500 | 0.003050 | 0.977941 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.977941 |
| 4 | 3 | (0.1, 0.3, 0.9) | (12.3000, 5.9000) | -1.155761 | 0.314818 | 0.031482 | 0.977941 | cộng | (0.003079, 0.009236, 0.027709) | 0.947154 |

$T_{final}=0.947154$, $\sum_n c_n\alpha_nT_n=(0.018005, 0.016369, 0.034841)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.965159, 0.963523, 0.981995)}\ \to\ \text{8-bit }(246, 246, 250)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0315$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(1,1)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (22.5000, 14.5000) | -4.932308 | 0.007210 | 0.000721 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (25.3571, 9.7381) | -5.124012 | 0.005952 | 0.000595 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (26.9444, 17.1667) | -7.107840 | 0.000819 | 0.000082 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (19.3000, 12.9000) | -3.347211 | 0.035182 | 0.003518 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/4 Gaussian thực sự đóng góp (4 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0007$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(14,1)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (9.5000, 14.5000) | -2.068609 | 0.126361 | 0.012636 | 1.000000 | cộng | (0.010109, 0.002527, 0.002527) | 0.987364 |
| 2 | 4 | (0.5, 0.5, 0.5) | (12.3571, 9.7381) | -1.719502 | 0.179155 | 0.017916 | 0.987364 | cộng | (0.008845, 0.008845, 0.008845) | 0.969675 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.9444, 17.1667) | -3.412830 | 0.032948 | 0.003295 | 0.969675 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.969675 |
| 4 | 3 | (0.1, 0.3, 0.9) | (6.3000, 12.9000) | -1.283731 | 0.277002 | 0.027700 | 0.969675 | cộng | (0.002686, 0.008058, 0.024174) | 0.942815 |

$T_{final}=0.942815$, $\sum_n c_n\alpha_nT_n=(0.021640, 0.019430, 0.035546)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.964454, 0.962244, 0.978360)}\ \to\ \text{8-bit }(246, 245, 249)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0277$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(1,14)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (22.5000, 1.5000) | -3.500459 | 0.030184 | 0.003018 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (25.3571, -3.2619) | -4.509031 | 0.011009 | 0.001101 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (26.9444, 4.1667) | -5.188837 | 0.005578 | 0.000558 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (19.3000, -0.1000) | -2.317144 | 0.098555 | 0.009855 | 1.000000 | cộng | (0.000986, 0.002957, 0.008870) | 0.990145 |

$T_{final}=0.990145$, $\sum_n c_n\alpha_nT_n=(0.000986, 0.002957, 0.008870)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.991130, 0.993101, 0.999014)}\ \to\ \text{8-bit }(253, 253, 255)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0099$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(14,14)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (9.5000, 1.5000) | -0.636760 | 0.529004 | 0.052900 | 1.000000 | cộng | (0.042320, 0.010580, 0.010580) | 0.947100 |
| 2 | 4 | (0.5, 0.5, 0.5) | (12.3571, -3.2619) | -1.124035 | 0.324966 | 0.032497 | 0.947100 | cộng | (0.015389, 0.015389, 0.015389) | 0.916322 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.9444, 4.1667) | -1.476466 | 0.228444 | 0.022844 | 0.916322 | cộng | (0.004187, 0.014653, 0.006280) | 0.895389 |
| 4 | 3 | (0.1, 0.3, 0.9) | (6.3000, -0.1000) | -0.246972 | 0.781163 | 0.078116 | 0.895389 | cộng | (0.006994, 0.020983, 0.062950) | 0.825445 |

$T_{final}=0.825445$, $\sum_n c_n\alpha_nT_n=(0.068890, 0.061605, 0.095199)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.894335, 0.887050, 0.920644)}\ \to\ \text{8-bit }(228, 226, 235)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0781$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(8,1)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (15.5000, 14.5000) | -3.101193 | 0.044995 | 0.004500 | 1.000000 | cộng | (0.003600, 0.000900, 0.000900) | 0.995500 |
| 2 | 4 | (0.5, 0.5, 0.5) | (18.3571, 9.7381) | -3.000422 | 0.049766 | 0.004977 | 0.995500 | cộng | (0.002477, 0.002477, 0.002477) | 0.990546 |
| 3 | 2 | (0.2, 0.7, 0.3) | (19.9444, 17.1667) | -4.824453 | 0.008031 | 0.000803 | 0.990546 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.990546 |
| 4 | 3 | (0.1, 0.3, 0.9) | (12.3000, 12.9000) | -1.974853 | 0.138782 | 0.013878 | 0.990546 | cộng | (0.001375, 0.004124, 0.012372) | 0.976799 |

$T_{final}=0.976799$, $\sum_n c_n\alpha_nT_n=(0.007451, 0.007501, 0.015749)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.984251, 0.984300, 0.992549)}\ \to\ \text{8-bit }(251, 251, 253)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0139$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(8,14)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (15.5000, 1.5000) | -1.669344 | 0.188371 | 0.018837 | 1.000000 | cộng | (0.015070, 0.003767, 0.003767) | 0.981163 |
| 2 | 4 | (0.5, 0.5, 0.5) | (18.3571, -3.2619) | -2.395948 | 0.091086 | 0.009109 | 0.981163 | cộng | (0.004469, 0.004469, 0.004469) | 0.972226 |
| 3 | 2 | (0.2, 0.7, 0.3) | (19.9444, 4.1667) | -2.896101 | 0.055238 | 0.005524 | 0.972226 | cộng | (0.001074, 0.003759, 0.001611) | 0.966855 |
| 4 | 3 | (0.1, 0.3, 0.9) | (12.3000, -0.1000) | -0.941183 | 0.390166 | 0.039017 | 0.966855 | cộng | (0.003772, 0.011317, 0.033951) | 0.929132 |

$T_{final}=0.929132$, $\sum_n c_n\alpha_nT_n=(0.024385, 0.023312, 0.043798)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.953517, 0.952444, 0.972930)}\ \to\ \text{8-bit }(243, 243, 248)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0390$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(1,8)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (22.5000, 7.5000) | -3.872189 | 0.020813 | 0.002081 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (25.3571, 2.7381) | -4.505062 | 0.011053 | 0.001105 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (26.9444, 10.1667) | -5.778463 | 0.003093 | 0.000309 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (19.3000, 5.9000) | -2.530059 | 0.079654 | 0.007965 | 1.000000 | cộng | (0.000797, 0.002390, 0.007169) | 0.992035 |

$T_{final}=0.992035$, $\sum_n c_n\alpha_nT_n=(0.000797, 0.002390, 0.007169)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.992831, 0.994424, 0.999203)}\ \to\ \text{8-bit }(253, 254, 255)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0080$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(24,8)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-0.5000, 7.5000) | -0.388940 | 0.677775 | 0.067778 | 1.000000 | cộng | (0.054222, 0.013556, 0.013556) | 0.932222 |
| 2 | 4 | (0.5, 0.5, 0.5) | (2.3571, 2.7381) | -0.090535 | 0.913442 | 0.091344 | 0.932222 | cộng | (0.042577, 0.042577, 0.042577) | 0.847069 |
| 3 | 2 | (0.2, 0.7, 0.3) | (3.9444, 10.1667) | -0.833321 | 0.434604 | 0.043460 | 0.847069 | cộng | (0.007363, 0.025770, 0.011044) | 0.810255 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-3.7000, 5.9000) | -0.303583 | 0.738168 | 0.073817 | 0.810255 | cộng | (0.005981, 0.017943, 0.053829) | 0.750445 |

$T_{final}=0.750445$, $\sum_n c_n\alpha_nT_n=(0.110142, 0.099845, 0.121006)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.860587, 0.850290, 0.871451)}\ \to\ \text{8-bit }(219, 217, 222)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0913$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(17,1)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (6.5000, 14.5000) | -1.738183 | 0.175840 | 0.017584 | 1.000000 | cộng | (0.014067, 0.003517, 0.003517) | 0.982416 |
| 2 | 4 | (0.5, 0.5, 0.5) | (9.3571, 9.7381) | -1.265724 | 0.282035 | 0.028204 | 0.982416 | cộng | (0.013854, 0.013854, 0.013854) | 0.954708 |
| 3 | 2 | (0.2, 0.7, 0.3) | (10.9444, 17.1667) | -2.895869 | 0.055251 | 0.005525 | 0.954708 | cộng | (0.001055, 0.003692, 0.001582) | 0.949434 |
| 4 | 3 | (0.1, 0.3, 0.9) | (3.3000, 12.9000) | -1.106118 | 0.330841 | 0.033084 | 0.949434 | cộng | (0.003141, 0.009423, 0.028270) | 0.918022 |

$T_{final}=0.918022$, $\sum_n c_n\alpha_nT_n=(0.032117, 0.030486, 0.047223)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.950140, 0.948509, 0.965246)}\ \to\ \text{8-bit }(242, 242, 246)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0282$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(30,1)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-6.5000, 14.5000) | -1.738183 | 0.175840 | 0.017584 | 1.000000 | cộng | (0.014067, 0.003517, 0.003517) | 0.982416 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-3.6429, 9.7381) | -0.737486 | 0.478315 | 0.047831 | 0.982416 | cộng | (0.023495, 0.023495, 0.023495) | 0.935426 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.0556, 17.1667) | -2.110548 | 0.121172 | 0.012117 | 0.935426 | cộng | (0.002267, 0.007934, 0.003400) | 0.924091 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-9.7000, 12.9000) | -1.630289 | 0.195873 | 0.019587 | 0.924091 | cộng | (0.001810, 0.005430, 0.016290) | 0.905990 |

$T_{final}=0.905990$, $\sum_n c_n\alpha_nT_n=(0.041639, 0.040376, 0.046703)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.947630, 0.946367, 0.952693)}\ \to\ \text{8-bit }(242, 241, 243)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0478$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(17,14)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (6.5000, 1.5000) | -0.306333 | 0.736141 | 0.073614 | 1.000000 | cộng | (0.058891, 0.014723, 0.014723) | 0.926386 |
| 2 | 4 | (0.5, 0.5, 0.5) | (9.3571, -3.2619) | -0.674760 | 0.509279 | 0.050928 | 0.926386 | cộng | (0.023589, 0.023589, 0.023589) | 0.879207 |
| 3 | 2 | (0.2, 0.7, 0.3) | (10.9444, 4.1667) | -0.955498 | 0.384620 | 0.038462 | 0.879207 | cộng | (0.006763, 0.023671, 0.010145) | 0.845391 |
| 4 | 3 | (0.1, 0.3, 0.9) | (3.3000, -0.1000) | -0.067815 | 0.934434 | 0.093443 | 0.845391 | cộng | (0.007900, 0.023699, 0.071097) | 0.766395 |

$T_{final}=0.766395$, $\sum_n c_n\alpha_nT_n=(0.097144, 0.085682, 0.119554)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.863538, 0.852077, 0.885948)}\ \to\ \text{8-bit }(220, 217, 226)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0934$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(30,14)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-6.5000, 1.5000) | -0.306333 | 0.736141 | 0.073614 | 1.000000 | cộng | (0.058891, 0.014723, 0.014723) | 0.926386 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-3.6429, -3.2619) | -0.166036 | 0.847015 | 0.084702 | 0.926386 | cộng | (0.039233, 0.039233, 0.039233) | 0.847920 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.0556, 4.1667) | -0.152816 | 0.858288 | 0.085829 | 0.847920 | cộng | (0.014555, 0.050943, 0.021833) | 0.775144 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-9.7000, -0.1000) | -0.585293 | 0.556942 | 0.055694 | 0.775144 | cộng | (0.004317, 0.012951, 0.038854) | 0.731973 |

$T_{final}=0.731973$, $\sum_n c_n\alpha_nT_n=(0.116997, 0.117850, 0.114643)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.848969, 0.849823, 0.846615)}\ \to\ \text{8-bit }(216, 217, 216)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0847$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(24,1)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-0.5000, 14.5000) | -1.449059 | 0.234791 | 0.023479 | 1.000000 | cộng | (0.018783, 0.004696, 0.004696) | 0.976521 |
| 2 | 4 | (0.5, 0.5, 0.5) | (2.3571, 9.7381) | -0.690895 | 0.501127 | 0.050113 | 0.976521 | cộng | (0.024468, 0.024468, 0.024468) | 0.927585 |
| 3 | 2 | (0.2, 0.7, 0.3) | (3.9444, 17.1667) | -2.179237 | 0.113128 | 0.011313 | 0.927585 | cộng | (0.002099, 0.007345, 0.003148) | 0.917091 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-3.7000, 12.9000) | -1.127111 | 0.323968 | 0.032397 | 0.917091 | cộng | (0.002971, 0.008913, 0.026740) | 0.887380 |

$T_{final}=0.887380$, $\sum_n c_n\alpha_nT_n=(0.048321, 0.045423, 0.059052)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.935702, 0.932803, 0.946432)}\ \to\ \text{8-bit }(239, 238, 241)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0501$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(24,14)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-0.5000, 1.5000) | -0.017210 | 0.982938 | 0.098294 | 1.000000 | cộng | (0.078635, 0.019659, 0.019659) | 0.901706 |
| 2 | 4 | (0.5, 0.5, 0.5) | (2.3571, -3.2619) | -0.110439 | 0.895441 | 0.089544 | 0.901706 | cộng | (0.040371, 0.040371, 0.040371) | 0.820964 |
| 3 | 2 | (0.2, 0.7, 0.3) | (3.9444, 4.1667) | -0.229518 | 0.794917 | 0.079492 | 0.820964 | cộng | (0.013052, 0.045682, 0.019578) | 0.755704 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-3.7000, -0.1000) | -0.085204 | 0.918325 | 0.091832 | 0.755704 | cộng | (0.006940, 0.020819, 0.062458) | 0.686306 |

$T_{final}=0.686306$, $\sum_n c_n\alpha_nT_n=(0.138998, 0.126531, 0.142066)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.825304, 0.812837, 0.828372)}\ \to\ \text{8-bit }(210, 207, 211)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0895$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(17,8)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (6.5000, 7.5000) | -0.678063 | 0.507599 | 0.050760 | 1.000000 | cộng | (0.040608, 0.010152, 0.010152) | 0.949240 |
| 2 | 4 | (0.5, 0.5, 0.5) | (9.3571, 2.7381) | -0.659706 | 0.517003 | 0.051700 | 0.949240 | cộng | (0.024538, 0.024538, 0.024538) | 0.900164 |
| 3 | 2 | (0.2, 0.7, 0.3) | (10.9444, 10.1667) | -1.554986 | 0.211192 | 0.021119 | 0.900164 | cộng | (0.003802, 0.013308, 0.005703) | 0.881153 |
| 4 | 3 | (0.1, 0.3, 0.9) | (3.3000, 5.9000) | -0.284531 | 0.752367 | 0.075237 | 0.881153 | cộng | (0.006630, 0.019889, 0.059666) | 0.814858 |

$T_{final}=0.814858$, $\sum_n c_n\alpha_nT_n=(0.075578, 0.067886, 0.100059)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.890436, 0.882744, 0.914917)}\ \to\ \text{8-bit }(227, 225, 233)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0752$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(40,8)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-16.5000, 7.5000) | -2.261358 | 0.104209 | 0.010421 | 1.000000 | cộng | (0.008337, 0.002084, 0.002084) | 0.989579 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-13.6429, 2.7381) | -1.333969 | 0.263430 | 0.026343 | 0.989579 | cộng | (0.013034, 0.013034, 0.013034) | 0.963511 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-12.0556, 10.1667) | -1.757754 | 0.172432 | 0.017243 | 0.963511 | cộng | (0.003323, 0.011630, 0.004984) | 0.946897 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-19.7000, 5.9000) | -2.636207 | 0.071632 | 0.007163 | 0.946897 | cộng | (0.000678, 0.002035, 0.006105) | 0.940114 |

$T_{final}=0.940114$, $\sum_n c_n\alpha_nT_n=(0.025372, 0.028783, 0.026207)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.965486, 0.968897, 0.966321)}\ \to\ \text{8-bit }(246, 247, 246)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0263$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(33,1)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-9.5000, 14.5000) | -2.068609 | 0.126361 | 0.012636 | 1.000000 | cộng | (0.010109, 0.002527, 0.002527) | 0.987364 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-6.6429, 9.7381) | -0.947463 | 0.387724 | 0.038772 | 0.987364 | cộng | (0.019141, 0.019141, 0.019141) | 0.949081 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-5.0556, 17.1667) | -2.265053 | 0.103825 | 0.010382 | 0.949081 | cộng | (0.001971, 0.006898, 0.002956) | 0.939228 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-12.7000, 12.9000) | -2.049826 | 0.128757 | 0.012876 | 0.939228 | cộng | (0.001209, 0.003628, 0.010884) | 0.927134 |

$T_{final}=0.927134$, $\sum_n c_n\alpha_nT_n=(0.032430, 0.032194, 0.035508)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.959565, 0.959328, 0.962643)}\ \to\ \text{8-bit }(245, 245, 245)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0388$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(46,1)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-22.5000, 14.5000) | -4.932308 | 0.007210 | 0.000721 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-19.6429, 9.7381) | -3.295497 | 0.037050 | 0.003705 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-18.0556, 17.1667) | -4.389419 | 0.012408 | 0.001241 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-25.7000, 12.9000) | -5.161648 | 0.005732 | 0.000573 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/4 Gaussian thực sự đóng góp (4 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0007$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(33,14)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-9.5000, 1.5000) | -0.636760 | 0.529004 | 0.052900 | 1.000000 | cộng | (0.042320, 0.010580, 0.010580) | 0.947100 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-6.6429, -3.2619) | -0.380516 | 0.683509 | 0.068351 | 0.947100 | cộng | (0.032368, 0.032368, 0.032368) | 0.882365 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-5.0556, 4.1667) | -0.303315 | 0.738367 | 0.073837 | 0.882365 | cộng | (0.013030, 0.045606, 0.019545) | 0.817214 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-12.7000, -0.1000) | -1.003287 | 0.366672 | 0.036667 | 0.817214 | cộng | (0.002996, 0.008989, 0.026968) | 0.787249 |

$T_{final}=0.787249$, $\sum_n c_n\alpha_nT_n=(0.090714, 0.097543, 0.089461)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.877963, 0.884791, 0.876710)}\ \to\ \text{8-bit }(224, 226, 224)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0684$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(46,14)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-22.5000, 1.5000) | -3.500459 | 0.030184 | 0.003018 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-19.6429, -3.2619) | -2.748064 | 0.064052 | 0.006405 | 1.000000 | cộng | (0.003203, 0.003203, 0.003203) | 0.993595 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-18.0556, 4.1667) | -2.410321 | 0.089786 | 0.008979 | 0.993595 | cộng | (0.001784, 0.006245, 0.002676) | 0.984674 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-25.7000, -0.1000) | -4.108416 | 0.016434 | 0.001643 | 0.984674 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.984674 |

$T_{final}=0.984674$, $\sum_n c_n\alpha_nT_n=(0.004987, 0.009447, 0.005879)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.989661, 0.994121, 0.990553)}\ \to\ \text{8-bit }(252, 254, 253)$$

*Nhận xét:* 2/4 Gaussian thực sự đóng góp (2 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0090$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(40,1)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-16.5000, 14.5000) | -3.321478 | 0.036099 | 0.003610 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-13.6429, 9.7381) | -1.921396 | 0.146402 | 0.014640 | 1.000000 | cộng | (0.007320, 0.007320, 0.007320) | 0.985360 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-12.0556, 17.1667) | -3.115176 | 0.044371 | 0.004437 | 0.985360 | cộng | (0.000874, 0.003060, 0.001312) | 0.980988 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-19.7000, 12.9000) | -3.464169 | 0.031299 | 0.003130 | 0.980988 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.980988 |

$T_{final}=0.980988$, $\sum_n c_n\alpha_nT_n=(0.008195, 0.010381, 0.008632)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.989182, 0.991368, 0.989619)}\ \to\ \text{8-bit }(252, 253, 252)$$

*Nhận xét:* 2/4 Gaussian thực sự đóng góp (2 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0146$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(40,14)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-16.5000, 1.5000) | -1.889628 | 0.151128 | 0.015113 | 1.000000 | cộng | (0.012090, 0.003023, 0.003023) | 0.984887 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-13.6429, -3.2619) | -1.364957 | 0.255392 | 0.025539 | 0.984887 | cộng | (0.012577, 0.012577, 0.012577) | 0.959734 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-12.0556, 4.1667) | -1.144090 | 0.318514 | 0.031851 | 0.959734 | cộng | (0.006114, 0.021398, 0.009171) | 0.929165 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-19.7000, -0.1000) | -2.414026 | 0.089454 | 0.008945 | 0.929165 | cộng | (0.000831, 0.002494, 0.007481) | 0.920853 |

$T_{final}=0.920853$, $\sum_n c_n\alpha_nT_n=(0.031612, 0.039491, 0.032250)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.952465, 0.960344, 0.953104)}\ \to\ \text{8-bit }(243, 245, 243)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0255$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(33,8)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-9.5000, 7.5000) | -1.008490 | 0.364769 | 0.036477 | 1.000000 | cộng | (0.029182, 0.007295, 0.007295) | 0.963523 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-6.6429, 2.7381) | -0.354377 | 0.701610 | 0.070161 | 0.963523 | cộng | (0.033801, 0.033801, 0.033801) | 0.895921 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-5.0556, 10.1667) | -0.912665 | 0.401453 | 0.040145 | 0.895921 | cộng | (0.007193, 0.025177, 0.010790) | 0.859954 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-12.7000, 5.9000) | -1.223804 | 0.294109 | 0.029411 | 0.859954 | cộng | (0.002529, 0.007588, 0.022763) | 0.834662 |

$T_{final}=0.834662$, $\sum_n c_n\alpha_nT_n=(0.072705, 0.073861, 0.074649)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.907367, 0.908523, 0.909311)}\ \to\ \text{8-bit }(231, 232, 232)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0702$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(8,24)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (15.5000, -8.5000) | -2.151216 | 0.116343 | 0.011634 | 1.000000 | cộng | (0.009307, 0.002327, 0.002327) | 0.988366 |
| 2 | 4 | (0.5, 0.5, 0.5) | (18.3571, -13.2619) | -3.507051 | 0.029985 | 0.002999 | 0.988366 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.988366 |
| 3 | 2 | (0.2, 0.7, 0.3) | (19.9444, -5.8333) | -3.034077 | 0.048119 | 0.004812 | 0.988366 | cộng | (0.000951, 0.003329, 0.001427) | 0.983610 |
| 4 | 3 | (0.1, 0.3, 0.9) | (12.3000, -10.1000) | -1.583554 | 0.205244 | 0.020524 | 0.983610 | cộng | (0.002019, 0.006056, 0.018169) | 0.963422 |

$T_{final}=0.963422$, $\sum_n c_n\alpha_nT_n=(0.012277, 0.011712, 0.021923)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.975699, 0.975134, 0.985345)}\ \to\ \text{8-bit }(249, 249, 251)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0205$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(1,17)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (22.5000, -1.5000) | -3.500459 | 0.030184 | 0.003018 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (25.3571, -6.2619) | -4.696034 | 0.009131 | 0.000913 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (26.9444, 1.1667) | -5.084353 | 0.006193 | 0.000619 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (19.3000, -3.1000) | -2.379437 | 0.092603 | 0.009260 | 1.000000 | cộng | (0.000926, 0.002778, 0.008334) | 0.990740 |

$T_{final}=0.990740$, $\sum_n c_n\alpha_nT_n=(0.000926, 0.002778, 0.008334)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.991666, 0.993518, 0.999074)}\ \to\ \text{8-bit }(253, 253, 255)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0093$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(14,17)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (9.5000, -1.5000) | -0.636760 | 0.529004 | 0.052900 | 1.000000 | cộng | (0.042320, 0.010580, 0.010580) | 0.947100 |
| 2 | 4 | (0.5, 0.5, 0.5) | (12.3571, -6.2619) | -1.315541 | 0.268329 | 0.026833 | 0.947100 | cộng | (0.012707, 0.012707, 0.012707) | 0.921686 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.9444, 1.1667) | -1.367976 | 0.254622 | 0.025462 | 0.921686 | cộng | (0.004694, 0.016428, 0.007040) | 0.898218 |
| 4 | 3 | (0.1, 0.3, 0.9) | (6.3000, -3.1000) | -0.307720 | 0.735121 | 0.073512 | 0.898218 | cộng | (0.006603, 0.019809, 0.059427) | 0.832188 |

$T_{final}=0.832188$, $\sum_n c_n\alpha_nT_n=(0.066324, 0.059523, 0.089754)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.898512, 0.891712, 0.921942)}\ \to\ \text{8-bit }(229, 227, 235)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0735$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(1,30)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (22.5000, -14.5000) | -4.932308 | 0.007210 | 0.000721 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (25.3571, -19.2619) | -6.931707 | 0.000976 | 0.000098 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (26.9444, -11.8333) | -6.097829 | 0.002248 | 0.000225 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (19.3000, -16.1000) | -3.949375 | 0.019267 | 0.001927 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/4 Gaussian thực sự đóng góp (4 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0007$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(14,30)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (9.5000, -14.5000) | -2.068609 | 0.126361 | 0.012636 | 1.000000 | cộng | (0.010109, 0.002527, 0.002527) | 0.987364 |
| 2 | 4 | (0.5, 0.5, 0.5) | (12.3571, -19.2619) | -3.570728 | 0.028135 | 0.002814 | 0.987364 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.987364 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.9444, -11.8333) | -2.364091 | 0.094035 | 0.009403 | 0.987364 | cộng | (0.001857, 0.006499, 0.002785) | 0.978079 |
| 4 | 3 | (0.1, 0.3, 0.9) | (6.3000, -16.1000) | -1.870966 | 0.153975 | 0.015397 | 0.978079 | cộng | (0.001506, 0.004518, 0.013554) | 0.963019 |

$T_{final}=0.963019$, $\sum_n c_n\alpha_nT_n=(0.013472, 0.013544, 0.018867)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.976491, 0.976564, 0.981886)}\ \to\ \text{8-bit }(249, 249, 250)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0154$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(8,17)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (15.5000, -1.5000) | -1.669344 | 0.188371 | 0.018837 | 1.000000 | cộng | (0.015070, 0.003767, 0.003767) | 0.981163 |
| 2 | 4 | (0.5, 0.5, 0.5) | (18.3571, -6.2619) | -2.585376 | 0.075368 | 0.007537 | 0.981163 | cộng | (0.003697, 0.003697, 0.003697) | 0.973768 |
| 3 | 2 | (0.2, 0.7, 0.3) | (19.9444, 1.1667) | -2.789460 | 0.061454 | 0.006145 | 0.973768 | cộng | (0.001197, 0.004189, 0.001795) | 0.967784 |
| 4 | 3 | (0.1, 0.3, 0.9) | (12.3000, -3.1000) | -1.002644 | 0.366908 | 0.036691 | 0.967784 | cộng | (0.003551, 0.010653, 0.031958) | 0.932275 |

$T_{final}=0.932275$, $\sum_n c_n\alpha_nT_n=(0.023515, 0.022306, 0.041218)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.955790, 0.954582, 0.973493)}\ \to\ \text{8-bit }(244, 243, 248)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0367$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(8,30)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (15.5000, -14.5000) | -3.101193 | 0.044995 | 0.004500 | 1.000000 | cộng | (0.003600, 0.000900, 0.000900) | 0.995500 |
| 2 | 4 | (0.5, 0.5, 0.5) | (18.3571, -19.2619) | -4.831556 | 0.007974 | 0.000797 | 0.995500 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.995500 |
| 3 | 2 | (0.2, 0.7, 0.3) | (19.9444, -11.8333) | -3.793588 | 0.022515 | 0.002251 | 0.995500 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.995500 |
| 4 | 3 | (0.1, 0.3, 0.9) | (12.3000, -16.1000) | -2.568978 | 0.076614 | 0.007661 | 0.995500 | cộng | (0.000763, 0.002288, 0.006864) | 0.987874 |

$T_{final}=0.987874$, $\sum_n c_n\alpha_nT_n=(0.004362, 0.003188, 0.007764)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.992236, 0.991062, 0.995638)}\ \to\ \text{8-bit }(253, 253, 254)$$

*Nhận xét:* 2/4 Gaussian thực sự đóng góp (2 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0077$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(1,24)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (22.5000, -8.5000) | -3.982331 | 0.018642 | 0.001864 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (25.3571, -13.2619) | -5.612051 | 0.003654 | 0.000365 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (26.9444, -5.8333) | -5.334003 | 0.004825 | 0.000482 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (19.3000, -10.1000) | -2.962287 | 0.051701 | 0.005170 | 1.000000 | cộng | (0.000517, 0.001551, 0.004653) | 0.994830 |

$T_{final}=0.994830$, $\sum_n c_n\alpha_nT_n=(0.000517, 0.001551, 0.004653)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.995347, 0.996381, 0.999483)}\ \to\ \text{8-bit }(254, 254, 255)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0052$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(24,24)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-0.5000, -8.5000) | -0.499082 | 0.607088 | 0.060709 | 1.000000 | cộng | (0.048567, 0.012142, 0.012142) | 0.939291 |
| 2 | 4 | (0.5, 0.5, 0.5) | (2.3571, -13.2619) | -1.240017 | 0.289379 | 0.028938 | 0.939291 | cộng | (0.013591, 0.013591, 0.013591) | 0.912110 |
| 3 | 2 | (0.2, 0.7, 0.3) | (3.9444, -5.8333) | -0.351057 | 0.703943 | 0.070394 | 0.912110 | cộng | (0.012841, 0.044945, 0.019262) | 0.847903 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-3.7000, -10.1000) | -0.721240 | 0.486149 | 0.048615 | 0.847903 | cộng | (0.004122, 0.012366, 0.037099) | 0.806682 |

$T_{final}=0.806682$, $\sum_n c_n\alpha_nT_n=(0.079121, 0.083044, 0.082093)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.885803, 0.889726, 0.888775)}\ \to\ \text{8-bit }(226, 227, 227)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0704$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(17,17)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (6.5000, -1.5000) | -0.306333 | 0.736141 | 0.073614 | 1.000000 | cộng | (0.058891, 0.014723, 0.014723) | 0.926386 |
| 2 | 4 | (0.5, 0.5, 0.5) | (9.3571, -6.2619) | -0.867305 | 0.420082 | 0.042008 | 0.926386 | cộng | (0.019458, 0.019458, 0.019458) | 0.887470 |
| 3 | 2 | (0.2, 0.7, 0.3) | (10.9444, 1.1667) | -0.846084 | 0.429092 | 0.042909 | 0.887470 | cộng | (0.007616, 0.026656, 0.011424) | 0.849389 |
| 4 | 3 | (0.1, 0.3, 0.9) | (3.3000, -3.1000) | -0.128207 | 0.879671 | 0.087967 | 0.849389 | cộng | (0.007472, 0.022416, 0.067247) | 0.774671 |

$T_{final}=0.774671$, $\sum_n c_n\alpha_nT_n=(0.093437, 0.083253, 0.112851)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.868108, 0.857924, 0.887523)}\ \to\ \text{8-bit }(221, 219, 226)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0880$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(30,17)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-6.5000, -1.5000) | -0.306333 | 0.736141 | 0.073614 | 1.000000 | cộng | (0.058891, 0.014723, 0.014723) | 0.926386 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-3.6429, -6.2619) | -0.363085 | 0.695527 | 0.069553 | 0.926386 | cộng | (0.032216, 0.032216, 0.032216) | 0.861953 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.0556, 1.1667) | -0.039395 | 0.961371 | 0.096137 | 0.861953 | cộng | (0.016573, 0.058006, 0.024860) | 0.779088 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-9.7000, -3.1000) | -0.644141 | 0.525113 | 0.052511 | 0.779088 | cộng | (0.004091, 0.012273, 0.036820) | 0.738177 |

$T_{final}=0.738177$, $\sum_n c_n\alpha_nT_n=(0.111772, 0.117218, 0.108619)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.849948, 0.855395, 0.846795)}\ \to\ \text{8-bit }(217, 218, 216)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0961$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(17,30)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (6.5000, -14.5000) | -1.738183 | 0.175840 | 0.017584 | 1.000000 | cộng | (0.014067, 0.003517, 0.003517) | 0.982416 |
| 2 | 4 | (0.5, 0.5, 0.5) | (9.3571, -19.2619) | -3.126995 | 0.043849 | 0.004385 | 0.982416 | cộng | (0.002154, 0.002154, 0.002154) | 0.978108 |
| 3 | 2 | (0.2, 0.7, 0.3) | (10.9444, -11.8333) | -1.838192 | 0.159105 | 0.015910 | 0.978108 | cộng | (0.003112, 0.010894, 0.004669) | 0.962546 |
| 4 | 3 | (0.1, 0.3, 0.9) | (3.3000, -16.1000) | -1.689909 | 0.184536 | 0.018454 | 0.962546 | cộng | (0.001776, 0.005329, 0.015986) | 0.944784 |

$T_{final}=0.944784$, $\sum_n c_n\alpha_nT_n=(0.021110, 0.021893, 0.026326)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.965893, 0.966677, 0.971109)}\ \to\ \text{8-bit }(246, 247, 248)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0185$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(30,30)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-6.5000, -14.5000) | -1.738183 | 0.175840 | 0.017584 | 1.000000 | cộng | (0.014067, 0.003517, 0.003517) | 0.982416 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-3.6429, -19.2619) | -2.642289 | 0.071198 | 0.007120 | 0.982416 | cộng | (0.003497, 0.003497, 0.003497) | 0.975421 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.0556, -11.8333) | -1.014143 | 0.362713 | 0.036271 | 0.975421 | cộng | (0.007076, 0.024766, 0.010614) | 0.940042 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-9.7000, -16.1000) | -2.199151 | 0.110897 | 0.011090 | 0.940042 | cộng | (0.001042, 0.003127, 0.009382) | 0.929617 |

$T_{final}=0.929617$, $\sum_n c_n\alpha_nT_n=(0.025683, 0.034907, 0.027010)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.955300, 0.964524, 0.956627)}\ \to\ \text{8-bit }(244, 246, 244)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0363$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(24,17)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-0.5000, -1.5000) | -0.017210 | 0.982938 | 0.098294 | 1.000000 | cộng | (0.078635, 0.019659, 0.019659) | 0.901706 |
| 2 | 4 | (0.5, 0.5, 0.5) | (2.3571, -6.2619) | -0.305409 | 0.736822 | 0.073682 | 0.901706 | cộng | (0.033220, 0.033220, 0.033220) | 0.835267 |
| 3 | 2 | (0.2, 0.7, 0.3) | (3.9444, 1.1667) | -0.117946 | 0.888744 | 0.088874 | 0.835267 | cộng | (0.014847, 0.051964, 0.022270) | 0.761033 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-3.7000, -3.1000) | -0.144765 | 0.865226 | 0.086523 | 0.761033 | cộng | (0.006585, 0.019754, 0.059262) | 0.695186 |

$T_{final}=0.695186$, $\sum_n c_n\alpha_nT_n=(0.133286, 0.124596, 0.134411)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.828472, 0.819782, 0.829597)}\ \to\ \text{8-bit }(211, 209, 212)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0983$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(24,30)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-0.5000, -14.5000) | -1.449059 | 0.234791 | 0.023479 | 1.000000 | cộng | (0.018783, 0.004696, 0.004696) | 0.976521 |
| 2 | 4 | (0.5, 0.5, 0.5) | (2.3571, -19.2619) | -2.575607 | 0.076108 | 0.007611 | 0.976521 | cộng | (0.003716, 0.003716, 0.003716) | 0.969089 |
| 3 | 2 | (0.2, 0.7, 0.3) | (3.9444, -11.8333) | -1.100707 | 0.332636 | 0.033264 | 0.969089 | cộng | (0.006447, 0.022565, 0.009671) | 0.936853 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-3.7000, -16.1000) | -1.702863 | 0.182161 | 0.018216 | 0.936853 | cộng | (0.001707, 0.005120, 0.015359) | 0.919788 |

$T_{final}=0.919788$, $\sum_n c_n\alpha_nT_n=(0.030653, 0.036096, 0.033442)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.950441, 0.955884, 0.953229)}\ \to\ \text{8-bit }(242, 244, 243)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0333$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(17,24)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (6.5000, -8.5000) | -0.788206 | 0.454660 | 0.045466 | 1.000000 | cộng | (0.036373, 0.009093, 0.009093) | 0.954534 |
| 2 | 4 | (0.5, 0.5, 0.5) | (9.3571, -13.2619) | -1.796255 | 0.165919 | 0.016592 | 0.954534 | cộng | (0.007919, 0.007919, 0.007919) | 0.938696 |
| 3 | 2 | (0.2, 0.7, 0.3) | (10.9444, -5.8333) | -1.084228 | 0.338163 | 0.033816 | 0.938696 | cộng | (0.006349, 0.022220, 0.009523) | 0.906953 |
| 4 | 3 | (0.1, 0.3, 0.9) | (3.3000, -10.1000) | -0.706623 | 0.493307 | 0.049331 | 0.906953 | cộng | (0.004474, 0.013422, 0.040267) | 0.862213 |

$T_{final}=0.862213$, $\sum_n c_n\alpha_nT_n=(0.055114, 0.052654, 0.066802)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.917327, 0.914867, 0.929014)}\ \to\ \text{8-bit }(234, 233, 237)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0493$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(40,24)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-16.5000, -8.5000) | -2.371501 | 0.093341 | 0.009334 | 1.000000 | cộng | (0.007467, 0.001867, 0.001867) | 0.990666 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-13.6429, -13.2619) | -2.513010 | 0.081024 | 0.008102 | 0.990666 | cộng | (0.004013, 0.004013, 0.004013) | 0.982639 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-12.0556, -5.8333) | -1.249193 | 0.286736 | 0.028674 | 0.982639 | cộng | (0.005635, 0.019723, 0.008453) | 0.954463 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-19.7000, -10.1000) | -3.043727 | 0.047657 | 0.004766 | 0.954463 | cộng | (0.000455, 0.001365, 0.004094) | 0.949915 |

$T_{final}=0.949915$, $\sum_n c_n\alpha_nT_n=(0.017571, 0.026968, 0.018427)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.967485, 0.976883, 0.968341)}\ \to\ \text{8-bit }(247, 249, 247)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0287$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(33,17)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-9.5000, -1.5000) | -0.636760 | 0.529004 | 0.052900 | 1.000000 | cộng | (0.042320, 0.010580, 0.010580) | 0.947100 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-6.6429, -6.2619) | -0.578604 | 0.560681 | 0.056068 | 0.947100 | cộng | (0.026551, 0.026551, 0.026551) | 0.893998 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-5.0556, 1.1667) | -0.188969 | 0.827812 | 0.082781 | 0.893998 | cộng | (0.014801, 0.051804, 0.022202) | 0.819991 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-12.7000, -3.1000) | -1.061778 | 0.345840 | 0.034584 | 0.819991 | cộng | (0.002836, 0.008508, 0.025523) | 0.791633 |

$T_{final}=0.791633$, $\sum_n c_n\alpha_nT_n=(0.086508, 0.097443, 0.084856)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.878141, 0.889076, 0.876488)}\ \to\ \text{8-bit }(224, 227, 224)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0828$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(46,17)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-22.5000, -1.5000) | -3.500459 | 0.030184 | 0.003018 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-19.6429, -6.2619) | -2.950655 | 0.052305 | 0.005231 | 1.000000 | cộng | (0.002615, 0.002615, 0.002615) | 0.994769 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-18.0556, 1.1667) | -2.291969 | 0.101067 | 0.010107 | 0.994769 | cộng | (0.002011, 0.007038, 0.003016) | 0.984716 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-25.7000, -3.1000) | -4.165363 | 0.015524 | 0.001552 | 0.984716 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.984716 |

$T_{final}=0.984716$, $\sum_n c_n\alpha_nT_n=(0.004626, 0.009653, 0.005631)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.989342, 0.994369, 0.990347)}\ \to\ \text{8-bit }(252, 254, 253)$$

*Nhận xét:* 2/4 Gaussian thực sự đóng góp (2 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0101$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(33,30)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-9.5000, -14.5000) | -2.068609 | 0.126361 | 0.012636 | 1.000000 | cộng | (0.010109, 0.002527, 0.002527) | 0.987364 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-6.6429, -19.2619) | -2.862311 | 0.057137 | 0.005714 | 0.987364 | cộng | (0.002821, 0.002821, 0.002821) | 0.981722 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-5.0556, -11.8333) | -1.159711 | 0.313577 | 0.031358 | 0.981722 | cộng | (0.006157, 0.021549, 0.009235) | 0.950938 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-12.7000, -16.1000) | -2.615244 | 0.073150 | 0.007315 | 0.950938 | cộng | (0.000696, 0.002087, 0.006260) | 0.943982 |

$T_{final}=0.943982$, $\sum_n c_n\alpha_nT_n=(0.019782, 0.028984, 0.020844)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.963764, 0.972966, 0.964826)}\ \to\ \text{8-bit }(246, 248, 246)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0314$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(46,30)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-22.5000, -14.5000) | -4.932308 | 0.007210 | 0.000721 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-19.6429, -19.2619) | -5.253877 | 0.005227 | 0.000523 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-18.0556, -11.8333) | -3.245350 | 0.038955 | 0.003895 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-25.7000, -16.1000) | -5.712137 | 0.003306 | 0.000331 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/4 Gaussian thực sự đóng góp (4 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0007$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(40,17)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-16.5000, -1.5000) | -1.889628 | 0.151128 | 0.015113 | 1.000000 | cộng | (0.012090, 0.003023, 0.003023) | 0.984887 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-13.6429, -6.2619) | -1.565470 | 0.208990 | 0.020899 | 0.984887 | cộng | (0.010292, 0.010292, 0.010292) | 0.964304 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-12.0556, 1.1667) | -1.027587 | 0.357869 | 0.035787 | 0.964304 | cộng | (0.006902, 0.024157, 0.010353) | 0.929795 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-19.7000, -3.1000) | -2.471686 | 0.084442 | 0.008444 | 0.929795 | cộng | (0.000785, 0.002355, 0.007066) | 0.921943 |

$T_{final}=0.921943$, $\sum_n c_n\alpha_nT_n=(0.030069, 0.039826, 0.030733)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.952012, 0.961769, 0.952676)}\ \to\ \text{8-bit }(243, 245, 243)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0358$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(40,30)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-16.5000, -14.5000) | -3.321478 | 0.036099 | 0.003610 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-13.6429, -19.2619) | -3.859684 | 0.021075 | 0.002107 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-12.0556, -11.8333) | -1.988981 | 0.136835 | 0.013683 | 1.000000 | cộng | (0.002737, 0.009578, 0.004105) | 0.986317 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-19.7000, -16.1000) | -4.021548 | 0.017925 | 0.001793 | 0.986317 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.986317 |

$T_{final}=0.986317$, $\sum_n c_n\alpha_nT_n=(0.002737, 0.009578, 0.004105)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.989053, 0.995895, 0.990422)}\ \to\ \text{8-bit }(252, 254, 253)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0137$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(33,24)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-9.5000, -8.5000) | -1.118632 | 0.326726 | 0.032673 | 1.000000 | cộng | (0.026138, 0.006535, 0.006535) | 0.967327 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-6.6429, -13.2619) | -1.520486 | 0.218606 | 0.021861 | 0.967327 | cộng | (0.010573, 0.010573, 0.010573) | 0.946181 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-5.0556, -5.8333) | -0.415609 | 0.659939 | 0.065994 | 0.946181 | cộng | (0.012488, 0.043709, 0.018733) | 0.883739 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-12.7000, -10.1000) | -1.635759 | 0.194804 | 0.019480 | 0.883739 | cộng | (0.001722, 0.005165, 0.015494) | 0.866523 |

$T_{final}=0.866523$, $\sum_n c_n\alpha_nT_n=(0.050921, 0.065982, 0.051334)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.917445, 0.932505, 0.917858)}\ \to\ \text{8-bit }(234, 238, 234)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0660$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

### Camera 2

#### Tile 0, pixel $x=(8,8)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (0.5000, 7.5000) | -0.388728 | 0.677918 | 0.067792 | 1.000000 | cộng | (0.054233, 0.013558, 0.013558) | 0.932208 |
| 2 | 4 | (0.5, 0.5, 0.5) | (4.0714, 2.7381) | -0.153272 | 0.857896 | 0.085790 | 0.932208 | cộng | (0.039987, 0.039987, 0.039987) | 0.852234 |
| 3 | 2 | (0.2, 0.7, 0.3) | (6.6111, 10.1667) | -1.037057 | 0.354496 | 0.035450 | 0.852234 | cộng | (0.006042, 0.021148, 0.009063) | 0.822023 |
| 4 | 3 | (0.1, 0.3, 0.9) | (0.3000, 5.9000) | -0.217804 | 0.804283 | 0.080428 | 0.822023 | cộng | (0.006611, 0.019834, 0.059503) | 0.755909 |

$T_{final}=0.755909$, $\sum_n c_n\alpha_nT_n=(0.106874, 0.094527, 0.122111)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.862783, 0.850436, 0.878020)}\ \to\ \text{8-bit }(220, 217, 224)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0858$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(1,1)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (7.5000, 14.5000) | -1.786991 | 0.167463 | 0.016746 | 1.000000 | cộng | (0.013397, 0.003349, 0.003349) | 0.983254 |
| 2 | 4 | (0.5, 0.5, 0.5) | (11.0714, 9.7381) | -1.392676 | 0.248410 | 0.024841 | 0.983254 | cộng | (0.012212, 0.012212, 0.012212) | 0.958829 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.6111, 17.1667) | -3.374461 | 0.034237 | 0.003424 | 0.958829 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.958829 |
| 4 | 3 | (0.1, 0.3, 0.9) | (7.3000, 12.9000) | -1.316371 | 0.268107 | 0.026811 | 0.958829 | cộng | (0.002571, 0.007712, 0.023136) | 0.933122 |

$T_{final}=0.933122$, $\sum_n c_n\alpha_nT_n=(0.028180, 0.023274, 0.038698)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.961302, 0.956396, 0.971820)}\ \to\ \text{8-bit }(245, 244, 248)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0248$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(14,1)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-5.5000, 14.5000) | -1.629996 | 0.195930 | 0.019593 | 1.000000 | cộng | (0.015674, 0.003919, 0.003919) | 0.980407 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-1.9286, 9.7381) | -0.682462 | 0.505371 | 0.050537 | 0.980407 | cộng | (0.024773, 0.024773, 0.024773) | 0.930860 |
| 3 | 2 | (0.2, 0.7, 0.3) | (0.6111, 17.1667) | -2.082287 | 0.124645 | 0.012464 | 0.930860 | cộng | (0.002321, 0.008122, 0.003481) | 0.919257 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-5.7000, 12.9000) | -1.230274 | 0.292213 | 0.029221 | 0.919257 | cộng | (0.002686, 0.008059, 0.024176) | 0.892395 |

$T_{final}=0.892395$, $\sum_n c_n\alpha_nT_n=(0.045455, 0.044873, 0.056349)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.937850, 0.937268, 0.948744)}\ \to\ \text{8-bit }(239, 239, 242)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0505$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(1,14)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (7.5000, 1.5000) | -0.355141 | 0.701074 | 0.070107 | 1.000000 | cộng | (0.056086, 0.014021, 0.014021) | 0.929893 |
| 2 | 4 | (0.5, 0.5, 0.5) | (11.0714, -3.2619) | -0.877046 | 0.416010 | 0.041601 | 0.929893 | cộng | (0.019342, 0.019342, 0.019342) | 0.891208 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.6111, 4.1667) | -1.384094 | 0.250551 | 0.025055 | 0.891208 | cộng | (0.004466, 0.015630, 0.006699) | 0.868879 |
| 4 | 3 | (0.1, 0.3, 0.9) | (7.3000, -0.1000) | -0.291878 | 0.746859 | 0.074686 | 0.868879 | cộng | (0.006489, 0.019468, 0.058404) | 0.803986 |

$T_{final}=0.803986$, $\sum_n c_n\alpha_nT_n=(0.086383, 0.068462, 0.098466)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.890369, 0.872448, 0.902452)}\ \to\ \text{8-bit }(227, 222, 230)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0747$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(14,14)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-5.5000, 1.5000) | -0.198146 | 0.820250 | 0.082025 | 1.000000 | cộng | (0.065620, 0.016405, 0.016405) | 0.917975 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-1.9286, -3.2619) | -0.094206 | 0.910095 | 0.091010 | 0.917975 | cộng | (0.041772, 0.041772, 0.041772) | 0.834431 |
| 3 | 2 | (0.2, 0.7, 0.3) | (0.6111, 4.1667) | -0.125427 | 0.882120 | 0.088212 | 0.834431 | cộng | (0.014721, 0.051525, 0.022082) | 0.760824 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-5.7000, -0.1000) | -0.177810 | 0.837102 | 0.083710 | 0.760824 | cộng | (0.006369, 0.019107, 0.057320) | 0.697135 |

$T_{final}=0.697135$, $\sum_n c_n\alpha_nT_n=(0.128482, 0.128809, 0.137579)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.825617, 0.825944, 0.834714)}\ \to\ \text{8-bit }(211, 211, 213)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0910$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(8,1)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (0.5000, 14.5000) | -1.448848 | 0.234841 | 0.023484 | 1.000000 | cộng | (0.018787, 0.004697, 0.004697) | 0.976516 |
| 2 | 4 | (0.5, 0.5, 0.5) | (4.0714, 9.7381) | -0.740064 | 0.477083 | 0.047708 | 0.976516 | cộng | (0.023294, 0.023294, 0.023294) | 0.929928 |
| 3 | 2 | (0.2, 0.7, 0.3) | (6.6111, 17.1667) | -2.395192 | 0.091155 | 0.009116 | 0.929928 | cộng | (0.001695, 0.005934, 0.002543) | 0.921451 |
| 4 | 3 | (0.1, 0.3, 0.9) | (0.3000, 12.9000) | -1.040114 | 0.353414 | 0.035341 | 0.921451 | cộng | (0.003257, 0.009770, 0.029309) | 0.888886 |

$T_{final}=0.888886$, $\sum_n c_n\alpha_nT_n=(0.047033, 0.043694, 0.059843)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.935919, 0.932580, 0.948728)}\ \to\ \text{8-bit }(239, 238, 242)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0477$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(8,14)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (0.5000, 1.5000) | -0.016998 | 0.983145 | 0.098315 | 1.000000 | cộng | (0.078652, 0.019663, 0.019663) | 0.901685 |
| 2 | 4 | (0.5, 0.5, 0.5) | (4.0714, -3.2619) | -0.185328 | 0.830832 | 0.083083 | 0.901685 | cộng | (0.037457, 0.037457, 0.037457) | 0.826771 |
| 3 | 2 | (0.2, 0.7, 0.3) | (6.6111, 4.1667) | -0.422866 | 0.655166 | 0.065517 | 0.826771 | cộng | (0.010833, 0.037917, 0.016250) | 0.772603 |
| 4 | 3 | (0.1, 0.3, 0.9) | (0.3000, -0.1000) | -0.000560 | 0.999440 | 0.099944 | 0.772603 | cộng | (0.007722, 0.023165, 0.069495) | 0.695386 |

$T_{final}=0.695386$, $\sum_n c_n\alpha_nT_n=(0.134664, 0.118203, 0.142866)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.830051, 0.813589, 0.838252)}\ \to\ \text{8-bit }(212, 207, 214)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0983$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(1,8)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (7.5000, 7.5000) | -0.726871 | 0.483419 | 0.048342 | 1.000000 | cộng | (0.038674, 0.009668, 0.009668) | 0.951658 |
| 2 | 4 | (0.5, 0.5, 0.5) | (11.0714, 2.7381) | -0.826941 | 0.437385 | 0.043739 | 0.951658 | cộng | (0.020812, 0.020812, 0.020812) | 0.910034 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.6111, 10.1667) | -2.006612 | 0.134443 | 0.013444 | 0.910034 | cộng | (0.002447, 0.008564, 0.003670) | 0.897799 |
| 4 | 3 | (0.1, 0.3, 0.9) | (7.3000, 5.9000) | -0.502171 | 0.605216 | 0.060522 | 0.897799 | cộng | (0.005434, 0.016301, 0.048903) | 0.843463 |

$T_{final}=0.843463$, $\sum_n c_n\alpha_nT_n=(0.067366, 0.055346, 0.083053)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.910829, 0.898809, 0.926516)}\ \to\ \text{8-bit }(232, 229, 236)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0605$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(24,8)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-15.5000, 7.5000) | -1.837913 | 0.159149 | 0.015915 | 1.000000 | cộng | (0.012732, 0.003183, 0.003183) | 0.984085 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-11.9286, 2.7381) | -0.980828 | 0.375000 | 0.037500 | 0.984085 | cộng | (0.018452, 0.018452, 0.018452) | 0.947182 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-9.3889, 10.1667) | -1.304790 | 0.271230 | 0.027123 | 0.947182 | cộng | (0.005138, 0.017983, 0.007707) | 0.921491 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-15.7000, 5.9000) | -1.582155 | 0.205532 | 0.020553 | 0.921491 | cộng | (0.001894, 0.005682, 0.017046) | 0.902552 |

$T_{final}=0.902552$, $\sum_n c_n\alpha_nT_n=(0.038216, 0.045300, 0.046387)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.940767, 0.947852, 0.948939)}\ \to\ \text{8-bit }(240, 242, 242)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0375$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(17,1)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-8.5000, 14.5000) | -1.883603 | 0.152041 | 0.015204 | 1.000000 | cộng | (0.012163, 0.003041, 0.003041) | 0.984796 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-4.9286, 9.7381) | -0.827354 | 0.437205 | 0.043720 | 0.984796 | cộng | (0.021528, 0.021528, 0.021528) | 0.941740 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.3889, 17.1667) | -2.108075 | 0.121472 | 0.012147 | 0.941740 | cộng | (0.002288, 0.008008, 0.003432) | 0.930301 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-8.7000, 12.9000) | -1.473144 | 0.229204 | 0.022920 | 0.930301 | cộng | (0.002132, 0.006397, 0.019191) | 0.908978 |

$T_{final}=0.908978$, $\sum_n c_n\alpha_nT_n=(0.038111, 0.038973, 0.047191)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.947089, 0.947951, 0.956169)}\ \to\ \text{8-bit }(242, 242, 244)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0437$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(30,1)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-21.5000, 14.5000) | -4.238527 | 0.014429 | 0.001443 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-17.9286, 9.7381) | -2.793298 | 0.061219 | 0.006122 | 1.000000 | cộng | (0.003061, 0.003061, 0.003061) | 0.993878 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-15.3889, 17.1667) | -3.623740 | 0.026683 | 0.002668 | 0.993878 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.993878 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-21.7000, 12.9000) | -3.664118 | 0.025627 | 0.002563 | 0.993878 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.993878 |

$T_{final}=0.993878$, $\sum_n c_n\alpha_nT_n=(0.003061, 0.003061, 0.003061)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.996939, 0.996939, 0.996939)}\ \to\ \text{8-bit }(254, 254, 254)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0061$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(17,14)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-8.5000, 1.5000) | -0.451754 | 0.636511 | 0.063651 | 1.000000 | cộng | (0.050921, 0.012730, 0.012730) | 0.936349 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-4.9286, -3.2619) | -0.222338 | 0.800644 | 0.080064 | 0.936349 | cộng | (0.037484, 0.037484, 0.037484) | 0.861381 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.3889, 4.1667) | -0.158946 | 0.853042 | 0.085304 | 0.861381 | cộng | (0.014696, 0.051436, 0.022044) | 0.787901 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-8.7000, -0.1000) | -0.414225 | 0.660852 | 0.066085 | 0.787901 | cộng | (0.005207, 0.015621, 0.046862) | 0.735833 |

$T_{final}=0.735833$, $\sum_n c_n\alpha_nT_n=(0.108308, 0.117271, 0.119120)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.844140, 0.853103, 0.854953)}\ \to\ \text{8-bit }(215, 218, 218)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0801$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(30,14)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-21.5000, 1.5000) | -2.806678 | 0.060405 | 0.006041 | 1.000000 | cộng | (0.004832, 0.001208, 0.001208) | 0.993959 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-17.9286, -3.2619) | -2.115658 | 0.120554 | 0.012055 | 0.993959 | cộng | (0.005991, 0.005991, 0.005991) | 0.981977 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-15.3889, 4.1667) | -1.708118 | 0.181207 | 0.018121 | 0.981977 | cộng | (0.003559, 0.012456, 0.005338) | 0.964183 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-21.7000, -0.1000) | -2.577227 | 0.075984 | 0.007598 | 0.964183 | cộng | (0.000733, 0.002198, 0.006594) | 0.956857 |

$T_{final}=0.956857$, $\sum_n c_n\alpha_nT_n=(0.015115, 0.021853, 0.019131)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.971972, 0.978710, 0.975988)}\ \to\ \text{8-bit }(248, 250, 249)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0181$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(24,1)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-15.5000, 14.5000) | -2.898032 | 0.055132 | 0.005513 | 1.000000 | cộng | (0.004411, 0.001103, 0.001103) | 0.994487 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-11.9286, 9.7381) | -1.615750 | 0.198742 | 0.019874 | 0.994487 | cộng | (0.009882, 0.009882, 0.009882) | 0.974722 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-9.3889, 17.1667) | -2.640719 | 0.071310 | 0.007131 | 0.974722 | cộng | (0.001390, 0.004866, 0.002085) | 0.967772 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-15.7000, 12.9000) | -2.423003 | 0.088655 | 0.008866 | 0.967772 | cộng | (0.000858, 0.002574, 0.007722) | 0.959192 |

$T_{final}=0.959192$, $\sum_n c_n\alpha_nT_n=(0.016541, 0.018424, 0.020792)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.975733, 0.977616, 0.979984)}\ \to\ \text{8-bit }(249, 249, 250)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0199$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(24,14)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-15.5000, 1.5000) | -1.466183 | 0.230805 | 0.023080 | 1.000000 | cộng | (0.018464, 0.004616, 0.004616) | 0.976920 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-11.9286, -3.2619) | -0.971629 | 0.378466 | 0.037847 | 0.976920 | cộng | (0.018487, 0.018487, 0.018487) | 0.939946 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-9.3889, 4.1667) | -0.709632 | 0.491825 | 0.049183 | 0.939946 | cộng | (0.009246, 0.032360, 0.013869) | 0.893717 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-15.7000, -0.1000) | -1.349022 | 0.259494 | 0.025949 | 0.893717 | cộng | (0.002319, 0.006957, 0.020872) | 0.870526 |

$T_{final}=0.870526$, $\sum_n c_n\alpha_nT_n=(0.048516, 0.062420, 0.057844)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.919042, 0.932946, 0.928370)}\ \to\ \text{8-bit }(234, 238, 237)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0492$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(17,8)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-8.5000, 7.5000) | -0.823484 | 0.438900 | 0.043890 | 1.000000 | cộng | (0.035112, 0.008778, 0.008778) | 0.956110 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-4.9286, 2.7381) | -0.213489 | 0.807761 | 0.080776 | 0.956110 | cộng | (0.038615, 0.038615, 0.038615) | 0.878879 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.3889, 10.1667) | -0.762431 | 0.466531 | 0.046653 | 0.878879 | cộng | (0.008200, 0.028702, 0.012301) | 0.837877 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-8.7000, 5.9000) | -0.640406 | 0.527078 | 0.052708 | 0.837877 | cộng | (0.004416, 0.013249, 0.039746) | 0.793714 |

$T_{final}=0.793714$, $\sum_n c_n\alpha_nT_n=(0.086344, 0.089344, 0.099441)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.880058, 0.883058, 0.893155)}\ \to\ \text{8-bit }(224, 225, 228)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0808$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(40,8)$ — center (tâm tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 2, pixel $x=(33,1)$ — corner trên-trái (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 2, pixel $x=(46,1)$ — corner trên-phải (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 2, pixel $x=(33,14)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 2, pixel $x=(46,14)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 2, pixel $x=(40,1)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 2, pixel $x=(40,14)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 2, pixel $x=(33,8)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 3, pixel $x=(8,24)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (0.5000, -8.5000) | -0.498871 | 0.607216 | 0.060722 | 1.000000 | cộng | (0.048577, 0.012144, 0.012144) | 0.939278 |
| 2 | 4 | (0.5, 0.5, 0.5) | (4.0714, -13.2619) | -1.336230 | 0.262835 | 0.026283 | 0.939278 | cộng | (0.012344, 0.012344, 0.012344) | 0.914591 |
| 3 | 2 | (0.2, 0.7, 0.3) | (6.6111, -5.8333) | -0.527264 | 0.590217 | 0.059022 | 0.914591 | cộng | (0.010796, 0.037787, 0.016194) | 0.860610 |
| 4 | 3 | (0.1, 0.3, 0.9) | (0.3000, -10.1000) | -0.638679 | 0.527989 | 0.052799 | 0.860610 | cộng | (0.004544, 0.013632, 0.040895) | 0.815171 |

$T_{final}=0.815171$, $\sum_n c_n\alpha_nT_n=(0.076261, 0.075906, 0.081578)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.891432, 0.891077, 0.896749)}\ \to\ \text{8-bit }(227, 227, 229)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0607$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(1,17)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (7.5000, -1.5000) | -0.355141 | 0.701074 | 0.070107 | 1.000000 | cộng | (0.056086, 0.014021, 0.014021) | 0.929893 |
| 2 | 4 | (0.5, 0.5, 0.5) | (11.0714, -6.2619) | -1.087297 | 0.337127 | 0.033713 | 0.929893 | cộng | (0.015675, 0.015675, 0.015675) | 0.898543 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.6111, 1.1667) | -1.263194 | 0.282750 | 0.028275 | 0.898543 | cộng | (0.005081, 0.017784, 0.007622) | 0.873137 |
| 4 | 3 | (0.1, 0.3, 0.9) | (7.3000, -3.1000) | -0.355515 | 0.700813 | 0.070081 | 0.873137 | cộng | (0.006119, 0.018357, 0.055072) | 0.811947 |

$T_{final}=0.811947$, $\sum_n c_n\alpha_nT_n=(0.082961, 0.065838, 0.092389)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.894907, 0.877784, 0.904336)}\ \to\ \text{8-bit }(228, 224, 231)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0701$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(14,17)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-5.5000, -1.5000) | -0.198146 | 0.820250 | 0.082025 | 1.000000 | cộng | (0.065620, 0.016405, 0.016405) | 0.917975 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-1.9286, -6.2619) | -0.287698 | 0.749988 | 0.074999 | 0.917975 | cộng | (0.034424, 0.034424, 0.034424) | 0.849128 |
| 3 | 2 | (0.2, 0.7, 0.3) | (0.6111, 1.1667) | -0.012258 | 0.987817 | 0.098782 | 0.849128 | cộng | (0.016776, 0.058715, 0.025163) | 0.765250 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-5.7000, -3.1000) | -0.234991 | 0.790578 | 0.079058 | 0.765250 | cộng | (0.006050, 0.018150, 0.054449) | 0.704751 |

$T_{final}=0.704751$, $\sum_n c_n\alpha_nT_n=(0.122869, 0.127693, 0.130441)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.827620, 0.832444, 0.835192)}\ \to\ \text{8-bit }(211, 212, 213)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0750$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(1,30)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (7.5000, -14.5000) | -1.786991 | 0.167463 | 0.016746 | 1.000000 | cộng | (0.013397, 0.003349, 0.003349) | 0.983254 |
| 2 | 4 | (0.5, 0.5, 0.5) | (11.0714, -19.2619) | -3.425105 | 0.032546 | 0.003255 | 0.983254 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.983254 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.6111, -11.8333) | -2.205755 | 0.110167 | 0.011017 | 0.983254 | cộng | (0.002166, 0.007583, 0.003250) | 0.972421 |
| 4 | 3 | (0.1, 0.3, 0.9) | (7.3000, -16.1000) | -1.931521 | 0.144928 | 0.014493 | 0.972421 | cộng | (0.001409, 0.004228, 0.012684) | 0.958328 |

$T_{final}=0.958328$, $\sum_n c_n\alpha_nT_n=(0.016973, 0.015160, 0.019283)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.975301, 0.973488, 0.977611)}\ \to\ \text{8-bit }(249, 248, 249)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0167$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(14,30)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-5.5000, -14.5000) | -1.629996 | 0.195930 | 0.019593 | 1.000000 | cộng | (0.015674, 0.003919, 0.003919) | 0.980407 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-1.9286, -19.2619) | -2.552881 | 0.077857 | 0.007786 | 0.980407 | cộng | (0.003817, 0.003817, 0.003817) | 0.972774 |
| 3 | 2 | (0.2, 0.7, 0.3) | (0.6111, -11.8333) | -0.988326 | 0.372199 | 0.037220 | 0.972774 | cộng | (0.007241, 0.025345, 0.010862) | 0.936567 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-5.7000, -16.1000) | -1.783026 | 0.168129 | 0.016813 | 0.936567 | cộng | (0.001575, 0.004724, 0.014172) | 0.920821 |

$T_{final}=0.920821$, $\sum_n c_n\alpha_nT_n=(0.028307, 0.037804, 0.032769)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.949128, 0.958625, 0.953590)}\ \to\ \text{8-bit }(242, 244, 243)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0372$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(8,17)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (0.5000, -1.5000) | -0.016998 | 0.983145 | 0.098315 | 1.000000 | cộng | (0.078652, 0.019663, 0.019663) | 0.901685 |
| 2 | 4 | (0.5, 0.5, 0.5) | (4.0714, -6.2619) | -0.386554 | 0.679394 | 0.067939 | 0.901685 | cộng | (0.030630, 0.030630, 0.030630) | 0.840426 |
| 3 | 2 | (0.2, 0.7, 0.3) | (6.6111, 1.1667) | -0.306129 | 0.736291 | 0.073629 | 0.840426 | cộng | (0.012376, 0.043316, 0.018564) | 0.778546 |
| 4 | 3 | (0.1, 0.3, 0.9) | (0.3000, -3.1000) | -0.060721 | 0.941086 | 0.094109 | 0.778546 | cộng | (0.007327, 0.021980, 0.065941) | 0.705278 |

$T_{final}=0.705278$, $\sum_n c_n\alpha_nT_n=(0.128984, 0.115589, 0.134798)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.834262, 0.820867, 0.840076)}\ \to\ \text{8-bit }(213, 209, 214)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0983$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(8,30)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (0.5000, -14.5000) | -1.448848 | 0.234841 | 0.023484 | 1.000000 | cộng | (0.018787, 0.004697, 0.004697) | 0.976516 |
| 2 | 4 | (0.5, 0.5, 0.5) | (4.0714, -19.2619) | -2.685257 | 0.068204 | 0.006820 | 0.976516 | cộng | (0.003330, 0.003330, 0.003330) | 0.969856 |
| 3 | 2 | (0.2, 0.7, 0.3) | (6.6111, -11.8333) | -1.266733 | 0.281751 | 0.028175 | 0.969856 | cộng | (0.005465, 0.019128, 0.008198) | 0.942530 |
| 4 | 3 | (0.1, 0.3, 0.9) | (0.3000, -16.1000) | -1.621666 | 0.197569 | 0.019757 | 0.942530 | cộng | (0.001862, 0.005586, 0.016759) | 0.923908 |

$T_{final}=0.923908$, $\sum_n c_n\alpha_nT_n=(0.029445, 0.032741, 0.032984)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.953353, 0.956650, 0.956892)}\ \to\ \text{8-bit }(243, 244, 244)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0282$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(1,24)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (7.5000, -8.5000) | -0.837014 | 0.433002 | 0.043300 | 1.000000 | cộng | (0.034640, 0.008660, 0.008660) | 0.956700 |
| 2 | 4 | (0.5, 0.5, 0.5) | (11.0714, -13.2619) | -2.058029 | 0.127705 | 0.012771 | 0.956700 | cộng | (0.006109, 0.006109, 0.006109) | 0.944482 |
| 3 | 2 | (0.2, 0.7, 0.3) | (13.6111, -5.8333) | -1.474614 | 0.228867 | 0.022887 | 0.944482 | cộng | (0.004323, 0.015131, 0.006485) | 0.922866 |
| 4 | 3 | (0.1, 0.3, 0.9) | (7.3000, -10.1000) | -0.941583 | 0.390010 | 0.039001 | 0.922866 | cộng | (0.003599, 0.010798, 0.032393) | 0.886873 |

$T_{final}=0.886873$, $\sum_n c_n\alpha_nT_n=(0.048671, 0.040698, 0.053647)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.935545, 0.927571, 0.940521)}\ \to\ \text{8-bit }(239, 237, 240)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0433$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(24,24)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-15.5000, -8.5000) | -1.948055 | 0.142551 | 0.014255 | 1.000000 | cộng | (0.011404, 0.002851, 0.002851) | 0.985745 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-11.9286, -13.2619) | -2.053773 | 0.128250 | 0.012825 | 0.985745 | cộng | (0.006321, 0.006321, 0.006321) | 0.973103 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-9.3889, -5.8333) | -0.845752 | 0.429235 | 0.042923 | 0.973103 | cộng | (0.008354, 0.029238, 0.012531) | 0.931334 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-15.7000, -10.1000) | -1.960659 | 0.140766 | 0.014077 | 0.931334 | cộng | (0.001311, 0.003933, 0.011799) | 0.918224 |

$T_{final}=0.918224$, $\sum_n c_n\alpha_nT_n=(0.027390, 0.042343, 0.033502)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.945614, 0.960567, 0.951726)}\ \to\ \text{8-bit }(241, 245, 243)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0429$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(17,17)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-8.5000, -1.5000) | -0.451754 | 0.636511 | 0.063651 | 1.000000 | cộng | (0.050921, 0.012730, 0.012730) | 0.936349 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-4.9286, -6.2619) | -0.411962 | 0.662349 | 0.066235 | 0.936349 | cộng | (0.031009, 0.031009, 0.031009) | 0.874330 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.3889, 1.1667) | -0.047562 | 0.953551 | 0.095355 | 0.874330 | cộng | (0.016674, 0.058360, 0.025012) | 0.790958 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-8.7000, -3.1000) | -0.469917 | 0.625054 | 0.062505 | 0.790958 | cộng | (0.004944, 0.014832, 0.044495) | 0.741519 |

$T_{final}=0.741519$, $\sum_n c_n\alpha_nT_n=(0.103549, 0.116932, 0.113247)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.845068, 0.858451, 0.854765)}\ \to\ \text{8-bit }(215, 219, 218)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0954$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(30,17)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-21.5000, -1.5000) | -2.806678 | 0.060405 | 0.006041 | 1.000000 | cộng | (0.004832, 0.001208, 0.001208) | 0.993959 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-17.9286, -6.2619) | -2.288522 | 0.101416 | 0.010142 | 0.993959 | cộng | (0.005040, 0.005040, 0.005040) | 0.983879 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-15.3889, 1.1667) | -1.604466 | 0.200997 | 0.020100 | 0.983879 | cộng | (0.003955, 0.013843, 0.005933) | 0.964103 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-21.7000, -3.1000) | -2.626463 | 0.072334 | 0.007233 | 0.964103 | cộng | (0.000697, 0.002092, 0.006276) | 0.957130 |

$T_{final}=0.957130$, $\sum_n c_n\alpha_nT_n=(0.014525, 0.022183, 0.018457)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.971655, 0.979313, 0.975587)}\ \to\ \text{8-bit }(248, 250, 249)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0201$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(17,30)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-8.5000, -14.5000) | -1.883603 | 0.152041 | 0.015204 | 1.000000 | cộng | (0.012163, 0.003041, 0.003041) | 0.984796 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-4.9286, -19.2619) | -2.660386 | 0.069921 | 0.006992 | 0.984796 | cộng | (0.003443, 0.003443, 0.003443) | 0.977910 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.3889, -11.8333) | -1.031362 | 0.356521 | 0.035652 | 0.977910 | cộng | (0.006973, 0.024405, 0.010459) | 0.943046 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-8.7000, -16.1000) | -2.011497 | 0.133788 | 0.013379 | 0.943046 | cộng | (0.001262, 0.003785, 0.011355) | 0.930429 |

$T_{final}=0.930429$, $\sum_n c_n\alpha_nT_n=(0.023841, 0.034674, 0.028298)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.954269, 0.965103, 0.958727)}\ \to\ \text{8-bit }(243, 246, 244)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0357$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(30,30)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-21.5000, -14.5000) | -4.238527 | 0.014429 | 0.001443 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-17.9286, -19.2619) | -4.464320 | 0.011513 | 0.001151 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-15.3889, -11.8333) | -2.621772 | 0.072674 | 0.007267 | 1.000000 | cộng | (0.001453, 0.005087, 0.002180) | 0.992733 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-21.7000, -16.1000) | -4.140072 | 0.015922 | 0.001592 | 0.992733 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.992733 |

$T_{final}=0.992733$, $\sum_n c_n\alpha_nT_n=(0.001453, 0.005087, 0.002180)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.994186, 0.997820, 0.994913)}\ \to\ \text{8-bit }(254, 254, 254)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0073$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(24,17)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-15.5000, -1.5000) | -1.466183 | 0.230805 | 0.023080 | 1.000000 | cộng | (0.018464, 0.004616, 0.004616) | 0.976920 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-11.9286, -6.2619) | -1.152228 | 0.315932 | 0.031593 | 0.976920 | cộng | (0.015432, 0.015432, 0.015432) | 0.946055 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-9.3889, 1.1667) | -0.602411 | 0.547490 | 0.054749 | 0.946055 | cộng | (0.010359, 0.036257, 0.015539) | 0.894260 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-15.7000, -3.1000) | -1.401238 | 0.246292 | 0.024629 | 0.894260 | cộng | (0.002202, 0.006607, 0.019822) | 0.872235 |

$T_{final}=0.872235$, $\sum_n c_n\alpha_nT_n=(0.046458, 0.062912, 0.055409)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.918693, 0.935147, 0.927644)}\ \to\ \text{8-bit }(234, 238, 237)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0547$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(24,30)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-15.5000, -14.5000) | -2.898032 | 0.055132 | 0.005513 | 1.000000 | cộng | (0.004411, 0.001103, 0.001103) | 0.994487 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-11.9286, -19.2619) | -3.361546 | 0.034682 | 0.003468 | 0.994487 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.994487 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-9.3889, -11.8333) | -1.604253 | 0.201040 | 0.020104 | 0.994487 | cộng | (0.003999, 0.013995, 0.005998) | 0.974494 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-15.7000, -16.1000) | -2.927756 | 0.053517 | 0.005352 | 0.974494 | cộng | (0.000522, 0.001565, 0.004694) | 0.969279 |

$T_{final}=0.969279$, $\sum_n c_n\alpha_nT_n=(0.008931, 0.016662, 0.011794)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.978209, 0.985941, 0.981073)}\ \to\ \text{8-bit }(249, 251, 250)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0201$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(17,24)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-8.5000, -8.5000) | -0.933626 | 0.393126 | 0.039313 | 1.000000 | cộng | (0.031450, 0.007863, 0.007863) | 0.960687 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-4.9286, -13.2619) | -1.334564 | 0.263273 | 0.026327 | 0.960687 | cộng | (0.012646, 0.012646, 0.012646) | 0.935395 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-2.3889, -5.8333) | -0.281188 | 0.754887 | 0.075489 | 0.935395 | cộng | (0.014122, 0.049428, 0.021184) | 0.864783 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-8.7000, -10.1000) | -1.037448 | 0.354358 | 0.035436 | 0.864783 | cộng | (0.003064, 0.009193, 0.027580) | 0.834139 |

$T_{final}=0.834139$, $\sum_n c_n\alpha_nT_n=(0.061283, 0.079130, 0.069272)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.895422, 0.913269, 0.903411)}\ \to\ \text{8-bit }(228, 233, 230)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0755$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(40,24)$ — center (tâm tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 5, pixel $x=(33,17)$ — corner trên-trái (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 5, pixel $x=(46,17)$ — corner trên-phải (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 5, pixel $x=(33,30)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 5, pixel $x=(46,30)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 5, pixel $x=(40,17)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 5, pixel $x=(40,30)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 5, pixel $x=(33,24)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

### Camera 3

#### Tile 0, pixel $x=(8,8)$ — center (tâm tile)

$\mathcal G_T=\{3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | (0.1, 0.3, 0.9) | (24.3000, 1.9000) | -3.568285 | 0.028204 | 0.002820 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/1 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=1$, $\alpha_n(x)=0.0028$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(1,1)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | (0.1, 0.3, 0.9) | (31.3000, 8.9000) | -6.442829 | 0.001592 | 0.000159 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/1 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=1$, $\alpha_n(x)=0.0002$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(14,1)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | (0.1, 0.3, 0.9) | (18.3000, 8.9000) | -2.547090 | 0.078309 | 0.007831 | 1.000000 | cộng | (0.000783, 0.002349, 0.007048) | 0.992169 |

$T_{final}=0.992169$, $\sum_n c_n\alpha_nT_n=(0.000783, 0.002349, 0.007048)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.992952, 0.994518, 0.999217)}\ \to\ \text{8-bit }(253, 254, 255)$$

*Nhận xét:* 1/1 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=1$, $\alpha_n(x)=0.0078$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(1,14)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | (0.1, 0.3, 0.9) | (31.3000, -4.1000) | -5.912956 | 0.002704 | 0.000270 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/1 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=1$, $\alpha_n(x)=0.0003$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(14,14)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | (0.1, 0.3, 0.9) | (18.3000, -4.1000) | -2.078021 | 0.125178 | 0.012518 | 1.000000 | cộng | (0.001252, 0.003755, 0.011266) | 0.987482 |

$T_{final}=0.987482$, $\sum_n c_n\alpha_nT_n=(0.001252, 0.003755, 0.011266)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.988734, 0.991238, 0.998748)}\ \to\ \text{8-bit }(252, 253, 255)$$

*Nhận xét:* 1/1 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=1$, $\alpha_n(x)=0.0125$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(8,1)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | (0.1, 0.3, 0.9) | (24.3000, 8.9000) | -4.094080 | 0.016671 | 0.001667 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/1 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=1$, $\alpha_n(x)=0.0017$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(8,14)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | (0.1, 0.3, 0.9) | (24.3000, -4.1000) | -3.596948 | 0.027407 | 0.002741 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/1 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=1$, $\alpha_n(x)=0.0027$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 0, pixel $x=(1,8)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | (0.1, 0.3, 0.9) | (31.3000, 1.9000) | -5.899405 | 0.002741 | 0.000274 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/1 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=1$, $\alpha_n(x)=0.0003$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(24,8)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (14.5000, 2.5000) | -1.334545 | 0.263278 | 0.026328 | 1.000000 | cộng | (0.021062, 0.005266, 0.005266) | 0.973672 |
| 2 | 4 | (0.5, 0.5, 0.5) | (16.6429, -2.0238) | -1.629433 | 0.196041 | 0.019604 | 0.973672 | cộng | (0.009544, 0.009544, 0.009544) | 0.954584 |
| 3 | 2 | (0.2, 0.7, 0.3) | (17.2778, 5.7222) | -2.021149 | 0.132503 | 0.013250 | 0.954584 | cộng | (0.002530, 0.008854, 0.003795) | 0.941936 |
| 4 | 3 | (0.1, 0.3, 0.9) | (8.3000, 1.9000) | -0.439630 | 0.644275 | 0.064428 | 0.941936 | cộng | (0.006069, 0.018206, 0.054618) | 0.881249 |

$T_{final}=0.881249$, $\sum_n c_n\alpha_nT_n=(0.039205, 0.041869, 0.073222)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.920454, 0.923119, 0.954471)}\ \to\ \text{8-bit }(235, 235, 243)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0644$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(17,1)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (21.5000, 9.5000) | -3.522954 | 0.029512 | 0.002951 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (23.6429, 4.9762) | -3.609291 | 0.027071 | 0.002707 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (24.2778, 12.7222) | -4.704113 | 0.009058 | 0.000906 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (15.3000, 8.9000) | -1.934980 | 0.144427 | 0.014443 | 1.000000 | cộng | (0.001444, 0.004333, 0.012998) | 0.985557 |

$T_{final}=0.985557$, $\sum_n c_n\alpha_nT_n=(0.001444, 0.004333, 0.012998)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.987002, 0.989890, 0.998556)}\ \to\ \text{8-bit }(252, 252, 255)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0144$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(30,1)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (8.5000, 9.5000) | -1.094903 | 0.334572 | 0.033457 | 1.000000 | cộng | (0.026766, 0.006691, 0.006691) | 0.966543 |
| 2 | 4 | (0.5, 0.5, 0.5) | (10.6429, 4.9762) | -0.895505 | 0.408401 | 0.040840 | 0.966543 | cộng | (0.019737, 0.019737, 0.019737) | 0.927069 |
| 3 | 2 | (0.2, 0.7, 0.3) | (11.2778, 12.7222) | -1.930132 | 0.145129 | 0.014513 | 0.927069 | cộng | (0.002691, 0.009418, 0.004036) | 0.913615 |
| 4 | 3 | (0.1, 0.3, 0.9) | (2.3000, 8.9000) | -0.525764 | 0.591104 | 0.059110 | 0.913615 | cộng | (0.005400, 0.016201, 0.048604) | 0.859611 |

$T_{final}=0.859611$, $\sum_n c_n\alpha_nT_n=(0.054594, 0.052048, 0.079068)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.914204, 0.911658, 0.938679)}\ \to\ \text{8-bit }(233, 232, 239)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0591$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(17,14)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (21.5000, -3.5000) | -2.837794 | 0.058555 | 0.005855 | 1.000000 | cộng | (0.004684, 0.001171, 0.001171) | 0.994145 |
| 2 | 4 | (0.5, 0.5, 0.5) | (23.6429, -8.0238) | -3.521833 | 0.029545 | 0.002955 | 0.994145 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.994145 |
| 3 | 2 | (0.2, 0.7, 0.3) | (24.2778, -0.2778) | -3.487263 | 0.030584 | 0.003058 | 0.994145 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.994145 |
| 4 | 3 | (0.1, 0.3, 0.9) | (15.3000, -4.1000) | -1.479943 | 0.227651 | 0.022765 | 0.994145 | cộng | (0.002263, 0.006790, 0.020369) | 0.971513 |

$T_{final}=0.971513$, $\sum_n c_n\alpha_nT_n=(0.006948, 0.007961, 0.021540)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.978460, 0.979473, 0.993052)}\ \to\ \text{8-bit }(250, 250, 253)$$

*Nhận xét:* 2/4 Gaussian thực sự đóng góp (2 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0228$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(30,14)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (8.5000, -3.5000) | -0.503734 | 0.604270 | 0.060427 | 1.000000 | cộng | (0.048342, 0.012085, 0.012085) | 0.939573 |
| 2 | 4 | (0.5, 0.5, 0.5) | (10.6429, -8.0238) | -1.000616 | 0.367653 | 0.036765 | 0.939573 | cộng | (0.017272, 0.017272, 0.017272) | 0.905029 |
| 3 | 2 | (0.2, 0.7, 0.3) | (11.2778, -0.2778) | -0.752551 | 0.471163 | 0.047116 | 0.905029 | cộng | (0.008528, 0.029849, 0.012792) | 0.862388 |
| 4 | 3 | (0.1, 0.3, 0.9) | (2.3000, -4.1000) | -0.131531 | 0.876752 | 0.087675 | 0.862388 | cộng | (0.007561, 0.022683, 0.068049) | 0.786778 |

$T_{final}=0.786778$, $\sum_n c_n\alpha_nT_n=(0.081703, 0.081889, 0.110199)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.868480, 0.868667, 0.896976)}\ \to\ \text{8-bit }(221, 222, 229)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0877$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(24,1)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (14.5000, 9.5000) | -1.961456 | 0.140653 | 0.014065 | 1.000000 | cộng | (0.011252, 0.002813, 0.002813) | 0.985935 |
| 2 | 4 | (0.5, 0.5, 0.5) | (16.6429, 4.9762) | -1.899246 | 0.149681 | 0.014968 | 0.985935 | cộng | (0.007379, 0.007379, 0.007379) | 0.971177 |
| 3 | 2 | (0.2, 0.7, 0.3) | (17.2778, 12.7222) | -2.961864 | 0.051722 | 0.005172 | 0.971177 | cộng | (0.001005, 0.003516, 0.001507) | 0.966154 |
| 4 | 3 | (0.1, 0.3, 0.9) | (8.3000, 8.9000) | -0.925128 | 0.396481 | 0.039648 | 0.966154 | cộng | (0.003831, 0.011492, 0.034476) | 0.927848 |

$T_{final}=0.927848$, $\sum_n c_n\alpha_nT_n=(0.023466, 0.025200, 0.046174)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.951314, 0.953048, 0.974022)}\ \to\ \text{8-bit }(243, 243, 248)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0396$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(24,14)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (14.5000, -3.5000) | -1.326907 | 0.265297 | 0.026530 | 1.000000 | cộng | (0.021224, 0.005306, 0.005306) | 0.973470 |
| 2 | 4 | (0.5, 0.5, 0.5) | (16.6429, -8.0238) | -1.915479 | 0.147271 | 0.014727 | 0.973470 | cộng | (0.007168, 0.007168, 0.007168) | 0.959134 |
| 3 | 2 | (0.2, 0.7, 0.3) | (17.2778, -0.2778) | -1.766159 | 0.170989 | 0.017099 | 0.959134 | cộng | (0.003280, 0.011480, 0.004920) | 0.942734 |
| 4 | 3 | (0.1, 0.3, 0.9) | (8.3000, -4.1000) | -0.502832 | 0.604815 | 0.060482 | 0.942734 | cộng | (0.005702, 0.017105, 0.051316) | 0.885716 |

$T_{final}=0.885716$, $\sum_n c_n\alpha_nT_n=(0.037374, 0.041060, 0.068710)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.923090, 0.926775, 0.954426)}\ \to\ \text{8-bit }(235, 236, 243)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0605$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 1, pixel $x=(17,8)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (21.5000, 2.5000) | -2.868791 | 0.056768 | 0.005677 | 1.000000 | cộng | (0.004541, 0.001135, 0.001135) | 0.994323 |
| 2 | 4 | (0.5, 0.5, 0.5) | (23.6429, -2.0238) | -3.283645 | 0.037491 | 0.003749 | 0.994323 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.994323 |
| 3 | 2 | (0.2, 0.7, 0.3) | (24.2778, 5.7222) | -3.752012 | 0.023470 | 0.002347 | 0.994323 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.994323 |
| 4 | 3 | (0.1, 0.3, 0.9) | (15.3000, 1.9000) | -1.431852 | 0.238866 | 0.023887 | 0.994323 | cộng | (0.002375, 0.007125, 0.021376) | 0.970572 |

$T_{final}=0.970572$, $\sum_n c_n\alpha_nT_n=(0.006917, 0.008261, 0.022511)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.977489, 0.978833, 0.993083)}\ \to\ \text{8-bit }(249, 250, 253)$$

*Nhận xét:* 2/4 Gaussian thực sự đóng góp (2 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0239$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(40,8)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-1.5000, 2.5000) | -0.053971 | 0.947459 | 0.094746 | 1.000000 | cộng | (0.075797, 0.018949, 0.018949) | 0.905254 |
| 2 | 4 | (0.5, 0.5, 0.5) | (0.6429, -2.0238) | -0.028130 | 0.972262 | 0.097226 | 0.905254 | cộng | (0.044007, 0.044007, 0.044007) | 0.817240 |
| 3 | 2 | (0.2, 0.7, 0.3) | (1.2778, 5.7222) | -0.242809 | 0.784421 | 0.078442 | 0.817240 | cộng | (0.012821, 0.044874, 0.019232) | 0.753134 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-7.7000, 1.9000) | -0.371311 | 0.689830 | 0.068983 | 0.753134 | cộng | (0.005195, 0.015586, 0.046758) | 0.701180 |

$T_{final}=0.701180$, $\sum_n c_n\alpha_nT_n=(0.137821, 0.123417, 0.128946)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.839001, 0.824597, 0.830126)}\ \to\ \text{8-bit }(214, 210, 212)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0972$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(33,1)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (5.5000, 9.5000) | -0.824967 | 0.438249 | 0.043825 | 1.000000 | cộng | (0.035060, 0.008765, 0.008765) | 0.956175 |
| 2 | 4 | (0.5, 0.5, 0.5) | (7.6429, 4.9762) | -0.553563 | 0.574898 | 0.057490 | 0.956175 | cộng | (0.027485, 0.027485, 0.027485) | 0.901205 |
| 3 | 2 | (0.2, 0.7, 0.3) | (8.2778, 12.7222) | -1.574059 | 0.207202 | 0.020720 | 0.901205 | cộng | (0.003735, 0.013071, 0.005602) | 0.882532 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-0.7000, 8.9000) | -0.487467 | 0.614180 | 0.061418 | 0.882532 | cộng | (0.005420, 0.016261, 0.048783) | 0.828328 |

$T_{final}=0.828328$, $\sum_n c_n\alpha_nT_n=(0.071700, 0.065582, 0.090635)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.900028, 0.893911, 0.918963)}\ \to\ \text{8-bit }(230, 228, 234)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0575$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(46,1)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-7.5000, 9.5000) | -0.913572 | 0.401089 | 0.040109 | 1.000000 | cộng | (0.032087, 0.008022, 0.008022) | 0.959891 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-5.3571, 4.9762) | -0.303846 | 0.737975 | 0.073797 | 0.959891 | cộng | (0.035419, 0.035419, 0.035419) | 0.889054 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-4.7222, 12.7222) | -1.262075 | 0.283066 | 0.028307 | 0.889054 | cộng | (0.005033, 0.017616, 0.007550) | 0.863887 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-13.7000, 8.9000) | -1.564775 | 0.209135 | 0.020914 | 0.863887 | cộng | (0.001807, 0.005420, 0.016260) | 0.845821 |

$T_{final}=0.845821$, $\sum_n c_n\alpha_nT_n=(0.074346, 0.066477, 0.067251)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.920166, 0.912297, 0.913071)}\ \to\ \text{8-bit }(235, 233, 233)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0738$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(33,14)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (5.5000, -3.5000) | -0.255488 | 0.774538 | 0.077454 | 1.000000 | cộng | (0.061963, 0.015491, 0.015491) | 0.922546 |
| 2 | 4 | (0.5, 0.5, 0.5) | (7.6429, -8.0238) | -0.703112 | 0.495042 | 0.049504 | 0.922546 | cộng | (0.022835, 0.022835, 0.022835) | 0.876876 |
| 3 | 2 | (0.2, 0.7, 0.3) | (8.2778, -0.2778) | -0.405540 | 0.666617 | 0.066662 | 0.876876 | cộng | (0.011691, 0.040918, 0.017536) | 0.818422 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-0.7000, -4.1000) | -0.107266 | 0.898287 | 0.089829 | 0.818422 | cộng | (0.007352, 0.022055, 0.066166) | 0.744904 |

$T_{final}=0.744904$, $\sum_n c_n\alpha_nT_n=(0.103841, 0.101299, 0.122028)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.848745, 0.846203, 0.866932)}\ \to\ \text{8-bit }(216, 216, 221)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0898$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(46,14)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-7.5000, -3.5000) | -0.438084 | 0.645271 | 0.064527 | 1.000000 | cộng | (0.051622, 0.012905, 0.012905) | 0.935473 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-5.3571, -8.0238) | -0.645964 | 0.524157 | 0.052416 | 0.935473 | cộng | (0.024517, 0.024517, 0.024517) | 0.886439 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-4.7222, -0.2778) | -0.132824 | 0.875619 | 0.087562 | 0.886439 | cộng | (0.015524, 0.054333, 0.023286) | 0.808821 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-13.7000, -4.1000) | -1.245379 | 0.287832 | 0.028783 | 0.808821 | cộng | (0.002328, 0.006984, 0.020952) | 0.785541 |

$T_{final}=0.785541$, $\sum_n c_n\alpha_nT_n=(0.093990, 0.098739, 0.081660)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.879531, 0.884280, 0.867201)}\ \to\ \text{8-bit }(224, 225, 221)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0876$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(40,1)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-1.5000, 9.5000) | -0.618592 | 0.538702 | 0.053870 | 1.000000 | cộng | (0.043096, 0.010774, 0.010774) | 0.946130 |
| 2 | 4 | (0.5, 0.5, 0.5) | (0.6429, 4.9762) | -0.170324 | 0.843392 | 0.084339 | 0.946130 | cộng | (0.039898, 0.039898, 0.039898) | 0.866334 |
| 3 | 2 | (0.2, 0.7, 0.3) | (1.2778, 12.7222) | -1.157500 | 0.314271 | 0.031427 | 0.866334 | cộng | (0.005445, 0.019058, 0.008168) | 0.839108 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-7.7000, 8.9000) | -0.816513 | 0.441970 | 0.044197 | 0.839108 | cộng | (0.003709, 0.011126, 0.033377) | 0.802022 |

$T_{final}=0.802022$, $\sum_n c_n\alpha_nT_n=(0.092148, 0.080856, 0.092217)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.894170, 0.882878, 0.894239)}\ \to\ \text{8-bit }(228, 225, 228)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=4$ (rank $n=2$, $\alpha_n(x)=0.0843$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(40,14)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-1.5000, -3.5000) | -0.099724 | 0.905087 | 0.090509 | 1.000000 | cộng | (0.072407, 0.018102, 0.018102) | 0.909491 |
| 2 | 4 | (0.5, 0.5, 0.5) | (0.6429, -8.0238) | -0.423564 | 0.654709 | 0.065471 | 0.909491 | cộng | (0.029773, 0.029773, 0.029773) | 0.849946 |
| 3 | 2 | (0.2, 0.7, 0.3) | (1.2778, -0.2778) | -0.010126 | 0.989925 | 0.098993 | 0.849946 | cộng | (0.016828, 0.058897, 0.025241) | 0.765808 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-7.7000, -4.1000) | -0.469053 | 0.625595 | 0.062559 | 0.765808 | cộng | (0.004791, 0.014373, 0.043118) | 0.717899 |

$T_{final}=0.717899$, $\sum_n c_n\alpha_nT_n=(0.123798, 0.121144, 0.116234)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.841697, 0.839043, 0.834133)}\ \to\ \text{8-bit }(215, 214, 213)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0905$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 2, pixel $x=(33,8)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (5.5000, 2.5000) | -0.233094 | 0.792079 | 0.079208 | 1.000000 | cộng | (0.063366, 0.015842, 0.015842) | 0.920792 |
| 2 | 4 | (0.5, 0.5, 0.5) | (7.6429, -2.0238) | -0.355535 | 0.700798 | 0.070080 | 0.920792 | cộng | (0.032264, 0.032264, 0.032264) | 0.856263 |
| 3 | 2 | (0.2, 0.7, 0.3) | (8.2778, 5.7222) | -0.647982 | 0.523100 | 0.052310 | 0.856263 | cộng | (0.008958, 0.031354, 0.013437) | 0.811472 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-0.7000, 1.9000) | -0.024635 | 0.975666 | 0.097567 | 0.811472 | cộng | (0.007917, 0.023752, 0.071255) | 0.732299 |

$T_{final}=0.732299$, $\sum_n c_n\alpha_nT_n=(0.112506, 0.103212, 0.132799)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.844806, 0.835511, 0.865098)}\ \to\ \text{8-bit }(215, 213, 221)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0976$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 3, pixel $x=(8,24)$ — center (tâm tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 3, pixel $x=(1,17)$ — corner trên-trái (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 3, pixel $x=(14,17)$ — corner trên-phải (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 3, pixel $x=(1,30)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 3, pixel $x=(14,30)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 3, pixel $x=(8,17)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 3, pixel $x=(8,30)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 3, pixel $x=(1,24)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\varnothing$ — tile này không có Gaussian nào chạm tới (xem $\mathcal K_i$ ở §4.0).
Không vào vòng lặp `renderCUDA`, `T_{final}=1`, `n\_contrib=0`, kết quả trực tiếp là nền:

$$C(x)=T_{final}\cdot C_{bg}=(1,1,1)\ \to\ \text{8-bit }(255,255,255)$$

#### Tile 4, pixel $x=(24,24)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (14.5000, -13.5000) | -2.400768 | 0.090648 | 0.009065 | 1.000000 | cộng | (0.007252, 0.001813, 0.001813) | 0.990935 |
| 2 | 4 | (0.5, 0.5, 0.5) | (16.6429, -18.0238) | -3.453381 | 0.031638 | 0.003164 | 0.990935 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.990935 |
| 3 | 2 | (0.2, 0.7, 0.3) | (17.2778, -10.2778) | -2.472125 | 0.084405 | 0.008441 | 0.990935 | cộng | (0.001673, 0.005855, 0.002509) | 0.982571 |
| 4 | 3 | (0.1, 0.3, 0.9) | (8.3000, -14.1000) | -1.591438 | 0.203633 | 0.020363 | 0.982571 | cộng | (0.002001, 0.006003, 0.018008) | 0.962563 |

$T_{final}=0.962563$, $\sum_n c_n\alpha_nT_n=(0.010926, 0.013670, 0.022330)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.973488, 0.976233, 0.984892)}\ \to\ \text{8-bit }(248, 249, 251)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0204$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(17,17)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (21.5000, -6.5000) | -3.005658 | 0.049506 | 0.004951 | 1.000000 | cộng | (0.003960, 0.000990, 0.000990) | 0.995049 |
| 2 | 4 | (0.5, 0.5, 0.5) | (23.6429, -11.0238) | -3.819998 | 0.021928 | 0.002193 | 0.995049 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.995049 |
| 3 | 2 | (0.2, 0.7, 0.3) | (24.2778, -3.2778) | -3.545736 | 0.028847 | 0.002885 | 0.995049 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.995049 |
| 4 | 3 | (0.1, 0.3, 0.9) | (15.3000, -7.1000) | -1.669915 | 0.188263 | 0.018826 | 0.995049 | cộng | (0.001873, 0.005620, 0.016860) | 0.976316 |

$T_{final}=0.976316$, $\sum_n c_n\alpha_nT_n=(0.005834, 0.006610, 0.017850)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.982150, 0.982926, 0.994166)}\ \to\ \text{8-bit }(250, 251, 254)$$

*Nhận xét:* 2/4 Gaussian thực sự đóng góp (2 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0188$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(30,17)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (8.5000, -6.5000) | -0.693288 | 0.499930 | 0.049993 | 1.000000 | cộng | (0.039994, 0.009999, 0.009999) | 0.950007 |
| 2 | 4 | (0.5, 0.5, 0.5) | (10.6429, -11.0238) | -1.343220 | 0.261004 | 0.026100 | 0.950007 | cộng | (0.012398, 0.012398, 0.012398) | 0.925211 |
| 3 | 2 | (0.2, 0.7, 0.3) | (11.2778, -3.2778) | -0.820086 | 0.440394 | 0.044039 | 0.925211 | cộng | (0.008149, 0.028522, 0.012224) | 0.884466 |
| 4 | 3 | (0.1, 0.3, 0.9) | (2.3000, -7.1000) | -0.335535 | 0.714955 | 0.071496 | 0.884466 | cộng | (0.006324, 0.018971, 0.056912) | 0.821230 |

$T_{final}=0.821230$, $\sum_n c_n\alpha_nT_n=(0.066865, 0.069889, 0.091532)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.888095, 0.891119, 0.912762)}\ \to\ \text{8-bit }(226, 227, 233)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0715$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(17,30)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (21.5000, -19.5000) | -5.145636 | 0.005825 | 0.000582 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (23.6429, -24.0238) | -6.491552 | 0.001516 | 0.000152 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (24.2778, -16.2778) | -5.269354 | 0.005147 | 0.000515 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (15.3000, -20.1000) | -3.771378 | 0.023020 | 0.002302 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |

$T_{final}=1.000000$, $\sum_n c_n\alpha_nT_n=(0.000000, 0.000000, 0.000000)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(1.000000, 1.000000, 1.000000)}\ \to\ \text{8-bit }(255, 255, 255)$$

*Nhận xét:* 0/4 Gaussian thực sự đóng góp (4 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=1$ (rank $n=1$, $\alpha_n(x)=0.0006$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(30,30)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (8.5000, -19.5000) | -2.927258 | 0.053544 | 0.005354 | 1.000000 | cộng | (0.004283, 0.001071, 0.001071) | 0.994646 |
| 2 | 4 | (0.5, 0.5, 0.5) | (10.6429, -24.0238) | -4.207342 | 0.014886 | 0.001489 | 0.994646 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.994646 |
| 3 | 2 | (0.2, 0.7, 0.3) | (11.2778, -16.2778) | -2.582972 | 0.075549 | 0.007555 | 0.994646 | cộng | (0.001503, 0.005260, 0.002254) | 0.987131 |
| 4 | 3 | (0.1, 0.3, 0.9) | (2.3000, -20.1000) | -2.497803 | 0.082266 | 0.008227 | 0.987131 | cộng | (0.000812, 0.002436, 0.007309) | 0.979010 |

$T_{final}=0.979010$, $\sum_n c_n\alpha_nT_n=(0.006598, 0.008767, 0.010634)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.985609, 0.987778, 0.989644)}\ \to\ \text{8-bit }(251, 252, 252)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0082$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(24,17)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (14.5000, -6.5000) | -1.506450 | 0.221696 | 0.022170 | 1.000000 | cộng | (0.017736, 0.004434, 0.004434) | 0.977830 |
| 2 | 4 | (0.5, 0.5, 0.5) | (16.6429, -11.0238) | -2.237572 | 0.106717 | 0.010672 | 0.977830 | cộng | (0.005218, 0.005218, 0.005218) | 0.967395 |
| 3 | 2 | (0.2, 0.7, 0.3) | (17.2778, -3.2778) | -1.829512 | 0.160492 | 0.016049 | 0.967395 | cộng | (0.003105, 0.010868, 0.004658) | 0.951869 |
| 4 | 3 | (0.1, 0.3, 0.9) | (8.3000, -7.1000) | -0.700360 | 0.496407 | 0.049641 | 0.951869 | cộng | (0.004725, 0.014175, 0.042526) | 0.904618 |

$T_{final}=0.904618$, $\sum_n c_n\alpha_nT_n=(0.030784, 0.034695, 0.056836)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.935401, 0.939313, 0.961453)}\ \to\ \text{8-bit }(239, 240, 245)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0496$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(24,30)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (14.5000, -19.5000) | -3.697039 | 0.024797 | 0.002480 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (16.6429, -24.0238) | -5.012817 | 0.006652 | 0.000665 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (17.2778, -16.2778) | -3.574274 | 0.028036 | 0.002804 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (8.3000, -20.1000) | -2.834564 | 0.058744 | 0.005874 | 1.000000 | cộng | (0.000587, 0.001762, 0.005287) | 0.994126 |

$T_{final}=0.994126$, $\sum_n c_n\alpha_nT_n=(0.000587, 0.001762, 0.005287)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.994713, 0.995888, 0.999413)}\ \to\ \text{8-bit }(254, 254, 255)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0059$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 4, pixel $x=(17,24)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (21.5000, -13.5000) | -3.872724 | 0.020802 | 0.002080 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 2 | 4 | (0.5, 0.5, 0.5) | (23.6429, -18.0238) | -4.979973 | 0.006874 | 0.000687 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 3 | 2 | (0.2, 0.7, 0.3) | (24.2778, -10.2778) | -4.176964 | 0.015345 | 0.001535 | 1.000000 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 1.000000 |
| 4 | 3 | (0.1, 0.3, 0.9) | (15.3000, -14.1000) | -2.543364 | 0.078602 | 0.007860 | 1.000000 | cộng | (0.000786, 0.002358, 0.007074) | 0.992140 |

$T_{final}=0.992140$, $\sum_n c_n\alpha_nT_n=(0.000786, 0.002358, 0.007074)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.992926, 0.994498, 0.999214)}\ \to\ \text{8-bit }(253, 254, 255)$$

*Nhận xét:* 1/4 Gaussian thực sự đóng góp (3 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0079$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(40,24)$ — center (tâm tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-1.5000, -13.5000) | -1.262570 | 0.282926 | 0.028293 | 1.000000 | cộng | (0.022634, 0.005659, 0.005659) | 0.971707 |
| 2 | 4 | (0.5, 0.5, 0.5) | (0.6429, -18.0238) | -2.143779 | 0.117211 | 0.011721 | 0.971707 | cộng | (0.005695, 0.005695, 0.005695) | 0.960318 |
| 3 | 2 | (0.2, 0.7, 0.3) | (1.2778, -10.2778) | -0.753269 | 0.470825 | 0.047082 | 0.960318 | cộng | (0.009043, 0.031650, 0.013564) | 0.915104 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-7.7000, -14.1000) | -1.615225 | 0.198846 | 0.019885 | 0.915104 | cộng | (0.001820, 0.005459, 0.016377) | 0.896907 |

$T_{final}=0.896907$, $\sum_n c_n\alpha_nT_n=(0.039191, 0.048462, 0.041294)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.936099, 0.945369, 0.938202)}\ \to\ \text{8-bit }(239, 241, 239)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0471$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(33,17)$ — corner trên-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (5.5000, -6.5000) | -0.450048 | 0.637598 | 0.063760 | 1.000000 | cộng | (0.051008, 0.012752, 0.012752) | 0.936240 |
| 2 | 4 | (0.5, 0.5, 0.5) | (7.6429, -11.0238) | -1.055971 | 0.347854 | 0.034785 | 0.936240 | cộng | (0.016284, 0.016284, 0.016284) | 0.903673 |
| 3 | 2 | (0.2, 0.7, 0.3) | (8.2778, -3.2778) | -0.475166 | 0.621782 | 0.062178 | 0.903673 | cộng | (0.011238, 0.039332, 0.016857) | 0.847484 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-0.7000, -7.1000) | -0.314508 | 0.730148 | 0.073015 | 0.847484 | cộng | (0.006188, 0.018564, 0.055691) | 0.785605 |

$T_{final}=0.785605$, $\sum_n c_n\alpha_nT_n=(0.084717, 0.086931, 0.101583)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.870322, 0.872537, 0.887188)}\ \to\ \text{8-bit }(222, 222, 226)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=3$ (rank $n=4$, $\alpha_n(x)=0.0730$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(46,17)$ — corner trên-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-7.5000, -6.5000) | -0.654334 | 0.519788 | 0.051979 | 1.000000 | cộng | (0.041583, 0.010396, 0.010396) | 0.948021 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-5.3571, -11.0238) | -1.043262 | 0.352304 | 0.035230 | 0.948021 | cộng | (0.016700, 0.016700, 0.016700) | 0.914622 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-4.7222, -3.2778) | -0.211512 | 0.809360 | 0.080936 | 0.914622 | cộng | (0.014805, 0.051818, 0.022208) | 0.840596 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-13.7000, -7.1000) | -1.466652 | 0.230696 | 0.023070 | 0.840596 | cộng | (0.001939, 0.005818, 0.017453) | 0.821204 |

$T_{final}=0.821204$, $\sum_n c_n\alpha_nT_n=(0.075027, 0.084731, 0.066756)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.896231, 0.905935, 0.887960)}\ \to\ \text{8-bit }(229, 231, 226)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0809$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(33,30)$ — corner dưới-trái (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (5.5000, -19.5000) | -2.705707 | 0.066823 | 0.006682 | 1.000000 | cộng | (0.005346, 0.001336, 0.001336) | 0.993318 |
| 2 | 4 | (0.5, 0.5, 0.5) | (7.6429, -24.0238) | -3.964532 | 0.018977 | 0.001898 | 0.993318 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.993318 |
| 3 | 2 | (0.2, 0.7, 0.3) | (8.2778, -16.2778) | -2.247114 | 0.105704 | 0.010570 | 0.993318 | cộng | (0.002100, 0.007350, 0.003150) | 0.982818 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-0.7000, -20.1000) | -2.490808 | 0.082843 | 0.008284 | 0.982818 | cộng | (0.000814, 0.002443, 0.007328) | 0.974676 |

$T_{final}=0.974676$, $\sum_n c_n\alpha_nT_n=(0.008260, 0.011129, 0.011814)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.982936, 0.985805, 0.986490)}\ \to\ \text{8-bit }(251, 251, 252)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0106$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(46,30)$ — corner dưới-phải (góc tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-7.5000, -19.5000) | -3.003984 | 0.049589 | 0.004959 | 1.000000 | cộng | (0.003967, 0.000992, 0.000992) | 0.995041 |
| 2 | 4 | (0.5, 0.5, 0.5) | (-5.3571, -24.0238) | -4.144392 | 0.015853 | 0.001585 | 0.995041 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.995041 |
| 3 | 2 | (0.2, 0.7, 0.3) | (-4.7222, -16.2778) | -2.022728 | 0.132294 | 0.013229 | 0.995041 | cộng | (0.002633, 0.009215, 0.003949) | 0.981877 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-13.7000, -20.1000) | -3.703756 | 0.024631 | 0.002463 | 0.981877 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.981877 |

$T_{final}=0.981877$, $\sum_n c_n\alpha_nT_n=(0.006600, 0.010206, 0.004941)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.988477, 0.992084, 0.986818)}\ \to\ \text{8-bit }(252, 253, 252)$$

*Nhận xét:* 2/4 Gaussian thực sự đóng góp (2 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0132$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(40,17)$ — cạnh trên, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-1.5000, -6.5000) | -0.305962 | 0.736414 | 0.073641 | 1.000000 | cộng | (0.058913, 0.014728, 0.014728) | 0.926359 |
| 2 | 4 | (0.5, 0.5, 0.5) | (0.6429, -11.0238) | -0.800352 | 0.449171 | 0.044917 | 0.926359 | cộng | (0.020805, 0.020805, 0.020805) | 0.884749 |
| 3 | 2 | (0.2, 0.7, 0.3) | (1.2778, -3.2778) | -0.084632 | 0.918851 | 0.091885 | 0.884749 | cộng | (0.016259, 0.056907, 0.024389) | 0.803454 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-7.7000, -7.1000) | -0.683850 | 0.504670 | 0.050467 | 0.803454 | cộng | (0.004055, 0.012164, 0.036493) | 0.762906 |

$T_{final}=0.762906$, $\sum_n c_n\alpha_nT_n=(0.100032, 0.104604, 0.096415)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.862938, 0.867510, 0.859321)}\ \to\ \text{8-bit }(220, 221, 219)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0919$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(40,30)$ — cạnh dưới, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (-1.5000, -19.5000) | -2.612233 | 0.073371 | 0.007337 | 1.000000 | cộng | (0.005870, 0.001467, 0.001467) | 0.992663 |
| 2 | 4 | (0.5, 0.5, 0.5) | (0.6429, -24.0238) | -3.812603 | 0.022091 | 0.002209 | 0.992663 | $\alpha<1/255\to$ bỏ | (0.000000, 0.000000, 0.000000) | 0.992663 |
| 3 | 2 | (0.2, 0.7, 0.3) | (1.2778, -16.2778) | -1.877724 | 0.152938 | 0.015294 | 0.992663 | cộng | (0.003036, 0.010627, 0.004554) | 0.977481 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-7.7000, -20.1000) | -2.892890 | 0.055416 | 0.005542 | 0.977481 | cộng | (0.000542, 0.001625, 0.004875) | 0.972065 |

$T_{final}=0.972065$, $\sum_n c_n\alpha_nT_n=(0.009448, 0.013720, 0.010897)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.981512, 0.985784, 0.982962)}\ \to\ \text{8-bit }(250, 251, 251)$$

*Nhận xét:* 3/4 Gaussian thực sự đóng góp (1 bị cửa $\alpha<1/255$); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0153$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

#### Tile 5, pixel $x=(33,24)$ — cạnh trái, giữa (biên tile)

$\mathcal G_T=\{1, 4, 2, 3\}$ (đã theo depth tăng dần). Với mỗi
$n$, $\Delta=x-\mu'_n=(\Delta_u,\Delta_v)$:

| $n$ | $i$ | $c_i$ | $\Delta=(\Delta_u,\Delta_v)$ | power | $G_n=e^{\text{power}}$ | $\alpha_n(x)$ | $T_n$ (trước) | cửa | đóng góp $c_i\alpha_nT_n$ | $T_{n+1}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (0.8, 0.2, 0.2) | (5.5000, -13.5000) | -1.379403 | 0.251729 | 0.025173 | 1.000000 | cộng | (0.020138, 0.005035, 0.005035) | 0.974827 |
| 2 | 4 | (0.5, 0.5, 0.5) | (7.6429, -18.0238) | -2.343566 | 0.095985 | 0.009598 | 0.974827 | cộng | (0.004678, 0.004678, 0.004678) | 0.965470 |
| 3 | 2 | (0.2, 0.7, 0.3) | (8.2778, -10.2778) | -1.132418 | 0.322253 | 0.032225 | 0.965470 | cộng | (0.006223, 0.021779, 0.009334) | 0.934358 |
| 4 | 3 | (0.1, 0.3, 0.9) | (-0.7000, -14.1000) | -1.228254 | 0.292803 | 0.029280 | 0.934358 | cộng | (0.002736, 0.008207, 0.024622) | 0.906999 |

$T_{final}=0.906999$, $\sum_n c_n\alpha_nT_n=(0.033775, 0.039699, 0.043669)$.

$$C(x)=\sum_n c_n\alpha_nT_n+T_{final}\cdot(1,1,1)=\mathbf{(0.940774, 0.946699, 0.950669)}\ \to\ \text{8-bit }(240, 241, 242)$$

*Nhận xét:* 4/4 Gaussian thực sự đóng góp (không splat nào bị lọc); Gaussian đóng góp nhiều
nhất vào $C(x)$ là $i=2$ (rank $n=3$, $\alpha_n(x)=0.0322$) —
phù hợp trực giác: pixel càng gần $\mu'_i$ (Δ nhỏ) và Gaussian đó càng gần camera (T lớn) thì
đóng góp $c_i\alpha_nT_n$ càng lớn.

---

## 4.4 — Ảnh render đầy đủ $\hat I^{(v)}$ cho cả 3 camera (48×32 pixel)

144 pixel ở §4.3 minh hoạ đầy đủ phép toán, nhưng khối Rasterizer thực sự chạy vòng lặp đó cho **cả**
$48\times32=1536$ pixel mỗi camera ($4608$ pixel cho 3 camera). Dưới đây là kết quả đầy đủ, sinh bởi
chính script ở Phụ lục (không chép tay), theo 3 dạng biểu diễn giống hệt cách chương 9 trình bày cho
camera 1: (a) ASCII art độ sáng trung bình RGB, (b) bản đồ $T_{\text{final}}$ (in $\lfloor9T\rfloor$,
$9=$nền hoàn toàn), (c) bản đồ `n_contrib` (chỉ số 1-based của Gaussian cuối cùng đóng góp thật).
Mọi pixel **ngoài** hợp các compact box (tức `n_contrib=0`, không Gaussian nào cả 3 cửa lọc còn giữ)
nhận đúng màu nền $(1,1,1)$ theo định nghĩa — không có phép toán nào chạy cho các pixel đó ngoài
gán `out_color = bg`.

### Camera 1 — ảnh render đầy đủ

**(a) ASCII art độ sáng trung bình RGB** (tối→sáng `@%#*+=-:. `, nền trắng = khoảng trắng; mỗi ký tự = 1 pixel, hàng = $v=0..31$, cột = $u=0..47$):

```
                                                
                     ........                   
                  .............                 
                .................               
               ...................              
              .....................             
             .......................            
             ........................           
            .........................           
           ...........................          
           ...........................          
           ............................         
           ............::::............         
          ............::::::...........         
          ...........::::::::..........         
          ...........::::::::..........         
          ............:::::::..........         
           ...........::::::...........         
           ............::::............         
           ............................         
            ...........................         
            ..........................          
             .........................          
             ........................           
              ......................            
               .....................            
                ..................              
                  ...............               
                    ...........                 
                        ...                     
                                                
                                                
```

**(b) Bản đồ $T_{\text{final}}$** (in $\lfloor9T_{\text{final}}\rfloor$; $9=$ nền hoàn toàn $T=1$, nhỏ hơn = bị Gaussian che nhiều):

```
999888888888888888888888888888888888888888888899
998888888888888888888887777888888888888888888899
988888888888888888887777777777888888888888888889
988888888888888888777777777777778888888888888889
888888888888888877777777777777777788888888888889
888888888888888777777777777777777778888888888888
888888888888887777777777777777777778888888888888
888888888888877777777766666677777777888888888888
888888888888877777776666666666777777788888888888
888888888888777777766666666666677777788888888888
888888888888777777666666666666667777778888888888
888888888888777777666666666666667777778888888888
888888888887777776666666666666666777778888888888
888888888887777776666666666666666777778888888888
888888888887777776666666666666666777777888888888
888888888887777776666666666666666777777888888888
888888888887777776666666666666666777777888888888
888888888887777776666666666666666777778888888888
888888888888777777666666666666667777778888888888
888888888888777777666666666666667777778888888888
888888888888777777766666666666677777778888888888
888888888888877777776666666666777777788888888888
888888888888887777777766666677777777788888888888
888888888888887777777777777777777777888888888888
888888888888888777777777777777777778888888888888
988888888888888877777777777777777788888888888888
998888888888888888777777777777777888888888888888
998888888888888888887777777777788888888888888888
999888888888888888888887777788888888888888888888
999988888888888888888888888888888888888888888889
999998888888888888888888888888888888888888888899
999999888888888888888888888888888888888888888899
```

**(c) Bản đồ `n_contrib`** (chỉ số 1-based, theo thứ tự depth, của Gaussian cuối cùng thực sự đóng góp; $0=$ nền):

```
000444444444444444444444444444444444444322222200
004444444444444444444444444444444444444432222200
044444444444444444444444444444444444444433222220
044444444444444444444444444444444444444443332220
444444444444444444444444444444444444444443332220
444444444444444444444444444444444444444444333222
444444444444444444444444444444444444444444333322
444444444444444444444444444444444444444444433332
444444444444444444444444444444444444444444433332
444444444444444444444444444444444444444444433333
444444444444444444444444444444444444444444433333
444444444444444444444444444444444444444444433333
444444444444444444444444444444444444444444443333
444444444444444444444444444444444444444444443333
444444444444444444444444444444444444444444443333
444444444444444444444444444444444444444444443333
444444444444444444444444444444444444444444443333
444444444444444444444444444444444444444444433333
444444444444444444444444444444444444444444433333
444444444444444444444444444444444444444444433333
444444444444444444444444444444444444444444433333
444444444444444444444444444444444444444444433333
444444444444444444444444444444444444444444333333
444444444444444444444444444444444444444444333333
444444444444444444444444444444444444444443333333
044444444444444444444444444444444444444443333333
004444444444444444444444444444444444444433333333
004444444444444444444444444444444444444433333333
000444444444444444444444444444444444444333333333
000044444444444444444444444444444444443333333330
000004444444444444444444444444444444433333333300
000000444444444444444444444444444444333333333300
```

| Thống kê trên 1536 pixel (camera 1) | R | G | B |
|---|---|---|---|
| min | 0.824703 | 0.812837 | 0.824967 |
| max | 1.000000 | 1.000000 | 1.000000 |
| mean | 0.939598 | 0.940363 | 0.946267 |

| Đại lượng | min | max | mean |
|---|---|---|---|
| `final_Ts` | 0.685838 | 1.000000 | 0.897887 |
| `n_contrib` | 0 | 4 | 3.746094 |
| số Gaussian cộng thật (`contribs`) | 0 | 4 | 3.215495 |

Pixel nền (`n_contrib=0`, ngoài hợp mọi compact box): **42** / 1536 ($=2.73\%$). Pixel early-termination ($T(1-\alpha)<10^{-4}$): **0**.

**(d) Bảng đầy đủ giá trị 8-bit $(R,G,B)$ của toàn bộ 1536 pixel** (mỗi dòng là một hàng ảnh
$v=0..31$; trong dòng liệt kê tuần tự $u=0..47$ dạng `RRGGBB` hex — đây chính là toàn bộ nội dung
của $\hat I^{(cam1)}$, không lấy mẫu, tính bằng đúng công thức §4.2 cho từng pixel một):

```
v=00: FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FDFDFF FCFCFE FBFBFD FAFAFD FAF9FC F9F9FC F8F8FB F7F7FA F6F6FA F6F5F9 F4F4F8 F3F3F7 F3F2F6 F2F2F5 F2F1F5 F1F1F4 F1F1F4 F1F0F4 F1F1F3 F1F1F3 F2F1F4 F2F2F4 F3F3F4 F4F3F5 F4F4F6 F5F5F6 F6F6F7 F7F7F8 F8F8F9 F9F9F9 FAFAFA FBFBFB FCFDFC FDFDFD FEFEFE FEFEFE FEFEFE FEFEFE FEFEFE FFFFFF FFFFFF
v=01: FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFEFF FDFDFF FCFDFF FBFBFD FAFAFD F9F9FC F9F8FC F8F7FB F7F6FA F6F5F9 F4F4F8 F3F3F7 F2F2F6 F1F1F5 F1F0F4 F0EFF3 EFEFF3 EFEEF2 EFEEF2 EFEEF1 EFEEF1 EFEEF1 EFEFF1 F0F0F2 F1F0F2 F2F1F3 F3F2F4 F4F3F5 F5F5F5 F6F6F6 F7F7F7 F8F8F8 F9F9F9 FAFAFA FBFBFB FCFDFC FDFDFD FEFEFE FEFEFE FEFEFE FEFEFE FFFFFF FFFFFF
v=02: FFFFFF FEFEFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FBFBFD FAFAFD F9F9FC F9F8FC F8F7FB F7F6FA F6F5F9 F4F3F8 F3F2F7 F1F1F6 F0F0F5 EFEFF3 EEEDF2 EDEDF1 EDECF1 ECEBF0 ECEBEF ECEBEF ECEBEF ECECEF EDECEF EEEDEF EEEEF0 EFEFF1 F1F0F2 F2F2F3 F3F3F4 F4F4F5 F5F6F6 F7F7F7 F8F8F8 F9F9F9 FAFAFA FCFCFB FCFDFC FEFEFE FEFEFE FEFEFE FEFEFE FEFEFE FFFFFF
v=03: FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FCFCFE FAFAFD FAFAFC F9F9FC F8F7FB F7F6FA F5F5F9 F3F3F8 F2F2F7 F1F0F5 EFEFF4 EEEDF3 EDECF2 ECEBF0 EBEAEF EAE9EE EAE8ED E9E8ED E9E8EC E9E8EC EAE9EC EAE9EC EBEAED ECEBEE EDEDEF EEEEF0 F0F0F1 F1F1F2 F3F3F3 F4F4F5 F5F6F6 F7F7F7 F8F8F8 F9F9F9 FAFAFA FCFCFB FCFDFD FDFDFD FEFEFE FEFEFE FEFEFE FFFFFF
v=04: FEFEFF FEFEFF FDFEFF FDFDFF FDFDFF FCFDFF FBFBFD FAFAFD F9F9FC F8F8FB F7F6FB F5F5FA F3F3F8 F2F1F7 F0F0F5 EFEEF4 EDECF2 ECEBF1 EAE9F0 E9E8EE E8E7ED E7E6EC E7E5EB E7E5EA E6E5EA E7E5E9 E7E6E9 E8E6EA E8E8EA E9E9EB EBEAEC ECECED EEEDEF EFEFF0 F1F1F2 F2F3F3 F4F4F4 F5F6F6 F7F7F7 F8F9F8 F9FAFA FBFBFB FCFDFC FDFDFD FEFEFE FEFEFE FEFEFE FFFFFF
v=05: FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FCFCFE FAFAFD F9F9FC F8F8FC F7F7FB F6F5FA F4F3F8 F2F2F7 F0F0F6 EFEEF4 EDECF2 EBEAF1 EAE8EF E8E7ED E7E5EC E6E4EA E5E3E9 E4E2E8 E4E2E7 E4E2E7 E4E2E6 E4E3E7 E5E4E7 E6E5E8 E7E6E8 E8E8EA EAE9EB EBEBEC EDEDEE EFEFF0 F1F1F1 F3F3F3 F4F5F5 F6F6F6 F7F8F8 F9F9F9 FAFAFA FBFCFB FCFDFC FDFDFD FEFEFE FEFEFE FEFEFE
v=06: FEFEFF FDFEFF FDFDFF FDFDFF FCFDFF FBFBFE FAFAFD F9F9FC F7F7FB F6F6FA F5F4F9 F2F2F7 F0F0F6 EFEEF4 EDECF3 EBEAF1 E9E8EF E7E6ED E6E4EB E4E2EA E3E1E8 E2E0E7 E1DFE5 E1DFE4 E1DFE4 E1DFE4 E1E0E4 E2E1E4 E3E2E5 E4E3E6 E6E5E7 E8E7E8 E9E9EA EBEBEC EDEDEE EFF0F0 F1F2F2 F3F4F3 F5F5F5 F6F7F7 F8F8F8 F9FAF9 FBFBFA FCFDFC FDFDFD FDFEFD FEFEFE FEFEFE
v=07: FEFEFF FDFEFF FDFDFF FCFDFF FCFCFE FBFBFD F9F9FC F8F8FC F7F6FB F5F5FA F3F3F8 F1F1F7 EFEFF5 EDECF3 EBEAF1 E9E8EF E7E6ED E5E3EB E3E1E9 E2E0E7 E0DEE5 DFDDE4 DFDCE3 DEDCE2 DEDCE1 DEDCE1 DFDDE1 DFDEE1 E0DFE2 E2E1E3 E3E3E4 E5E5E6 E7E7E8 E9E9EA EBECEC EEEEEE F0F0F0 F2F2F2 F4F4F4 F5F6F6 F7F8F7 F8F9F9 FAFAFA FBFCFB FCFDFC FDFEFD FDFEFD FEFEFE
v=08: FEFEFF FDFEFF FDFDFF FCFDFF FBFBFE FAFAFD F9F9FC F8F7FB F6F6FA F4F4F9 F2F2F7 F0EFF6 EEEDF4 ECEBF2 E9E8F0 E7E6EE E5E3EC E3E1E9 E1DFE7 E0DDE5 DEDCE3 DDDAE1 DCD9E0 DCD9DF DBD9DE DCD9DE DCDADE DDDBDE DEDDDF DFDEE0 E1E0E2 E3E3E4 E5E5E6 E7E8E8 EAEAEA ECEDEC EEEFEF F0F1F1 F3F3F3 F4F5F5 F6F7F6 F8F9F8 F9FAF9 FBFBFB FCFDFC FDFDFD FDFEFD FEFEFE
v=09: FDFEFF FDFDFF FDFDFF FCFCFE FBFBFD FAF9FD F8F8FC F7F7FB F6F5FA F3F3F8 F1F1F7 EFEEF5 EDECF3 EAE9F1 E8E7EF E6E4EC E3E1EA E1DFE8 DFDDE5 DDDBE3 DCD9E1 DBD8DF DAD7DE D9D6DC D9D6DC D9D7DB DAD7DB DBD9DC DCDADD DDDCDE DFDEE0 E1E1E2 E3E3E4 E6E6E6 E8E9E8 EAEBEB EDEEED EFF0F0 F2F3F2 F4F5F4 F5F6F6 F7F8F7 F9F9F9 FBFBFA FCFCFB FCFDFD FDFEFD FDFEFE
v=10: FDFEFF FDFDFF FCFDFF FCFCFE FBFBFD F9F9FC F8F8FC F7F6FB F4F4F9 F2F2F8 F0F0F6 EEEDF4 ECEBF2 E9E8F0 E7E5EE E4E2EB E2E0E8 E0DDE6 DEDBE3 DCD9E1 DAD7DF D9D6DD D8D5DB D7D4DA D7D4D9 D7D4D9 D8D5D9 D9D7D9 DAD8DA DBDADC DDDDDD DFDFDF E2E2E2 E4E5E4 E7E7E7 E9EAE9 ECEDEC EEF0EE F1F2F1 F3F4F3 F5F6F5 F7F8F7 F8F9F8 FAFBFA FBFCFB FCFDFC FDFEFD FDFEFD
v=11: FDFEFF FDFDFF FCFDFF FBFBFE FBFAFD F9F9FC F8F7FB F6F6FB F4F4F9 F2F1F7 F0EFF6 EDEDF4 EBEAF1 E8E7EF E6E4ED E3E1EA E1DEE7 DEDCE4 DCD9E2 DAD7DF D9D5DD D7D4DB D6D3D9 D6D2D8 D5D2D7 D5D3D7 D6D3D7 D7D5D7 D8D6D8 DAD9DA DCDBDC DEDEDE E0E1E0 E3E3E3 E5E6E5 E8E9E8 EBECEB EDEFED F0F1F0 F2F4F2 F4F6F4 F6F7F6 F8F9F8 FAFBF9 FBFCFB FCFDFC FDFEFD FDFEFD
v=12: FDFEFF FDFDFF FCFDFF FBFBFE FAFAFD F9F9FC F8F7FB F6F5FA F3F3F8 F1F1F7 EFEFF5 EDECF3 EAE9F1 E7E6EE E5E3EC E2E0E9 E0DDE6 DDDAE3 DBD8E1 D9D6DE D7D4DC D6D2D9 D5D1D8 D4D1D6 D4D1D5 D4D1D5 D5D2D5 D5D3D6 D7D5D7 D8D7D8 DADADA DCDDDC DFE0DF E1E3E1 E4E6E4 E7E9E7 EAECEA ECEEED EFF1EF F1F3F2 F4F5F4 F6F7F6 F7F9F7 F9FAF9 FBFCFA FCFDFC FDFEFD FDFEFD
v=13: FDFEFF FDFDFF FCFCFE FBFBFE FAFAFD F9F9FC F7F7FB F6F5FA F3F3F8 F1F1F7 EFEEF5 ECEBF3 EAE9F0 E7E6EE E4E2EB E2DFE8 DFDCE5 DDDAE2 DAD7E0 D8D5DD D7D3DA D5D1D8 D4D0D6 D3D0D5 D3D0D4 D3D0D4 D4D1D4 D4D2D4 D6D4D5 D7D7D7 D9D9D9 DBDCDB DEDFDE E0E2E0 E3E5E3 E6E8E6 E9EBE9 ECEEEC EEF0EF F1F3F1 F3F5F3 F5F7F5 F7F8F7 F9FAF9 FBFBFA FCFCFB FCFEFD FDFEFD
v=14: FDFEFF FDFDFF FCFCFE FBFBFD FAFAFD F9F8FC F7F7FB F5F5F9 F3F3F8 F1F1F6 EFEEF5 ECEBF2 EAE8F0 E7E5EE E4E2EB E1DFE8 DFDCE5 DCD9E2 DAD7DF D8D4DC D6D2DA D5D1D7 D4D0D6 D3CFD4 D2CFD3 D2D0D3 D3D1D3 D4D2D3 D5D4D4 D7D6D6 D8D9D8 DBDCDA DDDFDD E0E2E0 E3E5E2 E6E8E5 E8EBE8 EBEEEB EEF0EE F1F3F1 F3F5F3 F5F7F5 F7F8F7 F8FAF9 FAFBFA FBFCFB FCFEFD FDFEFD
v=15: FDFEFF FDFDFF FCFCFE FBFBFD FAFAFD F9F9FC F7F7FB F5F5F9 F3F3F8 F1F1F6 EFEEF5 ECEBF2 EAE8F0 E7E5ED E4E2EB E1DFE8 DFDCE5 DCD9E2 DAD7DF D8D4DC D6D3D9 D5D1D7 D3D0D5 D3CFD4 D2CFD3 D2D0D2 D3D1D2 D4D2D3 D5D4D4 D6D6D6 D8D9D7 DADCDA DDDFDC E0E2DF E2E5E2 E5E8E5 E8EBE8 EBEEEB EEF0EE F0F3F0 F3F5F3 F5F7F5 F7F8F7 F8FAF8 FAFBFA FBFCFB FCFEFD FDFEFD
v=16: FDFEFF FDFDFF FCFCFE FBFBFD FAFAFD F9F9FC F8F7FB F5F5F9 F3F3F8 F1F1F6 EFEEF5 EDECF2 EAE9F0 E7E6EE E4E3EB E2E0E8 DFDDE5 DDDAE2 DAD7DF D8D5DC D6D3DA D5D2D7 D4D1D5 D3D0D4 D3D0D3 D3D0D2 D3D1D2 D4D3D3 D5D5D4 D6D7D6 D8D9D7 DADCDA DDDFDC E0E2DF E2E5E2 E5E8E5 E8EBE8 EBEEEB EEF1EE F0F3F0 F3F5F3 F5F7F5 F7F8F7 F8FAF8 FAFBFA FBFCFB FCFEFD FDFEFD
v=17: FDFEFF FDFDFF FCFCFE FBFBFE FBFAFD FAF9FD F8F7FB F6F5FA F4F3F8 F2F1F7 EFEFF5 EDECF3 EAE9F0 E8E6EE E5E3EB E2E0E8 E0DEE5 DDDBE2 DBD8DF D9D6DD D7D4DA D6D3D8 D5D2D6 D4D1D5 D3D1D4 D3D1D3 D4D2D3 D4D4D4 D5D6D5 D7D8D6 D9DAD8 DBDDDA DDE0DD E0E3E0 E3E6E2 E6E9E5 E8ECE8 EBEEEB EEF1EE F0F3F1 F3F5F3 F5F7F5 F7F9F7 F9FBF9 FAFCFA FBFCFB FCFEFD FDFEFD
v=18: FDFEFF FDFDFF FCFCFE FCFBFE FBFBFD FAF9FD F8F8FB F6F6FA F4F4F8 F2F2F7 F0EFF5 EEEDF3 EBEAF1 E9E7EE E6E4EC E3E2E9 E1DFE6 DEDCE3 DCDAE0 DAD7DE D8D6DB D7D4D9 D6D3D7 D5D3D6 D4D3D5 D4D3D4 D5D4D4 D5D5D5 D6D7D6 D8D9D7 DADBD9 DCDEDB DEE1DD E1E4E0 E3E7E3 E6EAE6 E9ECE9 ECEFEC EEF1EE F1F4F1 F3F6F3 F5F7F5 F7F9F7 F9FBF9 FAFCFA FBFCFB FCFEFD FDFEFE
v=19: FEFEFF FDFEFF FDFDFF FCFCFE FBFBFD FAFAFD F9F9FC F6F6FA F5F4F9 F3F2F7 F1F0F6 EFEEF4 ECEBF1 EAE8EF E7E6EC E5E3EA E2E0E7 E0DEE4 DEDBE2 DCD9DF DAD8DC D8D6DA D7D5D9 D6D5D7 D6D5D6 D6D5D6 D6D6D6 D7D7D6 D8D9D7 D9DBD8 DBDDDA DDE0DC DFE2DF E2E5E1 E4E8E4 E7EAE7 EAEDEA ECF0EC EFF2EF F1F4F1 F3F6F4 F5F8F6 F7F9F7 F9FBF9 FBFCFA FCFDFC FCFEFD FDFEFE
v=20: FEFEFF FDFEFF FDFDFF FCFCFE FBFBFD FAFAFD F9F9FC F7F7FA F5F5F9 F4F3F8 F2F1F6 F0EFF4 EDECF2 EBEAF0 E9E7ED E6E5EB E4E2E8 E1E0E6 DFDDE3 DDDBE1 DCDADE DAD9DC D9D8DA D8D7D9 D8D7D8 D8D7D8 D8D8D8 D8D9D8 D9DBD9 DBDDDA DCDFDC DEE1DE E0E4E0 E3E6E3 E5E9E5 E8ECE8 EAEEEB EDF1ED EFF3F0 F2F5F2 F4F7F4 F6F8F6 F7F9F8 FAFBF9 FBFCFA FCFDFC FDFEFD FDFEFE
v=21: FEFEFF FDFEFF FDFDFF FCFCFE FCFBFD FBFAFD FAF9FC F8F8FB F6F6FA F4F4F8 F3F2F7 F1F0F5 EFEEF3 ECEBF1 EAE9EF E8E7EC E6E4EA E3E2E7 E1E0E5 DFDEE2 DEDCE0 DCDBDE DBDADD DADADB DADADA DADADA DADBDA DADCDA DBDDDB DDDFDC DEE1DE E0E3E0 E2E6E2 E4E8E4 E7EBE7 E9EDE9 ECEFEC EEF2EE F0F4F0 F2F5F3 F4F7F5 F6F9F6 F8FAF8 FAFBF9 FBFCFB FCFEFC FDFEFD FDFEFE
v=22: FEFEFF FEFEFF FDFEFF FCFCFE FCFCFE FBFBFD FAFAFD F9F8FB F7F7FA F5F5F9 F4F3F7 F2F1F6 F0EFF4 EEEDF2 ECEBF0 EAE9EE E7E6EB E5E4E9 E4E2E7 E2E1E5 E0DFE2 DFDEE1 DEDDDF DDDDDE DCDDDD DCDDDC DCDEDC DDDFDD DEE0DD DFE2DF E0E3E0 E2E5E2 E4E8E4 E6EAE6 E8ECE8 EBEEEB EDF1ED EFF3EF F1F5F1 F3F6F4 F5F8F5 F7F9F7 F9FBF9 FAFCFA FBFCFB FDFEFD FDFEFD FEFEFE
v=23: FEFEFF FEFEFF FDFEFF FDFDFF FCFCFE FCFBFD FBFAFD FAF9FC F8F8FB F6F6F9 F5F4F8 F3F3F7 F1F1F5 EFEFF3 EDEDF1 ECEBEF EAE9ED E8E7EB E6E5E9 E4E3E7 E3E2E5 E1E1E3 E0E0E2 E0E0E1 DFE0E0 DFE0DF DFE1DF DFE2DF E0E3E0 E1E4E1 E3E6E2 E4E8E4 E6EAE6 E8ECE8 EAEEEA ECF0EC EEF2EE F0F4F1 F2F5F3 F4F7F4 F6F8F6 F7FAF8 FAFBF9 FBFCFA FCFDFC FDFEFD FDFEFD FEFEFE
v=24: FEFEFF FEFEFF FEFEFF FDFEFF FCFCFE FCFCFE FBFBFD FBFAFD F9F9FB F8F7FA F6F6F9 F4F4F7 F3F2F6 F1F0F4 EFEFF3 EDEDF1 ECEBEF EAE9ED E8E8EB E7E6E9 E5E5E7 E4E4E6 E3E3E4 E2E3E3 E2E3E3 E2E3E2 E2E4E2 E2E4E2 E3E6E3 E4E7E4 E5E8E5 E7EAE7 E8ECE8 EAEEEA ECF0EC EEF1EE F0F3F0 F2F5F2 F3F6F4 F5F8F5 F7F9F7 F9FBF8 FAFCFA FBFCFB FCFEFD FDFEFD FDFEFE FEFFFE
v=25: FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFCFE FCFBFD FBFBFD FAF9FC F9F8FB F7F7F9 F6F5F8 F4F4F7 F3F2F6 F1F1F4 EFEFF2 EEEDF1 ECECEF EBEAED E9E9EB E8E8EA E7E7E9 E6E6E7 E5E6E6 E5E6E6 E5E6E5 E5E7E5 E5E7E5 E6E8E6 E7EAE7 E8EBE8 E9ECE9 EAEEEB ECF0EC EEF1EE EFF3F0 F1F5F1 F3F6F3 F5F7F5 F6F9F6 F8FAF8 FAFBF9 FAFCFA FCFEFC FDFEFD FDFEFD FDFEFE FEFFFE
v=26: FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFCFE FCFCFE FCFBFD FAFAFC F9F9FB F8F8FB F7F7F9 F5F5F8 F4F4F7 F3F2F5 F1F1F4 F0F0F2 EEEEF1 EDEDEF ECECEE EBEBEC EAEAEB E9EAEA E8E9E9 E8E9E9 E8E9E8 E8EAE8 E8EAE8 E8EBE9 E9ECE9 EAEDEA EBEFEC EDF0ED EEF2EE F0F3F0 F1F5F1 F3F6F3 F4F7F5 F6F8F6 F7F9F7 F9FBF9 FAFCFA FBFDFB FCFEFD FDFEFD FDFEFD FEFEFE FEFFFE
v=27: FFFFFF FFFFFF FEFEFF FEFEFF FEFEFF FDFEFF FDFCFE FCFCFD FCFBFD FAFAFC F9F9FB F8F8FA F7F7F9 F6F5F8 F4F4F7 F3F3F5 F2F2F4 F1F1F3 EFEFF1 EEEEF0 EDEEEF ECEDEE ECECED EBECEC EBECEB EAECEB EAEDEB EBEDEB EBEEEC ECEFEC EDF0ED EEF1EE EFF2EF F0F3F0 F2F5F2 F3F6F3 F4F7F5 F6F8F6 F7F9F7 F8FAF8 FAFCF9 FBFDFB FCFEFC FDFEFD FDFEFD FDFEFE FEFFFE FEFFFE
v=28: FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFE FDFCFE FCFCFD FBFBFC FAFAFC F9F9FB F8F8FA F7F7F9 F6F6F8 F5F5F7 F4F4F6 F3F3F5 F2F2F3 F1F1F2 F0F0F1 EFF0F0 EEEFEF EEEFEF EDEFEE EDEFEE EDEFEE EDF0EE EEF0EE EEF1EF EFF2F0 F0F3F0 F1F4F1 F2F5F2 F3F6F4 F5F7F5 F6F8F6 F7F9F7 F8FAF8 FAFBF9 FBFDFB FCFDFB FCFEFD FDFEFD FDFEFD FEFEFE FEFFFE FEFFFE
v=29: FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFE FDFCFE FCFCFD FBFBFC FAFAFC F9F9FB F9F9FA F8F8F9 F6F6F8 F6F6F7 F5F5F6 F4F4F5 F3F3F4 F2F3F3 F1F2F2 F1F2F2 F0F2F1 F0F1F1 F0F2F0 F0F2F0 F0F2F0 F0F3F1 F1F3F1 F1F4F2 F2F5F3 F3F6F3 F4F7F4 F5F8F5 F6F8F6 F7F9F7 F8FAF8 FAFCFA FBFCFB FCFDFB FCFEFD FDFEFD FDFEFD FEFEFE FEFFFE FEFFFE FFFFFF
v=30: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FEFEFF FDFDFE FDFCFE FCFCFD FBFBFC FAFAFC FAFAFB F9F9FA F8F8FA F8F8F9 F6F7F8 F6F6F7 F5F5F6 F4F5F5 F4F4F5 F3F4F4 F3F4F3 F2F4F3 F2F4F3 F2F4F3 F2F4F3 F3F5F3 F3F5F3 F4F6F4 F4F7F5 F5F7F5 F6F8F6 F7F9F7 F7FAF8 F8FAF9 FAFCFA FBFCFB FCFDFB FCFEFD FDFEFD FDFEFD FDFEFE FEFFFE FEFFFE FFFFFF FFFFFF
v=31: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FEFEFF FDFDFE FDFCFE FCFCFD FBFBFC FBFBFC FAFAFB F9FAFB F9F9FA F8F8F9 F8F8F9 F7F7F8 F6F7F7 F6F6F6 F5F6F6 F5F6F5 F5F6F5 F4F6F5 F4F6F5 F5F6F5 F5F7F5 F5F7F5 F6F8F6 F6F8F6 F7F9F7 F7F9F8 F8FAF8 F9FBF9 FBFCFA FBFDFB FCFDFB FCFEFD FDFEFD FDFEFD FDFEFE FEFEFE FEFFFE FEFFFE FFFFFF FFFFFF
```

### Camera 2 — ảnh render đầy đủ

**(a) ASCII art độ sáng trung bình RGB** (tối→sáng `@%#*+=-:. `, nền trắng = khoảng trắng; mỗi ký tự = 1 pixel, hàng = $v=0..31$, cột = $u=0..47$):

```
                                                
      ..........                                
    ..............                              
  ..................                            
 ....................                           
......................                          
.......................                         
........................                        
........................                        
.........................                       
.........................                       
...........:..............                      
.........:::::............                      
........:::::::...........                      
.......::::::::...........                      
.......::::::::...........                      
........:::::::...........                      
........:::::::...........                      
.........:::::............                      
..........................                      
..........................                      
.........................                       
.........................                       
........................                        
.......................                         
 .....................                          
  ...................                           
    ................                            
      ............                              
          ....                                  
                                                
                                                
```

**(b) Bản đồ $T_{\text{final}}$** (in $\lfloor9T_{\text{final}}\rfloor$; $9=$ nền hoàn toàn $T=1$, nhỏ hơn = bị Gaussian che nhiều):

```
888888888888888888888888888888889999999999999999
888888887777778888888888888888889999999999999999
888887777777777778888888888888889999999999999999
888777777777777777788888888888889999999999999999
887777777777777777778888888888889999999999999999
877777777777777777777888888888889999999999999999
777777777777777777777788888888889999999999999999
777777776666667777777778888888889999999999999999
777777666666666677777777888888889999999999999999
777776666666666667777777888888889999999999999999
777766666666666666777777788888889999999999999999
777766666666666666677777788888889999999999999999
777666666666666666677777788888889999999999999999
777666666666666666667777788888889999999999999999
777666666666666666667777778888889999999999999999
777666666666666666667777778888889999999999999999
777666666666666666667777778888889999999999999999
777666666666666666667777788888889999999999999999
777766666666666666677777788888889999999999999999
777766666666666666677777788888889999999999999999
777776666666666666777777788888889999999999999999
777777666666666667777777888888889999999999999999
777777776666666777777777888888889999999999999999
777777777777777777777778888888889999999999999999
877777777777777777777788888888889999999999999999
888777777777777777777888888888889999999999999999
888877777777777777778888888888889999999999999999
888888777777777777888888888888889999999999999999
888888888777777888888888888888889999999999999999
888888888888888888888888888888889999999999999999
888888888888888888888888888888889999999999999999
888888888888888888888888888888889999999999999999
```

**(c) Bản đồ `n_contrib`** (chỉ số 1-based, theo thứ tự depth, của Gaussian cuối cùng thực sự đóng góp; $0=$ nền):

```
444444444444444444444444444422220000000000000000
444444444444444444444444444442220000000000000000
444444444444444444444444444443220000000000000000
444444444444444444444444444444320000000000000000
444444444444444444444444444444430000000000000000
444444444444444444444444444444430000000000000000
444444444444444444444444444444430000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444440000000000000000
444444444444444444444444444444430000000000000000
444444444444444444444444444444430000000000000000
444444444444444444444444444444330000000000000000
444444444444444444444444444444330000000000000000
444444444444444444444444444443330000000000000000
444444444444444444444444444433330000000000000000
444444444444444444444444444333330000000000000000
444444444444444444444444443333330000000000000000
444444444444444444444444433333330000000000000000
```

| Thống kê trên 1536 pixel (camera 2) | R | G | B |
|---|---|---|---|
| min | 0.820120 | 0.810494 | 0.827216 |
| max | 1.000000 | 1.000000 | 1.000000 |
| mean | 0.944465 | 0.945763 | 0.949746 |

| Đại lượng | min | max | mean |
|---|---|---|---|
| `final_Ts` | 0.682599 | 1.000000 | 0.905896 |
| `n_contrib` | 0 | 4 | 2.630208 |
| số Gaussian cộng thật (`contribs`) | 0 | 4 | 2.546224 |

Pixel nền (`n_contrib=0`, ngoài hợp mọi compact box): **512** / 1536 ($=33.33\%$). Pixel early-termination ($T(1-\alpha)<10^{-4}$): **0**.

**(d) Bảng đầy đủ giá trị 8-bit $(R,G,B)$ của toàn bộ 1536 pixel** (mỗi dòng là một hàng ảnh
$v=0..31$; trong dòng liệt kê tuần tự $u=0..47$ dạng `RRGGBB` hex — đây chính là toàn bộ nội dung
của $\hat I^{(cam2)}$, không lấy mẫu, tính bằng đúng công thức §4.2 cho từng pixel một):

```
v=00: F7F6FA F7F6F9 F6F5F8 F5F4F8 F3F3F6 F3F2F6 F2F1F5 F1F1F4 F1F0F4 F1F0F4 F1F0F3 F1F0F3 F1F0F3 F1F1F4 F2F1F4 F2F2F4 F3F3F5 F4F4F6 F4F5F6 F5F6F7 F6F7F8 F7F8F9 F8F8F9 F9F9FA FAFAFB FBFCFC FBFCFD FDFDFE FEFEFE FEFEFE FEFEFE FEFEFE FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=01: F6F5F9 F5F4F8 F3F3F6 F2F1F5 F1F1F5 F1F0F4 F0EFF3 EFEEF2 EFEEF2 EEEEF2 EEEDF1 EEEEF1 EEEEF1 EFEEF2 EFEFF2 F0F0F2 F1F1F3 F2F2F4 F2F3F5 F4F4F6 F5F5F6 F6F6F7 F7F7F8 F8F8F9 F9F9FA FAFAFB FBFCFC FBFCFD FDFDFE FEFEFE FEFEFE FEFEFE FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=02: F5F3F7 F3F2F6 F2F0F5 F0EFF4 EFEEF3 EEEDF2 EDECF1 EDECF0 ECEBF0 ECEBEF EBEBEF EBEBEF ECEBEF ECECEF EDECF0 EDEDF0 EEEEF1 EFF0F2 F0F1F3 F2F2F4 F3F3F5 F4F5F6 F5F6F7 F7F7F8 F8F8F9 F9F9FA FAFAFB FBFCFC FCFCFD FDFEFD FEFEFE FEFEFE FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=03: F2F1F6 F1F0F4 F0EEF3 EEEDF2 EDECF1 ECEAF0 EBE9EF EAE9EE E9E8ED E9E8ED E9E8EC E9E8EC E9E8EC E9E9ED EAEAED EBEBEE ECECEF EDEDF0 EEEFF1 F0F0F2 F1F2F3 F2F3F4 F4F5F6 F5F6F7 F7F7F8 F8F8F9 F9FAFA FAFAFB FBFCFC FCFDFD FDFEFD FEFEFE FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=04: F1EFF4 EFEEF3 EEECF1 ECEAF0 EBE9EF E9E8ED E8E6EC E7E6EB E7E5EB E6E5EA E6E4EA E6E5EA E6E5EA E6E6EA E7E7EB E8E8EB E9E9EC EAEBED ECECEF EDEEF0 EFF0F1 F1F1F3 F2F3F4 F4F5F6 F5F6F7 F7F8F8 F8F9F9 F9FAFA FAFBFB FBFCFC FCFDFD FDFEFD FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=05: EFEDF3 EDEBF1 ECEAF0 EAE8EE E8E6ED E7E5EB E6E4EA E5E3E9 E4E2E8 E3E1E7 E3E1E7 E3E1E7 E3E2E7 E3E3E7 E4E4E8 E5E5E9 E6E7EA E8E8EB E9EAEC EBECEE EDEEEF EFF0F1 F0F2F3 F2F3F4 F4F5F6 F6F7F7 F7F8F8 F8F9F9 F9FAFA FBFCFC FCFDFD FDFEFD FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=06: EEEBF1 ECE9F0 EAE7EE E8E5EC E6E4EA E5E2E9 E3E1E7 E2DFE6 E1DFE5 E0DEE4 E0DEE4 E0DEE4 E0DFE4 E1E0E4 E1E1E5 E2E2E6 E4E4E7 E5E6E9 E7E8EA E9EAEC EBECEE EDEEEF EFF0F1 F1F2F3 F3F4F4 F4F6F6 F6F7F7 F7F8F9 F9FAFA FAFBFB FBFCFC FDFDFD FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=07: ECE9F0 EAE7EE E8E5EC E6E3EA E4E1E8 E2DFE6 E1DEE5 DFDDE4 DEDCE2 DEDBE2 DDDBE1 DDDBE1 DDDCE1 DEDDE2 DFDEE2 E0E0E3 E1E1E5 E3E3E6 E5E6E8 E7E8EA E9EAEC EBECEE EDEFEF EFF1F1 F1F3F3 F3F5F5 F5F6F6 F7F8F8 F8F9F9 F9FAFA FBFCFC FBFDFD FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=08: EAE8EE E8E5EC E6E3EA E4E1E8 E2DEE6 E0DDE4 DFDBE3 DDDAE1 DCD9E0 DBD8DF DBD8DE DBD8DE DBD9DE DBDADF DCDBE0 DDDDE1 DFDFE2 E0E1E4 E2E3E6 E4E6E8 E7E8EA E9EBEC EBEDEE EEF0F0 F0F2F2 F2F4F4 F4F6F6 F6F7F7 F7F9F8 F9FAFA FAFBFB FBFCFC FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=09: E9E6ED E7E3EB E4E1E9 E2DEE6 E0DCE4 DEDAE2 DCD9E0 DBD7DF DAD6DD D9D6DC D8D6DC D8D6DC D8D7DC D9D8DC DAD9DD DBDBDE DCDDE0 DEDFE2 E0E2E3 E2E4E6 E5E7E8 E7E9EA EAECEC ECEEEF EFF1F1 F1F3F3 F3F5F5 F5F7F6 F7F8F8 F8F9F9 F9FBFA FBFCFC FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=10: E8E5EC E6E2E9 E3DFE7 E1DDE5 DFDAE2 DDD8E0 DBD6DE D9D5DD D8D4DB D7D3DA D6D3DA D6D4D9 D6D4D9 D7D5DA D8D7DB D9D9DC DADBDE DCDDDF DEE0E2 E1E3E4 E3E5E6 E6E8E9 E8EBEB EBEDED EDF0F0 F0F2F2 F2F4F4 F4F6F6 F6F8F7 F8F9F9 F9FAFA FAFCFC FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=11: E7E3EB E5E0E8 E2DEE6 E0DBE3 DDD9E1 DBD7DF D9D5DD D8D3DB D6D2D9 D5D2D8 D5D1D7 D4D2D7 D4D2D7 D5D4D8 D6D5D9 D7D7DA D9D9DC DADCDE DDDEE0 DFE1E2 E2E4E5 E4E7E7 E7EAEA EAEDEC ECEFEF EFF1F1 F1F4F3 F3F6F5 F5F7F7 F7F9F8 F9FAFA FAFBFB FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=12: E6E2EA E4DFE7 E1DDE5 DFDAE2 DCD7E0 DAD5DD D8D3DB D6D2D9 D5D1D8 D4D0D7 D3D0D6 D3D0D5 D3D1D6 D3D2D6 D4D4D7 D6D6D8 D7D8DA D9DBDC DBDDDE DEE0E1 E0E3E3 E3E6E6 E6E9E9 E9ECEB ECEFEE EEF1F0 F1F3F3 F3F5F5 F5F7F6 F7F8F8 F8FAF9 FAFBFA FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=13: E6E2E9 E3DFE7 E1DCE4 DED9E1 DCD7DF D9D4DC D7D2DA D6D1D8 D4D0D7 D3CFD5 D2CFD5 D2CFD4 D2D0D4 D2D1D5 D3D3D6 D4D5D7 D6D7D9 D8DADB DADDDD DDE0E0 DFE3E2 E2E6E5 E5E9E8 E8EBEB EBEEED EEF1F0 F0F3F2 F2F5F4 F5F7F6 F6F8F8 F8FAF9 F9FBFA FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=14: E6E1E9 E3DEE6 E0DCE3 DED9E1 DBD6DE D9D4DC D7D2D9 D5D1D7 D4CFD6 D3CFD5 D2CFD4 D1CFD3 D1D0D3 D2D1D4 D3D3D5 D4D5D6 D5D7D8 D7DADA DADCDC DCDFDF DFE2E2 E2E5E4 E5E8E7 E7EBEA EAEEED EDF0EF F0F3F2 F2F5F4 F4F7F6 F6F8F7 F8FAF9 F9FBFA FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=15: E6E2E9 E3DFE6 E0DCE3 DED9E0 DBD6DE D9D4DB D7D2D9 D5D1D7 D4D0D5 D2CFD4 D2CFD3 D1CFD3 D1D0D3 D1D1D3 D2D3D4 D3D5D6 D5D7D8 D7DADA D9DCDC DCDFDF DEE2E1 E1E5E4 E4E8E7 E7EBEA EAEEEC EDF0EF F0F3F1 F2F5F4 F4F7F6 F6F8F7 F8FAF9 F9FBFA FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=16: E6E2E9 E4DFE6 E1DCE3 DED9E1 DCD7DE D9D5DC D7D3D9 D6D1D7 D4D0D6 D3D0D4 D2CFD3 D1D0D3 D1D0D3 D2D2D4 D2D3D4 D4D5D6 D5D7D8 D7DADA D9DDDC DCE0DE DEE3E1 E1E6E4 E4E9E7 E7EBEA EAEEEC EDF1EF EFF3F1 F2F5F4 F4F7F6 F6F8F7 F8FAF9 F9FBFA FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=17: E7E3E9 E4E0E7 E2DDE4 DFDAE1 DDD8DF DAD6DC D8D4DA D6D2D8 D5D1D6 D4D1D5 D3D1D4 D2D1D4 D2D2D4 D2D3D4 D3D4D5 D4D6D6 D6D8D8 D7DBDA DADEDC DCE0DF DFE3E1 E2E6E4 E4E9E7 E7ECEA EAEEED EDF1EF F0F3F1 F2F5F4 F4F7F6 F6F8F7 F8FAF9 F9FBFA FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=18: E8E4EA E5E1E7 E3DEE5 E0DCE2 DED9DF DBD7DD D9D5DB D7D4D9 D6D3D7 D5D2D6 D4D2D5 D3D2D5 D3D3D5 D3D4D5 D4D6D6 D5D8D7 D7DAD9 D8DCDB DADFDD DDE1DF DFE4E2 E2E7E5 E5EAE8 E8EDEA EBEFED EDF1EF F0F4F2 F2F5F4 F4F7F6 F6F9F7 F8FAF9 F9FBFA FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=19: E9E5EB E6E2E8 E4E0E6 E1DDE3 DFDBE1 DDD9DE DBD7DC D9D6DA D7D5D9 D6D4D8 D5D4D7 D5D4D6 D5D5D6 D5D6D7 D5D8D7 D7D9D9 D8DBDA DADEDC DCE0DE DEE3E1 E0E5E3 E3E8E6 E6EBE8 E9EDEB EBF0EE EEF2F0 F0F4F2 F3F6F4 F5F8F6 F7F9F8 F8FAF9 FAFBFA FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=20: EAE7EC E8E4EA E5E2E7 E3DFE5 E1DDE2 DFDBE0 DDDADE DBD8DC D9D7DB D8D7D9 D7D7D9 D7D7D8 D7D7D8 D7D8D8 D7DAD9 D8DBDA DADDDC DBE0DE DDE2E0 DFE4E2 E2E7E4 E4E9E7 E7ECE9 EAEEEC ECF1EE EFF3F1 F1F5F3 F3F6F5 F5F8F6 F7F9F8 F8FAF9 FAFBFA FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=21: ECE8ED E9E6EB E7E4E9 E5E2E6 E3E0E4 E1DEE2 DFDCE0 DDDBDE DBDADD DAD9DC D9D9DB D9DADA D9DADA D9DBDB D9DCDB DADEDC DCE0DE DDE2DF DFE4E1 E1E6E4 E3E9E6 E6EBE8 E8EDEB EBEFED EDF2EF F0F4F1 F2F5F3 F4F7F5 F6F8F7 F7FAF8 F9FBFA FAFCFC FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=22: EDEAEF EBE8EC E9E6EA E7E4E8 E5E2E6 E3E0E4 E1DFE2 DFDEE1 DEDDDF DDDCDE DCDCDD DBDCDD DBDDDD DBDEDD DCDFDE DDE0DF DEE2E0 DFE4E2 E1E6E3 E3E8E5 E5EAE7 E7EDEA EAEFEC ECF1EE EEF3F0 F1F4F2 F3F6F4 F5F8F6 F6F9F7 F8FAF9 F9FBFA FBFDFC FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=23: EFECF0 EDEAEE EBE8EC E9E7EA E7E5E8 E5E3E6 E3E2E5 E2E1E3 E1E0E2 DFE0E1 DFDFE0 DEE0E0 DEE0E0 DEE1E0 DFE2E0 DFE3E1 E0E5E2 E2E6E4 E3E8E6 E5EAE7 E7ECE9 E9EEEB EBF0ED EEF2EF F0F4F1 F2F5F3 F4F7F5 F5F8F7 F7FAF8 F8FBF9 FAFBFA FCFDFC FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=24: F0EEF2 EFEDF0 EDEBEE EBE9EC E9E8EB E8E6E9 E6E5E7 E5E4E6 E3E3E5 E2E3E4 E2E3E3 E1E3E3 E1E3E3 E1E4E3 E1E5E3 E2E6E4 E3E7E5 E4E9E6 E6EBE8 E7ECEA E9EEEB EBF0ED EDF2EF EFF3F1 F1F5F3 F3F6F4 F5F8F6 F6F9F7 F8FAF9 F9FBFA FAFDFC FCFEFD FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=25: F2F0F3 F0EFF2 EFEDF0 EDECEE ECEAED EAE9EB E9E8EA E7E7E9 E6E6E7 E5E6E7 E5E6E6 E4E6E6 E4E6E5 E4E7E6 E4E8E6 E5E9E7 E6EAE8 E7EBE9 E8EDEA EAEEEC EBF0ED EDF2EF EFF3F1 F1F5F2 F3F6F4 F4F7F5 F6F9F7 F7FAF8 F9FBF9 FAFBFA FCFEFC FDFEFD FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=26: F4F2F5 F2F1F3 F1EFF2 EFEEF0 EEEDEF ECECEE EBEBEC EAEAEB E9E9EA E8E9E9 E7E9E9 E7E9E9 E7E9E8 E7EAE9 E7EBE9 E8ECEA E9EDEA EAEEEB EBEFED ECF0EE EEF2EF EFF3F1 F1F5F2 F2F6F4 F4F7F5 F5F8F7 F7F9F8 F8FAF9 F9FBFA FBFDFC FCFEFD FDFEFD FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=27: F5F4F6 F4F3F5 F3F2F4 F1F0F2 F0EFF1 EFEEF0 EEEEEF EDEDEE ECECED EBECEC EAECEC EAECEB EAECEB EAEDEB EAEDEC EBEEEC EBEFED ECF0EE EDF1EF EFF2F0 F0F4F1 F1F5F3 F3F6F4 F4F7F5 F5F8F6 F7F9F8 F8FAF9 F9FBFA FAFCFB FCFEFD FDFEFD FDFEFD FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=28: F7F6F7 F5F5F6 F4F4F5 F3F3F4 F2F2F3 F1F1F2 F0F0F1 EFF0F0 EEEFEF EEEFEF EDEFEE EDEFEE EDEFEE EDEFEE EDF0EE EDF1EF EEF1EF EFF2F0 F0F3F1 F1F4F2 F2F5F3 F3F6F4 F4F7F5 F5F8F7 F7F9F8 F8FAF9 F9FBFA FAFCFB FCFEFC FDFEFD FDFEFD FDFEFE FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=29: F8F8F9 F7F6F8 F6F5F7 F5F5F6 F4F4F5 F3F3F4 F2F3F3 F2F2F2 F1F2F2 F0F1F1 F0F1F1 F0F1F1 EFF2F1 EFF2F1 F0F2F1 F0F3F1 F0F4F2 F1F4F2 F2F5F3 F3F6F4 F4F7F5 F5F8F6 F6F9F7 F7F9F8 F8FAF9 F9FBFA FAFCFB FCFEFC FCFEFD FDFEFD FDFEFD FEFEFE FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=30: F9F9FA F9F8F9 F8F8F9 F7F6F7 F6F6F7 F5F5F6 F4F5F5 F4F4F5 F3F4F4 F3F4F4 F2F4F3 F2F4F3 F2F4F3 F2F4F3 F2F4F3 F2F5F3 F3F5F4 F3F6F4 F4F7F5 F5F7F6 F6F8F7 F6F9F7 F7FAF8 F8FAF9 F9FBFA FAFCFB FCFEFC FCFEFD FDFEFD FDFEFD FEFEFE FEFFFE FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
v=31: FAFAFB FAF9FA F9F9FA F9F8F9 F8F8F9 F7F7F7 F6F7F7 F6F6F6 F5F6F6 F5F6F6 F4F6F5 F4F6F5 F4F6F5 F4F6F5 F4F6F5 F4F7F5 F5F7F6 F5F8F6 F6F8F7 F6F9F7 F7F9F8 F8FBF9 F9FBFA FAFCFA FAFCFB FCFEFC FCFEFD FDFEFD FDFEFD FDFEFE FEFFFE FEFFFE FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF
```

### Camera 3 — ảnh render đầy đủ

**(a) ASCII art độ sáng trung bình RGB** (tối→sáng `@%#*+=-:. `, nền trắng = khoảng trắng; mỗi ký tự = 1 pixel, hàng = $v=0..31$, cột = $u=0..47$):

```
                            ....................
                           .....................
                          ......................
                         .......................
                         .......................
                        ........................
                        ........................
                       .........................
                       .............:::::.......
                       ............:::::::......
                       ............:::::::......
                       ............:::::::......
                       .............::::::......
                       ..............::::.......
                       .........................
                        ........................
                        ........................
                         .......................
                         .......................
                          ......................
                           .....................
                            ....................
                              ..................
                               ...............  
                                  .........     
                                                
                                                
                                                
                                                
                                                
                                                
                                                
```

**(b) Bản đồ $T_{\text{final}}$** (in $\lfloor9T_{\text{final}}\rfloor$; $9=$ nền hoàn toàn $T=1$, nhỏ hơn = bị Gaussian che nhiều):

```
999999999999888888888888888887777777777777777777
999999999999888888888888888877777777777777777777
999999999998888888888888888777777777777777777777
999999999998888888888888887777777776666666677777
999999999988888888888888887777777666666666667777
999999999988888888888888877777776666666666666777
999999999988888888888888877777766666666666666677
999999999988888888888888777777766666666666666667
999999999988888888888888777777666666666666666667
999999999988888888888888777777666666666666666667
999999999988888888888888777777666666666666666667
999999999988888888888888777777666666666666666667
999999999988888888888888777777666666666666666667
999999999988888888888888777777766666666666666667
999999999988888888888888777777766666666666666677
999999999988888888888888877777776666666666666777
999999999999999988888888877777777666666666667777
999999999999999988888888887777777766666666677777
999999999999999988888888887777777777776777777777
999999999999999988888888888777777777777777777777
999999999999999988888888888877777777777777777777
999999999999999988888888888888777777777777777778
999999999999999988888888888888877777777777777788
999999999999999988888888888888888877777777788888
999999999999999988888888888888888888888888888888
999999999999999988888888888888888888888888888888
999999999999999988888888888888888888888888888888
999999999999999998888888888888888888888888888888
999999999999999999888888888888888888888888888888
999999999999999999988888888888888888888888888888
999999999999999999999888888888888888888888888888
999999999999999999999998888888888888888888888888
```

**(c) Bản đồ `n_contrib`** (chỉ số 1-based, theo thứ tự depth, của Gaussian cuối cùng thực sự đóng góp; $0=$ nền):

```
000000000000111144444444444444444444444444444444
000000000000111144444444444444444444444444444444
000000000001111144444444444444444444444444444444
000000000001111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000011111144444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000044444444444444444444444444444444
000000000000000004444444444444444444444444444444
000000000000000000444444444444444444444444444443
000000000000000000044444444444444444444444444333
000000000000000000000444444444444444444444433333
000000000000000000000004444444444444444443333333
```

| Thống kê trên 1536 pixel (camera 3) | R | G | B |
|---|---|---|---|
| min | 0.828640 | 0.816597 | 0.823634 |
| max | 1.000000 | 1.000000 | 1.000000 |
| mean | 0.948881 | 0.949328 | 0.955826 |

| Đại lượng | min | max | mean |
|---|---|---|---|
| `final_Ts` | 0.690158 | 1.000000 | 0.914561 |
| `n_contrib` | 0 | 4 | 2.667969 |
| số Gaussian cộng thật (`contribs`) | 0 | 4 | 2.375651 |

Pixel nền (`n_contrib=0`, ngoài hợp mọi compact box): **440** / 1536 ($=28.65\%$). Pixel early-termination ($T(1-\alpha)<10^{-4}$): **0**.

**(d) Bảng đầy đủ giá trị 8-bit $(R,G,B)$ của toàn bộ 1536 pixel** (mỗi dòng là một hàng ảnh
$v=0..31$; trong dòng liệt kê tuần tự $u=0..47$ dạng `RRGGBB` hex — đây chính là toàn bộ nội dung
của $\hat I^{(cam3)}$, không lấy mẫu, tính bằng đúng công thức §4.2 cho từng pixel một):

```
v=00: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FDFDFF FCFDFF FBFCFF FAFAFD F9F9FD F8F8FC F7F7FB F6F6FB F4F4F9 F3F3F8 F1F1F7 F0F0F6 EEEEF4 EDECF3 EBEBF1 EAE9F0 E9E8EE E8E7ED E7E6EB E7E5EA E6E4E9 E6E4E8 E6E4E7 E6E4E7 E7E4E7 E7E5E7 E8E6E7 E9E7E8 EAE8E9 EBE9EA EDEBEB EEECED
v=01: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FCFCFF FBFCFF FAFAFD F8F9FC F7F8FC F6F6FB F4F5F9 F3F3F8 F1F1F7 EFF0F6 EEEEF4 ECECF3 EBEAF1 E9E8EF E8E7EE E7E5EC E6E4EA E5E3E9 E4E2E7 E4E1E6 E3E1E5 E3E1E5 E4E1E4 E4E1E4 E5E2E4 E6E3E5 E7E4E5 E8E5E6 E9E7E7 EBE9E9 ECEAEA
v=02: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FDFDFF FCFDFF FBFCFF FAFBFE F9F9FD F8F8FC F7F7FB F4F5FA F3F3F9 F1F2F8 EFF0F6 EEEEF5 ECECF3 EAEAF1 E8E8EF E7E6EE E5E4EC E4E3EA E3E1E8 E2E0E6 E1DFE5 E1DEE4 E1DEE2 E1DEE2 E1DEE1 E1DEE1 E2DFE1 E3E0E2 E4E1E3 E6E3E4 E7E5E5 E9E7E7 EAE9E8
v=03: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FCFCFF FBFCFF F9FAFD F8F9FC F7F7FC F6F6FB F3F4F9 F2F2F8 F0F0F7 EEEEF5 ECECF4 EAEAF2 E8E8F0 E6E6EE E5E4EC E3E2EA E2E0E8 E0DEE6 DFDDE4 DFDCE2 DEDBE1 DEDBE0 DEDBDF DEDBDE DFDBDE E0DCDF E1DEDF E2DFE0 E3E1E1 E5E3E3 E7E5E4 E9E7E6
v=04: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFEFF FDFDFF FCFDFF FBFCFF FAFBFE F9F9FD F8F8FC F6F7FB F4F5FA F2F3F9 F1F1F7 EFEFF6 ECEDF4 EAEBF2 E8E8F0 E6E6EE E4E4EC E3E2EA E1E0E8 DFDEE6 DEDCE3 DDDBE1 DCD9E0 DCD9DE DCD8DD DCD8DC DCD8DC DDD9DC DDDADC DFDBDD E0DDDD E1DFDF E3E1E0 E5E3E2 E7E5E4
v=05: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FCFDFF FBFCFF FAFAFE F8F9FD F7F8FC F6F6FB F3F4F9 F1F2F8 EFF0F7 EDEEF5 EBECF3 E9E9F1 E7E7EF E5E4ED E3E2EB E1E0E8 DFDDE6 DDDBE4 DCDAE1 DBD8DF DAD7DD DAD6DC D9D6DB DAD6DA DAD6D9 DBD7D9 DBD8D9 DDD9DA DEDBDB E0DDDD E1DFDE E3E1E0 E5E4E2
v=06: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FCFCFF FBFCFF FAFAFD F8F8FC F7F7FC F4F5FA F3F3F9 F1F1F8 EFEFF6 ECEDF4 EAEAF3 E8E8F0 E5E5EE E3E3EC E1E0E9 DFDEE7 DDDCE4 DCDAE2 DAD8DF D9D6DD D8D5DB D8D4DA D8D4D8 D8D4D8 D8D4D7 D9D5D7 DAD6D7 DBD7D8 DCD9D9 DEDBDB E0DDDC E2E0DF E4E2E1
v=07: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FBFCFF FAFBFE F9FAFD F8F8FC F6F7FC F4F5FA F2F3F9 F0F1F7 EEEEF6 EBECF4 E9EAF2 E7E7F0 E4E4ED E2E2EB E0DFE8 DEDCE6 DCDAE3 DAD8E0 D9D6DE D8D5DC D7D3DA D6D3D8 D6D2D7 D6D2D6 D6D3D5 D7D3D5 D8D4D5 D9D6D6 DBD8D7 DDDAD9 DFDCDB E1DFDD E3E2DF
v=08: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FBFCFF FAFBFE F9FAFD F8F8FC F5F6FB F4F4FA F2F2F8 EFF0F7 EDEEF5 EBEBF3 E8E9F1 E6E6EF E3E3ED E1E1EA DFDEE7 DCDBE4 DBD9E2 D9D7DF D7D5DD D6D4DA D5D2D8 D5D1D7 D5D1D5 D5D1D4 D5D1D4 D6D2D4 D7D3D4 D8D5D5 DAD7D6 DCD9D8 DEDCDA E0DEDC E2E1DE
v=09: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FDFEFF FDFDFF FDFDFF FCFDFF FBFCFF FAFAFE F9F9FD F7F8FC F5F6FB F3F4F9 F1F2F8 EFF0F7 EDEEF5 EAEBF3 E8E8F1 E5E6EF E3E3EC E0E0E9 DEDDE7 DCDBE4 DAD8E1 D8D6DE D7D4DC D5D3D9 D5D2D7 D4D1D6 D4D0D4 D4D0D3 D4D1D3 D5D2D3 D6D3D3 D7D5D4 D9D6D5 DBD9D7 DDDBD9 DFDEDB E1E1DE
v=10: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FDFEFF FDFDFF FDFDFF FCFDFF FBFCFF FAFAFE F9F9FD F7F8FC F5F6FA F3F4F9 F1F2F8 EFF0F7 ECEDF5 EAEBF3 E7E8F1 E5E5EE E2E3EC E0E0E9 DEDDE6 DBDAE3 D9D8E0 D8D6DE D6D4DB D5D3D9 D4D1D7 D4D1D5 D3D0D4 D3D0D3 D4D1D2 D5D2D2 D6D3D3 D7D4D3 D9D6D5 DAD9D6 DCDBD8 DFDEDB E1E1DD
v=11: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FDFEFF FDFDFF FDFDFF FCFDFF FBFCFF FAFAFE F9F9FD F7F8FC F5F6FA F3F4F9 F1F2F8 EFF0F6 ECEDF5 EAEBF3 E7E8F1 E5E5EE E2E3EC E0E0E9 DDDDE6 DBDBE3 D9D8E0 D8D6DE D6D4DB D5D3D9 D4D2D7 D4D1D5 D3D1D3 D3D1D3 D4D1D2 D5D2D2 D6D3D2 D7D5D3 D8D7D5 DAD9D6 DCDCD8 DFDEDB E1E1DD
v=12: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FDFEFF FDFDFF FDFDFF FCFDFF FBFCFF FAFAFE F9F9FD F7F7FB F5F6FA F3F4F9 F1F2F8 EFF0F7 EDEEF5 EAEBF3 E8E8F1 E5E6EE E3E3EC E0E0E9 DEDEE6 DCDBE3 DAD9E1 D8D7DE D6D5DB D5D4D9 D4D2D7 D4D2D5 D4D1D4 D4D1D3 D4D2D2 D5D3D2 D6D4D3 D7D6D4 D9D8D5 DBDAD7 DDDCD9 DFDFDB E1E2DE
v=13: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FBFCFF FAFBFE F9FAFD F7F8FB F5F6FB F3F4F9 F2F2F8 EFF0F7 EDEEF5 EBECF3 E8E9F1 E6E6EF E3E4EC E1E1E9 DFDFE7 DCDCE4 DADAE1 D9D8DE D7D6DC D6D5DA D5D4D8 D5D3D6 D4D3D5 D4D3D4 D5D3D3 D6D4D3 D7D5D4 D8D7D5 D9D9D6 DBDBD8 DDDEDA DFE0DC E2E3DF
v=14: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FBFCFF FAFBFE FAFAFD F8F8FC F6F6FB F4F5FA F2F3F8 F0F1F7 EEEFF5 EBECF3 E9EAF1 E7E7EF E4E5ED E2E2EA E0E0E7 DDDEE5 DCDBE2 DAD9DF D8D8DD D7D6DB D6D5D9 D6D5D7 D6D4D6 D6D5D5 D6D5D5 D7D6D5 D8D7D5 D9D9D6 DADBD7 DCDDD9 DEDFDB E0E1DD E3E4E0
v=15: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FDFEFF FDFDFF FCFDFF FCFCFF FBFBFE FAFAFD F8F9FC F6F7FB F4F5FA F3F3F9 F1F2F7 EEEFF6 ECEDF4 EAEBF2 E8E9F0 E5E6ED E3E4EB E1E1E8 DFDFE6 DDDDE3 DBDBE1 DADADE D9D8DC D8D7DA D7D7D9 D7D7D8 D7D7D7 D7D7D6 D8D8D6 D9D9D7 DADBD8 DCDDD9 DDDFDB DFE1DC E1E3DF E4E6E1
v=16: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FBFBFE FAFAFD F8F9FC F7F7FB F5F6FA F3F4F9 F1F2F8 EFF0F6 EDEEF4 EBECF3 E9EAF1 E7E8EE E5E5EC E2E3EA E1E1E7 DFDFE5 DDDDE2 DCDCE0 DBDBDE DADADC D9D9DB D9D9DA D9D9D9 D9DAD9 DADBD9 DBDCD9 DCDDDA DDDFDB DFE1DD E1E3DE E3E5E0 E5E7E3
v=17: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FBFCFF FAFBFE FAFAFD F8F8FC F6F7FA F4F5F9 F2F3F8 F0F1F7 EFF0F5 ECEDF3 EAEBF1 E8E9EF E6E7ED E4E5EB E2E3E9 E1E1E6 DFE0E4 DEDEE2 DDDDE0 DCDDDF DBDCDD DBDCDC DBDCDB DBDCDB DCDDDB DDDEDC DEE0DC DFE1DD E1E3DF E3E5E1 E5E7E2 E7E9E5
v=18: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FCFCFF FBFBFE FAFAFD F8F9FC F6F7FB F5F6FA F3F4F9 F2F3F7 F0F1F6 EEEFF4 ECEDF3 EAEBF1 E8E9EF E6E7ED E5E6EA E3E4E8 E2E2E6 E0E1E4 DFE0E3 DEDFE1 DEDFE0 DEDFDF DEDFDE DEDFDE DEE0DE DFE1DE E0E2DF E2E4E0 E3E5E1 E5E7E3 E6E9E5 E8EBE7
v=19: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FCFDFF FBFBFE FAFBFD F9F9FC F8F8FC F6F7FA F4F5F9 F3F4F8 F1F2F7 EFF0F5 EEEFF4 ECEDF2 EAEBF0 E8EAEE E7E8EC E5E6EA E4E5E9 E3E4E7 E2E3E5 E1E2E4 E1E2E3 E0E2E2 E0E2E1 E1E2E1 E1E3E1 E2E4E1 E3E5E2 E4E6E3 E5E8E4 E7EAE5 E8EBE7 EAEDE9
v=20: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FCFDFF FCFDFF FBFBFE F9FAFD F8F9FC F7F8FB F6F6FA F4F5F9 F3F4F8 F1F2F6 EFF0F5 EEEFF3 ECEDF2 EBECF0 E9EAEE E8E9EC E7E8EB E5E7E9 E5E6E8 E4E5E6 E3E5E5 E3E5E5 E3E5E4 E3E5E4 E4E6E4 E4E7E4 E5E8E5 E6E9E6 E8EAE7 E9ECE8 EBEEE9 ECEFEB
v=21: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FDFDFF FCFDFF FBFCFE FBFBFD F9FAFC F8F9FC F7F7FA F5F6F9 F4F5F8 F3F4F7 F1F2F6 F0F1F5 EEEFF3 EDEEF2 ECEDF0 EAECEF E9EBED E8EAEC E7E9EA E7E8E9 E6E8E8 E6E8E7 E6E8E7 E6E8E7 E7E9E7 E7EAE7 E8EBE8 E9ECE8 EAEDE9 EBEEEA EDF0EC EEF1ED
v=22: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FDFDFF FDFDFF FCFDFF FBFBFE FAFAFD F9FAFC F8F9FB F7F7FA F5F6F9 F4F5F8 F3F4F7 F2F3F6 F0F1F5 EFF0F3 EEEFF2 EDEEF1 ECEDEF EBECEE EAECED EAEBEC E9EBEB E9EBEA E9EBEA E9EBEA E9ECEA EAEDEA EBEDEA EBEEEB ECEFEC EEF0ED EFF2EE F0F3EF
v=23: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FDFEFF FDFDFF FCFDFF FCFCFE FBFBFE FAFAFC F9FAFC F8F9FB F7F8FB F6F6F9 F4F5F8 F3F4F7 F2F3F6 F1F2F5 F0F1F4 EFF0F2 EEF0F1 EDEFF0 EDEEEF ECEEEE ECEEEE ECEEED ECEEED ECEEED ECEFED EDEFED EDF0ED EEF1EE EFF2EF F0F3EF F1F4F0 F2F5F2
v=24: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FDFEFF FDFEFF FDFDFF FCFDFF FCFCFE FBFBFD FAFAFC F9FAFC F8F9FB F7F8FA F6F7F9 F5F6F8 F4F5F7 F3F4F6 F2F3F5 F1F3F4 F1F2F3 F0F1F2 EFF1F2 EFF1F1 EFF0F0 EEF0F0 EEF0EF EEF1EF EFF1EF EFF2EF F0F2F0 F0F3F0 F1F4F1 F2F5F2 F3F5F3 F4F6F4
v=25: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FDFEFF FDFEFF FDFDFF FDFDFF FCFCFE FBFBFD FAFBFC F9FAFC F9F9FB F8F9FB F7F7F9 F6F7F9 F5F6F8 F4F5F7 F3F5F6 F3F4F5 F2F4F4 F2F3F4 F1F3F3 F1F3F3 F1F3F2 F1F3F2 F1F3F2 F1F3F2 F1F4F2 F2F4F2 F3F5F3 F3F6F3 F4F6F4 F5F7F5 F6F8F5
v=26: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FEFEFF FDFEFF FDFDFF FDFDFF FBFCFD FBFBFD FAFBFC FAFAFC F9FAFB F8F9FB F8F9FA F7F8F9 F6F7F8 F5F6F8 F5F6F7 F4F6F6 F4F5F6 F4F5F5 F3F5F5 F3F5F4 F3F5F4 F3F5F4 F3F5F4 F4F6F4 F4F6F4 F5F7F5 F5F7F5 F6F8F6 F6F8F6 F7F9F7
v=27: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FEFEFF FDFEFF FDFEFF FDFDFF FCFCFD FBFCFD FBFBFC FAFBFC FAFAFB F9FAFB F8F9FA F8F9FA F7F8F9 F7F8F8 F6F7F8 F6F7F7 F6F7F7 F5F7F6 F5F7F6 F5F7F6 F5F7F6 F5F7F6 F6F7F6 F6F8F6 F6F8F7 F7F9F7 F7F9F7 F8FAF8 F9FBF9
v=28: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FEFEFF FDFEFF FDFEFF FDFDFF FCFCFD FBFCFD FBFBFC FBFBFC FAFBFC FAFAFB F9FAFB F9FAFA F9F9FA F8F9FA F8F9F9 F8F9F9 F7F8F8 F7F8F8 F7F8F8 F7F9F8 F7F9F8 F7F9F8 F8FAF8 F8FAF9 F9FAF9 F9FBF9 FAFBFA FBFCFA
v=29: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FEFEFF FEFEFF FDFEFF FDFEFF FCFDFE FCFCFD FBFCFD FBFCFC FBFBFC FAFBFC FAFBFB FAFBFB FAFAFB F9FAFA F9FAFA F9FAFA F9FAFA F9FAFA F9FAFA F9FAFA F9FBFA F9FBFA FAFBFA FAFBFA FBFCFA FBFCFB FCFDFB
v=30: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FEFEFF FEFEFF FEFEFF FDFDFE FCFDFE FCFCFD FCFCFD FBFCFC FBFCFC FBFCFC FBFBFC FAFBFB FAFBFB FAFBFB FAFBFB FAFBFB FAFBFB FAFBFB FAFCFB FBFCFB FCFDFB FCFDFB FCFDFB FCFDFC FCFDFC
v=31: FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FFFFFF FEFEFF FEFEFF FEFEFF FEFEFF FEFEFF FDFEFE FDFDFE FCFDFD FCFCFD FCFCFD FCFCFC FBFCFC FBFCFC FBFCFC FBFCFC FBFCFC FBFCFC FBFCFC FCFDFC FCFDFC FCFDFC FCFDFC FDFDFC FDFEFD FDFEFD
```

---

## 4.5 — Pixel nền: định nghĩa và phạm vi hợp compact box

Một pixel $x$ là **nền thuần** khi và chỉ khi không có Gaussian nào trong $\mathcal G_{T(x)}$ vượt qua
cả hai cửa lọc đầu ($\text{power}\le0$ và $\alpha_n(x)\ge1/255$) — tương đương `n_contrib[x]=0`. Có hai
nguyên nhân dẫn tới điều này:

1. **Tile rỗng**: $\mathcal G_T=\varnothing$ (không Gaussian nào có tile này trong $\mathcal K_i$) — toàn bộ
   $16\times16=256$ pixel của tile đó là nền ngay từ đầu, không tốn phép toán nào (ví dụ tile 2, 5 của
   camera 2; tile 3 của camera 3).
2. **Tile không rỗng nhưng pixel nằm ngoài mọi ellipse hiệu dụng**: $\mathcal G_T\ne\varnothing$ nhưng tại
   pixel cụ thể, mọi $n\in\mathcal G_T$ đều bị cửa 1 hoặc cửa 2 loại (ví dụ pixel $(1,1)$ camera 1 ở §4.3 —
   tile 0 có đủ 4 Gaussian nhưng góc ảnh quá xa tâm mọi splat).

Ở cả hai trường hợp, `out_color = T_final * bg = 1*(1,1,1) = (1,1,1)` theo đúng công thức alpha-blending
với tổng rỗng — **không phải** một nhánh code riêng, mà là hệ quả tự nhiên của vòng lặp for chạy 0 lần
(trường hợp 1) hoặc $K$ lần nhưng không có lần `cộng` nào (trường hợp 2). Bảng dưới liệt kê số pixel
nền — phần bù của nó (pixel không nền) chính là **hợp các compact box đã lọc ellipse** chiếu lên lưới
pixel, và mọi pixel trong hợp đó đã được tính đầy đủ trong bản đồ `n_contrib`/ảnh ASCII ở §4.4 (tính
từng pixel một, không lấy mẫu):

| Camera | pixel nền | pixel không nền (∈ hợp compact box) | tỉ lệ nền |
|---|---|---|---|
| 1 | 42 | 1494 | 2.73% |
| 2 | 512 | 1024 | 33.33% |
| 3 | 440 | 1096 | 28.65% |

Camera 1 có tỉ lệ nền thấp nhất (splat phủ gần kín ảnh, như đã nhận xét ở Phần 3 §3.7: $R_{\text{tile}}=1$
cho camera 1 vì lưới quá nhỏ so với splat); camera 2, 3 có nền nhiều hơn vì tâm ảnh lệch khỏi cụm 4
Gaussian nên một nửa lưới tile ($T=2,5$ ở cam 2; $T=3$ ở cam 3, gần hết) không được splat nào chạm.

---

## 4.6 — Ghi chú đối chiếu với code (`diff-gaussian-rasterization_fastgs`)

Toàn bộ trình tự phép toán trong §4.1–§4.5 chép **đúng thứ tự lệnh** từ ba file nguồn CUDA thật của
submodule `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/`:

| Đại lượng / bước | File : dòng | Ghi chú |
|---|---|---|
| Khoá 64-bit `key = (tile<<32) | bit_cast(depth)` | `auxiliary.h:278-280` | `key <<= 32; key |= *((uint32_t*)&depth);` — đúng `bit_cast` float→uint32 dùng trong §4.1 |
| `duplicateWithKeys` (ghi khoá vào buffer chưa sort) | `rasterizer_impl.cu:70-99` | vòng lặp `off = offsets[idx-1]..offsets[idx]`, sinh đúng bảng "buffer trước sort" §4.1 |
| `InclusiveSum` cho `tiles_touched` → `point_offsets` | `rasterizer_impl.cu:329` | cho $\text{offset}_i=\sum_{j<i}K_j$ dùng trong §4.1 |
| Radix sort trên khoá 64-bit (giới hạn bit = $32+\lceil\log_2(\text{tile})\rceil$) | `rasterizer_impl.cu:356-366` | `cub::DeviceRadixSort::SortPairs`, tương đương `np.argsort(kind='stable')` trên `uint64` dùng trong script |
| `identifyTileRanges` (dựng `ranges[T]=[start,end)`) | `rasterizer_impl.cu:105-131` | so `key>>32` giữa idx và idx−1, đúng thuật toán trong §4.1 |
| `renderCUDA` — hàm rasterize chính, một thread/pixel, batch 256 Gaussian/lần nạp shared memory | `forward.cu:275-276` (khai báo), toàn thân `forward.cu:276-430` | mỗi CUDA block xử lý đúng 1 tile $16\times16=256$ thread |
| Vòng lặp dừng khi mọi pixel trong block đã `done` | `forward.cu:344-345` (`__syncthreads_count(done)==BLOCK_SIZE`) | tối ưu, không ảnh hưởng kết quả toán học |
| $\text{power}=-\tfrac12(A\,d_x^2+C\,d_y^2)-B\,d_xd_y$ | `forward.cu:380` | biến `con_o.x,con_o.y,con_o.z` $=(A,B,C)$ |
| Cửa 1: `power>0 → continue` | `forward.cu:381-382` | khớp cửa 1 §4.2 |
| $\alpha=\min(0.99,\ \alpha_i\cdot\exp(\text{power}))$ | `forward.cu:388` | khớp công thức $\alpha_n(x)$ §4.2 |
| Cửa 2: `alpha<1/255 → continue` | `forward.cu:390-391` | khớp cửa 2 §4.2 |
| `test_T = T*(1-alpha)` rồi cửa 3: `test_T<1e-4 → done=true` (thoát vòng, không cộng) | `forward.cu:393-397` | khớp cửa 3 và early-termination §4.2 |
| Cộng dồn màu: `C[ch] += color[ch]*alpha*T` | `forward.cu:402` | khớp $c_n\alpha_nT_n$ trong công thức blending |
| Cập nhật `T *= (1-alpha)` (ngầm qua `test_T`), `last_contributor = contributor` | `forward.cu:398-416` | dữ liệu này đi vào `n_contrib[x]` |
| Ghi `final_T[pix]`, `n_contrib[pix]` | `forward.cu:420-426` | dùng cho bản đồ §4.4 và backward chương 11 |
| `BlockReduce` lấy `max_contrib[tile]` | `forward.cu:434-439` | dùng cho `perTileBucketCount` |
| `perTileBucketCount` — số bucket backward mỗi tile $=\lceil|\mathcal G_T|/32\rceil$ | `rasterizer_impl.cu:133-…` | không dùng trực tiếp ở Phần 4 (thuộc chương 11) nhưng phụ thuộc `ranges[T]` đã dựng ở đây |

Ba camera trong lời giải này chỉ khác nhau ở **đầu vào** `ranges`/`point_list`/`con_o`/`means2D` do Phần 3
tính ra — **cùng một** kernel `renderCUDA`/`duplicateWithKeys`/`identifyTileRanges` chạy giống hệt cho cả
3 lần gọi (không có nhánh code theo camera trong rasterizer, camera chỉ ảnh hưởng qua phép chiếu ở Phần 3).

---

## 4.7 — Phụ lục: script numpy kiểm chứng toàn bộ chương

Script dưới đây (chỉ `numpy`, không `torch`, đúng tinh thần `scripts/ch0N_test.py` của chương 15/§16.4.4)
tái tạo **từ đầu** — từ 4 điểm SfM và 3 pose camera — toàn bộ pipeline chiếu (Phần 3) và rasterize
(Phần 4 này): chiếu tâm, $\Sigma'$, conic $(A,B,C)$, compact box, lọc tile theo ellipse (`processTiles`
rút gọn), sinh khoá 64-bit, sort, `identifyTileRanges`, rồi render alpha-blending cho cả 3 camera.
Mọi con số trong Phần 4 — bảng, ASCII art, bản đồ `final_T`/`n_contrib` — là **output trực tiếp** của
chính script này (không có số nào gõ tay), khớp lại đúng các bảng đã công bố ở chương 9 (`DOCS/BOOK/`
`09-differentiable-tile-rasterizer.md`, mục 9.2) cho camera 1: `img` min/max mỗi kênh
$(0.8247,1),(0.8128,1),(0.8250,1)$, `final_T` $\in[0.6858,1]$ trung bình $0.8979$, 42 pixel nền — khớp
chính xác đến 4 chữ số thập phân.

```python
import numpy as np
import struct

W, H = 48, 32
fx = fy = 40.0
tan_fovx, tan_fovy = 0.6, 0.4
TILE = 16
GRID_X, GRID_Y = 3, 2
LOWPASS = 0.3
MULT = 0.5
ALPHA_THRESH = 1.0 / 255.0
T_THRESH = 1e-4

mu = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.3, 0.5],
    [-0.4, -0.2, 1.0],
    [0.3, -0.5, 0.2],
])
colors = np.array([
    [0.8, 0.2, 0.2],
    [0.2, 0.7, 0.3],
    [0.1, 0.3, 0.9],
    [0.5, 0.5, 0.5],
])
alpha = np.array([0.1, 0.1, 0.1, 0.1])

# s_i^2 = trung bình khoảng cách bình phương tới 3 láng giềng (N=4 -> 3 điểm còn lại)
s2 = np.array([np.mean([np.sum((mu[i]-mu[j])**2) for j in range(4) if j != i])
               for i in range(4)])

cams = {1: np.array([0.0, 0.0, -4.0]),
        2: np.array([1.5, 0.0, -4.0]),
        3: np.array([-1.5, 0.5, -4.0])}

def ndc2pix(px, py):
    return ((px+1)*W-1)/2, ((py+1)*H-1)/2

def min_quad_form(A, B, C, dx0, dx1, dy0, dy1):
    """min tren hinh chu nhat cua dang toan phuong A dx^2+2B dx dy+C dy^2"""
    if dx0 <= 0 <= dx1 and dy0 <= 0 <= dy1:
        return 0.0
    best = np.inf
    for dxv in (dx0, dx1):
        dystar = min(max(-B*dxv/C if C != 0 else 0.0, dy0), dy1)
        best = min(best, A*dxv**2 + 2*B*dxv*dystar + C*dystar**2)
    for dyv in (dy0, dy1):
        dxstar = min(max(-B*dyv/A if A != 0 else 0.0, dx0), dx1)
        best = min(best, A*dxstar**2 + 2*B*dxstar*dyv + C*dyv**2)
    return best

results = {}
for cam_id, c in cams.items():
    t = mu - c[None, :]
    depth = t[:, 2].copy()
    tx_tz, ty_tz = t[:, 0]/t[:, 2], t[:, 1]/t[:, 2]
    mu2 = np.stack(ndc2pix(tx_tz/tan_fovx, ty_tz/tan_fovy), axis=1)

    lim_x, lim_y = 1.3*tan_fovx, 1.3*tan_fovy
    tx_hat = np.clip(tx_tz, -lim_x, lim_x) * depth
    ty_hat = np.clip(ty_tz, -lim_y, lim_y) * depth
    f_tz = 40.0 / depth
    J01 = -fx*tx_hat/depth**2
    J11 = -fy*ty_hat/depth**2

    Sigma11 = s2*(f_tz**2 + J01**2) + LOWPASS
    Sigma12 = s2*(J01*J11)
    Sigma22 = s2*(f_tz**2 + J11**2) + LOWPASS

    det = Sigma11*Sigma22 - Sigma12**2
    A, B, C = Sigma22/det, -Sigma12/det, Sigma11/det

    tth = MULT*2*np.log(255*alpha)
    half_x, half_y = np.sqrt(tth*Sigma11), np.sqrt(tth*Sigma22)
    bbox_min = mu2 - np.stack([half_x, half_y], axis=1)
    bbox_max = mu2 + np.stack([half_x, half_y], axis=1)
    rect_min = np.clip(np.floor(bbox_min/TILE), [0,0], [GRID_X, GRID_Y]).astype(int)
    rect_max = np.clip(np.floor(bbox_max/TILE)+1, [0,0], [GRID_X, GRID_Y]).astype(int)

    tile_lists, K = [], np.zeros(4, dtype=int)
    for i in range(4):
        tiles = []
        for ty in range(rect_min[i,1], rect_max[i,1]):
            for tx in range(rect_min[i,0], rect_max[i,0]):
                dx0, dx1 = tx*TILE-mu2[i,0], tx*TILE+TILE-mu2[i,0]
                dy0, dy1 = ty*TILE-mu2[i,1], ty*TILE+TILE-mu2[i,1]
                if min_quad_form(A[i],B[i],C[i],dx0,dx1,dy0,dy1) <= tth[i]:
                    tiles.append(ty*GRID_X+tx)
        tile_lists.append(sorted(tiles)); K[i] = len(tiles)

    results[cam_id] = dict(depth=depth, mu2=mu2, A=A, B=B, C=C,
                           tile_lists=tile_lists, K=K)

def f32_to_u32(x):
    return struct.unpack('>I', struct.pack('>f', np.float32(x)))[0]

# --- khoa 64-bit, sort, identifyTileRanges (per camera) ---
for cam_id, d in results.items():
    rows = []
    for i in range(4):
        for T in d['tile_lists'][i]:
            u32 = f32_to_u32(d['depth'][i])
            rows.append((T, i, (T << 32) | u32))
    srows = sorted(rows, key=lambda r: r[2])   # radix sort thay bang stable argsort
    ranges = {}
    for idx, (T, i, key) in enumerate(srows):
        if T not in ranges: ranges[T] = [idx, idx+1]
        else: ranges[T][1] = idx+1
    d['tile_members'] = {T: sorted([i for i in range(4) if T in d['tile_lists'][i]],
                                    key=lambda i: d['depth'][i]) for T in range(6)}

# --- renderCUDA (mo phong): mot pixel = mot vong lap Gaussian theo depth tang dan ---
def render(cam_id):
    d = results[cam_id]
    A, B, C, mu2 = d['A'], d['B'], d['C'], d['mu2']
    img = np.ones((H, W, 3))
    final_T = np.ones((H, W))
    n_contrib = np.zeros((H, W), dtype=int)
    for v in range(H):
        for u in range(W):
            T = (v//TILE)*GRID_X + (u//TILE)
            Tt, Cacc, last = 1.0, np.zeros(3), 0
            for rank, i in enumerate(d['tile_members'][T], start=1):
                dx, dy = mu2[i,0]-u, mu2[i,1]-v
                power = -0.5*(A[i]*dx*dx + C[i]*dy*dy) - B[i]*dx*dy
                if power > 0: continue
                a = min(0.99, alpha[i]*np.exp(power))
                if a < ALPHA_THRESH: continue
                test_T = Tt*(1-a)
                if test_T < T_THRESH: break
                Cacc += colors[i]*a*Tt
                Tt = test_T
                last = rank
            img[v,u] = Cacc + Tt*np.ones(3)
            final_T[v,u], n_contrib[v,u] = Tt, last
    return img, final_T, n_contrib

for cam_id in (1, 2, 3):
    img, fT, nc = render(cam_id)
    print(cam_id, img.reshape(-1,3).min(0), img.reshape(-1,3).max(0),
          fT.mean(), int((nc == 0).sum()))
```

Chạy script trên cho đúng $3\times(48,32,3)$ ảnh render và các con số thống kê đã liệt kê ở §4.4;
`assert` so khớp với bảng camera 1 của chương 9 (`min` $R=0.8247$, `max`$=1$, nền $=42$ pixel) là bước
kiểm chứng chéo bắt buộc trước khi tin các bảng camera 2, 3 (không có đối chiếu độc lập nào khác).

---

## 4.8 — Bài tập

**Bài tập 4.1.** Từ bảng §4.0, giải thích vì sao $\mathcal K_i$ của cả 4 Gaussian ở camera 2 **trùng nhau
hệt** ($\{0,1,3,4\}$) trong khi ở camera 3, Gaussian 3 có $\mathcal K_3=\{0,1,2,4,5\}$ khác 3 Gaussian
còn lại ($\{1,2,4,5\}$). Gợi ý: so sánh $B_3$ (dấu và độ lớn) với $B_1,B_2,B_4$ ở camera 3, và liên hệ
với lập luận "lọc ellipse chỉ có tác dụng khi $\rho\gg1$" của Phần 3 §3.7.

**Bài tập 4.2.** Tại camera 2, tile $T=2$ và $T=5$ hoàn toàn rỗng ($\mathcal G_T=\varnothing$). Tính số
cặp (tile, Gaussian) $P^{(cam2)}=\sum_iK_i$ trực tiếp từ bảng §4.0 và so với $P^{(cam1)}=24$: giảm bao
nhiêu phần trăm? Giải thích ý nghĩa cho chi phí `duplicateWithKeys`/radix sort (chương 9 §4.1–4.3) nếu
scene có nhiều Gaussian hơn nhưng cùng tỉ lệ tile-rỗng.

**Bài tập 4.3.** Chọn lại pixel "corner" của tile 1, camera 3 ở §4.3 — $x=(17,1)$ — có 3/4 Gaussian bị
cửa $\alpha<1/255$ loại và chỉ Gaussian 3 (đỏ... à lam) đóng góp. Tính tay lại $\alpha_n(x)$ cho $n=1$
(Gaussian 1) bằng công thức conic, rồi kiểm chứng ngưỡng: $0.1\cdot G_1<1/255\iff G_1<1/25.5\iff
\text{power}_1<\ln(1/25.5)=-3.239$. So $\text{power}_1=-3.523$ ở bảng — đúng dưới ngưỡng, khớp cửa lọc.

**Bài tập 4.4.** Với ba Gaussian tại một pixel theo thứ tự depth tăng dần $(c_n,\alpha_n)$ lần lượt là
$(\text{đỏ},0.5),(\text{lục},0.4),(\text{lam},0.6)$ (ví dụ kinh điển chương 9 §4.5), nếu đảo thứ tự
duyệt thành lam→lục→đỏ, tính lại $T_n$ và $C(x)$ từng bước rồi so với $(0.62,0.32,0.30)$ đúng thứ tự.
Áp dụng cùng phép đảo thứ tự cho pixel tile 4 "center" của **camera 1** ở §4.3
($\mathcal G_T=\{1,4,2,3\}\to$ đảo thành $\{3,2,4,1\}$) và tính $C(x)$ mới — so với
$(0.8858,0.8897,0.8888)$ đã tính ở bảng, định lượng sai số tuyệt đối do đảo thứ tự.

**Bài tập 4.5.** Tính số pixel nền lý thuyết tối thiểu cho camera 3 nếu compact box của cả 4 Gaussian
chỉ phủ đúng 4 tile $\{1,2,4,5\}$ (bỏ qua trường hợp đặc biệt của $G_3$ chạm thêm tile 0): $4\times256=$
bao nhiêu pixel không-nền tối đa? So với số đo thực tế ở §4.4/§4.5 ($1536-440=1096$) và giải thích chênh
lệch bằng vai trò của tile 0 (chỉ có $G_3$, phần lớn vẫn là nền vì $\mu'_3$ ở xa tile 0).

**Bài tập 4.6.** Với `mult`$=0.5$ và $\alpha=0.1$ cố định, hai hằng số $1/255$ và $10^{-4}$ đóng vai trò
khác nhau (bài tập gốc chương 9, §4.7). Tính $T$ nhỏ nhất lý thuyết đạt được tại một pixel nếu cả 4
Gaussian của camera 1 cùng đóng góp $\alpha_n(x)=\alpha_i=0.1$ tối đa ($x=\mu'_i$, $G_n=1$, giả định lý
tưởng không xảy ra thật). So với ngưỡng $10^{-4}$ và với `final_T` nhỏ nhất đo được thực tế ở bảng §4.4
(camera 1: $0.6858$) — vì sao chênh lệch lớn đến vậy?

**Bài tập 4.7.** Đối chiếu bảng `key` chưa-sort của camera 3 ở §4.1: dòng đầu tiên có tile$=1$ (không
phải tile$=0$) vì $\mathcal K_1^{(cam3)}=\{1,2,4,5\}$ không chứa tile 0. Giải thích `identifyTileRanges`
(`rasterizer_impl.cu:105-131`) xử lý thế nào khi tile 0 **không xuất hiện** trong toàn bộ buffer đã sort
— `ranges[0]` nhận giá trị gì, và vì sao đó vẫn là giá trị đúng để vòng lặp render bỏ qua tile 0.

---

## Đầu ra chuyển cho Phần 5

| Đại lượng | Giá trị / nguồn |
|---|---|
| $\hat I^{(1)},\hat I^{(2)},\hat I^{(3)}$ | ảnh render $48\times32\times3$, float64 (script) / float32 (kernel), RGB $\in[0,1]$, §4.4 |
| `final_Ts`, `n_contrib` mỗi camera | bản đồ §4.4, dùng cho backward chương 11 (Phần 6) |
| $P^{(cam)}$ | 24 (cam 1), 16 (cam 2), 17 (cam 3) — §4.0 |

> Đầu ra chuyển cho Phần 5 (Chương 10 — Loss & Metrics): ảnh render $\hat I^{(v)}$ cho cả 3 camera.

---

[← Phần 3](03-projection.md) | [Mục lục lời giải](00-muc-luc-loi-giai.md) | [Phần 5 →](05-loss-metrics.md)
