# Vì sao fastgs-lite nhanh hơn 3DGS — từ toán học 3DGS tới mô hình chi phí

> Tài liệu **tự chứa**: bắt đầu từ toán học 3D Gaussian Splatting, dựng ra **mô hình chi phí một vòng lặp huấn luyện**, rồi chỉ đúng chỗ mà fastgs-lite can thiệp vào từng số hạng của mô hình đó. Không giả định người đọc đã biết gì ngoài đại số tuyến tính cơ bản.
> Mọi công thức, hằng số và hành vi đều được đối chiếu với **code trong repo này** — không lấy con số nào từ paper upstream. Chỗ nào là suy luận chưa đo thì ghi rõ.
> Tên hàm/tham số trỏ thẳng tới mã nguồn: `utils/fast_utils.py`, `scene/gaussian_model.py`, `gaussian_renderer/__init__.py`, `train.py`, `submodules/diff-gaussian-rasterization_fastgs/`.
> Hai script chạy được, **không cần GPU**:
> - [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) — mô phỏng NumPy từng cơ chế (tên hàm ghi ngay đầu mỗi phần).
> - [`demos/fastgs_cost_model.py`](../demos/fastgs_cost_model.py) — **phép đo** ba tỉ số chi phí ở Phần VI.

**Mục lục**

| Phần | Nội dung |
|---|---|
| [I](#phần-i--nền-tảng-toán-học-3d-gaussian-splatting) | Nền tảng toán học 3DGS: blending, chiếu covariance, loss, và vì sao phải chiếu 2D |
| [II](#phần-ii--mô-hình-chi-phí-một-vòng-lặp-huấn-luyện) | Mô hình chi phí một vòng lặp — nơi mọi lập luận tăng tốc phải quy về |
| [III](#phần-iii--đòn-bẩy-1-giảm-n-điểm-số-nhất-quán-đa-góc-nhìn) | Đòn bẩy 1 — giảm $N$: điểm số nhất quán đa góc nhìn, densify AND, hai đường prune |
| [IV](#phần-iv--đòn-bẩy-2-giảm-k-compact-box) | Đòn bẩy 2 — giảm $K$: compact box, và vì sao hộp vuông $3\sigma$ của 3DGS lãng phí |
| [V](#phần-v--đòn-bẩy-3-giảm-nhịp-cập-nhật-tham-số) | Đòn bẩy 3 — giảm nhịp Adam, và phép tách learning rate SH |
| [VI](#phần-vi--phép-đo-fastgs-lite-vs-3dgs) | **Phép đo** fastgs-lite vs 3DGS: ba tỉ số, kết quả mô hình, giao thức A/B thật |
| [VII](#phần-vii--ba-điều-mã-nguồn-nói-mà-readme-upstream-không-nói) | Ba điều mã nguồn nói mà README upstream không nói |

---
---

# Phần I — Nền tảng toán học 3D Gaussian Splatting

Phần này là nền tảng để đọc Phần II trở đi. Nếu bạn đã nắm chắc 3DGS, có thể nhảy thẳng tới [Phần II](#phần-ii--mô-hình-chi-phí-một-vòng-lặp-huấn-luyện) — nhưng §1.5 (từ toán sang kernel) thì nên đọc, vì toàn bộ mô hình chi phí dựng trên đó.

## 1.1 — Alpha-blending dựa trên điểm (Point-based Blending)

### Công thức (3) — Pha trộn màu tại pixel

$$C=\sum_{i\in N}c_i\,\alpha_i\prod_{j=1}^{i-1}(1-\alpha_j)$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $C$ | Màu cuối cùng của pixel |
| $N$ | Số Gaussian chồng lên pixel đó, đã sắp xếp theo độ sâu (gần → xa) |
| $c_i$ | Màu của Gaussian thứ $i$ |
| $\alpha_i$ | Độ mờ (opacity) của Gaussian $i$ tại **đúng vị trí pixel đang xét** |
| $T_i=\prod_{j=1}^{i-1}(1-\alpha_j)$ | Độ truyền qua tích luỹ — phần ánh sáng còn lại sau khi bị các Gaussian phía trước che bớt |

**Bản chất:** GS biểu diễn cảnh bằng một tập **hữu hạn** các Gaussian 3D rời rạc. Khi render, các Gaussian phủ lên một pixel được sắp xếp theo độ sâu rồi blend tuần tự từ gần đến xa — điểm gần nhất đóng góp trọn vẹn $\alpha_1$, điểm sau bị chiết khấu bởi phần đã được "hấp thụ".

Công thức kế thừa từ dòng **point-based neural rendering** (Kopanas et al. 2021, 2022), được 3D Gaussian Splatting (Kerbl et al. 2023) dùng làm cơ chế blending chính.

**Ví dụ số:** 3 Gaussian phủ 1 pixel, sắp theo thứ tự gần→xa:

| i | màu $c_i$ | $\alpha_i$ | $T_i=\prod(1-\alpha_j)$ | đóng góp $T_i\alpha_ic_i$ |
|---|---|---|---|---|
| 1 | đỏ (1,0,0) | 0.5 | 1.0 | (0.5, 0, 0) |
| 2 | xanh lá (0,1,0) | 0.4 | 0.5 | (0, 0.2, 0) |
| 3 | xanh dương (0,0,1) | 0.6 | 0.3 | (0, 0, 0.18) |

$$C = (0.5,\ 0.2,\ 0.18)$$

Nếu nền trắng $(1,1,1)$, độ truyền qua còn lại $T_4=0.5\times0.6\times0.4=0.12$:

$$C_{\text{final}} = (0.5,0.2,0.18) + 0.12\times(1,1,1) = (0.62,\ 0.32,\ 0.30)$$

Gaussian gần nhất đóng góp nhiều nhất dù không có $\alpha$ cao nhất — vì chưa bị điểm nào che ($T_1=1$). Đây là lý do **depth sorting** cực kỳ quan trọng: đổi thứ tự sẽ ra màu khác hẳn dù $\alpha_i, c_i$ không đổi.

> **Đây cũng là mầm của toàn bộ vấn đề tốc độ.** Blending là tuần tự có thứ tự, nên trước khi blend phải **sort**. Sort cái gì, và bao nhiêu phần tử — chính là số hạng đắt nhất trong mô hình chi phí ở Phần II.

---

## 1.2 — Biểu diễn và chiếu 3D Gaussian

### Công thức (4) — Hàm phân phối 3D Gaussian cơ bản

$$G(x)=e^{-\frac{1}{2}x^T\Sigma^{-1}x}$$

- $x$: vị trí tương đối so với tâm Gaussian ($x = \mathbf{p} - \boldsymbol{\mu}$)
- $\Sigma$: ma trận hiệp phương sai, quyết định hình dạng, kích thước, hướng của "quả bóng" Gaussian
- $G(x)$: giá trị mật độ tại $x$, cực đại $=1$ tại tâm, giảm theo khoảng cách Mahalanobis

**Vai trò trong blending:** $\alpha_i$ ở công thức (3) không phải hằng số cho cả Gaussian — nó bằng opacity học được $\alpha'_i$ nhân với giá trị Gaussian tại đúng vị trí pixel:

$$\alpha_i = \alpha'_i \cdot G(x)$$

Càng xa tâm, $\alpha_i$ giảm **theo hàm mũ** — tạo biên mờ tự nhiên thay vì cạnh cứng. Tính chất "giảm theo hàm mũ" này là thứ Phần IV khai thác: sau một khoảng cách nhất định, đóng góp nhỏ hơn cả bước lượng tử của kênh 8-bit, nên **không cần rasterize nữa**.

> **Lưu ý về ký hiệu:** công thức (4) viết $\Sigma^{-1}$ như định nghĩa Gaussian **tổng quát**, dùng chung cho 3D (công thức 6) lẫn 2D. Khi thực sự tính $\alpha$ tại một pixel, $\Sigma$ ở đây phải hiểu ngầm là $\Sigma'$ (đã chiếu 2D, công thức 5), **không phải** $\Sigma$ 3D gốc. Cách viết rõ ràng hơn:
> $$\alpha_i(\mathbf{x}) = \alpha'_i \exp\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}'_i)^T{\Sigma'_i}^{-1}(\mathbf{x}-\boldsymbol{\mu}'_i)\right)$$

### Công thức (5) — Chiếu covariance từ 3D xuống 2D (Splatting)

$$\Sigma'=JW\Sigma W^T J^T$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $\Sigma$ | Hiệp phương sai 3D gốc (world space) |
| $W$ | Ma trận biến đổi view (world → camera space) |
| $J$ | Jacobian của phép chiếu xạ ảnh — tuyến tính hoá cục bộ phép chiếu phối cảnh phi tuyến |
| $\Sigma'$ | Hiệp phương sai 2D sau khi chiếu lên mặt phẳng ảnh (screen space) |

**Vai trò:** biến một ellipsoid Gaussian 3D thành một **vệt elip 2D** trên màn hình — "splat". Một Gaussian 3D chiếu xuống vẫn cho ra một Gaussian 2D, nên chỉ cần tính $\Sigma'$ **một lần mỗi Gaussian mỗi khung hình**, rồi dùng công thức (4) phiên bản 2D cho từng pixel.

Trong code, kernel không giữ $\Sigma'$ mà giữ **conic** — nghịch đảo của nó:

$$M=\Sigma'^{-1}=\begin{pmatrix}A&B\\B&C\end{pmatrix}$$

`forward.cu:232` đóng gói `conic = {cov.z, -cov.y, cov.x} / det`, rồi nhét cùng opacity vào một `float4` tên `con_o` (`.x=A, .y=B, .z=C, .w=o`). Toàn bộ Phần IV làm việc trên $M$ và $o$ này.

### Công thức (6) — Phân rã ma trận hiệp phương sai 3D

$$\Sigma=RSS^TR^T$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $R$ | Ma trận xoay 3×3 (tham số hoá bằng quaternion khi tối ưu) |
| $S$ | Ma trận tỷ lệ (scale), dạng đường chéo |

**Tại sao cần phân rã?** $\Sigma$ phải luôn bán xác định dương. Tối ưu trực tiếp các phần tử của $\Sigma$ bằng gradient descent rất dễ vi phạm ràng buộc này. Tham số hoá qua $R$ (trực giao — luôn hợp lệ) và $S$ (đường chéo dương) khiến $RSS^TR^T$ **luôn tự động** hợp lệ, bất kể cập nhật thế nào.

**Tham số học được của mỗi Gaussian** — đếm cho đúng, vì Phần II cần con số này:

| Nhóm | Số chiều |
|---|---|
| Vị trí $\boldsymbol{\mu}$ | 3 |
| Rotation quaternion | 4 |
| Scale $S$ | 3 |
| Opacity $\alpha'$ | 1 |
| SH bậc 0 (`features_dc`) | 3 |
| SH bậc 1–3 (`features_rest`) | 45 |
| **Tổng** | **59** |

Adam giữ 2 moment cho mỗi tham số ⇒ **~177 float mỗi Gaussian** chỉ riêng trạng thái optimizer. Đây là lý do số Gaussian $N$ vừa quyết định thời gian, vừa quyết định VRAM.

---

## 1.3 — Hàm mất mát

### Công thức (7) — Loss tổng quát

$$\mathcal{L}=(1-\lambda)\mathcal{L}_1+\lambda\,\mathcal{L}_{\text{D-SSIM}}$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathcal{L}_1$ | Sai số tuyệt đối trung bình giữa ảnh render và ground-truth |
| $\mathcal{L}_{\text{D-SSIM}}$ | Loss cấu trúc (Structural Dissimilarity) |
| $\lambda$ | Trọng số cân bằng ($\approx 0.2$ trong paper gốc; repo này dùng `--lambda_dssim 0.25`) |

Loss này lan truyền gradient ngược về mọi tham số của từng Gaussian qua chuỗi (5)→(4)→(3), đồng thời điều khiển **adaptive density control** (thêm/xoá/tách Gaussian).

Điểm cần nhớ cho Phần II: $\mathcal{L}$ tính trên **toàn ảnh $H\times W$** — chi phí của nó **không phụ thuộc $N$**. Đây là số hạng "cố định" đặt trần Amdahl cho mọi nỗ lực tăng tốc.

---

## 1.4 — Vì sao phải chiếu Gaussian xuống 2D?

### Hai cách khả dĩ để tính $\alpha$ tại một pixel

**Cách 1 — giữ nguyên 3D, ray-tracing (như NeRF):** bắn tia từ camera qua pixel → tìm giao với từng Gaussian → tính $x$ = giao điểm trừ tâm → đưa vào (4). Chi phí lớn: mỗi pixel phải duyệt/tính giao trong không gian 3D không có cấu trúc lưới.

**Cách 2 — chiếu xuống 2D rồi rasterize (GS):** tính trước $\Sigma'$ cho mỗi Gaussian → biết ngay bounding box pixel bị phủ → với mỗi pixel, $x=(u,v)-\boldsymbol{\mu}'$ chỉ là phép trừ 2D → đưa vào (4) với $\Sigma'^{-1}$.

### Vì sao không tính thẳng bằng $\Sigma^{-1}$ (3D)?

Khoảng cách Mahalanobis $x^T\Sigma^{-1}x$ trong 3D đo độ gần/xa trong không gian thật, nhưng cái cần để tô pixel là độ gần/xa trong **không gian ảnh** — hai đại lượng **không tương đương tuyến tính** vì phép chiếu phối cảnh phi tuyến:

- **Foreshortening theo góc nhìn:** Gaussian hình cầu nhìn nghiêng trông dẹt thành elip — phụ thuộc góc nhìn, điều $\Sigma$ 3D cố định không tự phản ánh.
- **Phối cảnh theo khoảng cách:** Gaussian gần trông to, xa trông nhỏ dù $\Sigma$ không đổi — quan hệ phi tuyến, nên cần $J$ để tuyến tính hoá cục bộ quanh tâm.

### Tóm tắt

| Không chiếu (giữ 3D) | Có chiếu xuống 2D |
|---|---|
| Ray-cast dò giao điểm cho từng pixel | Biết ngay bounding box pixel bị phủ |
| $x$ = kết quả giao tia–ellipsoid (tốn kém) | $x$ = phép trừ toạ độ 2D đơn giản |
| Không tận dụng rasterization của GPU | Tận dụng trực tiếp pipeline rasterize GPU |
| Chậm (giây–phút/khung hình) | Real-time (hàng trăm FPS) |

**Đóng góp cốt lõi của 3DGS** là biến bài toán render từ **ray-tracing** (đắt) thành **rasterization** (rẻ), mà vẫn giữ tính chất mượt của volume rendering nhờ biên Gaussian mờ dần.

---

## 1.5 — Từ toán sang kernel: rasterization theo tile

Đây là mắt xích nối Phần I với Phần II. Sau khi có $\Sigma'$, kernel **không** duyệt pixel trực tiếp — nó chia ảnh thành lưới **tile 16×16** rồi làm bốn bước:

| Bước | Kernel | Đơn vị chi phí |
|---|---|---|
| **(a) Preprocess** | `preprocessCUDA` (`forward.cu`) | mỗi **Gaussian**: chiếu $\Sigma\to\Sigma'$, nghịch đảo ra conic, đánh giá 48 hệ số SH ra màu RGB |
| **(b) Duplicate** | `duplicateToTilesTouched` (`auxiliary.h:318`) | mỗi **cặp (tile, Gaussian)**: sinh một khoá 64-bit `(tile_id, depth)` |
| **(c) Sort** | radix sort trên toàn bộ khoá | mỗi **cặp**, $O(P)$ với radix sort |
| **(d) Blend** | `renderCUDA` | mỗi **cặp**, nhân với số pixel trong tile mà splat còn đóng góp |

Gọi:

$$N=\text{số Gaussian},\qquad K=\text{số tile trung bình một splat chạm},\qquad P=N\cdot K$$

$P$ là **số cặp (tile, Gaussian)** — đại lượng trung tâm. Ba trong bốn bước trên tỉ lệ với $P$, không phải với $N$. Đây là điều quan trọng nhất cần rút ra từ Phần I:

> **Chi phí render không tỉ lệ với số Gaussian. Nó tỉ lệ với số Gaussian nhân số tile mỗi Gaussian.**
>
> Nên có **hai** cách làm nó rẻ đi: giảm $N$ (Phần III) hoặc giảm $K$ (Phần IV). fastgs-lite làm cả hai.

Backward chạy đúng cấu trúc đó theo chiều ngược (`backward.cu`), với chi phí cùng bậc — thường 2–3× forward vì phải tích luỹ gradient cho 59 tham số qua đường atomics.

---
---

# Phần II — Mô hình chi phí một vòng lặp huấn luyện

> Mô phỏng chạy được: [`demos/fastgs_cost_model.py`](../demos/fastgs_cost_model.py).

## 2.1 — Bốn số hạng

Một vòng lặp huấn luyện 3DGS làm đúng năm việc: render một ảnh, tính loss, backward, cập nhật tham số, và (định kỳ) densify. Gộp lại:

$$\boxed{\;T_{\text{iter}} \;=\; \underbrace{a\,N}_{\text{preprocess}} \;+\; \underbrace{b\,N K}_{\text{dup + sort + blend + backward}} \;+\; \underbrace{c\,N\cdot\mathbb{1}[\text{có Adam step}]}_{\text{optimizer}} \;+\; \underbrace{F}_{\text{loss, SSIM, IO}}\;}$$

| Số hạng | Tỉ lệ với | Vì sao |
|---|---|---|
| $a\,N$ | $N$ | Mỗi Gaussian: một phép chiếu $JW\Sigma W^TJ^T$, một phép nghịch đảo 2×2, một lần đánh giá SH bậc 3 (48 hệ số) |
| $b\,NK$ | $N\cdot K$ | Sinh khoá, radix sort, blend, và backward của blend — tất cả duyệt trên tập cặp (tile, Gaussian) |
| $c\,N$ | $N$ | Adam trên $59N$ tham số + $2\times 59N$ moment. Chỉ tính khi vòng đó thực sự `step()` |
| $F$ | — | $\mathcal{L}_1$ + SSIM trên ảnh $H\times W$, đọc ảnh GT, overhead Python/CUDA launch. **Không phụ thuộc $N$** |

Ba đại lượng có thể can thiệp: $N$, $K$, và tần suất $\mathbb{1}[\text{Adam}]$. $F$ thì không.

## 2.2 — Ba tỉ số, và vì sao chúng nhân nhau

Đặt tỉ số fastgs-lite so với 3DGS:

$$R_{\text{gauss}}=\frac{N_{\text{fast}}}{N_{\text{3dgs}}},\qquad
R_{\text{tile}}=\frac{K_{\text{fast}}}{K_{\text{3dgs}}},\qquad
R_{\text{adam}}=\frac{\#\text{step}_{\text{fast}}}{\#\text{step}_{\text{3dgs}}}$$

Số hạng rasterization — số hạng nặng nhất — co theo **tích**:

$$b\,N_{\text{fast}}K_{\text{fast}} = b\,N_{\text{3dgs}}K_{\text{3dgs}}\cdot R_{\text{gauss}}R_{\text{tile}}$$

Đây là điểm mấu chốt về mặt toán: hai đòn bẩy **nhân** nhau chứ không **cộng**. Giảm 3× số Gaussian và 3.4× số tile mỗi splat cho ra **~10×** trên số hạng rasterization, dù mỗi cái riêng lẻ chỉ khiêm tốn.

## 2.3 — Trần Amdahl

Cho $N\to 0$ và $K\to 0$, $T_{\text{iter}}\to F$. Nên tăng tốc tối đa có thể đạt được là:

$$S_{\max}=\frac{T_{\text{iter}}^{\text{3dgs}}}{F}=\frac{1}{f}\quad\text{với } f=\frac{F}{T_{\text{iter}}^{\text{3dgs}}}$$

Nếu loss + SSIM + IO chiếm 15% một vòng 3DGS thì **không cơ chế nào đưa tăng tốc vượt 6.7×**, dù có xoá sạch mọi Gaussian. Đây là lý do các con số tăng tốc thực tế của họ 3DGS nhanh đều nằm trong khoảng 2–5× chứ không phải 50×, và là lý do phải đo $f$ trước khi đặt kỳ vọng.

fastgs-lite còn **thêm** một chi phí mà 3DGS không có: `compute_gaussian_score_fastgs` render thêm $10\times2=20$ ảnh mỗi lần densify. Với `densification_interval=500` và `densify_until_iter=15000`: $30\times20=600$ render phụ trên tổng 30.000 vòng $\Rightarrow$ **+2% chi phí rasterization**. Nhỏ, nhưng phải trừ đi cho trung thực — và nó **tăng tuyến tính** nếu hạ `densification_interval` (xem §7.3).

---
---

# Phần III — Đòn bẩy 1: giảm $N$ (điểm số nhất quán đa góc nhìn)

3DGS gốc phình số Gaussian vì Adaptive Density Control nhân bản ở **mọi** vùng có gradient lớn, kể cả vùng đã hội tụ — tới hàng triệu điểm. fastgs-lite thay tiêu chí đó bằng câu hỏi khác hẳn:

> Thay vì *"Gaussian này có gradient lớn không?"* → *"Nhiều camera có cùng đồng ý rằng chỗ này đang sai không?"*

## 3.1 — Bản đồ lỗi nhị phân trên mỗi góc nhìn

> Mô phỏng: `normalize_minmax()`, `error_mask()`, `accum_metric_counts()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "1.").

Hàm `compute_gaussian_score_fastgs` (`utils/fast_utils.py:45`) chạy trước mỗi lần densify, lấy mẫu **10 camera ngẫu nhiên** (`sampling_cameras`, `fast_utils.py:10`), với từng camera làm 3 bước.

**Bước 1 — chuẩn hoá bản đồ lỗi L1 về $[0,1]$:**

$$e(u,v)=\frac{1}{3}\sum_{k\in\{R,G,B\}}\bigl|\,I_{\text{render}}(u,v,k)-I_{\text{gt}}(u,v,k)\,\bigr|$$

$$\hat{e}(u,v)=\frac{e(u,v)-\min_{u,v} e}{\max_{u,v} e-\min_{u,v} e}$$

Chuẩn hoá min–max khiến ngưỡng ở bước sau **không phụ thuộc độ sáng tuyệt đối của cảnh** — cảnh tối và cảnh sáng dùng chung một `loss_thresh`.

**Bước 2 — nhị phân hoá:**

$$m(u,v)=\begin{cases}1 & \hat{e}(u,v) > \tau_{\text{loss}}\\[4pt] 0 & \text{ngược lại}\end{cases}
\qquad \tau_{\text{loss}}=\texttt{loss\_thresh}\ (\text{mặc định }0.1)$$

`m` là `metric_map` (`fast_utils.py:82`): mặt nạ đánh dấu **những pixel mà mô hình hiện tại đang tái tạo tệ nhất**. `loss_thresh` thấp → nhiều pixel bị đánh dấu → giữ lại / sinh thêm nhiều Gaussian hơn.

**Bước 3 — đổ ngược mặt nạ pixel về từng Gaussian:** rasterizer được gọi lại với `get_flag=True` và `metric_map=m` (`fast_utils.py:84`). Mỗi Gaussian $i$ nhận về:

$$\text{counts}^{(v)}_i=\#\bigl\{\text{pixel }(u,v)\ \text{mà Gaussian }i\ \text{đóng góp và}\ m(u,v)=1\bigr\}$$

trả về ở `render_pkg["accum_metric_counts"]`. Trực giác: **Gaussian $i$ "chịu trách nhiệm" cho bao nhiêu pixel lỗi** ở góc nhìn $v$.

> **Vì sao lượt render thứ hai này rẻ?** Nó dùng lại đúng cấu trúc tile của lượt đầu — cùng $N$, cùng $K$, không backward. Nên chi phí bằng ~1 forward, và tổng overhead chỉ là con số +2% ở §2.3.

## 3.2 — Gộp nhiều góc nhìn thành hai điểm số

> Mô phỏng: `merge_views()` (mục "2.").

### Công thức (1) — Importance score (điều khiển việc *sinh thêm*)

$$\text{Importance}_i=\left\lfloor \frac{1}{V}\sum_{v=1}^{V}\text{counts}^{(v)}_i \right\rfloor
\qquad V=10$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $V$ | Số camera lấy mẫu (`num_cams = 10`) |
| $\text{counts}^{(v)}_i$ | Số pixel lỗi mà Gaussian $i$ phủ ở góc nhìn $v$ |
| $\lfloor\cdot\rfloor$ | Làm tròn xuống (`rounding_mode='floor'`, `fast_utils.py:102`) |

Đây là **số pixel-lỗi trung bình mỗi góc nhìn**. Vì lấy trung bình rồi floor, một Gaussian chỉ bị "một camera duy nhất" tố sai sẽ có điểm gần 0 — **phải sai một cách nhất quán trên nhiều góc nhìn** mới được điểm cao. Đó là ý nghĩa của "multi-view consistent".

**Phép floor không vô hại.** Với $V=10$, mọi Gaussian có tổng counts $< 10$ đều nhận điểm $0$. Nó là một bộ lọc nhiễu miễn phí: những đóng góp lẻ tẻ một-hai pixel bị triệt sạch trước khi so với ngưỡng.

### Công thức (2) — Pruning score (điều khiển việc *cắt bỏ*)

$$s_i=\sum_{v=1}^{V}\mathcal{L}^{(v)}_{\text{photo}}\cdot\text{counts}^{(v)}_i$$

$$\text{Pruning}_i=\frac{s_i-\min_j s_j}{\max_j s_j-\min_j s_j}\in[0,1]$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathcal{L}^{(v)}_{\text{photo}}$ | Loss ảnh của cả khung hình $v$: $(1-\lambda)\mathcal{L}_1+\lambda(1-\text{SSIM})$ với $\lambda=0.2$ **cứng trong code** (`compute_photometric_loss`, `fast_utils.py:30`) |
| $s_i$ | Điểm thô: số pixel-lỗi của Gaussian $i$, **nhân trọng số** bằng độ tệ toàn cục của góc nhìn đó |
| $\text{Pruning}_i$ | Điểm chuẩn hoá $[0,1]$; càng gần 1 = càng "vô dụng / gây hại" |

Khác biệt then chốt so với Importance: pruning score **nhân thêm $\mathcal{L}^{(v)}_{\text{photo}}$**. Một Gaussian phủ nhiều pixel lỗi trong một khung hình vốn đã render rất tệ sẽ bị phạt nặng hơn.

> **Bẫy:** $\lambda=0.2$ ở đây là hằng số viết thẳng trong `fast_utils.py:30`, **không** đọc `--lambda_dssim`. Preset của repo dùng `--lambda_dssim 0.25` cho loss train, nên loss dùng để *tối ưu* và loss dùng để *chấm điểm pruning* đang lệch nhau. Đổi `--lambda_dssim` không kéo theo chỗ này.

### Ví dụ số

3 Gaussian, $V=3$ camera:

| | cam 1 ($\mathcal{L}_{\text{photo}}=0.20$) | cam 2 ($0.05$) | cam 3 ($0.10$) |
|---|---|---|---|
| $G_A$ | 8 | 6 | 7 |
| $G_B$ | 12 | 0 | 0 |
| $G_C$ | 1 | 0 | 1 |

**Importance** (trung bình rồi floor):
- $G_A=\lfloor(8+6+7)/3\rfloor=\lfloor7.0\rfloor=7$ → **vượt ngưỡng 5** → ứng viên densify.
- $G_B=\lfloor12/3\rfloor=4$ → dưới ngưỡng: sai nhiều nhưng **chỉ ở 1 góc nhìn** → không densify (tránh nhồi Gaussian cho artefact cục bộ).
- $G_C=\lfloor2/3\rfloor=0$.

**Pruning** (điểm thô $s_i$):
- $s_A=0.20\cdot8+0.05\cdot6+0.10\cdot7=2.60$
- $s_B=0.20\cdot12=2.40$
- $s_C=0.20\cdot1+0.10\cdot1=0.30$

Chuẩn hoá: $\text{Pruning}_A=1.0$, $\text{Pruning}_B=\dfrac{2.40-0.30}{2.60-0.30}=0.913$, $\text{Pruning}_C=0.0$.

Ở lần "final prune" ($\tau=0.9$, §3.4), **cả $G_A$ và $G_B$ đều bị xoá** — chúng liên tục nằm dưới các pixel sai. $G_C$ giữ lại.

> **Nghịch lý biểu kiến:** $G_A$ vừa là ứng viên **densify** (Importance = 7) vừa là ứng viên **prune** (Pruning ≈ 1). Không mâu thuẫn: hai điểm số dùng ở **hai giai đoạn khác nhau** — densify chạy ở vòng < 15k để *thử thêm chi tiết*, final-prune chạy ở vòng > 15k để *dọn những gì không giúp được*.

## 3.3 — Densification có điều kiện kép

> Mô phỏng: `densify_masks()` và `final_prune_fastgs()` (mục "3.").

Hàm `densify_and_prune_fastgs` (`scene/gaussian_model.py:468`). Một Gaussian chỉ được nhân bản khi **thoả đồng thời hai điều kiện độc lập**.

### Điều kiện gradient — chọn *ở đâu* cần thêm chi tiết

$$\text{clone}_i:\ \lVert \bar{g}_i\rVert \ge \tau_{\text{grad}}\ \ \wedge\ \ \max(\text{scale}_i)\le \delta\cdot\text{extent}$$

$$\text{split}_i:\ \lVert \bar{g}^{\text{abs}}_i\rVert \ge \tau_{\text{grad}}^{\text{abs}}\ \ \wedge\ \ \max(\text{scale}_i) > \delta\cdot\text{extent}$$

| Ký hiệu | Ý nghĩa | Tham số CLI (mặc định) |
|---|---|---|
| $\bar{g}_i$ | Gradient vị trí 2D tích luỹ / số lần quan sát (`xyz_gradient_accum / denom`) | `--grad_thresh` (0.0002) |
| $\bar{g}^{\text{abs}}_i$ | Gradient **trị tuyệt đối** tích luỹ (kiểu Abs-GS) | `--grad_abs_thresh` (0.0012) |
| $\delta\cdot\text{extent}$ | Ngưỡng kích thước: nhỏ thì **clone** (thiếu mật độ), to thì **split** (thiếu độ mịn) | `--dense` (0.001) |

**Vì sao cần gradient trị tuyệt đối?** `add_densification_stats` (`:527-530`) tích luỹ hai đại lượng khác nhau từ cùng một tensor:

```python
self.xyz_gradient_accum[f]     += norm(viewspace_point_tensor.grad[f, :2])   # cộng có dấu
self.xyz_gradient_accum_abs[f] += norm(viewspace_point_tensor.grad[f, 2:])   # cộng trị tuyệt đối
```

Một Gaussian phủ lên biên vật thể nhận gradient **đẩy sang trái ở khung hình này, sang phải ở khung hình kia**. Tổng có dấu triệt tiêu về ~0, nên 3DGS gốc **không thấy** nó cần split — dù đó chính là chỗ cần thêm chi tiết nhất. Bản trị tuyệt đối không triệt tiêu:

$$\Bigl\lVert\sum_v g^{(v)}\Bigr\rVert \;\le\; \sum_v \bigl\lVert g^{(v)}\bigr\rVert$$

Dấu bằng chỉ xảy ra khi mọi $g^{(v)}$ cùng hướng. Khoảng cách giữa hai vế **chính là** tín hiệu "gradient dao động đổi dấu" mà Abs-GS khai thác — và cũng là lý do hai ngưỡng khác nhau ($0.0002$ vs $0.0012$, gấp 6 lần).

### Điều kiện nhất quán đa góc nhìn — lọc *cái nào* thực sự đáng thêm

$$\text{metric\_mask}_i = \bigl[\ \text{Importance}_i > 5\ \bigr]$$

```python
# scene/gaussian_model.py:494
metric_mask = importance_score > 5
self.densify_and_clone_fastgs(metric_mask, all_clones)   # AND theo từng phần tử
self.densify_and_split_fastgs(metric_mask, all_splits)
```

**Đây là điểm khác biệt cốt lõi với 3DGS gốc.** 3DGS densify mọi Gaussian có gradient lớn. fastgs-lite thêm phép **AND**. Hệ quả:

- Gaussian ở vùng đã hội tụ (gradient còn dư nhưng render đã đúng) → Importance thấp → **không nhân bản** → $N$ không phình.
- Gaussian ở artefact chỉ thấy từ 1–2 góc (floater phản chiếu chẳng hạn) → Importance thấp → **không được củng cố**.

Về mặt mô hình chi phí: phép AND tác động vào **tốc độ tăng trưởng** của $N$, mà $N$ lại xuất hiện trong cả ba số hạng biến thiên của $T_{\text{iter}}$. Vì densify chạy lặp lại (~30 lần), tác động là **luỹ thừa theo số lần densify**, không phải tuyến tính:

$$N_{\text{cuối}} \approx N_0\bigl[(1+r_{\text{spawn}})(1-r_{\text{prune}})\bigr]^{n_{\text{densify}}}$$

Một khác biệt nhỏ ở $r_{\text{spawn}}$ (ví dụ 0.10 → 0.06) khuếch đại thành khác biệt lớn ở $N_{\text{cuối}}$ sau 30 lần — đó là điều `mean_gaussians()` trong cost model minh hoạ.

## 3.4 — Hai đường xoá Gaussian (không phải ba tầng độc lập)

`densify_and_prune_fastgs` có một khối lọc trông như hai tầng, nhưng **chỉ gọi `prune_points` đúng một lần**:

```python
# scene/gaussian_model.py:499-518 (rút gọn)
prune_mask = (self.get_opacity < min_opacity).squeeze()          # tập ỨNG VIÊN
if max_screen_size:
    prune_mask |= (self.max_radii2D > max_screen_size)
    prune_mask |= (self.get_scaling.max(dim=1).values > 0.1 * extent)

scores = 1 - pruning_score
remove_budget = int(0.5 * torch.sum(prune_mask))                  # trần: một nửa ứng viên
if remove_budget:
    padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
    sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
    selected_pts_mask[sampled_indices] = True
    self.prune_points(torch.logical_and(prune_mask, selected_pts_mask))   # lệnh xoá DUY NHẤT
```

| | Vai trò | Chi tiết |
|---|---|---|
| `prune_mask` | **Không xoá gì cả** — chỉ là tập ứng viên | `opacity < 0.005`; cộng thêm `max_radii2D > 20 px` và `scale > 0.1·extent` **chỉ khi** `max_screen_size` khác `None`, tức chỉ từ vòng > `opacity_reset_interval` (3000) trở đi — `train.py:133`, `pipeline/trainer.py:126` |
| Lấy mẫu multinomial | Quyết định **ai** trong tập ứng viên thật sự bị xoá | Trần cứng bằng một nửa số ứng viên; trọng số $1/(10^{-6}+1-\text{Pruning}_i)$ nên Gaussian có `pruning_score` gần 1 gần như chắc chắn bị chọn. Gaussian vừa sinh ở lần densify này có trọng số 0 (nằm ngoài `scores.shape[0]`) nên miễn nhiễm |
| **`final_prune_fastgs`** | Đường xoá thứ hai, tách rời hẳn | `opacity < 0.1` **hoặc** `Pruning_i > 0.9`, xoá thẳng không lấy mẫu — `gaussian_model.py:533` |

Chú ý trọng số lấy mẫu $w_i=1/(10^{-6}+1-\text{Pruning}_i)$. Nó **phân kỳ** khi $\text{Pruning}_i\to 1$: Gaussian tệ nhất có $w=10^{6}$, trong khi Gaussian $\text{Pruning}=0.5$ chỉ có $w=2$. Multinomial không replacement với chênh lệch trọng số cỡ $10^6$ về thực chất là **sắp xếp giảm dần theo pruning score rồi lấy nửa đầu** — ngẫu nhiên chỉ còn tác dụng ở phần đuôi.

Hai điều dễ hiểu sai:

- **Mỗi lần densify, tối đa một nửa số Gaussian "đáng xoá" bị xoá.** Nếu `remove_budget == 0` (dưới 2 ứng viên) thì không xoá gì. `if remove_budget:` là chốt chặn chia-0, **không** phải công tắc bật/tắt — comment trong code (`"The budget is not necessary for our method"`, dòng 509) cho thấy nhóm tác giả coi cơ chế trần này là phần thừa kế, nhưng nó **vẫn chạy thật** mỗi lần densify.
- **`final_prune_fastgs` chỉ chạy trong khoảng `15_000 < iteration < 30_000`, mỗi 3000 vòng** (`train.py:153`, `pipeline/trainer.py:139`). Ở đúng vòng 30000 nó **không** chạy. Ngân sách dưới 15k — kể cả mặc định `7000` của `pipeline/config.py` — **không bao giờ chạm tới đường xoá này**, nên model giao ra là model chưa tỉa cuối.

---
---

# Phần IV — Đòn bẩy 2: giảm $K$ (compact box)

> Mô phỏng: `tiles_touched()` và `tiles_touched_3dgs()` trong [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) (mục "4."); phép **đo** đầy đủ trong [`demos/fastgs_cost_model.py`](../demos/fastgs_cost_model.py).
> Mã nguồn: `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/auxiliary.h:318-345`, comment ghi rõ *"This is built upon Speedy-Splat"*.

## 4.1 — 3DGS gốc dùng hộp **vuông**, và đó là chỗ lãng phí lớn nhất

Điều này hay bị nói sai. 3DGS gốc **không** dùng hình chữ nhật $3\sigma$ theo từng trục. Nó dùng một hộp **vuông**, cạnh quyết định bởi trục **dài nhất**:

```c
// forward.cu:238-240 — vẫn còn nguyên trong fork này (chỉ dùng cho radii[])
float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));    // <- max, không phải per-axis
```

rồi `getRect(p, my_radius, ...)` (`auxiliary.h:50`) dùng **cùng một `max_radius`** cho cả hai chiều. Nên:

$$K_{\text{3dgs}}=\left(\frac{2\cdot 3\sqrt{\lambda_{\max}}}{16}+1\right)^{\!2}$$

Hệ quả toán học: với một splat **dẹt** — tỉ lệ trục $\rho=\sigma_{\max}/\sigma_{\min}$ — hộp vuông có diện tích $\propto \sigma_{\max}^2$, trong khi ellipse thật chỉ có diện tích $\pi\sigma_{\max}\sigma_{\min}=\pi\sigma_{\max}^2/\rho$. Tỉ lệ lãng phí:

$$\frac{\text{diện tích hộp vuông}}{\text{diện tích ellipse}} = \frac{36\sigma_{\max}^2}{\pi\sigma_{\max}^2/\rho}=\frac{36}{\pi}\rho \;\approx\; 11.5\,\rho$$

**Và Gaussian sau khi train thì rất dẹt.** Adaptive density control ép chúng thành những đĩa mỏng áp vào bề mặt — đó là cách 3DGS biểu diễn mặt phẳng. Nên $\rho\gg1$ là trạng thái *bình thường*, không phải ngoại lệ. Đây là lý do lượng lãng phí lớn hơn nhiều so với trực giác "chữ nhật bao ellipse thì thừa 4 góc".

## 4.2 — Hộp được dựng từ conic và opacity, không từ $3\sigma$

fastgs-lite không co hộp $3\sigma$ lại. Nó **thay hẳn cách dựng hộp**, kế thừa **SnugBox của Speedy-Splat**, rồi thêm hệ số `mult` vào **ngưỡng level-set**.

Vùng cần rasterize là mặt cắt nơi splat còn đóng góp quá 1/255 mức xám — tức trên ngưỡng lượng tử của kênh 8-bit:

$$o\cdot\exp\!\Big(-\tfrac12\Delta^\top M\,\Delta\Big)\ \ge\ \frac{1}{255}
\qquad\Longleftrightarrow\qquad
\Delta^\top M\,\Delta\ \le\ t,\quad t=2\ln(255\,o)$$

```c
// auxiliary.h:336-338
float t = 2.0f * log(con_o.w * 255.0f);   // level-set o·G = 1/255
t = mult * t;                             // beta trong Compact Box
```

| Ký hiệu | Ý nghĩa | Nguồn |
|---|---|---|
| $M=\Sigma'^{-1}$ | conic 2D — nghịch đảo covariance đã chiếu (công thức (5), §1.2) | `con_o.x/.y/.z` |
| $o$ | opacity của Gaussian, sau sigmoid | `con_o.w` |
| $t$ | **ngưỡng bình phương khoảng cách Mahalanobis** của ellipse cần bao | `auxiliary.h:337` |
| `mult` | hệ số nhân vào $t$ — đây là toàn bộ tác dụng của `--mult` | `auxiliary.h:338` |
| $\text{disc}=B^2-AC$ | phải $<0$, nếu không ellipse suy biến và hàm trả 0 tile | `auxiliary.h:327-333` |

Hộp bao trục-song-song của ellipse $\Delta^\top M\Delta\le t$ có nửa cạnh **theo từng trục**:

$$\text{half-extent}_x=\sqrt{t\,\Sigma'_{11}},
\qquad
\text{half-extent}_y=\sqrt{t\,\Sigma'_{22}}$$

So sánh trực tiếp với §4.1: hộp này **tôn trọng dị hướng** ($\Sigma'_{11}\ne\Sigma'_{22}$) còn hộp vuông của 3DGS thì không. Toàn bộ hệ số $\rho$ trong công thức lãng phí biến mất ngay tại đây.

> **Chi tiết đọc code dễ nhầm:** `x_term`/`y_term` ở dòng 340-343 **không phải** nửa cạnh — đó là toạ độ điểm tiếp tuyến $h=\sqrt{-B^2t/(\text{disc}\cdot C)}$, dùng làm đầu vào cho `computeEllipseIntersection` để lấy biên chính xác.

## 4.3 — Ba hệ quả của việc `mult` nhân vào $t$ chứ không vào cạnh

**(a) Cạnh co theo $\sqrt{\texttt{mult}}$, diện tích co theo $\texttt{mult}$.**

| `mult` | $t$ (khi $o=1$) | Bán kính hiệu dụng | Diện tích hộp so với `mult=1` |
|---|---|---|---|
| `1.0` | $2\ln 255=11.08$ | $3.33\,\sigma'$ | 100% |
| `0.7` (Tanks&Temples, Deep Blending trong cả `train_base.sh` lẫn `train_big.sh`) | 7.76 | $2.79\,\sigma'$ | 70% |
| `0.5` (mặc định, `arguments/__init__.py:101`) | 5.54 | $2.35\,\sigma'$ | 50% |
| `0.3` | 3.32 | $1.82\,\sigma'$ | 30% |

Nên `mult=0.5` **giảm nửa** diện tích hộp, không phải giảm ba phần tư như cách hiểu "nhân thẳng vào cạnh".

**(b) Hộp phụ thuộc opacity** — điều hộp $3\sigma$ của 3DGS không làm được:

| $o$ | $t$ tại `mult=0.5` | Bán kính hiệu dụng | $K$ đo được ($\sigma'=8$px, đẳng hướng) |
|---|---|---|---|
| 1.0 | 5.54 | $2.35\,\sigma'$ | 12.0 |
| 0.5 | 4.85 | $2.20\,\sigma'$ | 9.0 |
| 0.1 | 3.24 | $1.80\,\sigma'$ | 6.0 |
| 0.02 | 1.63 | $1.28\,\sigma'$ | 5.0 |

Ý nghĩa: những Gaussian mờ — chính là loại đông đảo nhất trong giai đoạn giữa huấn luyện, ngay trước khi bị prune — gần như **miễn phí** về mặt rasterization. 3DGS trả giá đầy đủ cho chúng.

> Suy ra từ công thức, chưa đo: khi $o<1/255$ thì $t<0$ và các `sqrt` ở dòng 340-343 nhận đối số âm. Ngưỡng prune `min_opacity = 0.005` (`train.py:141`) nằm ngay trên $1/255=0.0039$, nên vùng này gần như không chạm tới trong thực tế — nhưng nó không được chặn tường minh bởi kiểm tra ellipse suy biến ở dòng 330.

**(c) `mult = 1.0` KHÔNG quay về hành vi 3DGS gốc.** Nó quay về SnugBox nguyên bản của Speedy-Splat. Xét **riêng bán kính**, ở `mult=1`, $o=1$ SnugBox cho $3.33\sigma'$ — **rộng hơn** $3\sigma$: nó bao đúng tới mức 1/255 chứ không cắt ở $3\sigma$. Cái làm nên tiết kiệm không phải bán kính mà là **(b)** cộng với **hộp theo từng trục** (§4.2) và **§4.4**. Trong codebase này **không có** đường quay lại cách dựng hộp của 3DGS.

## 4.4 — Tập tile chạm không phải hình chữ nhật

Sau khi có hộp, `processTiles` (`auxiliary.h:199-315`, phần *AccuTile* của Speedy-Splat) duyệt từng **lát tile** theo trục ngắn hơn (`isY = y_span < x_span`) và với mỗi lát gọi `computeEllipseIntersection` để lấy **giao chính xác của ellipse với lát đó**. Tile nào nằm trong hộp nhưng ngoài ellipse thì bị loại luôn, không vào danh sách sort.

Về mặt toán, đây là bài toán: đếm số tile $T$ sao cho

$$\min_{\Delta\in T}\ \Delta^\top M\,\Delta \;\le\; t$$

Trong giới hạn splat lớn (nhiều tile), tỉ lệ tile giữ lại tiến tới tỉ lệ diện tích ellipse trên diện tích hộp bao:

$$\frac{\pi\,\text{half}_x\,\text{half}_y}{4\,\text{half}_x\,\text{half}_y}=\frac{\pi}{4}\approx 0.785$$

Nghĩa là bước lọc ellipse tự nó tiết kiệm thêm **~21%** so với chỉ dùng hộp — và đó là con số §6.2 đo được (0.373 → 0.290).

> `mult` phải truyền **nhất quán cho cả train và render** — thấy rõ trong `train_base.sh` (dòng 10-13 train và 24-27 render đều `--mult 0.7`). Đường notebook không có rủi ro này: `pipeline/trainer.py:108` và `pipeline/submission.py:52` dùng chung một `cfg.mult`.

## 4.5 — Tổng hợp: công thức đầy đủ của $R_{\text{tile}}$

Gộp §4.1–§4.4, trong giới hạn splat lớn:

$$R_{\text{tile}}
=\frac{K_{\text{fast}}}{K_{\text{3dgs}}}
\;\approx\;
\underbrace{\frac{\pi}{4}}_{\text{lọc ellipse}}
\cdot
\underbrace{\frac{4\,t\,\sqrt{\Sigma'_{11}\Sigma'_{22}}}{36\,\lambda_{\max}}}_{\text{hộp theo trục vs hộp vuông}},
\qquad t=\texttt{mult}\cdot 2\ln(255\,o)$$

Ba biến điều khiển hiện rõ:

| Biến | Ảnh hưởng |
|---|---|
| `mult` | tuyến tính — $R_{\text{tile}}\propto\texttt{mult}$ |
| $o$ | qua $\ln$ — splat mờ được thưởng, nhưng lợi ích giảm dần |
| $\rho$ (độ dẹt) | **mạnh nhất** — $\sqrt{\Sigma'_{11}\Sigma'_{22}}/\lambda_{\max}\sim1/\rho$ |

Đo bằng `demos/fastgs_cost_model.py` ($\sigma_g=8$px, $o=1$, `mult=0.5`, lấy trung bình theo hướng $\theta$ và vị trí tâm):

| $\rho$ (tỉ lệ trục) | $K_{\text{3dgs}}$ | $K_{\text{fastgs}}$ | $R_{\text{tile}}$ |
|---|---|---|---|
| 1.0 (tròn) | 16.00 | 10.14 | **0.634** |
| 1.5 | 30.25 | 10.74 | **0.355** |
| 2.0 | 49.00 | 11.91 | **0.243** |
| 3.0 | 100.00 | 14.54 | **0.145** |
| 5.0 (rất dẹt) | 256.00 | 20.68 | **0.081** |

Đọc cột cuối: **splat càng dẹt, compact box càng thắng đậm.** Với splat tròn nó chỉ tiết kiệm 37%; với splat dẹt tỉ lệ 5:1 nó tiết kiệm hơn 12×. Vì Gaussian sau huấn luyện có xu hướng dẹt (§4.1), giá trị thực tế nằm ở nửa dưới bảng.

---
---

# Phần V — Đòn bẩy 3: giảm nhịp cập nhật tham số

## 5.1 — Lịch optimizer thưa dần

`scene/gaussian_model.py:225` — `optimizer_step(iteration)` **không** bước Adam mỗi vòng:

| Khoảng vòng lặp | `self.optimizer` (xyz, f_dc, opacity, scaling, rotation) | `self.shoptimizer` (`f_rest`) |
|---|---|---|
| `iteration <= 15000` (dòng 227) | mỗi vòng | **mỗi 16 vòng** |
| `15000 < iteration <= 20000` (dòng 233) | mỗi 32 vòng | mỗi 32 vòng |
| `iteration > 20000` (dòng 239) | mỗi 64 vòng | mỗi 64 vòng |

Đếm chính xác trên 30.000 vòng (`adam_steps()` trong cost model):

| | `optimizer` | `shoptimizer` | tổng |
|---|---|---|---|
| 3DGS gốc | 30.000 | 30.000 | 60.000 |
| fastgs-lite | 15.313 | 1.250 | **16.563** |
| tỉ số | 0.510 | **0.042** | **0.276** |

Đây là con số **chính xác**, không phải ước lượng — nó đọc thẳng từ vòng lặp trong code.

Gradient vẫn cộng dồn bình thường mỗi vòng (vì `zero_grad` chỉ gọi khi thực sự `step`), nên mỗi bước hiếm hoi đó áp một gradient đã tích luỹ. Về mặt tối ưu, nó gần với **gradient accumulation** hơn là bỏ bớt cập nhật — nhưng Adam chuẩn hoá theo $\sqrt{v}$, nên cộng dồn gradient rồi step một lần **không** tương đương với step nhiều lần: bước đi bị chuẩn hoá về cùng cỡ $\approx\text{lr}$ bất kể tích luỹ bao nhiêu. Đó chính là điều làm 15k–30k rẻ gần như miễn phí, và cũng là điều làm nó **học được rất ít** trong giai đoạn đó.

Cùng với `densify_until_iter = 15000` và `position_lr_max_steps = 30000`: **toàn bộ chi phí nằm ở 0–15k**, còn **15k–30k gần như miễn phí** nhưng vẫn chạy `final_prune_fastgs` (§3.4).

⇒ Hạ `--iterations` xuống 15k là lỗ: tiết kiệm rất ít thời gian mà mất phần tỉa cuối. Sàn hợp lý là **20000**, và khi rút ngắn phải đồng bộ `--position_lr_max_steps`, `--densify_until_iter`, cùng hai ngưỡng cứng ở dòng 227/233.

## 5.2 — Tách learning rate cho Spherical Harmonics

> Mô phỏng: `steps_to_converge()` (mục "5.").

3DGS gốc cũng đã tách SH bậc thấp / bậc cao, nhưng bằng **một** tham số: `feature_lr` cho `features_dc` và `feature_lr / 20` cho `features_rest`. fastgs-lite giữ nguyên phép chia 20 đó và thêm **hai flag độc lập**.

```python
# scene/gaussian_model.py:198-205
l = [ ...
      {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
      ... ]
sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]
```

| Tham số | Điều khiển | Mặc định (`arguments/__init__.py:97-98`) | **LR thực sự nạp vào Adam** |
|---|---|---|---|
| `--lowfeature_lr` | `features_dc` — SH bậc 0, màu cơ bản không phụ thuộc góc nhìn | `0.0025` | `0.0025` |
| `--highfeature_lr` | `features_rest` — SH bậc 1–3, phần màu đổi theo góc nhìn | `0.005` | **`0.005 / 20 = 0.00025`** |

⚠️ **Đọc kỹ cột cuối.** Con số bạn truyền vào `--highfeature_lr` bị chia 20 trước khi tới optimizer. Ở mặc định, SH bậc cao học với lr **nhỏ hơn 10 lần** SH bậc thấp. Ngay cả preset `0.02` của `train_base.sh`/`pipeline/config.py` cũng chỉ ra lr hiệu dụng `0.001`, vẫn **thấp hơn** `lowfeature_lr = 0.0025`.

Cộng với nhịp 1/16 ở §5.1, trong giai đoạn 0–15k `f_rest` nhận **lr nhỏ hơn 10× và số bước ít hơn 16×** so với `f_dc`. Đó là lý do vệt specular hội tụ chậm, quan sát trực tiếp trong [history-train.md](history-train.md) §1.7 (vệt sáng trên mặt tủ gỗ của `drjohnson` gần như biến mất sau 7000 vòng).

**Vì sao thiết kế như vậy:** SH bậc 0 mang phần lớn năng lượng màu — lr cao ở đó làm cả ảnh dao động. SH bậc 1–3 là hiệu chỉnh nhỏ quanh màu nền, và có 45 hệ số trên mỗi Gaussian (so với 3) — cho chúng lr đầy đủ ở nhịp đầy đủ vừa tốn thời gian optimizer vừa dễ overfit vào từng góc nhìn.

Giá trị theo cảnh trong `train_base.sh`:

| Cảnh | `--highfeature_lr` | LR hiệu dụng |
|---|---|---|
| `garden`, `room`, `counter`, `kitchen`, `bonsai` (MipNeRF360 nhiều phản xạ) | `0.02` | `0.001` |
| `truck` / `train` (Tanks&Temples) | `0.04` / `0.042` | `0.002` / `0.0021` |
| `drjohnson` / `playroom` (Deep Blending, trong nhà ít specular) | `0.0025` / `0.0015` | `0.000125` / `0.000075` |
| `bicycle`, `flowers`, `stump`, `treehill` (không truyền cờ) | mặc định `0.005` | `0.00025` |

> **Phép tách lr SH KHÔNG phải đòn bẩy tốc độ.** Nó là đòn bẩy *chất lượng* (và trong cấu hình mặc định, không rõ nó có giúp gì không). Thứ tăng tốc là **nhịp 1/16** ở §5.1, không phải giá trị lr. Đừng gộp hai thứ này.

> **Lưu ý về `--dense`.** Cách chia "trong nhà `0.001` / ngoài trời `0.004–0.01`" hay được nhắc kèm **không khớp `train_base.sh`**: `drjohnson` (trong nhà) dùng `--dense 0.013`, cao nhất cả file, còn `playroom` (cũng trong nhà) dùng `0.003`. Đừng chọn `--dense` theo nhãn indoor/outdoor — copy thẳng dòng của cảnh giống dataset của bạn nhất.

---
---

# Phần VI — Phép đo fastgs-lite vs 3DGS

> Chạy: `python demos/fastgs_cost_model.py` — không cần GPU, không cần dataset.

## 6.1 — Ba tỉ số, và tỉ số nào thật sự đo được

Mô hình ở Phần II quy toàn bộ câu hỏi "nhanh hơn bao nhiêu" về ba tỉ số. Chúng **không** cùng độ tin cậy:

| Tỉ số | Đo bằng gì | Độ tin cậy |
|---|---|---|
| $R_{\text{adam}}$ | Đếm vòng lặp trong `optimizer_step` | **Chính xác** — chỉ là số học trên code |
| $R_{\text{tile}}$ | Hình học: dựng hộp theo cả hai công thức, đếm tile trên quần thể splat mô phỏng | **Suy ra được** — phụ thuộc phân bố $(\sigma',\rho,o)$ giả định, nhưng cơ chế thì đúng |
| $R_{\text{gauss}}$ | Không có cách nào ngoài chạy thật cả hai | **Giả định** — đây là biến duy nhất bắt buộc phải đo A/B |

Nói thẳng: **repo này không có phép đo A/B nào giữa fastgs-lite và 3DGS.** Cost model dưới đây thay thế phép đo đó bằng một mô hình có tham số lộ thiên, để ít nhất biết *cái gì đóng góp bao nhiêu* và *cần đo cái gì*.

## 6.2 — Phép đo 1: số tile mỗi splat

Quần thể mô phỏng: 3000 splat, $\sigma'$ log-normal (trung vị 2.9 px), tỉ lệ trục log-normal, hướng đều, opacity Beta(2,1) — trung bình 0.66.

| `mult` | $K_{\text{3dgs}}$ | $K$ chỉ compact box | $K_{\text{fastgs}}$ (đủ) | $R_{\text{tile}}$ |
|---|---|---|---|---|
| 1.0 | 21.91 | 13.65 | 10.00 | 0.457 |
| 0.7 | 21.91 | 10.42 | 7.88 | 0.360 |
| **0.5** (mặc định) | 21.91 | 8.18 | **6.36** | **0.290** |
| 0.3 | 21.91 | 5.81 | 4.71 | 0.215 |

Tách hai nguồn tiết kiệm tại `mult=0.5`:

| | $K$ | tỉ lệ |
|---|---|---|
| Hộp vuông $3\sigma$ (3DGS) | 21.91 | 1.000 |
| Compact box, **chưa** lọc ellipse | 8.18 | 0.373 |
| **+ lọc tile theo ellipse** (AccuTile) | 6.36 | **0.290** |

Đọc: phần lớn tiết kiệm (1.000 → 0.373) đến từ **cách dựng hộp** — hộp theo từng trục + phụ thuộc opacity, tức §4.2–§4.3. Bước lọc ellipse chính xác đóng góp thêm 0.373 → 0.290, đúng bằng hệ số $\pi/4$ dự đoán ở §4.4.

## 6.3 — Phép đo 2: số lần Adam step (chính xác)

Bảng ở §5.1: $R_{\text{adam}}=16{,}563/60{,}000=\mathbf{0.276}$.

## 6.4 — Phép đo 3: số Gaussian (giả định, không phải đo)

Quỹ đạo mô hình, xuất phát 100k điểm SfM, `densification_interval=500`:

| | tham số | $N$ trung bình | $N$ cuối |
|---|---|---|---|
| 3DGS gốc | spawn 10%, prune 0.5% mỗi lần densify | 933.036 | 1.371.694 |
| fastgs-lite | spawn 6%, prune 1%, final-prune 5% | 292.773 | 329.750 |
| tỉ số | | **0.314** | 0.240 |

**Đây là chỗ yếu nhất của cả tài liệu.** Hai cặp `(spawn, prune)` là số giả định, chọn cho khớp với bậc độ lớn hay thấy: 3DGS kết thúc quanh 1–3 triệu Gaussian, các biến thể có prune kết thúc quanh 0.3–0.6 triệu. Nhưng chúng là *giả định*. Muốn biết thật thì phải chạy A/B (§6.6).

Điểm chắc chắn duy nhất: cơ chế densify AND (§3.3) **chỉ có thể giảm** $N$ so với 3DGS, không bao giờ tăng — vì nó là một phép AND thêm vào tập điều kiện. Nên $R_{\text{gauss}}\le 1$ là kết luận an toàn; giá trị cụ thể thì không.

## 6.5 — Gộp: tăng tốc theo mô hình

Chia chi phí một vòng 3DGS thành: preprocess 15%, rasterization 55%, Adam 15%, cố định ($\mathcal{L}$+SSIM+IO) 15%. Cộng overhead scoring +2% cho fastgs (§2.3):

| Cấu hình | Tăng tốc |
|---|---|
| **Cả ba đòn bẩy** | **3.83×** |
| Chỉ compact box (`mult=0.5`) | 1.64× |
| Chỉ giảm số Gaussian | 2.38× |
| Chỉ nhịp Adam thưa | 1.12× |
| *Trần Amdahl* ($N\to0$, $K\to0$) | *6.7×* |

Ba điều đáng chú ý:

1. **$1.64\times2.38\times1.12 = 4.37 > 3.83$.** Các đòn bẩy nhân nhau ở số hạng rasterization (§2.2) nhưng **không** nhân được ở số hạng cố định $F$ — đó là khoảng chênh. Càng gần trần Amdahl, thêm đòn bẩy càng ít tác dụng.
2. **Nhịp Adam chỉ đóng góp 1.12×** dù $R_{\text{adam}}=0.276$ — vì Adam chỉ chiếm 15% chi phí. Nó rẻ để làm, nhưng không phải nguồn tăng tốc chính.
3. **Giảm số Gaussian đóng góp nhiều nhất (2.38×)** vì $N$ xuất hiện trong **cả ba** số hạng biến thiên. Mà đó lại đúng là tỉ số kém tin cậy nhất. Nên: nếu chỉ đo được một thứ, hãy đo $N$.

Độ nhạy: đổi `SPLIT_FIXED` trong `demos/fastgs_cost_model.py` từ 0.15 xuống 0.05 sẽ đẩy trần Amdahl lên 20× và tổng tăng tốc lên đáng kể. Con số 3.83× **không** là hằng số của phương pháp — nó là hàm của bốn tham số chia chi phí, và tất cả đều đang là giả định.

## 6.6 — Giao thức A/B thật (repo chưa chạy)

Muốn thay $R_{\text{gauss}}$ giả định bằng số đo thật, cần đúng bốn thứ:

1. **Cùng máy, cùng dataset, cùng resolution.** Thời gian train không so được qua GPU khác nhau, và `--resolution 2` đổi $H\times W$ nên đổi cả $F$ lẫn $K$.
2. **Cùng ngân sách vòng lặp, tối thiểu 20.000** — dưới 15k thì `final_prune_fastgs` không chạy (§3.4) và fastgs bị đo thiếu một đòn bẩy.
3. **Ghi ba đại lượng chứ không chỉ wall-clock:**
   - $N$ tại vài mốc (1k, 5k, 10k, 15k, 30k) → cho $R_{\text{gauss}}$ thật
   - tổng `tiles_touched` mỗi vòng (kernel đã tính sẵn, chỉ cần cộng và log) → cho $R_{\text{tile}}$ thật
   - thời gian từng giai đoạn (preprocess / raster / loss / backward / optimizer) → cho bốn tham số chia chi phí ở §6.5
4. **Đối thủ đúng là 3DGS gốc**, không phải `--mult 1.0`. Như §4.3(c) nêu rõ, `mult=1` là SnugBox chứ không phải hộp vuông $3\sigma$ — codebase này **không có** đường quay lại hành vi 3DGS, nên phải cài đặt riêng repo upstream.

Có đủ bốn thứ đó thì cost model chuyển từ "mô hình" sang "phép đo", và mọi con số ở §6.5 có thể tính lại bằng cách thay ba tỉ số vào cùng công thức.

## 6.7 — So sánh theo mã nguồn

Bảng dưới chỉ liệt kê khác biệt **đọc được trực tiếp từ code trong repo này**:

| | 3DGS gốc | fastgs-lite |
|---|---|---|
| Tiêu chí densify | Chỉ gradient vị trí | Gradient **AND** `importance_score > 5` (`gaussian_model.py:494`) |
| Tín hiệu split | `densify_grad_threshold` | `grad_abs_thresh` trên gradient trị tuyệt đối, kiểu Abs-GS (`:527-530`) |
| Xoá Gaussian | Prune theo opacity/kích thước | Thêm lấy mẫu theo `pruning_score`, cộng `final_prune_fastgs` sau vòng 15k |
| Hộp bao khi rasterize | Hộp **vuông** cạnh $3\sqrt{\lambda_{\max}}$ (`forward.cu:240` + `getRect`) | Compact box theo từng trục: $t=\texttt{mult}\cdot 2\ln(255\,o)$ + giao ellipse–tile chính xác (Phần IV) |
| LR cho SH | Một `feature_lr` (và `/20` cho bậc cao) | Hai flag `lowfeature_lr` / `highfeature_lr` (vẫn `/20`), optimizer riêng |
| Nhịp Adam | Mỗi vòng, suốt 30k | Mỗi vòng tới 15k, rồi 1/32, rồi 1/64 (`optimizer_step`) |
| Chi phí phụ | — | 20 render phụ mỗi lần densify (`compute_gaussian_score_fastgs`) |

## 6.8 — Số đo thật duy nhất có trong repo

Một lần chạy fastgs-lite trên Colab free T4, ghi trong [`DOCS/assets/leaderboard.csv`](assets/leaderboard.csv) và thuật lại ở [history-train.md](history-train.md):

| | Giá trị |
|---|---|
| Cấu hình | `iterations=7000`, `resolution=2`, `densification_interval=500`, T4 14.56 GB |
| Thời gian train | 327.6 s cho 4 cảnh (67–95 s/cảnh, ~85 it/s) |
| Score trung bình (LPIPS-VGG, full-res) | 0.7643 |
| Số Gaussian | 129k–201k tuỳ cảnh |
| Peak VRAM | 0.68–1.16 GB |

Đây là ngân sách 7000 vòng, tức **chưa từng chạy `final_prune_fastgs`** (§3.4) — không so sánh được với bất kỳ con số published nào, vốn dùng ngân sách 30k. Con số $N$ (129k–201k) đáng chú ý: nó thấp hơn *một bậc độ lớn* so với 3DGS ở ngân sách đầy đủ, nhưng phần lớn chênh lệch đó đến từ **ngân sách ngắn**, không phải từ cơ chế — nên không dùng nó làm bằng chứng cho $R_{\text{gauss}}$.

## 6.9 — Kết luận

**Về mặt toán học, fastgs-lite nhanh hơn vì nó tấn công đúng cái tích $N\cdot K$** — số cặp (tile, Gaussian) — thay vì chỉ một thừa số:

- $K$ giảm vì hộp bao chuyển từ **vuông theo trục dài nhất** sang **theo từng trục, phụ thuộc opacity, lọc theo ellipse chính xác**. Với splat dẹt — trạng thái bình thường của Gaussian đã hội tụ — tiết kiệm lên tới hơn 10× (§4.5).
- $N$ giảm vì tiêu chí densify đổi từ *"gradient lớn"* sang *"gradient lớn **và** nhiều camera cùng thấy sai"*, và vì có thêm hai đường prune dựa trên cùng tín hiệu đó. Vì densify chạy lặp ~30 lần, khác biệt nhỏ ở tỉ lệ sinh khuếch đại thành khác biệt lớn ở $N$ cuối.
- Nhịp Adam thưa dần cắt số hạng thứ ba, nhưng đóng góp khiêm tốn (1.12×).

Tín hiệu nhất quán đa góc nhìn là thứ khiến hai đòn bẩy đầu khả thi: nó vừa **rẻ để tính** (20 render phụ mỗi `densification_interval` vòng, +2% chi phí), vừa **chọn lọc hơn gradient** — nên thêm Gaussian đúng chỗ, bỏ Gaussian đúng lúc.

Nhưng **repo chưa chứng minh được con số cụ thể**. Mô hình cho 3.83×; giá trị thật phụ thuộc $R_{\text{gauss}}$ và cách chia chi phí, cả hai đều chưa đo. §6.6 nói cách đo.

---
---

# Phần VII — Ba điều mã nguồn nói mà README upstream không nói

Sáu phần trên mô tả **thiết kế**. Ba mục dưới đây là kết quả đọc mã nguồn, và chúng quyết định cách cấu hình thực tế.

## 7.1 — Lịch optimizer bị đóng đinh theo ngân sách 30.000 vòng

Xem §5.1. Tóm tắt hệ quả: hạ `--iterations` xuống 15k là lỗ — tiết kiệm rất ít thời gian mà mất phần tỉa cuối. Sàn hợp lý là **20000**, và khi rút ngắn phải đồng bộ `--position_lr_max_steps`, `--densify_until_iter`, cùng hai ngưỡng cứng ở `gaussian_model.py:227` và `:233`.

## 7.2 — `--antialiasing` là cờ chết trong fork này

`arguments/__init__.py:71` khai báo `self.antialiasing = False`, nhưng `gaussian_renderer/__init__.py` dựng `GaussianRasterizationSettings(...)` **không có** trường đó, và `submodules/diff-gaussian-rasterization_fastgs/` không tham chiếu chữ `antialiasing` ở đâu. Truyền cờ chỉ bị bỏ qua âm thầm. Muốn chống răng cưa thật phải vá công thức Mip-Splatting vào kernel CUDA.

## 7.3 — `densification_interval` là lever thời gian bị bỏ quên

Mặc định trong `arguments/__init__.py` là `100`, nhưng `train_base.sh` thật sự dùng `500`. Vì `compute_gaussian_score_fastgs` render 10 camera × 2 lượt mỗi lần gọi, khác biệt là **~145 lần gọi so với ~29 lần** trong khoảng 500–15000.

Quy về mô hình chi phí (§2.3): overhead scoring là

$$\frac{\text{densify\_until}}{\text{interval}}\times\frac{20}{\text{total\_iters}}$$

| `densification_interval` | Số lần gọi (tới 15k) | Overhead raster |
|---|---|---|
| 500 | 30 | **+2.0%** |
| 200 | 75 | +5.0% |
| 100 (mặc định trong `arguments/`) | 150 | **+10.0%** |

Ở `interval=100`, riêng bước chấm điểm đã ăn 10% chi phí rasterization — đủ để triệt tiêu phần lớn lợi ích của nhịp Adam thưa (1.12×). Đây là lý do preset thật dùng 500.

---

## Đọc tiếp

| Tài liệu | Nội dung |
|---|---|
| [`demos/fastgs_cost_model.py`](../demos/fastgs_cost_model.py) | **Phép đo Phần VI** — chạy `python demos/fastgs_cost_model.py`, không cần GPU. Sửa `SPLIT_*` để thử giả định khác |
| [`demos/fastgs_mechanisms.py`](../demos/fastgs_mechanisms.py) | Mô phỏng NumPy từng cơ chế ở Phần III–V (`python demos/fastgs_mechanisms.py`) |
| [`fastgs-acceleration-method.ipynb`](../fastgs-acceleration-method.ipynb) | Notebook dựng pipeline `pipeline/` thật trên Colab T4 (check GPU → config → load dataset → smoke test → train → analytics → submission.zip) — cần GPU |
| [colab-t4-guide.md](colab-t4-guide.md) | Hướng dẫn vận hành pipeline Colab T4: preset, theo dõi tiến độ, xử lý sự cố |
| [history-train.md](history-train.md) | Số đo thật của các phiên train đã chạy — nguồn của §6.8 |
| [README.md](README.md) | Tổng quan repo và cách bắt đầu |

Mã nguồn tương ứng: `utils/fast_utils.py`, `scene/gaussian_model.py:198–244` và `:468–540`, `train.py:126–158`, `gaussian_renderer/__init__.py`, `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/auxiliary.h:175–345` và `forward.cu:228–270`.
