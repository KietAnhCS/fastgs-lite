# Toán học trong 3D Gaussian Splatting

## Phần 1: Alpha-blending dựa trên điểm (Point-based Blending)

### Công thức (3) — Pha trộn màu tại pixel

$$C=\sum_{i\in N}c_i\,\alpha_i\prod_{j=1}^{i-1}(1-\alpha_j)$$

**Ý nghĩa từng ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---|---|
| $C$ | Màu cuối cùng của pixel |
| $N$ | Số Gaussian chồng lên pixel đó, đã sắp xếp theo độ sâu (gần → xa) |
| $c_i$ | Màu của Gaussian thứ $i$ |
| $\alpha_i$ | Độ mờ (opacity) của Gaussian $i$ tại đúng vị trí pixel đang xét |
| $\prod_{j=1}^{i-1}(1-\alpha_j)$ | Độ truyền qua tích lũy $T_i$ — phần ánh sáng còn lại sau khi đã bị các Gaussian phía trước "che" bớt |

**Bản chất:** GS biểu diễn cảnh bằng một tập **hữu hạn** các Gaussian 3D rời rạc. Khi render, các Gaussian phủ lên một pixel được sắp xếp theo độ sâu rồi blend tuần tự từ gần đến xa — điểm gần nhất đóng góp trọn vẹn $\alpha_1$, điểm sau bị chiết khấu bởi phần đã được "hấp thụ" từ các điểm trước.

Đây là công thức kế thừa từ dòng nghiên cứu **point-based neural rendering** (Kopanas et al. 2021, 2022), được 3D Gaussian Splatting (Kerbl et al. 2023) áp dụng làm cơ chế blending chính.

**Ví dụ số:** 3 Gaussian phủ 1 pixel, sắp theo thứ tự gần→xa:

| i | màu $c_i$ | $\alpha_i$ | $T_i=\prod(1-\alpha_j)$ | đóng góp $T_i\alpha_ic_i$ |
|---|---|---|---|---|
| 1 | đỏ (1,0,0) | 0.5 | 1.0 | (0.5, 0, 0) |
| 2 | xanh lá (0,1,0) | 0.4 | 0.5 | (0, 0.2, 0) |
| 3 | xanh dương (0,0,1) | 0.6 | 0.3 | (0, 0, 0.18) |

$$C = (0.5,\ 0.2,\ 0.18)$$

Nếu nền trắng $(1,1,1)$, độ truyền qua còn lại $T_4=0.5\times0.6\times0.4=0.12$:

$$C_{final} = (0.5,0.2,0.18) + 0.12\times(1,1,1) = (0.62,\ 0.32,\ 0.30)$$

Gaussian gần nhất đóng góp nhiều nhất dù không có $\alpha$ cao nhất — vì chưa bị điểm nào che ($T_1=1$). Đây là lý do **depth sorting** cực kỳ quan trọng: đổi thứ tự sẽ ra màu khác hẳn dù $\alpha_i, c_i$ không đổi.

---

## Phần 2: Biểu diễn và Chiếu 3D Gaussian

### Công thức (4) — Hàm phân phối 3D Gaussian cơ bản

$$G(x)=e^{-\frac{1}{2}x^T\Sigma^{-1}x}$$

**Ý nghĩa:**
- $x$: vị trí tương đối so với tâm Gaussian ($x = \mathbf{p} - \boldsymbol{\mu}$)
- $\Sigma$: ma trận hiệp phương sai, quyết định hình dạng, kích thước, hướng của "quả bóng" Gaussian
- $G(x)$: giá trị mật độ Gaussian tại $x$, cực đại $=1$ tại tâm, giảm dần theo khoảng cách Mahalanobis

**Vai trò trong blending:** $\alpha_i$ trong công thức (3) không phải hằng số cho cả Gaussian — nó bằng opacity học được $\alpha'_i$ nhân với giá trị Gaussian tại đúng vị trí pixel:

$$\alpha_i = \alpha'_i \cdot G(x)$$

Càng xa tâm Gaussian, $\alpha_i$ càng giảm nhanh — tạo hiệu ứng mờ dần tự nhiên ở biên thay vì cạnh cứng.

> **Lưu ý về ký hiệu:** Công thức (4) viết $\Sigma^{-1}$ như một định nghĩa Gaussian **tổng quát**, dùng chung cho cả trường hợp 3D (công thức 6) lẫn 2D. Khi thực sự tính $\alpha$ tại một pixel màn hình, $\Sigma$ ở đây phải được hiểu ngầm là $\Sigma'$ (đã chiếu 2D, từ công thức 5), **không phải** $\Sigma$ 3D gốc. Cách viết chung ký hiệu này dễ gây hiểu lầm — cách viết rõ ràng hơn nên là $\alpha_i(\mathbf{x}) = \alpha'_i \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}'_i)^T{\Sigma'_i}^{-1}(\mathbf{x}-\boldsymbol{\mu}'_i)\right)$ để phân biệt rõ với $\Sigma$ 3D.

---

### Công thức (5) — Phép chiếu Covariance từ 3D xuống 2D (Splatting)

$$\Sigma'=JW\Sigma W^T J^T$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $\Sigma$ | Ma trận hiệp phương sai 3D gốc (world space) |
| $W$ | Ma trận biến đổi view (world → camera space) |
| $J$ | Ma trận Jacobian của phép chiếu xạ ảnh — tuyến tính hóa cục bộ phép chiếu phối cảnh phi tuyến |
| $\Sigma'$ | Ma trận hiệp phương sai 2D sau khi chiếu lên mặt phẳng ảnh (screen space) |

**Vai trò:** Biến một "quả bóng" Gaussian 3D (ellipsoid) thành một **vệt elip 2D** trên màn hình — gọi là "splat". Một Gaussian 3D chiếu xuống vẫn cho ra một Gaussian 2D, nên chỉ cần tính $\Sigma'$ một lần mỗi Gaussian mỗi khung hình, rồi dùng công thức (4) phiên bản 2D để tính $\alpha$ tại từng pixel.

---

### Công thức (6) — Phân rã Ma trận Hiệp phương sai 3D

$$\Sigma=RSS^TR^T$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $R$ | Ma trận xoay 3×3 (tham số hóa bằng quaternion khi tối ưu) |
| $S$ | Ma trận tỷ lệ (scale), dạng đường chéo |

**Tại sao cần phân rã?** $\Sigma$ phải luôn bán xác định dương để có ý nghĩa vật lý hợp lệ. Tối ưu trực tiếp các phần tử của $\Sigma$ bằng gradient descent rất dễ vi phạm ràng buộc này. Tham số hóa qua $R$ (trực giao — luôn hợp lệ) và $S$ (đường chéo dương) khiến $RSS^TR^T$ **luôn tự động** hợp lệ, bất kể cập nhật thế nào.

**Tham số học được của mỗi Gaussian:** vị trí $\boldsymbol{\mu}$ (3), rotation quaternion (4), scale $S$ (3), opacity $\alpha'$ (1), màu qua hệ số Spherical Harmonics.

---

## Phần 3: Hàm Mất mát (Loss Function)

### Công thức (7) — Hàm Loss tổng quát

$$\mathcal{L}=(1-\lambda)\mathcal{L}_1+\lambda\,\mathcal{L}_{\text{D-SSIM}}$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathcal{L}_1$ | Sai số tuyệt đối trung bình giữa ảnh render và ground-truth |
| $\mathcal{L}_{\text{D-SSIM}}$ | Loss cấu trúc (Structural Dissimilarity) — đo tương đồng cấu trúc cục bộ |
| $\lambda$ | Trọng số cân bằng (thường $\approx 0.2$ trong paper gốc) |

Kết hợp cả hai giúp Gaussian được tối ưu vừa đúng màu vừa giữ chi tiết cấu trúc sắc nét. Loss này lan truyền gradient ngược về tất cả tham số của từng Gaussian ($\boldsymbol{\mu}, R, S, \alpha', c$) qua chuỗi (5)→(4)→(3), đồng thời điều khiển **adaptive density control** (thêm/xóa/tách Gaussian) trong huấn luyện.

---

## Phần 4: Tại sao phải chiếu Gaussian xuống 2D?

### 4.1. Hai cách khả dĩ để tính $\alpha$ tại một pixel

**Cách 1 — Giữ nguyên 3D, ray-tracing (như NeRF):**
1. Bắn tia 3D từ camera qua pixel
2. Tìm giao điểm giữa tia và từng Gaussian
3. Tính $x$ = vị trí giao điểm 3D trừ tâm Gaussian, đưa vào công thức (4)
4. Lặp cho mọi Gaussian có khả năng giao với tia

→ Chi phí lớn: mỗi pixel phải duyệt/tính giao điểm trong không gian 3D không có cấu trúc lưới. Đây là lý do render kiểu ray-marching chậm.

**Cách 2 — Chiếu Gaussian xuống 2D trước, rồi rasterize (GS):**
1. Tính trước $\Sigma' = JW\Sigma W^TJ^T$ cho mỗi Gaussian — ra elip 2D cố định trên màn hình
2. Xác định ngay bounding box các pixel bị elip phủ (không cần dò bằng tia)
3. Với mỗi pixel $(u,v)$ trong bounding box, $x = (u,v) - \boldsymbol{\mu}'$ — chỉ là phép trừ tọa độ 2D
4. Đưa $x$ vào công thức (4) với $\Sigma'^{-1}$ để ra $\alpha$ tại đúng pixel

→ Đây là bài toán **rasterization** kinh điển, đã được GPU tối ưu hàng chục năm.

### 4.2. Vì sao không thể tính thẳng bằng $\Sigma^{-1}$ (3D)?

Khoảng cách Mahalanobis $x^T\Sigma^{-1}x$ trong 3D đo độ gần/xa trong không gian thật, nhưng cái cần để tô pixel là độ gần/xa trong **không gian ảnh** — hai đại lượng này **không tương đương tuyến tính** vì phép chiếu phối cảnh phi tuyến:

- **Foreshortening theo góc nhìn:** Gaussian hình cầu nhìn nghiêng sẽ trông dẹt thành elip — phụ thuộc góc nhìn hiện tại, điều $\Sigma$ 3D cố định không tự phản ánh.
- **Phối cảnh theo khoảng cách:** Gaussian gần trông to hơn, xa trông nhỏ hơn dù $\Sigma$ không đổi — quan hệ phi tuyến, nên cần Jacobian $J$ để tuyến tính hóa cục bộ quanh tâm Gaussian.

$\Sigma^{-1}$ 3D cho biết "hình dạng thật" trong không gian, nhưng không cho biết Gaussian trông ra sao và ở đâu trên màn hình tại góc camera hiện tại. Thiếu bước chiếu, không thể biết Gaussian phủ pixel nào, méo ra sao, và α suy giảm theo hướng nào trên ảnh 2D.

### 4.3. Tóm tắt lợi ích của việc chiếu 2D

| Không chiếu (giữ 3D) | Có chiếu xuống 2D |
|---|---|
| Ray-cast dò giao điểm cho từng pixel | Biết ngay bounding box pixel bị phủ |
| $x$ = kết quả giao tia–ellipsoid (tốn kém) | $x$ = phép trừ tọa độ 2D đơn giản |
| Không tận dụng rasterization của GPU | Tận dụng trực tiếp pipeline rasterize GPU |
| Dễ sai lệch nếu không xử lý đúng độ méo phối cảnh | $J$ đảm bảo elip 2D đúng với góc nhìn thực tế |
| Chậm (giây–phút/khung hình) | Real-time (hàng trăm FPS) |

**Kết luận:** đóng góp cốt lõi của 3D Gaussian Splatting là biến bài toán render từ **ray-tracing** (đắt) thành **rasterization** (rẻ), mà vẫn giữ tính chất vật lý mượt mà của volume rendering nhờ hình dạng Gaussian mờ dần tự nhiên ở biên.