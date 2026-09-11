# Chương 8 — Tổng hợp: mô hình chi phí và ba đòn bẩy

> Gom bảy chương trước thành một công thức duy nhất, rồi chỉ ra mỗi khối của sơ đồ ảnh hưởng tới số hạng nào.
> Mô phỏng chạy được không cần GPU: `demos/fastgs_cost_model.py`.

## 8.1 — Một vòng lặp huấn luyện làm gì

Theo đúng thứ tự sơ đồ:

| # | Khối | Việc | Chi phí tỉ lệ với |
|---|---|---|---|
| 1 | Projection (ch. 3) | chiếu $N$ Gaussian, nghịch đảo conic, SH, đếm tile | $N$ |
| 2 | Rasterizer (ch. 4) | sinh khoá, sort, blend trên $P=NK$ cặp | $NK$ |
| 3 | Image (ch. 5) | $\mathcal L_1$ + SSIM trên $H\times W$ | — |
| 4 | Gradient Flow (ch. 6) | backward qua blend ($\le NK$), qua projection ($N$), Adam ($59N$ tham số) | $NK$, $N$ |
| 5 | ADC (ch. 7), định kỳ | 20 render phụ + densify/prune | (overhead) |

## 8.2 — Mô hình chi phí

$$
\boxed{\ T_{\text{iter}}=\underbrace{a\,N}_{\text{preprocess}}+\underbrace{b\,NK}_{\text{dup+sort+blend+backward}}+\underbrace{c\,N\cdot\mathbb 1[\text{Adam step}]}_{\text{optimizer}}+\underbrace{F}_{\text{loss, SSIM, IO}}\ }
$$

Ba đại lượng có thể can thiệp: $N$, $K$, $\mathbb 1[\text{Adam}]$. $F$ thì không.

## 8.3 — Ba đòn bẩy của FastGS-lite ↔ ba khối của sơ đồ

| Đòn bẩy | Khối | Cơ chế | Tỉ số |
|---|---|---|---|
| Giảm $K$ | **Projection** | compact box $t=\texttt{mult}\cdot2\ln(255\alpha)$ + lọc ellipse | $R_{\text{tile}}=K_{\text{fast}}/K_{\text{3dgs}}$ — đo được (ví dụ $9/25=0.36$) |
| Giảm $N$ | **Adaptive Density Control** | Importance/Pruning score, densify AND, prune multinomial, final prune | $R_{\text{gauss}}=N_{\text{fast}}/N_{\text{3dgs}}$ — giả định, chưa đo A/B |
| Giảm nhịp Adam | **Gradient Flow** | lịch $1\to1/32\to1/64$, SH $1/16$ | $R_{\text{adam}}=16563/59998=0.276$ — đếm chính xác |

Số hạng nặng nhất co theo **tích**:

$$
b\,N_{\text{fast}}K_{\text{fast}}=b\,N_{\text{3dgs}}K_{\text{3dgs}}\cdot R_{\text{gauss}}R_{\text{tile}}
$$

Hai đòn bẩy **nhân** nhau, không cộng.

## 8.4 — Trần Amdahl

$$
S_{\max}=\frac{T^{\text{3dgs}}_{\text{iter}}}{F}=\frac1f,\qquad f=\frac{F}{T^{\text{3dgs}}_{\text{iter}}}
$$

Nếu loss + SSIM + IO chiếm 15% thì không cơ chế nào vượt $6.7\times$. Đây là lý do các con số tăng tốc thực tế của họ 3DGS nhanh nằm ở $2$–$5\times$.

## 8.5 — Kết quả mô hình (giả định chia chi phí 15/55/15/15, overhead scoring +2%)

| Cấu hình | Tăng tốc |
|---|---|
| Cả ba đòn bẩy | **3.83×** |
| Chỉ compact box | 1.64× |
| Chỉ giảm $N$ | 2.38× |
| Chỉ Adam thưa | 1.12× |
| Trần Amdahl | 6.7× |

$1.64\times2.38\times1.12=4.37>3.83$: các đòn bẩy nhân nhau ở $bNK$ nhưng không nhân được ở $F$. Adam chỉ đóng góp 1.12× dù $R_{\text{adam}}=0.276$ vì Adam chỉ chiếm 15%. Giảm $N$ đóng góp nhiều nhất vì $N$ xuất hiện trong cả ba số hạng biến thiên — và cũng là tỉ số kém tin cậy nhất.

## 8.6 — Overhead riêng của FastGS

`compute_gaussian_score_fastgs` render thêm $10\times2=20$ ảnh mỗi lần densify. Với `densify_from_iter=500`, `densification_interval=500`, `densify_until_iter=15000`:

$$
n_{\text{densify}}=\frac{14500-1000}{500}+1=28,\qquad 28\times20=560\ \text{forward phụ}
$$

Quy đổi: cận trên $560/30000\approx+1.9\%$; sát thực tế hơn (1 vòng $\approx3$ forward) $\approx+0.6\%$. Tăng tuyến tính khi hạ `densification_interval` (mặc định 100 → 145 lần densify → $\approx+10\%$).

## 8.7 — Những gì FastGS-lite KHÔNG đổi

Toàn bộ toán render và tối ưu:

$$
G(x),\quad\Sigma=RSS^\top R^\top,\quad\Sigma'=JW\Sigma W^\top J^\top+0.3I,\quad M=\Sigma'^{-1},\quad
C=\sum c_n\alpha_nT_n,\quad\text{Adam},\quad\mathcal L=(1-\lambda)\mathcal L_1+\lambda\mathcal L_{\text{D-SSIM}}
$$

Không dòng nào của sáu thay đổi nằm trong chuỗi render. Chúng chỉ quyết định **cái gì** đi vào chuỗi đó ($N$, $K$) và **khi nào** tham số được cập nhật.

## 8.8 — Cảnh báo khi cấu hình

| Điều chỉnh | Hệ quả cần biết |
|---|---|
| `--iterations < 20000` | mất `final_prune_fastgs` (chạy $15000<t<30000$); tiết kiệm rất ít vì 15k–30k vốn rẻ |
| Rút ngắn `--iterations` | phải đồng bộ `--position_lr_max_steps`, `--densify_until_iter`, và hai ngưỡng cứng 15000/20000 trong `optimizer_step` |
| `--highfeature_lr X` | lr thật vào Adam là $X/20$ |
| `--mult 1.0` | về SnugBox, **không** về 3DGS gốc; `getRect` đã bị xoá |
| `--dense` | không chọn theo nhãn indoor/outdoor — copy dòng của cảnh giống dataset nhất trong `train_base.sh` |
| So sánh A/B với 3DGS | cùng máy, cùng dataset, cùng resolution, tối thiểu 20k vòng |
