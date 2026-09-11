# Chương 4 — Differentiable Tile Rasterizer

> Khối biến các splat 2D thành ảnh. **Toán học giữ nguyên 3DGS**; FastGS-lite chỉ đưa vào ít cặp (tile, Gaussian) hơn.
> Code: `rasterizer_impl.cu` (sort, ranges), `forward.cu` (`renderCUDA`).

## 4.1 — Đơn vị công việc: cặp (tile, Gaussian)

Ảnh chia thành lưới tile $16\times16$. Chi phí rasterize **không** tỉ lệ với $N$ mà với

$$
P=\sum_{i=1}^{N}K_i\approx N\cdot K
$$

Ba trong bốn bước dưới đây tỉ lệ với $P$.

## 4.2 — Prefix sum và sinh khoá

$$
\text{offset}_i=\sum_{j<i}K_j
$$

Mỗi cặp $(T,i)$ nhận một khoá 64-bit (`auxiliary.h:278-280`):

$$
\boxed{\ \text{key}(T,i)=\bigl(\text{id}(T)\ll32\bigr)\ \big|\ \text{bit\_cast}_{u32}(\text{depth}_i)\ }
$$

32 bit cao gom entry cùng tile; 32 bit thấp xếp theo độ sâu trong tile. `bit_cast` float→uint32 giữ đúng thứ tự chỉ khi depth dương — bảo đảm bởi cull $t_z>0.2$ (chương 3.2).

## 4.3 — Radix sort

$$
T_{\text{sort}}=O(P)=O(NK)
$$

Chỉ sort $32+\lceil\log_2(\text{số tile})\rceil$ bit thay vì 64 (ảnh 1600×1040 → 45 bit). Sau sort, `identifyTileRanges` cho $\text{ranges}[T]=[\text{start},\text{end})$ — danh sách Gaussian $\mathcal G_T$ của tile $T$ theo độ sâu tăng dần.

## 4.4 — Alpha tại pixel

Với pixel $x=(u,v)$ trong tile $T$, Gaussian $n\in\mathcal G_T$, $\Delta=x-\mu'_n$:

$$
\text{power}_n(x)=-\tfrac12\bigl(A_n\Delta_u^2+2B_n\Delta_u\Delta_v+C_n\Delta_v^2\bigr)=-\tfrac12\Delta^\top M_n\Delta
$$

$$
G_n(x)=e^{\text{power}_n(x)},\qquad
\boxed{\ \alpha_n(x)=\min\bigl(0.99,\ \alpha_n\,G_n(x)\bigr)\ }
$$

Ba cửa loại, đúng thứ tự code:

| Điều kiện | Hành động | Ý nghĩa |
|---|---|---|
| $\text{power}_n>0$ | bỏ splat | $M$ không PSD (lỗi số học) |
| $\alpha_n(x)<1/255$ | bỏ splat | dưới bước lượng tử 8-bit — cùng ngưỡng với compact box |
| $T\,(1-\alpha_n)<10^{-4}$ | dừng pixel | early termination |

$1/255$ lọc **từng splat**, $10^{-4}$ dừng **cả vòng lặp pixel** — hai hằng khác nhau, hai việc khác nhau.

## 4.5 — Alpha-blending front-to-back

$$
T_1=1,\qquad T_{n+1}=T_n\bigl(1-\alpha_n(x)\bigr)
$$

$$
\boxed{\ C(x)=\sum_{n\in\mathcal G_T}c_n\,\alpha_n(x)\,T_n\;+\;T_{\text{final}}\cdot C_{\text{bg}}\ }
$$

Ví dụ: ba Gaussian $(\text{đỏ},0.5),(\text{lục},0.4),(\text{lam},0.6)$, nền trắng:

| $n$ | $c_n$ | $\alpha_n$ | $T_n$ | đóng góp |
|---|---|---|---|---|
| 1 | (1,0,0) | 0.5 | 1.0 | (0.5, 0, 0) |
| 2 | (0,1,0) | 0.4 | 0.5 | (0, 0.2, 0) |
| 3 | (0,0,1) | 0.6 | 0.3 | (0, 0, 0.18) |

$T_4=0.12$ → $C=(0.5,0.2,0.18)+0.12\,(1,1,1)=(0.62,0.32,0.30)$. Gaussian gần nhất đóng góp nhiều nhất dù $\alpha$ không cao nhất — đó là lý do phải sort theo depth.

Mỗi tile là một CUDA block 256 thread (một thread/pixel), duyệt Gaussian theo batch 256 nạp vào shared memory. Block dừng khi **mọi** pixel đã bão hoà (`__syncthreads_count(done)==256`).

## 4.6 — Dữ liệu lưu cho backward

| Mảng | Nội dung | Dùng cho |
|---|---|---|
| `final_Ts[x]` | $T_{\text{final}}$ | gradient của số hạng background |
| `n_contrib[x]` | số Gaussian thực sự đóng góp tại pixel | backward bỏ qua splat sau early termination |
| `sampled_T`, `sampled_ar` | checkpoint mỗi 32 Gaussian: $T$ và $C^{\le n}$ | backward duyệt **xuôi** (chương 6) |
| `max_contrib[T]` | $\max_{x\in T}n_{\text{contrib}}(x)$ | cắt cả warp trong backward |

Số bucket của tile: $\#\text{bucket}_T=\lceil|\mathcal G_T|/32\rceil$. Backward khởi chạy $\sum_T\#\text{bucket}_T$ warp — thường **ít hơn** $NK$ nhờ early termination, nên $bNK$ là cận trên cho backward.

## 4.7 — Đầu ra của khối

$$
I_{\text{rend}}=\{C(x)\}_{x\in H\times W}
$$

đi sang khối **Image** (chương 5) để tính loss; cùng với `final_Ts`, `n_contrib`, checkpoint đi ngược vào **Gradient Flow** (chương 6).
