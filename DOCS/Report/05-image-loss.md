# Chương 5 — Image → Loss và Metrics

> Khối cuối của Operation Flow. Ảnh render được so với ground-truth để (a) sinh loss cho Gradient Flow và (b) chấm điểm.
> **FastGS-lite giữ nguyên loss.** Code: `train.py:100-103`, `pipeline/trainer.py:118-120`, `utils/loss_utils.py`, `pipeline/score.py`.

## 5.1 — Loss huấn luyện

$$
\boxed{\ \mathcal L=(1-\lambda)\,\mathcal L_1+\lambda\,\mathcal L_{\text{D-SSIM}}\ }
$$

$$
\mathcal L_1=\frac{1}{3HW}\sum_{x,\text{ch}}\bigl|I_{\text{rend}}(x)-I_{\text{gt}}(x)\bigr|,
\qquad
\mathcal L_{\text{D-SSIM}}=1-\text{SSIM}(I_{\text{rend}},I_{\text{gt}})
$$

| Tham số | Giá trị | Nguồn |
|---|---|---|
| $\lambda$ | $0.2$ | `arguments/__init__.py:82` |
| $\lambda$ (preset notebook) | $0.25$ | `pipeline/config.py:44` |

SSIM dùng cửa sổ Gaussian $11\times11$, $\sigma=1.5$, $C_1=0.01^2$, $C_2=0.03^2$:

$$
\text{SSIM}=\frac{(2\mu_1\mu_2+C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)}
$$

Hai implementation cho cùng công thức: `fused_ssim` (kernel CUDA, dùng trong vòng lặp train) và `utils/loss_utils.ssim` (PyTorch thuần, dùng khi chấm báo cáo). Cả hai dùng zero-padding `same`.

**Điểm cần nhớ cho mô hình chi phí:** $\mathcal L$ tính trên toàn ảnh $H\times W$ — chi phí **không phụ thuộc $N$**. Đây là số hạng $F$ ở chương 8, đặt trần Amdahl cho mọi nỗ lực tăng tốc.

## 5.2 — Không có loss nào khác

Toàn bộ loss chỉ gồm hai số hạng trên. Không có mask loss, pose loss, depth loss. Hàm `get_loss`/`compute_photometric_loss` trong `utils/fast_utils.py` chỉ phục vụ việc tính điểm số multi-view của chương 7, không nằm trong công thức loss chính.

## 5.3 — Nền ngẫu nhiên

Khi `random_background=True`, $C_{\text{bg}}\sim\mathcal U[0,1]^3$ mỗi vòng. Số hạng $T_{\text{final}}C_{\text{bg}}$ ở chương 4.5 khi đó phạt các Gaussian "trong suốt nửa vời" ở vùng nền — chúng phải hoặc đục hẳn hoặc biến mất.

## 5.4 — Metrics chấm điểm

Ba metric của cuộc thi, tính trên mỗi ảnh test rồi trung bình theo cảnh:

$$
\text{PSNR}=20\log_{10}\frac{1}{\sqrt{\text{MSE}}},\qquad
\text{MSE}=\frac{1}{3HW}\sum_{x,\text{ch}}\bigl(I_{\text{rend}}-I_{\text{gt}}\bigr)^2
$$

$$
\text{LPIPS}=\sum_{l}\frac{1}{H_lW_l}\sum_{h,w}\bigl\lVert w_l\odot(\hat\phi_l^{\text{rend}}-\hat\phi_l^{\text{gt}})_{hw}\bigr\rVert_2^2
$$

với $\hat\phi_l$ là đặc trưng lớp $l$ của mạng (AlexNet/VGG) đã chuẩn hoá theo kênh, đầu vào ảnh được map về $[-1,1]$.

Điểm tổng hợp:

$$
\boxed{\ \text{Score}=0.4\,(1-\text{LPIPS})+0.3\,\text{SSIM}+0.3\,\operatorname{clamp}\Bigl(\frac{\text{PSNR}}{30},0,1\Bigr)\ }
$$

### Hai lỗi đã sửa trong repo

| Lỗi | Trước | Sau |
|---|---|---|
| PSNR trên tensor `[3,H,W]` không batch dim | `view(shape[0], -1)` → 3 PSNR riêng theo kênh rồi trung bình → **luôn cao hơn** PSNR thật (Jensen: $\text{mean}(-\log m_c)\ge-\log\text{mean}(m_c)$) | `utils/image_utils.py` tự thêm batch dim khi input 3D |
| LPIPS nhận ảnh $[0,1]$ | hằng số `ScalingLayer` (mean $-.030/-.088/-.188$, std $.458/.448/.450$) giả định input $[-1,1]$ | `lpipsPyTorch.lpips(..., normalize=True)` map $2x-1$ trước khi vào mạng |

Số PSNR/LPIPS trong `history-train.md` được tính trước hai sửa này, không so sánh trực tiếp được với số sau sửa.

### Ba chỗ phụ thuộc bộ chấm của ban tổ chức

1. **SSIM**: Gaussian $11\times11$ (Wang 2004, code hiện tại) hay uniform $7\times7$ (`skimage` mặc định)?
2. **Thứ tự trung bình**: $\operatorname{clamp}(\overline{\text{PSNR}}/30)$ hay $\overline{\operatorname{clamp}(\text{PSNR}_i/30)}$ — khác nhau khi có ảnh $>30$ dB.
3. **Chấm trên float hay PNG 8-bit** đã lưu.

## 5.5 — Live score ≠ điểm cuối

| | Live (trong lúc train) | Cuối (báo cáo) |
|---|---|---|
| LPIPS net | AlexNet | VGG |
| Độ phân giải | `resolution=2` | full-res |
| Số view | `eval_views=6` | toàn bộ test |

Chênh lệch ~0.1 Score giữa hai cột là do thiết kế, không phải lỗi.

## 5.6 — Đầu ra của khối

$\mathcal L$ (một scalar) đi ngược vào **Gradient Flow** (chương 6). Metrics chỉ để theo dõi, không tham gia tối ưu.
