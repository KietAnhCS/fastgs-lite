[← Phần 6](06-gradient-adam.md) · [Mục lục lời giải](00-muc-luc-loi-giai.md) · Bài toán lớn — Phần 7/8

# Phần 7 — Chương 12: Adaptive Density Control (densify **AND** prune) tại $t=5000$

> **Đầu vào nhận từ Phần 6 (Chương 11): 236 tham số đã cập nhật + thống kê gradient — xem [Phần 6](06-gradient-adam.md).**
> Cụ thể: $4$ Gaussian $\times 59$ tham số $=236$ tham số đã trải qua đúng **một** bước Adam tại $t=5000$ (chương 11), cộng bộ đếm tích luỹ gradient `accum`, `accum`$^{\text{abs}}$, `denom` (chương 6.3 / 11.3) đã cộng dồn qua 3 view của cảnh đồ chơi.
>
> Chương nguồn lý thuyết: [`12-adaptive-density-control.md`](../12-adaptive-density-control.md) (viết tắt **"ch.12"** trong toàn Phần 7 này). Mọi con số densify/prune dưới đây **chép nguyên văn** từ mục 7.1–7.8 của ch.12 (phiên bản 4-Gaussian/3-camera "cảnh đồ chơi dùng chung"), **không** dùng ví dụ 5-Gaussian/16-pixel/3-view của `3.md` (mục "12.3 Ví dụ số end-to-end" trong ch.12) — đó là một cảnh đồ chơi *khác*, chỉ minh hoạ ý tưởng, không thuộc pipeline COLMAP của Chương 16.

---

## 7.0 — Vì sao dùng lại nguyên số của ch.12, và tại sao $t=5000$ không cần biết chính xác

### 7.0.1 Cảnh này có đúng là cảnh của đề bài không?

Có. §16.1 của [Chương 16 — Đề bài](../16-bai-toan-lon-de-bai.md) định nghĩa **chính cảnh đồ chơi dùng chung** ở [§6.2](../06-initialization.md#62-cảnh-đồ-chơi-dùng-chung-cho-các-bài-kiểm-định-số-chương-6–13): 4 điểm SfM, 3 camera pinhole, ảnh $48\times32$. Đây đúng là cảnh mà mục "12.2 Kiểm định số" của ch.12 đã tính tay bằng `scripts/ch07_test.py` (numpy + scipy, seed 0). Vì vậy Phần 7 này **không tính lại** projection/render/loss — nó **thừa hưởng** các số đã công bố (μ, s, α, k, gradient, Importance, Pruning) làm input, đúng tinh thần Quy ước 1 ở §16.4: "không phát sinh số liệu mới mâu thuẫn với các chương đã xuất bản".

### 7.0.2 Sai số do Adam-update ở Phần 6 có làm lệch các quyết định densify/prune không?

Phần 6 (chương 11) cập nhật $\mu_i$, $\tilde s_i$, $\tilde\alpha_i$, $k_i$ bằng **một** bước Adam tại $t=5000$. Bước dịch chuyển lớn nhất trong một bước Adam bị chặn trên bởi learning rate của từng nhóm (chương 11.4 / ch.6.4):

$$
|\Delta\theta| \le \eta_\theta \quad\text{(vì } \bigl|\hat m/(\sqrt{\hat v}+\epsilon)\bigr|\le 1\text{ xấp xỉ, do } \hat m,\hat v \text{ cùng dấu độ lớn)}
$$

| Nhóm | $\eta$ tại $t=5000$ (xấp xỉ) | Ảnh hưởng tới quyết định ADC |
|---|---|---|
| `xyz` | $\eta_{xyz}(5000)\cdot\text{extent} \ll 1.6\times10^{-4}\cdot1.690\approx2.7\times10^{-4}$ | $\mu_i$ lệch $\lesssim 10^{-4}$ — không đổi footprint hữu hình đáng kể so với biên $\hat e>0.1$ |
| `scaling` | $\eta_s=0.005$ (không gian log) $\Rightarrow \Delta s_i \approx s_i\cdot0.005\approx0.004$–$0.006$ | so với biên $\delta\cdot\text{extent}=0.0017$ và $\max s_i\in[0.85,1.12]$: lệch $\ll$ khoảng cách tới biên (biên cách xa scale hiện tại tới $\sim500\times$) |
| `opacity` | $\eta_\alpha=0.025$ (không gian logit) | $\alpha_i$ lệch $\lesssim0.006$ quanh $0.1$ — không đổi kết luận $\alpha<0.005$ (còn cách xa) |

Vì mọi ngưỡng trong ADC ở chế độ $500<t<15000,\ t>3000$ (bảng 7.1 của ch.12) **không phụ thuộc trực tiếp vào $t$** — chỉ phụ thuộc việc $t$ có nằm trong dải hay không (`iteration < densify_until_iter`, `t > 3000` bật `size_threshold`) — và vì biên độ dịch chuyển một bước Adam nhỏ hơn khoảng cách tới mọi ngưỡng quyết định (bảng trên) từ 10 đến hơn 500 lần, **kết luận densify/prune tại $t=5000$ giống hệt kết luận đã tính trong ch.12** dù ta dùng $\mu_i,s_i,\alpha_i$ trước hay sau bước Adam của Phần 6. Phần 7 vì vậy dùng lại **nguyên văn** các giá trị đầu vào của ch.12 (bảng "Đầu vào của khối" của mục 12.2), vừa đúng số đã công bố, vừa đúng về bản chất vật lý của bài toán.

> **Kết luận formal:** với mọi $t\in(3000,15000)$, hàm quyết định densify/prune là hàm bậc thang (threshold) của $(\bar g,\bar g^{\text{abs}}, \max s, \text{Importance}, \alpha, \text{Pruning})$ — không có số hạng $t$ tường minh nào trong công thức 7.2–7.6 của ch.12. $t$ chỉ quyết định **có được gọi hay không** (lịch ở bảng 7.1). Với $t=5000$: $500<5000<15000$ ✅ và $5000>3000$ ✅ ⟹ khối chạy đúng với `size_threshold=20` bật (cả ba điều kiện của $\mathcal C$ hoạt động).

### 7.0.3 Bảng đầu vào (chép nguyên từ ch.12, mục 12.2 "Đầu vào của khối")

| $i$ | $\mu_i$ | $s_i$ (đẳng hướng, 3 trục bằng nhau) | $\tilde s_i=\log s_i$ | $\alpha_i$ | $\tilde\alpha_i=\sigma^{-1}(\alpha_i)$ | $k_{i,00}$ (R,G,B) | $q_i$ |
|---|---|---|---|---|---|---|---|
| 1 | $(0,0,0)$ | 0.8505 | $-0.161943$ | 0.1 | $-2.197225$ | $(1.063,-1.063,-1.063)$ | $(1,0,0,0)$ |
| 2 | $(0.5,0.3,0.5)$ | 0.9434 | $-0.058267$ | 0.1 | $-2.197225$ | $(-1.063,0.7090,-0.7090)$ | $(1,0,0,0)$ |
| 3 | $(-0.4,-0.2,1.0)$ | 1.1150 | $+0.108898$ | 0.1 | $-2.197225$ | $(-1.418,-0.7090,1.418)$ | $(1,0,0,0)$ |
| 4 | $(0.3,-0.5,0.2)$ | 0.8888 | $-0.117861$ | 0.1 | $-2.197225$ | $(0,0,0)$ | $(1,0,0,0)$ |

$R_i=I$ (quaternion đơn vị $\Rightarrow$ không xoay), $f_{\text{rest}}$: 45 hệ số $=0$ cho cả 4 Gaussian (SH bậc cao chưa kích hoạt — chương 6/ch.1).

$$
\bar c=(0,\ 0.1667,\ -4),\qquad \text{extent}=1.1\cdot\max_v\lVert c_v-\bar c\rVert=1.690250,\qquad \delta\,\text{extent}=0.0016902,\qquad 0.1\,\text{extent}=0.169025
$$

Đây chính là bảng đã dùng để tính projection/render (Phần 2–5, chương 7–10) và gradient (Phần 6, chương 11) của bài toán lớn — số liệu **khớp 100%** với các phần trước, không có mâu thuẫn.

---

## 7.1 — Lịch chạy tại $t=5000$ (chép công thức từ mục 7.1 ch.12)

| Việc | Điều kiện | Tại $t=5000$ | Kết luận |
|---|---|---|---|
| Tích luỹ thống kê gradient | `iteration < densify_until_iter` (mọi $t<15000$) | $5000<15000$ | ✅ đã tích luỹ (Phần 6) |
| Densify + prune (`densify_and_prune_fastgs`) | `densification_interval`, $500<t<15000$ | $500<5000<15000$ | ✅ **CHẠY** — đây là khối ta tính trong Phần 7 |
| `size_threshold` (bật cả 3 điều kiện của $\mathcal C$) | $t>3000$ | $5000>3000$ | ✅ bật |
| Opacity reset (`reset_opacity`, mỗi 3000 vòng) | $t \bmod 3000=0$ | $5000\bmod3000=2000\ne0$ | ❌ **KHÔNG** chạy ở $t=5000$ (lần kế tiếp là $t=6000$) |
| `final_prune_fastgs` | mỗi 3000 vòng, $15000<t<30000$ | $5000<15000$ | ❌ chưa tới giai đoạn này — xem mục 7.9 (minh hoạ) |

Vì `size_threshold` bật, tập ứng viên prune $\mathcal C$ dùng đủ ba điều kiện $\alpha<0.005 \vee r^{2D}>20\,\text{px} \vee \max s>0.1\,\text{extent}$ (mục 7.5 dưới đây) — đúng như ch.12 giả định "$500<t<15000,\ t>3000$".

---

## 7.2 — Importance và Pruning: bảy bước của `compute_gaussian_score_fastgs`

### 7.2.0 Trích code

```python
# utils/fast_utils.py:10-19
def sampling_cameras(my_viewpoint_stack):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''
    num_cams = 10
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    return camlist
```

```python
# utils/fast_utils.py:33-93 (rút gọn phần công thức)
def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    for view in range(len(camlist)):
        render_image = render_fastgs(...)["render"]                         # ①-embedded
        photometric_loss = compute_photometric_loss(...)                    # ⑥
        l1_loss_norm = get_loss(render_image, gt_image)                     # ①②
        metric_map = (l1_loss_norm > args.loss_thresh).int()                # ③
        render_pkg = render_fastgs(..., get_flag=True, metric_map=metric_map)
        accum_loss_counts = render_pkg["accum_metric_counts"]               # ④
        if DENSIFY:
            full_metric_counts += accum_loss_counts
        full_metric_score += photometric_loss * accum_loss_counts           # ⑦ (tích luỹ)
    pruning_score = (full_metric_score - min) / (max - min)                 # ⑦ minmax
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist),
                                      rounding_mode='floor')                 # ⑤
    return importance_score, pruning_score
```

`sampling_cameras` lấy $V=10$ camera **có hoàn lại theo pop ngẫu nhiên** từ `viewpoint_stack`. Cảnh đồ chơi của Chương 16 chỉ có **3 camera thật** (`cam01.png`, `cam02.png`, `cam03.png` — §16.1). Theo đúng quy ước mà bản thân ch.12 đã nêu ở mục 12.2 (dòng đầu file test):

> "**Khác với code thật:** $V=3$ camera (cảnh chỉ có 3) thay vì $V=10$ của `sampling_cameras`. Mọi công thức giữ nguyên."

Phần 7 **lặp lại đúng quy ước này**: dùng cả 3 camera thật của cảnh (không lấy mẫu lại, không nhân bản), $V=3$ thay cho $V=10$ trong công thức ⑤ và trong vòng lặp $\sum_{v=1}^V$ của công thức ⑦. Đây không phải một xấp xỉ tuỳ tiện — nó là cách duy nhất hợp lý khi tổng số camera của cảnh nhỏ hơn `num_cams=10`: `sampling_cameras` vốn giả định `viewpoint_stack` có nhiều hơn 10 phần tử (huấn luyện thật có hàng chục–hàng trăm ảnh), còn cảnh đồ chơi chỉ có 3.

### 7.2.1 ① Sai số màu từng pixel

$$
e_v(x)=\frac13\sum_{\text{ch}\in RGB}\bigl|I^{(v)}_{\text{rend},\text{ch}}(x)-I^{(v)}_{\text{gt},\text{ch}}(x)\bigr|
$$

Render dùng $\alpha=0.1$ (mô hình, cột 7.0.3), ground-truth dùng $\alpha=0.9$ (quy ước §16.1: "ảnh GT ... với mọi tham số như trên nhưng opacity $\alpha=0.9$ thay vì $0.1$"). Vì mô hình còn rất mờ ($\alpha$ nhỏ) so với GT ($\alpha$ lớn), sai số lớn gần như toàn ảnh — kết quả thật của script `ch07_test.py`:

| view | $\min_x e_v$ | $\max_x e_v$ | $\text{mean}_x e_v$ |
|---|---|---|---|
| 1 | 0.006232 | 0.438925 | 0.259278 |
| 2 | 0.000000 | 0.441639 | 0.222367 |
| 3 | 0.000000 | 0.440789 | 0.204733 |

### 7.2.2 ② Chuẩn hoá min–max theo ảnh

$$
\hat e_v(x)=\frac{e_v(x)-\min_xe_v}{\max_xe_v-\min_xe_v}
$$

| view | mẫu số $(\max-\min)$ | ngưỡng $e$ tương đương $\hat e_v>0.1$ |
|---|---|---|
| 1 | $0.438925-0.006232=0.432693$ | $e>0.006232+0.1\times0.432693=0.049502$ |
| 2 | $0.441639-0=0.441639$ | $e>0.044164$ |
| 3 | $0.440789-0=0.440789$ | $e>0.044079$ |

Ý nghĩa: ngưỡng $\hat e>0.1$ **tự co giãn** theo dải sai số của từng view — view "tốt" và view "tệ" đều cho ra số pixel lỗi tương đương về tỉ lệ.

### 7.2.3 ③ Mặt nạ nhị phân

$$
m_v(x)=\mathbb 1\bigl[\hat e_v(x)>\tau_{\text{loss}}\bigr],\qquad \tau_{\text{loss}}=0.1
$$

| view | số pixel bật / 1536 ($48\times32$) |
|---|---|
| 1 | **1409** |
| 2 | **1077** |
| 3 | **1042** |

### 7.2.4 ④ Đổ mặt nạ về từng Gaussian

$$
\text{counts}^{(v)}_i=\sum_{x\in\Omega^{(v)}_i}m_v(x)
$$

$\Omega^{(v)}_i$ là footprint **hữu hình** — tập pixel Gaussian $i$ thực sự đóng góp sau cả ba cửa loại: cull bằng compact box (`mult`$=0.5$), $\alpha\ge1/255$, $T\ge10^{-4}$ (đúng chỗ `forward.cu:406-408` gọi `atomicAdd(metricCount)`).

| | view 1 | view 2 | view 3 | $\sum_v$ | $\lvert\Omega\rvert$ (v1,v2,v3) |
|---|---|---|---|---|---|
| $G_1$ | 1248 | 954 | 930 | **3132** | 1248, 954, 930 |
| $G_2$ | 1181 | 1003 | 869 | **3053** | 1197, 1005, 869 |
| $G_3$ | 1245 | 978 | 977 | **3200** | 1301, 978, 990 |
| $G_4$ | 1180 | 973 | 770 | **2923** | 1193, 974, 770 |

Kiểm tra chéo view 1: $\sum_i\text{counts}^{(1)}_i=1248+1181+1245+1180=4854$ trong khi chỉ 1409 pixel bật — một pixel lỗi "tố cáo" **mọi** Gaussian đóng góp vào nó (các footprint chồng lấn nhau nhiều vì 4 điểm gần nhau so với kích thước ảnh).

### 7.2.5 ⑤ Importance score — thay số đầy đủ cho cả 4 Gaussian

$$
\boxed{\ \text{Importance}_i=\Bigl\lfloor\frac1V\sum_{v=1}^{V}\text{counts}^{(v)}_i\Bigr\rfloor\ },\qquad V=3\ \text{(quy ước §16.1/7.2.0)}
$$

**$G_1$:**
$$
\text{Importance}_1=\Bigl\lfloor\frac{3132}{3}\Bigr\rfloor=\lfloor1044.00\rfloor=1044
$$

**$G_2$:**
$$
\text{Importance}_2=\Bigl\lfloor\frac{3053}{3}\Bigr\rfloor=\lfloor1017.6\overline6\rfloor=1017
$$

**$G_3$:**
$$
\text{Importance}_3=\Bigl\lfloor\frac{3200}{3}\Bigr\rfloor=\lfloor1066.6\overline6\rfloor=1066
$$

**$G_4$:**
$$
\text{Importance}_4=\Bigl\lfloor\frac{2923}{3}\Bigr\rfloor=\lfloor974.3\overline3\rfloor=974
$$

| | $\sum_v\text{counts}$ | $/3$ | Importance (floor) | $>5$? |
|---|---|---|---|---|
| $G_1$ | 3132 | 1044.00 | **1044** | ✅ |
| $G_2$ | 3053 | 1017.67 | **1017** | ✅ |
| $G_3$ | 3200 | 1066.67 | **1066** | ✅ |
| $G_4$ | 2923 | 974.33 | **974** | ✅ |

Cả 4 Gaussian phủ gần hết ảnh và cả ảnh đều sai (mô hình $\alpha=0.1$ khác rất xa GT $\alpha=0.9$), nên Importance rất lớn — ngưỡng $5$ của công thức densify (mục 7.3) không chặn ai ở cảnh đồ chơi này (xem Bài tập 12.2/7.2 về ý nghĩa của việc này).

### 7.2.6 ⑥ Photometric loss toàn ảnh

$$
E^{(v)}_{\text{photo}}=(1-\lambda)\mathcal L^{(v)}_1+\lambda\bigl(1-\text{SSIM}^{(v)}\bigr),\qquad\lambda=0.2
$$

SSIM cài theo `utils/loss_utils.py`: cửa sổ gaussian $11\times11$, $\sigma=1.5$, zero-pad, $C_1=10^{-4}$, $C_2=9\times10^{-4}$.

**View 1:**
$$
E^{(1)}_{\text{photo}}=0.8\times0.259278+0.2\times(1-0.647773)=0.207422+0.070445=0.277867
$$

**View 2:**
$$
E^{(2)}_{\text{photo}}=0.8\times0.222367+0.2\times(1-0.702127)=0.177893+0.059575=0.237468
$$

**View 3:**
$$
E^{(3)}_{\text{photo}}=0.8\times0.204733+0.2\times(1-0.704586)=0.163786+0.059083=0.222869
$$

| view | $\mathcal L_1$ | SSIM | $0.8\mathcal L_1$ | $0.2(1-\text{SSIM})$ | $E_{\text{photo}}$ |
|---|---|---|---|---|---|
| 1 | 0.259278 | 0.647773 | 0.207422 | 0.070445 | **0.277867** |
| 2 | 0.222367 | 0.702127 | 0.177893 | 0.059575 | **0.237468** |
| 3 | 0.204733 | 0.704586 | 0.163786 | 0.059083 | **0.222869** |

### 7.2.7 ⑦ Pruning score — thay số đầy đủ cho cả 4 Gaussian

$$
\boxed{\ \text{Pruning}_i=\text{minmax}_i\Bigl(\sum_{v=1}^{V}\text{counts}^{(v)}_i\cdot E^{(v)}_{\text{photo}}\Bigr)\ }
$$

min–max chạy **qua 4 Gaussian** (khác bước ② chạy qua pixel).

**$G_1$:**
$$
\text{cnt}\cdot E = 1248\times0.277867+954\times0.237468+930\times0.222869 = 346.778+226.545+207.268=780.591
$$

**$G_2$:**
$$
1181\times0.277867+1003\times0.237468+869\times0.222869=328.161+238.181+193.673=760.015
$$

(chú ý: $\text{counts}^{(1)}_2=1181$ dùng ở bước ⑦ khác $\lvert\Omega^{(1)}_2\rvert=1197$ vì $\text{counts}$ chỉ đếm phần pixel **lỗi** ($m_v=1$) trong footprint, còn $\lvert\Omega\rvert$ đếm cả footprint — hai số này chỉ trùng nhau khi toàn bộ footprint đều là pixel lỗi.)

**$G_3$:**
$$
1245\times0.277867+978\times0.237468+977\times0.222869=345.945+232.244+217.743=795.932
$$

**$G_4$:**
$$
1180\times0.277867+973\times0.237468+770\times0.222869=327.883+231.056+171.609=730.549
$$

Thô $=(780.591,\ 760.015,\ 795.932,\ 730.549)$, $\min=730.549$ ($G_4$), $\max=795.932$ ($G_3$), mẫu số $=795.932-730.549=65.383$.

$$
\text{Pruning}_1=\frac{780.591-730.549}{65.383}=\frac{50.042}{65.383}=0.7654
$$
$$
\text{Pruning}_2=\frac{760.015-730.549}{65.383}=\frac{29.466}{65.383}=0.4507
$$
$$
\text{Pruning}_3=\frac{795.932-730.549}{65.383}=\frac{65.383}{65.383}=1.0000
$$
$$
\text{Pruning}_4=\frac{730.549-730.549}{65.383}=0.0000
$$

| | $\text{cnt}\cdot E_1$ | $\text{cnt}\cdot E_2$ | $\text{cnt}\cdot E_3$ | thô | Pruning |
|---|---|---|---|---|---|
| $G_1$ | 346.778 | 226.545 | 207.268 | 780.591 | **0.7654** |
| $G_2$ | 328.161 | 238.181 | 193.673 | 760.015 | **0.4507** |
| $G_3$ | 345.945 | 232.244 | 217.743 | 795.932 | **1.0000** |
| $G_4$ | 327.883 | 231.056 | 171.609 | 730.549 | **0.0000** |

### 7.2.8 Bảng tổng — kết quả của khối 7.2 (đầu vào cho 7.3 và 7.6)

| | $\sum_v\text{counts}$ | Importance | Pruning |
|---|---|---|---|
| $G_1$ | 3132 | **1044** | **0.7654** |
| $G_2$ | 3053 | **1017** | **0.4507** |
| $G_3$ | 3200 | **1066** | **1.0000** |
| $G_4$ | 2923 | **974** | **0.0000** |

Số này **giống hệt** bảng đã công bố ở mục 12.2 ch.12 — khớp chéo hoàn chỉnh.

---

## 7.3 — Densify: phép AND, tính cho cả 4 Gaussian

### 7.3.0 Công thức (ch.12, §7.3, `gaussian_model.py:449-462`)

```python
# scene/gaussian_model.py:442-462
grad_vars = self.xyz_gradient_accum / self.denom
grads_abs = self.xyz_gradient_accum_abs / self.denom
grad_qualifiers      = torch.norm(grad_vars, dim=-1) >= args.grad_thresh
grad_qualifiers_abs  = torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh
clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
split_qualifiers = torch.max(self.get_scaling, dim=1).values >  args.dense*extent
all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)
metric_mask = importance_score > 5
self.densify_and_clone_fastgs(metric_mask, all_clones)
self.densify_and_split_fastgs(metric_mask, all_splits)
```

$$
\boxed{\ \text{clone}_i=\bigl[\lVert\bar g_i\rVert\ge\tau_{\text{grad}}\bigr]\wedge\bigl[\max s_i\le\delta\,\text{extent}\bigr]\wedge\bigl[\text{Importance}_i>5\bigr]\ }
$$
$$
\boxed{\ \text{split}_i=\bigl[\lVert\bar g^{\text{abs}}_i\rVert\ge\tau^{\text{abs}}_{\text{grad}}\bigr]\wedge\bigl[\max s_i>\delta\,\text{extent}\bigr]\wedge\bigl[\text{Importance}_i>5\bigr]\ }
$$

| Ký hiệu | Giá trị | CLI |
|---|---|---|
| $\tau_{\text{grad}}$ | $2\times10^{-4}$ | `--grad_thresh` |
| $\tau^{\text{abs}}_{\text{grad}}$ | $1.2\times10^{-3}$ | `--grad_abs_thresh` |
| $\delta$ | $0.001$ | `--dense` |

### 7.3.1 Gradient tích luỹ $\bar g_i,\bar g^{\text{abs}}_i$ — nhận từ Phần 6

$\bar g_i=\text{accum}_i/\text{denom}_i$, $\bar g^{\text{abs}}_i=\text{accum}^{\text{abs}}_i/\text{denom}_i$ (công thức 6.3 / 11.3, `add_densification_stats`, `scene/gaussian_model.py:493-496`):

```python
# scene/gaussian_model.py:493-496
def add_densification_stats(self, viewspace_point_tensor, update_filter):
    self.xyz_gradient_accum[update_filter]     += norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1)
    self.xyz_gradient_accum_abs[update_filter] += norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1)
    self.denom[update_filter] += 1
```

Số liệu (chép nguyên từ mục 7.3 ch.12, tính bằng sai phân hữu hạn $\pm0.5$ px qua cả 3 view $\equiv 3$ lần gọi `add_densification_stats`, `denom_i=3`):

| | accum | accum$^{\text{abs}}$ | denom | $\bar g_i$ | $\bar g^{\text{abs}}_i$ |
|---|---|---|---|---|---|
| $G_1$ | 2.125e−2 | 9.550e−2 | 3 | **7.082e−3** | **3.183e−2** |
| $G_2$ | 1.779e−2 | 9.509e−2 | 3 | **5.931e−3** | **3.170e−2** |
| $G_3$ | 1.794e−2 | 9.295e−2 | 3 | **5.980e−3** | **3.098e−2** |
| $G_4$ | 1.810e−2 | 7.198e−2 | 3 | **6.035e−3** | **2.399e−2** |

Đối chiếu với gradient giải tích của Phần 6 (chương 11.2, `backward.cu`): sai lệch 1–10% do bước sai phân $0.5$ px thô (L1 không trơn tại $|\cdot|=0$), nhưng **cùng kết luận định tính**: cả 4 Gaussian vượt xa cả hai ngưỡng.

### 7.3.1bis — Truy nguyên $\bar g_i,\bar g^{\text{abs}}_i$ về từng view (cộng dồn thủ công)

Bảng dưới chép nguyên các cột $g$ (NDC, có dấu, đơn vị đã nhân $W/2=24$ theo trục $u$ và $H/2=16$ theo trục $v$) và $g^{\text{abs}}$ (trị tuyệt đối từng pixel rồi mới cộng — `backward.cu:589-597`) của mục 7.3 ch.12, rồi cộng thủ công từng bước để ra đúng cột `accum`, `accum`$^{\text{abs}}$ đã dùng ở trên — đây chính là việc `add_densification_stats` làm sau **mỗi** view (3 lần gọi $\equiv$ `denom_i`$\mathrel{+}=1$ ba lần):

**$G_1$:**

$$
\lVert g^{(v=1)}_1\rVert=4.90\times10^{-4},\quad \lVert g^{(v=2)}_1\rVert=9.87\times10^{-3},\quad \lVert g^{(v=3)}_1\rVert=1.09\times10^{-2}
$$
$$
\text{accum}_1 = 4.90\times10^{-4}+9.87\times10^{-3}+1.09\times10^{-2} = 2.126\times10^{-2}\ (\approx2.125\times10^{-2}\text{, làm tròn})
$$
$$
\lVert g^{\text{abs},(v=1)}_1\rVert=3.94\times10^{-2},\quad \lVert g^{\text{abs},(v=2)}_1\rVert=2.87\times10^{-2},\quad \lVert g^{\text{abs},(v=3)}_1\rVert=2.74\times10^{-2}
$$
$$
\text{accum}^{\text{abs}}_1 = 3.94\times10^{-2}+2.87\times10^{-2}+2.74\times10^{-2}=9.55\times10^{-2}
$$
$$
\text{denom}_1=1+1+1=3\quad\Rightarrow\quad \bar g_1=\frac{2.125\times10^{-2}}{3}=7.082\times10^{-3},\qquad \bar g^{\text{abs}}_1=\frac{9.550\times10^{-2}}{3}=3.183\times10^{-2}
$$

**$G_2$:**
$$
\text{accum}_2=1.72\times10^{-3}+3.08\times10^{-3}+1.30\times10^{-2}=1.779\times10^{-2},\qquad
\text{accum}^{\text{abs}}_2=3.75\times10^{-2}+3.26\times10^{-2}+2.51\times10^{-2}=9.509\times10^{-2}
$$
$$
\bar g_2=\frac{1.779\times10^{-2}}{3}=5.931\times10^{-3},\qquad \bar g^{\text{abs}}_2=\frac{9.509\times10^{-2}}{3}=3.170\times10^{-2}
$$

**$G_3$:**
$$
\text{accum}_3=1.74\times10^{-3}+9.96\times10^{-3}+6.24\times10^{-3}=1.794\times10^{-2},\qquad
\text{accum}^{\text{abs}}_3=3.73\times10^{-2}+2.65\times10^{-2}+2.92\times10^{-2}=9.295\times10^{-2}
$$
$$
\bar g_3=\frac{1.794\times10^{-2}}{3}=5.980\times10^{-3},\qquad \bar g^{\text{abs}}_3=\frac{9.295\times10^{-2}}{3}=3.098\times10^{-2}
$$

**$G_4$:**
$$
\text{accum}_4=3.07\times10^{-3}+5.17\times10^{-3}+9.87\times10^{-3}=1.810\times10^{-2},\qquad
\text{accum}^{\text{abs}}_4=3.02\times10^{-2}+2.47\times10^{-2}+1.71\times10^{-2}=7.198\times10^{-2}
$$
$$
\bar g_4=\frac{1.810\times10^{-2}}{3}=6.035\times10^{-3},\qquad \bar g^{\text{abs}}_4=\frac{7.198\times10^{-2}}{3}=2.399\times10^{-2}
$$

Ghi chú quan trọng: cột `accum` cộng **chuẩn** $\lVert g^{(v)}\rVert$ (đã khai căn) của từng view, **không** cộng vector $g^{(v)}$ trước rồi mới khai căn — đây chính là điều mục 6.3 (Phần 6/chương 11) nhấn mạnh: "Lấy `norm` **trước** khi cộng qua iteration — sau đó mọi số đều không âm, không còn gì triệt tiêu." Nếu cộng vector trước ($g_1=g^{(1)}_1+g^{(2)}_1+g^{(3)}_1$ rồi mới lấy chuẩn), kết quả sẽ khác (thường nhỏ hơn, vì các thành phần dấu khác nhau có thể triệt tiêu bớt) — đây chính xác là hiện tượng đã minh hoạ ở ví dụ view 1 của $G_1$ (mục 6.1/11.1): gradient có dấu $4.9\times10^{-4}$ rất nhỏ so với trị tuyệt đối $3.9\times10^{-2}$ do triệt tiêu ngay **trong một view**; còn ở đây ta đang nói về triệt tiêu **giữa các view**, một hiệu ứng riêng biệt mà quy ước "chuẩn hoá trước, cộng sau" của `add_densification_stats` cũng ngăn chặn.

### 7.3.2 Kiểm tra ngưỡng gradient — từng Gaussian

**$G_1$:**
$$
\bar g_1=7.082\times10^{-3} \ge \tau_{\text{grad}}=2\times10^{-4}\ \Rightarrow\ \text{✅ (gấp } 35.4\times\text{)}
$$
$$
\bar g^{\text{abs}}_1=3.183\times10^{-2}\ge\tau^{\text{abs}}_{\text{grad}}=1.2\times10^{-3}\ \Rightarrow\ \text{✅ (gấp } 26.5\times\text{)}
$$

**$G_2$:**
$$
\bar g_2=5.931\times10^{-3}\ge2\times10^{-4}\ \Rightarrow\ \text{✅ (gấp }29.7\times\text{)},\qquad
\bar g^{\text{abs}}_2=3.170\times10^{-2}\ge1.2\times10^{-3}\ \Rightarrow\ \text{✅ (gấp }26.4\times\text{)}
$$

**$G_3$:**
$$
\bar g_3=5.980\times10^{-3}\ge2\times10^{-4}\ \Rightarrow\ \text{✅ (gấp }29.9\times\text{)},\qquad
\bar g^{\text{abs}}_3=3.098\times10^{-2}\ge1.2\times10^{-3}\ \Rightarrow\ \text{✅ (gấp }25.8\times\text{)}
$$

**$G_4$:**
$$
\bar g_4=6.035\times10^{-3}\ge2\times10^{-4}\ \Rightarrow\ \text{✅ (gấp }30.2\times\text{)},\qquad
\bar g^{\text{abs}}_4=2.399\times10^{-2}\ge1.2\times10^{-3}\ \Rightarrow\ \text{✅ (gấp }20.0\times\text{)}
$$

$\Rightarrow$ `grad_qualifiers`$=[T,T,T,T]$, `grad_qualifiers_abs`$=[T,T,T,T]$.

### 7.3.3 Kiểm tra ngưỡng kích thước — từng Gaussian

$$
\delta\cdot\text{extent}=0.001\times1.690250=0.0016902
$$

**$G_1$:** $\max s_1=0.8505$. So sánh $0.8505 \le 0.0016902$? **Sai** $\Rightarrow$ `clone_qualifiers`$_1=$ False, `split_qualifiers`$_1=$ True ($0.8505>0.0016902$).

**$G_2$:** $\max s_2=0.9434 > 0.0016902$ $\Rightarrow$ split.

**$G_3$:** $\max s_3=1.1150 > 0.0016902$ $\Rightarrow$ split.

**$G_4$:** $\max s_4=0.8888 > 0.0016902$ $\Rightarrow$ split.

Cả 4 scale ($0.85$–$1.12$) lớn hơn ngưỡng $\delta\cdot\text{extent}=0.0017$ tới **500–660 lần** — hệ quả trực tiếp của việc 4 điểm SfM cách nhau xa so với kích thước Gaussian khởi tạo (khoảng cách 3-NN lớn ở bước KNN khởi tạo, chương 6/ch.1). Không Gaussian nào lọt vào nhánh clone.

### 7.3.4 Cổng Importance $>5$ — từng Gaussian

Từ bảng 7.2.8: Importance $=(1044,1017,1066,974)$, tất cả $\gg5$ $\Rightarrow$ `metric_mask`$=[T,T,T,T]$.

### 7.3.5 Bảng quyết định AND đầy đủ

| | $\bar g_i\ge\tau_{\text{grad}}$? | $\bar g^{\text{abs}}_i\ge\tau^{\text{abs}}_{\text{grad}}$? | $\max s_i\le\delta\,\text{extent}$? | $\max s_i>\delta\,\text{extent}$? | Importance$>5$? | clone$_i$ | split$_i$ | **Kết luận** |
|---|---|---|---|---|---|---|---|---|
| $G_1$ | ✅ | ✅ | ❌ | ✅ | ✅ | False | **True** | **SPLIT** |
| $G_2$ | ✅ | ✅ | ❌ | ✅ | ✅ | False | **True** | **SPLIT** |
| $G_3$ | ✅ | ✅ | ❌ | ✅ | ✅ | False | **True** | **SPLIT** |
| $G_4$ | ✅ | ✅ | ❌ | ✅ | ✅ | False | **True** | **SPLIT** |

`densify_and_clone_fastgs(metric_mask, all_clones)`: `selected_pts_mask = metric_mask ∧ all_clones = [T,T,T,T] ∧ [F,F,F,F] = [F,F,F,F]` $\Rightarrow$ **clone $=\varnothing$**.

`densify_and_split_fastgs(metric_mask, all_splits)`: `selected_pts_mask = metric_mask ∧ all_splits = [T,T,T,T] ∧ [T,T,T,T] = [T,T,T,T]` $\Rightarrow$ **split $=\{G_1,G_2,G_3,G_4\}$** (cả 4).

So với 3DGS gốc (không có cổng Importance): 3DGS cũng sẽ split cả 4 (vì gradient và scale đã đủ điều kiện độc lập với Importance) — ở cảnh đồ chơi này **phép AND không chặn ai** vì Importance quá lớn (mô hình $\alpha=0.1$ khác GT $\alpha=0.9$ trên toàn ảnh). Đây là điểm khác biệt so với ví dụ `3.md` (5-Gaussian) nơi phép AND **có** chặn $G_1,G_4,G_5$.

---

## 7.4 — Cơ chế split: 4 Gaussian gốc → 8 Gaussian con

### 7.4.0 Trích code

```python
# scene/gaussian_model.py:396-418
def densify_and_split_fastgs(self, metric_mask, filter, N=2):
    selected_pts_mask[:mask.shape[0]] = mask
    stds  = self.get_scaling[selected_pts_mask].repeat(N,1)          # std = s_i (không phải s_i^2)
    means = torch.zeros((stds.size(0), 3))
    samples = torch.normal(mean=means, std=stds)                     # ε^(j) ~ N(0, diag(s_i^2))
    rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N,1)
    new_scaling  = scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
    new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
    new_features_dc   = self._features_dc[selected_pts_mask].repeat(N,1,1)
    new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
    new_opacity  = self._opacity[selected_pts_mask].repeat(N,1)
    self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)
    prune_filter = torch.cat((selected_pts_mask, torch.zeros(N*selected_pts_mask.sum(), dtype=bool)))
    self.prune_points(prune_filter)          # bản gốc bị xoá ngay sau khi con được thêm vào
```

Công thức (ch.12, mục 7.3, $N=2$ con/gốc, $j=1,2$):

$$
\epsilon^{(j)}\sim\mathcal N\bigl(0,\operatorname{diag}(s_{i,1}^2,s_{i,2}^2,s_{i,3}^2)\bigr),\qquad
\mu^{(j)}=\mu_i+R(q_i)\,\epsilon^{(j)},\qquad
\tilde s^{(j)}=\log\frac{s_i}{0.8\cdot2}=\log\frac{s_i}{1.6}
$$

Vì Gaussian đẳng hướng ($s_{i,1}=s_{i,2}=s_{i,3}=s_i$), $\epsilon^{(j)}\sim\mathcal N(0, s_i^2 I_3)$ — mỗi thành phần độc lập, độ lệch chuẩn $s_i$. $R_i=I$ (quaternion đơn vị) $\Rightarrow \mu^{(j)}=\mu_i+\epsilon^{(j)}$ (không cần xoay). **rotation, opacity, f_dc, f_rest được thừa kế nguyên vẹn** từ Gaussian cha (`repeat(N,1)` không đổi giá trị, chỉ nhân bản). Bản gốc bị xoá ngay ($1\to2$, không phải $1\to3$).

### 7.4.1 Split $G_1$ ($\mu_1=(0,0,0)$, $s_1=0.8505$, seed 0)

Mẫu $\epsilon^{(1)},\epsilon^{(2)}\sim\mathcal N(0,\ 0.8505^2 I_3)$ (seed 0, kết quả thật của `np.random.normal`):

$$
\epsilon^{(1)}=(0.1069,\,-0.1124,\,0.5447),\qquad \epsilon^{(2)}=(0.0892,\,-0.4556,\,0.3075)
$$

$$
\mu^{(1)}_{G_1}=\mu_1+I\cdot\epsilon^{(1)}=(0,0,0)+(0.1069,-0.1124,0.5447)=(0.1069,\,-0.1124,\,0.5447)
$$
$$
\mu^{(2)}_{G_1}=\mu_1+I\cdot\epsilon^{(2)}=(0.0892,\,-0.4556,\,0.3075)
$$

$$
\tilde s^{(j)}_{G_1}=\log\frac{0.8505}{1.6}=\log(0.531563)=-0.631906,\qquad s^{(j)}_{G_1}=e^{-0.631906}=0.5316
$$

Cả hai con $G_1c_1,G_1c_2$ dùng chung $\tilde s=-0.6319$, $s=0.5316$; kế thừa $q=(1,0,0,0)$, $\tilde\alpha=-2.197225$ ($\alpha=0.1$), $k_{00}=(1.063,-1.063,-1.063)$, $f_{\text{rest}}=0\ (45\text{ hệ số})$.

### 7.4.2 Split $G_2$ ($\mu_2=(0.5,0.3,0.5)$, $s_2=0.9434$)

$$
\epsilon^{(1)}=(1.2302,\,0.8935,\,-0.6639),\qquad \epsilon^{(2)}=(-1.1938,\,-0.5880,\,0.0390)
$$
$$
\mu^{(1)}_{G_2}=(0.5,0.3,0.5)+(1.2302,0.8935,-0.6639)=(1.7302,\,1.1935,\,-0.1639)
$$
$$
\mu^{(2)}_{G_2}=(0.5,0.3,0.5)+(-1.1938,-0.5880,0.0390)=(-0.6938,\,-0.2880,\,0.5390)
$$
$$
\tilde s^{(j)}_{G_2}=\log\frac{0.9434}{1.6}=\log(0.589625)=-0.528292,\qquad s^{(j)}_{G_2}=0.5896
$$

Kế thừa $q=(1,0,0,0)$, $\tilde\alpha=-2.197225$, $k_{00}=(-1.063,0.7090,-0.7090)$, $f_{\text{rest}}=0$.

### 7.4.3 Split $G_3$ ($\mu_3=(-0.4,-0.2,1.0)$, $s_3=1.1150$)

$$
\epsilon^{(1)}=(-2.5925,\,-0.2440,\,-1.3893),\qquad \epsilon^{(2)}=(-0.8165,\,-0.6069,\,-0.3527)
$$
$$
\mu^{(1)}_{G_3}=(-0.4,-0.2,1.0)+(-2.5925,-0.2440,-1.3893)=(-2.9925,\,-0.4440,\,-0.3893)
$$
$$
\mu^{(2)}_{G_3}=(-0.4,-0.2,1.0)+(-0.8165,-0.6069,-0.3527)=(-1.2165,\,-0.8069,\,0.6473)
$$
$$
\tilde s^{(j)}_{G_3}=\log\frac{1.1150}{1.6}=\log(0.696875)=-0.361091,\qquad s^{(j)}_{G_3}=0.6969
$$

Kế thừa $q=(1,0,0,0)$, $\tilde\alpha=-2.197225$, $k_{00}=(-1.418,-0.7090,1.418)$, $f_{\text{rest}}=0$.

### 7.4.4 Split $G_4$ ($\mu_4=(0.3,-0.5,0.2)$, $s_4=0.8888$)

$$
\epsilon^{(1)}=(0.3659,\,0.9266,\,-0.1142),\qquad \epsilon^{(2)}=(1.2145,\,-0.5912,\,0.3124)
$$
$$
\mu^{(1)}_{G_4}=(0.3,-0.5,0.2)+(0.3659,0.9266,-0.1142)=(0.6659,\,0.4266,\,0.0858)
$$
$$
\mu^{(2)}_{G_4}=(0.3,-0.5,0.2)+(1.2145,-0.5912,0.3124)=(1.5145,\,-1.0912,\,0.5124)
$$
$$
\tilde s^{(j)}_{G_4}=\log\frac{0.8888}{1.6}=\log(0.5555)=-0.587855,\qquad s^{(j)}_{G_4}=0.5555
$$

Kế thừa $q=(1,0,0,0)$, $\tilde\alpha=-2.197225$, $k_{00}=(0,0,0)$, $f_{\text{rest}}=0$.

### 7.4.5 Bảng tổng — 8 Gaussian con (theo đúng thứ tự `repeat(N,1)` của code)

`repeat(2,1)` trên tensor $(4,\cdot)$ xếp thành $(8,\cdot)$ theo khối: bản sao thứ nhất của cả 4 Gaussian trước, rồi bản sao thứ hai — tức thứ tự thật trong bộ nhớ là $[G_1c_1,G_2c_1,G_3c_1,G_4c_1,\,G_1c_2,G_2c_2,G_3c_2,G_4c_2]$ (chỉ số $1..8$):

| # | tên | $\mu$ | $\tilde s$ | $s=e^{\tilde s}$ | $\tilde\alpha$ | $\alpha$ | $q$ | $k_{00}$ (R,G,B) |
|---|---|---|---|---|---|---|---|---|
| 1 | $G_1c_1$ | $(0.1069,-0.1124,0.5447)$ | $-0.6319$ | $0.5316$ | $-2.197225$ | 0.1 | $(1,0,0,0)$ | $(1.063,-1.063,-1.063)$ |
| 2 | $G_2c_1$ | $(1.7302,1.1935,-0.1639)$ | $-0.5283$ | $0.5896$ | $-2.197225$ | 0.1 | $(1,0,0,0)$ | $(-1.063,0.7090,-0.7090)$ |
| 3 | $G_3c_1$ | $(-2.9925,-0.4440,-0.3893)$ | $-0.3611$ | $0.6969$ | $-2.197225$ | 0.1 | $(1,0,0,0)$ | $(-1.418,-0.7090,1.418)$ |
| 4 | $G_4c_1$ | $(0.6659,0.4266,0.0858)$ | $-0.5879$ | $0.5555$ | $-2.197225$ | 0.1 | $(1,0,0,0)$ | $(0,0,0)$ |
| 5 | $G_1c_2$ | $(0.0892,-0.4556,0.3075)$ | $-0.6319$ | $0.5316$ | $-2.197225$ | 0.1 | $(1,0,0,0)$ | $(1.063,-1.063,-1.063)$ |
| 6 | $G_2c_2$ | $(-0.6938,-0.2880,0.5390)$ | $-0.5283$ | $0.5896$ | $-2.197225$ | 0.1 | $(1,0,0,0)$ | $(-1.063,0.7090,-0.7090)$ |
| 7 | $G_3c_2$ | $(-1.2165,-0.8069,0.6473)$ | $-0.3611$ | $0.6969$ | $-2.197225$ | 0.1 | $(1,0,0,0)$ | $(-1.418,-0.7090,1.418)$ |
| 8 | $G_4c_2$ | $(1.5145,-1.0912,0.5124)$ | $-0.5879$ | $0.5555$ | $-2.197225$ | 0.1 | $(1,0,0,0)$ | $(0,0,0)$ |

$f_{\text{rest}}$: 45 hệ số $=0$ cho cả 8 con (kế thừa nguyên từ cha, chưa kích hoạt SH bậc cao). `max_radii2D` của mọi con $=0$ (chưa được render lần nào — thuộc tính theo dõi màn hình reset về $0$ khi mới thêm vào, `densification_postfix`).

Kiểm tra: $s^{(j)}=s_i/1.6$ cho cả 4 gốc nằm trong $[0.5316,\ 0.6969]\subset[0.53,\ 0.70]$ — **khớp đúng** khoảng đã công bố ở ch.12 ("max s = s/1.6 ∈ [0.53,0.70]").

Đúng như quy ước 7.8 của ch.12: mỗi split ròng $N\mathrel{+}=1$ (2 con $-$ 1 gốc bị xoá). $N: 4\to8$ sau bước split (trước khi prune chạy).

---

## 7.5 — Tập ứng viên prune $\mathcal C$ trên quần thể 8 Gaussian con

### 7.5.0 Công thức

$$
\mathcal C=\bigl\{i:\ \alpha_i<0.005\ \vee\ r^{2D}_i>20\text{ px}\ \vee\ \max s_i>0.1\,\text{extent}\bigr\}
$$

Hai điều kiện sau chỉ bật khi $t>3000$ (`size_threshold` không `None`); ở $t=5000$ cả ba đều bật (mục 7.1). $0.1\cdot\text{extent}=0.169025$.

```python
# scene/gaussian_model.py:464-468
prune_mask = (self.get_opacity < min_opacity).squeeze()
if max_screen_size:
    big_points_vs = self.max_radii2D > max_screen_size
    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
```

Điểm mấu chốt: **hàm này chạy trên quần thể sau split** (đã có 8 Gaussian con, 4 bản gốc đã bị xoá) — không phải trên 4 Gaussian gốc. `min_opacity=0.005`, `max_screen_size=20`.

### 7.5.1 Kiểm tra từng con

| # | tên | $\alpha_i$ | $\alpha_i<0.005$? | $r^{2D}_i$ (`max_radii2D`) | $r^{2D}_i>20$? | $\max s_i=s^{(j)}$ | $\max s_i>0.169025$? | $\in\mathcal C$? |
|---|---|---|---|---|---|---|---|---|
| 1 | $G_1c_1$ | 0.1 | ❌ | 0 (mới, chưa render) | ❌ | 0.5316 | ✅ | **✅** |
| 2 | $G_2c_1$ | 0.1 | ❌ | 0 | ❌ | 0.5896 | ✅ | **✅** |
| 3 | $G_3c_1$ | 0.1 | ❌ | 0 | ❌ | 0.6969 | ✅ | **✅** |
| 4 | $G_4c_1$ | 0.1 | ❌ | 0 | ❌ | 0.5555 | ✅ | **✅** |
| 5 | $G_1c_2$ | 0.1 | ❌ | 0 | ❌ | 0.5316 | ✅ | **✅** |
| 6 | $G_2c_2$ | 0.1 | ❌ | 0 | ❌ | 0.5896 | ✅ | **✅** |
| 7 | $G_3c_2$ | 0.1 | ❌ | 0 | ❌ | 0.6969 | ✅ | **✅** |
| 8 | $G_4c_2$ | 0.1 | ❌ | 0 | ❌ | 0.5555 | ✅ | **✅** |

Không con nào có $\alpha<0.005$ (tất cả kế thừa $\alpha=0.1$) và không con nào có $r^{2D}>20$ (chưa từng được render nên `max_radii2D`$=0$ mặc định) — nhưng **mọi** con đều có $\max s^{(j)}=s_i/1.6\in[0.53,0.70]$ lớn hơn hẳn $0.1\cdot\text{extent}=0.169$ (gấp $3.1$–$4.1$ lần), vì split chỉ chia scale cho $1.6$ trong khi ngưỡng $0.1\cdot\text{extent}$ rất nhỏ so với kích thước Gaussian ban đầu ($\approx0.85$–$1.12$).

$$
\boxed{\ \lvert\mathcal C\rvert=8\ }
$$

$$
\text{remove\_budget}=\lfloor0.5\times8\rfloor=\boxed{4}
$$

---

## 7.6 — Pruning-score, trọng số multinomial, và tập xoá $\mathcal S$

### 7.6.0 Công thức và điểm mấu chốt về chỉ số

$$
w_i=\frac{1}{10^{-6}+(1-\text{Pruning}_i)},\qquad
\mathcal S\sim\text{Multinomial}\bigl(w,\ \text{remove\_budget}=4,\ \text{không hoàn lại}\bigr)
$$

```python
# scene/gaussian_model.py:470-483
scores = 1 - pruning_score
remove_budget = int(0.5 * to_remove)         # = 4
if remove_budget:
    n_init_points = self.get_xyz.shape[0]                     # = 8 (đã split)
    padded_importance = torch.zeros((n_init_points,))
    padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())   # scores.shape[0] = 4!
    sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
    selected_pts_mask[sampled_indices] = True
    final_prune = torch.logical_and(prune_mask, selected_pts_mask)
    self.prune_points(final_prune)
```

**Đây là chỗ dễ hiểu sai nhất trong toàn bộ chương 12**: `pruning_score` (và do đó `scores = 1 - pruning_score`) được `compute_gaussian_score_fastgs` tính **trước khi** `densify_and_prune_fastgs` chạy — tức tính trên **4 Gaussian gốc** ($G_1..G_4$, mục 7.2), **không phải** trên 8 Gaussian con. Vậy `scores.shape[0] = 4`, trong khi `n_init_points = 8` (đã split). Dòng `padded_importance[:scores.shape[0]] = ...` chỉ gán 4 phần tử **đầu tiên** của `padded_importance` (kích thước 8) bằng trọng số của 4 Gaussian gốc — **theo đúng chỉ số**, không đánh chỉ số lại theo Gaussian con nào tương ứng. 4 phần tử còn lại ($5..8$) giữ nguyên $0$ (khởi tạo `torch.zeros`).

Vì thứ tự quần thể sau split là $[G_1c_1,G_2c_1,G_3c_1,G_4c_1,\,G_1c_2,G_2c_2,G_3c_2,G_4c_2]$ (mục 7.4.5), 4 trọng số dương rơi đúng vào **bản sao thứ nhất** của mỗi Gaussian gốc ($G_1c_1,G_2c_1,G_3c_1,G_4c_1$), còn bản sao thứ hai ($G_1c_2,\dots,G_4c_2$) có trọng số $0$ — **miễn nhiễm với multinomial**, bất kể Pruning gốc của chúng là bao nhiêu.

### 7.6.1 Tính $w_i$ từ Pruning score của 4 Gaussian gốc (mục 7.2.8)

$$
w_1=\frac{1}{10^{-6}+(1-0.7654)}=\frac{1}{10^{-6}+0.2346}=\frac{1}{0.234601}=4.2621
$$
$$
w_2=\frac{1}{10^{-6}+(1-0.4507)}=\frac{1}{10^{-6}+0.5493}=\frac{1}{0.549301}=1.8204
$$
$$
w_3=\frac{1}{10^{-6}+(1-1.0000)}=\frac{1}{10^{-6}+0}=\frac{1}{10^{-6}}=1\,000\,000
$$
$$
w_4=\frac{1}{10^{-6}+(1-0.0000)}=\frac{1}{10^{-6}+1}=\frac{1}{1.000001}=0.999999\approx1.0000
$$

$$
\text{padded\_importance}=(w_1,w_2,w_3,w_4,0,0,0,0)=(4.2621,\ 1.8204,\ 1\,000\,000,\ 1.0000,\ 0,0,0,0)
$$

### 7.6.2 Xác suất chọn $p_i=w_i/\sum w$

$$
\sum w = 4.2621+1.8204+1\,000\,000+1.0000+0+0+0+0 = 1\,000\,007.0825
$$

$$
p_1=\frac{4.2621}{1\,000\,007.08}=4.262\times10^{-6},\qquad
p_2=\frac{1.8204}{1\,000\,007.08}=1.820\times10^{-6}
$$
$$
p_3=\frac{1\,000\,000}{1\,000\,007.08}=0.999993,\qquad
p_4=\frac{1.0000}{1\,000\,007.08}=1.000\times10^{-6}
$$
$$
p_5=p_6=p_7=p_8=0
$$

Ba ý nghĩa từ mục 7.4 ch.12, kiểm chứng bằng số cụ thể ở đây:

1. **Weight phân kỳ khi Pruning$\to1$**: $w_3=10^6$ so với $w_4=1.0$ (Pruning$=0$) — chênh lệch $10^6$ lần chỉ vì Pruning $1.0000$ so với $0.0000$. Về thực chất, $G_3c_1$ **chắc chắn** bị chọn ở lượt rút đầu tiên ($p_3=0.999993\approx1$).
2. **$\mathcal C$ không xoá ai** — nó chỉ là **ứng viên**: cả 8 Gaussian con đều $\in\mathcal C$ (mục 7.5), nhưng chỉ 4 bị rút bởi multinomial.
3. **Lấy mẫu trên cả $n\_init\_points=8$**, không phải chỉ trên $\mathcal C$ hay chỉ trên 4 Gaussian còn trọng số dương — nhưng vì đúng 4 phần tử có trọng số dương và budget $=4$, kết quả **tất định**: multinomial không hoàn lại buộc phải rút hết 4 phần tử dương đó (4 phần tử còn lại có $p=0$, không thể được rút).

### 7.6.3 Tập rút $\mathcal S$ và tập xoá cuối

$$
\mathcal S=\{1,2,3,4\}\quad(\text{chỉ số trong quần thể 8 Gaussian con})=\{G_1c_1,G_2c_1,G_3c_1,G_4c_1\}
$$

$$
\text{final\_prune}=\mathcal C\cap\mathcal S=\{1,2,3,4\}\cap\{1,2,3,4\}=\{1,2,3,4\}
$$

(vì $\mathcal C=\{1,\dots,8\}$ toàn bộ quần thể — mục 7.5.1 — nên giao với $\mathcal S$ đơn giản bằng $\mathcal S$.)

$$
\boxed{\ \text{Xoá: } G_1c_1,\ G_2c_1,\ G_3c_1,\ G_4c_1\ }
$$

$$
\boxed{\ N=8-4=4\ }
$$

Kết quả này (seed 0) **tất định về mặt tập hợp** — không phụ thuộc seed thật sự, vì đúng 4 trọng số dương và budget $=4$ luôn buộc cả 4 phần tử dương phải được rút hết (chỉ thứ tự rút giữa chúng phụ thuộc seed, không ảnh hưởng tập kết quả). Đây **khớp đúng** con số đã công bố ở ch.12 mục "Thứ tự thật trong `densify_and_prune_fastgs`": *"xoá $G_1c_1,G_2c_1,G_3c_1,G_4c_1$ → $N=8-4=4$"*.

### 7.6.4 Đối chiếu với công thức lý thuyết đơn giản (mục 7.8 ch.12)

Công thức lý thuyết không kể tới việc split chạy trước và đánh số lại:

$$
N\leftarrow N+|\text{clone}|+|\text{split}|-|\text{xoá}| = 4+0+4-2=6
$$

(giả định xoá theo cách tính trực tiếp trên 4 Gaussian gốc, budget $=\lfloor0.5\times4\rfloor=2$, $\mathcal S=\{1,3\}=\{G_1,G_3\}$ — bảng chi tiết ở mục 7.6.5 ch.12). Đây **không phải** kết quả thật thi hành bởi code, chỉ là công thức tổng quát $N\leftarrow N+|\text{clone}|+|\text{split}|-|\text{xoá}|$ minh hoạ ý nghĩa cấu trúc. Kết quả **thật sự** theo đúng thứ tự thực thi của `densify_and_prune_fastgs` (split trước → tính $\mathcal C$, $w$, $\mathcal S$ trên quần thể 8 → xoá) là $N=4$ như mục 7.6.3. Phần 7 này (bám theo pipeline code thật của Chương 16) dùng $N=4$ làm quần thể chính thức chuyển giao cho Phần 8.

---

## 7.7 — Ép opacity sau densify

### 7.7.0 Công thức

$$
\tilde\alpha_i\leftarrow\sigma^{-1}\bigl(\min(\alpha_i,\,0.8)\bigr)
$$

```python
# scene/gaussian_model.py:485-487 (chạy vô điều kiện ở cuối densify_and_prune_fastgs)
opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
self._opacity = optimizable_tensors["opacity"]
```

Áp dụng cho 4 Gaussian còn lại sau prune ($G_1c_2,G_2c_2,G_3c_2,G_4c_2$), mỗi cái $\alpha=0.1$:

$$
\min(0.1,\ 0.8)=0.1\quad\Rightarrow\quad \tilde\alpha\leftarrow\sigma^{-1}(0.1)=\log\frac{0.1}{0.9}=-2.197225\quad(\text{không đổi})
$$

Vì $\alpha=0.1<0.8$, phép $\min$ không có tác dụng — opacity giữ nguyên giá trị đã kế thừa từ Gaussian cha. Thao tác này **luôn** xoá trạng thái Adam $(m,v)$ của nhóm `opacity` về $(0,0)$ (nhưng giữ nguyên bộ đếm `step`) — không ảnh hưởng tới giá trị $\tilde\alpha$ đã tính ở trên, chỉ ảnh hưởng tới bước Adam **tiếp theo** (không phải bước tại $t=5000$, đã chạy xong ở Phần 6).

### 7.7.1 Reset opacity định kỳ — có chạy ở $t=5000$ không?

$$
\text{reset\_opacity: }\ \tilde\alpha_i\leftarrow\sigma^{-1}\bigl(\min(\alpha_i,\,0.01)\bigr),\qquad\text{mỗi 3000 vòng}
$$

$$
5000 \bmod 3000 = 2000 \ne 0 \quad\Rightarrow\quad \boxed{\text{KHÔNG chạy ở } t=5000}
$$

(Lần reset gần nhất trước $t=5000$ là $t=3000$; lần kế tiếp là $t=6000$.) Nếu **có** chạy (minh hoạ công thức, không áp dụng ở bước này): $\sigma^{-1}(\min(0.1,0.01))=\sigma^{-1}(0.01)=\log\frac{0.01}{0.99}=-4.595120$.

---

## 7.8 — Quần thể thật sự chuyển giao cho Phần 8: $\mathcal G_1$, $N=4$

Đây là kết quả **duy nhất và chính thức** của khối densify+prune tại $t=5000$ (không lẫn với các con số minh hoạ ở mục 7.6.4 hay mục 7.9 dưới đây).

### 7.8.1 Bảng tóm tắt theo nhóm tham số

**$\mu$ (vị trí, 3 số/Gaussian):**

| Gaussian | $\mu_x$ | $\mu_y$ | $\mu_z$ |
|---|---|---|---|
| $\mathcal G_{1,1}=G_1c_2$ | 0.0892 | −0.4556 | 0.3075 |
| $\mathcal G_{1,2}=G_2c_2$ | −0.6938 | −0.2880 | 0.5390 |
| $\mathcal G_{1,3}=G_3c_2$ | −1.2165 | −0.8069 | 0.6473 |
| $\mathcal G_{1,4}=G_4c_2$ | 1.5145 | −1.0912 | 0.5124 |

**$\tilde q$ (quaternion thô, 4 số/Gaussian) — cả 4 kế thừa nguyên vẹn:**

| Gaussian | $q_w$ | $q_x$ | $q_y$ | $q_z$ |
|---|---|---|---|---|
| $\mathcal G_{1,1..4}$ | 1 | 0 | 0 | 0 |

**$\tilde s$ và $s=e^{\tilde s}$ (scale, 3 trục bằng nhau — đẳng hướng):**

| Gaussian | $\tilde s$ (mỗi trục) | $s$ (mỗi trục) |
|---|---|---|
| $\mathcal G_{1,1}=G_1c_2$ | −0.631906 | 0.5316 |
| $\mathcal G_{1,2}=G_2c_2$ | −0.528292 | 0.5896 |
| $\mathcal G_{1,3}=G_3c_2$ | −0.361091 | 0.6969 |
| $\mathcal G_{1,4}=G_4c_2$ | −0.587855 | 0.5555 |

**$\tilde\alpha$ và $\alpha=\sigma(\tilde\alpha)$ (opacity, 1 số/Gaussian, sau ép opacity mục 7.7):**

| Gaussian | $\tilde\alpha$ | $\alpha$ |
|---|---|---|
| $\mathcal G_{1,1..4}$ | −2.197225 | 0.1 |

**$k_{00}$ / $f_{\text{dc}}$ (SH bậc 0, 3 số/Gaussian — R,G,B):**

| Gaussian | $f_{\text{dc},0}$ (R) | $f_{\text{dc},1}$ (G) | $f_{\text{dc},2}$ (B) |
|---|---|---|---|
| $\mathcal G_{1,1}=G_1c_2$ | 1.063 | −1.063 | −1.063 |
| $\mathcal G_{1,2}=G_2c_2$ | −1.063 | 0.7090 | −0.7090 |
| $\mathcal G_{1,3}=G_3c_2$ | −1.418 | −0.7090 | 1.418 |
| $\mathcal G_{1,4}=G_4c_2$ | 0 | 0 | 0 |

**$f_{\text{rest}}$ (SH bậc 1–3, 45 số/Gaussian) — cả 4 Gaussian: 45 số $0$** (SH bậc cao chưa được kích hoạt trong lịch $D(t)$ — chương 6/ch.1 §1.3; kích hoạt dần theo `oneupSHdegree` mỗi 1000 vòng, độc lập với ADC).

### 7.8.2 Đếm tham số

$$
4\ \text{Gaussian}\times(3_\mu+4_q+3_s+1_\alpha+3_{f_{dc}}+45_{f_{\text{rest}}})=4\times59=236\ \text{tham số}
$$

Đúng $236$ tham số — **giống số lượng của $\mathcal G_0$** ($4\times59$) dù $N$ không đổi ($4\to4$) — chỉ là quần thể **khác** (8 con sinh ra rồi 4 bị xoá, còn lại 4 con khác hẳn 4 cha về $\mu,\tilde s$ nhưng giữ nguyên $\tilde\alpha,q,k,f_{\text{rest}}$).

### 7.8.3 Bảng tổng hợp cuối — $\mathcal G_1$ đầy đủ (dùng để ghi `.ply` ở Phần 8)

| $i$ | tên gốc | $\mu$ | $q$ | $s$ (đẳng hướng) | $\alpha$ | $k_{00}$ (R,G,B) | $f_{\text{rest}}$ |
|---|---|---|---|---|---|---|---|
| 1 | $G_1c_2$ | $(0.0892,-0.4556,0.3075)$ | $(1,0,0,0)$ | $0.5316$ | $0.1$ | $(1.063,-1.063,-1.063)$ | $0\times45$ |
| 2 | $G_2c_2$ | $(-0.6938,-0.2880,0.5390)$ | $(1,0,0,0)$ | $0.5896$ | $0.1$ | $(-1.063,0.7090,-0.7090)$ | $0\times45$ |
| 3 | $G_3c_2$ | $(-1.2165,-0.8069,0.6473)$ | $(1,0,0,0)$ | $0.6969$ | $0.1$ | $(-1.418,-0.7090,1.418)$ | $0\times45$ |
| 4 | $G_4c_2$ | $(1.5145,-1.0912,0.5124)$ | $(1,0,0,0)$ | $0.5555$ | $0.1$ | $(0,0,0)$ | $0\times45$ |

$$
N\leftarrow N+|\text{clone}|+|\text{split}|-|\text{xoá}|,\quad\text{diễn dịch trên quần thể thật (mục 7.6):}\quad 4\ \xrightarrow{\text{split }\times4}\ 8\ \xrightarrow{\text{xoá }\times4}\ \boxed{N=4}
$$

`max_radii2D` của cả 4 Gaussian còn lại reset về $0$ (chưa render lại — sẽ được cập nhật ở lần forward pass kế tiếp, không thuộc phạm vi Phần 7). Adam state $(m,v)$ của nhóm `opacity` bị xoá về $(0,0)$ cho cả 4 (do bước ép opacity mục 7.7), state của các nhóm khác giữ nguyên các slot tương ứng với Gaussian con thứ hai (được `cat_tensors_to_optimizer` sao chép từ cha khi split, rồi `_prune_optimizer` cắt bỏ 4 slot bị xoá).

---

## 7.9 — Minh hoạ thêm cho giai đoạn sau: `final_prune_fastgs` ($15000<t<30000$)

> **Lưu ý quan trọng:** mục này là **minh hoạ thêm cho giai đoạn sau, không phải kết quả tại $t=5000$.** Bảng lịch chạy ở mục 7.1 đã xác nhận $t=5000<15000$ nên `final_prune_fastgs` **chưa** được gọi trong bài toán lớn ở bước (7) — quần thể chính thức chuyển cho Phần 8 vẫn là $N=4$ của mục 7.8, **không** bị trừ thêm gì ở đây.

### 7.9.1 Vì sao trình bày minh hoạ này

ch.12 (§7.6, và mục "12.2 Kiểm định số" §7.6) tự thân trình bày thêm một lần gọi giả định ở giai đoạn $15000<t<30000$ để minh hoạ cơ chế `final_prune_fastgs` — cơ chế mới hoàn toàn so với 3DGS gốc, không nằm trong `densify_and_prune_fastgs`. Để giữ **đầy đủ nội dung** của chương 12 (yêu cầu đề bài Phần 7), ta chép lại minh hoạ này **nguyên trạng** như ch.12 đã tính — áp dụng lên **4 Gaussian gốc $G_1$–$G_4$** (dùng đúng Pruning score đã tính ở mục 7.2.7, **không phải** 4 Gaussian con của mục 7.8, vì ch.12 không cung cấp Pruning score mới cho quần thể con — việc đó đòi hỏi chạy lại toàn bộ khối 7.2 ở một forward pass mới tại $t\approx15000+$, ngoài phạm vi số liệu đã công bố).

### 7.9.2 Công thức

```python
# scene/gaussian_model.py:498-505
def final_prune_fastgs(self, min_opacity, pruning_score=None):
    prune_mask  = (self.get_opacity < min_opacity).squeeze()
    scores_mask = pruning_score > 0.9
    final_prune = torch.logical_or(prune_mask, scores_mask)
    self.prune_points(final_prune)
```

$$
\boxed{\ \text{final\_prune}_i=\bigl[\alpha_i<0.1\bigr]\ \vee\ \bigl[\text{Pruning}_i>0.9\bigr]\ }
$$

(ở đây `min_opacity` truyền vào là $0.1$ — khác `min_opacity=0.005` dùng trong `densify_and_prune_fastgs` — đọc từ `train.py:153-158`.)

### 7.9.3 Thay số cho $G_1$–$G_4$ (Pruning score của mục 7.2.7)

| | $\alpha$ | $\alpha<0.1$? | Pruning | Pruning$>0.9$? | **Kết quả** |
|---|---|---|---|---|---|
| $G_1$ | 0.1 | ❌ (so sánh **chặt**, $0.1\not<0.1$) | 0.7654 | ❌ | giữ |
| $G_2$ | 0.1 | ❌ | 0.4507 | ❌ | giữ |
| $G_3$ | 0.1 | ❌ | **1.0000** | ✅ | 🗑️ **XOÁ** |
| $G_4$ | 0.1 | ❌ | 0.0000 | ❌ | giữ |

$\alpha=0.1$ nằm **đúng trên biên**: `get_opacity < min_opacity` (`0.1 < 0.1`) là so sánh chặt nên **không** ai bị xoá theo điều kiện thứ nhất — chỉ cần opacity học lệch xuống $0.0999$ (rất dễ xảy ra qua nhiều bước Adam) là cả 4 sẽ bị xoá theo điều kiện này. Duy nhất $G_3$ bị xoá, qua điều kiện thứ hai: $\text{Pruning}_3=1.0000>0.9$.

### 7.9.4 "Nghịch lý biểu kiến"

$G_3$ vừa là Gaussian **duy nhất** kích hoạt nhánh split đầu tiên với Importance cao nhất ($1066$, mục 7.2.5 — thực ra cả 4 đều split, nhưng $G_3$ có Importance cao nhất), vừa là Gaussian **duy nhất** bị xoá ở tỉa cuối ($\text{Pruning}=1.0$, cao nhất). Đây **không phải mâu thuẫn thiết kế**: hai chỉ số đo hai thứ khác nhau — Importance đo "còn nhiều pixel lỗi ở vùng nó phủ" (lý do để **đặt cược** thêm chi tiết bằng split), còn Pruning đo "so với các Gaussian khác, phần đóng góp của nó vào loss vẫn tệ nhất sau khi đã cân bằng theo `counts`" (lý do để **thanh lý** khoản cược nếu ở vòng lặp sau nó vẫn không cứu vãn được chất lượng). $G_3$ minh hoạ đúng chu trình "đặt cược giai đoạn đầu, thanh lý giai đoạn cuối nếu không hiệu quả".

---

## 7.8bis — Vector tham số đầy đủ 59 chiều (dạng liệt kê, sẵn sàng cho `.ply`)

Mục này liệt kê **tường minh từng số** (không viết tắt "$0\times45$") của cả 4 Gaussian trong $\mathcal G_1$, theo đúng thứ tự 59 tham số dùng xuyên suốt sách: $[\mu_x,\mu_y,\mu_z,\ q_w,q_x,q_y,q_z,\ \tilde s_1,\tilde s_2,\tilde s_3,\ \tilde\alpha,\ f_{dc,0},f_{dc,1},f_{dc,2},\ f_{rest,0..44}]$. Đây chính là dữ liệu thô Phần 8 cần để ghi từng dòng vertex của `point_cloud.ply` (sau khi chèn thêm 3 số $n_x=n_y=n_z=0$ để đủ 62 cột).

**$\mathcal G_{1,1}$ ($=G_1c_2$):**

```
mu        = [ 0.0892, -0.4556,  0.3075]
rotation  = [ 1.0,  0.0,  0.0,  0.0]
scaling   = [-0.631906, -0.631906, -0.631906]        # tilde_s, 3 trục bằng nhau (đẳng hướng)
opacity   = [-2.197225]                                # tilde_alpha, tương đương alpha = 0.1
f_dc      = [ 1.063, -1.063, -1.063]
f_rest    = [0.0]*45   # 45 hệ số SH bậc 1-3, tất cả bằng 0 (chưa kích hoạt)
```

**$\mathcal G_{1,2}$ ($=G_2c_2$):**

```
mu        = [-0.6938, -0.2880,  0.5390]
rotation  = [ 1.0,  0.0,  0.0,  0.0]
scaling   = [-0.528292, -0.528292, -0.528292]
opacity   = [-2.197225]
f_dc      = [-1.063,  0.7090, -0.7090]
f_rest    = [0.0]*45
```

**$\mathcal G_{1,3}$ ($=G_3c_2$):**

```
mu        = [-1.2165, -0.8069,  0.6473]
rotation  = [ 1.0,  0.0,  0.0,  0.0]
scaling   = [-0.361091, -0.361091, -0.361091]
opacity   = [-2.197225]
f_dc      = [-1.418, -0.7090,  1.418]
f_rest    = [0.0]*45
```

**$\mathcal G_{1,4}$ ($=G_4c_2$):**

```
mu        = [ 1.5145, -1.0912,  0.5124]
rotation  = [ 1.0,  0.0,  0.0,  0.0]
scaling   = [-0.587855, -0.587855, -0.587855]
opacity   = [-2.197225]
f_dc      = [ 0.0,  0.0,  0.0]
f_rest    = [0.0]*45
```

Kiểm đếm: mỗi khối trên có $3+4+3+1+3+45=59$ số $\Rightarrow$ $4\times59=236$ số cho toàn bộ $\mathcal G_1$ — khớp đúng công thức đếm ở mục 7.8.2.

### Đối chiếu ngược: 8 Gaussian con trung gian (trước bước prune, mục 7.4.5) ở dạng vector đầy đủ

Liệt kê lại để Phần 8 (nếu cần kiểm tra bước trung gian, ví dụ đối chiếu chi phí $T_{\text{opt}}(N=8)$ thoáng qua trước khi prune, chương 13) có sẵn dữ liệu — các trường `rotation`, `opacity`, `f_rest` giống hệt bảng trên (kế thừa nguyên từ cha), chỉ khác `mu`, `scaling`, `f_dc`:

| # | tên | $[\mu_x,\mu_y,\mu_z]$ | $\tilde s$ (×3) | $[f_{dc,0},f_{dc,1},f_{dc,2}]$ | còn sống sau prune? |
|---|---|---|---|---|---|
| 1 | $G_1c_1$ | $[0.1069,-0.1124,0.5447]$ | $-0.631906$ | $[1.063,-1.063,-1.063]$ | ❌ xoá (mục 7.6.3) |
| 2 | $G_2c_1$ | $[1.7302,1.1935,-0.1639]$ | $-0.528292$ | $[-1.063,0.7090,-0.7090]$ | ❌ xoá |
| 3 | $G_3c_1$ | $[-2.9925,-0.4440,-0.3893]$ | $-0.361091$ | $[-1.418,-0.7090,1.418]$ | ❌ xoá |
| 4 | $G_4c_1$ | $[0.6659,0.4266,0.0858]$ | $-0.587855$ | $[0,0,0]$ | ❌ xoá |
| 5 | $G_1c_2$ | $[0.0892,-0.4556,0.3075]$ | $-0.631906$ | $[1.063,-1.063,-1.063]$ | ✅ giữ $\to\mathcal G_{1,1}$ |
| 6 | $G_2c_2$ | $[-0.6938,-0.2880,0.5390]$ | $-0.528292$ | $[-1.063,0.7090,-0.7090]$ | ✅ giữ $\to\mathcal G_{1,2}$ |
| 7 | $G_3c_2$ | $[-1.2165,-0.8069,0.6473]$ | $-0.361091$ | $[-1.418,-0.7090,1.418]$ | ✅ giữ $\to\mathcal G_{1,3}$ |
| 8 | $G_4c_2$ | $[1.5145,-1.0912,0.5124]$ | $-0.587855$ | $[0,0,0]$ | ✅ giữ $\to\mathcal G_{1,4}$ |

---

## 7.8ter — Bảng đối chiếu số học từng bước với chương 12 (checklist)

Bảng dưới liệt kê **từng con số quan trọng** đã dùng trong Phần 7 và xác nhận nó trùng khớp (✅) với con số đã công bố trong `12-adaptive-density-control.md`, không phát sinh số liệu mới (Quy ước 1, §16.4):

| # | Đại lượng | Giá trị Phần 7 | Giá trị ch.12 | Khớp? |
|---|---|---|---|---|
| 1 | extent | 1.690250 | 1.690250 | ✅ |
| 2 | $\delta\cdot$extent | 0.0016902 | 0.001690 | ✅ |
| 3 | $0.1\cdot$extent | 0.169025 | 0.169025 | ✅ |
| 4 | $\sum_v\text{counts}$ (G1..G4) | 3132, 3053, 3200, 2923 | 3132, 3053, 3200, 2923 | ✅ |
| 5 | Importance (G1..G4) | 1044, 1017, 1066, 974 | 1044, 1017, 1066, 974 | ✅ |
| 6 | $E_{\text{photo}}$ (v1,v2,v3) | 0.277867, 0.237468, 0.222869 | 0.277867, 0.237468, 0.222869 | ✅ |
| 7 | Pruning (G1..G4) | 0.7654, 0.4507, 1.0000, 0.0000 | 0.7654, 0.4507, 1.0000, 0.0000 | ✅ |
| 8 | $\bar g_i$ (G1..G4) | 7.082e-3, 5.931e-3, 5.980e-3, 6.035e-3 | 7.082e-3, 5.931e-3, 5.980e-3, 6.035e-3 | ✅ |
| 9 | $\bar g^{\text{abs}}_i$ (G1..G4) | 3.183e-2, 3.170e-2, 3.098e-2, 2.399e-2 | 3.183e-2, 3.170e-2, 3.098e-2, 2.399e-2 | ✅ |
| 10 | nhánh densify | split ×4, clone ×0 | split ×4, clone ×0 | ✅ |
| 11 | $s^{(j)}$ con (G1..G4) | 0.5316, 0.5896, 0.6969, 0.5555 | 0.5316, 0.5896, 0.6969, 0.5555 (khoảng $[0.53,0.70]$) | ✅ |
| 12 | $\mu^{(j)}$ 8 con | bảng 7.4.5 | bảng mục 7.3 "Split cụ thể (seed 0)" | ✅ |
| 13 | $\lvert\mathcal C\rvert$ (trên 8 con) | 8 | 8 | ✅ |
| 14 | remove budget (trên 8 con) | 4 | 4 | ✅ |
| 15 | $w_i$ (G1..G4 gốc) | 4.2621, 1.8204, $10^6$, 1.0000 | 4.2621, 1.8204, $10^6$, 1.0000 | ✅ |
| 16 | $p_i$ (G1..G4 gốc) | 4e-6, 2e-6, 0.999993, 1e-6 | 4e-6, 2e-6, 0.999993, 1e-6 | ✅ |
| 17 | tập xoá (thứ tự thật code) | $\{G_1c_1,G_2c_1,G_3c_1,G_4c_1\}$ | $\{G_1c_1,G_2c_1,G_3c_1,G_4c_1\}$ | ✅ |
| 18 | $N$ sau densify+prune (thật) | 4 | 4 | ✅ |
| 19 | $N$ theo công thức tổng quát (không kể lệch chỉ số) | 6 | 6 | ✅ |
| 20 | ép opacity | $\tilde\alpha=-2.197225$ (không đổi) | $\tilde\alpha=-2.197225$ (không đổi) | ✅ |
| 21 | reset\_opacity ở $t=5000$? | Không ($5000\bmod3000\ne0$) | (không nêu trực tiếp, suy từ bảng 7.1) | ✅ (suy luận nhất quán) |
| 22 | `final_prune` minh hoạ | xoá duy nhất $G_3$ | xoá duy nhất $G_3$ | ✅ |

Toàn bộ 22 mục đối chiếu đều khớp — Phần 7 không phát sinh số liệu mâu thuẫn với chương 12 đã xuất bản, đúng yêu cầu Quy ước 1 của §16.4 (Chương 16 — Đề bài).

---

## 7.10 — Ghi chú đối chiếu với code

| Mục | Vị trí code | Ghi chú |
|---|---|---|
| Lấy mẫu $V$ camera | `utils/fast_utils.py:10-19` (`sampling_cameras`), `:13` (`num_cams=10`) | Cảnh đồ chơi chỉ có 3 camera → $V=3$ (quy ước đã nêu ở đầu file test ch.12) |
| 7 bước Importance/Pruning | `utils/fast_utils.py:33-93` (`compute_gaussian_score_fastgs`) | `get_loss` dòng 21-25 (bước ①②), `metric_map` dòng 70 (bước ③), `accum_metric_counts` từ `render_pkg` dòng 74 (bước ④, tính trong `forward.cu:406-408`), `importance_score` dòng 90 (bước ⑤), `photometric_loss` dòng 27-31 (bước ⑥), `pruning_score` dòng 87 (bước ⑦) |
| Ngưỡng $\tau_{\text{loss}}=0.1$ | `arguments/__init__.py:90` | |
| $\lambda=0.2$ | `utils/fast_utils.py:30` | trong `compute_photometric_loss` |
| Densify AND | `scene/gaussian_model.py:433-462` (`densify_and_prune_fastgs`, phần đầu) | `grad_thresh`, `grad_abs_thresh`, `dense` đọc từ `args` (CLI) |
| $\tau_{\text{grad}}=2\times10^{-4}$ | `--grad_thresh`, `arguments/__init__.py` | |
| $\tau^{\text{abs}}_{\text{grad}}=1.2\times10^{-3}$ | `--grad_abs_thresh` | |
| $\delta=0.001$ | `--dense`, `arguments/__init__.py:95` | |
| Cổng Importance$>5$ | `scene/gaussian_model.py:459` (`metric_mask = importance_score > 5`) | |
| Clone | `scene/gaussian_model.py:420-431` (`densify_and_clone_fastgs`) | không chạy ở cảnh này (nhánh clone rỗng) |
| Split | `scene/gaussian_model.py:396-418` (`densify_and_split_fastgs`) | `N=2`, `stds = get_scaling` (không phải bình phương — nhưng `torch.normal(std=stds)` lấy `stds` làm độ lệch chuẩn nên tương đương $\text{diag}(s^2)$), scale mới chia `0.8*N=1.6` |
| Tập ứng viên prune $\mathcal C$ | `scene/gaussian_model.py:464-468` | `min_opacity=0.005` (đọc từ `train.py`), `max_screen_size=20` chỉ truyền khi $t>3000$ |
| `remove_budget` | `scene/gaussian_model.py:472` (`int(0.5 * to_remove)`) | |
| Trọng số + multinomial | `scene/gaussian_model.py:473-483` | **chú ý `padded_importance[:scores.shape[0]]`** — nguồn gốc "lệch chỉ số" phân tích ở mục 7.6.0 |
| Ép opacity | `scene/gaussian_model.py:485-487` | chạy **vô điều kiện** mỗi lần gọi `densify_and_prune_fastgs`, không phân biệt có prune hay không |
| Tích luỹ gradient | `scene/gaussian_model.py:493-496` (`add_densification_stats`) | `xyz_gradient_accum` (cột có dấu), `xyz_gradient_accum_abs` (cột trị tuyệt đối) — nguồn `backward.cu:589-597` |
| `final_prune_fastgs` | `scene/gaussian_model.py:498-505` | `min_opacity=0.1` (khác 0.005!), ngưỡng Pruning $0.9$ đọc trực tiếp trong hàm |
| Lịch gọi trong vòng lặp huấn luyện | `train.py:127-158` | densify/prune mỗi `densification_interval` trong $(500,15000)$; opacity reset mỗi 3000; `final_prune_fastgs` mỗi 3000 trong $(15000,30000)$ |

---

## 7.11 — Phụ lục: script kiểm chứng bằng numpy

```python
import numpy as np

np.random.seed(0)

# ---- Đầu vào (mục 7.0.3) ----
mu    = np.array([[0,0,0],[0.5,0.3,0.5],[-0.4,-0.2,1.0],[0.3,-0.5,0.2]], dtype=float)
s     = np.array([0.8505, 0.9434, 1.1150, 0.8888])
alpha = np.full(4, 0.1)
extent = 1.690250
delta_extent = 0.001 * extent
tenpct_extent = 0.1 * extent

# ---- 7.2: Importance / Pruning (kết quả đã cho từ script render ch07_test.py) ----
counts_v = np.array([[1248,954,930],[1181,1003,869],[1245,978,977],[1180,973,770]])  # counts^(v)_i, v=1..3
importance = np.floor(counts_v.sum(axis=1) / 3).astype(int)
E_photo = np.array([0.277867, 0.237468, 0.222869])
raw = (counts_v * E_photo).sum(axis=1)
pruning = (raw - raw.min()) / (raw.max() - raw.min())
print("Importance:", importance)          # [1044 1017 1066  974]
print("Pruning:   ", np.round(pruning,4)) # [0.7654 0.4507 1.     0.    ]

# ---- 7.3: densify AND ----
g_bar     = np.array([7.082e-3, 5.931e-3, 5.980e-3, 6.035e-3])
g_abs_bar = np.array([3.183e-2, 3.170e-2, 3.098e-2, 2.399e-2])
tau_grad, tau_abs = 2e-4, 1.2e-3

grad_ok     = g_bar >= tau_grad
grad_abs_ok = g_abs_bar >= tau_abs
clone_size_ok = s <= delta_extent
split_size_ok = s >  delta_extent
importance_ok = importance > 5

all_clone = clone_size_ok & grad_ok & importance_ok
all_split = split_size_ok & grad_abs_ok & importance_ok
print("clone:", all_clone, " split:", all_split)   # clone: [F F F F]  split: [T T T T]

# ---- 7.4: split (seed 0) ----
children_mu, children_s = [], []
for i in range(4):
    eps = np.random.normal(0.0, s[i], size=(2,3))
    for j in range(2):
        children_mu.append(mu[i] + eps[j])
    s_child = s[i] / 1.6
    children_s += [s_child, s_child]
children_mu = np.array(children_mu)
children_s  = np.array(children_s)
print("children mu:\n", np.round(children_mu, 4))
print("children s: ", np.round(children_s, 4))

# ---- 7.5: candidate set C trên 8 con ----
big_ws = children_s > tenpct_extent
C = np.where(big_ws)[0]
print("|C| =", len(C))                       # 8
remove_budget = int(0.5 * len(C))
print("remove_budget =", remove_budget)       # 4

# ---- 7.6: trọng số + multinomial (chỉ 4 phần tử đầu có trọng số dương) ----
scores = 1 - pruning                          # từ 4 Gaussian GỐC, kích thước 4
padded_w = np.zeros(8)
padded_w[:4] = 1.0 / (1e-6 + scores)
print("padded_w:", padded_w)

def weighted_sample_without_replacement(w, k, rng):
    w = w.copy().astype(float)
    idx = []
    for _ in range(k):
        p = w / w.sum()
        choice = rng.choice(len(w), p=p)
        idx.append(choice)
        w[choice] = 0.0
    return np.array(idx)

rng = np.random.default_rng(0)
S = weighted_sample_without_replacement(padded_w, remove_budget, rng)
print("S =", sorted(S))                       # {0,1,2,3} == {G1c1,G2c1,G3c1,G4c1}

final_prune = np.zeros(8, dtype=bool)
final_prune[S] = True
N_after = 8 - final_prune.sum()
print("N sau prune =", N_after)               # 4

# ---- 7.7: ép opacity ----
alpha_children = np.full(8, 0.1)
alpha_clamped = np.minimum(alpha_children, 0.8)
tilde_alpha = np.log(alpha_clamped / (1 - alpha_clamped))
print("tilde_alpha (unchanged):", tilde_alpha[0])   # -2.197225

# ---- 7.9: final_prune minh hoạ trên 4 Gaussian gốc ----
final_prune_illustrative = (alpha < 0.1) | (pruning > 0.9)
print("final_prune (minh hoạ, KHÔNG áp dụng ở t=5000):", final_prune_illustrative)  # [F F T F]
```

Kết quả in ra khớp với mọi bảng số ở mục 7.2–7.9 (Importance $=(1044,1017,1066,974)$; Pruning $=(0.7654,0.4507,1.0000,0.0000)$; split cả 4; $\lvert\mathcal C\rvert=8$; budget $=4$; $\mathcal S=\{0,1,2,3\}$ tương ứng chỉ số 0-based của $\{G_1c_1,G_2c_1,G_3c_1,G_4c_1\}$; $N=4$; opacity không đổi; `final_prune` minh hoạ chỉ xoá $G_3$).

---

## 7.12 — Bài tập

**Bài tập 7.1.** Mục 7.0.2 lập luận rằng bước Adam của Phần 6 tại $t=5000$ không đủ lớn để đổi bất kỳ quyết định densify/prune nào. Tính cụ thể: với $\eta_{xyz}(5000)\cdot\text{extent}\approx2.7\times10^{-4}$ (chương 11.4), Gaussian $G_1$ có thể dịch chuyển tối đa bao nhiêu theo mỗi trục sau một bước Adam? So với khoảng cách trung bình giữa các Gaussian ($\|\mu_i-\mu_j\|\sim0.5$–$1.5$), hãy giải thích vì sao dịch chuyển này không đổi footprint hữu hình $\Omega_i^{(v)}$ đáng kể.

**Bài tập 7.2.** Mục 7.2.5 cho thấy Importance của cả 4 Gaussian nằm trong $[974,1066]$, vượt xa ngưỡng $5$. Giải thích bằng lời (dựa vào §16.1 — $\alpha$ mô hình $=0.1$ so với $\alpha$ GT $=0.9$): tại sao gần như toàn bộ ảnh bị đánh dấu lỗi ở bước ③, khiến Importance của *mọi* Gaussian đều rất lớn? Nếu mô hình đã học gần hội tụ ($\alpha\to0.9$), em kỳ vọng Importance sẽ thay đổi như thế nào, và ngưỡng $5$ khi đó có ý nghĩa lọc thực sự ra sao?

**Bài tập 7.3.** Mục 7.6.0 chỉ ra rằng `padded_importance[:scores.shape[0]]` (`gaussian_model.py:478`) gán trọng số Pruning của 4 Gaussian **gốc** cho 4 phần tử **đầu tiên** của quần thể 8 Gaussian con — đúng chỉ số, không đánh lại theo cha-con thật. Vẽ sơ đồ (bằng bảng hoặc chữ) minh hoạ tương ứng chỉ số $1..8$ với tên $G_ic_j$ (mục 7.4.5), và chỉ ra cụ thể: 4 Gaussian nào "vô tình" nhận được trọng số dương (dù đại diện đúng Pruning của cha), và 4 Gaussian nào "vô tình" miễn nhiễm hoàn toàn (trọng số 0) dù thuộc cùng 4 gốc.

**Bài tập 7.4.** So sánh hai con số "$N$ mới": (a) công thức tổng quát $N=4+0+4-2=6$ (mục 7.6.4, tính trực tiếp trên 4 Gaussian gốc với budget $=\lfloor0.5\times4\rfloor=2$); (b) kết quả thật $N=8-4=4$ (mục 7.6.3, tính đúng thứ tự thực thi code). Giải thích: vì sao budget khác nhau ($2$ so với $4$) dù cùng công thức $\lfloor0.5\times|\mathcal C|\rfloor$? (Gợi ý: $|\mathcal C|$ được tính trên quần thể nào ở mỗi trường hợp.)

**Bài tập 7.5.** Mục 7.7.1 tính $5000\bmod3000=2000\ne0$ nên `reset_opacity` không chạy ở $t=5000$. Tìm $t$ nhỏ nhất $>5000$ mà `reset_opacity` sẽ chạy. Nếu bài toán lớn (Chương 16) giả định densify/prune chạy lặp lại mỗi 500 vòng từ $t=500$ đến $t=15000$ (bảng 7.1 ch.12: `densification_interval`, preset $500$), hãy liệt kê tất cả các mốc $t$ trong khoảng $(3000,15000)$ mà **cả hai** sự kiện (densify/prune **và** opacity reset) trùng nhau.

**Bài tập 7.6.** Mục 7.9 trình bày `final_prune_fastgs` minh hoạ dùng Pruning score của **4 Gaussian gốc** ($G_1$–$G_4$), không phải 4 Gaussian con thật sự còn lại sau $t=5000$ ($G_1c_2,G_2c_2,G_3c_2,G_4c_2$). Giải thích vì sao Phần 7 không thể tính Pruning score mới cho 4 Gaussian con này mà không "phát sinh số liệu mới" (vi phạm Quy ước 1, §16.4) — cần những gì (ảnh render mới, GT mới, ...) để tính đúng?

**Bài tập 7.7.** Tổng hợp toàn bộ Phần 7: vẽ lại sơ đồ luồng $N: 4\xrightarrow{\text{split}\times4}8\xrightarrow{\text{prune}\times4}4$ kèm theo bảng 59 tham số/Gaussian đầy đủ ở mục 7.8.3, rồi so sánh với $\mathcal G_0$ (Phần 1, chương 6): những tham số nào **thay đổi** ($\mu$, $\tilde s$), những tham số nào **giữ nguyên** ($\tilde\alpha$, $q$, $k_{00}$, $f_{\text{rest}}$)? Từ đó giải thích vì sao Phần 8 (chương 13, xuất `.ply`) có thể coi $\mathcal G_1$ là "cùng cấu trúc 59 tham số/Gaussian" với $\mathcal G_0$ dù nội dung số liệu khác hẳn.

---

## Đầu ra chuyển cho Phần 8

**Đầu ra chuyển cho Phần 8 (Chương 13 — Cost model + xuất `.ply`): $\mathcal G_1$ = quần thể Gaussian sau densify/prune tại $t=5000$ (bảng đầy đủ ở trên).**

Tóm tắt: $N=4$ Gaussian, mỗi Gaussian 59 tham số ($\mu$: 3, $\tilde q$: 4, $\tilde s$: 3 — đẳng hướng, $\tilde\alpha$: 1, $f_{\text{dc}}$: 3, $f_{\text{rest}}$: 45) — bảng số đầy đủ ở mục 7.8.3. Phần 8 dùng $N=4$ này (thay vì $N_0=4$ của $\mathcal G_0$ ban đầu — cùng số lượng nhưng nội dung Gaussian khác hẳn) để: (a) so sánh mô hình chi phí $T_{\text{iter}}=T_{\text{fwd}}(K)+T_{\text{bwd}}(K)+T_{\text{opt}}(N)$ trước/sau bước densify+prune, và (b) ghi ra `point_cloud.ply` theo layout 62 cột ($x,y,z,n_x,n_y,n_z,\text{f\_dc}_{0..2},\text{f\_rest}_{0..44},\text{opacity},\text{scale}_{0..2},\text{rot}_{0..3}$) — trong đó $n_x=n_y=n_z=0$ (normal không được pipeline này sử dụng, giữ chỗ theo đúng định dạng 3DGS gốc), `scale`$_{0..2}=\tilde s$ (cùng giá trị lặp lại 3 trục vì đẳng hướng), `opacity`$=\tilde\alpha$ (giá trị logit, chưa qua sigmoid — đúng quy ước lưu trữ `.ply` của 3DGS).

[Phần 8 →](08-cost-model-ply.md)
