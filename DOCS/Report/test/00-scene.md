# Cảnh đồ chơi dùng chung cho toàn bộ bài test số (chương 1 → 8)

> Mọi file `NN-test.md` trong thư mục này tính trên **cùng một cảnh** dưới đây, để số liệu đầu ra
> của chương trước là đầu vào của chương sau. Chỉ dùng `numpy` (không có torch trong môi trường).
> Script sinh số nằm ở `scripts/chNN_test.py`; số trong markdown phải khớp với số script in ra.

## Ảnh và camera

| Đại lượng | Giá trị |
|---|---|
| $W\times H$ | $48\times32$ px → lưới tile $16\times16$: $3\times2=6$ tile |
| $\tan(\text{fov}_x/2)$ | $0.6$ → $f_x=W/(2\cdot0.6)=40$ |
| $\tan(\text{fov}_y/2)$ | $0.4$ → $f_y=H/(2\cdot0.4)=40$ |
| $z_n,\ z_f$ | $0.01,\ 100$ |

Ba camera, **xoay đơn vị** (nhìn dọc $+z$ của world), chỉ khác vị trí $c_v$:

| $v$ | $c_v$ (world) | $W_v$ (world→camera, 4×4) |
|---|---|---|
| 1 | $(0,\ 0,\ -4)$ | $W_v=[\,I_3\mid -c_v\,]$ (hàng cuối $0,0,0,1$), tức $t_v=\mu-c_v$ |
| 2 | $(1.5,\ 0,\ -4)$ | tương tự |
| 3 | $(-1.5,\ 0.5,\ -4)$ | tương tự |

Ground-truth $I^{(v)}_{\text{gt}}$: khi một chương cần ảnh GT, **định nghĩa** GT = ảnh render của cảnh
với tham số "đúng" là tham số khởi tạo nhưng opacity $\alpha=0.9$ thay vì $0.1$ (mô hình chưa học sẽ mờ hơn GT).

## Điểm SfM (4 điểm)

| $k$ | $p_k$ | $c_k$ (RGB) |
|---|---|---|
| 1 | $(0.0,\ 0.0,\ 0.0)$ | $(0.8,\ 0.2,\ 0.2)$ đỏ |
| 2 | $(0.5,\ 0.3,\ 0.5)$ | $(0.2,\ 0.7,\ 0.3)$ lục |
| 3 | $(-0.4,\ -0.2,\ 1.0)$ | $(0.1,\ 0.3,\ 0.9)$ lam |
| 4 | $(0.3,\ -0.5,\ 0.2)$ | $(0.5,\ 0.5,\ 0.5)$ xám |

$N_0=4$, nên 3-NN của mỗi điểm là 3 điểm còn lại.

## Hằng số (chương 0)

tile 16 px · low-pass 0.3 · clamp $1.3\tan(\text{fov}/2)$ · ngưỡng alpha $1/255$ · dừng sớm $10^{-4}$ ·
`mult` $=0.5$ · $\tau_{\text{grad}}=2\times10^{-4}$ · $\tau^{\text{abs}}_{\text{grad}}=1.2\times10^{-3}$ ·
$\tau_{\text{loss}}=0.1$ · $\delta=0.001$ · $\lambda=0.2$.

## Quy ước viết

- Mỗi bước: **công thức → thay số → kết quả** (làm tròn 4 chữ số có nghĩa, nhưng script giữ full precision).
- Cuối file: bảng "Đầu vào của khối" và "Đầu ra của khối" để chương sau lấy dùng.
- Ghi rõ chỗ nào là **kiểm chứng đối chiếu code** (gọi hàm thật trong repo: `utils/graphics_utils.py`,
  `utils/sh_utils.py`, `utils/general_utils.py`, `utils/loss_utils.py` nếu không cần torch) và chỗ nào là tính tay theo công thức.
