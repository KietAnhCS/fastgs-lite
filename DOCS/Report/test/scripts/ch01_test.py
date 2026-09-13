"""
Bài test số chương 1 — SfM Points -> Initialization.
Chỉ dùng numpy (môi trường không có torch nên không import được utils/*.py).
Mọi công thức được chép lại đúng từ code, có ghi số dòng tham chiếu:

  - scene/gaussian_model.py:137-160   create_from_pcd
  - scene/dataset_readers.py:45-66    getNerfppNorm
  - utils/sh_utils.py:26,114          C0, RGB2SH
  - utils/general_utils.py:18         inverse_sigmoid
  - submodules/simple-knn/simple_knn.cu:148-184  boxMeanDist (distCUDA2: trung bình
    bình phương khoảng cách tới 3 láng giềng gần nhất, loại bỏ chính điểm)

Chạy:  python DOCS/Report/test/scripts/ch01_test.py
"""
import numpy as np

np.set_printoptions(precision=10, suppress=True)


def fmt(x, sig=4):
    """In 4 chữ số có nghĩa (giống markdown)."""
    return f"{x:.{sig}g}"


# --------------------------------------------------------------------------
# 1.1 Đầu vào — cảnh đồ chơi (DOCS/Report/test/00-scene.md)
# --------------------------------------------------------------------------
points = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.3, 0.5],
    [-0.4, -0.2, 1.0],
    [0.3, -0.5, 0.2],
], dtype=np.float64)

colors = np.array([
    [0.8, 0.2, 0.2],
    [0.2, 0.7, 0.3],
    [0.1, 0.3, 0.9],
    [0.5, 0.5, 0.5],
], dtype=np.float64)

cam_centers = np.array([
    [0.0, 0.0, -4.0],
    [1.5, 0.0, -4.0],
    [-1.5, 0.5, -4.0],
], dtype=np.float64)

N0 = points.shape[0]
V = cam_centers.shape[0]
MAX_SH_DEGREE = 3

print("=" * 72)
print("1.1 DAU VAO")
print("=" * 72)
print(f"N0 = {N0} diem SfM, V = {V} camera, max_sh_degree = {MAX_SH_DEGREE}")
for k in range(N0):
    print(f"  p_{k+1} = {points[k]}   c_{k+1} = {colors[k]}")
for v in range(V):
    print(f"  c_cam_{v+1} = {cam_centers[v]}")


# --------------------------------------------------------------------------
# Các hàm chép lại từ code
# --------------------------------------------------------------------------
C0 = 0.28209479177387814            # utils/sh_utils.py:26


def RGB2SH(rgb):                     # utils/sh_utils.py:114
    return (rgb - 0.5) / C0


def inverse_sigmoid(x):              # utils/general_utils.py:18
    return np.log(x / (1 - x))


def distCUDA2(pts):
    """simple_knn.cu:148-184 boxMeanDist — với mỗi điểm: 3 bình phương khoảng cách
    nhỏ nhất tới các điểm KHÁC (i == idx bị bỏ qua), rồi chia 3."""
    P = pts.shape[0]
    out = np.zeros(P)
    for i in range(P):
        d2 = np.sum((pts - pts[i]) ** 2, axis=1)
        d2 = np.delete(d2, i)                 # loại chính nó
        best3 = np.sort(d2)[:3]
        out[i] = best3.sum() / 3.0
    return out


# --------------------------------------------------------------------------
# 1.2 Mỗi điểm SfM sinh một Gaussian — create_from_pcd (gaussian_model.py:137-160)
# --------------------------------------------------------------------------
print()
print("=" * 72)
print("1.2 KHOI TAO 59 THAM SO / GAUSSIAN")
print("=" * 72)

# --- mu_i = p_i  (line 139, 154)
mu = points.copy()

# --- bảng ||p_i - p_j||^2 đầy đủ
print("\n[a] Ma tran binh phuong khoang cach ||p_i - p_j||^2:")
D2 = np.zeros((N0, N0))
for i in range(N0):
    for j in range(N0):
        D2[i, j] = np.sum((points[i] - points[j]) ** 2)
header = "        " + "".join(f"{'p_'+str(j+1):>14}" for j in range(N0))
print(header)
for i in range(N0):
    print(f"  p_{i+1}  " + "".join(f"{D2[i, j]:14.6f}" for j in range(N0)))

# --- d2_knn3, clamp, scale  (line 147-148)
print("\n[b] d2_knn3 = (1/3) * tong 3 gia tri nho nhat (khac chinh no); clamp_min 1e-7;")
print("    s~_i = log(sqrt(d2));  s_i = exp(s~_i) = sqrt(d2)")
dist2_raw = distCUDA2(points)
dist2 = np.maximum(dist2_raw, 1e-7)          # torch.clamp_min(..., 0.0000001)
scaling_raw = np.log(np.sqrt(dist2))          # torch.log(torch.sqrt(dist2))
scaling = np.repeat(scaling_raw[:, None], 3, axis=1)   # [...,None].repeat(1,3)
scale_act = np.exp(scaling)                   # scaling_activation = torch.exp

for i in range(N0):
    others = np.delete(D2[i], i)
    terms = " + ".join(f"{t:.6f}" for t in others)
    print(f"  i={i+1}: d2_knn3 = ({terms})/3 = {dist2_raw[i]:.10f}"
          f"  -> clamp: {dist2[i]:.10f}")
    print(f"        s~_i = log(sqrt({dist2[i]:.10f})) = {scaling_raw[i]:.10f}"
          f"   s_i = exp(s~_i) = {scale_act[i, 0]:.10f}")

# --- rotation  (line 149-150)
rots = np.zeros((N0, 4))
rots[:, 0] = 1.0
print(f"\n[c] q~_i = {rots[0]} (moi Gaussian)")

# --- opacity  (line 152)
alpha_init = 0.1
opacity_raw = inverse_sigmoid(alpha_init * np.ones((N0, 1)))
print(f"\n[d] alpha~_i = inverse_sigmoid(0.1) = log(0.1/0.9) = {opacity_raw[0, 0]:.10f}")
print(f"    kiem tra sigmoid(alpha~) = {1/(1+np.exp(-opacity_raw[0,0])):.10f}")

# --- SH  (line 140-143, 155-156)
fused_color = RGB2SH(colors)
n_coef = (MAX_SH_DEGREE + 1) ** 2            # 16
features = np.zeros((N0, 3, n_coef))
features[:, :3, 0] = fused_color
features[:, 3:, 1:] = 0.0
features_dc = np.transpose(features[:, :, 0:1], (0, 2, 1))     # (N,1,3)
features_rest = np.transpose(features[:, :, 1:], (0, 2, 1))    # (N,15,3)
print(f"\n[e] k_00 = (c - 0.5)/C0,  C0 = {C0:.10f}")
for i in range(N0):
    parts = ", ".join(f"({colors[i,ch]}-0.5)/C0={fused_color[i,ch]:.10f}" for ch in range(3))
    print(f"  i={i+1}: {parts}")
print(f"    features_dc.shape = {features_dc.shape}, features_rest.shape = {features_rest.shape}"
      f" -> {features_rest.shape[1]*features_rest.shape[2]} he so bac cao = 0, "
      f"max|features_rest| = {np.abs(features_rest).max()}")

n_params = 3 + 4 + 3 + 1 + 3 * n_coef
print(f"\n[f] So tham so / Gaussian = 3 (mu) + 4 (q) + 3 (s) + 1 (alpha) + {3*n_coef} (SH) = {n_params}")


# --------------------------------------------------------------------------
# 1.3 extent — getNerfppNorm (dataset_readers.py:45-66)
# --------------------------------------------------------------------------
print()
print("=" * 72)
print("1.3 EXTENT (getNerfppNorm)")
print("=" * 72)

# Trong code: W2C = getWorld2View2(R, T); C2W = inv(W2C); c_v = C2W[:3, 3].
# Voi xoay don vi, W2C = [[I, -c_v],[0,1]] nen C2W[:3,3] = c_v. Kiem tra bang inv thuc:
cam_centers_from_inv = []
for v in range(V):
    W2C = np.eye(4)
    W2C[:3, 3] = -cam_centers[v]
    C2W = np.linalg.inv(W2C)
    cam_centers_from_inv.append(C2W[:3, 3:4])
cc = np.hstack(cam_centers_from_inv)                # (3, V)  -- np.hstack(cam_centers)
avg = np.mean(cc, axis=1, keepdims=True)            # avg_cam_center
dist = np.linalg.norm(cc - avg, axis=0, keepdims=True)
diagonal = np.max(dist)
radius = diagonal * 1.1
translate = -avg.flatten()

print(f"  c_v (tu inv(W2C)) = \n{cc.T}")
print(f"  c_bar = mean = {avg.flatten()}")
for v in range(V):
    print(f"  ||c_{v+1} - c_bar|| = ||{(cc[:, v]-avg.flatten())}|| = {dist[0, v]:.10f}")
print(f"  max = {diagonal:.10f}")
print(f"  extent = 1.1 * max = {radius:.10f}")
print(f"  translate = -c_bar = {translate}")

DELTA = 0.001            # arguments/__init__.py:95  self.dense
ETA_XYZ = 0.00016        # arguments/__init__.py:76  position_lr_init
thr_clone = DELTA * radius            # gaussian_model.py:451-452
thr_big = 0.1 * radius                # gaussian_model.py:467
lr_xyz = ETA_XYZ * radius             # gaussian_model.py:168
print(f"\n  Ba nguong dung extent:")
print(f"    delta*extent   = {DELTA} * {radius:.10f} = {thr_clone:.10f}")
print(f"    0.1*extent     = {thr_big:.10f}")
print(f"    eta_xyz*extent = {ETA_XYZ} * {radius:.10f} = {lr_xyz:.10f}")
print(f"  So sanh max(s_i) voi delta*extent (clone neu <=, split neu >):")
for i in range(N0):
    tag = "clone" if scale_act[i].max() <= thr_clone else "split"
    print(f"    i={i+1}: max s_i = {scale_act[i].max():.10f}  ->  {tag}")


# --------------------------------------------------------------------------
# 1.4 Bo nho optimizer — training_setup (gaussian_model.py:161-177)
# --------------------------------------------------------------------------
print()
print("=" * 72)
print("1.4 BO NHO TRANG THAI OPTIMIZER")
print("=" * 72)
per_gauss_floats = 3 * n_params
bytes_f32 = 4
print(f"  {n_params} tham so x 3 (theta, m, v) = {per_gauss_floats} float / Gaussian")
for N in [N0, 10**6]:
    floats = per_gauss_floats * N
    b = floats * bytes_f32
    print(f"  N = {N:>8d}: {floats:>12,d} float = {b:>14,d} byte = {b/1024**2:.4f} MiB = {b/1e6:.4f} MB")
# tach theo optimizer / shoptimizer
n_main = 3 + 4 + 3 + 1 + 3     # xyz, rot, scale, opacity, f_dc
n_sh = 3 * (n_coef - 1)        # f_rest = 45
print(f"  Trong do optimizer (mu,q,s,alpha,k00): {n_main} tham so -> {3*n_main} float/Gaussian;"
      f" shoptimizer (k_lm, l>=1): {n_sh} tham so -> {3*n_sh} float/Gaussian")


# --------------------------------------------------------------------------
# 1.5 Dau ra cua khoi
# --------------------------------------------------------------------------
print()
print("=" * 72)
print("1.5 DAU RA CUA KHOI  G_0 = {theta_i}, i=1..4  (full precision)")
print("=" * 72)
for i in range(N0):
    print(f"  Gaussian {i+1}:")
    print(f"    mu      = {mu[i]}")
    print(f"    q~      = {rots[i]}")
    print(f"    s~      = {scaling[i]}   (s = {scale_act[i]})")
    print(f"    alpha~  = {opacity_raw[i,0]:.10f}   (alpha = 0.1)")
    print(f"    k_00    = {fused_color[i]}")
    print(f"    k_lm    = 0 (45 he so)")
print(f"  extent = {radius:.10f}")
print(f"  spatial_lr_scale = extent = {radius:.10f}")

print()
print("=" * 72)
print("BANG TOM TAT (4 chu so co nghia)")
print("=" * 72)
print("| i | mu | d2_knn3 | s~ (x3) | s (x3) | q~ | alpha~ | k_00 |")
for i in range(N0):
    print(f"| {i+1} | ({', '.join(fmt(x) for x in mu[i])}) | {fmt(dist2[i])} | {fmt(scaling_raw[i])} "
          f"| {fmt(scale_act[i,0])} | (1,0,0,0) | {fmt(opacity_raw[i,0])} "
          f"| ({', '.join(fmt(x) for x in fused_color[i])}) |")
print(f"| extent | {fmt(radius)} | delta*extent | {fmt(thr_clone)} | 0.1*extent | {fmt(thr_big)} "
      f"| eta*extent | {fmt(lr_xyz)} |")
