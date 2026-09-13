"""
Test số cho Chương 2 — 3D Gaussians (DOCS/Report/02-3d-gaussians.md)
Cảnh đồ chơi: DOCS/Report/test/00-scene.md (4 điểm SfM, 3 camera).

Chỉ dùng numpy (môi trường không có torch). Các công thức được CHÉP LẠI từ:
  - scene/gaussian_model.py:33-40        : activation (exp / sigmoid / normalize)
  - scene/gaussian_model.py:137-160      : create_from_pcd (khởi tạo chương 1, tính lại tại đây)
  - utils/general_utils.py:66-99         : build_rotation, build_scaling_rotation
  - utils/sh_utils.py                    : C0, C1, C2, C3, eval_sh, RGB2SH
  - cuda_rasterizer/forward.cu:24-77     : computeColorFromSH (+0.5, clamp, cờ clamped)
  - cuda_rasterizer/forward.cu:124-158   : computeCov3D
  - train.py:81-82                       : oneupSHdegree mỗi 1000 vòng

Chạy:  python DOCS/Report/test/scripts/ch02_test.py
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ----------------------------------------------------------------------------
# Hằng số SH — chép nguyên văn từ utils/sh_utils.py (file đó import torch nên
# không import trực tiếp được trong môi trường numpy-only)
# ----------------------------------------------------------------------------
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435]

# Kiểm tra hằng số bằng công thức giải tích
assert abs(C0 - 0.5 * np.sqrt(1 / np.pi)) < 1e-15
assert abs(C1 - 0.5 * np.sqrt(3 / np.pi)) < 1e-15


def RGB2SH(rgb):        # utils/sh_utils.py
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def eval_sh(deg, sh, dirs):
    """Chép từ utils/sh_utils.py::eval_sh, sh: [..., C, (deg+1)^2], dirs: [..., 3]."""
    assert 0 <= deg <= 3
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff
    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result + C2[0] * xy * sh[..., 4] + C2[1] * yz * sh[..., 5]
                      + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                      + C2[3] * xz * sh[..., 7] + C2[4] * (xx - yy) * sh[..., 8])
            if deg > 2:
                result = (result + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                          + C3[1] * xy * z * sh[..., 10]
                          + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                          + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                          + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                          + C3[5] * z * (xx - yy) * sh[..., 14]
                          + C3[6] * x * (xx - 3 * yy) * sh[..., 15])
    return result


def compute_color_from_sh(deg, sh, dirs):
    """forward.cu::computeColorFromSH: eval_sh + 0.5, ghi cờ clamped, rồi max(.,0)."""
    raw = eval_sh(deg, sh, dirs) + 0.5
    clamped = raw < 0
    return np.maximum(raw, 0.0), clamped, raw


# ----------------------------------------------------------------------------
# Activation & covariance — chép từ scene/gaussian_model.py, utils/general_utils.py
# ----------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def normalize(q):       # torch.nn.functional.normalize (dim=-1, eps bỏ qua)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def build_rotation(r):
    """utils/general_utils.py:66-86 — chuẩn hoá bên trong rồi dựng R."""
    norm = np.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2 + r[:, 3] ** 2)
    q = r / norm[:, None]
    R = np.zeros((q.shape[0], 3, 3))
    rr, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - rr * z)
    R[:, 0, 2] = 2 * (x * z + rr * y)
    R[:, 1, 0] = 2 * (x * y + rr * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - rr * x)
    R[:, 2, 0] = 2 * (x * z - rr * y)
    R[:, 2, 1] = 2 * (y * z + rr * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    """utils/general_utils.py:88-99 — L = R @ diag(s)."""
    L = np.zeros((s.shape[0], 3, 3))
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return R @ L


def strip_lowerdiag(L):   # utils/general_utils.py — 6 phần tử (xx,xy,xz,yy,yz,zz)
    u = np.zeros((L.shape[0], 6))
    u[:, 0] = L[:, 0, 0]; u[:, 1] = L[:, 0, 1]; u[:, 2] = L[:, 0, 2]
    u[:, 3] = L[:, 1, 1]; u[:, 4] = L[:, 1, 2]; u[:, 5] = L[:, 2, 2]
    return u


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    """scene/gaussian_model.py:26-30"""
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual = L @ np.transpose(L, (0, 2, 1))
    return strip_lowerdiag(actual), actual


def gaussian_density(x, mu, Sigma):
    d = x - mu
    return float(np.exp(-0.5 * d @ np.linalg.solve(Sigma, d)))


def sh_degree_schedule(t, max_deg=3):
    """train.py:81-82: mỗi 1000 vòng active_sh_degree += 1, trần max_sh_degree."""
    return min(max_deg, t // 1000)


# ----------------------------------------------------------------------------
# Cảnh đồ chơi (00-scene.md)
# ----------------------------------------------------------------------------
P = np.array([[0.0, 0.0, 0.0],
              [0.5, 0.3, 0.5],
              [-0.4, -0.2, 1.0],
              [0.3, -0.5, 0.2]])
COL = np.array([[0.8, 0.2, 0.2],
                [0.2, 0.7, 0.3],
                [0.1, 0.3, 0.9],
                [0.5, 0.5, 0.5]])
CAM = np.array([[0.0, 0.0, -4.0],
                [1.5, 0.0, -4.0],
                [-1.5, 0.5, -4.0]])
N = P.shape[0]
MAX_SH = 3


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


# ============================================================================
# Chương 1 tính lại: khởi tạo (create_from_pcd)
# ============================================================================
hdr("[CH1 tính lại] Khởi tạo từ 4 điểm SfM (create_from_pcd)")
D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)      # khoảng cách bình phương
dist2 = np.zeros(N)
for i in range(N):
    others = np.sort(np.delete(D2[i], i))[:3]              # 3-NN (= 3 điểm còn lại)
    dist2[i] = others.mean()
dist2 = np.maximum(dist2, 1e-7)
s_tilde = np.log(np.sqrt(dist2))[:, None].repeat(3, 1)     # log-scale, đẳng hướng
q_tilde = np.zeros((N, 4)); q_tilde[:, 0] = 1.0
a_tilde = inverse_sigmoid(0.1 * np.ones((N, 1)))
k00 = RGB2SH(COL)                                          # features_dc  [N,3]
k_rest = np.zeros((N, 3, (MAX_SH + 1) ** 2 - 1))           # features_rest [N,3,15]
mu = P.copy()
for i in range(N):
    print(f"G{i+1}: mu={mu[i]}  d2_knn3={dist2[i]:.6f}  s~={s_tilde[i,0]:.6f}  "
          f"q~={q_tilde[i]}  a~={a_tilde[i,0]:.6f}  k00={k00[i]}")

# ============================================================================
# 2.1 Activation
# ============================================================================
hdr("[2.1] Activation: q~ -> q (normalize), s~ -> s = exp, a~ -> alpha = sigmoid")
q = normalize(q_tilde)
s = np.exp(s_tilde)
alpha = sigmoid(a_tilde)
print(f"{'i':>2} | {'q~':>22} | {'||q~||':>7} | {'q':>22} | {'s~':>9} | {'s=exp(s~)':>10} | "
      f"{'a~':>9} | {'alpha':>7}")
for i in range(N):
    print(f"{i+1:>2} | {str(q_tilde[i]):>22} | {np.linalg.norm(q_tilde[i]):7.4f} | {str(q[i]):>22} | "
          f"{s_tilde[i,0]:9.6f} | {s[i,0]:10.6f} | {a_tilde[i,0]:9.6f} | {alpha[i,0]:7.4f}")
n_params = {"mu": mu.shape[1], "q~": q_tilde.shape[1], "s~": s_tilde.shape[1],
            "a~": a_tilde.shape[1], "k00 (features_dc)": k00.shape[1],
            "k_lm l>=1 (features_rest)": k_rest.shape[1] * k_rest.shape[2]}
print("Đếm tham số/Gaussian:", n_params, " tổng =", sum(n_params.values()))
assert sum(n_params.values()) == 59

# ============================================================================
# 2.2 Covariance 3D
# ============================================================================
hdr("[2.2a] R(q) với q=(1,0,0,0) và Sigma cho 4 Gaussian khởi tạo")
R_id = build_rotation(q_tilde)
print("R(1,0,0,0) =\n", R_id[0])
assert np.allclose(R_id[0], np.eye(3))
cov6, cov33 = build_covariance_from_scaling_rotation(s, 1.0, q_tilde)
for i in range(N):
    print(f"G{i+1}: s={s[i,0]:.6f}  s^2={s[i,0]**2:.6f}  Sigma(6)={cov6[i]}")

hdr("[2.2b] Quaternion không tầm thường q~=(0.9,0.1,0.3,0.2) -> R -> Sigma với s của G1")
qt = np.array([[0.9, 0.1, 0.3, 0.2]])
nq = np.linalg.norm(qt)
qn = qt / nq
print(f"||q~|| = sqrt(0.81+0.01+0.09+0.04) = sqrt({(qt**2).sum():.4f}) = {nq:.6f}")
print("q = q~/||q~|| =", qn[0])
rr, x, y, z = qn[0]
print("Các thành phần: r=%.6f x=%.6f y=%.6f z=%.6f" % (rr, x, y, z))
print("R00 = 1-2(y^2+z^2) = 1-2(%.6f+%.6f) = %.6f" % (y*y, z*z, 1-2*(y*y+z*z)))
print("R01 = 2(xy - rz)   = 2(%.6f-%.6f) = %.6f" % (x*y, rr*z, 2*(x*y-rr*z)))
print("R02 = 2(xz + ry)   = 2(%.6f+%.6f) = %.6f" % (x*z, rr*y, 2*(x*z+rr*y)))
print("R10 = 2(xy + rz)   = %.6f" % (2*(x*y+rr*z)))
print("R11 = 1-2(x^2+z^2) = %.6f" % (1-2*(x*x+z*z)))
print("R12 = 2(yz - rx)   = %.6f" % (2*(y*z-rr*x)))
print("R20 = 2(xz - ry)   = %.6f" % (2*(x*z-rr*y)))
print("R21 = 2(yz + rx)   = %.6f" % (2*(y*z+rr*x)))
print("R22 = 1-2(x^2+y^2) = %.6f" % (1-2*(x*x+y*y)))
R = build_rotation(qt)[0]
print("R =\n", R)
print("R R^T =\n", R @ R.T, "\ndet R =", np.linalg.det(R))
assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12
s1 = s[0:1]
L = build_scaling_rotation(s1, qt)[0]
print("L = R S =\n", L)
Sig = L @ L.T
print("Sigma = L L^T =\n", Sig)
print("Sigma (6 phần tử xx,xy,xz,yy,yz,zz) =", strip_lowerdiag(Sig[None])[0])
print("Đối xứng:", np.allclose(Sig, Sig.T))
ev = np.linalg.eigvalsh(Sig)
print("Trị riêng:", ev, " >=0:", bool((ev >= -1e-12).all()))
print("det Sigma =", np.linalg.det(Sig), "  (s1*s2*s3)^2 =", float(np.prod(s1[0]) ** 2))
print("Vì S = s*I (đẳng hướng): Sigma = s^2 R R^T = s^2 I ->", np.allclose(Sig, s1[0, 0] ** 2 * np.eye(3)))
# Anisotropic example so rotation actually matters
s_aniso = np.array([[0.6, 0.3, 0.1]])
L2 = build_scaling_rotation(s_aniso, qt)[0]
Sig2 = L2 @ L2.T
print("\n[Bổ sung] s dị hướng = (0.6,0.3,0.1) cùng q trên -> Sigma =\n", Sig2)
ev2 = np.linalg.eigvalsh(Sig2)
print("Trị riêng:", ev2, " so với s^2 =", np.sort(s_aniso[0] ** 2),
      " det =", np.linalg.det(Sig2), " (0.6*0.3*0.1)^2 =", (0.6 * 0.3 * 0.1) ** 2)
assert np.allclose(np.sort(ev2), np.sort(s_aniso[0] ** 2))
# Đối chiếu computeCov3D (forward.cu): glm column-major, M = S*R, Sigma = M^T M
# Trong ký hiệu hàng-major, tương đương Sigma = R S S^T R^T -> cùng kết quả:
Sig_cuda = (R @ np.diag(s_aniso[0])) @ (R @ np.diag(s_aniso[0])).T
assert np.allclose(Sig_cuda, Sig2)

# ============================================================================
# 2.3 Mật độ Gaussian
# ============================================================================
hdr("[2.3] G_i(x) tại 3 điểm: x=mu, x=mu+(s,0,0), x=mu+(2s,0,0)")
print("Lý thuyết: 1, e^{-1/2}=%.6f, e^{-2}=%.6f" % (np.exp(-0.5), np.exp(-2)))
for i in range(N):
    Si = cov33[i]
    vals = [gaussian_density(mu[i] + np.array([k * s[i, 0], 0, 0]), mu[i], Si) for k in (0, 1, 2)]
    print(f"G{i+1}: s={s[i,0]:.6f}  G(mu)={vals[0]:.6f}  G(mu+s e_x)={vals[1]:.6f}  G(mu+2s e_x)={vals[2]:.6f}")
# with the rotated anisotropic Sigma the same holds along the principal axis
d_axis = R[:, 0] * 0.6  # cột 0 của R = trục chính, độ dài s1
print("Sigma dị hướng xoay: G(mu + s1*R[:,0]) =", gaussian_density(mu[0] + d_axis, mu[0], Sig2),
      " G(mu + 2 s1 R[:,0]) =", gaussian_density(mu[0] + 2 * d_axis, mu[0], Sig2))

# ============================================================================
# 2.4 Màu SH
# ============================================================================
hdr("[2.4a] Hướng nhìn d = (mu_i - c_1)/|| || từ camera 1, c_1 = " + str(CAM[0]))
dirs = mu - CAM[0]
norms = np.linalg.norm(dirs, axis=1, keepdims=True)
dirs = dirs / norms
for i in range(N):
    print(f"G{i+1}: mu-c1={mu[i]-CAM[0]}  ||.||={norms[i,0]:.6f}  d=(x,y,z)={dirs[i]}")

hdr("[2.4b] Màu bậc 0 (t=0, D=0): c = max(0, 0.5 + C0*k00)")
sh_full = np.concatenate([k00[:, :, None], k_rest], axis=2)  # [N,3,16]
c0, cl0, raw0 = compute_color_from_sh(0, sh_full, dirs)
for i in range(N):
    print(f"G{i+1}: k00={k00[i]}  C0*k00={C0*k00[i]}  c={c0[i]}  SfM={COL[i]}  khớp={np.allclose(c0[i], COL[i])}")
assert np.allclose(c0, COL)
print("Không phụ thuộc camera: c(cam2)==c(cam1):",
      np.allclose(compute_color_from_sh(0, sh_full, normalize(mu - CAM[1]))[0], c0))

hdr("[2.4c] Bậc 1 cho G1: k1,-1=(0.1,0,0) k10=(0,0.2,0) k11=(0,0,-0.3)")
k1m1 = np.array([0.1, 0.0, 0.0]); k10 = np.array([0.0, 0.2, 0.0]); k11 = np.array([0.0, 0.0, -0.3])
x, y, z = dirs[0]
print(f"d = (x,y,z) = ({x:.6f}, {y:.6f}, {z:.6f}),  C1 = {C1:.6f}")
t1 = -C1 * y * k1m1; t2 = C1 * z * k10; t3 = -C1 * x * k11
print(f"-C1*y*k1,-1 = -{C1:.6f}*{y:.6f}*{k1m1} = {t1}")
print(f"+C1*z*k10   = +{C1:.6f}*{z:.6f}*{k10} = {t2}")
print(f"-C1*x*k11   = -{C1:.6f}*{x:.6f}*{k11} = {t3}")
sh1 = sh_full.copy()
sh1[0, :, 1] = k1m1; sh1[0, :, 2] = k10; sh1[0, :, 3] = k11
c1, cl1, raw1 = compute_color_from_sh(1, sh1[0:1], dirs[0:1])
print("Tổng bậc 0: C0*k00 =", C0 * k00[0])
print("raw = 0.5 + C0*k00 + l1 =", raw1[0], " -> c =", c1[0], " clamped =", cl1[0])
assert np.allclose(raw1[0], 0.5 + C0 * k00[0] + t1 + t2 + t3)

hdr("[2.4d] Trường hợp dẫn tới clamp: G1, k10 = (0, 0.2, -1.5), camera 1")
k10c = np.array([0.0, 0.2, -1.5])
sh2 = sh1.copy(); sh2[0, :, 2] = k10c
c2, cl2, raw2 = compute_color_from_sh(1, sh2[0:1], dirs[0:1])
print(f"+C1*z*k10 = {C1:.6f}*{z:.6f}*{k10c} = {C1*z*k10c}")
print("raw =", raw2[0], " -> c = max(raw,0) =", c2[0], " clamped =", cl2[0])
assert cl2[0, 2] and c2[0, 2] == 0.0
# Cùng hệ số, nhìn từ camera 3 (hướng khác) -> màu khác
d3 = normalize(mu[0:1] - CAM[2])
c3, cl3, raw3 = compute_color_from_sh(1, sh2[0:1], d3)
print("Cùng hệ số, camera 3: d =", d3[0], " raw =", raw3[0], " c =", c3[0], " clamped =", cl3[0])

hdr("[2.4e] Kiểm chứng eval_sh bậc 3 trên 16 hệ số ngẫu nhiên (cùng công thức, chỉ để chắc chắn khớp)")
rng = np.random.default_rng(0)
sh_rand = rng.normal(size=(1, 3, 16)) * 0.1
for deg in range(4):
    print(f"deg={deg}: eval_sh+0.5 =", eval_sh(deg, sh_rand, dirs[0:1])[0] + 0.5)

# ============================================================================
# 2.5 / lịch D(t)
# ============================================================================
hdr("[2.5] Lịch D(t) = min(3, floor(t/1000)) và số float SH hoạt động, N=4")
print(f"{'t':>6} | {'D':>2} | {'(D+1)^2':>8} | {'x3 kênh':>8} | {'trên N=4':>9} | {'dc':>3} | {'rest hoạt động':>15}")
for t in (0, 999, 1000, 1999, 2000, 2999, 3000, 15000):
    D = sh_degree_schedule(t)
    per_ch = (D + 1) ** 2
    per_g = 3 * per_ch
    print(f"{t:>6} | {D:>2} | {per_ch:>8} | {per_g:>8} | {per_g*N:>9} | {3:>3} | {per_g-3:>15}")
print("Trần: 16 hệ số/kênh x 3 = 48 float SH/Gaussian (3 dc + 45 rest); tổng 59 tham số.")

# ============================================================================
# 2.6 Đầu ra của khối tại t=0
# ============================================================================
hdr("[2.6] Đầu ra của khối tại t=0 (D=0): mu_i, Sigma_i (6), alpha_i, c_i bậc 0")
print(f"{'i':>2} | {'mu':>24} | {'Sigma (xx,xy,xz,yy,yz,zz)':>60} | {'alpha':>6} | c (bậc 0)")
for i in range(N):
    print(f"{i+1:>2} | {str(mu[i]):>24} | {str(cov6[i]):>60} | {alpha[i,0]:6.4f} | {c0[i]}")
print("s_i (dùng ở chương 3 khi cần):", s[:, 0])
