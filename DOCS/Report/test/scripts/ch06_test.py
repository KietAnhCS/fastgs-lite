"""
Bai test so chuong 6 -- Gradient Flow (backward qua blend, gradient 4 cot,
tich luy thong ke ADC, Adam, lich step thua dan).
Chi dung numpy (moi truong khong co torch). MOI dao ham giai tich deu duoc
kiem chung bang sai phan huu han trung tam (central finite difference).

Cong thuc chep tu code, ghi so dong:
  - submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu:79-118  computeCov2D
  - .../forward.cu:207-244   p_proj, conic, ndc2Pix ; auxiliary.h:45-48  ndc2Pix
  - .../forward.cu:429-430   pixel_colors (KHONG co nen) / out_color = C + T*bg
  - .../backward.cu:549-597  renderCUDA backward (dL_dalpha 3 so hang, dG_ddel, 4 cot mean2D)
  - scene/gaussian_model.py:167-181  training_setup (lr 6 nhom, eps=1e-15)
  - scene/gaussian_model.py:190-209  optimizer_step ; train.py:161-162  `if iteration < opt.iterations`
  - scene/gaussian_model.py:494-497  add_densification_stats
  - utils/general_utils.py:29-50     get_expon_lr_func
  - arguments/__init__.py:76-93      lr mac dinh

Renderer numpy toi gian: lay tu chuong 3-4 (00-scene.md, 03-projection.md, 04-rasterizer.md).
Chay:  python DOCS/Report/test/scripts/ch06_test.py
"""
import numpy as np

np.set_printoptions(precision=10, suppress=True, linewidth=140)


def fmt(x, sig=4):
    return f"{x:.{sig}g}"


def relerr(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ============================================================================
# 0. Canh do choi (DOCS/Report/test/00-scene.md)
# ============================================================================
W, H = 48, 32
TANX, TANY = 0.6, 0.4
FX, FY = W / (2 * TANX), H / (2 * TANY)          # 40, 40
ZNEAR, ZFAR = 0.01, 100.0
BG = np.array([1.0, 1.0, 1.0])                   # nen trang
CAMS = np.array([[0.0, 0.0, -4.0], [1.5, 0.0, -4.0], [-1.5, 0.5, -4.0]])
P_SFM = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.5], [-0.4, -0.2, 1.0], [0.3, -0.5, 0.2]])
C_SFM = np.array([[0.8, 0.2, 0.2], [0.2, 0.7, 0.3], [0.1, 0.3, 0.9], [0.5, 0.5, 0.5]])
N = 4
ALPHA_MODEL, ALPHA_GT = 0.1, 0.9
TAU_GRAD, TAU_ABS = 2e-4, 1.2e-3
LOWPASS, CLAMP = 0.3, 1.3
ALPHA_MIN, T_STOP = 1.0 / 255.0, 1e-4

# --- Khoi tao (lay tu chuong 1-2): mu = p, s = sqrt(mean d^2 toi 3 lang gieng), R = I, Sigma = s^2 I
D2 = ((P_SFM[:, None, :] - P_SFM[None, :, :]) ** 2).sum(-1)
d2_knn = np.array([np.sort(D2[i])[1:4].mean() for i in range(N)])
S_INIT = np.sqrt(d2_knn)
SIGMA3D = np.array([np.eye(3) * s * s for s in S_INIT])

# --- extent (chuong 1, getNerfppNorm): 1.1 * max ||c_v - c_bar||
c_bar = CAMS.mean(0)
EXTENT = 1.1 * np.linalg.norm(CAMS - c_bar, axis=1).max()

print("=" * 78)
print("0. Canh do choi & khoi tao (lay tu chuong 1-4)")
print("=" * 78)
print(f"  W x H = {W} x {H}, fx = fy = {FX:g}, bg = {BG}")
for i in range(N):
    print(f"  Gaussian {i+1}: mu = {P_SFM[i]}, s = {S_INIT[i]:.10f}, c = {C_SFM[i]}")
print(f"  c_bar = {c_bar}, extent = 1.1 * {np.linalg.norm(CAMS - c_bar, axis=1).max():.10f} = {EXTENT:.10f}")


# ============================================================================
# Renderer numpy (chuong 3-4)
# ============================================================================
def proj_matrix():
    """utils/graphics_utils.py:51-71 getProjectionMatrix (z_sign = 1)."""
    top = TANY * ZNEAR; right = TANX * ZNEAR
    P = np.zeros((4, 4))
    P[0, 0] = 2 * ZNEAR / (2 * right)
    P[1, 1] = 2 * ZNEAR / (2 * top)
    P[3, 2] = 1.0
    P[2, 2] = ZFAR / (ZFAR - ZNEAR)
    P[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    return P


PROJ = proj_matrix()


def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def project(cam_idx, mu, sigma3d):
    """Tra ve dict: t (view), depth, ndc (2,), mu_pix (2,), cov2d (2x2), conic (A,B,C)."""
    t = mu - CAMS[cam_idx]                         # W_v = [I | -c_v]
    p_hom = PROJ @ np.append(t, 1.0)
    p_w = 1.0 / (p_hom[3] + 1e-7)
    ndc = p_hom[:2] * p_w
    mu_pix = np.array([ndc2pix(ndc[0], W), ndc2pix(ndc[1], H)])
    # computeCov2D: clamp, J, Sigma' = J Sigma J^T + 0.3 I  (W = I)
    limx, limy = CLAMP * TANX, CLAMP * TANY
    tx = np.clip(t[0] / t[2], -limx, limx) * t[2]
    ty = np.clip(t[1] / t[2], -limy, limy) * t[2]
    tz = t[2]
    J = np.array([[FX / tz, 0.0, -FX * tx / (tz * tz)],
                  [0.0, FY / tz, -FY * ty / (tz * tz)]])
    cov = J @ sigma3d @ J.T + LOWPASS * np.eye(2)
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    conic = np.array([cov[1, 1] / det, -cov[0, 1] / det, cov[0, 0] / det])
    mid = 0.5 * (cov[0, 0] + cov[1, 1])
    lam1 = mid + np.sqrt(max(0.1, mid * mid - det))
    radius = np.ceil(3.0 * np.sqrt(lam1))
    return dict(t=t, depth=t[2], ndc=ndc, mu_pix=mu_pix, cov=cov, conic=conic, radius=radius)


def blend_pixel(items, bg=BG):
    """items: list (c, alpha_pix) theo depth. Tra ve C_nobg, T_final, (khong xet cua loai)."""
    T = 1.0; C = np.zeros(3)
    for c, a in items:
        C = C + c * a * T
        T = T * (1 - a)
    return C, T


def render(cam_idx, ndc_all, conic_all, alpha_all, colors, depth_all, record=False):
    """Render toan anh theo chuong 4: d = mu' - x, 3 cua loai, front-to-back, nen trang.
    ndc_all: (N,2) toa do NDC cua tam (de FD theo NDC nhu code, ddelx_dx = W/2).
    Tra ve img (H,W,3) va (neu record) danh sach contributor moi pixel."""
    order = np.argsort(depth_all)
    mu_pix = np.stack([ndc2pix(ndc_all[:, 0], W), ndc2pix(ndc_all[:, 1], H)], 1)
    img = np.zeros((H, W, 3))
    contribs = [[None] * W for _ in range(H)] if record else None
    for py in range(H):
        for px in range(W):
            T = 1.0; C = np.zeros(3); lst = []
            for n in order:
                A, B, Cc = conic_all[n]
                du, dv = mu_pix[n, 0] - px, mu_pix[n, 1] - py
                power = -0.5 * (A * du * du + Cc * dv * dv) - B * du * dv
                if power > 0:
                    continue
                G = np.exp(power)
                a = min(0.99, alpha_all[n] * G)
                if a < ALPHA_MIN:
                    continue
                test_T = T * (1 - a)
                if test_T < T_STOP:
                    break
                C = C + colors[n] * a * T
                if record:
                    lst.append(dict(n=n, du=du, dv=dv, G=G, a=a, T=T))
                T = test_T
            img[py, px] = C + T * BG
            if record:
                contribs[py][px] = lst
    return img, contribs


def scene_params(cam_idx, alpha):
    prj = [project(cam_idx, P_SFM[i], SIGMA3D[i]) for i in range(N)]
    ndc = np.array([p["ndc"] for p in prj])
    conic = np.array([p["conic"] for p in prj])
    depth = np.array([p["depth"] for p in prj])
    return prj, ndc, conic, depth, np.full(N, alpha), C_SFM.copy()


def l1_loss(img, gt):
    return np.abs(img - gt).mean()          # = sum |.| / (3 H W)


# ============================================================================
# Render camera 1: GT (alpha 0.9) va mo hinh (alpha 0.1)
# ============================================================================
print("\n" + "=" * 78)
print("Chieu + render camera 1 (lay tu chuong 3-4)")
print("=" * 78)
prj1, ndc1, conic1, depth1, alpha1, col1 = scene_params(0, ALPHA_MODEL)
for i, p in enumerate(prj1):
    print(f"  G{i+1}: t = {p['t']}, ndc = {p['ndc']}, mu' = {p['mu_pix']}, "
          f"Sigma' = [{p['cov'][0,0]:.6f} {p['cov'][0,1]:.6f} {p['cov'][1,1]:.6f}], "
          f"conic (A,B,C) = {p['conic']}, radius = {p['radius']:g}")
_, ndc_gt, conic_gt, depth_gt, alpha_gt, _ = scene_params(0, ALPHA_GT)
GT1, _ = render(0, ndc_gt, conic_gt, alpha_gt, C_SFM, depth_gt)
IMG1, CONTRIB1 = render(0, ndc1, conic1, alpha1, col1, depth1, record=True)
L1_0 = l1_loss(IMG1, GT1)
print(f"  L1(I_rend, I_gt) camera 1 = {L1_0:.10f}")
dL_dC_img = np.sign(IMG1 - GT1) / (3 * H * W)       # chuong 5: dL/dC = sign(.)/(3HW)

# ============================================================================
# 6.1 Backward qua blend tai MOT pixel co >= 2 Gaussian dong gop
# ============================================================================
print("\n" + "=" * 78)
print("6.1 Backward qua blend tai mot pixel")
print("=" * 78)
# tim pixel co nhieu contributor nhat, uu tien gan tam G1
best = None
mu1 = prj1[0]["mu_pix"]
for py in range(H):
    for px in range(W):
        k = len(CONTRIB1[py][px])
        dist = np.hypot(px - mu1[0], py - mu1[1])
        key = (-k, dist)
        if best is None or key < best[0]:
            best = (key, px, py)
_, PX, PY = best
lst = CONTRIB1[PY][PX]
print(f"  Pixel chon x = ({PX}, {PY}), so contributor = {len(lst)} "
      f"(pixel co nhieu contributor nhat, gan tam G1 = {mu1})")
print(f"  I_rend(x) = {IMG1[PY, PX]}, I_gt(x) = {GT1[PY, PX]}, dL/dC = {dL_dC_img[PY, PX]}")
items = [(col1[e["n"]], e["a"]) for e in lst]
C_tot, T_final = blend_pixel(items)
print(f"  C^tot (khong nen) = {C_tot}, T_final = {T_final:.10f}, C(x) = C^tot + T_final*bg = {C_tot + T_final*BG}")
print("\n  Bang contributor theo depth:")
print("  | n | Gaussian | d=(mu'-x) | G | alpha_n(x) | T_n | C^{<=n} |")
Cle = np.zeros(3); rows_61 = []
for j, e in enumerate(lst):
    Cle = Cle + col1[e["n"]] * e["a"] * e["T"]
    rows_61.append((j, e, Cle.copy()))
    print(f"  | {j+1} | G{e['n']+1} | ({e['du']:.4f},{e['dv']:.4f}) | {e['G']:.6f} | {e['a']:.6f} | {e['T']:.6f} | {Cle} |")

# dC/dalpha_n theo cong thuc hieu hai tien to (3 so hang) + FD
print("\n  dC_ch/dalpha_n = c_n T_n - (C^tot - C^{<=n})/(1-a_n) - T_final*bg/(1-a_n)   [backward.cu:559-568]")
H_FD = 1e-6
dL_dalpha_pix = []
for j, e, Cle in rows_61:
    n = e["n"]; a = e["a"]; T = e["T"]
    term1 = col1[n] * T
    term2 = -(C_tot - Cle) / (1 - a)
    term3 = -T_final * BG / (1 - a)
    dC = term1 + term2 + term3
    # FD: nhieu alpha_n(x) tai pixel nay
    it_p = list(items); it_p[j] = (col1[n], a + H_FD)
    it_m = list(items); it_m[j] = (col1[n], a - H_FD)
    Cp, Tp = blend_pixel(it_p); Cm, Tm = blend_pixel(it_m)
    dC_fd = ((Cp + Tp * BG) - (Cm + Tm * BG)) / (2 * H_FD)
    print(f"\n  n={j+1} (G{n+1}): term1 c_n T_n = {term1}")
    print(f"           term2 -(C^tot-C^<=n)/(1-a) = {term2}")
    print(f"           term3 -T_final*bg/(1-a)   = {term3}")
    print(f"     analytic dC/dalpha = {dC}")
    print(f"     finite-diff        = {dC_fd}")
    print(f"     rel.err (max ch)   = {max(relerr(dC[c], dC_fd[c]) for c in range(3)):.3e}")
    dLda = float(dL_dC_img[PY, PX] @ dC)
    dL_dalpha_pix.append(dLda)
    print(f"     dL/dalpha_n(x) = sum_ch dL/dC_ch * dC_ch/dalpha = {dLda:.10e}")

# dG/dDelta, dL/dA,B,C tai pixel cho tung contributor
print("\n  dG/dd_u = -G(A d_u + B d_v),  dG/dd_v = -G(C d_v + B d_u),  d = mu' - x  [backward.cu:576-580]")
print("  dL/dG = alpha_n * dL/dalpha_n(x);  dL/dmu'_u = dL/dG * dG/dd_u * W/2 ; dL/dmu'_v = ... * H/2")
print("  dL/dA = -1/2 G d_u^2 dL/dG, dL/dB = -1/2 G d_u d_v dL/dG (code: -0.5*gdx*d.y), dL/dC = -1/2 G d_v^2 dL/dG")


def pixel_loss_with_override(j, key, val):
    """Loss L1 (chia 3HW) tai pixel (PX,PY) khi thay doi 1 dai luong cua contributor j."""
    T = 1.0; C = np.zeros(3)
    for jj, e in enumerate(lst):
        n = e["n"]; A, B, Cc = conic1[n]; du, dv = e["du"], e["dv"]
        if jj == j:
            if key == "du": du = val
            elif key == "dv": dv = val
            elif key == "A": A = val
            elif key == "B": B = val
            elif key == "C": Cc = val
        power = -0.5 * (A * du * du + Cc * dv * dv) - B * du * dv
        G = np.exp(power); a = min(0.99, alpha1[n] * G)
        C = C + col1[n] * a * T; T = T * (1 - a)
    Cpix = C + T * BG
    return np.abs(Cpix - GT1[PY, PX]).sum() / (3 * H * W)


rows_grad = []
for j, e, _ in rows_61:
    n = e["n"]; A, B, Cc = conic1[n]; du, dv, G = e["du"], e["dv"], e["G"]
    dLda = dL_dalpha_pix[j]
    dL_dG = alpha1[n] * dLda
    dG_du = -G * (A * du + B * dv)
    dG_dv = -G * (Cc * dv + B * du)
    # FD cho dG/dd (G thuan tuy)
    def Gf(u, v):
        return np.exp(-0.5 * (A * u * u + Cc * v * v) - B * u * v)
    dG_du_fd = (Gf(du + H_FD, dv) - Gf(du - H_FD, dv)) / (2 * H_FD)
    dG_dv_fd = (Gf(du, dv + H_FD) - Gf(du, dv - H_FD)) / (2 * H_FD)
    dL_dA = -0.5 * G * du * du * dL_dG
    dL_dB = -0.5 * G * du * dv * dL_dG
    dL_dCc = -0.5 * G * dv * dv * dL_dG
    fd = {}
    for key, base in (("A", A), ("B", B), ("C", Cc), ("du", du), ("dv", dv)):
        fd[key] = (pixel_loss_with_override(j, key, base + H_FD) - pixel_loss_with_override(j, key, base - H_FD)) / (2 * H_FD)
    dL_du = dL_dG * dG_du; dL_dv = dL_dG * dG_dv
    print(f"\n  n={j+1} (G{n+1}): dL/dG = {dL_dG:.10e}")
    print(f"    | dai luong | analytic | finite-diff | rel.err |")
    print(f"    | dG/dd_u | {dG_du:.10e} | {dG_du_fd:.10e} | {relerr(dG_du, dG_du_fd):.2e} |")
    print(f"    | dG/dd_v | {dG_dv:.10e} | {dG_dv_fd:.10e} | {relerr(dG_dv, dG_dv_fd):.2e} |")
    print(f"    | dL/dd_u (pixel) | {dL_du:.10e} | {fd['du']:.10e} | {relerr(dL_du, fd['du']):.2e} |")
    print(f"    | dL/dd_v (pixel) | {dL_dv:.10e} | {fd['dv']:.10e} | {relerr(dL_dv, fd['dv']):.2e} |")
    print(f"    | dL/dA | {dL_dA:.10e} | {fd['A']:.10e} | {relerr(dL_dA, fd['A']):.2e} |")
    print(f"    | dL/dB (quy uoc code, -1/2 G d_u d_v) | {dL_dB:.10e} | {fd['B']:.10e} | {relerr(dL_dB, fd['B']):.2e} |")
    print(f"    | dL/dB thuc = 2 x code (power co 2B d_u d_v; backward.cu:212-214 nhan lai 2*dL_dconic.y) | {2*dL_dB:.10e} | {fd['B']:.10e} | {relerr(2*dL_dB, fd['B']):.2e} |")
    print(f"    | dL/dC | {dL_dCc:.10e} | {fd['C']:.10e} | {relerr(dL_dCc, fd['C']):.2e} |")
    print(f"    dL/dmu'_u (x) = dL/dG*dG/dd_u*W/2 = {dL_du * W / 2:.10e}; dL/dmu'_v (x) = {dL_dv * H / 2:.10e}")
    rows_grad.append((j, n, dL_dG, dG_du, dG_du_fd, dG_dv, dG_dv_fd, dL_dA, fd['A'], dL_dB, fd['B'], dL_dCc, fd['C']))


# ============================================================================
# 6.1 FastGS: gradient 4 cot cong don qua toan bo pixel (backward.cu:583-597)
# ============================================================================
def backward_mean2d(cam_idx, ndc_all, conic_all, alpha_all, colors, depth_all, gt):
    """Tra ve g (N,4): [sum tmp_x, sum tmp_y, sum|tmp_x|, sum|tmp_y|] va visible mask (co contributor)."""
    img, contribs = render(cam_idx, ndc_all, conic_all, alpha_all, colors, depth_all, record=True)
    dLdC = np.sign(img - gt) / (3 * H * W)
    g = np.zeros((N, 4)); vis = np.zeros(N, bool)
    for py in range(H):
        for px in range(W):
            lst = contribs[py][px]
            if not lst:
                continue
            items = [(colors[e["n"]], e["a"]) for e in lst]
            C_tot, T_final = blend_pixel(items)
            Cle = np.zeros(3)
            for e in lst:
                n = e["n"]; a = e["a"]; T = e["T"]
                vis[n] = True
                Cle = Cle + colors[n] * a * T
                dC = colors[n] * T - (C_tot - Cle) / (1 - a) - T_final * BG / (1 - a)
                dLda = float(dLdC[py, px] @ dC)
                dL_dG = alpha_all[n] * dLda
                A, B, Cc = conic_all[n]; du, dv, G = e["du"], e["dv"], e["G"]
                tmp_x = dL_dG * (-G * (A * du + B * dv)) * (0.5 * W)
                tmp_y = dL_dG * (-G * (Cc * dv + B * du)) * (0.5 * H)
                g[n, 0] += tmp_x; g[n, 1] += tmp_y
                g[n, 2] += abs(tmp_x); g[n, 3] += abs(tmp_y)
    return g, vis, img


print("\n" + "=" * 78)
print("6.1 FastGS: gradient 4 cot (cong don qua toan bo pixel, camera 1)")
print("=" * 78)
G1, VIS1, _ = backward_mean2d(0, ndc1, conic1, alpha1, col1, depth1, GT1)
print("  | i | g_u | g_v | g_abs_u | g_abs_v | ||g|| | ||g_abs|| | ||g|| <= ||g_abs|| |")
for i in range(N):
    ng = np.linalg.norm(G1[i, :2]); nga = np.linalg.norm(G1[i, 2:])
    print(f"  | {i+1} | {G1[i,0]:.6e} | {G1[i,1]:.6e} | {G1[i,2]:.6e} | {G1[i,3]:.6e} | {ng:.6e} | {nga:.6e} | {ng <= nga} |")

# FD kiem chung g co dau: dich ndc cua Gaussian i, render lai toan anh, L1
print("\n  Kiem chung g co dau bang FD: dich mu'_ndc cua G_i +-h (h=1e-6), render lai toan anh, L1")
print("  (code: dL_dmean2D la gradient theo NDC, ddelx_dx = W/2 ; mu'_pix = ((ndc+1)W-1)/2)")
H_FD2 = 1e-6
print("  | i | truc | analytic g | finite-diff | rel.err |")
fd_rows = []
for i in range(N):
    for ax, name in ((0, "u"), (1, "v")):
        nd_p = ndc1.copy(); nd_p[i, ax] += H_FD2
        nd_m = ndc1.copy(); nd_m[i, ax] -= H_FD2
        Lp = l1_loss(render(0, nd_p, conic1, alpha1, col1, depth1)[0], GT1)
        Lm = l1_loss(render(0, nd_m, conic1, alpha1, col1, depth1)[0], GT1)
        gfd = (Lp - Lm) / (2 * H_FD2)
        print(f"  | {i+1} | {name} | {G1[i,ax]:.10e} | {gfd:.10e} | {relerr(G1[i,ax], gfd):.2e} |")
        fd_rows.append((i, name, G1[i, ax], gfd))

# ============================================================================
# 6.3 Tich luy thong ke ADC: 3 iteration voi 3 camera
# ============================================================================
print("\n" + "=" * 78)
print("6.3 add_densification_stats: 3 iteration, camera 1,2,3 (gaussian_model.py:494-497)")
print("=" * 78)
accum = np.zeros(N); accum_abs = np.zeros(N); denom = np.zeros(N)
G_all = []
for v in range(3):
    prj, ndc, conic, depth, alpha, col = scene_params(v, ALPHA_MODEL)
    _, ndc_g, conic_g, depth_g, alpha_g, _ = scene_params(v, ALPHA_GT)
    gt, _ = render(v, ndc_g, conic_g, alpha_g, C_SFM, depth_g)
    g, vis, img = backward_mean2d(v, ndc, conic, alpha, col, depth, gt)
    radii = np.array([p["radius"] for p in prj])
    print(f"\n  Camera {v+1}: c_v = {CAMS[v]}, L1 = {l1_loss(img, gt):.10f}, radii = {radii}, visible(radii>0) = {radii > 0}")
    print("  | i | g_u | g_v | g_abs_u | g_abs_v | ||g||_2 | ||g_abs||_2 |")
    for i in range(N):
        print(f"  | {i+1} | {g[i,0]:.6e} | {g[i,1]:.6e} | {g[i,2]:.6e} | {g[i,3]:.6e} | "
              f"{np.linalg.norm(g[i,:2]):.6e} | {np.linalg.norm(g[i,2:]):.6e} |")
    upd = radii > 0
    accum[upd] += np.linalg.norm(g[upd, :2], axis=1)
    accum_abs[upd] += np.linalg.norm(g[upd, 2:], axis=1)
    denom[upd] += 1
    G_all.append(g)
gbar = accum / denom; gbar_abs = accum_abs / denom
print(f"\n  Sau 3 iteration (tau_grad = {TAU_GRAD}, tau_abs = {TAU_ABS}):")
print("  | i | accum | accum_abs | denom | g_bar | g_bar_abs | g_bar >= tau_grad | g_bar_abs >= tau_abs |")
for i in range(N):
    print(f"  | {i+1} | {accum[i]:.6e} | {accum_abs[i]:.6e} | {denom[i]:g} | {gbar[i]:.6e} | {gbar_abs[i]:.6e} | "
          f"{gbar[i] >= TAU_GRAD} | {gbar_abs[i] >= TAU_ABS} |")

# ============================================================================
# 6.4 Adam bang so cho 1 tham so
# ============================================================================
print("\n" + "=" * 78)
print("6.4 Adam: 3 buoc voi gradient lap lai g = g_bar_x(G1)  (eps = 1e-15, beta = (0.9, 0.999))")
print("=" * 78)
LR_GROUPS = {
    "xyz (t=0)": 0.00016 * EXTENT,
    "f_dc": 0.0025,
    "opacity": 0.025,
    "scaling": 0.005,
    "rotation": 0.001,
    "f_rest (0.005/20)": 0.005 / 20.0,
}
B1, B2, EPS = 0.9, 0.999, 1e-15
g_adam = float(np.mean([G_all[v][0, 0] for v in range(3)]))   # gradient x (co dau) cua G1, trung binh 3 camera
print(f"  g = mean_v g_u(G1) = {g_adam:.10e}  (dau: {'+' if g_adam > 0 else '-'})")
m = v_ = 0.0
print("  | k | m | v | m_hat | v_hat | m_hat/(sqrt(v_hat)+eps) |")
ratios = []
for k in range(1, 4):
    m = B1 * m + (1 - B1) * g_adam
    v_ = B2 * v_ + (1 - B2) * g_adam ** 2
    mh = m / (1 - B1 ** k); vh = v_ / (1 - B2 ** k)
    r = mh / (np.sqrt(vh) + EPS)
    ratios.append(r)
    print(f"  | {k} | {m:.10e} | {v_:.10e} | {mh:.10e} | {vh:.10e} | {r:.10f} |")
print("\n  Delta_theta = eta * m_hat/(sqrt(v_hat)+eps) cho tung nhom lr (buoc 1,2,3):")
print("  | nhom | eta | Delta_theta k=1 | k=2 | k=3 | |Delta|/eta |")
for name, lr in LR_GROUPS.items():
    d = [lr * r for r in ratios]
    print(f"  | {name} | {lr:.6e} | {d[0]:.6e} | {d[1]:.6e} | {d[2]:.6e} | {abs(d[0])/lr:.6f} |")
print("  Thu voi g nho hon 1000 lan:")
gs = g_adam / 1000
m = v_ = 0.0
for k in range(1, 4):
    m = B1 * m + (1 - B1) * gs; v_ = B2 * v_ + (1 - B2) * gs ** 2
r = (m / (1 - B1 ** 3)) / (np.sqrt(v_ / (1 - B2 ** 3)) + EPS)
print(f"    g' = {gs:.6e} -> m_hat/(sqrt(v_hat)+eps) = {r:.10f}  => buoc di van ~ eta, khong phu thuoc |g|")

# ============================================================================
# 6.4 Decay eta_xyz(t)
# ============================================================================
print("\n" + "=" * 78)
print("6.4 Decay eta_xyz(t) = exp((1-t/T) ln(lr_init) + (t/T) ln(lr_final)), T = 30000 (general_utils.py:29-50)")
print("=" * 78)
LR_INIT = 0.00016 * EXTENT; LR_FINAL = 0.0000016 * EXTENT


def expon_lr(step, lr_init=LR_INIT, lr_final=LR_FINAL, max_steps=30000):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    t = np.clip(step / max_steps, 0, 1)
    return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)


print(f"  lr_init = 0.00016 * {EXTENT:.10f} = {LR_INIT:.10e}, lr_final = 0.0000016 * extent = {LR_FINAL:.10e}")
print("  | t | t/T | eta_xyz(t) | eta/lr_init |")
for t in (0, 7500, 15000, 30000):
    e = expon_lr(t)
    print(f"  | {t} | {t/30000:.4f} | {e:.10e} | {e/LR_INIT:.6f} |")

# ============================================================================
# 6.4 Lich step: dem bang vong lap that theo optimizer_step (gaussian_model.py:190-209)
# ============================================================================
print("\n" + "=" * 78)
print("6.4 Lich step thua dan: dem vong lap t = 1..29999 (train.py:161 `if iteration < opt.iterations`)")
print("=" * 78)


def optimizer_step_flags(it):
    """Chep dung dieu kien gaussian_model.py:190-209. Tra ve (main_step, sh_step)."""
    if it <= 15000:
        return True, (it % 16 == 0)
    elif it <= 20000:
        return (it % 32 == 0), (it % 32 == 0)
    else:
        return (it % 64 == 0), (it % 64 == 0)


ITERS = 30000
n_main = n_sh = 0
n_main_seg = {"<=15000": 0, "(15000,20000]": 0, ">20000": 0}
n_sh_seg = {"<=15000": 0, "(15000,20000]": 0, ">20000": 0}
for it in range(1, ITERS + 1):
    if it < ITERS:                       # train.py:161
        ms, ss = optimizer_step_flags(it)
        seg = "<=15000" if it <= 15000 else ("(15000,20000]" if it <= 20000 else ">20000")
        n_main += ms; n_sh += ss
        n_main_seg[seg] += ms; n_sh_seg[seg] += ss
print(f"  optimizer  : {n_main_seg}  tong = {n_main}")
print(f"  shoptimizer: {n_sh_seg}  tong = {n_sh}")
print(f"  tong 2 optimizer = {n_main + n_sh};  3DGS goc = 29999 + 29999 = 59998; ti so = {(n_main+n_sh)/59998:.4f}")
print(f"  So boi cua 32 trong (15000, 20000] = {n_main_seg['(15000,20000]']}  (20000 = 32*625, 15000/32 = {15000/32:.3f})")
print(f"  So boi cua 64 trong (20000, 29999] = {n_main_seg['>20000']}")
print(f"  Ky vong tai lieu: 15313 / 1250 / 16563 / 157 -> code cho: {n_main} / {n_sh} / {n_main+n_sh} / {n_main_seg['(15000,20000]']}")

# ============================================================================
# 6.4 g_eff: cong 64 gradient roi 1 step Adam vs 64 step nho
# ============================================================================
print("\n" + "=" * 78)
print("6.4 g_eff = sum 64 gradient roi step 1 lan  vs  64 step nho (cung m, v ban dau = 0)")
print("=" * 78)
rng = np.random.default_rng(0)
g_seq = g_adam * (1.0 + 0.3 * rng.standard_normal(64))     # 64 gradient cung dau, cung co
eta = LR_GROUPS["xyz (t=0)"]


def adam_run(grads, lr):
    m = v_ = 0.0; total = 0.0
    for k, g in enumerate(grads, 1):
        m = B1 * m + (1 - B1) * g; v_ = B2 * v_ + (1 - B2) * g * g
        total -= lr * (m / (1 - B1 ** k)) / (np.sqrt(v_ / (1 - B2 ** k)) + EPS)
    return total


d_one = adam_run([g_seq.sum()], eta)
d_64 = adam_run(g_seq, eta)
print(f"  eta = eta_xyz = {eta:.6e}, sum g = {g_seq.sum():.6e} (|g| trung binh {np.abs(g_seq).mean():.6e})")
print(f"  1 step voi g_eff = sum: Delta_theta = {d_one:.10e}  (= -eta * sign(g) chinh xac: {d_one/eta:.6f} eta)")
print(f"  64 step nho          : Delta_theta = {d_64:.10e}  (= {d_64/eta:.4f} eta)")
print(f"  ti so 64 step / 1 step = {d_64/d_one:.4f}")

# ============================================================================
# Dau ra cua khoi (cho chuong 7)
# ============================================================================
print("\n" + "=" * 78)
print("DAU RA CUA KHOI (chuong 7 dung)")
print("=" * 78)
print("  | i | accum | accum_abs | denom | g_bar | g_bar_abs | >= tau_grad | >= tau_abs |")
for i in range(N):
    print(f"  | {i+1} | {accum[i]:.6e} | {accum_abs[i]:.6e} | {denom[i]:g} | {gbar[i]:.6e} | {gbar_abs[i]:.6e} | "
          f"{gbar[i] >= TAU_GRAD} | {gbar_abs[i] >= TAU_ABS} |")
