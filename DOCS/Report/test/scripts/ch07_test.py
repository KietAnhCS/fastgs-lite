"""
Test số cho Chương 7 — Adaptive Density Control (DOCS/Report/07-adaptive-density-control.md)
Cảnh đồ chơi: DOCS/Report/test/00-scene.md (4 điểm SfM, 3 camera, ảnh 48x32, fx=fy=40).

Chỉ dùng numpy + scipy (không có torch). Seed RNG = 0.

Các công thức được CHÉP LẠI từ:
  - scene/gaussian_model.py:137-160      : create_from_pcd (khởi tạo chương 1, tính lại tại đây)
  - cuda_rasterizer/forward.cu:79-118    : computeCov2D  (J clamp 1.3 tan, +0.3 low-pass)
  - cuda_rasterizer/forward.cu:225-242   : conic, r2D = ceil(3 sqrt(lambda_max)), ndc2Pix
  - cuda_rasterizer/auxiliary.h:295-330  : compact box t = mult*2*ln(255 alpha), mult = 0.5 (cull tile)
  - cuda_rasterizer/forward.cu:376-415   : renderCUDA: 3 cửa loại (alpha<1/255, T<1e-4, tile) + đếm
                                           metricCount[i] += 1 khi Gaussian i THỰC SỰ đóng góp vào pixel
                                           có metric_map == 1 (cờ DENSIFY / get_flag)
  - utils/loss_utils.py                  : l1_loss, ssim (gaussian 11x11 sigma=1.5, zero-pad, C1=1e-4, C2=9e-4)
  - utils/fast_utils.py:33-90            : compute_gaussian_score_fastgs (7 bước)
  - scene/gaussian_model.py:433-486      : densify_and_prune_fastgs
  - scene/gaussian_model.py:498-504      : final_prune_fastgs
  - scene/gaussian_model.py:494-496      : add_densification_stats (norm cột 0-1 và cột 2-3)
  - train.py:127-158                     : lịch chạy
  - scene/dataset_readers.py getNerfppNorm : extent = 1.1 * max ||c_v - c_bar||

Renderer numpy tối giản (lấy từ chương 3–5): khởi tạo mu = p, s = sqrt(mean d^2 tới 3 láng giềng),
R = I, Sigma = s^2 I, màu = c_k (SH bậc 0, không dùng SH bậc cao vì ta chỉ cần màu hằng);
projection mỗi camera: t = mu - c_v; mu' = ndc2Pix; J clamp 1.3 tan; Sigma' = J Sigma J^T + 0.3 I;
conic; render mọi Gaussian theo depth với 3 cửa loại, nền trắng. Renderer trả thêm footprint hữu hình
Omega_i^{(v)} = tập pixel mà Gaussian i thực sự đóng góp (qua được 3 cửa loại).

Chạy:  python DOCS/Report/test/scripts/ch07_test.py
"""
import numpy as np
from scipy.ndimage import convolve

np.set_printoptions(precision=6, suppress=True, linewidth=140)
SEED = 0

# ----------------------------------------------------------------------------
# Hằng số (chương 0)
# ----------------------------------------------------------------------------
W, H = 48, 32
TANX, TANY = 0.6, 0.4
FX, FY = W / (2 * TANX), H / (2 * TANY)          # 40, 40
BLOCK = 16
GRID_X, GRID_Y = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK   # 3 x 2 tile
LOWPASS = 0.3
ALPHA_MIN = 1.0 / 255.0
T_EARLY = 1e-4
MULT = 0.5
TAU_GRAD = 2e-4
TAU_GRAD_ABS = 1.2e-3
TAU_LOSS = 0.1
DELTA = 0.001
LAMBDA = 0.2
V_CAMS = 3       # 3 camera thay vì V=10 của sampling_cameras (fast_utils.py:13) — cảnh chỉ có 3 camera

# Cảnh đồ chơi (00-scene.md)
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
BG = np.ones(3)  # nền trắng


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


# ============================================================================
# Khởi tạo (chương 1/2 tính lại)
# ============================================================================
hdr("[CH1-2 tính lại] Khởi tạo: mu = p, s = sqrt(mean d^2 3-NN), R = I, Sigma = s^2 I, màu = c_k")
D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
dist2 = np.zeros(N)
for i in range(N):
    dist2[i] = np.sort(np.delete(D2[i], i))[:3].mean()
dist2 = np.maximum(dist2, 1e-7)
S = np.sqrt(dist2)[:, None].repeat(3, 1)          # scale (đẳng hướng)
S_TILDE = np.log(S)
R_ROT = np.tile(np.eye(3), (N, 1, 1))              # q = (1,0,0,0) -> R = I
SIGMA3 = np.array([np.diag(S[i] ** 2) for i in range(N)])
ALPHA_MODEL = 0.1 * np.ones(N)
ALPHA_GT = 0.9 * np.ones(N)
for i in range(N):
    print(f"G{i+1}: mu={P[i]}  s={S[i,0]:.6f}  s~={S_TILDE[i,0]:.6f}  alpha_model=0.1  alpha_gt=0.9  c={COL[i]}")


# ============================================================================
# Projection (chương 3) — chép từ forward.cu computeCov2D / preprocessCUDA
# ============================================================================
def ndc2Pix(v, S_):
    return ((v + 1.0) * S_ - 1.0) * 0.5


def project(mu, Sigma, cam):
    """Trả về mu' (pixel), Sigma' (2x2), conic (A,B,C), depth, r2D, và t (camera space)."""
    t = mu - cam                                   # R_cam = I  -> t = mu - c_v
    tx, ty, tz = t
    # NDC theo getProjectionMatrix: x_ndc = tx / (tanx * tz)
    ndc = np.array([tx / (TANX * tz), ty / (TANY * tz)])
    mu2 = np.array([ndc2Pix(ndc[0], W), ndc2Pix(ndc[1], H)])
    # computeCov2D
    limx, limy = 1.3 * TANX, 1.3 * TANY
    txc = min(limx, max(-limx, tx / tz)) * tz
    tyc = min(limy, max(-limy, ty / tz)) * tz
    J = np.array([[FX / tz, 0.0, -(FX * txc) / (tz * tz)],
                  [0.0, FY / tz, -(FY * tyc) / (tz * tz)]])
    Sig2 = J @ Sigma @ J.T
    Sig2[0, 0] += LOWPASS
    Sig2[1, 1] += LOWPASS
    det = Sig2[0, 0] * Sig2[1, 1] - Sig2[0, 1] ** 2
    conic = np.array([Sig2[1, 1] / det, -Sig2[0, 1] / det, Sig2[0, 0] / det])
    mid = 0.5 * (Sig2[0, 0] + Sig2[1, 1])
    lam1 = mid + np.sqrt(max(0.1, mid * mid - det))
    lam2 = mid - np.sqrt(max(0.1, mid * mid - det))
    r2d = int(np.ceil(3.0 * np.sqrt(max(lam1, lam2))))
    return mu2, Sig2, conic, tz, r2d, t


def tiles_touched(mu2, conic, alpha):
    """Compact box (auxiliary.h:295-330, SnugBox): Gaussian chỉ được gán vào tile giao với ellipse
       Delta^T M Delta <= t_i,  t_i = mult * 2 ln(255 alpha).  Ở đây kiểm tra giao tile–ellipse chính xác
       bằng cách lấy min của dạng toàn phương lồi trên hình chữ nhật tile (biên + tâm)."""
    A, B, C = conic
    t_lvl = MULT * 2.0 * np.log(alpha * 255.0)
    if t_lvl <= 0:
        return np.zeros((GRID_Y, GRID_X), dtype=bool)

    def q(dx, dy):
        return A * dx * dx + 2 * B * dx * dy + C * dy * dy

    def min_on_rect(x0, x1, y0, y1):
        # tâm ellipse nằm trong tile
        if x0 <= mu2[0] <= x1 and y0 <= mu2[1] <= y1:
            return 0.0
        best = np.inf
        # 4 cạnh: cố định x hoặc y, tối thiểu 1D rồi clamp
        for xf in (x0, x1):
            dx = xf - mu2[0]
            dy_star = -B * dx / C
            dy = min(max(dy_star, y0 - mu2[1]), y1 - mu2[1])
            best = min(best, q(dx, dy))
        for yf in (y0, y1):
            dy = yf - mu2[1]
            dx_star = -B * dy / A
            dx = min(max(dx_star, x0 - mu2[0]), x1 - mu2[0])
            best = min(best, q(dx, dy))
        return best

    out = np.zeros((GRID_Y, GRID_X), dtype=bool)
    for ty in range(GRID_Y):
        for tx in range(GRID_X):
            x0, x1 = tx * BLOCK, min((tx + 1) * BLOCK, W) - 1
            y0, y1 = ty * BLOCK, min((ty + 1) * BLOCK, H) - 1
            out[ty, tx] = min_on_rect(x0, x1, y0, y1) <= t_lvl
    return out


# ============================================================================
# Rasterizer (chương 4) — renderCUDA với 3 cửa loại + footprint hữu hình
# ============================================================================
PIX_X, PIX_Y = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
TILE_OF_PIX = (PIX_Y // BLOCK).astype(int), (PIX_X // BLOCK).astype(int)


def prep_view(cam, alpha_vec, mu_override=None):
    """Tiền xử lý mọi Gaussian cho 1 camera. mu_override: dict {i: mu2' (pixel)} để sai phân hữu hạn."""
    items = []
    for i in range(N):
        mu2, Sig2, conic, depth, r2d, t = project(P[i], SIGMA3[i], cam)
        if mu_override is not None and i in mu_override:
            mu2 = mu_override[i]
        tiles = tiles_touched(mu2, conic, alpha_vec[i])
        items.append(dict(i=i, mu2=mu2, Sig2=Sig2, conic=conic, depth=depth, r2d=r2d,
                          tiles=tiles, alpha=alpha_vec[i], color=COL[i], t=t))
    return items


def render(items):
    """Trả về ảnh [3,H,W], footprint bool [N,H,W], T_final."""
    order = sorted(range(N), key=lambda k: items[k]["depth"])   # sort theo depth
    C = np.zeros((3, H, W))
    T = np.ones((H, W))
    done = np.zeros((H, W), dtype=bool)
    foot = np.zeros((N, H, W), dtype=bool)
    for k in order:
        g = items[k]
        A, B, Cc = g["conic"]
        dx = g["mu2"][0] - PIX_X
        dy = g["mu2"][1] - PIX_Y
        power = -0.5 * (A * dx * dx + Cc * dy * dy) - B * dx * dy
        alpha = np.minimum(0.99, g["alpha"] * np.exp(power))
        in_tile = g["tiles"][TILE_OF_PIX]                       # cửa 1: cull tile
        active = (~done) & in_tile & (power <= 0) & (alpha >= ALPHA_MIN)   # cửa 2: alpha >= 1/255
        test_T = T * (1 - alpha)
        newly_done = active & (test_T < T_EARLY)                # cửa 3: T < 1e-4 -> dừng
        contrib = active & ~newly_done
        for ch in range(3):
            C[ch] += np.where(contrib, g["color"][ch] * alpha * T, 0.0)
        T = np.where(contrib, test_T, T)
        done |= newly_done
        foot[g["i"]] = contrib
    C += T[None] * BG[:, None, None]                            # nền trắng
    return C, foot, T


# ============================================================================
# Loss (chương 5) — chép utils/loss_utils.py
# ============================================================================
def gaussian_1d(window_size=11, sigma=1.5):
    g = np.array([np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
    return g / g.sum()


WIN = np.outer(gaussian_1d(), gaussian_1d())       # 11x11


def conv(img):  # zero-pad như F.conv2d(padding=5)
    return convolve(img, WIN, mode="constant", cval=0.0)


def ssim(img1, img2):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        a, b = img1[ch], img2[ch]
        mu1, mu2 = conv(a), conv(b)
        s1 = conv(a * a) - mu1 ** 2
        s2 = conv(b * b) - mu2 ** 2
        s12 = conv(a * b) - mu1 * mu2
        m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        vals.append(m)
    return float(np.mean(vals))


def l1(img1, img2):
    return float(np.abs(img1 - img2).mean())


# ============================================================================
# Render GT (alpha=0.9) và mô hình (alpha=0.1) cho 3 camera
# ============================================================================
hdr("[Render] GT (alpha=0.9) và mô hình (alpha=0.1), 3 camera — projection + footprint hữu hình")
GT, REND, FOOT, ITEMS = [], [], [], []
for v in range(V_CAMS):
    it_gt = prep_view(CAM[v], ALPHA_GT)
    it_md = prep_view(CAM[v], ALPHA_MODEL)
    img_gt, _, _ = render(it_gt)
    img_md, foot, Tf = render(it_md)
    GT.append(img_gt); REND.append(img_md); FOOT.append(foot); ITEMS.append(it_md)
    print(f"\n-- camera {v+1}, c_v={CAM[v]} --")
    for g in it_md:
        i = g["i"]
        print(f"G{i+1}: t={g['t']}  mu'=({g['mu2'][0]:.4f},{g['mu2'][1]:.4f})  "
              f"Sigma'=[[{g['Sig2'][0,0]:.4f},{g['Sig2'][0,1]:.4f}],[.,{g['Sig2'][1,1]:.4f}]]  "
              f"conic=({g['conic'][0]:.5f},{g['conic'][1]:.5f},{g['conic'][2]:.5f})  depth={g['depth']:.3f}  "
              f"r2D={g['r2d']}  tiles={g['tiles'].sum()}/6  |Omega|={int(foot[i].sum())}px")
    print(f"T_final: min={Tf.min():.4f} max={Tf.max():.4f}")

# ============================================================================
# 7.2 — Bảy bước của compute_gaussian_score_fastgs
# ============================================================================
hdr("[7.2 ①] e_v(x) = (1/3) sum_ch |I_rend - I_gt|  (fast_utils.py:22 get_loss)")
E, EHAT, MASK = [], [], []
for v in range(V_CAMS):
    e = np.abs(REND[v] - GT[v]).mean(0)
    E.append(e)
    print(f"view {v+1}: min={e.min():.6f}  max={e.max():.6f}  mean={e.mean():.6f}")

hdr("[7.2 ②] minmax theo ảnh: e_hat = (e - min)/(max - min)")
for v in range(V_CAMS):
    e = E[v]
    eh = (e - e.min()) / (e.max() - e.min())
    EHAT.append(eh)
    print(f"view {v+1}: mẫu số = {e.max()-e.min():.6f}; ngưỡng e tương đương với tau=0.1: "
          f"e > {e.min() + 0.1*(e.max()-e.min()):.6f}")

hdr("[7.2 ③] m_v(x) = 1[e_hat > 0.1]  (metric_map)")
for v in range(V_CAMS):
    m = (EHAT[v] > TAU_LOSS).astype(int)
    MASK.append(m)
    print(f"view {v+1}: số pixel bật = {int(m.sum())}/{H*W}")
    # in bản đồ mask thu gọn
    for row in range(H):
        print("   " + "".join("#" if m[row, c] else "." for c in range(W)))

hdr("[7.2 ④] counts_i^(v) = sum_{x in Omega_i^(v)} m_v(x)  (forward.cu:406-408 atomicAdd metricCount)")
COUNTS = np.zeros((N, V_CAMS), dtype=int)
for v in range(V_CAMS):
    for i in range(N):
        COUNTS[i, v] = int((FOOT[v][i] & (MASK[v] == 1)).sum())
print(f"{'':>4} | {'view1':>6} | {'view2':>6} | {'view3':>6} | {'sum':>5}   (|Omega| = kích thước footprint)")
for i in range(N):
    om = " ".join(f"{int(FOOT[v][i].sum()):>3}" for v in range(V_CAMS))
    print(f"G{i+1:<3} | {COUNTS[i,0]:>6} | {COUNTS[i,1]:>6} | {COUNTS[i,2]:>6} | {COUNTS[i].sum():>5}   |Omega|=[{om}]")
for v in range(V_CAMS):
    print(f"kiểm tra chéo view {v+1}: sum_i counts = {COUNTS[:,v].sum()} vs số pixel bật = {MASK[v].sum()} "
          f"(khác nhau vì 1 pixel tố nhiều Gaussian)")

hdr("[7.2 ⑤] Importance_i = floor((1/V) sum_v counts)  V=3")
IMPORTANCE = COUNTS.sum(1) // V_CAMS
for i in range(N):
    print(f"G{i+1}: sum={COUNTS[i].sum():>3}  /3 = {COUNTS[i].sum()/3:.4f}  floor = {IMPORTANCE[i]}  >5? {IMPORTANCE[i] > 5}")

hdr("[7.2 ⑥] E_photo^(v) = 0.8 L1 + 0.2 (1 - SSIM)  (fast_utils.py:27-31)")
EPHOTO = np.zeros(V_CAMS)
for v in range(V_CAMS):
    L1v, Sv = l1(REND[v], GT[v]), ssim(REND[v], GT[v])
    EPHOTO[v] = (1 - LAMBDA) * L1v + LAMBDA * (1 - Sv)
    print(f"view {v+1}: L1={L1v:.6f}  SSIM={Sv:.6f}  0.8*L1={0.8*L1v:.6f}  0.2*(1-SSIM)={0.2*(1-Sv):.6f}  "
          f"E_photo={EPHOTO[v]:.6f}")

hdr("[7.2 ⑦] Pruning_i = minmax_i( sum_v counts_i^(v) * E_photo^(v) )")
RAW = (COUNTS * EPHOTO[None, :]).sum(1)
PRUNING = (RAW - RAW.min()) / (RAW.max() - RAW.min())
print(f"{'':>4} | " + " | ".join(f"{'cnt*E'+str(v+1):>10}" for v in range(V_CAMS)) + f" | {'thô':>10} | {'Pruning':>8}")
for i in range(N):
    terms = " | ".join(f"{COUNTS[i,v]*EPHOTO[v]:>10.6f}" for v in range(V_CAMS))
    print(f"G{i+1:<3} | {terms} | {RAW[i]:>10.6f} | {PRUNING[i]:>8.4f}")
print(f"min thô = {RAW.min():.6f}, max thô = {RAW.max():.6f}")

hdr("[7.2 Bảng tổng]")
print(f"{'':>4} | {'sum counts':>10} | {'Importance':>10} | {'Pruning':>8}")
for i in range(N):
    print(f"G{i+1:<3} | {COUNTS[i].sum():>10} | {IMPORTANCE[i]:>10} | {PRUNING[i]:>8.4f}")

# ============================================================================
# 7.3 — Densify
# ============================================================================
hdr("[7.3] Gradient theo mu' bằng sai phân hữu hạn (không có 06-test.md) — loss L1, dịch mu' ±0.5 px")
print("Ghi chú: code tích luỹ gradient theo toạ độ NDC của screenspace_points (backward.cu nhân W/2, H/2),")
print("nên g_ndc = g_pix * (W/2, H/2). Cột có dấu = tổng qua pixel của dL/dmu'; cột abs = tổng qua pixel")
print("của |dL/dmu'| theo từng pixel (backward.cu:589-597). add_densification_stats: accum += ||g||,")
print("accum_abs += ||g_abs||, denom += 1 cho mỗi view (3 view <=> 3 iteration).")
HSTEP = 0.5
ACCUM = np.zeros(N); ACCUM_ABS = np.zeros(N); DENOM = np.zeros(N)
GRAD_TABLE = []
for v in range(V_CAMS):
    base_items = ITEMS[v]
    for i in range(N):
        mu2 = base_items[i]["mu2"]
        g_signed = np.zeros(2); g_abs = np.zeros(2)
        for ax, scale in ((0, W / 2), (1, H / 2)):
            imgs = []
            for sgn in (+1, -1):
                mo = mu2.copy(); mo[ax] += sgn * HSTEP
                img, _, _ = render(prep_view(CAM[v], ALPHA_MODEL, mu_override={i: mo}))
                imgs.append(img)
            # per-pixel loss map l_x = mean_ch |C - gt| / (H W)   (L1 = sum_x l_x)
            lp = np.abs(imgs[0] - GT[v]).mean(0) / (H * W)
            lm = np.abs(imgs[1] - GT[v]).mean(0) / (H * W)
            dl_pix = (lp - lm) / (2 * HSTEP)                # dL_x / dmu'_pix  (map HxW)
            g_signed[ax] = dl_pix.sum() * scale             # tổng có dấu, đổi sang NDC
            g_abs[ax] = np.abs(dl_pix).sum() * scale        # tổng trị tuyệt đối từng pixel
        visible = base_items[i]["tiles"].any()
        if visible:
            ACCUM[i] += np.linalg.norm(g_signed)
            ACCUM_ABS[i] += np.linalg.norm(g_abs)
            DENOM[i] += 1
        GRAD_TABLE.append((v, i, g_signed, g_abs))
        print(f"view {v+1} G{i+1}: g=({g_signed[0]:+.3e},{g_signed[1]:+.3e}) ||g||={np.linalg.norm(g_signed):.3e}  "
              f"g_abs=({g_abs[0]:.3e},{g_abs[1]:.3e}) ||g_abs||={np.linalg.norm(g_abs):.3e}  visible={visible}")
GBAR = ACCUM / DENOM
GBAR_ABS = ACCUM_ABS / DENOM
print(f"\n{'':>4} | {'accum':>10} | {'accum_abs':>10} | {'denom':>5} | {'g_bar':>10} | {'g_bar_abs':>10} | "
      f"{'>=2e-4':>6} | {'>=1.2e-3':>8}")
for i in range(N):
    print(f"G{i+1:<3} | {ACCUM[i]:>10.3e} | {ACCUM_ABS[i]:>10.3e} | {int(DENOM[i]):>5} | {GBAR[i]:>10.3e} | "
          f"{GBAR_ABS[i]:>10.3e} | {str(GBAR[i] >= TAU_GRAD):>6} | {str(GBAR_ABS[i] >= TAU_GRAD_ABS):>8}")

hdr("[7.3] extent = 1.1 * max ||c_v - c_bar||  (getNerfppNorm)")
CBAR = CAM.mean(0)
DISTS = np.linalg.norm(CAM - CBAR, axis=1)
EXTENT = 1.1 * DISTS.max()
print(f"c_bar = {CBAR};  ||c_v - c_bar|| = {DISTS};  extent = 1.1 * {DISTS.max():.6f} = {EXTENT:.6f}")
print(f"delta*extent = {DELTA*EXTENT:.6f};  0.1*extent = {0.1*EXTENT:.6f}")

hdr("[7.3] Nhánh clone / split và phép AND với Importance > 5")
MAXS = S.max(1)
clone_q = MAXS <= DELTA * EXTENT
split_q = MAXS > DELTA * EXTENT
grad_q = GBAR >= TAU_GRAD
grad_abs_q = GBAR_ABS >= TAU_GRAD_ABS
all_clones = clone_q & grad_q
all_splits = split_q & grad_abs_q
metric = IMPORTANCE > 5
CLONE = all_clones & metric
SPLIT = all_splits & metric
print(f"{'':>4} | {'max s':>8} | {'nhánh':>6} | {'grad ok':>7} | {'3DGS gốc':>9} | {'Imp>5':>5} | {'FastGS-lite':>11}")
for i in range(N):
    branch = "clone" if clone_q[i] else "split"
    gok = grad_q[i] if clone_q[i] else grad_abs_q[i]
    orig = all_clones[i] or all_splits[i]
    res = "CLONE" if CLONE[i] else ("SPLIT" if SPLIT[i] else "chặn" if orig else "không")
    print(f"G{i+1:<3} | {MAXS[i]:>8.4f} | {branch:>6} | {str(bool(gok)):>7} | {str(bool(orig)):>9} | "
          f"{str(bool(metric[i])):>5} | {res:>11}")
print(f"3DGS gốc: clone={int(all_clones.sum())}, split={int(all_splits.sum())}; "
      f"FastGS-lite: clone={int(CLONE.sum())}, split={int(SPLIT.sum())}")

hdr("[7.3] Split cụ thể (densify_and_split_fastgs): eps ~ N(0, diag(s^2)), mu^(j) = mu + R eps, s~^(j) = log(s/1.6)")
rng = np.random.default_rng(SEED)
split_idx = np.where(SPLIT)[0]
simulated = False
if len(split_idx) == 0:
    split_idx = np.array([0]); simulated = True
    print("Không Gaussian nào đủ điều kiện split -> GIẢ LẬP split Gaussian 1.")
SPLIT_CHILDREN = []
for i in split_idx:
    stds = np.tile(S[i], (2, 1))
    eps = rng.normal(0.0, stds)                       # torch.normal(mean=0, std=stds), 2 mẫu
    new_mu = P[i][None] + (R_ROT[i] @ eps.T).T
    new_s_tilde = np.log(S[i] / (0.8 * 2))
    print(f"G{i+1}: s={S[i]}  eps^(1)={eps[0]}  eps^(2)={eps[1]}")
    for j in range(2):
        print(f"   con {j+1}: mu = {P[i]} + {eps[j]} = {new_mu[j]};  s~ = log({S[i,0]:.6f}/1.6) = {new_s_tilde[0]:.6f}"
              f"  -> s = {np.exp(new_s_tilde[0]):.6f}")
        SPLIT_CHILDREN.append((i, new_mu[j], new_s_tilde))
print("Clone (densify_and_clone_fastgs): theta_new = theta_i (sao chép nguyên).")

# ============================================================================
# 7.4 — Prune có trọng số
# ============================================================================
def multinomial_no_replacement(w, k, rng_):
    """torch.multinomial(w, k, replacement=False): rút lần lượt theo xác suất w/sum(w) trên phần còn lại."""
    w = w.astype(float).copy()
    out = []
    for _ in range(k):
        if w.sum() <= 0:          # torch báo lỗi khi hết trọng số dương; ở đây dừng
            break
        p = w / w.sum()
        j = rng_.choice(len(w), p=p)
        out.append(j)
        w[j] = 0.0
    return np.array(out, dtype=int)


hdr("[7.4] Tập ứng viên C (t>3000): alpha<0.005 ∨ r2D_max>20 ∨ max s>0.1 extent")
R2D_MAX = np.array([max(ITEMS[v][i]["r2d"] for v in range(V_CAMS)) for i in range(N)])
c_alpha = ALPHA_MODEL < 0.005
c_r2d = R2D_MAX > 20
c_ws = MAXS > 0.1 * EXTENT
CAND = c_alpha | c_r2d | c_ws
print(f"{'':>4} | {'alpha':>6} | {'<0.005':>6} | {'r2D max':>7} | {'>20':>5} | {'max s':>8} | {'>0.169':>6} | {'∈C':>4}")
for i in range(N):
    print(f"G{i+1:<3} | {ALPHA_MODEL[i]:>6.3f} | {str(bool(c_alpha[i])):>6} | {R2D_MAX[i]:>7} | {str(bool(c_r2d[i])):>5} | "
          f"{MAXS[i]:>8.4f} | {str(bool(c_ws[i])):>6} | {str(bool(CAND[i])):>4}")
print(f"t <= 3000 (size_threshold=None): C = {{alpha<0.005}} = {np.where(c_alpha)[0]+1} -> budget 0, không xoá.")
print(f"t >  3000: |C| = {int(CAND.sum())}, C = {np.where(CAND)[0]+1}")
BUDGET = int(0.5 * CAND.sum())
W_I = 1.0 / (1e-6 + (1 - PRUNING))
print(f"remove_budget = floor(0.5 * {int(CAND.sum())}) = {BUDGET}")
print(f"w_i = 1/(1e-6 + (1 - Pruning_i)) = {W_I};  xác suất rút đầu = {W_I / W_I.sum()}")
rng = np.random.default_rng(SEED)
if BUDGET > 0:
    SAMPLED = multinomial_no_replacement(W_I, BUDGET, rng)
    Smask = np.zeros(N, dtype=bool); Smask[SAMPLED] = True
    REMOVE = CAND & Smask
    print(f"S (seed 0) = {np.sort(SAMPLED)+1};  xoá = C ∩ S = {np.where(REMOVE)[0]+1}")
else:
    REMOVE = np.zeros(N, dtype=bool)
    print("budget = 0 -> không xoá.")
# tần suất trên 1000 lần cho cảnh
if BUDGET > 0:
    rng = np.random.default_rng(SEED)
    freq = np.zeros(N); nrem = []
    for _ in range(1000):
        s_ = multinomial_no_replacement(W_I, BUDGET, rng)
        m_ = np.zeros(N, dtype=bool); m_[s_] = True
        rm = CAND & m_
        freq += rm; nrem.append(rm.sum())
    print(f"1000 lần (seed 0): tần suất xoá từng Gaussian = {freq/1000};  số xoá thật TB = {np.mean(nrem):.3f} "
          f"so với budget {BUDGET}")

hdr("[7.4b] Thứ tự THẬT trong densify_and_prune_fastgs: split TRƯỚC, prune SAU trên quần thể mới")
print("Code: densify_and_split_fastgs xoá bản gốc và nối con [copy1 G1..G4, copy2 G1..G4] (repeat(2,1)),")
print("rồi prune_mask tính trên quần thể MỚI; padded_importance[:scores.shape[0]] gán trọng số cũ theo CHỈ SỐ")
print("(không đánh lại), phần dư = 0 (miễn nhiễm). max_radii2D của con = 0 (densification_postfix).")
if not simulated:
    pop = [(i, j) for j in range(2) for i in split_idx]          # thứ tự repeat(2,1)
    keep_orig = [i for i in range(N) if not SPLIT[i]]
    pop = [(i, -1) for i in keep_orig] + pop                       # bản gốc không split đứng trước
    n_pop = len(pop)
    s_pop = np.array([S[i, 0] if j < 0 else S[i, 0] / 1.6 for (i, j) in pop])
    r_pop = np.array([R2D_MAX[i] if j < 0 else 0 for (i, j) in pop])
    a_pop = np.full(n_pop, 0.1)
    cand_pop = (a_pop < 0.005) | (r_pop > 20) | (s_pop > 0.1 * EXTENT)
    budget_pop = int(0.5 * cand_pop.sum())
    w_pop = np.zeros(n_pop); w_pop[:N] = W_I
    print(f"quần thể sau split: {[f'G{i+1}' if j<0 else f'G{i+1}c{j+1}' for (i,j) in pop]}")
    print(f"max s = {s_pop};  ∈C = {cand_pop};  |C| = {int(cand_pop.sum())}, budget = {budget_pop}")
    print(f"w (theo chỉ số cũ) = {w_pop}")
    rng = np.random.default_rng(SEED)
    s_pop_idx = multinomial_no_replacement(w_pop, budget_pop, rng)
    m_pop = np.zeros(n_pop, dtype=bool); m_pop[s_pop_idx] = True
    rm_pop = cand_pop & m_pop
    print(f"S = {sorted(int(x)+1 for x in s_pop_idx)} (chỉ số 1-based trong quần thể mới) -> xoá = "
          f"{[f'G{pop[k][0]+1}c{pop[k][1]+1}' for k in np.where(rm_pop)[0]]}")
    N_CODE = n_pop - int(rm_pop.sum())
    print(f"N theo thứ tự code = {n_pop} - {int(rm_pop.sum())} = {N_CODE}")

hdr("[7.4 GIẢ ĐỊNH N=10] Pruning cho trước, C = {3,6,8,9,10}, 1000 lần seed 0")
PR10 = np.array([0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
C10 = np.zeros(10, dtype=bool); C10[[2, 5, 7, 8, 9]] = True
W10 = 1.0 / (1e-6 + (1 - PR10))
B10 = int(0.5 * C10.sum())
print(f"|C| = {int(C10.sum())}, remove_budget = floor(0.5*5) = {B10}")
print(f"{'i':>3} | {'Pruning':>8} | {'w_i':>12} | {'p_i = w/sum w':>13} | {'∈C':>3}")
for i in range(10):
    print(f"{i+1:>3} | {PR10[i]:>8.2f} | {W10[i]:>12.4f} | {W10[i]/W10.sum():>13.6f} | {str(bool(C10[i])):>3}")
rng = np.random.default_rng(SEED)
freq10 = np.zeros(10); nrem10 = []; freqS10 = np.zeros(10)
for _ in range(1000):
    s_ = multinomial_no_replacement(W10, B10, rng)
    m_ = np.zeros(10, dtype=bool); m_[s_] = True
    freqS10 += m_
    rm = C10 & m_
    freq10 += rm; nrem10.append(rm.sum())
print(f"tần suất ∈ S      : {freqS10/1000}")
print(f"tần suất bị xoá   : {freq10/1000}")
print(f"số xoá thật TB = {np.mean(nrem10):.3f} so với budget {B10} (min {min(nrem10)}, max {max(nrem10)})")

# ============================================================================
# 7.5 — Ép opacity
# ============================================================================
hdr("[7.5] alpha~ <- sigma^-1(min(alpha, 0.8)) (sau mỗi densify) và sigma^-1(min(alpha, 0.01)) (reset_opacity)")
for a in (0.1, 0.9):
    print(f"alpha={a}: logit trước = {inverse_sigmoid(a):+.6f};  (1) min(a,0.8)={min(a,0.8)} -> {inverse_sigmoid(min(a,0.8)):+.6f}"
          f";  (2) min(a,0.01)={min(a,0.01)} -> {inverse_sigmoid(min(a,0.01)):+.6f}")

# ============================================================================
# 7.6 — final_prune_fastgs
# ============================================================================
hdr("[7.6] final_prune = [alpha < 0.1] ∨ [Pruning > 0.9]")
FP = (ALPHA_MODEL < 0.1) | (PRUNING > 0.9)
for i in range(N):
    print(f"G{i+1}: alpha={ALPHA_MODEL[i]} <0.1? {ALPHA_MODEL[i] < 0.1} (đúng biên, so sánh chặt)  "
          f"Pruning={PRUNING[i]:.4f} >0.9? {PRUNING[i] > 0.9}  -> {'XOÁ' if FP[i] else 'giữ'}")
print(f"Kịch bản N=10 (giả định alpha=0.5 cho mọi Gaussian): Pruning>0.9 -> xoá {np.where(PR10 > 0.9)[0]+1} "
      f"(Gaussian 7 có Pruning=0.9, không > 0.9 -> giữ)")

# ============================================================================
# 7.3 cuối — luỹ thừa N
# ============================================================================
hdr("[7.3 cuối] N_cuối ≈ N0 [(1+r_spawn)(1-r_prune)]^n,  r_spawn=0.15, r_prune=0.05")
base = (1 + 0.15) * (1 - 0.05)
print(f"(1+0.15)(1-0.05) = {base:.4f}")
for N0 in (4, 100_000):
    for n in (28, 145):
        print(f"N0={N0:>7}, n={n:>3}: hệ số = {base**n:.6e}  ->  N_cuối ≈ {N0*base**n:.6e}")

# ============================================================================
# 7.8 — Đầu ra của khối
# ============================================================================
hdr("[7.8] N <- N + |clone| + |split| - |xoá|")
n_clone, n_split, n_rm = int(CLONE.sum()), int(SPLIT.sum()), int(REMOVE.sum())
N_NEW = N + n_clone + n_split - n_rm
print(f"N = {N} + {n_clone} + {n_split} - {n_rm} = {N_NEW}   (split: bản gốc bị xoá, thêm 2 con -> ròng +1)")
if simulated:
    print(f"Nếu tính cả split GIẢ LẬP G1: N = {N} + {n_clone} + 1 - {n_rm} = {N + n_clone + 1 - n_rm}")
print("Danh sách Gaussian sau ADC (theo công thức 7.8: 8 con sau split, trừ 2 chỉ số bị rút {1,3} = G1c1, G3c1")
print("theo thứ tự repeat(2,1) [G1c1,G2c1,G3c1,G4c1,G1c2,G2c2,G3c2,G4c2]):")
if not simulated:
    children = [(i, j) for j in range(2) for i in split_idx]           # thứ tự repeat(2,1)
    child_mu = {}
    for (i, m, st) in SPLIT_CHILDREN:
        child_mu.setdefault(i, []).append((m, st))
    removed_idx = set(int(x) for x in np.where(REMOVE)[0])          # chỉ số gốc bị rút (0-based)
    for k, (i, j) in enumerate(children):
        m, st = child_mu[i][j]
        tag = "XOÁ (chỉ số %d bị rút)" % (k + 1) if (k < N and k in removed_idx) else "giữ"
        print(f"  G{i+1}c{j+1}: mu={m} s~={st[0]:.6f} s={np.exp(st[0]):.6f} "
              f"alpha~={inverse_sigmoid(min(0.1, 0.8)):+.6f}  -> {tag}")
    for i in np.where(CLONE)[0]:
        print(f"  G{i+1}' (clone): mu={P[i]} s={S[i,0]:.6f}")
