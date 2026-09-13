"""
Bài test số chương 5 — Image -> Loss va Metrics.
Chỉ dùng numpy + scipy (scipy chỉ cho conv2d; môi trường không có torch).

Mọi công thức được chép lại đúng từ code, có ghi tham chiếu:

  - train.py:100-103                 Ll1, ssim, loss = (1-λ)·L1 + λ·(1-SSIM)
  - utils/loss_utils.py:20           l1_loss = mean |a-b|   (mean trên 3·H·W)
  - utils/loss_utils.py:26-66        gaussian(11, 1.5), create_window, _ssim
  - utils/image_utils.py:14-26       mse (đã sửa batch dim), psnr = 20 log10(1/sqrt(mse))
  - pipeline/score.py:12-18          composite_score
  - arguments/__init__.py:82         λ = 0.2 ;  pipeline/config.py:44  λ = 0.25

Renderer numpy tối giản (LẤY TỪ CHƯƠNG 3–4, chép lại vừa đủ để sinh ảnh):
  - khởi tạo (ch.1): μ=p, s=sqrt(mean d² tới 3 láng giềng), R=I, Σ=s²I, màu=c_k
  - projection (ch.3): t=μ-c_v, μ' theo ndc2Pix, J với clamp 1.3·tan, Σ'=JΣJᵀ+0.3I, conic
  - rasterizer (ch.4): duyệt mọi Gaussian theo depth tăng dần tại mọi pixel,
    3 cửa loại power>0, α<1/255, T<1e-4; C = Σ c α T + T_final·C_bg; nền trắng.
  Không lọc tile (ảnh 48×32 quá nhỏ, mọi Gaussian chạm mọi tile không ảnh hưởng kết quả:
  tile chỉ là cách chia việc, không đổi toán học).

GT (00-scene.md): render cùng cảnh với α = 0.9; mô hình chưa học: α = 0.1.

Chạy:  python DOCS/Report/test/scripts/ch05_test.py
"""
import numpy as np
from scipy.signal import convolve2d

np.set_printoptions(precision=6, suppress=True)


def fmt(x, sig=4):
    return f"{x:.{sig}g}"


# --------------------------------------------------------------------------
# Cảnh đồ chơi (DOCS/Report/test/00-scene.md)
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

W, H = 48, 32
TAN_X, TAN_Y = 0.6, 0.4
FX, FY = W / (2 * TAN_X), H / (2 * TAN_Y)        # = 40, 40
LOWPASS = 0.3
ALPHA_MIN = 1.0 / 255.0
T_STOP = 1e-4
BG_WHITE = np.ones(3)

# --------------------------------------------------------------------------
# Lấy từ chương 1: scale khởi tạo = sqrt(mean d² tới 3 láng giềng gần nhất)
# --------------------------------------------------------------------------
def init_scales(p):
    n = p.shape[0]
    s = np.zeros(n)
    for k in range(n):
        d2 = np.sum((p - p[k]) ** 2, axis=1)
        d2 = np.sort(d2)[1:4]                     # bỏ chính nó, lấy 3 gần nhất
        s[k] = np.sqrt(np.mean(d2))
    return s


# --------------------------------------------------------------------------
# Lấy từ chương 3: projection cho một camera (xoay đơn vị)
# --------------------------------------------------------------------------
def project(mu, scales, cam_c):
    """Trả về: mean2D [N,2] (pixel), conic [N,3] (A,B,C), depth [N]."""
    t = mu - cam_c                                # W_v = [I | -c_v]
    tz = t[:, 2]
    # ndc2Pix: x_ndc = t_x/(t_z·tan) ; pix = ((x_ndc+1)·W - 1)/2
    x_ndc = t[:, 0] / (tz * TAN_X)
    y_ndc = t[:, 1] / (tz * TAN_Y)
    mean2d = np.stack([((x_ndc + 1) * W - 1) / 2, ((y_ndc + 1) * H - 1) / 2], axis=1)
    # computeCov2D: clamp rồi Jacobian
    lim_x, lim_y = 1.3 * TAN_X, 1.3 * TAN_Y
    tx_h = np.clip(t[:, 0] / tz, -lim_x, lim_x) * tz
    ty_h = np.clip(t[:, 1] / tz, -lim_y, lim_y) * tz
    conic = np.zeros((mu.shape[0], 3))
    for i in range(mu.shape[0]):
        J = np.array([[FX / tz[i], 0, -FX * tx_h[i] / tz[i] ** 2],
                      [0, FY / tz[i], -FY * ty_h[i] / tz[i] ** 2],
                      [0, 0, 0]])
        Sigma = (scales[i] ** 2) * np.eye(3)      # R = I, Σ = s²I
        cov = J @ Sigma @ J.T                     # W_v xoay = I
        c11, c12, c22 = cov[0, 0] + LOWPASS, cov[0, 1], cov[1, 1] + LOWPASS
        det = c11 * c22 - c12 ** 2
        conic[i] = [c22 / det, -c12 / det, c11 / det]
    return mean2d, conic, tz


# --------------------------------------------------------------------------
# Lấy từ chương 4: alpha-blending front-to-back, toàn ảnh, không lọc tile
# --------------------------------------------------------------------------
def render(mu, scales, cols, opacity, cam_c, bg):
    mean2d, conic, depth = project(mu, scales, cam_c)
    order = np.argsort(depth)                     # sort theo depth tăng dần
    ys, xs = np.mgrid[0:H, 0:W]
    px = np.stack([xs, ys], axis=-1).astype(np.float64)   # pixf = (pix.x, pix.y)
    T = np.ones((H, W))
    C = np.zeros((H, W, 3))
    done = np.zeros((H, W), dtype=bool)
    n_contrib = np.zeros((H, W), dtype=int)
    for i in order:
        d = px - mean2d[i]
        A, B, Cc = conic[i]
        power = -0.5 * (A * d[..., 0] ** 2 + Cc * d[..., 1] ** 2) - B * d[..., 0] * d[..., 1]
        alpha = np.minimum(0.99, opacity[i] * np.exp(power))
        use = (~done) & (power <= 0) & (alpha >= ALPHA_MIN)
        test_T = T * (1 - alpha)
        stop = use & (test_T < T_STOP)
        done |= stop
        use &= ~stop
        C[use] += cols[i] * (alpha[use] * T[use])[:, None]
        T[use] = test_T[use]
        n_contrib[use] += 1
    C += T[..., None] * bg
    return C.transpose(2, 0, 1), T, n_contrib   # [3,H,W] như tensor của repo


# --------------------------------------------------------------------------
# utils/loss_utils.py — chép lại bằng numpy
# --------------------------------------------------------------------------
def l1_loss(a, b):                                # loss_utils.py:20
    return np.mean(np.abs(a - b))


def gaussian(window_size, sigma):                 # loss_utils.py:26
    g = np.array([np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
                  for x in range(window_size)])
    return g / g.sum()


def create_window(window_size):                   # loss_utils.py:30
    g = gaussian(window_size, 1.5)[:, None]
    return g @ g.T                                # outer product, tổng = 1


def conv_same(img, window):
    """F.conv2d(padding=5, groups=channel): zero-pad 'same', từng kênh."""
    return np.stack([convolve2d(img[c], window, mode="same", boundary="fill", fillvalue=0)
                     for c in range(img.shape[0])])


def ssim_full(img1, img2, window_size=11):        # loss_utils.py:46-66
    window = create_window(window_size)
    mu1, mu2 = conv_same(img1, window), conv_same(img2, window)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = conv_same(img1 * img1, window) - mu1_sq
    sigma2_sq = conv_same(img2 * img2, window) - mu2_sq
    sigma12 = conv_same(img1 * img2, window) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(), dict(mu1=mu1, mu2=mu2, s1=sigma1_sq, s2=sigma2_sq,
                                 s12=sigma12, map=ssim_map)


def ssim(img1, img2):
    return ssim_full(img1, img2)[0]


# --------------------------------------------------------------------------
# utils/image_utils.py, pipeline/score.py
# --------------------------------------------------------------------------
def mse(a, b):                                    # image_utils.py:21 (đã sửa)
    return np.mean((a - b) ** 2)


def psnr(a, b):                                   # image_utils.py:25
    return 20 * np.log10(1.0 / np.sqrt(mse(a, b)))


def psnr_old_bug(a, b):
    """Lỗi cũ: view(shape[0]=3, -1) → MSE từng kênh → 3 PSNR → mean."""
    m = ((a - b) ** 2).reshape(3, -1).mean(1)
    return (20 * np.log10(1.0 / np.sqrt(m))).mean(), m


def composite_score(psnr_v, ssim_v, lpips_v, psnr_max=30.0):   # score.py:12
    psnr_norm = np.clip(psnr_v / psnr_max, 0.0, 1.0)
    return 0.4 * (1 - lpips_v) + 0.3 * ssim_v + 0.3 * psnr_norm, psnr_norm


# ==========================================================================
if __name__ == "__main__":
    N = points.shape[0]
    scales = init_scales(points)
    cam1 = cam_centers[0]
    ALPHA_MODEL, ALPHA_GT = 0.1, 0.9

    print("=" * 72)
    print("5.0 RENDER (lay tu chuong 1/3/4) — camera 1, nen trang")
    print("=" * 72)
    print(f"W x H = {W} x {H}, fx = fy = {FX:g}, N = {N}")
    print("scale khoi tao s_k = sqrt(mean d^2 toi 3 lang gieng):", scales)
    mean2d, conic, depth = project(points, scales, cam1)
    for k in range(N):
        print(f"  G{k+1}: depth={fmt(depth[k])}  mu'=({fmt(mean2d[k,0])}, {fmt(mean2d[k,1])})"
              f"  conic (A,B,C)=({fmt(conic[k,0])}, {fmt(conic[k,1])}, {fmt(conic[k,2])})")
    print("thu tu depth tang dan:", [f"G{i+1}" for i in np.argsort(depth)])

    I_rend, T_rend, nc_rend = render(points, scales, colors, np.full(N, ALPHA_MODEL), cam1, BG_WHITE)
    I_gt, T_gt, nc_gt = render(points, scales, colors, np.full(N, ALPHA_GT), cam1, BG_WHITE)

    def stats(name, I, T, nc):
        print(f"{name}: shape={I.shape} min={fmt(I.min())} max={fmt(I.max())} mean={fmt(I.mean())}"
              f" | T_final: min={fmt(T.min())} mean={fmt(T.mean())}"
              f" | so pixel co >=1 splat dong gop={int((nc>0).sum())}/{H*W}"
              f" | pixel toan nen (T=1)={int((T==1).sum())}")
    stats("I_rend (alpha=0.1)", I_rend, T_rend, nc_rend)
    stats("I_gt   (alpha=0.9)", I_gt, T_gt, nc_gt)
    # pixel gan tam G1 nhat
    cx, cy = int(round(mean2d[0, 0])), int(round(mean2d[0, 1]))
    print(f"pixel tai tam G1 (x={cx}, y={cy}): I_rend={I_rend[:, cy, cx]}  I_gt={I_gt[:, cy, cx]}")

    # ----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("5.1 LOSS HUAN LUYEN")
    print("=" * 72)
    diff = np.abs(I_rend - I_gt)
    L1_sum = diff.sum()
    L1 = l1_loss(I_rend, I_gt)
    print(f"sum |I_rend - I_gt| = {fmt(L1_sum, 6)}   (3HW = {3*H*W})")
    print(f"L1 = sum / 3HW = {fmt(L1_sum,6)} / {3*H*W} = {fmt(L1, 6)}")
    print(f"  (kiem tra l1_loss numpy = {fmt(L1, 6)})")
    print(f"  max |diff| = {fmt(diff.max())} tai pixel {np.unravel_index(diff.argmax(), diff.shape)}")

    g1 = gaussian(11, 1.5)
    print("\nGaussian 1D 11 tap, sigma=1.5 (chuan hoa):")
    print("  ", np.array2string(g1, precision=6))
    print(f"   tong = {g1.sum():.6f}")
    win = create_window(11)
    print(f"cua so 2D 11x11 = outer(g,g): tam = {fmt(win[5,5],6)}, goc = {fmt(win[0,0],6)}, tong = {win.sum():.6f}")

    S, aux = ssim_full(I_rend, I_gt)
    print(f"\nSSIM(I_rend, I_gt) = {fmt(S, 6)}")
    print("minh hoa tai pixel tam G1, kenh R (c=0):")
    c = 0
    m1, m2 = aux['mu1'][c, cy, cx], aux['mu2'][c, cy, cx]
    s1, s2, s12 = aux['s1'][c, cy, cx], aux['s2'][c, cy, cx], aux['s12'][c, cy, cx]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * m1 * m2 + C1) * (2 * s12 + C2)
    den = (m1 ** 2 + m2 ** 2 + C1) * (s1 + s2 + C2)
    print(f"  mu1={fmt(m1,6)} mu2={fmt(m2,6)} sigma1^2={fmt(s1,6)} sigma2^2={fmt(s2,6)} sigma12={fmt(s12,6)}")
    print(f"  tu so  = (2*{fmt(m1)}*{fmt(m2)}+1e-4)(2*{fmt(s12)}+9e-4) = {fmt(num,6)}")
    print(f"  mau so = ({fmt(m1)}^2+{fmt(m2)}^2+1e-4)({fmt(s1)}+{fmt(s2)}+9e-4) = {fmt(den,6)}")
    print(f"  ssim_map = {fmt(num/den,6)}   (kiem tra map[c,y,x] = {fmt(aux['map'][c,cy,cx],6)})")
    print(f"  ssim_map: min={fmt(aux['map'].min())} max={fmt(aux['map'].max())}"
          f" ; so pixel map==1 (vung nen, ca hai anh deu trang): {int(np.isclose(aux['map'],1).sum())}/{3*H*W}")

    D_SSIM = 1 - S
    for lam in (0.2, 0.25):
        loss = (1 - lam) * L1 + lam * D_SSIM
        print(f"\nlambda={lam}: L = (1-{lam})*{fmt(L1,6)} + {lam}*(1-{fmt(S,6)})"
              f" = {fmt((1-lam)*L1,6)} + {fmt(lam*D_SSIM,6)} = {fmt(loss, 6)}")
    LOSS_02 = 0.8 * L1 + 0.2 * D_SSIM
    LOSS_025 = 0.75 * L1 + 0.25 * D_SSIM

    # ----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("5.3 NEN NGAU NHIEN")
    print("=" * 72)
    rng = np.random.RandomState(0)
    C_bg = rng.rand(3)
    print(f"seed 0 -> C_bg = {C_bg}")
    I_rend_bg, T_bg, _ = render(points, scales, colors, np.full(N, ALPHA_MODEL), cam1, C_bg)
    assert np.allclose(T_bg, T_rend)
    L1_bg = l1_loss(I_rend_bg, I_gt)
    S_bg = ssim(I_rend_bg, I_gt)
    loss_bg = 0.8 * L1_bg + 0.2 * (1 - S_bg)
    # phan tach: I_rend_bg - I_rend = T_final (C_bg - 1)
    delta = I_rend_bg - I_rend
    pred = T_rend[None] * (C_bg[:, None, None] - 1.0)
    print(f"I_rend_bg - I_rend = T_final*(C_bg - C_white): max sai so so voi cong thuc = {fmt(np.abs(delta-pred).max(),3)}")
    print(f"L1 nen trang  = {fmt(L1,6)}   SSIM = {fmt(S,6)}   loss = {fmt(LOSS_02,6)}")
    print(f"L1 nen ngau nhien = {fmt(L1_bg,6)}   SSIM = {fmt(S_bg,6)}   loss = {fmt(loss_bg,6)}")
    print(f"thay doi L1: {L1_bg-L1:+.6f} (x{fmt(L1_bg/L1,4)})")
    # phan L1 tang o pixel nen thuan (T=1) va pixel co splat
    pure_bg = (T_rend == 1)
    l1_pure = np.abs(I_rend_bg - I_gt)[:, pure_bg].sum()
    l1_splat = np.abs(I_rend_bg - I_gt)[:, ~pure_bg].sum()
    print(f"phan tich tong |diff| nen ngau nhien: pixel nen thuan (T=1, {int(pure_bg.sum())} px) = {fmt(l1_pure,6)}"
          f" ; pixel co splat ({int((~pure_bg).sum())} px) = {fmt(l1_splat,6)}")
    print(f"  pixel nen thuan: |C_bg - 1| = {np.abs(C_bg-1)}  tong 3 kenh = {fmt(np.abs(C_bg-1).sum())}"
          f" x {int(pure_bg.sum())} px = {fmt(np.abs(C_bg-1).sum()*pure_bg.sum(),6)}")
    print(f"  -> voi GT nen trang, moi pixel co T_final>0 bi phat |T_final*(C_bg-1)|:"
          f" Gaussian phai duc han (T_final->0) hoac bien mat.")
    # pixel tam G1: T_final nho hon, phat it hon
    print(f"  tai tam G1: T_final={fmt(T_rend[cy,cx])}  I_rend={I_rend[:,cy,cx]}  I_rend_bg={I_rend_bg[:,cy,cx]}")
    # Ky vong theo C_bg ~ U[0,1]^3: E|C_bg - 1| = 0.5 moi kenh -> phat ky vong = 0.5*T_final tai pixel nen
    print("\nKy vong theo C_bg~U[0,1]: E|C_bg-1| = 0.5/kenh -> sai so ky vong tai pixel nen thuan = 0.5*T_final")
    print(f"  T_final trung binh (alpha=0.1) = {fmt(T_rend.mean(),6)} -> phat ky vong ~ {fmt(0.5*T_rend.mean(),6)}/kenh/pixel"
          f" ; voi GT (alpha=0.9) T_final = {fmt(T_gt.mean(),6)} -> {fmt(0.5*T_gt.mean(),6)}")
    print("L1 nen ngau nhien theo 5 seed (GT nen trang co dinh):")
    for sd in range(5):
        cb = np.random.RandomState(sd).rand(3)
        I_s, _, _ = render(points, scales, colors, np.full(N, ALPHA_MODEL), cam1, cb)
        print(f"  seed {sd}: C_bg={cb}  L1={fmt(l1_loss(I_s, I_gt),6)}"
              f"  (L1 tai 42 px nen thuan = {fmt(np.abs(cb-1).sum()*pure_bg.sum()/(3*H*W),6)} so voi 0 khi nen trang)")
    print("  NOTE: o canh do choi GT toi hon nen trang (mean I_gt = %s) nen nen toi tinh co GAN GT hon;"
          " so hang phat dung nghia la o 42 px nen thuan: sai so 0 -> |C_bg-1|." % fmt(I_gt.mean()))

    # ----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("5.4 METRICS")
    print("=" * 72)
    MSE = mse(I_rend, I_gt)
    PSNR = psnr(I_rend, I_gt)
    print(f"MSE = sum (diff^2)/3HW = {fmt(((I_rend-I_gt)**2).sum(),6)} / {3*H*W} = {fmt(MSE,6)}")
    print(f"PSNR = 20 log10(1/sqrt({fmt(MSE,6)})) = {fmt(PSNR,6)} dB")
    PSNR_old, m_c = psnr_old_bug(I_rend, I_gt)
    print(f"Loi cu: MSE tung kenh (R,G,B) = {m_c}")
    print(f"        PSNR tung kenh = {20*np.log10(1/np.sqrt(m_c))}")
    print(f"        mean(PSNR_c) = {fmt(PSNR_old,6)} dB  >=  PSNR dung {fmt(PSNR,6)} dB"
          f"  (chenh +{fmt(PSNR_old-PSNR,4)} dB)")
    print(f"  Jensen: -10 log10 la loi => mean(-log m_c) >= -log(mean m_c); mean(m_c)={fmt(m_c.mean(),6)} = MSE")
    assert PSNR_old >= PSNR - 1e-12

    LPIPS_ASSUMED = 0.30
    score, pn = composite_score(PSNR, S, LPIPS_ASSUMED)
    print(f"\nLPIPS: KHONG tinh duoc (can mang AlexNet/VGG) -> GIA DINH LPIPS = {LPIPS_ASSUMED}")
    print(f"PSNR_norm = clamp({fmt(PSNR,6)}/30, 0, 1) = {fmt(pn,6)}")
    print(f"Score = 0.4*(1-{LPIPS_ASSUMED}) + 0.3*{fmt(S,6)} + 0.3*{fmt(pn,6)}"
          f" = {fmt(0.4*(1-LPIPS_ASSUMED),4)} + {fmt(0.3*S,6)} + {fmt(0.3*pn,6)} = {fmt(score,6)}")

    ps = np.array([25.0, 32.0, 35.0])
    a = np.clip(ps.mean() / 30, 0, 1)
    b = np.clip(ps / 30, 0, 1).mean()
    print(f"\nThu tu trung binh, PSNR 3 view = {ps}:")
    print(f"  clamp(mean/30) = clamp({fmt(ps.mean(),6)}/30) = {fmt(a,6)}")
    print(f"  mean(clamp/30) = mean({np.clip(ps/30,0,1)}) = {fmt(b,6)}")
    print(f"  chenh lech 0.3*({fmt(a,6)} - {fmt(b,6)}) = {fmt(0.3*(a-b),6)} diem Score")

    # ----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("5.5 DO NHAY: loss theo alpha mo hinh (GT co dinh alpha=0.9)")
    print("=" * 72)
    print(f"{'alpha':>6} | {'L1':>10} | {'SSIM':>10} | {'D-SSIM':>10} | {'loss(0.2)':>10} | {'PSNR':>8}")
    sens = []
    for a_m in (0.1, 0.3, 0.5, 0.7, 0.9):
        I_a, _, _ = render(points, scales, colors, np.full(N, a_m), cam1, BG_WHITE)
        l1a, sa = l1_loss(I_a, I_gt), ssim(I_a, I_gt)
        la = 0.8 * l1a + 0.2 * (1 - sa)
        pa = psnr(I_a, I_gt) if mse(I_a, I_gt) > 0 else np.inf
        sens.append((a_m, l1a, sa, 1 - sa, la, pa))
        print(f"{a_m:>6} | {l1a:>10.6f} | {sa:>10.6f} | {1-sa:>10.6f} | {la:>10.6f} | {pa:>8.3f}")
    assert sens[-1][4] == 0.0

    # ----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("5.6 DAU RA CUA KHOI")
    print("=" * 72)
    print(f"L1        = {L1:.6f}")
    print(f"SSIM      = {S:.6f}")
    print(f"D-SSIM    = {D_SSIM:.6f}")
    print(f"Loss l=0.2  = {LOSS_02:.6f}")
    print(f"Loss l=0.25 = {LOSS_025:.6f}")
    print(f"MSE       = {MSE:.6f}")
    print(f"PSNR      = {PSNR:.4f} dB")
    print(f"PSNR(loi cu) = {PSNR_old:.4f} dB")
    print(f"LPIPS     = {LPIPS_ASSUMED} (gia dinh)")
    print(f"Score     = {score:.6f}")
    print(f"T_final mean (nen trang, alpha=0.1) = {T_rend.mean():.6f}")
