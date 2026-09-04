"""Mo hinh chi phi fastgs-lite vs 3DGS - phep do bang so, khong can GPU.

Chay:  python demos/fastgs_cost_model.py

Tai lieu di kem: DOCS/fastgs-acceleration-method.md (Phan IV-VI).

Ba ti so duoi day la ba "phep do" ma repo nay that su lam duoc ma khong can
chay A/B tren GPU:

  R_tile   - ti so so cap (tile, Gaussian) moi splat, do bang cach dem tile
             that tren mot quan the splat mo phong. So sanh:
               * 3DGS goc: hop VUONG nua canh ceil(3*sqrt(lambda_max))
                 (forward.cu:240 + getRect, auxiliary.h:50)
               * fastgs  : compact box t = mult*2*ln(255*o) + loc tile theo
                 ellipse chinh xac (auxiliary.h:318-345, AccuTile)
  R_adam   - ti so so lan Adam that su step trong 30k vong. Day la con so
             CHINH XAC, doc thang tu optimizer_step (gaussian_model.py:225).
  R_gauss  - ti so so Gaussian trung binh. Day la mo hinh DINH TINH: repo
             khong co phep do A/B nen spawn/prune rate la tham so gia dinh.

R_tile va R_adam suy ra tu ma nguon + hinh hoc, khong phai tu paper.
R_gauss phai tu do moi biet - script chi cho thay no nhay the nao.
"""

import numpy as np

TILE = 16
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. Hinh hoc: dem tile cho mot splat
# ---------------------------------------------------------------------------
def conic_from_axes(lam1, lam2, theta):
    """Sigma' = R diag(lam1, lam2) R^T  ->  conic M = Sigma'^{-1} = (A, B, C)."""
    c, s = np.cos(theta), np.sin(theta)
    sxx = lam1 * c * c + lam2 * s * s
    sxy = (lam1 - lam2) * c * s
    syy = lam1 * s * s + lam2 * c * c
    det = sxx * syy - sxy * sxy
    return syy / det, -sxy / det, sxx / det   # A, B, C


def sigma_axes(lam1, lam2, theta):
    """Phuong sai theo truc man hinh: Sigma'_11 va Sigma'_22."""
    c, s = np.cos(theta), np.sin(theta)
    return lam1 * c * c + lam2 * s * s, lam1 * s * s + lam2 * c * c


def tiles_3dgs(lam1, lam2):
    """3DGS goc: hop VUONG nua canh 3*sqrt(lambda_max), moi tile cham deu tinh.

    getRect() chia toa do cho BLOCK, nen so tile moi chieu la
    ceil((p+r)/16) - floor((p-r)/16). Lay ki vong theo vi tri tam p trong tile
    thi trung binh bang 2r/16 + 1.
    """
    r = np.ceil(3.0 * np.sqrt(max(lam1, lam2)))
    n = 2.0 * r / TILE + 1.0
    return n * n


def _min_quadratic_on_box(A, B, C, x0, x1, y0, y1, iters=30):
    """min cua Q(dx,dy)=A dx^2 + 2B dx dy + C dy^2 tren hop, bang coordinate
    descent (Q loi, 2 bien -> CD hoi tu ve nghiem toan cuc)."""
    x = min(max(0.0, x0), x1)
    y = min(max(0.0, y0), y1)
    for _ in range(iters):
        x = min(max(-B * y / A, x0), x1)
        y = min(max(-B * x / C, y0), y1)
    return A * x * x + 2 * B * x * y + C * y * y


def tiles_fastgs(lam1, lam2, theta, opacity, mult, px=None, py=None):
    """fastgs: compact box + loc tile theo ellipse chinh xac.

    Dem so tile T sao cho  min_{d in T} d^T M d <= t,  t = mult*2*ln(255*o).
    Day dung la tap tile ma AccuTile giu lai (auxiliary.h:199-315).
    """
    t = mult * 2.0 * np.log(255.0 * opacity)
    if t <= 0.0:
        return 0.0
    A, B, C = conic_from_axes(lam1, lam2, theta)
    s11, s22 = sigma_axes(lam1, lam2, theta)
    hx, hy = np.sqrt(t * s11), np.sqrt(t * s22)
    if px is None:
        px, py = RNG.uniform(0, TILE), RNG.uniform(0, TILE)
    ix0, ix1 = int(np.floor((px - hx) / TILE)), int(np.floor((px + hx) / TILE))
    iy0, iy1 = int(np.floor((py - hy) / TILE)), int(np.floor((py + hy) / TILE))
    count = 0
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            q = _min_quadratic_on_box(A, B, C,
                                      ix * TILE - px, (ix + 1) * TILE - px,
                                      iy * TILE - py, (iy + 1) * TILE - py)
            if q <= t:
                count += 1
    return float(count)


def tiles_compact_box_only(lam1, lam2, theta, opacity, mult):
    """Compact box nhung KHONG loc ellipse - de tach rieng hai nguon tiet kiem."""
    t = mult * 2.0 * np.log(255.0 * opacity)
    if t <= 0.0:
        return 0.0
    s11, s22 = sigma_axes(lam1, lam2, theta)
    nx = 2.0 * np.sqrt(t * s11) / TILE + 1.0
    ny = 2.0 * np.sqrt(t * s22) / TILE + 1.0
    return nx * ny


# ---------------------------------------------------------------------------
# 2. Quan the splat mo phong
# ---------------------------------------------------------------------------
def sample_population(n=3000, log_sigma_mean=1.1, log_sigma_std=0.9,
                      aniso_std=0.5, opacity_a=2.0, opacity_b=1.0):
    """sigma' (px) log-normal, ti le truc log-normal, opacity Beta lech ve 1."""
    s = np.clip(np.exp(RNG.normal(log_sigma_mean, log_sigma_std, n)), 0.5, 120.0)
    ratio = np.exp(RNG.normal(0.0, aniso_std, n))
    lam1, lam2 = (s * ratio) ** 2, (s / ratio) ** 2
    theta = RNG.uniform(0, np.pi, n)
    o = np.clip(RNG.beta(opacity_a, opacity_b, n), 0.005, 0.999)
    return lam1, lam2, theta, o


def tile_stats(pop, mult):
    lam1, lam2, theta, o = pop
    k3 = np.mean([tiles_3dgs(a, b) for a, b in zip(lam1, lam2)])
    kbox = np.mean([tiles_compact_box_only(a, b, t, oo, mult)
                    for a, b, t, oo in zip(lam1, lam2, theta, o)])
    kf = np.mean([tiles_fastgs(a, b, t, oo, mult)
                  for a, b, t, oo in zip(lam1, lam2, theta, o)])
    return k3, kbox, kf


def sweep_opacity(sigma=8.0, mult=0.5, opacities=(0.999, 0.5, 0.1, 0.02)):
    """Tai lap bang DOCS/fastgs-acceleration-method.md Section 4.3(b).

    Splat DANG HUONG sigma' = 8 px. Lay trung binh theo huong theta va vi tri
    tam trong tile de con so khong phu thuoc mot mau ngau nhien duy nhat.
    """
    rows = []
    for o in opacities:
        t = mult * 2.0 * np.log(255.0 * o)
        vals = [tiles_fastgs(sigma ** 2, sigma ** 2, th, o, mult, px, py)
                for th in np.linspace(0, np.pi, 24, endpoint=False)
                for px in np.linspace(0, TILE, 8, endpoint=False)
                for py in np.linspace(0, TILE, 8, endpoint=False)]
        rows.append((o, t, np.sqrt(t), float(np.mean(vals))))
    return rows


def sweep_aniso(sigma_g=8.0, opacity=0.999, mult=0.5, rhos=(1.0, 1.5, 2.0, 3.0, 5.0)):
    """Tai lap bang DOCS/fastgs-acceleration-method.md Section 4.5.

    CANH BAO VE KY HIEU: tham so `rho` o day dat sigma_max = sigma_g * rho va
    sigma_min = sigma_g / rho, nen TI LE TRUC THAT (sigma_max/sigma_min) bang
    rho^2, KHONG phai rho. Cot "ti le truc" duoi day in ra rho^2.
    """
    rows = []
    for rho in rhos:
        smax, smin = sigma_g * rho, sigma_g / rho
        k3 = tiles_3dgs(smax ** 2, smin ** 2)
        vals = [tiles_fastgs(smax ** 2, smin ** 2, th, opacity, mult, px, py)
                for th in np.linspace(0, np.pi, 24, endpoint=False)
                for px in np.linspace(0, TILE, 8, endpoint=False)
                for py in np.linspace(0, TILE, 8, endpoint=False)]
        kf = float(np.mean(vals))
        rows.append((rho, rho * rho, k3, kf, kf / k3))
    return rows


# ---------------------------------------------------------------------------
# 3. Nhip Adam - con so chinh xac tu optimizer_step()
# ---------------------------------------------------------------------------
def adam_steps(total_iters=30000, fastgs=True):
    """Dem so lan .step() cua (optimizer chinh, shoptimizer).

    QUAN TRONG: train.py:159 va pipeline/trainer.py:143 deu bao boc loi goi
    bang `if iteration < opt.iterations:` -> vong CUOI CUNG (30000) KHONG step.
    Vi 30000 chia het cho 64, neu duyet toi 30000 se dem du dung 1 buoc cho
    ca `optimizer` lan `shoptimizer`. Nen range dung la range(1, total_iters).
    """
    main = sh = 0
    for it in range(1, total_iters):        # KHONG bao gom vong cuoi
        if not fastgs:                      # 3DGS goc: moi tham so, moi vong
            main += 1
            sh += 1
            continue
        if it <= 15000:                     # gaussian_model.py:227
            main += 1
            if it % 16 == 0:
                sh += 1
        elif it <= 20000:                   # :233
            if it % 32 == 0:
                main += 1
                sh += 1
        else:                               # :239
            if it % 64 == 0:
                main += 1
                sh += 1
    return main, sh


# ---------------------------------------------------------------------------
# 4. Quy dao so Gaussian (DINH TINH - spawn/prune rate la gia dinh)
# ---------------------------------------------------------------------------
def mean_gaussians(total_iters=30000, interval=500, spawn=0.06, prune=0.01,
                   final_prune_frac=0.0, start=100_000):
    n, traj = float(start), []
    for it in range(1, total_iters + 1):
        if it < 15000 and it % interval == 0:
            n = n * (1 + spawn) * (1 - prune)
        if final_prune_frac and 15000 < it < 30000 and it % 3000 == 0:
            n *= (1 - final_prune_frac)
        traj.append(n)
    return float(np.mean(traj)), float(traj[-1])


# ---------------------------------------------------------------------------
# 5. Gop thanh mo hinh chi phi mot vong lap
# ---------------------------------------------------------------------------
#   T_iter = a*N + b*N*K + c*N*[co Adam step] + F
#
# a,b,c,F duoc hieu chinh sao cho o diem lam viec cua 3DGS (N_REF Gaussian,
# K_REF tile/splat) bon thanh phan chiem dung ti le SPLIT_* duoi day.
# SPLIT_* la GIA DINH theo profiling 3DGS thuong thay - doi chung de xem
# ket luan nhay den dau. F la phan KHONG scale theo N (nap anh, L1 + SSIM
# tren anh HxW, overhead Python/CUDA launch) - chinh no dat tran Amdahl.
SPLIT_PRE = 0.15        # preprocess: chieu Sigma', danh gia SH        ~ N
SPLIT_RASTER = 0.55     # duplicate + sort + blend + backward blend    ~ N*K
SPLIT_ADAM = 0.15       # Adam tren N*59 tham so                       ~ N (khi step)
SPLIT_FIXED = 0.15      # loss, SSIM, IO, launch overhead              hang so

N_REF, K_REF = 0.933, 21.91   # N tinh bang trieu Gaussian


def _weights():
    return (SPLIT_PRE / N_REF,
            SPLIT_RASTER / (N_REF * K_REF),
            SPLIT_ADAM / N_REF,
            SPLIT_FIXED)


def iteration_cost(n_gauss, k_tiles, adam_frac, score_overhead=0.0):
    """n_gauss tinh bang trieu. score_overhead: phan tram chi phi raster cong them
    cho compute_gaussian_score_fastgs (fastgs moi co)."""
    a, b, c, f = _weights()
    return (a * n_gauss
            + b * n_gauss * k_tiles * (1.0 + score_overhead)
            + c * n_gauss * adam_frac
            + f)


def scoring_overhead(total_iters=30000, densify_until=15000, interval=500,
                     num_cams=10, renders_per_cam=2):
    """compute_gaussian_score_fastgs render num_cams*renders_per_cam anh phu
    moi lan densify -> quy ve ti le so voi 1 render/vong."""
    calls = densify_until // interval
    return calls * num_cams * renders_per_cam / total_iters


def main():
    pop = sample_population()
    lam1, lam2, theta, o = pop
    sig = np.sqrt(np.sqrt(lam1 * lam2))
    print(f"Quan the mo phong: {len(lam1)} splat, sigma' trung vi = "
          f"{np.median(sig):.1f} px, opacity trung binh = {o.mean():.2f}")

    print()
    print("=" * 70)
    print("PHEP DO 1 - So tile moi splat (hinh hoc, suy tu ma nguon)")
    print("=" * 70)
    print(f"{'mult':>6} | {'K_3dgs':>8} | {'K_box':>8} | {'K_fastgs':>9} | {'R_tile':>7}")
    print("-" * 52)
    k3 = None
    results = {}
    for mult in (1.0, 0.7, 0.5, 0.3):
        k3, kbox, kf = tile_stats(pop, mult)
        results[mult] = (k3, kbox, kf)
        print(f"{mult:>6.1f} | {k3:>8.2f} | {kbox:>8.2f} | {kf:>9.2f} | {kf / k3:>7.3f}")
    print()
    print("Tach hai nguon tiet kiem tai mult=0.5:")
    k3, kbox, kf = results[0.5]
    print(f"  hop vuong 3-sigma (3DGS)        K = {k3:7.2f}   1.000")
    print(f"  compact box, chua loc ellipse   K = {kbox:7.2f}   {kbox / k3:.3f}")
    print(f"  + loc tile theo ellipse         K = {kf:7.2f}   {kf / k3:.3f}")

    print()
    print("PHU LUC 1a - K theo opacity (sigma'=8px dang huong, mult=0.5)")
    print("  [tai lap bang Section 4.3(b) cua tai lieu]")
    print(f"{'opacity':>8} | {'t':>6} | {'ban kinh':>10} | {'K_fastgs':>9}")
    print("-" * 44)
    for o, t, r, kf in sweep_opacity():
        print(f"{o:>8.3f} | {t:>6.2f} | {r:>8.2f}*s | {kf:>9.2f}")

    print()
    print("PHU LUC 1b - K theo do det (sigma_g=8px, o=1, mult=0.5)")
    print("  [tai lap bang Section 4.5 cua tai lieu]")
    print("  rho la THAM SO: sigma_max=8*rho, sigma_min=8/rho")
    print("  => ti le truc THAT = rho^2 (cot thu hai)")
    print(f"{'rho':>5} | {'ti le truc':>10} | {'K_3dgs':>8} | {'K_fastgs':>9} | {'R_tile':>7}")
    print("-" * 54)
    for rho, ratio, k3_, kf_, r_ in sweep_aniso():
        print(f"{rho:>5.1f} | {ratio:>9.1f}:1 | {k3_:>8.2f} | {kf_:>9.2f} | {r_:>7.3f}")

    print()
    print("=" * 70)
    print("PHEP DO 2 - So lan Adam step trong 30.000 vong (CHINH XAC tu code)")
    print("=" * 70)
    m3, s3 = adam_steps(fastgs=False)
    mf, sf = adam_steps(fastgs=True)
    print(f"{'':<12} | {'optimizer':>10} | {'shoptimizer':>12} | {'tong':>9}")
    print("-" * 52)
    print(f"{'3DGS goc':<12} | {m3:>10,} | {s3:>12,} | {m3 + s3:>9,}")
    print(f"{'fastgs-lite':<12} | {mf:>10,} | {sf:>12,} | {mf + sf:>9,}")
    print(f"{'ti so':<12} | {mf / m3:>10.3f} | {sf / s3:>12.3f} | "
          f"{(mf + sf) / (m3 + s3):>9.3f}")

    print()
    print("=" * 70)
    print("PHEP DO 3 - So Gaussian trung binh (DINH TINH, spawn/prune gia dinh)")
    print("=" * 70)
    n3_mean, n3_end = mean_gaussians(spawn=0.10, prune=0.005)
    nf_mean, nf_end = mean_gaussians(spawn=0.06, prune=0.010, final_prune_frac=0.05)
    print(f"{'':<12} | {'N trung binh':>13} | {'N cuoi':>11}")
    print("-" * 42)
    print(f"{'3DGS goc':<12} | {n3_mean:>13,.0f} | {n3_end:>11,.0f}")
    print(f"{'fastgs-lite':<12} | {nf_mean:>13,.0f} | {nf_end:>11,.0f}")
    print(f"{'ti so':<12} | {nf_mean / n3_mean:>13.3f} | {nf_end / n3_end:>11.3f}")

    print()
    print("=" * 70)
    print("GOP - chi phi 30.000 vong theo mo hinh")
    print("=" * 70)
    k3, _, kf = results[0.5]
    r_tile = kf / k3
    r_adam = (mf + sf) / (m3 + s3)
    r_gauss = nf_mean / n3_mean
    ov = scoring_overhead()
    n3m, nfm = n3_mean / 1e6, nf_mean / 1e6

    c3 = iteration_cost(n3m, k3, 1.0)
    cf = iteration_cost(nfm, kf, r_adam, ov)
    print(f"  R_tile  (mult=0.5, hinh hoc)   = {r_tile:.3f}")
    print(f"  R_gauss (GIA DINH)             = {r_gauss:.3f}")
    print(f"  R_adam  (chinh xac)            = {r_adam:.3f}")
    print(f"  overhead scoring (fastgs)      = +{100 * ov:.1f}% chi phi raster")
    print()
    print("  Bat tung don bay mot, so voi 3DGS goc:")
    print(f"{'':<34} | {'tang toc':>9}")
    print("  " + "-" * 46)
    combos = [
        ("ca ba don bay", iteration_cost(nfm, kf, r_adam, ov)),
        ("chi compact box (mult=0.5)", iteration_cost(n3m, kf, 1.0, 0.0)),
        ("chi giam so Gaussian", iteration_cost(nfm, k3, 1.0, ov)),
        ("chi nhip Adam thua", iteration_cost(n3m, k3, r_adam, 0.0)),
    ]
    for name, c in combos:
        print(f"  {name:<32} | {c3 / c:>8.2f}x")
    print()
    print(f"  Tran Amdahl (N -> 0, K -> 0): {1.0 / SPLIT_FIXED:.1f}x")
    print("  -> phan chi phi khong scale theo N (loss/SSIM/IO) dat tran cung.")
    print()
    print("  LUU Y: day la MO HINH, khong phai so do A/B tren GPU. R_gauss la")
    print("  tham so gia dinh - no la thu duy nhat phai do that moi biet.")
    print("  Xem DOCS/fastgs-acceleration-method.md Phan VI de biet cach chay A/B.")


if __name__ == "__main__":
    main()
