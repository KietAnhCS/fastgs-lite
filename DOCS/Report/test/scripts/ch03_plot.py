"""
Chương 3 — Vẽ hình minh hoạ (matplotlib + numpy, không torch).
Import lại ch03_test.py để lấy đúng số (RESULT, MU2D, hàm processTiles, getRect_3dgs...).
PNG -> DOCS/Report/test/figures/ch03_<tên>.png
"""
import contextlib
import io
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    import ch03_test as T  # chạy toàn bộ test, lấy số

FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'axes.linewidth': 1.1, 'axes.titleweight': 'bold', 'axes.labelsize': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.framealpha': 0.9, 'legend.edgecolor': '0.75',
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.dpi': 450,
})
COL = ['#d62728', '#2ca02c', '#1f77b4', '#7f7f7f']
NAMES = ['G1', 'G2', 'G3', 'G4']
BLK = 16


def save(fig, name):
    path = os.path.join(FIG_DIR, f"ch03_{name}.png")
    fig.savefig(path, dpi=450, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("saved", os.path.relpath(path, os.path.join(HERE, "..")))


def ellipse_patch(mu, cov, level, **kw):
    """Ellipse {Δ : Δᵀ Σ'^{-1} Δ = level}  (level = t hoặc 9 cho 3σ, 1 cho 1σ)."""
    lam, vec = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    ang = math.degrees(math.atan2(vec[1, 0], vec[0, 0]))
    return Ellipse(mu, 2 * math.sqrt(level * lam[0]), 2 * math.sqrt(level * lam[1]), angle=ang, **kw)


def draw_grid(ax, W, H, blk=BLK, label_ids=True, gx=None, fs=8):
    gx = gx or W // blk
    for x in range(0, W + 1, blk):
        ax.axvline(x, color='k', lw=0.8, alpha=0.6)
    for y in range(0, H + 1, blk):
        ax.axhline(y, color='k', lw=0.8, alpha=0.6)
    if label_ids:
        for ty in range(H // blk):
            for tx in range(W // blk):
                ax.text(tx * blk + 1, ty * blk + 1.2, f"{ty * gx + tx}", fontsize=fs, va='top', ha='left',
                        color='0.35', fontweight='bold')
    ax.set_xlim(-0.5, W + 0.5)
    ax.set_ylim(H + 0.5, -0.5)  # y hướng xuống như ảnh
    ax.set_aspect('equal')
    ax.grid(False)


# =========================================================================================
# 1. ch03_projection.png
# =========================================================================================
def fig_projection():
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], hspace=0.35, wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    draw_grid(ax, T.W, T.H, fs=9)
    for i in range(T.N):
        R = T.RESULT[(1, i)]
        mu, cov = R['mu2'], R['cov']
        ax.add_patch(ellipse_patch(mu, cov, 1.0, fc=COL[i], ec=COL[i], alpha=0.12, lw=1.6, ls='-'))
        ax.add_patch(ellipse_patch(mu, cov, 1.0, fc='none', ec=COL[i], lw=1.6))
        ax.plot(mu[0], mu[1], 'o', color=COL[i], ms=7, mec='k', mew=0.6, zorder=5)
        dx, dy = [(1.0, -1.6), (1.0, 1.9), (-9.5, 1.9), (1.0, -1.6)][i]
        ax.text(mu[0] + dx, mu[1] + dy,
                f"{NAMES[i]} ({mu[0]:.2f}, {mu[1]:.2f})\n" + r"$\sigma'$=" + f"{math.sqrt(cov[0,0]):.2f} px",
                fontsize=8, color=COL[i], fontweight='bold', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))
    ax.plot(23.5, 15.5, '+', color='k', ms=10, mew=1.2, zorder=4)
    ax.set_title(r"Camera 1: $\mu'_i$ và ellipse $1\sigma$ của $\Sigma'_i$ trên ảnh $48\times32$ (lưới tile 16 px, id = $t_y\cdot3+t_x$)",
                 fontsize=10)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel, hướng xuống)")
    ax.set_xticks(range(0, 49, 8))
    ax.set_yticks(range(0, 33, 8))

    for k, v in enumerate([2, 3]):
        axs = fig.add_subplot(gs[k, 1])
        draw_grid(axs, T.W, T.H, fs=8)
        for i in range(T.N):
            R = T.RESULT[(v, i)]
            mu, cov = R['mu2'], R['cov']
            axs.add_patch(ellipse_patch(mu, cov, 1.0, fc=COL[i], ec=COL[i], alpha=0.10, lw=1.2))
            axs.add_patch(ellipse_patch(mu, cov, 1.0, fc='none', ec=COL[i], lw=1.2))
            axs.plot(mu[0], mu[1], 'o', color=COL[i], ms=5, mec='k', mew=0.5, zorder=5)
            off = [(-1.0, -1.6), (1.0, 1.6), (-1.0, 1.6), (1.0, -1.6)][i]
            axs.text(mu[0] + off[0], mu[1] + off[1], NAMES[i], fontsize=7, ha='right' if off[0] < 0 else 'left',
                     color=COL[i], fontweight='bold', va='center', zorder=6,
                     bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))
        txt = "\n".join(f"{NAMES[i]} ({T.RESULT[(v, i)]['mu2'][0]:.2f}, {T.RESULT[(v, i)]['mu2'][1]:.2f})"
                        for i in range(T.N))
        axs.text(46.5 if v == 2 else 1.5, 30.5, txt, fontsize=7, ha='right' if v == 2 else 'left', va='bottom',
                 zorder=7, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.5', alpha=0.95), linespacing=1.3)
        c = T.CAMS[v]
        axs.set_title(f"Camera {v}: $c_{v}$ = ({c[0]:g}, {c[1]:g}, {c[2]:g}) — tâm lệch "
                      + ("trái" if v == 2 else "phải, lên trên"), fontsize=9)
        axs.set_xticks(range(0, 49, 16))
        axs.set_yticks(range(0, 33, 16))
        axs.tick_params(labelsize=8)
    fig.suptitle(r"Chiếu tâm $\mu'=(f_x t_x/t_z+23.5,\ f_y t_y/t_z+15.5)$ và covariance $\Sigma'=J\Sigma J^\top+0.3I$ (4 Gaussian, 3 camera)",
                 fontsize=11, y=0.96)
    save(fig, "projection")


# =========================================================================================
# 2. ch03_jacobian.png
# =========================================================================================
def fig_jacobian():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw=dict(width_ratios=[1.15, 1]))
    # ---- trái: sơ đồ EWA, mặt cắt (x, z) của camera 1, G1
    s = T.S_INIT[0]
    tz = 4.0
    f = T.FX
    sig_px = f * s / tz
    ax1.set_aspect('equal')
    ax1.plot(0, 0, 'k^', ms=10, zorder=5)
    ax1.text(0, 0.12, "camera $c_1$ (pinhole)", fontsize=8, ha='center', va='bottom')
    # trục quang
    ax1.annotate("", xy=(5.3, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='0.4', lw=1))
    ax1.text(5.35, 0, "$z$ (depth)", fontsize=8, va='center')
    # Gaussian 3D: vòng tròn bán kính s tại t_z = 4
    ax1.add_patch(Circle((tz, 0), s, fc=COL[0], ec=COL[0], alpha=0.18, lw=1.5))
    ax1.add_patch(Circle((tz, 0), s, fc='none', ec=COL[0], lw=1.5))
    ax1.plot(tz, 0, 'o', color=COL[0], ms=5)
    ax1.annotate("", xy=(tz, s), xytext=(tz, 0), arrowprops=dict(arrowstyle='<->', color=COL[0], lw=1))
    ax1.text(tz + 0.08, s / 2, f"$s$ = {s:.4f}", fontsize=8, color=COL[0], va='center')
    ax1.text(tz, -s - 0.15, f"G1: $\\Sigma=s^2I$, $t_z$ = {tz:.0f}", fontsize=8.5, ha='center', va='top',
             color=COL[0], fontweight='bold')
    # mặt phẳng ảnh ở z = 1 (đơn vị chuẩn hoá; f = 40 px ứng với 1 đơn vị)
    zi = 1.0
    ax1.plot([zi, zi], [-0.75, 0.75], color='k', lw=2)
    ax1.text(zi, -0.85, "mặt phẳng ảnh\n($z=1$, $f_x=f_y=40$ px/đv)", fontsize=8, ha='center', va='top')
    # tia chiếu biên (tiếp tuyến gần đúng: x = ±s tại z = tz)
    for sgn in (1, -1):
        ax1.plot([0, tz], [0, sgn * s], color=COL[0], lw=0.9, ls='--', alpha=0.8)
        ax1.plot([zi, zi], [0, sgn * s / tz], color=COL[0], lw=4, solid_capstyle='butt', alpha=0.9)
    ax1.annotate("", xy=(zi + 0.08, s / tz), xytext=(zi + 0.08, 0), arrowprops=dict(arrowstyle='<->', color=COL[0], lw=1))
    ax1.annotate(f"$s/t_z$ = {s/tz:.4f}\n$\\to \\sigma' = f s/t_z$ = {sig_px:.2f} px", (zi + 0.1, s / tz / 2),
                 xytext=(1.45, 1.05), fontsize=8,
                 ha='left', va='center', color=COL[0], arrowprops=dict(arrowstyle='->', color=COL[0], lw=0.8))
    ax1.text(2.6, -1.35,
             r"$J_{11}=f_x/t_z$ = 10;  $\Sigma'_{11}=J_{11}^2 s^2+0.3$ = " + f"{f*f/tz/tz*s*s:.2f} + 0.3 = {T.RESULT[(1,0)]['cov'][0,0]:.2f}"
             + "\n" + r"$\sigma'=\sqrt{\Sigma'_{11}}$ = " + f"{math.sqrt(T.RESULT[(1,0)]['cov'][0,0]):.2f} px"
             + r"  (ellipse $1\sigma$; hộp $3\sigma$: $r=\lceil 3\sigma'\rceil$ = " + f"{T.RESULT[(1,0)]['r']})",
             fontsize=8.5, ha='center', va='top',
             bbox=dict(boxstyle='round', fc='#fff7e6', ec='#e0a040'))
    ax1.set_xlim(-0.4, 6.2)
    ax1.set_ylim(-2.2, 1.6)
    ax1.set_xlabel("z (camera), đơn vị world")
    ax1.set_ylabel("x (camera)")
    ax1.set_title("EWA: tuyến tính hoá phép chiếu phối cảnh tại tâm (G1, camera 1)", fontsize=10)

    # ---- phải: σ' theo t_z
    tzs = np.linspace(2.0, 8.0, 300)
    for i in range(T.N):
        s_i = T.S_INIT[i]
        ax2.plot(tzs, f * s_i / tzs, color=COL[i], lw=1.4, label=f"{NAMES[i]}: $s$={s_i:.3f}")
        R = T.RESULT[(1, i)]
        tz_i = R['depth']
        sig_i = math.sqrt(R['cov'][0, 0])
        ax2.plot(tz_i, sig_i, 'o', color=COL[i], ms=7, mec='k', mew=0.6, zorder=5)
        ax2.annotate(f"$t_z$={tz_i:g} → {sig_i:.2f} px", (tz_i, sig_i), xytext=[(-100, 26), (16, 4), (10, 14), (8, -26)][i], arrowprops=dict(arrowstyle='->', color=COL[i], lw=0.8),
                     textcoords='offset points', fontsize=8, color=COL[i], fontweight='bold')
    ax2.set_xlabel("$t_z$ (depth trong camera)")
    ax2.set_ylabel(r"$\sigma' = \sqrt{\Sigma'_{11}}$ (pixel)")
    ax2.set_title(r"$\sigma'(t_z)=f\,s/t_z$ (đường) và $\sqrt{(f s/t_z)^2+0.3}$ tại depth thật (điểm)", fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylim(0, 25)
    save(fig, "jacobian")


# =========================================================================================
# 3. ch03_boxes.png
# =========================================================================================
def draw_boxes_panel(ax, W, H, mu, cov, t, r, half, alpha, title):
    grid = (W // BLK, H // BLK)
    A_, B_, C_ = T.RESULT[(1, 0)]['A'], T.RESULT[(1, 0)]['B'], T.RESULT[(1, 0)]['C']
    r_min, r_max = T.getRect_3dgs(mu, r, grid)
    K3 = (r_max[0] - r_min[0]) * (r_max[1] - r_min[1])
    fg = T.duplicateToTilesTouched(mu, (A_, B_, C_, alpha), grid, T.MULT)
    draw_grid(ax, W, H, fs=7, label_ids=(W <= 64))
    # tile 3DGS (vàng nhạt), tile box (cam), tile giữ (xanh)
    ids_box = [ty * grid[0] + tx for ty in range(fg['rect_min'][1], fg['rect_max'][1])
               for tx in range(fg['rect_min'][0], fg['rect_max'][0])]
    for ty in range(r_min[1], r_max[1]):
        for tx in range(r_min[0], r_max[0]):
            ax.add_patch(Rectangle((tx * BLK, ty * BLK), BLK, BLK, fc='#ffe680', ec='none', alpha=0.9, zorder=0))
    for tid in ids_box:
        tx, ty = tid % grid[0], tid // grid[0]
        ax.add_patch(Rectangle((tx * BLK, ty * BLK), BLK, BLK, fc='#ffb060', ec='none', alpha=0.9, zorder=0.5))
    for tid in fg['ids']:
        tx, ty = tid % grid[0], tid // grid[0]
        ax.add_patch(Rectangle((tx * BLK, ty * BLK), BLK, BLK, fc='#8fd18f', ec='none', alpha=0.9, zorder=1))
    # 3 lớp hình
    ax.add_patch(Rectangle((mu[0] - r, mu[1] - r), 2 * r, 2 * r, fc='none', ec='#b8860b', lw=2, ls='--', zorder=3))
    ax.add_patch(Rectangle((mu[0] - half[0], mu[1] - half[1]), 2 * half[0], 2 * half[1], fc='none', ec='#d95f02',
                           lw=2, zorder=3))
    ax.add_patch(ellipse_patch(mu, cov, t, fc='none', ec='#1b7837', lw=2.2, zorder=4))
    ax.add_patch(ellipse_patch(mu, cov, 1.0, fc='none', ec=COL[0], lw=1, ls=':', zorder=4))
    ax.plot(mu[0], mu[1], 'o', color=COL[0], ms=6, mec='k', mew=0.6, zorder=5)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    return K3, fg['K_box'], fg['count'], r_min, r_max, fg


def fig_boxes():
    R = T.RESULT[(1, 0)]
    mu, cov, t, r, half = R['mu2'], R['cov'], R['t'], R['r'], R['half']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[1, 1.35]))
    # panel 1: ảnh thật 48x32
    K3, Kb, Kf, rmin, rmax, fg = draw_boxes_panel(ax1, T.W, T.H, mu, cov, t, r, half, T.ALPHA, "")
    ax1.set_title(f"G1, camera 1, ảnh $48\\times32$ (lưới $3\\times2$): mọi hộp bị clamp vào lưới\n"
                  f"$K^{{\\mathrm{{3dgs}}}}$ = {K3} (clamp)   $K^{{\\mathrm{{box}}}}$ = {Kb}   $K^{{\\mathrm{{fastgs}}}}$ = {Kf}",
                  fontsize=9.5)
    ax1.set_xlim(-8, 56)
    ax1.set_ylim(52, -20)
    ax1.text(mu[0] - r + 0.5, mu[1] - r - 0.8, f"hộp 3DGS: $r=\\lceil3\\sqrt{{\\lambda_{{\\max}}}}\\rceil$ = {r}, cạnh {2*r}",
             fontsize=8, color='#8a6508', va='bottom')
    ax1.text(mu[0] - half[0] + 0.5, mu[1] - half[1] - 0.8,
             f"compact box: half = $\\sqrt{{t\\,\\Sigma'_{{11}}}}$ = {half[0]:.2f}, cạnh {2*half[0]:.1f}",
             fontsize=8, color='#d95f02', va='bottom')
    ax1.text(mu[0] + half[0] + 1.0, mu[1] + half[1] - 1,
             f"$\\mu'$ = ({mu[0]}, {mu[1]})\n$t$ = {t:.3f}\n$\\sqrt{{t}}\\sigma'$ = {half[0]:.2f}",
             fontsize=8, color='#1b7837', va='bottom', ha='left')
    ax1.text(-7, 50.5, f"vàng: hộp 3DGS chạm; cam: compact box chạm; xanh: giữ sau lọc ellipse.\n"
             f"Ở $48\\times32$ ba tập trùng nhau (cả 6 tile) → $R_{{\\mathrm{{tile}}}}$ = 1",
             fontsize=7.5, va='bottom', color='0.25')
    # panel 2: ảnh giả định 160x128, tâm đặt ở giữa ảnh
    W2, H2 = 160, 128
    mu2 = np.array([(W2 - 1) / 2, (H2 - 1) / 2])
    K3b, Kbb, Kfb, rmin2, rmax2, fg2 = draw_boxes_panel(ax2, W2, H2, mu2, cov, t, r, half, T.ALPHA, "")
    ax2.set_title(
        f"Cùng $\\Sigma'$, ảnh giả định $160\\times128$ (lưới $10\\times8$), $\\mu'$ = ({mu2[0]}, {mu2[1]}), không clamp\n"
        f"$K^{{\\mathrm{{3dgs}}}}$ = {K3b} ({rmax2[0]-rmin2[0]}$\\times${rmax2[1]-rmin2[1]})   "
        f"$K^{{\\mathrm{{box}}}}$ = {Kbb} ({fg2['x_span']}$\\times${fg2['y_span']})   "
        f"$K^{{\\mathrm{{fastgs}}}}$ = {Kfb}   →  $R_{{\\mathrm{{tile}}}}$ = {Kfb}/{K3b} = {Kfb/K3b:.3f}", fontsize=9.5)
    ax2.set_xlim(-0.5, W2 + 0.5)
    ax2.set_ylim(H2 + 0.5, -0.5)
    ax2.set_xticks(range(0, W2 + 1, 16))
    ax2.set_yticks(range(0, H2 + 1, 16))
    ax2.tick_params(labelsize=7)
    ax2.text(mu2[0] - r + 1, mu2[1] - r - 1.5, f"hộp $3\\sigma$: {2*r}$\\times${2*r} px", fontsize=8, color='#8a6508',
             va='bottom')
    ax2.text(mu2[0] + half[0] + 1, mu2[1] - half[1] + 3, f"compact box\n{2*half[0]:.1f}$\\times${2*half[1]:.1f} px",
             fontsize=8, color='#d95f02', va='top')
    ax2.text(mu2[0] + 1.5, mu2[1] - 2, r"ellipse $\Delta^\top M\Delta = t$", fontsize=8, color='#1b7837')
    ax2.text(mu2[0] - r, mu2[1] + r + 3, "diện tích: hộp 3DGS 2704 px² → compact box "
             f"{(2*half[0])*(2*half[1]):.0f} px² (−{100*(1-(2*half[0])*(2*half[1])/(4*r*r)):.0f}%)",
             fontsize=8, va='top', color='0.25')
    fig.suptitle("Ba lớp bao splat: hộp vuông 3DGS (vàng, nét đứt) ⊃ compact box FastGS (cam) ⊃ ellipse level-set (xanh)",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "boxes")


# =========================================================================================
# 4. ch03_boxmd.png  (ví dụ box.md)
# =========================================================================================
def fig_boxmd():
    cov = np.array([[117.0, 54.0], [54.0, 36.0]])
    mu = np.array([120.0, 88.0])
    grid = (20, 20)
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    A, B, C = cov[1, 1] / det, -cov[0, 1] / det, cov[0, 0] / det
    mid = 0.5 * (cov[0, 0] + cov[1, 1])
    lam_max = mid + math.sqrt(mid * mid - det)
    lam_min = mid - math.sqrt(mid * mid - det)
    r = math.ceil(3 * math.sqrt(lam_max))
    rmin, rmax = T.getRect_3dgs(mu, r, grid)
    K3 = (rmax[0] - rmin[0]) * (rmax[1] - rmin[1])
    fg = T.duplicateToTilesTouched(mu, (A, B, C, 1.0), grid, T.MULT)
    t = fg['t']
    hx, hy = math.sqrt(t * cov[0, 0]), math.sqrt(t * cov[1, 1])
    ids3 = {ty * grid[0] + tx for ty in range(rmin[1], rmax[1]) for tx in range(rmin[0], rmax[0])}
    idsb = {ty * grid[0] + tx for ty in range(fg['rect_min'][1], fg['rect_max'][1])
            for tx in range(fg['rect_min'][0], fg['rect_max'][0])}
    idsf = set(fg['ids'])

    fig, ax = plt.subplots(figsize=(13, 8.5))
    x0, x1 = rmin[0] * BLK - 6, rmax[0] * BLK + 78
    y0, y1 = rmin[1] * BLK - 20, rmax[1] * BLK + 6
    for tid in ids3:
        tx, ty = tid % grid[0], tid // grid[0]
        if tid in idsf:
            fc = '#8fd18f'
        elif tid in idsb:
            fc = '#ffb060'
        else:
            fc = '#ffe680'
        ax.add_patch(Rectangle((tx * BLK, ty * BLK), BLK, BLK, fc=fc, ec='k', lw=0.6, alpha=0.9, zorder=0))
        ax.text(tx * BLK + 1, ty * BLK + 1.5, f"({tx},{ty})", fontsize=7, va='top', color='0.35')
    for x in range(int(x0) // BLK * BLK, int(x1) + BLK, BLK):
        ax.axvline(x, color='k', lw=0.5, alpha=0.4)
    for y in range(int(y0) // BLK * BLK, int(y1) + BLK, BLK):
        ax.axhline(y, color='k', lw=0.5, alpha=0.4)
    ax.add_patch(Rectangle((mu[0] - r, mu[1] - r), 2 * r, 2 * r, fc='none', ec='#b8860b', lw=2.2, ls='--', zorder=3))
    ax.add_patch(Rectangle((mu[0] - hx, mu[1] - hy), 2 * hx, 2 * hy, fc='none', ec='#d95f02', lw=2.2, zorder=3))
    ax.add_patch(ellipse_patch(mu, cov, 9.0, fc='none', ec='#7b3294', lw=1.6, ls='-.', zorder=4))
    ax.add_patch(ellipse_patch(mu, cov, t, fc='none', ec='#1b7837', lw=2.4, zorder=4))
    ax.plot(mu[0], mu[1], 'o', color='k', ms=6, zorder=5)
    ax.text(mu[0] + 1.5, mu[1] - 1.5, "$\\mu'$ = (120, 88)", fontsize=9, fontweight='bold', va='bottom')
    # tiếp điểm x_term / y_term
    xt, yt = fg['x_term'], fg['y_term']
    ax.plot([mu[0] + hx, mu[0] - hx], [mu[1] + yt, mu[1] - yt], 's', color='#d95f02', ms=5, zorder=6)
    ax.plot([mu[0] + xt, mu[0] - xt], [mu[1] + hy, mu[1] - hy], 's', color='#d95f02', ms=5, zorder=6)
    ax.annotate(f"tiếp điểm ($x_{{\\mathrm{{term}}}}$={xt:.2f}, half$_y$={hy:.2f})", (mu[0] + xt, mu[1] + hy),
                xytext=(44, -26), textcoords='offset points', fontsize=8, color='#d95f02',
                arrowprops=dict(arrowstyle='->', color='#d95f02', lw=0.8))
    # nhãn hình
    ax.text(rmin[0] * BLK + 1, rmin[1] * BLK - 1, f"hộp 3DGS: $r=\\lceil3\\sqrt{{144}}\\rceil$ = {r}, {2*r}$\\times${2*r} px → "
            f"rect ({rmin[0]},{rmin[1]})→({rmax[0]},{rmax[1]}) = {rmax[0]-rmin[0]}$\\times${rmax[1]-rmin[1]} = {K3} tile",
            fontsize=8.5, color='#8a6508', va='bottom')
    ax.text(mu[0] - hx, mu[1] + hy + 1.5, f"compact box: half = ({hx:.2f}, {hy:.2f}), {2*hx:.2f}$\\times${2*hy:.2f} px → "
            f"rect ({fg['rect_min'][0]},{fg['rect_min'][1]})→({fg['rect_max'][0]},{fg['rect_max'][1]}) = "
            f"{fg['x_span']}$\\times${fg['y_span']} = {fg['K_box']} tile",
            fontsize=8.5, color='#d95f02', va='top')
    ax.text(rmax[0] * BLK + 5, rmin[1] * BLK + 8, "ellipse $3\\sigma$ (tím, chấm gạch): $\\Delta^\\top M\\Delta = 9$\n"
            f"ellipse level-set (xanh): $\\Delta^\\top M\\Delta = t$ = {t:.3f}\n"
            f"tile giữ sau lọc (xanh lá) = {fg['count']}", fontsize=8.5, ha='left', va='top', color='#1b7837',
            bbox=dict(boxstyle='round', fc='white', ec='#1b7837', alpha=0.9))
    ax.text(x0 + 2, y0 + 1.5,
            f"$\\Sigma'$ = (117, 54, 36), $\\lambda$ = ({lam_max:.0f}, {lam_min:.0f}), $\\rho$ = {math.sqrt(lam_max/lam_min):.0f}, "
            f"$\\alpha$ = 1, mult = 0.5, $t$ = 0.5·2ln(255) = {t:.3f}\n"
            f"$K$: {K3} (3DGS) → {fg['K_box']} (compact box) → {fg['count']} (lọc ellipse);   "
            f"$R_{{\\mathrm{{tile}}}}$ = {fg['count']}/{K3} = {fg['count']/K3:.2f}",
            fontsize=9.5, va='top', fontweight='bold',
            bbox=dict(boxstyle='round', fc='#fff7e6', ec='#e0a040'))
    # legend giả
    from matplotlib.lines import Line2D
    handles = [Rectangle((0, 0), 1, 1, fc='#ffe680', ec='k', lw=0.5), Rectangle((0, 0), 1, 1, fc='#ffb060', ec='k', lw=0.5),
               Rectangle((0, 0), 1, 1, fc='#8fd18f', ec='k', lw=0.5),
               Line2D([], [], color='#b8860b', ls='--', lw=2), Line2D([], [], color='#d95f02', lw=2),
               Line2D([], [], color='#7b3294', ls='-.', lw=1.6), Line2D([], [], color='#1b7837', lw=2.4)]
    labels = [f"chỉ hộp 3DGS chạm ({len(ids3 - idsb)} tile)", f"compact box chạm nhưng ellipse bỏ ({len(idsb - idsf)} tile)",
              f"giữ sau lọc ellipse ({len(idsf)} tile)", "hộp vuông 3DGS $r$ = 36", "compact box 50.9 × 28.2",
              "ellipse $3\\sigma$", "ellipse $\\Delta^\\top M\\Delta = t$"]
    ax.legend(handles, labels, fontsize=8, loc='upper right', framealpha=0.95)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks(range(int(x0) // BLK * BLK + BLK, int(x1), BLK))
    ax.set_yticks(range(int(y0) // BLK * BLK + BLK, int(y1), BLK))
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel, hướng xuống)")
    ax.set_title("Ví dụ box.md (§3.8): $\\Sigma'$ dẹt ($\\rho$ = 4) — 25 → 15 → 9 tile, $R_{\\mathrm{tile}}$ = 0.36", fontsize=11)
    save(fig, "boxmd")


# =========================================================================================
# 5. ch03_t_vs_alpha.png
# =========================================================================================
def fig_t_vs_alpha():
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    alphas = np.linspace(1 / 255, 1.0, 600)
    for mult, col, ls in [(0.5, '#1b7837', '-'), (1.0, '#7b3294', '--')]:
        tv = mult * 2 * np.log(255 * alphas)
        ax.plot(alphas, tv, color=col, lw=2, ls=ls, label=f"mult = {mult}: $t = {mult}\\cdot2\\ln(255\\alpha)$")
    # vùng alpha < 1/255
    a_lo = np.linspace(0.002, 1 / 255, 50)
    ax.plot(a_lo, 0.5 * 2 * np.log(255 * a_lo), color='#1b7837', lw=1.2, alpha=0.5)
    ax.axvspan(0.001, 1 / 255, color='#d62728', alpha=0.10)
    ax.axvline(1 / 255, color='#d62728', lw=1, ls=':')
    ax.axhline(0, color='k', lw=0.8)
    ax.text(1 / 255 * 1.05, -0.55, r"$\alpha<1/255 \Rightarrow 255\alpha<1 \Rightarrow t<0$" "\n(không có ellipse → không rasterize;\nngưỡng alpha 1/255 của chương 0)",
            fontsize=8, color='#d62728', va='top')
    marks = [(0.02, '#7f7f7f'), (0.1, '#d62728'), (1.0, '#1f77b4')]
    for a, col in marks:
        tv = 0.5 * 2 * math.log(255 * a)
        ax.plot(a, tv, 'o', color=col, ms=8, mec='k', mew=0.7, zorder=5)
        ax.annotate(f"$\\alpha$ = {a:g} → $t$ = {tv:.2f}, $\\sqrt{{t}}$ = {math.sqrt(tv):.2f}$\\sigma'$",
                    (a, tv), xytext=(12, -18 if a < 0.5 else 10), textcoords='offset points', fontsize=8.5, color=col,
                    fontweight='bold', arrowprops=dict(arrowstyle='->', color=col, lw=0.8))
        tv1 = 2 * math.log(255 * a)
        ax.plot(a, tv1, 'o', color=col, ms=5, mfc='white', mec=col, mew=1.2, zorder=5)
        ax.annotate(f"{tv1:.2f}", (a, tv1), xytext=(6, 4), textcoords='offset points', fontsize=7.5, color='#7b3294')
    ax.text(0.1, 6.15, "toy scene: $\\alpha$ = 0.1 (khởi tạo) → $t$ = 3.24 → half = $\\sqrt{3.24\\cdot72.6}$ = 15.3 px",
            fontsize=8.5, color='0.25')
    ax.set_xscale('log')
    ax.set_xlim(0.0025, 1.15)
    ax.set_ylim(-1.2, 12)
    ax.set_xlabel(r"$\alpha$ (opacity, thang log)")
    ax.set_ylabel(r"$t$ (ngưỡng level-set $\Delta^\top M\Delta \leq t$)")
    ax.set_title(r"Ngưỡng $t = \mathrm{mult}\cdot 2\ln(255\alpha)$ theo opacity; hộp = $\mu' \pm \sqrt{t\,\Sigma'_{jj}}$", fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    # trục phụ: bán kính hiệu dụng sqrt(t) (đơn vị sigma')
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    tick_t = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]
    ax2.set_yticks(tick_t)
    ax2.set_yticklabels([f"{math.sqrt(v):.2f}" if v > 0 else "0" for v in tick_t])
    ax2.set_ylabel(r"bán kính hiệu dụng $\sqrt{t}$ (đơn vị $\sigma'$); 3DGS cố định $3\sigma'$ ($t$ = 9)")
    ax2.grid(False)
    ax.axhline(9, color='#b8860b', lw=1.2, ls='--')
    ax.text(0.0027, 9.15, "hộp 3DGS: $r = 3\\sigma'$ ⇔ $t$ = 9 (không phụ thuộc $\\alpha$)", fontsize=8, color='#8a6508')
    save(fig, "t_vs_alpha")


# =========================================================================================
# 6. ch03_rtile.png
# =========================================================================================
def fig_rtile():
    cams = [1, 2, 3]
    s3 = [sum(T.RESULT[(v, i)]['K3'] for i in range(T.N)) for v in cams]
    sb = [sum(T.RESULT[(v, i)]['fg']['K_box'] for i in range(T.N)) for v in cams]
    sf = [sum(T.RESULT[(v, i)]['fg']['count'] for i in range(T.N)) for v in cams]
    s3.append(sum(s3)); sb.append(sum(sb)); sf.append(sum(sf))
    labels = ["cam 1", "cam 2", "cam 3", "3 camera"]
    x = np.arange(4)
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    b1 = ax.bar(x - w, s3, w, color='#ffe680', ec='#b8860b', lw=1.2, label=r"$\sum K^{\mathrm{3dgs}}$ (hộp vuông $3\sigma$)")
    b2 = ax.bar(x, sb, w, color='#ffb060', ec='#d95f02', lw=1.2, label=r"$\sum K^{\mathrm{box}}$ (compact box)")
    b3 = ax.bar(x + w, sf, w, color='#8fd18f', ec='#1b7837', lw=1.2, label=r"$\sum K^{\mathrm{fastgs}}$ (sau lọc ellipse)")
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{int(b.get_height())}", ha='center',
                    va='bottom', fontsize=9.5, fontweight='bold')
    for k in range(4):
        rt = sf[k] / s3[k]
        ax.text(x[k], max(s3[k], sb[k], sf[k]) + 6.5, f"$R_{{\\mathrm{{tile}}}}$ = {sf[k]}/{s3[k]} = {rt:.3f}",
                ha='center', fontsize=9.5, color='#1b7837', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#1b7837'))
    ax.text(1, 43, "lọc ellipse chỉ bớt 1 tile (cam 3, G3, tile 3: 3.562 > t = 3.239);\ncompact box bớt 14 tile (72 → 58) vì $\\rho\\approx1$",
            fontsize=8.5, ha='center', va='top', color='0.25', bbox=dict(boxstyle='round', fc='#f5f5f5', ec='0.6'))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("tổng số tile chạm  $\\sum_i K_i$  (4 Gaussian)")
    ax.set_ylim(0, 92)
    ax.set_title("Số tile chạm theo 3 cách đếm và $R_{\\mathrm{tile}} = \\sum K^{\\mathrm{fastgs}} / \\sum K^{\\mathrm{3dgs}}$", fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='x')
    save(fig, "rtile")


if __name__ == "__main__":
    fig_projection()
    fig_jacobian()
    fig_boxes()
    fig_boxmd()
    fig_t_vs_alpha()
    fig_rtile()
