"""
Vẽ hình minh hoạ cho bài test số chương 6 (Gradient Flow).
Số liệu lấy lại bằng cách import scripts/ch06_test.py (chỉ numpy, không torch) —
toàn bộ biến trung gian (bảng 6.1, 6.3, 6.4) được dùng trực tiếp, không chép tay số.
Chạy:  python DOCS/Report/test/scripts/ch06_plot.py
Xuất:  DOCS/Report/test/figures/ch06_*.png
"""
import os
import io
import sys
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, HERE)

# Import ch06_test (script in rất nhiều -> nuốt stdout)
with contextlib.redirect_stdout(io.StringIO()):
    import ch06_test as T

GCOL = ['#d62728', '#2ca02c', '#1f77b4', '#7f7f7f']
GNAME = ['G1', 'G2', 'G3', 'G4']


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=450, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("saved", os.path.normpath(path))


def sci(x, d=3):
    """1.234e-04 -> mathtext 1.234×10^{-4}"""
    if x == 0:
        return "0"
    m, e = f"{x:.{d}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


# ==========================================================================
# 1. ch06_dC_dalpha.png — 3 số hạng của dC/dalpha_n tại pixel (23,15), kênh R
# ==========================================================================
def fig_dC_dalpha():
    PX, PY = T.PX, T.PY
    lst = T.lst
    items = [(T.col1[e["n"]], e["a"]) for e in lst]
    C_tot, T_final = T.blend_pixel(items)
    h = T.H_FD
    terms = []; ana = []; fd = []; err = []; gidx = []
    for j, e, Cle in T.rows_61:
        n = e["n"]; a = e["a"]; Tn = e["T"]
        t1 = T.col1[n] * Tn
        t2 = -(C_tot - Cle) / (1 - a)
        t3 = -T_final * T.BG / (1 - a)
        dC = t1 + t2 + t3
        it_p = list(items); it_p[j] = (T.col1[n], a + h)
        it_m = list(items); it_m[j] = (T.col1[n], a - h)
        Cp, Tp = T.blend_pixel(it_p); Cm, Tm = T.blend_pixel(it_m)
        dC_fd = ((Cp + Tp * T.BG) - (Cm + Tm * T.BG)) / (2 * h)
        terms.append((t1[0], t2[0], t3[0])); ana.append(dC[0]); fd.append(dC_fd[0])
        err.append(max(T.relerr(dC[c], dC_fd[c]) for c in range(3))); gidx.append(n)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(4); w = 0.24
    tcol = ['#ff9896', '#c5b0d5', '#c49c94']
    tlab = [r'(1) $c_n T_n$', r'(2) $-(C^{\mathrm{tot}}-C^{\leq n})/(1-\alpha_n)$',
            r'(3) $-T_{\mathrm{final}}C_{\mathrm{bg}}/(1-\alpha_n)$']
    for k in range(3):
        vals = [t[k] for t in terms]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=tcol[k], edgecolor='k', linewidth=0.6, label=tlab[k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.02), f"{v:+.4f}",
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=7.5)
    ax.plot(x, ana, 'o', color='k', ms=9, label=r'analytic $\partial C_R/\partial\alpha_n$ = (1)+(2)+(3)', zorder=5)
    ax.plot(x, fd, 'x', color='#e6550d', ms=11, mew=2.2, label=r'FD ($h=10^{-6}$)', zorder=6)
    for i in range(4):
        ax.text(x[i] + 0.36, ana[i], f"{ana[i]:+.5f}\nsai số tđ {err[i]:.1e}", fontsize=7.5, va='center',
                ha='left', color='k', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={i+1}\n{GNAME[g]} ($\\alpha_n$={lst[i]['a']:.4f})" for i, g in enumerate(gidx)])
    for lab, g in zip(ax.get_xticklabels(), gidx):
        lab.set_color(GCOL[g])
    ax.set_ylabel(r'$\partial C_R/\partial\alpha_n$ (kênh R)')
    ax.set_title(rf'Ba số hạng của $\partial C/\partial\alpha_n$ tại pixel $x=({PX},{PY})$, camera 1 — '
                 rf'$T_{{\mathrm{{final}}}}={T_final:.4f}$, $C^{{\mathrm{{tot}}}}_R={C_tot[0]:.4f}$', fontsize=10.5)
    ax.set_ylim(-0.95, 1.0)
    ax.set_xlim(-0.6, 3.9)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.text(0.01, 0.02, 'Số hạng nền (3) lớn nhất ở mọi n; bỏ nó thì n=1 đổi dấu (+0.737 thay vì −0.029).\n'
            'n=4 (G3, contributor cuối) có số hạng (2) = 0.',
            transform=ax.transAxes, fontsize=8, va='bottom', bbox=dict(fc='#fff8e1', ec='#999'))
    save(fig, "ch06_dC_dalpha.png")


# ==========================================================================
# 2. ch06_grad_field.png — trường gradient per-pixel dL/dmu'_1 của G1 (camera 1)
# ==========================================================================
def per_pixel_grad(cam_idx, ndc_all, conic_all, alpha_all, colors, depth_all, gt, target):
    """Giống backward_mean2d nhưng trả về (H,W,2) gradient dL/dmu'_ndc của Gaussian `target` tại từng pixel."""
    W, H = T.W, T.H
    img, contribs = T.render(cam_idx, ndc_all, conic_all, alpha_all, colors, depth_all, record=True)
    dLdC = np.sign(img - gt) / (3 * H * W)
    field = np.zeros((H, W, 2))
    for py in range(H):
        for px in range(W):
            lst = contribs[py][px]
            if not lst:
                continue
            items = [(colors[e["n"]], e["a"]) for e in lst]
            C_tot, T_final = T.blend_pixel(items)
            Cle = np.zeros(3)
            for e in lst:
                n = e["n"]; a = e["a"]; Tn = e["T"]
                Cle = Cle + colors[n] * a * Tn
                if n != target:
                    continue
                dC = colors[n] * Tn - (C_tot - Cle) / (1 - a) - T_final * T.BG / (1 - a)
                dL_dG = alpha_all[n] * float(dLdC[py, px] @ dC)
                A, B, Cc = conic_all[n]; du, dv, G = e["du"], e["dv"], e["G"]
                field[py, px, 0] = dL_dG * (-G * (A * du + B * dv)) * (0.5 * W)
                field[py, px, 1] = dL_dG * (-G * (Cc * dv + B * du)) * (0.5 * H)
    return field, img


def fig_grad_field():
    field, _ = per_pixel_grad(0, T.ndc1, T.conic1, T.alpha1, T.col1, T.depth1, T.GT1, target=0)
    g = field.sum((0, 1)); gabs = np.abs(field).sum((0, 1))
    # kiểm tra khớp với 4 cột của ch06_test
    assert np.allclose(g, T.G1[0, :2]) and np.allclose(gabs, T.G1[0, 2:])
    ng, nga = np.linalg.norm(g), np.linalg.norm(gabs)
    mag = np.hypot(field[..., 0], field[..., 1])
    mu1 = T.prj1[0]["mu_pix"]
    W, H = T.W, T.H
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    # --- (a) quiver
    ax = axes[0]
    ax.set_facecolor('#f7f7f7')
    ax.grid(False)
    u, v = field[..., 0], field[..., 1]
    sgn = np.sign(u)
    cmap_pts = np.where(sgn < 0, '#1f77b4', '#d62728')
    q = ax.quiver(X, Y, u, v, color=cmap_pts.ravel(), angles='xy', scale_units='xy',
                  scale=1.4e-4, width=0.0035, headwidth=3.5, headlength=4.5, minlength=0.05)
    ax.plot(mu1[0], mu1[1], marker='*', color=GCOL[0], ms=16, mec='k', zorder=5)
    ax.axvline(mu1[0], color=GCOL[0], ls='--', lw=1, alpha=0.7)
    ax.text(mu1[0] + 0.6, 0.6, rf"$\mu'_1=({mu1[0]:.2f},{mu1[1]:.2f})$", color=GCOL[0], fontsize=9, va='top')
    ax.set_xlim(-1, W); ax.set_ylim(H, -1)
    ax.set_aspect('equal')
    ax.set_xlabel('u (px)'); ax.set_ylabel('v (px)')
    ax.set_title(r"(a) Trường gradient per-pixel $\partial\mathcal{L}/\partial\mu'_1$ (G1, camera 1)", fontsize=10.5)
    # Mũi tên hội tụ về tâm từ hai phía: nửa trái có g_u>0 (chỉ sang phải, đỏ),
    # nửa phải có g_u<0 (chỉ sang trái, xanh) — tổng có dấu vì thế gần triệt tiêu.
    nleft = int((u[:, :24] > 0).sum()); nright = int((u[:, 24:] < 0).sum())
    ax.text(0.02, 0.97, f"nửa trái: {nleft} px có $g_u>0$ (mũi tên → phải, về phía tâm)", transform=ax.transAxes,
            fontsize=8.5, color='#d62728', va='top', bbox=dict(fc='white', ec='#d62728', alpha=0.9))
    ax.text(0.98, 0.97, f"nửa phải: {nright} px có $g_u<0$ (mũi tên → trái, về phía tâm)", transform=ax.transAxes,
            fontsize=8.5, color='#1f77b4', va='top', ha='right', bbox=dict(fc='white', ec='#1f77b4', alpha=0.9))
    ax.legend(handles=[Line2D([], [], color='#1f77b4', lw=2, label=r'$g_u<0$'),
                       Line2D([], [], color='#d62728', lw=2, label=r'$g_u>0$'),
                       Line2D([], [], marker='*', color=GCOL[0], mec='k', ms=12, ls='', label='tâm G1')],
              loc='lower left', fontsize=8)
    # --- (b) heatmap |g|
    ax = axes[1]
    ax.grid(False)
    im = ax.imshow(mag, cmap='magma', origin='upper', interpolation='nearest')
    ax.plot(mu1[0], mu1[1], marker='*', color='cyan', ms=14, mec='k')
    cb = fig.colorbar(im, ax=ax, fraction=0.037, pad=0.02)
    cb.set_label(r"$\|\partial\mathcal{L}/\partial\mu'_1\|_x$")
    cb.formatter.set_powerlimits((0, 0)); cb.update_ticks()
    ax.set_xlabel('u (px)'); ax.set_ylabel('v (px)')
    ax.set_title(r"(b) Độ lớn $|\partial\mathcal{L}/\partial\mu'_1|$ tại từng pixel", fontsize=10.5)
    ax.text(0.5, -0.16,
            rf"$\|\sum_x g_x\| = \|g\| = {sci(ng)}$   vs   $\|\sum_x |g_x|\| = \|g^{{\mathrm{{abs}}}}\| = {sci(nga)}$"
            f"\n$g=({g[0]:+.3e},\\ {g[1]:+.3e})$,  $g^{{\\mathrm{{abs}}}}=({gabs[0]:.3e},\\ {gabs[1]:.3e})$  →  tỉ số {nga/ng:.1f}×",
            transform=ax.transAxes, fontsize=9.5, ha='center', va='top',
            bbox=dict(fc='#fff8e1', ec='#999'))
    ax.text(0.5, -0.36, "Tổng có dấu triệt tiêu 77× vì hai nửa footprint kéo ngược chiều — 3DGS gốc (cột 0–1) không thấy,\n"
            "FastGS (cột 2–3, trị tuyệt đối) thấy.",
            transform=ax.transAxes, fontsize=8.5, ha='center', va='top', style='italic')
    save(fig, "ch06_grad_field.png")


# ==========================================================================
# 3. ch06_4cols.png — ‖g‖ vs ‖g_abs‖ cho 4 Gaussian; ḡ, ḡ_abs sau 3 camera
# ==========================================================================
def fig_4cols():
    G1 = T.G1
    ng = np.linalg.norm(G1[:, :2], axis=1); nga = np.linalg.norm(G1[:, 2:], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw=dict(width_ratios=[1, 1.15]))
    x = np.arange(4); w = 0.36
    # (a)
    ax = axes[0]
    b1 = ax.bar(x - w / 2, ng, w, color=[c for c in GCOL], alpha=0.45, edgecolor='k', hatch='//',
                label=r'$\|g\|$ có dấu (cột 0–1)')
    b2 = ax.bar(x + w / 2, nga, w, color=[c for c in GCOL], edgecolor='k', label=r'$\|g^{\mathrm{abs}}\|$ (cột 2–3)')
    ax.set_yscale('log')
    for i in range(4):
        ax.text(x[i] - w / 2, ng[i] * 1.15, f"{ng[i]:.2e}", ha='center', fontsize=7.5, rotation=90, va='bottom')
        ax.text(x[i] + w / 2, nga[i] * 1.15, f"{nga[i]:.2e}", ha='center', fontsize=7.5, rotation=90, va='bottom')
        ax.annotate(f"×{nga[i]/ng[i]:.1f}", (x[i], np.sqrt(ng[i] * nga[i])), ha='center', va='center', fontsize=9,
                    fontweight='bold', bbox=dict(fc='white', ec='k', boxstyle='round,pad=0.2'))
    ax.axhline(T.TAU_GRAD, color='#1f77b4', ls='--', lw=1.3)
    ax.text(3.45, T.TAU_GRAD * 1.1, r'$\tau_{\mathrm{grad}}=2\times10^{-4}$', color='#1f77b4', fontsize=8.5, ha='right')
    ax.axhline(T.TAU_ABS, color='#d62728', ls='--', lw=1.3)
    ax.text(3.45, T.TAU_ABS * 1.1, r'$\tau^{\mathrm{abs}}_{\mathrm{grad}}=1.2\times10^{-3}$', color='#d62728', fontsize=8.5, ha='right')
    ax.set_xticks(x); ax.set_xticklabels(GNAME)
    for lab, c in zip(ax.get_xticklabels(), GCOL):
        lab.set_color(c); lab.set_fontweight('bold')
    ax.set_ylim(1e-4, 0.3)
    ax.set_ylabel('chuẩn L2 của gradient (camera 1)')
    ax.set_title(r'(a) Gradient 4 cột, camera 1: $\|g\|$ vs $\|g^{\mathrm{abs}}\|$', fontsize=10.5)
    ax.legend(handles=[Patch(fc='#bbbbbb', ec='k', hatch='//', label=r'$\|g\|$ có dấu (cột 0–1)'),
                       Patch(fc='#555555', ec='k', label=r'$\|g^{\mathrm{abs}}\|$ trị tuyệt đối (cột 2–3)')],
              fontsize=8.5, loc='upper left')
    # (b) accumulate over 3 cameras
    ax = axes[1]
    per_cam = np.array([[np.linalg.norm(g[i, :2]) for i in range(4)] for g in T.G_all])       # (3,4)
    per_cam_abs = np.array([[np.linalg.norm(g[i, 2:]) for i in range(4)] for g in T.G_all])
    camcol = ['#444444', '#ff7f0e', '#9467bd']
    bottom = np.zeros(4); bottom_a = np.zeros(4)
    for v in range(3):
        ax.bar(x - w / 2, per_cam[v], w, bottom=bottom, color=camcol[v], alpha=0.5, edgecolor='k', hatch='//',
               label=f'cam {v+1}: $\\|g\\|$' if v == 0 else None)
        ax.bar(x + w / 2, per_cam_abs[v], w, bottom=bottom_a, color=camcol[v], edgecolor='k',
               label=f'cam {v+1}' )
        bottom += per_cam[v]; bottom_a += per_cam_abs[v]
    for i in range(4):
        ax.text(x[i] - w / 2, bottom[i] + 0.002, f"accum\n{T.accum[i]:.3e}\n$\\bar g$={T.gbar[i]:.3e}", ha='center',
                va='bottom', fontsize=7.2)
        ax.text(x[i] + w / 2, bottom_a[i] + 0.002, f"accum$^{{\\mathrm{{abs}}}}$\n{T.accum_abs[i]:.3e}\n$\\bar g^{{\\mathrm{{abs}}}}$={T.gbar_abs[i]:.3e}",
                ha='center', va='bottom', fontsize=7.2)
    ax.axhline(3 * T.TAU_GRAD, color='#1f77b4', ls='--', lw=1.2)
    ax.text(-0.55, 3 * T.TAU_GRAD + 0.0015, r'$3\tau_{\mathrm{grad}}=6\times10^{-4}$', color='#1f77b4', fontsize=8)
    ax.axhline(3 * T.TAU_ABS, color='#d62728', ls='--', lw=1.2)
    ax.text(-0.55, 3 * T.TAU_ABS + 0.0015, r'$3\tau^{\mathrm{abs}}=3.6\times10^{-3}$', color='#d62728', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(GNAME)
    for lab, c in zip(ax.get_xticklabels(), GCOL):
        lab.set_color(c); lab.set_fontweight('bold')
    ax.set_ylim(0, 0.135)
    ax.set_ylabel(r'accum $=\sum_v \|g^{(v)}\|$ (trái, gạch) · accum$^{\mathrm{abs}}$ (phải)')
    ax.set_title(r'(b) Tích luỹ 3 camera: $\bar g=\mathrm{accum}/3$ — cả 4 Gaussian vượt 2 ngưỡng', fontsize=10.5)
    ax.legend(handles=[Patch(fc=camcol[v], ec='k', label=f'camera {v+1}') for v in range(3)], fontsize=8.5, loc='upper left')
    fig.tight_layout()
    save(fig, "ch06_4cols.png")


# ==========================================================================
# 4. ch06_fd_check.png — scatter analytic vs FD, mọi đại lượng đã kiểm
# ==========================================================================
def fig_fd_check():
    pts = []   # (label, analytic, fd, group)
    # dC/dalpha (3 kênh × 4 n) — tính lại như fig 1
    items = [(T.col1[e["n"]], e["a"]) for e in T.lst]
    C_tot, T_final = T.blend_pixel(items); h = T.H_FD
    for j, e, Cle in T.rows_61:
        n = e["n"]; a = e["a"]; Tn = e["T"]
        dC = T.col1[n] * Tn - (C_tot - Cle) / (1 - a) - T_final * T.BG / (1 - a)
        it_p = list(items); it_p[j] = (T.col1[n], a + h)
        it_m = list(items); it_m[j] = (T.col1[n], a - h)
        Cp, Tp = T.blend_pixel(it_p); Cm, Tm = T.blend_pixel(it_m)
        dC_fd = ((Cp + Tp * T.BG) - (Cm + Tm * T.BG)) / (2 * h)
        for c in range(3):
            pts.append((rf"$\partial C/\partial\alpha$", dC[c], dC_fd[c], 0))
    for (j, n, dL_dG, dG_du, dG_du_fd, dG_dv, dG_dv_fd, dL_dA, fdA, dL_dB, fdB, dL_dCc, fdC) in T.rows_grad:
        pts.append((r"$\partial G/\partial d$", dG_du, dG_du_fd, 1))
        pts.append((r"$\partial G/\partial d$", dG_dv, dG_dv_fd, 1))
        pts.append((r"$\partial\mathcal{L}_x/\partial A,C$", dL_dA, fdA, 2))
        pts.append((r"$\partial\mathcal{L}_x/\partial A,C$", dL_dCc, fdC, 2))
        pts.append((r"$\partial\mathcal{L}_x/\partial B$ thực $=2\times$code", 2 * dL_dB, fdB, 3))
        pts.append((r"$\partial\mathcal{L}_x/\partial B$ code ($\frac{1}{2}$)", dL_dB, fdB, 4))
    for (i, name, ga, gfd) in T.fd_rows:
        pts.append((r"$g$ 8 thành phần (toàn ảnh, theo NDC)", ga, gfd, 5))

    groups = {}
    for lab, a, f, g in pts:
        groups.setdefault(g, [lab, [], []]); groups[g][1].append(a); groups[g][2].append(f)
    mk = ['o', 's', '^', 'D', 'X', 'P']
    col = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#e6550d', '#17becf']

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    lin = 1e-8
    ax.set_xscale('symlog', linthresh=lin); ax.set_yscale('symlog', linthresh=lin)
    lim = 2.0
    ax.plot([-lim, lim], [-lim, lim], 'k-', lw=1, alpha=0.6, label='$y=x$')
    ax.plot([-lim, lim], [-2 * lim, 2 * lim], 'k:', lw=1, alpha=0.6, label='$y=2x$')
    ok_err = 0.0
    for g in sorted(groups):
        lab, a, f = groups[g]
        a = np.array(a); f = np.array(f)
        errs = np.array([T.relerr(x, y) for x, y in zip(a, f)])
        if g != 4:
            ok_err = max(ok_err, errs.max())
        ax.scatter(a, f, marker=mk[g], s=60 if g != 5 else 80, color=col[g], edgecolor='k', linewidth=0.6, zorder=4,
                   label=f"{lab}  (n={len(a)}, max sai số tđ {errs.max():.1e})")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('analytic (công thức chép từ backward.cu)')
    ax.set_ylabel(r'sai phân hữu hạn trung tâm ($h=10^{-6}$)')
    ax.set_title('Kiểm chứng FD toàn bộ chuỗi backward tại pixel (23,15) và toàn ảnh camera 1\n'
                 rf'trục symlog (tuyến tính trong $\pm10^{{-8}}$) — max sai số tương đối (trừ B code) $= {ok_err:.1e}$',
                 fontsize=10.5)
    ax.legend(fontsize=8, loc='upper left')
    # annotate B code points
    lab, a, f = groups[4]
    ax.annotate(r"$\partial\mathcal{L}_x/\partial B$ theo quy ước code nằm trên $y=2x$:" "\n"
                r"backward.cu:212-214 nhân lại $2\cdot$dL_dconic.y",
                xy=(a[2], f[2]), xytext=(0.55, 0.12), textcoords='axes fraction', fontsize=8.5,
                arrowprops=dict(arrowstyle='->', color='#e6550d'), color='#e6550d',
                bbox=dict(fc='white', ec='#e6550d'))
    ax.text(0.98, 0.02, f"tổng {len(pts)} cặp analytic/FD", transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5)
    save(fig, "ch06_fd_check.png")


# ==========================================================================
# 5. ch06_adam.png — Adam 3 bước, decay eta_xyz(t), g_eff
# ==========================================================================
def fig_adam():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1, 0.9]))
    # (a)
    ax = axes[0]
    names = list(T.LR_GROUPS.keys()); lrs = np.array(list(T.LR_GROUPS.values()))
    short = ['xyz (t=0)', 'f_dc', 'opacity', 'scaling', 'rotation', 'f_rest']
    x = np.arange(len(names)); w = 0.26
    kcol = ['#c6dbef', '#6baed6', '#08519c']
    for k in range(3):
        d = lrs * T.ratios[k]
        bars = ax.bar(x + (k - 1) * w, np.abs(d), w, color=kcol[k], edgecolor='k', lw=0.5, label=f'bước k={k+1}')
    for i in range(len(names)):
        ax.text(x[i], lrs[i] * 1.25, f"η={lrs[i]:.3g}\n|Δθ|/η={abs(T.ratios[2]):.6f}", ha='center', va='bottom', fontsize=7)
    ax.set_yscale('log'); ax.set_ylim(1e-4, 0.3)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.5, rotation=15)
    ax.set_ylabel(r'$|\Delta\theta| = \eta\cdot\hat m/(\sqrt{\hat v}+\epsilon)$')
    ax.set_title(f'(a) Adam 3 bước, $g={T.g_adam:.3e}$ lặp lại\n'
                 rf'$\hat m/(\sqrt{{\hat v}}+\epsilon) = {T.ratios[0]:.4f},\ {T.ratios[1]:.4f},\ {T.ratios[2]:.4f}$ → $\Delta\theta=\eta$', fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    mvals = []
    m = v_ = 0.0
    for k in range(1, 4):
        m = T.B1 * m + (1 - T.B1) * T.g_adam; v_ = T.B2 * v_ + (1 - T.B2) * T.g_adam ** 2
        mvals.append((m, v_))
    ax.text(0.98, 0.97, "m: " + ", ".join(f"{mm:.3e}" for mm, _ in mvals) + "\nv: " + ", ".join(f"{vv:.3e}" for _, vv in mvals)
            + f"\ng/1000 → tỉ số 0.9999999935", transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            bbox=dict(fc='white', ec='#999'))
    # (b)
    ax = axes[1]
    t = np.linspace(0, 30000, 601)
    ax.plot(t, [T.expon_lr(tt) for tt in t], color='#d62728', lw=2)
    ax.set_yscale('log')
    for tt in (0, 7500, 15000, 30000):
        e = T.expon_lr(tt)
        ax.plot(tt, e, 'o', color='k', ms=6, zorder=5)
        ax.annotate(f"t={tt}\nη={e:.3e}\n(×{e/T.LR_INIT:.4g})", (tt, e), xytext=(8, 8 if tt < 20000 else 22),
                    textcoords='offset points', fontsize=8, ha='left' if tt < 20000 else 'right',
                    bbox=dict(fc='white', ec='#999', alpha=0.9))
    ax.axvspan(15000, 20000, color='#ffe08a', alpha=0.3); ax.axvspan(20000, 30000, color='#ffb3b3', alpha=0.25)
    ax.set_xlabel('iteration t'); ax.set_ylabel(r'$\eta_{xyz}(t)$')
    ax.set_xlim(0, 30000); ax.set_ylim(1.5e-6, 6e-4)
    ax.set_title(r'(b) $\eta_{xyz}(t)=\exp[(1-\frac{t}{T})\ln\eta_0+\frac{t}{T}\ln\eta_T]$' '\n'
                 rf'$\eta_0=1.6\times10^{{-4}}\cdot\mathrm{{extent}}={T.LR_INIT:.3e}$, $\eta_T=\eta_0/100$', fontsize=10)
    # (c)
    ax = axes[2]
    eta = T.eta
    vals = [abs(T.d_one) / eta, abs(T.d_64) / eta]
    bars = ax.bar([0, 1], vals, 0.55, color=['#7f7f7f', '#2ca02c'], edgecolor='k')
    for b, v, d in zip(bars, vals, (T.d_one, T.d_64)):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"Δθ = {d:.3e}\n= {v:.1f} η", ha='center', va='bottom', fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['1 step với\n$g_{\\mathrm{eff}}=\\sum_{64} g$\n(giai đoạn >20k)',
                                              '64 step nhỏ\n(giai đoạn ≤15k)'], fontsize=8.5)
    ax.set_ylabel(r'$|\Delta\theta|/\eta$'); ax.set_ylim(0, 78)
    ax.set_title(f'(c) $g_{{\\mathrm{{eff}}}}$: $\\sum g={T.g_seq.sum():.3e}$, $\\eta={eta:.3e}$\n'
                 f'tỉ số 64 step / 1 step = {T.d_64/T.d_one:.1f}×', fontsize=10)
    fig.tight_layout()
    save(fig, "ch06_adam.png")


# ==========================================================================
# 6. ch06_schedule.png — lịch step thưa dần
# ==========================================================================
def fig_schedule():
    ITERS = 30000
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axes[0]
    tt = np.array([1, 15000, 15000.0001, 20000, 20000.0001, 29999])
    f_main = np.array([1, 1, 1 / 32, 1 / 32, 1 / 64, 1 / 64])
    f_sh = np.array([1 / 16, 1 / 16, 1 / 32, 1 / 32, 1 / 64, 1 / 64])
    ax.step(tt, f_main, where='post', color='#1f77b4', lw=2.4, label='optimizer (xyz, opacity, scaling, rotation, f_dc)')
    ax.step(tt, f_sh, where='post', color='#e6550d', lw=2.4, ls='--', label='shoptimizer (f_rest)')
    ax.set_yscale('log'); ax.set_ylim(1 / 64 / 2.5, 3)
    ax.set_xlim(0, 30000)
    ax.axvspan(0, 15000, color='#c7e9c0', alpha=0.35); ax.axvspan(15000, 20000, color='#ffe08a', alpha=0.35)
    ax.axvspan(20000, 30000, color='#ffb3b3', alpha=0.3)
    ax.set_yticks([1, 1 / 16, 1 / 32, 1 / 64]); ax.set_yticklabels(['1', '1/16', '1/32', '1/64'])
    ax.set_ylabel('tần suất step (step / iteration)'); ax.set_xlabel('iteration t (step khi t < 30000, train.py:161)')
    seg_txt = [(7500, f"t ≤ 15000\noptimizer: {T.n_main_seg['<=15000']}\nshoptimizer: {T.n_sh_seg['<=15000']}\n(t mod 16 = 0)"),
               (17500, f"(15000, 20000]\ncả hai: t mod 32 = 0\n{T.n_main_seg['(15000,20000]']} step\n= 625 − 468"),
               (25000, f"(20000, 29999]\ncả hai: t mod 64 = 0\n{T.n_main_seg['>20000']} step")]
    for xx, s in seg_txt:
        ax.text(xx, 0.55, s, ha='center', va='top', fontsize=8.5, bbox=dict(fc='white', ec='#999', alpha=0.9))
    ax.text(0.99, 0.04, f"tích luỹ: optimizer {T.n_main_seg['<=15000']}+{T.n_main_seg['(15000,20000]']}+{T.n_main_seg['>20000']} = {T.n_main}\n"
            f"shoptimizer {T.n_sh_seg['<=15000']}+{T.n_sh_seg['(15000,20000]']}+{T.n_sh_seg['>20000']} = {T.n_sh}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, bbox=dict(fc='#fff8e1', ec='#999'))
    ax.set_title('Lịch step thưa dần của optimizer_step (gaussian_model.py:190-209), đếm bằng vòng lặp t = 1..29999', fontsize=10.5)
    ax.legend(fontsize=8.5, loc='upper right')
    # (b) bar so sanh
    ax = axes[1]
    tot3d = 29999 + 29999; totf = T.n_main + T.n_sh
    ax.bar([0], [29999], 0.5, color='#1f77b4', edgecolor='k', label='optimizer')
    ax.bar([0], [29999], 0.5, bottom=[29999], color='#e6550d', edgecolor='k', label='shoptimizer')
    ax.bar([1], [T.n_main], 0.5, color='#1f77b4', edgecolor='k')
    ax.bar([1], [T.n_sh], 0.5, bottom=[T.n_main], color='#e6550d', edgecolor='k')
    ax.text(0, 29999 / 2, "29999", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    ax.text(0, 29999 * 1.5, "29999", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    ax.text(1, T.n_main / 2, f"{T.n_main}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    ax.text(1, T.n_main + T.n_sh + 800, f"shoptimizer {T.n_sh}", ha='center', va='bottom', color='#e6550d', fontsize=9)
    ax.text(0, tot3d + 800, f"tổng {tot3d}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(1, T.n_main + T.n_sh + 3600, f"tổng {totf}\n= {totf/tot3d:.3f} × 59998", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['3DGS gốc\n(step mỗi iteration)', 'FastGS-lite\n(thưa dần)'])
    ax.set_ylim(0, 72000); ax.set_ylabel('số lần gọi optimizer.step()')
    ax.set_title(f'Tổng số step: {tot3d} → {totf} (tỉ số {totf/tot3d:.4f})', fontsize=10.5)
    ax.legend(fontsize=8.5, loc='upper right')
    fig.tight_layout()
    save(fig, "ch06_schedule.png")


if __name__ == "__main__":
    fig_dC_dalpha()
    fig_grad_field()
    fig_4cols()
    fig_fd_check()
    fig_adam()
    fig_schedule()
