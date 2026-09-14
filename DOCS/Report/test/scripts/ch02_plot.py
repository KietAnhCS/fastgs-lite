"""
Hình minh hoạ cho Test số — Chương 2: 3D Gaussians (DOCS/Report/test/02-test.md)

Chỉ dùng numpy + matplotlib (không torch). Các hàm tính số được CHÉP từ
scripts/ch02_test.py (file đó in ra stdout khi import nên không import trực tiếp).
Số ghi trên hình là số tính lại tại đây và phải trùng với 02-test.md.

Chạy:  python DOCS/Report/test/scripts/ch02_plot.py
PNG:   DOCS/Report/test/figures/ch02_*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

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
GCOL = ['#d62728', '#2ca02c', '#1f77b4', '#7f7f7f']            # G1..G4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Chép từ ch02_test.py (utils/sh_utils.py, utils/general_utils.py, gaussian_model.py)
# ---------------------------------------------------------------------------
C0 = 0.28209479177387814
C1 = 0.4886025119029199


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_rotation(r):
    norm = np.sqrt((r ** 2).sum(-1))
    q = r / norm
    rr, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - rr * z), 2 * (x * z + rr * y)],
                     [2 * (x * y + rr * z), 1 - 2 * (x * x + z * z), 2 * (y * z - rr * x)],
                     [2 * (x * z - rr * y), 2 * (y * z + rr * x), 1 - 2 * (x * x + y * y)]])


def cov_from_sr(s, q):          # Sigma = R S S^T R^T
    L = build_rotation(q) @ np.diag(s)
    return L @ L.T


# Cảnh đồ chơi (00-scene.md) — số khởi tạo chương 1 tính lại như ch02_test.py
P = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.5], [-0.4, -0.2, 1.0], [0.3, -0.5, 0.2]])
COL = np.array([[0.8, 0.2, 0.2], [0.2, 0.7, 0.3], [0.1, 0.3, 0.9], [0.5, 0.5, 0.5]])
D2 = ((P[:, None] - P[None]) ** 2).sum(-1)
dist2 = np.array([np.sort(np.delete(D2[i], i))[:3].mean() for i in range(4)])
s_tilde = np.log(np.sqrt(dist2))                 # (-0.161943, -0.058267, 0.108898, -0.117861)
s = np.exp(s_tilde)                              # (0.850490, 0.943398, 1.115049, 0.888819)
a_tilde = np.log(0.1 / 0.9)                      # -2.197225
k00 = (COL - 0.5) / C0


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=450, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', os.path.normpath(path))


# ===========================================================================
# Hình 1: activation
# ===========================================================================
def fig_activation():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle('2.1 — Activation: tham số thô → tham số dùng trong render', fontsize=13)

    # (a) quaternion normalize
    qt = np.array([0.9, 0.1, 0.3, 0.2])
    nq = np.linalg.norm(qt)
    qn = qt / nq
    a = ax[0]
    xs = np.arange(4); w = 0.38
    b1 = a.bar(xs - w / 2, qt, w, color='#bbbbbb', label=r'$\tilde q$ (thô)')
    b2 = a.bar(xs + w / 2, qn, w, color=GCOL[0], label=r'$q=\tilde q/\|\tilde q\|$')
    for r, v in zip(b1, qt):
        a.text(r.get_x() + r.get_width() / 2, v + 0.015, f'{v:.2f}', ha='center', fontsize=8, color='#555555')
    for r, v in zip(b2, qn):
        a.text(r.get_x() + r.get_width() / 2, v + 0.015, f'{v:.4f}', ha='center', fontsize=8, color=GCOL[0])
    a.set_xticks(xs); a.set_xticklabels(['r', 'x', 'y', 'z'])
    a.set_ylim(0, 1.35)
    a.set_title(r'(a) $q=\tilde q/\|\tilde q\|$', fontsize=11)
    a.text(0.97, 0.95, r'$\|\tilde q\|=\sqrt{0.95}=%.4f$' % nq + '\n' + r'$\|q\|=%.4f$' % np.linalg.norm(qn),
           transform=a.transAxes, ha='right', va='top', fontsize=9,
           bbox=dict(boxstyle='round', fc='white', ec='#999999'))
    a.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.42, 0.98))
    a.set_ylabel('giá trị thành phần')

    # (b) s = exp(s~)
    a = ax[1]
    t = np.linspace(-0.4, 0.4, 200)
    a.plot(t, np.exp(t), color='black', lw=1.5, label=r'$s=\exp(\tilde s)$')
    for i in range(4):
        a.plot(s_tilde[i], s[i], 'o', color=GCOL[i], ms=8, zorder=5)
        if i == 3:
            xy_t, ha = (s_tilde[i] + 0.05, s[i] - 0.16), 'left'
        elif i == 2:
            xy_t, ha = (s_tilde[i] - 0.02, s[i] + 0.10), 'center'
        else:
            xy_t, ha = (s_tilde[i] - 0.02, s[i] + 0.07), 'right'
        a.annotate(f'G{i+1}: exp({s_tilde[i]:.4f}) = {s[i]:.4f}', (s_tilde[i], s[i]),
                   xytext=xy_t, fontsize=8, color=GCOL[i], ha=ha,
                   arrowprops=dict(arrowstyle='-', color=GCOL[i], lw=0.7))
    a.set_xlabel(r'$\tilde s$ (log-scale, thô)'); a.set_ylabel(r'$s$')
    a.set_title(r'(b) $s=\exp(\tilde s)$ — luôn $>0$', fontsize=11)
    a.set_ylim(0.6, 1.55)
    a.legend(loc='upper left', fontsize=9)

    # (c) alpha = sigmoid
    a = ax[2]
    t = np.linspace(-6, 6, 300)
    a.plot(t, sigmoid(t), color='black', lw=1.5, label=r'$\alpha=\sigma(\tilde\alpha)=1/(1+e^{-\tilde\alpha})$')
    al = sigmoid(a_tilde)
    a.plot(a_tilde, al, 'o', color=GCOL[0], ms=8, zorder=5)
    a.axhline(al, color=GCOL[0], ls=':', lw=1); a.axvline(a_tilde, color=GCOL[0], ls=':', lw=1)
    a.annotate(r'$\tilde\alpha=\log(0.1/0.9)=%.6f$' % a_tilde + '\n' + r'$\alpha=1/(1+9)=%.4f$ (cả 4 Gaussian)' % al,
               (a_tilde, al), xytext=(-5.8, 0.62), fontsize=9, color=GCOL[0],
               arrowprops=dict(arrowstyle='->', color=GCOL[0]))
    a.set_xlabel(r'$\tilde\alpha$ (thô)'); a.set_ylabel(r'$\alpha$')
    a.set_title(r'(c) $\alpha=\mathrm{sigmoid}(\tilde\alpha)\in(0,1)$', fontsize=11)
    a.set_ylim(-0.05, 1.05)
    a.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    save(fig, 'ch02_activation.png')


# ===========================================================================
# Hình 2: covariance
# ===========================================================================
def ellipse_from_cov2(C, **kw):
    """Ellipse 1σ của ma trận 2x2 C."""
    ev, evec = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))     # trục lớn = vector riêng lớn nhất
    return Ellipse((0, 0), 2 * np.sqrt(ev[1]), 2 * np.sqrt(ev[0]), angle=ang, **kw), ev, evec


def fig_covariance():
    qt = np.array([0.9, 0.1, 0.3, 0.2])
    R = build_rotation(qt)
    fig, ax = plt.subplots(1, 4, figsize=(18, 5.6))
    fig.suptitle(r'2.2 — $\Sigma = R S S^\mathrm{T} R^\mathrm{T}$: (a) đẳng hướng G1;  (b)–(d) dị hướng $s=(0.6,0.3,0.1)$ '
                 r'với $\tilde q=(0.9,0.1,0.3,0.2)$ → ellipse 1σ bị $R(q)$ xoay', fontsize=12)

    # (a) G1 isotropic
    a = ax[0]
    r1 = s[0]
    Sig1 = cov_from_sr(np.full(3, r1), qt)
    a.add_patch(Ellipse((0, 0), 2 * r1, 2 * r1, fc=GCOL[0], alpha=0.25, ec=GCOL[0], lw=2))
    a.annotate('', (r1, 0), (0, 0), arrowprops=dict(arrowstyle='->', color='black'))
    a.text(r1 / 2, 0.06, r'$r=s_1=\sqrt{0.7233}=%.4f$' % r1, fontsize=8.5, ha='center')
    a.plot(0, 0, 'k+', ms=8)
    a.text(0.02, 0.98, (r'$q=(1,0,0,0)\Rightarrow R=I$' + '\n' + r'$\Sigma=s_1^2 I=0.7233\,I$' + '\n' + r'$\det\Sigma=0.3785$'),
           transform=a.transAxes, va='top', fontsize=8, bbox=dict(boxstyle='round', fc='white', ec='#999999'))
    a.text(0.98, 0.02, 'xoay theo bất kỳ q\nvẫn là hình tròn', transform=a.transAxes, va='bottom', ha='right',
           fontsize=8, color='#555555')
    a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5); a.set_aspect('equal')
    a.set_xlabel('x'); a.set_ylabel('y')
    a.set_title(r'(a) G1: lát $xy$ của $\Sigma=s_1^2 I$ (ellipse 1σ)', fontsize=10)

    # (b)-(d) anisotropic rotated on 3 planes
    sa = np.array([0.6, 0.3, 0.1])
    Sig = cov_from_sr(sa, qt)
    Sig0 = np.diag(sa ** 2)
    planes = [((0, 1), 'x', 'y'), ((0, 2), 'x', 'z'), ((1, 2), 'y', 'z')]
    labs = ['(b)', '(c)', '(d)']
    for k, ((i, j), ni, nj) in enumerate(planes):
        a = ax[k + 1]
        C = Sig[np.ix_([i, j], [i, j])]
        C0_ = Sig0[np.ix_([i, j], [i, j])]
        e0 = Ellipse((0, 0), 2 * sa[i], 2 * sa[j], fc='none', ec='#999999', lw=1.2, ls='--', label='chưa xoay ($R=I$)')
        a.add_patch(e0)
        e, ev, evec = ellipse_from_cov2(C)
        e.set(fc=GCOL[0], alpha=0.25, ec=GCOL[0], lw=2, label=r'xoay bởi $R(q)$')
        a.add_patch(e)
        # principal axes of the 2D slice
        for m in range(2):
            v = evec[:, m] * np.sqrt(ev[m])
            a.annotate('', (v[0], v[1]), (0, 0), arrowprops=dict(arrowstyle='->', color=GCOL[0], lw=1))
        a.plot(0, 0, 'k+', ms=8)
        a.text(0.03, 0.97, (r'$\Sigma_{%s%s}=%.4f$' % (ni, ni, C[0, 0]) + '\n' +
                            r'$\Sigma_{%s%s}=%.4f$' % (ni, nj, C[0, 1]) + '\n' +
                            r'$\Sigma_{%s%s}=%.4f$' % (nj, nj, C[1, 1])),
               transform=a.transAxes, va='top', fontsize=8.5, bbox=dict(boxstyle='round', fc='white', ec='#999999'))
        a.set_xlim(-0.75, 0.75); a.set_ylim(-0.75, 0.75); a.set_aspect('equal')
        a.set_xlabel(ni); a.set_ylabel(nj)
        a.set_title(f'{labs[k]} lát ${ni}{nj}$ của $\\Sigma$ dị hướng, xoay bởi $R(q)$', fontsize=10)
        if k == 0:
            a.legend(loc='lower right', fontsize=8)
    ev3 = np.linalg.eigvalsh(Sig)
    fig.text(0.5, 0.01, (r'Trị riêng của $\Sigma$ (3×3) = (%.2f, %.2f, %.2f) $= (s_3^2, s_2^2, s_1^2)$; '
                           r'$\det\Sigma=%.2e=(0.6\cdot0.3\cdot0.1)^2$; vector riêng = cột của $R$'
                           % (ev3[0], ev3[1], ev3[2], np.linalg.det(Sig))),
             ha='center', va='bottom', fontsize=9.5,
             bbox=dict(boxstyle='round', fc='#fff8e1', ec='#e0a800'))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, 'ch02_covariance.png')


# ===========================================================================
# Hình 3: mật độ
# ===========================================================================
def fig_density():
    r1 = s[0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.suptitle(r'2.3 — Mật độ $G(x)=\exp\left(-\frac{1}{2} (x-\mu)^\mathrm{T}\Sigma^{-1}(x-\mu)\right)$ của G1, '
                 r'$\Sigma=s_1^2 I$, $s_1=%.4f$' % r1, fontsize=12)
    a = ax[0]
    d = np.linspace(-3 * r1, 3 * r1, 400)
    a.plot(d, np.exp(-0.5 * d ** 2 / r1 ** 2), color=GCOL[0], lw=2, label=r'$G(\mu+d\,e_x)=\exp(-\frac{1}{2} d^2/s_1^2)$')
    pts = [(0, 1.0, r'$d=0$: $G=1$'),
           (r1, np.exp(-0.5), r'$d=s_1=%.4f$: $G=e^{-1/2}=%.4f$' % (r1, np.exp(-0.5))),
           (2 * r1, np.exp(-2), r'$d=2s_1=%.4f$: $G=e^{-2}=%.4f$' % (2 * r1, np.exp(-2)))]
    for k, (x0, y0, lab) in enumerate(pts):
        a.plot(x0, y0, 'o', color='black', ms=7, zorder=5)
        a.axvline(x0, color='#999999', ls=':', lw=0.8)
        a.annotate(lab, (x0, y0), xytext=(x0 + 0.25, y0 + 0.12 - 0.03 * k), fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    a.set_xlabel(r'$d$ = khoảng cách tới $\mu$ dọc trục $x$'); a.set_ylabel(r'$G(x)$')
    a.set_xlim(-3 * r1, 3 * r1); a.set_ylim(0, 1.25)
    a.set_title('(a) Lát cắt 1D — cùng luật cho cả 4 Gaussian (chỉ khác $s_i$)', fontsize=10)
    a.legend(loc='upper left', fontsize=8.5)

    a = ax[1]
    L = 2.6
    xs = np.linspace(-L, L, 261); X, Y = np.meshgrid(xs, xs)
    G = np.exp(-0.5 * (X ** 2 + Y ** 2) / r1 ** 2)
    im = a.imshow(G, extent=[-L, L, -L, L], origin='lower', cmap='magma', vmin=0, vmax=1)
    a.contour(X, Y, G, levels=[np.exp(-2), np.exp(-0.5)], colors='white', linewidths=0.8, linestyles='--')
    plt.colorbar(im, ax=a, fraction=0.046, pad=0.03, label=r'$G(x)$')
    for (x0, y0, lab) in pts:
        a.plot(x0, 0, 'o', color='cyan', mec='black', ms=7, zorder=5)
    a.annotate('G=1.0000', (0, 0), xytext=(-2.4, 1.9), color='white', fontsize=9,
               arrowprops=dict(arrowstyle='->', color='white'))
    a.annotate(r'G=0.6065 (d=s$_1$)', (r1, 0), xytext=(0.4, -1.6), color='white', fontsize=9,
               arrowprops=dict(arrowstyle='->', color='white'))
    a.annotate(r'G=0.1353 (d=2s$_1$)', (2 * r1, 0), xytext=(0.3, 2.0), color='white', fontsize=9,
               arrowprops=dict(arrowstyle='->', color='white'))
    a.text(0.02, 0.02, 'vòng nét đứt: 1σ và 2σ', transform=a.transAxes, color='white', fontsize=8)
    a.grid(False)
    a.set_xlabel('x'); a.set_ylabel('y')
    a.set_title(r'(b) Lát $xy$ ($z=0$) — đẳng hướng nên đường mức là hình tròn', fontsize=10)
    fig.tight_layout()
    save(fig, 'ch02_density.png')


# ===========================================================================
# Hình 4: SH
# ===========================================================================
def fig_sh():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1.3, 1]))
    fig.suptitle('2.4 — Màu SH: bậc 0 (không phụ thuộc hướng), bậc 1 (phụ thuộc hướng) và clamp', fontsize=13)

    # (a) degree-0 swatches
    a = ax[0]
    c0 = np.maximum(0, 0.5 + C0 * k00)
    for i in range(4):
        a.add_patch(Rectangle((0, 3 - i), 1, 0.9, fc=c0[i], ec='black'))
        a.text(1.1, 3 - i + 0.45,
               f'G{i+1}: $k_{{00}}$=({k00[i,0]:.3f}, {k00[i,1]:.3f}, {k00[i,2]:.3f})\n'
               f'$c$ = ({c0[i,0]:.2f}, {c0[i,1]:.2f}, {c0[i,2]:.2f}) = màu SfM ✓',
               va='center', fontsize=8.5, color=GCOL[i])
    a.set_xlim(-0.1, 4.3); a.set_ylim(-0.2, 4.1)
    a.axis('off')
    a.set_title(r'(a) Bậc 0: $c=\max(0,\ 0.5+C_0 k_{00})$, $C_0=%.4f$' % C0, fontsize=10)

    # (b) G1 degree-1 color vs z of viewing direction
    a = ax[1]
    zs = np.linspace(-1, 1, 201)
    base = 0.5 + C0 * k00[0]                           # (0.8, 0.2, 0.2)
    k10 = np.array([0.0, 0.2, 0.0])
    cols = np.clip(base[None, :] + C1 * zs[:, None] * k10[None, :], 0, 1)
    a.imshow(cols[None, :, :], extent=[-1, 1, -0.14, -0.02], aspect='auto', interpolation='nearest')
    for ch, cc, nm in zip(range(3), ['#d62728', '#2ca02c', '#1f77b4'], ['R', 'G', 'B']):
        a.plot(zs, cols[:, ch], color=cc, lw=2, label=f'kênh {nm}')
    zc = 1.0; g1 = base[1] + C1 * zc * 0.2
    a.plot(zc, g1, 'o', color='black', ms=7, zorder=5)
    a.annotate(r'cam 1: $\vec d=(0,0,1)$, $z=1$' + '\n' + r'$G=0.2+C_1\cdot1\cdot0.2=%.4f$' % g1,
               (zc, g1), xytext=(-0.05, 0.55), fontsize=8.5,
               arrowprops=dict(arrowstyle='->', color='black'))
    a.plot(0, base[1], 's', color='black', ms=5)
    a.annotate(r'$z=0$: $G=0.2$', (0, base[1]), xytext=(0.15, 0.30), fontsize=8.5,
               arrowprops=dict(arrowstyle='->', color='black'))
    a.plot(-1, base[1] - C1 * 0.2, 'o', color='black', ms=5)
    a.text(-0.98, base[1] - C1 * 0.2 + 0.03, f'$z=-1$: G={base[1]-C1*0.2:.4f}', fontsize=8.5)
    a.text(0.5, -0.08, r'màu $c_{G1}(\vec d)$', ha='center', va='center', fontsize=8, color='white')
    a.set_xlabel(r'$z$ của hướng nhìn $\vec d$ (chỉ $k_{10}$ đi với $z$)')
    a.set_ylabel('giá trị kênh')
    a.set_ylim(-0.15, 0.95); a.set_xlim(-1.02, 1.02)
    a.set_title(r'(b) G1, bậc 1: $c=0.5+C_0k_{00}+C_1 z\,k_{10}$, $k_{10}=(0,0.2,0)$', fontsize=10)
    a.legend(loc='upper left', fontsize=8, ncol=3)

    # (c) clamp case
    a = ax[2]
    k10c = np.array([0.0, 0.2, -1.5])
    raw = base + C1 * 1.0 * k10c
    clamped = np.maximum(raw, 0)
    xs = np.arange(3); w = 0.38
    b1 = a.bar(xs - w / 2, raw, w, color='#bbbbbb', label='raw = 0.5 + C0·k00 + C1·z·k10')
    b2 = a.bar(xs + w / 2, clamped, w, color=['#d62728', '#2ca02c', '#1f77b4'], label='c = max(raw, 0)')
    for r, v in zip(b1, raw):
        a.text(r.get_x() + w / 2, v + (0.02 if v >= 0 else -0.02), f'{v:.4f}', ha='center',
               va='bottom' if v >= 0 else 'top', fontsize=8.5)
    for r, v in zip(b2, clamped):
        a.text(r.get_x() + w / 2, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=8.5)
    a.axhline(0, color='black', lw=1)
    a.set_xticks(xs); a.set_xticklabels(['R', 'G', 'B'])
    a.set_ylim(-0.85, 1.05)
    a.annotate('clamped = (F, F, T)\nkênh B bị cắt → backward\nchặn gradient kênh B', (2 + w / 2, 0.0),
               xytext=(0.55, 0.72), fontsize=8.5, color='#1f77b4',
               arrowprops=dict(arrowstyle='->', color='#1f77b4'))
    a.text(2 - w / 2, -0.72, r'$0.2+C_1\cdot(-1.5)$', ha='center', fontsize=8, color='#555555')
    a.set_title(r'(c) Clamp: $k_{10}=(0,0.2,-1.5)$, $\vec d=(0,0,1)$', fontsize=10)
    a.legend(loc='lower left', fontsize=7.5)
    fig.tight_layout()
    save(fig, 'ch02_sh.png')


# ===========================================================================
# Hình 5: lịch tăng bậc SH
# ===========================================================================
def fig_sh_schedule():
    N = 4
    t = np.arange(0, 4001)
    D = np.minimum(3, t // 1000)
    fig, a = plt.subplots(figsize=(10, 4.6))
    a.step(t, D, where='post', color='black', lw=2, label=r'$D(t)=\min(3,\lfloor t/1000\rfloor)$')
    a.set_xlabel('vòng lặp $t$ (train.py: `if iteration % 1000 == 0: oneupSHdegree()`)')
    a.set_ylabel('bậc SH hoạt động $D$')
    a.set_yticks([0, 1, 2, 3]); a.set_ylim(-0.3, 4.8); a.set_xlim(0, 4000)
    a2 = a.twinx()
    perg = 3 * (D + 1) ** 2
    a2.step(t, perg, where='post', color='#ff7f0e', lw=2, ls='--', label='float SH / Gaussian $=3(D+1)^2$')
    a2.step(t, perg * N, where='post', color='#9467bd', lw=1.5, ls='-.', label=f'× N={N} Gaussian')
    a2.set_ylabel('số float SH đọc trong forward', color='#ff7f0e')
    a2.set_ylim(-15, 265)
    a2.grid(False)
    for k, tt in enumerate([0, 1000, 2000, 3000]):
        d = min(3, tt // 1000); pg = 3 * (d + 1) ** 2
        a.annotate(f'$t$={tt}: D={d}\n{pg} float ({(d+1)**2}/kênh ×3)\n×4 = {pg*N}',
                   (tt + 500, d), xytext=(tt + 420, d + 0.45), fontsize=8, ha='center', va='bottom',
                   bbox=dict(boxstyle='round', fc='white', ec='#999999'))
        a.plot(tt, d, 'o', color='black', ms=5)
        a.plot(tt, d, 'o', color='black', ms=5)
        a2.plot(tt, pg, 'o', color='#ff7f0e', ms=5)
    a.text(0.99, 0.03, 'sau t=3000 giữ nguyên D=3 (max_sh_degree)\n'
           'tổng tham số luôn 59/Gaussian; 3 dc + 45 rest luôn nằm trong optimizer',
           transform=a.transAxes, ha='right', va='bottom', fontsize=8, color='#555555')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8.5)
    a.set_title('2.5 — Lịch tăng bậc SH và số float SH hoạt động (3 → 12 → 27 → 48)', fontsize=12)
    fig.tight_layout()
    save(fig, 'ch02_sh_schedule.png')


if __name__ == '__main__':
    fig_activation()
    fig_covariance()
    fig_density()
    fig_sh()
    fig_sh_schedule()
