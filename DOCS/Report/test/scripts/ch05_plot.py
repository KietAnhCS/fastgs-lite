"""
Vẽ hình minh hoạ cho bài test số chương 5 (Loss & Metrics).
Import lại renderer / loss / metric từ ch05_test.py (không sửa file đó).
Chỉ dùng numpy + scipy + matplotlib (không torch).

Chạy:  python DOCS/Report/test/scripts/ch05_plot.py
Xuất:  DOCS/Report/test/figures/ch05_*.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch05_test import (points, colors, cam_centers, init_scales, render, gaussian,   # noqa: E402
                       create_window, l1_loss, ssim_full, mse, psnr, psnr_old_bug,
                       composite_score, project, BG_WHITE, W, H)

FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.facecolor': 'white'})
SAVE = dict(dpi=150, bbox_inches='tight')

C_REND, C_GT, C_L1, C_SSIM, C_LOSS, C_PSNR = '#1f77b4', '#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'


def out(name):
    return os.path.join(FIG_DIR, f"ch05_{name}.png")


# --------------------------------------------------------------------------
# Dữ liệu (giống hệt ch05_test.py)
# --------------------------------------------------------------------------
N = points.shape[0]
scales = init_scales(points)
cam1 = cam_centers[0]
ALPHA_MODEL, ALPHA_GT = 0.1, 0.9
I_rend, T_rend, _ = render(points, scales, colors, np.full(N, ALPHA_MODEL), cam1, BG_WHITE)
I_gt, T_gt, _ = render(points, scales, colors, np.full(N, ALPHA_GT), cam1, BG_WHITE)
mean2d, _, _ = project(points, scales, cam1)
cx, cy = int(round(mean2d[0, 0])), int(round(mean2d[0, 1]))     # tâm G1 = (24, 16)

diff = np.abs(I_rend - I_gt)
L1_sum, L1 = diff.sum(), l1_loss(I_rend, I_gt)
S, aux = ssim_full(I_rend, I_gt)
D_SSIM = 1 - S
MSE, PSNR = mse(I_rend, I_gt), psnr(I_rend, I_gt)
PSNR_old, m_c = psnr_old_bug(I_rend, I_gt)
psnr_c = 20 * np.log10(1 / np.sqrt(m_c))
LPIPS_ASSUMED = 0.30
score, pn = composite_score(PSNR, S, LPIPS_ASSUMED)


def to_img(I):
    return np.clip(I.transpose(1, 2, 0), 0, 1)


# ==========================================================================
# Hình 1: L1
# ==========================================================================
def fig_l1():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    (a1, a2), (a3, a4) = axes
    for ax in (a1, a2, a3):
        ax.grid(False)
    a1.imshow(to_img(I_rend), interpolation='nearest')
    a1.set_title(r'$I_\mathrm{rend}$ ($\alpha=0.1$, mô hình chưa học)' + f'\nmean = {I_rend.mean():.4f}', fontsize=10)
    a2.imshow(to_img(I_gt), interpolation='nearest')
    a2.set_title(r'$I_\mathrm{gt}$ ($\alpha=0.9$)' + f'\nmean = {I_gt.mean():.4f}', fontsize=10)
    for ax in (a1, a2):
        ax.plot(cx, cy, 'w+', ms=10, mew=1.5)
        ax.annotate('tâm G1 (24,16)', (cx, cy), xytext=(cx + 6, cy - 6), color='k', fontsize=8,
                    arrowprops=dict(arrowstyle='-', color='k', lw=0.8))
        ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')

    dmean = diff.mean(0)
    im = a3.imshow(dmean, cmap='magma', interpolation='nearest', vmin=0)
    cb = fig.colorbar(im, ax=a3, fraction=0.035, pad=0.02)
    cb.set_label(r'$\frac{1}{3}\sum_{ch}|I_\mathrm{rend}-I_\mathrm{gt}|$', fontsize=9)
    a3.set_title(r'$|I_\mathrm{rend}-I_\mathrm{gt}|$ trung bình 3 kênh', fontsize=10)
    ymax, xmax = np.unravel_index(dmean.argmax(), dmean.shape)
    a3.plot(xmax, ymax, 'c^', ms=7)
    a3.annotate(f'max TB kênh = {dmean.max():.4f}\ntại (x,y)=({xmax},{ymax})', (xmax, ymax),
                xytext=(xmax + 8, ymax + 10), color='k', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='k', lw=0.8))
    a3.text(0.02, 0.97, r'$\mathcal{L}_1=\frac{\sum|\mathrm{diff}|}{3HW}=\frac{%.2f}{%d}=%.4f$' % (L1_sum, 3 * H * W, L1),
            transform=a3.transAxes, va='top', ha='left', fontsize=10, color='k',
            bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.9))
    a3.set_xlabel('x (px)'); a3.set_ylabel('y (px)')

    d = diff.ravel()
    a4.hist(d, bins=40, color=C_L1, alpha=0.85, edgecolor='white')
    a4.axvline(L1, color=C_GT, ls='--', lw=1.5)
    a4.text(L1 + 0.01, a4.get_ylim()[1] * 0.9, f'mean = {L1:.4f}', color=C_GT, fontsize=9)
    a4.axvline(diff.max(), color='k', ls=':', lw=1.2)
    a4.text(diff.max() - 0.01, a4.get_ylim()[1] * 0.55, f'max = {diff.max():.4f}\n(kênh B, (y,x)=(16,22))',
            color='k', fontsize=8, ha='right')
    med = np.median(d)
    a4.text(0.98, 0.97, f'n = {d.size} giá trị (3HW)\nmedian = {med:.4f}\n≤0.05: {100*np.mean(d<=0.05):.1f}%\n>0.4: {100*np.mean(d>0.4):.1f}%',
            transform=a4.transAxes, va='top', ha='right', fontsize=8,
            bbox=dict(boxstyle='round', fc='white', ec='0.5'))
    a4.set_xlabel(r'sai số per-pixel $|I_\mathrm{rend}-I_\mathrm{gt}|$ (mọi kênh)')
    a4.set_ylabel('số pixel·kênh')
    a4.set_title(r'Histogram sai số → $\mathcal{L}_1$ = trung bình', fontsize=10)
    fig.suptitle(r'Hình 5.1 — $\mathcal{L}_1$ giữa ảnh render ($\alpha=0.1$) và GT ($\alpha=0.9$), camera 1, $48\times32$', fontsize=12)
    fig.tight_layout()
    fig.savefig(out('l1'), **SAVE)
    plt.close(fig)


# ==========================================================================
# Hình 2: SSIM
# ==========================================================================
def fig_ssim():
    g = gaussian(11, 1.5)
    win = create_window(11)
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], width_ratios=[1, 1, 1.15])
    a_win = fig.add_subplot(gs[0, 0])
    a_prof = fig.add_subplot(gs[0, 1])
    a_txt = fig.add_subplot(gs[0, 2])
    a_map = fig.add_subplot(gs[1, 0:2])
    gs_p = gs[1, 2].subgridspec(1, 2, wspace=0.35)
    a_p1, a_p2 = fig.add_subplot(gs_p[0]), fig.add_subplot(gs_p[1])

    a_win.grid(False)
    im = a_win.imshow(win, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=a_win, fraction=0.046, pad=0.03)
    a_win.set_title(r'(a) cửa sổ $11\times11 = g\,g^\top$, $\sigma=1.5$' + f'\ntâm = {win[5,5]:.5f}, góc = {win[0,0]:.2e}, tổng = {win.sum():.3f}', fontsize=9)
    a_win.set_xticks(range(0, 11, 2)); a_win.set_yticks(range(0, 11, 2))

    a_prof.bar(range(11), g, color='#4c72b0', alpha=0.85)
    a_prof.plot(range(11), g, 'o-', color='k', ms=3, lw=1)
    for i, v in enumerate(g):
        a_prof.text(i, v + 0.006, f'{v:.4f}' if v > 0.005 else f'{v:.1e}', ha='center', va='bottom', fontsize=6.5,
                    rotation=90 if v < 0.05 else 0)
    a_prof.set_ylim(0, 0.34)
    a_prof.set_xticks(range(11)); a_prof.set_xticklabels([str(i - 5) for i in range(11)])
    a_prof.set_xlabel('offset tap'); a_prof.set_ylabel('g[i]')
    a_prof.set_title(r'(a) profile 1D 11 tap, $\sum g = 1$', fontsize=9)

    a_txt.axis('off')
    c = 0
    m1, m2 = aux['mu1'][c, cy, cx], aux['mu2'][c, cy, cx]
    s1, s2, s12 = aux['s1'][c, cy, cx], aux['s2'][c, cy, cx], aux['s12'][c, cy, cx]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * m1 * m2 + C1) * (2 * s12 + C2)
    den = (m1 ** 2 + m2 ** 2 + C1) * (s1 + s2 + C2)
    txt = (r'(c) tại tâm G1 $(x,y)=(24,16)$, kênh R' + '\n\n'
           r'$\mu_1=%.4f,\ \mu_2=%.4f$' % (m1, m2) + '\n'
           r'$\sigma_1^2=%.3e,\ \sigma_2^2=%.3e$' % (s1, s2) + '\n'
           r'$\sigma_{12}=%.3e$ (< 0: cấu trúc ngược)' % s12 + '\n'
           r'$C_1=10^{-4},\ C_2=9\times10^{-4}$' + '\n\n'
           r'$\mathrm{SSIM}=\frac{(2\mu_1\mu_2+C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)}$' + '\n\n'
           r'$=\frac{(%.4f+10^{-4})(%.3e+9\cdot10^{-4})}{(%.4f+10^{-4})(%.3e+9\cdot10^{-4})}$' % (2 * m1 * m2, 2 * s12, m1 ** 2 + m2 ** 2, s1 + s2) + '\n\n'
           r'$=\frac{%.6g}{%.6g}=\mathbf{%.4f}$' % (num, den, num / den))
    a_txt.text(0.0, 0.5, txt, transform=a_txt.transAxes, fontsize=9.5, va='center', ha='left',
               bbox=dict(boxstyle='round,pad=0.6', fc='#fff8e6', ec='#c9a227'))

    a_map.grid(False)
    smap = aux['map'].mean(0)
    im2 = a_map.imshow(smap, cmap='RdYlGn', vmin=-0.1, vmax=1, interpolation='nearest')
    cb = fig.colorbar(im2, ax=a_map, fraction=0.025, pad=0.02)
    cb.set_label('SSIM per-pixel (TB 3 kênh)')
    a_map.plot(cx, cy, 'k+', ms=12, mew=2)
    a_map.annotate(f'tâm G1: map[R] = {aux["map"][0, cy, cx]:.4f}', (cx, cy), xytext=(cx + 5, cy - 8), fontsize=9,
                   arrowprops=dict(arrowstyle='->', lw=1))
    a_map.text(0.01, 0.97, f'(b) mean(ssim_map) = SSIM = {S:.4f} → D-SSIM = {D_SSIM:.4f}\n'
               f'map: min = {aux["map"].min():.4f}, max = {aux["map"].max():.4f}; không pixel nào = 1',
               transform=a_map.transAxes, va='top', fontsize=9.5,
               bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))
    a_map.set_xlabel('x (px)'); a_map.set_ylabel('y (px)')
    a_map.set_title(r'(b) bản đồ SSIM per-pixel giữa $I_\mathrm{rend}$ và $I_\mathrm{gt}$ (zero-pad, cửa sổ 11×11)', fontsize=10)
    # (c) patch 11x11 kênh R quanh tâm G1 → μ1, μ2 là trung bình có trọng số Gauss
    ys, xs = slice(cy - 5, cy + 6), slice(cx - 5, cx + 6)
    for ax, img, name, mu in ((a_p1, I_rend, r'$I_\mathrm{rend}$ kênh R', m1), (a_p2, I_gt, r'$I_\mathrm{gt}$ kênh R', m2)):
        ax.grid(False)
        ax.imshow(img[0, ys, xs], cmap='Reds_r', vmin=0.2, vmax=1.0, interpolation='nearest')
        ax.plot(5, 5, 'k+', ms=9, mew=1.5)
        ax.set_xticks([0, 5, 10]); ax.set_xticklabels(['19', '24', '29'], fontsize=8)
        ax.set_yticks([0, 5, 10]); ax.set_yticklabels(['11', '16', '21'], fontsize=8)
        ax.set_title(name + '\n11×11 quanh G1, $\\mu$ = %.4f' % mu, fontsize=8.5)
    fig.suptitle(r'Hình 5.2 — SSIM: cửa sổ Gauss, bản đồ SSIM và minh hoạ thay số tại một pixel', fontsize=12)
    fig.tight_layout()
    fig.savefig(out('ssim'), **SAVE)
    plt.close(fig)


# ==========================================================================
# Hình 3: Loss = (1-λ)L1 + λ D-SSIM
# ==========================================================================
def fig_loss():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    lams = [0.2, 0.25]
    labels = [r'$\lambda=0.2$' + '\n(arguments/__init__.py:82)', r'$\lambda=0.25$' + '\n(pipeline/config.py:44)']
    x = np.arange(2)
    p1 = np.array([(1 - l) * L1 for l in lams])
    p2 = np.array([l * D_SSIM for l in lams])
    a1.bar(x, p1, 0.5, color=C_L1, label=r'$(1-\lambda)\,\mathcal{L}_1$')
    a1.bar(x, p2, 0.5, bottom=p1, color=C_SSIM, label=r'$\lambda\,\mathcal{L}_\mathrm{D\text{-}SSIM}$'.replace(r'\text', r'\mathrm'))
    for i in range(2):
        a1.text(x[i], p1[i] / 2, f'{1-lams[i]:.2f}×{L1:.4f}\n= {p1[i]:.4f}', ha='center', va='center', color='white', fontsize=9)
        a1.text(x[i], p1[i] + p2[i] / 2, f'{lams[i]:.2f}×{D_SSIM:.4f}\n= {p2[i]:.4f}', ha='center', va='center', color='k', fontsize=9)
        a1.text(x[i], p1[i] + p2[i] + 0.006, r'$\mathcal{L}=%.4f$' % (p1[i] + p2[i]), ha='center', va='bottom', fontsize=11, weight='bold')
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=9)
    a1.set_ylim(0, 0.40)
    a1.set_ylabel('giá trị loss')
    a1.set_title(r'$\mathcal{L}=(1-\lambda)\mathcal{L}_1+\lambda(1-\mathrm{SSIM})$' + f'\n' + r'$\mathcal{L}_1=%.4f$, D-SSIM $=%.4f$' % (L1, D_SSIM), fontsize=10)
    a1.legend(loc='upper center', fontsize=9, ncol=2)

    # loss theo λ liên tục
    ll = np.linspace(0, 1, 101)
    a2.plot(ll, (1 - ll) * L1 + ll * D_SSIM, color=C_LOSS, lw=2, label=r'$\mathcal{L}(\lambda)$')
    a2.axhline(L1, color=C_L1, ls='--', lw=1, label=r'$\mathcal{L}_1=%.4f$' % L1)
    a2.axhline(D_SSIM, color=C_SSIM, ls='--', lw=1, label=r'D-SSIM $=%.4f$' % D_SSIM)
    for l, m in zip(lams, ['o', 's']):
        v = (1 - l) * L1 + l * D_SSIM
        a2.plot(l, v, m, color='k', ms=7)
        a2.annotate(f'λ={l}: {v:.4f}', (l, v), xytext=(l + 0.05, v - 0.012), fontsize=9,
                    arrowprops=dict(arrowstyle='->', lw=0.8))
    a2.set_xlabel(r'$\lambda$'); a2.set_ylabel(r'$\mathcal{L}$')
    a2.set_title(r'$\mathcal{L}$ tuyến tính theo $\lambda$: nội suy giữa $\mathcal{L}_1$ và D-SSIM' + '\n(D-SSIM > L1 nên tăng λ làm loss tăng nhẹ)', fontsize=10)
    a2.legend(fontsize=8, loc='lower right')
    fig.suptitle('Hình 5.3 — Hai thành phần của loss huấn luyện (camera 1, α mô hình = 0.1)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out('loss'), **SAVE)
    plt.close(fig)


# ==========================================================================
# Hình 4: độ nhạy theo α mô hình
# ==========================================================================
def sensitivity(alphas):
    rows = []
    for a_m in alphas:
        I_a, _, _ = render(points, scales, colors, np.full(N, a_m), cam1, BG_WHITE)
        l1a = l1_loss(I_a, I_gt)
        sa, _ = ssim_full(I_a, I_gt)
        la = 0.8 * l1a + 0.2 * (1 - sa)
        m = mse(I_a, I_gt)
        pa = psnr(I_a, I_gt) if m > 0 else np.inf
        rows.append((a_m, l1a, sa, 1 - sa, la, pa))
    return np.array(rows)


def fig_sensitivity():
    fine = sensitivity(np.round(np.arange(0.05, 0.951, 0.05), 2))
    coarse = sensitivity([0.1, 0.3, 0.5, 0.7, 0.9])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    al = fine[:, 0]
    a1.plot(al, fine[:, 1], '-', color=C_L1, lw=1.5, label=r'$\mathcal{L}_1$')
    a1.plot(al, fine[:, 3], '-', color=C_SSIM, lw=1.5, label='D-SSIM = 1 − SSIM')
    a1.plot(al, fine[:, 4], '-', color=C_LOSS, lw=2.5, label=r'$\mathcal{L}$ ($\lambda=0.2$)')
    for j, colr in ((1, C_L1), (3, C_SSIM), (4, C_LOSS)):
        a1.plot(coarse[:, 0], coarse[:, j], 'o', color=colr, ms=6)
    for r in coarse:
        a1.annotate(f'{r[4]:.4f}', (r[0], r[4]), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8.5, color=C_LOSS, weight='bold')
        if r[0] < 0.85:
            a1.annotate(f'{r[1]:.4f}', (r[0], r[1]), xytext=(0, -13), textcoords='offset points', ha='center', fontsize=8, color=C_L1)
            a1.annotate(f'{r[3]:.4f}', (r[0], r[3]), xytext=(14, 4), textcoords='offset points', ha='left', fontsize=8, color=C_SSIM)
    a1.axvline(0.9, color='k', ls=':', lw=1)
    a1.annotate('GT: α = 0.9\n→ loss = 0', (0.9, 0.0), xytext=(0.62, 0.12), fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=1), bbox=dict(boxstyle='round', fc='#eef', ec='0.6'))
    a1.plot(0.9, 0, '*', color='k', ms=12)
    a1.annotate('khởi tạo α = 0.1', (0.1, coarse[0, 4]), xytext=(0.3, 0.3), fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=1))
    a1.set_xlabel(r'$\alpha$ mô hình (GT cố định $\alpha=0.9$)')
    a1.set_ylabel('loss')
    a1.set_ylim(-0.02, 0.42)
    a1.set_title(r'$\mathcal{L}_1$, D-SSIM, $\mathcal{L}$ theo $\alpha$ mô hình' + '\n(đường mịn: bước 0.05; điểm: 5 giá trị trong bảng)', fontsize=10)
    a1.legend(fontsize=9, loc='upper right')

    ps = fine[:, 5].copy()
    ps[~np.isfinite(ps)] = np.nan            # ngắt đường tại α = 0.9 (PSNR = ∞)
    a2.plot(al, ps, '.-', color=C_PSNR, lw=2, ms=5)
    for a_, p_ in ((0.85, ps[al == 0.85][0]), (0.95, ps[al == 0.95][0])):
        a2.annotate(f'{p_:.2f}', (a_, p_), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8, color=C_PSNR)
    a2.plot(coarse[:-1, 0], coarse[:-1, 5], 'o', color=C_PSNR, ms=6)
    for r in coarse[:-1]:
        a2.annotate(f'{r[5]:.3f} dB', (r[0], r[5]), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8.5)
    a2.axhline(30, color='0.4', ls='--', lw=1)
    a2.text(0.06, 30.6, 'trần clamp PSNR/30 → 1 (Score)', fontsize=8.5, color='0.3')
    a2.axvline(0.9, color='k', ls=':', lw=1)
    a2.annotate('α = 0.9: MSE = 0\n→ PSNR = ∞ (ngắt đường)', (0.9, 46), xytext=(0.45, 43), fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=1), bbox=dict(boxstyle='round', fc='#eef', ec='0.6'))
    a2.set_ylim(0, 50)
    a2.set_xlabel(r'$\alpha$ mô hình'); a2.set_ylabel('PSNR (dB)')
    a2.set_title(r'PSNR theo $\alpha$ mô hình (tăng đơn điệu, phân kỳ tại GT)', fontsize=10)
    fig.suptitle('Hình 5.4 — Độ nhạy của loss và PSNR theo opacity mô hình (camera 1, nền trắng)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out('sensitivity'), **SAVE)
    plt.close(fig)
    return coarse


# ==========================================================================
# Hình 5: PSNR lỗi cũ (Jensen) + thứ tự trung bình / clamp
# ==========================================================================
def fig_psnr_bug():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[0.9, 1.1, 1.2]))
    ch = ['R', 'G', 'B']
    cc = ['#d62728', '#2ca02c', '#1f77b4']
    a1.bar(ch, m_c, color=cc, alpha=0.85)
    for i, v in enumerate(m_c):
        a1.text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    a1.axhline(MSE, color='k', ls='--', lw=1.2)
    a1.text(2.45, MSE + 0.002, f'mean = MSE\n= {MSE:.4f}', ha='right', va='bottom', fontsize=9)
    a1.set_ylim(0, 0.145)
    a1.set_ylabel(r'MSE$_c$'); a1.set_title('MSE theo kênh (trung bình = MSE toàn ảnh)', fontsize=10)

    x = np.arange(3)
    a2.bar(x, psnr_c, 0.55, color=cc, alpha=0.85)
    for i, v in enumerate(psnr_c):
        a2.text(i, v + 0.2, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    a2.axhline(PSNR_old, color='#8c564b', ls='--', lw=1.6)
    a2.axhline(PSNR, color='k', ls='-', lw=1.6)
    a2.text(2.45, PSNR_old + 0.12, f'lỗi cũ: mean(PSNR$_c$) = {PSNR_old:.4f} dB', ha='right', va='bottom', fontsize=9, color='#8c564b')
    a2.text(2.45, PSNR - 0.12, f'đúng: PSNR(MSE) = {PSNR:.4f} dB', ha='right', va='top', fontsize=9, color='k')
    a2.annotate('', (2.85, PSNR_old), (2.85, PSNR), arrowprops=dict(arrowstyle='<->', color='#8c564b', lw=1.2, shrinkA=0, shrinkB=0))
    a2.text(2.85, PSNR_old + 0.55, f'chênh\n+{PSNR_old-PSNR:.4f} dB', fontsize=8.5, color='#8c564b', va='bottom', ha='center')
    a2.set_xlim(-0.5, 3.3)
    a2.set_xticks(x); a2.set_xticklabels(ch)
    a2.set_ylim(8, 14)
    a2.set_ylabel('PSNR (dB)')
    a2.set_title('Jensen: $-10\\log_{10}$ lồi ⇒ mean$_c$(PSNR$_c$) ≥ PSNR(mean$_c$ MSE$_c$)', fontsize=9.5)

    ps = np.array([25.0, 32.0, 35.0])
    cl = np.clip(ps / 30, 0, 1)
    a_val = np.clip(ps.mean() / 30, 0, 1)
    b_val = cl.mean()
    xx = np.arange(3)
    a3.bar(xx - 0.18, ps / 30, 0.36, color='0.75', label='PSNR$_i$/30 (chưa clamp)')
    a3.bar(xx + 0.18, cl, 0.36, color=C_PSNR, alpha=0.85, label='clamp(PSNR$_i$/30)')
    for i in range(3):
        a3.text(xx[i] - 0.18, ps[i] / 30 + 0.01, f'{ps[i]/30:.4f}', ha='center', va='bottom', fontsize=8)
        a3.text(xx[i] + 0.18, cl[i] + 0.01, f'{cl[i]:.4f}', ha='center', va='bottom', fontsize=8)
    a3.axhline(1.0, color='k', ls=':', lw=1)
    a3.axhline(a_val, color='#8c564b', ls='-', lw=1.6)
    a3.axhline(b_val, color=C_PSNR, ls='--', lw=1.6)
    a3.text(2.5, a_val + 0.015, f'clamp(mean/30)\n= clamp({ps.mean():.4f}/30)\n= {a_val:.4f}  (code repo)', fontsize=8, color='#8c564b', va='bottom', ha='left')
    a3.text(2.5, b_val - 0.015, f'mean(clamp)\n= {b_val:.4f}', fontsize=8, color=C_PSNR, va='top', ha='left')
    a3.text(0.5, 0.06, f'chênh Score = 0.3×({a_val:.4f} − {b_val:.4f})\n= {0.3*(a_val-b_val):.4f}', transform=a3.transAxes,
            ha='center', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.5'))
    a3.set_xticks(xx); a3.set_xticklabels([f'view {i+1}\nPSNR={p:g}' for i, p in enumerate(ps)], fontsize=8.5)
    a3.set_ylim(0, 1.5); a3.set_xlim(-0.5, 3.7)
    a3.set_ylabel('PSNR chuẩn hoá')
    a3.set_title('Thứ tự trung bình ↔ clamp (3 view: 25/32/35 dB)', fontsize=10)
    a3.legend(fontsize=7.5, loc='upper left', ncol=2)
    fig.suptitle('Hình 5.5 — PSNR: lỗi cũ "theo kênh rồi trung bình" và thứ tự trung bình/clamp trong Score', fontsize=12)
    fig.tight_layout()
    fig.savefig(out('psnr_bug'), **SAVE)
    plt.close(fig)


# ==========================================================================
# Hình 6: Score
# ==========================================================================
def fig_score(coarse):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[0.8, 1.2]))
    parts = [0.4 * (1 - LPIPS_ASSUMED), 0.3 * S, 0.3 * pn]
    names = [f'0.4·(1 − LPIPS)\n= 0.4·(1 − {LPIPS_ASSUMED:.2f})', f'0.3·SSIM\n= 0.3·{S:.4f}', f'0.3·clamp(PSNR/30)\n= 0.3·{pn:.4f}']
    cols = ['#8c564b', C_SSIM, C_PSNR]
    bottom = 0
    for p, n, c in zip(parts, names, cols):
        a1.bar([0], [p], 0.5, bottom=bottom, color=c, alpha=0.9, label=n)
        a1.text(0, bottom + p / 2, f'{p:.4f}', ha='center', va='center', color='white', fontsize=11, weight='bold')
        bottom += p
    a1.text(0, bottom + 0.015, f'Score = {score:.4f}', ha='center', va='bottom', fontsize=12, weight='bold')
    # trần
    a1.bar([1], [0.4, 0.3, 0.3][0], 0.5, color='#8c564b', alpha=0.35)
    a1.bar([1], [0.3], 0.5, bottom=0.4, color=C_SSIM, alpha=0.35)
    a1.bar([1], [0.3], 0.5, bottom=0.7, color=C_PSNR, alpha=0.35)
    for b0, p in ((0, 0.4), (0.4, 0.3), (0.7, 0.3)):
        a1.text(1, b0 + p / 2, f'{p:.1f}', ha='center', va='center', fontsize=10, color='0.3')
    a1.text(1, 1.015, 'trần = 1.0', ha='center', va='bottom', fontsize=10, color='0.3')
    a1.set_xticks([0, 1]); a1.set_xticklabels(['α mô hình = 0.1\n(LPIPS giả định 0.30)', 'tối đa\n(LPIPS=0, SSIM=1, PSNR≥30)'], fontsize=9)
    a1.set_ylim(0, 1.15)
    a1.set_ylabel('Score')
    a1.set_title('Score = 0.4(1−LPIPS) + 0.3·SSIM + 0.3·clamp(PSNR/30)', fontsize=10)
    a1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, 0.93))

    fine = sensitivity(np.round(np.arange(0.05, 0.951, 0.05), 2))
    al = fine[:, 0]
    pn_a = np.clip(np.where(np.isfinite(fine[:, 5]), fine[:, 5], 1e9) / 30, 0, 1)
    sc = 0.4 * (1 - LPIPS_ASSUMED) + 0.3 * fine[:, 2] + 0.3 * pn_a
    sc0 = 0.4 * 1.0 + 0.3 * fine[:, 2] + 0.3 * pn_a
    a2.stackplot(al, 0.4 * (1 - LPIPS_ASSUMED) * np.ones_like(al), 0.3 * fine[:, 2], 0.3 * pn_a,
                 colors=cols, alpha=0.35, labels=['0.4(1−LPIPS), LPIPS = 0.30 cố định', '0.3·SSIM', '0.3·clamp(PSNR/30)'])
    a2.plot(al, sc, '-', color='k', lw=2, label='Score (LPIPS = 0.30)')
    a2.plot(al, sc0, '--', color='0.4', lw=1.5, label='Score nếu LPIPS = 0 (trần 1.0)')
    for r in coarse:
        p_r = min(r[5] / 30, 1.0) if np.isfinite(r[5]) else 1.0
        s_r = 0.4 * (1 - LPIPS_ASSUMED) + 0.3 * r[2] + 0.3 * p_r
        a2.plot(r[0], s_r, 'o', color='k', ms=6)
        a2.annotate(f'{s_r:.4f}', (r[0], s_r), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8.5, weight='bold')
    a2.axhline(1.0, color='k', ls=':', lw=1)
    a2.text(0.06, 1.02, 'trần Score = 1.0 (đường gạch chạm trần khi α ≥ 0.8: SSIM → 1, PSNR ≥ 30)', fontsize=8.5)
    a2.axhline(0.88, color='0.5', ls=':', lw=1)
    a2.text(0.06, 0.885, 'trần khi LPIPS = 0.30: 0.28 + 0.3 + 0.3 = 0.88', fontsize=8.5, color='0.4')
    a2.set_xlabel(r'$\alpha$ mô hình (GT $\alpha=0.9$)'); a2.set_ylabel('Score')
    a2.set_ylim(0, 1.12); a2.set_xlim(0.05, 0.95)
    a2.set_title('Score theo α mô hình (SSIM, PSNR từ bảng độ nhạy; LPIPS cố định)', fontsize=10)
    a2.legend(fontsize=7.5, loc='lower right')
    fig.suptitle('Hình 5.6 — Điểm tổng hợp Score và trần 1.0', fontsize=12)
    fig.tight_layout()
    fig.savefig(out('score'), **SAVE)
    plt.close(fig)


if __name__ == "__main__":
    fig_l1(); print("->", out('l1'))
    fig_ssim(); print("->", out('ssim'))
    fig_loss(); print("->", out('loss'))
    coarse = fig_sensitivity(); print("->", out('sensitivity'))
    fig_psnr_bug(); print("->", out('psnr_bug'))
    fig_score(coarse); print("->", out('score'))
    print(f"L1={L1:.6f} SSIM={S:.6f} loss={0.8*L1+0.2*D_SSIM:.6f} PSNR={PSNR:.4f} PSNR_old={PSNR_old:.4f} Score={score:.6f}")
