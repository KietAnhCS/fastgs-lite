"""
Vẽ hình minh hoạ cho bài test số chương 1 (SfM Points -> Initialization).
Số liệu lấy lại bằng cách chép các hàm trong scripts/ch01_test.py (chỉ numpy, không torch).
Chạy:  python DOCS/Report/test/scripts/ch01_plot.py
Xuất:  DOCS/Report/test/figures/ch01_*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

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

# Màu nhất quán cho 4 Gaussian và 3 camera
GCOL = ['#d62728', '#2ca02c', '#1f77b4', '#7f7f7f']
CAMCOL = ['black', 'tab:orange', 'tab:purple']
RGBCOL = ['#d62728', '#2ca02c', '#1f77b4']


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=450, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("saved", os.path.normpath(path))


# --------------------------------------------------------------------------
# Số liệu — chép lại từ ch01_test.py
# --------------------------------------------------------------------------
points = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.5], [-0.4, -0.2, 1.0], [0.3, -0.5, 0.2]])
colors = np.array([[0.8, 0.2, 0.2], [0.2, 0.7, 0.3], [0.1, 0.3, 0.9], [0.5, 0.5, 0.5]])
cam_centers = np.array([[0.0, 0.0, -4.0], [1.5, 0.0, -4.0], [-1.5, 0.5, -4.0]])
N0, V = 4, 3
C0 = 0.28209479177387814            # utils/sh_utils.py:26


def RGB2SH(rgb):                     # utils/sh_utils.py:114
    return (rgb - 0.5) / C0


def inverse_sigmoid(x):              # utils/general_utils.py:18
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def distCUDA2(pts):                  # simple_knn.cu:148-184 boxMeanDist
    P = pts.shape[0]
    out = np.zeros(P)
    for i in range(P):
        d2 = np.sum((pts - pts[i]) ** 2, axis=1)
        d2 = np.delete(d2, i)
        out[i] = np.sort(d2)[:3].sum() / 3.0
    return out


D2 = np.array([[np.sum((points[i] - points[j]) ** 2) for j in range(N0)] for i in range(N0)])
dist2 = np.maximum(distCUDA2(points), 1e-7)
s_tilde = np.log(np.sqrt(dist2))
s_act = np.exp(s_tilde)
alpha_tilde = inverse_sigmoid(0.1)
k00 = RGB2SH(colors)

cbar = cam_centers.mean(axis=0)
cam_d = np.linalg.norm(cam_centers - cbar, axis=1)
extent = 1.1 * cam_d.max()
n_coef = 16
n_params = 3 + 4 + 3 + 1 + 3 * n_coef      # 59

# ==========================================================================
# Hình 1: cảnh 3D + nhìn từ trên
# ==========================================================================
fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
p_off = [(-0.75, 0.0, -0.35), (0.15, 0.1, 0.25), (-0.6, 0.0, 0.5), (0.2, -0.3, -0.45)]
for i in range(N0):
    ax.scatter(*points[i], color=colors[i], s=900 * s_act[i] ** 2, edgecolor='k', depthshade=False)
    ax.text(points[i, 0] + p_off[i][0], points[i, 1] + p_off[i][1], points[i, 2] + p_off[i][2],
            f"$p_{i+1}$", fontsize=10, color=GCOL[i], fontweight='bold')
c_off = [(0.15, -0.2, -0.25), (0.15, -0.2, -0.25), (-1.2, -0.2, -0.25)]
for v in range(V):
    ax.scatter(*cam_centers[v], marker='^', s=120, color=CAMCOL[v], edgecolor='k')
    ax.quiver(*cam_centers[v], 0, 0, 1.2, color=CAMCOL[v], arrow_length_ratio=0.25, linewidth=1.5)
    ax.text(cam_centers[v, 0] + c_off[v][0], cam_centers[v, 1] + c_off[v][1], cam_centers[v, 2] + c_off[v][2],
            f"cam {v+1}", fontsize=9, color=CAMCOL[v], fontweight='bold')
ax.scatter(*cbar, marker='*', s=200, color='gold', edgecolor='k')
ax.text(cbar[0] + 0.15, cbar[1] + 0.3, cbar[2] + 0.35, r"$\bar c$", fontsize=10)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_xlim(-2, 2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-4.5, 1.5)
ax.view_init(elev=22, azim=-60)
ax.set_title(r"Cảnh đồ chơi: 4 điểm SfM (màu $c_k$, cỡ $\propto s_i$), 3 camera nhìn $+z$", fontsize=10)
# bảng toạ độ ghi ở góc (tránh chồng chữ lên điểm)
lines = [f"$p_{i+1}$ = ({points[i,0]:g}, {points[i,1]:g}, {points[i,2]:g}),  $s_{i+1}$ = {s_act[i]:.4f}"
         for i in range(N0)]
lines += [f"$c_{v+1}$ = ({cam_centers[v,0]:g}, {cam_centers[v,1]:g}, {cam_centers[v,2]:g})" for v in range(V)]
lines += [r"$\bar c$ = (0, 0.1667, −4)"]
for k, (ln, col) in enumerate(zip(lines, GCOL + CAMCOL + ['k'])):
    ax.text2D(0.0, 0.97 - 0.045 * k, ln, transform=ax.transAxes, fontsize=8, color=col)

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_aspect('equal')
for r, ls, lab in [(cam_d.max(), '--', r"$\max_v\|c_v-\bar c\|$ = " + f"{cam_d.max():.4f}"),
                   (extent, '-', f"extent = 1.1 × {cam_d.max():.4f} = {extent:.4f}")]:
    ax2.add_patch(Circle((cbar[0], cbar[1]), r, fill=False, ls=ls, lw=1.6, color='tab:red', label=lab))
for v in range(V):
    ax2.scatter(cam_centers[v, 0], cam_centers[v, 1], marker='^', s=140, color=CAMCOL[v], edgecolor='k', zorder=5)
    ax2.plot([cbar[0], cam_centers[v, 0]], [cbar[1], cam_centers[v, 1]], color=CAMCOL[v], lw=1, alpha=0.7)
    mx, my = (cbar[0] + cam_centers[v, 0]) / 2, (cbar[1] + cam_centers[v, 1]) / 2
    lab_xy = [(0.55, 0.06), (mx, my + 0.12), (mx, my + 0.12)][v]
    ax2.text(lab_xy[0], lab_xy[1], r"$\|c_{%d}-\bar c\|$=%.4f" % (v + 1, cam_d[v]), fontsize=8, color=CAMCOL[v],
             ha='center', va='bottom')
    ax2.annotate(f"cam {v+1}", (cam_centers[v, 0], cam_centers[v, 1]), textcoords='offset points',
                 xytext=(0, -16), ha='center', fontsize=8, color=CAMCOL[v])
ax2.scatter(cbar[0], cbar[1], marker='*', s=220, color='gold', edgecolor='k', zorder=6)
ax2.annotate(r"$\bar c$=(0, 0.1667)", (cbar[0], cbar[1]), textcoords='offset points', xytext=(-10, 6),
             ha='right', fontsize=8)
p_lab_off = [(-28, -4), (6, 4), (6, 4), (6, 4)]
for i in range(N0):
    ax2.scatter(points[i, 0], points[i, 1], color=colors[i], s=60, edgecolor='k', zorder=4)
    ax2.annotate(f"$p_{i+1}$", (points[i, 0], points[i, 1]), textcoords='offset points', xytext=p_lab_off[i],
                 fontsize=8, color=GCOL[i])
ax2.set_xlim(-2.1, 2.1); ax2.set_ylim(-1.8, 2.1)
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title(r"Nhìn từ trên (mặt xy): extent = $1.1\cdot\max_v\|c_v-\bar c\|_2$", fontsize=10)
ax2.legend(loc='upper left', fontsize=8)
fig.tight_layout()
save(fig, "ch01_scene3d.png")

# ==========================================================================
# Hình 2: heatmap ||p_i - p_j||^2 + bar d2_knn3 -> s~ -> s
# ==========================================================================
fig = plt.figure(figsize=(13, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
ax = fig.add_subplot(gs[0, 0])
ax.grid(False)
im = ax.imshow(D2, cmap='YlOrRd', vmin=0, vmax=D2.max())
for i in range(N0):
    for j in range(N0):
        ax.text(j, i, f"{D2[i,j]:.2f}", ha='center', va='center', fontsize=11,
                color='white' if D2[i, j] > 0.8 else 'black')
ax.set_xticks(range(N0)); ax.set_yticks(range(N0))
ax.set_xticklabels([f"$p_{j+1}$" for j in range(N0)]); ax.set_yticklabels([f"$p_{i+1}$" for i in range(N0)])
for t, c in zip(ax.get_xticklabels(), GCOL):
    t.set_color(c)
for t, c in zip(ax.get_yticklabels(), GCOL):
    t.set_color(c)
ax.set_title(r"Ma trận $\|p_i-p_j\|^2$ ($N_0=4$ → 3-NN = 3 điểm còn lại)", fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

gs_r = gs[0, 1].subgridspec(1, 3, wspace=0.45)
titles = [r"$d^2_{\mathrm{knn3}}=\frac{1}{3}\sum_j\|p_i-p_j\|^2$",
          r"$\tilde s_i=\log\sqrt{d^2_{\mathrm{knn3}}}$",
          r"$s_i=\exp\tilde s_i=\sqrt{d^2}$"]
vals = [dist2, s_tilde, s_act]
for k in range(3):
    a = fig.add_subplot(gs_r[0, k])
    bars = a.bar(range(N0), vals[k], color=GCOL, edgecolor='k')
    for i, b in enumerate(bars):
        h = vals[k][i]
        off = 0.02 * max(abs(vals[k]).max(), 1)
        a.text(b.get_x() + b.get_width() / 2, h + (off if h >= 0 else -off),
               f"{h:.4f}", ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
    a.set_xticks(range(N0)); a.set_xticklabels([f"G{i+1}" for i in range(N0)])
    a.set_title(titles[k], fontsize=9, pad=10)
    a.axhline(0, color='k', lw=0.8)
    lo, hi = vals[k].min(), vals[k].max()
    pad = 0.35 * max(hi - lo, 0.3)
    a.set_ylim(min(0, lo) - pad, max(0, hi) + pad)
fig.suptitle("Chương 1(b): từ khoảng cách 3-NN đến scale khởi tạo", fontsize=11)
save(fig, "ch01_knn.png")

# ==========================================================================
# Hình 3: activation sigmoid & exp
# ==========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.linspace(-6, 6, 400)
a1.plot(x, sigmoid(x), color='tab:blue', lw=2, label=r"$\alpha=\sigma(\tilde\alpha)=1/(1+e^{-\tilde\alpha})$")
a1.scatter([alpha_tilde], [0.1], s=90, color='#d62728', zorder=5, edgecolor='k')
a1.plot([alpha_tilde, alpha_tilde], [0, 0.1], ls=':', color='#d62728')
a1.plot([-6, alpha_tilde], [0.1, 0.1], ls=':', color='#d62728')
a1.annotate(r"$\tilde\alpha=\log(0.1/0.9)$" + "\n" + r"$=%.4f$,  $\alpha=0.1$" % alpha_tilde,
            (alpha_tilde, 0.1), textcoords='offset points', xytext=(15, 30), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#d62728'))
at9 = inverse_sigmoid(0.9)
a1.scatter([at9], [0.9], s=60, color='tab:gray', zorder=5, edgecolor='k')
a1.annotate("GT (00-scene): " + r"$\alpha=0.9$" + "\n" + r"$\tilde\alpha=%.4f$" % at9,
            (at9, 0.9), textcoords='offset points', xytext=(-120, -40), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='gray'))
a1.set_xlabel(r"$\tilde\alpha$ (tham số học)"); a1.set_ylabel(r"$\alpha$ (opacity)")
a1.set_title(r"(a) Opacity: $\tilde\alpha_i=\sigma^{-1}(0.1)$ cho cả 4 Gaussian", fontsize=10)
a1.legend(loc='upper left', fontsize=8)
a1.set_xlim(-6, 6); a1.set_ylim(0, 1)

x2 = np.linspace(-0.4, 0.4, 300)
a2.plot(x2, np.exp(x2), color='tab:blue', lw=2, label=r"$s=\exp(\tilde s)$")
for i in range(N0):
    a2.scatter([s_tilde[i]], [s_act[i]], s=90, color=GCOL[i], edgecolor='k', zorder=5)
    a2.annotate(r"G%d: $\tilde s$=%.4f → $s$=%.4f" % (i + 1, s_tilde[i], s_act[i]),
                (s_tilde[i], s_act[i]), textcoords='offset points',
                xytext=(12, -14) if i in (0, 3) else (-12, 10), ha='left' if i in (0, 3) else 'right',
                fontsize=8, color=GCOL[i],
                arrowprops=dict(arrowstyle='-', color=GCOL[i], alpha=0.6))
a2.axhline(1, color='gray', ls=':', lw=1); a2.axvline(0, color='gray', ls=':', lw=1)
a2.set_xlabel(r"$\tilde s$ (tham số học, mỗi trục)"); a2.set_ylabel(r"$s$ (scale)")
a2.set_title(r"(b) Scale: $s_i=\exp\tilde s_i=\sqrt{d^2_{\mathrm{knn3}}}$", fontsize=10)
a2.legend(loc='upper left', fontsize=8)
a2.set_xlim(-0.4, 0.4); a2.set_ylim(0.65, 1.5)
fig.tight_layout()
save(fig, "ch01_activation.png")

# ==========================================================================
# Hình 4: SH bậc 0: ô màu + bar k00
# ==========================================================================
fig = plt.figure(figsize=(12, 4.8))
gs = fig.add_gridspec(2, N0, height_ratios=[1, 4], hspace=0.05, wspace=0.35)
ymax = np.abs(k00).max() * 1.35
for i in range(N0):
    a0 = fig.add_subplot(gs[0, i])
    a0.grid(False)
    a0.add_patch(Rectangle((0, 0), 1, 1, color=colors[i]))
    a0.set_xlim(0, 1); a0.set_ylim(0, 1); a0.set_xticks([]); a0.set_yticks([])
    a0.set_title(f"G{i+1}: $c$=({colors[i,0]:g}, {colors[i,1]:g}, {colors[i,2]:g})", fontsize=9, color=GCOL[i])
    a = fig.add_subplot(gs[1, i])
    bars = a.bar(['R', 'G', 'B'], k00[i], color=RGBCOL, edgecolor='k')
    for b, hgt, ch in zip(bars, k00[i], range(3)):
        a.text(b.get_x() + b.get_width() / 2, hgt + (0.06 if hgt >= 0 else -0.06),
               f"{hgt:.4f}", ha='center', va='bottom' if hgt >= 0 else 'top', fontsize=8)
    a.axhline(0, color='k', lw=0.8)
    a.set_ylim(-ymax, ymax)
    a.set_xlabel(r"$(%g-0.5,\ %g-0.5,\ %g-0.5)/C_0$" % tuple(colors[i]), fontsize=8)
    if i == 0:
        a.set_ylabel(r"$k_{00}=(c-0.5)/C_0$")
    if i == 3:
        a.text(1, ymax * 0.55, "xám $c=0.5$\n→ $k_{00}=0$\n(đúng tâm SH2RGB)", ha='center', fontsize=8,
               color=GCOL[3], bbox=dict(boxstyle='round', fc='white', ec=GCOL[3]))
fig.suptitle(r"Chương 1(e): hệ số SH bậc 0, $C_0=%.4f$, $1/C_0=%.4f$; 45 hệ số $l\geq1$ đều bằng 0" % (C0, 1 / C0),
             fontsize=11, y=1.0)
save(fig, "ch01_sh.png")

# ==========================================================================
# Hình 5: đếm tham số & bộ nhớ
# ==========================================================================
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
a = fig.add_subplot(gs[0, 0])
groups = [(r"$\mu$", 3, '#1f77b4'), (r"$\tilde q$", 4, '#ff7f0e'), (r"$\tilde s$", 3, '#2ca02c'),
          (r"$\tilde\alpha$", 1, '#d62728'), (r"$k_{00}$", 3, '#9467bd'), (r"$k_{lm},l\geq1$", 45, '#7f7f7f')]
rows = [r"$\theta$ (tham số)", "$m$ (Adam moment 1)", "$v$ (Adam moment 2)"]
for r, rname in enumerate(rows):
    left = 0
    for name, n, col in groups:
        a.barh(r, n, left=left, color=col, edgecolor='k', alpha=1 if r == 0 else 0.55,
               label=f"{name}: {n}" if r == 0 else None)
        if n >= 3:
            a.text(left + n / 2, r, str(n), ha='center', va='center', fontsize=8,
                   color='white' if n == 45 else 'black')
        left += n
    a.text(left + 1, r, f"= {left}", va='center', fontsize=9)
a.axvline(14, color='k', ls='--', lw=1)
a.text(13.5, -0.45, "optimizer: 14 tham số → 42 float", ha='right', va='center', fontsize=8)
a.text(14.5, -0.45, "shoptimizer: 45 tham số → 135 float", ha='left', va='center', fontsize=8)
a.set_yticks(range(3)); a.set_yticklabels(rows)
a.set_ylim(-0.75, 2.9)
a.set_xlim(0, 68); a.set_xlabel("số float / Gaussian")
a.invert_yaxis()
a.set_title(r"59 tham số/Gaussian, ×3 với Adam $(\theta, m, v)$ = 177 float", fontsize=10)
a.legend(loc='lower right', fontsize=8, ncol=3)

b = fig.add_subplot(gs[0, 1])
Ns = [4, 10 ** 6]
byts = [3 * n_params * N * 4 for N in Ns]
bars = b.bar(["$N_0=4$", "$N=10^6$"], byts, color=['tab:blue', 'tab:red'], edgecolor='k', width=0.5)
b.set_yscale('log')
b.set_ylim(1e2, 1e11)
labels = [f"177×4 = 708 float\n= {byts[0]:,} byte\n= 2.832 kB",
          f"1.77×10$^8$ float\n= {byts[1]/1e8:.2f}×10$^8$ byte\n= {byts[1]/1e6:.1f} MB = {byts[1]/1024**2:.1f} MiB"]
for bar_, lab in zip(bars, labels):
    b.text(bar_.get_x() + bar_.get_width() / 2, bar_.get_height() * 1.6, lab, ha='center', va='bottom', fontsize=8)
b.set_ylabel("byte (float32 = 4 byte), thang log")
b.set_title(r"Bộ nhớ tham số + Adam = $177\cdot N\cdot 4$ byte", fontsize=10)
save(fig, "ch01_memory.png")
