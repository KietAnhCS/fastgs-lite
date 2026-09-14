"""
Vẽ hình minh hoạ Chương 7 — Adaptive Density Control (DOCS/Report/test/07-test.md).
Import lại toàn bộ số liệu từ ch07_test.py (numpy + scipy, seed 0; không torch).

Chạy:  python DOCS/Report/test/scripts/ch07_plot.py
PNG:   DOCS/Report/test/figures/ch07_<tên>.png
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, HERE)

with contextlib.redirect_stdout(io.StringIO()):
    import ch07_test as T   # chạy toàn bộ test, seed 0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.linewidth": 1.1, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.framealpha": 0.9, "legend.edgecolor": "0.75",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 450,
})
GCOL = ["#d62728", "#2ca02c", "#1f77b4", "#7f7f7f"]
GNAME = [r"$G_1$", r"$G_2$", r"$G_3$", r"$G_4$"]
N, V = T.N, T.V_CAMS


def save(fig, name):
    path = os.path.join(FIG, f"ch07_{name}.png")
    fig.savefig(path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", os.path.normpath(path))


def to_img(C):
    return np.clip(np.transpose(C, (1, 2, 0)), 0, 1)


# ============================================================================
# Hình 1: bản đồ lỗi 3 view x 4 cột
# ============================================================================
fig, axes = plt.subplots(V, 4, figsize=(14, 7.6))
for v in range(V):
    e, eh, m = T.E[v], T.EHAT[v], T.MASK[v]
    ax = axes[v, 0]
    ax.imshow(to_img(T.REND[v]), interpolation="nearest")
    ax.set_title(rf"view {v+1}: $I_{{\mathrm{{rend}}}}$ ($\alpha=0.1$)", fontsize=10)
    ax = axes[v, 1]
    ax.imshow(to_img(T.GT[v]), interpolation="nearest")
    ax.set_title(rf"view {v+1}: $I_{{\mathrm{{gt}}}}$ ($\alpha=0.9$)", fontsize=10)
    ax = axes[v, 2]
    im = ax.imshow(e, cmap="magma", vmin=0, vmax=0.45, interpolation="nearest")
    ax.set_title(r"$e_v(x)=\frac{1}{3}\sum_{ch}|I_{\mathrm{rend}}-I_{\mathrm{gt}}|$", fontsize=10)
    ax.text(0.02, 0.97, f"min={e.min():.4f}\nmax={e.max():.4f}\nmean={e.mean():.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8, color="white",
            bbox=dict(boxstyle="round", fc="black", alpha=0.55, ec="none"))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax = axes[v, 3]
    ax.imshow(m, cmap=ListedColormap(["white", "#d62728"]), vmin=0, vmax=1, interpolation="nearest")
    thr = e.min() + 0.1 * (e.max() - e.min())
    ax.set_title(rf"$m_v=\mathbb{{1}}[\hat e_v>0.1]$  ($e>{thr:.4f}$)", fontsize=10)
    ax.text(0.02, 0.97, f"bật {int(m.sum())}/{T.H*T.W} px", transform=ax.transAxes, va="top",
            ha="left", fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="gray"))
for ax in axes.ravel():
    ax.grid(False)
    ax.set_xticks([0, 16, 32, 47]); ax.set_yticks([0, 16, 31])
    ax.tick_params(labelsize=7)
fig.suptitle("Bước ①–③: sai số màu $e_v(x)$, chuẩn hoá min–max $\\hat e_v$ và mặt nạ nhị phân $m_v$ (3 view, 48×32 px)",
             fontsize=12)
fig.tight_layout()
save(fig, "error_maps")

# ============================================================================
# Hình 2: footprint + counts + Importance
# ============================================================================
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
v = 0
for i in range(N):
    ax = fig.add_subplot(gs[0, i])
    rgb = np.ones((T.H, T.W, 3))
    # mask lỗi: xám nhạt
    rgb[T.MASK[v] == 1] = [0.82, 0.82, 0.82]
    # footprint: màu Gaussian, đậm ở chỗ mask bật
    col = np.array(matplotlib.colors.to_rgb(GCOL[i]))
    f = T.FOOT[v][i]
    rgb[f & (T.MASK[v] == 0)] = 0.45 * col + 0.55
    rgb[f & (T.MASK[v] == 1)] = col
    ax.imshow(rgb, interpolation="nearest")
    ax.grid(False)
    mu2 = T.ITEMS[v][i]["mu2"]
    ax.plot(mu2[0], mu2[1], "k+", ms=10, mew=2)
    ax.set_title(f"{GNAME[i]} view 1: $|\\Omega_i|$={int(f.sum())} px", fontsize=10, color=GCOL[i])
    ax.text(0.02, 0.97, f"counts$_i^{{(1)}}$ = {T.COUNTS[i, v]}", transform=ax.transAxes, va="top",
            fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="gray"))
    ax.set_xticks([0, 16, 32, 47]); ax.set_yticks([0, 16, 31]); ax.tick_params(labelsize=7)
leg = [Patch(fc="#d1d1d1", label="mask lỗi $m_1(x)=1$ (ngoài Ω$_i$)"),
       Patch(fc="#888888", label="Ω$_i$ ∩ mask (đậm) = pixel được đếm"),
       Patch(fc="#cccccc", ec="gray", label="Ω$_i$ ∖ mask (nhạt) = không đếm"),
       Line2D([], [], marker="+", color="k", ls="", ms=8, mew=2, label="$\\mu'_i$")]
fig.legend(handles=leg, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.945), frameon=True)

# bảng counts 4x3 heatmap
ax = fig.add_subplot(gs[1, 0:2])
im = ax.imshow(T.COUNTS, cmap="Blues", vmin=700, vmax=1300)
for i in range(N):
    for vv in range(V):
        ax.text(vv, i, str(T.COUNTS[i, vv]), ha="center", va="center", fontsize=11,
                color="white" if T.COUNTS[i, vv] > 1100 else "black")
ax.set_xticks(range(V)); ax.set_xticklabels([f"view {k+1}" for k in range(V)])
ax.set_yticks(range(N)); ax.set_yticklabels(GNAME)
for lab, c in zip(ax.get_yticklabels(), GCOL):
    lab.set_color(c)
ax.grid(False)
ax.set_title(r"Bước ④: counts$_i^{(v)}=\sum_{x\in\Omega_i^{(v)}} m_v(x)$", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="counts")

# bar sum counts + Importance
ax = fig.add_subplot(gs[1, 2:4])
x = np.arange(N)
sums = T.COUNTS.sum(1)
b1 = ax.bar(x - 0.2, sums, 0.38, color=GCOL, alpha=0.5, edgecolor=GCOL, label=r"$\sum_v$ counts")
b2 = ax.bar(x + 0.2, T.IMPORTANCE, 0.38, color=GCOL, label=r"Importance $=\lfloor\frac{1}{3}\sum_v\mathrm{counts}\rfloor$")
for k in range(N):
    ax.text(x[k] - 0.2, sums[k] + 30, str(sums[k]), ha="center", fontsize=9)
    ax.text(x[k] + 0.2, T.IMPORTANCE[k] + 30, str(T.IMPORTANCE[k]), ha="center", fontsize=10, fontweight="bold")
ax.axhline(5, color="k", ls="--", lw=1)
ax.text(3.45, 60, "ngưỡng Importance > 5", ha="right", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(GNAME)
for lab, c in zip(ax.get_xticklabels(), GCOL):
    lab.set_color(c)
ax.set_ylim(0, 4600)
ax.set_title("Bước ⑤: Importance (cả 4 ≫ 5 → không ai bị chặn densify)", fontsize=11)
ax.legend(fontsize=9, loc="upper center", ncol=2)
fig.suptitle("Bước ④–⑤: footprint hữu hình $\\Omega_i^{(1)}$ chồng mask lỗi, counts và Importance", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.9])
save(fig, "footprint_counts")

# ============================================================================
# Hình 3: E_photo, Pruning, w_i
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
vc = ["#9467bd", "#8c564b", "#e377c2"]
L1s = [T.l1(T.REND[v], T.GT[v]) for v in range(V)]
Ss = [T.ssim(T.REND[v], T.GT[v]) for v in range(V)]
xs = np.arange(V)
ax.bar(xs, [0.8 * l for l in L1s], 0.55, color="#4c72b0", label=r"$0.8\,\mathcal{L}_1$")
ax.bar(xs, [0.2 * (1 - s) for s in Ss], 0.55, bottom=[0.8 * l for l in L1s], color="#dd8452",
       label=r"$0.2\,(1-\mathrm{SSIM})$")
for v in range(V):
    ax.text(xs[v], T.EPHOTO[v] + 0.006, f"{T.EPHOTO[v]:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.text(xs[v], 0.8 * L1s[v] / 2, f"L1={L1s[v]:.4f}", ha="center", va="center", fontsize=8, color="white")
    ax.text(xs[v], 0.8 * L1s[v] + 0.2 * (1 - Ss[v]) / 2, f"SSIM={Ss[v]:.3f}", ha="center", va="center", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels([f"view {v+1}" for v in range(V)])
ax.set_ylim(0, 0.33)
ax.set_title(r"(a) Bước ⑥: $E^{(v)}_{\mathrm{photo}}=0.8\mathcal{L}_1+0.2(1-\mathrm{SSIM})$", fontsize=10)
ax.legend(fontsize=9, loc="upper right")

ax = axes[1]
x = np.arange(N)
ax.bar(x - 0.2, T.RAW, 0.38, color=GCOL, alpha=0.45, edgecolor=GCOL)
for k in range(N):
    ax.text(x[k] - 0.2, T.RAW[k] + 4, f"{T.RAW[k]:.1f}", ha="center", fontsize=8)
ax.set_ylim(700, 830)
ax.set_ylabel(r"thô: $\sum_v \mathrm{counts}_i^{(v)}E^{(v)}_{\mathrm{photo}}$ (nhạt, trục trái)", fontsize=9)
ax.axhline(T.RAW.min(), color="gray", ls=":", lw=1); ax.axhline(T.RAW.max(), color="gray", ls=":", lw=1)
ax.text(1.5, T.RAW.min() + 1.5, f"min thô = {T.RAW.min():.1f} → Pruning 0", fontsize=7, color="gray", ha="center")
ax.text(1.5, T.RAW.max() + 1.5, f"max thô = {T.RAW.max():.1f} → Pruning 1", fontsize=7, color="gray", ha="center")
ax2 = ax.twinx()
ax2.bar(x + 0.2, T.PRUNING, 0.38, color=GCOL)
for k in range(N):
    ax2.text(x[k] + 0.2, T.PRUNING[k] + 0.02, f"{T.PRUNING[k]:.3f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_ylim(0, 1.18)
ax2.set_ylabel("Pruning = minmax qua 4 Gaussian (đậm, trục phải)", fontsize=9)
ax2.grid(False)
ax.set_xticks(x); ax.set_xticklabels(GNAME)
for lab, c in zip(ax.get_xticklabels(), GCOL):
    lab.set_color(c)
ax.set_title("(b) Bước ⑦: điểm thô → Pruning", fontsize=10)

ax = axes[2]
ax.bar(x, T.W_I, 0.55, color=GCOL)
for k in range(N):
    lab = f"{T.W_I[k]:.3g}" if T.W_I[k] < 1e5 else r"$10^6$"
    ax.text(x[k], T.W_I[k] * 1.4, lab, ha="center", fontsize=10, fontweight="bold")
ax.set_yscale("log"); ax.set_ylim(0.5, 1e7)
ax.set_xticks(x); ax.set_xticklabels(GNAME)
for lab, c in zip(ax.get_xticklabels(), GCOL):
    lab.set_color(c)
ax.set_ylabel(r"$w_i$ (log)")
ax.set_title(r"(c) $w_i=1/(10^{-6}+1-\mathrm{Pruning}_i)$", fontsize=10)
p = T.W_I / T.W_I.sum()
ax.text(0.03, 0.95, "xác suất rút lần 1:\n" + "\n".join(f"{GNAME[k]}: {p[k]:.1e}" for k in range(N)),
        transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="gray"))
fig.suptitle("Bước ⑥–⑦: photometric loss từng view, điểm Pruning và trọng số rút multinomial", fontsize=12)
fig.tight_layout()
save(fig, "scores")

# ============================================================================
# Hình 4: sơ đồ quyết định densify
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
dext = T.DELTA * T.EXTENT
for ax, g, tau, tname, gname in ((axes[0], T.GBAR, T.TAU_GRAD, r"$\tau_{\mathrm{grad}}=2\times10^{-4}$", r"$\bar g_i$ (có dấu)"),
                                 (axes[1], T.GBAR_ABS, T.TAU_GRAD_ABS, r"$\tau^{\mathrm{abs}}_{\mathrm{grad}}=1.2\times10^{-3}$", r"$\bar g^{\mathrm{abs}}_i$")):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(3e-5, 3e-1); ax.set_ylim(2e-4, 5)
    # vùng
    ax.axvspan(tau, 3e-1, ymin=0, ymax=1, color="#2ca02c", alpha=0.06)
    ax.axvline(tau, color="k", ls="--", lw=1.2)
    ax.axhline(dext, color="purple", ls="-.", lw=1.2)
    ax.text(tau * 1.1, 3.2, tname, fontsize=9, ha="left")
    ax.text(3.5e-5, dext * 1.25, rf"$\delta\cdot\mathrm{{extent}}={dext:.5f}$", fontsize=9, color="purple")
    ax.text(0.03, 0.62, "vùng SPLIT\n(grad ≥ τ, s > δ·extent)", transform=ax.transAxes, fontsize=9,
            color="#2ca02c", ha="left", alpha=0.9)
    ax.text(0.62, 0.62, "SPLIT", transform=ax.transAxes, fontsize=16, color="#2ca02c", alpha=0.35, fontweight="bold")
    ax.text(0.62, 0.12, "CLONE", transform=ax.transAxes, fontsize=16, color="#1f77b4", alpha=0.35, fontweight="bold")
    ax.text(0.05, 0.12, "không densify", transform=ax.transAxes, fontsize=11, color="gray", alpha=0.8)
    for i in range(N):
        ax.scatter(g[i], T.MAXS[i], s=110, color=GCOL[i], edgecolor="k", zorder=5)
    for i in range(N):
        ax.text(0.97, 0.50 - 0.05 * i, f"{GNAME[i]}: ({g[i]:.2e}, {T.MAXS[i]:.3f})", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=GCOL[i],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.text(0.97, 0.55, "(gradient, max s) của 4 Gaussian:", transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    ax.set_xlabel(gname + " (NDC, log)")
    ax.set_ylabel(r"$\max_k s_{i,k}$ (log)")
    ax.set_title(f"{gname} so với {tname}", fontsize=10)
axes[0].text(0.5, 0.02, "3DGS gốc: nhánh clone dùng $\\bar g$", transform=axes[0].transAxes, ha="center", fontsize=8, color="gray")
axes[1].text(0.5, 0.02, "3DGS gốc: nhánh split dùng $\\bar g^{\\mathrm{abs}}$", transform=axes[1].transAxes, ha="center", fontsize=8, color="gray")
txt = ("Quyết định FastGS-lite = (gradient ≥ τ) AND (Importance > 5)\n"
       + "  ".join(f"{GNAME[i]}: Imp={T.IMPORTANCE[i]} ✓" for i in range(N))
       + "\n→ clone 0, split 4 (δ·extent nhỏ hơn mọi s nên không ai vào nhánh clone)")
fig.text(0.5, -0.09, txt, ha="center", fontsize=10, bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0a800"))
fig.suptitle("7.3 — Sơ đồ quyết định densify: gradient trung bình vs kích thước Gaussian", fontsize=12)
fig.tight_layout()
save(fig, "densify_decision")

# ============================================================================
# Hình 5: split
# ============================================================================
children = {}
for (i, m, st) in T.SPLIT_CHILDREN:
    children.setdefault(i, []).append((m, float(np.exp(st[0]))))
removed = set(int(k) for k in np.where(T.REMOVE)[0])   # chỉ số gốc bị rút -> G_i c_1 (thứ tự repeat(2,1))

OFF_ORIG = [(-70, 22), (26, -16), (-30, 26), (34, -30)]
OFF_CH = [[(-52, -24), (-38, -30)], [(8, 8), (-42, 12)], [(10, -12), (-16, -22)], [(12, 8), (10, -12)]]
fig, axes = plt.subplots(1, 2, figsize=(16, 7.4))
for ax, (a, b, la, lb) in zip(axes, ((0, 1, "x", "y"), (0, 2, "x", "z"))):
    ax.set_aspect("equal")
    for i in range(N):
        mu = T.P[i]; s = T.S[i, 0]
        ax.add_patch(Circle((mu[a], mu[b]), s, fc=GCOL[i], alpha=0.07, ec=GCOL[i], lw=1.5, ls="--"))
        ax.plot(mu[a], mu[b], "o", color=GCOL[i], ms=8, mec="k", zorder=6)
        ax.annotate(f"{GNAME[i]} s={s:.3f}", (mu[a], mu[b]), xytext=OFF_ORIG[i], textcoords="offset points",
                    fontsize=8, color=GCOL[i], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=GCOL[i], lw=0.6, alpha=0.7))
        for j, (m, sc) in enumerate(children[i]):
            ax.add_patch(Circle((m[a], m[b]), sc, fc=GCOL[i], alpha=0.25, ec=GCOL[i], lw=1.2))
            ax.add_patch(FancyArrowPatch((mu[a], mu[b]), (m[a], m[b]), arrowstyle="-|>", mutation_scale=12,
                                         color=GCOL[i], lw=1.2, alpha=0.9, zorder=5))
            ax.plot(m[a], m[b], "s", color=GCOL[i], ms=5, mec="k", zorder=6)
            tag = f"$G_{i+1}c_{j+1}$"
            if j == 0 and i in removed:
                ax.plot(m[a], m[b], "x", color="k", ms=16, mew=3, zorder=7)
                tag += " (rút)"
            ax.annotate(tag, (m[a], m[b]), xytext=OFF_CH[i][j], textcoords="offset points", fontsize=7.5,
                        color=GCOL[i], bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                        arrowprops=dict(arrowstyle="-", color=GCOL[i], lw=0.6, alpha=0.7))
    ax.set_xlabel(la); ax.set_ylabel(lb)
    ax.set_xlim(-4.2, 3.2); ax.set_ylim(-2.4, 2.6) if b == 1 else ax.set_ylim(-1.6, 2.4)
    ax.set_title(f"Mặt phẳng {la}{lb} (top-view)", fontsize=10)
leg = [Line2D([], [], marker="o", color="gray", mec="k", ls="--", label="gốc: vòng tròn bán kính $s_i$ (nét đứt)"),
       Line2D([], [], marker="s", color="gray", mec="k", ls="-", label="con: bán kính $s_i/1.6$, $\\mu+\\epsilon$, $\\epsilon\\sim N(0,s_i^2)$ seed 0"),
       Line2D([], [], marker="x", color="k", ls="", ms=10, mew=3, label="bị multinomial rút (chỉ số {1,3} → $G_1c_1$, $G_3c_1$)")]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("7.3 split ×4 rồi 7.4 prune: N = 4 → 8 con (bán kính s/1.6 = " +
             ", ".join(f"{children[i][0][1]:.3f}" for i in range(N)) + ") → 6 (công thức 7.8)", fontsize=12)
fig.text(0.5, 0.905, "N: 4 → 8 → 6", ha="center", fontsize=14, fontweight="bold",
         bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0a800"))
fig.tight_layout(rect=[0, 0.03, 1, 0.9])
save(fig, "split")

# ============================================================================
# Hình 6: kịch bản N=10 + luỹ thừa N
# ============================================================================
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
x10 = np.arange(1, 11)
cc = ["#d62728" if c else "#9e9e9e" for c in T.C10]

ax = fig.add_subplot(gs[0, 0])
ax.bar(x10, T.PR10, color=cc, edgecolor="k", lw=0.5)
for k in range(10):
    ax.text(x10[k], T.PR10[k] + 0.02, f"{T.PR10[k]:.2f}", ha="center", fontsize=8)
ax.set_ylim(0, 1.15); ax.set_xticks(x10)
ax.set_title("Pruning$_i$ cho trước (đỏ = $i\\in\\mathcal{C}=\\{3,6,8,9,10\\}$)", fontsize=10)
ax.set_xlabel("i")

ax = fig.add_subplot(gs[0, 1])
ax.bar(x10, T.W10, color=cc, edgecolor="k", lw=0.5)
for k in range(10):
    ax.text(x10[k], T.W10[k] * 1.4, f"{T.W10[k]:.4g}" if T.W10[k] < 1e5 else r"$10^6$", ha="center", fontsize=7.5)
ax.set_yscale("log"); ax.set_ylim(0.5, 1e7); ax.set_xticks(x10)
ax.set_title(r"$w_i=1/(10^{-6}+1-\mathrm{Pruning}_i)$ (log)", fontsize=10)
ax.set_xlabel("i")

ax = fig.add_subplot(gs[0, 2])
fS = T.freqS10 / 1000; fR = T.freq10 / 1000
ax.bar(x10 - 0.2, fS, 0.4, color="#bbbbbb", edgecolor="k", lw=0.5, label=r"tần suất $\in\mathcal{S}$ (được rút)")
ax.bar(x10 + 0.2, fR, 0.4, color=cc, edgecolor="k", lw=0.5, label=r"tần suất bị xoá $=\mathcal{C}\cap\mathcal{S}$")
for k in range(10):
    ax.text(x10[k] - 0.2, fS[k] + 0.015, f"{fS[k]:.3f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
    ax.text(x10[k] + 0.2, fR[k] + 0.015, f"{fR[k]:.3f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
ax.set_ylim(0, 1.45); ax.set_xticks(x10)
ax.set_title(f"1000 lần rút (seed 0), budget = ⌊0.5·5⌋ = {T.B10}", fontsize=10)
ax.set_xlabel("i")
ax.legend(fontsize=8, loc="upper left")
ax.text(0.03, 0.72, f"budget = {T.B10}\nxoá thật TB = {np.mean(T.nrem10):.3f}\n(min {min(T.nrem10)}, max {max(T.nrem10)})",
        transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0a800"))

ax = fig.add_subplot(gs[1, :])
base = (1 + 0.15) * (1 - 0.05)
n = np.arange(0, 146)
for N0, c in ((4, "#1f77b4"), (100_000, "#d62728")):
    ax.plot(n, N0 * base ** n, color=c, lw=2, label=rf"$N_0={N0:,}$")
    for nn in (28, 145):
        val = N0 * base ** nn
        ax.plot(nn, val, "o", color=c, ms=7, mec="k", zorder=5)
        ax.annotate(f"n={nn}: {val:.3g}", (nn, val), xytext=(-8, 10), textcoords="offset points",
                    fontsize=9, color=c, ha="right" if nn == 145 else "left")
ax.axvline(28, color="gray", ls=":"); ax.axvline(145, color="gray", ls=":")
ax.text(28, 1.5, "interval 500\n→ n = 28", ha="center", fontsize=8, color="gray")
ax.text(145, 1.5, "interval 100\n→ n = 145", ha="right", fontsize=8, color="gray")
ax.set_yscale("log"); ax.set_ylim(1, 1e11); ax.set_xlim(0, 150)
ax.set_xlabel("n = số lần densify"); ax.set_ylabel(r"$N_{\mathrm{cuối}}$ (log)")
ax.set_title(rf"$N_{{\mathrm{{cuối}}}}\approx N_0[(1+r_s)(1-r_p)]^n$, $r_s=0.15$, $r_p=0.05$ → cơ số $={base:.4f}$", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
fig.suptitle("7.4 kịch bản N=10: sampling multinomial có trọng số; 7.3 cuối: luỹ thừa của số Gaussian", fontsize=12)
fig.tight_layout()
save(fig, "prune_multinomial")
