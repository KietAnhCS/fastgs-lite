"""
Hình minh hoạ tiêu chí densify (clone / split) trong ADC của 3DGS (Kerbl 2023).
Công thức (SPEC mục 3):
    clone_i = [||g_bar_i|| >= tau_grad] AND [max(s_i) <= delta*extent]
    split_i = [||g_bar_i|| >= tau_grad] AND [max(s_i) >  delta*extent]
    tau_grad = 2e-4, delta = percent_dense = 0.01
Prune theo kích thước (SPEC mục 5): max(s_i) > 0.1*extent.

Xuất 3 hình vào cùng thư mục với script:
    fig_05_decision_plane.png
    fig_06_scene_extent.png
    fig_07_threshold_sweep.png
Chỉ dùng numpy + matplotlib. Chạy: python make_fig_densify.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Patch
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ cấu hình
SEED = 20231
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TAU_GRAD = 2e-4        # ngưỡng gradient (densify_grad_threshold)
DELTA = 0.01           # percent_dense
PRUNE_SCALE = 0.1      # ngưỡng prune theo kích thước (× extent)

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["legend.fontsize"] = 8

C_KEEP = "#d9d9d9"     # giữ nguyên
C_CLONE = "#a8dadc"    # clone (xanh lam nhạt)
C_SPLIT = "#f4a582"    # split (cam nhạt)
C_PRUNE = "#c9c9ff"    # vùng bị prune (tím nhạt)
C_TAU = "#c0392b"      # màu vạch tau_grad
C_DELTA = "#1f4e79"    # màu vạch delta*extent
C_PR = "#6a3d9a"       # màu vạch prune

saved = []


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    saved.append(path)


# ------------------------------------------------------------------ tiện ích
def decide(g, s_rel):
    """Trả về 'keep' / 'clone' / 'split' theo công thức 3 (s_rel = max(s)/extent)."""
    if g < TAU_GRAD:
        return "keep"
    return "clone" if s_rel <= DELTA * 1.0 else "split"


# ======================================================================
# HÌNH 5 — mặt phẳng quyết định
# ======================================================================
def fig_05():
    rng = np.random.default_rng(SEED)
    x_lo, x_hi = 1e-5, 2e-3
    y_lo, y_hi = 1e-3, 0.3

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(10.2, 5.6),
                                  gridspec_kw={"width_ratios": [2.9, 1.35]})
    axr.axis("off")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # 3 vùng màu
    ax.fill_betweenx([y_lo, y_hi], x_lo, TAU_GRAD, color=C_KEEP, alpha=0.9, zorder=0)
    ax.fill_betweenx([y_lo, DELTA], TAU_GRAD, x_hi, color=C_CLONE, alpha=0.9, zorder=0)
    ax.fill_betweenx([DELTA, y_hi], TAU_GRAD, x_hi, color=C_SPLIT, alpha=0.9, zorder=0)
    # vùng prune theo kích thước (gạch chéo phía trên 0.1*extent)
    ax.fill_betweenx([PRUNE_SCALE, y_hi], x_lo, x_hi, facecolor="none",
                     hatch="///", edgecolor=C_PR, linewidth=0, alpha=0.55, zorder=1)

    # vạch ngưỡng
    ax.axvline(TAU_GRAD, color=C_TAU, lw=1.8, zorder=2)
    ax.axhline(DELTA, color=C_DELTA, lw=1.8, zorder=2)
    ax.axhline(PRUNE_SCALE, color=C_PR, lw=1.6, ls="--", zorder=2)
    ax.text(TAU_GRAD * 1.07, 1.25e-3, r"$\tau_{grad}=2\cdot10^{-4}$",
            color=C_TAU, fontsize=9, ha="left", va="bottom", zorder=5)
    ax.text(1.15e-5, DELTA * 1.12, r"$\delta\cdot extent = 0.01\,extent$",
            color=C_DELTA, fontsize=9, ha="left", va="bottom", zorder=5)
    ax.text(1.15e-5, PRUNE_SCALE * 1.12, r"prune: $\max(s_i) > 0.1\,extent$",
            color=C_PR, fontsize=9, ha="left", va="bottom", zorder=5)

    # nhãn vùng
    ax.text(4.5e-5, 0.04, "GIỮ NGUYÊN\n$||\\bar{g}_i|| < \\tau_{grad}$",
            ha="center", va="center", fontsize=10, color="#444", zorder=5)
    ax.text(7e-4, 2.8e-3, "CLONE\n$||\\bar{g}_i|| \\geq \\tau_{grad}$ và $\\max(s_i) \\leq \\delta\\,extent$",
            ha="center", va="center", fontsize=9.5, color="#0b525b", zorder=5)
    ax.text(7e-4, 0.045, "SPLIT\n$||\\bar{g}_i|| \\geq \\tau_{grad}$ và $\\max(s_i) > \\delta\\,extent$",
            ha="center", va="center", fontsize=9.5, color="#7f2704", zorder=5)

    # ~300 điểm mô phỏng: log-normal cho g, log-uniform-ish cho s
    n = 300
    g = np.exp(rng.normal(np.log(1e-4), 1.0, n))
    s = np.exp(rng.normal(np.log(0.006), 1.1, n))
    g = np.clip(g, x_lo * 1.2, x_hi / 1.2)
    s = np.clip(s, y_lo * 1.2, y_hi / 1.2)
    cols = {"keep": "#6b6b6b", "clone": "#1b7d84", "split": "#c1440e"}
    dec = np.array([decide(gi, si) for gi, si in zip(g, s)])
    for k, c in cols.items():
        m = dec == k
        ax.scatter(g[m], s[m], s=9, color=c, alpha=0.55, linewidths=0, zorder=3)

    # 6 điểm ví dụ A..F
    examples = [
        ("A", 3e-4, 0.004, "clone"),
        ("B", 5e-4, 0.03, "split"),
        ("C", 1e-4, 0.004, "giữ"),
        ("D", 1e-4, 0.05, "giữ"),
        ("E", 2e-4, 0.01, "clone (đúng biên ≥, ≤)"),
        ("F", 8e-4, 0.15, "split, rồi bị prune"),
    ]
    offsets = {"A": (8, -12), "B": (8, 6), "C": (8, -12), "D": (8, 6),
               "E": (-14, 8), "F": (8, 6)}
    for lab, gx, sy, _ in examples:
        ax.scatter([gx], [sy], s=70, marker="o", facecolor="white",
                   edgecolor="black", linewidths=1.3, zorder=6)
        ax.annotate(lab, (gx, sy), xytext=offsets[lab], textcoords="offset points",
                    fontsize=10, fontweight="bold", zorder=7)

    # bảng nhỏ
    cell = [[lab, f"{gx:.0e}", f"{sy:g}", d] for lab, gx, sy, d in examples]
    tb = axr.table(cellText=cell,
                   colLabels=["", r"$||\bar{g}_i||$", r"$\max(s_i)/ext$", "Quyết định"],
                   colWidths=[0.08, 0.2, 0.24, 0.48],
                   cellLoc="left", colLoc="left",
                   bbox=[0.0, 0.56, 1.0, 0.40], zorder=8)
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    axr.text(0.0, 0.975, "Ví dụ A..F (toạ độ và quyết định)", fontsize=9,
             fontweight="bold", va="bottom", transform=axr.transAxes)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor("#888")
        cl.set_facecolor("white" if r > 0 else "#eeeeee")
        cl.set_alpha(0.95)
    axr.text(0.0, 0.535, "F: max(s) > 0.1·extent nên bị prune ở bước sau",
             fontsize=7.5, color=C_PR, va="top", transform=axr.transAxes)

    # legend
    handles = [
        Patch(color=C_KEEP, label="Giữ nguyên"),
        Patch(color=C_CLONE, label="Clone"),
        Patch(color=C_SPLIT, label="Split"),
        Patch(facecolor="none", edgecolor=C_PR, hatch="///", label="Bị prune (max(s) > 0.1 extent)"),
        Line2D([], [], color=C_TAU, lw=1.8, label=r"$\tau_{grad}$"),
        Line2D([], [], color=C_DELTA, lw=1.8, label=r"$\delta\cdot extent$"),
        Line2D([], [], marker="o", color="black", mfc="white", ls="", label="Điểm ví dụ A..F"),
    ]
    axr.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.46),
               framealpha=0.95, ncol=1, title="Chú giải")

    ax.set_xlabel(r"$||\bar{g}_i||$  (chuẩn gradient vị trí 2D trung bình, thang log)")
    ax.set_ylabel(r"$\max(s_i)\,/\,extent$  (kích thước lớn nhất so với cảnh, thang log)")
    ax.set_title("Mặt phẳng quyết định densify: giữ nguyên / clone / split "
                 r"($\tau_{grad}=2\cdot10^{-4}$, $\delta=0.01$)")
    save(fig, "fig_05_decision_plane.png")
    return examples


# ======================================================================
# HÌNH 6 — scene extent
# ======================================================================
def fig_06():
    rng = np.random.default_rng(SEED + 1)
    # camera nằm trên vòng quanh cảnh (chừa một khoảng trống), point cloud ở giữa
    n_cam = 28
    ang = np.linspace(0.12 * np.pi, 1.88 * np.pi, n_cam) + rng.normal(0, 0.03, n_cam)
    rad = 6.0 + rng.normal(0, 0.4, n_cam)
    cams = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    pts = rng.normal(0, 1.4, (350, 2)) * np.array([1.3, 0.8])

    center = cams.mean(axis=0)                       # tâm các camera
    dist = np.linalg.norm(cams - center, axis=1)
    extent = 1.1 * dist.max()                        # cách Inria tính
    i_far = int(dist.argmax())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.8, 5.0),
                                  gridspec_kw={"width_ratios": [1.4, 1]})
    # --- panel trái
    ax.scatter(pts[:, 0], pts[:, 1], s=4, color="#7a7a7a", alpha=0.6, label="Point cloud (SfM)")
    ax.scatter(cams[:, 0], cams[:, 1], marker="^", s=38, color="#1f4e79",
               edgecolor="white", linewidths=0.5, label="Camera")
    ax.scatter([center[0]], [center[1]], marker="x", s=70, color="black", lw=1.8,
               label="Tâm các camera", zorder=5)
    ax.plot([center[0], cams[i_far, 0]], [center[1], cams[i_far, 1]],
            color="#c0392b", lw=1.2, ls="--", zorder=4)
    mid = (center + cams[i_far]) / 2
    ax.annotate(r"$d_{max}$ (camera xa tâm nhất)", mid, xytext=(10, 0), textcoords="offset points",
                fontsize=8, color="#c0392b", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.add_patch(Circle(center, extent, fill=False, color="#c0392b", lw=1.8,
                        label=f"extent = 1.1 × d_max ≈ {extent:.2f}"))

    # 2 Gaussian ellipse với max(s) = 0.01·extent và 0.1·extent
    s_small = DELTA * extent
    s_big = PRUNE_SCALE * extent
    e1 = Ellipse((-2.4, 2.0), 2 * s_small, 2 * s_small * 0.55, angle=25,
                 facecolor=C_CLONE, edgecolor=C_DELTA, lw=1.5, zorder=6)
    e2 = Ellipse((2.6, -1.4), 2 * s_big, 2 * s_big * 0.5, angle=-20,
                 facecolor=C_PRUNE, edgecolor=C_PR, lw=1.5, alpha=0.85, zorder=6)
    ax.add_patch(e1)
    ax.add_patch(e2)
    ax.annotate(f"Gaussian có max(s) = 0.01·extent ≈ {s_small:.3f}\n(ngưỡng clone/split, rất nhỏ)",
                (-2.4, 2.0), xytext=(-4.6, 5.3), fontsize=8, color=C_DELTA, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_DELTA, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_DELTA, alpha=0.9))
    ax.annotate(f"Gaussian có max(s) = 0.1·extent ≈ {s_big:.2f}\n(ngưỡng prune theo kích thước)",
                (2.6 + s_big * 0.5, -1.4 - 0.3), xytext=(1.0, -4.3), fontsize=8, color=C_PR, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_PR, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_PR, alpha=0.9))
    ax.set_aspect("equal")
    lim = extent * 1.12
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_xlabel("x (đơn vị cảnh)")
    ax.set_ylabel("y (đơn vị cảnh)")
    ax.set_title("extent theo cách tính của Inria:\n1.1 × khoảng cách xa nhất từ camera tới tâm các camera")
    ax.legend(loc="lower left", framealpha=0.95, fontsize=7.5)
    ax.grid(alpha=0.25)

    # --- panel phải: thước tỉ lệ
    bars = [
        (f"0.01·extent ≈ {s_small:.3f}", s_small, C_CLONE, C_DELTA, "ngưỡng clone ↔ split (δ = 0.01)"),
        (f"0.1·extent ≈ {s_big:.2f}", s_big, C_PRUNE, C_PR, "ngưỡng prune theo kích thước"),
        (f"extent ≈ {extent:.2f}", extent, "#f8d7c4", "#c0392b", "kích thước cảnh (Inria: cameras_extent)"),
    ]
    ys = [2, 1, 0]
    for (lab, L, fc, ec, note), y in zip(bars, ys):
        ax2.barh(y, L, height=0.5, color=fc, edgecolor=ec, lw=1.4)
        if L > 0.5 * extent:   # thanh dài: ghi chữ bên trong thanh
            ax2.text(0.02 * extent, y + 0.05, lab, va="center", fontsize=8.5, fontweight="bold")
            ax2.text(0.02 * extent, y - 0.17, note, va="center", fontsize=7.5, color="#444")
        else:
            ax2.text(L + extent * 0.02, y + 0.05, lab, va="center", fontsize=8.5, fontweight="bold")
            ax2.text(L + extent * 0.02, y - 0.17, note, va="center", fontsize=7.5, color="#444")
    ax2.set_yticks([])
    ax2.set_xlim(0, extent * 1.02)
    ax2.set_xticks(np.arange(0, extent, 1.0))
    ax2.set_ylim(-0.6, 2.9)
    ax2.set_xlabel("độ dài (đơn vị cảnh)")
    ax2.set_title("Thước tỉ lệ  0.01·extent : 0.1·extent : extent\n= 1 : 10 : 100")
    ax2.text(0.02 * extent, 2.75,
             "Cùng δ = 0.01 nhưng ngưỡng tuyệt đối δ·extent\nthay đổi theo kích thước cảnh (extent).",
             fontsize=8, va="top", color="#333")
    ax2.grid(axis="x", alpha=0.3)
    save(fig, "fig_06_scene_extent.png")
    return extent, s_small, s_big


# ======================================================================
# HÌNH 7 — sensitivity of thresholds
# ======================================================================
def fig_07():
    rng = np.random.default_rng(SEED + 2)
    N = 10000
    # ||g_bar||: log-normal, median ~4e-5, đuôi dài (hầu hết < tau_grad)
    g = np.exp(rng.normal(np.log(4e-5), 1.0, N))
    # max(s)/extent: log-normal, median ~0.004, dùng cho đường chia clone/split
    s_rel = np.exp(rng.normal(np.log(0.004), 0.9, N))

    sel = g >= TAU_GRAD
    n_over = int(sel.sum())
    frac_over = n_over / N * 100

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 4.3),
                                       gridspec_kw={"width_ratios": [1.35, 1, 1]})
    # --- (a) histogram
    bins = np.logspace(-6, -2, 70)
    counts, edges = np.histogram(g, bins=bins)
    ax.set_xscale("log")
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color="#9e9e9e", edgecolor="white", lw=0.3, label="Tất cả Gaussian")
    over = edges[:-1] >= TAU_GRAD
    ax.bar(edges[:-1][over], counts[over], width=np.diff(edges)[over], align="edge",
           color="#c0392b", edgecolor="white", lw=0.3, label=r"Vượt $\tau_{grad}$ (được densify)")
    ax.axvspan(TAU_GRAD, 1e-2, color="#c0392b", alpha=0.07)
    ax.axvline(TAU_GRAD, color=C_TAU, lw=1.8)
    ax.text(TAU_GRAD * 1.15, counts.max() * 1.22, r"$\tau_{grad}=2\cdot10^{-4}$",
            color=C_TAU, fontsize=9)
    ax.text(TAU_GRAD * 1.15, counts.max() * 1.02,
            f"{frac_over:.1f}% vượt ngưỡng\n({n_over}/{N} Gaussian)",
            color=C_TAU, fontsize=9)
    ax.set_ylim(0, counts.max() * 1.38)
    ax.set_xlim(1e-6, 1e-2)
    ax.set_xlabel(r"$||\bar{g}_i||$ (thang log)")
    ax.set_ylabel("số Gaussian")
    ax.set_title(f"(a) Phân bố mô phỏng $||\\bar{{g}}_i||$, N = {N}\n(log-normal, đuôi dài)")
    ax.legend(loc="upper left")

    # --- (b) quét tau_grad
    taus = np.logspace(np.log10(5e-5), np.log10(1e-3), 80)
    pct_densify = np.array([(g >= t).mean() * 100 for t in taus])
    pct_def = frac_over
    ax2.plot(taus, pct_densify, color=C_TAU, lw=2)
    ax2.set_xscale("log")
    ax2.axvline(TAU_GRAD, color=C_TAU, lw=1, ls="--", alpha=0.6)
    ax2.axhline(pct_def, color=C_TAU, lw=1, ls="--", alpha=0.6)
    ax2.scatter([TAU_GRAD], [pct_def], s=80, color=C_TAU, edgecolor="black", zorder=6)
    ax2.annotate(f"mặc định $\\tau_{{grad}}$ = 2e-4\n→ {pct_def:.1f}% được densify",
                 (TAU_GRAD, pct_def), xytext=(14, 22), textcoords="offset points",
                 fontsize=8.5, color=C_TAU,
                 arrowprops=dict(arrowstyle="->", color=C_TAU))
    for t in (1e-4, 5e-4):
        v = (g >= t).mean() * 100
        ax2.scatter([t], [v], s=30, color="white", edgecolor=C_TAU, zorder=6)
        ax2.annotate(f"{v:.1f}%", (t, v), xytext=(6, 4), textcoords="offset points",
                     fontsize=8, color=C_TAU)
    ax2.set_xlim(5e-5, 1e-3)
    ax2.set_ylim(0, pct_densify.max() * 1.25)
    ax2.set_xlabel(r"$\tau_{grad}$ (thang log)")
    ax2.set_ylabel("% Gaussian được densify (clone + split)")
    ax2.set_title(r"(b) Quét $\tau_{grad}$ (giữ $\delta$ = 0.01)" + "\nngưỡng nhỏ hơn → densify nhiều hơn")
    ax2.grid(alpha=0.3, which="both")

    # --- (c) quét delta
    deltas = np.logspace(np.log10(0.002), np.log10(0.05), 80)
    pct_clone = np.array([(s_rel[sel] <= d).mean() * 100 for d in deltas])
    clone_def = (s_rel[sel] <= DELTA).mean() * 100
    ax3.plot(deltas, pct_clone, color=C_DELTA, lw=2, label="% clone (max(s) ≤ δ·extent)")
    ax3.plot(deltas, 100 - pct_clone, color="#c1440e", lw=2, ls="--", label="% split (max(s) > δ·extent)")
    ax3.set_xscale("log")
    ax3.axvline(DELTA, color="#444", lw=1, ls="--", alpha=0.6)
    ax3.scatter([DELTA], [clone_def], s=80, color=C_DELTA, edgecolor="black", zorder=6)
    ax3.scatter([DELTA], [100 - clone_def], s=80, color="#c1440e", edgecolor="black", zorder=6)
    ax3.annotate(f"mặc định δ = 0.01\n→ {clone_def:.1f}% clone / {100 - clone_def:.1f}% split",
                 (DELTA, clone_def), xytext=(0.0155, 47), textcoords="data",
                 fontsize=8.5, color=C_DELTA, ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_DELTA))
    ax3.set_xlim(0.002, 0.05)
    ax3.set_ylim(0, 105)
    ax3.set_xticks([0.002, 0.005, 0.01, 0.02, 0.05])
    ax3.set_xticklabels(["0.002", "0.005", "0.01", "0.02", "0.05"])
    ax3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax3.set_xlabel("δ = percent_dense (thang log)")
    ax3.set_ylabel("% trong số Gaussian được densify")
    ax3.set_title(f"(c) Quét δ (giữ $\\tau_{{grad}}$ = 2e-4, {n_over} Gaussian)\nδ lớn hơn → nhiều clone, ít split")
    ax3.legend(loc="lower right", fontsize=7.5)
    ax3.grid(alpha=0.3, which="both")

    save(fig, "fig_07_threshold_sweep.png")
    return frac_over, pct_def, clone_def, float(np.median(g)), n_over


if __name__ == "__main__":
    ex = fig_05()
    extent, s_small, s_big = fig_06()
    frac_over, pct_def, clone_def, med_g, n_over = fig_07()

    print("Đã lưu:")
    for p in saved:
        print("  ", p)
    print("\nSố liệu ví dụ (hình 5):")
    for lab, gx, sy, d in ex:
        print(f"  {lab}: ||g_bar||={gx:.0e}, max(s)/extent={sy:g} -> {d}")
    print(f"\nHình 6: extent ≈ {extent:.3f}, 0.01·extent ≈ {s_small:.4f}, 0.1·extent ≈ {s_big:.3f}")
    print(f"Hình 7: median ||g_bar|| ≈ {med_g:.2e}; {n_over}/10000 = {frac_over:.2f}% vượt tau_grad=2e-4; "
          f"với delta=0.01: {clone_def:.1f}% clone / {100-clone_def:.1f}% split trong số được densify")
