"""
Hinh minh hoa cho PHAN 4 cua chuong Adaptive Density Control (FastGS-lite):
Densify — tich luy gradient, dieu kien AND, clone vs split, luy thua N.

Moi ham fig_4_k() xuat dung 1 file PNG "4_k_ten-ngan.png" vao cung thu muc.
Chi dung numpy + matplotlib (khong torch). Cac so lieu vi du lay tu file goc
12-adaptive-density-control.md (canh do choi 4 Gaussian / 3 camera, va vi du
5 Gaussian cua 3.md).

Chay: python adc_figures/part4_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

# ---- hang so cua repo (arguments/__init__.py) --------------------------------
TAU_GRAD = 2e-4        # --grad_thresh
TAU_ABS = 1.2e-3       # --grad_abs_thresh
DENSE = 0.001          # --dense  (percent_dense)
IMP_THRESH = 5         # importance_score > 5
DENSIFY_FROM = 500
DENSIFY_UNTIL = 15000

# ---- canh do choi (12-adaptive-density-control.md, muc 12.2) -----------------
CAMS = np.array([[0.0, 0.0, -4.0], [1.5, 0.0, -4.0], [-1.5, 0.5, -4.0]])
TOY_MU = np.array([[0, 0, 0], [0.5, 0.3, 0.5], [-0.4, -0.2, 1.0], [0.3, -0.5, 0.2]], dtype=float)
TOY_S = np.array([0.8505, 0.9434, 1.1150, 0.8888])
# split seed 0 (bang trong file goc): epsilon cua 2 con
TOY_EPS1 = np.array([[0.1069, -0.1124, 0.5447], [1.2302, 0.8935, -0.6639],
                     [-2.5925, -0.2440, -1.3893], [0.3659, 0.9266, -0.1142]])
TOY_EPS2 = np.array([[0.0892, -0.4556, 0.3075], [-1.1938, -0.5880, 0.0390],
                     [-0.8165, -0.6069, -0.3527], [1.2145, -0.5912, 0.3124]])

# ---- vi du 5 Gaussian cua 3.md ----------------------------------------------
G5_NAME = ["G1", "G2", "G3", "G4", "G5"]
G5_G = np.array([0.00025, 0.00008, 0.00031, 0.00011, 0.00042])
G5_GABS = np.array([0.00040, 0.00030, 0.00055, 0.00150, 0.00070])
G5_IMP = np.array([0, 0, 6, 3, 1])
G5_BIG = np.array([False, False, False, True, False])   # max s > delta*extent


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved:", path)


def ellipse_pts(mu, cov, k=1.0, n=200):
    """Diem tren duong muc k-sigma cua Gaussian 2D."""
    t = np.linspace(0, 2 * np.pi, n)
    w, V = np.linalg.eigh(cov)
    circ = np.stack([np.cos(t), np.sin(t)])
    pts = (V @ np.diag(np.sqrt(np.maximum(w, 0))) @ circ) * k
    return mu[0] + pts[0], mu[1] + pts[1]


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# =============================================================================
# 4.1  Gradient co dau vs tri tuyet doi: triet tieu trong anh va giua cac view
# =============================================================================
def fig_4_1():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # (a) trong MOT anh: footprint doi xung qua bien vat the
    ax = axes[0]
    xs = np.linspace(-1, 1, 9)
    ys = np.linspace(-1, 1, 9)
    X, Y = np.meshgrid(xs, ys)
    w = np.exp(-(X ** 2 + Y ** 2) / 0.5)
    gx = -np.sign(X) * w * 0.25      # nua trai keo trai, nua phai keo phai
    gy = np.zeros_like(gx)
    ax.axvline(0, color="k", lw=2)
    ax.quiver(X, Y, gx, gy, color="tab:red", scale=3, width=0.006)
    circ = plt.Circle((0, 0), 1.0, fill=False, ls="--", color="tab:blue")
    ax.add_patch(circ)
    sum_signed = np.hypot(gx.sum(), gy.sum())
    sum_abs = np.hypot(np.abs(gx).sum(), np.abs(gy).sum())
    ax.set_title("(a) Trong 1 ảnh: biên vật thể\n"
                 f"||sum g_x|| = {sum_signed:.2f}   vs   sum|g_x| = {sum_abs:.2f}", fontsize=9)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")

    # (b) giua 4 view: 4 vector g^(v) huong khac nhau
    ax = axes[1]
    angles = np.deg2rad([20, 110, 200, 290])
    mags = np.array([1.0, 0.9, 1.1, 0.95])
    G = np.stack([mags * np.cos(angles), mags * np.sin(angles)], 1)
    cols = ["tab:red", "tab:orange", "tab:green", "tab:purple"]
    for k, (g, c) in enumerate(zip(G, cols)):
        ax.annotate("", xy=g, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2))
        ax.text(g[0] * 1.15, g[1] * 1.15, f"g^({k + 1})", color=c, ha="center")
    vsum = G.sum(0)
    ax.annotate("", xy=vsum, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=3))
    ax.text(vsum[0] + 0.05, vsum[1] - 0.15, "tổng vector", fontsize=8)
    ax.plot(0, 0, "ko")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.set_title("(b) 4 view: cộng VECTOR trước\n"
                 f"||sum_v g^(v)|| = {np.linalg.norm(vsum):.2f}  (gần 0)", fontsize=9)
    ax.set_xlabel("g_u"); ax.set_ylabel("g_v")

    # (c) accum = sum ||g^(v)||, so sanh 3 cach tich luy
    ax = axes[2]
    norms = np.linalg.norm(G, axis=1)
    vals = [np.linalg.norm(vsum), norms.sum(), norms.sum() / 4]
    labels = ["||sum_v g||\n(SAI: cộng vector)", "sum_v ||g||\n= accum (code)", "accum/denom\n= g_bar"]
    bars = ax.bar(labels, vals, color=["lightgray", "tab:blue", "tab:cyan"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center")
    ax.set_ylim(0, 4.6)
    ax.set_title("(c) add_densification_stats lấy norm TRƯỚC\nrồi mới cộng qua iteration", fontsize=9)
    ax.set_ylabel("độ lớn")
    fig.suptitle("Hai tầng triệt tiêu: trong ảnh (cần cột abs) và giữa view (cần norm trước khi cộng)", fontsize=10)
    save(fig, "4_1_gradient-cancellation.png")


# =============================================================================
# 4.2  Duong tich luy accum / denom / g_bar theo iteration
# =============================================================================
def fig_4_2():
    rng = np.random.default_rng(0)
    T = 100                      # 1 chu ky densification_interval
    it = np.arange(1, T + 1)
    # Gaussian A: luon huu hinh, gradient ~ 3e-4 ; B: huu hinh 40%, gradient lon 6e-4 ; C: hoi tu, gradient 5e-5
    vis = {"A (luôn thấy)": np.ones(T, bool),
           "B (thấy 40% view)": rng.random(T) < 0.4,
           "C (đã hội tụ)": np.ones(T, bool)}
    gnorm = {"A (luôn thấy)": np.abs(rng.normal(3e-4, 8e-5, T)),
             "B (thấy 40% view)": np.abs(rng.normal(6e-4, 1.5e-4, T)),
             "C (đã hội tụ)": np.abs(rng.normal(5e-5, 2e-5, T))}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, c in zip(vis, ["tab:blue", "tab:orange", "tab:green"]):
        v = vis[name]
        acc = np.cumsum(np.where(v, gnorm[name], 0.0))
        den = np.cumsum(v.astype(float))
        gbar = np.where(den > 0, acc / np.maximum(den, 1), 0.0)
        axes[0].plot(it, acc, color=c, label=name)
        axes[1].plot(it, den, color=c, label=name)
        axes[2].plot(it, gbar, color=c, label=name)
    axes[0].set_title("xyz_gradient_accum (cộng dồn ||g||)")
    axes[1].set_title("denom (số lần hữu hình)")
    axes[2].set_title("g_bar = accum / denom")
    axes[2].axhline(TAU_GRAD, color="r", ls="--", label="τ_grad = 2e-4")
    for ax in axes:
        ax.set_xlabel("iteration trong 1 chu kỳ (interval = 100)")
        ax.grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    axes[0].legend(fontsize=7)
    axes[2].annotate("B: denom nhỏ hơn\nnhưng g_bar vẫn cao\n(chia theo số lần THẤY)",
                     xy=(60, 6e-4), xytext=(25, 7.5e-4), fontsize=7,
                     arrowprops=dict(arrowstyle="->"))
    fig.suptitle("Tích luỹ trong 1 chu kỳ densify: accum tăng tuyến tính, denom chỉ đếm khi radii>0, g_bar là trung bình", fontsize=10)
    save(fig, "4_2_accum-denom-curve.png")


# =============================================================================
# 4.3  Mat phang (g_bar, Importance): 4 vung, 5 Gaussian cua 3.md
# =============================================================================
def fig_4_3():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # (a) clone plane: g_bar vs Importance
    ax = axes[0]
    xmax, ymax = 6e-4, 8
    ax.add_patch(Rectangle((TAU_GRAD, IMP_THRESH), xmax - TAU_GRAD, ymax - IMP_THRESH,
                           color="tab:green", alpha=0.25, label="grad đạt AND Imp>5 → CLONE"))
    ax.add_patch(Rectangle((TAU_GRAD, 0), xmax - TAU_GRAD, IMP_THRESH,
                           color="tab:red", alpha=0.15, label="grad đạt nhưng Imp≤5: 3DGS clone, FastGS CHẶN"))
    ax.add_patch(Rectangle((0, IMP_THRESH), TAU_GRAD, ymax - IMP_THRESH,
                           color="tab:blue", alpha=0.12, label="Imp>5 nhưng grad thấp: không"))
    ax.axvline(TAU_GRAD, color="k", ls="--", lw=1)
    ax.axhline(IMP_THRESH, color="k", ls="--", lw=1)
    for n, g, imp, big in zip(G5_NAME, G5_G, G5_IMP, G5_BIG):
        if big:
            ax.plot(g, imp, "s", color="gray", ms=9)
            ax.text(g + 1e-5, imp + 0.25, n + " (to → xét split)", fontsize=8, color="gray")
        else:
            ax.plot(g, imp, "o", color="k", ms=8)
            ax.text(g + 1e-5, imp + 0.25, n, fontsize=9)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.3, ymax)
    ax.set_xlabel("||g_bar||  (gradient có dấu, trung bình)")
    ax.set_ylabel("Importance")
    ax.set_title("(a) Nhánh CLONE (max s ≤ δ·extent)")
    ax.legend(fontsize=7, loc="upper left")
    ax.text(TAU_GRAD * 1.03, IMP_THRESH - 0.45, "τ_grad = 2e-4", fontsize=8)

    # (b) split plane: g_abs vs Importance
    ax = axes[1]
    xmax = 2e-3
    ax.add_patch(Rectangle((TAU_ABS, IMP_THRESH), xmax - TAU_ABS, ymax - IMP_THRESH,
                           color="tab:orange", alpha=0.3, label="abs đạt AND Imp>5 → SPLIT"))
    ax.add_patch(Rectangle((TAU_ABS, 0), xmax - TAU_ABS, IMP_THRESH,
                           color="tab:red", alpha=0.15, label="abs đạt nhưng Imp≤5: CHẶN"))
    ax.axvline(TAU_ABS, color="k", ls="--", lw=1)
    ax.axhline(IMP_THRESH, color="k", ls="--", lw=1)
    for n, g, imp, big in zip(G5_NAME, G5_GABS, G5_IMP, G5_BIG):
        if big:
            ax.plot(g, imp, "s", color="k", ms=9)
            ax.text(g - 3.5e-4, imp + 0.3, n + " (split bị chặn, Imp=3)", fontsize=8)
        else:
            ax.plot(g, imp, "o", color="gray", ms=7)
            ax.text(g + 2e-5, imp + 0.25, n, fontsize=8, color="gray")
    ax.set_xlim(0, xmax); ax.set_ylim(-0.3, ymax)
    ax.set_xlabel("||g_bar_abs||  (gradient trị tuyệt đối, trung bình)")
    ax.set_ylabel("Importance")
    ax.set_title("(b) Nhánh SPLIT (max s > δ·extent)")
    ax.legend(fontsize=7, loc="upper left")
    ax.text(TAU_ABS * 1.02, ymax - 0.6, "τ_abs = 1.2e-3", fontsize=8)
    fig.suptitle("Phép AND của FastGS-lite: chỉ G3 (grad đạt, Imp=6) được clone; G1, G5 bị chặn; G4 bị chặn ở nhánh split", fontsize=10)
    save(fig, "4_3_and-plane-5gaussians.png")


# =============================================================================
# 4.4  500 Gaussian mo phong tren mat phang (g_bar, max s) — mo rong ch7_clone_region
# =============================================================================
def fig_4_4():
    rng = np.random.default_rng(1)
    N = 500
    extent = 1.69025
    thr = DENSE * extent          # 0.00169
    gbar = 10 ** rng.uniform(-5, -2.3, N)
    gabs = gbar * 10 ** rng.uniform(0.3, 1.6, N)       # abs luon lon hon co dau
    maxs = 10 ** rng.uniform(-3.8, -1.5, N)
    imp = rng.integers(0, 12, N)
    small = maxs <= thr
    grad_ok = gbar >= TAU_GRAD
    abs_ok = gabs >= TAU_ABS
    imp_ok = imp > IMP_THRESH
    clone3d = small & grad_ok
    split3d = (~small) & abs_ok
    clone = clone3d & imp_ok
    split = split3d & imp_ok
    blocked = (clone3d | split3d) & ~imp_ok
    none = ~(clone3d | split3d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.axvspan(TAU_GRAD, 1e-2, ymin=0, ymax=1, color="tab:green", alpha=0.07)
    ax.scatter(gbar[none], maxs[none], s=10, color="lightgray", label=f"không densify ({none.sum()})")
    ax.scatter(gbar[clone], maxs[clone], s=18, color="tab:green", label=f"CLONE ({clone.sum()})")
    ax.scatter(gbar[split], maxs[split], s=18, color="tab:orange", marker="^", label=f"SPLIT ({split.sum()})")
    ax.scatter(gbar[blocked], maxs[blocked], s=26, color="tab:red", marker="x",
               label=f"3DGS densify, FastGS CHẶN (Imp≤5) ({blocked.sum()})")
    ax.axvline(TAU_GRAD, color="k", ls="--", lw=1); ax.text(TAU_GRAD * 1.1, 2e-2, "τ_grad", fontsize=8)
    ax.axhline(thr, color="k", ls=":", lw=1); ax.text(1.2e-5, thr * 1.2, "δ·extent = 0.00169", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("||g_bar|| (có dấu)"); ax.set_ylabel("max s_i")
    ax.set_title("(a) Mặt phẳng (g_bar, max s): dưới ngưỡng kích thước → clone, trên → split")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    cats = ["3DGS gốc\n(chỉ gradient)", "FastGS-lite\n(AND Importance>5)"]
    c3 = clone3d.sum(); s3 = split3d.sum()
    ax.bar(cats, [c3, clone.sum()], color="tab:green", label="clone")
    ax.bar(cats, [s3, split.sum()], bottom=[c3, clone.sum()], color="tab:orange", label="split")
    ax.text(0, c3 + s3 + 3, f"{c3 + s3} / {N}", ha="center")
    ax.text(1, clone.sum() + split.sum() + 3, f"{clone.sum() + split.sum()} / {N}", ha="center")
    ax.set_ylabel("số Gaussian được densify trong 1 lần gọi")
    ax.set_title("(b) Cùng 500 Gaussian: phép AND cắt bớt r_spawn")
    ax.legend()
    fig.suptitle("Mô phỏng 500 Gaussian (Importance ~ U{0..11}): phép AND chỉ giữ ~ 1/2 ứng viên gradient", fontsize=10)
    save(fig, "4_4_clone-region-500.png")


# =============================================================================
# 4.5  Scene extent tu camera (getNerfppNorm) — canh do choi
# =============================================================================
def fig_4_5():
    center = CAMS.mean(0)
    d = np.linalg.norm(CAMS - center, axis=1)
    diag = d.max()
    extent = 1.1 * diag
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.scatter(CAMS[:, 0], CAMS[:, 1], s=80, color="tab:blue", zorder=3)
    for k, c in enumerate(CAMS):
        ax.text(c[0] + 0.05, c[1] + 0.06, f"cam {k + 1} ({c[0]}, {c[1]}, {c[2]})", fontsize=8)
        ax.plot([center[0], c[0]], [center[1], c[1]], color="gray", lw=1, ls="--")
        ax.text((center[0] + c[0]) / 2, (center[1] + c[1]) / 2 - 0.12, f"{d[k]:.4f}", fontsize=7, color="gray")
    ax.plot(center[0], center[1], "r*", ms=14, label=f"c_bar = ({center[0]:.2f}, {center[1]:.4f}, {center[2]:.0f})")
    circ = plt.Circle((center[0], center[1]), diag, fill=False, color="tab:red", ls="--", label=f"diag = max ||c_v - c_bar|| = {diag:.4f}")
    circ2 = plt.Circle((center[0], center[1]), extent, fill=False, color="tab:green", lw=2, label=f"extent = 1.1 * diag = {extent:.4f}")
    ax.add_patch(circ); ax.add_patch(circ2)
    ax.scatter(TOY_MU[:, 0], TOY_MU[:, 1], marker="x", color="k", label="4 Gaussian (x, y)")
    ax.set_aspect("equal"); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.2, 2.4)
    ax.set_xlabel("x (world)"); ax.set_ylabel("y (world)")
    ax.set_title("(a) getNerfppNorm: tâm camera trung bình, bán kính lớn nhất × 1.1")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    labels = ["δ·extent\n(ngưỡng\nclone/split)", "0.1·extent\n(ngưỡng prune\n'quá to')", "min s_i", "max s_i", "extent"]
    vals = [DENSE * extent, 0.1 * extent, TOY_S.min(), TOY_S.max(), extent]
    cols = ["tab:red", "tab:purple", "tab:blue", "tab:blue", "tab:green"]
    ax.bar(labels, vals, color=cols)
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=8)
    for k, v in enumerate(vals):
        ax.text(k, v * 1.25, f"{v:.4g}", ha="center", fontsize=8)
    ax.set_ylabel("độ dài (world unit, log)")
    ax.set_title("(b) Ba mức kích thước: 0.00169 << 0.85..1.12 → cả 4 vào nhánh split")
    fig.suptitle("Scene extent chỉ phụ thuộc VỊ TRÍ CAMERA, không phụ thuộc điểm SfM hay Gaussian", fontsize=10)
    save(fig, "4_5_scene-extent.png")


# =============================================================================
# 4.6  So do quyet dinh densify (ve bang hop va mui ten)
# =============================================================================
def fig_4_6():
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.9); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec="k", fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(p, q, text="", color="k"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=14, color=color, lw=1.3))
        if text:
            ax.text((p[0] + q[0]) / 2 + 0.08, (p[1] + q[1]) / 2, text, fontsize=8, color=color)

    box(4.3, 5.3, 3.4, 0.7, "Gaussian i, mỗi 100 (500) vòng, 500 < t < 15000", fc="#eef")
    box(4.3, 4.2, 3.4, 0.7, "max s_i ≤ δ·extent ?", fc="#ffd")
    arrow((6, 5.3), (6, 4.9))
    # nhanh nho
    box(0.6, 2.9, 3.6, 0.8, "NHỎ (under-reconstruction)\n||g_bar_i|| ≥ τ_grad = 2e-4 ?", fc="#dfd")
    box(7.8, 2.9, 3.6, 0.8, "TO (over-reconstruction)\n||g_bar_abs_i|| ≥ τ_abs = 1.2e-3 ?", fc="#fed")
    arrow((4.3, 4.55), (2.4, 3.7), "có")
    arrow((7.7, 4.55), (9.6, 3.7), "không")
    box(0.6, 1.6, 3.6, 0.8, "Importance_i > 5 ?\n(mới của FastGS-lite)", fc="#dfd")
    box(7.8, 1.6, 3.6, 0.8, "Importance_i > 5 ?\n(mới của FastGS-lite)", fc="#fed")
    arrow((2.4, 2.9), (2.4, 2.4), "có")
    arrow((9.6, 2.9), (9.6, 2.4), "có")
    box(0.6, 0.2, 3.6, 0.9, "CLONE: θ_new = θ_i\n1 → 2 (bản gốc giữ nguyên)", fc="tab:green", ec="tab:green", fs=9)
    box(7.8, 0.2, 3.6, 0.9, "SPLIT: 2 con, μ + R·ε, s/1.6\nxoá gốc: 1 → 2", fc="tab:orange", ec="tab:orange", fs=9)
    arrow((2.4, 1.6), (2.4, 1.1), "có")
    arrow((9.6, 1.6), (9.6, 1.1), "có")
    box(4.6, 0.6, 2.8, 0.9, "KHÔNG densify\n(3DGS gốc: vẫn densify nếu\nchỉ gradient đạt)", fc="#eee", fs=8)
    arrow((4.2, 3.3), (5.2, 1.5), "không", color="gray")
    arrow((7.8, 3.3), (6.8, 1.5), "không", color="gray")
    arrow((4.2, 2.0), (4.6, 1.3), "không", color="tab:red")
    arrow((7.8, 2.0), (7.4, 1.3), "không", color="tab:red")
    ax.text(6, 6.5, "3 điều kiện nối bằng AND: kích thước chọn NHÁNH, gradient chọn CÓ/KHÔNG, Importance lọc thêm",
            ha="center", fontsize=8.5, style="italic", bbox=dict(fc="white", ec="none"))
    ax.set_title("Sơ đồ quyết định densify trong densify_and_prune_fastgs (gaussian_model.py:449-462)", fontsize=10)
    save(fig, "4_6_decision-diagram.png")


# =============================================================================
# 4.7  Under vs over reconstruction (1D) — vi sao nho thi clone, to thi split
# =============================================================================
def fig_4_7():
    x = np.linspace(-3, 3, 600)

    def g(x, m, s, a=1.0):
        return a * np.exp(-0.5 * ((x - m) / s) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    # under: target rong, Gaussian nho
    ax = axes[0]
    target = g(x, 0, 1.4)
    fit = g(x, 0.0, 0.45)
    ax.fill_between(x, target, color="lightgray", label="vùng cần phủ (GT)")
    ax.plot(x, fit, color="tab:green", lw=2, label="1 Gaussian NHỎ (chưa phủ hết)")
    ax.plot(x, g(x, -0.55, 0.45) + g(x, 0.55, 0.45), color="tab:green", ls="--", lw=1.5,
            label="sau clone + tối ưu: 2 bản tách ra")
    ax.set_title("(a) UNDER-reconstruction: Gaussian nhỏ hơn vùng cần phủ\n→ gradient kéo nó giãn ra → CLONE (thêm bản để phủ)")
    ax.legend(fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("mật độ / màu")
    # over: target chi tiet, Gaussian to
    ax = axes[1]
    target = g(x, -0.8, 0.3) + g(x, 0.8, 0.3, 0.8)
    fit = g(x, 0.0, 1.2, 0.75)
    ax.fill_between(x, target, color="lightgray", label="chi tiết nhỏ (GT)")
    ax.plot(x, fit, color="tab:orange", lw=2, label="1 Gaussian TO (phủ lấn cả 2 chi tiết)")
    ax.plot(x, g(x, -0.8, 0.3) + g(x, 0.8, 0.3, 0.8), color="tab:orange", ls="--", lw=1.5,
            label="sau split + tối ưu: 2 con nhỏ hơn")
    ax.set_title("(b) OVER-reconstruction: Gaussian to hơn chi tiết\n→ gradient hai nửa ngược chiều (abs lớn) → SPLIT (thay bằng 2 con nhỏ)")
    ax.legend(fontsize=7)
    ax.set_xlabel("x")
    fig.suptitle("Cùng một tín hiệu 'gradient vị trí lớn' nhưng hai bệnh khác nhau, chẩn đoán bằng KÍCH THƯỚC", fontsize=10)
    save(fig, "4_7_under-vs-over.png")


# =============================================================================
# 4.8  Split hinh hoc: ellipse goc + 2 con (2D), ellipsoid 3D
# =============================================================================
def fig_4_8():
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 2, 1)
    mu = np.array([0.0, 0.0])
    s = np.array([1.0, 0.4])
    theta = np.deg2rad(30)
    R = rot2(theta)
    cov = R @ np.diag(s ** 2) @ R.T
    ex, ey = ellipse_pts(mu, cov, 1.0)
    ax.plot(ex, ey, "k--", lw=1.5, label="gốc, 1 sigma (s = (1.0, 0.4), xoay 30°)")
    ex, ey = ellipse_pts(mu, cov, 2.0)
    ax.plot(ex, ey, "k:", lw=1, label="gốc, 2 sigma")
    # mau eps trong he cuc bo roi xoay
    eps = rng.normal(0, 1, (300, 2)) * s
    pts = (R @ eps.T).T
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color="lightblue", label="300 mẫu ε ~ N(0, diag s²), xoay R")
    # 2 con
    e2 = rng.normal(0, 1, (2, 2)) * s
    kids = (R @ e2.T).T + mu
    covk = R @ np.diag((s / 1.6) ** 2) @ R.T
    for j, k in enumerate(kids):
        kx, ky = ellipse_pts(k, covk, 1.0)
        ax.plot(kx, ky, color="tab:red", lw=2)
        ax.plot(k[0], k[1], "o", color="tab:red")
        ax.text(k[0] + 0.08, k[1] + 0.08, f"con {j + 1}", color="tab:red")
    ax.plot([], [], color="tab:red", lw=2, label="2 con: cùng R, scale s/1.6")
    ax.annotate("", xy=kids[0], xytext=mu, arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.text(0.35 * kids[0][0], 0.35 * kids[0][1] - 0.15, "R(q)·ε", color="tab:red", fontsize=8)
    ax.set_aspect("equal"); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.0, 2.0)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(a) 2D: μ_con = μ + R(q)·ε ; s_con = s / 1.6 ; xoá gốc")
    ax.set_xlabel("x (world)"); ax.set_ylabel("y (world)")

    # 3D
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    sph = np.stack([np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V)], -1)

    def draw(center, scale, R3, color, alpha):
        P = (sph * scale) @ R3.T + center
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=color, alpha=alpha, linewidth=0)

    s3 = np.array([1.0, 0.5, 0.3])
    cz, sz = np.cos(np.deg2rad(25)), np.sin(np.deg2rad(25))
    R3 = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    draw(np.zeros(3), s3, R3, "gray", 0.15)
    e3 = rng.normal(0, 1, (2, 3)) * s3
    for j in range(2):
        c = R3 @ e3[j]
        draw(c, s3 / 1.6, R3, "tab:red", 0.5)
        ax.text(c[0], c[1], c[2] + 0.35, f"con {j + 1}", color="tab:red", fontsize=8)
    ax.set_box_aspect([1, 1, 1]); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(-1.6, 1.6)
    ax.set_title("(b) 3D: ellipsoid gốc (xám) và 2 con (đỏ), cùng hướng R(q)")
    fig.suptitle("densify_and_split_fastgs: stds = s, samples = N(0, s), rots = build_rotation(q), scale / (0.8 * N)", fontsize=10)
    save(fig, "4_8_split-geometry.png")


# =============================================================================
# 4.9  Phan bo con khi lap 1000 lan + he so 1.6
# =============================================================================
def fig_4_9():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) 1000 lan split cung 1 Gaussian: phan bo tam con
    ax = axes[0]
    s = np.array([1.0, 0.4]); R = rot2(np.deg2rad(30))
    cov = R @ np.diag(s ** 2) @ R.T
    kids = (R @ (rng.normal(0, 1, (2000, 2)) * s).T).T
    ax.hexbin(kids[:, 0], kids[:, 1], gridsize=28, cmap="Blues", mincnt=1)
    for k, ls in [(1, "--"), (2, ":")]:
        ex, ey = ellipse_pts(np.zeros(2), cov, k)
        ax.plot(ex, ey, "k", ls=ls, lw=1.2, label=f"{k} sigma của gốc")
    d2 = np.einsum("ij,jk,ik->i", kids, np.linalg.inv(cov), kids)
    ax.set_title(f"(a) 2000 tâm con: {np.mean(d2 <= 1) * 100:.0f}% trong 1 sigma,\n"
                 f"{np.mean(d2 <= 4) * 100:.0f}% trong 2 sigma của gốc")
    ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_xlabel("x"); ax.set_ylabel("y")

    # (b) 1D: mat do goc vs tong 2 con voi cac uoc so khac nhau
    ax = axes[1]
    x = np.linspace(-4, 4, 800)
    sp = 1.0
    parent = np.exp(-0.5 * (x / sp) ** 2)
    ax.plot(x, parent, "k", lw=2, label="gốc: s = 1")
    off = 0.8   # |eps| dien hinh ~ 0.8 s
    for div, c in [(1.0, "tab:gray"), (1.6, "tab:red"), (2.0, "tab:blue"), (3.0, "tab:green")]:
        sk = sp / div
        mix = np.exp(-0.5 * ((x - off) / sk) ** 2) + np.exp(-0.5 * ((x + off) / sk) ** 2)
        ax.plot(x, mix, color=c, lw=1.6 if div == 1.6 else 1, ls="-" if div == 1.6 else "--",
                label=f"2 con lệch ±{off}, s/{div:g}")
    ax.set_title("(b) Ước số scale con: /1 chồng lấn, /3 để hở,\n/1.6 = 0.8·N: vẫn phủ gốc mà đã tách")
    ax.legend(fontsize=7); ax.set_xlabel("x"); ax.set_ylabel("mật độ (chưa chuẩn hoá)")

    # (c) canh do choi seed 0: 4 goc va 8 con, chieu (x, y)
    ax = axes[2]
    cols = ["tab:blue", "tab:green", "tab:purple", "tab:brown"]
    for i in range(4):
        c = plt.Circle(TOY_MU[i, :2], TOY_S[i], fill=False, ls="--", color=cols[i], lw=1.2)
        ax.add_patch(c)
        ax.plot(TOY_MU[i, 0], TOY_MU[i, 1], "x", color=cols[i])
        ax.text(TOY_MU[i, 0] + 0.05, TOY_MU[i, 1] + 0.05, f"G{i + 1}", color=cols[i], fontsize=8)
        for eps, tag in [(TOY_EPS1[i], "c1"), (TOY_EPS2[i], "c2")]:
            m = TOY_MU[i] + eps
            ck = plt.Circle(m[:2], TOY_S[i] / 1.6, fill=False, color=cols[i], lw=1.8)
            ax.add_patch(ck)
            ax.plot(m[0], m[1], "o", color=cols[i], ms=4)
            ax.text(m[0] + 0.04, m[1] - 0.12, f"G{i + 1}{tag}", color=cols[i], fontsize=6.5)
    ax.set_aspect("equal"); ax.set_xlim(-4.2, 3.2); ax.set_ylim(-2.6, 2.8)
    ax.set_title("(c) Cảnh đồ chơi, seed 0 (bảng file gốc): 4 gốc (đứt) → 8 con (liền)\nbán kính s/1.6 = 0.53..0.70, chiếu (x, y)", fontsize=8.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle("Kết quả split là NGẪU NHIÊN theo hình dạng gốc: R = I ở cảnh đồ chơi nên ε cộng thẳng vào μ", fontsize=10)
    save(fig, "4_9_split-distribution.png")


# =============================================================================
# 4.10 Clone: ban sao y nguyen, tach ra nho Adam state khac nhau
# =============================================================================
def fig_4_10():
    # 1D: target = 2 bump lech; 2 Gaussian cung vi tri x=0, cung s, cung a; toi uu vi tri bang Adam
    x = np.linspace(-3, 3, 601)
    dx = x[1] - x[0]

    def g(x, m, s=0.35, a=1.0):
        return a * np.exp(-0.5 * ((x - m) / s) ** 2)

    target = g(x, -0.7, 0.35, 0.9) + g(x, 0.9, 0.35, 1.0)

    def loss_grad(m1, m2):
        r = g(x, m1) + g(x, m2) - target
        L = np.sum(r ** 2) * dx
        d1 = np.sum(2 * r * g(x, m1) * (x - m1) / 0.35 ** 2) * dx
        d2 = np.sum(2 * r * g(x, m2) * (x - m2) / 0.35 ** 2) * dx
        return L, d1, d2

    lr, b1, b2, eps = 0.02, 0.9, 0.999, 1e-8
    # ban goc da co lich su Adam (m, v tu cac buoc truoc); ban clone co (0, 0) (cat_tensors_to_optimizer)
    m = np.array([0.0, 0.0]); v = np.array([0.0, 0.0])
    pos = np.array([0.0, 0.0])
    # mo phong lich su: goc da chay 30 buoc mot minh
    p0 = 0.0
    mo, vo = 0.0, 0.0
    for k in range(1, 31):
        r = g(x, p0) - target
        d = np.sum(2 * r * g(x, p0) * (x - p0) / 0.35 ** 2) * dx
        mo = b1 * mo + (1 - b1) * d; vo = b2 * vo + (1 - b2) * d ** 2
        p0 -= lr * (mo / (1 - b1 ** k)) / (np.sqrt(vo / (1 - b2 ** k)) + eps)
    pos[:] = p0
    m[0], v[0] = mo, vo               # goc giu Adam state
    steps_o = 30
    T = 250
    traj = np.zeros((T + 1, 2)); traj[0] = pos
    Ls = []
    for k in range(1, T + 1):
        L, d1, d2 = loss_grad(pos[0], pos[1])
        Ls.append(L)
        grad = np.array([d1, d2])
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
        t_o = steps_o + k; t_c = k     # buoc bias-correction khac nhau? (code: step giu nguyen -> dung cung t)
        # code thuc: `step` cua ca nhom giu nguyen, chi (m, v) cua ban moi bang 0
        t = steps_o + k
        mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
        pos = pos - lr * mhat / (np.sqrt(vhat) + eps)
        traj[k] = pos

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.fill_between(x, target, color="lightgray", label="GT (2 bump lệch nhau)")
    ax.plot(x, g(x, traj[0, 0]), "tab:green", lw=2, label="gốc và clone: TRÙNG NHAU tại t=0")
    ax.plot(x, g(x, traj[-1, 0]), "tab:green", ls="--", label=f"gốc sau {T} bước")
    ax.plot(x, g(x, traj[-1, 1]), "tab:red", ls="--", label=f"clone sau {T} bước")
    ax.set_title("(a) Clone: θ_new = θ_i, không dịch chuyển")
    ax.legend(fontsize=7); ax.set_xlabel("x")
    ax = axes[1]
    ax.plot(traj[:, 0], "tab:green", label="μ gốc (Adam m, v kế thừa)")
    ax.plot(traj[:, 1], "tab:red", label="μ clone (Adam m, v = 0)")
    ax.axhline(-0.7, color="gray", ls=":"); ax.axhline(0.9, color="gray", ls=":")
    ax.text(T * 0.6, -0.65, "bump 1", fontsize=8, color="gray"); ax.text(T * 0.6, 0.95, "bump 2", fontsize=8, color="gray")
    ax.set_title("(b) Cùng gradient ở bước đầu, bước Adam khác nhau → tách đối xứng")
    ax.set_xlabel("bước tối ưu sau clone"); ax.set_ylabel("vị trí μ")
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(np.abs(traj[:, 0] - traj[:, 1]), "k")
    ax.set_yscale("log")
    ax.set_title("(c) |μ_gốc − μ_clone| theo bước (log)")
    ax.set_xlabel("bước tối ưu sau clone")
    fig.suptitle("Clone không cần dịch: exp_avg / exp_avg_sq của bản mới = 0 (cat_tensors_to_optimizer) là đủ để phá đối xứng", fontsize=10)
    save(fig, "4_10_clone-symmetry-break.png")


# =============================================================================
# 4.11 Luy thua N: N_{k+1} = N_k (1 + f_clone + f_split)(1 - f_prune)
# =============================================================================
def n_densify(interval):
    its = np.arange(DENSIFY_FROM + 1, DENSIFY_UNTIL)
    return int(np.sum(its % interval == 0))


def fig_4_11():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    n = np.arange(0, 146)
    for (rs, rp), c in [((0.15, 0.05), "tab:red"), ((0.10, 0.05), "tab:orange"),
                        ((0.05, 0.05), "tab:green"), ((0.05, 0.10), "tab:blue"), ((0.02, 0.02), "gray")]:
        q = (1 + rs) * (1 - rp)
        ax.plot(n, q ** n, color=c, label=f"r_spawn={rs}, r_prune={rp}: q={q:.4f}")
    ax.axvline(28, color="k", ls=":"); ax.text(29, 2, "n=28\n(interval 500)", fontsize=8)
    ax.axvline(144, color="k", ls=":"); ax.text(118, 2, "n=144\n(interval 100)", fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("số lần densify n"); ax.set_ylabel("N / N_0 (log)")
    ax.set_title("(a) N/N_0 = q^n với q = (1+r_spawn)(1−r_prune)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    ax.annotate(f"1.0925^28 = {1.0925 ** 28:.2f}", xy=(28, 1.0925 ** 28), xytext=(40, 1.0925 ** 28 * 0.3),
                fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.annotate(f"1.0925^144 = {1.0925 ** 144:.2e}", xy=(144, 1.0925 ** 144), xytext=(70, 1.0925 ** 144),
                fontsize=8, arrowprops=dict(arrowstyle="->"))

    ax = axes[1]
    it = np.arange(0, 30001)
    for interval, ls in [(100, "-"), (500, "--")]:
        for (rs, rp), c in [((0.15, 0.05), "tab:red"), ((0.05, 0.05), "tab:green")]:
            q = (1 + rs) * (1 - rp)
            cnt = np.cumsum((it > DENSIFY_FROM) & (it < DENSIFY_UNTIL) & (it % interval == 0))
            ax.plot(it, 1e5 * q ** cnt, color=c, ls=ls,
                    label=f"interval {interval}, r_spawn={rs}, r_prune={rp}")
    ax.axvspan(DENSIFY_FROM, DENSIFY_UNTIL, color="yellow", alpha=0.08)
    ax.text(7000, 1.3e5, "cửa sổ densify 500..15000", fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("N (N_0 = 100 000, log)")
    ax.set_title("(b) N(t): interval 100 gọi 144 lần, interval 500 gọi 28 lần")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    fig.suptitle("Cùng r_spawn, r_prune: số lần gọi và hệ số nhân quyết định N cuối theo LUỸ THỪA", fontsize=10)
    save(fig, "4_11_n-growth.png")


# =============================================================================
# 4.12 Reset thong ke sau densify va max_radii2D
# =============================================================================
def fig_4_12():
    rng = np.random.default_rng(3)
    T = 400
    it = np.arange(1, T + 1)
    vis = rng.random(T) < 0.7
    gn = np.abs(rng.normal(3e-4, 1e-4, T))
    radii = rng.integers(5, 30, T)
    acc = np.zeros(T); den = np.zeros(T); mr = np.zeros(T)
    a = d = m = 0.0
    for k in range(T):
        if vis[k]:
            a += gn[k]; d += 1; m = max(m, radii[k])
        acc[k], den[k], mr[k] = a, d, m
        if it[k] % 100 == 0:          # densify -> densification_postfix reset ve 0
            a = d = m = 0.0
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(it, acc, color="tab:blue"); axes[0].set_ylabel("xyz_gradient_accum")
    axes[1].plot(it, den, color="tab:orange"); axes[1].set_ylabel("denom")
    axes[2].plot(it, mr, color="tab:purple"); axes[2].set_ylabel("max_radii2D (px)")
    axes[2].axhline(20, color="r", ls="--"); axes[2].text(5, 21, "max_screen_size = 20 (chỉ bật khi t > 3000)", fontsize=8, color="r")
    for ax in axes:
        for k in range(100, T + 1, 100):
            ax.axvline(k, color="k", ls=":", lw=1)
        ax.grid(alpha=0.3)
    axes[0].set_title("Sau mỗi lần densify (mỗi 100 vòng) densification_postfix đưa cả 3 mảng về 0 cho TOÀN BỘ N (kể cả Gaussian không được densify)")
    axes[2].set_xlabel("iteration (đường chấm đứng: densify_and_prune_fastgs)")
    save(fig, "4_12_stats-reset.png")


# =============================================================================
# 4.13 Bo cuc chi so sau split: repeat(N,1), prune_filter, padded_importance
# =============================================================================
def fig_4_13():
    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(0, 14.5); ax.set_ylim(0, 6.0); ax.axis("off")
    X0 = 4.6

    def row(y, cells, colors, label, fs=8):
        ax.text(0.1, y + 0.25, label, fontsize=8.2, va="center", ha="left", weight="bold")
        for k, (c, col) in enumerate(zip(cells, colors)):
            ax.add_patch(Rectangle((X0 + k * 0.8, y), 0.76, 0.5, fc=col, ec="k", lw=0.8))
            ax.text(X0 + k * 0.8 + 0.38, y + 0.25, c, ha="center", va="center", fontsize=fs)

    P = "#cde"; C1 = "#fdd"; C2 = "#dfd"; Z = "#eee"
    row(5.2, ["G1", "G2", "G3", "G4"], [P] * 4, "1. trước split (N=4)\n   mask = [1,1,1,1]")
    row(4.3, ["G1", "G2", "G3", "G4", "G1c1", "G2c1", "G3c1", "G4c1", "G1c2", "G2c2", "G3c2", "G4c2"],
        [P] * 4 + [C1] * 4 + [C2] * 4, "2. densification_postfix:\n   cat(gốc, con), thứ tự repeat(N,1)")
    row(3.4, ["1", "1", "1", "1", "0", "0", "0", "0", "0", "0", "0", "0"],
        [P] * 4 + [Z] * 8, "3. prune_filter =\n   cat(mask, zeros(N·sum)) → xoá gốc")
    row(2.5, ["G1c1", "G2c1", "G3c1", "G4c1", "G1c2", "G2c2", "G3c2", "G4c2"],
        [C1] * 4 + [C2] * 4, "4. sau split (N=8):\n   accum = denom = max_radii2D = 0")
    row(1.6, ["4.26", "1.82", "1e6", "1.00", "0", "0", "0", "0"],
        [C1] * 4 + [Z] * 4, "5. padded_importance[:4] =\n   1/(1e-6+1-Pruning) theo CHỈ SỐ CŨ")
    ax.text(0.1, 0.95, "Hậu quả: 4 con đầu tiên 'thừa kế' trọng số prune của G1..G4 (theo chỉ số, không theo danh tính);\n"
                       "4 con sau có trọng số 0 → miễn nhiễm trong lần này. Trong huấn luyện thật tỉ lệ split nhỏ nên lệch ít.",
            fontsize=8.5, style="italic")
    ax.text(0.1, 0.25, "Thứ tự trong densify_and_prune_fastgs: (1) g_bar, g_bar_abs, tmp_radii = radii  (2) clone  (3) split  "
                       "(4) prune_mask + multinomial  (5) opacity ← min(alpha, 0.8)  (6) tmp_radii = None",
            fontsize=8.5)
    ax.set_title("Bố cục chỉ số khi split cả 4 Gaussian của cảnh đồ chơi (gaussian_model.py:396-418, 473-476)", fontsize=10)
    save(fig, "4_13_index-layout.png")


if __name__ == "__main__":
    print("n_densify(100) =", n_densify(100), " n_densify(500) =", n_densify(500))
    fig_4_1(); fig_4_2(); fig_4_3(); fig_4_4(); fig_4_5(); fig_4_6(); fig_4_7()
    fig_4_8(); fig_4_9(); fig_4_10(); fig_4_11(); fig_4_12(); fig_4_13()
    print("done")
