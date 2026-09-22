"""Hình minh hoạ bước prune cứng trong ADC (Kerbl 2023, Inria mặc định).

    xoa_i = [alpha_i < 0.005] OR [r_i^2D > 20px, chỉ khi t > 3000] OR [max(s_i) > 0.1*extent]

Chỉ dùng numpy + matplotlib. Chạy: python make_fig_prune.py
Xuất: fig_12_prune_alpha.png, fig_13_prune_radius2d.png,
      fig_14_prune_scale.png, fig_15_prune_or_region.png
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 12345
N = 20000
ALPHA_MIN = 0.005
R2D_MAX = 20.0
SCALE_FRAC = 0.1

RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
GREY = "#7f7f7f"

saved = []


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    saved.append(path)


# ---------------------------------------------------------------------------
# Mô phỏng chung 20000 Gaussian: alpha, scale s, r2D có tương quan
# ---------------------------------------------------------------------------
def simulate(rng, n=N, extent=10.0, f=1000.0):
    """Trả về alpha, max_s, r2d (px) cho n Gaussian.

    - alpha: hỗn hợp 3 thành phần: đa số 0.3-0.99 (Beta), một cụm nhỏ gần 0
      (Gaussian còn lại sau reset opacity chưa được "hồi"), một ít trung gian.
    - max_s: log-normal, một đuôi nhỏ rất to (Gaussian "floater" phủ cả cảnh).
    - z: độ sâu log-normal; r2d = f * s / z  (gần đúng, xem fig 13).
    Tương quan: Gaussian to (s lớn) có xu hướng alpha thấp hơn (mờ, "sương").
    """
    comp = rng.choice(3, size=n, p=[0.80, 0.06, 0.14])
    alpha = np.empty(n)
    m0 = comp == 0
    alpha[m0] = 0.3 + 0.69 * rng.beta(2.0, 1.3, size=m0.sum())
    m1 = comp == 1
    alpha[m1] = np.abs(rng.normal(0.0, 0.004, size=m1.sum())) + 1e-4  # cụm gần 0
    m2 = comp == 2
    alpha[m2] = 10 ** rng.uniform(-2.3, -0.5, size=m2.sum())  # 0.005 .. 0.3
    alpha = np.clip(alpha, 1e-4, 0.99)

    # scale: log-normal quanh 0.5% extent, đuôi 3% Gaussian to bất thường
    log_s = rng.normal(np.log(0.002 * extent), 0.8, size=n)
    big = rng.random(n) < 0.03
    log_s[big] = rng.normal(np.log(0.09 * extent), 0.5, size=big.sum())
    max_s = np.exp(log_s)

    # tương quan: Gaussian to -> alpha giảm (nhân hệ số < 1)
    frac = max_s / extent
    damp = np.exp(-6.0 * np.clip(frac - 0.03, 0, None))
    alpha = np.clip(alpha * damp, 1e-4, 0.99)

    # độ sâu: log-normal quanh 2 (đơn vị cảnh), min 0.3
    z = np.exp(rng.normal(np.log(4.0), 0.9, size=n))
    z = np.clip(z, 0.3, None)
    # bán kính chiếu: trục lớn không luôn vuông góc tia nhìn -> hệ số 0.3..1
    s_proj = max_s * rng.uniform(0.3, 1.0, size=n)
    r2d = f * s_proj / z
    return alpha, max_s, r2d, z


# ---------------------------------------------------------------------------
# Hình 12 — điều kiện alpha < 0.005
# ---------------------------------------------------------------------------
def fig12():
    rng = np.random.default_rng(SEED)
    alpha, _, _, _ = simulate(rng)
    n_rm = int((alpha < ALPHA_MIN).sum())
    pct = 100.0 * n_rm / len(alpha)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- panel trái: histogram trục log ---
    bins = np.logspace(-4, 0, 70)
    ax1.hist(alpha[alpha >= ALPHA_MIN], bins=bins, color=BLUE, alpha=0.85,
             label=f"giữ lại ({len(alpha) - n_rm:,})")
    ax1.hist(alpha[alpha < ALPHA_MIN], bins=bins, color=RED, alpha=0.85,
             label=f"bị xoá ({n_rm:,}, {pct:.1f}%)")
    ax1.axvspan(1e-4, ALPHA_MIN, color=RED, alpha=0.08)
    ax1.axvline(ALPHA_MIN, color=RED, lw=1.8, ls="--")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1e-4, 1.0)
    ax1.set_ylim(0.8, 2e4)
    ax1.set_xlabel(r"độ mờ $\alpha_i$ (trục log)")
    ax1.set_ylabel("số Gaussian (trục log)")
    ax1.set_title(f"Histogram $\\alpha$ của N = {len(alpha):,} Gaussian")
    ax1.text(ALPHA_MIN * 1.15, 1.2e4,
             r"$\alpha_{\min}=0.005$", color=RED, va="top", fontsize=9)
    ax1.text(4e-4, 5e3, f"xoá\n{n_rm:,} ({pct:.1f}%)", color=RED,
             ha="center", va="center", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, frameon=False,
               bbox_to_anchor=(0.0, 0.70))
    ax1.grid(axis="y", alpha=0.25)

    # --- panel phải: 4 ellipse với alpha khác nhau ---
    ax2.set_facecolor("#f3efe4")
    # nền có hoạ tiết để thấy độ trong suốt
    for k in range(0, 12):
        ax2.axhline(k * 0.25, color="#d9d2bf", lw=0.8, zorder=0)
        ax2.axvline(k * 0.5, color="#d9d2bf", lw=0.8, zorder=0)
    alphas = [0.9, 0.3, 0.02, 0.004]
    xs = [0.75, 2.0, 3.25, 4.5]
    for a, x in zip(alphas, xs):
        base = "#1f5fbf"
        e = Ellipse((x, 1.55), 1.05, 0.7, angle=20, facecolor=base,
                    alpha=max(a, 0.0), edgecolor="none", zorder=2)
        ax2.add_patch(e)
        # viền nét đứt để thấy vị trí kể cả khi trong suốt
        e2 = Ellipse((x, 1.55), 1.05, 0.7, angle=20, facecolor="none",
                     edgecolor=GREY, lw=0.8, ls=":", zorder=3)
        ax2.add_patch(e2)
        lbl = rf"$\alpha={a}$"
        col = RED if a < ALPHA_MIN else "black"
        ax2.text(x, 0.85, lbl, ha="center", va="top", fontsize=9, color=col,
                 fontweight="bold" if a < ALPHA_MIN else "normal")
        status = "xoá" if a < ALPHA_MIN else "giữ"
        ax2.text(x, 0.62, status, ha="center", va="top", fontsize=9, color=col)
        if a < ALPHA_MIN:
            ax2.plot([x - 0.5, x + 0.5], [1.2, 1.9], color=RED, lw=2.2, zorder=4)
            ax2.plot([x - 0.5, x + 0.5], [1.9, 1.2], color=RED, lw=2.2, zorder=4)
    ax2.set_xlim(0, 5.25)
    ax2.set_ylim(0.3, 2.6)
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Cùng một Gaussian, độ mờ khác nhau trên nền")
    ax2.text(2.625, 2.42,
             r"$\alpha<0.005$ ≈ trong suốt: đóng góp màu $\alpha_i T_i c_i \approx 0$"
             "\n→ xoá không đổi ảnh, giảm N",
             ha="center", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.6))

    fig.suptitle(r"Prune điều kiện 1: $\alpha_i < 0.005$ (Gaussian gần trong suốt)",
                 fontsize=11)
    save(fig, "fig_12_prune_alpha.png")
    return n_rm, pct


# ---------------------------------------------------------------------------
# Hình 13 — điều kiện r^2D > 20 px (chỉ khi t > 3000)
# ---------------------------------------------------------------------------
def fig13():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # ---------- panel trái: sơ đồ camera pinhole ----------
    ax1.set_xlim(-0.6, 6.4)
    ax1.set_ylim(-3.6, 1.7)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(r"Chiếu Gaussian lên ảnh: $r^{2D} \approx f\,s/z$ (pixel)")

    # tâm camera
    ax1.plot(0, 0, "ko", ms=6, zorder=5)
    ax1.text(-0.05, -0.2, "tâm camera\n(pinhole)", ha="right", va="top", fontsize=8)
    # mặt phẳng ảnh tại x = f (vẽ f = 1.2 đơn vị hình)
    fpx = 1.2
    ax1.plot([fpx, fpx], [-1.1, 1.1], color=GREY, lw=1.6)
    ax1.text(fpx, 1.18, "mặt phẳng ảnh", ha="center", va="bottom", fontsize=8, color=GREY)
    ax1.annotate("", xy=(fpx, 1.0), xytext=(0, 1.0),
                 arrowprops=dict(arrowstyle="<->", color=GREY, lw=1))
    ax1.text(fpx / 2, 1.05, "f (px)", ha="center", va="bottom", fontsize=8, color=GREY)
    # trục quang
    ax1.plot([0, 6.1], [0, 0], color="k", lw=0.7, ls="--")

    # 2 Gaussian minh hoạ, cùng s, khác z
    demo = [(1.9, 0.55, RED, "s lớn, gần → r²ᴰ lớn", -1.35),
            (5.2, 0.55, GREEN, "s lớn, xa → r²ᴰ nhỏ", -1.7)]
    for zx, rs, col, lab, yz in demo:
        c = Circle((zx, 0), rs, facecolor=col, alpha=0.25, edgecolor=col, lw=1.4)
        ax1.add_patch(c)
        rproj = fpx * rs / zx
        for sgn in (+1, -1):
            ax1.plot([0, zx], [0, sgn * rs], color=col, lw=0.9, alpha=0.8)
            ax1.plot([fpx, fpx], [0, sgn * rproj], color=col, lw=3.5, alpha=0.9,
                     solid_capstyle="butt")
        # bán kính s (từ tâm lên đỉnh)
        ax1.plot([zx, zx], [0, rs], color=col, lw=1.2)
        ax1.text(zx + 0.06, rs / 2, "s", color=col, fontsize=9, va="center")
        ax1.plot(zx, 0, "o", color=col, ms=3)
        # độ sâu z: mũi tên ngang bên dưới
        ax1.annotate("", xy=(zx, yz), xytext=(0, yz),
                     arrowprops=dict(arrowstyle="<->", color=col, lw=0.9))
        ax1.plot([zx, zx], [-rs, yz], color=col, lw=0.6, ls=":")
        ax1.text(zx / 2, yz + 0.05, f"z", ha="center", va="bottom", fontsize=9, color=col)
        ax1.text(zx, rs + 0.1, lab, ha="center", va="bottom", fontsize=8, color=col)
    ax1.plot([0, 0], [0, -1.75], color=GREY, lw=0.6, ls=":")
    # nhãn r2D trên mặt phẳng ảnh
    ax1.text(fpx + 0.08, 0.42, r"$r^{2D}$", fontsize=9, color=RED, va="center")

    # hộp công thức + ví dụ
    ax1.text(-0.5, -2.0,
             r"$r^{2D} \approx \dfrac{f\cdot s}{z}$  (xấp xỉ, f tính bằng px)." "\n"
             "Inria: max_radii2D = max theo mọi view đã render (rasterizer cập nhật mỗi vòng).",
             fontsize=8.3, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.35", fc="#fbfbfb", ec=GREY, lw=0.6))
    ex = [("s = 0.01, z = 1, f = 1000 px  ⇒  r²ᴰ = 10 px  → giữ", BLUE),
          ("s = 0.03, z = 1, f = 1000 px  ⇒  r²ᴰ = 30 px  → xoá", RED),
          ("s = 0.03, z = 3, f = 1000 px  ⇒  r²ᴰ = 10 px  → giữ", BLUE)]
    for k, (t, col) in enumerate(ex):
        ax1.text(-0.3, -2.75 - 0.27 * k, t, fontsize=8.5, color=col,
                 ha="left", va="top", fontweight="bold" if col == RED else "normal")

    # ---------- panel phải: ảnh 200x150 với các đĩa ----------
    W, H = 200, 150
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    ax2.set_aspect("equal")
    ax2.set_facecolor("#f3efe4")
    ax2.set_ylabel("v (px)")
    ax2.set_title("Ảnh 200×150 px — ngưỡng $r^{2D}$ = 20 px", pad=8)
    ax1.set_anchor("N")
    ax2.set_anchor("N")
    for k in range(0, W + 1, 20):
        ax2.axvline(k, color="#ddd6c4", lw=0.5, zorder=0)
    for k in range(0, H + 1, 20):
        ax2.axhline(k, color="#ddd6c4", lw=0.5, zorder=0)
    disks = [(30, 40, 5), (85, 45, 15), (150, 40, 20), (95, 108, 35)]
    for (u, v, r) in disks:
        rm = r > R2D_MAX
        col = RED if rm else BLUE
        c = Circle((u, v), r, facecolor=col, alpha=0.35, edgecolor=col, lw=1.6, zorder=2)
        ax2.add_patch(c)
        ax2.plot([u, u + r], [v, v], color=col, lw=1.2, zorder=3)
        ax2.text(u, v - r - 4, f"r = {r} px" + (" → xoá" if rm else " → giữ"),
                 ha="center", va="bottom", fontsize=8.5, color=col,
                 fontweight="bold" if rm else "normal", zorder=4)
        if rm:
            d = r * 0.7
            ax2.plot([u - d, u + d], [v - d, v + d], color=RED, lw=2.2, zorder=4)
            ax2.plot([u - d, u + d], [v + d, v - d], color=RED, lw=2.2, zorder=4)
    ax2.set_xlabel("u (px)\n\nchỉ áp dụng khi t > 3000 (sau lần reset opacity đầu tiên);\n"
                   "t ≤ 3000: điều kiện này bị bỏ qua, đĩa to vẫn được giữ")

    fig.suptitle(r"Prune điều kiện 2: $r_i^{2D} > 20$ px (Gaussian chiếm quá nhiều pixel), chỉ khi $t>3000$",
                 fontsize=11)
    save(fig, "fig_13_prune_radius2d.png")


# ---------------------------------------------------------------------------
# Hình 14 — điều kiện max(s) > 0.1 * extent
# ---------------------------------------------------------------------------
def fig14():
    rng = np.random.default_rng(SEED + 1)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    extent = 1.0  # bán kính cảnh (đơn vị chuẩn hoá)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    ax.set_title(r"Prune điều kiện 3: $\max(s_i) > 0.1\cdot$extent  (Gaussian to quá 10% cảnh)",
                 fontsize=11)

    # vòng tròn extent
    ax.add_patch(Circle((0, 0), extent, facecolor="#f3efe4", edgecolor=GREY,
                        lw=1.6, ls="--", zorder=0))
    th = np.deg2rad(160)
    ax.annotate("", xy=(extent * np.cos(th), extent * np.sin(th)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(-0.5, 0.12, "extent\n(bán kính cảnh)", color=GREY, fontsize=8.5,
            ha="center", va="bottom", rotation=0)
    ax.plot(0, 0, "k+", ms=8, zorder=5)
    # đám điểm camera/nền nhỏ để "cảnh" có nội dung
    pts = rng.normal(0, 0.35, size=(160, 2))
    pts = pts[np.linalg.norm(pts, axis=1) < 0.95]
    ax.scatter(pts[:, 0], pts[:, 1], s=4, color="#b8b0a0", zorder=1)

    # vòng tròn ngưỡng 0.1 extent (tham chiếu)
    ax.add_patch(Circle((0, 0), 0.1 * extent, facecolor="none", edgecolor=RED,
                        lw=1.0, ls=":", zorder=2))
    ax.text(0.0, -0.13, "0.1·extent", color=RED, fontsize=7.5, ha="center", va="top")

    fracs = [0.02, 0.05, 0.10, 0.18]
    centers = [(-0.45, 0.62), (-0.55, -0.45), (0.62, -0.50), (0.50, 0.55)]
    angles = [15, -30, 40, -20]
    for fr, (cx, cy), ang in zip(fracs, centers, angles):
        smax = fr * extent
        rm = smax > SCALE_FRAC * extent
        col = RED if rm else BLUE
        # ellipse: bán trục lớn = smax (vẽ 1 sigma), nhỏ = 0.55 smax
        e = Ellipse((cx, cy), 2 * smax, 2 * 0.55 * smax, angle=ang,
                    facecolor=col, alpha=0.35, edgecolor=col, lw=1.6, zorder=3)
        ax.add_patch(e)
        # vẽ trục lớn
        dx, dy = smax * np.cos(np.deg2rad(ang)), smax * np.sin(np.deg2rad(ang))
        ax.plot([cx, cx + dx], [cy, cy + dy], color=col, lw=1.0, zorder=4)
        lab = f"max(s)/extent = {fr:.2f}"
        if rm:
            lab += "  → xoá"
            d = smax * 0.8
            ax.plot([cx - d, cx + d], [cy - d, cy + d], color=RED, lw=2.4, zorder=5)
            ax.plot([cx - d, cx + d], [cy + d, cy - d], color=RED, lw=2.4, zorder=5)
        else:
            lab += "  → giữ"
        ax.text(cx, cy - smax * 0.55 - 0.05, lab, ha="center", va="top", fontsize=8.5,
                color=col, fontweight="bold" if rm else "normal", zorder=6)

    ax.text(0, -1.2,
            "Gaussian với bán trục lớn nhất vượt 10% kích thước cảnh là vô lý:\n"
            "nó phủ lên quá nhiều vật thể khác nhau (thường là 'sương'/floater to),\n"
            "không mô tả được bề mặt nào → xoá ngay, bất kể $\\alpha$ hay gradient.\n"
            "So sánh: ngưỡng clone/split dùng $\\delta$ = 0.01·extent (nhỏ hơn 10 lần).",
            ha="center", va="top", fontsize=8.3,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.6))
    ax.set_ylim(-1.75, 1.25)
    save(fig, "fig_14_prune_scale.png")


# ---------------------------------------------------------------------------
# Hình 15 — Venn 3 vòng + bar chart theo t
# ---------------------------------------------------------------------------
def venn_counts(alpha, max_s, r2d, extent, use_r2d):
    A = alpha < ALPHA_MIN
    B = (r2d > R2D_MAX) if use_r2d else np.zeros_like(A)
    C = max_s > SCALE_FRAC * extent
    d = {
        "A": A, "B": B, "C": C,
        "only_A": A & ~B & ~C, "only_B": B & ~A & ~C, "only_C": C & ~A & ~B,
        "AB": A & B & ~C, "AC": A & C & ~B, "BC": B & C & ~A, "ABC": A & B & C,
        "OR": A | B | C,
    }
    return {k: int(v.sum()) for k, v in d.items()}


def fig15():
    rng = np.random.default_rng(SEED)
    extent = 10.0
    alpha, max_s, r2d, _ = simulate(rng, extent=extent)
    c_late = venn_counts(alpha, max_s, r2d, extent, use_r2d=True)    # t=9000
    c_early = venn_counts(alpha, max_s, r2d, extent, use_r2d=False)  # t=2000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------- Venn ----------
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_xlim(-2.4, 2.4)
    ax1.set_ylim(-2.75, 2.25)
    ax1.set_title(f"Vùng OR (t = 9000, N = {N:,}): xoá = hợp của 3 điều kiện")
    R = 1.25
    cA = (-0.72, 0.42)
    cB = (0.72, 0.42)
    cC = (0.0, -0.78)
    cols = [BLUE, ORANGE, GREEN]
    for (cx, cy), col in zip([cA, cB, cC], cols):
        ax1.add_patch(Circle((cx, cy), R, facecolor=col, alpha=0.22, edgecolor=col, lw=1.8))
    ax1.text(cA[0] - 0.55, cA[1] + 1.15, r"A: $\alpha<0.005$", color=BLUE, ha="center",
             fontsize=9.5, fontweight="bold")
    ax1.text(cB[0] + 0.55, cB[1] + 1.15, r"B: $r^{2D}>20$ px", color=ORANGE, ha="center",
             fontsize=9.5, fontweight="bold")
    ax1.text(cC[0], cC[1] - R - 0.12, r"C: $\max(s)>0.1\cdot$extent", color=GREEN,
             ha="center", va="top", fontsize=9.5, fontweight="bold")

    def put(x, y, key, size=10):
        ax1.text(x, y, f"{c_late[key]:,}", ha="center", va="center", fontsize=size,
                 fontweight="bold")

    put(-1.25, 0.72, "only_A")
    put(1.25, 0.72, "only_B")
    put(0.0, -1.35, "only_C")
    put(0.0, 0.85, "AB")
    put(-0.72, -0.35, "AC")
    put(0.72, -0.35, "BC")
    put(0.0, 0.05, "ABC")
    ax1.text(0, 2.0,
             f"tổng vùng hợp (A ∪ B ∪ C) = {c_late['OR']:,} Gaussian bị xoá "
             f"({100*c_late['OR']/N:.1f}%)",
             ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.6))
    ax1.text(0, -2.72,
             f"|A| = {c_late['A']:,}   |B| = {c_late['B']:,}   |C| = {c_late['C']:,}   "
             f"(ngoài 3 vòng: {N - c_late['OR']:,} giữ lại)",
             ha="center", va="bottom", fontsize=8.5, color="black")

    # ---------- Bar chart ----------
    labels = [r"A: $\alpha<0.005$", r"B: $r^{2D}>20$px", r"C: $\max(s)>0.1$ext", "OR tổng\n(xoá thật)"]
    early = [c_early["A"], 0, c_early["C"], c_early["OR"]]
    late = [c_late["A"], c_late["B"], c_late["C"], c_late["OR"]]
    x = np.arange(4)
    w = 0.36
    b1 = ax2.bar(x - w / 2, early, w, color="#9ecae1", edgecolor=BLUE,
                 label="t = 2000 (chưa dùng B, t ≤ 3000)")
    b2 = ax2.bar(x + w / 2, late, w, color=RED, alpha=0.8, edgecolor="#8b0000",
                 label="t = 9000 (đủ 3 điều kiện)")
    for bars, vals in ((b1, early), (b2, late)):
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + max(late) * 0.012,
                     f"{v:,}" if v > 0 else "—", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("số Gaussian thoả điều kiện")
    ax2.set_ylim(0, max(late) * 1.45)
    ax2.set_title("Đóng góp từng điều kiện tại 2 mốc (cùng quần thể mô phỏng)")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(0.5, 0.84,
             "OR tổng < A+B+C vì các điều kiện chồng lấn\n"
             "prune cứng: xoá ngay, không ứng viên, không ngân sách",
             transform=ax2.transAxes, ha="center", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.6))

    fig.suptitle(r"Prune cứng: $\mathrm{xoa}_i = [\alpha_i<0.005]\ \vee\ [r_i^{2D}>20\,\mathrm{px},\ t>3000]\ \vee\ [\max(s_i)>0.1\cdot\mathrm{extent}]$",
                 fontsize=11)
    save(fig, "fig_15_prune_or_region.png")
    return c_early, c_late


if __name__ == "__main__":
    n_rm, pct = fig12()
    fig13()
    fig14()
    c_early, c_late = fig15()
    print("Đã lưu:")
    for p in saved:
        print("  ", p)
    print(f"\n[fig12] alpha<0.005: {n_rm} / {N} = {pct:.2f}%")
    print("[fig15] t=2000 (không B):", c_early)
    print("[fig15] t=9000 (có B):   ", c_late)
