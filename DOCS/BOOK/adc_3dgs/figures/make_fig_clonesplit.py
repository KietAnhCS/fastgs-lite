"""
Hình minh hoạ Clone / Split / kế toán N trong ADC của 3DGS (Kerbl 2023).
Chạy:  python DOCS/BOOK/adc_3dgs/figures/make_fig_clonesplit.py
Xuất:  fig_08_clone_before_after.png, fig_09_split_geometry.png,
       fig_10_split_samples.png, fig_11_N_accounting.png (cùng thư mục script)
Chỉ dùng numpy + matplotlib, seed cố định.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 3
rng = np.random.default_rng(SEED)

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["savefig.dpi"] = 170

C_OLD = "#1f77b4"     # Gaussian gốc
C_NEW = "#d62728"     # con 1
C_NEW2 = "#ff7f0e"    # con 2
C_OBJ = "#444444"     # viền vật thể
C_GREY = "#999999"

SPLIT_FACTOR = 1.6
saved = []


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)


def rot2d(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def gauss_ellipse(ax, mu, s, deg, k=2.0, **kw):
    """Ellipse mức k-sigma của Gaussian 2D scale s=(sx,sy), xoay deg độ."""
    e = Ellipse(mu, width=2 * k * s[0], height=2 * k * s[1], angle=deg, **kw)
    ax.add_patch(e)
    return e


# ---------------------------------------------------------------------------
# Hình 8: Clone trước / sau
# ---------------------------------------------------------------------------
def fig08():
    s = (0.006, 0.003)
    deg = 30.0
    alpha = 0.6
    mu = np.array([0.100, 0.050])

    fig = plt.figure(figsize=(10.2, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.7])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_t = fig.add_subplot(gs[0, 2])
    ax_t.axis("off")

    # đường viền vật thể: cạnh chi tiết (đường cong nhỏ + góc nhọn)
    xs = np.linspace(0.075, 0.130, 200)
    edge = 0.043 + 0.10 * (xs - 0.075) + 0.004 * np.sin((xs - 0.075) * 250)

    for ax, title in zip(axes, ["Trước clone: N = 1", "Sau clone: N = 2"]):
        ax.plot(xs, edge, color=C_OBJ, lw=1.6, zorder=1)
        ax.fill_between(xs, edge, 0.030, color="#e8e8e8", zorder=0)
        ax.text(0.129, 0.0312, "vật thể (cạnh chi tiết)", ha="right", va="bottom",
                fontsize=7.5, color=C_OBJ)
        ax.set_xlim(0.075, 0.130)
        ax.set_ylim(0.030, 0.070)
        ax.set_aspect("equal")
        ax.set_xlabel("x (đơn vị cảnh)")
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("y (đơn vị cảnh)")

    # panel trái: 1 Gaussian
    gauss_ellipse(axes[0], mu, s, deg, fc=C_OLD, ec=C_OLD, alpha=alpha, lw=1.2, zorder=3)
    axes[0].plot(*mu, "o", color="k", ms=3, zorder=4)
    axes[0].annotate(r"$\theta_i$: $\mu_i$, $s_i$=(0.006, 0.003)," "\n" r"$q_i$=30°, $\alpha_i$=0.6, SH$_i$",
                     xy=mu, xytext=(0.079, 0.063), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8), va="top")
    axes[0].text(0.079, 0.0335,
                 r"$\|\bar g_i\| \geq \tau_{grad}$ và max($s_i$) $\leq \delta\cdot$extent"
                 "\n→ Gaussian NHỎ dưới tái tạo → CLONE",
                 fontsize=7.5, va="bottom", color="#333333",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))

    # panel phải: 2 Gaussian trùng vị trí (vẽ lệch 0.0008 để nhìn thấy)
    off = np.array([0.0006, -0.0006])
    gauss_ellipse(axes[1], mu, s, deg, fc=C_OLD, ec=C_OLD, alpha=alpha, lw=1.2, zorder=3)
    gauss_ellipse(axes[1], mu + off, s, deg, fc=C_NEW, ec=C_NEW, alpha=alpha, lw=1.2,
                  ls="--", zorder=4)
    axes[1].plot(*mu, "o", color="k", ms=3, zorder=5)
    axes[1].annotate("gốc $\\theta_i$ (giữ nguyên)", xy=mu, xytext=(0.079, 0.066),
                     fontsize=8, color=C_OLD, arrowprops=dict(arrowstyle="->", lw=0.8, color=C_OLD))
    axes[1].annotate("bản sao $\\theta_{new}=\\theta_i$", xy=mu + off, xytext=(0.108, 0.066),
                     fontsize=8, color=C_NEW, arrowprops=dict(arrowstyle="->", lw=0.8, color=C_NEW))
    axes[1].text(0.079, 0.0335,
                 "Hai Gaussian TRÙNG vị trí\n(vẽ lệch một chút chỉ để nhìn thấy).\n"
                 "Tách ra nhờ gradient ở các\nvòng tối ưu sau.",
                 fontsize=7.5, va="bottom", color="#333333",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))

    # bảng tham số
    rows = [("μ", "(0.100, 0.050)", "(0.100, 0.050)"),
            ("s", "(0.006, 0.003)", "(0.006, 0.003)"),
            ("q (góc)", "30°", "30°"),
            ("α", "0.6", "0.6"),
            ("SH", "c_i", "c_i")]
    ax_t.text(0.5, 0.93, "Bảng tham số: θ_new = θ_i", ha="center", fontsize=9, transform=ax_t.transAxes)
    tbl = ax_t.table(cellText=[[a, b, c] for a, b, c in rows],
                     colLabels=["", "gốc θ_i", "mới θ_new"], colWidths=[0.22, 0.39, 0.39],
                     cellLoc="center", colLoc="center",
                     bbox=[0.0, 0.30, 1.0, 0.55], zorder=6)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    ax_t.text(0.5, 0.20, "Mọi trường sao chép y nguyên:\nμ, s, q, α, hệ số SH.\n"
              "Không đổi scale, không xoá gốc.\n\nN: 1 → 2",
              ha="center", va="top", fontsize=8, transform=ax_t.transAxes)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#f0f0f0")
        elif c == 2:
            cell.set_text_props(color=C_NEW)
    fig.suptitle("Clone: sao chép y nguyên θ, scale giữ, gốc giữ  →  N: 1 → 2", fontsize=10)
    save(fig, "fig_08_clone_before_after.png")


# ---------------------------------------------------------------------------
# Hình 9: hình học của split
# ---------------------------------------------------------------------------
def fig09():
    s = np.array([0.06, 0.02])
    deg = 40.0
    R = rot2d(deg)
    mu = np.array([0.0, 0.0])
    s_child = s / SPLIT_FACTOR

    # hai mẫu local e^(j) ~ N(0, diag(s^2)) — chọn seed để hai con nằm hai phía trục dài
    rng9 = np.random.default_rng(37)
    e = rng9.normal(size=(2, 2)) * s          # (2 mẫu, 2 chiều)
    if e[0, 0] * e[1, 0] > 0:                 # (không xảy ra với seed 37) đảm bảo 2 con ở 2 phía
        e[1, 0] *= -1
    children = mu + (R @ e.T).T               # mu^(j) = mu_i + R e^(j)

    fig = plt.figure(figsize=(10.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.75])
    ax_l = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_f = fig.add_subplot(gs[0, 2])

    # ---- panel local
    ax_l.set_title("Khung local: $e^{(j)} \\sim \\mathcal{N}(0, \\mathrm{diag}(s_i^2))$")
    smp = rng9.normal(size=(400, 2)) * s
    ax_l.scatter(smp[:, 0], smp[:, 1], s=4, color=C_GREY, alpha=0.5, label="mẫu $e$")
    gauss_ellipse(ax_l, (0, 0), s, 0, k=1, fc="none", ec=C_OLD, lw=1.2, ls="--")
    gauss_ellipse(ax_l, (0, 0), s, 0, k=2, fc="none", ec=C_OLD, lw=0.8, ls=":")
    lab_pos_l = [(0.10, 0.10), (-0.07, -0.115)]
    for j, (col, mk) in enumerate([(C_NEW, "s"), (C_NEW2, "D")]):
        ax_l.plot(e[j, 0], e[j, 1], mk, color=col, ms=7, zorder=5)
        ax_l.annotate(f"$e^{{({j+1})}}$ = ({e[j,0]:+.3f}, {e[j,1]:+.3f})", xy=e[j],
                      xytext=lab_pos_l[j], fontsize=8, color=col, ha="center",
                      arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    ax_l.annotate("", xy=(0.11, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax_l.annotate("", xy=(0, 0.11), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax_l.text(0.112, -0.006, "$x_{loc}$\n(σ=$s_x$=0.06)", fontsize=7.5, va="top")
    ax_l.text(-0.006, 0.112, "$y_{loc}$ (σ=$s_y$=0.02)", fontsize=7.5, ha="right", va="bottom")
    ax_l.set_xlim(-0.17, 0.19)
    ax_l.set_ylim(-0.17, 0.17)
    ax_l.set_aspect("equal")
    ax_l.grid(alpha=0.25, lw=0.5)
    ax_l.text(0.02, 0.02, "nét đứt: 1σ, chấm: 2σ", transform=ax_l.transAxes, fontsize=7.5)

    # ---- panel world
    ax_w.set_title("Khung thế giới: $\\mu^{(j)} = \\mu_i + R(q_i)\\,e^{(j)}$")
    # đoạn cong mà Gaussian đang gánh (cong rõ hơn để thấy 1 ellipse không ôm hết)
    tt = np.linspace(-0.13, 0.13, 200)
    curve = np.stack([tt, 2.2 * tt ** 2 - 0.012], 1) @ rot2d(deg).T
    ax_w.plot(curve[:, 0], curve[:, 1], color=C_OBJ, lw=2.0, zorder=1, label="đoạn cong (vật thể)")
    # Gaussian gốc bị xoá (nét đứt)
    gauss_ellipse(ax_w, mu, s, deg, k=2, fc=C_OLD, ec=C_OLD, alpha=0.10, lw=1.4, ls="--", zorder=2)
    gauss_ellipse(ax_w, mu, s, deg, k=1, fc="none", ec=C_OLD, lw=0.8, ls="--", zorder=2)
    ax_w.plot(*mu, "x", color=C_OLD, ms=8, mew=2, zorder=6)
    tip = mu + R[:, 0] * 0.115
    ax_w.annotate("gốc $\\theta_i$ (nét đứt, 2σ)\nbị XOÁ sau split", xy=tip,
                  xytext=(0.23, 0.19), fontsize=8, color=C_OLD, ha="right", va="top",
                  arrowprops=dict(arrowstyle="->", color=C_OLD, lw=0.8))
    # trục local xoay
    for vec, lab, dx in [(R[:, 0] * 0.09, "$R e_x$ (trục dài)", (0.10, -0.01)), (R[:, 1] * 0.06, "$R e_y$", (-0.03, 0.0))]:
        ax_w.annotate("", xy=vec, xytext=mu, arrowprops=dict(arrowstyle="->", lw=1.3, color="#555555"))
        ax_w.text(vec[0] + dx[0], vec[1] + dx[1], lab, fontsize=7.5, color="#555555", ha="center", zorder=8,
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    # hai con
    lab_pos_w = [(0.15, -0.10), (-0.12, -0.17)]
    for j, (col, mk) in enumerate([(C_NEW, "s"), (C_NEW2, "D")]):
        gauss_ellipse(ax_w, children[j], s_child, deg, k=2, fc=col, ec=col, alpha=0.35, lw=1.2, zorder=4)
        gauss_ellipse(ax_w, children[j], s_child, deg, k=1, fc="none", ec=col, lw=0.8, zorder=4)
        ax_w.plot(children[j, 0], children[j, 1], mk, color=col, ms=6, zorder=7)
        ax_w.annotate("", xy=children[j], xytext=mu,
                      arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, ls="-"), zorder=6)
        ax_w.annotate(f"$\\mu^{{({j+1})}}$ = ({children[j,0]:+.3f}, {children[j,1]:+.3f})",
                      xy=children[j], xytext=lab_pos_w[j],
                      fontsize=8, color=col, ha="center",
                      arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    ax_w.text(0.02, 0.98, "con: $s$ = (0.06/1.6, 0.02/1.6)\n      = (0.0375, 0.0125)\ngóc $q$ giữ = 40°",
              transform=ax_w.transAxes, fontsize=8, ha="left", va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))
    ax_w.set_xlim(-0.25, 0.25)
    ax_w.set_ylim(-0.21, 0.21)
    ax_w.set_aspect("equal")
    ax_w.grid(alpha=0.25, lw=0.5)
    ax_w.legend(loc="lower left", fontsize=7.5)

    # ---- panel công thức
    ax_f.axis("off")
    ax_f.set_title("Công thức split (2D)")
    txt = (
        "Ma trận xoay, góc $q$ = 40°:\n"
        "$R(q) = [\\,\\cos q,\\ -\\sin q\\,;\\ \\sin q,\\ \\cos q\\,]$\n"
        "$= [\\,%.3f,\\ %.3f\\,;\\ %.3f,\\ %.3f\\,]$\n"
        "(hàng 1 ; hàng 2)\n\n" % (R[0, 0], R[0, 1], R[1, 0], R[1, 1]) +
        "$e^{(j)} \\sim \\mathcal{N}(0, \\mathrm{diag}(s_i^2))$,  $j = 1, 2$\n"
        "$\\mu^{(j)} = \\mu_i + R(q_i)\\, e^{(j)}$\n"
        "$s^{(j)} = s_i / 1.6$,   $q^{(j)} = q_i$,\n"
        "$\\alpha^{(j)} = \\alpha_i$,   SH$^{(j)}$ = SH$_i$\n"
        "Gốc $\\theta_i$ bị xoá  →  N: 1 → 2 (+1)\n\n"
        "Ví dụ trong hình (seed cố định):\n"
        f"$e^{{(1)}}$ = ({e[0,0]:+.3f}, {e[0,1]:+.3f})\n"
        f"$\\mu^{{(1)}}$ = ({children[0,0]:+.3f}, {children[0,1]:+.3f})\n"
        f"$e^{{(2)}}$ = ({e[1,0]:+.3f}, {e[1,1]:+.3f})\n"
        f"$\\mu^{{(2)}}$ = ({children[1,0]:+.3f}, {children[1,1]:+.3f})\n"
        f"$\\|\\mu^{{(1)}}-\\mu_i\\|$ = {np.linalg.norm(children[0]-mu):.3f}, "
        f"$\\|\\mu^{{(2)}}-\\mu_i\\|$ = {np.linalg.norm(children[1]-mu):.3f}"
    )
    ax_f.text(0.0, 0.97, txt, va="top", ha="left", fontsize=8, transform=ax_f.transAxes,
              linespacing=1.55)
    fig.suptitle("Split: Gaussian TO ($\\max(s_i) > \\delta\\cdot$extent) gánh một đoạn cong → 2 con nhỏ hơn 1.6×",
                 fontsize=10)
    save(fig, "fig_09_split_geometry.png")
    return e, children


# ---------------------------------------------------------------------------
# Hình 10: Monte Carlo mẫu split
# ---------------------------------------------------------------------------
def fig10():
    s = np.array([0.06, 0.02])
    deg = 40.0
    R = rot2d(deg)
    n_pairs = 2000
    e = rng.normal(size=(n_pairs, 2, 2)) * s          # 2000 cặp, mỗi cặp 2 con, 2 chiều
    pos = (e.reshape(-1, 2) @ R.T)                    # mu^(j) - mu_i trong khung thế giới
    dist = np.linalg.norm(pos, axis=1)
    smax = s.max()
    dnorm = dist / smax
    # chiếu lên trục dài / ngắn
    proj_long = e.reshape(-1, 2)[:, 0]
    proj_short = e.reshape(-1, 2)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))

    # ---- scatter
    ax = axes[0]
    ax.scatter(pos[:, 0], pos[:, 1], s=2.5, color=C_NEW, alpha=0.35, label=f"{2*n_pairs} con (2000 cặp)")
    gauss_ellipse(ax, (0, 0), s, deg, k=1, fc="none", ec=C_OLD, lw=1.3, ls="--", label="gốc 1σ")
    gauss_ellipse(ax, (0, 0), s, deg, k=2, fc="none", ec=C_OLD, lw=0.9, ls=":", label="gốc 2σ")
    ax.plot(0, 0, "x", color="k", ms=8, mew=2)
    for vec, lab in [(R[:, 0] * 0.15, "trục dài"), (R[:, 1] * 0.08, "trục ngắn")]:
        ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"))
        ax.text(vec[0] * 1.12, vec[1] * 1.12, lab, fontsize=7.5, ha="center", color="#333333")
    ax.set_aspect("equal")
    ax.set_xlim(-0.22, 0.22)
    ax.set_ylim(-0.22, 0.22)
    ax.set_xlabel("$\\mu^{(j)}_x - \\mu_{i,x}$")
    ax.set_ylabel("$\\mu^{(j)}_y - \\mu_{i,y}$")
    ax.set_title("Vị trí con quanh gốc (mật độ theo trục dài)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.text(0.98, 0.02,
            f"σ theo trục dài: {proj_long.std():.4f} (lý thuyết 0.06)\n"
            f"σ theo trục ngắn: {proj_short.std():.4f} (lý thuyết 0.02)",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))
    ax.grid(alpha=0.25, lw=0.5)

    # ---- histogram khoảng cách
    ax = axes[1]
    ax.hist(dnorm, bins=50, color=C_NEW, alpha=0.75, edgecolor="white", lw=0.4)
    med = np.median(dnorm)
    mean = dnorm.mean()
    p90 = np.percentile(dnorm, 90)
    ax.axvline(med, color="k", lw=1.0, ls="--")
    ax.axvline(1.0, color=C_OLD, lw=1.0, ls=":")
    ax.text(med - 0.04, ax.get_ylim()[1] * 0.95, f"trung vị = {med:.2f}", fontsize=8, ha="right")
    ax.text(1.04, ax.get_ylim()[1] * 0.86, "1·max(s)", fontsize=8, color=C_OLD)
    ax.set_xlim(0, 3.6)
    ax.set_xlabel("$\\|\\mu^{(j)} - \\mu_i\\| \\,/\\, \\max(s_i)$")
    ax.set_ylabel("số con")
    ax.set_title("Khoảng cách con – gốc (chuẩn hoá theo max($s_i$)=0.06)")
    ax.text(0.98, 0.55,
            f"trung bình = {mean:.2f}\n90% con nằm trong {p90:.2f}·max(s)\n"
            f"tỉ lệ > 2·max(s): {(dnorm > 2).mean()*100:.1f}%",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))
    ax.grid(alpha=0.25, lw=0.5, axis="y")

    # ---- bar chart diện tích
    ax = axes[2]
    a_child = 1 / SPLIT_FACTOR ** 2
    labels = ["gốc\n$s_i$", "1 con\n$s_i/1.6$", "2 con\ncộng lại"]
    vals = [1.0, a_child, 2 * a_child]
    cols = [C_OLD, C_NEW, C_NEW2]
    bars = ax.bar(labels, vals, color=cols, width=0.6, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.axhline(1.0, color=C_GREY, lw=0.8, ls="--")
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("diện tích ellipse (tương đối, gốc = 1)")
    ax.set_title("Hình dạng con: diện tích ∝ $s_x s_y$")
    ax.text(0.5, 0.97,
            "1 con: $1/1.6^2$ = 0.391 gốc\n2 con: $2/1.6^2$ = 0.781 gốc\n"
            "(mỗi trục co 1.6×, thể tích 3D co $1.6^3$ = 4.1×)",
            transform=ax.transAxes, fontsize=7.5, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))
    ax.grid(alpha=0.25, lw=0.5, axis="y")

    fig.suptitle(f"Monte Carlo split: {n_pairs} cặp con từ cùng Gaussian gốc $s_i$=(0.06, 0.02), $q_i$=40° (seed={SEED})",
                 fontsize=10)
    save(fig, "fig_10_split_samples.png")
    return dict(std_long=proj_long.std(), std_short=proj_short.std(), med=med, mean=mean,
                p90=p90, frac_gt2=(dnorm > 2).mean())


# ---------------------------------------------------------------------------
# Hình 11: kế toán N
# ---------------------------------------------------------------------------
def fig11():
    N_old, n_clone, n_split, n_prune = 1000, 120, 80, 45
    steps = [("N cũ", N_old), ("+ clone\n(+n_clone)", n_clone),
             ("+ con split\n(+2·n_split)", 2 * n_split), ("− gốc split\n(−n_split)", -n_split),
             ("− prune\n(−n_prune)", -n_prune)]
    N_new = N_old + n_clone + n_split - n_prune

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))

    # ---- waterfall
    run = 0
    xs = np.arange(len(steps) + 1)
    for k, (lab, v) in enumerate(steps):
        if k == 0:
            ax.bar(k, v, color=C_OLD, width=0.6)
            ax.text(k, v + 12, f"{v}", ha="center", fontsize=8.5)
            run = v
        else:
            col = "#2ca02c" if v > 0 else "#d62728"
            bottom = run if v > 0 else run + v
            ax.bar(k, abs(v), bottom=bottom, color=col, width=0.6)
            ax.plot([k - 1 + 0.3, k - 0.3], [run, run], color=C_GREY, lw=0.8, ls=":")
            ax.text(k, run + max(v, 0) + 12, f"{v:+d}", ha="center", fontsize=8.5, color=col)
            run += v
    ax.plot([len(steps) - 1 + 0.3, len(steps) - 0.3], [run, run], color=C_GREY, lw=0.8, ls=":")
    ax.bar(len(steps), N_new, color=C_OLD, width=0.6)
    ax.text(len(steps), N_new + 12, f"{N_new}", ha="center", fontsize=8.5, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in steps] + ["N mới"], fontsize=8)
    ax.set_ylim(900, 1330)
    ax.set_ylabel("số Gaussian N")
    ax.set_title("Kế toán N trong MỘT bước densify + prune")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.text(0.52, 0.04,
            "$N_{new} = N + n_{clone} + n_{split} - n_{prune}$\n"
            f"= {N_old} + {n_clone} + {n_split} − {n_prune} = {N_new}\n"
            "(split: +2 con, −1 gốc → ròng +1 mỗi Gaussian tách)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))

    # ---- tăng trưởng N(t)
    n_steps = 144
    r = 0.015
    N0 = 100_000
    t = np.arange(n_steps + 1)
    N_exp = N0 * (1 + r) ** t
    N_lin = N0 * (1 + r * t)
    ax2.semilogy(t, N_exp, color=C_NEW, lw=1.8, label=f"luỹ thừa $N_0(1+r)^t$, r = 1.5%")
    ax2.semilogy(t, N_lin, color=C_OLD, lw=1.8, ls="--", label="tuyến tính $N_0(1+rt)$")
    ax2.axhline(N0, color=C_GREY, lw=0.8, ls=":")
    ax2.set_xlabel("bước densify t (mỗi 100 vòng, 500 < iter < 15000)")
    ax2.set_ylabel("N (log)")
    ax2.set_title("N(t) sau 144 bước densify: luỹ thừa vs tuyến tính")
    ax2.set_xlim(0, n_steps)
    ax2.set_ylim(8e4, 1.5e6)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3, lw=0.5, which="both")
    ax2.annotate(f"t=144: {N_exp[-1]/1e3:,.0f}k  (×{N_exp[-1]/N0:.2f})", xy=(n_steps, N_exp[-1]),
                 xytext=(88, 1.05e6), fontsize=8, color=C_NEW,
                 arrowprops=dict(arrowstyle="->", color=C_NEW, lw=0.8))
    ax2.annotate(f"t=144: {N_lin[-1]/1e3:,.0f}k  (×{N_lin[-1]/N0:.2f})", xy=(n_steps, N_lin[-1]),
                 xytext=(60, 1.55e5), fontsize=8, color=C_OLD,
                 arrowprops=dict(arrowstyle="->", color=C_OLD, lw=0.8))
    ax2.text(2, 8.6e4, f"$N_0$ = {N0//1000}k", fontsize=8, color=C_GREY)
    for tm in (72,):
        ax2.plot(tm, N_exp[tm], "o", color=C_NEW, ms=4)
        ax2.text(tm + 4, N_exp[tm] * 0.82, f"t={tm}: {N_exp[tm]/1e3:.0f}k", fontsize=7.5, color=C_NEW)

    fig.suptitle("Mỗi bước: clone +1, split ròng +1, prune −1  →  N tăng theo cấp số nhân nếu tỉ lệ densify không đổi",
                 fontsize=10)
    save(fig, "fig_11_N_accounting.png")
    return dict(N_new=N_new, N_exp_end=N_exp[-1], N_lin_end=N_lin[-1], N_exp_72=N_exp[72])


if __name__ == "__main__":
    fig08()
    e, ch = fig09()
    st = fig10()
    acc = fig11()
    print("Đã lưu:")
    for p in saved:
        print("  ", p)
    print("\nSố liệu ví dụ:")
    print("  fig09 e^(1), e^(2):", np.round(e, 4).tolist())
    print("  fig09 mu^(1), mu^(2):", np.round(ch, 4).tolist())
    print("  fig10:", {k: round(float(v), 4) for k, v in st.items()})
    print("  fig11:", {k: round(float(v), 1) for k, v in acc.items()})
