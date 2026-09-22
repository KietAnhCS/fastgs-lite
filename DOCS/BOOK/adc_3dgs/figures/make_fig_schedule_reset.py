"""
Hình cho phần LỊCH CHẠY ADC và RESET OPACITY (3DGS, Kerbl 2023 – tham số Inria mặc định).

Xuất 4 hình vào cùng thư mục với file này:
  fig_01_timeline.png         – timeline 0..30000 vòng (tích luỹ / densify+prune / reset / đóng băng)
  fig_16_sigmoid_inverse.png  – sigmoid, sigmoid^-1 và ánh xạ reset alpha <- min(alpha, 0.01)
  fig_17_opacity_trajectory.png – quỹ đạo alpha(t) của 4 Gaussian qua các lần reset
  fig_18_hist_reset.png       – histogram alpha của 20000 Gaussian trước / ngay sau / 500 vòng sau reset

Chỉ dùng numpy + matplotlib. Chạy: python make_fig_schedule_reset.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SEED = 2023
rng = np.random.default_rng(SEED)

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8
DPI = 170

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
saved = []

# ---------------------------------------------------------------- hằng số Inria
T_END = 30000
T_DENSIFY_UNTIL = 15000          # densify_until_iter
T_DENSIFY_FROM = 500             # densify_from_iter (điều kiện iteration > 500)
DENSIFY_INTERVAL = 100
RESET_INTERVAL = 3000
ALPHA_RESET = 0.01               # min(alpha, 0.01)
ALPHA_PRUNE = 0.005              # min_opacity
ALPHA_MAX = 0.99                 # trần sigmoid thực tế trong huấn luyện

densify_iters = np.array([t for t in range(1, T_DENSIFY_UNTIL)
                          if t > T_DENSIFY_FROM and t % DENSIFY_INTERVAL == 0])
reset_iters = np.array([t for t in range(1, T_DENSIFY_UNTIL)
                        if t % RESET_INTERVAL == 0])
assert len(densify_iters) == 144, len(densify_iters)
assert list(reset_iters) == [3000, 6000, 9000, 12000]

C_ACC = "#4C72B0"
C_DEN = "#DD8452"
C_RES = "#C44E52"
C_FRZ = "#8C8C8C"
C_OK = "#2A9D8F"
C_DEAD = "#C44E52"
C_MID = "#E9C46A"
C_NEW = "#7B61FF"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(a):
    return np.log(a / (1.0 - a))


# ============================================================ FIG 01: TIMELINE
def fig_01_timeline():
    fig = plt.figure(figsize=(9.2, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0])
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])
    axt.axis("off")

    rows = {
        "a": (3.0, "(a) Tích luỹ gradient\n(accum, denom)"),
        "b": (2.0, "(b) Densify + prune\n(mỗi 100 vòng)"),
        "c": (1.0, "(c) Reset opacity\n(mỗi 3000 vòng)"),
        "d": (0.0, "(d) N đóng băng\n(chỉ tối ưu tham số)"),
    }
    h = 0.6

    # (a) tích luỹ gradient: dải liên tục 0..15000
    y = rows["a"][0]
    ax.add_patch(Rectangle((0, y - h / 2), T_DENSIFY_UNTIL, h, color=C_ACC, alpha=0.85, lw=0))
    ax.text(T_DENSIFY_UNTIL / 2, y, "mọi vòng t < 15000", ha="center", va="center",
            color="white", fontsize=8.5, fontweight="bold")

    # (b) densify+prune: 144 vạch
    y = rows["b"][0]
    ax.vlines(densify_iters, y - h / 2, y + h / 2, color=C_DEN, lw=0.7)
    ax.text(T_DENSIFY_UNTIL + 300, y,
            "144 vạch: t = 600, 700, ..., 14900\n(iteration > 500 và iteration % 100 == 0\n và iteration < 15000)",
            ha="left", va="center", fontsize=7.5, color=C_DEN)

    # (c) reset opacity: 4 vạch đậm
    y = rows["c"][0]
    ax.vlines(reset_iters, y - h / 2, y + h / 2, color=C_RES, lw=3.0)
    for t in reset_iters:
        ax.text(t, y - h / 2 - 0.04, f"{t}", ha="center", va="top", fontsize=7.5, color=C_RES)

    # (d) sau 15000: vùng xám
    y = rows["d"][0]
    ax.add_patch(Rectangle((T_DENSIFY_UNTIL, y - h / 2), T_END - T_DENSIFY_UNTIL, h,
                           color=C_FRZ, alpha=0.55, lw=0))
    ax.text((T_DENSIFY_UNTIL + T_END) / 2, y, "N đóng băng – chỉ tối ưu θ (µ, s, q, α, SH)",
            ha="center", va="center", fontsize=8.5, color="black")

    # mốc 15000
    ax.axvline(T_DENSIFY_UNTIL, color="black", ls="--", lw=1.0)
    ax.text(T_DENSIFY_UNTIL + 150, 3.55, "densify_until_iter = 15000", fontsize=8, va="center")

    # ghi chú r2D > 20px chỉ bật từ t > 3000
    ax.annotate("prune theo r2D > 20px chỉ bật khi t > 3000",
                xy=(3000, rows["b"][0] - h / 2), xytext=(4600, rows["b"][0] - h / 2 - 0.22),
                fontsize=7.5, color="black", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlim(-300, T_END + 300)
    ax.set_ylim(-0.55, 3.75)
    ax.set_yticks([v[0] for v in rows.values()])
    ax.set_yticklabels([v[1] for v in rows.values()])
    ax.set_xticks(np.arange(0, T_END + 1, 3000))
    ax.set_xlabel("vòng lặp huấn luyện t")
    ax.set_title("Lịch chạy ADC (Inria mặc định): 30000 vòng huấn luyện")
    ax.grid(axis="x", ls=":", alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # bảng tổng số lần
    cells = [
        ["Sự kiện", "Điều kiện (Inria)", "Khoảng", "Số lần"],
        ["Tích luỹ gradient", "t < 15000", "0 .. 14999", "mỗi vòng"],
        ["Densify + prune", "t > 500 và t % 100 == 0 và t < 15000", "600 .. 14900", f"{len(densify_iters)}"],
        ["Reset opacity", "t % 3000 == 0 và t < 15000", "3000 .. 12000", f"{len(reset_iters)}"],
        ["N đóng băng", "t ≥ 15000", "15000 .. 30000", "–"],
    ]
    tb = axt.table(cellText=cells[1:], colLabels=cells[0], loc="center", cellLoc="center",
                   colWidths=[0.2, 0.42, 0.2, 0.12])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.15)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        if r == 0:
            cell.set_facecolor("#EEEEEE")
            cell.set_text_props(fontweight="bold")

    p = os.path.join(OUT_DIR, "fig_01_timeline.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    saved.append(p)


# ================================================= FIG 16: SIGMOID / INVERSE / RESET MAP
def fig_16_sigmoid_inverse():
    fig = plt.figure(figsize=(9.6, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    marks = [(0.005, "#C44E52", "α=0.005 (ngưỡng xoá)"),
             (0.01, "#DD8452", "α=0.01 (reset)"),
             (0.99, "#2A9D8F", "α=0.99 (trần)")]

    # --- panel 1: sigmoid
    x = np.linspace(-7, 7, 600)
    ax1.plot(x, sigmoid(x), color="#4C72B0", lw=1.6)
    for a, c, lab in marks:
        xa = logit(a)
        ax1.plot([xa], [a], "o", color=c, ms=5)
        ax1.plot([xa, xa], [0, a], color=c, ls=":", lw=0.9)
        ax1.plot([-7, xa], [a, a], color=c, ls=":", lw=0.9)
    ax1.text(-6.8, 0.88, "x = −5.293 → α = 0.005\nx = −4.595 → α = 0.01", fontsize=7.5,
             ha="left", va="top")
    ax1.text(4.6, 0.72, "x = +4.595\n→ α = 0.99", fontsize=7.5, ha="center", va="top")
    ax1.set_xlabel("x (tham số thô, được tối ưu)")
    ax1.set_ylabel("α = sigmoid(x)")
    ax1.set_title("α = sigmoid(x) = 1 / (1 + e^{−x})")
    ax1.set_xlim(-7, 7)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(ls=":", alpha=0.5)

    # --- panel 2: inverse sigmoid
    a = np.linspace(0.001, 0.999, 800)
    ax2.plot(a, logit(a), color="#4C72B0", lw=1.6)
    for aa, c, lab in marks:
        xa = logit(aa)
        ax2.plot([aa], [xa], "o", color=c, ms=5, label=f"{lab}: x = {xa:+.3f}")
        ax2.plot([aa, aa], [-7.5, xa], color=c, ls=":", lw=0.9)
        ax2.plot([0, aa], [xa, xa], color=c, ls=":", lw=0.9)
    ax2.set_xlabel("α")
    ax2.set_ylabel("x = sigmoid⁻¹(α) = ln(α / (1 − α))")
    ax2.set_title("Hàm ngược sigmoid⁻¹(α)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-7.5, 7.5)
    ax2.grid(ls=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=7)

    # --- panel 3: ánh xạ reset  alpha_sau = min(alpha_truoc, 0.01)
    x3 = np.linspace(0, 1, 1001)
    y3 = np.minimum(x3, ALPHA_RESET)
    ax3.plot(x3, y3, color=C_RES, lw=2.2, label="α_sau = min(α_trước, 0.01)")
    ax3.plot(x3, x3, color="#999999", ls="--", lw=1.0, label="y = x (không đổi)")
    ax3.axhline(ALPHA_PRUNE, color="#C44E52", ls=":", lw=1.0)
    ax3.axhline(ALPHA_RESET, color="#DD8452", ls=":", lw=1.0)
    ax3.axvline(ALPHA_RESET, color="#DD8452", ls=":", lw=1.0)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(1e-3, 1.0)
    ax3.set_ylim(1e-3, 1.0)
    ax3.set_xlabel("α trước reset (log)")
    ax3.set_ylabel("α ngay sau reset (log)")
    ax3.set_title("Reset: α ← sigmoid⁻¹(min(α, 0.01)) tại t = 3000k")
    ax3.grid(ls=":", alpha=0.5, which="both")
    ax3.annotate("α < 0.01: giữ nguyên\n(y = x)", xy=(3e-3, 3e-3), xytext=(1.3e-3, 0.06),
                 fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax3.annotate("α ≥ 0.01: tất cả về đúng 0.01\n(chỉ nhỉnh hơn ngưỡng xoá 0.005)",
                 xy=(0.3, ALPHA_RESET), xytext=(0.03, 0.25), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax3.text(0.013, ALPHA_PRUNE * 0.92, "ngưỡng xoá 0.005", fontsize=7, color="#C44E52", va="top", ha="left")
    ax3.text(0.013, ALPHA_RESET * 1.12, "mức reset 0.01", fontsize=7, color="#DD8452", va="bottom", ha="left")
    ax3.legend(loc="lower right", fontsize=7)

    p = os.path.join(OUT_DIR, "fig_16_sigmoid_inverse.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    saved.append(p)


# ============================================ FIG 17: OPACITY TRAJECTORY
def simulate_alpha(t, birth, target, rate, resets, noise=0.0, die_after=None):
    """
    Mô phỏng logit-space: x tăng theo tốc độ `rate` về target (x_target = logit(target)),
    reset: x <- logit(min(sigmoid(x), 0.01)). Trả về alpha(t) (nan trước khi sinh / sau khi chết).
    die_after: nếu không None, sau reset ở mốc này Gaussian không hồi phục (rate âm) và bị prune
               khi alpha < 0.005 tại densify kế tiếp; trả thêm t_prune.
    """
    x = np.full(len(t), np.nan)
    xt = logit(target)
    cur = logit(0.1)  # alpha khởi tạo Inria = 0.1
    dead_mode = False
    t_prune = None
    for k, tt in enumerate(t):
        if tt < birth:
            continue
        if tt == birth:
            cur = logit(0.1)
        if tt in resets and tt > birth:
            cur = logit(min(sigmoid(cur), ALPHA_RESET))
            if die_after is not None and tt == die_after:
                dead_mode = True
        if t_prune is not None:
            break
        if dead_mode:
            cur = cur - 0.012 + noise * rng.normal()
            if sigmoid(cur) < ALPHA_PRUNE:
                # bị prune ở lần densify kế tiếp
                nxt = densify_iters[densify_iters >= tt]
                t_prune = int(nxt[0]) if len(nxt) else tt
                x[k] = sigmoid(cur)
                # giữ đường đến mốc prune
                kk = k
                while kk < len(t) and t[kk] <= t_prune:
                    cur = cur - 0.012
                    x[kk] = sigmoid(cur)
                    kk += 1
                break
        else:
            cur = cur + rate * (xt - cur) + noise * rng.normal()
            cur = min(cur, logit(ALPHA_MAX))
        x[k] = sigmoid(cur)
    return x, t_prune


def fig_17_opacity_trajectory():
    t = np.arange(0, T_DENSIFY_UNTIL + 1, 10)
    resets = set(int(r) for r in reset_iters)

    a_ok, _ = simulate_alpha(t, 0, 0.90, 0.010, resets, noise=0.02)
    a_mid, _ = simulate_alpha(t, 0, 0.40, 0.006, resets, noise=0.02)
    a_dead, t_prune = simulate_alpha(t, 0, 0.60, 0.006, resets, noise=0.02, die_after=6000)
    a_new, _ = simulate_alpha(t, 9000, 0.85, 0.010, resets, noise=0.02)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(t, a_ok, color=C_OK, lw=1.8, label="(i) hữu ích: hồi phục nhanh về ~0.9 sau mỗi reset")
    ax.plot(t, a_mid, color=C_MID, lw=1.8, label="(iii) hữu ích vừa: hồi phục chậm về ~0.4")
    ax.plot(t, a_dead, color=C_DEAD, lw=1.8, label="(ii) 'ảo'/chết: sau reset 6000 không hồi phục → bị prune")
    ax.plot(t, a_new, color=C_NEW, lw=1.8, label="(iv) sinh ra (clone/split) tại t = 9000, α₀ = 0.1")

    # X đánh dấu prune
    if t_prune is not None:
        ax.plot([t_prune], [ALPHA_PRUNE], marker="x", color=C_DEAD, ms=11, mew=2.5, ls="none")
        ax.annotate(f"bị xoá tại densify t = {t_prune}\n(α < 0.005)",
                    xy=(t_prune, ALPHA_PRUNE), xytext=(t_prune + 400, 0.0025),
                    fontsize=7.5, color=C_DEAD, arrowprops=dict(arrowstyle="->", color=C_DEAD, lw=0.8))

    for r in reset_iters:
        ax.axvline(r, color=C_RES, ls="--", lw=1.0, alpha=0.8)
        ax.text(r, 1.35, f"reset\n{r}", ha="center", va="bottom", fontsize=7.5, color=C_RES)
    ax.axhline(ALPHA_MAX, color="#2A9D8F", ls=":", lw=1.0)
    ax.axhline(ALPHA_RESET, color="#DD8452", ls=":", lw=1.0)
    ax.axhline(ALPHA_PRUNE, color="#C44E52", ls=":", lw=1.0)
    ax.text(14900, ALPHA_MAX * 1.08, "trần 0.99", ha="right", va="bottom", fontsize=7.5, color="#2A9D8F")
    ax.text(14900, ALPHA_RESET * 1.12, "mức reset 0.01", ha="right", va="bottom", fontsize=7.5, color="#DD8452")
    ax.text(14900, ALPHA_PRUNE * 0.88, "ngưỡng xoá 0.005", ha="right", va="top", fontsize=7.5, color="#C44E52")

    ax.set_yscale("log")
    ax.set_ylim(1.5e-3, 2.4)
    ax.set_xlim(0, T_DENSIFY_UNTIL)
    ax.set_xticks(np.arange(0, T_DENSIFY_UNTIL + 1, 1500))
    ax.set_xlabel("vòng lặp t")
    ax.set_ylabel("α(t) (thang log)")
    ax.set_title("Quỹ đạo opacity α(t) của 4 Gaussian qua 4 lần reset (mô phỏng minh hoạ)")
    ax.grid(ls=":", alpha=0.5, which="major")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5, framealpha=0.95)

    p = os.path.join(OUT_DIR, "fig_17_opacity_trajectory.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    saved.append(p)
    return t_prune


# ==================================================== FIG 18: HISTOGRAM RESET
def fig_18_hist_reset():
    N = 20000
    frac_useful = 0.72
    n_use = int(N * frac_useful)
    n_junk = N - n_use

    # --- trước reset t = 6000: đa số cao; nhóm vô dụng có alpha thấp-trung
    x_use = rng.normal(logit(0.80), 1.1, n_use)
    x_junk = rng.normal(logit(0.15), 1.4, n_junk)
    x_before = np.concatenate([x_use, x_junk])
    x_before = np.clip(x_before, logit(0.0012), logit(ALPHA_MAX))
    a_before = sigmoid(x_before)

    # --- ngay sau reset: min(alpha, 0.01)
    a_after = np.minimum(a_before, ALPHA_RESET)

    # --- 500 vòng sau: hữu ích hồi phục lên cao; vô dụng rơi xuống < 0.005 và bị xoá
    x_after = logit(a_after)
    x_use2 = x_after[:n_use] + rng.normal(6.5, 1.0, n_use)
    x_junk2 = x_after[n_use:] + rng.normal(-1.6, 0.7, n_junk)
    # một phần nhỏ nhóm "junk" cũng hồi phục vừa phải (không xoá)
    k_recover = int(0.12 * n_junk)
    x_junk2[:k_recover] = x_after[n_use:][:k_recover] + rng.normal(2.5, 0.8, k_recover)
    x_500 = np.concatenate([x_use2, x_junk2])
    x_500 = np.clip(x_500, logit(0.0005), logit(ALPHA_MAX))
    a_500 = sigmoid(x_500)

    # bin rộng 0.05 dex, 0.01 nằm đúng giữa một bin (tránh tách đôi cột 0.01)
    bins = 10.0 ** np.arange(-3.325, 0.026, 0.05)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=False)
    titles = ["(1) Ngay trước reset t = 6000", "(2) Ngay sau reset (t = 6000⁺)", "(3) 500 vòng sau (t = 6500)"]
    datas = [a_before, a_after, a_500]
    cols = [C_ACC, C_RES, C_OK]

    stats = {}
    for ax, ttl, d, c in zip(axes, titles, datas, cols):
        ax.hist(d, bins=bins, color=c, alpha=0.85, edgecolor="white", lw=0.3)
        ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
        ax.axvline(ALPHA_PRUNE, color="#C44E52", ls=":", lw=1.1)
        ax.axvline(ALPHA_RESET, color="#DD8452", ls=":", lw=1.1)
        ax.set_title(ttl)
        ax.set_xlabel("α (thang log)")
        ax.grid(ls=":", alpha=0.4, axis="y")
        p_lt_prune = 100 * np.mean(d < ALPHA_PRUNE)
        p_at_reset = 100 * np.mean(np.isclose(d, ALPHA_RESET))
        p_hi = 100 * np.mean(d > 0.5)
        p_mid = 100 * np.mean((d >= ALPHA_RESET) & (d <= 0.5))
        p_low = 100 * np.mean((d >= ALPHA_PRUNE) & (d < ALPHA_RESET))
        stats[ttl] = dict(lt_prune=p_lt_prune, at_reset=p_at_reset, hi=p_hi, mid=p_mid, low=p_low)
    axes[0].set_ylabel("số Gaussian (N = 20000)")

    s = stats[titles[0]]
    axes[0].text(0.03, 0.97, f"α > 0.5: {s['hi']:.1f}%\n0.01 ≤ α ≤ 0.5: {s['mid']:.1f}%\nα < 0.005: {s['lt_prune']:.1f}%",
                 transform=axes[0].transAxes, va="top", ha="left", fontsize=7.5,
                 bbox=dict(boxstyle="round", fc="white", ec="#BBBBBB"))
    s = stats[titles[1]]
    axes[1].text(0.97, 0.97, f"α = 0.01 đúng: {s['at_reset']:.1f}%\n0.005 ≤ α < 0.01: {s['low']:.1f}%\nα < 0.005: {s['lt_prune']:.1f}%",
                 transform=axes[1].transAxes, va="top", ha="right", fontsize=7.5,
                 bbox=dict(boxstyle="round", fc="white", ec="#BBBBBB"))
    axes[1].annotate("gần như toàn bộ\nchụm tại 0.01", xy=(ALPHA_RESET, 0.35), xycoords=("data", "axes fraction"),
                     xytext=(0.06, 0.35), textcoords=("data", "axes fraction"), fontsize=7.5,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    s = stats[titles[2]]
    axes[2].text(0.03, 0.97, f"hồi phục (α > 0.5): {s['hi']:.1f}%\ntrung bình (0.01–0.5): {s['mid']:.1f}%\n0.005 ≤ α < 0.01: {s['low']:.1f}%\nα < 0.005 → bị xoá: {s['lt_prune']:.1f}%",
                 transform=axes[2].transAxes, va="top", ha="left", fontsize=7.5,
                 bbox=dict(boxstyle="round", fc="white", ec="#BBBBBB"))
    axes[2].text(ALPHA_PRUNE * 0.92, 0.42, "vô dụng\n→ prune", transform=axes[2].get_xaxis_transform(),
                 ha="right", va="center", fontsize=7.5, color="#C44E52")
    axes[2].text(0.15, 0.42, "hữu ích\n→ hồi phục", transform=axes[2].get_xaxis_transform(),
                 ha="center", va="center", fontsize=7.5, color=C_OK)
    for ax in axes:
        ax.text(ALPHA_PRUNE * 0.93, 0.62, "0.005", transform=ax.get_xaxis_transform(), ha="right", va="center",
                fontsize=7, color="#C44E52")
        ax.text(ALPHA_RESET * 1.08, 0.62, "0.01", transform=ax.get_xaxis_transform(), ha="left", va="center",
                fontsize=7, color="#DD8452")

    fig.suptitle("Phân bố opacity α của 20000 Gaussian quanh lần reset t = 6000 (mô phỏng minh hoạ)", fontsize=10)

    p = os.path.join(OUT_DIR, "fig_18_hist_reset.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    saved.append(p)
    return stats


if __name__ == "__main__":
    fig_01_timeline()
    fig_16_sigmoid_inverse()
    t_prune = fig_17_opacity_trajectory()
    stats = fig_18_hist_reset()

    print("Số lần densify+prune:", len(densify_iters), "| đầu:", densify_iters[0], "| cuối:", densify_iters[-1])
    print("Mốc reset:", list(reset_iters))
    print("sigmoid^-1(0.005) = %.3f, sigmoid^-1(0.01) = %.3f, sigmoid^-1(0.99) = %.3f"
          % (logit(0.005), logit(0.01), logit(0.99)))
    print("fig_17: Gaussian (ii) bị prune tại t =", t_prune)
    for k, v in stats.items():
        print("fig_18", k, {kk: round(vv, 1) for kk, vv in v.items()})
    print("Đã lưu:")
    for p in saved:
        print("  ", p)
