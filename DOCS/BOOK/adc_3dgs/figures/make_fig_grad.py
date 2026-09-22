"""
Sinh 3 hinh minh hoa cho phan "tich luy gradient" cua ADC (Kerbl 2023):
  fig_02_accum_denom.png   - accum_i / denom_i / g_bar_i cho 1 Gaussian qua 100 vong
  fig_03_grad_cancel.png   - gradient pixel nguoc chieu triet tieu trong kernel
  fig_04_gbar_history.png  - g_bar_i(t) cua 5 Gaussian qua 15000 vong, reset moi 100 vong

Chay:  python DOCS/BOOK/adc_3dgs/figures/make_fig_grad.py
Chi dung numpy + matplotlib, seed co dinh.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 2023
TAU_GRAD = 2e-4

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8

# Bang mau trung tinh, phan biet duoc trong ca nen sang/toi
C_BLUE = "#3B6FD1"
C_ORANGE = "#E07A2F"
C_GREEN = "#2F9E63"
C_RED = "#C7434B"
C_PURPLE = "#8A5BC7"
C_GRAY = "#7A7A7A"

saved = []


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    saved.append(path)


# ---------------------------------------------------------------------------
# Hinh 2: accum_i, denom_i, g_bar_i cho 1 Gaussian qua 100 vong
# ---------------------------------------------------------------------------
def fig02():
    rng = np.random.default_rng(SEED)
    T = 100
    t = np.arange(1, T + 1)

    # 1_i(t): Gaussian co duoc chieu vao khung hinh (visible) hay khong
    p_vis = 0.72
    vis = (rng.random(T) < p_vis).astype(int)
    # ep 1 doan bi che lien tiep cho de thay
    vis[40:52] = 0

    # ||sum_x dL/dmu'_i|| moi vong, chi co y nghia khi 1_i(t)=1.
    # Gia tri co 1e-4..6e-4, tang dan trong 100 vong dau (Gaussian dang o vi tri sai)
    base = 1.6e-4 + 2.0e-4 * (1 - np.exp(-t / 35.0))
    noise = np.exp(rng.normal(0.0, 0.32, T))
    g_norm = np.clip(base * noise, 1e-4, 6e-4)
    g_norm_masked = g_norm * vis  # vong khong nhin thay -> khong cong

    accum = np.cumsum(g_norm_masked)
    denom = np.cumsum(vis)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_bar = np.where(denom > 0, accum / np.maximum(denom, 1), 0.0)

    above = np.where((g_bar >= TAU_GRAD) & (denom > 0))[0]
    t_cross = int(t[above[0]]) if above.size else None

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)

    # (a) chuoi 1_i(t) va gradient moi vong
    ax = axes[0]
    ax.bar(t, g_norm_masked * 1e4, width=0.9, color=C_BLUE, alpha=0.85,
           label=r"$\|\sum_x \partial L/\partial\mu'_i\|$ (vòng nhìn thấy)")
    ax.bar(t[vis == 0], np.full((vis == 0).sum(), 6.3), width=0.9,
           color=C_GRAY, alpha=0.18, label=r"$\mathbb{1}_i(t)=0$ (không nhìn thấy)")
    ax.set_ylim(0, 6.6)
    ax.set_ylabel(r"gradient mỗi vòng  ($\times 10^{-4}$)")
    ax.set_title(r"(a) Chuỗi $\mathbb{1}_i(t)$ và chuẩn gradient vị trí 2D mỗi vòng của 1 Gaussian")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax2 = ax.twinx()
    ax2.step(t, vis, where="mid", color=C_RED, lw=1.0, alpha=0.8)
    ax2.set_ylim(-0.1, 1.1 * 6.6)  # ep duong 0/1 nam sat day
    ax2.set_yticks([0, 1])
    ax2.set_ylabel(r"$\mathbb{1}_i(t)$", color=C_RED)
    ax2.tick_params(axis="y", colors=C_RED)

    # (b) accum va denom tich luy
    ax = axes[1]
    ax.plot(t, accum * 1e4, color=C_BLUE, lw=1.8,
            label=r"$\mathrm{accum}_i(t)=\sum_{t'\leq t}\mathbb{1}_i(t')\,\|\sum_x \partial L/\partial\mu'_i\|$")
    ax.set_ylabel(r"accum$_i$  ($\times 10^{-4}$)", color=C_BLUE)
    ax.tick_params(axis="y", colors=C_BLUE)
    ax3 = ax.twinx()
    ax3.plot(t, denom, color=C_ORANGE, lw=1.8,
             label=r"$\mathrm{denom}_i(t)=\sum_{t'\leq t}\mathbb{1}_i(t')$")
    ax3.set_ylabel(r"denom$_i$ (số vòng nhìn thấy)", color=C_ORANGE)
    ax3.tick_params(axis="y", colors=C_ORANGE)
    ax.axvspan(40.5, 52.5, color=C_GRAY, alpha=0.15)
    ax.annotate("bị che 12 vòng:\ncả 2 đường đi ngang", xy=(46, accum[45] * 1e4),
                xytext=(56, accum[30] * 1e4), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    ax.set_title("(b) Tích luỹ tử số accum$_i$ và mẫu số denom$_i$ (chỉ cộng khi nhìn thấy)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)

    # (c) g_bar = accum/denom
    ax = axes[2]
    ax.plot(t, g_bar * 1e4, color=C_GREEN, lw=1.8,
            label=r"$\bar g_i(t)=\mathrm{accum}_i/\mathrm{denom}_i$")
    ax.axhline(TAU_GRAD * 1e4, color=C_RED, ls="--", lw=1.2,
               label=r"$\tau_{grad}=2\times10^{-4}$")
    ax.fill_between(t, TAU_GRAD * 1e4, g_bar * 1e4, where=g_bar >= TAU_GRAD,
                    color=C_GREEN, alpha=0.15)
    if t_cross is not None:
        ax.axvline(t_cross, color=C_GREEN, ls=":", lw=1.0)
        ax.annotate(f"vượt ngưỡng lần đầu tại t={t_cross}\n"
                    f"$\\bar g_i$={g_bar[t_cross-1]*1e4:.2f}e-4, denom={denom[t_cross-1]}",
                    xy=(t_cross, g_bar[t_cross - 1] * 1e4),
                    xytext=(t_cross + 8, 1.05), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    ax.annotate(f"t=100: $\\bar g_i$={g_bar[-1]*1e4:.2f}e-4 ≥ $\\tau_{{grad}}$\n"
                f"→ ứng viên densify (accum={accum[-1]*1e4:.1f}e-4, denom={denom[-1]})",
                xy=(100, g_bar[-1] * 1e4), xytext=(58, 4.2), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    ax.set_ylim(0, 5.0)
    ax.set_ylabel(r"$\bar g_i$  ($\times 10^{-4}$)")
    ax.set_xlabel("vòng lặp t (trong 1 chu kỳ 100 vòng giữa 2 lần densify)")
    ax.set_title(r"(c) Gradient trung bình $\bar g_i(t)$ hội tụ và so với ngưỡng $\tau_{grad}$")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 101)

    save(fig, "fig_02_accum_denom.png")
    return dict(t_cross=t_cross, gbar_end=g_bar[-1], accum_end=accum[-1],
                denom_end=int(denom[-1]), n_vis=int(vis.sum()),
                gmin=g_norm[vis == 1].min(), gmax=g_norm[vis == 1].max())


# ---------------------------------------------------------------------------
# Hinh 3: gradient pixel nguoc chieu triet tieu trong kernel
# ---------------------------------------------------------------------------
def pixel_grads(mu, sig, mu_gt, sig_gt, n=9, alpha=0.8):
    """dL/dmu' theo tung pixel cho 1 splat 2D dang cap (isotropic), L = 1/2 sum (I - I_gt)^2.
    I(x) = alpha*exp(-|x-mu|^2/(2 sig^2)); dI/dmu = I(x)*(x-mu)/sig^2."""
    xs = np.arange(n) - (n - 1) / 2.0
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X, Y], -1)
    d = P - mu
    d_gt = P - mu_gt
    I = alpha * np.exp(-(d ** 2).sum(-1) / (2 * sig ** 2))
    I_gt = alpha * np.exp(-(d_gt ** 2).sum(-1) / (2 * sig_gt ** 2))
    resid = I - I_gt
    g = (resid * I / sig ** 2)[..., None] * d  # dL/dmu tai moi pixel, shape (n,n,2)
    return X, Y, I, I_gt, g


def fig03():
    n = 9
    cases = [
        dict(title="(a) Gaussian lệch vị trí (đúng kích thước)",
             mu=np.array([1.2, 0.6]), sig=1.6, mu_gt=np.array([0.0, 0.0]), sig_gt=1.6),
        dict(title="(b) Gaussian đúng vị trí nhưng quá to",
             mu=np.array([0.0, 0.0]), sig=2.3, mu_gt=np.array([0.0, 0.0]), sig_gt=1.4),
    ]
    stats = []
    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.9])

    for k, c in enumerate(cases):
        X, Y, I, I_gt, g = pixel_grads(c["mu"], c["sig"], c["mu_gt"], c["sig_gt"], n)
        g_sum = g.reshape(-1, 2).sum(0)
        abs_sum = np.linalg.norm(g.reshape(-1, 2), axis=1).sum()
        stats.append((np.linalg.norm(g_sum), abs_sum))

        ax = fig.add_subplot(gs[0, k])
        resid = I - I_gt
        vmax = np.abs(resid).max()
        ax.imshow(resid, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  extent=[-n / 2, n / 2, -n / 2, n / 2], origin="lower", alpha=0.75)
        # duong dong muc Gaussian hien tai (do) va GT (den)
        ax.contour(X, Y, I, levels=[0.3], colors=[C_RED], linewidths=1.3)
        ax.contour(X, Y, I_gt, levels=[0.3], colors=["k"], linewidths=1.3, linestyles="--")
        scale = np.abs(g).max() * 1.7
        ax.quiver(X, Y, g[..., 0], g[..., 1], angles="xy", scale_units="xy", scale=scale,
                  width=0.006, color="#1c1c1c")
        # mui ten tong
        gs_n = g_sum / scale
        ax.annotate("", xy=(c["mu"][0] + gs_n[0] * 0.25, c["mu"][1] + gs_n[1] * 0.25),
                    xytext=tuple(c["mu"]),
                    arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2.6, mutation_scale=16))
        ax.plot(*c["mu"], "o", color=C_RED, ms=5)
        ax.set_xticks(np.arange(-4, 5)); ax.set_yticks(np.arange(-4, 5))
        ax.grid(True, color="w", lw=0.4, alpha=0.6)
        ax.set_title(c["title"])
        ax.set_xlabel("pixel x"); ax.set_ylabel("pixel y")
        ax.text(0.02, 0.98,
                r"$\|\sum_x g_x\|$ = %.2f" % stats[-1][0] + "\n" +
                r"$\sum_x \|g_x\|$ = %.2f" % stats[-1][1],
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", fc="w", ec="0.6", alpha=0.9))

    handles = [
        Line2D([0], [0], color=C_RED, lw=1.3, label="Gaussian hiện tại (mức 0.3)"),
        Line2D([0], [0], color="k", lw=1.3, ls="--", label="Gaussian đích (GT)"),
        Line2D([0], [0], color="#1c1c1c", marker=r"$\rightarrow$", ms=12, lw=0,
               label=r"$g_x=\partial L/\partial\mu'$ tại pixel x"),
        Line2D([0], [0], color=C_GREEN, lw=2.6, label=r"$\sum_x g_x$ (Inria dùng chuẩn của vector này)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, framealpha=0.9)

    # (c) bar chart
    ax = fig.add_subplot(gs[0, 2])
    labels = ["(a) lệch vị trí", "(b) sai kích thước"]
    x = np.arange(2)
    w = 0.36
    v_signed = [s[0] for s in stats]
    v_abs = [s[1] for s in stats]
    b1 = ax.bar(x - w / 2, v_signed, w, color=C_GREEN, label=r"$\|\sum_x g_x\|$ (có dấu, Inria dùng)")
    b2 = ax.bar(x + w / 2, v_abs, w, color=C_GRAY, alpha=0.6, hatch="//",
                label=r"$\sum_x \|g_x\|$ (không dấu, Inria KHÔNG dùng)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("độ lớn gradient (đơn vị tuỳ ý, cùng thang)")
    ax.set_ylim(0, max(v_abs) * 1.55)
    ax.set_title("(c) Tổng có dấu vs tổng không dấu")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5)
    ratio_b = v_signed[1] / v_abs[1]
    ax.text(0.5, 0.42, "Ở (b) các $g_x$ đối xứng\ntriệt tiêu: tỉ số chỉ %.1e\n→ $\\bar g_i$ nhỏ, không densify\n"
            "dù ảnh vẫn sai" % ratio_b,
            transform=ax.transAxes, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round", fc="#fff7e6", ec=C_ORANGE))

    fig.suptitle(r"Inria chỉ tích luỹ $\|\sum_x \partial L/\partial\mu'_i|_x\|$: gradient pixel ngược chiều triệt tiêu trong kernel", fontsize=10.5)
    save(fig, "fig_03_grad_cancel.png")
    return stats


# ---------------------------------------------------------------------------
# Hinh 4: g_bar_i(t) cua 5 Gaussian qua 15000 vong, reset moi 100 vong
# ---------------------------------------------------------------------------
def fig04():
    rng = np.random.default_rng(SEED + 1)
    T = 15000
    t = np.arange(1, T + 1)
    RESET_OP = [3000, 6000, 9000, 12000]

    def opacity_bump(amp, width=400):
        b = np.zeros(T)
        for r in RESET_OP:
            m = t > r
            b[m] += amp * np.exp(-(t[m] - r) / width)
        return b

    # Bien do gradient "sach" theo thoi gian cho tung loai Gaussian
    profiles = {}
    # 1. vien vat the: gradient lon, giam cham khi anh dan khop
    profiles["Viền vật thể"] = dict(
        base=1.2e-4 + 4.6e-4 * np.exp(-t / 5500.0) + opacity_bump(1.5e-4),
        p_vis=0.9, sigma=0.35, birth=1, color=C_RED)
    # 2. gan vien: lo lung quanh nguong
    profiles["Gần viền"] = dict(
        base=1.9e-4 + 0.4e-4 * np.sin(t / 900.0) - 0.3e-4 * (t / T) + opacity_bump(0.8e-4),
        p_vis=0.85, sigma=0.35, birth=1, color=C_ORANGE)
    # 3. nen phang: gradient nho
    profiles["Nền phẳng"] = dict(
        base=3.0e-5 + 1.5e-5 * np.exp(-t / 3000.0) + opacity_bump(0.5e-4, 250),
        p_vis=0.95, sigma=0.4, birth=1, color=C_BLUE)
    # 4. thinh thoang bi che: chi nhin thay ~30% vong -> denom nho -> g_bar rat nhieu
    profiles["Thỉnh thoảng bị che"] = dict(
        base=2.2e-4 + 1.0e-4 * np.exp(-t / 6000.0) + opacity_bump(1.0e-4),
        p_vis=0.3, sigma=0.6, birth=1, color=C_PURPLE)
    # 5. Gaussian moi sinh (clone tai t=3000): gradient cao roi giam nhanh
    tb = 3000
    profiles["Mới sinh (clone tại t=3000)"] = dict(
        base=1.0e-4 + 5.5e-4 * np.exp(-np.clip(t - tb, 0, None) / 1200.0) + opacity_bump(1.0e-4),
        p_vis=0.9, sigma=0.35, birth=tb, color=C_GREEN)

    densify_ts = np.arange(600, 15000, 100)  # 500 < t < 15000, moi 100 vong
    densify_ts = densify_ts[densify_ts < 15000]
    results = {}
    for name, p in profiles.items():
        vis = (rng.random(T) < p["p_vis"]).astype(float)
        vis[t < p["birth"]] = 0.0
        # doan bi che lien tuc cho Gaussian 4
        if name.startswith("Thỉnh"):
            for s in rng.integers(0, T - 400, 12):
                vis[s:s + 300] = 0.0
        g = p["base"] * np.exp(rng.normal(0, p["sigma"], T)) * vis
        # reset accum/denom ve 0 sau moi lan densify (t % 100 == 0)
        gbar = np.zeros(T)
        acc = 0.0
        den = 0.0
        for i in range(T):
            acc += g[i]
            den += vis[i]
            gbar[i] = acc / den if den > 0 else 0.0
            if t[i] % 100 == 0:  # thong ke bi xoa sau khi densify/prune doc xong
                acc = 0.0
                den = 0.0
        # gia tri dung tai luc densify: gbar tai t = k*100 (truoc khi reset)
        g_at = gbar[densify_ts - 1]
        hit = densify_ts[g_at >= TAU_GRAD]
        results[name] = dict(gbar=gbar, hit=hit, g_at=g_at, color=p["color"])

    names = list(profiles.keys())
    fig, axes = plt.subplots(5, 2, figsize=(11.5, 8.8), sharey="row",
                             gridspec_kw=dict(width_ratios=[2.4, 1.0]))
    for r, name in enumerate(names):
        res = results[name]
        gb = res["gbar"] * 1e4
        ymax = max(np.percentile(gb[gb > 0], 99.5) * 1.25, 0.6)
        for cidx, (lo, hi) in enumerate([(0, T), (0, 2000)]):
            ax = axes[r, cidx]
            m = (t >= lo) & (t <= hi)
            ax.plot(t[m], gb[m], color=res["color"], lw=0.55 if cidx == 0 else 0.9)
            ax.axhline(TAU_GRAD * 1e4, color="k", ls="--", lw=0.9)
            hit = res["hit"][(res["hit"] >= lo) & (res["hit"] <= hi)]
            ax.plot(hit, gb[hit - 1], "o", color="k", ms=3.2 if cidx == 0 else 4.5,
                    mfc=res["color"], mew=0.8, zorder=5)
            for rr in RESET_OP:
                if lo <= rr <= hi:
                    ax.axvline(rr, color=C_GRAY, ls=":", lw=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, ymax)
            ax.grid(True, alpha=0.25)
            if cidx == 0:
                ax.set_ylabel(r"$\bar g_i$ ($\times10^{-4}$)")
                ax.text(0.99, 0.95, f"{name}:  vượt ngưỡng {len(res['hit'])}/{len(densify_ts)} lần densify",
                        transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
                        bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
            else:
                ax.axvspan(0, 500, color=C_GRAY, alpha=0.12)
                ax.tick_params(labelleft=False)
            if r == 4:
                ax.set_xlabel("vòng lặp t" + ("" if cidx == 0 else "  (phóng to 0..2000)"))
            if r == 0:
                ax.set_title("Toàn bộ giai đoạn densify 0..15000 (reset stats sau mỗi 100 vòng)" if cidx == 0
                             else "Phóng to 0..2000: răng cưa do reset")
    axes[2, 1].text(250, axes[2, 1].get_ylim()[1] * 0.06, "chưa densify\n(t≤500)", ha="center", va="bottom",
                    fontsize=7.5, color=C_GRAY)
    axes[4, 1].text(1000, axes[4, 1].get_ylim()[1] * 0.5, "chưa tồn tại\n(được clone tại t=3000)",
                    ha="center", va="center", fontsize=8, color=C_GRAY)

    handles = [
        Line2D([0], [0], color="k", ls="--", lw=0.9, label=r"$\tau_{grad}=2\times10^{-4}$"),
        Line2D([0], [0], color="k", marker="o", mfc="w", ms=5, lw=0,
               label=r"mốc densify (t=k·100) mà $\bar g_i \geq \tau_{grad}$ → clone/split"),
        Line2D([0], [0], color=C_GRAY, ls=":", lw=0.9, label="reset opacity (t=3000, 6000, 9000, 12000)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, framealpha=0.9)
    fig.suptitle(r"Lịch sử $\bar g_i(t)=$accum$_i$/denom$_i$ của 5 Gaussian: giá trị chỉ được đọc tại mốc densify rồi xoá về 0",
                 fontsize=10.5)
    save(fig, "fig_04_gbar_history.png")
    return {k: (len(v["hit"]), v["hit"][:5].tolist(), len(densify_ts)) for k, v in results.items()}


if __name__ == "__main__":
    s2 = fig02()
    s3 = fig03()
    s4 = fig04()
    print("== fig02:", s2)
    print("== fig03 (||sum g||, sum ||g||):", s3)
    print("== fig04 (so lan vuot nguong, 5 moc dau, tong moc):", s4)
    print("Da luu:")
    for p in saved:
        print("  ", p)
