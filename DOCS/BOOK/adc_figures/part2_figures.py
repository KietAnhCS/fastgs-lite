"""
Hinh minh hoa cho PHAN 2 cua chuong Adaptive Density Control (FastGS-lite):
  buoc (4) do mat na loi ve tung Gaussian  +  buoc (5) Importance score.

Moi ham fig_2_k() xuat dung 1 file PNG `2_k_ten-ngan.png` vao cung thu muc.
Chi dung numpy + matplotlib (khong torch). Chu thich trong hinh viet
tieng Viet co dau (font DejaVu Sans mac dinh render tot).

Chay:  python adc_figures/part2_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

RNG = np.random.default_rng(0)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved:", path)


# ---------------------------------------------------------------------------
# Du lieu vi du 4x4 / 5 Gaussian / 3 view (dung so cua 3.md)
# Pixel danh so 1..16 theo hang (row-major).
# ---------------------------------------------------------------------------
OMEGA = {
    "G1": [{1, 2, 5, 6}, {1, 2, 5, 6}, {1, 2, 5, 6}],
    "G2": [{9, 10, 13, 14}, {9, 10, 13, 14}, {9, 10, 13, 14}],
    "G3": [{7, 8, 11, 12, 15, 16}, {7, 8, 11, 12, 15, 16}, {7, 8, 11, 12, 15, 16}],
    "G4": [{10, 11, 12, 14, 15, 16}, {11, 12, 15, 16}, {11, 12, 16}],
    "G5": [{3, 4, 7, 8, 12, 15, 16}, set(), set()],
}
MASK = [
    {7, 8, 11, 12, 15, 16},
    {7, 8, 11, 12, 15, 16},
    {7, 8, 10, 11, 12, 15, 16},
]
GNAMES = list(OMEGA.keys())
GCOLORS = ["#d62728", "#2ca02c", "#1f77b4", "#7f7f7f", "#ff7f0e"]


def counts_matrix():
    C = np.zeros((5, 3), dtype=int)
    S = np.zeros((5, 3), dtype=int)
    for i, g in enumerate(GNAMES):
        for v in range(3):
            C[i, v] = len(OMEGA[g][v] & MASK[v])
            S[i, v] = len(OMEGA[g][v])
    return C, S


def pix_rc(p):
    """pixel 1..16 -> (row, col) 0-based"""
    return (p - 1) // 4, (p - 1) % 4


def draw_grid4(ax, title=""):
    ax.set_xlim(0, 4)
    ax.set_ylim(4, 0)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(True, color="k", lw=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    for p in range(1, 17):
        r, c = pix_rc(p)
        ax.text(c + 0.5, r + 0.5, f"P{p}", ha="center", va="center", fontsize=7, color="0.35")
    if title:
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# 2.1  Mot Gaussian chieu thanh ellipse tren luoi pixel: 3 dinh nghia footprint
# ---------------------------------------------------------------------------
def fig_2_1():
    W = H = 24
    mu = np.array([11.3, 12.6])
    # covariance 2D
    a, b, th = 4.2, 2.2, np.deg2rad(30)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Sig = R @ np.diag([a**2, b**2]) @ R.T
    Sinv = np.linalg.inv(Sig)
    opacity = 0.6

    yy, xx = np.mgrid[0:H, 0:W]
    px = np.stack([xx + 0.5, yy + 0.5], -1) - mu
    maha = np.einsum("...i,ij,...j->...", px, Sinv, px)
    G = np.exp(-0.5 * maha)
    alpha = np.minimum(0.99, opacity * G)

    # (a) ellipse hinh hoc 3 sigma ; (b) compact box level-set mult=0.5 ; (c) alpha >= 1/255
    t_cb = 0.5 * 2 * np.log(255 * opacity)       # level-set: maha <= t_i = mult * 2 ln(255 alpha) (mult = 0.5)
    fp_geom = maha <= 9.0
    fp_cb = maha <= t_cb
    fp_alpha = alpha >= 1 / 255
    # cua thu 3: T cua pixel (do cac Gaussian phia truoc) < 1e-4 -> pixel dung som
    T_front = np.ones((H, W))
    T_front[(xx >= 14) & (yy <= 10)] = 5e-5     # goc tren-phai da bao hoa
    fp_final = fp_alpha & fp_cb & (T_front >= 1e-4)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    panels = [
        (fp_geom, "(a) ellipse hình học 3-sigma\n|Omega| = %d px" % fp_geom.sum()),
        (fp_cb, "(b) sau cull compact box (mult=0.5)\n|Omega| = %d px" % fp_cb.sum()),
        (fp_alpha, "(c) chỉ giữ alpha ≥ 1/255\n|Omega| = %d px" % fp_alpha.sum()),
        (fp_final, "(d) footprint HỮU HÌNH: (b) & (c) & T≥1e-4\n|Omega| = %d px" % fp_final.sum()),
    ]
    for ax, (fp, ttl) in zip(axes, panels):
        ax.imshow(alpha, cmap="Greys", vmin=0, vmax=1, extent=(0, W, H, 0), alpha=0.35)
        ax.imshow(np.where(fp, 1.0, np.nan), cmap="Reds", vmin=0, vmax=1.6, extent=(0, W, H, 0), alpha=0.8)
        for k in range(1, 4):
            ax.add_patch(Ellipse(mu, 2 * k * a, 2 * k * b, angle=np.rad2deg(th), fill=False, lw=0.8, ls="--", color="k"))
        ax.plot(*mu, "k+", ms=8)
        ax.set_xticks(np.arange(0, W + 1, 4))
        ax.set_yticks(np.arange(0, H + 1, 4))
        ax.grid(True, color="0.7", lw=0.4)
        ax.set_title(ttl, fontsize=8.5)
        ax.set_aspect("equal")
    axes[3].add_patch(Rectangle((14, 0), 10, 11, fill=False, ec="blue", lw=1.5, ls=":"))
    axes[3].text(14.3, 1.6, "vùng T<1e-4\n(bị che)", color="blue", fontsize=7)
    fig.suptitle("Bước 4: footprint Omega_i^v của MỘT Gaussian không phải ellipse hình học", fontsize=10)
    save(fig, "2_1_footprint-mot-gaussian.png")


# ---------------------------------------------------------------------------
# 2.2  Nhieu Gaussian chong nhau: mot pixel thuoc nhieu footprint, che khuat
# ---------------------------------------------------------------------------
def fig_2_2():
    W = H = 20
    yy, xx = np.mgrid[0:H, 0:W]
    P = np.stack([xx + 0.5, yy + 0.5], -1)

    def gauss(mu, a, b, th, op):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Sig = R @ np.diag([a**2, b**2]) @ R.T
        d = P - np.array(mu)
        maha = np.einsum("...i,ij,...j->...", d, np.linalg.inv(Sig), d)
        return np.minimum(0.99, op * np.exp(-0.5 * maha)), maha, Sig

    # theo thu tu depth (gan -> xa)
    specs = [
        ("A (gần, alpha=0.95)", (7.5, 8.5), 3.0, 2.0, 0.4, 0.95),
        ("B (giữa, alpha=0.5)", (11.5, 10.5), 3.5, 2.5, -0.6, 0.5),
        ("C (xa, alpha=0.7)", (9.5, 12.5), 4.0, 2.0, 1.0, 0.7),
    ]
    T = np.ones((H, W))
    fps = []
    for name, mu, a, b, th, op in specs:
        al, maha, Sig = gauss(mu, a, b, th, op)
        t_cb = 0.5 * 2 * np.log(255 * op)
        in_box = maha <= t_cb
        ok_alpha = al >= 1 / 255
        alive = T >= 1e-4  # early-out kiem tra T*(1-alpha) < 1e-4 ; xap xi bang T
        fp = in_box & ok_alpha & alive
        fps.append((name, fp, mu, a, b, th))
        T = np.where(in_box & ok_alpha, T * (1 - al), T)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax = axes[0]
    cols = ["#d62728", "#2ca02c", "#1f77b4"]
    for (name, fp, mu, a, b, th), c in zip(fps, cols):
        ax.imshow(np.where(fp, 1, np.nan), cmap=matplotlib.colors.ListedColormap([c]), extent=(0, W, H, 0), alpha=0.35)
        ax.add_patch(Ellipse(mu, 2 * 3 * a, 2 * 3 * b, angle=np.rad2deg(th), fill=False, lw=1, color=c))
        ax.text(mu[0], mu[1], name.split()[0], color=c, fontsize=10, weight="bold", ha="center")
    ax.set_title("3 Gaussian chồng nhau (ellipse 3-sigma,\n vùng tô = footprint hữu hình)", fontsize=9)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.grid(True, color="0.8", lw=0.4); ax.set_xticks(range(0, W + 1, 4)); ax.set_yticks(range(0, H + 1, 4))

    ax = axes[1]
    nfp = sum(fp.astype(int) for _, fp, *_ in fps)
    im = ax.imshow(nfp, cmap="YlOrRd", vmin=0, vmax=3, extent=(0, W, H, 0))
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], label="số Gaussian 'chạm' pixel")
    ax.set_title("Mỗi pixel thuộc BAO NHIÊU footprint?\n(1 pixel lỗi → tố cáo tất cả)", fontsize=9)
    ax.set_aspect("equal")

    ax = axes[2]
    im = ax.imshow(T, cmap="viridis", vmin=0, vmax=1, extent=(0, W, H, 0))
    fig.colorbar(im, ax=ax, label="T còn lại sau A,B,C")
    ax.set_title("Độ trong suốt tích luỹ T\n(T<1e-4 → Gaussian sau bị loại khỏi Omega)", fontsize=9)
    ax.set_aspect("equal")
    save(fig, "2_2_nhieu-gaussian-chong-nhau.png")


# ---------------------------------------------------------------------------
# 2.3  Vi du 4x4: giao mat na m_v va footprint Omega_i^v cho 5 Gaussian x 3 view
# ---------------------------------------------------------------------------
def fig_2_3():
    C, S = counts_matrix()
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.6))
    for v in range(3):
        ax = axes[v, 0]
        draw_grid4(ax, f"view {v+1}: mặt nạ m_v\n({len(MASK[v])} px lỗi)")
        for p in MASK[v]:
            r, c = pix_rc(p)
            ax.add_patch(Rectangle((c, r), 1, 1, color="k", alpha=0.55))
        for i, g in enumerate(GNAMES):
            ax = axes[v, i + 1]
            om = OMEGA[g][v]
            inter = om & MASK[v]
            ttl = f"{g}, view {v+1}\n|Omega|={S[i,v]}  counts={C[i,v]}"
            if not om:
                ttl = f"{g}, view {v+1}\nOmega = rỗng (bị che)"
            draw_grid4(ax, ttl)
            for p in MASK[v]:
                r, c = pix_rc(p)
                ax.add_patch(Rectangle((c, r), 1, 1, color="k", alpha=0.18))
            for p in om:
                r, c = pix_rc(p)
                ax.add_patch(Rectangle((c, r), 1, 1, fill=False, ec=GCOLORS[i], lw=2.5))
            for p in inter:
                r, c = pix_rc(p)
                ax.add_patch(Rectangle((c + 0.15, r + 0.15), 0.7, 0.7, color=GCOLORS[i], alpha=0.75))
    fig.suptitle("Bước 4 trên ví dụ 4x4: xám = pixel lỗi m_v ; khung màu = Omega_i^v ; ô tô đặc = giao (được đếm)", fontsize=10)
    save(fig, "2_3_giao-mat-na-footprint.png")


# ---------------------------------------------------------------------------
# 2.4  Bar chart counts / |Omega| moi Gaussian moi view
# ---------------------------------------------------------------------------
def fig_2_4():
    C, S = counts_matrix()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(5)
    w = 0.26
    ax = axes[0]
    for v in range(3):
        ax.bar(x + (v - 1) * w, S[:, v], w, color="0.8", edgecolor="0.4")
        ax.bar(x + (v - 1) * w, C[:, v], w, color=["#d62728", "#2ca02c", "#1f77b4"][v], label=f"view {v+1}: counts")
        for i in range(5):
            ax.text(x[i] + (v - 1) * w, S[i, v] + 0.1, f"{C[i,v]}/{S[i,v]}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(GNAMES)
    ax.set_ylabel("số pixel")
    ax.set_title("counts_i^v (màu) trên nền |Omega_i^v| (xám)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    tot_err = [len(m) for m in MASK]
    tot_cnt = C.sum(0)
    ax.bar(np.arange(3) - 0.18, tot_err, 0.36, label="số pixel lỗi của view (|m_v|)", color="0.4")
    ax.bar(np.arange(3) + 0.18, tot_cnt, 0.36, label="sum_i counts_i^v", color="#ff7f0e")
    for v in range(3):
        ax.text(v - 0.18, tot_err[v] + 0.2, str(tot_err[v]), ha="center", fontsize=8)
        ax.text(v + 0.18, tot_cnt[v] + 0.2, str(tot_cnt[v]), ha="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"view {v+1}" for v in range(3)])
    ax.set_title("Một pixel lỗi 'tố cáo' MỌI Gaussian chạm nó\n⇒ sum_i counts > số pixel lỗi", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "2_4_bar-counts-moi-view.png")


# ---------------------------------------------------------------------------
# 2.5  Importance 5 Gaussian + heatmap Gaussian x view
# ---------------------------------------------------------------------------
def fig_2_5():
    C, S = counts_matrix()
    V = 3
    mean = C.sum(1) / V
    imp = np.floor(mean).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=6)
    for i in range(5):
        for v in range(3):
            txt = str(C[i, v]) if S[i, v] else "0 (rỗng)"
            ax.text(v, i, txt, ha="center", va="center", fontsize=8, color="w" if C[i, v] > 3 else "k")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"view {v+1}" for v in range(3)])
    ax.set_yticks(range(5)); ax.set_yticklabels(GNAMES)
    ax.set_title("Ma trận counts (5 x 3)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    x = np.arange(5)
    ax.bar(x, mean, color="0.75", label="(1/V) sum_v counts (chưa floor)")
    ax.bar(x, imp, color=GCOLORS, label="Importance = floor(...)", width=0.5)
    ax.axhline(5, color="r", ls="--", lw=1.2, label="ngưỡng 5 (cần > 5)")
    for i in range(5):
        ax.text(x[i], mean[i] + 0.1, f"{mean[i]:.2f} → {imp[i]}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(GNAMES)
    ax.set_ylim(0, 7.2)
    ax.set_title("Bước 5: Importance_i (chỉ G3 vượt ngưỡng)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    alts = {
        "mean (code)": mean,
        "max_v": C.max(1),
        "sum_v": C.sum(1),
    }
    w = 0.26
    for k, (name, val) in enumerate(alts.items()):
        ax.bar(x + (k - 1) * w, val, w, label=name)
    ax.axhline(5, color="r", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(GNAMES)
    ax.set_title("Nếu thay mean bằng max hay sum?\n(G5: max=5, sum=5 ; mean=1.67)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "2_5_importance-heatmap.png")


# ---------------------------------------------------------------------------
# 2.6  Mo phong 200 Gaussian: footprint size khac nhau -> phan bo Importance
# ---------------------------------------------------------------------------
def simulate_population(n=200, V=10, rng=None):
    rng = rng or np.random.default_rng(1)
    size = np.exp(rng.uniform(np.log(4), np.log(3000), n)).astype(int)      # |Omega| moi view
    rho = rng.beta(0.7, 4.0, n)                                              # ti le pixel loi trong footprint
    pvis = rng.uniform(0.3, 1.0, n)                                           # xac suat huu hinh o mot view
    counts = np.zeros((n, V), dtype=int)
    for v in range(V):
        vis = rng.uniform(size=n) < pvis
        counts[:, v] = np.where(vis, rng.binomial(size, rho), 0)
    return size, rho, pvis, counts


def fig_2_6():
    size, rho, pvis, counts = simulate_population()
    V = counts.shape[1]
    imp = counts.sum(1) // V
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    bins = np.concatenate([[0, 1, 2, 3, 4, 5, 6], np.geomspace(8, imp.max() + 1, 14)])
    ax.hist(imp, bins=bins, color="#1f77b4", edgecolor="k")
    ax.axvline(5.5, color="r", ls="--", label="Importance > 5")
    ax.set_xscale("symlog", linthresh=10); ax.set_xlim(-0.5, imp.max() * 1.3)
    ax.set_xlabel("Importance_i"); ax.set_ylabel("số Gaussian")
    ax.set_title(f"Phân bố Importance (200 Gaussian, V={V})\n{(imp>5).sum()} vượt ngưỡng, {(imp==0).sum()} bằng 0", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    sc = ax.scatter(size, imp + 0.5, c=rho, cmap="plasma", s=18, edgecolor="k", lw=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(5.5, color="r", ls="--")
    xs = np.geomspace(4, 3000, 50)
    for r in [0.05, 0.2, 0.5]:
        ax.plot(xs, xs * r * np.mean(pvis) + 0.5, ":", color="0.4", lw=0.8)
        ax.text(xs[-1], xs[-1] * r * np.mean(pvis), f"rho={r}", fontsize=6.5, color="0.4")
    ax.set_xlabel("|Omega_i| (px, log)"); ax.set_ylabel("Importance + 0.5 (log)")
    ax.set_title("Importance ~ |Omega| * rho * p_vis\nGaussian to dễ vượt ngưỡng hơn", fontsize=9)
    fig.colorbar(sc, ax=ax, label="rho = tỉ lệ pixel lỗi trong footprint")

    ax = axes[2]
    ratio = counts.sum(1) / np.maximum(1, (counts > 0).sum(1)) / size   # ti le loi trung binh khi huu hinh
    ax.scatter(size, ratio, c=(imp > 5), cmap="coolwarm", s=18, edgecolor="k", lw=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("|Omega_i| (px, log)"); ax.set_ylabel("counts / |Omega| (chuẩn hoá theo diện tích)")
    ax.set_title("Đỏ = Importance>5, xanh = không.\nGaussian nhỏ có tỉ lệ lỗi cao vẫn bị chặn", fontsize=9)
    save(fig, "2_6_mo-phong-200-gaussian.png")


# ---------------------------------------------------------------------------
# 2.7  Monte Carlo: uoc luong Importance voi V = 1, 3, 10, 30 so voi toan bo
# ---------------------------------------------------------------------------
def fig_2_7():
    rng = np.random.default_rng(7)
    V_all = 200
    # ba kieu Gaussian: (A) loi nhat quan, (B) artefact 1 goc nhin, (C) bien gioi
    def make_counts(kind):
        if kind == "A":
            vis = rng.uniform(size=V_all) < 0.9
            return np.where(vis, rng.poisson(8, V_all), 0)
        if kind == "B":
            vis = rng.uniform(size=V_all) < 0.05
            return np.where(vis, rng.poisson(60, V_all), 0)
        if kind == "C":
            vis = rng.uniform(size=V_all) < 0.6
            return np.where(vis, rng.poisson(9, V_all), 0)
    kinds = {"A: lỗi nhất quán": "A", "B: artefact 1 góc": "B", "C: biên giới": "C"}
    Vs = [1, 3, 10, 30]
    trials = 3000

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for k, (label, kind) in enumerate(kinds.items()):
        pop = make_counts(kind)
        true_imp = pop.sum() / V_all
        means, stds, pgt5, lo, hi = [], [], [], [], []
        for V in Vs:
            est = np.empty(trials)
            for t in range(trials):
                idx = rng.choice(V_all, V, replace=False)
                est[t] = np.floor(pop[idx].sum() / V)
            means.append(est.mean()); stds.append(est.std()); pgt5.append((est > 5).mean())
            lo.append(np.percentile(est, 5)); hi.append(np.percentile(est, 95))
        ax = axes[0]
        yerr = [np.maximum(0, np.array(means) - np.array(lo)), np.maximum(0, np.array(hi) - np.array(means))]
        ax.errorbar(np.array(Vs) * (1 + 0.06 * (k - 1)), means, yerr=yerr, fmt="o-", capsize=4, label=label, color=GCOLORS[k])
        ax.axhline(true_imp, color=GCOLORS[k], ls=":", lw=1)
        ax = axes[1]
        ax.plot(Vs, pgt5, "o-", label=label, color=GCOLORS[k])
        ax = axes[2]
        ax.plot(Vs, stds, "o-", label=label, color=GCOLORS[k])
    axes[0].set_xscale("log"); axes[0].set_xticks(Vs); axes[0].set_xticklabels(Vs)
    axes[0].axhline(5, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("V (số view lấy mẫu)"); axes[0].set_ylabel("Importance ước lượng (mean ± std)")
    axes[0].set_title("Ước lượng Importance theo V\n(nét chấm: giá trị với V = 200 view)", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("log"); axes[1].set_xticks(Vs); axes[1].set_xticklabels(Vs)
    axes[1].set_xlabel("V"); axes[1].set_ylabel("P(Importance > 5)")
    axes[1].set_title("Xác suất qua ngưỡng densify", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[2].set_xscale("log"); axes[2].set_xticks(Vs); axes[2].set_xticklabels(Vs)
    axes[2].set_xlabel("V"); axes[2].set_ylabel("độ lệch chuẩn của ước lượng")
    axes[2].set_title("Phương sai giảm ~ 1/V", fontsize=9)
    axes[2].legend(fontsize=7)
    save(fig, "2_7_monte-carlo-V.png")


# ---------------------------------------------------------------------------
# 2.8  3DGS (chi gradient) vs FastGS-lite (gradient AND Importance>5)
# ---------------------------------------------------------------------------
def fig_2_8():
    rng = np.random.default_rng(3)
    n = 400
    grad = np.exp(rng.normal(np.log(2e-4), 0.9, n))
    size, rho, pvis, counts = simulate_population(n=n, V=10, rng=rng)
    imp = counts.sum(1) // 10
    tau = 2e-4
    g_ok = grad >= tau
    i_ok = imp > 5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.scatter(grad[~g_ok], imp[~g_ok] + 0.5, s=12, color="0.7", label="gradient < tau: cả hai đều bỏ qua")
    ax.scatter(grad[g_ok & i_ok], imp[g_ok & i_ok] + 0.5, s=14, color="#2ca02c", label="3DGS densify & FastGS densify")
    ax.scatter(grad[g_ok & ~i_ok], imp[g_ok & ~i_ok] + 0.5, s=14, color="#d62728", marker="x", label="3DGS densify, FastGS CHẶN (Importance≤5)")
    ax.axvline(tau, color="k", ls="--", lw=0.8); ax.axhline(5.5, color="k", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("||g_i|| (gradient tích luỹ, log)"); ax.set_ylabel("Importance_i + 0.5 (log)")
    ax.set_title("Phép AND: chiều thứ hai lọc thêm ứng viên", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")

    ax = axes[1]
    labels = ["3DGS gốc\n(gradient)", "FastGS-lite\n(gradient AND Imp>5)"]
    vals = [g_ok.sum(), (g_ok & i_ok).sum()]
    ax.bar(labels, vals, color=["0.5", "#2ca02c"])
    for k, vv in enumerate(vals):
        ax.text(k, vv + 3, str(vv), ha="center")
    ax.set_ylabel("số Gaussian được densify (trong 400)")
    ax.set_title(f"Giam {100*(1-vals[1]/vals[0]):.0f}% ứng viên mỗi lần densify\n⇒ tác động luỹ thừa lên N", fontsize=9)
    save(fig, "2_8_3dgs-vs-fastgs.png")


# ---------------------------------------------------------------------------
# 2.9  So do luong du lieu / tensor shape cua buoc 4-5
# ---------------------------------------------------------------------------
def fig_2_9():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#1f77b4"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.8)

    def arrow(x0, y0, x1, y1, text=""):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=12, lw=1))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", fontsize=7, color="0.3")

    # hang tren: 1 view
    box(0.2, 3.6, 2.2, 1.2, "render_fastgs(cam_v)\n-> render [3,H,W]")
    box(2.8, 3.6, 2.2, 1.2, "get_loss:\ne_v = mean_ch|r-g| [H,W]\nminmax -> e_hat [H,W]")
    box(5.4, 3.6, 2.2, 1.2, "metric_map =\n(e_hat > 0.1).int()\n[H,W] in {0,1}")
    box(8.0, 3.6, 2.6, 1.2, "render_fastgs(..., get_flag=True,\nmetric_map)  -> kernel forward.cu\nmetricCount [N] int32 (zeros)")
    box(11.0, 3.6, 1.9, 1.2, "accum_metric_counts\n= counts^(v) [N]", fc="#fdecec", ec="#d62728")
    arrow(2.4, 4.2, 2.8, 4.2); arrow(5.0, 4.2, 5.4, 4.2); arrow(7.6, 4.2, 8.0, 4.2); arrow(10.6, 4.2, 11.0, 4.2)

    # kernel detail
    box(8.0, 1.9, 2.6, 1.3, "trong kernel, mỗi (pixel, Gaussian):\n power<=0 ; alpha>=1/255 ; T>=1e-4\n if metric_map[pix]==1:\n   atomicAdd(metricCount[id], 1)", fc="#fff7e6", ec="#ff7f0e")
    arrow(9.3, 3.6, 9.3, 3.2)

    # hang duoi: cong don qua V view
    box(0.2, 0.3, 3.4, 1.2, "full_metric_counts += counts^(v)\n(lặp v = 1..V, V = len(camlist) = 10)\n[N] int")
    box(4.2, 0.3, 3.4, 1.2, "importance_score =\ntorch.div(full_metric_counts, V,\nrounding_mode='floor')  [N] int", fc="#e8f5e9", ec="#2ca02c")
    box(8.2, 0.3, 4.6, 1.2, "full_metric_score += E_photo^(v) * counts^(v)\n-> pruning_score = minmax_N(...)  [N] float\n(bước 6-7, phần sau)", fc="#f3f3f3", ec="0.5")
    arrow(11.9, 3.6, 1.9, 1.5, "mỗi view")
    arrow(3.6, 0.9, 4.2, 0.9)
    arrow(11.9, 3.6, 10.5, 1.5)
    ax.text(0.2, 5.0, "Luồng dữ liệu bước 4-5 (utils/fast_utils.py: compute_gaussian_score_fastgs ; forward.cu:406-408)", fontsize=9, weight="bold")
    save(fig, "2_9_so-do-tensor.png")


# ---------------------------------------------------------------------------
# 2.10  Hieu ung floor va nguong chat ">5"
# ---------------------------------------------------------------------------
def fig_2_10():
    V = 10
    tot = np.arange(0, 80)
    imp = tot // V
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    ax = axes[0]
    ax.step(tot, imp, where="post", color="#1f77b4", label="floor(sum/V)")
    ax.plot(tot, tot / V, "--", color="0.5", label="sum/V (không floor)")
    ax.axhline(5, color="r", ls="--", lw=1)
    ax.axvspan(60, 80, color="#2ca02c", alpha=0.15, label="qua ngưỡng: sum ≥ 60 = 6V")
    ax.axvspan(50, 60, color="#d62728", alpha=0.12, label="sum trong [50,60): mean trong [5,6) → floor = 5, KHÔNG qua")
    ax.set_xlabel("sum_v counts_i^v  (V = 10)"); ax.set_ylabel("Importance_i")
    ax.set_title("Floor + so sánh chặt '> 5' ⇔ cần ít nhất 60 pixel lỗi\ntổng trên 10 view (trung bình 6 px/view)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")

    ax = axes[1]
    # Gaussian huu hinh o k/10 view, moi view bi to c pixel: can c >= 60/k
    k = np.arange(1, 11)
    for c in [6, 10, 20, 60]:
        ax.plot(k, np.floor(c * k / 10), "o-", label=f"{c} px lỗi mỗi view hữu hình")
    ax.axhline(5, color="r", ls="--", lw=1)
    ax.set_xlabel("số view (trong 10) mà Gaussian hữu hình và bị tô")
    ax.set_ylabel("Importance")
    ax.set_title("Cùng số pixel lỗi mỗi view, ít view hữu hình → bị dìm\n(tính nhất quán đa góc nhìn)", fontsize=9)
    ax.legend(fontsize=6.5)
    save(fig, "2_10_floor-va-nguong.png")


# ---------------------------------------------------------------------------
# 2.11  Canh do choi 48x32 (kiem dinh so): counts ~ |Omega| vi ca anh deu sai
# ---------------------------------------------------------------------------
def fig_2_11():
    counts = np.array([[1248, 954, 930], [1181, 1003, 869], [1245, 978, 977], [1180, 973, 770]])
    omega = np.array([[1248, 954, 930], [1197, 1005, 869], [1301, 978, 990], [1193, 974, 770]])
    mask_on = np.array([1409, 1077, 1042])
    names = ["G1", "G2", "G3", "G4"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    x = np.arange(4); w = 0.26
    for v in range(3):
        ax.bar(x + (v - 1) * w, omega[:, v], w, color="0.85", edgecolor="0.5")
        ax.bar(x + (v - 1) * w, counts[:, v], w, color=["#d62728", "#2ca02c", "#1f77b4"][v], label=f"view {v+1}")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("pixel")
    ax.set_title("Cảnh 48x32: counts (màu) gần bằng |Omega| (xám)\nvì gần như mọi pixel đều lỗi", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    frac = counts / omega
    im = ax.imshow(frac, cmap="Greens", vmin=0.9, vmax=1.0)
    for i in range(4):
        for v in range(3):
            ax.text(v, i, f"{frac[i,v]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"view {v+1}" for v in range(3)])
    ax.set_yticks(range(4)); ax.set_yticklabels(names)
    ax.set_title("counts / |Omega| (tỉ lệ pixel footprint bị tô)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[2]
    ax.bar(np.arange(3) - 0.18, mask_on, 0.36, color="0.4", label="|m_v| pixel lỗi (trên 1536)")
    ax.bar(np.arange(3) + 0.18, counts.sum(0), 0.36, color="#ff7f0e", label="sum_i counts_i^v")
    for v in range(3):
        ax.text(v + 0.18, counts.sum(0)[v] + 30, str(counts.sum(0)[v]), ha="center", fontsize=8)
        ax.text(v - 0.18, mask_on[v] + 30, str(mask_on[v]), ha="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"view {v+1}" for v in range(3)])
    ax.set_title("4854 = sum counts >> 1409 pixel lỗi (view 1)\nImportance = 1044/1017/1066/974 >> 5", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "2_11_canh-do-choi-48x32.png")


# ---------------------------------------------------------------------------
# 2.12  Gaussian to vs nho: cung ti le loi, Importance khac nhau (anh 2D)
# ---------------------------------------------------------------------------
def fig_2_12():
    rng = np.random.default_rng(11)
    W = H = 40
    yy, xx = np.mgrid[0:H, 0:W]
    P = np.stack([xx + 0.5, yy + 0.5], -1)
    # mat na loi: mot dai cheo + nhieu ngau nhien
    mask = (np.abs((xx - 0.9 * yy) - 4) < 5) | (rng.uniform(size=(H, W)) < 0.08)

    def fp(mu, s, op=0.7):
        d = P - np.array(mu)
        maha = (d ** 2).sum(-1) / s**2
        al = np.minimum(0.99, op * np.exp(-0.5 * maha))
        t_cb = 0.5 * 2 * np.log(255 * op)
        return (maha <= t_cb) & (al >= 1 / 255)

    gs = [("nhỏ, s=0.7", (12, 8), 0.7), ("vừa, s=3", (22, 20), 3.0), ("to, s=6", (26, 27), 6.0)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    ax = axes[0]
    ax.imshow(mask, cmap="Greys", extent=(0, W, H, 0), alpha=0.6)
    rows = []
    label_xy = [(12, 4.5), (35, 20), (26, 8)]   # vi tri nhan (tranh chong nhau)
    for (name, mu, s), c, (lx, ly) in zip(gs, GCOLORS, label_xy):
        f = fp(mu, s)
        cnt = (f & mask).sum(); tot = f.sum()
        rows.append((name, cnt, tot))
        ax.contour(xx + 0.5, yy + 0.5, f.astype(float), levels=[0.5], colors=[c], linewidths=1.5)
        ax.text(lx, ly, f"{name}\n{cnt}/{tot}", color=c, fontsize=7, ha="center", va="center")
    ax.set_title("Cùng mặt nạ lỗi, 3 Gaussian kích thước khác nhau", fontsize=9)
    ax.set_aspect("equal")
    ax = axes[1]
    ax.bar([r[0] for r in rows], [r[1] for r in rows], color=GCOLORS[:3])
    ax.axhline(5, color="r", ls="--"); ax.set_title("counts (1 view) = Importance nếu\nnhất quán qua V view", fontsize=9)
    for k, r in enumerate(rows):
        ax.text(k, r[1] + 0.5, str(r[1]), ha="center", fontsize=8)
    ax = axes[2]
    ax.bar([r[0] for r in rows], [r[1] / r[2] for r in rows], color=GCOLORS[:3])
    ax.set_ylim(0, 1); ax.set_title("counts / |Omega| (chuẩn hoá diện tích)\n- code KHÔNG dùng", fontsize=9)
    for k, r in enumerate(rows):
        ax.text(k, r[1] / r[2] + 0.02, f"{r[1]/r[2]:.2f}", ha="center", fontsize=8)
    ax = axes[3]
    # tuong quan tren pham vi lien tuc
    ss = np.linspace(0.8, 8, 40)
    cnts = []
    for s in ss:
        f = fp((22, 20), s)
        cnts.append((f & mask).sum())
    ax.plot(ss, cnts, "k-")
    ax.axhline(5, color="r", ls="--")
    ax.set_xlabel("s (px) của Gaussian tại (22,20)"); ax.set_ylabel("counts")
    ax.set_title("counts tăng ~ s^2: Gaussian to\ntự động 'quan trọng' hơn", fontsize=9)
    save(fig, "2_12_to-vs-nho.png")


if __name__ == "__main__":
    for fn in [fig_2_1, fig_2_2, fig_2_3, fig_2_4, fig_2_5, fig_2_6,
               fig_2_7, fig_2_8, fig_2_9, fig_2_10, fig_2_11, fig_2_12]:
        fn()
    print("done.")
