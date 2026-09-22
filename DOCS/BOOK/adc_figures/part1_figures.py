"""
Hinh minh hoa cho PHAN 1 cua chuong 12 (Adaptive Density Control, FastGS-lite):
  - Lich chay ADC trong vong lap train (timeline 0 -> 30000)
  - Buoc (1) sai so mau tung pixel e_v(x)
  - Buoc (2) chuan hoa min-max theo anh
  - Buoc (3) nguong nhi phan tau_loss = 0.1

Moi ham fig_1_k() xuat dung 1 file PNG ten `1_k_ten-ngan.png` vao cung thu muc.
So lieu vi du 4x4 / 3 view lay dung tu `3.md` (muc 12.3 cua file goc).

Chay: python adc_figures/part1_figures.py
Chi dung numpy + matplotlib.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved:", path)


# ---------------------------------------------------------------------------
# Du lieu vi du 4x4 (view 1) — dung tu 3.md
# ---------------------------------------------------------------------------
REND1 = np.array([
    [0.48, 0.51, 0.47], [0.43, 0.38, 0.44], [0.57, 0.60, 0.61], [0.37, 0.34, 0.36],
    [0.54, 0.54, 0.56], [0.48, 0.50, 0.49], [0.40, 0.42, 0.41], [0.40, 0.40, 0.40],
    [0.32, 0.30, 0.29], [0.42, 0.44, 0.49], [0.39, 0.41, 0.40], [0.39, 0.40, 0.40],
    [0.39, 0.35, 0.39], [0.43, 0.41, 0.43], [0.45, 0.44, 0.43], [0.35, 0.35, 0.34],
]).reshape(4, 4, 3)

GT1 = np.array([
    [0.50, 0.50, 0.50], [0.42, 0.40, 0.44], [0.60, 0.58, 0.62], [0.35, 0.36, 0.34],
    [0.55, 0.53, 0.57], [0.48, 0.47, 0.49], [0.70, 0.66, 0.62], [0.68, 0.60, 0.58],
    [0.30, 0.31, 0.29], [0.45, 0.44, 0.46], [0.72, 0.68, 0.64], [0.75, 0.70, 0.64],
    [0.38, 0.37, 0.39], [0.41, 0.42, 0.40], [0.66, 0.62, 0.58], [0.80, 0.74, 0.70],
]).reshape(4, 4, 3)

# View 2, 3 cho san o dang e_v (da qua buoc 1)
E2 = np.array([.01, .02, .01, .02, .02, .01, .13, .11, .02, .01, .15, .14, .01, .02, .12, .16]).reshape(4, 4)
E3 = np.array([.02, .01, .02, .03, .01, .02, .19, .17, .02, .12, .21, .20, .01, .02, .16, .24]).reshape(4, 4)

TAU = 0.1


def pixel_error(rend, gt):
    """Buoc (1): e_v(x) = 1/3 * sum_ch |rend - gt|   (torch.mean(torch.abs(.), 0))"""
    return np.mean(np.abs(rend - gt), axis=-1)


def minmax(e):
    """Buoc (2): (e - min) / (max - min)   (get_loss trong fast_utils.py)"""
    return (e - e.min()) / (e.max() - e.min())


E1 = pixel_error(REND1, GT1)
E1 = np.round(E1, 2)  # dung nhu bang trong 3.md (.02 .01 ... .40)


def annotate(ax, M, fmt="{:.2f}", color_fn=None, fontsize=8):
    H, W = M.shape
    for i in range(H):
        for j in range(W):
            c = "white" if (color_fn is None and M[i, j] > 0.5 * M.max()) else "black"
            if color_fn is not None:
                c = color_fn(M[i, j])
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fontsize, color=c)


def pixel_labels(ax):
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(["c0", "c1", "c2", "c3"])
    ax.set_yticklabels(["r0", "r1", "r2", "r3"])


# ---------------------------------------------------------------------------
# 1_1  Timeline lich chay ADC
# ---------------------------------------------------------------------------
def fig_1_1():
    T = 30000
    dens_from, dens_until, interval, reset = 500, 15000, 500, 3000
    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Band: tich luy thong ke gradient (t < 15000)
    ax.add_patch(Rectangle((0, 3.6), dens_until, 0.8, color="#c7d9f1", alpha=0.9))
    ax.text(dens_until / 2, 4.0, "tích luỹ accum / accum_abs / denom (t < 15000)",
            ha="center", va="center", fontsize=8.5)

    # Densify + prune (500 < t < 15000, moi 500 preset)
    dens = np.arange(dens_from + interval, dens_until, interval)
    ax.vlines(dens, 2.65, 3.35, color="#2a7d2e", lw=1.2)
    ax.text(dens_until + 300, 3.0, f"densify + prune (multinomial)\nmỗi {interval} vòng, {len(dens)} lần",
            va="center", fontsize=8.5, color="#2a7d2e")

    # Opacity reset (moi 3000, toan bo vi trong nhanh t < densify_until)
    resets = np.arange(reset, dens_until, reset)
    ax.vlines(resets, 1.65, 2.35, color="#b8860b", lw=2.2)
    ax.text(dens_until + 300, 2.0, "reset_opacity (mỗi 3000, chỉ khi t < 15000)\n"
            "alpha <- min(alpha, 0.01)", va="center", fontsize=8.5, color="#b8860b")

    # Final prune (15000 < t < 30000, moi 3000)
    fp = np.arange(18000, T, 3000)
    ax.vlines(fp, 0.65, 1.35, color="#c0392b", lw=2.5)
    for t in fp:
        ax.text(t, 0.45, str(t), ha="center", fontsize=7.5, color="#c0392b")
    ax.text(dens_until + 300, 1.0, "", va="center")
    ax.text(16000, 1.55, "final_prune_fastgs: alpha < 0.1  OR  Pruning > 0.9  (xoá thẳng, 4 lần)",
            fontsize=8.5, color="#c0392b")

    # Moc quan trong
    for t, lab, dy in [(500, "500\ndensify_from", -0.6), (3000, "3000\nsize_threshold bật", -0.15),
                       (15000, "15000\ndensify_until", -0.15), (30000, "30000\nkết thúc (KHÔNG final prune)", -0.15)]:
        ax.axvline(t, color="k", ls=":", lw=0.8)
        ax.text(t, dy, lab, ha="center", va="top", fontsize=7.5)

    ax.set_xlim(-300, T + 9000)
    ax.set_ylim(-1.3, 4.7)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["final prune", "opacity reset", "densify+prune", "thống kê grad"])
    ax.set_xlabel("vòng lặp t")
    ax.set_title("Lịch chạy Adaptive Density Control trong train.py (preset densification_interval = 500)")
    ax.grid(axis="x", alpha=0.2)
    save(fig, "1_1_timeline.png")


# ---------------------------------------------------------------------------
# 1_2  N(t) dinh tinh: 3DGS vs FastGS-lite
# ---------------------------------------------------------------------------
def fig_1_2():
    rng = np.random.default_rng(0)
    T = 30000
    t = np.arange(0, T + 1, 100)
    N0 = 100_000

    def simulate(r_spawn, r_prune, r_final, interval=500):
        N = np.zeros_like(t, dtype=float)
        n = N0
        for k, tt in enumerate(t):
            if 500 < tt < 15000 and tt % interval == 0:
                n = n * (1 + r_spawn) * (1 - r_prune)
            if tt % 3000 == 0 and 15000 < tt < 30000:
                n = n * (1 - r_final)
            N[k] = n
        return N

    N_3dgs = simulate(0.085, 0.035, 0.0)
    N_fast = simulate(0.045, 0.030, 0.12)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, N_3dgs / 1e6, color="#7f8c8d", lw=2, label="3DGS gốc (chỉ gradient, không final prune)")
    ax.plot(t, N_fast / 1e6, color="#2a7d2e", lw=2, label="FastGS-lite (AND Importance>5, final prune)")
    ax.axvspan(500, 15000, color="#c7d9f1", alpha=0.35, label="giai đoạn densify (500..15000)")
    for ft in [18000, 21000, 24000, 27000]:
        ax.axvline(ft, color="#c0392b", ls="--", lw=0.9)
    ax.text(22500, N_fast.max() / 1e6 * 0.92, "final prune\n(mỗi 3000)", color="#c0392b", ha="center", fontsize=8.5)
    ax.set_xlabel("vòng lặp t")
    ax.set_ylabel("N (triệu Gaussian)  —  MINH HOẠ ĐỊNH TÍNH")
    ax.set_title("N(t): mỗi lần densify là một hệ số nhân; giảm r_spawn → tác động luỹ thừa")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, "1_2_N_qualitative.png")


# ---------------------------------------------------------------------------
# 1_3  Buoc (1): GT, render, |diff| tung kenh, e_v — view 1 cua 3.md
# ---------------------------------------------------------------------------
def fig_1_3():
    diff = np.abs(REND1 - GT1)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))

    ax = axes[0, 0]
    ax.imshow(GT1, interpolation="nearest")
    ax.set_title("I_gt (view 1, RGB thật)")
    pixel_labels(ax)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"P{i*4+j+1}", ha="center", va="center", fontsize=8, color="white")

    ax = axes[0, 1]
    ax.imshow(REND1, interpolation="nearest")
    ax.set_title("I_rend (view 1)")
    pixel_labels(ax)

    ax = axes[0, 2]
    im = ax.imshow(E1, cmap="magma", vmin=0, vmax=0.45, interpolation="nearest")
    ax.set_title("e_v(x) = 1/3 * sum_ch |diff|   (bước 1)")
    annotate(ax, E1, color_fn=lambda v: "white" if v < 0.3 else "black")
    pixel_labels(ax)
    fig.colorbar(im, ax=ax, fraction=0.046)

    for k, (name, cm) in enumerate([("R", "Reds"), ("G", "Greens"), ("B", "Blues")]):
        ax = axes[1, k]
        im = ax.imshow(diff[..., k], cmap=cm, vmin=0, vmax=0.5, interpolation="nearest")
        ax.set_title(f"|I_rend - I_gt| kênh {name}")
        annotate(ax, diff[..., k], color_fn=lambda v: "white" if v > 0.3 else "black")
        pixel_labels(ax)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Bước (1): sai số màu từng pixel — ảnh 4x4 view 1 (số liệu 3.md). e(P7) = (0.30+0.24+0.21)/3 = 0.25")
    save(fig, "1_3_pixel_error_view1.png")


# ---------------------------------------------------------------------------
# 1_4  Vi sao L1 (khong phai L2) o buoc (1)
# ---------------------------------------------------------------------------
def fig_1_4():
    diff = np.abs(REND1 - GT1)
    e_l1 = np.mean(diff, axis=-1)
    e_l2 = np.sqrt(np.mean(diff ** 2, axis=-1))   # RMS theo kenh
    e_sq = np.mean(diff ** 2, axis=-1)           # MSE theo kenh (khong lay can)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))

    ax = axes[0]
    d = np.linspace(0, 0.5, 200)
    ax.plot(d, d, label="|d|  (L1)", color="#2a7d2e", lw=2)
    ax.plot(d, d ** 2, label="d^2  (L2 không căn)", color="#c0392b", lw=2)
    ax.axvline(0.05, color="k", ls=":", lw=0.8)
    ax.text(0.052, 0.42, "sai số nhỏ 0.05:\nL1 = 0.05, L2 = 0.0025", fontsize=8)
    ax.set_xlabel("|I_rend - I_gt| trên 1 kênh")
    ax.set_ylabel("đóng góp vào e")
    ax.set_title("L1 tuyến tính, L2 'bóp' sai số nhỏ")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    idx = np.arange(1, 17)
    order = np.argsort(-e_l1.ravel())
    ax.bar(idx - 0.2, e_l1.ravel()[order], width=0.4, label="L1 (code)", color="#2a7d2e")
    ax.bar(idx + 0.2, e_sq.ravel()[order], width=0.4, label="MSE (d^2)", color="#c0392b")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"P{o+1}" for o in order], rotation=60, fontsize=7)
    ax.set_title("16 pixel view 1, sắp giảm dần theo L1")
    ax.set_ylabel("e(x)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    ratio_l1 = e_l1.max() / np.median(e_l1)
    ratio_sq = e_sq.max() / np.median(e_sq)
    ax.bar(["L1", "RMS", "MSE"], [ratio_l1, e_l2.max() / np.median(e_l2), ratio_sq],
           color=["#2a7d2e", "#888", "#c0392b"])
    for k, v in enumerate([ratio_l1, e_l2.max() / np.median(e_l2), ratio_sq]):
        ax.text(k, v * 1.03, f"{v:.0f}x", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("max(e) / median(e)   (log)")
    ax.set_title("Độ 'kéo dài đuôi' của phân phối sai số")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Bước (1) dùng L1: ổn định với outlier, đơn vị = đơn vị màu; MSE làm min-max ở bước (2) càng nhạy outlier hơn")
    save(fig, "1_4_l1_vs_l2.png")


# ---------------------------------------------------------------------------
# 1_5  Buoc (2): e_v truoc / sau min-max cho 3 view
# ---------------------------------------------------------------------------
def fig_1_5():
    views = [("view 1", E1), ("view 2", E2), ("view 3", E3)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for k, (name, E) in enumerate(views):
        En = minmax(E)
        thr_abs = E.min() + TAU * (E.max() - E.min())

        ax = axes[0, k]
        vals = np.sort(E.ravel())
        ax.bar(np.arange(16), vals, color=np.where(vals > thr_abs, "#c0392b", "#7fb3d5"))
        ax.axhline(thr_abs, color="k", ls="--", lw=1)
        ax.text(0.2, thr_abs + 0.01, f"ngưỡng tương đương e > {thr_abs:.4f}", fontsize=8)
        ax.set_ylim(0, 0.45)
        ax.set_title(f"{name}: e_v thô  (min {E.min():.2f}, max {E.max():.2f})")
        ax.set_xlabel("16 pixel sắp tăng dần")
        ax.set_ylabel("e_v(x)")
        ax.grid(axis="y", alpha=0.25)

        ax = axes[1, k]
        valsn = np.sort(En.ravel())
        ax.bar(np.arange(16), valsn, color=np.where(valsn > TAU, "#c0392b", "#7fb3d5"))
        ax.axhline(TAU, color="k", ls="--", lw=1)
        ax.text(0.2, TAU + 0.02, "tau_loss = 0.1", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{name}: e_hat sau min-max  →  {int((En > TAU).sum())} pixel bật")
        ax.set_xlabel("16 pixel sắp tăng dần")
        ax.set_ylabel("e_hat_v(x)")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Bước (2): view 2 'tốt' (max 0.16) và view 1 'tệ' (max 0.40) đều được kéo về [0,1] → cùng 1 ngưỡng 0.1 cho ra số pixel lỗi tương đương")
    save(fig, "1_5_minmax_3views.png")


# ---------------------------------------------------------------------------
# 1_6  Buoc (2): mot pixel outlier keo tat ca ve gan 0
# ---------------------------------------------------------------------------
def fig_1_6():
    E_out = E1.copy()
    E_out[0, 0] = 0.90   # gia su P1 bi loi cuc lon (vd. pixel chet / vat the chuyen dong)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))
    for r, (E, lab) in enumerate([(E1, "gốc"), (E_out, "có outlier e(P1)=0.90")]):
        En = minmax(E)
        M = (En > TAU).astype(int)

        ax = axes[r, 0]
        im = ax.imshow(E, cmap="magma", vmin=0, vmax=0.9, interpolation="nearest")
        annotate(ax, E, color_fn=lambda v: "white" if v < 0.5 else "black")
        ax.set_title(f"e_v {lab}")
        pixel_labels(ax)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[r, 1]
        im = ax.imshow(En, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        annotate(ax, En, color_fn=lambda v: "white" if v < 0.6 else "black")
        ax.set_title(f"e_hat = minmax(e)   [max-min = {E.max()-E.min():.2f}]")
        pixel_labels(ax)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[2 - 1 if False else r, 2]
        ax.imshow(M, cmap=ListedColormap(["#eeeeee", "#c0392b"]), vmin=0, vmax=1, interpolation="nearest")
        annotate(ax, M, fmt="{:d}", color_fn=lambda v: "white" if v == 1 else "black")
        ax.set_title(f"mask e_hat > 0.1  →  {M.sum()} pixel")
        pixel_labels(ax)

    fig.suptitle("Bước (2) nhạy outlier: mẫu số 0.39 → 0.89, ngưỡng tương đương 0.049 → 0.099, P15 (e=0.18) vẫn bật nhưng P7..P16 giảm gần một nửa e_hat")
    save(fig, "1_6_outlier.png")


# ---------------------------------------------------------------------------
# 1_7  So sanh min-max / z-score / percentile
# ---------------------------------------------------------------------------
def fig_1_7():
    rng = np.random.default_rng(1)
    # Mo phong ban do sai so 48x32 "giong that": nen nho + 2 vung loi + 1 outlier
    H, W = 32, 48
    yy, xx = np.mgrid[0:H, 0:W]
    E = 0.01 + 0.01 * rng.random((H, W))
    E += 0.25 * np.exp(-((xx - 33) ** 2 + (yy - 20) ** 2) / (2 * 4.5 ** 2))
    E += 0.12 * np.exp(-((xx - 12) ** 2 + (yy - 9) ** 2) / (2 * 3.5 ** 2))
    E[3, 44] = 0.95  # outlier don le

    mm = minmax(E)
    z = (E - E.mean()) / E.std()
    z01 = (z - z.min()) / (z.max() - z.min())      # z-score cung chi la affine -> cung thu tu
    p5, p95 = np.percentile(E, [5, 95])
    pc = np.clip((E - p5) / (p95 - p5), 0, 1)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    maps = [("e_v thô", E, "magma", None), ("min-max (code)", mm, "viridis", TAU),
            ("z-score (affine, cùng thứ tự)", z, "viridis", None), ("percentile 5-95 (clip)", pc, "viridis", TAU)]
    for k, (name, M, cm, thr) in enumerate(maps):
        ax = axes[0, k]
        im = ax.imshow(M, cmap=cm, interpolation="nearest")
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax = axes[1, k]
        if thr is None and k == 2:
            # z-score: chon nguong sao cho cung so pixel voi min-max de so sanh
            mask = z > np.quantile(z, 1 - (mm > TAU).mean())
            ax.set_title(f"mask z (cùng số pixel = {mask.sum()})")
        elif thr is None:
            mask = E > 0.05
            ax.set_title(f"mask e > 0.05 tuyệt đối: {mask.sum()} px")
        else:
            mask = M > thr
            ax.set_title(f"mask > {thr}: {mask.sum()} px")
        ax.imshow(mask, cmap=ListedColormap(["#eeeeee", "#c0392b"]), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Bước (2): min-max phụ thuộc DUY NHẤT vào min và max (outlier góc phải trên); percentile bền vững hơn nhưng code dùng min-max vì rẻ và đủ tốt")
    save(fig, "1_7_norm_compare.png")


# ---------------------------------------------------------------------------
# 1_8  Buoc (3): mat na theo nhieu tau (3 view x 4 tau)
# ---------------------------------------------------------------------------
def fig_1_8():
    taus = [0.05, 0.1, 0.2, 0.5]
    views = [("view 1", E1), ("view 2", E2), ("view 3", E3)]
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.2))
    for r, (name, E) in enumerate(views):
        En = minmax(E)
        for c, tau in enumerate(taus):
            M = (En > tau).astype(int)
            ax = axes[r, c]
            ax.imshow(M, cmap=ListedColormap(["#eeeeee", "#c0392b"]), vmin=0, vmax=1, interpolation="nearest")
            annotate(ax, En, color_fn=lambda v, t=tau: "white" if v > t else "#555", fontsize=7.5)
            thr_abs = E.min() + tau * (E.max() - E.min())
            ax.set_title(f"{name}, tau={tau}: {M.sum()}/16 bật\n(e > {thr_abs:.3f})", fontsize=8.5)
            pixel_labels(ax)
            if r == 1 and c == 1:
                for s in ax.spines.values():
                    s.set_edgecolor("#2a7d2e"); s.set_linewidth(2.5)
    fig.suptitle("Bước (3): mặt nạ m_v = 1[e_hat > tau]; ô trong ghi e_hat. tau=0.1 (code, khung xanh) tách sạch 'nền' và 'vùng lỗi' ở cả 3 view")
    save(fig, "1_8_tau_masks.png")


# ---------------------------------------------------------------------------
# 1_9  Duong cong % pixel loi vs tau
# ---------------------------------------------------------------------------
def fig_1_9():
    taus = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for name, E, col in [("view 1", E1, "#c0392b"), ("view 2", E2, "#2a7d2e"), ("view 3", E3, "#2874a6")]:
        En = minmax(E)
        frac = [(En > t).mean() * 100 for t in taus]
        ax.plot(taus, frac, label=name, color=col, lw=2)
    ax.axvline(TAU, color="k", ls="--", lw=1)
    ax.text(TAU + 0.01, 80, "tau_loss = 0.1", fontsize=8)
    ax.set_xlabel("tau")
    ax.set_ylabel("% pixel được đánh dấu")
    ax.set_title("Ảnh 4x4 (3.md): bậc thang vì chỉ 16 pixel;\n0.1 nằm trên 'bậc phẳng' rộng")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Anh mo phong 48x32 voi 3 muc "chat luong" khac nhau
    ax = axes[1]
    rng = np.random.default_rng(2)
    H, W = 32, 48
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.exp(-((xx - 30) ** 2 + (yy - 18) ** 2) / (2 * 6 ** 2)) + 0.6 * np.exp(-((xx - 10) ** 2 + (yy - 8) ** 2) / (2 * 4 ** 2))
    for amp, lab, col in [(0.40, "ảnh 'tệ' (max ~0.40)", "#c0392b"), (0.16, "ảnh 'tốt' (max ~0.16)", "#2a7d2e"),
                          (0.05, "ảnh rất tốt (max ~0.05)", "#2874a6")]:
        E = 0.005 + 0.01 * rng.random((H, W)) + amp * base
        En = minmax(E)
        frac = [(En > t).mean() * 100 for t in taus]
        ax.plot(taus, frac, label=lab, color=col, lw=2)
    ax.axvline(TAU, color="k", ls="--", lw=1)
    ax.set_xlabel("tau")
    ax.set_ylabel("% pixel được đánh dấu")
    ax.set_title("Mô phỏng 48x32: sau min-max, 3 ảnh khác chất lượng\ncho đường cong gần trùng nhau")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, "1_9_tau_curve.png")


# ---------------------------------------------------------------------------
# 1_10  Nguong tuyet doi tuong duong theo view: e > min + tau*(max-min)
# ---------------------------------------------------------------------------
def fig_1_10():
    views = [("view 1\n(3.md)", 0.01, 0.40), ("view 2\n(3.md)", 0.01, 0.16), ("view 3\n(3.md)", 0.01, 0.24),
             ("view 1\n(test 48x32)", 0.006232, 0.438925), ("view 2\n(test)", 0.0, 0.441639), ("view 3\n(test)", 0.0, 0.440789)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(views))
    mins = np.array([v[1] for v in views]); maxs = np.array([v[2] for v in views])
    thr = mins + TAU * (maxs - mins)
    ax.bar(x, maxs - mins, bottom=mins, color="#d6eaf8", edgecolor="#2874a6", label="[min, max] của e_v")
    ax.scatter(x, thr, color="#c0392b", zorder=5, label="ngưỡng tuyệt đối tương đương e > min + 0.1(max-min)")
    for xi, t, mn, mx in zip(x, thr, mins, maxs):
        ax.text(xi + 0.08, t, f"{t:.4f}", va="center", fontsize=8, color="#c0392b")
        ax.text(xi, mx + 0.01, f"max {mx:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in views], fontsize=8)
    ax.set_ylabel("e_v")
    ax.set_title("Bước (2)+(3): cùng tau = 0.1 nhưng ngưỡng tuyệt đối mỗi view mỗi khác (view càng 'tệ' ngưỡng càng cao)")
    ax.set_ylim(0, 0.56)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "1_10_equiv_threshold.png")


# ---------------------------------------------------------------------------
# 1_11  Truong hop bien: max = min (chia 0), NaN > tau = False; vai tro epsilon
# ---------------------------------------------------------------------------
def fig_1_11():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))

    # (a) anh sai so hang so -> max=min
    E_const = np.full((4, 4), 0.03)
    with np.errstate(invalid="ignore", divide="ignore"):
        En_const = (E_const - E_const.min()) / (E_const.max() - E_const.min())
    M_const = (En_const > TAU)
    ax = axes[0]
    ax.imshow(np.nan_to_num(En_const, nan=0.0), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "NaN", ha="center", va="center", fontsize=8, color="white")
    ax.set_title(f"e_v hằng số 0.03 → (0/0) = NaN\nNaN > 0.1 = False → mask {M_const.sum()} px\n(không lỗi, không crash)", fontsize=8.5)
    pixel_labels(ax)

    # (b) e_hat cua pixel P15 (0.18) theo mau so khi them epsilon
    ax = axes[1]
    eps = np.logspace(-8, -1, 100)
    denom = 0.39
    e_hat = (0.18 - 0.01) / (denom + eps)
    ax.semilogx(eps, e_hat, color="#2a7d2e", lw=2)
    ax.axhline((0.18 - 0.01) / denom, color="k", ls=":", lw=0.8)
    ax.set_xlabel("epsilon thêm vào mẫu số")
    ax.set_ylabel("e_hat(P15) view 1")
    ax.set_title("Nếu thêm eps (code KHÔNG thêm):\nảnh hưởng < 1% khi eps < 1e-3", fontsize=8.5)
    ax.grid(alpha=0.25)

    # (c) mau so nho (anh gan hoi tu) -> nhieu bi phong to
    ax = axes[2]
    rng = np.random.default_rng(3)
    noise = 0.002 * rng.random((4, 4))
    E_small = 0.004 + noise
    En_small = minmax(E_small)
    ax.imshow(En_small, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    annotate(ax, En_small, color_fn=lambda v: "white" if v < 0.6 else "black")
    ax.set_title(f"e trong [0.004, 0.006] (ảnh gần hội tụ)\nmin-max vẫn kéo về [0,1]\n→ {int((En_small>TAU).sum())}/16 px 'lỗi'", fontsize=8.5)
    pixel_labels(ax)

    fig.suptitle("Trường hợp biên của bước (2): max = min cho NaN (an toàn nhờ so sánh), mẫu số nhỏ làm nhiễu trở thành 'lỗi' (giá của chuẩn hoá theo ảnh)")
    save(fig, "1_11_edge_cases.png")


# ---------------------------------------------------------------------------
# 1_12  Ban do luong du lieu buoc (1)-(2)-(3) voi shape tensor (torch ops)
# ---------------------------------------------------------------------------
def fig_1_12():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off")
    boxes = [
        (0.02, "render_image\nI_rend  [3,H,W]\n\ngt_image\nI_gt  [3,H,W]", "#d6eaf8"),
        (0.24, "torch.abs(rend - gt)\n[3,H,W]\n\ntorch.mean(., 0)\n-> l1_loss  [H,W]   (bước 1)", "#fdebd0"),
        (0.47, "(l1 - min) / (max - min)\n-> l1_loss_norm  [H,W]\ntrong [0,1]   (bước 2)\n\n(torch.min / torch.max toàn ảnh)", "#e9f7ef"),
        (0.70, "(l1_loss_norm > loss_thresh).int()\n-> metric_map  [H,W]\n{0,1}   (bước 3)\n\nloss_thresh = 0.1", "#fadbd8"),
    ]
    for x, txt, col in boxes:
        ax.add_patch(Rectangle((x, 0.15), 0.21, 0.7, facecolor=col, edgecolor="#444"))
        ax.text(x + 0.105, 0.5, txt, ha="center", va="center", fontsize=7.6, family="monospace")
    for x in [0.23, 0.46, 0.69]:
        ax.annotate("", xy=(x + 0.01, 0.5), xytext=(x - 0.0, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.02, "get_loss() trong utils/fast_utils.py:21-25  →  metric_map đi vào render_fastgs(..., get_flag=True, metric_map=...)  (bước 4, phần sau)",
            ha="center", fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Ba bước đầu của compute_gaussian_score_fastgs: shape tensor và phép torch tương ứng")
    save(fig, "1_12_torch_flow.png")


if __name__ == "__main__":
    fig_1_1()
    fig_1_2()
    fig_1_3()
    fig_1_4()
    fig_1_5()
    fig_1_6()
    fig_1_7()
    fig_1_8()
    fig_1_9()
    fig_1_10()
    fig_1_11()
    fig_1_12()
    print("done.")
