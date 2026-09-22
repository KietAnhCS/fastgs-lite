"""
Hinh minh hoa cho PHAN 3 cua chuong 12 (Adaptive Density Control):
buoc (6) photometric loss toan anh, buoc (7) Pruning score, tong hop 7 buoc,
do nhay, chi phi.

Moi ham fig_3_k() xuat dung 1 file PNG `3_k_ten-ngan.png` vao cung thu muc.
Chi dung numpy + matplotlib + scipy.ndimage (khong torch).

Chay: python adc_figures/part3_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.ndimage import convolve, gaussian_filter

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

LAMBDA = 0.2


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved:", path)


# ---------------------------------------------------------------------------
# Du lieu goc (3.md: anh 4x4, 5 Gaussian, 3 view) va canh do choi (48x32, 4 G)
# ---------------------------------------------------------------------------
G_NAMES = ["G1", "G2", "G3", "G4", "G5"]
COUNTS_3MD = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [6, 6, 6],
                       [4, 4, 3],
                       [5, 0, 0]], dtype=float)          # 5 x 3
L1_3MD = np.array([0.111250, 0.060000, 0.090625])
SSIM_3MD = np.array([0.72, 0.88, 0.80])

# canh do choi (12.2, output that cua ch07_test.py)
L1_TOY = np.array([0.259278, 0.222367, 0.204733])
SSIM_TOY = np.array([0.647773, 0.702127, 0.704586])
COUNTS_TOY = np.array([[1248, 954, 930],
                       [1181, 1003, 869],
                       [1245, 978, 977],
                       [1180, 973, 770]], dtype=float)


def e_photo(l1, ssim, lam=LAMBDA):
    return (1 - lam) * l1 + lam * (1 - ssim)


def minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min())


def scores(counts, E, V=None):
    """counts: N x V, E: V.  Tra ve (importance, pruning, raw)."""
    V = counts.shape[1] if V is None else V
    imp = np.floor(counts.sum(1) / V)
    raw = (counts * E[None, :]).sum(1)
    return imp, minmax(raw), raw


# ---------------------------------------------------------------------------
# SSIM numpy (giong utils/loss_utils.py: cua so gauss 11x11 sigma 1.5, zero-pad)
# ---------------------------------------------------------------------------
def _gauss_window(size=11, sigma=1.5):
    g = np.array([np.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) for x in range(size)])
    g = g / g.sum()
    return np.outer(g, g)


def ssim_np(img1, img2, size=11, sigma=1.5):
    """img: (C,H,W) trong [0,1]. Tra ve mean SSIM (nhu size_average=True)."""
    w = _gauss_window(size, sigma)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    maps = []
    for c in range(img1.shape[0]):
        a, b = img1[c], img2[c]
        f = lambda x: convolve(x, w, mode="constant", cval=0.0)
        mu1, mu2 = f(a), f(b)
        s1 = f(a * a) - mu1 ** 2
        s2 = f(b * b) - mu2 ** 2
        s12 = f(a * b) - mu1 * mu2
        m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        maps.append(m)
    return float(np.mean(maps))


# ---------------------------------------------------------------------------
# Mo phong 300 Gaussian, V = 10 view
# ---------------------------------------------------------------------------
def simulate_population(N=300, V=10, seed=0):
    rng = np.random.default_rng(seed)
    # footprint huu hinh (so pixel) — log-normal, 0 khi bi che
    vis = rng.random((N, V)) < rng.uniform(0.3, 1.0, size=(N, 1))     # xac suat huu hinh
    omega = np.exp(rng.normal(4.0, 0.8, size=(N, V))).astype(int) + 1  # ~50-300 px
    omega = omega * vis
    # ti le pixel loi trong footprint: moi Gaussian co "do sai" rieng, moi view nhieu them
    p_err = np.clip(rng.beta(1.2, 4.0, size=(N, 1)) + rng.normal(0, 0.05, size=(N, V)), 0, 1)
    counts = rng.binomial(omega, p_err).astype(float)
    # E_photo moi view: view "te" co L1 lon, SSIM thap
    l1 = rng.uniform(0.03, 0.18, size=V)
    ssim = np.clip(1 - l1 * rng.uniform(1.5, 3.0, size=V), 0, 1)
    E = e_photo(l1, ssim)
    return counts, E, l1, ssim


# ===========================================================================
# 3_1  Bar L1 / (1-SSIM) / E_photo cho 3 view — 3.md va canh do choi
# ===========================================================================
def fig_3_1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, l1, ssim, title in [(axes[0], L1_3MD, SSIM_3MD, "Ví dụ 3.md (ảnh 4x4, SSIM cho sẵn)"),
                                (axes[1], L1_TOY, SSIM_TOY, "Cảnh đồ chơi 12.2 (48x32, SSIM tính thật)")]:
        E = e_photo(l1, ssim)
        x = np.arange(3)
        wdt = 0.2
        ax.bar(x - 1.5 * wdt, l1, wdt, label="L1", color="#4c72b0")
        ax.bar(x - 0.5 * wdt, 1 - ssim, wdt, label="1 - SSIM", color="#dd8452")
        ax.bar(x + 0.5 * wdt, (1 - LAMBDA) * l1, wdt, label="0.8 L1", color="#8fb3e0")
        ax.bar(x + 0.5 * wdt, LAMBDA * (1 - ssim), wdt, bottom=(1 - LAMBDA) * l1,
               label="0.2 (1-SSIM)", color="#f2b98a")
        ax.bar(x + 1.5 * wdt, E, wdt, label="E_photo = tổng", color="#55a868", edgecolor="k")
        for i in range(3):
            ax.text(x[i] + 1.5 * wdt, E[i] + 0.005, f"{E[i]:.4f}", ha="center", fontsize=7.5)
            ax.text(x[i] - 1.5 * wdt, l1[i] + 0.005, f"{l1[i]:.3f}", ha="center", fontsize=7)
            ax.text(x[i] - 0.5 * wdt, 1 - ssim[i] + 0.005, f"{1 - ssim[i]:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"view {v + 1}" for v in range(3)])
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("giá trị")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.suptitle("Bước (6): E_photo^(v) = 0.8 L1 + 0.2 (1 - SSIM) — cột xanh lá = trọng số của view trong bước (7)")
    save(fig, "3_1_ephoto-3view.png")


# ===========================================================================
# 3_2  Cung L1, khac SSIM: blur vs noise
# ===========================================================================
def _base_image(H=64, W=64, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] / H
    img = np.zeros((3, H, W))
    # nen gradient + hai vat the co canh sac + soc
    img[0] = 0.35 + 0.3 * xx
    img[1] = 0.4 + 0.2 * yy
    img[2] = 0.5 - 0.2 * xx
    box = (xx > 0.2) & (xx < 0.55) & (yy > 0.25) & (yy < 0.6)
    img[:, box] = np.array([0.85, 0.25, 0.2])[:, None]
    circ = (xx - 0.72) ** 2 + (yy - 0.7) ** 2 < 0.03
    img[:, circ] = np.array([0.2, 0.3, 0.9])[:, None]
    stripes = ((xx * 40).astype(int) % 2 == 0) & (yy > 0.8)
    img[:, stripes] = 0.9
    img += rng.normal(0, 0.01, img.shape)
    return np.clip(img, 0, 1)


def _match_l1_noise(base, target_l1, seed=2):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, base.shape)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        img = np.clip(base + mid * noise, 0, 1)
        l1 = np.abs(img - base).mean()
        if l1 < target_l1:
            lo = mid
        else:
            hi = mid
    return np.clip(base + 0.5 * (lo + hi) * noise, 0, 1)


def fig_3_2():
    base = _base_image()
    blur = np.stack([gaussian_filter(base[c], 1.6) for c in range(3)])
    l1_blur = np.abs(blur - base).mean()
    noise = _match_l1_noise(base, l1_blur)
    l1_noise = np.abs(noise - base).mean()
    s_blur, s_noise = ssim_np(blur, base), ssim_np(noise, base)
    E_blur, E_noise = e_photo(l1_blur, s_blur), e_photo(l1_noise, s_noise)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    for ax, im, t in [(axes[0, 0], base, "GT"),
                      (axes[0, 1], blur, f"Render A: mờ (blur)\nL1={l1_blur:.4f}  SSIM={s_blur:.3f}  E={E_blur:.4f}"),
                      (axes[0, 2], noise, f"Render B: nhiễu (noise)\nL1={l1_noise:.4f}  SSIM={s_noise:.3f}  E={E_noise:.4f}")]:
        ax.imshow(np.transpose(im, (1, 2, 0)))
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    for ax, im, t in [(axes[1, 1], blur, "e(x) của A"), (axes[1, 2], noise, "e(x) của B")]:
        err = np.abs(im - base).mean(0)
        h = ax.imshow(err, cmap="magma", vmin=0, vmax=0.25)
        ax.set_title(t + f"  (mean = {err.mean():.4f})", fontsize=9)
        ax.axis("off")
    fig.colorbar(h, ax=axes[1, 1:], shrink=0.8, label="|render - GT| trung bình 3 kênh")
    ax = axes[1, 0]
    ax.bar([0, 1], [l1_blur, l1_noise], 0.35, label="L1", color="#4c72b0")
    ax.bar([0.4, 1.4], [1 - s_blur, 1 - s_noise], 0.35, label="1 - SSIM", color="#dd8452")
    ax.set_xticks([0.2, 1.2])
    ax.set_xticklabels(["A: mờ", "B: nhiễu"])
    ax.set_title("Cùng L1, 1-SSIM khác nhau", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Vì sao bước (6) cần SSIM: L1 không phân biệt 'mờ' với 'nhiễu', SSIM thì có")
    save(fig, "3_2_l1-vs-ssim.png")
    return dict(l1_blur=l1_blur, l1_noise=l1_noise, s_blur=s_blur, s_noise=s_noise)


# ===========================================================================
# 3_3  E_photo theo lambda cho 3 view (3.md va toy) — cac duong co cat nhau?
# ===========================================================================
def fig_3_3():
    lam = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, l1, ssim, title in [(axes[0], L1_3MD, SSIM_3MD, "3.md"), (axes[1], L1_TOY, SSIM_TOY, "cảnh đồ chơi")]:
        for v in range(3):
            ax.plot(lam, e_photo(l1[v], ssim[v], lam), label=f"view {v + 1}: L1={l1[v]:.3f}, SSIM={ssim[v]:.2f}")
        ax.axvline(LAMBDA, color="k", ls="--", lw=1)
        ax.text(LAMBDA + 0.01, ax.get_ylim()[1] * 0.97, "lambda = 0.2 (code)", va="top", fontsize=8)
        ax.set_xlabel("lambda")
        ax.set_ylabel("E_photo^(v)")
        ax.set_title(f"E_photo(lambda) = (1-lambda) L1 + lambda (1-SSIM) — {title}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
        # ti so E_1/E_2
        E0 = e_photo(l1, ssim, 0.0)
        E1 = e_photo(l1, ssim, 1.0)
        ax.text(0.02, 0.05, f"tỉ số max/min: lambda=0 → {E0.max() / E0.min():.2f}   lambda=1 → {E1.max() / E1.min():.2f}",
                transform=ax.transAxes, fontsize=8, bbox=dict(fc="w", ec="0.7"))
    save(fig, "3_3_ephoto-theo-lambda.png")


# ===========================================================================
# 3_4  Heatmap dong gop counts * E cua tung view vao pruning score
# ===========================================================================
def fig_3_4():
    E = e_photo(L1_3MD, SSIM_3MD)
    contrib = COUNTS_3MD * E[None, :]
    raw = contrib.sum(1)
    pr = minmax(raw)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1, 1.3, 0.9]))
    # counts
    ax = axes[0]
    h = ax.imshow(COUNTS_3MD, cmap="Blues")
    for i in range(5):
        for j in range(3):
            ax.text(j, i, f"{int(COUNTS_3MD[i, j])}", ha="center", va="center", fontsize=10,
                    color="w" if COUNTS_3MD[i, j] > 4 else "k")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"view {v + 1}\nE={E[v]:.4f}" for v in range(3)])
    ax.set_yticks(range(5)); ax.set_yticklabels(G_NAMES)
    ax.set_title("counts_i^(v)  (bước 4)")
    # contrib
    ax = axes[1]
    h = ax.imshow(contrib, cmap="Oranges")
    for i in range(5):
        for j in range(3):
            ax.text(j, i, f"{contrib[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="w" if contrib[i, j] > 0.6 else "k")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"view {v + 1}" for v in range(3)])
    ax.set_yticks(range(5)); ax.set_yticklabels(G_NAMES)
    ax.set_title("counts_i^(v) * E_photo^(v)  (đóng góp từng view)")
    fig.colorbar(h, ax=ax, shrink=0.8)
    # raw + pruning
    ax = axes[2]
    y = np.arange(5)[::-1]
    ax.barh(y, raw, color="#dd8452", label="thô = tổng hàng")
    for i in range(5):
        ax.text(raw[i] + 0.03, y[i], f"{raw[i]:.4f} → Pruning {pr[i]:.3f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(G_NAMES)
    ax.set_xlim(0, 3.7)
    ax.set_xlabel("tổng có trọng số (trước minmax)")
    ax.set_title("bước 7: minmax qua 5 Gaussian")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "3_4_heatmap-dong-gop.png")


# ===========================================================================
# 3_5  Scatter Importance vs Pruning: 5 Gaussian 3.md + 300 mo phong
# ===========================================================================
def fig_3_5():
    E3 = e_photo(L1_3MD, SSIM_3MD)
    imp3, pr3, _ = scores(COUNTS_3MD, E3)
    counts, E, _, _ = simulate_population()
    imp, pr, raw = scores(counts, E)
    rho = np.corrcoef(np.argsort(np.argsort(imp)), np.argsort(np.argsort(pr)))[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(imp3, pr3, s=90, c="#c44e52", zorder=3)
    for i in range(5):
        ax.annotate(G_NAMES[i], (imp3[i], pr3[i]), xytext=(6, 4), textcoords="offset points", fontsize=10)
    ax.axvline(5, color="r", ls="--", lw=1, label="Importance > 5 (densify)")
    ax.axhline(0.9, color="k", ls="--", lw=1, label="Pruning > 0.9 (final prune)")
    ax.set_xlabel("Importance_i = floor(mean_v counts)")
    ax.set_ylabel("Pruning_i (minmax qua N)")
    ax.set_title("5 Gaussian của 3.md (V = 3)")
    ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.text(0.05, 1.03, "G3: góc trên-phải = vừa được clone (đầu) vừa bị xoá (cuối)", fontsize=8)

    ax = axes[1]
    vis_frac = (counts > 0).mean(1)
    sc = ax.scatter(imp, pr, s=14, c=vis_frac, cmap="viridis", alpha=0.85)
    fig.colorbar(sc, ax=ax, label="tỉ lệ view thấy Gaussian (|Omega| > 0)")
    ax.axvline(5, color="r", ls="--", lw=1)
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.set_xlabel("Importance_i")
    ax.set_ylabel("Pruning_i")
    ax.set_title(f"300 Gaussian mô phỏng, V = 10 (Spearman rho = {rho:.3f})")
    ax.grid(alpha=0.3)
    fig.suptitle("Importance và Pruning cùng tăng nhưng KHÔNG phải hàm của nhau: cùng Importance, Pruning trải rộng")
    save(fig, "3_5_scatter-imp-pruning.png")
    return rho


# ===========================================================================
# 3_6  Vi sao Pruning != 1 - Importance (va cung khong = Importance chuan hoa)
# ===========================================================================
def fig_3_6():
    # 3 truong hop cung tong counts = 18 nhung phan bo khac nhau qua 3 view
    E = e_photo(L1_3MD, SSIM_3MD)   # 0.145, 0.072, 0.1125
    cases = {
        "A: đều\n6/6/6": np.array([6, 6, 6.]),
        "B: dồn view tệ\n18/0/0": np.array([18, 0, 0.]),
        "C: dồn view tốt\n0/18/0": np.array([0, 18, 0.]),
        "D: 17/0/0\n(tổng 17)": np.array([17, 0, 0.]),
    }
    names = list(cases)
    C = np.stack([cases[k] for k in names])
    imp = np.floor(C.sum(1) / 3)
    raw = (C * E[None, :]).sum(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, imp, 0.38, color="#4c72b0", label="Importance = floor(sum/3)")
    ax.bar(x + 0.2, raw, 0.38, color="#dd8452", label="thô = sum_v counts * E_v")
    for i in range(len(names)):
        ax.text(x[i] - 0.2, imp[i] + 0.05, f"{int(imp[i])}", ha="center", fontsize=9)
        ax.text(x[i] + 0.2, raw[i] + 0.05, f"{raw[i]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_title("Cùng tổng counts → cùng Importance, nhưng 'thô' khác nhau")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    # right: raw as function of which view the counts fall into
    ax = axes[1]
    counts, Es, _, _ = simulate_population()
    imp_s, pr_s, raw_s = scores(counts, Es)
    # nhom theo Importance, ve khoang Pruning
    levels = np.unique(imp_s)
    for lv in levels[:25]:
        sel = pr_s[imp_s == lv]
        ax.plot([lv, lv], [sel.min(), sel.max()], color="0.6", lw=2)
        ax.scatter([lv] * len(sel), sel, s=8, color="#c44e52")
    ax.plot(levels[:25], [np.mean(pr_s[imp_s == lv]) for lv in levels[:25]], "k-", lw=1, label="trung bình")
    ax.set_xlabel("Importance_i (mô phỏng 300 G)")
    ax.set_ylabel("Pruning_i")
    ax.set_title("Cùng Importance, Pruning trải một khoảng rộng (thanh xám)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Pruning không phải 1 - Importance: nó giữ thêm thông tin 'sai ở view nào' và bỏ floor")
    save(fig, "3_6_pruning-vs-1-minus-imp.png")


# ===========================================================================
# 3_7  So do khoi 7 buoc voi shape tensor
# ===========================================================================
def fig_3_7():
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#eaf2fb"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="0.3", lw=1)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9, weight="bold")
        ax.text(x + w / 2, y + 0.12, body, ha="center", va="bottom", fontsize=7.6, family="monospace")

    def arrow(x0, y0, x1, y1, label, dy=0.12, color="0.25"):
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2)
        ax.add_patch(a)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=7.2, color=color,
                family="monospace")

    # hang tren: render + (1)(2)(3)
    box(0.2, 4.4, 2.2, 1.4, "render view v", "render_fastgs\n[3,H,W] + GT [3,H,W]\nO(N_vis*pix)", fc="#f7f7f7")
    arrow(2.4, 5.1, 3.0, 5.1, "[3,H,W] x2")
    box(3.0, 4.4, 2.0, 1.4, "(1) sai số pixel", "e_v = mean_ch |.|\n[H,W]\nO(HW)")
    arrow(5.0, 5.1, 5.6, 5.1, "[H,W]")
    box(5.6, 4.4, 2.0, 1.4, "(2) minmax ảnh", "(e-min)/(max-min)\n[H,W] in [0,1]\nO(HW)")
    arrow(7.6, 5.1, 8.2, 5.1, "[H,W]")
    box(8.2, 4.4, 2.0, 1.4, "(3) mặt nạ", "m_v = 1[e_hat > 0.1]\n[H,W] int\nO(HW)")
    arrow(10.2, 5.1, 10.8, 5.1, "[H,W]")
    box(10.8, 4.4, 1.6, 1.4, "(4) đổ về G", "DENSIFY=True\natomicAdd\n[N] int\nO(sum|Omega|)")

    # (6) tu render
    box(0.2, 2.3, 2.2, 1.3, "(6) loss toàn ảnh", "0.8 L1 + 0.2(1-SSIM)\nscalar E_v\nO(HW*121)", fc="#fde9d9")
    arrow(1.3, 4.4, 1.3, 3.6, "[3,H,W] x2", dy=-0.05)

    # tich luy qua V view
    box(4.2, 2.3, 4.2, 1.3, "tích luỹ qua V = 10 view (vòng for trong compute_gaussian_score_fastgs)",
        "full_metric_counts += counts_v        [N]\nfull_metric_score  += E_v * counts_v  [N]", fc="#eef7ea")
    arrow(11.6, 4.4, 8.4, 3.6, "counts_v [N]", dy=0.1)
    arrow(2.4, 2.95, 4.2, 2.95, "E_v scalar", dy=0.05)

    # (5) va (7)
    box(2.2, 0.3, 3.4, 1.3, "(5) Importance", "floor(full_counts / V)\n[N] int  — O(N)")
    box(7.0, 0.3, 3.6, 1.3, "(7) Pruning", "minmax_i(full_score)\n[N] float in [0,1] — O(N)", fc="#fde9d9")
    arrow(5.2, 2.3, 4.0, 1.6, "[N] int", dy=0.05)
    arrow(7.4, 2.3, 8.6, 1.6, "[N] float", dy=0.05)
    ax.text(3.9, 0.05, "-> densify AND (Importance > 5)", ha="center", fontsize=8, color="#2a6f2a")
    ax.text(8.8, 0.05, "-> w_i = 1/(1e-6 + 1 - Pruning), final prune > 0.9", ha="center", fontsize=8, color="#a03a2a")
    ax.text(6.25, 6.0, "Luồng dữ liệu 7 bước: ảnh [3,H,W] → pixel [H,W] → Gaussian [N] (mỗi view), rồi gộp qua V view",
            ha="center", fontsize=10, weight="bold")
    ax.text(6.25, 5.85, "màu xanh: chạy trên GPU trong kernel/torch; đỏ: bước dùng E_photo; xanh lá: bộ tích luỹ",
            ha="center", fontsize=8, color="0.4")
    save(fig, "3_7_so-do-7-buoc.png")


# ===========================================================================
# 3_8  Do nhay theo lambda: Pruning cua 5 G (3.md) khi lambda thay doi + bo trong so
# ===========================================================================
def fig_3_8():
    lams = np.linspace(0, 1, 51)
    P = np.zeros((len(lams), 5))
    for k, lam in enumerate(lams):
        E = e_photo(L1_3MD, SSIM_3MD, lam)
        P[k] = scores(COUNTS_3MD, E)[1]
    # khong trong so: E = 1
    P_nw = minmax(COUNTS_3MD.sum(1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for i in range(5):
        ax.plot(lams, P[:, i], label=G_NAMES[i], lw=2)
        ax.scatter([1.05], [P_nw[i]], s=30, marker="s")
        ax.text(1.07, P_nw[i], f"{G_NAMES[i]} bỏ w {P_nw[i]:.3f}", fontsize=7, va="center")
    ax.axvline(LAMBDA, color="k", ls="--", lw=1); ax.text(0.21, 0.5, "lambda=0.2", fontsize=8)
    ax.axhline(0.9, color="0.4", ls=":", lw=1)
    ax.set_xlim(0, 1.45); ax.set_xlabel("lambda   (ô vuông: bỏ trọng số, E_v = 1)")
    ax.set_ylabel("Pruning_i")
    ax.set_title("Pruning của 5 G (3.md) theo lambda")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.45, 0.78), ncol=5)
    # right: rank-change bump chart across schemes — them G6 gia dinh (0/6/0: chi sai o view TOT nhat)
    C6 = np.vstack([COUNTS_3MD, [[0, 6, 0]]])
    names6 = G_NAMES + ["G6*"]
    schemes = {}
    for lam in [0.0, 0.2, 0.5, 1.0]:
        schemes[f"lam={lam}"] = scores(C6, e_photo(L1_3MD, SSIM_3MD, lam))[1]
    schemes["bỏ trọng số"] = minmax(C6.sum(1))
    raw02 = (C6 * e_photo(L1_3MD, SSIM_3MD)[None, :]).sum(1)
    schemes["chia max"] = raw02 / raw02.max()
    schemes["z-score"] = (raw02 - raw02.mean()) / raw02.std()
    keys = list(schemes)
    ax = axes[1]
    for i in range(6):
        ranks = [int(np.sum(schemes[k] > schemes[k][i])) + 1 for k in keys]   # 1 = cao nhat
        ax.plot(range(len(keys)), ranks, "-o", label=names6[i], lw=2.5 if i in (4, 5) else 1.2,
                ls="-" if i != 5 else "--")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=20, fontsize=8)
    ax.set_yticks(range(1, 7)); ax.invert_yaxis()
    ax.set_ylabel("hạng (1 = Pruning cao nhất)")
    ax.set_title("Thứ hạng 5 G + G6* giả định (0/6/0: chỉ sai ở view 2, view tốt nhất)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Độ nhạy: lambda chỉ đổi giá trị; bỏ trọng số E_v đảo hạng G5 ↔ G6*; chuẩn hoá khác không đổi hạng")
    save(fig, "3_8_do-nhay-lambda-5g.png")
    return schemes


# ===========================================================================
# 3_9  Rank-change tren 300 G mo phong: lambda / bo trong so / chuan hoa
# ===========================================================================
def fig_3_9():
    counts, E, l1, ssim = simulate_population()
    base = scores(counts, E)[1]
    N = counts.shape[0]
    variants = {
        "lambda=0 (chỉ L1)": scores(counts, e_photo(l1, ssim, 0.0))[1],
        "lambda=0.5": scores(counts, e_photo(l1, ssim, 0.5))[1],
        "lambda=1 (chỉ D-SSIM)": scores(counts, e_photo(l1, ssim, 1.0))[1],
        "bỏ trọng số (E_v=1)": minmax(counts.sum(1)),
        "E_v thay bằng rank(E_v)": scores(counts, np.argsort(np.argsort(E)) + 1.0)[1],
    }
    rank_base = np.argsort(np.argsort(-base))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    ax = axes[0]
    ax.scatter(np.sort(E), np.sort(E) * 0 + 0, s=1, alpha=0)  # placeholder de giu khung
    order = np.argsort(E)
    ax.bar(range(len(E)), E[order], color="#dd8452")
    ax.set_xticks(range(len(E))); ax.set_xticklabels([f"v{o + 1}" for o in order], fontsize=7)
    ax.set_title(f"E_photo của 10 view mô phỏng (max/min = {E.max() / E.min():.2f})", fontsize=9)
    ax.set_ylabel("E_v")
    for ax, (name, var) in zip(axes[1:], variants.items()):
        rank_var = np.argsort(np.argsort(-var))
        delta = rank_var - rank_base
        ax.scatter(rank_base, rank_var, s=8, alpha=0.7, c="#4c72b0")
        ax.plot([0, N], [0, N], "k--", lw=0.8)
        top = base > 0.9
        ax.scatter(rank_base[top], rank_var[top], s=18, c="#c44e52", label="Pruning>0.9 ở gốc")
        moved = np.mean(np.abs(delta) > 0.05 * N) * 100
        kept = np.mean((var > 0.9) == top) * 100
        ax.set_title(f"{name}\n{moved:.0f}% đổi hạng >5% N; tập >0.9 trùng {kept:.1f}%", fontsize=8.5)
        ax.set_xlabel("hạng gốc (lambda=0.2)"); ax.set_ylabel("hạng mới")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Độ nhạy trên 300 Gaussian: xếp hạng thay đổi ra sao khi đổi lambda / bỏ trọng số / đổi chuẩn hoá")
    save(fig, "3_9_rank-change-300g.png")


# ===========================================================================
# 3_10  Chi phi tinh diem so: stacked bar theo densification_interval
# ===========================================================================
def fig_3_10():
    F, U, total = 500, 15000, 30000
    intervals = [500, 200, 100]
    n_dens = [len([i for i in range(1, U) if i > F and i % I == 0]) for I in intervals]
    fwd_per_iter = 3.0           # 1 vong train ~ 3 forward-equivalent
    extra_fwd = [n * 20 for n in n_dens]     # 10 view x 2 render
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = np.arange(len(intervals))
    train_units = total * fwd_per_iter
    ax.bar(x, [train_units] * 3, 0.55, color="#9fb8d3", label="train 30k vòng (x3 forward-eq)")
    ax.bar(x, extra_fwd, 0.55, bottom=[train_units] * 3, color="#c44e52", label="render phụ của scoring (10 view x 2)")
    for i in range(3):
        pct = extra_fwd[i] / train_units * 100
        ax.text(x[i], train_units + extra_fwd[i] + 1500, f"n_densify={n_dens[i]}\n+{extra_fwd[i]} fwd\n= +{pct:.1f}%",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"interval {I}" for I in intervals])
    ax.set_ylabel("forward-equivalent")
    ax.set_ylim(0, 135000)
    ax.set_title("Chi phí scoring so với cả quá trình train\n(sát thực tế, 1 vòng = 3 fwd)")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    Is = np.arange(50, 1001, 10)
    nd = np.array([len([i for i in range(1, U) if i > F and i % I == 0]) for I in Is])
    ax.plot(Is, nd * 20 / total * 100, label="cận trên (1 render = 1 vòng)", color="#c44e52")
    ax.plot(Is, nd * 20 / (total * 3) * 100, label="sát thực tế (1 vòng = 3 fwd)", color="#4c72b0")
    for I, n in zip(intervals, n_dens):
        ax.scatter([I], [n * 20 / total * 100], color="#c44e52", zorder=3)
        ax.scatter([I], [n * 20 / (total * 3) * 100], color="#4c72b0", zorder=3)
        ax.text(I + 10, n * 20 / total * 100 + 0.3, f"{n * 20 / total * 100:.1f}%", fontsize=8)
    ax.set_xlabel("densification_interval"); ax.set_ylabel("chi phí phụ (%)")
    ax.set_title("Chi phí phụ (overhead) ~ 1/interval\n(V = 10 view, 2 render mỗi view)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "3_10_chi-phi-scoring.png")
    return n_dens


# ===========================================================================
# 3_11  Tu Pruning den w_i (log) va hai nguong 0.9 / w
# ===========================================================================
def fig_3_11():
    p = np.linspace(0, 1, 1001)
    w = 1 / (1e-6 + 1 - p)
    E3 = e_photo(L1_3MD, SSIM_3MD)
    _, pr3, _ = scores(COUNTS_3MD, E3)
    pr_toy = np.array([0.7654, 0.4507, 1.0, 0.0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogy(p, w, color="#4c72b0", lw=2)
    ax.axvline(0.9, color="k", ls="--", lw=1, label="final prune > 0.9")
    for i in range(5):
        ax.scatter([pr3[i]], [1 / (1e-6 + 1 - pr3[i])], color="#c44e52", zorder=3)
        ax.annotate(G_NAMES[i] + " (3.md)", (pr3[i], 1 / (1e-6 + 1 - pr3[i])), xytext=(5, 3),
                    textcoords="offset points", fontsize=7.5, color="#c44e52")
    for i in range(4):
        ax.scatter([pr_toy[i]], [1 / (1e-6 + 1 - pr_toy[i])], color="#55a868", marker="s", zorder=3)
    ax.set_xlabel("Pruning_i"); ax.set_ylabel("w_i = 1/(1e-6 + 1 - Pruning_i)  [log]")
    ax.set_title("Trọng số multinomial (7.4): phân kỳ khi Pruning → 1")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.text(0.03, 2e5, "chấm đỏ: 3.md, ô vuông xanh: cảnh đồ chơi", fontsize=8)
    ax = axes[1]
    # xac suat bi rut o luot dau (p_i = w_i / sum w) cho 3.md
    w3 = 1 / (1e-6 + 1 - pr3)
    ax.bar(G_NAMES, w3 / w3.sum(), color="#c44e52")
    for i in range(5):
        ax.text(i, w3[i] / w3.sum() + 0.01, f"{w3[i] / w3.sum():.2e}", ha="center", fontsize=8)
    ax.set_ylabel("p_i = w_i / sum w (lượt rút đầu)")
    ax.set_title("3.md: G3 chiếm ~100% xác suất lượt đầu")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "3_11_pruning-to-w.png")


# ===========================================================================
# 3_12  Anh huong cua ban than chuan hoa min-max qua N: outlier keo ca quan the
# ===========================================================================
def fig_3_12():
    counts, E, _, _ = simulate_population()
    raw = (counts * E[None, :]).sum(1)
    pr = minmax(raw)
    raw_out = raw.copy()
    raw_out[np.argmax(raw_out)] *= 3.0      # mot Gaussian sai rat nang xuat hien
    pr_out = minmax(raw_out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.hist(pr, bins=40, alpha=0.7, label=f"gốc: {np.sum(pr > 0.9)} G > 0.9, {np.sum(pr > 0.5)} G > 0.5")
    ax.hist(pr_out, bins=40, alpha=0.7, label=f"1 outlier x3: {np.sum(pr_out > 0.9)} G > 0.9, {np.sum(pr_out > 0.5)} G > 0.5")
    ax.axvline(0.9, color="k", ls="--", lw=1)
    ax.set_xlabel("Pruning_i"); ax.set_ylabel("số Gaussian")
    ax.set_title("min-max qua N: một outlier kéo cả quần thể về gần 0")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.scatter(pr, pr_out, s=8)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.axhline(0.9, color="k", ls=":", lw=1); ax.axvline(0.9, color="k", ls=":", lw=1)
    ax.set_xlabel("Pruning gốc"); ax.set_ylabel("Pruning khi có outlier")
    ax.set_title("Thứ hạng giữ nguyên (đơn điệu), ngưỡng tuyệt đối 0.9 thì không")
    ax.grid(alpha=0.3)
    save(fig, "3_12_minmax-outlier.png")
    return int(np.sum(pr > 0.9)), int(np.sum(pr_out > 0.9)), int(np.sum(pr > 0.5)), int(np.sum(pr_out > 0.5))


if __name__ == "__main__":
    fig_3_1()
    r2 = fig_3_2()
    fig_3_3()
    fig_3_4()
    rho = fig_3_5()
    fig_3_6()
    fig_3_7()
    sch = fig_3_8()
    fig_3_9()
    nd = fig_3_10()
    fig_3_11()
    r12 = fig_3_12()
    print("--- so lieu de ghi vao markdown ---")
    print("fig_3_2:", {k: round(v, 4) for k, v in r2.items()})
    print("fig_3_5 spearman:", round(rho, 3))
    print("fig_3_8 schemes:")
    for k, v in sch.items():
        print("   ", k, np.round(v, 3))
    print("fig_3_10 n_densify:", nd)
    print("fig_3_12 (>0.9 goc, >0.9 outlier, >0.5 goc, >0.5 outlier):", r12)
