"""
Hình minh hoạ chương 4 (Tile Rasterizer) cho bài test số — chỉ matplotlib + numpy.
Dùng lại renderer trong ch04_test.py (import, không sửa) và dữ liệu sẵn:
  ch04_render_cam1.npy, ch04_render_cam1_gt.npy, ch04_aux_cam1.npz

Chạy:  python DOCS/Report/test/scripts/ch04_plot.py
Sinh:  DOCS/Report/test/figures/ch04_*.png
"""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, HERE)
import ch04_test as T4   # noqa: E402  (chỉ import hàm, main() không chạy)

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
GCOL = ["#d62728", "#2ca02c", "#1f77b4", "#7f7f7f"]     # G1..G4
GNAME = ["G1 (đỏ)", "G2 (lục)", "G3 (lam)", "G4 (xám)"]
W, H, BLK, GX, GY = T4.W, T4.H, T4.BLOCK_X, T4.GRID_X, T4.GRID_Y

img = np.load(os.path.join(HERE, "ch04_render_cam1.npy")).astype(np.float64)
gt = np.load(os.path.join(HERE, "ch04_render_cam1_gt.npy")).astype(np.float64)
aux = np.load(os.path.join(HERE, "ch04_aux_cam1.npz"))
mu2, conic, depth = aux["mu2"], aux["conic"], aux["depth"]
ranges, point_list, max_contrib = aux["ranges"], aux["point_list"], aux["max_contrib"]
final_T, n_contrib = aux["final_T"], aux["n_contrib"]
final_T_gt, n_contrib_gt = aux["final_T_gt"], aux["n_contrib_gt"]

R = T4.run(0.1)      # để lấy proj/alpha/cols cho render_pixel (trace từng pixel)
proj, alpha, cols = R["proj"], R["alpha"], R["cols"]


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[saved]", os.path.abspath(path))


def draw_tiles(ax, number=True):
    for tx in range(GX + 1):
        ax.axvline(tx * BLK - 0.5, color="k", lw=1.0, alpha=0.7)
    for ty in range(GY + 1):
        ax.axhline(ty * BLK - 0.5, color="k", lw=1.0, alpha=0.7)
    if number:
        for T in range(GX * GY):
            ty, tx = divmod(T, GX)
            ax.text(tx * BLK + 0.5, ty * BLK + 0.5, f"T{T}", fontsize=8, fontweight="bold",
                    color="k", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


# ==========================================================================
# 1. ch04_render.png — I_rend, I_gt, |diff| + lưới tile + tâm μ' + 3 pixel ví dụ
# ==========================================================================
def fig_render():
    diff = np.abs(img - gt).mean(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    panels = [
        (img, r"$I_{\mathrm{rend}}$ ($\alpha=0.1$, mô hình khởi tạo)"),
        (gt, r"$I_{\mathrm{gt}}$ ($\alpha=0.9$)"),
        (None, r"$|I_{\mathrm{rend}}-I_{\mathrm{gt}}|$ (trung bình RGB), mean=%.4f" % np.abs(img - gt).mean()),
    ]
    px_ex = [((24, 16), "(a) (24,16)"), ((25, 15), "(b) (25,15)"), ((0, 0), "(c) (0,0)")]
    for ax, (im, title) in zip(axes, panels):
        ax.grid(False)
        if im is not None:
            ax.imshow(im, interpolation="nearest", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        else:
            h = ax.imshow(diff, interpolation="nearest", cmap="magma", vmin=0, vmax=diff.max(),
                          extent=(-0.5, W - 0.5, H - 0.5, -0.5))
            cb = fig.colorbar(h, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("|diff|")
            ax.text(1, H - 1.2, f"max={diff.max():.3f}", color="white", fontsize=8, ha="left")
        draw_tiles(ax)
        for i in range(4):
            ax.plot(mu2[i, 0], mu2[i, 1], "o", ms=6, mfc=GCOL[i], mec="white", mew=1.2)
            off = {0: (-70, 22), 1: (6, -12), 2: (-82, -4), 3: (6, 4)}[i]
            ax.annotate(rf"$\mu'_{i+1}$=({mu2[i,0]:.1f},{mu2[i,1]:.1f})", (mu2[i, 0], mu2[i, 1]),
                        xytext=off, textcoords="offset points", fontsize=7, color=GCOL[i], fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=GCOL[i], lw=0.6))
        for (px, py), lab in px_ex:
            ax.add_patch(Rectangle((px - 0.5, py - 0.5), 1, 1, fill=False, ec="#ff7f0e", lw=1.6))
            offl = {(24, 16): (8, -30), (25, 15): (60, 22), (0, 0): (8, -14)}[(px, py)]
            ax.annotate(lab, (px, py), xytext=offl, textcoords="offset points",
                        fontsize=7, color="#b35900", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="#ff7f0e", lw=0.8))
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
        ax.set_xticks(range(0, W, 8)); ax.set_yticks(range(0, H, 8))
        ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
        ax.set_title(title, fontsize=10)
    fig.suptitle(r"Ảnh $48\times32$, lưới tile 16 px (T0..T5), tâm $\mu'_i$ và 3 pixel ví dụ (khung cam)", fontsize=11)
    fig.tight_layout()
    save(fig, "ch04_render.png")


# ==========================================================================
# 2. ch04_keys.png — khoá 64-bit: lưới tile×Gaussian (chưa sort) → timeline sau sort + ranges
# ==========================================================================
def fig_keys():
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.5)

    # --- (trên) lưới 6 tile × 4 Gaussian, ô ghi key hex, thứ tự ghi buffer (vị trí) ---
    ax = fig.add_subplot(gs[0]); ax.grid(False)
    d32 = [T4.bit_cast_u32(depth[i]) for i in range(4)]
    for i in range(4):
        for T in range(6):
            pos = i * 6 + T
            key = (T << 32) | d32[i]
            ax.add_patch(Rectangle((T, 3 - i), 1, 1, fc=GCOL[i], ec="white", lw=2, alpha=0.28))
            ax.text(T + 0.5, 3 - i + 0.62, f"0x{T}|{d32[i]:08X}", ha="center", va="center", fontsize=8.5,
                    family="monospace", fontweight="bold", color="k")
            ax.text(T + 0.5, 3 - i + 0.25, f"vị trí {pos}", ha="center", va="center", fontsize=7.5, color="#333")
    for i in range(4):
        ax.text(-0.15, 3 - i + 0.5, f"G{i+1}  depth={depth[i]:g}\n" + r"$\mathrm{bit\_cast}$=0x" + f"{d32[i]:08X}",
                ha="right", va="center", fontsize=8.5, color=GCOL[i], fontweight="bold")
    for T in range(6):
        ax.text(T + 0.5, 4.1, f"tile T{T}\nid≪32 = 0x{T}_00000000", ha="center", va="bottom", fontsize=7.5)
    ax.set_xlim(-3.2, 6.1); ax.set_ylim(-0.1, 4.9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(r"Bước 4.2 — $\mathrm{key}(T,i)=(\mathrm{id}(T)\ll32)\,|\,\mathrm{bit\_cast}_{u32}(\mathrm{depth}_i)$;  "
                 r"$P=\sum K_i=24$ cặp, ghi theo $\mathrm{offset}_i=(0,6,12,18)$ (chưa sort)", fontsize=10.5)

    # --- (dưới) timeline 24 ô sau sort, chia 6 đoạn tile, ranges ---
    ax2 = fig.add_subplot(gs[1]); ax2.grid(False)
    keys_s = aux["keys_sorted"]
    for idx in range(24):
        gid = int(point_list[idx]); T = int(keys_s[idx] >> np.uint64(32))
        ax2.add_patch(Rectangle((idx, 0), 1, 1, fc=GCOL[gid], ec="white", lw=1.5, alpha=0.85))
        ax2.text(idx + 0.5, 0.62, f"G{gid+1}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax2.text(idx + 0.5, 0.28, f"{depth[gid]:g}", ha="center", va="center", fontsize=7.5, color="white")
        ax2.text(idx + 0.5, -0.18, str(idx), ha="center", va="top", fontsize=7.5, color="#333")
    for T in range(6):
        a, b = ranges[T]
        ax2.plot([a + 0.05, b - 0.05], [1.18, 1.18], color="k", lw=2)
        ax2.plot([a + 0.05, a + 0.05], [1.08, 1.28], color="k", lw=1.5)
        ax2.plot([b - 0.05, b - 0.05], [1.08, 1.28], color="k", lw=1.5)
        ax2.text((a + b) / 2, 1.34, f"T{T}: ranges=[{a},{b})\n" + r"$\mathcal{G}_T$=[G1,G4,G2,G3]",
                 ha="center", va="bottom", fontsize=8)
        if T > 0:
            ax2.axvline(a, color="k", lw=2)
    ax2.text(-0.3, 0.5, "point_list\n(sau sort)", ha="right", va="center", fontsize=9)
    ax2.text(-0.3, -0.2, "idx", ha="right", va="top", fontsize=8)
    ax2.set_xlim(-3.2, 24.2); ax2.set_ylim(-0.6, 2.1)
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values(): s.set_visible(False)
    ax2.set_title("Bước 4.3 — sau radix sort 35 bit (32 + ⌈log₂6⌉): trong mỗi tile depth tăng dần "
                  "4.0 < 4.2 < 4.5 < 5.0 ⇒ [G1, G4, G2, G3]; identifyTileRanges → ranges[T]", fontsize=10.5)
    # mũi tên nối 2 phần
    fig.text(0.5, 0.455, r"$\Downarrow$  DeviceRadixSort::SortPairs (ổn định, sort theo khoá uint64)",
             ha="center", va="center", fontsize=11, fontweight="bold")
    save(fig, "ch04_keys.png")


# ==========================================================================
# 3. ch04_bitcast.png — bit_cast float→uint32 đơn điệu với depth>0; phản chứng depth âm
# ==========================================================================
def fig_bitcast():
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    # --- trái: depth > 0, phóng to ---
    xs = np.linspace(0.5, 8.0, 1500)
    ys = np.float32(xs).view(np.uint32).astype(np.float64)
    ax.plot(xs, ys, color="#1f77b4", lw=2, label=r"$\mathrm{bit\_cast}_{u32}(\mathrm{float32}(d))$, $d>0$: đơn điệu tăng")
    for d, c, dy in ((3.5, "#2ca02c", 40), (4.0, "#d62728", -45), (2.0, "#9467bd", 40)):
        u = T4.bit_cast_u32(d)
        ax.plot(d, u, "o", ms=8, color=c, mec="k", zorder=5)
        ax.annotate(f"depth={d} → 0x{u:08X}\n= {u}", (d, u), xytext=(-30, dy), textcoords="offset points",
                    fontsize=8.5, color=c, fontweight="bold", arrowprops=dict(arrowstyle="->", color=c))
    for i in range(4):
        u = T4.bit_cast_u32(depth[i])
        ax.plot(depth[i], u, "s", ms=6, color=GCOL[i], mec="k", zorder=6)
        ax.annotate(f"G{i+1}: {depth[i]:g} → 0x{u:08X}", (depth[i], u), xytext=(8, -14 - 11 * (i % 2)),
                    textcoords="offset points", fontsize=7.5, color=GCOL[i], fontweight="bold")
    for k in range(0, 4):     # ranh giới exponent: 1,2,4,8
        ax.axvline(2 ** k, color="gray", ls=":", lw=0.8)
        ax.text(2 ** k, T4.bit_cast_u32(8.0) + 6e6, f"exp={127 + k}", fontsize=7, color="gray", ha="left", va="bottom")
    ax.text(0.55, T4.bit_cast_u32(7.0), "mỗi bậc exponent: +0x00800000 = +8388608\n"
            "trong 1 bậc: mantissa tăng tuyến tính theo d\n"
            "⇒ 3.5 < 4.0 ⇔ 0x40600000 < 0x40800000 ✓\n"
            "⇒ 4 < 4.2 < 4.5 < 5 ⇔ 0x40800000 < 0x40866666 < 0x40900000 < 0x40A00000 ✓",
            fontsize=8, va="top", bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#999"))
    ax.set_xlim(0.4, 8.3)
    ax.set_ylim(T4.bit_cast_u32(0.5) - 3e6, T4.bit_cast_u32(8.0) + 2.2e7)
    ax.set_xlabel(r"depth $d$ (float32), chỉ $d>0$ nhờ cull $t_z\leq0.2$ (chương 3.2)")
    ax.set_ylabel("uint32 = 32 bit thấp của khoá")
    yt = [T4.bit_cast_u32(v) for v in (0.5, 1, 2, 4, 8)]
    ax.set_yticks(yt); ax.set_yticklabels([f"{v}\n0x{v:08X}" for v in yt], fontsize=7.5)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("depth dương: bit_cast giữ thứ tự (IEEE-754 sign|exp|mantissa)", fontsize=10.5)
    # --- phải: phản chứng depth âm, toàn thang uint32 ---
    vals = [(-1.0, "red"), (2.0, "#9467bd"), (3.5, "#2ca02c"), (4.0, "#d62728")]
    for k, (d, c) in enumerate(vals):
        u = T4.bit_cast_u32(d)
        axr.bar(k, u, 0.6, color=c, alpha=0.85, edgecolor="k")
        axr.text(k, u + 4e7, f"depth={d}\n0x{u:08X}\n{u:,}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=c)
    axr.axhline(2 ** 31, color="red", ls="--", lw=1.2)
    axr.text(3.3, 2 ** 31 + 3e7, r"$2^{31}$ = 0x80000000 (bit dấu)", ha="right", va="bottom", fontsize=8, color="red")
    axr.set_xticks(range(4)); axr.set_xticklabels(["−1.0\n(âm)", "2.0", "3.5", "4.0"])
    axr.set_ylim(0, 4.0e9); axr.ticklabel_format(axis="y", style="plain")
    axr.set_ylabel("uint32")
    axr.set_title("phản chứng: −1.0 < 2.0 nhưng 0xBF800000 > 0x40000000\n⇒ sort uint32 xếp −1.0 SAU 2.0 (sai front-to-back)",
                  fontsize=9.5, color="red")
    axr.text(1.5, 3.75e9, r"không xảy ra trong pipeline vì đã cull $t_z\leq0.2$", ha="center", fontsize=8,
             bbox=dict(boxstyle="round", fc="#fff3f3", ec="red"))
    fig.tight_layout()
    save(fig, "ch04_bitcast.png")


# ==========================================================================
# 4. ch04_blend_pixel.png — T_n và đóng góp từng Gaussian tại (24,16), (25,15); ví dụ 3 Gaussian
# ==========================================================================
def trace(px, py):
    tid = (py // BLK) * GX + (px // BLK)
    C, T, last, cnt, done, rows = T4.render_pixel(px, py, tid, R["ranges"], R["vals_s"], proj, alpha, cols)
    return C, T, rows


def blend_panel(axT, axC, rows, C_final, T_final, title, names, colors_bar):
    n = len(rows)
    Ts = [1.0] + [r.get("T_next", r["T"]) for r in rows]
    x = np.arange(n + 1)
    axT.step(x, Ts, where="post", color="k", lw=2)
    axT.plot(x, Ts, "ko", ms=4)
    for k, t in enumerate(Ts):
        axT.annotate(f"{t:.4g}", (k, t), xytext=(5, 6), textcoords="offset points", ha="left", va="bottom", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    axT.set_xticks(x); axT.set_xticklabels([f"$T_{k+1}$" if k < n else r"$T_{final}$" for k in range(n + 1)])
    axT.set_ylim(0, 1.15); axT.set_ylabel("$T_n$ (transmittance)")
    axT.set_title(title, fontsize=10)
    # bar đóng góp
    chans = ["R", "G", "B"]; chan_col = ["#d62728", "#2ca02c", "#1f77b4"]
    width = 0.24
    bottoms = np.zeros(3)
    for k, r in enumerate(rows):
        contrib = np.asarray(r["contrib"])
        a = r["alpha"]
        xs = np.arange(3) * 1.0
        axC.bar(xs, contrib, width * 3.2, bottom=bottoms, color=colors_bar[k], edgecolor="white", lw=0.8,
                label=f"{names[k]}: α={a:.4g}, T={r['T']:.4g} → c·α·T=({contrib[0]:.3g},{contrib[1]:.3g},{contrib[2]:.3g})")
        bottoms += contrib
    axC.bar(np.arange(3), T_final * np.ones(3), width * 3.2, bottom=bottoms, color="white", edgecolor="k",
            hatch="//", lw=1, label=rf"$T_{{final}}\cdot C_{{bg}}$ = {T_final:.4g}·(1,1,1)")
    tops = bottoms + T_final
    for c in range(3):
        axC.text(c, tops[c] + 0.015, f"{C_final[c]:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        axC.text(c, bottoms[c] / 2, f"Σ={bottoms[c]:.4f}", ha="center", va="center", fontsize=7.5, color="white",
                 bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45))
    axC.set_xticks(range(3)); axC.set_xticklabels([f"kênh {ch}" for ch in chans])
    for lab, cc in zip(axC.get_xticklabels(), chan_col):
        lab.set_color(cc); lab.set_fontweight("bold")
    axC.set_ylim(0, 1.12); axC.set_ylabel(r"$C=\sum c_n\alpha_nT_n+T_{final}C_{bg}$")
    axC.legend(fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, -0.42), framealpha=0.9)
    axC.set_title(f"C = ({C_final[0]:.3f}, {C_final[1]:.3f}, {C_final[2]:.3f})", fontsize=10)


def fig_blend():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2), gridspec_kw=dict(height_ratios=[0.8, 1.25]))
    for j, (px, py) in enumerate([(24, 16), (25, 15)]):
        C, T, rows = trace(px, py)
        tid = (py // BLK) * GX + (px // BLK)
        names = [f"G{r['gid']+1}" for r in rows]
        colors_bar = [GCOL[r["gid"]] for r in rows]
        blend_panel(axes[0, j], axes[1, j], rows, C, T,
                    rf"pixel ({px},{py}), tile T{tid}, $\mathcal{{G}}_T$=[{', '.join(names)}]", names, colors_bar)
    # ví dụ 3 Gaussian trong chương 4.5
    ex_c = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float); ex_a = [0.5, 0.4, 0.6]
    Tt = 1.0; rows = []
    for n in range(3):
        contrib = ex_c[n] * ex_a[n] * Tt
        rows.append(dict(T=Tt, alpha=ex_a[n], contrib=contrib, T_next=Tt * (1 - ex_a[n])))
        Tt *= (1 - ex_a[n])
    Cex = sum(r["contrib"] for r in rows) + Tt
    blend_panel(axes[0, 2], axes[1, 2], rows, Cex, Tt,
                "ví dụ chương 4.5: (đỏ,0.5), (lục,0.4), (lam,0.6), nền trắng",
                ["đỏ", "lục", "lam"], ["#d62728", "#2ca02c", "#1f77b4"])
    fig.suptitle(r"Alpha-blending front-to-back: $C \leftarrow C+c_n\alpha_nT_n$, $T_{n+1}=T_n(1-\alpha_n)$, cuối cùng $C \leftarrow C+T_{final}C_{bg}$",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "ch04_blend_pixel.png")


# ==========================================================================
# 5. ch04_gates.png — α_n(x) dọc hàng y=16; ngưỡng 1/255; T_final(x), n_contrib(x)
# ==========================================================================
def fig_gates():
    y = 16
    xs = np.arange(W)
    xf = np.linspace(-0.5, W - 0.5, 600)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    thr = 1.0 / 255.0
    order = [0, 3, 1, 2]        # G1, G4, G2, G3 (thứ tự trong G_T)
    for i in order:
        A, B, Cc = conic[i]
        dx = mu2[i, 0] - xf; dy = mu2[i, 1] - y
        power = -0.5 * (A * dx * dx + Cc * dy * dy) - B * dx * dy
        a = np.minimum(0.99, alpha[i] * np.exp(power))
        ax1.plot(xf, a, color=GCOL[i], lw=2, label=rf"G{i+1}: $\alpha_{{{i+1}}}(x)$, max={a.max():.4g} tại x≈{xf[a.argmax()]:.1f}")
        below = a < thr
        ax1.fill_between(xf, 1e-5, a, where=below, color=GCOL[i], alpha=0.15)
    # vùng bị cửa α<1/255 (bất kỳ Gaussian) — gạch chéo ở dưới ngưỡng
    ax1.axhspan(1e-5, thr, color="red", alpha=0.07, hatch="///", ec="red", lw=0)
    ax1.axhline(thr, color="red", ls="--", lw=1.5, label=r"cửa 2: $\alpha_n(x)<1/255=0.003922$ → bỏ splat")
    ax1.axhline(0.1, color="k", ls=":", lw=1, label=r"$\alpha_i=0.1$ (đỉnh lý thuyết khi $x=\mu'_i$)")
    ax1.text(0.2, thr * 0.55, "vùng bị cửa 2 (α<1/255): splat bị bỏ, T không đổi", color="red", fontsize=8.5, va="top")
    ax1.text(W - 0.8, 0.13, "cửa 1 (power>0): không bao giờ xảy ra vì $-\\frac{1}{2}\\Delta^\\top M\\Delta\\leq0$\n"
             "cửa 3 ($T(1-\\alpha)<10^{-4}$): 0 pixel với 4 Gaussian, α=0.1", ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#999"))
    for i in range(4):
        ax1.axvline(mu2[i, 0], color=GCOL[i], ls="-.", lw=0.8, alpha=0.6)
    ax1.set_yscale("log"); ax1.set_ylim(1e-4, 0.3)
    ax1.set_ylabel(r"$\alpha_n(x)=\min(0.99,\alpha_ie^{\mathrm{power}})$  (log)")
    ax1.set_title(rf"Hàng pixel $v={y}$ (qua tâm $\mu'_1$=(23.5,15.5)), $u=0..47$: alpha từng Gaussian và cửa loại", fontsize=10.5)
    ax1.legend(fontsize=7.8, loc="lower center", ncol=2)
    # T_final và n_contrib dọc hàng
    Trow = final_T[y]; nrow = n_contrib[y]
    ax2.step(xs, Trow, where="mid", color="k", lw=2, label=rf"$T_{{final}}(x)$: min={Trow.min():.4f} tại u={Trow.argmin()}, max={Trow.max():.4f}")
    ax2.set_ylabel(r"$T_{final}$"); ax2.set_ylim(0.6, 1.05)
    ax2b = ax2.twinx(); ax2b.grid(False)
    ax2b.bar(xs, nrow, width=0.85, color="#ff7f0e", alpha=0.35, label="n_contrib(x) = last_contributor")
    ax2b.set_ylim(0, 5); ax2b.set_ylabel("n_contrib", color="#b35900")
    for x0 in (0, 8, 16, 24, 32, 40, 47):
        ax2.text(x0, Trow[x0] + 0.012, f"{Trow[x0]:.3f}", ha="center", fontsize=7.5)
        ax2b.text(x0, nrow[x0] + 0.1, str(int(nrow[x0])), ha="center", fontsize=7.5, color="#b35900")
    for tx in range(1, GX):
        ax2.axvline(tx * BLK - 0.5, color="k", lw=1, alpha=0.5)
        ax1.axvline(tx * BLK - 0.5, color="k", lw=1, alpha=0.5)
    for T in range(3, 6):
        ax2.text((T - 3) * BLK + 8, 1.03, f"tile T{T} ({ranges[T,0]}..{ranges[T,1]-1})", ha="center", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax2.set_xlabel("u (px)"); ax2.set_xlim(-0.5, W - 0.5); ax2.set_xticks(range(0, W, 4))
    ax2.set_title(rf"$T_{{final}}$ và n_contrib dọc hàng $v={y}$ (α=0.1; các tile T3, T4, T5)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "ch04_gates.png")


# ==========================================================================
# 6. ch04_aux_maps.png — heatmap final_T, n_contrib (α=0.1, 0.9) + bar #bucket, max_contrib
# ==========================================================================
def fig_aux():
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.75], hspace=0.38, wspace=0.28)
    maps = [
        (final_T, r"final_T ($\alpha=0.1$)", "viridis", (0, 1)),
        (final_T_gt, r"final_T ($\alpha=0.9$, GT)", "viridis", (0, 1)),
        (n_contrib, r"n_contrib ($\alpha=0.1$)", "Oranges", (0, 4)),
        (n_contrib_gt, r"n_contrib ($\alpha=0.9$, GT)", "Oranges", (0, 4)),
    ]
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (m, title, cmap, (vmin, vmax)), (r, c) in zip(maps, pos):
        ax = fig.add_subplot(gs[r, c]); ax.grid(False)
        h = ax.imshow(m.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                      extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        fig.colorbar(h, ax=ax, fraction=0.03, pad=0.02)
        draw_tiles(ax)
        for i in range(4):
            ax.plot(mu2[i, 0], mu2[i, 1], "o", ms=5, mfc=GCOL[i], mec="white", mew=1)
        extra = ""
        if m.dtype.kind in "ui":
            extra = f"  |  #pixel=0 (nền): {int((m == 0).sum())}"
        ax.set_title(f"{title}\nmin={m.min():.4g}  max={m.max():.4g}  mean={m.mean():.4g}{extra}", fontsize=9.5)
        ax.set_xticks(range(0, W, 8)); ax.set_yticks(range(0, H, 8))
    # bar #bucket và max_contrib per tile
    ax = fig.add_subplot(gs[0, 2])
    Ts = np.arange(6); sizes = ranges[:, 1] - ranges[:, 0]; nb = (sizes + 31) // 32
    ax.bar(Ts - 0.2, sizes, 0.4, color="#1f77b4", label=r"$|\mathcal{G}_T|$ = ranges[T].y − ranges[T].x")
    ax.bar(Ts + 0.2, nb, 0.4, color="#ff7f0e", label=r"#bucket$_T$ = ⌈$|\mathcal{G}_T|$/32⌉")
    for T in Ts:
        ax.text(T - 0.2, sizes[T] + 0.08, str(int(sizes[T])), ha="center", fontsize=8)
        ax.text(T + 0.2, nb[T] + 0.08, str(int(nb[T])), ha="center", fontsize=8)
    ax.set_xticks(Ts); ax.set_xticklabels([f"T{T}" for T in Ts]); ax.set_ylim(0, 5.4)
    ax.set_title(f"perTileBucketCount — tổng bucket = {int(nb.sum())} warp backward", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right")
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(Ts, max_contrib, 0.55, color="#9467bd")
    for T in Ts:
        ax.text(T, max_contrib[T] + 0.08, str(int(max_contrib[T])), ha="center", fontsize=8)
    ax.set_xticks(Ts); ax.set_xticklabels([f"T{T}" for T in Ts]); ax.set_ylim(0, 5.4)
    ax.set_title("max_contrib[T] = BlockReduce Max(last_contributor) trong tile (α=0.1)", fontsize=9.5)
    fig.suptitle("Dữ liệu lưu cho backward (mục 4.6, 4.7): final_T, n_contrib, #bucket, max_contrib", fontsize=12)
    save(fig, "ch04_aux_maps.png")


if __name__ == "__main__":
    fig_render()
    fig_keys()
    fig_bitcast()
    fig_blend()
    fig_gates()
    fig_aux()
