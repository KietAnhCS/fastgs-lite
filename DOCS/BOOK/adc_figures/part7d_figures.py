"""
PHAN 7 (d): hai cai tien cuoi cua FastGS-lite tren toy 2D (engine part7_engine.py, cache part7_cache.pkl)

  7_9_tran-opacity-08.png       : Cai tien 4 — tran opacity α ← min(α, 0.8) sau moi ADC
  7_10_final-prune-tren-toy.png : Cai tien 5 — final prune tai t ∈ FINAL_AT = (1000, 1100)
"""
from part7_engine import *   # noqa: F401,F403
from matplotlib.gridspec import GridSpec

C3, CF = "tab:blue", "tab:orange"          # mau 3DGS goc / FastGS-lite


# =============================================================================
# 7.9  Tran opacity 0.8
# =============================================================================
def _alpha_hist(ax, a3, af):
    """Histogram α tren truc log cua (1-α) — dan nhan theo α de doc truc tiep."""
    x3, xf = 1 - a3, 1 - af
    lo = max(min(x3.min(), xf.min()) * 0.8, 1e-3)
    bins = np.logspace(np.log10(lo), np.log10(0.5), 26)
    ax.hist(x3, bins=bins, color=C3, alpha=0.55, label=f"3DGS gốc (N = {len(a3)})")
    ax.hist(xf, bins=bins, color=CF, alpha=0.55, label=f"FastGS-lite (N = {len(af)})")
    ax.hist(x3, bins=bins, histtype="step", color=C3, lw=1.2)
    ax.hist(xf, bins=bins, histtype="step", color=CF, lw=1.2)
    ax.set_xscale("log"); ax.invert_xaxis()
    ticks = [0.5, 0.8, 0.9, 0.95, 0.99, 0.997]
    ax.set_xticks([1 - t for t in ticks]); ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xticks([], minor=True)
    ax.axvline(1 - 0.8, color="k", ls="--", lw=1.2)
    ax.axvline(1 - 0.99, color="tab:red", ls=":", lw=1.2)
    ymax = ax.get_ylim()[1]
    ax.text(1 - 0.8, ymax * 0.98, " trần 0.8 ", ha="right", va="top", fontsize=8)
    ax.text(1 - 0.99, ymax * 0.98, " 0.99 ", ha="left", va="top", fontsize=8, color="tab:red")
    n3, nf = int((a3 > 0.8).sum()), int((af > 0.8).sum())
    n3h, nfh = int((a3 > 0.99).sum()), int((af > 0.99).sum())
    ax.text(0.03, 0.80, f"α > 0.8 : {n3} / {len(a3)}\nα > 0.99: {n3h} / {len(a3)}", transform=ax.transAxes,
            color=C3, fontsize=8.5, va="top",
            bbox=dict(fc="w", ec=C3, alpha=0.85, boxstyle="round,pad=0.25"))
    ax.text(0.03, 0.58, f"α > 0.8 : {nf} / {len(af)}\nα > 0.99: {nfh} / {len(af)}", transform=ax.transAxes,
            color=CF, fontsize=8.5, va="top",
            bbox=dict(fc="w", ec=CF, alpha=0.85, boxstyle="round,pad=0.25"))
    ax.set_xlabel("opacity α cuối (trục log theo 1−α, phải = đặc hơn)")
    ax.set_ylabel("số Gaussian")
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.0, 0.86))
    ax.grid(alpha=0.3, axis="y")
    return n3, nf, n3h, nfh


def fig_7_9():
    R = full_run_compare()
    p3, h3, _ = R["3dgs"]
    pf, hf, _ = R["fastgs"]
    a3, af = sigmoid(p3["lo"]), sigmoid(pf["lo"])
    I3, ch3 = render(p3, return_cache=True)
    If, chf = render(pf, return_cache=True)
    T3 = ch3["Tend"].reshape(H, W); Tf = chf["Tend"].reshape(H, W)
    # chi xet pixel co vat the (T_end lon o nen la binh thuong: nen la BG khong co Gaussian)
    obj = np.abs(TARGET - BG).sum(-1) > 0.05
    m3, mf = T3[obj].mean(), Tf[obj].mean()
    f3, ff = (T3[obj] < 0.01).mean(), (Tf[obj] < 0.01).mean()

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.0))
    n3, nf, n3h, nfh = _alpha_hist(axes[0], a3, af)
    axes[0].set_title("(a) Phân bố opacity cuối (t = %d)\ntrần 0.8 áp tại mỗi ADC (đến t = %d),\nsau đó α được tự tăng lại"
                      % (T_TOTAL, ADC_UNTIL), fontsize=9.5)

    kw = dict(extent=(0, W, H, 0), cmap="viridis", vmin=0, vmax=0.3, interpolation="nearest")
    im = axes[1].imshow(T3, **kw)
    axes[1].set_title(f"(b) 3DGS gốc: $T_{{end}}(x)$ còn lại sau mọi Gaussian\n"
                      f"trong vật thể (viền trắng): mean = {m3:.3f}\n{f3*100:.0f}% pixel có $T_{{end}}$ < 0.01 (bị chặn hết)",
                      fontsize=9.5)
    axes[2].imshow(Tf, **kw)
    axes[2].set_title(f"(c) FastGS-lite: $T_{{end}}(x)$\n"
                      f"trong vật thể: mean = {mf:.3f}\n{ff*100:.0f}% pixel có $T_{{end}}$ < 0.01", fontsize=9.5)
    for ax in axes[1:3]:
        ax.set_xticks([]); ax.set_yticks([])
        ax.contour(np.arange(W) + 0.5, np.arange(H) + 0.5, obj.astype(float), levels=[0.5],
                   colors="w", linewidths=0.6, alpha=0.7)
    cb = fig.colorbar(im, ax=axes[1:3], fraction=0.025, pad=0.02)
    cb.set_label("$T_{end}$ (tối = ánh sáng bị chặn hết)")

    # (d) minh hoa 1D
    ax = axes[3]
    x = np.linspace(-4, 4, 400)
    Gf = np.exp(-0.5 * x ** 2)                     # Gaussian phia truoc, σ = 1
    Gb = np.exp(-0.5 * ((x - 0.8) / 1.3) ** 2)     # Gaussian phia sau (chi de ve nen)
    ax.fill_between(x, 0, Gb, color="tab:green", alpha=0.12, label="$G_{back}(x)$ (Gaussian phía sau)")
    ax.fill_between(x, 0, Gf, color="tab:gray", alpha=0.15, label="$G_{front}(x)$ (Gaussian phía trước)")
    for a, col in ((0.99, "tab:red"), (0.8, "tab:blue")):
        T = 1 - a * Gf
        ax.plot(x, T, color=col, lw=2.2, label=f"$T = 1 - {a}\\,G_{{front}}(x)$")
        ax.annotate(f"T(0) = {1-a:.2f}", xy=(0, 1 - a), xytext=(1.6, 1 - a + (0.12 if a == 0.99 else 0.08)),
                    fontsize=8.5, color=col, arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
    ax.set_ylim(0, 1.05); ax.set_xlim(-4, 4)
    ax.set_xlabel("x (đơn vị σ của Gaussian trước)")
    ax.set_ylabel("T mà Gaussian phía sau 'nhìn thấy'")
    ax.set_title("(d) 1D: gradient của Gaussian phía sau ∝ T\n"
                 "α_front = 0.99 → T(0) = 0.01\nα_front = 0.8 → T(0) = 0.20  (gấp 20×)", fontsize=9.5)
    ax.text(0.5, 0.50, "∂L/∂(θ_back) ∝ T·α_back·G_back\nα_front = 0.8 giữ 20% ánh sáng\ncho lớp sau → vẫn học được",
            transform=ax.transAxes, ha="center", fontsize=8.5,
            bbox=dict(fc="lightyellow", ec="k", lw=0.6, boxstyle="round,pad=0.35"))
    ax.legend(loc="lower left", fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle("Cải tiến 4 — trần opacity 0.8: Gaussian phía sau vẫn nhận gradient qua T, không bị 'chôn' vĩnh viễn",
                 fontsize=12)
    save(fig, "7_9_tran-opacity-08.png")
    print(f"  7.9: α>0.8  3DGS {n3}/{len(a3)}, FastGS {nf}/{len(af)} | α>0.99  3DGS {n3h}, FastGS {nfh}")
    print(f"  7.9: T_end mean (toàn ảnh) 3DGS {T3.mean():.3f}, FastGS {Tf.mean():.3f}; "
          f"trên vật thể 3DGS {m3:.3f}, FastGS {mf:.3f}")


# =============================================================================
# 7.10  Final prune
# =============================================================================
def fig_7_10():
    R = full_run_compare()
    p3, h3, _ = R["3dgs"]
    pf, hf, _ = R["fastgs"]
    fd = hf["final_diag"]
    ts = sorted(fd.keys())

    fig = plt.figure(figsize=(17, 8.6))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.15, 1, 1, 1.05])
    stats = {}
    for r, t in enumerate(ts):
        d = fd[t]
        a, pr, rm = d["alpha"], d["pruning"], d["removed"]
        n_b, n_a = len(a), len(d["p_after"]["mu"])
        dI = np.abs(d["I_after"] - d["I_before"]).max()
        l1b = np.abs(d["I_before"] - TARGET).mean(); l1a = np.abs(d["I_after"] - TARGET).mean()
        n_alpha, n_pr = int((a < FINAL_ALPHA).sum()), int((pr > FINAL_PRUNE_THR).sum())
        stats[t] = dict(n_b=n_b, n_a=n_a, rm=int(rm.sum()), dI=dI, l1b=l1b, l1a=l1a, n_alpha=n_alpha, n_pr=n_pr)

        # --- cot 1: scatter (α, Pruning) ---
        ax = fig.add_subplot(gs[r, 0])
        ax.axvspan(0, FINAL_ALPHA, color="tab:red", alpha=0.13, lw=0)
        ax.axhspan(FINAL_PRUNE_THR, 1.05, color="tab:orange", alpha=0.18, lw=0)
        ax.axvline(FINAL_ALPHA, color="tab:red", ls="--", lw=1); ax.axhline(FINAL_PRUNE_THR, color="tab:orange", ls="--", lw=1)
        ax.scatter(a[~rm], pr[~rm], s=22, color="tab:gray", alpha=0.75, label=f"giữ ({int((~rm).sum())})", zorder=3)
        ax.scatter(a[rm], pr[rm], s=70, marker="x", color="tab:red", lw=1.8, label=f"xoá ({int(rm.sum())})", zorder=4)
        ax.text(FINAL_ALPHA / 2, 0.5, f"α < {FINAL_ALPHA}\n({n_alpha})", ha="center", fontsize=8, color="tab:red")
        ax.text(0.55, 0.975, f"Pruning > {FINAL_PRUNE_THR}  ({n_pr})", ha="center", va="center", fontsize=8, color="tab:orange")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("opacity α"); ax.set_ylabel("Pruning score (min-max của counts)")
        ax.set_title(f"t = {t}: vùng chữ L bị xoá — [α < {FINAL_ALPHA}] ∨ [Pruning > {FINAL_PRUNE_THR}]\n"
                     f"N = {n_b}: xoá {int(rm.sum())} (α thấp: {n_alpha}, Pruning cao: {n_pr})")
        ax.legend(loc="center right", fontsize=8); ax.grid(alpha=0.3)

        # --- cot 2: I_before + ellipse do dut ---
        ax = fig.add_subplot(gs[r, 1])
        show_img(ax, d["I_before"], f"TRƯỚC final prune (t = {t}): N = {n_b} → {n_a}\nL1 = {l1b:.4f}; ellipse đỏ đứt = Gaussian bị xoá")
        pb = d["p_before"]
        keep_p = {k: pb[k][~rm] for k in KEYS}; rm_p = {k: pb[k][rm] for k in KEYS}
        draw_ellipses(ax, keep_p, k=1.5, colors=["0.5"] * len(keep_p["mu"]), lw=0.6, alpha=0.35)
        draw_ellipses(ax, rm_p, k=1.5, colors=["red"] * len(rm_p["mu"]), lw=1.8, ls="--", alpha=1.0)

        # --- cot 3: I_after ---
        ax = fig.add_subplot(gs[r, 2])
        show_img(ax, d["I_after"], f"SAU final prune: N = {n_a}\nmax|ΔI| = {dI:.3f}; L1 {l1b:.4f} → {l1a:.4f}")
        draw_ellipses(ax, rm_p, k=1.5, colors=["red"] * len(rm_p["mu"]), lw=0.9, ls=":", alpha=0.8)

    # --- cot 4 (span 2 hang): N(t) ---
    ax = fig.add_subplot(gs[:, 3])
    t0, t1 = 900, T_TOTAL
    tt = np.arange(t0, t1 + 1)
    ax.step(tt, h3["N"][t0:t1 + 1], where="post", color=C3, lw=2.2, label="3DGS gốc (không có final prune)")
    ax.step(tt, hf["N"][t0:t1 + 1], where="post", color=CF, lw=2.2, label="FastGS-lite")
    for t in ts:
        ax.axvline(t, color="k", ls=":", lw=0.9)
        s = stats[t]
        ax.annotate(f"final prune t = {t}\n{s['n_b']} → {s['n_a']}  (−{s['rm']})",
                    xy=(t, s["n_a"]), xytext=(t + 12, s["n_a"] - 3.2), fontsize=8.5, color=CF,
                    arrowprops=dict(arrowstyle="->", color=CF, lw=0.9))
    ax.axvline(ADC_UNTIL, color="tab:green", ls="--", lw=1)
    ax.text(ADC_UNTIL + 3, max(h3["N"][t0:t1 + 1]) + 0.6, f"ADC cuối\n(t = {ADC_UNTIL})", fontsize=8, color="tab:green")
    ax.set_xlim(t0, t1); ax.set_ylim(min(hf["N"][t0:t1 + 1]) - 6, max(h3["N"][t0:t1 + 1]) + 4)
    ax.set_xlabel("iteration"); ax.set_ylabel("N (số Gaussian)")
    ax.set_title("N(t) trong [900, 1200]\n3DGS phẳng; FastGS hai bậc xuống")
    ax.legend(loc="lower left", fontsize=8.5); ax.grid(alpha=0.3)
    ax.text(0.97, 0.90, f"L1 cuối (t = {T_TOTAL}):\n3DGS {h3['loss'][-1]:.4f} (N = {len(p3['mu'])})\n"
            f"FastGS {hf['loss'][-1]:.4f} (N = {len(pf['mu'])})",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(fc="lightyellow", ec="k", lw=0.6, boxstyle="round,pad=0.35"))

    fig.suptitle("Cải tiến 5 — final prune sau cửa sổ densify: xoá [α<0.1] ∨ [Pruning>0.9], 3DGS gốc không có bước này",
                 fontsize=12)
    save(fig, "7_10_final-prune-tren-toy.png")
    for t, s in stats.items():
        print(f"  7.10 t={t}: N {s['n_b']}→{s['n_a']}, xoá {s['rm']} (α<0.1: {s['n_alpha']}, Pr>0.9: {s['n_pr']}), "
              f"max|ΔI| = {s['dI']:.3f}, L1 {s['l1b']:.4f} → {s['l1a']:.4f}")


if __name__ == "__main__":
    fig_7_9()
    fig_7_10()
