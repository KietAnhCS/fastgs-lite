"""
Hinh minh hoa cho PHAN 7 (tiep) cua chuong Adaptive Density Control:
so sanh ADC 3DGS goc vs FastGS-lite tren toy 2D (engine: part7_engine.py).

  7_4  loss / N(t) / clone-split-prune  (3 run: 3DGS, FastGS, khong ADC)
  7_5  render cuoi + sai so + ellipse   (3 hang, layout nhu fig_6_10)
  7_6  cai tien 1: AND voi Importance   (mat na loi, ellipse to mau theo nhom, bar Importance)

Chay: python adc_figures/part7b_figures.py
"""
from part7_engine import *   # noqa: F401,F403
from matplotlib.patches import Patch

C_3DGS, C_FAST, C_NO = "tab:red", "tab:blue", "tab:gray"
C_CLONE, C_SPLIT, C_PRUNE = "tab:green", "tab:red", "tab:gray"


def _final_row_title(name, p, h):
    return f"{name}\nrender cuối: N = {len(p['mu'])}, L1 = {h['loss'][-1]:.4f}"


# =============================================================================
# 7.4  Loss, N(t), clone/split/prune — 3DGS goc vs FastGS-lite
# =============================================================================
def fig_7_4():
    R = full_run_compare()
    _, h3, _ = R["3dgs"]
    _, hf, _ = R["fastgs"]
    _, hn, _ = R["no"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # (a) loss
    ax = axes[0]
    ax.semilogy(h3["loss"], color=C_3DGS, lw=1.3, label=f"ADC 3DGS gốc (L1 cuối {h3['loss'][-1]:.4f})")
    ax.semilogy(hf["loss"], color=C_FAST, lw=1.3, label=f"ADC FastGS-lite (L1 cuối {hf['loss'][-1]:.4f})")
    ax.semilogy(hn["loss"], color=C_NO, lw=1.3, label=f"không ADC, N = {hn['N'][0]} (L1 cuối {hn['loss'][-1]:.4f})")
    for (t, *_r) in h3["adc"]:
        ax.axvline(t, color="k", lw=0.5, alpha=0.25)
    ax.axvline(RESET_AT, color="tab:purple", ls=":", lw=1.2)
    ax.text(RESET_AT - 8, ax.get_ylim()[1] * 0.7, "reset α", color="tab:purple", fontsize=8, va="top", ha="right")
    for k, t in enumerate(FINAL_AT):
        ax.axvline(t, color=C_FAST, ls=":", lw=1.2)
        if k == 0:
            ax.text(t + 8, ax.get_ylim()[1] * 0.7, "final prune\n(FastGS)", color=C_FAST, fontsize=8, va="top")
    ax.set_xlabel("iteration"); ax.set_ylabel("L1 loss")
    ax.set_title("(a) Loss — vạch mờ = mốc ADC; chấm = reset α / final prune")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)

    # (b) N(t)
    ax = axes[1]
    ax.step(np.arange(len(h3["N"])), h3["N"], color=C_3DGS, where="post", lw=1.4, label="ADC 3DGS gốc")
    ax.step(np.arange(len(hf["N"])), hf["N"], color=C_FAST, where="post", lw=1.4, label="ADC FastGS-lite")
    ax.plot(hn["N"], color=C_NO, lw=1.2, label=f"không ADC, N = {hn['N'][0]}")
    ax.axvspan(ADC_FROM, ADC_UNTIL, color="tab:orange", alpha=0.08, label="cửa sổ densify")
    ax.axhline(N_MAX, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.text(5, N_MAX + 1.2, f"trần N = {N_MAX} (toy)", fontsize=7.5, color="k", alpha=0.7)
    for (t, nr) in hf["final"]:
        k1 = int(t == FINAL_AT[1])
        ax.annotate(f"final prune −{nr}", xy=(t, hf["N"][t]), xytext=(t - 330 + 130 * k1, hf["N"][t] - 32 - 14 * k1),
                    fontsize=8, color=C_FAST,
                    arrowprops=dict(arrowstyle="->", color=C_FAST, lw=1))
    ax.text(len(h3["N"]) - 5, h3["N"][-1] - 2.5, f"N = {h3['N'][-1]}", color=C_3DGS, fontsize=8, ha="right", va="top")
    ax.text(len(hf["N"]) - 5, hf["N"][-1] - 2.5, f"N = {hf['N'][-1]}", color=C_FAST, fontsize=8, ha="right", va="top")
    ax.set_xlabel("iteration"); ax.set_ylabel("N")
    ax.set_title("(b) Số Gaussian $N(t)$ — FastGS kết thúc với N ít hơn")
    ax.set_ylim(0, N_MAX + 12)
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.08)); ax.grid(alpha=0.3)

    # (c) clone/split/prune theo moc, 2 phuong phap canh nhau
    ax = axes[2]
    ts = np.array([a[0] for a in h3["adc"]])
    wdt = K_ADC * 0.13
    vals3 = np.array([[a[1], a[2], a[3]] for a in h3["adc"]])
    valsf = np.array([[a[1], a[2], a[3]] for a in hf["adc"]])
    cols = [C_CLONE, C_SPLIT, C_PRUNE]
    offs = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    for j in range(3):
        ax.bar(ts + offs[j] * wdt, vals3[:, j], width=wdt, color=cols[j], hatch="//", edgecolor="k", lw=0.4, alpha=0.75)
        ax.bar(ts + offs[3 + j] * wdt, valsf[:, j], width=wdt, color=cols[j], edgecolor="k", lw=0.4)
    # final prune cua FastGS: cot rieng (mau prune, vien xanh)
    for (t, nr) in hf["final"]:
        ax.bar(t, nr, width=2 * wdt, color=C_PRUNE, edgecolor=C_FAST, lw=1.4)
        ax.text(t, nr + 0.4, f"final\n−{nr}", ha="center", va="bottom", fontsize=7.5, color=C_FAST)
    ax.axvline((ADC_UNTIL + FINAL_AT[0]) / 2, color="k", ls=":", lw=0.8, alpha=0.5)
    handles = [Patch(facecolor=C_CLONE, label="clone"), Patch(facecolor=C_SPLIT, label="split"),
               Patch(facecolor=C_PRUNE, label="prune"),
               Patch(facecolor="w", edgecolor="k", hatch="//", label="3DGS gốc (gạch chéo)"),
               Patch(facecolor="w", edgecolor="k", label="FastGS-lite (đặc)")]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper right")
    ax.set_xticks(list(ts) + list(FINAL_AT))
    ax.set_xticklabels([str(t) for t in ts] + [str(t) for t in FINAL_AT], fontsize=7.5)
    ax.set_xlabel("mốc ADC (iteration)"); ax.set_ylabel("số Gaussian")
    ax.set_title("(c) Clone / split / prune tại mỗi mốc\n3DGS gốc (gạch chéo) và FastGS-lite (đặc) cạnh nhau")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(vals3.max(), valsf.max()) * 1.35)
    save(fig, "7_4_loss-N-clone-split-prune.png")


# =============================================================================
# 7.5  Doi chieu cuoi: 3DGS goc / FastGS-lite / khong ADC
# =============================================================================
def fig_7_5():
    R = full_run_compare()
    p3, h3, _ = R["3dgs"]
    pf, hf, _ = R["fastgs"]
    pn, hn, _ = R["no"]
    rows = [("ADC 3DGS gốc: %d → %d Gaussian" % (h3["N"][0], len(p3["mu"])), p3, h3),
            ("ADC FastGS-lite: %d → %d Gaussian\nL1 ≈ 3DGS gốc, N ít hơn (%d vs %d)"
             % (hf["N"][0], len(pf["mu"]), len(pf["mu"]), len(p3["mu"])), pf, hf),
            ("Không ADC, N cố định = %d" % len(pn["mu"]), pn, hn)]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.4))
    for r, (name, p, h) in enumerate(rows):
        I = render(p)
        show_img(axes[r, 0], I, f"{name}\nrender cuối, N = {len(p['mu'])}, L1 = {h['loss'][-1]:.4f}")
        im = show_err(axes[r, 1], I, "sai số $|I-T|$")
        show_img(axes[r, 2], TARGET * 0.3 + 0.7, f"ellipse 1.5σ (N = {len(p['mu'])})")
        draw_ellipses(axes[r, 2], p, k=1.5, colors=gauss_colors(len(p["mu"])), lw=1.2)
    for r, c in [(0, C_3DGS), (1, C_FAST), (2, C_NO)]:
        for ax in axes[r]:
            for sp in ax.spines.values():
                sp.set_edgecolor(c); sp.set_linewidth(1.6)
    fig.colorbar(im, ax=axes[:, 1], fraction=0.03)
    save(fig, "7_5_cuoi-3dgs-fastgs-khong-adc.png")


# =============================================================================
# 7.6  Cai tien 1: AND voi Importance > IMP_THR
# =============================================================================
def _pick_diag(hist, prefer=300):
    d = hist["diag"][prefer]
    if d["blocked_by_imp"].sum() > 0:
        return prefer, d
    t_best = max(hist["diag"], key=lambda t: (hist["diag"][t]["blocked_by_imp"].sum(), -t))
    return t_best, hist["diag"][t_best]


def fig_7_6():
    R = full_run_compare()
    _, hf, _ = R["fastgs"]
    t, d = _pick_diag(hf, 300)
    p = d["p_before"]
    n = len(p["mu"])
    dens3 = d["clone_3dgs"] | d["split_3dgs"]
    densf = d["clone_fastgs"] | d["split_fastgs"]
    both = dens3 & densf
    blocked = d["blocked_by_imp"]
    fonly = densf & ~dens3
    none = ~(both | blocked | fonly)
    C_BOTH, C_BLK, C_FONLY, C_NONE = "tab:green", "tab:red", "tab:orange", "0.55"
    cols = np.array([C_NONE] * n, dtype=object)
    cols[fonly] = C_FONLY; cols[blocked] = C_BLK; cols[both] = C_BOTH
    lws = np.where(none, 0.8, 1.6)

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35]))
    # (a) render truoc ADC
    show_img(axes[0], d["I_before"], f"(a) Render ngay trước mốc ADC t = {t}\nN = {n}")
    # (b) mat na loi
    ax = axes[1]
    show_img(ax, d["I_before"] * 0.55 + 0.45)
    m = d["mask"].astype(float)
    ax.imshow(np.dstack([np.ones_like(m), np.zeros_like(m), np.ones_like(m) * 0.2, m * 0.55]),
              extent=(0, W, H, 0), interpolation="nearest")
    ax.set_title(f"(b) Mặt nạ lỗi (bước ①②③): $\\hat e(x) > 0.1$\n"
                 f"{int(d['mask'].sum())} pixel lỗi ({100 * d['mask'].mean():.0f}% ảnh), đè lên render")
    # (c) ellipse to mau theo nhom
    ax = axes[2]
    show_img(ax, TARGET * 0.3 + 0.7)
    for i in np.argsort(none)[::-1]:  # ve nhom 'khong ai' truoc, nhom mau len sau
        s1, s2 = np.exp(p["logs"][i])
        e = Ellipse(p["mu"][i], 3 * s1, 3 * s2, angle=np.degrees(p["th"][i]),
                    fill=False, ec=cols[i], lw=lws[i], alpha=0.95 if not none[i] else 0.7)
        ax.add_patch(e)
        ax.plot(*p["mu"][i], ".", color=cols[i], ms=4)
        if blocked[i]:
            ax.plot(*p["mu"][i], "x", color=C_BLK, ms=11, mew=2.2)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    handles = [Line2D([], [], color=C_BOTH, lw=1.6, label=f"densify ở cả hai ({int(both.sum())})"),
               Line2D([], [], color=C_BLK, lw=1.6, marker="x", ms=8, mew=2,
                      label=f"3DGS densify, FastGS CHẶN vì Imp ≤ {IMP_THR} ({int(blocked.sum())})"),
               Line2D([], [], color=C_FONLY, lw=1.6, label=f"chỉ FastGS densify — split theo |g| ({int(fonly.sum())})"),
               Line2D([], [], color=C_NONE, lw=0.8, label=f"không ai ({int(none.sum())})")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, framealpha=0.9)
    ax.set_title("(c) Ellipse 1.5σ, tô màu theo quyết định\n(X đỏ = gradient muốn đổi nhưng render không sai)")
    # (d) bar Importance
    ax = axes[3]
    idx = np.arange(n)
    ax.bar(idx, d["imp"], color=list(cols), edgecolor="k", lw=0.3, width=0.8)
    for i in np.where(blocked)[0]:
        ax.plot(i, d["imp"][i] + 0.5, "x", color=C_BLK, ms=8, mew=2, clip_on=False)
    ax.axhline(IMP_THR, color="k", ls="--", lw=1.1)
    ax.text(n - 0.5, IMP_THR * 1.25, f"ngưỡng Importance = {IMP_THR}", ha="right", va="bottom", fontsize=8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, d["imp"].max() * 1.6)
    ax.set_yticks([0, 5, 10, 20, 50, 100, 200]); ax.set_yticklabels(["0", "5", "10", "20", "50", "100", "200"])
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_xlabel("chỉ số Gaussian $i$"); ax.set_ylabel("Importance$_i$ = số pixel lỗi trong footprint")
    ax.set_title(f"(d) Importance từng Gaussian (V = 1 view), màu như (c)\n"
                 f"{int((d['imp'] <= IMP_THR).sum())}/{n} Gaussian có Imp ≤ {IMP_THR} → không được clone/split")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Cải tiến 1 — AND với Importance>5: gradient nói 'muốn đổi', mặt nạ lỗi nói 'render có thật sự sai không'",
                 fontsize=11.5, fontweight="bold")
    save(fig, "7_6_and-importance-tren-toy.png")
    return t, int(blocked.sum())


# =============================================================================
if __name__ == "__main__":
    t_all = time.time()
    fig_7_4()
    fig_7_5()
    t6, nb = fig_7_6()
    print(f"7_6: dùng mốc t = {t6}, số Gaussian bị chặn = {nb}")
    print(f"Tổng thời gian: {time.time() - t_all:.1f}s")
