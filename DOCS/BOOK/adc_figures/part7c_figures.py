"""
PHAN 7 (c): hai cai tien "vi mo" cua FastGS so voi ADC 3DGS goc, minh hoa tren toy 2D
(engine part7_engine.py, cache part7_cache.pkl).

  7_7_abs-vs-signed-split.png       : Cai tien 2 — split theo Σ|g| (gradient tri tuyet doi)
  7_8_prune-hard-vs-multinomial.png : Cai tien 3 — prune cung (3DGS) vs prune co trong so (FastGS)
"""
from part7_engine import *   # noqa: F401,F403
from part7_engine import _xx, _yy   # noqa

N_ADAM = 80          # so buoc Adam sau khi (khong) split trong hinh 7.7
SPLIT_SEED = SEED + 77


def _copy(p):
    return {k: p[k].copy() for k in KEYS}


def _adam_run(p, n_steps):
    st = adam_init(p)
    for _ in range(n_steps):
        _, g, _, _ = loss_and_grad(p)
        adam_step(p, g, st)
    return st


def _pick_abs_only():
    """Chon (t, i*) trong fastgs: Gaussian to ma FastGS split nho |g| nhung 3DGS bo qua.
    Uu tien Gaussian khong 'qua to' (smax ≤ S_HUGE) de phong to duoc; diem = (gabs/τ_abs)/(gbar/τ)."""
    R = full_run_compare()
    diag = R["fastgs"][1]["diag"]
    best = None
    for t in sorted(diag):
        d = diag[t]
        idx = np.where(d["split_only_abs"])[0]
        for i in idx:
            s_i = np.exp(d["p_before"]["logs"][i])
            if s_i.max() > S_HUGE or s_i.min() < 1.5:
                continue          # bo Gaussian qua to / qua det (kho nhin khi phong to)
            score = (d["gabs"][i] / TAU_ABS) / max(d["gbar"][i] / TAU_GRAD, 1e-9)
            if best is None or score > best[0]:
                best = (score, t, int(i))
    if best is not None:
        return best[1], best[2]
    # du phong: Gaussian to co ti so gabs/gbar lon nhat
    for t in sorted(diag):
        d = diag[t]
        big = d["smax"] > S_BIG
        if big.any():
            r = np.where(big, d["gabs"] / np.maximum(d["gbar"], 1e-12), -1)
            return t, int(r.argmax())
    raise RuntimeError("khong tim duoc Gaussian to")


# =============================================================================
# 7.7  Split theo Σ|g| thay cho Σg
# =============================================================================
def fig_7_7():
    R = full_run_compare()
    t_sel, i_star = _pick_abs_only()
    d = R["fastgs"][1]["diag"][t_sel]
    p0 = _copy(d["p_before"])
    n = len(p0["mu"])
    loss0, g, I0, ch = loss_and_grad(p0)
    gpx, gpy = g["pix"]
    gsum = np.linalg.norm(g["mu"][i_star]); gabs_now = np.linalg.norm(g["mu_abs"][i_star])
    gbar_i, gabs_i = d["gbar"][i_star], d["gabs"][i_star]
    mu = p0["mu"][i_star]; s = np.exp(p0["logs"][i_star])
    r_zoom = max(3.0 * s.max(), 10.0)
    zoom = dict(xlim=(max(0, mu[0] - r_zoom), min(W, mu[0] + r_zoom)),
                ylim=(min(H, mu[1] + r_zoom), max(0, mu[1] - r_zoom)))

    # (c) khong split: 80 buoc Adam tu trang thai p_before
    p_no = _copy(p0)
    _adam_run(p_no, N_ADAM)
    loss_no, _, I_no, _ = loss_and_grad(p_no)
    # (d) split i*: do_split (seed co dinh) roi 80 buoc Adam
    p_sp = _copy(p0)
    st = adam_init(p_sp)
    rng = np.random.default_rng(SPLIT_SEED)
    do_split(p_sp, st, np.array([i_star]), rng)
    loss_sp0, _, _, _ = loss_and_grad(p_sp)
    child_idx = np.arange(len(p_sp["mu"]) - 2, len(p_sp["mu"]))   # 2 con nam cuoi
    p_sp1 = _copy(p_sp)
    st = adam_init(p_sp)
    for _ in range(N_ADAM):
        _, gg, _, _ = loss_and_grad(p_sp); adam_step(p_sp, gg, st)
    loss_sp, _, I_sp, _ = loss_and_grad(p_sp)

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.6))
    # ---- (a) gradient pixel trong footprint cua i* ----
    ax = axes[0]
    show_img(ax, TARGET * 0.35 + 0.65)
    ax.set_title(f"(a) t = {t_sel}, Gaussian i* = {i_star}: $s$ = ({s[0]:.1f}, {s[1]:.1f}) px, Imp = {int(d['imp'][i_star])}\n"
                 f"‖Σ$_x g_x$‖ = {gsum*1e5:.1f}×10⁻⁵  vs  Σ$_x$|$g_x$| = {gabs_now*1e5:.0f}×10⁻⁵ (gấp {gabs_now/gsum:.0f} lần)\n"
                 f"tích luỹ: ‖ḡ‖ = {gbar_i/TAU_GRAD:.2f}τ < τ → 3DGS bỏ qua\n"
                 f"‖ḡ$_{{abs}}$‖ = {gabs_i/TAU_ABS:.2f}τ$_{{abs}}$ ≥ τ$_{{abs}}$ → FastGS split", fontsize=9)
    others = [j for j in range(n) if j != i_star]
    draw_ellipses(ax, {k: p0[k][others] for k in KEYS}, k=1.5, colors=["0.55"] * len(others), lw=0.6, alpha=0.6)
    draw_ellipses(ax, {k: p0[k][[i_star]] for k in KEYS}, k=2, colors=["tab:orange"], lw=2.2)
    gx = gpx[i_star].reshape(H, W); gy = gpy[i_star].reshape(H, W)
    fp = (ch["G"][i_star] > np.exp(-0.5 * 4.0)).reshape(H, W)       # trong ellipse 2σ
    step = 2
    mag = np.sqrt(gx ** 2 + gy ** 2)
    sc = 1.6 / (mag[fp].max() + 1e-12)
    m = fp[::step, ::step]
    ax.quiver((_xx[::step, ::step] + 0.5)[m], (_yy[::step, ::step] + 0.5)[m],
              (-gx[::step, ::step] * sc)[m], (-gy[::step, ::step] * sc)[m],
              color="tab:red", angles="xy", scale_units="xy", scale=1, width=0.005, alpha=0.9)
    # mui ten tong (co dau), phong dai de thay
    tot = -g["mu"][i_star]; tot = tot / (np.linalg.norm(tot) + 1e-12) * 0.18 * r_zoom
    ax.annotate("", xy=mu + tot, xytext=mu, arrowprops=dict(arrowstyle="-|>", color="k", lw=2.2))
    ax.set(**zoom)
    ax.text(zoom["xlim"][0] + 0.4, zoom["ylim"][0] - 0.4,
            f"đỏ: −g từng pixel (kéo về nhiều phía, triệt tiêu nhau)\nđen: hướng −Σg; độ lớn chỉ bằng 1/{gabs_now/gsum:.0f} tổng |g|",
            fontsize=8, va="bottom", bbox=dict(fc="w", ec="none", alpha=0.85))

    # ---- (b) bar ‖ḡ‖ va ‖ḡ_abs‖ cho moi Gaussian to ----
    ax = axes[1]
    big = np.where(d["smax"] > S_BIG)[0]
    x = np.arange(len(big)); wd = 0.4
    c_sig = ["tab:orange" if j == i_star else "tab:blue" for j in big]
    c_abs = ["tab:orange" if j == i_star else "tab:green" for j in big]
    ax.bar(x - wd / 2, d["gbar"][big], wd, color=c_sig, alpha=0.85, label="‖ḡ‖ (có dấu, 3DGS)")
    ax.bar(x + wd / 2, d["gabs"][big], wd, color=c_abs, alpha=0.85, hatch="//", edgecolor="w", label="‖ḡ$_{abs}$‖ (|g|, FastGS)")
    ax.axhline(TAU_GRAD, color="tab:blue", ls="--", lw=1.2); ax.axhline(TAU_ABS, color="tab:green", ls="--", lw=1.2)
    ax.text(len(big) - 0.5, TAU_GRAD * 1.12, f"τ = {TAU_GRAD:.0e}", ha="right", fontsize=8, color="tab:blue")
    ax.text(len(big) - 0.5, TAU_ABS * 1.12, f"τ$_{{abs}}$ = {TAU_ABS:.0e}", ha="right", fontsize=8, color="tab:green")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([str(j) for j in big], fontsize=6, rotation=90)
    ax.set_xlabel("chỉ số Gaussian to (max s > S_BIG)"); ax.set_ylabel("gradient trung bình tích luỹ")
    k_star = int(np.where(big == i_star)[0][0])
    ax.axvspan(k_star - 0.5, k_star + 0.5, color="tab:orange", alpha=0.18)
    ax.annotate(f"i* = {i_star}", xy=(k_star, d["gabs"][i_star]), xytext=(k_star, d["gabs"][i_star] * 3),
                ha="center", fontsize=8, color="tab:orange", arrowprops=dict(arrowstyle="->", color="tab:orange"))
    n3 = int(d["split_3dgs"].sum()); nf = int(d["split_fastgs"].sum()); noa = int(d["split_only_abs"].sum())
    ax.set_title(f"(b) {len(big)} Gaussian to tại t = {t_sel}\n3DGS split (‖ḡ‖ ≥ τ): {n3}\n"
                 f"FastGS split (‖ḡ$_{{abs}}$‖ ≥ τ$_{{abs}}$ ∧ Imp > {IMP_THR}): {nf}\n"
                 f"chỉ |g| bắt được: {noa} Gaussian (cam = i*)", fontsize=9)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, which="both")

    # ---- (c) khong split ----
    ax = axes[2]
    show_img(ax, I_no, f"(c) KHÔNG split i* (như 3DGS)\n{N_ADAM} bước Adam: L1 = {loss0:.4f} → {loss_no:.4f}")
    draw_ellipses(ax, {k: p_no[k][[i_star]] for k in KEYS}, k=2, colors=["tab:orange"], lw=2.0)
    ax.set(**zoom)
    # ---- (d) co split ----
    ax = axes[3]
    show_img(ax, I_sp, f"(d) SPLIT i* thành 2 con (FastGS)\nL1 = {loss0:.4f} → {loss_sp0:.4f} (ngay sau split)\n→ {loss_sp:.4f} sau {N_ADAM} bước Adam")
    draw_ellipses(ax, {k: p0[k][[i_star]] for k in KEYS}, k=2, colors=["tab:orange"], lw=1.2, ls="--", alpha=0.6)
    draw_ellipses(ax, {k: p_sp1[k][child_idx] for k in KEYS}, k=2, colors=["tab:red", "tab:purple"], lw=1.0, ls=":", alpha=0.8)
    draw_ellipses(ax, {k: p_sp[k][child_idx] for k in KEYS}, k=2, colors=["tab:red", "tab:purple"], lw=2.0)
    ax.set(**zoom)
    ax.text(zoom["xlim"][0] + 0.4, zoom["ylim"][0] - 0.4,
            "cam đứt: gốc; chấm: 2 con ngay sau split; đậm: sau Adam",
            fontsize=8, va="bottom", bbox=dict(fc="w", ec="none", alpha=0.85))
    fig.suptitle("Cải tiến 2 — split dùng Σ|g| thay Σg: gradient pixel triệt tiêu nhau trong một ảnh nên ‖ḡ‖ nhỏ "
                 "dù Gaussian đang bị kéo nhiều phía", fontsize=11.5)
    save(fig, "7_7_abs-vs-signed-split.png")
    return dict(t=t_sel, i_star=i_star, gsum=gsum, gabs=gabs_now, gbar_acc=gbar_i, gabs_acc=gabs_i,
                loss0=loss0, loss_no=loss_no, loss_split0=loss_sp0, loss_split=loss_sp)


# =============================================================================
# 7.8  Prune cung (3DGS) vs prune multinomial (FastGS)
# =============================================================================
def _pick_prune_t():
    R = full_run_compare()
    df = R["fastgs"][1]["diag"]
    if 600 in df and df[600]["C"].sum() > 0:
        return 600
    return max(df, key=lambda t: df[t]["C"].sum())


def fig_7_8():
    R = full_run_compare()
    t_sel = _pick_prune_t()
    d3 = R["3dgs"][1]["diag"][t_sel]
    df = R["fastgs"][1]["diag"][t_sel]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    rows = [("3DGS gốc", d3, False), ("FastGS", df, True)]
    stats = {}
    for r, (name, d, is_f) in enumerate(rows):
        pd_ = d["p_after_densify"]; pa = d["p_after"]
        alpha = sigmoid(pd_["lo"]); smax = np.exp(pd_["logs"]).max(1)
        C, rm = d["C"], d["removed"]
        keptC = C & ~rm
        nC, nrm = int(C.sum()), int(rm.sum())
        budget = int(0.5 * nC)
        n = len(alpha)
        # ---- cot 1: bar alpha ----
        ax = axes[r, 0]
        cols = np.array(["0.6"] * n, dtype=object)
        cols[keptC] = "tab:orange"; cols[rm] = "tab:red"
        ax.bar(np.arange(n), alpha, color=list(cols), width=0.8)
        ax.axhline(EPS_OPA, color="tab:red", ls="--", lw=1.2)
        ax.text(n + 0.5, EPS_OPA, f"ε = {EPS_OPA}", ha="left", va="center", fontsize=8.5, color="tab:red")
        ax.set_xlim(-1, n + 9)
        hug = np.where(C & (smax > S_HUGE))[0]
        ax.plot(hug, alpha[hug] * 1.35, "v", color="k", ms=4, label=f"ứng viên vì max s > S_HUGE = {S_HUGE:.0f} px")
        if is_f and d["S"] is not None:
            Sidx = np.where(d["S"])[0]
            ax.plot(Sidx, np.full(len(Sidx), 0.7), "*", color="tab:purple", ms=7, ls="none",
                    label=f"rút multinomial S (|S| = {len(Sidx)}, w = 1/(1−Pruning))")
        ax.set_yscale("log"); ax.set_ylim(1e-3, 60)   # cho tren de dat legend
        ax.set_xlabel("chỉ số Gaussian (sau densify)"); ax.set_ylabel("opacity α (log)")
        ax.grid(axis="y", alpha=0.3, which="both")
        if is_f:
            ttl = (f"({'d' if r else 'a'}) {name}, t = {t_sel}: |C| = {nC} (α < ε ∨ max s > S_HUGE)\n"
                   f"budget ⌊0.5|C|⌋ = {budget}; xoá C ∩ S = {nrm}  (cam: ứng viên được giữ)")
        else:
            ttl = (f"({'d' if r else 'a'}) {name}, t = {t_sel}: |C| = {nC} (α < ε ∨ max s > S_HUGE)\n"
                   f"xoá NGAY toàn bộ C: {nrm} / {nC}  (đỏ)")
        ax.set_title(ttl)
        handles = [plt.Rectangle((0, 0), 1, 1, color="tab:red", label="bị xoá"),
                   plt.Rectangle((0, 0), 1, 1, color="0.6", label="không phải ứng viên")]
        if is_f:
            handles.insert(1, plt.Rectangle((0, 0), 1, 1, color="tab:orange", label="ứng viên C được giữ"))
        h2, l2 = ax.get_legend_handles_labels()
        ax.legend(handles=handles + h2, fontsize=7.5, loc="upper left", ncol=2, framealpha=0.95)
        # ---- cot 2: render sau densify + ellipse ----
        ax = axes[r, 1]
        show_img(ax, d["I_after_densify"],
                 f"({'e' if r else 'b'}) Render sau densify, N = {n}\nđỏ đứt = sẽ xoá ({nrm}); cam = ứng viên giữ lại ({int(keptC.sum())})")
        others = np.where(~C)[0]
        draw_ellipses(ax, {k: pd_[k][others] for k in KEYS}, k=1.5, colors=["0.75"] * len(others), lw=0.5, alpha=0.6)
        if keptC.any():
            ki = np.where(keptC)[0]
            draw_ellipses(ax, {k: pd_[k][ki] for k in KEYS}, k=1.5, colors=["tab:orange"] * len(ki), lw=1.6)
        if rm.any():
            ri = np.where(rm)[0]
            draw_ellipses(ax, {k: pd_[k][ri] for k in KEYS}, k=1.5, colors=["tab:red"] * len(ri), lw=1.6, ls="--")
        # ---- cot 3: render sau prune ----
        ax = axes[r, 2]
        dI = float(np.abs(d["I_after"] - d["I_after_densify"]).max())
        na = len(pa["mu"])
        show_img(ax, d["I_after"], f"({'f' if r else 'c'}) Render SAU prune: N = {n} → {na}\nmax|ΔI| so với trước prune = {dI:.3f}")
        draw_ellipses(ax, pa, k=1.5, colors=["0.75"] * na, lw=0.5, alpha=0.6)
        if is_f:
            ax.text(1, H - 1, f"sau đó α ← min(α, {ALPHA_CAP})", fontsize=8, va="bottom",
                    bbox=dict(fc="w", ec="none", alpha=0.85))
        stats[name] = dict(nC=nC, budget=budget, removed=nrm, N_after=na, dI=dI)
    axes[0, 0].annotate("3DGS gốc", xy=(-0.22, 0.5), xycoords="axes fraction", rotation=90, va="center", fontsize=12, fontweight="bold")
    axes[1, 0].annotate("FastGS", xy=(-0.22, 0.5), xycoords="axes fraction", rotation=90, va="center", fontsize=12, fontweight="bold")
    fig.suptitle("Cải tiến 3 — 3DGS xoá NGAY mọi α<ε (cả cụm); FastGS lập ứng viên C rồi rút multinomial theo Pruning, "
                 "xoá C∩S ≤ ⌊0.5|C|⌋", fontsize=11.5)
    save(fig, "7_8_prune-hard-vs-multinomial.png")
    return dict(t=t_sel, **stats)


if __name__ == "__main__":
    t_all = time.time()
    s7 = fig_7_7(); print("7.7:", s7)
    s8 = fig_7_8(); print("7.8:", s8)
    print(f"Tổng thời gian: {time.time() - t_all:.1f}s")
