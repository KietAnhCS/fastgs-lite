"""
Engine dung chung cho PHAN 7 cua chuong 12: mo phong toy 2D (tai su dung part6_figures.py)
voi HAI bien the Adaptive Density Control chay tren CUNG anh muc tieu, cung seed:

  mode = "3dgs"   : ADC cua 3DGS goc (Kerbl 2023)
      clone  = [‖ḡ‖ ≥ τ] ∧ [max s ≤ S_BIG]
      split  = [‖ḡ‖ ≥ τ] ∧ [max s >  S_BIG]          (gradient CO DAU cho ca hai nhanh)
      prune  = xoa NGAY moi Gaussian α < EPS_OPA  (∨ max s > S_HUGE sau moc reset)
      khong tran opacity, khong Importance, khong multinomial, khong final prune

  mode = "fastgs" : ADC cua FastGS-lite (chuong 12)
      Importance_i = so pixel loi trong footprint (mat na min-max > 0.1), V = 1 view
      clone  = [‖ḡ‖ ≥ τ] ∧ [max s ≤ S_BIG] ∧ [Imp > IMP_THR]
      split  = [‖ḡ_abs‖ ≥ τ_abs] ∧ [max s > S_BIG] ∧ [Imp > IMP_THR]
      prune  : C = {α < EPS_OPA ∨ max s > S_HUGE}; w_i = 1/(1e-6 + 1 - Pruning_i);
               S ~ Multinomial(w, ⌊0.5|C|⌋, khong hoan lai); xoa = C ∩ S
      sau moi ADC: α ← min(α, 0.8)
      final prune tai FINAL_AT: xoa [α < 0.1] ∨ [Pruning > 0.9]

Dung: from part7_engine import *  ;  R = full_run_compare()
  R["3dgs"], R["fastgs"], R["no"]  → (p_cuoi, hist, snaps)
  hist["adc"]  : list (t, n_clone, n_split, n_prune)
  hist["diag"] : dict t → chan doan tai moc t (gbar, gabs, smax, imp, pruning, alpha, cac mask ...)
Ket qua cache vao part7_cache.pkl (xoa file de chay lai).
"""
import os, sys, time, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part6_figures import *   # noqa: F401,F403  (engine toy: render, loss_and_grad, adam, do_clone, do_split, ...)
from part6_figures import _cat, _keep, _xx, _yy   # noqa

# ---- hang so rieng cua phan 7 -------------------------------------------------
IMP_THR = 5          # Importance > 5 (giong code that; V = 1 view trong toy)
S_HUGE = 10.0        # "qua to" (px) — vai 0.1·extent cua code that; chi bat sau RESET_AT
ALPHA_CAP = 0.8      # tran opacity sau moi ADC (FastGS)
FINAL_AT = (1000, 1100)   # moc final prune (FastGS), sau ADC_UNTIL = 900
FINAL_ALPHA = 0.1
FINAL_PRUNE_THR = 0.9
CACHE = os.path.join(OUT, "part7_cache.pkl")


def logit(a):
    return np.log(a / (1 - a))


# ---- diem so multi-view (V = 1) --------------------------------------------------
def error_mask(I):
    """Buoc ①②③: e(x) = mean_ch |I - T|, min-max theo anh, mat na > 0.1."""
    e = np.abs(I.reshape(-1, 3) - TARGET_FLAT).mean(1)
    d = e.max() - e.min()
    ehat = (e - e.min()) / d if d > 0 else np.zeros_like(e)
    return ehat > 0.1, e, ehat


def footprints(ch):
    """Footprint huu hinh: da qua 3 cua (3σ, w ≥ 1/255, T ≥ 1e-4)."""
    return (ch["w"] >= 1.0 / 255) & (ch["T"] >= 1e-4)


def scores(I, ch):
    """Buoc ④⑤⑦ voi V = 1: counts, Importance (=counts), Pruning = minmax(counts)."""
    m, e, ehat = error_mask(I)
    fp = footprints(ch)
    counts = (fp & m[None, :]).sum(1)
    imp = counts.copy()
    d = counts.max() - counts.min()
    pruning = (counts - counts.min()) / d if d > 0 else np.zeros(len(counts))
    return dict(mask=m.reshape(H, W), e=e.reshape(H, W), ehat=ehat.reshape(H, W),
                fp=fp, counts=counts, imp=imp, pruning=pruning)


def multinomial_no_replacement(w, k, rng):
    w = np.asarray(w, float)
    k = int(min(k, (w > 0).sum()))
    if k <= 0:
        return np.array([], int)
    return rng.choice(len(w), size=k, replace=False, p=w / w.sum())


# ---- mot moc ADC theo mode ------------------------------------------------------
def adc_step_mode(mode, p, st, accum, accum_abs, denom, rng, I, ch, t):
    n = len(p["mu"])
    gbar = accum / np.maximum(denom, 1)
    gabs = accum_abs / np.maximum(denom, 1)
    smax = np.exp(p["logs"]).max(1)
    big = smax > S_BIG
    sc = scores(I, ch)
    imp, pruning = sc["imp"], sc["pruning"]
    alpha0 = sigmoid(p["lo"])

    # quyet dinh theo 3DGS goc (luon tinh de doi chieu)
    clone_3 = (gbar >= TAU_GRAD) & ~big
    split_3 = (gbar >= TAU_GRAD) & big
    # quyet dinh theo FastGS
    clone_f = (gbar >= TAU_GRAD) & ~big & (imp > IMP_THR)
    split_f = (gabs >= TAU_ABS) & big & (imp > IMP_THR)

    clone_m, split_m = (clone_3, split_3) if mode == "3dgs" else (clone_f, split_f)
    clone_m, split_m = clone_m.copy(), split_m.copy()

    # tran N (chi de script chay nhanh) — giong part6
    budget = N_MAX - n
    if clone_m.sum() + split_m.sum() > budget:
        gsel = gbar if mode == "3dgs" else gabs
        score = np.where(clone_m, gbar / TAU_GRAD, 0) + np.where(split_m, gsel / TAU_ABS, 0)
        order = np.argsort(-score)
        chosen = np.zeros(n, bool)
        chosen[order[:max(budget, 0)]] = True
        clone_m &= chosen; split_m &= chosen

    diag = dict(p_before={k: p[k].copy() for k in KEYS}, I_before=I.copy(),
                gbar=gbar, gabs=gabs, smax=smax, imp=imp, pruning=pruning, alpha=alpha0,
                counts=sc["counts"], mask=sc["mask"], ehat=sc["ehat"], fp=sc["fp"],
                clone_3dgs=clone_3, split_3dgs=split_3, clone_fastgs=clone_f, split_fastgs=split_f,
                clone=clone_m.copy(), split=split_m.copy(),
                blocked_by_imp=((clone_3 | split_3) & ~(imp > IMP_THR)),
                split_only_abs=(split_f & ~split_3), split_only_signed=(split_3 & ~(gabs >= TAU_ABS) & big))

    clone_idx = np.where(clone_m)[0]
    split_idx = np.where(split_m)[0]
    n_score = n
    if len(clone_idx):
        do_clone(p, st, clone_idx)
    if len(split_idx):
        do_split(p, st, split_idx, rng)
    n_new = len(p["mu"])

    # ---- prune ----
    alpha = sigmoid(p["lo"])
    smax_new = np.exp(p["logs"]).max(1)
    cand = alpha < EPS_OPA
    if t > RESET_AT:
        cand |= smax_new > S_HUGE
    diag["p_after_densify"] = {k: p[k].copy() for k in KEYS}
    diag["I_after_densify"] = render(p)
    if mode == "3dgs":
        removed = cand.copy()
        diag["C"] = cand.copy(); diag["S"] = None; diag["w"] = None
    else:
        # trong so theo chi so CU (padded 0 cho Gaussian moi) — dung thu tu that cua code
        w = np.zeros(n_new)
        # sau split goc bi xoa: chi so 0..n_score-1 cua quan the moi
        n_pad = min(n_score, n_new)
        w[:n_pad] = 1.0 / (1e-6 + 1.0 - pruning[:n_pad])
        budget_rm = int(0.5 * cand.sum())
        S = np.zeros(n_new, bool)
        if budget_rm > 0:
            S[multinomial_no_replacement(w, budget_rm, rng)] = True
        removed = cand & S
        diag["C"] = cand.copy(); diag["S"] = S.copy(); diag["w"] = w.copy()
    diag["removed"] = removed.copy()
    n_prune = int(removed.sum())
    if n_prune:
        _keep(p, st, ~removed)
    # ---- tran opacity (FastGS) ----
    if mode == "fastgs":
        p["lo"] = np.minimum(p["lo"], logit(ALPHA_CAP))
    diag["p_after"] = {k: p[k].copy() for k in KEYS}
    diag["I_after"] = render(p)
    return len(clone_idx), len(split_idx), n_prune, diag


def final_prune_step(p, st, I, ch):
    sc = scores(I, ch)
    alpha = sigmoid(p["lo"])
    rm = (alpha < FINAL_ALPHA) | (sc["pruning"] > FINAL_PRUNE_THR)
    diag = dict(p_before={k: p[k].copy() for k in KEYS}, I_before=I.copy(),
                alpha=alpha, pruning=sc["pruning"], removed=rm.copy())
    if rm.sum() and rm.sum() < len(alpha):
        _keep(p, st, ~rm)
    diag["p_after"] = {k: p[k].copy() for k in KEYS}
    diag["I_after"] = render(p)
    return int(rm.sum()), diag


# ---- vong train theo mode -------------------------------------------------------
def train_mode(mode, p, T_total=T_TOTAL, snapshots=(), seed=SEED):
    """mode ∈ {"3dgs", "fastgs", "no"}."""
    rng = np.random.default_rng(seed + 1000)
    st = adam_init(p)
    n = len(p["mu"])
    accum = np.zeros(n); accum_abs = np.zeros(n); denom = np.zeros(n)
    hist = dict(loss=[], N=[], adc=[], diag={}, final=[], final_diag={}, alpha_win={})
    snaps = {}
    for t in range(1, T_total + 1):
        loss, g, I, ch = loss_and_grad(p)
        hist["loss"].append(loss); hist["N"].append(len(p["mu"]))
        if t - 1 in snapshots:
            snaps[t - 1] = dict(p={k: p[k].copy() for k in KEYS}, I=I.copy(), loss=loss)
        if RESET_AT - 49 <= t <= RESET_AT + 50:
            hist["alpha_win"][t] = sigmoid(p["lo"]).copy()
        vis = ch["w"].max(1) > 1e-3
        accum[vis] += np.linalg.norm(g["mu"][vis], axis=1)
        accum_abs[vis] += np.linalg.norm(g["mu_abs"][vis], axis=1)
        denom[vis] += 1
        adam_step(p, g, st)
        if mode == "no":
            continue
        if t == RESET_AT:
            p["lo"] = np.minimum(p["lo"], logit(ALPHA_RESET))
        if ADC_FROM <= t <= ADC_UNTIL and t % K_ADC == 0:
            I2, ch2 = render(p, return_cache=True)
            nc, ns, npn, diag = adc_step_mode(mode, p, st, accum, accum_abs, denom, rng, I2, ch2, t)
            hist["adc"].append((t, nc, ns, npn))
            hist["diag"][t] = diag
            n = len(p["mu"])
            accum = np.zeros(n); accum_abs = np.zeros(n); denom = np.zeros(n)
        if mode == "fastgs" and t in FINAL_AT:
            I2, ch2 = render(p, return_cache=True)
            nr, fd = final_prune_step(p, st, I2, ch2)
            hist["final"].append((t, nr))
            hist["final_diag"][t] = fd
            n = len(p["mu"])
            accum = np.zeros(n); accum_abs = np.zeros(n); denom = np.zeros(n)
    loss, g, I, ch = loss_and_grad(p)
    hist["loss"].append(loss); hist["N"].append(len(p["mu"]))
    snaps[T_total] = dict(p={k: p[k].copy() for k in KEYS}, I=I.copy(), loss=loss)
    return p, hist, snaps


SNAPS_AT = (0, 100, 300, 600, 1000)
_R = {}


def full_run_compare(force=False):
    """Chay 3 lan: 3dgs, fastgs, no-ADC — cung p0, cung seed. Cache ra pickle."""
    if _R and not force:
        return _R
    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as f:
            _R.update(pickle.load(f))
        return _R
    rng = np.random.default_rng(SEED)
    N0 = 12
    p0 = init_sfm_like(N0, rng, s=3.0)
    out = dict(p0=p0, snaps_at=SNAPS_AT)
    for mode in ("3dgs", "fastgs", "no"):
        t0 = time.time()
        res = train_mode(mode, {k: v.copy() for k, v in p0.items()}, snapshots=SNAPS_AT)
        p, hist, snaps = res
        print(f"  run {mode:7s}: N {N0} → {len(p['mu'])}, L1 {hist['loss'][0]:.4f} → {hist['loss'][-1]:.4f}, "
              f"{time.time() - t0:.1f}s")
        out[mode] = res
    tmp = CACHE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, CACHE)
    _R.update(out)
    return _R


def snap_titles(hist, keys, T_total=T_TOTAL):
    titles = []
    adc_t = [a[0] for a in hist["adc"]]
    for t in keys:
        k = sum(1 for a in adc_t if a <= t)
        if t == 0:
            titles.append("t = 0 (khởi tạo)")
        elif t >= T_total:
            titles.append(f"t = {t} (cuối)")
        else:
            titles.append(f"t = {t} (sau ADC lần {k})")
    return titles


if __name__ == "__main__":
    R = full_run_compare(force="--force" in sys.argv)
    for mode in ("3dgs", "fastgs"):
        p, hist, _ = R[mode]
        print(mode, "ADC log (t, clone, split, prune):", hist["adc"])
        if hist["final"]:
            print(mode, "final prune:", hist["final"])
