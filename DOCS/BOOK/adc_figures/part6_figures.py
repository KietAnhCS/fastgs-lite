"""
Hinh minh hoa cho PHAN 6 cua chuong Adaptive Density Control:
vi du truc quan toy 2D Gaussian splatting — Gaussian THIEU, DU, clone, split,
prune, va cac Gaussian tu "chia vung" sau nhieu vong toi uu + ADC.

Moi ham fig_6_k() xuat dung 1 file PNG "6_k_ten-ngan.png" vao cung thu muc.
Chi dung numpy + matplotlib (khong torch). Gradient tinh giai tich (backward
alpha-compositing giong 3DGS, thu tu depth = thu tu chi so), toi uu bang Adam.

Chay: python adc_figures/part6_figures.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

# ---- hang so cua mo hinh do choi -------------------------------------------
H = W = 64
BG = np.array([0.93, 0.91, 0.86])          # mau nen
SEED = 7
K_ADC = 100                                 # densification_interval (toy)
T_TOTAL = 1200                              # so vong train
ADC_FROM, ADC_UNTIL = 100, 900              # cua so densify (toy)
TAU_GRAD = 4.0e-5                           # nguong ‖ḡ‖   (clone)  — don vi toy
TAU_ABS = 4.0e-4                            # nguong ‖ḡ_abs‖ (split) — don vi toy
S_BIG = 3.5                                 # max s > S_BIG (pixel) → "to" → split
EPS_OPA = 0.03                              # opacity < eps → prune
RESET_AT = 550                              # moc reset opacity (toy): alpha ← min(alpha, ALPHA_RESET)
ALPHA_RESET = 0.02
N_MAX = 80                                  # tran N de script chay nhanh
LR = dict(mu=0.4, logs=0.02, th=0.02, col=0.005, lo=0.05)   # ti le col:lo ~ code that (0.0025 : 0.05)

# luoi pixel (tam pixel)
_yy, _xx = np.mgrid[0:H, 0:W]
PX = _xx.ravel() + 0.5
PY = _yy.ravel() + 0.5


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved:", path)


# =============================================================================
# Anh muc tieu: hinh tron to mau deu + dai manh + hang cham nho + o vuong nho
# =============================================================================
def make_target():
    img = np.tile(BG, (H, W, 1)).astype(float)
    x, y = _xx + 0.5, _yy + 0.5
    # hinh tron to, mau deu (vung phang)
    img[(x - 22) ** 2 + (y - 24) ** 2 <= 14.5 ** 2] = (0.22, 0.45, 0.80)
    # dai manh (2 px) — vung chi tiet
    img[(np.abs(y - 45.5) <= 1.0) & (x >= 36) & (x <= 61)] = (0.12, 0.12, 0.12)
    # hang 5 cham nho, mau xen ke — vung chi tiet
    for k, cx in enumerate([38.5, 43.5, 48.5, 53.5, 58.5]):
        col = (0.85, 0.20, 0.20) if k % 2 == 0 else (0.20, 0.65, 0.25)
        img[(x - cx) ** 2 + (y - 55) ** 2 <= 2.3 ** 2] = col
    # o vuong nho
    img[(np.abs(x - 52) <= 3.5) & (np.abs(y - 12) <= 3.5)] = (0.95, 0.72, 0.10)
    return img


TARGET = make_target()
TARGET_FLAT = TARGET.reshape(-1, 3)


def target_detail_map():
    """Do 'chi tiet' cua anh muc tieu: chuan gradient anh (Sobel don gian)."""
    g = TARGET.mean(-1)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) / 2
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) / 2
    return np.sqrt(gx ** 2 + gy ** 2)


DETAIL = target_detail_map()


# =============================================================================
# Mo hinh: N Gaussian 2D — mu (N,2), logs (N,2), th (N,), col (N,3), lo (N,) logit alpha
# =============================================================================
def init_params(n, rng, s=5.0, spread=None):
    if spread is None:
        mu = rng.uniform(6, W - 6, size=(n, 2))
    else:
        mu = np.array(spread, float)
    return dict(
        mu=mu,
        logs=np.log(np.full((n, 2), s, float)),
        th=rng.uniform(-np.pi / 4, np.pi / 4, size=n),
        col=rng.uniform(0.3, 0.7, size=(n, 3)),
        lo=np.full(n, np.log(0.6 / 0.4)),
    )


def init_sfm_like(n, rng, s=3.0):
    """Khoi tao kieu SfM: diem chi co o cho co texture (bien/chi tiet), mau lay tu anh."""
    prob = DETAIL.ravel() ** 2
    prob /= prob.sum()
    idx = rng.choice(H * W, size=n, replace=False, p=prob)
    mu = np.stack([PX[idx], PY[idx]], 1) + rng.normal(scale=0.5, size=(n, 2))
    col = np.clip(TARGET_FLAT[idx] + rng.normal(scale=0.05, size=(n, 3)), 0, 1)
    return dict(mu=mu, logs=np.log(np.full((n, 2), s, float)),
                th=rng.uniform(-np.pi / 4, np.pi / 4, size=n), col=col,
                lo=np.full(n, np.log(0.6 / 0.4)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cov_of(p, i):
    s1, s2 = np.exp(p["logs"][i])
    c, s = np.cos(p["th"][i]), np.sin(p["th"][i])
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([s1 ** 2, s2 ** 2]) @ R.T


def render(p, return_cache=False):
    """Alpha-compositing theo thu tu chi so (Gaussian 0 gan camera nhat)."""
    mu, s, th = p["mu"], np.exp(p["logs"]), p["th"]
    alpha = sigmoid(p["lo"])
    col = np.clip(p["col"], 0, 1)
    dx = PX[None, :] - mu[:, 0:1]                  # (N, HW)
    dy = PY[None, :] - mu[:, 1:2]
    c, sn = np.cos(th)[:, None], np.sin(th)[:, None]
    u = c * dx + sn * dy
    v = -sn * dx + c * dy
    inv1 = 1.0 / s[:, 0:1] ** 2
    inv2 = 1.0 / s[:, 1:2] ** 2
    q = u ** 2 * inv1 + v ** 2 * inv2
    G = np.exp(-0.5 * q) * (q < 9.0)               # footprint = 3 sigma
    w = np.minimum(alpha[:, None] * G, 0.99)       # (N, HW)
    Tcum = np.cumprod(1.0 - w, axis=0)             # T_{i+1}
    T = np.vstack([np.ones((1, w.shape[1])), Tcum[:-1]])   # T_i = prod_{j<i}(1-w_j)
    Tw = T * w                                     # (N, HW) trong so dong gop cua i tai x
    I = Tw.T @ col + Tcum[-1][:, None] * BG[None, :]       # (HW, 3)
    if not return_cache:
        return I.reshape(H, W, 3)
    cache = dict(u=u, v=v, inv1=inv1, inv2=inv2, G=G, w=w, T=T, Tw=Tw, Tend=Tcum[-1],
                 alpha=alpha, col=col, c=c, sn=sn)
    return I.reshape(H, W, 3), cache


def loss_and_grad(p):
    """L1 loss va gradient giai tich cho moi tham so + thong ke gradient vi tri."""
    I, ch = render(p, return_cache=True)
    Iflat = I.reshape(-1, 3)
    diff = Iflat - TARGET_FLAT
    loss = np.abs(diff).mean()
    dL_dI = np.sign(diff) / diff.size               # (HW, 3)

    T, w, G, Tw = ch["T"], ch["w"], ch["G"], ch["Tw"]
    # dI/dw_i = T_i c_i - S_i/(1-w_i), S_i = tong dong gop cua moi Gaussian phia SAU i (+ nen)
    # chi can tich vo huong voi dL/dI, nen lam viec voi (N, HW) thay vi (N, HW, 3)
    cdot = ch["col"] @ dL_dI.T                              # (N, HW): c_i . dL/dI
    sc = Tw * cdot                                          # (N, HW): dong gop_i . dL/dI
    back = np.cumsum(sc[::-1], axis=0)[::-1]                # sum_{k>=i}
    S_dot = back - sc + (ch["Tend"] * (BG @ dL_dI.T))[None]  # sum_{k>i} + nen
    dL_dw = T * cdot - S_dot / (1.0 - w)                    # (N, HW)
    g = {}
    g["col"] = Tw @ dL_dI                                   # (N, 3)
    a = ch["alpha"]
    dL_dG = dL_dw * a[:, None]
    g["lo"] = (dL_dw * G).sum(1) * a * (1 - a)
    u, v, inv1, inv2, c, sn = ch["u"], ch["v"], ch["inv1"], ch["inv2"], ch["c"], ch["sn"]
    dG_dmux = G * (u * c * inv1 - v * sn * inv2)
    dG_dmuy = G * (u * sn * inv1 + v * c * inv2)
    gpx = dL_dG * dG_dmux                          # gradient vi tri TUNG PIXEL
    gpy = dL_dG * dG_dmuy
    g["mu"] = np.stack([gpx.sum(1), gpy.sum(1)], 1)                 # co dau
    g["mu_abs"] = np.stack([np.abs(gpx).sum(1), np.abs(gpy).sum(1)], 1)  # tri tuyet doi
    g["logs"] = np.stack([(dL_dG * G * u ** 2 * inv1).sum(1),
                          (dL_dG * G * v ** 2 * inv2).sum(1)], 1)
    g["th"] = (dL_dG * (-G * u * v * (inv1 - inv2))).sum(1)
    g["pix"] = (gpx, gpy)
    return loss, g, I, ch


# ---- Adam ------------------------------------------------------------------
KEYS = ["mu", "logs", "th", "col", "lo"]


def adam_init(p):
    return dict(m={k: np.zeros_like(p[k]) for k in KEYS},
                v={k: np.zeros_like(p[k]) for k in KEYS}, t=0)


def adam_step(p, g, st, b1=0.9, b2=0.999, eps=1e-12):
    st["t"] += 1
    t = st["t"]
    for k in KEYS:
        st["m"][k] = b1 * st["m"][k] + (1 - b1) * g[k]
        st["v"][k] = b2 * st["v"][k] + (1 - b2) * g[k] ** 2
        mh = st["m"][k] / (1 - b1 ** t)
        vh = st["v"][k] / (1 - b2 ** t)
        p[k] = p[k] - LR[k] * mh / (np.sqrt(vh) + eps)
    p["logs"] = np.clip(p["logs"], np.log(0.7), np.log(18.0))
    p["mu"] = np.clip(p["mu"], -4, W + 4)
    p["col"] = np.clip(p["col"], 0, 1)
    p["lo"] = np.clip(p["lo"], -6, 6)


# ---- thao tac ADC (toy) ------------------------------------------------------
def _cat(p, st, new):
    """Noi Gaussian moi vao cuoi; trang thai Adam cua ban moi = 0 (nhu code that)."""
    for k in KEYS:
        p[k] = np.concatenate([p[k], new[k]], 0)
        st["m"][k] = np.concatenate([st["m"][k], np.zeros_like(new[k])], 0)
        st["v"][k] = np.concatenate([st["v"][k], np.zeros_like(new[k])], 0)


def _keep(p, st, mask):
    for k in KEYS:
        p[k] = p[k][mask]
        st["m"][k] = st["m"][k][mask]
        st["v"][k] = st["v"][k][mask]


def do_clone(p, st, idx):
    new = {k: p[k][idx].copy() for k in KEYS}
    _cat(p, st, new)


def do_split(p, st, idx, rng, n_child=2):
    """Con lay mau tu hinh dang goc: mu + R eps, eps ~ N(0, diag s^2); scale/1.6; xoa goc."""
    s = np.exp(p["logs"][idx])                       # (M, 2)
    th = p["th"][idx]
    new = {k: np.repeat(p[k][idx], n_child, axis=0) for k in KEYS}
    eps = rng.normal(size=(len(idx) * n_child, 2)) * np.repeat(s, n_child, axis=0)
    c, sn = np.cos(np.repeat(th, n_child)), np.sin(np.repeat(th, n_child))
    rot = np.stack([c * eps[:, 0] - sn * eps[:, 1], sn * eps[:, 0] + c * eps[:, 1]], 1)
    new["mu"] = new["mu"] + rot
    new["logs"] = new["logs"] - np.log(0.8 * n_child)
    _cat(p, st, new)
    n_old = len(p["mu"]) - len(idx) * n_child
    keep = np.ones(len(p["mu"]), bool)
    keep[idx] = False
    _keep(p, st, keep)
    return eps, rot


def adc_step(p, st, accum, accum_abs, denom, rng, record=None):
    """Mot moc densify+prune toy. Tra ve (n_clone, n_split, n_prune)."""
    n = len(p["mu"])
    gbar = accum / np.maximum(denom, 1)
    gabs = accum_abs / np.maximum(denom, 1)
    smax = np.exp(p["logs"]).max(1)
    big = smax > S_BIG
    clone_m = (gbar >= TAU_GRAD) & ~big
    split_m = (gabs >= TAU_ABS) & big
    # tran N (chi de script chay nhanh): uu tien gradient lon
    budget = N_MAX - n
    if clone_m.sum() + split_m.sum() > budget:
        score = np.where(clone_m, gbar / TAU_GRAD, 0) + np.where(split_m, gabs / TAU_ABS, 0)
        order = np.argsort(-score)
        chosen = np.zeros(n, bool)
        chosen[order[:max(budget, 0)]] = True
        clone_m &= chosen; split_m &= chosen
    clone_idx = np.where(clone_m)[0]
    split_idx = np.where(split_m)[0]
    if len(clone_idx):
        do_clone(p, st, clone_idx)
    if len(split_idx):
        do_split(p, st, split_idx, rng)
    # prune theo opacity
    alpha = sigmoid(p["lo"])
    keep = alpha >= EPS_OPA
    n_prune = int((~keep).sum())
    if record is not None:
        record["p_before"] = {k: p[k].copy() for k in KEYS}
        record["I_before"] = render(p)
        record["keep"] = keep.copy()
    if n_prune:
        _keep(p, st, keep)
    if record is not None:
        record["I_after"] = render(p)
        record["p_after"] = {k: p[k].copy() for k in KEYS}
    return len(clone_idx), len(split_idx), n_prune


def train(p, use_adc=True, T_total=T_TOTAL, snapshots=(), seed=SEED, log_every=1):
    """Vong lap train toy: gradient descent (Adam) + ADC moi K_ADC vong."""
    rng = np.random.default_rng(seed + 1000)
    st = adam_init(p)
    n = len(p["mu"])
    accum = np.zeros(n); accum_abs = np.zeros(n); denom = np.zeros(n)
    hist = dict(loss=[], N=[], adc=[])      # adc: (t, n_clone, n_split, n_prune)
    hist["alpha_win"] = {}                  # t → alpha (cua so quanh moc reset, chi so on dinh)
    hist["prune_rec"] = {}
    snaps = {}
    win_lo = (RESET_AT // K_ADC) * K_ADC + 1
    win_hi = win_lo - 1 + K_ADC
    for t in range(1, T_total + 1):
        loss, g, I, ch = loss_and_grad(p)
        hist["loss"].append(loss); hist["N"].append(len(p["mu"]))
        if t - 1 in snapshots:
            snaps[t - 1] = dict(p={k: p[k].copy() for k in KEYS}, I=I.copy(), loss=loss)
        if use_adc and win_lo <= t <= win_hi:
            hist["alpha_win"][t] = sigmoid(p["lo"]).copy()
            if t == RESET_AT:
                hist["I_reset"] = I.copy()
        vis = ch["w"].max(1) > 1e-3
        accum[vis] += np.linalg.norm(g["mu"][vis], axis=1)
        accum_abs[vis] += np.linalg.norm(g["mu_abs"][vis], axis=1)
        denom[vis] += 1
        adam_step(p, g, st)
        if use_adc and t == RESET_AT:
            p["lo"] = np.minimum(p["lo"], np.log(ALPHA_RESET / (1 - ALPHA_RESET)))
        if use_adc and ADC_FROM <= t <= ADC_UNTIL and t % K_ADC == 0:
            rec = hist["prune_rec"] if t == win_hi else None
            nc, ns, npn = adc_step(p, st, accum, accum_abs, denom, rng, record=rec)
            hist["adc"].append((t, nc, ns, npn))
            n = len(p["mu"])
            accum = np.zeros(n); accum_abs = np.zeros(n); denom = np.zeros(n)
    loss, g, I, ch = loss_and_grad(p)
    hist["loss"].append(loss); hist["N"].append(len(p["mu"]))
    snaps[T_total] = dict(p={k: p[k].copy() for k in KEYS}, I=I.copy(), loss=loss)
    return p, hist, snaps


# ---- tien ich ve -------------------------------------------------------------
def show_img(ax, img, title=None):
    ax.imshow(np.clip(img, 0, 1), extent=(0, W, H, 0), interpolation="nearest")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)


def show_err(ax, img, title=None, vmax=0.5):
    e = np.abs(img - TARGET).mean(-1)
    im = ax.imshow(e, extent=(0, W, H, 0), cmap="magma", vmin=0, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return im


def draw_ellipses(ax, p, k=2.0, colors=None, lw=1.2, alpha=1.0, ls="-", label_idx=False):
    n = len(p["mu"])
    for i in range(n):
        s1, s2 = np.exp(p["logs"][i])
        col = colors[i] if colors is not None else "w"
        e = Ellipse(p["mu"][i], 2 * k * s1, 2 * k * s2, angle=np.degrees(p["th"][i]),
                    fill=False, ec=col, lw=lw, alpha=alpha, ls=ls)
        ax.add_patch(e)
        ax.plot(p["mu"][i, 0], p["mu"][i, 1], ".", color=col, ms=4, alpha=alpha)
        if label_idx:
            ax.text(p["mu"][i, 0] + 1, p["mu"][i, 1] - 1, str(i), color=col, fontsize=7)


def gauss_colors(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def territory_map(p):
    """Moi pixel → chi so Gaussian dong gop alpha lon nhat (T_i w_i); -1 = nen."""
    _, ch = render(p, return_cache=True)
    contrib = ch["T"] * ch["w"]                   # (N, HW)
    best = contrib.argmax(0)
    bestv = contrib.max(0)
    best = np.where(bestv > ch["Tend"], best, -1)  # nen thang neu T_end lon hon
    return best.reshape(H, W)


def show_territory(ax, p, title=None):
    terr = territory_map(p)
    n = len(p["mu"])
    cols = gauss_colors(n)
    rgb = np.tile(np.array([0.97, 0.97, 0.97]), (H, W, 1))
    for i in range(n):
        rgb[terr == i] = cols[i][:3]
    ax.imshow(rgb, extent=(0, W, H, 0), interpolation="nearest")
    # bien cua vat the muc tieu de doi chieu
    ax.contour(_xx + 0.5, _yy + 0.5, TARGET.mean(-1), levels=[0.55], colors="k", linewidths=0.5)
    draw_ellipses(ax, p, k=1.0, colors=["k"] * n, lw=0.5, alpha=0.6)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)


def arrows_per_gaussian(ax, p, g, scale, color="r", key="mu"):
    """Mui ten -grad mu (huong Gaussian muon dich) tai moi tam."""
    for i in range(len(p["mu"])):
        d = -g[key][i] * scale
        ax.annotate("", xy=(p["mu"][i, 0] + d[0], p["mu"][i, 1] + d[1]),
                    xytext=(p["mu"][i, 0], p["mu"][i, 1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6))


# =============================================================================
# 6.1  Anh muc tieu + luoi pixel; mot Gaussian 2D va footprint
# =============================================================================
def fig_6_1():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    ax = axes[0]
    show_img(ax, TARGET, "(a) Ảnh mục tiêu $64\\times64$ + lưới pixel")
    for k in range(0, W + 1, 4):
        ax.axvline(k, color="k", lw=0.25, alpha=0.5)
        ax.axhline(k, color="k", lw=0.25, alpha=0.5)
    ax.text(22, 24, "vùng phẳng\n(hình tròn)", ha="center", va="center", color="w", fontsize=8)
    ax.text(48, 40, "vùng chi tiết\n(dải + chấm)", ha="center", va="center", color="k", fontsize=8)

    # (b) mot Gaussian: heatmap G + 3 duong muc
    p1 = dict(mu=np.array([[40.0, 26.0]]), logs=np.log(np.array([[7.0, 3.0]])),
              th=np.array([np.radians(30)]), col=np.array([[0.2, 0.2, 0.2]]), lo=np.array([2.0]))
    _, ch = render(p1, return_cache=True)
    G = ch["G"][0].reshape(H, W)
    ax = axes[1]
    ax.imshow(G, extent=(0, W, H, 0), cmap="Greys", vmin=0, vmax=1)
    for k in range(0, W + 1, 4):
        ax.axvline(k, color="k", lw=0.2, alpha=0.4); ax.axhline(k, color="k", lw=0.2, alpha=0.4)
    for kk, ls, lab in [(1, "-", "1σ"), (2, "--", "2σ"), (3, ":", "3σ = footprint")]:
        s1, s2 = np.exp(p1["logs"][0])
        ax.add_patch(Ellipse(p1["mu"][0], 2 * kk * s1, 2 * kk * s2, angle=30, fill=False,
                             ec="tab:red", lw=1.4, ls=ls))
        ax.text(p1["mu"][0, 0] + kk * s1 * np.cos(np.radians(30)) + 1,
                p1["mu"][0, 1] + kk * s1 * np.sin(np.radians(30)) + 1, lab, color="tab:red", fontsize=8)
    ax.plot(*p1["mu"][0], "o", color="tab:red", ms=4)
    ax.text(p1["mu"][0, 0] - 3, p1["mu"][0, 1] - 3, "$\\mu$", color="tab:red", fontsize=10)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) Một Gaussian 2D: $G(x)=\\exp(-d^\\top\\Sigma^{-1}d/2)$,\n"
                 "$s=(7,3)$ px, $\\theta=30°$; pixel ngoài 3σ không được chạm")

    # (c) render cua Gaussian do len nen: alpha * G * (c - bg) + bg
    ax = axes[2]
    show_img(ax, render(p1), "(c) Render: $I(x)=T_i\\,\\alpha_iG_i(x)\\,c_i+\\dots$\n"
                             "($\\alpha=0.88$, màu xám đậm, một Gaussian)")
    ax.add_patch(Ellipse(p1["mu"][0], 6 * 7.0, 6 * 3.0, angle=30, fill=False, ec="tab:red", lw=1, ls=":"))
    ax.text(20, 58, "Ngoài footprint: chỉ còn màu nền", fontsize=8)
    save(fig, "6_1_muc-tieu-va-gaussian-2d.png")


# =============================================================================
# 6.2  THIEU (under-reconstruction): it Gaussian nho — render, sai so, gradient
# =============================================================================
def under_state():
    rng = np.random.default_rng(SEED)
    spread = [[14, 18], [30, 30], [20, 34], [40, 45], [55, 55], [52, 12]]
    p = init_params(6, rng, s=3.0, spread=spread)
    p["col"] = np.array([[0.22, 0.45, 0.80]] * 3 + [[0.12, 0.12, 0.12], [0.5, 0.4, 0.25], [0.95, 0.72, 0.10]])
    p["lo"][:] = np.log(0.85 / 0.15)
    return p


def fig_6_2():
    p = under_state()
    loss, g, I, ch = loss_and_grad(p)
    gn = np.linalg.norm(g["mu"], axis=1)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    show_img(axes[0], I, f"(a) Render với 6 Gaussian nhỏ ($s=3$ px)\nL1 = {loss:.4f}")
    draw_ellipses(axes[0], p, k=2, colors=["w"] * 6, lw=1)
    show_img(axes[1], TARGET, "(b) Mục tiêu")
    im = show_err(axes[2], I, "(c) Sai số $|I-T|$ (trung bình 3 kênh)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    ax = axes[3]
    show_img(ax, TARGET * 0.35 + 0.65, "(d) Mũi tên $-\\nabla_\\mu L$ tại mỗi Gaussian\n(độ dài ∝ ‖∇μ‖; số = ‖∇μ‖×10⁵)")
    draw_ellipses(ax, p, k=2, colors=["tab:blue"] * 6, lw=1.2)
    arrows_per_gaussian(ax, p, g, scale=12.0 / gn.max())
    for i in range(6):
        ax.text(p["mu"][i, 0] + 2.5, p["mu"][i, 1] + 4.5, f"{gn[i]*1e5:.0f}", fontsize=8,
                color="k", bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
    ax.text(2, 62, f"ngưỡng clone toy $\\tau$ = {TAU_GRAD*1e5:.0f}×10⁻⁵", fontsize=8)
    save(fig, "6_2_thieu-gaussian.png")


# =============================================================================
# 6.3  DU / QUA TO (over-reconstruction): 1 Gaussian to phu vung chi tiet
# =============================================================================
def over_state():
    rng = np.random.default_rng(SEED)
    p = init_params(1, rng, s=1.0, spread=[[48.5, 50.0]])
    p["logs"] = np.log(np.array([[13.0, 6.5]]))
    p["th"] = np.array([0.0])
    p["col"] = np.array([[0.45, 0.35, 0.30]])
    p["lo"] = np.array([np.log(0.8 / 0.2)])
    return p


def fig_6_3():
    p = over_state()
    loss, g, I, ch = loss_and_grad(p)
    gn = np.linalg.norm(g["mu"][0]); gabs = np.linalg.norm(g["mu_abs"][0])
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.4))
    show_img(axes[0], I, f"(a) Render với 1 Gaussian to $s=(13,6.5)$ px\nphủ cả dải + 5 chấm; L1 = {loss:.4f}")
    draw_ellipses(axes[0], p, k=2, colors=["w"], lw=1.2)
    show_img(axes[1], TARGET, "(b) Mục tiêu\n(vùng chi tiết ở góc dưới phải)")
    im = show_err(axes[2], I, "(c) Sai số: mờ đều,\nsai cả ở chấm lẫn khoảng trống")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    ax = axes[3]
    show_img(ax, TARGET * 0.35 + 0.65, "(d) Gradient vị trí từng pixel trong footprint\n"
                                        f"có dấu ‖Σ$_x g_x$‖ = {gn*1e5:.0f}×10⁻⁵\n"
                                        f"trị tuyệt đối Σ$_x$|$g_x$| = {gabs*1e5:.0f}×10⁻⁵ (gấp {gabs/gn:.1f} lần)")
    draw_ellipses(ax, p, k=2, colors=["tab:orange"], lw=1.5)
    gpx, gpy = g["pix"]
    gx = gpx[0].reshape(H, W); gy = gpy[0].reshape(H, W)
    step = 3
    mag = np.sqrt(gx ** 2 + gy ** 2)
    sc = 2.2 / (mag.max() + 1e-12)
    ax.quiver(_xx[::step, ::step] + 0.5, _yy[::step, ::step] + 0.5,
              -gx[::step, ::step] * sc, -gy[::step, ::step] * sc,  # huong -grad (truc y anh huong xuong)
              color="tab:red", angles="xy", scale_units="xy", scale=1, width=0.004, alpha=0.9)
    ax.set_xlim(30, 64); ax.set_ylim(64, 34)
    ax.text(31, 63, "Mũi tên kéo về nhiều phía → tổng có dấu nhỏ,\ntổng trị tuyệt đối lớn → cần SPLIT",
            fontsize=8, va="bottom", bbox=dict(fc="w", ec="none", alpha=0.85))
    save(fig, "6_3_du-gaussian-qua-to.png")


# =============================================================================
# 6.4  CLONE: sao chep y nguyen → hai ban tach ra sau vai buoc (quy dao mu)
# =============================================================================
def set_target(img):
    """Doi anh muc tieu toan cuc (dung cho vai hinh minh hoa cuc bo)."""
    global TARGET, TARGET_FLAT
    TARGET = img
    TARGET_FLAT = img.reshape(-1, 3)


def two_blob_target():
    img = np.tile(BG, (H, W, 1)).astype(float)
    x, y = _xx + 0.5, _yy + 0.5
    for cx in (17.0, 27.0):
        img[(x - cx) ** 2 + (y - 22) ** 2 <= 3.6 ** 2] = (0.85, 0.20, 0.20)
    return img


def fig_6_4():
    # canh nho: hai dom do cach nhau 10 px (vung can phu RONG hon Gaussian), mot Gaussian nho o giua (lech trai 0.5 px)
    main_target = TARGET
    set_target(two_blob_target())
    rng = np.random.default_rng(SEED)
    p = init_params(1, rng, s=4.0, spread=[[21.5, 22.0]])
    p["th"][:] = 0.0
    p["col"] = np.array([[0.85, 0.20, 0.20]]); p["lo"] = np.array([np.log(0.9 / 0.1)])
    st = adam_init(p)
    # 5 buoc mot minh de goc co trang thai Adam khac 0
    for _ in range(5):
        _, g, _, _ = loss_and_grad(p); adam_step(p, g, st)
    p_before = {k: p[k].copy() for k in KEYS}
    loss_b, g_b, I_b, _ = loss_and_grad(p)
    do_clone(p, st, np.array([0]))
    traj = [p["mu"].copy()]
    dist = []
    for _ in range(120):
        _, g, _, _ = loss_and_grad(p); adam_step(p, g, st)
        traj.append(p["mu"].copy()); dist.append(np.linalg.norm(p["mu"][0] - p["mu"][1]))
    traj = np.array(traj)
    loss_a, _, I_a, _ = loss_and_grad(p)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    ax = axes[0]
    show_img(ax, TARGET * 0.35 + 0.65, f"(a) Trước clone: 1 Gaussian nhỏ\n‖∇μ‖ = {np.linalg.norm(g_b['mu'][0])*1e5:.0f}×10⁻⁵ ≥ τ; vùng cần phủ rộng hơn → CLONE")
    draw_ellipses(ax, p_before, k=2, colors=["tab:green"], lw=1.6)
    arrows_per_gaussian(ax, p_before, g_b, scale=10.0 / np.linalg.norm(g_b["mu"][0]), color="tab:green")
    ax.set_xlim(0, 44); ax.set_ylim(44, 0)
    ax = axes[1]
    show_img(ax, TARGET * 0.35 + 0.65, "(b) Ngay sau clone: 2 bản TRÙNG NHAU\n(gốc: liền; bản sao: đứt, Adam $m=v=0$)")
    draw_ellipses(ax, p_before, k=2, colors=["tab:green"], lw=1.8)
    draw_ellipses(ax, p_before, k=2.15, colors=["tab:red"], lw=1.4, ls="--")
    ax.set_xlim(0, 44); ax.set_ylim(44, 0)
    ax = axes[2]
    show_img(ax, TARGET * 0.35 + 0.65, "(c) Sau 120 bước Adam: hai bản tách ra\n(quỹ đạo μ: gốc xanh, bản sao đỏ)")
    ax.plot(traj[:, 0, 0], traj[:, 0, 1], "-", color="tab:green", lw=1.5)
    ax.plot(traj[:, 1, 0], traj[:, 1, 1], "-", color="tab:red", lw=1.5)
    draw_ellipses(ax, p, k=2, colors=["tab:green", "tab:red"], lw=1.6)
    ax.set_xlim(0, 44); ax.set_ylim(44, 0)
    ax = axes[3]
    ax.semilogy(np.arange(1, 121), np.maximum(dist, 1e-6), color="k")
    ax.set_xlabel("bước sau clone"); ax.set_ylabel("‖μ_gốc − μ_sao‖ (pixel)")
    ax.set_title(f"(d) Khoảng cách hai bản\nL1: {loss_b:.4f} → {loss_a:.4f}")
    ax.grid(alpha=0.3)
    save(fig, "6_4_clone-truoc-sau.png")
    set_target(main_target)


# =============================================================================
# 6.5  SPLIT: Gaussian to → 2 con lay mau tu hinh dang goc, scale/1.6, xoa goc
# =============================================================================
def fig_6_5():
    rng = np.random.default_rng(SEED + 3)
    p = over_state()
    st = adam_init(p)
    for _ in range(10):
        _, g, _, _ = loss_and_grad(p); adam_step(p, g, st)
    p0 = {k: p[k].copy() for k in KEYS}
    loss0, g0, I0, _ = loss_and_grad(p)
    # dam mau eps de ve
    s = np.exp(p0["logs"][0]); c, sn = np.cos(p0["th"][0]), np.sin(p0["th"][0])
    cloud = np.random.default_rng(SEED + 99).normal(size=(300, 2)) * s   # rng rieng, khong anh huong split
    cloud = np.stack([c * cloud[:, 0] - sn * cloud[:, 1], sn * cloud[:, 0] + c * cloud[:, 1]], 1) + p0["mu"][0]
    eps, rot = do_split(p, st, np.array([0]), rng)
    p1 = {k: p[k].copy() for k in KEYS}
    loss1, _, I1, _ = loss_and_grad(p)
    traj = [p["mu"].copy()]
    for _ in range(150):
        _, g, _, _ = loss_and_grad(p); adam_step(p, g, st); traj.append(p["mu"].copy())
    traj = np.array(traj)
    loss2, _, I2, _ = loss_and_grad(p)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    zoom = dict(xlim=(28, 64), ylim=(64, 32))
    ax = axes[0, 0]
    show_img(ax, TARGET * 0.35 + 0.65, f"(a) Trước split: 1 Gaussian to, $s$=({s[0]:.1f},{s[1]:.1f}) px\n300 mẫu $R\\epsilon$, $\\epsilon\\sim N(0,\\mathrm{{diag}}\\,s^2)$")
    ax.plot(cloud[:, 0], cloud[:, 1], ".", color="tab:blue", ms=2, alpha=0.5)
    draw_ellipses(ax, p0, k=2, colors=["tab:orange"], lw=1.8)
    ax.set(**zoom)
    ax = axes[0, 1]
    show_img(ax, TARGET * 0.35 + 0.65, "(b) Ngay sau split: 2 con ở $\\mu+R\\epsilon^{(j)}$, $s/1.6$\ngốc (đứt, mờ) bị xoá")
    draw_ellipses(ax, p0, k=2, colors=["tab:orange"], lw=1.2, ls="--", alpha=0.5)
    draw_ellipses(ax, p1, k=2, colors=["tab:red", "tab:purple"], lw=1.8)
    for j in range(2):
        ax.annotate("", xy=p1["mu"][j], xytext=p0["mu"][0], arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.set(**zoom)
    ax = axes[0, 2]
    show_img(ax, TARGET * 0.35 + 0.65, "(c) Sau 150 bước Adam: hai con định vị lại,\nmỗi con lo một phần vùng chi tiết")
    for j, col in enumerate(["tab:red", "tab:purple"]):
        ax.plot(traj[:, j, 0], traj[:, j, 1], "-", color=col, lw=1.4)
    draw_ellipses(ax, p, k=2, colors=["tab:red", "tab:purple"], lw=1.8)
    ax.set(**zoom)
    show_img(axes[1, 0], I0, f"(d) Render trước split — L1 = {loss0:.4f}"); axes[1, 0].set(**zoom)
    show_img(axes[1, 1], I1, f"(e) Render ngay sau split — L1 = {loss1:.4f}\n(thường xấu đi tạm thời)"); axes[1, 1].set(**zoom)
    show_img(axes[1, 2], I2, f"(f) Render sau 150 bước — L1 = {loss2:.4f}"); axes[1, 2].set(**zoom)
    save(fig, "6_5_split-truoc-sau.png")


# =============================================================================
# Chay day du (dung chung cho 6.6 – 6.11)
# =============================================================================
_RUNS = {}


def full_run():
    if "adc" in _RUNS:
        return _RUNS
    rng = np.random.default_rng(SEED)
    N0 = 12
    p0 = init_sfm_like(N0, rng, s=3.0)
    snaps_at = (0, 100, 300, 600, 1000)
    t0 = time.time()
    p_adc, hist_adc, snaps_adc = train({k: v.copy() for k, v in p0.items()}, True, snapshots=snaps_at)
    print(f"  run ADC: N {N0} → {len(p_adc['mu'])}, L1 {hist_adc['loss'][0]:.4f} → {hist_adc['loss'][-1]:.4f}, {time.time()-t0:.1f}s")
    t0 = time.time()
    p_no, hist_no, snaps_no = train({k: v.copy() for k, v in p0.items()}, False)
    print(f"  run no-ADC N={N0}: L1 → {hist_no['loss'][-1]:.4f}, {time.time()-t0:.1f}s")
    n_end = len(p_adc["mu"])
    rng2 = np.random.default_rng(SEED)
    p_big0 = init_params(n_end, rng2, s=4.0)
    t0 = time.time()
    p_big, hist_big, _ = train(p_big0, False)
    print(f"  run no-ADC N={n_end} ngẫu nhiên: L1 → {hist_big['loss'][-1]:.4f}, {time.time()-t0:.1f}s")
    _RUNS.update(adc=(p_adc, hist_adc, snaps_adc), no=(p_no, hist_no, snaps_no),
                 big=(p_big, hist_big), p0=p0, snaps_at=snaps_at)
    return _RUNS


def _snap_titles(hist, snaps_at):
    titles = []
    adc_t = [a[0] for a in hist["adc"]]
    for t in snaps_at:
        k = sum(1 for a in adc_t if a <= t)
        if t == 0:
            titles.append("t = 0 (khởi tạo)")
        elif t >= T_TOTAL:
            titles.append(f"t = {t} (cuối)")
        else:
            titles.append(f"t = {t} (sau ADC lần {k})")
    return titles


# =============================================================================
# 6.6  Chuoi thoi gian: render + ellipse tai cac moc
# =============================================================================
def fig_6_6():
    R = full_run()
    p_adc, hist, snaps = R["adc"]
    keys = sorted(snaps.keys())
    titles = _snap_titles(hist, keys)
    fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 6.6))
    for j, t in enumerate(keys):
        sn = snaps[t]
        n = len(sn["p"]["mu"])
        show_img(axes[0, j], sn["I"], f"{titles[j]}\nN = {n}, L1 = {sn['loss']:.4f}")
        show_img(axes[1, j], TARGET * 0.3 + 0.7)
        draw_ellipses(axes[1, j], sn["p"], k=1.5, colors=gauss_colors(n), lw=1.3)
    axes[0, 0].set_ylabel("Render", fontsize=10)
    axes[1, 0].set_ylabel("Ellipse 1.5σ của từng Gaussian\ntrên nền mục tiêu (mờ)", fontsize=10)
    fig.suptitle("Gradient descent + ADC lặp lại: các Gaussian tự dịch, co giãn, nhân lên và phủ dần ảnh mục tiêu", fontsize=11)
    save(fig, "6_6_chuoi-thoi-gian.png")


# =============================================================================
# 6.7  "Chia vung": pixel → Gaussian dong gop lon nhat
# =============================================================================
def fig_6_7():
    R = full_run()
    p_adc, hist, snaps = R["adc"]
    keys = sorted(snaps.keys())
    titles = _snap_titles(hist, keys)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.7))
    for j, t in enumerate(keys):
        sn = snaps[t]
        show_territory(axes[j], sn["p"], f"{titles[j]}\nN = {len(sn['p']['mu'])}")
    fig.suptitle("Lãnh thổ của từng Gaussian (pixel tô theo Gaussian có $T_i\\alpha_iG_i$ lớn nhất; trắng = nền; "
                 "đường đen = biên vật thể mục tiêu)", fontsize=10)
    save(fig, "6_7_chia-vung.png")


# =============================================================================
# 6.8  Duong cong loss, N, so clone/split/prune
# =============================================================================
def fig_6_8():
    R = full_run()
    _, hist, _ = R["adc"]
    _, hist_no, _ = R["no"]
    _, hist_big = R["big"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axes[0]
    ax.semilogy(hist["loss"], color="tab:blue", label="có ADC")
    ax.semilogy(hist_no["loss"], color="tab:gray", label=f"không ADC, N = {hist_no['N'][0]}")
    ax.semilogy(hist_big["loss"], color="tab:olive", ls="--", label=f"không ADC, N = {hist_big['N'][0]} (ngẫu nhiên)")
    for (t, nc, ns, npn) in hist["adc"]:
        ax.axvline(t, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("iteration"); ax.set_ylabel("L1 loss"); ax.set_title("(a) Loss (vạch dọc = mốc ADC)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.step(np.arange(len(hist["N"])), hist["N"], color="tab:blue", where="post", label="có ADC")
    ax.plot(hist_no["N"], color="tab:gray", label="không ADC")
    ax.axvspan(ADC_FROM, ADC_UNTIL, color="tab:orange", alpha=0.08, label="cửa sổ densify")
    ax.set_xlabel("iteration"); ax.set_ylabel("N"); ax.set_title("(b) Số Gaussian $N(t)$ — bậc thang tại mốc ADC")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    ts = [a[0] for a in hist["adc"]]
    ncl = [a[1] for a in hist["adc"]]; nsp = [a[2] for a in hist["adc"]]; npr = [a[3] for a in hist["adc"]]
    wdt = K_ADC * 0.27
    ax.bar(np.array(ts) - wdt, ncl, width=wdt, color="tab:green", label="clone")
    ax.bar(np.array(ts), nsp, width=wdt, color="tab:red", label="split")
    ax.bar(np.array(ts) + wdt, npr, width=wdt, color="tab:gray", label="prune")
    ax.set_xlabel("mốc ADC (iteration)"); ax.set_ylabel("số Gaussian"); ax.set_title("(c) Clone / split / prune tại mỗi mốc")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    save(fig, "6_8_duong-cong-loss-N.png")


# =============================================================================
# 6.9  Histogram kich thuoc dau vs cuoi; scatter kich thuoc vs do chi tiet
# =============================================================================
def footprint_detail(p):
    """Trung binh gradient anh muc tieu trong footprint (q<9) cua tung Gaussian."""
    _, ch = render(p, return_cache=True)
    G = ch["G"]
    out = []
    for i in range(len(p["mu"])):
        m = G[i] > 1e-6
        out.append(DETAIL.ravel()[m].mean() if m.any() else 0.0)
    return np.array(out)


def fig_6_9():
    R = full_run()
    p_adc, hist, snaps = R["adc"]
    p0 = snaps[0]["p"]
    s0 = np.exp(p0["logs"]).max(1); s1 = np.exp(p_adc["logs"]).max(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    bins = np.linspace(0, 18, 19)
    ax.hist(s0, bins=bins, alpha=0.6, color="tab:gray", label=f"t = 0 (N = {len(s0)})")
    ax.hist(s1, bins=bins, alpha=0.6, color="tab:blue", label=f"cuối (N = {len(s1)})")
    ax.axvline(S_BIG, color="tab:red", ls="--", label=f"ngưỡng to/nhỏ toy = {S_BIG} px")
    ax.set_xlabel("max scale $\\max_k s_{i,k}$ (pixel)"); ax.set_ylabel("số Gaussian")
    ax.set_title("(a) Phân bố kích thước: đầu vs cuối"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    det = footprint_detail(p_adc)
    alpha = sigmoid(p_adc["lo"])
    sc = ax.scatter(det, s1, c=alpha, cmap="viridis", s=28, edgecolor="k", lw=0.3, vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, label="opacity α")
    ax.set_xlabel("độ chi tiết vùng phủ (gradient ảnh mục tiêu trung bình trong footprint)")
    ax.set_ylabel("max scale (pixel)")
    ax.set_title("(b) Gaussian cuối: to ở vùng phẳng, nhỏ ở vùng chi tiết")
    ax.grid(alpha=0.3)
    save(fig, "6_9_kich-thuoc-vs-chi-tiet.png")


# =============================================================================
# 6.10 Doi chieu: khong ADC vs co ADC
# =============================================================================
def fig_6_10():
    R = full_run()
    p_adc, hist, _ = R["adc"]
    p_no, hist_no, _ = R["no"]
    p_big, hist_big = R["big"]
    rows = [("Không ADC, N cố định = %d" % len(p_no["mu"]), p_no, hist_no),
            ("Không ADC, N = %d ngẫu nhiên từ đầu" % len(p_big["mu"]), p_big, hist_big),
            ("Có ADC: %d → %d" % (hist["N"][0], len(p_adc["mu"])), p_adc, hist)]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11))
    for r, (name, p, h) in enumerate(rows):
        I = render(p)
        show_img(axes[r, 0], I, f"{name}\nrender cuối, L1 = {h['loss'][-1]:.4f}")
        im = show_err(axes[r, 1], I, "sai số $|I-T|$")
        show_img(axes[r, 2], TARGET * 0.3 + 0.7, "ellipse 1.5σ")
        draw_ellipses(axes[r, 2], p, k=1.5, colors=gauss_colors(len(p["mu"])), lw=1.2)
    fig.colorbar(im, ax=axes[:, 1], fraction=0.03)
    save(fig, "6_10-khong-adc-vs-co-adc.png".replace("6_10-", "6_10_"))


# =============================================================================
# 6.11 Prune: opacity giam dan → bi xoa, anh khong doi
# =============================================================================
def fig_6_11():
    R = full_run()
    p_adc, hist, snaps = R["adc"]
    rec = hist["prune_rec"]
    ts = sorted(hist["alpha_win"].keys())
    A = np.array([hist["alpha_win"][t] for t in ts])          # (T_win, N) — N on dinh trong cua so
    n_win = A.shape[1]
    keep = rec["keep"]
    # Gaussian moi (clone/split tai moc cuoi cua so) nam sau n_win → khong co lich su
    keep_old = keep[:n_win]
    t_prune = ts[-1] + 1
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
    ax = axes[0]
    for i in range(n_win):
        col = "tab:red" if not keep_old[i] else "tab:gray"
        ax.plot(ts, A[:, i], color=col, lw=1.4 if not keep_old[i] else 0.7, alpha=1 if not keep_old[i] else 0.5)
    ax.axvline(RESET_AT, color="k", ls=":", lw=1)
    ax.text(RESET_AT + 1, 0.93, f"reset: α ← min(α, {ALPHA_RESET})", fontsize=8)
    ax.axhline(EPS_OPA, color="tab:red", ls="--", lw=1)
    ax.text(ts[0] + 1, EPS_OPA + 0.02, f"ε = {EPS_OPA}: dưới ngưỡng tại mốc t={t_prune} → prune", fontsize=8, color="tab:red")
    ax.set_xlabel("iteration"); ax.set_ylabel("opacity α"); ax.set_ylim(0, 1)
    ax.set_title(f"(a) Opacity trong cửa sổ [{ts[0]}, {ts[-1]}]\nxám: hồi phục; đỏ: không hồi phục ({int((~keep_old).sum())} Gaussian)")
    ax.grid(alpha=0.3)
    show_img(axes[1], hist["I_reset"], f"(b) Render ngay trước reset (t = {RESET_AT})\nL1 = {hist['loss'][RESET_AT-1]:.4f}")
    pb = rec["p_before"]
    show_img(axes[2], rec["I_before"], f"(c) t = {t_prune}, TRƯỚC prune: N = {len(pb['mu'])}\nellipse đỏ = Gaussian có α < ε")
    cols = ["tab:red" if not keep[i] else "tab:gray" for i in range(len(pb["mu"]))]
    draw_ellipses(axes[2], pb, k=1.5, colors=cols, lw=1.2, alpha=0.9)
    pa = rec["p_after"]
    dI = np.abs(rec["I_after"] - rec["I_before"]).max()
    show_img(axes[3], rec["I_after"], f"(d) SAU prune: N = {len(pa['mu'])}\nmax|ΔI| = {dI:.4f} — ảnh gần như không đổi")
    draw_ellipses(axes[3], pa, k=1.5, colors=["tab:gray"] * len(pa["mu"]), lw=0.8, alpha=0.7)
    save(fig, "6_11_prune-opacity.png")


# =============================================================================
if __name__ == "__main__":
    t_all = time.time()
    for f in [fig_6_1, fig_6_2, fig_6_3, fig_6_4, fig_6_5, fig_6_6, fig_6_7,
              fig_6_8, fig_6_9, fig_6_10, fig_6_11]:
        t0 = time.time()
        f()
        print(f"   {f.__name__}: {time.time()-t0:.1f}s")
    R = full_run()
    _, hist, _ = R["adc"]
    print("ADC toy log (t, clone, split, prune):", hist["adc"])
    print(f"Tổng thời gian: {time.time()-t_all:.1f}s")
