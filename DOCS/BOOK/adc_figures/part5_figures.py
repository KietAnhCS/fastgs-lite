"""
Hinh minh hoa cho PHAN 5 chuong Adaptive Density Control (FastGS-lite):
prune multinomial, ep opacity, tia cuoi, doi chieu 3DGS vs FastGS-lite,
dau ra khoi, so do luong du lieu.

Moi ham fig_5_k() xuat 1 PNG "5_k_ten-ngan.png" vao cung thu muc adc_figures/.
Chi dung numpy + matplotlib (khong torch). Mo phong deu dat seed co dinh.

Chay: python adc_figures/part5_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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
# Cac ham dung chung
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inverse_sigmoid(a):
    return np.log(a / (1.0 - a))


def weights_from_pruning(P):
    """w_i = 1 / (1e-6 + (1 - Pruning_i))  -- dung nhu gaussian_model.py:478"""
    return 1.0 / (1e-6 + (1.0 - np.asarray(P, dtype=float)))


def multinomial_no_replacement(rng, w, k):
    """Rut k chi so khong hoan lai, xac suat ti le w tren phan con lai
    (giong torch.multinomial(w, k, replacement=False))."""
    w = np.asarray(w, dtype=float).copy()
    out = []
    for _ in range(k):
        s = w.sum()
        if s <= 0:
            break
        i = rng.choice(len(w), p=w / s)
        out.append(i)
        w[i] = 0.0
    return np.array(out, dtype=int)


def exact_prob_in_S_budget2(w):
    """Xac suat ly thuyet chi so i nam trong S khi budget = 2, khong hoan lai."""
    w = np.asarray(w, dtype=float)
    W = w.sum()
    p1 = w / W
    p = p1.copy()
    for i in range(len(w)):
        for j in range(len(w)):
            if j != i:
                p[i] += p1[j] * w[i] / (W - w[j])
    return p


# Kich ban GIA DINH N = 10 (so tu file goc 12-adaptive-density-control.md)
P10 = np.array([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
C10 = np.array([3, 6, 8, 9, 10]) - 1          # tap ung vien (chi so 0-based)
BUDGET10 = int(0.5 * len(C10))                # = 2


# ---------------------------------------------------------------------------
# 5.1  Trong so va xac suat bi chon cua 10 Gaussian gia dinh
# ---------------------------------------------------------------------------
def fig_5_1():
    w = weights_from_pruning(P10)
    p = w / w.sum()
    idx = np.arange(1, 11)
    inC = np.isin(idx - 1, C10)
    col = np.where(inC, "#d62728", "#7f7f7f")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    ax.bar(idx, P10, color=col)
    ax.set_xticks(idx)
    ax.set_xlabel("Gaussian i")
    ax.set_ylabel("Pruning_i")
    ax.set_title("Pruning score (đỏ = thuộc tập ứng viên C)")
    ax.axhline(0.9, ls="--", c="k", lw=0.8)
    ax.text(1, 0.92, "0.9 (ngưỡng final prune)", fontsize=7)

    ax = axes[1]
    ax.bar(idx, w, color=col)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xlabel("Gaussian i")
    ax.set_ylabel("w_i = 1/(1e-6 + 1 - Pruning_i)  (log)")
    ax.set_title("Trọng số: phân kỳ khi Pruning → 1")
    for i, v in zip(idx, w):
        ax.text(i, v * 1.4, f"{v:.3g}", ha="center", fontsize=6.5)

    ax = axes[2]
    ax.bar(idx, p, color=col)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xlabel("Gaussian i")
    ax.set_ylabel("p_i = w_i / sum w  (log)")
    ax.set_title("Xác suất rút ở lượt đầu: G10 chiếm 99.99%")
    ax.set_ylim(1e-7, 2)
    for i, v in zip(idx, p):
        ax.text(i, v * 1.6, f"{v:.1e}", ha="center", fontsize=6)
    save(fig, "5_1_trong-so-N10.png")


# ---------------------------------------------------------------------------
# 5.2  Mo phong 2000 lan lay mau: tan suat bi xoa vs xac suat ly thuyet
# ---------------------------------------------------------------------------
def fig_5_2():
    rng = np.random.default_rng(0)
    w = weights_from_pruning(P10)
    T = 2000
    inS = np.zeros(10)
    removed = np.zeros(10)
    n_removed = []
    for _ in range(T):
        S = multinomial_no_replacement(rng, w, BUDGET10)
        inS[S] += 1
        rm = np.intersect1d(S, C10)
        removed[rm] += 1
        n_removed.append(len(rm))
    inS /= T
    removed /= T
    p_th = exact_prob_in_S_budget2(w)
    idx = np.arange(1, 11)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    ax = axes[0]
    bw = 0.38
    ax.bar(idx - bw / 2, p_th, bw, label="P(i ∈ S) lý thuyết", color="#1f77b4")
    ax.bar(idx + bw / 2, inS, bw, label=f"tần suất i ∈ S ({T} lần)", color="#aec7e8")
    ax.set_xticks(idx)
    ax.set_xlabel("Gaussian i")
    ax.set_ylabel("xác suất / tần suất")
    ax.set_title("Được rút vào S (budget 2)")
    ax.legend(fontsize=7)

    ax = axes[1]
    inC = np.isin(idx - 1, C10)
    ax.bar(idx - bw / 2, np.where(inC, p_th, 0), bw, color="#d62728", label="lý thuyết: P(i ∈ S) nếu i ∈ C")
    ax.bar(idx + bw / 2, removed, bw, color="#ff9896", label="tần suất BỊ XOÁ (S ∩ C)")
    ax.set_xticks(idx)
    ax.set_xlabel("Gaussian i")
    ax.set_title("Bị xoá = S ∩ C: G7 hay được rút nhưng không xoá")
    ax.legend(fontsize=7)

    ax = axes[2]
    vals, cnts = np.unique(n_removed, return_counts=True)
    ax.bar(vals, cnts / T, color="#2ca02c")
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("số Gaussian bị xoá thực tế")
    ax.set_ylabel("tỉ lệ")
    ax.set_title(f"budget = 2, xoá thực tế TB = {np.mean(n_removed):.3f}")
    for v, c in zip(vals, cnts):
        ax.text(v, c / T + 0.01, f"{c / T:.3f}", ha="center", fontsize=8)
    save(fig, "5_2_mo-phong-2000-lan.png")
    return removed, np.mean(n_removed), p_th


# ---------------------------------------------------------------------------
# 5.3  Top-k cung vs multinomial tren 300 Gaussian mo phong
# ---------------------------------------------------------------------------
def _population_300(seed=1):
    rng = np.random.default_rng(seed)
    N = 300
    raw = rng.beta(2.0, 2.5, N)                 # diem tho
    P = (raw - raw.min()) / (raw.max() - raw.min())   # minmax qua N nhu code
    # tap ung vien: 40% quan the (opacity thap / to qua) -- chon ngau nhien, doc lap voi score
    cand = rng.random(N) < 0.40
    return rng, N, P, cand


def fig_5_3():
    rng, N, P, cand = _population_300()
    C = np.where(cand)[0]
    budget = int(0.5 * len(C))
    w = weights_from_pruning(P)

    # (a) top-k cung: sap xep giam dan Pruning trong C, lay budget phan tu
    order = C[np.argsort(-P[C])]
    hard_rm = order[:budget]

    # (b) multinomial tren ca N, roi giao voi C (1 lan rut, seed 0)
    rng2 = np.random.default_rng(0)
    S = multinomial_no_replacement(rng2, w, budget)
    multi_rm = np.intersect1d(S, C)

    # (c) tan suat bi xoa qua 500 lan rut
    T = 500
    freq = np.zeros(N)
    for _ in range(T):
        S_ = multinomial_no_replacement(rng2, w, budget)
        freq[np.intersect1d(S_, C)] += 1
    freq /= T

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    ax = axes[0, 0]
    ax.scatter(np.arange(N), P, s=8, c="#bbbbbb", label="không thuộc C")
    ax.scatter(C, P[C], s=12, c="#1f77b4", label=f"ứng viên C (|C|={len(C)})")
    ax.scatter(hard_rm, P[hard_rm], s=40, marker="x", c="#d62728", label=f"top-k cứng xoá ({len(hard_rm)})")
    ax.axhline(P[order[budget - 1]], ls="--", c="#d62728", lw=0.8)
    ax.set_xlabel("chỉ số Gaussian")
    ax.set_ylabel("Pruning_i")
    ax.set_title("(a) Ngưỡng cứng: xoá đúng budget, cắt sạch trên 1 đường")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[0, 1]
    ax.scatter(np.arange(N), P, s=8, c="#bbbbbb")
    ax.scatter(C, P[C], s=12, c="#1f77b4")
    ax.scatter(S, P[S], s=30, facecolors="none", edgecolors="#ff7f0e", label=f"S (rút trên cả N, |S|={len(S)})")
    ax.scatter(multi_rm, P[multi_rm], s=40, marker="x", c="#d62728", label=f"xoá = S ∩ C ({len(multi_rm)})")
    ax.set_xlabel("chỉ số Gaussian")
    ax.set_ylabel("Pruning_i")
    ax.set_title("(b) Multinomial (seed 0): xoá ít hơn budget, có ngẫu nhiên")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 21)
    keep_hard = np.setdiff1d(np.arange(N), hard_rm)
    keep_multi = np.setdiff1d(np.arange(N), multi_rm)
    ax.hist(P, bins, alpha=0.35, color="#7f7f7f", label="trước prune")
    ax.hist(P[keep_hard], bins, histtype="step", lw=1.6, color="#d62728", label="còn lại sau top-k cứng")
    ax.hist(P[keep_multi], bins, histtype="step", lw=1.6, color="#1f77b4", ls="--", label="còn lại sau multinomial")
    ax.set_xlabel("Pruning_i")
    ax.set_ylabel("số Gaussian")
    ax.set_title("(c) Phân bố score còn lại")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.scatter(P[C], freq[C], s=14, c="#1f77b4")
    ax.set_xlabel("Pruning_i (chỉ ứng viên C)")
    ax.set_ylabel(f"tần suất bị xoá ({T} lần rút)")
    ax.set_title("(d) Xác suất bị xoá tăng mạnh khi Pruning → 1")
    ax.axvline(P[order[budget - 1]], ls="--", c="#d62728", lw=0.8)
    ax.text(P[order[budget - 1]] + 0.01, 0.5, "ngưỡng top-k", fontsize=7, color="#d62728")
    save(fig, "5_3_topk-vs-multinomial.png")
    return len(C), budget, len(hard_rm), len(multi_rm)


# ---------------------------------------------------------------------------
# 5.4  Vi sao ngau nhien tot hon nguong cung: tranh xoa ca cum
# ---------------------------------------------------------------------------
def fig_5_4():
    rng = np.random.default_rng(3)
    N = 300
    xy = rng.uniform(0, 10, (N, 2))
    # cum "nong" o goc: score cao dong deu 0.86..0.95; noi khac 0.55..0.92
    center = np.array([7.5, 7.5])
    d = np.linalg.norm(xy - center, axis=1)
    incl = d < 1.6
    P = rng.uniform(0.55, 0.92, N)
    P[incl] = rng.uniform(0.86, 0.95, incl.sum())
    cand = np.ones(N, bool)          # gia dinh moi Gaussian deu la ung vien
    C = np.where(cand)[0]
    budget = int(0.15 * N)           # xoa 15% de hinh de nhin
    w = weights_from_pruning(P)

    order = np.argsort(-P)
    hard_rm = order[:budget]
    S = multinomial_no_replacement(rng, w, budget)
    multi_rm = np.intersect1d(S, C)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax = axes[0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=P, s=16, cmap="viridis", vmin=0.5, vmax=1)
    ax.add_patch(plt.Circle(center, 1.6, fill=False, ls="--", color="r"))
    ax.set_title("Pruning_i theo vị trí (vòng đỏ: cụm điểm cao)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Pruning_i")

    for ax, rm, name in [(axes[1], hard_rm, "top-k cứng"), (axes[2], multi_rm, "multinomial")]:
        ax.scatter(xy[:, 0], xy[:, 1], c="#cccccc", s=10)
        ax.scatter(xy[rm, 0], xy[rm, 1], c="#d62728", s=22, marker="x")
        ax.add_patch(plt.Circle(center, 1.6, fill=False, ls="--", color="r"))
        n_in = np.isin(rm, np.where(incl)[0]).sum()
        ax.set_title(f"{name}: xoá {len(rm)}, trong cụm {n_in}/{incl.sum()}")
        ax.set_aspect("equal")
    save(fig, "5_4_tranh-xoa-ca-cum.png")


# ---------------------------------------------------------------------------
# 5.5  sigmoid va inverse_sigmoid voi cac moc opacity quan trong
# ---------------------------------------------------------------------------
def fig_5_5():
    x = np.linspace(-7, 7, 600)
    a = np.linspace(0.001, 0.999, 600)
    marks = [(0.005, "prune 0.005"), (0.01, "reset 0.01"), (0.1, "final prune 0.1"), (0.8, "clamp 0.8")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(x, sigmoid(x), lw=2)
    for v, lab in marks:
        xv = inverse_sigmoid(v)
        ax.plot([xv, xv], [0, v], ls=":", c="gray", lw=0.8)
        ax.plot([-7, xv], [v, v], ls=":", c="gray", lw=0.8)
        ax.plot(xv, v, "o", c="#d62728", ms=4)
        ax.annotate(f"{lab}\nlogit={xv:.3f}", (xv, v), xytext=(6, -2), textcoords="offset points", fontsize=7)
    ax.set_xlabel("logit  (tham số _opacity thực sự được học)")
    ax.set_ylabel("alpha = sigmoid(logit)")
    ax.set_title("sigmoid: đầu ra luôn trong (0,1)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(a, inverse_sigmoid(a), lw=2)
    for v, lab in marks:
        ax.plot(v, inverse_sigmoid(v), "o", c="#d62728", ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (thang log)")
    ax.set_ylabel("inverse_sigmoid(alpha) = log(alpha/(1-alpha))")
    ax.set_title("inverse_sigmoid: 0.9 → 0.8 là bước nhỏ, 0.9 → 0.01 là bước rất xa")
    ax.grid(alpha=0.3, which="both")
    ax.annotate("", xy=(0.8, inverse_sigmoid(0.8)), xytext=(0.9, inverse_sigmoid(0.9)),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2))
    ax.annotate("", xy=(0.01, inverse_sigmoid(0.01)), xytext=(0.9, inverse_sigmoid(0.9)),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
    ax.text(0.5, 1.9, "(1) 2.197 → 1.386", color="#2ca02c", fontsize=8)
    ax.text(0.03, -1.0, "(2) 2.197 → −4.595", color="#d62728", fontsize=8)
    save(fig, "5_5_sigmoid-inverse.png")


# ---------------------------------------------------------------------------
# 5.6  Duong opacity cua vai Gaussian theo iteration, co cac moc reset
# ---------------------------------------------------------------------------
def _simulate_opacity(T=15000, reset_every=3000, seed=0):
    """Mo phong don gian: logit tang theo 'suc keo' k_i cua gradient (Gaussian
    huu ich k>0, vo dung k<=0), clamp 0.8 sau moi densify (100 vong),
    reset ve inverse_sigmoid(0.01) moi 3000 vong. Chi de minh hoa."""
    rng = np.random.default_rng(seed)
    names = ["hữu ích (k=+0.006)", "trung bình (k=+0.002)", "vô dụng (k=−0.001)", "hồi sinh muộn (k=0 rồi +0.004)"]
    k = np.array([0.006, 0.002, -0.001, 0.0])
    logit = np.full(4, inverse_sigmoid(0.1))
    traj = np.zeros((T + 1, 4))
    traj[0] = sigmoid(logit)
    for t in range(1, T + 1):
        kk = k.copy()
        if t > 7000:
            kk[3] = 0.004
        logit = logit + kk + rng.normal(0, 0.003, 4)
        if t % 100 == 0 and 500 < t < 15000:
            logit = np.minimum(logit, inverse_sigmoid(0.8))
        if t % reset_every == 0:
            logit = np.minimum(logit, inverse_sigmoid(0.01))
        traj[t] = sigmoid(logit)
    return traj, names


def fig_5_6():
    traj, names = _simulate_opacity()
    T = traj.shape[0] - 1
    t = np.arange(T + 1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, n in enumerate(names):
        ax.plot(t, traj[:, i], lw=1.4, label=n)
    for r in range(3000, T + 1, 3000):
        ax.axvline(r, c="k", ls="--", lw=0.8)
        ax.text(r + 60, 0.93, f"reset\n{r}", fontsize=7)
    ax.axhline(0.8, c="gray", ls=":", lw=0.8)
    ax.text(200, 0.81, "clamp 0.8 sau mỗi densify", fontsize=7, color="gray")
    ax.axhline(0.005, c="#d62728", ls=":", lw=0.8)
    ax.text(200, 0.015, "prune 0.005", fontsize=7, color="#d62728")
    ax.axhline(0.01, c="#ff7f0e", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.002, 1.05)
    ax.set_xlabel("iteration t")
    ax.set_ylabel("alpha_i (log)")
    ax.set_title("Mô phỏng (minh hoạ): reset 3000 vòng kéo alpha về 0.01, Gaussian hữu ích tự leo lại, vô dụng rơi xuống ngưỡng prune")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "5_6_opacity-theo-iteration.png")


# ---------------------------------------------------------------------------
# 5.7  Histogram opacity truoc / ngay sau reset / 500 vong sau
# ---------------------------------------------------------------------------
def fig_5_7():
    rng = np.random.default_rng(5)
    N = 20000
    # truoc reset: hon hop -- 60% huu ich (alpha cao), 40% "lo lung"
    useful = rng.random(N) < 0.6
    alpha0 = np.where(useful, rng.beta(6, 2, N), rng.beta(1.5, 4, N))
    alpha0 = np.clip(alpha0, 1e-4, 0.999)
    logit0 = inverse_sigmoid(alpha0)
    logit1 = np.minimum(logit0, inverse_sigmoid(0.01))
    # 500 vong sau: huu ich leo lai nhanh, vo dung leo cham / dung yen
    gain = np.where(useful, rng.normal(3.5, 0.8, N), rng.normal(-0.3, 0.9, N))
    logit2 = logit1 + gain
    a1, a2 = sigmoid(logit1), sigmoid(logit2)

    bins = np.logspace(-3.5, 0, 45)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    for ax, a, ttl in [(axes[0], alpha0, "trước reset"), (axes[1], a1, "ngay sau reset: min(alpha, 0.01)"),
                       (axes[2], a2, "~500 vòng sau (minh hoạ)")]:
        ax.hist(a, bins, color="#1f77b4", alpha=0.8)
        ax.set_xscale("log")
        ax.axvline(0.005, c="#d62728", ls="--", lw=1)
        ax.axvline(0.01, c="#ff7f0e", ls=":", lw=1)
        ax.set_xlabel("alpha (log)")
        ax.set_title(ttl)
        below = (a < 0.005).mean()
        ax.text(0.02, 0.92, f"alpha<0.005: {below * 100:.1f}%", transform=ax.transAxes, fontsize=8, color="#d62728")
    axes[0].set_ylabel("số Gaussian")
    save(fig, "5_7_histogram-truoc-sau-reset.png")


# ---------------------------------------------------------------------------
# 5.8  Adam sau khi xoa moment: buoc dau tien lon hon lr (khong bias-correct)
# ---------------------------------------------------------------------------
def fig_5_8():
    lr, b1, b2, eps = 0.025, 0.9, 0.999, 1e-15
    g = 0.5   # gradient hang so (gia dinh) tren logit opacity
    steps = 40

    def run(step0):
        m = v = 0.0
        out = []
        for s in range(1, steps + 1):
            t = step0 + s
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            out.append(lr * mhat / (np.sqrt(vhat) + eps))
        return np.array(out)

    fresh = run(0)          # step = 0: bias-correct dung
    kept = run(6000)        # step giu nguyen 6000: bias-correct ~ 1

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(np.arange(1, steps + 1), kept / lr, "o-", ms=3, label="reset (m,v)=(0,0), step GIỮ NGUYÊN (code thật)")
    ax.plot(np.arange(1, steps + 1), fresh / lr, "s-", ms=3, label="Adam mới hoàn toàn (step=0, bias-correct đúng)")
    ax.axhline(1, c="gray", ls=":")
    ax.set_xlabel("số bước sau reset")
    ax.set_ylabel("|bước cập nhật| / lr")
    ax.set_title(f"Bước cập nhật đầu tiên sau reset ≈ {kept[0] / lr:.2f} × lr (= 0.1/sqrt(0.001)), rồi hội tụ về 1 × lr")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "5_8_adam-sau-reset.png")


# ---------------------------------------------------------------------------
# 5.9  Timeline toan bo 30k vong: densify / reset / final prune
# ---------------------------------------------------------------------------
def fig_5_9():
    fig, ax = plt.subplots(figsize=(12, 3.2))
    # densify moi 100 (500 < t < 15000)
    dens = np.arange(600, 15000, 100)
    ax.vlines(dens, 0.0, 0.25, color="#1f77b4", lw=0.4, alpha=0.6)
    ax.text(7500, 0.28, "densify + prune multinomial: mỗi 100 vòng, 500 < t < 15000 (145 lần)", ha="center", fontsize=8, color="#1f77b4")
    resets = np.arange(3000, 15001, 3000)
    ax.vlines(resets, 0.45, 0.7, color="#ff7f0e", lw=2)
    for r in resets:
        ax.text(r, 0.72, f"{r}", ha="center", fontsize=7, color="#ff7f0e")
    ax.text(7500, 0.82, "reset_opacity: t % 3000 == 0 và t < 15000 (5 lần: 3k..15k)", ha="center", fontsize=8, color="#ff7f0e")
    fp = [18000, 21000, 24000, 27000]
    ax.vlines(fp, 0.95, 1.2, color="#d62728", lw=2)
    for r in fp:
        ax.text(r, 1.22, f"{r}", ha="center", fontsize=7, color="#d62728")
    ax.vlines([30000], 0.95, 1.2, color="#d62728", lw=2, ls=":")
    ax.text(30000, 1.22, "30000\nKHÔNG chạy\n(t < 30000)", ha="center", fontsize=7, color="#d62728")
    ax.text(22500, 1.36, "final_prune_fastgs: t % 3000 == 0, 15000 < t < 30000 (4 lần)", ha="center", fontsize=8, color="#d62728")
    ax.axvspan(0, 15000, color="#1f77b4", alpha=0.04)
    ax.axvspan(15000, 30000, color="#d62728", alpha=0.04)
    ax.text(15000, -0.12, "15000 = densify_until_iter", ha="center", fontsize=7)
    ax.set_xlim(0, 31000)
    ax.set_ylim(-0.2, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("iteration t")
    ax.set_title("Lịch chạy của ba cơ chế ADC trên 30 000 vòng")
    save(fig, "5_9_timeline-final-prune.png")


# ---------------------------------------------------------------------------
# 5.10  Vung OR tren mat phang (opacity, s_p) voi 5 Gaussian cua 3.md
# ---------------------------------------------------------------------------
def fig_5_10():
    alpha = np.array([0.85, 0.62, 0.40, 0.07, 0.55])
    sp = np.array([0.000, 0.057, 1.000, 0.610, 0.367])
    names = ["G1", "G2", "G3", "G4", "G5"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.add_patch(Rectangle((0, 0), 0.1, 1.0, color="#d62728", alpha=0.18, label="alpha < 0.1  → xoá"))
    ax.add_patch(Rectangle((0, 0.9), 1.0, 0.1, color="#ff7f0e", alpha=0.25, label="s_p > 0.9  → xoá"))
    ax.axvline(0.1, c="#d62728", ls="--", lw=1)
    ax.axhline(0.9, c="#ff7f0e", ls="--", lw=1)
    for n, a, s in zip(names, alpha, sp):
        rm = (a < 0.1) or (s > 0.9)
        ax.plot(a, s, "X" if rm else "o", ms=11, c="#d62728" if rm else "#2ca02c")
        ax.annotate(f"{n} ({a}, {s})", (a, s), xytext=(7, 5), textcoords="offset points", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("alpha_i (opacity)")
    ax.set_ylabel("s_p (Pruning_i)")
    ax.set_title("final_prune = [alpha < 0.1] OR [s_p > 0.9]: G3 (score) và G4 (opacity) bị xoá")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "5_10_vung-OR-final-prune.png")


# ---------------------------------------------------------------------------
# 5.11  N(t) tu 15k -> 30k giam bac thang (minh hoa)
# ---------------------------------------------------------------------------
def fig_5_11():
    t = np.arange(15000, 30001)
    N = np.full(t.shape, 420_000.0)
    drops = {18000: 0.12, 21000: 0.08, 24000: 0.05, 27000: 0.03}
    for tt, r in drops.items():
        N[t >= tt] *= (1 - r)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.step(t, N / 1e3, where="post", lw=2)
    for tt, r in drops.items():
        ax.axvline(tt, c="#d62728", ls="--", lw=0.8)
        ax.text(tt + 100, N[t == tt][0] / 1e3 + 5, f"-{r * 100:.0f}%", fontsize=8, color="#d62728")
    ax.axvline(30000, c="gray", ls=":", lw=1)
    ax.text(29200, N[-1] / 1e3 + 20, "30000: không prune", fontsize=7, ha="right")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("N (nghìn Gaussian)")
    ax.set_title("N(t) trong giai đoạn tỉa cuối (MINH HOẠ, tỉ lệ giảm giả định giảm dần)")
    ax.grid(alpha=0.3)
    save(fig, "5_11_N-bac-thang-15k-30k.png")


# ---------------------------------------------------------------------------
# 5.12  Radar so sanh dinh tinh 3DGS vs FastGS-lite
# ---------------------------------------------------------------------------
def fig_5_12():
    cats = ["Tín hiệu densify\n(số điều kiện)", "Tín hiệu prune\n(số điều kiện)", "Ngẫu nhiên\ntrong prune",
            "Tỉa cuối", "Chi phí/lần densify\n(render phụ)", "Kiểm soát N"]
    # thang 0..3 dinh tinh
    gs = np.array([2, 1, 0, 0, 0, 1])
    fg = np.array([3, 3, 3, 3, 3, 3])
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])

    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1, polar=True)
    for v, lab, c in [(gs, "3DGS", "#7f7f7f"), (fg, "FastGS-lite", "#1f77b4")]:
        vv = np.concatenate([v, v[:1]])
        ax.plot(ang_c, vv, "o-", lw=2, c=c, label=lab)
        ax.fill(ang_c, vv, alpha=0.15, color=c)
    ax.set_xticks(ang)
    ax.set_xticklabels(cats, fontsize=7)
    ax.tick_params(axis="x", pad=22)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3"], fontsize=6)
    ax.set_title("Định tính (thang 0-3, tự đặt)", fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.15), fontsize=8)

    ax = fig.add_subplot(1, 2, 2)
    labels = ["N trung bình\n(nghìn)", "R_gauss", "render phụ /\nlần densify"]
    v3 = [933, 1.0, 0]
    vf = [293, 0.314, 20]
    x = np.arange(3)
    for i in range(3):
        a3 = ax.bar(x[i] - 0.2, v3[i] / max(v3[i], vf[i], 1e-9), 0.38, color="#7f7f7f")
        af = ax.bar(x[i] + 0.2, vf[i] / max(v3[i], vf[i], 1e-9), 0.38, color="#1f77b4")
        ax.text(x[i] - 0.2, v3[i] / max(v3[i], vf[i], 1e-9) + 0.02, f"{v3[i]:g}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, vf[i] / max(v3[i], vf[i], 1e-9) + 0.02, f"{vf[i]:g}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("tỉ lệ so với giá trị lớn hơn")
    ax.set_ylim(0, 1.2)
    ax.set_title("Số liệu tham chiếu chương 13 (R_gauss = 0.314 là GIẢ ĐỊNH)", fontsize=9)
    ax.legend([a3, af], ["3DGS", "FastGS-lite"], fontsize=8)
    save(fig, "5_12_radar-so-sanh.png")


# ---------------------------------------------------------------------------
# 5.13  N(t) dien hinh 3DGS vs FastGS-lite tren cung truc (minh hoa)
# ---------------------------------------------------------------------------
def fig_5_13():
    t = np.arange(0, 30001)
    N0 = 100_000.0

    def curve(r_spawn, r_prune, reset_drop, final_drops, N0=N0):
        N = np.full(t.shape, N0)
        cur = N0
        for tt in t[1:]:
            if 500 < tt < 15000 and tt % 100 == 0:
                cur *= (1 + r_spawn) * (1 - r_prune)
            if tt % 3000 == 0 and tt < 15000:
                cur *= (1 - reset_drop)      # xoa alpha<0.005 o lan densify ke tiep (gop vao day)
            if tt in final_drops:
                cur *= (1 - final_drops[tt])
            N[tt] = cur
        return N

    # He so chon de N trung binh ~ 933k va ~ 293k (chuong 13) -- minh hoa
    N3 = curve(0.0227, 0.003, 0.04, {})
    Nf = curve(0.0155, 0.004, 0.04, {18000: 0.12, 21000: 0.08, 24000: 0.05, 27000: 0.03})
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t, N3 / 1e6, lw=2, c="#7f7f7f", label=f"3DGS (minh hoạ) − TB {N3.mean() / 1e6:.2f} M")
    ax.plot(t, Nf / 1e6, lw=2, c="#1f77b4", label=f"FastGS-lite (minh hoạ) − TB {Nf.mean() / 1e6:.2f} M")
    for r in range(3000, 15001, 3000):
        ax.axvline(r, c="#ff7f0e", ls=":", lw=0.7)
    for r in [18000, 21000, 24000, 27000]:
        ax.axvline(r, c="#d62728", ls="--", lw=0.7)
    ax.axvline(15000, c="k", lw=0.8)
    ax.text(15100, ax.get_ylim()[1] * 0.02 + N3.max() / 1e6 * 0.9, "hết densify", fontsize=7)
    ax.set_xlabel("iteration t")
    ax.set_ylabel("N (triệu Gaussian)")
    ax.set_title("N(t) điển hình − MINH HOẠ định tính, hệ số chọn tay để khớp N trung bình chương 13 (933k vs 293k)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "5_13_N-t-3dgs-vs-fastgs.png")
    return N3.mean(), Nf.mean()


# ---------------------------------------------------------------------------
# 5.14  Cat / ghep tensor va trang thai Adam (_prune_optimizer, cat_tensors_to_optimizer)
# ---------------------------------------------------------------------------
def fig_5_14():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    def draw_rows(ax, x, y0, labels, colors, w=1.4, h=0.42, title=None):
        for k, (lab, c) in enumerate(zip(labels, colors)):
            ax.add_patch(Rectangle((x, y0 - k * h), w, h * 0.9, color=c, ec="k", lw=0.5))
            ax.text(x + w / 2, y0 - k * h + h * 0.45, lab, ha="center", va="center", fontsize=7)
        if title:
            ax.text(x + w / 2, y0 + h * 1.35, title, ha="center", fontsize=8, weight="bold")

    # (a) split + prune: 4 goc -> 8 con -> prune
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 5.5)
    ax.axis("off")
    ax.set_title("(a) densify_and_split rồi prune: hàng tensor & Adam (m, v)", fontsize=9)
    g = ["#aec7e8"] * 4
    draw_rows(ax, 0.2, 4.2, ["G1", "G2", "G3", "G4"], g, title="param (N=4)")
    draw_rows(ax, 1.8, 4.2, ["m,v", "m,v", "m,v", "m,v"], g, title="Adam")
    ax.annotate("", xy=(3.5, 3.4), xytext=(3.25, 3.4), arrowprops=dict(arrowstyle="->"))
    ax.text(3.35, 5.2, "cat_tensors_to_optimizer\n(+8 con, m=v=0)", fontsize=7, ha="center")
    kids = ["G1c1", "G2c1", "G3c1", "G4c1", "G1c2", "G2c2", "G3c2", "G4c2"]
    draw_rows(ax, 3.7, 4.2, ["G1", "G2", "G3", "G4"] + kids, g + ["#ffbb78"] * 8, h=0.36, title="N = 12")
    draw_rows(ax, 5.3, 4.2, ["m,v"] * 4 + ["0,0"] * 8, g + ["#ffbb78"] * 8, h=0.36)
    ax.text(7.0, 5.2, "prune_points(mask gốc)\n_prune_optimizer: giữ hàng ~mask", fontsize=7, ha="center")
    ax.annotate("", xy=(7.4, 3.4), xytext=(7.0, 3.4), arrowprops=dict(arrowstyle="->"))
    draw_rows(ax, 7.6, 4.2, kids, ["#ffbb78"] * 8, h=0.36, title="N = 8")
    draw_rows(ax, 9.0, 4.2, ["0,0"] * 8, ["#ffbb78"] * 8, h=0.36, w=0.9)
    ax.text(5, -1.2, "Con mới vào Adam với (m, v) = (0, 0) nhưng 'step' của nhóm là chung → bias-correct ≈1 (xem hình 5.8)", fontsize=7, ha="center")

    # (b) replace_tensor_to_optimizer cho opacity
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 5.5)
    ax.axis("off")
    ax.set_title("(b) replace_tensor_to_optimizer('opacity'): chỉ nhóm opacity, xoá (m, v)", fontsize=9)
    groups = ["xyz", "f_dc", "opacity", "scaling", "rotation", "f_rest (shopt)"]
    cols = ["#c7c7c7"] * 6
    cols[2] = "#ff9896"
    draw_rows(ax, 0.3, 4.2, groups, cols, w=2.0, h=0.5, title="param_groups")
    draw_rows(ax, 2.6, 4.2, ["giữ", "giữ", "m=v=0\nlogit←min(.,0.8|0.01)", "giữ", "giữ", "giữ"], cols, w=2.6, h=0.5, title="trạng thái sau ép opacity")
    ax.text(6.8, 3.4, "Cả hai đường:\n(1) sau mỗi densify: min(alpha, 0.8)\n(2) reset 3000 vòng: min(alpha, 0.01)\n\nChỉ opacity mất moment;\nxyz/scale/rot/SH giữ nguyên m, v.\nstep của Adam KHÔNG đổi.", fontsize=8, va="center")
    save(fig, "5_14_optimizer-cat-ghep.png")


# ---------------------------------------------------------------------------
# 5.15  So do khoi toan bo ADC
# ---------------------------------------------------------------------------
def fig_5_15():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f0fe", fs=7.5, ec="#1f77b4"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, text=None, c="k"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=12, color=c, lw=1))
        if text:
            ax.text((x0 + x1) / 2 + 0.05, (y0 + y1) / 2, text, fontsize=6.5, color=c)

    box(0.3, 8.6, 3.2, 1.1, "Mỗi vòng t < 15000\nmax_radii2D, accum, accum_abs, denom\n(add_densification_stats)", "#f5f5f5", ec="gray")
    box(4.2, 8.6, 4.2, 1.1, "t > 500, t % 100 == 0\ncompute_gaussian_score_fastgs (V=10 view)\n→ Importance_i, Pruning_i", "#fff3e0", ec="#ff7f0e")
    box(9.2, 8.6, 3.5, 1.1, "t % 3000 == 0, 15000 < t < 30000\ncompute_gaussian_score_fastgs\n→ Pruning_i (chỉ cần s_p)", "#fdecea", ec="#d62728")

    box(0.3, 6.6, 3.9, 1.3, "DENSIFY (AND)\nclone: |g| ≥ 2e-4 & max s ≤ dense·extent & Imp > 5\nsplit: |g_abs| ≥ 1.2e-3 & max s > dense·extent & Imp > 5\ncat_tensors_to_optimizer (+clone, +2 con) ; prune gốc split")
    box(4.6, 6.6, 3.9, 1.3, "PRUNE MULTINOMIAL\nC = {a<0.005 | r2D>20 | max s>0.1 extent}\nbudget = floor(0.5|C|), w = 1/(1e-6+1-Pruning)\nS ~ multinomial(w, budget) ; xoá = C ∩ S", "#fff3e0", ec="#ff7f0e")
    box(8.9, 6.6, 3.8, 1.3, "ÉP OPACITY (1)\nlogit ← inverse_sigmoid(min(a, 0.8))\nreplace_tensor_to_optimizer('opacity')\n(m, v) ← 0, step giữ", "#e8f5e9", ec="#2ca02c")
    box(4.6, 4.6, 3.9, 1.1, "RESET (2)  t % 3000 == 0, t < 15000\nlogit ← inverse_sigmoid(min(a, 0.01))\nreplace_tensor_to_optimizer → (m, v) ← 0", "#e8f5e9", ec="#2ca02c")
    box(8.9, 4.6, 3.8, 1.1, "FINAL PRUNE (thẳng, không lấy mẫu)\nxoá = [a < 0.1] OR [Pruning > 0.9]\nprune_points → _prune_optimizer", "#fdecea", ec="#d62728")
    box(0.3, 4.6, 3.9, 1.1, "3D GAUSSIANS (chương 2)\n_xyz, _f_dc, _f_rest, _opacity, _scaling, _rotation\nN ← N + |clone| + |split| − |xoá|", "#ede7f6", ec="#673ab7", fs=8)
    box(0.3, 2.4, 12.4, 1.2,
        "Đầu ra của khối mỗi lần gọi: 6 tensor tham số + exp_avg/exp_avg_sq của 6 nhóm (2 optimizer) đổi số hàng;\n"
        "xyz_gradient_accum / _abs / denom / max_radii2D bị RESET về 0 sau densification_postfix (bắt đầu chu kỳ tích luỹ mới);\n"
        "tmp_radii = None. Vòng lặp tiếp theo render với N mới, Adam step chung (step không reset).", "#f5f5f5", fs=7.5, ec="gray")

    arrow(2.0, 8.6, 2.0, 7.9)
    arrow(6.3, 8.6, 6.3, 7.9, "Importance, Pruning", "#ff7f0e")
    arrow(4.2, 7.25, 4.6, 7.25, "N'", "k")
    arrow(8.5, 7.25, 8.9, 7.25)
    # score cuoi -> final prune: di vong theo le phai de khong cat hop EP OPACITY
    ax.plot([12.7, 12.9, 12.9], [9.15, 9.15, 5.15], c="#d62728", lw=1)
    arrow(12.9, 5.15, 12.7, 5.15, "", "#d62728")
    ax.text(12.93, 7.0, "Pruning", fontsize=6.5, color="#d62728", rotation=90, va="center")
    arrow(10.8, 6.6, 6.6, 5.7, "cùng vòng, sau densify", "#2ca02c")
    arrow(8.9, 5.15, 8.5, 5.15, "", "#d62728")
    arrow(4.6, 5.15, 4.2, 5.15, "", "#2ca02c")
    arrow(2.2, 4.6, 2.2, 3.6)
    arrow(2.2, 6.6, 2.2, 5.7)
    ax.text(0.4, 1.9, "Thứ tự thật trong densify_and_prune_fastgs: clone → split → prune multinomial (trên quần thể mới, trọng số theo chỉ số cũ) → ép opacity 0.8.", fontsize=7.5)
    ax.text(0.4, 1.4, "reset_opacity chạy ngay sau (cùng vòng) khi t % 3000 == 0; final_prune ở nhánh riêng, không densify.", fontsize=7.5)
    ax.text(0.4, 0.7, "Ký hiệu: a = alpha = sigmoid(_opacity); r2D = max_radii2D; extent = scene.cameras_extent; dense = 0.001.", fontsize=7.5, color="gray")
    ax.set_title("Sơ đồ khối toàn bộ Adaptive Density Control (FastGS-lite)", fontsize=11)
    save(fig, "5_15_so-do-khoi-ADC.png")


# ---------------------------------------------------------------------------
# 5.16  Thu tu that: split truoc, prune sau, trong so theo chi so cu (canh 4 Gaussian)
# ---------------------------------------------------------------------------
def fig_5_16():
    w_old = np.array([4.2621, 1.8204, 1e6, 1.0])
    kids = ["G1c1", "G2c1", "G3c1", "G4c1", "G1c2", "G2c2", "G3c2", "G4c2"]
    w_pad = np.concatenate([w_old, np.zeros(4)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.bar(range(8), np.maximum(w_pad, 1e-2), color=["#d62728"] * 4 + ["#7f7f7f"] * 4)
    ax.set_yscale("log")
    ax.set_xticks(range(8))
    ax.set_xticklabels(kids, fontsize=8)
    ax.set_ylabel("padded_importance (log; 0 vẽ thành 1e-2)")
    ax.set_title("Sau split 4 gốc → 8 con: trọng số cũ gán theo CHỈ SỐ 0..3")
    for i, v in enumerate(w_pad):
        ax.text(i, max(v, 1e-2) * 1.5, f"{v:g}", ha="center", fontsize=7)
    ax = axes[1]
    ax.axis("off")
    txt = ("Cảnh 4 Gaussian (kiểm định số, t > 3000):\n\n"
           "1. split cả 4  →  N = 8 con, max_radii2D con = 0, max s = s/1.6 > 0.169\n"
           "2. prune_mask = cả 8 (vì max s > 0.1 extent)  →  |C| = 8, budget = 4\n"
           "3. padded_importance[:4] = w cũ  →  chỉ 4 phần tử dương\n"
           "4. multinomial(w, 4) bắt buộc rút đúng {0,1,2,3}\n"
           "5. xoá = C ∩ S = {G1c1, G2c1, G3c1, G4c1}  →  N = 4\n\n"
           "Công thức 7.8 (thứ tự lý thuyết): 4 + 0 + 4 − 2 = 6\n"
           "Thứ tự thật của code: 8 − 4 = 4\n\n"
           "Trong huấn luyện thật tỉ lệ split mỗi lần ~ vài %, nên lệch nhỏ.")
    ax.text(0.0, 0.95, txt, va="top", fontsize=8.5, linespacing=1.35)
    save(fig, "5_16_thu-tu-that-code.png")


if __name__ == "__main__":
    fig_5_1()
    removed, mean_rm, p_th = fig_5_2()
    print("5.2 tan suat bi xoa:", np.round(removed, 3), "xoa TB:", round(mean_rm, 3))
    print("5.2 P(i in S) ly thuyet:", np.round(p_th, 4))
    print("5.3 |C|, budget, hard, multi:", fig_5_3())
    fig_5_4()
    fig_5_5()
    fig_5_6()
    fig_5_7()
    fig_5_8()
    fig_5_9()
    fig_5_10()
    fig_5_11()
    fig_5_12()
    print("5.13 N TB:", fig_5_13())
    fig_5_14()
    fig_5_15()
    fig_5_16()
