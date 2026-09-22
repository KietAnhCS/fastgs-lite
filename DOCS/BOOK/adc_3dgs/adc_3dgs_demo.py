"""
adc_3dgs_demo.py — Mô phỏng TOY cơ chế Adaptive Density Control (ADC) của
3D Gaussian Splatting gốc (Kerbl et al. 2023, cấu hình Inria mặc định).

ĐÂY LÀ TOY MODEL: không render ảnh, không có hàm mất mát thật. Gradient
vị trí trên màn hình dL/dmu' được GIẢ LẬP: cao ở các Gaussian nằm gần viền một
đường tròn (vùng "chi tiết cao"), thấp ở nơi khác. Mục đích duy nhất là minh
hoạ đúng LỊCH và CÔNG THỨC 1–6 của ADC (xem _SPEC.md):

  (1) Lịch: tích luỹ gradient mỗi vòng t < 15000; densify + prune mỗi 100 vòng
      trong khoảng 500 < t < 15000 (144 lần); reset opacity tại
      t = 3000, 6000, 9000, 12000; sau 15000 số Gaussian N đóng băng.
  (2) g_bar_i = accum_i / denom_i, với
      accum_i = sum_t 1_i(t) * || sum_x dL/dmu'_i |_x ||,  denom_i = sum_t 1_i(t).
  (3) clone_i = [||g_bar_i|| >= tau_grad] AND [max(s_i) <= delta * extent]
      split_i = [||g_bar_i|| >= tau_grad] AND [max(s_i) >  delta * extent]
      tau_grad = 2e-4, delta = percent_dense = 0.01.
  (4) Clone: copy nguyên theta, gốc giữ  -> N + 1.
      Split: mu^(j) = mu_i + R(q_i) e^(j), e^(j) ~ N(0, diag(s_i^2)),
             scale = s_i / 1.6, gốc xoá, 2 con        -> N + 1.
  (5) Prune cứng: xoá_i = [alpha_i < 0.005] OR [r_i^2D > 20 px, chỉ khi t > 3000]
                          OR [max(s_i) > 0.1 * extent].
  (6) Reset opacity: alpha_i <- sigmoid^-1(min(alpha_i, 0.01)) tại t = 3000k.
      Ở đây alpha được lưu ở dạng XÁC SUẤT (0..1) nên reset là alpha <- min(alpha, 0.01);
      trong code thật alpha lưu dạng logit nên mới có sigmoid^-1.

Không gian là 2D để dễ vẽ; mở rộng 3D chỉ cần mu (N,3), scale (N,3), rot là
quaternion (N,4) thay cho góc xoay (N,), và r^2D tính qua phép chiếu thật.

Chạy:  python DOCS/BOOK/adc_3dgs/adc_3dgs_demo.py
Kết quả: DOCS/BOOK/adc_3dgs/adc_3dgs_demo.png + thống kê in ra màn hình.
Chỉ dùng numpy + matplotlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "adc_3dgs_demo.png")

# Vùng "chi tiết cao" giả lập: viền đường tròn tâm (0.5, 0.5), bán kính 0.3
CIRCLE_C = np.array([0.5, 0.5])
CIRCLE_R = 0.3


# ----------------------------------------------------------------------------
# Cấu hình ADC — các hằng số Inria mặc định, KHÔNG đổi số
# ----------------------------------------------------------------------------
@dataclass
class ADCConfig:
    tau_grad: float = 2e-4                 # ngưỡng ||g_bar||  (công thức 3)
    percent_dense: float = 0.01            # delta               (công thức 3)
    opacity_reset_iters: tuple = (3000, 6000, 9000, 12000)  # (công thức 1, 6)
    densify_every: int = 100               # (công thức 1)
    densify_from_iter: int = 500           # (công thức 1)
    densify_until_iter: int = 15000        # (công thức 1)
    alpha_min: float = 0.005               # (công thức 5)
    radius_2d_max_px: float = 20.0         # (công thức 5)
    scale_max_ratio: float = 0.1           # (công thức 5)
    scale_shrink_on_split: float = 1.6     # (công thức 4)
    reset_opacity_value: float = 0.01      # (công thức 6)
    total_iters: int = 20000               # chạy quá 15000 để thấy N đứng yên
    extent: float = 1.0                    # kích thước cảnh (cạnh hình vuông [0,1]^2)

    # --- tham số CHỈ của phần giả lập (không thuộc ADC) ---
    focal_px: float = 800.0                # r^2D ~ max(s) * focal / depth, depth = 1
    visibility_p: float = 0.7              # xác suất 1_i(t) = 1
    alpha_grow_per_iter: float = 0.0006    # opacity tự tăng giữa 2 lần reset (trung bình)
    alpha_decay_per_iter: float = 0.004    # Gaussian "chết" bị giảm alpha
    dying_fraction: float = 0.04           # tỉ lệ Gaussian mới bị gán "chết"
    grad_rim_level: float = 4.0            # gradient ở viền ≈ 4 * tau_grad
    grad_bg_level: float = 0.3             # gradient nơi khác ≈ 0.3 * tau_grad
    grad_band: float = 0.03                # bề rộng băng viền gradient cao
    density_sat: float = 1.0               # bão hoà gradient khi đông hàng xóm
    density_cell: float = 0.02             # ô lưới đếm mật độ cục bộ
    pos_lr: float = 0.002                  # "bước tối ưu" kéo Gaussian về viền
    pos_noise: float = 3e-4                # nhiễu vị trí mỗi vòng
    snapshot_iters: tuple = (500, 3000, 9000, 15000)


# ----------------------------------------------------------------------------
# Đám mây Gaussian 2D
# ----------------------------------------------------------------------------
@dataclass
class GaussianCloud:
    """Tập Gaussian 2D. Mọi mảng có cùng chiều dài N.

    mu    (N,2)  tâm
    scale (N,2)  bán trục s_i (độ lệch chuẩn theo 2 trục riêng)
    rot   (N,)   góc xoay theta_i — đóng vai trò R(q_i) trong 2D
    alpha (N,)   opacity dạng xác suất (code thật lưu logit)
    accum (N,)   sum_t 1_i(t) * ||g_i(t)||          (công thức 2)
    denom (N,)   sum_t 1_i(t)                        (công thức 2)
    rate  (N,)   [chỉ giả lập] tốc độ đổi alpha mỗi vòng (+: sống, -: chết)
    """

    mu: np.ndarray
    scale: np.ndarray
    rot: np.ndarray
    alpha: np.ndarray
    accum: np.ndarray = field(default=None)
    denom: np.ndarray = field(default=None)
    rate: np.ndarray = field(default=None)

    def __post_init__(self):
        n = len(self.mu)
        if self.accum is None:
            self.accum = np.zeros(n)
        if self.denom is None:
            self.denom = np.zeros(n)
        if self.rate is None:
            self.rate = np.zeros(n)
        self.check()

    _FIELDS = ("mu", "scale", "rot", "alpha", "accum", "denom", "rate")

    @property
    def n(self) -> int:
        return len(self.mu)

    def check(self):
        n = self.n
        for f in self._FIELDS:
            arr = getattr(self, f)
            assert len(arr) == n, f"mảng {f} có {len(arr)} phần tử, mong {n}"
        assert self.mu.shape == (n, 2) and self.scale.shape == (n, 2)

    def reset_grad_stats(self):
        """Đặt lại accum, denom = 0 sau mỗi lần densify (như code gốc)."""
        self.accum = np.zeros(self.n)
        self.denom = np.zeros(self.n)
        self.check()

    def grow(self, other: "GaussianCloud"):
        """Nối thêm các Gaussian mới (clone / con của split)."""
        other.check()
        for f in self._FIELDS:
            setattr(self, f, np.concatenate([getattr(self, f), getattr(other, f)], axis=0))
        self.check()

    def keep(self, mask: np.ndarray):
        """Giữ lại các Gaussian có mask = True (xoá gốc bị split, prune)."""
        assert mask.shape == (self.n,), f"mask {mask.shape} không khớp N={self.n}"
        for f in self._FIELDS:
            setattr(self, f, getattr(self, f)[mask])
        self.check()

    def copy(self) -> "GaussianCloud":
        return GaussianCloud(**{f: getattr(self, f).copy() for f in self._FIELDS})

    def max_scale(self) -> np.ndarray:
        return self.scale.max(axis=1)


def rot2d(theta: np.ndarray) -> np.ndarray:
    """Ma trận xoay 2D (N,2,2) — R(q_i) trong 2D."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


# ----------------------------------------------------------------------------
# Giả lập gradient và visibility
# ----------------------------------------------------------------------------
def local_density(mu: np.ndarray, cell: float) -> np.ndarray:
    """Số Gaussian trong cùng ô lưới (cell x cell) — đo "đã đủ dày chưa"."""
    ncell = int(np.ceil(1.0 / cell))
    ij = np.clip((mu / cell).astype(int), 0, ncell - 1)
    flat = ij[:, 0] * ncell + ij[:, 1]
    counts = np.bincount(flat, minlength=ncell * ncell)
    return counts[flat].astype(float)


def fake_gradient_norms(mu: np.ndarray, rng: np.random.Generator, cfg: ADCConfig):
    """Giả lập || sum_x dL/dmu'_i |_x || cho từng Gaussian, và visibility 1_i(t).

    - Gaussian nằm trong băng |dist - 0.3| < grad_band quanh viền tròn nhận
      gradient ≈ grad_rim_level * tau_grad (vùng chi tiết cao, ảnh chưa khớp).
    - Nơi khác gradient ≈ grad_bg_level * tau_grad (< tau_grad).
    - Cộng nhiễu nhân log-normal.
    - Gradient bị chia bớt khi ô lưới đã đông (mô phỏng: nhiều Gaussian cùng
      phủ một vùng thì mỗi cái nhận ít lỗi hơn) — nếu không có, N sẽ bùng nổ mũ.
    - 1_i(t) ~ Bernoulli(visibility_p): Gaussian có nằm trong khung nhìn không.
    """
    dist = np.linalg.norm(mu - CIRCLE_C, axis=1)
    in_band = np.abs(dist - CIRCLE_R) < cfg.grad_band
    base = np.where(in_band, cfg.grad_rim_level, cfg.grad_bg_level) * cfg.tau_grad
    noise = np.exp(rng.normal(0.0, 0.35, size=len(mu)))
    dens = local_density(mu, cfg.density_cell)
    g = base * noise / (1.0 + dens / cfg.density_sat)
    visible = rng.random(len(mu)) < cfg.visibility_p
    return g, visible


def accumulate(cloud: GaussianCloud, g: np.ndarray, visible: np.ndarray):
    # công thức (2): accum += 1_i(t) * ||g_i||, denom += 1_i(t)
    cloud.accum += visible * g
    cloud.denom += visible.astype(float)


def compute_g_bar(cloud: GaussianCloud) -> np.ndarray:
    # công thức (2): g_bar_i = accum_i / denom_i (denom = 0 -> g_bar = 0)
    return np.where(cloud.denom > 0, cloud.accum / np.maximum(cloud.denom, 1.0), 0.0)


# ----------------------------------------------------------------------------
# Bước tối ưu giả lập (không thuộc ADC)
# ----------------------------------------------------------------------------
def fake_optimizer_step(cloud: GaussianCloud, rng: np.random.Generator, cfg: ADCConfig):
    """Mô phỏng ảnh hưởng của gradient descent lên theta giữa 2 lần densify:
    - Gaussian gần viền bị kéo về đúng viền (clone tách khỏi gốc nhờ bước này,
      như trong code thật hai bản sao nhận gradient khác nhau và trôi ra).
    - Nhiễu vị trí nhỏ.
    - Opacity tự tăng (rate > 0) hoặc giảm dần với Gaussian "chết" (rate < 0).
    """
    d = cloud.mu - CIRCLE_C
    dist = np.linalg.norm(d, axis=1) + 1e-9
    near = np.abs(dist - CIRCLE_R) < 2 * cfg.grad_band
    radial = d / dist[:, None]
    step = -cfg.pos_lr * (dist - CIRCLE_R)[:, None] * radial * near[:, None]
    cloud.mu = cloud.mu + step + rng.normal(0.0, cfg.pos_noise, size=cloud.mu.shape)
    cloud.mu = np.clip(cloud.mu, 0.0, 1.0)
    # công thức (6) ghi chú: giữa 2 lần reset, alpha tự do lên tới 0.99
    cloud.alpha = np.clip(cloud.alpha + cloud.rate, 0.0, 0.99)


def new_rates(n: int, rng: np.random.Generator, cfg: ADCConfig) -> np.ndarray:
    dying = rng.random(n) < cfg.dying_fraction
    grow = cfg.alpha_grow_per_iter * rng.uniform(0.3, 1.7, size=n)   # mỗi Gaussian hồi alpha nhanh/chậm khác nhau
    return np.where(dying, -cfg.alpha_decay_per_iter, grow)


# ----------------------------------------------------------------------------
# ADC: densify (clone / split) + prune
# ----------------------------------------------------------------------------
def densify_and_prune(cloud: GaussianCloud, t: int, rng: np.random.Generator,
                      cfg: ADCConfig) -> dict:
    """Một lần densify + prune. Trả về thống kê {clone, split, prune}.

    Thứ tự bắt buộc để tránh lệch kích thước mảng:
      1. tính g_bar, mask clone/split trên N cũ
      2. tạo Gaussian mới (bản clone, 2 con của split) từ N cũ
      3. xoá gốc bị split, rồi nối các Gaussian mới
      4. tính mask prune trên cloud MỚI (N mới) và lọc
      5. reset accum/denom cho toàn bộ
    """
    n_old = cloud.n
    g_bar = compute_g_bar(cloud)
    max_s = cloud.max_scale()
    thr_scale = cfg.percent_dense * cfg.extent

    # công thức (3): điều kiện clone/split
    high = g_bar >= cfg.tau_grad
    clone_mask = high & (max_s <= thr_scale)
    split_mask = high & (max_s > thr_scale)
    assert clone_mask.shape == (n_old,) and split_mask.shape == (n_old,)

    # công thức (4) — Clone: copy nguyên theta, gốc giữ nguyên
    clones = GaussianCloud(
        mu=cloud.mu[clone_mask].copy(),
        scale=cloud.scale[clone_mask].copy(),
        rot=cloud.rot[clone_mask].copy(),
        alpha=cloud.alpha[clone_mask].copy(),
        accum=np.zeros(clone_mask.sum()),
        denom=np.zeros(clone_mask.sum()),
        rate=new_rates(int(clone_mask.sum()), rng, cfg),
    )

    # công thức (4) — Split: 2 con, mu^(j) = mu_i + R(q_i) e^(j), e^(j) ~ N(0, diag(s_i^2)),
    # scale = s_i / 1.6, gốc bị xoá
    idx = np.nonzero(split_mask)[0]
    k = 2
    par_mu = np.repeat(cloud.mu[idx], k, axis=0)
    par_s = np.repeat(cloud.scale[idx], k, axis=0)
    par_rot = np.repeat(cloud.rot[idx], k)
    par_alpha = np.repeat(cloud.alpha[idx], k)
    e = rng.normal(0.0, 1.0, size=par_mu.shape) * par_s           # e ~ N(0, diag(s^2))
    R = rot2d(par_rot)                                             # R(q_i)
    child_mu = par_mu + np.einsum("nij,nj->ni", R, e)
    child_mu = np.clip(child_mu, 0.0, 1.0)
    children = GaussianCloud(
        mu=child_mu,
        scale=par_s / cfg.scale_shrink_on_split,
        rot=par_rot,
        alpha=par_alpha,
        accum=np.zeros(len(par_mu)),
        denom=np.zeros(len(par_mu)),
        rate=new_rates(len(par_mu), rng, cfg),
    )

    # xoá gốc bị split (mask trên N cũ), rồi nối clone + con
    cloud.keep(~split_mask)
    cloud.grow(clones)
    cloud.grow(children)
    n_after_grow = cloud.n
    assert n_after_grow == n_old + clone_mask.sum() + split_mask.sum()  # mỗi clone/split -> N+1

    # công thức (5): prune cứng — mask tính trên cloud SAU KHI grow
    max_s_new = cloud.max_scale()
    r2d_px = max_s_new * cfg.focal_px                 # r^2D ~ max(s) * focal / depth (depth=1)
    prune = cloud.alpha < cfg.alpha_min
    if t > cfg.opacity_reset_iters[0]:                # chỉ khi t > 3000
        prune |= r2d_px > cfg.radius_2d_max_px
    prune |= max_s_new > cfg.scale_max_ratio * cfg.extent
    assert prune.shape == (n_after_grow,)
    cloud.keep(~prune)

    cloud.reset_grad_stats()
    return {"clone": int(clone_mask.sum()), "split": int(split_mask.sum()),
            "prune": int(prune.sum())}


def maybe_reset_opacity(cloud: GaussianCloud, t: int, cfg: ADCConfig) -> bool:
    """công thức (6): tại t = 3000k, alpha_i <- sigmoid^-1(min(alpha_i, 0.01)).

    alpha ở đây lưu dạng xác suất nên chỉ cần min(alpha, 0.01); trong code thật
    tham số là logit nên lưu sigmoid^-1(.) rồi sigmoid lại khi render.
    """
    if t in cfg.opacity_reset_iters:
        cloud.alpha = np.minimum(cloud.alpha, cfg.reset_opacity_value)
        return True
    return False


# ----------------------------------------------------------------------------
# Vòng mô phỏng
# ----------------------------------------------------------------------------
def rim_fraction(mu: np.ndarray, band: float = 0.05) -> float:
    dist = np.linalg.norm(mu - CIRCLE_C, axis=1)
    return float(np.mean(np.abs(dist - CIRCLE_R) < band))


def init_cloud(n0: int, rng: np.random.Generator, cfg: ADCConfig) -> GaussianCloud:
    mu = rng.uniform(0.0, 1.0, size=(n0, 2))
    scale = rng.uniform(0.005, 0.03, size=(n0, 2))
    rot = rng.uniform(0.0, 2 * np.pi, size=n0)
    alpha = np.full(n0, 0.1)                       # 3DGS khởi tạo opacity = 0.1
    return GaussianCloud(mu=mu, scale=scale, rot=rot, alpha=alpha,
                         rate=new_rates(n0, rng, cfg))


def run_simulation(cfg: ADCConfig, seed: int = 0, n0: int = 300) -> dict:
    rng = np.random.default_rng(seed)
    cloud = init_cloud(n0, rng, cfg)
    n_hist = np.zeros(cfg.total_iters + 1, dtype=int)
    n_hist[0] = cloud.n
    snapshots = {}
    totals = {"clone": 0, "split": 0, "prune": 0, "densify_calls": 0}
    log = []  # (t, clone, split, prune)
    rim0 = rim_fraction(cloud.mu)

    for t in range(1, cfg.total_iters + 1):
        # công thức (1): tích luỹ gradient chỉ khi t < densify_until_iter
        if t < cfg.densify_until_iter:
            g, vis = fake_gradient_norms(cloud.mu, rng, cfg)
            accumulate(cloud, g, vis)

        fake_optimizer_step(cloud, rng, cfg)

        # công thức (1): densify + prune mỗi 100 vòng, 500 < t < 15000
        if (t % cfg.densify_every == 0 and cfg.densify_from_iter < t < cfg.densify_until_iter):
            st = densify_and_prune(cloud, t, rng, cfg)
            for k_ in ("clone", "split", "prune"):
                totals[k_] += st[k_]
            totals["densify_calls"] += 1
            log.append((t, st["clone"], st["split"], st["prune"]))

        n_hist[t] = cloud.n
        if t in cfg.snapshot_iters:           # chụp TRƯỚC reset opacity để thấy alpha
            snapshots[t] = cloud.copy()

        # công thức (6): reset opacity (sau densify cùng vòng, như train.py gốc)
        maybe_reset_opacity(cloud, t, cfg)

    return {"cloud": cloud, "n_hist": n_hist, "snapshots": snapshots,
            "totals": totals, "log": log, "n0": n0, "rim0": rim0,
            "rim1": rim_fraction(cloud.mu)}


# ----------------------------------------------------------------------------
# Vẽ hình
# ----------------------------------------------------------------------------
def make_figure(res: dict, cfg: ADCConfig, path: str):
    n_hist = res["n_hist"]
    snaps = res["snapshots"]
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])

    # --- N(t) ---
    ax = fig.add_subplot(gs[0, :])
    ax.plot(np.arange(len(n_hist)), n_hist, color="#1f4e79", lw=1.6, label="N(t)")
    for i, r in enumerate(cfg.opacity_reset_iters):
        ax.axvline(r, color="#c0392b", ls="--", lw=1,
                   label="reset opacity (3000k)" if i == 0 else None)
    ax.axvline(cfg.densify_from_iter, color="#7f8c8d", ls=":", lw=1,
               label="densify_from_iter = 500")
    ax.axvline(cfg.densify_until_iter, color="#2e8b57", ls="-.", lw=1.3,
               label="densify_until_iter = 15000 (N đóng băng)")
    ax.set_xlabel("vòng lặp t")
    ax.set_ylabel("số Gaussian N")
    ax.set_title(f"N(t): {res['n0']} → {n_hist[-1]} Gaussian; "
                 f"{res['totals']['densify_calls']} lần densify, "
                 f"clone {res['totals']['clone']}, split {res['totals']['split']}, "
                 f"prune {res['totals']['prune']}")
    ax.set_xlim(0, cfg.total_iters)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # --- 3 scatter ---
    plot_iters = [it for it in (500, 3000, 9000) if it in snaps]
    all_alpha = np.concatenate([snaps[it].alpha for it in plot_iters])
    vmin, vmax = 0.0, float(all_alpha.max())
    theta = np.linspace(0, 2 * np.pi, 200)
    sc = None
    for j, it in enumerate(plot_iters):
        axs = fig.add_subplot(gs[1, j])
        c = snaps[it]
        size = (c.max_scale() / 0.03) ** 2 * 60 + 2      # diện tích ∝ scale^2
        sc = axs.scatter(c.mu[:, 0], c.mu[:, 1], s=size, c=c.alpha, cmap="viridis",
                         vmin=vmin, vmax=vmax, alpha=0.75, linewidths=0)
        axs.plot(CIRCLE_C[0] + CIRCLE_R * np.cos(theta), CIRCLE_C[1] + CIRCLE_R * np.sin(theta),
                 "r--", lw=1, label="vùng chi tiết cao (r = 0.3)")
        axs.set_xlim(0, 1)
        axs.set_ylim(0, 1)
        axs.set_aspect("equal")
        axs.set_title(f"t = {it}: N = {c.n}, ở viền {100 * rim_fraction(c.mu):.0f}%")
        axs.set_xticks([0, 0.5, 1])
        axs.set_yticks([0, 0.5, 1])
        if j == 0:
            axs.legend(loc="lower left", fontsize=7)
    cb = fig.colorbar(sc, ax=fig.axes[1:], shrink=0.85, pad=0.02)
    cb.set_label("opacity α (kích thước điểm ∝ max(s))")

    fig.suptitle("Toy ADC của 3DGS gốc (Kerbl 2023): densify mỗi 100 vòng trong 500<t<15000, "
                 "reset opacity tại 3000k, prune α<0.005 / r²ᴰ>20px / max(s)>0.1·extent",
                 fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    cfg = ADCConfig()
    res = run_simulation(cfg, seed=0, n0=300)
    make_figure(res, cfg, OUT_PNG)

    tot = res["totals"]
    n_hist = res["n_hist"]
    print("=== Toy ADC 3DGS ===")
    print(f"N ban đầu           : {res['n0']}")
    print(f"N cuối (t={cfg.total_iters}) : {n_hist[-1]}")
    print(f"N tại t=15000        : {n_hist[cfg.densify_until_iter]}  "
          f"(đứng yên sau đó: {np.all(n_hist[cfg.densify_until_iter:] == n_hist[-1])})")
    print(f"số lần densify       : {tot['densify_calls']}")
    print(f"tổng clone / split / prune : {tot['clone']} / {tot['split']} / {tot['prune']}")
    print(f"tỉ lệ Gaussian ở băng viền |dist-0.3|<0.05 : đầu {100 * res['rim0']:.1f}%  "
          f"→ cuối {100 * res['rim1']:.1f}%")
    for it, c in sorted(res["snapshots"].items()):
        print(f"  snapshot t={it:5d}: N={c.n:5d}, ở viền {100 * rim_fraction(c.mu):.1f}%, "
              f"alpha trung bình {c.alpha.mean():.3f}")
    print("Đã lưu:")
    print(f"  {OUT_PNG}")


if __name__ == "__main__":
    main()
