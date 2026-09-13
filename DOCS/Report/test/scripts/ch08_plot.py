"""
Hình minh hoạ cho Test số — Chương 8: mô hình chi phí và ba đòn bẩy
(DOCS/Report/test/08-test.md)

Chỉ dùng numpy + matplotlib (không torch). Ba tỉ số được tính lại tại đây
(R_adam đếm vòng lặp, R_tile từ quần thể demo / ví dụ 3.8 / cảnh đồ chơi,
R_gauss từ demos.mean_gaussians) — cùng công thức với scripts/ch08_test.py
(file đó in ra stdout khi import nên không import trực tiếp). Hàm hình học
được import lại từ demos/fastgs_cost_model.py, không sửa.

Chạy:  python DOCS/Report/test/scripts/ch08_plot.py
PNG:   DOCS/Report/test/figures/ch08_*.png
"""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.grid': True, 'grid.alpha': 0.3})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "demos"))
import fastgs_cost_model as fcm  # noqa: E402

OUT = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT, exist_ok=True)
TILE = 16

C_PRE, C_RAS, C_ADAM, C_FIX = '#4e79a7', '#e15759', '#59a14f', '#9c9c9c'   # aN, bNK, cN·1, F
C_COUNT, C_MEAS, C_ASSUME = '#2ca02c', '#1f77b4', '#ff7f0e'                # đếm / đo / giả định


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("saved", os.path.relpath(path, ROOT))


# ===========================================================================
# Tính lại ba tỉ số (chép công thức của ch08_test.py)
# ===========================================================================
def adam_count(total_iters=30000, fastgs=True):
    """Trả về (main, sh, mảng tích luỹ main, mảng tích luỹ sh) duyệt t=1..total-1."""
    main = sh = 0
    cm, cs = np.zeros(total_iters + 1), np.zeros(total_iters + 1)
    for t in range(1, total_iters):
        if not fastgs:
            main += 1
            sh += 1
        elif t <= 15000:
            main += 1
            if t % 16 == 0:
                sh += 1
        elif t <= 20000:
            if t % 32 == 0:
                main += 1
                sh += 1
        else:
            if t % 64 == 0:
                main += 1
                sh += 1
        cm[t], cs[t] = main, sh
    cm[total_iters], cs[total_iters] = main, sh
    return main, sh, cm, cs


m3, s3, cm3, cs3 = adam_count(fastgs=False)
mf, sf, cmf, csf = adam_count(fastgs=True)
R_ADAM = (mf + sf) / (m3 + s3)
assert (mf, sf) == fcm.adam_steps(fastgs=True)

# R_tile (i): ví dụ 3.8
S = np.array([[117.0, 54.0], [54.0, 36.0]])
mu = np.array([120.0, 88.0])
mid = (S[0, 0] + S[1, 1]) / 2
rad = math.sqrt(((S[0, 0] - S[1, 1]) / 2) ** 2 + S[0, 1] ** 2)
lmax, lmin = mid + rad, mid - rad
r3 = math.ceil(3 * math.sqrt(lmax))
K3_ex = int(np.prod(np.floor((mu + r3 + 15) / TILE) - np.floor((mu - r3) / TILE)))
theta_ex = 0.5 * math.atan2(2 * S[0, 1], S[0, 0] - S[1, 1])
Kf_ex = fcm.tiles_fastgs(lmax, lmin, theta_ex, 1.0, 0.5, mu[0] % TILE, mu[1] % TILE)
R_TILE_EX = Kf_ex / K3_ex

# R_tile (ii): quần thể demo, seed 0, tính từng splat (cùng thứ tự RNG với tile_stats)
fcm.RNG = np.random.default_rng(0)
pop = fcm.sample_population()
lam1, lam2, theta, opac = pop
K3_arr = np.array([fcm.tiles_3dgs(a, b) for a, b in zip(lam1, lam2)])
Kbox_arr = np.array([fcm.tiles_compact_box_only(a, b, t, o, 0.5) for a, b, t, o in zip(lam1, lam2, theta, opac)])
Kf_arr = np.array([fcm.tiles_fastgs(a, b, t, o, 0.5) for a, b, t, o in zip(lam1, lam2, theta, opac)])
K3_pop, Kbox_pop, Kf_pop = K3_arr.mean(), Kbox_arr.mean(), Kf_arr.mean()
R_TILE_POP = Kf_pop / K3_pop
# K_fastgs theo mult (thứ tự 1.0 trước để khớp 9.99 của ch08_test, rồi 0.25, 0.75)
K_MULT = {0.5: Kf_pop}
for m in (1.0, 0.25, 0.75):
    K_MULT[m] = fcm.tile_stats(pop, m)[2]
OPAC_ROWS = fcm.sweep_opacity()      # (o, t, sqrt t, K_fastgs), sigma'=8 px

# R_tile (iii): cảnh đồ chơi camera 1, alpha=0.1, mult=0.5
pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.5], [-0.4, -0.2, 1.0], [0.3, -0.5, 0.2]])
W, H, fx, fy = 48, 32, 40.0, 40.0
grid = (math.ceil(W / TILE), math.ceil(H / TILE))
t_toy = 0.5 * 2 * math.log(255 * 0.1)
D2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
P3c = Pfc = P3n = Pfn = 0
for i in range(4):
    s = math.sqrt(np.sort(D2[i])[1:4].mean())
    tx, ty, tz = pts[i, 0], pts[i, 1], pts[i, 2] + 4.0
    mux, muy = fx * tx / tz + (W - 1) / 2, fy * ty / tz + (H - 1) / 2
    S11 = (fx / tz) ** 2 * s * s + 0.3
    r = math.ceil(3 * math.sqrt(S11))
    half = math.sqrt(t_toy * S11)
    cl = lambda v, g: max(0, min(g, int(v)))
    P3c += (cl((mux + r + 15) / TILE, grid[0]) - cl((mux - r) / TILE, grid[0])) * \
           (cl((muy + r + 15) / TILE, grid[1]) - cl((muy - r) / TILE, grid[1]))
    Pfc += (cl((mux + half) / TILE + 1, grid[0]) - cl((mux - half) / TILE, grid[0])) * \
           (cl((muy + half) / TILE + 1, grid[1]) - cl((muy - half) / TILE, grid[1]))
    P3n += (math.floor((mux + r + 15) / TILE) - math.floor((mux - r) / TILE)) * \
           (math.floor((muy + r + 15) / TILE) - math.floor((muy - r) / TILE))
    Pfn += (math.floor((mux + half) / TILE + 1) - math.floor((mux - half) / TILE)) * \
           (math.floor((muy + half) / TILE + 1) - math.floor((muy - half) / TILE))
R_TILE_TOY, R_TILE_TOY_NC = Pfc / P3c, Pfn / P3n

# R_gauss (giả định demo)
n3_mean, _ = fcm.mean_gaussians(spawn=0.10, prune=0.005)
nf_mean, _ = fcm.mean_gaussians(spawn=0.06, prune=0.010, final_prune_frac=0.05)
R_GAUSS = nf_mean / n3_mean
OV = fcm.scoring_overhead()          # 0.02

print(f"R_adam={R_ADAM:.4f} R_tile: ex={R_TILE_EX:.4f} pop={R_TILE_POP:.4f} toy={R_TILE_TOY:.4f}"
      f" (no clamp {R_TILE_TOY_NC:.4f}) R_gauss={R_GAUSS:.4f}")
print(f"K_3dgs={K3_pop:.2f} K_box={Kbox_pop:.2f} K_fastgs={Kf_pop:.2f}  K_mult={ {k: round(v, 2) for k, v in K_MULT.items()} }")


# ===========================================================================
# Mô hình chi phí
# ===========================================================================
def split_for(F):
    rest = 1 - F
    return (0.15 / 0.85 * rest, 0.55 / 0.85 * rest, 0.15 / 0.85 * rest, F)


def terms(rg, rt, ra, split=(0.15, 0.55, 0.15, 0.15), ov=0.0):
    a, b, c, f = split
    return np.array([a * rg, b * rg * rt * (1 + ov), c * rg * ra, f])


CONFIGS = [  # (nhãn, rg, rt, ra, ov)
    ("3DGS\n(chuẩn hoá)", 1, 1, 1, 0),
    ("FastGS\ncả ba đòn bẩy", R_GAUSS, R_TILE_POP, R_ADAM, OV),
    ("chỉ $K$\n(compact box)", 1, R_TILE_POP, 1, 0),
    ("chỉ $N$\n(ADC)", R_GAUSS, 1, 1, OV),
    ("chỉ Adam\nthưa", 1, 1, R_ADAM, 0),
]
TERMS = [terms(rg, rt, ra, ov=ov) for _, rg, rt, ra, ov in CONFIGS]
TOT = [t.sum() for t in TERMS]
SPEED = [1 / t for t in TOT]
S_ALL, S_K, S_N, S_A = SPEED[1:]
PROD = S_K * S_N * S_A
print("speed:", [round(s, 3) for s in SPEED], "prod", round(PROD, 3))


# ===========================================================================
# Hình 1 — stacked bar T_iter
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5.6))
x = np.arange(len(CONFIGS))
labels = ['$aN$ (Projection)', '$bNK$ (Rasterizer + backward)', r'$cN\cdot\mathbb{1}[\mathrm{Adam}]$', '$F$ (loss/SSIM/IO)']
cols = [C_PRE, C_RAS, C_ADAM, C_FIX]
bottom = np.zeros(len(CONFIGS))
for k in range(4):
    vals = np.array([t[k] for t in TERMS])
    ax.bar(x, vals, 0.6, bottom=bottom, color=cols[k], label=labels[k], edgecolor='white', linewidth=0.8)
    for i, v in enumerate(vals):
        if v >= 0.045:
            ax.text(x[i], bottom[i] + v / 2, f"{v:.3f}", ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
    bottom += vals
for i in range(len(CONFIGS)):
    ax.text(x[i], TOT[i] + 0.02, f"$T$ = {TOT[i]:.4f}\n{SPEED[i]:.2f}×", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(0.15, color='#555', ls='--', lw=1, label='trần Amdahl: $F$ = 0.15 → tối đa 6.67×')
ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in CONFIGS])
ax.set_ylabel(r'$T_{\mathrm{iter}} / T^{\mathrm{3dgs}}$')
ax.set_ylim(0, 1.25)
ax.set_title(r'$T_{\mathrm{iter}} = aN + bNK + cN\cdot\mathbb{1}[\mathrm{Adam}] + F$, chia 15/55/15/15; '
             f'$R_g$={R_GAUSS:.3f}, $R_t$={R_TILE_POP:.3f}, $R_a$={R_ADAM:.3f}, ov +2%', fontsize=10)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
save(fig, 'ch08_cost_stack.png')


# ===========================================================================
# Hình 2 — ba tỉ số theo độ tin cậy + tốc độ riêng lẻ vs tích
# ===========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1.6, 1]})
ratios = [
    (r'$R_{\mathrm{adam}}$' + '\n16563/\n59998', R_ADAM, C_COUNT),
    (r'$R_{\mathrm{tile}}$' + '\nví dụ 3.8\n9/25', R_TILE_EX, C_MEAS),
    (r'$R_{\mathrm{tile}}$' + '\nquần thể\n3000 splat', R_TILE_POP, C_MEAS),
    (r'$R_{\mathrm{tile}}$' + '\nđồ chơi\nclamp 3×2', R_TILE_TOY, C_MEAS),
    (r'$R_{\mathrm{tile}}$' + '\nđồ chơi\nkhông clamp', R_TILE_TOY_NC, C_MEAS),
    (r'$R_{\mathrm{gauss}}$' + '\n$N_{tb}$\n293k/933k', R_GAUSS, C_ASSUME),
]
xs = np.arange(len(ratios))
a1.bar(xs, [r[1] for r in ratios], 0.62, color=[r[2] for r in ratios], edgecolor='black', linewidth=0.6)
for i, (_, v, _) in enumerate(ratios):
    a1.text(i, v + 0.015, f"{v:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
a1.set_xticks(xs)
a1.set_xticklabels([r[0] for r in ratios], fontsize=8.5)
a1.set_ylim(0, 1.3)
a1.axhline(1, color='k', lw=0.8, ls=':')
a1.set_ylabel('tỉ số FastGS / 3DGS (1 = không tiết kiệm)')
a1.set_title('Ba tỉ số đi vào mô hình chi phí — màu theo độ tin cậy', fontsize=10)
a1.legend(handles=[Patch(color=C_COUNT, label='đếm chính xác từ code (optimizer_step)'),
                   Patch(color=C_MEAS, label='đo được bằng hình học (auxiliary.h)'),
                   Patch(color=C_ASSUME, label='giả định (spawn/prune demo) — cần A/B')],
          loc='upper right', fontsize=8.5, framealpha=0.95)

names = ['chỉ $K$', 'chỉ $N$', 'chỉ Adam', 'tích 3 số\nriêng lẻ', 'cả ba\n(mô hình)']
vals = [S_K, S_N, S_A, PROD, S_ALL]
bcol = [C_MEAS, C_ASSUME, C_COUNT, '#bbbbbb', '#7b3294']
bars = a2.bar(np.arange(5), vals, 0.62, color=bcol, edgecolor='black', linewidth=0.6)
bars[3].set_hatch('//')
for i, v in enumerate(vals):
    a2.text(i, v + 0.08, f"{v:.2f}×", ha='center', va='bottom', fontsize=10, fontweight='bold')
a2.axhline(1 / 0.15, color='red', ls='--', lw=1.2)
a2.text(4.3, 1 / 0.15 + 0.1, 'trần $1/F$ = 6.67×', ha='right', va='bottom', fontsize=9, color='red')
a2.annotate('', xy=(4, S_ALL), xytext=(3, PROD), arrowprops=dict(arrowstyle='->', color='#555', lw=1))
a2.text(3.5, 5.15, f'−{PROD - S_ALL:.2f}: $F$ không co', ha='center', fontsize=8.5, color='#555')
a2.set_xticks(np.arange(5))
a2.set_xticklabels(names, fontsize=9)
a2.set_ylim(0, 7.6)
a2.set_ylabel('tốc độ = $1/T$')
a2.set_title(f'Tốc độ từng đòn bẩy: {S_K:.2f}×{S_N:.2f}×{S_A:.2f} = {PROD:.2f} > {S_ALL:.2f}', fontsize=10)
save(fig, 'ch08_levers.png')


# ===========================================================================
# Hình 3 — Amdahl: tốc độ theo F
# ===========================================================================
Fs = np.linspace(0.02, 0.5, 200)
s_all = np.array([1 / terms(R_GAUSS, R_TILE_POP, R_ADAM, split_for(F), OV).sum() for F in Fs])
s_k = np.array([1 / terms(1, R_TILE_POP, 1, split_for(F)).sum() for F in Fs])
s_n = np.array([1 / terms(R_GAUSS, 1, 1, split_for(F), OV).sum() for F in Fs])
s_a = np.array([1 / terms(1, 1, R_ADAM, split_for(F)).sum() for F in Fs])
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(Fs, 1 / Fs, 'r--', lw=1.5, label=r'trần Amdahl $S_{\max} = 1/F$')
ax.plot(Fs, s_all, color='#7b3294', lw=2.5, label='cả ba đòn bẩy')
ax.plot(Fs, s_n, color=C_ASSUME, lw=1.3, label='chỉ $N$')
ax.plot(Fs, s_k, color=C_MEAS, lw=1.3, label='chỉ $K$')
ax.plot(Fs, s_a, color=C_COUNT, lw=1.3, label='chỉ Adam')
for F, dy in ((0.10, 0.9), (0.15, 0.9), (0.25, 0.9)):
    sp = 1 / terms(R_GAUSS, R_TILE_POP, R_ADAM, split_for(F), OV).sum()
    ax.plot(F, sp, 'o', color='#7b3294', ms=7, zorder=5)
    ax.annotate(f'$F$={F:.2f} → {sp:.2f}×', xy=(F, sp), xytext=(F + 0.03, sp + dy),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='#555', lw=0.8))
    ax.plot(F, 1 / F, 's', color='red', ms=5, zorder=5)
ax.annotate('trần 1/0.15 = 6.67×', xy=(0.15, 1 / 0.15), xytext=(0.2, 8.6), fontsize=9, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
ax.text(0.10 + 0.005, 10.0, '10.0×', fontsize=8, color='red', va='bottom')
ax.text(0.25 + 0.005, 4.0, '4.0×', fontsize=8, color='red', va='bottom')
ax.axvspan(0.10, 0.25, color='#7b3294', alpha=0.06)
ax.text(0.175, 0.5, 'dải giả định $F$ = 10–25%\n→ cả ba 2.87–4.60×', ha='center', fontsize=8.5, color='#7b3294')
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 12)
ax.set_xlabel('tỉ lệ $F$ (phần không scale theo $N$: loss/SSIM/IO), ba phần còn lại giữ tỉ lệ 15:55:15')
ax.set_ylabel('tốc độ = $1/T$')
ax.set_title(r'Trần Amdahl: tốc độ cả ba theo $F$ ($R_g$=0.314, $R_t$=0.289, $R_a$=0.276, ov +2%)', fontsize=10)
ax.legend(loc='upper right', fontsize=9)
save(fig, 'ch08_amdahl.png')


# ===========================================================================
# Hình 4 — đếm Adam step tích luỹ + R_adam theo --iterations
# ===========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.5, 1]})
t = np.arange(0, 30001)
a1.step(t, cm3 + cs3, where='post', color='#7f7f7f', lw=2, label='3DGS: optimizer + shoptimizer (2 step/vòng)')
a1.step(t, cmf, where='post', color=C_MEAS, lw=2, label='FastGS optimizer (1 → 1/32 → 1/64)')
a1.step(t, csf, where='post', color=C_ASSUME, lw=2, label='FastGS shoptimizer (1/16 → 1/32 → 1/64)')
a1.step(t, cmf + csf, where='post', color=C_COUNT, lw=1.3, ls='--', label='FastGS tổng')
for xv in (15000, 20000):
    a1.axvline(xv, color='k', lw=0.8, ls=':')
a1.text(14700, 47000, '15k: 1/32', ha='right', fontsize=8.5)
a1.text(20300, 47000, '20k: 1/64', ha='left', fontsize=8.5)
a1.text(29900, m3 + s3 + 1200, f'{m3 + s3}', ha='right', fontsize=9.5, fontweight='bold', color='#555')
a1.text(29900, mf + sf + 1200, f'tổng {mf + sf}', ha='right', fontsize=9.5, fontweight='bold', color=C_COUNT)
a1.text(29900, mf - 2600, f'{mf}', ha='right', fontsize=9.5, fontweight='bold', color=C_MEAS)
a1.text(29900, sf + 1200, f'{sf}', ha='right', fontsize=9.5, fontweight='bold', color=C_ASSUME)
a1.text(7500, 16500, f'15000 (t≤15k)\n+157 (bội 32)\n+156 (bội 64)', fontsize=8.5, color=C_MEAS, ha='center')
a1.text(7500, 3300, '937 = 15000//16', fontsize=8.5, color=C_ASSUME, ha='center')
a1.set_xlim(0, 30000)
a1.set_ylim(0, 68000)
a1.set_xlabel('vòng lặp $t$ (vòng 30000 không step: train.py:160)')
a1.set_ylabel('số lần .step() tích luỹ')
a1.set_title(f'Số Adam step tích luỹ → $R_{{\\mathrm{{adam}}}}$ = {mf + sf}/{m3 + s3} = {R_ADAM:.4f}', fontsize=10)
a1.legend(loc='upper left', fontsize=8.5, framealpha=0.95)

its = [15000, 20000, 30000]
rads, fps = [], []
for it in its:
    m3i, s3i, _, _ = adam_count(total_iters=it, fastgs=False)
    mfi, sfi, _, _ = adam_count(total_iters=it, fastgs=True)
    rads.append((mfi + sfi) / (m3i + s3i))
    fps.append(len([u for u in range(1, it + 1) if u % 3000 == 0 and 15000 < u < 30000]))
b = a2.bar(np.arange(3), rads, 0.72, color=[C_ASSUME, C_MEAS, C_COUNT], edgecolor='black', linewidth=0.6)
for i, (r, f) in enumerate(zip(rads, fps)):
    a2.text(i, r + 0.012, f"$R_{{\\mathrm{{adam}}}}$ = {r:.3f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    a2.text(i, r / 2, f"final_prune\n{f} lần", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
a2.set_xticks(np.arange(3))
a2.set_xticklabels([f'--iterations\n{it}' for it in its])
a2.set_ylim(0, 0.68)
a2.set_ylabel(r'$R_{\mathrm{adam}}$')
a2.set_title('Sổ tay 8.8: rút ngắn --iterations mất phần Adam "rẻ"', fontsize=10)
save(fig, 'ch08_adam_count.png')


# ===========================================================================
# Hình 5 — phân bố K trên quần thể demo + K_fastgs theo mult và alpha
# ===========================================================================
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1.5, 1, 1]})


def ecdf(a):
    s = np.sort(a)
    return s, np.arange(1, len(s) + 1) / len(s)


for arr, name, col in ((K3_arr, r'$K^{\mathrm{3dgs}}$ hộp vuông $3\sigma$', '#7f7f7f'),
                       (Kbox_arr, r'$K^{\mathrm{box}}$ compact box', C_ASSUME),
                       (Kf_arr, r'$K^{\mathrm{fastgs}}$ + lọc ellipse', C_MEAS)):
    xs_, ys_ = ecdf(arr)
    a1.step(xs_, ys_, where='post', color=col, lw=1.8, label=f'{name}: mean {arr.mean():.2f}')
    a1.axvline(arr.mean(), color=col, ls='--', lw=1)
a1.text(K3_pop * 1.1, 0.12, f'{K3_pop:.2f}', color='#7f7f7f', fontsize=9, fontweight='bold')
a1.text(Kbox_pop * 1.1, 0.05, f'{Kbox_pop:.2f}', color=C_ASSUME, fontsize=9, fontweight='bold')
a1.text(Kf_pop * 0.45, 0.05, f'{Kf_pop:.2f}', color=C_MEAS, fontsize=9, fontweight='bold', ha='right')
a1.set_xscale('log')
a1.set_xlim(0.8, 3000)
a1.set_ylim(0, 1.02)
a1.set_xlabel('số tile $K$ mỗi splat (log)')
a1.set_ylabel('ECDF (3000 splat, seed 0, mult = 0.5)')
a1.set_title(f'$R_{{\\mathrm{{tile}}}}$ = {Kf_pop:.2f}/{K3_pop:.2f} = {R_TILE_POP:.3f}  '
             f'(hộp ×{Kbox_pop / K3_pop:.3f}, ellipse ×{Kf_pop / Kbox_pop:.3f})', fontsize=10)
a1.legend(loc='lower right', fontsize=8.5, framealpha=0.95)

ms = sorted(K_MULT)
kv = [K_MULT[m] for m in ms]
a2.plot(ms, kv, 'o-', color=C_MEAS, lw=2, ms=7)
for m, k in zip(ms, kv):
    a2.text(m, k + 0.35, f'{k:.2f}\n$R_t$={k / K3_pop:.3f}', ha='center', fontsize=8.5)
a2.axhline(K3_pop, color='#7f7f7f', ls='--', lw=1)
a2.text(0.62, K3_pop - 0.9, f'$K^{{\\mathrm{{3dgs}}}}$ = {K3_pop:.2f} (không đổi theo mult)', ha='center', fontsize=8.5, color='#555')
a2.set_xticks(ms)
a2.set_xlim(0.15, 1.1)
a2.set_ylim(0, 25)
a2.set_xlabel(r'mult ($t = \mathrm{mult}\cdot 2\ln(255\alpha)$)')
a2.set_ylabel(r'$K^{\mathrm{fastgs}}$ trung bình')
a2.set_title('mult 1.0 vẫn < 3DGS (SnugBox)', fontsize=10)

ops = [r[0] for r in OPAC_ROWS]
kop = [r[3] for r in OPAC_ROWS]
tt = [r[1] for r in OPAC_ROWS]
a3.plot(ops, kop, 's-', color=C_ASSUME, lw=2, ms=7)
for o, k, tv in zip(ops, kop, tt):
    a3.text(o, k + 0.3, f'{k:.2f}\n$t$={tv:.2f}', ha='center', fontsize=8.5)
a3.set_xscale('log')
a3.set_xticks(ops)
a3.set_xticklabels([str(o) for o in ops])
a3.set_xlim(0.012, 1.6)
a3.set_ylim(0, 13)
a3.set_xlabel(r'opacity $\alpha$ (log)')
a3.set_ylabel(r'$K^{\mathrm{fastgs}}$ ($\sigma\prime$ = 8 px, mult = 0.5)')
a3.set_title(r'$K$ theo $\alpha$: splat mờ → hộp nhỏ (Phụ lục 1a)', fontsize=10)
save(fig, 'ch08_rtile_demo.png')


# ===========================================================================
# Hình 6 — overhead theo densification_interval
# ===========================================================================
def count_densify(iters=30000, from_it=500, until=15000, interval=500):
    return [u for u in range(1, iters + 1) if u < until and u > from_it and u % interval == 0]


ivs = [100, 200, 500, 1000]
nd = [len(count_densify(interval=i)) for i in ivs]
fwd = [n * 10 * 2 for n in nd]
ov_up = [100 * f / 30000 for f in fwd]
ov_3 = [o / 3 for o in ov_up]
n_fp = len([u for u in range(1, 30001) if u % 3000 == 0 and 15000 < u < 30000])
print("densify:", dict(zip(ivs, nd)), "fwd:", fwd, "ov%:", [round(o, 2) for o in ov_up], "final_prune", n_fp)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5))
xs = np.arange(4)
a1.bar(xs, nd, 0.6, color=C_COUNT, edgecolor='black', linewidth=0.6)
for i in range(4):
    ts = count_densify(interval=ivs[i])
    a1.text(xs[i], nd[i] + 3, f'$n$ = {nd[i]}\n$t$ = {ts[0]}…{ts[-1]}\n→ {fwd[i]} forward phụ',
            ha='center', va='bottom', fontsize=8.5, color='black')
    a1.text(xs[i], nd[i] / 2, f'{nd[i]}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
a1.set_xticks(xs)
a1.set_xticklabels([f'interval {i}' for i in ivs])
a1.set_ylim(0, 190)
a1.set_ylabel('số lần densify (train.py:132: 500 < $t$ < 15000, $t$ % interval = 0)')
a1.set_title(f'$n_{{\\mathrm{{densify}}}}$ = ($t_{{cuối}}$ − $t_{{đầu}}$)/interval + 1, forward phụ = $n$·10 cam·2 render\n'
             f'(final_prune_fastgs: {n_fp} lần × 20 = {n_fp * 20} forward, không tính)', fontsize=9.5)

a2.bar(xs - 0.2, ov_up, 0.4, color='#e15759', edgecolor='black', linewidth=0.6, label='cận trên: forward phụ / 30000')
a2.bar(xs + 0.2, ov_3, 0.4, color='#f4a582', edgecolor='black', linewidth=0.6, label='÷3 (1 vòng ≈ 3 forward)')
for i in range(4):
    a2.text(xs[i] - 0.2, ov_up[i] + 0.15, f'{ov_up[i]:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    a2.text(xs[i] + 0.2, ov_3[i] + 0.15, f'{ov_3[i]:.2f}%', ha='center', va='bottom', fontsize=8.5)
a2.axhline(100 * OV, color='k', ls=':', lw=1)
a2.text(3.4, 100 * OV + 0.15, 'demo scoring_overhead() = 2.00% (30 lần)', ha='right', fontsize=8, color='#555')
a2.annotate(f'interval 500 → {nd[2]} lần\n→ {ov_up[2]:.2f}% (chương 8.6: 1.9%)', xy=(xs[2] - 0.2, ov_up[2]),
            xytext=(1.5, 6.2), fontsize=9, arrowprops=dict(arrowstyle='->', color='#555'))
a2.annotate(f'interval 100 → {nd[0]} lần\n→ {ov_up[0]:.2f}% (chương ghi 145, ≈10%)', xy=(xs[0] - 0.2, ov_up[0]),
            xytext=(0.7, 10.6), fontsize=9, arrowprops=dict(arrowstyle='->', color='#555'))
a2.set_xticks(xs)
a2.set_xticklabels([f'interval {i}' for i in ivs])
a2.set_ylim(0, 12.5)
a2.set_ylabel('overhead scoring (% chi phí 30000 vòng)')
a2.set_title('Overhead compute_gaussian_score_fastgs theo densification_interval', fontsize=10)
a2.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
save(fig, 'ch08_overhead.png')
