"""
Bài test số chương 8 — Tổng hợp: mô hình chi phí và ba đòn bẩy.
Chỉ dùng numpy. Tính lại bằng tay ba tỉ số R_adam, R_tile, R_gauss, mô hình
T_iter = aN + bNK + cN·1[Adam] + F, trần Amdahl, overhead scoring, và bảng
"sổ tay" cho các thay đổi cấu hình ở mục 8.8.

Tham chiếu code:
  - scene/gaussian_model.py:190-209   optimizer_step (lịch 1 / 1/32 / 1/64, SH 1/16)
  - train.py:66                        for iteration in range(1, iterations+1)
  - train.py:127,132                   iteration < densify_until_iter
                                       and iteration > densify_from_iter
                                       and iteration % densification_interval == 0
  - train.py:152                       final prune: it % 3000 == 0 and 15000 < it < 30000
  - train.py:160                       if iteration < opt.iterations: optimizer_step
  - utils/fast_utils.py:13             num_cams = 10; mỗi cam render 2 lần (:63, :72)
  - arguments/__init__.py:75-92        iterations=30000, densification_interval=100,
                                       densify_from_iter=500, densify_until_iter=15000,
                                       highfeature_lr=0.005 ; gaussian_model.py:174  lr/20
  - submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/auxiliary.h:312-345
                                       t = mult*2ln(255 o); rect từ bbox
  - demos/fastgs_cost_model.py         hàm được import lại để đối chiếu

Chạy:  python DOCS/Report/test/scripts/ch08_test.py
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "demos"))
import fastgs_cost_model as fcm  # noqa: E402

TILE = 16


def hdr(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def sub(s):
    print()
    print("--- " + s)


# ==========================================================================
# 8.3(a)  R_adam — đếm vòng lặp theo lịch chương 6.4 / gaussian_model.py:190
# ==========================================================================
def adam_count(total_iters=30000, fastgs=True):
    """Đếm số lần .step() của optimizer chính và shoptimizer.
    train.py:160: `if iteration < opt.iterations` -> vòng cuối KHÔNG step,
    nên duyệt t = 1..total_iters-1."""
    main = sh = 0
    per_phase = {"main_le15k": 0, "main_15k_20k": 0, "main_gt20k": 0,
                 "sh_le15k": 0, "sh_15k_20k": 0, "sh_gt20k": 0}
    for t in range(1, total_iters):
        if not fastgs:
            main += 1
            sh += 1
            continue
        if t <= 15000:
            main += 1
            per_phase["main_le15k"] += 1
            if t % 16 == 0:
                sh += 1
                per_phase["sh_le15k"] += 1
        elif t <= 20000:
            if t % 32 == 0:
                main += 1
                sh += 1
                per_phase["main_15k_20k"] += 1
                per_phase["sh_15k_20k"] += 1
        else:
            if t % 64 == 0:
                main += 1
                sh += 1
                per_phase["main_gt20k"] += 1
                per_phase["sh_gt20k"] += 1
    return main, sh, per_phase


hdr("8.3(a)  R_adam — đếm số lần Adam step (lịch gaussian_model.py:190-209)")
print("Công thức: 1_main(t) = 1 (t<=15000) ; [t%32=0] (15000<t<=20000) ; [t%64=0] (t>20000)")
print("           1_SH(t)   = [t%16=0] (t<=15000) ; = 1_main(t) (t>15000)")
print("Duyệt t = 1 .. 29999 (vòng 30000 không step vì train.py:160 `iteration < opt.iterations`)")
m3, s3, _ = adam_count(fastgs=False)
mf, sf, ph = adam_count(fastgs=True)
print(f"  main   : t<=15k = {ph['main_le15k']}, (15k,20k] bội 32 = {ph['main_15k_20k']}, "
      f">20k bội 64 = {ph['main_gt20k']}  -> tổng {mf}")
print(f"  SH     : t<=15k bội 16 = {ph['sh_le15k']}, (15k,20k] = {ph['sh_15k_20k']}, "
      f">20k = {ph['sh_gt20k']}  -> tổng {sf}")
print(f"  3DGS   : main = {m3}, SH = {s3}, tổng = {m3 + s3}")
print(f"  FastGS : main = {mf}, SH = {sf}, tổng = {mf + sf}")
R_ADAM = (mf + sf) / (m3 + s3)
print(f"  R_adam = {mf + sf}/{m3 + s3} = {R_ADAM:.6f}  (chương 8.3 ghi 0.276)")
print(f"  R_main = {mf}/{m3} = {mf / m3:.4f} ; R_SH = {sf}/{s3} = {sf / s3:.4f}")
# Kiểm tra nhẩm
print("  Kiểm tra nhẩm: bội 32 trong (15000,20000] = 20000/32 - floor(15000/32) = "
      f"{20000 // 32 - 15000 // 32} ; bội 64 trong (20000,29999] = "
      f"{29999 // 64 - 20000 // 64} ; bội 16 trong [1,15000] = {15000 // 16}")
d_m, d_s = fcm.adam_steps(fastgs=True)
print(f"  Đối chiếu demos.adam_steps(fastgs=True) = ({d_m}, {d_s}) -> "
      f"{'KHỚP' if (d_m, d_s) == (mf, sf) else 'LỆCH'}")
# Nếu KHÔNG bỏ vòng cuối:
mf2, sf2, _ = adam_count(total_iters=30001, fastgs=True)
m32, s32, _ = adam_count(total_iters=30001, fastgs=False)
print(f"  Nếu đếm nhầm cả vòng 30000 (chia hết 64): FastGS {mf2 + sf2}, 3DGS {m32 + s32} "
      f"-> R = {(mf2 + sf2) / (m32 + s32):.6f} (sai lệch nhỏ nhưng không đúng code)")


# ==========================================================================
# 8.3(b)  R_tile
# ==========================================================================
hdr("8.3(b)  R_tile — ba cách")

# ---- (i) ví dụ chương 3.8 -------------------------------------------------
sub("(i) Ví dụ chương 3.8 (Sigma' = [[117,54],[54,36]], mu'=(120,88), alpha=1, mult=0.5)")
S = np.array([[117.0, 54.0], [54.0, 36.0]])
mu = np.array([120.0, 88.0])
mid = (S[0, 0] + S[1, 1]) / 2
rad = math.sqrt(((S[0, 0] - S[1, 1]) / 2) ** 2 + S[0, 1] ** 2)
lmax, lmin = mid + rad, mid - rad
print(f"  lambda_max = {mid:.4g} + {rad:.4g} = {lmax:.6g}, lambda_min = {lmin:.6g}, rho = {math.sqrt(lmax / lmin):.4g}")
r3 = math.ceil(3 * math.sqrt(lmax))
print(f"  r = ceil(3*sqrt({lmax:.6g})) = {r3}")
# grid rộng vô hạn (ảnh lớn) -> không clamp
rect_min = np.floor((mu - r3) / TILE).astype(int)
rect_max = np.floor((mu + r3 + 15) / TILE).astype(int)
K3_ex = int(np.prod(rect_max - rect_min))
print(f"  rect_min = floor(({mu[0]:.0f}-{r3})/16, ({mu[1]:.0f}-{r3})/16) = {rect_min.tolist()} ; "
      f"rect_max = floor((mu+r+15)/16) = {rect_max.tolist()} -> K^3dgs = "
      f"{rect_max[0] - rect_min[0]}x{rect_max[1] - rect_min[1]} = {K3_ex}")
t_ex = 0.5 * 2 * math.log(255 * 1.0)
hx, hy = math.sqrt(t_ex * S[0, 0]), math.sqrt(t_ex * S[1, 1])
print(f"  t = 0.5*2ln(255) = {t_ex:.4f} ; half_x = sqrt(t*117) = {hx:.4f}, half_y = sqrt(t*36) = {hy:.4f}")
bmin = mu - np.array([hx, hy])
bmax = mu + np.array([hx, hy])
rmin_f = (bmin / TILE).astype(int)          # auxiliary.h:334 (int)(bbox_min/BLOCK)
rmax_f = (bmax / TILE + 1).astype(int)      # auxiliary.h:338 (int)(bbox_max/BLOCK + 1)
Kbox_ex = int(np.prod(rmax_f - rmin_f))
print(f"  compact box: rect_min = int(bbox_min/16) = {rmin_f.tolist()}, "
      f"rect_max = int(bbox_max/16 + 1) = {rmax_f.tolist()} -> K^box = "
      f"{rmax_f[0] - rmin_f[0]}x{rmax_f[1] - rmin_f[1]} = {Kbox_ex}")
# lọc ellipse: dùng hàm của demos với px,py cố định = vị trí tâm trong tile
Minv = np.linalg.inv(S)
A, B, C = Minv[0, 0], Minv[0, 1], Minv[1, 1]
print(f"  conic (A,B,C) = ({A:.6g}, {B:.6g}, {C:.6g})  (chương 3.8: 1/36, -1/24, 13/144)")
theta = 0.5 * math.atan2(2 * S[0, 1], S[0, 0] - S[1, 1])
# Đếm tile giao ellipse bằng cùng hàm _min_quadratic_on_box của demos, toạ độ tuyệt đối
Kf_ex = 0
rows = {}
for iy in range(rmin_f[1], rmax_f[1]):
    for ix in range(rmin_f[0], rmax_f[0]):
        q = fcm._min_quadratic_on_box(A, B, C, ix * TILE - mu[0], (ix + 1) * TILE - mu[0],
                                      iy * TILE - mu[1], (iy + 1) * TILE - mu[1])
        if q <= t_ex:
            Kf_ex += 1
            rows[iy] = rows.get(iy, 0) + 1
print(f"  lọc ellipse (min_T d^T M d <= t): theo hàng y = {[rows[k] for k in sorted(rows)]} -> K^fastgs = {Kf_ex}")
R_TILE_EX = Kf_ex / K3_ex
print(f"  R_tile (3.8) = {Kf_ex}/{K3_ex} = {R_TILE_EX:.4f}")
print(f"  đối chiếu demos.tiles_fastgs(lam, theta, o=1, mult=0.5, px, py) = "
      f"{fcm.tiles_fastgs(lmax, lmin, theta, 1.0, 0.5, mu[0] % TILE, mu[1] % TILE):.0f}")

# ---- (ii) quần thể mô phỏng ------------------------------------------------
sub("(ii) Quần thể mô phỏng của demos/fastgs_cost_model.py (3000 splat, seed 0)")
fcm.RNG = np.random.default_rng(0)   # reset seed cho giống lúc chạy demo
pop = fcm.sample_population()
k3_pop, kbox_pop, kf_pop = fcm.tile_stats(pop, 0.5)
R_TILE_POP = kf_pop / k3_pop
print(f"  K_3dgs = {k3_pop:.4f}, K_box = {kbox_pop:.4f}, K_fastgs = {kf_pop:.4f}")
print(f"  R_tile (quần thể, mult=0.5) = {R_TILE_POP:.4f}  (demo in 0.290)")
print(f"  hai nguồn: hộp theo trục {kbox_pop / k3_pop:.3f}, lọc ellipse thêm {kf_pop / kbox_pop:.3f}")

# ---- (iii) cảnh đồ chơi ----------------------------------------------------
sub("(iii) Cảnh đồ chơi 00-scene.md, camera 1, alpha = 0.1, mult = 0.5")
print("  XẤP XỈ: Sigma' ~ (fx/t_z)^2 s^2 + 0.3 trên đường chéo, BỎ phần ngoài chéo của J")
print("          (J có số hạng -fx*t_x/t_z^2 ở cột 3; splat đẳng hướng nên bỏ qua để tính tay).")
print("          K^fastgs lấy = hộp compact (không lọc ellipse; ellipse tròn nên hộp = lọc trong hầu hết ô).")
pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.5], [-0.4, -0.2, 1.0], [0.3, -0.5, 0.2]])
W, H, fx, fy = 48, 32, 40.0, 40.0
grid = (math.ceil(W / TILE), math.ceil(H / TILE))
alpha = 0.1
t_toy = 0.5 * 2 * math.log(255 * alpha)
print(f"  lưới tile = {grid[0]}x{grid[1]} ; t = 0.5*2ln(255*0.1) = {t_toy:.4f}")
D2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
K3_toy, Kf_toy = [], []
print(f"  {'i':>2} | {'s':>7} | {'t_z':>4} | {'mu_x,mu_y':>13} | {'S11':>7} | {'r':>3} | {'rect3':>11} | K3 | {'half':>6} | {'rectF':>11} | Kf")
for i in range(4):
    d2 = np.sort(D2[i])[1:4]
    s = math.sqrt(d2.mean())                                      # distCUDA2 -> sqrt
    tx, ty, tz = pts[i, 0], pts[i, 1], pts[i, 2] + 4.0            # camera 1: c=(0,0,-4)
    mux = fx * tx / tz + (W - 1) / 2                              # ndc2Pix
    muy = fy * ty / tz + (H - 1) / 2
    S11 = (fx / tz) ** 2 * s * s + 0.3
    r = math.ceil(3 * math.sqrt(S11))
    # 3DGS getRect (auxiliary.h gốc): min(grid, max(0, (p-r)/16)), min(grid, max(0,(p+r+15)/16))
    x0 = min(grid[0], max(0, int((mux - r) / TILE)))
    y0 = min(grid[1], max(0, int((muy - r) / TILE)))
    x1 = min(grid[0], max(0, int((mux + r + 15) / TILE)))
    y1 = min(grid[1], max(0, int((muy + r + 15) / TILE)))
    k3 = (x1 - x0) * (y1 - y0)
    half = math.sqrt(t_toy * S11)
    # fastgs auxiliary.h:333-340
    fx0 = max(0, min(grid[0], int((mux - half) / TILE)))
    fy0 = max(0, min(grid[1], int((muy - half) / TILE)))
    fx1 = max(0, min(grid[0], int((mux + half) / TILE + 1)))
    fy1 = max(0, min(grid[1], int((muy + half) / TILE + 1)))
    kf = (fx1 - fx0) * (fy1 - fy0)
    K3_toy.append(k3)
    Kf_toy.append(kf)
    print(f"  {i + 1:>2} | {s:7.4f} | {tz:4.1f} | ({mux:5.2f},{muy:5.2f}) | {S11:7.3f} | {r:>3} | "
          f"[{x0},{x1})x[{y0},{y1}) | {k3:>2} | {half:6.3f} | [{fx0},{fx1})x[{fy0},{fy1}) | {kf:>2}")
K3_toy_sum, Kf_toy_sum = sum(K3_toy), sum(Kf_toy)
R_TILE_TOY = Kf_toy_sum / K3_toy_sum
print(f"  P^3dgs = sum K = {K3_toy_sum}, P^fastgs = {Kf_toy_sum} -> R_tile (đồ chơi) = {R_TILE_TOY:.4f}")
print("  Nhận xét: ảnh 48x32 chỉ có 6 tile, cả hai hộp đều bị clamp vào lưới -> R_tile = 1;")
print("  không clamp thì hộp 3DGS 2r ~ 50 px ≈ 4x4 tile, hộp fastgs 2*half ~ 30 px ≈ 3x3 tile.")
# Không clamp (ảnh vô hạn) để thấy chênh lệch
K3_nc, Kf_nc = 0, 0
for i in range(4):
    d2 = np.sort(D2[i])[1:4]
    s = math.sqrt(d2.mean())
    tx, ty, tz = pts[i, 0], pts[i, 1], pts[i, 2] + 4.0
    mux = fx * tx / tz + (W - 1) / 2
    muy = fy * ty / tz + (H - 1) / 2
    S11 = (fx / tz) ** 2 * s * s + 0.3
    r = math.ceil(3 * math.sqrt(S11))
    k3 = (math.floor((mux + r + 15) / TILE) - math.floor((mux - r) / TILE)) * \
         (math.floor((muy + r + 15) / TILE) - math.floor((muy - r) / TILE))
    half = math.sqrt(t_toy * S11)
    kf = (math.floor((mux + half) / TILE + 1) - math.floor((mux - half) / TILE)) * \
         (math.floor((muy + half) / TILE + 1) - math.floor((muy - half) / TILE))
    K3_nc += k3
    Kf_nc += kf
print(f"  Không clamp (ảnh vô hạn): P^3dgs = {K3_nc}, P^fastgs = {Kf_nc} -> R_tile = {Kf_nc / K3_nc:.4f}")


# ==========================================================================
# 8.3(c)  R_gauss — GIẢ ĐỊNH
# ==========================================================================
hdr("8.3(c)  R_gauss — GIẢ ĐỊNH (không có phép đo A/B trong repo)")
sub("Công thức đề bài: N_cuối = N0 [(1+r_spawn)(1-r_prune)]^n")
N0 = 4
g3 = (1 + 0.20) * (1 - 0.05)
gf = (1 + 0.12) * (1 - 0.08)
n3_end_a = N0 * g3 ** 145
nf_end_a = N0 * gf ** 28
print(f"  3DGS  : g = 1.20*0.95 = {g3:.4f}, n = 145 -> N_cuối = 4*{g3:.4f}^145 = {n3_end_a:.4e}")
print(f"  FastGS: g = 1.12*0.92 = {gf:.4f}, n = 28  -> N_cuối = 4*{gf:.4f}^28  = {nf_end_a:.4f}")
R_GAUSS_A = nf_end_a / n3_end_a
print(f"  R_gauss (bộ số đề bài) = {R_GAUSS_A:.3e}")
print("  -> Bộ số này phi thực tế: 1.14^145 nổ tung (3DGS thật dừng ở vài triệu Gaussian vì")
print("     prune opacity/size và VRAM). Chỉ minh hoạ R_gauss nhạy theo n và r như thế nào.")
sub("Bộ số của demos/fastgs_cost_model.py (đi vào bảng 8.5), N0 = 100 000, interval 500")
n3_mean, n3_end = fcm.mean_gaussians(spawn=0.10, prune=0.005)
nf_mean, nf_end = fcm.mean_gaussians(spawn=0.06, prune=0.010, final_prune_frac=0.05)
print(f"  3DGS  : spawn 0.10, prune 0.005, 29 lần (t=500..14500 vì demo dùng t<15000, t%500==0)")
print(f"          N_tb = {n3_mean:,.0f}, N_cuối = {n3_end:,.0f}")
print(f"  FastGS: spawn 0.06, prune 0.010, + final prune 5% x4 (18k,21k,24k,27k)")
print(f"          N_tb = {nf_mean:,.0f}, N_cuối = {nf_end:,.0f}")
R_GAUSS = nf_mean / n3_mean
print(f"  R_gauss (N trung bình) = {R_GAUSS:.4f} ; theo N_cuối = {nf_end / n3_end:.4f}")
# kiểm tra bằng tay: 3DGS N_cuối = 1e5*(1.10*0.995)^29
print(f"  kiểm tay: 1e5*(1.10*0.995)^29 = {1e5 * (1.10 * 0.995) ** 29:,.0f} ; "
      f"1e5*(1.06*0.99)^29*(0.95)^4 = {1e5 * (1.06 * 0.99) ** 29 * 0.95 ** 4:,.0f}")
print("  LƯU Ý: demo đếm 29 lần densify (t=500 cũng tính) trong khi train.py:132 (t>500) cho 28 — lệch 1 lần.")


# ==========================================================================
# 8.2 / 8.4 / 8.5  Mô hình chi phí
# ==========================================================================
hdr("8.2–8.5  T_iter = aN + bNK + cN·1[Adam] + F, chuẩn hoá T^3dgs = 1, chia 15/55/15/15")


def T(rg, rt, ra, split=(0.15, 0.55, 0.15, 0.15), ov=0.0):
    a, b, c, f = split
    return a * rg + b * rg * rt * (1 + ov) + c * rg * ra + f


def table(split, ov, rg, rt, ra, label):
    a, b, c, f = split
    print(f"  [{label}] split = {split}, overhead scoring = {100 * ov:.2f}%")
    combos = [("cả ba", rg, rt, ra, ov),
              ("chỉ K (compact box)", 1, rt, 1, 0),
              ("chỉ N (ADC)", rg, 1, 1, ov),
              ("chỉ Adam thưa", 1, 1, ra, 0)]
    out = {}
    for name, g, t, ad, o in combos:
        tf = T(g, t, ad, split, o)
        out[name] = 1 / tf
        print(f"    {name:<22}: T = {a}*{g:.3f} + {b}*{g:.3f}*{t:.3f}*{1 + o:.3f} + {c}*{g:.3f}*{ad:.3f} + {f}"
              f" = {tf:.4f} -> tốc độ {1 / tf:.3f}x")
    prod = out["chỉ K (compact box)"] * out["chỉ N (ADC)"] * out["chỉ Adam thưa"]
    print(f"    tích ba đòn bẩy riêng lẻ = {prod:.3f}x  vs  cả ba = {out['cả ba']:.3f}x  ; trần Amdahl 1/F = {1 / f:.3f}x")
    return out


OV_DEMO = fcm.scoring_overhead()          # 30*20/30000 = 0.02
res_demo = table((0.15, 0.55, 0.15, 0.15), OV_DEMO, R_GAUSS, R_TILE_POP, R_ADAM,
                 "R từ demo: R_gauss=0.314 (giả định), R_tile=0.290 (quần thể), R_adam=0.276")
print("  So với bảng 8.5 (3.83, 1.64, 2.38, 1.12): "
      + ", ".join(f"{v:.2f}" for v in [res_demo['cả ba'], res_demo['chỉ K (compact box)'],
                                        res_demo['chỉ N (ADC)'], res_demo['chỉ Adam thưa']]))
print()
res_ex = table((0.15, 0.55, 0.15, 0.15), 0.0, R_GAUSS, R_TILE_EX, R_ADAM,
               "R_tile = 0.36 (ví dụ 3.8), không overhead")
print()
res_toy = table((0.15, 0.55, 0.15, 0.15), 0.0, R_GAUSS, R_TILE_TOY, R_ADAM,
                "R_tile = 1.0 (cảnh đồ chơi, clamp lưới 3x2)")

sub("Bảng nhạy theo F (giữ tỉ lệ 15:55:15 cho ba phần biến thiên)")
print(f"  {'F':>5} | {'split':>28} | {'S_max=1/F':>9} | {'cả ba':>7} | {'chỉ K':>6} | {'chỉ N':>6} | {'chỉ Adam':>8}")
SENS = []
for F in (0.10, 0.15, 0.25):
    rest = 1 - F
    split = (0.15 / 0.85 * rest, 0.55 / 0.85 * rest, 0.15 / 0.85 * rest, F)
    all3 = 1 / T(R_GAUSS, R_TILE_POP, R_ADAM, split, OV_DEMO)
    onlyK = 1 / T(1, R_TILE_POP, 1, split, 0)
    onlyN = 1 / T(R_GAUSS, 1, 1, split, OV_DEMO)
    onlyA = 1 / T(1, 1, R_ADAM, split, 0)
    SENS.append((F, split, 1 / F, all3, onlyK, onlyN, onlyA))
    print(f"  {F:>5.2f} | ({split[0]:.3f},{split[1]:.3f},{split[2]:.3f},{split[3]:.2f}) | {1 / F:>9.2f} | "
          f"{all3:>7.2f} | {onlyK:>6.2f} | {onlyN:>6.2f} | {onlyA:>8.2f}")


# ==========================================================================
# 8.6  Overhead riêng của FastGS
# ==========================================================================
hdr("8.6  Overhead scoring — đếm bằng vòng lặp thật trên điều kiện train.py:127,132,152")


def count_densify(iters=30000, from_it=500, until=15000, interval=500):
    ts = [t for t in range(1, iters + 1)
          if t < until and t > from_it and t % interval == 0]
    return ts


def count_final_prune(iters=30000):
    return [t for t in range(1, iters + 1) if t % 3000 == 0 and 15000 < t < 30000]


NUM_CAMS, RENDERS_PER_CAM = 10, 2
for interval in (500, 100):
    ts = count_densify(interval=interval)
    formula = (14500 - 1000) // interval + 1 if interval == 500 else (14900 - 600) // interval + 1
    print(f"  interval = {interval}: đếm vòng lặp -> n_densify = {len(ts)} "
          f"(đầu {ts[0]}, cuối {ts[-1]}) ; công thức (t_cuối - t_đầu)/interval + 1 = {formula}")
    extra = len(ts) * NUM_CAMS * RENDERS_PER_CAM
    print(f"    forward phụ = {len(ts)}*{NUM_CAMS}*{RENDERS_PER_CAM} = {extra} ; "
          f"/30000 = {100 * extra / 30000:.2f}% (cận trên) ; /3 = {100 * extra / 30000 / 3:.2f}% (1 vòng ≈ 3 forward)")
fp = count_final_prune()
print(f"  final_prune_fastgs (t%3000==0, 15000<t<30000): {fp} -> {len(fp)} lần, "
      f"thêm {len(fp) * NUM_CAMS * RENDERS_PER_CAM} forward ({100 * len(fp) * 20 / 30000:.2f}%)")
print(f"  demos.scoring_overhead() = 15000//500 * 20 / 30000 = {fcm.scoring_overhead():.4f} "
      f"(30 lần, đếm cả t=500 và t=15000 — nhiều hơn code 2 lần) ; đếm đúng = {28 * 20 / 30000:.4f}")
N_DENS_500 = len(count_densify(interval=500))
N_DENS_100 = len(count_densify(interval=100))


# ==========================================================================
# 8.8  Bảng "sổ tay" cấu hình
# ==========================================================================
hdr("8.8  Sổ tay: mỗi thay đổi cấu hình -> con số bị ảnh hưởng")
# --iterations 15000 / 20000
for it in (15000, 20000, 30000):
    m3i, s3i, _ = adam_count(total_iters=it, fastgs=False)
    mfi, sfi, _ = adam_count(total_iters=it, fastgs=True)
    fpi = [t for t in range(1, it + 1) if t % 3000 == 0 and 15000 < t < 30000]
    print(f"  --iterations {it}: FastGS step = {mfi}+{sfi} = {mfi + sfi}, 3DGS = {m3i + s3i}, "
          f"R_adam = {(mfi + sfi) / (m3i + s3i):.4f} ; final_prune chạy {len(fpi)} lần {fpi}")
print(f"  --highfeature_lr 0.005 (mặc định) -> lr Adam f_rest = 0.005/20 = {0.005 / 20} ; "
      f"--highfeature_lr 0.02 -> {0.02 / 20}  (gaussian_model.py:174)")
for mult in (0.5, 1.0):
    for a_ in (1.0, 0.1):
        tt = mult * 2 * math.log(255 * a_)
        print(f"  --mult {mult}, alpha={a_}: t = {tt:.4f}, half = {math.sqrt(tt):.4f}*sigma', "
              f"diện tích hộp ∝ t = {tt:.3f}")
print(f"  mult 0.5 -> 1.0: t x{2:.0f}, half x{math.sqrt(2):.4f}, diện tích x2 ; K_fastgs quần thể: "
      f"{fcm.tile_stats(pop, 0.5)[2]:.2f} -> {fcm.tile_stats(pop, 1.0)[2]:.2f}")
print(f"  --densification_interval 100 (mặc định arguments) : n_densify = {N_DENS_100}, "
      f"forward phụ = {N_DENS_100 * 20}, = {100 * N_DENS_100 * 20 / 30000:.2f}% cận trên")
print(f"  --densify_until_iter 20000 (interval 500): n_densify = {len(count_densify(until=20000))} "
      f"(nhưng optimizer chỉ step 1/32 sau 15k -> densify trên gradient tích luỹ)")


# ==========================================================================
# Bảng tổng hợp
# ==========================================================================
hdr("TỔNG HỢP")
print(f"  R_adam  = {R_ADAM:.4f}   (ĐẾM ĐƯỢC chính xác từ code)")
print(f"  R_tile  = {R_TILE_EX:.4f} (3.8) / {R_TILE_POP:.4f} (quần thể) / {R_TILE_TOY:.4f} (đồ chơi, clamp)   (ĐO ĐƯỢC bằng hình học)")
print(f"  R_gauss = {R_GAUSS:.4f}   (GIẢ ĐỊNH — spawn/prune của demo)")
print(f"  Tốc độ cả ba (R_tile quần thể) = {res_demo['cả ba']:.3f}x ; chỉ K {res_demo['chỉ K (compact box)']:.3f}x ; "
      f"chỉ N {res_demo['chỉ N (ADC)']:.3f}x ; chỉ Adam {res_demo['chỉ Adam thưa']:.3f}x ; trần {1 / 0.15:.2f}x")
