"""Mô phỏng thu nhỏ các cơ chế tăng tốc của fastgs-lite (bản trước đây nằm trong notebook).

Chạy: `python demos/fastgs_mechanisms.py` (chỉ cần numpy; matplotlib là tuỳ chọn).
Giải thích công thức đi kèm: DOCS/fastgs-acceleration-method.md
"""

import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)


# --- 1. Bản đồ lỗi nhị phân trên mỗi góc nhìn -----------------------------------
def normalize_minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def error_mask(height=6, width=8, loss_thresh=0.1):
    """Chuẩn hoá min-max bản đồ lỗi rồi ngưỡng thành mặt nạ pixel sai."""
    error = rng.uniform(0.0, 0.05, size=(height, width))
    error[0:3, 0:3] += rng.uniform(0.2, 0.6, size=(3, 3))
    error_hat = normalize_minmax(error)
    return error_hat, (error_hat > loss_thresh).astype(int)


def accum_metric_counts(mask, footprint_map):
    """Đổ ngược mặt nạ pixel về từng Gaussian: đếm pixel sai mà Gaussian đó phủ."""
    counts = np.zeros(len(footprint_map), dtype=int)
    for index, pixels in footprint_map.items():
        counts[index] = sum(mask[r, c] for r, c in pixels)
    return counts


# --- 2. Gộp nhiều góc nhìn thành importance + pruning ----------------------------
def merge_views(counts, photometric_loss):
    importance = np.floor(counts.mean(axis=1)).astype(int)
    raw = (counts * photometric_loss).sum(axis=1)
    pruning = (raw - raw.min()) / (raw.max() - raw.min())
    return importance, raw, pruning


# --- 3. Densification có điều kiện kép ------------------------------------------
def densify_masks(grad, grad_abs, scale_max, importance_score,
                  grad_thresh=0.0002, grad_abs_thresh=0.0012, size_cut=0.001):
    small = scale_max <= size_cut
    clone_3dgs = (grad >= grad_thresh) & small
    split_3dgs = (grad_abs >= grad_abs_thresh) & ~small
    metric = importance_score > 5
    return clone_3dgs, split_3dgs, clone_3dgs & metric, split_3dgs & metric


def final_prune_fastgs(opacity, pruning_score, min_opacity=0.1):
    return (opacity < min_opacity) | (pruning_score > 0.9)


# --- 4. Compact box: số tile mỗi splat (--mult) ----------------------------------
def tiles_touched_3dgs(sigma_x=22.0, sigma_y=14.0, tile=16):
    """3DGS gốc: hộp chữ nhật 3-sigma, không nhìn tới opacity."""
    tiles_x = np.ceil((2 * 3 * sigma_x) / tile) + 1
    tiles_y = np.ceil((2 * 3 * sigma_y) / tile) + 1
    return int(tiles_x * tiles_y)


def compact_box_halfwidths(mult, sigma_x=22.0, sigma_y=14.0, opacity=1.0):
    """Nua canh hop bao cua Compact Box (auxiliary.h:336-343).

    Nguong level-set noi opacity*G = 1/255:  t = mult * 2*ln(255*o).
    Hop bao ellipse  d^T M d <= t  co nua canh  sqrt(t) * sigma  theo moi truc,
    tuc canh co theo sqrt(mult) — KHONG phai theo mult.
    """
    t = mult * 2.0 * np.log(255.0 * opacity)
    if t <= 0.0:  # o < 1/255: sqrt trong kernel nhan doi so am
        return 0.0, 0.0, t
    k = np.sqrt(t)
    return k * sigma_x, k * sigma_y, t


def tiles_touched(mult, sigma_x=22.0, sigma_y=14.0, opacity=1.0, tile=16):
    """So tile hop compact box cham toi (chan tren: kernel con cat theo ellipse)."""
    half_x, half_y, _ = compact_box_halfwidths(mult, sigma_x, sigma_y, opacity)
    if half_x == 0.0:
        return 0
    tiles_x = np.ceil((2 * half_x) / tile) + 1
    tiles_y = np.ceil((2 * half_y) / tile) + 1
    return int(tiles_x * tiles_y)


# --- 5. Tách learning rate cho SH ------------------------------------------------
def steps_to_converge(lr, target=1.0, tol=0.05, max_steps=20000):
    weight = 0.0
    for step in range(1, max_steps + 1):
        weight += lr * (target - weight)
        if abs(target - weight) < tol:
            return step
    return max_steps


# --- 6. Quỹ đạo số Gaussian ------------------------------------------------------
def gaussian_trajectory(iters, spawn_rate, prune_rate, final_prune_frac=0.0, start=100_000):
    count, trajectory = start, []
    for iteration in iters:
        if iteration < 15000:
            count = count * (1 + spawn_rate) * (1 - prune_rate)
        if final_prune_frac and iteration >= 15000 and iteration % 3000 == 0:
            count *= (1 - final_prune_frac)
        trajectory.append(count)
    return np.array(trajectory)


def main():
    print("== 1. mặt nạ pixel sai trên một góc nhìn ==")
    error_hat, mask = error_mask()
    print(mask, f"\n{mask.sum()} pixel sai / {mask.size}")

    footprints = {
        0: [(r, c) for r in range(0, 3) for c in range(0, 3)],
        1: [(r, c) for r in range(0, 2) for c in range(0, 5)],
        2: [(r, c) for r in range(3, 6) for c in range(3, 8)],
        3: [(0, 0), (5, 7)],
    }
    counts_single = accum_metric_counts(mask, footprints)
    for index, count in enumerate(counts_single):
        print(f"  gaussian {index}: phủ {len(footprints[index]):2d} pixel, {count} pixel sai")

    print("\n== 2. gộp 3 góc nhìn ==")
    counts = np.array([[8, 6, 7], [12, 0, 0], [1, 0, 1]], dtype=float)
    importance, raw, pruning = merge_views(counts, np.array([0.20, 0.05, 0.10]))
    for name, imp, r, p in zip("ABC", importance, raw, pruning):
        print(f"  G_{name}: importance={imp} raw={r:.2f} pruning={p:.3f}")
    print("  densify (importance>5):", [f"G_{n}" for n, i in zip("ABC", importance) if i > 5])

    print("\n== 3. điều kiện AND khi densify ==")
    n = 12
    clone_3dgs, split_3dgs, clone_fast, split_fast = densify_masks(
        rng.uniform(0, 0.0006, n), rng.uniform(0, 0.0030, n),
        rng.uniform(0, 0.004, n), rng.integers(0, 12, n))
    print(f"  3dgs densify {clone_3dgs.sum() + split_3dgs.sum()} điểm,"
          f" fastgs-lite densify {clone_fast.sum() + split_fast.sum()} điểm")
    removed = final_prune_fastgs(rng.uniform(0, 1, n), rng.uniform(0, 1, n))
    print(f"  final prune xoá {removed.sum()} / {n}")

    print("\n== 4. compact box (--mult) ==")
    box_3dgs = tiles_touched_3dgs()
    print(f"  3dgs (hộp 3-sigma):    ~{box_3dgs:3d} tile/splat")
    for mult in (1.0, 0.7, 0.5):
        half_x, _, t = compact_box_halfwidths(mult)
        count = tiles_touched(mult)
        print(f"  mult={mult}: t={t:5.2f}  bán kính {half_x / 22.0:.2f}·sigma  "
              f"~{count:3d} tile/splat ({count / box_3dgs:.0%} so với 3dgs)")
    print("  luu y: day chi la HOP BAO. mult=1 cho 3.33 sigma, rong hon hop 3-sigma cua 3dgs;")
    print("  tiet kiem that den tu (a) hop co theo opacity va (b) processTiles cat theo ellipse.")
    print("  canh co theo sqrt(mult) => dien tich hop co theo mult; hop phu thuoc ca opacity:")
    for opacity in (1.0, 0.5, 0.1, 0.01):
        half_x, _, t = compact_box_halfwidths(0.5, opacity=opacity)
        print(f"    mult=0.5, opacity={opacity:<5}: t={t:5.2f} -> bán kính {half_x / 22.0:.2f}·sigma")

    print("\n== 5. highfeature_lr (chú ý phép chia 20 ở gaussian_model.py:205) ==")
    print(f"  lowfeature_lr=0.0025 -> lr thực 0.00250: "
          f"~{steps_to_converge(0.0025):5d} bước tới 95%, cập nhật mỗi vòng")
    for lr in (0.005, 0.02, 0.04):
        eff = lr / 20.0
        print(f"  highfeature_lr={lr:<6} -> lr thực {eff:<7.5f}: "
              f"~{steps_to_converge(eff):5d} bước tới 95%, cập nhật mỗi 16 vòng (<=15k)")

    print("\n== 6. quỹ đạo số Gaussian ==")
    iters = np.arange(0, 30001, 500)
    n_3dgs = gaussian_trajectory(iters, 0.06, 0.01)
    n_fast = gaussian_trajectory(iters, 0.02, 0.015, final_prune_frac=0.08)
    print(f"  cuối cùng: 3dgs {n_3dgs[-1]:,.0f} điểm | fastgs-lite {n_fast[-1]:,.0f} điểm"
          f" ({n_fast[-1] / n_3dgs[-1]:.0%})")


if __name__ == "__main__":
    main()
