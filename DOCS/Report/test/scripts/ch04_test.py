"""
Bài test số chương 4 — Differentiable Tile Rasterizer (cảnh đồ chơi 00-scene.md).
Chỉ dùng numpy. Chép đúng thứ tự phép toán của:

  - submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu
        preprocessCUDA (ndc2Pix, computeCov2D, conic)          -> "lấy từ chương 3"
        renderCUDA (power, 3 cửa loại, alpha-blending, final_T, n_contrib, max_contrib)
  - cuda_rasterizer/auxiliary.h
        duplicateToTilesTouched / processTiles / computeEllipseIntersection (AccuTile)
        key = (tile_id << 32) | bit_cast_u32(depth)
  - cuda_rasterizer/rasterizer_impl.cu
        InclusiveSum (offsets), DeviceRadixSort::SortPairs, identifyTileRanges,
        perTileBucketCount  (#bucket = (|G_T| + 31) / 32)

Chạy:  python DOCS/Report/test/scripts/ch04_test.py
Sinh:  DOCS/Report/test/scripts/ch04_render_cam1.npy      (ảnh render, alpha = 0.1)
       DOCS/Report/test/scripts/ch04_render_cam1.ppm      (PPM P3 viết tay)
       DOCS/Report/test/scripts/ch04_render_cam1_gt.npy   (ảnh "GT", alpha = 0.9)
       DOCS/Report/test/scripts/ch04_render_cam1_gt.ppm
       DOCS/Report/test/scripts/ch04_aux_cam1.npz         (final_T, n_contrib, ranges, ... cho chương 6)
       DOCS/Report/test/04-test.md                        (báo cáo, số lấy thẳng từ script)
"""
import math
import os
import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=140)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "..", "04-test.md")

# --------------------------------------------------------------------------
# Hằng số (chương 0) + cảnh (00-scene.md)
# --------------------------------------------------------------------------
BLOCK_X = BLOCK_Y = 16
LOWPASS = 0.3
CLAMP_FACTOR = 1.3
ALPHA_MIN = 1.0 / 255.0
T_STOP = 1e-4
MULT = 0.5

W, H = 48, 32
TAN_X, TAN_Y = 0.6, 0.4
FX = W / (2 * TAN_X)          # 40
FY = H / (2 * TAN_Y)          # 40
GRID_X = (W + BLOCK_X - 1) // BLOCK_X   # 3
GRID_Y = (H + BLOCK_Y - 1) // BLOCK_Y   # 2
NUM_TILES = GRID_X * GRID_Y             # 6

points = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.3, 0.5],
    [-0.4, -0.2, 1.0],
    [0.3, -0.5, 0.2],
], dtype=np.float64)
colors = np.array([
    [0.8, 0.2, 0.2],
    [0.2, 0.7, 0.3],
    [0.1, 0.3, 0.9],
    [0.5, 0.5, 0.5],
], dtype=np.float64)
cam_centers = np.array([
    [0.0, 0.0, -4.0],
    [1.5, 0.0, -4.0],
    [-1.5, 0.5, -4.0],
], dtype=np.float64)
N = points.shape[0]
BG = np.array([1.0, 1.0, 1.0])   # nền TRẮNG

LOG = []          # dòng in ra stdout + dùng cho markdown


def P(s=""):
    print(s)
    LOG.append(s)


def f4(x):
    return f"{x:.4g}"


def vec(v, sig=4):
    return "(" + ", ".join(f"{a:.{sig}g}" for a in np.atleast_1d(v)) + ")"


# ==========================================================================
# LẤY TỪ CHƯƠNG 1: khởi tạo (mu = p, s = sqrt(mean d^2 tới 3 láng giềng), R = I, alpha)
# ==========================================================================
def init_gaussians(alpha_value):
    mu = points.copy()
    s = np.zeros(N)
    for k in range(N):
        d2 = np.sum((points - points[k]) ** 2, axis=1)
        d2 = np.sort(d2[np.arange(N) != k])[:3]
        s[k] = math.sqrt(max(d2.mean(), 1e-7))
    cov3d = np.array([s_k ** 2 * np.eye(3) for s_k in s])     # R = I -> Sigma = s^2 I
    alpha = np.full(N, alpha_value)
    return mu, s, cov3d, alpha, colors.copy()


# ==========================================================================
# LẤY TỪ CHƯƠNG 3: projection cho camera v (forward.cu preprocessCUDA / computeCov2D)
# ==========================================================================
def ndc2Pix(v, S):                       # auxiliary.h:45
    return ((v + 1.0) * S - 1.0) * 0.5


def project(mu, cov3d, cam_c):
    """Trả về per-Gaussian: t (camera), depth, mu' (pixel), Sigma' (2x2), conic (A,B,C)."""
    out = []
    for i in range(N):
        t = mu[i] - cam_c                                   # W_v: R = I, t = mu - c_v
        depth = t[2]
        if depth <= 0.2:
            out.append(None)
            continue
        # chiếu tâm: p_proj = (t_x / tan_x, t_y / tan_y) / (t_z + 1e-7)
        p_w = depth + 1e-7
        px = (t[0] / TAN_X) / p_w
        py = (t[1] / TAN_Y) / p_w
        mu2 = np.array([ndc2Pix(px, W), ndc2Pix(py, H)])
        # computeCov2D: clamp rồi Jacobian
        limx, limy = CLAMP_FACTOR * TAN_X, CLAMP_FACTOR * TAN_Y
        txtz, tytz = t[0] / t[2], t[1] / t[2]
        tx_hat = min(limx, max(-limx, txtz)) * t[2]
        ty_hat = min(limy, max(-limy, tytz)) * t[2]
        J = np.array([
            [FX / t[2], 0.0, -(FX * tx_hat) / (t[2] * t[2])],
            [0.0, FY / t[2], -(FY * ty_hat) / (t[2] * t[2])],
            [0.0, 0.0, 0.0],
        ])
        Wm = np.eye(3)                                       # R = I
        cov = J @ Wm @ cov3d[i] @ Wm.T @ J.T
        S11 = cov[0, 0] + LOWPASS
        S12 = cov[0, 1]
        S22 = cov[1, 1] + LOWPASS
        det = S11 * S22 - S12 * S12
        A, B, C = S22 / det, -S12 / det, S11 / det
        out.append(dict(t=t, depth=depth, p_proj=(px, py), mu2=mu2, J=J,
                        S=(S11, S12, S22), det=det, conic=(A, B, C),
                        clamped=(abs(txtz) > limx or abs(tytz) > limy)))
    return out


# ==========================================================================
# LẤY TỪ CHƯƠNG 3: compact box + AccuTile (auxiliary.h duplicateToTilesTouched / processTiles)
# ==========================================================================
def computeEllipseIntersection(con_o, disc, t, p, isY, coord):
    A, B, C, _ = con_o
    p_u = p[1] if isY else p[0]
    p_v = p[0] if isY else p[1]
    coeff = A if isY else C
    h = coord - p_u
    sqrt_term = math.sqrt(disc * h * h + t * coeff)
    return ((-B * h - sqrt_term) / coeff + p_v,
            (-B * h + sqrt_term) / coeff + p_v)


def duplicateToTilesTouched(p, con_o):
    """Trả về (t, half_x, half_y, bbox_min, bbox_max, rect_min, rect_max, tiles[list of tile_id])."""
    A, B, C, o = con_o
    disc = B * B - A * C
    if A <= 0 or C <= 0 or disc >= 0:
        return None
    t = 2.0 * math.log(o * 255.0)
    t = MULT * t
    x_term = math.sqrt(-(B * B * t) / (disc * A))
    x_term = x_term if B < 0 else -x_term
    y_term = math.sqrt(-(B * B * t) / (disc * C))
    y_term = y_term if B < 0 else -y_term
    bbox_argmin = (p[1] - y_term, p[0] - x_term)      # (y, x) như code
    bbox_argmax = (p[1] + y_term, p[0] + x_term)
    bbox_min = (computeEllipseIntersection(con_o, disc, t, p, True, bbox_argmin[0])[0],
                computeEllipseIntersection(con_o, disc, t, p, False, bbox_argmin[1])[0])
    bbox_max = (computeEllipseIntersection(con_o, disc, t, p, True, bbox_argmax[0])[1],
                computeEllipseIntersection(con_o, disc, t, p, False, bbox_argmax[1])[1])
    rect_min = (max(0, min(GRID_X, int(bbox_min[0] / BLOCK_X))),
                max(0, min(GRID_Y, int(bbox_min[1] / BLOCK_Y))))
    rect_max = (max(0, min(GRID_X, int(bbox_max[0] / BLOCK_X + 1))),
                max(0, min(GRID_Y, int(bbox_max[1] / BLOCK_Y + 1))))
    y_span = rect_max[1] - rect_min[1]
    x_span = rect_max[0] - rect_min[0]
    half_x = bbox_max[0] - p[0]
    half_y = bbox_max[1] - p[1]
    if y_span * x_span == 0:
        return dict(t=t, half=(half_x, half_y), bbox_min=bbox_min, bbox_max=bbox_max,
                    rect_min=rect_min, rect_max=rect_max, tiles=[], isY=None)
    isY = y_span < x_span
    tiles = processTiles(con_o, disc, t, p, bbox_min, bbox_max, bbox_argmin, bbox_argmax,
                         rect_min, rect_max, isY)
    return dict(t=t, half=(half_x, half_y), bbox_min=bbox_min, bbox_max=bbox_max,
                rect_min=rect_min, rect_max=rect_max, tiles=tiles, isY=isY,
                rect_count=x_span * y_span)


def processTiles(con_o, disc, t, p, bbox_min, bbox_max, bbox_argmin, bbox_argmax,
                 rect_min, rect_max, isY):
    BLOCK_U = BLOCK_Y if isY else BLOCK_X
    BLOCK_V = BLOCK_X if isY else BLOCK_Y
    if isY:
        rect_min = (rect_min[1], rect_min[0]); rect_max = (rect_max[1], rect_max[0])
        bbox_min = (bbox_min[1], bbox_min[0]); bbox_max = (bbox_max[1], bbox_max[0])
        bbox_argmin = (bbox_argmin[1], bbox_argmin[0]); bbox_argmax = (bbox_argmax[1], bbox_argmax[0])
    tiles = []
    intersect_max_line = (bbox_max[1], bbox_min[1])
    min_line = rect_min[0] * BLOCK_U
    if bbox_min[0] <= min_line:
        intersect_min_line = computeEllipseIntersection(con_o, disc, t, p, isY, rect_min[0] * BLOCK_U)
    else:
        intersect_min_line = intersect_max_line
    for u in range(rect_min[0], rect_max[0]):
        max_line = min_line + BLOCK_U
        if max_line <= bbox_max[0]:
            intersect_max_line = computeEllipseIntersection(con_o, disc, t, p, isY, max_line)
        if min_line <= bbox_argmin[1] < max_line:
            ellipse_min = bbox_min[1]
        else:
            ellipse_min = min(intersect_min_line[0], intersect_max_line[0])
        if min_line <= bbox_argmax[1] < max_line:
            ellipse_max = bbox_max[1]
        else:
            ellipse_max = max(intersect_min_line[1], intersect_max_line[1])
        min_tile_v = max(rect_min[1], min(rect_max[1], int(ellipse_min / BLOCK_V)))
        max_tile_v = min(rect_max[1], max(rect_min[1], int(ellipse_max / BLOCK_V + 1)))
        for v in range(min_tile_v, max_tile_v):
            tile_id = (u * GRID_X + v) if isY else (v * GRID_X + u)
            tiles.append(tile_id)
        intersect_min_line = intersect_max_line
        min_line = max_line
    return tiles


# ==========================================================================
# CHƯƠNG 4 — 4.2 khoá, 4.3 sort + ranges, 4.4-4.5 render, 4.6 dữ liệu backward
# ==========================================================================
def bit_cast_u32(depth):
    return int(np.float32(depth).view(np.uint32))


def build_keys(proj, tiles_per_g):
    """4.2: offset_i = prefix sum K_i ; key = (tile << 32) | bit_cast(depth)."""
    K = np.array([len(tiles_per_g[i]) if tiles_per_g[i] is not None else 0 for i in range(N)])
    offsets = np.cumsum(K)                       # InclusiveSum như cub
    keys, vals, rows = [], [], []
    for i in range(N):
        if K[i] == 0:
            continue
        off = 0 if i == 0 else offsets[i - 1]
        depth = proj[i]["depth"]
        d32 = bit_cast_u32(depth)
        for tile in tiles_per_g[i]:
            key = (tile << 32) | d32
            keys.append(key); vals.append(i)
            rows.append((off, tile, i, float(np.float32(depth)), d32, key))
            off += 1
    return K, offsets, np.array(keys, dtype=np.uint64), np.array(vals, dtype=np.uint32), rows


def sort_and_ranges(keys, vals):
    """4.3: radix sort (ổn định) rồi identifyTileRanges."""
    order = np.argsort(keys, kind="stable")
    keys_s, vals_s = keys[order], vals[order]
    L = len(keys_s)
    ranges = np.zeros((NUM_TILES, 2), dtype=np.int64)
    for idx in range(L):
        currtile = int(keys_s[idx] >> np.uint64(32))
        if idx == 0:
            ranges[currtile, 0] = 0
        else:
            prevtile = int(keys_s[idx - 1] >> np.uint64(32))
            if currtile != prevtile:
                ranges[prevtile, 1] = idx
                ranges[currtile, 0] = idx
        if idx == L - 1:
            ranges[currtile, 1] = L
    return order, keys_s, vals_s, ranges


def render_pixel(px, py, tile_id, ranges, point_list, proj, alpha, cols, trace=False):
    """renderCUDA cho một pixel; trả về (C_out, T_final, last_contributor, contribs, done, rows)."""
    T = 1.0
    C = np.zeros(3)
    contributor = 0
    last_contributor = 0
    contribs = 0
    done = False
    rows = []
    pixf = (float(px), float(py))
    for k in range(ranges[tile_id, 0], ranges[tile_id, 1]):
        if done:
            break
        gid = int(point_list[k])
        contributor += 1
        xy = proj[gid]["mu2"]
        d = (xy[0] - pixf[0], xy[1] - pixf[1])          # d = mu' - x  (code); Delta = x - mu' = -d
        A, B, Cc = proj[gid]["conic"]
        power = -0.5 * (A * d[0] * d[0] + Cc * d[1] * d[1]) - B * d[0] * d[1]
        row = dict(n=contributor, gid=gid, c=cols[gid], delta=(-d[0], -d[1]), power=power,
                   G=math.exp(power) if power <= 0 else float("nan"), T=T)
        if power > 0.0:
            row["status"] = "power>0 → bỏ"; rows.append(row); continue
        a = min(0.99, alpha[gid] * math.exp(power))
        row["alpha"] = a
        if a < ALPHA_MIN:
            row["status"] = "α<1/255 → bỏ"; row["contrib"] = np.zeros(3); rows.append(row); continue
        test_T = T * (1 - a)
        if test_T < T_STOP:
            row["status"] = "T(1−α)<1e-4 → dừng"; row["contrib"] = np.zeros(3); rows.append(row)
            done = True
            continue
        contrib = cols[gid] * a * T
        C += contrib
        T = test_T
        last_contributor = contributor
        contribs += 1
        row["status"] = "cộng"; row["contrib"] = contrib; row["T_next"] = T
        rows.append(row)
    C_out = C + T * BG
    return C_out, T, last_contributor, contribs, done, rows


def render_image(ranges, point_list, proj, alpha, cols):
    img = np.zeros((H, W, 3))
    final_T = np.zeros((H, W))
    n_contrib = np.zeros((H, W), dtype=np.int64)
    n_real = np.zeros((H, W), dtype=np.int64)
    early = np.zeros((H, W), dtype=bool)
    max_contrib = np.zeros(NUM_TILES, dtype=np.int64)
    for py in range(H):
        for px in range(W):
            tile_id = (py // BLOCK_Y) * GRID_X + (px // BLOCK_X)
            C, T, last, cnt, done, _ = render_pixel(px, py, tile_id, ranges, point_list, proj, alpha, cols)
            img[py, px] = C; final_T[py, px] = T; n_contrib[py, px] = last; n_real[py, px] = cnt
            early[py, px] = done
            max_contrib[tile_id] = max(max_contrib[tile_id], last)
    return img, final_T, n_contrib, n_real, early, max_contrib


def write_ppm(path, img):
    q = np.clip(np.round(img * 255), 0, 255).astype(int)
    with open(path, "w") as f:
        f.write(f"P3\n{W} {H}\n255\n")
        for py in range(H):
            f.write(" ".join(f"{q[py, px, 0]} {q[py, px, 1]} {q[py, px, 2]}" for px in range(W)) + "\n")


def ascii_art(img):
    chars = "@%#*+=-:. "     # tối → sáng (nền trắng = ' ')
    lum = img.mean(axis=2)
    lines = []
    for py in range(H):
        lines.append("".join(chars[min(9, int(lum[py, px] * 9.999))] for px in range(W)))
    return lines


def ascii_int(arr):
    return ["".join(str(min(9, int(v))) for v in row) for row in arr]


# ==========================================================================
# Chạy toàn bộ pipeline cho một giá trị alpha
# ==========================================================================
def run(alpha_value):
    mu, s, cov3d, alpha, cols = init_gaussians(alpha_value)
    proj = project(mu, cov3d, cam_centers[0])
    tiles = []
    for i in range(N):
        if proj[i] is None:
            tiles.append(None); continue
        con_o = (*proj[i]["conic"], alpha[i])
        tiles.append(duplicateToTilesTouched(proj[i]["mu2"], con_o))
    tiles_per_g = [t["tiles"] if t is not None else [] for t in tiles]
    K, offsets, keys, vals, rows = build_keys(proj, tiles_per_g)
    order, keys_s, vals_s, ranges = sort_and_ranges(keys, vals)
    img, final_T, n_contrib, n_real, early, max_contrib = render_image(ranges, vals_s, proj, alpha, cols)
    return dict(mu=mu, s=s, cov3d=cov3d, alpha=alpha, cols=cols, proj=proj, tiles=tiles,
                tiles_per_g=tiles_per_g, K=K, offsets=offsets, keys=keys, vals=vals, rows=rows,
                order=order, keys_s=keys_s, vals_s=vals_s, ranges=ranges, img=img, final_T=final_T,
                n_contrib=n_contrib, n_real=n_real, early=early, max_contrib=max_contrib)


# ==========================================================================
# MAIN — in + ghi markdown
# ==========================================================================
def main():
    R = run(0.1)
    mu, s, alpha, cols, proj, tiles = R["mu"], R["s"], R["alpha"], R["cols"], R["proj"], R["tiles"]

    P("# Chương 4 — Test số: Differentiable Tile Rasterizer (camera 1, nền trắng)")
    P("")
    P("> Cảnh: [`00-scene.md`](00-scene.md). Script: [`scripts/ch04_test.py`](scripts/ch04_test.py) — mọi số dưới đây")
    P("> do script in ra (file markdown này được script sinh trực tiếp, không chép tay).")
    P("> Thứ tự phép toán chép từ `forward.cu:renderCUDA`, `auxiliary.h:duplicateToTilesTouched/processTiles`,")
    P("> `rasterizer_impl.cu:duplicateWithKeys/identifyTileRanges/perTileBucketCount`.")
    P(">")
    P("> Chỉ dùng camera 1: $c_1=(0,0,-4)$, $R=I$ ⇒ $t_i=\\mu_i-c_1$, $f_x=f_y=40$, ảnh $48\\times32$ = lưới $3\\times2$ tile.")
    P("")

    # ---------------------------------------------------------------- 4.0
    P("## 4.0 — Đầu vào của khối (lấy từ chương 1 và chương 3, tính lại trong script)")
    P("")
    P("### Khởi tạo (lấy từ chương 1)")
    P("")
    P("$\\mu_i=p_i$, $s_i=\\sqrt{\\text{mean}(d^2\\text{ tới 3 láng giềng})}$, $R_i=I$ ⇒ $\\Sigma_i=s_i^2I$, $\\alpha_i=0.1$, $c_i=$ màu SfM.")
    P("")
    P("| $i$ | $\\mu_i$ | $s_i$ | $s_i^2$ | $\\alpha_i$ | $c_i$ |")
    P("|---|---|---|---|---|---|")
    for i in range(N):
        P(f"| {i+1} | {vec(mu[i])} | {f4(s[i])} | {f4(s[i]**2)} | {alpha[i]} | {vec(cols[i])} |")
    P("")
    P("### Projection camera 1 (lấy từ chương 3)")
    P("")
    P("$t_i=\\mu_i-c_1$, depth $=t_z$; $p^{proj}=(t_x/0.6,\\ t_y/0.4)/(t_z+10^{-7})$; "
      "$\\mu'=\\bigl(\\tfrac{(p_x+1)\\cdot48-1}{2},\\tfrac{(p_y+1)\\cdot32-1}{2}\\bigr)$ (`ndc2Pix`).")
    P("")
    P("| $i$ | $t_i$ | depth | $p^{proj}$ | $\\mu'_i$ (px) | clamp $1.3\\tan$ kích hoạt? |")
    P("|---|---|---|---|---|---|")
    for i in range(N):
        p = proj[i]
        P(f"| {i+1} | {vec(p['t'])} | {f4(p['depth'])} | {vec(p['p_proj'])} | {vec(p['mu2'])} | "
          f"{'có' if p['clamped'] else 'không'} ($|t_x/t_z|$={f4(abs(p['t'][0]/p['t'][2]))}≤0.78, $|t_y/t_z|$={f4(abs(p['t'][1]/p['t'][2]))}≤0.52) |")
    P("")
    P("$J=\\begin{pmatrix}f_x/t_z&0&-f_x\\hat t_x/t_z^2\\\\0&f_y/t_z&-f_y\\hat t_y/t_z^2\\\\0&0&0\\end{pmatrix}$, "
      "$\\Sigma'=J\\,\\Sigma_i\\,J^\\top+0.3I$ (vì $W$ có $R=I$), $\\det=\\Sigma'_{11}\\Sigma'_{22}-\\Sigma'^2_{12}$, "
      "$M=(A,B,C)=(\\Sigma'_{22},-\\Sigma'_{12},\\Sigma'_{11})/\\det$.")
    P("")
    P("| $i$ | $J$ (2 hàng đầu) | $\\Sigma'_{11}$ | $\\Sigma'_{12}$ | $\\Sigma'_{22}$ | $\\det$ | $A$ | $B$ | $C$ |")
    P("|---|---|---|---|---|---|---|---|---|")
    for i in range(N):
        p = proj[i]; J = p["J"]; S11, S12, S22 = p["S"]; A, B, C = p["conic"]
        P(f"| {i+1} | [{f4(J[0,0])}, 0, {f4(J[0,2])}; 0, {f4(J[1,1])}, {f4(J[1,2])}] | {f4(S11)} | {f4(S12)} | {f4(S22)} | {f4(p['det'])} | {f4(A)} | {f4(B)} | {f4(C)} |")
    P("")
    P("Ví dụ thay số cho $i=1$: $t=(0,0,4)$, $J_{11}=40/4=10$, $\\Sigma'_{11}=10^2\\cdot s_1^2+0.3="
      f"{f4(100*s[0]**2)}+0.3={f4(proj[0]['S'][0])}$, $\\Sigma'_{{12}}=0$ (vì $\\hat t_x=\\hat t_y=0$), "
      f"$A=1/{f4(proj[0]['S'][0])}={f4(proj[0]['conic'][0])}$.")
    P("")
    P("### Compact box + lọc tile theo ellipse (lấy từ chương 3, `duplicateToTilesTouched`)")
    P("")
    P("$t_i=\\texttt{mult}\\cdot2\\ln(255\\alpha_i)=0.5\\cdot2\\ln(25.5)=" + f4(tiles[0]["t"]) + "$ (giống nhau cho mọi $i$ vì $\\alpha=0.1$). "
      "$\\text{half}_x=\\sqrt{t\\,\\Sigma'_{11}}$, $\\text{half}_y=\\sqrt{t\\,\\Sigma'_{22}}$ "
      "(code tính qua `computeEllipseIntersection` tại tiếp điểm — cùng giá trị). Tile id $=t_y\\cdot3+t_x$.")
    P("")
    P("| $i$ | $t_i$ | $\\text{half}_x$ | $\\text{half}_y$ | Box $[x_{min},x_{max}]\\times[y_{min},y_{max}]$ | rect tile (min..max) | #tile hộp | $\\mathcal K_i$ (AccuTile) | $K_i$ |")
    P("|---|---|---|---|---|---|---|---|---|")
    for i in range(N):
        tb = tiles[i]; S11, S12, S22 = proj[i]["S"]
        hx, hy = math.sqrt(tb["t"] * S11), math.sqrt(tb["t"] * S22)
        P(f"| {i+1} | {f4(tb['t'])} | {f4(hx)} | {f4(hy)} | [{f4(tb['bbox_min'][0])}, {f4(tb['bbox_max'][0])}]×[{f4(tb['bbox_min'][1])}, {f4(tb['bbox_max'][1])}] | "
          f"x:{tb['rect_min'][0]}..{tb['rect_max'][0]-1}, y:{tb['rect_min'][1]}..{tb['rect_max'][1]-1} | {tb.get('rect_count', 0)} | {tb['tiles']} | {len(tb['tiles'])} |")
    P("")
    Ktot = int(R["K"].sum())
    P(f"$P=\\sum_iK_i={'+'.join(str(int(k)) for k in R['K'])}=\\mathbf{{{Ktot}}}$ cặp (tile, Gaussian) = `num_rendered`.")
    P("")
    P("Nhận xét: với cảnh này $\\text{half}_{x,y}\\approx15$ px trên ảnh $48\\times32$ nên hộp của mọi Gaussian phủ cả 6 tile "
      "và bước lọc ellipse (AccuTile) không loại thêm tile nào ($K_i=K^{\\text{box}}_i=6$). "
      "Trong `processTiles`, `isY = y_span < x_span` = (2 < 3) = True → quét theo lát $y$.")
    P("")

    # ---------------------------------------------------------------- 4.2
    P("## 4.2 — Prefix sum và sinh khoá")
    P("")
    P("![Khoá 64-bit và thứ tự sau sort](figures/ch04_keys.png)")
    P("")
    P("*Hình: 24 cặp (tile, Gaussian) xếp theo 6 tile; mỗi ô ghi khoá rút gọn (tile_id | bit_cast(depth)); sau sort mọi tile đều có cùng thứ tự [G1, G4, G2, G3] theo độ sâu tăng dần.*")
    P("")
    P("![bit_cast float→uint32 giữ đúng thứ tự khi depth dương](figures/ch04_bitcast.png)")
    P("")
    P("*Hình: giá trị uint32 tăng đơn điệu theo depth dương (4.0 → 0x40800000 lớn hơn 3.5 → 0x40600000); điểm phản chứng depth âm (−1.0) lại cho uint32 lớn hơn depth dương nhỏ (2.0) — lý do phải cull t_z ≤ 0.2.*")
    P("")
    P("$\\text{offset}_i=\\sum_{j<i}K_j$ (code: `InclusiveSum` rồi lấy `offsets[idx-1]`, $i=1$ lấy 0):")
    P("")
    P("| $i$ | $K_i$ | InclusiveSum | $\\text{offset}_i$ (vị trí ghi đầu tiên) |")
    P("|---|---|---|---|")
    for i in range(N):
        off = 0 if i == 0 else int(R["offsets"][i - 1])
        P(f"| {i+1} | {int(R['K'][i])} | {int(R['offsets'][i])} | {off} |")
    P("")
    P("$\\text{key}(T,i)=(\\text{id}(T)\\ll32)\\ |\\ \\text{bit\\_cast}_{u32}(\\text{float32}(\\text{depth}_i))$ — "
      "numpy: `np.float32(depth).view(np.uint32)`. Bảng chưa sort (thứ tự ghi vào buffer):")
    P("")
    P("| vị trí | tile | $i$ | depth (float32) | bit_cast (hex) | key (hex 64-bit) |")
    P("|---|---|---|---|---|---|")
    for (off, tile, i, depth, d32, key) in R["rows"]:
        P(f"| {off} | {tile} | {i+1} | {depth:.6g} | 0x{d32:08X} | 0x{key:016X} |")
    P("")
    P("Kiểm tra thay số cho $i=1$: depth $=4.0$ → float32 = sign 0, exp $=129$ (0x81), mantissa 0 → "
      f"0x{bit_cast_u32(4.0):08X}; tile 1 → key = (1≪32)|0x{bit_cast_u32(4.0):08X} = 0x{(1<<32)|bit_cast_u32(4.0):016X}.")
    P("")

    # ---------------------------------------------------------------- 4.3
    P("## 4.3 — Radix sort và `identifyTileRanges`")
    P("")
    nbits = 32 + math.ceil(math.log2(NUM_TILES))
    P(f"Số bit sort: $32+\\lceil\\log_2 6\\rceil=32+3={nbits}$ (`getHigherMsb(6)` = 3). "
      "Ở đây dùng `np.argsort(kind='stable')` trên khoá `uint64` — cùng kết quả vì radix sort ổn định.")
    P("")
    P("Sau sort (`point_list_keys`, `point_list`):")
    P("")
    P("| idx | key (hex) | tile = key≫32 | $i$ = point_list[idx] | depth |")
    P("|---|---|---|---|---|")
    for idx in range(len(R["keys_s"])):
        key = int(R["keys_s"][idx]); gid = int(R["vals_s"][idx])
        P(f"| {idx} | 0x{key:016X} | {key >> 32} | {gid+1} | {f4(proj[gid]['depth'])} |")
    P("")
    P("`identifyTileRanges`: tại idx mà tile đổi so với idx−1 → `ranges[prev].y = idx`, `ranges[curr].x = idx`; idx=0 → `ranges[t].x=0`; idx=L−1 → `ranges[t].y=L`.")
    P("")
    P("| tile $T$ | $[\\text{start},\\text{end})$ | $|\\mathcal G_T|$ | $\\mathcal G_T$ theo depth tăng dần |")
    P("|---|---|---|---|")
    for T in range(NUM_TILES):
        a, b = R["ranges"][T]
        G = [int(R["vals_s"][k]) + 1 for k in range(a, b)]
        P(f"| {T} | [{a}, {b}) | {b-a} | {G} (depth {[f4(proj[g-1]['depth']) for g in G]}) |")
    P("")
    P("**Minh hoạ bit_cast giữ thứ tự khi depth dương:**")
    P("")
    for d in (3.5, 4.0):
        P(f"- depth $={d}$ → float32 bits = 0x{bit_cast_u32(d):08X} = {bit_cast_u32(d)} (uint32)")
    P(f"  ⇒ $3.5<4.0$ và $0x{bit_cast_u32(3.5):08X}<0x{bit_cast_u32(4.0):08X}$ ✓ (cùng dấu, exponent/mantissa của IEEE-754 tăng đơn điệu).")
    P("")
    P("**Phản chứng với depth âm** (không bao giờ xảy ra nhờ cull $t_z>0.2$ ở chương 3.2):")
    P("")
    for d in (-1.0, 2.0):
        P(f"- depth $={d}$ → 0x{bit_cast_u32(d):08X} = {bit_cast_u32(d)} (uint32)")
    P(f"  ⇒ $-1.0<2.0$ nhưng $0x{bit_cast_u32(-1.0):08X}>0x{bit_cast_u32(2.0):08X}$ (bit dấu là bit cao nhất) — "
      "sort uint32 sẽ xếp $-1.0$ **sau** $2.0$: sai thứ tự front-to-back. Cull là điều kiện đúng đắn của khoá.")
    P("")

    # ---------------------------------------------------------------- 4.4-4.5
    img, final_T, n_contrib, n_real, early = R["img"], R["final_T"], R["n_contrib"], R["n_real"], R["early"]
    np.save(os.path.join(HERE, "ch04_render_cam1.npy"), img.astype(np.float32))
    write_ppm(os.path.join(HERE, "ch04_render_cam1.ppm"), img)

    P("## 4.4 – 4.5 — Alpha tại pixel và alpha-blending front-to-back (render đầy đủ 48×32)")
    P("")
    P("![Alpha-blending front-to-back tại 2 pixel + ví dụ chương](figures/ch04_blend_pixel.png)")
    P("")
    P("*Hình: transmittance T_n giảm dần và đóng góp từng Gaussian cho pixel (24,16) và (25,15); subplot tái hiện đúng ví dụ 3 Gaussian của chương 4.5, ra (0.62, 0.32, 0.30).*")
    P("")
    P("![Ba cửa loại dọc theo một hàng pixel](figures/ch04_gates.png)")
    P("")
    P("*Hình: quét ngang hàng y=16 qua tâm G1 — đường α_n(x) của 4 Gaussian so với ngưỡng 1/255, cùng T_final(x) và n_contrib(x) dọc hàng đó.*")
    P("")
    P("Với pixel $x=(u,v)$ (tâm pixel = toạ độ nguyên, code: `pixf = {pix.x, pix.y}`), tile $T=(v\\!\\div\\!16)\\cdot3+(u\\!\\div\\!16)$, "
      "duyệt $\\mathcal G_T$ theo `point_list[ranges[T].x .. ranges[T].y)`. Code tính `d = μ' − pixf` (= $-\\Delta$), dạng toàn phương đối xứng nên")
    P("")
    P("$$\\text{power}=-\\tfrac12(A\\,d_x^2+C\\,d_y^2)-B\\,d_xd_y=-\\tfrac12\\Delta^\\top M\\Delta,\\qquad "
      "\\alpha_n(x)=\\min(0.99,\\ \\alpha_n e^{\\text{power}})$$")
    P("")
    P("Ba cửa: `power>0 → continue`; `alpha<1/255 → continue`; `T(1−alpha)<1e-4 → done`. "
      "Sau đó $C\\mathrel{+}=c_n\\alpha_nT$, $T\\leftarrow T(1-\\alpha_n)$, `last_contributor = contributor`. "
      "Kết thúc: `out_color = C + T·bg`, `final_T = T`, `n_contrib = last_contributor`.")
    P("")
    P("### Ảnh kết quả (độ sáng trung bình RGB, ký tự tối → sáng `@%#*+=-:. `, nền trắng = khoảng trắng)")
    P("")
    P("```")
    for line in ascii_art(img):
        P(line)
    P("```")
    P("")
    P("Lưu: `scripts/ch04_render_cam1.npy` (float32, shape (32, 48, 3), HWC, RGB ∈ [0,1]) và `scripts/ch04_render_cam1.ppm` (P3, 8-bit).")
    P("")
    P("![I_rend, I_gt, |diff| với lưới tile và pixel ví dụ](figures/ch04_render.png)")
    P("")
    P("*Hình: ba ảnh 48×32 phóng to cạnh nhau (α=0.1, α=0.9 \"GT\", và trị tuyệt đối hiệu số); lưới tile 16 px, tâm 4 Gaussian, và 3 pixel ví dụ (24,16)/(25,15)/(0,0) được đánh dấu.*")
    P("")

    # 3 example pixels
    mu1 = proj[0]["mu2"]
    pa = (int(round(mu1[0])), int(round(mu1[1])))
    # (b): pixel có nhiều Gaussian cộng thật nhất; phá hoà bằng tổng alpha_n(x) lớn nhất, loại pixel (a)
    best, pb = -1.0, None
    for py in range(H):
        for px in range(W):
            if n_real[py, px] != n_real.max() or (px, py) == pa:
                continue
            tid = (py // BLOCK_Y) * GRID_X + (px // BLOCK_X)
            _, _, _, _, _, rws = render_pixel(px, py, tid, R["ranges"], R["vals_s"], proj, alpha, cols)
            sa = sum(r.get("alpha", 0.0) for r in rws if r["status"] == "cộng")
            if sa > best:
                best, pb = sa, (px, py)
    bg_cands = np.argwhere(n_contrib == 0)
    if len(bg_cands):
        # ưu tiên góc (0,0) nếu là nền
        if n_contrib[0, 0] == 0:
            pc = (0, 0)
        else:
            pc = (int(bg_cands[0][1]), int(bg_cands[0][0]))
    else:
        pc = None

    def detail(px, py, label):
        tile_id = (py // BLOCK_Y) * GRID_X + (px // BLOCK_X)
        C, T, last, cnt, done, rows = render_pixel(px, py, tile_id, R["ranges"], R["vals_s"], proj, alpha, cols, trace=True)
        P(f"### {label}: pixel $x=({px},{py})$, tile $T={tile_id}$, $\\mathcal G_T$ = {[int(R['vals_s'][k])+1 for k in range(R['ranges'][tile_id,0], R['ranges'][tile_id,1])]}")
        P("")
        P("| $n$ | Gaussian $i$ | $c_n$ | $\\Delta=x-\\mu'_n$ | power | $G_n=e^{\\text{power}}$ | $\\alpha_n(x)=\\min(0.99,\\alpha_iG_n)$ | $T_n$ | cửa | đóng góp $c_n\\alpha_nT_n$ | $T_{n+1}$ |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            a = r.get("alpha", float("nan"))
            contrib = r.get("contrib", np.zeros(3))
            Tn1 = r.get("T_next", r["T"])
            P(f"| {r['n']} | {r['gid']+1} | {vec(r['c'])} | {vec(r['delta'])} | {f4(r['power'])} | {f4(r['G'])} | {f4(a)} | {f4(r['T'])} | {r['status']} | {vec(contrib)} | {f4(Tn1)} |")
        if not rows:
            P("| – | – | – | – | – | – | – | 1 | $\\mathcal G_T=\\varnothing$ | (0,0,0) | 1 |")
        P("")
        P(f"$T_{{final}}={f4(T)}$, `n_contrib` = {last}, số Gaussian cộng thật = {cnt}, early-termination = {done}.")
        P("")
        P(f"$C(x)=\\sum c_n\\alpha_nT_n+T_{{final}}\\cdot(1,1,1)={vec(C - T*BG)}+{f4(T)}\\cdot(1,1,1)=\\mathbf{{{vec(C)}}}$ "
          f"→ 8-bit {tuple(int(v) for v in np.clip(np.round(C*255),0,255))}.")
        P("")
        return C, T, last, cnt

    P("### Ba pixel trình bày chi tiết")
    P("")
    P(f"(a) pixel gần tâm Gaussian 1 nhất: $\\mu'_1={vec(mu1)}$ → làm tròn $({pa[0]},{pa[1]})$. "
      f"(b) pixel có nhiều Gaussian cộng thật nhất (argmax `contribs` = {int(n_real.max())}; phá hoà bằng $\\sum_n\\alpha_n(x)$ lớn nhất, loại pixel (a)): $({pb[0]},{pb[1]})$, $\\sum_n\\alpha_n(x)={f4(best)}$. "
      f"(c) pixel nền (`n_contrib`=0): $({pc[0]},{pc[1]})$." if pc else "(c) không có pixel nền.")
    P("")
    Ca = detail(pa[0], pa[1], "(a) Gần tâm Gaussian 1")
    Cb = detail(pb[0], pb[1], "(b) Chồng nhiều Gaussian nhất")
    Cc = detail(pc[0], pc[1], "(c) Pixel nền") if pc else None

    # ví dụ 3 Gaussian trong chương
    P("### Kiểm tra lại ví dụ 3 Gaussian trong chương 4.5")
    P("")
    ex_c = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    ex_a = [0.5, 0.4, 0.6]
    T = 1.0; C = np.zeros(3)
    P("| $n$ | $c_n$ | $\\alpha_n$ | $T_n$ | đóng góp | $T_{n+1}$ |")
    P("|---|---|---|---|---|---|")
    for n in range(3):
        contrib = ex_c[n] * ex_a[n] * T
        C += contrib
        Tn1 = T * (1 - ex_a[n])
        P(f"| {n+1} | {vec(ex_c[n])} | {ex_a[n]} | {f4(T)} | {vec(contrib)} | {f4(Tn1)} |")
        T = Tn1
    Cex = C + T * BG
    P("")
    P(f"$T_4={f4(T)}$ → $C={vec(C)}+{f4(T)}\\cdot(1,1,1)=\\mathbf{{{vec(Cex)}}}$ — khớp $(0.62,0.32,0.30)$ trong chương: "
      f"{'✓' if np.allclose(Cex, [0.62, 0.32, 0.30]) else '✗'}.")
    P("")

    # ---------------------------------------------------------------- 4.6
    P("## 4.6 — Dữ liệu lưu cho backward")
    P("")
    P("![final_T, n_contrib và thống kê bucket](figures/ch04_aux_maps.png)")
    P("")
    P("*Hình: bản đồ final_T và n_contrib toàn ảnh cho cả α=0.1 và α=0.9 (GT); GT có final_T nhỏ hơn nhiều (nhiều Gaussian đục hơn) nhưng vẫn không có pixel early-termination trong cảnh 4-Gaussian này.*")
    P("")
    P("`final_Ts[x]` (bản đồ $T_{final}$, in $\\lfloor 9T\\rfloor$ mỗi pixel: 9 = nền hoàn toàn, nhỏ = bị che nhiều):")
    P("")
    P("```")
    for line in ascii_int(np.floor(final_T * 9.999)):
        P(line)
    P("```")
    P("")
    P("`n_contrib[x]` (= `last_contributor`: chỉ số 1-based của Gaussian cuối cùng đóng góp trong $\\mathcal G_T$):")
    P("")
    P("```")
    for line in ascii_int(n_contrib):
        P(line)
    P("```")
    P("")
    P("| Thống kê trên 1536 pixel | min | max | mean |")
    P("|---|---|---|---|")
    P(f"| `final_Ts` | {f4(final_T.min())} | {f4(final_T.max())} | {f4(final_T.mean())} |")
    P(f"| `n_contrib` (last_contributor) | {int(n_contrib.min())} | {int(n_contrib.max())} | {f4(n_contrib.mean())} |")
    P(f"| số Gaussian cộng thật (`contribs`) | {int(n_real.min())} | {int(n_real.max())} | {f4(n_real.mean())} |")
    P(f"| pixel có `n_contrib`=0 (nền) | | | {int((n_contrib==0).sum())} pixel |")
    P(f"| pixel early-termination ($T(1-\\alpha)<10^{{-4}}$) | | | {int(early.sum())} pixel |")
    P("")
    P("`max_contrib[T]` (BlockReduce Max của `last_contributor` trong tile) và số bucket `perTileBucketCount` = $\\lceil|\\mathcal G_T|/32\\rceil$:")
    P("")
    P("| tile $T$ | $|\\mathcal G_T|$ | `max_contrib[T]` | #bucket$_T$ | pixel trong tile | pixel early-term |")
    P("|---|---|---|---|---|---|")
    total_buckets = 0
    for T in range(NUM_TILES):
        a, b = R["ranges"][T]; nb = (b - a + 31) // 32; total_buckets += nb
        ty, tx = divmod(T, GRID_X)
        sl = (slice(ty*16, min(ty*16+16, H)), slice(tx*16, min(tx*16+16, W)))
        P(f"| {T} | {b-a} | {int(R['max_contrib'][T])} | {nb} | {16*16} | {int(early[sl].sum())} |")
    P("")
    P(f"Tổng warp backward $=\\sum_T\\#\\text{{bucket}}_T={total_buckets}$; mỗi bucket chứa tối đa 32 Gaussian nên xử lý được tới "
      f"{total_buckets*32} entry, trong khi $P=\\sum_iK_i={Ktot}$ — cận trên $bNK$ ở đây là {total_buckets} warp × 256 pixel. "
      "`sampled_T`/`sampled_ar` được ghi tại `j % 32 == 0`, tức 1 checkpoint (T=1, C=0) mỗi bucket vì mọi $|\\mathcal G_T|\\le32$.")
    P("")

    # ---------------------------------------------------------------- 4.7 GT alpha = 0.9
    G = run(0.9)
    np.save(os.path.join(HERE, "ch04_render_cam1_gt.npy"), G["img"].astype(np.float32))
    write_ppm(os.path.join(HERE, "ch04_render_cam1_gt.ppm"), G["img"])
    np.savez(os.path.join(HERE, "ch04_aux_cam1.npz"),
             final_T=final_T.astype(np.float32), n_contrib=n_contrib.astype(np.uint32),
             n_real=n_real.astype(np.uint32), ranges=R["ranges"], point_list=R["vals_s"],
             keys_sorted=R["keys_s"], max_contrib=R["max_contrib"],
             mu2=np.array([p["mu2"] for p in proj]), conic=np.array([p["conic"] for p in proj]),
             depth=np.array([p["depth"] for p in proj]), alpha=alpha, colors=cols,
             final_T_gt=G["final_T"].astype(np.float32), n_contrib_gt=G["n_contrib"].astype(np.uint32))

    P("## 4.7 — Cùng cảnh với $\\alpha=0.9$ (ảnh \"GT\" cho chương 5)")
    P("")
    P("Chỉ $\\alpha$ đổi ⇒ $t_i=0.5\\cdot2\\ln(255\\cdot0.9)=" + f4(G["tiles"][0]["t"]) + "$ (thay vì " + f4(tiles[0]["t"]) + ") → hộp rộng hơn, $K_i$ có thể tăng:")
    P("")
    P("| $i$ | $\\text{half}_x$ | $\\text{half}_y$ | $\\mathcal K_i$ | $K_i$ |")
    P("|---|---|---|---|---|")
    for i in range(N):
        tb = G["tiles"][i]; S11, _, S22 = G["proj"][i]["S"]
        P(f"| {i+1} | {f4(math.sqrt(tb['t']*S11))} | {f4(math.sqrt(tb['t']*S22))} | {tb['tiles']} | {len(tb['tiles'])} |")
    P("")
    P(f"$P^{{gt}}={int(G['K'].sum())}$; ranges: " + ", ".join(f"T{T}=[{a},{b})" for T, (a, b) in enumerate(G["ranges"])) + ".")
    P("")
    P("```")
    for line in ascii_art(G["img"]):
        P(line)
    P("```")
    P("")
    P("| Thống kê ($\\alpha=0.9$) | min | max | mean |")
    P("|---|---|---|---|")
    P(f"| `final_Ts` | {f4(G['final_T'].min())} | {f4(G['final_T'].max())} | {f4(G['final_T'].mean())} |")
    P(f"| `n_contrib` | {int(G['n_contrib'].min())} | {int(G['n_contrib'].max())} | {f4(G['n_contrib'].mean())} |")
    P(f"| pixel early-termination | | | **{int(G['early'].sum())}** pixel |")
    P(f"| pixel nền (`n_contrib`=0) | | | {int((G['n_contrib']==0).sum())} pixel |")
    P("")
    P(f"Số pixel bị early-termination với $\\alpha=0.9$: **{int(G['early'].sum())}** (với $\\alpha=0.1$: {int(early.sum())}). "
      "Với 4 Gaussian, $T$ nhỏ nhất có thể là $(1-0.9)^4=10^{-4}$ đúng bằng ngưỡng, còn $\\alpha_n(x)<\\alpha$ tại mọi pixel không trùng tâm nên $T$ không xuống dưới $10^{-4}$; "
      f"$T_{{final}}$ nhỏ nhất đo được = {f4(G['final_T'].min())}.")
    P("")
    P("Lưu: `scripts/ch04_render_cam1_gt.npy`, `scripts/ch04_render_cam1_gt.ppm`.")
    P("")

    # ---------------------------------------------------------------- Đầu vào / Đầu ra
    P("## Đầu vào của khối")
    P("")
    P("| Đại lượng | Giá trị / nguồn |")
    P("|---|---|")
    P("| $(\\mu'_i, M_i, \\alpha_i, c_i, \\text{depth}_i, \\mathcal K_i)$ | bảng mục 4.0 (chương 3, camera 1) |")
    P(f"| $P=\\sum K_i$ | {Ktot} (α=0.1), {int(G['K'].sum())} (α=0.9) |")
    P("| $C_{bg}$ | (1,1,1) |")
    P("| lưới tile | 3×2, tile 16×16, `BLOCK_SIZE`=256 thread |")
    P("")
    P("## Đầu ra của khối")
    P("")
    P("$I_{rend}=\\{C(x)\\}$, shape **(H, W, 3) = (32, 48, 3)**, float32, RGB ∈ [0,1], HWC, gốc (0,0) góc trên-trái (chương 5 cần CHW thì `transpose(2,0,1)`).")
    P("")
    P("| Ảnh | file | kênh | min | max | mean |")
    P("|---|---|---|---|---|---|")
    for name, path, im in (("$I_{rend}$ (α=0.1, mô hình khởi tạo)", "scripts/ch04_render_cam1.npy", img),
                           ("$I_{gt}$ (α=0.9, GT của chương 5)", "scripts/ch04_render_cam1_gt.npy", G["img"])):
        for ch, cn in enumerate("RGB"):
            P(f"| {name if ch == 0 else ''} | {path if ch == 0 else ''} | {cn} | {f4(im[..., ch].min())} | {f4(im[..., ch].max())} | {f4(im[..., ch].mean())} |")
    P("")
    l1 = np.abs(img - G["img"]).mean()
    P(f"Tham khảo nhanh cho chương 5: $\\text{{mean}}|I_{{rend}}-I_{{gt}}|$ (L1 trung bình trên 32·48·3 giá trị) $={f4(l1)}$.")
    P("")
    P("Dữ liệu cho backward (chương 6) lưu ở `scripts/ch04_aux_cam1.npz`: `final_T` (32,48), `n_contrib` (32,48), `n_real`, "
      "`ranges` (6,2), `point_list` (P,), `keys_sorted`, `max_contrib` (6,), `mu2` (4,2), `conic` (4,3), `depth`, `alpha`, `colors`, "
      "cùng `final_T_gt`, `n_contrib_gt` cho bản α=0.9.")
    P("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"\n[đã ghi] {os.path.abspath(OUT_MD)}")

    # tóm tắt cho báo cáo
    print("\n=== TÓM TẮT ===")
    print(f"(a) pixel {pa}: C = {Ca[0]}")
    print(f"(b) pixel {pb}: C = {Cb[0]}")
    if Cc: print(f"(c) pixel {pc}: C = {Cc[0]}")
    print(f"img mean per channel: {img.reshape(-1,3).mean(0)}, gt mean: {G['img'].reshape(-1,3).mean(0)}")


if __name__ == "__main__":
    main()
