"""
Chương 3 — Camera + Projection: bài test số trên cảnh đồ chơi (DOCS/Report/test/00-scene.md).

Chỉ dùng numpy. Chép lại đúng logic của:
  - utils/graphics_utils.py:51-71        getProjectionMatrix
  - scene/cameras.py:54-56               world_view_transform / full_proj_transform (quy ước vector hàng)
  - cuda_rasterizer/auxiliary.h:45-48    ndc2Pix
  - cuda_rasterizer/auxiliary.h:131-156  in_frustum   (cull t_z <= 0.2)
  - cuda_rasterizer/forward.cu:79-118    computeCov2D (clamp 1.3 tan, J, W, +0.3)
  - cuda_rasterizer/forward.cu:227-244   conic, lambda, my_radius, point_image
  - cuda_rasterizer/auxiliary.h:159-174  computeEllipseIntersection
  - cuda_rasterizer/auxiliary.h:177-291  processTiles  (AccuTile)
  - cuda_rasterizer/auxiliary.h:294-362  duplicateToTilesTouched (SnugBox + compact box)
  - getRect của 3DGS gốc (đã bị xoá khỏi auxiliary.h, chép lại để so sánh)

Khởi tạo (chương 1) được tính lại tại chỗ: mu = p, s = sqrt(mean d^2 tới 3 láng giềng),
q = (1,0,0,0) -> R = I, Sigma = s^2 I, alpha = 0.1, màu = c_k.
"""
import math
import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=140)

# ----------------------------------------------------------------------------------
# Cảnh đồ chơi (00-scene.md)
# ----------------------------------------------------------------------------------
W, H = 48, 32
BLOCK_X = BLOCK_Y = 16
GRID = (W // BLOCK_X, H // BLOCK_Y)  # (3, 2)
TAN_X, TAN_Y = 0.6, 0.4
FX = W / (2 * TAN_X)  # 40
FY = H / (2 * TAN_Y)  # 40
ZNEAR, ZFAR = 0.01, 100.0
MULT = 0.5
ALPHA = 0.1

CAMS = {1: np.array([0.0, 0.0, -4.0]),
        2: np.array([1.5, 0.0, -4.0]),
        3: np.array([-1.5, 0.5, -4.0])}

P_SFM = np.array([[0.0, 0.0, 0.0],
                  [0.5, 0.3, 0.5],
                  [-0.4, -0.2, 1.0],
                  [0.3, -0.5, 0.2]])
C_SFM = np.array([[0.8, 0.2, 0.2],
                  [0.2, 0.7, 0.3],
                  [0.1, 0.3, 0.9],
                  [0.5, 0.5, 0.5]])
N = len(P_SFM)


def hr(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ----------------------------------------------------------------------------------
# Khởi tạo lại (chương 1) — không chờ bot khác
# ----------------------------------------------------------------------------------
hr("Chương 1 (tính lại): mu, s, Sigma, alpha")
D2 = ((P_SFM[:, None, :] - P_SFM[None, :, :]) ** 2).sum(-1)
S_INIT = np.zeros(N)
for i in range(N):
    others = [j for j in range(N) if j != i]
    d2 = np.sort(D2[i, others])[:3]
    S_INIT[i] = math.sqrt(max(d2.mean(), 1e-7))
    print(f"  G{i+1}: mu={P_SFM[i]}, d2_knn3={d2}, mean={d2.mean():.6f}, s={S_INIT[i]:.6f}, s^2={S_INIT[i]**2:.6f}")
SIGMA3D = [s * s * np.eye(3) for s in S_INIT]  # R = I -> Sigma = s^2 I


# ----------------------------------------------------------------------------------
# 3.1  Ma trận camera
# ----------------------------------------------------------------------------------
def getWorld2View(c):
    """W_v (world->camera, quy ước cột): R = I, t = -c."""
    Rt = np.eye(4)
    Rt[:3, 3] = -c
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Chép nguyên văn utils/graphics_utils.py:51-71 (quy ước cột)."""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = np.zeros((4, 4))
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


FOVX = 2 * math.atan(TAN_X)
FOVY = 2 * math.atan(TAN_Y)
P_COL = getProjectionMatrix(ZNEAR, ZFAR, FOVX, FOVY)

hr("3.1  Đầu vào từ camera")
print(f"fovx = {FOVX:.6f} rad, fovy = {FOVY:.6f} rad, tan = ({math.tan(FOVX/2):.6f}, {math.tan(FOVY/2):.6f})")
print(f"fx = {FX}, fy = {FY}")
print("P (quy ước cột, graphics_utils.getProjectionMatrix):")
print(P_COL)

# scene/cameras.py:54-56 — code lưu ma trận CHUYỂN VỊ và dùng vector hàng:
#   world_view_transform = W_v^T ;  projection_matrix = P^T ;  full_proj_transform = W_v^T @ P^T = (P W_v)^T
#   p_hom = [mu, 1] @ full_proj_transform   (transformPoint4x4: matrix[0]x + matrix[4]y + matrix[8]z + matrix[12])
VIEW_ROW = {v: getWorld2View(c).T for v, c in CAMS.items()}          # world_view_transform
PROJ_ROW = P_COL.T                                                   # projection_matrix
FULL_ROW = {v: VIEW_ROW[v] @ PROJ_ROW for v in CAMS}                 # full_proj_transform

for v in CAMS:
    print(f"\n-- Camera {v}: c_v = {CAMS[v]}")
    print("W_v (quy ước cột, t = mu - c):")
    print(getWorld2View(CAMS[v]))
    if v == 1:
        print("world_view_transform = W_v^T (code, vector hàng):")
        print(VIEW_ROW[v])
        print("full_proj_transform = W_v^T @ P^T (code):")
        print(FULL_ROW[v])
        print("Kiểm tra: full_proj_transform == (P @ W_v)^T ?",
              np.allclose(FULL_ROW[v], (P_COL @ getWorld2View(CAMS[v])).T))
    else:
        print("full_proj_transform (gọn): hàng 4 =", FULL_ROW[v][3], " | khối 3x3 trên-trái = diag",
              np.diag(FULL_ROW[v][:3, :3]), " | cột 4 =", FULL_ROW[v][:, 3])


# ----------------------------------------------------------------------------------
# helper chép từ auxiliary.h / forward.cu
# ----------------------------------------------------------------------------------
def transformPoint4x3(p, M):
    return (np.append(p, 1.0) @ M)[:3]


def transformPoint4x4(p, M):
    return np.append(p, 1.0) @ M


def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def computeCov2D(mean, focal_x, focal_y, tan_fovx, tan_fovy, Sigma, viewmatrix_row, verbose=False):
    """forward.cu:79-118. viewmatrix_row = world_view_transform (W_v^T)."""
    t = transformPoint4x3(mean, viewmatrix_row)
    limx, limy = 1.3 * tan_fovx, 1.3 * tan_fovy
    txtz, tytz = t[0] / t[2], t[1] / t[2]
    tx_hat = min(limx, max(-limx, txtz)) * t[2]
    ty_hat = min(limy, max(-limy, tytz)) * t[2]
    tz = t[2]
    J = np.array([[focal_x / tz, 0.0, -(focal_x * tx_hat) / (tz * tz)],
                  [0.0, focal_y / tz, -(focal_y * ty_hat) / (tz * tz)],
                  [0.0, 0.0, 0.0]])
    Wm = viewmatrix_row[:3, :3].T   # glm column-major: W = (W_v)_{3x3}
    cov3 = J @ Wm @ Sigma @ Wm.T @ J.T
    cov = cov3[:2, :2].copy()
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3
    info = dict(t=t, txtz=txtz, tytz=tytz, limx=limx, limy=limy, tx_hat=tx_hat, ty_hat=ty_hat,
                J=J, W=Wm, cov3=cov3)
    return cov, info


def computeEllipseIntersection(con_o, disc, t, p, isY, coord):
    """auxiliary.h:159-174. con_o = (A, B, C, alpha)."""
    p_u = p[1] if isY else p[0]
    p_v = p[0] if isY else p[1]
    coeff = con_o[0] if isY else con_o[2]
    h = coord - p_u
    sqrt_term = math.sqrt(disc * h * h + t * coeff)
    return ((-con_o[1] * h - sqrt_term) / coeff + p_v,
            (-con_o[1] * h + sqrt_term) / coeff + p_v)


def processTiles(con_o, disc, t, p, bbox_min, bbox_max, bbox_argmin, bbox_argmax,
                 rect_min, rect_max, grid, isY, log=None):
    """auxiliary.h:177-291. Trả về (tiles_count, danh sách tile id = ty*grid.x + tx)."""
    BLOCK_U = BLOCK_Y if isY else BLOCK_X
    BLOCK_V = BLOCK_X if isY else BLOCK_Y
    if isY:
        rect_min = (rect_min[1], rect_min[0]); rect_max = (rect_max[1], rect_max[0])
        bbox_min = (bbox_min[1], bbox_min[0]); bbox_max = (bbox_max[1], bbox_max[0])
        bbox_argmin = (bbox_argmin[1], bbox_argmin[0]); bbox_argmax = (bbox_argmax[1], bbox_argmax[0])
    tiles_count = 0
    ids = []
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
        tiles_count += max_tile_v - min_tile_v
        if log is not None:
            log.append(dict(u=u, min_line=min_line, max_line=max_line,
                            ellipse_min=ellipse_min, ellipse_max=ellipse_max,
                            min_tile_v=min_tile_v, max_tile_v=max_tile_v,
                            n=max_tile_v - min_tile_v))
        for vv in range(min_tile_v, max_tile_v):
            ids.append(u * grid[0] + vv if isY else vv * grid[0] + u)
        intersect_min_line = intersect_max_line
        min_line = max_line
    return tiles_count, ids


def duplicateToTilesTouched(p, con_o, grid, mult, log=None):
    """auxiliary.h:294-362. Trả về dict đầy đủ các đại lượng trung gian."""
    A, B, C, o = con_o
    disc = B * B - A * C
    out = dict(disc=disc)
    if A <= 0 or C <= 0 or disc >= 0:
        out.update(count=0, ids=[])
        return out
    t = 2.0 * math.log(o * 255.0)
    t = mult * t
    x_term = math.sqrt(-(B * B * t) / (disc * A))
    x_term = x_term if B < 0 else -x_term
    y_term = math.sqrt(-(B * B * t) / (disc * C))
    y_term = y_term if B < 0 else -y_term
    bbox_argmin = (p[1] - y_term, p[0] - x_term)
    bbox_argmax = (p[1] + y_term, p[0] + x_term)
    bbox_min = (computeEllipseIntersection(con_o, disc, t, p, True, bbox_argmin[0])[0],
                computeEllipseIntersection(con_o, disc, t, p, False, bbox_argmin[1])[0])
    bbox_max = (computeEllipseIntersection(con_o, disc, t, p, True, bbox_argmax[0])[1],
                computeEllipseIntersection(con_o, disc, t, p, False, bbox_argmax[1])[1])
    rect_min = (max(0, min(grid[0], int(bbox_min[0] / BLOCK_X))),
                max(0, min(grid[1], int(bbox_min[1] / BLOCK_Y))))
    rect_max = (max(0, min(grid[0], int(bbox_max[0] / BLOCK_X + 1))),
                max(0, min(grid[1], int(bbox_max[1] / BLOCK_Y + 1))))
    y_span = rect_max[1] - rect_min[1]
    x_span = rect_max[0] - rect_min[0]
    out.update(t=t, x_term=x_term, y_term=y_term, bbox_argmin=bbox_argmin, bbox_argmax=bbox_argmax,
               bbox_min=bbox_min, bbox_max=bbox_max, rect_min=rect_min, rect_max=rect_max,
               x_span=x_span, y_span=y_span, K_box=x_span * y_span)
    if y_span * x_span == 0:
        out.update(count=0, ids=[], isY=None)
        return out
    isY = y_span < x_span
    out["isY"] = isY
    cnt, ids = processTiles(con_o, disc, t, p, bbox_min, bbox_max, bbox_argmin, bbox_argmax,
                            rect_min, rect_max, grid, isY, log)
    out.update(count=cnt, ids=sorted(ids))
    return out


def getRect_3dgs(p, max_radius, grid):
    """getRect của 3DGS gốc (đã bị xoá khỏi auxiliary.h của FastGS)."""
    rect_min = (min(grid[0], max(0, int((p[0] - max_radius) / BLOCK_X))),
                min(grid[1], max(0, int((p[1] - max_radius) / BLOCK_Y))))
    rect_max = (min(grid[0], max(0, int((p[0] + max_radius + BLOCK_X - 1) / BLOCK_X))),
                min(grid[1], max(0, int((p[1] + max_radius + BLOCK_Y - 1) / BLOCK_Y))))
    return rect_min, rect_max


def exact_tile_min(M, p, t, grid, rect_min=None, rect_max=None):
    """Kiểm tra CHÍNH XÁC  min_{Delta in T} Delta^T M Delta <= t  cho từng tile T của lưới.
    T = [16tx, 16(tx+1)] x [16ty, 16(ty+1)] (đóng). Hàm bậc hai lồi trên hình chữ nhật:
    min đạt tại tâm (nếu tâm trong T) hoặc trên 4 cạnh (min 1-D của parabol, kẹp vào đoạn)."""
    A, B, C = M[0, 0], M[0, 1], M[1, 1]

    def q(dx, dy):
        return A * dx * dx + 2 * B * dx * dy + C * dy * dy

    def min_on_segment(x0, x1, y0, y1, along_x):
        # tối ưu 1-D của q trên đoạn thẳng (cạnh)
        if along_x:
            dy = y0 - p[1]
            # q(dx) = A dx^2 + 2B dx dy + C dy^2 -> dx* = -B dy / A
            dxs = -B * dy / A
            dx = min(max(dxs, x0 - p[0]), x1 - p[0])
            return q(dx, dy)
        else:
            dx = x0 - p[0]
            dys = -B * dx / C
            dy = min(max(dys, y0 - p[1]), y1 - p[1])
            return q(dx, dy)

    res = {}
    for ty in range(grid[1]):
        for tx in range(grid[0]):
            x0, x1 = tx * BLOCK_X, (tx + 1) * BLOCK_X
            y0, y1 = ty * BLOCK_Y, (ty + 1) * BLOCK_Y
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                m = 0.0
            else:
                m = min(min_on_segment(x0, x1, y0, y0, True),
                        min_on_segment(x0, x1, y1, y1, True),
                        min_on_segment(x0, x0, y0, y1, False),
                        min_on_segment(x1, x1, y0, y1, False))
            res[ty * grid[0] + tx] = m
    ids = sorted(k for k, m in res.items() if m <= t)
    return res, ids


# ----------------------------------------------------------------------------------
# 3.2 / 3.3  t_i và mu'_i cho 4 Gaussian x 3 camera
# ----------------------------------------------------------------------------------
hr("3.2  Cull theo frustum: t_i = W_v [mu;1]  (transformPoint4x3 với world_view_transform)")
T_VIEW = {}
for v in CAMS:
    for i in range(N):
        t = transformPoint4x3(P_SFM[i], VIEW_ROW[v])
        T_VIEW[(v, i)] = t
        print(f"  cam {v}, G{i+1}: t = ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})   t_z <= 0.2 ? {t[2] <= 0.2}  -> {'CULL' if t[2] <= 0.2 else 'giữ'}")

hr("3.3  Chiếu tâm: p_hom = [mu,1] @ full_proj_transform ; p_proj = xyz/(w+1e-7) ; ndc2Pix")
MU2D = {}
for v in CAMS:
    for i in range(N):
        p_hom = transformPoint4x4(P_SFM[i], FULL_ROW[v])
        p_w = 1.0 / (p_hom[3] + 1e-7)
        p_proj = p_hom[:3] * p_w
        mu2 = np.array([ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H)])
        MU2D[(v, i)] = mu2
        print(f"  cam {v}, G{i+1}: p_hom = ({p_hom[0]:+.5f}, {p_hom[1]:+.5f}, {p_hom[2]:+.5f}, {p_hom[3]:+.5f})"
              f"  p_proj = ({p_proj[0]:+.6f}, {p_proj[1]:+.6f}, {p_proj[2]:+.6f})"
              f"  mu' = ({mu2[0]:.4f}, {mu2[1]:.4f}) px")

# ----------------------------------------------------------------------------------
# 3.4 - 3.7  Camera 1 chi tiết (và các camera khác gọn)
# ----------------------------------------------------------------------------------
RESULT = {}
for v in CAMS:
    hr(f"3.4 -> 3.7  Camera {v}" + ("  (CHI TIẾT)" if v == 1 else "  (gọn)"))
    for i in range(N):
        cov, info = computeCov2D(P_SFM[i], FX, FY, TAN_X, TAN_Y, SIGMA3D[i], VIEW_ROW[v])
        t = info["t"]
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[0, 1]
        det_inv = 1.0 / det
        A, B, C = cov[1, 1] * det_inv, -cov[0, 1] * det_inv, cov[0, 0] * det_inv
        disc = B * B - A * C
        mid = 0.5 * (cov[0, 0] + cov[1, 1])
        lam1 = mid + math.sqrt(max(0.1, mid * mid - det))
        lam2 = mid - math.sqrt(max(0.1, mid * mid - det))
        lam_max_exact = mid + math.sqrt(mid * mid - det)
        lam_min_exact = mid - math.sqrt(mid * mid - det)
        my_radius = math.ceil(3.0 * math.sqrt(max(lam1, lam2)))
        rho = math.sqrt(lam_max_exact / lam_min_exact)
        mu2 = MU2D[(v, i)]
        # 3DGS gốc
        r_min, r_max = getRect_3dgs(mu2, my_radius, GRID)
        K3 = (r_max[0] - r_min[0]) * (r_max[1] - r_min[1])
        ids3 = sorted(ty * GRID[0] + tx for ty in range(r_min[1], r_max[1]) for tx in range(r_min[0], r_max[0]))
        # FastGS
        con_o = (A, B, C, ALPHA)
        log = []
        fg = duplicateToTilesTouched(mu2, con_o, GRID, MULT, log)
        tval = MULT * 2.0 * math.log(255.0 * ALPHA)
        half_x = math.sqrt(tval * cov[0, 0])
        half_y = math.sqrt(tval * cov[1, 1])
        M = np.array([[A, B], [B, C]])
        exact_min, ids_exact = exact_tile_min(M, mu2, tval, GRID)

        RESULT[(v, i)] = dict(mu2=mu2, cov=cov, det=det, A=A, B=B, C=C, disc=disc,
                              lam_max=lam_max_exact, lam_min=lam_min_exact, rho=rho, r=my_radius,
                              rect3=(r_min, r_max), K3=K3, ids3=ids3, t=tval, half=(half_x, half_y),
                              fg=fg, ids_exact=ids_exact, exact_min=exact_min, depth=t[2])

        if v == 1:
            print(f"\n----- G{i+1}: mu = {P_SFM[i]}, s = {S_INIT[i]:.6f}, s^2 = {S_INIT[i]**2:.6f}")
            print(f"  t = ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")
            print(f"  t_x/t_z = {info['txtz']:+.6f} (lim ±{info['limx']:.2f}) -> t̂_x = {info['tx_hat']:+.6f} ;"
                  f"  t_y/t_z = {info['tytz']:+.6f} (lim ±{info['limy']:.2f}) -> t̂_y = {info['ty_hat']:+.6f}"
                  f"   clamp có tác dụng? x:{abs(info['txtz']) > info['limx']} y:{abs(info['tytz']) > info['limy']}")
            print("  J =\n" + str(info["J"]))
            print("  W (3x3 của W_v) =\n" + str(info["W"]))
            print("  Sigma'_3x3 = J W Sigma W^T J^T =\n" + str(info["cov3"]))
            print(f"  Sigma' (2x2, +0.3 đường chéo) = [[{cov[0,0]:.6f}, {cov[0,1]:.6f}], [{cov[0,1]:.6f}, {cov[1,1]:.6f}]]")
            print(f"  3.5: det = {det:.6f} ; A = {A:.6f}, B = {B:.6f}, C = {C:.6f} ; disc = B^2 - AC = {disc:.6f} (<0: {disc < 0})"
                  f" ; -1/det = {-1/det:.6f}")
            print(f"  3.6 (3DGS): mid = {mid:.6f}, lam_max = {lam_max_exact:.6f}, lam_min = {lam_min_exact:.6f}, rho = {rho:.4f}"
                  f" ; code(max(0.1,.)): lam1 = {lam1:.6f}, lam2 = {lam2:.6f} ; r = ceil(3 sqrt lam_max) = ceil({3*math.sqrt(max(lam1,lam2)):.4f}) = {my_radius}")
            print(f"        mu' = ({mu2[0]:.4f}, {mu2[1]:.4f}) ; rect_min = {r_min}, rect_max = {r_max} -> K^3dgs = {K3}, tile = {ids3}")
            print(f"  3.6 (FastGS): t = {MULT}*2 ln(255*{ALPHA}) = {tval:.6f} ; half_x = sqrt(t*S11) = {half_x:.6f} ; half_y = sqrt(t*S22) = {half_y:.6f}")
            print(f"        Box x = [{mu2[0]-half_x:.4f}, {mu2[0]+half_x:.4f}] ; y = [{mu2[1]-half_y:.4f}, {mu2[1]+half_y:.4f}]")
            print(f"        code: x_term = {fg['x_term']:+.6f}, y_term = {fg['y_term']:+.6f}")
            print(f"              bbox_min (x,y) = ({fg['bbox_min'][0]:.4f}, {fg['bbox_min'][1]:.4f}) ; bbox_max = ({fg['bbox_max'][0]:.4f}, {fg['bbox_max'][1]:.4f})")
            print(f"              bbox_argmin (y,x) = ({fg['bbox_argmin'][0]:.4f}, {fg['bbox_argmin'][1]:.4f}) ; bbox_argmax (y,x) = ({fg['bbox_argmax'][0]:.4f}, {fg['bbox_argmax'][1]:.4f})")
            print(f"              rect_min = {fg['rect_min']}, rect_max = {fg['rect_max']} ; x_span = {fg['x_span']}, y_span = {fg['y_span']} -> K^box = {fg['K_box']}")
            print(f"  3.7 (AccuTile): isY = {fg['isY']} ; quét theo {'y (lát ngang)' if fg['isY'] else 'x (lát dọc)'}")
            for L in log:
                print(f"        lát u={L['u']}: [{L['min_line']:.0f},{L['max_line']:.0f}) -> ellipse_min = {L['ellipse_min']:.4f}, ellipse_max = {L['ellipse_max']:.4f}"
                      f" -> tile v từ {L['min_tile_v']} đến {L['max_tile_v']-1} ({L['n']} tile)")
            print(f"        K^fastgs = {fg['count']} ; tập tile id (ty*3+tx) = {fg['ids']}")
            print(f"        Kiểm tra chính xác min_T Delta^T M Delta <= t = {tval:.4f}:")
            for k in sorted(exact_min):
                print(f"          tile {k} (tx={k%GRID[0]},ty={k//GRID[0]}): min = {exact_min[k]:.4f} -> {'GIỮ' if exact_min[k] <= tval else 'loại'}")
            print(f"        tập tile chính xác = {ids_exact} ; trùng với processTiles? {ids_exact == fg['ids']}")
        else:
            print(f"  G{i+1}: mu'=({mu2[0]:.3f},{mu2[1]:.3f}) S'=[{cov[0,0]:.4f},{cov[0,1]:.4f},{cov[1,1]:.4f}] "
                  f"A,B,C=({A:.5f},{B:.5f},{C:.5f}) r={my_radius} K3dgs={K3} {ids3} | half=({half_x:.3f},{half_y:.3f}) "
                  f"Kbox={fg['K_box']} Kfastgs={fg['count']} {fg['ids']} exact={ids_exact}")

# ----------------------------------------------------------------------------------
# Tổng hợp R_tile
# ----------------------------------------------------------------------------------
hr("Tổng hợp R_tile")
for v in CAMS:
    s3 = sum(RESULT[(v, i)]["K3"] for i in range(N))
    sb = sum(RESULT[(v, i)]["fg"]["K_box"] for i in range(N))
    sf = sum(RESULT[(v, i)]["fg"]["count"] for i in range(N))
    se = sum(len(RESULT[(v, i)]["ids_exact"]) for i in range(N))
    print(f"  cam {v}: sum K^3dgs = {s3}, sum K^box = {sb}, sum K^fastgs = {sf} (exact {se}) -> R_tile = {sf}/{s3} = {sf/s3:.4f}")
S3 = sum(RESULT[k]["K3"] for k in RESULT)
SB = sum(RESULT[k]["fg"]["K_box"] for k in RESULT)
SF = sum(RESULT[k]["fg"]["count"] for k in RESULT)
print(f"  3 camera: sum K^3dgs = {S3}, sum K^box = {SB}, sum K^fastgs = {SF} -> R_tile = {SF/S3:.4f}")

# ----------------------------------------------------------------------------------
# 3.8  Kiểm chứng ví dụ box.md bằng cùng code
# ----------------------------------------------------------------------------------
hr("3.8  Kiểm chứng ví dụ box.md: Sigma'=[[117,54],[54,36]], mu'=(120,88), alpha=1, mult=0.5, lưới 20x20")
cov = np.array([[117.0, 54.0], [54.0, 36.0]])
mu2 = np.array([120.0, 88.0])
grid_big = (20, 20)
det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
A, B, C = cov[1, 1] / det, -cov[0, 1] / det, cov[0, 0] / det
mid = 0.5 * (cov[0, 0] + cov[1, 1])
lam_max = mid + math.sqrt(mid * mid - det); lam_min = mid - math.sqrt(mid * mid - det)
r = math.ceil(3 * math.sqrt(lam_max))
rmin, rmax = getRect_3dgs(mu2, r, grid_big)
K3 = (rmax[0] - rmin[0]) * (rmax[1] - rmin[1])
log = []
fg = duplicateToTilesTouched(mu2, (A, B, C, 1.0), grid_big, MULT, log)
tval = MULT * 2 * math.log(255.0)
hx, hy = math.sqrt(tval * cov[0, 0]), math.sqrt(tval * cov[1, 1])
print(f"  det = {det:.4f}, A,B,C = ({A:.6f}, {B:.6f}, {C:.6f}) = (1/36, -1/24, 13/144)? "
      f"{np.allclose([A,B,C],[1/36,-1/24,13/144])}")
print(f"  lam_max = {lam_max:.4f}, lam_min = {lam_min:.4f}, rho = {math.sqrt(lam_max/lam_min):.4f}, r = {r}")
print(f"  t = {tval:.4f} (kỳ vọng 5.541) ; half_x = {hx:.4f} (kỳ vọng 25.46) ; half_y = {hy:.4f} (kỳ vọng 14.12)")
print(f"  3DGS: rect = {rmin}..{rmax} -> K^3dgs = {K3} (kỳ vọng 25)")
print(f"  FastGS: bbox_min = ({fg['bbox_min'][0]:.3f},{fg['bbox_min'][1]:.3f}) bbox_max = ({fg['bbox_max'][0]:.3f},{fg['bbox_max'][1]:.3f})"
      f" rect = {fg['rect_min']}..{fg['rect_max']} -> K^box = {fg['K_box']} (kỳ vọng 15)")
print(f"  isY = {fg['isY']}")
for L in log:
    print(f"    lát u={L['u']}: [{L['min_line']:.0f},{L['max_line']:.0f}) -> [{L['ellipse_min']:.3f}, {L['ellipse_max']:.3f}] -> {L['n']} tile (v {L['min_tile_v']}..{L['max_tile_v']-1})")
print(f"  K^fastgs = {fg['count']} (kỳ vọng 9) ; R_tile = {fg['count']}/{K3} = {fg['count']/K3:.3f}")
M = np.array([[A, B], [B, C]])
_, ids_exact = exact_tile_min(M, mu2, tval, grid_big)
print(f"  Tập tile processTiles = {fg['ids']}")
print(f"  Tập tile chính xác   = {ids_exact}  (số = {len(ids_exact)})")
ok = (K3 == 25 and fg['K_box'] == 15 and fg['count'] == 9 and abs(hx - 25.46) < 0.01 and abs(hy - 14.12) < 0.01 and abs(tval - 5.541) < 0.001)
print(f"  KHỚP 25 -> 15 -> 9, half = 25.46/14.12, t = 5.541 ?  {ok}")

# ----------------------------------------------------------------------------------
# 3.10  Đầu ra của khối cho camera 1
# ----------------------------------------------------------------------------------
hr("3.10  Đầu ra của khối — camera 1 (đi vào chương 4)")
print("| i | mu'_i (px) | A | B | C | alpha_i | c_i | depth=t_z | K_i (tile id) | K_i |")
Ptot = 0
for i in range(N):
    R = RESULT[(1, i)]
    Ptot += R["fg"]["count"]
    print(f"| {i+1} | ({R['mu2'][0]:.4f}, {R['mu2'][1]:.4f}) | {R['A']:.6f} | {R['B']:.6f} | {R['C']:.6f} | {ALPHA} | "
          f"({C_SFM[i][0]}, {C_SFM[i][1]}, {C_SFM[i][2]}) | {R['depth']:.4f} | {R['fg']['ids']} | {R['fg']['count']} |")
print(f"P = sum K_i = {Ptot}")
