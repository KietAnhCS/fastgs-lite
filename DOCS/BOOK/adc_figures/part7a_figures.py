"""
PHAN 7 (a): hinh chuoi thoi gian + chia vung cho hai bien the ADC (3DGS goc vs FastGS-lite)
tren cung toy 2D, cung seed. Dung engine part7_engine.py (cache part7_cache.pkl).

  7_1_3dgs-chuoi-thoi-gian.png   : render + ellipse 1.5σ tai 6 moc, mode 3dgs
  7_2_fastgs-chuoi-thoi-gian.png : cung layout, mode fastgs
  7_3_chia-vung-3dgs-vs-fastgs.png : ban do lanh tho, 2 hang (3dgs / fastgs) x 6 moc
"""
from part7_engine import *   # noqa: F401,F403

MODE_NAME = {"3dgs": "3DGS gốc", "fastgs": "FastGS-lite"}


def _timeline(mode, name, suptitle):
    R = full_run_compare()
    p_end, hist, snaps = R[mode]
    keys = sorted(snaps.keys())
    titles = snap_titles(hist, keys)
    fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 6.6))
    for j, t in enumerate(keys):
        sn = snaps[t]
        n = len(sn["p"]["mu"])
        show_img(axes[0, j], sn["I"], f"{titles[j]}\nN = {n}, L1 = {sn['loss']:.4f}")
        show_img(axes[1, j], TARGET * 0.3 + 0.7)
        draw_ellipses(axes[1, j], sn["p"], k=1.5, colors=gauss_colors(n), lw=1.3)
    axes[0, 0].set_ylabel("Render", fontsize=10)
    axes[1, 0].set_ylabel("Ellipse 1.5σ của từng Gaussian\ntrên nền mục tiêu (mờ)", fontsize=10)
    fig.suptitle(suptitle, fontsize=11)
    save(fig, name)


# =============================================================================
# 7.1  Chuoi thoi gian: ADC cua 3DGS goc
# =============================================================================
def fig_7_1():
    _timeline("3dgs", "7_1_3dgs-chuoi-thoi-gian.png",
              "ADC của 3DGS gốc trên toy 2D: clone/split chỉ theo gradient có dấu, "
              "prune cứng α<ε, không trần opacity")


# =============================================================================
# 7.2  Chuoi thoi gian: ADC cua FastGS-lite
# =============================================================================
def fig_7_2():
    _timeline("fastgs", "7_2_fastgs-chuoi-thoi-gian.png",
              "ADC của FastGS-lite trên cùng toy: AND Importance, split theo |g|, "
              "prune multinomial, trần 0.8, final prune tại t=1000/1100")


# =============================================================================
# 7.3  Chia vung: 3DGS goc vs FastGS-lite, 2 hang x 6 moc
# =============================================================================
def fig_7_3():
    R = full_run_compare()
    modes = ("3dgs", "fastgs")
    keys = sorted(R[modes[0]][2].keys())
    fig, axes = plt.subplots(len(modes), len(keys), figsize=(3.0 * len(keys), 6.8))
    for i, mode in enumerate(modes):
        _, hist, snaps = R[mode]
        titles = snap_titles(hist, keys)
        for j, t in enumerate(keys):
            sn = snaps[t]
            show_territory(axes[i, j], sn["p"], f"{titles[j]}\nN = {len(sn['p']['mu'])}")
        axes[i, 0].set_ylabel(MODE_NAME[mode], fontsize=11)
    fig.suptitle("Lãnh thổ của từng Gaussian, 3DGS gốc (trên) và FastGS-lite (dưới) "
                 "(pixel tô theo Gaussian có $T_i\\alpha_iG_i$ lớn nhất; trắng = nền; "
                 "đường đen = biên vật thể mục tiêu)", fontsize=10)
    save(fig, "7_3_chia-vung-3dgs-vs-fastgs.png")


if __name__ == "__main__":
    fig_7_1()
    fig_7_2()
    fig_7_3()
