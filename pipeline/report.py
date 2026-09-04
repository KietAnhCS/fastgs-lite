"""Data analytics: gộp lịch sử huấn luyện, bảng so sánh scene và biểu đồ."""

import json
import os

METRIC_COLS = ["score", "psnr", "psnr_norm", "ssim", "lpips"]


def history_frame(results, save_to=None):
    """Gộp `history` của mọi scene thành một DataFrame (và ghi CSV nếu muốn)."""
    import pandas as pd

    rows = [row for result in results for row in result.get("history", [])]
    frame = pd.DataFrame(rows)
    if save_to and not frame.empty:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        frame.to_csv(save_to, index=False)
    return frame


def leaderboard(results, submissions=None, save_to=None):
    """Bảng điểm từng scene + điểm trung bình toàn bộ scene (giống bảng xếp hạng)."""
    import pandas as pd

    final = {r["scene"]: r for r in results}
    rows = []
    for scene, result in final.items():
        row = dict(scene=scene, iters=result.get("iterations"),
                   n_gauss=result.get("n_gauss"),
                   train_s=round(result.get("total_time_s", 0.0), 1),
                   peak_vram_gb=round(result.get("peak_vram_gb", 0.0), 2))
        row.update({f"live_{k}": result.get(k) for k in METRIC_COLS})
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("scene")
    if submissions:
        sub = pd.DataFrame([{**{"scene": s["scene"], "images": s["images"],
                                "size": f"{s['width']}x{s['height']}"},
                             **{k: s.get(k) for k in METRIC_COLS}} for s in submissions])
        frame = frame.join(sub.set_index("scene"))

    numeric = frame.select_dtypes("number")
    frame.loc["MEAN"] = numeric.mean().reindex(frame.columns)
    frame = frame.round(4)
    if save_to:
        frame.to_csv(save_to)
    if "score" in frame.columns:
        print(f"Điểm leaderboard (trung bình các scene): {frame.loc['MEAN', 'score']:.4f}")
    return frame


def plot_training(history, save_to=None):
    """4 khối: Score theo iteration, ΔScore mỗi mốc, số Gaussian, thời gian."""
    import matplotlib.pyplot as plt

    if history.empty:
        print("không có lịch sử để vẽ")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    scenes = list(history["scene"].unique())

    ax = axes[0, 0]
    for scene in scenes:
        part = history[history.scene == scene]
        ax.plot(part["iter"], part["score"], marker="o", ms=3, label=scene)
    ax.set(xlabel="iteration", ylabel="Score", title="Score = .4(1-LPIPS)+.3SSIM+.3PSNR_norm")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for scene in scenes:
        part = history[history.scene == scene]
        ax.plot(part["iter"], part["psnr"], marker="o", ms=3, label=f"{scene} PSNR")
        ax.plot(part["iter"], part["ssim"] * 30, ls="--", lw=1, label=f"{scene} SSIM×30")
        ax.plot(part["iter"], part["lpips"] * 30, ls=":", lw=1, label=f"{scene} LPIPS×30")
    ax.set(xlabel="iteration", ylabel="PSNR (dB) / metric×30", title="Ba thành phần metric")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    if "d_score" in history.columns:
        for scene in scenes:
            part = history[history.scene == scene].dropna(subset=["d_score"])
            ax.bar(part["iter"], part["d_score"], width=max(1, history["iter"].max() / 40),
                   alpha=.6, label=scene)
        ax.axhline(0, c="k", lw=.8)
    ax.set(xlabel="iteration", ylabel="ΔScore mỗi mốc đánh giá", title="Tiến bộ giữa hai lần chấm")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for scene in scenes:
        part = history[history.scene == scene]
        ax.plot(part["n_gauss"], part["score"], marker="o", ms=3, label=scene)
    ax.set(xlabel="số Gaussian", ylabel="Score", title="Chi phí mô hình đổi lấy chất lượng")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    return fig


def plot_leaderboard(board, save_to=None):
    """So sánh Score và ba metric giữa các scene."""
    import matplotlib.pyplot as plt

    frame = board.drop(index="MEAN", errors="ignore")
    columns = [c for c in ("score", "live_score") if c in frame.columns]
    if not columns:
        print("chưa có cột điểm để so sánh")
        return None
    column = columns[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].bar(frame.index, frame[column], color="#3b7dd8")
    axes[0].axhline(frame[column].mean(), ls="--", c="crimson",
                    label=f"trung bình {frame[column].mean():.4f}")
    axes[0].set(ylabel="Score", title="Score theo scene")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.3, axis="y")

    prefix = "" if column == "score" else "live_"
    metrics = [m for m in ("psnr_norm", "ssim", "lpips") if f"{prefix}{m}" in frame.columns]
    width = 0.8 / max(1, len(metrics))
    for offset, metric in enumerate(metrics):
        positions = [i + offset * width for i in range(len(frame))]
        axes[1].bar(positions, frame[f"{prefix}{metric}"], width=width, label=metric)
    axes[1].set_xticks([i + 0.4 for i in range(len(frame))])
    axes[1].set_xticklabels(frame.index, rotation=30)
    axes[1].set(title="Thành phần metric (LPIPS càng thấp càng tốt)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.3, axis="y")

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    return fig


def show_samples(cfg, scene, n=3, save_to=None):
    """Ghép ảnh render cạnh ground-truth để soi hình học và vị trí vật thể."""
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from pipeline import data as data_mod

    manifest_path = os.path.join(cfg.submission_scene_dir(scene), "_manifest.json")
    if not os.path.exists(manifest_path):
        print("chưa render scene", scene)
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    image_root = os.path.join(data_mod.scene_path(cfg, scene), cfg.images_dir)
    picks = manifest["files"][:: max(1, len(manifest["files"]) // n)][:n]

    fig, axes = plt.subplots(2, len(picks), figsize=(4.2 * len(picks), 6.4), squeeze=False)
    for column, entry in enumerate(picks):
        render = Image.open(os.path.join(cfg.submission_scene_dir(scene), entry["file"]))
        axes[0][column].imshow(np.asarray(render))
        axes[0][column].set_title(f"render {entry['file']}", fontsize=9)
        gt_path = None
        for ext in (".png", ".jpg", ".JPG", ".jpeg"):
            candidate = os.path.join(image_root, entry["source"] + ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break
        if gt_path:
            axes[1][column].imshow(np.asarray(Image.open(gt_path)))
            axes[1][column].set_title(f"gt {entry['source']}", fontsize=9)
        for row in (0, 1):
            axes[row][column].axis("off")

    fig.suptitle(f"{scene}: render vs ground-truth")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig
