<div align="center">

# fastgs-lite

**A trimmed [FastGS](https://github.com/fastgs/FastGS) fork that trains 3D Gaussian Splatting on a single free Google Colab T4.**

[🌐 Upstream homepage](https://fastgs.github.io/) · [📄 Paper (arXiv:2511.04283)](https://arxiv.org/abs/2511.04283) · [🤗 Pre-trained models](https://huggingface.co/Goodsleepeverday/fastgs)

</div>

---

FastGS replaces the question *"does this Gaussian have a large gradient?"* with *"do several cameras agree that this region is wrong?"*. That multi-view consistency signal is cheap to compute (10 extra renders) and far more selective, so Gaussians get added in the right place, pruned at the right time, and every rasterization step stays small.

This fork keeps that method intact and rebuilds everything around it for a **16 GB GPU and 12.7 GB of system RAM**.

## What this fork changes

| | Upstream FastGS | fastgs-lite |
|---|---|---|
| Target hardware | 24 GB VRAM, RTX 4090 | **free Colab T4** (~15 GB VRAM, ~12.7 GB RAM) |
| CUDA extensions | git submodules | **vendored in-tree** — `git clone` is enough, no `--recursive` |
| Entry point | `train.py` + shell scripts | same, plus a **documented notebook** that rebuilds the training loop with instrumentation |
| Progress signal | L1 loss + EMA | **composite Score logged every 1000 iterations**, with the per-1000 delta |
| Memory | assumes headroom | **anti-OOM playbook** so the model is still retrievable after training |
| Docs | README + wiki | [`DOCS/`](DOCS/README.md) — math foundation, mechanism walkthrough, Colab guide |

Extra task branches from upstream (dynamic scenes, sparse view, surface reconstruction, SLAM) are **not** included here — see [Not included](#not-included).

## Quick start

### Colab (the path this fork is built for)

Open [`fastgs-acceleration-method.ipynb`](fastgs-acceleration-method.ipynb) in Colab, set the runtime to **T4 GPU**, and run top to bottom. The notebook is glue only — every step is a function in [`pipeline/`](pipeline).

| Step | Notebook cell | Module |
|---|---|---|
| Install deps, check the GPU | 1 | `pipeline/env.py` |
| One config object, everything in one place | 2 | `pipeline/config.py` |
| Download data, list scenes, data profile | 3 | `pipeline/data.py` |
| Smoke test (a few hundred iterations) | 4 | `pipeline/run.py` |
| Train every scene, live `Score` + progress % | 5 | `pipeline/trainer.py`, `pipeline/score.py` |
| Analytics: per-scene tables and plots | 6 | `pipeline/report.py` |
| `submission.zip`, format check, auto-download | 7 | `pipeline/submission.py`, `pipeline/deliver.py` |

Scoring follows the competition definition: `Score = 0.4 (1 - LPIPS) + 0.3 SSIM + 0.3 clamp(PSNR / PSNR_max, 0, 1)`,
averaged over scenes. The submission is `submission.zip` with one folder per scene holding `0001.png`, `0002.png`, ...
at the ground-truth resolution.

To use your own data, point `data_root` at a folder of COLMAP scenes (or set `dataset_url` to a ZIP). For full-quality
results set `iterations=30000`. The FastGS mechanisms themselves are simulated, CUDA-free, in
[`demos/fastgs_mechanisms.py`](demos/fastgs_mechanisms.py).

### Local

```bash
git clone https://github.com/KietAnhCS/fastgs-lite.git
cd fastgs-lite

# Windows only
SET DISTUTILS_USE_SDK=1

conda env create --file environment.yml
conda activate fastgs

pip install ./submodules/diff-gaussian-rasterization_fastgs
pip install ./submodules/fused-ssim
pip install ./submodules/simple-knn
```

Requirements: CUDA-ready GPU with compute capability 7.0+, a C++ compiler compatible with your PyTorch build, and CUDA SDK 11. Upstream's reference setup was conda CUDA 11.6 / system 12.2 / `nvcc` 11.8 — the `nvcc` version is what matters for compiling the extensions.

### Dataset layout

```
datasets/
├── mipnerf360/{bicycle,flowers,garden,...}
├── db/{playroom,drjohnson}
└── tanksandtemples/{truck,train}
```

MipNeRF360 scenes are [hosted by the authors](https://jonbarron.info/mipnerf360/); SfM data for Tanks&Temples and Deep Blending is [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

## Documentation

Deep dives live in [`DOCS/`](DOCS/README.md). **They are written in Vietnamese**; this README is the English entry point.

| Document | Covers | Notebook part |
|---|---|---|
| [DOCS/gaussian-splatting-math.md](DOCS/gaussian-splatting-math.md) | 3DGS foundations: covariance projection, alpha blending, the loss | — |
| [DOCS/fastgs-acceleration-method.md](DOCS/fastgs-acceleration-method.md) | FastGS mechanisms: multi-view consistency score, dual-condition densification, compact box, split SH learning rates | 1–6 |
| [DOCS/colab-t4-guide.md](DOCS/colab-t4-guide.md) | Colab T4 playbook: tuned preset with per-parameter reasoning, the progress score and where its marginal value lies, the anti-OOM rules, three upgrades, CUDA roadmap | 7–9 |
| [DOCS/DIGITAL-TWIN-GS-PIPELINE-{1,2,3}.md](DOCS/README.md) | Archived line-by-line anatomy of an older DroneSplat-era `train.py`. The general 3DGS walkthrough still holds, but **every flag name must be re-checked against the code** | — |

## Reading the code: five flags that do nothing

These are declared in `arguments/__init__.py` but never read anywhere in this codebase. Passing them is silently ignored — no error, no effect.

| Flag | Superseded by | Evidence |
|---|---|---|
| `--antialiasing` | *(nothing — feature absent)* | `gaussian_renderer/__init__.py` builds `GaussianRasterizationSettings` without the field, and the vendored rasterizer never references it |
| `--feature_lr` | `--lowfeature_lr` | `scene/gaussian_model.py:200` reads `lowfeature_lr` for `f_dc` |
| `--shfeature_lr` | `--highfeature_lr` | `scene/gaussian_model.py:205` reads `highfeature_lr` for `f_rest` |
| `--percent_dense` | `--dense` | assigned at `gaussian_model.py:193`, never read again; densification uses `args.dense` |
| `--densify_grad_threshold` | `--grad_thresh` | no reference outside `arguments/__init__.py` |

Two more things the upstream README does not mention:

**The optimizer schedule is hard-wired to a 30,000-iteration budget.** `scene/gaussian_model.py:225` steps Adam every iteration up to 15k, then every 32nd up to 20k, then every 64th. Together with `densify_until_iter = 15000`, all the cost sits in 0–15k while 15k–30k is nearly free but still runs `final_prune_fastgs`. Cutting `--iterations` below 20000 therefore saves little and gives up the final pruning; if you do shorten it, also sync `--position_lr_max_steps`, `--densify_until_iter`, and the two hard-coded thresholds.

**`--densification_interval` is an underused time lever.** It defaults to `100`, but `train_base.sh` uses `500`. Since each densification step renders 10 sampled cameras twice for scoring, that is roughly 29 scoring passes instead of 145 over iterations 500–15000.

## Training and evaluation

Reproduce the paper's benchmark sweep:

```bash
bash train_base.sh   # standard
bash train_big.sh    # higher quality, --mult 0.7 on large scenes
```

Single scene, tuned for a free T4:

```bash
python train.py -s datasets/mipnerf360/counter -m output/counter -i images -r 2 --eval \
  --iterations 30000 \
  --densification_interval 500 \
  --lambda_dssim 0.25 \
  --highfeature_lr 0.02 \
  --loss_thresh 0.07 \
  --grad_abs_thresh 0.0012

python render.py  -m output/counter --skip_train --mult 0.5
python metrics.py -m output/counter
```

`-r 2` for indoor and `-r 4` for outdoor are the standard MipNeRF360 evaluation resolutions, not a compromise. `--mult` must match between `train.py` and `render.py`. Per-scene values for `--grad_abs_thresh`, `--dense` and `--highfeature_lr` are in `train_base.sh` — start from the scene most like yours.

<details>
<summary><b>Full command line reference</b></summary>

#### FastGS-specific

| Flag | Default | Meaning |
|---|---|---|
| `--loss_thresh` | `0.1` | Threshold on the normalized loss map. Lower keeps more Gaussians. `garden` uses `0.06` |
| `--grad_thresh` | `0.0002` | Gradient threshold for clone (as in vanilla 3DGS) |
| `--grad_abs_thresh` | `0.0012` | Absolute-gradient threshold for split (as in Abs-GS) |
| `--dense` | `0.001` | Fraction of scene extent above which a point is split rather than cloned. Outdoor scenes use `0.004`–`0.01` |
| `--highfeature_lr` | `0.005` | LR for high-order SH (`features_rest`). Note `gaussian_model.py:205` divides it by 20, so `0.02` means an effective `0.001` |
| `--lowfeature_lr` | `0.0025` | LR for low-order SH (`features_dc`) |
| `--mult` | `0.5` | Compact-box multiplier controlling how many tiles each splat touches. `0.7` for large or cluttered scenes |
| `--optimizer_type` | `default` | `default` uses the staged Adam cadence; `sparse_adam` steps only visible Gaussians every iteration |

#### Data and model

| Flag | Default | Meaning |
|---|---|---|
| `--source_path` / `-s` | — | COLMAP or synthetic-NeRF dataset directory |
| `--model_path` / `-m` | `output/<random>` | Where the trained model is written |
| `--images` / `-i` | `images` | Image subdirectory inside the COLMAP dataset |
| `--resolution` / `-r` | `-1` | `1`, `2`, `4`, `8` select full, 1/2, 1/4, 1/8 resolution. Any other value rescales width to that number. Unset with width above 1.6K auto-rescales to 1.6K |
| `--eval` | off | Use the MipNeRF360-style train/test split (every 8th image is held out) |
| `--white_background` / `-w` | off | White instead of black background |
| `--data_device` | `cuda` | Use `cpu` for large or high-resolution datasets to cut VRAM at a small speed cost |
| `--sh_degree` | `3` | Spherical harmonics order, at most 3 |

#### Optimization

| Flag | Default | Meaning |
|---|---|---|
| `--iterations` | `30000` | Total iterations. See the note above before lowering this |
| `--lambda_dssim` | `0.2` | Weight of the DSSIM term in the loss |
| `--opacity_lr` | `0.025` | Opacity learning rate |
| `--scaling_lr` | `0.005` | Scaling learning rate |
| `--rotation_lr` | `0.001` | Rotation learning rate |
| `--position_lr_init` | `0.00016` | Initial position learning rate |
| `--position_lr_final` | `0.0000016` | Final position learning rate |
| `--position_lr_delay_mult` | `0.01` | Position LR delay multiplier |
| `--position_lr_max_steps` | `30000` | Steps over which the position LR anneals. Keep equal to `--iterations` |
| `--densify_from_iter` | `500` | Densification starts here |
| `--densify_until_iter` | `15000` | Densification stops here |
| `--densification_interval` | `100` | How often to densify. `train_base.sh` uses `500` |
| `--opacity_reset_interval` | `3000` | How often opacity is reset |
| `--random_background` | off | Randomize the background colour each iteration |

#### Runtime and debugging

| Flag | Default | Meaning |
|---|---|---|
| `--test_iterations` | `[30000]` | Iterations at which the test set is evaluated |
| `--save_iterations` | `[30000]` + `--iterations` | Iterations at which the Gaussian model is written |
| `--checkpoint_iterations` | `[30000]` | Iterations at which a resumable checkpoint is stored |
| `--start_checkpoint` | — | Checkpoint to resume from |
| `--quiet` | off | Suppress stdout |
| `--debug` | off | Dump rasterizer state on failure |
| `--debug_from` | `-1` | Enable debug mode only after this iteration (debugging is slow) |
| `--detect_anomaly` | off | Enable autograd anomaly detection |
| `--convert_SHs_python` | off | Compute SH forward/backward in PyTorch instead of CUDA |
| `--compute_cov3D_python` | off | Compute 3D covariance in PyTorch instead of CUDA |
| `--websockets`, `--ip`, `--port` | off, `127.0.0.1`, `6009` | Live viewer bridge |

</details>

Like MipNeRF360 and vanilla 3DGS, this targets images in the 1–1.6K pixel range. Arbitrary sizes are accepted and downscaled when wider than 1600 px; force full resolution with `-r 1`.

## Viewers

The representation is identical to vanilla 3DGS, so the official [SIBR viewer](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#interactive-viewers) works, as does the browser-based [SuperSplat](https://superspl.at/editor).

## Not included

These upstream task branches are outside this fork's scope. Use the upstream repository for them:

- [Fast-D3DGS](https://github.com/fastgs/FastGS/tree/fast-d3dgs) — dynamic scenes
- [Fast-DropGaussian](https://github.com/fastgs/FastGS/tree/fast-dropgaussian) — sparse-view reconstruction
- [Fast-PGSR](https://github.com/fastgs/FastGS/tree/fast-pgsr) — surface reconstruction

## Acknowledgements

**FastGS is not my work.** This repository is a fork; the method, the CUDA extensions and the original README are by Ren, Wen, Fang and Lu. My contribution here is the Colab T4 packaging, the instrumented training notebook, and the documentation in `DOCS/`. Questions about the method itself belong upstream at **renshiwei@mail.nankai.edu.cn**.

FastGS builds on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [Taming-3DGS](https://github.com/humansensinglab/taming-3dgs), [Speedy-Splat](https://github.com/j-alex-hanson/speedy-splat) and [Abs-GS](https://github.com/TY424/AbsGS), with thanks to the authors of [DashGaussian](https://github.com/YouyuChen0207/DashGaussian).

**License**: follow the licenses of 3DGS, Taming-3DGS and Speedy-Splat. See [LICENSE](LICENSE) and [LICENSE_ORIGINAL.md](LICENSE_ORIGINAL.md).

## Citation

```bibtex
@article{ren2025fastgs,
  title={FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  author={Ren, Shiwei and Wen, Tianci and Fang, Yongchun and Lu, Biao},
  journal={arXiv preprint arXiv:2511.04283},
  year={2025}
}
```
