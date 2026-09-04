<div align="center">

# fastgs-lite

**A trimmed [FastGS](https://github.com/fastgs/FastGS) fork that trains 3D Gaussian Splatting on a single free Google Colab T4.**

</div>

---

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

## Results

A complete reference run of the notebook on free-tier Colab. Every number below is read from the run's own
artifacts — [`DOCS/assets/leaderboard.csv`](DOCS/assets/leaderboard.csv) (final metrics),
[`DOCS/assets/history.csv`](DOCS/assets/history.csv) (training dynamics), and `fastgs_models/<scene>/cfg_args`
(the exact `Namespace` each scene was trained with) — and nothing is rounded by hand. Both CSVs and the two
figures below (2.2 MB) are committed so the tables can be re-derived; the models and renders are not (see
[Artifacts](#artifacts)). A narrative version with per-session commentary is in
[DOCS/history-train.md](DOCS/history-train.md) (Vietnamese).

### Experimental setup

| | |
|---|---|
| Accelerator | NVIDIA Tesla T4, 14.56 GB VRAM (Colab free tier) |
| Host | 12.7 GB system RAM |
| Software | PyTorch 2.11.0+cu128 · CUDA 12.8 · Python 3.13.15 |
| Datasets | Deep Blending (`drjohnson`, `playroom`) and Tanks&Temples (`train`, `truck`), from `tandt_db.zip` |
| Split | `--eval` with `llffhold=8`: every 8th camera held out. Deterministic, no sampling |
| Runner | `pipeline/run.py` (`setup → load_data → smoke_test → run_all → analytics → finish`) |
| Date | 2026-09-04 |

Scene sizes after the split, from `cameras.json` and the submission folders:

| Scene | Cameras | Train views | Test views | Evaluation resolution |
|---|---|---|---|---|
| drjohnson | 263 | 230 | 33 | 1332×876 |
| playroom | 225 | 196 | 29 | 1264×832 |
| train | 301 | 263 | 38 | 980×545 |
| truck | 251 | 219 | 32 | 979×546 |

Optimization was identical across scenes (`cfg_args`, abridged):

```
iterations=7000            position_lr_max_steps=7000    sh_degree=3
resolution=2               submission_resolution=1       lambda_dssim=0.25
densification_interval=500 densify_from_iter=500         densify_until_iter=15000
opacity_reset_interval=3000 grad_abs_thresh=0.0012       grad_thresh=0.0002
dense=0.001                highfeature_lr=0.02           lowfeature_lr=0.0025
mult=0.5                   optimizer_type='default'      separate_sh=True
```

Note the **training/evaluation resolution mismatch**: models were fit at `resolution=2` but scored on renders at
the native resolution (`submission_resolution=1`). This is the competition protocol, not an oversight, and it is
the single largest known handicap in this run.

### Metric

`pipeline/score.py` implements the competition definition

```
Score = 0.4 (1 − LPIPS) + 0.3 SSIM + 0.3 clamp(PSNR / PSNR_max, 0, 1),    PSNR_max = 30 dB
```

reported as the unweighted mean over scenes. Two distinct measurement protocols appear in the artifacts and must
not be compared with each other:

| | `live_*` columns | reported columns |
|---|---|---|
| When | every 1000 iterations, during training | after training, on the final model |
| Views | 6 sampled test views (`eval_views=6`) | all test views (29–38) |
| Resolution | `resolution=2` | native |
| LPIPS backbone | AlexNet (`lpips_net_live`) | VGG (`lpips_net_report`) |

### Table 1 — Final quality

Reported columns of `leaderboard.csv`. LPIPS is VGG-based at native resolution; lower is better.

| Scene | Score ↑ | PSNR (dB) ↑ | PSNR_norm | SSIM ↑ | LPIPS ↓ | Gaussians |
|---|---|---|---|---|---|---|
| playroom | **0.8198** | 28.457 | 0.9486 | 0.8799 | 0.3219 | 129,383 |
| drjohnson | **0.8027** | 27.408 | 0.9136 | 0.8677 | 0.3293 | 174,003 |
| truck | **0.7482** | 22.231 | 0.7410 | 0.7851 | **0.2742** | 187,287 |
| train | **0.6867** | 19.755 | 0.6585 | 0.7241 | 0.3202 | 201,129 |
| **Mean** | **0.7643** | **24.463** | 0.8154 | 0.8142 | 0.3114 | 172,951 |

![Per-scene Score against the 0.7644 mean, and the three metric components side by side](DOCS/assets/leaderboard.png)

> **Figure 1.** *Left:* final Score per scene, dashed line at the 0.7644 mean. *Right:* the three components that
> enter the Score — `psnr_norm` (blue), `SSIM` (orange), `LPIPS` (green, lower is better). The component view is
> what makes the indoor/outdoor split legible: LPIPS bars are nearly level across all four scenes, while
> `psnr_norm` drops sharply on `train` and `truck`. Axis labels are Vietnamese, matching `DOCS/`.

The two indoor Deep Blending scenes lead the two outdoor Tanks&Temples scenes by 0.06–0.13 Score **while using
fewer Gaussians** (129k vs 201k). The gap is carried entirely by the PSNR term — `PSNR_norm` is 0.66 for `train`
against 0.95 for `playroom` — whereas `truck` has the best LPIPS in the table (0.2742). Fidelity of fine texture is
not what separates these scenes; absolute radiometric accuracy on large-depth outdoor geometry is.

For reference, the pipeline's smoke test on `drjohnson` at 300 iterations already scores 0.6012 under the live
protocol (PSNR 20.93, SSIM 0.7073, LPIPS 0.5506) — 67% of that scene's final live Score of 0.8966 in 4.3% of the
iteration budget, consistent with the multi-view score front-loading Gaussian placement, though this repository contains no ablation isolating that mechanism.

### Qualitative results

Held-out test views, produced by `report.show_samples(cfg, scene, n=3)`. **Top row: render. Bottom row: ground
truth.** Views are sampled evenly across each scene's test set, not hand-picked. The two scenes shown are the
extremes of Table 1 — the strongest indoor result and the weakest outdoor one.

![drjohnson: three rendered test views above their ground-truth counterparts](DOCS/assets/samples_drjohnson.png)

> **Figure 3.** `drjohnson` (Score 0.8027, PSNR 27.41). Geometry, pose and colour hold up: the first view is
> near-indistinguishable from its ground truth. The residual errors are all high-frequency or view-dependent —
> the radiator fins and the bookcase glazing bars soften, book spines lose their separation, and the specular
> highlight on the mahogany cabinet (bright in the ground truth, third column) is largely missing, since
> view-dependent lobes are exactly what the high-order SH coefficients encode and `highfeature_lr` is divided by
> 20 in `gaussian_model.py:205`. Corners also blur more than centres. This is what LPIPS ≈ 0.33 looks like.

![train: three rendered test views above their ground-truth counterparts](DOCS/assets/samples_train.png)

> **Figure 4.** `train` (Score 0.6867, PSNR 19.75) — the weakest scene, and the failure is unmistakable: **the
> sky**. Ground truth is clean, uniform blue; the renders fill it with grey-white streaks and blotches. The
> locomotive itself is reconstructed well, and so is anything with parallax to lock onto, but a textureless
> region at effectively infinite depth gives the multi-view score nothing to disagree about, so stray Gaussians
> survive there. Ballast gravel and the "WESTERN PACIFIC" lettering also flatten, and the third view smears
> toward the frame edge.
>
> This is precisely why the deficit lands in `psnr_norm` (0.66 vs `drjohnson`'s 0.91) and not in LPIPS, which is
> nearly identical for the two scenes (0.3202 vs 0.3293): sky occupies a large fraction of the pixels, so its
> error dominates a per-pixel metric like PSNR, while a perceptual metric weights it far less. It also suggests
> the cheapest available fix here is a larger `--dense` (`train_base.sh` gives `train` itself `0.01`, against the
> `0.001` used in this run) rather than a longer schedule. Note `--dense` does not split cleanly along indoor/outdoor
> in `train_base.sh` — `drjohnson` is indoor and uses `0.013`, the highest value in the file.

Regenerate these, or the two scenes not shown, with one call per scene:

```python
report.show_samples(cfg, "playroom", n=3, save_to="samples_playroom.png")
```

### Table 2 — Training dynamics

Live Score at each checkpoint (`history.csv`), all four scenes:

| Iteration | drjohnson | playroom | train | truck |
|---|---|---|---|---|
| 1000 | 0.6578 | 0.6991 | 0.6360 | 0.7596 |
| 2000 | 0.7598 | 0.7909 | 0.7102 | 0.8213 |
| **3000** | *0.2954* | *0.2315* | *0.2324* | *0.2264* |
| 4000 | 0.8489 | 0.8949 | 0.7863 | 0.8634 |
| 5000 | 0.8818 | 0.9158 | 0.7993 | 0.8804 |
| **6000** | *0.3116* | *0.2523* | *0.2424* | *0.2304* |
| 7000 | **0.8966** | **0.9167** | **0.8123** | **0.8821** |

![Four panels: Score per iteration, the three metric components, per-checkpoint deltas, and Score against Gaussian count](DOCS/assets/training.png)

> **Figure 2.** All four scenes, every 1000 iterations. *Top left:* Score — the two V-shaped notches at 3000 and
> 6000 are opacity resets, identical in phase across scenes. *Top right:* PSNR in dB alongside SSIM and LPIPS
> rescaled ×30 to share the axis. *Bottom left:* ΔScore between consecutive checkpoints, showing the −0.6 drop
> and the matching +0.6 recovery. *Bottom right:* Score against Gaussian count — the trajectory moves right
> monotonically (densification never stops before 15000) while Score oscillates, so the horizontal excursions at
> low Score are the reset artifacts, not a quality/size trade-off curve.

The collapses at 3000 and 6000 (PSNR falls to 6.0–11.1 dB on all four scenes simultaneously) are a **measurement
artifact, not divergence**. `opacity_reset_interval = 3000` makes `reset_opacity()` fire exactly on those
iterations, and `score_every = 1000` samples the model in the same iteration, before any recovery step. Full
recovery takes fewer than 1000 iterations in every case.

Two practical consequences:

- Do not set `iterations` to a multiple of 3000 while densification is still active — the run would terminate on
  a reset. `iterations = 7000` lands 1000 steps after the last reset, so the delivered model is intact.
  Budgets of 15000+ are unaffected because `densify_until_iter = 15000` closes that branch.
- Offsetting `score_every` (e.g. 1100) yields a monotone curve without changing the optimization.

### Table 3 — Cost and storage

| Scene | Train (s) | Iter/s | Peak VRAM (GB) | Peak RAM (GB) | `point_cloud.ply` | Bytes/Gaussian |
|---|---|---|---|---|---|---|
| playroom | 67.2 | 104.2 | 0.89 | 9.59 | 32.1 MB | 248 |
| drjohnson | 74.8 | 93.6 | **1.16** | 3.74 | 43.2 MB | 248 |
| truck | 91.0 | 76.9 | 0.68 | 9.59 | 46.4 MB | 248 |
| train | 94.6 | 74.0 | 0.78 | 9.59 | 49.9 MB | 248 |
| **Total / mean** | **327.6** | 85.5 | 0.88 | — | 171.6 MB | 248 |

End-to-end wall clock was **14 minutes**, of which 5.5 minutes is optimization; the remainder is the one-time
CUDA-extension build (~6–7 min, skipped on later sessions via `/content/.deps_ok`), dataset download, native-
resolution rendering of 132 test images, and VGG-LPIPS scoring.

248 B/Gaussian is exactly the uncompressed 3DGS record — 62 float32 fields (3 position, 3 normal, 3 `f_dc`,
45 `f_rest`, 1 opacity, 3 scale, 4 rotation) — confirming no compression is applied at save time.

### Artifacts

`run.finish` produces two archives and downloads them to the machine running the browser:

| Artifact | Contents | Size | In git? |
|---|---|---|---|
| `submission.zip` → `submission/` | 132 PNGs at native resolution, `<scene>/0001.png…` contiguous | 82 MB | no |
| `fastgs_models.zip` → `fastgs_models/` | 4 × (`point_cloud.ply`, `cameras.json`, `cfg_args`) plus the CSVs and figures | 165 MB | no |
| [`DOCS/assets/`](DOCS/assets) | `leaderboard.csv`, `history.csv` and the four figures above | 2.2 MB | **yes** |

Only the evidence needed to re-derive every table and figure above is committed. The 247 MB of renders and point
clouds is deliberately excluded — both folders are in `.gitignore`, because a git object is permanent: adding a
`.ply` per run would grow the clone size of the repository forever, for every user, even after deletion. Publish
those through GitHub Releases, Hugging Face, or Drive instead, and link them. To regenerate them locally, run the
notebook with the config under [Reproduction](#reproduction).

`pipeline/submission.py` validated scene names, filename contiguity, per-scene image counts and image sizes:
`problems = []`.

### Observations

**1. At 7000 iterations, `final_prune_fastgs` never executes.** Its guard is `15_000 < iteration < 30_000`
(`pipeline/trainer.py`), so a 7k run delivers *un-pruned* models. This is the most likely explanation for both
the storage figures above and the LPIPS floor around 0.31: the third pruning stage that FastGS relies on for its
compactness claims was never reached. Any comparison of these numbers against published FastGS or 3DGS results
is therefore invalid — those use the 30k budget.

**2. VRAM is not the constraint on a T4; host RAM is.** Peak VRAM was 1.16 GB, 8.0% of the 14.56 GB available,
while host RAM reached 9.59 GB of 12.7 GB — 91% of the pipeline's own 10.5 GB soft limit. The guidance in
[DOCS/colab-t4-guide.md](DOCS/colab-t4-guide.md) to lower resolution for VRAM reasons is unnecessary at this
budget; `resolution=1` is affordable and would remove the train/eval mismatch.

**3. The `live_*` → reported drop (0.8769 → 0.7643) is protocol, not regression.** Decomposed: LPIPS rises
0.1140 → 0.3114 (AlexNet → VGG, and half → native resolution) and PSNR falls 26.23 → 24.46 dB (evaluation at
native resolution on a model fit at half resolution). SSIM moves least (0.8675 → 0.8142). Cross-session
comparisons must hold the protocol fixed.

**4. Internal consistency.** The reported Score reproduces exactly from its components, e.g. for `drjohnson`:
`0.4(1 − 0.3293) + 0.3(0.8677) + 0.3(0.9136) = 0.8027`, and the mean Score is the unweighted mean of the four
scene Scores (0.76435). No post-hoc weighting was applied.

### Reproduction

```python
cfg = Config(data_root="/content/data", scenes=(), resolution=2, iterations=7000,
             score_every=1000, eval_views=6, psnr_max=30.0, submission_resolution=1)
```

then `Runtime → Run all` on a T4. `pipeline/config.py` supplies the FastGS flags
(`--densification_interval 500 --lambda_dssim 0.25 --highfeature_lr 0.02 --loss_thresh 0.07
--grad_abs_thresh 0.0012`) and `pipeline/trainer.py:26` syncs `--position_lr_max_steps` to `iterations`.

**Threats to validity.** This is a *single* run: no seed is fixed, camera order within each epoch is shuffled
randomly, and no repeats were performed, so no variance estimate or error bars are available and small
differences between scenes should not be over-read. Four scenes from two datasets is a narrow sample. The 7k
budget is a third of the schedule the optimizer cadence in `scene/gaussian_model.py:225` is written for, and as
noted above it skips final pruning entirely. Treat Table 1 as a **reproducible baseline for this repository**,
not as a benchmark result.

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

`-r 2` for indoor and `-r 4` for outdoor are the standard MipNeRF360 evaluation resolutions, not a compromise. `--mult` must match between `train.py` and `render.py`. Per-scene values for `--grad_abs_thresh`, `--dense` and `--highfeature_lr` are in `train_base.sh` — copy the line for the scene most like yours rather than guessing from an indoor/outdoor rule.

<details>
<summary><b>Full command line reference</b></summary>

#### FastGS-specific

| Flag | Default | Meaning |
|---|---|---|
| `--loss_thresh` | `0.1` | Threshold on the normalized loss map. Lower keeps more Gaussians. `garden` uses `0.06` |
| `--grad_thresh` | `0.0002` | Gradient threshold for clone (as in vanilla 3DGS) |
| `--grad_abs_thresh` | `0.0012` | Absolute-gradient threshold for split (as in Abs-GS) |
| `--dense` | `0.001` | Fraction of scene extent above which a point is split rather than cloned. `train_base.sh` uses `0.003`–`0.013` per scene; the value does not track indoor/outdoor |
| `--highfeature_lr` | `0.005` | LR for high-order SH (`features_rest`). `gaussian_model.py:205` divides it by 20, so `0.02` means an effective `0.001` — still below `--lowfeature_lr`. Its optimizer also steps only every 16th iteration up to 15k |
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
