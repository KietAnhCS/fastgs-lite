# DIGITAL TWIN GS PIPELINE (3/3) — Lưu, render, chấm điểm, submission

> Phần cuối của bộ ba tài liệu tham chiếu, đi từ lúc `Scene.save`/`GaussianModel.save_ply` ghi
> `point_cloud.ply` xuống đĩa, qua hai đường render (`render.py` CLI và
> `pipeline/submission.py::render_scene` dùng để nộp bài), tới cách chấm điểm
> (`pipeline/score.py`, `lpipsPyTorch`, PSNR/SSIM) và đóng gói `submission.zip` đúng hợp đồng
> cuộc thi. Mọi số liệu lấy trực tiếp từ mã nguồn hiện tại trong repo — không phải bản
> "DroneSplat" cũ (`save_pose`, `render_video.py`, `scripts/make_gif.py`,
> `scripts/check_save_ram.py`, `metrics.json` **không tồn tại** trong codebase này).

---

## MỤC LỤC

**PHẦN VI — LƯU MÔ HÌNH VÀ RENDER**

- [34. `Scene.save` → `save_ply` — bố cục cột của một Gaussian](#34)
- [35. Cây thư mục `output/<scene>/` sau khi train xong](#35)
- [36. Checkpoint định kỳ trong `pipeline/trainer.py`](#36)
- [37. `render.py` — đường CLI](#37)
- [38. `Scene.__init__` nhánh `load_iteration` → `load_ply`](#38)
- [39. `pipeline.submission.render_scene` — đường submission](#39)
- [40. `build_zip` và `verify` — hợp đồng nộp bài](#40)

**PHẦN VII — CHẤM ĐIỂM VÀ PHỤ LỤC**

- 41. `metrics.py`
- 42. `lpipsPyTorch`
- 43. PSNR và SSIM
- 44. `composite_score` và bảng leaderboard
- 45. Bảng hằng số toàn hệ thống
- 46. Bảng tra nhanh bước ↔ file ↔ hàm
- 47. Chẩn đoán sự cố
- 48. Thuật ngữ

---

# PHẦN VI — LƯU MÔ HÌNH VÀ RENDER

## 34. `Scene.save` → `save_ply` — bố cục cột của một Gaussian

`Scene.save(iteration)` (`scene/__init__.py:83-85`):

```python
def save(self, iteration):
    point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
```

Chỉ ghép đường dẫn `point_cloud/iteration_<n>/point_cloud.ply` rồi gọi thẳng
`GaussianModel.save_ply` (`scene/gaussian_model.py:260-277`).

### `construct_list_of_attributes` (`scene/gaussian_model.py:246-258`)

Danh sách tên cột theo đúng thứ tự được sinh ra:

```python
def construct_list_of_attributes(self):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(self._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(self._rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l
```

`_features_dc` có shape `(P, 1, 3)` (1 hệ số SH bậc 0 × 3 kênh màu RGB), `_features_rest` có
shape `(P, (max_sh_degree+1)²-1, 3)`. Với `sh_degree = 3` (giá trị mặc định của
`ModelParams.sh_degree`, dùng xuyên suốt pipeline này):

| Nhóm cột | Công thức | Số lượng khi `sh_degree=3` |
|---|---|---|
| `x, y, z` | vị trí | 3 |
| `nx, ny, nz` | pháp tuyến — **luôn ghi 0**, xem dưới | 3 |
| `f_dc_0..2` | `_features_dc.shape[1]*shape[2]` = 1×3 | 3 |
| `f_rest_0..44` | `_features_rest.shape[1]*shape[2]` = 15×3, với 15 = (3+1)²−1 | 45 |
| `opacity` | 1 giá trị opacity thô (chưa qua sigmoid) | 1 |
| `scale_0..2` | `_scaling.shape[1]` | 3 |
| `rot_0..3` | `_rotation.shape[1]` (quaternion) | 4 |
| **Tổng** | 3+3+3+45+1+3+4 | **62 cột float** |

Vậy mỗi Gaussian ở `sh_degree=3` chiếm **62 giá trị `float32` (`f4`)** trong PLY, tức
`62 × 4 = 248 byte/vertex` (chưa tính header text của PLY). Con số 45 cho `f_rest` khớp với
assertion phía đọc lại: `load_ply` yêu cầu
`len(extra_f_names) == 3*(max_sh_degree+1)**2 - 3 = 3*16-3 = 45` (`scene/gaussian_model.py:299`).

### Cách ghi (`save_ply`, dòng 260-277)

```python
def save_ply(self, path):
    mkdir_p(os.path.dirname(path))
    xyz = self._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = self._opacity.detach().cpu().numpy()
    scale = self._scaling.detach().cpu().numpy()
    rotation = self._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
```

Ghi chú quan trọng:

- `normals = np.zeros_like(xyz)` — cột `nx, ny, nz` **luôn bằng 0**, không phải pháp tuyến bề
  mặt thật; đây là hành vi gốc của 3DGS (Gaussian không có khái niệm pháp tuyến), giữ cột chỉ
  để tương thích định dạng PLY tiêu chuẩn (nhiều viewer/loader mong có `nx,ny,nz`).
- `.transpose(1, 2)` trên `_features_dc`/`_features_rest` đổi thứ tự trục trước khi
  `.flatten(start_dim=1)`, để khi ghi ra thứ tự cột là "duyệt theo hệ số SH trước, kênh màu
  sau" — đúng thứ tự mà `load_ply` mong đợi khi đọc ngược lại (xem §38).
- `opacities`, `scale`, `rotation` được ghi **thô**, chưa qua activation (`sigmoid` cho
  opacity, `exp` cho scale, chuẩn hoá cho rotation) — activation chỉ áp dụng lúc dùng
  (`get_opacity`, `get_scaling`, `get_rotation`), không áp dụng lúc lưu.
- `mkdir_p(os.path.dirname(path))` tạo cây thư mục `point_cloud/iteration_<n>/` nếu chưa có.
- `PlyElement.describe(elements, 'vertex')` + `PlyData([el]).write(path)` (thư viện `plyfile`)
  ghi một element PLY tên `vertex`, mặc định ở dạng **binary** (little-endian trên hầu hết máy)
  vì không truyền `text=True` — mỗi vertex là 248 byte liên tiếp theo đúng thứ tự cột ở trên.

## 35. Cây thư mục `output/<scene>/` sau khi train xong

Ví dụ minh hoạ cho một scene tên `truck`, train xong ở `iteration=7000` với `save_every=2000`
đã dọn hết checkpoint trung gian (`keep_last_checkpoint=True`, mặc định):

```
output/
├── truck/
│   ├── cfg_args                              # ghi bởi build_args (pipeline/trainer.py:20-32)
│   ├── input.ply                             # copy nguyên point cloud COLMAP/Blender ban đầu
│   ├── cameras.json                          # toàn bộ camera (test rồi tới train) dạng JSON
│   └── point_cloud/
│       └── iteration_7000/
│           └── point_cloud.ply               # Scene.save(7000) — bản cuối cùng
├── history.csv                               # ghi bởi pipeline.run.analytics → report.history_frame
├── leaderboard.csv                           # ghi bởi pipeline.run.analytics → report.leaderboard
├── training.png                              # ghi bởi pipeline.run.analytics → report.plot_training
├── leaderboard.png                           # ghi bởi pipeline.run.analytics → report.plot_leaderboard
└── results.json                              # ghi bởi pipeline.run.run_all (không phải analytics)
```

Ai ghi file nào, khi nào:

| File | Ghi bởi | Thời điểm |
|---|---|---|
| `output/<scene>/cfg_args` | `pipeline.trainer.build_args` (nếu `write_cfg=True`, mặc định khi gọi từ `train_scene`) — hoặc `train.py::prepare_output_and_logger` khi chạy CLI gốc | Ngay khi bắt đầu train scene đó, trước vòng lặp |
| `output/<scene>/input.ply`, `cameras.json` | `Scene.__init__` (`scene/__init__.py:50-60`), chỉ khi `load_iteration` không được truyền (tức đang train mới, không phải render lại) | Ngay khi `Scene` khởi tạo, trước vòng lặp train |
| `output/<scene>/point_cloud/iteration_<n>/point_cloud.ply` | `Scene.save` qua `pipeline.trainer._save_checkpoint` | Mỗi `save_every` vòng + vòng cuối (§36) |
| `output/history.csv` | `pipeline.report.history_frame` gọi từ `pipeline.run.analytics` | Sau khi `run_all` train xong toàn bộ scene, ở bước "cell 6" của notebook |
| `output/leaderboard.csv` | `pipeline.report.leaderboard` gọi từ `pipeline.run.analytics` | Cùng bước trên |
| `output/training.png` | `pipeline.report.plot_training` gọi từ `pipeline.run.analytics` | Cùng bước trên |
| `output/leaderboard.png` | `pipeline.report.plot_leaderboard` gọi từ `pipeline.run.analytics` | Cùng bước trên |
| `output/results.json` | `pipeline.run.run_all` trực tiếp (`pipeline/run.py:63-65`), **không phải** `analytics` | Ngay sau vòng `for scene in scenes` train+render xong hết, trước khi gọi `analytics` |

Lưu ý: `history.csv`, `leaderboard.csv`, `training.png`, `leaderboard.png`, `results.json` nằm
ở **`output_root` gốc** (`cfg.output_root`, ví dụ `/content/output`), không nằm bên trong từng
thư mục `<scene>/` — vì `cfg.model_path(scene) = os.path.join(cfg.output_root, scene)`
(`pipeline/config.py:65-66`) trong khi các file phân tích được ghi thẳng vào
`os.path.join(cfg.output_root, "history.csv")` v.v. (`pipeline/run.py:71-75`).

Nếu `save_every` khác 0 và scene chưa train xong (checkpoint giữa chừng còn tồn tại vì phiên
bị ngắt), cây thư mục sẽ có thêm `point_cloud/iteration_<k>/` với `k` là mốc checkpoint gần
nhất — xem §36.

## 36. Checkpoint định kỳ trong `pipeline/trainer.py`

`_save_checkpoint` (`pipeline/trainer.py:35-42`):

```python
def _save_checkpoint(scene_obj, iteration, previous=None, drop_previous=True):
    """Lưu `.ply` rồi xoá checkpoint giữa chừng trước đó (đĩa Colab không rộng)."""
    import shutil

    scene_obj.save(iteration)
    if drop_previous and previous is not None and previous != iteration:
        stale = os.path.join(scene_obj.model_path, "point_cloud", f"iteration_{previous}")
        shutil.rmtree(stale, ignore_errors=True)
    return iteration
```

Được gọi ở hai chỗ trong `train_scene`:

1. **Trong vòng lặp**, dòng `if cfg.save_every and iteration % cfg.save_every == 0 and iteration < iterations:` — với `cfg.save_every` mặc định `2000` (`pipeline/config.py`). Đặt `save_every=0` thì điều kiện `cfg.save_every` (falsy) luôn sai → **tắt hẳn checkpoint giữa chừng**, chỉ còn lưu ở vòng cuối.
2. **Sau vòng lặp**, luôn luôn gọi một lần cuối `_save_checkpoint(scene_obj, iterations, saved_at, cfg.keep_last_checkpoint)` — bất kể `save_every` là bao nhiêu, đảm bảo luôn có bản `.ply` tại `iteration_<iterations>`.

`keep_last_checkpoint` (mặc định `True`) truyền vào tham số `drop_previous`: nếu bật, mỗi lần
lưu mốc mới sẽ `shutil.rmtree` xoá thư mục `iteration_<previous>` — chỉ giữ **một** bản
`.ply` trên đĩa tại một thời điểm (đĩa Colab không rộng). Đặt `False` để giữ mọi mốc, phục vụ
so sánh chất lượng theo iteration, đổi lại tốn đĩa hơn.

**Giới hạn phải nêu rõ:** `Scene.save`/`save_ply` chỉ ghi tham số Gaussian
(`_xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation`) — **không** ghi trạng
thái optimizer Adam (`exp_avg`, `exp_avg_sq`), không ghi bộ đếm densify
(`xyz_gradient_accum`, `denom`, `max_radii2D`), không ghi số iteration đã chạy. Khác với
`torch.save(checkpoint)` kiểu gốc 3DGS (`(model_params, first_iter)`, dùng bởi
`train.py --checkpoint_iterations` + `--start_checkpoint`), file `.ply` này **không resumable**
— nếu phiên Colab đứt và bạn nạp lại `.ply` để train tiếp, thực chất là train lại từ đầu với
Gaussian ban đầu đã có nhiều điểm hơn, không phải resume đúng nghĩa optimizer state.

## 37. `render.py` — đường CLI

Đây là script gốc 3DGS, độc lập với `pipeline/`, dùng khi muốn render lại một model đã train
mà không qua Colab pipeline.

### Đọc lại `cfg_args`: `get_combined_args`

`get_combined_args` (`arguments/__init__.py:106-126`):

```python
def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        ...
        cfgfile_string = cfg_file.read()
    except TypeError:
        ...
    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
```

Đọc file text `cfg_args` (được `prepare_output_and_logger` hoặc `pipeline.trainer.build_args`
ghi bằng `str(Namespace(**vars(args)))`), `eval()` chuỗi đó thành lại một `Namespace`, rồi
**đè** bằng bất kỳ cờ nào người dùng truyền thêm trên dòng lệnh (chỉ đè nếu giá trị command
line khác `None`). Nhờ vậy chỉ cần `--model_path` + `--iteration` là đủ, không phải gõ lại
`-s`, `-r`, `--eval`, ... như lúc train.

### Cờ dòng lệnh của `render.py` (dòng 66-73)

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--iteration` | `-1` | chọn mốc; `render_sets` không tự resolve `-1` thành "mới nhất" — việc đó nằm ở `Scene.__init__` khi gọi `searchForMaxIteration` (xem §38) |
| `--skip_train` | `False` (action `store_true`) | bỏ qua render tập train |
| `--skip_test` | `False` | bỏ qua render tập test |
| `--quiet` | `False` | truyền vào `safe_state` |
| `--mult` | `0.5` | hệ số compact-box fastgs-lite truyền thẳng vào `render_fastgs` |

### `render_sets` → `render_set`

`render_sets` (dòng 49-61) tạo `GaussianModel`, `Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)`, dựng `background` từ `dataset.white_background`, rồi gọi `render_set` cho `"train"` (nếu không `--skip_train`) và `"test"` (nếu không `--skip_test`).

`render_set` (dòng 26-44):

```python
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    ...
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render_fastgs(view, gaussians, pipeline, background, args.mult)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    ...
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")
```

Ghi ra `<model_path>/test/ours_<iteration>/renders/00000.png, 00001.png, ...` (đệm **5 chữ
số**, `{0:05d}`, khác với submission dùng 4 chữ số — xem §40) và ảnh ground-truth tương ứng ở
`.../gt/`. Số thứ tự `idx` là chỉ số duyệt trong `views` — **không** sắp theo `image_name` như
`render_scene` submission làm (§39); thứ tự phụ thuộc `Scene.__init__` (không shuffle vì
`shuffle=False`, nên giữ nguyên thứ tự do `sceneLoadTypeCallbacks` trả về). Cuối cùng in FPS
trung bình = `1 / (tổng thời gian render / số ảnh)`.

`render.py` **không tính PSNR/SSIM/LPIPS** — nó chỉ ghi ảnh ra đĩa; việc chấm điểm off-line
trên các thư mục `renders`/`gt` này (kiểu `metrics.py::process_folders` của bản cũ) **không
tồn tại** trong codebase hiện tại. Điểm số được tính ở nơi khác: `pipeline/score.py` (trong
lúc train) và `pipeline/submission.py::render_scene` (lúc render nộp bài) — xem PHẦN VII.

## 38. `Scene.__init__` nhánh `load_iteration` → `load_ply`

Khi gọi `Scene(dataset, gaussians, load_iteration=<n hoặc -1>, shuffle=False)`
(`scene/__init__.py:23-81`):

```python
if load_iteration:
    if load_iteration == -1:
        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
    else:
        self.loaded_iter = load_iteration
```

`searchForMaxIteration` (`utils/system_utils.py:26-28`):

```python
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
```

Liệt kê mọi thư mục con của `point_cloud/` (dạng `iteration_<n>`), tách số sau dấu `_` cuối
cùng, lấy giá trị lớn nhất — nếu `point_cloud/` không tồn tại hoặc rỗng thì `os.listdir` /
`max([])` ném lỗi (không có xử lý ngoại lệ nào ở đây).

Vì `load_iteration` khác `None`/`0`, khối ghi `input.ply` + `cameras.json` (dòng 50-60) **bị
bỏ qua** — đúng như mô tả trong §35 rằng những file đó chỉ ghi khi train mới.

Cuối `__init__`, do `self.loaded_iter` truthy:

```python
if self.loaded_iter:
    self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud",
                                          "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
else:
    self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
```

### `load_ply` (`scene/gaussian_model.py:284-325`) đọc lại 62 cột

Đọc `x,y,z` → `xyz`; `opacity` → `opacities`; ba cột `f_dc_0..2` → `features_dc` shape
`(P,3,1)`; lọc mọi property tên bắt đầu `f_rest_`, sắp theo số hậu tố tăng dần
(`sorted(..., key=lambda x: int(x.split('_')[-1]))`), rồi:

```python
assert len(extra_f_names) == 3*(self.max_sh_degree + 1) ** 2 - 3   # = 45 khi sh_degree=3
features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
```

— đúng 45 cột như tính ở §34. Tương tự lọc `scale_*` và `rot*` (chú ý `rot*` không có dấu
gạch dưới bắt buộc ngay sau — `startswith("rot")` khớp cả `rotation` nếu có, nhưng ở đây chỉ
có `rot_0..rot_3`). Tổng số cột đọc lại: 3(xyz) + 1(opacity) + 3(f_dc) + 45(f_rest) + 3(scale)
+ 4(rot) = **59 giá trị được đọc trực tiếp**, cộng 3 cột `nx,ny,nz` bị **bỏ qua hoàn toàn**
(không có dòng nào đọc `plydata.elements[0]["nx"]`) — khớp với việc chúng luôn là 0 lúc ghi.
Vậy 59 + 3 (normals không đọc) = 62 cột tổng cộng trong file, đúng số cột đã ghi ở §34.

Sáu tensor được gán lại làm `nn.Parameter` mới, `requires_grad_(True)`, trên `"cuda"`:
`_xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation`. Chú ý
`_features_dc`/`_features_rest` được `.transpose(1, 2)` lại **sau khi** `torch.tensor(...)` —
đảo ngược đúng phép `.transpose(1, 2)` đã làm lúc `save_ply`, để khôi phục shape gốc
`(P, 1, 3)` / `(P, 15, 3)` (kênh màu ở trục cuối) từ shape lưu trên đĩa `(P, 3, 1)` / `(P, 3, 15)`.

Dòng cuối:

```python
self.active_sh_degree = self.max_sh_degree
```

Vì file `.ply` đã lưu đủ toàn bộ hệ số SH tới bậc tối đa (được ghi ra bất kể
`active_sh_degree` lúc train là bao nhiêu — `construct_list_of_attributes` luôn dùng full
shape của `_features_rest`), khi load lại để **render/suy luận** không cần tăng dần SH degree
như lúc train (`oneupSHdegree` mỗi 1000 vòng) — mô hình đã "chín", nên set thẳng
`active_sh_degree = max_sh_degree` để dùng toàn bộ chi tiết màu sắc góc nhìn ngay từ đầu.

## 39. `pipeline.submission.render_scene` — đường submission

`render_scene(cfg, scene, iterations=None, score=True)` (`pipeline/submission.py:21-83`) là
đường render dùng để **nộp bài**, khác hẳn `render.py` ở trên. Đi theo đúng thứ tự trong code:

1. **`iterations = int(iterations or cfg.iterations)`** — mặc định dùng `cfg.iterations` nếu không truyền.
2. **`build_args(cfg, scene, iterations=iterations, resolution=cfg.submission_resolution, write_cfg=False)`** (`pipeline/trainer.py:14-32`) — dựng lại `Namespace` tham số 3DGS cho scene này:
   - `resolution=cfg.submission_resolution` (mặc định `1`) ghi đè `cfg.resolution` (mặc định `2` lúc train) → **render đúng độ phân giải gốc**, không downscale như lúc train.
   - `write_cfg=False` → **không ghi đè** `cfg_args` đã có từ lúc train (tránh làm hỏng file cấu hình gốc chỉ vì render submission ở độ phân giải khác).
3. **`GaussianModel(dataset.sh_degree, optimizer_type="default")`** rồi **`Scene(dataset, gaussians, load_iteration=iterations, shuffle=False)`** — nạp đúng `.ply` tại `iteration_<iterations>` theo cơ chế `load_ply` ở §38. `shuffle=False` giữ nguyên thứ tự camera gốc (không quan trọng vì bước sau sắp lại thủ công).
4. **`background`** dựng từ `dataset.white_background`, giống mọi nơi khác trong repo.
5. **`cams = sorted(scene_obj.getTestCameras(), key=lambda c: c.image_name)`** — sắp xếp theo **tên ảnh gốc** (chuỗi), không theo thứ tự trong `cameras.json`, đảm bảo thứ tự file nộp bài **ổn định và tái lập được** giữa các lần chạy.
6. **Vòng lặp** `for index, cam in enumerate(..., start=1)`:
   - `rendered = torch.clamp(render_fastgs(...)["render"], 0.0, 1.0)` — **clamp về [0,1]** trước khi lưu, tránh giá trị âm/vượt 1 từ alpha-blend gây lỗi màu khi `torchvision.utils.save_image` tự nhân 255.
   - `name = _image_name(index, cfg)` = `f"{index:0{cfg.submission_digits}d}{cfg.submission_ext}"` → với mặc định `submission_digits=4`, `submission_ext=".png"`, ra `0001.png`, `0002.png`, ... (đệm **4 chữ số**, khác 5 chữ số của `render.py`).
   - `torchvision.utils.save_image(rendered, os.path.join(out_dir, name))` ghi thẳng vào `cfg.submission_scene_dir(scene)` = `submission/<scene>/`.
   - Ghi vào `files`: `dict(file=name, source=cam.image_name, width=cam.image_width, height=cam.image_height)`.
   - **Nếu `score=True`**: lấy `gt = torch.clamp(cam.original_image.to("cuda")[:3], 0.0, 1.0)` (chỉ khi ảnh gốc có sẵn — với dữ liệu eval-split kiểu 3DGS, test camera vẫn có `original_image` nạp từ đĩa), tính `psnr_fn`, `ssim_fn`, `lpips_fn(..., net_type=cfg.lpips_net_report)` (mặc định `"vgg"`) rồi append vào các list; `del gt` ngay sau đó để giải phóng VRAM.
   - `del rendered` cuối mỗi vòng — dọn VRAM liên tục vì vòng lặp có thể chạy hàng trăm camera độ phân giải gốc (nặng hơn nhiều so với train ở `resolution=2`).
7. Sau vòng lặp: dựng `info = dict(scene=..., images=len(files), out_dir=..., width=..., height=..., files=files)`; nếu có `psnrs` (tức `score=True` và có ground-truth), tính trung bình cộng ba metric rồi gọi `composite_score(psnr_val, ssim_val, lpips_val, cfg.psnr_max)` để ra `score_val, psnr_norm`, in log một dòng tổng kết; nếu không có ground-truth thì in dòng "(không có ground-truth để chấm)" và **không** có khoá `score` trong `info`.
8. **`_manifest.json`** được ghi vào chính `out_dir` (`submission/<scene>/_manifest.json`), chứa toàn bộ `info` (kể cả `files` — danh sách ánh xạ tên file nộp bài ↔ tên ảnh gốc ↔ kích thước ↔ metric tổng). File này **không** được `build_zip` đưa vào zip (§40 lọc theo đuôi `submission_ext`), nó chỉ để tiện tra cứu/so sánh cục bộ.
9. Cuối hàm: `del gaussians, scene_obj; gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()` — dọn RAM/VRAM trước khi trả `info` về cho `run_all`/`render_all` xử lý scene kế tiếp.

**Điểm phải nói rõ:** toàn bộ hàm này chạy **trong cùng tiến trình Python** đang giữ notebook/pipeline (import trực tiếp `scene.Scene`, `gaussian_renderer.render_fastgs`, ...) — **không** phải gọi `!python render.py` như một subprocess con. Hệ quả: (a) không có cách nào "giới hạn" bộ nhớ của bước render này tách biệt khỏi tiến trình chính — nếu nó rò VRAM/RAM, ảnh hưởng trực tiếp tới các scene train sau; (b) không tận dụng được việc một subprocess kết thúc sẽ tự động trả toàn bộ VRAM về hệ điều hành — dọn dẹp hoàn toàn phụ thuộc vào các lệnh `del`/`gc.collect()`/`torch.cuda.empty_cache()` thủ công ở cuối hàm; (c) đổi lại, tránh được chi phí khởi động lại CUDA context và nạp lại thư viện mỗi lần render một scene — nhanh hơn đáng kể khi `run_all` lặp qua nhiều scene liên tiếp trên Colab.

## 40. `build_zip` và `verify` — hợp đồng nộp bài

### `build_zip(cfg, scenes=None)` (`pipeline/submission.py:96-113`)

```python
scenes = scenes or sorted(d for d in os.listdir(cfg.submission_dir)
                          if os.path.isdir(os.path.join(cfg.submission_dir, d)))
if os.path.exists(cfg.submission_zip):
    os.remove(cfg.submission_zip)

with zipfile.ZipFile(cfg.submission_zip, "w", zipfile.ZIP_STORED) as zf:
    for scene in scenes:
        folder = cfg.submission_scene_dir(scene)
        for name in sorted(os.listdir(folder)):
            if not name.endswith(cfg.submission_ext):
                continue
            zf.write(os.path.join(folder, name), f"{scene}/{name}")
```

- Nếu không truyền `scenes`, tự liệt kê mọi thư mục con của `cfg.submission_dir` (sắp theo tên).
- Xoá zip cũ nếu đã tồn tại trước khi ghi mới (tránh cộng dồn file thừa từ lần chạy trước).
- Lọc `name.endswith(cfg.submission_ext)` — nghĩa là **`_manifest.json` bị loại khỏi zip**, chỉ ảnh `.png` (hoặc đuôi `cfg.submission_ext` khác nếu đổi cấu hình) được đóng gói.
- Đường dẫn trong zip là `f"{scene}/{name}"` — đúng cây thư mục:

```
submission.zip
├── <scene_1>/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── <scene_2>/
│   └── ...
└── ...
```

- Tên file tuân theo `_image_name(index, cfg) = f"{index:0{cfg.submission_digits}d}{cfg.submission_ext}"` — với mặc định (`submission_digits=4`, `submission_ext=".png"`) là `0001.png, 0002.png, ...`, index bắt đầu từ 1 và liên tục theo đúng thứ tự camera đã sort ở §39 bước 5.
- Dùng `zipfile.ZIP_STORED` (lưu trữ nguyên văn, **không** nén thêm) thay vì `ZIP_DEFLATED` — vì PNG đã tự nén bên trong định dạng của nó, nén zip thêm lần nữa gần như không giảm dung lượng mà lại tốn CPU/RAM đáng kể khi zip hàng nghìn ảnh trên Colab (bộ nhớ giới hạn).
- In ra `f"{cfg.submission_zip} | {len(scenes)} scene | {total} ảnh | {size_mb:.1f} MB"` để xác nhận nhanh trước khi tải về.

### `verify(cfg, expected=None)` (`pipeline/submission.py:116-149`)

Trả về `(frame, problems)` — `frame` là `pandas.DataFrame` cột `scene, images, width, height`; `problems` là list chuỗi cảnh báo (rỗng = hợp lệ). Các bước kiểm tra, đúng thứ tự trong code:

1. Mở zip, lọc file có đuôi `cfg.submission_ext`; với mỗi tên, tách `scene, _, file = name.partition("/")` — nếu `file` rỗng (ảnh nằm ở gốc zip, không trong thư mục con nào) thì báo lỗi `"ảnh nằm ngoài thư mục scene: {name}"`.
2. Với mỗi scene tìm thấy: sắp tên file, so với `wanted = [_image_name(i, cfg) for i in range(1, len(files)+1)]` — nếu khác thì báo `"{scene}: tên file không liên tục ({files[:3]} ...)"` (bắt cả trường hợp thiếu số, nhảy số, hoặc sai định dạng đệm số).
3. Nếu một scene có 0 ảnh (không xảy ra trong luồng thường vì scene rỗng sẽ không có key trong `by_scene`, nhưng code vẫn thủ điều kiện `if not files`) → báo `"{scene}: không có ảnh nào"`.
4. Mở ảnh đầu tiên của mỗi scene bằng `PIL.Image.open` để lấy `(width, height)`, đưa vào `rows` cho bảng hiển thị.
5. Nếu truyền `expected` (danh sách scene mong đợi — `run.finish` truyền đúng `scenes` đã train qua `run_all`): tính `missing = [s for s in expected if s not in by_scene]` và `extra = [s for s in by_scene if s not in expected]`, báo tương ứng `"thiếu scene: {missing}"` / `"thừa scene: {extra}"`.
6. In `"OK: submission hợp lệ"` nếu `problems` rỗng, ngược lại in `"CẢNH BÁO:\n- " + "\n- ".join(problems)"`.

**Những gì `verify()` KHÔNG thể kiểm tra** — cần nói thẳng với người dùng để tránh ảo tưởng "verify OK = chắc chắn được điểm":

- Nó không biết **số lượng test-pose thật** mà ban tổ chức sẽ chấm cho mỗi scene — chỉ so với `expected` do chính người dùng truyền vào (thường là danh sách scene, không phải số ảnh mỗi scene). Nếu bộ test cục bộ (`Scene.getTestCameras()`, quyết định bởi `llffhold` lúc train) có số lượng camera khác với test set thật của ban tổ chức, `verify()` sẽ báo "OK" một cách sai lệch.
- Nó không có quyền truy cập **độ phân giải ảnh chuẩn** của ban tổ chức để so khớp — chỉ đọc kích thước ảnh đầu tiên trong zip để hiển thị, không đối chiếu với bất kỳ giá trị tham chiếu nào bên ngoài.
- Nó **không kiểm tra nội dung hình ảnh** — không đánh giá "hình học đúng, vật thể đúng vị trí, ảnh thực tế và nhất quán" như luật cuộc thi yêu cầu. Việc đó chỉ có thể kiểm bằng mắt qua `pipeline.report.show_samples` (so ảnh render cạnh ảnh ground-truth), không có bước tự động nào trong `verify()`.
- Tóm lại, `verify()` chỉ đảm bảo **tính nhất quán nội bộ** của file zip (tên liên tục, đúng cấu trúc thư mục, không thiếu/thừa scene so với danh sách bạn tự cung cấp) — không phải một bộ chấm điểm giả lập.

---

# PHẦN VII — CHẤM ĐIỂM VÀ PHỤ LỤC

## 41. `metrics.py` — đường CLI

`metrics.py` là script chấm điểm độc lập, chạy sau khi `render.py` đã xuất ảnh vào `test/ours_<n>/renders` và `test/ours_<n>/gt` (hoặc `train/...`). Gọi bằng:

```
python metrics.py -m <model_path_1> [<model_path_2> ...]
```

**`readImages(renders_dir, gt_dir)`** (`metrics.py:24-34`) ghép cặp ảnh render với ảnh ground-truth **theo tên file**: nó liệt kê `os.listdir(renders_dir)` rồi với mỗi `fname` mở luôn `gt_dir / fname` — không có bước đối chiếu danh sách, không sort tường minh trước khi ghép theo index (nhưng vì cùng `fname` nên phép mở file là an toàn miễn là `renders_dir` và `gt_dir` chứa đúng các file trùng tên). Mỗi ảnh được `tf.to_tensor(...).unsqueeze(0)[:, :3, :, :].cuda()` — ép về đúng 3 kênh (bỏ alpha nếu có) và đẩy lên GPU. Hàm trả về ba list son song: `renders`, `gts`, `image_names`.

**Vòng lặp chính** nằm trong `evaluate(model_paths)` (`metrics.py:36-93`):
- Với mỗi `scene_dir` trong `model_paths`, script tìm `test_dir = Path(scene_dir) / "test"` rồi lặp qua từng `method` (tức từng thư mục con kiểu `ours_<n>` do `render.py` tạo ra).
- Với mỗi `method`, gọi `readImages` rồi với từng ảnh tính `ssim(renders[idx], gts[idx])` (từ `utils/loss_utils.py`), `psnr(renders[idx], gts[idx])` (từ `utils/image_utils.py`), và `lpips(renders[idx], gts[idx], net_type='vgg')` — **CLI luôn dùng mạng VGG cho LPIPS**, không có tuỳ chọn đổi sang alex.
- **Vào `results.json`** (ghi ở `scene_dir + "/results.json"`, `metrics.py:88-89`): chỉ ba số trung bình toàn method — `{"SSIM": mean, "PSNR": mean, "LPIPS": mean}` — dạng `full_dict[scene_dir][method]`.
- **Vào `per_view.json`** (`scene_dir + "/per_view.json"`, `metrics.py:90-91`): điểm SSIM/PSNR/LPIPS **của từng ảnh riêng lẻ**, dạng `{"SSIM": {tên_file: giá_trị, ...}, "PSNR": {...}, "LPIPS": {...}}` — dùng để soi ảnh nào tệ nhất.
- Các biến `full_dict_polytopeonly` / `per_view_dict_polytopeonly` được khởi tạo (`metrics.py:40-41,49-50,59-60`) nhưng **không bao giờ được điền dữ liệu hay ghi ra file** — đây là phần thừa kế từ mã gốc 3DGS, không có tác dụng trong bản này; không cần bận tâm.

**Cái bẫy `except` trần** (`metrics.py:92-93`):
```python
except:
    print("Unable to compute metrics for model", scene_dir)
```
Toàn bộ thân vòng lặp cho một `scene_dir` — kể cả việc mở thư mục, đọc ảnh, tính SSIM/PSNR/LPIPS, và ghi JSON — nằm trong khối `try` bắt đầu ở `metrics.py:45`. Bất kỳ lỗi nào (thiếu thư mục `test/`, `renders`/`gt` không khớp tên file, ảnh hỏng, hết VRAM khi chạy LPIPS, lỗi ghi đĩa...) đều bị nuốt gọn: script chỉ in một dòng "Unable to compute metrics for model ..." rồi **chuyển sang scene tiếp theo mà không để lại bất kỳ dấu vết nào khác** — không traceback, không exit code khác 0, và quan trọng nhất là **không có `results.json`/`per_view.json` nào được ghi ra cho scene đó**. Nếu chạy hàng loạt nhiều scene, một scene lỗi sẽ âm thầm biến mất khỏi báo cáo cuối cùng thay vì làm dừng cả script — đây là điểm cần kiểm tra thủ công (xem §47) chứ không thể tin rằng "chạy xong không báo lỗi" nghĩa là "mọi scene đều có điểm".

## 42. `lpipsPyTorch` — LPIPS chạy thế nào

Hàm công khai `lpips(x, y, net_type='alex', version='0.1')` (`lpipsPyTorch/__init__.py:6-21`) chỉ đơn giản khởi tạo `LPIPS(net_type, version).to(x.device)` rồi gọi nó như một criterion — không cache theo lần gọi, nghĩa là **mỗi lệnh gọi `lpips(...)` xây dựng lại toàn bộ mạng backbone** (kể cả nạp state dict). Đây là lý do các đường chấm điểm tần suất cao (đánh giá trong lúc train) nên dùng ít view/khoảng cách xa nhau hơn là gọi liên tục.

`LPIPS.forward(x, y)` (`lpipsPyTorch/modules/lpips.py:29-34`): trích đặc trưng `feat_x = self.net(x)`, `feat_y = self.net(y)` (list theo từng layer mục tiêu), tính `diff = (feat_x - feat_y)**2` theo từng layer, đưa qua các lớp tuyến tính 1×1 `self.lin` (trọng số học sẵn, không có bias — `lpipsPyTorch/modules/networks.py:23-33`), lấy trung bình không gian `.mean((2,3), True)`, rồi **cộng dồn qua tất cả các layer** (`torch.sum(torch.cat(res, 0), 0, True)`) để ra một số LPIPS duy nhất cho mỗi ảnh (giá trị càng nhỏ càng giống nhau).

**`alex` vs `vgg`** (`lpipsPyTorch/modules/networks.py:12-20,66-96`):
| net_type | Backbone | Layer trích đặc trưng | Số kênh mỗi layer |
|---|---|---|---|
| `alex` | `torchvision.models.alexnet(pretrained=True).features` | `[2, 5, 8, 10, 12]` | `[64, 192, 384, 256, 256]` |
| `squeeze` | `squeezenet1_1(pretrained=True).features` | `[2, 5, 8, 10, 11, 12, 13]` | `[64,128,256,384,384,512,512]` |
| `vgg` | `vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features` | `[4, 9, 16, 23, 30]` | `[64,128,256,512,512]` |

`vgg` đi qua backbone sâu và nặng hơn `alex` (VGG16 có nhiều tham số và tốn FLOPs hơn AlexNet đáng kể), nên **chậm hơn rõ rệt cho mỗi lần gọi** — đúng như tên biến `lpips_net_report` (chỉ dùng khi chấm báo cáo cuối, số lần gọi ít) so với `lpips_net_live` (dùng liên tục trong lúc train, cần nhanh).

**Chuẩn hoá đầu vào**: `BaseNet.z_score(x)` (`lpipsPyTorch/modules/lpips.py` gọi gián tiếp qua `networks.py:50-51`) trừ `mean = [-.030, -.088, -.188]` và chia `std = [.458, .448, .450]` — đây là hằng số chuẩn hoá riêng của LPIPS (khác hẳn ImageNet mean/std thông thường), áp dụng **trước khi** đưa ảnh vào backbone. Vì vậy `x, y` truyền vào hàm `lpips(...)` phải là ảnh RGB đã ở thang **[0, 1]** (giống định dạng `torchvision.transforms.functional.to_tensor` hoặc `torch.clamp(render, 0, 1)`) — module tự lo phần chuẩn hoá tiếp theo, người gọi không cần tự chuẩn hoá theo ImageNet.

**Trọng số học sẵn tải từ đâu**: `get_state_dict(net_type, version)` (`lpipsPyTorch/modules/utils.py:11-30`) tải file `.pth` từ:
```
https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/<net_type>.pth
```
qua `torch.hub.load_state_dict_from_url(..., progress=True)`, sau đó đổi tên khoá (`lin` → bỏ, `model.` → bỏ) để khớp với `LinLayers`. Đây là các file rất nhỏ (chỉ chứa trọng số của các lớp tuyến tính 1×1, không phải trọng số backbone) — cỡ vài chục KB, tải gần như tức thời nếu có mạng; backbone (`alexnet`/`vgg16`/`squeezenet1_1`) thì tải trọng số ImageNet tiêu chuẩn của `torchvision` (mục "tải trọng số lần đầu" — vgg16 khoảng 500+ MB, alexnet nhẹ hơn nhiều, ~230 MB) qua cơ chế cache mặc định của `torchvision.models` (thư mục `~/.cache/torch/hub/checkpoints`). Nếu máy chạy offline hoàn toàn, cả hai lần tải này đều thất bại (xem §47).

**Nơi mỗi mạng được dùng**:
| Vị trí | net_type | Biến cấu hình |
|---|---|---|
| `metrics.py:74` (CLI) | `'vgg'` | cố định trong code, không cấu hình được |
| `pipeline/score.py::evaluate_cameras` (chấm nhanh trong lúc train) | tham số `lpips_net`, gọi từ `trainer.py:158` với `cfg.lpips_net_live` | `Config.lpips_net_live = "alex"` (`pipeline/config.py:39`) |
| `pipeline/submission.py::render_scene` (chấm báo cáo/nộp bài) | `cfg.lpips_net_report` (`pipeline/submission.py:62`) | `Config.lpips_net_report = "vgg"` (`pipeline/config.py:40`) |

## 43. PSNR và SSIM

**`utils/image_utils.py::psnr(img1, img2)`** (dòng 17-19):
```python
mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
return 20 * torch.log10(1.0 / torch.sqrt(mse))
```
Đây là công thức PSNR chuẩn cho ảnh đã chuẩn hoá về `[0, 1]` (`MAX_I = 1`, nên `20*log10(1) - 10*log10(MSE) = -10*log10(MSE) = 20*log10(1/sqrt(MSE))`). `.view(img1.shape[0], -1).mean(1, keepdim=True)` gộp phẳng mọi kênh màu và pixel rồi lấy trung bình theo batch — hàm trả về **một tensor có shape `[batch, 1]`** (không phải một số vô hướng); các nơi gọi (`metrics.py:73`, `pipeline/score.py:40`, `pipeline/submission.py`) đều tự `.mean()`/`.item()` thêm để lấy số.

**`utils/loss_utils.py::ssim(img1, img2, window_size=11, size_average=True)`** (dòng 36-44, `_ssim` dòng 46-66):
- Xây cửa sổ Gauss 2D kích thước **11×11** (`window_size=11`), 1 chiều Gauss có `sigma=1.5` (`gaussian(window_size, 1.5)`, dòng 26-28) rồi nhân ngoài (`_1D_window.mm(_1D_window.t())`) để ra 2D, sau đó `expand` ra theo số kênh (`create_window`, dòng 30-34).
- `_ssim` tính SSIM bằng convolution theo từng kênh riêng (`F.conv2d(..., groups=channel)`) — tức **xử lý mỗi kênh màu độc lập** rồi mới gộp, không chuyển sang grayscale trước.
- Hằng số ổn định `C1 = 0.01**2`, `C2 = 0.03**2` (dòng 17-18 và lặp lại dòng 58-59) — giá trị chuẩn theo bài báo SSIM gốc (giả định ảnh trong `[0,1]`, dynamic range = 1).
- `size_average=True` (mặc định) trả về **một số vô hướng** = trung bình toàn bộ bản đồ SSIM trên mọi pixel, mọi kênh, mọi ảnh trong batch (`ssim_map.mean()`); nếu `False` thì trả theo từng ảnh trong batch (`ssim_map.mean(1).mean(1).mean(1)`).

**`fused_ssim`** (thư viện ngoài, `from fused_ssim import fused_ssim as fast_ssim`, dùng ở `train.py:18,102,226`, `pipeline/trainer.py:62`, `pipeline/score.py:27,41`): cùng là chỉ số SSIM (cùng công thức toán, cửa sổ Gauss, hằng số C1/C2 tương đương) nhưng **là một kernel CUDA hợp nhất (fused)** viết riêng để chạy nhanh hơn implementation thuần PyTorch của `utils/loss_utils.py::ssim` — tránh việc gọi `F.conv2d` nhiều lần rời rạc. Về mặt giá trị hai hàm cho kết quả gần như tương đương (đều là SSIM cửa sổ Gauss 11×11), nhưng khác nhau về **hiệu năng**, không phải về ý nghĩa chỉ số.

**Ai dùng cái nào**:
| Nơi | Hàm SSIM | Lý do |
|---|---|---|
| Vòng lặp loss huấn luyện (`train.py:102`), test report giữa chừng (`train.py:226`), `pipeline/trainer.py`, `pipeline/score.py::evaluate_cameras` | `fused_ssim` | Gọi hàng nghìn lần mỗi phiên train → cần nhanh |
| `metrics.py::evaluate` (CLI chấm điểm cuối) | `utils/loss_utils.py::ssim` | Chạy một lần, ít ảnh, ưu tiên đơn giản/không phụ thuộc thêm submodule |

## 44. `composite_score` và bảng leaderboard

`pipeline/score.py::composite_score(psnr_val, ssim_val, lpips_val, psnr_max=30.0)` (dòng 12-18):
```python
psnr_norm = clamp(psnr_val / psnr_max, 0.0, 1.0)
score = 0.4 * (1 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm
```
Ba trọng số cố định ở đầu file (`pipeline/score.py:9`): `W_LPIPS = 0.4`, `W_SSIM = 0.3`, `W_PSNR = 0.3`.

**Ý nghĩa của `torch.clamp(psnr_val / psnr_max, 0.0, 1.0)`**: PSNR (đơn vị dB, không có trần tự nhiên) được chia cho `psnr_max` rồi chặn trên tại 1.0. Với `psnr_max = 30.0` mặc định (`Config.psnr_max`, `pipeline/config.py:38`, và tham số mặc định của chính `composite_score`): **một khi PSNR của scene vượt qua 30 dB, phần đóng góp PSNR vào score bị đóng băng ở mức tối đa (0.3 điểm) — cải thiện PSNR thêm nữa (31, 35, 40 dB...) không mang lại thêm điểm nào.** Đây là điểm khuyến khích rõ ràng: tối ưu hoá sau ngưỡng 30 dB nên dồn sang giảm LPIPS/tăng SSIM thay vì cố nặn thêm PSNR.

**Độ nhạy (đạo hàm) của score theo từng metric** (trong miền `psnr_val < psnr_max`, không tính bão hoà clamp):
- `∂score/∂lpips_val = -0.4` — mỗi 0.1 giảm LPIPS cho +0.04 điểm.
- `∂score/∂ssim_val = +0.3` — mỗi 0.1 tăng SSIM cho +0.03 điểm.
- `∂score/∂psnr_val = 0.3 / psnr_max = 0.01` (với `psnr_max=30`) khi `psnr_val < psnr_max`, và **bằng 0** khi `psnr_val ≥ psnr_max` — mỗi 1 dB PSNR tăng thêm cho +0.01 điểm, nhưng chỉ tính đến ngưỡng.
Vì hệ số của LPIPS (0.4) lớn nhất và đạo hàm tuyệt đối (0.4) cũng lớn nhất trong ba số, **LPIPS là đòn bẩy mạnh nhất lên score trên một đơn vị thay đổi thực tế** của chỉ số đó (dù thang giá trị của mỗi metric khác nhau nên so sánh "1 đơn vị" không hoàn toàn công bằng — LPIPS và SSIM đều nằm trong `[0,1]` nên so sánh trực tiếp được, PSNR thì thang dB khác hẳn).

**Bảng minh hoạ (ví dụ, không phải số đo thật)**:
| PSNR (dB) | SSIM | LPIPS | psnr_norm (psnr_max=30) | Score |
|---|---|---|---|---|
| 25 | 0.80 | 0.20 | 0.833 | 0.4×0.80 + 0.3×0.80 + 0.3×0.833 = 0.320+0.240+0.250 = **0.810** |
| 30 | 0.90 | 0.10 | 1.000 | 0.4×0.90 + 0.3×0.90 + 0.3×1.00 = 0.360+0.270+0.300 = **0.930** |
| 35 | 0.90 | 0.10 | 1.000 (bị chặn) | giống hệt hàng trên = **0.930** (thêm 5 dB PSNR không đổi gì) |
| 20 | 0.70 | 0.30 | 0.667 | 0.4×0.70 + 0.3×0.70 + 0.3×0.667 = 0.280+0.210+0.200 = **0.690** |

**`pipeline/report.py::leaderboard(results, submissions=None, save_to=None)`** (dòng 21-49): dựng một `DataFrame`, mỗi hàng là một scene (`row["live_score"]`, `row["live_psnr"]`,... lấy từ kết quả đánh giá trong lúc train — tiền tố `live_`; nếu có `submissions` — kết quả từ `pipeline/submission.py::render_scene` — thì join thêm các cột không tiền tố `score, psnr, psnr_norm, ssim, lpips`, tức **điểm ở chất lượng nộp bài đầy đủ**). Cuối cùng:
```python
numeric = frame.select_dtypes("number")
frame.loc["MEAN"] = numeric.mean().reindex(frame.columns)
```
Hàng `MEAN` là **trung bình cộng theo từng cột số của mọi scene** (không trọng số theo số ảnh/số Gaussian) — và dòng `frame.loc['MEAN', 'score']` chính là "điểm leaderboard (trung bình các scene)" được in ra (`pipeline/report.py:48`). Vì vậy leaderboard cuối cùng = trung bình `composite_score` của từng scene, đúng như mô tả kiến trúc: `Score` từng scene tính theo công thức ở trên, leaderboard = mean qua scene.

## 45. Bảng hằng số toàn hệ thống

| Hằng số | Giá trị | Vị trí | Vai trò |
|---|---|---|---|
| SH degree tối đa | 3 | `arguments/__init__.py:49` (`ModelParams.sh_degree`) | Bậc cầu điều hoà tối đa cho màu phụ thuộc góc nhìn |
| Nhịp tăng SH degree | mỗi 1000 iteration | `train.py:80-82` (`if iteration % 1000 == 0: gaussians.oneupSHdegree()`) | Tăng dần độ phức tạp màu sắc, tránh học SH bậc cao quá sớm |
| `min_opacity` (giai đoạn densify) | 0.005 | `train.py:140` (gọi `densify_and_prune_fastgs`) | Ngưỡng opacity để prune trong pha densify |
| `min_opacity` (giai đoạn cuối) | 0.1 | `train.py:158` (gọi `final_prune_fastgs`) | Ngưỡng opacity cao hơn khi prune lần cuối, dọn Gaussian yếu |
| Ngưỡng kích thước màn hình (`max_screen_size`) | 20 | `train.py:133` (`size_threshold = 20 if iteration > opt.opacity_reset_interval else None`) | Pixel — Gaussian chiếm view lớn hơn ngưỡng này (sau lần reset opacity đầu) bị coi là "quá to", có thể bị prune |
| `percent_dense` | 0.001 | `arguments/__init__.py:83` (`OptimizationParams.percent_dense`) | Đặt vào `self.percent_dense` của `GaussianModel` (`scene/gaussian_model.py:193`); tham chiếu tỉ lệ với `extent` cảnh để phân loại clone/split ở 3DGS gốc |
| `dense` (fastgs-lite, đóng vai trò percent_dense) | 0.001 | `arguments/__init__.py:100` (`OptimizationParams.dense`) | Dùng trực tiếp trong `densify_and_prune_fastgs`: `clone_qualifiers = scaling.max <= args.dense*extent`, `split_qualifiers = scaling.max > args.dense*extent` (`scene/gaussian_model.py:486-487`) |
| `opacity_reset_interval` | 3000 | `arguments/__init__.py:86` | Chu kỳ (iteration) reset opacity về thấp; cũng là mốc bật ngưỡng screen-size 20 |
| `densify_from_iter` | 500 | `arguments/__init__.py:87` | Iteration bắt đầu tính densify |
| `densify_until_iter` | 15000 | `arguments/__init__.py:88` | Iteration dừng densify/prune theo gradient |
| `densification_interval` | 100 (mặc định gốc) — nhưng pipeline Colab override thành **500** | `arguments/__init__.py:85`; override tại `pipeline/config.py:43` (`train_extra_args: ["--densification_interval", "500", ...]`) | Chu kỳ (số iteration) giữa hai lần chạy densify_and_prune |
| `position_lr_max_steps` | 30000 | `arguments/__init__.py:76` | Số bước để lịch suy giảm learning-rate vị trí (`position_lr_init` → `position_lr_final`) hoàn tất — `pipeline/trainer.py::build_args` ghi đè bằng đúng số vòng train; đường CLI giữ nguyên 30000. |
| `lambda_dssim` | 0.2 (mặc định gốc) — pipeline Colab override thành **0.25** | `arguments/__init__.py:82`; override `pipeline/config.py:44` | Trọng số D-SSIM trong loss: `loss = (1-λ)*L1 + λ*(1-SSIM)` (`train.py:103`) |
| `mult` | 0.5 | `arguments/__init__.py:101` (`OptimizationParams.mult`); cũng là `Config.mult` (`pipeline/config.py:37`) | Hệ số nhân "compact box" kiểm soát số tile mỗi splat chiếm (đặc thù fastgs-lite) |
| `loss_thresh` | 0.1 (mặc định gốc) — pipeline Colab override thành **0.07** | `arguments/__init__.py:95`; override `pipeline/config.py:46` | Ngưỡng loss dùng trong tính điểm multi-view của fastgs-lite (`utils/fast_utils.py`) |
| `grad_abs_thresh` | 0.0012 | `arguments/__init__.py:93` (mặc định gốc trùng với override `pipeline/config.py:46`) | Ngưỡng gradient tuyệt đối để đánh dấu ứng viên "split" (`grad_qualifiers_abs`, `scene/gaussian_model.py:484`) |
| `grad_thresh` | 0.0002 | `arguments/__init__.py:96` | Ngưỡng gradient (norm) để đánh dấu ứng viên "clone" (`grad_qualifiers`, `scene/gaussian_model.py:483`) |
| `lowfeature_lr` | 0.0025 | `arguments/__init__.py:98` | Learning rate cho nhóm feature "thấp" (đặc thù fastgs-lite, tách khỏi `feature_lr` gốc) |
| `highfeature_lr` | 0.005 (mặc định gốc) — pipeline Colab override thành **0.02** | `arguments/__init__.py:91`; override `pipeline/config.py:45` | Learning rate cho nhóm feature "cao" |
| Ngưỡng importance-score trong densify mask | `importance_score > 5` | `scene/gaussian_model.py:494` (`metric_mask = importance_score > 5`) | Gaussian phải được tối thiểu ~6 lượt "phiếu" đa góc nhìn mới được coi là ứng viên densify hợp lệ |
| Ngưỡng pruning-score cuối | `pruning_score > 0.9` | `scene/gaussian_model.py:538` (`final_prune_fastgs`) | Gaussian có điểm nhất quán đa góc nhìn chuẩn hoá > 0.9 (tức rất kém — xem §48) bị prune ở bước dọn cuối |
| Kích thước tile rasterizer | 16 × 16 pixel | `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/config.h:16-17` (`BLOCK_X 16`, `BLOCK_Y 16`) | Kích thước ô lưới dùng để phân vùng màn hình khi rasterize |
| `psnr_max` | 30.0 | `pipeline/config.py:38` (`Config.psnr_max`); mặc định trùng trong `pipeline/score.py:12,23` | Ngưỡng chuẩn hoá PSNR trong `composite_score` (xem §44) |
| `llffhold` | 8 | `ModelParams.llffhold` (`arguments/__init__.py`, cờ `--llffhold`), mặc định hàm `readColmapSceneInfo` (`scene/dataset_readers.py:132`) và `Config.llffhold` (`pipeline/config.py`) — cả ba cùng giá trị 8; `Scene` truyền cờ này xuống reader. | Cứ 8 ảnh COLMAP thì 1 ảnh (`idx % llffhold == 0`) làm test/hold-out, còn lại làm train |
| `ram_soft_limit_gb` | 10.5 | `pipeline/config.py:41` | Ngưỡng RAM mềm nội bộ pipeline Colab, không phải hằng số của lõi 3DGS/fastgs-lite |

Lưu ý quan trọng: nhiều tham số của `OptimizationParams` (`arguments/__init__.py`) có **giá trị mặc định gốc** nhưng bị **override bởi `Config.train_extra_args`** (`pipeline/config.py:42-48`) khi chạy qua pipeline Colab — cụ thể `densification_interval` (100→500), `lambda_dssim` (0.2→0.25), `highfeature_lr` (0.005→0.02), `loss_thresh` (0.1→0.07), `grad_abs_thresh` (0.0012→0.0012, không đổi). Khi đọc log/`cfg_args` của một lần chạy cụ thể, giá trị hiệu lực là giá trị sau override, không phải giá trị mặc định trong `arguments/__init__.py`.

## 46. Bảng tra nhanh bước ↔ file ↔ hàm

| Bước | Đường CLI (`train.py`/`render.py`/`metrics.py`) | Đường `pipeline/` |
|---|---|---|
| Cài đặt/thiết lập tham số | `arguments/__init__.py` (`ModelParams`, `OptimizationParams`, `PipelineParams`) | `pipeline/trainer.py::build_args` dựng `Namespace` rồi gọi thẳng các lớp trên |
| Chuẩn bị dữ liệu / đọc scene COLMAP | `scene/__init__.py::Scene`, `scene/dataset_readers.py::readColmapSceneInfo` | `pipeline/data.py` (tìm scene, tải/giải nén dataset, tính `n_test` theo `llffhold`) |
| Vòng lặp huấn luyện | `train.py` (hàm `training`, vòng `for iteration in range(...)`) | `pipeline/trainer.py::train_scene` (gọi `render_fastgs`, `compute_gaussian_score_fastgs`, `densify_and_prune_fastgs`) |
| Đánh giá nhanh trong lúc train | `train.py::training_report` (bị comment ở dòng gọi, xem PHẦN VI) | `pipeline/score.py::evaluate_cameras`, gọi định kỳ từ `pipeline/trainer.py:157-158` theo `cfg.score_every` |
| Lưu checkpoint `.ply` | `scene/__init__.py::Scene.save`, gọi tại `train.py` khi `iteration in saving_iterations` | `pipeline/trainer.py::_save_checkpoint`, điều khiển bởi `cfg.save_every` / `cfg.keep_last_checkpoint` |
| Render ảnh test/train | `render.py` (`render_sets`, `render_set`) | `pipeline/submission.py::render_scene` |
| Chấm điểm CLI (SSIM/PSNR/LPIPS-vgg) | `metrics.py::evaluate` → `results.json` + `per_view.json` | `pipeline/submission.py::render_scene` (score inline) + `pipeline/score.py::composite_score` |
| Tổng hợp báo cáo/so sánh scene | không có tương đương CLI trực tiếp (`full_eval.py` gộp nhiều scene nhưng không tính leaderboard) | `pipeline/report.py::leaderboard`, `plot_training`, `plot_leaderboard`, `show_samples` |
| Đóng gói nộp bài (submission.zip) | không có tương đương CLI | `pipeline/submission.py::render_scene` (ghi ảnh) rồi lệnh zip trong `pipeline/run.py`; kiểm tra bằng `pipeline/submission.py::verify` |
| Tải kết quả về máy / lưu Drive | không có tương đương CLI | `pipeline/deliver.py::pack_models`, `copy_to_drive`, tải file `.zip` |

## 47. Chẩn đoán sự cố

| Triệu chứng | Nguyên nhân (theo code) | Cách xử lý |
|---|---|---|
| CUDA out of memory trên T4 | VRAM 16 GB của T4 không đủ khi số Gaussian tăng nhanh trong pha densify, hoặc `mult`/độ phân giải quá cao; `compute_gaussian_score_fastgs` giữ nhiều tensor tạm khi tính điểm đa góc nhìn | Giảm `resolution` (tăng `-r`), giảm `iterations`/tần suất densify, hoặc gọi `torch.cuda.empty_cache()` (đã có sau `densify_and_prune_fastgs`, `scene/gaussian_model.py:531`); giảm `eval_views`/`max_views` khi chấm điểm live |
| RAM (CPU) OOM khi nén `submission.zip` | `pipeline/submission.py`/`pipeline/deliver.py` giữ nhiều ảnh/đường dẫn trong bộ nhớ khi zip nhiều scene liên tiếp trên Colab (RAM hệ thống giới hạn, xem `Config.ram_soft_limit_gb = 10.5`) | Zip theo từng scene rồi giải phóng, theo dõi `pipeline/env.py::mem`/`show_mem`; hạ `ram_soft_limit_gb` để cảnh báo sớm hơn |
| `Could not recognize scene type!` | `scene/__init__.py:49` — `Scene.__init__` không tìm thấy `sparse/` (COLMAP) hoặc `transforms_train.json` (Blender) trong `source_path` | Kiểm tra đường dẫn `-s`/`data_root`, đảm bảo cấu trúc thư mục scene đúng chuẩn COLMAP (`sparse/0/...`) hoặc Blender |
| Zero test camera / không chấm được | Chạy thiếu cờ `--eval`; khi đó `readColmapSceneInfo` không tách hold-out theo `llffhold` mà đưa hết ảnh vào train (`scene/dataset_readers.py:149-150` chỉ tách khi `eval=True`) | Luôn truyền `--eval` (đã có sẵn trong `pipeline/trainer.py::build_args`, `argv` chứa `"--eval"`); nếu chạy `train.py` tay thì phải tự thêm |
| Ảnh submission sai kích thước | `Config.submission_resolution` khác 1 khiến `render_scene` render ở độ phân giải bị scale thay vì đúng kích thước ảnh gốc (`pipeline/config.py:55` ghi rõ "1 = render đúng kích thước ảnh gốc") | Đặt `submission_resolution = 1` trước khi render nộp bài; chạy `pipeline/submission.py::verify` để soi kích thước ảnh thực tế trong zip |
| Tải trọng số LPIPS thất bại khi offline | `get_state_dict` (`lpipsPyTorch/modules/utils.py:17-20`) và backbone `torchvision.models.*(True/weights=...)` đều gọi `torch.hub`/tải từ Internet; không có cơ chế fallback cục bộ | Tải trước khi mất mạng (chạy một lần lúc còn mạng để cache vào `~/.cache/torch/hub`), hoặc chuẩn bị sẵn cache trên máy Colab |
| `verify()` báo tên file không liên tục | `pipeline/submission.py:139-141` — so khớp danh sách file thực tế với dãy `0001.png, 0002.png, ...` sinh từ `_image_name`; lệch nghĩa là thiếu ảnh, đánh số nhảy cóc, hoặc ảnh nằm sai thư mục scene | Render lại toàn bộ scene bằng `render_scene` (đảm bảo không bị ngắt giữa chừng), kiểm tra `by_scene` trong log lỗi |
| Phiên Colab bị ngắt giữa lúc train một scene | Không có auto-resume tích hợp; mất tiến trình iteration hiện tại nếu chưa tới mốc lưu | `save_every` (`Config.save_every`, mặc định 2000, `pipeline/config.py:34`) quyết định tần suất lưu `.ply` định kỳ — giảm giá trị này để hạn chế mất việc khi phiên chết giữa chừng; `keep_last_checkpoint=True` chỉ giữ checkpoint mới nhất nên không "quay lại" iteration cũ hơn được |
| Thiếu điểm số cho một scene mà không có lỗi rõ ràng | Bẫy `except:` trần trong `metrics.py:92-93` nuốt mọi lỗi khi chấm CLI, chỉ in một dòng cảnh báo và bỏ qua scene, không ghi `results.json`/`per_view.json` (xem §41) | Không tin tưởng "chạy xong không crash" là đủ; luôn kiểm tra sự tồn tại của `results.json` cho từng scene sau khi chạy `metrics.py`, hoặc tạm sửa thành `except Exception as e: print(e)` khi debug |

## 48. Thuật ngữ

| Thuật ngữ | Giải thích |
|---|---|
| Gaussian (3D Gaussian) | Đơn vị biểu diễn cảnh: một phân bố Gauss 3D có vị trí, hiệp phương sai (scale + rotation), opacity và màu (SH), được rasterize thành "vết" (splat) trên ảnh |
| SH (Spherical Harmonics — cầu điều hoà) | Khai triển hàm màu phụ thuộc góc nhìn; bậc (degree) càng cao càng biểu diễn được hiệu ứng phản chiếu/góc nhìn phức tạp, đổi lại tốn bộ nhớ hơn |
| Opacity | Độ mờ/đục của một Gaussian, dùng sigmoid nghịch đảo (`inverse_sigmoid`) để tham số hoá; Gaussian có opacity quá thấp bị coi là "vô hình" và bị prune |
| Densify (làm dày) | Quá trình thêm Gaussian mới ở vùng thiếu chi tiết, gồm hai thao tác: clone và split |
| Clone | Nhân đôi một Gaussian nhỏ (đang thiếu chi tiết nhưng đã đủ nhỏ) thành hai bản giữ nguyên scale, dịch nhẹ vị trí |
| Split | Tách một Gaussian lớn thành N Gaussian con nhỏ hơn (scale chia nhỏ theo hệ số `0.8*N`), dùng khi Gaussian đã đủ lớn nhưng vẫn thiếu chi tiết |
| Prune (tỉa) | Xoá Gaussian không cần thiết: opacity quá thấp, kích thước màn hình/không gian quá lớn, hoặc bị đánh dấu bởi pruning-score cao |
| Tile | Ô lưới 16×16 pixel (`BLOCK_X`, `BLOCK_Y`) mà rasterizer CUDA dùng để phân vùng và song song hoá việc vẽ splat lên ảnh |
| Splat | Kết quả chiếu 2D của một Gaussian 3D lên mặt phẳng ảnh khi rasterize |
| Importance score | Số nguyên đếm số lượt (qua nhiều camera lấy mẫu) một Gaussian được "bỏ phiếu" là quan trọng cho chất lượng render đa góc nhìn; dùng để lọc ứng viên densify (`metric_mask = importance_score > 5`) |
| Pruning score | Điểm chuẩn hoá 0..1 đo mức độ *kém* nhất quán đa góc nhìn của một Gaussian (giá trị càng cao càng nên loại bỏ); dùng trong cả pha densify (lấy mẫu theo trọng số `1/(1e-6+1-pruning_score)`) và pha cuối (`pruning_score > 0.9` → prune) |
| Hold-out | Tập ảnh/camera bị giữ lại không dùng để train, dùng làm test để đo khả năng tổng quát hoá — chọn bằng `idx % llffhold == 0` |
| Novel view synthesis | Tổng hợp ảnh từ góc nhìn (camera pose) mới không có trong tập ảnh gốc, dựa trên mô hình 3D đã học |
| LPIPS | "Learned Perceptual Image Patch Similarity" — chỉ số khác biệt cảm nhận, dùng đặc trưng mạng CNN đã học (AlexNet/VGG16/SqueezeNet) thay vì so khớp pixel; càng thấp càng giống |
| SSIM | "Structural Similarity Index" — đo độ tương đồng cấu trúc (độ sáng, tương phản, cấu trúc cục bộ) qua cửa sổ trượt Gauss; càng cao (gần 1) càng giống |
| PSNR | "Peak Signal-to-Noise Ratio" — đo sai khác pixel-wise qua MSE, đơn vị dB; càng cao càng giống |
| Compact box | Vùng bao (bounding) được co gọn quanh mỗi splat để giới hạn số tile nó chạm tới khi rasterize, điều khiển bởi hệ số `mult` |
| COLMAP sparse | Kết quả tái tạo camera pose + point cloud thưa từ COLMAP structure-from-motion, đọc bởi `scene/dataset_readers.py::readColmapSceneInfo`, nằm trong thư mục `sparse/0/` của scene |
| Extent | Kích thước (bán kính) không gian của cảnh, tính từ vị trí các camera (`scene.cameras_extent`); dùng làm đơn vị co giãn cho các ngưỡng clone/split và ngưỡng kích thước loại bỏ Gaussian quá to (`0.1 * extent`) |
