Dưới đây là các đoạn code được tách riêng theo từng giai đoạn của pipeline huấn luyện (dựa trên FastGS - biến thể tăng tốc của 3D Gaussian Splatting).

## 1. Import & khởi tạo phụ thuộc

```python
import torch
import numpy as np
import os, random, time
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
```
Nạp các module: renderer (`render_fastgs`), loss (L1, SSIM, LPIPS), scene/model của Gaussian Splatting, và các hàm phụ trợ đặc thù của FastGS (`compute_gaussian_score_fastgs`, `sampling_cameras`).

## 2. Setup scene, model, optimizer, checkpoint

```python
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
```
Khởi tạo `GaussianModel`, `Scene`, thiết lập optimizer, và khôi phục từ checkpoint nếu có. Xác định màu nền train (trắng/đen).

## 3. Chuẩn bị camera stack & timer

```python
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background
```
Dùng CUDA event để đo thời gian render/optimize riêng biệt. Tạo danh sách camera huấn luyện dạng "stack" để lấy ngẫu nhiên không lặp.

## 4. Vòng lặp chính — Network GUI (nếu bật websockets)

```python
    for iteration in range(first_iter, opt.iterations + 1):
        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
```
Cho phép xem preview render trực tiếp qua giao diện web (websocket) trong lúc train.

## 5. Cập nhật learning rate & SH degree

```python
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
```
Learning rate được lịch trình theo iteration (decay). Cứ mỗi 1000 iter thì tăng bậc Spherical Harmonics (chi tiết màu sắc góc nhìn) lên 1 mức.

## 6. Chọn camera ngẫu nhiên

```python
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)
```
Lấy ngẫu nhiên 1 camera chưa dùng trong "epoch" hiện tại; khi hết thì reset lại toàn bộ.

## 7. Render + tính loss + backward

```python
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()
```
Đây là phần lõi: render ảnh từ Gaussians theo camera đã chọn, so sánh với ảnh gốc bằng loss kết hợp L1 + D-SSIM, rồi lan truyền ngược gradient.

## 8. Logging progress bar & lưu checkpoint

```python
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            optim_start.record()
```
Loss được làm mượt bằng EMA để hiển thị. Lưu model tại các iteration được chỉ định.

## 9. Densification (thêm/tách Gaussian) — đặc trưng FastGS

```python
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)                    
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
```
Khác với 3DGS gốc (chỉ dựa vào gradient tích lũy), FastGS tính điểm "importance/pruning" dựa trên **nhiều camera lấy mẫu cùng lúc** (`sampling_cameras` + `compute_gaussian_score_fastgs`) để densify/prune nhất quán đa góc nhìn (multiview-consistent), thay vì chỉ theo 1 view như bản gốc.

## 10. Pruning cuối chu kỳ (sau iteration 15k, trước 30k) — tối ưu riêng của FastGS

```python
            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)                    
                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
```
Giai đoạn model đã hội tụ cơ bản, nên có thể prune (loại bỏ Gaussian dư thừa) mạnh tay hơn (`min_opacity=0.1` cao hơn) mà không ảnh hưởng chất lượng render — giúp giảm số Gaussian, tăng tốc độ render.

## 11. Optimizer step & đo thời gian

```python
            if iteration < opt.iterations:
                gaussians.optimizer_step(iteration)

            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3
```
Cập nhật tham số bằng optimizer, đồng bộ CUDA để đo chính xác thời gian mỗi iteration (render + optimize).

## 12. Kết thúc train — in thống kê

```python
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
```

## 13. Setup output dir & logger

```python
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    return tb_writer
```
Tạo thư mục output, lưu config, khởi tạo TensorBoard writer.

## 14. Hàm đánh giá (test/train) — hiện đang bị comment out trong vòng lặp chính

```python
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs):
    ...
    for config in validation_configs:
        ...
        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        ssim_test += fast_ssim(...).mean().double()
        lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
```
Tính PSNR, SSIM, LPIPS trên tập test và một số ảnh train để log lên TensorBoard.

## 15. Entry point `__main__` — parse argument & chạy training

```python
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    ...
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    safe_state(args.quiet)
    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args.websockets)
```

---

**Tóm tắt pipeline tổng thể:**
1. Load scene + init Gaussian model → 2. Vòng lặp iteration: render → tính loss (L1+SSIM) → backward → 3. Densify/prune Gaussian (dùng kỹ thuật multiview-consistent riêng của FastGS) → 4. Optimizer step → 5. Lặp lại đến khi đủ iteration → 6. Save model cuối cùng.

Điểm khác biệt cốt lõi so với 3DGS gốc nằm ở bước densify/prune (mục 9, 10) — dùng `compute_gaussian_score_fastgs` trên nhiều camera lấy mẫu thay vì chỉ dựa vào gradient của 1 view, giúp chọn lọc Gaussian chính xác và nhất quán hơn giữa các góc nhìn.