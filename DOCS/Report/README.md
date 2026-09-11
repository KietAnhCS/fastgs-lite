# Report — Quy trình FastGS-lite theo sơ đồ pipeline 3DGS

> Báo cáo này đi theo **đúng thứ tự các khối trong sơ đồ pipeline** của 3D Gaussian Splatting, và ở mỗi khối chỉ ra
> phần nào là 3DGS gốc (giữ nguyên) và phần nào là FastGS-lite (thay đổi). Mọi công thức đối chiếu với code trong repo,
> không lấy con số nào từ paper upstream.
>
> Nguồn tham khảo (cùng thư mục `DOCS/`):
> - [`fastgs-acceleration-method.md`](../fastgs-acceleration-method.md) — tài liệu gốc, đầy đủ nhất (Phần I–IX)
> - [`3.md`](../3.md) — ví dụ số end-to-end cho Adaptive Density Control (chương 7)
> - [`box.md`](../box.md), [`box2.md`](../box2.md) — bài tập và ba công thức đếm tile cho compact box (chương 3)

![pipeline](../../pipe.png)

```
SfM Points ──► Initialization ──► 3D Gaussians ──► Projection ──► Differentiable ──► Image
                                       ▲    ▲          ▲           Tile Rasterizer      │
                        Camera ────────┼────┼──────────┘                 ▲              │
                                       │    │                            │              │
                                       │    └──── gradient ◄─────────────┴── gradient ◄─┘
                                       │
                              Adaptive Density Control
```

Mũi tên đen = **Operation Flow** (chương 1 → 5). Mũi tên xanh = **Gradient Flow** (chương 6). Vòng quay về từ
**Adaptive Density Control** (chương 7).

## Mục lục

| Chương | Khối trong sơ đồ | File | FastGS thay đổi gì |
|---|---|---|---|
| 0 | Ký hiệu chung | [00-ky-hieu.md](00-ky-hieu.md) | — |
| 1 | SfM Points → Initialization | [01-initialization.md](01-initialization.md) | giữ nguyên |
| 2 | 3D Gaussians | [02-3d-gaussians.md](02-3d-gaussians.md) | giữ nguyên |
| 3 | Camera + Projection | [03-projection.md](03-projection.md) | **compact box, lọc tile theo ellipse** (giảm $K$) |
| 4 | Differentiable Tile Rasterizer | [04-rasterizer.md](04-rasterizer.md) | giữ nguyên toán; đầu vào nhỏ hơn |
| 5 | Image → Loss & Metrics | [05-image-loss.md](05-image-loss.md) | giữ nguyên loss; ghi chú metrics |
| 6 | Gradient Flow | [06-gradient-flow.md](06-gradient-flow.md) | **gradient trị tuyệt đối, Adam thưa, lr SH** |
| 7 | Adaptive Density Control | [07-adaptive-density-control.md](07-adaptive-density-control.md) | **Importance / Pruning score, densify AND, prune multinomial, final prune** (giảm $N$) |
| 8 | Tổng hợp: mô hình chi phí | [08-tong-hop.md](08-tong-hop.md) | ba tỉ số nhân nhau |

## Kết luận trước khi đọc

FastGS-lite **không đổi một dòng nào của toán render** — $G(x)$, $\Sigma=RSS^\top R^\top$, $\Sigma'=JW\Sigma W^\top J^\top$,
alpha-blend, backward qua blend, update rule của Adam, loss $\mathcal L=(1-\lambda)\mathcal L_1+\lambda\mathcal L_{\text{D-SSIM}}$
đều giữ nguyên. Cái bị thay là ba **công thức điều khiển**:

| Điều khiển | Khối | Hiệu ứng lên chi phí $T_{\text{iter}}=aN+bNK+cN\cdot\mathbb 1_{\text{step}}+F$ |
|---|---|---|
| Cặp (tile, Gaussian) nào được đưa vào sort/blend | Projection | giảm $K$ |
| Gaussian nào được tồn tại | Adaptive Density Control | giảm $N$ |
| Khi nào gọi `optimizer.step()` | Gradient Flow | giảm $\mathbb 1_{\text{step}}$ |
