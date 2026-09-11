#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def _batched(img1, img2):
    # Ảnh [C, H, W] không có batch dim: nếu để nguyên, shape[0] = C và MSE bị tính
    # riêng từng kênh rồi trung bình PSNR theo kênh -> luôn cao hơn PSNR toàn ảnh.
    if img1.dim() == 3:
        img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    return img1, img2

def mse(img1, img2):
    img1, img2 = _batched(img1, img2)
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2)))
