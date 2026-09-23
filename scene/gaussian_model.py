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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, fov2focal
from utils.general_utils import strip_symmetric, build_scaling_rotation

# --- MCMC densification (3DGS-MCMC) support ---------------------------------
# Ported from the closed-form relocation formula (Eq. 9 of the 3DGS-MCMC paper),
# as implemented in faster-gaussian-splatting/FasterGSCudaBackend/.../kernels_mcmc.cuh.
# Kept as module-level helpers (not GaussianModel state) so this stays a pure
# additive diff on top of the existing class.
_MCMC_MAX_N_SAMPLES = 51
_mcmc_cum_coeff_cache = {}

def _mcmc_get_cum_coefficients(device):
    """Cache of cumulative binomial coefficients used by the relocation formula.

    cum[n, k] = sum_{n'=k}^{n} C(n', k) * (-1)^k / sqrt(k + 1), for k <= n' (else 0).
    """
    key = str(device)
    cached = _mcmc_cum_coeff_cache.get(key)
    if cached is None:
        n_max = _MCMC_MAX_N_SAMPLES
        coeff = np.zeros((n_max, n_max), dtype=np.float64)
        for n in range(n_max):
            binom = 1.0
            sign = 1.0
            for k in range(n + 1):
                coeff[n, k] = binom * sign / np.sqrt(k + 1)
                binom *= (n - k) / (k + 1)
                sign = -sign
        cached = torch.tensor(np.cumsum(coeff, axis=0), dtype=torch.float32, device=device)
        _mcmc_cum_coeff_cache[key] = cached
    return cached


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.spatial_lr_scale = 0
        # Mip-Splatting-style 3D anti-aliasing filter — luôn bật.
        # See faster-gaussian-splatting/Model.py:150-201 and filter3d.cu for the reference formulation.
        self._filter_3d = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        xyz_gradient_accum_abs, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        if self._filter_3d is not None and self._filter_3d.shape[0] == self._scaling.shape[0]:
            # Clamp the raw (log-space) scale from below so screen-space footprint never
            # drops under one pixel worth of extent -> removes aliasing when zooming/downsampling.
            # Densify/prune inside densify_and_prune_fastgs changes the point count before
            # compute_3d_filter() re-syncs _filter_3d, so a stale (mismatched) filter is
            # skipped here rather than crashing on the broadcast.
            return self.scaling_activation(torch.maximum(self._scaling, self._filter_3d))
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]

        # Fused-CUDA Adam (Faster-GS-inspired), luôn dùng: xem utils/fused_adam.py.
        from utils.fused_adam import FusedAdam
        self.optimizer = FusedAdam(l, lr=0.0, eps=1e-15)
        self.shoptimizer = FusedAdam(sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def optimizer_step(self, iteration):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        elif iteration <= 20000:
            if iteration % 32 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 64 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)

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

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # abs
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score = None, pruning_score = None):
        
        ''' 
            Densification and Pruning based on FastGS criteria:
            1.  The gaussians candidate for densification are selected based on the gradient of their position first.
            2.  Then, based on their average metric score (computed over multiple sampled views), they are either densified (cloned) or split.
                This is our main contribution compared to the vanilla 3DGS.
            3.  Finally, gaussians with low opacity or very large size are pruned.
        '''
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # This is our multi-view consisent metric for densification
        # We use this metric to further filter the candidates for densification, which is similar to taming 3dgs.
        metric_mask = importance_score > 5

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        scores = 1 - pruning_score 
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)

        # The budget is not necessary for our method.
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            self.prune_points(final_prune)
        
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def final_prune_fastgs(self, min_opacity, pruning_score = None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(final_prune)

    def _reorder_optimizer(self, order):
        """Applies `order` (a permutation of indices) to every optimizable tensor and its
        Adam state, following the same pattern as `_prune_optimizer`/`cat_tensors_to_optimizer`."""
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][order]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][order]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(group["params"][0][order].requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][order].requires_grad_(True))

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def apply_morton_ordering(self):
        """Reorders all Gaussians (and their optimizer state) by 3D Morton code.

        Purely a memory-locality optimization (nearby Gaussians end up nearby in memory,
        which the rasterizer's tile-based sort/blend benefits from) — does not change any
        rendering math, densification/pruning logic, or the number of Gaussians. Safe to
        call at any point during training; the training loop calls it every
        `morton_reorder_interval` iterations (default 5000, always on).
        """
        with torch.no_grad():
            xyz = self._xyz.detach()
            mins = xyz.min(dim=0).values
            maxs = xyz.max(dim=0).values
            span = (maxs - mins).clamp_min(1e-8)
            n_bits = 10  # 3 * 10 = 30-bit code, fits comfortably in int64
            scale = float((1 << n_bits) - 1)
            quantized = ((xyz - mins) / span * scale).clamp(0, scale).long()

            def spread_bits(v):
                # Interleave `n_bits` bits of v with two zero bits after each bit (part1by2).
                v = v & ((1 << n_bits) - 1)
                v = (v | (v << 16)) & 0x030000FF
                v = (v | (v << 8)) & 0x0300F00F
                v = (v | (v << 4)) & 0x030C30C3
                v = (v | (v << 2)) & 0x09249249
                return v

            code = (spread_bits(quantized[:, 0])
                    | (spread_bits(quantized[:, 1]) << 1)
                    | (spread_bits(quantized[:, 2]) << 2))
            order = torch.argsort(code)

            optimizable_tensors = self._reorder_optimizer(order)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.max_radii2D = self.max_radii2D[order]
            self.xyz_gradient_accum = self.xyz_gradient_accum[order]
            self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[order]
            self.denom = self.denom[order]
            if getattr(self, "tmp_radii", None) is not None:
                self.tmp_radii = self.tmp_radii[order]
            if self._filter_3d is not None:
                self._filter_3d = self._filter_3d[order]

    def setup_3d_filter(self, cameras, filter_variance=0.2):
        """Enables the Mip-Splatting-style 3D anti-aliasing filter and computes it for the
        first time. `cameras` is any iterable of `scene.cameras.Camera` (e.g.
        `scene.getTrainCameras()`). Ported from faster-gaussian-splatting/Model.py:150-201
        ("optimized formulation": clamps scale directly in log-space, no opacity correction
        needed) — see that file for the reference derivation."""
        max_focal = 1e-12
        for cam in cameras:
            max_focal = max(max_focal, fov2focal(cam.FoVx, cam.image_width), fov2focal(cam.FoVy, cam.image_height))
        self.distance2filter = (filter_variance ** 0.5) / max_focal
        self.compute_3d_filter(cameras)

    def compute_3d_filter(self, cameras, clipping_tolerance=0.15):
        """Recomputes the 3D filter buffer; must be re-run whenever the number of Gaussians
        changes (after densify/prune) or after `setup_3d_filter`. Filter is always active
        (see `get_scaling`), gated only by `self._filter_3d is not None`."""
        means = self._xyz.detach()
        n_points = means.shape[0]
        ones = torch.ones((n_points, 1), device=means.device, dtype=means.dtype)
        homogeneous = torch.cat([means, ones], dim=1)

        filter_3d = torch.full((n_points, 1), fill_value=float("inf"), device=means.device, dtype=torch.float32)
        visibility_mask = torch.zeros((n_points, 1), device=means.device, dtype=torch.bool)

        for cam in cameras:
            p_view = homogeneous @ cam.world_view_transform  # row-vector convention (see scene/cameras.py)
            z = p_view[:, 2:3]
            x = p_view[:, 0:1]
            y = p_view[:, 1:2]

            width, height = float(cam.image_width), float(cam.image_height)
            focal_x = fov2focal(cam.FoVx, width)
            focal_y = fov2focal(cam.FoVy, height)
            bounds_factor = clipping_tolerance + 0.5
            max_x_shifted = bounds_factor * width
            max_y_shifted = bounds_factor * height
            # principal point assumed centered -> principal_offset terms cancel out
            left, right = -max_x_shifted / focal_x, max_x_shifted / focal_x
            top, bottom = -max_y_shifted / focal_y, max_y_shifted / focal_y

            in_view = (z >= cam.znear) & (z <= cam.zfar)
            in_view &= (x >= left * z) & (x <= right * z)
            in_view &= (y >= top * z) & (y <= bottom * z)

            candidate = self.distance2filter * z
            update = in_view & (candidate < filter_3d)
            filter_3d = torch.where(update, candidate, filter_3d)
            visibility_mask |= in_view

        if visibility_mask.any():
            filter_3d_max = filter_3d[visibility_mask].max()
            filter_3d = torch.where(visibility_mask, filter_3d, filter_3d_max)
        else:
            filter_3d = torch.zeros_like(filter_3d)

        # store in log-space so it can be used directly as a lower clamp on `self._scaling`
        self._filter_3d = filter_3d.clamp_min(1e-12).log()

    # --- MCMC densification (3DGS-MCMC), alternative densification mode --------
    # Not wired into any training control-flow: densify_and_prune_fastgs/
    # final_prune_fastgs are the only densification path used. Kept here as a
    # ready-to-use alternative for manual benchmarking (call directly instead
    # of densify_and_prune_fastgs if you want to compare).

    def _mcmc_reset_state(self, indices):
        """Zero the Adam moment estimates at `indices` (both optimizer and shoptimizer)."""
        optimizers = [self.optimizer]
        if self.shoptimizer:
            optimizers.append(self.shoptimizer)
        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"][indices] = 0
                    stored_state["exp_avg_sq"][indices] = 0

    def _mcmc_relocate(self, old_opacity, old_scale, n_samples):
        """Closed-form relocation (3DGS-MCMC Eq. 9): redistribute one Gaussian's
        opacity/scale across `n_samples` copies of itself so the sum stays consistent.

        old_opacity: (M, 1) activated opacity in (0, 1).
        old_scale:   (M, 3) activated scale.
        n_samples:   (M,) int64, number of copies each sampled Gaussian ended up with (>=1).
        Returns (new_opacity (M, 1), new_scale (M, 3)), both activated.
        """
        n_samples = n_samples.clamp(min=1, max=_MCMC_MAX_N_SAMPLES)
        old_opacity_flat = old_opacity.flatten()
        new_opacity = 1.0 - (1.0 - old_opacity_flat).pow(1.0 / n_samples.float())

        k_idx = torch.arange(_MCMC_MAX_N_SAMPLES, device=old_opacity.device, dtype=torch.float32)
        power = new_opacity.unsqueeze(1).pow(k_idx.unsqueeze(0) + 1.0)  # (M, max_n) = new_opacity^(k+1)

        cum_coeff = _mcmc_get_cum_coefficients(old_opacity.device)  # (max_n, max_n)
        gathered = cum_coeff[(n_samples - 1).long()]  # (M, max_n), zero for k > n_samples-1
        denominator = (gathered * power).sum(dim=1).clamp_min(1e-12)

        scaling_factor = old_opacity_flat / denominator
        new_scale = scaling_factor.unsqueeze(1) * old_scale
        return new_opacity.unsqueeze(1), new_scale

    @torch.no_grad()
    def mcmc_densification(self, min_opacity, cap_max):
        """3DGS-MCMC densification: relocate dead Gaussians by resampling from alive
        ones (weighted by opacity), then grow the population up to `cap_max`.
        This is an alternative to densify_and_prune_fastgs. Not called anywhere in the
        default training loop; invoke it manually if you want to benchmark it.
        """
        eps = torch.finfo(torch.float32).eps
        dead_mask = (self.get_opacity.flatten() <= min_opacity) | (self._rotation.pow(2).sum(dim=1) < 1e-8)
        n_dead = int(dead_mask.sum().item())
        if n_dead > 0:
            dead_indices = torch.where(dead_mask)[0]
            alive_indices = torch.where(~dead_mask)[0]
            opacities = self.get_opacity.flatten()
            sampled_local = torch.multinomial(opacities[alive_indices], n_dead, replacement=True)
            sampled_indices = alive_indices[sampled_local]

            _, inverse, counts_per_unique = sampled_indices.unique(sorted=False, return_inverse=True, return_counts=True)
            counts = counts_per_unique[inverse] + 1  # +1 for the original Gaussian being kept alive too

            new_opacity, new_scale = self._mcmc_relocate(
                opacities[sampled_indices].unsqueeze(1),
                self.get_scaling[sampled_indices],
                counts,
            )
            new_opacity_raw = self.inverse_opacity_activation(new_opacity.clamp(min_opacity, 1.0 - eps))
            new_scale_raw = self.scaling_inverse_activation(new_scale.clamp_min(1e-8))

            self._opacity[sampled_indices] = new_opacity_raw
            self._scaling[sampled_indices] = new_scale_raw

            self._xyz[dead_indices] = self._xyz[sampled_indices]
            self._features_dc[dead_indices] = self._features_dc[sampled_indices]
            self._features_rest[dead_indices] = self._features_rest[sampled_indices]
            self._opacity[dead_indices] = new_opacity_raw
            self._scaling[dead_indices] = new_scale_raw
            self._rotation[dead_indices] = self._rotation[sampled_indices]

            self._mcmc_reset_state(sampled_indices)

        current_n_points = self.get_xyz.shape[0]
        n_target = min(cap_max, int(1.05 * current_n_points))
        n_added = max(0, n_target - current_n_points)
        if n_added > 0:
            opacities = self.get_opacity.flatten()
            sampled_indices = torch.multinomial(opacities, n_added, replacement=True)

            _, inverse, counts_per_unique = sampled_indices.unique(sorted=False, return_inverse=True, return_counts=True)
            counts = counts_per_unique[inverse] + 1

            new_opacity, new_scale = self._mcmc_relocate(
                opacities[sampled_indices].unsqueeze(1),
                self.get_scaling[sampled_indices],
                counts,
            )
            new_opacity_raw = self.inverse_opacity_activation(new_opacity.clamp(min_opacity, 1.0 - eps))
            new_scale_raw = self.scaling_inverse_activation(new_scale.clamp_min(1e-8))

            self._opacity[sampled_indices] = new_opacity_raw
            self._scaling[sampled_indices] = new_scale_raw

            new_xyz = self._xyz[sampled_indices].clone()
            new_features_dc = self._features_dc[sampled_indices].clone()
            new_features_rest = self._features_rest[sampled_indices].clone()
            new_rotation = self._rotation[sampled_indices].clone()

            optimizable_tensors = self.cat_tensors_to_optimizer({
                "xyz": new_xyz,
                "f_dc": new_features_dc,
                "f_rest": new_features_rest,
                "opacity": new_opacity_raw.clone(),
                "scaling": new_scale_raw.clone(),
                "rotation": new_rotation,
            })
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            if getattr(self, "tmp_radii", None) is not None:
                self.tmp_radii = torch.cat((self.tmp_radii, torch.zeros(n_added, device="cuda")))

            self._mcmc_reset_state(sampled_indices)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def mcmc_add_noise(self, current_lr):
        """Add positional noise scaled inversely with opacity (3DGS-MCMC): stable,
        high-opacity Gaussians barely move, low-opacity ones explore more.
        `current_lr` is the current xyz learning rate times opt.mcmc_noise_lr.
        """
        rotation_matrices = build_rotation(self.get_rotation)  # (N, 3, 3)
        variance = self.get_scaling.pow(2)  # (N, 3), == exp(2 * raw_scale)
        cov3d = rotation_matrices @ torch.diag_embed(variance) @ rotation_matrices.transpose(1, 2)

        noise = torch.randn_like(self._xyz)
        transformed_noise = torch.bmm(cov3d, noise.unsqueeze(-1)).squeeze(-1)

        opacity = self.get_opacity.flatten()
        op_sigmoid = torch.sigmoid(0.5 - 100.0 * opacity)
        noise_factor = current_lr * op_sigmoid

        self._xyz.add_(noise_factor.unsqueeze(1) * transformed_noise)
