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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

# Camera models whose first parameters are the focal length(s). Distortion
# coefficients are ignored: the renderer is a pinhole model, so a distorted
# dataset is treated as its pinhole approximation instead of failing outright.
SINGLE_FOCAL_MODELS = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                       "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"}
DUAL_FOCAL_MODELS = {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
                     "THIN_PRISM_FISHEYE"}

def colmap_focals(intr):
    """Return (fx, fy) for a COLMAP intrinsics record."""
    if intr.model in SINGLE_FOCAL_MODELS:
        return intr.params[0], intr.params[0]
    if intr.model in DUAL_FOCAL_MODELS:
        return intr.params[0], intr.params[1]
    raise ValueError("Colmap camera model not handled: {}".format(intr.model))

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    n_missing = 0
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        focal_length_x, focal_length_y = colmap_focals(intr)
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if not os.path.exists(image_path):
            # The COLMAP model is reconstructed over every frame, but a
            # split only ships its own images. Frames without a file
            # belong to another split; skip them instead of crashing.
            n_missing += 1
            continue
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    if n_missing:
        print("Skipped {}/{} COLMAP cameras with no image in {}".format(
            n_missing, len(cam_extrinsics), images_folder))
    if not cam_infos:
        raise RuntimeError(
            "No COLMAP camera matched an image in {}".format(images_folder))
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def generate_random_point_cloud(cam_infos, nerf_normalization, n_points=100_000,
                                 carve=True, carve_in_all_frustums=False):
    """Faster-GS-style random initialization fallback (see faster-gaussian-splatting/utils.py
    `carve()` for the reference formulation), used when COLMAP has no usable points3D.

    Samples `n_points` uniformly inside the scene's bounding sphere (from `nerf_normalization`),
    then optionally keeps only points inside at least one camera's frustum (`carve_in_all_frustums
    =False`) or every camera's frustum (`True`). Pure numpy — no torch/CUDA required. Falls back
    to the uncarved point set if carving would remove everything.
    """
    center = -nerf_normalization["translate"]
    radius = nerf_normalization["radius"]

    # uniform sampling inside a sphere: sample direction + radius^(1/3)
    directions = np.random.normal(size=(n_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-8)
    radii = radius * np.cbrt(np.random.uniform(0.0, 1.0, size=(n_points, 1)))
    xyz = center[None, :] + directions * radii

    if carve and cam_infos:
        visible_counts = np.zeros(n_points, dtype=np.int32)
        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)  # 4x4, world -> camera (column-vector convention)
            cam_coords = xyz @ W2C[:3, :3].T + W2C[:3, 3]
            z = cam_coords[:, 2]
            x = cam_coords[:, 0]
            y = cam_coords[:, 1]
            tan_x = np.tan(cam.FovX * 0.5)
            tan_y = np.tan(cam.FovY * 0.5)
            in_frustum = (z > 1e-4) & (np.abs(x) <= z * tan_x) & (np.abs(y) <= z * tan_y)
            visible_counts += in_frustum.astype(np.int32)

        keep_mask = (visible_counts == len(cam_infos)) if carve_in_all_frustums else (visible_counts > 0)
        if keep_mask.any():
            xyz = xyz[keep_mask]
        # else: carving removed every point -> keep the uncarved set instead of an empty cloud

    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    rgb = SH2RGB(shs) * 255.0
    normals = np.zeros_like(xyz)
    return xyz, rgb, BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normals)

def readColmapSceneInfo(path, images, eval, llffhold=8, random_init_force=False,
                         random_init_n_points=100_000, random_init_carving=True):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    used_ply_path = ply_path
    if pcd is None or random_init_force:
        reason = "no usable points3D" if pcd is None else "random_init_force=True"
        print(f"Falling back to random point cloud initialization ({reason}): "
              f"{random_init_n_points} points, carving={random_init_carving}")
        xyz, rgb, pcd = generate_random_point_cloud(
            train_cam_infos, nerf_normalization,
            n_points=random_init_n_points, carve=random_init_carving)
        # Cache to a distinct file so real COLMAP points3D.ply is never overwritten.
        used_ply_path = os.path.join(path, "sparse/0/points3D_random_init.ply")
        storePly(used_ply_path, xyz, rgb)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=used_ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}