import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

from util import get_image_to_tensor_balanced, get_mask_to_tensor

import sys
sys.path.insert(0, "../model")
from camera_model import ortho2rotation, rotation2orth, make_rand_axis, R_axis_angle

image_to_tensor = get_image_to_tensor_balanced()
mask_to_tensor = get_mask_to_tensor()

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_real_data(basedir, half_res=False, args=None, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(
            os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r'
        ) as fp:
            metas[s] = json.load(fp)
    
    all_imgs = []
    all_poses = []

    counts = [0]
    for i, s in enumerate(splits):
        meta = metas[s]
        imgs = []
        poses = []

        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip
        
        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, f"{s}/{frame['file_path']}")
            img = imageio.imread(fname)
            imgs.append(img)

            pose = np.array(frame["transform_matrix"])
            poses.append(pose)

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    i_train, _, _ = i_split

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    focal_x = float(metas["train"]["fl_x"])
    focal_y = float(metas["train"]["fl_y"])
    cx = float(metas["train"]["cx"])
    cy = float(metas["train"]["cy"])

    noisy_focal_x = focal_x
    noisy_focal_y = focal_y
    if args.initial_noise_size_intrinsic != 0.0:
        noisy_focal_x = focal_x * (1 + args.initial_noise_size_intrinsic)
        noisy_focal_y = focal_y * (1 + args.initial_noise_size_intrinsic)
        print("Starting with noise in intrinsic parameters")
    
    poses_update = poses.copy()

    if args.initial_noise_size_rotation != 0.0:
        
        angle_noise = (
            np.random.rand(poses.shape[0], 1) - 0.5
        ) * 2 * args.initial_noise_size_rotation * np.pi / 180
        rotation_axis = mask_rand_axis(poses.shape[0])
        rotmat = R_axis_angle(rotation_axis, angle_noise)

        poses_update[i_train, :3, :3] = rotmat[i_train] @ poses_update[i_train, :3, :3]

        print("Starting with noise in rotation")
    
    if args.initial_noise_size_translation != 0.0:
        individual_noise = (
            np.random.rand(poses.shape[0], 3) - 0.5
        ) * 2 * args.initial_noise_size_translation

        assert (
            np.all(
                np.abs(individual_noise) < args.initial_noise_size_translation
            )
        )

        poses_update[i_train, :3, 3] = poses_update[i_train, :3, 3] + individual_noise[i_train]
        print("Starting with noise in translation")

    if args.run_without_colmap != "none":
        if args.run_without_colmap in ["both", "rot"]:
            print("Starting without colmap initialization in rotation")
            base_array = np.array([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]
            ])[None]
            poses_update[i_train, :3, :3] = base_array
        
        if args.run_without_colmap in ["both", "trans"]:
            print("Starting without colmap initialization in translation")
            poses_update[i_train, :3, :3] = 0.
        
    intrinsic_gt = torch.tensor([
        [focal_x, 0, cx, 0],
        [0, focal_y, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).cuda().float()

    print(f"Original focal length: {focal_x}, {focal_y}")
    print(f"Initial noisy focal length: {noisy_focal_x}, {noisy_focal_y}")

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(
                -180, 180, 40 + 1
            )[:-1]
        ],
        dim=0
    )

    extrinsic_gt = torch.from_numpy(poses).cuda().float()

    return imgs, poses_update, render_poses, [H, W, noisy_focal_x, noisy_focal_y], cx, cy, i_split, (intrinsic_gt, extrinsic_gt)