#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import math
import torch
import torchvision
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from engines.FLAME import FLAMEDense
from utils.renderer_utils import Mesh_Renderer, Point_Renderer

class Lightning_Engine:
    def __init__(self, device='cuda', lazy_init=True):
        self._device = device

    def init_model(self, calibration_results, image_size=512):
        # camera params
        self.image_size = image_size
        self.verts_scale = calibration_results['verts_scale']
        self.focal_length = calibration_results['focal_length'].to(self._device)
        self.principal_point = calibration_results['principal_point'].to(self._device)
        # build flame
        self.flame_model = FLAMEDense(n_shape=100, n_exp=50).to(self._device)
        # print('Lightning Engine Init Done.')

    def _build_cameras_kwargs(self, batch_size):
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self._device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': self.principal_point.repeat(batch_size, 1), 'focal_length': self.focal_length, 
            'image_size': screen_size, 'device': self._device,
        }
        return cameras_kwargs

    def lightning_optimize(self, track_frames, batch_emoca, shape_code, batch_frames=None, steps=500):
        batch_size = len(track_frames)
        cameras_kwargs = self._build_cameras_kwargs(batch_size)
        # flame params
        base_rotation = batch_emoca['emoca_pose'][:, :3].clone().float()
        batch_emoca['emoca_pose'][:, :3] *= 0
        if shape_code is None:
            shape_code = batch_emoca['emoca_shape'].float()
        else:
            shape_code = shape_code[None].expand(batch_size, -1)
        vertices, pred_lmk_68, pred_lmk_dense = self.flame_model(
            shape_params=shape_code, 
            expression_params=batch_emoca['emoca_expression'].float(), 
            pose_params=batch_emoca['emoca_pose'].float()
        )
        vertices = vertices * self.verts_scale
        pred_lmk_68, pred_lmk_dense = pred_lmk_68 * self.verts_scale, pred_lmk_dense * self.verts_scale
        # build params
        cameras = PerspectiveCameras(**cameras_kwargs)
        base_transform_p3d = self.transform_emoca_to_p3d(
            base_rotation, pred_lmk_dense, batch_emoca['lmks_dense'], self.image_size
        )
        rotation = torch.nn.Parameter(matrix_to_rotation_6d(base_transform_p3d[:, :3, :3]))
        base_transform_p3d[:, :3, 3] = base_transform_p3d[:, :3, 3] * self.focal_length / 13.0 * self.verts_scale / 5.0
        translation = torch.nn.Parameter(base_transform_p3d[:, :, 3])
        expression = torch.nn.Parameter(batch_emoca['emoca_expression'].float())
        params = [
            {'params': [translation], 'lr': 0.01}, {'params': [rotation], 'lr': 0.005},
            {'params': [expression], 'lr': 0.002}
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)
        # run
        for idx in range(steps):
            vertices, _, pred_lmk_dense = self.flame_model(
                shape_params=shape_code, 
                expression_params=expression, 
                pose_params=batch_emoca['emoca_pose'].float()
            )
            vertices, pred_lmk_dense = vertices*self.verts_scale, pred_lmk_dense*self.verts_scale
            points_dense = cameras.transform_points_screen(
                pred_lmk_dense, R=rotation_6d_to_matrix(rotation), T=translation
            )[..., :2]

            loss_key_lmk = lmk_loss(
                points_dense, batch_emoca['lmks_dense'][:, self.flame_model.mediapipe_idx], self.image_size
            ) * 10
            all_loss = loss_key_lmk
            # print(all_loss)
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()
        if batch_frames is not None:
            # point_render = Point_Renderer(image_size=self.image_size, device=self._device)
            with torch.no_grad():
                mesh_render = Mesh_Renderer(
                    512, faces=self.flame_model.get_faces().cpu().numpy(), device=self._device
                )
                # points_image = point_render(vertices)
                images, alpha_images = mesh_render(vertices, cameras)
                vis_images = []
                alpha_images = alpha_images.expand(-1, 3, -1, -1)
                for idx, frame in enumerate(batch_frames[:20]):
                    vis_i = frame.clone()
                    vis_i[alpha_images[idx]>0.5] *= 0.5
                    vis_i[alpha_images[idx]>0.5] += (images[idx, alpha_images[idx]>0.5] * 0.5)
                    # vis_image = torchvision.utils.make_grid([vis_i.cpu(), points_image[idx].cpu(), ], nrow=2)
                    vis_images.append(vis_i.cpu()/255.0)
                visualization = torchvision.utils.make_grid(vis_images, nrow=4)
        else:
            visualization = None
        lightning_results = {}
        transform_matrix = torch.cat([rotation_6d_to_matrix(rotation), translation[:, :, None]], dim=-1)
        for idx, name in enumerate(track_frames):
            lightning_results[name] = {
                'emoca_shape': shape_code[idx].detach().float().cpu(),
                'emoca_expression': batch_emoca['emoca_expression'][idx].detach().float().cpu(),
                'emoca_pose': batch_emoca['emoca_pose'][idx].detach().float().cpu(),
                'transform_matrix': transform_matrix[idx].detach().float().cpu(),
            }
        return lightning_results, visualization

    @staticmethod
    def transform_emoca_to_p3d(emoca_base_rotation, pred_lmks, gt_lmks, image_size):
        device = emoca_base_rotation.device
        batch_size = emoca_base_rotation.shape[0]
        initial_trans = torch.tensor([[0, 0, 5000.0/image_size]]).to(device)
        emoca_base_rotation[:, 1] += math.pi
        emoca_base_rotation = emoca_base_rotation[:, [2, 1, 0]]
        emoca_base_rotation = batch_rodrigues(emoca_base_rotation)
        base_transform = torch.cat([
                transform_inv(emoca_base_rotation), 
                initial_trans.reshape(1, -1, 1).repeat(batch_size, 1, 1)
            ], dim=-1
        )
        base_transform_p3d = transform_opencv_to_p3d(base_transform)
        # find translate
        cameras = PerspectiveCameras(
            device=device,
            image_size=torch.tensor([[image_size, image_size]], device=device).repeat(batch_size, 1)
        )
        pred_lmks = cameras.transform_points_screen(
            pred_lmks.clone(),
            R=base_transform_p3d[:, :3, :3], T=base_transform_p3d[:, :3, 3], 
            principal_point=torch.zeros(batch_size, 2), focal_length=5000.0/image_size
        )[..., :2]
        trans_xy = (pred_lmks.mean(dim=1)[..., :2] - gt_lmks.mean(dim=1)[..., :2]) * 2 / image_size
        base_transform_p3d[:, :2, 3] = trans_xy
        return base_transform_p3d


def lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask=None):
    size = torch.tensor([1 / image_size, 1 / image_size], device=opt_lmks.device).float()[None, None, ...]
    diff = torch.pow(opt_lmks - target_lmks, 2)
    if lmk_mask is None:
        return (diff * size).mean()
    else:
        return (diff * size * lmk_mask).mean()


def intrinsic_opencv_to_p3d(focal_length, principal_point, image_size):
    half_size = image_size/2
    focal_length = focal_length / half_size
    principal_point = -(principal_point - half_size) / half_size
    return focal_length, principal_point


def transform_opencv_to_p3d(opencv_transform, verts_scale=1, type='w2c'):
    assert type in ['w2c', 'c2w']
    if opencv_transform.dim() == 3:
        return torch.stack([transform_opencv_to_p3d(t, verts_scale, type) for t in opencv_transform])
    if type == 'c2w':
        if opencv_transform.shape[-1] != opencv_transform.shape[-2]:
            new_transform = torch.eye(4).to(opencv_transform.device)
            new_transform[:3, :] = opencv_transform
            opencv_transform = new_transform
        opencv_transform = torch.linalg.inv(opencv_transform) # c2w to w2c
    rotation = opencv_transform[:3, :3]
    rotation = rotation.permute(1, 0)
    rotation[:, :2] *= -1
    if opencv_transform.shape[-1] == 4:
        translation = opencv_transform[:3, 3] * verts_scale
        translation[:2] *= -1
        rotation = torch.cat([rotation, translation.reshape(-1, 1)], dim=-1)
    return rotation


def transform_inv(transforms):
    if transforms.dim() == 3:
        return torch.stack([transform_opencv_to_p3d(t) for t in transforms])
    if transforms.shape[-1] != transforms.shape[-2]:
        new_transform = torch.eye(4)
        new_transform[:3, :] = transforms
        transforms = new_transform
    transforms = torch.linalg.inv(transforms)
    return transforms[:3]


def batch_rodrigues(rot_vecs,):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat
