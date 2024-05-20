#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import torch
import pickle
import random
import numpy as np
from tqdm.rich import tqdm
from copy import deepcopy
sys.path.append('./')

from utils.lmdb_utils import LMDBEngine
from engines.core_engine import TrackEngine

class Tracker:
    def __init__(self, focal_length=12.0, device='cuda'):
        self._device = device
        self.tracker = TrackEngine(focal_length=focal_length, device=device)

    def track_lmdb(self, lmdb_path, dir_path=None, ):
        # build name
        data_name = os.path.basename(lmdb_path)
        output_path = os.path.join(f'outputs/{dir_path}', data_name) if dir_path else f'outputs/{data_name}'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # build video data
        print('Load lmdb data...')
        lmdb_engine = LMDBEngine(lmdb_path, write=False)
        # track base
        base_results = self.run_base(lmdb_engine, output_path, )
        # track lightning
        lightning_results = self.run_lightning(base_results, lmdb_engine, output_path)
        self.run_visualization(base_results, lightning_results, lmdb_engine, output_path)
        lmdb_engine.close()

    def run_base(self, lmdb_engine, output_path, ):
        print('Track with emoca/mica/insightface/mediapipe/...')
        all_keys = lmdb_engine.keys()
        if not os.path.exists(os.path.join(output_path, 'base.pkl')):
            base_results = {}
            for key in tqdm(all_keys, ncols=80, colour='#95bb72'):
                base_results[key] = self.tracker.track_base(lmdb_engine[key])
            with open(os.path.join(output_path, 'base.pkl'), 'wb') as f:
                pickle.dump(base_results, f)
        else:
            with open(os.path.join(output_path, 'base.pkl'), 'rb') as f:
                base_results = pickle.load(f)
        return base_results

    def run_lightning(self, base_results, lmdb_engine, output_path,):
        print('Track lightning...')
        if not os.path.exists(os.path.join(output_path, 'lightning.pkl')):
            lightning_result = self.tracker.track_lightning(base_results, lmdb_engine, output_path)
            with open(os.path.join(output_path, 'lightning.pkl'), 'wb') as f:
                pickle.dump(lightning_result, f)
        else:
            with open(os.path.join(output_path, 'lightning.pkl'), 'rb') as f:
                lightning_result = pickle.load(f)
        return lightning_result

    def run_visualization(self, base_results, data_result, lmdb_engine, output_path):
        import torchvision
        from engines.FLAME import FLAMEDense
        from utils.renderer_utils import Mesh_Renderer
        self.verts_scale = self.tracker.calibration_results['verts_scale']
        self.focal_length = self.tracker.calibration_results['focal_length']
        self.principal_point = self.tracker.calibration_results['principal_point']
        self.flame_model = FLAMEDense(n_shape=100, n_exp=50).to(self._device)
        print('Visualize results...')
        mesh_render = Mesh_Renderer(
            512, faces=self.flame_model.get_faces().cpu().numpy(), device=self._device
        )
        frames = list(data_result.keys())
        random.shuffle(frames)
        frames = frames[:20]
        vis_images = []
        for frame in tqdm(frames, ncols=80, colour='#95bb72'):
            vertices, _, _ = self.flame_model(
                shape_params=torch.tensor(data_result[frame]['mica_shape'], device=self._device)[None],
                expression_params=torch.tensor(data_result[frame]['emoca_expression'], device=self._device)[None],
                pose_params=torch.tensor(data_result[frame]['emoca_pose'], device=self._device)[None].float()
            )
            vertices = vertices*self.verts_scale
            gt_kps = torch.tensor(base_results[frame]['kps'])[None]
            gt_lmk_68 = torch.tensor(base_results[frame]['lmks'])[None]
            gt_lmk_dense = torch.tensor(base_results[frame]['lmks_dense'])[self.flame_model.mediapipe_idx.cpu()][None]
            images, alpha_images = mesh_render(
                vertices, transform_matrix=torch.tensor(data_result[frame]['transform_matrix'], device=self._device)[None],
                focal_length=self.focal_length, principal_point=self.principal_point
            )
            alpha_images = alpha_images[0].expand(3, -1, -1)
            vis_image_0 = lmdb_engine[frame].to(self._device).float()
            vis_image_1 = vis_image_0.clone()
            vis_image_1[alpha_images>0.5] *= 0.3
            vis_image_1[alpha_images>0.5] += (images[0, alpha_images>0.5] * 0.7)
            vis_image_1 = vis_image_1.cpu().to(torch.uint8)
            vis_image_1 = torchvision.utils.draw_keypoints(vis_image_1, gt_kps, colors="white", radius=3)
            vis_image_1 = torchvision.utils.draw_keypoints(vis_image_1, gt_lmk_68, colors="red", radius=1.5)
            vis_image_1 = torchvision.utils.draw_keypoints(vis_image_1, gt_lmk_dense, colors="blue", radius=1.5)
            vis_image = torchvision.utils.make_grid([vis_image_0.cpu(), vis_image_1.cpu(), images[0].cpu()], nrow=3, padding=0)
            vis_images.append(vis_image/255.0)
        torchvision.utils.save_image(vis_images, os.path.join(output_path, 'tracked.jpg'), nrow=2, padding=0)


def list_all_files(dir_path):
    pair = os.walk(dir_path)
    result = []
    for path, dirs, files in pair:
        if len(files):
            for file_name in files:
                result.append(os.path.join(path, file_name))
    return result


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    # warnings.simplefilter("ignore", category=FutureWarning, lineno=0, append=False)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', '-l', required=True, type=str)
    parser.add_argument('--outdir_path', '-d', default='', type=str)
    args = parser.parse_args()
    tracker = Tracker(focal_length=12.0, device='cuda')
    tracker.track_lmdb(args.lmdb_path, dir_path=args.outdir_path, )
   
