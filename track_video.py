#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import torch
import pickle
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

    def track_video(self, video_path, dir_path=None, synthesis=False, no_vis=False):
        # build name
        data_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(f'outputs/{dir_path}', data_name) if dir_path else f'outputs/{data_name}'
        # build video data
        print('Processing video data...')
        fps = self.tracker.build_video(video_path, output_path, matting=True, background=0.0)
        lmdb_engine = LMDBEngine(os.path.join(output_path, 'img_lmdb'), write=False)
        # track base
        base_results = self.run_base(lmdb_engine, output_path)
        # track lightning
        lightning_results = self.run_lightning(base_results, lmdb_engine, output_path)
        if synthesis:
            synthesis_results = self.run_synthesis(base_results, lightning_results, lmdb_engine, output_path)
            synthesis_results = run_smoothing(synthesis_results, output_path)
            if not no_vis:
                self.run_visualization(base_results, synthesis_results, lmdb_engine, output_path, fps=fps)
        else:
            lightning_results = run_smoothing(lightning_results, output_path)
            if not no_vis:
                self.run_visualization(base_results, lightning_results, lmdb_engine, output_path, fps=fps)
        lmdb_engine.close()

    def run_base(self, lmdb_engine, output_path,):
        print('Track with emoca/mica/insightface/mediapipe/...')
        if not os.path.exists(os.path.join(output_path, 'base.pkl')):
            base_results = {}
            for key in tqdm(lmdb_engine.keys(), ncols=80, colour='#95bb72'):
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

    def run_synthesis(self, base_results, lightning_results, lmdb_engine, output_path,):
        print('Track synthesis...')
        if not os.path.exists(os.path.join(output_path, 'synthesis.pkl')):
            synthesis_result = self.tracker.track_synthesis(base_results, lightning_results, lmdb_engine, output_path)
            with open(os.path.join(output_path, 'synthesis.pkl'), 'wb') as f:
                pickle.dump(synthesis_result, f)
        else:
            with open(os.path.join(output_path, 'synthesis.pkl'), 'rb') as f:
                synthesis_result = pickle.load(f)
        return synthesis_result

    def run_visualization(self, base_results, data_result, lmdb_engine, output_path, fps=25.0):
        import torchvision
        from engines.FLAME import FLAMEDense, FLAMETex
        from utils.renderer_utils import Mesh_Renderer, Texture_Renderer
        self.verts_scale = self.tracker.calibration_results['verts_scale']
        self.focal_length = self.tracker.calibration_results['focal_length']
        self.principal_point = self.tracker.calibration_results['principal_point']
        self.flame_model = FLAMEDense(n_shape=100, n_exp=50).to(self._device)
        print('Visualize results...')
        if 'tex_params' in data_result[list(data_result.keys())[0]]:
            texture_params = torch.tensor(
                data_result[list(data_result.keys())[0]]['tex_params'], device=self._device
            )[None]
            self.flame_texture = FLAMETex(n_tex=50).to(self._device)
            mesh_render = Texture_Renderer(
                tuv = self.flame_texture.get_tuv(), flame_mask=self.flame_texture.get_face_mask(), 
                device=self._device
            )
            albedos = self.flame_texture(texture_params, image_size=512)
            sh_params = torch.tensor(
                data_result[list(data_result.keys())[0]]['sh_params'], device=self._device
            )[None]
            vis_texture = True
        else:
            mesh_render = Mesh_Renderer(
                512, faces=self.flame_model.get_faces().cpu().numpy(), device=self._device
            )
            vis_texture = False
        frames = list(data_result.keys())
        frames = sorted(frames, key=lambda x: int(x.split('_')[-1]))[:300]
        vis_images = []
        for frame in tqdm(frames, ncols=80, colour='#95bb72'):
            vertices, _, _ = self.flame_model(
                shape_params=torch.tensor(data_result[frame]['mica_shape'], device=self._device)[None],
                expression_params=torch.tensor(data_result[frame]['emoca_expression'], device=self._device)[None],
                pose_params=torch.tensor(data_result[frame]['emoca_pose'], device=self._device)[None].float()
            )
            vertices = vertices * self.verts_scale
            gt_lmk_68 = torch.tensor(base_results[frame]['lmks'])[None]
            gt_lmk_dense = torch.tensor(base_results[frame]['lmks_dense'])[self.flame_model.mediapipe_idx.cpu()][None]
            if vis_texture:
                images, masks_all, masks_face = mesh_render(
                    vertices, albedos, image_size=512, transform_matrix=torch.tensor(data_result[frame]['transform_matrix'], device=self._device)[None],
                    focal_length=self.focal_length, principal_point=self.principal_point, lights=sh_params
                )
                images = (images * 255.0).clamp(0, 255)
                masks_face = masks_face[0].expand(3, -1, -1)
                vis_image_0 = lmdb_engine[frame].to(self._device).float()
                vis_image_1 = vis_image_0.clone()
                vis_image_1[masks_face] = images[0, masks_face]
                vis_image_1 = vis_image_1.cpu().to(torch.uint8)
                vis_image_1 = torchvision.utils.draw_keypoints(vis_image_1, gt_lmk_68, colors="red", radius=1.5)
                vis_image_1 = torchvision.utils.draw_keypoints(vis_image_1, gt_lmk_dense, colors="blue", radius=1.5)
                vis_image = torchvision.utils.make_grid([vis_image_0.cpu(), vis_image_1.cpu(), images[0].cpu()], nrow=3, padding=0)[None]
                vis_images.append(vis_image)
            else:
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
                vis_image_1 = torchvision.utils.draw_keypoints(vis_image_1, gt_lmk_68, colors="red", radius=1.5)
                vis_image_1 = torchvision.utils.draw_keypoints(vis_image_1, gt_lmk_dense, colors="blue", radius=1.5)
                vis_image = torchvision.utils.make_grid([vis_image_0.cpu(), vis_image_1.cpu(), images[0].cpu()], nrow=3, padding=0)[None]
                vis_images.append(vis_image)
        vis_images = torch.cat(vis_images, dim=0).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(
            os.path.join(output_path, 'tracked.mp4'), vis_images, fps=fps
        )


def run_smoothing(lightning_result, output_path):
    from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
    def smooth_params(data, alpha=0.5):
        smoothed_data = [data[0]]  # Initialize the smoothed data with the first value of the input data
        for i in range(1, len(data)):
            smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[i-1]
            smoothed_data.append(smoothed_value)
        return smoothed_data
    def kalman_smooth_params(data, ):
        from pykalman import KalmanFilter
        kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=data.shape[-1])
        smoothed_data = kf.em(data).smooth(data)[0]
        return smoothed_data
    smoothed_results = {}
    shapecode, expression, pose, rotates, translates = [], [], [], [], []
    frames = list(lightning_result.keys())
    frames = sorted(frames, key=lambda x: int(x.split('_')[-1]))
    for frame_name in frames:
        smoothed_results[frame_name] = deepcopy(lightning_result[frame_name])
        transform_matrix = smoothed_results[frame_name]['transform_matrix']
        rotates.append(matrix_to_rotation_6d(torch.tensor(transform_matrix[:3, :3])).numpy())
        translates.append(transform_matrix[:3, 3])
        shapecode.append(smoothed_results[frame_name]['mica_shape'])
        # pose.append(smoothed_results[frame_name]['emoca_pose'])
        # expression.append(smoothed_results[frame_name]['emoca_expression'])
    # pose = smooth_params(np.stack(pose), alpha=0.95)
    # expression = smooth_params(np.stack(expression), alpha=0.95)
    shapecode = np.stack(shapecode).mean(axis=0)
    if len(rotates) < 2000:
        print('Running kalman smoothing...')
        rotates = kalman_smooth_params(np.stack(rotates))
        translates = kalman_smooth_params(np.stack(translates))
    else:
        print('Run smoothing...')
        rotates = smooth_params(np.stack(rotates), alpha=0.8)
        translates = smooth_params(np.stack(translates), alpha=0.8)
    # rotates = kalman_smooth_params(np.stack(rotates))
    # translates = kalman_smooth_params(np.stack(translates))
    for fidx, frame_name in enumerate(frames):
        # smoothed_results[frame_name]['emoca_pose'] = pose[fidx]
        # smoothed_results[frame_name]['emoca_expression'] = expression[fidx]
        rotation = rotation_6d_to_matrix(torch.tensor(rotates[fidx])).numpy()
        affine_matrix = np.concatenate([rotation, translates[fidx][:, None]], axis=-1)
        smoothed_results[frame_name]['transform_matrix'] = affine_matrix
        smoothed_results[frame_name]['mica_shape'] = shapecode
    with open(os.path.join(output_path, 'smoothed.pkl'), 'wb') as f:
        pickle.dump(smoothed_results, f)
    return smoothed_results


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
    parser.add_argument('--video_path', '-v', required=True, type=str)
    parser.add_argument('--outdir_path', '-d', default='', type=str)
    # parser.add_argument('--split_id', '-s', default=0, type=int)
    parser.add_argument('--synthesis', action='store_true')
    parser.add_argument('--no_vis', action='store_true')
    args = parser.parse_args()
    tracker = Tracker(focal_length=12.0, device='cuda')
    if not os.path.isdir(args.video_path):
        tracker.track_video(args.video_path, dir_path=args.outdir_path, synthesis=args.synthesis, no_vis=args.no_vis)
    else:
        all_videos = list_all_files(args.video_path)
        all_videos = [v for v in all_videos if v.endswith('.mp4')]
        all_videos = sorted(all_videos)
        # all_videos = [v for i, v in enumerate(all_videos) if i % 2 == args.split_id]
        for vidx, video_path in enumerate(all_videos):
            print('Processing {}/{}......'.format(vidx+1, len(all_videos)))
            print(video_path)
            tracker.track_video(video_path, dir_path=args.outdir_path, synthesis=args.synthesis, no_vis=args.no_vis)
