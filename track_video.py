#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
sys.path.append('./')

from utils.lmdb_utils import LMDBEngine
from engines.core_engine import TrackEngine

class Tracker:
    def __init__(self, focal_length=8.0, device='cuda'):
        self.tracker = TrackEngine(focal_length=focal_length, device=device)

    def track_video(self, video_path, dir_path=None):
        # build name
        data_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(f'outputs/{dir_path}', data_name) if dir_path else f'outputs/{data_name}'
        if os.path.exists(os.path.join(output_path, 'lightning.pkl')):
            print('Done.')
            return
        # build video data
        print('Processing video data...')
        self.tracker.build_video(video_path, output_path, matting=True, background=0.0)
        lmdb_engine = LMDBEngine(os.path.join(output_path, 'img_lmdb'), write=False)
        print('Done.')
        # track emoca
        emoca_results = self.run_emoca(lmdb_engine, output_path)
        # track lightning
        lightning_result = self.run_lightning(emoca_results, lmdb_engine, output_path)
        lmdb_engine.close()
        # smoothing_result = run_smoothing(lightning_result, output_path)

    def run_emoca(self, lmdb_engine, output_path,):
        print('Track with emoca...')
        if not os.path.exists(os.path.join(output_path, 'emoca.pkl')):
            emoca_results = {}
            for key in tqdm(lmdb_engine.keys()):
                emoca_results[key] = self.tracker.track_emoca(lmdb_engine[key])
            with open(os.path.join(output_path, 'emoca.pkl'), 'wb') as f:
                pickle.dump(emoca_results, f)
        else:
            with open(os.path.join(output_path, 'emoca.pkl'), 'rb') as f:
                emoca_results = pickle.load(f)
        print('Done.')
        return emoca_results

    def run_lightning(self, emoca_results, lmdb_engine, output_path,):
        print('Track lightning...')
        if not os.path.exists(os.path.join(output_path, 'lightning.pkl')):
            lightning_result = self.tracker.track_lightning(emoca_results, lmdb_engine, output_path)
            with open(os.path.join(output_path, 'lightning.pkl'), 'wb') as f:
                pickle.dump(lightning_result, f)
        else:
            with open(os.path.join(output_path, 'lightning.pkl'), 'rb') as f:
                lightning_result = pickle.load(f)
        print('Done.')
        return lightning_result


def run_smoothing(lightning_result, output_path):
    print('Run smoothing...')
    from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
    def smooth_params(data, alpha=0.5):
        smoothed_data = [data[0]]  # Initialize the smoothed data with the first value of the input data
        for i in range(1, len(data)):
            smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[i-1]
            smoothed_data.append(smoothed_value)
        return smoothed_data
    smoothed_results = {}
    expression, pose, rotates, translates = [], [], [], []
    max_frame_id = max([int(k.split('_')[1]) for k in lightning_result.keys()])
    for fidx in range(max_frame_id+1):
        frame_name = f'img_{fidx}'
        smoothed_results[frame_name] = deepcopy(lightning_result[frame_name])
        transform_matrix = smoothed_results[frame_name]['transform_matrix']
        rotates.append(matrix_to_rotation_6d(torch.tensor(transform_matrix[:3, :3])).numpy())
        translates.append(transform_matrix[:3, 3])
        pose.append(smoothed_results[frame_name]['emoca_pose'])
        expression.append(smoothed_results[frame_name]['emoca_expression'])
    pose = smooth_params(np.stack(pose), alpha=0.9)
    expression = smooth_params(np.stack(expression), alpha=0.9)
    rotates = smooth_params(np.stack(rotates), alpha=0.5)
    translates = smooth_params(np.stack(translates), alpha=0.5)

    for fidx in range(max_frame_id+1):
        frame_name = f'img_{fidx}'
        smoothed_results[frame_name]['emoca_pose'] = pose[fidx]
        smoothed_results[frame_name]['emoca_expression'] = expression[fidx]
        rotation = rotation_6d_to_matrix(torch.tensor(rotates[fidx])).numpy()
        affine_matrix = np.concatenate([rotation, translates[fidx][:, None]], axis=-1)
        smoothed_results[frame_name]['transform_matrix'] = affine_matrix
    print('Done')
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v', required=True, type=str)
    parser.add_argument('--outdir_path', '-d', default='', type=str)
    parser.add_argument('--split_id', '-s', default=0, type=int)
    args = parser.parse_args()
    tracker = Tracker(focal_length=8.0, device='cuda')
    if not os.path.isdir(args.video_path):
        tracker.track_video(args.video_path, dir_path=args.outdir_path)
    else:
        all_videos = list_all_files(args.video_path)
        all_videos = [v for v in all_videos if v.endswith('.mp4')]
        all_videos = sorted(all_videos)
        # all_videos = [v for i, v in enumerate(all_videos) if i % 2 == args.split_id]
        for vidx, video_path in enumerate(all_videos):
            if '002468' in video_path:
                continue
            print('Processing {}/{}......'.format(vidx+1, len(all_videos)))
            print(video_path)
            tracker.track_video(video_path, dir_path=args.outdir_path)
