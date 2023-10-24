#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import torch
import pickle
import random
sys.path.append('./')
import torchvision
from tqdm import tqdm

from engines.FLAME import FLAMEDense
from utils.lmdb_utils import LMDBEngine
from utils.renderer_utils import Mesh_Renderer

def visualize_video(tracked_path, output_path):
    lmdb_engine = LMDBEngine(os.path.join(tracked_path, 'img_lmdb'), write=False)
    with open(os.path.join(tracked_path, 'lightning.pkl'), 'rb') as f:
        params = pickle.load(f)
    frame_keys = sorted(params.keys(), key=lambda x: int(x.split('_')[-1]))
    flame_model = FLAMEDense(n_shape=100, n_exp=50).cuda()
    mesh_renderer = Mesh_Renderer(
        256, faces=flame_model.get_faces().cpu().numpy(), device='cuda'
    )
    images = []
    for frame_key in tqdm(frame_keys[:50]):
        ori_image = lmdb_engine[frame_key]
        ori_image = torchvision.transforms.functional.resize(ori_image, 256, antialias=True)
        this_params = params[frame_key]
        for pkey in this_params.keys():
            this_params[pkey] = torch.tensor(this_params[pkey]).float().cuda()[None]
        vertices, _, _ = flame_model(
            shape_params=this_params['emoca_shape'], 
            expression_params=this_params['emoca_expression'], 
            pose_params=this_params['emoca_pose']
        )
        vertices *= 5.0
        mesh_images, _ = mesh_renderer(
            vertices, transform_matrix=this_params['transform_matrix'], 
            focal_length=8.0, principal_point=torch.tensor([0.0, 0.0])
        )
        mesh_images = mesh_images.cpu()
        image = torchvision.utils.make_grid([ori_image, mesh_images[0]], nrow=2)
        images.append(image)
    lmdb_engine.close()
    images = torch.stack(images, dim=0)
    images = images.to(torch.uint8).permute(0, 2, 3, 1).cpu()
    torchvision.io.write_video(output_path, images, fps=25.0, )


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
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.video_path, 'img_lmdb')):
        video_dirs = os.listdir(args.video_path)
        video_path = os.path.join(args.video_path, random.choice(video_dirs))
    else:
        video_path = args.video_path
    visualize_video(video_path, '{}.mp4'.format(os.path.basename(video_path)))
