"""
Script to extract floorplan from 3D layout

- The starting vertex is the point that has the mimimum x
- All vertices and edges are sorted clockwise
- Only extract single closed layout (i.e. no inner room)


"""

import argparse
from traceback import print_tb
import numpy as np
import os

from scenecad_utils import get_floor

    

def config():
    a = argparse.ArgumentParser(description='extract floorplan')
    a.add_argument('--scannet_planes_path', default='scannet_planes', type=str, help='path to scannet planes folder')
    a.add_argument('--scans_transform_path', default='scans_transform', type=str, help='path to scans transform matrix (converted to axisAlignment)')
    a.add_argument('--out_path', default='2Dfloor_planes', type=str, help='path to output floor folder')
    
    args = a.parse_args()
    return args

def main(args):
    SCANNET_PLANES_PATH = args.scannet_planes_path
    SCANS_TRANSFORM_PATH = args.scans_transform_path
    FLOOR_OUT_PATH = args.out_path

    if not os.path.exists(FLOOR_OUT_PATH):
        os.mkdir(FLOOR_OUT_PATH)

    scene_files = os.listdir(SCANNET_PLANES_PATH)
    scenes = np.unique(np.array([file.split('.')[0] for file in scene_files]))

    ignored_scenes = []

    for scene in scenes:
        if not get_floor(scene, SCANNET_PLANES_PATH, SCANS_TRANSFORM_PATH, FLOOR_OUT_PATH):
            ignored_scenes.append(scene)

    print(ignored_scenes)


if __name__ == '__main__':
    main(config())