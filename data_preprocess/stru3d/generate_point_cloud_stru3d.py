import argparse
import os
from tqdm import tqdm
from PointCloudReaderPanorama import PointCloudReaderPanorama


def config():
    a = argparse.ArgumentParser(description='Generate point cloud for Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    args = a.parse_args()
    return args

def main(args):
    print("Creating point cloud from perspective views...")
    data_root = args.data_root
    data_parts = os.listdir(data_root)

    for part in tqdm(data_parts):
        scenes = os.listdir(os.path.join(data_root, part, 'Structured3D'))
        for scene in tqdm(scenes):
            scene_path = os.path.join(data_root, part, 'Structured3D', scene)
            reader = PointCloudReaderPanorama(scene_path, random_level=0, generate_color=True, generate_normal=False)
            save_path = os.path.join(data_root, part, 'Structured3D', scene, 'point_cloud.ply')
            reader.export_ply(save_path)
            

if __name__ == "__main__":

    main(config())