import argparse
import json
import numpy as np
import cv2
import os
import sys
from scenecad_utils import transform
from tqdm import tqdm

from scenecad_utils import generate_density, normalize_annotations, generate_coco_dict

sys.path.append('../.')
from common_utils import read_scene_pc, export_density


def config():
    a = argparse.ArgumentParser(description='Generate coco format data for SceneCAD')
    a.add_argument('--data_root', default='ScanNet', type=str, help='path to raw scannet folder')
    a.add_argument('--scannet_floor_path', default='2Dfloor_planes', type=str, help='path to 2D floor planes folder')
    a.add_argument('--scans_transform_path', default='scans_transform', type=str, help='path to scans transform matrix (converted to axisAlignment)')
    a.add_argument('--output', default='coco_scenecad', type=str, help='path to output floor density folder')
    a.add_argument('--train_list', default='scenecad/meta_data/scannetv2_train.txt', type=str, help='path to official train scene list')
    a.add_argument('--val_list', default='scenecad/meta_data/scannetv2_val.txt', type=str, help='path to official val scene list')
    a.add_argument('--normal_map', default=True, type=str, help='if return normal map')
    a.add_argument('--viz', default=True, type=str, help='visualize for sanity check')
    args = a.parse_args()
    return args

def main(args):
    SCANNET_RAW_PATH = args.data_root
    SCANNET_FLOOR_PATH = args.scannet_floor_path
    SCANS_TRANSFORM_PATH = args.scans_transform_path
    FLOOR_OUT_PATH = args.output

    ANNO_OUT_PATH = os.path.join(FLOOR_OUT_PATH, 'annotations')

    ### prepare
    if not os.path.exists(FLOOR_OUT_PATH):
        os.mkdir(FLOOR_OUT_PATH)

    train_img_folder = os.path.join(FLOOR_OUT_PATH, 'train')
    val_img_folder = os.path.join(FLOOR_OUT_PATH, 'val')
    for img_folder in [train_img_folder, val_img_folder]:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

    if not os.path.exists(ANNO_OUT_PATH):
        os.mkdir(ANNO_OUT_PATH)

    return_normal = args.normal_map
    if return_normal:
        NORMAL_OUT_PATH = os.path.join(FLOOR_OUT_PATH, 'normal')
        if not os.path.exists(NORMAL_OUT_PATH):
            os.mkdir(NORMAL_OUT_PATH)

    viz = args.viz
    if viz:
        VIS_OUT_PATH = os.path.join(FLOOR_OUT_PATH, 'vis')
        if not os.path.exists(VIS_OUT_PATH):
            os.mkdir(VIS_OUT_PATH)


    scene_files = os.listdir(SCANNET_FLOOR_PATH)
    scenes = np.unique(np.array([file.split('.')[0] for file in scene_files]))

    train_list = np.loadtxt(args.train_list,dtype='str').tolist()
    val_list = np.loadtxt(args.val_list,dtype='str').tolist()

    instance_id = 0

    coco_train_json_path = os.path.join(ANNO_OUT_PATH, 'train.json')
    coco_val_json_path = os.path.join(ANNO_OUT_PATH, 'val.json')

    coco_train_dict = {}
    coco_train_dict["images"] = []
    coco_train_dict["annotations"] = []
    coco_train_dict["categories"] = []

    coco_val_dict = {}
    coco_val_dict["images"] = []
    coco_val_dict["annotations"] = []
    coco_val_dict["categories"] = []

    type_dict = {"supercategory": "room", "id": 0, "name": "room"}
    coco_train_dict["categories"].append(type_dict)
    coco_val_dict["categories"].append(type_dict)

    ### begin processing
    for scene in tqdm(scenes):
        
        # load pre-generated point cloud 
        ply_path = os.path.join(SCANNET_RAW_PATH, 'scans', scene, '%s_vh_clean_2.ply' % (scene))

        points = read_scene_pc(ply_path)
        xyz = points[:, :3]
        xyz = transform(scene, xyz, SCANS_TRANSFORM_PATH)

        ### project point cloud to density map
        density, normalization_dict, normal = generate_density(xyz, normal=True)

        ### rescale raw annotations

        polys, heat_annot = normalize_annotations(scene, SCANNET_FLOOR_PATH, normalization_dict)

        ### prepare coco dict
        scene_id = scene.split('scene')[-1].split('_')
        img_id = int(scene_id[0]) * 100 + int(scene_id[1])
        scan_id = f"{img_id:06d}"
        img_dict = {}
        img_dict["file_name"] = scan_id + '.png'
        img_dict["id"] = img_id
        img_dict["width"] = 256
        img_dict["height"] = 256


        ### viz
        if viz:
            density_uint8 = (density * 255).astype(np.uint8)
            density_img_vis = np.repeat(np.expand_dims(density_uint8 ,2), 3, axis=2)
            for i, corner in enumerate(polys):
                if i == len(polys)-1:
                    cv2.line(density_img_vis, (round(corner[0]), round(corner[1])), (round(polys[0][0]), round(polys[0][1])), (0, 252, 252), 1)
                else:
                    cv2.line(density_img_vis, (round(corner[0]), round(corner[1])), (round(polys[i+1][0]), round(polys[i+1][1])), (0, 252, 252), 1)
                cv2.circle(density_img_vis, (round(corner[0]), round(corner[1])), 3, (0,0,255), -1)
                cv2.putText(density_img_vis, str(i), (round(corner[0]), round(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(VIS_OUT_PATH, scan_id+'.png'), density_img_vis)

        if normal is not None:    
            cv2.imwrite(os.path.join(NORMAL_OUT_PATH, scan_id+'.png'), normal)

        polygons_list = generate_coco_dict([polys], instance_id, img_id)

        assert len(polygons_list) == 1
        instance_id += len(polygons_list)

        if scene in train_list:
            coco_train_dict["images"].append(img_dict)
            coco_train_dict["annotations"] += polygons_list
            export_density(density, train_img_folder, scan_id)
        elif scene in val_list:
            coco_val_dict["images"].append(img_dict)
            coco_val_dict["annotations"] += polygons_list
            export_density(density, val_img_folder, scan_id)

        print(scene)


    with open(coco_train_json_path, 'w') as f:
        json.dump(coco_train_dict, f)
    with open(coco_val_json_path, 'w') as f:
        json.dump(coco_val_dict, f)

if __name__ == '__main__':

    main(config())