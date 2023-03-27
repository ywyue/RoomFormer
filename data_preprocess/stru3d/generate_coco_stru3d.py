import argparse
import json
import os
import sys
from tqdm import tqdm
from stru3d_utils import generate_density, normalize_annotations, parse_floor_plan_polys, generate_coco_dict

sys.path.append('../.')
from common_utils import read_scene_pc, export_density


### Note: Some scenes have missing/wrong annotations. These are the indices that you should additionally exclude 
### to be consistent with MonteFloor and HEAT:
invalid_scenes_ids = [76, 183, 335, 491, 663, 681, 703, 728, 865, 936, 985, 986, 1009, 1104, 1155, 1221, 1282, 
                     1365, 1378, 1635, 1745, 1772, 1774, 1816, 1866, 2037, 2076, 2274, 2334, 2357, 2580, 2665, 
                     2706, 2713, 2771, 2868, 3156, 3192, 3198, 3261, 3271, 3276, 3296, 3342, 3387, 3398, 3466, 3496]

type2id = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

def config():
    a = argparse.ArgumentParser(description='Generate coco format data for Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--output', default='coco_stru3d', type=str, help='path to output folder')
    
    args = a.parse_args()
    return args

def main(args):
    data_root = args.data_root
    data_parts = os.listdir(data_root)

    ### prepare
    outFolder = args.output
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    annotation_outFolder = os.path.join(outFolder, 'annotations')
    if not os.path.exists(annotation_outFolder):
        os.mkdir(annotation_outFolder)

    train_img_folder = os.path.join(outFolder, 'train')
    val_img_folder = os.path.join(outFolder, 'val')
    test_img_folder = os.path.join(outFolder, 'test')

    for img_folder in [train_img_folder, val_img_folder, test_img_folder]:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

    coco_train_json_path = os.path.join(annotation_outFolder, 'train.json')
    coco_val_json_path = os.path.join(annotation_outFolder, 'val.json')
    coco_test_json_path = os.path.join(annotation_outFolder, 'test.json')

    coco_train_dict = {"images":[],"annotations":[],"categories":[]}
    coco_val_dict = {"images":[],"annotations":[],"categories":[]}
    coco_test_dict = {"images":[],"annotations":[],"categories":[]}

    for key, value in type2id.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        coco_train_dict["categories"].append(type_dict)
        coco_val_dict["categories"].append(type_dict)
        coco_test_dict["categories"].append(type_dict)

    ### begin processing
    instance_id = 0
    for part in tqdm(data_parts):
        scenes = os.listdir(os.path.join(data_root, part, 'Structured3D'))
        for scene in tqdm(scenes):
            scene_path = os.path.join(data_root, part, 'Structured3D', scene)
            scene_id = scene.split('_')[-1]

            if int(scene_id) in invalid_scenes_ids:
                print('skip {}'.format(scene))
                continue
            
            # load pre-generated point cloud 
            ply_path = os.path.join(scene_path, 'point_cloud.ply')
            points = read_scene_pc(ply_path)
            xyz = points[:, :3]

            ### project point cloud to density map
            density, normalization_dict = generate_density(xyz, width=256, height=256)
            
            ### rescale raw annotations
            normalized_annos = normalize_annotations(scene_path, normalization_dict)

            ### prepare coco dict
            img_id = int(scene_id)
            img_dict = {}
            img_dict["file_name"] = scene_id + '.png'
            img_dict["id"] = img_id
            img_dict["width"] = 256
            img_dict["height"] = 256

            ### parse annotations
            polys = parse_floor_plan_polys(normalized_annos)
            polygons_list = generate_coco_dict(normalized_annos, polys, instance_id, img_id, ignore_types=['outwall'])

            instance_id += len(polygons_list)

            ### train
            if int(scene_id) < 3000:
                coco_train_dict["images"].append(img_dict)
                coco_train_dict["annotations"] += polygons_list
                export_density(density, train_img_folder, scene_id)

            ### val
            elif int(scene_id) >= 3000 and int(scene_id) < 3250:
                coco_val_dict["images"].append(img_dict)
                coco_val_dict["annotations"] += polygons_list
                export_density(density, val_img_folder, scene_id)

            ### test
            else:
                coco_test_dict["images"].append(img_dict)
                coco_test_dict["annotations"] += polygons_list
                export_density(density, test_img_folder, scene_id)
            
            
            print(scene_id)


    with open(coco_train_json_path, 'w') as f:
        json.dump(coco_train_dict, f)
    with open(coco_val_json_path, 'w') as f:
        json.dump(coco_val_dict, f)
    with open(coco_test_json_path, 'w') as f:
        json.dump(coco_test_dict, f)


if __name__ == "__main__":

    main(config())