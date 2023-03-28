from pathlib import Path

import torch
import torch.utils.data

from pycocotools.coco import COCO
from PIL import Image
import cv2

from util.poly_ops import resort_corners
from detectron2.data import transforms as T
from torch.utils.data import Dataset
import numpy as np
import os
from copy import deepcopy

from detectron2.data.detection_utils import annotations_to_instances, transform_instance_annotations
from detectron2.structures import BoxMode


class MultiPoly(Dataset):
    def __init__(self, img_folder, ann_file, transforms, semantic_classes):
        super(MultiPoly, self).__init__()

        self.root = img_folder
        self._transforms = transforms
        self.semantic_classes = semantic_classes
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.prepare = ConvertToCocoDict(self.root, self._transforms)

    def get_image(self, path):
        return Image.open(os.path.join(self.root, path))
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: COCO format dict
        """
        coco = self.coco
        img_id = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        ### Note: here is a hack which assumes door/window have category_id 16, 17 in structured3D
        if self.semantic_classes == -1:
            target = [t for t in target if t['category_id'] not in [16, 17]]

        path = coco.loadImgs(img_id)[0]['file_name']

        record = self.prepare(img_id, path, target)

        return record


class ConvertToCocoDict(object):
    def __init__(self, root, augmentations):
        self.root = root
        self.augmentations = augmentations

    def __call__(self, img_id, path, target):

        file_name = os.path.join(self.root, path)

        img = np.array(Image.open(file_name))
        w, h = img.shape

        record = {}
        record["file_name"] = file_name
        record["height"] = h
        record["width"] = w
        record['image_id'] = img_id
        
        for obj in target: obj["bbox_mode"] = BoxMode.XYWH_ABS

        record['annotations'] = target


        if self.augmentations is None:
            record['image'] = (1/255) * torch.as_tensor(np.ascontiguousarray(np.expand_dims(img, 0)))
            record['instances'] = annotations_to_instances(target, (h, w), mask_format="polygon")
        else:
            aug_input = T.AugInput(img)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            record['image'] = (1/255) * torch.as_tensor(np.array(np.expand_dims(image, 0)))
            
            annos = [
                transform_instance_annotations(
                    obj, transforms, image.shape[:2]
                    )
                    for obj in record.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                    ]
            # resort corners after augmentation: so that all corners start from upper-left counterclockwise
            for anno in annos:
                anno['segmentation'][0] = resort_corners(anno['segmentation'][0])

            record['instances'] = annotations_to_instances(annos, (h, w), mask_format="polygon")
            
        return record

def make_poly_transforms(image_set):

    if image_set == 'train':
        return T.AugmentationList([
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomRotation([0.0, 90.0, 180.0, 270.0], expand=False, center=None, sample_style="choice")
            ]) 
        
    if image_set == 'val' or image_set == 'test':
        return None

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.dataset_root)
    assert root.exists(), f'provided data path {root} does not exist'

    PATHS = {
        "train": (root / "train", root / "annotations" / 'train.json'),
        "val": (root / "val", root / "annotations" / 'val.json'),
        "test": (root / "test", root / "annotations" / 'test.json')
    }

    img_folder, ann_file = PATHS[image_set]
    
    dataset = MultiPoly(img_folder, ann_file, transforms=make_poly_transforms(image_set), semantic_classes=args.semantic_classes)
    
    return dataset
