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
    def __init__(self, img_folder, ann_file, transforms, semantic_classes,
                 use_depth_input=False, depth_root=None, input_channels=1):
        super(MultiPoly, self).__init__()

        self.root = img_folder
        self._transforms = transforms
        self.semantic_classes = semantic_classes
        self.use_depth_input = use_depth_input
        self.depth_root = depth_root
        self.input_channels = input_channels
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.prepare = ConvertToCocoDict(
            self.root,
            self._transforms,
            use_depth_input=self.use_depth_input,
            depth_root=self.depth_root,
            input_channels=self.input_channels,
        )

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
    def __init__(self, root, augmentations, use_depth_input=False,
                 depth_root=None, input_channels=1):
        self.root = root
        self.augmentations = augmentations
        self.use_depth_input = use_depth_input
        self.depth_root = Path(depth_root) if depth_root is not None else None
        self.input_channels = input_channels

    def _read_image(self, file_name):
        img = np.array(Image.open(file_name))
        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return img

    def _resolve_depth_path(self, path):
        if self.depth_root is not None:
            depth_path = self.depth_root / path
            if depth_path.exists():
                return depth_path
            raise FileNotFoundError(
                f"Depth input is enabled, but {depth_path} does not exist."
            )

        image_path = Path(self.root) / path
        split_dir = Path(self.root)
        candidates = [
            split_dir.parent / "depth" / path,
            split_dir.parent / "depths" / path,
            image_path.with_name(image_path.stem + "_depth" + image_path.suffix),
            image_path.with_name(image_path.stem + ".npy"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Depth input is enabled, but no depth map was found for "
            f"{path}. Pass --depth_root or place depth maps in a sibling "
            "depth/ directory."
        )

    def _read_depth(self, path, image_shape):
        depth_path = self._resolve_depth_path(path)
        if depth_path.suffix == ".npy":
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path))
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.shape[:2] != image_shape[:2]:
            depth = cv2.resize(depth, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32)
        depth_min, depth_max = np.nanmin(depth), np.nanmax(depth)
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)
        return depth[:, :, None]

    def _build_input(self, path, file_name):
        img = self._read_image(file_name)
        if self.use_depth_input:
            if self.input_channels != 4:
                raise ValueError("--use_depth_input expects --input_channels 4")
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            else:
                img = img[:, :, :3]
            depth = self._read_depth(path, img.shape)
            img = np.concatenate([img.astype(np.float32) / 255.0, depth], axis=2)
        else:
            if self.input_channels == 1:
                img = img[:, :, :1]
            elif self.input_channels == 3:
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                else:
                    img = img[:, :, :3]
            else:
                raise ValueError("--input_channels must be 1 or 3 unless --use_depth_input is set")
            img = img.astype(np.float32) / 255.0
        return img.astype(np.float32)

    def _to_chw_tensor(self, image):
        if image.ndim == 2:
            image = image[:, :, None]
        return torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    def __call__(self, img_id, path, target):

        file_name = os.path.join(self.root, path)

        img = self._build_input(path, file_name)
        h, w = img.shape[:2]

        record = {}
        record["file_name"] = file_name
        record["height"] = h
        record["width"] = w
        record['image_id'] = img_id
        
        for obj in target: obj["bbox_mode"] = BoxMode.XYWH_ABS

        record['annotations'] = target


        if self.augmentations is None:
            record['image'] = self._to_chw_tensor(img)
            record['instances'] = annotations_to_instances(target, (h, w), mask_format="polygon")
        else:
            aug_input = T.AugInput(img)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            record['image'] = self._to_chw_tensor(image)
            
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

            record['instances'] = annotations_to_instances(annos, image.shape[:2], mask_format="polygon")
            
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
    
    depth_root = None
    if getattr(args, "depth_root", None):
        depth_root = Path(args.depth_root) / image_set
    dataset = MultiPoly(
        img_folder,
        ann_file,
        transforms=make_poly_transforms(image_set),
        semantic_classes=args.semantic_classes,
        use_depth_input=getattr(args, "use_depth_input", False),
        depth_root=depth_root,
        input_channels=getattr(args, "input_channels", 1),
    )
    
    return dataset
