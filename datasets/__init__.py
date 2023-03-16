import torch.utils.data

from .poly_data import build as build_poly


def build_dataset(image_set, args):
    if args.dataset_name == 'stru3d' or args.dataset_name == 'scenecad':
        return build_poly(image_set, args)
    raise ValueError(f'dataset {args.dataset_name} not supported')
