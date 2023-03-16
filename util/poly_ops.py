"""
Utilities for polygon manipulation.
"""
import torch
import numpy as np


def is_clockwise(points):
    """Check whether a sequence of points is clockwise ordered
    """
    # points is a list of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0

def resort_corners(corners):
    """Resort a sequence of corners so that the first corner starts
       from upper-left and counterclockwise ordered in image
    """
    corners = corners.reshape(-1, 2)
    x_y_square_sum = corners[:,0]**2 + corners[:,1]**2 
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    ## sort points clockwise (counterclockwise in image)
    if not is_clockwise(corners_sorted[:,:2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners_sorted.reshape(-1)


def get_all_order_corners(corners):
    """Get all possible permutation of a polygon
    """
    length = int(len(corners) / 2)
    all_corners = torch.stack([corners.roll(i*2) for i in range(length)])
    return all_corners


def pad_gt_polys(gt_instances, num_queries_per_poly, device):
    """Pad the ground truth polygons so that they have a uniform length
    """

    room_targets = []
    # padding ground truth on-fly
    for gt_inst in gt_instances:
        room_dict = {}
        room_corners = []
        corner_labels = []
        corner_lengths = []

        for i, poly in enumerate(gt_inst.gt_masks.polygons):
            corners = torch.from_numpy(poly[0]).to(device)
            corners = torch.clip(corners, 0, 255) / 255
            corner_lengths.append(len(corners))

            corners_pad = torch.zeros(num_queries_per_poly*2, device=device)
            corners_pad[:len(corners)] = corners

            labels = torch.ones(int(len(corners)/2), dtype=torch.int64).to(device)
            labels_pad = torch.zeros(num_queries_per_poly, device=device)
            labels_pad[:len(labels)] = labels
            room_corners.append(corners_pad)
            corner_labels.append(labels_pad)

        room_dict = {
            'coords': torch.stack(room_corners),
            'labels': torch.stack(corner_labels),
            'lengths': torch.tensor(corner_lengths, device=device),
            'room_labels': gt_inst.gt_classes
        }
        room_targets.append(room_dict)


    return room_targets


