import numpy as np
import os
import cv2
from plyfile import PlyData, PlyElement

def read_scene_pc(file_path):
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        dtype = plydata['vertex'].data.dtype
    print('dtype of file{}: {}'.format(file_path, dtype))

    points_data = np.array(plydata['vertex'].data.tolist())

    return points_data


def is_clockwise(points):
    # points is a list of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0

def resort_corners(corners):
    # re-find the starting point and sort corners clockwisely
    x_y_square_sum = corners[:,0]**2 + corners[:,1]**2 
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    ## sort points clockwise
    if not is_clockwise(corners_sorted[:,:2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners
    

def export_density(density_map, out_folder, scene_id):
    density_path = os.path.join(out_folder, scene_id+'.png')
    density_uint8 = (density_map * 255).astype(np.uint8)
    cv2.imwrite(density_path, density_uint8)