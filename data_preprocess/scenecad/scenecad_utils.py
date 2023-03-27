"""
utils to process SceneCAD layout data

"""

import json
import math
import numpy as np
from shapely.geometry import Polygon
import os
import sys
import open3d as o3d

sys.path.append('../data_preprocess')
from common_utils import is_clockwise, resort_corners

def get_axis_align_matrix(scan_name, scans_transform_path):
    meta_file = scans_transform_path + '/'+os.path.join(scan_name,scan_name+'.txt')
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    return axis_align_matrix


def transform(scan_name,mesh_vertices,scans_transform_path):
    axis_align_matrix = get_axis_align_matrix(scan_name, scans_transform_path)
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]
    return mesh_vertices


def get_floor(scan_name, scannet_planes_path, scans_transform_path, out_path):

    print("Processing scan: ", scan_name)

    with open(scannet_planes_path+'/'+scan_name+'.json','r') as quad_file:
        plane_dict = json.load(quad_file)
    edge_dict = plane_dict['edges']
    vert_dict=plane_dict['verts']
    
    for i in range(0,len(vert_dict)):
       temp = vert_dict[i][1]
       vert_dict[i][1] = - vert_dict[i][2]
       vert_dict[i][2] = temp

    verts = np.array(vert_dict)
    verts = transform(scan_name,verts,scans_transform_path)

    floor_egdes = []
    z_center = verts[:,2].mean()

    ## extract points and egdes on the floor
    for egde in edge_dict:
        # if np.all(verts[egde][:,2] < z_center) & (np.abs(verts[egde][0,2]-verts[egde][1,2]) < 1):
        if np.all(verts[egde][:,2] < z_center):
            floor_egdes.append(egde)

    floor_egdes = np.array(floor_egdes)
    vert_idx = np.unique(floor_egdes)
    vert_idx_dict = dict(zip(np.arange(0, vert_idx.shape[0], 1), vert_idx))
    for k, v in vert_idx_dict.items(): floor_egdes[floor_egdes==v] = k


    floor_points = verts[vert_idx]

    ## find most upper corner (x minimum)
    # upper_corner_idx = np.argmin(floor_points, 0)[0]
    # V2: find most upper corner (x+y minimum)
    # upper_corner_idx = np.argmin(floor_points[:,0] + floor_points[:,1])
    x_y_square_sum = floor_points[:,0]**2 + floor_points[:,1]**2 
    lower_left_points = (floor_points[:,0] < 0) & (floor_points[:,1] < 0)
    x_y_square_sum[np.invert(lower_left_points)] =0
    upper_corner_idx = np.argmax(x_y_square_sum)


    adjacent_edges = floor_egdes[(floor_egdes[:,0]==upper_corner_idx) | (floor_egdes[:,1]==upper_corner_idx)]
    adjacent_vertices = {e for l in adjacent_edges.tolist() for e in l}
    adjacent_vertices.remove(upper_corner_idx)
    if not len(adjacent_vertices) == 2:
        print("Ignored scan: ", scan_name)
        return False 
    adj_v1, adj_v2 = adjacent_vertices


    ## sort points and edges from the starting point
    edge_sorted = []
    points_sorted = []
    edge_sorted.append([0, 1])
    points_sorted.append(floor_points[upper_corner_idx])
    points_sorted.append(floor_points[adj_v1])
    vert_idx_dict_sorted = {0:upper_corner_idx, 1:adj_v1}
    idx = 2

    adj_v_next = adj_v1
    adj_v_before = upper_corner_idx
    
    while not adj_v_next == adj_v2:
        
        adjacent_edges_next = floor_egdes[(floor_egdes[:,0]==adj_v_next) | (floor_egdes[:,1]==adj_v_next)]
        adjacent_vertices = {e for l in adjacent_edges_next.tolist() for e in l}
        adjacent_vertices.remove(adj_v_next)
        adjacent_vertices.remove(adj_v_before)

        if not len(adjacent_vertices) == 1:
            print("Ignored scan: ", scan_name)
            return False

        adj_v_before = adj_v_next
        (adj_v_next,)=adjacent_vertices

        edge_sorted.append([idx-1, idx])
        points_sorted.append(floor_points[adj_v_next])
        vert_idx_dict_sorted[idx] = adj_v_next
        idx+=1
    edge_sorted.append([idx-1, 0])
    edge_sorted = np.array(edge_sorted)
    points_sorted = np.array(points_sorted)

    ## sort points clockwise
    if not is_clockwise(points_sorted[:,:2].tolist()):
        points_sorted[1:] = np.flip(points_sorted[1:], 0)

    toSave = {'floor_verts': points_sorted.tolist(), 'floor_edges':edge_sorted.tolist()}
    with open(out_path+'/'+scan_name+'.json', 'w') as fp:
        json.dump(toSave, fp)

    return True


def get_floor_multiRoom(scan_name, scannet_planes_path, scans_transform_path, out_path):

    print("Processing scan: ", scan_name)

    with open(scannet_planes_path+'/'+scan_name+'.json','r') as quad_file:
        plane_dict = json.load(quad_file)
    edge_dict = plane_dict['edges']
    vert_dict=plane_dict['verts']
    
    for i in range(0,len(vert_dict)):
       temp = vert_dict[i][1]
       vert_dict[i][1] = - vert_dict[i][2]
       vert_dict[i][2] = temp

    verts = np.array(vert_dict)
    verts = transform(scan_name,verts,scans_transform_path)

    floor_egdes = []
    z_center = verts[:,2].mean()

    ## extract points and egdes on the floor
    for egde in edge_dict:
        # if np.all(verts[egde][:,2] < z_center) & (np.abs(verts[egde][0,2]-verts[egde][1,2]) < 1):
        if np.all(verts[egde][:,2] < z_center):
            floor_egdes.append(egde)

    floor_egdes = np.array(floor_egdes)
    vert_idx = np.unique(floor_egdes)
    vert_idx_dict = dict(zip(np.arange(0, vert_idx.shape[0], 1), vert_idx))
    for k, v in vert_idx_dict.items(): floor_egdes[floor_egdes==v] = k


    floor_points = verts[vert_idx]

    pg = {'corners': floor_points[:,:2], 'edges': floor_egdes}


    ## find most upper corner (x minimum)
    # upper_corner_idx = np.argmin(floor_points, 0)[0]
    # V2: find most upper corner (x+y minimum)
    # upper_corner_idx = np.argmin(floor_points[:,0] + floor_points[:,1])
    x_y_square_sum = floor_points[:,0]**2 + floor_points[:,1]**2 
    lower_left_points = (floor_points[:,0] < 0) & (floor_points[:,1] < 0)
    x_y_square_sum[np.invert(lower_left_points)] =0
    upper_corner_idx = np.argmax(x_y_square_sum)


    adjacent_edges = floor_egdes[(floor_egdes[:,0]==upper_corner_idx) | (floor_egdes[:,1]==upper_corner_idx)]
    adjacent_vertices = {e for l in adjacent_edges.tolist() for e in l}
    adjacent_vertices.remove(upper_corner_idx)
    if not len(adjacent_vertices) == 2:
        print("Ignored scan: ", scan_name)
        return False 
    adj_v1, adj_v2 = adjacent_vertices


    ## sort points and edges from the starting point
    edge_sorted = []
    points_sorted = []
    edge_sorted.append([0, 1])
    points_sorted.append(floor_points[upper_corner_idx])
    points_sorted.append(floor_points[adj_v1])
    vert_idx_dict_sorted = {0:upper_corner_idx, 1:adj_v1}
    idx = 2

    adj_v_next = adj_v1
    adj_v_before = upper_corner_idx
    
    while not adj_v_next == adj_v2:
        
        adjacent_edges_next = floor_egdes[(floor_egdes[:,0]==adj_v_next) | (floor_egdes[:,1]==adj_v_next)]
        adjacent_vertices = {e for l in adjacent_edges_next.tolist() for e in l}
        adjacent_vertices.remove(adj_v_next)
        adjacent_vertices.remove(adj_v_before)

        if not len(adjacent_vertices) == 1:
            print("Ignored scan: ", scan_name)
            return False

        adj_v_before = adj_v_next
        (adj_v_next,)=adjacent_vertices

        edge_sorted.append([idx-1, idx])
        points_sorted.append(floor_points[adj_v_next])
        vert_idx_dict_sorted[idx] = adj_v_next
        idx+=1
    edge_sorted.append([idx-1, 0])
    edge_sorted = np.array(edge_sorted)
    points_sorted = np.array(points_sorted)

    ## sort points clockwise
    if not is_clockwise(points_sorted[:,:2].tolist()):
        points_sorted[1:] = np.flip(points_sorted[1:], 0)

    toSave = {'floor_verts': points_sorted.tolist(), 'floor_edges':edge_sorted.tolist()}
    with open(out_path+'/'+scan_name+'.json', 'w') as fp:
        json.dump(toSave, fp)

    return True


def get_floor_corners(scan_name, floorplan_path):
    with open(floorplan_path+'/'+scan_name+'.json','r') as quad_file:
        floor_dict = json.load(quad_file)
    edge_dict = floor_dict['floor_edges']
    vert_dict=floor_dict['floor_verts']
    verts = np.array(vert_dict)

    return verts, edge_dict


def normalize_point(corner, min_x, width, min_y, height):
        img_x = np.clip(int(math.floor((corner[0] - min_x) * 1.0 / width * 256)), 0, 255)
        img_y = np.clip(int(math.floor((corner[1] - min_y) * 1.0 / height * 256)), 0, 255)
        return img_x, img_y

def generate_density(xyz, normal=False):


    # xyz = xyz * 1000.0
    # xyz[:, :2] = np.round(xyz[:, :2] / 10) * 10.
    # xyz[:, 2] = np.round(xyz[:, 2] / 100) * 100.
    # unique_coords, unique_ind = np.unique(xyz[:, 0:3], return_index=True, axis=0)
    # xyz = xyz[unique_ind]

    if normal:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals()

    mins = xyz.min(0, keepdims=True)
    maxs = xyz.max(0, keepdims=True)

    max_range = (maxs - mins)[:, :2].max()
    padding = max_range * 0.05

    mins = (maxs + mins) / 2 - max_range / 2
    mins -= padding
    max_range += padding * 2

    xyz = (xyz - mins) / max_range  # re-scale coords into [0.0, 1.0]

    coordinates = np.round(xyz[:,:2] * 256)
    coordinates = np.minimum(np.maximum(coordinates, 0),
                                255)

    density = np.zeros((256, 256), dtype=np.float32)

    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)

    unique_coordinates = unique_coordinates.astype(np.int32)

    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    density = density / np.max(density)

    normalization_dict = {}
    normalization_dict["min_x"] = mins[0][0]
    normalization_dict["min_y"] = mins[0][1]
    normalization_dict["max_range"] = max_range


    if normal:
        normals = np.array(pcd.normals)
        normals_map = np.zeros((density.shape[0], density.shape[1], 3))

        # count_map = np.ones((density.shape[0], density.shape[1], 3))
        for i, unique_coord in enumerate(unique_coordinates):
            normals_indcs = np.argwhere(np.all(coordinates[::10] == unique_coord, axis=1))[:,0]
            normals_map[unique_coordinates[i, 1], unique_coordinates[i, 0], :] = np.mean(normals[::10][normals_indcs, :], axis=0)
            # if len(normals_indcs) > 0:
            #     count_map[unique_coordinates[i, 1], unique_coordinates[i, 0], :] = len(normals_indcs)

        normals_map = (np.clip(normals_map,0,1) * 255).astype(np.uint8)

    else:
        normals_map = None

    return density, normalization_dict, normals_map

def normalize_annotations(scan_name, SCANNET_FLOOR_PATH, normalization_dict):

    corners, edge_dict = get_floor_corners(scan_name, SCANNET_FLOOR_PATH)

    # corners = corners *1000

    corners_in_img = []
    min_x = normalization_dict["min_x"]
    min_y = normalization_dict["min_y"]
    width = height = normalization_dict["max_range"]
    for corner in corners:
        corner_in_img = normalize_point(corner, min_x, width, min_y, height)
        corners_in_img.append(corner_in_img[:2])
    corners_in_img = np.stack(corners_in_img).astype(np.float64)

    # for heat format annotation
    heat_annot = dict()
    edge_dict = np.array(edge_dict)
    for i, corner in enumerate(corners_in_img):
        adjacent_edge = edge_dict[(edge_dict[:,0]==i) | (edge_dict[:,1]==i)]
        adjacent_idx = {e for l in adjacent_edge.tolist() for e in l}
        adjacent_idx.remove(i)

        heat_annot[tuple(corners_in_img[i])] = [tuple(corners_in_img[j]) for j in adjacent_idx]


    return corners_in_img, heat_annot


def generate_coco_dict(polygons, curr_instance_id, curr_img_id):


    coco_annotation_dict_list = []

    for poly_ind, polygon in enumerate(polygons):

        poly_shapely = Polygon(polygon)
        area = poly_shapely.area
        
        rectangle_shapely = poly_shapely.envelope


        coco_seg_poly = []
        poly_sorted = resort_corners(polygon)

        for p in poly_sorted:
            coco_seg_poly += list(p)

        # Slightly wider bounding box
        bound_pad = 2
        bb_x, bb_y = rectangle_shapely.exterior.xy
        bb_x = np.unique(bb_x)
        bb_y = np.unique(bb_y)
        bb_x_min = np.maximum(np.min(bb_x) - bound_pad, 0)
        bb_y_min = np.maximum(np.min(bb_y) - bound_pad, 0)

        bb_x_max = np.minimum(np.max(bb_x) + bound_pad, 256 - 1)
        bb_y_max = np.minimum(np.max(bb_y) + bound_pad, 256 - 1)

        bb_width = (bb_x_max - bb_x_min)
        bb_height = (bb_y_max - bb_y_min)

        coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]

        coco_annotation_dict = {
                "segmentation": [coco_seg_poly],
                "area": area,
                "iscrowd": 0,
                "image_id": curr_img_id,
                "bbox": coco_bb,
                "category_id": 0,
                "id": curr_instance_id}
        
        coco_annotation_dict_list.append(coco_annotation_dict)
        curr_instance_id += 1


    return coco_annotation_dict_list
