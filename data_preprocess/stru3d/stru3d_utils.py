"""
This code is an adaptation that uses Structured 3D for the code base.

Reference: https://github.com/bertjiazheng/Structured3D
"""

import numpy as np
from shapely.geometry import Polygon
import os
import json
import sys

sys.path.append('../data_preprocess')
from common_utils import resort_corners

type2id = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}



def generate_density(point_cloud, width=256, height=256):

    ps = point_cloud * -1
    ps[:,0] *= -1
    ps[:,1] *= -1

    image_res = np.array((width, height))

    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    max_m_min = max_coords - min_coords

    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min

    normalization_dict = {}
    normalization_dict["min_coords"] = min_coords
    normalization_dict["max_coords"] = max_coords
    normalization_dict["image_res"] = image_res


    # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
    coordinates = \
        np.round(
            (ps[:, :2] - min_coords[None, :2]) / (max_coords[None,:2] - min_coords[None, :2]) * image_res[None])
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                                image_res - 1)

    density = np.zeros((height, width), dtype=np.float32)

    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    # print(np.unique(counts))
    # counts = np.minimum(counts, 1e2)

    unique_coordinates = unique_coordinates.astype(np.int32)

    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    density = density / np.max(density)


    return density, normalization_dict




def normalize_point(point, normalization_dict):

    min_coords = normalization_dict["min_coords"]
    max_coords = normalization_dict["max_coords"]
    image_res = normalization_dict["image_res"]

    point_2d = \
        np.round(
            (point[:2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * image_res)
    point_2d = np.minimum(np.maximum(point_2d, np.zeros_like(image_res)),
                            image_res - 1)

    point[:2] = point_2d.tolist()

    return point

def normalize_annotations(scene_path, normalization_dict):
    annotation_path = os.path.join(scene_path, "annotation_3d.json")
    with open(annotation_path, "r") as f:
        annotation_json = json.load(f)

    for line in annotation_json["lines"]:
        point = line["point"]
        point = normalize_point(point, normalization_dict)
        line["point"] = point

    for junction in annotation_json["junctions"]:
        point = junction["coordinate"]
        point = normalize_point(point, normalization_dict)
        junction["coordinate"] = point

    return annotation_json

def parse_floor_plan_polys(annos):
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

        if semantic['type'] == 'outwall':
            outerwall_planes = semantic['planeID']

    # extract hole vertices
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])

    # outerwall_floor = []
    # for planeID in outerwall_planes:
    #     lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
    #     lineIDs = np.setdiff1d(lineIDs, lines_holes)
    #     junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
    #     for start, end in junction_pairs:
    #         if start in junction_floor and end in junction_floor:
    #             outerwall_floor.append([start, end])

    # outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    # polygons.append([outerwall_polygon[0], 'outwall'])

    return polygons

def convert_lines_to_vertices(lines):
    """
    convert line representation to polygon vertices

    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons



def generate_coco_dict(annos, polygons, curr_instance_id, curr_img_id, ignore_types):

    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])

    coco_annotation_dict_list = []

    for poly_ind, (polygon, poly_type) in enumerate(polygons):
        if poly_type in ignore_types:
            continue

        polygon = junctions[np.array(polygon)]

        poly_shapely = Polygon(polygon)
        area = poly_shapely.area

        # assert area > 10
        # if area < 100:
        if poly_type not in ['door', 'window'] and area < 100:
            continue
        if poly_type in ['door', 'window'] and area < 1:
            continue
        
        rectangle_shapely = poly_shapely.envelope

        ### here we convert door/window annotation into a single line
        if poly_type in ['door', 'window']:
            assert polygon.shape[0] == 4
            midp_1 = (polygon[0] + polygon[1])/2
            midp_2 = (polygon[1] + polygon[2])/2
            midp_3 = (polygon[2] + polygon[3])/2
            midp_4 = (polygon[3] + polygon[0])/2

            dist_1_3 = np.square(midp_1 -midp_3).sum()
            dist_2_4 = np.square(midp_2 -midp_4).sum()
            if dist_1_3 > dist_2_4:
                polygon = np.row_stack([midp_1, midp_3])
            else:
                polygon = np.row_stack([midp_2, midp_4])

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
                "category_id": type2id[poly_type],
                "id": curr_instance_id}
        
        coco_annotation_dict_list.append(coco_annotation_dict)
        curr_instance_id += 1


    return coco_annotation_dict_list