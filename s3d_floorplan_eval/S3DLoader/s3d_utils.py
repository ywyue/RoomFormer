"""
This code is an adaptation that uses Structured 3D for the code base.

Reference: https://github.com/bertjiazheng/Structured3D
"""

import numpy as np
import cv2
from shapely.geometry import Polygon
import random

type2id = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17, 'outwall':-1}

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

    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    polygons.append([outerwall_polygon[0], 'outwall'])

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

def generate_floorplan(annos, polygons, height, width, ignore_types, include_types=None, fillpoly=True, constant_color=False, shuffle=False):
    """
    plot floorplan

    """

    floor_map = np.zeros((height, width))

    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])

    room_ind = 0
    if shuffle:
        room_ind = np.random.randint(0, 2)

    polygons_list = []
    polygons_type_list = []
    for poly_ind, (polygon, poly_type) in enumerate(polygons):
        if poly_type in ignore_types:
            continue
        if include_types is not None and poly_type not in include_types:
            continue

        polygon = junctions[np.array(polygon)].astype(np.int32)

        poly_shapely = Polygon(polygon)
        area = poly_shapely.area

        # assert area > 10
        # if area < 100:
        #     continue
        if poly_type not in ['door', 'window'] and area < 100:
            continue
        if poly_type in ['door', 'window'] and area < 1:
            continue
        
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

        polygons_list.append(polygon)
        polygons_type_list.append(type2id[poly_type])

    if shuffle:
        random.shuffle(polygons_list)
    for poly_ind, polygon in enumerate(polygons_list):

        if shuffle:
            room_ind += np.random.randint(1, 2)
        else:
            room_ind += 1

        if fillpoly:
            if constant_color:
                clr = 1.
            else:
                clr = room_ind
            cv2.fillPoly(floor_map, [polygon], color=clr)
        else:
            assert constant_color
            cv2.polylines(floor_map, [polygon.astype(np.int32)], isClosed=True, color=1., thickness=3)

    return floor_map, polygons_list, polygons_type_list