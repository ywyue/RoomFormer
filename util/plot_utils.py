"""
Utilities for floorplan visualization.
"""
import torch
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.patches as mpatches

import cv2 
import numpy as np
from imageio import imsave

from shapely.geometry import LineString
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch


colors_12 = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58230",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd7b4"
]

semantics_cmap = {
    0: '#e6194b',
    1: '#3cb44b',
    2: '#ffe119',
    3: '#0082c8',
    4: '#f58230',
    5: '#911eb4',
    6: '#46f0f0',
    7: '#f032e6',
    8: '#d2f53c',
    9: '#fabebe',
    10: '#008080',
    11: '#e6beff',
    12: '#aa6e28',
    13: '#fffac8',
    14: '#800000',
    15: '#aaffc3',
    16: '#808000',
    17: '#ffd7b4'
}

semantics_label = {
    0: 'Living Room',
    1: 'Kitchen',
    2: 'Bedroom',
    3: 'Bathroom',
    4: 'Balcony',
    5: 'Corridor',
    6: 'Dining room',
    7: 'Study',
    8: 'Studio',
    9: 'Store room',
    10: 'Garden',
    11: 'Laundry room',
    12: 'Office',
    13: 'Basement',
    14: 'Garage',
    15: 'Misc.',
    16: 'Door',
    17: 'Window'
}


BLUE = '#6699cc'
GRAY = '#999999'
DARKGRAY = '#333333'
YELLOW = '#ffcc33'
GREEN = '#339933'
RED = '#ff3333'
BLACK = '#000000'


def plot_floorplan_with_regions(regions, corners=None, edges=None, scale=256):
    """Draw floorplan map where different colors indicate different rooms
    """
    colors = colors_12

    regions = [(region * scale / 256).round().astype(np.int) for region in regions]

    # define the color map
    room_colors = [colors[i] for i in range(len(regions))]

    colorMap = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in room_colors]
    colorMap = np.asarray(colorMap)
    if len(regions) > 0:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap], axis=0).astype(
            np.uint8)
    else:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0)], axis=0).astype(
            np.uint8)
    # when using opencv, we need to flip, from RGB to BGR
    colorMap = colorMap[:, ::-1]

    alpha_channels = np.zeros(colorMap.shape[0], dtype=np.uint8)
    alpha_channels[1:len(regions) + 1] = 150

    colorMap = np.concatenate([colorMap, np.expand_dims(alpha_channels, axis=-1)], axis=-1)

    room_map = np.zeros([scale, scale]).astype(np.int32)
    # sort regions
    if len(regions) > 1:
        avg_corner = [region.mean(axis=0) for region in regions]
        ind = np.argsort(np.square(np.array(avg_corner)).sum(axis=1), axis=0)
        regions = np.array(regions)[ind]

    for idx, polygon in enumerate(regions):
        cv2.fillPoly(room_map, [polygon], color=idx + 1)

    image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 4))

    pointColor = (0,0,0,255)
    lineColor = (0,0,0,255)

    for region in regions:
        for i, point in enumerate(region):
            if i == len(region)-1:
                cv2.line(image, tuple(point), tuple(region[0]), color=lineColor, thickness=5)
            else:    
                cv2.line(image, tuple(point), tuple(region[i+1]), color=lineColor, thickness=5)

    for region in regions:
        for i, point in enumerate(region):
            cv2.circle(image, tuple(point), color=pointColor, radius=12, thickness=-1)
            cv2.circle(image, tuple(point), color=(255, 255, 255, 0), radius=6, thickness=-1)

    return image


def plot_score_map(corner_map, scores):
    """Draw score map overlaid on the density map
    """
    score_map = np.zeros([356, 356, 3])
    score_map[100:, 50:306] = corner_map
    cv2.putText(score_map, 'room_prec: '+str(round(scores['room_prec']*100, 1)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (252, 252, 0), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'room_rec: '+str(round(scores['room_rec']*100, 1)), (190, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (252, 252, 0), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'corner_prec: '+str(round(scores['corner_prec']*100, 1)), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'corner_rec: '+str(round(scores['corner_rec']*100, 1)), (190, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'angles_prec: '+str(round(scores['angles_prec']*100, 1)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'angles_rec: '+str(round(scores['angles_rec']*100, 1)), (190, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (0, 255, 0), 1, cv2.LINE_AA)

    return score_map


def plot_room_map(preds, room_map, im_size=256):
    """Draw room polygons overlaid on the density map
    """
    for i, corner in enumerate(preds):
        if i == len(preds)-1:
            cv2.line(room_map, (round(corner[0]), round(corner[1])), (round(preds[0][0]), round(preds[0][1])), (252, 252, 0), 2)
        else:
            cv2.line(room_map, (round(corner[0]), round(corner[1])), (round(preds[i+1][0]), round(preds[i+1][1])), (252, 252, 0), 2)
        cv2.circle(room_map, (round(corner[0]), round(corner[1])), 2, (0, 0, 255), 2)
        cv2.putText(room_map, str(i), (round(corner[0]), round(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
    return room_map


def plot_anno(img, annos, save_path, transformed=False, draw_poly=True, draw_bbx=True, thickness=2):
    """Visualize annotation
    """
    img = np.repeat(np.expand_dims(img,2), 3, axis=2)
    num_inst = len(annos)

    bbx_color = (0, 255, 0)
    # poly_color = (0, 255, 0)
    for j in range(num_inst):
        
        if draw_bbx:
            bbox = annos[j]['bbox']
            if transformed: 
                start_point = (round(bbox[0]), round(bbox[1]))
                end_point = (round(bbox[2]), round(bbox[3]))
            else:
                start_point = (round(bbox[0]), round(bbox[1]))
                end_point = (round(bbox[0]+bbox[2]), round(bbox[1]+bbox[3]))
            # Blue color in BGR
            img = cv2.rectangle(img, start_point, end_point, bbx_color, thickness)

        if draw_poly:
            verts = annos[j]['segmentation'][0]
            if isinstance(verts, list):
                verts = np.array(verts)
            verts = verts.reshape(-1,2)

            for i, corner in enumerate(verts):
                if i == len(verts)-1:
                    cv2.line(img, (round(corner[0]), round(corner[1])), (round(verts[0][0]), round(verts[0][1])), (0, 252, 252), 1)
                else:
                    cv2.line(img, (round(corner[0]), round(corner[1])), (round(verts[i+1][0]), round(verts[i+1][1])), (0, 252, 252), 1)
                cv2.circle(img, (round(corner[0]), round(corner[1])), 2, (255, 0, 0), 2)
                cv2.putText(img, str(i), (round(corner[0]), round(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, (0, 255, 0), 1, cv2.LINE_AA)

    imsave(save_path, img)


def plot_coords(ax, ob, color=BLACK, zorder=1, alpha=1, linewidth=1):
    x, y = ob.xy
    ax.plot(x, y, color=color, zorder=zorder, alpha=alpha, linewidth=linewidth, solid_joinstyle='miter')


def plot_corners(ax, ob, color=BLACK, zorder=1, alpha=1):
    x, y = ob.xy
    ax.scatter(x, y, color=color, marker='o')

def get_angle(p1, p2):
    """Get the angle of this line with the horizontal axis.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return angle

def filled_arc(e1, e2, direction, radius, ax, color):
    """Draw arc for door
    """
    angle = get_angle(e1,e2)
    if direction == 'counterclock':
        theta1 = angle
        theta2 = angle + 90.0
    else:
        theta1 = angle - 90.0
        theta2 = angle
    circ = mpatches.Wedge(e1, radius, theta1, theta2, fill=True, color=color, linewidth=1, ec='#000000')
    ax.add_patch(circ)


def plot_semantic_rich_floorplan(polygons, file_name, prec=None, rec=None):
    """plot semantically-rich floorplan (i.e. with additional room label, door, window)
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    polygons_windows = []
    polygons_doors = []

    # Iterate over rooms to draw black outline
    for (poly, poly_type) in polygons:
        if len(poly) > 2:
            polygon = Polygon(poly)
            if poly_type != 16 and poly_type != 17:
                plot_coords(ax, polygon.exterior, alpha=1.0, linewidth=10)

    # Iterate over all predicted polygons (rooms, doors, windows)
    for (poly, poly_type) in polygons:
        if poly_type == 'outqwall':  # unclear what is this?
            pass
        elif poly_type == 16:  # Door
            door_length = math.dist(poly[0], poly[1])
            polygons_doors.append([poly, poly_type, door_length])
        elif poly_type == 17:  # Window
            polygons_windows.append([poly, poly_type])
        else: # regular room
            polygon = Polygon(poly)
            patch = PolygonPatch(polygon, facecolor='#FFFFFF', alpha=1.0, linewidth=0)
            ax.add_patch(patch)
            patch = PolygonPatch(polygon, facecolor=semantics_cmap[poly_type], alpha=0.5, linewidth=1, capstyle='round', edgecolor='#000000FF')
            ax.add_patch(patch)
            ax.text(np.mean(poly[:, 0]), np.mean(poly[:, 1]), semantics_label[poly_type], size=6, horizontalalignment='center', verticalalignment='center')


    # Compute door size statistics (median)
    door_median_size = np.median([door_length for (_, _, door_length) in polygons_doors])

    # Draw doors
    for (poly, poly_type, door_size) in polygons_doors:

        door_size_y = np.abs(poly[0,1]-poly[1,1])
        door_size_x = np.abs(poly[0,0]-poly[1,0])
        if door_size_y > door_size_x:
            if poly[1,1] > poly[0,1]:
                e1 = poly[0]
                e2 = poly[1]
            else:
                e1 = poly[1]
                e2 = poly[0]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, 'clock', door_size, ax, 'white')
            else:
                filled_arc(e1, e2, 'clock', door_size/2, ax, 'white')
                filled_arc(e2, e1, 'counterclock', door_size/2, ax, 'white')

        else:
            if poly[1,0] > poly[0,0]:
                e1 = poly[1]
                e2 = poly[0]
            else:
                e1 = poly[0]
                e2 = poly[1]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, 'counterclock', door_size, ax, 'white')
            else:
                filled_arc(e1, e2, 'counterclock', door_size/2, ax, 'white')
                filled_arc(e2, e1, 'clock', door_size/2, ax, 'white')


    # Draw windows
    for (line, line_type) in polygons_windows:
        line = LineString(line)
        poly = line.buffer(1.5, cap_style=2)
        patch = PolygonPatch(poly, facecolor='#FFFFFF', alpha=1.0, linewidth=1, linestyle='dashed')
        ax.add_patch(patch)

    title = ''
    if prec is not None:
        title = 'prec: ' + str(round(prec * 100, 1)) + ', rec: ' + str(round(rec * 100, 1))
    plt.title(file_name.split('/')[-1] + ' ' + title)
    plt.axis('equal')
    plt.axis('off')

    print(f'>>> {file_name}')
    # fig.savefig(file_name[:-3]+'svg', dpi=fig.dpi, format='svg')
    fig.savefig(file_name, dpi=fig.dpi)