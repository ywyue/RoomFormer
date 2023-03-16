"""
Utilities for floorplan visualization.
"""
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.patches as mpatches

import cv2 
import numpy as np
from imageio import imsave

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


def filled_arc(center, radius, theta1, theta2, ax, color):
    """Draw arc for door
    """
    circ = mpatches.Wedge(center, radius, theta1, theta2, fill=True, color=color, linewidth=1, ec='#000000')
    pt1 = (radius * (np.cos(theta1*np.pi/180.)) + center[0],
           radius * (np.sin(theta1*np.pi/180.)) + center[1])
    pt2 = (radius * (np.cos(theta2*np.pi/180.)) + center[0],
           radius * (np.sin(theta2*np.pi/180.)) + center[1])
    pt3 = center
    ax.add_patch(circ)



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
    ax.plot(x, y, color=color, zorder=zorder, alpha=alpha, linewidth=linewidth)


def plot_corners(ax, ob, color=BLACK, zorder=1, alpha=1):
    x, y = ob.xy
    ax.scatter(x, y, color=color, marker='o')