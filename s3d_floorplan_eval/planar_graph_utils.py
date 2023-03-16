import os
import numpy as np
import cv2
from scipy import ndimage
from shapely.geometry import Polygon


def extract_regions(adj_mat, corners, corner_sorted):
    all_regions = list()
    cur_idx = 0
    corners = corners.astype(np.int)
    nb_orders = _sort_neighours(adj_mat, corners)
    while cur_idx is not None:
        regions = _get_regions_for_corner(cur_idx, adj_mat, nb_orders)
        all_regions.extend(regions)
        cur_idx = _get_new_start(adj_mat, cur_idx, corners)

    outwall_idx = get_outwall(all_regions, corners, corner_sorted)
    all_regions.pop(outwall_idx)

    # all_regions = filter_regions(all_regions) # only used for drawing visualization
    # return all_regions

    all_regions_coords = [corners[regions] for regions in all_regions]
    return all_regions_coords


def get_outwall(all_regions, corners, corner_sorted):
    """
        Find the outermost boundary loop, which should be discarded
    """
    if corner_sorted:
        regions_for_top_bot = np.nonzero([(0 in region and len(corners) - 1 in region) for region in all_regions])[0]
        if len(regions_for_top_bot) == 1:
            return regions_for_top_bot[0]
        else:
            areas = [_compute_region_area(corners[all_regions[idx]]) for idx in range(len(all_regions))]
            max_idx = np.argmax(areas)
        return max_idx
    else:
        areas = [_compute_region_area(corners[all_regions[idx]]) for idx in range(len(all_regions))]
        max_idx = np.argmax(areas)
        return max_idx



# def filter_regions(all_regions):
#     areas = [_compute_region_area(corners[all_regions[idx]]) for idx in range(len(all_regions))]
#     all_regions = [region for idx, region in enumerate(all_regions) if areas[idx] > 20]
#     return all_regions


def _compute_region_area(region):
    edge_map = np.zeros([256, 256])
    for idx, c in enumerate(region[:-1]):
        cv2.line(edge_map, tuple(c), tuple(region[idx + 1]), 1, 3)
    reverse_edge_map = 1 - edge_map
    label, num_features = ndimage.label(reverse_edge_map)
    if num_features < 2:
        return 0
        # import pdb; pdb.set_trace()
        # raise ValueError('Invalid region structure')
    bg_label = label[0, 0]
    num_labels = [(label == l).sum() for l in range(1, num_features + 1)]
    num_labels[bg_label - 1] = 0
    room_label = np.argmax(num_labels) + 1
    area = (label == room_label).sum()
    return area


def _get_regions_for_corner(cur_idx, adj_mat, nb_orders):
    regions = list()
    if adj_mat[cur_idx].sum() == 0:
        assert ValueError('Zero-degree corner, should not reach here')
    # elif adj_mat[cur_idx].sum() == 1:  # remove the connection if only one neighbour
    #     other_idx = nb_orders[0]
    #     import pdb; pdb.set_trace()
    #     adj_mat[cur_idx, other_idx] = 0
    else:
        v_s = cur_idx
        know_v_q = False
        while v_s is not None:
            if not know_v_q:
                v_p, v_q = _find_wedge_nbs(v_s, nb_orders, adj_mat)
                if v_p is None:  # cannot find proper wedge, remove this corner
                    adj_mat[v_s, :] = 0
                    adj_mat[:, v_s] = 0
                    break
            else:
                assert v_q is not None, 'v_q should be known here'
                v_p = _find_wedge_third_v(v_q, v_s, nb_orders, adj_mat, dir=-1)
                if v_p is None:
                    adj_mat[v_s, :] = 0
                    adj_mat[:, v_s] = 0
                    break
            cur_region = [v_p, v_s, ]
            # try:
            assert adj_mat[v_p, v_s] == 1, 'Wrong connection matrix!'
            # except:
            #     import pdb; pdb.set_trace()
            adj_mat[v_p, v_s] = 0
            region_i = 0
            closed_polygon = False
            while v_q is not None:  # tracking the current region
                cur_region.append(v_q)
                assert adj_mat[v_s, v_q] == 1, 'Wrong connection matrix!'
                adj_mat[v_s, v_q] = 0
                # update the nb order list for the current v_s
                if v_q == cur_region[0]:  # get a closed polygon
                    closed_polygon = True
                    break
                else:
                    v_p = cur_region[region_i + 1]
                    v_s = cur_region[region_i + 2]
                    v_q = _find_wedge_third_v(v_p, v_s, nb_orders, adj_mat, dir=1)
                    if v_q == None:
                        closed_polygon = False
                        break
                    region_i += 1

            if closed_polygon:  # find a closed region, keep iteration
                regions.append(cur_region)
                found_next = False
                for temp_i in range(1, len(cur_region)):
                    if adj_mat[cur_region[temp_i], cur_region[temp_i - 1]] == 1:
                        found_next = True
                        v_s_idx = temp_i
                        break
                if not found_next:
                    v_s = None
                else:
                    v_s = cur_region[v_s_idx]
                    v_q = cur_region[v_s_idx - 1]
                    know_v_q = True
            else:  # no closed region, directly quit the search for the current v_s
                break
    return regions


def _find_wedge_nbs(v_s, nb_orders, adj_mat):
    sorted_nbs = nb_orders[v_s]
    start_idx = 0
    while True:
        if start_idx == -len(sorted_nbs):
            return None, None
        v_p, v_q = sorted_nbs[start_idx], sorted_nbs[start_idx - 1]
        if adj_mat[v_p, v_s] == 1 and adj_mat[v_s, v_q] == 1:
            return v_p, v_q
        else:
            start_idx -= 1


def _find_wedge_third_v(v1, v2, nb_orders, adj_mat, dir):
    sorted_nbs = nb_orders[v2]
    v1_idx = sorted_nbs.index(v1)
    if dir == 1:
        v3_idx = v1_idx - 1
        while adj_mat[v2, sorted_nbs[v3_idx]] == 0:
            if sorted_nbs[v3_idx] == v1:
                return None
            v3_idx -= 1
    elif dir == -1:
        v3_idx = v1_idx + 1 if v1_idx <= len(sorted_nbs) - 2 else 0
        while adj_mat[sorted_nbs[v3_idx], v2] == 0:
            if sorted_nbs[v3_idx] == v1:
                return None
            v3_idx = v3_idx + 1 if v3_idx <= len(sorted_nbs) - 2 else 0
    else:
        raise ValueError('unknown dir {}'.format(dir))
    return sorted_nbs[v3_idx]


def _get_new_start(adj_mat, cur_idx, corners):
    for i in range(cur_idx, len(corners)):
        if adj_mat[i].sum() > 0:
            return i
    return None


def _sort_neighours(adj_mat, corners):
    nb_orders = dict()
    for idx, c in enumerate(corners):
        nb_ids = np.nonzero(adj_mat[idx])[0]
        nb_degrees = [_compute_degree(c, corners[other_idx]) for other_idx in nb_ids]
        degree_ranks = np.argsort(nb_degrees)
        sort_nb_ids = [nb_ids[i] for i in degree_ranks]
        nb_orders[idx] = sort_nb_ids
    return nb_orders


def _compute_degree(c1, c2):
    vec = (c2[0] - c1[0], -(c2[1] - c1[1]))  # note that the y direction should be flipped (image coord system)
    cos = (vec[0] * 1 + vec[1] * 0) / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    theta = np.arccos(cos)
    if vec[1] < 0:
        theta = np.pi * 2 - theta
    return theta


def preprocess_pg(pg):
    corners = pg['corners']
    edge_pairs = pg['edges']
    adj_mat = np.zeros([len(corners), len(corners)])
    for edge_pair in edge_pairs:
        c1, c2 = edge_pair
        adj_mat[c1][c2] = 1
        adj_mat[c2][c1] = 1

    return corners, adj_mat


def cleanup_pg(pg):
    corners = pg['corners']
    edge_pairs = pg['edges']
    adj_list = [[] for _ in range(len(corners))]

    for edge_pair in edge_pairs:
        adj_list[edge_pair[0]].append(edge_pair[1])
        adj_list[edge_pair[1]].append(edge_pair[0])

    for idx in range(len(corners)):
        if len(adj_list[idx]) < 2:
            _remove_corner(idx, adj_list)

    new_corners = list()
    removed_ids = list()
    old_to_new = dict()
    counter = 0
    for c_i in range(len(adj_list)):
        if len(adj_list[c_i]) > 0:
            assert len(adj_list[c_i]) >= 2
            new_corners.append(corners[c_i])
            old_to_new[c_i] = counter
            counter += 1
        else:
            removed_ids.append(c_i)

    new_edges = list()
    for c_i_1 in range(len(adj_list)):
        for c_i_2 in adj_list[c_i_1]:
            if c_i_1 < c_i_2:
                new_edge = (old_to_new[c_i_1], old_to_new[c_i_2])
                new_edges.append(new_edge)
    new_corners = np.array(new_corners)
    new_edges = np.array(new_edges)
    new_pg = {
        'corners': new_corners,
        'edges': new_edges,
    }
    return new_pg


def _remove_corner(idx, adj_list):
    assert len(adj_list[idx]) <= 1
    if len(adj_list[idx]) == 0:
        return
    nbs = list(adj_list[idx])
    adj_list[idx].pop(0)
    for nb in nbs:
        adj_list[nb].remove(idx)
        if len(adj_list[nb]) < 2:
            _remove_corner(nb, adj_list)


def get_regions_from_pg(pg, corner_sorted):
    pg = cleanup_pg(pg)
    corners, adj_mat = preprocess_pg(pg)
    if len(corners) == 0:
        regions = []
    else:
        regions = extract_regions(adj_mat, corners, corner_sorted)
    return regions


def convert_annot(annot):
    corners = np.array(list(annot.keys()))
    corners_mapping = {tuple(c): idx for idx, c in enumerate(corners)}
    edges = set()
    for corner, connections in annot.items():
        idx_c = corners_mapping[tuple(corner)]
        for other_c in connections:
            idx_other_c = corners_mapping[tuple(other_c)]
            if (idx_c, idx_other_c) not in edges and (idx_other_c, idx_c) not in edges:
                edges.add((idx_c, idx_other_c))
    edges = np.array(list(edges))
    pg_data = {
        'corners': corners,
        'edges': edges
    }
    return pg_data


colors_12 = [
    "#DCECC9",
    "#B3DDCC",
    "#8ACDCE",
    "#62BED2",
    "#46AACE",
    "#3D91BE",
    "#3677AE",
    "#2D5E9E",
    "#24448E",
    "#1C2B7F",
    "#162165",
    "#11174B",
]


def plot_floorplan_with_regions(regions, corners, edges, scale):
    colors = colors_12[:8]

    regions = [(region * scale / 256).round().astype(np.int) for region in regions]
    corners = (corners * scale / 256).round().astype(np.int)

    # define the color map
    room_colors = [colors[i % 8] for i in range(len(regions))]

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
        ind = np.argsort(np.array(avg_corner)[:, 0], axis=0)
        regions = np.array(regions)[ind]

    for idx, polygon in enumerate(regions):
        cv2.fillPoly(room_map, [polygon], color=idx + 1)

    image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 4))

    pointColor = tuple((np.array([0.95, 0.3, 0.3, 1]) * 255).astype(np.uint8).tolist())
    for point in corners:
        cv2.circle(image, tuple(point), color=pointColor, radius=12, thickness=-1)
        cv2.circle(image, tuple(point), color=(255, 255, 255, 255), radius=6, thickness=-1)

    for edge in edges:
        c1 = corners[edge[0]]
        c2 = corners[edge[1]]
        cv2.line(image, tuple(c1), tuple(c2), color=(0, 0, 0, 255), thickness=3)

    return image

