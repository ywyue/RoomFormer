"""
This is a hack implementation for evaluation on SceneCAD

Mostly copy-paste from Evaluator.py (from MonteFloor) with small modification
"""

import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial import Delaunay
import os
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString

corner_metric_thresh = 20
angle_metric_thresh = 5



# colormap_255 = [[i, i, i] for i in range(40)]

class Evaluator_SceneCAD():
    def __init__(self, data_rw=None, options=None):
        self.data_rw = data_rw
        self.options = options

        self.device = torch.device("cuda")

    def polygonize_mask(self, mask, degree, return_mask=True):
        h, w = mask.shape[0], mask.shape[1]
        mask = mask

        room_mask = 255 * (mask == 1)
        room_mask = room_mask.astype(np.uint8)
        room_mask_inv = 255 - room_mask

        ret, thresh = cv2.threshold(room_mask_inv, 250, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        cnt = contours[0]
        max_area = cv2.contourArea(cnt)

        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        perimeter = cv2.arcLength(cnt, True)
        # epsilon = 0.01 * cv2.arcLength(cnt, True)
        epsilon = degree * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # approx = np.concatenate([approx, approx[0][None]], axis=0)
        approx = approx.astype(np.int32).reshape((-1, 2))

        # approx_tensor = torch.tensor(approx, device=self.device)

        # return approx_tensor
        if return_mask:
            room_filled_map = np.zeros((h, w))
            cv2.fillPoly(room_filled_map, [approx], color=1.)

            return approx, room_filled_map
        else:
            return approx

    def print_res_str_for_latex(self, quant_result_dict):

        str_fields = ""
        str_values = ""

        avg_value_prec = 0
        avg_value_rec = 0
        for k_ind, k in enumerate(quant_result_dict.keys()):
            str_fields += " & " + k
            str_values += " & %.2f " % quant_result_dict[k]

            if k_ind % 2 == 0:
                avg_value_prec += quant_result_dict[k] / 3
            else:
                avg_value_rec += quant_result_dict[k] / 3

            str_fields += "tm_prec & tm_rec"

        str_values += " & %.2f " % avg_value_prec
        str_values += " & %.2f " % avg_value_rec

        str_fields += " \\\\"
        str_values += " \\\\"

        print(str_fields)
        print(str_values)

    def calc_gradient(self, room_map):
        grad_x = np.abs(room_map[:, 1:] - room_map[:, :-1])
        grad_y = np.abs(room_map[1:] - room_map[:-1])

        grad_xy = np.zeros_like(room_map)
        grad_xy[1:] = grad_y
        grad_xy[:, 1:] = np.maximum(grad_x, grad_xy[:,1:])

        plt.figure()
        plt.axis("off")
        plt.imshow(grad_xy, cmap="gray")
        # plt.show()
        plt.savefig("grad.png", bbox_inches='tight')

        plt.figure()
        plt.axis("off")
        plt.imshow(room_map, cmap="gray")
        # plt.show()
        plt.savefig("joint_mask.png", bbox_inches='tight')
        assert False

    def evaluate_scene(self, room_polys, gt_polys, show=False, name="ours", dataset_type="scenecad"):

        gt_polys_list = [np.concatenate([poly, poly[None, 0]]) for poly in gt_polys]
        room_polys = [np.concatenate([poly, poly[None, 0]]) for poly in room_polys]
        
        ignore_mask_region = None

        # img_size = (joint_room_map.shape[0], joint_room_map.shape[1])
        img_size = (256, 256)
        quant_result_dict = self.get_quantitative(gt_polys_list, ignore_mask_region, room_polys, None, img_size, dataset_type=dataset_type)

        return quant_result_dict

    def get_quantitative(self, gt_polys, ignore_mask_region, pred_polys=None, masks_list=None, img_size=(256, 256), dataset_type="s3d"):
        def get_room_metric():
            pred_overlaps = [False] * len(pred_room_map_list)

            for pred_ind1 in range(len(pred_room_map_list) - 1):
                pred_map1 = pred_room_map_list[pred_ind1]

                for pred_ind2 in range(pred_ind1 + 1, len(pred_room_map_list)):
                    pred_map2 = pred_room_map_list[pred_ind2]

                    if dataset_type == "s3d":
                        kernel = np.ones((5, 5), np.uint8)
                    else:
                        kernel = np.ones((3, 3), np.uint8)

                    # todo: for our method, the rooms share corners and edges, need to check here
                    pred_map1_er = cv2.erode(pred_map1, kernel)
                    pred_map2_er = cv2.erode(pred_map2, kernel)

                    intersection = (pred_map1_er + pred_map2_er) == 2
                    # intersection = (pred_map1 + pred_map2) == 2

                    intersection_area = np.sum(intersection)

                    if intersection_area >= 1:
                        pred_overlaps[pred_ind1] = True
                        pred_overlaps[pred_ind2] = True

            # import pdb; pdb.set_trace()
            room_metric = [np.bool((1 - pred_overlaps[ind]) * pred2gt_exists[ind]) for ind in range(len(pred_polys))]

            return room_metric

        def get_corner_metric():

            room_corners_metric = []
            for pred_poly_ind, gt_poly_ind in enumerate(pred2gt_indices):
                p_poly = pred_polys[pred_poly_ind][:-1] # Last vertex = First vertex

                p_poly_corner_metrics = [False] * p_poly.shape[0]
                if not room_metric[pred_poly_ind]:
                    room_corners_metric += p_poly_corner_metrics
                    continue

                gt_poly = gt_polys[gt_poly_ind][:-1]

                # for v in p_poly:
                #     v_dists = np.linalg.norm(v[None,:] - gt_poly, axis=1, ord=2)
                #     v_min_dist = np.min(v_dists)
                #
                #     v_tp = v_min_dist <= 10
                #     room_corners_metric.append(v_tp)

                for v in gt_poly:
                    v_dists = np.linalg.norm(v[None,:] - p_poly, axis=1, ord=2)
                    v_min_dist_ind = np.argmin(v_dists)
                    v_min_dist = v_dists[v_min_dist_ind]

                    if not p_poly_corner_metrics[v_min_dist_ind]:
                        v_tp = v_min_dist <= corner_metric_thresh
                        p_poly_corner_metrics[v_min_dist_ind] = v_tp

                room_corners_metric += p_poly_corner_metrics

            return room_corners_metric

        def get_angle_metric():

            def get_line_vector(p1, p2):
                p1 = np.concatenate((p1, np.array([1])))
                p2 = np.concatenate((p2, np.array([1])))

                line_vector = -np.cross(p1, p2)

                return line_vector

            def get_poly_orientation(my_poly):
                angles_sum = 0
                for v_ind, _ in enumerate(my_poly):
                    if v_ind < len(my_poly) - 1:
                        v_sides = my_poly[[v_ind - 1, v_ind, v_ind, v_ind + 1], :]
                    else:
                        v_sides = my_poly[[v_ind - 1, v_ind, v_ind, 0], :]

                    v1_vector = get_line_vector(v_sides[0], v_sides[1])
                    v1_vector = v1_vector / (np.linalg.norm(v1_vector, ord=2) + 1e-4)
                    v2_vector = get_line_vector(v_sides[2], v_sides[3])
                    v2_vector = v2_vector / (np.linalg.norm(v2_vector, ord=2) + 1e-4)

                    orientation = (v_sides[1, 1] - v_sides[0, 1]) * (v_sides[3, 0] - v_sides[1, 0]) - (
                            v_sides[3, 1] - v_sides[1, 1]) * (
                                          v_sides[1, 0] - v_sides[0, 0])

                    v1_vector_2d = v1_vector[:2] / (v1_vector[2] + 1e-4)
                    v2_vector_2d = v2_vector[:2] / (v2_vector[2] + 1e-4)

                    v1_vector_2d = v1_vector_2d / (np.linalg.norm(v1_vector_2d, ord=2) + 1e-4)
                    v2_vector_2d = v2_vector_2d / (np.linalg.norm(v2_vector_2d, ord=2) + 1e-4)

                    angle_cos = v1_vector_2d.dot(v2_vector_2d)
                    angle_cos = np.clip(angle_cos, -1, 1)

                    # G.T. has clockwise orientation, remove minus in the equation

                    angle = np.sign(orientation) * np.abs(np.arccos(angle_cos))
                    angle_degree = angle * 180 / np.pi

                    angles_sum += angle_degree

                return np.sign(angles_sum)

            def get_angle_v_sides(inp_v_sides, poly_orient):
                v1_vector = get_line_vector(inp_v_sides[0], inp_v_sides[1])
                v1_vector = v1_vector / (np.linalg.norm(v1_vector, ord=2) + 1e-4)
                v2_vector = get_line_vector(inp_v_sides[2], inp_v_sides[3])
                v2_vector = v2_vector / (np.linalg.norm(v2_vector, ord=2) + 1e-4)

                orientation = (inp_v_sides[1, 1] - inp_v_sides[0, 1]) * (inp_v_sides[3, 0] - inp_v_sides[1, 0]) - (
                        inp_v_sides[3, 1] - inp_v_sides[1, 1]) * (
                                      inp_v_sides[1, 0] - inp_v_sides[0, 0])

                v1_vector_2d = v1_vector[:2] / (v1_vector[2]+ 1e-4)
                v2_vector_2d = v2_vector[:2] / (v2_vector[2]+ 1e-4)

                v1_vector_2d = v1_vector_2d / (np.linalg.norm(v1_vector_2d, ord=2) + 1e-4)
                v2_vector_2d = v2_vector_2d / (np.linalg.norm(v2_vector_2d, ord=2) + 1e-4)

                angle_cos = v1_vector_2d.dot(v2_vector_2d)
                angle_cos = np.clip(angle_cos, -1, 1)

                angle = poly_orient * np.sign(orientation) * np.arccos(angle_cos)
                angle_degree = angle * 180 / np.pi

                return angle_degree

            room_angles_metric = []
            for pred_poly_ind, gt_poly_ind in enumerate(pred2gt_indices):
                p_poly = pred_polys[pred_poly_ind][:-1] # Last vertex = First vertex

                p_poly_angle_metrics = [False] * p_poly.shape[0]
                if not room_metric[pred_poly_ind]:
                    room_angles_metric += p_poly_angle_metrics
                    continue

                gt_poly = gt_polys[gt_poly_ind][:-1]

                # for v in p_poly:
                #     v_dists = np.linalg.norm(v[None,:] - gt_poly, axis=1, ord=2)
                #     v_min_dist = np.min(v_dists)
                #
                #     v_tp = v_min_dist <= 10
                #     room_corners_metric.append(v_tp)

                gt_poly_orient = get_poly_orientation(gt_poly)
                p_poly_orient = get_poly_orientation(p_poly)

                for v_gt_ind, v in enumerate(gt_poly):
                    v_dists = np.linalg.norm(v[None,:] - p_poly, axis=1, ord=2)
                    v_ind = np.argmin(v_dists)
                    v_min_dist = v_dists[v_ind]

                    if v_min_dist > corner_metric_thresh:
                        # room_angles_metric.append(False)
                        continue

                    if v_ind < len(p_poly) - 1:
                        v_sides = p_poly[[v_ind - 1, v_ind, v_ind, v_ind + 1], :]
                    else:
                        v_sides = p_poly[[v_ind - 1, v_ind, v_ind, 0], :]

                    v_sides = v_sides.reshape((4,2))
                    pred_angle_degree = get_angle_v_sides(v_sides, p_poly_orient)

                    # Note: replacing some variables with values from the g.t. poly

                    if v_gt_ind < len(gt_poly) - 1:
                        v_sides = gt_poly[[v_gt_ind - 1, v_gt_ind, v_gt_ind, v_gt_ind + 1], :]
                    else:
                        v_sides = gt_poly[[v_gt_ind - 1, v_gt_ind, v_gt_ind, 0], :]

                    v_sides = v_sides.reshape((4, 2))
                    gt_angle_degree = get_angle_v_sides(v_sides, gt_poly_orient)

                    angle_metric = np.abs(pred_angle_degree - gt_angle_degree)

                    # room_angles_metric.append(angle_metric < 5)
                    p_poly_angle_metrics[v_ind] = angle_metric <= angle_metric_thresh

                    # if angle_metric > 5:
                    #     print(v_gt_ind, angle_metric)
                    #     print(pred_angle_degree, gt_angle_degree)
                    #     input("?")


                room_angles_metric += p_poly_angle_metrics

            for am, cm in zip(room_angles_metric, corner_metric):
                assert not (cm == False and am == True), "cm: %d am: %d" %(cm, am)

            return room_angles_metric

        def poly_map_sort_key(x):
            return np.sum(x[1])

        h, w = img_size

        gt_room_map_list = []
        for room_ind, poly in enumerate(gt_polys):
            room_map = np.zeros((h, w))
            cv2.fillPoly(room_map, [poly], color=1.)

            gt_room_map_list.append(room_map)

        gt_polys_sorted_indcs = [i[0] for i in sorted(enumerate(gt_room_map_list), key=poly_map_sort_key, reverse=True)]

        gt_polys = [gt_polys[ind] for ind in gt_polys_sorted_indcs]
        gt_room_map_list = [gt_room_map_list[ind] for ind in gt_polys_sorted_indcs]

        if pred_polys is not None:
            pred_room_map_list = []
            for room_ind, poly in enumerate(pred_polys):
                room_map = np.zeros((h, w))
                cv2.fillPoly(room_map, [poly], color=1.)

                pred_room_map_list.append(room_map)
        else:
            pred_room_map_list = masks_list

        gt2pred_indices = [-1] * len(gt_polys)
        gt2pred_exists = [False] * len(gt_polys)

        for gt_ind, gt_map in enumerate(gt_room_map_list):

            best_iou = 0.
            best_ind = -1
            for pred_ind, pred_map in enumerate(pred_room_map_list):

                # intersection = (1 - ignore_mask_region) * ((pred_map + gt_map) == 2)
                # union = (1 - ignore_mask_region) * ((pred_map + gt_map) >= 1)
                intersection = (pred_map + gt_map) == 2
                union = (pred_map + gt_map) >= 1

                iou = np.sum(intersection) / (np.sum(union) + 1)

                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_ind = pred_ind

            #         plt.figure()
            #         plt.subplot(121)
            #         plt.imshow(pred_map)
            #         plt.subplot(122)
            #         plt.imshow(gt_map)
            #         plt.show()
            # if best_ind == -1:
            #     plt.figure()
            #     plt.imshow(gt_map)
            #     plt.show()

            gt2pred_indices[gt_ind] = best_ind
            gt2pred_exists[gt_ind] = best_ind != -1

            # if best_ind == -1:
            #     plt.figure()
            #     plt.imshow(gt_map)
            #     plt.show()

        pred2gt_exists = [True if pred_ind in gt2pred_indices else False for pred_ind, _ in enumerate(pred_polys)]
        pred2gt_indices = [gt2pred_indices.index(pred_ind) if pred_ind in gt2pred_indices else -1 for pred_ind, _ in enumerate(pred_polys)]

        # print(gt2pred_indices)
        # print(pred2gt_indices)
        # assert False

        # import pdb; pdb.set_trace()
        room_metric = get_room_metric()
        if len(pred_polys) == 0:
            room_metric_prec = 0
        else:
            room_metric_prec = sum(room_metric) / float(len(pred_polys))
        room_metric_rec = sum(room_metric) / float(len(gt_polys))


        corner_metric = get_corner_metric()
        pred_corners_n = sum([poly.shape[0] - 1 for poly in pred_polys])
        gt_corners_n = sum([poly.shape[0] - 1 for poly in gt_polys])

        if pred_corners_n > 0:
            corner_metric_prec = sum(corner_metric) / float(pred_corners_n)
        else:
            corner_metric_prec = 0
        corner_metric_rec = sum(corner_metric) / float(gt_corners_n)


        angles_metric = get_angle_metric()

        if pred_corners_n > 0:
            angles_metric_prec = sum(angles_metric) / float(pred_corners_n)
        else:
            angles_metric_prec = 0
        angles_metric_rec = sum(angles_metric) / float(gt_corners_n)

        assert room_metric_prec <= 1
        assert room_metric_rec <= 1
        assert corner_metric_prec <= 1
        assert corner_metric_rec <= 1
        assert angles_metric_prec <= 1
        assert angles_metric_rec <= 1

        result_dict = {
            'room_iou': best_iou,
            'room_prec': room_metric_prec,
            'room_rec': room_metric_rec,
            'corner_prec': corner_metric_prec,
            'corner_rec': corner_metric_rec,
            'angles_prec': angles_metric_prec,
            'angles_rec': angles_metric_rec,
        }

        return result_dict
