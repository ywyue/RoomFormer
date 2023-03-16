import numpy as np
import cv2
import torch
import os
import time

from s3d_floorplan_eval.DataRW.DataRW import DataRW
from s3d_floorplan_eval.S3DLoader.S3DLoader import S3DLoader

class S3DRW(DataRW):
    def __init__(self, options, mode):
        """
        Class for accessing FloorNet dataset related data

        :param options:
        """
        # initialize the base class variables
        super(DataRW, self).__init__()

        self.options = options

        self.dataset_path = options.dataset_path
        self.scene_id = options.scene_id

        self.mcts_path = options.mcts_path
        self.creation_time = int(time.time())

        self.device = torch.device("cpu")

        # mode = "train"
        # mode = "online_eval"
        # For validation only
        # self.loader = S3DLoader(options, 'online_eval').dataset
        self.loader = S3DLoader(options, mode).dataset

        # gt_sample = iter(floornet_loader.dataset[int(self.scene_id)])
        # self.gt_sample = floornet_loader.load_sample(list(iter(floornet_loader.dataset))[int(self.scene_id)])

        if mode == "online_eval":
            scene_ind = int(self.scene_id[6:]) - 3000
        elif mode == "test":
            scene_ind = int(self.scene_id[6:]) - 3250
        elif mode == "train":
            scene_ind = int(self.scene_id[6:])
        else:
            assert False

        # print(len(list(iter(self.s3d_loader.data))))
        self.gt_sample = gt_sample = self.loader[scene_ind]
        self.gt_sample["density_map"] = torch.tensor(self.gt_sample["density_map"][None], device=self.device)
        self.gt_sample["room_map"] = torch.tensor(self.gt_sample["room_map"][None,:,:,None], device=self.device)
        self.gt_sample["wall_map"] = torch.tensor(self.gt_sample["wall_map"][None,:,:,None], device=self.device)


        self.density_map = self.gt_sample['density_map'][:,:,:,None]

        self.h, self.w = self.density_map.shape[1], self.density_map.shape[2]

        self.generate_input_map_from_props = self.generate_input_dict_from_room_props

    def get_gt_solution(self):
        """
        Read top-view density map of the scene

        :return:
        """
        img_path = os.path.join(self.dataset_path, str(self.scene_id) + "_density.png")
        density_map = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)[:,:, 0][None,:,:,None]

        density_map = torch.from_numpy(density_map).to(self.device)

        dm_min = torch.min(density_map)
        dm_max = torch.max(density_map)

        density_map = (density_map - dm_min) / (dm_max - dm_min)

        return density_map.type(torch.cuda.FloatTensor)

    def polygonize_mask(self, pm, return_mask=True):
        pm_np = pm.cpu().numpy()

        room_mask = 255 * (pm_np == 1)
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

        # define main island contour approx. and hull
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # approx = np.concatenate([approx, approx[0][None]], axis=0)
        approx = approx.astype(np.int32).reshape((1, -1, 2))

        if return_mask:
            room_filled_map = np.zeros((self.h, self.w))
            cv2.fillPoly(room_filled_map, approx, color=1.)

            room_filled_map = torch.tensor(room_filled_map[:,:], dtype=torch.float32, device=self.device)

            return room_filled_map
        else:
            approx_tensor = torch.tensor(approx, device=self.device)
            return approx_tensor

    def generate_input_dict_from_room_props(self, room_prop_list, score_function, use_thresh=False):
        """

        :param room_prop_list:
        :type room_prop_list: list of FloorPlanRoomProp
        :param score_function:
        :return:
        """

        if score_function == "room_maskrcnn_iou":
            inputs = self.generate_input_dict_for_room_maskrcnn_iou(room_prop_list)
        elif score_function == "room_iou":
            inputs = self.generate_input_dict_for_room_iou(room_prop_list, use_thresh=use_thresh)
        else:
            assert "generate_input_dict_from_room_props for %s not implemented" % score_function

        return inputs








