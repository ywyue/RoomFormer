from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

class MCSSOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MCSSFloor options")

        # PATHS
        self.parser.add_argument("--mcts_path",
                                 type=str,
                                 help="the name of the MonteFloorNet model",
                                 default="/media/sinisa/Sinisa_hdd_data/Sinisa_Projects/corridor_localisation/experiments/MonteFloorNet_experiments/room_shape_experiments/Structured3D_test/")


        self.parser.add_argument("--dataset_path",
                                 type=str,
                                 help="the name of the MonteFloorNet model",
                                 default="s3d_floorplan_eval/montefloor_data")

        self.parser.add_argument("--dataset_type",
                                 type=str,
                                 help="the name of the dataset type",
                                 choices=["floornet", "s3d", "fsp"],
                                 default="s3d")
        self.parser.add_argument("--scene_id",
                                 type=str,
                                 help="the name of the scene",
                                 default="val")

        self.parser.add_argument("--min_scene_ind",
                                 type=int,
                                 help="the name of the scene",
                                 default=0)
        self.parser.add_argument("--max_scene_ind",
                                 type=int,
                                 help="the name of the scene",
                                 default=251)

        # MonteFloorNet options
        # self.parser.add_argument("--model_S_path",
        #                          help="the name of the MonteFloorNet model",
        #                          default="/home/sinisa/tmp/current_experiments/montefloornet_S_model_camera_ready16/best_models/weights_935")


        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=256)

    def parse(self):
        self.options, unknown = self.parser.parse_known_args()
        return self.options
