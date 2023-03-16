import copy
import functools
import numpy as np
import os

from Evaluator.Evaluator import Evaluator
from options import MCSSOptions
from DataRW.S3DRW import S3DRW
from DataRW.wrong_annotatios import wrong_s3d_annotations_list
from planar_graph_utils import get_regions_from_pg


room_polys_def = [np.array([[191, 150],
       [191,  70],
       [222,  70],
       [222, 150],
       [191, 150]]), np.array([[232,  65],
       [232,  11],
       [202,  11],
       [202,  65],
       [232,  65]]), np.array([[ 47,  50],
       [ 47, 150],
       [ 24, 150],
       [ 24,  50],
       [ 47,  50]]), np.array([[199, 156],
       [199, 234],
       [146, 234],
       [146, 156],
       [199, 156]]), np.array([[109, 184],
       [120, 184],
       [120, 156],
       [ 50, 156],
       [ 50, 234],
       [109, 234],
       [109, 184]]), np.array([[110, 234],
       [144, 234],
       [144, 187],
       [110, 187],
       [110, 234]]), np.array([[ 50,  50],
       [ 50, 150],
       [123, 150],
       [123, 184],
       [144, 184],
       [144, 150],
       [190, 150],
       [190,  70],
       [108,  70],
       [108,  50],
       [ 50,  50]])]

# pg_base = 'results/npy_heat_s3d_256/'
pg_base = 'results/test_gt/'
# pg_base = 'results/test_eval2/'

options = MCSSOptions()
opts = options.parse()

if __name__ == '__main__':

    # data_rw = FloorNetRW(opts)

    if opts.scene_id == "val":

        opts.scene_id = "scene_03250" # Temp. value
        data_rw = S3DRW(opts)
        scene_list = data_rw.loader.scenes_list

        quant_result_dict = None
        quant_result_maskrcnn_dict = None
        scene_counter = 0
        for scene_ind, scene in enumerate(scene_list):
            if int(scene[6:]) in wrong_s3d_annotations_list:
                continue

            print("------------")
            curr_opts = copy.deepcopy(opts)
            curr_opts.scene_id = scene
            curr_data_rw = S3DRW(curr_opts)
            print("Running Evaluation for scene %s" % scene)

            evaluator = Evaluator(curr_data_rw, curr_opts)

            # TODO load your room polygons into room_polys, list of polygons (n x 2)
            # room_polys = np.array([[[0,0], [200, 0], [200, 200]]]) # Placeholder

            pg_path = os.path.join(pg_base, scene[6:] + '.npy')
            example_pg = np.load(pg_path, allow_pickle=True).tolist()
            example_pg['corners'] = example_pg['corners'][:8]
            example_pg['edges'] = example_pg['edges'][:8]
            regions = get_regions_from_pg(example_pg, corner_sorted=True)
            # regions = [np.array(re, dtype=np.int32) for re in example_pg]
            room_polys = regions
            # room_polys = room_polys_def # Placeholder


            quant_result_dict_scene =\
                evaluator.evaluate_scene(room_polys=room_polys)

            if quant_result_dict is None:
                quant_result_dict = quant_result_dict_scene
            else:
                for k in quant_result_dict.keys():
                    quant_result_dict[k] += quant_result_dict_scene[k]

            scene_counter += 1

            # break

        for k in quant_result_dict.keys():
            quant_result_dict[k] /= float(scene_counter)

        print("Our: ", quant_result_dict)

        print("Ours")
        evaluator.print_res_str_for_latex(quant_result_dict)
