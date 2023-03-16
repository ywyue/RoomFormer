import os
import numpy as np
import cv2
from planar_graph_utils import get_regions_from_pg, plot_floorplan_with_regions

# example_pg = {
#     (127, 20): [(20, 120), (234, 120)],
#     (20, 120): [(127, 20), (234, 120), (20, 240)],
#     (234, 120): [(127, 20), (20, 120), (234, 240)],
#     (20, 240): [(20, 120), (234, 240)],
#     (234, 240): [(234, 120), (20, 240)],
# }

# pg_base = '../results/npy_heat_s3d_256/'
pg_base = '../results/test_gt/'
viz_base = './viz_gt'
if not os.path.exists(viz_base):
    os.makedirs(viz_base)

for filename in sorted(os.listdir(pg_base)):
    pg_path = os.path.join(pg_base, filename)
    example_pg = np.load(pg_path, allow_pickle=True).tolist()

    corners = example_pg['corners']
    corners = corners.astype(np.int)
    edges = example_pg['edges']

    print('Processing file: {}'.format(filename))
    regions = get_regions_from_pg(example_pg, corner_sorted=True)
    print('num of extracted regions {}'.format(len(regions)))
    floorplan_image = plot_floorplan_with_regions(regions, corners, edges, scale=1000)
    cv2.imwrite(os.path.join(viz_base, '{}.png'.format(filename[:-4])), floorplan_image)
