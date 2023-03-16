import os
import numpy as np
import cv2




source = '../Structured3D/montefloor_data/test/'
dst = './viz_density'

for dirname in sorted(os.listdir(source)):
    density_path = os.path.join(source, dirname, 'density.png')
    density_img = cv2.imread(density_path)
    density = 255 - density_img
    out_path = os.path.join(dst, dirname + '.png')
    out_alpha_path = os.path.join(dst, dirname + '_alpha.png')
    alphas = np.zeros([density.shape[0], density.shape[1], 1], dtype=np.int32)
    alphas[density_img.sum(axis=-1) > 0] = 255
    density_alpha = np.concatenate([density, alphas], axis=-1)
    cv2.imwrite(out_path, density)
    cv2.imwrite(out_alpha_path, density_alpha)