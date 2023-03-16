import numpy as np


def rotate_poly(poly, angle, flip_h):
    """
    Rotate poly

    :param poly:
    :return:
    """

    px, py = poly[:, 0], poly[:, 1]

    angle_rad = angle * np.pi / 180

    qx = np.cos(angle_rad) * px - np.sin(angle_rad) * py
    qy = np.sin(angle_rad) * px + np.cos(angle_rad) * py

    if flip_h:
        qx = -qx

    rotated_poly = np.zeros_like(poly)
    rotated_poly[:, 0] = qx
    rotated_poly[:, 1] = qy

    # print("p",  poly)
    # print("r", rotated_poly)

    return rotated_poly