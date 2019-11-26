import os

import numpy as np

from deephar.utils import *

def get_bbox_from_poses(poses, afmat, scale=1.5):
    if poses.ndim == 3:
        p = poses[:, :, 0:2]
        c = poses[:, :, -2:-1] > 0.25
    elif poses.ndim == 4:
        p = poses[0, :, :, 0:2]
        c = poses[0, :, :, -2:-1] > 0.25
    else:
        raise ValueError('Invalid poses shape {}'.format(poses.shape))

    baux = get_valid_bbox_array(p, jprob=c, relsize=scale)
    baux = np.array([min(baux[:,0]), min(baux[:,1]),
        max(baux[:,2]), max(baux[:,3])])
    baux = np.reshape(baux, (2, 2))
    baux = transform_2d_points(afmat, baux, transpose=True,
            inverse=True)
    baux = np.reshape(baux, (4,))
    bbox = np.array([min(baux[0], baux[2]), min(baux[1], baux[3]),
        max(baux[0], baux[2]), max(baux[1], baux[3])])

    return bbox

