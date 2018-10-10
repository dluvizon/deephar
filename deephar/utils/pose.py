import numpy as np

class _pa16j():
    """Pose alternated with 16 joints (like Penn Action with three more
    joints on the spine.
    """
    num_joints = 16
    joint_names = ['pelvis', 'thorax', 'neck', 'head',
            'r_shoul', 'l_shoul', 'r_elb', 'l_elb', 'r_wrist', 'l_wrist',
            'r_hip', 'l_hip', 'r_knww', 'l_knee', 'r_ankle', 'l_ankle']

    """Horizontal flip mapping"""
    map_hflip = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]

    """Projections from other layouts to the PA16J standard"""
    map_from_mpii = [6, 7, 8, 9, 12, 13, 11, 14, 10, 15, 2, 3, 1, 4, 0, 5]
    map_from_ntu = [0, 20, 2, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]

    """Projections of PA16J to other formats"""
    map_to_pa13j = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    map_to_jhmdb = [2, 1, 3, 4, 5, 10, 11, 6, 7, 12, 13, 8, 9, 14, 15]
    map_to_mpii = [14, 12, 10, 11, 13, 15, 0, 1, 2, 3, 8, 6, 4, 5, 7, 9]
    map_to_lsp = [14, 12, 10, 11, 13, 15, 8, 6, 4, 5, 7, 9, 2, 3]

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm']
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
    links = [[0, 1], [1, 2], [2, 3], [4, 6], [6, 8], [5, 7], [7, 9],
            [10, 12], [12, 14], [11, 13], [13, 15]]

class _pa17j():
    """Pose alternated with 17 joints (like _pa16j, with the middle spine).
    """
    num_joints = 17

    """Horizontal flip mapping"""
    map_hflip = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 16]

    """Projections from other layouts to the PA17J standard"""
    map_from_h36m = \
            [0, 12, 13, 15, 25, 17, 26, 18, 27, 19, 1, 6, 2, 7, 3, 8, 11]
    map_from_ntu = _pa16j.map_from_ntu + [1]
    map_from_mpii3dhp = \
            [4, 5, 6, 7, 14, 9, 15, 10, 16, 11, 23, 18, 24, 19, 25, 20, 3]
    map_from_mpii3dhp_te = \
            [14, 1, 16, 0, 2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13, 15]

    """Projections of PA17J to other formats"""
    map_to_pa13j = _pa16j.map_to_pa13j
    map_to_mpii = [14, 12, 10, 11, 13, 15, 0, 1, 2, 3, 8, 6, 4, 5, 7, 9]
    map_to_pa16j = list(range(16))

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm']
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 0]
    links = [[0, 16], [16, 1], [1, 2], [2, 3], [4, 6], [6, 8], [5, 7], [7, 9],
            [10, 12], [12, 14], [11, 13], [13, 15]]

class _pa20j():
    """Pose alternated with 20 joints. Similar to _pa16j, but with one more
    joint for hands and feet.
    """
    num_joints = 20

    """Horizontal flip mapping"""
    map_hflip = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16,
            19, 18]

    """Projections from other layouts to the PA20J standard"""
    map_from_h36m = [0, 12, 13, 15, 25, 17, 26, 18, 27, 19, 30, 22, 1, 6, 2,
            7, 3, 8, 4, 9]
    map_from_ntu = [0, 20, 2, 3, 4, 8, 5, 9, 6, 10, 7, 11, 12, 16, 13, 17, 14,
            18, 15, 19]

    """Projections of PA20J to other formats"""
    map_to_mpii = [16, 14, 12, 13, 15, 17, 0, 1, 2, 3, 8, 6, 4, 5, 7, 9]
    map_to_pa13j = [3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]
    map_to_pa16j = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm']
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4]
    links = [[0, 1], [1, 2], [2, 3], [4, 6], [6, 8], [8, 10], [5, 7], [7, 9],
            [9, 11], [12, 14], [14, 16], [16, 18], [13, 15], [15, 17], [17, 19]]

class _pa21j():
    """Pose alternated with 21 joints. Similar to _pa20j, but with one more
    joint referent to the 16th joint from _pa17j, for compatibility with H36M.
    """
    num_joints = 21

    """Horizontal flip mapping"""
    map_hflip = _pa20j.map_hflip + [20]

    """Projections from other layouts to the PA21J standard"""
    map_from_h36m = _pa20j.map_from_h36m + [11]
    map_from_ntu = _pa20j.map_from_ntu + [1]

    """Projections of PA20J to other formats"""
    map_to_mpii = _pa20j.map_to_mpii
    map_to_pa13j = _pa20j.map_to_pa13j
    map_to_pa16j = _pa20j.map_to_pa16j
    map_to_pa17j = _pa20j.map_to_pa16j + [20]

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm']
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 0]
    links = [[0, 20], [20, 1], [1, 2], [2, 3], [4, 6], [6, 8], [8, 10], [5, 7], [7, 9],
            [9, 11], [12, 14], [14, 16], [16, 18], [13, 15], [15, 17], [17, 19]]

class coco17j():
    """Original layout for the MS COCO dataset."""
    num_joints = 17
    dim = 2

    """Horizontal flip mapping"""
    map_hflip = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm', 'w']
    cmap = [0, 0, 0, 5, 5, 0, 0, 2, 1, 2, 1, 0, 0, 4, 3, 4, 3]
    links = [[13, 15], [13, 11], [14, 16], [14, 12], [11, 12], [5, 11], [6,
        12], [5, 6], [7, 5], [8, 6], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
        [3, 1], [4, 2], [3, 5], [4, 6]]


class pa16j2d(_pa16j):
    dim = 2

class pa16j3d(_pa16j):
    dim = 3

class pa17j2d(_pa17j):
    dim = 2

class pa17j3d(_pa17j):
    dim = 3

class pa20j3d(_pa20j):
    dim = 3

class pa21j3d(_pa21j):
    dim = 3

class ntu25j3d():
    num_joints = 25
    dim = 3


def _func_and(x):
    if x.all():
        return 1
    return 0

def get_visible_joints(x, margin=0.0):

    visible = np.apply_along_axis(_func_and, axis=1, arr=(x > margin))
    visible *= np.apply_along_axis(_func_and, axis=1, arr=(x < 1 - margin))

    return visible

def get_valid_joints(x):
    return np.apply_along_axis(_func_and, axis=1, arr=(x > -1e6))

def convert_pa17j3d_to_pa16j(p, dim=3):
    assert p.shape == (pa17j3d.num_joints, pa17j3d.dim)
    return p[pa17j3d.map_to_pa16j,0:dim].copy()

def convert_sequence_pa17j3d_to_pa16j(seqp, dim=3):
    assert seqp.shape[1:] == (pa17j3d.num_joints, pa17j3d.dim)
    x = np.zeros((len(seqp), _pa16j.num_joints, dim))
    for i in range(len(seqp)):
        x[i,:] = convert_pa17j3d_to_pa16j(seqp[i], dim=dim)
    return x

def write_poselist(filename, poses):
    """ Write a pose list to a text file.
    In the text file, every row corresponds to one pose and the columns are:
    {x1, y1, x2, y2, ...}

        Inputs: 'filename'
                'poses' [nb_samples, nb_joints, 2]
    """
    nb_samples, nb_joints, dim = poses.shape
    x = poses.copy()
    x = np.reshape(x, (nb_samples, nb_joints * dim))
    np.savetxt(filename, x, fmt='%.6f', delimiter=',')

def assign_knn_confidence(c, num_iter=2):
    assert c.ndim == 2 and c.shape[1] == 1, \
            'Invalid confidence shape {}'.format(c.shape)

    def _search_knn(refp):
        cs = c[list(refp), 0]
        if np.isnan(cs).all():
            return np.nan
        if np.nanmean(cs) < 0.5:
            return 0.1
        return 0.9

    for _ in range(num_iter):
        for i in range(len(c)):
            if np.isnan(c[i, 0]):
                c[i, 0] = _search_knn(dsl80j3d.neighbors[i])

