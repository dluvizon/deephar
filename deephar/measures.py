import numpy as np

from deephar.utils import *


def _norm(x, axis=None):
    return np.sqrt(np.sum(np.power(x, 2), axis=axis))

def _valid_joints(y, min_valid=-1e6):
    def and_all(x):
        if x.all():
            return 1
        return 0

    return np.apply_along_axis(and_all, axis=1, arr=(y > min_valid))

def mean_distance_error(y_true, y_pred):
    """Compute the mean distance error on predicted samples, considering
    only the valid joints from y_true.

    # Arguments
        y_true: [num_samples, nb_joints, dim]
        y_pred: [num_samples, nb_joints, dim]

    # Return
        The mean absolute error on valid joints.
    """

    assert y_true.shape == y_pred.shape
    num_samples = len(y_true)

    dist = np.zeros(y_true.shape[0:2])
    valid = np.zeros(y_true.shape[0:2])

    for i in range(num_samples):
        valid[i,:] = _valid_joints(y_true[i])
        dist[i,:] = _norm(y_true[i] - y_pred[i], axis=1)

    match = dist * valid
    # print ('Maximum valid distance: {}'.format(match.max()))
    # print ('Average valid distance: {}'.format(match.mean()))

    return match.sum() / valid.sum()

def pckh(y_true, y_pred, head_size, refp=0.5):
    """Compute the PCKh measure (using refp of the head size) on predicted
    samples.

    # Arguments
        y_true: [num_samples, nb_joints, 2]
        y_pred: [num_samples, nb_joints, 2]
        head_size: [num_samples, 1]

    # Return
        The PCKh score.
    """

    assert y_true.shape == y_pred.shape
    assert len(y_true) == len(head_size)
    num_samples = len(y_true)

    # Ignore the joints 6 and 7 (pelvis and thorax respectively), according
    # to the file 'annolist2matrix.m'
    used_joints = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 8, 9]
    y_true = y_true[:, used_joints, :]
    y_pred = y_pred[:, used_joints, :]
    dist = np.zeros((num_samples, len(used_joints)))
    valid = np.zeros((num_samples, len(used_joints)))

    for i in range(num_samples):
        valid[i,:] = _valid_joints(y_true[i])
        dist[i,:] = _norm(y_true[i] - y_pred[i], axis=1) / head_size[i]
    match = (dist <= refp) * valid

    return match.sum() / valid.sum()


def pck3d(y_true, y_pred, refp=150):
    """Compute the PCK3D measure (using refp as the threshold) on predicted
    samples.

    # Arguments
        y_true: [num_samples, nb_joints, 3]
        y_pred: [num_samples, nb_joints, 3]

    # Return
        The PCKh score.
    """

    assert y_true.shape == y_pred.shape
    num_samples = len(y_true)

    # Ignore the joints 6 and 7 (pelvis and thorax respectively), according
    # to the file 'annolist2matrix.m'
    used_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    y_true = y_true[:, used_joints, :]
    y_pred = y_pred[:, used_joints, :]
    dist = np.zeros((num_samples, len(used_joints)))
    valid = np.zeros((num_samples, len(used_joints)))

    for i in range(num_samples):
        valid[i,:] = _valid_joints(y_true[i])
        dist[i,:] = _norm(y_true[i] - y_pred[i], axis=1)
    match = (dist <= refp) * valid

    return match.sum() / valid.sum()


def pckh_per_joint(y_true, y_pred, head_size, pose_layout, refp=0.5, verbose=1):
    """Compute the PCKh measure (using refp of the head size) on predicted
    samples per joint and output the results.

    # Arguments
        y_true: [num_samples, nb_joints, 2]
        y_pred: [num_samples, nb_joints, 2]
        head_size: [num_samples, 1]
        pose_layout: from deephar.utils.pose
    """

    assert y_true.shape == y_pred.shape
    assert len(y_true) == len(head_size)

    num_samples = len(y_true)
    num_joints = pose_layout.num_joints
    dist = np.zeros((num_samples, num_joints))
    valid = np.zeros((num_samples, num_joints))

    for i in range(num_samples):
        valid[i,:] = _valid_joints(y_true[i])
        dist[i,:] = _norm(y_true[i] - y_pred[i], axis=1) / head_size[i]

    for j in range(num_joints):
        jname = pose_layout.joint_names[j]
        space = 7*' '
        ss = len(space) - len(jname)
        if verbose:
            printc(HEADER, jname + space[0:ss] + '| ')
    if verbose:
        print ('')

    match = (dist <= refp) * valid
    for j in range(num_joints):
        pck = match[:, j].sum() / valid[:, j].sum()
        if verbose:
            printc(OKBLUE, ' %.2f | ' % (100 * pck))
    if verbose:
        print ('')


def pck_torso(y_true, y_pred, refp=0.2):
    """ Compute the PCK (using 0.2 of the torso size) on predicted samples.

        Input:  y_true [nb_samples, nb_joints, 2]
                y_pred [nb_samples, nb_joints, 2]

        Return: The PCK score [1]
    """
    assert y_true.shape == y_pred.shape
    nb_samples, _, nb_joints = y_true.shape

    dist = np.zeros((nb_samples, nb_joints))
    valid = np.zeros((nb_samples, nb_joints))
    torso = _norm(y_true[:,:,5] - y_true[:,:,10], axis=1)

    for i in range(nb_samples):
        valid[i,:] = _valid_joints(y_true[i])
        dist[i,:] = _norm(y_true[i] - y_pred[i], axis=0) / torso[i]
    match = (dist <= refp) * valid

    return match.sum() / valid.sum()

