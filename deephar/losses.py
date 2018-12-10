from keras import backend as K
from keras.losses import binary_crossentropy

import tensorflow as tf


def _reset_invalid_joints(y_true, y_pred):
    """Reset (set to zero) invalid joints, according to y_true, and compute the
    number of valid joints.
    """
    idx = K.cast(K.greater(y_true, 0.), 'float32')
    y_true = idx * y_true
    y_pred = idx * y_pred
    num_joints = K.clip(K.sum(idx, axis=(-1, -2)), 1, None)
    return y_true, y_pred, num_joints


def elasticnet_loss_on_valid_joints(y_true, y_pred):
    y_true, y_pred, num_joints = _reset_invalid_joints(y_true, y_pred)
    l1 = K.sum(K.abs(y_pred - y_true), axis=(-1, -2)) / num_joints
    l2 = K.sum(K.square(y_pred - y_true), axis=(-1, -2)) / num_joints
    return l1 + l2


def elasticnet_bincross_loss_on_valid_joints(y_true, y_pred):
    idx = K.cast(K.greater(y_true, 0.), 'float32')
    num_joints = K.clip(K.sum(idx, axis=(-1, -2)), 1, None)

    l1 = K.abs(y_pred - y_true)
    l2 = K.square(y_pred - y_true)
    bc = 0.01*K.binary_crossentropy(y_true, y_pred)
    dummy = 0. * y_pred

    return K.sum(tf.where(K.cast(idx, 'bool'), l1 + l2 + bc, dummy),
            axis=(-1, -2)) / num_joints


def l1_loss_on_valid_joints(y_true, y_pred):
    y_true, y_pred, num_joints = _reset_invalid_joints(y_true, y_pred)
    return K.sum(K.abs(y_pred - y_true), axis=(-1, -2)) / num_joints


def l2_loss_on_valid_joints(y_true, y_pred):
    y_true, y_pred, num_joints = _reset_invalid_joints(y_true, y_pred)
    return K.sum(K.square(y_pred - y_true), axis=(-1, -2)) / num_joints


def pose_regression_loss(pose_loss, visibility_weight):

    def _pose_regression_loss(y_true, y_pred):
        video_clip = K.ndim(y_true) == 4
        if video_clip:
            """The model was time-distributed, so there is one additional
            dimension.
            """
            p_true = y_true[:, :, :, 0:-1]
            p_pred = y_pred[:, :, :, 0:-1]
            v_true = y_true[:, :, :, -1]
            v_pred = y_pred[:, :, :, -1]
        else:
            p_true = y_true[:, :, 0:-1]
            p_pred = y_pred[:, :, 0:-1]
            v_true = y_true[:, :, -1]
            v_pred = y_pred[:, :, -1]

        if pose_loss == 'l1l2':
            ploss = elasticnet_loss_on_valid_joints(p_true, p_pred)
        elif pose_loss == 'l1':
            ploss = l1_loss_on_valid_joints(p_true, p_pred)
        elif pose_loss == 'l2':
            ploss = l2_loss_on_valid_joints(p_true, p_pred)
        elif pose_loss == 'l1l2bincross':
            ploss = elasticnet_bincross_loss_on_valid_joints(p_true, p_pred)
        else:
            raise Exception('Invalid pose_loss option ({})'.format(pose_loss))

        vloss = binary_crossentropy(v_true, v_pred)

        if video_clip:
            """If time-distributed, average the error on video frames."""
            vloss = K.mean(vloss, axis=-1)
            ploss = K.mean(ploss, axis=-1)

        return ploss + visibility_weight*vloss

    return _pose_regression_loss

