import os

import numpy as np
import operator
import json

from keras.callbacks import Callback

from deephar.measures import mean_distance_error
from deephar.utils import *

def eval_human36m_sc_error(model, x, pose_w, afmat, rootz, scam, action,
        resol_z=2000., batch_size=8, map_to_pa17j=None, logdir=None,
        verbose=True):

    from deephar.data.human36m import ACTION_LABELS

    assert len(x) == len(pose_w) == len(afmat) == len(scam) == len(action)

    input_shape = model.input_shape
    if len(input_shape) == 5:
        """Video clip processing."""
        num_frames = input_shape[1]
        num_batches = int(len(x) / num_frames)

        x = x[0:num_batches*num_frames]
        x = np.reshape(x, (num_batches, num_frames,) + x.shape[1:])

        pose_w = pose_w[0:num_batches*num_frames]
        afmat = afmat[0:num_batches*num_frames]
        rootz = rootz[0:num_batches*num_frames]
        scam = scam[0:num_batches*num_frames]
        action = action[0:num_batches*num_frames]

    num_blocks = len(model.outputs)
    num_spl = len(x)
    cams = []

    y_true_w = pose_w.copy()
    if map_to_pa17j is not None:
        y_true_w = y_true_w[:, map_to_pa17j, :]
    y_pred_w = np.zeros((num_blocks,) + y_true_w.shape)
    if rootz.ndim == 1:
        rootz = np.expand_dims(rootz, axis=-1)

    pred = model.predict(x, batch_size=batch_size, verbose=1)

    """Move the root joints from g.t. poses to origin."""
    y_true_w -= y_true_w[:,0:1,:]

    if verbose:
        printc(WARNING, 'Avg. mm. error:')

    lower_err = np.inf
    lower_i = -1
    scores = []

    for b in range(num_blocks):

        if num_blocks > 1:
            y_pred = pred[b]
        else:
            y_pred = pred

        if len(input_shape) == 5:
            """Remove the temporal dimension."""
            y_pred = y_pred[:, :, :, 0:3]
            y_pred = np.reshape(y_pred, (-1, y_pred.shape[2], y_pred.shape[3]))
        else:
            y_pred = y_pred[:, :, 0:3]

        """Project normalized coordiates to the image plane."""
        y_pred[:, :, 0:2] = transform_pose_sequence(
            afmat.copy(), y_pred[:, :, 0:2], inverse=True)

        """Recover the absolute Z."""
        y_pred[:, :, 2] = (resol_z * (y_pred[:, :, 2] - 0.5)) + rootz
        if map_to_pa17j is not None:
            y_pred_uvd = y_pred[:, map_to_pa17j, 0:3]
        else:
            y_pred_uvd = y_pred[:, :, 0:3]

        """Do the camera inverse projection."""
        for j in range(len(y_pred_uvd)):
            cam = camera_deserialize(scam[j])
            y_pred_w[b, j, :, :] = cam.inverse_project(y_pred_uvd[j])

        """Move the root joint from predicted poses to the origin."""
        y_pred_w[b, :, :, :] -= y_pred_w[b, :, 0:1, :]

        err_w = mean_distance_error(y_true_w[:, 0:, :], y_pred_w[b, :, 0:, :])
        scores.append(err_w)
        if verbose:
            printc(WARNING, ' %.1f' % err_w)

        """Keep the best prediction and its index."""
        if err_w < lower_err:
            lower_err = err_w
            lower_i = b

    if verbose:
        printcn('', '')

    if logdir is not None:
        np.save('%s/y_pred_w.npy' % logdir, y_pred_w)
        np.save('%s/y_true_w.npy' % logdir, y_true_w)

    """Select only the best prediction."""
    y_pred_w = y_pred_w[lower_i]

    """Compute error per action."""
    num_act = len(ACTION_LABELS)
    y_pred_act = {}
    y_true_act = {}
    for i in range(num_act):
        y_pred_act[i] = None
        y_true_act[i] = None

    act = lambda x: action[x, 0]
    for i in range(len(y_pred_w)):
        if y_pred_act[act(i)] is None:
            y_pred_act[act(i)] = y_pred_w[i:i+1]
            y_true_act[act(i)] = y_true_w[i:i+1]
        else:
            y_pred_act[act(i)] = np.concatenate(
                    [y_pred_act[act(i)], y_pred_w[i:i+1]], axis=0)
            y_true_act[act(i)] = np.concatenate(
                    [y_true_act[act(i)], y_true_w[i:i+1]], axis=0)

    for i in range(num_act):
        if y_pred_act[i] is None:
            continue
        err = mean_distance_error(y_true_act[i][:,0:,:], y_pred_act[i][:,0:,:])
        printcn(OKBLUE, '%s: %.1f' % (ACTION_LABELS[i], err))

    printcn(WARNING, 'Final averaged error (mm): %.3f' % lower_err)

    return scores


class H36MEvalCallback(Callback):

    def __init__(self, x, pw, afmat, rootz, scam, action, batch_size=24,
            eval_model=None, map_to_pa17j=None, logdir=None):

        self.x = x
        self.pw = pw
        self.afmat = afmat
        self.rootz = rootz
        self.scam = scam
        self.action = action
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.map_to_pa17j = map_to_pa17j
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        scores = eval_human36m_sc_error(model, self.x, self.pw, self.afmat,
                self.rootz, self.scam, self.action, batch_size=self.batch_size,
                map_to_pa17j=self.map_to_pa17j)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'h36m_val.json'), 'w') as f:
                json.dump(self.logarray, f)

        cur_best = min(scores)
        self.scores[epoch] = cur_best

        printcn(OKBLUE, 'Best score is %.1f at epoch %d' % \
                (self.best_score, self.best_epoch))


    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the minimum value from a dict
            return min(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the minimum value from a dict
            return self.scores[self.best_epoch]
        else:
            return np.inf

