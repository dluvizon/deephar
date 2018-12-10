import os

import numpy as np
import json

from keras.callbacks import Callback

from deephar.measures import pckh
from deephar.measures import pckh_per_joint
from deephar.utils import *


def refine_pred(model, frames, afmat, bbox, ds, mode, outidx,
        num_iter=2,
        winsize_scale=1.50,
        momentum=0.8,
        batch_size=8):

    pred_out = []
    refined_bbox = []

    for t in range(num_iter):

        ds.set_custom_bboxes(mode, refined_bbox)
        pred = model.predict(frames, batch_size=batch_size, verbose=1)[outidx]

        A = afmat[:]
        curr_bbox = bbox[:]
        if len(refined_bbox) == 0:
            refined_bbox = curr_bbox.copy()

        pred = transform_pose_sequence(A.copy(), pred, inverse=True)
        pred_out.append(pred)
        if t == num_iter - 1:
            break

        for i in range(len(pred)):
            x1 = np.min(pred[i, :, 0])
            y1 = np.min(pred[i, :, 1])
            x2 = np.max(pred[i, :, 0])
            y2 = np.max(pred[i, :, 1])
            objpos_p = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            wsize_p = winsize_scale * max(x2 - x1, y2 - y1)

            objpos_t, winsize_t = bbox_to_objposwin(curr_bbox[i])
            objpos_p = momentum*objpos_t + (1 - momentum)*objpos_p

            refined_bbox[i,:] = objposwin_to_bbox(objpos_p, (wsize_p, wsize_p))

    ds.clear_custom_bboxes(mode)

    return pred_out

def absulute_pred(model, frames, afmat, outidx, batch_size=8):

    pred = model.predict(frames, batch_size=batch_size, verbose=1)[outidx]
    A = afmat[:]
    pred = transform_pose_sequence(A.copy(), pred, inverse=True)

    return pred


def eval_singleperson_pckh(model, fval, pval, afmat_val, headsize_val,
        win=None, batch_size=8, refp=0.5, map_to_pa16j=None, pred_per_block=1,
        verbose=1):

    input_shape = model.get_input_shape_at(0)
    if len(input_shape) == 5:
        """Video clip processing."""
        num_frames = input_shape[1]
        num_batches = int(len(fval) / num_frames)

        fval = fval[0:num_batches*num_frames]
        fval = np.reshape(fval, (num_batches, num_frames,) + fval.shape[1:])

        pval = pval[0:num_batches*num_frames]
        afmat_val = afmat_val[0:num_batches*num_frames]
        headsize_val = headsize_val[0:num_batches*num_frames]

    num_blocks = int(len(model.outputs) / pred_per_block)
    inputs = [fval]
    if win is not None:
        num_blocks -= 1
        inputs.append(win)

    pred = model.predict(inputs, batch_size=batch_size, verbose=1)
    if win is not None:
        del pred[0]

    A = afmat_val[:]
    y_true = pval[:]

    y_true = transform_pose_sequence(A.copy(), y_true, inverse=True)
    if map_to_pa16j is not None:
        y_true = y_true[:, map_to_pa16j, :]
    scores = []
    if verbose:
        printc(WARNING, 'PCKh on validation:')

    for b in range(num_blocks):

        if num_blocks > 1:
            y_pred = pred[pred_per_block*b]
        else:
            y_pred = pred

        if len(input_shape) == 5:
            """Remove the temporal dimension."""
            y_pred = y_pred[:, :, :, 0:2]
            y_pred = np.reshape(y_pred, (-1, y_pred.shape[2], y_pred.shape[3]))
        else:
            y_pred = y_pred[:, :, 0:2]

        if map_to_pa16j is not None:
            y_pred = y_pred[:, map_to_pa16j, :]

        y_pred = transform_pose_sequence(A.copy(), y_pred, inverse=True)
        s = pckh(y_true, y_pred, headsize_val, refp=refp)
        if verbose:
            printc(WARNING, ' %.1f' % (100*s))
        scores.append(s)

        if b == num_blocks-1:
            if verbose:
                printcn('', '')
            pckh_per_joint(y_true, y_pred, headsize_val, pa16j2d,
                    verbose=verbose)

    return scores


class MpiiEvalCallback(Callback):

    def __init__(self, fval, pval, afmat_val, headsize_val,
            win=None, batch_size=16, eval_model=None, map_to_pa16j=None,
            pred_per_block=1, logdir=None):

        self.fval = fval
        self.pval = pval[:, :, 0:2]
        self.afmat_val = afmat_val
        self.headsize_val = headsize_val
        self.win = win
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.map_to_pa16j = map_to_pa16j
        self.pred_per_block = pred_per_block
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        scores = eval_singleperson_pckh(model, self.fval, self.pval,
                self.afmat_val, self.headsize_val, win=self.win,
                batch_size=self.batch_size, map_to_pa16j=self.map_to_pa16j,
                pred_per_block=self.pred_per_block)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'mpii_val.json'), 'w') as f:
                json.dump(self.logarray, f)

        cur_best = max(scores)
        self.scores[epoch] = cur_best

        printcn(OKBLUE, 'Best score is %.1f at epoch %d' % \
                (100*self.best_score, self.best_epoch))


    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the maximum value from a dict
            return max(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the maximum value from a dict
            return self.scores[self.best_epoch]
        else:
            return 0

