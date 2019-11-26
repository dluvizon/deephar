import os

import numpy as np
import json
import time

from keras.callbacks import Callback

from deephar.data import BatchLoader
from deephar.utils import *


def eval_singleclip_gt_bbox_generator(model, datagen, verbose=1):

    num_blocks = len(model.outputs)
    num_samples = len(datagen)

    start = time.time()
    for i in range(num_samples):
        if verbose > 1:
            printcn('', 'pred %05d/%05d' % (i+1, num_samples))

        [x], [y] = datagen[i]
        if 'y_true' not in locals():
            y_true = np.zeros((num_samples,) + y.shape[1:])
            y_pred = np.zeros((num_samples, num_blocks) + y.shape[1:])

        y_true[i, :] = y
        pred = model.predict(x)
        for b in range(num_blocks):
            y_pred[i, b, :] = pred[b]

    dt = time.time() - start

    if verbose:
        printc(WARNING, 'NTU, single-clip, GT bbox, action acc.%:')

    scores = []
    for b in range(num_blocks):
        correct = np.equal(np.argmax(y_true, axis=-1),
                np.argmax(y_pred[:, b, :], axis=-1), dtype=np.float)
        scores.append(sum(correct) / len(correct))
        if verbose:
            printc(WARNING, ' %.1f ' % (100*scores[-1]))

    if verbose:
        printcn('', '\n%d samples in %.1f sec: %.1f clips per sec' \
                % (num_samples, dt, num_samples / dt))

    return scores


def eval_multiclip_dataset(model, ntu, subsampling, bboxes_file=None,
        logdir=None, verbose=1):
    """If bboxes_file if not given, use ground truth bounding boxes."""

    num_samples = ntu.get_length(TEST_MODE)
    num_blocks = len(model.outputs)

    """Save and reset some original configs from the dataset."""
    org_hflip = ntu.dataconf.fixed_hflip
    org_use_gt_bbox = ntu.use_gt_bbox

    cnt_corr = 0
    cnt_total = 0

    action_shape = (num_samples,) + ntu.get_shape('ntuaction')
    a_true = np.zeros(action_shape)
    a_pred = np.ones((num_blocks,) + action_shape)
    missing_clips = {}

    if bboxes_file is not None:
        with open(bboxes_file, 'r') as fid:
            bboxes_data = json.load(fid)
        ntu.use_gt_bbox = False
        bboxes_info = 'Using bounding boxes from file "{}"'.format(bboxes_file)
    else:
        bboxes_data = None
        ntu.use_gt_bbox = True
        bboxes_info = 'Using ground truth bounding boxes.'

    for i in range(num_samples):
        if verbose:
            printc(OKBLUE, '%04d/%04d\t' % (i, num_samples))

        frame_list = ntu.get_clip_index(i, TEST_MODE, subsamples=[subsampling])

        """Variable to hold all preditions for this sequence.
        2x frame_list due to hflip.
        """
        allpred = np.ones((num_blocks, 2*len(frame_list)) + action_shape[1:])

        for f in range(len(frame_list)):
            for hflip in range(2):
                preds_clip = []
                try:
                    ntu.dataconf.fixed_hflip = hflip # Force horizontal flip

                    bbox = None
                    if bboxes_data is not None:
                        key = '%04d.%d.%03d.%d' % (i, subsampling, f, hflip)
                        try:
                            bbox = np.array(bboxes_data[key])
                        except:
                            warning('Missing bounding box key ' + str(key))

                    """Load clip and predict action."""
                    data = ntu.get_data(i, TEST_MODE, frame_list=frame_list[f],
                            bbox=bbox)
                    a_true[i, :] = data['ntuaction']

                    pred = model.predict(np.expand_dims(data['frame'], axis=0))
                    for b in range(num_blocks):
                        allpred[b, 2*f+hflip, :] = pred[b][0]
                        a_pred[b, i, :] *= pred[b][0]

                    if np.argmax(a_true[i]) != np.argmax(a_pred[-1, i]):
                        missing_clips['%04d.%03d.%d' % (i, f, hflip)] = [
                                int(np.argmax(a_true[i])),
                                int(np.argmax(a_pred[-1, i]))]

                except Exception as e:
                    warning('eval_multiclip, exception on sample ' \
                            + str(i) + ' frame ' + str(f) + ': ' + str(e))

        if verbose:
            cor = int(np.argmax(a_true[i]) == np.argmax(a_pred[-1, i]))

            cnt_total += 1
            cnt_corr += cor
            printnl('%d : %.1f' % (cor, 100 * cnt_corr / cnt_total))

    if logdir is not None:
        np.save('%s/a_pred.npy' % logdir, a_pred)
        np.save('%s/a_true.npy' % logdir, a_true)
        with open(os.path.join(logdir, 'missing-clips.json'), 'w') as fid:
            json.dump(missing_clips, fid)

    a_true = np.expand_dims(a_true, axis=0)
    a_true = np.tile(a_true, (num_blocks, 1, 1))
    correct = np.argmax(a_true, axis=-1) == np.argmax(a_pred, axis=-1)
    scores = 100*np.sum(correct, axis=-1) / num_samples
    if verbose:
        printcn(WARNING, 'NTU, multi-clip. ' + bboxes_info + '\n')
        printcn(WARNING, np.array2string(np.array(scores), precision=2))
        printcn(WARNING, 'NTU best: %.2f' % max(scores))

    ntu.dataconf.fixed_hflip = org_hflip
    ntu.use_gt_bbox = org_use_gt_bbox

    return scores


class NtuEvalCallback(Callback):

    def __init__(self, data, eval_model=None, logdir=None):

        assert type(data) == BatchLoader, \
                'data must be a BatchLoader instance, ' \
                + 'got {} instead'.format(data)

        self.data = data
        self.eval_model = eval_model
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        scores = eval_singleclip_gt_bbox_generator(model, self.data)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'ntu_val.json'), 'w') as f:
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

# Aliases.
eval_singleclip_generator = eval_singleclip_gt_bbox_generator
