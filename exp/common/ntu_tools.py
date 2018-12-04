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
        if verbose:
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
