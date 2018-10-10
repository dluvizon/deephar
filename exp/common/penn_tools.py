import os

import numpy as np
import json
import time

from keras.callbacks import Callback

from deephar.data import BatchLoader
from deephar.utils import *


def eval_singleclip_gt_bbox(model, x_te, action_te, batch_size=1, verbose=1):

    num_blocks = len(model.outputs)
    start = time.time()

    pred = model.predict(x_te, batch_size=batch_size, verbose=verbose)
    dt = time.time() - start

    if verbose:
        printc(WARNING, 'PennAction, single-clip, GT bbox, action acc.%:')

    scores = []
    for b in range(num_blocks):

        y_pred = pred[b]
        correct = np.equal(np.argmax(action_te, axis=-1),
                np.argmax(y_pred, axis=-1), dtype=np.float)
        scores.append(sum(correct) / len(correct))

        if verbose:
            printc(WARNING, ' %.1f' % (100*scores[-1]))

    if verbose:
        printcn('', '\n%d samples in %.1f sec: %.1f clips per sec' \
                % (len(x_te), dt, len(x_te) / dt))

    return scores


def eval_singleclip_gt_bbox_generator(model, datagen, verbose=1, logdir=None):

    num_blocks = len(model.outputs)
    num_samples = len(datagen)
    start = time.time()

    for i in range(num_samples):
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
        printc(WARNING, 'PennAction, single-clip, GT bbox, action acc.%:')

    if logdir is not None:
        logpath = os.path.join(logdir, 'single-clip')
        mkdir(logpath)

    scores = []
    for b in range(num_blocks):
        correct = np.equal(np.argmax(y_true, axis=-1),
                np.argmax(y_pred[:, b, :], axis=-1), dtype=np.float)
        scores.append(sum(correct) / len(correct))
        if verbose:
            printc(WARNING, ' %.1f ' % (100*scores[-1]))

        if logdir is not None:
            np.save(logpath + '/%02d.npy' % b, correct)

    if verbose:
        printcn('', '\n%d samples in %.1f sec: %.1f clips per sec' \
                % (num_samples, dt, num_samples / dt))

    return scores


class PennActionEvalCallback(Callback):

    def __init__(self, data, batch_size=1, eval_model=None,
            logdir=None):

        self.data = data
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        if type(self.data) == BatchLoader:
            scores = eval_singleclip_gt_bbox_generator(model, self.data)
        else:
            scores = eval_singleclip_gt_bbox(model, self.data[0],
                    self.data[1], batch_size=self.batch_size)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'penn_val.json'), 'w') as f:
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
eval_singleclip = eval_singleclip_gt_bbox
eval_singleclip_generator = eval_singleclip_gt_bbox_generator

