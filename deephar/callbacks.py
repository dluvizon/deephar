import os

import json
import numpy as np
from keras.callbacks import Callback

from .utils import *

class SaveModel(Callback):

    def __init__(self, filepath, model_to_save=None, save_best_only=False,
            callback_to_monitor=None, verbose=1):

        if save_best_only and callback_to_monitor is None:
            warning('Cannot save the best model with no callback monitor')

        self.filepath = filepath
        self.model_to_save = model_to_save
        self.save_best_only = save_best_only
        self.callback_to_monitor = callback_to_monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if self.model_to_save is not None:
            model = self.model_to_save
        else:
            model = self.model

        filename = self.filepath.format(epoch=epoch + 1)

        if self.best_epoch == epoch + 1 or not self.save_best_only:
            if self.verbose:
                printnl('Saving model @epoch=%05d to %s' \
                        % (epoch + 1, filename))
            model.save_weights(filename)

    @property
    def best_epoch(self):
        if self.callback_to_monitor is not None:
            return self.callback_to_monitor.best_epoch
        else:
            return None

