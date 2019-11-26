import os

import numpy as np

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ProgbarLogger

from keras.utils import Sequence
from keras.utils import OrderedEnqueuer

from deephar.data import BatchLoader
from deephar.utils import *


class TrainerOnGenerator(object):
    """This class basically is a wrapper to the method 'fit_generator' from
    Keras, despite that it also can configure user callbacks, tensorboard,
    learning rate scheduler, and model saving.

    The built-in learning rate scheduler depends on a validation callback
    with an attribute 'best_epoch'.

    # Arguments
        logdir: Path to where all the logs and weights will be saved.
    """

    def __init__(self, model, gen_tr, gpu_model=None, steps_per_epoch=None,
            initial_lr=1e-3, lr_factor=0.1, lr_patience=10, minimum_lr=1e-7,
            epochs=1, verbose=1, workers=1, shuffle=True, initial_epoch=0,
            validation_callbacks=None, custom_lr_scheduler=None,
            save_tensor_board=False, weights_fname='weights.hdf5', logdir=None):

        self.model = model
        if gpu_model is not None:
            self.gpu_model = gpu_model
        else:
            self.gpu_model = model

        self.gen_tr = gen_tr
        self.steps_per_epoch = steps_per_epoch

        self.initial_lr = initial_lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_wait = 0
        self.minimum_lr = minimum_lr

        self.epochs = epochs
        self.verbose = verbose
        self.workers = workers
        self.shuffle = shuffle
        self.initial_epoch = initial_epoch

        self.val_cb = validation_callbacks
        self.callbacks = []
        self.weights_fname = weights_fname
        self.logdir = logdir

        if self.val_cb is not None:
            if not isinstance(self.val_cb, list):
                self.val_cb = [self.val_cb]

            self.callbacks += self.val_cb

            if custom_lr_scheduler is None:
                lrscheduler = LearningRateScheduler(
                        self.learningrate_scheduler)
                self.callbacks.append(lrscheduler)

        if custom_lr_scheduler is not None:
                lrscheduler = LearningRateScheduler(custom_lr_scheduler)
                self.callbacks.append(lrscheduler)

        if (self.logdir is not None) and save_tensor_board:
            tensorboard = TensorBoard(log_dir=self.logdir)
            self.callbacks.append(tensorboard)

        if len(self.callbacks) == 0:
            self.callbacks = None # Reset if not used


    def learningrate_scheduler(self, epoch, lr):
        best_epoch = self.val_cb[-1].best_epoch
        if epoch == self.initial_epoch:
            lr = self.initial_lr

        elif best_epoch == epoch:
            self.lr_wait = 0
            if self.logdir is not None:
                self.model.save_weights(
                        os.path.join(self.logdir, self.weights_fname))
        else:
            """Increase the waiting time if it was not the best epoch."""
            self.lr_wait += 1

        if self.lr_wait >= self.lr_patience:
            self.lr_wait = 0

            """Reduce the learning rate and (re)load the best model."""
            lr *= self.lr_factor

            if self.logdir is not None:
                printcn(OKGREEN,
                        'Reloading weights from epoch %03d' % best_epoch)
                self.model.load_weights(
                        os.path.join(self.logdir, self.weights_fname))

            if lr < self.minimum_lr:
                printcn(FAIL, 'Minimum learning rate reached!')
                self.gpu_model.stop_training = True
            else:
                printcn(OKGREEN, 'Setting learning rate to: %g' % lr)

        return lr

    def train(self):
        self.gpu_model.fit_generator(self.gen_tr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                workers=self.workers,
                use_multiprocessing=False,
                shuffle=self.shuffle,
                initial_epoch=self.initial_epoch)


class MultiModelTrainer(object):
    """This class is much more than a wrapper to the method 'fit_generator'
    from Keras. It is able to train a list o models, given a corresponding
    list of data generator (one per model), by training each model with one
    batch of its corresponding data. Actually, it is supposed that each model
    here is a small part of a bigger model (the full model), which is actually
    used for saveing weights.
    """

    def __init__(self, models, generators, workers=1, shuffle=True,
            max_queue_size=10, print_full_losses=False):

        assert len(models) == len(generators), \
                'ValueError: models and generators should be lists of same size'

        if type(workers) is not list:
            workers = len(models)*[workers]

        self.models = models
        self.output_generators = []
        self.batch_logs = {}
        self.print_full_losses = print_full_losses

        metric_names = []

        batch_size = 0
        for i in range(len(models)):
            assert isinstance(generators[i], BatchLoader), \
                    'Only BatchLoader class is supported'
            batch_size += generators[i].get_batch_size()
            enqueuer = OrderedEnqueuer(generators[i], shuffle=shuffle)
            enqueuer.start(workers=workers[i], max_queue_size=max_queue_size)
            self.output_generators.append(enqueuer.get())

            metric_names.append('loss%d' % i)
            if self.print_full_losses:
                for out in models[i].outputs:
                    metric_names.append(out.name.split('/')[0])

        self.batch_logs['size'] = batch_size
        self.metric_names = metric_names

    def train(self, epochs, steps_per_epoch, initial_epoch=0,
            end_of_epoch_callback=None, verbose=1):

        epoch = initial_epoch

        logger = ProgbarLogger(count_mode='steps')
        logger.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'metrics': self.metric_names})
        logger.on_train_begin()

        while epoch < epochs:
            step = 0
            batch = 0

            logger.on_epoch_begin(epoch)

            while step < steps_per_epoch:

                self.batch_logs['batch'] = batch
                logger.on_batch_begin(batch, self.batch_logs)

                for i in range(len(self.models)):
                    x, y = next(self.output_generators[i])
                    outs = self.models[i].train_on_batch(x, y)

                    if not isinstance(outs, list):
                        outs = [outs]
                    if self.print_full_losses:
                        for l, o in zip(self.metric_names, outs):
                            self.batch_logs[l] = o
                    else:
                        self.batch_logs[self.metric_names[i]] = outs[0]

                logger.on_batch_end(batch, self.batch_logs)

                step += 1
                batch += 1

            logger.on_epoch_end(epoch)
            if end_of_epoch_callback is not None:
                end_of_epoch_callback(epoch)

            epoch += 1

