import sys

import numpy as np

import random
from multiprocessing import Queue
import queue
import threading

from keras.utils import Sequence

from deephar.utils import *

class BatchLoader(Sequence):
    """Loader class for generic datasets, based on the Sequence class from
    Keras.

    One (or more) object(s) implementing a dataset should be provided.
    The required functions are 'get_length(self, mode)' and
    'get_data(self, key, mode)'. The first returns an integer, and the last
    returns a dictionary containing the data for a given pair of (key, mode).

    # Arguments
        dataset: A dataset object, or a list of dataset objects (for multiple
            datasets), which are merged by this class.
        x_dictkeys: Key names (strings) to constitute the baches of X data
            (input).
        y_dictkeys: Identical to x_dictkeys, but for Y data (labels).
            All given datasets must provide those keys.
        batch_size: Number of samples in each batch. If multiple datasets, it
            can be a list with the same length of 'dataset', where each value
            corresponds to the number of samples from the respective dataset,
            or it can be a single value, which corresponds to the number of
            samples from *each* dataset.
        num_predictions: number of predictions (y) that should be repeated for
            training.
        mode: TRAIN_MODE, TEST_MODE, or VALID_MODE.
        shuffle: boolean to shuffle *samples* (not batches!) or not.
        custom_dummy_dictkey: Allows to generate dummy outputs for each batch.
            Should be defined as a list of tuples, each with three values:
            (dictkey, shape, value). It is useful to include an action label for
            a sequence poses from pose-only datasets, e.g., when mixturing MPII
            and Human3.6M for training with action recognition at the same time
            (to further mergning with an action dataset).
    """
    BATCH_HOLD = 4

    def __init__(self, dataset, x_dictkeys, y_dictkeys, mode,
            batch_size=24, num_predictions=1, shuffle=True,
            custom_dummy_dictkey=[]):

        if not isinstance(dataset, list):
            dataset = [dataset]
        self.datasets = dataset
        self.x_dictkeys = x_dictkeys
        self.y_dictkeys = y_dictkeys
        self.allkeys = x_dictkeys + y_dictkeys

        """Include custom dictkeys into the output list."""
        self.custom_dummy_dictkey = custom_dummy_dictkey
        self.custom_dictkeys = []
        for dummyout in self.custom_dummy_dictkey:
            assert dummyout[0] not in self.y_dictkeys, \
                    'dummy key {} already in y_dictkeys!'.format(dummyout[0])
            self.custom_dictkeys.append(dummyout[0])
        self.y_dictkeys += self.custom_dictkeys

        """Make sure that all datasets have the same shapes for all dictkeys"""
        for dkey in self.allkeys:
            for i in range(1, len(self.datasets)):
                assert self.datasets[i].get_shape(dkey) == \
                        self.datasets[i-1].get_shape(dkey), \
                        'Incompatible dataset shape for dictkey {}'.format(dkey)

        self.batch_sizes = batch_size
        if not isinstance(self.batch_sizes, list):
            self.batch_sizes = len(self.datasets)*[self.batch_sizes]

        assert len(self.datasets) == len(self.batch_sizes), \
                'dataset and batch_size should be lists with the same length.'

        if isinstance(num_predictions, int):
            self.num_predictions = len(self.y_dictkeys)*[num_predictions]
        elif isinstance(num_predictions, list):
            self.num_predictions = num_predictions
        else:
            raise ValueError(
                'Invalid num_predictions ({})'.format(num_predictions))

        assert len(self.num_predictions) == len(self.y_dictkeys), \
                'num_predictions and y_dictkeys not matching'

        self.mode = mode
        self.shuffle = shuffle

        """Create one lock object for each dataset in case of data shuffle."""
        if self.shuffle:
            self.qkey = []
            self.lock = []
            for d in range(self.num_datasets):
                maxsize = self.datasets[d].get_length(self.mode) \
                        + BatchLoader.BATCH_HOLD*self.batch_sizes[d]
                self.qkey.append(Queue(maxsize=maxsize))
                self.lock.append(threading.Lock())

    def __len__(self):
        dataset_len = []
        for d in range(self.num_datasets):
            dataset_len.append(
                    int(np.ceil(self.datasets[d].get_length(self.mode) /
                        float(self.batch_sizes[d]))))

        return max(dataset_len)


    def __getitem__(self, idx):
        data_dict = self.get_data(idx, self.mode)

        """Convert the dictionary of samples to a list for x and y."""
        x_batch = []
        for dkey in self.x_dictkeys:
            x_batch.append(data_dict[dkey])

        y_batch = []
        for i, dkey in enumerate(self.y_dictkeys):
            for _ in range(self.num_predictions[i]):
                y_batch.append(data_dict[dkey])

        return x_batch, y_batch

    def get_batch_size(self):
        return sum(self.batch_sizes)

    def get_data(self, idx, mode):
        """Get the required data by mergning all the datasets as specified
        by the object's parameters."""
        data_dict = {}
        for dkey in self.allkeys:
            data_dict[dkey] = np.empty((sum(self.batch_sizes),) \
                    + self.datasets[0].get_shape(dkey))

        """Add custom dummy data."""
        for dummyout in self.custom_dummy_dictkey:
            dkey, dshape, dvalue = dummyout
            data_dict[dkey] = dvalue * np.ones(dshape)

        batch_cnt = 0
        for d in range(len(self.datasets)):
            for i in range(self.batch_sizes[d]):
                if self.shuffle:
                    key = self.get_shuffled_key(d)
                else:
                    key = idx*self.batch_sizes[d] + i
                    if key >= self.datasets[d].get_length(mode):
                        key -= self.datasets[d].get_length(mode)

                data = self.datasets[d].get_data(key, mode)
                for dkey in self.allkeys:
                    data_dict[dkey][batch_cnt, :] = data[dkey]

                batch_cnt += 1

        return data_dict

    def get_shape(self, dictkey):
        """Inception of get_shape method.
        First check if it is a custom key.
        """
        for dummyout in self.custom_dummy_dictkey:
            if dictkey == dummyout[0]:
                return dummyout[1]
        return (sum(self.batch_sizes),) + self.datasets[0].get_shape(dictkey)

    def get_length(self, mode):
        assert mode == self.mode, \
                'You are mixturing modes! {} with {}'.format(mode, self.mode)
        return len(self)

    def get_shuffled_key(self, dataset_idx):
        assert self.shuffle, \
                'There is not sense in calling this function if shuffle=False!'

        key = None
        with self.lock[dataset_idx]:
            min_samples = BatchLoader.BATCH_HOLD*self.batch_sizes[dataset_idx]
            if self.qkey[dataset_idx].qsize() <= min_samples:
                """Need to fill that list."""
                num_samples = self.datasets[dataset_idx].get_length(self.mode)
                newlist = list(range(num_samples))
                random.shuffle(newlist)
                try:
                    for j in newlist:
                        self.qkey[dataset_idx].put(j, False)
                except queue.Full:
                    pass
            key = self.qkey[dataset_idx].get()

        return key

    @property
    def num_datasets(self):
        return len(self.datasets)


