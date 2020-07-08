import os
import sys

import numpy as np
import json

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from deephar.config import ModelConfig
from deephar.config import DataConfig
dconf = DataConfig(scales=[0.9], hflips=[0], chpower=[1])

from deephar.data import PennAction
from deephar.data import BatchLoader

from deephar.models import spnet
from deephar.utils import *

from keras.models import Model

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))

from generic import get_bbox_from_poses

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

cfg = ModelConfig(dconf.input_shape, pa16j2d, num_pyramids=8, num_levels=4, action_pyramids=[])

"""Load dataset"""
datapath = 'datasets/PennAction'
penn = PennAction(datapath, dconf, poselayout=pa16j2d, topology='frames',
        use_gt_bbox=False)

"""Build and compile the network."""
model = spnet.build(cfg)

# The weights for predicting the bounding boxes are avaiable upon request.
# If needed, please contact the authors by email.
model.load_weights(
        'output/mpii_spnet_51b_741a720/weights_mpii_spnet_8b4l_050.hdf5')

"""Squeeze the model for only one output."""
model = Model(model.input, model.outputs[-1])


def predict_frame_bboxes(mode):
    bboxes = {}

    num_samples = penn.get_length(mode)

    for i in range(num_samples):
        printnl('%d: %06d/%06d' % (mode, i+1, num_samples))

        data = penn.get_data(i, mode)
        poses = model.predict(np.expand_dims(data['frame'], axis=0))
        bbox = get_bbox_from_poses(poses, data['afmat'], scale=1.5)
        seq_idx = data['seq_idx']
        f = data['frame_list'][0]
        bboxes['%d.%d' % (seq_idx, f)] = bbox.astype(int).tolist()

    return bboxes

bbox_tr = predict_frame_bboxes(TRAIN_MODE)
bbox_te = predict_frame_bboxes(TEST_MODE)
bbox_val = predict_frame_bboxes(VALID_MODE)

jsondata = [bbox_te, bbox_tr, bbox_val]

filename = os.path.join(datapath, 'pred_bboxes_penn.json')
with open(filename, 'w') as fid:
    json.dump(jsondata, fid)

