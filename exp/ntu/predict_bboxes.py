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

from deephar.data import Ntu
from deephar.data import BatchLoader

from deephar.models import spnet
from deephar.utils import *

from keras.models import Model

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from datasetpath import datasetpath
from generic import get_bbox_from_poses

from keras.models import Model

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

cfg = ModelConfig(dconf.input_shape, pa17j3d, num_pyramids=8,
        action_pyramids=[], num_levels=4)

full_model = spnet.build(cfg)
full_model.load_weights(
        'output/pose_baseline_3dp_02_94226e0/weights_posebaseline_060.hdf5')

"""Squeeze the model for only one output."""
model = Model(full_model.input, full_model.outputs[-1])
model.summary()

"""Load dataset"""
ntu = Ntu(datasetpath('NTU'), dconf, poselayout=pa17j3d, topology='frames',
        use_gt_bbox=False)

def predict_frame_bboxes(mode):
    bboxes = {}

    num_samples = ntu.get_length(mode)
    for i in range(num_samples):
        printnl('mode %d: %07d/%07d' % (mode, i+1, num_samples))

        data = ntu.get_data(i, mode)
        poses = model.predict(np.expand_dims(data['frame'], axis=0))
        bbox = get_bbox_from_poses(poses, data['afmat'], scale=1.5)
        seq_idx = data['seq_idx']
        f = data['frame_list'][0]
        bboxes['%d.%d' % (seq_idx, f)] = bbox.astype(int).tolist()

    return bboxes

bbox_te = predict_frame_bboxes(TEST_MODE)
bbox_tr = predict_frame_bboxes(TRAIN_MODE)
bbox_val = predict_frame_bboxes(VALID_MODE)

jsondata = [bbox_te, bbox_tr, bbox_val]
filename = os.path.join(logdir, 'pred_bboxes_ntu.json')
with open(filename, 'w') as fid:
    json.dump(jsondata, fid)

