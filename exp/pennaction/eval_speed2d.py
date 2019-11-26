import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from deephar.config import pennaction_dataconf
from deephar.config import ModelConfig

from deephar.data import PennAction
from deephar.data import BatchLoader

from deephar.models import split_model
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from datasetpath import datasetpath

from keras.models import Model
import time

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
num_clips = 250
cfg = ModelConfig((num_frames,) + pennaction_dataconf.input_shape, pa16j2d,
        num_actions=[15], num_pyramids=6, action_pyramids=[1, 2, 3, 4, 5, 6],
        num_levels=4, pose_replica=True,
        num_pose_features=160, num_visual_features=160)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)


"""Load PennAction"""
penn_seq = PennAction(datasetpath('Penn_Action'), pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=False,
        pred_bboxes_file='pred_bboxes_penn.json', clip_size=num_frames)


"""Build the full model"""
full_model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(
        'output/penn_multimodel_trial-07-full_2e9fa5a/weights_mpii+penn_ar_028.hdf5',
        by_name=True)

"""Pre-load some samples from PennAction."""
penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TEST_MODE,
        batch_size=num_clips, shuffle=False)
[x], [y] = penn_te[0]

num_blocks = len(full_model.outputs) // 2
fps_list = []
for bidx in range(num_blocks):
    """Build a new model considering a single prediction block."""
    inp = full_model.input
    outputs = full_model.outputs[2*bidx:2*bidx+2]
    m = Model(inp, outputs)

    """Warming up the new model."""
    _ = m.predict(x[0:1])

    start = time.time()
    _ = m.predict(x, batch_size=2, verbose=1)
    end = time.time()
    fps = num_clips * num_frames / (end - start)
    fps_list.append(fps)

print (fps_list)

