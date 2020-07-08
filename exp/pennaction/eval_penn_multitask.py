import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from deephar.config import mpii_dataconf
from deephar.config import pennaction_dataconf
from deephar.config import ModelConfig

from deephar.data import MpiiSinglePerson
from deephar.data import PennAction
from deephar.data import BatchLoader

from deephar.models import split_model
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from mpii_tools import eval_singleperson_pckh
from penn_tools import eval_singleclip_generator
from penn_tools import eval_multiclip_dataset

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper
annothelper.check_pennaction_dataset()

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
cfg = ModelConfig((num_frames,) + pennaction_dataconf.input_shape, pa16j2d,
        num_actions=[15], num_pyramids=6, action_pyramids=[5, 6],
        num_levels=4, pose_replica=True,
        num_pose_features=160, num_visual_features=160)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)


"""Load datasets"""
mpii = MpiiSinglePerson('datasets/MPII', dataconf=mpii_dataconf,
        poselayout=pa16j2d)

# Check file with bounding boxes
penn_data_path = 'datasets/PennAction'
penn_bbox_file = 'penn_pred_bboxes_multitask.json'
if os.path.isfile(os.path.join(penn_data_path, penn_bbox_file)) == False:
    print (f'Error: file {penn_bbox_file} not found in {penn_data_path}!')
    print (f'\nPlease download it from https://drive.google.com/file/d/1qXpEKF0d9KxmQdd2_QSIA1c3WGj1D3Y3/view?usp=sharing')
    sys.stdout.flush()
    sys.exit()

penn_seq = PennAction(penn_data_path, pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=False,
        pred_bboxes_file='penn_pred_bboxes_multitask.json', clip_size=num_frames)


"""Build the full model"""
full_model = spnet.build(cfg)

weights_file = 'weights/weights_mpii+penn_ar_028.hdf5'
if os.path.isfile(weights_file) == False:
    print (f'Error: file {weights_file} not found!')
    print (f'\nPlease download it from https://drive.google.com/file/d/106yIhqNN-TrI34SX81q2xbU-NczcQj6I/view?usp=sharing')
    sys.stdout.flush()
    sys.exit()

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(weights_file, by_name=True)

"""This call splits the model into its parts: pose estimation and action
recognition, so we can evaluate each part separately on its respective datasets.
"""
models = split_model(full_model, cfg, interlaced=False,
        model_names=['2DPose', '2DAction'])


"""Trick to pre-load validation samples from MPII."""
mpii_val = BatchLoader(mpii, ['frame'], ['pose', 'afmat', 'headsize'],
        VALID_MODE, batch_size=mpii.get_length(VALID_MODE), shuffle=False)
printnl('Pre-loading MPII validation data...')
[x_val], [p_val, afmat_val, head_val] = mpii_val[0]


"""Define a loader for PennAction test samples. """
penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TEST_MODE,
        batch_size=1, shuffle=False)

"""Evaluate on 2D action recognition (PennAction)."""
s = eval_singleclip_generator(models[1], penn_te)
print ('Best score on PennAction (single-clip): ' + str(s))

s = eval_multiclip_dataset(models[1], penn_seq,
        subsampling=pennaction_dataconf.fixed_subsampling)
print ('Best score on PennAction (multi-clip): ' + str(s))

"""Evaluate on 2D pose estimation (MPII)."""
s = eval_singleperson_pckh(models[0], x_val, p_val[:, :, 0:2], afmat_val, head_val)
print ('Best score on MPII: ' + str(s))


