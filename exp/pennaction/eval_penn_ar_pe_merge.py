import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from keras.utils.data_utils import get_file

from deephar.config import pennaction_dataconf

from deephar.data import PennAction
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.models import action
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from penn_tools import eval_singleclip_generator
from penn_tools import eval_multiclip_dataset

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_pennaction_dataset()

weights_file = 'weights_AR_merge_ep074_26-10-17.h5'
TF_WEIGHTS_PATH = \
        'https://github.com/dluvizon/deephar/releases/download/v0.3/' \
        + weights_file
md5_hash = 'f53f89257077616a79a6c1cd1702d50f'

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')


num_frames = 16
use_bbox = False
num_blocks = 4
batch_size = 2
input_shape = pennaction_dataconf.input_shape
num_joints = 16
num_actions = 15

"""Build pose and action models."""
model_pe = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
        concat_pose_confidence=False)

model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=2, pose_net_version='v1',
        full_trainable=False)

"""Load pre-trained model."""
weights_path = get_file(weights_file, TF_WEIGHTS_PATH, md5_hash=md5_hash,
        cache_subdir='models')
model.load_weights(weights_path)

"""Load PennAction dataset."""
penn_seq = PennAction('datasets/PennAction', pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=use_bbox,
        clip_size=num_frames)

penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TEST_MODE,
        batch_size=1, shuffle=False)


printcn(OKGREEN, 'Evaluation on PennAction multi-clip using predicted bboxes')
eval_multiclip_dataset(model, penn_seq,
        bboxes_file='datasets/PennAction/penn_pred_bboxes_16f.json',
        logdir=logdir)
