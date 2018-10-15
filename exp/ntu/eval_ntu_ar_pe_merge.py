import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from keras.utils.data_utils import get_file

from deephar.config import ntu_ar_dataconf

from deephar.data import Ntu
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.models import action
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_ntu_dataset()

weights_file = ''
TF_WEIGHTS_PATH = \
        'https://github.com/dluvizon/deephar/releases/download/v0.3/' \
        + weights_file
md5_hash = ''

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')


num_frames = 20
num_blocks = 4
batch_size = 2
depth_maps = 8
num_joints = 20
num_actions = 60
pose_dim = 3
input_shape = ntu_ar_dataconf.input_shape

"""Build the pose estimation model."""
model_pe = reception.build(input_shape, num_joints, dim=pose_dim,
        num_blocks=num_blocks, depth_maps=depth_maps, ksize=(5, 5))

"""Build the full model using the previous pose estimation one."""
model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=pose_dim,
        num_context_per_joint=0, pose_net_version='v2')

"""Load pre-trained model."""
model.load_weights('weights_0052.h5')

"""Load NTU dataset."""
ntu = Ntu(datasetpath('NTU'), ntu_dataconf, poselayout=pa21j3d,
        topology='sequences', use_gt_bbox=True, clip_size=num_frames, num_S=1)

ntu_te = BatchLoader(ntu, ['frame'], ['ntuaction'], TEST_MODE,
        batch_size=1, shuffle=False)

printcn(OKGREEN, 'Evaluation on NTU single-clip using GT bboxes')

