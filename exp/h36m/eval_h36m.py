import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from keras.models import Model
from keras.layers import concatenate
from keras.utils.data_utils import get_file

from deephar.config import human36m_dataconf

from deephar.data import Human36M
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from h36m_tools import eval_human36m_sc_error

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_h36m_dataset()

weights_file = 'weights_3DPE_H36M_cvpr18_Nov-2017.h5'
TF_WEIGHTS_PATH = \
        'https://github.com/dluvizon/deephar/releases/download/v0.2/' \
        + weights_file
md5_hash = 'af79f83ad939117d4ccc2cf1d4bd37d2'


logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_blocks = 8
batch_size = 24
input_shape = human36m_dataconf.input_shape
num_joints = pa17j3d.num_joints

model = reception.build(input_shape, num_joints, dim=3, num_blocks=num_blocks,
        ksize=(5, 5), concat_pose_confidence=False)

"""Load pre-trained model."""
weights_path = get_file(weights_file, TF_WEIGHTS_PATH, md5_hash=md5_hash,
        cache_subdir='models')
model.load_weights(weights_path)

"""Merge pose and visibility as a single output."""
outputs = []
for b in range(int(len(model.outputs) / 2)):
    outputs.append(concatenate([model.outputs[2*b], model.outputs[2*b + 1]],
        name='blk%d' % (b + 1)))
model = Model(model.input, outputs, name=model.name)


"""Load Human3.6M dataset."""
h36m = Human36M('datasets/Human3.6M', dataconf=human36m_dataconf,
        poselayout=pa17j3d, topology='frames')

h36m_val = BatchLoader(h36m, ['frame'],
        ['pose_w', 'pose_uvd', 'afmat', 'camera', 'action'], VALID_MODE,
        batch_size=h36m.get_length(VALID_MODE), shuffle=False)
printcn(OKBLUE, 'Preloading Human3.6M validation samples...')
[x_val], [pw_val, puvd_val, afmat_val, scam_val, action] = h36m_val[0]

eval_human36m_sc_error(model, x_val, pw_val, afmat_val, puvd_val[:,0,2],
        scam_val, action, batch_size=24)

