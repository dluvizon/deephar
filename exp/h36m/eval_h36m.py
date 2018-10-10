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

weights_file = ''
TF_WEIGHTS_PATH = \
        'https://github.com/dluvizon/deephar/releases/download/v0.2/' \
        + weights_file
md5_hash = ''


logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)

num_blocks = 8
batch_size = 24
input_shape = human36m_dataconf.input_shape
num_joints = pa17j3d.num_joints

model = reception.build(input_shape, num_joints, dim=3, num_blocks=num_blocks,
        ksize=(5, 5))
model.load_weights('weights_0042.h5')

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

