import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from deephar.config import mpii_dataconf
from deephar.config import human36m_dataconf
from deephar.config import pennaction_dataconf
from deephar.config import ntu_pe_dataconf

from deephar.config import ModelConfig

from deephar.data import MpiiSinglePerson
from deephar.data import Human36M
from deephar.data import PennAction
from deephar.data import Ntu
from deephar.data import BatchLoader

from deephar.losses import pose_regression_loss
from keras.optimizers import RMSprop

from keras.callbacks import LambdaCallback
from deephar.callbacks import SaveModel

from deephar.models import spnet
from deephar.trainer import TrainerOnGenerator
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from datasetpath import datasetpath
from mpii_tools import MpiiEvalCallback
from h36m_tools import H36MEvalCallback


logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

cfg = ModelConfig(mpii_dataconf.input_shape, pa17j3d, num_pyramids=8,
        action_pyramids=[], num_levels=4)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)

start_lr = 0.001
weights_path = os.path.join(logdir, 'weights_posebaseline_{epoch:03d}.hdf5')

batch_size_mpii = 14
batch_size_ar = 2

"""Load datasets"""
mpii = MpiiSinglePerson(datasetpath('MPII'), dataconf=mpii_dataconf,
        poselayout=pa17j3d)

h36m = Human36M(datasetpath('Human3.6M'), dataconf=human36m_dataconf,
        poselayout=pa17j3d, topology='frames')

penn_sf = PennAction(datasetpath('Penn_Action'), pennaction_dataconf,
        poselayout=pa17j3d, topology='frames', use_gt_bbox=True)

ntu_sf = Ntu(datasetpath('NTU'), ntu_pe_dataconf, poselayout=pa17j3d,
        topology='frames', use_gt_bbox=True)

"""Create an object to load data from all datasets."""
data_tr = BatchLoader([mpii, h36m, penn_sf, ntu_sf], ['frame'], ['pose'],
        TRAIN_MODE, batch_size=[batch_size_mpii, batch_size_mpii, batch_size_ar,
            batch_size_ar], num_predictions=num_predictions, shuffle=True)

"""MPII validation samples."""
mpii_val = BatchLoader(mpii, ['frame'], ['pose', 'afmat', 'headsize'],
        VALID_MODE, batch_size=mpii.get_length(VALID_MODE), shuffle=True)
printcn(OKBLUE, 'Pre-loading MPII validation data...')
[x_val], [p_val, afmat_val, head_val] = mpii_val[0]
mpii_callback = MpiiEvalCallback(x_val, p_val, afmat_val, head_val,
        map_to_pa16j=pa17j3d.map_to_pa16j, logdir=logdir)

"""Human3.6H validation samples."""
h36m_val = BatchLoader(h36m, ['frame'],
        ['pose_w', 'pose_uvd', 'afmat', 'camera', 'action'], VALID_MODE,
        batch_size=h36m.get_length(VALID_MODE), shuffle=True)
printcn(OKBLUE, 'Preloading Human3.6M validation samples...')
[x_val], [pw_val, puvd_val, afmat_val, scam_val, action] = h36m_val[0]

h36m_callback = H36MEvalCallback(x_val, pw_val, afmat_val,
        puvd_val[:,0,2], scam_val, action, logdir=logdir)


model = spnet.build(cfg)

loss = pose_regression_loss('l1l2bincross', 0.01)
model.compile(loss=loss, optimizer=RMSprop(lr=start_lr))
model.summary()

callbacks = []
callbacks.append(SaveModel(weights_path))
callbacks.append(mpii_callback)
callbacks.append(h36m_callback)

steps_per_epoch = mpii.get_length(TRAIN_MODE) // batch_size_mpii

model.fit_generator(data_tr,
        steps_per_epoch=steps_per_epoch,
        epochs=60,
        callbacks=callbacks,
        workers=8,
        initial_epoch=0)

