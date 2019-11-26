import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from deephar.config import mpii_dataconf
from deephar.config import human36m_dataconf
from deephar.config import ntu_dataconf
from deephar.config import ntu_pe_dataconf
from deephar.config import ModelConfig
from deephar.config import DataConfig

from deephar.data import MpiiSinglePerson
from deephar.data import Human36M
from deephar.data import PennAction
from deephar.data import Ntu
from deephar.data import BatchLoader

from deephar.losses import pose_regression_loss
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import keras.backend as K

from deephar.callbacks import SaveModel

from deephar.trainer import MultiModelTrainer
from deephar.models import compile_split_models
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from datasetpath import datasetpath

from mpii_tools import MpiiEvalCallback
from h36m_tools import H36MEvalCallback
from ntu_tools import NtuEvalCallback

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
cfg = ModelConfig((num_frames,) + ntu_dataconf.input_shape, pa17j3d,
        # num_actions=[60], num_pyramids=8, action_pyramids=[5, 6, 7, 8],
        num_actions=[60], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=192, num_visual_features=192)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

start_lr = 0.01
action_weight = 0.1
batch_size_mpii = 3
batch_size_h36m = 4
batch_size_ntu = 8 #1
batch_clips = 4 # 8/4

"""Load datasets"""
mpii = MpiiSinglePerson(datasetpath('MPII'), dataconf=mpii_dataconf,
        poselayout=pa17j3d)

# h36m = Human36M(datasetpath('Human3.6M'), dataconf=human36m_dataconf,
        # poselayout=pa17j3d, topology='frames')

ntu_sf = Ntu(datasetpath('NTU'), ntu_pe_dataconf, poselayout=pa17j3d,
        topology='frames', use_gt_bbox=True)

ntu = Ntu(datasetpath('NTU'), ntu_dataconf, poselayout=pa17j3d,
        topology='sequences', use_gt_bbox=True, clip_size=num_frames)

ntu_s1 = Ntu(datasetpath('NTU'), ntu_dataconf, poselayout=pa17j3d,
        topology='sequences', use_gt_bbox=True, clip_size=num_frames)
        # topology='sequences', use_gt_bbox=True, clip_size=num_frames, num_S=1)

pe_data_tr = BatchLoader([ntu_sf], ['frame'], ['pose'], TRAIN_MODE,
        batch_size=[batch_size_ntu],
        shuffle=True)
pe_data_tr = BatchLoader(pe_data_tr, ['frame'], ['pose'], TRAIN_MODE,
        batch_size=batch_clips, num_predictions=num_predictions, shuffle=False)

ar_data_tr = BatchLoader(ntu, ['frame'], ['ntuaction'], TRAIN_MODE,
        batch_size=batch_clips, num_predictions=num_action_predictions,
        shuffle=True)

"""Build the full model"""
full_model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
# full_model.load_weights(
        # 'output/pose_baseline_3dp_02_94226e0/weights_posebaseline_060.hdf5',
        # by_name=True)
# full_model.load_weights(
        # 'output/ntu_spnet_trial-03_fa9d2e2/weights_3dp+ntu_ar_050.hdf5',
        # by_name=True)
        # 'output/ntu_spnet_trial-03-ft2_0ae2bf7/weights_3dp+ntu_ar_058.hdf5')
# full_model.load_weights(
        # 'output/ntu_spnet_trial_06_nopose_f_512a239/weights_3dp+ntu_ar_020.hdf5',
        # by_name=True)

"""Trick to pre-load validation samples and generate the eval. callback."""
mpii_val = BatchLoader(mpii, ['frame'], ['pose', 'afmat', 'headsize'],
        VALID_MODE, batch_size=mpii.get_length(VALID_MODE), shuffle=False)
printnl('Pre-loading MPII validation data...')
[mpii_x_val], [mpii_p_val, mpii_afmat_val, mpii_head_val] = mpii_val[0]

"""Human3.6H validation samples."""
# h36m_val = BatchLoader(h36m, ['frame'],
        # ['pose_w', 'pose_uvd', 'afmat', 'camera', 'action'], VALID_MODE,
        # batch_size=h36m.get_length(VALID_MODE), shuffle=False)
# printcn(OKBLUE, 'Preloading Human3.6M validation samples...')
# [h36m_x_val], [h36m_pw_val, h36m_puvd_val, h36m_afmat_val, h36m_scam_val, \
        # h36m_action] = h36m_val[0]

"""NTU subset of testing samples"""
ntu_te = BatchLoader(ntu_s1, ['frame'], ['ntuaction'], TEST_MODE,
        batch_size=1, shuffle=False)

"""Save model callback."""
save_model = SaveModel(os.path.join(logdir,
    'weights_3dp+ntu_ar_{epoch:03d}.hdf5'), model_to_save=full_model)


def prepare_training(pose_trainable, lr):
    optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
    # optimizer = RMSprop(lr=lr)
    models = compile_split_models(full_model, cfg, optimizer,
            pose_trainable=pose_trainable, ar_loss_weights=action_weight,
            copy_replica=cfg.pose_replica)
    full_model.summary()

    """Create validation callbacks."""
    # mpii_callback = MpiiEvalCallback(mpii_x_val, mpii_p_val, mpii_afmat_val,
            # mpii_head_val, eval_model=models[0], pred_per_block=1,
            # map_to_pa16j=pa17j3d.map_to_pa16j, batch_size=1, logdir=logdir)

    # h36m_callback = H36MEvalCallback(h36m_x_val, h36m_pw_val, h36m_afmat_val,
            # h36m_puvd_val[:,0,2], h36m_scam_val, h36m_action,
            # batch_size=1, eval_model=models[0], logdir=logdir)

    ntu_callback = NtuEvalCallback(ntu_te, eval_model=models[1], logdir=logdir)

    def end_of_epoch_callback(epoch):

        save_model.on_epoch_end(epoch)
        # if epoch == 0 or epoch >= 50:
        # mpii_callback.on_epoch_end(epoch)
        # h36m_callback.on_epoch_end(epoch)

        ntu_callback.on_epoch_end(epoch)

        if epoch in [58, 70]:
            lr = float(K.get_value(optimizer.lr))
            newlr = 0.1*lr
            K.set_value(optimizer.lr, newlr)
            printcn(WARNING, 'lr_scheduler: lr %g -> %g @ %d' \
                    % (lr, newlr, epoch))

    return end_of_epoch_callback, models



steps_per_epoch = ntu.get_length(TRAIN_MODE) // batch_clips

fcallback, models = prepare_training(False, start_lr)
# trainer = MultiModelTrainer(models[1:], [ar_data_tr], workers=8,
        # print_full_losses=True)
# trainer.train(50, steps_per_epoch=steps_per_epoch, initial_epoch=0,
        # end_of_epoch_callback=fcallback)

"""Joint learning the full model."""
# steps_per_epoch = mpii.get_length(TRAIN_MODE) // (batch_size_mpii * batch_clips)

fcallback, models = prepare_training(True, 0.1*start_lr)
# trainer = MultiModelTrainer(models, [pe_data_tr, ar_data_tr], workers=8,
trainer = MultiModelTrainer(models, [pe_data_tr, ar_data_tr], workers=4,
        print_full_losses=True)
trainer.train(90, steps_per_epoch=steps_per_epoch, initial_epoch=20,
        end_of_epoch_callback=fcallback)

