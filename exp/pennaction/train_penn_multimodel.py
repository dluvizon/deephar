import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from deephar.config import mpii_dataconf
from deephar.config import pennaction_dataconf
from deephar.config import pennaction_pe_dataconf
from deephar.config import ModelConfig

from deephar.data import MpiiSinglePerson
from deephar.data import PennAction
from deephar.data import BatchLoader

from deephar.losses import pose_regression_loss
from keras.optimizers import RMSprop
import keras.backend as K

from deephar.callbacks import SaveModel

from deephar.trainer import MultiModelTrainer
from deephar.models import compile_split_models
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from datasetpath import datasetpath

from mpii_tools import MpiiEvalCallback
from penn_tools import PennActionEvalCallback

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
cfg = ModelConfig((num_frames,) + pennaction_dataconf.input_shape, pa16j2d,
        num_actions=[15], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=160, num_visual_features=160)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

start_lr = 0.001
action_weight = 0.01
batch_size_mpii = int(0.8 * num_frames)
# batch_size_penn = num_frames - batch_size_mpii
batch_size_penn = num_frames
batch_clips = 4 # 8/4

"""Load datasets"""
mpii = MpiiSinglePerson(datasetpath('MPII'), dataconf=mpii_dataconf,
        poselayout=pa16j2d)

penn_sf = PennAction(datasetpath('Penn_Action'), pennaction_pe_dataconf,
        poselayout=pa16j2d, topology='frames', use_gt_bbox=True)

penn_seq = PennAction(datasetpath('Penn_Action'), pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=True,
        clip_size=num_frames)

# pe_data_tr = BatchLoader([mpii, penn_sf], ['frame'], ['pose'], TRAIN_MODE,
pe_data_tr = BatchLoader([mpii], ['frame'], ['pose'], TRAIN_MODE,
        # batch_size=[batch_size_mpii, batch_size_penn], shuffle=True)
        batch_size=[batch_size_penn], shuffle=True)
pe_data_tr = BatchLoader(pe_data_tr, ['frame'], ['pose'], TRAIN_MODE,
        batch_size=batch_clips, num_predictions=num_predictions, shuffle=False)

ar_data_tr = BatchLoader(penn_seq, ['frame'], ['pennaction'], TRAIN_MODE,
        batch_size=batch_clips, num_predictions=num_action_predictions,
        shuffle=True)

"""Build the full model"""
full_model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
# Here it is recommended to load a model pre-trained (few epochs) on pose estimation!
#full_model.load_weights(
#        'output/mpii_spnet_51_f47147e/weights_mpii_spnet_8b4l_039.hdf5',
#        by_name=True)

# from keras.models import Model
# full_model = Model(full_model.input,
        # [full_model.outputs[5], full_model.outputs[11]], name=full_model.name)
# cfg.num_pyramids = 1
# cfg.num_levels = 2
# cfg.action_pyramids = [2]

"""Trick to pre-load validation samples and generate the eval. callback."""
mpii_val = BatchLoader(mpii, ['frame'], ['pose', 'afmat', 'headsize'],
        VALID_MODE, batch_size=mpii.get_length(VALID_MODE), shuffle=False)
printnl('Pre-loading MPII validation data...')
[x_val], [p_val, afmat_val, head_val] = mpii_val[0]

penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TEST_MODE,
        batch_size=1, shuffle=False)

"""Save model callback."""
save_model = SaveModel(os.path.join(logdir,
    'weights_mpii+penn_ar_{epoch:03d}.hdf5'), model_to_save=full_model)


def prepare_training(pose_trainable, lr):
    optimizer = RMSprop(lr=lr)
    models = compile_split_models(full_model, cfg, optimizer,
            pose_trainable=pose_trainable, ar_loss_weights=action_weight,
            copy_replica=cfg.pose_replica)
    full_model.summary()

    """Create validation callbacks."""
    mpii_callback = MpiiEvalCallback(x_val, p_val, afmat_val, head_val,
            eval_model=models[0], pred_per_block=1, batch_size=1, logdir=logdir)
    penn_callback = PennActionEvalCallback(penn_te, eval_model=models[1],
            logdir=logdir)

    def end_of_epoch_callback(epoch):

        save_model.on_epoch_end(epoch)
        mpii_callback.on_epoch_end(epoch)
        penn_callback.on_epoch_end(epoch)

        if epoch in [15, 25]:
            lr = float(K.get_value(optimizer.lr))
            newlr = 0.1*lr
            K.set_value(optimizer.lr, newlr)
            printcn(WARNING, 'lr_scheduler: lr %g -> %g @ %d' \
                    % (lr, newlr, epoch))

    return end_of_epoch_callback, models

steps_per_epoch = mpii.get_length(TRAIN_MODE) // batch_size_mpii

fcallback, models = prepare_training(False, start_lr)
trainer = MultiModelTrainer(models[1:], [ar_data_tr], workers=12,
        print_full_losses=True)
trainer.train(2, steps_per_epoch=steps_per_epoch, initial_epoch=0,
        end_of_epoch_callback=fcallback)

"""Joint learning the full model."""
fcallback, models = prepare_training(True, start_lr)
trainer = MultiModelTrainer(models, [pe_data_tr, ar_data_tr], workers=12,
        print_full_losses=True)
trainer.train(30, steps_per_epoch=steps_per_epoch, initial_epoch=2,
        end_of_epoch_callback=fcallback)

