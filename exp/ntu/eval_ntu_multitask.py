import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from deephar.config import human36m_dataconf
from deephar.config import ntu_dataconf
from deephar.config import ModelConfig
from deephar.config import DataConfig

from deephar.data import Human36M
from deephar.data import Ntu
from deephar.data import BatchLoader

from deephar.trainer import MultiModelTrainer
from deephar.models import split_model
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))

from h36m_tools import eval_human36m_sc_error
from ntu_tools import eval_multiclip_dataset

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
cfg = ModelConfig((num_frames,) + ntu_dataconf.input_shape, pa17j3d,
        num_actions=[60], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=192, num_visual_features=192)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

"""Load datasets"""
# h36m = Human36M(datasetpath('Human3.6M'), dataconf=human36m_dataconf,
        # poselayout=pa17j3d, topology='frames')

ntu_data_path = 'datasets/NTU'

ntu = Ntu(ntu_data_path, ntu_dataconf, poselayout=pa17j3d,
        topology='sequences', use_gt_bbox=True, clip_size=num_frames) #, num_S=1)
#print ('WARNING!! USING ONLY S1 FOR EVALUATION!')

"""Build the full model"""
full_model = spnet.build(cfg)

weights_file = 'output/ntu_spnet_trial-03-ft_replica_0ae2bf7/weights_3dp+ntu_ar_062.hdf5'

if os.path.isfile(weights_file) == False:
    print (f'Error: file {weights_file} not found!')
    print (f'\nPlease download it from  https://drive.google.com/file/d/1I6GftXEkL5nohLA60Vi6faW0rvTZg6Kx/view?usp=sharing')
    sys.stdout.flush()
    sys.exit()


"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(weights_file,
        #'output/ntu_spnet_trial_06_nopose_g_512a239/weights_3dp+ntu_ar_030.hdf5',
        by_name=True)

"""Split model to simplify evaluation."""
models = split_model(full_model, cfg, interlaced=False,
        model_names=['3DPose', '3DAction'])

"""Human3.6H validation samples."""
# h36m_val = BatchLoader(h36m, ['frame'],
        # ['pose_w', 'pose_uvd', 'afmat', 'camera', 'action'], VALID_MODE,
        # batch_size=h36m.get_length(VALID_MODE), shuffle=False)
# printcn(OKBLUE, 'Preloading Human3.6M validation samples...')
# [h36m_x_val], [h36m_pw_val, h36m_puvd_val, h36m_afmat_val, h36m_scam_val, \
        # h36m_action] = h36m_val[0]

"""NTU subset of testing samples"""
ntu_te = BatchLoader(ntu, ['frame'], ['ntuaction'], TEST_MODE, batch_size=1,
        shuffle=False)

"""Evaluate on Human3.6M using 3D poses."""
# s = eval_human36m_sc_error(models[0], h36m_x_val, h36m_pw_val, h36m_afmat_val,
        # h36m_puvd_val[:,0,2], h36m_scam_val, h36m_action, batch_size=2)

s = eval_multiclip_dataset(models[1], ntu,
        subsampling=ntu_dataconf.fixed_subsampling, logdir=logdir)
