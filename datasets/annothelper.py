import os
import sys

import deephar
from keras.utils.data_utils import get_file

ORIGIN = 'https://github.com/dluvizon/deephar/releases/download/'

def check_mpii_dataset():
    version = 'v0.1'
    try:
        mpii_path = os.path.join(os.getcwd(), 'datasets/MPII/')
        annot_path = get_file(mpii_path + 'annotations.mat',
                ORIGIN + version + '/mpii_annotations.mat',
                md5_hash='cc62b1bb855bf4866d19bc0637526930')

        if os.path.isdir(mpii_path + 'images') is False:
            raise Exception('MPII dataset (images) not found! '
                    'You must download it by yourself from '
                    'http://human-pose.mpi-inf.mpg.de')

    except:
        sys.stderr.write('Error checking MPII dataset!\n')
        raise

def check_h36m_dataset():
    version = 'v0.2'
    try:
        h36m_path = os.path.join(os.getcwd(), 'datasets/Human3.6M/')
        annot_path = get_file(h36m_path + 'annotations.mat',
                ORIGIN + 'h36m_annotations.mat',
                md5_hash='4067d52db61737fbebdec850238d87dd')

        if os.path.isdir(h36m_path + 'images') is False:
            raise Exception('Human3.6M dataset (images) not found! '
                    'You must download it by yourself from '
                    'http://vision.imar.ro/human3.6m '
                    'and extract the video files!')

    except:
        sys.stderr.write('Error checking Human3.6M dataset!\n')
        raise

