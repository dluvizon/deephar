import os
import sys

import deephar
from keras.utils.data_utils import get_file

ORIGIN = 'https://github.com/dluvizon/deephar/releases/download/v0.1/'

def check_mpii_dataset():
    try:
        mpii_path = os.path.join(os.getcwd(), 'datasets/MPII/')
        annot_path = get_file(mpii_path + 'annotations.mat',
                ORIGIN + 'mpii_annotations.mat',
                md5_hash='cc62b1bb855bf4866d19bc0637526930')

        if os.path.isdir(mpii_path + 'images') is False:
            raise Exception('MPII dataset (images) not found! '
                    'You must download it by yourself from '
                    'http://human-pose.mpi-inf.mpg.de')

    except:
        sys.stderr.write('Error checking MPII dataset!\n')
        raise

