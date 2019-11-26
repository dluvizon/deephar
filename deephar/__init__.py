from __future__ import absolute_import

import sys
if sys.version_info[0] < 3:
    sys.stderr.write('You must use Python 3\n')
    sys.exit()

__version__ = '0.5.0'

info = 'Initializing deephar v{}\n'.format(__version__)
sys.stderr.write(info)

import os

"""Change this line if you want to use a custom (git) version of keras."""
keras_git = os.environ.get('HOME') + '/git/fchollet/keras'
info = 'Using keras'
if os.path.isdir(keras_git):
    sys.path.insert(0, keras_git)
    info += ' from "{}"'.format(keras_git)

try:
    sys.stderr.write('CUDA_VISIBLE_DEVICES: '
            + str(os.environ['CUDA_VISIBLE_DEVICES']) + '\n')
except:
    sys.stderr.write('CUDA_VISIBLE_DEVICES not defined\n')

import keras
info += ' version "{}"\n'.format(keras.__version__)

from . import data
from . import models
from . import utils

sys.stderr.write(info)
