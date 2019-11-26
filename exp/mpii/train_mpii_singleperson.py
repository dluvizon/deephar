import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from deephar.config import mpii_sp_dataconf

from deephar.data import MpiiSinglePerson
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.losses import pose_regression_loss

from keras.models import Model
from keras.layers import concatenate
from keras.utils.data_utils import get_file
from keras.optimizers import RMSprop

from deephar.utils import *

from keras.callbacks import LearningRateScheduler
from deephar.callbacks import SaveModel

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from mpii_tools import eval_singleperson_pckh
from mpii_tools import MpiiEvalCallback

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_mpii_dataset()

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

weights_file = os.parh.join(logdir, 'weights_mpii_{epoch:03d}.h5')

"""Architecture configuration."""
num_blocks = 8
batch_size = 24
input_shape = mpii_sp_dataconf.input_shape
num_joints = 16

model = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5))


"""Load the MPII dataset."""
mpii = MpiiSinglePerson('datasets/MPII', dataconf=mpii_sp_dataconf)

data_tr = BatchLoader(mpii, ['frame'], ['pose'], TRAIN_MODE,
        batch_size=batch_size, num_predictions=num_blocks, shuffle=True)

"""Pre-load validation samples and generate the eval. callback."""
mpii_val = BatchLoader(mpii, x_dictkeys=['frame'],
        y_dictkeys=['pose', 'afmat', 'headsize'], mode=VALID_MODE,
        batch_size=mpii.get_length(VALID_MODE), num_predictions=1,
        shuffle=False)
printcn(OKBLUE, 'Pre-loading MPII validation data...')
[x_val], [p_val, afmat_val, head_val] = mpii_val[0]
eval_callback = MpiiEvalCallback(x_val, p_val, afmat_val, head_val,
        eval_model=model, batch_size=2, pred_per_block=1, logdir=logdir)

loss = pose_regression_loss('l1l2bincross', 0.01)
model.compile(loss=loss, optimizer=RMSprop())
model.summary()

def lr_scheduler(epoch, lr):

    if epoch in [80, 100]:
        newlr = 0.2*lr
        printcn(WARNING, 'lr_scheduler: lr %g -> %g @ %d' % (lr, newlr, epoch))
    else:
        newlr = lr
        printcn(OKBLUE, 'lr_scheduler: lr %g @ %d' % (newlr, epoch))

    return newlr

callbacks = []
callbacks.append(SaveModel(weights_file))
callbacks.append(LearningRateScheduler(lr_scheduler))
callbacks.append(eval_callback)

steps_per_epoch = mpii.get_length(TRAIN_MODE) // batch_size

model.fit_generator(data_tr,
        steps_per_epoch=steps_per_epoch,
        epochs=120,
        callbacks=callbacks,
        workers=4,
        initial_epoch=0)

