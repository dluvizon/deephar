# -*- coding: utf-8 -*-
"""Define the ReceptionNet for human pose estimation for Keras and TensorFlow.

The network is defined as:

-------   ------
|Input|-->|Stem|--> [...],
-------   ------

end every block:

                     -----------------------------------------------
                     |             --------------------------------|
           --------- |  ---------- |  ---------      ---------     |
    [...]->|rBlockN|--->|SepConvN|--->|RegMapN|-(H)->|fReMapN|--->(+)-->[...]
           ---------    ----------    ---------      ---------

For dim = 2 (2D poses):

                  |-->(sSAM)-------------------
         |--(Hs)--|                           |
         |        |-->(sjProp)--> *visible*   |
    H -> |                                    |
         |        |-->(cSAM)----------------(Agg)--> *pose*
         |--(Hc)--|                           |
                  |-->(cjProp)----------------|
"""
import numpy as np

from keras.models import Model
from keras.optimizers import RMSprop

from deephar.utils import *
from deephar.layers import *
from deephar.models.blocks import build_softargmax_1d
from deephar.models.blocks import build_softargmax_2d
from deephar.models.blocks import build_joints_probability
from deephar.models.blocks import build_context_aggregation

from deephar.objectives import elasticnet_loss_on_valid_joints


def _sepconv_residual(x, out_size, name, kernel_size=(3, 3)):
    shortcut_name = name + '_shortcut'
    reduce_name = name + '_reduce'

    num_filters = K.int_shape(x)[-1]
    if num_filters == out_size:
        ident = x
    else:
        ident = act_conv_bn(x, out_size, (1, 1), name=shortcut_name)

    if out_size < num_filters:
        x = act_conv_bn(x, out_size, (1, 1), name=reduce_name)

    x = separable_act_conv_bn(x, out_size, kernel_size, name=name)
    x = add([ident, x])

    return x

def _stem(inp, old_model=False):

    xi = Input(shape=K.int_shape(inp)[1:]) # 256 x 256 x 3

    x = conv_bn_act(xi, 32, (3, 3), strides=(2, 2))
    if not old_model:
        x = conv_bn_act(x, 32, (3, 3))
    x = conv_bn_act(x, 64, (3, 3))

    if old_model:
        a = conv_bn_act(x, 32, (3, 3), strides=(2, 2))
    else:
        a = conv_bn_act(x, 96, (3, 3), strides=(2, 2))
    b = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([a, b])

    a = conv_bn_act(x, 64, (1, 1))
    a = conv_bn(a, 96, (3, 3))
    b = conv_bn_act(x, 64, (1, 1))
    b = conv_bn_act(b, 64, (5, 1))
    b = conv_bn_act(b, 64, (1, 5))
    b = conv_bn(b, 96, (3, 3))
    x = concatenate([a, b])

    a = act_conv_bn(x, 192, (3, 3), strides=(2, 2))
    b = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = concatenate([a, b])

    if not old_model:
        x = _sepconv_residual(x, 3*192, name='sepconv1')

    model = Model(xi, x, name='Stem')
    x = model(inp)

    if old_model:
        x = _sepconv_residual(x, 512, name='sepconv1')

    return x


def build_reception_block(inp, name, ksize=(3, 3)):
    input_shape = K.int_shape(inp)[1:]
    size = input_shape[-1]

    xi = Input(shape=input_shape)
    a = _sepconv_residual(xi, size, name='sepconv_l1', kernel_size=ksize)

    low1 = MaxPooling2D((2, 2))(xi)
    low1 = act_conv_bn(low1, int(size/2), (1, 1))
    low1 = _sepconv_residual(low1, int(size/2), name='sepconv_l2_1',
            kernel_size=ksize)
    b = _sepconv_residual(low1, int(size/2), name='sepconv_l2_2',
            kernel_size=ksize)

    c = MaxPooling2D((2, 2))(low1)
    c = _sepconv_residual(c, int(size/2), name='sepconv_l3_1',
            kernel_size=ksize)
    c = _sepconv_residual(c, int(size/2), name='sepconv_l3_2',
            kernel_size=ksize)
    c = _sepconv_residual(c, int(size/2), name='sepconv_l3_3',
            kernel_size=ksize)
    c = UpSampling2D((2, 2))(c)

    b = add([b, c])
    b = _sepconv_residual(b, size, name='sepconv_l2_3', kernel_size=ksize)
    b = UpSampling2D((2, 2))(b)
    x = add([a, b])

    model = Model(inputs=xi, outputs=x, name=name)

    return model(inp)


def build_sconv_block(inp, name=None, ksize=(3, 3)):
    input_shape = K.int_shape(inp)[1:]

    xi = Input(shape=input_shape)
    x = separable_act_conv_bn(xi, input_shape[-1], ksize)

    model = Model(inputs=xi, outputs=x, name=name)

    return model(inp)


def build_regmap_block(inp, num_maps, name=None):
    input_shape = K.int_shape(inp)[1:]

    xi = Input(shape=input_shape)
    x = act_conv(xi, num_maps, (1, 1))

    model = Model(inputs=xi, outputs=x, name=name)

    return model(inp)


def build_fremap_block(inp, num_filters, name=None):
    input_shape = K.int_shape(inp)[1:]

    xi = Input(shape=input_shape)
    x = act_conv_bn(xi, num_filters, (1, 1))

    model = Model(inputs=xi, outputs=x, name=name)

    return model(inp)


def pose_regression_2d_context(h, num_joints, sam_s_model,
        sam_c_model, jprob_c_model, agg_model, jprob_s_model):

    # Split heatmaps for specialized and contextual information
    hs = Lambda(lambda x: x[:,:,:,:num_joints])(h)
    hc = Lambda(lambda x: x[:,:,:,num_joints:])(h)

    # Soft-argmax and joint probability for each heatmap
    ps = sam_s_model(hs)
    pc = sam_c_model(hc)
    vc = jprob_c_model(hc)

    pose = agg_model([ps, pc, vc])
    visible = jprob_s_model(hs)

    return pose, visible, hs


def pose_regression_2d(h, sam_s_model, jprob_s_model):

    pose = sam_s_model(h)
    visible = jprob_s_model(h)

    return pose, visible, h


def pose_regression_3d(h, num_joints, depth_maps, sam_s_model, sam_z_model):
    assert K.int_shape(h)[-1] == depth_maps * num_joints

    def _reshape_heatmaps(x):
        x = K.expand_dims(x, axis=-1)
        x = K.reshape(x, (-1, K.int_shape(x)[1], K.int_shape(x)[2],
            depth_maps, num_joints))

        return x

    h = Lambda(_reshape_heatmaps)(h)
    hxy = Lambda(lambda x: K.mean(x, axis=3))(h)
    hz = Lambda(lambda x: K.mean(x, axis=(1, 2)))(h)
    # hxy = Lambda(lambda x: K.max(x, axis=3))(h)
    # hz = Lambda(lambda x: K.max(x, axis=(1, 2)))(h)

    # hxy_s = act_channel_softmax(hxy)
    # hz_s = act_depth_softmax(hz)

    pxy = sam_s_model(hxy)
    pz = sam_z_model(hz)
    pose = concatenate([pxy, pz])

    vxy = GlobalMaxPooling2D()(hxy)
    vz = GlobalMaxPooling1D()(hz)
    v = add([vxy, vz])
    v = Lambda(lambda x: K.expand_dims(x, axis=-1))(v)
    visible = Activation('sigmoid')(v)

    return pose, visible, hxy


def build(input_shape, num_joints, dim,
        num_context_per_joint=None,
        alpha=0.8,
        num_blocks=4,
        depth_maps=16,
        ksize=(3, 3),
        export_heatmaps=False,
        export_vfeat_block=None,
        old_model=False,
        concat_pose_confidence=True):

    if dim == 2:
        if num_context_per_joint is None:
            num_context_per_joint = 2

        num_heatmaps = (num_context_per_joint + 1) * num_joints

    elif dim == 3:
        assert num_context_per_joint == None, \
                'For 3D pose estimation, contextual heat maps are not allowed.'
        num_heatmaps = depth_maps * num_joints
    else:
        raise ValueError('"dim" must be 2 or 3 and not (%d)' % dim)

    inp = Input(shape=input_shape)
    outputs = []
    vfeat = None

    x = _stem(inp, old_model=old_model)

    num_rows, num_cols, num_filters = K.int_shape(x)[1:]

    # Build the soft-argmax models (no parameters) for specialized and
    # contextual maps.
    sams_input_shape = (num_rows, num_cols, num_joints)
    sam_s_model = build_softargmax_2d(sams_input_shape, rho=0, name='sSAM')
    jprob_s_model = build_joints_probability(sams_input_shape, name='sjProb')

    # Build the aggregation model (no parameters)
    if num_context_per_joint is not None:
        samc_input_shape = (num_rows, num_cols, num_heatmaps - num_joints)
        sam_c_model = build_softargmax_2d(samc_input_shape, rho=0,
                name='cSAM')
        jprob_c_model = build_joints_probability(samc_input_shape,
                name='cjProb')
        agg_model = build_context_aggregation(num_joints,
                num_context_per_joint, alpha, name='Agg')

    if dim == 3:
        samz_input_shape = (depth_maps, num_joints)
        sam_z_model = build_softargmax_1d(samz_input_shape, name='zSAM')

    for bidx in range(num_blocks):
        block_shape = K.int_shape(x)[1:]
        x = build_reception_block(x, name='rBlock%d' % (bidx + 1), ksize=ksize)

        if export_vfeat_block == (bidx+1):
            vfeat = x

        ident_map = x
        x = build_sconv_block(x, name='SepConv%d' % (bidx + 1), ksize=ksize)
        h = build_regmap_block(x, num_heatmaps, name='RegMap%d' % (bidx + 1))

        if dim == 2:
            if num_context_per_joint is not None:
                pose, visible, hm = pose_regression_2d_context(h, num_joints,
                        sam_s_model, sam_c_model, jprob_c_model, agg_model,
                        jprob_s_model)
            else:
                pose, visible, hm = pose_regression_2d(h, sam_s_model,
                        jprob_s_model)
        else:
            pose, visible, hm = pose_regression_3d(h, num_joints, depth_maps,
                    sam_s_model, sam_z_model)

        if concat_pose_confidence:
            outputs.append(concatenate([pose, visible]))
        else:
            outputs.append(pose)
            outputs.append(visible)

        if export_heatmaps:
            outputs.append(hm)

        if bidx < num_blocks - 1:
            h = build_fremap_block(h, block_shape[-1],
                    name='fReMap%d' % (bidx + 1))
            x = add([ident_map, x, h])

    if vfeat is not None:
        outputs.append(vfeat)

    model = Model(inputs=inp, outputs=outputs)

    return model


def compile(model, ptr, vtr, num_y_per_branch=1):
    """Create a list with ground truth, loss functions and loss weights.
    """
    yholder_tr = []
    losses = []
    loss_weights = []
    num_blocks = int(len(model.output) / (num_y_per_branch + 1))

    printcn(OKBLUE,
            'Compiling model with %d outputs per branch and %d branches.' %
            (num_y_per_branch, num_blocks))

    for i in range(num_blocks):
        for j in range(num_y_per_branch):
            yholder_tr.append(ptr)
            losses.append(elasticnet_loss_on_valid_joints)
            loss_weights.append(1.)
        yholder_tr.append(vtr)
        losses.append('binary_crossentropy')
        loss_weights.append(0.01)

    printcn(OKBLUE, 'loss_weights: ' + str(loss_weights))
    model.compile(loss=losses, optimizer=RMSprop(), loss_weights=loss_weights)

    return yholder_tr

