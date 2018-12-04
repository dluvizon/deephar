from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD

from deephar.activations import channel_softmax_2d
from deephar.models.blocks import build_context_aggregation

from deephar.utils import *
from deephar.layers import *

from deephar.objectives import elasticnet_loss_on_valid_joints


def action_top(x, name=None):
    x = global_max_min_pooling(x)
    x = Activation('softmax', name=name)(x)
    return x


def build_act_pred_block(x, num_out, name=None, last=False, include_top=True):

    num_features = K.int_shape(x)[-1]

    ident = x
    x = act_conv_bn(x, int(num_features/2), (1, 1))
    x = act_conv_bn(x, num_features, (3, 3))
    x = add([ident, x])

    ident = x
    x1 = act_conv_bn(x, num_features, (3, 3))
    x = max_min_pooling(x1, (2, 2))
    action_hm = act_conv(x, num_out, (3, 3))
    y = action_hm
    if include_top:
        y = action_top(y)

    if not last:
        action_hm = UpSampling2D((2, 2))(action_hm)
        action_hm = act_conv_bn(action_hm, num_features, (3, 3))
        x = add([ident, x1, action_hm])

    return x, y


def build_pose_model(num_joints, num_actions, num_temp_frames=None, pose_dim=2,
        name=None, include_top=True, network_version='v1'):

    y = Input(shape=(num_temp_frames, num_joints, pose_dim))
    p = Input(shape=(num_temp_frames, num_joints, 1))

    ## Pose information
    mask = Lambda(lambda x: K.tile(x, [1, 1, 1, pose_dim]))(p)
    x = Lambda(lambda x: x[0] * x[1])([y, mask])

    if network_version == 'v1':
        a = conv_bn_act(x, 8, (3, 1))
        b = conv_bn_act(x, 16, (3, 3))
        c = conv_bn_act(x, 24, (3, 5))
        x = concatenate([a, b, c])
        a = conv_bn(x, 56, (3, 3))
        b = conv_bn(x, 32, (1, 1))
        b = conv_bn(b, 56, (3, 3))
        x = concatenate([a, b])
        x = max_min_pooling(x, (2, 2))
    elif network_version == 'v2':
        a = conv_bn_act(x, 12, (3, 1))
        b = conv_bn_act(x, 24, (3, 3))
        c = conv_bn_act(x, 36, (3, 5))
        x = concatenate([a, b, c])
        a = conv_bn(x, 112, (3, 3))
        b = conv_bn(x, 64, (1, 1))
        b = conv_bn(b, 112, (3, 3))
        x = concatenate([a, b])
        x = max_min_pooling(x, (2, 2))
    else:
        raise Exception('Unkown network version "{}"'.format(network_version))

    x, y1 = build_act_pred_block(x, num_actions, name='y1',
            include_top=include_top)
    x, y2 = build_act_pred_block(x, num_actions, name='y2',
            include_top=include_top)
    x, y3 = build_act_pred_block(x, num_actions, name='y3',
            include_top=include_top)
    _, y4 = build_act_pred_block(x, num_actions, name='y4',
            include_top=include_top, last=True)
    x = [y1, y2, y3, y4]

    model = Model(inputs=[y, p], outputs=x, name=name)

    return model


def build_visual_model(num_joints, num_actions, num_features,
        num_temp_frames=None, name=None, include_top=True):

    inp = Input(shape=(num_temp_frames, num_joints, num_features))
    x = conv_bn(inp, 256, (1, 1))
    x = MaxPooling2D((2, 2))(x)
    x, y1 = build_act_pred_block(x, num_actions, name='y1',
            include_top=include_top)
    x, y2 = build_act_pred_block(x, num_actions, name='y2',
            include_top=include_top)
    x, y3 = build_act_pred_block(x, num_actions, name='y3',
            include_top=include_top)
    _, y4 = build_act_pred_block(x, num_actions, name='y4',
            include_top=include_top, last=True)
    model = Model(inp, [y1, y2, y3, y4], name=name)

    return model


def _get_2d_pose_estimation_from_model(inp, model_pe, num_joints, num_blocks,
        num_context_per_joint, full_trainable=False):

    num_frames = K.int_shape(inp)[1]

    stem = model_pe.get_layer('Stem')
    stem.trainable = full_trainable

    i = 1
    recep_block = model_pe.get_layer('rBlock%d' % i)
    recep_block.trainable = full_trainable

    x1 = TimeDistributed(stem, name='td_%s' % stem.name)(inp)
    xb1 = TimeDistributed(recep_block, name='td_%s' % recep_block.name)(x1)

    inp_pe = Input(shape=K.int_shape(xb1)[2:])
    sep_conv = model_pe.get_layer('SepConv%d' % i)
    reg_map = model_pe.get_layer('RegMap%d' % i)
    fre_map = model_pe.get_layer('fReMap%d' % i)
    x2 = sep_conv(inp_pe)
    x3 = fre_map(reg_map(x2))
    x = add([inp_pe, x2, x3])

    for i in range(2, num_blocks):
        recep_block = model_pe.get_layer('rBlock%d' % i)
        sep_conv = model_pe.get_layer('SepConv%d' % i)
        reg_map = model_pe.get_layer('RegMap%d' % i)
        fre_map = model_pe.get_layer('fReMap%d' % i)
        x1 = recep_block(x)
        x2 = sep_conv(x1)
        x3 = fre_map(reg_map(x2))
        x = add([x1, x2, x3])

    recep_block = model_pe.get_layer('rBlock%d' % num_blocks)
    sep_conv = model_pe.get_layer('SepConv%d' % num_blocks)
    reg_map = model_pe.get_layer('RegMap%d' % num_blocks)
    x = recep_block(x)
    x = sep_conv(x)
    x = reg_map(x)

    model1 = Model(inp_pe, x, name='PoseReg')
    model1.trainable = full_trainable

    num_heatmaps = (num_context_per_joint + 1) * num_joints
    num_rows = K.int_shape(model1.output)[1]
    num_cols = K.int_shape(model1.output)[2]

    sams_input_shape = (num_frames, num_rows, num_cols, num_joints)
    samc_input_shape = \
            (num_frames, num_rows, num_cols, num_heatmaps - num_joints)

    # Build the time distributed models
    model_pe.get_layer('sSAM').trainable = full_trainable
    sam_s_model = TimeDistributed(model_pe.get_layer('sSAM'),
            input_shape=sams_input_shape, name='sSAM')

    if num_context_per_joint > 0:
        model_pe.get_layer('cSAM').trainable = full_trainable
        sam_c_model = TimeDistributed(model_pe.get_layer('cSAM'),
                input_shape=samc_input_shape, name='cSAM')

    model_pe.get_layer('sjProb').trainable = False
    jprob_s_model = TimeDistributed(model_pe.get_layer('sjProb'),
            input_shape=sams_input_shape, name='sjProb')

    if num_context_per_joint > 0:
        model_pe.get_layer('cjProb').trainable = False
        jprob_c_model = TimeDistributed(model_pe.get_layer('cjProb'),
                input_shape=samc_input_shape, name='cjProb')

    agg_model = build_context_aggregation(num_joints,
            num_context_per_joint, 0.8, num_frames=num_frames, name='Agg')

    h = TimeDistributed(model1, name='td_Model1')(xb1)
    if num_context_per_joint > 0:
        hs = Lambda(lambda x: x[:,:,:,:, :num_joints])(h)
        hc = Lambda(lambda x: x[:,:,:,:, num_joints:])(h)
    else:
        hs = h

    ys = sam_s_model(hs)
    if num_context_per_joint > 0:
        yc = sam_c_model(hc)
        pc = jprob_c_model(hc)
        y = agg_model([ys, yc, pc])
    else:
        y = ys

    p = jprob_s_model(Lambda(lambda x: 4*x)(hs))

    hs = TimeDistributed(Activation(channel_softmax_2d()),
            name='td_ChannelSoftmax')(hs)

    return y, p, hs, xb1


def _get_3d_pose_estimation_from_model(inp, model_pe, num_joints, num_blocks,
        depth_maps, full_trainable=False):

    num_frames = K.int_shape(inp)[1]

    model_pe.summary()

    stem = model_pe.get_layer('Stem')
    stem.trainable = full_trainable

    i = 1
    recep_block = model_pe.get_layer('rBlock%d' % i)
    recep_block.trainable = full_trainable

    x1 = TimeDistributed(stem, name='td_%s' % stem.name)(inp)
    xb1 = TimeDistributed(recep_block, name='td_%s' % recep_block.name)(x1)

    inp_pe = Input(shape=K.int_shape(xb1)[2:])
    sep_conv = model_pe.get_layer('SepConv%d' % i)
    reg_map = model_pe.get_layer('RegMap%d' % i)
    fre_map = model_pe.get_layer('fReMap%d' % i)
    x2 = sep_conv(inp_pe)
    x3 = fre_map(reg_map(x2))
    x = add([inp_pe, x2, x3])

    for i in range(2, num_blocks):
        recep_block = model_pe.get_layer('rBlock%d' % i)
        sep_conv = model_pe.get_layer('SepConv%d' % i)
        reg_map = model_pe.get_layer('RegMap%d' % i)
        fre_map = model_pe.get_layer('fReMap%d' % i)
        x1 = recep_block(x)
        x2 = sep_conv(x1)
        x3 = fre_map(reg_map(x2))
        x = add([x1, x2, x3])

    recep_block = model_pe.get_layer('rBlock%d' % num_blocks)
    sep_conv = model_pe.get_layer('SepConv%d' % num_blocks)
    reg_map = model_pe.get_layer('RegMap%d' % num_blocks)
    x = recep_block(x)
    x = sep_conv(x)
    x = reg_map(x)

    model1 = Model(inp_pe, x, name='PoseReg')
    model1.trainable = full_trainable

    num_rows = K.int_shape(model1.output)[1]
    num_cols = K.int_shape(model1.output)[2]

    sams_input_shape = (num_frames, num_rows, num_cols, num_joints)
    samz_input_shape = (num_frames, depth_maps, num_joints)

    # Build the time distributed models
    model_pe.get_layer('sSAM').trainable = full_trainable
    sam_s_model = TimeDistributed(model_pe.get_layer('sSAM'),
            input_shape=sams_input_shape, name='sSAM')

    model_pe.get_layer('zSAM').trainable = full_trainable
    sam_z_model = TimeDistributed(model_pe.get_layer('zSAM'),
            input_shape=samz_input_shape, name='zSAM')

    h = TimeDistributed(model1, name='td_Model1')(xb1)
    assert K.int_shape(h)[-1] == depth_maps * num_joints

    def _reshape_heatmaps(x):
        x = K.expand_dims(x, axis=-1)
        x = K.reshape(x, (-1, K.int_shape(x)[1], K.int_shape(x)[2],
            K.int_shape(x)[3], depth_maps, num_joints))

        return x

    h = Lambda(_reshape_heatmaps)(h)
    hxy = Lambda(lambda x: K.mean(x, axis=4))(h)
    hz = Lambda(lambda x: K.mean(x, axis=(2, 3)))(h)

    pxy = sam_s_model(hxy)
    pz = sam_z_model(hz)
    pose = concatenate([pxy, pz])

    vxy = TimeDistributed(GlobalMaxPooling2D(), name='td_GlobalMaxPooling2D',
            input_shape=K.int_shape(hxy)[1:])(hxy)
    vz = TimeDistributed(GlobalMaxPooling1D(), name='td_GlobalMaxPooling1D',
            input_shape=K.int_shape(hz)[1:])(hz)
    v = add([vxy, vz])
    v = Lambda(lambda x: 2*K.expand_dims(x, axis=-1))(v)
    visible = Activation('sigmoid')(v)

    hxy = TimeDistributed(Activation(channel_softmax_2d()),
            name='td_ChannelSoftmax')(hxy)

    return pose, visible, hxy, xb1


def build_guided_visual_model(model_pe, num_actions, input_shape, num_frames,
        num_joints, num_blocks, num_context_per_joint=2):

    inp = Input(shape=(num_frames,) + input_shape)
    _, _, hs, xb1 = _get_2d_pose_estimation_from_model(inp, model_pe, num_joints,
            num_blocks, num_context_per_joint,
            num_context_per_joint=num_context_per_joint)

    f = kronecker_prod(hs, xb1)
    num_features = K.int_shape(f)[-1]
    model_ar = build_visual_model(num_joints, num_actions, num_features,
            num_temp_frames=num_frames, name='GuidedVisAR')

    x = model_ar(f)
    model = Model(inp, x)

    return model


def build_merge_model(model_pe,
        num_actions,
        input_shape,
        num_frames,
        num_joints,
        num_blocks,
        pose_dim=2,
        depth_maps=8,
        num_context_per_joint=2,
        pose_net_version='v1',
        output_poses=False,
        weighted_merge=True,
        ar_pose_weights=None,
        ar_visual_weights=None,
        full_trainable=False):

    inp = Input(shape=(num_frames,) + input_shape)
    outputs = []

    if pose_dim == 2:
        y, p, hs, xb1 = _get_2d_pose_estimation_from_model(inp, model_pe,
                num_joints, num_blocks, num_context_per_joint,
                full_trainable=full_trainable)
    elif pose_dim == 3:
        y, p, hs, xb1 = _get_3d_pose_estimation_from_model(inp, model_pe,
                num_joints, num_blocks, depth_maps,
                full_trainable=full_trainable)

    if output_poses:
        outputs.append(y)
        outputs.append(p)

    model_pose = build_pose_model(num_joints, num_actions, num_frames,
            pose_dim=pose_dim, include_top=False, name='PoseAR',
            network_version=pose_net_version)
    # model_pose.trainable = False
    if ar_pose_weights is not None:
        model_pose.load_weights(ar_pose_weights)
    out_pose = model_pose([y, p])

    f = kronecker_prod(hs, xb1)
    num_features = K.int_shape(f)[-1]
    model_vis = build_visual_model(num_joints, num_actions, num_features,
            num_temp_frames=num_frames, include_top=False, name='GuidedVisAR')
    # model_vis.trainable = False
    if ar_visual_weights is not None:
        model_vis.load_weights(ar_visual_weights)
    out_vis = model_vis(f)

    for i in range(len(out_pose)):
        outputs.append(action_top(out_pose[i], name='p%d' % (i+1)))

    for i in range(len(out_vis)):
        outputs.append(action_top(out_vis[i], name='v%d' % (i+1)))

    p = out_pose[-1]
    v = out_vis[-1]

    def _heatmap_weighting(inp):
        num_filters = K.int_shape(inp)[-1]
        conv = SeparableConv2D(num_filters, (1, 1),
                use_bias=False)
        x = conv(inp)
        w = conv.get_weights()
        w[0].fill(1.)
        w[1].fill(0)
        for i in range(num_filters):
            w[1][0, 0, i, i] = 1.
        conv.set_weights(w)

        return x

    if weighted_merge:
        p = _heatmap_weighting(p)
        v = _heatmap_weighting(v)

    m = add([p, v])
    outputs.append(action_top(m, name='m'))

    model = Model(inp, outputs)

    return model


def compile(model, lr=0.001, momentum=0.95, loss_weights=None,
        pose_predicted=False):

    if pose_predicted:
        losses = []
        losses.append(elasticnet_loss_on_valid_joints)
        losses.append('binary_crossentropy')
        for i in range(len(model.output) - 2):
            losses.append('categorical_crossentropy')

        model.compile(loss=losses,
                optimizer=SGD(lr=lr, momentum=momentum, nesterov=True),
                loss_weights=loss_weights)
    else:
        model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=lr, momentum=momentum, nesterov=True),
                metrics=['acc'], loss_weights=loss_weights)

