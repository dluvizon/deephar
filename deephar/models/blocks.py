import tensorflow as tf

from keras.models import Model

from deephar.layers import *
from deephar.utils import *


def conv_block(inp, kernel_size, filters, last_act=True):

    filters1, filters2, filters3 = filters

    x = conv_bn_act(inp, filters1, (1, 1))
    x = conv_bn_act(x, filters2, kernel_size)
    x = conv_bn(x, filters3, (1, 1))

    shortcut = conv_bn(inp, filters3, (1, 1))
    x = add([x, shortcut])
    if last_act:
        x = Activation('relu')(x)

    return x

def identity_block(inp, kernel_size, filters, last_act=True):

    filters1, filters2, filters3 = filters

    x = conv_bn_act(inp, filters1, (1, 1))
    x = conv_bn_act(x, filters2, kernel_size)
    x = conv_bn(x, filters3, (1, 1))

    x = add([x, inp])
    if last_act:
        x = Activation('relu')(x)

    return x


def stem_inception_v4(x, image_div=8):
    """Entry-flow network (stem) *based* on Inception_v4."""

    assert image_div in [4, 8, 16, 32], \
            'Invalid image_div ({}).'.format(image_div)

    x = conv_bn_act(x, 32, (3, 3), strides=(2, 2))
    x = conv_bn_act(x, 32, (3, 3))
    if image_div is 32:
        x = MaxPooling2D((2, 2))(x)
    x = conv_bn_act(x, 64, (3, 3))

    a = conv_bn_act(x, 96, (3, 3), strides=(2, 2))
    b = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([a, b])

    a = conv_bn_act(x, 64, (1, 1))
    a = conv(a, 96, (3, 3))
    b = conv_bn_act(x, 64, (1, 1))
    b = conv_bn_act(b, 64, (5, 1))
    b = conv_bn_act(b, 64, (1, 5))
    b = conv(b, 96, (3, 3))
    x = concatenate([a, b])
    x = BatchNormalization(axis=-1, scale=False)(x)

    if image_div is not 4:
        a = act_conv_bn(x, 192, (3, 3), strides=(2, 2))
        b = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = concatenate([a, b])

    if image_div in [16, 32]:
        a = act_conv_bn(x, 192, (3, 3), strides=(2, 2))
        b = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = concatenate([a, b])

    if image_div is 4:
        x = residual(x, int_size=112, out_size=2*192+64, convtype='normal',
                name='residual0')
    else:
        x = residual(x, int_size=144, out_size=3*192, convtype='normal',
                name='residual0')

    return x


def stem_residual_eccv(x, image_div=8):
    """Entry-flow network (stem) *based* on ResNet ('residual' option)."""

    assert image_div in [4, 8, 16, 32], \
            'Invalid image_div ({}).'.format(image_div)

    x = conv_bn_act(x, 64, (7, 7), strides=(2, 2), padding='same')
    a = conv_bn_act(x, 128, (3, 3), padding='same')
    b = conv_bn_act(x, 128, (1, 1), padding='same')
    x = add([a, b])
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = residual(x, int_size=128, out_size=256, convtype='normal', name='rn0')
    x = residual(x, int_size=128, out_size=256, convtype='normal', name='rn1')

    if image_div is 4:
        x = residual(x, out_size=256, convtype='normal', name='rn3')

    else:
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = residual(x, int_size=192, out_size=384, convtype='normal',
                name='rn3')
        x = residual(x, int_size=192, out_size=384, convtype='normal',
                name='rn4')

        if image_div in [16, 32]:
            x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
            x = residual(x, int_size=256, out_size=512, convtype='normal',
                    name='rn5')
            x = residual(x, int_size=256, out_size=512, convtype='normal',
                    name='rn6')

            if image_div is 32:
                x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    return x



def reception_block(x, num_levels, kernel_size, int_size=None,
        convtype='depthwise', name=None):

    def hourglass(x, n):
        up1 = residual(x, kernel_size=kernel_size, int_size=int_size,
                convtype=convtype)

        low = MaxPooling2D((2, 2))(x)

        if n == num_levels:
            low = act_conv_bn(low, int(K.int_shape(x)[-1] / 2), (1, 1))
        low = residual(low, kernel_size=kernel_size, int_size=int_size,
                convtype=convtype)

        if n > 2:
            low = hourglass(low, n-1)
        else:
            low = residual(low, kernel_size=kernel_size,
                    int_size=int_size,
                    convtype=convtype)

        if n == num_levels:
            low = residual(low, kernel_size=kernel_size,
                    out_size=K.int_shape(x)[-1], int_size=int_size,
                    convtype=convtype)
        else:
            low = residual(low, kernel_size=kernel_size,
                    int_size=int_size, convtype=convtype)

        up2 = UpSampling2D((2, 2))(low)

        x = add([up1, up2])

        return x

    x = hourglass(x, num_levels)

    return x


def build_keypoints_regressor(input_shape, dim, num_maps, sam_model, prob_model,
        name=None, verbose=0):

    assert num_maps >= 1, \
            'The number of maps should be at least 1 (%d given)' % num_maps

    inputs = []
    inputs3d = []
    p_concat = []
    v_concat = []

    # Auxiliary functions
    v_tile = Lambda(lambda x: K.tile(x, (1, 1, dim)))
    # This depends on TensorFlow because keras does not implement divide.
    tf_div = Lambda(lambda x: tf.divide(x[0], x[1]))

    for i in range(num_maps):
        h = Input(shape=input_shape)
        inputs.append(h)
        h_s = act_channel_softmax(h)
        p = sam_model(h_s)
        v = prob_model(h_s)

        if dim == 3:
            d = Input(shape=input_shape)
            inputs3d.append(d)
            d_s = Activation('sigmoid')(d)
            dm = multiply([d_s, h_s])
            z = Lambda(lambda x: K.sum(x, axis=(1, 2)))(dm)
            z = Lambda(lambda x: K.expand_dims(x, axis=-1))(z)
            p = concatenate([p, z])

        if num_maps > 1:
            t = v_tile(v)
            p = multiply([p, v_tile(v)])

        p_concat.append(p)
        v_concat.append(v)

    if num_maps > 1:
        p = add(p_concat)
        v_sum = add(v_concat)
        p = tf_div([p, v_tile(v_sum)])
        v = maximum(v_concat)
    else:
        p = p_concat[0]
        v = v_concat[0]

    model = Model(inputs+inputs3d, [p, v], name=name)
    if verbose:
        model.summary()

    return model


def build_context_aggregation(num_joints, num_context, alpha,
        num_frames=1, name=None):

    inp = Input(shape=(num_joints * num_context, 1))
    d = Dense(num_joints, use_bias=False)

    x = Lambda(lambda x: K.squeeze(x, axis=-1))(inp)
    x = d(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    w = d.get_weights()
    w[0].fill(0)
    for j in range(num_joints):
        start = j*num_context
        w[0][j * num_context : (j + 1) * num_context, j] = 1.
    d.set_weights(w)
    d.trainable = False

    ctx_sum = Model(inputs=inp, outputs=x)
    ctx_sum.trainable = False
    if num_frames > 1:
        ctx_sum = TimeDistributed(ctx_sum,
                input_shape=(num_frames,) + K.int_shape(inp)[1:])

    # Define auxiliary layers.
    mul_alpha = Lambda(lambda x: alpha * x)
    mul_1alpha = Lambda(lambda x: (1 - alpha) * x)

    # This depends on TensorFlow because keras does not implement divide.
    tf_div = Lambda(lambda x: tf.divide(x[0], x[1]))

    if num_frames == 1:
        # Define inputs
        ys = Input(shape=(num_joints, 2))
        yc = Input(shape=(num_joints * num_context, 2))
        pc = Input(shape=(num_joints * num_context, 1))

        # Split contextual predictions in x and y and do computations separately
        xi = Lambda(lambda x: x[:,:, 0:1])(yc)
        yi = Lambda(lambda x: x[:,:, 1:2])(yc)
    else:
        ys = Input(shape=(num_frames, num_joints, 2))
        yc = Input(shape=(num_frames, num_joints * num_context, 2))
        pc = Input(shape=(num_frames, num_joints * num_context, 1))

        # Split contextual predictions in x and y and do computations separately
        xi = Lambda(lambda x: x[:,:,:, 0:1])(yc)
        yi = Lambda(lambda x: x[:,:,:, 1:2])(yc)

    pxi = multiply([xi, pc])
    pyi = multiply([yi, pc])

    pc_sum = ctx_sum(pc)
    pxi_sum = ctx_sum(pxi)
    pyi_sum = ctx_sum(pyi)
    pc_div = Lambda(lambda x: x / num_context)(pc_sum)
    pxi_div = tf_div([pxi_sum, pc_sum])
    pyi_div = tf_div([pyi_sum, pc_sum])
    yc_div = concatenate([pxi_div, pyi_div])

    ys_alpha = mul_alpha(ys)
    yc_div_1alpha = mul_1alpha(yc_div)

    y = add([ys_alpha, yc_div_1alpha])

    model = Model(inputs=[ys, yc, pc], outputs=y, name=name)
    model.trainable = False

    return model


def build_softargmax_1d(input_shape, name=None):

    if name is None:
        name_sm = None
    else:
        name_sm = name + '_softmax'

    inp = Input(shape=input_shape)
    x = act_depth_softmax(inp, name=name_sm)

    x = lin_interpolation_1d(x)

    model = Model(inputs=inp, outputs=x, name=name)
    model.trainable = False

    return model


def build_softargmax_2d(input_shape, rho=0., name=None):

    if name is None:
        name_sm = None
    else:
        name_sm = name + '_softmax'

    inp = Input(shape=input_shape)
    x = act_channel_softmax(inp, name=name_sm)
    if rho > 0:
        x = kl_divergence_regularizer(x, rho=rho)

    x_x = lin_interpolation_2d(x, axis=0)
    x_y = lin_interpolation_2d(x, axis=1)
    x = concatenate([x_x, x_y])

    model = Model(inputs=inp, outputs=x, name=name)
    model.trainable = False

    return model


def build_joints_probability(input_shape, name=None, verbose=0):

    inp = Input(shape=input_shape)

    x = inp
    x = AveragePooling2D((2, 2), strides=(1, 1))(x)
    x = Lambda(lambda x: 4*x)(x)
    x = GlobalMaxPooling2D()(x)

    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    model = Model(inputs=inp, outputs=x, name=name)
    if verbose:
        model.summary()

    return model

