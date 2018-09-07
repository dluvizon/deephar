import tensorflow as tf

from keras.models import Model

from deephar.layers import *
from deephar.utils import *


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

    x_x = lin_interpolation_2d(x, dim=0)
    x_y = lin_interpolation_2d(x, dim=1)
    x = concatenate([x_x, x_y])

    model = Model(inputs=inp, outputs=x, name=name)
    model.trainable = False

    return model


def build_joints_probability(input_shape, name=None):

    num_rows, num_cols = input_shape[0:2]
    inp = Input(shape=input_shape)

    x = MaxPooling2D((num_rows, num_cols))(inp)
    x = Activation('sigmoid')(x)

    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    model = Model(inputs=inp, outputs=x, name=name)

    return model


