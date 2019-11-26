from keras.models import Model

from deephar.utils import *
from deephar.layers import *

from ..activations import channel_softmax_2d


def concat_tensorlist(t):
    assert isinstance(t, list), 't should be a list, got ({})'.format(t)

    if len(t) > 1:
        return concatenate(t)
    return t[0]


def add_tensorlist(t):
    assert isinstance(t, list), 't should be a list, got ({})'.format(t)

    if len(t) > 1:
        return add(t)
    return t[0]


def residual_unit(x, kernel_size, strides=(1, 1), out_size=None,
        convtype='depthwise', shortcut_act=True,
        features_div=2, name=None):
    """(Separable) Residual Unit implementation.
    """
    assert convtype in ['depthwise', 'normal'], \
            'Invalid convtype ({}).'.format(convtype)

    num_filters = K.int_shape(x)[-1]
    if out_size is None:
        out_size = num_filters

    skip_conv = (num_filters != out_size) or (strides != (1, 1))

    if skip_conv:
        x = BatchNormalization(name=appstr(name, '_bn1'))(x)

    shortcut = x
    if skip_conv:
        if shortcut_act:
            shortcut = relu(shortcut, name=appstr(name, '_shortcut_act'))
        shortcut = conv2d(shortcut, out_size, (1, 1), strides=strides,
                name=appstr(name, '_shortcut_conv'))

    if not skip_conv:
        x = BatchNormalization(name=appstr(name, '_bn1'))(x)
    x = relu(x, name=appstr(name, '_act1'))

    if convtype == 'depthwise':
        x = sepconv2d(x, out_size, kernel_size, strides=strides,
                name=appstr(name, '_conv1'))
    else:
        x = conv2d(x, int(out_size / features_div), (1, 1),
                name=appstr(name, '_conv1'))
        middle_bn_name = appstr(name, '_bn2')
        x = BatchNormalization(name=middle_bn_name)(x)
        x = relu(x, name=appstr(name, '_act2'))
        x = conv2d(x, out_size, kernel_size, strides=strides,
                name=appstr(name, '_conv2'))

    x = add([shortcut, x])

    return x


def downscaling_unit(x, cfg, out_size=None, name=None):
    """Downscaling Unit using depth wise separable convolutions"""

    kernel_size = cfg.kernel_size
    downsampling_type = cfg.downsampling_type

    if out_size is None:
        out_size = K.int_shape(x)[-1]

    s1 = (2, 2) if downsampling_type == 'conv' else (1, 1)
    if downsampling_type == 'maxpooling':
        x = maxpooling2d(x, (2, 2))

    x = residual_unit(x, kernel_size, out_size=out_size, strides=s1,
            name=appstr(name, '_r0'))

    return x


def upscaling_unit(x, cfg, out_size=None, name=None):
    """Upscaling Unit using depth wise separable convolutions"""

    kernel_size = cfg.kernel_size
    downsampling_type = cfg.downsampling_type

    if out_size is None:
        out_size = K.int_shape(x)[-1]

    if downsampling_type == 'maxpooling':
        x = upsampling2d(x, (2, 2))
        x = residual_unit(x, kernel_size, out_size=out_size,
                name=appstr(name, '_r0'))
    else:
        x = BatchNormalization(name=appstr(name, '_bn1'))(x)
        x = relu(x, name=appstr(name, '_act1'))
        x = conv2dtranspose(x, out_size, (2, 2), strides=(2, 2),
                name=appstr(name, '_convtrans1'))

    return x


def set_trainable_layers(model, keyword, pos_trainable, neg_trainable=None):

    def trainable_flag(curr, newval):
        return newval if newval in [True, False] else curr

    for i in range(len(model.layers)):
        name = model.layers[i].name
        if '_xy_x' in name or '_xy_y' in name \
                or '_xy2_x' in name or '_xy2_y' in name:
            warning('Unchanged layer {}'.format(name))
            continue

        if keyword in name:
            model.layers[i].trainable = \
                    trainable_flag(model.layers[i].trainable, pos_trainable)
        else:
            model.layers[i].trainable = \
                    trainable_flag(model.layers[i].trainable, neg_trainable)


def copy_replica_layers(model):
    for i in range(len(model.layers)):
        if '_replica' in model.layers[i].name:
            rname = model.layers[i].name
            lname = rname.split('_replica')[0]
            worg = model.get_layer(lname).get_weights()
            wrep = model.get_layer(rname).get_weights()
            wrep[0][:] = worg[0][:]
            model.get_layer(rname).set_weights(wrep)


def compile_model(model, loss, optimizer, loss_weights=None):

    nout = len(model.outputs)
    if loss_weights is not None:
        if isinstance(loss_weights, list):
            assert len(loss_weights) == nout, \
                    'loss_weights incompatible with model'
        else:
            loss_weights = nout*[loss_weights]

    if isinstance(loss, list):
        assert nout == len(loss), 'loss not corresponding to the model outputs'

    model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights)


# Aliases.
residual = residual_unit
downscaling = downscaling_unit
upscaling = upscaling_unit

