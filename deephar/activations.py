from keras import backend as K

def channel_softmax_2d(alpha=1):

    def _channel_softmax_2d(x):
        assert K.ndim(x) in [4, 5], \
                'Input tensor must have ndim 4 or 5 ({})'.format(K.ndim(x))

        if alpha != 1:
            x = alpha * x
        e = K.exp(x - K.max(x, axis=(-3, -2), keepdims=True))
        s = K.clip(K.sum(e, axis=(-3, -2), keepdims=True), K.epsilon(), None)

        return e / s

    return _channel_softmax_2d

def channel_softmax_1d():

    def _channel_softmax_1d(x):
        ndim = K.ndim(x)
        if ndim == 3:
            e = K.exp(x - K.max(x, axis=(1,), keepdims=True))
            s = K.sum(e, axis=(1,), keepdims=True)
            return e / s
        else:
            raise ValueError('This function is specific for 3D tensors. '
                    'Here, ndim=' + str(ndim))

    return _channel_softmax_1d
