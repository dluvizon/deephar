from keras import backend as K

def elasticnet_loss_on_valid_joints(y_true, y_pred):
    idx = K.cast(K.greater(y_true, -1e6), 'float32')
    y_true = idx * y_true
    y_pred = idx * y_pred
    l1 = K.sum(K.abs(y_pred - y_true), axis=(-2, -1))
    l2 = K.sum(K.square(y_pred - y_true), axis=(-2, -1))
    return l1 + l2

