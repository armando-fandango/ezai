from tensorflow.keras import backend as K
from tensorflow.keras import losses as k_losses

import six
from .constants import _EPSILON
from .util import np_util as npu

def smape_k(y_true, y_pred):
    diff = K.abs(y_true - y_pred) / K.clip(K.abs(y_true) + K.abs(y_pred),
                                           _EPSILON, None)
    return 100. * K.mean(diff, axis=-1)


mse_k = k_losses.mse
mae_k = k_losses.mae
mape_k = k_losses.mape


smape_np = npu.smape
mse_np = npu.mse
mae_np = npu.mae
mape_np = npu.mape

def getfunc(fname, objects=globals()):
    if isinstance(fname, six.string_types):
        fn = objects.get(str(fname))
    elif callable(fname):
        fn = fname
    else:
        fn = None

    if fn is None:
        raise (ValueError('No such function: {0}'.format(fname)))
    else:
        return fn
