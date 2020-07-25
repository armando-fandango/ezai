# unit_axis =1 means one column and unit_axis = 0 means one row
import numpy as np
from ..constants import _EPSILON

"""
Notes:

expand_dims

Expands the numpy array by adding a unit axis
Refer to
https://medium.com/@zhang_yang/reshaping-numpy-arrays-newaxis-v-s-reshape-v-s-expand-dims-521289b73c8a

y = expand_dims(x,axis=x.ndims-1)  #  : to :,1
#np.array(x)[np.newaxis, :] 

y = expand_dims(x,axis=0) # : to 1,:
#np.array(x)[:,np.newaxis]
"""

def shift(a, n, fill_value=np.nan, copy = False):
    """
    :param a: numpy array to be shifted
    :param n: shift by number of rows +ve means push to bottom, -ve pull up
    :param fill_value: fill the shifted cells with this value
    :return:  shifted array
    """
    if copy:
        a = a.copy()
    if n == 0:
        a = a
    else:
        #result = np.full_like(a, fill_value)
        if n > 0:
            a[n:] = a[:-n]
            a[:n] = np.full_like(a[:n],fill_value)
        else:
            a[:n] = a[-n:]
            a[n:] = np.full_like(a[n:],fill_value)
    return a

def zero2one(x, copy=False):
    if isinstance(x, np.ndarray):
        if copy:
            x = x.copy()
        x[x == 0.0] = 1.0
    elif np.isscalar(x) and x== 0.0:
        # if we are fitting on 1D arrays
        x = 1.0
    return x

# Metrics

def smape(y_true, y_pred):
    """
    based on chen and yang's formulae on https://robjhyndman.com/hyndsight/smape/
    -- removed multiply by 2 to get value between 0,1
    -- multiplied by 100 to get percentage
    :param y_true:
    :param y_pred:
    :return:
    """
    diff = np.abs(y_true - y_pred) / np.clip(np.abs(y_true) + np.abs(y_pred),
                                             _EPSILON, None)
    return 100. * np.mean(diff)


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def mape(y_true, y_pred):
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true),
                                              _EPSILON, None))
    return 100. * np.mean(diff)