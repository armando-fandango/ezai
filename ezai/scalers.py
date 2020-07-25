import numpy as np
from . import np_util

class StandardScaler():
    def __init__(self):
        self._fitted = False

    def fit(self, x):
        self._mean = np.nanmean(x, axis=0, dtype=np.float64)
        self._sd = np_util.zero2one(np.nanstd(x, axis=0, dtype=np.float64))
        self._fitted=True
        return self

    def transform(self, x, copy = False):
        if self._fitted:
            if copy:
                x = x.copy()
            x -= self._mean
            x /= self._sd
        else:
            raise(ValueError('npscaler not fitted'))
        return x

    def fit_transform(self, x, copy = False):
        return self.fit(x).transform(x, copy)


    def inverse_transform(self, x, copy = False):
        if self._fitted:
            if copy:
                x = x.copy()
            x *= self._sd
            x += self._mean
        else:
            raise(ValueError('npscaler not fitted'))
        return x

class MinMaxScaler():
    def __init__(self):
        self._fitted = False

    def fit(self, x, x_range=(0,1)):
        self._x_range=x_range
        self._x_min = np.nanmin(x, axis=0)
        self._x_max = np.nanmax(x, axis=0)
        self._scale = (self._x_range[1] - self._x_range[0]) / np_util.zero2one(self._x_max - self._x_min)
        self._add_val = self._x_range[0] - self._x_min * self._scale
        self._fitted=True
        return self

    def transform(self, x, copy = False):
        if self._fitted:
            if copy:
                x = x.copy()
            x *= self._scale
            x += self._add_val
        else:
            raise(ValueError('npscaler not fitted'))
        return x

    def fit_transform(self, x, x_range=(0,1), copy = False):
        return self.fit(x, x_range).transform(x, copy)


    def inverse_transform(self, x, copy = False):
        if self._fitted:
            if copy:
                x = x.copy()
            x -= self._add_val
            x /= self._scale
        else:
            raise(ValueError('npscaler not fitted'))
        return x
