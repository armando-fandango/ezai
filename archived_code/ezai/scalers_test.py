import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

import unittest
from . import scalers

# Make some data to be used many times
rs = np.random.RandomState(0)
n_cols = 4
n_rows = 10
offsets = rs.uniform(-1, 1, size=n_cols)
scales = rs.uniform(1, 10, size=n_cols)
X_2d = rs.randn(n_rows, n_cols) * scales + offsets
X_1row = X_2d[0, :].reshape(1, n_cols)
X_1col = X_2d[:, 0].reshape(n_rows, 1)
X_list_1row = X_1row.tolist()
X_list_1col = X_1col.tolist()

class TestScalers(unittest.TestCase):


    def test_standard_scaler_1d(self):
        # Test scaling of dataset along single axis
        for X in [X_1row, X_1col, X_list_1row, X_list_1row]:

            scaler = scalers.StandardScaler()
            X_scaled = scaler.fit(X).transform(X, copy=True)

            if isinstance(X, list):
                X = np.array(X)  # cast only after scaling done

            if X.shape[0] == 1:
                assert_almost_equal(scaler._mean, X.ravel())
                assert_almost_equal(scaler._sd, np.ones(n_cols))
                assert_array_almost_equal(X_scaled.mean(axis=0),
                                          np.zeros_like(n_cols))
                assert_array_almost_equal(X_scaled.std(axis=0),
                                          np.zeros_like(n_cols))
            else:
                assert_almost_equal(scaler._mean, X.mean())
                assert_almost_equal(scaler._sd, X.std())
                assert_array_almost_equal(X_scaled.mean(axis=0),
                                          np.zeros_like(n_cols))
                assert_array_almost_equal(X_scaled.mean(axis=0), .0)
                assert_array_almost_equal(X_scaled.std(axis=0), 1.)

            # check inverse transform
            X_scaled_back = scaler.inverse_transform(X_scaled)
            assert_array_almost_equal(X_scaled_back, X)

        # Constant feature
        X = np.ones((n_rows,n_cols))
        scaler = scalers.StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)
        assert_almost_equal(scaler._mean, 1.)
        assert_almost_equal(scaler._sd, 1.)
        assert_array_almost_equal(X_scaled.mean(axis=0), .0)
        assert_array_almost_equal(X_scaled.std(axis=0), .0)

    def test_standard_scaler_dtype(self):
        # Ensure scaling does not affect dtype
        for dtype in [np.float16, np.float32, np.float64]:
            X = X_2d.copy().astype(dtype)
            scaler = scalers.StandardScaler()
            X_scaled = scaler.fit(X).transform(X)
            assert X.dtype == X_scaled.dtype
            assert scaler._mean.dtype == np.float64 # may cause numerical issues if of X.dtype
            assert scaler._sd.dtype == np.float64

    def test_standard_scaler_2d(self):

        X = X_2d.copy()

        X[:, 0] = 0.0
        scaler = scalers.StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)
        assert not np.any(np.isnan(X_scaled))

        assert_array_almost_equal(X_scaled.mean(axis=0), n_cols * [0.0])
        assert_array_almost_equal(X_scaled.std(axis=0), [0.] + (n_cols-1) * [1.])
        # Check that X has been copied
        assert X_scaled is not X

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled, copy=True)
        assert X_scaled_back is not X
        assert X_scaled_back is not X_scaled
        assert_array_almost_equal(X_scaled_back, X)

        X_scaled = scaler.fit(X).transform(X, copy=False)
        assert not np.any(np.isnan(X_scaled))
        assert_array_almost_equal(X_scaled.mean(axis=0), n_cols * [0.0])
        assert_array_almost_equal(X_scaled.std(axis=0), [0.] + (n_cols-1) * [1.])
        # Check that X has not been copied
        assert X_scaled is X

        X[:, 0] = 1.0
        scaler = scalers.StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)
        assert not np.any(np.isnan(X_scaled))
        assert_array_almost_equal(X_scaled.mean(axis=0), n_cols * [0.0])
        assert_array_almost_equal(X_scaled.std(axis=0), [0.] + (n_cols-1) * [1.])
        # Check that X has not been copied
        assert X_scaled is not X


    def test_scaler_float16_overflow(self):
        # Test if the npscaler will not overflow on float16 numpy arrays
        rs = np.random.RandomState(0)
        # float16 has a maximum of 65500.0. On the worst case 5 * 200000 is 100000
        # which is enough to overflow the data type
        X = rs.uniform(n_rows, n_cols, [200000, 1]).astype(np.float16)

        with np.errstate(over='raise'):
            scaler = scalers.StandardScaler()
            X_scaled = scaler.fit(X).transform(X)

        # Calculate the float64 equivalent to verify result
        X_scaled_f64 = scalers.StandardScaler().fit_transform(X.astype(np.float64))

        # Overflow calculations may cause -inf, inf, or nan. Since there is no nan
        # input, all of the outputs should be finite. This may be redundant since a
        # FloatingPointError exception will be thrown on overflow above.
        assert np.all(np.isfinite(X_scaled))

        # The normal distribution is very unlikely to go above 4. At 4.0-8.0 the
        # float16 precision is 2^-8 which is around 0.004. Thus only 2 decimals are
        # checked to account for precision differences.
        assert_array_almost_equal(X_scaled, X_scaled_f64, decimal=2)

    def test_min_max_scaler_1d(self):
        # Test scaling of dataset along single axis
        for X in [X_1row, X_1col, X_list_1row, X_list_1row]:

            scaler = scalers.MinMaxScaler()
            X_scaled = scaler.fit(X).transform(X, copy=True)

            if isinstance(X, list):
                X = np.array(X)  # cast only after scaling done

            if X.shape[0] == 1:
                assert_array_almost_equal(X_scaled.min(axis=0),
                                          np.zeros(n_cols))
                assert_array_almost_equal(X_scaled.max(axis=0),
                                          np.zeros(n_cols))
            else:
                assert_array_almost_equal(X_scaled.min(axis=0), .0)
                assert_array_almost_equal(X_scaled.max(axis=0), 1.)

            # check inverse transform
            X_scaled_back = scaler.inverse_transform(X_scaled, copy=True)
            assert_array_almost_equal(X_scaled_back, X)

        # Constant feature
        X = np.ones((5, 1))
        scaler = scalers.MinMaxScaler()
        X_scaled = scaler.fit(X).transform(X,copy=True)
        assert X_scaled.min() >= 0.
        assert X_scaled.max() <= 1.

    def test_min_max_scaler_2d(self):

        X = X_2d.copy()
        scaler = scalers.MinMaxScaler()
        # default params
        X_trans = scaler.fit_transform(X, copy=True)
        assert_array_almost_equal(X_trans.min(axis=0), 0)
        assert_array_almost_equal(X_trans.max(axis=0), 1)
        X_trans_inv = scaler.inverse_transform(X_trans, copy=True)
        assert_array_almost_equal(X, X_trans_inv)

        # not default params: min=1, max=2
        scaler = scalers.MinMaxScaler()
        X_trans = scaler.fit_transform(X,x_range=(1, 2),copy=True)
        assert_array_almost_equal(X_trans.min(axis=0), 1)
        assert_array_almost_equal(X_trans.max(axis=0), 2)
        X_trans_inv = scaler.inverse_transform(X_trans,copy=True)
        assert_array_almost_equal(X, X_trans_inv)

        # min=-.5, max=.6
        scaler = scalers.MinMaxScaler()
        X_trans = scaler.fit_transform(X,x_range=(-.5, .6),copy=True)
        assert_array_almost_equal(X_trans.min(axis=0), -.5)
        assert_array_almost_equal(X_trans.max(axis=0), .6)
        X_trans_inv = scaler.inverse_transform(X_trans, copy=True)
        assert_array_almost_equal(X, X_trans_inv)

        # raises on invalid range
        #npscaler = MinMaxScaler(feature_range=(2, 1))
        #with pytest.raises(ValueError):
        #    npscaler.fit(X)

if __name__ == '__main__':
    unittest.main()

