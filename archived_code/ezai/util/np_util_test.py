import numpy as np

from numpy.testing import assert_array_equal

import unittest
from . import np_util
import ddt
#from ddt import ddt,data,unpack

data1D = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1)
data2D = np.array([[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19]]).transpose()

@ddt.ddt
class Test_np_util(unittest.TestCase):

    """
        def test_zero2one_array_copy(self):
            s1 = np.array([0, 1, 2, 3])
            s2 = nputil.zero2one(s1, copy=True)

            self.assertFalse(s1[0] == s2[0])
            assert_array_equal(s1, np.array([0, 1, 2, 3]))
            assert_array_equal(s2, np.array([1, 1, 2, 3]))

        def test_zero2one_array_inplace(self):
            s1 = np.array([0, 1, 2, 3])
            s2 = nputil.zero2one(s1)  # default copy=False

            self.assertTrue(s1[0] == s2[0])
            assert_array_equal(s1, np.array([1, 1, 2, 3]))
            assert_array_equal(s2, np.array([1, 1, 2, 3]))

        def test_zero2one_scalar(self):
            s1 = 0
            s2 = nputil.zero2one(s1)

            self.assertFalse(s1 == s2)
    """
    @ddt.data(
        {'a':data1D.copy(), 'n':0, 'fill_value':99,
         'e_b': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1)},
        {'a':data1D.copy(), 'n':1, 'fill_value':99,
         'e_b': np.array([99, 0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1,1)},
        {'a':data1D.copy(), 'n':-1, 'fill_value':99,
         'e_b': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 99]).reshape(-1,1)},
    )
    @ddt.unpack
    def test_shift_inplace(self,a,n,fill_value,e_b,copy=False):
        b = np_util.shift(a, n, fill_value, copy)
        self.assertIs(a,b)
        assert_array_equal(b,e_b)

    @ddt.data(
        {'a':data1D.copy(), 'n':0, 'fill_value':99,
         'e_b': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1)},
        {'a':data1D.copy(), 'n':1, 'fill_value':99,
         'e_b': np.array([99, 0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1,1)},
        {'a':data1D.copy(), 'n':-1, 'fill_value':99,
         'e_b': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 99]).reshape(-1,1)},
    )
    @ddt.unpack
    def test_shift_copy(self,a,n,fill_value,e_b,copy=True):
        b = np_util.shift(a, n, fill_value, copy)
        self.assertIsNot(a,b)
        assert_array_equal(b,e_b)


if __name__ == '__main__':
    unittest.main()

