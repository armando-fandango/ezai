import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal

import unittest
import ddt

from . import temporal

np1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
np2d = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]).transpose()

n_rows = 10
idx = pd.date_range('2020-01-01', periods=n_rows, freq='H')
s = pd.Series(np.linspace(1,10,n_rows), index=idx)
df1d = pd.DataFrame({'col1':np.linspace(1,10,n_rows)},
                    index=idx)
df2d = pd.DataFrame({'col1':np.linspace(1,10,n_rows),'col2':np.linspace(11,20,n_rows)},
                    index=idx)
@ddt.ddt
class TestTDUtil(unittest.TestCase):

    #def td_to_xy(td, n_tx=1, n_ty=1, x_idx=[0], y_idx=[0], h=1):
    @ddt.data(
        # iterative or recursive strategy n_x_vars = input window
        {'data': np1d, 'n_tx': 1, 'n_ty': 1, 'h':1,
         'x': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8],dtype=np.float32).reshape(-1,1),
         'y': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9],dtype=np.float32).reshape(-1,1)
         },
        {'data': np2d, 'n_tx': 1, 'n_ty': 1, 'h':1,
         'x': np.array([[0, 10],[1,11], [2,12], [3,13],[4,14],[5,15], [6,16], [7,17],[8,18]],dtype=np.float32),
         'y': np.array([[1,11], [2,12], [3,13],[4,14],[5,15], [6,16], [7,17],[8,18],[9,19]],dtype=np.float32)
         },
        {'data': np2d, 'n_tx': 2, 'n_ty': 1, 'h':1,
         'x': np.array([[0, 10,1,11],[1,11,2,12], [2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17], [7,17,8,18]],dtype=np.float32),
         'y': np.array([[2,12], [3,13],[4,14],[5,15], [6,16], [7,17],[8,18],[9,19]],dtype=np.float32)
         },
        # MO strategy n_y = output window
        {'data': np2d, 'n_tx': 1, 'n_ty': 2, 'h':1,
         'x': np.array([[0, 10],[1,11], [2,12], [3,13],[4,14],[5,15], [6,16], [7,17]],dtype=np.float32),
         'y': np.array([[1,11,2,12], [2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17], [7,17,8,18],[8,18,9,19]],dtype=np.float32)
         },
        {'data': np2d, 'n_tx': 2, 'n_ty': 2, 'h':1,
         'x': np.array([[0, 10,1,11],[1,11,2,12], [2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17]],dtype=np.float32),
         'y': np.array([[2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17], [7,17,8,18],[8,18,9,19]],dtype=np.float32)
         },
        # increase h for direct strategy i.e. h=2 here
        {'data': np2d, 'n_tx': 2, 'n_ty': 2, 'h':2,
         'x': np.array([[0, 10,1,11],[1,11,2,12], [2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16]],dtype=np.float32),
         'y': np.array([[3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17], [7,17,8,18],[8,18,9,19]],dtype=np.float32)
         },
        {'data': np1d, 'n_tx': 1, 'n_ty': 1, 'h':2,
         'x': np.array([0, 1, 2, 3, 4, 5, 6, 7],dtype=np.float32).reshape(-1,1),
         'y': np.array([2, 3, 4, 5, 6, 7, 8, 9],dtype=np.float32).reshape(-1,1)
         },
        {'data': np2d, 'n_tx': 1, 'n_ty': 1, 'h':2,
         'x': np.array([[0, 10],[1,11], [2,12], [3,13],[4,14],[5,15], [6,16], [7,17]],dtype=np.float32),
         'y': np.array([[2,12], [3,13],[4,14],[5,15], [6,16], [7,17],[8,18],[9,19]],dtype=np.float32)
         }
    )
    @ddt.unpack
    def test_np_to_xy(self,data,x,y,n_tx,n_ty,h):
        xy = temporal.np_to_xy(data, n_tx=n_tx, n_ty=n_ty, h=h, dim3=False)
        assert_array_equal(x,xy[0])
        assert_array_equal(y,xy[1])

        xy = temporal.np_to_xy(data, n_tx=n_tx, n_ty=n_ty, h=h, dim3=True)
        #print('data.shape={},n_tx={}, n_ty={}, h={}, td_xy[0].shape={},td_xy[1].shape={}'.format(data.shape,n_tx, n_ty, h, td_xy[0].shape,td_xy[1].shape) )
        assert_array_equal(x.reshape(-1,n_tx,data.shape[1]),xy[0])
        assert_array_equal(y.reshape(-1,n_ty,data.shape[1]),xy[1])

    """
    #def td_to_xy(td, n_tx=1, n_ty=1, x_idx=[0], y_idx=[0], h=1):
    
    @ddt.data(
        # iterative or recursive strategy n_x_vars = input window
        {'data': np1d, 'n_tx': 1, 'n_ty': 1, 'h':1,
         'x': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8],dtype=np.float32).reshape(-1,1),
         'y': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9],dtype=np.float32).reshape(-1,1)
         },
        {'data': np2d, 'n_tx': 1, 'n_ty': 1, 'h':1,
         'x': np.array([[0, 10],[1,11], [2,12], [3,13],[4,14],[5,15], [6,16], [7,17],[8,18]],dtype=np.float32),
         'y': np.array([[1,11], [2,12], [3,13],[4,14],[5,15], [6,16], [7,17],[8,18],[9,19]],dtype=np.float32)
         },
        {'data': np2d, 'n_tx': 2, 'n_ty': 1, 'h':1,
         'x': np.array([[0, 10,1,11],[1,11,2,12], [2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17], [7,17,8,18]],dtype=np.float32),
         'y': np.array([[2,12], [3,13],[4,14],[5,15], [6,16], [7,17],[8,18],[9,19]],dtype=np.float32)
         },
        # MO strategy n_y = output window
        {'data': np2d, 'n_tx': 1, 'n_ty': 2, 'h':1,
         'x': np.array([[0, 10],[1,11], [2,12], [3,13],[4,14],[5,15], [6,16], [7,17]],dtype=np.float32),
         'y': np.array([[1,11,2,12], [2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17], [7,17,8,18],[8,18,9,19]],dtype=np.float32)
         },
        {'data': np2d, 'n_tx': 2, 'n_ty': 2, 'h':1,
         'x': np.array([[0, 10,1,11],[1,11,2,12], [2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17]],dtype=np.float32),
         'y': np.array([[2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17], [7,17,8,18],[8,18,9,19]],dtype=np.float32)
         },
        # increase h for direct strategy i.e. h=2 here
        {'data': np2d, 'n_tx': 2, 'n_ty': 2, 'h':2,
         'x': np.array([[0, 10,1,11],[1,11,2,12], [2,12,3,13], [3,13,4,14],[4,14,5,15],[5,15,6,16]],dtype=np.float32),
         'y': np.array([[3,13,4,14],[4,14,5,15],[5,15,6,16], [6,16,7,17], [7,17,8,18],[8,18,9,19]],dtype=np.float32)
         },
        {'data': np1d, 'n_tx': 1, 'n_ty': 1, 'h':2,
         'x': np.array([0, 1, 2, 3, 4, 5, 6, 7],dtype=np.float32).reshape(-1,1),
         'y': np.array([2, 3, 4, 5, 6, 7, 8, 9],dtype=np.float32).reshape(-1,1)
         },
        {'data': np2d, 'n_tx': 1, 'n_ty': 1, 'h':2,
         'x': np.array([[0, 10],[1,11], [2,12], [3,13],[4,14],[5,15], [6,16], [7,17]],dtype=np.float32),
         'y': np.array([[2,12], [3,13],[4,14],[5,15], [6,16], [7,17],[8,18],[9,19]],dtype=np.float32)
         }
    )
    @ddt.unpack
    def test_df_to_xy(self,data,x,y,n_tx,n_ty,h):
    To manually test for now:
    n_rows = 10
    idx = pd.date_range('2020-01-01', periods=n_rows, freq='5T')
    s = pd.Series(np.linspace(1,10,n_rows), index=idx)
    df1d = pd.DataFrame({'col1':np.linspace(1,10,n_rows)},
                        index=idx)
    df2d = pd.DataFrame({'col1':np.linspace(1,10,n_rows),'col2':np.linspace(11,20,n_rows)},
                        index=idx)

xy_df, _ = temporal.df_to_xy(df1d,dim3=False)
ipd(xy_df)

xy_df, _ = temporal.df_to_xy(df2d,dim3=False)
ipd(xy_df)

xy_df, _ = temporal.df_to_xy(df2d,dim3=False,n_tx=2)
ipd(xy_df)

xy_df, _ = temporal.df_to_xy(df2d,dim3=False,n_ty=2)
ipd(xy_df)

xy_df, _ = temporal.df_to_xy(df2d,dim3=False,n_tx=2,n_ty=2)
ipd(xy_df)

xy_df, _ = temporal.df_to_xy(df1d,dim3=False,n_tx=1,n_ty=1,h=2)
ipd(xy_df)

xy_df, _ = temporal.df_to_xy(df2d,dim3=False, x_cols=['col1'],y_cols=['col2'])
ipd(xy_df)


    
        pass
        #xy = temporal.np_to_xy(data,n_tx=n_tx, n_ty=n_ty, h=h, dim3=False)
        #assert_array_equal(x,xy[0])
        #assert_array_equal(y,xy[1])

        #xy = temporal.np_to_xy(data,n_tx=n_tx, n_ty=n_ty, h=h, dim3=True)
        #print('data.shape={},n_tx={}, n_ty={}, h={}, td_xy[0].shape={},td_xy[1].shape={}'.format(data.shape,n_tx, n_ty, h, td_xy[0].shape,td_xy[1].shape) )
        #assert_array_equal(x.reshape(-1,n_tx,data.shape[1]),xy[0])
        #assert_array_equal(y.reshape(-1,n_ty,data.shape[1]),xy[1])
    """