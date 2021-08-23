import datetime
import itertools

import numpy as np
import pandas as pd
import dask.dataframe as dd

from archived_code.ezai.data.dataset import Dataset
from archived_code.ezai.util import np_util
from archived_code.ezai.util import log_util

logger = log_util.get_logger()

from typing import Union, Optional

dftypes = Union[pd.DataFrame, pd.Series, dd.DataFrame, dd.Series]

"""
Notes:
        n_tx and n_ty are number of x and y timesteps respectively
            if x = 1 then input is one-step
            if x > 1 then input is multi-step
            if y = 1 then output is one-step-ahead
            if y > 1 then output is multi-step-ahead
        h is prediction horizon for direct strategy: 1 to n
            returns x : {n_tx-1,...,t}, y : {t+h,...,t+h+n_ty-1}
                        t = n_t
                   for:   t-(n_tx-1),t,-1       t+h,t+h+n_ty,1
            in 3d mode should return (samples, time_steps, features)
                                        n_rows, n_tx, n_vx
        x_idx is the list of columns to be used as input or feature with length n_vx
        y_idx is the list of columns to be used as output or target with length n_vy
        total inputs n_cx = n_vx * n_tx
        total outputs n_cy = n_vy * n_ty

"""
class TemporalDataset(Dataset):
    """
    This class can contain single or multiple time series in single dataframe/table
    - Date col could be present or not - should we breakinto sequence ?
    - Date col could be index or not
    - Categorical columns - for groupby
    - Numerical columns - features/targets
    Usecases:
    - convert to XY format for MLDataset
    - find missing values
    - plotting and visualization

    design:
    df: contains dataframe
    dt_col: None means no date column or date column not set yet
            Otherwise a string that represents the date column
    freq: one of the values from
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """
    def __init__(self,
                 df: Optional[dftypes] = None,
                 dt_col: Optional[str] = None,
                 freq: Optional[str] = None,
                 copy:bool=True):
        Dataset.__init__(self)

        # Basic checks
        #  1. does dt_col exist and is datetime ?
        #  2. Is time series empty ? or has at least 3 rows ?
        if df is None:
            log_util.log_and_raise(ValueError('df is passed as NoneType'), logger)
        #if not isinstance(df,dftypes):
        #    log_util.log_and_raise(ValueError('df not one of the legal dftypes'),logger)
        if len(df)<3:
            #logger.error("ValueError: " + message)
            log_util.log_and_raise(ValueError('df must have minimum 3 rows'), logger)

        self.df = df.copy() if copy else df

        if dt_col is not None:
            self.df = self.df.set_index(dt_col).sort_index()

        self.freq = freq
        #inferred_freq = df[dt_col].inferred_freq  # Infer frequency
        #if (freq is None) and (freq != inferred_freq):
        #    log_util.log_and_raise(ValueError('The inferred frequency does not '
        #                                      'match the value of the "freq" '
        #                                      'argument.',logger))

    """ use is_unique
    def has_dup_dt(self, col=None):
        return df_util.has_dups(self.df,col)
    """
    def is_unique(self, col=None):
        dates = self.df.index if col is None else self.df[col]
        return dates.is_unique

    def infer_freq(self, col=None):
        dates =  self.df.index if col is None else self.df[col]
        return pd.infer_freq(dates)

    def dt_gap_counts(self,col=None):
        dates =  self.df.index if col is None else self.df[col]
        return dt_gap_counts(dates)

    def dt_gap_fill(self, freq, fill_value=None):
        self.df = dt_gap_fill(self.df,freq,fill_value=fill_value)


#TODO: can use existing function with stepwise slicing

def dfs_to_xy(td_x:pd.DataFrame, td_y:pd.DataFrame,
              x_cols=None, y_cols=None,
              n_tx=1, n_ty=1, n_tx_step = 1, n_ty_step=1, h=1,
              step_method = 'mean',
              dim3=True):
    """
    return x values at a freq n_tx_step
    return y values at a freq n_ty_step
    n_ty_step // n_tx_step should be zero
    e.g. 1,6 means 5 min x and 30 min y
    2,6 means 10 min x and 30 min y
    3,6 mean 15 min x and 30 min y
    4,6 would be wrong as it would be 20 min x and 30 min y
    2,3 or 3,4 also wrong same way
    2,3 or 3,4 or 4,6 could be possible but then would reduce windowing

    n_tx  >= n_ty_step/n_tx_step and n_tx is multiple of n_ty_step / n_tx_step
    e.g. for 6/1 it has to be multiple of 6
    for 6/2 it hs to be multiple of 3
    for 6/3 it has to be multiple of 2

    If dim3 is True then return X, Y 3D numpy arrays
    else returns xy_df, new_x_cols, new_y_cols
    - DataFrame should be indexed before sending

    :param td:
    :param x_cols:
    :param y_cols:
    :param sort_cols:
    :param other_cols:
    :param n_tx:
    :param n_ty:
    :param h:
    :param dim3:
    :return:
    """

    if x_cols is None:
        x_cols = td_x.columns.tolist()

    if y_cols is None:
        y_cols = td_y.columns.tolist()

    # compute numeric indices
    xy_cols = list(set(x_cols + y_cols))
    x_idx = [x_cols.index(col) for col in x_cols]
    y_idx = [y_cols.index(col) for col in y_cols]

    # why are you sorting it here ?
    #td_sorted = td.sort_values(sort_cols).reset_index(drop=True)
    # we have to send here two numpy arrays because they are
    #    sampled at different intervals
    x, y = nps_to_xy(td_x[x_cols].to_numpy(),
                     td_y[y_cols].to_numpy(),
                    n_tx=n_tx,
                    n_ty=n_ty,
                    n_tx_step=n_tx_step,
                    n_ty_step=n_ty_step,
                    h=h,
                    x_idx=x_idx,
                    y_idx=y_idx, dim3=dim3)

    new_x_cols = [
        '{}_t-{}'.format(col, x)
        for x, col in itertools.product(range(n_tx - 1, -1, -1), x_cols)
    ]
    new_y_cols = ['{}_t+{}'.format(col, x) for x, col in
                  itertools.product(range(h, h + n_ty), y_cols)]

    if dim3 is False: #means we are returning a dataframe
        # now lets compute column names

        #new_x_cols, new_y_cols = new_xy_cols(x_cols, y_cols, n_tx, n_ty, h)
        #xy_df = td.head(x.shape[1]).index #[other_cols]
        xy_df = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1,
                          ignore_index=True)
        #print('xy_df:',xy_df.shape)
        #print(new_x_cols,new_y_cols)
        #print(td.iloc[:x.shape[0],:].index)
        xy_df = xy_df.set_index(td_y.index[:y.shape[0]])
        xy_df.columns = new_x_cols + new_y_cols
        return xy_df, (new_x_cols, new_y_cols)
    else:
        return (x, y), (new_x_cols, new_y_cols)

def nps_to_xy(td_x: np.ndarray,td_y: np.ndarray,
              n_tx=1, n_ty=1,
             n_tx_step=1, n_ty_step=1, h=1,
             x_idx=None, y_idx=None, dim3=True):
    """
        n_tx and n_ty are number of x and y timesteps respectively
            if x = 1 then input is one timestep
            if x > 1 then input is multistep
            if y = 1 then output is one-step-ahead
            if y > 1 then output is multi-step-ahead
        h is prediction horizon for direct strategy
            returns x : {n_tx-1,...,t}, y : {t+h,...,t+h+n_ty-1}
                            n_rows, n_tx * n_vx
            in 3d mode should return (samples, time_steps, features)
                                        n_rows, n_tx, n_vx
        x_idx is the list of columns to be used as input or feature with length n_vx
        y_idx is the list of columns to be used as output or target with length n_vy
        total inputs n_cx = n_vx * n_tx
        total outputs n_cy = n_vy * n_ty

        Post-steps:
        Time Series should not be divided into train, valid and test
        Because doing that would make us loose some values
        Hence after making it xy, then we convert to train, val, test

        Pre-steps:
        However timeseries should be normalized before applying this step

        #  to be converted based on strategy
        # for single step ahead prediction :
        #   input series : {t-n_tx,...,t}, output series : {t+h} h=1 or more
        # for multi step ahead :
        #   iterative : input series : {t-n_tx,...,t}, output series : {t+1}
        #        and columns of out_vec in input series
        #   direct : input series : {t-n_tx,...,t}, output series : {t+h}
        #   MIMO : input series : {t-n_tx,...,t}, output series : {t+1,...,t+n_ty}
        # test set is always going to be : input series : {t-n_tx,...,t}, output series : {t+1,...,t+n_ty}

    """

    # if numpy array is 1D, make it columnar vector

    td_x = np.expand_dims(td_x,axis=1) if td_x.ndim == 1 else td_x # : to :,1
    td_y = np.expand_dims(td_y,axis=1) if td_y.ndim == 1 else td_y # : to :,1

    # print(n_tx, n_ty, h, x_idx, y_idx, td.shape)
    td_x_n_cols = td_x.shape[1]
    td_x_n_rows = td_x.shape[0]

    td_y_n_cols = td_y.shape[1]
    td_y_n_rows = td_y.shape[0]

    n_step_div = n_ty_step // n_tx_step # this is basically sliding size of window
    y_skip_from_top = (n_tx // n_step_div) + h - 1

    if x_idx is None:
        x_idx = range(0, td_x_n_cols)
    if y_idx is None:
        y_idx = range(0, td_y_n_cols)

    # y_cols_x_idx = [x_idx.index(i) for i in y_idx]

    n_vx = len(x_idx)
    n_vy = len(y_idx)

    # TODO: Validation of other options

    """
    # print(td_rows) 1 2 3 4 5 6 7 8 9      1 2   3 4
                                            2 3   4 5
                                            3
                                            4
                                            5
                                            6 7   8 9
    """
    n_rows = td_x_n_rows - ((n_tx + n_ty) - 1 + h - 1)
    n_y_rows = td_y.shape[0] - y_skip_from_top
    # print(n_rows)
    shapeX = (n_rows, n_tx, n_vx) if dim3 else (n_rows, n_vx * n_tx)
    shapeY = (n_y_rows, n_ty, n_vy) if dim3 else (n_y_rows, n_vy * n_ty)
    dataX = np.empty(shape=shapeX,
                     dtype=np.float32)
    dataY = np.empty(shape=shapeY,
                     dtype=np.float32)
    step = 1
    if dim3: # returns n_rows, n_tx, n_vx
        from_idx = n_tx
        to_idx = n_rows
        # for i in range(from_idx, to_idx):
        #    idx = range(i-n_tx, i, step )
        #    dataX = td[idx,x_idx]

        # print('dataX.shape {},td.shape {},n_tx {},x_idx {}'.format(dataX.shape,td.shape,n_tx,x_idx))

        for i in range(0, n_rows):
            # print('i {} dataX[i].shape {}, td[i:n_tx,x_idx].shape {}'.format(i, dataX[i].shape,td[i:n_tx,x_idx].shape))
            #print(i,'td_x:',td_x.shape,'dataX:',dataX.shape)
            dataX[i] = td_x[i:i + n_tx, x_idx]
        for i in range(0, n_y_rows):
            #print(i,'td_y:',td_y.shape,'dataY:',dataY.shape)
            dataY[i] = td_y[i +y_skip_from_top :
                            i +y_skip_from_top + n_ty , y_idx]
        # now drop the columns
        dataX = dataX[::n_step_div]
        #dataY = dataY[::n_step_div]
    else: # n_rows, n_tx * n_vx
        from_col = 0
        # print(from_col)
        # print(from_col+n_vx)

        for i in range(n_tx, 0, -1):
            dataX[:, from_col:from_col + n_vx] = np_util.shift(
                td_x[:, x_idx], i)[n_tx:n_rows + n_tx]
            from_col = from_col + n_vx

        # forecast sequence (t+h, ... t+h+n_ty)
        from_col = 0
        for i in range(0, n_ty):
            # y_cols.append(shift(td,-i))
            dataY[:, from_col:from_col + n_vy] = np_util.shift(
                td_y[:, y_idx], -(i + y_skip_from_top))[:-y_skip_from_top]
            from_col = from_col + n_vy
        dataX = dataX[::n_step_div]
        #dataY = dataY[::n_step_div]

    return dataX, dataY

def df_to_xy(td:pd.DataFrame, x_cols=None,
                y_cols=None, n_tx=1, n_ty=1, n_ty_step=1, h=1,
                dim3=True):
    """
    If dim3 is True then return X, Y 3D numpy arrays
    else returns xy_df, new_x_cols, new_y_cols
    - DataFrame should be indexed before sending

    :param td:
    :param x_cols:
    :param y_cols:
    :param sort_cols:
    :param other_cols:
    :param n_tx:
    :param n_ty:
    :param h:
    :param dim3:
    :return:
    """
    if td is None:
        raise ValueError(
            'the parameter td not provided')
    elif not isinstance(td, pd.DataFrame):
        raise ValueError(
            'the parameter td is not pandas dataframe')

    if x_cols is None:
        x_cols = td.columns.tolist()

    if y_cols is None:
        y_cols = td.columns.tolist()

    # compute numeric indices
    xy_cols = list(set(x_cols + y_cols))
    x_idx = [xy_cols.index(col) for col in x_cols]
    y_idx = [xy_cols.index(col) for col in y_cols]

    # why are you sorting it here ?
    #td_sorted = td.sort_values(sort_cols).reset_index(drop=True)
    x, y = np_to_xy(td[xy_cols].to_numpy(),
                    n_tx=n_tx,
                    n_ty=n_ty,
                    n_ty_step=n_ty_step,
                    h=h,
                    x_idx=x_idx,
                    y_idx=y_idx, dim3=dim3)

    new_x_cols = [
        '{}_t-{}'.format(col, x)
        for x, col in itertools.product(range(n_tx - 1, -1, -1), x_cols)
    ]
    new_y_cols = ['{}_t+{}'.format(col, x) for x, col in
                  itertools.product(range(h, h + n_ty,n_ty_step), y_cols)]

    if dim3 is False: #means we are returning a dataframe
        # now lets compute column names

        #new_x_cols, new_y_cols = new_xy_cols(x_cols, y_cols, n_tx, n_ty, h)
        #xy_df = td.head(x.shape[1]).index #[other_cols]
        xy_df = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1,
                          ignore_index=True)
        #print(td.iloc[:x.shape[0],:].index)
        xy_df = xy_df.set_index(td.index[:x.shape[0]])
        xy_df.columns = new_x_cols + new_y_cols
        return xy_df, (new_x_cols, new_y_cols)
    else:
        return (x, y), (new_x_cols, new_y_cols)

def np_to_xy(td: np.ndarray, n_tx=1, n_ty=1, n_ty_step=1, h=1,
             x_idx=None, y_idx=None, dim3=True):
    """
        n_tx and n_ty are number of x and y timesteps respectively
            if x = 1 then input is one timestep
            if x > 1 then input is multistep
            if y = 1 then output is one-step-ahead
            if y > 1 then output is multi-step-ahead
        h is prediction horizon for direct strategy
            returns x : {n_tx-1,...,t}, y : {t+h,...,t+h+n_ty-1}
                            n_rows, n_tx * n_vx
            in 3d mode should return (samples, time_steps, features)
                                        n_rows, n_tx, n_vx
        x_idx is the list of columns to be used as input or feature with length n_vx
        y_idx is the list of columns to be used as output or target with length n_vy
        total inputs n_cx = n_vx * n_tx
        total outputs n_cy = n_vy * n_ty

        Post-steps:
        Time Series should not be divided into train, valid and test
        Because doing that would make us loose some values
        Hence after making it xy, then we convert to train, val, test

        Pre-steps:
        However timeseries should be normalized before applying this step

        #  to be converted based on strategy
        # for single step ahead prediction :
        #   input series : {t-n_tx,...,t}, output series : {t+h} h=1 or more
        # for multi step ahead :
        #   iterative : input series : {t-n_tx,...,t}, output series : {t+1}
        #        and columns of out_vec in input series
        #   direct : input series : {t-n_tx,...,t}, output series : {t+h}
        #   MIMO : input series : {t-n_tx,...,t}, output series : {t+1,...,t+n_ty}
        # test set is always going to be : input series : {t-n_tx,...,t}, output series : {t+1,...,t+n_ty}

    """

    if td is None:
        raise ValueError(
            'the parameter td not provided')
    elif not isinstance(td, np.ndarray):
        raise ValueError(
            'the parameter td is not numpy array')

    # if numpy array is 1D, make it columnar vector
    td = np.expand_dims(td,axis=1) if td.ndim == 1 else td # : to :,1
    # print(n_tx, n_ty, h, x_idx, y_idx, td.shape)
    td_cols = td.shape[1]
    td_rows = td.shape[0]

    if x_idx is None:
        x_idx = range(0, td_cols)
    if y_idx is None:
        y_idx = range(0, td_cols)

    # y_cols_x_idx = [x_idx.index(i) for i in y_idx]

    n_vx = len(x_idx)
    n_vy = len(y_idx)

    # TODO: Validation of other options

    """
    # print(td_rows) 1 2 3 4 5 6 7 8 9      1 2   3 4
                                            2 3   4 5
                                            3
                                            4
                                            5
                                            6 7   8 9
    """
    n_rows = td_rows - (n_tx + (n_ty * n_ty_step) - 1 + h - 1)
    # print(n_rows)
    shapeX = (n_rows, n_tx, n_vx) if dim3 else (n_rows, n_vx * n_tx)
    shapeY = (n_rows, n_ty, n_vy) if dim3 else (n_rows, n_vy * n_ty)
    dataX = np.empty(shape=shapeX,
                     dtype=np.float32)
    dataY = np.empty(shape=shapeY,
                     dtype=np.float32)
    step = 1
    if dim3: # returns n_rows, n_tx, n_vx
        from_idx = n_tx
        to_idx = n_rows
        # for i in range(from_idx, to_idx):
        #    idx = range(i-n_tx, i, step )
        #    dataX = td[idx,x_idx]

        # print('dataX.shape {},td.shape {},n_tx {},x_idx {}'.format(dataX.shape,td.shape,n_tx,x_idx))

        for i in range(0, n_rows):
            # print('i {} dataX[i].shape {}, td[i:n_tx,x_idx].shape {}'.format(i, dataX[i].shape,td[i:n_tx,x_idx].shape))
            dataX[i] = td[i:i + n_tx, x_idx]
            if (i+1)%n_ty_step == 0:
                dataY[((i+1)//n_ty_step)-1] = td[i + n_tx + h - 1 :i + n_tx + h - 1  + ( n_ty * n_ty_step), y_idx]


    else: # n_rows, n_tx * n_vx
        from_col = 0
        # print(from_col)
        # print(from_col+n_vx)

        for i in range(n_tx, 0, -1):
            dataX[:, from_col:from_col + n_vx] = np_util.shift(
                td[:, x_idx], i)[n_tx:n_rows + n_tx]
            from_col = from_col + n_vx

        # forecast sequence (t+h, ... t+h+n_ty)
        from_col = 0
        for i in range(0, n_ty):
            # y_cols.append(shift(td,-i))
            dataY[:, from_col:from_col + n_vy] = np_util.shift(
                td[:, y_idx], -((((i+1) * n_ty_step) -1) + h - 1))[n_tx:n_rows + n_tx]
            from_col = from_col + n_vy

    return dataX, dataY



    #_df: Optional[dftypes] = field(init=False,repr=False)

   # by default first datetime column is dt_col

    #@property
    #def df(self):
    #    return self._df

    #@df.setter
    #def df(self, df: Optional[dftypes] = None):
    #    self._df = df
def dt_gap_fill(df, freq, fill_value=None):
    return df.asfreq(freq,fill_value=fill_value)
    #resample(freq)

# TODO: I think should also work for any kind of filed not just dates

"""
    if df is None:
        print('None sent as Data')
        return df
    if isinstance(df,pd.DatetimeIndex):
        df = df.to_series() # can we use Pandas.Series() instead ?
    else:
        if col is None:
            df = df.index.to_series()
        else:
            df = df[col]
    return df.sort_values().diff().value_counts(sort=False)
"""

def freq_as_timedelta(freq):
    if isinstance(freq,str):
        freq = pd.tseries.frequencies.to_offset(freq)
    if isinstance(freq, pd.tseries.offsets.DateOffset):
        freq = pd.to_timedelta(freq)
    return freq


def next_weekday(d=datetime.datetime.today(), weekday=0, next=True):
    """
    Returns the date when same day of week happens next or prev
    :param d: date from which to find
    :param weekday: # 0 = Monday, 1=Tuesday, 2=Wednesday
    :param next: next weekday or previous weekday
    :return:
    """
    days_ahead = weekday - d.weekday()
    if next:
        if days_ahead < 0:  # Target day already happened this week
            days_ahead += 7
    else:
        if days_ahead > 0:  # Target day already happened this week
            days_ahead -= 7
    return d + datetime.timedelta(days_ahead)


def interpolate(df, dt_col=None, freq=None, new_freq=None):
    """
    Incoming DF: Either index or dt_col should be DateTime
    freq is either provided or inferred
    new_freq is always provided

    only one zid is provided here
    Also fills missing dates in dt_col
    :param df:
    :param id_col:
    :param dt_col:
    :param freq:
    :return:
    """
    if freq == new_freq:
        new_freq = None
    #1 interpolate any missing values on current freq
    if dt_col is not None: # overwrite index with dt_col
        df = df.set_index(dt_col).sort_values(dt_col)
    else: # we assume that index has the date
        df = df.copy() #dont modify original df
    df = df.resample(rule=freq).interpolate()

    #2 resample to new frequency
    if new_freq is not None:
        df = df.resample(rule=new_freq).mean()
    if dt_col is not None:
        df = df.reset_index()  # restore the dt_col from index
    return df