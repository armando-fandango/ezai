#dataframe utilities
import pandas as pd
import numpy as np

def get_first_dt_col(df):
    dt_col = None
    for col in df.columns:
        # from pandas.api.types import is_datetime64_any_dtype
        if np.issubdtype(df[col].dtype, np.datetime64):
            dt_col = col
            break
    return dt_col

def as_ordered_category(series, order_type = str, ascending=True):
    cats = series.unique()
    original_type = type(cats[0])
    if isinstance(cats,np.ndarray):
        cats = np.sort(cats)
        if not ascending:
            cats = np.flipud(cats)
    else: #isinstance(cats,pd.Categorical):
        cats = sorted(cats.astype(order_type), reverse=(not ascending))
    cats = list(map(original_type,cats))

    if not isinstance(cats,pd.Categorical):
        series = series.astype('category')
    series = series.cat.set_categories(cats, ordered=True)
    return series

def remove_unused_categories(df):
    for col in df.select_dtypes(include='category'):
        df[col]=df[col].cat.remove_unused_categories()
    return df

def reset_cat_index(df):
    '''Returns DataFrame with index as columns'''
    index_df = df.index.to_frame(index=False)
    df = df.reset_index(drop=True)
    #  In merge is important the order in which you pass the dataframes
    # if the index contains a Categorical.
    # pd.merge(df, index_df, left_index=True, right_index=True) does not work
    return pd.merge(index_df, df, left_index=True, right_index=True)

def gap_counts(series):
    """
    Returns the series consisting of gaps and how many data having those gaps
    :param series: Series is series or Index
    :param col:
    :return:
    """
    if isinstance(series, pd.Index):
        series = pd.Series(series)

    return series.sort_values().diff().value_counts(sort=False)

def nullity_sort(df, ascending=True, axis=0):
    """
    Sorts a DataFrame according to its nullity, in either ascending or descending order.

    :param df: The DataFrame object being sorted.
    :param ascending: The sorting method: either "ascending", "descending", or None (default).
    :param axis: 0 or rows and 1 or columns
    :return: The nullity-sorted DataFrame.
    """
    axis = df._get_axis_number(axis)

    if ascending:
        idx = np.argsort(df.count(axis=axis).values)
    else:
        idx = np.flipud(np.argsort(df.count(axis=axis).values))

    if axis: # axis==1
        return df.iloc[idx, :]
    else: #axis ==0
        return df.iloc[:, idx]

#TODO : Better argument word for filter please

def nullity_filter(df, filter='top', p=0, n=0, axis=0, ascending=None):
    """
    Filters a DataFrame according to its nullity, using some combination of 'top' and 'bottom' numerical and
    percentage values. Percentages and numerical thresholds can be specified simultaneously: for example,
    to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns, use
    `nullity_filter(df, filter='top', p=.75, n=5)`.

    :param df: The DataFrame whose columns are being filtered.
    :param filter: The orientation of the filter being applied to the DataFrame. One of, "top", "bottom",
    or None (default). The filter will simply return the DataFrame if you leave the filter argument unspecified or
    as None.
    :param p: A completeness ratio cut-off. If non-zero the filter will limit the DataFrame to columns with at least p
    completeness. Input should be in the range [0, 1].
    :param n: A numerical cut-off. If non-zero no more than this number of columns will be returned.
    :return: The nullity-filtered `DataFrame`.
    """

    def n_idx(counts,filter,n,ascending):
        idx = np.argsort(counts)
        if filter=='top':
            idx = idx[-n:]
        elif filter == 'bottom':
            idx = idx[:n]

        if ascending is None:
            idx = np.sort(idx)
        elif ascending is False:
            idx = np.flipud(idx)

        return idx # to reverse the effect of argsort

    def p_idx(counts,filter,p):
        counts = counts / counts.shape[0]
        idx=[]
        if filter == 'top':
            idx = [c >= p for c in counts ]
        elif filter == 'bottom':
            idx = [c <= p for c in counts ]
        return idx

    axis = df._get_axis_number(axis)
    counts = df.count(axis=axis).values
    if p: # filter the counts by percentage
        idx = p_idx(counts,filter,p)
        counts = counts[idx]
    n = n or df.shape[0] # if n is zero, set it to all
    # filter by top n or bottom n counts ad sort
    idx = n_idx(counts,filter,n,ascending)

    if axis:  # axis ==1 or columns
        return df.iloc[idx, :]
    else:   # axis = 0 or rows
        return df.iloc[:, idx]

def std_colnames(df, colnames_dic):
    """
    Also returns the dataframe
    :param df:
    :return:
    """
    colnames_dic = {key: val for key, val in colnames_dic.items() if key in df}
    return df.rename(columns=colnames_dic)

def std_colcodes(df, colcodes_dic):
    """
    Also returns the dataframe
    :param df:
    :return:
    """
    return df.replace(colcodes_dic)

#TODO: Timeseries generator
def generate_df():
    dt = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')
    df = pd.DataFrame(dt,columns=['dt'])
    df['values'] = np.random.randint(0,100,size=(len(dt)))
    return df

"""
    No need of this function, just use is_unique()
    def has_dups(df, col=None):
    col = df.index if col is
    return df[col].count() > df[col].nunique()
"""

# copied from https://github.com/pandas-dev/pandas/blob/3e88e170a436b5e3cfdfc29f8a7416220658c616/pandas/io/formats/info.py#L44
from typing import Union
def _sizeof_fmt(num: Union[int, float], size_qualifier: str='+') -> str:
    """
    Return size in human readable format.
    Parameters
    ----------
    num : int
        Size in bytes.
    size_qualifier : str
        Either empty, or '+' (if lower bound).
    Returns
    -------
    str
        Size in human readable format.
    Examples
    --------
    >>> _sizeof_fmt(23028, '')
    '22.5 KB'
    >>> _sizeof_fmt(23028, '+')
    '22.5+ KB'
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f}{size_qualifier} {x}"
        num /= 1024.0
    return f"{num:3.1f}{size_qualifier} PB"

def memory_usage(df):
    return _sizeof_fmt(df.memory_usage(deep=True).sum())

