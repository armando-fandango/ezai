"""
dataset classes representing RITIS data
"""
from bidict import frozenbidict
import os
import math
import pandas as pd

from archived_code.ezai.data import datasets_root
from archived_code.ezai.util import dict_util
from archived_code.ezai.util import df_util
from archived_code.ezai.util import vis_util
from archived_code.ezai.data import temporal

from matplotlib import pyplot as plt

# TODO: Reafctor into geo utils if more datasets need it
dir2code = frozenbidict({
    "North": "N",
    "East": "E",
    "West": "W",
    "South": "S"
})


# TODO: obsolete can be deleted once the above bidict is in use
"""
CODE2DIR = {
    "N": "North",
    "E": "East",
    "W": "West",
    "S": "South"
}
"""

# Columns present in the data
ID = 'id'  # ''zone_id'
DT = 'dt'  # ''measurement_start'
SPD = 'speed'
VOL = 'volume'  # veh / x minute
OCC = 'occupancy'  # % of time detector was occupied
LAT = 'lat'
LON = 'lon'
RD = 'road'


# dictionary representing the standardized column names
# TODO: see if you need a bidict here
STD_COLNAMES = {'measurement_start': DT, 'zone_id': ID, 'latitude': LAT,
                'longitude': LON, 'freeway': RD}

# this dictionary helps convert columns into codes for saving soace
# TODO if bidict doesnt work then find a way to extract just the dict here
STD_COLCODES = {"direction": dir2code}

def dt_gap_fill(df,freq=None):
    """
    Only does zone for now
    TODO: also implement for lane
    :param freq:
    :return:
    """
    # Gap counts would be a series with date Index and counts as values
    gap_count = df_util.gap_counts(df.index)
    # Find the frequency from gap counts
    if freq is None:  # If freq is not provided in arguments
        freq = gap_count.index.min()
    print('Inferred Frequency by counting gaps :', freq)
    print("Filling the gaps in dates")
    df = temporal.dt_gap_fill(df, freq).sort_index() # This must also set the index.freq
    print('Current frequency from index :',df.index.freq)
    return df

def unpivot(df,index=DT, column=ID):
    df = df.melt(value_name='val',ignore_index=False)\
        .pivot_table(index=[column, index],columns='var',values='val')\
        .reset_index(level=0).head()

def pivot(df,values=None, index=DT, column=ID):
    """
    Only pivots zone for now
    TODO: pivot for lane too
    :param index:
    :param column:
    :param values:
    :return:
    """
    # Find the column names if valus is none
    if values is None:
        values = df.columns.drop([column, index]).tolist()

    # Remove the categories that are not used
    df = df_util.remove_unused_categories(df.copy())

    # Pivot the table to bring ID from rows as column headings
    df = df.pivot_table(index=index,
                        columns=column,
                        values=values,
                        dropna=False)

    if df.columns.names[0] is None:
        df = df.rename_axis(columns=['var', column])

    return df

# works for multi-series, multi-var, singleseries, single-var
# Nullity analysis specific to RITIS Data
def nullity_analysis(df):
    """

    :param df: is already pivoted
    :param index:
    :param column:
    :param values:
    :param freq:
    :return:
    """
    #cant use till the fix the PR from tabulate import tabulate
    ## TODO: doesnt make sense to find freq if we know its multi_series
    # freq = df.index.freq
    # print('Current frequency from index :',freq)
    # if freq is None:
    #    freq = df.index.inferred_freq
    #    print('Inferred Frequency by pandas :', freq)
    #    if freq is None:

    # TODO: for multivariate series - analyse all variables together

    # Count how many not-na values exist
    notna_count = df.count() #returs series
    s = notna_count / notna_count.shape[0]

    # TODO: By what other values would you need it to be sorted ?

    # Draw nullity plots for each var - with sorted by ID
    print("Sorting notna-count by id")
    s = s.sort_index()
    df = df.reindex(columns=s.index)
    for var in df.columns.levels[0]:
        #col_list = s.loc[var,:].index
        nullity_plot(df.loc[:,var],var)

    # Draw nullity plots for each var - with sorted by %
    print("Sorting notna-count by percentage")
    """ The above one liner does this more elegently
    s = s.reset_index(0)
    s = s.sort_values(by=s.columns.tolist(),ascending=False)
    s = s.set_index(s.columns[0], append=True). \
            swaplevel(0,-1)#.rename(columns={0:'percentage_values_present'})
            # should this be -1 in swaplevel above
    df = df.reindex(columns=s.index)
    """
    for var in df.columns.levels[0]:
        df1 = df_util.nullity_filter(df.loc[:, var], p=0, n=0, axis=0, ascending=False)
        #col_list = s.loc[var,:].index
        nullity_plot(df1,var)
        del df1

def nullity_plot(df,var):
    # if orientation = "left" then plot can have only 50 columns
    # otherwise also plot an have only 50 columns
    n_cols_in_plot=40
    n_rows_in_plot=100
    if df.index.freq is None:
        freq = None
    else: # use n_cols_in_plot for left oreitn else use n_rows_***
        freq =  (math.ceil(df.shape[0] / n_cols_in_plot)) * df.index.freq

    #plt.xticks(rotation=90)
    #plt.plot(list(df.index),df.count(axis=1))
    #plt.show()
    #TODO replace with proper import after PR is pulled
    #m_path = os.path.join(os.path.expanduser('~'),'projects','missingno')
    #msno = util.m_load('missingno',m_path)
    fig = vis_util.nullity_plot_matrix(df.iloc[:, :],
                                       freq=freq,
                                       fontsize=12,
                                       labels=True,
                                       sparkline=True, orientation='left')
    #fig.suptitle(var,fontsize='xx-large')
    #plt.savefig('matrix_left_nosparkline.png', bbox_inches='tight')
    plt.show()

class RITISDetector():
    """
    Class representing data from the RITIS Detectors

    Three files are produced bu RITIS Detectors:

    The Detector Identification File has following columns:
    zone_id,display_name,state,rtmc,timezone,road,direction,location_description,lane_type,organization,detector_type,latitude,longitude,bearing,default_speed,interval,length

    The Lane Readings file has the following columns :
    zone_id,lane_number,lane_id,measurement_start,speed,volume,occupancy,quality

    The Zone Readings file has the following columns :
    zone_id,measurement_start,speed,volume,occupancy,quality
    # zone_id	measurement_start	speed	volume	occupancy	volume_weighted_speed	factored_up_volume
    """

    dataset_name = 'ritis'
    def __init__(self, subset_name):
        self.subset_name = subset_name
        self.dataset_home = os.path.join(datasets_root,self.dataset_name,subset_name)
        self.df = dict_util.DictObj({
            'meta': None,
            'lane': None,
            'zone': None,
            'event': None
        })
        self.csv_name = dict_util.DictObj({
            'meta': 'Detector_Identification.csv',
            'lane': 'Lane_Readings.csv',
            'zone': 'Zone_Readings.csv',
            'event': 'Events.csv'
        })

        self.dt_cols = dict_util.DictObj({
            'meta': False,
            'lane': False,
            'zone': ['measurement_start'],
            'event': False
        })

    def load_raw(self, kind=[], filetype = 'parquet', folder=None):
        """
        Loads the CSV files for this dataset

        :param kind: a set or list or tuple # ['meta','zone','lane','event']:
        :param folder: optional
        :return:
        """
        folder = folder or os.path.join(self.dataset_home,'raw')

        dtype_dict={'zone_id':'category',
                    'display_name':'category',
                    'state':'category',
                    'rtmc':'string',
                    'timezone':'category',
                    'road':'string',
                    'direction':'category',
                    'location_description':'string',
                    'lane_type':'category',
                    'organization':'category',
                    'detector_type':'category',
                    'latitude':'float',
                    'longitude':'float',
                    'bearing':'string',
                    'default_speed':'float',
                    'interval':'float',
                    'length':'float',
                    'speed': 'float',
                    'volume': 'float',
                    'occupancy': 'float',
                    'quality': 'string'
            }
        for k in ['meta'] + kind:  # ['meta','zone','lane','event']:

            if filetype == 'csv':
                filename = os.path.join(folder, '{}.csv'.format(k))
                df = pd.read_csv(filename,
                                     parse_dates=self.dt_cols[k],
                                     dtype = dtype_dict,
                                     index_col=None)
                # drop Unnamed columns
                df = df.drop(df.filter(regex="Unname"),axis=1)
            elif filetype=='parquet':
                filename = os.path.join(folder, '{}.parquet'.format(k))
                df = pd.read_parquet(filename, engine='pyarrow')
            else:
                raise ValueError('{} not supported yet.'.format(filetype))

            #self.df[k] = self.std_df(self.df[k],v['cols'] if 'cols' in v.keys() else None)

            #1 Standardize the column names
            #2 Filter the columns
            #3 Standardize the column codes
            df = df_util.std_colnames(df, STD_COLNAMES)
            col_list = list(df.columns)
            df = df_util.std_colcodes(df, STD_COLCODES)

            #TODO: Why ?????
            #4 replace - with _
            for col in ['road']:
                if col in col_list:
                    df.loc[:, col] = df.loc[:, col].str.replace('-', '_')

            #6 set columns as category - order the categories

            #Geo sorted categoricals
            if ID in col_list:
                if k=='meta':
                    df.loc[:, ID] = df_util.as_ordered_category(df.loc[:, ID],
                                                                order_type = 'geo',
                                                                ascending=True,
                                                                pts = df.loc[:,['lon','lat']])
                else:
                    df.loc[:, ID] = df_util.as_ordered_category(df.loc[:, ID],
                                    self.df['meta'].loc[:, ID].cat.categories)

            df = df_util.remove_unused_categories(df)

            self.df[k] = df
        return self

    def info(self):

        if self.df.meta is None:
            print('---------')
            print('No meta data')
            print('---------')
        else:
            print('---------')
            print('meta data')
            print('---------')
            df = self.df.meta
            print('Total records: ', df.id.nunique())
            print('Memory usage: ', df_util.memory_usage(df))
            if not df[ID].is_unique:
                print('Error : has duplicated zone_id')

        if self.df.lane is None:
            print('---------')
            print('No lane data')
            print('---------')
        else:
            print('---------')
            print('lane data coming soon')
            print('---------')

        if self.df.zone is None:
            print('---------')
            print('No zone data')
            print('---------')
        else:
            print('---------')
            print('zone data')
            print('---------')
            df = self.df.zone
            print('Total records: ', df.id.count())
            print('Memory usage: ', df_util.memory_usage(df))
            print('Unique id: ', df.id.nunique())
            print('dt stats:')
            print('min: ', df.dt.min())
            print('max: ', df.dt.max())
            print('unique: ', df.dt.nunique())
            print('common ids in zone and meta:', len(self.common_ids()))

        if self.df.event is None:
            print('---------')
            print('No event data')
            print('---------')
        else:
            print('---------')
            print('event data coming soon')
            print('---------')

    def common_ids(self, kind=['zone','lane','event']):
        """
        Find common id between meta file and zone/lane/event file
        if kind = None, it will find common between all available dataframes
        :param kind:
        :return:
        """
        sets=[]
        for k in kind:
            if self.df[k] is not None:
                sets.append(set(self.df[k].id.unique()))

        if (self.df.meta is None) or (not sets):
            common = ()
        else:
            common = set(self.df.meta.id.unique()).intersection(*sets)
        return common

    def filter_by_common_id(self):
        common_id_set = self.common_ids()
        #df = self.df.meta
        #df_list = [df[df.id.isin(common_id_set)].reset_index(drop=True)]
        for k in ['meta','zone','lane','event']:
            if self.df[k] is not None:
                df = self.df[k]
                self.df[k] = df.query('id in @common_id_set')\
                    .reset_index(drop=True)
                df_util.remove_unused_categories(self.df[k])
        return self
