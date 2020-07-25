import os
import math
import time
import tempfile

import shutil

from ezai.util import util
from ezai.data import image

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#from mpl_toolkits.basemap import Basemap


import geopandas as gpd
import folium
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from contextlib import contextmanager
import cv2

"""
General Images size: w x h in inch

1-col : 3.5 (1063) x 
1.5 col : 5 or 5.5 (1654) x
2 col 7.2 to 7.5 (2244) x

"""
# TODO: Basemap is not working, uncomment when its working
# TODO: Since basemap is deprecated in favor of cartopy, add that

def map_with_basemap(df):
    #Hack to fix missing PROJ env var
    conda_dir = os.environ['CONDA_PREFIX']
    proj_lib = os.path.join(conda_dir, 'share', 'proj')
    os.environ["PROJ_LIB"] = proj_lib
    #shutil.copyfile('epsg',os.path.join(proj_lib,'epsg'))

    #lat = df.lat.values
    #lon = df.lon.values

    # determine range to print based on min, max lat and lon of the data
    margin = 1.5 # buffer to add to the range
    lat_min = min(df.lat) - margin
    lat_max = max(df.lat) + margin
    lon_min = min(df.lon) - margin
    lon_max = max(df.lon) + margin

    # create map using BASEMAP
    plt.figure(figsize=(12,6))
    m = Basemap(llcrnrlon=lon_min,
                llcrnrlat=lat_min,
                urcrnrlon=lon_max,
                urcrnrlat=lat_max,
                lat_0=(lat_max - lat_min)/2,
                lon_0=(lon_max-lon_min)/2,
                projection='merc',
                resolution = 'f', # l, i, h, f
                area_thresh=10000.,
                )

    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color = 'white',lake_color='aqua')

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawcounties()

    # convert lat and lon to map projection coordinates
    lons, lats = m(df.lon.values, df.lat.values)
    # plot points as red dots
    m.scatter(lons, lats, marker = 'o', color='r', zorder=5)
    plt.show()


def map_with_geopandas(df, width=5, height=5):
    gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lon,df.lat))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # We restrict to South America.
    ax = world[world.continent == 'North America'].plot(color='white',
                                                        figsize=(width,height),
                                                        edgecolor='black')

    minx, miny, maxx, maxy = gdf.total_bounds
    margin = 1.8 # buffer to add to the range
    ax.set_xlim(minx-margin, maxx+margin)
    ax.set_ylim(miny-margin, maxy+margin)
    # We can now plot our ``GeoDataFrame``.
    gdf.plot(ax=ax, color='red')
    #plt.axis("off")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.show()



def map_with_folium(df, width=5, height=5 ):
    # from inch to pixels at 300dpi
    dpi = plt.gcf().dpi
    width = width  * dpi #55
    height = height * dpi  #55
    #create a map
    f_map = folium.Map(width=width,
                   height=height,
                   #location=[35.22,-120.69],
                   #%zoom_start = 11,
                   prefer_canvas=True)


    def plotDot(point):
        folium.CircleMarker(
            location=(point.lat, point.lon),
            radius=1,
            popup='id:{}'.format(point.id),
            #fill=True,
            #fill_color='black'
            color='red').add_to(f_map)


    #use df.apply(,axis=1) to iterate through every row in your dataframe
    df[['id','lat','lon']].apply(plotDot, axis=1)

    #Set the zoom to the maximum possible
    f_map.fit_bounds(f_map.get_bounds(), padding=(20, 20))

    #Save the map to an HTML file
    #f_map.save('map.html')

    return f_map

def folium_to_png(map,filename):
    """
    Folium's inbuilt function doesnt allow to capture the shot of map only.

    :param map:
    :param filename:
    :return:
    """
    @contextmanager
    def temp_html_filepath(data):
        """Yields the path of a temporary HTML file containing data."""
        filepath = ''
        try:
            fid, filepath = tempfile.mkstemp(suffix='.html', prefix='folium_')
            os.write(fid, data.encode('utf8') if isinstance(data, str) else data)
            os.close(fid)
            yield filepath
        finally:
            if os.path.isfile(filepath):
                os.remove(filepath)

    delay = 3
    png = None # return None if it doesnt work

    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Firefox(options=options)

    html = map.get_root().render()
    with temp_html_filepath(html) as fname:
        # We need the tempfile to avoid JS security issues.
        driver.get('file:///{path}'.format(path=fname))
        #    driver.maximize_window()
        time.sleep(delay)
        png  = driver.find_element_by_class_name('folium-map').screenshot_as_png
        with open(filename, 'wb') as file:
            file.write(png)
        driver.quit()
    return png
"""
I solved this with a few extra lines of code after calling msno.matrix. In my df, I had a column called year and I wanted to see if there were some years that had missing values. Therefore, my code looked like this:

df = df.sort_values(by=['year'])

fontsize = 20
    
fig, ax = plt.subplots(1, 1, figsize=[20, 14])
msno.matrix(df=df, ax=ax, color=(0.2, 0.2, 0.2), sparkline=False, fontsize=fontsize)

years = list(df['year'].unique())
ylim_start, ylim_end = ax.get_ylim()
step_size = df.shape[0] / len(years)
_ = ax.yaxis.set_ticks(np.arange(ylim_end, ylim_start, step_size))
_ = ax.yaxis.set_ticklabels(years, fontsize=fontsize)

"""

def nullity_plot_matrix(df,
           figsize=None, width_ratios=(15, 1), height_ratios=(15, 1), color=(0, 0, 0),
           na_color = (0.6,0,0), notna_color=(0,0.6,0),
           fontsize=16, labels=None, sparkline=True,
           freq=None, orientation=None):
    """
    A matrix visualization of the nullity of the given DataFrame.

    :param df: The `DataFrame` being mapped.
    :param figsize: The size of the figure to display.
    :param fontsize: The figure's font size. Default to 16.
    :param labels: Whether or not to display the column names. Defaults to the underlying data labels when there are
    50 columns or less, and no labels when there are more than 50 columns.
    :param sparkline: Whether or not to display the sparkline. Defaults to True.
    :param width_ratios: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15, 1)`.
    Does nothing if `sparkline=False`.
    :param color: The color of the filled columns. Default is `(0.25, 0.25, 0.25)`.
    :param orientation: The way the matrix plot is oriented. Defaults to vertical if there are less than or equal to 50
    columns and horizontal if there are more.
    :return: If `inline` is False, the underlying `matplotlib.figure` object. Else, nothing.
    """

    n_rows  = df.shape[0]
    n_cols  = df.shape[1]
    col_range = list(range(0, n_cols))
    row_range = list(range(0, n_rows))

    if orientation is None:
        if n_cols > 50:
            orientation = 'left'  # index is horizontal, columns are vertical
        else:
            orientation = 'bottom'

    # z is the color-mask array, g is a NxNx3 matrix. Apply the z color-mask to set the RGB of each pixel.
    z = df.notnull().values
    g = np.zeros((n_rows, n_cols, 3))

    g[z < 0.5] = na_color
    g[z > 0.5] = notna_color

    if not (orientation=="top" or orientation=="bottom"):
        g = g.transpose((1,0,2))

    # Set up the matplotlib grid layout. A unary subplot if no sparkline, a left-right splot if yes sparkline.
    if figsize is None: #TODO: What is the max figsize in notebook possible ?
        if (orientation=="top" or orientation=="bottom"):
            figsize =(25,min(60,math.ceil(n_rows * 0.005)))
        else:
            #figsize =(25, min(25,3+ math.ceil(n_cols * 0.4)))
            figsize =(25, min(60, math.ceil(n_cols * 0.4)))

    fig = plt.figure(figsize=figsize)
    if sparkline:
        gs = fig.add_gridspec(2, 2, width_ratios=width_ratios, wspace=0.08,
                              height_ratios=height_ratios, hspace=0.04)
        axv = fig.add_subplot(gs[0,1])
        axh = fig.add_subplot(gs[1,0])
    else:
        gs = fig.add_gridspec(1, 1)
    ax0 = fig.add_subplot(gs[0,0])

    # Create the nullity plot.
    ax0.imshow(g, interpolation='none')

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(b=False)
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)

    # Enumerate the column_ticks
    # The labels argument is set to None by default.
    # If the user specifies it in the argument, respect that specification.
    # If 'left' orientation display always, else display for <= 50 columns and do not display for > 50.

    if labels or (labels is None and (orientation not in ['top','bottom'] or len(df.columns) <= 50)):
        column_ticks = list(range(0, n_cols))
        column_ticklabels = list(df.columns)
    else:
        column_ticks = []
        column_ticklabels = []

    # Enumerate the index_ticklabels - Adds Timestamps ticks if freq is not None, else set up the two top-bottom row ticks.
    if freq:
        index_ticks=[]
        index_ticklabels = []

        if type(df.index) == pd.PeriodIndex:
            ts_array = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).values

            index_ticklabels = pd.date_range(df.index.to_timestamp().date[0],
                                             df.index.to_timestamp().date[-1],
                                             freq=freq).map(lambda t:
                                                            t.strftime('%Y-%m-%d'))

        elif type(df.index) == pd.DatetimeIndex:
            ts_array = pd.date_range(df.index[0], df.index[-1],
                                     freq=freq).values

            index_ticklabels = pd.date_range(df.index[0], df.index[-1],
                                             freq=freq).map(lambda t:
                                                            t.strftime('%Y-%m-%d'))
        else:
            raise KeyError('Dataframe index must be PeriodIndex or DatetimeIndex.')
        try:
            for value in ts_array:
                index_ticks.append(df.index.get_loc(value))
        except KeyError:
            raise KeyError('Could not divide time index into desired frequency.')

    else:
        index_ticks = [0, df.shape[0] - 1]
        index_ticklabels = [1, df.shape[0]]

    ha = 'left'
    va = 'center'

    if (orientation=="top" or orientation=="bottom"):
        ax0.set_xticks(column_ticks)
        ax0.set_xticklabels(column_ticklabels, ha=ha, fontsize=int(fontsize * 1.25), rotation=45)
        ax0.set_yticks(index_ticks)
        ax0.set_yticklabels(index_ticklabels, va=va, fontsize=int(fontsize * 1.25 ), rotation=0)
        for pt in col_range[:-1]:
            ax0.axvline(pt + 0.5, linestyle='-', color='white')
        for pt in row_range[:-1:40]:  # TODO: Replace 40 with number dervided from calculations
            ax0.axhline(pt+0.5, linestyle='-', color='white')
    else:
        ax0.set_xticks(index_ticks)
        ax0.set_xticklabels(index_ticklabels, ha=ha, fontsize=int(fontsize  * 1.25), rotation=45)
        ax0.set_yticks(column_ticks)
        ax0.set_yticklabels(column_ticklabels, va=va, fontsize=int(fontsize * 1.25), rotation=0)
        # Create the inter-column vertical grid.
        for pt in row_range[:-1:40]: # TODO: Replace 40 with number dervided from calculations
            ax0.axvline(pt + 0.5, linestyle='-', color='white')
        for pt in col_range[:-1]:
            ax0.axhline(pt+0.5, linestyle='-', color='white')

    if sparkline:
        col_counts = df.count(axis=1)
        row_counts = df.count(axis=0)
        col_completeness = col_counts.tolist()
        row_completeness = row_counts.tolist()
        min_col_completeness = min(col_completeness)
        max_col_completeness = max(col_completeness)
        min_row_completeness = min(row_completeness)
        max_row_completeness = max(row_completeness)

        # Set up the sparkline, remove the border element.
        for ax in [axh,axv]:
            ax.grid(b=False)
            ax.set_aspect('auto')
            ax.set_xmargin(0)
            ax.set_ymargin(0)
            # GH 25
            if int(mpl.__version__[0]) <= 1:
                ax.set_axis_bgcolor((1, 1, 1))
            else:
                ax.set_facecolor((1, 1, 1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        if (orientation=="top" or orientation=="bottom"):
            # Plot sparkline---plot is sideways so the x and y axis are reversed.
            axv.plot(list(reversed(col_completeness)), row_range, color=color)
            axh.plot(col_range, row_completeness, color=color)
        else:
            # Plot sparkline---plot is sideways so the x and y axis are reversed.
            axv.plot(list(reversed(row_completeness)), col_range, color=color)
            axh.plot(row_range, col_completeness, color=color)

        if labels:
            # Figure out what case to display the label in: mixed, upper, lower.
            label = 'Data\nCompleteness'
            if str(df.columns[0]).islower():
                label = label.lower()
            if str(df.columns[0]).isupper():
                label = label.upper()

            # Set up and rotate the sparkline label.
            ha = 'center'
            va = 'top'
            if (orientation=="top" or orientation=="bottom"):
                #axv.tick_params(axis='x', length=20, width=2)
                axv.spines['bottom'].set_visible(True)
                axv.set_xticks([min_col_completeness,  max_col_completeness])
                axv.set_xticklabels(['{:.1f}'.format(min_col_completeness/n_cols),
                                     '{:.1f}'.format(max_col_completeness/n_cols)],
                                    ha='center', va='top', fontsize=int(fontsize * 1.25))
                #axv.xaxis.tick_top()
                axv.set_yticks([])
                axv.set_xlabel(label, fontsize=int(fontsize))

                axh.spines['right'].set_visible(True)
                axh.set_yticks([min_row_completeness,  max_row_completeness])
                axh.set_yticklabels(['{:.1f}'.format(min_row_completeness/n_rows),
                                     '{:.1f}'.format(max_row_completeness/n_rows)],
                                    ha='left', va='center', fontsize=int(fontsize * 1.25))
                axh.yaxis.tick_right()
                axh.set_xticks([])
            else:
                #axv.tick_params(axis='x', length=20, width=2)
                axv.spines['bottom'].set_visible(True)
                axv.set_xticks([min_row_completeness,  max_row_completeness])
                axv.set_xticklabels(['{:.1f}'.format(min_row_completeness/n_rows),
                                     '{:.1f}'.format(max_row_completeness/n_rows)],
                                    ha='center', va='top', fontsize=int(fontsize * 1.25))
                #axv.xaxis.tick_top()
                axv.set_yticks([])
                axv.set_xlabel(label, fontsize=int(fontsize))

                axh.spines['right'].set_visible(True)
                axh.set_yticks([min_col_completeness,  max_col_completeness])
                axh.set_yticklabels(['{:.1f}'.format(min_col_completeness/n_cols),
                                     '{:.1f}'.format(max_col_completeness/n_cols)],
                                    ha='left', va='center', fontsize=int(fontsize * 1.25))
                axh.yaxis.tick_right()
                axh.set_xticks([])


                #axh.set_yticks([min_col_completeness + (max_col_completeness - min_col_completeness) / 2])
                #axh.set_yticklabels([label], rotation=45, ha=ha, fontsize=fontsize)
                #axh.xaxis.tick_top()
                #axh.set_xticks([])
        else:
            axh.set_xticks([])
            axh.set_yticks([])
            axv.set_xticks([])
            axv.set_yticks([])

        # Remove tick mark (only works after plotting).
        axv.xaxis.set_ticks_position('none')
        axh.yaxis.set_ticks_position('none')

    return fig

#TODO: doesnt work
#TODO: what if there are multiple files to show ???
def show_image_file(filename,width=5,height=5):
    # reading png image file
    img = cv2.cvtColor(image.load_image(filename),cv2.COLOR_BGR2RGB)
    # show image
    plt.figure(figsize = (width,height))
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.show()

# TODO doesnt work with odd number of subplots
# Always supply 4 dim array : n, h, w, c
# if sending single image use np.expand_dims(img,axis=0)
# convert one hot labels with nputil.argmax(labels)

def image_display(images, titles=[], n_cols=1, figsize=(8,8)):
    """

    :param images: ndarray
    :param labels: list of labels
    :param n_cols: number of columns to display
    :param figsize: figure size in width,height
    :return:
    """
    if images.ndim < 4:
        raise Exception('image array is not 4D')
    if images.shape[3] > 3:
        raise Exception('3+ channels not supported')
    #TODO: Why are we doing this ??? do we really need to squeeze?
    if images.shape[3] == 1:
        images = np.squeeze(images,axis=3) # remove the last axis if it is 1
        #images.reshape(images.shape[0:3])

    n_rows = math.ceil(images.shape[0] / n_cols)
    #-(-images.shape[0] // n_cols)
    # double minus is for upside down floor division to get ceiling division

    fig, axs = plt.subplots(n_rows,n_cols,figsize=figsize)

    if n_rows * n_cols == 1:
        axs = [axs]
    else:
        axs = axs.flat

    """ old loop
    for image, label, ax in itertools.zip_longest(images, labels, axs):
        if image is not None:
            ax.imshow(image)
        if label is not None:
            ax.set_title(label)
        ax.axis('off')
    """

    for image, title, ax in itertools.zip_longest(images, titles, axs):
        if image is not None:
            ax.imshow(image)
        if title is not None:
            ax.set_title(title)
        #if (image is None) and (title is None):
        #    ax.remove()
        ax.axis('off')

    fig.subplots_adjust(hspace = 0)
    fig.tight_layout()
    return fig,axs
