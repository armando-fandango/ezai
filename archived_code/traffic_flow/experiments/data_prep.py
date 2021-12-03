import os

import archived_code.ezai.data.temporal
from archived_code.ezai.data import temporal, ritis
from archived_code.ezai.util import util
from archived_code.ezai.util import filesystem_util
from archived_code.ezai.util import log_util
from archived_code.ezai.util import dict_util
from archived_code.ezai import scalers

import pandas as pd
import numpy as np

#TODO Replace os.path with pathlib.Path

logger = log_util.get_logger(__name__)

def n3_build_data(rd: ritis.RITISDetector, conf, expdata_home, n_adj_id=0):
    # adj_id 0 means select all
    # 1-5 speed field only
    # 1-15 speed field with 15 min agg
    # 3-5 speed, vol, occ fields
    # 3-15 speed, vol, occ fields with 15 min agg
    # f - no derived features added
    # tf - time features added - hour of day, day of week
    # sf - spatial features added - previous and next id
    # stf - spatial and time features added
    #           <>-prep-<prep_format_id> for raw data pre-processed in parquet
    #               one set of raw data can be prepped in multiple prep folders
    #           <>-num-<numeric_format_id> for data in numeric format e.g. npz, ready for ML
    #               one raw -> n prep -> m numeric formats
    #           <>/exp/<experiment_id> for experiments e,g, pems_d5-n11_id30
    #               on each numeric format we may run multiple experiments

    # folder:   <> is dataset_home/
    #           <>/exp_<>/ for data in numeric format e.g. npz, ready for ML
    #               one raw -> n prep -> m numeric formats
    rd.filter_by_common_id() #just in case it wasnt done before
    conf = conf.deepcopy()

    ID = ritis.ID
    DT = ritis.DT
    SPD = ritis.SPD
    VOL = ritis.VOL
    OCC = ritis.OCC
    zone = rd.df.zone
    meta = rd.df.meta

    filesystem_util.makedir(expdata_home)

    # Get the proper grouping from meta for loop by road, direction
    road_list = list(meta.road.unique())
    dir_list = list(meta.direction.unique())
    x_cols_list = [['speed'], ['speed', 'volume', 'occupancy']]
    n_vx_list = [len(l) for l in x_cols_list]
    n_agg_list = [5,10,15]
    derived_features_list = ['f', 'tf', 'sf', 'stf']
    dict_util.save_to_json({'road': road_list,
                            'dir':dir_list,
                            'x_cols':x_cols_list,
                            'n_agg':n_agg_list,
                            'derived_features':derived_features_list
                            },
                           os.path.join(expdata_home, 'trial_lists.json'))

    meta_gdf = meta.sort_values([ID]).groupby(['road', 'direction'],observed=True)
    print('# of ids:')
    print(meta_gdf.id.count())
    # keep only those groups that have id's at leat one more than
    #  number of adjacent ids required
    # TODO: Why... if no adjacent then groups that have at least two sensors?
    meta_gdf = meta_gdf.filter(lambda x: x.id.count() > max(n_adj_id + 1, 2))
    # get top n_adj_id+2 from each highway
    if n_adj_id:
        meta_gdf = meta_gdf.groupby(['road', 'direction'],observed=True)\
                       .head(n_adj_id + 2)
    meta_gdf = meta_gdf.groupby(['road', 'direction'],observed=True)
    print('# of ids after filtering for n_adj_id = {}:'.format(n_adj_id))
    print(meta_gdf.id.count())



    for meta_idx, meta_grp in meta_gdf:
        # loop for getting contiguous id on each (highway, direction pair)
        id_list = meta_grp.id.unique().tolist()

        conf.road = meta_idx[0]
        conf.direction = meta_idx[1]

        for cols in x_cols_list:
            conf.x_cols = cols
            conf.n_vx = len(cols)

            # load only selected columns and selected ids, months

            df = zone.loc[(zone[ID].isin(id_list)), [DT, ID] + conf.x_cols]
#            print('df',df.loc[df.id=='6399',:])
            if conf.months:
                df = df.loc[df[DT].dt.month.isin(conf.months), : ]
            else:
                conf.months = df[DT].dt.month.unique().tolist() # for sake of saving

            for n_agg in n_agg_list:
                conf.n_agg = n_agg
                conf.n_tx_step = n_agg // 5
                conf.n_ty_step = 30 // 5
                if conf.n_tx_duration:
                    conf.n_tx = conf.n_tx_duration // (n_agg // 5)
                expdid = '{}-{}-{}-{}-{}'.format(rd.subset_name, conf.road,
                                                 conf.direction, conf.n_vx,
                                                 conf.n_agg)
                parquet_folder = os.path.join(expdata_home, expdid)
                filesystem_util.makedir(parquet_folder)
                print('saving id_list.json and <id>.parquet files in \n {}'
                      .format(parquet_folder))
                # Not saving first and last for spatial analysis purpose
                dict_util.save_to_json({'id_list': id_list[1:-1]},
                                       os.path.join(parquet_folder, 'id_list.json'))
                meta_grp.to_parquet(os.path.join(
                    parquet_folder, 'meta.parquet'),
                    engine='pyarrow')
                id_dict = {}

                #print('df',df.loc[df.id=='6399',:])
                for id_key, id_grp in df.groupby([ID],observed=True):
                    tgdf = id_grp.drop(columns=[ID])
                    idf_x = pd.DataFrame(columns=tgdf.columns)
                    idf_y = pd.DataFrame(columns=tgdf.columns)

                    #print('tgdf',id_key,tgdf.head())
                    #print('idf',idf.head())
                    for dt_key, dt_grp in tgdf.groupby(
                            [tgdf[DT].dt.year, tgdf[DT].dt.month],observed=True):
                        if conf.trim_partial_weeks:
                            from_ts = temporal.next_weekday(dt_grp[DT].min())
                            to_ts = temporal.next_weekday(dt_grp[DT].max(),
                                                          weekday=6,
                                                          next=False)
                            tdf = dt_grp[((dt_grp[DT] >= from_ts) &
                                          (dt_grp[DT] <= to_ts))]
                        else:
                            tdf = dt_grp
                        # interpolate all available weeks for one single id
                        tdf_x = archived_code.ezai.data.temporal.interpolate(tdf, DT, freq='5T',
                                                                             new_freq='{}T'.format(conf.n_agg))
                        tdf_y = archived_code.ezai.data.temporal.interpolate(tdf, DT, freq='5T',
                                                                             new_freq='30T'.format(conf.n_agg))
                        #TODO: replace above with n_ty_step * 5T
                        idf_x = idf_x.append(tdf_x, ignore_index=True)
                        idf_y = idf_y.append(tdf_y, ignore_index=True)

                # idf has interpolated data of selected id, all months and years
                    # take only selected weekdays
                    if conf.weekdays:
                        idf_x = idf_x[idf_x[DT].dt.weekday.isin(conf.weekdays)]
                        idf_y = idf_y[idf_y[DT].dt.weekday.isin(conf.weekdays)]
                    idf_x = idf_x.set_index(DT).loc[:,conf.x_cols]
                    idf_y = idf_y.set_index(DT).loc[:,conf.y_cols]
                    id_dict[id_key] = (idf_x,idf_y)
                    # save interpolated data and metadata for each zone
                    idf_x.to_parquet(os.path.join(parquet_folder,
                                                '{}-x.parquet'.format(id_key)),
                                   engine='pyarrow')
                    idf_y.to_parquet(os.path.join(parquet_folder,
                                                  '{}-y.parquet'.format(id_key)),
                                     engine='pyarrow')

                    print('id=', id_key, 'x,y shapes=', idf_x.shape,idf_y.shape)

                # save ids covered in this data
                print('Saving NPZ and conf.json files:')
                for derived_features in derived_features_list:
                    conf.derived_features = derived_features
                    data_folder = os.path.join(parquet_folder,
                                               conf.derived_features)
                    filesystem_util.makedir(data_folder)
                    print(data_folder)
                    # now lets do feature engineering and save

                    for i in range(1, len(id_list) - 1):
                        conf1 = conf.deepcopy()
                        conf1.id = id_list[i]
                        id_df_x = id_dict[conf1.id][0].copy()
                        id_df_y = id_dict[conf1.id][1].copy()

                        if 's' in conf1.derived_features:
                            conf1.id_prev = id_list[i - 1]
                            conf1.id_next = id_list[i + 1]

                            x_dfs = [
                                id_dict[conf1.id_prev][0], id_df_x,
                                id_dict[conf1.id_next][0]
                            ]

                            k = np.arange(len(x_dfs)).astype(str)
                            id_df_x = pd.concat(x_dfs, join='inner', axis=1, keys=k)
                            id_df_x.columns = id_df_x.columns.map('_'.join)
                            conf1.x_cols = list(id_df_x.columns)
                            conf1.y_cols = ['1_speed']
                            id_df_y.columns = conf1.y_cols
                            conf1.n_vx = len(conf1.x_cols)

                        if 't' in conf1.derived_features:
                            conf1.x_cols.append('dow')
                            conf1.x_cols.append('hod')

                            conf1.n_vx += 2

                            id_df_x['dow'] = id_df_x.index.dayofweek
                            id_df_x['hod'] = id_df_x.index.hour

                        conf1.xy_cols = list(set(conf1.x_cols+conf1.y_cols))
                        #print('conf1.xcols ',conf1.x_cols)
                        #print('conf1.ycols ',conf1.y_cols)
                        #print('conf1.xycols ',conf1.xy_cols)
                        conf_filename = os.path.join(
                            data_folder, '{}-conf.json'.format(conf1.id))
                        conf1.save_to_json(conf_filename)
                        #zid = conf1.id
                        # Why are we trying to convert index to column here ?
                        # Because we need to group by on weeks as partial weeks is true in config
                        id_df_x = id_df_x.reset_index(drop=False)
                        id_df_y = id_df_y.reset_index(drop=False)

                        #print('saving ....',zid)
                        # create temporal dataset with dataframe
                        # scale to minmax and save scaler
                        #print(conf1.xy_cols)
                        #print(id_df.head())
                        scaler_x = scalers.MinMaxScaler()\
                            .fit(id_df_x.loc[:,conf1.x_cols].to_numpy())
                        #td.scaler_y = MinMaxScaler().fit(td.get_ycols_as_np())
                        #td.scaler_x = MinMaxScaler().fit(td.get_xcols_as_np())

                        id_df_x[conf1.x_cols] = scaler_x.transform(id_df_x[conf1.x_cols])
                        #id_df_y[conf1.y_cols] = scaler_x.transform(id_df_y[conf1.y_cols])

                        # convert to xy by year, month, week and then append together
                        # to df wont work anymore because we are converting to dim3
                        xy = n3_to_xy((id_df_x,id_df_y), conf=conf1)  # dim3=True by default
                        # td.tvt_xy_split()
                        #filename = os_path.join(data_folder, '{}-scaler-y.pkl'.format(conf1.id))
                        #pickle_dump(td.scaler_y, open(filename, 'wb'))

                        #filename = os_path.join(data_folder, '{}-scaler-x.pkl'.format(conf1.id))
                        #pickle_dump(td.scaler_x, open(filename, 'wb'))

                        filename = os.path.join(data_folder, '{}.npz'.format(conf1.id))
                        np.savez_compressed(filename, x=xy[0], y=xy[1].reshape(-1, conf1.n_ty))
                        #print('done saving ',zid)

    print("all done")

def n3_to_xy(dfxy, conf, dim3 = True):

    # if dim3 is true then get 3D x and Y numpy arrays
    # if dim3 is False then get merged xy df
    n_tx = conf.n_tx
    n_ty = conf.n_ty
    x_cols = conf.x_cols
    y_cols = conf.y_cols
    n_tx_step = conf.n_tx_step
    n_ty_step = conf.n_ty_step
    h = conf.h #TODO: Handle edge case h<1 as h should be >=1

    df_x = dfxy[0]
    df_y = dfxy[1]

    # convert to xy by id, year, month, week and then append together
    #TODO not all data would be weekly hence fix this
    gdf_x = df_x.groupby(
        [df_x['dt'].dt.year, df_x['dt'].dt.month, df_x['dt'].dt.isocalendar().week])

    gdf_y = df_y.groupby(
        [df_y['dt'].dt.year, df_y['dt'].dt.month, df_y['dt'].dt.isocalendar().week])

    #        n_rows = td_rows - (n_tx + n_ty - 1 + h - 1)
    #        shapeX = (n_rows, n_tx, n_vx) if dim3 else (n_rows, n_vx * n_tx)
    #        shapeY = (n_rows, n_ty, n_vy) if dim3 else (n_rows, n_vy * n_ty)
    # python list is much faster then the one below
    #xy_np = [np.empty((0,n_tx,n_vx)), np.empty((0,n_ty,n_vy))]
    xy_np = [[],[]]
    xy_df = pd.DataFrame()
    for key, group_x in gdf_x:
        group_y = gdf_y.get_group(key)
        #util.sflush()
        #print('for week ',key,' #records: ',len(group))
        n_rows = group_y.shape[0] - (n_tx // (n_ty_step // n_tx_step)) + h - 1
        if n_rows < 10:
            print('not enough rows, only {}, thus skipping {}'.format(len(group_x),key))
        else:
            xy, _ = temporal.dfs_to_xy(group_x, group_y,
                                       n_tx=n_tx,
                                       n_ty=n_ty,
                                       n_tx_step=n_tx_step,
                                       n_ty_step=n_ty_step,
                                       h=h,
                                       x_cols=x_cols,
                                       y_cols=y_cols,
                                       dim3 = dim3)
            if dim3:
                #xy_np[0].append(xy_np[0],[xy[0]], axis=0)
                #xy_np[1].append(xy_np[1],[xy[1]], axis=0)
                # python list append is faster then above
                xy_np[0].append(xy[0])  # obj_n is numpy arrays x and y
                xy_np[1].append(xy[1])
            else:
                #xy_df, xy_x_cols, xy_y_cols = xy
                xy_df = xy_df.append(xy, ignore_index=True)
                #xy_list.append(xy) # obj_1 is dataframe

    if dim3:
        xy = [np.vstack(xy_np[0]),np.vstack(xy_np[1])]
    else:
        # TODO:if it starts taking more memory then concat might not work.
        xy = xy_df #pd.concat(xy_list)
    return xy

def n2_build_data(rd: ritis.RITISDetector, conf, expdata_home, n_adj_id=0):
    # adj_id 0 means select all
    # 1-5 speed field only
    # 1-15 speed field with 15 min agg
    # 3-5 speed, vol, occ fields
    # 3-15 speed, vol, occ fields with 15 min agg
    # f - no derived features added
    # tf - time features added - hour of day, day of week
    # sf - spatial features added - previous and next id
    # stf - spatial and time features added
    #           <>-prep-<prep_format_id> for raw data pre-processed in parquet
    #               one set of raw data can be prepped in multiple prep folders
    #           <>-num-<numeric_format_id> for data in numeric format e.g. npz, ready for ML
    #               one raw -> n prep -> m numeric formats
    #           <>/exp/<experiment_id> for experiments e,g, pems_d5-n11_id30
    #               on each numeric format we may run multiple experiments

    # folder:   <> is dataset_home/
    #           <>/exp_<>/ for data in numeric format e.g. npz, ready for ML
    #               one raw -> n prep -> m numeric formats
    rd.filter_by_common_id() #just in case it wasnt done before
    conf = conf.deepcopy()

    ID = ritis.ID
    DT = ritis.DT
    SPD = ritis.SPD
    VOL = ritis.VOL
    OCC = ritis.OCC
    zone = rd.df.zone
    meta = rd.df.meta

    # Get the proper grouping from meta for loop by road, direction

    meta_gdf = meta.sort_values([ID]).groupby(['road', 'direction'],observed=True)
    print('# of ids:')
    print(meta_gdf.id.count())
    # keep only those groups that have id's at leat one more than
    #  number of adjacent ids required
    # TODO: Why... if no adjacent then groups that have at least two sensors?
    meta_gdf = meta_gdf.filter(lambda x: x.id.count() > max(n_adj_id + 1, 2))
    # get top n_adj_id+2 from each highway
    if n_adj_id:
        meta_gdf = meta_gdf.groupby(['road', 'direction'],observed=True) \
            .head(n_adj_id + 2)
    meta_gdf = meta_gdf.groupby(['road', 'direction'],observed=True)
    print('# of ids after filtering for n_adj_id = {}:'.format(n_adj_id))
    print(meta_gdf.id.count())

    filesystem_util.makedir(expdata_home)

    for meta_idx, meta_grp in meta_gdf:
        # loop for getting contiguous id on each (highway, direction pair)
        id_list = meta_grp.id.unique().tolist()

        conf.road = meta_idx[0]
        conf.direction = meta_idx[1]

        for cols in [['speed'], ['speed', 'volume', 'occupancy']]:
            conf.x_cols = cols
            conf.n_vx = len(cols)

            # load only selected columns and selected ids, months

            df = zone.loc[(zone[ID].isin(id_list)), [DT, ID] + conf.x_cols]
            #            print('df',df.loc[df.id=='6399',:])
            if conf.months:
                df = df.loc[df[DT].dt.month.isin(conf.months), : ]
            else:
                conf.months = df[DT].dt.month.unique().tolist() # for sake of saving

            for n_agg in [5, 10, 15]:
                conf.n_agg = n_agg

                expdid = '{}-{}-{}-{}-{}'.format(rd.subset_name, conf.road,
                                                 conf.direction, conf.n_vx,
                                                 conf.n_agg)
                parquet_folder = os.path.join(expdata_home, expdid)
                filesystem_util.makedir(parquet_folder)
                print('saving id_list.json and <id>.parquet files in \n {}'
                      .format(parquet_folder))
                # Not saving first and last for spatial analysis purpose
                dict_util.save_to_json({'id_list': id_list[1:-1]},
                                       os.path.join(parquet_folder, 'id_list.json'))
                id_dict = {}

                #print('df',df.loc[df.id=='6399',:])
                for id_key, id_grp in df.groupby([ID],observed=True):
                    tgdf = id_grp.drop(columns=[ID])
                    idf = pd.DataFrame(columns=tgdf.columns)
                    #print('tgdf',id_key,tgdf.head())
                    #print('idf',idf.head())
                    for dt_key, dt_grp in tgdf.groupby(
                            [tgdf[DT].dt.year, tgdf[DT].dt.month],observed=True):
                        if conf.trim_partial_weeks:
                            from_ts = temporal.next_weekday(dt_grp[DT].min())
                            to_ts = temporal.next_weekday(dt_grp[DT].max(),
                                                          weekday=6,
                                                          next=False)
                            tdf = dt_grp[((dt_grp[DT] >= from_ts) &
                                          (dt_grp[DT] <= to_ts))]
                        else:
                            tdf = dt_grp
                        # interpolate all available weeks for one single id
                        tdf = archived_code.ezai.data.temporal.interpolate(tdf, DT, freq='5T',
                                                                           new_freq='{}T'.format(conf.n_agg))
                        idf = idf.append(tdf, ignore_index=True)
                    # idf has interpolated data of selected id, all months and years
                    # take only selected weekdays
                    if conf.weekdays:
                        idf = idf[idf[DT].dt.weekday.isin(conf.weekdays)]
                    id_dict[id_key] = idf.set_index(DT)
                    # save interpolated data and metadata for each zone
                    meta_grp.to_parquet(os.path.join(
                        parquet_folder, '{}-meta.parquet'.format(id_key)),
                        engine='pyarrow')
                    idf.to_parquet(os.path.join(parquet_folder,
                                                '{}-data.parquet'.format(id_key)),
                                   engine='pyarrow')
                    print('id=', id_key, 'data_items=', len(idf))

                # save ids covered in this data
                print('Saving NPZ and conf.json files:')
                for derived_features in ['f', 'tf', 'sf', 'stf']:
                    conf.derived_features = derived_features
                    data_folder = os.path.join(parquet_folder,
                                               conf.derived_features)
                    filesystem_util.makedir(data_folder)
                    print(data_folder)
                    # now lets do feature engineering and save

                    for i in range(1, len(id_list) - 1):
                        conf1 = conf.deepcopy()
                        conf1.id = id_list[i]
                        id_df = id_dict[conf1.id].copy()

                        if 's' in conf1.derived_features:
                            conf1.id_prev = id_list[i - 1]
                            conf1.id_next = id_list[i + 1]

                            dfs = [
                                id_dict[conf1.id_prev], id_df,
                                id_dict[conf1.id_next]
                            ]
                            k = np.arange(len(dfs)).astype(str)
                            id_df = pd.concat(dfs, join='inner', axis=1, keys=k)
                            id_df.columns = id_df.columns.map('_'.join)
                            conf1.x_cols = list(id_df.columns)
                            conf1.y_cols = ['1_speed']
                            conf1.n_vx = len(conf1.x_cols)

                        if 't' in conf1.derived_features:
                            conf1.x_cols.append('dow')
                            conf1.x_cols.append('hod')

                            conf1.n_vx += 2

                            id_df['dow'] = id_df.index.dayofweek
                            id_df['hod'] = id_df.index.hour

                        conf1.xy_cols = list(set(conf1.x_cols+conf1.y_cols))
                        conf_filename = os.path.join(
                            data_folder, '{}-conf.json'.format(conf1.id))
                        conf1.save_to_json(conf_filename)
                        #zid = conf1.id
                        id_df = id_df.reset_index(drop=False)
                        #print('saving ....',zid)
                        # create temporal dataset with dataframe
                        # scale to minmax and save scaler
                        #print(conf1.xy_cols)
                        #print(id_df.head())
                        scaler_xy = scalers.MinMaxScaler().fit(id_df[conf1.xy_cols].to_numpy())
                        #td.scaler_y = MinMaxScaler().fit(td.get_ycols_as_np())
                        #td.scaler_x = MinMaxScaler().fit(td.get_xcols_as_np())

                        id_df[conf1.xy_cols] = scaler_xy.transform(id_df[conf1.xy_cols])
                        # convert to xy by year, month, week and then append together
                        # to df wont work anymore because we are converting to dim3
                        xy = n2_to_xy(id_df, conf=conf1)  # dim3=True by default
                        # td.tvt_xy_split()
                        #filename = os_path.join(data_folder, '{}-scaler-y.pkl'.format(conf1.id))
                        #pickle_dump(td.scaler_y, open(filename, 'wb'))

                        #filename = os_path.join(data_folder, '{}-scaler-x.pkl'.format(conf1.id))
                        #pickle_dump(td.scaler_x, open(filename, 'wb'))

                        filename = os.path.join(data_folder, '{}.npz'.format(conf1.id))
                        np.savez_compressed(filename, x=xy[0], y=xy[1].reshape(-1, conf1.n_ty))
                        #print('done saving ',zid)

    print("all done")

def n2_to_xy(df, conf, dim3 = True):

    # if dim3 is true then get 3D x and Y numpy arrays
    # if dim3 is False then get merged xy df
    n_tx = conf.n_tx
    n_ty = conf.n_ty
    x_cols = conf.x_cols
    y_cols = conf.y_cols
    h = conf.h #TODO: Handle edge case h<1 as h should be >=1

    # convert to xy by id, year, month, week and then append together
    #TODO not all data would be weekly hence fix this
    gdf = df.groupby(
        [df['dt'].dt.year, df['dt'].dt.month, df['dt'].dt.isocalendar().week])

    #        n_rows = td_rows - (n_tx + n_ty - 1 + h - 1)
    #        shapeX = (n_rows, n_tx, n_vx) if dim3 else (n_rows, n_vx * n_tx)
    #        shapeY = (n_rows, n_ty, n_vy) if dim3 else (n_rows, n_vy * n_ty)
    # python list is much faster then the one below
    #xy_np = [np.empty((0,n_tx,n_vx)), np.empty((0,n_ty,n_vy))]
    xy_np = [[],[]]
    xy_df = pd.DataFrame()
    for key, group in gdf:
        #util.sflush()
        #print('for week ',key,' #records: ',len(group))
        n_rows = len(group) - (n_tx + n_ty - 1 + h - 1)
        if n_rows < 10:
            print('not enough rows, only {}, thus skipping {}'.format(len(group),key))
        else:
            xy, _ = temporal.df_to_xy(group,
                                      n_tx=n_tx,
                                      n_ty=n_ty,
                                      h=h,
                                      x_cols=x_cols,
                                      y_cols=y_cols,
                                      dim3 = dim3)
            if dim3:
                #xy_np[0].append(xy_np[0],[xy[0]], axis=0)
                #xy_np[1].append(xy_np[1],[xy[1]], axis=0)
                # python list append is faster then above
                xy_np[0].append(xy[0])  # obj_n is numpy arrays x and y
                xy_np[1].append(xy[1])
            else:
                #xy_df, xy_x_cols, xy_y_cols = xy
                xy_rdf = xy_rdf.append(xy, ignore_index=True)
                #xy_list.append(xy) # obj_1 is dataframe

    if dim3:
        xy = [np.vstack(xy_np[0]),np.vstack(xy_np[1])]
    else:
        # TODO:if it starts taking more memory then concat might not work.
        xy = xy_rdf #pd.concat(xy_list)
    return xy

def build_ds_exp_1(rd: ritis.RITISDetector, conf, expdata_folder, n_adj_id=0):
    # adj_id 0 means select all
    # 1-5 speed field only
    # 1-15 speed field with 15 min agg
    # 3-5 speed, vol, occ fields
    # 3-15 speed, vol, occ fields with 15 min agg
    # f - no derived features added
    # tf - time features added - hour of day, day of week
    # sf - spatial features added - previous and next id
    # stf - spatial and time features added
    #           <>-prep-<prep_format_id> for raw data pre-processed in parquet
    #               one set of raw data can be prepped in multiple prep folders
    #           <>-num-<numeric_format_id> for data in numeric format e.g. npz, ready for ML
    #               one raw -> n prep -> m numeric formats
    #           <>/exp/<experiment_id> for experiments e,g, pems_d5-n11_id30
    #               on each numeric format we may run multiple experiments

    # folder:   <> is dataset_home/
    #           <>/exp_<>/ for data in numeric format e.g. npz, ready for ML
    #               one raw -> n prep -> m numeric formats
    rd.filter_by_common_id() #just in case it wasnt done before

    ID = ritis.ID
    DT = ritis.DT
    SPD = ritis.SPD
    VOL = ritis.VOL
    OCC = ritis.OCC
    zone = rd.df.zone
    meta = rd.df.meta



    meta_grouped = meta.groupby(['road', 'direction'])
    logger.info('# of ids:')
    logger.info(meta_grouped.id.count())

    expdata_folder = os.path.join(rd.dataset_home, 'exp',expdata_folder)
    filesystem_util.makedir(expdata_folder)

    meta_grouped = meta_grouped.filter(lambda x: x.id.count() > max(n_adj_id + 1, 2))
    # get top n_adj_id+2 from each highway
    if n_adj_id:
        meta_grouped = meta_grouped.groupby(['road', 'direction']).head(n_adj_id + 2)
    meta_grouped = meta_grouped.groupby(['road', 'direction'])
    print('# of ids after filtering for n_adj_id = {}:'.format(n_adj_id))
    print(meta_grouped.id.count())
    conf.setdefault('months', list(zone[DT].dt.month.unique()))
    conf.setdefault('trim_partial_weeks', True)

    for meta_idx, meta_grp in meta_grouped:  # loop for getting contiguous id on each (highway, direction pair)

        meta_grp = meta_grp.sort_values([ID])
        id_list = meta_grp.id.unique().tolist()
        conf.road = meta_idx[0]
        conf.direction = meta_idx[1]

        for cols in [['speed'], ['speed', 'volume', 'occupancy']]:
            conf.x_cols = cols
            conf.n_vx = len(cols)

            # load only selected columns and selected ids, months
            #                rd = rd.load(cols=[conf.dt_col, conf.id_col] + conf.x_cols)
            df = zone.loc[((zone[ID].isin(id_list)) & (
                zone[DT].dt.month.isin(conf.months))), [DT, ID] + conf.x_cols]

            for n_agg in [5, 10, 15]:
                conf.n_agg = n_agg

                expdid = '{}-{}-{}-{}-{}'.format(self.subset_name, conf.road,
                                                 conf.direction, conf.n_vx,
                                                 conf.n_agg)
                parquet_folder = os_path.join(expdata_folder, expdid)
                util.makedir(parquet_folder)
                print(parquet_folder)
                util.save_to_json({'id_list': id_list[1:-1]}, os_path.join(parquet_folder, 'id_list.json'))
                id_dict = {}
                for id_key, id_grp in df.groupby([ID]):
                    gdf = id_grp.drop(columns=[ID])
                    rdf = pd.DataFrame(columns=gdf.columns)

                    for dt_key, dt_grp in gdf.groupby(
                            [gdf[DT].dt.year, gdf[DT].dt.month]):

                        if conf.trim_partial_weeks:
                            from_ts = archived_code.ezai.data.temporal.next_weekday(dt_grp[DT].min())
                            to_ts = archived_code.ezai.data.temporal.next_weekday(dt_grp[DT].max(),
                                                                                  weekday=6,
                                                                                  next=False)
                            mdf = dt_grp[((dt_grp[DT] >= from_ts) &
                                          (dt_grp[DT] <= to_ts))]

                        else:
                            mdf = dt_grp
                        # interpolate all available weeks for one single id
                        mdf = archived_code.ezai.data.temporal.interpolate(mdf,
                                                                           DT,
                                                                           freq='5T',
                                                                           sfreq='{}T'.format(conf.n_agg))
                        rdf = rdf.append(mdf, ignore_index=True)

                    # rdf has interpolated data of selected id, all months and years
                    # take only selected weekdays
                    rdf = rdf[rdf[DT].dt.weekday.isin(conf.weekdays)]
                    id_dict[id_key] = rdf.set_index(DT)
                    # save interpolated data and metadata for each zone
                    meta_grp.to_parquet(os_path.join(
                        parquet_folder, '{}-meta.parquet'.format(id_key)),
                        engine='fastparquet')
                    rdf.to_parquet(os_path.join(parquet_folder,
                                                '{}-data.parquet'.format(id_key)),
                                   engine='fastparquet')

                    print('id=', id_key, 'data_items=', len(rdf))
                # save ids covered in this data
                for derived_features in ['f', 'tf', 'sf', 'stf']:
                    conf.derived_features = derived_features
                    data_folder = os.path.join(parquet_folder,
                                               conf.derived_features)
                    util.makedir(data_folder)
                    print(data_folder)
                    # now lets do feature engineering and save

                    for i in range(1, len(id_list) - 1):
                        conf1 = conf.deepcopy()
                        conf1.id = id_list[i]
                        id_df = id_dict[conf1.id].copy()

                        if 's' in conf1.derived_features:
                            conf1.id_prev = id_list[i - 1]
                            conf1.id_next = id_list[i + 1]

                            dfs = [
                                id_dict[conf1.id_prev], id_df,
                                id_dict[conf1.id_next]
                            ]
                            k = np.arange(len(dfs)).astype(str)
                            id_df = pd.concat(dfs, join='inner', axis=1, keys=k)
                            id_df.columns = id_df.columns.map('_'.join)
                            conf1.x_cols = list(id_df.columns)
                            conf1.y_cols = ['1_speed']
                            conf1.n_vx = len(conf1.x_cols)

                        if 't' in conf1.derived_features:
                            conf1.x_cols.append('dow')
                            conf1.x_cols.append('hod')
                            conf1.n_vx += 2

                            id_df['dow'] = id_df.index.dayofweek
                            id_df['hod'] = id_df.index.hour

                        conf_filename = os_path.join(
                            data_folder, '{}-conf.json'.format(conf1.id))
                        conf1.save_to_json(conf_filename)
                        #zid = conf1.id
                        id_df = id_df.reset_index(drop=False)

                        #print('saving ....',zid)
                        # create temporal dataset with dataframe
                        td = TemporalDataset(data=id_df,dt_col=DT,x_cols=conf1.x_cols,y_cols=conf1.y_cols)
                        # scale to minmax and save scaler
                        td.scaler_xy = MinMaxScaler().fit(td.get_xycols_as_np())
                        td.scaler_y = MinMaxScaler().fit(td.get_ycols_as_np())
                        td.scaler_x = MinMaxScaler().fit(td.get_xcols_as_np())
                        td.transform_xy_cols()
                        # convert to xy by year, month, week and then append together
                        # to df wont work anymore because we are converting to dim3
                        td.to_xy(n_tx=conf1.n_tx, n_ty=conf1.n_ty, h=conf1.h)  # dim3=True by default
                        # td.tvt_xy_split()
                        filename = os_path.join(data_folder, '{}-scaler-y.pkl'.format(conf1.id))
                        pickle_dump(td.scaler_y, open(filename, 'wb'))

                        filename = os_path.join(data_folder, '{}-scaler-x.pkl'.format(conf1.id))
                        pickle_dump(td.scaler_x, open(filename, 'wb'))

                        filename = os_path.join(data_folder, '{}.npz'.format(conf1.id))
                        np.savez_compressed(filename, x=td.xy_np[0], y=td.xy_np[1].reshape(-1, conf1.n_ty))
                        #print('done saving ',zid)

    print("all done")
