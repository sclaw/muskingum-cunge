import pandas as pd
import numpy as np
import json
import os
from muskingumcunge.controller import create_project
import netCDF4 as nc


# Load data
# paths = r"/users/k/l/klawson1/netfiles/ciroh/floodplainsData/retrospective/lamoille/2002/mc_run.json"
# paths = r"/users/k/l/klawson1/netfiles/ciroh/floodplainsData/retrospective/lamoille/2019/mc_run.json"
paths = r"/users/k/l/klawson1/netfiles/ciroh/floodplainsData/retrospective/otter/2011/mc_run.json"
# paths = r"/users/k/l/klawson1/netfiles/ciroh/floodplainsData/retrospective/otter/2019/mc_run.json"
# paths = r"/users/k/l/klawson1/netfiles/ciroh/floodplainsData/retrospective/lewis/1996/mc_run.json"


def clean(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    if 'datetime' in df.columns:
        df = df[df['datetime'].notnull()]
        df["datetime"] = df["datetime"].map(lambda x: x + " 00:00:00" if len(x) == 10 else x)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    return df

def find_root(reaches):
    dataset = nc.Dataset(param_path)
    comids = np.array(dataset.variables['link'][:])
    to = np.array(dataset.variables['to'][:])
    outflow_dict = {us: ds for us, ds in zip(comids, to)}
    r1 = reaches[0]  # just need to trace one reach to outlet
    outflow = outflow_dict[r1]
    while outflow in reaches:
        r1 = outflow
        outflow = outflow_dict[r1]
    return r1

def extract_from_root(root):
    print(f'Extracting data from root {root}')
    print(f'Loading parameters from {param_path}')
    dataset = nc.Dataset(param_path)
    comids = np.array(dataset.variables['link'][:])
    to = np.array(dataset.variables['to'][:])

    print('traversing network...')
    keep = list()
    q = [root]
    while len(q) > 0:
        cur = q.pop()
        children = comids[to == cur]
        q.extend(children)
        keep.extend(children)

    print('extracting subset...')
    indices = np.where(np.isin(comids, keep))
    data = {}
    for var_name in ['link', 'to', 'n', 'nCC', 'order', 'Length', 'So', 'ChSlp', 'BtmWdth', 'TopWdthCC', 'TopWdth', 'NHDWaterbodyComID', 'MusK', 'MusX', 'Kchan']:
        var_data = dataset.variables[var_name][:]
        if var_data.ndim == 0:
            continue
        elif var_data.ndim > 1:
            data[var_name] = var_data[indices, :]
        else:
            data[var_name] = var_data[indices]

    # Close the dataset
    dataset.close()

    print('creating DataFrame...')
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df


run_dir = r'/netfiles/ciroh/floodplainsData/retrospective/basin_run_files'
all_paths = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.json')]

for paths in all_paths:
    print(f'Processing {paths}')

    # load data
    if not os.path.exists(paths):
        create_project(paths)  # Optional make template
    with open(paths, 'r') as f:
        paths = json.load(f)
    lateral_path = paths['lateral_path']
    head_path = paths['head_path']
    lake_path = paths['lake_path']
    out_path = paths['meta_path']
    if os.path.exists(out_path):
        continue
    meta_path = r"/users/k/l/klawson1/netfiles/ciroh/floodplainsData/runs/NWM/network/reach_data.csv"
    param_path = r"/users/k/l/klawson1/netfiles/ciroh/floodplainsData/runs/NWM/working/RouteLink_CONUS.nc"
    da_path = r"/users/k/l/klawson1/netfiles/ciroh/floodplainsData/runs/NWM/working/drain_areas.csv"
    geom_dir = paths['geometry_dir']
    ds_hydrographs = os.path.join(paths['base_dir'], 'downstream_hydrographs.csv')

    ds = pd.read_csv(ds_hydrographs)
    ds = clean(ds)
    reaches = [int(i) for i in ds.columns]
    root = find_root(reaches)
    meta_df = extract_from_root(root)
    meta_df = meta_df.sort_values("link")
    meta_df = meta_df.rename(columns={"link": paths['id_field'], "to": paths['to_id_field'], 'Length': 'length', 'So': 'slope'})
    meta_df = meta_df.drop_duplicates()  # remove completely duplicated rows
    meta_df = meta_df.set_index("comid")

    # load DAs
    das = pd.read_csv(da_path)
    das = das.set_index("ID")
    meta_df = meta_df.join(das, how='left')
    replace_nan = lambda x: np.nanmean(meta_df['TotDASqKm']) if np.isnan(x) else x
    meta_df['TotDASqKm'] = meta_df['TotDASqKm'].map(replace_nan)  # maybe should update later

    # Remove divergences by keeping largest connection first
    count = len(meta_df)
    meta_df = meta_df.sort_values('order', ascending=False)
    meta_df = meta_df[~meta_df.index.duplicated(keep='first')].copy().sort_index()
    print(f'Removed {count - len(meta_df)} divergences')

    # Check for single root
    roots = list(set(meta_df['ds_comid']).difference(set(meta_df.index)))
    if len(roots) > 1:
        raise RuntimeError(f'Multiple roots detected: {roots}')

    # Handle missing data by grabbing next ds reach
    missing = meta_df[meta_df[['n', 'nCC', 'slope', 'ChSlp', 'BtmWdth', 'TopWdthCC', 'TopWdth', 'TotDASqKm']].isnull().values.any(axis=1)].index
    print(f'Handling missing data for {len(missing)} reaches')
    print(missing)
    for col in missing:
        isna = True
        ds = col
        while isna:
            ds = meta_df.loc[ds, 'ds_comid']
            if ds in missing:
                continue
            else:
                isna = False
        if ds == roots[0]:
            meta_df.loc[col, ['n', 'ncc', 'TotDASqKm', 'slope']] = [0.06, 0.12, meta_df['TotDASqKm'].max(), np.exp(np.mean(np.log(meta_df['slope'])))]
        else:
            meta_df.loc[col, ['n', 'ncc', 'TotDASqKm', 'slope']] = meta_df.loc[ds, ['n', 'ncc', 'TotDASqKm', 'slope']]
        meta_df.loc[col, 'length'] = meta_df['length'].mean()

    # Add lakes to headwater forcings
    try:
        lake_df = pd.read_csv(lake_path) 
        tmp_cols = list(lake_df.columns)
        tmp_cols.remove('Unnamed: 0')
        tmp_cols.remove('datetime')
        lakes = [float(c) for c in tmp_cols]
        force_lakes = list()
        for l in lakes:
            # Find root of each lake
            lake_reaches = list(meta_df[meta_df['NHDWaterbodyComID'] == l].index)
            lake_root = lake_reaches[0]
            outflow = meta_df.loc[lake_root, 'ds_comid']
            while outflow in lake_reaches:
                lake_root = meta_df.loc[lake_root, 'ds_comid']
                outflow = meta_df.loc[lake_root, 'ds_comid']
            force_lakes.append(str(int(lake_root)))
        # ds[force_lakes].to_csv(os.path.join(paths['base_dir'], 'lake2.csv'), index=False)
    except:
        force_lakes = list()

    # Prune upstream of headwater forcings
    head_df = pd.read_csv(head_path)  # reaches we don't want to route through
    tmp_cols = list(head_df.columns)
    tmp_cols.remove('Unnamed: 0')
    tmp_cols.remove('datetime')
    heads = [float(c) for c in tmp_cols]
    small = list(meta_df[meta_df['TotDASqKm'] < 5.2].index)  # also treat small reaches as headwaters
    heads.extend(small)
    heads.extend([float(i) for i in force_lakes])
    prune_counter = 0
    for h in heads:
        us = list(meta_df[meta_df['ds_comid'] == h].index)
        while len(us) > 0:
            tmp = us.pop()
            us.extend(list(meta_df[meta_df['ds_comid'] == tmp].index))
            meta_df = meta_df.drop(index=tmp).copy()
            prune_counter += 1
    print(f'Removed {prune_counter} reaches to align with headwater forcings')

    # # Prune upstream of lake forcings
    # lake_df = pd.read_csv(lake_path) 
    # tmp_cols = list(lake_df.columns)
    # tmp_cols.remove('Unnamed: 0')
    # tmp_cols.remove('datetime')
    # lakes = [float(c) for c in tmp_cols]
    # prune_counter = 0
    # for l in lakes:
    #     # Find root of each lake
    #     lake_reaches = list(meta_df[meta_df['NHDWaterbodyComID'] == l].index)
    #     lake_root = lake_reaches[0]
    #     outflow = meta_df.loc[lake_root, 'ds_comid']
    #     while outflow in lake_reaches:
    #         lake_root = meta_df.loc[lake_root, 'ds_comid']
    #         outflow = meta_df.loc[lake_root, 'ds_comid']
    #     lake_root_cache = meta_df.loc[lake_root]
    #     us = list(meta_df[meta_df['ds_comid'] == lake_root].index)
    #     while len(us) > 0:
    #         tmp = us.pop()
    #         us.extend(list(meta_df[meta_df['ds_comid'] == tmp].index))
    #         meta_df = meta_df.drop(index=tmp).copy()
    #         prune_counter += 1
    #     meta_df.loc[lake_root] = lake_root_cache
    # print(f'Removed {prune_counter} reaches to align with lake forcings')

    # Export metadata
    meta_df['comid'] = meta_df.index
    meta_df = meta_df[['comid', 'ds_comid', 'n', 'nCC', 'order', 'TotDASqKm', 'length', 'slope', 'ChSlp', 'BtmWdth', 'TopWdthCC', 'TopWdth', 'NHDWaterbodyComID', 'MusK', 'MusX', 'Kchan']]
    meta_df.to_csv(out_path, index=False)
