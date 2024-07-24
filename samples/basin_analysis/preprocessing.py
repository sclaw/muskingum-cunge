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
    if 'NA' in df.columns:
        df = df.drop(columns='NA')
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

    # Establish file structure and paths
    if not os.path.exists(paths):
        create_project(paths)  # Optional make template
    with open(paths, 'r') as f:
        paths = json.load(f)
    lateral_path = paths['lateral_path']
    inflow_path = paths['inflow_path']
    out_path = paths['meta_path']
    geom_dir = paths['geometry_dir']

    ds_hydrographs = os.path.join(paths['base_dir'], 'downstream_hydrographs.csv')
    head_path = os.path.join(paths['base_dir'], 'headwaters.csv')
    lake_path = os.path.join(paths['base_dir'], 'lake.csv')
    
    meta_path = r"/netfiles/ciroh/floodplainsData/runs/NWM/network/reach_data.csv"
    param_path = r"/netfiles/ciroh/floodplainsData/runs/NWM/working/RouteLink_CONUS.nc"
    da_path = r"/netfiles/ciroh/floodplainsData/runs/NWM/working/drain_areas.csv"
    if os.path.exists(out_path):
        print(f'{out_path} already exists, skipping')
        continue
    
    # Load data
    ds = pd.read_csv(ds_hydrographs)  # treating this as the model domain
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
            lake_root = lake_reaches[0]  # initialize
            outflow = meta_df.loc[lake_root, 'ds_comid']
            while outflow in lake_reaches:
                lake_root = meta_df.loc[lake_root, 'ds_comid']  # update lake root
                outflow = meta_df.loc[lake_root, 'ds_comid']
            force_lakes.append(str(int(lake_root)))
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

    # Check that all final network have valid geometry
    geom_area_path = os.path.join(geom_dir, 'area.csv')
    geom_area = pd.read_csv(geom_area_path)
    valid_geom = geom_area.columns[~(geom_area == 0).all(axis=0)].astype(int)
    missing_geom = list(set(meta_df.index).difference(set(valid_geom)))
    if len(missing_geom) > 0:
        print(f'Missing geometry for {len(missing_geom)} reaches')
        print(missing_geom)
    meta_df['valid_geom'] = meta_df.index.isin(valid_geom)

    # make an inflows file
    headwaters = list(set(meta_df.index).difference(set(meta_df['ds_comid'])))
    headwaters = [str(h) for h in headwaters]
    try:
        inflow_df = ds[headwaters].copy()
    except KeyError:
        missing = list(set(headwaters).difference(set(ds.columns)))
        missing_ds = pd.DataFrame(index=missing, columns=['missing_ds'])
        missing_ds['missing_ds'] = True
        missing_ds.to_csv(os.path.join(paths['base_dir'], 'missing_ds.csv'))
        raise RuntimeError(f'Missing downstream reaches in {ds_hydrographs}')
    inflow_df.to_csv(inflow_path)

    # Make an initial conditions file
    init_flow = pd.DataFrame(ds.loc[ds.index[0], meta_df.index.astype(str)])
    init_flow = init_flow.rename(columns={ds.index[0]: 'init_discharge'})
    init_flow.to_csv(paths['init_conditions_path'])

    # Export metadata
    meta_df['comid'] = meta_df.index
    meta_df = meta_df[['comid', 'ds_comid', 'n', 'nCC', 'order', 'TotDASqKm', 'length', 'slope', 'ChSlp', 'BtmWdth', 'TopWdthCC', 'TopWdth', 'NHDWaterbodyComID', 'MusK', 'MusX', 'Kchan', 'valid_geom']]
    meta_df.to_csv(out_path, index=False)
