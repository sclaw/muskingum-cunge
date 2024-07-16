import os
import sys
import json
import pandas as pd
import numpy as np
from muskingumcunge.reach import CompoundReach, CustomReach, MuskingumReach, WRFCompound
from muskingumcunge.network import Network


def create_project(path):
    template = {
        "base_dir": os.path.dirname(path),
        "meta_path": os.path.join(os.path.dirname(path), "reach_meta.csv"),
        "lateral_path": os.path.join(os.path.dirname(path), "laterals.csv"),
        "head_path": os.path.join(os.path.dirname(path), "headwaters.csv"),
        "lake_path": os.path.join(os.path.dirname(path), "lake.csv"),
        "geometry_dir": "/users/k/l/klawson1/netfiles/ciroh/floodplainsData/runs/NWM/geometry",
        "geometry_source": "NWM",
        "optimize_dx": True,
        "conserve_mass": False,
        "lat_addition": "middle",
        "id_field": "comid",
        "to_id_field": "ds_comid",
        "mode": 'basin',
        "out_dir": os.path.join(os.path.dirname(path), 'outputs'),
    }
    os.makedirs(template['out_dir'], exist_ok=True)
    with open(path, 'w') as f:
        json.dump(template, f, indent=4)

def clean_ts(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    if 'datetime' in df.columns:
        df["datetime"] = df["datetime"].map(lambda x: x + " 00:00:00" if len(x) == 10 else x)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    return df

def load_geom(meta_df, source='NWM', geom_dir=None):
    reaches = dict()
    geom_error_count = 0
    if source == 'HAND':
        hand_tw = pd.read_csv(os.path.join(geom_dir, 'area.csv'))
        hand_el = pd.read_csv(os.path.join(geom_dir, 'el.csv'))
        hand_wp = pd.read_csv(os.path.join(geom_dir, 'p.csv'))
        hand_ar = pd.read_csv(os.path.join(geom_dir, 'vol.csv'))
    for r in list(meta_df.index):
        ch_n = meta_df.loc[r, 'n']
        fp_n = meta_df.loc[r, 'nCC']
        slope = meta_df.loc[r, 'slope']
        length = meta_df.loc[r, 'length']
        da = meta_df.loc[r, 'TotDASqKm']

        tw = meta_df.loc[r, 'TopWdth']
        bw = meta_df.loc[r, 'BtmWdth']
        z = 2 / (meta_df.loc[r, 'ChSlp'])
        tw_cc = meta_df.loc[r, 'TopWdthCC']
        bf = (tw - bw) / (z)

        if source == 'NWM':
            reaches[r] = WRFCompound(bw, z, bf, tw_cc, ch_n, fp_n, slope, length, max_stage=20*bf, stage_resolution=999)
        elif source == 'NWM_Regression':
            tw = 2.44 * (da ** 0.34)
            a_ch = 0.75 * (da ** 0.53)
            bf = (a_ch / tw) * 1.25
            bw = ((2 * a_ch) / bf) - tw
            z = (tw - bw) / bf
            tw_cc = 3 * tw
            reaches[r] = WRFCompound(bw, z, bf, tw_cc, ch_n, fp_n, slope, length, max_stage=20*bf, stage_resolution=999)
        elif source == 'HAND':
            try:
                tw = hand_tw[r].to_numpy() / length
                if np.all(tw == 0):
                    raise RuntimeError('All top-width values are zero')
                wp = hand_wp[r].to_numpy() / length
                ar = hand_ar[r].to_numpy() / length
                el = hand_el[r].to_numpy()
                n = np.ones(el.shape) * fp_n
                n[el < bf] = ch_n
                n[el >= bf] = fp_n
                reaches[r] = CustomReach(n, slope, length, el, tw, ar, wp)
            except Exception as e:
                # Error catching.  Default to NWM channel
                print(f'Error loading HAND data for reach {r}.  Defaulting to NWM channel')
                print(f'Error: {e}')
                geom_error_count += 1
                reaches[r] = WRFCompound(bw, z, bf, tw_cc, ch_n, fp_n, slope, length, max_stage=10*bf, stage_resolution=len(hand_tw))
        elif source == 'MUSKINGUM':
            x = meta_df.loc[r, 'MusX']
            k = meta_df.loc[r, 'MusK']
            reaches[r] = MuskingumReach(k, x)

    print(f'Error loading {geom_error_count} / {len(reaches)} reaches')
    return reaches

def execute_by_reach(meta_path):
    # Load data
    with open(meta_path, 'r') as f:
        paths = json.load(f)

    meta_df = pd.read_csv(paths['meta_path'])
    meta_df['comid'] = meta_df['comid'].astype(int).astype(str)
    meta_df['ds_comid'] = meta_df['ds_comid'].astype(int).astype(str)
    meta_df = meta_df.set_index("comid")

    upstream = pd.read_csv(os.path.join(paths['base_dir'], 'upstream_hydrographs.csv'))
    upstream = clean_ts(upstream)
    upstream = upstream.reindex(sorted(upstream.columns), axis=1)

    # Set up reaches
    reaches = load_geom(meta_df, source=paths['geometry_source'], geom_dir=paths['geometry_dir'])

    # export celerity curves
    stages = {r: reaches[r].geometry['stage'] for r in reaches}
    stage_df = pd.DataFrame().from_dict(stages, orient='columns')
    stage_df.to_csv(os.path.join(paths['out_dir'], "stage.csv"))
    celerities = {r: reaches[r].geometry['celerity'] for r in reaches}
    celerity_df = pd.DataFrame().from_dict(celerities, orient='columns')
    celerity_df.to_csv(os.path.join(paths['out_dir'], "celerity.csv"))
    discharge = {r: reaches[r].geometry['discharge'] for r in reaches}
    discharge_df = pd.DataFrame().from_dict(discharge, orient='columns')
    discharge_df.to_csv(os.path.join(paths['out_dir'], "discharge.csv"))

    # Run
    out_df = pd.DataFrame(index=upstream.index, columns=upstream.columns)
    stage_df = pd.DataFrame(index=upstream.index, columns=upstream.columns)
    dt = (upstream.index[1] - upstream.index[0]).seconds / 3600
    counter = 1
    for r in upstream.columns:
        if counter % 100 == 0:
            print(f"{counter} / {len(upstream.columns)}")
        try:
            reach = reaches[r]
            outflow = reach.route_hydrograph(upstream[r].to_numpy(), dt)
        except Exception as e:
            print(f"Error routing reach {r}: ({type(e).__name__}) {e}")
            outflow = np.zeros(len(upstream))
        out_df[r] = outflow
        stage_df[r] = np.interp(outflow, reach.geometry['discharge'], reach.geometry['stage'])

    os.makedirs(paths['out_dir'], exist_ok=True)
    out_df.to_csv(os.path.join(paths['out_dir'], "model_outflows.csv"))
    stage_df.to_csv(os.path.join(paths['out_dir'], "model_outflows_stage.csv"))

def execute(meta_path):
    # Load data
    with open(meta_path, 'r') as f:
        paths = json.load(f)

    meta_df = pd.read_csv(paths['meta_path'])
    meta_df['comid'] = meta_df['comid'].astype(int).astype(str)
    meta_df['ds_comid'] = meta_df['ds_comid'].astype(int).astype(str)
    meta_df = meta_df.set_index("comid")
    edge_dict = meta_df['ds_comid'].to_dict()

    lateral_df = pd.read_csv(paths['lateral_path'])
    lateral_df = clean_ts(lateral_df)
    lateral_df = lateral_df.reindex(sorted(lateral_df.columns), axis=1)

    head_df = pd.read_csv(paths['head_path'])
    head_df = clean_ts(head_df)
    head_df = head_df.reindex(sorted(head_df.columns), axis=1)

    lake_df = pd.read_csv(paths['lake_path'])
    lake_df = clean_ts(lake_df)
    lake_df = lake_df.reindex(sorted(lake_df.columns), axis=1)

    downstream = pd.read_csv(os.path.join(paths['base_dir'], 'downstream_hydrographs.csv'))
    downstream = clean_ts(downstream)
    downstream = downstream.reindex(sorted(downstream.columns), axis=1)

    # pre process lakes
    tmp_cols = list(lake_df.columns)
    lakes = [float(c) for c in tmp_cols]
    for l in lakes:
        # Find root of each lake
        lake_reaches = list(meta_df[meta_df['NHDWaterbodyComID'] == l].index)
        lake_root = lake_reaches[0]
        outflow = meta_df.loc[lake_root, 'ds_comid']
        while outflow in lake_reaches:
            lake_root = meta_df.loc[lake_root, 'ds_comid']
            outflow = meta_df.loc[lake_root, 'ds_comid']
        head_df[lake_root] = downstream[lake_root]

    # resample to 5 minute increments
    lateral_df = lateral_df.resample('5T').ffill()
    head_df = head_df.resample('5T').ffill()
    lake_df = lake_df.resample('5T').ffill()

    # Error checking
    missing_forcings = list(set(meta_df.index).difference(set(lateral_df.columns)))
    print(f'Missing forcing data for: {missing_forcings}')

    network = Network(edge_dict)

    # Set up reaches
    reaches = load_geom(meta_df, source=paths['geometry_source'], geom_dir=paths['geometry_dir'])

    # export celerity curves
    stages = {r: reaches[r].geometry['stage'] for r in reaches}
    stage_df = pd.DataFrame().from_dict(stages, orient='columns')
    stage_df.to_csv(os.path.join(paths['out_dir'], "stage.csv"))
    celerities = {r: reaches[r].geometry['celerity'] for r in reaches}
    celerity_df = pd.DataFrame().from_dict(celerities, orient='columns')
    celerity_df.to_csv(os.path.join(paths['out_dir'], "celerity.csv"))
    discharge = {r: reaches[r].geometry['discharge'] for r in reaches}
    discharge_df = pd.DataFrame().from_dict(discharge, orient='columns')
    discharge_df.to_csv(os.path.join(paths['out_dir'], "discharge.csv"))
                                 
    # Set up run
    network = Network(edge_dict)
    network.load_forcings(lateral_df)
    network.load_headwater_forcings(head_df)
    network.load_reaches(reaches)

    # Set up initial outflows
    for n in network.post_order:
        obs_out = downstream[n].to_numpy()[0]
        if np.isnan(obs_out):
            obs_out = None
        network.init_outflows[n] = obs_out
    network.load_initial_conditions(network.init_outflows)  # kind of unnecessary

    # Run model
    network.run_event(optimize_dx=paths['optimize_dx'], conserve_mass=paths['conserve_mass'], lat_addition=paths['lat_addition'])
    os.makedirs(paths['out_dir'], exist_ok=True)
    out_path = os.path.join(paths['out_dir'], "model_outflows.csv")
    network.out_df.to_csv(out_path)
    out_path = os.path.join(paths['out_dir'], "model_outflows_stage.csv")
    network.stage_df.to_csv(out_path)

    return network

def handler(path):
    with open(path, 'r') as f:
        dict = json.load(f)

    if dict['mode'] == 'reach':
        execute_by_reach(path)
    elif dict['mode'] == 'basin':
        execute(path)

if __name__ == '__main__':
    path = sys.argv[1]
    handler(path)