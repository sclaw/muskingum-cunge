import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from muskingumcunge.reach import CustomReach
from scipy.ndimage import gaussian_filter1d
import time


def create_run_template():
    path_dict = {'elevation_path': 'path',
                 'area_path': 'path',
                 'volume_path': 'path',
                 'perimeter_path': 'path',
                 'reach_path': 'path'}
    run_dict = {'reaches_completed': list(),
                'run_date': None}
    combo_dict = {'data_paths': path_dict,
                  'run_meta': run_dict}
    out_path = os.path.join(os.path.dirname(__file__), 'run_template.json')
    out_file = open(out_path, "w")
    json.dump(combo_dict, out_file, indent=6)

def load_run(path):
    # Loads pandas dataframes and run metadata from a json
    in_file = open(path)
    in_dict = json.load(in_file)
    data_dict = {i[0].replace('_path', ''): pd.read_csv(i[1]) for i in in_dict['data_paths'].items()}
    reach_data = data_dict['reach']
    reach_data['ReachCode'] = reach_data['ReachCode'].astype(np.int64).astype(str)
    reach_data = reach_data.set_index('ReachCode')
    del data_dict['reach']
    run_meta = in_dict['run_meta']
    return data_dict, reach_data, run_meta


def add_bathymetry(geom, da, slope, shape='rectangle'):
    filter_arg = np.argmin(geom['elevation'] < 0.015)
    top_width = geom['area'][filter_arg]
    top_width = max(1, top_width)
    flowrate = (0.4962 * da) / 35.3147
    n = 0.035
    stage_space = np.linspace(0, 10, 1000)

    if shape == 'trapezoid':
        area = (0.8 * top_width) * stage_space  # Follum 2023 assumed Bw=0.6Tw
        perimeter = (0.6 * top_width) + (2 * (((stage_space ** 2) + ((0.2 * top_width) ** 2)) ** 0.5))
    elif shape == 'rectangle':
        area = (stage_space * top_width)
        perimeter = (top_width + (2 * stage_space))

    flowrate_space = (1 / n) * (stage_space * top_width) * (slope ** 0.5) * ((area / perimeter) ** (2 / 3))
    add_depth = np.interp(flowrate, flowrate_space, stage_space)
    add_volume = np.interp(flowrate, flowrate_space, area)
    add_perimeter = np.interp(flowrate, flowrate_space, perimeter)

    geom = {i: geom[i][filter_arg:] for i in geom}

    geom['area'] = np.insert(geom['area'], 0, perimeter[0])

    geom['elevation'] -= geom['elevation'][0]
    geom['elevation'] += add_depth
    geom['elevation'] = np.insert(geom['elevation'], 0, 0)

    geom['volume'] -= geom['volume'][0]
    geom['volume'] += add_volume
    geom['volume'] = np.insert(geom['volume'], 0, 0)

    geom['perimeter'] -= geom['perimeter'][0]
    geom['perimeter'] += add_perimeter
    geom['perimeter'] = np.insert(geom['perimeter'], 0, 0)

    return geom


def variance(q, t):
    n = np.sum(q)
    t_bar = (q * t) / n
    return np.sum(((t - t_bar) ** 2) * q) / (n - 1)


def execute(run_path):
    ### temp docstring.  Method to route several hydrographs through all reaches in a dataset and record relevant metrics
    # Load run info
    geometry, reach_data, run_meta = load_run(run_path)

    # Clean input data
    valid_columns = set(reach_data.index)
    for col in geometry:
        dataset = geometry[col]
        tmp_cols = dataset.columns[(dataset != 0).any(axis=0)]
        valid_columns = valid_columns.intersection(tmp_cols)
    valid_columns = sorted(valid_columns)

    # Temporary debugging override
    # valid_columns = ['4300103002217']
    # valid_columns = ['4300102001933']
    # valid_columns = ['4300102003621']

    # Setup Hydrographs
    t_tp_ordinates = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5])
    q_qp_ordinates = np.array([0, 0.03, 0.1, 0.19, 0.31, 0.47, 0.66, 0.82, 0.93, 0.99, 1, 0.99, 0.93, 0.86, 0.78, 0.68, 0.56, 0.46, 0.39, 0.33, 0.28, 0.207, 0.147, 0.107, 0.077, 0.055, 0.04, 0.029, 0.021, 0.015, 0.011, 0.005, 0])
    q2 = lambda da: 48.2 * (da ** 0.869) / 35.3147
    q10 = lambda da: 101 * (da ** 0.847) / 35.3147
    q50 = lambda da: 164 * (da ** 0.833) / 35.3147
    q100 = lambda da: 197 * (da ** 0.827) / 35.3147
    q_funcs = [q2, q10, q50, q100]
    q_labels = ['q2', 'q10', 'q50', 'q100']
    durations = [5, 10, 15]
    d_labels = ['short', 'med', 'long']
    hydrographs = list()
    for func, q_label in zip(q_funcs, q_labels):
            for length, d_label in zip(durations, d_labels):
                label = '_'.join([q_label, d_label])
                attenuation_label = '_'.join([label, 'att'])
                lag_label = '_'.join([label, 'lag'])
                reach_data[attenuation_label] = np.nan
                reach_data[lag_label] = np.nan
                h_dict = {'attenuation_label': attenuation_label,
                          'lag_label': lag_label,
                          'function': func,
                          'length': length}
                hydrographs.append(h_dict)
    dt = 0.03

    # Route
    counter = 1
    t_start = time.perf_counter()
    for reach in valid_columns:
        print(f'{counter} / {len(valid_columns)} | {round((len(valid_columns) - counter) * ((time.perf_counter() - t_start) / counter), 1)} seconds left')
        counter += 1
        # Subset data
        tmp_geom = {i: geometry[i][reach].to_numpy() for i in geometry}
        tmp_meta = reach_data.loc[reach]
        # slope = tmp_meta['slope(m/m)']
        # length = tmp_meta['length(m)']
        # da = tmp_meta['DA(sqkm)']
        slope = tmp_meta['slope']
        length = tmp_meta['length']
        da = tmp_meta['TotDASqKm']
        tmp_geom['elevation'] = np.linspace(0, 5 * (0.26 * (da ** 0.287)), 999)  # Temporary until we fix utilities.py

        # Convert 3D to 2D perspective
        tmp_geom['area'] = tmp_geom['area'] / length
        tmp_geom['volume'] = tmp_geom['volume'] / length
        tmp_geom['perimeter'] = tmp_geom['perimeter'] / length

        # Add bathymetry
        cache_el = tmp_geom['elevation'].copy()
        cache_width = tmp_geom['area'].copy()

        tmp_geom = add_bathymetry(tmp_geom, da, slope)

        # fig, ax = plt.subplots()
        # ax.plot(cache_width, cache_el, label='og')
        # # ax.plot(tmp_geom['area'], tmp_geom['elevation'] - (tmp_geom['elevation'][-1] - cache_el[-1]), label='added', ls='dashed')
        # ax.plot(tmp_geom['area'], tmp_geom['elevation'], label='added', ls='dashed')
        # plt.legend()
        # plt.show()

        # Create reach
        mc_reach = CustomReach(0.035, slope, 1000, tmp_geom['elevation'], tmp_geom['area'], tmp_geom['volume'], tmp_geom['perimeter'])

        dqs = mc_reach.geometry['discharge'][1:] - mc_reach.geometry['discharge'][:-1]
        das = mc_reach.geometry['area'][1:] - mc_reach.geometry['area'][:-1]
        dq_da = dqs / das
        dq_da[0] = dq_da[1]
        dq_da = np.append(dq_da, dq_da[-1])
        # fig, ax = plt.subplots()
        # ax.plot(mc_reach.geometry['stage'], dq_da, label='raw', alpha=0.7)

        kernel = 50
        dq_da[np.isnan(dq_da)] = 0.0001
        # dq_da_cum = np.cumsum(dq_da)
        # dq_da_cum = dq_da_cum[kernel:] - dq_da_cum[:-kernel]
        # dq_da_cum /= kernel
        # dq_da_2 = dq_da
        # dq_da_2[int(kernel / 2):-int(kernel / 2)] = dq_da_cum
        # dq_da_2[:int(kernel / 2)] = np.linspace(dq_da_2[0], dq_da_2[int(kernel / 2)], int(kernel / 2))
        # dq_da_2[-int(kernel / 2):] = dq_da_2[-int(kernel / 2):].mean()
        dq_da_2 = gaussian_filter1d(dq_da, 15)
        dq_da_2[dq_da_2 < 0.0001] = 0.0001
        dq_da_2[abs((dq_da - dq_da[0]) / dq_da[0]) < 0.1] = dq_da[0]
        # ax.plot(mc_reach.geometry['stage'], dq_da_2, label='smooth')
        mc_reach.geometry['celerity'] = dq_da_2

        # kernel = 100
        # dqs = mc_reach.geometry['discharge'][1::kernel] - mc_reach.geometry['discharge'][:-1:kernel]
        # das = mc_reach.geometry['area'][1::kernel] - mc_reach.geometry['area'][:-1:kernel]
        # dq_da = dqs / das
        # dq_da[0] = dq_da[1]
        # ax.plot(mc_reach.geometry['stage'][::kernel], dq_da, label='smooth')


        # plt.legend()
        # plt.show()

        # Route hydrographs
        run_sum = 0
        for h_dict in hydrographs:
            tmp_flows = q_qp_ordinates * h_dict['function'](da)
            tmp_times = t_tp_ordinates * h_dict['length']
            timesteps = np.arange(0, 48, dt)
            inflows = np.interp(timesteps, tmp_times, tmp_flows)

            outflows = mc_reach.route_hydrograph_c(inflows, dt)            

            pct_attenuation = (inflows.max() - outflows.max()) / inflows.max()
            lag = (np.argmax(outflows) - np.argmax(inflows)) * dt
            reach_data.loc[reach, h_dict['attenuation_label']] = pct_attenuation
            reach_data.loc[reach, h_dict['lag_label']] = lag

            var1 = variance(outflows, timesteps)
            var2 = variance(inflows, timesteps)
            print(f"{h_dict['attenuation_label'].ljust(20)} {round(var1 / var2, 5)}")
            fig, ax = plt.subplots()
            ax.plot(timesteps, inflows, label='inflows')
            ax.plot(timesteps, outflows, label='outflows')
            plt.title(h_dict['attenuation_label'])
            plt.legend()
            plt.show()

            run_sum += pct_attenuation
        print((run_sum * 100) / 12)

    reach_data = reach_data.loc[valid_columns]
    reach_data.to_csv(run_meta['out_path'])

if __name__ == '__main__':
    run_path = 'muskingum-cunge/samples/CIROH/run_2.json'
    execute(run_path)
