import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from muskingumcunge.reach import CustomReach
from scipy.ndimage import gaussian_filter1d
import time


### Static Data ###
T_TP_ORDINATES = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5])
Q_QP_ORDINATES = np.array([0, 0.03, 0.1, 0.19, 0.31, 0.47, 0.66, 0.82, 0.93, 0.99, 1, 0.99, 0.93, 0.86, 0.78, 0.68, 0.56, 0.46, 0.39, 0.33, 0.28, 0.207, 0.147, 0.107, 0.077, 0.055, 0.04, 0.029, 0.021, 0.015, 0.011, 0.005, 0])
PEAK_FLOW_REGRESSION = {
    'Q2': lambda da: 48.2 * ((da / 2.59) ** 0.869) / 35.3147,  # Cubic Meters per Second.  From Olson 2014
    'Q10': lambda da: 101 * ((da / 2.59) ** 0.847) / 35.3147,
    'Q50': lambda da: 164 * ((da / 2.59) ** 0.833) / 35.3147,
    'Q100': lambda da: 197 * ((da / 2.59) ** 0.827) / 35.3147}
DURATION_REGRESSION = {
    'Q2_Short': lambda da: 94.5 * (da ** 0.268) / 60,  # Hours.  From Lawson 2023
    'Q2_Medium': lambda da: 94.5 * (da ** 0.384) / 60,
    'Q2_Long': lambda da: 94.5 * (da ** 0.511) / 60,
    'Q10_Short': lambda da: 314 * (da ** 0.268) / 60,
    'Q10_Medium': lambda da: 314 * (da ** 0.384) / 60,
    'Q10_Long': lambda da: 314 * (da ** 0.511) / 60,
    'Q50_Short': lambda da: 534 * (da ** 0.268) / 60,
    'Q50_Medium': lambda da: 534 * (da ** 0.384) / 60,
    'Q50_Long': lambda da: 534 * (da ** 0.511) / 60,
    'Q100_Short': lambda da: 628 * (da ** 0.268) / 60,
    'Q100_Medium': lambda da: 628 * (da ** 0.384) / 60,
    'Q100_Long': lambda da: 628 * (da ** 0.511) / 60
}

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


def variance(q, t):
    n = np.sum(q)
    t_bar = (q * t) / n
    return np.sum(((t - t_bar) ** 2) * q) / (n - 1)


def execute(run_path, debug_plots=False):
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
    valid_columns = ['4300108000015','4300103001544', '4300103004224', '4300107001632', '4300108006151', '4300103001342', '4300102006820', '4300103003575', '4300108001804', '4300101001766', '4300103004201', '4300108006451', '4300108010065', '4300108009146', '4300102002584', '4300105002588', '4300103003747', '4300107003106']

    # Setup Hydrographs
    hydrographs = ['Q2_Short', 'Q2_Medium', 'Q2_Long', 'Q10_Short', 'Q10_Medium', 'Q10_Long', 'Q50_Short', 'Q50_Medium', 'Q50_Long', 'Q100_Short', 'Q100_Medium', 'Q100_Long']
    results_dict = dict()
    results_dict['ReachCode'] = list()
    results_dict['DASqKm'] = list()
    results_dict['slope'] = list()
    results_dict['peak_loc_error'] = list()
    results_dict['peak_val_error'] = list()
    results_dict['dt_error'] = list()
    for h in hydrographs:
        results_dict['_'.join([h, 'lag'])] = list()
        results_dict['_'.join([h, 'pct_attenuation'])] = list()
        results_dict['_'.join([h, 'raw_attenuation'])] = list()

    # Route
    counter = 1
    t_start = time.perf_counter()
    for reach in valid_columns:
        print(f'{counter} / {len(valid_columns)} | {round((len(valid_columns) - counter) * ((time.perf_counter() - t_start) / counter), 1)} seconds left')
        counter += 1
        
        # Subset data
        tmp_geom = {i: geometry[i][reach].to_numpy() for i in geometry}
        tmp_meta = reach_data.loc[reach]
        slope = tmp_meta['slope']
        length = tmp_meta['length']
        da = tmp_meta['TotDASqKm']

        # Convert 3D to 2D perspective
        tmp_geom['area'] = tmp_geom['area'] / length
        tmp_geom['volume'] = tmp_geom['volume'] / length
        tmp_geom['perimeter'] = tmp_geom['perimeter'] / length

        # Create reach
        mc_reach = CustomReach(0.035, slope, 1000, tmp_geom['elevation'], tmp_geom['area'], tmp_geom['volume'], tmp_geom['perimeter'])

        dqs = mc_reach.geometry['discharge'][1:] - mc_reach.geometry['discharge'][:-1]
        das = mc_reach.geometry['area'][1:] - mc_reach.geometry['area'][:-1]
        dq_da = dqs / das
        dq_da[0] = dq_da[1]
        dq_da = np.append(dq_da, dq_da[-1])

        dq_da[np.isnan(dq_da)] = 0.0001
        dq_da_2 = gaussian_filter1d(dq_da, 15)
        dq_da_2[dq_da_2 < 0.0001] = 0.0001
        dq_da_2[abs((dq_da - dq_da[0]) / dq_da[0]) < 0.1] = dq_da[0]
        mc_reach.geometry['celerity'] = dq_da_2

        # Route hydrographs
        results_dict['ReachCode'].append(reach)
        results_dict['DASqKm'].append(da)
        results_dict['slope'].append(slope)
        results_dict['peak_loc_error'].append(False)
        results_dict['peak_val_error'].append(False)
        results_dict['dt_error'].append(False)
        if debug_plots:
            fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
            axs[0, 1].plot(mc_reach.geometry['discharge'], mc_reach.geometry['celerity'], c='k')
            axs[0, 2].plot(mc_reach.geometry['top_width'], mc_reach.geometry['stage'], c='k')
            ax2 = axs[1, 2].twinx()
        for hydrograph in hydrographs:
            # Calculate hydrograph ordinates
            magnitude = hydrograph.split('_')[0]
            tmp_flows = Q_QP_ORDINATES * PEAK_FLOW_REGRESSION[magnitude](da)
            tmp_times = T_TP_ORDINATES * DURATION_REGRESSION[hydrograph](da)
            timesteps = np.arange(0, 3*DURATION_REGRESSION[hydrograph](da), run_meta['dt'])
            inflows = np.interp(timesteps, tmp_times, tmp_flows)

            # Route hydrograph
            outflows, errors = mc_reach.route_hydrograph_c(inflows, run_meta['dt'])
            
            # Log results
            raw_attenuation = inflows.max() - outflows.max()
            pct_attenuation = raw_attenuation / inflows.max()
            lag = (np.argmax(outflows) - np.argmax(inflows)) * run_meta['dt']
            results_dict['_'.join([hydrograph, 'lag'])].append(lag)
            results_dict['_'.join([hydrograph, 'pct_attenuation'])].append(pct_attenuation)
            results_dict['_'.join([hydrograph, 'raw_attenuation'])].append(raw_attenuation)
            results_dict['peak_loc_error'][-1] = results_dict['peak_loc_error'][-1] or errors[0]
            results_dict['peak_val_error'][-1] = results_dict['peak_val_error'][-1] or errors[1]
            results_dict['dt_error'][-1] = results_dict['dt_error'][-1] or errors[2]

            # Debug
            if debug_plots:
                if magnitude == 'Q2':
                    c = '#007ACC'
                elif magnitude == 'Q10':
                    c = '#D7263D'
                elif magnitude == 'Q50':
                    c = '#99C24D'
                elif magnitude == 'Q100':
                    c = '#6A0572'
                axs[0, 0].plot(timesteps, inflows, c=c)
                axs[1, 0].plot(timesteps, outflows, c=c)
                axs[1, 2].scatter(max(inflows), pct_attenuation, c='k', marker='x')
                ax2.scatter(max(inflows), raw_attenuation, fc='none', ec='gray')
                axs[1, 1].scatter(max(inflows), lag, c='k', marker='x')
                axs[0, 1].axvline(max(inflows), c='k', ls='dashed')
                axs[0, 2].axhline(np.interp(max(inflows), mc_reach.geometry['discharge'], mc_reach.geometry['stage']), c='k', ls='dashed')

        # Debug
        if debug_plots:            
            axs[0, 0].set(xlabel='Time (hours)', ylabel='Flow (cms)')
            axs[0, 0].text(0.95, 0.95, 'Inflow', transform=axs[0, 0].transAxes, horizontalalignment='right', verticalalignment='top')
            axs[1, 0].set(xlabel='Time (hours)', ylabel='Flow (cms)')
            axs[1, 0].text(0.95, 0.95, 'Outflow', transform=axs[1, 0].transAxes, horizontalalignment='right', verticalalignment='top')
            axs[0, 1].set(xlabel='Discharge (cms)', ylabel='Celerity (m/s)')
            axs[0, 2].set(xlabel='Width (m)', ylabel='Stage (m)')
            axs[1, 2].set(xlabel='Discharge (cms)', ylabel='Percent Attenuation (x)')
            ax2.set(ylabel='Raw Attenuation (o)')
            axs[1, 1].set(xlabel='Discharge (cms)', ylabel='Lag')
            fig.suptitle(f'{reach} | slope={slope} m/m | DA={da} sqkm')
            fig.tight_layout()
            plt.show()

    out_data = pd.DataFrame(results_dict)
    out_data = out_data.set_index('ReachCode')
    os.makedirs(os.path.dirname(run_meta['out_path']), exist_ok=True)
    out_data.to_csv(run_meta['out_path'])

if __name__ == '__main__':
    run_path = 'samples/CIROH/run_3.json'
    execute(run_path, debug_plots=True)
