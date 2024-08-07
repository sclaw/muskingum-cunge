import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from muskingumcunge.reach import CustomReach
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import time
import sys


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
BKF_STAGE_REGRESSION = lambda da: (0.26 * (da ** 0.287))

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


def execute(meta_path, debug_plots=False):
    ### temp docstring.  Method to route several hydrographs through all reaches in a dataset and record relevant metrics
    # Load run info
    with open(meta_path, 'r') as f:
        run_dict = json.loads(f.read())
    geometry = {i: pd.read_csv(os.path.join(run_dict['geometry_directory'], f'{i}.csv')) for i in run_dict['fields_of_interest']}
    reach_data = pd.read_csv(run_dict['reach_meta_path'])
    reach_data[run_dict['id_field']] = reach_data[run_dict['id_field']].astype(np.int64).astype(str)
    reach_data = reach_data.set_index(run_dict['id_field'])
    mannings_n = 0.095

    # Setup Hydrographs
    hydrographs = ['Q2_Short', 'Q2_Medium', 'Q2_Long', 'Q10_Short', 'Q10_Medium', 'Q10_Long', 'Q50_Short', 'Q50_Medium', 'Q50_Long', 'Q100_Short', 'Q100_Medium', 'Q100_Long']
    results_dict = dict()
    results_dict[run_dict['id_field']] = list()
    results_dict['DASqKm'] = list()
    results_dict['slope'] = list()
    for h in hydrographs:
        results_dict['_'.join([h, 'diffusion_number'])] = list()
        results_dict['_'.join([h, 'pct_attenuation'])] = list()
        results_dict['_'.join([h, 'pct_attenuation_per_km'])] = list()
        results_dict['_'.join([h, 'cms_attenuation'])] = list()
        results_dict['_'.join([h, 'cms_attenuation_per_km'])] = list()
        results_dict['_'.join([h, 'skewness'])] = list()
        results_dict['_'.join([h, 'mass_conserve'])] = list()
        results_dict['_'.join([h, 'dx'])] = list()
        results_dict['_'.join([h, 'subreaches'])] = list()

    # Route
    counter = 1
    t_start = time.perf_counter()
    reaches = reach_data.index.to_list()
    for reach in reaches:
        print(f'{counter} / {len(reaches)} | {round((len(reaches) - counter) * ((time.perf_counter() - t_start) / counter), 1)} seconds left | {(len(reaches) - counter)} left | | {((time.perf_counter() - t_start) / counter)} rate')
        counter += 1

        if debug_plots:
            fig, axs = plt.subplots(ncols=5, nrows=4, figsize=(16, 10))
        
        # Subset data
        tmp_geom = {i: geometry[i][reach].to_numpy() for i in geometry}
        tmp_meta = reach_data.loc[reach]
        slope = tmp_meta['slope']
        length = tmp_meta['length']
        da = tmp_meta['TotDASqKm']

        # Convert 3D to 2D perspective
        tmp_geom['area'] = tmp_geom['area'] / length
        tmp_geom['vol'] = tmp_geom['vol'] / length
        tmp_geom['p'] = tmp_geom['p'] / length

        # Handle null geometries
        if np.all(tmp_geom['area'] < 1):
            print('NULL GEOMETRY')
            results_dict[run_dict['id_field']].append(reach)
            results_dict['DASqKm'].append(da)
            results_dict['slope'].append(slope)
            for hydrograph in hydrographs:
                results_dict['_'.join([hydrograph, 'diffusion_number'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'pct_attenuation'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'pct_attenuation_per_km'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'cms_attenuation'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'cms_attenuation_per_km'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'skewness'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'mass_conserve'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'dx'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'subreaches'])].append(np.nan)
            plt.close()
            continue

        # Create reach
        mc_reach = CustomReach(mannings_n, slope, 1000, tmp_geom['el'], tmp_geom['area'], tmp_geom['vol'], tmp_geom['p'])

        # Route hydrographs
        results_dict[run_dict['id_field']].append(reach)
        results_dict['DASqKm'].append(da)
        results_dict['slope'].append(slope)
        if debug_plots:
            
            for row, reg in zip(axs, PEAK_FLOW_REGRESSION):
                row[0].set_xlabel('Time (hours)')
                row[1].set_xlabel('Celerity (m/s)')
                row[2].set_xlabel('Top Width (m)')

                max_q = 1.1 * PEAK_FLOW_REGRESSION[reg](da)
                for col in range(3):
                    row[col].set_ylim(0, max_q)
                    row[col].set_ylabel('Discharge (CMS)')

                row[1].plot(mc_reach.geometry['celerity'], mc_reach.geometry['discharge'], c='k')
                row[2].plot(mc_reach.geometry['top_width'], mc_reach.geometry['discharge'], c='k')

                if mc_reach.geometry['discharge'].max() > max_q:
                    peak_loc = np.argmax(mc_reach.geometry['discharge'] > max_q)
                    peak_loc = max([peak_loc, 1])
                    max_c = mc_reach.geometry['celerity'][:peak_loc].max() 
                    max_w = mc_reach.geometry['top_width'][:peak_loc].max()
                else:
                    max_c = 1
                    max_w = 1
                row[1].set_xlim(0, max_c)
                row[2].set_xlim(0, max_w)

                q2el = lambda q: np.interp(q, mc_reach.geometry['discharge'], mc_reach.geometry['stage'])
                el2q = lambda el: np.interp(el, mc_reach.geometry['stage'], mc_reach.geometry['discharge'])
                q2normel = lambda q: np.interp(q, mc_reach.geometry['discharge'], mc_reach.geometry['stage']) / BKF_STAGE_REGRESSION(da)
                normel2q = lambda el: np.interp(el * BKF_STAGE_REGRESSION(da), mc_reach.geometry['stage'], mc_reach.geometry['discharge'])

                celerity_secax = row[1].secondary_yaxis('right', functions=(q2el, el2q))
                section_secax = row[2].secondary_yaxis('right', functions=(q2normel, normel2q))
                celerity_secax.set_ylabel('Stage (m)')
                section_secax.set_ylabel('Stage / BKF Depth')

            att_gs = axs[0, 3].get_gridspec()
            for ax in axs[:, 3]:
                ax.remove()
            att_ax = fig.add_subplot(att_gs[:, 3])
            att_ax.set_xlabel('Percent Attenuation')
            att_ax.set_xlim(0, 50)

            length_gs = axs[0, 4].get_gridspec()
            for ax in axs[:, 4]:
                ax.remove()
            length_ax = fig.add_subplot(length_gs[:, 4])
            length_ax.set_xlabel('Length for one Time to Rise (m)')     
        
        for hydrograph in hydrographs:
            # Calculate hydrograph ordinates
            magnitude = hydrograph.split('_')[0]
            peak = PEAK_FLOW_REGRESSION[magnitude](da)
            if peak > mc_reach.geometry['discharge'].max():
                results_dict['_'.join([hydrograph, 'diffusion_number'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'pct_attenuation'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'pct_attenuation_per_km'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'cms_attenuation'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'cms_attenuation_per_km'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'skewness'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'mass_conserve'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'dx'])].append(np.nan)
                results_dict['_'.join([hydrograph, 'subreaches'])].append(np.nan)
                continue
            tmp_flows = Q_QP_ORDINATES * peak
            tmp_times = T_TP_ORDINATES * DURATION_REGRESSION[hydrograph](da)
            dt = (tmp_times[10] / 20)  # From USACE guidance.  dt may be t_rise / 20
            timesteps = np.arange(0, 5*DURATION_REGRESSION[hydrograph](da), dt)
            inflows = np.interp(timesteps, tmp_times, tmp_flows)

            # Prepare model run
            # Ponce method
            qref = 0.5 * peak
            cref = np.interp(qref, mc_reach.geometry['discharge'], mc_reach.geometry['celerity'])
            twref = np.interp(qref, mc_reach.geometry['discharge'], mc_reach.geometry['top_width'])
            dist_to_rise = (tmp_times[10] * 60 * 60) * cref
            dxc = dt * 60 * 60 * cref  # Courant length
            dxd = (qref / twref) / (mc_reach.slope * cref)  # characteristic reach length
            dxmax = 0.5 * (dxc + dxd)
            peak_loc = np.argmax(mc_reach.geometry['discharge'] > peak)
            peak_loc = max([peak_loc, 1])
            cmax = mc_reach.geometry['celerity'][:peak_loc].max()  # I think the fact that this is the max should cover us for making all stable
            dxmin = cmax * (dt * 60 * 60)
            dx = max([dxmin, dxmax])
            subreaches = int(np.ceil(dist_to_rise / dx))
            dx = dist_to_rise / subreaches
            mc_reach.reach_length = dx

            # Route hydrograph
            outflows = inflows.copy()
            for iter in range(subreaches):
                try:
                    outflows = mc_reach.route_hydrograph(outflows, (dt * 60 * 60), lateral=None, initial_outflow=None, short_ts=False, solver='fread-c')
                except AssertionError:
                    outflows = np.repeat(np.nan, outflows.shape[0])
                    break
            
            # Log results
            raw_attenuation = inflows.max() - outflows.max()
            attenuation_per_km = raw_attenuation / (dx * subreaches / 1000)
            pct_attenuation = raw_attenuation / inflows.max()
            pct_attenuation_km = (1 - ((1 - pct_attenuation) ** (1000 / (dx * subreaches))))
            tmp_diff_number = ((9 * np.pi) / 50) * (((mannings_n ** (6 / 5)) * (max(inflows) ** (1 / 5))) / (((slope * 100) ** (8 / 5)) * (dt * 20)))
            peak_loc = np.argmax(outflows)
            dqs = outflows[1:] - outflows[:-1]
            rising_dqs = np.abs(dqs[:peak_loc])
            falling_dqs = np.abs(dqs[peak_loc:])
            skewness = 1 - (falling_dqs.mean() / rising_dqs.mean())
            # volume conservation a la todini 2007
            conserved = inflows.sum() / outflows.sum()

            results_dict['_'.join([hydrograph, 'diffusion_number'])].append(tmp_diff_number)
            results_dict['_'.join([hydrograph, 'pct_attenuation'])].append(pct_attenuation)
            results_dict['_'.join([hydrograph, 'pct_attenuation_per_km'])].append(pct_attenuation_km)
            results_dict['_'.join([hydrograph, 'cms_attenuation'])].append(raw_attenuation)
            results_dict['_'.join([hydrograph, 'cms_attenuation_per_km'])].append(attenuation_per_km)
            results_dict['_'.join([hydrograph, 'skewness'])].append(skewness)
            results_dict['_'.join([hydrograph, 'mass_conserve'])].append(conserved)
            results_dict['_'.join([hydrograph, 'dx'])].append(dx)
            results_dict['_'.join([hydrograph, 'subreaches'])].append(subreaches)
            

            # Debug
            if debug_plots:
                row_dict = {'Q2': 0, 'Q10': 1, 'Q50': 2, 'Q100': 3}
                row = row_dict[magnitude]
                axs[row, 0].plot(timesteps, inflows, c='darkgray')
                axs[row, 0].plot(timesteps, outflows, c='k')

        # Debug
        if debug_plots:
            width = 0.25
            position_dict = {'Q2': 3, 'Q10': 2, 'Q50': 1, 'Q100': 0}
            offset_dict = {'Short': 0, 'Medium': 1, 'Long': 2}
            loc_list = list()
            attenuations = list()
            att_colors = list()
            for hydrograph in hydrographs:
                loc = position_dict[hydrograph.split('_')[0]] + (offset_dict[hydrograph.split('_')[1]] * width)
                loc_list.append(loc)
                att = results_dict['_'.join([hydrograph, 'pct_attenuation'])][-1] * 100
                attenuations.append(abs(att))
                if att < 0:
                    att_colors.append('k')
                else:
                    att_colors.append('darkorange')
                
            att_ax.barh(loc_list, attenuations, width, label=hydrographs, color=att_colors)
            att_ax.set_yticks(loc_list)
            att_ax.set_yticklabels(hydrographs)
            length_ax.set_yticks(loc_list)
            length_ax.set_yticklabels(hydrographs)

            fig.suptitle(f'{reach} | slope={slope} m/m | DA={da} sqkm | iterations={iter}')
            fig.tight_layout()
            fig.savefig(os.path.join(run_dict['muskingum_diagnostics'], f'{reach}.png'), dpi=100)
            plt.close()

    out_data = pd.DataFrame(results_dict)
    out_data = out_data.set_index(run_dict['id_field'])
    os.makedirs(os.path.dirname(run_dict['muskingum_path']), exist_ok=True)
    out_data.to_csv(run_dict['muskingum_path'])

if __name__ == '__main__':
    run_path = sys.argv[1]
    execute(run_path, debug_plots=False)
