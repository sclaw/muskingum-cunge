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
    valid_columns = ['4300101000139', '4300101000318', '4300101000385', '4300101000389', '4300101000425', '4300101000515', '4300101000545', '4300101000585', '4300101000764', '4300101000878', '4300101000922', '4300101000988', '4300101001076', '4300101001234', '4300101001550', '4300101001569', '4300101001692', '4300101001903', '4300101001905', '4300101001993', '4300101001997', '4300101002410', '4300102000152', '4300102000162', '4300102000225', '4300102000257', '4300102000267', '4300102000306', '4300102000326', '4300102000383', '4300102000427', '4300102000506', '4300102000514', '4300102000702', '4300102000707', '4300102000731', '4300102001022', '4300102001079', '4300102001144', '4300102001313', '4300102001347', '4300102001415', '4300102001462', '4300102001496', '4300102001873', '4300102001903', '4300102001931', '4300102001933', '4300102002052', '4300102002287', '4300102002819', '4300102002822', '4300102003036', '4300102003067', '4300102003080', '4300102003165', '4300102003203', '4300102003212', '4300102003275', '4300102003340', '4300102003543', '4300102003673', '4300102003847', '4300102003934', '4300102003970', '4300102004158', '4300102004542', '4300102004607', '4300102004660', '4300102004716', '4300102004753', '4300102004792', '4300102004925', '4300102004999', '4300102005157', '4300102005217', '4300102005264', '4300102005476', '4300102005511', '4300102005627', '4300102005729', '4300102005738', '4300102006175', '4300102006333', '4300102006373', '4300102006918', '4300102007370', '4300102007396', '4300102007522', '4300102007625', '4300102007687', '4300103000300', '4300103000363', '4300103000600', '4300103000616', '4300103000680', '4300103001266', '4300103001278', '4300103001315', '4300103001333', '4300103001642', '4300103001648', '4300103001868', '4300103002303', '4300103002540', '4300103002563', '4300103002633', '4300103002695', '4300103002755', '4300103002856', '4300103002890', '4300103002899', '4300103002923', '4300103003334', '4300103003532', '4300103003711', '4300103003922', '4300103004087', '4300103004319', '4300103004404', '4300103004590', '4300103004709', '4300103004723', '4300103004801', '4300103005065', '4300103005067', '4300103005178', '4300105000056', '4300105000114', '4300105000175', '4300105000200', '4300105000237', '4300105000326', '4300105000337', '4300105000570', '4300105000614', '4300105000634', '4300105000801', '4300105000831', '4300105000912', '4300105000924', '4300105000947', '4300105000983', '4300105000994', '4300105001009', '4300105001024', '4300105001035', '4300105001268', '4300105001279', '4300105001297', '4300105001305', '4300105001343', '4300105001406', '4300105001500', '4300105001617', '4300105001727', '4300105001792', '4300105001797', '4300105001938', '4300105002021', '4300105002027', '4300105002077', '4300105002203', '4300105002276', '4300105002615', '4300105002637', '4300105002716', '4300105002791', '4300105002985', '4300105002987', '4300105003015', '4300105003041', '4300105003125', '4300105003243', '4300105003249', '4300105003333', '4300105003356', '4300105003381', '4300105003435', '4300105003523', '4300105003918', '4300105003923', '4300105004101', '4300105004174', '4300105004237', '4300105004269', '4300105004275', '4300105004294', '4300105004342', '4300105004429', '4300105004431', '4300105004602', '4300105004699', '4300105004745', '4300105004770', '4300105004780', '4300105004791', '4300105004840', '4300105004944', '4300105004968', '4300105005003', '4300105005012', '4300105005036', '4300105005037', '4300105005059', '4300105005073', '4300105005317', '4300105005326', '4300105005402', '4300105005521', '4300105005614', '4300105005695', '4300105005720', '4300107000229', '4300107000327', '4300107000372', '4300107000787', '4300107000802', '4300107000874', '4300107001001', '4300107001116', '4300107001520', '4300107001562', '4300107001566', '4300107001747', '4300107001812', '4300107002032', '4300107002101', '4300107002237', '4300107002259', '4300107002370', '4300107002393', '4300107002443', '4300107002467', '4300107002617', '4300107002694', '4300107002830', '4300107002993', '4300107002999', '4300108000033', '4300108000264', '4300108000267', '4300108000342', '4300108000397', '4300108000426', '4300108000461', '4300108000706', '4300108000741', '4300108000742', '4300108000824', '4300108000843', '4300108001016', '4300108001288', '4300108001316', '4300108001335', '4300108001368', '4300108001491', '4300108001792', '4300108001959', '4300108002063', '4300108002181', '4300108002302', '4300108002464', '4300108002837', '4300108003229', '4300108003579', '4300108003765', '4300108003817', '4300108003850', '4300108004334', '4300108004648', '4300108005152', '4300108005240', '4300108005325', '4300108005675', '4300108005905', '4300108005918', '4300108005982', '4300108005989', '4300108006171', '4300108006197', '4300108006393', '4300108007023', '4300108008015', '4300108008295', '4300108008308', '4300108008463', '4300108008480', '4300108008628', '4300108008798', '4300108009369', '4300108009494', '4300108009536', '4300108009573', '4300108010639', '4300108010740', '4300108011088', '4300108013124', '4300108013130']
    
    
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
            fig.savefig(r"/netfiles/ciroh/floodplainsData/runs/3/muskingum-cunge/diagnostics/{}_lowdt.png".format(reach), dpi=300)
            plt.show()

    out_data = pd.DataFrame(results_dict)
    out_data = out_data.set_index('ReachCode')
    os.makedirs(os.path.dirname(run_meta['out_path']), exist_ok=True)
    # out_data.to_csv(run_meta['out_path'])

if __name__ == '__main__':
    run_path = 'samples/CIROH/run_3.json'
    execute(run_path, debug_plots=True)
