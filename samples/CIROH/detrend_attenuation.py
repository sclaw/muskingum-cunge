import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import pwlf
import os


def segmented(in_path, plot_live=False):
    working_dir = os.path.dirname(in_path)
    in_path = r"G:\floodplainsData\runs\3\muskingum-cunge\mc_out.csv"
    in_data = pd.read_csv(in_path)

    metadata_cols = ['ReachCode', 'DASqKm', 'slope', 'peak_loc_error', 'peak_val_error', 'dt_error']
    cols = [c for c in in_data.columns if c not in metadata_cols]
    wrk_df = in_data[cols]
    wrk_df['ReachCode'] = in_data['ReachCode']
    wrk_df['slope'] = np.log(in_data['slope'])
    # Set negative attenuation to minimum attenuation
    for c in cols:
        wrk_df[c] = np.log(wrk_df[c])
    wrk_df = wrk_df.dropna(axis=0)
    og_scatter = wrk_df.copy()
    
    for col in cols:
        # Model
        x0 = np.array([min(wrk_df['slope']), np.log(10 ** -5), np.log(0.0025), max(wrk_df['slope'])])
        my_pwlf = pwlf.PiecewiseLinFit(wrk_df['slope'], wrk_df[col])
        my_pwlf.fit_with_breaks(x0)

        s_space = np.linspace(min(wrk_df['slope']), max(wrk_df['slope']), num=1000)
        pred = my_pwlf.predict(s_space)
        wrk_df[col] = wrk_df[col] - my_pwlf.predict(wrk_df['slope'])

        # KDE
        resid_space = np.linspace(-5, 5, 1000)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(wrk_df[col].values[:, np.newaxis])
        density = kde.score_samples(resid_space[:, np.newaxis])

        y_log = True
        if not y_log:
            low_att = np.exp(low_att)
            med_att = np.exp(med_att)
            high_att = np.exp(high_att)
            mean = np.exp(mean)
            med_pred = np.exp(med_pred)
            high_pred = np.exp(high_pred)

        fig, axs = plt.subplots(ncols=2, figsize=(13.33, 7.5))
        axs[0].scatter(og_scatter['slope'], og_scatter[col], c='#B10DC9', s=3, alpha=0.2, label='MC Model Results')
        axs[0].plot(s_space, pred, ls='solid', c='k', alpha=0.7, label='Regression')
        axs[0].set(xlabel='Slope (log m/m)', ylabel=f'log of {col}')
        axs[0].legend()

        axs[1].fill_between(resid_space, np.exp(density), fc="#B10DC9", alpha=0.9, ec='k')
        axs[1].set(xlabel='Regression Residual (log cms)')

        fig.suptitle(col)
        if plot_live:
            plt.show()
        else:
            fig.savefig(os.path.join(working_dir, 'diagnostics', 'detrending', f'{col}.png'), dpi=300)
        plt.close()
    
    for c in metadata_cols:
        wrk_df[c] = in_data[c]
    wrk_df.to_csv(os.path.join(working_dir, 'detrended_data.csv'), index=False)

if __name__ == '__main__':
    path = r"G:\floodplainsData\runs\3\muskingum-cunge\mc_out.csv"
    segmented(path)