import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import pwlf


def v1():
    in_path = r"G:\floodplainsData\runs\3\muskingum-cunge\mc_out.csv"
    in_data = pd.read_csv(in_path)

    hydrographs = ['Q2_Short', 'Q2_Medium', 'Q2_Long', 'Q10_Short', 'Q10_Medium', 'Q10_Long', 'Q50_Short', 'Q50_Medium', 'Q50_Long', 'Q100_Short', 'Q100_Medium', 'Q100_Long']

    low_df = in_data[in_data['slope'] == (10 ** -5)].copy()
    med_df = in_data[np.logical_and((in_data['slope'] <= 0.0025), (in_data['slope'] != (10 ** -5)))].copy()
    high_df = in_data[in_data['slope'] > 0.0025].copy()
    
    for h in hydrographs:
        slopes = in_data['slope'].to_numpy()
        attenuation_type = 'raw_attenuation'
        col_label = '_'.join([h, attenuation_type])
        attenuations = in_data[col_label].to_numpy()
        
        null_mask = (attenuations > 0)
        remove_null = False
        if remove_null:
            attenuations = attenuations[null_mask]
            attenuations = np.log(attenuations)
            slopes = slopes[null_mask]
        else:
            in_data[col_label][in_data[col_label] <= 0] = in_data[col_label][in_data[col_label] > 0].min()

        low_mask = (slopes == (10 ** -5))
        med_mask = np.logical_and((slopes <= 0.0025), ~low_mask)
        high_mask = (slopes > 0.0025)
        slopes = np.log(slopes)

        low_slopes = slopes[low_mask]
        med_slopes = slopes[med_mask]
        high_slopes = slopes[high_mask]
        # attenuations[attenuations <= 0] = attenuations[attenuations > 0].min()

        low_att = attenuations[low_mask]
        med_att = attenuations[med_mask]
        high_att = attenuations[high_mask]

        # Fit models and get residuals
        
        resid_slopes = in_data['slope'].to_numpy()
        resid_att = in_data[col_label].to_numpy()
        low_resid_slopes = resid_slopes[resid_slopes == (10 ** -5)]
        low_resid_att = resid_att[resid_slopes == (10 ** -5)]
        med_resid_slopes = resid_slopes[np.logical_and((resid_slopes <= 0.0025), ~(resid_slopes == (10 ** -5)))]
        med_resid_att = resid_att[np.logical_and((resid_slopes <= 0.0025), ~(resid_slopes == (10 ** -5)))]
        high_resid_slopes = resid_slopes[resid_slopes > 0.0025]
        high_resid_att = resid_att[resid_slopes > 0.0025]

        # Model 1
        mean = low_att.mean()
        low_df[col_label] = low_df[col_label] - np.exp(mean)
        low_resid = np.log(low_resid_att - np.exp(mean))

        # Model 2
        med_reg = np.polyfit(med_slopes, med_att, 1)
        med_s_space = np.linspace(np.log(10 ** -5), np.log(0.0025), 500)
        med_pred = med_reg[1] + (med_s_space * med_reg[0])
        low_df[col_label] = low_df[col_label] - (med_reg[1] + (low_df['slope'] * med_reg[0]))

        # Model 3
        high_reg = np.polyfit(high_slopes, high_att, 1)
        high_s_space = np.linspace(np.log(0.0025), max(slopes), 500)
        high_pred = high_reg[1] + (high_s_space * high_reg[0])
        low_df[col_label] = low_df[col_label] - (med_reg[1] + (low_df['slope'] * med_reg[0]))
        high_resid = high_att - (high_reg[1] + (high_slopes * high_reg[0]))

        # KDE
        resid_space = np.linspace(-5, 5, 1000)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(low_resid[:, np.newaxis])
        low_density = kde.score_samples(resid_space[:, np.newaxis])
        low_density = np.exp(low_density) * (len(low_resid) / sum([len(low_resid), len(med_resid), len(high_resid)]))
        kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(med_resid[:, np.newaxis])
        med_density = kde.score_samples(resid_space[:, np.newaxis])
        med_density = np.exp(med_density) * (len(med_density) / sum([len(low_resid), len(med_resid), len(high_resid)]))
        med_density = med_density + low_density
        kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(high_resid[:, np.newaxis])
        high_density = kde.score_samples(resid_space[:, np.newaxis])
        high_density = np.exp(high_density) * (len(high_density) / sum([len(low_resid), len(med_resid), len(high_resid)]))
        high_density = med_density + high_density

        y_log = True
        if not y_log:
            low_att = np.exp(low_att)
            med_att = np.exp(med_att)
            high_att = np.exp(high_att)
            mean = np.exp(mean)
            med_pred = np.exp(med_pred)
            high_pred = np.exp(high_pred)

        fig, axs = plt.subplots(ncols=2, figsize=(11, 8.5))
        axs[0].scatter(low_slopes, low_att, s=3, alpha=0.2, c='#2ECC40', label='Low Slope Series')
        axs[0].scatter(med_slopes, med_att, s=3, alpha=0.2, c='#001F3F', label='Medium Slope Series')
        axs[0].scatter(high_slopes, high_att, s=3, alpha=0.2, c='#FF4136', label='High Slope Series')
        axs[0].plot([min(slopes), np.log(10 ** -5)], [mean, mean], ls='solid', c='k', alpha=0.7, label='Low Regression')
        axs[0].plot(med_s_space, med_pred, ls='dashed', c='k', alpha=0.7, label='Medium Regression')
        axs[0].plot(high_s_space, high_pred, ls='dotted', c='k', alpha=0.7, label='High Regression')
        axs[0].set(xlabel='Slope (log m/m)', ylabel='Raw Attenuation (log cms)')
        axs[0].legend()

        axs[1].fill_between(resid_space, low_density, fc="#2ECC40", alpha=0.9, ec='k')
        axs[1].fill_between(resid_space, med_density, low_density, fc="#001F3F", alpha=0.9, ec='k')
        axs[1].fill_between(resid_space, high_density, med_density, fc="#FF4136", alpha=0.9, ec='k')
        # axs[1].hist(np.concatenate([high_resid, med_resid, low_resid]), bins=50)
        axs[1].set(xlabel='Regression Residual (log cms)')

        fig.suptitle(h)
        plt.show()


def v2():
    in_path = r"G:\floodplainsData\runs\3\muskingum-cunge\mc_out.csv"
    in_data = pd.read_csv(in_path)

    hydrographs = ['Q2_Short', 'Q2_Medium', 'Q2_Long', 'Q10_Short', 'Q10_Medium', 'Q10_Long', 'Q50_Short', 'Q50_Medium', 'Q50_Long', 'Q100_Short', 'Q100_Medium', 'Q100_Long']

    for h in hydrographs:
        slopes = in_data['slope'].to_numpy()
        attenuation_type = 'raw_attenuation'
        col_label = '_'.join([h, attenuation_type])
        attenuations = in_data[col_label].to_numpy()
        null_mask = (attenuations > 0)
        attenuations = attenuations[null_mask]
        attenuations = np.log(attenuations)
        slopes = slopes[null_mask]
        slopes = np.log(slopes)

        # Model
        x0 = np.array([min(slopes), np.log(10 ** -5), np.log(0.0025), max(slopes)])
        my_pwlf = pwlf.PiecewiseLinFit(slopes, attenuations)
        my_pwlf.fit_with_breaks(x0)

        s_space = np.linspace(min(slopes), max(slopes), num=10000)
        pred = my_pwlf.predict(s_space)
        residuals = attenuations - my_pwlf.predict(slopes)

        fig, axs = plt.subplots(ncols=2, figsize=(11, 8.5))
        axs[0].scatter(slopes, attenuations, s=3, alpha=0.2, c='#2ECC40', label='Modeled Attenuation')
        axs[0].axvline(np.log(1.1 * (10 ** -5)), c='k', alpha=0.2)
        axs[0].axvline(np.log(0.0025), c='k', alpha=0.2)
        axs[0].plot(s_space, pred, ls='dashed', c='k', alpha=0.7, label='Regression')
        axs[0].set(xlabel='Slope (log m/m)', ylabel='Raw Attenuation (log cms)')
        axs[0].legend()

        axs[1].hist(residuals, bins=50)
        axs[1].set(xlabel='Regression Residual (log cms)')

        fig.suptitle(h)
        plt.show()

def slope_breaks(value):
    classification = None
    if value == (10 ** -5):
        classification = 'low'
    elif (value <= 0.0025) and (value != (10 ** -5)):
        classification = 'med'
    elif value > 0.0025:
        classification = 'high'
    return classification

def v3():
    in_path = r"G:\floodplainsData\runs\3\muskingum-cunge\mc_out.csv"
    in_data = pd.read_csv(in_path)

    hydrographs = ['Q2_Short', 'Q2_Medium', 'Q2_Long', 'Q10_Short', 'Q10_Medium', 'Q10_Long', 'Q50_Short', 'Q50_Medium', 'Q50_Long', 'Q100_Short', 'Q100_Medium', 'Q100_Long']
    attenuation_type = 'pct_attenuation'
    cols = ['_'.join([h, attenuation_type]) for h in hydrographs]
    wrk_df = in_data[cols]
    wrk_df['ReachCode'] = in_data['ReachCode']
    wrk_df['slope'] = in_data['slope']
    # Set negative attenuation to minimum attenuation
    for c in cols:
        # wrk_df[c][wrk_df[c] <= 0] = np.nanmin(wrk_df[c][wrk_df[c] > 0])
        wrk_df[c] = np.log(wrk_df[c])
    wrk_df = wrk_df.dropna(axis=0)

    wrk_df['slope_class'] = wrk_df['slope'].apply(slope_breaks)
    colors = {'low': '#2ECC40','med': '#001F3F','high': '#FF4136'}
    wrk_df['c'] = wrk_df['slope_class'].map(colors)
    wrk_df['slope'] = np.log(wrk_df['slope'])

    low_df = wrk_df[wrk_df['slope_class'] == 'low']
    med_df = wrk_df[wrk_df['slope_class'] == 'med']
    high_df = wrk_df[wrk_df['slope_class'] == 'high']
    
    for col in cols:
        # Model 1
        low_nan_mask = (~low_df[col].isna())
        mean = np.nanmean(low_df[col][low_nan_mask])
        low_df[col] = low_df[col] - np.exp(mean)
        low_df[col][~low_nan_mask] = np.nanmin(low_df[col][low_nan_mask])

        # Model 2
        med_nan_mask = (~med_df[col].isna())
        med_reg = np.polyfit(med_df['slope'][med_nan_mask], med_df[col][med_nan_mask], 1)
        med_s_space = np.linspace(np.log(10 ** -5), np.log(0.0025), 500)
        med_pred = med_reg[1] + (med_s_space * med_reg[0])
        med_df[col] = med_df[col] - (med_reg[1] + (med_df['slope'] * med_reg[0]))
        med_df[col][~med_nan_mask] = np.nanmin(med_df[col][med_nan_mask])

        # Model 3
        high_nan_mask = (~high_df[col].isna())
        high_reg = np.polyfit(high_df['slope'][high_nan_mask], high_df[col][high_nan_mask], 1)
        high_s_space = np.linspace(np.log(0.0025), max(wrk_df['slope']), 500)
        high_pred = high_reg[1] + (high_s_space * high_reg[0])
        high_df[col] = high_df[col] - (high_reg[1] + (high_df['slope'] * high_reg[0]))
        high_df[col][~high_nan_mask] = np.nanmin(high_df[col][high_nan_mask])

        # KDE
        resid_space = np.linspace(-5, 5, 1000)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(low_df[col].values[:, np.newaxis])
        low_density = kde.score_samples(resid_space[:, np.newaxis])
        low_density = np.exp(low_density) * (len(low_df) / len(wrk_df))
        kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(med_df[col].values[:, np.newaxis])
        med_density = kde.score_samples(resid_space[:, np.newaxis])
        med_density = np.exp(med_density) * (len(med_df) / len(wrk_df))
        med_density = med_density + low_density
        kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(high_df[col].values[:, np.newaxis])
        high_density = kde.score_samples(resid_space[:, np.newaxis])
        high_density = np.exp(high_density) * (len(high_density) / len(wrk_df))
        high_density = med_density + high_density

        y_log = True
        if not y_log:
            low_att = np.exp(low_att)
            med_att = np.exp(med_att)
            high_att = np.exp(high_att)
            mean = np.exp(mean)
            med_pred = np.exp(med_pred)
            high_pred = np.exp(high_pred)

        fig, axs = plt.subplots(ncols=2, figsize=(13.33, 7.5))
        axs[0].scatter(wrk_df['slope'], wrk_df[col], c=wrk_df['c'], s=3, alpha=0.2, label='Low Slope Series')
        axs[0].plot([min(wrk_df['slope']), np.log(10 ** -5)], [mean, mean], ls='solid', c='k', alpha=0.7, label='Low Regression')
        axs[0].plot(med_s_space, med_pred, ls='dashed', c='k', alpha=0.7, label='Medium Regression')
        axs[0].plot(high_s_space, high_pred, ls='dotted', c='k', alpha=0.7, label='High Regression')
        axs[0].set(xlabel='Slope (log m/m)', ylabel='Raw Attenuation (log cms)')
        axs[0].legend()

        axs[1].fill_between(resid_space, low_density, fc="#2ECC40", alpha=0.9, ec='k')
        axs[1].fill_between(resid_space, med_density, low_density, fc="#001F3F", alpha=0.9, ec='k')
        axs[1].fill_between(resid_space, high_density, med_density, fc="#FF4136", alpha=0.9, ec='k')
        axs[1].set(xlabel='Regression Residual (log cms)')

        fig.suptitle(col)
        fig.savefig(r'G:\floodplainsData\runs\3\muskingum-cunge\3pc_detrend\{}.png'.format(col), dpi=300)
        # plt.show()


def segmented_2():
    in_path = r"G:\floodplainsData\runs\3\muskingum-cunge\mc_out.csv"
    in_data = pd.read_csv(in_path)

    hydrographs = ['Q2_Short', 'Q2_Medium', 'Q2_Long', 'Q10_Short', 'Q10_Medium', 'Q10_Long', 'Q50_Short', 'Q50_Medium', 'Q50_Long', 'Q100_Short', 'Q100_Medium', 'Q100_Long']
    attenuation_type = 'raw_attenuation'
    cols = ['_'.join([h, attenuation_type]) for h in hydrographs]
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
        axs[0].set(xlabel='Slope (log m/m)', ylabel='Raw Attenuation (log cms)')
        axs[0].legend()

        axs[1].fill_between(resid_space, np.exp(density), fc="#B10DC9", alpha=0.9, ec='k')
        axs[1].set(xlabel='Regression Residual (log cms)')

        fig.suptitle(col)
        # fig.savefig(r'G:\floodplainsData\runs\3\muskingum-cunge\segmented_detrend_pct\{}.png'.format(col), dpi=300)
        plt.show()
    
    # wrk_df.to_csv(r'G:\floodplainsData\runs\3\muskingum-cunge\detrended_{}.csv'.format(attenuation_type), index=False)

# v3()
segmented_2()