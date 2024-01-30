import numpy as np


def parabolic_hydrograph_generator(timestamps, peak, duration):
    a = np.array([[(duration / 2) ** 2, duration / 2],
                  [duration ** 2, duration]])
    b = np.array([[peak],
                  [0]])
    coeffs = np.linalg.inv(a).dot(b)
    flows = (coeffs[0] * (timestamps ** 2)) + (coeffs[1] * timestamps)
    flows[flows < 1] = 1
    return flows

def triangular(timestamps, peak, duration, peak_ratio=0.375):
    rise_rate = peak / (peak_ratio * duration)
    peak_time = peak_ratio * duration
    flows = timestamps * rise_rate

    fall_rate = peak / ((1 - peak_ratio) * duration)
    flows[timestamps > peak_time] = peak - ((timestamps[timestamps > peak_time] - peak_time) * fall_rate)

    flows[np.argmax(flows)] = peak
    flows[flows < 0] = 0

    return flows


def duration_over_threshold_hydrograph(timestamps, a, b, ri, da_sqkm, peak_ratio=0.17):
    # get ffa parameters from regression
    log_da = np.log(da_sqkm)
    alpha = np.exp(2.439459 + (log_da * 0.774109))
    xi = np.exp(3.163143 + (log_da * 0.804755))
    k = -0.49031 + (log_da * 0.064844)

    # calc max flowrate and make interpolation space
    q_ri = xi + ((alpha * (1 - ((1 / ri) ** k))) / k)
    q_space = np.linspace(1, q_ri, 5000)

    # calc durations for each intermediate flowrate
    lambda_q = (1 - ((k / alpha) * (q_space - xi))) ** (-1 / k)
    numerator = np.log((1 / ri) * lambda_q)
    denomenator = -a * (q_space ** b)
    d = numerator * denomenator
    d /= 60 # convert minutes to hours

    # interpolate flow grid
    t0_func = lambda d: (peak_ratio * timestamps[-1]) - (peak_ratio * d)
    t1_func = lambda d: (peak_ratio * timestamps[-1]) + ((1 - peak_ratio) * d)
    rising_limb_x = t0_func(d)
    falling_limb_x = t1_func(d[::-1])
    x_series = np.array([*rising_limb_x, *falling_limb_x])
    y_series = np.array([*q_space, *q_space[::-1]])
    out_flowrates = np.interp(timestamps, x_series, y_series)

    # Ensure peak flowrate
    out_flowrates[np.argmax(out_flowrates)] = q_ri
    return out_flowrates

def duration_over_threshold_hydrograph_2(timestamps, a, b, ri, da_sqkm, peak_ratio=0.17):
    # get ffa parameters from regression
    log_da = np.log(da_sqkm)
    alpha = np.exp(2.439459 + (log_da * 0.774109))
    xi = np.exp(3.163143 + (log_da * 0.804755))
    k = -0.49031 + (log_da * 0.064844)

    # calc max flowrate and make interpolation space
    q_ri = xi + ((alpha * (1 - ((1 / ri) ** k))) / k)
    q_space = np.linspace(1, q_ri, 5000)

    # calc durations for each intermediate flowrate
    lambda_q = (1 - ((k / alpha) * (q_space - xi))) ** (-1 / k)
    numerator = np.log((1 / ri) * lambda_q)
    denomenator = -a * (q_space ** b)
    d = numerator * denomenator
    d /= 60 # convert minutes to hours
    peak_ratios = (q_space * 1.14 * (10 ** -6)) + 0.05
    peak_ratios = np.linspace(0.05, 0.1, len(q_space))

    # interpolate flow grid
    rising_limb_x = (peak_ratios * timestamps[-1]) - (peak_ratios * d)
    falling_limb_x = (peak_ratios[::-1] * timestamps[-1]) + ((1 - peak_ratios[::-1]) * d[::-1])
    x_series = np.array([*rising_limb_x, *falling_limb_x])
    y_series = np.array([*q_space, *q_space[::-1]])
    out_flowrates = np.interp(timestamps, x_series, y_series)

    # Ensure peak flowrate
    out_flowrates[np.argmax(out_flowrates)] = q_ri
    return out_flowrates

def scs_hydrograph(peak, duration, timesteps):
    t_tp_ordinates = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5])
    q_qp_ordinates = np.array([0.03, 0.03, 0.1, 0.19, 0.31, 0.47, 0.66, 0.82, 0.93, 0.99, 1, 0.99, 0.93, 0.86, 0.78, 0.68, 0.56, 0.46, 0.39, 0.33, 0.28, 0.207, 0.147, 0.107, 0.077, 0.055, 0.04, 0.029, 0.021, 0.015, 0.011, 0.005, 0.005])

    tmp_flows = q_qp_ordinates * peak
    tmp_times = t_tp_ordinates * duration

    out_flowrates = np.interp(timesteps, tmp_times, tmp_flows)
    # Ensure peak flowrate
    out_flowrates[np.argmax(out_flowrates)] = peak
    return out_flowrates