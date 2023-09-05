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

def duration_over_threshold_hydrograph(timestamps, a, b, ri, peak_ratio=0.17):
    flowrate_func = lambda d: (d / (-a * np.log((1 / ri) ** 2))) ** (1 / b)
    t0_func = lambda d: (peak_ratio * timestamps[-1]) - (peak_ratio * d)
    t1_func = lambda d: (peak_ratio * timestamps[-1]) + ((1 - peak_ratio) * d)
    rising_limb_x = t0_func(timestamps[::-1])
    rising_limb_y = flowrate_func(timestamps[::-1])
    rising_limb_y[-1] = rising_limb_y[-2]
    falling_limb_x = t1_func(timestamps)
    falling_limb_y = flowrate_func(timestamps)
    falling_limb_y[0] = falling_limb_y[1]
    x_series = np.array([*rising_limb_x, *falling_limb_x])
    y_series = np.array([*rising_limb_y, *falling_limb_y])
    out_flowrates = np.interp(timestamps, x_series, y_series)
    return out_flowrates


