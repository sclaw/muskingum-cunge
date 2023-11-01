import numpy as np
import cython
from libc.math cimport log, exp

def croute(py_inflows, dt, reach_length, slope, geometry, max_iter=1000, verbose=False):
    inflows: cython.double[:] = py_inflows
    outflows: cython.double[:] = np.empty_like(py_inflows)
    geom_log_q: cython.double[:] = geometry['log_q']
    geom_log_w: cython.double[:] = geometry['log_width']
    geom_q: cython.double[:] = geometry['discharge']
    geom_c: cython.double[:] = geometry['celerity']
    series_length: cython.int = len(py_inflows)
    i: cython.int
    max_c: cython.float = 0
    min_c: cython.float = 999999
    q_guess: cython.float
    last_guess: cython.float
    counter: cython.int
    reach_q: cython.float
    log_reach_q: cython.float
    b_tmp: cython.float
    c_tmp: cython.float
    k_tmp: cython.float
    x_tmp: cython.float
    c0: cython.float
    c1: cython.float
    c2: cython.float
    min_flow: cython.float = py_inflows.min()
    max_iter: cython.int = max_iter
    reach_length: cython.float = reach_length
    dt: cython.float = dt
    slope: cython.float = slope
    min_travel_time: cython.float

    outflows[0] = inflows[0]

    if np.argmax(inflows) < 20:
        print('dt too large')
        print(f'hydrograph peak of {max(inflows)} is at index {np.argmax(inflows)}')
    if np.max(inflows) > np.max(geom_q):
        print(f'WARNING: inflow {round(np.max(inflows), 1)} greater than max flowrate of {round(np.max(geom_q), 1)}')
    
    for i in range(series_length - 1):
        q_guess = (inflows[i] + inflows[i + 1] + outflows[i]) / 3
        last_guess = q_guess * 2
        counter = 1
        while abs(last_guess - q_guess) > 0.003:
            counter += 1
            last_guess = q_guess
            reach_q = (inflows[i] + inflows[i + 1] + outflows[i] + q_guess) / 4

            # Interpolate
            log_reach_q = log(reach_q)
            b_tmp = exp(np.interp(log_reach_q, geom_log_q, geom_log_w))
            c_tmp = np.interp(log_reach_q, geom_log_q, geom_c)

            k_tmp = (reach_length / c_tmp) / (60 * 60)
            x_tmp = 0.5 - (reach_q / (2 * c_tmp * b_tmp * slope * reach_length))
            x_tmp = max(0, x_tmp)
            x_tmp = min(0.5, x_tmp)
            
            max_c = max(max_c, c_tmp)
            min_c = min(min_c, c_tmp)

            c0 = ((dt / k_tmp) - (2 * x_tmp)) / ((2 * (1 - x_tmp)) + (dt / k_tmp))
            c1 = ((dt / k_tmp) + (2 * x_tmp)) / ((2 * (1 - x_tmp)) + (dt / k_tmp))
            c2 = ((2 * (1 - x_tmp)) - (dt / k_tmp)) / ((2 * (1 - x_tmp)) + (dt / k_tmp))

            q_guess = (c0 * inflows[i + 1]) + (c1 * inflows[i]) + (c2 * outflows[i])
            q_guess = max(min_flow, q_guess)
            if counter == max_iter:
                last_guess = q_guess
        outflows[i + 1] = q_guess

    min_travel_time = (reach_length / max_c) / (60 * 60)
    if min_travel_time < dt:
        print('dt too large')
        print(f'Minimum travel time is {min_travel_time} hours')
    return np.array(outflows)