import numpy as np
import cython
from libc.math cimport log, exp

def croute(py_inflows, dt, reach_length, slope, geometry, max_iter=1000):
    inflows: cython.double[:] = py_inflows
    outflows: cython.double[:] = np.empty_like(py_inflows)
    geom_log_q: cython.double[:] = geometry['log_q']
    geom_log_w: cython.double[:] = geometry['log_width']
    geom_q: cython.double[:] = geometry['discharge']
    geom_c: cython.double[:] = geometry['celerity']
    series_length: cython.int = len(py_inflows)
    i: cython.int
    q_guess: cython.float
    last_guess: cython.float
    counter: cython.int
    reach_q: cython.float
    log_reach_q: cython.float
    b_tmp: cython.float
    c_tmp: cython.float
    courant: cython.float
    reynold: cython.float
    c0: cython.float
    c1: cython.float
    c2: cython.float
    min_flow: cython.float = py_inflows.min()
    max_iter: cython.int = max_iter
    reach_length: cython.float = reach_length
    dt: cython.float = dt
    slope: cython.float = slope

    assert np.log(max(inflows)) < max(geom_log_q), 'Rating Curve does not cover range of flowrates in hydrograph'

    outflows[0] = inflows[0]
    
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

            courant = c_tmp * dt * 60 * 60 / reach_length
            reynold = reach_q / (slope * c_tmp * reach_length * b_tmp)
            
            c0 = (-1 + courant + reynold) / (1 + courant + reynold)
            c1 = (1 + courant - reynold) / (1 + courant + reynold)
            c2 = (1 - courant + reynold) / (1 + courant + reynold)

            q_guess = (c0 * inflows[i + 1]) + (c1 * inflows[i]) + (c2 * outflows[i])
            q_guess = max(min_flow, q_guess)
            if counter == max_iter:
                last_guess = q_guess
        outflows[i + 1] = q_guess

    return np.array(outflows)