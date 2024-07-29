import numpy as np
import cython
from libc.math cimport log, exp

def croute_fread(inflows: cython.double[:], outflows: cython.double[:], dt: cython.float, reach_length: cython.float, slope: cython.float, geometry, short_ts: cython.bint):
    geom_log_q: cython.double[:] = geometry['log_q']
    geom_log_w: cython.double[:] = geometry['log_width']
    geom_q: cython.double[:] = geometry['discharge']
    geom_c: cython.double[:] = geometry['celerity']
    series_length: cython.int = len(inflows)
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
    min_flow: cython.float = np.min(inflows)
    max_iter: cython.int = 1000

    assert np.log(max(inflows)) < max(geom_log_q), 'Rating Curve does not cover range of flowrates in hydrograph'
    
    for i in range(series_length - 1):
        if short_ts:
            us_cur = inflows[i]
        else:
            us_cur = inflows[i + 1]
        q_guess = (inflows[i] + us_cur + outflows[i]) / 3
        last_guess = q_guess * 2
        counter = 1
        while abs(last_guess - q_guess) > 0.003:
            counter += 1
            last_guess = q_guess
            reach_q = (inflows[i] + us_cur + outflows[i] + q_guess) / 4

            # Interpolate
            log_reach_q = log(reach_q)
            b_tmp = exp(np.interp(log_reach_q, geom_log_q, geom_log_w))
            c_tmp = np.interp(log_reach_q, geom_log_q, geom_c)

            courant = c_tmp * dt / reach_length
            reynold = reach_q / (slope * c_tmp * reach_length * b_tmp)
            
            c0 = (-1 + courant + reynold) / (1 + courant + reynold)
            c1 = (1 + courant - reynold) / (1 + courant + reynold)
            c2 = (1 - courant + reynold) / (1 + courant + reynold)

            q_guess = (c0 * us_cur) + (c1 * inflows[i]) + (c2 * outflows[i])
            q_guess = max(min_flow, q_guess)
            if counter == max_iter:
                last_guess = q_guess
        outflows[i + 1] = q_guess

    return np.array(outflows)

def interpolate(x: cython.double, xp: cython.double[:], yp: cython.double[:]):
    i: cython.int
    for i in range(len(xp) - 1):
        if xp[i] <= x <= xp[i + 1]:
            return yp[i] + ((x - xp[i]) * ((yp[i + 1] - yp[i]) / (xp[i + 1] - xp[i])))
    return 0.0

def croute_wrf(
        inflows: cython.double[:],
        outflows: cython.double[:],
        lateral: cython.double[:],
        depth: cython.double,
        stages: cython.double[:],
        cks: cython.double[:],
        twls: cython.double[:],
        qs: cython.double[:],
        slope: cython.double,
        dx: cython.double,
        dt: cython.double,
        short_ts: cython.bint):
    
    max_iter: cython.int = 100
    mindepth: cython.double = 0.01
    i: cython.int

    Qj_0: cython.double
    Qj: cython.double
    iters: cython.int
    rerror: cython.double
    aerror: cython.double
    tries: cython.int

    h: cython.double
    h_0: cython.double
    qup: cython.double = 0
    quc: cython.double = 0
    qdp: cython.double = 0
    qdc: cython.double = 0
    depth: cython.double
    Ck: cython.double
    Twl: cython.double
    q: cython.double
    Km: cython.double
    X: cython.double
    D: cython.double
    C1: cython.double = 0
    C2: cython.double = 0
    C3: cython.double = 0
    C4: cython.double = 0
    h_1: cython.double

    for i in range(len(inflows) - 1):
        if not ((inflows[i] > 0.0) or (inflows[i + 1] > 0.0) or (outflows[i] > 0.0)):
            depth = 0.0
            qdc = 0.0
            outflows[i + 1] = qdc
            continue
        Qj_0 = 0.0
        iters = 0
        rerror = 1.0
        aerror = 0.01
        tries = 0

        h     = (depth * 1.33) + mindepth
        h_0   = (depth * 0.67)

        while rerror > 0.01 and aerror >= mindepth and iters <= max_iter:
            qup = inflows[i]
            if short_ts:
                quc = inflows[i]
            else:
                quc = inflows[i + 1]
            qdp = outflows[i]

            # Lower Interval
            Ck = interpolate(h_0, stages, cks)
            Twl = interpolate(h_0, stages, twls)
            q = interpolate(h_0, stages, qs)

            if Ck > 0.000000:
                Km = max(dt, dx / Ck)
            else:
                Km = dt
            
            if Ck > 0.0:
                X = 0.5*(1-(Qj_0/(2.0*Twl*slope*Ck*dx)))
                X = min(0.5, max(0.0, X))
            else:
                X = 0.5

            D = (Km*(1.000 - X) + dt/2.0000)

            C1 = (Km*X + dt/2.000000)/D
            C2 = (dt/2.0000 - Km*X)/D
            C3 = (Km*(1.00000000-X)-dt/2.000000)/D
            C4 = (lateral[i]*dt)/D
            Qj_0 = ((C1*qup) + (C2*quc) + (C3*qdp) + C4) - q

            # Upper Interval
            Ck = interpolate(h, stages, cks)
            Twl = interpolate(h, stages, twls)
            q = interpolate(h, stages, qs)

            if Ck > 0.000000:
                Km = max(dt, dx / Ck)
            else:
                Km = dt
            
            if Ck > 0.0:
                X = 0.5*(1-(((C1*qup)+(C2*quc)+(C3*qdp) + C4)/(2.0*Twl*slope*Ck*dx)))
                X = min(0.5, max(0.25, X))
            else:
                X = 0.5

            D = (Km*(1.000 - X) + dt/2.0000)

            C1 = (Km*X + dt/2.000000)/D
            C2 = (dt/2.0000 - Km*X)/D
            C3 = (Km*(1.00000000-X)-dt/2.000000)/D
            C4 = (lateral[i]*dt)/D
            Qj = ((C1*qup) + (C2*quc) + (C3*qdp) + C4) - q

            # Secant Method
            if Qj - Qj_0 != 0:
                h_1 = h - ((Qj * (h_0 - h))/(Qj_0 - Qj))
                if h_1 < 0.0:
                    h_1 = h
            else:
                h_1 = h
            
            if h > 0.0:
                rerror = abs((h_1 - h) / h)
                aerror = abs(h_1 - h)
            else:
                rerror = 0.0
                aerror = 0.9
            
            h_0 = max(0.0, h)
            h = max(0.0, h_1)
            iters += 1

            if h < mindepth:
                break

            if iters >= max_iter:
                tries += 1
                if tries <= 4:
                    h = h * 1.33
                    h_0 = h_0 * 0.67
                    max_iter += 25

        if ((C1*qup)+(C2*quc)+(C3*qdp) + C4) < 0:
            if C4 < 0 and abs(C4) > ((C1*qup)+(C2*quc)+(C3*qdp)):
                qdc = 0.0
            else:
                qdc =  max(((C1*qup)+(C2*quc)+C4), ((C1*qup)+(C3*qdp)+C4))
        else:
            qdc =  ((C1*qup)+(C2*quc)+(C3*qdp) + C4)
        if qdc < 0.0:
            qdc = 0.0
        outflows[i + 1] = qdc
        depth = h

    return np.array(outflows)