import numpy as np
from scipy.ndimage import gaussian_filter1d
from .route import croute

class BaseReach:
    geometry = {'stage': None, 
                'top_width': None,
                'area': None,
                'wetted_perimeter': None,
                'hydraulic_radius': None,
                'mannings_n': None}
    rating_curve = {'stage': None,
                    'discharge': None}
    muskingum_params = {'stage': None,
                        'k': None,
                        'x': None}

    def __init__(self, width, mannings_n, slope, reach_length, max_stage=10, stage_resolution=50):
        self.width = width
        self.mannings_n = mannings_n
        self.slope = slope
        self.reach_length = reach_length
        self.max_stage = max_stage
        self.resolution = stage_resolution

        self.generate_geometry()

    def generate_geometry(self):
        geom = self.geometry
        geom['stage'] = np.linspace(0.001, self.max_stage, self.resolution)
        geom['top_width'] = np.repeat(self.width, self.resolution)
        geom['log_width'] = np.log(geom['top_width'])
        geom['area'] = geom['stage'] * geom['top_width']
        geom['log_area'] = np.log(geom['area'])
        geom['wetted_perimeter'] = self.width + (2 * geom['stage'])
        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']
        geom['mannings_n'] = np.repeat(self.mannings_n, self.resolution)
        geom['discharge'] = (1 / geom['mannings_n']) * geom['area'] * (geom['hydraulic_radius'] ** (2 / 3)) * (self.slope ** 0.5)
        geom['log_q'] = np.log(geom['discharge'])
        dq = geom['discharge'][1:] - geom['discharge'][:-1]
        da = geom['area'][1:] - geom['area'][:-1]
        dq_da = dq / da
        geom['celerity'] = np.append(dq_da, dq_da[-1])

    def calculate_x_ref(self, inflows, dt):
        q_ref = min(inflows) + ((max(inflows) + min(inflows)) / 2)
        b_ref = np.exp(np.interp(np.log(q_ref), self.geometry['log_q'], self.geometry['log_width']))
        c_ref = np.interp(np.log(q_ref), self.geometry['log_q'], self.geometry['celerity'])
        x_ref = 0.5 * ((c_ref * dt * 60 * 60) + (q_ref / (b_ref * self.slope * c_ref)))
        return x_ref

    def route_hydrograph_c(self, inflows, dt):
        reach_length = self.reach_length
        slope = self.slope
        geometry = self.geometry

        return croute(inflows, dt, reach_length, slope, geometry)

    def route_hydrograph(self, inflows, dt, lateral=None, max_iter=1000, initial_outflow=None):
        dt = dt * 60 * 60  # MUST DELETE AFTER DEBUGGING
        outflows = list()
        if initial_outflow is not None:
            outflows.append(initial_outflow)
        else:
            outflows.append(inflows[0])
        assert max(inflows) < max(self.geometry['discharge']), 'Rating Curve does not cover range of flowrates in hydrograph'

        if lateral is None:
            lateral = np.zeros_like(inflows)
        lateral = (lateral[:-1] + lateral[1:]) / 2
        lateral = np.append(lateral, 0)
        for i in range(len(inflows) - 1):
            # q_guess = sum([inflows[i], inflows[i + 1], outflows[i], lateral[i]]) / 3
            q_guess = sum([inflows[i], inflows[i], outflows[i], lateral[i]]) / 3
            last_guess = q_guess * 2
            counter = 1
            while abs(last_guess - q_guess) > 0.003:  # from handbook of hydrology page 328
                counter += 1
                last_guess = q_guess.copy()
                # reach_q = sum([inflows[i], inflows[i + 1], outflows[i], q_guess]) / 4
                reach_q = sum([inflows[i], inflows[i], outflows[i], q_guess]) / 4

                # Interpolate
                log_reach_q = np.log(reach_q)
                b_tmp = np.exp(np.interp(log_reach_q, self.geometry['log_q'], self.geometry['log_width']))
                c_tmp = np.interp(log_reach_q, self.geometry['log_q'], self.geometry['celerity'])

                # courant = c_tmp * dt * 60 * 60 / self.reach_length
                # reynold = reach_q / (self.slope * c_tmp * self.reach_length * b_tmp)
                
                # c0 = (-1 + courant + reynold) / (1 + courant + reynold)
                # c1 = (1 + courant - reynold) / (1 + courant + reynold)
                # c2 = (1 - courant + reynold) / (1 + courant + reynold)
                # c3 = (2 * courant) / (1 + courant + reynold)

                # q_guess = (c0 * inflows[i + 1]) + (c1 * inflows[i]) + (c2 * outflows[i]) + (c3 * lateral[i])
                # q_guess = max(min(inflows), q_guess)

                # WRF way
                if c_tmp > 0:
                    Km = self.reach_length / c_tmp
                    Km = max(dt, Km)
                else:
                    Km = dt
                X = 0.5*(1-(reach_q/(2.0*b_tmp*self.slope*c_tmp*self.reach_length)))
                X = min(0.5, max(0.0, X))
                D = (Km*(1.000 - X) + (dt)/2.0000)

                C1 =  (Km*X + dt/2.000000)/D
                C2 =  (dt/2.0000 - Km*X)/D
                C3 =  (Km*(1.00000000-X)-dt/2.000000)/D
                C4 =  (lateral[i]*dt)/D
                # q_guess = ((C1*inflows[i]) + (C2*inflows[i + 1]) + (C3*outflows[i]))
                q_guess = ((C1*inflows[i]) + (C2*inflows[i]) + (C3*outflows[i]))
                
                if counter == max_iter:
                    last_guess = q_guess
            outflows.append(q_guess)


        return np.array(outflows)

    def route_hydrograph_mct(self, inflows, dt, max_iter=1000):
        outflows = list()
        outflows.append(inflows[0])
        assert max(inflows) < max(self.geometry['discharge']), 'Rating Curve does not cover range of flowrates in hydrograph'

        # warm up parameters
        qref1 = (inflows[0] + outflows[0]) / 2
        aref1 = np.exp(np.interp(np.log(qref1), self.geometry['log_q'], self.geometry['log_area']))
        bref1 = np.exp(np.interp(np.log(qref1), self.geometry['log_q'], self.geometry['log_width']))
        cref1 = np.interp(np.log(qref1), self.geometry['log_q'], self.geometry['celerity'])
        betaref1 = (cref1 * aref1) / qref1
        courantref1 = (cref1 / betaref1) * (dt * 60 * 60 / self.reach_length)
        reynoldref1 = qref1 / (betaref1 * bref1 * self.slope * cref1 * self.reach_length)
        for i in range(len(inflows) - 1):
            q_guess = outflows[i] + (inflows[i + 1] - inflows[i])
            last_guess = q_guess * 2
            counter = 1
            while abs(last_guess - q_guess) > 0.003:  # from handbook of hydrology page 328
                counter += 1
                last_guess = q_guess.copy()
                
                qref2 = (inflows[i + 1] + q_guess) / 2
                # interpolate
                aref2 = np.exp(np.interp(np.log(qref2), self.geometry['log_q'], self.geometry['log_area']))
                bref2 = np.exp(np.interp(np.log(qref2), self.geometry['log_q'], self.geometry['log_width']))
                cref2 = np.interp(np.log(qref2), self.geometry['log_q'], self.geometry['celerity'])                
                betaref2 = (cref2 * aref2) / qref2
                courantref2 = (cref2 / betaref2) * (dt * 60 * 60 / self.reach_length)                
                reynoldref2 = qref2 / (betaref2 * bref2 * self.slope * cref2 * self.reach_length)

                # MCT parameters
                c0 = (-1 + courantref1 + reynoldref1) / (1 + courantref2 + reynoldref2)
                c1 = ((1 + courantref1 - reynoldref1) / (1 + courantref2 + reynoldref2)) * (courantref2 / courantref1)
                c2 = ((1 - courantref1 + reynoldref1) / (1 + courantref2 + reynoldref2)) * (courantref2 / courantref1)

                # Estimate outflow
                q_guess = (c0 * inflows[i + 1]) + (c1 * inflows[i]) + (c2 * outflows[i])
                q_guess = max(min(inflows), q_guess)
                if counter == max_iter:
                    last_guess = q_guess
            outflows.append(q_guess)
            if (cref2 / betaref2) > 1:
                print((cref2 / betaref2))
            qref1 = qref2
            aref1 = aref2
            bref1 = bref2
            cref1 = bref2
            betaref1 = betaref2
            courantref1 = courantref2
            reynoldref1 = reynoldref2

        return np.array(outflows)
    
    def route_hydrograph_wrf_secant(self, inflows, dt, lateral=None, max_iter=1000, initial_outflow=None):
        # Secant method solver from WRF-HYDRO 
        dt = dt * 60 * 60  # MUST DELETE AFTER DEBUGGING
        dx = self.reach_length
        outflows = list()
        if initial_outflow is not None:
            outflows.append(initial_outflow)
        else:
            outflows.append(inflows[0])

        if max(inflows) > max(self.geometry['discharge']):
            add_res = 200
            add_stages = np.linspace(0, 9 * self.geometry['stage'][-1], add_res)
            add_widths = np.repeat(self.geometry['top_width'][-1], add_res)
            add_areas = (add_stages * add_widths) + self.geometry['area'][-1]
            add_wp = (2 * add_stages) + self.geometry['wetted_perimeter'][-1]
            add_hr = add_areas / add_wp
            add_n = np.repeat(self.geometry['mannings_n'][-1], add_res)
            add_q = (1 / add_n) * add_areas * (add_hr ** (2 / 3)) * (self.slope ** 0.5)
            dpdy = np.repeat(2, add_res)
            k_prime = (5 / 3) - ((2 / 3) * (add_areas / (add_widths * add_wp)) * dpdy)
            add_c = k_prime * (add_q / add_areas)

            self.geometry['stage'] = np.append(self.geometry['stage'], add_stages + self.geometry['stage'][-1])
            self.geometry['top_width'] = np.append(self.geometry['top_width'], add_widths)
            self.geometry['area'] = np.append(self.geometry['area'], add_areas)
            self.geometry['wetted_perimeter'] = np.append(self.geometry['wetted_perimeter'], add_wp)
            self.geometry['hydraulic_radius'] = np.append(self.geometry['hydraulic_radius'], add_hr)
            self.geometry['mannings_n'] = np.append(self.geometry['mannings_n'], add_n)
            self.geometry['discharge'] = np.append(self.geometry['discharge'], add_q)
            self.geometry['celerity'] = np.append(self.geometry['celerity'], add_c)
                
            assert max(inflows) < max(self.geometry['discharge']), 'Rating Curve does not cover range of flowrates in hydrograph'

        if lateral is None:
            lateral = np.zeros_like(inflows)
        lateral = (lateral[:-1] + lateral[1:]) / 2
        lateral = np.append(lateral, 0)

        # local variables
        mindepth = 0.01
        max_iter = 100
        short_ts = True

        # initialize secant method
        depth = np.interp(outflows[0], self.geometry['discharge'], self.geometry['stage'])
        h     = (depth * 1.33) + mindepth
        h_0   = (depth * 0.67)

        for i in range(len(inflows) - 1):
            if not ((inflows[i] > 0.0) or (inflows[i + 1] > 0.0) or (outflows[i] > 0.0)):
                outflows.append(0.0)
                continue
            Qj_0 = 0.0
            iter = 0
            rerror = 1.0
            aerror = 0.01
            tries = 0

            while rerror > 0.01 and aerror >= mindepth and iter <= max_iter:
                qup = inflows[i]
                if short_ts:
                    quc = inflows[i]
                else:
                    quc = inflows[i + 1]
                qdp = outflows[i]

                # Lower Interval
                Ck = np.interp(h_0, self.geometry['stage'], self.geometry['celerity'])
                Twl = np.interp(h_0, self.geometry['stage'], self.geometry['top_width'])
                q = np.interp(h_0, self.geometry['stage'], self.geometry['discharge'])
                
                if Ck > 0.000000:
                    Km = max(dt, dx / Ck)
                else:
                    Km = dt
                
                X = 0.5*(1-(Qj_0/(2.0*Twl*self.slope*Ck*dx)))
                X = min(0.5, max(0.0, X))

                D = (Km*(1.000 - X) + dt/2.0000)

                C1 = (Km*X + dt/2.000000)/D
                C2 = (dt/2.0000 - Km*X)/D
                C3 = (Km*(1.00000000-X)-dt/2.000000)/D
                C4 = (lateral[i]*dt)/D
                Qj_0 = ((C1*quc) + (C2*qup) + (C3*qdp) + C4) - q

                # Upper Interval
                Ck = np.interp(h, self.geometry['stage'], self.geometry['celerity'])
                Twl = np.interp(h, self.geometry['stage'], self.geometry['top_width'])
                q = np.interp(h, self.geometry['stage'], self.geometry['discharge'])

                if Ck > 0.000000:
                    Km = max(dt, dx / Ck)
                else:
                    Km = dt
                
                if Ck > 0.0:
                    X = 0.5*(1-(((C1*qup)+(C2*quc)+(C3*qdp) + C4)/(2.0*Twl*self.slope*Ck*dx)))
                    X = min(0.5, max(0.25, X))
                else:
                    X = 0.5

                D = (Km*(1.000 - X) + dt/2.0000)

                C1 = (Km*X + dt/2.000000)/D
                C2 = (dt/2.0000 - Km*X)/D
                C3 = (Km*(1.00000000-X)-dt/2.000000)/D
                C4 = (lateral[i]*dt)/D
                Qj = ((C1*quc) + (C2*qup) + (C3*qdp) + C4) - q

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
                iter += 1

                if h < mindepth:
                    break

                if iter >= max_iter:
                    tries += 1
                    if tries <= 4:
                        h = h * 1.33
                        h_0 = h_0 * 0.67
                        max_iter += 25

            qdc =  ((C1*qup)+(C2*quc)+(C3*qdp) + C4)
            if qdc < 0.0:
                qdc = 0.0
            outflows.append(qdc)

        return np.array(outflows)

    def optimize_route_params(self, inflows, dt):
        # Ponce method
        qref = (inflows.max() + inflows.min()) / 2
        cref = np.interp(qref, self.geometry['discharge'], self.geometry['celerity'])
        twref = np.interp(qref, self.geometry['discharge'], self.geometry['top_width'])
        dxc = dt * 60 * 60 * cref  # Courant length
        dxd = (qref / twref) / (self.slope * cref)  # characteristic reach length
        dxmax = 0.5 * (dxc + dxd)
        peak_loc = np.argmax(self.geometry['discharge'] > inflows.max())
        peak_loc = max([peak_loc, 1])
        cmax = self.geometry['celerity'][:peak_loc].max()  # I think the fact that this is the max should cover us for making all stable
        dxmin = cmax * (dt * 60 * 60)
        dx = max([dxmin, dxmax])
        subreaches = int(np.ceil(self.reach_length / dx))
        dx = self.reach_length / subreaches
        return dx, subreaches
   
class TrapezoidalReach(BaseReach):

    def __init__(self, bottom_width, side_slope, mannings_n, slope, reach_length, max_stage=10, stage_resolution=50):
        self.bottom_width = bottom_width
        self.side_slope = side_slope
        self.mannings_n = mannings_n
        self.slope = slope
        self.reach_length = reach_length
        self.max_stage = max_stage
        self.resolution = stage_resolution

        self.generate_geometry()
    
    def generate_geometry(self):
        geom = self.geometry
        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)
        geom['top_width'] = self.bottom_width + (geom['stage'] * self.side_slope)
        geom['log_width'] = np.log(geom['top_width'])
        geom['area'] = (((geom['stage'] * self.side_slope) + (2 * self.bottom_width)) / 2) * geom['stage']
        geom['wetted_perimeter'] = ((((geom['stage'] * self.side_slope) ** 2) + (geom['stage'] ** 2)) ** 0.5) + self.bottom_width
        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']
        geom['mannings_n'] = np.repeat(self.mannings_n, self.resolution)
        geom['discharge'] = (1 / geom['mannings_n']) * geom['area'] * (geom['hydraulic_radius'] ** (2 / 3)) * (self.slope ** 0.5)
        geom['log_q'] = np.log(geom['discharge'])
        dq = geom['discharge'][1:] - geom['discharge'][:-1]
        da = geom['area'][1:] - geom['area'][:-1]
        dq_da = dq / da
        geom['celerity'] = np.append(dq_da, dq_da[-1])

class CompoundReach(BaseReach):

    def __init__(self, bottom_width, side_slope, bankfull_depth, floodplain_width, channel_n, floodplain_n, slope, reach_length, max_stage=10, stage_resolution=50):
        self.bottom_width = bottom_width
        self.side_slope = side_slope
        self.bankfull_depth = bankfull_depth
        assert floodplain_width >= self.bottom_width + (self.side_slope * self.bankfull_depth), 'floodplain width smaller than channel width at bankfull depth'
        self.floodplain_width = floodplain_width
        self.channel_n = channel_n
        self.floodplain_n = floodplain_n
        self.slope = slope
        self.reach_length = reach_length
        self.max_stage = max_stage
        self.resolution = stage_resolution

        self.generate_geometry()
    
    def generate_geometry(self):
        geom = self.geometry

        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)
        geom['top_width'] = self.bottom_width + (geom['stage'] * self.side_slope)
        geom['area'] = (((geom['stage'] * self.side_slope) + (2 * self.bottom_width)) / 2) * geom['stage']
        geom['wetted_perimeter'] = (((((geom['stage'] * (self.side_slope / 2)) ** 2) + (geom['stage'] ** 2)) ** 0.5) * 2) + self.bottom_width
        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']
        geom['discharge'] = (1 / self.channel_n) * geom['area'] * (geom['hydraulic_radius'] ** (2 / 3)) * (self.slope ** 0.5)

        # Add compound channel
        stage_subset = geom['stage'][geom['stage'] > self.bankfull_depth] - self.bankfull_depth
        tw_at_bkf = np.interp(self.bankfull_depth, geom['stage'], geom['top_width'])
        a_at_bkf = np.interp(self.bankfull_depth, geom['stage'], geom['area'])
        q_at_bkf = np.interp(self.bankfull_depth, geom['stage'], geom['discharge'])
        area = stage_subset * self.floodplain_width
        perimeter = (self.floodplain_width - tw_at_bkf) + (2 * stage_subset)
        radius = area / perimeter
        discharge = (1 / self.floodplain_n) * area * (radius ** (2 / 3)) * (self.slope ** 0.5)

        geom['wetted_perimeter'][geom['stage'] > self.bankfull_depth] = perimeter
        geom['area'][geom['stage'] > self.bankfull_depth] = area + a_at_bkf
        geom['top_width'][geom['stage'] > self.bankfull_depth] = self.floodplain_width
        geom['discharge'][geom['stage'] > self.bankfull_depth] = discharge + q_at_bkf

        dq = geom['discharge'][1:] - geom['discharge'][:-1]
        da = geom['area'][1:] - geom['area'][:-1]
        dq_da = dq / da
        geom['celerity'] = np.append(dq_da, dq_da[-1])

        # dpdy = (geom['wetted_perimeter'][1:] - geom['wetted_perimeter'][:-1]) / (geom['stage'][1:] - geom['stage'][:-1])
        # dpdy = np.append(dpdy, dpdy[-1])
        # dpdy[dpdy < 0.0001] = 0.0001
        # k_prime = (5 / 3) - ((2 / 3) * (geom['area'] / (geom['top_width'] * geom['wetted_perimeter'])) * dpdy)
        # geom['celerity'] = k_prime * (geom['discharge'] / geom['area'])

        geom['celerity'][geom['celerity'] < 0.0001] = 0.0001
        geom['log_q'] = np.log(geom['discharge'])

        geom['log_width'] = np.log(geom['top_width'])

class USACERectangle(BaseReach):
    """ Equations from EM 1110-2-1417
    https://www.publications.usace.army.mil/portals/76/publications/engineermanuals/em_1110-2-1417.pdf
    """
    
    def generate_geometry(self):
        geom = self.geometry
        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)
        geom['area'] = geom['stage'] * self.width
        self.alpha = (1 / self.mannings_n) * (self.slope ** 0.5) * (self.width ** (-2 / 3))
        self.m = (5 / 3)

    def generate_rating_curve(self):
        self.rating_curve['stage'] = self.geometry['stage']
        self.rating_curve['discharge'] = self.alpha * (self.geometry['area'] ** self.m)

    def generate_muskingum_params(self):
        self.muskingum_params['stage'] = self.geometry['stage']
        c = self.alpha * self.m * (self.geometry['area'] ** (self.m - 1))
        c[0] = c[1]

        self.muskingum_params['k'] = self.reach_length / c
        self.muskingum_params['k'] /= (60 * 60)  # Convert seconds to hours
        self.muskingum_params['x'] = (1 / 2) - (self.rating_curve['discharge'] / (2 * c * self.width * self.slope * self.reach_length))
        self.muskingum_params['x'][self.muskingum_params['x'] < 0] = 0

class CustomReach(BaseReach):
    geometry = {'stage': None, 
                'top_width': None,
                'area': None,
                'wetted_perimeter': None,
                'hydraulic_radius': None,
                'mannings_n': None}
    rating_curve = {'stage': None,
                    'discharge': None}
    muskingum_params = {'stage': None,
                        'k': None,
                        'x': None}

    def __init__(self, mannings_n, slope, reach_length, stages, top_widths, areas, perimeters):
        self.mannings_n = mannings_n
        self.slope = slope
        self.reach_length = reach_length

        self.generate_geometry(stages, top_widths, areas, perimeters)

    def generate_geometry(self, stages, top_widths, areas, perimeters):
        geom = {'stage': None, 
                'top_width': None,
                'area': None,
                'wetted_perimeter': None,
                'hydraulic_radius': None,
                'mannings_n': None,
                'discharge': None}
        geom['stage'] = stages
        geom['top_width'] = top_widths
        geom['log_width'] = np.log(geom['top_width'])
        geom['area'] = areas
        geom['wetted_perimeter'] = perimeters
        geom['wetted_perimeter'][geom['wetted_perimeter'] <= 0] = geom['wetted_perimeter'][geom['wetted_perimeter'] > 0].min()
        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']
        if type(self.mannings_n) == np.ndarray:
            geom['mannings_n'] = self.mannings_n
        else:
            geom['mannings_n'] = np.repeat(self.mannings_n, len(stages))
        geom['discharge'] = (1 / geom['mannings_n']) * geom['area'] * (geom['hydraulic_radius'] ** (2 / 3)) * (self.slope ** 0.5)
        geom['discharge'][geom['discharge'] <= 0] = geom['discharge'][geom['discharge'] > 0].min()
        geom['discharge'] = self.clean_looped_rating_curve(geom['discharge'])
        geom['log_q'] = np.log(geom['discharge'])

        dp = geom['wetted_perimeter'][1:] - geom['wetted_perimeter'][:-1]
        dy = geom['stage'][1:] - geom['stage'][:-1]
        dp_dy = dp / dy
        dp_dy[0] = dp_dy[1]
        dp_dy[np.isnan(dp_dy)] = 0.0001
        dp_dy = gaussian_filter1d(dp_dy, 15)
        dp_dy[dp_dy < 0.0001] = 0.0001
        dp_dy = np.append(dp_dy, dp_dy[-1])
        k_prime = (5 / 3) - ((2 / 3)*(geom['area'] / (geom['top_width'] * geom['wetted_perimeter'])) * dp_dy)
        geom['celerity'] = k_prime * (geom['discharge'] / geom['area'])
        geom['celerity'][0] = geom['celerity'][1]
        geom['celerity'] = np.nan_to_num(geom['celerity'])
        ## TODO:  geom['celerity'][<0] = 0
        self.geometry = geom
        

    def clean_looped_rating_curve(self, discharge):
        # Enforce monotonic discharge inreases
        q_last = np.nan
        for ind, q in enumerate(discharge):
            if q < q_last:
                discharge[ind] = q_last
            else:
                q_last = q
        return discharge

class MuskingumReach:
    
    def __init__(self, k, x):
        self.k = k
        self.x = x

    def route_hydrograph(self, inflows, dt, lateral=None):
        outflows = list()
        outflows.append((inflows[0]))

        c0 = ((dt / self.k) - (2 * self.x)) / ((2 * (1 - self.x)) + (dt / self.k))
        c1 = ((dt / self.k) + (2 * self.x)) / ((2 * (1 - self.x)) + (dt / self.k))
        c2 = ((2 * (1 - self.x)) - (dt / self.k)) / ((2 * (1 - self.x)) + (dt / self.k))
        for i in range(len(inflows) - 1):
            q_out = (c0 * inflows[i + 1]) + (c1 * inflows[i]) + (c2 * outflows[i])
            q_out = max(min(inflows), q_out)
            outflows.append(q_out)
        return outflows

class WRFCompound(BaseReach):
    """ Equations from lines 276 & 390 of WRF-HYDRO gh page 
    https://github.com/NCAR/wrf_hydro_nwm_public/blob/v5.2.0-rc2/trunk/NDHMS/Routing/module_channel_routing.F
    """
    
    def __init__(self, bottom_width, side_slope, bankfull_depth, floodplain_width, channel_n, floodplain_n, slope, reach_length, max_stage=10, stage_resolution=50):
        self.Bw = bottom_width
        self.z = side_slope / 2  # Should be rise/run per side
        self.bfd = bankfull_depth
        self.TwCC = floodplain_width
        self.n = channel_n
        self.nCC = floodplain_n
        self.So = slope
        self.slope = slope
        self.reach_length = reach_length
        self.max_stage = max_stage
        self.resolution = stage_resolution

        self.generate_geometry()

    def generate_geometry(self):
        geom = dict()
        geom = {'stage': None, 
                'top_width': None,
                'area': None,
                'wetted_perimeter': None,
                'hydraulic_radius': None,
                'mannings_n': None,
                'discharge': None}
        
        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)
        for param in ['top_width', 'area', 'wetted_perimeter', 'hydraulic_radius', 'mannings_n', 'discharge', 'celerity']:
            geom[param] = np.zeros(geom['stage'].shape)

        for ind, h_0 in enumerate(geom['stage']):
            Ck, WP, WPC, n, nCC, AREA, AREAC, R, So = self.get_geom_at_stage(h_0)
            if h_0 > self.bfd:
                Twl = self.TwCC
            else:
                Twl = self.Bw + 2.0*self.z*h_0
            q = ((1/(((WP*n)+(WPC*nCC))/(WP+WPC))) * (AREA+AREAC) * (R**(2./3.)) * np.sqrt(So))
            geom['top_width'][ind] = Twl
            geom['area'][ind] = AREA + AREAC
            geom['wetted_perimeter'][ind] = WP + WPC
            geom['hydraulic_radius'][ind] = R
            geom['mannings_n'][ind] = ((n * WP) + (nCC * WPC)) / (WP + WPC)
            geom['celerity'][ind] = Ck
            geom['discharge'][ind] = q
        
        self.geometry = geom

    def get_geom_at_stage(self, h_0):
        Bw = self.Bw
        bfd = self.bfd
        z = self.z
        TwCC = self.TwCC
        So = self.So
        n = self.n
        nCC = self.nCC

        WPC = 0
        AREAC = 0

        if h_0 > bfd:
            AREA = (Bw + bfd * z) * bfd
            AREAC = TwCC * (h_0 - bfd)
            WP = (Bw + 2.0 * bfd * np.sqrt(1.0 + z*z))
            WPC = TwCC + (2.0 * (h_0 - bfd))
            R = (AREA + AREAC) / (WP + WPC)
        else:
            AREA = (Bw + h_0 * z) * h_0
            WP = (Bw + 2.0 * h_0 * np.sqrt(1.0 + z*z))
            if WP > 0:
                R = AREA / WP
            else:
                R = 0.0
        
        if h_0 > bfd:
            Ck = ((np.sqrt(So)/n)*((5./3.)*R**(2./3.) - ((2./3.)*R**(5./3.)*(2.0*np.sqrt(1.0 + z*z)/(Bw+2.0*bfd*z))))*AREA + ((np.sqrt(So)/(nCC))*(5./3.)*(h_0-bfd)**(2./3.))*AREAC)/(AREA+AREAC)
            Ck = max(0.0, Ck)
        else:
            if h_0 > 0.0:
                Ck = max(0.0,(np.sqrt(So)/n)*((5./3.)*R**(2./3.) - ((2./3.)*R**(5./3.)*(2.0*np.sqrt(1.0 + z*z)/(Bw+2.0*h_0*z)))))
            else:
                Ck = 0.0
        return Ck, WP, WPC, n, nCC, AREA, AREAC, R, So


class SigmoidalReach(BaseReach):
    geometry = {'stage': None, 
                'top_width': None,
                'area': None,
                'wetted_perimeter': None,
                'hydraulic_radius': None,
                'mannings_n': None}
    rating_curve = {'stage': None,
                    'discharge': None}
    muskingum_params = {'stage': None,
                        'k': None,
                        'x': None}

    def __init__(self, mannings_n, slope, reach_length, ch_w, fp_w, bkf_el, fp_s, max_stage=10, stage_resolution=50):
        self.mannings_n = mannings_n
        self.slope = slope
        self.reach_length = reach_length
        self.ch_w = ch_w
        self.fp_w = fp_w
        self.bkf_el = bkf_el
        self.fp_s = fp_s
        self.geometry = None
        self.max_stage = max_stage
        self.resolution = stage_resolution

        self.generate_geometry()

    def generate_geometry(self):
        geom = self.geometry

        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)
        geom['mannings_n'] = np.repeat(self.mannings_n, len(geom['stage']))
        
        L = self.fp_w - self.ch_w
        d_el = geom['stage'][1:] - geom['stage'][:-1]
        d_el = np.append(d_el, d_el[-1])

        geom['top_width'] = (L / (1 + np.exp(-self.fp_s * (geom['stage'] - self.bkf_el)))) + self.ch_w
        geom['log_width'] = np.log(geom['top_width'])

        geom['area'] = d_el * geom['top_width']
        geom['area'] = np.cumsum(geom['area'])
        d_w = geom['top_width'][1:] - geom['top_width'][:-1]
        d_w = np.append(d_w, d_w[-1])
        geom['wetted_perimeter'] = np.sqrt((d_el ** 2) + (d_w ** 2))
        geom['wetted_perimeter'] = np.cumsum(geom['wetted_perimeter']) + self.ch_w
        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']

        rhp = (geom['hydraulic_radius'][1:] - geom['hydraulic_radius'][:-1]) / (geom['stage'][1:] - geom['stage'][:-1])
        rhp = np.append(rhp, rhp[-1])

        geom['discharge'] = (1 / geom['mannings_n']) * geom['area'] * (geom['hydraulic_radius'] ** (2 / 3)) * (self.slope ** 0.5)
        geom['discharge'][geom['discharge'] <= 0] = geom['discharge'][geom['discharge'] > 0].min()
        geom['log_q'] = np.log(geom['discharge'])

        dp = geom['wetted_perimeter'][1:] - geom['wetted_perimeter'][:-1]
        dy = geom['stage'][1:] - geom['stage'][:-1]
        dp_dy = dp / dy
        dp_dy[0] = dp_dy[1]
        dp_dy[np.isnan(dp_dy)] = 0.0001
        dp_dy[dp_dy < 0.0001] = 0.0001
        dp_dy = np.append(dp_dy, dp_dy[-1])
        k_prime = (5 / 3) - ((2 / 3)*(geom['area'] / (geom['top_width'] * geom['wetted_perimeter'])) * dp_dy)
        geom['celerity'] = k_prime * (geom['discharge'] / geom['area'])
        geom['celerity'][0] = geom['celerity'][1]
        geom['celerity'] = np.nan_to_num(geom['celerity'])
        