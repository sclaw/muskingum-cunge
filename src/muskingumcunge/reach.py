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

    def route_hydrograph(self, inflows, dt, lateral=None, max_iter=1000):
        outflows = list()
        outflows.append((inflows[0]))
        assert max(inflows) < max(self.geometry['discharge']), 'Rating Curve does not cover range of flowrates in hydrograph'

        if lateral is None:
            lateral = np.zeros_like(inflows)
        lateral = (lateral[:-1] + lateral[1:]) / 2
        lateral = np.append(lateral, 0)
        for i in range(len(inflows) - 1):
            q_guess = sum([inflows[i], inflows[i + 1], outflows[i], lateral[i]]) / 3
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

                courant = c_tmp * dt * 60 * 60 / self.reach_length
                reynold = reach_q / (self.slope * c_tmp * self.reach_length * b_tmp)
                
                c0 = (-1 + courant + reynold) / (1 + courant + reynold)
                c1 = (1 + courant - reynold) / (1 + courant + reynold)
                c2 = (1 - courant + reynold) / (1 + courant + reynold)
                c3 = (2 * courant) / (1 + courant + reynold)

                q_guess = (c0 * inflows[i + 1]) + (c1 * inflows[i]) + (c2 * outflows[i]) + (c3 * lateral[i])
                q_guess = max(min(inflows), q_guess)
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
        self.z = side_slope  # Should be rise/run per side
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
        geom = {'stage': None, 
                'top_width': None,
                'area': None,
                'wetted_perimeter': None,
                'hydraulic_radius': None,
                'mannings_n': None,
                'discharge': None}

        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)

        # Trapezoidal Channel
        geom['area'] = (self.Bw + geom['stage'] * self.z) * geom['stage']
        geom['wetted_perimeter'] = (self.Bw + 2.0 * geom['stage'] * np.sqrt(1.0 + self.z*self.z))
        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']
        read_celerity = lambda y, r, n: (np.sqrt(self.So)/n)*((5./3.)*r**(2./3.) - ((2./3.)*r**(5./3.)*(2.0*np.sqrt(1.0 + self.z*self.z)/(self.Bw+2.0*y*self.z))))
        geom['celerity'] = read_celerity(geom['stage'], geom['hydraulic_radius'], self.n)
        geom['mannings_n'] = np.repeat(self.n, self.resolution)
        geom['top_width'] = self.Bw + (2 * self.z * geom['stage'])

        # Compound Channel
        mask = (geom['stage'] > self.bfd)
        area_trap = (self.Bw + self.bfd * self.z) * self.bfd
        wetted_trap = (self.Bw + 2.0 * self.bfd * np.sqrt(1.0 + self.z*self.z))
        hr_trap = area_trap / wetted_trap
        celerity_trap = read_celerity(self.bfd, hr_trap, self.n)
        area_CC = (geom['stage'] - self.bfd) * self.TwCC
        wetted_CC = (self.TwCC + (2.0 * (geom['stage'] - self.bfd)))
        radius = (area_CC + area_trap) / (wetted_CC + wetted_trap)
        celerity_cc = read_celerity(geom['stage'], radius, self.nCC)
        celerity_CC = ((celerity_trap * area_trap) + (celerity_cc * area_CC)) / (area_trap + area_CC)
        n_cc = ((self.n * wetted_trap) + (wetted_CC * self.nCC)) / (wetted_trap + wetted_CC)

        geom['area'][mask] = area_CC[mask]
        geom['wetted_perimeter'][mask] = wetted_CC[mask]
        geom['hydraulic_radius'][mask] = radius[mask]
        geom['celerity'][mask] = celerity_CC[mask]
        if np.any(geom['celerity'] < 0) or np.any(np.isnan(geom['celerity'])) or np.any(np.isinf(geom['celerity'])):
            print('bad celerity')
        geom['mannings_n'][mask] = n_cc[mask]
        geom['top_width'][mask] = self.TwCC

        geom['discharge'] = (1.0 / geom['mannings_n']) * geom['area'] * (geom['hydraulic_radius'] ** (2.0 / 3.0)) * (self.So ** 0.5)
        geom['log_q'] = np.log(geom['discharge'])
        geom['log_width'] = np.log(geom['top_width'])
        self.geometry = geom


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
        