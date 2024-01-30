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

    def route_hydrograph(self, inflows, dt, max_iter=1000):
        outflows = list()
        outflows.append((inflows[0]))
        assert max(inflows) < max(self.geometry['discharge']), 'Rating Curve does not cover range of flowrates in hydrograph'
        
        for i in range(len(inflows) - 1):
            q_guess = sum([inflows[i], inflows[i + 1], outflows[i]]) / 3
            last_guess = q_guess * 2
            counter = 1
            while abs(last_guess - q_guess) > 0.003:  # from handbook of hydrology page 328
                counter += 1
                last_guess = q_guess.copy()
                reach_q = sum([inflows[i], inflows[i + 1], outflows[i], q_guess]) / 4

                # Interpolate
                log_reach_q = np.log(reach_q)
                b_tmp = np.exp(np.interp(log_reach_q, self.geometry['log_q'], self.geometry['log_width']))
                c_tmp = np.interp(log_reach_q, self.geometry['log_q'], self.geometry['celerity'])

                courant = c_tmp * dt * 60 * 60 / self.reach_length
                reynold = reach_q / (self.slope * c_tmp * self.reach_length * b_tmp)
                
                c0 = (-1 + courant + reynold) / (1 + courant + reynold)
                c1 = (1 + courant - reynold) / (1 + courant + reynold)
                c2 = (1 - courant + reynold) / (1 + courant + reynold)

                q_guess = (c0 * inflows[i + 1]) + (c1 * inflows[i]) + (c2 * outflows[i])
                q_guess = max(min(inflows), q_guess)
                if counter == max_iter:
                    last_guess = q_guess
            outflows.append(q_guess)

        return np.array(outflows)

    
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
        geom = self.geometry
        geom['stage'] = stages
        geom['top_width'] = top_widths
        geom['log_width'] = np.log(geom['top_width'])
        geom['area'] = areas
        geom['wetted_perimeter'] = perimeters
        geom['wetted_perimeter'][geom['wetted_perimeter'] <= 0] = geom['wetted_perimeter'][geom['wetted_perimeter'] > 0].min()
        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']
        geom['mannings_n'] = np.repeat(self.mannings_n, len(stages))
        geom['discharge'] = (1 / geom['mannings_n']) * geom['area'] * (geom['hydraulic_radius'] ** (2 / 3)) * (self.slope ** 0.5)
        geom['discharge'][geom['discharge'] <= 0] = geom['discharge'][geom['discharge'] > 0].min()
        self.clean_looped_rating_curve()
        geom['log_q'] = np.log(geom['discharge'])
        dq = geom['discharge'][1:] - geom['discharge'][:-1]
        da = geom['area'][1:] - geom['area'][:-1]
        dq_da = dq / da
        dq_da[0] = dq_da[1]
        dq_da[np.isnan(dq_da)] = 0.0001
        dq_da = gaussian_filter1d(dq_da, 15)
        dq_da[dq_da < 0.0001] = 0.0001
        geom['celerity'] = np.append(dq_da, dq_da[-1])
        

    def clean_looped_rating_curve(self):
        # Enforce monotonic discharge inreases
        q_last = np.nan
        for ind, q in enumerate(self.geometry['discharge']):
            if q < q_last:
                self.geometry['discharge'][ind] = q_last
            else:
                q_last = q


class MuskingumReach:
    
    def __init__(self, k, x):
        self.k = k
        self.x = x

    def route_hydrograph(self, inflows, dt):
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