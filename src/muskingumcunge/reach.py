import numpy as np


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
        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)
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
        q_ref = (max(inflows) + min(inflows)) / 2
        stage_ref = np.interp(q_ref, self.geometry['discharge'], self.geometry['stage'])
        b_ref = np.interp(stage_ref, self.geometry['stage'], self.geometry['top_width'])
        c_ref = np.interp(stage_ref, self.geometry['stage'], self.geometry['celerity'])
        x_ref = 0.5 * ((c_ref * dt * 60 * 60) + (q_ref / (b_ref * self.slope * c_ref)))
        return x_ref


    def route_hydrograph(self, inflows, dt):
        if np.argmax(inflows) < 20:
            print('dt too large')
            print(f'hydrograph peak of {max(inflows)} is at index {np.argmax(inflows)}')
        outflows = list()
        outflows.append((inflows[0]))
        max_c = 0
        min_c = 99999
        x_ref = self.calculate_x_ref(inflows, dt)
        if self.reach_length > x_ref:
                print(f'WARNING: reach length {self.reach_length} greater than x_ref of {round(x_ref, 1)}')
        for i in range(len(inflows) - 1):
            if inflows[i] > max(self.geometry['discharge']):
                    print(f'WARNING: inflow {round(inflows[i], 1)} greater than max flowrate of {round(max(self.geometry["discharge"]), 1)}')
            q_guess = sum([inflows[i], inflows[i - 1], outflows[i - 1]]) / 3
            last_guess = q_guess * 2
            counter = 1
            while abs(last_guess - q_guess) > 0.003:
                counter += 1
                last_guess = q_guess
                q_guess = sum([inflows[i], inflows[i - 1], outflows[i - 1]], q_guess) / 4

                # Interpolate
                log_q_guess = np.log(q_guess)
                b_tmp = np.exp(np.interp(log_q_guess, self.geometry['log_q'], self.geometry['log_width']))
                c_tmp = np.interp(log_q_guess, self.geometry['log_q'], self.geometry['celerity'])

                k_tmp = (self.reach_length / c_tmp) / (60 * 60)
                x_tmp = 0.5 - (q_guess / (2 * c_tmp * b_tmp * self.slope * self.reach_length))
                
                max_c = max(max_c, c_tmp)
                min_c = min(min_c, c_tmp)

                c0 = ((dt / k_tmp) - (2 * x_tmp)) / ((2 * (1 - x_tmp)) + (dt / k_tmp))
                c1 = ((dt / k_tmp) + (2 * x_tmp)) / ((2 * (1 - x_tmp)) + (dt / k_tmp))
                c2 = ((2 * (1 - x_tmp)) - (dt / k_tmp)) / ((2 * (1 - x_tmp)) + (dt / k_tmp))

                q_guess = (c0 * inflows[i + 1]) + (c1 * inflows[i]) + (c2 * outflows[i])
                
            q_guess = max(min(inflows), q_guess)
            outflows.append(q_guess)

        min_travel_time = (self.reach_length / max_c) / (60 * 60)
        if min_travel_time < dt:
            print('dt too large')
            print(f'Minimum travel time is {min_travel_time} hours')
        return outflows
    
class TrapezoidalReach(BaseReach):

    def __init__(self, bottom_width, side_slope, mannings_n, slope, reach_length, max_stage=10, stage_resolution=50):
        self.bottom_width = bottom_width
        self.side_slope = side_slope
        self.mannings_n = mannings_n
        self.slope = slope
        self.reach_length = reach_length
        self.max_stage = max_stage
        self.resolution = stage_resolution

        self.calculate_parameters()
    
    def generate_geometry(self):
        geom = self.geometry
        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)
        geom['top_width'] = self.bottom_width + (geom['stage'] * self.side_slope)
        geom['area'] = (((geom['stage'] * self.side_slope) + (2 * self.bottom_width)) / 2) * geom['stage']
        geom['wetted_perimeter'] = ((((geom['stage'] * self.side_slope) ** 2) + (geom['stage'] ** 2)) ** 0.5) + self.bottom_width
        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']
        geom['mannings_n'] = np.repeat(self.mannings_n, self.resolution)
        geom['dpdy'] = np.repeat(2, self.resolution)
        geom['dpdy'] = (geom['wetted_perimeter'][1:] - geom['wetted_perimeter'][:-1]) / (geom['stage'][1:] - geom['stage'][:-1])
        geom['dpdy'] = np.append(geom['dpdy'], geom['dpdy'][-1])

class CompoundReach(BaseReach):

    def __init__(self, bottom_width, side_slope, bankfull_depth, floodplain_width, mannings_n, slope, reach_length, max_stage=10, stage_resolution=50):
        self.bottom_width = bottom_width
        self.side_slope = side_slope
        self.bankfull_depth = bankfull_depth
        assert floodplain_width >= self.bottom_width + (self.side_slope * self.bankfull_depth), 'floodplain width smaller than channel width at bankfull depth'
        self.floodplain_width = floodplain_width
        self.mannings_n = mannings_n
        self.slope = slope
        self.reach_length = reach_length
        self.max_stage = max_stage
        self.resolution = stage_resolution

        self.calculate_parameters()
    
    def generate_geometry(self):
        geom = self.geometry
        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)
        geom['top_width'] = self.bottom_width + (geom['stage'] * self.side_slope)
        geom['area'] = (((geom['stage'] * self.side_slope) + (2 * self.bottom_width)) / 2) * geom['stage']
        geom['wetted_perimeter'] = ((((geom['stage'] * self.side_slope) ** 2) + (geom['stage'] ** 2)) ** 0.5) + self.bottom_width

        # Add compound channel
        geom['top_width'][geom['stage'] >= self.bankfull_depth] = self.floodplain_width
        area_at_bkf = np.interp(self.bankfull_depth, geom['stage'], geom['area'])
        area_after_bkf = area_at_bkf + (self.floodplain_width * (geom['stage'] - self.bankfull_depth))
        geom['area'][geom['stage'] > self.bankfull_depth] = area_after_bkf[geom['stage'] > self.bankfull_depth]
        p_at_bkf = np.interp(self.bankfull_depth, geom['stage'], geom['wetted_perimeter'])
        tw_at_bkf = np.interp(self.bankfull_depth, geom['stage'], geom['top_width'])
        p_after_bkf = p_at_bkf + (2 * (geom['stage'] - self.bankfull_depth)) + (self.floodplain_width - tw_at_bkf)
        geom['wetted_perimeter'][geom['stage'] > self.bankfull_depth] = p_after_bkf[geom['stage'] > self.bankfull_depth]

        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']
        geom['mannings_n'] = np.repeat(self.mannings_n, self.resolution)
        geom['dpdy'] = np.repeat(2, self.resolution)
        geom['dpdy'] = (geom['wetted_perimeter'][1:] - geom['wetted_perimeter'][:-1]) / (geom['stage'][1:] - geom['stage'][:-1])
        geom['dpdy'] = np.append(geom['dpdy'], geom['dpdy'][-1])


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

