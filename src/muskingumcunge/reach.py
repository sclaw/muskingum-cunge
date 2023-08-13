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

        self.calculate_parameters()


    def calculate_parameters(self):
        self.generate_geometry()
        self.generate_rating_curve()
        self.generate_muskingum_params()

    def generate_geometry(self):
        geom = self.geometry
        geom['stage'] = np.linspace(0, self.max_stage, self.resolution)
        geom['top_width'] = np.repeat(self.width, self.resolution)
        geom['area'] = geom['stage'] * geom['top_width']
        geom['wetted_perimeter'] = self.width + (2 * geom['stage'])
        geom['hydraulic_radius'] = geom['area'] / geom['wetted_perimeter']
        geom['mannings_n'] = np.repeat(self.mannings_n, self.resolution)
        geom['dpdy'] = np.repeat(2, self.resolution)

    def generate_rating_curve(self):
        self.rating_curve['stage'] = self.geometry['stage']
        self.rating_curve['discharge'] = (1 / self.geometry['mannings_n']) * self.geometry['area'] * (self.geometry['hydraulic_radius'] ** (2 / 3)) * (self.slope ** 0.5)

    def generate_muskingum_params(self):
        self.muskingum_params['stage'] = self.geometry['stage']
        k_prime = (5 / 3) - ((2 / 3) * (self.geometry['area'] / (self.geometry['top_width'] * self.geometry['wetted_perimeter'])) * self.geometry['dpdy'])
        tmp_area = self.geometry['area']
        tmp_area[0] = tmp_area[1]  # Approximation to make this computationally stable at stage=0
        c = (self.rating_curve['discharge'] / tmp_area) * k_prime
        c[0] = c[1]  # Approximation to make this computationally stable at stage=0
        self.muskingum_params['k'] = self.reach_length / c
        self.muskingum_params['k'] /= (60 * 60)  # Convert seconds to hours
        self.muskingum_params['x'] = (1 / 2) - (self.rating_curve['discharge'] / (2 * c * self.geometry['top_width'] * self.slope * self.reach_length))
        self.muskingum_params['x'][self.muskingum_params['x'] < 0] = 0

    def route_hydrograph(self, inflows, dt):
        outflows = list()
        outflows.append((inflows[0]))
        for i in range(len(inflows) - 1):
            stage = np.interp(inflows[i], self.rating_curve['discharge'], self.rating_curve['stage'])
            if stage > self.max_stage:
                print(f'WARNING: stage {round(stage, 1)} greater than max stage of {round(self.max_stage, 1)}')
            k_tmp = np.interp(stage, self.muskingum_params['stage'], self.muskingum_params['k'])
            x_tmp = np.interp(stage, self.muskingum_params['stage'], self.muskingum_params['x'])

            c0 = ((dt / k_tmp) - (2 * x_tmp)) / ((2 * (1 - x_tmp)) + (dt / k_tmp))
            c1 = ((dt / k_tmp) + (2 * x_tmp)) / ((2 * (1 - x_tmp)) + (dt / k_tmp))
            c2 = ((2 * (1 - x_tmp)) - (dt / k_tmp)) / ((2 * (1 - x_tmp)) + (dt / k_tmp))

            tmp_out = (c0 * inflows[i + 1]) + (c1 * inflows[i]) + (c2 * outflows[i])
            tmp_out = max(0, tmp_out)
            outflows.append(tmp_out)

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

