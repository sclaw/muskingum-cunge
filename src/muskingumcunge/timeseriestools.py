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