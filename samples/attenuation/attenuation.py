import numpy as np
import matplotlib.pyplot as plt
from muskingumcunge.reach import BaseReach
from muskingumcunge.timeseriestools import parabolic_hydrograph_generator


def slope_vs_attenuation():
    # Generate hydrographs
    simulations = 25
    max_flow = 1000
    max_duration = 10
    np.random.seed(5)
    timestamps = np.linspace(0, 20, 200)
    hydrographs = list()
    for s in range(simulations):
        peak = max_flow * np.random.random()
        duration = max_duration * np.random.random()
        if duration < 0.3:
            duration = 0.3
        inflows = parabolic_hydrograph_generator(timestamps, peak, duration)
        hydrographs.append(inflows)

    # run comparison
    fig, ax = plt.subplots()
    dt = 0.1
    for n in [0.1, 0.05, 0.01]:
        slopes = np.logspace(-1, -4, 20)
        attenuations = list()
        slope_log = list()
        for slope in slopes:
            reach = BaseReach(100, n, slope, 1000)
            for hydrograph in hydrographs:
                outflows = reach.route_hydrograph(hydrograph, 0.1)
                attenuation = (np.max(hydrograph) - np.max(outflows)) / np.max(hydrograph)
                # attenuation = (np.max(hydrograph) - np.max(outflows))
                attenuations.append(attenuation)
                slope_log.append(slope)
        ax.scatter(slope_log, attenuations, label=n, alpha=0.6, s=15)
    ax.set_ylabel('Percent Attenuation of Inflow Peak')
    ax.set_xlabel('Slope (m/m)')
    ax.set_xscale('log')
    plt.legend(title='mannings n')
    plt.show()

slope_vs_attenuation()