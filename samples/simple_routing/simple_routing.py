import numpy as np
import matplotlib.pyplot as plt
from muskingumcunge.reach import TrapezoidalReach
from muskingumcunge.timeseriestools import parabolic_hydrograph_generator

# Generate hydrograph
timestamps = np.linspace(0, 20, 200)
peak = 1000
duration = 10
inflows = parabolic_hydrograph_generator(timestamps, peak, duration)

# Build reach
reach = TrapezoidalReach(100, 100, 0.1, 0.001, 1000)

# Route flows
outflows = reach.route_hydrograph(inflows, 0.1)

# Plot results
fig, ax = plt.subplots()
ax.plot(inflows, label='inflow')
ax.plot(outflows, label='outflow')
plt.show()