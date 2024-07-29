import numpy as np
import matplotlib.pyplot as plt
from muskingumcunge.reach import TrapezoidalReach
from muskingumcunge.timeseriestools import parabolic_hydrograph_generator

# Generate hydrograph
start = 0  # hours
stop = 24  # hours
dt = 0.1  # hours
timestamps = np.arange(start, stop, dt)
peak = 1000
duration = 10
inflows = parabolic_hydrograph_generator(timestamps, peak, duration)

# Build reach
reach = TrapezoidalReach(100, 100, 0.1, 0.001, 2000)

# Route flows
outflows = reach.route_hydrograph(inflows, 360, solver='fread')

# Plot results
fig, ax = plt.subplots()
ax.plot(timestamps, inflows, label='inflow')
ax.plot(timestamps, outflows, label='outflow')
ax.set_xlabel('Time(hrs)')
ax.set_ylabel('Flowrate(cms)')
plt.legend()
plt.tight_layout()
plt.savefig('simple_routing.png', dpi=300)
# plt.show()