import os
from muskingumcunge.controller import handler

run_dir = r'/netfiles/ciroh/floodplainsData/retrospective/basin_run_files'
all_paths = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.json')]

for path in all_paths:
    handler(path)