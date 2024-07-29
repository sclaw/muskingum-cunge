import os
import json
from muskingumcunge.controller import handler
import numpy as np

run_dir = r'/netfiles/ciroh/floodplainsData/retrospective/basin_run_files'
all_paths = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.json')]

for path in all_paths:
    with open(path, 'r') as f:
        dict = json.load(f)

    handler(path)