import os
import json


def create_project(data_path, run_path, geom_type):
    template = {
        "base_dir": data_path,
        "meta_path": os.path.join(data_path, "reach_meta.csv"),
        "lateral_path": os.path.join(data_path, "laterals.csv"),
        "inflow_path": os.path.join(data_path, "inflows.csv"),
        "init_conditions_path": os.path.join(data_path, "init_conditions.csv"),
        "geometry_dir": "/netfiles/ciroh/floodplainsData/runs/NWM/geometry",
        "geometry_source": geom_type.upper(),
        "dt": "5m",
        "optimize_dx": False,
        "conserve_mass": False,
        "short_ts": True,
        "lat_addition": "top",
        "id_field": "comid",
        "to_id_field": "ds_comid",
        "mode": 'basin',
        "out_dir": os.path.join(data_path, f'{geom_type.lower()}_outputs'),
    }
    os.makedirs(template['out_dir'], exist_ok=True)
    with open(run_path, 'w') as f:
        json.dump(template, f, indent=4)

base_dir = r"/netfiles/ciroh/floodplainsData/retrospective"
runs_path = os.path.join(base_dir, 'basin_run_files')
catchments = ['lamoille', 'lewis', 'mettawee', 'missisquoi', 'otter', 'winooski']
for catchment in catchments:
    tmp_dir = os.path.join(base_dir, catchment)
    folders = [f for f in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, f))]
    for f in folders:
        data_path = os.path.join(tmp_dir, f)
        geom_type = 'nwm'
        run_path = os.path.join(runs_path, f'{catchment}_{f}_{geom_type}.json')
        create_project(data_path, run_path, geom_type)
        geom_type = 'hand'
        run_path = os.path.join(runs_path, f'{catchment}_{f}_{geom_type}.json')
        create_project(data_path, run_path, geom_type)
