from collections import defaultdict
import pandas as pd
import numpy as np
import time


class Network:

    def __init__(self, edge_dict):
        self.edge_dict = edge_dict  # key=u/s, value=d/s
        self.chlid_dict = defaultdict(list)
        for u, v in edge_dict.items():
            self.chlid_dict[v].append(u)

        roots = set(edge_dict.values()).difference(set(edge_dict.keys()))
        if len(roots) > 1:
            raise ValueError(f'Multiple roots detected: {roots}')
        self.root = set(edge_dict.values()).difference(set(edge_dict.keys())).pop()
        self.headwaters = list()
        self.post_order = list()
        self.calculate_post_order()
        self.reach_dict = None
        self.forcing_df = None
        self.channel_outflows = {k: None for k in self.edge_dict.keys()}
        self.channel_outflows_stage = {k: None for k in self.edge_dict.keys()}
        self.init_outflows = {k: None for k in self.edge_dict.keys()}
        self.out_df = None

    def calculate_post_order(self):
        self.headwaters = list()
        self.post_order = list()
        q = [self.root]
        working = True
        while working:
            cur = q.pop()
            children = self.chlid_dict[cur]
            if len(children) == 0:
                self.post_order.append(cur)
                self.headwaters.append(cur)
            elif all([c in self.post_order for c in children]):
                self.post_order.append(cur)
            else:
                q.append(cur)
                q.extend(children)
            if len(q) == 0:
                working = False
        self.post_order = self.post_order[:-1]  # remove root

    def export_post_order(self, path):
        out_df = pd.DataFrame(self.post_order, columns=['comid'])
        out_df['order'] = range(1, len(out_df) + 1)
        out_df.to_csv(path, index=False)

    def load_reaches(self, reach_dict):
        self.reach_dict = reach_dict  # Todo:  add informational note about whether all reaches in post order have been loaded

    def load_forcings(self, forcing_df):
        self.forcing_df = forcing_df
        self.forcing_df = self.forcing_df[self.forcing_df.columns.intersection(self.post_order)].copy()
    
    def load_headwater_forcings(self, headwater_forcings):
        headwater_forcings = headwater_forcings.fillna(0)
        missing = list(set(self.headwaters).difference(set(headwater_forcings.columns)))
        print(f"Missing headwater forcings for: {missing}")
        for n in headwater_forcings.columns:
            self.channel_outflows[n] = headwater_forcings[n].to_numpy()
            self.headwaters.append(n)
        for n in missing:
            self.channel_outflows[n] = np.zeros(len(self.forcing_df))

    def load_initial_conditions(self, initial_conditions):
        # loads initial outflows for each reach
        self.init_outflows = initial_conditions

    def run_event(self, optimize_dx=True, conserve_mass=False, lat_addition='middle', short_ts=False):
        # calculate total inflow
        t_start = time.perf_counter()
        dt = (self.forcing_df.index[1] - self.forcing_df.index[0]).seconds
        counter = 0
        pct5 = int(len(self.post_order) / 20) + 1
        for node in self.post_order:
            counter += 1
            if counter % pct5 == 0:
                pct_done = int(100 * counter / len(self.post_order))
                print(f'{pct_done}% done')
                
            reach = self.reach_dict[node]
            if node in self.headwaters:
                continue
            else:
                children = self.chlid_dict[node]
                us_hydro = [self.channel_outflows[c] for c in children]
                us_hydro = np.sum(us_hydro, axis=0)
            
            laterals = self.forcing_df[node].to_numpy()
            if lat_addition == 'top':
                us_hydro = us_hydro + laterals
    
            if optimize_dx:
                dx, subreaches = reach.optimize_route_params(us_hydro, dt)
                reach.reach_length = dx
            else:
                subreaches = 1

            routed = us_hydro
            for i in range(subreaches):
                try:
                    if lat_addition == 'middle':
                        l = laterals
                    else:
                        l = None
                    if i == 0 and node in self.init_outflows:
                        init_out = self.init_outflows[node]
                    else:
                        init_out = None
                    routed = reach.route_hydrograph(routed, dt, lateral=l, initial_outflow=init_out, short_ts=short_ts, solver='wrf-c')
                except AssertionError as e:
                    print(f"Error routing {node}: {e}")
            
            if lat_addition == 'bottom':
                routed = routed + laterals

            # enforce mass conservation
            if conserve_mass:
                in_flow = us_hydro.sum() + laterals.sum()
                out_flow = routed.sum()
                routed = routed * (in_flow / out_flow)
            self.channel_outflows[node] = routed
            self.channel_outflows_stage[node] = np.interp(routed, reach.geometry['discharge'], reach.geometry['stage'])
        
        print(f'Routing complete in {round(time.perf_counter() - t_start, 1)} seconds')

        # Post-processing
        out_df = pd.DataFrame(self.channel_outflows)
        out_df['dt'] = self.forcing_df.index
        self.out_df = out_df.set_index('dt')
        stage_df = pd.DataFrame(self.channel_outflows_stage)
        stage_df['dt'] = self.forcing_df.index
        self.stage_df = stage_df.set_index('dt')


