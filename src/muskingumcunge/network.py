from collections import defaultdict
import pandas as pd
import numpy as np


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

    def load_lake_forcings(self, lake_forcings):
        lake_forcings = lake_forcings.fillna(0)
        for n in lake_forcings.columns:
            self.channel_outflows[n] = lake_forcings[n].to_numpy()
            self.headwaters.append(n)

    def run_event(self, optimize_dx=True, conserve_mass=False, lat_addition='middle'):
        # calculate total inflow
        volume_in = self.forcing_df.sum().sum() + np.sum([self.channel_outflows[c].sum() for c in self.headwaters])
        dt = (self.forcing_df.index[1] - self.forcing_df.index[0]).seconds / 3600
        counter = 1
        for node in self.post_order:
            if counter % 100 == 0:
                print(f"{counter} / {len(self.post_order)}")
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
                    routed = reach.route_hydrograph(routed, dt, lateral=l)
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
        
        # calculate total outflow
        us_root = self.chlid_dict[self.root]
        us_hydro = [self.channel_outflows[c] for c in us_root]
        self.channel_outflows[self.root] = np.sum(us_hydro, axis=0)
        volume_out = self.channel_outflows[self.root].sum()
        print(f"Volume conservation = {round(100 *volume_out / volume_in, 2)} %")

        # Post-processing
        out_df = pd.DataFrame(self.channel_outflows)
        out_df['dt'] = self.forcing_df.index
        self.out_df = out_df.set_index('dt')


