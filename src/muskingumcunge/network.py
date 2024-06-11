from collections import defaultdict
import pandas as pd
import numpy as np


class Network:

    def __init__(self, edge_dict):
        self.edge_dict = edge_dict  # key=u/s, value=d/s
        self.chlid_dict = defaultdict(list)
        for u, v in edge_dict.items():
            self.chlid_dict[v].append(u)
        self.headwaters = list(set(edge_dict.keys()).difference(set(edge_dict.values())))
        roots = set(edge_dict.values()).difference(set(edge_dict.keys()))
        if len(roots) > 1:
            raise ValueError(f'Multiple roots detected: {roots}')
        self.root = set(edge_dict.values()).difference(set(edge_dict.keys())).pop()
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
        self.post_order = self.post_order[:-1]

        self.reach_dict = None
        self.forcing_df = None
        self.channel_outflows = {k: None for k in self.edge_dict.keys()}
        self.out_df = None

    def export_post_order(self, path):
        out_df = pd.DataFrame(self.post_order, columns=['comid'])
        out_df['order'] = range(1, len(out_df) + 1)
        out_df.to_csv(path, index=False)

    def load_reaches(self, reach_dict):
        self.reach_dict = reach_dict

    def load_forcings(self, forcing_df):
        self.forcing_df = forcing_df
        self.forcing_df = self.forcing_df[self.forcing_df.columns.intersection(self.post_order)].copy()
        for n in self.headwaters:
            self.channel_outflows[n] = np.zeros(self.forcing_df.shape[0])
    
    def load_headwater_forcings(self, headwater_forcings):
        for n in self.headwaters:
            self.channel_outflows[n] = headwater_forcings[n].to_numpy()

    def run_event(self):
        # calculate total inflow
        volume_in = self.forcing_df.sum().sum() + np.sum([self.channel_outflows[c].sum() for c in self.headwaters])
        dt = (self.forcing_df.index[1] - self.forcing_df.index[0]).seconds / 3600
        for node in self.post_order:
            reach = self.reach_dict[node]
            if node in self.headwaters:
                us_hydro = self.channel_outflows[node]
            else:
                children = self.chlid_dict[node]
                us_hydro = [self.channel_outflows[c] for c in children]
                us_hydro = np.sum(us_hydro, axis=0)
            routed = reach.route_hydrograph(us_hydro, dt)
            # routed = us_hydro
            outflow = routed + self.forcing_df[node]
            self.channel_outflows[node] = outflow
        
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


