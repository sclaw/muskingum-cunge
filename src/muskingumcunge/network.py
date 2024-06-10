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

    def export_post_order(self, path):
        out_df = pd.DataFrame(self.post_order, columns=['comid'])
        out_df['order'] = range(1, len(out_df) + 1)
        out_df.to_csv(path, index=False)

    def load_reaches(self, reach_dict):
        self.reach_dict = reach_dict

    def load_forcings(self, forcing_path):
        self.forcing_df = pd.read_csv(forcing_path, index_col=0, parse_dates=True)
        for n in self.headwaters:
            self.channel_outflows[n] = np.zeros(self.forcing_df.shape[0])

    def run_event(self):
        for node in self.post_order:
            reach = self.reach_dict[node]
            children = self.chlid_dict[node]
            us_hydro = [self.channel_outflows[c] for c in children]
            us_hydro = np.sum(us_hydro, axis=0)
            # routed = reach.route_hydrograph(us_hydro)
            routed = us_hydro
            outflow = routed + self.forcing_df[node]
            self.channel_outflows[node] = outflow


