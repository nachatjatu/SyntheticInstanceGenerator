import pickle
import os
import numpy as np
import yaml
from scipy.stats import gaussian_kde
from pyproj import Transformer
import osmnx as ox
from scipy.special import logsumexp, gammaln
from scipy.interpolate import interp1d
from names_generator import generate_name
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# GLOBAL CONSTANTS
INDO_CRS = "EPSG:23867"             # Indonesian Projected CRS
LL_CRS = "EPSG:4326"                # WGS84 Lat/Lon
MIN_CAPACITY, MAX_CAPACITY = 2, 9
RES = 250                           # Grid resolution in meters
MAX_DIST = 63000                    # Maximum sampling distance

class InstanceGenerator:
    def __init__(self, farmers_df, ints_df, 
                 graph_path="../FactoredPlatformSolver/data/graph_0-14960_00.pickle"):
        # CRS transformers
        self.xy_to_ll = Transformer.from_crs(INDO_CRS, LL_CRS, always_xy=True)
        self.ll_to_xy = Transformer.from_crs(LL_CRS, INDO_CRS, always_xy=True)

        # load data
        self.farmers_df = farmers_df
        self.ints_df = ints_df
        self.G_proj, self.bbox_m = self._init_graph(graph_path)
        
        # create spacial grid
        x_ax = np.arange(self.bbox_m[0], self.bbox_m[2], RES)
        y_ax = np.arange(self.bbox_m[1], self.bbox_m[3], RES)
        gx, gy = np.meshgrid(x_ax, y_ax, indexing='ij')
        self.grid_coords = np.vstack([gx.ravel(), gy.ravel()])

        # KDEs
        self.int_spatial_kde = self._init_int_kde()
        self.farmer_spatial_kde = self._init_farmer_kde()
        
        # precompute farmer spatial priors on grid
        p_spatial = self.farmer_spatial_kde.evaluate(self.grid_coords)
        self.p_spatial = p_spatial / (p_spatial.sum() + 1e-20)

        # initialize distance KDE
        self.gamma_lookups = self._init_gamma_kdes()

        # cache historical statistics
        self.hist_quantities = (self.farmers_df.groupby('int_id')['quantity']
                                .apply(list).to_dict())
        
        counts_df = (self.farmers_df.groupby(['int_id', 'date'])
                    .size().reset_index(name='count'))
        self.hist_n_farmers = counts_df.groupby('int_id')['count'].apply(list).to_dict()

        self.ints = {}
        self.mills = [{'id': 'SKIP', 'location': [-0.682643, 102.501522]}]

        # sigma values for clustering intensity (precomputed)
        self.sigmas = {
            'Dodi Lesmana': 2500, 'Purnomo': 2500, 'Isna': 4000,
            'Agus Wibowo': 12000, 'Nurmala': 13500, 'yaya suhayat': 12500,
            'Agus Yasir': 6500, 'Ngatinu': 5500, 'Samsuri': 13000,
            'Riki Mandala': 30500, 'Syafrial': 9500, 'Yaman Saragih': 3500,
            'Khairul': 4500, 'Ndoharo': 2500
        }


    def _init_graph(self, graph_path):
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        G_proj = ox.project_graph(G, to_crs=INDO_CRS)
        nodes_proj, _ = ox.graph_to_gdfs(G_proj)
        return G_proj, nodes_proj.total_bounds
    

    def _init_int_kde(self, int_bw=0.2):
        coords = self.ints_df.drop_duplicates(['int_id'])[['int_x', 'int_y']].T
        return gaussian_kde(coords, bw_method=int_bw)


    def _init_farmer_kde(self, farmer_bw=0.2):
        coords = self.farmers_df.drop_duplicates(['farmer_x', 'farmer_y'])[['farmer_x', 'farmer_y']].T
        return gaussian_kde(coords, bw_method=farmer_bw)
    
    
    def _init_gamma_kdes(self):
        int_to_dists = (self.farmers_df
                        .drop_duplicates(['int_id', 'farmer_x', 'farmer_y'])
                        .groupby('int_id')['distance']
                        .apply(np.array).to_dict())
        
        lookups = {}
        x_eval = np.linspace(0, MAX_DIST + 10000, 2000)
        
        for i_id, dists in int_to_dists.items():
            n = len(dists)
            h = 0.1 * np.mean(dists) + 1e-6
            shape = (dists / h)
            
            pdf_values = np.zeros_like(x_eval)
            for i in range(n):
                s, scale = shape[i], h
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Gamma log-PDF: (s-1)*log(x) - x/scale - (log(gamma(s)) + s*log(scale))
                    log_pdf = (
                        (s - 1) * np.log(x_eval + 1e-10) 
                        - (x_eval / scale) 
                        - (gammaln(s + 1e-10) + s * np.log(scale))
                    )
                pdf_values += np.exp(log_pdf)
            
            pdf_values /= n
            lookups[i_id] = interp1d(x_eval, pdf_values, fill_value=(0,0), bounds_error=False)
            
        return lookups
    
    
    def _init_sigmas(self):
        self.sigmas = {int_id: self.find_mle_sigma_adaptive(int_id) for int_id in self.ints}


    def gen_ints(self, n_ints, seed):
        ss = np.random.SeedSequence(seed)

        # One independent RNG stream per intermediary
        child_seeds = ss.spawn(n_ints)
        rngs = [np.random.default_rng(child_seed) for child_seed in child_seeds]

        ints = {}
        names = set()
        types = list(self.gamma_lookups.keys())

        for i in range(n_ints):
            rng = rngs[i]

            while True:
                int_id = generate_name(seed=int(rng.integers(0, 2**32 - 1)))

                if int_id not in names:
                    names.add(int_id)
                    break

            int_type = rng.choice(types)
            
            # rejection sampling within bounding box
            while True:
                sample = self.int_spatial_kde.resample(1, seed=rng).flatten()
                if (self.bbox_m[0] <= sample[0] <= self.bbox_m[2] and 
                    self.bbox_m[1] <= sample[1] <= self.bbox_m[3]):
                    int_xy = sample
                    break
            
            lon, lat = self.xy_to_ll.transform(int_xy[0], int_xy[1])
            ints[int_id] = {'xy': int_xy, 'll': (lat, lon), 'type': int_type}
        self.ints = ints


    def gen_farmers(self, int_xy, int_type, n_farmers, rng, sigma=500):
        # precompute distances from grid to int
        dist_lookup = self.gamma_lookups[int_type]
        grid_points = self.grid_coords.T 
        dists = np.linalg.norm(grid_points - int_xy, axis=1)
        
        # compute base log probabilities
        p_dist_raw = dist_lookup(dists)
        log_p_base = np.log(p_dist_raw + 1e-20) + np.log(self.p_spatial + 1e-20)
        log_p_base -= logsumexp(log_p_base)

        locs = []
        sigma_sq_2 = 2 * (sigma ** 2)
        acc_exp_kernels = np.zeros(len(grid_points))

        for k in range(n_farmers):
            if k == 0:
                log_p_cond = log_p_base
            else:
                # bayesian update: clustering influence (add in log space)
                log_local_factor = np.log(acc_exp_kernels + 1e-20) - np.log(k)
                log_p_cond = log_p_base + log_local_factor
                log_p_cond -= logsumexp(log_p_cond)

            p_sampling = np.exp(log_p_cond)
            
            # numerical stability fallback
            if np.isnan(p_sampling).any() or p_sampling.sum() == 0:
                p_sampling = self.p_spatial

            # sample farmer locations
            idx = rng.choice(len(p_sampling), p=p_sampling/p_sampling.sum())
            sampled_xy = self.grid_coords[:, idx]
            locs.append(sampled_xy)
            
            # update kernel for next farmer in sequence
            new_dist_sq = np.sum((grid_points - sampled_xy)**2, axis=1)
            acc_exp_kernels += np.exp(-new_dist_sq / sigma_sq_2)

        return np.array(locs)
    

    def gen_instance(self, instance_id, seed, write=False, plot=False, scale_factor=1.0):

        farmers, ints = [], []

        ss = np.random.SeedSequence(seed)
        int_ids = list(self.ints.keys())

        # One independent RNG stream per intermediary
        child_seeds = ss.spawn(len(int_ids))
        rngs = {
            int_id: np.random.default_rng(child_seed)
            for int_id, child_seed in zip(int_ids, child_seeds)
        }

        for int_id in int_ids:

            int_data = self.ints[int_id]

            # get int's seed
            rng = rngs[int_id]

            # sample intermediary type and location
            int_type, int_xy, int_ll = int_data['type'], int_data['xy'], int_data['ll']

            # sample number of farmers in intermediary's network
            
            n_farmers = rng.choice(self.hist_n_farmers[int_type])
            raw_n = n_farmers * scale_factor
            n_farmers = int(np.floor(raw_n) + (rng.random() < (raw_n % 1))) # Bernoulli using scale_factor
            
            if n_farmers > 0:
                # generate farmer locations
                sigma = self.sigmas.get(int_type, 5000)
                farmer_xys = self.gen_farmers(int_xy, int_type, n_farmers, rng, sigma=sigma)

                # generate farmer quantities
                qs = []
                for _ in range(n_farmers):
                    q = rng.choice(self.hist_quantities[int_type], replace=True)
                    qs.append(q)
                qs = np.array(qs)

                # rescale quantities to fit intermediary capacity constraints
                total_q = qs.sum()
                if total_q >= MAX_CAPACITY:
                    sf = (MAX_CAPACITY - 0.01) / total_q
                elif total_q < MIN_CAPACITY:
                    sf = MIN_CAPACITY / total_q
                else:
                    sf = 1
                qs_scaled = qs * sf

                # format and append farmers
                routes = []
                for f in range(n_farmers):
                    f_id = f'{int_id}_f{f}'
                    f_lon, f_lat = self.xy_to_ll.transform(farmer_xys[f][0], farmer_xys[f][1])
                    
                    farmers.append({
                        'id': f_id, 
                        'location': [f_lat, f_lon],
                        'quantity': float(qs_scaled[f]),
                        'intermediary': int_id
                    })
                    routes.append(f_id)

                ints.append({
                    'id': int_id, 
                    'capacity': MAX_CAPACITY, 
                    'location': list(int_ll), 
                    'routes': [routes]
                })
        
        # write and plot (if desired)
        instance = {'instance_id': instance_id,
                    'farmers': farmers, 
                    'intermediaries': ints, 
                    'mills': self.mills}

        if write:
            os.makedirs("data/instances", exist_ok=True)
            with open(f'data/instances/{instance_id}.yaml', 'w') as file:
                yaml.dump(instance, file, default_flow_style=False)   

        if plot:
            self.plot_instance(instance) 

        return instance
    

    def plot_instance(self, instance_data):
        plt.figure(figsize=(14, 11))
        
        # 1. Plot the Farmer KDE Background
        x_coords = np.unique(self.grid_coords[0])
        y_coords = np.unique(self.grid_coords[1])
        Z = self.p_spatial.reshape(len(x_coords), len(y_coords))
        
        plt.imshow(
            Z.T, 
            origin='lower', 
            extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
            cmap='magma', # 'magma' or 'viridis' provide excellent contrast for density
            aspect='equal'
        )
        
        # 2. Plot the Road Map (from OSMnx G_proj)
        # Extract edges from the projected graph
        lines = []
        for u, v, data in self.G_proj.edges(data=True):
            if 'geometry' in data:
                # Plot the line-string geometry
                xs, ys = data['geometry'].xy
                lines.append(list(zip(xs, ys)))
            else:
                # Fallback to straight lines between nodes if geometry is missing
                u_node = self.G_proj.nodes[u]
                v_node = self.G_proj.nodes[v]
                lines.append([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
                
        lc = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.4, zorder=1)
        plt.gca().add_collection(lc)

        # Plot Mill
        for mill in instance_data['mills']:
            x, y = self.ll_to_xy.transform(mill['location'][1], mill['location'][0])
            plt.scatter(x, y, c='white', marker='*', s=400, label="Mill", zorder=10)

        # Plot Clusters
        colors = plt.get_cmap('tab10', len(instance_data['intermediaries']))
        farmer_lookup = {f['id']: f for f in instance_data['farmers']}

        for i, intermediary in enumerate(instance_data['intermediaries']):
            color = colors(i)
            ix, iy = self.ll_to_xy.transform(intermediary['location'][1], intermediary['location'][0])
            
            # Intermediary marker
            plt.scatter(ix, iy, color=color, marker='s', s=120, edgecolors='k', zorder=9, label=intermediary['id'])

            # Farmer markers and lines
            for f_id in intermediary['routes'][0]:
                farmer = farmer_lookup[f_id]
                fx, fy = self.ll_to_xy.transform(farmer['location'][1], farmer['location'][0])
                
                plt.scatter(fx, fy, color=color, s=40, edgecolors='white', zorder=8)
                plt.plot([ix, fx], [iy, fy], color=color, lw=1.5, alpha=0.6, zorder=7)

        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.tight_layout()
        plt.show()

    
    def find_mle_sigma_adaptive(self, int_type, start_sigma=2500, step=500):
            best_sigma = start_sigma
            best_ll = -np.inf
            current_sigma = start_sigma
            
            # Precompute base prior for the specific intermediary location
            int_data = self.ints_df[self.ints_df['int_id'] == int_type].iloc[0]
            int_xy = np.array([int_data['int_x'], int_data['int_y']])
            dists = np.linalg.norm(self.grid_coords.T - int_xy, axis=1)
            
            log_p_base = np.log(self.gamma_lookups[int_type](dists) + 1e-20) + np.log(self.p_spatial + 1e-20)
            
            # Pre-map historical farmers to grid indices
            daily_groups = self.farmers_df[self.farmers_df['int_id'] == int_type].groupby('date')
            historical_indices = []
            for _, group in daily_groups:
                coords = group[['farmer_x', 'farmer_y']].values
                indices = [np.argmin(np.sum((self.grid_coords.T - c) ** 2, axis=1)) for c in coords]
                historical_indices.append(indices)

            # Hill-climbing optimization
            while True:
                total_ll = 0
                sigma_sq_2 = 2 * (current_sigma ** 2)
                
                for f_indices in historical_indices:
                    acc_exp_kernels = np.zeros(len(self.grid_coords[0]))
                    
                    for k, target_idx in enumerate(f_indices):
                        if k == 0:
                            log_p_cond = log_p_base
                        else:
                            log_local_factor = np.log(acc_exp_kernels + 1e-20) - np.log(k)
                            log_p_cond = log_p_base + log_local_factor
                        
                        log_p_cond -= logsumexp(log_p_cond)
                        total_ll += log_p_cond[target_idx]
                        
                        # Update kernel density for sequential evaluation
                        sampled_xy = self.grid_coords[:, target_idx]
                        dist_sq = np.sum((self.grid_coords.T - sampled_xy)**2, axis=1)
                        acc_exp_kernels += np.exp(-dist_sq / sigma_sq_2)
                
                if total_ll > best_ll:
                    best_ll = total_ll
                    best_sigma = current_sigma
                    current_sigma += step
                else:
                    break # Likelihood began decreasing
                    
            return best_sigma