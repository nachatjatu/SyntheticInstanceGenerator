import pickle
import random
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from pyproj import Transformer
import osmnx as ox
from scipy.special import logsumexp, gammaln
from scipy.interpolate import interp1d

# GLOBAL CONSTANTS
INDO_CRS = "EPSG:23867"
LL_CRS = "EPSG:4326"
MIN_CAPACITY, MAX_CAPACITY = 2, 9
RES = 250  
MAX_DIST = 63000

class InstanceGenerator:
    """
    Generates synthetic farmer-intermediary instances using a hybrid 
    Bayesian prior (Gamma + Spatial) with local sequential clustering.
    """
    def __init__(self, farmers_df, ints_df, 
                 graph_path="../FactoredPlatformSolver/data/graph_0-14960_00.pickle"):
        
        # CRS transformers: always_xy=True ensures (Lon, Lat) order
        self.xy_to_ll = Transformer.from_crs(INDO_CRS, LL_CRS, always_xy=True)
        self.ll_to_xy = Transformer.from_crs(LL_CRS, INDO_CRS, always_xy=True)

        self.farmers_df = farmers_df
        self.ints_df = ints_df
        self.G_proj, self.bbox_m = self._load_graph(graph_path)
        
        # 1. Setup Grid
        x_ax = np.arange(self.bbox_m[0], self.bbox_m[2], RES)
        y_ax = np.arange(self.bbox_m[1], self.bbox_m[3], RES)
        gx, gy = np.meshgrid(x_ax, y_ax, indexing='ij')
        self.grid_coords = np.vstack([gx.ravel(), gy.ravel()])

        # 2. Setup Spatial Priors
        self.int_spatial_kde = self._init_int_kde()
        self.farmer_spatial_kde = self._init_farmer_kde()
        
        # Precompute global farmer spatial prior on the grid
        p_spatial = self.farmer_spatial_kde.evaluate(self.grid_coords)
        self.p_spatial = p_spatial / (p_spatial.sum() + 1e-20)

        # 3. Setup Distance Priors (Gamma)
        self.gamma_lookups = self._init_gamma_kdes()

        # 4. Cache Historical Stats
        self.hist_quantities = (self.farmers_df.groupby('int_id')['quantity']
                                .apply(list).to_dict())
        
        counts_df = (self.farmers_df.groupby(['int_id', 'date'])
                    .size().reset_index(name='count'))
        self.hist_n_farmers = counts_df.groupby('int_id')['count'].apply(list).to_dict()

        self.ints = {}
        self.mills = [{'id': 'SKIP', 'location': (-0.682643, 102.501522)}]

    def _load_graph(self, graph_path):
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        G_proj = ox.project_graph(G, to_crs=INDO_CRS)
        nodes_proj, _ = ox.graph_to_gdfs(G_proj)
        return G_proj, nodes_proj.total_bounds

    def _init_int_kde(self):
        coords = self.ints_df.drop_duplicates(['int_id'])[['int_x', 'int_y']].T
        return gaussian_kde(coords, bw_method='scott')

    def _init_farmer_kde(self, farmer_bw=0.2):
        coords = self.farmers_df.drop_duplicates(['farmer_x', 'farmer_y'])[['farmer_x', 'farmer_y']].T
        return gaussian_kde(coords, bw_method=farmer_bw)

    def _init_gamma_kdes(self):
        """Builds lookup tables for the Gamma Mixture distance model."""
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
                    # Gamma log-PDF
                    log_pdf = (
                        (s - 1) * np.log(x_eval + 1e-10) 
                        - (x_eval / scale) 
                        - (gammaln(s + 1e-10) + s * np.log(scale))
                    )
                pdf_values += np.exp(log_pdf)
            
            pdf_values /= n
            lookups[i_id] = interp1d(x_eval, pdf_values, fill_value=(0,0), bounds_error=False)
            
        return lookups

    def gen_ints(self, n_ints):
        """Generates intermediary centers within the graph bounding box."""
        ints = {}
        types = list(self.gamma_lookups.keys())
        for i in range(n_ints):
            int_id = f'int_{i}'
            int_type = random.choice(types)
            while True:
                sample = self.int_spatial_kde.resample(1).flatten()
                if (self.bbox_m[0] <= sample[0] <= self.bbox_m[2] and 
                    self.bbox_m[1] <= sample[1] <= self.bbox_m[3]):
                    int_xy = sample
                    break
            lon, lat = self.xy_to_ll.transform(int_xy[0], int_xy[1])
            ints[int_id] = {'xy': int_xy, 'll': (lat, lon), 'type': int_type}
        self.ints = ints

    def get_sigmas(self):
        self.sigmas = {int_id: self.find_mle_sigma_adaptive(int_id, 2500, 500) for int_id in self.ints}

    def gen_int_pickups_log(self, int_xy, int_type, n_farmers, sigma=500):
        """Sequential sampling using Log-Base Prior + Gaussian Kernels."""
        dist_lookup = self.gamma_lookups[int_type]
        grid_points = self.grid_coords.T 
        dists = np.linalg.norm(grid_points - int_xy, axis=1)
        
        # 1. Base Prior (Distance PDF * Spatial Density)
        p_dist_raw = dist_lookup(dists)
        log_p_base = np.log(p_dist_raw + 1e-20) + np.log(self.p_spatial + 1e-20)
        log_p_base -= logsumexp(log_p_base)

        locs = []
        sigma_sq_2 = 2 * (sigma**2)
        acc_exp_kernels = np.zeros(len(grid_points))

        for k in range(n_farmers):
            if k == 0:
                log_p_cond = log_p_base
            else:
                # Add clustering influence (Product in linear, Add in log)
                # log_local_factor = np.log(acc_exp_kernels + 1e-20)
                log_local_factor = np.log(acc_exp_kernels + 1e-20) - np.log(k)
                log_p_cond = log_p_base + log_local_factor
                log_p_cond -= logsumexp(log_p_cond)

            p_sampling = np.exp(log_p_cond)
            
            # Robustness fallback
            if np.isnan(p_sampling).any() or p_sampling.sum() == 0:
                p_sampling = self.p_spatial

            idx = np.random.choice(len(p_sampling), p=p_sampling/p_sampling.sum())
            sampled_xy = self.grid_coords[:, idx]
            locs.append(sampled_xy)
            
            # Efficient O(N_grid) KDE update
            new_dist_sq = np.sum((grid_points - sampled_xy)**2, axis=1)
            acc_exp_kernels += np.exp(-new_dist_sq / sigma_sq_2)

        qs = np.random.choice(self.hist_quantities[int_type], size=n_farmers, replace=True)
        return np.array(locs), qs

    def gen_instance(self, sigmas):
        """Orchestrates full instance generation."""
        farmers, ints = [], []

        for int_id, int_data in self.ints.items():
            int_type, int_xy, int_ll = int_data['type'], int_data['xy'], int_data['ll']
            n_farmers = np.random.choice(self.hist_n_farmers[int_type])
            
            if n_farmers > 0:
                # Get the specific sigma for this intermediary type
                s_val = sigmas.get(int_type, 500)
                locs, qs = self.gen_int_pickups_log(int_xy, int_type, n_farmers, s_val)
                
                routes = []
                for f in range(n_farmers):
                    f_id = f'{int_id}_f{f}'
                    f_lon, f_lat = self.xy_to_ll.transform(locs[f][0], locs[f][1])
                    
                    farmers.append({
                        'id': f_id, 
                        'location': (f_lat, f_lon), # Store as (Lat, Lon)
                        'quantity': float(qs[f])
                    })
                    routes.append(f_id)

                ints.append({
                    'id': int_id, 
                    'capacity': MAX_CAPACITY, 
                    'location': int_ll, 
                    'routes': routes
                })

        return {'farmers': farmers, 'intermediaries': ints, 'mills': self.mills}
    

    def find_mle_sigma_adaptive(self, int_type, start_sigma=1000, step=100):
        """
        Finds the optimal sigma by climbing the likelihood surface 
        until results begin to worsen.
        """
        best_sigma = start_sigma
        best_ll = -np.inf
        current_sigma = start_sigma
        
        # Precompute log_p_base (Base Prior: Distance * Spatial)
        int_data = self.ints_df[self.ints_df['int_id'] == int_type].iloc[0]
        int_xy = np.array([int_data['int_x'], int_data['int_y']])
        dists = np.linalg.norm(self.grid_coords.T - int_xy, axis=1)
        
        log_p_base = np.log(self.gamma_lookups[int_type](dists) + 1e-20) + np.log(self.p_spatial + 1e-20)
        
        # Get historical daily routes
        daily_groups = self.farmers_df[self.farmers_df['int_id'] == int_type].groupby('date')
        historical_indices = []
        
        # Pre-map historical farmers to grid indices to save time in the loop
        for _, group in daily_groups:
            coords = group[['farmer_x', 'farmer_y']].values
            indices = [np.argmin(np.sum((self.grid_coords.T - c)**2, axis=1)) for c in coords]
            historical_indices.append(indices)

        while True:
            total_ll = 0
            sigma_sq_2 = 2 * (current_sigma**2)
            print(current_sigma)
            
            for f_indices in historical_indices:
                acc_exp_kernels = np.zeros(len(self.grid_coords[0]))
                
                for k, target_idx in enumerate(f_indices):
                    # Conditional Log-Prob
                    if k == 0:
                        log_p_cond = log_p_base
                    else:
                        # log_local_factor = np.log(acc_exp_kernels + 1e-20)
                        log_local_factor = np.log(acc_exp_kernels + 1e-20) - np.log(k)
                        log_p_cond = log_p_base + log_local_factor
                    
                    log_p_cond -= logsumexp(log_p_cond)
                    total_ll += log_p_cond[target_idx]
                    
                    # Update kernels
                    sampled_xy = self.grid_coords[:, target_idx]
                    dist_sq = np.sum((self.grid_coords.T - sampled_xy)**2, axis=1)
                    acc_exp_kernels += np.exp(-dist_sq / sigma_sq_2)
            
            # Termination logic: If likelihood drops, the previous sigma was the peak
            if total_ll > best_ll:
                best_ll = total_ll
                best_sigma = current_sigma
                current_sigma += step
            else:
                break # Peak found
                
        return best_sigma