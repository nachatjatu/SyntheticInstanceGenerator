import sys
from scipy.stats import gaussian_kde
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from collections import defaultdict

sys.path.insert(0, '../FactoredPlatformSolver/src')


# GLOBAL CONSTANTS - MODIFY AS NEEDED
INDO_CRS = "EPSG:23867"
LL_CRS = "EPSG:4326"
MIN_CAPACITY = 4
MAX_CAPACITY = 9
HARVEST_CYCLE = 14


class InstanceGenerator:
    def __init__(self,
                 farmers_df: pd.DataFrame, 
                 ints_df: pd.DataFrame, 
                 mills_df: pd.DataFrame):

        # initialize crs transformers
        self.xy_to_ll = Transformer.from_crs(INDO_CRS, LL_CRS)
        self.ll_to_xy = Transformer.from_crs(LL_CRS, INDO_CRS)

        # store data
        self.farmers_df = farmers_df
        self.ints_df = ints_df
        self.mills_df = mills_df

        # pre-compute historical distance distributions
        self.hist_n_farmers = self._get_historical_n_farmers()
        self.hist_distances = self._get_historical_distances()
        self.hist_quantities = self._get_historical_quantities()

        # initialize KDEs over data
        self.farmer_kde = self._init_kde('farmer_x', 'farmer_y', self.farmers_df)
        self.int_kde = self._init_kde('int_x', 'int_y', self.ints_df)
        self.mills_kde = self._init_kde('mill_x', 'mill_y', self.mills_df)

        # initialize setup
        self.intermediaries = {}
        self.mills = {}
        self.farmers = {}
    
    # ==========================================================================
    # helper functions for extracting historical data
    # (# of farmers per intermediary type, match distances, and quantities)
    # ==========================================================================

    def _get_historical_n_farmers(self):
        return (self.farmers_df
                .drop_duplicates(['farmer_lat', 'farmer_lon'])
                .groupby('int_id')
                .size()
                .to_dict())

    def _get_historical_distances(self):
        return (self.farmers_df
                .groupby('int_id')['distance']
                .apply(list)
                .to_dict())
    
    def _get_historical_quantities(self):
        return (self.farmers_df
                .groupby('int_id')['quantity']
                .apply(list)
                .to_dict())
    
    def _init_kde(self, x_col: str, y_col: str, df: pd.DataFrame):
        coords = df[[x_col, y_col]].drop_duplicates()
        return gaussian_kde(coords.T, bw_method=0.2)
    
    # ==========================================================================
    # helper functions for generating intermediaries, mills, and farmers
    # ==========================================================================
    
    def generate_ints_mills_farmers(self, n_ints: int, n_mills: int):
        # set instance parameters
        self.n_ints = n_ints
        self.n_mills = n_mills
        # generate synthetic data
        self.intermediaries = self.generate_ints()
        self.mills = self.generate_mills()
        self.farmers = self.generate_farmers()


    def generate_ints(self, jitter=2):
        """
        Generates a synthetic population of intermediaries by sampling locations 
        from a KDE and anchoring farmer counts to historical empirical data.

        Each synthetic intermediary is assigned a type from the historical data. 
        The number of farmers associated with that intermediary is derived from 
        the historical count for that specific type, modified by a discrete 
        integer perturbation (jitter) for variety.

        Args:
            jitter (int, optional): The maximum inclusive integer range to 
                randomly add or subtract from the historical farmer count. 

        Returns:
            dict: A mapping of synthetic intermediary IDs to their attributes, 
                including capacity bounds, spatial coordinates (XY and Lat/Lon), 
                assigned type, and the perturbed farmer count.
        """
        types = self.ints_df.int_id.unique().tolist() 
        # generate synthetic ints with str ID, 
        # types sampled uniformly, and coords from KDE
        int_ids = [f'int_{i}' for i in range(self.n_ints)]
        int_types = random.choices(types, k=self.n_ints)
        int_xys = self.int_kde.resample(self.n_ints).T

        ints = {}

        for i, int_id in enumerate(int_ids):
            int_xy = int_xys[i]
            int_ll = self.xy_to_ll.transform(int_xy[0], int_xy[1])
            int_type = int_types[i]
            int_n_farmers = max(
                1, 
                (self.hist_n_farmers[int_type] + 
                 random.randint(-jitter, jitter))
            ) # take historical n_farmers from type, then perturb +/- jitter

            ints[int_id] = {
                    'min_capacity': MIN_CAPACITY,
                    'max_capacity': MAX_CAPACITY,
                    'xy': int_xy,
                    'll': int_ll,
                    'type': int_type,
                    'n_farmers': int_n_farmers
                }

        return ints
    

    def generate_mills(self):
        """
        Generates synthetic mill locations by resampling from the historical 
        spatial density distribution (KDE).

        Each mill is assigned a unique ID and a spatial coordinate sampled from 
        the `mills_kde`. The coordinates are then projected from the local 
        projected CRS (XY) to geographic coordinates (Lat/Lon).

        Returns:
            dict: A dictionary mapping mill IDs to their spatial attributes, 
                containing both projected 'xy' coordinates and 'll' 
                (Latitude, Longitude) tuples.
        """
        mill_ids = [f'mill_{m}' for m in range(self.n_mills)]
        mill_xys = self.mills_kde.resample(self.n_mills).T

        mills = {}

        for m, mill_id in enumerate(mill_ids):
            mill_xy = mill_xys[m]
            mill_ll = self.xy_to_ll.transform(mill_xy[0], mill_xy[1])

            mills[mill_id] = {
                    'xy': mill_xy,
                    'll': mill_ll
                }
    
        return mills
    
    
    def generate_farmers(self):
        """
        Generates sets of synthetic farmers for each intermediary based on 
        historical distance and quantity distributions.

        For each intermediary, this method samples historical distances and uses 
        a constrained 1D circular sampling technique. This ensures that while 
        farmers are placed at realistic distances from their intermediary, their  
        angular position is weighted by the global spatial density (KDE) of 
        farmers in the region.

        Returns:
            dict: A nested mapping where keys are intermediary IDs and values 
                are dicts of associated farmers and their attributes (location, 
                quantity).
        """
        def sample_1d_circs(center, radii, kde, n_farmers, n_grid=360):
            """
            Samples coordinates along a 1D circular arc around a center point, 
            weighted by a 2D Kernel Density Estimate.

            Args:
                center (array-like): The (X, Y) coordinate of the intermediary.
                radii (np.array): Array of distances (one for each farmer) 
                    sampled from historical data.
                kde (scipy.stats.gaussian_kde): The global farmer spatial 
                    density model.
                n_farmers (int): # of farmers to generate for this intermediary.
                n_grid (int, optional): The angular resolution of the search 
                    circle. Defaults to 360.

            Returns:
                list: A list of (X, Y) tuples for the generated farmers.
            """
            thetas = np.linspace(0, 2*np.pi, n_grid)

            cxs = center[0] + radii[:, np.newaxis] * np.cos(thetas)
            cys = center[1] + radii[:, np.newaxis] * np.sin(thetas)

            points = np.vstack([cxs.ravel(), cys.ravel()])

            densities = kde.evaluate(points)

            density_matrix = densities.reshape(n_farmers, n_grid)

            final_coords = []
            for f in range(n_farmers):
                probs = density_matrix[f] / density_matrix[f].sum()
                idx = np.random.choice(n_grid, p=probs)
                final_coords.append((cxs[f, idx], cys[f, idx]))

            return final_coords

        # generate farmer sets for each intermediary
        int_to_farmers = {}

        for int_id, i in self.intermediaries.items():
            int_type = i['type']
            int_dists = self.hist_distances[int_type]
            int_farmer_quantities = self.hist_quantities[int_type]
            int_n_farmers = i['n_farmers']

            # sample distance from type distribution
            distances = random.choices(int_dists, k=int_n_farmers)
            quantities = random.choices(int_farmer_quantities, k=int_n_farmers)

            # sample locations from 1D circle on farmer KDE
            farmers = {}

            farmer_xys = sample_1d_circs(
                i['xy'], np.array(distances), self.farmer_kde, int_n_farmers)

            for f in range(int_n_farmers):
                farmer_xy = farmer_xys[f]
                farmer_ll = self.xy_to_ll.transform(farmer_xy[0], farmer_xy[1])
                farmers[f'{int_id}_farmer_{f}'] = {
                    'll': farmer_ll,
                    'xy': farmer_xy,
                    'quantity': quantities[f]
                }

            int_to_farmers[int_id] = farmers

        return int_to_farmers
    

    def generate_instances(self, n_instances):
        # keep track of 14 day pickup schedules
        cooldowns = {
            int_id: {
                farmer_id: 0 for farmer_id in self.farmers[int_id].keys()
            } for int_id in self.intermediaries.keys()
        }

        instances = []

        for t in range(n_instances):
            inst_farmers, inst_ints, inst_mills = [], [], []

            # sample pickup subsets for each intermediary
            for int_id, int_data in self.intermediaries.items():
                # get active set of farmers
                active = []
                for farmer_id in self.farmers[int_id].keys():
                    if cooldowns[int_id][farmer_id] == 0:
                        active.append(farmer_id)

                n_to_pickup = random.randint(0, 5)
                pickup_ids = random.sample(
                    active, 
                    k=min(n_to_pickup, len(active))
                )


                farmers = []

                if sum([self.farmers[int_id][farmer_id]['quantity'] 
                        for farmer_id 
                        in pickup_ids]) >= int_data['min_capacity']:
                    q = 0.0

                    for farmer_id in pickup_ids:
                        farmer_location = list(self.farmers[int_id][farmer_id]['ll'])
                        farmer_yield = self.farmers[int_id][farmer_id]['quantity']

                        remaining_space = int_data['max_capacity'] - q
                        
                        if remaining_space <= 0.01:
                            break

                        pickup_quantity = min(farmer_yield, remaining_space - 0.01)

                        if pickup_quantity > 0:
                            cooldowns[int_id][farmer_id] = HARVEST_CYCLE
                            farmers.append({
                                'id': farmer_id,
                                'location': farmer_location,
                                'quantity': pickup_quantity,
                                'intermediary': int_id
                            })
                            q += pickup_quantity

                inst_farmers += farmers

                routes = [f['id'] for f in farmers]

                inst_int = {
                    'id': int_id,
                    'capacity': int_data['max_capacity'],
                    'location': list(int_data['ll']),
                    'routes': routes
                }

                inst_ints.append(inst_int)

            # decrement cooldowns
            for int_id in cooldowns:
                for farmer_id in cooldowns[int_id]:
                    if cooldowns[int_id][farmer_id] > 0:
                        cooldowns[int_id][farmer_id] -= 1

            for mill_id, mill_data in self.mills.items():
                inst_mills.append({
                    'id': mill_id,
                    'location': list(mill_data['ll'])
                })

            inst_mills[-1]['id'] = 'SKIP' # change if necessary
            
            instances.append(
                    {
                        'instance_id': str(t),
                        'farmers': inst_farmers,
                        'intermediaries': inst_ints,
                        'mills': inst_mills
                    }
                )
        
        return instances
