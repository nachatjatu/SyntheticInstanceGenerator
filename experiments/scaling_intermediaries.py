"""Experiment 6 script: similar to experiment_4 with structured pricing.

The routine performs a grid of heuristic runs, recording results for
vanilla, structured, and domination pricing options.  The script accepts a
single numeric argument and writes outputs to ``data/results_6``.

Usage::
    python experiments/experiment_6.py <n_id>
"""

import sys
sys.path.insert(0, "src")  # src/ contains core library modules

from farmers_intermediaries import Instance
from road_graphs import RoadGraph
import pickle
import pandas as pd
import pricing
from pricing import Optimizer
from datetime import datetime, timedelta
import sys
import numpy as np
import json
from instance_generator import InstanceGenerator

if len(sys.argv) != 2:
    raise ValueError("Please provide n_id as a single argument when running the script.")
n_id = int(sys.argv[1])


GRAPH_PATH = "data/graph_0-14960_00.pickle"
RESULTS_PATH = f"data/results_sc/{n_id}.json"

FARMERS_PATH = "data/farmers.csv"
INTS_PATH = "data/ints.csv"
MILLS_PATH = "data/mills.csv"


# =========================
# LOAD STATIC DATA
# =========================
farmers_df = pd.read_csv(FARMERS_PATH)
ints_df = pd.read_csv(INTS_PATH)
mills_df = pd.read_csv(MILLS_PATH)

with open(GRAPH_PATH, "rb") as f:
    GRAPH = pickle.load(f)

# Initialize generator once
instance_generator = InstanceGenerator(farmers_df, ints_df, GRAPH_PATH)

np.random.seed(n_id) # set random seed for reproducibility and randomness

sim_size = 1

def reset_quantities():
    return {farmer.id: min(max(np.floor((farmer.quantity + np.random.uniform(-0.5,0.5))*10)/10,0.1),9.0) for farmer in Platform.farmers}

def reset_fixed_costs():
    return {intermediary.id: Platform.dist_to_mill[intermediary.id]*4+np.random.normal(0, 100000) for intermediary in Platform.intermediaries}


def diagnose_instance(platform):
    route_totals = []

    for intermediary in platform.intermediaries:
        for hist_set in intermediary.additional_info["hist_sets"]:
            q = sum(platform.farmer_by_id[f_id].quantity for f_id in hist_set)
            route_totals.append(q)

    total_quantity = sum(f.quantity for f in platform.farmers)

    diagnostics = {
        "n_farmers": len(platform.farmers),
        "n_intermediaries": len(platform.intermediaries),
        "total_quantity": float(total_quantity),
        "hist_route_min": float(np.min(route_totals)) if route_totals else None,
        "hist_route_mean": float(np.mean(route_totals)) if route_totals else None,
        "hist_route_max": float(np.max(route_totals)) if route_totals else None,
        "num_infeasible_hist_routes": int(sum(q > platform.truck_capacity for q in route_totals)),
        "farmers_per_intermediary_mean": len(platform.farmers) / len(platform.intermediaries),
        "quantity_per_intermediary_mean": total_quantity / len(platform.intermediaries),
    }

    print("Instance diagnostics:", diagnostics)
    return diagnostics

# =========================
# CORE BUILDERS
# =========================
def build_instance(instance_id, seed=n_id):
    """
    Generate a fresh instance and attach the road graph.
    """
    instance_generator.gen_ints(14, seed)

    instance_dict = instance_generator.gen_instance(
        instance_id,
        write=False,
        plot=False,
        seed=seed,
        scale_factor = 1.0
    )

    platform = Instance.from_dict(instance_dict)
    platform.set_graph(RoadGraph(GRAPH))

    return platform, instance_dict


results = []
print(f"Running grid")
for sim_n in range(sim_size):
    sampled_epsilon = 2

    Platform, instance_dict = build_instance(1, seed=n_id)

    diagnose_instance(Platform)
    epsilon = {int.id: sampled_epsilon for int in Platform.intermediaries}


    with open("data/graph_0-14960_00.pickle", 'rb') as pickle_file:
        G = pickle.load(pickle_file)

    Platform.set_graph(RoadGraph(G))

    #new_hist_matching = reset_relationships()
    farmer_quantities = {farmer.id: farmer.quantity for farmer in Platform.farmers}
    print(farmer_quantities)
    print(f"Running simulation index {sim_n} of {sim_size}")
    het_costs = reset_fixed_costs()
    parameters = {
        "epsilon":epsilon, 
        "solver": "gurobi", 
        "het_costs": het_costs,
    }
    farmer_dirt_to_mill = {farmer.id: farmer.dirt_to_mill for farmer in Platform.farmers}
    farmer_paved_to_mill = {farmer.id: farmer.paved_to_mill for farmer in Platform.farmers}


    opt = Optimizer(Platform, parameters)
    base_matchings = opt.base_matchings

    summary_vanilla = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": False,
        "domination": False,})
    
    opt = Optimizer(Platform, parameters, base_matchings=base_matchings)
    
    summary_structured = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": True,
        "domination": False,})
    
    opt = Optimizer(Platform, parameters, base_matchings=base_matchings)
    
    summary_domination = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": False,
        "domination": True,})

    results.append({
        "instance_str": n_id,
        "cost": het_costs,
        "epsilon": epsilon,
        "farmer_quantities": farmer_quantities,
        #"new_hist_matching": new_hist_matching,
        "summary_vanilla": summary_vanilla.to_dict(),
        "summary_structured": summary_structured.to_dict(),
        "summary_domination": summary_domination.to_dict(),
        #"farmer_locations": farmer_locations,
        "farmer_dirt_to_mill": farmer_dirt_to_mill,
        "famer_paved_to_mill": farmer_paved_to_mill,
    })

    print(f"Simulation index {sim_n} completed with farmer_quantities {farmer_quantities} and het_costs {het_costs}")
    print(f"Profits are vanilla: {summary_vanilla.max_int_welf_sol.profit}, structured: {summary_structured.max_int_welf_sol.profit}, domination: {summary_domination.max_int_welf_sol.profit}")
    print(f"Profit percentage is: {summary_vanilla.max_int_welf_sol.profit / (np.sum(list(farmer_quantities.values())) * Platform.fruit_price)}")

# Save the result to a JSON file
def convert(obj):
    if isinstance(obj, (np.float64, np.int64)):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


with open(f"data/results_6/{n_id}.json", 'w') as f:
    json.dump(results, f, indent=4, default=convert)