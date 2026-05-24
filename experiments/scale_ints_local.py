import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
from pprint import pprint

sys.path.insert(0, "src")

from farmers_intermediaries import Instance
from road_graphs import RoadGraph
from pricing import Optimizer
from instance_generator import InstanceGenerator

# =========================
# CONFIG
# =========================
TEXTWIDTH = 80
N_INTS = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]    # 10 n_ints
EPSILONS = [2]                     # 1 epsilon
INSTANCE_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]         # 6 instances
TYPE = "low"

GRAPH_PATH = "data/graph_0-14960_00_new.pickle"
FARMERS_PATH = "data/farmers.csv"
FARMERS_2_PATH = "data/farmers_2.csv"
INTS_PATH = "data/ints.csv"

# =========================
# CORE BUILDERS
# =========================
def build_instance(instance_generator, graph, instance_id, n_ints, seed, set_type="high"):
    print(f'Building instance {instance_id} with n_ints = {n_ints}, seed = {seed}, type = {set_type}')

    instance_generator.gen_ints(n_ints, seed, set_type=set_type)

    instance_dict = instance_generator.gen_instance(
        instance_id,
        write=False,
        plot=False,
        seed=seed
    )

    platform = Instance.from_dict(instance_dict)
    platform.set_graph(RoadGraph(graph))

    msg = f" Instance Details ".center(TEXTWIDTH, "-")
    print(msg)
    print("Intermediaries:")
    pprint(platform.intermediaries)
    print()
    print("Farmers:")
    pprint(platform.farmers)
    print()

    return platform, instance_dict



def run_single_simulation(platform, epsilon, graph):
    print(f'Running simulation with epsilon = {epsilon}')

    platform.set_graph(RoadGraph(graph))

    epsilons = {intermediary.id: epsilon for intermediary in platform.intermediaries}
    het_costs = {intermediary.id: (platform.dist_to_mill[intermediary.id] * 4) for intermediary in platform.intermediaries}

    parameters = {
        "epsilon": epsilons,
        "solver": "gurobi",
        "het_costs": het_costs,
    }

    opt = Optimizer(platform, parameters)
    base_matchings = opt.base_matchings

    summary_vanilla = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": False,
        "domination": False,})
    
    opt = Optimizer(platform, parameters, base_matchings=base_matchings)
    
    summary_structured = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": True,
        "domination": False,})
    
    opt = Optimizer(platform, parameters, base_matchings=base_matchings)
    
    summary_domination = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": False,
        "domination": True,})

    farmer_quantities = {f.id: f.quantity for f in platform.farmers}

    print(" Profits ".center(TEXTWIDTH, "-"))
    print(f"Vanilla: {summary_vanilla.max_int_welf_sol.profit}")
    print(f"Structured: {summary_structured.max_int_welf_sol.profit}")
    print(f"Domination: {summary_domination.max_int_welf_sol.profit}")
    print(f"Vanilla Profit %: {summary_vanilla.max_int_welf_sol.profit / (np.sum(list(farmer_quantities.values())) * platform.fruit_price) * 100}")
    print()

    return {
        "epsilon": epsilon,
        "cost": het_costs,
        "farmer_quantities": farmer_quantities,
        "summary_vanilla": summary_vanilla.to_dict(),
        "summary_structured": summary_structured.to_dict(),
        "summary_domination": summary_domination.to_dict(),
        "farmer_dirt_to_mill": {f.id: f.dirt_to_mill for f in platform.farmers},
        "farmer_paved_to_mill": {f.id: f.paved_to_mill for f in platform.farmers},
    }



# =========================
# JSON SERIALIZATION
# =========================
def convert(obj):
    if isinstance(obj, (np.float64, np.int64)):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


# =========================
# MAIN EXPERIMENT LOOP
# =========================
def main():
    # load static data
    farmers_df = pd.read_csv(FARMERS_PATH)
    farmers_2_df = pd.read_csv(FARMERS_2_PATH)
    ints_df = pd.read_csv(INTS_PATH)
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    sweep_grid = list(itertools.product(INSTANCE_SEEDS, EPSILONS, N_INTS))

    for n_id in range(100):
        instance_id, epsilon, n_ints = sweep_grid[n_id]
        
        msg = f" Experiment {n_id}: n_ints = {n_ints}, instance_seed = {instance_id}, epsilon = {epsilon} "
        print(msg.center(TEXTWIDTH, "="))
        print()

        # initialize generator and platform
        instance_generator = InstanceGenerator(farmers_df, farmers_2_df, ints_df, GRAPH_PATH)
        platform, _ = build_instance(instance_generator, graph, n_id, n_ints, seed=instance_id, set_type=TYPE)

        # run one stochastic solve
        sim_result = run_single_simulation(platform, epsilon, graph)

        # add metadata
        sim_result.update({
            "instance_id": instance_id,
            "n_id": n_id,
        })

        # save results
        results_path = Path(f"data/results_scale_ints_local/{n_id}.json")

        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, "w") as f:
            json.dump(sim_result, f, indent=4, default=convert)

        print(f"Results saved to {results_path}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()