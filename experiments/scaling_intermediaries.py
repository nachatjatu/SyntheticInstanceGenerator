import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")

from farmers_intermediaries import Instance
from road_graphs import RoadGraph
from pricing import Optimizer
from instance_generator import InstanceGenerator


# =========================
# CLI ARGUMENTS
# =========================
if len(sys.argv) != 2:
    raise ValueError("Please provide n_id as a single argument.")

n_id = int(sys.argv[1])

# =========================
# CONFIG
# =========================
SIM_SIZE = 1
N_INTS_LIST = [i for i in range(6, 30+1, 4)]
N_INSTANCES = 4

GRAPH_PATH = "data/graph_0-14960_00.pickle"

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


# =========================
# STOCHASTIC COMPONENTS
# =========================
def reset_quantities(platform, rng):
    return {
        farmer.id: min(
            max(
                np.floor((farmer.quantity + rng.uniform(-0.5, 0.5)) * 10) / 10,
                0.1,
            ),
            9.0,
        )
        for farmer in platform.farmers
    }


def reset_fixed_costs(platform, rng):
    return {
        intermediary.id: (
            platform.dist_to_mill[intermediary.id] * 4
            + rng.normal(0, 100000)
        )
        for intermediary in platform.intermediaries
    }


def sample_epsilon(platform, rng):
    return {
        intermediary.id: rng.uniform(0, 6.0)
        for intermediary in platform.intermediaries
    }




# =========================
# CORE BUILDERS
# =========================
def build_instance(instance_id, n_ints, seed=n_id):
    """
    Generate a fresh instance and attach the road graph.
    """
    instance_generator.gen_ints(n_ints, seed)

    instance_dict = instance_generator.gen_instance(
        instance_id,
        write=False,
        plot=False,
        seed=seed
    )

    platform = Instance.from_dict(instance_dict)
    platform.set_graph(RoadGraph(GRAPH))

    return platform, instance_dict


def apply_quantity_perturbation(instance_dict, platform, rng):
    """
    Rebuild platform with perturbed quantities.
    """
    quantities = reset_quantities(platform, rng)
    return Instance.from_dict(instance_dict, opt_quantities=quantities)


def run_single_simulation(instance_dict, platform, rng):
    """
    Runs one optimization on a (possibly perturbed) platform.
    """
    platform = apply_quantity_perturbation(instance_dict, platform, rng)

    platform.set_graph(RoadGraph(GRAPH))

    epsilon = sample_epsilon(platform, rng)
    het_costs = reset_fixed_costs(platform, rng)

    parameters = {
        "epsilon": epsilon,
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

    # Collect outputs

    farmer_quantities = {f.id: f.quantity for f in platform.farmers}

    print(f"Profits are vanilla: {summary_vanilla.max_int_welf_sol.profit}, structured: {summary_structured.max_int_welf_sol.profit}, domination: {summary_domination.max_int_welf_sol.profit}")
    print(f"Profit percentage is: {summary_vanilla.max_int_welf_sol.profit / (np.sum(list(farmer_quantities.values())) * platform.fruit_price)}")

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

    n_ints = N_INTS_LIST[n_id % len(N_INTS_LIST)]
    instance_id = (n_id // len(N_INTS_LIST)) % N_INSTANCES

    print(f"Starting experiment {n_id} with {n_ints} ints and instance {instance_id}")

    platform, instance_dict = build_instance(n_id, n_ints, seed=instance_id)

    rng = np.random.default_rng(n_id)

    # Run one stochastic solve
    sim_result = run_single_simulation(instance_dict, platform, rng)

    # Add metadata
    sim_result.update({
        "instance_id": instance_id,
        "n_id": n_id,
    })

    # Save results
    results_path = Path(f"data/results_scaling_ints/{instance_id}/{n_id}.json")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(sim_result, f, indent=4, default=convert)

    print(f"Results saved to {results_path}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()