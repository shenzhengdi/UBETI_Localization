from dataclasses import dataclass
import numpy as np
import torch
import os
from collections import defaultdict
from tqdm import tqdm
from typing import List
from get_boundary import get_boundray
from get_dnhop import get_dnhop
from node_area import node_area
from compute_objective_values import compute_objective_values
from fast_nondominated_sorting import fast_nondominated_sorting
from selection import get_selection
from crossover import crossover
from mutation import mutation
from update_parents import update_parents


INPUT_DATA_DIR = "input_data"
PROBLEM_TO_DATA_FILE_MAP = {
    "random": "test.txt",
    "Oshape": "testO.txt",
    "Cshape": "testC.txt",
    "Xshape": "testX.txt",
}


@dataclass
class NSGA_Solution:
    POP: torch.Tensor
    distance1: torch.tensor
    distance2: torch.tensor
    beacon_coords: torch.Tensor
    unknown_coords: torch.Tensor
    hops: torch.tensor
    weights: List


def solve_gaii(
    radius,
    num_beacons,
    problem_name,  # Use a key from `PROBLEM_TO_DATA_FILE_MAP`.
    num_all_nodes=100,
    max_iter=500,
    logging_step=50,
    display_logging=True,
    border_length_m=100,
    cp=0.9,
    mp=0.1,
    num_trials=2,
    population_size=20,
    estimate_exp_errors=False,
    add_mean_pop_in_crossover=True,
    add_min_max_pop_in_crossover=True,
    remove_mean_pops_from_stats_in_selection=False,
    remove_min_max_pops_from_stats_in_selection=False,
    random_pair_in_crossover=False,
    use_upper_bound_estimation=False,
    unweight_values_exceeding_dist_upper_bound=False,
    convert_solution_to_dict=False,
):
    num_unknown_nodes = num_all_nodes - num_beacons
    lower_bounds, upper_bounds = get_boundray(num_unknown_nodes)
    upper_bounds = torch.from_numpy(upper_bounds)
    lower_bounds = torch.from_numpy(lower_bounds)
    solutions = []

    logged_metrics_all_trials = defaultdict(list)

    for trial in range(num_trials):
        all_node_coords = load_input_data(
            problem_name,
            num_all_nodes=num_all_nodes,
        )
        assert all_node_coords.shape[0] >= num_all_nodes
        beacon_coords = all_node_coords[:num_beacons, :]
        unknown_coords = all_node_coords[num_beacons : num_all_nodes, :]

        dnhop_result = get_dnhop(
            all_node_coords=all_node_coords,
            num_beacons=num_beacons,
            radius=radius,
            estimate_exp_errors=estimate_exp_errors,
            use_upper_bound_estimation=use_upper_bound_estimation,
            unweight_values_exceeding_dist_upper_bound=unweight_values_exceeding_dist_upper_bound,
        )
        
        # Convert numpy arrays to torch tensors.
        beacon_coords = torch.from_numpy(beacon_coords)
        unknown_coords = torch.from_numpy(unknown_coords)

        upper_bounds, lower_bounds = node_area(upper_bounds, lower_bounds, radius, dnhop_result.hops, beacon_coords)

        # Compute fitness scores of ground truth unknown node positions.
        gt_fitness_scores = compute_objective_values(
            POP=unknown_coords.unsqueeze(0),
            distances1=dnhop_result.distance1,
            distances2=dnhop_result.distance2,
            beacon_coords=beacon_coords,
            weights=dnhop_result.weights,
        )

        logged_metrics_single_trial = defaultdict(list)
        # Initialize population.
        EPOP = torch.rand(population_size, 2, num_unknown_nodes) * (upper_bounds - lower_bounds).unsqueeze(0) \
            + lower_bounds.unsqueeze(0)
        samples = torch.rand(int(population_size / 4), 2, num_unknown_nodes)
        # Use estimated coordinates in EPOP for every 4 population if the estimated
        # coordinate is in the range of (lower_bound, upper_bound).
        estimated_coords_in_bounds = (dnhop_result.estimated_coords > lower_bounds) & (
            dnhop_result.estimated_coords < upper_bounds)
        use_estimated_coords_mask = estimated_coords_in_bounds.unsqueeze(0) & (samples < 0.5)
        EPOP[::4] = torch.where(use_estimated_coords_mask, dnhop_result.estimated_coords, EPOP[::4])
        
        # Compute objective values.
        fitness_scores = compute_objective_values(
            POP=EPOP.permute(0, 2, 1),
            distances1=dnhop_result.distance1,
            distances2=dnhop_result.distance2,
            beacon_coords=beacon_coords,
            weights=dnhop_result.weights,
        )
        
        # Fast non-dominated sorting.
        frontier_sort_idx = fast_nondominated_sorting(fitness_scores)

        (
            NPOP,
            Npal,
            new_frontier_sort_idx,
            new_padis,
        ) = get_selection(
            POP=EPOP,
            objectives=fitness_scores,
            frontier_sort_idx=frontier_sort_idx,
        )
        
        NPOP = crossover(
            POP=NPOP,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            cp=cp,
            add_mean=add_mean_pop_in_crossover,
            add_min=add_min_max_pop_in_crossover,
            add_max=add_min_max_pop_in_crossover,
            random_pair_in_crossover=random_pair_in_crossover,
        )
        
        NPOP = mutation(
            POP=NPOP,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            mp=mp,
        )
        
        # NSGA iteration.
        NSGA_iter = tqdm(range(max_iter), desc="NSGA", disable=not display_logging)
        ite_cnt = 0
        for ite in NSGA_iter:
            ite_cnt += 1
            
            combined_pop = torch.cat([NPOP, EPOP], dim=0)
            
            new_fitness_scores = compute_objective_values(
                POP=NPOP.permute(0, 2, 1),
                distances1=dnhop_result.distance1,
                distances2=dnhop_result.distance2,
                beacon_coords=beacon_coords,
                weights=dnhop_result.weights,
            )
            
            combined_fitness_scores = torch.cat([new_fitness_scores, fitness_scores], dim=0)
            
            combined_frontier_sort_idx = fast_nondominated_sorting(combined_fitness_scores)
            
            (
                EPOP, fitness_scores, frontier_sort_idx, selected_combined_pop_indices
            ) = update_parents(
                combined_pop=combined_pop,
                combined_objectives=combined_fitness_scores,
                combined_frontier_sort_idx=combined_frontier_sort_idx,
                population_size=population_size,
            )

            combined_frontier_sort_idx = fast_nondominated_sorting(combined_fitness_scores)
            (
                EPOP, fitness_scores, frontier_sort_idx, selected_combined_pop_indices
            ) = update_parents(
                combined_pop=combined_pop,
                combined_objectives=combined_fitness_scores,
                combined_frontier_sort_idx=combined_frontier_sort_idx,
                population_size=population_size,
            )

            # Selection, crossover and mutation.
            NPOP, _, _, _ = get_selection(
                POP=EPOP,
                objectives=fitness_scores,
                frontier_sort_idx=frontier_sort_idx,
            )
            NPOP = crossover(
                POP=NPOP,
                upper_bounds=upper_bounds,
                lower_bounds=lower_bounds,
                cp=cp,
                add_mean=add_mean_pop_in_crossover,
                add_min=add_min_max_pop_in_crossover,
                add_max=add_min_max_pop_in_crossover,
                random_pair_in_crossover=random_pair_in_crossover,
            )
            NPOP = mutation(
                POP=NPOP,
                upper_bounds=upper_bounds,
                lower_bounds=lower_bounds,
                mp=mp,
            )
        
        nsga_solution = NSGA_Solution(
            POP=EPOP.permute(0, 2, 1),
            distance1=dnhop_result.distance1,
            distance2=dnhop_result.distance2,
            beacon_coords=beacon_coords,
            unknown_coords=unknown_coords,
            hops=dnhop_result.hops.permute(1, 0),
            weights=dnhop_result.weights,
        )
        if convert_solution_to_dict:
            nsga_solution = nsga_solution.__dict__

        solutions.append(nsga_solution)

        # End of trial.

    return solutions


def load_input_data(problem_name, num_all_nodes=None):
    filename = os.path.join(INPUT_DATA_DIR, PROBLEM_TO_DATA_FILE_MAP[problem_name])
    node_coords = []
    with open(filename, 'r') as file:
        # Read the contents of the file
        lines = file.readlines()
        # First line is x coordinates, second line is y coordinates.
        assert len(lines) == 2

        # Process each line
        for line in lines:
            numbers = line.split()
            numbers = [float(num) for num in numbers]
            node_coords.append(numbers)

    node_coords = np.asarray(node_coords, dtype=float).T
        
    if num_all_nodes is not None:
        node_coords = node_coords[:num_all_nodes, :]

    return node_coords
