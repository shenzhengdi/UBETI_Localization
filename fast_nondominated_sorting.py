import numpy as np
import torch
from typing import Tuple

from fast_nondominated_sorting_cpp import generate_frontier_info, update_frontier_info


def fast_nondominated_sorting(
    objectives,
):
    population_size, num_objectives = objectives.shape
    S = -np.ones((population_size, population_size), dtype=int)

    frontier_idx = np.zeros((population_size, ), dtype=int)
    frontier_sort_idx = np.zeros((population_size, ), dtype=int)
    
    is_first_dominating = _find_dominating_pairs(objectives)

    S, frontier_idx, frontier_sort_idx = generate_frontier_info(
        frontier_sort_idx,
        frontier_idx,
        S,
        is_first_dominating.cpu().numpy(),
    )
    
    frontier_sort_idx = update_frontier_info(
        frontier_sort_idx, frontier_idx, S
    )
    frontier_sort_idx = torch.as_tensor(frontier_sort_idx, dtype=torch.long, device=objectives.device)

    return frontier_sort_idx


def _find_dominating_pairs(
    objectives,
):
    is_first_equal_or_better = (objectives.unsqueeze(1) <= objectives.unsqueeze(0)).all(dim=2)
    is_equal = (objectives.unsqueeze(1) == objectives.unsqueeze(0)).all(dim=2)
    
    is_first_dominating = is_first_equal_or_better & (~is_equal)
    
    return is_first_dominating
