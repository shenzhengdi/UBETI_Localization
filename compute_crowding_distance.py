import torch
from typing import Optional, Tuple


def compute_crowding_distance(
    tPOP,
    tpa,
    pop_indices=None,
):
    num_pop, num_objectives = tpa.shape
    padis = torch.zeros((num_pop, ), dtype=torch.float)
    
    if num_pop == 0:
        return tPOP, tpa, padis, pop_indices

    # Sort populations for each objective.
    sorted_tpa, sorted_indices = torch.sort(tpa, dim=0)

    for i in range(num_objectives):
        tpa = tpa[sorted_indices[:, i]]
        tPOP = tPOP[sorted_indices[:, i]]
        
        padis = padis[sorted_indices[:, i]]
        padis[0] = float("inf")
        padis[-1] = float("inf")
        tpai1 = tpa[2:num_pop, i]
        tpad1 = tpa[0:num_pop - 2, i]
        fimin = min(tpa[:, i])
        fimax = max(tpa[:, i])
        padis[1:num_pop - 1] = padis[1:num_pop - 1] + (tpai1 - tpad1) / (fimax - fimin + 1e-4)
        
        if pop_indices is not None:
            pop_indices = pop_indices[sorted_indices[:, i]]

    return tPOP, tpa, padis, pop_indices
