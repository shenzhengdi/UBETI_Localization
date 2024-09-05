import torch
from typing import Dict, Optional, Tuple


def crossover(
    POP,
    upper_bounds,
    lower_bounds,
    cp,
    add_mean=False,
    add_min=False,
    add_max=False,
    random_pair_in_crossover=False,
):
    eta_c = 15

    population_size = POP.size(0)
    num_unknown_nodes = POP.size(2)

    step_max, _ = torch.max(POP, dim=0)
    step_min, _ = torch.min(POP, dim=0)
    step_length = (step_max - step_min) / (population_size + 1e-6)
    
    # Whether population `i * 2` and `i * 2 + 1` should do crossover.
    do_cross_mask = (torch.rand(int(population_size / 2)) < cp)

    step_rate1 = torch.rand(int(population_size / 2), POP.size(1), POP.size(2))
    step_rate2 = torch.rand(int(population_size / 2), POP.size(1), POP.size(2))
    
    r2 = torch.rand(int(population_size / 2), POP.size(1), POP.size(2))
    r3 = torch.rand(int(population_size / 2), POP.size(1), POP.size(2))
    
    par1 = POP[::2] + step_rate1 * step_length.unsqueeze(0)
    par2 = POP[1::2] + step_rate2 * step_length.unsqueeze(0)
    if random_pair_in_crossover:
        population_indices = torch.randperm(par1.size(0))
        par2 = par2[population_indices]

    ymin = torch.minimum(par1, par2)
    ymax = torch.maximum(par1, par2)
    
    do_cross_mask = do_cross_mask.unsqueeze(1).unsqueeze(2) & (
        (r2 < 0.5) & (torch.abs(par1 - par2) > 1e-10)
    )
    delta_y = torch.where(
        ymin - lower_bounds.unsqueeze(0) > upper_bounds.unsqueeze(0) - ymax,
        upper_bounds.unsqueeze(0) - ymax,
        ymin - lower_bounds.unsqueeze(0)
    )
    
    beta = 1 + 2 * delta_y / (ymax - ymin)
    expp = eta_c + 1
    beta = 1 / beta
    alpha = 2.0 - torch.pow(beta, expp)
    
    alpha = torch.where(
        r3 <= 1 / alpha,
        alpha * r3,
        1 / (2.0 - alpha * r3),
    )
    
    expp = 1 / (eta_c + 1.0)
    betaq = torch.pow(alpha, expp)
    
    child1 = 0.5 * (ymin + ymax - betaq * (ymax - ymin))
    child2 = 0.5 * (ymin + ymax + betaq * (ymax - ymin))
    
    rand_for_child1 = torch.rand(int(population_size / 2), POP.size(1), POP.size(2))
    child1 = torch.where(
        (child1 > upper_bounds.unsqueeze(0)) | (child1 < lower_bounds.unsqueeze(0)),
        rand_for_child1 * (upper_bounds - lower_bounds).unsqueeze(0) + lower_bounds.unsqueeze(0),
        child1,
    )
    
    rand_for_child2 = torch.rand(int(population_size / 2), POP.size(1), POP.size(2))
    child2 = torch.where(
        (child2 > upper_bounds.unsqueeze(0)) | (child2 < lower_bounds.unsqueeze(0)),
        rand_for_child2 * (upper_bounds - lower_bounds).unsqueeze(0) + lower_bounds.unsqueeze(0),
        child2,
    )
    
    NPOP = POP.clone()
    NPOP[::2][do_cross_mask] = child1[do_cross_mask]
    NPOP[1::2][do_cross_mask] = child2[do_cross_mask]
    
    additional_candidates = []
    if add_mean:
        mean_pop = POP.mean(dim=0)
        additional_candidates.append(mean_pop.unsqueeze(0))
    if add_min:
        min_pop = step_min
        additional_candidates.append(min_pop.unsqueeze(0))
    if add_max:
        max_pop = step_max
        additional_candidates.append(max_pop.unsqueeze(0))
    
    if len(additional_candidates) > 0:
        NPOP = torch.cat(
            [NPOP] + additional_candidates, dim=0
        )

    return NPOP
