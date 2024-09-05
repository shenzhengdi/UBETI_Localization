import torch

from compute_crowding_distance import compute_crowding_distance


def update_parents(
    combined_pop,
    combined_objectives,
    combined_frontier_sort_idx,
    population_size,
):
    assert combined_pop.size(0) >= population_size
    
    selected_pop = []
    selected_objectives = []
    selected_frontier_sort_idx = []
    selected_combined_pop_indices = []
    
    i = 1
    while len(selected_pop) < population_size:
        Fisign, = torch.where(combined_frontier_sort_idx == i)
        if len(selected_pop) + len(Fisign) <= population_size:
            selected_pop += combined_pop[Fisign].tolist()
            selected_objectives += combined_objectives[Fisign].tolist()
            selected_frontier_sort_idx += combined_frontier_sort_idx[Fisign].tolist()
            selected_combined_pop_indices += Fisign.tolist()
        else:
            num_needed = population_size - len(selected_pop)
            ttPOP, ttpa, ttI, pop_indices = compute_crowding_distance(
                tPOP=combined_pop[Fisign],
                tpa=combined_objectives[Fisign],
                pop_indices=Fisign,
            )
            
            _, ttI_sort_indices = torch.sort(-ttI)
            ttI_sort_indices = ttI_sort_indices[:num_needed]
            
            selected_pop += ttPOP[ttI_sort_indices].tolist()
            selected_objectives += ttpa[ttI_sort_indices].tolist()
            selected_frontier_sort_idx += [i] * num_needed
            selected_combined_pop_indices += pop_indices.tolist()
        
        i += 1
    
    selected_pop = torch.as_tensor(selected_pop, dtype=combined_pop.dtype)
    selected_objectives = torch.as_tensor(selected_objectives, dtype=combined_objectives.dtype)
    selected_frontier_sort_idx = torch.as_tensor(selected_frontier_sort_idx, dtype=torch.long)
    
    return selected_pop, selected_objectives, selected_frontier_sort_idx, selected_combined_pop_indices
