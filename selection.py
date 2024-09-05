import torch

from compute_crowding_distance import compute_crowding_distance


def get_selection(
    POP,
    objectives,
    frontier_sort_idx,
):
    population_size = POP.size(0)
    padis = torch.zeros((population_size, ), dtype=torch.float)
    for i in range(max(frontier_sort_idx)):
        pop_mask = frontier_sort_idx == i
        tpa = objectives[pop_mask]
        tPOP = POP[pop_mask]
        updated_features, updated_objectives, ttI, _ = compute_crowding_distance(tPOP=tPOP, tpa=tpa)
        
        POP[pop_mask] = updated_features
        objectives[pop_mask] = updated_objectives
        frontier_sort_idx[pop_mask] = i + torch.zeros((pop_mask.sum(), ), dtype=torch.long)
        padis[pop_mask] = ttI
    
    NPOP = POP.clone()
    Npal = objectives.clone()
    new_frontier_sort_idx = frontier_sort_idx.clone()
    new_padis = padis.clone()
    samples = torch.ceil(torch.rand(population_size) * (population_size - 0.5) - 0.5).to(torch.long)

    mask = (samples != torch.arange(0, population_size))
    mask = mask & (
        frontier_sort_idx > torch.gather(frontier_sort_idx, 0, samples) | \
        ((frontier_sort_idx == torch.gather(frontier_sort_idx, 0, samples)) & \
        (padis < torch.gather(padis, 0, samples)))
    )
    
    selected_samples = samples[mask]
    NPOP[mask] = torch.gather(POP, 0, selected_samples.unsqueeze(1).unsqueeze(2).expand(-1, POP.size(1), POP.size(2)))
    Npal[mask] = torch.gather(objectives, 0, selected_samples.unsqueeze(1).expand(-1, objectives.size(1)))
    new_frontier_sort_idx[mask] = frontier_sort_idx[selected_samples]
    new_padis[mask] = padis[selected_samples]

    return (
        NPOP,
        Npal,
        new_frontier_sort_idx,
        new_padis,
    )
