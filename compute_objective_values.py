import torch
import numpy as np
from typing import List, Optional, Tuple
from hop_helper import (
    compute_minimum_hops_from_distances,
)


def compute_objective_values(
    POP,
    distances1,
    distances2,
    beacon_coords,
    weights,
):
    dist_loss1 = compute_distance_losses(
        POP=POP,
        distances=distances1,
        beacon_coords=beacon_coords,
        weights=weights[0],
    )
    dist_loss2 = compute_distance_losses(
        POP=POP,
        distances=distances2,
        beacon_coords=beacon_coords,
        weights=weights[1],
    )
    
    objectives = torch.stack([dist_loss1, dist_loss2], dim=1)
    return objectives


def compute_distance_losses(
    POP,
    distances,
    beacon_coords,
    weights=None,
):
    distances_to_beacons = _compute_distance_to_beacons(POP=POP, beacon_coords=beacon_coords)
    distance_errors = torch.square(
        distances_to_beacons - distances.unsqueeze(0)
    )

    if weights is None:
        return distance_errors.mean(dim=2).mean(dim=1)

    else:
        normalizers = weights.sum(dim=1)
        weighted_errors = (distance_errors * weights.unsqueeze(0)).sum(dim=2) / normalizers.unsqueeze(0)
        weighted_errors = weighted_errors.mean(dim=1)
        return weighted_errors


def _compute_distance_to_beacons(
    POP,
    beacon_coords,
):
    delta = POP.unsqueeze(2) - beacon_coords.unsqueeze(0).unsqueeze(1)
    distances = np.linalg.norm(delta.numpy(), axis=3)
    distances = torch.as_tensor(distances, dtype=torch.float)
    return distances
