import numpy as np
import torch


def node_area(
    upper_bounds,
    lower_bounds,
    radius,
    hops,
    beacon_coords,
):
    beacon_coords = beacon_coords.permute(1, 0)
    lower_bounds_all_beacons = beacon_coords.unsqueeze(2) - radius * hops.unsqueeze(0)
    upper_bounds_all_beacons = beacon_coords.unsqueeze(2) + radius * hops.unsqueeze(0)
    
    lower_bounds = torch.maximum(
        lower_bounds,
        torch.amax(lower_bounds_all_beacons, dim=1),
    )
    upper_bounds = torch.minimum(
        upper_bounds,
        torch.amin(upper_bounds_all_beacons, dim=1),
    )
    return upper_bounds, lower_bounds
