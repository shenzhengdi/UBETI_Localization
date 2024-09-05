from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from typing import List
from scipy.spatial import distance
import scipy.integrate as integrate
import math

from hop_helper import (
    compute_minimum_hops_from_distances,
    INF_HOP,
)


@dataclass
class DVFeatures:
    radius: float
    hops: torch.tensor
    distance1: torch.tensor
    distance2: torch.tensor
    estimated_coords: torch.tensor
    weights: List


def get_dnhop(
    all_node_coords,
    num_beacons,
    radius,
    estimate_exp_errors,
    use_upper_bound_estimation,
    unweight_values_exceeding_dist_upper_bound,
):
    num_all_nodes = all_node_coords.shape[0]
    # Pairwise Euclidean distance.
    gt_distances_all_nodes = distance.cdist(all_node_coords, all_node_coords)

    hops = compute_minimum_hops_from_distances(
        gt_distances_all_nodes=gt_distances_all_nodes,
        radius=radius,
    )

    beacon_hops = hops[:num_beacons, :num_beacons]
    beacon_distances = gt_distances_all_nodes[:num_beacons, :num_beacons]
    est_ave_dist_for_anchors = beacon_distances.sum(axis=1) / beacon_hops.sum(axis=1)

    anchor_to_unknown_hops = hops[:num_beacons, num_beacons:]
    max_hop = (anchor_to_unknown_hops * (anchor_to_unknown_hops<INF_HOP)).max()
    max_hop_per_anchor = np.amax(anchor_to_unknown_hops * (anchor_to_unknown_hops<INF_HOP), axis=1)

    num_unknown_nodes = num_all_nodes - num_beacons
    # Count the number of unknown nodes with a certain hop to each anchor node.
    hop_to_anchor_cnt = np.zeros((max_hop, num_beacons), dtype=int)
    dist_hop = np.zeros((max_hop, num_beacons), dtype=float)
    for anchor_idx in range(num_beacons):
        for unknown_idx in range(num_unknown_nodes):
            for num_hop in range(max_hop):
                if anchor_to_unknown_hops[anchor_idx, unknown_idx] < num_hop + 2:
                    hop_to_anchor_cnt[num_hop, anchor_idx] += 1

        up_b = np.zeros(max_hop, dtype=float)
        up_b[0] = 2 * hop_to_anchor_cnt[0, anchor_idx] / (2 * hop_to_anchor_cnt[0, anchor_idx] + 1) * radius
        dist_hop[0, anchor_idx] = _compute_probability_based_distance_estimation(
            lower_bound=0.0, upper_bound=up_b[0]
        )

        for num_hop in range(1, max_hop_per_anchor[anchor_idx]):
            up_b[num_hop] = expected_distance(radius, up_b[num_hop - 1], num_hop + 1, hop_to_anchor_cnt[num_hop, anchor_idx])
            dist_hop[num_hop, anchor_idx] = _compute_probability_based_distance_estimation(
                lower_bound=dist_hop[num_hop - 1, anchor_idx], upper_bound=up_b[num_hop]
            )

    anchor_to_unknown_hops = torch.from_numpy(anchor_to_unknown_hops)

    # Estimate distance by hops.
    distance1 = torch.from_numpy(est_ave_dist_for_anchors).unsqueeze(1) * anchor_to_unknown_hops
    distance1 = distance1.permute(1, 0)
    distance2 = torch.zeros(num_unknown_nodes, num_beacons, dtype=torch.float)
    for anchor_idx in range(num_beacons):
        for unknown_idx in range(num_unknown_nodes):
            kk = anchor_to_unknown_hops[anchor_idx, unknown_idx]
            distance2[unknown_idx, anchor_idx] = dist_hop[kk-1, anchor_idx]

    # Least square.
    X = estimate_unknown_positions_by_least_square(
        all_node_coords=all_node_coords,
        num_beacons=num_beacons,
        distances=distance1,
    )
    
    beacon_coords = torch.from_numpy(all_node_coords[:num_beacons, :])

    # Get fitness score weights.
    weights1 = torch.ones(
        distance1.shape[0], distance1.shape[1], dtype=torch.float,
    )
    weights2 = torch.ones(
        distance2.shape[0], distance2.shape[1], dtype=torch.float,
    )

    weights = [weights1, weights2]
    
    (
        distance_upper_bound,
        reference_anchor_indices,
    ) = get_distance_upper_bound_by_triangle_inequality(
        beacon_coords=beacon_coords,
        hops=anchor_to_unknown_hops.permute(1, 0),
        radius=radius,
    )

    distance1_exceed = distance1 > distance_upper_bound
    distance2_exceed = distance2 > distance_upper_bound
    
    if unweight_values_exceeding_dist_upper_bound:
        weights1 = torch.where(
            distance1_exceed,
            0 * weights1, weights1
        )
        weights2 = torch.where(
            distance2_exceed,
            0 * weights2, weights2
        )

        weights = [weights1, weights2]

    if use_upper_bound_estimation:
        distance1 = torch.minimum(distance1, distance_upper_bound)
        distance2 = torch.minimum(distance2, distance_upper_bound)

    return DVFeatures(
        radius=radius,
        hops=anchor_to_unknown_hops,
        distance1=distance1,
        distance2=distance2,
        estimated_coords=torch.from_numpy(X),
        weights=weights,
    )


def _compute_probability_based_distance_estimation(
    lower_bound, upper_bound
):
    if lower_bound == upper_bound:
        return float("nan")
    numerator, _ = integrate.quad(lambda r: 2 * math.pi * (r**2), lower_bound, upper_bound)
    denominator, _ = integrate.quad(lambda r: 2 * math.pi * r, lower_bound, upper_bound)
    if denominator == 0:
        raise Exception(f"lower_bound = {lower_bound}, upper_bound = {upper_bound}")
    return numerator / denominator


def _compupte_expected_distance_error_estimation(
    lower_bound, upper_bound, est_distance
):
    if lower_bound == upper_bound:
        return float("nan")
    numerator, _ = integrate.quad(lambda r: 2 * math.pi * r * abs(r - est_distance), lower_bound, upper_bound)
    denominator, _ = integrate.quad(lambda r: 2 * math.pi * r, lower_bound, upper_bound)
    return numerator / denominator


def expected_distance(
    radius,
    e2,
    num_hop,
    hop_to_anchor_cnt,
):
    lr = 0.98
    e2 = e2 * lr
    num_intervals = 2500
    upper_bound = radius * num_hop
    
    if num_hop > 1:
        lower_bound = radius * (num_hop - 2)
    else:
        lower_bound = 0
    
    subinterval_size = (upper_bound - lower_bound) / num_intervals
    x = lower_bound + subinterval_size * (np.arange(1, num_intervals + 1) - 0)
    pc = np.power(x / (num_hop * radius), 2 * hop_to_anchor_cnt)

    x_sq = np.power(x, 2)
    acos1 = np.arccos(((x_sq + e2**2 - radius**2) / (2 * x * e2)).astype(complex))
    s1 = 0.5 * radius**2 * (2 * acos1 - np.sin(2 * acos1))
    s1 = np.real(s1)
    
    mask = (x_sq > (e2**2 - radius**2))
    cosine0 = (x_sq - e2**2 + radius**2) / (2 * x * radius)
    cosine = np.where(
        mask,
        (x_sq - e2**2 + radius**2) / (2 * x * radius),
        (e2**2 - x_sq - radius**2) / (2 * x * radius),
    ).astype(complex)
    acos2 = np.arccos(cosine)
    s2 = 0.5 * radius**2 * (2 * acos2 - np.sin(2 * acos2))
    s2 = np.real(s2)
    
    s = math.pi * e2**2
    p2 = 1 - np.power(1 - (s1 + s2) / s, 10)
    f = pc * p2
    
    # Bayesian.
    if np.sum(f) == 0:
        raise Exception(radius, e2, num_hop, hop_to_anchor_cnt, acos1, lower_bound, upper_bound, cosine0)
    f = f / np.sum(f)
    s_x = f * x
    exp_dis = np.sum(s_x)
    return exp_dis


def estimate_unknown_positions_by_least_square(
    all_node_coords,
    num_beacons,
    distances,
):
    num_unknown_nodes = all_node_coords.shape[0] - num_beacons

    a = np.zeros((num_beacons - 1, 2), dtype=float)
    for i in range(num_beacons - 1):
        for j in range(2):
            a[i, j] = all_node_coords[i, j] - all_node_coords[num_beacons - 1, j]
    a = a * 2
    
    X = np.zeros((2, num_unknown_nodes), dtype=float)
    b = np.zeros((num_beacons - 1, 1), dtype=float)
    for unknown_idx in range(num_unknown_nodes):
        for anchor_idx in range(num_beacons - 1):
            b[anchor_idx, 0] = -distances[unknown_idx, anchor_idx]**2 + distances[unknown_idx, num_beacons - 1]**2 + \
                all_node_coords[anchor_idx, 0]**2 - all_node_coords[num_beacons - 1, 0]**2 \
                + all_node_coords[anchor_idx, 1]**2 - all_node_coords[num_beacons - 1, 1]**2
        x1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(a.T, a)), a.T), b)
        X[0, unknown_idx] = x1[0, 0]
        X[1, unknown_idx] = x1[1, 0]
    
    return X


def get_distance_upper_bound_by_triangle_inequality(
    beacon_coords,
    hops,
    radius,
    eps=0.1,
):
    """According to triangle inequality, |UA1| - |UA2| <= |A1A2|. Also,
    |UA2| <= hops(U, A2) * R. Therefore, |UA1| <= |UA2| + |A1A2| <= hops(U, A2) * R + |A1A2|.
    """
    beacon_to_beacon_dist = torch.cdist(beacon_coords, beacon_coords)

    # Initialize upper bounds.
    upper_bounds = hops * radius

    # Get upper bounds between unknown node U and anchor node A1 based on the data with another anchor node A2.
    new_upper_bounds = upper_bounds.unsqueeze(1) + beacon_to_beacon_dist.unsqueeze(0)

    # Choose the minimum upper bound among all A2.
    new_upper_bounds, reference_anchor_indices = torch.min(new_upper_bounds, dim=2)

    assert torch.all(new_upper_bounds <= upper_bounds + 1e-3)

    upper_bounds = new_upper_bounds

    return upper_bounds, reference_anchor_indices
