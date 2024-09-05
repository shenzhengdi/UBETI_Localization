import numpy as np


INF_HOP = 100000

def compute_minimum_hops_from_distances(
    gt_distances_all_nodes,
    radius,
):
    num_all_nodes = gt_distances_all_nodes.shape[0]
    hops = np.zeros((num_all_nodes, num_all_nodes), dtype=int)
    
    # Initialize hops.
    for node_idx1 in range(num_all_nodes):
        for node_idx2 in range(num_all_nodes):
            if gt_distances_all_nodes[node_idx1, node_idx2] <= radius and gt_distances_all_nodes[node_idx1, node_idx2] > 0:
                hops[node_idx1, node_idx2] = 1
            elif node_idx1 == node_idx2:
                hops[node_idx1, node_idx2] = 0
            else:
                hops[node_idx1, node_idx2] = INF_HOP
    
    # Compute minimum hops.
    for k in range(num_all_nodes):
        for i in range(num_all_nodes):
            for j in range(num_all_nodes):
                if hops[i, j] == -1 or hops[i, k] + hops[k, j] < hops[i, j]:
                    hops[i, j] = hops[i, k] + hops[k, j]

    return hops
