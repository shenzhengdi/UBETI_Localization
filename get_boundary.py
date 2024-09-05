import numpy as np


def get_boundray(
    num_unknown_nodes,
):
    lower_bounds = np.zeros((2, num_unknown_nodes), dtype=float)
    upper_bounds = np.ones((2, num_unknown_nodes), dtype=float) * 100
    return lower_bounds, upper_bounds
