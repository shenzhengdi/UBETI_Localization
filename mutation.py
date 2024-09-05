import torch


def mutation(
    POP,
    upper_bounds,
    lower_bounds,
    mp,
):
    eta_m = 20

    population_size = POP.size(0)

    NPOP = POP.clone()
    r1 = torch.rand(POP.shape)
    mutation_mask = (r1 <= mp)
    
    NPOP[mutation_mask] = _get_mutations_at_masked_data(
        NPOP[mutation_mask],
        upper_bounds.unsqueeze(0).expand(population_size, -1, -1)[mutation_mask],
        lower_bounds.unsqueeze(0).expand(population_size, -1, -1)[mutation_mask],
        eta_m=eta_m,
    )
    
    return NPOP


def _get_mutations_at_masked_data(
    features,
    upper_bounds,
    lower_bounds,
    eta_m,
):
    feature_larger_than_lb_mask = features > lower_bounds
    
    # If feature is larger than lower bound, update the feature as follows:
    features[feature_larger_than_lb_mask] = _update_features_larger_than_lower_bound(
        features=features[feature_larger_than_lb_mask], 
        upper_bounds=upper_bounds[feature_larger_than_lb_mask],
        lower_bounds=lower_bounds[feature_larger_than_lb_mask],
        eta_m=eta_m,
    )

    # If feature is smaller than lower bound, update the feature as follows:
    features[~feature_larger_than_lb_mask] = _update_features_smaller_than_lower_bound(
        upper_bounds=upper_bounds[~feature_larger_than_lb_mask],
        lower_bounds=lower_bounds[~feature_larger_than_lb_mask],
    )
    
    return features


def _update_features_larger_than_lower_bound(
    features,
    upper_bounds,
    lower_bounds,
    eta_m,
):
    r = torch.rand(features.shape)
    indi = 1 / (eta_m + 1)

    delta = torch.minimum(features - lower_bounds, upper_bounds - features) \
        / (upper_bounds - lower_bounds)
    
    xy = 1 - delta
    pw = torch.pow(xy, eta_m + 1)
    
    val = torch.where(
        r <= 0.5,
        2 * r + (1 - 2 * r) * pw,
        2 * (1 - r) + 2 * (r - 0.5) * pw,
    )
    val_pow_indi = torch.pow(val, indi)
    deltaq = torch.where(
        r <= 0.5,
        val_pow_indi - 1,
        1 - val_pow_indi,
    )
    
    features += deltaq * (upper_bounds - lower_bounds)
    
    out_of_bound_mask = (features > upper_bounds) | (features < lower_bounds)
    features[out_of_bound_mask] = torch.rand(out_of_bound_mask.sum()) * (
        upper_bounds - lower_bounds)[out_of_bound_mask] + lower_bounds[out_of_bound_mask]
    
    return features

    
def _update_features_smaller_than_lower_bound(
    upper_bounds,
    lower_bounds,
):
    r = torch.rand(upper_bounds.shape)
    return r * (upper_bounds - lower_bounds) + lower_bounds
