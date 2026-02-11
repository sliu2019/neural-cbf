import math
from typing import Dict, Any, Optional

import numpy as np


# 10-dimensional rotational domain: [gamma, beta, alpha, dgamma, dbeta, dalpha, phi, theta, dphi, dtheta]
_DEFAULT_THRESH = np.array(
    [math.pi / 3, math.pi / 3, math.pi, 20, 20, 20, math.pi / 3, math.pi / 3, 20, 20],
    dtype=np.float32,
)
_DEFAULT_X_LIM = np.concatenate((-_DEFAULT_THRESH[:, None], _DEFAULT_THRESH[:, None]), axis=1)


def approx_volume(
    param_dict: dict,
    cbf_obj,
    n_samples: int,
    x_lim: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Estimates invariant-set volume via rejection-sampling Monte Carlo.

    Samples uniformly in x_lim (rotational states; translational states fixed at 0)
    and counts the fraction inside {x : max_i phi_i(x) <= 0}.  Volume is reported as
    a percentage of the default domain volume for comparability.

    Args:
        param_dict: Must contain x_dim and x_lim.
        cbf_obj: Has phi_star_fn(x) returning (N, r+1) numpy array.
        n_samples: Number of Monte Carlo samples.
        x_lim: Optional (x_dim, 2) override for the sampling domain.

    Returns:
        Dict with key 'percent_of_domain_volume'.
    """
    x_dim = param_dict["x_dim"]
    if x_lim is None:
        x_lim = param_dict["x_lim"]

    box_side_lengths = x_lim[:, 1] - x_lim[:, 0]

    batch_size = 50
    n_inside = 0
    for _ in range(math.ceil(float(n_samples) / batch_size)):
        samples = np.random.rand(batch_size, x_dim) * box_side_lengths + x_lim[:, 0]
        # Append zeros for the 6 translational states
        samples = np.concatenate((samples, np.zeros((batch_size, 6))), axis=1)
        phi_vals = cbf_obj.phi_star_fn(samples)
        n_inside += int(np.sum(phi_vals.max(axis=1) <= 0))

    fraction_inside = float(n_inside) / n_samples
    default_volume = np.prod(_DEFAULT_X_LIM[:, 1] - _DEFAULT_X_LIM[:, 0])
    percent_of_domain_volume = fraction_inside * 100 * np.prod(box_side_lengths) / default_volume

    return {"percent_of_domain_volume": percent_of_domain_volume}
