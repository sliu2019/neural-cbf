"""Invariant-set volume estimation for the quadcopter-pendulum CBF.

Two algorithms are provided:
  * approx_volume   — rejection-sampling Monte Carlo estimate
  * bfs_approx_volume — BFS on a uniform grid starting from the origin cell
"""
import math
from queue import Queue
from typing import Dict, Any, Optional, List

import numpy as np


# Default state-space domain used for normalising volume fractions.
# 10-dimensional rotational state: [gamma, beta, alpha, dgamma, dbeta, dalpha,
#                                   phi, theta, dphi, dtheta]
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
    """Estimates invariant-set volume via rejection sampling.

    Samples uniformly in the x_lim box (rotational states only; translational
    states are fixed at zero) and counts the fraction inside the safe set
    {x : max_i phi_i(x) <= 0}.

    Volume is reported as a percentage of the default domain volume so that
    results are comparable across methods with different x_lim choices.

    Args:
        param_dict: Problem parameter dictionary (must contain x_dim, x_lim).
        cbf_obj: Object with a phi_star_fn(x) method returning (N, r+1) numpy array.
        n_samples: Total number of Monte Carlo samples.
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
        # Sample uniformly inside the bounding box (rotational states only)
        samples = np.random.rand(batch_size, x_dim) * box_side_lengths + x_lim[:, 0]
        # Append zeros for the 6 translational states expected by phi_star_fn
        samples = np.concatenate((samples, np.zeros((batch_size, 6))), axis=1)

        phi_vals = cbf_obj.phi_star_fn(samples)
        max_phi_vals = phi_vals.max(axis=1)
        n_inside += int(np.sum(max_phi_vals <= 0))

    fraction_inside = float(n_inside) / n_samples
    default_volume = np.prod(_DEFAULT_X_LIM[:, 1] - _DEFAULT_X_LIM[:, 0])
    percent_of_domain_volume = (
        fraction_inside * 100 * np.prod(box_side_lengths) / default_volume
    )

    return {"percent_of_domain_volume": percent_of_domain_volume}


def bfs_approx_volume(
    param_dict: dict,
    cbf_obj,
    axes_grid_size: List[float],
) -> Dict[str, Any]:
    """Estimates invariant-set volume via BFS on a uniform grid.

    Starting from the origin cell (assumed safe), explores all grid-adjacent
    cells that are inside the safe set via breadth-first search.

    Args:
        param_dict: Problem parameter dictionary (must contain x_dim, x_lim).
        cbf_obj: Object with a phi_star_fn(x) method returning (N, r+1) numpy array.
        axes_grid_size: Per-dimension step size for the grid (length x_dim).

    Returns:
        Dict with keys: n_cells_occupied, total_absolute_volume,
                        cell_absolute_volume, percent_of_domain_volume.
    """
    x_dim = param_dict["x_dim"]
    x_lim = param_dict["x_lim"]

    queue: Queue = Queue()
    visited: set = set()

    start_node = tuple(np.zeros(x_dim))
    queue.put(start_node)
    visited.add(start_node)

    def _children(node: tuple):
        """Returns grid-adjacent nodes within x_lim that haven't been visited."""
        np_node = np.reshape(np.array(node), (1, -1))
        candidates = np.tile(np_node, (2 * x_dim, 1))
        for i in range(x_dim):
            candidates[2 * i, i] -= axes_grid_size[i]
            candidates[2 * i + 1, i] += axes_grid_size[i]

        in_domain = np.logical_and(
            np.all(candidates > x_lim[:, 0], axis=1),
            np.all(candidates < x_lim[:, 1], axis=1),
        )
        return [tuple(c) for c in candidates[np.nonzero(in_domain)[0]].tolist()]

    n_cells_occupied = 0
    while not queue.empty():
        current_node = queue.get()
        np_node = np.reshape(np.array(current_node), (1, -1))
        phi_vals = cbf_obj.phi_star_fn(np_node)
        max_phi_val = phi_vals.max(axis=1)

        if max_phi_val <= 0:
            n_cells_occupied += 1
            for child in _children(current_node):
                if child not in visited:
                    queue.put(child)
                    visited.add(child)

    cell_volume = np.prod(axes_grid_size)
    total_volume = n_cells_occupied * cell_volume
    domain_volume = np.prod(x_lim[:, 1] - x_lim[:, 0])
    percent_of_domain_volume = total_volume / domain_volume

    return {
        "n_cells_occupied": n_cells_occupied,
        "total_absolute_volume": total_volume,
        "cell_absolute_volume": cell_volume,
        "percent_of_domain_volume": percent_of_domain_volume,
    }
