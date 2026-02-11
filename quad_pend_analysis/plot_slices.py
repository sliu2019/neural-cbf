"""2-D slice visualizations of the CBF invariant set for the quadcopter-pendulum.

Public API
----------
plot_invariant_set_slices   -- plots a grid of 2-D projections of φ*(x) ≤ 0
plot_interesting_slices     -- convenience wrapper with experiment-standard slice set
"""
import math
import os
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# Colour palette (shared between standalone and overlay modes)
_RED_RGB = np.array([199, 115, 113]) / 255.0       # outside invariant set
_PURPLE_RGB = np.array([125, 99, 167]) / 255.0     # inside invariant set (standalone)
_DARK_PURPLE_RGB = np.array([114, 81, 150]) / 255.0  # boundary contour (standalone)
_BLUE_RGB = np.array([118, 131, 202]) / 255.0      # inside invariant set (overlay)
_DARK_BLUE_RGB = np.array([92, 100, 137]) / 255.0  # boundary contour (overlay)


def _eval_phi_on_grid(
    phi_star_fn,
    input_grid: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Evaluates phi_star_fn over a pre-built input grid in batches.

    Args:
        phi_star_fn: Callable (N, x_dim) → (N, r+1) torch module.
        input_grid: (N_total, x_dim) float32 numpy array.
        batch_size: Number of rows per forward pass.

    Returns:
        phi_vals: (N_total, r+1) numpy array.
    """
    all_size = input_grid.shape[0]
    phi_vals = []
    for k in range(math.ceil(all_size / batch_size)):
        batch = input_grid[k * batch_size: min(all_size, (k + 1) * batch_size)]
        batch_torch = torch.from_numpy(batch.astype("float32"))
        batch_phi = phi_star_fn(batch_torch)
        phi_vals.append(batch_phi.detach().cpu().numpy())
    return np.concatenate(phi_vals, axis=0)


def plot_invariant_set_slices(
    phi_star_fn,
    param_dict: dict,
    samples: Optional[np.ndarray] = None,
    which_params: Optional[List[List[str]]] = None,
    constants_for_other_params: Optional[List[np.ndarray]] = None,
    fnm: Optional[str] = None,
    fldr_path: Optional[str] = None,
    pass_axs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Plots 2-D projections of the CBF invariant set φ*(x) ≤ 0.

    For each pair of state dimensions, evaluates φ* on a fine 2-D grid while
    holding all other dimensions at their specified constant values.  Pixels
    are coloured by feasibility; a contour line marks the φ* = 0 boundary.

    Modes:
      * Standalone (pass_axs=None): creates a new figure with purple/red
        colouring.  Saves to disk.
      * Overlay (pass_axs=<axes array>): draws on existing axes with blue/red
        colouring so a second CBF can be visually compared against the first.
        The figure is NOT saved — the caller is responsible.

    Args:
        phi_star_fn: NeuralPhi module (torch, evaluated in no-grad mode).
        param_dict: Problem parameter dict with keys x_lim, x_dim,
                    state_index_dict.
        samples: Optional (N, x_dim) boundary samples to scatter-plot.
        which_params: List of [param1, param2] state-name pairs to visualize.
                      Defaults to the experiment-standard set of 8 slices.
        constants_for_other_params: Per-slice (x_dim,) background state vector.
                                    Must match len(which_params) if provided.
        fnm: Filename stem for the saved PNG (standalone mode only).
        fldr_path: Directory for saved PNG (standalone mode only).
        pass_axs: Pre-existing (n_row, n_per_row) axes array for overlay mode.

    Returns:
        axs: The matplotlib axes array used for plotting.
    """
    x_lim = param_dict["x_lim"]
    x_dim = param_dict["x_dim"]
    state_index_dict = param_dict["state_index_dict"]

    # Build default slice set if not provided
    if which_params is None:
        which_params = []
        constants_for_other_params = []
        angle = math.pi / 6

        # Angle–velocity pairs for each rigid body
        for p1, p2 in [("phi", "dphi"), ("theta", "dtheta"), ("gamma", "dgamma"), ("beta", "dbeta")]:
            which_params.append([p1, p2])
            constants_for_other_params.append(np.zeros(x_dim))

        # Quad–pendulum coupling slices at zero and with angular velocity offset
        for pair in [("gamma", "phi"), ("beta", "theta")]:
            which_params.append(list(pair))
            constants_for_other_params.append(np.zeros(x_dim))
            x = np.zeros(x_dim)
            x[state_index_dict["d" + pair[0]]] = 5
            which_params.append(list(pair))
            constants_for_other_params.append(x)

    n_slices = len(which_params)
    n_per_row = 2
    n_rows = math.ceil(n_slices / n_per_row)

    if pass_axs is None:
        _, axs = plt.subplots(n_rows, n_per_row, squeeze=False, figsize=(6, 4 * n_rows))
    else:
        axs = pass_axs

    for idx, (param1, param2) in enumerate(which_params):
        row, col = divmod(idx, n_per_row)
        if row >= axs.shape[0]:
            break
        ax = axs[row, col]

        ind1 = state_index_dict[param1]
        ind2 = state_index_dict[param2]

        # Build the 2-D evaluation grid
        delta = 0.01
        xs = np.arange(x_lim[ind1, 0], x_lim[ind1, 1], delta)
        ys = np.arange(x_lim[ind2, 0], x_lim[ind2, 1], delta)[::-1]  # y-axis: top→bottom
        X_grid, Y_grid = np.meshgrid(xs, ys)

        if constants_for_other_params:
            bg = np.reshape(constants_for_other_params[idx], (1, -1))
            grid_input = np.tile(bg, (X_grid.size, 1))
        else:
            grid_input = np.zeros((X_grid.size, x_dim))
        grid_input[:, ind1] = X_grid.flatten()
        grid_input[:, ind2] = Y_grid.flatten()

        # Evaluate CBF and determine feasibility
        batch_size = max(1, int(X_grid.size / 5))
        phi_vals = _eval_phi_on_grid(phi_star_fn, grid_input, batch_size)
        S_vals = np.max(phi_vals, axis=1)        # scalar feasibility: > 0 ⟹ infeasible
        phi_signs = np.reshape(np.sign(S_vals), X_grid.shape)

        print("Slice (%s, %s): any feasible? %s" % (param1, param2, np.any(phi_signs < 0)))

        # Build RGBA image
        img = np.zeros((*phi_signs.shape, 4))
        red_inds = np.argwhere(phi_signs == 1)
        blue_inds = np.argwhere(phi_signs == -1)

        if pass_axs is None:
            # Standalone: purple/red with dark-purple contour
            img[red_inds[:, 0], red_inds[:, 1], :] = np.append(_RED_RGB, 0.5)
            img[blue_inds[:, 0], blue_inds[:, 1], :] = np.append(_PURPLE_RGB, 0.8)
            ax.imshow(img, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
            ax.contour(
                X_grid, Y_grid, np.reshape(phi_vals[:, -1], X_grid.shape),
                levels=[0.0], colors=[np.append(_DARK_PURPLE_RGB, 1.0)],
                linewidths=(2,), zorder=1,
            )
        else:
            # Overlay: blue/red with dark-blue contour
            img[red_inds[:, 0], red_inds[:, 1], :] = np.append(_RED_RGB, 0.5)
            img[blue_inds[:, 0], blue_inds[:, 1], :] = np.append(_BLUE_RGB, 0.7)
            ax.imshow(img, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
            ax.contour(
                X_grid, Y_grid, np.reshape(phi_vals[:, -1], X_grid.shape),
                levels=[0.0], colors=[np.append(_DARK_BLUE_RGB, 1.0)],
                linewidths=(2,), zorder=1,
            )

        ax.set_aspect(2.0 / ax.get_data_ratio(), adjustable="box")

        # Optional: scatter boundary samples
        if samples is not None:
            ax.scatter(samples[:, ind1], samples[:, ind2], s=0.5)

        sz = 30
        ax.set_xlabel(param1, fontsize=sz)
        ax.set_ylabel(param2, fontsize=sz)
        ax.set_aspect("equal")
        ax.tick_params(axis="x", labelsize=sz)
        ax.tick_params(axis="y", labelsize=sz)

    if pass_axs is None:
        if fnm is None:
            fnm = time.strftime("%m_%d_%H:%M:%S")
        if fldr_path is None:
            fldr_path = "./log/boundary_sampling"
        os.makedirs(fldr_path, exist_ok=True)
        save_path = os.path.join(fldr_path, fnm + ".png")
        plt.savefig(save_path, bbox_inches="tight")
        print("Saved slice plot: %s" % save_path)

    return axs


def plot_interesting_slices(
    phi_star_fn,
    param_dict: dict,
    save_fldr_path: str,
    checkpoint_number: int,
) -> None:
    """Generates the standard set of invariant-set slices used in experiments.

    Produces a single multi-panel PNG covering:
      - Per-axis angle/velocity pairs (phi/dphi, theta/dtheta, gamma/dgamma, beta/dbeta)
      - Cross-velocity coupling (dtheta/dbeta, dphi/dgamma)
      - Quad–pendulum alignment effects at pi/6 offset (theta/beta, phi/gamma)

    Args:
        phi_star_fn: Trained NeuralPhi module.
        param_dict: Problem parameter dict.
        save_fldr_path: Directory where the PNG will be saved.
        checkpoint_number: Used in the output filename for traceability.
    """
    state_index_dict = param_dict["state_index_dict"]
    x_dim = param_dict["x_dim"]
    angle = math.pi / 6

    which_params = []
    constants = []

    # Angle–velocity slices
    for p1, p2 in [("phi", "dphi"), ("theta", "dtheta"), ("gamma", "dgamma"), ("beta", "dbeta")]:
        which_params.append([p1, p2])
        constants.append(np.zeros(x_dim))

    # Cross-axis velocity coupling
    for p1, p2 in [("dtheta", "dbeta"), ("dphi", "dgamma")]:
        which_params.append([p1, p2])
        constants.append(np.zeros(x_dim))

    # Coupled quad–pendulum slices (misaligned then aligned)
    # theta/beta: vary phi and gamma as background constants
    for phi_sign in [-1, 1]:
        x = np.zeros(x_dim)
        x[state_index_dict["phi"]] = phi_sign * angle
        x[state_index_dict["gamma"]] = angle
        which_params.append(["theta", "beta"])
        constants.append(x)

    # phi/gamma: vary theta and beta as background constants
    for theta_sign in [-1, 1]:
        x = np.zeros(x_dim)
        x[state_index_dict["theta"]] = theta_sign * angle
        x[state_index_dict["beta"]] = angle
        which_params.append(["phi", "gamma"])
        constants.append(x)

    fnm = "slices_ckpt_%i" % checkpoint_number
    plot_invariant_set_slices(
        phi_star_fn, param_dict,
        which_params=which_params,
        constants_for_other_params=constants,
        fnm=fnm,
        fldr_path=save_fldr_path,
    )
    plt.clf()
    plt.close()
