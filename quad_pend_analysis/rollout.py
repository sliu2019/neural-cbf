"""Closed-loop rollout infrastructure for the quadcopter-pendulum CBF.

Functions
---------
sample_inside_safe_set      -- rejection sampling of initial states inside the CBF safe set
simulate_rollout            -- single closed-loop trajectory
run_rollouts                -- sequential rollout collection
run_rollouts_multiproc      -- parallel rollout collection
extract_statistics          -- aggregates per-rollout dicts into scalar metrics
convert_angle_to_negpi_pi   -- wraps an angle to (-π, π]
"""
import multiprocessing as mp
from typing import Optional, Tuple

import numpy as np
import torch


# Fixed seed for repeatability across sequential runs
torch.manual_seed(2022)
np.random.seed(2022)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_inside_safe_set(
    param_dict: dict, cbf_obj, target_n: int
) -> Tuple[np.ndarray, float]:
    """Rejection-samples initial states uniformly inside the CBF safe set.

    Samples uniformly in the rotational state bounding box (x_lim) and appends
    zeros for the 6 translational states.  Repeats until target_n safe samples
    are found.

    Args:
        param_dict: Must contain x_dim and x_lim.
        cbf_obj: PhiNumpy instance; phi_star_fn(x) → (N, r+1) array.
        target_n: Number of safe initial states to return.

    Returns:
        x0s: (target_n, 16) array of safe initial states.
        percent_inside: Fraction of all sampled points that were inside (×100).
    """
    x_dim = param_dict["x_dim"]
    x_lim = param_dict["x_lim"]
    box_side_lengths = x_lim[:, 1] - x_lim[:, 0]

    x0s = np.empty((0, 16))
    batch_size = 1000
    n_found = 0
    n_total_sampled = 0

    while n_found < target_n:
        samples = np.random.rand(batch_size, x_dim) * box_side_lengths + x_lim[:, 0]
        # Translational states are not part of the CBF domain; fix at zero
        samples_full = np.concatenate((samples, np.zeros((batch_size, 6))), axis=1)

        phi_vals = cbf_obj.phi_star_fn(samples_full)
        safe_mask = phi_vals.max(axis=1) <= 0
        safe_samples = samples_full[safe_mask]

        x0s = np.concatenate((x0s, safe_samples), axis=0)
        n_found += int(np.sum(safe_mask))
        n_total_sampled += batch_size

    x0s = x0s[:target_n]
    percent_inside = float(n_found) / n_total_sampled * 100.0
    return x0s, percent_inside


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def convert_angle_to_negpi_pi(angle: np.ndarray) -> np.ndarray:
    """Wraps angle(s) to the interval (-π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def simulate_rollout(
    env, n_steps_max: int, cbf_controller, x0: np.ndarray,
    random_seed: Optional[int] = None,
) -> dict:
    """Simulates a single closed-loop trajectory under the CBF safety filter.

    The rollout terminates after n_steps_max steps or after the safe control
    has been applied and the system has returned to the interior for 1 step
    (whichever comes first).

    Args:
        env: QuadPendEnv instance.
        n_steps_max: Maximum number of integration steps.
        cbf_controller: CBFController instance.
        x0: Initial state (1, 16).
        random_seed: Optional seed for reproducibility in parallel runs.

    Returns:
        info_dict: Dict of numpy arrays, one entry per timestep, with keys:
            x, u, apply_u_safe, inside_boundary, on_boundary,
            outside_boundary, phi_vals, qp_slack, qp_rhs, qp_lhs,
            impulses, dist_between_xs, phi_grad_mag, phi_grad.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    x = x0
    xs = [x]
    us = []
    step_data = None

    # Cyclical angle indices that need wrapping
    cyclic_inds = [0, 1, 2, 6, 7]

    u_safe_applied = False
    t_since_safe = 0

    for t in range(n_steps_max):
        u, debug = cbf_controller.compute_control(t, x)
        x_dot = env.x_dot_open_loop(x, u)
        x = x + env.dt * x_dot

        # Wrap cyclical angles
        x[:, cyclic_inds] = convert_angle_to_negpi_pi(x[:, cyclic_inds])
        u = np.reshape(u, (4,))
        x = np.reshape(x, (1, 16))

        us.append(u)
        xs.append(x)

        if step_data is None:
            step_data = {k: [v] for k, v in debug.items()}
        else:
            for k, v in step_data.items():
                v.append(debug[k])

        if debug["apply_u_safe"]:
            u_safe_applied = True

        if u_safe_applied:
            t_since_safe += 1
        if t_since_safe > 1:
            break

    info_dict = {k: np.array(v) for k, v in step_data.items()}
    info_dict["x"] = np.concatenate([np.reshape(xi, (1, 16)) for xi in xs], axis=0)
    info_dict["u"] = np.array(us)
    return info_dict


# ---------------------------------------------------------------------------
# Multi-rollout runners
# ---------------------------------------------------------------------------

def run_rollouts(env, n_desired: int, n_steps_max: int, cbf_controller) -> dict:
    """Sequentially collects n_desired valid rollouts.

    A rollout is valid if the safe control was triggered at least once
    (i.e., the trajectory reached or crossed the CBF boundary).

    Args:
        env: QuadPendEnv instance.
        n_desired: Number of valid rollouts to collect.
        n_steps_max: Max steps per rollout.
        cbf_controller: CBFController instance.

    Returns:
        info_dicts: Dict mapping key → list of per-rollout arrays.
    """
    info_dicts = None
    n_collected = 0

    while n_collected < n_desired:
        print("Rollout %i / %i" % (n_collected, n_desired))
        x0, _ = sample_inside_safe_set(cbf_controller.param_dict, cbf_controller.cbf_obj, 1)
        info_dict = simulate_rollout(env, n_steps_max, cbf_controller, x0)

        # Skip rollouts that never touched the boundary
        if not np.any(info_dict["apply_u_safe"]):
            continue

        if info_dicts is None:
            info_dicts = {k: [v] for k, v in info_dict.items()}
        else:
            for k, v in info_dicts.items():
                v.append(info_dict[k])
        n_collected += 1

    return info_dicts


def run_rollouts_multiproc(
    env, n_desired: int, n_steps_max: int, cbf_controller,
    verbose: bool = False, n_proc: Optional[int] = None,
) -> dict:
    """Collects n_desired valid rollouts using a multiprocessing pool.

    Uses the 'spawn' start method to avoid fork-related issues with some
    backends (e.g., CUDA).

    Args:
        env: QuadPendEnv instance.
        n_desired: Number of valid rollouts to collect.
        n_steps_max: Max steps per rollout.
        cbf_controller: CBFController instance.
        verbose: Print running total of collected rollouts if True.
        n_proc: Number of worker processes (defaults to cpu_count).

    Returns:
        info_dicts: Dict mapping key → list of per-rollout arrays.
    """
    if n_proc is None:
        n_proc = mp.cpu_count()

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(n_proc)

    base_args = [env, n_steps_max, cbf_controller]
    info_dicts = None
    n_collected = 0

    while n_collected < n_desired:
        # Sample a fresh batch of start states for this worker batch
        x0s, _ = sample_inside_safe_set(
            cbf_controller.param_dict, cbf_controller.cbf_obj, n_proc
        )
        batch_args = [base_args + [x0s[i]] for i in range(n_proc)]
        results = pool.starmap(simulate_rollout, batch_args)

        for info_dict in results:
            if not np.any(info_dict["apply_u_safe"]):
                continue
            if n_collected >= n_desired:
                break

            if info_dicts is None:
                info_dicts = {k: [v] for k, v in info_dict.items()}
            else:
                for k, v in info_dicts.items():
                    v.append(info_dict[k])
            n_collected += 1

        if verbose:
            print("Collected %i / %i rollouts" % (n_collected, n_desired))

    print("Collected %i rollouts (target: %i)" % (n_collected, n_desired))
    return info_dicts


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def extract_statistics(info_dicts: dict, env, cbf_controller, param_dict: dict) -> dict:
    """Aggregates per-rollout data into scalar summary statistics.

    Computed metrics
    ----------------
    Boundary crossing fractions:
        percent_on_in / percent_on_out / percent_on_on
        N_transitions, N_on_in, N_on_out, N_on_on

    CBF-boundary state properties:
        min_phi, mean_phi          — φ* value at boundary crossings
        mean_dist, max_dist        — ||x_{t+1} - x_t|| at boundary
        mean_phi_grad, max_phi_grad — ||∇φ*(x)|| at boundary

    Box-exit fractions (state leaves the training domain):
        percent_on_in_outside_box / percent_on_out_outside_box / percent_on_on_outside_box
        N_on_in_outside_box / …

    QP violation statistics:
        mean_violation_amount, std_violation_amount

    Per-state box exit counts:
        N_count_exit_on_<state_name> for each state dimension

    Args:
        info_dicts: Output of run_rollouts / run_rollouts_multiproc.
        env: QuadPendEnv instance.
        cbf_controller: CBFController instance.
        param_dict: Problem parameter dict (x_lim, state_index_dict).

    Returns:
        stat_dict: Dict of scalar metrics.
    """
    stat_dict = {}
    x_lim = param_dict["x_lim"]
    N_rollout = len(info_dicts["apply_u_safe"])

    inside  = info_dicts["inside_boundary"]
    on      = info_dicts["on_boundary"]
    outside = info_dicts["outside_boundary"]

    # Count boundary-crossing transitions: on→in, on→out, on→on
    on_in_rl  = [on[i][:-1] * inside[i][1:]  for i in range(N_rollout)]
    on_out_rl = [on[i][:-1] * outside[i][1:] for i in range(N_rollout)]
    on_on_rl  = [on[i][:-1] * on[i][1:]      for i in range(N_rollout)]

    on_in_count  = int(np.sum([np.sum(r) for r in on_in_rl]))
    on_out_count = int(np.sum([np.sum(r) for r in on_out_rl]))
    on_on_count  = int(np.sum([np.sum(r) for r in on_on_rl]))
    total = on_in_count + on_out_count + on_on_count
    print("Total transitions: %i  (N rollouts: %i)" % (total, N_rollout))

    stat_dict["N_transitions"]  = total
    stat_dict["percent_on_in"]  = on_in_count  / float(total) * 100
    stat_dict["percent_on_out"] = on_out_count / float(total) * 100
    stat_dict["percent_on_on"]  = on_on_count  / float(total) * 100
    stat_dict["N_on_in"]  = on_in_count
    stat_dict["N_on_out"] = on_out_count
    stat_dict["N_on_on"]  = on_on_count

    # φ* values at the boundary
    phis     = [rl[:, -1] for rl in info_dicts["phi_vals"]]
    gap_phis = [phis[i] * on[i] for i in range(N_rollout)]
    on_total = float(np.sum([np.sum(r) for r in on]))
    stat_dict["min_phi"]  = float(np.min([np.min(r) for r in gap_phis]))
    stat_dict["mean_phi"] = float(np.sum([np.sum(r) for r in gap_phis]) / on_total)

    # Dynamics step-size at boundary
    dist = info_dicts["dist_between_xs"]
    dist_on = [dist[i] * on[i] for i in range(N_rollout)]
    stat_dict["mean_dist"] = float(np.sum([np.sum(r) for r in dist_on]) / on_total)
    stat_dict["max_dist"]  = float(np.max([np.max(r) for r in dist_on]))

    # CBF gradient magnitude at boundary
    grad_mag = info_dicts["phi_grad_mag"]
    grad_on  = [grad_mag[i] * on[i] for i in range(N_rollout)]
    stat_dict["mean_phi_grad"] = float(np.sum([np.sum(r) for r in grad_on]) / on_total)
    stat_dict["max_phi_grad"]  = float(np.max([np.max(r) for r in grad_on]))

    # Box-exit analysis
    xs = info_dicts["x"]
    out_box = [
        np.logical_or(
            np.any(rl[:, :10] < x_lim[:, 0], axis=1),
            np.any(rl[:, :10] > x_lim[:, 1], axis=1),
        )
        for rl in xs
    ]

    def _count_box_exits(transition_rl):
        return int(np.sum([np.sum(out_box[i][:-2] * transition_rl[i]) for i in range(N_rollout)]))

    on_in_box  = _count_box_exits(on_in_rl)
    on_out_box = _count_box_exits(on_out_rl)
    on_on_box  = _count_box_exits(on_on_rl)

    stat_dict["percent_on_in_outside_box"]  = on_in_box  / float(max(1, on_in_count))  * 100
    stat_dict["percent_on_out_outside_box"] = on_out_box / float(max(1, on_out_count)) * 100
    stat_dict["percent_on_on_outside_box"]  = on_on_box  / float(max(1, on_on_count))  * 100
    stat_dict["N_on_in_outside_box"]  = on_in_box
    stat_dict["N_on_out_outside_box"] = on_out_box
    stat_dict["N_on_on_outside_box"]  = on_on_box

    # Per-state exit counts
    out_box_states = [
        np.logical_or(rl[:, :10] < x_lim[:, 0], rl[:, :10] > x_lim[:, 1])
        for rl in xs
    ]
    state_inds = np.concatenate([np.argwhere(rl)[:, 1] for rl in out_box_states])
    state_id_to_name = {v: k for k, v in param_dict["state_index_dict"].items()}
    for val, cnt in zip(*np.unique(state_inds, return_counts=True)):
        stat_dict["N_count_exit_on_%s" % state_id_to_name[val]] = int(cnt)

    # QP constraint violation amounts (only for on→out transitions)
    qp_slacks = info_dicts["qp_slack"]
    violation_amounts = []
    for i in range(N_rollout):
        if np.any(on_out_rl[i]):
            slack = qp_slacks[i][:-1].astype(float)
            slack[slack == None] = 0.0
            violation_amounts.append(float(np.sum(on_out_rl[i] * slack)))
    stat_dict["mean_violation_amount"] = float(np.mean(violation_amounts)) + cbf_controller.eps_bdry
    stat_dict["std_violation_amount"]  = float(np.std(violation_amounts))

    return stat_dict
