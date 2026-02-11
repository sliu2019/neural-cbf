"""Entry point for quadcopter-pendulum CBF evaluation experiments.

Experiments
-----------
average_boundary  -- sample boundary points uniformly, measure % CBF-infeasible
worst_boundary    -- adversarial boundary search via projected gradient ascent
rollout           -- closed-loop simulations with the CBF safety filter
volume            -- invariant-set volume estimation

Usage
-----
    python -m quad_pend_analysis.run_exps \\
        --which_cbf ours \\
        --exp_name_to_load quad_pend_ESG_reg_speedup_better_counterexs_seed_0 \\
        --checkpoint_number_to_load 1020 \\
        --which_experiments average_boundary worst_boundary rollout volume \\
        --run_length medium \\
        --save_fnm my_run

Common gotchas
--------------
* param_dicts must match across compared CBFs.
* Use --run_length to set sample counts; override individual counts via the
  dedicated flags for finer control.
* --worst_boundary reuses boundary samples from --average_boundary when both
  are enabled in the same run (saves time).
"""
import argparse
import os
import pickle

import numpy as np
import torch

from quad_pend_analysis.load_cbf import load_phi_and_params
from quad_pend_analysis.volume import approx_volume, bfs_approx_volume
from quad_pend_analysis.plot_slices import plot_interesting_slices

from phi_numpy_wrapper import PhiNumpy
from flying_rollout_experiment import run_rollouts, run_rollouts_multiproc, extract_statistics
from rollout_envs.quad_pend_env import FlyingInvertedPendulumEnv
from flying_cbf_controller import CBFController

from src.critic import Critic
from src.saturation_risk import SaturationRisk
from src.problems.quad_pend import XDot, ULimitSetVertices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_saturation_risk(
    torch_phi_star_fn, param_dict: dict, device: torch.device
) -> SaturationRisk:
    """Constructs a SaturationRisk module on the given device."""
    x_dim = param_dict["x_dim"]
    u_dim = param_dict["u_dim"]
    xdot_fn = XDot(param_dict, device).to(device)
    uvertices_fn = ULimitSetVertices(param_dict, device).to(device)
    torch_phi_star_fn = torch_phi_star_fn.to(device)
    sat_risk = SaturationRisk(
        torch_phi_star_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device,
        logger=None, args=None,
    )
    return sat_risk.to(device)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_exps(args: argparse.Namespace) -> None:
    """Runs the experiments specified in args.which_experiments.

    Results are accumulated in experiment_dict and saved to a pickle file
    after each sub-experiment so that partial results are not lost on failure.
    """
    # ------------------------------------------------------------------
    # Apply run-length presets
    # ------------------------------------------------------------------
    if args.run_length == "short":
        args.boundary_n_samples = 10
        args.worst_boundary_n_samples = 10
        args.rollout_N_rollout = 10
        args.N_samp_volume = 100
    elif args.run_length == "medium":
        args.boundary_n_samples = 1000
        args.worst_boundary_n_samples = 1000
        args.rollout_N_rollout = 1000
        args.N_samp_volume = 100_000
    elif args.run_length == "long":
        args.boundary_n_samples = 10_000
        args.worst_boundary_n_samples = 10_000
        args.rollout_N_rollout = 5_000
        args.N_samp_volume = 1_000_000
        # Tighter boundary sampling for well-trained CBFs
        args.boundary_gaussian_t = 0.1
        args.worst_boundary_gaussian_t = 0.1

    # ------------------------------------------------------------------
    # Load CBF and build helper objects
    # ------------------------------------------------------------------
    device = torch.device("cpu")

    if args.which_cbf == "ours":
        torch_phi_star_fn, param_dict = load_phi_and_params(
            args.exp_name_to_load, args.checkpoint_number_to_load
        )
        numpy_phi_star_fn = PhiNumpy(torch_phi_star_fn)
        save_fldr = "./log/%s" % args.exp_name_to_load

    else:
        raise NotImplementedError(
            "--which_cbf '%s' is not supported in this version.  "
            "Only 'ours' (neural CBF) is available." % args.which_cbf
        )

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    os.makedirs(save_fldr, exist_ok=True)
    save_path = os.path.join(save_fldr, "%s_exp_data.pkl" % args.save_fnm)
    experiment_dict: dict = {"args": vars(args)}

    def _save():
        with open(save_path, "wb") as fh:
            pickle.dump(experiment_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # Build saturation risk (needed by average_boundary and worst_boundary)
    # ------------------------------------------------------------------
    saturation_risk = _build_saturation_risk(torch_phi_star_fn, param_dict, device)
    torch_x_lim = torch.tensor(param_dict["x_lim"]).to(device)

    # ------------------------------------------------------------------
    # Experiment: average_boundary
    # ------------------------------------------------------------------
    boundary_samples = None  # may be reused by worst_boundary
    if "average_boundary" in args.which_experiments:
        n_samples = args.boundary_n_samples
        critic = Critic(
            torch_x_lim, device, logger=None,
            n_samples=n_samples,
            gaussian_t=args.boundary_gaussian_t,
            projection_lr=args.boundary_projection_lr,
        )
        boundary_samples, sample_debug = critic._sample_points_on_boundary(
            torch_phi_star_fn, n_samples
        )

        obj_values = saturation_risk(boundary_samples)
        n_infeasible = int(torch.sum(obj_values > 0))
        percent_infeasible = float(n_infeasible) * 100.0 / n_samples

        infeas_idx = torch.argwhere(obj_values > 0)[:, 0]
        mean_infeas = float(torch.mean(obj_values[infeas_idx])) if len(infeas_idx) > 0 else 0.0
        std_infeas = float(torch.std(obj_values[infeas_idx])) if len(infeas_idx) > 1 else 0.0

        experiment_dict.update({
            "percent_infeasible": percent_infeasible,
            "n_infeasible": n_infeasible,
            "mean_infeasible_amount": mean_infeas,
            "std_infeasible_amount": std_infeas,
            "average_boundary_debug_dict": sample_debug,
        })
        _save()

        print("Percent infeasible: %.3f" % percent_infeasible)
        print("Mean, std infeas. amount: %.3f +/- %.3f" % (mean_infeas, std_infeas))

    # ------------------------------------------------------------------
    # Experiment: worst_boundary
    # ------------------------------------------------------------------
    if "worst_boundary" in args.which_experiments:
        n_samples = args.worst_boundary_n_samples
        n_opt_steps = args.worst_boundary_n_opt_steps

        critic = Critic(
            torch_x_lim, device, logger=None,
            n_samples=n_samples,
            max_n_steps=n_opt_steps,
            gaussian_t=args.worst_boundary_gaussian_t,
            projection_lr=args.worst_boundary_projection_lr,
        )
        # Iteration=0 uses the full step budget (no exponential decay)
        x_worst, _X, _debug = critic.opt(saturation_risk, torch_phi_star_fn, iteration=0)

        x_worst_row = x_worst.view(1, -1)
        worst_value = float(torch.max(saturation_risk(x_worst_row)).detach())

        experiment_dict.update({
            "worst_infeasible_amount": worst_value,
            "worst_x": x_worst_row.detach().cpu().numpy(),
        })
        _save()

        print("Worst infeas. amount: %.3f" % worst_value)

    # ------------------------------------------------------------------
    # Experiment: rollout
    # ------------------------------------------------------------------
    if "rollout" in args.which_experiments:
        N_rollout = args.rollout_N_rollout
        N_steps_max = int(args.rollout_T_max / args.rollout_dt)
        print("Rollout timesteps per trajectory: %i" % N_steps_max)

        model_param_dict = param_dict

        # Optional model-mismatch / noise settings
        if args.mismatched_model_parameter is not None:
            real_param_dict = param_dict.copy()
            for param, val in zip(
                args.mismatched_model_parameter,
                args.mismatched_model_parameter_true_value,
            ):
                real_param_dict[param] = val
            env = FlyingInvertedPendulumEnv(
                model_param_dict=model_param_dict,
                real_param_dict=real_param_dict,
                dynamics_noise_spread=args.dynamics_noise_spread,
            )
        else:
            env = FlyingInvertedPendulumEnv(
                model_param_dict=model_param_dict,
                dynamics_noise_spread=args.dynamics_noise_spread,
            )
        env.dt = args.rollout_dt

        cbf_controller = CBFController(env, numpy_phi_star_fn, param_dict, args)

        if N_rollout < 10:
            info_dicts = run_rollouts(env, N_rollout, N_steps_max, cbf_controller)
        else:
            info_dicts = run_rollouts_multiproc(
                env, N_rollout, N_steps_max, cbf_controller,
                verbose=True, n_proc=args.n_proc,
            )

        experiment_dict["rollout_info_dicts"] = info_dicts
        _save()

        stat_dict = extract_statistics(info_dicts, env, cbf_controller, param_dict)
        experiment_dict["rollout_stat_dict"] = stat_dict
        _save()

        for key, value in stat_dict.items():
            print("%s: %.3f" % (key, value))

    # ------------------------------------------------------------------
    # Experiment: volume
    # ------------------------------------------------------------------
    if "volume" in args.which_experiments:
        if args.volume_alg == "sample":
            vol_data = approx_volume(param_dict, numpy_phi_star_fn, args.N_samp_volume)
        elif args.volume_alg == "bfs_grid":
            assert args.bfs_axes_grid_size is not None, "--bfs_axes_grid_size required for bfs_grid"
            vol_data = bfs_approx_volume(param_dict, numpy_phi_star_fn, args.bfs_axes_grid_size)
        else:
            raise ValueError("Unknown volume_alg: %s" % args.volume_alg)

        experiment_dict.update(vol_data)
        _save()
        print("percent_of_domain_volume: %.4f" % vol_data["percent_of_domain_volume"])

    # ------------------------------------------------------------------
    # Analysis: plot_slices
    # ------------------------------------------------------------------
    if "plot_slices" in args.which_analyses:
        plot_interesting_slices(
            torch_phi_star_fn, param_dict,
            save_fldr_path=save_fldr,
            checkpoint_number=args.checkpoint_number_to_load,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation experiments for the quadcopter-pendulum neural CBF"
    )

    # Which CBF to evaluate
    parser.add_argument(
        "--which_cbf", type=str,
        choices=["ours"],
        required=True,
        help="Which CBF variant to evaluate. Currently only 'ours' (neural CBF) is supported.",
    )
    parser.add_argument("--exp_name_to_load", type=str, required=True,
                        help="Experiment subdirectory under ./log/ to load from.")
    parser.add_argument("--checkpoint_number_to_load", type=int, default=0,
                        help="Which saved checkpoint to load weights from.")
    parser.add_argument("--save_fnm", type=str, default="debug",
                        help="Short description of this run, used as filename stem.")

    # Which experiments / analyses to run
    parser.add_argument(
        "--which_experiments", nargs="+",
        default=["average_boundary", "worst_boundary", "rollout", "volume"],
        help="Subset of: average_boundary, worst_boundary, rollout, volume.",
    )
    parser.add_argument(
        "--which_analyses", nargs="+",
        default=["plot_slices"],
        help="Post-processing analyses to run. Currently: plot_slices.",
    )

    # Sample-count presets
    parser.add_argument(
        "--run_length", type=str, choices=["short", "medium", "long"],
        help="Sets boundary/volume/rollout sample counts as a bundle.",
    )

    # Fine-grained sample counts (override run_length)
    parser.add_argument("--boundary_n_samples", type=int, default=1000)
    parser.add_argument("--boundary_gaussian_t", type=float, default=1.0,
                        help="Gaussian temperature for boundary point sampling.")
    parser.add_argument("--boundary_projection_lr", type=float, default=1e-2,
                        help="Learning rate for boundary projection optimizer.")

    parser.add_argument("--worst_boundary_n_samples", type=int, default=1000)
    parser.add_argument("--worst_boundary_n_opt_steps", type=int, default=50)
    parser.add_argument("--worst_boundary_gaussian_t", type=float, default=1.0)
    parser.add_argument("--worst_boundary_projection_lr", type=float, default=1e-2)

    # Rollout settings
    parser.add_argument("--rollout_N_rollout", type=int, default=500)
    parser.add_argument("--rollout_dt", type=float, default=1e-4)
    parser.add_argument("--rollout_T_max", type=float, default=1.0)
    parser.add_argument("--rollout_u_ref", type=str,
                        choices=["unactuated", "LQR", "MPC"], default="unactuated")
    parser.add_argument("--rollout_LQR_q", type=float, default=0.1)
    parser.add_argument("--rollout_LQR_r", type=float, default=1.0)

    # Model-mismatch / noise robustness
    parser.add_argument("--dynamics_noise_spread", type=float, default=0.0,
                        help="Std dev of zero-mean Gaussian dynamics noise.")
    parser.add_argument("--mismatched_model_parameter", type=str, nargs="+",
                        help="Parameter name(s) to change in the real environment.")
    parser.add_argument("--mismatched_model_parameter_true_value", type=float, nargs="+",
                        help="True value(s) for mismatched parameters.")

    # Volume estimation
    parser.add_argument("--volume_alg", type=str,
                        choices=["sample", "bfs_grid"], default="sample")
    parser.add_argument("--N_samp_volume", type=int, default=100_000,
                        help="Number of Monte Carlo samples for volume estimation.")
    parser.add_argument("--bfs_axes_grid_size", type=float, nargs="+",
                        help="Per-dimension grid step sizes for bfs_grid volume algorithm.")

    # Parallelism
    parser.add_argument("--n_proc", type=int, default=36,
                        help="Number of worker processes for parallel rollouts.")

    args = parser.parse_args()
    run_exps(args)
