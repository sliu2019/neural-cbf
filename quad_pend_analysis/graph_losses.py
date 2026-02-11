"""Training-curve and loss visualizations for quadcopter-pendulum CBF experiments.

Functions
---------
graph_losses          -- full multi-panel diagnostic plot for one experiment run
plot_training_losses  -- compact multi-seed training-curve figure (for paper)
"""
import datetime
import os
import pickle
import types

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data_and_args(exp_name: str):
    """Loads the pickled data dict and reconstructs an args namespace."""
    data_path = "./log/%s/data.pkl" % exp_name
    args_path = "./log/%s/args.txt" % exp_name
    import json
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(args_path, "r") as f:
        args = types.SimpleNamespace(**json.load(f))
    return data, args


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def graph_losses(exp_name: str, debug: bool = True) -> None:
    """Plots training and test metrics for a single experiment run.

    Saves three PNG files under ./log/<exp_name>/:
      * <exp_name>_loss.png          — train/test losses vs iteration
      * <exp_name>_loss_vs_time.png  — test losses vs wall-clock time
      * <exp_name>_timing_debug.png  — critic timing breakdown (if debug=True)

    Data key convention (post-refactor):
      train_attack_losses            — worst counterexample value per iteration
      train_attack_t_total_opt       — total critic time per iteration
      train_attack_t_init            — boundary sampling time
      train_attack_t_grad_step       — per-step grad-ascent times (list of lists)
      train_attack_t_reproject       — per-step reprojection times (list of lists)
      train_attack_n_segments_sampled
      train_attack_n_opt_steps
      train_attack_dist_diff_after_proj
      train_attack_final_best_counterex_value
      train_attack_init_best_counterex_value

    Args:
        exp_name: Experiment name (subdirectory under ./log/).
        debug: If True, also save timing and critic-quality debug plots.
    """
    data, args = _load_data_and_args(exp_name)
    save_dir = "./log/%s" % exp_name

    n_it_so_far = len(data["train_losses"]) - 1

    # ------------------------------------------------------------------
    # Figure 1: train and test losses vs iteration
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(
        "Metrics for %s at iteration %i/%i" % (exp_name, n_it_so_far, args.learner_n_steps),
        fontsize=8,
    )

    # Train panel
    axs[0].set_title("Train metrics")
    axs[0].plot(data["train_losses"], label="objective value", linewidth=0.5)
    axs[0].plot(data["train_attack_losses"], label="worst counterex value", linewidth=0.5)
    axs[0].plot(data["train_reg_losses"], label="reg value", linewidth=0.5)
    axs[0].axhline(0, color="k", linewidth=0.5)
    axs[0].legend(loc=(1.04, 0))

    # Test panel
    axs[1].set_title("Test metrics")
    per_n = args.n_test_loss_step
    test_iters = np.arange(0, n_it_so_far, per_n)

    approx_v = np.array(data["V_approx_list"])
    axs[1].plot(test_iters, approx_v, label="approx. volume", color="green", linewidth=1.0)

    boundary_obj_vals = data["boundary_samples_obj_values"]
    pct_infeas = [
        np.sum(v > 0) * 100.0 / v.size for v in boundary_obj_vals
    ]
    avg_infeas = np.array([np.mean((v > 0) * v) for v in boundary_obj_vals])
    std_infeas = np.array([np.std((v > 0) * v) for v in boundary_obj_vals])

    n = min(len(test_iters), len(pct_infeas))
    axs[1].plot(test_iters[:n], pct_infeas[:n], label="% infeas. at boundary", color="blue", linewidth=1.0)
    axs[1].fill_between(
        test_iters[:n],
        avg_infeas[:n] - std_infeas[:n],
        avg_infeas[:n] + std_infeas[:n],
        alpha=0.5, color="orange",
    )
    axs[1].plot(test_iters[:n], avg_infeas[:n], label="avg infeas.", linewidth=0.5, color="orange")
    axs[1].set_ylim([-5, 40])
    axs[1].legend(loc=(1.04, 0))

    fig.tight_layout()
    plt.xlabel("Iterations")
    plt.savefig(os.path.join(save_dir, "%s_loss.png" % exp_name))
    plt.clf()
    plt.cla()

    # ------------------------------------------------------------------
    # Figure 2: test losses vs wall-clock time
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1)
    ax.set_title(
        "Metrics for %s at iteration %i/%i" % (exp_name, n_it_so_far, args.learner_n_steps)
    )
    train_times = np.array(data["train_loop_times"])[::per_n][: len(pct_infeas)] / 3600.0

    ax.plot(train_times, approx_v[: len(train_times)], label="approx. volume", color="green", linewidth=1.0)
    ax.plot(train_times, pct_infeas[: len(train_times)], label="% infeas. at boundary", color="blue", linewidth=1.0)
    ax.plot(train_times, avg_infeas[: len(train_times)], label="avg infeas.", linewidth=0.5, color="orange")
    ax.legend(loc="upper right")
    plt.xlabel("Clock time (hours)")
    plt.savefig(os.path.join(save_dir, "%s_loss_vs_time.png" % exp_name))
    plt.clf()
    plt.cla()

    # ------------------------------------------------------------------
    # Summary statistics to stdout
    # ------------------------------------------------------------------
    print("Average approx volume: %.3f" % np.mean(approx_v))
    pct_infeas_thresh = 1.0
    below_thresh = np.argwhere(np.array(pct_infeas) < pct_infeas_thresh).flatten()
    print("Checkpoints with < %.1f%% infeasible at boundary:" % pct_infeas_thresh)
    for k in range(min(len(below_thresh), 10)):
        idx = int(np.sort(below_thresh)[k])
        real_iter = idx * per_n
        t = datetime.timedelta(seconds=data["train_loop_times"][real_iter])
        print("  ckpt %i: %.3f%% infeas., avg infeas %.3f, train loss %.3f  [%s]" % (
            real_iter, pct_infeas[idx], avg_infeas[idx],
            data["train_attack_losses"][real_iter], t,
        ))

    if not debug:
        return

    # ------------------------------------------------------------------
    # Figure 3: critic timing debug
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle(
        "Timing debug for %s at iteration %i/%i" % (exp_name, n_it_so_far, args.learner_n_steps),
        fontsize=8,
    )
    axs[0].set_title("Test metric timing")
    axs[0].plot(test_iters, data["test_t_total"], label="Test metric time", linewidth=0.5)
    axs[0].legend(loc=(1.04, 0))

    axs[1].set_title("Critic timing")
    axs[1].plot(data["train_attack_t_total_opt"], label="Total critic time", linewidth=0.5)
    axs[1].plot(data["train_attack_t_init"], label="Boundary sampling time", linewidth=0.5)
    axs[1].plot(
        [np.mean(x) for x in data["train_attack_t_grad_step"]],
        label="Avg grad step time", linewidth=0.5,
    )
    axs[1].plot(
        [np.mean(x) for x in data["train_attack_t_reproject"]],
        label="Avg reproject time", linewidth=0.5,
    )
    axs[1].legend(loc=(1.04, 0))

    axs[2].set_title("Critic quality")
    axs[2].plot(data["train_attack_n_segments_sampled"], label="Segments sampled", linewidth=0.5)
    axs[2].plot(data["train_attack_n_opt_steps"], label="Opt steps", linewidth=0.5)
    axs[2].legend(loc=(1.04, 0))

    fig.tight_layout()
    plt.xlabel("Iterations")
    plt.savefig(os.path.join(save_dir, "%s_timing_debug.png" % exp_name))
    plt.clf()
    plt.cla()

    # ------------------------------------------------------------------
    # Figure 4: counterexample improvement and gradient norms
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle(
        "Debug for %s at iteration %i/%i" % (exp_name, n_it_so_far, args.learner_n_steps),
        fontsize=8,
    )
    axs[0].set_title("Counterex improvement after critic opt")
    diff = (
        np.array(data["train_attack_final_best_counterex_value"])
        - np.array(data["train_attack_init_best_counterex_value"])
    )
    axs[0].plot(diff, linewidth=0.5)
    axs[0].axhline(0, color="red", linewidth=0.5)

    axs[1].set_title("Reproj constraint improvement per step")
    avg_dist_diff = [np.mean(x) for x in data["train_attack_dist_diff_after_proj"]]
    axs[1].plot(avg_dist_diff, linewidth=0.5)

    axs[2].set_title("CBF gradient magnitudes")
    axs[2].plot(data["reg_grad_norms"], linewidth=0.5, label="Reg loss")
    axs[2].plot(data["grad_norms"], linewidth=0.5, label="Combined loss")
    axs[2].legend(loc=(1.04, 0))

    fig.tight_layout()
    plt.xlabel("Iterations")
    plt.savefig(os.path.join(save_dir, "%s_debug.png" % exp_name))
    plt.clf()
    plt.cla()

    # ------------------------------------------------------------------
    # Figure 5: standalone gradient-norm plot
    # ------------------------------------------------------------------
    plt.title("CBF gradient magnitudes")
    plt.plot(data["reg_grad_norms"], linewidth=0.5, label="Reg loss")
    plt.plot(data["grad_norms"], linewidth=0.5, label="Combined loss")
    plt.legend(loc=(1.04, 0))
    plt.xlabel("Iterations")
    plt.savefig(os.path.join(save_dir, "%s_grad_norms.png" % exp_name))
    plt.clf()
    plt.cla()


def plot_training_losses(
    exp_names=None,
    last_checkpoint_numbers=None,
    save_path="./log/corl/sat_risk_for_all_seeds_subfigs.svg",
) -> None:
    """Plots training curves for multiple seeds on aligned subplots.

    Each subplot shows:
      - Dashed: training saturation risk (worst counterex value)
      - Solid:  test metric (% infeasible states on boundary)

    Args:
        exp_names: List of experiment names. Defaults to the 5-seed CoRL runs.
        last_checkpoint_numbers: Per-seed iteration to truncate at.
        save_path: Where to save the figure.
    """
    if exp_names is None:
        exp_names = [
            "quad_pend_ESG_reg_speedup_better_counterexs_seed_%i" % i for i in range(5)
        ]
    if last_checkpoint_numbers is None:
        last_checkpoint_numbers = [250, 400, 550, 300, 175]

    n_seeds = len(exp_names)
    plt.figure(dpi=1200)
    fig, axs = plt.subplots(n_seeds, sharex=True)
    fig.suptitle("Losses throughout training for %i random seeds" % n_seeds)
    fig.text(0.5, 0.04, "Training iteration", ha="center", va="center")
    fig.text(0.06, 0.5, "Train, test losses", ha="center", va="center", rotation="vertical")

    colors = plt.rcParams["axes.prop_cycle"]()
    x_ub = max(last_checkpoint_numbers) + 50
    all_times = []

    for i, (exp_name, last_ckpt) in enumerate(zip(exp_names, last_checkpoint_numbers)):
        data, args = _load_data_and_args(exp_name)
        c = next(colors)["color"]

        axs[i].set_xlim([0, x_ub])

        # Training loss (dashed)
        axs[i].plot(
            data["train_attack_losses"][:last_ckpt],
            linewidth=1.0, color=c, linestyle="--",
        )

        # Test metric (solid): % infeasible at boundary
        per_n = args.n_test_loss_step
        test_iters = np.arange(0, last_ckpt + per_n, per_n)
        boundary_obj_vals = data["boundary_samples_obj_values"]
        pct_infeas = [
            np.sum(v > 0) * 100.0 / v.size for v in boundary_obj_vals
        ][: len(test_iters)]
        axs[i].plot(test_iters, pct_infeas, linewidth=1.0, color=c, linestyle="-")

        # Zero reference line
        axs[i].axhline(0, color="k", linestyle="--", linewidth=0.8)

        # Timing
        t_sec = data["train_loop_times"][last_ckpt]
        all_times.append(t_sec)
        h, m = int(t_sec // 3600), int((t_sec % 3600) // 60)
        print("Training took %i:%02i h:m" % (h, m))

    avg_h = np.mean(all_times) / 3600.0
    std_h = np.std(all_times) / 3600.0
    print("Average training time: %.3f +/- %.3f h" % (avg_h, std_h))

    legend_str = "Dashed: train loss (sat. risk)\nSolid: test loss (% non-sat. states)"
    props = dict(boxstyle="round", facecolor="whitesmoke")
    fig.text(0.6, 0.855, legend_str, va="center", ha="left", bbox=props, size="small")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=1200)
    print("Saved: %s" % save_path)
