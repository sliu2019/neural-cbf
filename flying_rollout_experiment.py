import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys, IPython
from cvxopt import solvers
solvers.options['show_progress'] = False
import argparse
import pickle

from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
from flying_cbf_controller import CBFController
from flying_plot_utils import load_phi_and_params, plot_invariant_set_slices
from rollout_cbf_classes.flying_our_cbf_class import OurCBF


# Fixed seed for repeatability
torch.manual_seed(2022)
np.random.seed(2022)

def sample_x0s(param_dict, cbf_obj, N_samp):
	"""
	Uses rejection sampling to sample uniformly in the invariant set

	Note: assumes invariant set is defined as follows:
	x0 in S if max(phi_array(x)) <= 0
	"""
	# print("inside sample_x0s")
	# IPython.embed()
	# Define some variables
	x_dim = param_dict["x_dim"]
	x_lim = param_dict["x_lim"]
	box_side_lengths = x_lim[:, 1] - x_lim[:, 0]
	x0s = np.empty((0, 16))

	M = 50
	N_samp_found = 0
	i = 0
	while N_samp_found < N_samp:
		print(i)
		# Sample in box
		samples = np.random.rand(M, x_dim)
		samples = samples*box_side_lengths + x_lim[:, 0]
		samples = np.concatenate((samples, np.zeros((M, 6))), axis=1) # Add translational states as zeros

		# Check if samples in invariant set
		phi_vals = cbf_obj.phi_fn(samples)
		max_phi_vals = phi_vals.max(axis=1)

		# Save good samples
		ind = np.argwhere(max_phi_vals <= 0).flatten()
		samples_inside = samples[ind]
		x0s = np.concatenate((x0s, samples_inside), axis=0)
		N_samp_found += len(ind)
		i += 1

	# Could be more than N_samp currently; truncate to exactly N_samp
	x0s = x0s[:N_samp]
	return x0s

def compute_exits(phi_vals):
	phi_max = np.max(phi_vals, axis=2)
	rollouts_any_exits = np.any(phi_max > 0, axis=1)
	any_exits = np.any(rollouts_any_exits)
	print("Any exits?", any_exits)
	if any_exits:
		print("Percent exits: ", np.mean(rollouts_any_exits))
		print("Which rollout_results have exits:", rollouts_any_exits)
		print("How many exits per rollout: ", np.sum(phi_max > 0, axis=1))

		# Compute magnitude of exits
		pos_phi_inds = np.argwhere(phi_max > 0)
		pos_phi_max = phi_max[pos_phi_inds[:, 0], pos_phi_inds[:, 1]]
		print("Phi max mean and std: %f +/- %f" % (np.mean(pos_phi_max), np.std(pos_phi_max)))
		print("Phi max maximum: %f" % (np.max(pos_phi_max)))


def sanity_check(info_dicts):
	# print("before sanity checks")
	# for key, value in info_dicts.items():
	# 	print(key, value.shape)
	# IPython.embed()
	# 1. Check that all rollout_results touched the invariant set boundary. If not, increase T_max
	# 2. Compute the number of exits for each rollout
	# 3. Compute the number of rollout_results without any exits
	# debug_dict = {"x": x, "u": u, "apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
	print("*********************************************************\n")
	apply_u_safe = info_dicts["apply_u_safe"]  # (N_rollout, T_max)
	rollouts_any_safe_ctrl = np.any(apply_u_safe, axis=1)
	print("Did we apply safe control?", np.all(rollouts_any_safe_ctrl))
	if np.all(rollouts_any_safe_ctrl) == False:
		print("Which rollout_results did we apply safe control?", rollouts_any_safe_ctrl)
		false_ind = np.argwhere(np.logical_not(rollouts_any_safe_ctrl))
		x = info_dicts["x"]
		x_for_false = x[false_ind.flatten()]
		# theta_for_false = x_for_false[:, :, 1]
		# thetadot_for_false = x_for_false[:, :, 3]

	phi_vals = info_dicts["phi_vals"]  # (N_rollout, T_max, r+1)
	compute_exits(phi_vals)

	phi_star = phi_vals[:, :, -1]
	rollouts_any_phistar_pos = np.any(phi_star > 0, axis=1)
	any_phistar_pos = np.any(rollouts_any_phistar_pos)

	print("Any phi_star positive?", any_phistar_pos)
	if any_phistar_pos:
		print("Which rollout_results had phi_star positive:", rollouts_any_phistar_pos)

def simulate_rollout(env, x0, N_dt, cbf_controller):
	# print("Inside simulate_rollout")
	# IPython.embed()

	x = x0.copy()
	# 	debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
	xs = [x]
	us = []
	dict = None

	# IPython.embed()
	for t in range(N_dt):
		print("rollout, step %i" % t)
		# IPython.embed()
		# if t == 10:
		# 	IPython.embed()

		u, debug_dict = cbf_controller.compute_control(t, x)  # Define this
		x_dot = env.x_dot_open_loop(x, u)
		x = x + env.dt * x_dot

		us.append(u)
		xs.append(x)

		if dict is None:
			dict = {key: [value] for (key, value) in debug_dict.items()}
		else:
			for key, value in dict.items():
				value.append(debug_dict[key])

	dict = {key: np.array(value) for (key, value) in dict.items()}
	dict["x"] = np.array(xs)
	dict["u"] = np.array(us)

	# print("At the end of a rollout")
	# IPython.embed()
	return dict

def run_rollouts(env, N_rollout, x0s, N_dt, cbf_controller, log_fldr):
	info_dicts = None
	for i in range(N_rollout):
		info_dict = simulate_rollout(env, x0s[i], N_dt, cbf_controller)

		if info_dicts is None:
			info_dicts = info_dict
			# Dict comprehension is: dict_variable = {key: value for (key, value) in dictonary.items()}
			info_dicts = {key: value[None] for (key, value) in info_dicts.items()}
		else:
			info_dicts = {key: np.concatenate((value, info_dict[key][None]), axis=0) for (key, value) in
			              info_dicts.items()}

	# Save data
	save_fpth = os.path.join(log_fldr, "data.pkl")
	with open(save_fpth, 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return info_dicts

def run_rollout_experiment(args):
	# IPython.embed()
	log_folder = args.log_folder
	which_cbf = args.which_cbf
	exp_name = args.exp_name
	checkpoint_number = args.checkpoint_number

	log_fldrpth = os.path.join("./rollout_results/flying", log_folder)
	if not os.path.exists(log_fldrpth):
		os.makedirs(log_fldrpth)
	# save_prefix = "%s/%s_" % (log_fldrpth, which_cbf)

	# TODO: Tianhao, Weiye, add clauses here
	if which_cbf == "ours":
		phi_fn, param_dict = load_phi_and_params(exp_name, checkpoint_number)
		cbf_obj = OurCBF(phi_fn, param_dict) # numpy wrapper
	else:
		raise NotImplementedError
		# Create your own cbf_obj
		# Use the following to create param_dict
		# from main import create_flying_param_dict
		# param_dict = create_flying_param_dict()

	env = FlyingInvertedPendulumEnv(param_dict)

	# TODO: fill out run arguments
	N_rollout = 1
	T_max = 1.5  # in seconds
	N_dt = int(T_max / env.dt)

	cbf_controller = CBFController(env, cbf_obj, param_dict)
	# x_lim = cbf_controller.env.x_lim

	# Get x0's
	x0s = sample_x0s(param_dict, cbf_obj, N_rollout)
	# print("main file, ln 199")
	# IPython.embed()

	#####################################
	# Plot x0 samples and invariant set
	#####################################
	# if which_cbf == "ours":
	# 	x0s_no_translation = x0s[:, :10]
	# 	plot_invariant_set_slices(phi_fn, param_dict, samples=x0s_no_translation, fnm="x0s", fldr_path=log_fldrpth) # which_params=None
	# else:
	# 	raise NotImplementedError
	# 	# TODO: Suggest slightly modify above function to be used with cbf_obj (numpy wrapper of phi_fn), instead of phi_fn

	#####################################
	# Run multiple rollout_results
	#####################################
	info_dicts = run_rollouts(env, N_rollout, x0s, N_dt, cbf_controller, log_fldrpth)

	#####################################
	# Sanity checks
	#####################################
	# IPython.embed()
	sanity_check(info_dicts)

	#####################################
	# Plot trajectories
	#####################################
	if which_cbf == "ours":
		rollouts = info_dicts["x"]
		plot_invariant_set_slices(phi_fn, param_dict, rollouts=rollouts, fnm="traj", fldr_path=log_fldrpth, which_params=[["phi", "theta"]])
	else:
		raise NotImplementedError
		# TODO: Suggest slightly modify above function to be used with cbf_obj (numpy wrapper of phi_fn), instead of phi_fn

	#####################################
	# Plot EXITED trajectories ONLY (we choose the 5 with the largest violation)
	#####################################
	if which_cbf == "ours":
		rollouts = info_dicts["x"]
		phi_vals = info_dicts["phi_vals"]  # (N_rollout, T_max, r+1)
		phi_max = np.max(phi_vals, axis=2)
		rollouts_any_exits = np.any(phi_max > 0, axis=1)
		any_exits = np.any(rollouts_any_exits)

		if any_exits:
			exit_rollout_inds = np.argsort(np.max(phi_max, axis=1)).flatten()[::-1]
			exit_rollout_inds = exit_rollout_inds[:min(5, np.sum(rollouts_any_exits))]

			exiting_rollouts = [rollouts for i in exit_rollout_inds]
			plot_invariant_set_slices(phi_fn, param_dict, rollouts=exiting_rollouts, fnm="exiting_traj", fldr_path=log_fldrpth, which_params=[["phi", "theta"]])
	else:
		raise NotImplementedError
		# TODO: Suggest slightly modify above function to be used with cbf_obj (numpy wrapper of phi_fn), instead of phi_fn

if __name__ == "__main__":
	# TODO: something wrong with parser, prevents us from passing arguments in
	parser = argparse.ArgumentParser(description='Rollout experiment for flying')
	parser.add_argument('--log_folder', type=str, default="debug")
	parser.add_argument('--which_cbf', type=str, default="ours")

	parser.add_argument('--exp_name', type=str, default="flying_inv_pend_first_run", help="for our CBF")
	parser.add_argument('--checkpoint_number', type=int, default=3080, help="for our CBF")

	args = parser.parse_args()
	# IPython.embed()
	run_rollout_experiment(args)

	# python flying_rollout_experiment.py --which_cbf ours --exp_name first_run --checkpoint_number 3080
