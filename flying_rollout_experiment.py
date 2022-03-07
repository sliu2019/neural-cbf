import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys, IPython
from cvxopt import solvers
solvers.options['show_progress'] = False
import argparse
import pickle, math

from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
from flying_cbf_controller import CBFController
from flying_plot_utils import load_phi_and_params, plot_invariant_set_slices
from rollout_cbf_classes.flying_our_cbf_class import OurCBF

# Fixed seed for repeatability
torch.manual_seed(2022)
np.random.seed(2022)

def sample_inside_safe_set(param_dict, cbf_obj, N_samp):
	# TODO; modify this somehow?
	# Or can just increase n_samp for now? Or just find points close to boundary?
	"""
	Uses rejection sampling to sample uniformly in the invariant set

	Note: assumes invariant set is defined as follows:
	x0 in S if max(phi_array(x)) <= 0
	"""
	# print("inside sample_inside_safe_set")
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
		# print(i)
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

	# print("Inside sample x0s")
	# IPython.embed()
	# Could be more than N_samp currently; truncate to exactly N_samp
	x0s = x0s[:N_samp]
	percent_inside = float(N_samp_found)/(M*i)
	return x0s, percent_inside

"""
TODO: 3/3
1. No longer max length: just a generous length that if you hit it without applying safe control, discard the rollouts 
a. Probably, as result, print the number of desired rollouts vs. actual collected rollouts 
2. 
TODO: add to stat dict things like dt, counts (not just percentages), so that you get shown what you want 
"""

def extract_statistics(info_dicts, env, param_dict):
	# TODO: 3/3, for variable length rollouts
	# TODO: proofread this!
	print("Inside extract_statistics")
	IPython.embed()

	stat_dict = {}
	x_lim = param_dict["x_lim"]

	stat_dict["dt"] = env.dt
	# for key, value in info_dicts.items():
	# 	print(key, value.shape)
	"""
	1. Check how many rollouts didn't apply safe control; if too many, then increase T_max
	2. For all rollouts that applied safe control, how many exited?
	a. Count on-on: and how many inside/outside state box?
	b. on-in: inside or outside state box?
	c. on-out: inside or outside state box? 
	3. For each time the boundary was hit, how many times did the safe control prevent exit?
	"""
	# How many rollouts applied safe control before they were terminated?
	apply_u_safe = info_dicts["apply_u_safe"]  # list of len N_rollout; elem is array of size T_max
	N_rollout = len(apply_u_safe)
	stat_dict["N_rollout"] = N_rollout

	# rollouts_any_safe_ctrl = np.any(apply_u_safe, axis=1)
	safe_ctrl_rl = [np.any(rl) for rl in apply_u_safe]
	percent_rollouts_with_safe_ctrl = np.sum(safe_ctrl_rl)/float(N_rollout)
	stat_dict["percent_rollouts_with_safe_ctrl"] = percent_rollouts_with_safe_ctrl

	# Count transitions
	inside = info_dicts["inside_boundary"] # (n_rollouts, t_max)?
	on = info_dicts["on_boundary"]
	outside = info_dicts["outside_boundary"]

	on_in_rl = [on[i][:-1]*inside[i][1:] for i in range(N_rollout)]
	on_out_rl = [on[i][:-1]*outside[i][1:] for i in range(N_rollout)]
	on_on_rl = [on[i][:-1]*on[i][1:] for i in range(N_rollout)]

	on_in_count = np.sum([np.sum(rl) for rl in on_in_rl])
	on_out_count = np.sum([np.sum(rl) for rl in on_out_rl])
	on_on_count = np.sum([np.sum(rl) for rl in on_on_rl])

	total_transitions = on_in_count + on_out_count + on_on_count
	stat_dict["N_transitions"] = total_transitions
	stat_dict["percent_on_in"] = on_in_count/float(total_transitions)
	stat_dict["percent_on_out"] = on_out_count/float(total_transitions)
	stat_dict["percent_on_on"] = on_on_count/float(total_transitions)

	stat_dict["N_on_in"] = on_in_count
	stat_dict["N_on_out"] = on_out_count
	stat_dict["N_on_on"] = on_on_count

	# Find out box exits
	xs = info_dicts["x"]
	outside_box_rl = [np.logical_or(np.any(rl < x_lim[:, 0], axis=1), np.any(rl > x_lim[:, 1], axis=1)) for rl in xs]
	on_in_outside_box_rl = [np.logical_or(outside_box_rl[i], on_in_rl[i]) for i in range(N_rollout)]
	on_out_outside_box_rl = [np.logical_or(outside_box_rl[i], on_out_rl[i]) for i in range(N_rollout)]
	on_on_outside_box_rl = [np.logical_or(outside_box_rl[i], on_on_rl[i]) for i in range(N_rollout)]

	on_in_outside_box_count = np.sum([np.sum(rl) for rl in on_in_outside_box_rl])
	on_out_outside_box_count = np.sum([np.sum(rl) for rl in on_out_outside_box_rl])
	on_on_outside_box_count = np.sum([np.sum(rl) for rl in on_on_outside_box_rl])

	stat_dict["percent_on_in_outside_box"] = on_in_outside_box_count/float(on_in_count)
	stat_dict["percent_on_out_outside_box"] = on_out_outside_box_count/float(on_out_count)
	stat_dict["percent_on_on_outside_box"] = on_on_outside_box_count/float(on_on_count)

	return stat_dict

def simulate_rollout(env, x0, N_dt, cbf_controller):
		# print("Inside simulate_rollout")
	# IPython.embed()

	x = x0.copy()
	xs = [x]
	us = []
	dict = None

	# IPython.embed()
	u_safe_applied = False
	t_since = 0
	for t in range(N_dt): # TODO: end criteria should be different
		# print("rollout, step %i" % t)
		# print(x)
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

		# TODO: reread: should these be elif or if?
		if debug_dict["apply_u_safe"]:
			u_safe_applied = True
		if u_safe_applied:
			t_since += 1

		if t_since > 2:
			break

	dict = {key: np.array(value) for (key, value) in dict.items()}
	dict["x"] = np.array(xs)
	dict["u"] = np.array(us)

	print("At the end of a rollout")
	IPython.embed()
	return dict

def run_rollouts(env, N_rollout, x0s, N_dt, cbf_controller, log_fldr):
	info_dicts = None
	for i in range(N_rollout):
		print("Rollout %i" % i)
		info_dict = simulate_rollout(env, x0s[i], N_dt, cbf_controller)

		if info_dicts is None:
			info_dicts = info_dict
			# Dict comprehension is: dict_variable = {key: value for (key, value) in dictonary.items()}
			# info_dicts = {key: value[None] for (key, value) in info_dicts.items()}
			info_dicts = {key: [value] for (key, value) in info_dicts.items()} # TODO: 3/3
		else:
			# info_dicts = {key: np.concatenate((value, info_dict[key][None]), axis=0) for (key, value) in
			# 			  info_dicts.items()}
			# TODO: 3/3
			for key, value in info_dicts.items():
				value.append(info_dict[key])
	# Save data
	save_fpth = os.path.join(log_fldr, "data.pkl")
	with open(save_fpth, 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("At the end of all our rollouts")
	IPython.embed()
	return info_dicts

def run_rollout_experiment(args):
	which_cbf = args.which_cbf
	exp_name = args.exp_name
	checkpoint_number = args.checkpoint_number


	if which_cbf == "ours":
		phi_fn, param_dict = load_phi_and_params(exp_name, checkpoint_number)
		cbf_obj = OurCBF(phi_fn, param_dict) # numpy wrapper
		log_fldrpth = "./log/%s" % exp_name
	else:
		# TODO: Tianhao, Weiye, add clauses here
		raise NotImplementedError
		# Create your own cbf_obj
		# Use the following to create param_dict
		# from main import create_flying_param_dict
		# param_dict = create_flying_param_dict()

	env = FlyingInvertedPendulumEnv(param_dict)

	# TODO: fill out run arguments
	N_rollout = 10
	T_max = 0.075 #1.5  # in seconds
	N_dt = int(T_max / env.dt)
	print("Number of timesteps: %f" % N_dt)

	cbf_controller = CBFController(env, cbf_obj, param_dict)

	# Get x0's
	x0s, _ = sample_inside_safe_set(param_dict, cbf_obj, N_rollout)

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

	# IPython.embed()
	#####################################
	# Sanity checks
	#####################################
	stat_dict = extract_statistics(info_dicts, env, param_dict)

	# Finally, approximate volume of invariant set
	N_samp = 1000
	_, percent_inside = sample_inside_safe_set(param_dict, cbf_obj, N_samp)
	stat_dict["percent_inside"] = percent_inside
	stat_dict["N_samp"] = N_samp

	with open(os.path.join(log_fldrpth, "stat_dict.pkl"), 'wb') as handle:
		pickle.dump(stat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	for key, value in stat_dict.items():
		print("%s: %.3f" % (key, value))

	sys.exit(0)
	#####################################
	# Plot trajectories
	#####################################
	# if which_cbf == "ours":
	# 	rollouts = info_dicts["x"]
	# 	plot_invariant_set_slices(phi_fn, param_dict, rollouts=rollouts, fnm="traj", fldr_path=log_fldrpth, which_params=[["phi", "theta"]])
	# else:
	# 	raise NotImplementedError
		# TODO: Suggest slightly modify above function to be used with cbf_obj (numpy wrapper of phi_fn), instead of phi_fn

	#####################################
	# Plot EXITED trajectories ONLY (we choose the 5 with the largest violation)
	#####################################
	# TODO: should we be maximizing h instead of phi?
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
	# TODO: something wrong with parser, prevents us from passing arguments in; it is clashing with another instance of ArgumentParser which is being imported
	parser = argparse.ArgumentParser(description='Rollout experiment for flying')
	parser.add_argument('--which_cbf', type=str, default="ours")

	parser.add_argument('--exp_name', type=str, default="flying_inv_pend_reg_weight_10", help="for our CBF") # flying_inv_pend_first_run
	parser.add_argument('--checkpoint_number', type=int, default=510, help="for our CBF")

	args = parser.parse_args()
	run_rollout_experiment(args)

	# python flying_rollout_experiment.py --which_cbf ours --exp_name first_run --checkpoint_number 3080
