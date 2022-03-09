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
2. TODO: make print out informative, so when Weiye/Tianhao runs, you can see all relevant info at a glance 
TODO: also make sure that saved info is informative
TODO: add to stat dict things like dt, counts (not just percentages), so that you get shown what you want 
"""

def extract_statistics(info_dicts, env, param_dict):
	# TODO: 3/3, for variable length rollouts
	# TODO: proofread this!
	print("Inside extract_statistics")
	# IPython.embed()

	stat_dict = {}
	x_lim = param_dict["x_lim"]

	# stat_dict["dt"] = env.dt
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
	apply_u_safe = info_dicts["apply_u_safe"]  # list of len N_desired_rollout; elem is array of size T_max
	N_desired_rollout = len(apply_u_safe)
	stat_dict["N_desired_rollout"] = N_desired_rollout

	"""safe_ctrl_rl = [np.any(rl) for rl in apply_u_safe]
	percent_rollouts_with_safe_ctrl = np.sum(safe_ctrl_rl)/float(N_desired_rollout)
	stat_dict["percent_rollouts_with_safe_ctrl"] = percent_rollouts_with_safe_ctrl"""


	# Count transitions
	inside = info_dicts["inside_boundary"] # (n_rollouts, t_max)?
	on = info_dicts["on_boundary"]
	outside = info_dicts["outside_boundary"]
	# IPython.embed()

	on_in_rl = [on[i][:-1]*inside[i][1:] for i in range(N_desired_rollout)]
	on_out_rl = [on[i][:-1]*outside[i][1:] for i in range(N_desired_rollout)]
	on_on_rl = [on[i][:-1]*on[i][1:] for i in range(N_desired_rollout)]

	on_in_count = np.sum([np.sum(rl) for rl in on_in_rl])
	on_out_count = np.sum([np.sum(rl) for rl in on_out_rl])
	on_on_count = np.sum([np.sum(rl) for rl in on_on_rl])
	# IPython.embed()

	total_transitions = on_in_count + on_out_count + on_on_count
	print("Total transitions, N desired rollout: ", total_transitions, N_desired_rollout)
	assert total_transitions == N_desired_rollout # TODO

	stat_dict["N_transitions"] = total_transitions
	stat_dict["percent_on_in"] = (on_in_count/float(total_transitions))*100
	stat_dict["percent_on_out"] = (on_out_count/float(total_transitions))*100
	stat_dict["percent_on_on"] = (on_on_count/float(total_transitions))*100
	# IPython.embed()

	stat_dict["N_on_in"] = on_in_count
	stat_dict["N_on_out"] = on_out_count
	stat_dict["N_on_on"] = on_on_count

	# Find out box exits
	xs = info_dicts["x"]
	outside_box_rl = [np.logical_or(np.any(rl[:, :10] < x_lim[:, 0][None], axis=1), np.any(rl[:, :10] > x_lim[:, 1][None], axis=1)) for rl in xs]
	on_in_outside_box_rl = [outside_box_rl[i][:-2]*on_in_rl[i] for i in range(N_desired_rollout)] # checks if x_i in x_i --> x_f transition is outside. This from [:-2]
	on_out_outside_box_rl = [outside_box_rl[i][:-2]*on_out_rl[i] for i in range(N_desired_rollout)]
	on_on_outside_box_rl = [outside_box_rl[i][:-2]*on_on_rl[i] for i in range(N_desired_rollout)]
	# IPython.embed()

	on_in_outside_box_count = np.sum([np.any(rl) for rl in on_in_outside_box_rl]) # TODO: is any the right thing here?
	on_out_outside_box_count = np.sum([np.any(rl) for rl in on_out_outside_box_rl])
	on_on_outside_box_count = np.sum([np.any(rl) for rl in on_on_outside_box_rl])

	stat_dict["percent_on_in_outside_box"] = (on_in_outside_box_count/float(max(1, on_in_count)))*100
	stat_dict["percent_on_out_outside_box"] = (on_out_outside_box_count/float(max(1, on_out_count)))*100
	stat_dict["percent_on_on_outside_box"] = (on_on_outside_box_count/float(max(1, on_on_count)))*100
	# IPython.embed()

	stat_dict["N_on_in_outside_box"] = on_in_outside_box_count
	stat_dict["N_on_out_outside_box"] = on_out_outside_box_count
	stat_dict["N_on_on_outside_box"] = on_on_outside_box_count

	return stat_dict

def simulate_rollout(env, N_steps_max, cbf_controller):
	# print("Inside simulate_rollout")
	# IPython.embed()

	# Compute x0
	x0, _ = sample_inside_safe_set(cbf_controller.param_dict, cbf_controller.cbf_obj, 1)

	# Initialize data structures
	x = x0.flatten()
	xs = [x]
	us = []
	dict = None

	# IPython.embed()
	u_safe_applied = False
	t_since = 0
	for t in range(N_steps_max): # TODO: end criteria should be different
		print("rollout, step %i" % t)
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
		if t_since > 2: # 3 steps after first applying safe control
			break

	dict = {key: np.array(value) for (key, value) in dict.items()}
	dict["x"] = np.array(xs)
	dict["u"] = np.array(us)

	# print("At the end of a rollout")
	# IPython.embed()
	return dict

def run_rollouts(env, N_desired_rollout, N_steps_max, cbf_controller, log_fldr):
	info_dicts = None
	N_rollout = 0
	# for i in range(N_desired_rollout):
	while N_rollout < N_desired_rollout:
		print("Rollout %i" % N_rollout)
		info_dict = simulate_rollout(env, N_steps_max, cbf_controller)

		# print("inside run_rollouts()")
		# IPython.embed()

		if not np.any(info_dict["apply_u_safe"]):
			continue

		# Store data
		if info_dicts is None:
			info_dicts = {key: [value] for (key, value) in info_dict.items()}
		else:
			for key, value in info_dicts.items():
				value.append(info_dict[key])

		# Indicate this rollout has been recorded
		N_rollout += 1

	# Save data
	save_fpth = os.path.join(log_fldr, "rollouts.pkl")
	with open(save_fpth, 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# print("At the end of all our rollouts")
	# IPython.embed()
	return info_dicts

def run_rollout_experiment(args):

	which_cbf = args.which_cbf
	exp_name_to_load = args.exp_name_to_load
	checkpoint_number_to_load = args.checkpoint_number_to_load

	log_fldr_base = "./rollout_results/flying/%s" % which_cbf

	if which_cbf == "ours":
		phi_fn, param_dict = load_phi_and_params(exp_name_to_load, checkpoint_number_to_load)
		cbf_obj = OurCBF(phi_fn, param_dict) # numpy wrapper
		log_fldrpth = os.path.join(log_fldr_base, "exp_%s_ckpt_%i" % (exp_name_to_load, checkpoint_number_to_load))
	if which_cbf == "low":
		from main import create_flying_param_dict
		param_dict = create_flying_param_dict(args)
		from rollout_cbf_classes.flying_ssa import SSA
		cbf_obj = SSA(param_dict)
		log_fldrpth = os.path.join(log_fldr_base, "c1_%.2f_c2_%.2f" % (cbf_obj.c1, cbf_obj.c2))

	# Making log folder
	if not os.path.exists(log_fldrpth):
		os.makedirs(log_fldrpth)
	print("Saving data at: %s" % log_fldrpth)

	# Save args
	with open(os.path.join(log_fldrpth, "args.pkl"), 'wb') as handle:
		pickle.dump(args.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Experimental settings
	N_desired_rollout = args.N_rollout
	T_max = args.T_max
	# T_max = 0.075
	N_steps_max = int(T_max / args.dt)
	# print("Number of timesteps: %f" % N_steps_max)

	# Create core classes: environment, controller
	env = FlyingInvertedPendulumEnv(param_dict)
	env.dt = args.dt
	cbf_controller = CBFController(env, cbf_obj, param_dict)

	# print("before running rollouts")
	# IPython.embed()
	#####################################
	# Run multiple rollout_results
	#####################################
	info_dicts = run_rollouts(env, N_desired_rollout, N_steps_max, cbf_controller, log_fldrpth)

	#####################################
	# Compute numbers
	#####################################
	stat_dict = extract_statistics(info_dicts, env, param_dict)

	# Finally, approximate volume of invariant set
	_, percent_inside = sample_inside_safe_set(param_dict, cbf_obj, args.N_samp_volume)
	stat_dict["vol_approximation"] = percent_inside

	# Save numbers
	with open(os.path.join(log_fldrpth, "stat_dict.pkl"), 'wb') as handle:
		pickle.dump(stat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	for key, value in args.__dict__.items():
		print(key, value)

	# IPython.embed()
	for key, value in stat_dict.items():
		print("%s: %.3f" % (key, value))

	sys.exit(0)
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
	# if which_cbf == "ours":
	# 	rollouts = info_dicts["x"]
	# 	phi_vals = info_dicts["phi_vals"]  # (N_desired_rollout, T_max, r+1)
	# 	phi_max = np.max(phi_vals, axis=2)
	# 	rollouts_any_exits = np.any(phi_max > 0, axis=1)
	# 	any_exits = np.any(rollouts_any_exits)
	#
	# 	if any_exits:
	# 		exit_rollout_inds = np.argsort(np.max(phi_max, axis=1)).flatten()[::-1]
	# 		exit_rollout_inds = exit_rollout_inds[:min(5, np.sum(rollouts_any_exits))]
	#
	# 		exiting_rollouts = [rollouts for i in exit_rollout_inds]
	# 		plot_invariant_set_slices(phi_fn, param_dict, rollouts=exiting_rollouts, fnm="exiting_traj", fldr_path=log_fldrpth, which_params=[["phi", "theta"]])
	# else:
	# 	raise NotImplementedError
	# 	# TODO: Suggest slightly modify above function to be used with cbf_obj (numpy wrapper of phi_fn), instead of phi_fn

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Rollout experiment for flying')
	parser.add_argument('--which_cbf', type=str, default="ours")
	parser.add_argument('--exp_name_to_load', type=str, default="flying_inv_pend_reg_weight_10", help="for our CBF") # flying_inv_pend_first_run
	parser.add_argument('--checkpoint_number_to_load', type=int, default=510, help="for our CBF")

	parser.add_argument('--N_rollout', type=int, default=10)
	parser.add_argument('--dt', type=float, default=1e-4)
	parser.add_argument('--T_max', type=float, default=1e-1)

	parser.add_argument('--N_samp_volume', type=int, default=1000)

	args = parser.parse_args()

	# IPython.embed()
	run_rollout_experiment(args)
	"""
	Example command:
	python flying_rollout_experiment.py --which_cbf ours --exp_name_to_load flying_inv_pend_reg_weight_1 --checkpoint_number_to_load 1380 --N_rollout 2
	"""


