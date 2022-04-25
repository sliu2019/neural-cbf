import torch
import numpy as np
import os, sys
from cvxopt import solvers
solvers.options['show_progress'] = False
import argparse
import pickle

from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
from flying_cbf_controller import CBFController
from flying_plot_utils import load_phi_and_params
from rollout_cbf_classes.deprecated.flying_our_cbf_class import OurCBF
import multiprocessing as mp

from decimal import Decimal
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
	percent_inside = (float(N_samp_found)/(M*i))*100
	return x0s, percent_inside


def extract_statistics(info_dicts, env, param_dict):
	# print("Inside extract_statistics")
	# IPython.embed()

	stat_dict = {}
	x_lim = param_dict["x_lim"]

	# stat_dict["dt"] = env.dt
	# for key, value in info_dicts.items():
	# 	print(key, value.shape)

	# How many rollouts applied safe control before they were terminated?
	apply_u_safe = info_dicts["apply_u_safe"]  # list of len N_rollout; elem is array of size T_max
	N_rollout = len(apply_u_safe)

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

	# total_transitions = on_in_count + on_out_count # TODO
	total_transitions = on_in_count + on_out_count + on_on_count
	print("Total transitions, N rollout: ", total_transitions, N_rollout)
	# Note: If we're counting on-on transitions, a single rollout can have multiple transitions.
	# If not, then should have total_transitions == N_rollout
	# TODO: but then, which transition counts?
	# TOOD: perhaps we should only consider if the rollout ultimately ends up outside or inside...
	# if total_transitions != N_desired_rollout:
	# 	IPython.embed()

	stat_dict["N_transitions"] = total_transitions
	stat_dict["percent_on_in"] = (on_in_count/float(total_transitions))*100
	stat_dict["percent_on_out"] = (on_out_count/float(total_transitions))*100
	stat_dict["percent_on_on"] = (on_on_count/float(total_transitions))*100 # TODO
	# IPython.embed()

	stat_dict["N_on_in"] = on_in_count
	stat_dict["N_on_out"] = on_out_count
	stat_dict["N_on_on"] = on_on_count

	# Debug: how large is the gap? In terms of phi
	# IPython.embed()
	phis = [rl[:, -1] for rl in info_dicts["phi_vals"]]
	gap_phis = [phis[i]*on[i] for i in range(N_rollout)]
	min_phi = np.min([np.min(rl) for rl in gap_phis])
	mean_phi = np.sum([np.sum(rl) for rl in gap_phis])/np.sum([np.sum(rl) for rl in on])
	stat_dict["min_phi"] = min_phi
	stat_dict["mean_phi"] = mean_phi
	# IPython.embed()

	# Debug: why is the gap large? Hypothesis 1: dynamics are extreme
	dist_between_xs = info_dicts["dist_between_xs"]
	dist_between_xs_on_gap = [dist_between_xs[i]*on[i] for i in range(N_rollout)]
	mean_dist = np.sum([np.sum(rl) for rl in dist_between_xs_on_gap])/np.sum([np.sum(rl) for rl in on])
	max_dist = np.max([np.max(rl) for rl in dist_between_xs_on_gap])
	stat_dict["mean_dist"] = mean_dist
	stat_dict["max_dist"] = max_dist

	# Debug: why is the gap large? Hypothesis 2: phi steep near border
	phi_grad_mag = info_dicts["phi_grad_mag"]
	phi_grad_gap = [phi_grad_mag[i]*on[i] for i in range(N_rollout)]
	mean_phi_grad = np.sum([np.sum(rl) for rl in phi_grad_gap])/np.sum([np.sum(rl) for rl in on])
	max_phi_grad = np.max([np.max(rl) for rl in phi_grad_gap])
	stat_dict["mean_phi_grad"] = mean_phi_grad
	stat_dict["max_phi_grad"] = max_phi_grad

	# Debug: what do the large magnitude gradients look like?
	# Show me the state and corresponding gradient
	# TODO
	# phi_grad = info_dicts["phi_grad"]
	# xs = info_dicts["x"]
	# IPython.embed()

	# Find out box exits
	xs = info_dicts["x"]
	outside_box_rl = [np.logical_or(np.any(rl[:, :10] < x_lim[:, 0][None], axis=1), np.any(rl[:, :10] > x_lim[:, 1][None], axis=1)) for rl in xs]
	on_in_outside_box_rl = [outside_box_rl[i][:-2]*on_in_rl[i] for i in range(N_rollout)] # checks if x_i in x_i --> x_f transition is outside. This from [:-2]
	on_out_outside_box_rl = [outside_box_rl[i][:-2]*on_out_rl[i] for i in range(N_rollout)]
	on_on_outside_box_rl = [outside_box_rl[i][:-2]*on_on_rl[i] for i in range(N_rollout)]
	# IPython.embed()

	on_in_outside_box_count = np.sum([np.sum(rl) for rl in on_in_outside_box_rl]) # TODO: is any the right thing here?
	on_out_outside_box_count = np.sum([np.sum(rl) for rl in on_out_outside_box_rl])
	on_on_outside_box_count = np.sum([np.sum(rl) for rl in on_on_outside_box_rl])

	stat_dict["percent_on_in_outside_box"] = (on_in_outside_box_count/float(max(1, on_in_count)))*100
	stat_dict["percent_on_out_outside_box"] = (on_out_outside_box_count/float(max(1, on_out_count)))*100
	stat_dict["percent_on_on_outside_box"] = (on_on_outside_box_count/float(max(1, on_on_count)))*100
	# IPython.embed()

	stat_dict["N_on_in_outside_box"] = on_in_outside_box_count
	stat_dict["N_on_out_outside_box"] = on_out_outside_box_count
	stat_dict["N_on_on_outside_box"] = on_on_outside_box_count

	# Debug box exits
	# Which states are we exiting on?
	outside_box_states = [np.logical_or(rl[:, :10] < x_lim[:, 0][None], rl[:, :10] > x_lim[:, 1][None]) for rl in xs]

	state_ind = [np.argwhere(rl)[:, 1] for rl in outside_box_states]
	state_ind = np.concatenate(state_ind)
	values, counts = np.unique(state_ind, return_counts=True)

	state_index_dict = param_dict['state_index_dict']
	state_index_value_to_key_dict = dict(zip(state_index_dict.values(), state_index_dict.keys()))
	for i in range(len(values)):
		# stat_dict["N_count_exit_on_%i" % values[i]] = counts[i]
		stat_dict["N_count_exit_on_%s" % state_index_value_to_key_dict[values[i]]] = counts[i]

	# which rollouts exited?
	# outside_box_any_rl = [np.sum(rl) for rl in outside_box_rl]
	# which_rl = np.argwhere(outside_box_any_rl).flatten()

	# print("at the end of compute_stats")
	# IPython.embed()
	return stat_dict


def convert_angle_to_negpi_pi_interval(angle):
	new_angle = np.arctan2(np.sin(angle), np.cos(angle))
	return new_angle

def simulate_rollout(env, N_steps_max, cbf_controller, random_seed=None):
	# Random seed is for multiproc
	# print("Inside simulate_rollout")
	# IPython.embed()
	if random_seed:
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)

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

		# Mod on angles
		# TODO: hard-coded
		# IPython.embed()
		ind = [0, 1, 2, 6, 7]
		mod_ang = convert_angle_to_negpi_pi_interval(x[ind])
		x[ind] = mod_ang

		us.append(u)
		xs.append(x)

		if dict is None:
			dict = {key: [value] for (key, value) in debug_dict.items()}
		else:
			for key, value in dict.items():
				value.append(debug_dict[key])

		if debug_dict["apply_u_safe"]:
			u_safe_applied = True
		# if u_safe_applied and (debug_dict["outside_boundary"] or debug_dict["inside_boundary"]): # TODO
		# 	break

		# TODO
		if u_safe_applied:
			t_since += 1
		if t_since > 1:
			break

	# IPython.embed()
	dict = {key: np.array(value) for (key, value) in dict.items()}
	dict["x"] = np.array(xs)
	dict["u"] = np.array(us)

	# print("At the end of a rollout")
	# IPython.embed()
	return dict

def run_rollouts(env, N_desired_rollout, N_steps_max, cbf_controller):
	info_dicts = None
	N_rollout = 0
	# for i in range(N_desired_rollout):
	while N_rollout < N_desired_rollout:
		print("Rollout %i" % N_rollout)
		info_dict = simulate_rollout(env, N_steps_max, cbf_controller)

		# print("inside run_rollouts()")
		# IPython.embed()

		# if not (np.any(info_dict["apply_u_safe"]) and (info_dict["inside_boundary"][-1] or info_dict["outside_boundary"][-1])): # TODO
		# 	continue

		# TODO: if you change anything in this, you have to change it in run_rollouts_multiproc
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
	# save_fpth = os.path.join(log_fldr, "rollouts.pkl")
	# with open(save_fpth, 'wb') as handle:
	# 	pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# print("At the end of all our rollouts")
	# IPython.embed()
	return info_dicts

def run_rollouts_multiproc(env, N_desired_rollout, N_steps_max, cbf_controller):
	info_dicts = None
	N_rollout = 0
	it = 0

	n_cpu = mp.cpu_count()
	random_seeds = np.arange(10*N_desired_rollout)
	np.random.shuffle(random_seeds) # in place
	pool = mp.Pool(n_cpu)

	arg_tup = [env, N_steps_max, cbf_controller]
	duplicated_arg = list([arg_tup] * n_cpu)

	while N_rollout < N_desired_rollout:
		batch_random_seeds = random_seeds[it*n_cpu:(it+1)*n_cpu]
		final_arg = [duplicated_arg[i] + [batch_random_seeds[i]] for i in range(n_cpu)]
		result = pool.starmap(simulate_rollout, final_arg)

		for info_dict in result:
			# if not (np.any(info_dict["apply_u_safe"]) and (np.any(info_dict["inside_boundary"]) or np.any(info_dict["outside_boundary"]))): # TODO
			# if not (np.any(info_dict["apply_u_safe"]) and (info_dict["inside_boundary"][-1] or info_dict["outside_boundary"][-1])):  # TODO
			# 	continue

			# TODO: if you change anything in this, you have to change it in run_rollouts above!
			if not np.any(info_dict["apply_u_safe"]):
				continue
			if N_rollout == N_desired_rollout:
				break

			# Store data
			if info_dicts is None:
				info_dicts = {key: [value] for (key, value) in info_dict.items()}
			else:
				for key, value in info_dicts.items():
					value.append(info_dict[key])

			N_rollout += 1 		# Indicate this rollout has been recorded
		it += 1

	print("\n\n")
	print("Desired rollouts: %i" % N_desired_rollout)
	print("Number of collected rollouts: %i" % N_rollout)

	# Save data
	# save_fpth = os.path.join(log_fldr, "rollouts.pkl")
	# with open(save_fpth, 'wb') as handle:
	# 	pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
		log_fldrpth = os.path.join(log_fldr_base, "exp_%s_ckpt_%i_nrollout_%i_dt_%s" % (exp_name_to_load, checkpoint_number_to_load, args.N_rollout, '%.2E' % Decimal(args.dt)))
	if which_cbf == "low":
		# from main import create_flying_param_dict
		# param_dict = create_flying_param_dict(args)

		from src.argument import create_parser
		parser = create_parser()
		cbf_train_args, _ = parser.parse_known_args() # allows us to ignore unknown args

		from main import create_flying_param_dict
		param_dict = create_flying_param_dict(cbf_train_args) # default

		from rollout_cbf_classes.deprecated.flying_ssa import SSA
		cbf_obj = SSA(param_dict)

		# cbf_obj.c1 = args.low_c1
		cbf_obj.c2 = args.low_c2
		cbf_obj.c3 = args.low_c3

		# log_fldrpth = os.path.join(log_fldr_base, "c1_%.3f_c2_%.3f" % (cbf_obj.c1, cbf_obj.c2))
		log_fldrpth = os.path.join(log_fldr_base, "c2_%.3f_c3_%.3f" % (cbf_obj.c2, cbf_obj.c3))
		# IPython.embed()

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
	if N_desired_rollout < 10:
		info_dicts = run_rollouts(env, N_desired_rollout, N_steps_max, cbf_controller)
	else:
		info_dicts = run_rollouts_multiproc(env, N_desired_rollout, N_steps_max, cbf_controller)

	save_fpth = os.path.join(log_fldrpth, "rollouts.pkl")
	with open(save_fpth, 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#####################################
	# Compute numbers
	#####################################
	stat_dict = extract_statistics(info_dicts, env, param_dict)

	# Finally, approximate volume of invariant set
	_, percent_inside = sample_inside_safe_set(param_dict, cbf_obj, args.N_samp_volume)

	# print(percent_inside)
	stat_dict["vol_approximation"] = percent_inside

	# Save numbers
	with open(os.path.join(log_fldrpth, "stat_dict.pkl"), 'wb') as handle:
		pickle.dump(stat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	for key, value in args.__dict__.items():
		print(key, value)

	print("\n")
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
	parser.add_argument('--which_cbf', type=str, default="ours", choices=["ours", "low"])

	parser.add_argument('--exp_name_to_load', type=str, default="flying_inv_pend_reg_weight_10", help="for our CBF") # flying_inv_pend_first_run
	parser.add_argument('--checkpoint_number_to_load', type=int, default=510, help="for our CBF")

	# parser.add_argument('--low_c1', type=float, default=1.0) # TODO: permanently set to 1.0
	parser.add_argument('--low_c2', type=float, default=1.0)
	parser.add_argument('--low_c3', type=float, default=0.0)

	parser.add_argument('--N_rollout', type=int, default=10)
	parser.add_argument('--dt', type=float, default=1e-4)
	parser.add_argument('--T_max', type=float, default=1e-1)

	parser.add_argument('--N_samp_volume', type=int, default=100000)

	args = parser.parse_known_args()[0]

	# IPython.embed()
	run_rollout_experiment(args)
	""" 
	Example command:
	python flying_rollout_experiment.py --which_cbf ours --exp_name_to_load flying_inv_pend_reg_weight_1 --checkpoint_number_to_load 1380 --N_rollout 2
	
	python flying_rollout_experiment.py --which_cbf low --N_rollout 2

	python flying_rollout_experiment.py --which_cbf low --low_c2 10.0 --N_rollout 50
	python flying_rollout_experiment.py --which_cbf low --low_c2 2.0 --low_c3 1.0 --N_rollout 50
	python flying_rollout_experiment.py --which_cbf low --low_c2 10.0 --N_rollout 50
	
	Testing the volume sampler:
	See if increasing c2, c3 will decrease volume 
	
	Investigating box exits:
	python flying_rollout_experiment.py --which_cbf low --low_c2 0.1 --N_rollout 250
	
	
	python flying_rollout_experiment.py --which_cbf ours --exp_name_to_load flying_inv_pend_phi_format_1_seed_0 --checkpoint_number_to_load 60 --N_rollout 2 --N_samp_volume 10
	python flying_rollout_experiment.py --which_cbf ours --exp_name_to_load flying_inv_pend_phi_format_0_seed_0 --checkpoint_number_to_load 3370 --N_rollout 1000 --N_samp_volume 10
	"""


