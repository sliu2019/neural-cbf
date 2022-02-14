import torch
from plot_utils import plot_trajectories, plot_samples_invariant_set, \
	plot_exited_trajectories
from src.utils import *
from cvxopt import solvers

solvers.options['show_progress'] = False

import pickle

from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
from flying_cbf_controller import FlyingCBFController
from flying_plot_utils import *
from rollout_cbf_classes.flying_our_cbf_class import OurCBF

import argparse

# Fixed seed for repeatability
torch.manual_seed(2022)
np.random.seed(2022)


def sample_invariant_set(x_lim, cbf_obj, N_samp):
	"""
	Note: assumes invariant set is defined as follows:
	x0 in S if max(phi_array(x)) <= 0
	"""
	# IPython.embed()
	# Discretizes state space, then returns the subset of states in invariant set

	# delta = 0.01
	# x = np.arange(x_lim[0, 0], x_lim[0, 1], delta)
	# y = np.arange(x_lim[1, 0], x_lim[1, 1], delta)[::-1]  # need to reverse it
	# X, Y = np.meshgrid(x, y)
	#
	# ##### Plotting ######
	# sze = X.size
	# input = np.concatenate((np.zeros((sze, 1)), X.flatten()[:, None], np.zeros((sze, 1)), Y.flatten()[:, None]), axis=1)
	# phi_vals_on_grid = cbf_obj.phi_fn(input)  # N_samp x r+1
	#
	# max_phi_vals_on_grid = phi_vals_on_grid.max(axis=1)  # Assuming S = all phi_i <= 0
	# max_phi_vals_on_grid = np.reshape(max_phi_vals_on_grid, X.shape)
	# where_invariant = np.argwhere(max_phi_vals_on_grid <= 0)
	#
	# sample_ind = np.random.choice(np.arange(where_invariant.shape[0]), size=N_samp, replace=False)
	# global_ind = where_invariant[sample_ind]
	# sample_X = X[global_ind[:, 0], global_ind[:, 1]]
	# sample_Y = Y[global_ind[:, 0], global_ind[:, 1]]
	#
	# x0s = np.zeros((N_samp, 4))
	# x0s[:, 1] = sample_X
	# x0s[:, 3] = sample_Y
	#
	# return x0s, phi_vals_on_grid, X, Y
	raise NotImplementedError

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


def run_rollouts(env, N_rollout, x0s, N_dt, cbf_controller, save_prefix):
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
	with open(save_prefix + ".pkl", 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return info_dicts

def run_rollout_experiment(args):
	log_folder = args.log_folder
	which_cbf = args.which_cbf
	exp_name = args.exp_name
	checkpoint_number = args.checkpoint_number

	log_fldrpth = os.path.join("rollout_results", log_folder)
	if not os.path.exists(log_fldrpth):
		makedirs(log_fldrpth)

	env = FlyingInvertedPendulumEnv()

	# TODO: fill out run arguments
	N_rollout = 100
	T_max = 1.5  # in seconds
	N_dt = int(T_max / env.dt)

	# TODO: Tianhao, Weiye, add clauses here
	if which_cbf == "ours":
		phi_fn, param_dict = load_phi_and_params(exp_name, checkpoint_number)
		cbf_obj = OurCBF(phi_fn, param_dict) # numpy wrapper
	else:
		# create default param_dict
		from main import create_flying_param_dict
		param_dict = create_flying_param_dict()

	cbf_controller = FlyingCBFController(env, cbf_obj, param_dict)
	x_lim = cbf_controller.env.x_lim

	# Get x0's
	# x0s, phi_vals_on_grid, X, Y = sample_invariant_set(x_lim, cbf_obj, N_rollout) # TODO!!! How can we get uniform samples within the invariant set?
	# save_prefix = "./rollout_results/%s/%s_" % (log_folder, which_cbf)
	#
	# #####################################
	# # Plot x0 samples and invariant set
	# #####################################
	# phi_signs = plot_samples_invariant_set(x_lim, x0s, phi_vals_on_grid, X, save_prefix) # TODO: reuse plot_util
	#
	# # IPython.embed()
	# # sys.exit(0)
	#
	# #####################################
	# # Run multiple rollout_results
	# #####################################
	# info_dicts = run_rollouts(env, N_rollout, x0s, N_dt, cbf_controller, save_prefix) # TODO: check that this works
	#
	# #####################################
	# # Sanity checks
	# #####################################
	# sanity_check(info_dicts)
	#
	# #####################################
	# # Plot trajectories
	# #####################################
	# plot_trajectories(x_lim, N_rollout, x0s, phi_vals_on_grid, X, Y, phi_signs, info_dicts, save_prefix) # TODO: reuse plot_util
	#
	# #####################################
	# # Plot EXITED trajectories ONLY (we choose the 5 with the largest violation)
	# #####################################
	# plot_exited_trajectories(x_lim, x0s, phi_vals_on_grid, X, Y, phi_signs, info_dicts, save_prefix) # TODO: reuse plot_util

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Rollout experiment')
	parser.add_argument('--log_folder', type=str, default="debug")
	parser.add_argument('--which_cbf', type=str, choices=["ours", "hand_derived", "cma_es"])

	parser.add_argument('--exp_name', type=str, help="for our CBF")
	parser.add_argument('--checkpoint_number', type=int, help="for our CBF")

	# parser.add_argument('--reg_weight', type=float, default=1.0, help="only relevant for cma-es")
	args = parser.parse_args()
	run_rollout_experiment(args)
