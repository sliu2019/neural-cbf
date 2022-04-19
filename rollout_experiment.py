import torch
from plot_utils import plot_trajectories, plot_samples_invariant_set, plot_exited_trajectories
from src.utils import *
from cvxopt import solvers

solvers.options['show_progress'] = False

import pickle

from rollout_cbf_classes.deprecated.cma_es_evaluator import CartPoleEvaluator
from rollout_cbf_classes.deprecated.our_cbf_class import OurCBF
from rollout_cbf_classes.deprecated.normal_ssa_newsi import SSA
from rollout_envs.cart_pole_env import CartPoleEnv
from cbf_controller import CBFController

import sys, argparse

# Fixed seed for repeatability
torch.manual_seed(2022)
np.random.seed(2022)

# theta_safety_lim = math.pi/4.0 

def sample_invariant_set(x_lim, cbf_obj, N_samp):
	"""
	Note: assumes invariant set is defined as follows:
	x0 in S if max(phi_array(x)) <= 0
	"""
	# IPython.embed()
	# Discretizes state space, then returns the subset of states in invariant set
	delta = 0.01
	x = np.arange(x_lim[0, 0], x_lim[0, 1], delta)
	y = np.arange(x_lim[1, 0], x_lim[1, 1], delta)[::-1] # need to reverse it 
	X, Y = np.meshgrid(x, y)

	##### Plotting ######
	sze = X.size
	input = np.concatenate((np.zeros((sze, 1)), X.flatten()[:, None], np.zeros((sze, 1)), Y.flatten()[:, None]), axis=1)
	phi_vals_on_grid = cbf_obj.phi_fn(input) # N_samp x r+1

	max_phi_vals_on_grid = phi_vals_on_grid.max(axis=1) # Assuming S = all phi_i <= 0
	max_phi_vals_on_grid = np.reshape(max_phi_vals_on_grid, X.shape)
	where_invariant = np.argwhere(max_phi_vals_on_grid <= 0)

	sample_ind = np.random.choice(np.arange(where_invariant.shape[0]), size=N_samp, replace=False)
	global_ind = where_invariant[sample_ind]
	sample_X = X[global_ind[:, 0], global_ind[:, 1]]
	sample_Y = Y[global_ind[:, 0], global_ind[:, 1]]

	x0s = np.zeros((N_samp, 4))
	x0s[:, 1] = sample_X
	x0s[:, 3] = sample_Y

	return x0s, phi_vals_on_grid, X, Y

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
		u, debug_dict = cbf_controller.compute_control(t, x) # Define this
		x_dot = env.x_dot_open_loop(x, u)
		x = x + env.dt*x_dot

		us.append(u)
		xs.append(x)

		if dict is None:
			dict = {key:[value] for (key, value) in debug_dict.items()}
		else:
			for key, value in dict.items():
				value.append(debug_dict[key])

	dict = {key:np.array(value) for (key, value) in dict.items()}
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
	print("Did we apply safe control?" , np.all(rollouts_any_safe_ctrl))
	if np.all(rollouts_any_safe_ctrl) == False:
		print("Which rollout_results did we apply safe control?", rollouts_any_safe_ctrl)
		false_ind = np.argwhere(np.logical_not(rollouts_any_safe_ctrl))
		x = info_dicts["x"]
		x_for_false = x[false_ind.flatten()]
		theta_for_false = x_for_false[:, :, 1]
		thetadot_for_false = x_for_false[:, :, 3]
	
	phi_vals = info_dicts["phi_vals"] # (N_rollout, T_max, r+1)
	compute_exits(phi_vals)

	phi_star = phi_vals[:, :, -1]
	rollouts_any_phistar_pos = np.any(phi_star>0, axis=1)
	any_phistar_pos = np.any(rollouts_any_phistar_pos)

	print("Any phi_star positive?", any_phistar_pos)
	if any_phistar_pos:
		print("Which rollout_results had phi_star positive:", rollouts_any_phistar_pos)


def run_rollout(env, N_rollout, x0s, N_dt, cbf_controller, save_prefix):
	info_dicts = None
	for i in range(N_rollout):
		info_dict = simulate_rollout(env, x0s[i], N_dt, cbf_controller)

		if info_dicts is None:
			info_dicts = info_dict
			# Dict comprehension is: dict_variable = {key: value for (key, value) in dictonary.items()}
			info_dicts = {key: value[None] for (key, value) in info_dicts.items()}
		else:
			info_dicts = {key: np.concatenate((value, info_dict[key][None]), axis=0) for (key, value) in info_dicts.items()}

	# Save data
	with open(save_prefix+".pkl", 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	return info_dicts

def main(args):
	log_folder = args.log_folder
	which_cbf = args.which_cbf
	# reg_weight = args.reg_weight

	log_fldrpth = os.path.join("rollout_results", log_folder)
	if not os.path.exists(log_fldrpth):
		makedirs(log_fldrpth)

	env = CartPoleEnv()
 
	# TODO: fill out run arguments
	N_rollout = 100
	T_max = 1.5 # in seconds
	N_dt = int(T_max/env.dt)

	if which_cbf == "our_cbf_football":
		exp_name = "cartpole_reduced_debugpinch3_softplus_s1"
		checkpoint_number = 1450 
		cbf_obj = OurCBF(exp_name, checkpoint_number)
	elif which_cbf == "our_cbf_baguette":
		exp_name = "cartpole_reduced_debugpinch1_softplus_s1" 
		checkpoint_number = 1450 
		cbf_obj = OurCBF(exp_name, checkpoint_number)
	elif 'cmaes' in which_cbf:
		config_path = "rollout_cbf_classes/deprecated/cma_es_config.yaml"
		# params = run_cmaes(config_path) # TODO 

		# params = np.array([1.4, 0.2, 0.0])
		params = np.array([1.0, 0.1, 0.0])
		# params = np.array([1., 1, 0.0])
		# params = np.array([1.06870235, 0.14911268, 0.09038379])  # reg_weight = 0.1, boundary eps = 1e-2
		# params = np.array([2.04809693, 0.4586106, 0.06874261])
		cbf_obj = SSA(env)
		evaluator = CartPoleEvaluator()
		evaluator.evaluate(params)

		if params is not None:
			print("***********************************************")
			print("***********************************************")
			print(params)
			cbf_obj.set_params(params)
			print("***********************************************")
			print("***********************************************")
		else:
			print("Failed to run CMA-ES")
			sys.exit(0)

		# cbf_controller = CBFController(cbf_obj, eps_bdry=0.0, eps_outside=params[-1]) # Note for dot(phi) <= - k*phi, k is learned
	elif which_cbf == "ssa":
		params = np.array([1.0, 0.1, 0]) # Note: last one doesn't matter, overwritten by defaults in CBFController
		cbf_obj = SSA(env)
		cbf_obj.set_params(params)
	
	cbf_controller = CBFController(env, cbf_obj) 
	x_lim = cbf_controller.env.x_lim

	x0s, phi_vals_on_grid, X, Y = sample_invariant_set(x_lim, cbf_obj, N_rollout)
	save_prefix = "./rollout_results/%s/%s_" % (log_folder, which_cbf)
 
	#####################################
	# Plot x0 samples and invariant set
	#####################################
	phi_signs = plot_samples_invariant_set(x_lim, x0s, phi_vals_on_grid, X, save_prefix)

	# IPython.embed()
	# sys.exit(0)
 
	#####################################
	# Run multiple rollout_results
	#####################################
	info_dicts = run_rollout(env, N_rollout, x0s, N_dt, cbf_controller, save_prefix)
	
	#####################################
	# Sanity checks
	#####################################
	sanity_check(info_dicts)

	#####################################
	# Plot trajectories
	#####################################
	plot_trajectories(x_lim, N_rollout, x0s, phi_vals_on_grid, X, Y, phi_signs, info_dicts, save_prefix)

	#####################################
	# Plot EXITED trajectories ONLY (we choose the 5 with the largest violation)
	#####################################
	plot_exited_trajectories(x_lim, x0s, phi_vals_on_grid, X, Y, phi_signs, info_dicts, save_prefix)

	# IPython.embed()
 
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Rollout experiment')
	parser.add_argument('--log_folder', type=str, default="debug")
	parser.add_argument('--which_cbf', type=str, default="our_cbf_football")
	# parser.add_argument('--reg_weight', type=float, default=1.0, help="only relevant for cma-es")
	args = parser.parse_args()
	main(args)
