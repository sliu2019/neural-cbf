"""
Computes largest safe set in a brute-force way, by using MPC
"""
import do_mpc
import IPython
import numpy as np, math
import matplotlib.pyplot as plt
import torch
from casadi import *
import pickle
import argparse
import time
import sys
import logging
import multiprocessing as mp
import control
from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
import datetime
# Fixed seed for repeatability
np.random.seed(2022)

g = 9.81

from create_arg_parser import create_arg_parser
from flying_backup_set_utils import *


"""
parser.add_argument('--flow_dt', type=float, default=0.05, help="dt for discretized computation of the dynamics flow")
parser.add_argument('--flow_T', type=float, default=1.0, help="length (in total time) of dynamics flow")
parser.add_argument('--backup_LQR_q', type=float, default=1.0, help="assume backup controller is LQR")
parser.add_argument('--backup_LQR_r', type=float, default=1.0, help="assume backup controller is LQR")
"""
def run_backup_set_on_x0(args, x_batch):
	"""
	Checks if x0 are in the implicit viable set
	:param args:
	:param x_batch: (None, 16)
	:return: (None, )
	"""
	# print("inside run_backup_set_on_x0")
	# IPython.embed()

	sim_env = FlyingInvertedPendulumEnv(model_param_dict=param_dict, dt=args.flow_dt)
	xdot_fn = sim_env.x_dot_open_loop_model
	backup_K = compute_flying_inv_pend_LQR_matrix(args.backup_LQR_q, args.backup_LQR_r)

	h_fn = H(param_dict)
	hb_fn = Hb(param_dict)

	bs = x_batch.shape[0]
	whole_rollout_safe = (h_fn(torch.from_numpy(x_batch)) <= 0).numpy().flatten()
	dt = args.flow_dt

	for i in range(math.ceil(args.flow_T/dt)):
		u_batch = -x_batch@backup_K.T
		x_batch = x_batch + dt*xdot_fn(x_batch, u_batch)

		# Check if the whole trajectory was inside the safe set
		rollout_safe_at_time_i = (h_fn(torch.from_numpy(x_batch)) <= 0).numpy().flatten()
		whole_rollout_safe = rollout_safe_at_time_i*whole_rollout_safe

		# print(x_batch)

	# Compute on final state
	x_reach_invariant = (hb_fn(torch.from_numpy(x_batch)) <= 0).numpy().flatten()
	x_inside_implicit_safe_set = np.logical_and(x_reach_invariant, whole_rollout_safe)
	x_inside_implicit_safe_set = x_inside_implicit_safe_set.flatten()
	return x_inside_implicit_safe_set

def sample_inside_safe_set(args, target_N_samp_inside=None, N_samp=None, x_lim_max=None):
	# print("inside sample_inside_safe_set")
	# IPython.embed()
	if args.affix:
		save_fpth = "./flying_backup_set_outputs/volume_approx_%i.pkl" % args.affix
	else:
		if N_samp:
			save_fpth = "./flying_backup_set_outputs/volume_approx_N_samp_%i.pkl" % (N_samp)
		if target_N_samp_inside:
			save_fpth = "./flying_backup_set_outputs/volume_approx_target_N_samp_inside_%i.pkl" % (target_N_samp_inside)

	# Define some variables
	x_dim = param_dict["x_dim"]
	if x_lim_max is not None:
		# IPython.embed()
		x_lim = np.concatenate((-np.array(x_lim_max)[:, None], np.array(x_lim_max)[:, None]), axis=1)
	else:
		x_lim = param_dict["x_lim"]

	print("x_lim in sample_inside_safe_set")
	print(x_lim)
	box_side_lengths = x_lim[:, 1] - x_lim[:, 0]

	# IPython.embed()

	M = 1000
	N_samp_found = 0
	i = 0 # repetitions of loop below

	ctx = mp.get_context('fork')
	pool = ctx.Pool(args.n_proc)
	n_x0_per_proc = math.ceil(M / args.n_proc)
	x0s = np.zeros((0, 16))

	d = {"x0s": x0s, "N_samp": 0, "N_samp_inside": 0}
	while True:
		samples = np.random.rand(M, x_dim)
		samples = samples * box_side_lengths + x_lim[:, 0]
		samples = np.concatenate((samples, np.zeros((M, 6))), axis=1)

		arguments = [[args, samples[i * n_x0_per_proc:(i + 1) * n_x0_per_proc]] for i in range(args.n_proc)]

		# Launch threads
		t0 = time.perf_counter()
		result = pool.starmap(run_backup_set_on_x0, arguments)
		tf = time.perf_counter()

		for j, exists_soln_bools in enumerate(result):
			# IPython.embed()
			N_samp_found += np.sum(exists_soln_bools)
			ind = np.argwhere(exists_soln_bools).flatten()
			new_x0s = arguments[j][1][ind]
			# new_x0s = np.concatenate((new_x0s, np.zeros((new_x0s.shape[0], 6))), axis=1)
			# print(x0s.shape, new_x0s.shape)
			x0s = np.concatenate((x0s, new_x0s), axis=0)

		print("*******************************************")
		print("**********   Round %i completed   *********" % i)
		print("Found %i samples so far" % (N_samp_found))
		print("*******************************************")

		i += 1

		d = {"x0s": x0s, "N_samp": (M*i), "N_samp_inside": N_samp_found, "x_lim": x_lim}

		print("Saving at: ", save_fpth)
		with open(save_fpth, 'wb') as handle:
			pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

		if target_N_samp_inside:
			if N_samp_found >= target_N_samp_inside:
				break
		if N_samp:
			if (M*i) >= N_samp:
				break

		# print("after 1 round")
		# IPython.embed()

	# print("of sample inside safe set")
	# IPython.embed()
	x0s = x0s[:target_N_samp_inside]
	domain_volume = np.prod(2*param_dict["x_lim"][:, 1])
	domain_volume_sampled = np.prod(2*x_lim[:, 1])
	percent_inside = (float(N_samp_found) / (M * i)) * 100 * (domain_volume_sampled/domain_volume)
	return x0s, percent_inside

def plot_invariant_set(args):
	# print("inside plot_invariant_set")
	# IPython.embed()

	t0 = time.perf_counter()

	# Create useful variables
	delta = args.delta
	which_params = args.which_params
	x_lim = param_dict["x_lim"]
	ind1 = state_index_dict[which_params[0]]
	ind2 = state_index_dict[which_params[1]]
	fixed_states = [args.viz_gamma, args.viz_beta, args.viz_alpha, args.viz_dgamma, args.viz_dbeta, args.viz_dalpha, args.viz_phi, args.viz_theta, args.viz_dphi, args.viz_dtheta]
	fixed_state_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
	                     "dtheta"] #, "x", "y", "z", "dx", "dy", "dz"]
	if args.affix is None:
		flow_N_steps = math.ceil(args.flow_T / args.flow_dt)
		save_fpth_root = "./flying_backup_set_outputs/2d_viz_%s_%s_flow_dt_%.3f_length_%i_LQR_q_%.2f_r_%.2f" % (which_params[0], which_params[1], args.flow_dt, flow_N_steps, args.backup_LQR_q, args.backup_LQR_r)

		for j, fixed_state in enumerate(fixed_states):
			if fixed_state != 0:
				save_fpth_root += "_%s_%.2f" % (fixed_state_names[j], fixed_state)

		print("Saving 2D slice of invariant set at %s" % save_fpth_root)
	else:
		save_fpth_root = "./flying_backup_set_outputs/2d_viz_%s" % args.affix

	# Create logger
	# log_file_path = save_fpth_root + "_debug.out"
	# logging.basicConfig(filename=log_file_path, filemode='w', format='%(message)s', level=logging.DEBUG)

	# Create grid over 2D slice
	# Note: assuming that we're interested in 2 variables and the other vars = 0
	x = np.arange(x_lim[ind1, 0], x_lim[ind1, 1], delta)
	y = np.arange(x_lim[ind2, 0], x_lim[ind2, 1], delta)[::-1]  # need to reverse it
	X, Y = np.meshgrid(x, y)

	sze = X.size
	print("sze: ", sze)
	input = np.zeros((sze, 16))

	# TODO: hardcoded index mapping, doesn't really matter
	for j, fixed_state in enumerate(fixed_states):
		input[:, j] = fixed_state

	input[:, ind1] = X.flatten()
	input[:, ind2] = Y.flatten()

	# input[:, 0] = args.viz_gamma
	# input[:, 1] = args.viz_beta
	# input[:, 2] = args.viz_alpha
	# input[:, 3] = args.viz_dgamma
	# input[:, 4] = args.viz_dbeta
	# input[:, 5] = args.viz_dalpha
	# input[:, 6] = args.viz_phi
	# input[:, 7] = args.viz_theta
	# input[:, 8] = args.viz_dphi
	# input[:, 9] = args.viz_dtheta

	# IPython.embed()

	h_fn = H(param_dict)
	phi_0_vals_flat = h_fn(torch.from_numpy(input)).numpy().flatten()
	phi_0_vals = np.reshape(phi_0_vals_flat, X.shape)
	# neg_inds = np.argwhere(phi_0_vals <= 0) # (m, 2)
	flat_neg_inds = np.argwhere(phi_0_vals_flat <= 0)[:, 0]

	# IPython.embed()
	ctx = mp.get_context('fork')  # TODO: try spawn and fork
	pool = ctx.Pool(args.n_proc)
	n_x0 = flat_neg_inds.size
	n_x0_per_proc = math.ceil(n_x0 / args.n_proc)
	permuted_inds = np.random.permutation(flat_neg_inds)  # TODO: why do we need to do this?
	chunked_permuted_inds = [permuted_inds[i * n_x0_per_proc:(i + 1) * n_x0_per_proc] for i in range(args.n_proc)]
	arguments = [[args, input[x]] for x in chunked_permuted_inds]

	# Launch threads
	t0 = time.perf_counter()
	result = pool.starmap(run_backup_set_on_x0, arguments)
	tf = time.perf_counter()

	# print("line 246")
	# IPython.embed()
	# Don't multithread
	# result = []
	# for i in range(len(flat_neg_inds)):
	# 	r = run_backup_set_on_x0(args, [input[flat_neg_inds[i]]])
	# 	result.append(r)

	exists_soln_bools = np.zeros(sze)
	for i, r in enumerate(result):
		exists_soln_bools[chunked_permuted_inds[i]] = r  # TODO: does fork return RV in the same order as args?

	S_grid = np.reshape(exists_soln_bools, X.shape)
	t_per_mpc = []

	# Save data
	A_grid = (phi_0_vals <= 0).astype("int")
	percent_of_A_volume = np.sum(S_grid) * 100.0 / np.sum(A_grid)
	save_dict = {"A_grid": A_grid, "X": X, "Y": Y, "exists_soln_bools": exists_soln_bools, "S_grid": S_grid,
	             "args": args, "t_per_mpc": t_per_mpc, "t_total": (tf - t0), "percent_of_A_volume": percent_of_A_volume}
	with open(save_fpth_root + ".pkl", 'wb') as handle:
		pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Plotting
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(np.logical_not(S_grid), extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
	ax.set_aspect("equal")

	ax.set_xlabel(which_params[0])
	ax.set_ylabel(which_params[1])

	plt.savefig(save_fpth_root + ".png", bbox_inches='tight')
	plt.clf()
	plt.close()

def convert_angle_to_negpi_pi_interval(angle):
	new_angle = np.arctan2(np.sin(angle), np.cos(angle))
	return new_angle

def simulate_rollout(env, N_steps_max, controller, x0, random_seed=None):
	"""
	:param env:
	:param N_steps_max:
	:param controller:
	:param x0: (1,16)
	:param random_seed:
	:return:
	"""
	# Random seed is for multiproc
	if random_seed:
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)

	# Initialize data structures
	# Note: x is 1 x 16
	x0 = np.reshape(x0, (1, 16))
	x = x0
	xs = [x]
	us = []
	dict = None

	# IPython.embed()
	u_safe_applied = False
	t_since = 0
	for t in range(N_steps_max):
		# print("rollout, step %i" % t)
		u, debug_dict = controller.compute_control(t, x)  # Define this
		x_dot = env.x_dot_open_loop(x, u)
		x = x + env.dt * x_dot

		# Mod on angles
		ind = [0, 1, 2, 6, 7]
		mod_ang = convert_angle_to_negpi_pi_interval(x[:, ind])
		x[:, ind] = mod_ang

		u = np.reshape(u, (4))
		x = np.reshape(x, (1, 16))
		us.append(u) # (4)
		xs.append(x) # (1, 16)

		if dict is None:
			dict = {key: [value] for (key, value) in debug_dict.items()}
		else:
			for key, value in dict.items():
				value.append(debug_dict[key])

		if debug_dict["apply_u_safe"]:
			u_safe_applied = True

		if u_safe_applied:
			t_since += 1
		if t_since > 1:
			break

	# IPython.embed()
	dict = {key: np.array(value) for (key, value) in dict.items()}
	xs = [np.reshape(x, (1, 16)) for x in xs]
	dict["x"] = np.concatenate(xs, axis=0) # get rid of second dim = 1
	dict["u"] = np.array(us)

	return dict

def run_rollouts(x0s, env, controller, N_steps_max, save_folder):
	# Set up
	info_dicts = None
	N_rollout = 0
	t0 = time.perf_counter()

	for x0 in x0s:
		print("Rollout %i" % N_rollout)
		info_dict = simulate_rollout(env, N_steps_max, controller, x0)

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
	tf = time.perf_counter()
	sec_total = tf - t0
	info_dicts["t_total"] = sec_total

	stat_dict = extract_statistics(info_dicts)
	info_dicts["stat_dict"] = stat_dict
	with open(os.path.join(save_folder, "rollouts.pkl"), 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return info_dicts

def run_rollouts_multiproc(x0s, env, controller, N_steps_max, save_folder, verbose=False, n_proc=None):
	info_dicts = None
	N_rollout = 0
	N_x0 = len(x0s) # TODO: or .shape[0]?
	t0 = time.perf_counter()

	if n_proc is None:
		n_proc = mp.cpu_count()
	ctx = mp.get_context('spawn') # TODO: default "fork" is not compatible with tensorflow
	pool = ctx.Pool(n_proc)

	arg_tup = [env, N_steps_max, controller]
	duplicated_arg = list([arg_tup] * n_proc)

	N_it = math.ceil(N_x0/n_proc)
	for i in range(N_it):

		x0s_batch = x0s[i*n_proc:(i+1)*n_proc]
		final_arg = [duplicated_arg[i] + [x0s_batch[i]] for i in range(len(x0s_batch))]
		result = pool.starmap(simulate_rollout, final_arg)

		for info_dict in result:
			if not np.any(info_dict["apply_u_safe"]):
				print("invalid rollout")
				continue

			# Store data
			if info_dicts is None:
				info_dicts = {key: [value] for (key, value) in info_dict.items()}
			else:
				for key, value in info_dicts.items():
					value.append(info_dict[key])

			N_rollout += 1 		# Indicate this rollout has been recorded

		if verbose:
			print(N_rollout)

		if i == N_it - 1:
			tf = time.perf_counter()
			sec_total = tf - t0
			info_dicts["t_total"] = sec_total

		stat_dict = extract_statistics(info_dicts)
		info_dicts["stat_dict"] = stat_dict
		with open(os.path.join(save_folder, "rollouts.pkl"), 'wb') as handle:
			pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
		info_dicts.pop("stat_dict") # hack, to handle the concatenation of info_dict to info_dicts


	print("\n\n")
	print("N rollouts out of N x0: %i/%i" % (N_rollout, N_x0))

	return info_dicts


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='CBF synthesis')

	# General parameters
	parser.add_argument('--which_experiments', nargs='+', default=["rollout", "volume", "plot_slices"], type=str)
	parser.add_argument('--n_proc', default=36, type=int)
	parser.add_argument('--affix', help='the affix for the save folder', type=str)

	parser.add_argument('--dt', type=float, default=0.1, help="dt for simulation (environment and controller)")

	# MPC solver
	# parser.add_argument('--N_horizon', type=int, default=10)
	# parser.add_argument('--safety_violation_weight', default=1.0, type=float) # TODO: tune. Does larger weight give larger safe set?
	# parser.add_argument('--mpc_no_terminal_safety', action='store_true', help='do not enforce that terminal state lies in invariant set?')
	# parser.add_argument('--mpc_scale_states', action='store_true', help='scale states to same [-1, 1] range?') # TODO: not implemented

	# Backup set
	parser.add_argument('--flow_dt', type=float, default=0.1, help="dt for discretized computation of the dynamics flow")
	parser.add_argument('--flow_T', type=float, default=1.0, help="length (in total time) of dynamics flow")
	parser.add_argument('--backup_LQR_q', type=float, default=0.05, help="assume backup controller is LQR")
	parser.add_argument('--backup_LQR_r', type=float, default=1.0, help="assume backup controller is LQR")

	# Plotting
	parser.add_argument('--delta', type=float, default=0.1, help="discretization of grid over slice")
	parser.add_argument('--which_params', default=["phi", "dphi"], nargs='+', type=str, help="which 2 state variables")
	parser.add_argument('--viz_gamma', default=0.0, type=float)
	parser.add_argument('--viz_beta', default=0.0, type=float)
	parser.add_argument('--viz_alpha', default=0.0, type=float)
	parser.add_argument('--viz_dgamma', default=0.0, type=float)
	parser.add_argument('--viz_dbeta', default=0.0, type=float)
	parser.add_argument('--viz_dalpha', default=0.0, type=float)
	parser.add_argument('--viz_phi', default=0.0, type=float)
	parser.add_argument('--viz_theta', default=0.0, type=float)
	parser.add_argument('--viz_dphi', default=0.0, type=float)
	parser.add_argument('--viz_dtheta', default=0.0, type=float)

	# Rollout
	# parser.add_argument('--rollout_N_rollout', type=int, default=5) # TODO: deprecated
	parser.add_argument('--rollout_T_max', type=float, default=2.5)
	parser.add_argument('--rollout_load_x0_fnm', type=str, help='If you want to use saved, precomputed x0.')
	# parser.add_argument('--rollout_mpc_set_initial_guess', action='store_true', help='set initial guess for MPC variables (all u, all x)') # TODO: do this on default
	parser.add_argument('--rollout_u_ref', type=str, choices=["unactuated", "LQR"], default="unactuated")
	parser.add_argument('--rollout_LQR_q', type=float, default=0.05)
	parser.add_argument('--rollout_LQR_r', type=float, default=1.0)

	# Volume
	parser.add_argument('--volume_N_samp', type=int, default=0) # 100K
	parser.add_argument('--volume_target_N_samp_inside', type=int, default=0)
	parser.add_argument('--volume_x_lim_max', nargs='+', type=float, help="for sampling in a smaller box; specify x_lim array one entry at a time")

	args = parser.parse_known_args()[0]

	# IPython.embed()
	# print(args.which_experiments)
	if "volume" in args.which_experiments:
		# IPython.embed()
		# run_backup_set_on_x0(args, np.random.rand(10, 16))

		if args.volume_target_N_samp_inside != 0:
			x0, percent_inside = sample_inside_safe_set(args, target_N_samp_inside=args.volume_target_N_samp_inside, x_lim_max=args.volume_x_lim_max)
		elif args.volume_N_samp != 0:
			x0, percent_inside = sample_inside_safe_set(args, N_samp=args.volume_N_samp, x_lim_max=args.volume_x_lim_max)
		else:
			raise ValueError("need to set volume_N_samp or volume_target_N_samp_inside")
		print("percent inside:", percent_inside)

		# nohup python -u run_flying_backup_set_baseline.py --which_experiments volume --volume_N_samp 1000 --n_proc 12 &> mpc_volume.out &
		# nohup python -u run_flying_backup_set_baseline.py --which_experiments volume --volume_target_N_samp_inside 10 --n_proc 12 &> debug_backup_set_volume.out &
	if "plot_slices" in args.which_experiments:
		plot_invariant_set(args)

		# python run_flying_backup_set_baseline.py --which_experiments plot_slices --delta 0.5 --which_params theta dtheta
		# python run_flying_backup_set_baseline.py --which_experiments plot_slices --delta 0.5 --which_params gamma dgamma
		# python run_flying_backup_set_baseline.py --which_experiments plot_slices --delta 0.5 --which_params dtheta dbeta

	if "rollout" in args.which_experiments:
		assert args.rollout_load_x0_fnm is not None

		sim_env = FlyingInvertedPendulumEnv(model_param_dict=param_dict, dt=args.dt)
		controller = BackupSetController(sim_env, param_dict, args)

		rollout_load_x0_fnm = args.rollout_load_x0_fnm
		x0s = pickle.load(open("./flying_backup_set_outputs/%s.pkl" % rollout_load_x0_fnm, "rb"))["x0s"]
		N_x0 = x0s.shape[0]

		N_steps_max = math.ceil(args.rollout_T_max/args.dt)
		print("%i time steps max" % N_steps_max)

		save_folder = make_backup_set_rollouts_save_folder(args)
		# Save run's identifying info
		with open(os.path.join(save_folder, "args.pkl"), 'wb') as handle:
			pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

		with open(os.path.join(save_folder, "param_dict.pkl"), 'wb') as handle:
			pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		info_dicts = run_rollouts(x0s, sim_env, controller, N_steps_max, save_folder)

		# if N_x0 <= 10:
		# 	info_dicts = run_rollouts(x0s, sim_env, controller, N_steps_max, save_folder)
		# else:
		# 	info_dicts = run_rollouts_multiproc(x0s, sim_env, controller, N_steps_max, save_folder, verbose=True, n_proc=args.n_proc)


