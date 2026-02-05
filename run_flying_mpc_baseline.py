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
# TODO: switching to Pathos
# from pathos.multiprocessing import ProcessingPool as PathosPool
import multiprocessing as mp
import control
from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
import datetime
from global_settings import *
# Fixed seed for repeatability
np.random.seed(2022)

g = 9.81

from create_arg_parser import create_arg_parser

from flying_mpc_utils import *

def run_mpc_on_x0(args, x0_list):
	_, mpc = setup_solver(args)
	exists_soln_bools = []
	for i, x0 in enumerate(x0_list):
		mpc.reset_history()

		mpc.x0 = x0
		mpc.set_initial_guess()

		u0 = mpc.make_step(x0)

		soln_found_mpc, _ = check_solution_found_mpc(mpc)
		exists_soln_bools.append(soln_found_mpc)

		# Has shape (2*N_horizon) because records default (0) aux first.
		# pred_cost = mpc.data['_opt_aux_num']
		# pred_cost = np.reshape(pred_cost, (-1, 2))[:, 1]
		# print(pred_cost)
		# if np.any(pred_cost > 0):
		# 	exists_soln_bools.append(0)
		# else:
		# 	exists_soln_bools.append(1)

		# print(i, x0)
	return exists_soln_bools

def sample_inside_safe_set(args, target_N_samp_inside=None, N_samp=None, x_lim_max=None):
	# print("inside sample_inside_safe_set")
	# IPython.embed()
	if args.affix:
		save_fpth = "./flying_mpc_outputs/volume_approx_%i.pkl" % args.affix
	else:
		if N_samp:
			save_fpth = "./flying_mpc_outputs/volume_approx_N_samp_%i.pkl" % (N_samp)
		if target_N_samp_inside:
			save_fpth = "./flying_mpc_outputs/volume_approx_target_N_samp_inside_%i.pkl" % (target_N_samp_inside)

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

		arguments = [[args, samples[i * n_x0_per_proc:(i + 1) * n_x0_per_proc]] for i in range(args.n_proc)]

		# Launch threads
		t0 = time.perf_counter()
		result = pool.starmap(run_mpc_on_x0, arguments)
		tf = time.perf_counter()

		for j, exists_soln_bools in enumerate(result):
			# IPython.embed()
			N_samp_found += np.sum(exists_soln_bools)
			ind = np.argwhere(exists_soln_bools).flatten()
			new_x0s = arguments[j][1][ind]
			new_x0s = np.concatenate((new_x0s, np.zeros((new_x0s.shape[0], 6))), axis=1)
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

	# print("of sample inside safe set")
	# IPython.embed()
	x0s = x0s[:target_N_samp_inside]
	domain_volume = np.prod(2*param_dict["x_lim"][:, 1])
	domain_volume_sampled = np.prod(2*x_lim[:, 1])
	percent_inside = (float(N_samp_found) / (M * i)) * 100 * (domain_volume_sampled/domain_volume)
	return x0s, percent_inside

def convert_angle_to_negpi_pi_interval(angle):
	new_angle = np.arctan2(np.sin(angle), np.cos(angle))
	return new_angle

# def simulate_rollout(args, env, N_steps_max, controller, x0s, random_seed=None):
def simulate_rollout(args, N_steps_max, x0):
	"""
	:param args:
	:param N_steps_max:
	:param x0:
	:return:
	"""
	# TODO: note, we can't pass do-mpc objects to this function...
	# ...since they are Python-wrapped C+ objects and multiprocessing can't pickle that kind of project.

	# Create env and controller
	env = FlyingInvertedPendulumEnv(model_param_dict=param_dict, dt=args.dt)
	model, mpc = setup_solver(args)
	controller = MPCController(env, mpc, model, param_dict, args)

	# Initialize data structures
	x = x0 # 1 x 16
	xs = [x]
	Us = []
	dict = None

	# For computing initial gues s
	K = compute_flying_inv_pend_LQR_matrix(args)

	# IPython.embed()
	for t in range(N_steps_max):
		# print("inside inner loop of simulate_rollout")
		# IPython.embed()

		x_initial_guess = None
		u_initial_guess = None
		if t > 0 and args.rollout_mpc_set_initial_guess:
			# compute x_initial_guess, u_initial_guess
			prev_x_pred = controller.mpc.opt_x_num['_x'] # (timestep x scenario x collocation point x x_dim ) aka (horizon + 1 x 1 x 1 x 10)
			prev_u_pred = controller.mpc.opt_x_num['_u']

			# print(prev_u_pred)
			# print(prev_x_pred)

			prev_u_pred.pop(0)
			prev_x_pred.pop(0)

			u_k = prev_u_pred[-1][0] # (4, 1)
			x_k_plus_1 = prev_x_pred[-1][0][0] # (10, 1)

			U_invariant = -K@x_k_plus_1 # type: DM
			u_invariant = env.mixer_inv@(U_invariant + np.array([env.M*g, 0, 0, 0])[:, None]) # DM, 4x1

			u_k_plus_1 = [u_invariant]
			prev_u_pred.append(u_k_plus_1)

			# Fix dimensions and types
			x_k_plus_1_env = np.concatenate((np.array(x_k_plus_1)[:, 0], np.zeros(6)))[None] # (16,)
			u_k_plus_1_env = np.array(u_invariant).T
			x_dot = env.x_dot_open_loop(x_k_plus_1_env, u_k_plus_1_env) # (1, 16)
			# x_k_plus_2 = x_k_plus_1 + env.dt * DM(x_dot[:, :10].T) # DM
			x_k_plus_2 = x_k_plus_1 + env.dt * DM(x_dot.T) # DM
			x_k_plus_2 = [[x_k_plus_2]] # TODO: this may exceed the terminal limits, due to the nature of LQR control
			prev_x_pred.append(x_k_plus_2)

			# print(prev_u_pred)
			# print(prev_x_pred)

			x_initial_guess = prev_x_pred
			u_initial_guess = prev_u_pred

		# U, debug_dict = controller.compute_control(t, x[0, :10], x_initial_guess=x_initial_guess, u_initial_guess=u_initial_guess)
		U, debug_dict = controller.compute_control(t, x[0], x_initial_guess=x_initial_guess, u_initial_guess=u_initial_guess)

		x_dot = env.x_dot_open_loop(x, U)
		x = x + env.dt * x_dot

		# Mod on angles
		# TODO: hard-coded
		# IPython.embed()
		ind = [0, 1, 2, 6, 7]
		mod_ang = convert_angle_to_negpi_pi_interval(x[:, ind])
		x[:, ind] = mod_ang

		U = np.reshape(U, (4))
		x = np.reshape(x, (1, 16))
		Us.append(U)  # (4)
		xs.append(x)  # (1, 16)

		if dict is None:
			dict = {key: [value] for (key, value) in debug_dict.items()}
		else:
			for key, value in dict.items():
				value.append(debug_dict[key])

		# print("rollout, step %i" % t)

	dict = {key: np.array(value) for (key, value) in dict.items()}
	xs = [np.reshape(x, (1, 16)) for x in xs]
	dict["x"] = np.concatenate(xs, axis=0)  # get rid of second dim = 1
	dict["U"] = np.array(Us)

	# for k, v in dict.items():
	# 	if type(v) == np.ndarray:
	# 		print(k, v.shape)
	# 	else:
	# 		print(k, len(v), v[0].shape)

	return dict


def run_rollouts(args):
	save_folder = make_mpc_rollouts_save_folder(args)

	# Save run's identifying info
	with open(os.path.join(save_folder, "args.pkl"), 'wb') as handle:
		pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(os.path.join(save_folder, "param_dict.pkl"), 'wb') as handle:
		pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Useful variables
	rollout_load_x0_fnm = args.rollout_load_x0_fnm
	x0s = pickle.load(open("./flying_mpc_outputs/%s.pkl" % rollout_load_x0_fnm, "rb"))["x0s"]
	N_desired_rollout = x0s.shape[0]

	N_steps_max = math.ceil(args.rollout_T_max/args.dt)
	print("N steps max: ", N_steps_max)

	# Initialize counters and datasets
	info_dicts = None
	N_rollout = 0
	t0 = time.perf_counter()

	while N_rollout < N_desired_rollout:
		x0 = x0s[N_rollout][None]

		info_dict = simulate_rollout(args, N_steps_max, x0)

		# IPython.embed()
		# Store data
		if info_dicts is None:
			info_dicts = {key: [value] for (key, value) in info_dict.items()}
		else:
			for key, value in info_dicts.items():
				value.append(info_dict[key])

		print("Rollout %i" % N_rollout)
		tcurr = time.perf_counter()
		sec_elapsed = tcurr - t0
		dt = datetime.timedelta(seconds=sec_elapsed)
		print("Time so far:" + str(dt))


		# Indicate this rollout has been recorded
		N_rollout += 1

		# stat_dict = extract_statistics(info_dicts)
		# info_dicts["stat_dict"] = stat_dict
		# with open(save_fpth, 'wb') as handle:
		# 	pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
		# IPython.embed()

	tf = time.perf_counter()
	sec_total = tf - t0
	info_dicts["t_total"] = sec_total

	stat_dict = extract_statistics(info_dicts)
	info_dicts["stat_dict"] = stat_dict
	with open(os.path.join(save_folder, "rollouts.pkl"), 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return info_dicts

def run_rollouts_multiproc(args):
	save_folder = make_mpc_rollouts_save_folder(args)

	# Save run's identifying info
	with open(os.path.join(save_folder, "args.pkl"), 'wb') as handle:
		pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(os.path.join(save_folder, "param_dict.pkl"), 'wb') as handle:
		pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Useful variables
	rollout_load_x0_fnm = args.rollout_load_x0_fnm
	x0s = pickle.load(open("./flying_mpc_outputs/%s.pkl" % rollout_load_x0_fnm, "rb"))["x0s"]
	N_desired_rollout = x0s.shape[0]

	N_steps_max = math.ceil(args.rollout_T_max/args.dt)
	print("N steps max: ", N_steps_max)

	# Initialize counters and datasets
	info_dicts = None
	N_rollout = 0
	it = 0
	t0 = time.perf_counter()

	# TODO; time for spawn vs fork
	ctx = mp.get_context('spawn') # TODO: default "fork" is not compatible with tensorflow
	pool = ctx.Pool(args.n_proc)
	# pool = PathosPool(args.n_proc)

	arg_tup = [args, N_steps_max]
	duplicated_arg = list([arg_tup] * args.n_proc)

	while N_rollout < N_desired_rollout:
		x0_batch = x0s[it * args.n_proc:(it + 1) * args.n_proc]

		final_arg = [duplicated_arg[i] + [x0_batch[i][None]] for i in range(x0_batch.shape[0])]

		# print("before calling starmap")
		# IPython.embed()
		result = pool.starmap(simulate_rollout, final_arg) # Hopefully, the pool can run on fewer than pool_size inputs

		for info_dict in result:
			# Store data
			if info_dicts is None:
				info_dicts = {key: [value] for (key, value) in info_dict.items()}
			else:
				for key, value in info_dicts.items():
					value.append(info_dict[key])

			N_rollout += 1 		# Indicate this rollout has been recorded

		print("Rollout %i" % N_rollout)
		tcurr = time.perf_counter()
		sec_elapsed = tcurr - t0
		dt = datetime.timedelta(seconds=sec_elapsed)
		print("Time so far:" + str(dt))

		it += 1

		stat_dict = extract_statistics(info_dicts)
		info_dicts["stat_dict"] = stat_dict
		with open(os.path.join(save_folder, "rollouts.pkl"), 'wb') as handle:
			pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
		info_dicts.pop("stat_dict") # hack, to handle the concatenation of info_dict to info_dicts

		# print("end of multiproc loop")
		# IPython.embed()

	tf = time.perf_counter()
	sec_total = tf - t0
	info_dicts["t_total"] = sec_total

	# stat_dict = extract_statistics(info_dicts)
	# info_dicts["stat_dict"] = stat_dict
	# with open(save_fpth, 'wb') as handle:
	# 	pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return info_dicts

def plot_invariant_set(args, ax=None, x_lim=None):
	t0 = time.perf_counter()

	# Create useful variables
	N_horizon = args.N_horizon
	delta = args.delta
	dt = args.dt
	which_params = args.which_params
	if x_lim is None:
		x_lim = param_dict["x_lim"]
	ind1 = state_index_dict[which_params[0]]
	ind2 = state_index_dict[which_params[1]]

	if args.affix is None:
		save_fpth_root = "./flying_mpc_outputs/new_mpc_%s_%s_dt_%.3f_N_horizon_%i_delta_%.3f" % (
		which_params[0], which_params[1], dt, N_horizon, delta)
		print("Saving 2D slice of invariant set at %s" % save_fpth_root)
	else:
		save_fpth_root = "./flying_mpc_outputs/mpc_%s" % args.affix

	# Create logger
	log_file_path = save_fpth_root + "_debug.out"
	logging.basicConfig(filename=log_file_path, filemode='w', format='%(message)s', level=logging.DEBUG)

	# Create grid over 2D slice
	# Note: assuming that we're interested in 2 variables and the other vars = 0
	x = np.arange(x_lim[ind1, 0], x_lim[ind1, 1], delta)
	y = np.arange(x_lim[ind2, 0], x_lim[ind2, 1], delta)[::-1]  # need to reverse it
	X, Y = np.meshgrid(x, y)

	sze = X.size
	print("sze: ", sze)
	input = np.zeros((sze, 16))
	input[:, ind1] = X.flatten()
	input[:, ind2] = Y.flatten()
	phi_0_vals_flat = phi_0(input)
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
	result = pool.starmap(run_mpc_on_x0, arguments)
	tf = time.perf_counter()

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
	if ax is None:
		ax = fig.add_subplot(111)

	# ax.imshow(np.logical_not(S_grid), extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])

	green_rgb = np.array([0, 204, 102]) / (255.0)
	green_rgba = np.append(green_rgb, 0.8)
	dark_green_rgb = np.array([0, 128, 64]) / 255.0
	dark_green_rgba = np.append(dark_green_rgb, 0.8)

	red_rgba = np.append(red_rgb, 0.5)
	unsafe_color = red_rgba
	# unsafe_color = np.zeros(4)
	safe_color = dark_green_rgba
	# boundary_color = dark_green_rbg

	img = np.zeros((S_grid.shape[0], S_grid.shape[1], 4))
	outside_inds = np.argwhere(np.logical_not(S_grid))
	img[outside_inds[:, 0], outside_inds[:, 1], :] = unsafe_color
	inside_inds = np.argwhere(S_grid)
	img[inside_inds[:, 0], inside_inds[:, 1], :] = safe_color

	ax.imshow(img, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
	# ax.set_aspect("equal")
	# ax.set_aspect("box")
	ax.set_aspect(2.0 / ax.get_data_ratio(), adjustable='box')

	# phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
	# ax.contour(X, Y, S_grid, levels=[0.5], colors=([np.append(boundary_color, 1.0)]), linewidths=(2,), zorder=1) # No boundary here

	ax.set_aspect("equal")

	ax.set_xlabel(which_params[0])
	ax.set_ylabel(which_params[1])

	plt.savefig(save_fpth_root + ".png", bbox_inches='tight')
	plt.clf()
	plt.close()

	return ax


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='CBF synthesis')

	# General parameters
	parser.add_argument('--which_experiments', nargs='+', default=["average_boundary", "worst_boundary", "rollout", "volume", "plot_slices", "plot_corl"], type=str)
	parser.add_argument('--n_proc', default=36, type=int)
	parser.add_argument('--affix', help='the affix for the save folder', type=str)

	parser.add_argument('--dt', type=float, default=0.05, help="dt for both MPC and FIPEnv")

	# MPC solver
	parser.add_argument('--N_horizon', type=int, default=10)
	parser.add_argument('--safety_violation_weight', default=1.0, type=float) # TODO: tune. Does larger weight give larger safe set?
	parser.add_argument('--mpc_no_terminal_safety', action='store_true', help='do not enforce that terminal state lies in invariant set?')
	# parser.add_argument('--mpc_scale_states', action='store_true', help='scale states to same [-1, 1] range?') # TODO: not implemented

	# Plotting
	parser.add_argument('--delta', type=float, default=0.1, help="discretization of grid over slice")
	parser.add_argument('--which_params', default=["phi", "dphi"], nargs='+', type=str, help="which 2 state variables")

	# Rollout
	# parser.add_argument('--rollout_N_rollout', type=int, default=5) # TODO: deprecated
	parser.add_argument('--rollout_T_max', type=float, default=2.5)
	parser.add_argument('--rollout_load_x0_fnm', type=str, help='If you want to use saved, precomputed x0.')
	parser.add_argument('--rollout_mpc_set_initial_guess', action='store_true', help='set initial guess for MPC variables (all u, all x)') # TODO: do this on default
	parser.add_argument('--rollout_u_ref', type=str, choices=["unactuated", "LQR"], default="unactuated")
	parser.add_argument('--rollout_LQR_q', type=float, default=0.1)
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
		if args.volume_target_N_samp_inside != 0:
			x0, percent_inside = sample_inside_safe_set(args, target_N_samp_inside=args.volume_target_N_samp_inside, x_lim_max=args.volume_x_lim_max)
		elif args.volume_N_samp != 0:
			x0, percent_inside = sample_inside_safe_set(args, N_samp=args.volume_N_samp, x_lim_max=args.volume_x_lim_max)
		else:
			raise ValueError("need to set volume_N_samp or volume_target_N_samp_inside")
		print("percent inside:", percent_inside)

		# nohup python -u run_flying_mpc_baseline.py --which_experiments volume --volume_N_samp 1000 --n_proc 12 &> mpc_volume.out &
		# nohup python -u run_flying_mpc_baseline.py --which_experiments volume --volume_N_samp 1000 --n_proc 12 &> mpc_volume.out &

	if "plot_slices" in args.which_experiments:
		plot_invariant_set(args)

		# python run_flying_mpc_baseline.py --which_experiments plot_slices --delta 0.5 --which_params theta dtheta
		# python run_flying_mpc_baseline.py --which_experiments plot_slices --delta 0.5 --which_params gamma dgamma
		# python run_flying_mpc_baseline.py --which_experiments plot_slices --delta 0.5 --which_params dtheta dbeta

	if "rollout" in args.which_experiments:
		assert args.rollout_load_x0_fnm is not None

		# env = FlyingInvertedPendulumEnv(model_param_dict=param_dict, dt=args.dt)
		# model, mpc = setup_solver(args)
		# controller = MPCController(env, mpc, model, param_dict, args)

		rollout_load_x0_fnm = args.rollout_load_x0_fnm
		x0s = pickle.load(open("./flying_mpc_outputs/%s.pkl" % rollout_load_x0_fnm, "rb"))["x0s"]
		N_rollout = x0s.shape[0]

		if N_rollout < 10:
			info_dicts = run_rollouts(args) # This option is for debugging
		else:
			info_dicts = run_rollouts_multiproc(args)

		# TODO: implement post-processing
		# save_fpth = "flying_mpc_outputs/new_mpc_rollouts.pkl"
		# with open(save_fpth, 'wb') as handle:
		# 	pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# TODO: implement rollout multiprocessing
		# TODO: implement preload of x0. Specify filename. Manually make sure the MPC settings match between x0 and rollout, otherwise this is invalid.
		# TODO: save rollout data after every rollout. This is good because rollouts are time-consuming.

		# TODO: are we changing between the inputs and state representation of MPC and FIPEnv properly?
		# MPC has 10D state, FIPEnv has 16D
		# MPC has lower-level inputs, FIPEnv has higher-level and gravity-compensating inputs

		# TODO: implement collection of debug data. And also delete malformed data.
		# TODO: sanity check: {"motor_impulses": motor_impulses, "smooth_clamped_motor_impulses": smooth_clamped_motor_impulses} from FIPEnv should be the same
		# TODO: sanity check: 100% of the rollouts should be safe...
		# TODO: sanity check: at every timestep in every rollout, the MPC should yield a solution. If not, u_ref = 0.

	if "plot_corl" in args.which_experiments:
		"""
		Making 2d plots for corl 
		"""
		from plot_for_corl import *

		exp_name = "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0"
		checkpoint_number = 250

		params_to_viz_list = [args.which_params]
		fnm = "%s_%s_slice" % (args.which_params[0], args.which_params[1])

		fldr_path = os.path.join("./log", exp_name)

		ub = 20.0 # TODO
		thresh = np.array([math.pi / 3, math.pi / 3, math.pi, ub, ub, ub, math.pi / 3, math.pi / 3, ub, ub],
		                  dtype=np.float32)  # angular velocities bounds probably much higher in reality (~10-20 for drone, which can do 3 flips in 1 sec).

		x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)  # (13, 2)
		constants_for_other_params_list = [np.zeros(10)]
		###############################
		# Now, the baseline
		baseline_phi_fn, baseline_param_dict = load_philow_and_params()  # TODO: this assumes default param_dict for dynamics
		baseline_param_dict["x_lim"] = x_lim

		mu = [3.75977875, 0., 0.01]
		state_dict = {"ki": torch.tensor([[mu[2]]]),
		              "ci": torch.tensor([[mu[0]], [mu[1]]])}  # todo: this is not very generic
		baseline_phi_fn.load_state_dict(state_dict, strict=False)

		axs = plot_invariant_set_slices(baseline_phi_fn, baseline_param_dict, fldr_path=fldr_path,
		                                           which_params=params_to_viz_list,
		                                           constants_for_other_params=constants_for_other_params_list, fnm=fnm, safe_color="purple", delta=args.delta)
		# ###############################
		phi_fn, param_dict = load_phi_and_params(exp_name, checkpoint_number)
		param_dict["x_lim"] = x_lim
		axs = plot_invariant_set_slices(phi_fn, param_dict, fldr_path=fldr_path,
		                                           which_params=params_to_viz_list,
		                                           constants_for_other_params=constants_for_other_params_list, fnm=fnm,
		                                           pass_axs=axs, safe_color="blue", delta=args.delta)

		# Safe MPC
		# IPython.embed()
		ax = plot_invariant_set(args, ax=axs[0, 0], x_lim=x_lim)

		plt.savefig("./corl_media/%s_%s_delta_%.2f.png" % (args.which_params[0], args.which_params[1], args.delta), bbox_inches='tight')
		plt.clf()
		plt.close()

		"""
		call script as: 
		python run_flying_mpc_baseline.py --which_experiments plot_corl --delta 1.0 --which_params dtheta dbeta 
		python run_flying_mpc_baseline.py --which_experiments plot_corl --delta 0.1 --which_params dtheta dbeta 
		python run_flying_mpc_baseline.py --which_experiments plot_corl --delta 0.5 --which_params gamma dgamma
		python run_flying_mpc_baseline.py --which_experiments plot_corl --delta 0.5 --which_params theta dtheta	 
		"""

	# print("ln 235")
	# IPython.embed()




	""""# Test 1: do FIPenv and MPC model match?
	model, mpc = setup_solver(args)

	# Checking model is implemented correctly
	print("Checking model is implemented correctly")

	simulator = do_mpc.simulator.Simulator(model)
	simulator.set_param(t_step=args.dt)
	simulator.setup()

	np.random.seed(0)
	x0 = np.random.rand(16)
	u = np.random.uniform(size=(4))
	print("x0", x0)
	print("u", u)

	default_env = FlyingInvertedPendulumEnv(model_param_dict=param_dict, dt=args.dt)
	U = default_env.mixer@u[:, None] - np.array([param_dict["M"]*g, 0, 0, 0])[:, None]
	U = U.flatten()
	print(U)
	# IPython.embed()
	x_dot = default_env.x_dot_open_loop(x0, U)
	x1_env = x0 + args.dt*x_dot

	# mpc.x0 = x0[:10]
	# mpc.u0 = np.zeros(4)
	# mpc.set_initial_guess()
	# u = mpc.make_step(x0[:10])
	#
	# state_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta"]
	# pred = [mpc.data.prediction(('_x', state_name)) for state_name in state_names]
	# pred = np.concatenate(pred, axis=0)[:, :, 0]  # 10 x horizon
	# x1_mpc = pred[:, 1]

	simulator.x0 = x0[:10]
	meas_data = simulator.make_step(u[:, None])
	x1_mpc = meas_data

	print("\n")
	print(np.linalg.norm(x1_env.flatten()[:10] - x1_mpc.flatten()))
	print(x1_env.flatten()[:10])
	print(x1_mpc.flatten())"""

	"""# Test 2: does MPC and simulator match?
	model, mpc = setup_solver(args)

	# Checking model is implemented correctly
	print("Checking model is implemented correctly")

	simulator = do_mpc.simulator.Simulator(model)
	simulator.set_param(t_step=args.dt)
	simulator.setup()

	np.random.seed(0)
	x0 = np.random.rand(16)
	print("x0", x0)

	mpc.x0 = x0[:10]
	mpc.u0 = np.zeros(4)
	mpc.set_initial_guess()
	u = mpc.make_step(x0[:10])

	state_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta"]
	pred = [mpc.data.prediction(('_x', state_name)) for state_name in state_names]
	pred = np.concatenate(pred, axis=0)[:, :, 0]  # 10 x horizon
	x1_mpc = pred[:, 1]

	simulator.x0 = x0[:10]
	meas_data = simulator.make_step(u)
	x1_env = meas_data

	print("\n")
	print(np.linalg.norm(x1_env.flatten()[:10] - x1_mpc.flatten()))
	print(x1_env.flatten()[:10])
	print(x1_mpc.flatten())
	# print(pred[:, 0].flatten())
	IPython.embed()"""


	# # Look at mpc output
	# mpc.reset_history()
	# mpc.x0 = x
	# mpc.set_initial_guess()
	# u0 = mpc.make_step(x)
	#
	# # Has shape (2*N_horizon) because records default (0) aux first.
	# print("Looking at the output of MPC")
	# print(mpc.data["_opt_aux_num"])
	# print(mpc.data["_opt_x_num"].shape)
	# IPython.embed()
	# """

	# pred_cost = mpc.data['_opt_aux_num']
	# pred_cost = np.reshape(pred_cost, (-1, 2))[:, 1]
	# print(pred_cost)
	# if np.any(pred_cost > 0):
	# 	exists_soln_bools.append(0)
	# else:
	# 	exists_soln_bools.append(1)

	# x0, percent_inside = sample_inside_safe_set(args, 5)

	# IPython.embed()
	# x0 = [np.zeros((10))]
	# x0 = np.random.normal(scale=1.0, size=(5, 10))
	# rv = run_mpc_on_x0(args, x0)

	# env = FlyingInvertedPendulumEnv(model_param_dict=param_dict)
	# model, mpc = setup_solver(args)
	# controller = MPCController(env, mpc, model, param_dict, args)
	#
	# run_rollouts(args, env, controller)

	# plot_invariant_set(args)
	# python run_flying_mpc_baseline.py --delta 0.2 --N_horizon 10 --affix debug_new_mpc
