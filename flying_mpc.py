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

# Fixed seed for repeatability
np.random.seed(2022)

g = 9.81

from src.argument import create_parser

parser = create_parser()  # default
default_args = parser.parse_known_args()[0]
from main import create_flying_param_dict

param_dict = create_flying_param_dict(default_args)  # default
state_index_dict = param_dict["state_index_dict"]

def setup_solver(args):
	N_horizon = args.N_horizon
	dt = args.dt

	model_type = 'continuous'  # either 'discrete' or 'continuous'
	model = do_mpc.model.Model(model_type)

	# Define state vars
	# TODO: has to be done in this order: 	state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta"]
	gamma = model.set_variable(var_type='_x', var_name='gamma', shape=(1, 1))
	beta = model.set_variable(var_type='_x', var_name='beta', shape=(1, 1))
	alpha = model.set_variable(var_type='_x', var_name='alpha', shape=(1, 1))

	dgamma = model.set_variable(var_type='_x', var_name='dgamma', shape=(1, 1))
	dbeta = model.set_variable(var_type='_x', var_name='dbeta', shape=(1, 1))
	dalpha = model.set_variable(var_type='_x', var_name='dalpha', shape=(1, 1))

	phi = model.set_variable(var_type='_x', var_name='phi', shape=(1,1))
	theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))

	dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(1,1))
	dtheta = model.set_variable(var_type='_x', var_name='dtheta', shape=(1,1))

	# Define input
	u = model.set_variable(var_type='_u', var_name='u', shape=(4, 1)) # TODO: note this is lower-level controls!!!

	# Define dynamics
	k1 = param_dict["k1"]
	k2 = param_dict["k2"]
	l = param_dict["l"]
	M = param_dict["M"]
	J_x = param_dict["J_x"]
	J_y = param_dict["J_y"]
	J_z = param_dict["J_z"]
	L_p = param_dict["L_p"]
	delta_safety_limit = param_dict["delta_safety_limit"]

	Mixer = SX([[k1, k1, k1, k1], [0, -l * k1, 0, l * k1], [l * k1, 0, -l * k1, 0], [-k2, k2, -k2, k2]])  # mixer matrix

	R = SX.zeros(3, 3)
	R[0, 0] = cos(alpha) * cos(beta)
	R[0, 1] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)
	R[0, 2] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)
	R[1, 0] = sin(alpha) * cos(beta)
	R[1, 1] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma)
	R[1, 2] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)
	R[2, 0] = -sin(beta)
	R[2, 1] = cos(beta) * sin(gamma)
	R[2, 2] = cos(beta) * cos(gamma)

	k_x = R[0, 2]
	k_y = R[1, 2]
	k_z = R[2, 2]

	U = Mixer@u # TODO: high-level control inputs, (4x1)
	F = U[0, 0]

	###### Computing state derivatives
	J = SX([[J_x], [J_y], [J_z]])
	norm_torques = U[1:] * (1.0 / J)

	ddquad_angles = R@norm_torques
	ddgamma = ddquad_angles[0, 0]
	ddbeta = ddquad_angles[1, 0]
	ddalpha = ddquad_angles[2, 0]

	ddphi = (3.0) * (k_y * cos(phi) + k_z * sin(phi)) / (
				2 * M * L_p * cos(theta)) * F + 2 * dtheta * dphi * tan(theta)
	ddtheta = (3.0 * (
				-k_x * cos(theta) - k_y * sin(phi) * sin(theta) + k_z * cos(phi) * sin(
			theta)) / (2.0 * M * L_p)) * F - (dphi**2) * sin(theta) * cos(theta)

	model.set_rhs('gamma', dgamma)
	model.set_rhs('beta', dbeta)
	model.set_rhs('alpha', dalpha)

	model.set_rhs('dgamma', ddgamma)
	model.set_rhs('dbeta', ddbeta)
	model.set_rhs('dalpha', ddalpha)

	model.set_rhs('theta', dtheta)
	model.set_rhs('phi', dphi)

	model.set_rhs('dtheta', ddtheta)
	model.set_rhs('dphi', ddphi)

	# Define logging functions (we log the cost function)
	### Convert all angles to [-pi, pi] interval
	### TODO: this is important, as it matches our assumption in the phi0 function
	gamma_int = arctan2(sin(gamma), cos(gamma))
	beta_int = arctan2(sin(beta), cos(beta))
	phi_int = arctan2(sin(phi), cos(phi))
	theta_int = arctan2(sin(theta), cos(theta))

	cos_cos = cos(theta_int) * cos(phi_int)
	eps = 1e-4  # prevents nan when cos_cos = +/- 1 (at x = 0)
	signed_eps = -sign(cos_cos) * eps
	delta = acos(cos_cos + signed_eps)
	rho = delta ** 2 + gamma_int ** 2 + beta_int ** 2 - delta_safety_limit ** 2
	cost = fmax(0, rho)  # we are in a casadi symbolic environment
	# cost = rho # TODO
	model.set_expression('cost', cost)

	# Finally,
	model.setup()

	#######################
	# Create optimizer
	mpc = do_mpc.controller.MPC(model)
	# Last entry suppresses IPOPT output
	setup_mpc = {
		'n_horizon': N_horizon,
		't_step': dt,
		'store_full_solution': True,
		'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
	}

	mpc.set_param(**setup_mpc)

	# Define cost function
	### You have to redefine this, even though it's defined as cost above
	gamma_int = arctan2(sin(gamma), cos(gamma))
	beta_int = arctan2(sin(beta), cos(beta))
	phi_int = arctan2(sin(phi), cos(phi))
	theta_int = arctan2(sin(theta), cos(theta))

	cos_cos = cos(theta_int) * cos(phi_int)
	eps = 1e-4  # prevents nan when cos_cos = +/- 1 (at x = 0)
	signed_eps = -sign(cos_cos) * eps
	delta = acos(cos_cos + signed_eps)
	rho = delta ** 2 + gamma_int ** 2 + beta_int ** 2 - delta_safety_limit ** 2
	# lterm = rho
	lterm = fmax(0, rho)  # we are in a casadi symbolic environment
	mpc.set_objective(lterm=lterm, mterm=lterm)

	# Set state and control limits
	# No state limits
	mpc.bounds['lower', '_u', 'u'] = SX.zeros(4, 1)
	mpc.bounds['upper', '_u', 'u'] = SX.ones(4, 1)

	# Finally,
	mpc.setup()

	#################################
	return model, mpc

def phi_0(x):
	theta = x[:, [state_index_dict["theta"]]]
	phi = x[:, [state_index_dict["phi"]]]
	gamma = x[:, [state_index_dict["gamma"]]]
	beta = x[:, [state_index_dict["beta"]]]
	delta_safety_limit = param_dict["delta_safety_limit"]

	cos_cos = np.cos(theta) * np.cos(phi)
	eps = 1e-4  # prevents nan when cos_cos = +/- 1 (at x = 0)
	signed_eps = -np.sign(cos_cos) * eps
	delta = np.arccos(cos_cos + signed_eps)
	rv = delta ** 2 + gamma ** 2 + beta ** 2 - delta_safety_limit ** 2
	return rv

def run_mpc_on_x0(args, x0_list):
	model, mpc = setup_solver(args)
	exists_soln_bools = []
	for i, x0 in enumerate(x0_list):

		mpc.reset_history()

		mpc.x0 = x0
		mpc.set_initial_guess()

		u0 = mpc.make_step(x0)

		# Has shape (2*N_horizon) because records default (0) aux first.
		pred_cost = mpc.data['_opt_aux_num']
		pred_cost = np.reshape(pred_cost, (-1, 2))[:, 1]
		if np.any(pred_cost > 0):
			exists_soln_bools.append(0)
		else:
			exists_soln_bools.append(1)

		print(i, x0)

	return exists_soln_bools

def mpc_compute_invariant_set(args):
	t0 = time.perf_counter()

	# Create useful variables
	N_horizon = args.N_horizon
	delta = args.delta
	dt = args.dt
	which_params = args.which_params
	x_lim = param_dict["x_lim"]
	ind1 = state_index_dict[which_params[0]]
	ind2 = state_index_dict[which_params[1]]

	if args.affix is None:
		save_fpth_root = "./flying_mpc_outputs/mpc_%s_%s_dt_%.3f_N_horizon_%i_delta_%.3f" % (which_params[0], which_params[1], dt, N_horizon, delta)
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
	input = np.zeros((sze, 10))
	input[:, ind1] = X.flatten()
	input[:, ind2] = Y.flatten()
	phi_0_vals_flat = phi_0(input)
	phi_0_vals = np.reshape(phi_0_vals_flat, X.shape)
	# neg_inds = np.argwhere(phi_0_vals <= 0) # (m, 2)
	flat_neg_inds = np.argwhere(phi_0_vals_flat <= 0)[:, 0]

	# IPython.embed()
	ctx = mp.get_context('fork') # TODO: try spawn and fork
	pool = ctx.Pool(args.n_proc)
	n_x0 = flat_neg_inds.size
	n_x0_per_proc = math.ceil(n_x0/args.n_proc)
	permuted_inds = np.random.permutation(flat_neg_inds) # TODO: why do we need to do this?
	chunked_permuted_inds = [permuted_inds[i*n_x0_per_proc:(i+1)*n_x0_per_proc] for i in range(args.n_proc)]
	arguments = [[args, input[x]] for x in chunked_permuted_inds]

	# Launch threads
	t0 = time.perf_counter()
	result = pool.starmap(run_mpc_on_x0, arguments)
	tf = time.perf_counter()

	# print("line 246")
	# IPython.embed()
	# Don't multithread
	# result = []
	# for i in range(len(flat_neg_inds)):
	# 	r = run_mpc_on_x0(args, [input[flat_neg_inds[i]]])
	# 	result.append(r)

	exists_soln_bools = np.zeros(sze)
	for i, r in enumerate(result):
		exists_soln_bools[chunked_permuted_inds[i]] = r # TODO: does fork return RV in the same order as args?

	S_grid = np.reshape(exists_soln_bools, X.shape)
	t_per_mpc = []

	# Save data
	A_grid = (phi_0_vals <= 0).astype("int")
	percent_of_A_volume = np.sum(S_grid)*100.0/np.sum(A_grid)
	save_dict = {"A_grid": A_grid, "X": X, "Y": Y, "exists_soln_bools": exists_soln_bools, "S_grid": S_grid, "args": args, "t_per_mpc": t_per_mpc, "t_total": (tf-t0), "percent_of_A_volume": percent_of_A_volume}
	with open(save_fpth_root + ".pkl", 'wb') as handle:
		pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Plotting
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(np.logical_not(S_grid), extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
	ax.set_aspect("equal")

	plt.savefig(save_fpth_root + ".png", bbox_inches='tight')
	plt.clf()
	plt.close()

if __name__ == "__main__":
	# print("TODO: check against the default param dict in main()")
	# print(param_dict)
	# IPython.embed()
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--dt', type=float, default=0.05)
	parser.add_argument('--delta', type=float, default=0.1, help="discretization of grid over slice")
	parser.add_argument('--N_horizon', type=int, default=20)

	parser.add_argument('--which_params', default=["phi", "dphi"], nargs='+', type=str, help="which 2 state variables")

	parser.add_argument('--n_proc', default=36, type=int)

	parser.add_argument('--affix', help='the affix for the save folder', type=str)

	args = parser.parse_known_args()[0]

	mpc_compute_invariant_set(args)
	# setup_solver(args)


# if __name__ == "__main__":
# 	N_horizon = 5
# 	setup_solver(N_horizon)

"""
Query data using: 
mpc.data['success']
"""
# print("line 150")
# IPython.embed()
# print(u0.data.keys())

# TODO: fill out exists_soln_bools


# for i in range(T_max):
		# 	u0 = mpc.make_step(x0)
		# 	x0 = simulator.make_step(u0)

"""
import sys
 
file_path = 'randomfile.txt'
sys.stdout = open(file_path, "w")
print("This text will be added to the file")
"""
