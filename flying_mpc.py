"""
Estimates invariant set in a brute-force way, by using MPC
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
	# IPython.embed()
	N_horizon = args.N_horizon
	dt = args.dt

	model_type = 'continuous'  # either 'discrete' or 'continuous'
	model = do_mpc.model.Model(model_type)

	# Define state vars
	gamma = model.set_variable(var_type='_x', var_name='gamma', shape=(1, 1))
	dgamma = model.set_variable(var_type='_x', var_name='dgamma', shape=(1, 1))
	beta = model.set_variable(var_type='_x', var_name='beta', shape=(1, 1))
	dbeta = model.set_variable(var_type='_x', var_name='dbeta', shape=(1, 1))
	alpha = model.set_variable(var_type='_x', var_name='alpha', shape=(1, 1))
	dalpha = model.set_variable(var_type='_x', var_name='dalpha', shape=(1, 1))

	phi = model.set_variable(var_type='_x', var_name='phi', shape=(1,1))
	dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(1,1))
	theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
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
	# IPython.embed()
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

	# model.set_rhs('rpw', drpw)
	# model.set_rhs('drpw', ddrpw)

	model.set_rhs('theta', dtheta)
	model.set_rhs('phi', dphi)
	model.set_rhs('gamma', dgamma)
	model.set_rhs('beta', dbeta)
	model.set_rhs('alpha', dalpha)

	model.set_rhs('dtheta', ddtheta)
	model.set_rhs('dphi', ddphi)
	model.set_rhs('dgamma', ddgamma)
	model.set_rhs('dbeta', ddbeta)
	model.set_rhs('dalpha', ddalpha)

	# Set aux expressions
	cos_cos = cos(theta) * cos(phi)
	eps = 1e-4  # prevents nan when cos_cos = +/- 1 (at x = 0)
	signed_eps = -sign(cos_cos) * eps
	delta = acos(cos_cos + signed_eps)
	h = delta ** 2 + gamma ** 2 + beta ** 2 - delta_safety_limit ** 2
	cost = fmax(0, h)  # we are in a casadi symbolic environment
	model.set_expression('cost', cost)

	# Finally,
	model.setup()

	#######################
	# Create optimizer
	mpc = do_mpc.controller.MPC(model)
	setup_mpc = {
		'n_horizon': N_horizon,
		't_step': dt,
		'store_full_solution': True
	}

	mpc.set_param(**setup_mpc)

	# Define objective (minimized)
	cos_cos = cos(theta) * cos(phi)
	eps = 1e-4  # prevents nan when cos_cos = +/- 1 (at x = 0)
	signed_eps = -sign(cos_cos) * eps
	delta = acos(cos_cos + signed_eps)
	h = delta ** 2 + gamma ** 2 + beta ** 2 - delta_safety_limit ** 2
	lterm = fmax(0, h)  # we are in a casadi symbolic environment
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

def mpc_compute_invariant_set(args):
	N_horizon = args.N_horizon
	delta = args.delta
	dt = args.dt
	which_params = args.which_params
	x_lim = param_dict["x_lim"]
	# state_index_dict = param_dict["state_index_dict"]
	ind1 = state_index_dict[which_params[0]]
	ind2 = state_index_dict[which_params[1]]

	# Note: assuming that we're interested in 2 variables and the other vars = 0
	x = np.arange(x_lim[ind1, 0], x_lim[ind1, 1], delta)
	y = np.arange(x_lim[ind2, 0], x_lim[ind2, 1], delta)[::-1]  # need to reverse it
	X, Y = np.meshgrid(x, y)

	sze = X.size
	input = np.zeros((sze, 10))
	input[:, ind1] = X.flatten()
	input[:, ind2] = Y.flatten()
	# input = np.concatenate((np.zeros((sze, 1)), X.flatten()[:, None], np.zeros((sze, 1)), Y.flatten()[:, None]), axis=1)
	phi_0_vals = phi_0(input)
	phi_0_vals = np.reshape(phi_0_vals, X.shape)

	neg_inds = np.argwhere(phi_0_vals <= 0) # (m, 2)
	n_neg_inds = neg_inds.shape[0]

	exists_soln_bools = np.zeros(n_neg_inds)
	model, mpc = setup_solver(args)

	# print("./rollout_results/mpc_delta_%f_dt_%f_horizon_%i.png" % (delta, dt, N_horizon)) # TODO: replace this
	save_fpth_root = "./flying_mpc_outputs/mpc_%s_%s" % (which_params[0], which_params[1])
	# print(save_fpth_root)
	# print("ln 220")
	# IPython.embed()
	for i in range(n_neg_inds):
		# print("inside loop")
		# IPython.embed()
		t0 = time.perf_counter()
		mpc.reset_history()
		x0 = np.zeros((10, 1))

		x0[ind1] = X[neg_inds[i,0], neg_inds[i,1]]
		x0[ind2] = Y[neg_inds[i,0], neg_inds[i,1]]

		mpc.x0 = x0
		mpc.set_initial_guess()

		u0 = mpc.make_step(x0)

		# exists_soln_bools[i] = mpc.data['success'].item()

		# Has shape (2*N_horizon) because records default (0) aux first.
		pred_cost = mpc.data['_opt_aux_num']
		pred_cost = np.reshape(pred_cost, (-1, 2))[:, 1]
		if np.any(pred_cost != 0):
			# print("YAY")
			# IPython.embed()
			exists_soln_bools[i] = 0
		else:
			exists_soln_bools[i] = 1

		# print(mpc.data['_opt_aux_num'].shape, mpc.data['_opt_aux_num'])
		"""Please choose from dict_keys(['_time', '_x', '_y', '_u', '_z', '_tvp', '_p', '_aux', '_eps', 'opt_p_num', '_opt_x_num', '_opt_aux_num', '_lam_g_num', 'success', 't_wall_total'])"""
		t1 = time.perf_counter()
		print("t for mpc on a single x0: %.3f s" % (t1 -t0))

	# Save data
	# print("line 251")
	# IPython.embed()
	S_grid = np.zeros_like(X)
	S_grid[neg_inds[:, 0], neg_inds[:, 1]] = exists_soln_bools
	A_grid = (phi_0_vals <= 0).astype("int")
	save_dict = {"A_grid": A_grid, "X": X, "Y": Y, "exists_soln_bools": exists_soln_bools, "S_grid": S_grid, "args": args}
	with open(save_fpth_root + ".pkl", 'wb') as handle:
		pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Plotting
	signs = np.zeros_like(X)
	signs[neg_inds[:, 0], neg_inds[:, 1]] = exists_soln_bools
	signs = np.logical_not(signs) # get colors to match NN plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(signs, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
	ax.set_aspect("equal")

	plt.savefig(save_fpth_root + ".png", bbox_inches='tight')
	plt.clf()
	plt.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--dt', type=float, default=0.05)
	parser.add_argument('--delta', type=float, default=0.1, help="discretization of grid over slice")
	parser.add_argument('--N_horizon', type=int, default=20)

	parser.add_argument('--which_params', default=["phi", "dphi"], nargs='+', type=str, help="which 2 state variables")

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
