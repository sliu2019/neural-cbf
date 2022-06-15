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
from global_settings import *
from PIL import Image
"""
Pseudo-code:

phi_0
grid, find points inside phi_0 zero sublevel set 

for each point, solve MPC (can multithread this if it's super slow to get low resolution) 
If it's slow, would recommend saving progress from each point as it's received. 

solve mpc:
implement model 
implement optimizer 
yes, you can do a rollout in RHC fashion.
Unless Changliu meant to do MPC 1-step with a super long horizon (which I doubt). 

# TODO: you also need to try for 2-3 Horizon choices. What is T_max? 
"""

# Fixed seed for repeatability
torch.manual_seed(2022)
np.random.seed(2022)

g = 9.81
I = 1.2E-3
m = 0.127
M = 1.0731
l = 0.3365

theta_safety_lim = math.pi/4.0
max_force = 22.0

max_angular_velocity = 5.0 # state space constraint
x_lim = np.array([[-math.pi, math.pi], [-max_angular_velocity, max_angular_velocity]])

def phi_0(x_batch):
	# x_batch is (n, 4)
	theta = x_batch[:, 1]
	rv = theta**2 - theta_safety_lim**2
	return rv


def setup_solver(args):
	N_horizon = args.N_horizon
	dt = args.dt

	model_type = 'continuous'  # either 'discrete' or 'continuous'
	model = do_mpc.model.Model(model_type)

	# Define state vars
	x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
	theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
	dot_x = model.set_variable(var_type='_x', var_name='dot_x', shape=(1,1))
	dot_theta = model.set_variable(var_type='_x', var_name='dot_theta', shape=(1,1))

	# Define input
	u = model.set_variable(var_type='_u', var_name='u')

	# Define dynamics
	model.set_rhs('x', dot_x)
	model.set_rhs('theta', dot_theta)

	# from casadi import *
	# Note: has to use Casadi primitives, not numpy or math module
	denom = I*(M + m) + m*(l**2)*(M + m*(sin(theta)**2))
	numx = (I + m*(l**2))*(m*l*dot_theta**2*sin(theta)) - g*(m**2)*(l**2)*sin(theta)*cos(theta) + (I + m*l**2)*u
	numtheta = m*l*(-m*l*dot_theta**2*sin(theta)*cos(theta) + (M+m)*g*sin(theta)) + (-m*l*cos(theta))*u
	ddot_x_expr = numx/denom
	ddot_theta_expr = numtheta/denom

	model.set_rhs('dot_x', ddot_x_expr)
	model.set_rhs('dot_theta', ddot_theta_expr)

	# Set aux expressions
	# model = do_mpc.model.Model('continuous')
	# model.set_variable('_x', 'temperature', 4)  # 4 states
	# dt = model.x['temperature', 0] - model.x['temperature', 1]
	# model.set_expression('dtemp', dt)
	# # Query:
	# model.aux['dtemp', 0]  # 0th element of variable
	# model.aux['dtemp']  # all elements of variable
	cost = fmax(0, theta ** 2 - theta_safety_lim ** 2)  # we are in a casadi symbolic environment
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
	lterm = fmax(0, theta ** 2 - theta_safety_lim ** 2)  # we are in a casadi symbolic environment
	mpc.set_objective(lterm=lterm, mterm=lterm)

	# Set state and control limits
	# TODO: no state limits
	# mpc.bounds['lower', '_x', 'theta'] = x_lim[0, 0]
	# mpc.bounds['upper', '_x', 'theta'] = x_lim[0, 1]
	# mpc.bounds['lower', '_x', 'dot_theta'] = x_lim[1, 0]
	# mpc.bounds['upper', '_x', 'dot_theta'] = x_lim[0, 1]

	mpc.bounds['lower', '_u', 'u'] = -max_force
	mpc.bounds['upper', '_u', 'u'] = max_force

	# Finally,
	mpc.setup()

	#######################
	# Create simulator
	# simulator = do_mpc.simulator.Simulator(model)
	# simulator.set_param(t_step = dt)
	# simulator.setup()

	return model, mpc

def mpc_compute_invariant_set(args):
	N_horizon = args.N_horizon
	delta = args.delta
	dt = args.dt

	x = np.arange(x_lim[0, 0], x_lim[0, 1], delta)
	y = np.arange(x_lim[1, 0], x_lim[1, 1], delta)[::-1]  # need to reverse it
	X, Y = np.meshgrid(x, y)

	sze = X.size
	input = np.concatenate((np.zeros((sze, 1)), X.flatten()[:, None], np.zeros((sze, 1)), Y.flatten()[:, None]), axis=1)
	phi_0_vals = phi_0(input)
	phi_0_vals = np.reshape(phi_0_vals, X.shape)

	neg_inds = np.argwhere(phi_0_vals <= 0) # (m, 2)
	n_neg_inds = neg_inds.shape[0]

	exists_soln_bools = np.zeros(n_neg_inds)
	model, mpc = setup_solver(args)

	print("./rollout_results/mpc_delta_%f_dt_%f_horizon_%i.png" % (delta, dt, N_horizon))
	# IPython.embed()
	for i in range(n_neg_inds):
		mpc.reset_history()
		x0 = np.zeros((4, 1))

		x0[1] = X[neg_inds[i,0], neg_inds[i,1]]
		x0[3] = Y[neg_inds[i,0], neg_inds[i,1]]

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
		# IPython.embed()
	save_fpth_root = "./rollout_results/mpc_delta_%f_dt_%f_horizon_%i" % (delta, dt, N_horizon)

	# Save data
	save_dict = {"neg_inds": neg_inds, "X": X, "Y": Y, "exists_soln_bools":exists_soln_bools}
	with open(save_fpth_root + ".pkl", 'wb') as handle:
		pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Plotting
	signs = np.zeros_like(X)
	signs[neg_inds[:, 0], neg_inds[:, 1]] = exists_soln_bools
	signs = np.logical_not(signs) # get colors to match NN plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(signs, extent=x_lim.flatten())
	ax.set_aspect("equal")

	plt.savefig(save_fpth_root + ".png", bbox_inches='tight')
	plt.clf()
	plt.close()

	return exists_soln_bools

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--dt', type=float, default=0.05)
	parser.add_argument('--delta', type=float, default=0.1)
	parser.add_argument('--N_horizon', type=int, default=20)
	args = parser.parse_known_args()[0]

	mpc_compute_invariant_set(args)


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
