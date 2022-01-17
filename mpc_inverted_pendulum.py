"""
Estimates invariant set in a brute-force way, by using MPC
"""
import do_mpc
import IPython
import numpy as np, math
import matplotlib.pyplot as plt
import torch
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

dt = 0.005 # default 0.005

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


def setup_solver(N_horizon):
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

	denom = I*(M + m) + m*(l**2)*(M + m*(math.sin(theta)**2))
	numx = (I + m*(l**2))*(m*l*dot_theta**2*math.sin(theta)) - g*(m**2)*(l**2)*math.sin(theta)*math.cos(theta) + (I + m*l**2)*u
	numtheta = m*l*(-m*l*dot_theta**2*math.sin(theta)*math.cos(theta) + (M+m)*g*math.sin(theta)) + (-m*l*math.cos(theta))*u
	ddot_x_expr = numx/denom
	ddot_theta_expr = numtheta/denom

	model.set_rhs('dot_x', ddot_x_expr)
	model.set_rhs('dot_theta', ddot_theta_expr)

	# Finally,
	model.setup()

	#######################
	# Create optimizer
	mpc = do_mpc.controller.MPC(model)
	# TODO: check out parameters and what they mean
	setup_mpc = {
		'n_horizon': N_horizon, # TODO
		't_step': dt,
		'store_full_solution': True,
		'store_solver_stats': True
	}
	mpc.set_param(**setup_mpc)

	# Define objective (minimized)
	lterm = max(0, theta**2 - theta_safety_lim**2)
	mpc.set_objective(lterm=lterm)

	# Set state and control limits
	mpc.bounds['lower', '_x', 'theta'] = x_lim[0, 0]
	mpc.bounds['upper', '_x', 'theta'] = x_lim[0, 1]
	mpc.bounds['lower', '_x', 'dot_theta'] = x_lim[1, 0]
	mpc.bounds['upper', '_x', 'dot_theta'] = x_lim[0, 1]

	mpc.bounds['lower', '_u', 'u'] = -max_force
	mpc.bounds['upper', '_u', 'u'] = max_force

	# Finally,
	mpc.setup()

	#######################
	# Create simulator
	simulator = do_mpc.simulator.Simulator(model)
	simulator.set_param(t_step = dt)
	simulator.setup()

	return model, mpc, simulator


def mpc_compute_invariant_set(N_horizon):
	delta = 0.01
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
	model, mpc, simulator = setup_solver(N_horizon)

	# simulator.reset_history()
	# simulator.x0 = x0
	for i in range(n_neg_inds):
		mpc.reset_history()
		x0 = np.zeros((4, 1))

		x0[1] = X[neg_inds[i, 0]]
		x0[3] = Y[neg_inds[i, 1]]

		# simulator.x0 = x0
		mpc.x0 = x0
		mpc.set_initial_guess()

		u0 = mpc.make_step(x0)

		print("line 150")
		IPython.embed()
		print(u0.data.keys())


		# for i in range(T_max):
				# 	u0 = mpc.make_step(x0)
				# 	x0 = simulator.make_step(u0)



if __name__ == "__main__":
	N_horizon = 20
	mpc_compute_invariant_set(N_horizon)