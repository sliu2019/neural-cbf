import scipy as sp
import numpy as np 
import IPython 
import torch 
import math 
from plot_utils import create_phi_struct_load_xlim
from torch.autograd import grad
from src.utils import *
from scipy.integrate import solve_ivp
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import pickle

from rollout_cbf_classes.our_cbf_class import OurCBF
# TODO: Weiye, Tianhao: import here

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

def compute_u_ref(t, x):
	return 0  

class CBFController:
	def __init__(self, cbf_obj):
		# super().__init__()
		# variables = locals()  # dict of local names
		# self.__dict__.update(variables)  # __dict__ holds and object's attributes
		# del self.__dict__["self"]  # don't need `self`
		self.cbf_obj = cbf_obj

	def compute_control(self, t, x):
		############ Init log vars
		apply_u_safe = None
		u_ref = compute_u_ref(t, x)
		phi_vals = None
		qp_slack = None
		qp_lhs = None
		qp_rhs = None
		################

		# phi_vals = numpy_phi_fn(x) # This is an array of (1, r+1), where r is the degree
		# phi_grad = numpy_phi_grad(x)
		
		phi_vals = self.cbf_obj.phi_fn(x) # This is an array of (1, r+1), where r is the degree
		phi_grad = self.cbf_obj.phi_grad(x)

		x_next = x + dt*x_dot_open_loop(x, compute_u_ref(t, x)) # in the absence of safe control, the next state
		next_phi_val = self.cbf_obj.phi_fn(x_next)

		if phi_vals[0, -1] > 0: # Outside
			eps = 5.0 # TODO
			apply_u_safe = True
		elif phi_vals[0, -1] < 0 and next_phi_val[-1] >= 0: # On boundary. Note: cheating way to convert DT to CT
			eps = 1.0 # TODO
			apply_u_safe = True
		else: # Inside
			apply_u_safe = False
			debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
			return u_ref, debug_dict

		# Compute the control constraints
		# Get f(x), g(x); note it's a hack for scalar u
		f_x = x_dot_open_loop(x, 0)
		g_x = x_dot_open_loop(x, 1) - f_x

		lhs = phi_grad@g_x.T
		rhs = -phi_grad@f_x.T - eps

		# Computing control using QP
		# Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
		w = 1000.0 # slack weight

		qp_lhs = lhs.item()
		qp_rhs = rhs.item()
		Q = 2*np.array([[1.0, 0], [0, 0]])
		p = np.array([[-2.0*u_ref], [w]])
		G = np.array([[qp_lhs, -1.0], [1, 0], [-1, 0], [0, -1]])
		h = np.array([[qp_rhs], [max_force], [max_force], [0.0]])

		# try:
		sol_obj = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h))
		# except:
		# 	IPython.embed()
		sol_var = sol_obj['x']

		u_safe = sol_var[0]
		qp_slack = sol_var[1]

		debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "phi_vals":phi_vals.flatten(), "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs}
		return u_safe, debug_dict

	def sample_invariant_set(self, N_samp):
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
		phi_vals_on_grid = self.cbf_obj.phi_fn(input) # N_samp x r+1

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

def x_dot_open_loop(x, u):
	# u is scalar
	x_dot = np.zeros(4)

	x_dot[0] = x[2]
	x_dot[1] = x[3]

	theta = x[1]
	theta_dot = x[3]
	denom = I*(M + m) + m*(l**2)*(M + m*(math.sin(theta)**2))
	x_dot[2] = (I + m*(l**2))*(m*l*theta_dot**2*math.sin(theta)) - g*(m**2)*(l**2)*math.sin(theta)*math.cos(theta) + (I + m*l**2)*u
	x_dot[3] = m*l*(-m*l*theta_dot**2*math.sin(theta)*math.cos(theta) + (M+m)*g*math.sin(theta)) + (-m*l*math.cos(theta))*u

	x_dot[2] = x_dot[2]/denom
	x_dot[3] = x_dot[3]/denom

	return x_dot 

def simulate_rollout(x0, N_dt, cbf_controller):
	# print("Inside simulate_rollout")
	# IPython.embed()

	x = x0.copy()
	# 	debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
	xs = [x]
	us = []
	dict = None

	for t in range(N_dt):
		u, debug_dict = cbf_controller.compute_control(t, x) # Define this
		x_dot = x_dot_open_loop(x, u)
		x = x + dt*x_dot

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

if __name__ == "__main__":
	log_folder = "debug2"
	which_cbf = "our_cbf_football" 

	makedirs(os.path.join("rollout_results", log_folder))

	# TODO: fill out run arguments
	N_rollout = 5 
	T_max = 1.0 # in seconds
	N_dt = int(T_max/dt)

	if which_cbf == "our_cbf_football":
		exp_name = "cartpole_reduced_debugpinch3_softplus_s1"
		checkpoint_number = 1450 
		cbf_obj = OurCBF(exp_name, checkpoint_number)
	elif which_cbf == "our_cbf_baguette":
		exp_name = "cartpole_reduced_debugpinch1_softplus_s1" 
		checkpoint_number = 1450 
		cbf_obj = OurCBF(exp_name, checkpoint_number)
	# TODO: weiye + tianhao: import and fill in 

	### Test
	# x = np.random.normal(size=(4))
	# x_batch = np.random.normal(size=(2, 4))
	# cbf_obj.phi_fn(x_batch)
	# cbf_obj.phi_grad(x)
	######
	cbf_controller = CBFController(cbf_obj)

	x0s, phi_vals_on_grid, X, Y = cbf_controller.sample_invariant_set(N_rollout)

	#####################################
	# Plot x0 samples and invariant set
	#####################################

	# print("Check x0 plotting")
	# IPython.embed()

	fig = plt.figure()
	ax = fig.add_subplot(111)

	max_phi_vals_on_grid = phi_vals_on_grid.max(axis=1)
	phi_signs = np.reshape(np.sign(max_phi_vals_on_grid), X.shape)
	ax.imshow(phi_signs, extent=x_lim.flatten())
	ax.set_aspect("equal")
	ax.scatter(x0s[:, 1], x0s[:, 3])
	plt.savefig("./rollout_results/%s/%s_x0s.png" % (log_folder, which_cbf), bbox_inches='tight')
	plt.clf()
	plt.close()

	#####################################
	# Run multiple rollout_results
	#####################################

	info_dicts = None
	for i in range(N_rollout):
		info_dict = simulate_rollout(x0s[i], N_dt, cbf_controller)

		if info_dicts is None:
			info_dicts = info_dict
			# Dict comprehension is: dict_variable = {key: value for (key, value) in dictonary.items()}
			info_dicts = {key: value[None] for (key, value) in info_dicts.items()}
		else:
			info_dicts = {key: np.concatenate((value, info_dict[key][None]), axis=0) for (key, value) in info_dicts.items()}

	# Save data
	save_fpth = "./rollout_results/%s/%s.pkl" % (log_folder, which_cbf)
	with open(save_fpth, 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#####################################
	# Sanity checks
	#####################################

	# print("before sanity checks")
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
	phi_max = np.max(phi_vals, axis=2)
	rollouts_any_exits = np.any(phi_max > 0, axis=1)
	any_exits = np.any(rollouts_any_exits)
	print("Any exits?", any_exits)
	if any_exits:
		print("Percent exits: ", np.mean(rollouts_any_exits))
		print("Which rollout_results have exits:", rollouts_any_exits)
		print("How many exits per rollout: ", np.sum(phi_max > 0, axis=1))

	phi_star = phi_vals[:, :, -1]
	rollouts_any_phistar_pos = np.any(phi_star>0, axis=1)
	any_phistar_pos = np.any(rollouts_any_phistar_pos)

	print("Any phi_star positive?", any_phistar_pos)
	if any_phistar_pos:
		print("Which rollout_results had phi_star positive:", rollouts_any_phistar_pos)

	# print("after sanity checks")
	# # phi_vals (20, 100, 1, 3)
	#
	# for key, value in info_dicts.items():
	# 	print(key, value.shape)

	#####################################
	# Plot trajectories
	#####################################
	# print("Check plotted trajectories")
	# IPython.embed()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(phi_signs, extent=x_lim.flatten())
	ax.set_aspect("equal")
	ax.scatter(x0s[:, 1], x0s[:, 3])

	phi_star_on_grid = phi_vals_on_grid[:, -1]
	plt.contour(X, Y, np.reshape(phi_star_on_grid, X.shape), levels=[0.0],
	                 colors=('k',), linewidths=(2,))

	x = info_dicts["x"]
	for i in range(N_rollout):
		x_rl = x[i]
		plt.plot(x_rl[:, 1], x_rl[:, 3])
	plt.savefig("./rollout_results/%s/%s_trajectories.png" % (log_folder, which_cbf), bbox_inches='tight')
	plt.clf()
	plt.close()

	IPython.embed()