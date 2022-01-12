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

from rollout_cbf_classes import our_cbf_class
# Note: none of this is batched
# Everything done in numpy 

# exp_name = "cartpole_reduced_debugpinch3_softplus_s1" # "football"
# exp_name = "cartpole_reduced_debugpinch1_softplus_s1" # "baguette"
# checkpoint_number = 1450

# todo: LATER, you can load all the below from the experiment name
# dt = 0.005 # TODO
# dt = 1e-8
dt = 1e-5

g = 9.81
I = 1.2E-3
m = 0.127
M = 1.0731
l = 0.3365 

max_theta = math.pi/4.0 
max_force = 22.0
max_angular_velocity = 5.0 # state space constraint

x_lim = [[-math.pi, math.pi], [-5, 5]]

# phi_fn, x_lim = create_phi_struct_load_xlim(exp_name, checkpoint_number)
# phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
# load_model(phi_fn, phi_load_fpth)

# def convert_angle_to_negpi_pi_interval(angle):
# 	new_angle = np.arctan2(np.sin(angle), np.cos(angle))
# 	return new_angle

# def numpy_phi_fn(x):
# 	# Numpy wrapper 

# 	theta = convert_angle_to_negpi_pi_interval(x[1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
# 	assert theta < math.pi and theta > -math.pi
# 	x_trunc = np.array([theta, x[3]])
# 	x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)

# 	phi_output = phi_fn(x_input)
# 	# phi_vals = phi_output[0,-1].item()
# 	phi_vals = phi_output.detach().cpu().numpy().flatten()
# 	return phi_vals

# def numpy_phi_grad(x):
# 	# Computes grad of phi at x
# 	theta = convert_angle_to_negpi_pi_interval(x[1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
# 	assert theta < math.pi and theta > -math.pi
# 	x_trunc = np.array([theta, x[3]])
# 	x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)
# 	x_input.requires_grad = True

# 	# Compute phi grad
# 	phi_vals = phi_fn(x_input)
# 	phi_val = phi_vals[0,-1]
# 	phi_grad = grad([phi_val], x_input)[0]

# 	# Post op
# 	x_input.requires_grad = False
# 	phi_grad = phi_grad.detach().cpu().numpy()
# 	phi_grad = np.array([0, phi_grad[0, 0], 0, phi_grad[0, 1]])[None]

# 	return phi_grad

def compute_u_ref(t, x):
	return 0  

class CBFController:
	def __init__(self, cbf_obj):
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def compute_control(t, x):
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

		if phi_vals[-1] > 0: # Outside
			eps = 5.0 # TODO
			apply_u_safe = True
		elif phi_vals[-1] < 0 and next_phi_val[-1] >= 0: # On boundary. Note: cheating way to convert DT to CT
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
		w = 1000.0 # TODO: slack weight

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
		
		# Discretizes state space, then returns the subset of states in invariant set
		delta = 0.01
		x = np.arange(x_lim[0, 0], x_lim[0, 1], delta)
		y = np.arange(x_lim[1, 0], x_lim[1, 1], delta)[::-1] # need to reverse it # TODO
		X, Y = np.meshgrid(x, y)

		##### Plotting ######
		# input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
		# input = torch.from_numpy(input) # unnecessary 
		size = X.size
		input = np.concatenate((np.zeros((size, 1)), X.flatten()[:, None], np.zeros((size, 1)), Y.flatten()[:, None]), axis=1)
		phi_vals_on_grid = self.cbf_obj.phi_fn(input)

		# phi_vals = phi_vals.detach().cpu().numpy()
		# return phi_vals, X, Y # square array with 1, -1. Negative indicates inside invariant set

		# phi_vals_on_grid, X, Y = compute_phi_signs()
		# S_vals = torch.max(phi_vals, dim=1)[0] # S = all phi_i <= 0
		# phi_signs = torch.sign(S_vals).detach().cpu().numpy()
		# phi_signs = np.reshape(phi_signs, X.shape)

		# S_vals = phi_vals_on_grid.max(axis=1) # S = all phi_i <= 0
		# phi_signs = np.sign(S_vals)
		# phi_signs = np.reshape(phi_signs, X.shape) # binary array denoting where invariant set

		max_phi_vals_on_grid = phi_vals_on_grid.max(axis=1) # Assuming S = all phi_i <= 0
		where_invariant = np.argwhere(max_phi_vals_on_grid <= 0)

		sample_ind = np.random.choice(np.arange(where_invariant.shape[0]), size=N_rollout, replace=False)
		global_ind = where_invariant[sample_ind]
		sample_X = X[global_ind[:, 0], global_ind[:, 1]]
		sample_Y = Y[global_ind[:, 0], global_ind[:, 1]]

		# chosen_invariant = np.concatenate((sample_X[:, None], sample_Y[:, None]), axis=1)

		x0s = np.zeros((N_rollout, 4))
		x0s[:, 1] = sample_X
		x0s[:, 3] = sample_Y

		return x0s, phi_vals_on_grid, X, Y

# def compute_u_ours(t, x):
# 	############ Log
# 	apply_u_safe = None
# 	u_ref = compute_u_ref(t, x)
# 	phi_vals = None
# 	qp_slack = None
# 	qp_lhs = None
# 	qp_rhs = None
# 	################

# 	phi_vals = numpy_phi_fn(x) # This is an array of (1, r+1), where r is the degree
# 	phi_grad = numpy_phi_grad(x)

# 	x_next = x + dt*x_dot_open_loop(x, compute_u_ref(t, x)) # in the absence of safe control, the next state
# 	next_phi_val = numpy_phi_fn(x_next)

# 	if phi_vals[-1] > 0: # Outside
# 		eps = 5.0 # TODO
# 		apply_u_safe = True
# 	elif phi_vals[-1] < 0 and next_phi_val[-1] >= 0: # On boundary. Note: cheating way to convert DT to CT
# 		eps = 1.0 # TODO
# 		apply_u_safe = True
# 	else: # Inside
# 		apply_u_safe = False
# 		debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
# 		return u_ref, debug_dict

# 	# Compute the control constraints
# 	# Get f(x), g(x); note it's a hack for scalar u
# 	f_x = x_dot_open_loop(x, 0)
# 	g_x = x_dot_open_loop(x, 1) - f_x

# 	lhs = phi_grad@g_x.T
# 	rhs = -phi_grad@f_x.T - eps

# 	# Computing control using QP
# 	# Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
# 	w = 1000.0 # TODO: slack weight

# 	qp_lhs = lhs.item()
# 	qp_rhs = rhs.item()
# 	Q = 2*np.array([[1.0, 0], [0, 0]])
# 	p = np.array([[-2.0*u_ref], [w]])
# 	G = np.array([[qp_lhs, -1.0], [1, 0], [-1, 0], [0, -1]])
# 	h = np.array([[qp_rhs], [max_force], [max_force], [0.0]])

# 	# try:
# 	sol_obj = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h))
# 	# except:
# 	# 	IPython.embed()
# 	sol_var = sol_obj['x']

# 	u_safe = sol_var[0]
# 	qp_slack = sol_var[1]

# 	debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals.flatten()}
# 	return u_safe, debug_dict

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

# def x_dot_closed_loop(t, x):
# 	# Dynamics function 
# 	# Compute u
# 	u, _ = compute_u_ours(t, x)
# 	x_dot = x_dot_open_loop(x, u)
# 	return x_dot

def simulate_rollout(x0, N_dt, cbf_obj):
	# print("Inside simulate_rollout")
	# IPython.embed()

	x = x0.copy()
	# 	debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
	xs = [x]
	us = []
	dict = None

	for t in range(N_dt):
		u, debug_dict = cbf_obj.compute_control(t, x) # Define this
		x_dot = x_dot_open_loop(x, u)
		x = x + dt*x_dot
  
		us.append(u)
		xs.append(x)

		if dict is None:
			# dict = debug_dict
			dict = {key:[value] for (key, value) in debug_dict.items()}
		else:
			# IPython.embed()
			for key, value in dict.items():
				# if t == 1:
				# 	IPython.embed()
				# print(key, value)
				value.append(debug_dict[key])
				# dict[key] = value
				# print(new_value, type(new_value))
			# dict = {key:(value.append(debug_dict[key])) for (key, value) in dict.items()}
		# print(dict)

	dict = {key:np.array(value) for (key, value) in dict.items()}
	dict["x"] = np.array(xs)
	dict["u"] = np.array(us)

	# print("At the end of a rollout")
	# IPython.embed()
	return dict

# def compute_phi_signs():
# 	# Discretizes state space, gets phi on these states, then returns the subset of states in invariant set
# 	delta = 0.01
# 	x = np.arange(x_lim[0, 0], x_lim[0, 1], delta)
# 	y = np.arange(x_lim[1, 0], x_lim[1, 1], delta)[::-1] # need to reverse it # TODO
# 	X, Y = np.meshgrid(x, y)

# 	##### Plotting ######
# 	input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
# 	input = torch.from_numpy(input)
# 	phi_vals = phi_fn(input)

# 	phi_vals = phi_vals.detach().cpu().numpy()
# 	return phi_vals, X, Y # square array with 1, -1. Negative indicates inside invariant set

if __name__ == "__main__":
	log_folder = "debug"
	which_cbf = "our_cbf_baguette" 

	makedirs(os.path.join("rollouts", log_folder))

	# IPython.embed()
	N_rollout = 5 # TODO
	T_max = 1.0 # seconds def: 100
	N_dt = int(T_max/dt)


	if which_cbf == "our_cbf_football":
		exp_name = "cartpole_reduced_debugpinch3_softplus_s1"
		checkpoint_number = 1450 
		cbf_obj = our_cbf_class(exp_name, checkpoint_number)
	elif which_cbf == "our_cbf_baguette":
		exp_name = "cartpole_reduced_debugpinch1_softplus_s1" 
		checkpoint_number = 1450 
		cbf_obj = our_cbf_class(exp_name, checkpoint_number)
	# TODO: weiye + tianhao: import and fill in 

	cbf_controller = CBFController(cbf_obj)

	######################################
	"""phi_vals_on_grid, X, Y = compute_phi_signs()
	# S_vals = torch.max(phi_vals, dim=1)[0] # S = all phi_i <= 0
	# phi_signs = torch.sign(S_vals).detach().cpu().numpy()
	# phi_signs = np.reshape(phi_signs, X.shape)
	S_vals = phi_vals_on_grid.max(axis=1) # S = all phi_i <= 0
	phi_signs = np.sign(S_vals)
	phi_signs = np.reshape(phi_signs, X.shape)

	where_invariant = np.argwhere(phi_signs == -1)
	which = np.random.choice(np.arange(where_invariant.shape[0]), size=N_rollout, replace=False)
	chosen_invariant_ind = where_invariant[which]
	chosen_X = X[chosen_invariant_ind[:, 0], chosen_invariant_ind[:, 1]]
	chosen_Y = Y[chosen_invariant_ind[:, 0], chosen_invariant_ind[:, 1]]

	chosen_invariant = np.concatenate((chosen_X[:, None], chosen_Y[:, None]), axis=1)

	x0s = np.zeros((N_rollout, 4))
	x0s[:, 1] = chosen_invariant[:, 0]
	x0s[:, 3] = chosen_invariant[:, 1]"""

	x0s, phi_vals_on_grid, X, Y = cbf_controller.sample_invariant_set(N_rollout)

	#####################################
	# Plot x0 samples and invariant set
	#####################################

	print("Check x0 plotting")
	IPython.embed()

	fig = plt.figure()
	ax = fig.add_subplot(111)

	max_phi_vals_on_grid = phi_vals_on_grid.max(axis=1)
	phi_signs = np.reshape(np.sign(max_phi_vals_on_grid), X.shape)
	ax.imshow(phi_signs, extent=x_lim.flatten())
	ax.set_aspect("equal")
	ax.scatter(x0s[:, 1], x0s[:, 3])
	plt.savefig("./rollouts/%s/%s_x0s.png" % (log_folder, which_cbf), bbox_inches='tight')
	plt.clf()
	plt.close()

	#####################################
	# Run multiple rollouts 
	#####################################

	info_dicts = None
	for i in range(N_rollout):
		info_dict = simulate_rollout(x0s[i], N_dt, cbf_obj)

		if info_dicts is None:
			info_dicts = info_dict
			# Dict comprehension is: dict_variable = {key: value for (key, value) in dictonary.items()}
			info_dicts = {key: value[None] for (key, value) in info_dicts.items()}
		else:
			info_dicts = {key: np.concatenate((value, info_dict[key][None]), axis=0) for (key, value) in info_dicts.items()}

	# Save data
	save_fpth = "./rollouts/%s/%s.pkl" % (log_folder, which_cbf)
	with open(save_fpth, 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#####################################
	# Sanity checks
	#####################################

	print("before sanity checks")
	IPython.embed()
	# 1. Check that all rollouts touched the invariant set boundary. If not, increase T_max
	# 2. Compute the number of exits for each rollout
	# 3. Compute the number of rollouts without any exits
	# debug_dict = {"x": x, "u": u, "apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
	print("*********************************************************\n")
	apply_u_safe = info_dicts["apply_u_safe"]  # (N_rollout, T_max)
	rollouts_any_safe_ctrl = np.any(apply_u_safe, axis=1)
	print("Did we apply safe control?" , np.all(rollouts_any_safe_ctrl))
	if np.all(rollouts_any_safe_ctrl) == False:
		print("Which rollouts did we apply safe control?", rollouts_any_safe_ctrl)
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
		print("Which rollouts have exits:", rollouts_any_exits)
		print("How many exits per rollout: ", np.sum(phi_max > 0, axis=1))

	phi_star = phi_vals[:, :, -1]
	rollouts_any_phistar_pos = np.any(phi_star>0, axis=1)
	any_phistar_pos = np.any(rollouts_any_phistar_pos)

	print("Any phi_star positive?", any_phistar_pos)
	if any_phistar_pos:
		print("Which rollouts had phi_star positive:", rollouts_any_phistar_pos)

	# print("after sanity checks")
	# # phi_vals (20, 100, 1, 3)
	#
	# for key, value in info_dicts.items():
	# 	print(key, value.shape)

	#####################################
	# Plot trajectories
	#####################################
	print("Check plotted trajectories")
	IPython.embed()
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(phi_signs, extent=x_lim.flatten())
	ax.set_aspect("equal")
	ax.scatter(x0s[:, 1], chosen_invariant[:, 3])

	phi_star_on_grid = phi_vals_on_grid[:, -1]
	plt.contour(X, Y, np.reshape(phi_star_on_grid, X.shape), levels=[0.0],
	                 colors=('k',), linewidths=(2,))

	x = info_dicts["x"]
	for i in range(N_rollout):
		x_rl = x[i]
		plt.plot(x_rl[:, 1], x_rl[:, 3])
	plt.savefig("./rollouts/%s/%s_trajectories.png" % (log_folder, which_cbf), bbox_inches='tight')
	plt.clf()
	plt.close()

	IPython.embed()