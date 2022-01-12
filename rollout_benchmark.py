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
from cart_pole_env import CartPoleEnv

env = CartPoleEnv()

# Note: none of this is batched
# Everything done in numpy 

exp_name = "cartpole_reduced_debugpinch3_softplus_s1"
checkpoint_number = 1450

# todo: LATER, you can load all the below from the experiment name

phi_fn, x_lim = create_phi_struct_load_xlim(exp_name, checkpoint_number)
phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
load_model(phi_fn, phi_load_fpth)

def convert_angle_to_negpi_pi_interval(angle):
	new_angle = np.arctan2(np.sin(angle), np.cos(angle))
	return new_angle

def numpy_phi_fn(x):
	# Numpy wrapper 

	theta = convert_angle_to_negpi_pi_interval(x[1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
	assert theta < math.pi and theta > -math.pi
	x_trunc = np.array([theta, x[3]])
	x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)

	phi_output = phi_fn(x_input)
	# phi_vals = phi_output[0,-1].item()
	phi_vals = phi_output.detach().cpu().numpy()
	return phi_vals

def numpy_phi_grad(x):
	# Computes grad of phi at x
	theta = convert_angle_to_negpi_pi_interval(x[1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
	assert theta < math.pi and theta > -math.pi
	x_trunc = np.array([theta, x[3]])
	x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)
	x_input.requires_grad = True

	# Compute phi grad
	phi_vals = phi_fn(x_input)
	phi_val = phi_vals[0,-1]
	phi_grad = grad([phi_val], x_input)[0]

	# Post op
	x_input.requires_grad = False
	phi_grad = phi_grad.detach().cpu().numpy()
	phi_grad = np.array([0, phi_grad[0, 0], 0, phi_grad[0, 1]])[None]

	return phi_grad

def compute_u_ref(t, x):
	return 0  

def compute_u_ours(t, x):
	############ Log
	apply_u_safe = None
	u_ref = compute_u_ref(t, x)
	phi_vals = None
	qp_slack = None
	qp_lhs = None
	qp_rhs = None
	################

	phi_vals = numpy_phi_fn(x)
	phi_grad = numpy_phi_grad(x)

	x_next = x + env.dt * env.x_dot_open_loop(x, compute_u_ref(t, x))
	next_phi_val = numpy_phi_fn(x_next)

	if phi_vals[0, -1] > 0:
		eps = 5.0 # TODO
		apply_u_safe = True
	elif phi_vals[0, -1] < 0 and next_phi_val[0, -1] >= 0: # Note: cheating way to convert DT to CT
		eps = 1.0 # TODO
		apply_u_safe = True
	else:
		apply_u_safe = False
		debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
		return u_ref, debug_dict

	# Compute the control constraints
	# Get f(x), g(x); note it's a hack for scalar u
	f_x = env.x_dot_open_loop(x, 0)
	g_x = env.x_dot_open_loop(x, 1) - f_x

	lhs = phi_grad@g_x.T
	rhs = -phi_grad@f_x.T - eps

	# Computing control using QP
	# Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
	w = 1000.0 # TODO: slack weight

	qp_lhs = lhs.item()
	qp_rhs = rhs.item()
	Q = 2*np.array([[1.0, 0], [0, 0]])
	p = np.array([[-2.0*u_ref], [w]])
	G = np.array([[lhs, -1.0], [1, 0], [-1, 0], [0, -1]])
	h = np.array([[rhs], [env.max_force], [env.max_force], [0.0]])

	sol_obj = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h))
	sol_var = sol_obj['x']

	u_safe = sol_var[0]
	qp_slack = sol_var[1]

	debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
	return u_safe, debug_dict

def x_dot_closed_loop(t, x):
	# Dynamics function 
	# Compute u
	u, _ = compute_u_ours(t, x)
	x_dot = env.x_dot_open_loop(x, u)
	return x_dot

def simulate_rollout(x0, T_max=100):
	x = x0.copy()
	# 	debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
	xs = [x]
	us = []
	dict = None

	for t in range(T_max):
		u, debug_dict = compute_u_ours(t, x)
		x_dot = env.x_dot_open_loop(x, u)
		x = x + env.dt * x_dot

		us.append(u)
		xs.append(x)

		if dict is None:
			# dict = debug_dict
			dict = {key:[value] for (key, value) in debug_dict.items()}
		else:
			dict = {key:(value.append(debug_dict[key])) for (key, value) in debug_dict.items()}

	dict = {key:np.array(value) for (key, value) in dict.items()}
	dict["x"] = np.array(xs)
	dict["u"] = np.array(us)

	print("At the end of a rollout")
	IPython.embed()
	return dict

def compute_phi_signs():
	# Discretizes state space, gets phi on these states, then returns the subset of states in invariant set
	delta = 0.01
	x = np.arange(x_lim[0, 0], x_lim[0, 1], delta)
	y = np.arange(x_lim[1, 0], x_lim[1, 1], delta)[::-1] # need to reverse it # TODO
	X, Y = np.meshgrid(x, y)

	##### Plotting ######
	input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
	input = torch.from_numpy(input)
	phi_vals = phi_fn(input)

	S_vals = torch.max(phi_vals, dim=1)[0] # S = all phi_i <= 0
	phi_signs = torch.sign(S_vals).detach().cpu().numpy()
	phi_signs = np.reshape(phi_signs, X.shape)

	return phi_signs, X, Y # square array with 1, -1. Negative indicates inside invariant set

if __name__ == "__main__":
	log_folder = "debug"
	log_fname = "our_cbf"

	makedirs(os.path.join("rollouts", log_folder))

	N_rollout = 5 # TODO
	phi_signs, X, Y = compute_phi_signs()
	where_invariant = np.argwhere(phi_signs == -1)
	which = np.random.choice(np.arange(where_invariant.shape[0]), size=N_rollout, replace=False)
	chosen_invariant_ind = where_invariant[which]
	chosen_invariant = np.concatenate((np.reshape(X[chosen_invariant_ind[:, 0]], (-1, 1)), np.reshape(Y[chosen_invariant_ind[:, 1]], (-1, 1))), axis=1)

	x0s = np.zeros((N_rollout, 4))
	x0s[:, 1] = chosen_invariant[:, 0]
	x0s[:, 3] = chosen_invariant[:, 1]

	# Plot points
	print("Check plotted points")
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(phi_signs, extent=x_lim.flatten())
	ax.set_aspect("equal")
	ax.scatter(chosen_invariant[:, 0], chosen_invariant[:, 1])
	plt.savefig("./rollouts/%s/%s_x0.png" % (log_folder, log_fname), bbox_inches='tight')
	plt.clf()
	plt.close()
	IPython.embed()

	info_dicts = None
	for i in range(N_rollout):
		info_dict = simulate_rollout(x0s[i])

		if info_dicts is None:
			info_dicts = info_dict
			# Dict comprehension is: dict_variable = {key: value for (key, value) in dictonary.items()}
			info_dicts = {key: value[None] for (key, value) in info_dicts.items()}
		else:
			info_dicts = {key: np.concatenate((value, info_dict[key][None]), axis=0) for (key, value) in info_dicts.items()}

	# Save data
	save_fpth = "./rollouts/%s/%s.pkl" % (log_folder, log_fname)
	with open(save_fpth, 'wb') as handle:
		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Sanity checks
	print("before sanity checks")
	IPython.embed()
	# 1. Check that all rollouts touched the invariant set boundary. If not, increase T_max
	# 2. Compute the number of exits for each rollout
	# 3. Compute the number of rollouts without any exits
	# debug_dict = {"x": x, "u": u, "apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs":qp_rhs, "qp_lhs":qp_lhs, "phi_vals":phi_vals}
	apply_u_safe = info_dicts["apply_u_safe"]  # (N_rollout, T_max)
	rollouts_any_safe_ctrl = np.any(apply_u_safe, axis=1)
	print(np.any(rollouts_any_safe_ctrl))
	print(rollouts_any_safe_ctrl)

	phi_vals = info_dicts["phi_vals"] # (N_rollout, T_max, r+1)
	phi_max = np.max(phi_vals, axis=2)
	rollouts_any_exits = np.any(phi_max > 0, axis=1)
	print("Any exits?", np.any(rollouts_any_exits))
	print("Percent exits: ", np.mean(rollouts_any_exits))
	print("Which rollouts have exits:", rollouts_any_exits)
	print("How many exists per rollout: ", np.sum(phi_max > 0, axis=1))

	print("after sanity checks")
	IPython.embed()