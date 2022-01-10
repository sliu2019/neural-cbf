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
# Note: none of this is batched
# Everything done in numpy 

exp_name = "cartpole_reduced_debugpinch3_softplus_s1"
checkpoint_number = 1450

# todo: LATER, you can load all the below from the experiment name
eps_value = 10 # TODO? mag? check Changliu paper or Weiye, Tianhao
alpha = 0.1 # boundary buffer size
# This experiment has min phi value of -0.6
dt = 0.005

g = 9.81
I = 1.2E-3
m = 0.127
M = 1.0731
l = 0.3365 

max_theta = math.pi/4.0 
max_force = 22.0
max_angular_velocity = 5.0 # state space constraint

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
	# x_input.requires_grad = True

	# Compute phi grad
	phi_output = phi_fn(x_input)
	phi_val = phi_output[0,-1].item()
	# phi_grad = grad([phi_val], x_input)[0]

	# IPython.embed()
	return phi_val

def numpy_phi_grad(x):
	# Computes grad of phi at x 

	theta = convert_angle_to_negpi_pi_interval(x[1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
	assert theta < math.pi and theta > -math.pi
	x_trunc = np.array([theta, x[3]])
	x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)
	x_input.requires_grad = True

	# Compute phi grad
	phi_output = phi_fn(x_input)
	phi_val = phi_output[0,-1]
	phi_grad = grad([phi_val], x_input)[0]

	# Post op
	x_input.requires_grad = False
	phi_grad = phi_grad.detach().cpu().numpy()
	phi_grad = np.array([0, phi_grad[0, 0], 0, phi_grad[0, 1]])[None]

	return phi_grad

def u_ref(t, x):
	return 0  

def u_ours(t, x):
	phi_val = numpy_phi_fn(x)
	phi_grad = numpy_phi_grad(x)

	x_next = x + dt*x_dot_open_loop(x, u_ref(t, x))
	next_phi_val = numpy_phi_fn(x_next)

	debug_where_are_we = None
	if phi_val > 0:
		eps = eps_value
		debug_where_are_we = "Outside"
	# elif phi_val <= 0 and phi_val >= -alpha: # TODO: set this. Simin: doesn't work, a uniform border will never work well.
	elif phi_val < 0 and next_phi_val >= 0: # TODO: cheating way to convert DT to CT
		# eps = 0
		eps = 0
		debug_where_are_we = "On"
	else:
		# return np.zeros(1)
		debug_where_are_we = "Inside"
		print(debug_where_are_we)
		return u_ref(t, x)

	# Compute the control constraints
	# Get f(x), g(x)
	f_x = x_dot_open_loop(x, 0) # xdot defined globally
	g_x = x_dot_open_loop(x, 1) - f_x

	lhs = phi_grad@g_x.T
	rhs = -phi_grad@f_x.T - eps

	"""# Done computing CBF constraint
	u_safe = u_ref(t, x)

	# Cbf constraint
	ratio = (rhs/lhs).item()
	if lhs >= 0:
		u_safe = np.clip(u_safe, None, ratio)
	else:
		u_safe = np.clip(u_safe, ratio, None)

	# print()
	# IPython.embed()
	if debug_where_are_we == "On":
		print(ratio)
		IPython.embed()
	print(debug_where_are_we)
	print("u: %f" % u_safe)
	return u_safe"""

	# Computing control using QP
	# Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
	u_ref_input = u_ref(t, x)

	Q = np.array([[1, 0], [0, 0]])
	p = np.array([-2*u_ref_input, 1])
	G = np.array([[lhs, -1], [1, 0], [-1, 0], [0, -1]])
	h = np.array([rhs, max_force, -max_force, 0])

	sol_obj = solvers.qp(Q, p, G, h)
	sol_var = sol_obj['x']

	u_safe = sol_var[0]
	eps_slack = sol_var[1]
	debug_dict = {'eps_slack': eps_slack} # second RV

	return u_safe, debug_dict

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

def x_dot_closed_loop(t, x):
	# Dynamics function 
	# Compute u
	u = u_ours(t, x)
	# print("u: ", u)
	# Compute x_dot 
	x_dot = x_dot_open_loop(x, u)
	return x_dot


if __name__ == "__main__":
	theta_dot_init = 0 # TODO: try others later (can only be in the assumed range)
	theta_init = math.pi/8 # less than pi/4  
	x0 = np.array([0, theta_init, 0, theta_dot_init]) #x, theta, xdot, theta_dot
	t_span = [0, 20]

	"""
	sol = solve_ivp(x_dot_closed_loop, t_span, x0)
	assert sol.status==0
	# print("Solution status: ", sol.status)

	x_rollout = sol.y

	# rv = x_dot_closed_loop(0, x0)
	print("Rollout has %i steps" % x_rollout.shape[1])
	IPython.embed()
	"""

	T_max = 50
	x = x0.copy()
	x_rollout = [x]
	u_rollout = []
	eps_slack_rollout = []
	# IPython.embed()

	for t in range(T_max):
		u, debug_dict = u_ours(t, x)
		x_dot = x_dot_open_loop(x, u)
		x = x + dt*x_dot

		u_rollout.append(u)
		x_rollout.append(x)
		eps_slack_rollout.append(debug_dict["eps_slack"])

	# IPython.embed()

	# x = np.array([-0.00887287, 0.68213777, -0.08144935, 2.92963457])
	# x_next = np.array([-0.00928012, 0.69678594, -0.08291683, 3.01757929])
	#
	# phi_x = numpy_phi_fn(x)
	# phi_x_next = numpy_phi_fn(x_next)
	#
	# print(phi_x, phi_x_next)

	"""
	tensor([[-0.1515, -0.1116,  0.2843],
        [-0.1313, -0.0893,  0.3068]], grad_fn=<CatBackward>)
	"""


