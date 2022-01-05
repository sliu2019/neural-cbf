import scipy as sp 
import numpy as np 
import IPython 
import torch 
import math 
from plot_utils import load_phi_xlim  
from scipy.integrate import solve_ivp
# Note: none of this is batched
# Everything done in numpy 

exp_name = "TODO"
checkpoint_number = 999

eps_value = 10 # TODO? mag?

I = 1.2E-3
m = 0.127
M = 1.0731
l = 0.3365 

max_theta = math.pi/4.0 
max_force = 22.0

phi_fn, x_lim = load_phi_xlim(exp_name, checkpoint_number)

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
	phi_val = phi_output[0,-1]
	# phi_grad = grad([phi_val], x_input)[0]

	# Post op
	# x_input.requires_grad = False
	# phi_grad = phi_grad.detach().cpu().numpy()
	# phi_grad = np.array([0, phi_grad[0, 0], 0, phi_grad[0, 1]])[None]
	print("Confirm this is scalar")
	print(type(phi_val), phi_val)
	IPython.embed()

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

	if phi_val > 0:
		eps = eps_value
	elif phi_val <= 0 and phi_val >= -2.5: # TODO: set this 
	# elif phi_val < 0 and next_phi >= 0: # TODO: cheating way to convert DT to CT
		# eps = 0
		eps = 0 
	else:
		# return np.zeros(1)
		return 0 

	# Compute the control constraints
	# Get f(x), g(x)
	f_x = x_dot_open_loop(x, 0) # xdot defined globally
	g_x = x_dot_open_loop(x, 1) - f_x

	lhs = phi_grad@g_x.T
	rhs = -phi_grad@f_x.T - eps

	# Done computing CBF constraint
	u_safe = u_ref(t, x)

	# Cbf constraint
	if lhs >= 0:
		u_safe = np.clip(u_safe, None, rhs / lhs)
	else:
		u_safe = np.clip(u_safe, rhs / lhs, None)

	return u_safe

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

	# Compute x_dot 
	x_dot = x_dot_open_loop(x, u)
	return x_dot


if __name__ == "__main__":
	theta_dot_init = 0 # TODO: try others later (can only be in the assumed range)
	theta_init = math.pi/8 # less than pi/4  
	x0 = np.array([0, theta_init, 0, theta_dot_init]) #x, theta, xdot, theta_dot
	T = 100 

	# sol = solve_ivp(x_dot_closed_loop, T, x0)
	# print("Solution status: ", sol.status)

	# x_rollout = sol.y 

	rv = x_dot_closed_loop(0, x0)
	IPython.embed()








