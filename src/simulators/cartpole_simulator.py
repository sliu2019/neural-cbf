"""
Simulates the cartpole (inverted pendulum on cart) dynamical system.
"""

"""
Code structure:
controller function
step function for dynamics with dt argument

dynamics simulation loop
outputs array of states 

function that runs experiments, calling the simulation loop many times with different parameters 

animator: pass in array and save output 

use the visualizer to debug 
and create illustrations of good samples
"""

import numpy as np
import torch
from torch.autograd import grad
import IPython
import math
from main import Phi
from src.argument import parser
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

dt = 0.01

g = 9.81
I= 0.099
m= 0.2
M= 2
l= 0.5

# TODO
max_theta= math.pi / 10.0
max_force= 1.0

def step(X, u):
	"""
	Discrete time
	Computes next state from current and input
	"""
	X_dot = np.zeros(4)

	X_dot[0] = X[2]
	X_dot[1] = X[3]

	theta = X[1]
	theta_dot = X[3]
	denom = I*(M + m) + m*(l**2)*(M + m*(math.sin(theta)**2))
	X_dot[2] = (I + m*(l**2))*(m*l*theta_dot**2*math.sin(theta)) - g*(m**2)*(l**2)*math.sin(theta)*math.cos(theta) + (I + m*l**2)*u
	X_dot[3] = m*l*(-m*l*theta_dot**2*math.sin(theta)*math.cos(theta) + (M+m)*g*math.sin(theta)) + (-m*l*math.cos(theta))*u

	X_dot[2] = X_dot[2]/denom
	X_dot[3] = X_dot[3]/denom

	X_next = X + dt*X_dot
	return X_next

####################################################################################################################
# Loading trained phi
r = 2
x_dim = 4
u_dim = 1
x_lim = np.array([[-5, 5], [-math.pi/2.0, math.pi/2.0], [-10, 10], [-5, 5]], dtype=np.float32) # TODO

# Create phi
from src.problems.cartpole import H, XDot, ULimitSetVertices
param_dict = {
	"I": 0.099,
	"m": 0.2,
	"M": 2,
	"l": 0.5,
	"max_theta": math.pi / 10.0,
	"max_force": 1.0
}

h_fn = H(param_dict)
xdot_fn = XDot(param_dict)
# uvertices_fn = ULimitSetVertices(param_dict)

args = parser()
my_phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, args)
###################################################

class CBF_controller():
	def __init__(self, phi_fn, nominal_controller):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def __call__(self, X):
		"""
		X is (4) vec
		"""
		# Phi grad
		X_input = torch.from_numpy(X.astype("float32")).view(-1, x_dim)
		X_input.requires_grad = True
		phi_val = self.phi_fn(X_input) # TODO: need to refactor this
		phi_grad = grad([phi_val], X_input)[0]
		X_input.requires_grad = False

		# Get f(x), g(x)
		# honestly, this is a hacky way
		U_input = torch.zeros(1, 1)
		f_x = xdot_fn(X_input, U_input) # xdot defined globally
		U_input = torch.ones(1, 1)
		g_x = xdot_fn(X_input, U_input) - f_x

		# IPython.embed()
		lhs = phi_grad.mm(g_x.t())
		rhs = -phi_grad.mm(f_x.t())

		# To numpy
		lhs = lhs.cpu().detach().numpy()[0,0]
		rhs = rhs.cpu().detach().numpy()[0,0]
		# Done computing CBF constraint

		u_nom = self.nominal_controller(X)
		u_safe = u_nom
		u_safe = np.clip(u_safe, -max_force, max_force)

		# Cbf constraint
		if lhs >= 0:
			u_safe = np.clip(u_safe, None, rhs / lhs)
		else:
			u_safe = np.clip(u_safe, rhs / lhs, None)
		return u_safe

####################################################################################################################

def simulate_rollout(X0, controller, max_time=2): # max_time relative to dt
	"""
	Simulate one rollout and return data
	"""
	X = X0.copy()
	X_all = X[None]
	u_all = []
	i = 0
	while True:
		u = controller(X)
		X = step(X, u)

		X_all = np.concatenate((X_all, X[None]), axis=0)
		u_all.append(u)

		# TODO: when to terminate rollout? Just when cart position OOB or also when theta is?
		if np.any(X[:2]<x_lim[:2, 0]) or np.any(X[:2]>x_lim[:2, 1]):
			print("State out of bounds, terminating rollout")
			rollout_terminate_reason = "state out of bound"
			break
		elif dt*i > max_time:
			print("Timeout, terminating rollout")
			rollout_terminate_reason = "time"
			break

		if np.any(X[2:]<x_lim[2:, 0]) or np.any(X[2:]>x_lim[2:, 1]):
			print("Vel or angular vel is out of bounds! Check")
			print("X: ", X)
			print("X_lim", x_lim)
			# break

		i += 1

	return X_all, np.array(u_all), rollout_terminate_reason

def run_experiment(controller_type, save_fpth):
	"""
	Launches multiple experiments
	Calculates metrics across them

	Saves experimental data + computed metrics
	"""
	# TODO: multiple trials, with different initial data
	# TODO: initialization must be safe...so depends on the CBF
	X0_list = [np.zeros(4)]

	if controller_type == "my_cbf":
		nominal_controller = lambda x: 0.0
		controller = CBF_controller(my_phi_fn, nominal_controller)
	elif controller_type == "baseline_cbf":
		raise NotImplementedError # TODO
	else:
		raise NotImplementedError

	# Run 1 trial before you do a panel

	X_experiment = [] # n_rollouts, n_rollout_steps (may differ across rollouts), x_dim=4
	U_experiment = [] # n_rollouts, n_rollout_steps, 1
	terminate_reasons = []
	for X0 in X0_list:
		X_rollout, u_rollout, rollout_terminate_reason = simulate_rollout(X0, controller)

		X_experiment.append(X_rollout)
		U_experiment.append(u_rollout)
		terminate_reasons.append(rollout_terminate_reason)

	# Compute metrics
	compute_metrics(X_experiment, U_experiment, terminate_reasons)

	# Save
	save_dict = {"X_experiment": X_experiment, "U_experiment": U_experiment, "terminate_reasons": terminate_reasons}
	with open(save_fpth, 'wb') as handle:
		pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return X_experiment, U_experiment, terminate_reasons

def compute_metrics(X_experiment, U_experiment, terminate_reasons):
	# Number of rollouts with safety violation
	# Max angle across rollouts
	safety_violation_experiment = [] # list of bools
	max_abs_angle_experiment = []
	control_saturated_experiment = [] # list of counts
	for i in range(len(X_experiment)):
		X_rollout = X_experiment[i]
		U_rollout = U_experiment[i]

		if np.any(np.abs(X_rollout[:, 1]) > max_theta):
			safety_violation_experiment.append(1)
		else:
			safety_violation_experiment.append(0)

		max_abs_angle_experiment.append(np.max(np.abs(X_rollout[:, 1])))

		n_times_control_saturated_rollout = np.sum(np.abs(U_experiment) > max_force)
		control_saturated_experiment.append(n_times_control_saturated_rollout)

	print("Percent rollouts with violation: %f" % (np.mean(safety_violation_experiment)))
	print("Average maximum (absolute) angle: %f" % (np.mean(max_abs_angle_experiment)))
	print("Average number of times control exceeded threshold, and was saturated: %f" % (np.mean(control_saturated_experiment)))

def load_experiment(load_fpth):
	with open(load_fpth, 'rb') as handle:
		save_dict = pickle.load(handle)

	return save_dict["X_experiment"], save_dict["U_experiment"], save_dict["terminate_reasons"]

####################################################################################################################
# Animation utilities
fig = plt.figure()
ax = fig.add_subplot(111, aspect = 'equal', xlim = (x_lim[0, 0], x_lim[0, 1]), ylim = (-1, 1), title = "Inverted Pendulum Simulation")
ax.grid()

# animation parameters
origin = [0.0, 0.0]
dt = 0.02

pendulumArm = lines.Line2D(origin, origin, color='r')
cart = patches.Rectangle(origin, 0.5, 0.15, color='b')

def init():
	ax.add_patch(cart)
	ax.add_line(pendulumArm)
	return pendulumArm, cart


def animate_rollout(X_rollout, U_rollout, rollout_terminate_reason, save_fpth):
	def animate(i):
		# TODO: what is the theta=0 position in this animation code?
		xPos = X_rollout[i, 0]
		theta = X_rollout[i, 1]
		x = [origin[0] + xPos, origin[0] + xPos + l * np.sin(theta)]
		y = [origin[1], origin[1] - l * np.cos(theta)]

		pendulumArm.set_xdata(x)
		pendulumArm.set_ydata(y)
		cartPos = [origin[0] + xPos - cart.get_width() / 2, origin[1] - cart.get_height()]
		cart.set_xy(cartPos)
		return pendulumArm, cart

	anim = animation.FuncAnimation(fig, animate, init_func = init, interval = 1000*dt, blit = True) # interval: real time, interval between frames in ms
	# plt.show()
	FFwriter = animation.FFMpegWriter()
	anim.save(save_fpth, writer=FFwriter, fps=10)

####################################################################################################################
# Misc
def test_in_S(x0):
	# IPython.embed()
	x0 = x0.view(-1, x_dim)
	x0.requires_grad = True
	h_val = h_fn(x0)
	grad_h = grad([h_val], x0)[0]
	x0.requires_grad = False

	u = torch.zeros(1, 1)
	xdot_val = xdot_fn(x0, u)
	phi_1 = h_fn(x0) + my_phi_fn.ci[0]*grad_h.mm(xdot_val.t())

	print(h_fn(x0), phi_1, my_phi_fn(x0))
	if h_fn(x0) < 0 and phi_1 < 0 and my_phi_fn(x0) < 0:
		return True
	return False

####################################################################################################################

if __name__ == "__main__":
	x0 = torch.tensor([0.0, math.pi/2, 0, 0])
	print(test_in_S(x0))

	# TODO: run with nohup writing out to file! The printout will be really informative.
	# X_experiment, U_experiment, terminate_reasons = run_experiment("my_cbf", "./log/cartpole_default/simulations/debug.pkl")
	# animate_rollout(X_experiment[0], U_experiment[0], terminate_reasons[0], "./log/cartpole_default/simulations/debug.mp4")

"""
TODO's:
1. What is the nominal controller? Could be Lyapunov (like Ames' QP-CBF paper)
2. What's the baseline CBF?
3. *(Later) Ask Changliu and John what other controllers they want to see?
"""