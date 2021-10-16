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
from src.argument import parser
from src.utils import *
from main import Phi
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines


dt = 0.01
g = 9.81

dev = "cpu"
device = torch.device(dev)

# TODO: are we expecting reduced or full?
r = 2
x_dim = 4
u_dim = 1
# x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)
x_lim = np.array([[-5, 5], [-math.pi, math.pi], [-10, 10], [-5, 5]], dtype=np.float32)

####################################################################################################################
class XDot_numpy():
	def __init__(self, param_dict):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		self.I = param_dict["I"]
		self.M = param_dict["M"]
		self.m = param_dict["m"]
		self.l = param_dict["l"]

	def __call__(self, x, u):
		x_dot = np.zeros(4)

		x_dot[0] = x[2]
		x_dot[1] = x[3]

		theta = x[1]
		theta_dot = x[3]
		denom = self.I*(self.M + self.m) + self.m*(self.l**2)*(self.M + self.m*(math.sin(theta)**2))
		x_dot[2] = (self.I + self.m*(self.l**2))*(self.m*self.l*theta_dot**2*math.sin(theta)) - g*(self.m**2)*(self.l**2)*math.sin(theta)*math.cos(theta) + (self.I + self.m*self.l**2)*u
		x_dot[3] = self.m*self.l*(-self.m*self.l*theta_dot**2*math.sin(theta)*math.cos(theta) + (self.M+self.m)*g*math.sin(theta)) + (-self.m*self.l*math.cos(theta))*u

		x_dot[2] = x_dot[2]/denom
		x_dot[3] = x_dot[3]/denom
		return x_dot

class Cartpole_Simulator():
	def __init__(self, xdot_fn_numpy, param_dict, max_time=1.0):
		"""
		Simulates + animates
		"""
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		self.l = self.param_dict["l"]

	def step(self, x, u):
		x_dot = self.xdot_fn_numpy(x, u)
		x_next = x + dt*x_dot
		return x_next

	def simulate_rollout(self, x0, controller): # max_time relative to dt
		"""
		x0: (4) vector
		Simulate one rollout and return data
		"""
		x = x0.copy()

		x_all = [np.reshape(x0, (1, x_dim))]
		u_all = []
		u_preclip_all = []
		i = 0
		while True:
			u = controller(x)
			u_feas = np.clip(u, -self.param_dict["max_force"], self.param_dict["max_force"])

			x = self.step(x, u_feas)

			x_all.append(np.reshape(x, (1, x_dim)))
			u_preclip_all.append(np.reshape(u, (1, u_dim)))
			u_all.append(np.reshape(u_feas, (1, u_dim)))

			if dt*i > self.max_time:
				print("Timeout, terminating rollout")
				rollout_terminate_reason = "time"
				break

			i += 1

		x_all = np.concatenate(x_all, axis=0)
		u_preclip_all = np.concatenate(u_preclip_all, axis=0)
		u_all = np.concatenate(u_all, axis=0)
		return x_all, u_all, u_preclip_all

	def animate_rollout(self, x_rollout, save_fpth):
		# Animation utilities
		fig = plt.figure()
		ax = fig.add_subplot(111, aspect='equal', xlim=(x_lim[0, 0], x_lim[0, 1]), ylim=(-1, 1),
		                     title="Inverted Pendulum Simulation")
		ax.grid()

		# animation parameters
		origin = [0.0, 0.0]
		anim_dt = 0.02

		pendulumArm = lines.Line2D(origin, origin, color='r')
		cart = patches.Rectangle(origin, 0.5, 0.15, color='b')

		def init():
			ax.add_patch(cart)
			ax.add_line(pendulumArm)
			return pendulumArm, cart

		def animate(i):
			xPos = x_rollout[i, 0]
			theta = x_rollout[i, 1]
			x = [origin[0] + xPos, origin[0] + xPos + self.l * np.sin(theta)]
			y = [origin[1], origin[1] + self.l * np.cos(theta)]
			pendulumArm.set_xdata(x)
			pendulumArm.set_ydata(y)

			cartPos = [origin[0] + xPos - cart.get_width() / 2, origin[1] - cart.get_height()]
			cart.set_xy(cartPos)
			return pendulumArm, cart

		anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1000 * anim_dt,
		                               blit=True)  # interval: real time, interval between frames in ms
		# plt.show()
		FFwriter = animation.FFMpegWriter(fps=10)
		anim.save(save_fpth, writer=FFwriter)

####################################################################################################################

def load_trained_cbf(exp_name, checkpoint_number):
	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)

	args = load_args("./log/%s/args.txt" % exp_name)

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	if args.physical_difficulty == 'easy':
		param_dict = {
			"I": 0.021,
			"m": 0.25,
			"M": 1.00,
			"l": 0.5,
			"max_theta": math.pi / 2.0,
			"max_force": 15.0
		}
	elif args.physical_difficulty == 'hard':
		param_dict = {
			"I": 0.021,
			"m": 0.25,
			"M": 1.00,
			"l": 0.5,
			"max_theta": math.pi / 4.0,
			"max_force": 1.0
		}

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	x_e = torch.zeros(1, x_dim)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	###################################
	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)

	return phi_fn, param_dict

class CBF_controller():
	def __init__(self, phi_fn, xdot_fn_numpy, param_dict, eps=1.0):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def convert_angle_to_negpi_pi_interval(self, angle):
		new_angle = np.arctan2(np.sin(angle), np.cos(angle))
		return new_angle

	def __call__(self, x):
		"""
		x is (4) vec
		RV u is numpy (1) vec
		"""
		# Prepare first
		theta = self.convert_angle_to_negpi_pi_interval(x[1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
		print("Check that this angle is in [-pi, pi] range: %f" % theta)
		x_trunc = np.array([theta, x[3]])
		x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)
		x_input.requires_grad = True

		# Compute phi grad
		phi_output = self.phi_fn(x_input)
		phi_val = phi_output[0, -1]
		phi_grad = grad([phi_val], x_input)[0]

		# Post op
		x_input.requires_grad = False
		phi_grad = phi_grad.detach().cpu().numpy()
		phi_grad = np.array([0, phi_grad[0, 0], 0, phi_grad[0, 1]])[None]
		# IPython.embed()

		# Get eps
		# TODO: sketchy way to convert DT to CT
		if phi_val > 0:
			eps = self.eps
		elif phi_val < 0 and phi_val > -1e-3:
			eps = 0
		else:
			return np.zeros(1)

		# Compute the control constraints
		# Get f(x), g(x)
		# honestly, this is a hacky way
		f_x = self.xdot_fn_numpy(x, np.zeros(1)) # xdot defined globally
		g_x = self.xdot_fn_numpy(x, np.ones(1)) - f_x

		lhs = phi_grad@g_x.T
		rhs = -phi_grad@f_x.T - eps

		# Done computing CBF constraint
		u_nom = np.zeros(1)
		u_safe = u_nom
		# u_safe = np.clip(u_safe, -self.param_dict["max_force"], self.param_dict["max_force"])

		# Cbf constraint
		if lhs >= 0:
			u_safe = np.clip(u_safe, None, rhs / lhs)
		else:
			u_safe = np.clip(u_safe, rhs / lhs, None)
		# u_safe = np.clip(u_safe, -self.param_dict["max_force"], self.param_dict["max_force"])
		return u_safe

# def run_experiment(controller_type, save_fpth):
# 	"""
# 	Launches multiple experiments
# 	Calculates metrics across them
#
# 	Saves experimental data + computed metrics
# 	"""
# 	# TODO: multiple trials, with different initial data
# 	# TODO: initialization must be safe...so depends on the CBF
# 	X0_list = [np.zeros(4)]
#
# 	if controller_type == "my_cbf":
# 		nominal_controller = lambda x: 0.0
# 		controller = CBF_controller(my_phi_fn, nominal_controller)
# 	elif controller_type == "baseline_cbf":
# 		raise NotImplementedError # TODO
# 	else:
# 		raise NotImplementedError
#
# 	# Run 1 trial before you do a panel
#
# 	X_experiment = [] # n_rollouts, n_rollout_steps (may differ across rollouts), x_dim=4
# 	U_experiment = [] # n_rollouts, n_rollout_steps, 1
# 	terminate_reasons = []
# 	for X0 in X0_list:
# 		X_rollout, u_rollout, rollout_terminate_reason = simulate_rollout(X0, controller)
#
# 		X_experiment.append(X_rollout)
# 		U_experiment.append(u_rollout)
# 		terminate_reasons.append(rollout_terminate_reason)
#
# 	# Compute metrics
# 	compute_metrics(X_experiment, U_experiment, terminate_reasons)
#
# 	# Save
# 	save_dict = {"X_experiment": X_experiment, "U_experiment": U_experiment, "terminate_reasons": terminate_reasons}
# 	with open(save_fpth, 'wb') as handle:
# 		pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# 	return X_experiment, U_experiment, terminate_reasons
#
# def compute_metrics(X_experiment, U_experiment, terminate_reasons):
# 	# Number of rollouts with safety violation
# 	# Max angle across rollouts
# 	safety_violation_experiment = [] # list of bools
# 	max_abs_angle_experiment = []
# 	control_saturated_experiment = [] # list of counts
# 	for i in range(len(X_experiment)):
# 		X_rollout = X_experiment[i]
# 		U_rollout = U_experiment[i]
#
# 		if np.any(np.abs(X_rollout[:, 1]) > max_theta):
# 			safety_violation_experiment.append(1)
# 		else:
# 			safety_violation_experiment.append(0)
#
# 		max_abs_angle_experiment.append(np.max(np.abs(X_rollout[:, 1])))
#
# 		n_times_control_saturated_rollout = np.sum(np.abs(U_experiment) > max_force)
# 		control_saturated_experiment.append(n_times_control_saturated_rollout)
#
# 	print("Percent rollouts with violation: %f" % (np.mean(safety_violation_experiment)))
# 	print("Average maximum (absolute) angle: %f" % (np.mean(max_abs_angle_experiment)))
# 	print("Average number of times control exceeded threshold, and was saturated: %f" % (np.mean(control_saturated_experiment)))
#
# def load_experiment(load_fpth):
# 	with open(load_fpth, 'rb') as handle:
# 		save_dict = pickle.load(handle)
#
# 	return save_dict["X_experiment"], save_dict["U_experiment"], save_dict["terminate_reasons"]

####################################################################################################################


####################################################################################################################
# Misc
# def test_in_S(x0):
# 	# IPython.embed()
# 	x0 = x0.view(-1, x_dim)
# 	x0.requires_grad = True
# 	h_val = h_fn(x0)
# 	grad_h = grad([h_val], x0)[0]
# 	x0.requires_grad = False
#
# 	u = torch.zeros(1, 1)
# 	xdot_val = xdot_fn(x0, u)
# 	phi_1 = h_fn(x0) + my_phi_fn.ci[0]*grad_h.mm(xdot_val.t())
#
# 	print(h_fn(x0), phi_1, my_phi_fn(x0))
# 	if h_fn(x0) < 0 and phi_1 < 0 and my_phi_fn(x0) < 0:
# 		return True
# 	return False

####################################################################################################################

if __name__ == "__main__":
	checkpoint_number = 50
	phi_fn, param_dict = load_trained_cbf("cartpole_reduced_l_50_w_1e_1", checkpoint_number)
	# param_dict = {
	# 	"I": 0.021,
	# 	"m": 0.25,
	# 	"M": 1.00,
	# 	"l": 0.5,
	# 	"max_theta": math.pi / 2.0,
	# 	"max_force": 15.0
	# }
	# IPython.embed()

	xdot_fn_numpy = XDot_numpy(param_dict)
	cartpole_simulator = Cartpole_Simulator(xdot_fn_numpy, param_dict)

	x0 = np.array([0.0, math.pi/4, 0, 0])
	controller = CBF_controller(phi_fn, xdot_fn_numpy, param_dict)
	# controller = lambda x: np.zeros(1)

	x_rollout, u_rollout, u_preclip_rollout = cartpole_simulator.simulate_rollout(x0, controller)
	# print("rollout ended")
	# IPython.embed()
	# cartpole_simulator.animate_rollout(x_rollout, "./animations/test_cartpole_animation.mp4")
	cartpole_simulator.animate_rollout(x_rollout, "./animations/our_cbf_cartpole_animation.mp4")

# TODO: mod theta before applying controller
# TODO: CBF controller: needs to be different when applying controller at boundary vs on unsafe (eps >0)
# TODO (later): consider terminating when angle too large (would go through cart)

# TODO: run with nohup writing out to file! The printout will be really informative.
# X_experiment, U_experiment, terminate_reasons = run_experiment("my_cbf", "./log/cartpole_default/simulations/debug.pkl")
# animate_rollout(X_experiment[0], U_experiment[0], terminate_reasons[0], "./log/cartpole_default/simulations/debug.mp4")

"""
TODO's:
1. What is the nominal controller? Could be Lyapunov (like Ames' QP-CBF paper)
2. What's the baseline CBF?
3. *(Later) Ask Changliu and John what other controllers they want to see?
"""


