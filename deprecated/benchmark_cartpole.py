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
# from src.argument import parser
from src.utils import *
from main import Phi
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

from src.attacks.gradient_batch_attacker import GradientBatchAttacker
from src.utils import *
from main import *

dt = 0.01
g = 9.81

dev = "cpu"
device = torch.device(dev)

# Note: everything for 4D cartpole, not 2D pole
r = 2
x_dim = 4
u_dim = 1
# x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)
x_lim = np.array([[-5, 5], [-math.pi, math.pi], [-10, 10], [-5, 5]], dtype=np.float32)

mode = 'easy' # TODO
if mode == 'easy':
	param_dict = {
		"I": 0.021,
		"m": 0.25,
		"M": 1.00,
		"l": 0.5,
		"max_theta": math.pi / 2.0,
		"max_force": 15.0
	}

elif mode == 'hard':
	param_dict = {
		"I": 0.021,
		"m": 0.25,
		"M": 1.00,
		"l": 0.5,
		"max_theta": math.pi / 4.0,
		"max_force": 1.0
	}
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

	def simulate_rollout(self, x0, controller, stop_criterion="max_time"): # max_time relative to dt
		"""
		x0: (4) vector
		Simulate one rollout and return data
		Stop criteria: max_time or in_S
		"""
		x = x0.copy()

		x_all = [np.reshape(x0, (1, x_dim))]
		u_all = []
		u_preclip_all = []
		i = 0
		while True:
			# u = controller(x, self.xdot_fn_numpy) # TODO: remove
			u = 0.0
			u_feas = np.clip(u, -self.param_dict["max_force"], self.param_dict["max_force"])

			x = self.step(x, u_feas)

			x_all.append(np.reshape(x, (1, x_dim)))
			u_preclip_all.append(np.reshape(u, (1, u_dim)))
			u_all.append(np.reshape(u_feas, (1, u_dim)))

			if stop_criterion == "max_time" and dt*i > self.max_time:
				print("Timeout, terminating rollout")
				rollout_terminate_reason = "time"
				break
			elif stop_criterion == "in_S" and u == 0: # TODO: hacky
				print("In S, terminating rollout")
				break
			# print(x)
			# print(u)

			i += 1

		x_all = np.concatenate(x_all, axis=0)
		u_preclip_all = np.concatenate(u_preclip_all, axis=0)
		u_all = np.concatenate(u_all, axis=0)
		return x_all, u_all, u_preclip_all

def animate_rollout(x_rollout, save_fpth):
	l = param_dict["l"]

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
		x = [origin[0] + xPos, origin[0] + xPos + l * np.sin(theta)]
		y = [origin[1], origin[1] + l * np.cos(theta)]
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
class CBF_controller():
	def __init__(self, phi_fn, xdot_fn_numpy, param_dict, eps=1.0):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def convert_angle_to_negpi_pi_interval(self, angle):
		new_angle = np.arctan2(np.sin(angle), np.cos(angle))
		return new_angle

	def __call__(self, x, xdot_fn_numpy):
		"""
		x is (4) vec
		RV u is numpy (1) vec
		"""
		# Prepare first
		theta = self.convert_angle_to_negpi_pi_interval(x[1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
		# print("Check that this angle is in [-pi, pi] range: %f" % theta)
		assert theta < math.pi and theta > -math.pi
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
		x_dot = xdot_fn_numpy(x, 0.0)
		x_next = x + dt*x_dot
		x_next_trunc = np.array([x_next[1], x_next[3]])
		x_next_torch = torch.from_numpy(x_next_trunc.astype("float32")).view(-1, 2)
		next_phi = self.phi_fn(x_next_torch)[0, -1].item()

		# IPython.embed()
		if phi_val > 0:
			eps = self.eps
		elif phi_val < 0 and next_phi >= 0: # TODO: cheating way to convert DT to CT
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

def load_trained_cbf(exp_name, checkpoint_number):
	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)

	args = load_args("./log/%s/args.txt" % exp_name)

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	# if args.physical_difficulty == 'easy':
	# 	param_dict = {
	# 		"I": 0.021,
	# 		"m": 0.25,
	# 		"M": 1.00,
	# 		"l": 0.5,
	# 		"max_theta": math.pi / 2.0,
	# 		"max_force": 15.0
	# 	}
	# elif args.physical_difficulty == 'hard':
	# 	param_dict = {
	# 		"I": 0.021,
	# 		"m": 0.25,
	# 		"M": 1.00,
	# 		"l": 0.5,
	# 		"max_theta": math.pi / 4.0,
	# 		"max_force": 1.0
	# 	}

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	x_e = torch.zeros(1, x_dim)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	###################################
	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)

	return phi_fn

####################################################################################################################

def run_experiment(x0_list, simulator, controller, save_fpth, stop_criterion="max_time"):
	"""
	Launches multiple experiments
	Calculates metrics across them

	Saves experimental data + computed metrics
	"""

	x_experiment = [] # n_rollouts, n_rollout_steps (may differ across rollout_results), x_dim=4
	u_experiment = [] # n_rollouts, n_rollout_steps, 1
	u_preclip_experiment = []

	for x0 in x0_list:
		x_rollout, u_rollout, u_preclip_rollout = simulator.simulate_rollout(x0, controller, stop_criterion)

		x_experiment.append(x_rollout)
		u_experiment.append(u_rollout)
		u_preclip_experiment.append(u_preclip_rollout)

	# Compute metrics
	# compute_metrics(x_experiment, u_experiment, u_preclip_experiment)

	# Save
	save_dict = {"x_experiment": x_experiment, "u_experiment": u_experiment, "u_preclip_experiment": u_preclip_experiment, "x0_list": x0_list}
	with open(save_fpth, 'wb') as handle:
		pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return x_experiment, u_experiment, u_preclip_experiment

# def compute_metrics(x_experiment, u_experiment, u_preclip_experiment):
# 	max_theta = param_dict["max_theta"]
# 	max_force = param_dict["max_force"]
#
# 	# Number of rollout_results with safety violation
# 	# Max angle across rollout_results
# 	safety_violation_experiment = [] # list of bools
# 	max_abs_angle_experiment = []
# 	control_saturated_experiment = [] # list of counts
# 	for i in range(len(x_experiment)):
# 		x_rollout = x_experiment[i]
# 		u_rollout = u_experiment[i]
#
# 		safety_violation_experiment.append(np.sum(np.abs(x_rollout[:, 1]) > max_theta)) # TODO: sum to mean?
# 		max_abs_angle_experiment.append(np.max(np.abs(x_rollout[:, 1])))
# 		control_saturated_experiment.append(np.sum(np.abs(u_rollout) > max_force))
#
# 	print("Average number of violation: %f" % (np.mean(safety_violation_experiment)))
# 	print("Average maximum (absolute) angle: %f" % (np.mean(max_abs_angle_experiment)))
# 	print("Average number of times control exceeded threshold, and was saturated: %f" % (np.mean(control_saturated_experiment)))

def load_experiment(load_fpth):
	with open(load_fpth, 'rb') as handle:
		save_dict = pickle.load(handle)

	return save_dict

####################################################################################################################
def run_benchmark(phi_fn, other_phi_fn, save_fldr, which_tests=["FI", "FTC_G", "FTC_F", "S_vol"], n_x0=30):
	xdot_fn_numpy = XDot_numpy(param_dict)
	simulator = Cartpole_Simulator(xdot_fn_numpy, param_dict)

	controller = CBF_controller(phi_fn, xdot_fn_numpy, param_dict, eps=10.0) # TODO: eps

	# other_phi_fn = load_trained_cbf(other_exp_name, other_checkpoint_number)

	# Defining vars
	max_theta = param_dict["max_theta"]
	max_force = param_dict["max_force"]

	# Compute optimization objective
	# TODO: actually, do this in main.py

	# Create attacker
	logger = create_logger("log/discard", 'train', 'info')
	x_lim_pole = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)
	x_lim_pole = torch.tensor(x_lim_pole).to(device)
	attacker = GradientBatchAttacker(x_lim_pole, device, logger, n_samples=n_x0)

	# print("Before FI")
	# IPython.embed()
	############################################################################################
	# Compute FI performance
	"""
	Add options to attacker to sample n points on dG and dS or dG and not dS
	Let x0 be n points in dG and dS
	Implement metrics here + delete the function: 
	1. % rollout_results: 0/1 stay in S the whole rollout?
	2. amount of safe set violation, measured 2 ways
		a. number of timesteps outside of S over the rollout 
		b. max(phi_i) over the rollout
	3. amount of control limit violation, measured 2 ways
		a. max violation over rollout
		b. avg + std violation over rollout
		c. number of violations over rollout 
	"""
	if "FI" in which_tests:
		# TODO: uncomment
		# bdry_points = attacker.sample_points_on_boundary(phi_fn, mode="dG+dS")
		# x0_pole = bdry_points.detach().cpu().numpy()
		# x0_array = np.zeros((n_x0, 4))
		# x0_array[:, 1] = x0_pole[:, 0]
		# x0_array[:, 3] = x0_pole[:, 1]
		# x0_list = x0_array.tolist()

		# x0_list = [np.array([0, -1.5796646, 0, 1.0705544])]
		x0_list = [np.array([0, 0, 0, 1.0])]

		save_fpth = "./simulations/%s_(%i)_FI.pkl" % (exp_name, checkpoint_number)
		x_experiment, u_experiment, u_preclip_experiment = run_experiment(x0_list, simulator, controller, save_fpth)

		# Compute phi_vals
		phi_experiment = []
		for rl in x_experiment:
			pole_rl = np.concatenate((rl[:, [1]], rl[:, [3]]), axis=1).astype(np.float32)
			pole_rl = torch.from_numpy(pole_rl)
			phi_vals = phi_fn(pole_rl)
			phi_experiment.append(phi_vals.detach().numpy())

		############## Metrics ################
		in_S_always = [np.all(rl_phi <= 0) for rl_phi in phi_experiment]
		print("in_S_always", in_S_always)
		in_S_always = np.mean(in_S_always)
		print("Should be 1: ", in_S_always)

		n_tsteps_outside_S = [np.sum(np.any(rl_phi > 0, axis=1)) for rl_phi in phi_experiment]
		print("n_tsteps_outside_S", n_tsteps_outside_S)
		n_tsteps_outside_S = np.mean(n_tsteps_outside_S)
		print("Should be 0: ", n_tsteps_outside_S)

		max_phii = np.array([np.max(rl_phi) for rl_phi in phi_experiment])
		max_phii = np.mean(np.where(max_phii > 0, max_phii, 0))
		print("Max phi: ", max_phii)

		violations = [np.clip(np.abs(rl_u_preclip) - max_force, 0, None) for rl_u_preclip in u_preclip_experiment]
		avg_control_violations = np.mean([np.mean(x) for x in violations])
		number_control_violations = np.mean([np.sum(x > 0) for x in violations])
		print(avg_control_violations, number_control_violations)

		# animate_rollout(x_experiment[0], "./animations/debug_cbf_0.mp4")
		# animate_rollout(x_experiment[1], "./animations/debug_cbf_1.mp4")
		# animate_rollout(x_experiment[2], "./animations/debug_cbf_2.mp4")

		which_rl_to_animate = []
		for i in which_rl_to_animate:
			animate_rollout(x_experiment[i], "./animations/debug_cbf_%i.mp4" % i)

		# array([ 0, 29, 21, 11, 12,  2,  6, 10,  4, 23, 24,  1, 25, 18,  8, 19, 14,
		#         5,  3,  7, 15, 16, 27, 17,  9, 28, 13, 22, 20, 26])

		print(np.argsort(max_phii)[::-1])
		print(np.max(x_experiment[:, -1]))
		IPython.embed()
	############################################################################################
	# print("Before FTC G")
	# IPython.embed()
	# Compute FTC performance
	"""
	Part 1 
	Let x0 be n points in dG and not dS 
	Implement metrics
	1-3. same but for G
	
	Part 2 
	Let x0 be n points that are outside of G for both phi1 and phi2
		You can just sample a grid with density on binary search and evaluate with phi; count number of points for which phi1>0, phi2>0 
	Make the rollout_results longer 
	Implement metrics
	1. # steps to S
	2. amount of control limit violation 
	2. Check: is the performance monotonic along a rollout?
	"""
	if "FTC_G" in which_tests:
		bdry_points = attacker.sample_points_on_boundary(phi_fn, mode="dG/dS") # TODO
		x0_pole = bdry_points.detach().cpu().numpy()
		x0_array = np.zeros((n_x0, 4))
		x0_array[:, 1] = x0_pole[:, 0]
		x0_array[:, 3] = x0_pole[:, 1]
		x0_list = x0_array.tolist()

		save_fpth = "./simulations/%s_(%i)_FTC_G.pkl" % (exp_name, checkpoint_number) # TODO
		x_experiment, u_experiment, u_preclip_experiment = run_experiment(x0_list, simulator, controller, save_fpth)

		# Compute phi_vals
		phi_experiment = []
		for rl in x_experiment:
			pole_rl = np.concatenate((rl[:, [1]], rl[:, [3]]), axis=1).astype(np.float32)
			pole_rl = torch.from_numpy(pole_rl)
			phi_vals = phi_fn(pole_rl)
			phi_experiment.append(phi_vals.detach().numpy())

		############## Metrics ################
		in_G_always = [np.all(rl_phi[:, -1] <= 0) for rl_phi in phi_experiment] # TODO
		in_G_always = np.mean(in_G_always)

		n_tsteps_outside_G = [np.sum(rl_phi[:, -1] > 0) for rl_phi in phi_experiment] # TODO
		n_tsteps_outside_G = np.mean(n_tsteps_outside_G)

		max_phii = np.array([np.max(rl_phi[:, -1]) for rl_phi in phi_experiment])
		max_phii = np.mean(np.where(max_phii > 0, max_phii, 0))

		violations = [np.clip(np.abs(rl_u_preclip) - max_force, 0, None) for rl_u_preclip in u_preclip_experiment]
		avg_control_violations = np.mean([np.mean(x) for x in violations])
		number_control_violations = np.mean([np.sum(x != 0) for x in violations])

	############################################################################################
	# print("Before FTC F")
	# IPython.embed()
	# Compute x0
	if "FTC_F" in which_tests:
		delta = 0.75
		x = np.arange(x_lim[1, 0], x_lim[1, 1], delta)
		y = np.arange(x_lim[3, 0], x_lim[3, 1], delta)[::-1] # need to reverse it
		X, Y = np.meshgrid(x, y)
		input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
		input = torch.from_numpy(input)
		phi_vals = phi_fn(input)
		other_phi_vals = other_phi_fn(input)

		F_both = (phi_vals[:, -1] > 0)*(other_phi_vals[:, -1] > 0)
		ind = np.transpose(np.nonzero(F_both))
		ind = ind.detach().numpy()
		x0_pole = input[ind[0, :], :]
		x0_array = np.zeros((x0_pole.shape[0], 4))
		x0_array[:, 1] = x0_pole[:, 0]
		x0_array[:, 3] = x0_pole[:, 1]
		x0_list = x0_array.tolist()

		save_fpth = "./simulations/%s_(%i)_FTC_F.pkl" % (exp_name, checkpoint_number)
		x_experiment, u_experiment, u_preclip_experiment = run_experiment(x0_list, simulator, controller, save_fpth, stop_criterion="in_S") # TODO: make run longer
		phi_experiment = []
		for rl in x_experiment:
			pole_rl = np.concatenate((rl[:, [1]], rl[:, [3]]), axis=1).astype(np.float32)
			pole_rl = torch.from_numpy(pole_rl)
			phi_vals = phi_fn(pole_rl)
			phi_experiment.append(phi_vals.detach().numpy())

		############## Metrics ################
		steps_to_S = [np.transpose(np.nonzero(np.all(rl_phi < 0, axis=1))) for rl_phi in phi_experiment]
		steps_to_S = np.mean(steps_to_S)
		violations = [np.clip(np.abs(rl_u_preclip) - max_force, 0, None) for rl_u_preclip in u_preclip_experiment]
		avg_control_violations = np.mean([np.mean(x) for x in violations])
		number_control_violations = np.mean([np.sum(x != 0) for x in violations])
	############################################################################################
	# print("Computing S volume")
	# IPython.embed()
	# Compute S volume
	"""
	Eval on grid and count % in S
	"""
	if "S_vol" in which_tests:
		delta = 0.01
		x = np.arange(x_lim[1, 0], x_lim[1, 1], delta)
		y = np.arange(x_lim[3, 0], x_lim[3, 1], delta)[::-1] # need to reverse it
		X, Y = np.meshgrid(x, y)
		input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
		input = torch.from_numpy(input)
		phi_vals = phi_fn(input)

		vol_S = np.mean(np.all(phi_vals > 0, axis=1))
	############################################################################################

if __name__ == "__main__":
	checkpoint_number = 320 # TODO
	exp_name = "cartpole_reduced_new_h_l_50_w_1"
	# phi_fn = load_trained_cbf(exp_name, checkpoint_number)
	save_fldrnm = "%s_%i" % (exp_name, checkpoint_number)

	from src.problems.cartpole_reduced import H, XDot
	from deprecated.phi_baseline import PhiBaseline

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	ci = [2.0] # TODO
	beta = param_dict["max_theta"] - 0.1 # TODO
	other_phi_fn = PhiBaseline(h_fn, ci, beta, xdot_fn, r, x_dim, u_dim, device)

	# print("make sure phi works!")
	# IPython.embed()
	# run_benchmark(phi_fn, other_phi_fn, save_fldrnm, ["FI"], n_x0=30)

	xdot_fn_numpy = XDot_numpy(param_dict)
	simulator = Cartpole_Simulator(xdot_fn_numpy, param_dict)
	x0 = [0, 0, 0, 1.0]
	x_rollout, u_rollout, u_preclip_rollout = simulator.simulate_rollout(x0, None)
	print(np.max(x_rollout[:, -1]))
	IPython.embed()




