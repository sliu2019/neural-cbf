"""
Pseudo-code:
1. Load model

Make 2 plots:
1. Phase plot: start trajectories at various points, simulate full trajectories, plot on top of invariant set (do we converge to)?
2. Plot trajectories starting inside invariant set

Code structure:
use solveivp [can you run it?]
create dynamics function
create policy function to be called within it

do the plotting in the main function
"""
import numpy as np
from control import lqr
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import IPython
import os

from main import Phi

g = 9.81
class ClosedLoopDynamics():
	def __init__(self, param_dict, controller):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		self.I = param_dict["I"]
		self.M = param_dict["M"]
		self.m = param_dict["m"]
		self.l = param_dict["l"]

	def __call__(self, t, x):
		u = self.controller(x)

		x_dot = np.zeros(4)

		x_dot[0] = x[2]
		x_dot[1] = x[3]

		theta = x[1]
		theta_dot = x[3]
		denom = self.I * (self.M + self.m) + self.m * (self.l ** 2) * (self.M + self.m * (math.sin(theta) ** 2))
		x_dot[2] = (self.I + self.m * (self.l ** 2)) * (self.m * self.l * theta_dot ** 2 * math.sin(theta)) - g * (
					self.m ** 2) * (self.l ** 2) * math.sin(theta) * math.cos(theta) + (
					           self.I + self.m * self.l ** 2) * u
		x_dot[3] = self.m * self.l * (-self.m * self.l * theta_dot ** 2 * math.sin(theta) * math.cos(theta) + (
					self.M + self.m) * g * math.sin(theta)) + (-self.m * self.l * math.cos(theta)) * u

		x_dot[2] = x_dot[2] / denom
		x_dot[3] = x_dot[3] / denom
		return x_dot

class OurCBFController():
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

def load_our_cbf(exp_name, checkpoint_number):
	log_folder = "./log/%s" % exp_name
	args = load_args(os.path.join(log_folder, "args.txt"))
	with open(os.path.join(log_folder, "param_dict.pkl"), 'wb') as handle:
		param_dict = pickle.load(os.path.join(log_folder, "param_dict.pkl"))

	# Create phi
	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-args.max_angular_velocity, args.max_angular_velocity]], dtype=np.float32)

	dev = "cuda:0"
	device = torch.device(dev)

	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	x_e = torch.zeros(1, x_dim)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	###################################
	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)

	return phi_fn

def main():
	exp_name = "cartpole_reduced_64_64_40-2"
	checkpoint_number = 1020
	# Load model and load param_dict (pendulum parameters)
	phi_fn = load_our_cbf(exp_name, checkpoint_number)

	# Create controller
	controller = OurCBFController()
	# Pass into dynamics
	cld = ClosedLoopDynamics(param_dict, controller)

	# Pass closed-loop dynamics to solve_ivp


if __name__=="__main__":
	main()