import numpy as np
import torch
from torch.autograd import grad
import IPython
import math
from src.utils import *
import pickle
from plot_utils import create_phi_struct_load_xlim, plot_2d_attacks_from_loaded

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

from deprecated.phi_baseline import PhiBaseline


g = 9.81

dev = "cpu"
device = torch.device(dev)

# Note: everything for 4D cartpole, not 2D pole
r = 2
x_dim = 4
u_dim = 1
# x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)
x_lim = np.array([[-5, 5], [-math.pi, math.pi], [-10, 10], [-5, 5]], dtype=np.float32)

####################################################################################################################
class CBF_controller():
	def __init__(self, phi_fn, xdot_fn_numpy, param_dict, eps=5.0, eps_bdry=1.0, dt=0.01):
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
		x_next = x + self.dt*x_dot
		x_next_trunc = np.array([x_next[1], x_next[3]])
		x_next_torch = torch.from_numpy(x_next_trunc.astype("float32")).view(-1, 2)
		next_phi = self.phi_fn(x_next_torch)[0, -1].item()

		# IPython.embed()
		if phi_val > 0:
			eps = self.eps
		elif phi_val < 0 and next_phi >= 0: # TODO: cheating way to convert DT to CT
			eps = self.eps_bdry
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

		# Cbf constraint
		if lhs >= 0:
			u_safe = np.clip(u_safe, None, rhs / lhs)
		else:
			u_safe = np.clip(u_safe, rhs / lhs, None)
		# u_safe = np.clip(u_safe, -self.param_dict["max_force"], self.param_dict["max_force"]) # Don't need this, clipping outside this fn call
		return u_safe

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
	def __init__(self, xdot_fn_numpy, param_dict, max_time=1.0, dt=0.01):
		"""
		Simulates + animates
		"""
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		self.l = self.param_dict["l"]

	def step(self, x, u):
		x_dot = self.xdot_fn_numpy(x, u)
		x_next = x + self.dt*x_dot
		return x_next

	def simulate_rollout(self, x0, controller): # max_time relative to dt
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
			u = controller(x, self.xdot_fn_numpy)
			u_feas = np.clip(u, -self.param_dict["max_force"], self.param_dict["max_force"])

			x = self.step(x, u_feas)

			x_all.append(np.reshape(x, (1, x_dim)))
			u_preclip_all.append(np.reshape(u, (1, u_dim)))
			u_all.append(np.reshape(u_feas, (1, u_dim)))

			if self.dt*i > self.max_time:
				print("Timeout, terminating rollout")
				break

			i += 1

		x_all = np.concatenate(x_all, axis=0)
		u_preclip_all = np.concatenate(u_preclip_all, axis=0)
		u_all = np.concatenate(u_all, axis=0)
		return x_all, u_all, u_preclip_all

####################################################################################################################

def animate_rollout(x_rollout, param_dict, save_fpth):
	l = param_dict["l"]
	len = x_rollout.shape[0]

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

	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len, interval=1000 * anim_dt,
	                               blit=True)  # interval: real time, delay between frames in ms
	# plt.show()
	FFwriter = animation.FFMpegWriter(fps=10)
	anim.save(save_fpth, writer=FFwriter)

def plot_2d_contours(phi_fn, exp_name):
	delta = 0.01
	# delta = 0.005
	x = np.arange(x_lim[0, 0], x_lim[0, 1], delta)
	y = np.arange(x_lim[1, 0], x_lim[1, 1], delta)[::-1] # need to reverse it # TODO
	X, Y = np.meshgrid(x, y)

	input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
	input = torch.from_numpy(input)
	rv = phi_fn(input)
	# S_vals = torch.max(phi_vals, dim=1)[0] # S = all phi_i <= 0
	# phi_signs = torch.sign(S_vals).detach().cpu().numpy()
	# phi_signs = np.reshape(phi_signs, X.shape)

	# fig, axes = plt.subplots(1, 2) # (1, 2)
	fig, ax = plt.subplots(1, 1) # (1, 2)

	# red_rgba = np.append(red_rgb, 0.5)
	# blue_rgba = np.append(blue_rgb, 0.7)
	#
	# img = np.zeros((phi_signs.shape[0], phi_signs.shape[1], 4))
	# blue_inds = np.argwhere(phi_signs == 1)
	# img[blue_inds[:, 0], blue_inds[:, 1], :] = red_rgba
	# red_inds = np.argwhere(phi_signs == -1)
	# img[red_inds[:, 0], red_inds[:, 1], :] = blue_rgba



	# for ax in axes: # TODO
	phi_vals = rv[:, -1].detach().cpu().numpy()
	img = np.reshape(phi_vals, X.shape)
	ax.imshow(img, extent=[-math.pi, math.pi, -5, 5], cmap="RdBu_r")
	ax.set_aspect("equal")
	ax.contour(X, Y, img, levels=[0.0],
	            colors=(['k']), linewidths=(2,), zorder=1)

	ax.set_xlabel("angle")
	ax.set_ylabel("ang. vel.")

	# title = "Ckpt %i, k0 = %.4f, k1 = %.4f" % (checkpoint_number, phi_fn.k0[0, 0].item(), phi_fn.ci[0, 0].item())
	# plt.title(title)
	fname = "2d_contour_from_loaded_checkpoint_%i.png" % checkpoint_number
	fpth = "./log/%s/%s" % (exp_name, fname)
	print("saving at %s" % fpth)
	plt.savefig(fpth, bbox_inches='tight')
	plt.clf()
	plt.close()

if __name__ == "__main__":
	# exp_name = "cartpole_reduced_debugpinch1_softplus_s3"
	# checkpoint_number = 750

	exp_name = "cartpole_reduced_baseline_k_1e_2"
	checkpoint_number = 0
	x0 = [0, 0.2, 0, 1.0] # [x, theta, dx, dtheta]
	dt = 0.1
	max_time = 1.0

	param_dict = pickle.load(open("./log/%s/param_dict.pkl" % exp_name, "rb"))

	# IPython.embed()

	from src.problems.cartpole_reduced import H, XDot

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	xdot_fn_numpy = XDot_numpy(param_dict)

	# phi_fn, x_lim = create_phi_struct_load_xlim(exp_name)
	# phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	# load_model(phi_fn, phi_load_fpth)
	ci = [0.01] # weights on higher order terms
	phi_fn = PhiBaseline(h_fn, ci, xdot_fn, r, x_dim, u_dim, device)

	# save_model(phi_fn, "./checkpoint/cartpole_reduced_baseline_k_1e_2/checkpoint_0.pth")

	controller = CBF_controller(phi_fn, xdot_fn_numpy, param_dict, dt=dt) # Note: controller, simulator have their own dt defined
	simulator = Cartpole_Simulator(xdot_fn_numpy, param_dict, dt=dt, max_time=max_time)
	# print("after creating all the objects")
	# IPython.embed()

	# x_rollout, u_rollout, u_preclip_rollout = simulator.simulate_rollout(x0, controller)
	# print(np.max(x_rollout[:, -1]))

	# save_fpth = "./log/%s/test_animated_rollout.mp4" % exp_name
	# save_fpth = "./log/cartpole_reduced_baseline_k_1/test_animated_rollout.mp4"
	# animate_rollout(x_rollout, param_dict, save_fpth)
	#
	# print("finished animating")
	# print(u_rollout)
	# IPython.embed()

	# checkpoint_number = 0
	# exp_name = "cartpole_reduced_baseline_k_1e_2"
	# plot_2d_attacks_from_loaded(checkpoint_number, exp_name)

	plot_2d_contours(phi_fn, exp_name)






