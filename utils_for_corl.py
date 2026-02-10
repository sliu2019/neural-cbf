import numpy as np
import math
import IPython
import sys, os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from torch.autograd import grad
import torch
from torch.autograd.functional import jacobian

from phi_numpy_wrapper import PhiNumpy

# print("In flying pend exps")
# print(sys.path)
# import socket
# if socket.gethostname() == "nsh1609server4":
# 	# IPython.embed()
# 	sys.path.extend(['/home/simin/anaconda3/envs/si_feas_env/lib/python38.zip', '/home/simin/anaconda3/envs/si_feas_env/lib/python3.8', '/home/simin/anaconda3/envs/si_feas_env/lib/python3.8/lib-dynload', '/home/simin/anaconda3/envs/si_feas_env/lib/python3.8/site-packages'])
# from cmaes.utils import load_philow_and_params

from critic import Critic
from main import SaturationRisk

# For rollouts
# from rollout_envs.quad_pend_env import FlyingInvertedPendulumEnv
# from flying_cbf_controller import CBFController
from flying_rollout_experiment import *

# For plotting slices
from flying_plot_utils import plot_interesting_slices

class FlyingInvertedPendulumEnv():
	def __init__(self, param_dict=None):
		if param_dict is None:
			# Form a default param dict
			sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
			from create_arg_parser import create_arg_parser
			from problems.quad_pend import create_quad_pend_param_dict

			parser = create_arg_parser()  # default
			args = parser.parse_known_args()[0]
			self.param_dict = create_quad_pend_param_dict(args)  # default
		else:
			self.param_dict = param_dict

		self.__dict__.update(self.param_dict)  # __dict__ holds and object's attributes
		state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
		                     "dtheta", "x", "y", "z", "dx", "dy", "dz"]
		state_index_dict = dict(zip(state_index_names, torch.arange(len(state_index_names))))
		self.i = state_index_dict
		self.dt = 0.00005  # same as cartpole
		# self.dt = 1e-6
		self.g = 9.81

		# self.init_visualization() # TODO: simplifying out viz, for now

	# 	self.control_lim_verts = self.compute_control_lim_vertices()
	#
	# def compute_control_lim_vertices(self):
	# 	k1 = self.k1
	# 	k2 = self.k2
	# 	l = self.l
	#
	# 	M = torch.tensor(
	# 		[[k1, k1, k1, k1], [0, -l * k1, 0, l * k1], [l * k1, 0, -l * k1, 0], [-k2, k2, -k2, k2]])  # mixer matrix
	#
	# 	self.mixer = M
	#
	# 	r1 = torch.cat((torch.zeros(8), torch.ones(8)))
	# 	r2 = torch.cat((torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)))
	# 	r3 = torch.cat(
	# 		(torch.zeros(2), torch.ones(2), torch.zeros(2), torch.ones(2), torch.zeros(2), torch.ones(2), torch.zeros(2), torch.ones(2)))
	# 	r4 = torch.zeros(16)
	# 	r4[1::2] = 1.0
	# 	impulse_vert = torch.cat((r1[None], r2[None], r3[None], r4[None]),
	# 	                              axis=0)  # 16 vertices in the impulse control space
	#
	# 	# print("ulimitsetvertices")
	# 	# IPython.embed()
	#
	# 	force_vert = M @ impulse_vert - torch.tensor(
	# 		[[self.M * self.g], [0.0], [0.0], [0.0]])  # Fixed bug: was subtracting self.M*g (not just in the first row)
	# 	# force_vert = force_vert.T.astype("float32")
	# 	force_vert = force_vert.T
	# 	return force_vert

	def _f(self, x):
		# print("Inside f function")
		# IPython.embed()

		if len(x.shape) == 1:
			x = x[None]  # (1, 16)
		# print("Inside f")
		# IPython.embed()
		bs = x.shape[0]

		gamma = x[:, self.i["gamma"]]
		beta = x[:, self.i["beta"]]
		alpha = x[:, self.i["alpha"]]

		phi = x[:, self.i["phi"]]
		theta = x[:, self.i["theta"]]
		dphi = x[:, self.i["dphi"]]
		dtheta = x[:, self.i["dtheta"]]

		R = torch.zeros((bs, 3, 3))
		R[:, 0, 0] = torch.cos(alpha) * torch.cos(beta)
		R[:, 0, 1] = torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma) - torch.sin(alpha) * torch.cos(gamma)
		R[:, 0, 2] = torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma) + torch.sin(alpha) * torch.sin(gamma)
		R[:, 1, 0] = torch.sin(alpha) * torch.cos(beta)
		R[:, 1, 1] = torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)
		R[:, 1, 2] = torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.cos(alpha) * torch.sin(gamma)
		R[:, 2, 0] = -torch.sin(beta)
		R[:, 2, 1] = torch.cos(beta) * torch.sin(gamma)
		R[:, 2, 2] = torch.cos(beta) * torch.cos(gamma)

		k_x = R[:, 0, 2]
		k_y = R[:, 1, 2]
		k_z = R[:, 2, 2]

		###### Computing state derivatives

		ddphi = (3.0) * (k_y * torch.cos(phi) + k_z * torch.sin(phi)) * (
					self.M * self.g) / (2 * self.M * self.L_p * torch.cos(theta)) + 2 * dtheta * dphi * torch.tan(theta)
		ddtheta = (self.M * self.g)*(3.0 * (
					-k_x * torch.cos(theta) - k_y * torch.sin(phi) * torch.sin(theta) + k_z * torch.cos(phi) * torch.sin(theta)) / (
					           2.0 * self.M * self.L_p)) - torch.square(dphi) * torch.sin(theta) * torch.cos(theta)

		ddx = k_x * self.g
		ddy = k_y * self.g
		ddz = k_z * self.g - self.g

		# Including translational motion
		f = torch.vstack(
			[x[:, self.i["dgamma"]], x[:, self.i["dbeta"]], x[:, self.i["dalpha"]], torch.zeros(bs), torch.zeros(bs),
			 torch.zeros(bs), dphi, dtheta, ddphi, ddtheta, x[:, self.i["dx"]], x[:, self.i["dy"]], x[:, self.i["dz"]],
			 ddx, ddy, ddz]).T
		# f = torch.vstack(
		# 	[x[:, self.i["dgamma"]], x[:, self.i["dbeta"]], x[:, self.i["dalpha"]], torch.zeros(bs), torch.zeros(bs),
		# 	 torch.zeros(bs), dphi, dtheta, ddphi, ddtheta]).T
		return f

	def _g(self, x):
		# print("Inside g function")
		# IPython.embed()

		if len(x.shape) == 1:
			x = x[None]  # (1, 16)
		# print("g: returns matrix")
		# IPython.embed()
		bs = x.shape[0]

		gamma = x[:, self.i["gamma"]]
		beta = x[:, self.i["beta"]]
		alpha = x[:, self.i["alpha"]]

		phi = x[:, self.i["phi"]]
		theta = x[:, self.i["theta"]]

		R = torch.zeros((bs, 3, 3))
		R[:, 0, 0] = torch.cos(alpha) * torch.cos(beta)
		R[:, 0, 1] = torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma) - torch.sin(alpha) * torch.cos(gamma)
		R[:, 0, 2] = torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma) + torch.sin(alpha) * torch.sin(gamma)
		R[:, 1, 0] = torch.sin(alpha) * torch.cos(beta)
		R[:, 1, 1] = torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)
		R[:, 1, 2] = torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.cos(alpha) * torch.sin(gamma)
		R[:, 2, 0] = -torch.sin(beta)
		R[:, 2, 1] = torch.cos(beta) * torch.sin(gamma)
		R[:, 2, 2] = torch.cos(beta) * torch.cos(gamma)

		k_x = R[:, 0, 2]
		k_y = R[:, 1, 2]
		k_z = R[:, 2, 2]

		###### Computing state derivatives
		J_inv = torch.diag(torch.tensor([(1.0 / self.J_x), (1.0 / self.J_y), (1.0 / self.J_z)]))
		dd_drone_angles = R @ J_inv

		# print(J_inv, R)

		ddphi = (3.0) * (k_y * torch.cos(phi) + k_z * torch.sin(phi)) / (2 * self.M * self.L_p * torch.cos(theta))
		ddtheta = (3.0 * (
					-k_x * torch.cos(theta) - k_y * torch.sin(phi) * torch.sin(theta) + k_z * torch.cos(phi) * torch.sin(theta)) / (
					           2.0 * self.M * self.L_p))

		# Including translational motion
		g = torch.zeros((bs, 16, 4))
		g[:, 3:6, 1:] = dd_drone_angles
		g[:, 8, 0] = ddphi
		g[:, 9, 0] = ddtheta
		g[:, 13:, 0] = (1.0 / self.M) * torch.vstack([k_x, k_y, k_z]).T

		print(g)
		return g

	def x_dot_open_loop(self, x, u):
		f = self._f(x)
		g = self._g(x)

		rv = f + g @ u
		return rv

# A = np.array([[0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [-4.9050, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 4.9050, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, -4.9050, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 4.9050,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
#                   [0.0000, 9.8100, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [-9.8100, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
#
# B = np.array([[  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000, 200.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000, 200.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000, 111.1111],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [  1.1905,   0.0000,   0.0000,   0.0000]])

A = np.array([[ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-4.9050,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  4.9050,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000, -4.9050,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  4.9050,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
        [ 0.0000,  9.8100,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-9.8100,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
B = np.array([[  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000, 200.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000, 200.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000, 111.1111],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   0.0000],
        [  1.1905,   0.0000,   0.0000,   0.0000]])


if __name__ == "__main__":
	# pass
	"""param_dict = {
		"m": 0.8,
		"J_x": 0.005,
		"J_y": 0.005,
		"J_z": 0.009,
		"l": 1.5,
		"k1": 4.0,
		"k2": 0.05,
		"m_p": 0.04, # 5% of quad weight
		"L_p": 3.0, # Prev: 0.03
		'delta_safety_limit': math.pi / 4  # should be <= math.pi/4
	}
	param_dict["M"] = param_dict["m"] + param_dict["m_p"]
	state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
						 "dtheta"]  # excluded x, y, z
	state_index_dict = dict(zip(state_index_names, torch.arange(len(state_index_names))))
	param_dict["state_index_dict"] = state_index_dict


	torch.random.seed(3)
	default_env = FlyingInvertedPendulumEnv(param_dict)

	x = torch.random.rand(16)
	u = torch.random.rand(4)
	print(x, u)
	x_dot = default_env.x_dot_open_loop(x, u)

	print(x_dot)"""

	# Testing the batch refactoring
	"""env = FlyingInvertedPendulumEnv()

	# x = torch.random.random((16))
	# torch.random.seed(0)
	x_batch = torch.rand((10, 16))
	f_vals = env._f(x_batch)
	g_vals = env._g(x_batch)

	print(f_vals)
	print(g_vals)

	print("done")
	IPython.embed() #"""

	# env = FlyingInvertedPendulumEnv()
	# x_batch = torch.zeros((1, 16))
	#
	# # x_batch = torch.ones((1, 16))
	# A = torch.squeeze(jacobian(env._f, x_batch))
	# # g_jacobian = jacobian(env._g, x_batch)
	# B = torch.squeeze(env._g(x_batch))
	#
	# print(A)
	# print(B)
	# print("done")
	# # IPython.embed()

	"""
	A = torch.tensor([[ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-4.9050,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  4.9050,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000, -4.9050,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  4.9050,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
        [ 0.0000,  9.8100,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-9.8100,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
	
	B = torch.tensor([[[  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000, 200.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000, 200.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000, 111.1111],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   0.0000],
         [  1.1905,   0.0000,   0.0000,   0.0000]]])

	"""

	# python - u
	# run_flying_pend_exps.py - -save_fnm
	# debug_LQR - -which_cbf
	# ours - -exp_name_to_load
	# quad_pend_ESG_reg_speedup_better_attacks_seed_0 - -checkpoint_number_to_load
	# 250 - -which_experiments
	# rollout - -rollout_u_ref
	# LQR - -rollout_T_max
	# 10 - -rollout_dt
	# 0.1 - -rollout_LQR_q
	# 0.5 - -rollout_N_rollout
	# 1

	exp_name = "quad_pend_ESG_reg_speedup_better_attacks_seed_0"
	rollout_N_rollout = 1
	rollout_T_max = 2.5
	rollout_dt = 0.01

	class dotdict(dict):
		"""dot.notation access to dictionary attributes"""
		__getattr__ = dict.get
		__setattr__ = dict.__setitem__
		__delattr__ = dict.__delitem__

	torch_phi_star_fn, param_dict = load_phi_and_params(exp_name=exp_name,
	                                               checkpoint_number=250)
	numpy_phi_star_fn = PhiNumpy(torch_phi_star_fn)

	save_fldrpth = "./log/%s" % exp_name

	# Experimental settings
	N_desired_rollout = rollout_N_rollout
	T_max = rollout_T_max
	N_steps_max = int(T_max / rollout_dt)
	print("Number of timesteps: %f" % N_steps_max)

	# Create core classes: environment, controller
	env = FlyingInvertedPendulumEnv(param_dict)
	env.dt = rollout_dt
	controller_args = {"rollout_u_ref": "LQR", "rollout_LQR_q":1.0, "rollout_LQR_r":1.0}
	controller_args = dotdict(controller_args)
	cbf_controller = CBFController(env, numpy_phi_star_fn, param_dict, args)  # 2nd arg prev. "cbf_obj"

	x0 = np.random.normal(scale=0.1, size=(1, 16))
	d = simulate_rollout(env, N_steps_max, cbf_controller, x0)
	x = d["x"]
	print(x)

	IPython.embed()
