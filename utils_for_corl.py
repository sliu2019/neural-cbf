import numpy as np
import math
import IPython
import sys, os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from torch.autograd import grad
import torch
from torch.autograd.functional import jacobian

class FlyingInvertedPendulumEnv():
	def __init__(self, param_dict=None):
		if param_dict is None:
			# Form a default param dict
			sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
			from src.argument import create_parser
			from main import create_flying_param_dict

			parser = create_parser()  # default
			args = parser.parse_known_args()[0]
			self.param_dict = create_flying_param_dict(args)  # default
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

		self.control_lim_verts = self.compute_control_lim_vertices()

	def compute_control_lim_vertices(self):
		k1 = self.k1
		k2 = self.k2
		l = self.l

		M = torch.tensor(
			[[k1, k1, k1, k1], [0, -l * k1, 0, l * k1], [l * k1, 0, -l * k1, 0], [-k2, k2, -k2, k2]])  # mixer matrix

		self.mixer = M

		r1 = torch.cat((torch.zeros(8), torch.ones(8)))
		r2 = torch.cat((torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)))
		r3 = torch.cat(
			(torch.zeros(2), torch.ones(2), torch.zeros(2), torch.ones(2), torch.zeros(2), torch.ones(2), torch.zeros(2), torch.ones(2)))
		r4 = torch.zeros(16)
		r4[1::2] = 1.0
		impulse_vert = torch.cat((r1[None], r2[None], r3[None], r4[None]),
		                              axis=0)  # 16 vertices in the impulse control space

		# print("ulimitsetvertices")
		# IPython.embed()

		force_vert = M @ impulse_vert - torch.tensor(
			[[self.M * self.g], [0.0], [0.0], [0.0]])  # Fixed bug: was subtracting self.M*g (not just in the first row)
		# force_vert = force_vert.T.astype("float32")
		force_vert = force_vert.T
		return force_vert

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

		ddphi = (3.0) * (k_y * torch.cos(phi) + k_z * torch.sin(phi)) / (2 * self.M * self.L_p * torch.cos(theta)) * (
					self.M * self.g) + 2 * dtheta * dphi * torch.tan(theta)
		ddtheta = (3.0 * (
					-k_x * torch.cos(theta) - k_y * torch.sin(phi) * torch.sin(theta) + k_z * torch.cos(phi) * torch.sin(theta)) / (
					           2.0 * self.M * self.L_p)) * (self.M * self.g) - torch.square(dphi) * torch.sin(theta) * torch.cos(
			theta)

		ddx = k_x * self.g
		ddy = k_y * self.g
		ddz = k_z * self.g - self.g

		# Including translational motion
		f = torch.vstack(
			[x[:, self.i["dgamma"]], x[:, self.i["dbeta"]], x[:, self.i["dalpha"]], torch.zeros(bs), torch.zeros(bs),
			 torch.zeros(bs), dphi, dtheta, ddphi, ddtheta, x[:, self.i["dx"]], x[:, self.i["dy"]], x[:, self.i["dz"]],
			 ddx, ddy, ddz]).T
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

		# print(g)
		return g

	def x_dot_open_loop(self, x, u):
		f = self._f(x)
		g = self._g(x)

		rv = f + g @ u
		return rv

A = torch.tensor([[0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [-4.9050, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 4.9050, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, -4.9050, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 4.9050,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                  [0.0000, 9.8100, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [-9.8100, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])

B = torch.tensor([[[0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 200.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 200.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 111.1111],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000, 0.0000],
                   [1.1905, 0.0000, 0.0000, 0.0000]]])


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
	# print("done")
	# IPython.embed()
	#
	# A = torch.squeeze(jacobian(env._f, x_batch))
	# # g_jacobian = jacobian(env._g, x_batch)
	# B = env._g(x_batch)

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

