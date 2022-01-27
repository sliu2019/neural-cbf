import torch
import numpy as np

from torch import nn
import os, sys
import IPython
import math

g = 9.81
class H(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes

	def forward(self, x):
		# TODO (toy): implement
		# The way these are implemented should be batch compliant

		theta = x[:, 12]
		phi = x[:, 13]
		rv = torch.acos(torch.cos(theta)*torch.sin(phi)) - self.delta_safety_limit # TODO: torch.acos output is in [0, 2pi]?

		return rv

class XDot(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		# IPython.embed()

	def forward(self, x, u):
		# x: bs x 16, u: bs x 4
		# The way these are implemented should be batch compliant

		# Pre-computations
		# Compute the rotation matrix from quad to global frame
		# Extract the k_{x,y,z}
		# IPython.embed()
		gamma = x[:, 6]
		beta = x[:, 7]
		alpha = x[:, 8]

		phi = x[:, 12]
		theta = x[:, 13]
		dot_phi = x[:, 14]
		dot_theta = x[:, 15]

		R = torch.zeros((x.shape[0], 3, 3)) # TODO: is this the correct rotation?
		R[:, 0, 0] = torch.cos(alpha)*torch.cos(beta)
		R[:, 0, 1] = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma)
		R[:, 0, 2] = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)
		R[:, 1, 0] = torch.sin(alpha)*torch.cos(beta)
		R[:, 1, 1] = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma)
		R[:, 1, 2] = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
		R[:, 2, 0] = -torch.sin(beta)
		R[:, 2, 1] = torch.cos(beta)*torch.sin(gamma)
		R[:, 2, 2] = torch.cos(beta)*torch.cos(gamma)

		k_x = R[:, 0, 2]
		k_y = R[:, 1, 2]
		k_z = R[:, 2, 2]

		F = u[:, 0]
		# tau_gamma = u[:, 1]
		# tau_beta = u[:, 2]
		# tau_alpha = u[:, 3]
		###### Computing state derivatives
		ddotx = (1.0/self.M)*k_x*F
		ddoty = (1.0/self.M)*k_y*F
		ddotz = (1.0/self.M)*k_z*F - g

		ddot_quad_angles = torch.bmm(R, u[:, 1:, None]) # (N, 3, 1)
		ddot_quad_angles = ddot_quad_angles[:, :, 0]
		ddot_gamma = (1.0/self.J_x)*ddot_quad_angles[:, 0]
		ddot_beta = (1.0/self.J_y)*ddot_quad_angles[:, 1]
		ddot_alpha = (1.0/self.J_z)*ddot_quad_angles[:, 2]

		ddot_phi = (3.0)*(k_y*torch.cos(phi) + k_z*torch.sin(phi))/(2*self.M*self.L_p*torch.cos(theta))*F + 2*dot_theta*dot_phi*torch.tan(theta)
		ddot_theta = (3.0*(-k_x*torch.cos(theta)-k_y*torch.sin(phi)*torch.sin(theta) + k_z*torch.cos(phi)*torch.sin(theta))/(2.0*self.M*self.L_p))*F - torch.square(dot_phi)*torch.sin(theta)*torch.cos(theta)

		rv = torch.cat([x[:, [3]], x[:, [4]], x[:, [5]], ddotx[:, None], ddoty[:, None], ddotz[:, None], x[:, [9]], x[:, [10]], x[:, [11]], ddot_gamma[:, None], ddot_beta[:, None], ddot_alpha[:, None], dot_phi[:, None], dot_theta[:, None], ddot_phi[:, None], ddot_theta[:, None]], axis=1)
		return rv

class ULimitSetVertices(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device

		# IPython.embed()
		# precompute, just tile it in forward
		k1 = self.k1
		k2 = self.k2
		l = self.l
		M = np.array([[k1, k1, k1, k1], [0, -l*k1, 0, l*k1], [l*k1, 0, -l*k1, 0], [-k2, k2, -k2, k2]]) # mixer matrix

		r1 = np.concatenate((np.zeros(8), np.ones(8)))
		r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
		r3 = np.concatenate((np.zeros(2), np.ones(2),np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),np.zeros(2), np.ones(2)))
		r4 = np.zeros(16)
		r4[1::2] = 1.0
		impulse_vert = np.concatenate((r1[None], r2[None], r3[None], r4[None]), axis=0) # 8 vertices in the impulse control space

		force_vert = M@impulse_vert
		self.vert = torch.from_numpy(force_vert.T).to(self.device)

	def forward(self, x):
		# TODO (toy): implement
		# The way these are implemented should be batch compliant
		# (bs, n_vertices, u_dim) or (bs, 16, 4)

		# rv = torch.tensor([[self.max_force], [-self.max_force]]).to(self.device)
		# rv = rv.unsqueeze(dim=0)
		# rv = rv.expand(x.size()[0], -1, -1)

		# IPython.embed()
		rv = self.vert
		rv = rv.unsqueeze(dim=0)
		rv = rv.expand(x.shape[0], -1, -1)
		return rv

if __name__ == "__main__":
	param_dict = {
		"m": 0.8,
		"J_x": 0.005,
		"J_y": 0.005,
		"J_z": 0.009,
		"l": 1.5,
		"k1": 4.0,
		"k2": 0.05,
		"m_p": 0.04,
		"L_p": 0.03, # TODO?
		'delta_safety_limit': math.pi/5 # in radians; should be <= math.pi/4
	}
	param_dict["M"] = param_dict["m"] + param_dict["m_p"]

	##############################################

	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'
		dev = "cuda:%i" % (0)
		print("Using GPU device: %s" % dev)
	else:
		dev = "cpu"
	device = torch.device(dev)

	# h_fn = H(param_dict)
	# xdot_fn = XDot(param_dict)
	# uvertices_fn = ULimitSetVertices(param_dict, device)
	# TODO: test by forward and backward prop

	# N = 25
	# x = torch.rand(N, 16)
	# u = torch.rand(N, 4)

	# IPython.embed()
	# rv = h_fn(x)
	# rv = xdot_fn(x, u)
	# rv = uvertices_fn(x)

	# IPython.embed()