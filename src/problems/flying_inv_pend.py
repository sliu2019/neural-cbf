import torch
import numpy as np

from torch import nn
import os, sys
import IPython
import math

g = 9.81
class HMax(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Return value is size (bs, 1)

		# print("Inside HMax forward")
		# IPython.embed()
		theta = x[:, [self.i["theta"]]]
		phi = x[:, [self.i["phi"]]]
		gamma = x[:, [self.i["gamma"]]]
		beta = x[:, [self.i["beta"]]]

		cos_cos = torch.cos(theta)*torch.cos(phi)
		eps = 1e-4 # prevents nan when cos_cos = +/- 1 (at x = 0)
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos)*eps
		delta = torch.acos(cos_cos + signed_eps)
		rv = torch.maximum(torch.maximum(delta**2, gamma**2), beta**2) - self.delta_safety_limit**2
		return rv

class HSum(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Return value is size (bs, 1)

		# print("Inside HSum forward")
		# IPython.embed()
		theta = x[:, [self.i["theta"]]]
		phi = x[:, [self.i["phi"]]]
		gamma = x[:, [self.i["gamma"]]]
		beta = x[:, [self.i["beta"]]]

		cos_cos = torch.cos(theta)*torch.cos(phi)
		eps = 1e-4 # prevents nan when cos_cos = +/- 1 (at x = 0)
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos)*eps
		delta = torch.acos(cos_cos + signed_eps)
		rv = delta**2 + gamma**2 + beta**2 - self.delta_safety_limit**2

		return rv

class XDot(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device
		self.i = self.state_index_dict

	def forward(self, x, u):
		# x: bs x 10, u: bs x 4
		# The way these are implemented should be batch compliant

		# Pre-computations
		# Compute the rotation matrix from quad to global frame
		# Extract the k_{x,y,z}
		gamma = x[:, self.i["gamma"]]
		beta = x[:, self.i["beta"]]
		alpha = x[:, self.i["alpha"]]

		phi = x[:, self.i["phi"]]
		theta = x[:, self.i["theta"]]
		dphi = x[:, self.i["dphi"]]
		dtheta = x[:, self.i["dtheta"]]

		R = torch.zeros((x.shape[0], 3, 3), device=self.device) # is this the correct rotation?
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

		F = (u[:, 0] + self.M*g)

		###### Computing state derivatives
		# IPython.embed()
		J = torch.tensor([self.J_x, self.J_y, self.J_z]).to(self.device)
		norm_torques = u[:, 1:]*(1.0/J)

		ddquad_angles = torch.bmm(R, norm_torques[:, :, None]) # (N, 3, 1)
		ddquad_angles = ddquad_angles[:, :, 0]
		# ddgamma = (1.0/self.J_x)*ddquad_angles[:, 0]
		# ddbeta = (1.0/self.J_y)*ddquad_angles[:, 1]
		# ddalpha = (1.0/self.J_z)*ddquad_angles[:, 2]

		ddgamma = ddquad_angles[:, 0]
		ddbeta = ddquad_angles[:, 1]
		ddalpha = ddquad_angles[:, 2]

		ddphi = (3.0)*(k_y*torch.cos(phi) + k_z*torch.sin(phi))/(2*self.M*self.L_p*torch.cos(theta))*F + 2*dtheta*dphi*torch.tan(theta)
		ddtheta = (3.0*(-k_x*torch.cos(theta)-k_y*torch.sin(phi)*torch.sin(theta) + k_z*torch.cos(phi)*torch.sin(theta))/(2.0*self.M*self.L_p))*F - torch.square(dphi)*torch.sin(theta)*torch.cos(theta)

		# Excluding translational motion
		rv = torch.cat([x[:, [self.i["dgamma"]]], x[:, [self.i["dbeta"]]], x[:, [self.i["dalpha"]]], ddgamma[:, None], ddbeta[:, None], ddalpha[:, None], dphi[:, None], dtheta[:, None], ddphi[:, None], ddtheta[:, None]], axis=1)
		return rv

class ULimitSetVertices(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device

		# precompute, just tile it in forward()
		k1 = self.k1
		k2 = self.k2
		l = self.l
		M = np.array([[k1, k1, k1, k1], [0, -l*k1, 0, l*k1], [l*k1, 0, -l*k1, 0], [-k2, k2, -k2, k2]]) # mixer matrix

		r1 = np.concatenate((np.zeros(8), np.ones(8)))
		r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
		r3 = np.concatenate((np.zeros(2), np.ones(2),np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),np.zeros(2), np.ones(2)))
		r4 = np.zeros(16)
		r4[1::2] = 1.0
		impulse_vert = np.concatenate((r1[None], r2[None], r3[None], r4[None]), axis=0) # 16 vertices in the impulse control space

		# print("ulimitsetvertices")
		# IPython.embed()

		force_vert = M@impulse_vert - np.array([[self.M*g], [0.0], [0.0], [0.0]]) # Fixed bug: was subtracting self.M*g (not just in the first row)
		force_vert = force_vert.T.astype("float32")
		self.vert = torch.from_numpy(force_vert).to(self.device)

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# (bs, n_vertices, u_dim) or (bs, 16, 4)

		rv = self.vert
		rv = rv.unsqueeze(dim=0)
		rv = rv.expand(x.shape[0], -1, -1)
		return rv

if __name__ == "__main__":
	# param_dict = {
	# 	"m": 0.8,
	# 	"J_x": 0.005,
	# 	"J_y": 0.005,
	# 	"J_z": 0.009,
	# 	"l": 1.5,
	# 	"k1": 4.0,
	# 	"k2": 0.05,
	# 	"m_p": 0.04, # TODO?
	# 	"L_p": 0.03, # TODO?
	# 	'delta_safety_limit': math.pi/5 # in radians; should be <= math.pi/4
	# }
	# param_dict["M"] = param_dict["m"] + param_dict["m_p"]
	#
	# state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta"] # excluded x, y, z
	# state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
	# param_dict["state_index_dict"] = state_index_dict
	# ##############################################
	#
	# if torch.cuda.is_available():
	# 	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	# 	dev = "cuda:%i" % (0)
	# 	print("Using GPU device: %s" % dev)
	# else:
	# 	dev = "cpu"
	# device = torch.device(dev)
	#
	# # h_fn = H(param_dict)
	# xdot_fn = XDot(param_dict, device)
	# uvertices_fn = ULimitSetVertices(param_dict, device)
	#
	# N = 10
	# x = torch.rand(N, 10).to(device)
	# u = torch.rand(N, 4).to(device)
	#
	# uvert = uvertices_fn(x)

	# IPython.embed()
	# h_fn = HMax(param_dict)
	# h_fn = HSum(param_dict)
	# rv1 = h_fn(x)
	# print(rv1.shape)

	# x = torch.zeros(1, 10).to(device)
	# u = torch.zeros(1, 4).to(device)
	# rv2 = xdot_fn(x, u)
	# IPython.embed()
	# rv3 = uvertices_fn(x)

	# print(rv2.shape)
	# print(rv3.shape)
	# print(rv1.shape)
	# IPython.embed()

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
	state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
	param_dict["state_index_dict"] = state_index_dict

	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'
		dev = "cuda:%i" % (0)
		print("Using GPU device: %s" % dev)
	else:
		dev = "cpu"
	device = torch.device(dev)

	xdot_fn = XDot(param_dict, device)

	np.random.seed(3)
	x = np.random.rand(16)
	u = np.random.rand(4)

	x = x[:10]
	x = torch.from_numpy(x.astype("float32")).to(device)
	u = torch.from_numpy(u.astype("float32")).to(device)

	x = x.view(1, -1)
	u = u.view(1, -1)

	# N = 10
	# x = torch.rand(N, 10).to(device)
	# u = torch.rand(N, 4).to(device)

	xdot = xdot_fn(x, u)

	print(x, u)
	print(xdot)"""
	pass

	# comparing results to batched numpy

