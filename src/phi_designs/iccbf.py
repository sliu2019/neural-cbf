import torch
from torch import nn
from torch.autograd import grad
import IPython
import math
import numpy as np
import os

class ICCBF(nn.Module):
	def __init__(self, h_fn, xdot_fn, uvertices_fn, class_kappa_fns, x_dim, u_dim, device):
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		self.N = len(class_kappa_fns) # the number of "iterations" of the CBF; previously equal to degree

	def forward(self, x, grad_x=False):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		# RV is (bs, r+1)

		# print("inside forward")
		# IPython.embed()
		# bs = x.size()[0]

		if grad_x == False:
			orig_req_grad_setting = x.requires_grad # Basically only useful if x.requires_grad was False before
			x.requires_grad = True

		hi = self.h_fn(x) # bs x 1, the zeroth derivative

		hi_list = [hi] # bs x 1
		# f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device)) # bs x x_dim
		u_lim_set_vertices = self.uvertices_fn(x)  # (bs, n_vertices, u_dim), can be a function of x_batch
		n_vertices = u_lim_set_vertices.size()[1]
		U = torch.reshape(u_lim_set_vertices, (-1, self.u_dim))  # (bs x n_vertices, u_dim)

		X = (x.unsqueeze(1)).repeat(1, n_vertices, 1)  # (bs, n_vertices, x_dim)
		X = torch.reshape(X, (-1, self.x_dim))  # (bs x n_vertices, x_dim)

		# Evaluate every X against multiple U
		xdot = self.xdot_fn(X, U)

		# IPython.embed()
		for i in range(self.N): # N+1: just how it's defined in paper
			# phi_value = self.phi_fn(x)
			# grad_phi = grad([torch.sum(phi_value[:, -1])], x, create_graph=True)[0]  # check
			grad_hi = grad([torch.sum(hi)], x, create_graph=True)[0] # TODO: check

			grad_hi = (grad_hi.unsqueeze(1)).repeat(1, n_vertices, 1)
			grad_hi = torch.reshape(grad_hi, (-1, self.x_dim))

			# Dot product
			dot_hi_cand = xdot.unsqueeze(1).bmm(grad_hi.unsqueeze(2))
			dot_hi_cand = torch.reshape(dot_hi_cand, (-1, n_vertices))  # bs x n_vertices

			dot_hi, _ = torch.min(dot_hi_cand, 1)

			hiplus1 = dot_hi[:, None] + self.class_kappa_fns[i](hi)
			hiplus1 = hiplus1.view(-1, 1)

			hi_list.append(hiplus1)
			hi = hiplus1

		# IPython.embed()
		if grad_x == False:
			x.requires_grad = orig_req_grad_setting

		result = torch.cat(hi_list, dim=1)
		return result


class KappaPolynomial(nn.Module):
	# Note: this is specific to FlyingInvPend
	def __init__(self, coefficient, exponent):
		"""
		:param which_ind: flat numpy array
		"""
		super().__init__()

		assert coefficient > 0
		assert exponent > 0
		self.coefficient = coefficient
		self.exponent = exponent

	def forward(self, x):
		rv = self.coefficient*torch.pow(x, self.exponent)
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
		"m_p": 0.04, # 5% of quad weight
		"L_p": 3.0, # Prev: 0.03
		'delta_safety_limit': math.pi / 4  # should be <= math.pi/4
	}
	param_dict["M"] = param_dict["m"] + param_dict["m_p"]
	state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
	                     "dtheta"]  # excluded x, y, z
	state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))

	r = 2
	x_dim = len(state_index_names)
	u_dim = 4
	ub = 15
	thresh = np.array([math.pi / 3, math.pi / 3, math.pi, ub, ub, ub, math.pi / 3, math.pi / 3, ub, ub],
	                  dtype=np.float32) # angular velocities bounds probably much higher in reality (~10-20 for drone, which can do 3 flips in 1 sec).

	x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)  # (13, 2)

	# Save stuff in param dict
	param_dict["state_index_dict"] = state_index_dict
	param_dict["r"] = r
	param_dict["x_dim"] = x_dim
	param_dict["u_dim"] = u_dim
	param_dict["x_lim"] = x_lim

	# write args into the param_dict
	# Device
	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
		dev = "cuda:%i" % (0)
	else:
		dev = "cpu"
	device = torch.device(dev)

	from src.problems.flying_inv_pend import HSum, XDot, ULimitSetVertices

	h_fn = HSum(param_dict)

	xdot_fn = XDot(param_dict, device)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	# Class kappa functions
	kappa1 = KappaPolynomial(2, 1)
	kappa2 = KappaPolynomial(7, 0.5)
	kappa3 = KappaPolynomial(3, 2)

	class_kappa_fns = [kappa1, kappa2, kappa3]

	# print("before initializing class")
	# IPython.embed()
	iccbf_fn = ICCBF(h_fn, xdot_fn, uvertices_fn, class_kappa_fns, x_dim, u_dim, device)

	# Forward
	# IPython.embed()
	bs = 5
	x = torch.rand(bs, 10).to(device)
	x.requires_grad = True
	phi_value = iccbf_fn(x)

	# Test backward pass
	# IPython.embed()
	grad_phi = grad([torch.sum(phi_value[:, -1])], x, create_graph=True)[0]  # check


