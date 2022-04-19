from cmath import cos
import numpy as np
import math
from cvxopt import matrix, solvers
import torch
from torch.autograd import grad
from torch import nn
import IPython

"""
File contains torch module that implements the low-dimensional CBF
This can be wrapped with a numpy class
It can also be used within our training algorithm  
"""

# Noe: class is agnostic to r value, but assume it's r = 2
# Note: in our reshaping, we'll also assume the form of h (although the original class is agnostic to h)
class PhiLow(nn.Module):
	def __init__(self, h_fn, xdot_fn, r, x_dim, u_dim, device, args, param_dict):
		"""
		:param h_fn:
		:param xdot_fn:
		:param r:
		:param x_dim:
		:param u_dim:
		:param device:
		:param args:
		:param nn_input_modifier:
		:param x_e:
		"""
		# Later: args specifying how beta is parametrized
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		assert r >= 0

		# turn Namespace into dict
		args_dict = vars(args)
		self.delta_max = param_dict['delta_safety_limit']
		self.state_index_dict = self.param_dict["state_index_dict"]

		# Initialize params
		# Q: do I need to pad any away from 0 here? No, you can just specify the lower bound within CMA-ES
		self.ci = nn.Parameter(torch.rand(2, 1)) # reshaping parameters
		self.ki = nn.Parameter(torch.rand(r-1, 1)) # coeffs for higher order terms

	def forward(self, x, grad_x=False):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		# RV is (bs, r+1)

		# Compute higher-order Lie derivatives
		#####################################################################
		# Turn gradient tracking on for x
		bs = x.size()[0]
		if grad_x == False:
			orig_req_grad_setting = x.requires_grad  # Basically only useful if x.requires_grad was False before
			x.requires_grad = True

		h_ith_deriv = self.h_fn(x)  # bs x 1, the zeroth derivative

		h_derivs = h_ith_deriv  # bs x 1
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device))  # bs x x_dim

		for i in range(self.r - 1):
			grad_h_ith = grad([torch.sum(h_ith_deriv)], x, create_graph=True)[
				0]  # bs x x_dim; create_graph ensures gradient is computed through the gradient operation
			h_ith_deriv = (grad_h_ith.unsqueeze(dim=1)).bmm(f_val.unsqueeze(dim=2))  # bs x 1 x 1
			h_ith_deriv = h_ith_deriv[:, :, 0]  # bs x 1
			h_derivs = torch.cat((h_derivs, h_ith_deriv), dim=1)

		if grad_x == False:
			x.requires_grad = orig_req_grad_setting
		#####################################################################
		# Turn gradient tracking off for x
		result = h_derivs.mm(self.ki)

		##############################################
		##### Compute new_h (reshaped)
		theta = x[:, [self.state_index_dict["theta"]]]
		phi = x[:, [self.state_index_dict["phi"]]]
		gamma = x[:, [self.state_index_dict["gamma"]]]
		beta = x[:, [self.state_index_dict["beta"]]]

		cos_cos = torch.cos(theta)*torch.cos(phi)
		eps = 1e-4 # prevents nan when cos_cos = +/- 1 (at x = 0)
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos)*eps
		delta = torch.acos(cos_cos + signed_eps)
		new_h = (delta**2 + gamma**2 + beta**2)**(self.ci[0]) - (self.delta_max**2)**(self.ci[0]) + self.ci[1]

		phi_r_minus_1_star = result[:, [-1]] - result[:, [0]] + new_h

		result = torch.cat((result, phi_r_minus_1_star), dim=1)

		return result


if __name__ == "__main__":
	pass
