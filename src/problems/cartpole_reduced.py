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
		rv = torch.abs(x[:, [1]]) - self.max_theta # bs x 1
		return rv

class XDot(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		# IPython.embed()

	def forward(self, x, u):
		# x: bs x 2, u: bs x 1
		# The way these are implemented should be batch compliant
		theta = x[:, [0]]
		thetadot = x[:, [1]]

		# xddot_num = (self.I +self.m*(self.l**2))*(self.m*self.l*(thetadot**2)*torch.sin(theta)) - g*(self.m**2)*(self.l**2)*torch.sin(theta)*torch.cos(theta) + (self.I + self.m*(self.l**2))*u
		denom = self.I*(self.m + self.M) + self.m*(self.l**2)*(self.M + self.m*(torch.sin(theta)**2))
		# IPython.embed()
		thetaddot_num = self.m*self.l*(-self.m*self.l*(thetadot**2)*torch.sin(theta)*torch.cos(theta) + (self.M + self.m)*g*torch.sin(theta)) - self.m*self.l*torch.cos(theta)*u

		# print(xddot_num.size(), denom.size())
		# xddot = xddot_num/denom
		thetaddot = thetaddot_num/denom
		# print(xddot.size(), thetaddot.size())
		# rv = torch.cat((x[:, [2]], x[:, [3]], xddot, thetaddot), dim=1)
		rv = torch.cat((thetadot, thetaddot), dim=1)
		return rv

class ULimitSetVertices(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device

	def forward(self, x):
		# TODO (toy): implement
		# The way these are implemented should be batch compliant
		# rv is [2, 1]
		# (bs, n_vertices, u_dim) or (bs, 2, 1)
		rv = torch.tensor([[self.max_force], [-self.max_force]]).to(self.device)
		rv = rv.unsqueeze(dim=0)
		rv = rv.expand(x.size()[0], -1, -1)
		return rv

if __name__ == "__main__":
	# Params from p. 19 of http://ethesis.nitrkl.ac.in/6302/1/E-64.pdf
	param_dict = {
		"I": 0.099,
		"m": 0.2,
		"M": 2,
		"l": 0.5,
		"max_theta": math.pi/10.0,
		"max_force": 1.0
	}

	xdot_fn = XDot(param_dict)

	# Alternately
	param_dict = {
		"I": 0.006,
		"m": 0.2,
		"M": 0.5,
		"l": 0.3,
		"max_theta": math.pi/10.0,
		"max_force": 1.0
	}

	# Alternately alternately, from the blogpost
	param_dict = {
		"I": 0.021,
		"m": 0.25,
		"M": 1.00,
		"l": 0.5,
		"max_theta": math.pi/4.0,
		"max_force": 5.0
	}
