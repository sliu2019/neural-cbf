import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
from attacks.basic import BasicAttacker

class Phi(nn.Module):
	# TODO: get phi_i for all i from here. Use forward hooks

	def __init__(self, h_fn, xdot_fn, r, x_dim):
		# Later: args specifying how beta is parametrized
		super().__init__()
		self.h_fn = h_fn
		self.xdot_fn = xdot_fn
		self.r = r
		self.x_dim=x_dim

		assert r>=0
		self.ci = nn.Parameter(torch.randint(100, r)) # int from 0-100

		self.beta_net = nn.Sequential(
			nn.Linear(x_dim, 2*x_dim),
			nn.ReLU(),
			nn.Linear(2*x_dim, 1)
		)

	def forward(self, x):
		beta_value = nn.Softplus(self.h_fn(x)) - torch.log(2) + nn.Softplus(self.beta_net(x))

		# Polynomial multiplication as matrix multiplication
		ki = torch.tensor([[1]])
		for i in range(self.r):
			A = torch.zeros(torch.numel(ki)+1, 2)
			A[:-1, 0] = ki # copy?
			A[1:, 1] = ki

			ki = A.mm(torch.tensor([[1], [self.ci[i]]]))

		# Gather higher-order derivatives of h
		h = self.h_fn
		for i in range(self.r):
			grads = grad(h, x, create_graph=True)[0]
			h = grads.sum()

			if i == 0:
				h_gradients = grads.unsqueeze(0).clone() # ?
			else:
				h_gradients = torch.cat((h_gradients, grad.unsqueeze(0)), axis=0)

		result = beta_value + self.h_fn(x)
		if self.r:
			xdot = self.xdot_fn(x)
			result += ki.t().mm(h_gradients.mm(xdot))

		return result

class H(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		# TODO (toy): implement
		# The way these are implemented should be batch compliant
		return x[0]

class XDot(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, u):
		# TODO (toy): implement
		# The way these are implemented should be batch compliant
		return None

class Objective(nn.Module):
	def __init__(self, phi_fn, xdot_fn):
		super().__init__()
		self.phi_fn = phi_fn
		self.xdot_fn = xdot_fn

	# Currently, hard-coding minimization of u_lim vertices here.
	def forward(self, x):
		# TODO (toy): implement
		u_lim_set_vertices = torch.zeros((69, 4)) # (n_vertices, u_dim)
		x_batch = torch.tile(x.unsqueeze(0), (u_lim_set_vertices.size()[0], 1))
		xdot_batch = self.xdot_fn(x_batch, u_lim_set_vertices)

		grad_phi = grad(self.phi_fn, x, create_graph=True)[0] # [0] correct here?
		phidot_batch = xdot_batch.mm(grad_phi)
		result = nn.ReLU(torch.min(phidot_batch))

		return result

def main():
	# Define some parameters
	# TODO (toy): implement
	r = 2
	x_dim = 14
	u_dim = 4
	x_lim = np.zeros((x_dim, 2))

	# Define state space dict: is this necessary?

	# Create phi
	h_fn = H()
	xdot_fn = XDot()

	phi_fn = Phi(h_fn, xdot_fn, r, x_dim)

	# Create objective function
	objective_fn = Objective(phi_fn, xdot_fn)

	# Create attacker
	attacker = BasicAttacker(x_lim)

	# Pass everything to Trainer 


if __name__ == "__main__":
	main()
