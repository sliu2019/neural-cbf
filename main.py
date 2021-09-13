import torch
import numpy as np

from torch import nn
from torch.autograd import grad
from attacks.basic import BasicAttacker
from src.trainer import Trainer
from src.utils import *
from scipy.linalg import pascal
from src.argument import parser, print_args
import os, sys

class Phi(nn.Module):
	# TODO: get phi_i for all i from here. Use forward hooks
	# Note: currently, we have a implementation which is generic to any r. May be slow

	def __init__(self, h_fn, xdot_fn, r, x_dim, args):
		# Later: args specifying how beta is parametrized
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		assert r>=0
		self.eps = self.args.phi_eps

		self.ci = nn.Parameter(torch.randint(100, r)) # int from 0-100
		self.register_parameter("ci_tensor", self.ci)

		self.beta_net = nn.Sequential(
			nn.Linear(x_dim, 2*x_dim),
			nn.ReLU(),
			nn.Linear(2*x_dim, 1)
		)

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		beta_value = nn.Softplus(self.h_fn(x)) - torch.log(2) + nn.Softplus(self.beta_net(x))

		# Convert ci to ki
		ki = torch.tensor([[1]])
		for i in range(self.r): # A is current coeffs
			A = torch.zeros(torch.numel(ki)+1, 2)
			A[:-1, 0] = ki # copy?
			A[1:, 1] = ki

			ki = A.mm(torch.tensor([[1], [self.ci[i]]]))

		# Compute higher-order derivatives of h via finite differencing (sp. forward difference formulation)
		pasc = pascal(self.r+1, kind='lower') # r+1 because of 0 term
		alt_sign_vec = np.power(-np.ones_like(np.arange(self.r+1)), np.arange(self.r+1))[:, None]
		alt_sign_matrix = alt_sign_vec@alt_sign_vec
		coeff_matrix = pasc*alt_sign_matrix # rxr
		coeff_matrix = np.diag(np.power((1/self.eps)*self.ones(self.r+1), self.arange(self.r+1)))@coeff_matrix
		coeff_matrix = torch.tensor(coeff_matrix)

		# print("forward of phi in main.py")
		# print("check out coeff_matrix")
		# print(coeff_matrix)
		# IPython.embed()

		# Batched x?
		inputs = torch.tile(x.unsqueeze(1), (1, self.r+1, 1)) + (self.eps*torch.arange(self.r+1)).unsqueeze(0).unsqueeze(2) # (r, x_dim)
		inputs = torch.reshape(inputs, (-1, self.x_dim))
		outputs = self.h_fn(inputs)
		outputs = torch.reshape(outputs, (-1, 1, self.r+1))

		h_derivs = outputs.mm(coeff_matrix.t())

		result = beta_value + (h_derivs.mm(ki)).squeeze()

		return result

class H(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		# TODO (toy): implement
		# The way these are implemented should be batch compliant
		return None

class XDot(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, u):
		# TODO (toy): implement
		# The way these are implemented should be batch compliant
		return None

class ULimitSetVertices(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		# TODO (toy): implement
		# The way these are implemented should be batch compliant
		return None

class Objective(nn.Module):
	def __init__(self, phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim):
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def forward(self, x):
		# The way these are implemented should be batch compliant
		u_lim_set_vertices = self.uvertices_fn(x) # (bs, n_vertices, u_dim), can be a function of x_batch
		n_vertices = u_lim_set_vertices.size()[1]

		U = torch.reshape(u_lim_set_vertices, (-1, self.u_dim)) # (bs x n_vertices, u_dim)
		X = torch.tile(x.unsqueeze(1), (1, n_vertices, 1)) # (bs, n_vertices, x_dim)
		X = torch.reshape(X, (-1, self.x_dim)) # (bs x n_vertices, x_dim)

		xdot = self.xdot_fn(X, U)

		grad_phi = grad([self.phi_fn], x, create_graph=True)[0] # check
		grad_phi = torch.tile(grad_phi.unsqueeze(1), (1, n_vertices, 1))
		grad_phi = torch.reshape(grad_phi, (-1, self.x_dim))

		phidot_cand = xdot.unsqueeze(1).mm(grad_phi.unsqueeze(2))
		phidot_cand = torch.reshape(phidot_cand, (-1, n_vertices))

		phidot = torch.min(phidot_cand, 1)
		result = nn.ReLU(phidot)
		return result

def main(args):
	# Boilerplate for saving
	save_folder = '%s_%s' % (args.dataset, args.affix)

	log_folder = os.path.join(args.log_root, save_folder)
	model_folder = os.path.join(args.model_root, save_folder)

	makedirs(log_folder)
	makedirs(model_folder)

	setattr(args, 'log_folder', log_folder)
	setattr(args, 'model_folder', model_folder)

	logger = create_logger(log_folder, 'train', 'info')
	print_args(args, logger)

	args_savepth = os.join(log_folder, "args.txt")
	save_args(args, args_savepth)

	# Selecting problem
	if args.problem == "toy":
		r = 2
		x_dim = 14
		u_dim = 4
		x_lim = np.zeros((x_dim, 2))

		# Create phi
		h_fn = H()
		xdot_fn = XDot()
		uvertices_fn = ULimitSetVertices()
	else:
		raise NotImplementedError


	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, args)

	# Create objective function
	objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim)

	# Create attacker
	attacker = BasicAttacker(x_lim)

	# Create test attacker
	test_attacker = BasicAttacker(x_lim, stop_condition="threshold")

	# Pass everything to Trainer
	trainer = Trainer(args, logger, attacker, test_attacker)
	trainer.train(objective_fn, phi_fn, xdot_fn)

if __name__ == "__main__":
	args = parser()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	main(args)
