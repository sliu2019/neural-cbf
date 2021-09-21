import torch
import numpy as np

from torch import nn
from torch.autograd import grad
from src.attacks.basic_attacker import BasicAttacker
from src.trainer import Trainer
from src.utils import *
from scipy.linalg import pascal
from src.argument import parser, print_args
import os, sys
import math
import IPython

class Phi(nn.Module):
	# TODO: get phi_i for all i from here. Use forward hooks
	# Note: currently, we have a implementation which is generic to any r. May be slow

	def __init__(self, h_fn, xdot_fn, r, x_dim, u_dim, args):
		# Later: args specifying how beta is parametrized
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		assert r>=0
		# Note: by default, it registers parameters by their variable name
		self.ci = nn.Parameter(5*torch.rand(r-1, 1)) # int from 0-5 (if ci in small range, ki will be much larger)
		self.beta_net = nn.Sequential(
			nn.Linear(x_dim, 2*x_dim),
			nn.ReLU(),
			nn.Linear(2*x_dim, 1)
		)

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		beta_value = nn.functional.softplus(self.h_fn(x)) - torch.log(torch.tensor(2.0)) + nn.functional.softplus(self.beta_net(x))

		# Convert ci to ki
		ki = torch.tensor([[1.0]])
		for i in range(self.r-1): # A is current coeffs
			A = torch.zeros(torch.numel(ki)+1, 2)
			A[:-1, [0]] = ki
			A[1:, [1]] = ki

			ki = A.mm(torch.tensor([[1], [self.ci[i]]]))

		# Ultimately, ki should be r x 1
		# print("ci: ", self.ci)
		# print("ki: ", ki)

		# Compute higher-order Lie derivatives
		bs = x.size()[0]

		# TODO: is this the right way to compute a gradient within a forward function?
		# TODO: This does forward computation correctly; how does it affect the backwards pass?
		# print(x.requires_grad)
		x.requires_grad = True
		h_ith_deriv = self.h_fn(x) # bs x 1, the zeroth derivative
		h_derivs = h_ith_deriv # bs x 1
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim)) # bs x x_dim

		# print(h_ith_deriv)
		for i in range(self.r-1):
			# print(h_ith_deriv.size())
			grad_h_ith = grad([torch.sum(h_ith_deriv)], x, create_graph=True)[0] # bs x x_dim; create_graph ensures gradient is computed through the gradient operation

			# IPython.embed()
			h_ith_deriv = (grad_h_ith.unsqueeze(dim=1)).bmm(f_val.unsqueeze(dim=2)) # bs x 1 x 1
			h_ith_deriv = h_ith_deriv[:, :, 0] # bs x 1

			# print(h_ith_deriv)
			h_derivs = torch.cat((h_derivs, h_ith_deriv), dim=1)

		# TODO?
		# x.requires_grad = False
		result = beta_value + h_derivs.mm(ki) # bs x 1

		return result

# class H(nn.Module):
# 	def __init__(self):
# 		super().__init__()
#
# 	def forward(self, x):
# 		# TODO (toy): implement
# 		# The way these are implemented should be batch compliant
# 		return None
#
# class XDot(nn.Module):
# 	def __init__(self):
# 		super().__init__()
#
# 	def forward(self, x, u):
# 		# TODO (toy): implement
# 		# The way these are implemented should be batch compliant
# 		return None
#
# class ULimitSetVertices(nn.Module):
# 	def __init__(self):
# 		super().__init__()
#
# 	def forward(self, x):
# 		# TODO (toy): implement
# 		# The way these are implemented should be batch compliant
# 		return None

class Objective(nn.Module):
	def __init__(self, phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim):
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def forward(self, x):
		# The way these are implemented should be batch compliant
		u_lim_set_vertices = self.uvertices_fn(x) # (bs, n_vertices, u_dim), can be a function of x_batch
		print("Inside objective's forward function")
		print(u_lim_set_vertices.size(), u_lim_set_vertices)

		n_vertices = u_lim_set_vertices.size()[1]

		# Evaluate every X against multiple U
		U = torch.reshape(u_lim_set_vertices, (-1, self.u_dim)) # (bs x n_vertices, u_dim)
		# X = torch.tile(x.unsqueeze(1), (1, n_vertices, 1)) # (bs, n_vertices, x_dim)
		X = (x.unsqueeze(1)).repeat(1, n_vertices, 1) # (bs, n_vertices, x_dim)
		X = torch.reshape(X, (-1, self.x_dim)) # (bs x n_vertices, x_dim)

		xdot = self.xdot_fn(X, U)

		# TODO: does the backwards pass work on here if taking gradient wrt x?
		# TODO: this is needed for adversarial training
		x.requires_grad = True
		phi_value = self.phi_fn(x)
		grad_phi = grad([torch.sum(phi_value)], x, create_graph=True)[0] # check
		# x.requires_grad = False

		# IPython.embed()
		# grad_phi = torch.tile(grad_phi.unsqueeze(1), (1, n_vertices, 1))
		grad_phi = (grad_phi.unsqueeze(1)).repeat(1, n_vertices, 1)

		grad_phi = torch.reshape(grad_phi, (-1, self.x_dim))
		# print(grad_phi.size(), xdot.size())

		# Dot product
		phidot_cand = xdot.unsqueeze(1).bmm(grad_phi.unsqueeze(2))
		phidot_cand = torch.reshape(phidot_cand, (-1, n_vertices)) # bs x n_vertices

		phidot, _ = torch.min(phidot_cand, 1)
		result = nn.functional.relu(phidot)
		result = result.view(-1, 1) # ensures bs x 1

		# IPython.embed()
		return result

def main(args):
	# Boilerplate for saving
	save_folder = '%s_%s' % (args.problem, args.affix)

	log_folder = os.path.join(args.log_root, save_folder)
	model_folder = os.path.join(args.model_root, save_folder)

	makedirs(log_folder)
	makedirs(model_folder)

	setattr(args, 'log_folder', log_folder)
	setattr(args, 'model_folder', model_folder)

	logger = create_logger(log_folder, 'train', 'info')
	print_args(args, logger)

	args_savepth = os.path.join(log_folder, "args.txt")
	save_args(args, args_savepth)

	# print("Parsed and saved args")
	# IPython.embed()

	# Selecting problem
	if args.problem == "cartpole":
		r = 2 # TODO
		x_dim = 4
		u_dim = 1
		x_lim = np.array([[-5, 5], [-math.pi/2.0, math.pi/2.0], [-10, 10], [-5, 5]]) # TODO

		# Create phi
		from src.problems.cartpole import H, XDot, ULimitSetVertices
		param_dict = {
			"I": 0.099,
			"m": 0.2,
			"M": 2,
			"l": 0.5,
			"max_theta": math.pi / 10.0,
			"max_force": 1.0
		}

		h_fn = H(param_dict)
		xdot_fn = XDot(param_dict)
		uvertices_fn = ULimitSetVertices(param_dict)

		# print("after initializing problem")
		# bs = 5
		# x = torch.ones((bs, x_dim))
		# u = torch.ones((bs, u_dim))
		# h = h_fn(x)
		# xdot = xdot_fn(x, u)
		# uvertices = uvertices_fn(x)
		# print(h.size(), xdot.size(),uvertices.size())
		# IPython.embed()
	else:
		raise NotImplementedError


	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, args)

	# Tests: tests
	# print("Created CBF function")
	# print("Check on phi that I registered named parameters")
	# for name, param in phi_fn.named_parameters():
	# 	print(name, param.size())
	# print("Check that forward pass compiles")
	# print("Also check that forward pass is correct: c-k conversion and H0 derivs are nonzero")
	# x = torch.rand(10, x_dim)
	# phi_values = phi_fn(x)
	# print(phi_values.size())
	# IPython.embed()

	# Create objective function
	objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim)

	# TODO: tests
	# print("Created objective function")
	# print("Check that forward pass compiles")
	# print("Also check that objective value is >> 0, otherwise there's no point in optimizing")
	# x = torch.rand(10, x_dim)
	# obj_values = objective_fn(x)
	# print(obj_values.size())
	# IPython.embed()

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
