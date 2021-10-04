import torch
import numpy as np

from torch import nn
from torch.autograd import grad
from src.attacks.basic_attacker import BasicAttacker
from src.attacks.gradient_batch_attacker import GradientBatchAttacker
from src.trainer import Trainer
from src.utils import *
from scipy.linalg import pascal
from src.argument import parser, print_args
import os, sys
import math
import IPython
import time
# TODO: THIS MAKES PYTORCH DETERMINISTIC!! IF THAT ISN'T WHAT YOU WANT, remove it
torch.manual_seed(1)

class Phi(nn.Module):
	# Note: currently, we have a implementation which is generic to any r. May be slow

	def __init__(self, h_fn, xdot_fn, r, x_dim, u_dim, x_e=None):
		# Later: args specifying how beta is parametrized
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		assert r>=0
		# Note: by default, it registers parameters by their variable name
		self.ci = nn.Parameter(5*torch.rand(r-1, 1)) # int from 0-5 (if ci in small range, ki will be much larger) # TODO: param
		self.beta_net = nn.Sequential(
			nn.Linear(x_dim, 25*x_dim),
			nn.ReLU(),
			nn.Linear(25*x_dim, 1)
		) # TODO: param

		self.x_e = self.x_e.view(1, -1)

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		h_val = self.h_fn(x)
		if self.x_e is None:
			beta_value = nn.functional.softplus(self.beta_net(x)) + nn.functional.relu(h_val + torch.sign(h_val)) - 1.0
		else:
			beta_value = nn.functional.softplus(self.beta_net(x) - self.beta_net(self.x_e)) + nn.functional.relu(h_val + torch.sign(h_val)) - 1.0

		# IPython.embed()

		# Convert ci to ki
		ki = torch.tensor([[1.0]])
		ki_all = torch.zeros(self.r, self.r) # phi_i coefficients are in row i
		ki_all[0, 0:ki.numel()] = ki
		for i in range(self.r-1): # A is current coeffs
			A = torch.zeros(torch.numel(ki)+1, 2)
			A[:-1, [0]] = ki
			A[1:, [1]] = ki

			# Note: to preserve gradient flow, have to assign mat entries to ci not create with ci (i.e. torch.tensor([ci[0]]))
			binomial = torch.ones((2, 1))
			binomial[1] = self.ci[i]
			ki = A.mm(binomial)

			ki_all[i+1, 0:ki.numel()] = ki.view(1, -1)
		# Ultimately, ki should be r x 1

		# Compute higher-order Lie derivatives
		bs = x.size()[0]
		orig_req_grad_setting = x.requires_grad # Basically only useful if x.requires_grad was False before
		x.requires_grad = True

		h_ith_deriv = self.h_fn(x) # bs x 1, the zeroth derivative
		h_derivs = h_ith_deriv # bs x 1
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim)) # bs x x_dim

		for i in range(self.r-1):
			# print(h_ith_deriv.size())
			grad_h_ith = grad([torch.sum(h_ith_deriv)], x, create_graph=True)[0] # bs x x_dim; create_graph ensures gradient is computed through the gradient operation

			# IPython.embed()
			h_ith_deriv = (grad_h_ith.unsqueeze(dim=1)).bmm(f_val.unsqueeze(dim=2)) # bs x 1 x 1
			h_ith_deriv = h_ith_deriv[:, :, 0] # bs x 1

			# print(h_ith_deriv)
			h_derivs = torch.cat((h_derivs, h_ith_deriv), dim=1)

		x.requires_grad = orig_req_grad_setting
		# Old:
		# result = beta_value + h_derivs.mm(ki) # bs x 1
		# New: (bs, r+1)
		result = h_derivs.mm(ki_all.t())
		phi_r_minus_1_star = result[:, [-1]] - result[:, [0]] + beta_value
		result = torch.cat((result, phi_r_minus_1_star), dim=1)

		return result

class Objective(nn.Module):
	def __init__(self, phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, volume_term_weight=0.0, A_samples=None):
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		assert volume_term_weight >= 0.0
		if volume_term_weight:
			assert A_samples is not None

	def forward(self, x):
		# The way these are implemented should be batch compliant
		u_lim_set_vertices = self.uvertices_fn(x) # (bs, n_vertices, u_dim), can be a function of x_batch
		n_vertices = u_lim_set_vertices.size()[1]

		# Evaluate every X against multiple U
		U = torch.reshape(u_lim_set_vertices, (-1, self.u_dim)) # (bs x n_vertices, u_dim)
		X = (x.unsqueeze(1)).repeat(1, n_vertices, 1) # (bs, n_vertices, x_dim)
		X = torch.reshape(X, (-1, self.x_dim)) # (bs x n_vertices, x_dim)

		xdot = self.xdot_fn(X, U)

		orig_req_grad_setting = x.requires_grad
		x.requires_grad = True
		phi_value = self.phi_fn(x)
		grad_phi = grad([torch.sum(phi_value[:, -1])], x, create_graph=True)[0] # check
		x.requires_grad = orig_req_grad_setting

		grad_phi = (grad_phi.unsqueeze(1)).repeat(1, n_vertices, 1)
		grad_phi = torch.reshape(grad_phi, (-1, self.x_dim))

		# Dot product
		phidot_cand = xdot.unsqueeze(1).bmm(grad_phi.unsqueeze(2))
		phidot_cand = torch.reshape(phidot_cand, (-1, n_vertices)) # bs x n_vertices

		phidot, _ = torch.min(phidot_cand, 1)
		result = nn.functional.relu(phidot)
		result = result.view(-1, 1) # ensures bs x 1

		# Add volume term
		# Calculate the percentage of points in A which are not in S
		if self.volume_term_weight:
			# IPython.embed()
			phi_value_A_samples = self.phi_fn(self.A_samples)

			phi_value_pos_bool = torch.where(phi_value_A_samples >= 0.0, 1.0, 0.0)
			phi_value_pos = phi_value_A_samples*phi_value_pos_bool
			volume_term = torch.sum(phi_value_pos)

			# print(result)
			result = result + self.volume_term_weight*volume_term
			# print(result)
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
		r = 2
		x_dim = 4
		u_dim = 1
		x_lim = np.array([[-5, 5], [-math.pi/2.0, math.pi/2.0], [-10, 10], [-5, 5]], dtype=np.float32) # TODO

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

		# print("ln 167, main, check A sample creation")
		# IPython.embed()
		# n_samples = 50
		# rnge = torch.tensor([param_dict["max_theta"], x_lim[1:x_dim, 0]])
		# A_samples = torch.rand(n_samples, x_dim)*(2*rnge) - rnge # (n_samples, x_dim)
		# A_samples = torch.rand(10, x_dim)

		A_samples = None
		phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, x_e=torch.zeros(1, x_dim))
	else:
		A_samples = None
		phi_fn = None
		raise NotImplementedError


	# Create objective function
	objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, args.objective_volume_weight, A_samples)

	# Create attacker
	if args.train_attacker == "basic":
		attacker = BasicAttacker(x_lim, stopping_condition="early_stopping")
	elif args.train_attacker == "gradient_batch":
		attacker = GradientBatchAttacker(x_lim, stopping_condition=args.train_attacker_stopping_condition, n_samples=args.train_attacker_n_samples)

	# Create test attacker
	if args.test_attacker == "basic":
		test_attacker = BasicAttacker(x_lim, stopping_condition="early_stopping")
	elif args.test_attacker == "gradient_batch":
		test_attacker = GradientBatchAttacker(x_lim, stopping_condition=args.test_attacker_stopping_condition, n_samples=args.test_attacker_n_samples)

	# Test:
	# t0 = time.perf_counter()
	# test_attacker.opt(objective_fn, phi_fn)
	# t1 = time.perf_counter()
	# # print("Total time: %f s" % (t1-t0))
	#
	# test_attacker.opt(objective_fn, phi_fn)

	# Pass everything to Trainer
	# trainer = Trainer(args, logger, attacker, test_attacker)
	# trainer.train(objective_fn, phi_fn, xdot_fn)

	# Test load model
	# print(list(phi_fn.parameters()))
	# load_model(phi_fn, "./checkpoint/cartpole_default/checkpoint_1490.pth")
	# print(list(phi_fn.parameters()))
	# # IPython.embed()
	#
	# # Strong attack against model
	# test_attacker = GradientBatchAttacker(x_lim, stopping_condition=args.test_attacker_stopping_condition, n_samples=200)
	# t0 = time.perf_counter()
	# test_attacker.opt(objective_fn, phi_fn)
	# t1 = time.perf_counter()

	# TODO: test if gradient is propagated through volume term
	# TODO: check out magnitude of gradient term. What to set weights to?

	# Check the refactored phi function
	"""x = torch.rand(10, x_dim)
	# x.requires_grad = True
	phi_value = phi_fn(x)
	IPython.embed()
	for i in range(r+1):
		# phi_grad = grad([torch.sum(phi_value[:, i])], x, retain_graph=True)[0]
		# print(phi_grad)

		loss = torch.sum(phi_value[:, i])
		loss.backward(retain_graph=True)
		for param in phi_fn.parameters():
			print(param.grad)
		print("\n")
	# test_attacker.opt(objective_fn, phi_fn)"""

	# IPython.embed()
	# x = torch.zeros(1, 4)
	# print(phi_fn(x))

	# Check the refactored objective
	# x = torch.rand(10, x_dim)
	# loss = torch.sum(objective_fn(x))
	# loss.backward()
	# for name, param in phi_fn.named_parameters():
	# 	print(name)
	# 	print(param.grad)

	# Timing the projection



if __name__ == "__main__":
	args = parser()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	main(args)


