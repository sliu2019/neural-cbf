import numpy as np

import torch
from torch import nn
from torch.autograd import grad

from src.attacks.basic_attacker import BasicAttacker
from src.attacks.gradient_batch_attacker import GradientBatchAttacker
from src.trainer import Trainer
from src.utils import *
from src.argument import parser, print_args

from scipy.linalg import pascal
import os, sys
import math
import IPython
import time

# from global_settings import * # TODO: comment this out before a run

class Phi(nn.Module):
	# Note: currently, we have a implementation which is generic to any r. May be slow
	def __init__(self, h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=None):
		# Later: args specifying how beta is parametrized
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		assert r>=0

		# Note: by default, it registers parameters by their variable name
		self.ci = nn.Parameter(args.phi_ci_init_range*torch.rand(r-1, 1)) # if ci in small range, ki will be much larger
		self.a = nn.Parameter(torch.rand(1, 1)) # constrained to be positive, TODO pending change. Can also set this to constant 0.1
		# self.sigma = nn.Parameter(1e-2*torch.rand(1)) # TODO
		# print("################################################################")
		# print("Initial ci: ", self.ci)
		# print("Initial sig:", self.sigma)
		# print("################################################################")

		hidden_dims = args.phi_nn_dimension.split("-")
		hidden_dims = [int(h) for h in hidden_dims]

		net_layers = []
		# print(self.args["g_input_is_xy"])
		if self.args.g_input_is_xy:
			print("Warning: args.g_input_is_xy flag should only be used for reduced_cartpole problem")
			prev_dim = 1
		else:
			prev_dim = self.x_dim

		for hidden_dim in hidden_dims:
			net_layers.append(nn.Linear(prev_dim, hidden_dim))
			net_layers.append(nn.ReLU())
			prev_dim = hidden_dim
		net_layers.append(nn.Linear(prev_dim, 1))
		self.beta_net = nn.Sequential(*net_layers)

		# if self.x_e is not None:
		# 	self.x_e = self.x_e.view(1, -1)
		#
		# 	h_xe = self.h_fn(self.x_e)
		#
		# 	# print("Constructing CBF: in init")
		# 	IPython.embed()
		# 	# self.c = (-np.log(2.0))/h_xe # TODO: put below back
		# 	# self.c = (-np.log(2.0) - 0.1)/h_xe # TODO: pending change
		# 	self.c = (-torch.tensor(2.0) - self.a)/h_xe

	def forward(self, x, grad_x=False):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		h_val = self.h_fn(x)

		# print("in forward")
		# IPython.embed()
		if self.args.g_input_is_xy: # TODO: pending change
			beta_net_value = self.beta_net(x[:, 0]*x[:, 1])
			beta_net_xe_value = self.beta_net(self.x_e[:, 0]*self.x_e[:, 1])
		else:
			beta_net_value = self.beta_net(x)
			beta_net_xe_value = self.beta_net(self.x_e)

		if self.x_e is None:
			beta_value = nn.functional.softplus(beta_net_value) + nn.functional.softplus(h_val) - np.log(2) #+ self.sigma
		else:
			# alpha_value = self.c*h_val
			h_xe = self.h_fn(self.x_e)
			num = (-torch.tensor(2.0) - self.a).to(self.device)
			alpha_value = num/h_xe
			beta_value = nn.functional.softplus(beta_net_value-beta_net_xe_value) + alpha_value #+ self.sigma

		# Convert ci to ki
		ci = self.ci + 1e-2 # TODO: pending change
		ki = torch.tensor([[1.0]])
		ki_all = torch.zeros(self.r, self.r).to(self.device) # phi_i coefficients are in row i
		ki_all[0, 0:ki.numel()] = ki
		for i in range(self.r-1): # A is current coeffs
			A = torch.zeros(torch.numel(ki)+1, 2)
			A[:-1, [0]] = ki
			A[1:, [1]] = ki

			# Note: to preserve gradient flow, have to assign mat entries to ci not create with ci (i.e. torch.tensor([ci[0]]))
			binomial = torch.ones((2, 1))
			# binomial[1] = self.ci[i]
			binomial[1] = ci[i]
			ki = A.mm(binomial)

			ki_all[i+1, 0:ki.numel()] = ki.view(1, -1)
		# Ultimately, ki should be r x 1

		# Compute higher-order Lie derivatives
		bs = x.size()[0]
		if grad_x == False:
			orig_req_grad_setting = x.requires_grad # Basically only useful if x.requires_grad was False before
			x.requires_grad = True

		h_ith_deriv = self.h_fn(x) # bs x 1, the zeroth derivative
		h_derivs = h_ith_deriv # bs x 1
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device)) # bs x x_dim

		for i in range(self.r-1):
			grad_h_ith = grad([torch.sum(h_ith_deriv)], x, create_graph=True)[0] # bs x x_dim; create_graph ensures gradient is computed through the gradient operation
			# grad_h_ith = grad([torch.sum(h_ith_deriv)], x, retain_graph=True)[0] # TODO: for debugging only

			h_ith_deriv = (grad_h_ith.unsqueeze(dim=1)).bmm(f_val.unsqueeze(dim=2)) # bs x 1 x 1
			h_ith_deriv = h_ith_deriv[:, :, 0] # bs x 1

			h_derivs = torch.cat((h_derivs, h_ith_deriv), dim=1)

		if grad_x == False:
			x.requires_grad = orig_req_grad_setting
		# New: (bs, r+1)
		# TODO: turn back on
		# print("h_derivs")
		# print(h_derivs)
		# print("ki_all")
		# print(ki_all)

		result = h_derivs.mm(ki_all.t())
		phi_r_minus_1_star = result[:, [-1]] - result[:, [0]] + beta_value
		result = torch.cat((result, phi_r_minus_1_star), dim=1)

		return result

class Objective(nn.Module):
	def __init__(self, phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger):
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def forward(self, x):
		# IPython.embed()
		# The way these are implemented should be batch compliant
		u_lim_set_vertices = self.uvertices_fn(x) # (bs, n_vertices, u_dim), can be a function of x_batch
		n_vertices = u_lim_set_vertices.size()[1]

		# Evaluate every X against multiple U
		U = torch.reshape(u_lim_set_vertices, (-1, self.u_dim)) # (bs x n_vertices, u_dim)
		X = (x.unsqueeze(1)).repeat(1, n_vertices, 1) # (bs, n_vertices, x_dim)
		X = torch.reshape(X, (-1, self.x_dim)) # (bs x n_vertices, x_dim)

		# IPython.embed()
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

		# print(phidot_cand)
		phidot, min_indices = torch.min(phidot_cand, 1) # TODO: let second argument be _

		# print("Figure out shapes in objectvie fn")
		# IPython.embed()
		# verif_minimizing_u = (x[:, [0]]*(min_indices.view(-1, 1))-0.5) <= 0
		# assert torch.all(verif_minimizing_u)
		# print(min_indices)
		result = phidot
		# result = nn.functional.relu(phidot) # TODO
		result = result.view(-1, 1) # ensures bs x 1

		return result

class Regularizer(nn.Module):
	def __init__(self, phi_fn, device, volume_term_weight=0.0,
	             A_samples=None):
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		assert volume_term_weight >= 0.0
		if volume_term_weight:
			assert A_samples is not None

	def forward(self):
		reg = torch.tensor(0).to(self.device)
		if self.volume_term_weight:
			# IPython.embed()
			phi_value_A_samples = self.phi_fn(self.A_samples)
			max_phi_values = torch.max(phi_value_A_samples, dim=1)[0]

			sharp_sigmoid = 1.0/(1.0 + torch.exp(-10*max_phi_values)) # alpha = 10.0, instead of 1.0 as usual
			step_on_max = -(sharp_sigmoid + 0.001*nn.functional.relu(max_phi_values)) + 1.0

			reg = -self.volume_term_weight*torch.mean(step_on_max)
		return reg

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

	# Device
	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
		dev = "cuda:%i" % (args.gpu)
		print("Using GPU device: %s" % dev)
	else:
		dev = "cpu"
	device = torch.device(dev)

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
		uvertices_fn = ULimitSetVertices(param_dict, device)

		# print("ln 167, main, check A sample creation")
		# IPython.embed()
		# n_samples = 50
		# rnge = torch.tensor([param_dict["max_theta"], x_lim[1:x_dim, 0]])
		# A_samples = torch.rand(n_samples, x_dim)*(2*rnge) - rnge # (n_samples, x_dim)
		# A_samples = torch.rand(10, x_dim)

		A_samples = None
		x_e = torch.zeros(1, x_dim)
	elif args.problem == "cartpole_reduced":
		r = 2
		x_dim = 2
		u_dim = 1
		x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)

		# Create phi
		from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

		if args.physical_difficulty == 'easy':
			param_dict = {
				"I": 0.021,
				"m": 0.25,
				"M": 1.00,
				"l": 0.5,
				"max_theta": math.pi / 2.0,
				"max_force": 15.0
			}
		elif args.physical_difficulty == 'hard':
			param_dict = {
				"I": 0.021,
				"m": 0.25,
				"M": 1.00,
				"l": 0.5,
				"max_theta": math.pi / 4.0,
				"max_force": 1.0
			}

		h_fn = H(param_dict)
		xdot_fn = XDot(param_dict)
		uvertices_fn = ULimitSetVertices(param_dict, device)


		n_mesh_grain = 0.75 # TODO: increase or decrease?
		XXX = np.meshgrid(*[np.arange(r[0], r[1], n_mesh_grain) for r in x_lim])
		A_samples = np.concatenate([x.flatten()[:, None] for x in XXX], axis=1)
		A_samples = torch.from_numpy(A_samples.astype(np.float32))

		x_e = torch.zeros(1, x_dim)
	else:
		raise NotImplementedError
		A_samples = None
		x_e = None

	# Send all modules to the correct device
	h_fn = h_fn.to(device)
	xdot_fn = xdot_fn.to(device)
	uvertices_fn = uvertices_fn.to(device)
	if x_e is not None:
		x_e = x_e.to(device)
	if A_samples is not None:
		A_samples = A_samples.to(device)
	x_lim = torch.tensor(x_lim).to(device)

	# Create CBF
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)
	# Create objective function
	objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger)
	reg_fn = Regularizer(phi_fn, device, volume_term_weight=args.objective_volume_weight, A_samples=A_samples)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)
	objective_fn = objective_fn.to(device)
	reg_fn = reg_fn.to(device)

	# Create attacker
	if args.train_attacker == "basic":
		attacker = BasicAttacker(x_lim, device, stopping_condition="early_stopping")
	elif args.train_attacker == "gradient_batch":
		attacker = GradientBatchAttacker(x_lim, device, logger, n_samples=args.train_attacker_n_samples, stopping_condition=args.train_attacker_stopping_condition, lr=args.train_attacker_lr, projection_stop_threshold=args.train_attacker_projection_stop_threshold, projection_lr=args.train_attacker_projection_lr)

	# Create test attacker
	if args.test_attacker == "basic":
		test_attacker = BasicAttacker(x_lim, device, stopping_condition="early_stopping")
	elif args.test_attacker == "gradient_batch":
		test_attacker = GradientBatchAttacker(x_lim, device, logger, n_samples=args.test_attacker_n_samples, stopping_condition=args.test_attacker_stopping_condition, lr=args.test_attacker_lr, projection_stop_threshold=args.test_attacker_projection_stop_threshold, projection_lr=args.test_attacker_projection_lr)

	# Pass everything to Trainer
	# trainer = Trainer(args, logger, attacker, test_attacker)
	# trainer.train(objective_fn, reg_fn, phi_fn, xdot_fn)

	##############################################################
	#####################      Testing      ######################
	# Test of bug fix 11/8
	# IPython.embed()
	# X = torch.tensor([0.0, 0.0])
	# X = X.view(1, -1).to(device)
	# print(phi_fn(X))
	# print(phi_fn.ci)
	# IPython.embed()

	# Test there is grad of reg term wrt parameters
	# IPython.embed()
	# # gradients = grad([reg], list(phi_fn.parameters()))[0]

	# import torch.optim as optim
	# optimizer = optim.Adam(phi_fn.parameters())
	# reg = reg_fn()
	# reg.backward()
	# optimizer.step()

	# IPython.embed()


if __name__ == "__main__":
	args = parser()
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)
	# IPython.embed()
	main(args)


