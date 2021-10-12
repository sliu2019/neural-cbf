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
from global_settings import *

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
		self.ci = nn.Parameter(args.phi_ci_init_range*torch.rand(r-1, 1)) # if ci in small range, ki will be much largers

		hidden_dims = args.phi_nn_dimension.split("-")
		hidden_dims = [int(h) for h in hidden_dims]

		net_layers = []
		prev_dim = self.x_dim
		for hidden_dim in hidden_dims:
			net_layers.append(nn.Linear(prev_dim, hidden_dim))
			net_layers.append(nn.ReLU())
			prev_dim = hidden_dim
		net_layers.append(nn.Linear(prev_dim, 1))
		self.beta_net = nn.Sequential(*net_layers)

		# IPython.embed()
		# self.beta_net = nn.Sequential(
		# 	nn.Linear(x_dim, hidden_dim),
		# 	nn.ReLU(),
		# 	nn.Linear(hidden_dim, 1)
		# )

		# hidden_dim = 100
		# self.beta_net = nn.Sequential(
		# 	nn.Linear(x_dim, hidden_dim),
		# 	nn.ReLU(),
		# 	nn.Linear(hidden_dim, hidden_dim),
		# 	nn.ReLU(),
		# 	nn.Linear(hidden_dim, 1)
		# )
		#
		# state_dict = self.beta_net.state_dict() # 0.weight/bias and 2.weight/bias
		# state_dict['4.weight'] = torch.rand(state_dict['4.weight'].shape)*(-100.0)
		# self.beta_net.load_state_dict(state_dict)

		# def init_weights(m):
		# 	if isinstance(m, nn.Linear):
		# 		torch.nn.init.normal_(m.weight, std=1e-3)
		# 		m.bias.data.fill_(1e-5)
		# self.beta_net.apply(init_weights)

		# self.beta_net = nn.Sequential(
		# 	nn.Linear(x_dim, hidden_dim),
		# 	nn.ReLU(),
		# 	nn.Linear(hidden_dim, 1)
		# )

		# state_dict = self.beta_net.state_dict() # 0.weight/bias and 2.weight/bias
		# state_dict['2.weight'] = torch.rand(state_dict['2.weight'].shape)*(-50.0)
		# self.beta_net.load_state_dict(state_dict)

		if self.x_e is not None:
			self.x_e = self.x_e.view(1, -1)

			h_xe = self.h_fn(self.x_e)
			self.c = -np.log(2.0)/h_xe

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		h_val = self.h_fn(x)
		if self.x_e is None:
			# beta_value = nn.functional.softplus(self.beta_net(x)) + nn.functional.relu(h_val + torch.sign(h_val)) - 1.0
			beta_value = nn.functional.softplus(self.beta_net(x)) + nn.functional.softplus(h_val) - np.log(2)
		else:
			# beta_value = nn.functional.softplus(self.beta_net(x) - self.beta_net(self.x_e)) + nn.functional.relu(h_val + torch.sign(h_val)) - 1.0
			# alpha_value = 1.0/(1.0 + torch.exp(-self.c*x)) - (1.0/2)
			alpha_value = self.c*h_val
			beta_value = nn.functional.softplus(self.beta_net(x) - self.beta_net(self.x_e)) + alpha_value
			# beta_value = self.c*h_val
		# IPython.embed()

		# Convert ci to ki
		ki = torch.tensor([[1.0]])
		ki_all = torch.zeros(self.r, self.r).to(self.device) # phi_i coefficients are in row i
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
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device)) # bs x x_dim

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
	def __init__(self, phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger):
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def forward(self, x):
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

		phidot, _ = torch.min(phidot_cand, 1)
		result = nn.functional.relu(phidot)
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
		reg = torch.zeros(1, 1).to(self.device)
		if self.volume_term_weight:
			phi_value_A_samples = self.phi_fn(self.A_samples)

			# phi_value_pos_bool = torch.where(phi_value_A_samples >= 0.0, 1.0, 0.0)
			# phi_value_pos = phi_value_A_samples * phi_value_pos_bool
			# volume_term = torch.sum(phi_value_pos)

			phi_value_pos_bool = torch.where(phi_value_A_samples >= 0.0, 1.0, 0.0)
			phi_value_pos = phi_value_A_samples * phi_value_pos_bool
			num_pos_per_sample = torch.sum(phi_value_pos_bool, dim=1)
			numerator = torch.sum(phi_value_pos, dim=1)
			num_pos_per_sample = torch.clamp(num_pos_per_sample, min=1e-5)
			# avg_pos_phi_per_sample = num_pos_per_sample/denom
			avg_pos_phi_per_sample = numerator/num_pos_per_sample

			# fix divide by zeros
			# avg_pos_phi_per_sample[avg_pos_phi_per_sample != avg_pos_phi_per_sample] = 0.0
			volume_term = torch.mean(avg_pos_phi_per_sample)

			reg = self.volume_term_weight * volume_term
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

		n_samples = 50
		rnge = torch.tensor([param_dict["max_theta"], x_lim[1:x_dim, 1]])
		A_samples = torch.rand(n_samples, x_dim)*(2*rnge) - rnge # (n_samples, x_dim)
		print(A_samples)
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
		attacker = GradientBatchAttacker(x_lim, device, logger, stopping_condition=args.train_attacker_stopping_condition, n_samples=args.train_attacker_n_samples, projection_stop_threshold=args.train_attacker_projection_stop_threshold, projection_lr=args.train_attacker_projection_lr)

	# Create test attacker
	if args.test_attacker == "basic":
		test_attacker = BasicAttacker(x_lim, device, stopping_condition="early_stopping")
	elif args.test_attacker == "gradient_batch":
		test_attacker = GradientBatchAttacker(x_lim, device, logger, stopping_condition=args.test_attacker_stopping_condition, n_samples=args.test_attacker_n_samples, projection_stop_threshold=args.test_attacker_projection_stop_threshold, projection_lr=args.test_attacker_projection_lr)

	# Pass everything to Trainer
	trainer = Trainer(args, logger, attacker, test_attacker)
	trainer.train(objective_fn, reg_fn, phi_fn, xdot_fn)

	##############################################################
	#####################      Testing      ######################
	# TODO: Simin, on Sat you are here!
	# Testing newly design CBF
	# print("Debug cbf further")
	# IPython.embed()
	# phi_fn(x_e) # is 0?
	# Draw 2D plot
	# file_name = os.path.join(args.model_folder, f'checkpoint_0.pth')
	# save_model(phi_fn, file_name)

	# Test: initialization of attacks on manifold
	# IPython.embed()
	# X = test_attacker.sample_points_on_boundary(phi_fn)
	# X = X.detach().cpu().numpy()
	# save_pth = os.path.join(log_folder, "boundary_samples.npy")
	# np.save(save_pth, X)
	# print("Check points and saved correctly")
	# IPython.embed()

	# run this and viz
	# X = test_attacker.opt(objective_fn, phi_fn)
	# X = X.detach().cpu().numpy()
	# save_pth = os.path.join(log_folder, "boundary_samples_post_opt.npy")
	# np.save(save_pth, X)
	# print("Saved")
	# # print("Check points and saved correctly")
	# IPython.embed()

	# Test
	# IPython.embed()
	# X = torch.tensor([0, -5.0])
	# X = X.view(1, -1).to(device)
	# print(phi_fn(X))
	# print(phi_fn(-X))

	# Test: different phi init gives different plot
	# for name, param in phi_fn.named_parameters():
	# 	print(name, param)

	# file_name = os.path.join(args.model_folder, f'checkpoint_0.pth')
	# save_model(phi_fn, file_name)

	# Test:
	# t0 = time.perf_counter()
	# test_attacker.opt(objective_fn, phi_fn)
	# t1 = time.perf_counter()
	# # print("Total time: %f s" % (t1-t0))
	#
	# test_attacker.opt(objective_fn, phi_fn)

	# Test load model
	"""load_model(phi_fn, "./checkpoint/cartpole_reduced_exp1a/checkpoint_60.pth")
	for name, param in phi_fn.named_parameters():
		print(name, param)
	# print(list(phi_fn.parameters()))
	IPython.embed()"""

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

	# Check magnitude of volume regularization term
	# x = torch.rand(10, x_dim)
	# obj_value = objective_fn(x)

	# Timing the projection



if __name__ == "__main__":
	args = parser()
	main(args)


