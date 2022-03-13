import torch
from torch import nn
from torch.autograd import grad

from src.attacks.basic_attacker import BasicAttacker
from src.attacks.gradient_batch_attacker import GradientBatchAttacker
from src.attacks.gradient_batch_attacker_warmstart import GradientBatchWarmstartAttacker
from src.trainer import Trainer
from src.reg_samplers.boundary import BoundaryRegSampler
from src.reg_samplers.random import RandomRegSampler
from src.reg_samplers.fixed import FixedRegSampler

reg_samplers_name_to_class_dict = {"boundary": BoundaryRegSampler, "random": RandomRegSampler, "fixed": FixedRegSampler}
from src.utils import *
from src.argument import create_parser, print_args

import os
import math
import pickle

# from global_settings import * # TODO: comment this out before a run

class Phi(nn.Module):
	# Note: currently, we have a implementation which is generic to any r. May be slow
	def __init__(self, h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=None):
		# Later: args specifying how beta is parametrized
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		assert r>=0

		# turn Namespace into dict
		args_dict = vars(args)

		# Note: by default, it registers parameters by their variable name
		self.ci = nn.Parameter(args.phi_ci_init_range*torch.rand(r-1, 1)) # if ci in small range, ki will be much larger
		# rng = args.phi_k0_init_max - args.phi_k0_init_min
		# self.k0 = nn.Parameter(rng*torch.rand(1, 1) + args.phi_k0_init_min)
		self.k0 = nn.Parameter(args.phi_ci_init_range*torch.rand(1, 1))

		# IPython.embed()s
		# To enforce strict positivity for both
		self.ci_min = 1e-2
		self.k0_min = 1e-2

		print("At initialization: k0 is %f" % self.k0.item())
		#############################################################
		hidden_dims = args.phi_nn_dimension.split("-")
		hidden_dims = [int(h) for h in hidden_dims]

		net_layers = []
		prev_dim = self.x_dim

		phi_nnl = args_dict.get("phi_nnl", "relu") # return relu if var "phi_nnl" not on namespace
		for hidden_dim in hidden_dims:
			net_layers.append(nn.Linear(prev_dim, hidden_dim))
			if phi_nnl == "relu":
				net_layers.append(nn.ReLU())
			elif phi_nnl == "tanh":
				net_layers.append(nn.Tanh())
			prev_dim = hidden_dim
		net_layers.append(nn.Linear(prev_dim, 1))
		self.beta_net = nn.Sequential(*net_layers)

	def forward(self, x, grad_x=False):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		# RV is (bs, r+1)

		# print("inside phi's forward")
		# IPython.embed()
		k0 = self.k0 + self.k0_min
		ci = self.ci + self.ci_min

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
			binomial[1] = ci[i]
			ki = A.mm(binomial)

			ki_all[i+1, 0:ki.numel()] = ki.view(1, -1)
			# Ultimately, ki should be r x 1

		# Compute higher-order Lie derivatives
		#####################################################################
		# Turn gradient tracking on for x
		bs = x.size()[0]
		if grad_x == False:
			orig_req_grad_setting = x.requires_grad # Basically only useful if x.requires_grad was False before
			x.requires_grad = True

		if self.x_e is None:
			beta_net_value = self.beta_net(x)
			new_h = nn.functional.softplus(beta_net_value) + k0*self.h_fn(x)
		else:
			beta_net_value = self.beta_net(x)
			beta_net_xe_value = self.beta_net(self.x_e)
			new_h = torch.square(beta_net_value - beta_net_xe_value) + k0*self.h_fn(x)

		if self.args.phi_include_beta_deriv:
			h_ith_deriv = new_h  # bs x 1, the zeroth derivative
		else:
			h_ith_deriv = self.h_fn(x) # bs x 1, the zeroth derivative

		h_derivs = h_ith_deriv # bs x 1
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device)) # bs x x_dim

		for i in range(self.r-1):
			grad_h_ith = grad([torch.sum(h_ith_deriv)], x, create_graph=True)[0] # bs x x_dim; create_graph ensures gradient is computed through the gradient operation
			h_ith_deriv = (grad_h_ith.unsqueeze(dim=1)).bmm(f_val.unsqueeze(dim=2)) # bs x 1 x 1
			h_ith_deriv = h_ith_deriv[:, :, 0] # bs x 1
			h_derivs = torch.cat((h_derivs, h_ith_deriv), dim=1)

		if grad_x == False:
			x.requires_grad = orig_req_grad_setting
		#####################################################################
		# Turn gradient tracking off for x
		result = h_derivs.mm(ki_all.t())
		if self.args.phi_include_beta_deriv:
			phi_r_minus_1_star = result[:, [-1]]
		else:
			phi_r_minus_1_star = result[:, [-1]] - result[:, [0]] + new_h

		result = torch.cat((result, phi_r_minus_1_star), dim=1)

		# print(phi_r_minus_1_star)
		# print(result)
		# IPython.embed()
		return result

class Objective(nn.Module):
	def __init__(self, phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args):
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
		phidot, _ = torch.min(phidot_cand, 1)

		if self.args.no_softplus_on_obj:
			# print("ln 176, creating objective")
			# IPython.embed()
			result = phidot
		else:
			result = nn.functional.softplus(phidot) # using softplus on loss!!!
		result = result.view(-1, 1) # ensures bs x 1

		return result

class Regularizer(nn.Module):
	def __init__(self, phi_fn, device, reg_weight=0.0):
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		assert reg_weight >= 0.0

	def forward(self, x):
		all_phi_values = self.phi_fn(x)
		max_phi_values = torch.max(all_phi_values, dim=1)[0]
		reg = self.reg_weight*torch.mean(torch.sigmoid(0.3*max_phi_values))
		return reg

def create_flying_param_dict(args=None):
	# Args: for modifying the defaults through args
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
	ub = args.box_ang_vel_limit
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
	param_dict["L_p"] = args.pend_length

	# print("creating param_dict in main")
	# IPython.embed()
	return param_dict

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

	# Device
	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
		dev = "cuda:%i" % (args.gpu)
		print("Using GPU device: %s" % dev)
	else:
		dev = "cpu"
	device = torch.device(dev)

	# Selecting problem
	if args.problem == "cartpole_reduced":
		r = 2
		x_dim = 2
		u_dim = 1
		# x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)
		x_lim = np.array([[-math.pi, math.pi], [-args.max_angular_velocity, args.max_angular_velocity]], dtype=np.float32)

		# Create phi
		from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

		# param_dict = {
		# 	"I": 0.021,
		# 	"m": 0.25,
		# 	"M": 1.00,
		# 	"l": 0.5
		# }

		if args.physical_difficulty == 'easy': # medium length pole
			param_dict = {
				"I": 1.2E-3,
				"m": 0.127,
				"M": 1.0731,
				"l": 0.3365
				# "max_theta": math.pi / 2.0,
				# "max_force": 15.0
			}
		elif args.physical_difficulty == 'hard': # long pole
			param_dict = {
				"I": 7.88E-3,
				"m": 0.230,
				"M": 1.0731,
				"l": 0.6413
				# "max_theta": math.pi / 4.0,
				# "max_force": 1.0
			}

		param_dict["max_theta"] = args.max_theta
		param_dict["max_force"] = args.max_force

		h_fn = H(param_dict)
		xdot_fn = XDot(param_dict)
		uvertices_fn = ULimitSetVertices(param_dict, device)


		# n_mesh_grain = 0.75 # TODO: increase or decrease?
		n_mesh_grain = args.reg_sample_distance
		XXX = np.meshgrid(*[np.arange(r[0], r[1], n_mesh_grain) for r in x_lim])
		reg_samples = np.concatenate([x.flatten()[:, None] for x in XXX], axis=1)
		reg_samples = torch.from_numpy(reg_samples.astype(np.float32)).to(device)
		reg_sampler = FixedRegSampler(x_lim, device, logger, samples=reg_samples)

		if args.phi_include_xe:
			x_e = torch.zeros(1, x_dim)
		else:
			x_e = None
	elif args.problem == "flying_inv_pend":
		param_dict = create_flying_param_dict(args)

		r = param_dict["r"]
		x_dim = param_dict["x_dim"]
		u_dim = param_dict["u_dim"]
		x_lim = param_dict["x_lim"]

		# Create phi
		from src.problems.flying_inv_pend import HMax, HSum, XDot, ULimitSetVertices
		if args.h == "sum":
			h_fn = HSum(param_dict)
		elif args.h == "max":
			h_fn = HMax(param_dict)

		xdot_fn = XDot(param_dict, device)
		uvertices_fn = ULimitSetVertices(param_dict, device)

		reg_sampler = reg_samplers_name_to_class_dict[args.reg_sampler](x_lim, device, logger, n_samples=args.reg_n_samples)

		if args.phi_include_xe:
			x_e = torch.zeros(1, x_dim)
		else:
			x_e = None
	else:
		raise NotImplementedError

	# Save param_dict
	with open(os.path.join(log_folder, "param_dict.pkl"), 'wb') as handle:
		pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Send all modules to the correct device
	h_fn = h_fn.to(device)
	xdot_fn = xdot_fn.to(device)
	uvertices_fn = uvertices_fn.to(device)
	if x_e is not None:
		x_e = x_e.to(device)
	x_lim = torch.tensor(x_lim).to(device)

	# Create CBF, etc.
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)
	objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args)
	reg_fn = Regularizer(phi_fn, device, reg_weight=args.reg_weight)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)
	objective_fn = objective_fn.to(device)
	reg_fn = reg_fn.to(device)

	# Create attacker
	if args.train_attacker == "basic":
		attacker = BasicAttacker(x_lim, device, stopping_condition="early_stopping")
	elif args.train_attacker == "gradient_batch":
		attacker = GradientBatchAttacker(x_lim, device, logger, n_samples=args.train_attacker_n_samples, stopping_condition=args.train_attacker_stopping_condition, lr=args.train_attacker_lr, projection_tolerance=args.train_attacker_projection_tolerance, projection_lr=args.train_attacker_projection_lr)
	elif args.train_attacker == "gradient_batch_warmstart":
		attacker = GradientBatchWarmstartAttacker(x_lim, device, logger, n_samples=args.train_attacker_n_samples, stopping_condition=args.train_attacker_stopping_condition, max_n_steps=args.train_attacker_max_n_steps,lr=args.train_attacker_lr, projection_tolerance=args.train_attacker_projection_tolerance, projection_lr=args.train_attacker_projection_lr, projection_time_limit=args.train_attacker_projection_time_limit)

	# Create test attacker
	if args.test_attacker == "basic":
		test_attacker = BasicAttacker(x_lim, device, stopping_condition="early_stopping")
	elif args.test_attacker == "gradient_batch":
		test_attacker = GradientBatchAttacker(x_lim, device, logger, n_samples=args.test_attacker_n_samples, stopping_condition=args.test_attacker_stopping_condition, lr=args.test_attacker_lr, projection_tolerance=args.test_attacker_projection_tolerance, projection_lr=args.test_attacker_projection_lr)
	elif args.test_attacker == "gradient_batch_warmstart":
		test_attacker = GradientBatchWarmstartAttacker(x_lim, device, logger, n_samples=args.test_attacker_n_samples, stopping_condition=args.test_attacker_stopping_condition, max_n_steps=args.test_attacker_max_n_steps, lr=args.test_attacker_lr, projection_tolerance=args.test_attacker_projection_tolerance, projection_lr=args.test_attacker_projection_lr)

	# Pass everything to Trainer
	trainer = Trainer(args, logger, attacker, test_attacker, reg_sampler)
	trainer.train(objective_fn, reg_fn, phi_fn, xdot_fn)

	##############################################################
	#####################      Testing      ######################

	# 2.20 bug
	# x_rand = torch.rand(1, 10).to(device)
	# phi_vals = phi_fn(x_rand)
	# # IPython.embed()
	# obj = phi_vals[0, -1]
	# obj.backward()
	#
	# for n, p in phi_fn.named_parameters():
	# 	print(n, p.grad)
	# # IPython.embed()



if __name__ == "__main__":
	parser = create_parser()
	args = parser.parse_args()
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)
	main(args)
