import os

import matplotlib.pyplot as plt
import math
from dotmap import DotMap
import torch
import pickle
import time, os, glob, re, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phi_low_torch_module import PhiLow
from src.utils import *

# Make numpy and torch deterministic (for rand phi and attack/reg sampling)
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cpu")

def load_philow(phi_load_fpth=None):
	# TODO: commented below, because haven't implemented GD training of phi_low yet
	# if exp_name:
	# 	fnm = "./log/%s/args.txt" % exp_name
	# 	# args = load_args(fnm) # can't use, args conflicts with args in outer scope
	# 	with open(fnm, 'r') as f:
	# 		json_data = json.load(f)
	# 	args = DotMap(json_data)
	# 	param_dict = pickle.load(open("./log/%s/param_dict.pkl" % exp_name, "rb"))
	# else:
	from src.argument import create_parser
	parser = create_parser() # default
	args = parser.parse_args()

	from main import create_flying_param_dict
	param_dict = create_flying_param_dict(args) # default

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
	# uvertices_fn = ULimitSetVertices(param_dict, device)

	# reg_sampler = reg_samplers_name_to_class_dict[args.reg_sampler](x_lim, device, logger, n_samples=args.reg_n_samples)

	# if args.phi_include_xe:
	# 	x_e = torch.zeros(1, x_dim)
	# else:
	# 	x_e = None

	# Passing in subset of state to NN
	from src.utils import IndexNNInput, TransformEucNNInput
	state_index_dict = param_dict["state_index_dict"]
	# if args.phi_nn_inputs == "spherical":
	# 	nn_input_modifier = None
	# elif args.phi_nn_inputs == "euc":
	# 	nn_input_modifier = TransformEucNNInput(state_index_dict)

	# Send all modules to the correct device
	h_fn = h_fn.to(device)
	xdot_fn = xdot_fn.to(device)
	# uvertices_fn = uvertices_fn.to(device)
	# if x_e is not None:
	# 	x_e = x_e.to(device)
	# x_lim = torch.tensor(x_lim).to(device)

	# Create CBF, etc.
	# phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e, nn_input_modifier=nn_input_modifier)
	phi_fn = PhiLow(h_fn, xdot_fn, r, x_dim, u_dim, device, args, param_dict)
	# objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args)
	# reg_fn = Regularizer(phi_fn, device, reg_weight=args.reg_weight)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)
	# objective_fn = objective_fn.to(device)
	# reg_fn = reg_fn.to(device)

	# print("Phi param before load:")
	# print("k0, ci: ", phi_fn.k0, phi_fn.ci)
	#
	# if exp_name:
	# 	assert checkpoint_number is not None
	# 	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	# 	load_model(phi_fn, phi_load_fpth)
	# print("Phi param after load:")
	# print("k0, ci: ", phi_fn.k0, phi_fn.ci)

	load_model(phi_fn, phi_load_fpth)

	# for name, param_info in phi_fn.named_parameters():
	# 	print(name)
	# 	print(param_info)
	# IPython.embed()

	# return phi_fn, param_dict
	return phi_fn