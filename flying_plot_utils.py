import numpy as np
import IPython
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import math
from main import Phi, Objective, Regularizer
from src.argument import parser, print_args
from src.utils import *
import torch
import pickle
# from phi_baseline import PhiBaseline
from src.attacks.gradient_batch_attacker import GradientBatchAttacker
from src.attacks.gradient_batch_attacker_warmstart import GradientBatchWarmstartAttacker
from src.reg_sample_keeper import RegSampleKeeper

import time
from torch.autograd import grad
from torch import nn
from src.argument import parser, print_args

# Make numpy and torch deterministic
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cpu")

thresh = np.array([math.pi, math.pi / 2, math.pi / 2, 2, 2, 2, math.pi / 2, math.pi / 2, 2, 2], dtype=np.float32)
x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)

def load_phi():
	# IPython.embed()
	args = parser()

	param_dict = {
		"m": 0.8,
		"J_x": 0.005,
		"J_y": 0.005,
		"J_z": 0.009,
		"l": 1.5,
		"k1": 4.0,
		"k2": 0.05,
		"m_p": 0.04,
		"L_p": 0.03,
		'delta_safety_limit': math.pi / 4  # in radians; should be <= math.pi/4
	}
	param_dict["M"] = param_dict["m"] + param_dict["m_p"]
	state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
	                     "dtheta"]  # excluded x, y, z
	state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))

	r = 2
	x_dim = len(state_index_names)
	u_dim = 4

	# Save stuff in param dict
	param_dict["state_index_dict"] = state_index_dict
	param_dict["r"] = r
	param_dict["x_dim"] = x_dim
	param_dict["u_dim"] = u_dim
	param_dict["x_lim"] = x_lim

	# Create phi
	from src.problems.flying_inv_pend import H, XDot, ULimitSetVertices
	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict, device)
	# uvertices_fn = ULimitSetVertices(param_dict, device)

	x_e = torch.zeros(1, x_dim) # TODO: assume that we include origin

	h_fn = h_fn.to(device)
	xdot_fn = xdot_fn.to(device)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	phi_fn = phi_fn.to(device)

	#####################################################################
	# In a loop, plot 2D slices
	# logger = create_logger("./log/discard", 'train', 'info')
	# x_lim_torch = torch.tensor(x_lim).to(device)
	# attacker = GradientBatchWarmstartAttacker(x_lim_torch, device, logger, n_samples=args.train_attacker_n_samples,
	#                                           stopping_condition=args.train_attacker_stopping_condition,
	#                                           max_n_steps=args.train_attacker_max_n_steps, lr=args.train_attacker_lr,
	#                                           projection_tolerance=args.train_attacker_projection_tolerance,
	#                                           projection_lr=args.train_attacker_projection_lr)

	return phi_fn, param_dict

def plot_boundary_samples(phi_fn, samples, param_dict, which_params=None, fnm=None):
	"""
	Plots invariant set and projected boundary samples in 2D
	which_params: all or list of lists of length 2
	"""
	x_lim = param_dict["x_lim"]
	x_dim = param_dict["x_dim"]
	state_index_dict = param_dict["state_index_dict"]

	if which_params is None:
		params_to_viz = [["theta", "phi"], ["dtheta", "dphi"], ["theta", "dtheta"], ["phi", "dphi"], ["beta", "alpha"], ["gamma", "beta"], ["gamma", "alpha"], ["dbeta", "dalpha"], ["dgamma", "dbeta"], ["dgamma", "dalpha"]]
	else:
		params_to_viz = which_params

	# IPython.embed()
	n_per_row = 3
	n_row = math.ceil(len(params_to_viz)/float(n_per_row))

	n_per_row = min(len(params_to_viz), n_per_row)
	fig, axs = plt.subplots(n_row, n_per_row, squeeze=False)
	# IPython.embed()
	# axs[0, 0].plot(x, y)
	for i in range(n_row):
		for j in range(n_per_row):
			if i*n_per_row + j >= len(params_to_viz):
				break
			# for param_pair in params_to_viz:
			param_pair = params_to_viz[i*n_per_row + j]
			param1, param2 = param_pair

			ind1 = state_index_dict[param1]
			ind2 = state_index_dict[param2]

			delta = 0.01
			x = np.arange(x_lim[ind1, 0], x_lim[ind1, 1], delta)
			y = np.arange(x_lim[ind2, 0], x_lim[ind2, 1], delta)[::-1] # need to reverse it # TODO
			X, Y = np.meshgrid(x, y)

			###################################
			##### Plotting ######
			###################################
			## Get phi values
			input = np.zeros((X.size, x_dim))
			input[:, ind1] = X.flatten()
			input[:, ind2] = Y.flatten()

			batch_size = int(X.size/5)
			all_size = input.shape[0]

			phi_vals = []
			for k in range(math.ceil(all_size/batch_size)):
				batch_input = input[k*batch_size: min(all_size, (k+1)*batch_size)]
				batch_input = batch_input.astype("float32")
				batch_input_torch = torch.from_numpy(batch_input)

				batch_phi_vals = phi_fn(batch_input_torch)
				phi_vals.append(batch_phi_vals.detach().cpu().numpy())

			## Process phi values
			phi_vals = np.concatenate((phi_vals), axis=0)
			S_vals = np.max(phi_vals, axis=1)  # S = all phi_i <= 0
			phi_signs = np.sign(S_vals)
			phi_signs = np.reshape(phi_signs, X.shape)

			# fig = plt.figure()
			# ax = fig.add_subplot(111)
			# IPython.embed()
			axs[i, j].imshow(phi_signs, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
			axs[i, j].set_aspect("equal")
			# phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
			axs[i, j].contour(X, Y, np.reshape(phi_vals[:, -1], X.shape), levels=[0.0],
			                 colors=('k',), linewidths=(2,))


			## Plotting the sampled points
			axs[i, j].scatter(samples[:, ind1], samples[:, ind2], s=0.5) # projection (truncation)

			## Title
			# title = "%s vs. %s" % (param1, param2)
			axs[i, j].set_xlabel(param1)
			axs[i, j].set_ylabel(param2)

	if fnm is None:
		fnm = time.strftime('%m_%d_%H:%M:%S')
	save_fpth = "./log/boundary_sampling/regularization/%s.png" % fnm
	print("Saved at: %s" % save_fpth)
	plt.tight_layout(pad=0.5)
	plt.savefig(save_fpth, bbox_inches='tight')
	plt.clf()
	plt.close()


if __name__ == "__main__":
	phi_fn, param_dict = load_phi()
	# IPython.embed()

	# Make attacker and viz
	logger = create_logger("./log/discard", 'train', 'info')
	x_lim_torch = torch.tensor(x_lim).to(device)
	args = parser()
	# attacker = GradientBatchWarmstartAttacker(x_lim_torch, device, logger, n_samples=args.train_attacker_n_samples,
	#                                           stopping_condition=args.train_attacker_stopping_condition,
	#                                           max_n_steps=args.train_attacker_max_n_steps, lr=args.train_attacker_lr,
	#                                           projection_tolerance=args.train_attacker_projection_tolerance,
	#                                           projection_lr=args.train_attacker_projection_lr)
	# n_samples = 50
	# samples = attacker._sample_points_on_boundary(phi_fn, n_samples)


	# Make regularizer and viz
	reg_sample_keeper = RegSampleKeeper(x_lim_torch, device, logger, n_samples=30)
	samples = reg_sample_keeper.return_samples(phi_fn)

	# plot_boundary_samples(phi_fn, samples, param_dict, which_params=[["phi", "theta"], ["theta", "dtheta"], ["phi", "dphi"], ["beta", "alpha"], ["gamma", "beta"], ["gamma", "alpha"]])
	plot_boundary_samples(phi_fn, samples, param_dict, fnm="n30")

