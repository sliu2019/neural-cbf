import numpy as np
import IPython
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import math
from main import Phi, Objective, Regularizer
# from src.argument import parser, print_args
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
# from src.argument import parser, print_args

# Make numpy and torch deterministic (for rand phi and attack/reg sampling)
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cpu")

def load_phi_and_params(exp_name=None, checkpoint_number=None):
	if exp_name:
		args = load_args("./log/%s/args.txt" % exp_name)
		param_dict = pickle.load(open("./log/%s/param_dict.pkl" % exp_name, "rb"))
	else:
		from src.argument import parser
		args = parser() # default

		from main import create_flying_param_dict
		param_dict = create_flying_param_dict() # default

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

	A_samples = None
	if args.phi_include_xe:
		x_e = torch.zeros(1, x_dim)
	else:
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

	# Create CBF, etc.
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)
	# objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger)
	# reg_fn = Regularizer(phi_fn, device, reg_weight=args.reg_weight, A_samples=A_samples)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)
	# objective_fn = objective_fn.to(device)
	# reg_fn = reg_fn.to(device)

	print("Phi param before load:")
	# print(list(phi_fn.parameters())[0])
	print("k0, ci: ", phi_fn.k0, phi_fn.ci)

	if exp_name:
		assert checkpoint_number is not None
		phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
		load_model(phi_fn, phi_load_fpth)
	print("Phi param after load:")
	# print(list(phi_fn.parameters())[0])

	print("k0, ci: ", phi_fn.k0, phi_fn.ci)

	# for name, param_info in phi_fn.named_parameters():
	# 	print(name)
	# 	print(param_info)
	# IPython.embed()

	return phi_fn, param_dict


def plot_invariant_set_slices(phi_fn, param_dict, samples=None, rollouts=None, which_params=None, constants_for_other_params=None, fnm=None, fldr_path=None):
	"""
	Plots invariant set and (if necessary) projected boundary samples in 2D
	which_params: all or list of lists of length 2
	"""
	if rollouts is not None:
		print("Inside plot_invariant_set_slices() of fling_plot_utils")
		print("Have never debugged traj plotting using this")
		IPython.embed()

	x_lim = param_dict["x_lim"]
	x_dim = param_dict["x_dim"]
	state_index_dict = param_dict["state_index_dict"]

	if which_params is None:
		# params_to_viz = [["theta", "phi"], ["dtheta", "dphi"], ["theta", "dtheta"], ["phi", "dphi"], ["beta", "alpha"], ["gamma", "beta"], ["gamma", "alpha"], ["dbeta", "dalpha"], ["dgamma", "dbeta"], ["dgamma", "dalpha"]]
		# Default is intelligent plotting
		constants_for_other_params = []
		params_to_viz = []

		# Test
		params_to_viz.append(["gamma", "phi"])
		constants_for_other_params.append(np.zeros(16))

		# Test 2: gamma v phi
		params_to_viz.append(["gamma", "phi"])
		constants_for_other_params.append(np.zeros(16))


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
			if samples is not None:
				axs[i, j].scatter(samples[:, ind1], samples[:, ind2], s=0.5) # projection (truncation)


			## Plotting the included trajectories
			if rollouts is not None:
				N_rollout = len(rollouts) # rollouts is a ist
				for i in range(N_rollout):
					ith_rl = rollouts[i]
					axs[i, j].plot(ith_rl[:, ind1], ith_rl[:, ind2])

			## Title
			# title = "%s vs. %s" % (param1, param2)
			axs[i, j].set_xlabel(param1)
			axs[i, j].set_ylabel(param2)

	if fnm is None:
		fnm = time.strftime('%m_%d_%H:%M:%S')
	if fldr_path is None:
			fldr_path = "./log/boundary_sampling"

	# if fpth is not None:
	# 	save_fpth = "./log/%s/%s.png" % (fpth, fnm)
	# else:
	# 	save_fpth = "./log/boundary_sampling/%s.png" % fnm
	save_fpth = os.path.join(fldr_path, fnm + ".png")

	print("Saved at: %s" % save_fpth)
	plt.tight_layout(pad=0.5)
	plt.title("From %s" % fldr_path)
	plt.savefig(save_fpth, bbox_inches='tight')
	# plt.clf()
	# plt.close()

	return fig, axs

def load_attacks(exp_name, checkpoint_number):
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)

		# attacks_init = data["train_attack_X_init"][checkpoint_number+1]
		attacks = data["train_attack_X_final"][checkpoint_number+1]
		# best_attack = data["train_attacks"][checkpoint_number+1]

	return attacks

def graph_losses(exp_name):
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)

		train_attack_losses = data["train_attack_losses"]
		train_reg_losses = data["train_reg_losses"]
		train_losses = data["train_losses"]

		plt.plot(train_attack_losses[:1500], linewidth=0.5, label="train attack loss")
		plt.plot(train_reg_losses[:1500], linewidth=0.5, label="train reg loss")
		plt.plot(train_losses[:1500], linewidth=0.5, label="train total loss")
		plt.title("Losses for %s" % exp_name)

	plt.xlabel("Optimization steps")
	plt.legend(loc="upper right")


	plt.savefig("./log/%s/%s_loss.png" % (exp_name, exp_name))
	plt.clf()
	plt.cla()

	print(exp_name)
	print("Min attack loss achieved (desired <= 0): %.5f" % np.min(train_attack_losses))
	print("Min overall loss %.3f at checkpoint %i" % (np.min(train_losses), np.argmin(train_losses)))

def plot_cbf_3d_slices(phi_fn, param_dict, which_params = None, fnm = None, fpth = None):
	"""
	Plots CBF value against 2 inputs, for a total of 3 dimensions (value, input1, input2)
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
	# fig, axs = plt.subplots(n_row, n_per_row, squeeze=False, projection="3d")
	fig = plt.figure()
	# IPython.embed()
	# axs[0, 0].plot(x, y)
	for i in range(n_row):
		for j in range(n_per_row):
			if i*n_per_row + j >= len(params_to_viz):
				break

			# Create 3D axis on fig
			ax = fig.add_subplot(n_row, n_per_row, (i*n_per_row + j) + 1, projection='3d') # 3rd argument: using 1-indexing

			# for param_pair in params_to_viz:
			param_pair = params_to_viz[i*n_per_row + j]
			param1, param2 = param_pair

			ind1 = state_index_dict[param1]
			ind2 = state_index_dict[param2]

			delta = 0.01 # larger for 3D plotting, due to latency
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
			last_phi_vals = phi_vals[:, -1]
			Z = np.reshape(last_phi_vals, X.shape)

			## Finally, plot
			# IPython.embed()
			ax.set_xlim(x_lim[ind1, 0], x_lim[ind1, 1])
			ax.set_ylim(x_lim[ind2, 0], x_lim[ind2, 1])

			# ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1,
			#                 alpha=0.75)
			ax.plot_surface(X, Y, Z, cmap="autumn_r", linewidth = 0, antialiased = False)
			# ax.contour(X, Y, Z, 10, lw=3, colors="k", linestyles="solid")

			print("CBF min, max: %.3f, %.3f" % (np.min(Z), np.max(Z)))
			# IPython.embed()
			## Title
			ax.set_xlabel(param1)
			ax.set_ylabel(param2)

	if fnm is None:
		fnm = time.strftime('%m_%d_%H:%M:%S')

	if fpth is not None:
		save_fpth = "./log/%s/%s.png" % (fpth, fnm)
	else:
		save_fpth = "./log/flying_3d_viz/%s.png" % fnm

	print("Saved at: %s" % save_fpth)
	plt.tight_layout(pad=0.5)
	# plt.show()
	plt.savefig(save_fpth, bbox_inches='tight')
	plt.clf()
	plt.close()

if __name__ == "__main__":
	"""
	Code to visualize samples (from RegSampleKeeper or Attacker)
	phi_fn, param_dict = load_phi_and_params()

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

	# plot_invariant_set_slices(phi_fn, samples, param_dict, which_params=[["phi", "theta"], ["theta", "dtheta"], ["phi", "dphi"], ["beta", "alpha"], ["gamma", "beta"], ["gamma", "alpha"]])
	plot_invariant_set_slices(phi_fn, param_dict, samples, fnm="n30")
	"""

	########################################################
	#########     FILL OUT HERE !!!!   #####################
	### ****************************************************

	exp_names = ["flying_inv_pend_pend_5_beta_iterated", "flying_inv_pend_pend_3_beta_iterated_nn_128_128", "flying_inv_pend_pend_5_beta_flat_nn_128_128", "flying_inv_pend_pend_3_beta_iterated"]
	# checkpoint_numbers = [500, 1800, 800, 1800]
	checkpoint_numbers = [1420, 1450, 430, 1370]
	### ****************************************************
	########################################################
	for exp_name, checkpoint_number in zip(exp_names, checkpoint_numbers):
		graph_losses(exp_name)
		plt.clf()
		plt.close()

		# phi_fn, param_dict = load_phi_and_params(exp_name, checkpoint_number)
		#
		# samples = load_attacks(exp_name, checkpoint_number)
		#
		# fldr_path = os.path.join("./log", exp_name)
		# plot_invariant_set_slices(phi_fn, param_dict, fldr_path=fldr_path, fnm="viz_invar_set_ckpt_%i" % checkpoint_number)
		#
		# plt.clf()
		# plt.close()
		#
		# plot_invariant_set_slices(phi_fn, param_dict, samples=samples, fldr_path=fldr_path, fnm="viz_attacks_ckpt_%i" % checkpoint_number)
		#
		# # plot_cbf_3d_slices(phi_fn, param_dict, which_params = [["phi", "theta"]], fnm = "3d_viz_ckpt_%i" % checkpoint_number, fpth = exp_name)
		#
		# plt.clf()
		# plt.close()

		#################################################
		#### Graphing other debug info ##################

		# with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		# 	data = pickle.load(handle)
		#
		# 	train_attack_losses = data["train_attack_losses"]
		# 	train_reg_losses = data["train_reg_losses"]
		# 	train_losses = data["train_losses"]
		#
		# 	plt.plot(train_attack_losses[:1000], linewidth=0.5, label="train attack loss")
		# 	plt.plot(train_reg_losses[:1000], linewidth=0.5, label="train reg loss")
		# 	plt.plot(train_losses[:1000], linewidth=0.5, label="train total loss")
		# 	plt.title("Losses for %s" % exp_name)
		#
		# plt.xlabel("Optimization steps")
		# plt.legend(loc="upper right")
		#
		# plt.savefig("./log/%s/%s_loss.png" % (exp_name, exp_name))
		# plt.clf()
		# plt.cla()
		#
		# print("Min attack loss achieved (desired <= 0): %.5f" % np.min(train_attack_losses))
		# print("Min overall loss %.3f at checkpoint %i" % (np.min(train_losses), np.argmin(train_losses)))
		#
		# save_dict = {"test_losses": test_losses, "test_attack_losses": test_attack_losses,
		#              "test_reg_losses": test_reg_losses, "train_loop_times": train_loop_times,
		#              "train_attacks": train_attacks, "train_attack_X_init": train_attack_X_init,
		#              "train_attack_X_final": train_attack_X_final, "k0_grad": k0_grad, "ci_grad": ci_grad,
		#              "train_losses": train_losses, "train_attack_losses": train_attack_losses,
		#              "train_reg_losses": train_reg_losses, "train_attack_X_obj_vals": train_attack_X_obj_vals,
		#              "grad_norms": grad_norms, "reg_sample_keeper_X": reg_sample_keeper_X}
		#
		# additional_train_attack_dict = {"train_attack_X_init_reuse": train_attack_X_init_reuse,
		#                                 "train_attack_X_init_random": train_attack_X_init_random,
		#                                 "train_attack_init_best_attack_value": train_attack_init_best_attack_value,
		#                                 "train_attack_final_best_attack_value": train_attack_final_best_attack_value,
		#                                 "train_attack_t_init": train_attack_t_init,
		#                                 "train_attack_t_grad_steps": train_attack_t_grad_steps,
		#                                 "train_attack_t_reproject": train_attack_t_reproject,
		#                                 "train_attack_t_total_opt": train_attack_t_total_opt}
		#
		# reg_debug_dict = {"max_dists_X_reg": max_dists_X_reg, "times_to_compute_X_reg": times_to_compute_X_reg}