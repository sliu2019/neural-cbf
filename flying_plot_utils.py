import os

import matplotlib.pyplot as plt
import math
from dotmap import DotMap
import torch
import pickle
import time, os, glob, re
import numpy as np

from main import Phi, Objective, Regularizer
from src.utils import *

# Make numpy and torch deterministic (for rand phi and attack/reg sampling)
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cpu")

def load_phi_and_params(exp_name=None, checkpoint_number=None):
	if exp_name:
		fnm = "./log/%s/args.txt" % exp_name
		# args = load_args(fnm) # can't use, args conflicts with args in outer scope
		with open(fnm, 'r') as f:
			json_data = json.load(f)
		args = DotMap(json_data)
		param_dict = pickle.load(open("./log/%s/param_dict.pkl" % exp_name, "rb"))
	else:
		from src.argument import create_parser
		parser = create_parser() # default
		args = parser.parse_known_args()[0]

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
	uvertices_fn = ULimitSetVertices(param_dict, device)

	# reg_sampler = reg_samplers_name_to_class_dict[args.reg_sampler](x_lim, device, logger, n_samples=args.reg_n_samples)

	if args.phi_include_xe:
		x_e = torch.zeros(1, x_dim)
	else:
		x_e = None

	# Passing in subset of state to NN
	from src.utils import IndexNNInput, TransformEucNNInput
	state_index_dict = param_dict["state_index_dict"]
	if args.phi_nn_inputs == "spherical":
		nn_input_modifier = None
	elif args.phi_nn_inputs == "euc":
		nn_input_modifier = TransformEucNNInput(state_index_dict)

	# Send all modules to the correct device
	h_fn = h_fn.to(device)
	xdot_fn = xdot_fn.to(device)
	uvertices_fn = uvertices_fn.to(device)
	if x_e is not None:
		x_e = x_e.to(device)
	# x_lim = torch.tensor(x_lim).to(device)

	# Create CBF, etc.
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e, nn_input_modifier=nn_input_modifier)
	# objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args)
	# reg_fn = Regularizer(phi_fn, device, reg_weight=args.reg_weight)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)
	# objective_fn = objective_fn.to(device)
	# reg_fn = reg_fn.to(device)

	print("Phi param before load:")
	print("k0, ci: ", phi_fn.k0, phi_fn.ci)

	if exp_name:
		assert checkpoint_number is not None
		phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
		load_model(phi_fn, phi_load_fpth)
	print("Phi param after load:")
	print("k0, ci: ", phi_fn.k0, phi_fn.ci)

	# for name, param_info in phi_fn.named_parameters():
	# 	print(name)
	# 	print(param_info)
	# IPython.embed()

	return phi_fn, param_dict

def load_attacks(exp_name, checkpoint_number):
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)

		# attacks_init = data["train_attack_X_init"][checkpoint_number+1]
		attacks = data["train_attack_X_final"][checkpoint_number+1]
		# best_attack = data["train_attacks"][checkpoint_number+1]

	return attacks

def graph_losses(exp_name, debug=True):
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)

	args = load_args("./log/%s/args.txt" % exp_name)

	# IPython.embed()
	suptitle_size = 8 # default is 10
	n_it_so_far = len(data["train_losses"]) - 1
	fig, axs = plt.subplots(2, sharex=True)
	fig.suptitle('Metrics for %s at iteration %i/%i' % (exp_name, n_it_so_far, args.trainer_n_steps), fontsize=suptitle_size)

	#############################################
	axs[0].set_title("Train metrics")
	axs[0].plot(data["train_losses"], label="objective value", linewidth=0.5)
	axs[0].plot(data["train_attack_losses"], label="worst attack value", linewidth=0.5)
	axs[0].plot(data["train_reg_losses"], label="reg value", linewidth=0.5)
	axs[0].plot([0, n_it_so_far], [0, 0], color="k", linewidth=0.5)

	axs[0].legend(loc=(1.04,0))
	#############################################
	axs[1].set_title("Test metrics")
	per_n_iterations = args.n_test_loss_step
	test_iterations = np.arange(0, n_it_so_far, per_n_iterations)

	axs[1].plot(test_iterations, data["V_approx_list"], label="approx. volume", color="green", linewidth=1.0)

	boundary_samples_obj_values = data["boundary_samples_obj_values"]
	percent_infeas_at_boundary = [np.sum(boundary_samples_obj_value > 0) * 100 / boundary_samples_obj_value.size for boundary_samples_obj_value in boundary_samples_obj_values]
	infeas_values_list = [(boundary_samples_obj_value > 0) * boundary_samples_obj_value for boundary_samples_obj_value in boundary_samples_obj_values]
	average_infeas_amount = [np.mean(infeas_values) for infeas_values in infeas_values_list]
	average_infeas_amount = np.array(average_infeas_amount)
	std_infeas_amount = [np.std(infeas_values) for infeas_values in infeas_values_list]
	std_infeas_amount = np.array(std_infeas_amount)

	max_infeas_amount = [np.max(infeas_values) for infeas_values in infeas_values_list]

	axs[1].plot(test_iterations, percent_infeas_at_boundary, label="% infeas. at boundary", color="blue", linewidth=1.0)
	axs[1].fill_between(test_iterations, average_infeas_amount - std_infeas_amount, average_infeas_amount + std_infeas_amount, alpha=0.5, color="orange")
	axs[1].plot(test_iterations, average_infeas_amount, label="average infeas.", linewidth=0.5, color="orange")

	# axs[1].plot(test_iterations, max_infeas_amount, label="max infeas.", linewidth=0.5)

	axs[1].set_ylim([-5, 40])
	axs[1].legend(loc=(1.04,0))

	fig.tight_layout()
	plt.xlabel("Iterations") # aka opt. steps

	plt.savefig("./log/%s/%s_loss.png" % (exp_name, exp_name))
	plt.clf()
	plt.cla()

	#############################################
	#############################################
	if debug == True:
		# IPython.embed()
		###################################
		########## Timing debug ###########
		###################################
		n_it_so_far = len(data["train_losses"])
		fig, axs = plt.subplots(3, sharex=True)
		fig.suptitle('Timing debug for %s at iteration %i/%i' % (exp_name, n_it_so_far, args.trainer_n_steps), fontsize=suptitle_size)

		#############################################
		axs[0].set_title("Timing metrics")
		axs[0].plot(test_iterations, data["test_t_total"], label="Time to compute test metric", linewidth=0.5)
		axs[0].legend(loc=(1.04,0))

		#############################################
		axs[1].set_title("Attack timing metrics")
		axs[1].plot(data["train_attack_t_total_opt"], label="Total time for counterex", linewidth=0.5)
		axs[1].plot(data["train_attack_t_init"], label="Time to init", linewidth=0.5)
		train_attack_t_grad_steps = data["train_attack_t_grad_steps"]
		avg_t_grad_steps = [np.mean(x) for x in train_attack_t_grad_steps]
		axs[1].plot(avg_t_grad_steps, label="Avg time for grad steps", linewidth=0.5)
		train_attack_t_reproject = data["train_attack_t_reproject"]
		avg_t_reproject = [np.mean(x) for x in train_attack_t_reproject]
		axs[1].plot(avg_t_reproject, label="Avg time to reproject", linewidth=0.5)

		axs[1].legend(loc=(1.04,0))
		#############################################
		axs[2].set_title("Attack other metrics")
		axs[2].plot(data["train_attack_n_segments_sampled"], label="N seg sampled", linewidth=0.5)
		axs[2].plot(data["train_attack_n_opt_steps"], label="N opt steps", linewidth=0.5)
		axs[2].legend(loc=(1.04,0))

		fig.tight_layout()
		plt.xlabel("Iterations") # aka opt. steps
		plt.savefig("./log/%s/%s_timing_debug.png" % (exp_name, exp_name))
		plt.clf()
		plt.cla()

		###################################
		########## Other debug ###########
		###################################
		fig, axs = plt.subplots(4, sharex=True)
		fig.suptitle('Debug for %s at iteration %i/%i' % (exp_name, n_it_so_far, args.trainer_n_steps), fontsize=suptitle_size)

		#############################################
		axs[0].set_title("Attack improvement after opt.")
		diff = np.array(data["train_attack_final_best_attack_value"]) - np.array(data["train_attack_init_best_attack_value"])
		axs[0].plot(diff, linewidth=0.5)
		axs[0].plot([0, n_it_so_far], [0, 0], c="red", linewidth=0.5)

		#############################################
		axs[1].set_title("Attack constraint improv. after reproj.")
		dist_after_reproj = [np.mean(x) for x in data["train_attack_dist_diff_after_proj"]]
		axs[1].plot(dist_after_reproj, linewidth=0.5)

		#############################################
		axs[2].set_title("CBF gradient magnitudes")
		axs[2].plot(data["reg_grad_norms"], linewidth=0.5, label="From reg loss")
		axs[2].plot(data["grad_norms"], linewidth=0.5, label="From combined loss")
		axs[2].legend(loc=(1.04,0))

		#############################################
		axs[3].set_title("k0, ci over iterations")
		k0_list = [x.item() for x in data['k0_list']]
		ci_list = [x.item() for x in data["ci_list"]]
		axs[3].plot(k0_list, linewidth=0.5, label="k0")
		axs[3].plot(ci_list, linewidth=0.5, label="k1")
		axs[3].legend(loc=(1.04,0))

		#############################################
		fig.tight_layout()

		plt.xlabel("Iterations") # aka opt. steps

		plt.savefig("./log/%s/%s_debug.png" % (exp_name, exp_name))
		plt.clf()
		plt.cla()

		#############################################
		########## Grad mag detailed plot ###########
		#############################################
		plt.title("CBF gradient magnitudes")
		plt.plot(data["reg_grad_norms"], linewidth=0.5, label="From reg loss")
		plt.plot(data["grad_norms"], linewidth=0.5, label="From combined loss")
		plt.legend(loc=(1.04,0))
		plt.xlabel("Iterations") # aka opt. steps

		plt.savefig("./log/%s/%s_grad_norms.png" % (exp_name, exp_name))
		plt.clf()
		plt.cla()

	#############################################
	train_attack_losses = np.array(data["train_attack_losses"])
	approx_v = np.array(data["V_approx_list"])
	N_it = len(train_attack_losses)
	n_checkpoint_step = args.n_checkpoint_step

	print("For experiment %s, at iteration %i/%i" % (exp_name, n_it_so_far, args.trainer_n_steps))
	min_attack_ind = np.argmin(train_attack_losses)

	# print("Total iterations (so far): %i" % N_it)
	print("Average approx volume: %.3f" % (np.mean(approx_v)))

	# IPython.embed()
	# Out[24]: dict_keys(['test_losses', 'test_attack_losses', 'test_reg_losses', 'k0_grad', 'ci_grad', 'train_loop_times', 'train_losses', 'train_attack_losses', 'train_reg_losses', 'grad_norms', 'V_approx_list', 'boundary_samples_obj_values', 'ci_list', 'k0_list', 'train_attacks', 'train_attack_X_init', 'train_attack_X_init_reuse', 'train_attack_X_init_random', 'train_attack_X_final', 'train_attack_X_obj_vals', 'train_attack_X_phi_vals', 'train_attack_init_best_attack_value', 'train_attack_final_best_attack_value', 'train_attack_t_init', 'train_attack_t_grad_steps', 'train_attack_t_reproject', 'train_attack_t_total_opt', 'train_attack_diff_after_proj', 'train_attack_t_sample_boundary', 'train_attack_n_segments_sampled', 'train_attack_dist_diff_after_proj', 'reg_grad_norms'])

	print(np.min(train_attack_losses))

	# IPython.embed()
	# Answers the question: does the sawtoothing help?
	trunc_v = approx_v[4:]
	low_loss_ind = np.argwhere(train_attack_losses[100:] < 1.5).flatten()

	# v_diff = trunc_v[1:] - trunc_v[:-1]
	print("V avg: %.3f" % (np.mean(trunc_v)))
	# print("V diff: %.3f" % (v_diff[-1] - v_diff[0]))
	print("First: %.3f, best: %.3f" % (trunc_v[0], np.max(trunc_v)))
	# mean_diff = np.mean(v_diff)

	# print("average V diff step-to-step: %.3f +/- %.3f" % (mean_diff, np.std(v_diff)))
	# IPython.embed()
	# print("Min attack loss (desired <= 0): %.5f at checkpoint %i, with volume ~= %.3f" % (np.min(train_attack_losses), min_attack_ind, approx_v[round(min_attack_ind/float(args.n_checkpoint_step))]))
	# min_attack_ind_w_checkpoint = np.argmin(np.array(train_attack_losses)[::n_checkpoint_step])*n_checkpoint_step
	# print("Min attack loss (desired <= 0): %.5f at checkpoint %i, with volume ~= %.3f" % (train_attack_losses[min_attack_ind_w_checkpoint], min_attack_ind_w_checkpoint, approx_v[int(min_attack_ind_w_checkpoint/n_checkpoint_step)]))
	# # print("Min overall loss %.3f at checkpoint %i" % (np.min(train_losses), np.argmin(train_losses)))
	# # checkpoint_ind = min_attack_ind_w_checkpoint # TODO
	#
	# # Trying different tactics to select a training iteration
	#
	# # Balancing volume and train loss
	# train_attack_losses_at_checkpoints = train_attack_losses[::n_checkpoint_step]
	# m = len(train_attack_losses_at_checkpoints)
	# ind_sort_train_loss = np.argsort(train_attack_losses_at_checkpoints)
	# rank_train_loss = np.zeros(m)
	# rank_train_loss[ind_sort_train_loss] = np.arange(m)
	# rank_train_loss = rank_train_loss.astype(int)
	#
	# ind_sort_volume = np.argsort(-approx_v)
	# rank_volume = np.zeros(m)
	# rank_volume[ind_sort_volume] = np.arange(m)
	# rank_volume = rank_volume.astype(int)
	#
	# rank_sum = rank_volume + rank_train_loss # TODO: checkpoint selection criteria
	# # rank_sum = rank_train_loss
	# best_balanced_ind = np.argmin(rank_sum)
	#
	# checkpoint_ind = best_balanced_ind*n_checkpoint_step
	#
	# # IPython.embed()
	# # print(train_attack_losses)
	# # IPython.embed()
	# # return min_attack_ind_w_checkpoint # cause we're loading this checkpoint
	#
	# checkpoint_ind = int(checkpoint_ind)
	# print("At selected checkpoint %i: %.3f loss, %.3f volume" % (checkpoint_ind, train_attack_losses[checkpoint_ind], approx_v[int(checkpoint_ind/n_checkpoint_step)]))
	#
	# print("\n")
	# return checkpoint_ind

	return None

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

def plot_invariant_set_slices(phi_fn, param_dict, samples=None, rollouts=None, which_params=None, constants_for_other_params=None, fnm=None, fldr_path=None, checkpoint=None):
	"""
	Plots invariant set and (if necessary) projected boundary samples in 2D
	which_params: all or list of lists of length 2
	"""
	# IPython.embed()
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

		# Test: does it matter if pend angle and pend velocity are aligned?
		params_to_viz.extend([["phi", "dphi"], ["theta", "dtheta"], ["gamma", "dgamma"], ["beta", "dbeta"]])
		constants_for_other_params.extend([np.zeros(10)]*4)

		# Test: does it matter if pend and quad angle are aligned/not aligned?
		params_to_viz.append(["gamma", "phi"])
		constants_for_other_params.append(np.zeros(10))

		# More dangerous
		params_to_viz.append(["gamma", "phi"])
		x = np.zeros(10)
		x[state_index_dict["dgamma"]] = 5
		constants_for_other_params.append(x)

		# params_to_viz.append(["gamma", "phi"])
		# x = np.zeros(10)
		# x[state_index_dict["dphi"]] = 5
		# constants_for_other_params.append(x)

		params_to_viz.append(["beta", "theta"])
		constants_for_other_params.append(np.zeros(10))

		# More dangerous
		params_to_viz.append(["beta", "theta"])
		x = np.zeros(10)
		x[state_index_dict["dbeta"]] = 5
		constants_for_other_params.append(x)
	else:
		params_to_viz = which_params

	# IPython.embed()
	n_per_row = 2 # TODO
	n_row = math.ceil(len(params_to_viz)/float(n_per_row))

	n_per_row = min(len(params_to_viz), n_per_row)
	fig, axs = plt.subplots(n_row, n_per_row, squeeze=False, figsize=(6, 16)) # TODO: remove figsize
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
			if constants_for_other_params:
				# print("flying_plot_utils.py, ln 162")
				# IPython.embed()
				input = constants_for_other_params[i*n_per_row + j]
				# print(input)
				input = np.reshape(input, (1, -1))
				input = np.tile(input, (X.size, 1))
			else:
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
			# print(phi_signs)
			print("Any negative phi in the box?: ", np.any(phi_signs < 0))
			# IPython.embed()
			axs[i, j].imshow(phi_signs, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]], vmin=-1.0, vmax=1.0)
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
			if constants_for_other_params:
				const = constants_for_other_params[i * n_per_row + j]
				axs[i, j].set_title(const)

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
	# plt.tight_layout(pad=0.5)

	ki_str = "k0 = %.4f, k1 = %.4f" % (phi_fn.k0, phi_fn.ci[0])
	# fig.title(ki_str) # doesn't work, title goes on last subfigure
	if checkpoint: # little messy, little hacky, doesn't matter tho
		plt.suptitle("From %s, ckpt %i \n %s" % (fldr_path, checkpoint, ki_str))
	else:
		plt.suptitle("From %s \n %s" % (fldr_path, ki_str))

	plt.savefig(save_fpth, bbox_inches='tight')
	# plt.clf()
	# plt.close()

	return fig, axs

def debug(exp_name):
	"""
	Debugging:
	- train loss spikes
	- counterexamples (are they consistently close to the boundary? are they on dS, as well as phi(x) = 0?)
	- regularization (is reg grad magnitude proportional to volume?)
	"""
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)

		# IPython.embed()
		train_attack_losses = np.array(data["train_attack_losses"])
		train_attack_X_obj_vals = data["train_attack_X_obj_vals"]
		train_attack_X_phi_vals = data["train_attack_X_phi_vals"]
		# train_attack_X_phi_vals = [np.reshape(x, (x.shape[0], 1)) for x in train_attack_X_phi_vals] # TODO
		train_attack_X_init_random = data["train_attack_X_init_random"]
		train_attack_X_init_reuse = data["train_attack_X_init_reuse"]

		# Debug train loss spikes
		std = np.std(train_attack_losses)
		diff = train_attack_losses[1:] - train_attack_losses[:-1]
		spike_inds = np.argwhere(diff > 2*std).flatten() + 1 # TODO: using 2 std dev to define spikes here; 68=95=99 std dev confidence interval rule
		spike_bools = diff > 2*std # 1 index before the spike happens

		n_random_best = 0
		n_random = train_attack_X_init_random[1].shape[0]
		# print(n_random)
		for spike_ind in spike_inds:
			obj_vals = train_attack_X_obj_vals[spike_ind]
			# print(obj_vals.shape)
			max_ind = np.argmax(obj_vals)
			# print(max_ind)
			if max_ind < n_random:
				n_random_best += 1

		# IPython.embed()
		print(exp_name)
		print("We define spikes as outside 2 std dev")
		percent_random_best = n_random_best*100.0/float(max(1, len(spike_inds)))

		print("Percent of spikes caused by newly sampled counterexamples: %.3f" % percent_random_best)

		# Debug counterexamples: are they consistently close to the boundary? Are they on dS, as well as phi(x) = 0?

		# Find ind of best
		ind_best_X_over_rollout = [np.argmax(x) for x in train_attack_X_obj_vals]

		# TODO: replace
		max_dist_from_boundary_over_rollout = [np.max(np.abs(x[:, -1])) for x in train_attack_X_phi_vals] # THIS IS THE PHI=0 boundary that we're talking about here
		# max_dist_from_boundary_over_rollout = [np.max(np.abs(x)) for x in train_attack_X_phi_vals] # THIS IS THE PHI=0 boundary that we're talking about here
		dist_from_boundary_over_rollout_for_X_best = []
		for i, ind in enumerate(ind_best_X_over_rollout):
			dist = np.abs(train_attack_X_phi_vals[i][ind, -1])
			dist_from_boundary_over_rollout_for_X_best.append(dist)

		# Graphing
		plt.plot(max_dist_from_boundary_over_rollout, linewidth=0.5, label="max dist over all counterexamples")
		plt.plot(dist_from_boundary_over_rollout_for_X_best, linewidth=0.5, label="dist for best counterex")
		plt.title("Distances of counterexamples from phi(x) = 0")
		plt.legend(loc="upper right")
		plt.savefig("./log/%s/dist_of_counterexamples_from_boundary.png" % (exp_name))
		plt.clf()

		max_dist_from_dS_over_rollout = [np.max(np.abs(np.max(x, axis=1))) for x in train_attack_X_phi_vals]
		dist_from_dS_over_rollout_for_X_best = []
		for i, ind in enumerate(ind_best_X_over_rollout):
			dist = np.abs(np.max(train_attack_X_phi_vals[i][ind]))
			dist_from_dS_over_rollout_for_X_best.append(dist)

		# IPython.embed()
		# Sanity check:
		# print(np.any(np.array(dist_from_dS_over_rollout_for_X_best) > np.array(max_dist_from_dS_over_rollout)))

		# Graphing
		plt.plot(max_dist_from_dS_over_rollout, linewidth=0.5, label="max dist over all counterexamples")
		plt.plot(dist_from_dS_over_rollout_for_X_best, linewidth=0.5, label="dist for best counterex")
		plt.title("Distances of counterexamples from dS")
		plt.legend(loc="upper right")
		plt.savefig("./log/%s/dist_of_counterexamples_from_dS.png" % (exp_name))
		plt.clf()

		# Find if spikes in dist_from_dS_over_rollout_for_X_best coincide with spikes in train_attack_loss
		dist_from_dS_over_rollout_for_X_best = np.array(dist_from_dS_over_rollout_for_X_best)
		std = np.std(dist_from_dS_over_rollout_for_X_best)
		diff = dist_from_dS_over_rollout_for_X_best[1:] - dist_from_dS_over_rollout_for_X_best[:-1]
		spike_bools_dS = diff > 2*std

		n_overlap = np.sum(spike_bools*spike_bools_dS)
		print("Number of overlaps between dS and train loss spikes: %i" % n_overlap)
		print("As a percentage of n_spikes for train loss: %.3f" % (n_overlap*100.0/max(1, np.sum(spike_bools))))
		# Blurry overlap
		# TODO
		print("\n")

def find_attacker_n_reuse(exp_name):
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)

		train_attack_X_init_reuse = data["train_attack_X_init_reuse"]
		train_attack_X_init_random = data["train_attack_X_init_random"]

		# IPython.embed()

		for i in range(len(train_attack_X_init_reuse)):
			print(train_attack_X_init_reuse[i].shape[0])

def plot_interesting_slices(phi_fn, param_dict, save_fldrpth, checkpoint_number):
	"""
	:param phi_fn:
	:param param_dict:
	:param save_fldrpth:
	:param save_fnm:
	:param checkpoint: Is used for the title!
	:return:
	"""
	print("inside plot_interesting_slices")
	IPython.embed()
	# TODO
	params_to_viz_list = []
	constants_for_other_params_list = []
	fnms = []

	params_to_viz_list.extend([["phi", "dphi"], ["theta", "dtheta"], ["gamma", "dgamma"], ["beta", "dbeta"]])
	constants_for_other_params_list.extend([np.zeros(10)] * 4)
	fnms.extend(["phi_dphi", "theta_dtheta", "gamma_dgamma", "beta_dbeta"])

	# TODO: more slices
	params_to_viz_list.extend([["dtheta", "dbeta"], ["dphi", "dgamma"]])
	constants_for_other_params_list.extend([np.zeros(10)] * 2)
	fnms.extend(["dtheta_dbeta", "dphi_dgamma"])

	state_index_dict = param_dict["state_index_dict"]

	angle = math.pi / 6  # [pi/5-pi/7] is a popular range of choices

	params_to_viz_list.append(["theta", "beta"])  # aligned, misaligned
	x = np.zeros(10)
	x[state_index_dict["phi"]] = -angle
	x[state_index_dict["gamma"]] = angle
	constants_for_other_params_list.append(x)
	fnms.append("theta_beta_misaligned")

	params_to_viz_list.append(["theta", "beta"])  # aligned, misaligned
	x = np.zeros(10)
	x[state_index_dict["phi"]] = angle
	x[state_index_dict["gamma"]] = angle
	constants_for_other_params_list.append(x)
	fnms.append("theta_beta_aligned")

	params_to_viz_list.append(["phi", "gamma"])  # aligned, misaligned
	x = np.zeros(10)
	x[state_index_dict["theta"]] = -angle
	x[state_index_dict["beta"]] = angle
	constants_for_other_params_list.append(x)
	fnms.append("phi_gamma_misaligned")

	params_to_viz_list.append(["phi", "gamma"])  # aligned, misaligned
	x = np.zeros(10)
	x[state_index_dict["theta"]] = angle
	x[state_index_dict["beta"]] = angle
	constants_for_other_params_list.append(x)
	fnms.append("phi_gamma_aligned")

	# fnms = [x + "_ckpt_%i" % checkpoint_number for x in fnms]

	# TODO: plotting multiple slices at a time
	# fldr_path = os.path.join("./log", exp_name)
	fnm = "slices_ckpt_%i" % checkpoint_number
	plot_invariant_set_slices(phi_fn, param_dict, fldr_path=save_fldrpth, which_params=params_to_viz_list,
	                          constants_for_other_params=constants_for_other_params_list, fnm=fnm,
	                          checkpoint=checkpoint_number)
	plt.clf()
	plt.close()

if __name__ == "__main__":
	########################################################
	#########     FILL OUT HERE !!!!   #####################
	### ****************************************************

	# spherical, server 4
	# base_exp_names = ["flying_inv_pend_sphere_seed_0", "flying_inv_pend_sphere_seed_1", "flying_inv_pend_sphere_seed_2", "flying_inv_pend_sphere_softplus_seed_0", "flying_inv_pend_sphere_softplus_seed_1", "flying_inv_pend_sphere_softplus_seed_2"]

	# euclidean, server 5
	# base_exp_names = ["flying_inv_pend_euc_seed_0", "flying_inv_pend_euc_seed_1", "flying_inv_pend_euc_seed_2", "flying_inv_pend_euc_softplus_seed_0", "flying_inv_pend_euc_softplus_seed_1", "flying_inv_pend_euc_softplus_seed_2", "flying_inv_pend_euc_softplus_weighted_avg_seed_0", "flying_inv_pend_euc_softplus_weighted_avg_seed_1"]

	# base_exp_names = ["flying_inv_pend_euc_seed_0"]

	# base_exp_names = ["flying_inv_pend_euc_softplus_seed_1"]

	# server 4
	# base_exp_names = ["flying_inv_pend_ESG_reg_sigmoid_random_inside_sampler_weight_50", "flying_inv_pend_ESG_reg_sigmoid_random_inside_sampler_weight_150", "flying_inv_pend_ESG_reg_sigmoid_random_inside_sampler_weight_200", "flying_inv_pend_ESG_reg_softplus_random_sampler_weight_50", "flying_inv_pend_ESG_reg_softplus_random_sampler_weight_150", "flying_inv_pend_ESG_reg_softplus_random_sampler_weight_200", "flying_inv_pend_ESG_reg_softplus_random_inside_sampler_weight_50", "flying_inv_pend_ESG_reg_softplus_random_inside_sampler_weight_150"]

	# base_exp_names = ["flying_inv_pend_ESG_reg_sigmoid_random_inside_sampler_weight_50", "flying_inv_pend_ESG_reg_sigmoid_random_inside_sampler_weight_150", "flying_inv_pend_ESG_reg_sigmoid_random_inside_sampler_weight_200"]hio
	# find_attacker_n_reuse("flying_inv_pend_ESG_reg_sigmoid_random_inside_sampler_weight_50")

	# base_exp_names = ["flying_inv_pend_ESG_reg_softplus_random_sampler_weight_50", "flying_inv_pend_ESG_reg_softplus_random_sampler_weight_150", "flying_inv_pend_ESG_reg_softplus_random_sampler_weight_200"]

	# base_exp_names = ["flying_inv_pend_ESG_reg_softplus_random_inside_sampler_weight_50", "flying_inv_pend_ESG_reg_softplus_random_inside_sampler_weight_150"]

	# Server 4
	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_weight_150_seed_0", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_1", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_2", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_3", "flying_inv_pend_ESG_reg_speedup_weight_125_seed_0", "flying_inv_pend_ESG_reg_speedup_weight_125_seed_1", "flying_inv_pend_ESG_reg_speedup_weight_125_seed_2", "flying_inv_pend_ESG_reg_speedup_weight_125_seed_3"]

	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_weight_150_seed_1"]

	# Server 4
	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_weight_125_seed_0", "flying_inv_pend_ESG_reg_speedup_weight_125_seed_1", "flying_inv_pend_ESG_reg_speedup_weight_125_seed_2", "flying_inv_pend_ESG_reg_speedup_weight_125_seed_3"]

	# Server 5
	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_weight_175_seed_0", "flying_inv_pend_ESG_reg_speedup_weight_175_seed_1", "flying_inv_pend_ESG_reg_speedup_weight_175_seed_2", "flying_inv_pend_ESG_reg_speedup_weight_175_seed_3"]

	# server 5
	# base_exp_names = ["flying_inv_pend_ESG_reg_softplus_random_inside_sampler_weight_200"]

	##########################################################################################################
	# From Wednesday, April 13
	# Server 4
	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_weight_150_seed_0_again", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_1_again", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_2_again", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_3_again", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_4_again"]

	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_weight_150_seed_0_again", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_1_again"]
	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_weight_150_seed_2_again", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_3_again", "flying_inv_pend_ESG_reg_speedup_weight_150_seed_4_again"]

	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_weight_150_seed_0_again"]

	# Server 5
	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_1", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_2", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_3", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_4"]
	base_exp_names = ["flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0"]

	# base_exp_names = ["flying_inv_pend_ESG_reg_speedup_better_attacks_seed_3", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_4"]

	"""checkpoint_numbers = list(np.arange(0, 825, 50)) + [825]
	exp_names = ["flying_inv_pend_euc_softplus_seed_1"]*len(checkpoint_numbers)

	checkpoint_numbers = list(np.arange(0, 140, 25)) + [140]
	exp_names = ["flying_inv_pend_euc_softplus_weighted_avg_seed_1"]*len(checkpoint_numbers)"""


	# To visualize slices for a new experiment
	checkpoint_numbers = []
	exp_names = []
	for base_exp_name in base_exp_names:
		data = pickle.load(open("./log/%s/data.pkl" % base_exp_name, 'rb'))
		train_losses = data["train_losses"]
		n_it_so_far = len(train_losses)

		# TODO: hardcoded checkpoint save frequency
		it_per_slice = ((n_it_so_far/15.0)//5)*5
		print(base_exp_name)
		print("it_per_slice: %i" % it_per_slice)

		iterations = list(np.arange(0, n_it_so_far, it_per_slice).astype("int"))
		last_it = (n_it_so_far//5)*5
		if iterations[-1] < last_it:
			iterations.append(last_it)
		print(iterations)

		checkpoint_numbers.extend(iterations)
		exp_names.extend([base_exp_name]*len(iterations))
		#"""

	# To visualize slices for an ongoing experiment (already has slices visualized)
	"""checkpoint_numbers = []
	exp_names = []
	for base_exp_name in base_exp_names:
		# IPython.embed()
		search_dir = "./log/%s/" % base_exp_name
		files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
		files = [x for x in files if "slices" in x]
		files.sort(key=lambda x: os.path.getmtime(x))

		regex = re.compile('ckpt_([0-9]*)')
		last_ind = int(regex.findall(files[-1])[0])
		minus1_ind = int(regex.findall(files[-2])[0])
		minus2_ind = int(regex.findall(files[-3])[0])

		diff = minus1_ind - minus2_ind

		data = pickle.load(open("./log/%s/data.pkl" % base_exp_name, 'rb'))
		train_losses = data["train_losses"]
		n_it_so_far = len(train_losses)

		# TODO: hardcoded checkpoint save frequency
		it_per_slice = diff
		print(base_exp_name)
		print("it_per_slice: %i" % it_per_slice)

		iterations = list(np.arange(last_ind, n_it_so_far, it_per_slice).astype("int"))
		last_it = (n_it_so_far // 5) * 5
		if iterations[-1] < last_it:
			iterations.append(last_it)
		print(iterations)

		checkpoint_numbers.extend(iterations)
		exp_names.extend([base_exp_name] * len(iterations))
		# IPython.embed()
		#"""

	# IPython.embed()
	# IPython.embed()
	# exp_names = [exp_names[0]]

	"""
	base_exp_names = ["flying_inv_pend_phi_format_0_seed_0", "flying_inv_pend_phi_format_0_seed_1"]

	exp_names = []
	checkpoint_numbers = []
	for exp_name in base_exp_names:
		with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
			data = pickle.load(handle)
			train_attacks = data["train_attacks"]
			n_it = len(train_attacks)
			n_it_rounded = (n_it//10)*10
			# print(n_it, n_it_rounded)

		nums = list(np.arange(0, n_it_rounded, 50))
		checkpoint_numbers.extend(nums)

		exp_names.extend([exp_name]*len(nums))

		# IPython.embed()"""

	### ****************************************************
	########################################################
	# for exp_name in exp_names:
	# 	debug(exp_name)

	# TODO: check training progress
	# for exp_name in base_exp_names:
	# 	min_attack_loss_ind = graph_losses(exp_name)
	# 	# checkpoint_numbers.append(min_attack_loss_ind)

	# TODO: manually check attacks
	# with open("./log/%s/data.pkl" % "flying_inv_pend_phi_format_1_seed_0", 'rb') as handle:
	# 	data = pickle.load(handle)
	# 	train_attacks = data["train_attacks"]
	# 	# IPython.embed()
	# 	n_it = len(train_attacks)
	# 	for i in np.arange(0, n_it, 50):
	# 		print("It %i" % i)
	# 		attack = train_attacks[i]
	# 		# print(train_attacks[i])
	# 		print("gamma: %.2f, %.2f \t\t phi: %.2f, %.2f" % (attack[0], attack[3], attack[6], attack[8]))
	# 		print("beta: %.2f, %.2f \t\t theta: %.2f, %.2f" % (attack[1], attack[4], attack[7], attack[9]))
	# 		print("alpha: %.2f, %.2f" % (attack[2], attack[5]))
	# 		print("\n")
	#
	# 	IPython.embed()

	# TODO; plot ci values
	"""exp_name = "flying_inv_pend_phi_format_1_seed_1"
	k0_list = []
	k1_list = []
	n_it = 4000
	for checkpoint_number in np.arange(0, n_it, 10):
		phi_fn, param_dict = load_phi_and_params(exp_name, checkpoint_number)
		# IPython.embed()
		k0_list.append(phi_fn.k0.detach().cpu().numpy().item())
		k1_list.append(phi_fn.ci.detach().cpu().numpy().item())
	plt.plot(np.arange(0, n_it, 10), k0_list, label="k0")
	plt.plot(np.arange(0, n_it, 10), k1_list, label="k1")
	plt.title("k0, k1 over training iterations")
	plt.legend(loc="upper left")
	plt.savefig("./log/%s/k0_k1_plot" % exp_name)"""

	# TODO: plot pages of slices over many iterations

	for exp_name, checkpoint_number in zip(exp_names, checkpoint_numbers):

			phi_fn, param_dict = load_phi_and_params(exp_name, checkpoint_number)

			##################################################################
			# Slice visualization, with multiple subplots per plot
			# fldr_path = os.path.join("./log", exp_name)
			# plot_invariant_set_slices(phi_fn, param_dict, fldr_path=fldr_path, fnm="viz_invar_set_ckpt_%i" % checkpoint_number)
			# plt.clf()
			# plt.close()
			##################################################################
			# Slice visualization, one plot at a time

			# TODO
			params_to_viz_list = []
			constants_for_other_params_list = []
			fnms = []

			params_to_viz_list.extend([["phi", "dphi"], ["theta", "dtheta"], ["gamma", "dgamma"], ["beta", "dbeta"]])
			constants_for_other_params_list.extend([np.zeros(10)]*4)
			fnms.extend(["phi_dphi", "theta_dtheta", "gamma_dgamma", "beta_dbeta"])

			# TODO: more slices
			params_to_viz_list.extend([["dtheta", "dbeta"], ["dphi", "dgamma"]])
			constants_for_other_params_list.extend([np.zeros(10)]*2)
			fnms.extend(["dtheta_dbeta", "dphi_dgamma"])

			state_index_dict = param_dict["state_index_dict"]

			angle = math.pi/6 # [pi/5-pi/7] is a popular range of choices

			params_to_viz_list.append(["theta", "beta"]) # aligned, misaligned
			x = np.zeros(10)
			x[state_index_dict["phi"]] = -angle
			x[state_index_dict["gamma"]] = angle
			constants_for_other_params_list.append(x)
			fnms.append("theta_beta_misaligned")

			params_to_viz_list.append(["theta", "beta"]) # aligned, misaligned
			x = np.zeros(10)
			x[state_index_dict["phi"]] = angle
			x[state_index_dict["gamma"]] = angle
			constants_for_other_params_list.append(x)
			fnms.append("theta_beta_aligned")


			params_to_viz_list.append(["phi", "gamma"]) # aligned, misaligned
			x = np.zeros(10)
			x[state_index_dict["theta"]] = -angle
			x[state_index_dict["beta"]] = angle
			constants_for_other_params_list.append(x)
			fnms.append("phi_gamma_misaligned")

			params_to_viz_list.append(["phi", "gamma"]) # aligned, misaligned
			x = np.zeros(10)
			x[state_index_dict["theta"]] = angle
			x[state_index_dict["beta"]] = angle
			constants_for_other_params_list.append(x)
			fnms.append("phi_gamma_aligned")

			fnms = [x + "_ckpt_%i" % checkpoint_number for x in fnms]

			# params_to_viz_list = []
			# constants_for_other_params_list = []
			# fnms = []


			# IPython.embed()

			# TODO
			# for params_to_viz, constants_for_other_params, fnm in zip(params_to_viz_list, constants_for_other_params_list, fnms):
			# 	fldr_path = os.path.join("./log", exp_name)
			# 	plot_invariant_set_slices(phi_fn, param_dict, fldr_path=fldr_path, which_params=[params_to_viz], constants_for_other_params=[constants_for_other_params], fnm=fnm)
			#
			# 	plt.clf()
			# 	plt.close()

			# TODO: plotting multiple slices at a time
			fldr_path = os.path.join("./log", exp_name)
			fnm = "slices_ckpt_%i" % checkpoint_number
			plot_invariant_set_slices(phi_fn, param_dict, fldr_path=fldr_path, which_params=params_to_viz_list, constants_for_other_params=constants_for_other_params_list, fnm=fnm, checkpoint=checkpoint_number)
			plt.clf()
			plt.close()

			# Other plotting: samples on 2D slices, 3D slices, etc.
			# samples = load_attacks(exp_name, checkpoint_number)
			#
			# plot_invariant_set_slices(phi_fn, param_dict, samples=samples, fldr_path=fldr_path, fnm="viz_attacks_ckpt_%i" % checkpoint_number)
			#
			# # plot_cbf_3d_slices(phi_fn, param_dict, which_params = [["phi", "theta"]], fnm = "3d_viz_ckpt_%i" % checkpoint_number, fpth = exp_name)
			#
			# plt.clf()
			# plt.close()"""
