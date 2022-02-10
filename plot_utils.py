"""
Scratch for one-off code
"""
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

from torch.autograd import grad
from torch import nn

def graph_log_file_2(exp_name, mode='train'):
	"""
	Corresponds to above
	"""
	# exp_name = "cartpole_reduced_exp1a"
	# fig, axes = plt.subplots(2, 1)
	# print(axes.shape)
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)
		# IPython.embed()
		# timings = data["timings"]
		# test_losses = data["test_losses"]
		if mode == 'test':
			test_attack_losses = data["test_attack_losses"]
			test_reg_losses = data["test_reg_losses"]
			plt.plot(test_attack_losses, linewidth=0.5, label="Test attack loss")
			plt.plot(test_reg_losses, linewidth=0.5, label="Test reg loss")
			plt.title("Test loss for %s" % exp_name)
		elif mode == 'train':
			train_attack_losses = data["train_attack_losses"]
			train_reg_losses = data["train_reg_losses"]
			train_losses = data["train_losses"]

			plt.plot(train_attack_losses[:1000], linewidth=0.5, label="train attack loss")
			plt.plot(train_reg_losses[:1000], linewidth=0.5, label="train reg loss")
			plt.plot(train_losses[:1000], linewidth=0.5, label="train total loss")
			plt.title("Losses for %s" % exp_name)
	# plt.plot(timings, color='red', label="Runtime (hours)")
	# IPython.embed()
	plt.xlabel("Optimization steps")
	plt.legend(loc="upper right")
	# plt.title("Statistics throughout training")

	# Fix the tick interval to make comparison easy
	# plt.xticks(np.arange(40, -20, 5))

	plt.savefig("./log/%s/%s_%s_loss.png" % (exp_name, exp_name, mode))
	plt.clf()
	plt.cla()

	# print(train_attack_losses[-20:])
	# print(timings[-1])
	# totalsec = timings[-1]
	# print("Total training time: %i h, %i m" % (totalsec/3600, totalsec % 60))
	# print("%i h, %i m" % (totalsec/3600, totalsec % 60))
	print("%.5f" % np.min(train_attack_losses))
	# IPython.embed()

def create_phi_struct_load_xlim(exp_name, checkpoint_number):
	args = load_args("./log/%s/args.txt" % exp_name)
	dev = "cpu"
	device = torch.device(dev)
	param_dict = pickle.load(open("./log/%s/param_dict.pkl" % exp_name, "rb"))

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-args.max_angular_velocity, args.max_angular_velocity]], dtype=np.float32)

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	# if args.physical_difficulty == 'easy':  # medium length pole
	# 	param_dict = {
	# 		"I": 1.2E-3,
	# 		"m": 0.127,
	# 		"M": 1.0731,
	# 		"l": 0.3365
	# 	}
	# elif args.physical_difficulty == 'hard':  # long pole
	# 	param_dict = {
	# 		"I": 7.88E-3,
	# 		"m": 0.230,
	# 		"M": 1.0731,
	# 		"l": 0.6413
	# 	}
	#
	# param_dict["theta_safe_lim"] = args.theta_safe_lim
	# param_dict["max_force"] = args.max_force

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)

	if args.phi_include_xe:
		x_e = torch.zeros(1, x_dim)
	else:
		x_e = None

	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	return phi_fn, x_lim

def plot_3d(checkpoint_number, exp_name, fname=None):
	phi_fn, x_lim = create_phi_struct_load_xlim(exp_name, checkpoint_number)
	###################################
	# IPython.embed()
	delta = 0.1
	x = np.arange(x_lim[0, 0], x_lim[0, 1], delta)
	y = np.arange(x_lim[1, 0], x_lim[1, 1], delta)[::-1] # need to reverse it
	X, Y = np.meshgrid(x, y)

	# phi_load_fpth = "./checkpoint/cartpole_reduced_exp1a/checkpoint_69.pth"
	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)

	# print(phi_fn.a) # TODO
	# print(phi_fn.ci)
	##### Testing ######
	# state_dict = phi_fn.beta_net.state_dict()  # 0.weight/bias and 2.weight/bias
	# # print(torch.mean(state_dict["4.weight"]))
	# print(state_dict["2.weight"])
	# # print(state_dict)
	######

	input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
	input = torch.from_numpy(input)
	phi_i_vals = phi_fn(input)
	phi_vals = phi_i_vals[:, -1].detach().cpu().numpy()
	Z = np.reshape(phi_vals, X.shape)

	# IPython.embed()
	# S_vals = torch.max(phi_vals, dim=1)[0] # S = all phi_i <= 0
	# phi_signs = torch.sign(S_vals).detach().cpu().numpy()
	# phi_signs = np.reshape(phi_signs, X.shape)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	# fig, ax = plt.subplots(projection="3d")
	# ax.imshow(phi_signs, extent=[-math.pi, math.pi, -5.0, 5.0])
	# IPython.embed()
	# ax.set_ylim(-15.0, 15.0)
	ax.set_xlim(x_lim[0, 0], x_lim[0, 1])
	ax.set_ylim(x_lim[1, 0], x_lim[1, 1])

	# ax.set_xlim(-math.pi, math.pi)
	ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.75) #, bbox_inches='tight')
	# ax.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
	ax.contour(X, Y, Z, 10, lw=3, colors="k", linestyles="solid")
	# ax.set_aspect("equal")

	# ax.set_aspect("equal")

	# phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
	# ax.contour(X, Y, np.reshape(phi_vals_numpy, X.shape), levels=[0.0],
	#                  colors=('k',), linewidths=(2,))
	if fname is None:
		fname = "3d_checkpoint_%i" % checkpoint_number
	plt.title("Ckpt %i" % checkpoint_number)
	plt.savefig("./log/%s/%s" % (exp_name, fname))
	plt.clf()
	# plt.show()

def plot_2d_attacks_from_loaded(checkpoint_number, exp_name, fname=None):
	"""
	Plots binary +/- of CBF value
	Also plots training attacks (all of the candidate attacks, not just the maximizer)
	"""
	phi_fn, x_lim = create_phi_struct_load_xlim(exp_name, checkpoint_number)
	# print(out)
	# IPython.embed()
	###################################
	# IPython.embed()
	delta = 0.01
	# delta = 0.005
	x = np.arange(x_lim[0, 0], x_lim[0, 1], delta)
	y = np.arange(x_lim[1, 0], x_lim[1, 1], delta)[::-1] # need to reverse it # TODO
	X, Y = np.meshgrid(x, y)

	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)


	##### Plotting ######
	input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
	input = torch.from_numpy(input)
	phi_vals = phi_fn(input)
	# print(torch.min(phi_vals))
	# IPython.embed()
	S_vals = torch.max(phi_vals, dim=1)[0] # S = all phi_i <= 0
	phi_signs = torch.sign(S_vals).detach().cpu().numpy()
	phi_signs = np.reshape(phi_signs, X.shape)

	# fig, axes = plt.subplots(1, 2)
	#
	# for ax in axes:
	# 	ax.imshow(phi_signs, extent=[-math.pi, math.pi, -5.0, 5.0]) # TODO
	# 	ax.set_aspect("equal")
	#
	# 	phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
	# 	ax.contour(X, Y, np.reshape(phi_vals_numpy, X.shape), levels=[0.0],
	# 	                 colors=('k',), linewidths=(2,))
	#
	# # Plot attacks
	# with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
	# 	data = pickle.load(handle)
	#
	# 	attacks_init = data["train_attack_X_init"][checkpoint_number+1]
	# 	attacks = data["train_attack_X_final"][checkpoint_number+1]
	# 	best_attack = data["train_attacks"][checkpoint_number+1]
	#
	# axes[0].scatter(attacks_init[:, 0], attacks_init[:, 1], c="tab:orange", marker="x")
	# axes[1].scatter(attacks[:, 0], attacks[:, 1], c="tab:orange", marker="x")
	# axes[1].scatter(best_attack[0], best_attack[1], marker="D", c="c")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(phi_signs, extent=x_lim.flatten())
	ax.set_aspect("equal")
	phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
	plt.contour(X, Y, np.reshape(phi_vals_numpy, X.shape), levels=[0.0],
	                 colors=('k',), linewidths=(2,))

	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)

		attacks_init = data["train_attack_X_init"][checkpoint_number+1]
		attacks = data["train_attack_X_final"][checkpoint_number+1]
		best_attack = data["train_attacks"][checkpoint_number+1]

	ax.scatter(attacks[:, 0], attacks[:, 1], c="tab:orange", marker="x")
	ax.scatter(best_attack[0], best_attack[1], marker="D", c="c")

	### todo: delete
	# plt.scatter(0.68213777, 2.92963457, c="red")
	# plt.scatter(0.69678594, 3.01757929, c="blue")
	# input = torch.from_numpy(np.array([[0.68213777, 2.92963457], [0.69678594, 3.01757929]]).astype(np.float32))
	# phi_vals = phi_fn(input)
	# IPython.embed()

	# title = "Ckpt %i, k0 = %.4f, k1 = %.4f" % (checkpoint_number, phi_fn.k0[0, 0].item(), phi_fn.ci[0, 0].item())
	# plt.title(title)
	if fname is None:
		fname = "2d_attacks_from_loaded_checkpoint_%i.png" % checkpoint_number
	plt.savefig("./log/%s/%s" % (exp_name, fname), bbox_inches='tight')
	plt.clf()
	plt.close()

def plot_a_k(exp_name, n_it):
	a_list = []
	k_list = []
	for checkpoint_number in np.arange(0, n_it, 50):
		phi_fn, x_lim = create_phi_struct_load_xlim(exp_name, checkpoint_number)

		a = phi_fn.a[0, 0].item()
		k = phi_fn.ci[0, 0].item()

		a_list.append(a)
		k_list.append(k)

	plt.plot(a_list, label="A list")
	plt.plot(k_list, label="k list")
	plt.legend(loc="upper right")
	plt.savefig("./log/%s/a_k_plot.png" % exp_name)


def plot_trajectories(x_lim, N_rollout, x0s, phi_vals_on_grid, X, Y, phi_signs, info_dicts, save_prefix):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(phi_signs, extent=x_lim.flatten())
	ax.set_aspect("equal")
	ax.scatter(x0s[:, 1], x0s[:, 3])

	phi_star_on_grid = phi_vals_on_grid[:, -1]
	plt.contour(X, Y, np.reshape(phi_star_on_grid, X.shape), levels=[0.0],
					colors=('k',), linewidths=(2,))

	x = info_dicts["x"]
	for i in range(N_rollout):
		x_rl = x[i]
		plt.plot(x_rl[:, 1], x_rl[:, 3])
	plt.savefig(save_prefix+"trajectories.png", bbox_inches='tight')
	plt.clf()
	plt.close()

def plot_exited_trajectories(x_lim, x0s, phi_vals_on_grid, X, Y, phi_signs, info_dicts, save_prefix):
	phi_vals = info_dicts["phi_vals"] # (N_rollout, T_max, r+1)
	phi_max = np.max(phi_vals, axis=2)
	rollouts_any_exits = np.any(phi_max > 0, axis=1)
	any_exits = np.any(rollouts_any_exits)

	if any_exits:
		exit_rollout_inds = np.argsort(np.max(phi_max, axis=1)).flatten()[::-1]
		exit_rollout_inds = exit_rollout_inds[:min(5, np.sum(rollouts_any_exits))]
		# exit_rollout_inds = np.argwhere(rollouts_any_exits).flatten()
		# exit_rollout_inds = np.random.choice(exit_rollout_inds, size=5) # Note: sampling 5 (with replacement, so it'll work if there are fewer than 5)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_aspect("equal")

		ax.imshow(phi_signs, extent=x_lim.flatten())
		phi_star_on_grid = phi_vals_on_grid[:, -1]
		plt.contour(X, Y, np.reshape(phi_star_on_grid, X.shape), levels=[0.0],
						colors=('k',), linewidths=(2,))

		ax.scatter(x0s[exit_rollout_inds, 1], x0s[exit_rollout_inds, 3]) # x0s

		x = info_dicts["x"]
		for i in exit_rollout_inds:
			x_rl = x[i]
			plt.plot(x_rl[:, 1], x_rl[:, 3])

		for i in exit_rollout_inds: # to scatter on top of the trajectories
			x_rl = x[i]
			xi_exit_ind = np.argwhere(phi_max[i] > 0).flatten()
			plt.scatter(x_rl[xi_exit_ind, 1], x_rl[xi_exit_ind, 3], c="red", s=0.2)
		
		plt.savefig(save_prefix+"exiting_trajectories.png", bbox_inches='tight')
		plt.clf()
		plt.close()

def plot_samples_invariant_set(x_lim, x0s, phi_vals_on_grid, X, save_prefix):
	# print("Check x0 plotting")
	# IPython.embed()
	plt.clf()
	plt.cla()
	mpl.rcParams.update(mpl.rcParamsDefault)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	max_phi_vals_on_grid = phi_vals_on_grid.max(axis=1)
	phi_signs = np.reshape(np.sign(max_phi_vals_on_grid), X.shape)
	ax.imshow(phi_signs, extent=x_lim.flatten())
	ax.set_aspect("equal")
	plt.savefig(save_prefix+"invariant_set.png", bbox_inches='tight')

	# IPython.embed()
	inds = np.argwhere(max_phi_vals_on_grid <= 0)
	print(inds.size/max_phi_vals_on_grid.size)
	# sys.exit(0)

	ax.scatter(x0s[:, 1], x0s[:, 3])
	plt.savefig(save_prefix+"x0s.png", bbox_inches='tight')
	plt.clf()
	plt.close()
	return phi_signs

if __name__=="__main__":
	# December 16
	# exp_names = ["cartpole_reduced_64_64_64_40-1", "cartpole_reduced_64_64_64_40-2", "cartpole_reduced_64_64_64_45-1"]
	# n_it = [950, 950, 950]
	#
	# # for exp_name in exp_names:
	# # 	graph_log_file_2(exp_name)
	# #
	# # for i, exp_name in enumerate(exp_names):
	# # 	for checkpoint_number in np.arange(0, n_it[i], 50):
	# # 		print(checkpoint_number)
	# # 		plot_2d_attacks_from_loaded(checkpoint_number, exp_name)
	#
	# for i, exp_name in enumerate(exp_names):
	# 	plot_a_k(exp_name, n_it[i])
	# 	plt.clf()
	# 	plt.cla()

	####################################################################################
	# exp_names = ["cartpole_reduced_64_64_40-1", "cartpole_reduced_64_64_40-2", "cartpole_reduced_64_64_40-3", "cartpole_reduced_64_64_40-4", "cartpole_reduced_64_64_40-5"]
	# n_it = [1440, 1440, 1440, 1440, 1430]

	# exp_names = ["cartpole_reduced_64_64_60pts_gradient_avging_seed_1", "cartpole_reduced_64_64_60pts_gradient_avging_seed_2", "cartpole_reduced_64_64_60pts_gradient_avging_seed_3", "cartpole_reduced_64_64_60pts_gradient_avging_seed_4"]
	# n_it = [3000, 3000, 3000, 3000]

	# exp_names = ["cartpole_reduced_64_64_60pts_20weight_gdavg_newreg", "cartpole_reduced_64_64_60pts_40weight_gdavg_newreg", "cartpole_reduced_64_64_60pts_50weight_gdavg_newreg"]
	# n_it = [283, 425, 883]

	# exp_names = ["cartpole_reduced_reg_point3_sigmoid_regweight_10", "cartpole_reduced_reg_point3_sigmoid_regweight_50", "cartpole_reduced_reg_point3_sigmoid_regweight_100", "cartpole_reduced_reg_point3_sigmoid_regweight_250"]
	# n_it = [470, 540, 480, 540]

	# exp_names = ["cartpole_reduced_reg_point3_sigmoid_regweight_250_init_small", "cartpole_reduced_reg_point3_sigmoid_regweight_500_init_small", "cartpole_reduced_reg_point3_sigmoid_regweight_750_init_small"]
	# n_it = [2000, 3000, 3000]

	# exp_names = ["cartpole_reduced_debugpinch1", "cartpole_reduced_debugpinch2a", "cartpole_reduced_debugpinch2b", "cartpole_reduced_debugpinch2c", "cartpole_reduced_debugpinch3"]
	# n_it = [850, 980, 950, 850, 780]
	#
	# exp_names = exp_names[2:]
	# print(exp_names)

	# exp_names = ["cartpole_reduced_debugpinch1_softplus_s1", "cartpole_reduced_debugpinch1_softplus_s2", "cartpole_reduced_debugpinch1_softplus_s3",  "cartpole_reduced_debugpinch3_softplus_s1", "cartpole_reduced_debugpinch3_softplus_s2", "cartpole_reduced_debugpinch3_softplus_s3"]
	# n_it = [1500]*5

	# exp_names = ["cartpole_reduced_debugpinch1_resoftplus_s1", "cartpole_reduced_debugpinch1_resoftplus_s2", "cartpole_reduced_debugpinch1_resoftplus_s3"]
	# n_it = [1500, 1500, 1340]

	# exp_names = ["cartpole_reduced_debugpinch3_softplus_s1"]
	# n_it = [1450]
	#
	# plot_2d_attacks_from_loaded(1450, "cartpole_reduced_debugpinch3_softplus_s1")
	####################################################################################

	# for i, exp_name in enumerate(exp_names):
	# 	plot_a_k(exp_name, n_it[i])
	# 	plt.clf()
	# 	plt.cla()

	# for exp_name in exp_names:
	# 	graph_log_file_2(exp_name)

	# for i, exp_name in enumerate(exp_names):
	# 	for checkpoint_number in np.arange(0, n_it[i], 50):
	# 	# for checkpoint_number in np.arange(100, 250, 10):
	# 		print(checkpoint_number)
	# 		plot_2d_attacks_from_loaded(checkpoint_number, exp_name)

	# for i, exp_name in enumerate(exp_names):
	# 	for checkpoint_number in np.arange(0, n_it[i], 50):
	# 		print(checkpoint_number)
	# 		plot_3d(checkpoint_number, exp_name)

	# exp_names = ["cartpole_reduced_64_64_40_no_adam", "cartpole_reduced_64_64_40_no_adam_fast"]
	# n_it = [1800, 1600]
	#
	# for i, exp_name in enumerate(exp_names):
	# 	for checkpoint_number in np.arange(0, n_it[i], 100):
	# 		print(checkpoint_number)
	# 		plot_2d_attacks_from_loaded(checkpoint_number, exp_name)

	# plotting the best iterations (lowest total loss)

	# train_attack_losses = data["train_attack_losses"]
	# train_reg_losses = data["train_reg_losses"]
	# train_losses = data["train_losses"]

	# Among the iterations with subzero attack loss, find the one with lowest total loss
	# for exp_name in exp_names:
	# 	# IPython.embed()
	# 	args = load_args("./log/%s/args.txt" % exp_name)
	# 	n_checkpoint_step = args.n_checkpoint_step
	# 	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
	# 		data = pickle.load(handle)
	# 		train_losses = np.array(data["train_losses"])
	# 		train_attack_losses = np.array(data["train_attack_losses"])
	# 		train_losses_w_saved_checkpoint = np.array(train_losses)[::n_checkpoint_step]
	# 		train_attack_losses_w_saved_checkpoint = np.array(data["train_attack_losses"])[::n_checkpoint_step]
	#
	# 		ind_attack_subzero = np.argwhere(train_attack_losses_w_saved_checkpoint < 0)
	#
	# 		if len(ind_attack_subzero) > 0:
	# 			total_loss_subzero = train_losses[ind_attack_subzero*n_checkpoint_step] # total losses for checkpoints where attack loss < 0
	# 			ind_best = ind_attack_subzero[np.argmin(total_loss_subzero)]*n_checkpoint_step
	# 		else:
	# 			print("No subzero saved checkpoint")
	# 			ind_best = np.argmin(train_losses_w_saved_checkpoint)*n_checkpoint_step
	#
	# 		ind_best = ind_best.item() # TODO
	# 		checkpoint_number = ind_best
	# 		print("Best checkpoint is: %i" % checkpoint_number)
	# 		# IPython.embed()
	# 		# print("Total, attack, reg losses: %.4f, %.4f, %.4f" % (train_losses[ind_best].item(), data["train_attack_losses"][ind_best].item(), data["train_reg_losses"][ind_best].item()))
	# 		print("Total, attack, reg losses: %.4f, %.4f, %.4f" % (train_losses[ind_best], data["train_attack_losses"][ind_best], data["train_reg_losses"][ind_best]))
	# 		# plot_2d_attacks_from_loaded(checkpoint_number, exp_name, fname=("best_2d_attacks_from_loaded_checkpoint_%i" % checkpoint_number))
	# 		# plot_3d(checkpoint_number, exp_name, fname=("best_3d_checkpoint_%i" % checkpoint_number))

	exp_name = "flying_inv_pend_first_run"
	graph_log_file_2(exp_name)


	