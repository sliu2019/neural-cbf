"""
Scratch for one-off code
"""
import numpy as np
import IPython
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math
from main import Phi, Objective, Regularizer
from src.argument import parser, print_args
from src.utils import *
import torch
import pickle
from deprecated.phi_baseline import PhiBaseline
from src.attacks.gradient_batch_attacker import GradientBatchAttacker
from torch.autograd import grad
from torch import nn
def parse_log_file():
	"""
	Neglected to write test loss out to .npy file
	"""
	exp_name = "cartpole_reduced_exp1a"
	fpth = "./log/%s/train_log.txt" % exp_name
	with open(fpth) as file:
		lines = file.readlines()

		test_losses = []
		times = []
		for line in lines:
			if "test loss" in line:
				# IPython.embed()
				loss_str = re.search(r'loss: ([-+]?\d*\.?\d+)%', line).group(1)
				loss = float(loss_str)
				print(loss)
				test_losses.append(loss)
			elif "time spent" in line:
				# IPython.embed()
				time_str = re.search(r'far: ([-+]?\d*\:?\d+\:?\d+)', line).group(1)
				# print(time_str)
				hms = time_str.split(":")
				h_frac = 0
				h_frac += float(hms[0])
				h_frac += float(hms[0])/60.0
				h_frac += float(hms[0])/360.0
				# print(time_str, h_frac)
				times.append(h_frac)

		IPython.embed()
		log_array = np.concatenate((np.array(test_losses)[None], np.array(times)[None]), axis=0)
		np.save("./log/%s/train_log.npy" % exp_name, log_array)

def graph_log_file():
	"""
	Corresponds to above
	"""
	exp_name = "cartpole_reduced_exp1a"
	log_array = np.load("./log/%s/train_log.npy" % exp_name)
	plt.plot(log_array[0], linewidth=0.5, label="Test loss")
	plt.plot(log_array[1], color='red', label="Runtime (hours)")
	plt.xlabel("Optimization steps")
	plt.legend(loc="upper right")
	plt.title("Statistics throughout training")
	plt.savefig("./log/%s/test_loss.png" % exp_name)

	# plt.ylabel("Test loss")

def graph_log_file_2(exp_name, mode='train'):
	"""
	Corresponds to above
	"""
	# exp_name = "cartpole_reduced_exp1a"
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)
		# IPython.embed()
		timings = data["timings"]
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

			plt.plot(train_attack_losses, linewidth=0.5, label="train attack loss")
			plt.plot(train_reg_losses, linewidth=0.5, label="train reg loss")
			plt.plot(train_losses, linewidth=0.5, label="train total loss")
			plt.title("Train loss for %s" % exp_name)
	# plt.plot(timings, color='red', label="Runtime (hours)")
	# IPython.embed()
	plt.xlabel("Optimization steps")
	plt.legend(loc="upper right")
	# plt.title("Statistics throughout training")

	plt.savefig("./log/%s/%s_%s_loss.png" % (exp_name, exp_name, mode))
	plt.clf()
	plt.cla()

	# IPython.embed()

def plot_phi_2d_level_curve(phi_fn, phi_load_fpth, checkpoint_number, x_lim, which_2_state_vars):
	"""
	Plots phi's
	which_2_state_vars: 2 entry list of indices

	Hardcoded for cartpole
	"""
	# IPython.embed()
	# Load
	load_model(phi_fn, phi_load_fpth) # "./checkpoint/cartpole_default/checkpoint_1490.pth"

	# Make meshgrid
	ind1, ind2 = which_2_state_vars
	delta = 0.025
	x = np.arange(x_lim[ind1, 0], x_lim[ind1, 1], delta)
	y = np.arange(x_lim[ind2, 0], x_lim[ind2, 1], delta)
	X, Y = np.meshgrid(x, y)

	input_batch = torch.zeros((X.size, 4)) # Set the other two variables at 0
	input_batch[:, ind1] = torch.from_numpy(X.flatten())
	input_batch[:, ind2] = torch.from_numpy(Y.flatten())

	Z = phi_fn(input_batch)
	Z = Z.view(X.shape)
	Z = Z.cpu().detach().numpy()

	# Plot
	CS = plt.contour(X, Y, Z)
	plt.contour(X, Y, Z, levels=[0.0])
	# CS = plt.contour(X, Y, Z)
	plt.clabel(CS, inline=1, fontsize=10)
	# plt.savefig(phi_load_fpth[:-4] + "_2d_level_curve.png")
	plt.savefig("./log/cartpole_default/2d_level_curves/checkpoint_%i_vars_%i_%i.png" % (checkpoint_number, ind1, ind2))


def plot_phi_2d_level_curve_over_training():
	"""
	Hard-coded for visualizing cartpole-default training
	"""
	args = parser()

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
		"theta_safe_lim": math.pi / 10.0,
		"max_force": 1.0
	}

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	# uvertices_fn = ULimitSetVertices(param_dict)

	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, args)

	which_checkpoints = [1490] # TODO
	which_2_state_vars = [0, 1] # TODO

	for checkpoint_number in which_checkpoints:
		# call above function
		phi_load_fpth = "./checkpoint/cartpole_default/checkpoint_%i.pth" % checkpoint_number
		# print(phi_load_fpth)
		plot_phi_2d_level_curve(phi_fn, phi_load_fpth, checkpoint_number, x_lim, which_2_state_vars)


def plot_2d_binary(checkpoint_number, save_fnm, exp_name):
	"""
	Plots binary +/- of CBF value
	Also plots training attacks
	"""
	args = load_args("./log/%s/args.txt" % exp_name)
	# IPython.embed()
	# args = parser()
	dev = "cpu"
	device = torch.device(dev)

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	param_dict = {
		"I": 0.021,
		"m": 0.25,
		"M": 1.00,
		"l": 0.5,
		"theta_safe_lim": math.pi / 2.0,
		"max_force": 15.0
	}

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	# n_samples = 50
	# rnge = torch.tensor([param_dict["theta_safe_lim"], x_lim[1:x_dim, 1]])
	# A_samples = torch.rand(n_samples, x_dim) * (2 * rnge) - rnge  # (n_samples, x_dim)
	x_e = torch.zeros(1, x_dim)

	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	# plot_range_x = [-0.5, 0.5] # TODO [-math.pi, math.pi]
	# plot_range_y = [-5, -1] # TODO: [-5, 5]
	plot_range_x = [-math.pi, math.pi]
	plot_range_y = [-15, 15]
	###################################
	# IPython.embed()
	delta = 0.01
	x = np.arange(plot_range_x[0], plot_range_x[1], delta)
	y = np.arange(plot_range_y[0], plot_range_y[1], delta)[::-1] # need to reverse it
	X, Y = np.meshgrid(x, y)

	# phi_load_fpth = "./checkpoint/cartpole_reduced_exp1a/checkpoint_69.pth"
	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)

	# print(phi_fn.ci)
	phi_fn.ci = nn.Parameter(torch.tensor([[0.05]])) # TODO
	# IPython.embed()

	##### Testing ######
	# state_dict = phi_fn.beta_net.state_dict()  # 0.weight/bias and 2.weight/bias
	# # print(torch.mean(state_dict["4.weight"]))
	# print(state_dict["2.weight"])
	# # print(state_dict)

	# zero_output = phi_fn(torch.zeros(1, 2))
	# print(zero_output)
	# print(phi_fn.ci, phi_fn.sigma)
	######

	input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
	input = torch.from_numpy(input)
	phi_vals = phi_fn(input)
	# IPython.embed()
	S_vals = torch.max(phi_vals, dim=1)[0] # S = all phi_i <= 0
	phi_signs = torch.sign(S_vals).detach().cpu().numpy()
	phi_signs = np.reshape(phi_signs, X.shape)


	fig, ax = plt.subplots()
	ax.imshow(phi_signs, extent=[plot_range_x[0], plot_range_x[1], plot_range_y[0], plot_range_y[1]])
	ax.set_aspect("equal")

	phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
	# print(np.min(phi_vals_numpy))
	# print(np.sort(phi_vals_numpy)[:500])
	# ax.contour(X, Y, np.reshape(phi_vals_numpy, X.shape), levels=[0.0],
	#                  colors=('k',), linewidths=(2,))
	ax.contour(X, Y, np.reshape(phi_vals_numpy, X.shape))

	# Sanity check projection: plot vector field of gradients
	# IPython.embed()
	delta = 0.1
	x = np.arange(plot_range_x[0], plot_range_x[1], delta)
	y = np.arange(plot_range_y[0], plot_range_y[1], delta)[::-1] # need to reverse it
	X, Y = np.meshgrid(x, y)
	input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
	input = torch.from_numpy(input)
	input.requires_grad = True
	loss = torch.abs(phi_fn(input)[:, -1])
	grad_to_zero_level = grad([torch.sum(loss)], input)[0]
	input.requires_grad = False
	grad_to_zero_level = grad_to_zero_level.detach().cpu().numpy()

	plt.quiver(X.flatten(), Y.flatten(), grad_to_zero_level[:, 0], grad_to_zero_level[:, 1], color='w')

	# Get attacks
	# TODO: remove, plot recorded attack points
	# with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
	# 	data = pickle.load(handle)
	# 	train_attacks = data["train_attacks"]
	# 	test_losses = data["test_losses"]
	#
	# 	# IPython.embed()
	# 	print("Test loss is %f" % (test_losses[checkpoint_number].item()))
	# 	plt.scatter(train_attacks[checkpoint_number+1][0], train_attacks[checkpoint_number+1][1], c="tab:orange", marker="x") # +1 because of implementation quirk

	# TODO: remove, test dS/dG options on batch attacker
	# logger = create_logger("log/discard", 'train', 'info')
	# x_lim_pole = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)
	# x_lim_pole = torch.tensor(x_lim_pole).to(device)
	# from src.attacks.gradient_batch_attacker import GradientBatchAttacker
	# attacker = GradientBatchAttacker(x_lim_pole, device, logger, n_samples=2)
	#
	# dG_dS = attacker.sample_points_on_boundary(phi_fn, mode="dG+dS").detach().cpu().numpy()
	# plt.scatter(dG_dS[:, 0], dG_dS[:, 1], c="tab:orange", marker="x")
	# print("done")
	# dG_not_dS = attacker.sample_points_on_boundary(phi_fn, mode="dG/dS").detach().cpu().numpy()
	# plt.scatter(dG_not_dS[:, 0], dG_not_dS[:, 1], c="red", marker="x")

	# print(attacks)
	plt.savefig("./log/%s/%s" % (exp_name, save_fnm), dpi=600)

def plot_3d(checkpoint_number, exp_name):
	args = load_args("./log/%s/args.txt" % exp_name)
	dev = "cpu"
	device = torch.device(dev)

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	if args.physical_difficulty == 'easy':  # medium length pole
		param_dict = {
			"I": 1.2E-3,
			"m": 0.127,
			"M": 1.0731,
			"l": 0.3365
		}
	elif args.physical_difficulty == 'hard':  # long pole
		param_dict = {
			"I": 7.88E-3,
			"m": 0.230,
			"M": 1.0731,
			"l": 0.6413
		}
	param_dict["theta_safe_lim"] = args.theta_safe_lim
	param_dict["max_force"] = args.max_force

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	x_e = torch.zeros(1, x_dim)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)
	# IPython.embed()
	###################################
	# IPython.embed()
	delta = 0.1
	x = np.arange(-math.pi, math.pi, delta)
	y = np.arange(-15, 15, delta)[::-1] # need to reverse it
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
	ax.set_ylim(-15.0, 15.0)
	ax.set_xlim(-math.pi, math.pi)
	ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.75)
	# ax.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
	ax.contour(X, Y, Z, 10, lw=3, colors="k", linestyles="solid")
	# ax.set_aspect("equal")

	# ax.set_aspect("equal")

	# phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
	# ax.contour(X, Y, np.reshape(phi_vals_numpy, X.shape), levels=[0.0],
	#                  colors=('k',), linewidths=(2,))
	save_fnm = "3d_checkpoint_%i" % checkpoint_number
	plt.savefig("./log/%s/%s" % (exp_name, save_fnm))
	plt.clf()
	# plt.show()

def plot_2d_attacks(checkpoint_number, exp_name):
	"""
	Plots binary +/- of CBF value
	Also plots training attacks (all of the candidate attacks, not just the maximizer)
	"""
	args = load_args("./log/%s/args.txt" % exp_name)
	dev = "cpu"
	device = torch.device(dev)

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	param_dict = {
		"I": 0.021,
		"m": 0.25,
		"M": 1.00,
		"l": 0.5,
		"theta_safe_lim": math.pi / 2.0,
		"max_force": 15.0
	}

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	x_e = torch.zeros(1, x_dim)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	log_folder = "./log/discard"
	logger = create_logger(log_folder, 'train', 'info') # doesn't matter, isn't used
	x_lim_torch = torch.tensor(x_lim).to(device)
	# test_attacker = GradientBatchAttacker(x_lim_torch, device, logger,
	#                                       stopping_condition=args.test_attacker_stopping_condition,
	#                                       n_samples=n_attacks,
	#                                       projection_stop_threshold=args.test_attacker_projection_stop_threshold,
	#                                       projection_lr=args.test_attacker_projection_lr)
	# test_attacker = GradientBatchAttacker(x_lim_torch, device, logger,
	#                                       stopping_condition="n_steps", # TODO: replaceargs.test_attacker_stopping_condition,
	#                                       max_n_steps=100,
	#                                       n_samples=10, # TODO: replace w/ 50
	#                                       projection_stop_threshold=1e-1, #=args.test_attacker_projection_stop_threshold,
	#                                       projection_lr=1e-4, #args.test_attacker_projection_lr,
	#                                       early_stopping_min_delta=1e-5,
	#                                       early_stopping_patience=10,
	#                                       lr=1e-3
	#                                       )
	test_attacker = GradientBatchAttacker(x_lim_torch, device, logger,
	                                      stopping_condition="early_stopping", # TODO: replaceargs.test_attacker_stopping_condition,
	                                      n_samples=25, # TODO: replace w/ 50
	                                      projection_stop_threshold=1e-1, #=args.test_attacker_projection_stop_threshold,
	                                      projection_lr=1e-4, #args.test_attacker_projection_lr,
	                                      early_stopping_min_delta=1e-3,
	                                      early_stopping_patience=50,
	                                      lr=1e-3
	                                      )
	###################################
	# IPython.embed()
	delta = 0.01
	x = np.arange(-math.pi, math.pi, delta)
	y = np.arange(-5, 5, delta)[::-1] # need to reverse it
	X, Y = np.meshgrid(x, y)

	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)

	# TODO
	phi_fn.ci = nn.Parameter(torch.tensor([[0.1]]))

	##### Plotting ######
	input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
	input = torch.from_numpy(input)
	phi_vals = phi_fn(input)
	S_vals = torch.max(phi_vals, dim=1)[0] # S = all phi_i <= 0
	phi_signs = torch.sign(S_vals).detach().cpu().numpy()
	phi_signs = np.reshape(phi_signs, X.shape)


	fig, axes = plt.subplots(1, 2)

	for ax in axes:
		ax.imshow(phi_signs, extent=[-math.pi, math.pi, -5.0, 5.0])
		ax.set_aspect("equal")

		phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
		ax.contour(X, Y, np.reshape(phi_vals_numpy, X.shape), levels=[0.0],
		                 colors=('k',), linewidths=(2,))

	# Plot attacks
	objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger)
	attacks_init, attacks, best_attack, obj_vals = test_attacker.opt(objective_fn, phi_fn, debug=True, mode="dS") # TODO
	obj_vals = obj_vals.detach().cpu().numpy()

	inds = np.argsort(attacks[:, 0])

	# IPython.embed()
	obj_vals_init = objective_fn(attacks_init)
	obj_vals_init = obj_vals_init.detach().cpu().numpy()
	improvement = obj_vals - obj_vals_init
	if np.any(improvement < 0):
		print("Manifold optimization does not yield strict improvement")
		print(improvement)

		# neg_inds = np.where(improvement < 0)[0]
		# print("Negative improvement at these init attacks")
		# print(attacks_init[neg_inds])
		# print("Corr. ultimate attacks")
		# print(attacks[neg_inds])
		# IPython.embed()

	best_attack_improvement = np.max(obj_vals) - np.max(obj_vals_init)
	print("%f (+ %f)" % (np.max(obj_vals), best_attack_improvement))
	# print("Has objective value been maximized? final - init > 0")
	# print(obj_vals - obj_vals_init)

	# ids = np.nonzero(obj_vals)[0]
	# print(attacks[ids])
	# IPython.embed()
	attacks_init = attacks_init.detach().cpu().numpy()
	attacks = attacks.detach().cpu().numpy()
	best_attack = best_attack.detach().cpu().numpy()
	# print(best_attack)

	axes[0].scatter(attacks_init[:, 0], attacks_init[:, 1], c="tab:orange", marker="x")
	axes[1].scatter(attacks[:, 0], attacks[:, 1], c="tab:orange", marker="x")
	axes[1].scatter(best_attack[0], best_attack[1], marker="D", c="c")

	plt.savefig("./log/%s/%s" % (exp_name, "2d_attacks_checkpoint_%i.png" % checkpoint_number))

	# objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger)
	# test_attack = torch.tensor([-2.4602,  0.1066]).view(1, -1)
	# all_phi = objective_fn(test_attack)
	# print(all_phi)
	# IPython.embed()
	# print(attacks_init)

def plot_2d_attacks_from_loaded(checkpoint_number, exp_name):
	"""
	Plots binary +/- of CBF value
	Also plots training attacks (all of the candidate attacks, not just the maximizer)
	"""
	args = load_args("./log/%s/args.txt" % exp_name)
	dev = "cpu"
	device = torch.device(dev)

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-args.max_angular_velocity, args.max_angular_velocity]], dtype=np.float32) # TODO

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	if args.physical_difficulty == 'easy':  # medium length pole
		param_dict = {
			"I": 1.2E-3,
			"m": 0.127,
			"M": 1.0731,
			"l": 0.3365
		}
	elif args.physical_difficulty == 'hard':  # long pole
		param_dict = {
			"I": 7.88E-3,
			"m": 0.230,
			"M": 1.0731,
			"l": 0.6413
		}

	param_dict["theta_safe_lim"] = args.theta_safe_lim
	param_dict["max_force"] = args.max_force

	# param_dict = pickle.load(open("./log/%s/param_dict.pkl" % exp_name, "rb")) # TODO: replace with this

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	x_e = torch.zeros(1, x_dim)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	out = phi_fn(x_e)
	assert out[0, -1].item() <= 0
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

	# IPython.embed()
	title = "a = %.4f, k = %.4f" % (phi_fn.a[0, 0].item(), phi_fn.ci[0, 0].item())
	plt.title(title)
	plt.savefig("./log/%s/%s" % (exp_name, "2d_attacks_from_loaded_checkpoint_%i.png" % checkpoint_number))
	plt.clf()
	plt.close()

def debug_manifold_optimization(checkpoint_number, exp_name):
	args = load_args("./log/%s/args.txt" % exp_name)
	dev = "cpu"
	device = torch.device(dev)

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	param_dict = {
		"I": 0.021,
		"m": 0.25,
		"M": 1.00,
		"l": 0.5,
		"theta_safe_lim": math.pi / 2.0,
		"max_force": 15.0
	}

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)
	x_e = torch.zeros(1, x_dim)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)
	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)


	log_folder = "./log/discard"
	logger = create_logger(log_folder, 'train', 'info') # doesn't matter, isn't used
	x_lim_torch = torch.tensor(x_lim).to(device)
	# attacker = GradientBatchAttacker(x_lim_torch, device, logger,
	#                                       stopping_condition=args.test_attacker_stopping_condition,
	#                                       n_samples=10,
	#                                       projection_stop_threshold=args.test_attacker_projection_stop_threshold,
	#                                       projection_lr=args.test_attacker_projection_lr,
	#                                       early_stopping_min_delta=1e-7,
	#                                       early_stopping_patience=50,
	#                                       lr=2e-3, # TODO
	#                                       verbose=True,
	#                                       projection_time_limit=10000)
	attacker = GradientBatchAttacker(x_lim_torch, device, logger,
	                                      stopping_condition=args.test_attacker_stopping_condition,
	                                      n_samples=10,
	                                      projection_stop_threshold=1e-1, #=args.test_attacker_projection_stop_threshold,
	                                      projection_lr=1e-4, #args.test_attacker_projection_lr,
	                                      early_stopping_min_delta=1e-7,
	                                      early_stopping_patience=50,
	                                      lr=1e-4, # TODO: 2e-3
	                                      verbose=True)

	objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger)

	attack_init = torch.tensor([[-2.0407, 0.0992],
	        [-1.9117, 0.0940]])
	for i in range(200):
		attack_init = attacker.step(objective_fn, phi_fn, attack_init)
	IPython.embed()

def plot_ci_over_time(exp_name):
	args = load_args("./log/%s/args.txt" % exp_name)
	dev = "cpu"
	device = torch.device(dev)

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	param_dict = {
		"I": 0.021,
		"m": 0.25,
		"M": 1.00,
		"l": 0.5,
		"theta_safe_lim": math.pi / 2.0,
		"max_force": 15.0
	}

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	x_e = torch.zeros(1, x_dim)

	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	ci_list = []
	checkpoint_range = np.arange(0, 14000, 1000)
	for checkpoint_number in checkpoint_range:
		phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
		load_model(phi_fn, phi_load_fpth)

		# IPython.embed()
		# print(phi_fn.beta_net[0].bias)
		print(phi_fn.c)
		ci = phi_fn.ci
		ci = ci.detach().cpu().numpy()
		ci = ci[0, 0]
		ci_list.append(ci)

	# IPython.embed()
	print(ci_list)
	plt.plot(checkpoint_range, ci_list)
	plt.xlabel("Optimization steps")
	plt.title("Ci over training")

	plt.savefig("./log/%s/ci_throughout_training.png" % exp_name)
	plt.clf()
	plt.cla()

def test_reg_term(exp_name, checkpoint_number):
	"""
	Compute the reg term
	"""
	args = load_args("./log/%s/args.txt" % exp_name)
	dev = "cpu"
	device = torch.device(dev)

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-args.max_angular_velocity, args.max_angular_velocity]], dtype=np.float32) # TODO

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	if args.physical_difficulty == 'easy':  # medium length pole
		param_dict = {
			"I": 1.2E-3,
			"m": 0.127,
			"M": 1.0731,
			"l": 0.3365
		}
	elif args.physical_difficulty == 'hard':  # long pole
		param_dict = {
			"I": 7.88E-3,
			"m": 0.230,
			"M": 1.0731,
			"l": 0.6413
		}

	param_dict["theta_safe_lim"] = args.theta_safe_lim
	param_dict["max_force"] = args.max_force
	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	x_e = torch.zeros(1, x_dim)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	out = phi_fn(x_e)
	assert out[0, -1].item() <= 0
	###################################
	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)

	n_mesh_grain = 0.1
	XXX = np.meshgrid(*[np.arange(r[0], r[1], n_mesh_grain) for r in x_lim])
	A_samples = np.concatenate([x.flatten()[:, None] for x in XXX], axis=1)
	A_samples = torch.from_numpy(A_samples.astype(np.float32))
	A_samples = A_samples.to(device)

	reg_fn = Regularizer(phi_fn, device, reg_weight=1.0, A_samples=A_samples, sigmoid_weight=10.0, relu_weight=0.1) # TODO: add the other arguments here

	r_value = reg_fn()

	# r_value.backward()
	# for param in phi_fn.parameters():
	# 	print(param.grad)

	return r_value

def plot_a_k(exp_name, n_it):
	args = load_args("./log/%s/args.txt" % exp_name)
	dev = "cpu"
	device = torch.device(dev)

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-args.max_angular_velocity, args.max_angular_velocity]], dtype=np.float32) # TODO

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	if args.physical_difficulty == 'easy':  # medium length pole
		param_dict = {
			"I": 1.2E-3,
			"m": 0.127,
			"M": 1.0731,
			"l": 0.3365
		}
	elif args.physical_difficulty == 'hard':  # long pole
		param_dict = {
			"I": 7.88E-3,
			"m": 0.230,
			"M": 1.0731,
			"l": 0.6413
		}

	param_dict["theta_safe_lim"] = args.theta_safe_lim
	param_dict["max_force"] = args.max_force

	# param_dict = pickle.load(open("./log/%s/param_dict.pkl" % exp_name, "rb")) # TODO: replace with this

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	x_e = torch.zeros(1, x_dim)
	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)

	a_list = []
	k_list = []
	for checkpoint_number in np.arange(0, n_it, 50):
		phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
		load_model(phi_fn, phi_load_fpth)

		a = phi_fn.a[0, 0].item()
		k = phi_fn.ci[0, 0].item()

		a_list.append(a)
		k_list.append(k)

	plt.plot(a_list, label="A list")
	plt.plot(k_list, label="k list")
	plt.legend(loc="upper right")
	plt.savefig("./log/%s/a_k_plot.png" % exp_name)

if __name__=="__main__":
	# 11/19 experiment check-in
	# graph_log_file_2("cartpole_reduced_11_18_baseline")
	# graph_log_file_2("cartpole_reduced_64_64")
	# graph_log_file_2("cartpole_reduced_64_64_64")
	# graph_log_file_2("cartpole_reduced_64_64_gradient_avging")
	# graph_log_file_2("cartpole_reduced_64_64_xy")

	# exp_names = ["cartpole_reduced_11_18_baseline", "cartpole_reduced_64_64", "cartpole_reduced_64_64_64", "cartpole_reduced_64_64_gradient_avging", "cartpole_reduced_64_64_xy"]
	# n_it = [2500, 1000, 1000, 2500, 2500]
	#
	# for i, exp_name in enumerate(exp_names):
	# 	print(exp_name)
	# 	for checkpoint_number in np.arange(0, n_it[i], 50):
	# 		print(checkpoint_number)
	# 		plot_2d_attacks_from_loaded(checkpoint_number, exp_name)

	# 11/21 experiment check-in
	# exp_names = ["cartpole_reduced_64_64_gradient_avging_seed_2", "cartpole_reduced_64_64_gradient_avging_seed_3", "cartpole_reduced_64_64_gradient_avging_seed_4", "cartpole_reduced_64_64_gradient_avging_seed_5"]

	# n_it = [2000, 500, 800, 500]
	# for exp_name in exp_names:
	# 	graph_log_file_2(exp_name)

	# for i, exp_name in enumerate(exp_names):
	# 	for checkpoint_number in np.arange(0, n_it[i], 50):
	# 		plot_2d_attacks_from_loaded(checkpoint_number, exp_name)

	# exp_names.insert(0, "cartpole_reduced_64_64_gradient_avging")
	# for exp_name in exp_names:
	# 	graph_log_file_2(exp_name)

	# 11/23 here
	# checkpoint_number = 0
	# exp_name = "cartpole_reduced_64_64_gradient_avging_seed_2"
	#
	# for checkpoint_number in np.arange(0, 2500, 50):
	# 	r_value = test_reg_term(exp_name, checkpoint_number)
	# 	# print("Reg value: ", r_value)

	# exp_names = ["cartpole_reduced_64_64_gradient_avging_1", "cartpole_reduced_64_64_gradient_avging_10", "cartpole_reduced_64_64_gradient_avging_50", "cartpole_reduced_64_64_gradient_avging_100"]
	# n_it = [600, 1000, 1000, 1000]
	#
	# # for exp_name in exp_names:
	# # 	graph_log_file_2(exp_name)
	# #
	# # for i, exp_name in enumerate(exp_names):
	# # 	print(exp_name)
	# # 	for checkpoint_number in np.arange(0, n_it[i], 50):
	# # 		print(checkpoint_number)
	# # 		plot_2d_attacks_from_loaded(checkpoint_number, exp_name)
	#
	# plot_2d_attacks_from_loaded(990, "cartpole_reduced_64_64_gradient_avging_50")

	######################
	# exp_names = ["cartpole_reduced_64_64_40-1", "cartpole_reduced_64_64_40-2", "cartpole_reduced_64_64_64_gradient_avging_40-1", "cartpole_reduced_64_64_64_gradient_avging_40-2"]
	# n_it = [1440, 1440, 990, 980]

	# exp_names = ["cartpole_reduced_64_64_40-1", "cartpole_reduced_64_64_40-2", "cartpole_reduced_64_64_40-3", "cartpole_reduced_64_64_40-4", "cartpole_reduced_64_64_40-5"]
	# n_it = [1440, 1440, 1440, 1440, 1430]

	# exp_names = ["cartpole_reduced_64_64_gradient_avging_30-1", "cartpole_reduced_64_64_gradient_avging_30-2", "cartpole_reduced_64_64_gradient_avging_40-1", "cartpole_reduced_64_64_gradient_avging_40-2"]
	# n_it = [1270, 720, 1430, 1420]

	# for exp_name in exp_names:
	# 	graph_log_file_2(exp_name)

	# for i, exp_name in enumerate(exp_names):
	# 	for checkpoint_number in np.arange(0, n_it[i], 50):
	# 		print(checkpoint_number)
	# 		plot_2d_attacks_from_loaded(checkpoint_number, exp_name)

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

	exp_names = ["cartpole_reduced_64_64_40-1", "cartpole_reduced_64_64_40-2", "cartpole_reduced_64_64_40-3", "cartpole_reduced_64_64_40-4", "cartpole_reduced_64_64_40-5"]
	n_it = [1440, 1440, 1440, 1440, 1430]

	# for i, exp_name in enumerate(exp_names):
	# 	plot_a_k(exp_name, n_it[i])
	# 	plt.clf()
	# 	plt.cla()
	#
	# for i, exp_name in enumerate(exp_names):
	# 	for checkpoint_number in np.arange(0, n_it[i], 10):
	# 		print(checkpoint_number)
	# 		plot_2d_attacks_from_loaded(checkpoint_number, exp_name)

	# for i, exp_name in enumerate(exp_names):
	# 	for checkpoint_number in np.arange(0, n_it[i], 50):
	# 		print(checkpoint_number)
	# 		plot_3d(checkpoint_number, exp_name)

	# for checkpoint_number in np.arange(0, 1440, 50):
	# 	print(checkpoint_number)
	# 	plot_3d(checkpoint_number, "cartpole_reduced_64_64_40-1")

