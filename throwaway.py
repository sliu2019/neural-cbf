"""
Scratch for one-off code
"""
import numpy as np
import IPython
import re
import matplotlib.pyplot as plt
import math
from main import Phi, Objective
from src.argument import parser, print_args
from src.utils import *
import torch
import pickle

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

def graph_log_file_2(exp_name):
	"""
	Corresponds to above
	"""
	# exp_name = "cartpole_reduced_exp1a"
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)
		timings = data["timings"]
		# test_losses = data["test_losses"]
		test_attack_losses = data["test_attack_losses"]
		test_reg_losses = data["test_reg_losses"]
		plt.plot(test_attack_losses, linewidth=0.5, label="Test attack loss")
		plt.plot(test_reg_losses, linewidth=0.5, label="Test reg loss")
		# plt.plot(test_losses, linewidth=0.5, label="Test loss")
	# plt.plot(timings, color='red', label="Runtime (hours)")
	plt.xlabel("Optimization steps")
	plt.legend(loc="upper right")
	plt.title("Statistics throughout training")

	plt.savefig("./log/%s/test_loss.png" % exp_name)
	plt.clf()
	plt.cla()

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
		"max_theta": math.pi / 10.0,
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
		"max_theta": math.pi / 2.0,
		"max_force": 15.0
	}

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	# n_samples = 50
	# rnge = torch.tensor([param_dict["max_theta"], x_lim[1:x_dim, 1]])
	# A_samples = torch.rand(n_samples, x_dim) * (2 * rnge) - rnge  # (n_samples, x_dim)
	x_e = torch.zeros(1, x_dim)

	phi_fn = Phi(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e)
	###################################
	# IPython.embed()
	delta = 0.01
	x = np.arange(-math.pi, math.pi, delta)
	y = np.arange(-5, 5, delta)[::-1] # need to reverse it
	X, Y = np.meshgrid(x, y)

	# phi_load_fpth = "./checkpoint/cartpole_reduced_exp1a/checkpoint_69.pth"
	phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
	load_model(phi_fn, phi_load_fpth)


	##### Testing ######
	# state_dict = phi_fn.beta_net.state_dict()  # 0.weight/bias and 2.weight/bias
	# # print(torch.mean(state_dict["4.weight"]))
	# print(state_dict["2.weight"])
	# # print(state_dict)
	######

	input = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis=1).astype(np.float32)
	input = torch.from_numpy(input)
	phi_vals = phi_fn(input)
	# IPython.embed()
	S_vals = torch.max(phi_vals, dim=1)[0] # S = all phi_i <= 0
	phi_signs = torch.sign(S_vals).detach().cpu().numpy()
	phi_signs = np.reshape(phi_signs, X.shape)


	fig, ax = plt.subplots()
	ax.imshow(phi_signs, extent=[-math.pi, math.pi, -5.0, 5.0])
	ax.set_aspect("equal")

	phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
	ax.contour(X, Y, np.reshape(phi_vals_numpy, X.shape), levels=[0.0],
	                 colors=('k',), linewidths=(2,))

	# Get attacks
	with open("./log/%s/data.pkl" % exp_name, 'rb') as handle:
		data = pickle.load(handle)
		train_attacks = data["train_attacks"]
		# IPython.embed()
		test_losses = data["test_losses"]
		plt.scatter(train_attacks[checkpoint_number][0], train_attacks[checkpoint_number][1], c="white", marker="x")
		# for j in range(checkpoint_number):
		# 	if test_losses[j] > 0:
		# 		plt.scatter(train_attacks[j][0], train_attacks[j][1], c="white", s=0.2)

	# attacks = np.load("./log/cartpole_reduced_debug/boundary_samples_post_opt.npy")
	# plt.scatter(attacks[:, 0], attacks[:, 1], c="white", marker="x")

	# print(attacks)
	plt.savefig("./log/%s/%s" % (exp_name, save_fnm))

if __name__=="__main__":
	# for checkpoint_number in np.arange(0, 30, 10):
	# 	save_fnm = "2d_checkpoint_%i.png" % checkpoint_number
	# 	plot_2d_binary(checkpoint_number, save_fnm, "cartpole_reduced_devel")

	# graph_log_file_2("cartpole_reduced_devel")

	# graph_log_file_2("cartpole_reduced_l_50_w_5e_4")
	# graph_log_file_2("cartpole_reduced_l_50_w_1")
	# graph_log_file_2("cartpole_reduced_l_50_w_5e_1")

	# graph_log_file_2("cartpole_reduced_l_50_w_1e_1")
	for checkpoint_number in np.arange(0, 410, 10):
		save_fnm = "2d_checkpoint_%i.png" % checkpoint_number
		plot_2d_binary(checkpoint_number, save_fnm, "cartpole_reduced_l_50_w_5e_1")
