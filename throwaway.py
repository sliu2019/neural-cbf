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

def parse_log_file():
	"""
	Neglected to write test loss out to .npy file
	"""
	fpth = "./log/cartpole_default/train_log.txt"
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

		# IPython.embed()
		log_array = np.concatenate((np.array(test_losses)[None], np.array(times)[None]), axis=0)
		np.save("./log/cartpole_default/train_log.npy", log_array)

def graph_log_file():
	"""
	Corresponds to above
	"""
	log_array = np.load("./log/cartpole_default/train_log.npy")
	plt.plot(log_array[0], linewidth=0.5, label="Test loss")
	plt.plot(log_array[1], color='red', label="Runtime (hours)")
	plt.xlabel("Optimization steps")
	plt.legend(loc="upper right")
	plt.title("Statistics throughout training")

	plt.savefig("./log/cartpole_default/test_loss.png")

	# plt.ylabel("Test loss")

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


if __name__=="__main__":
	# parse_log_file()
	# graph_log_file()

	plot_phi_2d_level_curve_over_training()