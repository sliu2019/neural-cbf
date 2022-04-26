import os

import IPython
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

def load_philow_and_params(exp_name=None, checkpoint_number=None):
	"""
	Same behavor as flying_plot_utils.py/load_phi_and_params
	:param exp_name:
	:param checkpoint_number:
	:return:
	"""
	# TODO: assume default param_dict even if we pass in exp_name, checkpoint_number
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

	# Send all modules to the correct device
	h_fn = h_fn.to(device)
	xdot_fn = xdot_fn.to(device)

	# Create CBF, etc.
	phi_fn = PhiLow(h_fn, xdot_fn, x_dim, u_dim, device, param_dict)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)

	if exp_name:
		print("inside loadphilow_and_params, reloading model part")
		IPython.embed()
		data = pickle.load(open(os.path.join("./cmaes", exp_name, "data.pkl")))
		mus = data["mu"]
		mu = mus[checkpoint_number]
		state_dict = {"ki": torch.tensor([[mu[2]]]), "ci": torch.tensor([[mu[0]], [mu[1]]])}
		phi_fn.load_state_dict(state_dict, strict=False)

	# TODO: not loading a checkpoint
	# if phi_load_fpth:
	# 	load_model(phi_fn, phi_load_fpth)

	return phi_fn, param_dict

def plot(exp_name):
	print(exp_name)
	data_fpth = "./cmaes/%s/data.pkl" % exp_name
	args_fpth = "./cmaes/%s/args.pkl" % exp_name
	data = pickle.load(open(data_fpth, "rb"))
	args = pickle.load(open(args_fpth, "rb"))

	# pop = data["pop"]
	mus = data["mu"]
	sigma = data["sigma"]
	# objective_value = data["obj:objective_value"]
	# n_near_boundary = data["obj:n_near_boundary"]


	n_it_so_far = len(mus)
	n_it_total = args["epoch"]
	# Plot mu, sigma
	#############################################
	fig, axs = plt.subplots(2, sharex=True)
	fig.suptitle('Mu/sigma for %s at iteration %i/%i' % (exp_name, n_it_so_far, n_it_total))

	axs[0].set_title("Mu")
	axs[0].plot([x[0] for x in mus], label="mus[0]", linewidth=0.5)
	axs[0].plot([x[1] for x in mus], label="mus[1]", linewidth=0.5)
	axs[0].plot([x[2] for x in mus], label="mus[2]", linewidth=0.5)

	axs[0].legend(loc=(1.04,0))

	axs[1].set_title("Sigma")
	axs[1].plot([x[0, 0] for x in sigma], label="sigma[0, 0]", linewidth=0.5)
	axs[1].plot([x[1, 1] for x in sigma], label="sigma[1, 1]", linewidth=0.5)
	axs[1].plot([x[2, 2] for x in sigma], label="sigma[2, 2]", linewidth=0.5)

	axs[1].plot([x[1, 0] for x in sigma], label="sigma[0, 0]", linewidth=0.5)
	axs[1].plot([x[2, 0] for x in sigma], label="sigma[1, 1]", linewidth=0.5)
	axs[1].plot([x[2, 1] for x in sigma], label="sigma[2, 2]", linewidth=0.5)

	print(np.min([x[0, 0] for x in sigma]))
	print(np.min([x[1, 1] for x in sigma]))
	print(np.min([x[2, 2] for x in sigma]))
	print(np.min(np.abs([x[1, 0] for x in sigma])))
	print(np.min(np.abs([x[2, 0] for x in sigma])))
	print(np.min(np.abs([x[2, 1] for x in sigma])))

	axs[1].legend(loc=(1.04,0))

	fig.tight_layout()
	plt.xlabel("Iterations") # aka opt. steps

	plt.savefig("./cmaes/%s/%s_mu_sigma.png" % (exp_name, exp_name))
	plt.clf()
	plt.cla()
	#############################################
	# Plot debug
	rewards = data["rewards"]
	elite_ratio = args["elite_ratio"]
	n_elite = int(elite_ratio*args["populate_num"])
	# IPython.embed()
	avg_elite_rewards = [np.mean(np.sort(x)[::-1][:n_elite]) for x in rewards]
	# IPython.embed()
	if "mu_rewards" not in data:
		from cmaes_objective_flying_pend import FlyingPendEvaluator
		evaluators_dict = {"FlyingPendEvaluator": FlyingPendEvaluator}

		# IPython.embed()
		evaluator = args["evaluator"] # converted from class name (str) to class object
		mu_rewards = []
		for mu in mus:
			print(mu)
			mu_reward, debug_dict = evaluator.evaluate(mu)
			mu_rewards.append(mu_reward)

		data["mu_rewards"] = mu_rewards
		with open(os.path.join("cmaes", exp_name, "data.pkl"), 'wb') as handle:
			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# print("here")
	# IPython.embed()
	mu_rewards = data["mu_rewards"]
	if type(mu_rewards[0]) is tuple:
		# woops, mistake with old baseline_run_cmaes.py
		mu_rewards = [x[0] for x in mu_rewards]
		# resave
		data["mu_rewards"] = mu_rewards
		with open(os.path.join("cmaes", exp_name, "data.pkl"), 'wb') as handle:
			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# IPython.embed()

	plt.plot(mu_rewards, label="mu rewards")
	plt.plot(avg_elite_rewards, label="avg elite rewards")
	plt.legend(loc=(1.04,0))

	plt.plot()
	plt.title("Rewards")
	plt.savefig("./cmaes/%s/%s_rewards.png" % (exp_name, exp_name))
	plt.clf()
	plt.cla()

	# Plot objective components
	# Useful for reg weight tuning
	# IPython.embed()
	fig, axs = plt.subplots(3, sharex=True)
	fig.suptitle('Fitness function components for %s at iteration %i/%i' % (exp_name, n_it_so_far, n_it_total))
	percent_inside = data["obj:percentage_inside"]
	mean_percent_inside = [np.mean(x) for x in percent_inside]
	max_percent_inside = [np.max(x) for x in percent_inside]
	min_percent_inside = [np.min(x) for x in percent_inside]

	axs[0].fill_between(np.arange(n_it_so_far), min_percent_inside, max_percent_inside, alpha=0.5, label="percent inside", color="orange")
	axs[0].plot(mean_percent_inside, color="orange")
	axs[0].set_title("Percent inside")

	obj_value = data["obj:objective_value"]
	mean_obj_value = [np.mean(x) for x in obj_value]
	max_obj_value = [np.max(x) for x in obj_value]
	min_obj_value = [np.min(x) for x in obj_value]

	axs[1].fill_between(np.arange(n_it_so_far), min_obj_value, max_obj_value, alpha=0.5, label="obj value", color="blue")
	axs[1].plot(mean_obj_value, color="blue")
	axs[1].set_title("Obj value")

	n_near_boundary = data["obj:n_near_boundary"]
	mean_n_near_boundary = [np.mean(x) for x in n_near_boundary]
	max_n_near_boundary = [np.max(x) for x in n_near_boundary]
	min_n_near_boundary = [np.min(x) for x in n_near_boundary]

	print(mean_n_near_boundary)

	axs[2].fill_between(np.arange(n_it_so_far), min_n_near_boundary, max_n_near_boundary, alpha=0.5, label="obj value", color="purple")
	axs[2].plot(mean_n_near_boundary, color="purple")
	axs[2].set_title("N_near_boundary")

	# plt.legend(loc=(1.04,0))

	# print("mean_percent_inside")
	# print(mean_percent_inside)
	# print("mean_obj_value")
	# print(mean_obj_value)

	# plt.plot()
	# plt.title("Fitness function components")
	plt.savefig("./cmaes/%s/%s_fitness.png" % (exp_name, exp_name))
	plt.clf()
	plt.cla()

if __name__ == "__main__":
	# pass

	# exp_names = ["flying_pend_v1_n_feasible", "flying_pend_v2_n_feasible", "flying_pend_v3_n_feasible", "flying_pend_v1_avg_amount_infeasible", "flying_pend_v2_avg_amount_infeasible", "flying_pend_v3_avg_amount_infeasible", "flying_pend_v1_max_amount_infeasible", "flying_pend_v2_max_amount_infeasible", "flying_pend_v3_max_amount_infeasible"]
	#
	# exp_names = ["flying_pend_n_feasible_reg_weight_1e_1", "flying_pend_n_feasible_reg_weight_5e_2", "flying_pend_n_feasible_reg_weight_1e_2", "flying_pend_avg_amount_infeasible_reg_weight_10", "flying_pend_avg_amount_infeasible_reg_weight_50", "flying_pend_avg_amount_infeasible_reg_weight_100", "flying_pend_max_amount_infeasible_reg_weight_15", "flying_pend_max_amount_infeasible_reg_weight_75", "flying_pend_max_amount_infeasible_reg_weight_150"]
	# exp_names = ["flying_pend_v1_avg_amount_infeasible"]

	exp_names = ["flying_pend_n_feasible_reg_weight_5e_2", "flying_pend_n_feasible_reg_weight_1e_2"]

	for exp_name in exp_names:
		plot(exp_name)



