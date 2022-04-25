"""
Mega-script to run all flying pendulum experiments
"""
import pickle
import os, sys, time, IPython
import torch
import numpy as np
import math

from phi_numpy_wrapper import PhiNumpy
from phi_low_torch_module import PhiLow

from cmaes.utils import load_philow_and_params
# todo: tweak load_philow
from flying_plot_utils import load_phi_and_params

# todo: what about logging/saving conventions?
# ssave onto one file (reload, modify, resave)?
# or save onto a bunch of different metrics? <- prefer this

from src.attacks.gradient_batch_attacker_warmstart_faster import GradientBatchWarmstartFasterAttacker
from main import Objective

# For rollouts
# from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
# from flying_cbf_controller import CBFController
from flying_rollout_experiment import *

# For plotting slices
from flying_plot_utils import plot_interesting_slices


def approx_volume(param_dict, cbf_obj, N_samp):
	# Or can just increase n_samp for now? Or just find points close to boundary?
	"""
	Uses rejection sampling to sample uniformly in the invariant set

	Note: assumes invariant set is defined as follows:
	x0 in S if max(phi_array(x)) <= 0
	"""
	# print("inside sample_inside_safe_set")
	# IPython.embed()

	# Define some variables
	x_dim = param_dict["x_dim"]
	x_lim = param_dict["x_lim"]
	box_side_lengths = x_lim[:, 1] - x_lim[:, 0]

	M = 50
	n_inside = 0
	for i in range(math.ceil(float(N_samp)/M)):
		# Sample in box
		samples = np.random.rand(M, x_dim)
		samples = samples*box_side_lengths + x_lim[:, 0]
		samples = np.concatenate((samples, np.zeros((M, 6))), axis=1) # Add translational states as zeros

		# Check if samples in invariant set
		phi_vals = cbf_obj.phi_fn(samples)
		max_phi_vals = phi_vals.max(axis=1)
		n_inside += np.sum(max_phi_vals <= 0)

	percent_inside = float(n_inside)/N_samp
	return percent_inside

def run_exps(args):
	"""
	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
		dev = "cuda:%i" % (args.gpu)
		# print("Using GPU device: %s" % dev)
	else:
		dev = "cpu"
	# dev = "cpu"
	device = torch.device(dev)
	"""
	##### Logging #####
	experiment_dict = {}
	###################

	device = torch.device("cpu")
	# load phi, phi_torch
	if args.which_cbf == "ours":
		# print("loading phi for ours")
		# IPython.embed()

		torch_phi_fn, param_dict = load_phi_and_params(exp_name=args.exp_name_to_load, checkpoint_number=args.checkpoint_number_to_load)
		numpy_phi_fn = PhiNumpy(torch_phi_fn)

		save_fldrpth = "./log/%s" % args.exp_name_to_load
	elif args.which_cbf == "low-CMAES":
		# print("loading phi for low-CMAES")
		# IPython.embed()
		torch_phi_fn, param_dict = load_philow_and_params() # TODO: this assumes default param_dict for dynamics
		numpy_phi_fn = PhiNumpy(torch_phi_fn)

		data = pickle.load(open(os.path.join("cmaes", args.exp_name_to_load, "data.pkl"), "rb"))
		mu = data["mu"][args.checkpoint_number_to_load]

		state_dict = {"ki": torch.tensor([[mu[2]]]), "ci": torch.tensor([[mu[0]], [mu[1]]])} # todo: this is not very generic
		numpy_phi_fn.set_params(state_dict)
		# print("check: does this mutate torch_phi_fn or not?") # yes

		save_fldrpth = "./cmaes/%s" % args.exp_name_to_load
	else:
		raise NotImplementedError

	########## Saving and logging ############
	save_fpth = os.path.join(save_fldrpth, "%s_exp_data.pkl"% args.save_fnm)

	#############################################
	##### Form the torch objective function #####
	#############################################
	# print("before forming objective function")
	# print("Check if the learned Phi are on GPU. Probably yes. In that case, is that enough? Usually, Phi class takes a device on instantiation")
	# IPython.embed()

	# r = param_dict["r"]
	x_dim = param_dict["x_dim"]
	u_dim = param_dict["u_dim"]
	# x_lim = param_dict["x_lim"]

	# Create phi
	from src.problems.flying_inv_pend import XDot, ULimitSetVertices
	xdot_fn = XDot(param_dict, device)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	xdot_fn = xdot_fn.to(device)
	uvertices_fn = uvertices_fn.to(device)

	torch_phi_fn = torch_phi_fn.to(device)
	logger = None # doesn't matter, isn't used
	obj_args = None # currently not used, but sometimes used to set options

	objective_fn = Objective(torch_phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, obj_args)
	objective_fn = objective_fn.to(device)

	# call separate functions for each test
	if "average_boundary" in args.which_experiments:
		# TODO: later, add more pass-in options for this
		print("average_boundary")
		IPython.embed()

		n_samples = 10 # TODO: increase
		torch_x_lim = torch.tensor(param_dict["x_lim"]).to(device)
		attacker = GradientBatchWarmstartFasterAttacker(torch_x_lim, device, None) # o.w. default args
		boundary_samples, debug_dict = attacker._sample_points_on_boundary(torch_phi_fn, n_samples) # todo: n_samples to arg?
		# outputs are in torch

		obj_values = objective_fn(boundary_samples)

		# Compute metrics
		n_infeasible = int(torch.sum(obj_values > 0))
		percent_infeasible = float(n_infeasible)/n_samples

		print("Check! that you're only adding scalars, not tensors to the dict")

		experiment_dict["percent_infeasible"] = percent_infeasible
		experiment_dict["n_infeasible"] = n_infeasible

		infeas_ind = torch.argwhere(obj_values > 0).flatten()
		mean_infeasible_amount = float(torch.mean(obj_values[infeas_ind]))
		std_infeasible_amount = float(torch.std(obj_values[infeas_ind]))

		experiment_dict["mean_infeasible_amount"] = mean_infeasible_amount
		experiment_dict["std_infeasible_amount"] = std_infeasible_amount

		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("Percent infeasible: %.3f" % percent_infeasible)
		print("Mean, std infeas. amount: %.3f +/- %.3f" % (mean_infeasible_amount, std_infeasible_amount))
	if "worst_boundary" in args.which_experiments:
		# TODO: later, add more pass-in options for this
		print("worst_boundary")
		IPython.embed()
		"""
		For now, you can use your attacker 
		(But of course, you can write a slower, better test-time attacker) 
		"""
		# if you called average boundary, reuse the attacker + it's initialized boundary points
		# TODO: examine parameters of attacker!
		n_opt_steps = 5 # TODO: increase
		n_samples = 20 # TODO

		torch_x_lim = torch.tensor(param_dict["x_lim"]).to(device)
		attacker = GradientBatchWarmstartFasterAttacker(torch_x_lim, device, None, max_n_steps=n_opt_steps, n_samples=n_samples) # o.w. default args
		iteration = 0 # dictates the number of grad steps, if you're using a step schedule. but we're not.
		x_worst, debug_dict = attacker.opt(objective_fn, torch_phi_fn, iteration, debug=False)

		x_worst = torch.reshape(x_worst, (1, 10))
		obj_values = objective_fn(x_worst)

		worst_infeasible_amount = torch.max(obj_values)
		experiment_dict["worst_infeasible_amount"] = worst_infeasible_amount

		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("Worst infeas. amount: %.3f" % worst_infeasible_amount)
	if "rollout" in args.which_experiments:
		# pass
		print("rollout")
		print("you're probably going to have a lot of dimension issues, since you switched classes")
		IPython.embed()
		"""
		you're probably going to have to refactor rollout to be more modular....
		"""
		# TODO, replacing old numpy wrapper OurCBF
		# The difference is that the new one is batched, which may cause some dimension issues
		# You may have to refactor the original file too
		# TODO: add rollout arguments below

		# Experimental settings
		N_desired_rollout = args.rollout_N_rollout
		T_max = args.rollout_T_max
		N_steps_max = int(T_max / args.rollout_dt)
		print("Number of timesteps: %f" % N_steps_max)

		# Create core classes: environment, controller
		env = FlyingInvertedPendulumEnv(param_dict)
		env.dt = args.rollout_dt
		cbf_controller = CBFController(env, numpy_phi_fn, param_dict) # 2nd arg prev. "cbf_obj"

		# print("before running rollouts")
		# IPython.embed()
		#####################################
		# Run multiple rollout_results
		#####################################
		if N_desired_rollout < 10:
			info_dicts = run_rollouts(env, N_desired_rollout, N_steps_max, cbf_controller)
		else:
			info_dicts = run_rollouts_multiproc(env, N_desired_rollout, N_steps_max, cbf_controller)

		#####################################
		# Compute numbers
		#####################################
		stat_dict = extract_statistics(info_dicts, env, param_dict)
		# print(percent_inside)
		# stat_dict["rollout_info_dicts"] = info_dicts

		# Fill out experiment dict
		experiment_dict["rollout_info_dicts"] = info_dicts
		experiment_dict["rollout_stat_dict"] = stat_dict

		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		for key, value in stat_dict.items():
			print("%s: %.3f" % (key, value))
	if "volume" in args.which_experiments:
		print("volume")
		IPython.embed()
		# Finally, approximate volume of invariant set
		# Note: don't use this: because N_samp is the number of samples we want to find inside
		# That means it could run forever
		# On the other hand, if you specify the N_samples, it's a predictable run time
		# _, percent_inside = sample_inside_safe_set(param_dict, numpy_phi_fn, args.N_samp_volume)
		percent_inside = approx_volume(param_dict, cbf_obj, N_samp)
		experiment_dict["vol_approximation"] = percent_inside

		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("Approx. volume: %.3f" % percent_inside)

	# Maybe analysis is better done in a different folder
	if "plot_slices" in args.which_analyses:
		plot_interesting_slices(torch_phi_fn, param_dict, save_fldrpth, args.checkpoint_number_to_load)

if __name__ == "__main__":
	# from cmaes.cmas_argument import create_parse
	import argparse

	parser = argparse.ArgumentParser(description='All experiments for flying pendulum')
	parser.add_argument('--save_fnm', type=str, default="debug", required=True)
	parser.add_argument('--which_cbf', type=str, choices=["ours", "low-CMAES"], required=True)

	parser.add_argument('--exp_name_to_load', type=str, required=True) # flying_inv_pend_first_run
	parser.add_argument('--checkpoint_number_to_load', type=int, help="for our CBF")

	parser.add_argument('--which_experiments', nargs='+', default=["average_boundary", "worst_boundary", "rollout", "volume"], type=str)
	parser.add_argument('--which_analyses', nargs='+', default=["plot_slices"], type=str) # TODO: add "animate_rollout" later s


	# For rollout_experiment, TODO: rename
	parser.add_argument('--rollout_N_rollout', type=int, default=10)
	parser.add_argument('--rollout_dt', type=float, default=1e-4)
	parser.add_argument('--rollout_T_max', type=float, default=1e-1)

	# Volume
	parser.add_argument('--N_samp_volume', type=int, default=100) # TODO: default 100k

	args = parser.parse_known_args()[0]

	# IPython.embed()
	run_exps(args)

"""
Debug 

# Ours 
python run_flying_pend_exps.py --save_fnm debug --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 400 --rollout_N_rollout 2 

(ckpt 200 or 400) 

python run_flying_pend_exps.py --save_fnm debug --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 400 --rollout_N_rollout 2 --which_experiments volume 


# Low-CMAES
python run_flying_pend_exps.py --save_fnm debug --which_cbf low-CMAES --exp_name_to_load flying_pend_v3_avg_amount_infeasible --checkpoint_number_to_load 10 --rollout_N_rollout 2
(ckpt 10 or 12) 
"""