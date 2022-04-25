"""
Mega-script to run all flying pendulum experiments
"""
import pickle
import os, sys, time, IPython
import torch
import numpy as np

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
	save_fldrpth = ""
	save_fpth = os.join(save_fldrpth, "")# TODO
	###################

	device = torch.device("cpu") # todo: is this fine? or too slow?
	# load phi, phi_torch
	if args.which_cbf == "ours":
		torch_phi_fn, param_dict = load_phi_and_params(exp_name=args.exp_name_to_load, checkpoint_number=args.checkpoint_number_to_load)
		numpy_phi_fn = PhiNumpy(torch_phi_fn)
	elif args.which_cbf == "low-CMAES":
		# todo: create param_dict here? and then pass it load
		# So basically separate the two parts of load_phiload that are (1) get param_dict and (2) create phi using param_dict into 2 separate functions
		torch_phi_fn, param_dict = load_philow_and_params() # TODO: this assumes default param_dict for dynamics
		numpy_phi_fn = PhiNumpy(torch_phi_fn)

		data = pickle.load(open(os.path.join("cmaes", args.exp_name_to_load, "data.pkl")))
		mu = data["mu"][-1] # todo: defaulting to last mu

		state_dict = {"ki": torch.tensor([[mu[2]]]), "ci": torch.tensor([[mu[0]], [mu[1]]])} # todo: this is not very generic
		numpy_phi_fn.set_params(state_dict)
		# todo: does this mutate torch_phi_fn or not?
		print("check todo")
		IPython.embed()

	# Form the torch objective function
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
	# TODO: forgot if the learned Phi are on GPU. Probably yes. In that case, is that enough? Usually, Phi class takes a device on instantiation
	torch_phi_fn = torch_phi_fn.to(device)
	logger = None # doesn't matter, isn't used
	args = None # currently not used, but sometimes used to set options

	objective_fn = Objective(torch_phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args)
	objective_fn = objective_fn.to(device)

	# TODO: simin you are here! Some means to create Objective so i can evaluate it easily
	# TODO: i think what's tricky is we have Numpy vs. Torch, GPU vs, COU
	# call separate functions for each test
	if "average_boundary" in args.which_experiments:
		# TODO: everything in torch here
		# CPU? GPU?
		# logger = None, since it's never called
		print("average_boundary")
		IPython.embed()
		n_samples = 100000
		attacker = GradientBatchWarmstartFasterAttacker(param_dict["x_lim"], device, None) # o.w. default args
		boundary_samples, debug_dict = attacker._sample_points_on_boundary(torch_phi_fn, n_samples) # todo: n_samples to arg?
		# outputs are in torch

		obj_values = objective_fn(boundary_samples)

		# Compute metrics
		n_infeasible = torch.sum(obj_values > 0)
		percent_infeasible = float(n_infeasible)/n_samples

		print("Check! that you're only adding scalars, not tensors to the dict")

		experiment_dict["percent_infeasible"] = percent_infeasible
		experiment_dict["n_infeasible"] = n_infeasible

		infeas_ind = torch.argwhere(obj_values > 0).flatten()
		mean_infeasible_amount = torch.mean(obj_values[infeas_ind])
		std_infeasible_amount = torch.std(obj_values[infeas_ind])

		experiment_dict["mean_infeasible_amount"] = mean_infeasible_amount
		experiment_dict["std_infeasible_amount"] = std_infeasible_amount

		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		"""
		Create attacker w/ torch_phi_fn
		Call sample points on boundary  
		"""
	if "worst_boundary" in args.which_experiments:
		pass
		"""
		For now, you can use your attacker 
		(But of course, you can write a slower, better test-time attacker) 
		"""
		# TODO: you'll need an objective function for this too
		#     def opt(self, objective_fn, phi_fn, iteration, debug=False):
		# if you called average boundary, reuse the attacker + it's initialized boundary points
		attacker = GradientBatchWarmstartFasterAttacker(param_dict["x_lim"], device, None) # o.w. default args
		iteration = 0 # TODO: dictates the number of grad steps, if you're using a step schedule
		worst_boundary_samples, debug_dict = attacker.opt(objective_fn, torch_phi_fn, iteration, debug=False)

		obj_values = objective_fn(worst_boundary_samples)

		worst_infeasible_amount = torch.max(obj_values)
		experiment_dict["worst_infeasible_amount"] = worst_infeasible_amount

		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	if "rollout" in args.which_experiments:
		# pass
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
	if "volume" in args.which_experiments:
		# Finally, approximate volume of invariant set
		_, percent_inside = sample_inside_safe_set(param_dict, numpy_phi_fn, args.N_samp_volume)
		experiment_dict["vol_approximation"] = percent_inside

		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# TODO:
	# Maybe analysis is better done in a different folder
	if "plot_slices" in args.which_analysis:
		# TODO: check this
		plot_interesting_slices(torch_phi_fn, param_dict, save_fldrpth, args.checkpoint_number_to_load) # TODO: 2nd arg from arg obj? above we assume -1

	# Print and save
	# TODO: print


if __name__ == "__main__":
	# from cmaes.cmas_argument import create_parse
	import argparse

	parser = argparse.ArgumentParser(description='All experiments for flying pendulum')
	parser.add_argument('--which_cbf', type=str, choices=["ours", "low-CMAES"], required=True)

	parser.add_argument('--exp_name_to_load', type=str, required=True) # flying_inv_pend_first_run
	parser.add_argument('--checkpoint_number_to_load', type=int, help="for our CBF")

	parser.add_argument('--which_experiments', nargs='+', default=["average_boundary", "worst_boundary", "rollout", "volume"], type=str)
	parser.add_argument('--which_analysis', nargs='+', default=["plot_slices"], type=str) # TODO: add "animate_rollout" later s


	# For rollout_experiment, TODO: rename
	parser.add_argument('--rollout_N_rollout', type=int, default=10)
	parser.add_argument('--rollout_dt', type=float, default=1e-4)
	parser.add_argument('--rollout_T_max', type=float, default=1e-1)

	# Volume
	parser.add_argument('--N_samp_volume', type=int, default=100000)

	args = parser.parse_known_args()[0]

	# IPython.embed()
	run_exps(args)

"""
TODO: a couple of notes for sensible coding practices 

1. Remove numpy interface for updating state dict. That numpy wrapper is supposed to be agnostic
A: actually, the way it's implemented is agnostic 
 
2. What fully determines an experiment?
Resulting Phi 
Dynamics it assumed (param_dict) 

Relatedly: 
2.a) load_philow should have an input-output interface more like load_phi_and_params
We need param_dict out; it should take exp_name and checkpoint_number in 
You'll need to refactor all usages of it 

2.b) create a torch Objective 
It's a function of torch_Phi and param_dict only 

3. On refactoring rollout 
Cut and take lines 398-418, that will also get you the volume code 
Again, all of the things you need to instantiate only need numpy_phi and param_dict  

(Honestly, we probably should have built this script in that file. It's so similar!) 
"""