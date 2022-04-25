"""
Mega-script to run all flying pendulum experiments
"""
import pickle
import os, sys, time, IPython
import torch
import numpy as np

from phi_numpy_wrapper import PhiNumpy
from phi_low_torch_module import PhiLow

from cmaes.utils import load_philow
# todo: tweak load_philow
from flying_plot_utils import load_phi_and_params

# todo: what about logging/saving conventions?
# ssave onto one file (reload, modify, resave)?
# or save onto a bunch of different metrics? <- prefer this

from src.attacks.gradient_batch_attacker_warmstart_faster import GradientBatchWarmstartFasterAttacker
from main import Objective

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
	device = torch.device("cpu") # todo: is this fine? or too slow?
	# load phi, phi_torch
	if args.which_cbf == "ours":
		torch_phi_fn, param_dict = load_phi_and_params(exp_name=args.exp_name_to_load, checkpoint_number=args.checkpoint_number_to_load)
		numpy_phi_fn = PhiNumpy(torch_phi_fn)
	elif args.which_cbf == "low-CMAES":
		# todo: create param_dict here? and then pass it load
		# So basically separate the two parts of load_phiload that are (1) get param_dict and (2) create phi using param_dict into 2 separate functions
		torch_phi_fn = load_philow(phi_load_fpth=None) # TODO: this assumes default param_dict for dynamics
		numpy_phi_fn = PhiNumpy(torch_phi_fn)

		data = pickle.load(open(os.path.join("cmaes", args.exp_name_to_load, "data.pkl")))
		mu = data["mu"][-1] # todo: defaulting to last mu

		state_dict = {"ki": torch.tensor([[mu[2]]]), "ci": torch.tensor([[mu[0]], [mu[1]]])} # todo: this is not very generic
		numpy_phi_fn.set_params(state_dict)
		# todo: does this mutate torch_phi_fn or not?
		print("check todo")
		IPython.embed()

	# Form the objective function
	"""r = param_dict["r"]
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
	objective_fn = Objective(torch_phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args)
	reg_fn = Regularizer(phi_fn, device, reg_weight=args.reg_weight, reg_transform=args.reg_transform)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)
	objective_fn = objective_fn.to(device)"""
	# TODO: simin you are here! Some means to create Objective so i can evaluate it easily
	# TODO: i think what's tricky is we have Numpy vs. Torch, GPU vs, COU
	# call separate functions for each test
	if "average_boundary" in args.which_experiments:
		# logger = None, since it's never called
		n_samples = 100000
		attacker = GradientBatchWarmstartFasterAttacker(param_dict["x_lim"], device, None) # o.w. default args
		boundary_samples, debug_dict = attacker._sample_points_on_boundary(torch_phi_fn, n_samples) # todo: n_samples to arg?
		# outputs are in torch
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
	if "rollout" in args.which_experiments:
		pass
		"""
		you're probably going to have to refactor rollout to be more modular....
		"""
	if "volume" in args.which_experiments:
		pass

	# TODO:
	# Maybe analysis is better done in a different folder
	parser.add_argument('--which_analysis', nargs='+', default=["plot_slices", "animate_rollout"], type=str)

	# Collect data

	# Print and save


if __name__ == "__main__":
	# from cmaes.cmas_argument import create_parse
	import argparse

	parser = argparse.ArgumentParser(description='All experiments for flying pendulum')
	parser.add_argument('--which_cbf', type=str, choices=["ours", "low-CMAES"], required=True)

	parser.add_argument('--exp_name_to_load', type=str, required=True) # flying_inv_pend_first_run
	parser.add_argument('--checkpoint_number_to_load', type=int, help="for our CBF")

	parser.add_argument('--which_experiments', nargs='+', default=["average_boundary", "worst_boundary", "rollout", "volume"], type=str)
	parser.add_argument('--which_analysis', nargs='+', default=["plot_slices", "animate_rollout"], type=str)


	# For rollout_experiment, TODO: rename
	parser.add_argument('--N_rollout', type=int, default=10)
	parser.add_argument('--dt', type=float, default=1e-4)
	parser.add_argument('--T_max', type=float, default=1e-1)

	parser.add_argument('--N_samp_volume', type=int, default=100000)

	args = parser.parse_known_args()[0]

	# IPython.embed()
	run_exps(args)

"""
TODO: a couple of notes for sensible coding practices 

1. Remove numpy interface for updating state dict. That numpy wrapper is supposed to be agnostic 
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