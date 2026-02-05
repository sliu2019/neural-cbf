"""
Mega-script to run all flying pendulum experiments
"""
import torch
import math

from phi_numpy_wrapper import PhiNumpy

# print("In flying pend exps")
# print(sys.path)
# import socket
# if socket.gethostname() == "nsh1609server4":
# 	# IPython.embed()
# 	sys.path.extend(['/home/simin/anaconda3/envs/si_feas_env/lib/python38.zip', '/home/simin/anaconda3/envs/si_feas_env/lib/python3.8', '/home/simin/anaconda3/envs/si_feas_env/lib/python3.8/lib-dynload', '/home/simin/anaconda3/envs/si_feas_env/lib/python3.8/site-packages'])
# from cmaes.utils import load_philow_and_params

from critic import Critic
from main import SaturationRisk

# For rollouts
# from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
# from flying_cbf_controller import CBFController
from flying_rollout_experiment import *

# For plotting slices
from flying_plot_utils import plot_interesting_slices

def approx_volume(param_dict, cbf_obj, N_samp, x_lim=None):
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
	if x_lim is None:
		x_lim = param_dict["x_lim"]
	else:
		x_lim_command_line = np.array(x_lim)
		x_lim = np.concatenate((x_lim_command_line[::2][:, None], x_lim_command_line[1::2][:, None]), axis=1)
		print("tried to pass in x_lim to approx_volume; needs sanity check")
		IPython.embed() # TODO

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

	# percent_inside = float(n_inside)*100.0/N_samp
	# return percent_inside

	fraction_inside = float(n_inside)/N_samp
	# TODO: assumes this is the default domain
	thresh = np.array([math.pi / 3, math.pi / 3, math.pi, 20, 20, 20, math.pi / 3, math.pi / 3, 20, 20],
	                  dtype=np.float32) # angular velocities bounds probably much higher in reality (~10-20 for drone, which can do 3 flips in 1 sec).
	default_x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)  # (13, 2)

	percent_of_domain_volume = fraction_inside*100*np.prod(box_side_lengths)/np.prod(default_x_lim[:, 1] - default_x_lim[:, 0])
	# approx_volume = np.prod(box_side_lengths)*fraction_inside

	# print("Check modification to volume calculation")
	# IPython.embed()
	data = {"percent_of_domain_volume": percent_of_domain_volume}
	return data

def bfs_approx_volume(param_dict, cbf_obj, axes_grid_size):
	"""
	Using breadth-first search on the state space grid to approximate volume
	Assumes we start from origin cell, but we could also sample a cell to start with.
	"""

	from queue import Queue
	# Q contains states, x
	x_dim = param_dict["x_dim"]
	x_lim = param_dict["x_lim"]

	queue = Queue()
	visited = set()

	start_node = tuple(np.zeros(x_dim)) # TODO

	queue.put(start_node)
	visited.add(start_node)

	def children(node):
		# In: tuple
		# Out: list of tuples
		# IPython.embed()
		np_node = np.reshape(np.array(node), (1, -1))
		potential_children = np.tile(np_node, (2*x_dim, 1))
		for i in range(x_dim):
			potential_children[2*i, i] = potential_children[2*i, i] - axes_grid_size[i]
			potential_children[2*i + 1, i] = potential_children[2*i + 1, i] + axes_grid_size[i]

		in_domain = np.logical_and(np.all(potential_children > x_lim[:, 0], axis=1), np.all(potential_children < x_lim[:, 1], axis=1))
		# print(in_domain)
		in_domain_inds = np.nonzero(in_domain)[0]
		children = potential_children[in_domain_inds]

		rv = [tuple(child) for child in children.tolist()]
		return rv

	# print("inside bfs approx volume, debugging")
	# IPython.embed()

	n_cells_occupied = 0
	while not queue.empty():
		# print(list(queue.queue)) # Prints queue elements
		current_node = queue.get()
		np_current_node = np.reshape(np.array(current_node), (1, -1))
		phi_vals = cbf_obj.phi_fn(np_current_node)
		max_phi_vals = phi_vals.max(axis=1)

		if max_phi_vals <= 0:
			print(n_cells_occupied)
			n_cells_occupied += 1
			for child_node in children(current_node):
				if child_node not in visited:
					queue.put(child_node)
					visited.add(child_node)

	cell_absolute_volume = np.prod(axes_grid_size)
	total_absolute_volume = n_cells_occupied*cell_absolute_volume
	percent_of_domain_volume = total_absolute_volume/np.prod(x_lim[:, 1] - x_lim[:, 0])
	data = {"n_cells_occupied": n_cells_occupied, "total_absolute_volume": total_absolute_volume, "cell_absolute_volume": cell_absolute_volume, "percent_of_domain_volume": percent_of_domain_volume}
	return data


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
	# if args.debug_mode:
		# IPython.embed()
		# args.boundary_n_samples = 10
		# args.worst_boundary_n_samples = 10
		# args.worst_boundary_n_opt_steps = 10
		# args.rollout_N_rollout = 2
		# args.N_samp_volume = 100

	if args.run_length == "short":
		args.boundary_n_samples = 10
		args.worst_boundary_n_samples = 10
		args.rollout_N_rollout = 10
		args.N_samp_volume = 100
	elif args.run_length == "medium": # 1k
		args.boundary_n_samples = 1000
		args.worst_boundary_n_samples = 1000
		args.rollout_N_rollout = 1000
		args.N_samp_volume = 100000 # 100k
	elif args.run_length == "long": # 10k
		args.boundary_n_samples = 10000
		args.worst_boundary_n_samples = 10000
		args.rollout_N_rollout = 5000 # TODO: 10k too Slow
		args.N_samp_volume = 1000000 # 1m

		args.worst_boundary_gaussian_t = 0.1 # TODO
		args.boundary_gaussian_t = 0.1 # TODO

	# print("Sanity check if you have set args.run_length")
	# IPython.embed() # TODO
	##### Logging #####
	experiment_dict = {}
	args_dict = vars(args)
	experiment_dict["args"] = args_dict
	###################

	device = torch.device("cpu")
	# load phi, phi_torch
	# if args.which_cbf == "ours":
	
	# print("loading phi for ours")
	# IPython.embed()

	torch_phi_fn, param_dict = load_phi_and_params(exp_name=args.exp_name_to_load, checkpoint_number=args.checkpoint_number_to_load)
	numpy_phi_fn = PhiNumpy(torch_phi_fn)

	save_fldrpth = "./log/%s" % args.exp_name_to_load

	# elif args.which_cbf == "low-CMAES":
	# 	# print("loading phi for low-CMAES")
	# 	# IPython.embed()
	# 	from cmaes.utils import load_philow_and_params
	# 	torch_phi_fn, param_dict = load_philow_and_params() # TODO: this assumes default param_dict for dynamics
	# 	numpy_phi_fn = PhiNumpy(torch_phi_fn)

	# 	data = pickle.load(open(os.path.join("cmaes", args.exp_name_to_load, "data.pkl"), "rb"))
	# 	mu = data["mu"][args.checkpoint_number_to_load]
	# 	print("*************************************************")
	# 	print("Parameters found by CMAES are:")
	# 	print(mu)
	# 	print("*************************************************")
	# 	# IPython.embed()

	# 	state_dict = {"ki": torch.tensor([[mu[2]]]), "ci": torch.tensor([[mu[0]], [mu[1]]])} # todo: this is not very generic
	# 	numpy_phi_fn.set_params(state_dict)

	# 	save_fldrpth = "./cmaes/%s" % args.exp_name_to_load
	# elif args.which_cbf == "low-heuristic":
	# 	from cmaes.utils import load_philow_and_params
	# 	torch_phi_fn, param_dict = load_philow_and_params() # TODO: this assumes default param_dict for dynamics
	# 	numpy_phi_fn = PhiNumpy(torch_phi_fn)

	# 	# print("loading phi for low-heuristic")
	# 	# IPython.embed()
	# 	mu = args.low_cbf_params
	# 	state_dict = {"ki": torch.tensor([[mu[2]]]), "ci": torch.tensor([[mu[0]], [mu[1]]])} # todo: this is not very generic
	# 	numpy_phi_fn.set_params(state_dict)

	# 	save_fldrpth = "./low_heuristic/k1_%.2f_k2_%.2f_k3_%.2f" % (mu[2], mu[0], mu[1]) # Note: named using Overleaf convention, but args are passed using another convention (swap 1st entry to end)
	# 	if not os.path.exists(save_fldrpth):
	# 		os.makedirs(save_fldrpth)

		# print("ln 156")
		# IPython.embed()
	# elif args.which_cbf == "low-gradient":
	# 	torch_phi_fn, param_dict = load_phi_and_params(exp_name=args.exp_name_to_load, checkpoint_number=args.checkpoint_number_to_load)
	# 	numpy_phi_fn = PhiNumpy(torch_phi_fn)

	# 	save_fldrpth = "./log/%s" % args.exp_name_to_load
	# 	# IPython.embed()
	# elif args.which_cbf == "iccbf":
	# 	coeffs = args.iccbf_coefficients
	# 	exps = args.iccbf_exponents

	# 	from src.phi_designs.iccbf import KappaPolynomial, ICCBF
	# 	assert len(coeffs) == len(exps)
	# 	class_kappa_fns = []
	# 	for i in range(len(coeffs)):
	# 		kappa_i = KappaPolynomial(coeffs[i], exps[i])
	# 		class_kappa_fns.append(kappa_i)

	# 	# Everything else
	# 	from src.problems.flying_inv_pend import RhoSum, XDot, ULimitSetVertices
	# 	# NOTE: USING DEFAULT PARAM DICT
	# 	from create_arg_parser import create_arg_parser
	# 	parser = create_arg_parser() # default
	# 	parser_args = parser.parse_known_args()[0]
	# 	from main import create_flying_param_dict
	# 	param_dict = create_flying_param_dict(parser_args) # default

	# 	h_fn = RhoSum(param_dict)
	# 	xdot_fn = XDot(param_dict, device)
	# 	uvertices_fn = ULimitSetVertices(param_dict, device)

	# 	torch_phi_fn = ICCBF(h_fn, xdot_fn, uvertices_fn, class_kappa_fns, param_dict["x_dim"], param_dict["u_dim"], device)
	# 	numpy_phi_fn = PhiNumpy(torch_phi_fn)

<<<<<<< HEAD
		save_fldrpth = "./iccbf_outputs/" + "_".join(["%.2f-%.2f" % (coeffs[i], exps[i]) for i in range(len(coeffs))])
		if not os.path.exists(save_fldrpth):
			os.makedirs(save_fldrpth)
		# IPython.embed()
	else:
		numpy_phi_fn = None
		save_fldrpth = None
		param_dict = None
		raise NotImplementedError
||||||| parent of be3c453 (deleted some unnecessary folders and files)
		save_fldrpth = "./iccbf_outputs/" + "_".join(["%.3f-%.3f" % (coeffs[i], exps[i]) for i in range(len(coeffs))])
		if not os.path.exists(save_fldrpth):
			os.makedirs(save_fldrpth)
		# print(save_fldrpth)
		# IPython.embed()
	else:
		numpy_phi_fn = None
		save_fldrpth = None
		param_dict = None
		raise NotImplementedError
=======
	# 	save_fldrpth = "./iccbf_outputs/" + "_".join(["%.3f-%.3f" % (coeffs[i], exps[i]) for i in range(len(coeffs))])
	# 	if not os.path.exists(save_fldrpth):
	# 		os.makedirs(save_fldrpth)
	# 	# print(save_fldrpth)
	# 	# IPython.embed()
	# else:
	numpy_phi_fn = None
	save_fldrpth = None
	param_dict = None
	raise NotImplementedError
>>>>>>> be3c453 (deleted some unnecessary folders and files)

	########## Saving and logging ############
	save_fpth = os.path.join(save_fldrpth, "%s_exp_data.pkl" % args.save_fnm)

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

	saturation_risk = SaturationRisk(torch_phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, obj_args)
	saturation_risk = saturation_risk.to(device)

	# call separate functions for each test
	if "average_boundary" in args.which_experiments:
		# print("average_boundary")
		# IPython.embed()

		n_samples = args.boundary_n_samples
		torch_x_lim = torch.tensor(param_dict["x_lim"]).to(device)
<<<<<<< HEAD
<<<<<<< HEAD
		attacker = GradientBatchWarmstartFasterAttacker(torch_x_lim, device, None, gaussian_t=args.boundary_gaussian_t, verbose=True) # o.w. default args
		boundary_samples, debug_dict = attacker._sample_points_on_boundary(torch_phi_fn, n_samples)
||||||| parent of c225af8 (renamed: h to rho, attacker to critic, trainer to learner, objective to sat risk. Should be more consistent with paper)
		attacker = GradientBatchWarmstartFasterAttacker(torch_x_lim, device, None, gaussian_t=args.boundary_gaussian_t, verbose=True, projection_lr=args.boundary_projection_lr) # o.w. default args
		boundary_samples, debug_dict = attacker._sample_points_on_boundary(torch_phi_fn, n_samples)
=======
		critic = GradientBatchWarmstartFasterCritic(torch_x_lim, device, None, gaussian_t=args.boundary_gaussian_t, verbose=True, projection_lr=args.boundary_projection_lr) # o.w. default args
||||||| parent of b7ac0a4 (in the middle of simplifying options and hardcoding best options)
		critic = GradientBatchWarmstartFasterCritic(torch_x_lim, device, None, gaussian_t=args.boundary_gaussian_t, verbose=True, projection_lr=args.boundary_projection_lr) # o.w. default args
=======
		critic = Critic(torch_x_lim, device, None, gaussian_t=args.boundary_gaussian_t, verbose=True, projection_lr=args.boundary_projection_lr) # o.w. default args
>>>>>>> b7ac0a4 (in the middle of simplifying options and hardcoding best options)
		boundary_samples, debug_dict = critic._sample_points_on_boundary(torch_phi_fn, n_samples)
>>>>>>> c225af8 (renamed: h to rho, attacker to critic, trainer to learner, objective to sat risk. Should be more consistent with paper)
		# boundary_samples = torch.rand((10000, 10))*100
		# debug_dict = {}

		obj_values = saturation_risk(boundary_samples)

		# Compute metrics
		n_infeasible = int(torch.sum(obj_values > 0))
		percent_infeasible = float(n_infeasible)*100.0/n_samples

		experiment_dict["percent_infeasible"] = percent_infeasible
		experiment_dict["n_infeasible"] = n_infeasible

		infeas_ind = torch.argwhere(obj_values > 0)[:, 0]
		mean_infeasible_amount = float(torch.mean(obj_values[infeas_ind]))
		std_infeasible_amount = float(torch.std(obj_values[infeas_ind]))

		experiment_dict["mean_infeasible_amount"] = mean_infeasible_amount
		experiment_dict["std_infeasible_amount"] = std_infeasible_amount
		experiment_dict["average_boundary_debug_dict"] = debug_dict

		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("Percent infeasible: %.3f" % percent_infeasible)
		print("Mean, std infeas. amount: %.3f +/- %.3f" % (mean_infeasible_amount, std_infeasible_amount))
	if "worst_boundary" in args.which_experiments:
		# print("worst_boundary")
		# IPython.embed()
		"""
		For now, you can use your critic 
		(But of course, you can write a slower, better test-time critic) 
		"""
		n_samples = args.worst_boundary_n_samples
		n_opt_steps = args.worst_boundary_n_opt_steps

		torch_x_lim = torch.tensor(param_dict["x_lim"]).to(device)
<<<<<<< HEAD
<<<<<<< HEAD
		attacker = GradientBatchWarmstartFasterAttacker(torch_x_lim, device, None, max_n_steps=n_opt_steps, n_samples=n_samples, gaussian_t=args.worst_boundary_gaussian_t, verbose=True, p_reuse=1.0) # o.w. default args
||||||| parent of c225af8 (renamed: h to rho, attacker to critic, trainer to learner, objective to sat risk. Should be more consistent with paper)
		attacker = GradientBatchWarmstartFasterAttacker(torch_x_lim, device, None, max_n_steps=n_opt_steps, n_samples=n_samples, gaussian_t=args.worst_boundary_gaussian_t, verbose=True, p_reuse=1.0, projection_lr=args.worst_boundary_projection_lr) # o.w. default args
=======
		critic = GradientBatchWarmstartFasterCritic(torch_x_lim, device, None, max_n_steps=n_opt_steps, n_samples=n_samples, gaussian_t=args.worst_boundary_gaussian_t, verbose=True, p_reuse=1.0, projection_lr=args.worst_boundary_projection_lr) # o.w. default args
>>>>>>> c225af8 (renamed: h to rho, attacker to critic, trainer to learner, objective to sat risk. Should be more consistent with paper)
||||||| parent of b7ac0a4 (in the middle of simplifying options and hardcoding best options)
		critic = GradientBatchWarmstartFasterCritic(torch_x_lim, device, None, max_n_steps=n_opt_steps, n_samples=n_samples, gaussian_t=args.worst_boundary_gaussian_t, verbose=True, p_reuse=1.0, projection_lr=args.worst_boundary_projection_lr) # o.w. default args
=======
		critic = Critic(torch_x_lim, device, None, max_n_steps=n_opt_steps, n_samples=n_samples, gaussian_t=args.worst_boundary_gaussian_t, verbose=True, p_reuse=1.0, projection_lr=args.worst_boundary_projection_lr) # o.w. default args
>>>>>>> b7ac0a4 (in the middle of simplifying options and hardcoding best options)
		iteration = 0 # dictates the number of grad steps, if you're using a step schedule. but we're not.

		if "average_boundary" in args.which_experiments:
			# reuse the boundary points computed there
			# saves time
			critic.X_saved = boundary_samples
			obj_vals = saturation_risk(boundary_samples.view(-1, 10)) # TODO: hard-coded dim
			critic.obj_vals_saved = obj_vals
		x_worst, debug_dict = critic.opt(saturation_risk, torch_phi_fn, iteration, debug=True)

		x_worst = torch.reshape(x_worst, (1, 10))
		obj_values = saturation_risk(x_worst)

		worst_infeasible_amount = torch.max(obj_values) # could be negative
		experiment_dict["worst_infeasible_amount"] = worst_infeasible_amount.detach().cpu().numpy()
		experiment_dict["worst_x"] = x_worst.detach().cpu().numpy()
		# experiment_dict["worst_boundary_debug_dict"] = debug_dict

		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("Worst infeas. amount: %.3f" % worst_infeasible_amount)
	if "rollout" in args.which_experiments:
		# print("rollout")
		# IPython.embed()

		# Experimental settings
		N_desired_rollout = args.rollout_N_rollout
		T_max = args.rollout_T_max
		N_steps_max = int(T_max / args.rollout_dt)
		print("Number of timesteps: %f" % N_steps_max)

		# Create core classes: environment, controller
		model_param_dict = param_dict

		# TODO: rebuttal update
		if args.mismatched_model_parameter is not None:
			# IPython.embed()
			real_param_dict = param_dict.copy()
			for i, param in enumerate(args.mismatched_model_parameter):
				real_param_dict[param] = args.mismatched_model_parameter_true_value[i]
			print(real_param_dict)
			env = FlyingInvertedPendulumEnv(model_param_dict=model_param_dict, real_param_dict=real_param_dict, dynamics_noise_spread=args.dynamics_noise_spread)
		else:
			env = FlyingInvertedPendulumEnv(model_param_dict=model_param_dict, dynamics_noise_spread=args.dynamics_noise_spread)

		env.dt = args.rollout_dt
		cbf_controller = CBFController(env, numpy_phi_fn, param_dict, args) # 2nd arg prev. "cbf_obj"

		#####################################
		# Run multiple rollout_results
		#####################################
		if N_desired_rollout < 10:
			info_dicts = run_rollouts(env, N_desired_rollout, N_steps_max, cbf_controller)
		else:
			info_dicts = run_rollouts_multiproc(env, N_desired_rollout, N_steps_max, cbf_controller, verbose=True, n_proc=args.n_proc)

		experiment_dict["rollout_info_dicts"] = info_dicts
		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		#####################################
		# Compute numbers
		#####################################
		stat_dict = extract_statistics(info_dicts, env, cbf_controller, param_dict)

		# Fill out experiment dict
		experiment_dict["rollout_stat_dict"] = stat_dict
		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		for key, value in stat_dict.items():
			print("%s: %.3f" % (key, value))
	if "volume" in args.which_experiments:
		# Finally, approximate volume of invariant set

		# print("volume")
		# IPython.embed()
		"""parser.add_argument('--volume_alg', type=str, choices=['sample', 'bfs_grid'], default='sample')
		# For sampling alg
		parser.add_argument('--N_samp_volume', type=int, default=100000)  # 100K
		parser.add_argument('--volume_x_lim', nargs='+',
		                    help="if you want to shrink x_lim to help the approx be better; this should just be a flat list of form [LB1, UB1, LB2, UB2, etc.]")  # 100K
		# For BFS alg
		parser.add_argument('--bfs_axes_grid_size', type=float, nargs='+')  # 100K"""
		if args.volume_alg == "sample":
			vol_data = approx_volume(param_dict, numpy_phi_fn, args.N_samp_volume, args.volume_x_lim)
		elif args.volume_alg == "bfs_grid":
			assert args.bfs_axes_grid_size is not None
			# print("before calling bfs approx volume")
			# IPython.embed()
			vol_data = bfs_approx_volume(param_dict, numpy_phi_fn, args.bfs_axes_grid_size) # TODO

		# experiment_dict["percent_of_domain_volume"] = percent_of_domain_volume
		# print("after calling bfs approx volume")
		# IPython.embed()
		experiment_dict.update(vol_data)
		with open(save_fpth, 'wb') as handle:
			pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("percent_of_domain_volume: %f" % vol_data["percent_of_domain_volume"])

	# Maybe analysis is better done in a different folder
	if "plot_slices" in args.which_analyses:
		plot_interesting_slices(torch_phi_fn, param_dict, save_fldrpth, args.checkpoint_number_to_load)

"""
Instructions for running:

Use run_length (short, medium, long) to select how many samples you want to use for boundary sampling, volume sampling, rollout sampling 
However, this does not set parameters like boundary_gaussian_t, so if you want to set those smaller (1 -> 0.1), do so separately. 

Common gotchas:
1. Did you check if the param_dicts for different CBF's are the same? They have to be, if you're comparing them
"""
if __name__ == "__main__":
	# from cmaes.cmas_argument import create_parse
	import argparse

	parser = argparse.ArgumentParser(description='All experiments for flying pendulum')
	parser.add_argument('--save_fnm', type=str, default="debug", help="conscisely describes the hyperparameters of this run")
	# parser.add_argument('--which_cbf', type=str, choices=["ours", "low-CMAES", "low-heuristic", "low-gradient", "iccbf"], required=True)

	parser.add_argument('--exp_name_to_load', type=str) # flying_inv_pend_first_run
	parser.add_argument('--checkpoint_number_to_load', type=int, help="for our CBF", default=0)
	parser.add_argument('--low_cbf_params', type=float, nargs='+', help="for which_cbf == low-heuristic")
	parser.add_argument('--iccbf_coefficients', type=float, nargs='+')
	parser.add_argument('--iccbf_exponents', type=float, nargs='+')

	parser.add_argument('--which_experiments', nargs='+', default=["average_boundary", "worst_boundary", "rollout", "volume"], type=str)
	parser.add_argument('--which_analyses', nargs='+', default=["plot_slices"], type=str) # TODO: add "animate_rollout" later

	# For boundary stats
	parser.add_argument('--boundary_n_samples', type=int, default=1000) # TODO
	parser.add_argument('--boundary_gaussian_t', type=float, default=1.0) # TODO

	# For worst boundary
	parser.add_argument('--worst_boundary_n_samples', type=int, default=1000) # TODO
	parser.add_argument('--worst_boundary_n_opt_steps', type=int, default=50) # TODO
	parser.add_argument('--worst_boundary_gaussian_t', type=float, default=1.0) # TODO

	# For rollout_experiment
	parser.add_argument('--rollout_N_rollout', type=int, default=500)
	parser.add_argument('--rollout_dt', type=float, default=1e-4)
	parser.add_argument('--rollout_T_max', type=float, default=1.0)
	parser.add_argument('--rollout_u_ref', type=str, choices=["unactuated", "LQR", "MPC"], default="unactuated")
	parser.add_argument('--rollout_LQR_q', type=float, default=0.1)
	parser.add_argument('--rollout_LQR_r', type=float, default=1.0)

	# For rollout robustness experiments
	parser.add_argument('--dynamics_noise_spread', type=float, default=0.0, help='set std dev of zero-mean, Gaussian noise')
	parser.add_argument('--mismatched_model_parameter', type=str, nargs='+')
	parser.add_argument('--mismatched_model_parameter_true_value', type=float, nargs='+')

	# Volume
	parser.add_argument('--volume_alg', type=str, choices=['sample', 'bfs_grid'], default='sample')
	# For sampling alg
	parser.add_argument('--N_samp_volume', type=int, default=100000) # 100K
	parser.add_argument('--volume_x_lim', nargs='+', help="if you want to shrink x_lim to help the approx be better; this should just be a flat list of form [LB1, UB1, LB2, UB2, etc.]") # 100K
	# For BFS alg
	parser.add_argument('--bfs_axes_grid_size', type=float, nargs='+') # 100K

	# In debug mode (use fewer samples for everything)
	# parser.add_argument('--debug_mode', action="store_true") # 100K
	parser.add_argument('--run_length', type=str, choices=["short", "medium", "long"], help="determines the number of samples (the run length)")

	# This only affects rollout, AFAIK
	parser.add_argument('--n_proc', type=int, default=36)

	args = parser.parse_known_args()[0]

	# IPython.embed()
	run_exps(args)

"""
python run_flying_pend_exps.py --save_fnm debug --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --boundary_n_samples 100 --worst_boundary_n_samples 100 --rollout_N_rollout 100 

Debug 

# Ours 
python run_flying_pend_exps.py --save_fnm debug --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 400 --rollout_N_rollout 2 

(ckpt 200 or 400) 

python run_flying_pend_exps.py --save_fnm debug --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 400 --rollout_N_rollout 2 --which_experiments volume 


# Low-CMAES
python run_flying_pend_exps.py --save_fnm debug --which_cbf low-CMAES --exp_name_to_load flying_pend_v3_avg_amount_infeasible --checkpoint_number_to_load 10 --rollout_N_rollout 2
(ckpt 10 or 12) 

python run_flying_pend_exps.py --save_fnm debug --which_cbf low-CMAES --exp_name_to_load flying_pend_n_feasible_reg_weight_1e_1 --checkpoint_number_to_load 6 --rollout_N_rollout 2 --which_experiments volume 

python run_flying_pend_exps.py --save_fnm debug --which_cbf low-CMAES --exp_name_to_load flying_pend_n_feasible_reg_weight_1e_1 --checkpoint_number_to_load 6 --rollout_N_rollout 2 --which_experiments rollout

# Best candidates for ours 
python run_flying_pend_exps.py --save_fnm debug --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020
"""

"""
For experiment flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0, at iteration 1551/3000
Average approx volume: 0.247
-0.08455947
1020 0.26794875
V avg: 0.031
First: 0.120, best: 0.320
For experiment flying_inv_pend_ESG_reg_speedup_better_attacks_seed_1, at iteration 1606/3000
Average approx volume: 0.234
0.49223045
1605 0.6197671
V avg: 0.028
First: 0.160, best: 0.360
For experiment flying_inv_pend_ESG_reg_speedup_better_attacks_seed_2, at iteration 2021/3000
Average approx volume: 0.200
0.38371262
1975 0.38371262
V avg: 0.028
First: 0.640, best: 0.640
For experiment flying_inv_pend_ESG_reg_speedup_better_attacks_seed_3, at iteration 1951/3000
Average approx volume: 0.185
0.25222638
930 0.54474854
V avg: 0.024
First: 0.000, best: 0.200
For experiment flying_inv_pend_ESG_reg_speedup_better_attacks_seed_4, at iteration 1186/3000
Average approx volume: 0.163
0.49849012
220 0.49849012
V avg: 0.017
First: 0.200, best: 0.200
(si_feas_env) siminl@nsh1609s
"""