"""Command-line argument parser for neural CBF synthesis experiments.

This module defines all hyperparameters for the neural Control Barrier Function
(CBF) training pipeline described in "Safe control under input limits with neural
control barrier functions" (liu23e, CoRL 2022).

The parameters control:
- Problem specification (currently: flying_inv_pend only)
- Regularization strategy (volume maximization weight and sampling)
- Critic configuration (counterexample search via projected gradient ascent)
- Learner settings (learning rate, stopping conditions, checkpointing)
- Testing and logging (volume estimation, boundary sampling, output paths)

Default values correspond to the configuration used in the paper experiments.

References:
    liu23e.pdf Section 3 (Method), Section 4 (Experiments)
"""
import argparse
import math

def create_arg_parser():
	"""Creates argument parser for neural CBF training experiments.

	Returns:
		argparse.ArgumentParser: Configured parser with all training hyperparameters.
		                         Call parser.parse_known_args()[0] to get args Namespace.

	Parameter Groups:
		Problem: System specification (flying_inv_pend)
		Regularization: Volume maximization parameters (weight, sampling strategy)
		Critic: Counterexample search configuration (samples, gradient steps)
		Learner: Training loop settings (LR, stopping conditions)
		Testing: Validation metrics (volume estimation, boundary sampling)
		Logging: Checkpointing and output paths

	Note:
		Many parameters have been hardcoded during refactoring (marked with TODO comments).
		Active parameters represent the final simplified configuration.
	"""
	# Problem specification
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--problem', default='flying_inv_pend', help='problem specifies dynamics, rho definition, U_limits, etc.', choices=["flying_inv_pend"])

	# parser.add_argument('--rho', type=str, default='sum', choices=['max', 'sum', 'reg'], help='Chose the form of rho(x). For flying inv pend, chose between max and sum. For quadcopter, chose between sum and regular') # TODO: hardcode 

	# Phi (neural CBF) parameters defining the structure of the CBF
	# parser.add_argument('--phi_design', default="neural", type=str, choices=["neural", "low"]) # TODO: hardcode
	# parser.add_argument('--phi_nn_dimension', default="64-64", type=str, help='for neural CBF: specify the hidden dimension') # TODO: hardcode
	# parser.add_argument('--phi_nnl', default="tanh-tanh-none", type=str, choices=) # TODO: hardcode tanh-tanh-softplus from ablation_server_5 and 04-22-24
	# parser.add_argument('--phi_ci_init_range', default=1e-2, type=float, help='for neural CBF: c_i are initialized uniformly within the range [0, x]')
	# parser.add_argument('--phi_include_xe', action='store_true', help='for neural CBF') # TODO: hardcode as True
	# parser.add_argument('--phi_nn_inputs', type=str, default="spherical", choices=["spherical", "euc"], help='for neural CBF: which coordinates? spherical or euclidean') # TODO: hardcode euc

	# Parameters for cartpole only
	# parser.add_argument('--physical_difficulty', default='easy', choices=['hard', 'easy'], help='long or medium pole') # TODO: hardcode
	# parser.add_argument('--max_angular_velocity', default=5.0, type=float) # between 1-10 lol # TODO: move this to the system spec 
	# parser.add_argument('--max_theta', default=math.pi/4.0, type=float) # TODO: move this to the system spec 
	# parser.add_argument('--max_force', default=22.0, type=float) # TODO: move this to the system spec 

	# Parameters for flying cartpole only
	# parser.add_argument('--pend_length', default=3.0, type=float) # TODO: move this to the system spec 
	# parser.add_argument('--box_ang_vel_limit', default=20.0, type=float) # TODO: move this to the system spec 

	###################################################################################################################################
	# Regularization parameters (volume maximization - Eq. 4 in liu23e.pdf)
	parser.add_argument('--reg_weight', default=150.0, type=float, help='Weight on volume regularization term. Higher values encourage larger safe sets.') 
	# parser.add_argument('--reg_sample_distance', default=0.1, type=float, help='grid sampling param for the cartpole task') # TODO: unused?
	# parser.add_argument('--reg_sampler', type=str, default="random", choices=['boundary', 'random', 'fixed', 'random_inside', 'random_inside_then_boundary'], help="random_inside_then_boundary switches from RI to bdry after vol drops") # TODO: harcode - random_inside in best run (4-22-24) repro
	parser.add_argument('--reg_n_samples', type=int, default=250, help='Number of samples for Monte Carlo volume approximation')
	# parser.add_argument('--reg_transform', type=str, default="sigmoid", choices=["sigmoid", "softplus"]) # TODO: hardcode - sigmoid in best run (4-22-24 repro) 

	# parser.add_argument('--objective_option', type=str, default='weighted_average', choices=['regular', 'softplus', 'weighted_average', 'weighted_average_include_neg_phidot'], help="allow negative pays attention to phi < 0 as well") # TODO: what's this? harcode default

	###################################################################################################################################
	# Critic parameters (counterexample search - Algorithm 1 in liu23e.pdf)
	# Critic finds states on the boundary with highest saturation risk using projected gradient ascent

	# parser.add_argument('--critic', default='gradient_batch_warmstart_faster', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart', 'gradient_batch_warmstart_faster']) # TODO: hardcode 
	# parser.add_argument("--gradient_batch_warmstart_faster_speedup_method", type=str, default="sequential", choices=["sequential", "gpu_parallelized", "cpu_parallelized"]) # TODO: hardcode sequential 
	# parser.add_argument("--gradient_batch_warmstart_faster_sampling_method", type=str, default="gaussian", choices=["uniform", "gaussian"]) # TODO: hardcode gaussian 
	# parser.add_argument("--gradient_batch_warmstart_faster_gaussian_t", type=float, default=1.0) # TODO: could shrink as training progresses

	# parser.add_argument('--critic', default='gradient_batch_warmstart_faster', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart', 'gradient_batch_warmstart_faster']) # TODO: hardcode 
	# parser.add_argument("--gradient_batch_warmstart_faster_speedup_method", type=str, default="sequential", choices=["sequential", "gpu_parallelized", "cpu_parallelized"]) # TODO: hardcode sequential 
	# parser.add_argument("--gradient_batch_warmstart_faster_sampling_method", type=str, default="gaussian", choices=["uniform", "gaussian"]) # TODO: hardcode gaussian 
	# parser.add_argument("--gradient_batch_warmstart_faster_gaussian_t", type=float, default=1.0) # TODO: could shrink as training progresses
	
	# Gradient batch critic configuration
	parser.add_argument('--critic_n_samples', default=50, type=int, help='Batch size for counterexample search (number of boundary points to optimize)')
	# parser.add_argument('--critic_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping']) # Hardcode n steps
	parser.add_argument('--critic_max_n_steps', default=20, type=int, help='Maximum gradient ascent steps for finding counterexamples') 
	# parser.add_argument('--critic_use_n_step_schedule', action='store_true', help='use a schedule (starting with >>>max_n_steps and exponentially decreasing down to it') # TODO: hardcode True - best in 04-22-24 repro run 
	# parser.add_argument('--critic_p_reuse', default=0.7, type=float) # TODO: 0.0 in 04-22 run 

	# parser.add_argument('--critic_lr', default=1e-3, type=float)

	# parser.add_argument('--critic_projection_tolerance', default=1e-1, type=float, help='when to consider a point "projected"') 
	# parser.add_argument('--critic_projection_lr', default=1e-2, type=float) # changed from 1e-4 to increase proj speed
	# parser.add_argument('--critic_projection_time_limit', default=3.0, type=float)
	
	############################################################################
	# Testing/Validation parameters
	parser.add_argument('--test_N_volume_samples', default=2500, type=int, help='Number of Monte Carlo samples for safe set volume estimation')
	parser.add_argument('--test_N_boundary_samples', default=2500, type=int, help='Number of boundary samples for checking saturation violations')
	###################################################################################################################################

	# Learner parameters (training loop configuration)
	parser.add_argument('--learner_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping'], help='Training termination criterion')
	parser.add_argument('--learner_early_stopping_patience', default=100, type=int, help='Patience for early stopping (if enabled)')
	parser.add_argument('--learner_n_steps', default=3000, type=int, help='Number of training iterations (if stopping condition is n_steps)')
	parser.add_argument('--learner_lr', default=1e-3, type=float, help='Adam learning rate for neural CBF parameters')

	# Saving/logging configuration
	parser.add_argument('--random_seed', default=1, type=int, help='Random seed for reproducibility (PyTorch and NumPy)')
	parser.add_argument('--affix',  type=str, default='default', help='Suffix for experiment folder names (e.g., problem_affix)')
	parser.add_argument('--log_root', default='log',
	                    help='Root directory for log files and training data')
	parser.add_argument('--model_root', default='checkpoint', help='Root directory for model checkpoints')
	parser.add_argument('--n_checkpoint_step', type=int, default=5,
	                    help='Save checkpoint every N iterations')
	parser.add_argument('--n_test_loss_step', type=int, default=25,
	                    help='Compute test statistics (volume, boundary violations) every N iterations')

	# TODO: add lr for ci or Adam option; also for projection, etc.

	# Hardware configuration
	parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU device ID to use for training (CUDA required)')
	return parser
	# return parser.parse_known_args()


def print_args(args, logger=None):
	"""Prints all arguments to console or logger.

	Args:
		args: argparse.Namespace with parsed arguments
		logger: Optional logging.Logger instance. If None, prints to console.

	Note:
		Useful for logging hyperparameters at the start of training runs.
	"""
	for k, v in vars(args).items():
		if logger is not None:
			logger.info('{:<16} : {}'.format(k, v))
		else:
			print('{:<16} : {}'.format(k, v))