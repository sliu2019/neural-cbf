import argparse
import math

def create_parser():
	# Problem
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--problem', default='flying_inv_pend', help='problem specifies dynamics, h definition, U_limits, etc.', choices=["cartpole", "flying_inv_pend", "cartpole_reduced", "quadcopter"])

	# h(x) (user-specified SI)
	parser.add_argument('--h', type=str, default='sum', choices=['max', 'sum'], help='For flying inv pend, chose the form of h(x)')

	# Phi
	parser.add_argument('--phi_design', default="neural", type=str, choices=["neural", "low"])
	parser.add_argument('--phi_nn_dimension', default="64-64", type=str, help='for neural CBF: specify the hidden dimension')
	parser.add_argument('--phi_nnl', default="tanh-tanh-none", type=str, help='for neural CBF: can also do tanh-tanh-softplus')
	parser.add_argument('--phi_ci_init_range', default=1e-2, type=float, help='for neural CBF: c_i are initialized uniformly within the range [0, x]')
	# parser.add_argument('--phi_k0_init_min', default=0.0, type=float)
	# parser.add_argument('--phi_k0_init_max', default=1.0, type=float)
	parser.add_argument('--phi_include_xe', action='store_true', help='for neural CBF')
	parser.add_argument('--phi_nn_inputs', type=str, default="spherical", choices=["spherical", "euc"], help='for neural CBF: which coordinates? spherical or euclidean')

	# parser.add_argument('--phi_reshape_h', action='store_true', help='reshape h')
	# parser.add_argument('--phi_reshape_dh', action='store_true', help="reshape dh by setting dh = d/dt(h + reshape). h will be reshape independently by default")
	# parser.add_argument('--phi_format', type=int, default=0, choices=[0, 1, 2])
	"""
	Style 0: phi = phi_0 + gnn 
	Style 1: phi = phi_0_star + k_1 dot(phi_0)
	Style 2: phi = phi_0_star + k_1 dot(phi_0_star)
	"""

	# Parameters for cartpole only
	parser.add_argument('--physical_difficulty', default='easy', choices=['hard', 'easy'], help='long or medium pole')
	parser.add_argument('--max_angular_velocity', default=5.0, type=float) # between 1-10 lol
	parser.add_argument('--max_theta', default=math.pi/4.0, type=float)
	parser.add_argument('--max_force', default=22.0, type=float)

	# Parameters for flying cartpole only
	parser.add_argument('--pend_length', default=3.0, type=float)
	parser.add_argument('--box_ang_vel_limit', default=20.0, type=float)

	# Reg
	parser.add_argument('--reg_weight', default=0.0, type=float, help='the weight on the volume term')
	parser.add_argument('--reg_sample_distance', default=0.1, type=float, help='grid sampling param for the cartpole task')
	parser.add_argument('--reg_sampler', type=str, default="random", choices=['boundary', 'random', 'fixed', 'random_inside', 'random_inside_then_boundary'], help="random_inside_then_boundary switches from RI to bdry after vol drops")
	parser.add_argument('--reg_n_samples', type=int, default=250)
	parser.add_argument('--reg_transform', type=str, default="sigmoid", choices=["sigmoid", "softplus"])
	# parser.add_argument('--reg_xe', default=0.0, type=float) # deprecated

	# parser.add_argument('--g_input_is_xy', action='store_true')

	# Objective
	# parser.add_argument('--no_softplus_on_obj', action='store_true', help='removes softplus on the objective')
	# parser.add_argument('--trainer_average_gradients', action='store_true')

	parser.add_argument('--objective_option', type=str, default='regular', choices=['regular', 'softplus', 'weighted_average', 'weighted_average_include_neg_phidot'], help="allow negative pays attention to phi < 0 as well")

	###################################################################################################################################
	# # Reg sample keeper
	# parser.add_argument('--reg_n_samples', default=50, type=int)

	###################################################################################################################################
	# Attacker: train
	parser.add_argument('--train_attacker', default='gradient_batch_warmstart', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart', 'gradient_batch_warmstart_faster'])
	# parser.add_argument('--gradient_batch_warmstart2_proj_tactic', choices=['gd_step_timeout', 'adam_ba'])
	parser.add_argument("--gradient_batch_warmstart_faster_speedup_method", type=str, default="sequential", choices=["sequential", "gpu_parallelized", "cpu_parallelized"])
	parser.add_argument("--gradient_batch_warmstart_faster_sampling_method", type=str, default="uniform", choices=["uniform", "gaussian"])
	parser.add_argument("--gradient_batch_warmstart_faster_gaussian_t", type=float, default=1.0) # TODO: could shrink as training progresses

	# Gradient batch attacker
	parser.add_argument('--train_attacker_n_samples', default=60, type=int)
	parser.add_argument('--train_attacker_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping'])
	parser.add_argument('--train_attacker_max_n_steps', default=50, type=int) # TODO: 200?
	parser.add_argument('--train_attacker_p_reuse', default=0.7, type=float) # TODO: 200?
	parser.add_argument('--train_attacker_projection_tolerance', default=1e-1, type=float, help='when to consider a point "projected"')
	parser.add_argument('--train_attacker_projection_lr', default=1e-2, type=float) # changed from 1e-4 to increase proj speed
	parser.add_argument('--train_attacker_projection_time_limit', default=3.0, type=float)
	parser.add_argument('--train_attacker_lr', default=1e-3, type=float)

	parser.add_argument('--train_attacker_use_n_step_schedule', action='store_true', help='use a schedule (starting with >>>max_n_steps and exponentially decreasing down to it')

	# parser.add_argument('--train_attacker_adaptive_lr', action='store_true')
	############################################################################
	# Attacker: test
	# parser.add_argument('--test_attacker', default='gradient_batch', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart'])
	#
	# # Gradient batch attacker
	# parser.add_argument('--test_attacker_n_samples', default=50, type=int)
	# parser.add_argument('--test_attacker_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping'])
	# parser.add_argument('--test_attacker_max_n_steps', default=200, type=int∑) # TODO
	# parser.add_argument('--test_attacker_projection_tolerance', default=1e-1, type=float, help='when to consider a point "projected"')
	# parser.add_argument('--test_attacker_projection_lr', default=1e-4, type=float)
	# parser.add_argument('--test_attacker_lr', default=1e-3, type=float)

	parser.add_argument('--test_N_volume_samples', default=2500, type=int)
	parser.add_argument('--test_N_boundary_samples', default=2500, type=int)
	###################################################################################################################################

	# Trainer
	parser.add_argument('--trainer_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping'])
	parser.add_argument('--trainer_early_stopping_patience', default=100, type=int)
	parser.add_argument('--trainer_n_steps', default=3000, type=int, help='if stopping condition is n_steps, specify the number here')
	parser.add_argument('--trainer_lr', default=1e-3, type=float)
	# parser.add_argument('--train_mode', default='dG', choices=['dG', 'dS'])
	# parser.add_argument('--trainer_type', type=str, default="Adam")
	# parser.add_argument('--trainer_lr_scheduler', type=str, choices=["exponential_reduction", "reduce_on_plateau"]) # TODO: not implemented
	# parser.add_argument('--trainer_lr_scheduler_exponential_reduction_gamma', type=float, default=0.992, help="the multiplicative factor; lr goes by alpha^t") # TODO: not implemented

	# Saving/logging
	parser.add_argument('--random_seed', default=1, type=int)
	parser.add_argument('--affix', default='default', help='the affix for the save folder')
	parser.add_argument('--log_root', default='log',
	                    help='the directory to save the logs or other imformations (e.g. images)')
	parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
	parser.add_argument('--n_checkpoint_step', type=int, default=5,
	                    help='number of iterations to save a checkpoint')
	parser.add_argument('--n_test_loss_step', type=int, default=25,
	                    help='number of iterations to compute test loss; if negative, then never')

	# TODO: add lr for ci or Adam option; also for projection, etc.

	# Misc
	parser.add_argument('--gpu', '-g', default=0, type=int, help='which gpu to use')
	return parser
	# return parser.parse_known_args()


def print_args(args, logger=None):
	for k, v in vars(args).items():
		if logger is not None:
			logger.info('{:<16} : {}'.format(k, v))
		else:
			print('{:<16} : {}'.format(k, v))