import argparse
import math

def create_parser():
	# Problem
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--problem', default='cartpole_reduced', help='problem specifies dynamics, h definition, U_limits, etc.', choices=["cartpole", "flying_inv_pend", "cartpole_reduced"])

	# h(x) (user-specified SI)
	parser.add_argument('--h', type=str, default='sum', choices=['max', 'sum'], help='For flying inv pend, chose the form of h(x)')

	# Phi
	parser.add_argument('--phi_nn_dimension', default="64-64", type=str, help='specify the hidden dimension')
	parser.add_argument('--phi_nnl', default="tanh", type=str)
	parser.add_argument('--phi_ci_init_range', default=1e-2, type=float, help='c_i are initialized uniformly within the range [0, x]')
	# parser.add_argument('--phi_k0_init_min', default=0.0, type=float)
	# parser.add_argument('--phi_k0_init_max', default=1.0, type=float)
	parser.add_argument('--phi_include_xe', action='store_true')
	parser.add_argument('--phi_include_beta_deriv', action='store_true')

	# Parameters for cartpole only
	parser.add_argument('--physical_difficulty', default='easy', choices=['hard', 'easy'], help='long or medium pole')
	parser.add_argument('--max_angular_velocity', default=5.0, type=float) # between 1-10 lol
	parser.add_argument('--max_theta', default=math.pi/4.0, type=float)
	parser.add_argument('--max_force', default=22.0, type=float)

	# Parameters for flying cartpole only
	parser.add_argument('--pend_length', default=3.0, type=float)
	parser.add_argument('--box_ang_vel_limit', default=20.0, type=float)

	# Reg
	parser.add_argument('--reg_weight', default=1.0, type=float, help='the weight on the volume term')
	parser.add_argument('--reg_sample_distance', default=0.1, type=float)
	# parser.add_argument('--reg_xe', default=0.0, type=float) # deprecated

	# parser.add_argument('--g_input_is_xy', action='store_true')

	# Objective
	parser.add_argument('--no_softplus_on_obj', action='store_true', help='removes softplus on the objective')

	###################################################################################################################################
	# Reg sample keeper
	parser.add_argument('--reg_n_samples', default=50, type=int)

	###################################################################################################################################
	# Attacker: train
	parser.add_argument('--train_attacker', default='gradient_batch_warmstart', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart'])

	# Gradient batch attacker
	parser.add_argument('--train_attacker_n_samples', default=60, type=int)
	parser.add_argument('--train_attacker_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping'])
	parser.add_argument('--train_attacker_max_n_steps', default=50, type=int) # TODO: 200?
	parser.add_argument('--train_attacker_projection_tolerance', default=1e-1, type=float, help='when to consider a point "projected"')
	parser.add_argument('--train_attacker_projection_lr', default=1e-4, type=float)
	parser.add_argument('--train_attacker_projection_time_limit', default=3.0, type=float)
	parser.add_argument('--train_attacker_lr', default=1e-3, type=float)

	# parser.add_argument('--train_attacker_adaptive_lr', action='store_true')
	############################################################################
	# Attacker: test
	parser.add_argument('--test_attacker', default='gradient_batch', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart'])

	# Gradient batch attacker
	parser.add_argument('--test_attacker_n_samples', default=50, type=int)
	parser.add_argument('--test_attacker_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping'])
	parser.add_argument('--test_attacker_max_n_steps', default=200, type=int) # TODO
	parser.add_argument('--test_attacker_projection_tolerance', default=1e-1, type=float, help='when to consider a point "projected"')
	parser.add_argument('--test_attacker_projection_lr', default=1e-4, type=float)
	parser.add_argument('--test_attacker_lr', default=1e-3, type=float)
	# parser.add_argument('--test_attacker_adaptive_lr', action='store_true')
	###################################################################################################################################

	# Trainer
	parser.add_argument('--trainer_stopping_condition', default=['early_stopping'], choices=['n_steps', 'early_stopping'])
	parser.add_argument('--trainer_early_stopping_patience', default=100, type=int)
	parser.add_argument('--trainer_n_steps', default=1500, type=int, help='if stopping condition is n_steps, specify the number here')
	parser.add_argument('--trainer_lr', default=1e-3, type=float)
	parser.add_argument('--train_mode', default='dG', choices=['dG', 'dS'])
	parser.add_argument('--trainer_average_gradients', action='store_true')
	# parser.add_argument('--trainer_type', type=str, default="Adam")

	# Saving/logging
	parser.add_argument('--random_seed', default=1, type=int)
	parser.add_argument('--affix', default='default', help='the affix for the save folder')
	parser.add_argument('--log_root', default='log',
	                    help='the directory to save the logs or other imformations (e.g. images)')
	parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
	parser.add_argument('--n_checkpoint_step', type=int, default=10,
	                    help='number of iterations to save a checkpoint')
	parser.add_argument('--n_test_loss_step', type=int, default=-1,
	                    help='number of iterations to compute test loss; if negative, then never')

	# TODO: add lr for ci or Adam option; also for projection, etc.

	# Misc
	parser.add_argument('--gpu', '-g', default=0, type=int, help='which gpu to use')
	return parser 
	# return parser.parse_args()


def print_args(args, logger=None):
	for k, v in vars(args).items():
		if logger is not None:
			logger.info('{:<16} : {}'.format(k, v))
		else:
			print('{:<16} : {}'.format(k, v))