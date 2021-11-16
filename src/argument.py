import argparse
import math

def parser():
	# Problem
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--problem', default='cartpole_reduced', help='problem specifies dynamics, h definition, U_limits, etc.', choices=["cartpole", "challenge", "cartpole_reduced"])

	# Phi
	parser.add_argument('--phi_nn_dimension', default="6", type=str, help='specify the hidden dimension')
	parser.add_argument('--phi_ci_init_range', default=1e-2, type=float, help='c_i are initialized uniformly within the range [0, x]')

	# parser.add_argument('--physical_difficulty', default='easy', choices=['hard', 'easy'])
	parser.add_argument('--max_angular_velocity', default=15.0, type=float)
	parser.add_argument('--max_theta', default=math.pi/4.0, type=float)
	parser.add_argument('--max_force', default=30.0, type=float)

	parser.add_argument('--objective_volume_weight', default=1.0, type=float, help='the weight on the volume term')

	parser.add_argument('--g_input_is_xy', action='store_true')
	###################################################################################################################################
	# Attacker: train
	parser.add_argument('--train_attacker', default='gradient_batch', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart'])

	# Gradient batch attacker
	parser.add_argument('--train_attacker_n_samples', default=20, type=int)
	parser.add_argument('--train_attacker_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping'])
	parser.add_argument('--train_attacker_max_n_steps', default=200, type=int) # TODO
	parser.add_argument('--train_attacker_projection_tolerance', default=1e-1, type=float, help='when to consider a point "projected"')
	parser.add_argument('--train_attacker_projection_lr', default=1e-4, type=float)
	parser.add_argument('--train_attacker_lr', default=1e-3, type=float)
	# parser.add_argument('--train_attacker_adaptive_lr', action='store_true')
	############################################################################
	# Attacker: test
	parser.add_argument('--test_attacker', default='gradient_batch', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart'])

	# Gradient batch attacker
	parser.add_argument('--test_attacker_n_samples', default=30, type=int)
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

	# Saving/logging
	parser.add_argument('--random_seed', default=1, type=int)
	parser.add_argument('--affix', default='default', help='the affix for the save folder')
	parser.add_argument('--log_root', default='log',
	                    help='the directory to save the logs or other imformations (e.g. images)')
	parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
	parser.add_argument('--n_checkpoint_step', type=int, default=10,
	                    help='number of iterations to save a checkpoint')
	parser.add_argument('--n_test_loss_step', type=int, default=10,
	                    help='number of iterations to compute test loss and save data')

	# TODO: add lr for ci or Adam option; also for projection, etc.

	# Misc
	parser.add_argument('--gpu', '-g', default=0, type=int, help='which gpu to use')

	return parser.parse_args()


def print_args(args, logger=None):
	for k, v in vars(args).items():
		if logger is not None:
			logger.info('{:<16} : {}'.format(k, v))
		else:
			print('{:<16} : {}'.format(k, v))