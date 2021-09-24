import argparse


def parser():
	# Problem
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--problem', default='cartpole', help='problem specifies dynamics, h definition, U_limits, etc.', choices=["cartpole", "challenge"])

	# Phi
	# TODO: NN parametrization
	# TODO: ci initialization?

	###################################################################################################################################
	# Attacker: train
	parser.add_argument('--train_attacker', default='gradient_batch', choices=['basic', 'gradient_batch'])

	# Gradient batch attacker
	parser.add_argument('--train_attacker_n_samples', default=20)
	parser.add_argument('--train_attacker_stopping_condition', default='early_stopping', choices=['n_steps', 'early_stopping'])

	############################################################################
	# Attacker: test
	parser.add_argument('--test_attacker', default='gradient_batch', choices=['basic', 'gradient_batch'])

	# Gradient batch attacker
	parser.add_argument('--test_attacker_n_samples', default=30)
	parser.add_argument('--test_attacker_stopping_condition', default='early_stopping', choices=['n_steps', 'early_stopping'])
	###################################################################################################################################

	# Trainer
	# Saving/logging
	parser.add_argument('--affix', default='default', help='the affix for the save folder')
	parser.add_argument('--log_root', default='log',
	                    help='the directory to save the logs or other imformations (e.g. images)')
	parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
	parser.add_argument('--n_checkpoint_step', type=int, default=10,
	                    help='number of iterations to save a checkpoint')

	# TODO: add lr for ci

	# Misc
	parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')

	return parser.parse_args()


def print_args(args, logger=None):
	for k, v in vars(args).items():
		if logger is not None:
			logger.info('{:<16} : {}'.format(k, v))
		else:
			print('{:<16} : {}'.format(k, v))