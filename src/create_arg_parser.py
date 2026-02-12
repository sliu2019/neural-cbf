import argparse
import logging
from typing import Optional


def create_arg_parser() -> argparse.ArgumentParser:
	"""Creates argument parser for neural CBF training experiments.

	Returns:
		argparse.ArgumentParser: Configured parser with all training hyperparameters.
		                         Call parser.parse_known_args()[0] to get args Namespace.
	"""
	# Problem specification
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--problem', default='quad_pend', help='Problem specifies dynamics, safety function rho, input limits, etc.', choices=["quad_pend"])

	###################################################################################################################################
	# Regularization parameters (volume maximization - Eq. 7 in paper)
	parser.add_argument('--reg_weight', default=150.0, type=float, help='Weight on volume regularization term. Higher values encourage larger safe sets.')
	parser.add_argument('--reg_n_samples', type=int, default=250, help='Number of samples for Monte Carlo volume approximation')

	###################################################################################################################################
	# Critic parameters (counterexample search - function ComputeCE of Algorithm 1 in paper)
	parser.add_argument('--critic_n_samples', default=50, type=int, help='Batch size for counterexample search (number of boundary points to optimize)')
	parser.add_argument('--critic_max_n_steps', default=20, type=int, help='Maximum gradient ascent steps for finding counterexamples')

	###################################################################################################################################
	# Learner parameters (training loop configuration)
	parser.add_argument('--learner_lr', default=1e-3, type=float, help='Adam learning rate for neural CBF parameters')
	parser.add_argument('--learner_n_steps', default=3000, type=int, help='Number of training iterations (if stopping condition is n_steps)')

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

	# Hardware configuration
	parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU device ID to use for training (CUDA required)')
	return parser


def print_args(args: argparse.Namespace, logger: Optional[logging.Logger] = None) -> None:
	"""Prints all arguments to console or logger.

	Args:
		args: argparse.Namespace with parsed arguments
		logger: Optional logging.Logger instance. If None, prints to console.
	"""
	for k, v in vars(args).items():
		if logger is not None:
			logger.info('{:<16} : {}'.format(k, v))
		else:
			print('{:<16} : {}'.format(k, v))
