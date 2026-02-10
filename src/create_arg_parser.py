"""Command-line argument parser for neural CBF synthesis experiments.

This module defines all hyperparameters for the neural Control Barrier Function
(CBF) training pipeline described in "Safe control under input limits with neural
control barrier functions" (liu23e, CoRL 2022).

The parameters control:
- Problem specification (currently: quad_pend only)
- Regularization strategy (volume maximization weight and sampling)
- Critic configuration (counterexample search via projected gradient ascent)
- Learner settings (learning rate, stopping conditions, checkpointing)
- Testing and logging (volume estimation, boundary sampling, output paths)

Default values correspond to the configuration used in the paper experiments.

References:
    liu23e.pdf Section 3 (Method), Section 4 (Experiments)
"""
import argparse
import logging
from typing import Optional


def create_arg_parser() -> argparse.ArgumentParser:
	"""Creates argument parser for neural CBF training experiments.

	Returns:
		argparse.ArgumentParser: Configured parser with all training hyperparameters.
		                         Call parser.parse_known_args()[0] to get args Namespace.

	Parameter Groups:
		Problem: System specification (quad_pend)
		Regularization: Volume maximization parameters (weight, sampling strategy)
		Critic: Counterexample search configuration (samples, gradient steps)
		Learner: Training loop settings (LR, stopping conditions)
		Testing: Validation metrics (volume estimation, boundary sampling)
		Logging: Checkpointing and output paths
	"""
	# Problem specification
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--problem', default='quad_pend', help='problem specifies dynamics, rho definition, U_limits, etc.', choices=["quad_pend"])

	###################################################################################################################################
	# Regularization parameters (volume maximization - Eq. 4 in liu23e.pdf)
	parser.add_argument('--reg_weight', default=150.0, type=float, help='Weight on volume regularization term. Higher values encourage larger safe sets.')
	parser.add_argument('--reg_n_samples', type=int, default=250, help='Number of samples for Monte Carlo volume approximation')

	###################################################################################################################################
	# Critic parameters (counterexample search - Algorithm 1 in liu23e.pdf)
	# Critic finds states on the boundary with highest saturation risk using projected gradient ascent
	parser.add_argument('--critic_n_samples', default=50, type=int, help='Batch size for counterexample search (number of boundary points to optimize)')
	parser.add_argument('--critic_max_n_steps', default=20, type=int, help='Maximum gradient ascent steps for finding counterexamples')

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
