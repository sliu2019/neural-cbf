"""Utility functions and classes for neural CBF training.

This module provides supporting functionality for the neural Control Barrier Function
synthesis pipeline:
- Logging setup (console + file output)
- Model checkpointing (save/load PyTorch models)
- Early stopping for training convergence
- Input transformations for neural networks (coordinate system conversions)

The utilities support the training framework described in liu23e.pdf.
"""
import os
import json
import logging

import numpy as np
import torch
from torch import nn
import IPython
from dotmap import DotMap
import torch
import pickle
from src.create_arg_parser import create_arg_parser, print_args

def create_logger(save_path='', file_type='', level='debug'):
	"""Creates logger with both console and file output.

	Args:
		save_path: Directory path for log file. If empty, no file logging.
		file_type: Prefix for log filename (e.g., 'train' creates 'train_log.txt')
		level: Logging level ('debug' or 'info')

	Returns:
		logging.Logger: Configured logger instance
	"""

	if level == 'debug':
		_level = logging.DEBUG
	elif level == 'info':
		_level = logging.INFO

	logger = logging.getLogger()
	logger.setLevel(_level)

	cs = logging.StreamHandler()
	cs.setLevel(_level)
	logger.addHandler(cs)

	if save_path != '':
		file_name = os.path.join(save_path, file_type + '_log.txt')
		fh = logging.FileHandler(file_name, mode='w')
		fh.setLevel(_level)

		logger.addHandler(fh)

	return logger


def save_model(model, file_name):
	"""Saves PyTorch model state dict to file.

	Args:
		model: PyTorch nn.Module instance
		file_name: Path where model will be saved (.pth extension)
	"""
	torch.save(model.state_dict(), file_name)

def load_model(model, file_name):
	"""Loads PyTorch model state dict from file.

	Args:
		model: PyTorch nn.Module instance to load weights into
		file_name: Path to saved model checkpoint (.pth file)

	Note:
		Uses CPU mapping to enable loading GPU-trained models on CPU.
	"""
	model.load_state_dict(
		torch.load(file_name, map_location=lambda storage, loc: storage))

def makedirs(path):
	"""Creates directory if it doesn't exist (equivalent to mkdir -p).

	Args:
		path: Directory path to create
	"""
	if not os.path.exists(path):
		os.makedirs(path)

def save_args(args, file_name):
	"""Saves command-line arguments to JSON file for reproducibility.

	Args:
		args: argparse.Namespace with parsed arguments
		file_name: Path to JSON file where args will be saved
	"""
	with open(file_name, 'w') as f:
		json.dump(args.__dict__, f, indent=2)

def load_args(file_name):
	"""Loads previously saved arguments from JSON file.

	Args:
		file_name: Path to JSON file containing saved arguments

	Returns:
		argparse.Namespace: Loaded arguments with defaults from parser
		                     overridden by saved values
	"""
	parser = create_arg_parser()
	args = parser.parse_known_args()[0]
	# args = parser() # TODO
	with open(file_name, 'r') as f:
		args.__dict__ = json.load(f)
	return args


class EarlyStopping():
	"""Early stopping to terminate training when loss stops improving.

	Tracks best loss seen so far and counts consecutive epochs without
	sufficient improvement. Stops training after patience is exceeded.

	Attributes:
		patience: Number of epochs to wait before stopping
		min_delta: Minimum loss improvement to reset patience counter
		counter: Current count of non-improving epochs
		best_loss: Best loss value seen so far
		early_stop: Flag indicating whether to stop training
	"""
	def __init__(self, patience=3, min_delta=0):
		"""Initializes early stopping monitor.

		Args:
			patience: How many epochs to wait before stopping when loss
			         is not improving (default: 3)
			min_delta: Minimum loss decrease to count as improvement
			          (default: 0, any decrease counts)
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, test_loss):
		"""Updates early stopping state with new loss value.

		Args:
			test_loss: Current loss value (scalar or tensor)

		Side Effects:
			Updates self.best_loss, self.counter, and self.early_stop
		"""
		if self.best_loss == None:
			self.best_loss = test_loss
		elif self.best_loss - test_loss > self.min_delta:
			self.best_loss = test_loss
		elif self.best_loss - test_loss < self.min_delta:
			self.counter += 1
			# print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
			if self.counter >= self.patience:
				print('INFO: Early stopping')
				self.early_stop = True


class EarlyStoppingBatch():
	"""Batch-wise early stopping for adversarial optimization (maximization).

	Unlike regular EarlyStopping, this tracks improvement for each element in
	a batch independently. Training stops only when ALL batch elements have
	stopped improving. Used for critic's counterexample search where we
	maximize saturation risk across multiple candidate states.

	Note: This monitors MAXIMIZATION (increasing loss), not minimization.

	Attributes:
		patience: Number of steps to wait per batch element
		min_delta: Minimum loss increase to count as improvement
		counter: Per-element counter of non-improving steps (bs,)
		best_loss: Best loss per element (bs,)
		early_stop_vec: Per-element stopping flags (bs,)
		early_stop: Global flag (True when all elements stopped)
	"""
	def __init__(self, bs, patience=3, min_delta=1e-1):
		"""Initializes batch early stopping monitor.

		Args:
			bs: Batch size (number of independent optimization problems)
			patience: Steps to wait per element before marking as converged
			min_delta: Minimum loss increase to count as improvement (default: 0.1)
		"""
		self.patience = patience
		self.min_delta = min_delta

		self.counter = torch.zeros(bs)
		self.best_loss = None #torch.ones(bs)*float("inf")
		self.early_stop_vec = torch.zeros(bs, dtype=torch.bool)
		self.early_stop = False

	def __call__(self, test_loss):
		"""Updates early stopping state for each batch element.

		Args:
			test_loss: Current loss values for batch (bs, 1) tensor

		Side Effects:
			Updates best_loss, counter, early_stop_vec, and early_stop.
			Prints message when all elements have converged.
		"""
		if self.best_loss == None:
			self.best_loss = test_loss

		# print("Inside EarlyStoppingBatch")
		# IPython.embed()

		# print(self.best_loss, test_loss)
		improve_ind = torch.nonzero(test_loss - self.best_loss >= self.min_delta)
		nonimprove_ind = torch.nonzero(test_loss - self.best_loss < self.min_delta)
		self.best_loss[improve_ind] = test_loss[improve_ind]

		self.counter[nonimprove_ind] = self.counter[nonimprove_ind] + 1

		early_stop_ind = torch.nonzero(self.counter >= self.patience)
		self.early_stop_vec[early_stop_ind] = True

		# print(self.counter)
		if torch.all(self.early_stop_vec).item():
			print('INFO: Early stopping')
			self.early_stop = True

class IndexNNInput(nn.Module):
	"""Selects subset of input dimensions for neural network.

	Simple input preprocessor that selects specific state dimensions
	before passing to neural network. Used to train on partial state
	observations.

	Attributes:
		which_ind: Indices of dimensions to keep
		output_dim: Number of output dimensions
	"""
	def __init__(self, which_ind):
		"""Initializes dimension selector.

		Args:
			which_ind: Array or list of indices to select from input
		"""
		self.which_ind = which_ind
		self.output_dim = len(which_ind)

	def forward(self, x):
		"""Selects specified dimensions from input.

		Args:
			x: Input tensor (bs, input_dim)

		Returns:
			Tensor (bs, output_dim) with selected dimensions
		"""
		return x[:, self.which_ind]


class TransformEucNNInput(nn.Module):
	"""Transforms spherical coordinates to Euclidean for neural network input.

	Flying inverted pendulum uses spherical coordinates (angles), but neural
	networks work better with Euclidean representations. This transform converts:
	- quadcopter angles (α, β, γ) → direction vector (kx, ky, kz) + velocity
	- Pendulum angles (φ, θ) → Euclidean position + velocity

	This makes the representation more smooth and easier to learn.
	See liu23e.pdf Section 4 for coordinate system details.

	Attributes:
		state_index_dict: Mapping from state names to indices
		output_dim: Output dimension (12: 6 quad + 6 pendulum)
	"""
	def __init__(self, state_index_dict):
		"""Initializes coordinate transform.

		Args:
			state_index_dict: Dict mapping state names (e.g., 'alpha', 'phi')
			                 to their indices in the state vector
		"""
		super().__init__()
		self.state_index_dict = state_index_dict
		self.output_dim = 12

	def forward(self, x):
		"""Converts spherical state to Euclidean representation.

		Args:
			x: State tensor in spherical coordinates (bs, 10)
			   [γ, β, α, dγ, dβ, dα, φ, θ, dφ, dθ]

		Returns:
			Tensor (bs, 12) in Euclidean coordinates:
			[x_quad, y_quad, z_quad, vx_quad, vy_quad, vz_quad,
			 x_pend, y_pend, z_pend, vx_pend, vy_pend, vz_pend]
		"""
		alpha = x[:, self.state_index_dict["alpha"]]
		beta = x[:, self.state_index_dict["beta"]]
		gamma = x[:, self.state_index_dict["gamma"]]

		dalpha = x[:, self.state_index_dict["dalpha"]]
		dbeta = x[:, self.state_index_dict["dbeta"]]
		dgamma = x[:, self.state_index_dict["dgamma"]]

		phi = x[:, self.state_index_dict["phi"]]
		theta = x[:, self.state_index_dict["theta"]]

		dphi = x[:, self.state_index_dict["dphi"]]
		dtheta = x[:, self.state_index_dict["dtheta"]]

		# Convert quadcopter orientation (α, β, γ) to Euclidean direction vectors
		# These represent the quadcopter's attitude as a rotation matrix columns
		x_quad = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)
		y_quad = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
		z_quad = torch.cos(beta)*torch.cos(gamma)

		# Compute velocity of quadcopter direction vectors using chain rule
		# v = ∂pos/∂angles · angular_velocities
		d_x_quad_d_alpha = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
		d_x_quad_d_beta = -torch.cos(alpha)*torch.cos(beta)*torch.cos(gamma)
		d_x_quad_d_gamma = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma)
		v_x_quad = dalpha*d_x_quad_d_alpha + dbeta*d_x_quad_d_beta + dgamma*d_x_quad_d_gamma

		d_y_quad_d_alpha = -torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.sin(alpha)*torch.sin(gamma)
		d_y_quad_d_beta = -torch.sin(alpha)*torch.cos(beta)*torch.cos(gamma)
		d_y_quad_d_gamma = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma)
		v_y_quad = dalpha*d_y_quad_d_alpha + dbeta*d_y_quad_d_beta + dgamma*d_y_quad_d_gamma

		v_z_quad = dbeta*torch.sin(beta)*torch.cos(gamma) + dgamma*torch.cos(beta)*torch.sin(gamma)

		# Convert pendulum angles (φ, θ) to Euclidean position
		# Pendulum hangs below quadcopter, (φ, θ) define spherical coordinates
		x_pend = torch.sin(theta)*torch.cos(phi)
		y_pend = -torch.sin(phi)
		z_pend = torch.cos(theta)*torch.cos(phi)

		v_x_pend = -dtheta*torch.cos(theta)*torch.cos(phi) + dphi*torch.sin(theta)*torch.sin(phi)
		v_y_pend = dphi*torch.cos(phi)
		v_z_pend = dtheta*torch.sin(theta)*torch.cos(phi) + dphi*torch.cos(theta)*torch.sin(phi)

		rv = torch.cat([x_quad[:, None], y_quad[:, None], z_quad[:, None], v_x_quad[:, None], v_y_quad[:, None], v_z_quad[:, None], x_pend[:, None], y_pend[:, None], z_pend[:, None], v_x_pend[:, None], v_y_pend[:, None], v_z_pend[:, None]], dim=1)
		return rv
