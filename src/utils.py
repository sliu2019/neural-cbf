"""Utility functions and classes."""
import os
import json
import logging
from typing import Optional

import numpy as np
import torch
from torch import nn

from src.create_arg_parser import create_arg_parser, print_args


def create_logger(save_path: str = '', file_type: str = '', level: str = 'debug') -> logging.Logger:
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


def save_model(model: nn.Module, file_name: str) -> None:
	"""Saves PyTorch model state dict to file.

	Args:
		model: PyTorch nn.Module instance
		file_name: Path where model will be saved (.pth extension)
	"""
	torch.save(model.state_dict(), file_name)


def load_model(model: nn.Module, file_name: str) -> None:
	"""Loads PyTorch model state dict from file.

	Args:
		model: PyTorch nn.Module instance to load weights into
		file_name: Path to saved model checkpoint (.pth file)

	Note:
		Uses CPU mapping to enable loading GPU-trained models on CPU.
	"""
	model.load_state_dict(
		torch.load(file_name, map_location=lambda storage, loc: storage))


def makedirs(path: str) -> None:
	"""Creates directory if it doesn't exist (equivalent to mkdir -p).

	Args:
		path: Directory path to create
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def save_args(args, file_name: str) -> None:
	"""Saves command-line arguments to JSON file for reproducibility.

	Args:
		args: argparse.Namespace with parsed arguments
		file_name: Path to JSON file where args will be saved
	"""
	with open(file_name, 'w') as f:
		json.dump(args.__dict__, f, indent=2)


def load_args(file_name: str):
	"""Loads previously saved arguments from JSON file.

	Args:
		file_name: Path to JSON file containing saved arguments

	Returns:
		argparse.Namespace: Loaded arguments with defaults from parser
		                     overridden by saved values
	"""
	parser = create_arg_parser()
	args = parser.parse_known_args()[0]
	with open(file_name, 'r') as f:
		args.__dict__ = json.load(f)
	return args

class TransformEucNNInput(nn.Module):
	"""Transforms spherical coordinates to Euclidean for neural network input.

	Flying inverted pendulum uses spherical coordinates (angles), but neural
	networks work better with Euclidean representations. This transform converts:
	- quadcopter angles (α, β, γ) → direction vector (kx, ky, kz) + velocity
	- Pendulum angles (φ, θ) → Euclidean position + velocity

	This makes the representation more smooth and easier to learn.
	See paper Section 4 for coordinate system details.
q
	Attributes:
		state_index_dict: Mapping from state names to indices
		output_dim: Output dimension (12: 6 quad + 6 pendulum)
	"""
	def __init__(self, state_index_dict: dict) -> None:
		"""Initializes coordinate transform.

		Args:
			state_index_dict: Dict mapping state names (e.g., 'alpha', 'phi')
			                 to their indices in the state vector
		"""
		super().__init__()
		self.state_index_dict = state_index_dict
		self.output_dim = 12

	def forward(self, x: torch.Tensor) -> torch.Tensor:
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
