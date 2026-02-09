"""Regularization sampler for volume maximization.

This module provides sampling functionality for the volume regularization term
in neural CBF training. Samples are drawn uniformly from the zero-sublevel set
of the base safety specification ρ(x) ≤ 0, then used to compute the
regularization loss that encourages the learned safe set to be large.

See liu23e.pdf Eq. 4 for the regularization objective.
"""
import torch
import IPython
import numpy as np
import sys

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *

class RegSampler():
	"""Samples states uniformly from base safety specification zero-sublevel set.

	Uses rejection sampling to generate states satisfying ρ(x) ≤ 0, where ρ
	is the base safety specification (e.g., maximum angle from vertical).
	These samples are used for volume regularization loss computation.

	The rejection sampling ensures samples are distributed according to the
	uniform distribution within the safe set defined by ρ.

	Attributes:
		x_lim: State space bounds (x_dim, 2) with [min, max] per dimension
		device: PyTorch device for computation
		logger: Logger instance (currently unused)
		n_samples: Number of samples to generate per call
		x_dim: State space dimension
		x_lim_interval_sizes: Width of each state dimension (1, x_dim)
		bs: Batch size for evaluating φ (default: 100)

	Note:
		Currently samples from ρ(x) ≤ 0, not the full modified CBF φ*(x) ≤ 0.
		This is intentional as noted in TODO comment.
	"""
	def __init__(self, x_lim, device, logger, n_samples=250):
		"""Initializes regularization sampler.

		Args:
			x_lim: State space bounds tensor (x_dim, 2) with columns [min, max]
			device: PyTorch device for tensor operations
			logger: Logger instance for debugging (currently unused)
			n_samples: Number of samples to generate per call (default: 250)
		"""
		self.x_lim = x_lim
		self.device = device
		self.logger = logger
		self.n_samples = n_samples

		self.x_dim = x_lim.shape[0]
		self.x_lim_interval_sizes = np.reshape(x_lim[:, 1] - x_lim[:, 0], (1, self.x_dim))
		self.bs = 100  # Batch size for parallel φ evaluations

	def get_samples(self, phi_fn):
		"""Generates samples uniformly from ρ(x) ≤ 0 region using rejection sampling.

		Algorithm:
		1. Sample candidates uniformly in state space box
		2. Evaluate ρ(x) on candidates (zeroth component of φ)
		3. Keep only samples where ρ(x) ≤ 0
		4. Repeat until n_samples collected

		Args:
			phi_fn: Neural CBF function (returns (bs, r+1) with ρ in column 0)

		Returns:
			Tensor (n_samples, x_dim) of states satisfying ρ(x) ≤ 0

		Note:
			TODO: Currently samples from ρ(x) ≤ 0 instead of max φ_i(x) ≤ 0.
			This is the base safety specification, not the full CBF zero-sublevel set.
		"""
		# Rejection sampling: sample candidates and keep those satisfying ρ(x) ≤ 0
		samples = torch.empty((0, self.x_dim), device=self.device)

		n_samp_found = 0
		i = 0
		while n_samp_found < self.n_samples:
			# Sample candidates uniformly in state space hypercube
			candidate_samples_numpy = np.random.uniform(size=(self.bs, self.x_dim))*self.x_lim_interval_sizes + self.x_lim[:, [0]].T
			candidate_samples_torch = torch.from_numpy(candidate_samples_numpy.astype("float32")).to(self.device)

			# Evaluate φ(x) to get ρ(x) = φ_0(x)
			phi_vals = phi_fn(candidate_samples_torch)

			# Check which samples satisfy safety condition
			# TODO: Should use max φ_i(x) ≤ 0, but currently uses ρ(x) ≤ 0
			# max_phi_vals = torch.max(phi_vals, dim=1)[0]
			# ind = torch.nonzero(max_phi_vals <= 0).flatten()
			h_vals = phi_vals[:, 0]  # ρ(x) is the first component
			ind = torch.nonzero(h_vals <= 0).flatten()  # Indices where ρ(x) ≤ 0

			# Accumulate accepted samples
			samples_inside = candidate_samples_torch[ind]
			samples = torch.cat((samples, samples_inside), dim=0)

			n_samp_found += len(ind)
			i += 1

		# Truncate to exactly n_samples (rejection sampling may overshoot)
		samples = samples[:self.n_samples]

		return samples
