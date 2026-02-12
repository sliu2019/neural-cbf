"""Regularization sampler for volume maximization."""
import logging
import numpy as np
import torch


class RegSampler():
	"""Samples states uniformly such that φ*(x) ≤ 0. This is a good-enough proxy for sampling states that are near the safe set boundary of φ*."""
	def __init__(self, x_lim: torch.Tensor, device: torch.device, n_samples: int = 250) -> None:
		"""
		Args:
			x_lim: State space bounds tensor (x_dim, 2) with columns [min, max]
			device: PyTorch device for tensor operations
			n_samples: Number of samples to generate per call (default: 250)
		"""
		self.x_lim = x_lim
		self.device = device
		self.n_samples = n_samples

		self.x_dim = x_lim.shape[0]
		self.x_lim_interval_sizes = np.reshape(x_lim[:, 1] - x_lim[:, 0], (1, self.x_dim))
		self.bs = 100  # Batch size for parallel φ evaluations

	def get_samples(self, phi_star_fn: torch.nn.Module) -> torch.Tensor:
		"""Use rejection sampling.

		Args:
			phi_star_fn: Neural CBF function (returns (bs, r+1) with ρ in column 0)

		Returns:
			Tensor (n_samples, x_dim) of states satisfying φ*(x) ≤ 0
		"""
		samples = torch.empty((0, self.x_dim), device=self.device)

		n_samp_found = 0
		while n_samp_found < self.n_samples:
			# Sample self.bs candidates uniformly in bounded state space
			candidate_samples_numpy = np.random.uniform(size=(self.bs, self.x_dim))*self.x_lim_interval_sizes + self.x_lim[:, [0]].T
			candidate_samples_torch = torch.from_numpy(candidate_samples_numpy.astype("float32")).to(self.device)

			# Keep samples where φ*(x) ≤ 0 (base safety specification)
			phi_vals = phi_star_fn(candidate_samples_torch)
			h_vals = phi_vals[:, 0]  # ρ(x) is the first component
			ind = torch.nonzero(h_vals <= 0).flatten()

			# Accumulate accepted samples
			samples_inside = candidate_samples_torch[ind]
			samples = torch.cat((samples, samples_inside), dim=0)

			n_samp_found += len(ind)

		# Truncate to exactly n_samples (rejection sampling may overshoot)
		samples = samples[:self.n_samples]

		return samples
