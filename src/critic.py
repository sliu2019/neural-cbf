"""Critic for finding worst-case counterexamples via boundary optimization.

Approach: Projected Gradient Ascent on Boundary
1. Sample points on CBF zero-level set (Appendix Algorithm 2)
2. Perform projected gradient ascent to maximize saturation risk
3. Project back to boundary after each step to maintain φ(x)=0
4. Return worst-case counterexamples for learner
"""
import logging
import time
from typing import Any, Dict, Optional, Tuple
from collections.abc import Callable

import numpy as np
import torch
from torch.autograd import grad
import torch.optim as optim


class Critic():
	"""Finds worst-case counterexamples on CBF boundary via projected gradient ascent.

	The critic searches for states x where φ(x)=0 (on boundary) and saturation
	risk is maximized. Uses:
	- Gaussian sampling around safe set for initialization
	- Projected gradient ascent constrained to boundary manifold
	- Warmstart with previous counterexamples for efficiency

	Key Methods:
		opt(): Main optimization loop
		_sample_points_on_boundary(): Generate initial boundary samples
		_step(): Single projected gradient ascent step
		_project(): Project points onto φ=0 manifold

	Attributes:
		x_lim: State space bounds (x_dim, 2)
		device: PyTorch device
		logger: Logger instance
		n_samples: Batch size for counterexample search (default: 60)
		max_n_steps: Maximum gradient ascent iterations (default: 50)
		verbose: Print debugging info
		gaussian_t: Temperature for Gaussian sampling (default: 1.0)
		lr: Learning rate for gradient ascent (default: 1e-3)
		projection_tolerance: Convergence threshold for projection (default: 1e-1)
		projection_lr: LR for projection optimizer (default: 1e-2)
		projection_time_limit: Max time for single projection (default: 3.0s)
	"""
	def __init__(self, x_lim: torch.Tensor, device: torch.device, logger: logging.Logger,
	             n_samples: int = 60, max_n_steps: int = 50, verbose: bool = False) -> None:
		"""Initializes critic for counterexample search.

		Args:
			x_lim: State space bounds tensor (x_dim, 2)
			device: PyTorch device
			logger: Logger for debugging
			n_samples: Number of candidate counterexamples to optimize
			max_n_steps: Maximum gradient ascent steps per iteration
			verbose: Enable debug printing
		"""
		self.x_lim = x_lim
		self.device = device
		self.logger = logger
		self.n_samples = n_samples
		self.max_n_steps = max_n_steps
		self.verbose = verbose

		# Hardcoded hyperparameters (from best configuration)
		self.gaussian_t = 1.0  # Gaussian sampling temperature
		self.p_reuse = 0.0  # Warmstart reuse probability (disabled)
		self.projection_tolerance = 1e-1  # Projection convergence threshold
		self.projection_lr = 1e-2  # Projection optimizer LR
		self.projection_time_limit = 3.0  # Max projection time (seconds)
		self.lr = 1e-3  # Gradient ascent LR
		self.early_stopping_min_delta = 1e-3  # Early stopping threshold
		self.early_stopping_patience = 50  # Early stopping patience

		self.x_dim = self.x_lim.shape[0]

		# Compute relative volumes of hypercube facets (for uniform facet sampling)
		# Each facet is an (n-1)-dimensional hyperrectangle
		x_lim_interval_sizes = self.x_lim[:, 1] - self.x_lim[:, 0]
		x_lim_interval_sizes = x_lim_interval_sizes.view(1, -1)
		tiled = x_lim_interval_sizes.repeat(self.x_dim, 1)
		tiled = tiled - torch.eye(self.x_dim).to(self.device)*x_lim_interval_sizes + torch.eye(self.x_dim).to(device)
		vols = torch.prod(tiled, axis=1)
		vols = vols/torch.sum(vols)
		self.vols = vols.detach().cpu().numpy()  # For facet sampling probability
		self.hypercube_vol = torch.prod(x_lim_interval_sizes)

		# Warmstart: save counterexamples from previous iteration
		self.X_saved = None
		self.obj_vals_saved = None

		self.n_gpu = torch.cuda.device_count()

	def _project(self, phi_star_fn: Callable, X: torch.Tensor,
	             projection_n_grad_steps: Optional[int] = None) -> torch.Tensor:
		"""Projects points onto CBF zero-level set φ(x)=0 using gradient descent.

		Minimizes |φ(x)| via Adam optimizer until convergence or timeout.
		Used after gradient ascent steps to maintain boundary constraint.

		Args:
			phi_star_fn: Neural CBF function
			X: Points to project (list of tensors or single tensor)
			projection_n_grad_steps: Optional step limit (otherwise uses time limit)

		Returns:
			Projected points on boundary where φ(x) ≈ 0
		"""
		i = 0
		t1 = time.perf_counter()

		X_list = list(X)
		X_list = [X_mem.view(-1, self.x_dim) for X_mem in X_list]
		for X_mem in X_list:
			X_mem.requires_grad = True
		proj_opt = optim.Adam(X_list, lr=self.projection_lr)

		while True:
			proj_opt.zero_grad()

			loss = torch.sum(torch.abs(phi_star_fn(torch.cat(X_list), grad_x=True)[:, -1]))

			loss.backward()
			proj_opt.step()

			i += 1
			t_now = time.perf_counter()
			if torch.max(loss) < self.projection_tolerance:
				break

			if projection_n_grad_steps is not None:  # use step number limit
				if i == projection_n_grad_steps:
					break
			else:  # use time limit
				if (t_now - t1) > self.projection_time_limit:
					print("Attack: reprojection exited on timeout, max dist from =0 boundary: ", torch.max(loss).item())
					break

		for X_mem in X_list:
			X_mem.requires_grad = False
		rv_X = torch.cat(X_list)

		if self.verbose:
			if torch.max(loss) > self.projection_tolerance:
				print("Not on manifold, %f" % (torch.max(loss).item()))
		return rv_X

	def _step(self, saturation_risk: Callable, phi_star_fn: Callable,
	          X: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
		"""Single projected gradient ascent step on boundary manifold.

		Performs:
		1. Compute gradient of saturation risk: ∇_x L(x)
		2. Compute normal to manifold: n = ∇φ(x) / ||∇φ(x)||
		3. Project gradient: g_proj = g - (g·n)n (tangent to manifold)
		4. Take step: x_new = x - lr·g_proj
		5. Project x_new back onto boundary

		Args:
			saturation_risk: Loss function to maximize
			phi_star_fn: Neural CBF (defines manifold)
			X: Current boundary points (n_samples, x_dim)

		Returns:
			X_new: Updated points after projected gradient step
			debug_dict: Timing and convergence info
		"""
		t0_step = time.perf_counter()

		X_batch = X.view(-1, self.x_dim)
		X_batch.requires_grad = True

		obj_val = -saturation_risk(X_batch)  # maximizing
		obj_grad = grad([torch.sum(obj_val)], X_batch)[0]

		normal_to_manifold = grad([torch.sum(phi_star_fn(X_batch)[:, -1])], X_batch)[0]

		normal_to_manifold = normal_to_manifold/torch.norm(normal_to_manifold, dim=1)[:, None]  # normalize
		X_batch.requires_grad = False
		weights = obj_grad.unsqueeze(1).bmm(normal_to_manifold.unsqueeze(2))[:, 0]
		proj_obj_grad = obj_grad - weights*normal_to_manifold

		# Take a step
		X_new = X - self.lr*proj_obj_grad
		tf_grad_step = time.perf_counter()

		dist_before_proj = torch.mean(torch.abs(phi_star_fn(X_new)[:,-1]))
		X_new = self._project(phi_star_fn, X_new)
		dist_after_proj = torch.mean(torch.abs(phi_star_fn(X_new)[:,-1]))

		tf_reproject = time.perf_counter()

		# Wrap-around in state domain
		X_new = torch.minimum(torch.maximum(X_new, self.x_lim[:, 0]), self.x_lim[:, 1])
		dist_diff_after_proj = (dist_after_proj-dist_before_proj).detach().cpu().numpy()
		debug_dict = {"t_grad_step": (tf_grad_step-t0_step), "t_reproject": (tf_reproject-tf_grad_step), "dist_diff_after_proj": dist_diff_after_proj}

		return X_new, debug_dict

	def _sample_in_safe_set(self, phi_star_fn: Callable,
	                        random_seed: Optional[int] = None) -> torch.Tensor:
		"""Samples a single point uniformly inside the safe set φ(x) ≤ 0."""
		if random_seed:
			torch.manual_seed(random_seed)
			np.random.seed(random_seed)

		bs = 100
		N_samp_found = 0
		while N_samp_found < 1:
			# Sample in box
			unif = torch.rand((bs, self.x_dim)).to(self.device)
			samples_torch = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]

			# Check if samples in invariant set
			phi_vals = phi_star_fn(samples_torch)
			max_phi_vals = torch.max(phi_vals, dim=1)[0]

			# Save good samples
			ind = torch.argwhere(max_phi_vals <= 0).flatten()
			samples_torch_inside = samples_torch[ind]
			N_samp_found += len(ind)

		rv = samples_torch_inside[0]  # is flat shape already
		return rv

	def _sample_in_gaussian(self, safe_set_sample: torch.Tensor) -> torch.Tensor:
		"""Samples from Gaussian centered at a safe set point."""
		cov = 2*self.gaussian_t*torch.eye(self.x_dim).to(self.device)
		m = torch.distributions.MultivariateNormal(safe_set_sample, cov)
		sample_torch = m.sample()
		return sample_torch

	def _intersect_segment_with_manifold(self, p1: torch.Tensor, p2: torch.Tensor,
	                                     phi_star_fn: Callable) -> Optional[torch.Tensor]:
		"""Finds intersection of line segment [p1, p2] with CBF boundary φ(x)=0 via bisection."""
		diff = p2-p1

		left_weight = 0.0
		right_weight = 1.0
		left_val = phi_star_fn(p1.view(1, -1))[:, -1].item()
		right_val = phi_star_fn(p2.view(1, -1))[:, -1].item()
		left_sign = np.sign(left_val)
		right_sign = np.sign(right_val)

		if left_sign*right_sign > 0:
			return None

		t0 = time.perf_counter()
		while True:
			mid_weight = (left_weight + right_weight)/2.0
			mid_point = p1 + mid_weight*diff

			mid_val = phi_star_fn(mid_point.view(1, -1))[:, -1].item()
			mid_sign = np.sign(mid_val)
			if mid_sign*left_sign < 0:
				# go to the left side
				right_weight = mid_weight
				right_val = mid_val
			elif mid_sign*right_sign <= 0:
				left_weight = mid_weight
				left_val = mid_val

			if max(abs(left_val), abs(right_val)) < self.projection_tolerance:
				intersection_point = p1 + left_weight*diff
				break
			t1 = time.perf_counter()
			if (t1-t0) > 7:  # an arbitrary time limit
				print("Something is wrong in projection")
				print(left_val, right_val)
				print(left_weight, right_weight)
				print("p1:", p1)
				print("p2:", p2)
				return None
		return intersection_point

	def _sample_segment_intersect_boundary(self, phi_star_fn: Callable,
	                                       random_seed: Optional[int] = None) -> Optional[torch.Tensor]:
		"""Samples point on boundary using Gaussian sampling strategy.

		Algorithm (liu23e.pdf Appendix B.3):
		1. Sample x_safe inside safe set (φ < 0)
		2. Sample x_outer from Gaussian centered at x_safe
		3. Find intersection of segment [x_safe, x_outer] with boundary

		This biases sampling toward regions near safe set, improving
		efficiency when safe set is small.

		Args:
			phi_star_fn: Neural CBF
			random_seed: Optional seed for reproducibility

		Returns:
			Point on boundary (x_dim,) where φ(x)≈0, or None if no intersection
		"""
		center = self._sample_in_safe_set(phi_star_fn, random_seed=random_seed)
		outer = self._sample_in_gaussian(center)
		intersection = self._intersect_segment_with_manifold(center, outer, phi_star_fn)
		return intersection

	def _sample_points_on_boundary_sequential(self, phi_star_fn: Callable,
	                                          n_samples: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
		"""Generates n_samples points on CBF boundary via rejection sampling.

		Repeatedly calls _sample_segment_intersect_boundary until n_samples
		successful intersections found.

		Args:
			phi_star_fn: Neural CBF
			n_samples: Number of boundary points to generate

		Returns:
			samples: Tensor (n_samples, x_dim) on boundary
			debug_dict: Timing and efficiency statistics
		"""
		t0 = time.perf_counter()
		samples = torch.zeros((0, self.x_dim)).to(self.device)
		n_remaining_to_sample = n_samples

		n_segments_sampled = 0
		while n_remaining_to_sample > 0:
			if self.verbose:
				print(".", end=" ")
			intersection = self._sample_segment_intersect_boundary(phi_star_fn)
			if intersection is not None:
				samples = torch.cat((samples, intersection.view(1, -1)), dim=0)
				n_remaining_to_sample -= 1
				if self.verbose:
					print("\n")
					print(n_remaining_to_sample)

			n_segments_sampled += 1

		tf = time.perf_counter()
		debug_dict = {"t_sample_boundary": (tf- t0), "n_segments_sampled": n_segments_sampled}
		return samples, debug_dict

	def opt(self, saturation_risk: Callable, phi_star_fn: Callable, iteration: int,
	        debug: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
		"""Main optimization loop to find worst-case counterexamples.

		Algorithm:
		1. Initialize: Sample n_samples points on boundary (with optional warmstart)
		2. Iterate: Perform projected gradient ascent to maximize saturation_risk
		3. Return: Worst counterexample (state with highest saturation risk)

		Uses exponential schedule for max_n_steps: starts high, decays over iterations
		to save computation as CBF improves.

		Args:
			saturation_risk: Loss function L(x) to maximize
			phi_star_fn: Neural CBF defining boundary
			iteration: Current training iteration (for step schedule)
			debug: If True, return detailed debug information

		Returns:
			x: Worst counterexample (x_dim,) with highest saturation risk
			debug_dict: Timing, convergence, and sample info (if debug=True)
		"""
		t0_opt = time.perf_counter()

		if self.X_saved is None:
			X_init, boundary_sample_debug_dict = self._sample_points_on_boundary_sequential(phi_star_fn, self.n_samples)

			X_reuse_init = torch.zeros((0, self.x_dim))
			X_random_init = X_init
		else:
			n_target_reuse_samples = int(self.n_samples*self.p_reuse)

			inds = torch.argsort(self.obj_vals_saved, axis=0, descending=True).flatten()

			# Select distinct attacks (exclude near-duplicate points)
			inds_distinct = [inds[0]]
			for ind in inds[1:]:
				diff = self.X_saved[torch.tensor(inds_distinct)] - self.X_saved[ind]
				distances = torch.norm(diff.view(-1, self.x_dim), dim=1)
				if torch.any(distances <= 1e-1).item():
					continue
				inds_distinct.append(ind)
				if len(inds_distinct) >= n_target_reuse_samples:
					break

			n_reuse_samples = len(inds_distinct)
			n_random_samples = self.n_samples - n_reuse_samples
			X_reuse_init = self.X_saved[torch.tensor(inds_distinct)]
			X_reuse_init = self._project(phi_star_fn, X_reuse_init)  # reproject, since phi changed
			X_random_init, boundary_sample_debug_dict = self._sample_points_on_boundary_sequential(phi_star_fn, n_random_samples)
			X_init = torch.cat([X_random_init, X_reuse_init], axis=0)

		tf_init = time.perf_counter()

		X = X_init.clone()
		i = 0
		# logging
		t_grad_step = []
		t_reproject = []
		dist_diff_after_proj = []
		obj_vals = saturation_risk(X.view(-1, self.x_dim))
		init_best_attack_value = torch.max(obj_vals).item()

		max_n_steps = (0.5*self.max_n_steps)*np.exp(-iteration/75) + self.max_n_steps
		print("Max_n_steps: %i" % max_n_steps)
		while True:
			if self.verbose:
				print("Counterex. max. step #%i" % i)
			X, step_debug_dict = self._step(saturation_risk, phi_star_fn, X)

			# Logging
			t_grad_step.append(step_debug_dict["t_grad_step"])
			t_reproject.append(step_debug_dict["t_reproject"])
			dist_diff_after_proj.append(step_debug_dict["dist_diff_after_proj"])

			if (i > max_n_steps):
				break
			i += 1

		tf_opt = time.perf_counter()

		# Save for warmstart
		self.X_saved = X
		obj_vals = saturation_risk(X.view(-1, self.x_dim))
		self.obj_vals_saved = obj_vals

		# Return single worst-case attack
		max_ind = torch.argmax(obj_vals)

		if not debug:
			x = X[max_ind]
			return x, {}
		else:
			x = X[max_ind]
			final_best_attack_value = torch.max(obj_vals).item()

			t_init = tf_init - t0_opt
			t_total_opt = tf_opt - t0_opt

			debug_dict = {"X_init": X_init, "X_init_reuse": X_reuse_init, "X_init_random": X_random_init, "X_final": X, "X_obj_vals": obj_vals, "init_best_attack_value": init_best_attack_value, "final_best_attack_value": final_best_attack_value, "t_init": t_init, "t_grad_steps": t_grad_step, "t_reproject": t_reproject, "t_total_opt": t_total_opt, "dist_diff_after_proj": dist_diff_after_proj, "n_opt_steps": max_n_steps}
			debug_dict.update(boundary_sample_debug_dict)

			return x, debug_dict
