import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *

class GradientBatchAttacker():
	"""
	Gradient-based attack, but parallelized across many initializations
	"""
	# TODO: enforce argmax(phi_i) = r constraint (project to this subset of the manifold)
	# Note: this is not batch compliant.

	def __init__(self, x_lim, device, logger, n_samples=20, \
	             stopping_condition="n_steps", max_n_steps=10, early_stopping_min_delta=1e-2, early_stopping_patience=3,\
	             lr=1e-3, \
	             projection_stop_threshold=1e-3, projection_lr=1e-3, projection_time_limit=3, verbose=False):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		assert stopping_condition in ["n_steps", "early_stopping"]

		self.x_dim = self.x_lim.shape[0]

		# print("Initializing GradientBatchAttacker")
		# IPython.embed()
		# Compute 2n facets volume of n-dim hypercube (actually n facets because they come in pairs)
		x_lim_interval_sizes = self.x_lim[:, 1] - self.x_lim[:, 0]
		x_lim_interval_sizes = x_lim_interval_sizes.view(1, -1)
		tiled = x_lim_interval_sizes.repeat(self.x_dim, 1)
		# print(tiled.shape)
		tiled = tiled - torch.eye(self.x_dim).to(self.device)*x_lim_interval_sizes + torch.eye(self.x_dim).to(device)
		# print(tiled)
		vols = torch.prod(tiled, axis=1)
		vols = vols/torch.sum(vols)
		self.vols = vols.detach().cpu().numpy() # numpy
		self.hypercube_vol = torch.prod(x_lim_interval_sizes) # tensor const

	def project(self, phi_fn, x):
		# Until convergence
		i = 0
		t1 = time.perf_counter()

		# issue = False
		# if x[:, 1] == -5.0:
		# 	# issue = True # TODO
		# 	issue = False
		while True:
			# print(i)
			x_batch = x.view(-1, self.x_dim)
			x_batch.requires_grad = True
			loss = torch.abs(phi_fn(x_batch)[:, -1])
			grad_to_zero_level = grad([torch.sum(loss)], x_batch)[0]
			x_batch.requires_grad = False

			# IPython.embed()
			x = x - self.projection_lr*grad_to_zero_level
			# x = x - grad_to_zero_level
			# Clip to bounding box; no torch.clamp in torch 1.7.1
			# TODO: REMOVE THIS FOR PROBLEMS THAT ARE NOT REDUCED CARTPOLE!
			# Mod on angle before clipping (clipping angle will be redundant)
			# print(x[:, 0])
			x[:, 0] = torch.atan2(torch.sin(x[:, 0]), torch.cos(x[:, 0]))
			# print(x[:, 0])
			x = torch.minimum(torch.maximum(x, self.x_lim[:, 0]), self.x_lim[:, 1])
			# print(x[:, 0])

			i += 1

			# if issue:
			# 	# print(grad_to_zero_level)
			# 	print(x, grad_to_zero_level)
			# print(loss)
			# print(torch.max(loss), self.projection_stop_threshold)
			if torch.max(loss) < self.projection_stop_threshold:
				if self.verbose:
					print("reprojection was successful")
				break
			elif (time.perf_counter() - t1) > self.projection_time_limit and (torch.min(loss) <= self.projection_stop_threshold):
				# In case projection fails or takes too long on some samples
				# Keeps the same size X
				# Replace unprojected x with a projected x
				projected_x = x[torch.argmin(loss)]
				mask = loss > self.projection_stop_threshold
				mask = mask.type(torch.float32).view(x.shape[0], 1)
				inv_mask = (1-mask)
				print("inv_mask")
				print(inv_mask)
				x = inv_mask*x + mask.mm(projected_x.unsqueeze(0))
				break
			elif (time.perf_counter() - t1) > self.projection_time_limit and (torch.min(loss) > self.projection_stop_threshold):
				print("projected: ", x)
				print("loss: ", loss)
				print("min loss: ", torch.min(loss))
				print(self.projection_stop_threshold)
				IPython.embed() # TODO: for an actual run
				raise ValueError('Timed out because of non-convergence of projection')
			# print(torch.min(loss))

		# print("projected: ", x)
		# print("%i steps for projection" % i)
		t2 = time.perf_counter()
		# print("Done projecting in %f seconds" % (t2-t1))
		return x

	def step(self, objective_fn, phi_fn, x):
		# It makes less sense to use an adaptive LR method here, if you think about it
		t0 = time.perf_counter()
		x_batch = x.view(-1, self.x_dim)
		x_batch.requires_grad = True

		obj_val = -objective_fn(x_batch)
		obj_grad = grad([torch.sum(obj_val)], x_batch)[0]

		phi_val = phi_fn(x_batch)
		normal_to_manifold = grad([torch.sum(phi_val[:, -1])], x_batch)[0]
		# IPython.embed()
		normal_to_manifold = normal_to_manifold/torch.norm(normal_to_manifold, dim=1)[:, None] # normalize

		x_batch.requires_grad = False

		weights = obj_grad.unsqueeze(1).bmm(normal_to_manifold.unsqueeze(2))[:, 0]
		proj_obj_grad = obj_grad - weights*normal_to_manifold

		# IPython.embed()
		if self.verbose:
			print("unprojected grad:", obj_grad)
			print("projected grad: ", proj_obj_grad)

		# Take a step
		x = x - self.lr*proj_obj_grad

		t1 = time.perf_counter()
		# Clip to bounding box
		# Rationale for this step: everytime you take a step on x, clip to box

		# No torch.clamp in torch 1.7.1
		# TODO: REMOVE THIS FOR PROBLEMS THAT ARE NOT REDUCED CARTPOLE!
		# Mod on angle before clipping (clipping angle will be redundant)
		# print(x[:, 0])
		x[:, 0] = torch.atan2(torch.sin(x[:, 0]), torch.cos(x[:, 0]))
		# print(x[:, 0])
		x = torch.minimum(torch.maximum(x, self.x_lim[:, 0]), self.x_lim[:, 1])
		# print(x[:, 0])

		if self.verbose:
			print("After step:", x)
		# Project to surface
		x = self.project(phi_fn, x)
		if self.verbose:
			print("After reprojection", x)

		t2 = time.perf_counter()

		# print("Step: %f s" % (t1-t0))
		# print("Project: %f s" % (t2 -t1))
		return x

	def sample_in_cube(self):
		"""
		Samples uniformly in state space hypercube
		Returns 1 sample
		"""
		# samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
		unif = torch.rand(self.x_dim).to(self.device)
		sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
		return sample

	def sample_on_cube(self):
		"""
		Samples uniformly on state space hypercube
		Returns 1 sample
		"""
		# https://math.stackexchange.com/questions/2687807/uniquely-identify-hypercube-faces
		which_facet_pair = np.random.choice(np.arange(self.x_dim), p=self.vols)
		which_facet = np.random.choice([0, 1])

		# samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
		unif = torch.rand(self.x_dim).to(self.device)
		sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
		sample[which_facet_pair] = self.x_lim[which_facet_pair, which_facet]
		return sample

	def intersect_segment_with_manifold(self, p1, p2, phi_fn, rtol=1e-5, atol=1e-3):
		"""
		Atol? Reltol?
		"""
		# TODO: is the stopping condition correct?
		# self.logger.info("Intersecting segment with manifold")
		# print("Inside intersect_segment_withmanifold")
		# IPython.embed()
		diff = p2-p1

		left_weight = 0.0
		right_weight = 1.0
		left_val = phi_fn(p1.view(1, -1))[0, -1]
		right_val = phi_fn(p2.view(1, -1))[0, -1]
		left_sign = torch.sign(left_val)
		right_sign = torch.sign(right_val)

		if left_sign*right_sign > 0:
			return None

		# print("This segment intersects the surface")
		# IPython.embed()
		t0 = time.perf_counter()
		while True:
			mid_weight = (left_weight + right_weight)/2.0
			mid_point = p1 + mid_weight*diff

			mid_val = phi_fn(mid_point.view(1, -1))[0, -1]
			mid_sign = torch.sign(mid_val)
			if mid_sign*left_sign < 0:
				# go to the left side
				right_weight = mid_weight
				right_val = mid_val
			elif mid_sign*right_sign < 0:
				left_weight = mid_weight
				left_val = mid_val

			# Use this approach or the one below to prevent infinite loops
			# Approach #1
			if np.abs(left_weight - right_weight) < 1e-3:
				intersection_point = p1 + left_weight*diff
				break
			t1 = time.perf_counter()
			if (t1-t0)>100:
				# This clause is necessary for non-differentiable, continuous points (abrupt change)
				print("Something is wrong in projection")
				print(torch.abs(left_val - right_val))
				print(left_weight, right_weight)
				left_point = p1 + left_weight * diff
				right_point = p1 + right_weight * diff
				print(left_point, right_point)
				# IPython.embed()
				return None

			"""# Approach #2: CGAL uses this
			left_point = p1 + left_weight*diff
			right_point = p1 + right_weight*diff

			norm_diff = torch.norm(left_point-right_point)
			print(norm_diff, rtol*self.hypercube_vol)
			if norm_diff < rtol*self.hypercube_vol:
				intersection_point = p1 + left_weight*diff
				break
			"""

			"""
			# Approach #3
			t1 = time.perf_counter()
			if (t1-t0)>100:
				# This clause is necessary for non-differentiable, continuous points (abrupt change)
				print("Something is wrong in projection")
				print(torch.abs(left_val - right_val))
				print(left_weight, right_weight)
				left_point = p1 + left_weight * diff
				right_point = p1 + right_weight * diff
				print(left_point, right_point)
				return None
			if torch.abs(left_val - right_val) <= atol:
				intersection_point = p1 + left_weight*diff # arbitrary choice of left point
				break
			"""

			# IPython.embed()
		# self.logger.info("done")
		return intersection_point

	def sample_points_on_boundary(self, phi_fn, mode="dG"):
		"""
		Returns torch array of size (self.n_samples, self.x_dim)
		Mode between "dG", "dG+dS", "dG/dS"
		"""
		# Everything done in torch
		# self.logger.info("sampling points on the boundary")
		samples = []
		n_remaining_to_sample = self.n_samples

		center = self.sample_in_cube()
		n_segments_sampled = 0

		while n_remaining_to_sample > 0:
			# print(n_remaining_to_sample)
			outer = self.sample_on_cube()

			intersection = self.intersect_segment_with_manifold(center, outer, phi_fn)
			if intersection is not None:
				if mode == "dG":
					samples.append(intersection.view(1, -1))
					n_remaining_to_sample -= 1
				else:
					phi_val = phi_fn(intersection.view(1, -1))
					on_dS = torch.all(phi_val[0, :-1] <= 1e-6).item()
					# if mode == "dG/dS":
					# 	print(intersection, phi_val)
					# 	print(on_dS)
					if on_dS and mode=="dG+dS":
						samples.append(intersection.view(1, -1))
						n_remaining_to_sample -= 1
					elif (not on_dS) and mode=="dG/dS":
						samples.append(intersection.view(1, -1))
						n_remaining_to_sample -= 1
			else:
				center = self.sample_in_cube()
			n_segments_sampled += 1
			# self.logger.info("%i segments" % n_segments_sampled)

		samples = torch.cat(samples, dim=0)
		# self.logger.info("Done with sampling points on the boundary...")
		return samples

	def opt(self, objective_fn, phi_fn, test=False):

		X_init = self.sample_points_on_boundary(phi_fn)
		X = X_init.clone()
		i = 0
		early_stopping = EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta)
		while True:
			# print(i)
			X = self.step(objective_fn, phi_fn, X)

			if self.stopping_condition == "n_steps":
				if (i > self.max_n_steps):
					break
			elif self.stopping_condition == "early_stopping":
				obj_vals = objective_fn(X.view(-1, self.x_dim))
				obj_val = torch.max(obj_vals)

				early_stopping(obj_val)
				if early_stopping.early_stop:
					break
			i += 1

		# Returning a single attack
		obj_vals = objective_fn(X)
		max_ind = torch.argmax(obj_vals)

		if not test:
			x = X[max_ind]
			return x
		else: # TODO: this is terrible coding practice!!!??
			x = X[max_ind]
			return X_init, X, x, obj_vals

# TODO: for PGD x Adam, should I manually feed the clipped gradient to Adam?
# Workaround: custom backwards hook for the projection objective function. In this hook, clip gradient before returning
# Want to use Adam, because we want this to converge quickly
# How many steps does this take?? If too many, it's going to be an issue.