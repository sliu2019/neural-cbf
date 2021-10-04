import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *

# TODO: make sure GPU is engaged for this.
class GradientBatchAttacker():
	"""
	Gradient-based attack, but parallelized across many initializations
	"""
	# TODO: enforce argmax(phi_i) = r constraint (project to this subset of the manifold)
	# Note: this is not batch compliant.

	def __init__(self, x_lim, device, n_samples=20, \
	             stopping_condition="n_steps", max_n_steps=10, early_stopping_min_delta=1e-2, early_stopping_patience=3,\
	             lr=1e-3, \
	             projection_stop_threshold=1e-3, projection_lr=1e-3, projection_time_limit=60):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		assert stopping_condition in ["n_steps", "early_stopping"]

		self.x_dim = self.x_lim.shape[0]

	def project(self, phi_fn, x):
		# Until convergence
		i = 0
		t1 = time.perf_counter()

		print(self.projection_stop_threshold)
		while True:
			print(i)
			x_batch = x.view(-1, self.x_dim)
			x_batch.requires_grad = True
			loss = (phi_fn(x_batch)[:, -1])**2
			grad_to_zero_level = grad([torch.sum(loss)], x_batch)[0]
			x_batch.requires_grad = False

			# x = x - self.projection_lr*grad_to_zero_level
			x = x - grad_to_zero_level
			# Clip to bounding box
			# No torch.clamp in torch 1.7.1
			x = torch.minimum(torch.maximum(x, self.x_lim[:, 0]), self.x_lim[:, 1])

			i += 1

			print(torch.min(loss), torch.mean(loss), torch.max(loss))
			if torch.max(loss) < self.projection_stop_threshold:
				break
			elif (time.perf_counter() - t1) > self.projection_time_limit and (torch.min(loss) < self.projection_stop_threshold):
				# In case projection fails or takes too long on some samples
				# Keeps the same size X
				# Replace unprojected x with a projected x
				projected_x = x[torch.argmin(loss)]
				mask = loss > self.projection_stop_threshold
				mask = mask.type(torch.float32)
				x = (1-mask)*x + mask.mm(projected_x.unsqueeze(0))
				break

		print("%i steps for projection" % i)
		t2 = time.perf_counter()
		print("Done projecting in %f seconds" % (t2-t1))
		return x

	def step(self, objective_fn, phi_fn, x):
		# It makes less sense to use an adaptive LR method here, if you think about it
		# requires_grad_before = x.requires_grad

		# print("in step")
		# IPython.embed()
		t0 = time.perf_counter()
		x_batch = x.view(-1, self.x_dim)
		x_batch.requires_grad = True
		obj_val = -objective_fn(x_batch)
		phi_val = phi_fn(x_batch)
		obj_grad = grad([torch.sum(obj_val)], x_batch)[0]
		normal_to_manifold = grad([torch.sum(phi_val[:, -1])], x_batch)[0]
		x_batch.requires_grad = False

		weights = obj_grad.unsqueeze(1).bmm(normal_to_manifold.unsqueeze(2))[:, 0]
		proj_obj_grad = obj_grad - weights*normal_to_manifold
		# IPython.embed()
		# Take a step
		x = x - self.lr*proj_obj_grad

		t1 = time.perf_counter()
		# Clip to bounding box
		# Rationale for this step: everytime you take a step on x, clip to box

		# No torch.clamp in torch 1.7.1
		x = torch.minimum(torch.maximum(x, self.x_lim[:, 0]), self.x_lim[:, 1])

		# Project to surface
		x = self.project(phi_fn, x)

		t2 = time.perf_counter()

		print("Step: %f s" % (t1-t0))
		print("Project: %f s" % (t2 -t1))
		# x.requires_grad = requires_grad_before
		return x

	def opt(self, objective_fn, phi_fn):
		# Sample 1 point well within box
		random = torch.rand(self.n_samples, self.x_dim).to(self.device)
		X = random*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
		# X = X.to(self.device)
		# print(X)

		# Project to manifold
		t0 = time.perf_counter()
		X = self.project(phi_fn, X)
		t1 = time.perf_counter()
		# print(X)
		# IPython.embed()
		print("Time for initial projection: %f" % (t1-t0))

		# should_stop = False
		i = 0
		early_stopping = EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta)
		while True:
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
		print(torch.max(obj_vals))
		x = X[max_ind]
		return x


# TODO: there should be a more sophisticated way to detect convergence
# TODO: think - Anusha's code

# TODO: for PGD x Adam, should I manually feed the clipped gradient to Adam?
# Workaround: custom backwards hook for the projection objective function. In this hook, clip gradient before returning
# Want to use Adam, because we want this to converge quickly
# How many steps does this take?? If too many, it's going to be an issue.