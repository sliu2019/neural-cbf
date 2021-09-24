import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
import torch.optim as optim

# TODO: implement early stopping
class BasicAttacker():
	"""
	The most bare-bones, no frills attack.
	"""
	# TODO: enforce argmax(phi_i) = r constraint (project to this subset of the manifold)
	# Note: this is not batch compliant.

	def __init__(self, x_lim, stopping_condition="n_steps", max_n_steps=10, early_stopping=1e-2, lr=1e-3, projection_stop_threshold=1e-3, projection_lr=1e-3):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		assert stopping_condition in ["n_steps", "early_stopping"]

		self.x_dim = self.x_lim.shape[0]

	# TODO: for PGD x Adam, should I manually feed the clipped gradient to Adam?
	# Workaround: custom backwards hook for the projection objective function. In this hook, clip gradient before returning
	# Want to use Adam, because we want this to converge quickly
	# How many steps does this take?? If too many, it's going to be an issue.
	def project(self, phi_fn, x):
		# TODO: what happens if you can't reach the manifold via GD?
		# Until convergence
		loss = float("inf")
		while abs(loss) > self.projection_stop_threshold:
			x_batch = x.view(1, -1)
			x_batch.requires_grad = True
			loss = phi_fn(x_batch)**2
			grad_to_zero_level = grad([loss], x_batch)[0].squeeze()
			x_batch.requires_grad = False

			x = x - self.projection_lr*grad_to_zero_level
			# Clip to bounding box
			# No torch.clamp in torch 1.7.1
			x = torch.minimum(torch.maximum(x, torch.tensor(self.x_lim[:, 0])), torch.tensor(self.x_lim[:, 1]))
		return x

	def step(self, objective_fn, phi_fn, x):
		# It makes less sense to use an adaptive LR method here, if you think about it
		# requires_grad_before = x.requires_grad

		# IPython.embed()
		x_batch = x.view(1, -1)
		x_batch.requires_grad = True
		obj_val = -objective_fn(x_batch)
		# print(obj_val)
		phi_val = phi_fn(x_batch)
		obj_grad = grad([obj_val], x_batch)[0].squeeze()
		normal_to_manifold = grad([phi_val], x_batch)[0].squeeze()
		x_batch.requires_grad = False

		proj_obj_grad = obj_grad - torch.dot(obj_grad, normal_to_manifold)*normal_to_manifold
		# IPython.embed()
		# Take a step
		x = x - self.lr*proj_obj_grad

		# Clip to bounding box
		# Rationale for this step: everytime you take a step on x, clip to box

		# No torch.clamp in torch 1.7.1
		x = torch.minimum(torch.maximum(x, torch.tensor(self.x_lim[:, 0])), torch.tensor(self.x_lim[:, 1]))

		# Project to surface
		x = self.project(phi_fn, x)

		# x.requires_grad = requires_grad_before
		return x

	def opt(self, objective_fn, phi_fn):
		# Sample 1 point well within box
		x = torch.rand(self.x_dim)*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]

		# Project to manifold
		x = self.project(phi_fn, x)

		should_stop = False
		i = 0
		prev_objective = float("inf")
		while not should_stop:
			x = self.step(objective_fn, phi_fn, x)

			if self.stopping_condition == "n_steps":
				should_stop = (i > self.max_n_steps)
			elif self.stopping_condition == "early_stopping":
				objective = objective_fn(x.view(1, -1))[0, 0]
				# TODO: there should be a more sophisticated way to detect convergence
				# TODO: think - Anusha's code
				should_stop = abs(prev_objective-objective) < self.early_stopping
				prev_objective = objective
			i += 1
		return x

