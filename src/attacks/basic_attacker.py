import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
import torch.optim as optim


class BasicAttacker():
	"""
	The most bare-bones, no frills attack.
	"""
	# TODO: enforce argmax(phi_i) = r constraint (project to this subset of the manifold)
	# Note: this is not batch compliant.

	def __init__(self, x_lim, stopping_condition="steps", max_n_steps=10, stop_threshold=1e-1, lr=1e-3, projection_stop_threshold=1e-3, projection_lr=1e-3):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		assert stopping_condition in ["steps", "stop_threshold"]

		self.x_dim = self.x_lim.shape[0]

	# TODO: for PGD x Adam, should I manually feed the clipped gradient to Adam?
	# Workaround: custom backwards hook for the projection objective function. In this hook, clip gradient before returning
	# Want to use Adam, because we want this to converge quickly
	# How many steps does this take?? If too many, it's going to be an issue.
	def project(self, phi_fn, x):
		# requires_grad_before = x.requires_grad
		# x.requires_grad = True

		# optimizer = optim.Adam([x])

		# TODO: what happens if you can't converge?
		# Until convergence
		loss = float("inf")
		while abs(loss) > self.projection_stop_threshold:
			# IPython.embed()
			# prev_loss = loss

			x_batch = x.view(1, -1)
			x_batch.requires_grad = True
			loss = phi_fn(x_batch)**2
			grad_to_zero_level = grad([loss], x_batch)[0].squeeze()
			x_batch.requires_grad = False

			# IPython.embed()
			# print(grad_to_zero_level)
			x = x - self.projection_lr*grad_to_zero_level
			# print(x)
			# IPython.embed()

			# Clip to bounding box
			# No torch.clamp in torch 1.7.1
			x = torch.minimum(torch.maximum(x, torch.tensor(self.x_lim[:, 0])), torch.tensor(self.x_lim[:, 1]))

			# print(grad_to_zero_level)
			# print(loss)
			# print(abs(prev_loss - loss))

		# x.requires_grad = requires_grad_before
		return x

	def step(self, objective_fn, phi_fn, x):
		# It makes less sense to use an adaptive LR method here, if you think about it
		# requires_grad_before = x.requires_grad

		# IPython.embed()
		x_batch = x.view(1, -1)
		x_batch.requires_grad = True
		obj_val = -objective_fn(x_batch)
		print(obj_val)
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

		# print("This is initial x; is it inside bounds?")
		# print(x, self.x_lim)

		# Project to manifold
		x = self.project(phi_fn, x)

		# print("ln 85")
		# IPython.embed()

		should_stop = False
		# x.requires_grad = True
		i = 0
		prev_objective = float("inf")
		while not should_stop:
			x = self.step(objective_fn, phi_fn, x)

			if self.stopping_condition == "steps": # TODO: in the first run, optimize inner to convergence
				should_stop = (i > self.max_n_steps)
				i += 1
			elif self.stopping_conditon == "stop_threshold":
				# objective_fn.eval() # ?
				objective = objective_fn(x)
				# objective_fn.train() # ?
				should_stop = abs(prev_objective-objective) < self.stop_threshold
				prev_objective = objective

		# IPython.embed()
		return x

# TODO: how should .requires_grad be set here?