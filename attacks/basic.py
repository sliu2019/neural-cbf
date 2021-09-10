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
	# *Batch the implementations later

	def __init__(self, x_lim, max_n_steps=10, stop_threshold=1e-1, projection_stop_threshold=1e-3, stopping_condition="steps", lr=1e-3):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		self.x_dim = self.x_lim.size()[0]

	# TODO: for PGD x Adam, should I manually feed the clipped gradient to Adam?
	# Workaround: custom backwards hook for the projection objective function. In this hook, clip gradient before returning
	# Want to use Adam, because we want this to converge quickly
	def project(self, phi_fn, x): # How many steps does this take?? If too many, it's going to be an issue.
		x = x.clone()
		x.requires_grad = True

		optimizer = optim.Adam([x])

		# Until convergence
		prev_loss = float("inf")
		loss = 0
		while abs(prev_loss - loss) > self.projection_stop_threshold:
			optimizer.zero_grad()
			loss = phi_fn(x)**2
			loss.backward()
			optimizer.step()

			# Clip to bounding box
			x = torch.clamp(x, min=self.x_lim[:, 0], max=self.x_lim[:, 1])
		return x

	def step(self, objective_fn, phi_fn, x): # It makes less sense to use an adaptive LR method here, if you think about it
		# Compute gradient, project to tangent plane
		loss = -objective_fn(x)
		loss.backward()
		grad = x.grad
		x.grad = None

		phi_fn(x).backward()
		normal_vec = x.grad
		x.grad = None

		grad = grad - torch.dot(grad, normal_vec)*normal_vec
		x = x + self.lr*grad

		# Clip to bounding box
		# Rationale for this step: everytime you take a step on x, clip to box
		x = torch.clamp(x, min=self.x_lim[:, 0], max=self.x_lim[:, 1])

		# Project to surface
		x = self.project(phi_fn, x)
		return x

	def opt(self, objective_fn, phi_fn):
		projection_loss_fn = phi_fn**2

		# Sample 1 point well within box
		x = torch.rand(self.x_dim)*(self.x_lim[:, 1] - self.x_lim[:, 0]) + (self.x_lim[:, 1] + self.x_lim[:, 0])/2.0

		# Project to manifold
		x = self.project(phi_fn, x)

		# while (not stop_condition):
		x.requires_grad = True
		optimizer = optim.Adam(x) # Keep history between steps
		i = 0
		prev_objective = float("inf")
		while not stopping_condition:
			x = self.step(objective_fn, phi_fn, optimizer, x)

			if self.stopping_condition == "steps":
				stopping_condition = (i > self.max_n_steps)
				i +=1
			elif self.stopping_conditon == "stop_threshold":
				objective = objective_fn(x)
				stopping_condition = abs(prev_objective-objective) < self.stop_threshold
				prev_objective = objective

		return x
