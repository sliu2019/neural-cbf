import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad

class BasicAttacker():
	"""
	The most bare-bones, no frills attack.
	"""
	# TODO: enforce argmax(phi_i) = r constraint (project to this subset of the manifold)

	def __init__(self, x_lim, max_n_steps=10, min_change=1e-1, stopping_condition="steps"):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def project(self, X):
		pass

	def step(self):
		pass

	def opt(self, objective_fn, phi_fn):
		# Sample 1 point well within box

		# Project to manifold

		# while (not stop_condition):
		#   step
