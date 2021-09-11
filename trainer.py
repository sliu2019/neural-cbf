import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
from attacks.basic import BasicAttacker
import torch.optim as optim
from utils import save_model
import os, sys
import time

"""class Trainer:
	init():
		In:
			Attack class object
			(Later) optimization parameters

	train():
		In:
			phi_i (model)
			x_dot
			X_limit, objective function (U_limit implicitly)
		Fn:
			<Anything meta-training, mostly boilerplate:
				1. Training loop: bi-step (inner and outer)
				2. Saving model, computing and logging progress (call test)

	test():
		Either GD until (near) conv or MCMC"""


class Trainer():
	def __init__(self, args, logger, attacker, test_attacker):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def train(self, objective_fn, phi_fn, xdot_fn):
		# logging + saving model
		# for (stopping condition) (# iterations or convergence):
		#   call self.attacker
		#   optimize phi parameters

		# TODO: c_i require projected GD, so you'll need to modify Adam
		params_no_ci = [tup[1] for tup in phi_fn.named_parameters() if tup[0] != "ci_tensor"]
		ci = phi_fn.get_parameter("ci_tensor")

		optimizer = optim.Adam(params_no_ci)
		_iter = 0
		while True:
			# Inner min
			x = self.attacker.opt(self, objective_fn, phi_fn)

			# Outer max
			optimizer.zero_grad()
			ci.grad = None

			objective_value = objective_fn(x)
			objective_value.backwards()

			optimizer.step()

			ci = ci + (1e-3)*ci.grad # hardcoded LR
			ci = torch.clamp(ci, min=torch.zeros_like(ci)) # Project

			# Testing and logging at every iteration
			t1 = time.perf_counter()
			test_loss = self.test(phi_fn, objective_fn)
			t2 = time.perf_counter()

			self.logger.info('\n' + '=' * 20 + f' evaluation at iteration: {_iter} ' \
			            + '=' * 20)
			self.logger.info(f'test loss: {test_loss:.3f}%, spent: {t2 - t1:.3f} s')
			self.logger.info('=' * 28 + ' end of evaluation ' + '=' * 28 + '\n')

			# Saving at every _ iterations
			if _iter % self.args.n_checkpoint_step == 0:
				file_name = os.path.join(self.args.model_folder, f'checkpoint_{_iter}.pth')
				save_model(phi_fn, file_name)

			_iter += 1

	def test(self, phi_fn, objective_fn):
		# Inner min
		x = self.test_attacker.opt(self, objective_fn, phi_fn)
		objective_value = objective_fn(x)

		return objective_value
