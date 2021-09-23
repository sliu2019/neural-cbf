import torch

import torch.optim as optim
from src.utils import save_model
import os
import time
import IPython
from torch.autograd import grad

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


"""
Things to add to args:
ci_lr
"""

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
		p_dict = {p[0]:p[1] for p in phi_fn.named_parameters()}
		params_no_ci = [tup[1] for tup in phi_fn.named_parameters() if tup[0] != "ci"]
		ci = p_dict["ci"]
		# IPython.embed()

		# print("check params_no_ci")

		# TODO
		optimizer = optim.Adam(params_no_ci)
		_iter = 0
		prev_test_loss = float("inf")
		test_loss = 999
		# while abs(prev_test_loss-test_loss) <= self.args.stop_threshold: # TODO: put this back
		while _iter < 3:
			print("inside loop")

			# Inner min
			x = self.attacker.opt(objective_fn, phi_fn)

			# Outer max
			optimizer.zero_grad()
			# ci.requires_grad = True
			ci.grad = None

			x_batch = x.view(1, -1)
			objective_value = objective_fn(x_batch)
			objective_value.backward()

			# print("ci grad: ", ci.grad)
			optimizer.step()

			print(_iter)
			# if _iter == 1:
			# 	print("HERE!!!")
			# 	IPython.embed()
			# 	ci_grad = grad([objective_value], ci)
			print(ci.requires_grad)
			ci_grad = ci.grad
			print(ci_grad)
			if _iter==1:
				IPython.embed()
			with torch.no_grad():
				# Weird that it's even required
				print("ci efore: ", ci)
				new_ci = ci + (1e-3)*ci_grad
				new_ci = torch.maximum(new_ci, torch.zeros_like(new_ci)) # Project
				ci.copy_(new_ci)
				# ci = ci + (1e-3)*ci_grad # hardcoded LR
				print("ci after: ", ci)
				# ci = torch.clamp(ci, min=torch.zeros_like(ci)) # Project
				# No torch.clamp in torch 1.7.1
			print(ci.requires_grad)
			# Testing and logging at every iteration
			prev_test_loss = test_loss

			# IPython.embed()

			# print("TESTING!!!!!!")
			#TODO: bring this back
			"""t1 = time.perf_counter()
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

			"""
			_iter += 1

	def test(self, phi_fn, objective_fn):
		# Inner min
		x = self.test_attacker.opt(objective_fn, phi_fn)
		objective_value = objective_fn(x.view(1, -1))[0, 0]

		return objective_value
