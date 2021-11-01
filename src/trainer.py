import torch

import torch.optim as optim
from src.utils import save_model
import os
import time
import IPython
from torch.autograd import grad
from src.utils import *
import datetime
import pickle

class Trainer():
	def __init__(self, args, logger, attacker, test_attacker):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		self.save_folder = '%s_%s' % (self.args.problem, self.args.affix)
		self.log_folder = os.path.join(self.args.log_root, self.save_folder)
		self.data_save_fpth = os.path.join(self.log_folder, "data.pkl")

	def train(self, objective_fn, reg_fn, phi_fn, xdot_fn):
		# TODO: c_i require projected GD, so you'll need to modify Adam
		p_dict = {p[0]:p[1] for p in phi_fn.named_parameters()}
		params_no_ci = [tup[1] for tup in phi_fn.named_parameters() if tup[0] != "ci"]
		ci = p_dict["ci"]

		optimizer = optim.Adam(params_no_ci)
		_iter = 0
		t0 = time.perf_counter()

		# Set up saving
		# All these lists should be the same length
		test_attack_losses = []
		test_reg_losses = []
		test_losses = []
		timings = [] # in seconds
		train_attacks = [] # lists of 1D numpy tensors
		# test_attacks = []
		ci_lr = 1e-4

		early_stopping = EarlyStopping(patience=self.args.trainer_early_stopping_patience, min_delta=1e-2)
		while True:
			# Inner min
			x = self.attacker.opt(objective_fn, phi_fn)

			# Outer max
			optimizer.zero_grad()
			ci.grad = None

			x_batch = x.view(1, -1)
			# IPython.embed()
			objective_value = objective_fn(x_batch) + reg_fn()
			objective_value.backward()

			if torch.any(torch.isnan(phi_fn.ci.grad)):
				IPython.embed()

			optimizer.step()

			with torch.no_grad():
				new_ci = ci - ci_lr*ci.grad
				new_ci = torch.maximum(new_ci, torch.zeros_like(new_ci)) # Project to all positive
				ci.copy_(new_ci) # proper way to update

			# Testing and logging at every iteration
			t1 = time.perf_counter()
			test_attack_loss = self.test(phi_fn, objective_fn)
			test_reg_loss = reg_fn()
			test_loss = test_attack_loss + test_reg_loss
			t2 = time.perf_counter()

			self.logger.info('\n' + '=' * 20 + f' evaluation at iteration: {_iter} ' \
			            + '=' * 20)
			# IPython.embed()
			self.logger.info(f'test loss: {test_loss:.3f}%, time spent testing: {t2 - t1:.3f} s')
			self.logger.info(f'test attack loss: {test_attack_loss:.3f}%, reg loss: {test_reg_loss:.3f}%')

			# Time spent training so far?
			t_so_far = str(datetime.timedelta(seconds=(t2-t0)))
			self.logger.info('time spent training so far: %s' % t_so_far)
			self.logger.info('=' * 28 + ' end of evaluation ' + '=' * 28 + '\n')

			# Update save data
			test_attack_losses.append(test_attack_loss)
			test_reg_losses.append(test_reg_loss)
			test_losses.append(test_loss)

			timings.append(t_so_far)
			train_attack_numpy = x.detach().cpu().numpy()
			train_attacks.append(train_attack_numpy)

			# Saving at every _ iterations
			if _iter % self.args.n_checkpoint_step == 0:
				file_name = os.path.join(self.args.model_folder, f'checkpoint_{_iter}.pth')
				save_model(phi_fn, file_name)

				# save data too
				save_dict = {"test_losses": test_losses, "test_attack_losses": test_attack_losses, "test_reg_losses": test_reg_losses, "timings": timings, "train_attacks": train_attacks}
				print("Saving at: ", self.data_save_fpth)
				with open(self.data_save_fpth, 'wb') as handle:
					pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

			# Check for stopping
			if self.args.trainer_stopping_condition == "early_stopping":
				early_stopping(test_loss)
				if early_stopping.early_stop:
					break
			elif self.args.trainer_stopping_condition == "n_steps":
				if _iter > self.args.trainer_n_steps:
					break

			_iter += 1

	def test(self, phi_fn, objective_fn):
		# Inner min
		x = self.test_attacker.opt(objective_fn, phi_fn)
		objective_value = objective_fn(x.view(1, -1))[0, 0]

		return objective_value
