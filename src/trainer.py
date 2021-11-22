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
		p_dict = {p[0]:p[1] for p in phi_fn.named_parameters()}
		# proj_params = ["ci", "a"]
		# params_no_proj = [tup[1] for tup in phi_fn.named_parameters() if tup[0] not in proj_params]
		ci = p_dict["ci"]
		a = p_dict["a"]
		beta_net_0_weight = p_dict["beta_net.0.weight"]

		# optimizer = optim.Adam(params_no_proj) # TODO: pending change
		optimizer = optim.Adam(phi_fn.parameters(), lr=self.args.trainer_lr)
		_iter = 0
		t0 = time.perf_counter()

		# Set up saving
		# All these lists should be the same length
		test_attack_losses = []
		test_reg_losses = []
		test_losses = []
		timings = [] # in seconds
		train_attacks = [] # lists of 1D numpy tensors

		# Debug
		train_attack_losses = []
		train_reg_losses = []
		train_losses = []
		train_loss_debug = []
		train_attack_X_init = []
		train_attack_X_final = []
		train_attack_X_obj_vals = []
		ci_grad = []
		a_grad = []
		grad_norms = []
		# ci_lr = 1e-4
		# a_lr = 1e-4

		early_stopping = EarlyStopping(patience=self.args.trainer_early_stopping_patience, min_delta=1e-2)

		# TODO: remove (optional)
		# file_name = os.path.join(self.args.model_folder, f'checkpoint_{_iter}.pth')
		# save_model(phi_fn, file_name)
		while True:
			# Inner min
			X_init, X, x, X_obj_vals = self.attacker.opt(objective_fn, phi_fn, debug=True, mode=self.args.train_mode)

			if self.args.trainer_average_gradients:
				# IPython.embed()
				# Outer max
				optimizer.zero_grad()

				reg_value = reg_fn()
				obj = objective_fn(X)

				c = 0.1
				with torch.no_grad():
					# w = torch.nn.functional.softmax(obj, dim=0)
					w = torch.exp(c*obj)
					w = w/torch.sum(w)
				attack_value = torch.dot(w.flatten(), obj.flatten())
				objective_value = attack_value + reg_value
				objective_value.backward()

				# Logging for debugging # TODO
				beta_net_0_weight_grad = beta_net_0_weight.grad
				grad_norm = torch.norm(beta_net_0_weight_grad)
				grad_norm = grad_norm.detach().cpu().detach()
				grad_norms.append(grad_norm)

				optimizer.step()

				objective_value_after_step = objective_fn(X) + reg_fn()
			else:
				# Outer max
				optimizer.zero_grad()

				x_batch = x.view(1, -1)
				reg_value = reg_fn()
				attack_value = objective_fn(x_batch)[0, 0]
				objective_value = attack_value + reg_value
				objective_value.backward()

				# Logging for debugging # TODO
				beta_net_0_weight_grad = beta_net_0_weight.grad
				grad_norm = torch.norm(beta_net_0_weight_grad)
				grad_norm = grad_norm.detach().cpu().detach()
				grad_norms.append(grad_norm)

				optimizer.step()

				objective_value_after_step = objective_fn(x_batch) + reg_fn()

			print(grad_norms)
			# Clipping for positive parameters
			with torch.no_grad():
				clipped_ci = torch.maximum(ci, torch.zeros_like(ci))  # Project to all positive
				ci.copy_(clipped_ci)  # proper way to update

				clipped_a = torch.maximum(a, torch.zeros_like(a))  # Project to all positive
				a.copy_(clipped_a)

			tnow = time.perf_counter()

			# Output
			self.logger.info('\n' + '=' * 20 + f' evaluation at iteration: {_iter} ' \
			                 + '=' * 20)

			# IPython.embed()
			# objective_value = objective_value.detach().cpu().numpy()
			# reg_value = reg_value.detach().cpu().numpy()
			# attack_value = attack_value.detach().cpu().numpy()
			self.logger.info(f'train loss: {objective_value:.3f}%')
			self.logger.info(f'train attack loss: {attack_value:.3f}%, reg loss: {reg_value:.3f}%')
			t_so_far = str(datetime.timedelta(seconds=(tnow-t0)))
			self.logger.info('time spent training so far: %s' % t_so_far)

			# Gather data
			train_attack_losses.append(attack_value)
			train_reg_losses.append(reg_value)
			train_losses.append(objective_value)

			train_loss_debug.append(objective_value_after_step-objective_value)

			X_init_numpy = X_init.detach().cpu().numpy()
			X_numpy = X.detach().cpu().numpy()
			train_attack_X_init.append(X_init_numpy)
			train_attack_X_final.append(X_numpy)
			train_attack_X_obj_vals.append(X_obj_vals)
			a_grad.append(a.grad)
			ci_grad.append(ci.grad)

			timings.append(t_so_far)
			train_attack_numpy = x.detach().cpu().numpy()
			train_attacks.append(train_attack_numpy)

			# Saving at every _ iterations
			if _iter % self.args.n_checkpoint_step == 0:
				file_name = os.path.join(self.args.model_folder, f'checkpoint_{_iter}.pth')
				save_model(phi_fn, file_name)

				# save data too
				save_dict = {"test_losses": test_losses, "test_attack_losses": test_attack_losses, "test_reg_losses": test_reg_losses, "timings": timings, "train_attacks": train_attacks, "train_loss_debug": train_loss_debug, "train_attack_X_init": train_attack_X_init, "train_attack_X_final": train_attack_X_final, "a_grad":a_grad, "ci_grad":ci_grad, "train_losses":train_losses, "train_attack_losses": train_attack_losses, "train_reg_losses": train_reg_losses, "train_attack_X_obj_vals": train_attack_X_obj_vals, "grad_norms": grad_norms}
				self.logger.info(f'train loss: {objective_value:.3f}%')
				self.logger.info(f'train attack loss: {attack_value:.3f}%, reg loss: {reg_value:.3f}%')

				print("Saving at: ", self.data_save_fpth)
				with open(self.data_save_fpth, 'wb') as handle:
					pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

			if (self.args.n_test_loss_step > 0) and (_iter % self.args.n_test_loss_step == 0):
				# compute test loss
				t1 = time.perf_counter()
				test_attack_loss = self.test(phi_fn, objective_fn)
				test_reg_loss = reg_fn()
				test_loss = test_attack_loss + test_reg_loss
				t2 = time.perf_counter()

				# IPython.embed()
				self.logger.info(f'test loss: {test_loss:.3f}%, time spent testing: {t2 - t1:.3f} s')
				self.logger.info(f'test attack loss: {test_attack_loss:.3f}%, reg loss: {test_reg_loss:.3f}%')

				# Update save data
				test_attack_losses.append(test_attack_loss)
				test_reg_losses.append(test_reg_loss)
				test_losses.append(test_loss)

			# Rest of print output
			self.logger.info('=' * 28 + ' end of evaluation ' + '=' * 28 + '\n')

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
		x = self.test_attacker.opt(objective_fn, phi_fn, mode=self.args.train_mode)
		objective_value = objective_fn(x.view(1, -1))[0, 0]

		return objective_value
