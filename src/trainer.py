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
	def __init__(self, args, logger, attacker, test_attacker, reg_sample_keeper):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		self.save_folder = '%s_%s' % (self.args.problem, self.args.affix)
		self.log_folder = os.path.join(self.args.log_root, self.save_folder)
		self.data_save_fpth = os.path.join(self.log_folder, "data.pkl")

	def train(self, objective_fn, reg_fn, phi_fn, xdot_fn):
		p_dict = {p[0]:p[1] for p in phi_fn.named_parameters()}
		proj_params = ["ci", "k0"]
		params_no_proj = [tup[1] for tup in phi_fn.named_parameters() if tup[0] not in proj_params]
		ci = p_dict["ci"]
		k0 = p_dict["k0"]
		beta_net_0_weight = p_dict["beta_net.0.weight"]

		# if self.args.trainer_type == "Adam":
		# 	# TODO: need to fix this for projected parameters
		# 	# optimizer = optim.Adam(params_no_proj) # old
		# 	optimizer = optim.Adam(phi_fn.parameters(), lr=self.args.trainer_lr)
		# elif self.args.trainer_type == "LinearLR":
		# 	optimizer = optim.SGD(phi_fn.parameters(), self.args.trainer_lr)
		# 	scheduler = optim.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000, verbose=True)
		# 	# TODO: total_iters set experimentally
		# elif self.args.trainer_type == "ExponentialLR":
		# 	optimizer = optim.SGD(phi_fn.parameters(), self.args.trainer_lr)
		# 	gamma = 0.997
		# 	scheduler = optim.ExponentialLR(optimizer, gamma, verbose=True)

		optimizer = optim.Adam(phi_fn.parameters(), lr=self.args.trainer_lr)

		_iter = 0
		t0 = time.perf_counter()

		# Set up saving
		# All these lists should be the same length
		test_attack_losses = []
		test_reg_losses = []
		test_losses = []

		# Debug losses and overall timing
		train_attack_losses = []
		train_reg_losses = []
		train_losses = []
		train_loop_times = [] # in seconds

		# Debug attacks
		train_attacks = [] # lists of 1D numpy tensors
		train_attack_X_init = []
		train_attack_X_init_reuse = []
		train_attack_X_init_random = []
		train_attack_X_final = []
		train_attack_X_obj_vals = []
		train_attack_X_phi_vals = []

		train_attack_init_best_attack_value = []
		train_attack_final_best_attack_value = []

		# Debug attack timing
		train_attack_t_init = []
		train_attack_t_grad_steps = []
		train_attack_t_reproject = []
		train_attack_t_total_opt = []

		# Debug PGD on params + grad norms
		ci_grad = []
		k0_grad = []
		grad_norms = []

		# Debug reg sample keeper
		reg_sample_keeper_X = []
		max_dists_X_reg = []
		times_to_compute_X_reg = []

		ci_lr = 1e-4
		a_lr = 1e-4

		early_stopping = EarlyStopping(patience=self.args.trainer_early_stopping_patience, min_delta=1e-2)

		# TODO: remove (optional)
		# file_name = os.path.join(self.args.model_folder, f'checkpoint_{_iter}.pth')
		# save_model(phi_fn, file_name)
		while True:
			# Inner max
			# X_init, X, x, X_obj_vals = self.attacker.opt(objective_fn, phi_fn, debug=True, mode=self.args.train_mode)
			x, debug_dict = self.attacker.opt(objective_fn, phi_fn, debug=True)
			X = debug_dict["X"]

			optimizer.zero_grad()
			# ci.grad = None
			# k0.grad = None

			t0_xreg = time.perf_counter()
			if reg_fn.A_samples:
				doesnt_matter = torch.tensor(0)
				reg_value = reg_fn(doesnt_matter) # reg_fn depends on property A_samples, not the input
			elif reg_fn.A_samples is None and self.args.reg_weight:
				X_reg = self.reg_sample_keeper.return_samples(phi_fn)
				reg_value = reg_fn(X_reg)
			else:
				reg_value = torch.tensor(0)
			tf_xreg = time.perf_counter()

			if self.args.trainer_average_gradients:
				# Outer max
				obj = objective_fn(X)
				c = 0.1
				with torch.no_grad():
					w = torch.exp(c*obj)
					w = w/torch.sum(w)
				attack_value = torch.dot(w.flatten(), obj.flatten())
			else:
				# Outer max
				x_batch = x.view(1, -1)
				attack_value = objective_fn(x_batch)[0, 0]

			# print("Reg value: ", reg_value)
			objective_value = attack_value + reg_value
			objective_value.backward()
			optimizer.step()

			with torch.no_grad():
				# new_ci = ci - ci_lr*ci.grad
				new_ci = ci
				new_ci = torch.maximum(new_ci, torch.zeros_like(new_ci)) # Project to all positive
				ci.copy_(new_ci) # proper way to update

				# new_k0 = k0 - k0_lr*k0.grad
				new_k0 = k0
				new_k0 = torch.maximum(new_k0, torch.zeros_like(new_k0)) # Project to all positive
				k0.copy_(new_k0) # proper way to update

			tnow = time.perf_counter()

			# Logging for debugging
			# Make sure you detach before logging, otherwise you will accumulate memory over iterations and get an OOM
			self.logger.info('\n' + '=' * 20 + f' evaluation at iteration: {_iter} ' \
			                 + '=' * 20)

			self.logger.info(f'train loss: {objective_value:.3f}%')
			self.logger.info(f'train attack loss: {attack_value:.3f}%, reg loss: {reg_value:.3f}%')
			t_so_far = tnow-t0
			t_so_far_str = str(datetime.timedelta(seconds=t_so_far))
			self.logger.info('time spent training so far: %s' % t_so_far_str)

			self.logger.info('OOM debug. Mem allocated and reserved: %f, %f' % (torch.cuda.memory_allocated(self.args.gpu), torch.cuda.memory_reserved(self.args.gpu)))

			# IPython.embed()
			# Gather data
			train_attack_losses.append(attack_value.item()) # Have to use .item() to detach from graph, otherwise we retain each graph from each iteration
			train_reg_losses.append(reg_value.item())
			train_losses.append(objective_value.item())

			# Debug attacks
			X_init = debug_dict["X_init"]
			X_reuse_init = debug_dict["X_reuse_init"]
			X_random_init = debug_dict["X_random_init"]
			obj_vals = debug_dict["obj_vals"]
			init_best_attack_value = debug_dict["init_best_attack_value"]
			final_best_attack_value = debug_dict["final_best_attack_value"]
			X_phi_vals = debug_dict["phi_vals"]

			train_attack_X_init.append(X_init.detach().cpu().numpy())
			train_attack_X_final.append(X.detach().cpu().numpy())
			train_attack_X_obj_vals.append(obj_vals.detach().cpu().numpy())
			train_attack_X_phi_vals.append(X_phi_vals.detach().cpu().numpy())
			train_attack_numpy = x.detach().cpu().numpy()
			train_attacks.append(train_attack_numpy)
			train_attack_X_init_reuse.append(X_reuse_init.detach().cpu().numpy())
			train_attack_X_init_random.append(X_random_init.detach().cpu().numpy())
			train_attack_init_best_attack_value.append(init_best_attack_value)
			train_attack_final_best_attack_value.append(final_best_attack_value)

			self.logger.info(f'train attack loss increase: {final_best_attack_value-init_best_attack_value:.3f}')

			# Debug attack timing
			t_init = debug_dict["t_init"]
			t_grad_step = debug_dict["t_grad_step"]
			t_reproject = debug_dict["t_reproject"]
			t_total_opt = debug_dict["t_total_opt"]

			train_attack_t_init.append(t_init)
			train_attack_t_grad_steps.append(t_grad_step)
			train_attack_t_reproject.append(t_reproject)
			train_attack_t_total_opt.append(t_total_opt)
			self.logger.info(f'train attack init time: {t_init:.3f}s')
			self.logger.info(f'train attack avg grad step time: {np.mean(t_grad_step):.3f}s')
			self.logger.info(f'train attack avg reroj time: {np.mean(t_reproject):.3f}s')
			self.logger.info(f'train attack total time: {t_total_opt:.3f}s')

			# Gradient debug
			# IPython.embed()
			k0_grad.append(k0.grad.detach().cpu().numpy())
			ci_grad.append(ci.grad.detach().cpu().numpy())

			# Timing debug
			train_loop_times.append(t_so_far)

			# Recording for reg sample keeper
			if self.args.reg_weight and reg_fn.A_samples is None:
				phis_X_reg = phi_fn(X_reg)
				max_dist_X_reg = torch.max(torch.abs(torch.max(phis_X_reg, axis=1)[0])).item()
				# max_dist_X_reg = max_dist_X_reg.detach().cpu().numpy()
				max_dists_X_reg.append(max_dist_X_reg)
				reg_sample_keeper_X.append(X_reg.detach().cpu().numpy())
				times_to_compute_X_reg.append(tf_xreg-t0_xreg)
				self.logger.info(f'reg, total time: {(tf_xreg-t0_xreg):.3f}s')
				self.logger.info(f'reg, max dist: {max_dist_X_reg:.3f}')

			# Saving at every _ iterations
			if _iter % self.args.n_checkpoint_step == 0:
				file_name = os.path.join(self.args.model_folder, f'checkpoint_{_iter}.pth')
				save_model(phi_fn, file_name)

				# save data too
				save_dict = {"test_losses": test_losses, "test_attack_losses": test_attack_losses, "test_reg_losses": test_reg_losses, "train_loop_times": train_loop_times, "train_attacks": train_attacks, "train_attack_X_init": train_attack_X_init, "train_attack_X_final": train_attack_X_final, "k0_grad":k0_grad, "ci_grad":ci_grad, "train_losses":train_losses, "train_attack_losses": train_attack_losses, "train_reg_losses": train_reg_losses, "train_attack_X_obj_vals": train_attack_X_obj_vals, "train_attack_X_phi_vals": train_attack_X_phi_vals, "grad_norms": grad_norms, "reg_sample_keeper_X": reg_sample_keeper_X}

				additional_train_attack_dict = {"train_attack_X_init_reuse": train_attack_X_init_reuse, "train_attack_X_init_random": train_attack_X_init_random, "train_attack_init_best_attack_value": train_attack_init_best_attack_value, "train_attack_final_best_attack_value": train_attack_final_best_attack_value,"train_attack_t_init": train_attack_t_init, "train_attack_t_grad_steps": train_attack_t_grad_steps, "train_attack_t_reproject": train_attack_t_reproject, "train_attack_t_total_opt": train_attack_t_total_opt}

				reg_debug_dict = {"max_dists_X_reg": max_dists_X_reg, "times_to_compute_X_reg": times_to_compute_X_reg}

				# save_dict = save_dict + additional_train_attack_dict
				save_dict.update(additional_train_attack_dict)
				save_dict.update(reg_debug_dict)

				# IPython.embed()

				self.logger.info(f'train loss: {objective_value:.3f}%')
				self.logger.info(f'train attack loss: {attack_value:.3f}%, reg loss: {reg_value:.3f}%')

				print("Saving at: ", self.data_save_fpth)
				with open(self.data_save_fpth, 'wb') as handle:
					pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

			# TODO: test loss is not being computed
			# if (self.args.n_test_loss_step > 0) and (_iter % self.args.n_test_loss_step == 0):
			# 	# compute test loss
			# 	t1 = time.perf_counter()
			# 	test_attack_loss = self.test(phi_fn, objective_fn)
			# 	test_reg_loss = reg_fn(X_reg) # TODO: this won't work if self.args.reg_weight = 0
			# 	test_loss = test_attack_loss + test_reg_loss
			# 	t2 = time.perf_counter()
			#
			# 	# IPython.embed()
			# 	self.logger.info(f'test loss: {test_loss:.3f}%, time spent testing: {t2 - t1:.3f} s')
			# 	self.logger.info(f'test attack loss: {test_attack_loss:.3f}%, reg loss: {test_reg_loss:.3f}%')
			#
			# 	# Update save data
			# 	test_attack_losses.append(test_attack_loss.item())
			# 	test_reg_losses.append(test_reg_loss.item())
			# 	test_losses.append(test_loss.item())

			# Rest of print output
			self.logger.info('=' * 28 + ' end of evaluation ' + '=' * 28 + '\n')

			# Check for stopping
			if self.args.trainer_stopping_condition == "early_stopping":
				# early_stopping(test_loss) # TODO: should technically use test_loss, but we're not computing it
				early_stopping(objective_value)
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
