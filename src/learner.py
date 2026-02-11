"""Learner module for neural CBF training loop.

The learner minimizes the combined objective:
    Loss = max_x∈∂S (saturation_risk(x)) + regularization(x)
"""
import os
import time
import datetime
import pickle
import math
# from collections.abc import Callable

import numpy as np
import torch
import torch.optim as optim

from src.utils import save_model


class Learner():
	"""Performs neural CBF training via learner-critic optimization.

	The learner minimizes saturation risk + regularization using gradient descent,
	while the critic provides worst-case counterexamples. Training loop alternates:
	1. Critic finds states on boundary with highest saturation risk
	2. Learner updates CBF to reduce risk at those states
	3. Regularization encourages large safe set
	4. Periodic testing measures safe set volume and boundary violations
	"""
	def __init__(self, args, logger, critic, test_critic, reg_sampler, param_dict: dict,
	             device: torch.device) -> None:
		"""Initializes learner with training configuration.

		Args:
			args: argparse.Namespace with training hyperparameters
			logger: Logger instance
			critic: Critic for finding counterexamples
			test_critic: Separate critic for testing (uses same config)
			reg_sampler: Sampler for regularization loss
			param_dict: System parameters (x_lim, etc.)
			device: PyTorch device
		"""
		self.args = args
		self.logger = logger
		self.critic = critic
		self.test_critic = test_critic
		self.reg_sampler = reg_sampler
		self.param_dict = param_dict
		self.device = device

		self.save_folder = '%s_%s' % (self.args.problem, self.args.affix)
		self.log_folder = os.path.join(self.args.log_root, self.save_folder)
		self.data_save_fpth = os.path.join(self.log_folder, "data.pkl")

		# State space info for volume approximation
		x_lim = param_dict["x_lim"]
		self.x_lim = x_lim
		self.x_dim = x_lim.shape[0]
		self.x_lim_interval_sizes = np.reshape(x_lim[:, 1] - x_lim[:, 0], (1, self.x_dim))

		self.test_N_volume_samples = 2500
		self.test_N_boundary_samples = 2500

	@staticmethod
	def _init_data_dict() -> dict:
		"""Initializes empty lists for all logged quantities."""
		return {
			"train_loop_times": [],
			"train_losses": [],
			"train_counterex_losses": [],
			"train_reg_losses": [],
			"grad_norms": [],
			"reg_grad_norms": [],
			"V_approx_list": [],
			"boundary_samples_obj_values": [],
			"test_t_total": [],
			"test_t_boundary": [],
			"ci_list": [],
			"h_list": [],
			"train_counterexs": [],
			"train_counterex_X_init": [],
			"train_counterex_X_phi_vals": [],
			"train_counterex_init_best_counterex_value": [],
			"train_counterex_final_best_counterex_value": [],
			"train_counterex_t_init": [],
			"train_counterex_t_grad_step": [],
			"train_counterex_t_reproject": [],
			"train_counterex_t_total_opt": [],
			"train_counterex_t_sample_boundary": [],
			"train_counterex_n_segments_sampled": [],
			"train_counterex_dist_diff_after_proj": [],
			"train_counterex_n_opt_steps": [],
		}

	def _avg_grad_norm(self, phi_star_fn: torch.nn.Module) -> float:
		"""Computes average gradient norm over non-excluded parameters."""
		total, count = 0.0, 0
		for name, param in phi_star_fn.named_parameters():
			if name not in phi_star_fn.exclude_from_gradient_param_names:
				total += torch.linalg.norm(param.grad).item()
				count += 1
		return total / count

	def _compute_batch_risk(self, X: torch.Tensor,
	                          saturation_risk: torch.nn.Module) -> tuple:
		"""Computes softmax-weighted counterex loss over boundary samples.

		Only positive violations (φ̇ ≥ 0) contribute. Softmax concentrates
		on worst violations while maintaining gradients from all.

		Returns:
			counterex_value: Weighted loss for backprop
			max_value: Worst single violation (for logging)
		"""
		c = 0.1  # Temperature: higher c = more focus on worst case
		obj = saturation_risk(X)
		pos_inds = torch.where(obj >= 0)
		pos_obj = obj[pos_inds[0], pos_inds[1]].flatten()

		with torch.no_grad():
			w = torch.nn.functional.softmax(c * pos_obj, dim=0)
		counterex_value = torch.dot(w.flatten(), pos_obj.flatten())
		max_value = torch.max(obj)
		return counterex_value, max_value

	def _learner_step(self, counterex_value: torch.Tensor, reg_value: torch.Tensor,
	                  optimizer: torch.optim.Optimizer, pos_params: list,
	                  phi_star_fn: torch.nn.Module) -> tuple:
		"""Backward pass, optimizer step, and positive-parameter projection.

		Returns:
			reg_grad_norm: Gradient norm after reg backward (before counterex backward)
			total_grad_norm: Gradient norm after counterex backward
		"""
		reg_value.backward()
		reg_grad_norm = self._avg_grad_norm(phi_star_fn)

		counterex_value.backward() # collecting gradients on phi_star parameters
		total_grad_norm = self._avg_grad_norm(phi_star_fn)

		optimizer.step()

		with torch.no_grad(): # enforce positivity on ci and h oaraneters
			for param in pos_params:
				param.copy_(torch.maximum(param, torch.zeros_like(param)))

		return reg_grad_norm, total_grad_norm

	def _log_iteration(self, _iter: int, objective_value: torch.Tensor,
	                    max_value: torch.Tensor, reg_value: torch.Tensor,
	                    reg_grad_norm: float, total_grad_norm: float,
	                    x: torch.Tensor, X: torch.Tensor,
	                    critic_debug: dict, phi_star_fn: torch.nn.Module,
	                    t0: float) -> dict:
		"""Logs iteration stats to logger and assembles the per-iteration record.

		Returns:
			iteration_info: Dict of values to append to data_dict
		"""
		t_so_far = time.perf_counter() - t0
		t_so_far_str = str(datetime.timedelta(seconds=t_so_far))

		t_init = critic_debug["t_init"]
		t_grad_step = critic_debug["t_grad_step"]
		t_reproject = critic_debug["t_reproject"]
		t_total_opt = critic_debug["t_total_opt"]

		self.logger.info('\n' + '=' * 20 + f' evaluation at iteration: {_iter} ' + '=' * 20)
		self.logger.info(f'train total loss: {objective_value:.3f}%')
		self.logger.info(f'train max loss: {max_value:.3f}%, reg loss: {reg_value:.3f}%')
		self.logger.info('time spent training so far: %s', t_so_far_str)
		self.logger.info(f'train counterex total time: {t_total_opt:.3f}s')
		self.logger.info(f'train counterex init time: {t_init:.3f}s')
		self.logger.info(f'train counterex avg grad step time: {np.mean(t_grad_step):.3f}s')
		self.logger.info(f'train counterex avg reproj time: {np.mean(t_reproject):.3f}s')
		self.logger.info(f'Reg grad norm: {reg_grad_norm:.3f}')
		self.logger.info(f'total grad norm: {total_grad_norm:.3f}')
		self.logger.info(f'train counterex loss increase over inner max: {(critic_debug["final_best_counterex_value"] - critic_debug["init_best_counterex_value"]):.3f}')
		self.logger.info('OOM debug. Mem allocated and reserved: %f, %f',
		                 torch.cuda.memory_allocated(self.args.gpu),
		                 torch.cuda.memory_reserved(self.args.gpu))
		self.logger.debug('h: %s', phi_star_fn.h)
		self.logger.debug('ci: %s', phi_star_fn.ci)
		self.logger.info('=' * 28 + ' end of evaluation ' + '=' * 28 + '\n')

		return {
			"train_loop_times": t_so_far,
			"train_losses": objective_value,
			"train_counterex_losses": max_value,
			"train_reg_losses": reg_value,
			"grad_norms": total_grad_norm,
			"reg_grad_norms": reg_grad_norm,
			"train_counterexs": x,
			"train_counterex_X_phi_vals": phi_star_fn(X),
			"ci_list": phi_star_fn.ci,
			"h_list": phi_star_fn.h,
			**{"train_counterex_" + k: v for k, v in critic_debug.items()},
		}

	def _compute_test_stats(self, saturation_risk: torch.nn.Module,
	                     phi_star_fn: torch.nn.Module) -> dict:
		"""Computes test stats: safe set volume fraction and boundary violations.

		Returns:
			Dict with keys: V_approx_list, boundary_samples_obj_values,
			                test_t_total, test_t_boundary
		"""
		self.logger.info('\n' + '+' * 20 + ' computing test stats ' + '+' * 20)
		t0_test = time.perf_counter()

		# Monte Carlo safe set volume approximation
		samp_numpy = (np.random.uniform(size=(self.test_N_volume_samples, self.x_dim))
		              * self.x_lim_interval_sizes + self.x_lim[:, [0]].T)
		samp_torch = torch.from_numpy(samp_numpy.astype("float32")).to(self.device)
		M = 100 # Batch size for evaluating φ on test samples to avoid OOM; no grad needed here
		N_samples_inside = 0
		for k in range(math.ceil(self.test_N_volume_samples / float(M))):
			phi_vals_batch = phi_star_fn(samp_torch[k*M: min((k+1)*M, self.test_N_volume_samples)])
			N_samples_inside += torch.sum(torch.max(phi_vals_batch, axis=1)[0] <= 0.0)
		V_approx = (N_samples_inside * 100.0 / float(self.test_N_volume_samples)).item()

		self.logger.info(f'v approx: {V_approx:.3f}% of volume')

		# Sample states on boundary, evaluate percent infeasible and mean/std/max amount infeasible
		t0_test_boundary = time.perf_counter()
		boundary_samples, _ = self.test_critic._sample_points_on_boundary(
			phi_star_fn, self.test_N_boundary_samples)
		boundary_obj = saturation_risk(boundary_samples).detach().cpu().numpy()

		percent_infeas = np.sum(boundary_obj > 0) * 100 / boundary_obj.size
		self.logger.info(f'percentage infeasible at boundary: {percent_infeas:.2f}%')
		infeas_values = (boundary_obj > 0) * boundary_obj
		self.logger.info(f'mean, std amount infeasible at boundary: {np.mean(infeas_values):.2f} +/- {np.std(infeas_values):.2f}')
		self.logger.info(f'max amount infeasible at boundary: {np.max(infeas_values):.2f}')
		self.logger.info('\n' + '+' * 80)

		tf_test = time.perf_counter()

		return {
			"V_approx_list": V_approx,
			"boundary_samples_obj_values": boundary_obj,
			"test_t_total": tf_test - t0_test,
			"test_t_boundary": tf_test - t0_test_boundary,
		}

	def train(self, saturation_risk: torch.nn.Module, reg_fn: torch.nn.Module,
	          phi_star_fn: torch.nn.Module) -> None:
		"""Main training loop implementing Algorithm 1 from paper.

		Algorithm alternates between:
		- Critic: Find worst counterexamples (maximize saturation_risk)
		- Learner: Update CBF (minimize saturation_risk + regularization)

		Also, for saving, logging, and debugging:
		- Saves checkpoints to model_folder every n_checkpoint_step iterations
		- Logs training data to data.pkl
		- Prints progress and statistics to logger

		Args:
			saturation_risk: SaturationRisk loss function
			reg_fn: RegularizationLoss function
			phi_star_fn: Neural CBF (NeuralPhi instance)
		"""
		data_dict = self._init_data_dict()

		p_dict = {p[0]: p[1] for p in phi_star_fn.named_parameters()}
		pos_params = [p_dict[name] for name in phi_star_fn.pos_param_names]
		optimizer = optim.Adam(phi_star_fn.parameters(), lr=self.args.learner_lr)

		t0 = time.perf_counter()
		save_model(phi_star_fn, os.path.join(self.args.model_folder, 'checkpoint_0.pth'))

		for _iter in range(self.args.learner_n_steps + 1):
			# Evaluate regularization on sampled states (from ρ(x) ≤ 0 region)			
			X_reg = self.reg_sampler.get_samples(phi_star_fn)
			reg_value = reg_fn(X_reg)

			# Critic: find worst-case counterexamples on boundary
			x, X, critic_debug = self.critic.opt(saturation_risk, phi_star_fn, _iter)

			# Learner: update CBF to reduce saturation risk
			optimizer.zero_grad()
			counterex_value, max_value = self._compute_batch_risk(X, saturation_risk)
			objective_value = counterex_value + reg_value

			reg_grad_norm, total_grad_norm = self._learner_step(
				counterex_value, reg_value, optimizer, pos_params, phi_star_fn)

			# Log and record iteration stats
			iteration_info = self._log_iteration(
				_iter, objective_value, max_value, reg_value,
				reg_grad_norm, total_grad_norm, x, X, critic_debug, phi_star_fn, t0)
			for key, value in iteration_info.items():
				if torch.is_tensor(value):
					value = value.detach().cpu().numpy()
				data_dict[key].append(value)

			# Checkpoint
			if _iter % self.args.n_checkpoint_step == 0:
				save_model(phi_star_fn, os.path.join(self.args.model_folder, f'checkpoint_{_iter}.pth'))
				self.logger.info("Saving at: %s", self.data_save_fpth)
				with open(self.data_save_fpth, 'wb') as handle:
					pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

			# Periodic test stats
			if _iter % self.args.n_test_loss_step == 0:
				test_stats = self._compute_test_stats(saturation_risk, phi_star_fn)
				for key, value in test_stats.items():
					data_dict[key].append(value)
