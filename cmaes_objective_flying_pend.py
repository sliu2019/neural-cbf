import numpy as np
# import matplotlib.pyplot as plt
# from rollout_envs.cart_pole_env import CartPoleEnv
from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
# import seaborn as sns
# from rollout_cbf_classes.deprecated.normal_ssa_newsi import SSA
# from rollout_cbf_classes.deprecated.flying_pend_ssa import FlyingPendSSA
# from .normal_ssa import SSA
from phi_low_torch_module import PhiLow
from phi_numpy_wrapper import PhiNumpy
from cmaes.utils import load_philow
# from src.problems.flying_inv_pend import HSum, XDot # TODO: how to import this correctly?
# from main import create_flying_param_dict
# from src.argument import create_parser
import torch
import IPython
import math

class FlyingPendEvaluator(object):
	def __init__(self, arg_dict):
		self.arg_dict = arg_dict

		self.env = FlyingInvertedPendulumEnv()
		# Defining some var
		self.dt = self.env.dt
		self.reg_weight = arg_dict["FlyingPendEvaluator_reg_weight"]
		# n_samples = 100000
		self.n_samples = arg_dict["FlyingPendEvaluator_n_samples"]

		self.samples = []
		for i in range(self.n_samples):
			x_sample = (self.env.x_lim[:, 1] - self.env.x_lim[:, 0]) * np.random.random_sample(
				(len(self.env.x_lim[:, 0]),)) + self.env.x_lim[:, 0]

			x_sample = np.concatenate((x_sample, np.zeros(6))) # (self.n_samples, 10) --> (self.n_samples, 16)
			self.samples.append(x_sample)

		# Create CBF class
		self.ssa_torch_module = load_philow() # load clean phi_low, not from checkpoint
		self.ssa = PhiNumpy(self.ssa_torch_module)

		# For faster computation
		self.conp_bs = 100

		# Misc
		self.near_boundary_eps = 1e-2

	def set_params(self, params):
		state_dict = {"ki": torch.tensor([[params[2]]]), "ci": torch.tensor([[params[0]], [params[1]]])}
		self.ssa.set_params(state_dict)

	# def most_valid_control(self, grad_phi, x):
	# 	# print("inside most_valid_control")
	# 	# IPython.embed()
	# 	# todo: what's the vstack part for?
	# 	f = np.vstack(self.env._f(x))
	# 	g = np.vstack(self.env._g(x))
	#
	# 	# f = self._f_trunc_input(x)
	# 	# g = self._g_trunc_input(x)
	#
	# 	min_dot_phi = float("inf")
	# 	for u in self.env.control_lim_verts:
	# 		min_dot_phi = min(min_dot_phi, grad_phi @ f + grad_phi @ g @ u)
	#
	# 	return min_dot_phi

	# def near_boundary(self, phis):
	# 	# Technically, the only boundary we care about has phi[-1] = 0 and phi_i <= 0
	# 	phi = phis[0, -1]
	# 	eps = 1e-2
	# 	return abs(phi) < eps

	def compute_valid_invariant(self):
		# print("in compute_valid_invariant")
		# IPython.embed()

		in_invariant = 0
		valid = 0
		tot_near_boundary = 0

		"""i = 0
		for sample in self.samples: # Note: set of samples is fixed across CMA-ES opt.
			# print(i)
			x = sample
			phis = self.ssa.phi_fn(x)
			phi = phis[0, -1]

			if self.near_boundary(phis):
				tot_near_boundary += 1

				C = self.ssa.phi_grad(x)
				# d = -phi / self.dt if phi < 0 else -1 # TODO: objective differs. Aims for dot(phi) <= -alpha(phi(x))
				d = 0
				most_valid = self.most_valid_control(C, x)
				valid += (most_valid < d)

			if np.max(phis) <= 0:
				in_invariant += 1

			i += 1"""

		for i in math.ceil(self.n_samples/float(self.comp_bs)):
			IPython.embed()
			x_batch = self.samples[i*self.comp_bs:min((i+1)*self.comp_bs, self.n_samples)]

			# Check if on boundary
			phis_batch = self.ssa.phi_fn(x_batch)

			ind_near_boundary = np.argwhere(np.abs(phis_batch[:, -1]) < self.near_boundary_eps).flatten()
			tot_near_boundary += len(ind_near_boundary)

			# If yes, then compute feasibility of safe control
			f_batch = self.env._f(x_batch)
			g_batch = self.env._g(x_batch)

			grad_phi_batch = self.ssa.phi_grad(x_batch)

			# Starting to compute
			grad_phi_batch = np.reshape(grad_phi_batch, (-1, 1, 16))
			f_batch = np.reshape(f_batch, (-1, 16, 1))
			u_batch = np.tile(self.env.control_lim_verts[None], (x_batch.shape[0], 1, 1))
			phidot_for_all_u = grad_phi_batch @ f_batch + grad_phi_batch @ g_batch @ u_batch  # (bs, nu)

			min_phidot_over_all_u = np.min(phidot_for_all_u, axis=1)
			valid += np.sum(min_phidot_over_all_u < 0)

			n_in_S = np.sum(np.all(phis_batch < 0, axis=1))
			in_invariant += n_in_S


		# self.valid = valid # ?
		valid_rate = float(valid) / max(1, tot_near_boundary)
		print("valid / tot_near_boundary: ", valid, "/", tot_near_boundary) #

		#### Reg term ####
		in_invariant_rate = float(in_invariant) / self.n_samples
		return valid_rate, in_invariant_rate

	def evaluate(self, params):
		# print("Trying params: ", params)
		self.set_params(params)

		valid_rate, in_invariant_rate = self.compute_valid_invariant()
		self.valid_rate = valid_rate
		self.in_invariant_rate = in_invariant_rate
		rv = valid_rate + self.reg_weight * in_invariant_rate
		print("valid rate: ", valid_rate, "in_invariant_rate: ", in_invariant_rate, "params: ", params) # TODO: why printed and not logged?
		debug_dict = {"obj:valid_rate": valid_rate, "obj:in_invariant_rate": in_invariant_rate}

		# print("before returning from evaluate on Objective class")
		# IPython.embed()
		return rv, debug_dict

	# @property
	# def log(self):
	# 	# return "{} {}".format(str(self.coe), str(self.valid))
	# 	# return "{} {} {}".format(str(self.coe), str(self.valid))
	# 	# s = "Params: %s, valid rate: %f, volume rate: %f" % (str(self.coe), self.valid_rate, self.in_invariant_rate)
	# 	# return s
	# 	return ""