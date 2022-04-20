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

class FlyingPendEvaluator(object):
	def __init__(self, arg_dict):
		self.arg_dict = arg_dict

		self.env = FlyingInvertedPendulumEnv()
		# Defining some var
		self.dt = self.env.dt
		self.reg_weight = arg_dict["FlyingPendEvaluator_reg_weight"]
		# n_samples = 100000
		n_samples = arg_dict["FlyingPendEvaluator_n_samples"]

		self.samples = []
		for i in range(n_samples):
			x_sample = (self.env.x_lim[:, 1] - self.env.x_lim[:, 0]) * np.random.random_sample(
				(len(self.env.x_lim[:, 0]),)) + self.env.x_lim[:, 0]

			x_sample = np.concatenate((x_sample, np.zeros(6))) # (n_samples, 10) --> (n_samples, 16)
			self.samples.append(x_sample)

		# Create CBF class
		self.ssa_torch_module = load_philow() # load clean phi_low, not from checkpoint
		self.ssa = PhiNumpy(self.ssa_torch_module)

	def set_params(self, params):
		state_dict = {"ki": torch.tensor([[params[2]]]), "ci": torch.tensor([[params[0]], [params[1]]])}
		self.ssa.set_params(state_dict)

	def most_valid_control(self, grad_phi, x):
		# print("inside most_valid_control")
		# IPython.embed()
		# todo: what's the vstack part for?
		f = np.vstack(self.env._f(x))
		g = np.vstack(self.env._g(x))

		# f = self._f_trunc_input(x)
		# g = self._g_trunc_input(x)

		min_dot_phi = float("inf")
		for u in self.env.control_lim_verts:
			min_dot_phi = min(min_dot_phi, grad_phi @ f + grad_phi @ g @ u)

		return min_dot_phi

	def near_boundary(self, phis):
		# Technically, the only boundary we care about has phi[-1] = 0 and phi_i <= 0
		phi = phis[0, -1]
		eps = 1e-2
		return abs(phi) < eps

	# def _f_trunc_input(self, x):
	# 	"""
	# 	Numpy flying_inv_pend takes in 16D state, outputs 16D vec
	# 	We need 10D-10D mapping
	# 	:param x:
	# 	:return:
	# 	"""
	# 	padding = np.zeros((6))
	# 	x_padded = np.concatenate((x, padding))
	# 	f_out = self.env._f(x_padded)
	# 	f_out_trunc = f_out[:10]
	# 	return f_out_trunc
	#
	# def _g_trunc_input(self, x):
	# 	"""
	# 	Numpy flying_inv_pend takes in 16D state, outputs 16D vec
	# 	We need 10D-10D mapping
	# 	:param x:
	# 	:return:
	# 	"""
	# 	padding = np.zeros((6))
	# 	x_padded = np.concatenate((x, padding))
	# 	g_out = self.env._g(x_padded)
	# 	g_out_trunc = g_out[:10]
	# 	return g_out_trunc

	def compute_valid_invariant(self):
		# print("in compute_valid_invariant")
		# IPython.embed()

		in_invariant = 0
		valid = 0
		tot_cnt = 0

		for sample in self.samples: # Note: set of samples is fixed across CMA-ES opt.
			x = sample
			phis = self.ssa.phi_fn(x)
			phi = phis[0, -1]

			if self.near_boundary(phis):
				tot_cnt += 1

				C = self.ssa.phi_grad(x)
				# d = -phi / self.dt if phi < 0 else -1 # TODO: objective differs. Aims for dot(phi) <= -alpha(phi(x))
				d = 0
				most_valid = self.most_valid_control(C, x)
				valid += (most_valid < d)

			if np.max(phis) <= 0:
				in_invariant += 1

		# self.valid = valid # ?
		valid_rate = float(valid) / max(1, tot_cnt)
		print("valid / tot_cnt: ", valid, "/", tot_cnt) #

		#### Reg term ####
		in_invariant_rate = float(in_invariant) / len(self.samples)
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