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
# import socket, sys
# if socket.gethostname() == "nsh1609server4":
# 	# IPython.embed()
# 	sys.path.extend(['/home/simin/anaconda3/envs/si_feas_env/lib/python38.zip', '/home/simin/anaconda3/envs/si_feas_env/lib/python3.8', '/home/simin/anaconda3/envs/si_feas_env/lib/python3.8/lib-dynload', '/home/simin/anaconda3/envs/si_feas_env/lib/python3.8/site-packages'])
import sys
print("in objective file")
print(sys.path)
from cmaes.utils import load_philow_and_params
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
		self.n_samples = arg_dict["FlyingPendEvaluator_n_samples"]

		self.samples = np.random.random_sample((self.n_samples, len(self.env.x_lim[:, 0])))*(self.env.x_lim[:, 1] - self.env.x_lim[:, 0]) + self.env.x_lim[:, 0]
		self.samples = np.concatenate((self.samples, np.zeros((self.n_samples, 6))), axis=1) # (self.n_samples, 10) --> (self.n_samples, 16)

		# Create CBF class
		self.ssa_torch_module, _ = load_philow_and_params() # load clean phi_low, not from checkpoint
		self.ssa = PhiNumpy(self.ssa_torch_module)

		# For faster computation
		self.comp_bs = 100

		# Misc
		self.objective_type = arg_dict["FlyingPendEvaluator_objective_type"]
		self.near_boundary_eps = arg_dict["FlyingPendEvaluator_near_boundary_eps"]

	def set_params(self, params):
		state_dict = {"ki": torch.tensor([[params[2]]]), "ci": torch.tensor([[params[0]], [params[1]]])}
		self.ssa.set_params(state_dict)

	def evaluate(self, params):
		# print("Trying params: ", params)
		self.set_params(params)

		n_inside = 0
		objective_value = 0
		n_near_boundary = 0

		for i in range(math.ceil(self.n_samples/float(self.comp_bs))):

			x_batch = self.samples[i*self.comp_bs:min((i+1)*self.comp_bs, self.n_samples)]

			# Check if on boundary
			phis_batch = self.ssa.phi_fn(x_batch)

			ind_near_boundary = np.argwhere(np.abs(phis_batch[:, -1]) < self.near_boundary_eps).flatten()
			n_near_boundary += len(ind_near_boundary)

			# If yes, then compute feasibility of safe control
			x_near = x_batch[ind_near_boundary]
			f_batch = self.env._f(x_near)
			g_batch = self.env._g(x_near)

			grad_phi_batch = self.ssa.phi_grad(x_near)

			# Starting to compute
			grad_phi_batch = np.reshape(grad_phi_batch, (-1, 1, 16))
			f_batch = np.reshape(f_batch, (-1, 16, 1))
			u_batch = np.tile(self.env.control_lim_verts.T[None], (x_near.shape[0], 1, 1))
			phidot_for_all_u = grad_phi_batch @ f_batch + grad_phi_batch @ g_batch @ u_batch  # (bs, nu)
			phidot_for_all_u = phidot_for_all_u[:, 0]

			min_phidot_over_all_u = np.min(phidot_for_all_u, axis=1)

			n_in_S = np.sum(np.all(phis_batch < 0, axis=1))
			n_inside += n_in_S

			# self.n_feasible = n_feasible # ?
			# ["n_feasible", "avg_amount_infeasible", "max_amount_infeasible"]
			if self.objective_type == "n_feasible":
				n_feasible = np.sum(min_phidot_over_all_u < 0)
				objective_value += float(n_feasible)
			elif self.objective_type == "avg_amount_infeasible":
				objective_value -= np.sum((min_phidot_over_all_u > 0)*min_phidot_over_all_u)
			elif self.objective_type == "max_amount_infeasible":
				# IPython.embed()
				objective_value -= np.max((min_phidot_over_all_u > 0)*min_phidot_over_all_u, initial=0.0) # basically, a relu on max; 2nd arg lets you work with empty array

		# normalize objective by n_samples
		objective_value /= float(max(1, n_near_boundary))
		# print("n_feasible / n_near_boundary: ", n_feasible, "/", n_near_boundary) #
		# print("\n")

		#### Reg term ####
		percentage_inside = float(n_inside)*100.0 / self.n_samples
		# return objective_value, percentage_inside

		# objective_value, percentage_inside = self.compute_valid_invariant()
		# self.objective_value = objective_value
		# self.percentage_inside = percentage_inside
		rv = objective_value + self.reg_weight * percentage_inside
		print("objective value: ", objective_value, "percentage_inside: ", percentage_inside) # TODO: why printed and not logged?
		print("params: ", params)
		print("\n")
		debug_dict = {"obj:objective_value": objective_value, "obj:percentage_inside": percentage_inside, "obj:n_near_boundary": n_near_boundary}

		# print("before returning from evaluate on objective_value class")
		# IPython.embed()
		return rv, debug_dict

	# @property
	# def log(self):
	# 	# return "{} {}".format(str(self.coe), str(self.valid))
	# 	# return "{} {} {}".format(str(self.coe), str(self.valid))
	# 	# s = "Params: %s, valid rate: %f, volume rate: %f" % (str(self.coe), self.objective_value, self.percentage_inside)
	# 	# return s
	# 	return ""