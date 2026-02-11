import numpy as np
import math
from plot_utils import create_phi_struct_load_xlim
from torch.autograd import grad
from src.utils import *
from scipy.integrate import solve_ivp
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

from rollout_envs.quad_pend_env import QuadPendEnv
# from rollout_envs.cart_pole_env import CartPoleEnv


class CBFController:
	def __init__(self, env, cbf_obj, eps_bdry=1.0, eps_outside=5.0):
		# super().__init__()
		# variables = locals()  # dict of local names
		# self.__dict__.update(variables)  # __dict__ holds and object's attributes
		# del self.__dict__["self"]  # don't need `self`
		self.cbf_obj = cbf_obj
		self.env = env
		# self.param_dict = param_dict
		self.eps_bdry = eps_bdry
		self.eps_outside = eps_outside

	def compute_u_ref(self, t, x):
		return 0

	def compute_control(self, t, x):
		############ Init log vars
		apply_u_safe = None
		u_ref = self.compute_u_ref(t, x)
		phi_vals = None
		qp_slack = None
		qp_lhs = None
		qp_rhs = None
		################

		# phi_vals = numpy_phi_star_fn(x) # This is an array of (1, r+1), where r is the degree
		# phi_grad = numpy_phi_grad(x)

		phi_vals = self.cbf_obj.phi_star_fn(x)  # This is an array of (1, r+1), where r is the degree
		phi_grad = self.cbf_obj.phi_grad(x)

		x_next = x + self.env.dt * self.env.x_dot_open_loop(x, self.compute_u_ref(t,
		                                                                          x))  # in the absence of safe control, the next state
		next_phi_val = self.cbf_obj.phi_star_fn(x_next)

		# IPython.embed()
		if phi_vals[0, -1] > 0:  # Outside
			eps = self.eps_outside
			apply_u_safe = True
		elif phi_vals[0, -1] < 0 and next_phi_val[0, -1] >= 0:  # On boundary. Note: cheating way to convert DT to CT
			# eps = 1.0 # TODO
			eps = self.eps_bdry
			apply_u_safe = True
		else:  # Inside
			apply_u_safe = False
			debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs": qp_rhs,
			              "qp_lhs": qp_lhs, "phi_vals": phi_vals.flatten()}
			return u_ref, debug_dict

		# Compute the control constraints
		# Get f(x), g(x); note it's a hack for scalar u
		f_x = self.env.x_dot_open_loop(x, 0)
		g_x = self.env.x_dot_open_loop(x, 1) - f_x

		lhs = phi_grad @ g_x.T
		rhs = -phi_grad @ f_x.T - eps

		# Computing control using QP
		# Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
		w = 1000.0  # slack weight

		max_force = 22.0

		qp_lhs = lhs.item()
		qp_rhs = rhs.item()
		Q = 2 * np.array([[1.0, 0], [0, 0]])
		p = np.array([[-2.0 * u_ref], [w]])
		G = np.array([[qp_lhs, -1.0], [1, 0], [-1, 0], [0, -1]])
		rho = np.array([[qp_rhs], [max_force], [max_force], [0.0]])

		try:
			sol_obj = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(rho))
		except:
			# IPython.embed()
			exit(0)
		sol_var = sol_obj['x']

		u_safe = sol_var[0]
		qp_slack = sol_var[1]

		debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "phi_vals": phi_vals.flatten(),
		              "qp_slack": qp_slack, "qp_rhs": qp_rhs, "qp_lhs": qp_lhs}
		return u_safe, debug_dict