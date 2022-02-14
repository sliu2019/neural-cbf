import numpy as np
import math
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import IPython
from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv

class CBFController:
	def __init__(self, env, cbf_obj, param_dict, eps_bdry=1.0, eps_outside=5.0):
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		self.__dict__.update(self.param_dict)  # __dict__ holds and object's attributes

		# pre-compute mixer matrix
		self.mixer = np.array([[self.k1, self.k1, self.k1, self.k1], [0, -self.l*self.k1, 0, self.l*self.k1], [self.l*self.k1, 0, -self.l*self.k1, 0], [-self.k2, self.k2, -self.k2, self.k2]])

	def compute_u_ref(self, t, x):
		u_ref = np.zeros(self.u_dim)
		return u_ref

	def compute_control(self, t, x):
		print("in CBFcontroller, compute control")
		IPython.embed()
		############ Init log vars
		apply_u_safe = None
		u_ref = self.compute_u_ref(t, x)
		phi_vals = None
		qp_slack = None
		qp_lhs = None
		qp_rhs = None
		################

		phi_vals = self.cbf_obj.phi_fn(x)  # This is an array of (1, r+1), where r is the degree
		phi_grad = self.cbf_obj.phi_grad(x)

		x_next = x + self.env.dt * self.env.x_dot_open_loop(x, self.compute_u_ref(t, x))  # in the absence of safe control, the next state
		next_phi_val = self.cbf_obj.phi_fn(x_next)

		if phi_vals[0, -1] > 0:  # Outside
			eps = self.eps_outside
			apply_u_safe = True
		elif phi_vals[0, -1] < 0 and next_phi_val[0, -1] >= 0:  # On boundary. Note: cheating way to convert DT to CT
			eps = self.eps_bdry
			apply_u_safe = True
		else:  # Inside
			apply_u_safe = False
			debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs": qp_rhs,
			              "qp_lhs": qp_lhs, "phi_vals": phi_vals.flatten()}
			return u_ref, debug_dict

		# Compute the control constraints
		f_x = self.env._f(x)
		g_x = self.env._g(x)

		lhs = phi_grad.T @ g_x # 1 x 4
		rhs = -phi_grad.T @ f_x - eps

		# Computing control using QP
		# Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
		w = 1000.0  # slack weight

		Q = np.zeros((5, 5))
		Q[:4, :4] = 2*self.mixer.T@self.mixer
		p = np.array([-2*u_ref.T@self.mixer, w])
		p = p.T # column?

		G = np.zeros((10, 5))
		G[0, :4] = lhs@self.mixer
		G[0, 4] = -1.0
		G[1:5, 0:4] = -np.eye(4)
		G[5:9, 0:4] = np.eye(4)
		G[-1, -1] = -1.0

		h = np.array([rhs, np.zeros(4), np.ones(4), 0.0])
		h = h.T # column?

		try:
			sol_obj = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h))
		except:
			# IPython.embed()
			exit(0)
		sol_var = sol_obj['x']

		u_safe = sol_var[0]
		qp_slack = sol_var[1]

		debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "phi_vals": phi_vals.flatten(),
		              "qp_slack": qp_slack, "qp_rhs": qp_rhs, "qp_lhs": qp_lhs}
		return u_safe, debug_dict
