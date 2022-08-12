import numpy as np
import math
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False
import IPython
# from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
# from utils_for_corl import A, B
import control

g = 9.81
class CBFController:
	def __init__(self, env, cbf_obj, param_dict, args, eps_bdry=1.0, eps_outside=5.0):
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		self.__dict__.update(self.param_dict)  # __dict__ holds and object's attributes

		# pre-compute mixer matrix
		self.mixer = np.array([[self.k1, self.k1, self.k1, self.k1], [0, -self.l * self.k1, 0, self.l * self.k1],
		                       [self.l * self.k1, 0, -self.l * self.k1, 0], [-self.k2, self.k2, -self.k2, self.k2]])

		if self.args.rollout_u_ref == "LQR":
			L_p = param_dict["L_p"]
			M = param_dict["M"]
			J_x = param_dict["J_x"]
			J_y = param_dict["J_y"]
			J_z = param_dict["J_z"]

			A = np.zeros((10, 10))  # 10 x 10
			A[0:3, 3:6] = np.eye(3)
			A[6:8, 8:10] = np.eye(2)
			A[8, 0] = -3 * g / (2 * L_p)
			A[9, 1] = -3 * g / (2 * L_p)
			A[8, 6] = 3 * g / (2 * L_p)
			A[9, 7] = 3 * g / (2 * L_p)

			B = np.zeros((10, 4))
			B[3:6, 1:4] = np.diag([1.0 / J_x, 1.0 / J_y, 1.0 / J_z])

			# Use LQR to compute feedback portion of controller
			q = self.args.rollout_LQR_q
			r = self.args.rollout_LQR_r
			Q = q * np.eye(10)
			R = r * np.eye(4)
			K, S, E = control.lqr(A, B, Q, R)
			self.K = K

			# print(q)
			# print(K)
			# print(K)
			# IPython.embed()

	def compute_u_ref(self, t, x):
		if self.args.rollout_u_ref == "unactuated":
			u = np.zeros(self.u_dim)
		elif self.args.rollout_u_ref == "LQR":
			u = - self.K @ np.squeeze(x)
		return u

	def compute_control(self, t, x):
		# print("in CBFcontroller, compute control")
		# print(x)
		# IPython.embed()
		############ Init log vars
		apply_u_safe = None
		u_ref = self.compute_u_ref(t, x)
		phi_vals = None
		qp_slack = None
		qp_lhs = None
		qp_rhs = None
		impulses = None
		inside_boundary = False
		on_boundary = False
		outside_boundary = False
		################
		# apply_u_safe = False
		# inside_boundary = True
		# debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs": qp_rhs,
		#               "qp_lhs": qp_lhs, "phi_vals": [0], "impulses": impulses,
		#               "inside_boundary": inside_boundary, "on_boundary": on_boundary,
		#               "outside_boundary": outside_boundary, "dist_between_xs": 0,
		#               "phi_grad_mag": 0, "phi_grad": 0}
		# return u_ref, debug_dict


		phi_vals = self.cbf_obj.phi_fn(x)  # This is an array of (1, r+1), where r is the degree
		phi_grad = self.cbf_obj.phi_grad(x)

		# print(x.shape)
		x_next = x + self.env.dt * self.env.x_dot_open_loop(x, self.compute_u_ref(t,
		                                                                          x))  # in the absence of safe control, the next state
		next_phi_val = self.cbf_obj.phi_fn(x_next)

		dist_between_xs = np.linalg.norm(x_next - x)
		phi_grad_mag = np.linalg.norm(phi_grad)

		if phi_vals[0, -1] > 0:  # Outside
			# print("STATUS: Outside") # TODO
			eps = self.eps_outside
			apply_u_safe = True
			outside_boundary = True
		elif phi_vals[0, -1] < 0 and next_phi_val[0, -1] >= 0:  # On boundary. Note: cheating way to convert DT to CT
			# print("STATUS: On") # TODO
			eps = self.eps_bdry
			apply_u_safe = True
			on_boundary = True
		else:  # Inside
			# print("STATUS: Inside") # TODO
			apply_u_safe = False
			inside_boundary = True
			debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs": qp_rhs,
			              "qp_lhs": qp_lhs, "phi_vals": phi_vals.flatten(), "impulses": impulses,
			              "inside_boundary": inside_boundary, "on_boundary": on_boundary, "outside_boundary": outside_boundary, "dist_between_xs": dist_between_xs, "phi_grad_mag": phi_grad_mag, "phi_grad": phi_grad}
			return u_ref, debug_dict

		# IPython.embed()
		# Compute the control constraints
		f_x = self.env._f(x)
		f_x = np.reshape(f_x, (16, 1))
		g_x = self.env._g(x)

		phi_grad = np.reshape(phi_grad, (16, 1))
		lhs = phi_grad.T @ g_x  # 1 x 4
		rhs = -phi_grad.T @ f_x - eps
		rhs = rhs.item()  # scalar, not numpy array

		# Saving data
		qp_lhs = lhs
		qp_rhs = rhs

		# Computing control using QP
		# Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
		w = 1000.0  # slack weight

		# P = np.zeros((5, 5))
		# P[:4, :4] = 2 * self.mixer.T @ self.mixer
		# q = np.concatenate([-2 * u_ref.T @ self.mixer, np.array([w])])
		# q = np.reshape(q, (-1, 1))
		#
		# G = np.zeros((10, 5))
		# G[0, 0:4] = lhs @ self.mixer
		# G[0, 4] = -1.0
		# G[1:5, 0:4] = -np.eye(4)
		# G[5:9, 0:4] = np.eye(4)
		# G[-1, -1] = -1.0
		#
		# h = np.concatenate([np.array([rhs]), np.zeros(4), np.ones(4), np.zeros(1)])
		# h = np.reshape(h, (-1, 1))

		P = np.zeros((9, 9))
		P[:4, :4] = 2 *np.eye(4)
		q = np.zeros((9, 1))
		q[:4, 0] = -2*u_ref
		q[-1, 0] = w

		# G <= h
		G = np.zeros((10,9))
		G[0, :4] = lhs
		G[0, -1] = -1.0
		##
		G[1:5, 4:8] = -np.eye(4)
		G[5:9, 4:8] = np.eye(4)
		G[9, -1] = -1.0

		h = np.zeros((10, 1))
		h[0, 0] = rhs
		##
		h[5:9, 0] = 1.0

		A = np.zeros((4, 9))
		A[:4, :4] = -np.eye(4)
		A[:4, 4:8] = self.mixer
		b = np.array([self.M*g, 0, 0, 0])[:, None]

		# print("line 177, flying_cbf_controller")
		# IPython.embed()

		try:
			# sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
			sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
		except:
			# IPython.embed()
			print("QP solve was unsuccessful, with status: %s " % sol_obj["status"])
			print("Go to line 96 in flying_cbf_controller")
			IPython.embed()
			print("exiting")
			exit(0)

		# print("ln 94 in cbf controller")
		# print("Try to check out the properties on sol_obj")
		# print("So that we can debug exceptions")
		# IPython.embed()
		sol_var = np.array(sol_obj['x'])

		# u_safe = sol_var[0:4]
		# sol_impulses = sol_var[0:4]
		# u_safe = self.mixer @ np.reshape(sol_impulses, (4, 1))

		u_safe = sol_var[0:4]
		u_safe = np.reshape(u_safe, (4))
		qp_slack = sol_var[-1]

		# print("Slack: %.6f" % qp_slack) # TODO
		# print(sol_impulses, u_safe, qp_slack)
		impulses = sol_var[4:8]
		debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "phi_vals": phi_vals.flatten(),
		              "qp_slack": qp_slack, "qp_rhs": qp_rhs, "qp_lhs": qp_lhs, "impulses": impulses,
		             "inside_boundary": inside_boundary, "on_boundary": on_boundary, "outside_boundary": outside_boundary, "dist_between_xs": dist_between_xs, "phi_grad_mag": phi_grad_mag, "phi_grad": phi_grad.flatten()}
		return u_safe, debug_dict
