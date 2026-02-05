import do_mpc
import IPython
import numpy as np, math
from casadi import *
from copy import deepcopy
import os
import control
import torch
from torch.autograd.functional import jacobian
from torch.autograd import grad
from torch import nn

from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

from scipy.linalg import cholesky  # computes upper triangle by default, matches paper

# Imports from my module
from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
# from src.problems.flying_inv_pend import XDot

# Fixed seed for repeatability
np.random.seed(2022)

g = 9.81

from src.argument import create_parser

parser = create_parser()  # default
default_args = parser.parse_known_args()[0]
from main import create_flying_param_dict

param_dict = create_flying_param_dict(default_args)  # default
state_index_names = param_dict["state_index_names"]
state_index_dict = param_dict["state_index_dict"]
delta_safety_limit = param_dict["delta_safety_limit"]

def compute_flying_inv_pend_LQR_matrix(q, r):
	x_dim = 16

	L_p = param_dict["L_p"]
	M = param_dict["M"]
	J_x = param_dict["J_x"]
	J_y = param_dict["J_y"]
	J_z = param_dict["J_z"]

	A = np.zeros((x_dim, x_dim))  # 10 x 10
	A[0:3, 3:6] = np.eye(3)
	A[6:8, 8:10] = np.eye(2)
	A[8, 0] = -3 * g / (2 * L_p)
	A[9, 1] = -3 * g / (2 * L_p)
	A[8, 6] = 3 * g / (2 * L_p)
	A[9, 7] = 3 * g / (2 * L_p)
	A[10:13, 13:16] = np.eye(3)  # x,y,z
	A[13, 1] = g
	A[14, 0] = -g

	B = np.zeros((x_dim, 4))
	B[3:6, 1:4] = np.diag([1.0 / J_x, 1.0 / J_y, 1.0 / J_z])
	B[15, 0] = 1 / M

	Q = q * np.eye(x_dim)
	R = r * np.eye(4)
	K, S, E = control.lqr(A, B, Q, R)

	# print(K)
	# print("sanity-check the above LQR feedback matrix, if it's possible to understand it")
	# IPython.embed()
	return K


class H(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Return value is size (bs, 1)

		# print("Inside HSum forward")
		# IPython.embed()
		theta = x[:, [self.i["theta"]]]
		phi = x[:, [self.i["phi"]]]
		gamma = x[:, [self.i["gamma"]]]
		beta = x[:, [self.i["beta"]]]

		cos_cos = torch.cos(theta) * torch.cos(phi)
		eps = 1e-4  # prevents nan when cos_cos = +/- 1 (at x = 0)
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos) * eps
		delta = torch.acos(cos_cos + signed_eps)
		rv = delta ** 2 + gamma ** 2 + beta ** 2 - self.delta_safety_limit ** 2

		return rv


class Hb(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict

		self.radius = 1e-1

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Return value is size (bs, 1)

		# print("Inside Hb's forward")
		# IPython.embed()

		rv = torch.linalg.norm(x, dim=1) - (self.radius ** 2)
		rv = rv[:, None]

		return rv

class XDot(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device
		state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
		                     "dtheta", "x", "y", "z", "dx", "dy", "dz"]
		state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
		self.i = state_index_dict

	def forward(self, x, u):
		# x: bs x 12, u: bs x 4
		# The way these are implemented should be batch compliant

		# Pre-computations
		# Compute the rotation matrix from quad to global frame
		# Extract the k_{x,y,z}
		gamma = x[:, self.i["gamma"]]
		beta = x[:, self.i["beta"]]
		alpha = x[:, self.i["alpha"]]

		phi = x[:, self.i["phi"]]
		theta = x[:, self.i["theta"]]
		dphi = x[:, self.i["dphi"]]
		dtheta = x[:, self.i["dtheta"]]

		R = torch.zeros((x.shape[0], 3, 3), device=self.device)  # is this the correct rotation?
		R[:, 0, 0] = torch.cos(alpha) * torch.cos(beta)
		R[:, 0, 1] = torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma) - torch.sin(alpha) * torch.cos(gamma)
		R[:, 0, 2] = torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma) + torch.sin(alpha) * torch.sin(gamma)
		R[:, 1, 0] = torch.sin(alpha) * torch.cos(beta)
		R[:, 1, 1] = torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)
		R[:, 1, 2] = torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.cos(alpha) * torch.sin(gamma)
		R[:, 2, 0] = -torch.sin(beta)
		R[:, 2, 1] = torch.cos(beta) * torch.sin(gamma)
		R[:, 2, 2] = torch.cos(beta) * torch.cos(gamma)

		k_x = R[:, 0, 2]
		k_y = R[:, 1, 2]
		k_z = R[:, 2, 2]

		F = (u[:, 0] + self.M * g)

		###### Computing state derivatives
		J = torch.tensor([self.J_x, self.J_y, self.J_z]).to(self.device)
		# J_inv = np.diag([(1.0 / self.J_x), (1.0 / self.J_y), (1.0 / self.J_z)])
		norm_torques = u[:, 1:] * (1.0 / J)

		ddquad_angles = torch.bmm(R, norm_torques[:, :, None])  # (N, 3, 1)
		ddquad_angles = ddquad_angles[:, :, 0]

		ddgamma = ddquad_angles[:, 0]
		ddbeta = ddquad_angles[:, 1]
		ddalpha = ddquad_angles[:, 2]

		ddx = k_x * (F / self.M)
		ddy = k_y * (F / self.M)
		ddz = k_z * (F / self.M) - g

		ddphi = (3.0)*(k_y*torch.cos(phi) + k_z*torch.sin(phi))/(2*self.M*self.L_p*torch.cos(theta))*F + 2*dtheta*dphi*torch.tan(theta)
		ddtheta = (3.0*(-k_x*torch.cos(theta)-k_y*torch.sin(phi)*torch.sin(theta) + k_z*torch.cos(phi)*torch.sin(theta))/(2.0*self.M*self.L_p))*F - torch.square(dphi)*torch.sin(theta)*torch.cos(theta)

		# Excluding translational motion
		# IPython.embed()
		rv = torch.cat([x[:, [self.i["dgamma"]]], x[:, [self.i["dbeta"]]], x[:, [self.i["dalpha"]]],
		                ddgamma[:, None], ddbeta[:, None], ddalpha[:, None], dphi[:, None], dtheta[:, None], ddphi[:, None], ddtheta[:, None], x[:, [self.i["dx"]]], x[:, [self.i["dy"]]], x[:, [self.i["dz"]]], ddx[:, None], ddy[:, None],
		                ddz[:, None]], axis=1)
		return rv


"""
    parser.add_argument('--rollout_u_ref', type=str, choices=["unactuated", "LQR"], default="unactuated")
    parser.add_argument('--rollout_LQR_q', type=float, default=0.1)
    parser.add_argument('--rollout_LQR_r', type=float, default=1.0)
"""
"""
parser.add_argument('--flow_dt', type=float, default=0.05, help="dt for discretized computation of the dynamics flow")
parser.add_argument('--flow_T', type=float, default=1.0, help="length (in total time) of dynamics flow")
parser.add_argument('--backup_LQR_q', type=float, default=1.0, help="assume backup controller is LQR")
parser.add_argument('--backup_LQR_r', type=float, default=1.0, help="assume backup controller is LQR")
"""


def numpy_to_torch(x):
	"""
    Convert to float (from possibly double) and to torch array
    :param x:
    :return:
    """
	rv = torch.from_numpy(x.astype(np.float32))
	return rv


class BackupSetController:
	def __init__(self, sim_env, param_dict, args, eps_bdry=1.0, eps_outside=5.0):
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		self.__dict__.update(self.param_dict)  # __dict__ holds and object's attributes

		# Setting up
		self.flow_env = FlyingInvertedPendulumEnv(model_param_dict=param_dict,
		                                          dt=args.flow_dt)  # TODO: note, at flow_dt not dt
		self.backup_K = compute_flying_inv_pend_LQR_matrix(args.backup_LQR_q, args.backup_LQR_r)
		self.backup_K_torch = numpy_to_torch(self.backup_K)

		if self.args.rollout_u_ref == "LQR":
			self.ref_K = compute_flying_inv_pend_LQR_matrix(args.rollout_LQR_q, args.rollout_LQR_r)
		dev = "cpu"
		device = torch.device(dev)
		self.xdot_torch_function = XDot(param_dict, device)
		self.h_torch_function = H(param_dict)
		self.hb_torch_function = Hb(param_dict)

		# pre-compute mixer matrix
		self.mixer = np.array([[self.k1, self.k1, self.k1, self.k1], [0, -self.l * self.k1, 0, self.l * self.k1],
		                       [self.l * self.k1, 0, -self.l * self.k1, 0], [-self.k2, self.k2, -self.k2, self.k2]])

		self.mixer_inv = np.linalg.inv(self.mixer)
		self.torch_mixer_inv = torch.from_numpy(self.mixer_inv.astype("float32"))
		self.torch_mixer = torch.from_numpy(self.mixer.astype("float32"))

	def compute_backup_control(self, x):
		"""
        :param x: (1, 16)
        :return: (1, 4)
        """
		# print("compute_backup_control")
		# IPython.embed()
		u = -self.backup_K @ x.flatten()
		u = np.reshape(u, (1, self.u_dim))
		return u

	def f_cl_torch(self, x):
		"""
        :param x: (16)
        :return: (16)
        So that Jacobian will be (16, 16)
        """
		# print("Debug dimensions, etc. in f_cl_torch")
		# IPython.embed()
		# with torch.no_grad(): # TODO: u = -kx should be involved in the Jacobian
		u = -self.backup_K_torch @ x

		# TODO: should include clipping here
		if u.ndim == 1:
			u = u[None]
		u_gravity_comp = u + torch.tensor([self.M * g, 0, 0, 0])[None]  # u with gravity compensation
		motor_impulses = u_gravity_comp @ torch.t(self.torch_mixer_inv)
		smooth_clamped_motor_impulses = torch.clip(motor_impulses, 0, 1) # note: not smooth clamping
		smooth_clamped_u_gravity_comp = smooth_clamped_motor_impulses @ torch.t(self.torch_mixer)
		u_clipped = smooth_clamped_u_gravity_comp - torch.tensor([self.M * g, 0, 0, 0])[None]

		f_cl = self.xdot_torch_function(x[None], u_clipped)[0]
		return f_cl

	def compute_flow_with_gradients(self, x0):
		"""
        Computes flow and flow gradients at all timesteps
        :param x0: (1, 16)
        :return:
        """
		tsteps = int(self.args.flow_T / self.args.flow_dt)
		xi_list = [x0]
		Q0 = np.eye(16)
		Qi_list = [Q0]

		xi = x0
		Qi = Q0

		dt = self.args.flow_dt

		# f_cl_torch = lambda x: self.xdot_torch_function(x, self.backup_K@x)

		# TODO: just using finite differencing
		for i in range(tsteps):
			print("flow %i" % i)
			# For integration, simply using finite time differencing (first order approx)
			ui = self.compute_backup_control(xi)
			xi = xi + dt * self.flow_env.x_dot_open_loop(xi, ui)

			xi_torch = numpy_to_torch(xi.flatten())
			J_cl_torch = jacobian(self.f_cl_torch, xi_torch)
			J_cl = J_cl_torch.detach().cpu().numpy()
			# TODO: detach? dim?
			dQi = J_cl @ Qi
			# dQi = J_cl
			Qi = Qi + dt * dQi

			xi_list.append(xi)
			Qi_list.append(Qi)
			# print(xi, Qi)
			# print(dQi)
			# print(np.mean(dQi), np.std(dQi), np.min(dQi), np.max(dQi))
			print(np.linalg.norm(xi), np.linalg.norm(Qi))

		xi_array = np.array(xi_list)
		Qi_array = np.array(Qi_list)

		xi_array = np.squeeze(xi_array) # remove dimensions with size 1
		Qi_array = np.squeeze(Qi_array)
		# print("inside compute_flow_with_gradients")
		# IPython.embed()

		# print(np.sum(np.array(xi_array)))

		return xi_array, Qi_array

	def compute_u_ref(self, t, x):
		"""
        :param t:
        :param x: (1, 16)
        :return: (1, 4)
        """
		if self.args.rollout_u_ref == "unactuated":
			u = np.zeros(self.u_dim)
		elif self.args.rollout_u_ref == "LQR":
			u = - self.ref_K @ x.flatten()
		u = np.reshape(u, (1, self.u_dim))
		return u

	def check_outside(self, x):
		"""
        Check if x is inside or outside of the defined viable set
        :param x:
        :return:
        """
		# print("Check controller's check_outside")

		xi_array, _ = self.compute_flow_with_gradients(x)

		# print("Check controller's check_outside")
		# IPython.embed()
		xT = xi_array[-1][None]
		xT_torch = numpy_to_torch(xT)
		xi_array_torch = numpy_to_torch(xi_array)
		xT_in_compact_viable_set = (self.hb_torch_function(xT_torch) < 0).item()
		xi_all_safe = torch.all(self.h_torch_function(xi_array_torch) < 0).item()

		inside = xT_in_compact_viable_set * xi_all_safe
		outside = not inside
		return outside

	def solve_QP(self, x, eps, u_ref):
		# print("inside solve QP")
		# print("There are going to be a lot of dimension errors")
		# IPython.embed()

		xi_array, Qi_array = self.compute_flow_with_gradients(x)
		x_torch = numpy_to_torch(x)
		x_torch.requires_grad = True
		hb_grad = grad([self.hb_torch_function(x_torch)], x_torch)[0].detach().cpu().numpy()
		h_grad = grad([self.h_torch_function(x_torch)], x_torch)[0].detach().cpu().numpy()

		f_x = self.sim_env._f_model(x)
		f_x = np.reshape(f_x, (16, 1))
		g_x = self.sim_env._g_model(x)

		tsteps = len(xi_array)  # TODO: off by 1?
		n_input_cons = tsteps + 1  # TODO: number of input constraints
		lhs_matrix = np.zeros((n_input_cons, 4))
		rhs_matrix = np.zeros((n_input_cons))

		lhs_matrix[0, :] = np.squeeze(hb_grad @ Qi_array[-1] @ g_x)
		rhs_matrix[0] = np.squeeze(-hb_grad @ Qi_array[-1] @ f_x - eps)

		lhs_matrix[1:, :] = np.squeeze(h_grad @ Qi_array @ g_x)
		rhs_matrix[1:] = np.squeeze(-h_grad @ Qi_array @ f_x - eps)

		# Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
		w = 1000.0  # slack weight

		n_qp_var = 4 + 4 + n_input_cons  # U, u, and input cons slack
		P = np.zeros((n_qp_var, n_qp_var))
		P[:4, :4] = 2 * np.eye(4)
		q = np.zeros((n_qp_var, 1))
		q[:4, 0] = -2 * u_ref
		q[8:, 0] = w  # assume slacks are positive

		# G <= rho
		n_qp_ineq = n_input_cons * 2 + 4 * 2
		G = np.zeros((n_qp_ineq, n_qp_var))
		rho = np.zeros((n_qp_ineq, 1))
		# Input constraints
		G[:n_input_cons, :4] = lhs_matrix
		G[:n_input_cons, -n_input_cons:] = -np.eye(n_input_cons)  # adding slack to input constraint
		rho[:n_input_cons, 0] = rhs_matrix
		# u in [0, 1]
		G[n_input_cons:n_input_cons + 4, 4:8] = -np.eye(4)
		G[n_input_cons + 4:n_input_cons + 8, 4:8] = np.eye(4)
		rho[n_input_cons + 4:n_input_cons + 8, 0] = 1.0
		# slack > 0
		G[-n_input_cons:, -n_input_cons:] = -np.eye(n_input_cons)

		# A = b
		# U = M@u - [Mg, 0, 0, 0]
		A = np.zeros((4, n_qp_var))
		A[:, :4] = -np.eye(4)
		A[:, 4:8] = self.mixer
		b = np.array([self.M * g, 0, 0, 0])[:, None]

		try:
			sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(rho), matrix(A), matrix(b))
			print("qp_sol_status", sol_obj["status"])
		except:
			# IPython.embed()
			print("QP solve was unsuccessful, with status: %s " % sol_obj["status"])
			print("Go to line 96 in flying_cbf_controller")
			IPython.embed()
			print("exiting")
			exit(0)

		sol_var = np.array(sol_obj['x'])
		U = sol_var[0:4]
		U = np.reshape(U, (4))
		qp_slack = sol_var[-n_input_cons:]

		# print("Slack: %.6f" % qp_slack)
		# print(sol_impulses, u_safe, qp_slack)
		impulses = sol_var[4:8]
		debug_dict = {"qp_slack": qp_slack, "qp_sol_status": sol_obj["status"]}

		return U, debug_dict

	def compute_control(self, t, x):
		"""
        :param t:
        :param x: (1, 16)
        :return: U (1, 4)
        """
		# print("inside controller's compute_control()")
		# IPython.embed()

		###################################
		# print(x.shape)
		u_ref = self.compute_u_ref(t, x)
		# print(u_ref.shape)
		x_next = x + self.sim_env.dt * self.sim_env.x_dot_open_loop_model(x,
		                                                                  u_ref)  # in the absence of safe control, the next state
		# print(x_next.shape)

		x_outside_viable = self.check_outside(x)
		x_next_outside_viable = self.check_outside(x_next)

		# Default values
		apply_u_safe = False
		u_ref = self.compute_u_ref(t, x)
		qp_slack = None
		inside_boundary = False
		on_boundary = False
		outside_boundary = False

		if x_outside_viable:  # Outside
			print("Outside safe set")
			eps = self.eps_outside
			apply_u_safe = True
			outside_boundary = True

			U, qp_debug_dict = self.solve_QP(x, eps, u_ref)
			qp_slack = qp_debug_dict["qp_slack"]  # TODO: add
		elif (not x_outside_viable) and x_next_outside_viable:  # On boundary. Note: cheating way to convert DT to CT
			print("On safe set boundary")
			eps = self.eps_bdry
			apply_u_safe = True
			on_boundary = True

			U, qp_debug_dict = self.solve_QP(x, eps, u_ref)
			qp_slack = qp_debug_dict["qp_slack"]  # TODO: add
		else:  # Inside
			print("Inside safe set")
			apply_u_safe = False
			inside_boundary = True

		debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack,
		              "inside_boundary": inside_boundary, "on_boundary": on_boundary,
		              "outside_boundary": outside_boundary}

		# IPython.embed()
		return U, debug_dict


# def phi_0(x):
# 	"""
# 	:param x: (None, 16, 1)
# 	:return:
# 	"""
# 	theta = x[:, [state_index_dict["theta"]]]
# 	phi = x[:, [state_index_dict["phi"]]]
# 	gamma = x[:, [state_index_dict["gamma"]]]
# 	beta = x[:, [state_index_dict["beta"]]]
# 	delta_safety_limit = param_dict["delta_safety_limit"]
#
# 	cos_cos = np.cos(theta) * np.cos(phi)
# 	eps = 1e-4  # prevents nan when cos_cos = +/- 1 (at x = 0)
# 	signed_eps = -np.sign(cos_cos) * eps
# 	delta = np.arccos(cos_cos + signed_eps)
# 	rv = delta ** 2 + gamma ** 2 + beta ** 2 - delta_safety_limit ** 2
# 	return rv

# def extract_statistics(info_dicts):
# 	# TODO: any printing for logging should be done in this function
# 	# print(info_dicts['n_cons_viol'] )
#
# 	# print("Inside extract statistics")
# 	# IPython.embed()
# 	"""['obj_pred', 'U_pred', 'U_ref_pred', 'x_pred', 'u_pred', 'name_of_lb_viol', 'name_of_ub_viol', 'n_cons_viol', 'x',
# 	 'U', 'stat_dict']"""
#
# 	stat_dict = {}
# 	rollouts_x = info_dicts["x"]
# 	rollouts_U = info_dicts["U"]
# 	rollouts_obj_pred = info_dicts["obj_pred"]
# 	rollouts_U_pred = info_dicts["U_pred"]
# 	rollouts_U_ref_pred = info_dicts["U_ref_pred"]
# 	rollouts_x_pred = info_dicts["x_pred"]
# 	rollouts_u_pred = info_dicts["u_pred"]
# 	N_rollouts = len(rollouts_x)
#
# 	# Main statistic
# 	phi0_tol = 1e-2
# 	rollouts_which_unsafe = np.array([np.any(phi_0(rollout_x) > phi0_tol) for rollout_x in rollouts_x])
# 	rollouts_with_x0_safe = np.array([(phi_0(rollout_x)[0] <= phi0_tol) for rollout_x in rollouts_x]).flatten()
#
# 	# rollouts_phi0 = np.array([phi_0(rollout_x) for rollout_x in rollouts_x])
# 	rollouts_percent_unsafe = np.sum(rollouts_which_unsafe*rollouts_with_x0_safe)*100/N_rollouts
# 	rollouts_percent_safe = 100 - rollouts_percent_unsafe
#
# 	rollouts_which_safe = np.logical_not(rollouts_which_unsafe*rollouts_with_x0_safe)
# 	stat_dict["rollouts_which_safe"] = rollouts_which_safe
# 	stat_dict["rollouts_percent_safe"] = rollouts_percent_safe
# 	print("%.3f %% of rollouts safe" % rollouts_percent_safe)
#
# 	# Debugging
# 	unsafe_ind = np.argwhere(np.logical_not(rollouts_which_safe)).flatten()
# 	print("%i number of rollouts unsafe" % unsafe_ind.size)
#
# 	# Did MPC become infeasible at any point? And which constraints became infeasible?
# 	rollouts_with_infeas = np.sum(np.array(info_dicts["n_cons_viol"]), axis=1) > 0
# 	ind_unsafe_rollouts = np.argwhere(np.logical_not(rollouts_which_safe)).flatten()
# 	percent_unsafe_with_infeas = np.mean(rollouts_with_infeas[ind_unsafe_rollouts])*100
# 	percent_with_infeas = np.mean(rollouts_with_infeas)*100
# 	stat_dict["percent_unsafe_with_infeas"] = percent_unsafe_with_infeas
# 	stat_dict["percent_with_infeas"] = percent_with_infeas
#
# 	print("%.3f %% of rollouts have constraint violations" % percent_with_infeas)
# 	print("%.3f %% of unsafe rollouts have constraint violations" % percent_unsafe_with_infeas)
#
# 	# TODO: another possible reason for unsafe rollouts is that the initial state is unsafe
# 	# initial states were computed with a slightly diff MPC formulation than the one being used for rollouts
# 	# IPython.embed()
#
# 	return stat_dict

def extract_statistics(info_dicts):
	print("Inside extract_statistics")
	IPython.embed()

	stat_dict = {}
	x_lim = param_dict["x_lim"]

	# How many rollouts applied safe control before they were terminated?
	apply_u_safe = info_dicts["apply_u_safe"]  # list of len N_rollout; elem is array of size T_max
	N_rollout = len(apply_u_safe)

	# Count transitions
	inside = info_dicts["inside_boundary"]  # (n_rollouts, t_max)?
	on = info_dicts["on_boundary"]
	outside = info_dicts["outside_boundary"]

	on_in_rl = [on[i][:-1] * inside[i][1:] for i in range(N_rollout)]
	on_out_rl = [on[i][:-1] * outside[i][1:] for i in range(N_rollout)]
	on_on_rl = [on[i][:-1] * on[i][1:] for i in range(N_rollout)]

	on_in_count = np.sum([np.sum(rl) for rl in on_in_rl])
	on_out_count = np.sum([np.sum(rl) for rl in on_out_rl])
	on_on_count = np.sum([np.sum(rl) for rl in on_on_rl])

	# total_transitions = on_in_count + on_out_count # TODO
	total_transitions = on_in_count + on_out_count + on_on_count
	print("Total transitions, N rollout: ", total_transitions, N_rollout)
	# Note: If we're counting on-on transitions, a single rollout can have multiple transitions.
	# If not, then should have total_transitions == N_rollout
	# TODO: but then, which transition counts?
	# TOOD: perhaps we should only consider if the rollout ultimately ends up outside or inside...
	# if total_transitions != N_desired_rollout:
	# 	IPython.embed()

	stat_dict["N_transitions"] = total_transitions
	stat_dict["percent_on_in"] = (on_in_count / float(total_transitions)) * 100
	stat_dict["percent_on_out"] = (on_out_count / float(total_transitions)) * 100
	stat_dict["percent_on_on"] = (on_on_count / float(total_transitions)) * 100  # TODO
	# IPython.embed()

	stat_dict["N_on_in"] = on_in_count
	stat_dict["N_on_out"] = on_out_count
	stat_dict["N_on_on"] = on_on_count

	# Debug: how large is the gap? In terms of phi
	# IPython.embed()
	"""phis = [rl[:, -1] for rl in info_dicts["phi_vals"]]
    gap_phis = [phis[i]*on[i] for i in range(N_rollout)]
    min_phi = np.min([np.min(rl) for rl in gap_phis])
    mean_phi = np.sum([np.sum(rl) for rl in gap_phis])/np.sum([np.sum(rl) for rl in on])
    stat_dict["min_phi"] = min_phi
    stat_dict["mean_phi"] = mean_phi
    # IPython.embed()

    # Debug: why is the gap large? Hypothesis 1: dynamics are extreme
    dist_between_xs = info_dicts["dist_between_xs"]
    dist_between_xs_on_gap = [dist_between_xs[i]*on[i] for i in range(N_rollout)]
    mean_dist = np.sum([np.sum(rl) for rl in dist_between_xs_on_gap])/np.sum([np.sum(rl) for rl in on])
    max_dist = np.max([np.max(rl) for rl in dist_between_xs_on_gap])
    stat_dict["mean_dist"] = mean_dist
    stat_dict["max_dist"] = max_dist

    # Debug: why is the gap large? Hypothesis 2: phi steep near border
    phi_grad_mag = info_dicts["phi_grad_mag"]
    phi_grad_gap = [phi_grad_mag[i]*on[i] for i in range(N_rollout)]
    mean_phi_grad = np.sum([np.sum(rl) for rl in phi_grad_gap])/np.sum([np.sum(rl) for rl in on])
    max_phi_grad = np.max([np.max(rl) for rl in phi_grad_gap])
    stat_dict["mean_phi_grad"] = mean_phi_grad
    stat_dict["max_phi_grad"] = max_phi_grad"""

	# Debug: what do the large magnitude gradients look like?
	# Show me the state and corresponding gradient
	# TODO
	# phi_grad = info_dicts["phi_grad"]
	# xs = info_dicts["x"]
	# IPython.embed()

	# Find out box exits
	xs = info_dicts["x"]
	outside_box_rl = [
		np.logical_or(np.any(rl[:, :10] < x_lim[:, 0][None], axis=1), np.any(rl[:, :10] > x_lim[:, 1][None], axis=1))
		for rl in xs]
	on_in_outside_box_rl = [outside_box_rl[i][:-2] * on_in_rl[i] for i in
	                        range(N_rollout)]  # checks if x_i in x_i --> x_f transition is outside. This from [:-2]
	on_out_outside_box_rl = [outside_box_rl[i][:-2] * on_out_rl[i] for i in range(N_rollout)]
	on_on_outside_box_rl = [outside_box_rl[i][:-2] * on_on_rl[i] for i in range(N_rollout)]
	# IPython.embed()

	on_in_outside_box_count = np.sum([np.sum(rl) for rl in on_in_outside_box_rl])  # TODO: is any the right thing here?
	on_out_outside_box_count = np.sum([np.sum(rl) for rl in on_out_outside_box_rl])
	on_on_outside_box_count = np.sum([np.sum(rl) for rl in on_on_outside_box_rl])

	stat_dict["percent_on_in_outside_box"] = (on_in_outside_box_count / float(max(1, on_in_count))) * 100
	stat_dict["percent_on_out_outside_box"] = (on_out_outside_box_count / float(max(1, on_out_count))) * 100
	stat_dict["percent_on_on_outside_box"] = (on_on_outside_box_count / float(max(1, on_on_count))) * 100
	# IPython.embed()

	stat_dict["N_on_in_outside_box"] = on_in_outside_box_count
	stat_dict["N_on_out_outside_box"] = on_out_outside_box_count
	stat_dict["N_on_on_outside_box"] = on_on_outside_box_count

	# Debug box exits
	# Which states are we exiting on?
	outside_box_states = [np.logical_or(rl[:, :10] < x_lim[:, 0][None], rl[:, :10] > x_lim[:, 1][None]) for rl in xs]

	state_ind = [np.argwhere(rl)[:, 1] for rl in outside_box_states]
	state_ind = np.concatenate(state_ind)
	values, counts = np.unique(state_ind, return_counts=True)

	state_index_dict = param_dict['state_index_dict']
	state_index_value_to_key_dict = dict(zip(state_index_dict.values(), state_index_dict.keys()))
	for i in range(len(values)):
		# stat_dict["N_count_exit_on_%i" % values[i]] = counts[i]
		stat_dict["N_count_exit_on_%s" % state_index_value_to_key_dict[values[i]]] = counts[i]

	# which rollouts exited?
	# outside_box_any_rl = [np.sum(rl) for rl in outside_box_rl]
	# which_rl = np.argwhere(outside_box_any_rl).flatten()

	# Compute how much things are violated
	# print("at the end of compute_stats")
	# IPython.embed()

	qp_slacks = info_dicts["qp_slack"]
	violation_amounts = []
	for i in range(N_rollout):
		if np.any(on_out_rl[i]):
			qp_slack = qp_slacks[i][:-1]
			qp_slack[qp_slack == None] = 0
			amount = np.sum(on_out_rl[i] * qp_slack)
			violation_amounts.append(amount)
	# violation_amounts = [on_out_rl[i]*qp_slacks[i][:-1] for i in range(N_rollout)]
	# mean_violation_amount = np.mean(violation_amounts) + cbf_controller.eps_bdry # TODO: note the difference between this and below
	mean_violation_amount = np.mean(violation_amounts)
	std_violation_amount = np.std(violation_amounts)
	stat_dict["mean_violation_amount"] = mean_violation_amount
	stat_dict["std_violation_amount"] = std_violation_amount

	return stat_dict


"""
    parser.add_argument('--flow_dt', type=float, default=0.05, help="dt for discretized computation of the dynamics flow")
    parser.add_argument('--flow_T', type=float, default=1.0, help="length (in total time) of dynamics flow")
    parser.add_argument('--backup_LQR_q', type=float, default=1.0, help="assume backup controller is LQR")
    parser.add_argument('--backup_LQR_r', type=float, default=1.0, help="assume backup controller is LQR")

    # Plotting
    parser.add_argument('--delta', type=float, default=0.1, help="discretization of grid over slice")
    parser.add_argument('--which_params', default=["phi", "dphi"], nargs='+', type=str, help="which 2 state variables")

    # Rollout
    # parser.add_argument('--rollout_N_rollout', type=int, default=5) # TODO: deprecated
    parser.add_argument('--rollout_T_max', type=float, default=2.5)
    parser.add_argument('--rollout_load_x0_fnm', type=str, help='If you want to use saved, precomputed x0.')
    # parser.add_argument('--rollout_mpc_set_initial_guess', action='store_true', help='set initial guess for MPC variables (all u, all x)') # TODO: do this on default
    parser.add_argument('--rollout_u_ref', type=str, choices=["unactuated", "LQR"], default="unactuated")
    parser.add_argument('--rollout_LQR_q', type=float, default=0.1)
    parser.add_argument('--rollout_LQR_r', type=float, default=1.0)
"""


def make_backup_set_rollouts_save_folder(args):
	# print("Check out: make_backup_set_rollouts_save_folder")
	# IPython.embed()
	save_parent_folder = "./flying_backup_set_outputs/rollouts_loading_%s" % (args.rollout_load_x0_fnm)
	if not os.path.exists(save_parent_folder):
		os.makedirs(save_parent_folder)

	N_steps_max = math.ceil(args.rollout_T_max / args.dt)
	save_folder = "dt_%.2f_length_%i" % (args.dt, N_steps_max)

	flow_N_steps = math.ceil(args.flow_T / args.flow_dt)
	save_folder += "_flow_dt_%.2f_length_%i_backup_LQR_q_%.2f_r_%.2f" % (
		args.flow_dt, flow_N_steps, args.backup_LQR_q, args.backup_LQR_r)

	if args.rollout_u_ref == "unactuated":
		save_folder += "_u_ref_unact"
	elif args.rollout_u_ref == "LQR":
		save_folder += "_u_ref_LQR_q_%.2f_r_%.2f" % (args.rollout_LQR_q, args.rollout_LQR_r)

	# if args.rollout_mpc_set_initial_guess:
	#     save_folder += "_set_init_guess"

	# save_folder += ".pkl"
	save_fpth = save_parent_folder + "/" + save_folder

	if not os.path.exists(save_fpth):
		os.makedirs(save_fpth)

	print("Rollout data will be saved at folder: %s" % save_fpth)
	# IPython.embed()
	return save_fpth


def sample_unif_in_hyperellipsoid(S, z_hat, m_FA, Gamma_Threshold=1.0):
	"""
    :param S:
    :param z_hat:
    :param m_FA:
    :param Gamma_Threshold:
    :return: (nz, m_FA)
    """
	nz = S.shape[0]
	z_hat = z_hat.reshape(nz, 1)

	X_Cnz = np.random.normal(size=(nz, m_FA))

	rss_array = np.sqrt(np.sum(np.square(X_Cnz), axis=0))
	kron_prod = np.kron(np.ones((nz, 1)), rss_array)

	X_Cnz = X_Cnz / kron_prod  # Points uniformly distributed on hypersphere surface

	R = np.ones((nz, 1)) * (np.power(np.random.rand(1, m_FA), (1. / nz)))

	unif_sph = R * X_Cnz;  # m_FA points within the hypersphere
	T = np.asmatrix(cholesky(S))  # Cholesky factorization of S => S=T’T

	# IPython.embed()
	unif_ell = T.H * unif_sph;  # Hypersphere to hyperellipsoid mapping

	# Translation and scaling about the center
	z_fa = (unif_ell * np.sqrt(Gamma_Threshold) + (z_hat * np.ones((1, m_FA))))

	return np.array(z_fa)


if __name__ == "__main__":
	# IPython.embed()
	# import pickle
	# # info_dicts = pickle.load(open("flying_mpc_outputs/rollouts_loading_volume_approx_target_N_samp_inside_5000/dt_0.05000_length_200_u_ref_LQR_q_0.10_r_1.00/rollouts.pkl", "rb"))
	# # fpth = "flying_mpc_outputs/rollouts_loading_volume_approx_target_N_samp_inside_5000/dt_0.05000_length_200_u_ref_LQR_q_0.10_r_1.00/rollouts.pkl"
	# # fpth = "flying_mpc_outputs/rollouts_loading_volume_approx_target_N_samp_inside_5000/dt_0.05000_length_200_u_ref_LQR_q_1.00_r_1.00/rollouts.pkl"
	# fpth = "flying_mpc_outputs/rollouts_loading_volume_approx_target_N_samp_inside_5000/dt_0.05000_length_200_u_ref_unact/rollouts.pkl"
	# print(fpth)
	# info_dicts = pickle.load(open(fpth, "rb"))
	# stat_dict = extract_statistics(info_dicts)
	# # info_dicts["stat_dict"] = stat_dict
	# # with open(fpth, 'wb') as handle:
	# # 	pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Test hyperellipsoid sampling and see if uniform
	# IPython.embed()

	P = np.array([[-1, 1], [1, 1]])  # cartesian to ellipsoid
	# P = np.random.rand(2, 2)
	P = P * (1.0 / np.linalg.norm(P, axis=1))
	D = np.diag([1, 0.25])
	P_inv = np.linalg.inv(P)
	S = P @ D @ P_inv
	z_hat = np.zeros(2)
	m_FA = 500
	samples = sample_unif_in_hyperellipsoid(S, z_hat, m_FA, Gamma_Threshold=1.0)

	import matplotlib.pyplot as plt

	# IPython.embed()
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(samples[0, :], samples[1, :])
	ax.set_aspect("equal")
	plt.savefig("debug/test_hyperellip_sample.png")
# ax.imshow(np.logical_not(S_grid), extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
