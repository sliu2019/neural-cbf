import do_mpc
import IPython
import numpy as np, math
from casadi import *
from copy import deepcopy
import os
import control

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

# def compute_flying_inv_pend_LQR_matrix(args):
# 	L_p = param_dict["L_p"]
# 	M = param_dict["M"]
# 	J_x = param_dict["J_x"]
# 	J_y = param_dict["J_y"]
# 	J_z = param_dict["J_z"]
#
# 	A = np.zeros((10, 10))  # 10 x 10
# 	A[0:3, 3:6] = np.eye(3)
# 	A[6:8, 8:10] = np.eye(2)
# 	A[8, 0] = -3 * g / (2 * L_p)
# 	A[9, 1] = -3 * g / (2 * L_p)
# 	A[8, 6] = 3 * g / (2 * L_p)
# 	A[9, 7] = 3 * g / (2 * L_p)
#
# 	B = np.zeros((10, 4))
# 	B[3:6, 1:4] = np.diag([1.0 / J_x, 1.0 / J_y, 1.0 / J_z])
#
# 	# Use LQR to compute feedback portion of controller
# 	q = args.rollout_LQR_q
# 	r = args.rollout_LQR_r
# 	Q = q * np.eye(10)
# 	R = r * np.eye(4)
# 	K, S, E = control.lqr(A, B, Q, R)
#
# 	return K

def compute_flying_inv_pend_LQR_matrix(args):
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
	A[10:13, 13:16] = np.eye(3) # x,y,z
	A[13, 1] = g
	A[14, 0] = -g

	B = np.zeros((x_dim, 4))
	B[3:6, 1:4] = np.diag([1.0 / J_x, 1.0 / J_y, 1.0 / J_z])
	B[15, 0] = 1/M

	# print(A)
	# print(B)
	# print("double-check the above")
	# IPython.embed()
	# Use LQR to compute feedback portion of controller
	q = args.rollout_LQR_q
	r = args.rollout_LQR_r
	Q = q * np.eye(x_dim)
	R = r * np.eye(4)
	K, S, E = control.lqr(A, B, Q, R)

	# print(K)
	# print("sanity-check the above, if it's possible to understand it")
	# IPython.embed()
	return K

def setup_solver(args):
	# print(param_dict)
	N_horizon = args.N_horizon
	dt = args.dt

	# model_type = 'continuous'  # either 'discrete' or 'continuous'
	model_type = 'discrete'
	model = do_mpc.model.Model(model_type)

	# Define state vars
	# Have to be created in this order: state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta"]

	gamma = model.set_variable(var_type='_x', var_name='gamma', shape=(1, 1))
	beta = model.set_variable(var_type='_x', var_name='beta', shape=(1, 1))
	alpha = model.set_variable(var_type='_x', var_name='alpha', shape=(1, 1))

	dgamma = model.set_variable(var_type='_x', var_name='dgamma', shape=(1, 1))
	dbeta = model.set_variable(var_type='_x', var_name='dbeta', shape=(1, 1))
	dalpha = model.set_variable(var_type='_x', var_name='dalpha', shape=(1, 1))

	phi = model.set_variable(var_type='_x', var_name='phi', shape=(1, 1))
	theta = model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))

	dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(1, 1))
	dtheta = model.set_variable(var_type='_x', var_name='dtheta', shape=(1, 1))

	px = model.set_variable(var_type='_x', var_name='px', shape=(1, 1))
	py = model.set_variable(var_type='_x', var_name='py', shape=(1, 1))
	pz = model.set_variable(var_type='_x', var_name='pz', shape=(1, 1))

	dpx = model.set_variable(var_type='_x', var_name='dpx', shape=(1, 1))
	dpy = model.set_variable(var_type='_x', var_name='dpy', shape=(1, 1))
	dpz = model.set_variable(var_type='_x', var_name='dpz', shape=(1, 1))

	# Define input
	u = model.set_variable(var_type='_u', var_name='u', shape=(4, 1))  # TODO: note this is lower-level controls!!!

	# Define dynamics
	k1 = param_dict["k1"]
	k2 = param_dict["k2"]
	l = param_dict["l"]
	M = param_dict["M"]
	J_x = param_dict["J_x"]
	J_y = param_dict["J_y"]
	J_z = param_dict["J_z"]
	L_p = param_dict["L_p"]
	delta_safety_limit = param_dict["delta_safety_limit"]

	Mixer = SX([[k1, k1, k1, k1], [0, -l * k1, 0, l * k1], [l * k1, 0, -l * k1, 0], [-k2, k2, -k2, k2]])  # mixer matrix

	R = SX.zeros(3, 3)
	R[0, 0] = cos(alpha) * cos(beta)
	R[0, 1] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)
	R[0, 2] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)
	R[1, 0] = sin(alpha) * cos(beta)
	R[1, 1] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma)
	R[1, 2] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)
	R[2, 0] = -sin(beta)
	R[2, 1] = cos(beta) * sin(gamma)
	R[2, 2] = cos(beta) * cos(gamma)

	k_x = R[0, 2]
	k_y = R[1, 2]
	k_z = R[2, 2]

	u_high = Mixer @ u  # TODO: high-level control inputs, (4x1)
	F = u_high[0, 0] # TODO: No +/- M*g is correct

	###### Computing state derivatives
	J = SX([[J_x], [J_y], [J_z]])
	norm_torques = u_high[1:] * (1.0 / J)

	ddquad_angles = R @ norm_torques
	ddgamma = ddquad_angles[0, 0]
	ddbeta = ddquad_angles[1, 0]
	ddalpha = ddquad_angles[2, 0]

	ddphi = ((3.0 * (k_y * cos(phi) + k_z * sin(phi)) * F) / (
			2 * M * L_p * cos(theta))) + (2 * dtheta * dphi * tan(theta))
	ddtheta = ((3.0 * (
			-k_x * cos(theta) - k_y * sin(phi) * sin(theta) + k_z * cos(phi) * sin(
		theta)) * F) / (2.0 * M * L_p)) - (power(dphi,2) * sin(theta) * cos(theta))

	ddpx = (k_x/M)*F
	ddpy = (k_y/M)*F
	ddpz = (k_z/M)*F - g

	# TODO: Using discrete (not continuous) model
	gamma_next = gamma + dt*dgamma
	beta_next = beta + dt*dbeta
	alpha_next = alpha + dt*dalpha

	dgamma_next = dgamma + dt*ddgamma
	dbeta_next = dbeta + dt*ddbeta
	dalpha_next = dalpha + dt*ddalpha

	phi_next = phi + dt*dphi
	theta_next = theta + dt*dtheta

	dphi_next = dphi + dt*ddphi
	dtheta_next = dtheta + dt*ddtheta

	px_next = px + dt*dpx
	py_next = py + dt*dpy
	pz_next = pz + dt*dpz

	dpx_next = dpx + dt*ddpx
	dpy_next = dpy + dt*ddpy
	dpz_next = dpz + dt*ddpz

	model.set_rhs('gamma', gamma_next)
	model.set_rhs('beta', beta_next)
	model.set_rhs('alpha', alpha_next)

	model.set_rhs('dgamma', dgamma_next)
	model.set_rhs('dbeta', dbeta_next)
	model.set_rhs('dalpha', dalpha_next)

	model.set_rhs('theta', theta_next)
	model.set_rhs('phi', phi_next)

	model.set_rhs('dtheta', dtheta_next)
	model.set_rhs('dphi', dphi_next)

	model.set_rhs('px', px_next)
	model.set_rhs('py', py_next)
	model.set_rhs('pz', pz_next)

	model.set_rhs('dpx', dpx_next)
	model.set_rhs('dpy', dpy_next)
	model.set_rhs('dpz', dpz_next)

	# Set aux expressions
	# Can either be queried as model.aux['expr_name'] for the symbolic expression or mpc.data["_opt_aux_num"]
	if args.rollout_u_ref == "unactuated":
		U_ref = SX.zeros((4, 1))
	elif args.rollout_u_ref == "LQR":
		K = compute_flying_inv_pend_LQR_matrix(args)
		x = SX.zeros(16, 1)
		# x_const = np.zeros((10, 1)) # can't mix them like this
		x[0, 0] = gamma
		x[1, 0] = beta
		x[2, 0] = alpha
		x[3, 0] = dgamma
		x[4, 0] = dbeta
		x[5, 0] = dalpha
		x[6, 0] = phi
		x[7, 0] = theta
		x[8, 0] = dphi
		x[9, 0] = dtheta
		x[10, 0] = px
		x[11, 0] = py
		x[12, 0] = pz
		x[13, 0] = dpx
		x[14, 0] = dpy
		x[15, 0] = dpz
		# x = SX(x)
		U_ref = -K@x

	u_high = Mixer@u # need to define everything as a function of u below model.setup() function
	U = u_high - SX([[param_dict["M"]*g], [0], [0], [0]])
	obj = norm_2(U - U_ref) ** 2

	# Expressions are recorded in the order you defined them
	model.set_expression("obj", obj)
	model.set_expression("U", U)
	model.set_expression("U_ref", U_ref)

	# Finally,
	model.setup()

	#######################
	# print("inside setup_solver, ln 176")
	# IPython.embed()

	# Create optimizer
	mpc = do_mpc.controller.MPC(model)

	# Last entry suppresses IPOPT output
	# 'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0},
	setup_mpc = {
		'n_horizon': N_horizon,
		't_step': dt,
		'store_full_solution': True,
		'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
	}

	mpc.set_param(**setup_mpc)

	# Hard safety constraint
	cons_name = "safety"
	gamma_int = arctan2(sin(gamma), cos(gamma)) # mod to [-pi, pi] interval
	beta_int = arctan2(sin(beta), cos(beta))
	phi_int = arctan2(sin(phi), cos(phi))
	theta_int = arctan2(sin(theta), cos(theta))
	cos_cos = cos(theta_int) * cos(phi_int)
	eps = 1e-4  # prevents nan when cos_cos = +/- 1 (at x = 0)
	signed_eps = -sign(cos_cos) * eps
	delta = acos(cos_cos + signed_eps)
	cons_expr = delta ** 2 + gamma_int ** 2 + beta_int ** 2 - delta_safety_limit ** 2
	mpc.set_nl_cons(cons_name, cons_expr, ub=0) # constraint is cons_expr <= ub

	# TODO: to convert to soft constraint, fill out the last 3 parameters
	# set_nl_cons(self, expr_name, expr, ub=inf, soft_constraint=False, penalty_term_cons=1, maximum_violation=inf)
	# TODO: also check out the mpc.set_param() argument:
	# :param nl_cons_single_slack: If ``True``, soft-constraints set with :py:func:`set_nl_cons` introduce only a single slack variable for the entire horizon. Defaults to ``False``.
	#         :type nl_cons_single_slack: bool
	# TODO: to check the slack variable
	# # slack variables for soft constraints:
	# opt_x_num['_eps', time_step, scenario, _nl_cons_name]

	# SaturationRisk: |u-uref|
	if args.rollout_u_ref == "unactuated":
		U_ref = SX.zeros((4, 1))
	elif args.rollout_u_ref == "LQR":
		K = compute_flying_inv_pend_LQR_matrix(args)
		x = SX.zeros(16, 1)
		# x_const = np.zeros((10, 1)) # can't mix them like this
		x[0, 0] = gamma
		x[1, 0] = beta
		x[2, 0] = alpha
		x[3, 0] = dgamma
		x[4, 0] = dbeta
		x[5, 0] = dalpha
		x[6, 0] = phi
		x[7, 0] = theta
		x[8, 0] = dphi
		x[9, 0] = dtheta
		x[10, 0] = px
		x[11, 0] = py
		x[12, 0] = pz
		x[13, 0] = dpx
		x[14, 0] = dpy
		x[15, 0] = dpz
		# x = SX(x)
		U_ref = -K@x

	u_high = Mixer@u # need to define everything as a function of u below model.setup() function
	U = u_high - SX([[param_dict["M"]*g], [0], [0], [0]])
	# model.set_meas("U_ref", meas_noise=False)

	obj = norm_2(U - U_ref) ** 2
	end_state_term = SX.zeros((1))
	mpc.set_objective(lterm=obj, mterm=end_state_term)

	# mpc.set_rterm(u=1.0)

	# No state limits
	# Control limits:
	mpc.bounds['lower', '_u', 'u'] = SX.zeros(4, 1)
	mpc.bounds['upper', '_u', 'u'] = SX.ones(4, 1)

	# Add terminal bounds
	# print("line 205: added scaling")
	# IPython.embed()
	x_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta"]

	# TODO: notice that no terminal constraint on x, y, z
	eps = 1e-1
	if not args.mpc_no_terminal_safety:
		print("Adding terminal bounds")
		for i, x_name in enumerate(x_names):
			mpc.terminal_bounds['lower', '_x', x_name] = -eps
			mpc.terminal_bounds['upper', '_x', x_name] = eps

	# Also, scale all variables to [-1, 1] range to improve optimizer performance
	# TODO: would have to reg x, y, z axis to arb zero-cent hyper-rect
	# if args.mpc_scale_states:
	# 	print("Scaling all states to same range")
	# 	for i, x_name in enumerate(x_names):
	# 		x_max = param_dict["x_lim"][i][1]
	# 		print(x_name, x_max)
	# 		mpc.scaling['_x', x_name] = x_max

	# Finally,
	mpc.setup()

	# TODO: will default to x_0 = 0, u_0 = 0
	mpc.set_initial_guess()
	#################################
	return model, mpc

def phi_0(x):
	theta = x[:, [state_index_dict["theta"]]]
	phi = x[:, [state_index_dict["phi"]]]
	gamma = x[:, [state_index_dict["gamma"]]]
	beta = x[:, [state_index_dict["beta"]]]
	delta_safety_limit = param_dict["delta_safety_limit"]

	cos_cos = np.cos(theta) * np.cos(phi)
	eps = 1e-4  # prevents nan when cos_cos = +/- 1 (at x = 0)
	signed_eps = -np.sign(cos_cos) * eps
	delta = np.arccos(cos_cos + signed_eps)
	rv = delta ** 2 + gamma ** 2 + beta ** 2 - delta_safety_limit ** 2
	return rv

def check_solution_found_mpc(mpc):
	soln_found_dict = {}

	# print("Inside check_solution_found_mpc")
	# IPython.embed()

	# Which constraints are violated, if any
	# print("Check that this includes equality constraints, not just inequality")
	lb_bound_violation = mpc.opt_x_num.cat < mpc.lb_opt_x # TODO: this only checks terminal constraint and input limits
	ub_bound_violation = mpc.opt_x_num.cat > mpc.ub_opt_x

	opt_labels = mpc.opt_x.labels()
	labels_lb_viol = np.array(opt_labels)[np.where(lb_bound_violation)[0]]
	labels_ub_viol = np.array(opt_labels)[np.where(ub_bound_violation)[0]]

	soln_found_dict["name_of_lb_viol"] = labels_lb_viol
	soln_found_dict["name_of_ub_viol"] = labels_ub_viol

	# TODO: also need to check nonlinear constraint
	state_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta", "px", "py",
	               "pz", "dpx", "dpy", "dpz"]
	x_pred = [mpc.data.prediction(('_x', state_name)) for state_name in state_names]
	x_pred = np.concatenate(x_pred, axis=0)[:, :, 0].T # (horizon + 1, n_state)

	# IPython.embed()
	phi0_pred = phi_0(x_pred)
	n_safety_viol = np.sum(phi0_pred > 0)
	soln_found_dict["n_safety_viol"] = n_safety_viol

	n_cons_viol = len(labels_lb_viol) + len(labels_ub_viol) + n_safety_viol
	soln_found_dict["n_cons_viol"] = n_cons_viol

	soln_found = (n_cons_viol == 0)

	# IPython.embed()

	return soln_found, soln_found_dict


class MPCController:
	def __init__(self, env, mpc, model, param_dict, args):
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		self.__dict__.update(self.param_dict)  # __dict__ holds and object's attributes

		# pre-compute mixer matrix
		self.mixer = np.array([[self.k1, self.k1, self.k1, self.k1], [0, -self.l * self.k1, 0, self.l * self.k1],
		                       [self.l * self.k1, 0, -self.l * self.k1, 0], [-self.k2, self.k2, -self.k2, self.k2]])

	def collect_debug_data(self):
		# IPython.embed()

		debug_dict = {}

		# Collect MPC prediction (state, input, objective value, U_ref)
		opt_aux_num = self.mpc.data["_opt_aux_num"] # n_rollout_steps x (1+ n_total_aux_expression)
		opt_aux_num = np.reshape(opt_aux_num[-1, :], (-1, 10))

		obj_pred = opt_aux_num[:, 1]
		U_pred = opt_aux_num[:, 2:6]
		U_ref_pred = opt_aux_num[:, 6:]
		# print(obj_pred.shape, U_pred.shape, U_ref_pred.shape)

		debug_dict["obj_pred"] = obj_pred
		debug_dict["U_pred"] = U_pred
		debug_dict["U_ref_pred"] = U_ref_pred

		# What is the MPC prediction (state and input)?
		# state_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta"]
		state_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta", "px", "py", "pz", "dpx", "dpy", "dpz"]
		x_pred = [self.mpc.data.prediction(('_x', state_name)) for state_name in state_names]
		x_pred = np.concatenate(x_pred, axis=0)[:, :, 0]
		debug_dict["x_pred"] = x_pred.T  # (horizon + 1) x 10

		u_pred = self.mpc.data.prediction(('_u', 'u'))  # (4, 10, 1)
		u_pred = u_pred[:, :, 0]
		debug_dict['u_pred'] = u_pred.T  # horizon x 4

		debug_dict['success'] = self.mpc.data.success[-1, 0]

		# print("inside collect_debug_data")
		# IPython.embed()

		return debug_dict

	def compute_control(self, t, x, x_initial_guess=None, u_initial_guess=None):
		"""
		Determines if there exists a control input.
		If so, multiplies by mixer and returns it.
		:param t:
		:param x:
		:return:
		"""
		# print("Inside compute_control, ln 377 in utils")
		# IPython.embed()
		# print("before modifying, before make_step() call")
		# print(self.mpc.opt_x_num['_x'])
		# print(self.mpc.opt_x_num['_u'])
		if x_initial_guess and u_initial_guess:
			copy = deepcopy(self.mpc.opt_x_num)
			copy['_u'] = u_initial_guess
			copy['_x'] = x_initial_guess

			# self.mpc.data.update(_opt_x_num=copy)
			# self.mpc.opt_x_num(copy)
			self.mpc.opt_x_num = copy

			# print("after modifying, before make_step() call")
			# print(self.mpc.opt_x_num['_x'])
			# print(self.mpc.opt_x_num['_u'])

		u = self.mpc.make_step(x)

		# print("after make_step() call")
		# print(self.mpc.opt_x_num['_x'])
		# print(self.mpc.opt_x_num['_u'])
		# IPython.embed()

		soln_found, soln_found_dict = check_solution_found_mpc(self.mpc)
		debug_dict = self.collect_debug_data()

		debug_dict.update(soln_found_dict) # merges dicts

		U = self.mixer @ u - np.array([self.M * g, 0, 0, 0])[:, None]
		U = U.flatten()

		# print("before returning from complete_control")
		# IPython.embed()
		return U, debug_dict

def extract_statistics(info_dicts):
	# TODO: any printing for logging should be done in this function
	# print(info_dicts['n_cons_viol'] )

	# print("Inside extract statistics")
	# IPython.embed()
	"""['obj_pred', 'U_pred', 'U_ref_pred', 'x_pred', 'u_pred', 'name_of_lb_viol', 'name_of_ub_viol', 'n_cons_viol', 'x',
	 'U', 'stat_dict']"""

	stat_dict = {}
	rollouts_x = info_dicts["x"]
	rollouts_U = info_dicts["U"]
	rollouts_obj_pred = info_dicts["obj_pred"]
	rollouts_U_pred = info_dicts["U_pred"]
	rollouts_U_ref_pred = info_dicts["U_ref_pred"]
	rollouts_x_pred = info_dicts["x_pred"]
	rollouts_u_pred = info_dicts["u_pred"]
	N_rollouts = len(rollouts_x)

	# Main statistic
	phi0_tol = 1e-2
	rollouts_which_unsafe = np.array([np.any(phi_0(rollout_x) > phi0_tol) for rollout_x in rollouts_x])
	rollouts_with_x0_safe = np.array([(phi_0(rollout_x)[0] <= phi0_tol) for rollout_x in rollouts_x]).flatten()

	# rollouts_phi0 = np.array([phi_0(rollout_x) for rollout_x in rollouts_x])
	rollouts_percent_unsafe = np.sum(rollouts_which_unsafe*rollouts_with_x0_safe)*100/N_rollouts
	rollouts_percent_safe = 100 - rollouts_percent_unsafe

	rollouts_which_safe = np.logical_not(rollouts_which_unsafe*rollouts_with_x0_safe)
	stat_dict["rollouts_which_safe"] = rollouts_which_safe
	stat_dict["rollouts_percent_safe"] = rollouts_percent_safe
	print("%.3f %% of rollouts safe" % rollouts_percent_safe)

	# Debugging
	unsafe_ind = np.argwhere(np.logical_not(rollouts_which_safe)).flatten()
	print("%i number of rollouts unsafe" % unsafe_ind.size)

	# Did MPC become infeasible at any point? And which constraints became infeasible?
	rollouts_with_infeas = np.sum(np.array(info_dicts["n_cons_viol"]), axis=1) > 0
	ind_unsafe_rollouts = np.argwhere(np.logical_not(rollouts_which_safe)).flatten()
	percent_unsafe_with_infeas = np.mean(rollouts_with_infeas[ind_unsafe_rollouts])*100
	percent_with_infeas = np.mean(rollouts_with_infeas)*100
	stat_dict["percent_unsafe_with_infeas"] = percent_unsafe_with_infeas
	stat_dict["percent_with_infeas"] = percent_with_infeas

	print("%.3f %% of rollouts have constraint violations" % percent_with_infeas)
	print("%.3f %% of unsafe rollouts have constraint violations" % percent_unsafe_with_infeas)

	# TODO: another possible reason for unsafe rollouts is that the initial state is unsafe
	# initial states were computed with a slightly diff MPC formulation than the one being used for rollouts
	# IPython.embed()

	return stat_dict


def make_mpc_rollouts_save_folder(args):
	save_parent_folder = "./flying_mpc_outputs/rollouts_loading_%s" % (args.rollout_load_x0_fnm)
	if not os.path.exists(save_parent_folder):
		os.makedirs(save_parent_folder)

	N_steps_max = math.ceil(args.rollout_T_max / args.dt)
	save_folder = "dt_%.5f_length_%i" % (args.dt, N_steps_max)
	if args.rollout_u_ref == "unactuated":
		save_folder += "_u_ref_unact"
	elif args.rollout_u_ref == "LQR":
		save_folder += "_u_ref_LQR_q_%.2f_r_%.2f" % (args.rollout_LQR_q, args.rollout_LQR_r)

	if args.rollout_mpc_set_initial_guess:
		save_folder += "_set_init_guess"

	# save_folder += ".pkl"
	save_fpth = save_parent_folder + "/" + save_folder

	if not os.path.exists(save_fpth):
		os.makedirs(save_fpth)

	print("Rollout data will be saved at folder: %s" % save_fpth)
	# IPython.embed()
	return save_fpth

# def save_rollout_data(info_dicts, save_fpth):
# 	stat_dict = extract_statistics(info_dicts)

# 	# Fill out experiment dict
# 	info_dicts["stat_dict"] = stat_dict
#
# 	# for key, value in stat_dict.items():
# 	# 	print("%s: %.3f" % (key, value))
#
# 	with open(save_fpth, 'wb') as handle:
# 		pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
	# IPython.embed()
	import pickle
	# info_dicts = pickle.load(open("flying_mpc_outputs/rollouts_loading_volume_approx_target_N_samp_inside_5000/dt_0.05000_length_200_u_ref_LQR_q_0.10_r_1.00/rollouts.pkl", "rb"))
	# fpth = "flying_mpc_outputs/rollouts_loading_volume_approx_target_N_samp_inside_5000/dt_0.05000_length_200_u_ref_LQR_q_0.10_r_1.00/rollouts.pkl"
	# fpth = "flying_mpc_outputs/rollouts_loading_volume_approx_target_N_samp_inside_5000/dt_0.05000_length_200_u_ref_LQR_q_1.00_r_1.00/rollouts.pkl"
	fpth = "flying_mpc_outputs/rollouts_loading_volume_approx_target_N_samp_inside_5000/dt_0.05000_length_200_u_ref_unact/rollouts.pkl"
	print(fpth)
	info_dicts = pickle.load(open(fpth, "rb"))
	stat_dict = extract_statistics(info_dicts)
	# info_dicts["stat_dict"] = stat_dict
	# with open(fpth, 'wb') as handle:
	# 	pickle.dump(info_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)