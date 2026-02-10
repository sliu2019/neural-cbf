"""Main entry point for neural Control Barrier Function synthesis training.

This module implements the learner-critic training framework for synthesizing
neural CBFs that respect input saturation constraints, as described in
"Safe control under input limits with neural control barrier functions"
(liu23e, CoRL 2022).

The training optimizes a neural network to represent a modified CBF that
ensures safety even when control inputs saturate. The method uses:
- Learner: Minimizes saturation risk + volume regularization
- Critic: Finds worst-case counterexamples via gradient ascent on boundary

System: quadcopter-pendulum (10D state, 4D control)
Output: Trained neural CBF checkpoints and training statistics

References:
    liu23e.pdf Section 3 (Method), Section 4 (Experiments), Algorithm 1
"""
import torch
from torch import nn
from torch.autograd import grad
import os
import math
import pickle
from dataclasses import dataclass, field
from typing import Dict
import numpy as np


from src.create_arg_parser import create_arg_parser, print_args
from src.neural_phi import NeuralPhi

from src.critic import Critic
from src.learner import Learner
from src.reg_sampler import RegSampler

from src.utils import *

class SaturationRisk(nn.Module):
	"""Computes worst-case CBF derivative over control limit set vertices.

	This loss function implements the saturation avoidance objective from
	liu23e.pdf Eq. 4. For each state x on the boundary, it evaluates:

		min_{u ∈ vertices(U)} φ̇(x, u) = ∇φ(x)·f(x,u)

	where the minimum is taken over all vertices of the polytopic control
	limit set U. A negative value indicates the CBF derivative is negative
	even with worst-case saturated control, violating the forward invariance
	condition and potentially causing safety violations.

	The critic maximizes this loss to find states where saturation causes
	problems, and the learner minimizes it to eliminate such violations.

	Attributes:
		phi_fn: Neural CBF function returning (bs, r+1) with φ* in last column
		xdot_fn: System dynamics f(x,u) returning state derivatives (bs, x_dim)
		uvertices_fn: Returns control polytope vertices (bs, n_vertices, u_dim)
		x_dim: State space dimension
		u_dim: Control input dimension
		device: PyTorch device for computation
		logger: Logger instance (for debugging)
		args: Training arguments namespace (currently unused)

	Returns:
		When called: Tensor (bs, 1) with minimum φ̇ values (saturation risk)
	"""
	def __init__(self, phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args):
		super().__init__()
		self.phi_fn = phi_fn
		self.xdot_fn = xdot_fn
		self.uvertices_fn = uvertices_fn
		self.x_dim = x_dim
		self.u_dim = u_dim
		self.device = device
		self.logger = logger
		self.args = args

	def forward(self, x):
		"""Computes minimum CBF derivative over control limit vertices.

		Algorithm:
		1. Get all vertices of the control limit polytope U
		2. Compute φ̇(x, u) = ∇φ(x) · f(x,u) for each vertex u
		3. Return minimum over all vertices (best-case saturation at x)

		Args:
			x: Batch of states (bs, x_dim)

		Returns:
			$\mathcal{L}(x, \theta)$ values (bs, 1)
		"""
		# Get vertices of control limit set (polytope with n_vertices corners)
		u_lim_set_vertices = self.uvertices_fn(x)  # (bs, n_vertices, u_dim)
		n_vertices = u_lim_set_vertices.size()[1]

		# Reshape to evaluate all (state, control) pairs in parallel
		# We need to evaluate xdot at every (x[i], u_vertex[j]) combination
		U = torch.reshape(u_lim_set_vertices, (-1, self.u_dim))  # (bs*n_vertices, u_dim)
		X = (x.unsqueeze(1)).repeat(1, n_vertices, 1)  # (bs, n_vertices, x_dim)
		X = torch.reshape(X, (-1, self.x_dim))  # (bs*n_vertices, x_dim)

		# Compute state derivatives f(x,u) for all combinations
		xdot = self.xdot_fn(X, U)  # (bs*n_vertices, x_dim)

		# Compute gradient of CBF: ∇φ(x)
		orig_req_grad_setting = x.requires_grad
		x.requires_grad = True
		phi_value = self.phi_fn(x)  # (bs, r+1)
		grad_phi = grad([torch.sum(phi_value[:, -1])], x, create_graph=True)[0]  # (bs, x_dim)
		x.requires_grad = orig_req_grad_setting

		# Broadcast gradient to match all vertex evaluations
		grad_phi = (grad_phi.unsqueeze(1)).repeat(1, n_vertices, 1)  # (bs, n_vertices, x_dim)
		grad_phi = torch.reshape(grad_phi, (-1, self.x_dim))  # (bs*n_vertices, x_dim)

		# Compute Lie derivative: φ̇ = ∇φ · f(x,u) for all vertices
		phidot_cand = xdot.unsqueeze(1).bmm(grad_phi.unsqueeze(2))  # (bs*n_vertices, 1, 1)
		phidot_cand = torch.reshape(phidot_cand, (-1, n_vertices))  # (bs, n_vertices)

		# Take minimum over all control vertices (best-case saturation on a given state)
		phidot, _ = torch.min(phidot_cand, 1)  # (bs,)

		result = phidot
		result = result.view(-1, 1) # ensures bs x 1

		return result

class RegularizationLoss(nn.Module):
	"""Volume regularization loss to maximize the safe set.

	Implements the regularization term from liu23e.pdf Eq. 7. This term
	encourages the neural CBF to produce a large safe set by penalizing
	states where φ(x) is close to 0 (boundary).

	The loss uses sigmoid transform of max φ values:
		L_reg = weight · mean(sigmoid(0.3 · max_i φ_i(x)))

	States deep inside safe set (φ << 0) contribute little, while states
	near boundary (φ ≈ 0) contribute more. This provides gradient signal
	to expand the safe set.

	Attributes:
		phi_fn: Neural CBF function
		device: PyTorch device
		reg_weight: Coefficient for regularization term (default: 0.0)
	"""
	def __init__(self, phi_fn, device, reg_weight=0.0):
		super().__init__()
		self.phi_fn = phi_fn
		self.device = device
		self.reg_weight = reg_weight
		assert reg_weight >= 0.0

	def forward(self, x):
		"""Computes volume regularization loss.

		Args:
			x: Batch of sampled states (bs, x_dim), typically sampled
			   uniformly inside the safe set zero-sublevel

		Returns:
			Scalar regularization loss (encourages large safe set)
		"""
		# Evaluate all φ_i(x) for i=0..r (CBF and its higher-order terms)
		all_phi_values = self.phi_fn(x)  # (bs, r+1)

		# Safe set condition: max_i φ_i(x) ≤ 0
		# Taking max gives tightest constraint
		max_phi_values = torch.max(all_phi_values, dim=1)[0]  # (bs,)

		# Apply sigmoid transform: sigmoid(0.3·φ)
		# The 0.3 scaling factor adjusts sensitivity:
		# - States far inside (φ << 0): sigmoid ≈ 0, little gradient
		# - States near boundary (φ ≈ 0): sigmoid ≈ 0.5, large gradient 
		# - States outside (φ > 0): sigmoid ≈ 1, little gradient
		transform_of_max_phi = torch.sigmoid(0.3*max_phi_values)  # (bs,)

		# Weighted mean over batch
		reg = self.reg_weight*torch.mean(transform_of_max_phi)
		return reg


@dataclass
class QuadPendConfig:
	"""Configuration for quadcopter-pendulum system parameters.

	This dataclass provides type-safe configuration for the quadcopter with
	attached inverted pendulum system, replacing the untyped param_dict.

	Physical Parameters:
		m: quadcopter mass [kg]
		m_p: Pendulum mass [kg]
		M: Total mass (computed as m + m_p) [kg]
		L_p: Pendulum length [m]
		J_x, J_y, J_z: Moments of inertia [kg·m²]
		l: quadcopter arm length [m]
		k1: Thrust coefficient [N·s²]
		k2: Drag coefficient [N·m·s²]

	Safety Parameters:
		delta_safety_limit: Maximum allowed angle from vertical [rad]
		box_ang_vel_limit: Angular velocity bounds [rad/s]

	System Dimensions:
		x_dim: State space dimension (10)
		u_dim: Control input dimension (4)
		r: Relative degree of CBF (2)

	State Space:
		state_index_dict: Mapping from state names to indices
		x_lim: State space bounds (x_dim, 2) with [min, max] per dimension

	Reference:
		liu23e.pdf Section 4 for parameter values and system description
	"""
	# Physical parameters (quadcopter)
	m: float = 0.8
	J_x: float = 0.005
	J_y: float = 0.005
	J_z: float = 0.009
	l: float = 1.5
	k1: float = 4.0
	k2: float = 0.05

	# Physical parameters (pendulum)
	m_p: float = 0.04  # 5% of quadcopter weight
	L_p: float = 3.0

	# Safety parameters
	delta_safety_limit: float = math.pi / 4  # Should be <= π/4
	box_ang_vel_limit: float = 20.0

	# System dimensions (computed in __post_init__)
	r: int = 2
	x_dim: int = 10
	u_dim: int = 4

	# Derived parameters (computed in __post_init__)
	M: float = field(init=False)  # Total mass
	state_index_dict: Dict[str, int] = field(init=False)
	x_lim: np.ndarray = field(init=False)

	def __post_init__(self):
		"""Computes derived parameters from base parameters."""
		# Total mass
		self.M = self.m + self.m_p

		# State indexing
		state_index_names = [
			"gamma", "beta", "alpha",      # quadcopter angles
			"dgamma", "dbeta", "dalpha",   # quadcopter angular velocities
			"phi", "theta",                # Pendulum angles
			"dphi", "dtheta"               # Pendulum angular velocities
		]
		self.state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))

		# State space bounds
		ub = self.box_ang_vel_limit
		thresh = np.array([
			math.pi / 3, math.pi / 3, math.pi,  # quadcopter angle limits
			ub, ub, ub,                         # quadcopter velocity limits
			math.pi / 3, math.pi / 3,           # Pendulum angle limits
			ub, ub                              # Pendulum velocity limits
		], dtype=np.float32)

		self.x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)

	def to_dict(self) -> Dict:
		"""Converts config to dictionary for backward compatibility.

		Returns:
			dict: Parameter dictionary with all attributes
		"""
		return {
			"m": self.m,
			"J_x": self.J_x,
			"J_y": self.J_y,
			"J_z": self.J_z,
			"l": self.l,
			"k1": self.k1,
			"k2": self.k2,
			"m_p": self.m_p,
			"L_p": self.L_p,
			"M": self.M,
			"delta_safety_limit": self.delta_safety_limit,
			"box_ang_vel_limit": self.box_ang_vel_limit,
			"r": self.r,
			"x_dim": self.x_dim,
			"u_dim": self.u_dim,
			"state_index_dict": self.state_index_dict,
			"x_lim": self.x_lim,
		}


def create_quad_pend_param_dict(args=None):
	"""Creates parameter dictionary for flying inverted pendulum system.

	Instantiates QuadPendConfig with default parameters and converts
	to dictionary for backward compatibility with existing code.

	Args:
		args: Optional argparse.Namespace to override default parameters
		      (currently unused, parameters are hardcoded)

	Returns:
		dict: Parameter dictionary with keys:
			Physical: m, J_x, J_y, J_z, l, k1, k2, m_p, L_p, M
			Safety: delta_safety_limit, box_ang_vel_limit
			Dimensions: x_dim, u_dim, r
			Bounds: x_lim (state space bounds)
			Mappings: state_index_dict

	Reference:
		liu23e.pdf Section 4 for system description and parameters

	Note:
		This function now uses QuadPendConfig dataclass internally
		for type safety, but returns a dict for backward compatibility.
	"""
	# Create typed config with default parameters
	config = QuadPendConfig()

	# Convert to dict for backward compatibility with existing code
	param_dict = config.to_dict()

	return param_dict

def main(args):
	# Boilerplate for saving
	# IPython.embed()
	save_folder = '%s_%s' % (args.problem, args.affix)

	log_folder = os.path.join(args.log_root, save_folder)
	model_folder = os.path.join(args.model_root, save_folder)

	makedirs(log_folder)
	makedirs(model_folder)

	setattr(args, 'log_folder', log_folder)
	setattr(args, 'model_folder', model_folder)

	logger = create_logger(log_folder, 'train', 'info')
	print_args(args, logger)

	args_savepth = os.path.join(log_folder, "args.txt")
	save_args(args, args_savepth)

	# Device
	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
		dev = "cuda:%i" % (args.gpu)
	else:
		raise NotImplementedError
	device = torch.device(dev)

	# Selecting problem
	if args.problem == "quad_pend":
		param_dict = create_quad_pend_param_dict(args)

		r = param_dict["r"]
		x_dim = param_dict["x_dim"]
		u_dim = param_dict["u_dim"]
		x_lim = param_dict["x_lim"]

		# Create phi
		from problems.quad_pend import RhoSum, XDot, ULimitSetVertices
		# if args.rho == "sum":
		rho_fn = RhoSum(param_dict)
		# elif args.rho == "max":
		# 	rho_fn = RhoMax(param_dict)

		xdot_fn = XDot(param_dict, device)
		uvertices_fn = ULimitSetVertices(param_dict, device)

		# reg_sampler = reg_samplers_name_to_class_dict[args.reg_sampler](x_lim, device, logger, n_samples=args.reg_n_samples)
		reg_sampler = RegSampler(x_lim, device, logger, n_samples=args.reg_n_samples)

		# if args.phi_include_xe:
		x_e = torch.zeros(1, x_dim)
		# else:
			# x_e = None

		# Passing in subset of state to NN
		from src.utils import IndexNNInput, TransformEucNNInput
		state_index_dict = param_dict["state_index_dict"]
		# if args.phi_nn_inputs == "spherical":
		# 	nn_input_modifier = None
		# elif args.phi_nn_inputs == "euc":
		nn_input_modifier = TransformEucNNInput(state_index_dict)
	else:
		raise NotImplementedError

	# Save param_dict
	with open(os.path.join(log_folder, "param_dict.pkl"), 'wb') as handle:
		pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Send all modules to the correct device
	rho_fn = rho_fn.to(device)
	xdot_fn = xdot_fn.to(device)
	uvertices_fn = uvertices_fn.to(device)
	if x_e is not None:
		x_e = x_e.to(device)
	x_lim = torch.tensor(x_lim).to(device)

	# Create CBF, etc.
	# if args.phi_design == "neural":
	phi_fn = NeuralPhi(rho_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e, nn_input_modifier=nn_input_modifier)
	# elif args.phi_design == "low":
	# 	phi_fn = LowPhi(rho_fn, xdot_fn, x_dim, u_dim, device, param_dict)

	saturation_risk = SaturationRisk(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args)
	reg_fn = RegularizationLoss(phi_fn, device, reg_weight=args.reg_weight) #, reg_transform=args.reg_transform)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)
	saturation_risk = saturation_risk.to(device)
	reg_fn = reg_fn.to(device)

	# Create critic
	critic = Critic(x_lim, device, logger, n_samples=args.critic_n_samples,\
											# stopping_condition=args.critic_stopping_condition,
											max_n_steps=args.critic_max_n_steps, \
											# lr=args.critic_lr, \
											# projection_tolerance=args.critic_projection_tolerance,
											# projection_lr=args.critic_projection_lr,
											# projection_time_limit=args.critic_projection_time_limit,
											# critic_use_n_step_schedule=args.critic_use_n_step_schedule,
											# boundary_sampling_speedup_method=args.gradient_batch_warmstart_faster_speedup_method,
											# boundary_sampling_method=args.gradient_batch_warmstart_faster_sampling_method,
											# gaussian_t=args.gradient_batch_warmstart_faster_gaussian_t,
											# p_reuse=args.critic_p_reuse
											)

	# Create test critic
	# Note: doesn't matter that we're passing train params. We're only using test_critic to sample on boundary
	test_critic = Critic(x_lim, device, logger, n_samples=args.critic_n_samples, \
												# stopping_condition=args.critic_stopping_condition,
												max_n_steps=args.critic_max_n_steps, \
												# lr=args.critic_lr, \
												# projection_tolerance=args.critic_projection_tolerance,
												# projection_lr=args.critic_projection_lr,
												# projection_time_limit=args.critic_projection_time_limit,
												# critic_use_n_step_schedule=args.critic_use_n_step_schedule,
												# boundary_sampling_speedup_method=args.gradient_batch_warmstart_faster_speedup_method,
												# boundary_sampling_method=args.gradient_batch_warmstart_faster_sampling_method,
												# gaussian_t=args.gradient_batch_warmstart_faster_gaussian_t,
												# p_reuse=args.critic_p_reuse
												)


	# Pass everything to learner
	learner = Learner(args, logger, critic, test_critic, reg_sampler, param_dict, device)
	learner.train(saturation_risk, reg_fn, phi_fn, xdot_fn)

	##############################################################
	#####################      Testing      ######################

	### Fill out ###

	# Testing gradient_batch_critic_warmstart_2
	# cpu_handle = torch.device("cpu")
	# cpu_phi_fn = phi_fn.to(cpu_handle)

	# def surface_fn(x, grad_x=False):
	# 	return phi_fn(x, grad_x=grad_x)[:, -1]
	#
	# critic.opt(saturation_risk, phi_fn, 0, debug=True)

	# Checking to see if all 4 quadcopter inputs appear in phidot (that is, if grad_phi_g is nonzero)
	"""N = 5
	x = torch.rand(N, 12).to(device)
	x.requires_grad = True
	phi_value = phi_fn(x)
	grad_phi = grad([torch.sum(phi_value[:, -1])], x, create_graph=True)[0]  # check

	# IPython.embed()
	from src.problems.quadcopter import G
	g_fn = G(param_dict, device)

	g_x = g_fn(x)

	# IPython.embed()
	grad_phi_g = grad_phi[:, None]@g_x
	grad_phi_g = grad_phi_g[:, 0]
	print(grad_phi_g)"""


if __name__ == "__main__":
	# Parse command line arguments 
	parser = create_arg_parser()
	args = parser.parse_known_args()[0]

	# Set seeds 
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	main(args)
