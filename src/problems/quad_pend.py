"""Quadcopter-pendulum system: dynamics, safety specification, and control constraints."""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import numpy as np
from torch import nn

g = 9.81  # Gravitational acceleration [m/s^2]


class Rho(nn.Module):
	"""Base safety specification ρ(x) = δ² + γ² + β² - δ_limit².

	The safe set is {x : ρ(x) ≤ 0}, enforcing that the pendulum deviation
	angle δ and quadcopter tilt angles γ, β stay within δ_limit.

	δ is computed from pendulum angles (φ, θ) via cos(δ) = cos(φ)·cos(θ).
	"""
	def __init__(self, param_dict: dict) -> None:
		super().__init__()
		self.__dict__.update(param_dict)
		self.i = self.state_index_dict

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Computes ρ(x) = δ² + γ² + β² - δ_limit² for a batch of states.

		Args:
			x: Batch of states (bs, 10)

		Returns:
			Tensor (bs, 1): Safety specification values; ρ(x) ≤ 0 is safe
		"""
		theta = x[:, [self.i["theta"]]]  # Pendulum angle axis 1
		phi = x[:, [self.i["phi"]]]      # Pendulum angle axis 2
		gamma = x[:, [self.i["gamma"]]]  # quadcopter roll
		beta = x[:, [self.i["beta"]]]    # quadcopter pitch

		# Compute pendulum deviation angle δ via cos(δ) = cos(φ)·cos(θ)
		cos_cos = torch.cos(theta)*torch.cos(phi)

		# Small epsilon for numerical stability (prevents NaN at vertical)
		eps = 1e-4
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos)*eps

		delta = torch.acos(cos_cos + signed_eps)

		rv = delta**2 + gamma**2 + beta**2 - self.delta_safety_limit**2
		return rv


class XDot(nn.Module):
	"""System dynamics ẋ = f(x, u) for quadcopter-pendulum."""
	def __init__(self, param_dict: dict, device: torch.device) -> None:
		super().__init__()
		self.__dict__.update(param_dict)
		self.device = device
		self.i = self.state_index_dict

	def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
		"""Computes ẋ = f(x, u) for a batch of states and controls.

		Args:
			x: Batch of states (bs, 10)
			u: Batch of controls (bs, 4) as [F, τ_x, τ_y, τ_z]

		Returns:
			Tensor (bs, 10): State derivatives ẋ = [γ̇, β̇, α̇, γ̈, β̈, α̈, φ̇, θ̇, φ̈, θ̈]
		"""
		###############################################################################
		# Extract state variables
		###############################################################################
		gamma = x[:, self.i["gamma"]]   # quadcopter roll
		beta = x[:, self.i["beta"]]     # quadcopter pitch
		alpha = x[:, self.i["alpha"]]   # quadcopter yaw

		phi = x[:, self.i["phi"]]       # Pendulum angle 1
		theta = x[:, self.i["theta"]]   # Pendulum angle 2
		dphi = x[:, self.i["dphi"]]     # Pendulum angular velocity 1
		dtheta = x[:, self.i["dtheta"]] # Pendulum angular velocity 2

		###############################################################################
		# Rotation matrix R: body frame → global frame (ZYX Euler angles)
		###############################################################################
		# R = R_z(α) · R_y(β) · R_x(γ)
		R = torch.zeros((x.shape[0], 3, 3), device=self.device)

		R[:, 0, 0] = torch.cos(alpha)*torch.cos(beta)
		R[:, 0, 1] = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma)
		R[:, 0, 2] = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)

		R[:, 1, 0] = torch.sin(alpha)*torch.cos(beta)
		R[:, 1, 1] = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma)
		R[:, 1, 2] = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)

		R[:, 2, 0] = -torch.sin(beta)
		R[:, 2, 1] = torch.cos(beta)*torch.sin(gamma)
		R[:, 2, 2] = torch.cos(beta)*torch.cos(gamma)

		# Thrust direction k = R·[0, 0, 1]ᵀ (third column of R)
		k_x = R[:, 0, 2]
		k_y = R[:, 1, 2]
		k_z = R[:, 2, 2]

		# Total thrust (u[:,0] is deviation from hover)
		F = (u[:, 0] + self.M*g)

		###############################################################################
		# Quadcopter angular accelerations from torques
		###############################################################################
		J = torch.tensor([self.J_x, self.J_y, self.J_z]).to(self.device)

		norm_torques = u[:, 1:]*(1.0/J)  # (bs, 3)
		ddquad_angles = torch.bmm(R, norm_torques[:, :, None])[:, :, 0]  # (bs, 3)

		ddgamma = ddquad_angles[:, 0]  # Roll acceleration
		ddbeta = ddquad_angles[:, 1]   # Pitch acceleration
		ddalpha = ddquad_angles[:, 2]  # Yaw acceleration

		###############################################################################
		# Pendulum angular accelerations (Euler-Lagrange, coupled via thrust)
		###############################################################################
		ddphi = (3.0)*(k_y*torch.cos(phi) + k_z*torch.sin(phi))/(2*self.M*self.L_p*torch.cos(theta))*F + 2*dtheta*dphi*torch.tan(theta)

		ddtheta = (3.0*(-k_x*torch.cos(theta)-k_y*torch.sin(phi)*torch.sin(theta) + k_z*torch.cos(phi)*torch.sin(theta))/(2.0*self.M*self.L_p))*F - torch.square(dphi)*torch.sin(theta)*torch.cos(theta)

		###############################################################################
		# Assemble ẋ = [γ̇, β̇, α̇, γ̈, β̈, α̈, φ̇, θ̇, φ̈, θ̈]
		###############################################################################
		rv = torch.cat([
			x[:, [self.i["dgamma"]]],  # γ̇
			x[:, [self.i["dbeta"]]],   # β̇
			x[:, [self.i["dalpha"]]],  # α̇
			ddgamma[:, None],          # γ̈
			ddbeta[:, None],           # β̈
			ddalpha[:, None],          # α̈
			dphi[:, None],             # φ̇
			dtheta[:, None],           # θ̇
			ddphi[:, None],            # φ̈
			ddtheta[:, None]           # θ̈
		], axis=1)

		return rv


class ULimitSetVertices(nn.Module):
	"""Precomputes and returns vertices of the control input constraint polytope U."""
	def __init__(self, param_dict: dict, device: torch.device) -> None:
		super().__init__()
		self.__dict__.update(param_dict)
		self.device = device

		###############################################################################
		# Precompute control polytope vertices (16 vertices for 4 binary rotors)
		###############################################################################
		k1 = self.k1
		k2 = self.k2
		l = self.l

		# Mixer matrix: maps rotor impulses [0,1]^4 → (F, τ_x, τ_y, τ_z)
		M = np.array([
			[k1,   k1,    k1,   k1],    # Total thrust
			[0,   -l*k1,  0,    l*k1],  # Roll torque
			[l*k1, 0,    -l*k1, 0],     # Pitch torque
			[-k2,  k2,   -k2,   k2]     # Yaw torque
		])

		# All 16 vertices of the unit hypercube [0,1]^4
		r1 = np.concatenate((np.zeros(8), np.ones(8)))
		r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
		r3 = np.concatenate((np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),
		                     np.zeros(2), np.ones(2), np.zeros(2), np.ones(2)))
		r4 = np.zeros(16)
		r4[1::2] = 1.0

		impulse_vert = np.concatenate((r1[None], r2[None], r3[None], r4[None]), axis=0)  # (4, 16)

		# Transform to force/torque space and subtract hover thrust
		force_vert = M @ impulse_vert - np.array([[self.M*g], [0.0], [0.0], [0.0]])
		force_vert = force_vert.T.astype("float32")  # (16, 4)

		self.vert = torch.from_numpy(force_vert).to(self.device)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Returns control polytope vertices replicated for batch.

		Args:
			x: Batch of states (bs, 10) — used only for batch size

		Returns:
			Tensor (bs, 16, 4): Control polytope vertices for each state
		"""
		return self.vert.unsqueeze(0).expand(x.shape[0], -1, -1)  # (bs, 16, 4)


@dataclass
class QuadPendConfig:
	"""Parameters for the quadcopter-pendulum system.

	Physical Parameters:
		m: quadcopter mass [kg]
		m_p: Pendulum mass [kg]
		M: Total mass (m + m_p) [kg]
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

	# System dimensions
	r: int = 2
	x_dim: int = 10
	u_dim: int = 4

	# Derived parameters (computed in __post_init__)
	M: float = field(init=False)
	state_index_dict: Dict[str, int] = field(init=False)
	x_lim: np.ndarray = field(init=False)

	def __post_init__(self) -> None:
		"""Computes derived parameters from base parameters."""
		self.M = self.m + self.m_p

		state_index_names = [
			"gamma", "beta", "alpha",      # quadcopter angles
			"dgamma", "dbeta", "dalpha",   # quadcopter angular velocities
			"phi", "theta",                # Pendulum angles
			"dphi", "dtheta"               # Pendulum angular velocities
		]
		self.state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))

		ub = self.box_ang_vel_limit
		thresh = np.array([
			math.pi / 3, math.pi / 3, math.pi,  # quadcopter angle limits
			ub, ub, ub,                          # quadcopter velocity limits
			math.pi / 3, math.pi / 3,            # Pendulum angle limits
			ub, ub                               # Pendulum velocity limits
		], dtype=np.float32)

		self.x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)

	def to_dict(self) -> Dict:
		"""Converts config to dictionary for backward compatibility."""
		return {
			"m": self.m, "J_x": self.J_x, "J_y": self.J_y, "J_z": self.J_z,
			"l": self.l, "k1": self.k1, "k2": self.k2,
			"m_p": self.m_p, "L_p": self.L_p, "M": self.M,
			"delta_safety_limit": self.delta_safety_limit,
			"box_ang_vel_limit": self.box_ang_vel_limit,
			"r": self.r, "x_dim": self.x_dim, "u_dim": self.u_dim,
			"state_index_dict": self.state_index_dict,
			"x_lim": self.x_lim,
		}


def create_quad_pend_param_dict(args: Optional[object] = None) -> Dict:
	"""Creates parameter dictionary for the quadcopter-pendulum system.

	Args:
		args: Optional argparse.Namespace (currently unused; parameters are hardcoded)

	Returns:
		dict: System parameters — see QuadPendConfig for keys
	"""
	return QuadPendConfig().to_dict()
