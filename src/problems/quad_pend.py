import torch
import numpy as np
from torch import nn
g = 9.81  # Gravitational acceleration [m/s^2]

class RhoSum(nn.Module):
	"""
	Computes safety specification:
	    ρ(x) = δ² + γ² + β² - δ_limit²

	where:
	- δ: angle between pendulum axis and vertical (computed from φ, θ)
	- γ, β: quadcopter roll and pitch angles
	- δ_limit: limit on all angles

	The safe set is defined as {x : ρ(x) ≤ 0}.
	This means the safety objective is to keep the pendulum upright and prevent the quadcopter from rolling.
	"""
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # Unpack all physical parameters
		self.i = self.state_index_dict

	def forward(self, x):
		"""Computes ρ(x) = δ² + γ² + β² - δ_limit² for batch of states.

		Args:
			x: Batch of states (bs, 10)

		Returns:
			Tensor (bs, 1): Safety specification values
		"""
		# Extract angles
		theta = x[:, [self.i["theta"]]]  # Pendulum angle axis 1
		phi = x[:, [self.i["phi"]]]      # Pendulum angle axis 2
		gamma = x[:, [self.i["gamma"]]]  # quadcopter roll
		beta = x[:, [self.i["beta"]]]    # quadcopter pitch

		# Compute pendulum deviation angle δ from vertical
		# Uses: cos(δ) = cos(φ)·cos(θ)
		cos_cos = torch.cos(theta)*torch.cos(phi)

		# Add small epsilon for numerical stability (prevents NaN at vertical)
		eps = 1e-4
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos)*eps

		delta = torch.acos(cos_cos + signed_eps)

		# Sum-of-squares safety specification
		rv = delta**2 + gamma**2 + beta**2 - self.delta_safety_limit**2

		return rv

class XDot(nn.Module):
	"""System dynamics ẋ = f(x, u) for quadcopter-pendulum."""
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # Unpack all physical parameters
		self.device = device
		self.i = self.state_index_dict

	def forward(self, x, u):
		"""Computes ẋ = f(x, u) for batch of states and controls.

		Args:
			x: Batch of states (bs, 10)
			u: Batch of controls (bs, 4)

		Returns:
			Tensor (bs, 10): State derivatives ẋ
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
		# Compute rotation matrix R: body frame to global frame (ZYX Euler angles)
		###############################################################################
		# R = R_z(α) · R_y(β) · R_x(γ)
		R = torch.zeros((x.shape[0], 3, 3), device=self.device)

		# First row
		R[:, 0, 0] = torch.cos(alpha)*torch.cos(beta)
		R[:, 0, 1] = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma)
		R[:, 0, 2] = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)

		# Second row
		R[:, 1, 0] = torch.sin(alpha)*torch.cos(beta)
		R[:, 1, 1] = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma)
		R[:, 1, 2] = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)

		# Third row
		R[:, 2, 0] = -torch.sin(beta)
		R[:, 2, 1] = torch.cos(beta)*torch.sin(gamma)
		R[:, 2, 2] = torch.cos(beta)*torch.cos(gamma)

		# Extract thrust direction components in global frame
		# k = R·[0, 0, 1]ᵀ is the third column of R
		k_x = R[:, 0, 2]
		k_y = R[:, 1, 2]
		k_z = R[:, 2, 2]

		# Total thrust (control input F is deviation from hover)
		F = (u[:, 0] + self.M*g)

		###############################################################################
		# Compute quadcopter angular accelerations from torques
		###############################################################################
		J = torch.tensor([self.J_x, self.J_y, self.J_z]).to(self.device)

		# Normalize torques by moments of inertia
		norm_torques = u[:, 1:]*(1.0/J)  # (bs, 3)

		# Transform torques to global frame and extract angular accelerations
		ddquad_angles = torch.bmm(R, norm_torques[:, :, None])  # (bs, 3, 1)
		ddquad_angles = ddquad_angles[:, :, 0]  # (bs, 3)

		ddgamma = ddquad_angles[:, 0]  # Roll acceleration
		ddbeta = ddquad_angles[:, 1]   # Pitch acceleration
		ddalpha = ddquad_angles[:, 2]  # Yaw acceleration

		###############################################################################
		# Compute pendulum angular accelerations (coupled to quadcopter via thrust)
		###############################################################################
		# Derived from Euler-Lagrange equations with thrust coupling
		ddphi = (3.0)*(k_y*torch.cos(phi) + k_z*torch.sin(phi))/(2*self.M*self.L_p*torch.cos(theta))*F + 2*dtheta*dphi*torch.tan(theta)

		ddtheta = (3.0*(-k_x*torch.cos(theta)-k_y*torch.sin(phi)*torch.sin(theta) + k_z*torch.cos(phi)*torch.sin(theta))/(2.0*self.M*self.L_p))*F - torch.square(dphi)*torch.sin(theta)*torch.cos(theta)

		###############################################################################
		# Assemble state derivative vector
		###############################################################################
		# ẋ = [γ̇, β̇, α̇, γ̈, β̈, α̈, φ̇, θ̇, φ̈, θ̈]
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
	"""Computes vertices of control input constraint polytope U."""
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # Unpack all physical parameters
		self.device = device

		###############################################################################
		# Precompute control polytope vertices (16 vertices for 4 binary rotors)
		###############################################################################
		k1 = self.k1
		k2 = self.k2
		l = self.l

		# Mixer matrix
		M = np.array([
			[k1,   k1,    k1,   k1],    # Total thrust
			[0,   -l*k1,  0,    l*k1],  # Roll torque
			[l*k1, 0,    -l*k1, 0],     # Pitch torque
			[-k2,  k2,   -k2,   k2]     # Yaw torque
		])

		# Generate all 16 vertices of the unit hypercube [0,1]^4
		r1 = np.concatenate((np.zeros(8), np.ones(8)))
		r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
		r3 = np.concatenate((np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),
		                     np.zeros(2), np.ones(2), np.zeros(2), np.ones(2)))
		r4 = np.zeros(16)
		r4[1::2] = 1.0

		impulse_vert = np.concatenate((r1[None], r2[None], r3[None], r4[None]), axis=0)  # (4, 16)

		# Transform rotor impulses to force/torque space and subtract hover thrust
		force_vert = M @ impulse_vert - np.array([[self.M*g], [0.0], [0.0], [0.0]])
		force_vert = force_vert.T.astype("float32")  # (16, 4)

		self.vert = torch.from_numpy(force_vert).to(self.device)

	def forward(self, x):
		"""Returns control polytope vertices replicated for batch.

		Args:
			x: Batch of states (bs, 10) - only used for batch size

		Returns:
			Tensor (bs, 16, 4): Polytope vertices for each state
		"""
		rv = self.vert  # (16, 4)
		rv = rv.unsqueeze(dim=0)  # (1, 16, 4)
		rv = rv.expand(x.shape[0], -1, -1)  # (bs, 16, 4)
		return rv
