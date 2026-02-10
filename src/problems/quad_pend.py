"""Quadcopter-pendulum system dynamics and safety specifications.

The system has 10 state variables and 4 control inputs:

State: x = [γ, β, α, γ̇, β̇, α̇, φ, θ, φ̇, θ̇] ∈ R^10
  - γ, β, α: quadcopter roll, pitch, yaw
  - φ, θ: Pendulum angles from vertical
  - Velocities: Time derivatives of angles

Control: u = [F, τ_x, τ_y, τ_z] ∈ R^4
  - F: Total thrust (hover thrust subtracted)
  - τ_x, τ_y, τ_z: Torques about body axes

The safety objective is to keep the pendulum upright and prevent the quadcopter from rolling.
"""
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
	"""System dynamics ẋ = f(x, u) for quadcopter-pendulum.

	Computes the time derivatives of all state variables using:
	1. quadcopter rigid body dynamics (rotation matrix formulation)
	2. Pendulum dynamics (coupled to quadcopter through thrust vector)

	The dynamics are derived from Euler-Lagrange equations with the following
	key couplings:
	- quadcopter thrust affects pendulum acceleration via thrust direction
	- Pendulum motion does NOT feedback to quadcopter (underactuated assumption)

	State Derivatives:
	    ẋ = [γ̇, β̇, α̇, γ̈, β̈, α̈, φ̇, θ̇, φ̈, θ̈]ᵀ

	where:
	- quadcopter angular accelerations: Computed from torques via rotation matrix R
	- Pendulum angular accelerations: Depend on thrust direction and Coriolis terms

	Rotation Matrix R (ZYX Euler angles):
	The rotation matrix R transforms from body frame to global frame using
	intrinsic rotations: R = R_z(α) R_y(β) R_x(γ)

	The thrust direction vector in global frame is k = R·[0, 0, 1]ᵀ, with
	components k_x, k_y, k_z used in pendulum dynamics.

	Pendulum Dynamics:
	The pendulum accelerations follow from Lagrangian mechanics:
	    φ̈ = (3/2ML_p) · (k_y cos(φ) + k_z sin(φ))/cos(θ) · F + Coriolis terms
	    θ̈ = (3/2ML_p) · (-k_x cos(θ) - k_y sin(φ)sin(θ) + k_z cos(φ)sin(θ)) · F + Coriolis terms

	Args (in param_dict):
		M: Total mass (quadcopter + pendulum) [kg]
		L_p: Pendulum length [m]
		J_x, J_y, J_z: Moments of inertia [kg·m²]
		state_index_dict: Mapping from state names to indices

	Input:
		x: Batch of states (bs, 10)
		u: Batch of controls (bs, 4) = [F, τ_x, τ_y, τ_z]

	Returns:
		ẋ: State derivatives (bs, 10)

	Note:
		Translational motion (x, y, z positions) is excluded as it's decoupled
		from the safety-critical pendulum angles.
	"""
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
		# Compute rotation matrix R: body frame → global frame (ZYX Euler angles)
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
	"""Computes vertices of control input constraint polytope U.

	For the quadcopter-pendulum, the control limits come from physical
	rotor constraints: each of the 4 rotors has bounded RPM ω_i ∈ [0, ω_max].

	The control space is a 4D polytope with 2^4 = 16 vertices corresponding
	to all combinations of rotors at minimum (0) or maximum (ω_max) RPM.

	Mixer Matrix Transformation:
	The mixer matrix M maps rotor impulses [ω₁², ω₂², ω₃², ω₄²]ᵀ to forces/torques:

	    [F  ]       [k1   k1    k1   k1  ] [ω₁²]
	    [τ_x] = M · [0   -l·k1  0    l·k1] [ω₂²]
	    [τ_y]       [l·k1  0   -l·k1  0  ] [ω₃²]
	    [τ_z]       [-k2  k2   -k2   k2 ] [ω₄²]

	where:
	- k1: Thrust coefficient (thrust per unit rotor speed squared)
	- k2: Drag coefficient (torque per unit rotor speed squared)
	- l: Arm length from center to rotor

	The hover thrust M·g is subtracted so control input F represents
	deviation from hover.

	Args (in param_dict):
		k1: Thrust coefficient
		k2: Drag coefficient
		l: quadcopter arm length [m]
		M: Total mass (quad + pendulum) [kg]

	Input:
		x: Batch of states (bs, 10) - not used, only for batch size

	Returns:
		Tensor (bs, 16, 4): Control polytope vertices for each state in batch
		                    Each vertex is [F, τ_x, τ_y, τ_z]

	Note:
		Vertices are precomputed in __init__ for efficiency and simply
		replicated across the batch dimension in forward().
	"""
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

		# Mixer matrix: rotor impulses → [F, τ_x, τ_y, τ_z]
		M = np.array([
			[k1,   k1,    k1,   k1],    # Total thrust
			[0,   -l*k1,  0,    l*k1],  # Roll torque
			[l*k1, 0,    -l*k1, 0],     # Pitch torque
			[-k2,  k2,   -k2,   k2]     # Yaw torque
		])

		# Generate all 16 vertices of the unit hypercube [0,1]^4
		# Each rotor can be at 0 (off) or ω_max (full thrust)
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
