"""Flying inverted pendulum system dynamics and safety specifications.

This module implements the flying inverted pendulum system studied in
"Safe control under input limits with neural control barrier functions"
(liu23e, CoRL 2022). The system consists of:

1. **Quadrotor**: A 3D quadrotor with orientation angles (γ, β, α) representing
   roll, pitch, and yaw. The quadrotor provides thrust and torques.

2. **Inverted Pendulum**: A pendulum attached below the quadrotor with angles
   (φ, θ) representing swing angles from vertical in two axes.

State: x = [γ, β, α, γ̇, β̇, α̇, φ, θ, φ̇, θ̇] ∈ R^10
  - γ, β, α: Quadrotor roll, pitch, yaw
  - φ, θ: Pendulum angles from vertical
  - Velocities: Time derivatives of angles

Control: u = [F, τ_x, τ_y, τ_z] ∈ R^4
  - F: Total thrust (hover thrust subtracted)
  - τ_x, τ_y, τ_z: Torques about body axes

The safety objective is to keep the pendulum upright: |φ|, |θ| small.

Physical Model:
- Quadrotor mass: m ≈ 0.8 kg
- Pendulum mass: m_p ≈ 0.04 kg (5% of quadrotor)
- Pendulum length: L_p ≈ 3.0 m
- Moments of inertia: J_x, J_y, J_z
- Mixer matrix parameters: k1 (thrust-to-rotor), k2 (drag-to-rotor), l (arm length)

References:
    liu23e.pdf Section 5 (Flying Inverted Pendulum Experiment)
"""
import torch
import numpy as np

from torch import nn
import os, sys
import IPython
import math

g = 9.81  # Gravitational acceleration [m/s^2]
# class RhoMax(nn.Module):
# 	def __init__(self, param_dict):
# 		super().__init__()
# 		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
# 		self.i = self.state_index_dict

# 	def forward(self, x):
# 		# The way these are implemented should be batch compliant
# 		# Return value is size (bs, 1)

# 		# print("Inside HMax forward")
# 		# IPython.embed()
# 		theta = x[:, [self.i["theta"]]]
# 		phi = x[:, [self.i["phi"]]]
# 		gamma = x[:, [self.i["gamma"]]]
# 		beta = x[:, [self.i["beta"]]]

# 		cos_cos = torch.cos(theta)*torch.cos(phi)
# 		eps = 1e-4 # prevents nan when cos_cos = +/- 1 (at x = 0)
# 		with torch.no_grad():
# 			signed_eps = -torch.sign(cos_cos)*eps
# 		delta = torch.acos(cos_cos + signed_eps)
# 		rv = torch.maximum(torch.maximum(delta**2, gamma**2), beta**2) - self.delta_safety_limit**2
# 		return rv

class RhoSum(nn.Module):
	"""Base safety specification ρ(x) for flying inverted pendulum.

	Computes the sum-of-squares safety specification:
	    ρ(x) = δ² + γ² + β² - δ_limit²

	where:
	- δ: Angle between pendulum axis and vertical (computed from φ, θ)
	- γ, β: Quadrotor roll and pitch angles
	- δ_limit: Maximum allowed deviation from vertical (e.g., π/4)

	The safe set is defined as {x : ρ(x) ≤ 0}, meaning all angles
	remain within their safety limits. This sum-of-squares form ensures
	a convex safe region.

	Angle Computation:
	The pendulum deviation angle δ satisfies:
	    cos(δ) = cos(φ) · cos(θ)

	A small epsilon is added for numerical stability when cos(δ) ≈ ±1.

	Args (in param_dict):
		delta_safety_limit: Maximum allowed angle from vertical [radians]
		state_index_dict: Mapping from state names to indices in x

	Input:
		x: Batch of states (bs, 10)

	Returns:
		ρ(x): Safety specification values (bs, 1)
		      Negative values indicate safe states
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
		gamma = x[:, [self.i["gamma"]]]  # Quadrotor roll
		beta = x[:, [self.i["beta"]]]    # Quadrotor pitch

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
	"""System dynamics ẋ = f(x, u) for flying inverted pendulum.

	Computes the time derivatives of all state variables using:
	1. Quadrotor rigid body dynamics (rotation matrix formulation)
	2. Pendulum dynamics (coupled to quadrotor through thrust vector)

	The dynamics are derived from Euler-Lagrange equations with the following
	key couplings:
	- Quadrotor thrust affects pendulum acceleration via thrust direction
	- Pendulum motion does NOT feedback to quadrotor (underactuated assumption)

	State Derivatives:
	    ẋ = [γ̇, β̇, α̇, γ̈, β̈, α̈, φ̇, θ̇, φ̈, θ̈]ᵀ

	where:
	- Quadrotor angular accelerations: Computed from torques via rotation matrix R
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
		M: Total mass (quadrotor + pendulum) [kg]
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
		gamma = x[:, self.i["gamma"]]   # Quadrotor roll
		beta = x[:, self.i["beta"]]     # Quadrotor pitch
		alpha = x[:, self.i["alpha"]]   # Quadrotor yaw

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
		# Compute quadrotor angular accelerations from torques
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
		# Compute pendulum angular accelerations (coupled to quadrotor via thrust)
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

	For the flying inverted pendulum, the control limits come from physical
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
		l: Quadrotor arm length [m]
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

if __name__ == "__main__":
	# param_dict = {
	# 	"m": 0.8,
	# 	"J_x": 0.005,
	# 	"J_y": 0.005,
	# 	"J_z": 0.009,
	# 	"l": 1.5,
	# 	"k1": 4.0,
	# 	"k2": 0.05,
	# 	"m_p": 0.04, # TODO?
	# 	"L_p": 0.03, # TODO?
	# 	'delta_safety_limit': math.pi/5 # in radians; should be <= math.pi/4
	# }
	# param_dict["M"] = param_dict["m"] + param_dict["m_p"]
	#
	# state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta"] # excluded x, y, z
	# state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
	# param_dict["state_index_dict"] = state_index_dict
	# ##############################################
	#
	# if torch.cuda.is_available():
	# 	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	# 	dev = "cuda:%i" % (0)
	# 	print("Using GPU device: %s" % dev)
	# else:
	# 	dev = "cpu"
	# device = torch.device(dev)
	#
	# # h_fn = H(param_dict)
	# xdot_fn = XDot(param_dict, device)
	# uvertices_fn = ULimitSetVertices(param_dict, device)
	#
	# N = 10
	# x = torch.rand(N, 10).to(device)
	# u = torch.rand(N, 4).to(device)
	#
	# uvert = uvertices_fn(x)

	# IPython.embed()
	# h_fn = HMax(param_dict)
	# h_fn = HSum(param_dict)
	# rv1 = h_fn(x)
	# print(rv1.shape)

	# x = torch.zeros(1, 10).to(device)
	# u = torch.zeros(1, 4).to(device)
	# rv2 = xdot_fn(x, u)
	# IPython.embed()
	# rv3 = uvertices_fn(x)

	# print(rv2.shape)
	# print(rv3.shape)
	# print(rv1.shape)
	# IPython.embed()

	"""param_dict = {
		"m": 0.8,
		"J_x": 0.005,
		"J_y": 0.005,
		"J_z": 0.009,
		"l": 1.5,
		"k1": 4.0,
		"k2": 0.05,
		"m_p": 0.04, # 5% of quad weight
		"L_p": 3.0, # Prev: 0.03
		'delta_safety_limit': math.pi / 4  # should be <= math.pi/4
	}
	param_dict["M"] = param_dict["m"] + param_dict["m_p"]
	state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
	                     "dtheta"]  # excluded x, y, z
	state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
	param_dict["state_index_dict"] = state_index_dict

	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'
		dev = "cuda:%i" % (0)
		print("Using GPU device: %s" % dev)
	else:
		dev = "cpu"
	device = torch.device(dev)

	xdot_fn = XDot(param_dict, device)

	np.random.seed(3)
	x = np.random.rand(16)
	u = np.random.rand(4)

	x = x[:10]
	x = torch.from_numpy(x.astype("float32")).to(device)
	u = torch.from_numpy(u.astype("float32")).to(device)

	x = x.view(1, -1)
	u = u.view(1, -1)

	# N = 10
	# x = torch.rand(N, 10).to(device)
	# u = torch.rand(N, 4).to(device)

	xdot = xdot_fn(x, u)

	print(x, u)
	print(xdot)"""
	pass

	# comparing results to batched numpy

