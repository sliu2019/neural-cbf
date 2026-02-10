"""Saturation risk loss for neural CBF training.
The critic maximizes this to find worst-case states; the learner minimizes it to fix violations.
"""
import logging
from collections.abc import Callable

import torch
from torch import nn
from torch.autograd import grad

class SaturationRisk(nn.Module):
	"""Computes worst-case CBF derivative over control limit set vertices.

	For each state x, evaluates φ̇(x, u) = ∇φ(x)·f(x,u) at all vertices
	of the polytopic control limit set U and returns the minimum (worst case).

	A negative minimum means no saturated control can make φ̇ ≥ 0, i.e.,
	the CBF forward invariance condition is violated at x.

	Attributes:
		phi_fn: Neural CBF returning (bs, r+1) with φ* in last column
		xdot_fn: System dynamics f(x, u) returning (bs, x_dim)
		uvertices_fn: Returns control polytope vertices (bs, n_vertices, u_dim)
		x_dim: State space dimension
		u_dim: Control input dimension
		device: PyTorch device
		logger: Logger instance
		args: Training arguments namespace
	"""
	def __init__(self, phi_fn: nn.Module, xdot_fn: Callable, uvertices_fn: Callable,
	             x_dim: int, u_dim: int, device: torch.device,
	             logger: logging.Logger, args) -> None:
		super().__init__()
		self.phi_fn = phi_fn
		self.xdot_fn = xdot_fn
		self.uvertices_fn = uvertices_fn
		self.x_dim = x_dim
		self.u_dim = u_dim
		self.device = device
		self.logger = logger
		self.args = args

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Computes minimum CBF derivative over control limit vertices.

		Approach:
		1. Get all vertices of the control limit polytope U
		2. Compute φ̇(x, u) = ∇φ(x) · f(x, u) for each vertex u
		3. Return minimum over all vertices

		Args:
			x: Batch of states (bs, x_dim)

		Returns:
			Tensor (bs, 1) with minimum φ̇ values (saturation risk)
		"""
		# Get vertices of control limit set (polytope with n_vertices corners)
		u_lim_set_vertices = self.uvertices_fn(x)  # (bs, n_vertices, u_dim)
		n_vertices = u_lim_set_vertices.size()[1]

		# Reshape to evaluate all (state, control) pairs in parallel
		U = torch.reshape(u_lim_set_vertices, (-1, self.u_dim))  # (bs*n_vertices, u_dim)
		X = (x.unsqueeze(1)).repeat(1, n_vertices, 1)  # (bs, n_vertices, x_dim)
		X = torch.reshape(X, (-1, self.x_dim))  # (bs*n_vertices, x_dim)

		# Compute state derivatives f(x, u) for all combinations
		xdot = self.xdot_fn(X, U)  # (bs*n_vertices, x_dim)

		# Compute gradient of CBF: ∇φ(x)
		orig_req_grad_setting = x.requires_grad 
		x.requires_grad = True # So gradient flows through this forward pass 
		phi_value = self.phi_fn(x)  # (bs, r+1)
		grad_phi = grad([torch.sum(phi_value[:, -1])], x, create_graph=True)[0]  # (bs, x_dim)
		x.requires_grad = orig_req_grad_setting

		# Broadcast gradient to match all vertex evaluations
		grad_phi = (grad_phi.unsqueeze(1)).repeat(1, n_vertices, 1)  # (bs, n_vertices, x_dim)
		grad_phi = torch.reshape(grad_phi, (-1, self.x_dim))  # (bs*n_vertices, x_dim)

		# Compute Lie derivative: φ̇ = ∇φ · f(x, u) for all vertices
		phidot_cand = xdot.unsqueeze(1).bmm(grad_phi.unsqueeze(2))  # (bs*n_vertices, 1, 1)
		phidot_cand = torch.reshape(phidot_cand, (-1, n_vertices))  # (bs, n_vertices)

		# Take minimum over all control vertices
		phidot, _ = torch.min(phidot_cand, 1)  # (bs,)

		return phidot.view(-1, 1)  # (bs, 1)
