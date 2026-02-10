"""Neural Control Barrier Function (CBF) implementation using modified CBF construction.

This module implements a variation of the neural CBF design described in Sections 2 and 3:

    φ*(x) = [Π_{i=0}^{r-1} (1 + c_i ∂^i/∂t^i) ρ(x)] + ρ(x) - ρ*
	ρ* = (nn(x) - nn(x_e))^2 + h * ρ(x)

where:
- ρ(x) is the base safety specification (e.g., angles from vertical)
- ρ*(x) is CBF reshaping function, where nn(x) is a neural network
- k_0, c_i are learnable positive coefficients
- r is the relative degree of the dynamic system
- x_e is an equilibrium state
"""
from typing import Optional
from collections.abc import Callable

import torch
from torch import nn
from torch.autograd import grad


class NeuralPhi(nn.Module):
	"""Neural Control Barrier Function (CBF)"""
	def __init__(self, rho_fn: Callable, xdot_fn: Callable, r: int, x_dim: int,
	             u_dim: int, device: torch.device, args,
	             nn_input_modifier: Optional[nn.Module] = None,
	             x_e: Optional[torch.Tensor] = None) -> None:
		"""Initializes Neural CBF with modified structure.

		Args:
			rho_fn: Base safety specification function ρ: R^x_dim → R
			xdot_fn: System dynamics f: R^x_dim × R^u_dim → R^x_dim
			r: Relative degree (typically 2 for control-affine systems)
			x_dim: State space dimension
			u_dim: Control input dimension
			device: PyTorch device (cuda or cpu)
			args: Training arguments namespace (for future extensions)
			nn_input_modifier: Optional transform applied before ρ* network
			                   (e.g., TransformEucNNInput for coordinate conversion)
			x_e: Optional equilibrium point. If provided, uses squared difference
			     ρ*(x) - ρ*(x_e) for better centering around equilibrium
		"""
		super().__init__()
		self.rho_fn = rho_fn
		self.xdot_fn = xdot_fn
		self.r = r
		self.x_dim = x_dim
		self.u_dim = u_dim
		self.device = device
		self.args = args
		self.nn_input_modifier = nn_input_modifier
		self.x_e = x_e
		assert r >= 0

		# Initialize ci, h parameters with small, positive values for training stability
		small_pos = 1e-2
		self.ci = nn.Parameter(small_pos*torch.rand(r-1, 1))
		self.h = nn.Parameter(small_pos*torch.rand(1, 1))

		# Minimum values to enforce strict positivity after projection
		# During training, parameters are clipped to max(param, param_min)
		self.pos_param_names = ["ci", "h"]  # Parameters that must remain positive (enforced in learner)
		self.exclude_from_gradient_param_names = ["ci", "h"]  # Parameters excluded from standard gradient norm logging
		self.ci_min = 1e-2
		self.h_min = 1e-2

		self.rho_star_net = self._create_net()  # nn(x)

	def _create_net(self) -> nn.Sequential:
		"""Creates MLP nn(x) for the reshaping term ρ*(x).

		Architecture: 2-layer MLP with tanh activations, softplus output
		- Hidden layers: 64-64
		- Activations: tanh-tanh-softplus
		- Input: Either raw state or transformed state (via nn_input_modifier)
		- Output: nn(x) >= 0
		"""
		hidden_dims = [64, 64, 1]
		phi_nnls = ["tanh", "tanh", "softplus"]

		# Determine input dimension
		if self.nn_input_modifier is None:
			prev_dim = self.x_dim  # Use raw state
		else:
			prev_dim = self.nn_input_modifier.output_dim  # Use transformed state

		# Build network layer by layer
		net_layers = []
		for hidden_dim, phi_nnl in zip(hidden_dims, phi_nnls):
			net_layers.append(nn.Linear(prev_dim, hidden_dim))
			# Add activation function
			if phi_nnl == "tanh":
				net_layers.append(nn.Tanh())
			elif phi_nnl == "softplus":
				net_layers.append(nn.Softplus())  # Output layer: ensures ρ*(x) ≥ 0
			prev_dim = hidden_dim

		net = nn.Sequential(*net_layers)
		return net

	def forward(self, x: torch.Tensor, grad_x: bool = False) -> torch.Tensor:
		"""Computes φ*(x) and all intermediate φ_i(x) values in a differentiable way.

		This implements the forward pass of the neural CBF, computing:
		1. ρ*(x): reshaping function, using nn(x)
		2. φ_i(x) for i=0..r-1, as defined in Appendix 9.
		3. Finally, φ*(x) = φ_{r-1}(x) - ρ(x) + ρ*(x).

		Args:
			x: Batch of states (bs, x_dim)
			grad_x: If True, preserves the requires_grad setting of x (caller manages gradients).
			        If False, temporarily enables gradients for Lie derivative computation.

		Returns:
			Tensor (bs, r+1) where:
			- Column i contains φ_i(x) for i=0..r-1
			- Column r contains φ*(x)
			The last column φ*(x) is the final CBF used for safety certification.

		Note:
			Batch-compliant implementation. All operations support arbitrary batch sizes.
		"""
		# Enforce strict positivity
		h = self.h + self.h_min
		ci = self.ci + self.ci_min

		# Convert c_i coefficients to k_i via binomial expansion
		ki = torch.tensor([[1.0]])
		ki_list = torch.zeros(self.r, self.r).to(self.device)  # Row i stores coefficients for φ_i
		ki_list[0, 0:ki.numel()] = ki

		for i in range(self.r-1):
			# Build convolution matrix for multiplication with [1, c_i]
			A = torch.zeros(torch.numel(ki)+1, 2)
			A[:-1, [0]] = ki  # Shifted by 0
			A[1:, [1]] = ki   # Shifted by 1

			# Binomial coefficients [1, c_i]
			# Important: Use tensor assignment to preserve gradient flow through c_i
			binomial = torch.ones((2, 1))
			binomial[1] = ci[i]

			# Compute k_{i+1} via convolution
			ki = A.mm(binomial)
			ki_list[i+1, 0:ki.numel()] = ki.view(1, -1)

		###############################################################################
		# Compute reshaped safety specification ρ'(x) = h·ρ(x) + ρ*(x)
		###############################################################################
		bs = x.size()[0]

		if not grad_x:
			orig_req_grad_setting = x.requires_grad
			x.requires_grad = True

		# Compute ρ*(x) using neural network
		if self.x_e is None:
			# ρ*(x) = softplus(nn(x)) + h·ρ(x)
			if self.nn_input_modifier is None:
				rho_star_value = self.rho_star_net(x)
			else:
				rho_star_value = self.rho_star_net(self.nn_input_modifier(x))
			new_rho = nn.functional.softplus(rho_star_value) + h*self.rho_fn(x)
		else:
			# This centers the modification around equilibrium x_e
			if self.nn_input_modifier is None:
				rho_star_value = self.rho_star_net(x)
				rho_star_xe_value = self.rho_star_net(self.x_e)
			else:
				rho_star_value = self.rho_star_net(self.nn_input_modifier(x))
				rho_star_xe_value = self.rho_star_net(self.nn_input_modifier(self.x_e))
			new_rho = torch.square(rho_star_value - rho_star_xe_value) + h*self.rho_fn(x)

		###############################################################################
		# Compute Lie derivatives L_f^i(ρ) for i=0..r-1
		###############################################################################
		# The k-th Lie derivative satisfies the recursion:
		#   L_f^0(ρ) = ρ(x)
		#   L_f^{k+1}(ρ) = ∇(L_f^k(ρ)) · f(x,0)
		#
		# This is computed using automatic differentiation (grad) with create_graph=True
		# to allow differentiation through this forward() pass.

		ith_lie_deriv = self.rho_fn(x)  # L_f^0(ρ) = ρ(x) (bs, 1)
		lie_derivs_list = ith_lie_deriv  # Accumulate all derivatives (bs, 1)

		# Evaluate drift dynamics f(x,0) - control set to zero for Lie derivatives
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device))  # (bs, x_dim)

		for i in range(self.r-1):
			# Compute gradient: ∇(L_f^i(ρ))
			ith_grad_rho = grad([torch.sum(ith_lie_deriv)], x, create_graph=True)[0]  # (bs, x_dim)

			#  L_f^{i+1}(ρ) =  ∇(L_f^i(ρ)) · f(x,0)
			ith_lie_deriv = (ith_grad_rho.unsqueeze(dim=1)).bmm(f_val.unsqueeze(dim=2))  # (bs, 1, 1)
			ith_lie_deriv = ith_lie_deriv[:, :, 0]  # (bs, 1)
			lie_derivs_list = torch.cat((lie_derivs_list, ith_lie_deriv), dim=1)  # (bs, i+2)

		if not grad_x:
			x.requires_grad = orig_req_grad_setting

		# Compute terms φ_i(x) = Σ_{j=0}^{i} k_{ij} · L_f^j(ρ) from Appendix 9
		result = lie_derivs_list.mm(ki_list.t())  # (bs, r)

		# Compute final CBF: φ*(x) = φ_{r-1}(x) - ρ(x) + ρ*(x)
		rho_star = result[:, [-1]] - result[:, [0]] + new_rho  # (bs, 1)

		# Return all φ_i for i=0..r-1 plus final φ*
		result = torch.cat((result, rho_star), dim=1)  # (bs, r+1)

		return result
