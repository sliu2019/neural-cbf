"""Neural Control Barrier Function (CBF) implementation using modified CBF construction.

This module implements a variation of the neural CBF design described in Sections 2 and 3:

    φ*(x) = [Π_{i=0}^{r-1} (1 + c_i ∂^i/∂t^i) ρ(x)] + ρ(x) - ρ* 
	ρ* = (nn(x) - nn(x_e))^2 + k_0 * ρ(x)

where:
- ρ(x) is the base safety specification (e.g., angles from vertical)
- ρ*(x) is CBF reshaping function, where nn(x) is a neural network
- k_0, c_i are learnable positive coefficients
- r is the relative degree of the dynamic system 
- x_e is an equilibrium state
"""
import torch
from torch import nn
from torch.autograd import grad

class NeuralPhi(nn.Module):
	"""Neural Control Barrier Function using modified CBF construction.

	Implements the neural CBF design from liu23e.pdf Section 3.1. The CBF uses
	a modified higher-order structure to handle input saturation:

	    φ*(x) = [Π_{i=0}^{r-1} (1 + c_i ∂^i/∂t^i) ρ(x)] + ρ(x) - ρ* 

	The network learns:
	- ρ*(x): Neural network (MLP) that reshapes the base safety specification
	- c_i: Positive coefficients that define the modified CBF structure
	- k0: Scaling coefficient for ρ(x)

	The forward pass computes all φ_i(x) for i=0..r, where φ_r(x) = φ*(x)
	is the final CBF used for safety certification.

	Attributes:
		rho_fn: Base safety specification ρ(x) (e.g., RhoSum for angle constraints)
		xdot_fn: System dynamics f(x,u) for computing Lie derivatives
		r: Relative degree of CBF (default: 2)
		x_dim: State space dimension
		u_dim: Control input dimension
		device: PyTorch device for computation
		args: Training arguments (currently unused)
		nn_input_modifier: Optional input transform (e.g., spherical to Euclidean)
		x_e: Optional equilibrium point for centering ρ* network
		ci: Learnable coefficients (r-1,) - must be positive
		k0: Learnable scaling (1,) - must be positive
		rho_star_net: MLP for ρ*(x)
		pos_param_names: List of parameters constrained to be positive
		exclude_from_gradient_param_names: Parameters excluded from gradient updates

	Note:
		Implementation is generic to any relative degree r, which may be slower
		than a hardcoded r=2 version but provides flexibility.
	"""
	def __init__(self, rho_fn, xdot_fn, r, x_dim, u_dim, device, args, nn_input_modifier=None, x_e=None):
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

		# Initialize learnable parameters for modified CBF structure
		# Small initialization helps stability during early training
		self.phi_ci_init_range = 1e-2

		# c_i coefficients: define the modified CBF structure (r-1 coefficients)
		# These are converted to k_i via binomial expansion (see forward())
		self.ci = nn.Parameter(self.phi_ci_init_range*torch.rand(r-1, 1))

		# k0: scaling coefficient for ρ(x) term
		self.k0 = nn.Parameter(self.phi_ci_init_range*torch.rand(1, 1))

		# Minimum values to enforce strict positivity after projection
		# During training, parameters are clipped to max(param, param_min)
		self.ci_min = 1e-2
		self.k0_min = 1e-2

		print("At initialization: k0 is %f" % self.k0.item())

		# Create neural network ρ*(x) for reshaping ρ(x)
		self.rho_star_net = self._create_net()

		# Specify which parameters must remain positive (enforced in learner)
		self.pos_param_names = ["ci", "k0"]
		# Parameters to exclude from standard gradient updates (if needed)
		self.exclude_from_gradient_param_names = ["ci", "k0"]

	def _create_net(self):
		"""Creates MLP for ρ*(x) that reshapes the base safety specification.

		Architecture: 2-layer MLP with tanh activations, softplus output
		- Hidden layers: 64-64
		- Activations: tanh-tanh-softplus
		- Input: Either raw state or transformed state (via nn_input_modifier)
		- Output: Scalar ρ*(x) ≥ 0 (softplus ensures positivity)

		The softplus output ensures ρ*(x) ≥ 0, which helps with optimization
		stability and interpretation (ρ* enlarges the safe set).

		Returns:
			nn.Sequential: MLP network for ρ*(x)
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

	def forward(self, x, grad_x=False):
		"""Computes modified CBF φ*(x) and all intermediate φ_i(x) values.

		This implements the forward pass of the neural CBF, computing:
		1. ρ*(x): Neural network output
		2. ρ'(x) = - k0·ρ(x) + ρ*(x): Reshaped safety specification
		3. L_f^i(ρ): Lie derivatives for i=0..r-1
		4. φ_i(x): Modified CBF terms for i=0..r

		The modified CBF structure from liu23e.pdf Appendix A.1:
		    φ_i(x) = Σ_{j=0}^{i} k_{ij} · L_f^j(ρ)(x)

		where k_{ij} are computed from c_i via binomial expansion to ensure
		smoothness and proper conditioning.

		Args:
			x: Batch of states (bs, x_dim)
			grad_x: If True, preserves requires_grad setting of x.
			       If False, temporarily enables gradients for Lie derivative computation.

		Returns:
			Tensor (bs, r+1) where:
			- Column i contains φ_i(x) for i=0..r-1
			- Column r contains φ*(x) = φ_{r-1}(x) - φ_0(x) + ρ'(x)
			The last column φ*(x) is the final CBF used for safety certification.

		Note:
			Batch-compliant implementation. All operations support arbitrary batch sizes.
		"""
		# Enforce positivity constraints on learned parameters
		k0 = self.k0 + self.k0_min  # Ensures k0 ≥ k0_min > 0
		ci = self.ci + self.ci_min  # Ensures all c_i ≥ ci_min > 0

		# Convert c_i coefficients to k_i via binomial expansion
		# Starting with k_0 = [1], we build k_1, k_2, ..., k_{r-1} # TODO: is k_0 = 1?
		ki = torch.tensor([[1.0]])
		ki_all = torch.zeros(self.r, self.r).to(self.device)  # Row i stores coefficients for φ_i
		ki_all[0, 0:ki.numel()] = ki

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
			ki_all[i+1, 0:ki.numel()] = ki.view(1, -1)

		###############################################################################
		# Compute reshaped safety specification ρ'(x) = k0·ρ(x) + ρ*(x)
		###############################################################################
		bs = x.size()[0]

		# Temporarily enable gradients on x if needed (for Lie derivative computation)
		if grad_x == False:
			orig_req_grad_setting = x.requires_grad
			x.requires_grad = True

		# Compute ρ*(x) using neural network
		if self.x_e is None:
			# Standard case: ρ*(x) directly
			if self.nn_input_modifier is None:
				rho_star_value = self.rho_star_net(x)
			else:
				rho_star_value = self.rho_star_net(self.nn_input_modifier(x))
			# Apply softplus for numerical stability (ρ* ≥ 0)
			new_rho = nn.functional.softplus(rho_star_value) + k0*self.rho_fn(x)
		else:
			# Equilibrium-centered case: (ρ*(x) - ρ*(x_e))^2
			# This centers the modification around equilibrium x_e
			if self.nn_input_modifier is None:
				rho_star_value = self.rho_star_net(x)
				rho_star_xe_value = self.rho_star_net(self.x_e)
			else:
				rho_star_value = self.rho_star_net(self.nn_input_modifier(x))
				rho_star_xe_value = self.rho_star_net(self.nn_input_modifier(self.x_e))
			new_rho = torch.square(rho_star_value - rho_star_xe_value) + k0*self.rho_fn(x)

		###############################################################################
		# Compute Lie derivatives L_f^i(ρ) for i=0..r-1
		###############################################################################
		# The k-th Lie derivative satisfies the recursion:
		#   L_f^0(ρ) = ρ(x)
		#   L_f^{k+1}(ρ) = ∇(L_f^k(ρ)) · f(x,0)
		#
		# This is computed using automatic differentiation (grad) with create_graph=True
		# to preserve gradient flow for the outer training loop.

		rho_itrho_deriv = self.rho_fn(x)  # L_f^0(ρ) = ρ(x) (bs, 1)
		rho_derivs = rho_itrho_deriv  # Accumulate all derivatives (bs, 1)

		# Evaluate drift dynamics f(x,0) - control set to zero for Lie derivatives
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device))  # (bs, x_dim)

		# Recursively compute higher-order Lie derivatives
		for i in range(self.r-1):
			# Compute gradient: ∇(L_f^i(ρ))
			# create_graph=True enables second-order gradients needed for training
			grad_rho_ith = grad([torch.sum(rho_itrho_deriv)], x, create_graph=True)[0]  # (bs, x_dim)

			# Take directional derivative along f: ∇(L_f^i(ρ)) · f(x,0)
			# This gives L_f^{i+1}(ρ)
			rho_itrho_deriv = (grad_rho_ith.unsqueeze(dim=1)).bmm(f_val.unsqueeze(dim=2))  # (bs, 1, 1)
			rho_itrho_deriv = rho_itrho_deriv[:, :, 0]  # (bs, 1)

			# Accumulate derivative
			rho_derivs = torch.cat((rho_derivs, rho_itrho_deriv), dim=1)  # (bs, i+2)

		# Restore original gradient setting
		if grad_x == False:
			x.requires_grad = orig_req_grad_setting

		###############################################################################
		# Compute modified CBF terms φ_i(x) = Σ_{j=0}^{i} k_{ij} · L_f^j(ρ)
		###############################################################################
		result = rho_derivs.mm(ki_all.t())  # (bs, r)

		# Compute final CBF: φ*(x) = φ_{r-1}(x) - φ_0(x) + ρ'(x)
		# This is the complete modified CBF from liu23e.pdf Eq. 2
		phi_r_minus_1_star = result[:, [-1]] - result[:, [0]] + new_rho  # (bs, 1)

		# Return all φ_i for i=0..r-1 plus final φ*
		result = torch.cat((result, phi_r_minus_1_star), dim=1)  # (bs, r+1)

		return result
