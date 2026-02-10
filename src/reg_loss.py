"""Volume regularization loss for neural CBF training.

Implements the regularization term from liu23e.pdf Eq. 7 that encourages
the neural CBF to produce a large safe set. The loss is:

    L_reg(x) = weight · mean(sigmoid(0.3 · max_i φ_i(x)))

where states near the boundary (φ ≈ 0) contribute more gradient signal
than states deep inside or outside the safe set.

References:
    liu23e.pdf Eq. 7, Section 3.3
"""
import torch
from torch import nn


class RegularizationLoss(nn.Module):
	"""Volume regularization loss to maximize the safe set.

	Penalizes states where max_i φ_i(x) is close to 0 (boundary), providing
	gradient signal to push the boundary outward and enlarge the safe set.

	The sigmoid transform σ(0.3·φ) provides:
	- States deep inside (φ << 0): σ ≈ 0, little gradient
	- States near boundary (φ ≈ 0): σ ≈ 0.5, large gradient
	- States outside (φ > 0): σ ≈ 1, little gradient

	Attributes:
		phi_fn: Neural CBF function
		device: PyTorch device
		reg_weight: Loss coefficient (must be ≥ 0)
	"""
	def __init__(self, phi_fn: nn.Module, device: torch.device,
	             reg_weight: float = 0.0) -> None:
		super().__init__()
		self.phi_fn = phi_fn
		self.device = device
		self.reg_weight = reg_weight
		assert reg_weight >= 0.0

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Computes volume regularization loss.

		Args:
			x: Sampled states (bs, x_dim), typically from ρ(x) ≤ 0 region

		Returns:
			Scalar regularization loss (positive, encourages large safe set)
		"""
		all_phi_values = self.phi_fn(x)  # (bs, r+1)

		# Safe set condition: max_i φ_i(x) ≤ 0
		max_phi_values = torch.max(all_phi_values, dim=1)[0]  # (bs,)

		transform_of_max_phi = torch.sigmoid(0.3*max_phi_values)  # (bs,)

		return self.reg_weight*torch.mean(transform_of_max_phi)
