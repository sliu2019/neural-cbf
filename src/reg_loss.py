"""Volume regularization loss for neural CBF training; encourages neural CBF to have a large safe set."""
import torch
from torch import nn


class RegularizationLoss(nn.Module):
	"""Volume regularization loss to maximize the safe set.

	Penalizes states where max_i φ_i(x) is close to 0 (boundary), providing
	gradient signal to push the boundary outward and enlarge the safe set.

	The sigmoid transform provides:
	- States deep inside (max_i φ_i(x) << 0): little gradient
	- States near boundary (max_i φ_i(x) ≈ 0): large gradient
	- States outside (max_i φ_i(x) > 0): little gradient

	Attributes:
		phi_star_fn: Neural CBF function
		device: PyTorch device
		reg_weight: Loss coefficient (must be ≥ 0)
	"""
	def __init__(self, phi_star_fn: nn.Module, device: torch.device,
	             reg_weight: float = 0.0) -> None:
		super().__init__()
		self.phi_star_fn = phi_star_fn
		self.device = device
		self.reg_weight = reg_weight
		assert reg_weight >= 0.0

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Computes volume regularization loss.

		Args:
			x: Sampled states (bs, x_dim)

		Returns:
			Scalar regularization loss (positive, encourages large safe set)
		"""
		all_phi_values = self.phi_star_fn(x)  # (bs, r+1)

		# Safe set condition: max_i φ_i(x) ≤ 0
		max_phi_values = torch.max(all_phi_values, dim=1)[0]  # (bs,)

		transform_of_max_phi = torch.sigmoid(0.3*max_phi_values)  # (bs,)

		return self.reg_weight*torch.mean(transform_of_max_phi)
