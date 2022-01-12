import torch
import numpy as np

from torch import nn
from torch.autograd import grad
from src.attacks.basic_attacker import BasicAttacker
from src.attacks.gradient_batch_attacker import GradientBatchAttacker
from src.trainer import Trainer
from src.utils import *
from scipy.linalg import pascal
from src.argument import parser, print_args
import os, sys
import math
import IPython
import time
from global_settings import *

"""
Implements a baseline CBF 
"""
class PhiBaseline(nn.Module):
	# Note: currently, we have a implementation which is generic to any r. May be slow

	def __init__(self, h_fn, ci, beta, xdot_fn, r, x_dim, u_dim, device):
		# Later: args specifying how beta is parametrized
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		assert r>=0
		assert self.beta>=0
		assert np.all(np.array(ci)>0)


	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		beta_value = self.beta

		# Convert ci to ki
		ki = torch.tensor([[1.0]])
		ki_all = torch.zeros(self.r, self.r).to(self.device) # phi_i coefficients are in row i
		ki_all[0, 0:ki.numel()] = ki
		for i in range(self.r-1): # A is current coeffs
			A = torch.zeros(torch.numel(ki)+1, 2)
			A[:-1, [0]] = ki
			A[1:, [1]] = ki

			# Note: to preserve gradient flow, have to assign mat entries to ci not create with ci (i.e. torch.tensor([ci[0]]))
			binomial = torch.ones((2, 1))
			binomial[1] = self.ci[i]
			ki = A.mm(binomial)

			ki_all[i+1, 0:ki.numel()] = ki.view(1, -1)
		# Ultimately, ki should be r x 1

		# Compute higher-order Lie derivatives
		bs = x.size()[0]
		orig_req_grad_setting = x.requires_grad # Basically only useful if x.requires_grad was False before
		x.requires_grad = True

		h_ith_deriv = self.h_fn(x) # bs x 1, the zeroth derivative
		h_derivs = h_ith_deriv # bs x 1
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device)) # bs x x_dim

		for i in range(self.r-1):
			# print(h_ith_deriv.size())
			grad_h_ith = grad([torch.sum(h_ith_deriv)], x, create_graph=True)[0] # bs x x_dim; create_graph ensures gradient is computed through the gradient operation

			# IPython.embed()
			h_ith_deriv = (grad_h_ith.unsqueeze(dim=1)).bmm(f_val.unsqueeze(dim=2)) # bs x 1 x 1
			h_ith_deriv = h_ith_deriv[:, :, 0] # bs x 1

			# print(h_ith_deriv)
			h_derivs = torch.cat((h_derivs, h_ith_deriv), dim=1)

		x.requires_grad = orig_req_grad_setting
		# Old:
		# result = beta_value + h_derivs.mm(ki) # bs x 1
		# New: (bs, r+1)
		result = h_derivs.mm(ki_all.t())
		phi_r_minus_1_star = result[:, [-1]] - result[:, [0]] + beta_value
		result = torch.cat((result, phi_r_minus_1_star), dim=1)

		return result

if __name__=="__main__":
	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = "0"
		dev = "cuda:0"
		print("Using GPU device: %s" % dev)
	else:
		dev = "cpu"
	device = torch.device(dev)

	# TODO
	physical_difficulty = "easy"

	r = 2
	x_dim = 2
	u_dim = 1
	x_lim = np.array([[-math.pi, math.pi], [-5, 5]], dtype=np.float32)

	# Create phi
	from src.problems.cartpole_reduced import H, XDot, ULimitSetVertices

	if physical_difficulty == 'easy':
		param_dict = {
			"I": 0.021,
			"m": 0.25,
			"M": 1.00,
			"l": 0.5,
			"max_theta": math.pi / 2.0,
			"max_force": 15.0
		}
	elif physical_difficulty == 'hard':
		param_dict = {
			"I": 0.021,
			"m": 0.25,
			"M": 1.00,
			"l": 0.5,
			"max_theta": math.pi / 4.0,
			"max_force": 1.0
		}

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict)

	# ci = [2.0] # TODO
	# beta = param_dict["max_theta"] - 0.1 # TODO
	# phi_baseline = PhiBaseline(h_fn, ci, beta, xdot_fn, r, x_dim, u_dim, device)
	# save_model(phi_baseline, "./checkpoint/cartpole_baseline_cbf/checkpoint_0.pth")

	# load_model()
