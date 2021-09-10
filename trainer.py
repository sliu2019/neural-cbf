import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
from attacks.basic import BasicAttacker


"""class Trainer:
	init():
		In:
			Attack class object
			(Later) optimization parameters

	train():
		In:
			phi_i (model)
			x_dot
			X_limit, objective function (U_limit implicitly)
		Fn:
			<Anything meta-training, mostly boilerplate:
				1. Training loop: bi-step (inner and outer)
				2. Saving model, computing and logging progress (call test)

	test():
		Either GD until (near) conv or MCMC"""

