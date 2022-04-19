import scipy as sp
import numpy as np
import IPython
import torch
import math
from torch.autograd import grad

class PhiNumpy:
    def __init__(self, torch_phi_fn):
        self.torch_phi_fn = torch_phi_fn

    def set_params(self, state_dict):
        # TODO: probably should have some checks on this
        self.torch_phi_fn.load_state_dict(state_dict, strict=False)

    def _convert_angle_to_negpi_pi_interval(self, angle):
        new_angle = np.arctan2(np.sin(angle), np.cos(angle))
        return new_angle
    
    def _x_numpy_to_x_torch(self, x):
        # Slice off translational states, if they are present
        if len(x.shape) == 1:
            x = np.reshape(x, (1, -1))
        x = x[:, :10]

        # Wrap-around on cyclical angles
        ind_cyclical = [0, 1, 2, 6, 7]
        for i in ind_cyclical:
            x[:, i] = self._convert_angle_to_negpi_pi_interval(x[:, i])

        x_torch = torch.from_numpy(x.astype("float32"))
        # Q: how come we don't have to involve device = gpu?
        # A: because it is set as CPU elsewhere? Yes

        return x_torch

    def phi_fn(self, x): # Batched
        """
        :param x: (N_batch, 16)
        :return: (N_batch, r+1) where r is degree
        """

        x_torch = self._x_numpy_to_x_torch(x)
        phi_torch = self.torch_phi_fn(x_torch)
        phi_numpy = phi_torch.detach().cpu().numpy()

        return phi_numpy

    def phi_grad(self, x): # Not batched
        """
        :param x: (16)
        :return: (16)
        """
        x_torch = self._x_numpy_to_x_torch(x)
        x_torch.requires_grad = True

        # Compute phi grad
        phi_vals = self.torch_phi_fn(x_torch)
        phi_val = phi_vals[0, -1]
        phi_grad = grad([phi_val], x_torch)[0]

        # Post op
        x_torch.requires_grad = False
        phi_grad = phi_grad.detach().cpu().numpy().flatten()
        phi_grad = np.concatenate((phi_grad, np.zeros(6)))

        return phi_grad

