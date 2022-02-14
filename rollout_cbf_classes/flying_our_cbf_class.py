import scipy as sp
import numpy as np
import IPython
import torch
import math
from torch.autograd import grad

class OurCBF:
    def __init__(self, torch_phi_fn, param_dict):
        self.torch_phi_fn = torch_phi_fn
        self.param_dict = param_dict
        self.__dict__.update(self.param_dict)

    def convert_angle_to_negpi_pi_interval(self, angle):
        new_angle = np.arctan2(np.sin(angle), np.cos(angle))
        return new_angle

    def phi_fn(self, x): # Batched
        """
        :param x: (N_batch, 16)
        :return: (N_batch, r+1) where r is degree
        """
        print("inside ourcbfclass, phi_fn")
        IPython.embed()
        # Slice off translational states
        x = x[:, :10]
        x = np.reshape(x, (-1, self.x_dim))
        # Wrap-around on cyclical angles
        ind_cyclical = np.argwhere(self.x_lim[:, 1] == math.pi).flatten()
        for i in ind_cyclical:
            x[:, i] = self.convert_angle_to_negpi_pi_interval(x[:, i])

        # Warn if x outside of state box of phi
        if np.any(np.logical_or(x < self.x_lim[:, 0], x > self.x_lim[:, 1])):
            print("WARNING: evaluating phi at x outside of state box \n")

        # To torch
        x_torch = torch.from_numpy(x.astype("float32"))
        phi_torch = self.torch_phi_fn(x_torch)
        phi_numpy = phi_torch.detach().cpu().numpy()

        return phi_numpy

    def phi_grad(self, x): # Not batched
        """
        :param x: (16)
        :return: (16)
        """
        print("inside ourcbf, phi grad")
        IPython.embed()

        x = x[:, :10]
        x = np.reshape(x, (-1, self.x_dim))
        # Wrap-around on cyclical angles
        ind_cyclical = np.argwhere(self.x_lim[:, 1] == math.pi).flatten()
        for i in ind_cyclical:
            x[:, i] = self.convert_angle_to_negpi_pi_interval(x[:, i])

        # Warn if x outside of state box of phi
        if np.any(np.logical_or(x < self.x_lim[:, 0], x > self.x_lim[:, 1])):
            print("WARNING: evaluating phi at x outside of state box \n")

        x_torch = torch.from_numpy(x.astype("float32"))
        x_torch.requires_grad = True

        # Compute phi grad
        phi_vals = self.torch_phi_fn(x_torch)
        phi_val = phi_vals[0, -1]
        phi_grad = grad([phi_val], x_torch)[0]

        # Post op
        x_torch.requires_grad = False
        phi_grad = phi_grad.detach().cpu().numpy()
        phi_grad = np.concatenate((phi_grad, np.zeros(6)))
        # phi_grad = np.array([0, phi_grad[0, 0], 0, phi_grad[0, 1]])[None]

        return phi_grad

