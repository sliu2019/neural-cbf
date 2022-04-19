import scipy as sp
import numpy as np
import IPython
import torch
import math
from torch.autograd import grad

# TODO: how come we don't have to involve device = gpu?
# TODO: because it is set as CPU elsewhere?

class PhiNumpy:
    def __init__(self, torch_phi_fn):
        print("inside init of PhiNumpy")
        print("don't [pass torch_phi_fn as first argument! Pass the whole torch class instantiation")
        IPython.embed()
        # self.torch_phi_fn = torch_phi_fn
        self.torch_phi_fn = torch_phi_fn
        # self.param_dict = param_dict
        # self.__dict__.update(self.param_dict)
        # self.state_index_list = list(self.state_index_dict.keys())

    def set_params(self, state_dict):
        print("inside set_params of PhiNumpy")
        print("don't pass params as a list, pass it as a state dict")
        IPython.embed()

        self.torch_phi_fn.load_state_dict(state_dict, strict=False)

    def _convert_angle_to_negpi_pi_interval(self, angle):
        new_angle = np.arctan2(np.sin(angle), np.cos(angle))
        return new_angle
    
    def _x_numpy_to_x_torch(self, x):
        # Slice off translational states
        x = np.reshape(x, (-1, 16))
        x = x[:, :10]

        # Wrap-around on cyclical angles
        ind_cyclical = [0, 1, 2, 6, 7]
        for i in ind_cyclical:
            x[:, i] = self._convert_angle_to_negpi_pi_interval(x[:, i])

        x_torch = torch.from_numpy(x.astype("float32"))

        # Warn if x outside of state box of phi
        # TODO: is this actually being used or is it redundant?
        """if bs == 1: # NOTE: ONLY PRINTS WHEN RUNNING 1 ROLLOUT AT A TIME
            outside_box = np.logical_or(x < self.x_lim[:, 0], x > self.x_lim[:, 1]).flatten()
            if np.any(outside_box):
                print("WARNING: phi_fn( . ) evaluating phi at x outside of state box")
                ind_outside_box = np.argwhere(outside_box).flatten()
                for ind in ind_outside_box:
                    print(self.state_index_list[ind], x[0, ind])
                # print("States: " + ", ".join(state_vars_outside_box))"""

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

