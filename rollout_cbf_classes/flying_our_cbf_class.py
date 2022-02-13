import scipy as sp
import numpy as np
import IPython
import torch
import math
from plot_utils import create_phi_struct_load_xlim
from torch.autograd import grad
from src.utils import *
from scipy.integrate import solve_ivp
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import pickle

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
        :param x: (N_batch, 10)
        :return: (N_batch, r+1) where r is degree
        """
        """# print("here")
        # IPython.embed()
        x = np.reshape(x, (-1, 4))
        # Numpy wrapper
        theta = self.convert_angle_to_negpi_pi_interval(x[:, 1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
        # assert theta < math.pi and theta > -math.pi
        x_trunc = np.concatenate((theta[:, None], x[:, [3]]), axis=1)
        x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)

        # IPython.embed()
        phi_output = self.torch_phi_fn(x_input)
        phi_vals = phi_output.detach().cpu().numpy()"""


        print("inside ourcbfclass, debug")
        IPython.embed()

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
        :param x: (10)
        :return: (10)
        """

        print("inside phi grad")
        IPython.embed()
        # Computes grad of phi at x
        # theta = self.convert_angle_to_negpi_pi_interval(x[1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
        # assert theta < math.pi and theta > -math.pi
        # x_trunc = np.array([theta, x[3]])
        # x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)
        # x_input.requires_grad = True

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

        # phi_grad = np.array([0, phi_grad[0, 0], 0, phi_grad[0, 1]])[None]
        return phi_grad

