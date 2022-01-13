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
    def __init__(self, exp_name, checkpoint_number):
        super().__init__()
        variables = locals()  # dict of local names
        self.__dict__.update(variables)  # __dict__ holds and object's attributes
        del self.__dict__["self"]  # don't need `self`

        # print("here")
        self.torch_phi_fn, self.x_lim = create_phi_struct_load_xlim(exp_name, checkpoint_number)
        phi_load_fpth = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
        load_model(self.torch_phi_fn, phi_load_fpth)
        # print("actually here")

    def convert_angle_to_negpi_pi_interval(self, angle):
        new_angle = np.arctan2(np.sin(angle), np.cos(angle))
        return new_angle 

    def phi_fn(self, x): # Batched
        """
        :param x: (N_batch, 4)
        :return: (N_batch, r+1) where r is degree
        """
        # print("here")
        # IPython.embed()
        x = np.reshape(x, (-1, 4))
        # Numpy wrapper
        theta = self.convert_angle_to_negpi_pi_interval(x[:, 1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
        # assert theta < math.pi and theta > -math.pi
        x_trunc = np.concatenate((theta[:, None], x[:, [3]]), axis=1)
        x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)

        # IPython.embed()
        phi_output = self.torch_phi_fn(x_input)
        phi_vals = phi_output.detach().cpu().numpy()
        return phi_vals

    def phi_grad(self, x): # Not batched
        """
        :param x: (4)
        :return: (4)
        """
        # IPython.embed()
        # Computes grad of phi at x
        theta = self.convert_angle_to_negpi_pi_interval(x[1]) # Note: mod theta first, before applying cbf. Also, truncate the state.
        assert theta < math.pi and theta > -math.pi
        x_trunc = np.array([theta, x[3]])
        x_input = torch.from_numpy(x_trunc.astype("float32")).view(-1, 2)
        x_input.requires_grad = True

        # Compute phi grad
        phi_vals = self.torch_phi_fn(x_input)
        phi_val = phi_vals[0, -1]
        phi_grad = grad([phi_val], x_input)[0]

        # Post op
        x_input.requires_grad = False
        phi_grad = phi_grad.detach().cpu().numpy()
        phi_grad = np.array([0, phi_grad[0, 0], 0, phi_grad[0, 1]])[None]

        return phi_grad

