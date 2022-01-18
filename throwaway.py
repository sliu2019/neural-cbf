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
import pickle

with open("./rollout_results/mpc_delta_0.050000_dt_0.050000_horizon_30.pkl", 'rb') as handle:
	data = pickle.load(handle)

	n_points = data["X"].size
	n_inside = np.sum(data["exists_soln_bools"])

	print(n_inside/n_points)