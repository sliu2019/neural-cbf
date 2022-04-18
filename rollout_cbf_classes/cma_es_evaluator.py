import numpy as np
import matplotlib.pyplot as plt
from rollout_envs.cart_pole_env import CartPoleEnv
from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
import seaborn as sns
from rollout_cbf_classes.deprecated.normal_ssa_newsi import SSA
from rollout_cbf_classes.deprecated.flying_pend_ssa import FlyingPendSSA
# from .normal_ssa import SSA

class Evaluator(object):
    # Each evaluator must contain two major functions: set_params(params) and evaluate(params)
    def set_params(self, params):
        # Set the params to the desired component
        raise NotImplementedError
    
    def evaluate(self, params):
        # Execute and collect reward.
        raise NotImplementedError

    def visualize(self, final_params):
        # visualize the learned final params.
        pass

    @property
    def log(self):
        return ""