import numpy as np
import time
import subprocess
import sys, os
import copy
import matplotlib.pyplot as plt
import math
from rollout_cbf_classes.cart_pole_env import CartPoleEnv
import seaborn as sns
import IPython

# TODO: check if phi_fn, phi_grad are correct.
# TODO: check if evaluate is correct. Add reg term

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

class CartPoleEvaluator(object):
    def __init__(self):

        self.env = CartPoleEnv()

        self.coe = np.zeros(4) # parameters; initialization doesn't matter
        """
        What's in self.coe?
        0: power
        1: weight on dot_theta term 
        2: additive scalar
        3: weight dotphi < -weight*phi
        """

        # Sample state space
        n_theta = 200 # TODO
        n_ang = 400 # TODO

        # delta = 0.01
        # self.thetas = np.arange(-self.env.max_theta, self.env.max_theta, delta)
        # self.ang_vels = np.arange(-self.env.max_angular_velocity, self.env.max_angular_velocity, delta)

        self.thetas = np.linspace(0, self.env.max_theta, n_theta)
        self.ang_vels = np.linspace(-self.env.max_angular_velocity, self.env.max_angular_velocity, n_ang)
        m1, m2 = np.meshgrid(self.thetas, self.ang_vels)

        self.samples = []
        for i,j in np.ndindex(m1.shape):
            self.samples.append([j,i])

        # Defining some var
        self.max_u = np.vstack([self.env.max_force])
        self.dt = self.env.dt

        self.reg_weight = 1.0 # TODO: need to tune to get possible result

        self.k = 5.0 # TODO: fixed, not learning it. For ease of comparison
        # self.d_min = 1
        
    def phi_fn(self, x):
        """
        :param x: (N_batch, 4)
        :return: (N_batch, r+1) where r is degree
        """
        # batched
        x = np.reshape(x, (-1, 4))
        # x =  [x theta, dot_x, dot_theta]
        # phi = theta ** a1 - theta_max ** a1 + a2 * dot_theta + a3
        # phi_0 = theta ** 2 - theta_max ** 2

        # invariant set: (phi_(k+1) < 0 or dot_phi_k < 0 (with a valid control)) and phi_0 < 0 # TODO: Simin, this may not be correct.
        theta = x[:, 1]
        dot_theta = x[:, 3]

        phi_0 = theta**2 - self.env.theta_safe_lim**2
        phi_1 = phi_0 + self.coe[1]*theta*dot_theta
        phi = (theta**2)**(self.coe[0]/2.0) - self.env.theta_safe_lim ** self.coe[0] + self.coe[2] + self.coe[1] * theta*dot_theta # Note: x2/2 is a hack!!!

        phi_0 = np.reshape(phi_0, (-1, 1))
        phi_1 = np.reshape(phi_1, (-1, 1))
        phi = np.reshape(phi, (-1, 1))
        phis = np.concatenate((phi_0, phi_1, phi), axis=1)
        return phis

    def phi_grad(self, x):
        """
        :param x: (4)
        :return: (4)
        """
        # not batched
        theta = x[1]
        dot_theta = x[3]
        theta_sq = x[1]**2 # Note: hack
        rv = np.hstack([0, self.coe[0] * (theta_sq ** ((self.coe[0]-1)/2.0)) + self.coe[1]*dot_theta, 0, self.coe[1]*theta])

        # if np.any(np.isnan(rv)):
        #     print("Inside cma-es evaluator, numpy is nan")
        #     IPython.embed()
        return rv

    def set_params(self, params):
        self.coe = params

    def has_valid_control(self, C, d, x):
        # dot_x = f + g * u 
        f = np.vstack(self.env.x_dot_open_loop(x, 0))
        g = np.vstack(self.env.x_dot_open_loop(x, 1)) - f
        return (C @ f + C @ g @ self.max_u < d) or (C @ f - C @ g @ self.max_u < d)

    def evaluate(self, params):
        # self.set_params(params)
        # valid = 0

        # in_invariant = 0
        # for sample in self.samples:
        #     idx = sample
        #     x = [0, self.thetas[idx[0]], 0, self.ang_vels[idx[1]]]
        #     phis = self.phi_fn(x)
        #     phi = phis[0, -1]
        #     # C dot_x < d
        #     # phi(x_k) > 0 -> con: dot_phi(x_k) < -k*phi(x): C = self.grad_phi(x, o)  d = -phi*self.coe[3]
        #     # phi(x_k) < 0 -> con: phi(x_k+1) < 0:           C = self.grad_phi(x, o)  d = -phi/dt
        #     C = self.phi_grad(x)
        #     d = -phi/self.dt if phi < 0 else -phi*self.k
        #     has_valid = self.has_valid_control(C, d, x)
        #     valid += has_valid

        #     if np.max(phis) <= 0 and has_valid:
        #         in_invariant += 1

        # self.valid = valid
        # valid_rate = valid * 1.0 / len(self.samples)

        # #### Reg term ####
        # in_invariant_rate = float(in_invariant)/len(self.samples)

        # # Log
        # self.valid_rate = valid_rate
        # self.in_invariant_rate = in_invariant_rate

        # rv = valid_rate + self.reg_weight*in_invariant_rate
        # return rv

        # TODO: special objective on the boundary

        self.set_params(params)
        valid = 0
        on_bdry = 0
        
        in_invariant = 0
        for sample in self.samples:
            idx = sample
            x = [0, self.thetas[idx[0]], 0, self.ang_vels[idx[1]]]
            phis = self.phi_fn(x)
            phi = phis[0, -1]

            C = self.phi_grad(x)
            d = -phi / self.dt if phi < 0 else -phi * self.k # self.coe[3]
            has_valid = self.has_valid_control(C, d, x)
            
            if np.abs(phi) <= 1e-2: # near the boundary
                on_bdry += 1
                valid += has_valid

            if np.max(phis) <= 0 and has_valid:
                in_invariant += 1
        
        print("N samples on boundary: %i" % on_bdry)
        self.valid = valid
        valid_rate = float(valid)/max(on_bdry, 1)
        
        #### Reg term ####
        in_invariant_rate = float(in_invariant) / len(self.samples)
        
        # Log
        self.valid_rate = valid_rate
        self.in_invariant_rate = in_invariant_rate
        
        rv = valid_rate + self.reg_weight * in_invariant_rate

        print("valid_rate")
        print(valid_rate)
        print("in_invariant_rate")
        print(in_invariant_rate)
        return rv

    def visualize(self, params):
        """
        Visualizes where in the state space we have valid safe control...
        :param params:
        :return:
        """
        self.set_params(params)
        valid_cnt = np.zeros((len(self.thetas), len(self.ang_vels)))
        tot_cnt = 0
        for sample in self.samples:
            idx = sample
            x = [0, self.thetas[idx[0]], 0, self.ang_vels[idx[1]]]
            phis = self.phi_fn(x)
            phi = phis[0, -1]
            C = self.phi_grad(x)
            # d = -phi/self.dt if phi < 0 else -phi*self.coe[3]
            d = -phi/self.dt if phi < 0 else -phi*self.k
            if not self.has_valid_control(C, d, x):
                continue
            valid_cnt[idx[0], idx[1]] += 1
            tot_cnt += 1

        print("tot_cnt: ", tot_cnt)
        # plt.figure()
        sns.set_theme()
        ax = sns.heatmap(valid_cnt, cmap="YlGnBu", vmin=0)
        xticks = range(0, len(self.thetas), len(self.thetas)//5)
        yticks = range(0, len(self.ang_vels), len(self.ang_vels)//5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(np.array(self.thetas[xticks]).astype(float))
        ax.set_yticklabels(np.array(self.ang_vels[yticks]).astype(int))
        plt.ylim(0,len(self.ang_vels))
        plt.xlim(0,len(self.thetas))
        # plt.show()
        plt.savefig("./rollout_cbf_classes/cma_es_results/"+str(self.coe)+".png", dpi=300)
        # for key in valid_cnt.keys():
        #     # print(key)
        #     # print(valid_cnt[key])
        #     plt.scatter(key[0], key[1], color="b")
        # plt.show()

        return valid_cnt

    @property
    def log(self):
        # return "{} {}".format(str(self.coe), str(self.valid))
        # return "{} {} {}".format(str(self.coe), str(self.valid))
        s = "Params: %s, valid rate: %f, volume rate: %f" % (str(self.coe), self.valid_rate, self.in_invariant_rate)
        return s