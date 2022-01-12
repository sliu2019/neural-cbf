import numpy as np
import time
import subprocess
import sys, os
import copy
import matplotlib.pyplot as plt
import math
from rollout_cbf_classes.cart_pole_env import CartPoleEnv
import seaborn as sns

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

        self.coe = np.zeros(4)
        n_theta = 100
        n_ang = 100
        
        self.thetas = np.linspace(0, self.env.max_theta, n_theta)
        self.ang_vels = np.linspace(-self.env.max_angular_velocity, self.env.max_angular_velocity, n_ang)
        m1, m2 = np.meshgrid(self.thetas, self.ang_vels)

        self.samples = []
        for i,j in np.ndindex(m1.shape):
            self.samples.append([i,j])

        self.max_u = np.vstack([self.env.max_force])
        self.dt = self.env.dt

        self.d_min = 1
        
    def phi(self, x):
        # x =  [x theta, dot_x, dot_theta]
        # phi = theta ** a1 - theta_max ** a1 + a2 * dot_theta + a3
        # phi_0 = theta ** 2 - theta_max ** 2

        # invariant set: (phi < 0 or dot_phi < 0 (with a valid control)) and phi_0 < 0
        theta = x[1]
        dot_theta = x[3]
        return theta ** self.coe[0] - self.env.theta_safe_lim ** self.coe[0] + self.coe[1] * dot_theta + self.coe[2]

    def grad_phi(self, x):
        theta = x[1]
        dot_theta = x[3]
        return np.hstack([0, self.coe[0] * theta ** (self.coe[0]-1), 0, self.coe[1]])

    def set_params(self, params):
        self.coe = params

    def has_valid_control(self, C, d, x):
        # dot_x = f + g * u 
        f = np.vstack(self.env.x_dot_open_loop(x, 0))
        g = np.vstack(self.env.x_dot_open_loop(x, 1)) - f
        return (C @ f + C @ g @ self.max_u < d) or (C @ f - C @ g @ self.max_u < d)

    def evaluate(self, params):
        self.set_params(params)
        valid = 0
        for sample in self.samples:
            idx = sample
            x = [0, self.thetas[idx[0]], 0, self.ang_vels[idx[1]]]
            phi = self.phi(x)
            # C dot_x < d
            # phi(x_k) > 0 -> con: dot_phi(x_k) < -k*phi(x): C = self.grad_phi(x, o)  d = -phi/dt*self.coe[0]
            # phi(x_k) < 0 -> con: phi(x_k+1) < 0:           C = self.grad_phi(x, o)  d = -phi/dt
            C = self.grad_phi(x)
            d = -phi/self.dt if phi < 0 else -phi/self.dt*self.coe[2]
            valid += self.has_valid_control(C, d, x)

        self.valid = valid
        valid_rate = valid * 1.0 / len(self.samples)

        return valid_rate

    def visualize(self, params):
        self.set_params(params)
        valid_cnt = np.zeros((len(self.thetas), len(self.ang_vels)))
        tot_cnt = 0
        for sample in self.samples:
            idx = sample
            x = [0, self.thetas[idx[0]], 0, self.ang_vels[idx[1]]]
            phi = self.phi(x)
            C = self.grad_phi(x)
            d = -phi/self.dt if phi < 0 else -phi/self.dt*self.coe[2]
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
        plt.savefig("./cma_es_results/"+str(self.coe)+".png", dpi=300)
        # for key in valid_cnt.keys():
        #     # print(key)
        #     # print(valid_cnt[key])
        #     plt.scatter(key[0], key[1], color="b")
        # plt.show()

        return valid_cnt

    @property
    def log(self):
        return "{} {}".format(str(self.coe), str(self.valid))
