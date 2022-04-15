import numpy as np
import matplotlib.pyplot as plt
from rollout_envs.cart_pole_env import CartPoleEnv
from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv
import seaborn as sns
from .normal_ssa_newsi import SSA
from .flying_pend_ssa import FlyingPendSSA
# from .normal_ssa import SSA

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



class FlyingPendEvaluator(object):
    def __init__(self):

        self.env = FlyingInvertedPendulumEnv()

        self.coe = np.zeros(4) # parameters; initialization doesn't matter
        """
        phi_0 = delta - delta_max
        phi = (delta - delta_max)^c0 - c1 dot_delta + c2
        dot_phi < c3 * phi
        
        What's in self.coe?
        0: power
        1: weight on dot_theta term 
        2: additive scalar
        3: weight dotphi < -weight*phi
        """

        # Sample state space
        # n_theta = 30 # TODO
        # n_psi = 30 # TODO
        # n_dot_theta = 30 # TODO
        # n_dot_psi = 30 # TODO

        # delta = 0.01
        # self.thetas = np.arange(-self.env.max_theta, self.env.max_theta, delta)
        # self.psis = np.arange(-self.env.max_angular_velocity, self.env.max_angular_velocity, delta)
        
        # self.thetas = np.linspace(0, np.pi/4, n_theta)
        # self.psis = np.linspace(0, np.pi/4, n_psi)
        # m1, m2 = np.meshgrid(self.thetas, self.psis)
        
        n_samples = 100000
        
        # print("sampling range:", self.env.x_lim[6:10,0])
        # print("sampling range:", self.env.x_lim[6:10,1])
        # x_lim_low = self.env.x_lim[6:10,0]
        # x_lim_high = self.env.x_lim[6:10,1]
        self.samples = []
        
        for i in range(n_samples):
            # x_sample = np.zeros(16)
            # x_sample[6:10] = (x_lim_high-x_lim_low) * np.random.random_sample((4,)) + x_lim_low
            x_sample = (self.env.x_lim[:,1] - self.env.x_lim[:,0]) * np.random.random_sample((len(self.env.x_lim[:,0]),)) + self.env.x_lim[:,0]
            self.samples.append(x_sample)
        
        # Defining some var
        self.min_u = np.vstack(self.env.u_lim[:,0])
        self.max_u = np.vstack(self.env.u_lim[:,1])
        self.dt = self.env.dt

        self.reg_weight = 1 # TODO: need to tune to get possible result

        self.k = 5.0 # TODO: fixed, not learning it. For ease of comparison
        # self.d_min = 1
        
        self.ssa = FlyingPendSSA(FlyingInvertedPendulumEnv())
        # self.simp_ssa = FlyingPendSimpSSA(FlyingInvertedPendulumEnv())
        
    def phi_fn(self, x):
        return self.ssa.phi_fn(np.hstack(x))

    def phi_grad(self, x):
        return self.ssa.phi_grad(np.hstack(x))

    def set_params(self, params):
        """
        What's in self.coe?
        0: -dot phi < - c0
        1: power of delta term
        2: weight on dot_delta term 
        3: additive scalar
        """
        self.coe = params
        self.ssa.set_params(params)
        
    def most_valid_control(self, grad_phi, x):
        # dot_x = f + g * u 
        f = np.vstack(self.env._f(x))
        g = np.vstack(self.env._g(x))
        # return min(grad_phi @ f + grad_phi @ g @ self.min_u, grad_phi @ f + grad_phi @ g @ self.max_u)

        min_dot_phi = 1e9
        # print("verts:")
        for u in self.env.control_lim_verts:
            # print("u_vert: ", u, "dot_phi", grad_phi @ f + grad_phi @ g @ u)
            min_dot_phi = min(min_dot_phi, grad_phi @ f + grad_phi @ g @ u)
        
        # u_safe, debug_dict = self.ssa.compute_control(1, x)
        # print("u_qp: ", u_safe, "dot_phi: ", grad_phi @ f + grad_phi @ g @ u_safe)
        # exit()
        return min_dot_phi
    
    def near_boundary(self, phis):
        # print("phis")
        # print(phis)
        phi0 = phis[0,0]
        phi = phis[0,-1]
        eps = 1e-2
        # return abs(phi0) < eps or abs(phi) < eps
        return abs(phi) < eps

    # TODO: deprecated
    """
    def find_max_dot_phi(self):
        in_invariant = 0
        valid = 0
        tot_cnt = 0
        max_dot_phi = -1e9
        reg = 0
        
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        
        for sample in self.samples:
            idx = sample
            x = np.hstack([0, self.thetas[idx[0]], 0, self.psis[idx[1]]])
            
            phis = self.ssa.phi_fn(x)
            
            phi = phis[0, -1]
            # phi = np.max(phis)
            
            if self.near_boundary(phis):
                tot_cnt += 1
                
                C = self.ssa.phi_grad(x)
                # d = -phi/self.dt if phi < 0 else -phi*self.k
                d = -phi/self.dt if phi < 0 else -phi*self.k
                most_valid = self.most_valid_control(C, x)
                max_dot_phi = max(max_dot_phi, most_valid)
                
            if phi <= 0:
                # reg += sigmoid(phi)
                in_invariant += 1
        
        in_invariant_rate = float(in_invariant)/len(self.samples)
        return max_dot_phi, in_invariant_rate
        # """
    
    def compute_valid_invariant(self):
        in_invariant = 0
        in_invariant_valid = 0
        valid = 0
        tot_cnt = 0
        for sample in self.samples:
            # idx = sample
            # x = np.hstack([0, self.thetas[idx[0]], 0, self.psis[idx[1]]])
            x = sample
            # print("x")
            # print(x)
            phis = self.ssa.phi_fn(x)
            
            phi = phis[0, -1]
            # phi = np.max(phis)
            
            if self.near_boundary(phis):
                tot_cnt += 1
                
                C = self.ssa.phi_grad(x)
                # d = -phi/self.dt if phi < 0 else -phi*self.k
                d = -phi/self.dt if phi < 0 else -1
                most_valid = self.most_valid_control(C, x) # TODO: Simin, check this function. You made an MG update....
                weighted_valid = np.exp(0.1 * min(d - most_valid, 0))
                # most_valid < d => 1,   most_valid > d => 1 - exp(d-valid)
                # valid += weighted_valid
                # print(most_valid - d)
                valid += (most_valid < d)

            if np.max(phis) <= 0:
                in_invariant += 1

        self.valid = valid
        valid_rate = valid * 1.0 / max(1,tot_cnt)
        print("valid / tot_cnt: ", valid, "/", tot_cnt)
        # valid_rate = valid * 1.0 / len(self.samples)
        # self.valid = in_invariant_valid
        # valid_rate = in_invariant_valid * 1.0 / max(1, in_invariant)

        #### Reg term ####
        in_invariant_rate = float(in_invariant)/len(self.samples)
        return valid_rate, in_invariant_rate
        
    def evaluate(self, params):
        self.set_params(params)
        
        valid_rate, in_invariant_rate = self.compute_valid_invariant()
        self.valid_rate = valid_rate
        self.in_invariant_rate = in_invariant_rate
        rv = valid_rate + self.reg_weight*in_invariant_rate # TODO: Simin, set this weight
        print("valid rate: ", valid_rate,"in_invariant_rate: ", in_invariant_rate, "params: ", params)
        return rv
        
        # max_dot_phi, in_invariant_rate = self.find_max_dot_phi()
        # print("=======")
        # print(params)
        # print(max_dot_phi, " ", in_invariant_rate)
        # rv = -max_dot_phi + self.reg_weight*in_invariant_rate
        # return rv
    
    @property
    def log(self):
        # return "{} {}".format(str(self.coe), str(self.valid))
        # return "{} {} {}".format(str(self.coe), str(self.valid))
        # s = "Params: %s, valid rate: %f, volume rate: %f" % (str(self.coe), self.valid_rate, self.in_invariant_rate)
        # return s
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
        n_theta = 400 # TODO
        n_psi = 800 # TODO


        self.thetas = np.linspace(0, self.env.max_theta, n_theta)
        self.ang_vels = np.linspace(-self.env.max_angular_velocity, self.env.max_angular_velocity, n_psi)
        m1, m2 = np.meshgrid(self.thetas, self.ang_vels)

        self.samples = []
        for i,j in np.ndindex(m1.shape):
            self.samples.append([j,i])

        # Defining some var
        self.max_u = np.vstack([self.env.max_force])
        self.dt = self.env.dt

        # self.reg_weight = 0.1 # TODO: need to tune to get possible result
        
        self.reg_weight = 1 # TODO: need to tune to get possible result

        self.k = 5.0 # TODO: fixed, not learning it. For ease of comparison
        # self.d_min = 1
        
        self.ssa = SSA(CartPoleEnv())
        
    def phi_fn(self, x):
        return self.ssa.phi_fn(np.hstack(x))

    def phi_grad(self, x):
        return self.ssa.phi_grad(np.hstack(x))

    def set_params(self, params):
        self.coe = params
        self.ssa.c1 = params[0]
        self.ssa.c2 = params[1]
        self.ssa.c3 = params[2]

    def most_valid_control(self, C, x):
        # dot_x = f + g * u 
        f = np.vstack(self.env.x_dot_open_loop(x, 0))
        g = np.vstack(self.env.x_dot_open_loop(x, 1)) - f
        return min(C @ f + C @ g @ self.max_u, C @ f - C @ g @ self.max_u)
    
    def near_boundary(self, phis):
        phi0 = phis[0,0]
        phi = phis[0,-1]
        eps = 1e-2
        return abs(phi0) < eps or abs(phi) < eps

    def find_max_dot_phi(self):
        in_invariant = 0
        valid = 0
        tot_cnt = 0
        max_dot_phi = -1e9
        reg = 0
        
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        
        for sample in self.samples:
            idx = sample
            x = np.hstack([0, self.thetas[idx[0]], 0, self.ang_vels[idx[1]]])
            
            phis = self.ssa.phi_fn(x)
            
            # phi = phis[0, -1]
            phi = np.max(phis)
            
            if self.near_boundary(phis):
                tot_cnt += 1
                
                C = self.ssa.phi_grad(x)
                d = -phi/self.dt if phi < 0 else -phi*self.k
                most_valid = self.most_valid_control(C, x)
                max_dot_phi = max(max_dot_phi, most_valid)
                
            if phi <= 0:
                # reg += sigmoid(phi)
                in_invariant += 1
        
        in_invariant_rate = float(in_invariant)/len(self.samples)
        return max_dot_phi, in_invariant_rate
        
    
    def compute_valid_invariant(self):
        in_invariant = 0
        in_invariant_valid = 0
        valid = 0
        tot_cnt = 0
        for sample in self.samples:
            idx = sample
            x = np.hstack([0, self.thetas[idx[0]], 0, self.ang_vels[idx[1]]])
            
            phis = self.ssa.phi_fn(x)
            
            # phi = phis[0, -1]
            phi = np.max(phis)
            
            if self.near_boundary(phis):
                tot_cnt += 1
                
                C = self.ssa.phi_grad(x)
                d = -phi/self.dt if phi < 0 else -phi*self.k
                most_valid = self.most_valid_control(C, x)
                weighted_valid = np.exp(0.1 * min(d - most_valid, 0))
                # most_valid < d => 1,   most_valid > d => 1 - exp(d-valid)
                valid += weighted_valid

            if np.max(phis) <= 0:
                in_invariant += 1

        self.valid = valid
        valid_rate = valid * 1.0 / max(1,tot_cnt)
        print("valid / tot_cnt: ", valid, "/", tot_cnt)
        # valid_rate = valid * 1.0 / len(self.samples)
        # self.valid = in_invariant_valid
        # valid_rate = in_invariant_valid * 1.0 / max(1, in_invariant)

        #### Reg term ####
        in_invariant_rate = float(in_invariant)/len(self.samples)
        return valid_rate, in_invariant_rate
        
    def evaluate(self, params):
        self.set_params(params)
        
        # valid_rate, in_invariant_rate = self.compute_valid_invariant()
        # Log
        # self.valid_rate = valid_rate
        # self.in_invariant_rate = in_invariant_rate
        # rv = valid_rate + self.reg_weight*in_invariant_rate
        # print("valid rate: ", valid_rate,"in_invariant_rate: ", in_invariant_rate, "params: ", params)
        # return rv
        
        max_dot_phi, in_invariant_rate = self.find_max_dot_phi()
        print("=======")
        print(params)
        print(max_dot_phi, " ", in_invariant_rate)
        rv = -max_dot_phi + self.reg_weight*in_invariant_rate
        return rv

        # # TODO: special objective on the boundary

        # self.set_params(params)
        # valid = 0
        # on_bdry = 0
        
        # in_invariant = 0
        # for sample in self.samples:
        #     idx = sample
        #     x = [0, self.thetas[idx[0]], 0, self.ang_vels[idx[1]]]
        #     phis = self.phi_fn(x)
        #     phi = phis[0, -1]

        #     C = self.phi_grad(x)
        #     d = -phi / self.dt if phi < 0 else -phi * self.k # self.coe[3]
        #     has_valid = self.has_valid_control(C, d, x)
            
        #     if np.abs(phi) <= 1e-2: # near the boundary
        #         on_bdry += 1
        #         valid += has_valid

        #     if np.max(phis) <= 0 and has_valid:
        #         in_invariant += 1
        
        # print("N samples on boundary: %i" % on_bdry)
        # self.valid = valid
        # valid_rate = float(valid)/max(on_bdry, 1)
        
        # #### Reg term ####
        # in_invariant_rate = float(in_invariant) / len(self.samples)
        
        # # Log
        # self.valid_rate = valid_rate
        # self.in_invariant_rate = in_invariant_rate
        
        # rv = valid_rate + self.reg_weight * in_invariant_rate

        # print("valid_rate")
        # print(valid_rate)
        # print("in_invariant_rate")
        # print(in_invariant_rate)
        # return rv

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
            # C = self.phi_grad(x)
            # # d = -phi/self.dt if phi < 0 else -phi*self.coe[3]
            # d = -phi/self.dt if phi < 0 else -phi*self.k
            # if not self.has_valid_control(C, d, x):
            #     continue
            # valid_cnt[idx[0], idx[1]] += 1
            
            C = self.ssa.phi_grad(x)
            d = -phi/self.dt if phi < 0 else -phi*self.k
            most_valid = self.most_valid_control(C, x)
            weighted_valid = np.exp(0.1 * min(d - most_valid, 0))
            # most_valid < d => 1,   most_valid > d => 1 - exp(d-valid)
            valid_cnt[idx[0], idx[1]] += weighted_valid
            
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
        # s = "Params: %s, valid rate: %f, volume rate: %f" % (str(self.coe), self.valid_rate, self.in_invariant_rate)
        # return s
        return ""