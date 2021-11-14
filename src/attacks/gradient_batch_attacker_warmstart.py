import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *

class GradientBatchWarmstartAttacker():
    """
    Gradient-based attack, but parallelized across many initializations
    """
    # Note: this is not batch compliant.

    def __init__(self, x_lim, device, logger, n_samples=20, \
                 stopping_condition="n_steps", max_n_steps=10, early_stopping_min_delta=1e-3, early_stopping_patience=50,\
                 lr=1e-3, \
                 p_random=0.3,\
                 projection_stop_threshold=1e-3, projection_lr=1e-3, projection_time_limit=3, verbose=False):
        vars = locals()  # dict of local names
        self.__dict__.update(vars)  # __dict__ holds and object's attributes
        del self.__dict__["self"]  # don't need `self`

        assert stopping_condition in ["n_steps", "early_stopping"]

        self.x_dim = self.x_lim.shape[0]

        # Compute 2n facets volume of n-dim hypercube (actually n facets because they come in pairs)
        x_lim_interval_sizes = self.x_lim[:, 1] - self.x_lim[:, 0]
        x_lim_interval_sizes = x_lim_interval_sizes.view(1, -1)
        tiled = x_lim_interval_sizes.repeat(self.x_dim, 1)
        tiled = tiled - torch.eye(self.x_dim).to(self.device)*x_lim_interval_sizes + torch.eye(self.x_dim).to(device)
        vols = torch.prod(tiled, axis=1)
        vols = vols/torch.sum(vols)
        self.vols = vols.detach().cpu().numpy() # numpy
        self.hypercube_vol = torch.prod(x_lim_interval_sizes) # tensor const

        # For warmstart
        self.X_saved = None
        self.obj_vals_saved = None

    def project(self, phi_fn, x):
        # Until convergence
        i = 0
        t1 = time.perf_counter()

        x_list = list(x)
        x_list = [x_mem.view(-1, self.x_dim) for x_mem in x_list] # TODO
        for x_mem in x_list:
            x_mem.requires_grad = True
        proj_opt = optim.Adam(x_list, lr=self.projection_lr)

        while True:
            proj_opt.zero_grad()
            loss = torch.sum(torch.abs(phi_fn(torch.cat(x_list), grad_x=True)[:, -1]))
            loss.backward()
            proj_opt.step()

            i += 1
            t_now = time.perf_counter()
            if torch.max(loss) < self.projection_stop_threshold:
                # print("reprojection exited before timeout in %i steps" % i)
                break
            elif (t_now - t1) > self.projection_time_limit:
                # print("reprojection exited on timeout")
                break

        for x_mem in x_list:
            x_mem.requires_grad = False
        rv_x = torch.cat(x_list)
        return rv_x

    def step(self, objective_fn, phi_fn, x, mode="dG"):
        # It makes less sense to use an adaptive LR method here, if you think about it
        t0 = time.perf_counter()
        x_batch = x.view(-1, self.x_dim)
        x_batch.requires_grad = True

        obj_val = -objective_fn(x_batch)
        obj_grad = grad([torch.sum(obj_val)], x_batch)[0]

        phi_val = phi_fn(x_batch)
        normal_to_manifold = grad([torch.sum(phi_val[:, -1])], x_batch)[0]
        normal_to_manifold = normal_to_manifold/torch.norm(normal_to_manifold, dim=1)[:, None] # normalize

        x_batch.requires_grad = False

        weights = obj_grad.unsqueeze(1).bmm(normal_to_manifold.unsqueeze(2))[:, 0]
        proj_obj_grad = obj_grad - weights*normal_to_manifold

        if self.verbose:
            print("unprojected grad:", obj_grad)
            print("projected grad: ", proj_obj_grad)

        # Take a step
        x_new = x - self.lr*proj_obj_grad

        t1 = time.perf_counter()
        if self.verbose:
            print("After step:", x_new)
        x_new = self.project(phi_fn, x_new)
        if self.verbose:
            print("After reprojection", x_new)

        # TODO: hard-coded for reduced cartpole
        x_new[:, 0] = torch.atan2(torch.sin(x_new[:, 0]), torch.cos(x_new[:, 0]))
        x_new = torch.minimum(torch.maximum(x_new, self.x_lim[:, 0]), self.x_lim[:, 1])

        if mode=="dS":
            phi_x_new = phi_fn(x_new)
            exited_dS_ind = torch.nonzero(torch.max(phi_x_new[:, :-1], axis=1)[0] > 1e-6) # tol
            x_new[exited_dS_ind] = x[exited_dS_ind]

        return x_new

    def sample_in_cube(self):
        """
        Samples uniformly in state space hypercube
        Returns 1 sample
        """
        # samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
        unif = torch.rand(self.x_dim).to(self.device)
        sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
        return sample

    def sample_on_cube(self):
        """
        Samples uniformly on state space hypercube
        Returns 1 sample
        """
        # https://math.stackexchange.com/questions/2687807/uniquely-identify-hypercube-faces
        which_facet_pair = np.random.choice(np.arange(self.x_dim), p=self.vols)
        which_facet = np.random.choice([0, 1])

        # samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
        unif = torch.rand(self.x_dim).to(self.device)
        sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
        sample[which_facet_pair] = self.x_lim[which_facet_pair, which_facet]
        return sample

    def intersect_segment_with_manifold(self, p1, p2, phi_fn, rtol=1e-5, atol=1e-3):
        """
        Atol? Reltol?
        """
        diff = p2-p1

        left_weight = 0.0
        right_weight = 1.0
        left_val = phi_fn(p1.view(1, -1))[0, -1]
        right_val = phi_fn(p2.view(1, -1))[0, -1]
        left_sign = torch.sign(left_val)
        right_sign = torch.sign(right_val)

        if left_sign*right_sign > 0:
            return None

        t0 = time.perf_counter()
        while True:
            mid_weight = (left_weight + right_weight)/2.0
            mid_point = p1 + mid_weight*diff

            mid_val = phi_fn(mid_point.view(1, -1))[0, -1]
            mid_sign = torch.sign(mid_val)
            if mid_sign*left_sign < 0:
                # go to the left side
                right_weight = mid_weight
                right_val = mid_val
            elif mid_sign*right_sign <= 0:
                left_weight = mid_weight
                left_val = mid_val

            # Use this approach or the one below to prevent infinite loops
            # Approach #1
            if np.abs(left_weight - right_weight) < 1e-3:
                intersection_point = p1 + left_weight*diff
                break
            t1 = time.perf_counter()
            if (t1-t0)>7:
                # This clause is necessary for non-differentiable, continuous points (abrupt change)
                print("Something is wrong in projection")
                print(torch.abs(left_val - right_val))
                print(left_weight, right_weight)
                left_point = p1 + left_weight * diff
                right_point = p1 + right_weight * diff
                print(left_point, right_point)
                print(left_val, right_val)
                print(mid_val, mid_point, mid_sign)
                # IPython.embed()
                return None

        return intersection_point

    def sample_points_on_boundary(self, phi_fn, n_samples, mode="dG"):
        """
        Returns torch array of size (self.n_samples, self.x_dim)
        Mode between "dG", "dG+dS", "dG/dS"
        """
        # Everything done in torch
        samples = []
        n_remaining_to_sample = n_samples

        center = self.sample_in_cube()
        n_segments_sampled = 0
        while n_remaining_to_sample > 0:
            # print(n_remaining_to_sample)
            outer = self.sample_on_cube()

            intersection = self.intersect_segment_with_manifold(center, outer, phi_fn)
            valid = False
            if intersection is not None:
                if mode == "dG":
                    samples.append(intersection.view(1, -1))
                    n_remaining_to_sample -= 1
                    valid = True
                elif mode == "dS":
                    phi_val = phi_fn(intersection.view(1, -1))
                    on_dS = torch.all(phi_val[0, :-1] <= 1e-6).item() # tol
                    if on_dS:
                        samples.append(intersection.view(1, -1))
                        n_remaining_to_sample -= 1
                        valid = True

            if not valid:
                center = self.sample_in_cube()
            n_segments_sampled += 1
            # self.logger.info("%i segments" % n_segments_sampled)

        samples = torch.cat(samples, dim=0)
        # self.logger.info("Done with sampling points on the boundary...")
        return samples

    def opt(self, objective_fn, phi_fn, debug=False, mode="dG"):

        # print("Opt mode: ", mode)
        if self.X_saved is None:
            # print("here")
            X_init = self.sample_points_on_boundary(phi_fn, self.n_samples, mode=mode)
        else:
            # IPython.embed()
            n_random_samples = int(self.n_samples*self.p_random)
            n_reuse_samples = self.n_samples - n_random_samples
            X_random_init = self.sample_points_on_boundary(phi_fn, n_random_samples)

            inds = torch.argsort(self.obj_vals_saved, axis=0, descending=True).flatten()
            X_reuse_init = self.X_saved[inds[:n_reuse_samples]]
            X_reuse_init = self.project(phi_fn, X_reuse_init)

            X_init = torch.cat([X_random_init, X_reuse_init], axis=0)

        X = X_init.clone()
        i = 0
        early_stopping = EarlyStoppingBatch(self.n_samples, patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta)

        while True:
            # print(i)
            X = self.step(objective_fn, phi_fn, X, mode=mode)

            if self.stopping_condition == "n_steps":
                if (i > self.max_n_steps):
                    break
            elif self.stopping_condition == "early_stopping":
                obj_vals = objective_fn(X.view(-1, self.x_dim))

                early_stopping(obj_vals)
                if early_stopping.early_stop:
                    break
                elif i > 400: # Hard-coded n_{max iter}
                    break
            i += 1

        # Save for warmstart
        self.X_saved = X
        self.obj_vals_saved = obj_vals

        # Returning a single attack
        obj_vals = objective_fn(X)
        max_ind = torch.argmax(obj_vals)

        if not debug:
            x = X[max_ind]
            return x
        else: # TODO
            x = X[max_ind]
            return X_init, X, x, obj_vals

