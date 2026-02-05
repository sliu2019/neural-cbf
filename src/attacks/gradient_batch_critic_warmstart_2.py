import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *
import IPython
# import multiprocessing as mp

import torch.multiprocessing as mp
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

class GradientBatchWarmstartCritic2():
    """
    Faster and better than V1?

    Upgrades:
	1. Multi-processing for speed-up on boundary sampling
	2. Different projection routines
    """
    def __init__(self, x_lim, device, logger, n_samples=20, \
                 stopping_condition="n_steps", max_n_steps=10, early_stopping_min_delta=1e-3, early_stopping_patience=50,\
                 lr=1e-3, \
                 p_reuse=0.7,\
                 projection_tolerance=1e-1, projection_lr=1e-4, projection_time_limit=3.0, verbose=False, critic_use_n_step_schedule=False, proj_tactic="gd_step_timeout"): # TODO: verbose
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

        # For projection, adamBA
        self.n_vector_samples = 10
        self.torch_cpu = torch.device("cpu")

        # For multiproc
        # Otherwise, do not set this variable
        # self._surface_fn = None

    def _adamBA_subroutine(self, surface_fn, x, special_vector=None):
        """
        Everything done in torch, on the CPU
        """
        print("Inside _adamBA_subroutine")
        print("Check by seeing if dot product between special_vector and sampled_vectors is positive, near 1")
        IPython.embed()

        ###### Sample vectors
        special_vector_orth1 = torch.torch([-special_vector[1], special_vector[0], 0])
        special_vector_orth2 = torch.cross(special_vector, special_vector_orth1)

        sampled_eps_vectors = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=(self.n_vector_samples)) # n x 2
        sampled_eps_vectors = torch.from_numpy(sampled_eps_vectors)
        # sampled_eps_vectors = np.concatenate((sampled_eps_vectors, np.zeros(self.n_vector_samples, 1)), axis=1)

        proj_matrix = torch.cat((special_vector_orth1[:, None], special_vector_orth2[:, None]), axis=1) # 3 x 2
        # project to zero-centered plane orthogonal to special_vector
        sampled_eps_vectors = (proj_matrix@sampled_eps_vectors.T).T # n x 3

        sampled_vectors = special_vector[None, :] + sampled_eps_vectors
        sampled_vectors = torch.cat((sampled_vectors, special_vector))  # include special vector
        sampled_vectors = sampled_vectors/torch.norm(sampled_vectors, axis=1) # all length 1

        vectors = x + sampled_vectors
        surface_other_end = surface_fn(x)
        sign_other_end = torch.sign(surface_other_end)
        while vectors.size >= 0:
            # Check if intersecting with boundary
            surface_one_end = surface_fn(vectors)

            signs = torch.sign(surface_one_end)*sign_other_end
            inds = np.argwhere(signs <= 0)

            if len(inds) > 0:
                closest_proj = None
                closest_proj_dist = float("inf")
                for ind in inds:
                    rv = self._intersect_segment_with_manifold(x, vectors[ind], surface_fn)
                    proj_dist = torch.norm(rv-x)
                    if proj_dist < closest_proj_dist:
                        closest_proj = rv
                        closest_proj_dist = proj_dist
                return closest_proj

            # Are any exiting box? Remove now
            exit_inds = np.argwhere(np.logical_or(vectors <= self.x_lim[:, 0], vectors >= self.x_lim[:, 1], axis=1))
            vectors = np.delete(vectors, exit_inds, axis=0) # TODO: check axis

            # For the remaining, double in length
            vectors = vectors*2.0

        # Failed
        # Later, can do something more sophisticated in this case
        return None

    def _project_adamBA(self, surface_fn, X):
        """
        CPU multiprocessing
        Sampled line search projection
        Naturally times out: with success or no success (none of lines intersected with surface)

        Have projections return a debug dictionary that we accumulate, instead of print outs
        """
        print("inside project_adamBA")
        print("check which device everything is on")
        print("also check whether the devices are consistent")
        IPython.embed()

        # compute gradients here, using GPU (prolly faster)
        X.requires_grad = True
        gradient = grad([torch.sum(torch.abs(surface_fn(X)))], X)[0]
        X.requires_grad = False

        # use mp to call adamBA
        gradient_cpu = gradient.to(self.torch_cpu)
        X_cpu = X.to(self.torch_cpu)

        # n_cpu = mp.cpu_count()
        # pool = mp.Pool(n_cpu)
        n_points = X.shape[0]
        pool = mp.Pool(n_points) # TODO: mp

        final_arg = [[surface_fn, X_cpu[i], gradient_cpu[i]] for i in range(n_points)]

        # TODO: remove, merely for debug
        self._adamBA_subroutine(surface_fn, X_cpu[0], special_vector=gradient_cpu[0])
        print("ln 141")
        IPython.embed()
        result_cpu = pool.starmap(self._adamBA_subroutine, final_arg) # projected x's or None
        result = result_cpu.to(self.device)

        # For now, if projection unsuccessful, do nothing
        # Perhaps later, if None then call subroutine until RV is not None
        for i in range(n_points):
            if result[i] is None:
                result[i] = X[i]

        X_projected = torch.cat(result, dim=0)
        return X_projected

    def _project(self, surface_fn, X, projection_n_grad_steps=None):
        """
        GPU batched
        GD-based projection
        With timeout
        """
        # Until convergence
        i = 0
        t1 = time.perf_counter()

        X_list = list(X)
        X_list = [X_mem.view(-1, self.x_dim) for X_mem in X_list] # TODO
        for X_mem in X_list:
            X_mem.requires_grad = True
        proj_opt = optim.Adam(X_list, lr=self.projection_lr)

        while True:
            proj_opt.zero_grad()
            # print("Inside _project, check out surface_fn")
            # IPython.embed()
            loss = torch.sum(torch.abs(surface_fn(torch.cat(X_list), grad_x=True)))
            # loss = torch.sum(torch.abs(surface_fn(torch.cat(x_list))))
            loss.backward()
            proj_opt.step()

            i += 1
            # print(i)
            t_now = time.perf_counter()
            if torch.max(loss) < self.projection_tolerance:
                # if self.verbose:
                #     print("reprojection exited before timeout in %i steps" % i)
                break

            if projection_n_grad_steps is not None: # use step number limit
                if i == projection_n_grad_steps:
                    break
            else: # else, use time limit
                if (t_now - t1) > self.projection_time_limit:
                    # print("reprojection exited on timeout")
                    print("Attack: reprojection exited on timeout, max dist from =0 boundary: ", torch.max(loss).item())
                    break

            # print((t_now - t1), torch.max(loss))

        for X_mem in X_list:
            X_mem.requires_grad = False
        rv_X = torch.cat(X_list)

        if self.verbose:
            # if torch.max(loss) < self.projection_tolerance:
            #     print("Yes, on manifold")
            # else:
            if torch.max(loss) > self.projection_tolerance:
                print("Not on manifold, %f" % (torch.max(loss).item()))
        # IPython.embed()
        return rv_X

    def _step(self, saturation_risk, surface_fn, X):
        # It makes less sense to use an adaptive LR method here, if you think about it
        t0_step = time.perf_counter()

        X_batch = X.view(-1, self.x_dim)
        X_batch.requires_grad = True

        obj_val = -saturation_risk(X_batch) # maximizing
        obj_grad = grad([torch.sum(obj_val)], X_batch)[0]

        normal_to_manifold = grad([torch.sum(surface_fn(X_batch))], X_batch)[0]
        normal_to_manifold = normal_to_manifold/torch.norm(normal_to_manifold, dim=1)[:, None] # normalize
        X_batch.requires_grad = False
        weights = obj_grad.unsqueeze(1).bmm(normal_to_manifold.unsqueeze(2))[:, 0]
        proj_obj_grad = obj_grad - weights*normal_to_manifold

        # Take a step
        X_new = X - self.lr*proj_obj_grad
        tf_grad_step = time.perf_counter()

        dist_before_proj = torch.mean(torch.abs(surface_fn(X_new)))
        if self.proj_tactic == 'gd_step_timeout':
            X_new = self._project(surface_fn, X_new, projection_n_grad_steps=5) # TODO: set projection_n_grad_steps
        elif self.proj_tactic == 'adam_ba':
            X_new = self._project_adamBA(surface_fn, X)
        dist_after_proj = torch.mean(torch.abs(surface_fn(X_new)))

        tf_reproject = time.perf_counter()

        # Wrap-around in state domain
        X_new = torch.minimum(torch.maximum(X_new, self.x_lim[:, 0]), self.x_lim[:, 1])
        debug_dict = {"t_grad_step": (tf_grad_step-t0_step), "t_reproject": (tf_reproject-tf_grad_step), "diff_after_proj": (dist_after_proj-dist_before_proj)}

        print("Inside _step, line 242")
        IPython.embed()
        return X_new, debug_dict

    def _sample_in_cube(self, random_seed=None):
        """
        Samples uniformly in state space hypercube
        Returns 1 sample
        """
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        # samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
        unif = torch.rand(self.x_dim).to(self.device)
        sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
        return sample

    def _sample_on_cube(self, random_seed=None):
        """
        Samples uniformly on state space hypercube
        Returns 1 sample
        """
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        # https://math.stackexchange.com/questions/2687807/uniquely-identify-hypercube-faces
        which_facet_pair = np.random.choice(np.arange(self.x_dim), p=self.vols)
        which_facet = np.random.choice([0, 1])

        # samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
        unif = torch.rand(self.x_dim).to(self.device)
        sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
        sample[which_facet_pair] = self.x_lim[which_facet_pair, which_facet]
        return sample

    def _intersect_segment_with_manifold(self, p1, p2, surface_fn):
        # if surface_fn is None:
        #     surface_fn = self._surface_fn

        diff = p2-p1

        left_weight = 0.0
        right_weight = 1.0
        # left_val = surface_fn(p1.view(1, -1))[0, -1]
        # right_val = surface_fn(p2.view(1, -1))[0, -1]
        # print("inside _intersect_segment_with_manifold")
        # IPython.embed()
        left_val = surface_fn(p1.view(1, -1)).item()
        right_val = surface_fn(p2.view(1, -1)).item()
        left_sign = np.sign(left_val)
        right_sign = np.sign(right_val)

        if left_sign*right_sign > 0:
            # print("does not intersect")
            return None

        t0 = time.perf_counter()
        while True:
            # print(left_weight, right_weight)
            # print(left_val, right_val)
            mid_weight = (left_weight + right_weight)/2.0
            mid_point = p1 + mid_weight*diff

            # mid_val = surface_fn(mid_point.view(1, -1))[0, -1]
            mid_val = surface_fn(mid_point.view(1, -1)).item()
            mid_sign = np.sign(mid_val)
            if mid_sign*left_sign < 0:
                # go to the left side
                right_weight = mid_weight
                right_val = mid_val
            elif mid_sign*right_sign <= 0:
                left_weight = mid_weight
                left_val = mid_val

            # Use this approach or the one below to prevent infinite loops
            # Approach #1: previously used for discontinuous phi, but we shouldn't have discont. phi
            # if np.abs(left_weight - right_weight) < 1e-3:
            #     intersection_point = p1 + left_weight*diff
            #     break
            if max(abs(left_val), abs(right_val)) < self.projection_tolerance:
                intersection_point = p1 + left_weight*diff
                break
            t1 = time.perf_counter()
            if (t1-t0)>7: # an arbitrary time limit
                # This clause is necessary for non-differentiable, continuous points (abrupt change)
                print("Something is wrong in projection")
                print(left_val, right_val)
                # print(torch.abs(left_val - right_val))
                print(left_weight, right_weight)
                print("p1:", p1)
                print("p2:", p2)
                # left_point = p1 + left_weight * diff
                # right_point = p1 + right_weight * diff
                # print(left_point, right_point)
                # print(mid_val, mid_point, mid_sign)
                # IPython.embed()
                # print("out of time")
                return None
        # print("success")
        return intersection_point

    def _sample_segment_intersect_boundary(self, surface_fn, random_seed=None):
        outer = self._sample_on_cube(random_seed=random_seed)
        center = self._sample_in_cube(random_seed=random_seed)

        intersection = self._intersect_segment_with_manifold(center, outer, surface_fn)

        return intersection

    def _sample_points_on_boundary(self, surface_fn, n_samples):
        """
        Returns torch array of size (self.n_samples, self.x_dim)
        """
        # self._surface_fn = surface_fn # cheat way to pass to child processes
        # Everything done in torch
        samples = []
        n_remaining_to_sample = n_samples
        # n_segments_sampled = 0

        n_cpu = mp.cpu_count()
        pool = mp.Pool(n_cpu) # torch pool

        random_seeds = np.arange(100 * n_samples) # This should be enough. If it isn't, code will error out
        np.random.shuffle(random_seeds)  # in place

        # arg_tup = [surface_fn]
        # duplicated_arg = list([arg_tup] * n_cpu)

        # IPython.embed()
        it = 0
        while n_remaining_to_sample > 0:
            batch_random_seeds = random_seeds[it * n_cpu:(it + 1) * n_cpu]
            # final_arg = [duplicated_arg[i] + [batch_random_seeds[i]] for i in range(n_cpu)]
            final_arg = [[surface_fn, batch_random_seeds[i]] for i in range(n_cpu)]
            result = pool.starmap(self._sample_segment_intersect_boundary, final_arg)

            for intersection in result:
                if intersection is not None:
                    samples.append(intersection.view(1, -1))
                    n_remaining_to_sample -= 1 # could go negative!

            # n_segments_sampled += n_cpu
            it += 1
            self.logger.info("%i segments sampled" % (it*n_cpu))

        print("sample points on boundary")
        IPython.embed()
        samples = torch.cat(samples[:n_samples], dim=0)
        # self.logger.info("Done with sampling points on the boundary...")
        return samples

    def opt(self, saturation_risk, surface_fn, iteration, debug=False):
        t0_opt = time.perf_counter()

        if self.X_saved is None:
            X_init = self._sample_points_on_boundary(surface_fn, self.n_samples)

            X_reuse_init = torch.zeros((0, self.x_dim))
            X_random_init = X_init
        else:
            n_target_reuse_samples = int(self.n_samples*self.p_reuse)

            inds = torch.argsort(self.obj_vals_saved, axis=0, descending=True).flatten()

            # Some attacks will be very near each other. This helps to only select distinct attacks
            inds_distinct = [inds[0]]
            for ind in inds[1:]:
                diff = self.X_saved[torch.tensor(inds_distinct)] - self.X_saved[ind]
                distances = torch.norm(diff.view(-1, self.x_dim), dim=1)
                if torch.any(distances <= 1e-1).item(): # TODO: set this (distance which determines an "identical" point)
                    print("passed")
                    continue
                inds_distinct.append(ind)
                if len(inds_distinct) >= n_target_reuse_samples:
                    break

            n_reuse_samples = len(inds_distinct)
            n_random_samples= self.n_samples - n_reuse_samples
            # print("Actual percentage reuse: %f" % ((n_reuse_samples/self.n_samples)*100))
            X_reuse_init = self.X_saved[torch.tensor(inds_distinct)]
            # print("Reprojecting")
            X_reuse_init = self._project(surface_fn, X_reuse_init) # reproject, since phi changed
            # print("Sampling points on boundary")
            X_random_init = self._sample_points_on_boundary(surface_fn, n_random_samples)
            # print("Done")
            X_init = torch.cat([X_random_init, X_reuse_init], axis=0)

        tf_init = time.perf_counter()

        X = X_init.clone()
        i = 0
        early_stopping = EarlyStoppingBatch(self.n_samples, patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta)
        # logging
        t_grad_step = []
        t_reproject = []
        diff_after_proj = []
        obj_vals = saturation_risk(X.view(-1, self.x_dim))
        init_best_attack_value = torch.max(obj_vals).item()

        # critic_use_n_step_schedule
        max_n_steps = self.max_n_steps
        if self.critic_use_n_step_schedule:
            max_n_steps = (2*self.max_n_steps)*np.exp(-iteration/75) + self.max_n_steps
            print("Max_n_steps: %i" % max_n_steps)
        while True:
            # print("Inner max step #%i" % i)
            X, step_debug_dict = self._step(saturation_risk, surface_fn, X) # Take gradient steps on all candidate attacks
            # obj_vals = saturation_risk(X.view(-1, self.x_dim))

            # Logging
            t_grad_step.append(step_debug_dict["t_grad_step"])
            t_reproject.append(step_debug_dict["t_reproject"])
            diff_after_proj.append(step_debug_dict["diff_after_proj"])

            # Loop break condition
            if self.stopping_condition == "n_steps":
                if (i > max_n_steps):
                    break
            elif self.stopping_condition == "early_stopping":
                print("Not recommended to use this option; it will run for hundreds of steps before stopping")
                raise NotImplementedError
            # elif self.stopping_condition == "early_stopping": # Note: the stopping criteria is so strict that this is effectively the same as using n_steps = 400
            #     early_stopping(obj_vals)
            #     if early_stopping.early_stop:
            #         break
            #     elif i > 400: # Hard-coded n_{max iter}
            #         print("Critic exceeded time limit")
            #         break
            i += 1

        tf_opt = time.perf_counter()

        # Save for warmstart
        self.X_saved = X
        obj_vals = saturation_risk(X.view(-1, self.x_dim))
        self.obj_vals_saved = obj_vals

        # Returning a single attack
        max_ind = torch.argmax(obj_vals)

        if not debug:
            x = X[max_ind]
            return x, {}
        else:
            x = X[max_ind]
            final_best_attack_value = torch.max(obj_vals).item()

            t_init = tf_init - t0_opt
            t_total_opt = tf_opt - t0_opt

            # TODO: do not change the names in the dict here! Names are matched to learner.py
            debug_dict = {"X_init": X_init, "X_init_reuse": X_reuse_init, "X_init_random": X_random_init, "X_final": X, "X_obj_vals": obj_vals, "init_best_attack_value": init_best_attack_value, "final_best_attack_value": final_best_attack_value, "t_init": t_init, "t_grad_steps": t_grad_step, "t_reproject": t_reproject, "t_total_opt": t_total_opt}

            # debug_dict = {"X_init": X_init, "X_reuse_init": X_reuse_init, "X_random_init": X_random_init, "X": X, "obj_vals": obj_vals, "init_best_attack_value": init_best_attack_value, "final_best_attack_value": final_best_attack_value, "t_init": t_init, "t_grad_step": t_grad_step, "t_reproject": t_reproject, "t_total_opt": t_total_opt}

            print("check before returning from opt")
            IPython.embed()
            return x, debug_dict

