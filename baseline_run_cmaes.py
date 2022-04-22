import time

import numpy as np
import os
# import yaml
import math
import pickle, IPython
from cmaes_objective_flying_pend import FlyingPendEvaluator
import multiprocessing as mp

# evaluators_dict = {"CartPoleEvaluator": CartPoleEvaluator, "FlyingPendEvaluator": FlyingPendEvaluator}
evaluators_dict = {"FlyingPendEvaluator": FlyingPendEvaluator}


class CMAESLearning(object):
	def __init__(self, CMAES_args):
		"""
		================================================================================
		Initialize the learning module. Create learning log.
		"""
		self.cmaes_args = CMAES_args

		# Logging
		self.log_fldrpth = os.path.join("cmaes", "flying_pend_" + CMAES_args["exp_name"])  # Hard-coded the prefix
		if not os.path.exists(self.log_fldrpth):
			os.makedirs(self.log_fldrpth)
		print("Saving data at: %s" % self.log_fldrpth)

		with open(os.path.join(self.log_fldrpth, "args.pkl"), 'wb') as handle:
			pickle.dump(CMAES_args, handle, protocol=pickle.HIGHEST_PROTOCOL)

		self.data = {"pop": [], "rewards": [], "mu": [], "sigma": []}
		self.evaluator = self.cmaes_args["evaluator"]

		# For multiprocessing
		# Deprecated, not using
		# self.n_proc = mp.cpu_count()
		# self.n_proc = 150
		# self.pool = mp.Pool(self.n_proc)

	def regulate_params(self, params):
		"""
		================================================================================
		Regulate params by upper bound and lower bound. And convert params to integer if required by the user.
		"""
		params = np.maximum(params, self.cmaes_args["lower_bound"])  # lower bound
		params = np.minimum(params, self.cmaes_args["upper_bound"])  # upper bound
		if "param_is_int" in self.cmaes_args:
			for i in range(params.shape[1]):
				if self.cmaes_args["param_is_int"][i]:
					params[:, [i]] = np.vstack([int(round(x)) for x in params[:, i]])
		# params = [ int(params[i]) if self.cmaes_args["param_is_int"][i] else params[i] ]
		return params

	def populate(self, mu, sigma):
		"""
		================================================================================
		Populate n members using the current estimates of mu and S
		"""
		self.population = np.random.multivariate_normal(mu, sigma, self.cmaes_args["populate_num"])
		self.population = self.regulate_params(self.population)

	def evaluate(self, mu):
		"""
		===============================================================================
		Evaluate a set of weights (a mu) by interacting with the environment and
		return the average total reward over multiple repeats.
		"""
		reward, debug_dict = self.evaluator.evaluate(mu)
		return reward, debug_dict

	def step(self, mu, sigma):
		"""
		===============================================================================
		Perform an iteration of CMA-ES by evaluating all members of the current
		population and then updating mu and S with the top self.cmaes_args["elite_ratio"] proportion of members
		and updateing the weights of the policy networks.
		"""
		self.populate(mu, sigma)

		# pool = mp.Pool(self.n_proc)

		# Refactored, since self.evaluate returns a tuple now
		rewards = []
		all_debug_dicts = None

		"""for i in range(math.ceil(self.cmaes_args["populate_num"]/float(self.n_proc))):
			print(i)
			arguments = self.population[i*self.n_proc:min((i+1)*self.n_proc, self.cmaes_args["populate_num"])]
			arguments = arguments.tolist()
			arguments = [[x] for x in arguments] # input convention for starmap...
			# IPython.embed()
			results = pool.starmap(self.evaluate, arguments)

			for result in results:
				reward, debug_dict = result
				rewards.append(reward)
				if all_debug_dicts is None:
					all_debug_dicts = {k: [v] for (k, v) in debug_dict.items()}
				else:
					for k, v in all_debug_dicts.items():
						all_debug_dicts[k].append(debug_dict[k])"""

		for pop_member in self.population:
			reward, debug_dict = self.evaluate(pop_member)

			rewards.append(reward)
			if all_debug_dicts is None:
				all_debug_dicts = {k: [v] for (k, v) in debug_dict.items()}
			else:
				for k, v in all_debug_dicts.items():
					all_debug_dicts[k].append(debug_dict[k])

		# rewards = np.array(rewards)
		indexes = np.argsort(-np.array(rewards))
		"""
		===============================================================================
		best members are the top self.cmaes_args["elite_ratio"] proportion of members with the highest 
		evaluation rewards.
		"""
		best_members = self.population[indexes[0:int(self.cmaes_args["elite_ratio"] * self.cmaes_args["populate_num"])]]
		mu = np.mean(best_members, axis=0)

		# IPython.embed()
		sigma = np.cov(best_members.T) + self.noise
		# except:
		#     IPython.embed()
		# print("avg best mu in this epoch:")
		# print(mu)

		# Logging
		# print("ln 112, logging")
		# print("Check that data dict is being created properly (new data is scalar or list)")
		# IPython.embed()
		for k, v in all_debug_dicts.items():
			if k not in self.data:
				self.data[k] = []
			self.data[k].append(v)

		self.data["pop"].append(self.population)
		self.data["rewards"].append(rewards)
		self.data["mu"].append(mu)
		self.data["sigma"].append(sigma)

		# data = {}
		# data["pop"] = self.population
		# data["rewards"] = rewards
		# data["mu"] = mu
		# data["sigma"] = sigma
		# data.update(all_debug_dicts)

		with open(os.path.join(self.log_fldrpth, "data.pkl"), 'wb') as handle:
			pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# IPython.embed()
		return mu, sigma

	def learn(self):
		# print("inside learn")
		# IPython.embed()
		mu = self.cmaes_args["init_params"]
		bound_range = np.array(self.cmaes_args["upper_bound"]) - np.array(self.cmaes_args["lower_bound"])
		sigma = np.diag((self.cmaes_args["init_sigma_ratio"] * bound_range) ** 2)
		self.noise = np.diag((self.cmaes_args["noise_ratio"] * bound_range) ** 2)

		for i in range(self.cmaes_args["epoch"]):
			t0 = time.perf_counter()
			mu, sigma = self.step(mu, sigma)
			tf = time.perf_counter()
			print("epoch took %.3f s" % (tf - t0))
			# print("learning")
			print("mu:", mu)
			print("sigma:", sigma)
			print(self.evaluate(mu)[0])

		# print("Final best param:")
		# print(mu)
		# print("Final reward:")
		# print(self.evaluate(mu)[0])
		# self.evaluator.visualize(mu) # TODO: what happened to this function? No longer exists

		return mu


if __name__ == "__main__":
	# from cmaes.cmas_argument import create_parse
	import argparse
	parser = argparse.ArgumentParser(description='CMAES algorithm on CBF parameters')

	# Parameters
	parser.add_argument('--epoch', default=3, type=int)
	parser.add_argument('--elite_ratio', default=0.06, type=float)
	parser.add_argument('--populate_num', default=50, type=int)

	parser.add_argument('--lower_bound', nargs='+', default=[1.0, 0.0, 0.01], type=float, help="first, third required > 0") # TODO; c1 LB is 1.0 or 0.0?
	parser.add_argument('--upper_bound', nargs='+', default=[50.0, 50.0, 50.0], type=float, help="Tianhao use very different UB?") # TODO: perhaps range too large?
	parser.add_argument('--init_params', nargs='+', default=[1.0, 0.0, 1.0], type=float, help="exponent, added scalar, multiplier on dot term") # TODO: better default? Although, does it matter much if sigma=0.3?

	parser.add_argument('--init_sigma_ratio', default=0.3, type=float, help="initial sigma = init_sigma_ratio * (upper_bound - lower_bound)")
	parser.add_argument('--noise_ratio', default=0.01, type=int, help="noise = noise_ratio * (upper_bound - lower_bound)")

	parser.add_argument('--evaluator', default="FlyingPendEvaluator", type=str)
	# parser.add_argument('--exp_prefix', default="flying_pend", type=str)
	parser.add_argument('--exp_name', default="debug", type=str)

	# Objective specific
	parser.add_argument('--FlyingPendEvaluator_reg_weight', default=1.0, type=float)
	parser.add_argument('--FlyingPendEvaluator_n_samples', default=100000, type=int)
	parser.add_argument('--FlyingPendEvaluator_objective_type   ', default="n_feasible", type=int, choices=["n_feasible", "avg_amount_infeasible", "max_amount_infeasible"]) # note: we are maximizing
	parser.add_argument('--FlyingPendEvaluator_near_boundary_eps', default=1e-2, type=float, help="abs(phi) <= eps defines samples considered on the boundary")

	# parser = create_parser()
	args = parser.parse_known_args()[0]
	arg_dict = vars(args)
	# IPython.embed()

	arg_dict["evaluator"] = evaluators_dict[arg_dict["evaluator"]](arg_dict)  # pass a class instance

	learner = CMAESLearning(arg_dict)
	mu = learner.learn()


	"""
	python baseline_run_cmaes.py --FlyingPendEvaluator_n_samples 150
	
	python baseline_run_cmaes.py
	"""