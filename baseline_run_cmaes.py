import numpy as np
import os
# import yaml
import pickle, IPython
from cmaes.cmas_argument import create_parser
from cmaes_objective_flying_pend import FlyingPendEvaluator

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

		self.data_dict = {"pop": [], "rewards": [], "mu": [], "sigma": []}

		self.evaluator = self.cmaes_args["evaluator"]

		print("in init")
		IPython.embed()

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

		# reward - parameter relationship is not stochastic, so we don't need to evaluate it multiple times
		# rewards = []
		# repeat_times = 1  # test multiple times to reduce randomness
		# for i in range(repeat_times):
		# 	reward = self.evaluator.evaluate(mu)
		# 	rewards.append(reward)
		#
		# print('Rewards: {}'.format(rewards))
		#
		# reward = np.mean(rewards)

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
		# rewards = np.array(list(map(self.evaluate, self.population)))
		# Refactored, since self.evaluate returns a tuple now
		rewards = []
		all_debug_dicts = None
		for pop_member in self.population:
			reward, debug_dict = self.evaluate(pop_member)
			rewards.append(reward)
			if all_debug_dicts is None:
				all_debug_dicts = {k: list(v) for k, v in debug_dict.items()}
			else:
				for k, v in all_debug_dicts.items():
					all_debug_dicts[k].append(debug_dict[k])
		indexes = np.argsort(-rewards)
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
		print("avg best mu in this epoch:")
		print(mu)

		# Logging
		print("ln 112, logging")
		print("Check that data dict is being created properly (new data is scalar or list)")
		IPython.embed()
		for k, v in all_debug_dicts.items():
			if k not in self.data:
				self.data[k] = []
			self.data[k].append(v)

		self.data["pop"].append(self.population)
		self.data["rewards"].append(self.rewards)
		self.data["mu"].append(mu)
		self.data["sigma"].append(sigma)
		with open(os.path.join(self.log_fldrpth, "data.pkl"), 'wb') as handle:
			pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

		return mu, sigma

	def learn(self):
		# print("inside learn")
		# IPython.embed()
		mu = self.cmaes_args["init_params"]
		bound_range = np.array(self.cmaes_args["upper_bound"]) - np.array(self.cmaes_args["lower_bound"])
		sigma = np.diag((self.cmaes_args["init_sigma_ratio"] * bound_range) ** 2)
		self.noise = np.diag((self.cmaes_args["noise_ratio"] * bound_range) ** 2)

		for i in range(self.cmaes_args["epoch"]):
			mu, sigma = self.step(mu, sigma)
			print("learning")

		print("Final best param:")
		print(mu)
		print("Final reward:")
		print(self.evaluate(mu)[0])
		# self.evaluator.visualize(mu) # TODO: what happened to this function? No longer exists

		return mu


if __name__ == "__main__":
	parser = create_parser()
	args = parser.parse_args()
	arg_dict = vars(args)
	arg_dict["evaluator"] = evaluators_dict[arg_dict["evaluator"]](arg_dict)  # pass a class instance

	learner = CMAESLearning(arg_dict)
	mu = learner.learn()
