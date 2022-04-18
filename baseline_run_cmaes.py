import numpy as np
import time
import subprocess
import os, sys
from datetime import datetime
# import yaml
import pickle, IPython

from src.cmas_argument import create_parser
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

		# print("in CMAES learning obj init")
		# IPython.embed()

		# now = datetime.now()
		# timestamp = now.strftime("%m-%d_%H:%M")
		# log_name = CMAES_args["exp_prefix"] + "_epoch:" + str(CMAES_args["epoch"]) + "_populate_num:" + str(CMAES_args["populate_num"]) + "_elite_ratio:" + str(CMAES_args["elite_ratio"]) + "_init_sigma_ratio:" + str(CMAES_args["init_sigma_ratio"]) + "_noise_ratio:" + str(CMAES_args["noise_ratio"]) + "_date:" + timestamp
		# self.log = open(os.path.dirname(os.path.abspath(__file__)) + "/cma_es_logs/" + log_name + ".txt","w")

		# Logging
		self.log_fldrpth = "flying_pend_" + CMAES_args["exp_name"] # Hard-coded the prefix
		if not os.path.exists(self.log_fldrpth):
			os.makedirs(self.log_fldrpth)
		print("Saving data at: %s" % self.log_fldrpth)

		with open(os.path.join(self.log_fldrpth, "args.pkl"), 'wb') as handle:
			pickle.dump(CMAES_args, handle, protocol=pickle.HIGHEST_PROTOCOL)

		self.data_dict = {"pop": [], "rewards": [], "mu": [], "sigma": []}

		self.evaluator = self.cmaes_args["evaluator"]

		# print(self.cmaes_args)
		# IPython.embed()

	def regulate_params(self, params):
		"""
		================================================================================
		Regulate params by upper bound and lower bound. And convert params to integer if required by the user.
		"""
		params = np.maximum(params, self.cmaes_args["lower_bound"]) # lower bound
		params = np.minimum(params, self.cmaes_args["upper_bound"]) # upper bound
		if "param_is_int" in self.cmaes_args:
			for i in range(params.shape[1]):
				if self.cmaes_args["param_is_int"][i]:
					params[:,[i]] = np.vstack([int(round(x)) for x in params[:,i]])
			# params = [ int(params[i]) if self.cmaes_args["param_is_int"][i] else params[i] ]
		return params

	def populate(self, mu, sigma):
		"""
		================================================================================
		Populate n members using the current estimates of mu and S
		"""
		self.population = np.random.multivariate_normal(mu, sigma, self.cmaes_args["populate_num"])
		self.population = self.regulate_params(self.population)

	def evaluate(self, mu, log=True):
		"""
		===============================================================================
		Evaluate a set of weights (a mu) by interacting with the environment and
		return the average total reward over multiple repeats.
		"""

		rewards = []
		repeat_times = 1  # test multiple times to reduce randomness
		for i in range(repeat_times):
			reward = self.evaluator.evaluate(mu)
			rewards.append(reward)

		print('Rewards: {}'.format(rewards))

		reward = np.mean(rewards)
		if log:
			self.log.write("{} {}".format(str(mu), reward))
			self.log.write(self.evaluator.log+"\n")
			self.log.flush()
		return reward

	def step(self, mu, sigma):
		"""
		===============================================================================
		Perform an iteration of CMA-ES by evaluating all members of the current
		population and then updating mu and S with the top self.cmaes_args["elite_ratio"] proportion of members
		and updateing the weights of the policy networks.
		"""
		self.populate(mu, sigma)
		rewards = np.array(list(map(self.evaluate, self.population)))
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
		IPython.embed()
		self.data["pop"].append(self.population)
		self.data["rewards"].append(self.rewards)
		self.data["mu"].append(mu)
		self.data["sigma"].append(sigma)

		return mu, sigma

	def learn(self):
		# print("inside learn")
		# IPython.embed()
		mu = self.cmaes_args["init_params"]
		bound_range = np.array(self.cmaes_args["upper_bound"]) - np.array(self.cmaes_args["lower_bound"])
		sigma = np.diag((self.cmaes_args["init_sigma_ratio"] * bound_range)**2)
		self.noise = np.diag((self.cmaes_args["noise_ratio"] * bound_range)**2)

		for i in range(self.cmaes_args["epoch"]):
			self.log.write("epoch {}\n".format(i))
			mu, sigma = self.step(mu, sigma)
			print("learning")

		print("Final best param:")
		print(mu)
		print("Final reward:")
		print(self.evaluate(mu, log=False))
		# self.evaluator.visualize(mu)

		# Note: add save
		# save_fpth = "./cma_es_results/%s" %
		# with open(save_fpth, 'wb') as handle:
		#     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		return mu

# def run_cmaes(config_path):
# 	with open(config_path, 'r') as stream:
# 		try:
# 			print("in run_cmaes() of baseline_run_cmaes.py")
# 			IPython.embed()
# 			config = yaml.safe_load(stream)
# 			# config["evaluator"] = eval("cma_es_evaluator."+config["evaluator"]+"()")
# 			config["evaluator"] = evaluators_dict[config["evaluator"]]
# 			learner = CMAESLearning(config)
# 			mu = learner.learn() # TODO: key line
# 			return mu
# 		except yaml.YAMLError as exc:
# 			print(exc)
# 			return None



if __name__ == "__main__":

	# if len(sys.argv) < 2:
	#     print("===============================================================================")
	#     print("Please pass in the learning config file path. Pre-defined files are in config")
	#     print("===============================================================================")
	# run_cmaes(sys.argv[1])

	parser = create_parser()
	args = parser.parse_args()
	arg_dict = vars(args)

	# print("In main of baseline_run_cmaes.py")
	# print(arg_dict)
	# IPython.embed()
	# Is this a dict?
	arg_dict["evaluator"] = evaluators_dict[arg_dict["evaluator"]]
	learner = CMAESLearning(arg_dict)
	mu = learner.learn()

