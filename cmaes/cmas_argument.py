import argparse
import math

def create_parser():
	# Problem
	parser = argparse.ArgumentParser(description='CMAES algorithm on CBF parameters')

	# Parameters
	parser.add_argument('--epoch', default=3, type=int)
	parser.add_argument('--elite_ratio', default=0.06, type=float)
	parser.add_argument('--populate_num', default=50, type=int)

	parser.add_argument('--init_params', nargs='+', default=[1.0, 0.0, 1.0], type=float, help="exponent, added scalar, multiplier on dot term")
	parser.add_argument('--lower_bound', nargs='+', default=[0.01, 0.0, 0.01], type=float, help="first, third required > 0")
	parser.add_argument('--upper_bound', nargs='+', default=[50.0, 5.0, 50.0], type=float, help="Tianhao use very different UB?")

	parser.add_argument('--init_sigma_ratio', default=0.3, type=float, help="initial sigma = init_sigma_ratio * (upper_bound - lower_bound)")
	parser.add_argument('--noise_ratio', default=0.01, type=int, help="noise = noise_ratio * (upper_bound - lower_bound)")

	parser.add_argument('--evaluator', default="FlyingPendEvaluator", type=str)
	# parser.add_argument('--exp_prefix', default="flying_pend", type=str)
	parser.add_argument('--exp_name', default="debug", type=str)

	# SaturationRisk specific
	parser.add_argument('--FlyingPendEvaluator_reg_weight', default=1.0, type=int)
	parser.add_argument('--FlyingPendEvaluator_n_samples', default=10e4, type=int)

	return parser


# def print_args(args, logger=None):
# 	for k, v in vars(args).items():
# 		if logger is not None:
# 			logger.info('{:<16} : {}'.format(k, v))
# 		else:
# 			print('{:<16} : {}'.format(k, v))