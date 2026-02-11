"""Main entry point for nCBF training.
Sets up the training environment from command-line arguments and launches the learner-critic training loop.
"""
import os
import pickle

import torch
import numpy as np

from problems.quad_pend import create_quad_pend_param_dict
from src.create_arg_parser import create_arg_parser, print_args
from src.neural_phi import NeuralPhi
from src.critic import Critic
from src.learner import Learner
from src.reg_sampler import RegSampler
from src.saturation_risk import SaturationRisk
from src.reg_loss import RegularizationLoss
from src.utils import makedirs, create_logger, save_args, TransformEucNNInput


def main(args):
	save_folder = '%s_%s' % (args.problem, args.affix)

	log_folder = os.path.join(args.log_root, save_folder)
	model_folder = os.path.join(args.model_root, save_folder)

	makedirs(log_folder)
	makedirs(model_folder)

	setattr(args, 'log_folder', log_folder)
	setattr(args, 'model_folder', model_folder)

	logger = create_logger(log_folder, 'train', 'info')
	print_args(args, logger)

	args_savepth = os.path.join(log_folder, "args.txt")
	save_args(args, args_savepth)

	# Device
	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
		dev = "cuda:%i" % (args.gpu)
	else:
		raise NotImplementedError
	device = torch.device(dev)

	# Problem setup
	if args.problem == "quad_pend":
		param_dict = create_quad_pend_param_dict(args)

		r = param_dict["r"]
		x_dim = param_dict["x_dim"]
		u_dim = param_dict["u_dim"]
		x_lim = param_dict["x_lim"]

		from src.problems.quad_pend import Rho, XDot, ULimitSetVertices
		rho_fn = Rho(param_dict)
		xdot_fn = XDot(param_dict, device)
		uvertices_fn = ULimitSetVertices(param_dict, device)

		reg_sampler = RegSampler(x_lim, device, n_samples=args.reg_n_samples)

		x_e = torch.zeros(1, x_dim)

		state_index_dict = param_dict["state_index_dict"]
		nn_input_modifier = TransformEucNNInput(state_index_dict)
	else:
		raise NotImplementedError

	# Save param_dict
	with open(os.path.join(log_folder, "param_dict.pkl"), 'wb') as handle:
		pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Send all modules to the correct device
	rho_fn = rho_fn.to(device)
	xdot_fn = xdot_fn.to(device)
	uvertices_fn = uvertices_fn.to(device)
	if x_e is not None:
		x_e = x_e.to(device)
	x_lim = torch.tensor(x_lim).to(device)

	# Create CBF and loss functions
	phi_star_fn = NeuralPhi(rho_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e, nn_input_modifier=nn_input_modifier)
	saturation_risk = SaturationRisk(phi_star_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args)
	reg_fn = RegularizationLoss(phi_star_fn, device, reg_weight=args.reg_weight)

	phi_star_fn = phi_star_fn.to(device)
	saturation_risk = saturation_risk.to(device)
	reg_fn = reg_fn.to(device)

	# Create critic and test critic
	critic = Critic(x_lim, device, logger, n_samples=args.critic_n_samples,
	                max_n_steps=args.critic_max_n_steps)
	test_critic = Critic(x_lim, device, logger, n_samples=args.critic_n_samples, # TODO: is this correct?
	                     max_n_steps=args.critic_max_n_steps)

	# Launch training
	learner = Learner(args, logger, critic, test_critic, reg_sampler, param_dict, device)
	learner.train(saturation_risk, reg_fn, phi_star_fn)


if __name__ == "__main__":
	parser = create_arg_parser()
	args = parser.parse_known_args()[0]

	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	main(args)
