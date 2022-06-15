"""
Script for checking our solution against the MPC solution
"""
import numpy as np

from flying_rollout_experiment import *
from phi_numpy_wrapper import PhiNumpy
import matplotlib.pyplot as plt

def check_our_2D_slice(args):
	torch_phi_fn, param_dict = load_phi_and_params(exp_name=args.exp_name_to_load,
	                                               checkpoint_number=args.checkpoint_number_to_load)
	state_index_dict = param_dict["state_index_dict"]
	numpy_phi_fn = PhiNumpy(torch_phi_fn)

	env = FlyingInvertedPendulumEnv(param_dict)
	env.dt = args.rollout_dt
	cbf_controller = CBFController(env, numpy_phi_fn, param_dict)  # 2nd arg prev. "cbf_obj"

	#####################################
	# Create x0
	#####################################
	which_params = args.which_params
	x_lim = param_dict["x_lim"]
	ind1 = state_index_dict[which_params[0]]
	ind2 = state_index_dict[which_params[1]]

	x = np.arange(x_lim[ind1, 0], x_lim[ind1, 1], args.delta)
	y = np.arange(x_lim[ind2, 0], x_lim[ind2, 1], args.delta)[::-1]  # need to reverse it
	X, Y = np.meshgrid(x, y)

	sze = X.size
	input = np.zeros((sze, 16))
	input[:, ind1] = X.flatten()
	input[:, ind2] = Y.flatten()
	phi_vals = numpy_phi_fn.phi_fn(input) # can take (*, 16) input

	ind_S_interior = np.argwhere(np.max(phi_vals, axis=1) <= 0)[:, 0]
	N_rollout = len(ind_S_interior)

	info_dicts = None
	T_max = args.rollout_T_max
	N_steps_max = int(T_max / args.rollout_dt)

	# IPython.embed()
	# print("ln 45: before rollouts")
	print("Total rollouts: %i" % len(ind_S_interior))
	for i, ind in enumerate(ind_S_interior):
		print("Rollout %i" % i)
		info_dict = simulate_rollout(env, N_steps_max, cbf_controller, input[ind])
		if not np.any(info_dict["apply_u_safe"]):
			print("Error: rollouts not long enough")
			IPython.embed()

		if info_dicts is None:
			info_dicts = {key: [value] for (key, value) in info_dict.items()}
		else:
			for key, value in info_dicts.items():
				value.append(info_dict[key])

	# inside = info_dicts["inside_boundary"] # (n_rollouts, t_max)?
	# print("ln 57: done with rollouts")
	# IPython.embed()
	on = info_dicts["on_boundary"]
	outside = info_dicts["outside_boundary"]

	on_out_rl = [np.sum(x) for x in [on[i][:-1] * outside[i][1:] for i in range(N_rollout)]]
	# IPython.embed()
	safe_set_mask = np.zeros(sze)
	safe_set_mask[ind_S_interior] = np.logical_not(on_out_rl)
	safe_set_mask = np.reshape(safe_set_mask, X.shape)

	candidate_safe_set_mask = np.zeros(sze)
	candidate_safe_set_mask[ind] = 1.0
	candidate_safe_set_mask = np.reshape(candidate_safe_set_mask, X.shape)

	# create image
	image = 0.5*safe_set_mask + 0.5*candidate_safe_set_mask

	# IPython.embed()
	fig, ax = plt.subplots()
	ax.imshow(image, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
	ax.set_aspect("equal")

	plt.savefig("./sanity_check_outputs/sanity_check_mpc.png")

if __name__ == "__main__":
	# from cmaes.cmas_argument import create_parse
	import argparse

	parser = argparse.ArgumentParser(description='All experiments for flying pendulum')
	parser.add_argument('--which_params', default=["phi", "dphi"], nargs='+', type=str, help="which 2 state variables")
	parser.add_argument('--delta', type=float, default=0.1, help="discretization of grid over slice")
	parser.add_argument('--rollout_dt', type=float, default=1e-4)
	parser.add_argument('--rollout_T_max', type=float, default=1.0)

	parser.add_argument('--exp_name_to_load', type=str) # flying_inv_pend_first_run
	parser.add_argument('--checkpoint_number_to_load', type=int, help="for our CBF", default=0)

	args = parser.parse_known_args()[0]

	# IPython.embed()
	check_our_2D_slice(args)

	# python sanity_check_mpc.py --which_params theta dtheta --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250