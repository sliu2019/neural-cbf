from flying_plot_utils import *
from global_settings import *
from phi_numpy_wrapper import PhiNumpy
import torch

def plot_invariant_set_slices(phi_fn, param_dict, samples=None, rollouts=None, which_params=None, constants_for_other_params=None, fnm=None, fldr_path=None, pass_axs=None):
	"""
	Plots invariant set and (if necessary) projected boundary samples in 2D
	which_params: all or list of lists of length 2
	"""
	# IPython.embed()
	if rollouts is not None:
		print("Inside plot_invariant_set_slices() of fling_plot_utils")
		print("Have never debugged traj plotting using this")
		IPython.embed()

	x_lim = param_dict["x_lim"]
	x_dim = param_dict["x_dim"]
	state_index_dict = param_dict["state_index_dict"]

	if which_params is None:
		# params_to_viz = [["theta", "phi"], ["dtheta", "dphi"], ["theta", "dtheta"], ["phi", "dphi"], ["beta", "alpha"], ["gamma", "beta"], ["gamma", "alpha"], ["dbeta", "dalpha"], ["dgamma", "dbeta"], ["dgamma", "dalpha"]]
		# Default is intelligent plotting
		constants_for_other_params = []
		params_to_viz = []

		# Test: does it matter if pend angle and pend velocity are aligned?
		params_to_viz.extend([["phi", "dphi"], ["theta", "dtheta"], ["gamma", "dgamma"], ["beta", "dbeta"]])
		constants_for_other_params.extend([np.zeros(10)]*4)

		# Test: does it matter if pend and quad angle are aligned/not aligned?
		params_to_viz.append(["gamma", "phi"])
		constants_for_other_params.append(np.zeros(10))

		# More dangerous
		params_to_viz.append(["gamma", "phi"])
		x = np.zeros(10)
		x[state_index_dict["dgamma"]] = 5
		constants_for_other_params.append(x)

		# params_to_viz.append(["gamma", "phi"])
		# x = np.zeros(10)
		# x[state_index_dict["dphi"]] = 5
		# constants_for_other_params.append(x)

		params_to_viz.append(["beta", "theta"])
		constants_for_other_params.append(np.zeros(10))

		# More dangerous
		params_to_viz.append(["beta", "theta"])
		x = np.zeros(10)
		x[state_index_dict["dbeta"]] = 5
		constants_for_other_params.append(x)
	else:
		params_to_viz = which_params

	# IPython.embed()
	n_per_row = 2 # TODO
	n_row = math.ceil(len(params_to_viz)/float(n_per_row))

	n_per_row = min(len(params_to_viz), n_per_row)
	if not pass_axs:
		_, axs = plt.subplots(n_row, n_per_row, squeeze=False, figsize=(6, 16)) # TODO: remove figsize
	else:
		axs = pass_axs
	# IPython.embed()
	# axs[0, 0].plot(x, y)
	for i in range(n_row):
		for j in range(n_per_row):
			if i*n_per_row + j >= len(params_to_viz):
				break
			# for param_pair in params_to_viz:
			param_pair = params_to_viz[i*n_per_row + j]
			param1, param2 = param_pair

			ind1 = state_index_dict[param1]
			ind2 = state_index_dict[param2]

			delta = 0.01
			x = np.arange(x_lim[ind1, 0], x_lim[ind1, 1], delta)
			y = np.arange(x_lim[ind2, 0], x_lim[ind2, 1], delta)[::-1] # need to reverse it # TODO
			X, Y = np.meshgrid(x, y)

			###################################
			##### Plotting ######
			###################################
			## Get phi values
			if constants_for_other_params:
				# print("flying_plot_utils.py, ln 162")
				# IPython.embed()
				input = constants_for_other_params[i*n_per_row + j]
				# print(input)
				input = np.reshape(input, (1, -1))
				input = np.tile(input, (X.size, 1))
			else:
				input = np.zeros((X.size, x_dim))
			input[:, ind1] = X.flatten()
			input[:, ind2] = Y.flatten()

			batch_size = int(X.size/5)
			all_size = input.shape[0]

			phi_vals = []
			for k in range(math.ceil(all_size/batch_size)):
				batch_input = input[k*batch_size: min(all_size, (k+1)*batch_size)]
				batch_input = batch_input.astype("float32")
				batch_input_torch = torch.from_numpy(batch_input)

				batch_phi_vals = phi_fn(batch_input_torch)
				phi_vals.append(batch_phi_vals.detach().cpu().numpy())

			## Process phi values
			phi_vals = np.concatenate((phi_vals), axis=0)
			S_vals = np.max(phi_vals, axis=1)  # S = all phi_i <= 0
			phi_signs = np.sign(S_vals)
			phi_signs = np.reshape(phi_signs, X.shape)


			print("Any negative phi in the box?: ", np.any(phi_signs < 0))
			# TODO: Add better colors for CORL
			# IPython.embed()
			if not pass_axs:
				# print("ln 123")
				# IPython.embed()
				red_rgba = np.append(red_rgb, 0.5)
				dark_purple_rgb = np.array([114,81,150])/255.0
				purple_rgba = np.append(np.array([125,99,167])/255.0, 0.8)

				# dark_purple_rgb = np.array([119, 101, 140])/255.0
				# purple_rgba = np.append(np.array([199, 169, 234])/255.0, 0.8)

				# dark_purple_rgb = np.array([255, 217, 152])/255.0
				# purple_rgba = np.append(np.array([255, 252, 152])/255.0, 0.8)

				img = np.zeros((phi_signs.shape[0], phi_signs.shape[1], 4))
				red_inds = np.argwhere(phi_signs == 1)
				img[red_inds[:, 0], red_inds[:, 1], :] = red_rgba
				blue_inds = np.argwhere(phi_signs == -1)
				img[blue_inds[:, 0], blue_inds[:, 1], :] = purple_rgba

				axs[i, j].imshow(img, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
				# axs[i, j].set_aspect("equal")
				# axs[i, j].set_aspect("box")
				axs[i, j].set_aspect(2.0 / axs[i, j].get_data_ratio(), adjustable='box')

				# phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
				axs[i, j].contour(X, Y, np.reshape(phi_vals[:, -1], X.shape), levels=[0.0],
				                  colors=([np.append(dark_purple_rgb, 1.0)]), linewidths=(2,), zorder=1)
			else:
				# print("ln 141")
				# IPython.embed()
				red_rgba = np.append(red_rgb, 0.5)
				blue_rgba = np.append(blue_rgb, 0.7)

				img = np.zeros((phi_signs.shape[0], phi_signs.shape[1], 4))
				red_inds = np.argwhere(phi_signs == 1)
				img[red_inds[:, 0], red_inds[:, 1], :] = red_rgba
				blue_inds = np.argwhere(phi_signs == -1)
				img[blue_inds[:, 0], blue_inds[:, 1], :] = blue_rgba

				axs[i, j].imshow(img, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]])
				# axs[i, j].set_aspect("equal")
				# axs[i, j].set_aspect("box")
				axs[i, j].set_aspect(2.0 / axs[i, j].get_data_ratio(), adjustable='box')

				# phi_vals_numpy = phi_vals[:, -1].detach().cpu().numpy()
				axs[i, j].contour(X, Y, np.reshape(phi_vals[:, -1], X.shape), levels=[0.0],
				           colors=([np.append(dark_blue_rgb, 1.0)]), linewidths=(2,), zorder=1)

			# Old below
			"""axs[i, j].imshow(phi_signs, extent=[x_lim[ind1, 0], x_lim[ind1, 1], x_lim[ind2, 0], x_lim[ind2, 1]], vmin=-1.0, vmax=1.0)
			axs[i, j].set_aspect("equal")
			axs[i, j].contour(X, Y, np.reshape(phi_vals[:, -1], X.shape), levels=[0.0],
							 colors=('k',), linewidths=(2,))"""


			## Plotting the sampled points
			if samples is not None:
				axs[i, j].scatter(samples[:, ind1], samples[:, ind2], s=0.5) # projection (truncation)


			## Plotting the included trajectories
			if rollouts is not None:
				N_rollout = len(rollouts) # rollouts is a ist
				for i in range(N_rollout):
					ith_rl = rollouts[i]
					axs[i, j].plot(ith_rl[:, ind1], ith_rl[:, ind2])

			## Title
			# title = "%s vs. %s" % (param1, param2)
			axs[i, j].set_xlabel(param1)
			axs[i, j].set_ylabel(param2)
			# TODO: uncomment
			# if constants_for_other_params:
			# 	const = constants_for_other_params[i * n_per_row + j]
			# 	axs[i, j].set_title(const)

	if fnm is None:
		fnm = time.strftime('%m_%d_%H:%M:%S')
	if fldr_path is None:
			fldr_path = "./log/boundary_sampling"

	# if fpth is not None:
	# 	save_fpth = "./log/%s/%s.png" % (fpth, fnm)
	# else:
	# 	save_fpth = "./log/boundary_sampling/%s.png" % fnm
	save_fpth = os.path.join(fldr_path, fnm + ".png")

	print("Saved at: %s" % save_fpth)
	# plt.tight_layout(pad=0.5)

	# IPython.embed()
	if hasattr(phi_fn, "k0"):
		# cbf is type "ours"
		ki_str = "k0 = %.4f, k1 = %.4f" % (phi_fn.k0, phi_fn.ci[0])
	elif hasattr(phi_fn, "ki"):
		ki_str = "ci = %.4f, %.4f, ki = %.4f" % (phi_fn.ci[0, 0], phi_fn.ci[1, 0], phi_fn.ki[0, 0])
	else:
		ki_str = ""
	# fig.title(ki_str) # doesn't work, title goes on last subfigure

	# TODO: uncomment
	# if checkpoint: # little messy, little hacky, doesn't matter tho
	# 	plt.suptitle("From %s, ckpt %i \n %s" % (fldr_path, checkpoint, ki_str))
	# else:
	# 	plt.suptitle("From %s \n %s" % (fldr_path, ki_str))

	plt.savefig(save_fpth, bbox_inches='tight')
	# plt.clf()
	# plt.close()

	return axs

def load_philow_and_params(exp_name=None, checkpoint_number=None):
	"""
	Same behavor as flying_plot_utils.py/load_phi_and_params
	:param exp_name:
	:param checkpoint_number:
	:return:
	"""
	# TODO: assume default param_dict even if we pass in exp_name, checkpoint_number
	# if exp_name:
	# 	fnm = "./log/%s/args.txt" % exp_name
	# 	# args = load_args(fnm) # can't use, args conflicts with args in outer scope
	# 	with open(fnm, 'r') as f:
	# 		json_data = json.load(f)
	# 	args = DotMap(json_data)
	# 	param_dict = pickle.load(open("./log/%s/param_dict.pkl" % exp_name, "rb"))
	# else:
	from src.argument import create_parser
	parser = create_parser() # default
	args = parser.parse_known_args()[0]

	from main import create_flying_param_dict
	param_dict = create_flying_param_dict(args) # default

	r = param_dict["r"]
	x_dim = param_dict["x_dim"]
	u_dim = param_dict["u_dim"]
	x_lim = param_dict["x_lim"]


	# Create phi
	from src.problems.flying_inv_pend import HMax, HSum, XDot
	if args.h == "sum":
		h_fn = HSum(param_dict)
	elif args.h == "max":
		h_fn = HMax(param_dict)

	xdot_fn = XDot(param_dict, device)

	# Send all modules to the correct device
	h_fn = h_fn.to(device)
	xdot_fn = xdot_fn.to(device)

	# Create CBF, etc.
	phi_fn = LowPhi(h_fn, xdot_fn, x_dim, u_dim, device, param_dict)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)

	if exp_name:
		print("inside loadphilow_and_params, reloading model part")
		IPython.embed()
		data = pickle.load(open(os.path.join("./cmaes", exp_name, "data.pkl")))
		mus = data["mu"]
		mu = mus[checkpoint_number]
		state_dict = {"ki": torch.tensor([[mu[2]]]), "ci": torch.tensor([[mu[0]], [mu[1]]])}
		phi_fn.load_state_dict(state_dict, strict=False)

	# TODO: not loading a checkpoint
	# if phi_load_fpth:
	# 	load_model(phi_fn, phi_load_fpth)

	return phi_fn, param_dict

if __name__ == "__main__":
	exp_name = "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0"
	checkpoint_number = 250

	# params_to_viz_list = [["phi", "dphi"]] # TODO
	# params_to_viz_list = [["beta", "dbeta"]] # TODO
	params_to_viz_list = [["gamma", "dgamma"]] # TODO
	# params_to_viz_list = [["theta", "dtheta"]] # TODO
	# params_to_viz_list = [["dtheta", "dbeta"]] # TODO
	# params_to_viz_list = [["dphi", "dgamma"]] # TODO

	# fnm = "beta_dbeta_slice" # TODO
	# fnm = "theta_dtheta_slice" # TODO
	# fnm = "theta_dtheta_larger_slice" # TODO
	# fnm = "dtheta_dbeta_slice" # TODO
	# fnm = "dphi_dgamma_slice" # TODO
	fnm = "gamma_dgamma_larger_slice" # TODO
	fldr_path = os.path.join("./log", exp_name)

	ub = 15.0
	thresh = np.array([math.pi / 3, math.pi / 3, math.pi, ub, ub, ub, math.pi / 3, math.pi / 3, ub, ub],
	                  dtype=np.float32) # angular velocities bounds probably much higher in reality (~10-20 for drone, which can do 3 flips in 1 sec).

	x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)  # (13, 2)
	constants_for_other_params_list = [np.zeros(10)]
	###############################
	# Now, the baseline
	baseline_phi_fn, baseline_param_dict = load_philow_and_params()  # TODO: this assumes default param_dict for dynamics
	baseline_param_dict["x_lim"] = x_lim

	mu = [3.75977875, 0., 0.01]
	state_dict = {"ki": torch.tensor([[mu[2]]]),
	              "ci": torch.tensor([[mu[0]], [mu[1]]])}  # todo: this is not very generic
	baseline_phi_fn.load_state_dict(state_dict, strict=False)

	axs = plot_invariant_set_slices(baseline_phi_fn, baseline_param_dict, fldr_path=fldr_path,
	                                which_params=params_to_viz_list,
	                                constants_for_other_params=constants_for_other_params_list, fnm=fnm)

	###############################
	phi_fn, param_dict = load_phi_and_params(exp_name, checkpoint_number)
	param_dict["x_lim"] = x_lim
	axs = plot_invariant_set_slices(phi_fn, param_dict, fldr_path=fldr_path, which_params=params_to_viz_list,
	                          constants_for_other_params=constants_for_other_params_list, fnm=fnm, pass_axs=axs)

	# print("ln 235")
	# IPython.embed()

