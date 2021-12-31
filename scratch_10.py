import matplotlib.pyplot as plt
import numpy as np
import IPython
import math

# def reg(x_arr):
# 	x_arr = x_arr + 1.586586586586587
# 	softplus = np.log(1 + np.exp(x_arr))
# 	sigmoid = 1 / (1.0 + np.exp(-x_arr))
# 	rv = -(sigmoid + 0.1*softplus) + 1
# 	return rv
#
# # print(reg(np.array([0])))
#
# x_vals = np.linspace(-5, 30, 1000)
# y_vals = reg(x_vals)
#
# # ind= np.argmin(np.abs(y_vals))
# # print(y_vals[ind-1], y_vals[ind], y_vals[ind+1])
# # print(ind)
# # print(x_vals[ind+1])
#
# # IPython.embed()
# plt.plot(x_vals, y_vals)
# plt.show()

# from src.utils import *
# args = load_args("./log/cartpole_reduced_64_64_60pts_gradient_avging_seed_1/args.txt")
# print(args.reg_relu_weight) # 0.1

#################
def plot_regularization_samples(n_mesh_grain=0.1):
	max_angular_velocity = 5.0
	x_lim = np.array([[-math.pi, math.pi], [-max_angular_velocity, max_angular_velocity]], dtype=np.float32)
	XXX = np.meshgrid(*[np.arange(r[0], r[1], n_mesh_grain) for r in x_lim])
	A_samples = np.concatenate([x.flatten()[:, None] for x in XXX], axis=1)

	print("N samples: %i" % A_samples.shape[0])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(A_samples[:, 0], A_samples[:, 1], s=0.5)
	# ax.imshow(phi_signs, extent=x_lim.flatten())
	ax.set_aspect("equal")

	plt.show()
	# IPython.embed()

if __name__ == "__main__":
	plot_regularization_samples(0.2)

