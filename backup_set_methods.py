"""
Simulates stabilization to an equilibrium
"""
import numpy as np
import control
import matplotlib.pyplot as plt
import IPython
import math
from rollout_envs.flying_inv_pend_env import FlyingInvertedPendulumEnv

# Physical parameters
g = 9.81
# m = 0.8
# Jx = 0.005
# Jy = 0.005
# Jz = 0.009
# l = 0.3 / 2
#
# k1 = 4.0
# k2 = 0.05

dt = 0.01

np.random.seed(0)

param_dict = {
	"m": 0.8,
	"J_x": 0.005,
	"J_y": 0.005,
	"J_z": 0.009,
	"l": 1.5,
	"k1": 4.0,
	"k2": 0.05,
	"m_p": 0.04, # 5% of quad weight
	"L_p": 3.0, # Prev: 0.03
	'delta_safety_limit': math.pi / 4  # should be <= math.pi/4
}
param_dict["M"] = param_dict["m"] + param_dict["m_p"]

ub = 20
thresh = np.array([math.pi / 3, math.pi / 3, math.pi, ub, ub, ub, math.pi / 3, math.pi / 3, ub, ub],
                  dtype=np.float32)  # angular velocities bounds probably much higher in reality (~10-20 for drone, which can do 3 flips in 1 sec).
x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)  # (13, 2)

env = FlyingInvertedPendulumEnv(param_dict)

state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
                     "dtheta", "x", "y", "z", "dx", "dy", "dz"]
state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))

def simulate_quadrotor(xdot_fn, u_fn, timesteps=1000):
	k1 = param_dict["k1"]
	k2 = param_dict["k2"]
	l = param_dict["l"]

	x_all = []
	u_impulses_all = []
	u_forces_all = []
	distances = []

	eps = np.random.normal(scale=0.5, size=10)  # let x0 be approximately in the 0-centered norm ball
	target = np.zeros(10)  # Stabilize to origin
	xcurr = target + eps
	xcurr = np.concatenate((xcurr, np.zeros(6))) # adding [x, y, z] and derivatives to the end

	x_all.append(xcurr)

	M = np.array(
		[[k1, k1, k1, k1], [0, -l * k1, 0, l * k1], [l * k1, 0, -l * k1, 0], [-k2, k2, -k2, k2]])  # Mixer matrix

	print("Initial state: ", xcurr)

	for t in range(timesteps):
		ucurr = u_fn(xcurr)  # call feedback controller

		# TODO
		# Ui = np.linalg.solve(M,
		#                      ucurr)  # Convert from high-level control inputs (forces, torques) to low-level (impulses)
		# Ui = np.clip(Ui, 0, 1)  # Enforce control limits
		# ucurr = M @ Ui  # Convert back to high-level control inputs
		_, debug_dict = env.clip_u(ucurr)
		Ui = debug_dict["smooth_clamped_motor_impulses"]

		xdot = xdot_fn(xcurr, ucurr)  # Call dynamics equation
		xnext = xcurr + np.squeeze(xdot) * dt  # Simulate system via finite differencing

		# Log data
		x_all.append(xnext)
		u_impulses_all.append(Ui)
		u_forces_all.append(ucurr)

		xcurr = xnext

		# IPython.embed()
		dist = np.linalg.norm(target - xcurr[0:10])
		distances.append(dist)

		print(dist)

	return x_all, u_impulses_all, u_forces_all, distances

def xdot_fn(x, u):
	xdot = env.x_dot_open_loop(x, u)
	return xdot

##### Animation stuff below
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation

def animate_quadcopter(r, param_dict, save_fpth):
	# Define parameters
	r = np.array(r)

	# Note: lengths are not proportional
	quad_axis_length = 1.0
	e1 = np.array([1.0, 0, 0])*quad_axis_length # note: will double the stated length
	e2 = np.array([0, 1.0, 0])*quad_axis_length

	pend_axis_length = quad_axis_length
	e3 = np.array([0, 0, 1.0])*pend_axis_length

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Define patches, etc.
	origin = np.zeros(3)
	quad_axis1 = art3d.Line3D([1, -1], [0, 0], [0, 0], color='k')
	quad_axis2 = art3d.Line3D([0, 0], [1, -1], [0, 0], color='k')
	pend = art3d.Line3D([0, 0], [0, 0], [0, 1], color='r')
	# traj, = ax.plot3D([], [], [], marker=".", color="grey")

	xs = r[:, state_index_dict["x"]]
	ys = r[:, state_index_dict["y"]]
	zs = r[:, state_index_dict["z"]]
	ax.plot(xs, ys, zs)

	set_axes_equal(ax)
	ax.set_box_aspect([1, 1, 1])

	# IPython.embed()
	def init():
		# Place everything onto axes
		ax.add_line(quad_axis1)
		ax.add_line(quad_axis2)
		ax.add_line(pend)

		rv = [quad_axis1, quad_axis2, pend]
		return rv


	def animate(i):
		# Update quadcopter 
		gamma = r[i, state_index_dict["gamma"]]
		beta = r[i, state_index_dict["beta"]]
		alpha = r[i, state_index_dict["alpha"]]

		phi = r[i, state_index_dict["phi"]]
		theta = r[i, state_index_dict["theta"]]

		R = np.zeros((3, 3))
		R[0, 0] = np.cos(alpha) * np.cos(beta)
		R[0, 1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
		R[0, 2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
		R[1, 0] = np.sin(alpha) * np.cos(beta)
		R[1, 1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
		R[1, 2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
		R[2, 0] = -np.sin(beta)
		R[2, 1] = np.cos(beta) * np.sin(gamma)
		R[2, 2] = np.cos(beta) * np.cos(gamma)

		r1 = R@e1
		r2 = R@e2
		p = r[i, [state_index_dict["x"], state_index_dict["y"], state_index_dict["z"]]]

		ax1 = np.concatenate([(p-r1)[:, None], (p+r1)[:, None]], axis=1)
		quad_axis1.set_data_3d(ax1[0], ax1[1], ax1[2])

		ax2 = np.concatenate([(p-r2)[:, None], (p+r2)[:, None]], axis=1)
		quad_axis2.set_data_3d(ax2[0], ax2[1], ax2[2])

		R_pend = np.zeros((3, 3))

		R_pend[0, 0] = np.cos(0) * np.cos(theta)
		R_pend[0, 1] = np.cos(0) * np.sin(theta) * np.sin(phi) - np.sin(0) * np.cos(phi)
		R_pend[0, 2] = np.cos(0) * np.sin(theta) * np.cos(phi) + np.sin(0) * np.sin(phi)
		R_pend[1, 0] = np.sin(0) * np.cos(theta)
		R_pend[1, 1] = np.sin(0) * np.sin(theta) * np.sin(phi) + np.cos(0) * np.cos(phi)
		R_pend[1, 2] = np.sin(0) * np.sin(theta) * np.cos(phi) - np.cos(0) * np.sin(phi)
		R_pend[2, 0] = -np.sin(theta)
		R_pend[2, 1] = np.cos(theta) * np.sin(phi)
		R_pend[2, 2] = np.cos(theta) * np.cos(phi)

		r3 = R_pend@e3
		ax3 = np.concatenate([p[:, None], (p+r3)[:, None]], axis=1)
		pend.set_data_3d(ax3[0], ax3[1], ax3[2])

		# Finally, plot the trajectory
		# traj.set_data(r[:i, state_index_dict["x"]], r[:i, state_index_dict["y"]])
		# traj.set_3d_properties(r[:i, state_index_dict["z"]], "z")

		rv = [quad_axis1, quad_axis2, pend]
		return rv

	# IPython.embed()
	print(r.shape[0])
	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=r.shape[0],
	                               blit=True)  # interval: real time, delay between frames in ms, 20
	FFwriter = animation.FFMpegWriter(fps=60)
	anim.save(save_fpth, writer=FFwriter, dpi=600)

def plot_fancy_3D(r, save_fpth):
	r = np.array(r)

	quad_axis_length = 1.0
	e1 = np.array([1.0, 0, 0])*quad_axis_length # note: will double the stated length
	e2 = np.array([0, 1.0, 0])*quad_axis_length

	pend_axis_length = quad_axis_length
	e3 = np.array([0, 0, 1.0])*pend_axis_length

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# ax.set_aspect("equal")

	# Option 2: aspect ratio 1:1:1 in view space
	# ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
	# set_axes_equal(ax)
	xs = r[:, state_index_dict["x"]]
	ys = r[:, state_index_dict["y"]]
	zs = r[:, state_index_dict["z"]]
	ax.plot(xs, ys, zs)

	set_axes_equal(ax)
	# ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
	ax.set_box_aspect([1, 1, 1])  # IMPORTANT - this is the new, key line

	skip_length = 50 # TODO: set skip length
	n_skip = r.shape[0]//skip_length
	# IPython.embed()
	for i in np.arange(0, r.shape[0], skip_length):
		gamma = r[i, state_index_dict["gamma"]]
		beta = r[i, state_index_dict["beta"]]
		alpha = r[i, state_index_dict["alpha"]]

		phi = r[i, state_index_dict["phi"]]
		theta = r[i, state_index_dict["theta"]]

		R = np.zeros((3, 3))
		R[0, 0] = np.cos(alpha) * np.cos(beta)
		R[0, 1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
		R[0, 2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
		R[1, 0] = np.sin(alpha) * np.cos(beta)
		R[1, 1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
		R[1, 2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
		R[2, 0] = -np.sin(beta)
		R[2, 1] = np.cos(beta) * np.sin(gamma)
		R[2, 2] = np.cos(beta) * np.cos(gamma)

		r1 = R@e1
		r2 = R@e2
		p = r[i, [state_index_dict["x"], state_index_dict["y"], state_index_dict["z"]]]

		# IPython.embed()
		# print(i/float(n_skip))
		ax1 = np.concatenate([(p-r1)[:, None], (p+r1)[:, None]], axis=1)
		quad_axis1 = art3d.Line3D(ax1[0], ax1[1], ax1[2], color='k', alpha=(float(i)/r.shape[0]))
		ax.add_line(quad_axis1)

		ax2 = np.concatenate([(p-r2)[:, None], (p+r2)[:, None]], axis=1)
		quad_axis2 = art3d.Line3D(ax2[0], ax2[1], ax2[2], color='k', alpha=(float(i)/r.shape[0]))
		ax.add_line(quad_axis2)

		R_pend = np.zeros((3, 3))
		R_pend[0, 0] = np.cos(0) * np.cos(theta)
		R_pend[0, 1] = np.cos(0) * np.sin(theta) * np.sin(phi) - np.sin(0) * np.cos(phi)
		R_pend[0, 2] = np.cos(0) * np.sin(theta) * np.cos(phi) + np.sin(0) * np.sin(phi)
		R_pend[1, 0] = np.sin(0) * np.cos(theta)
		R_pend[1, 1] = np.sin(0) * np.sin(theta) * np.sin(phi) + np.cos(0) * np.cos(phi)
		R_pend[1, 2] = np.sin(0) * np.sin(theta) * np.cos(phi) - np.cos(0) * np.sin(phi)
		R_pend[2, 0] = -np.sin(theta)
		R_pend[2, 1] = np.cos(theta) * np.sin(phi)
		R_pend[2, 2] = np.cos(theta) * np.cos(phi)

		r3 = R_pend@e3
		ax3 = np.concatenate([p[:, None], (p+r3)[:, None]], axis=1)
		pend = art3d.Line3D(ax3[0], ax3[1], ax3[2], color='r', alpha=(float(i)/r.shape[0]))
		ax.add_line(pend)

	plt.savefig(save_fpth)
	# plt.show()

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def batch_simulate_T_seconds(x_batch, xdot_fn, u_fn, T):
	for i in range(T/float(dt)):
		u_batch = u_fn(x_batch)
		x_batch = x_batch + dt*xdot_fn(x_batch, u_batch)

	return x_batch

def compute_backup_set(params_to_viz, xdot_fn, u_fn, save_fpth, T=3.0):
	tol = 1e-3
	# params_to_viz is a tuple of parameter names
	ind1, ind2 = params_to_viz

	delta = 0.01  # larger for 3D plotting, due to latency
	x = np.arange(x_lim[ind1, 0], x_lim[ind1, 1], delta)
	y = np.arange(x_lim[ind2, 0], x_lim[ind2, 1], delta)[::-1]  # need to reverse it # TODO
	X, Y = np.meshgrid(x, y)

	processing_batch_size = 100
	inside_implicit_ss = np.zeros(X.size)
	for i in range(math.ceil(X.size/processing_batch_size)):
		ind1_batch = X.flatten()[i * processing_batch_size:max(X.size, (i + 1) * processing_batch_size)]
		ind2_batch = Y.flatten()[i * processing_batch_size:max(X.size, (i + 1) * processing_batch_size)]
		x_batch = np.zeros((ind1_batch.size, 16))
		x_batch[:, state_index_dict[ind1]] = ind1_batch
		x_batch[:, state_index_dict[ind2]] = ind2_batch

		x_batch_T = batch_simulate_T_seconds(x_batch, xdot_fn, u_fn, T)
		x_batch_inside = np.linalg.norm(x_batch_T[:, :10], axis=1) < tol
		inside_implicit_ss[i * processing_batch_size:max(X.size, (i + 1) * processing_batch_size)] = x_batch_inside

	img = np.reshape(inside_implicit_ss, X.shape)

	# np.save('test3.npy', a)  # .npy extension is added if not given
	# d = np.load('test3.npy')
	np.save(save_fpth + ".npy", img)
	plt.plot(img)
	plt.savefig(save_fpth + ".png")
	plt.show()


if __name__ == "__main__":
	# IPython.embed()
	L_p = param_dict["L_p"]
	M = param_dict["M"]
	J_x = param_dict["J_x"]
	J_y = param_dict["J_y"]
	J_z = param_dict["J_z"]

	A = np.zeros((10, 10)) # 10 x 10
	A[0:3, 3:6] = np.eye(3)
	A[6:8, 8:10] = np.eye(2)
	A[8, 0] = -3*g/(2*L_p)
	A[9, 1] = -3*g/(2*L_p)
	A[8, 6] = 3*g/(2*L_p)
	A[9, 7] = 3*g/(2*L_p)

	# print(A)

	B = np.zeros((10, 4))
	B[3:6, 1:4] = np.diag([1.0/J_x, 1.0/J_y, 1.0/J_z])

	# print(B)

	# Check that the system is controllable
	# C = control.ctrb(A, B)
	# rk = np.linalg.matrix_rank(C)
	# assert rk == 10

	# Use LQR to compute feedback portion of controller
	q = 0.25 # 1 is the limit
	r = 1
	Q = q * np.eye(10)
	R = r * np.eye(4)
	K, S, E = control.lqr(A, B, Q, R)

	print(K)

	def u_fn(x):
		x = x[:10] # truncate, if necessary. control loop uses full 16D
		u = -K @ x
		return u

	# u_fn = lambda x: np.array([M * g, 0, 0, 0]) - K @ x  # control policy

	# Sanity check the dynamics
	# print(xdot_fn(np.zeros(16), np.array([0, 0, 0, 0]))) # check origin is equilibrium

	# IPython.embed()

	x_all, u_impulses_all, u_forces_all, distances = simulate_quadrotor(xdot_fn, u_fn, timesteps=250) # TODO: 2000

	# Plotting

	# Plot distance from origin over time
	plt.plot(np.arange(len(distances)) * dt, distances, linewidth=0.5)
	plt.title("Distance from origin, for q=%f, r=%f" % (q, r))
	plt.xlabel("Time so far")
	plt.savefig("backup_set_outputs/dist_q_%f_r_%f.png" % (q, r))
	# plt.show()
	plt.clf()

	# Plot control inputs
	u_impulses_all = np.array(u_impulses_all)
	n = u_impulses_all.shape[0]
	t = np.arange(n) * dt
	# ax = fig.add_subplot(111, projection='3d')
	fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
	fig.suptitle("Control inputs (impulses) over time")
	# plt.subplot(1, 2, 1)
	axs[0].plot(t, u_impulses_all[:, 0])
	axs[0].set_title("Motor 1")
	axs[1].plot(t, u_impulses_all[:, 1], linewidth=0.5)
	axs[1].set_title("Motor 2")
	axs[2].plot(t, u_impulses_all[:, 2])
	axs[2].set_title("Motor 3")
	axs[3].plot(t, u_impulses_all[:, 3], linewidth=0.5)
	axs[3].set_title("Motor 4")
	fig.savefig("backup_set_outputs/u_q_%f_r_%f.png" % (q, r))
	plt.clf()

	# IPython.embed()

	# Plot 3D trajectory
	from mpl_toolkits.mplot3d import Axes3D

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# x_all = np.array(x_all)  # 1001 x 10
	# # IPython.embed()
	# plt.title("Trajectory of closed loop system, for q=%f, r=%f" % (q, r))
	# ax.plot(x_all[:, 6], x_all[:, 7], x_all[:, 8])
	# interval = 50
	# plt.show()

	# plot_fancy_3D(x_all, "backup_set_outputs/fancy_3d_plot_q_%f_r_%f.png" % (q,r))

	animate_quadcopter(x_all, param_dict, "backup_set_outputs/animation_q_%f_r_%f.gif" % (q,r))









