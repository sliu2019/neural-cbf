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

env = FlyingInvertedPendulumEnv(param_dict)

def simulate_quadrotor(xdot_fn, u_fn, timesteps=1000):
	k1 = param_dict["k_1"]
	k2 = param_dict["k_2"]
	l = param_dict["l"]

	x_all = []
	u_impulses_all = []
	u_forces_all = []
	distances = []

	eps = np.random.normal(scale=1.0, size=10)  # let x0 be approximately in the 0-centered norm ball
	target = np.zeros(10)  # Stabilize to origin
	xcurr = target + eps
	xcurr = np.concatenate((xcurr, np.zeros(6))) # adding [x, y, z] and derivatives to the end

	x_all.append(xcurr)

	M = np.array(
		[[k1, k1, k1, k1], [0, -l * k1, 0, l * k1], [l * k1, 0, -l * k1, 0], [-k2, k2, -k2, k2]])  # Mixer matrix

	for t in range(timesteps):
		ucurr = u_fn(xcurr)  # call feedback controller

		Ui = np.linalg.solve(M,
		                     ucurr)  # Convert from high-level control inputs (forces, torques) to low-level (impulses)
		Ui = np.clip(Ui, 0, 1)  # Enforce control limits
		ucurr = M @ Ui  # Convert back to high-level control inputs

		xdot = xdot_fn(xcurr, ucurr)  # Call dynamics equation
		xnext = xcurr + xdot * dt  # Simulate system via finite differencing

		# Log data
		x_all.append(xnext)
		u_impulses_all.append(Ui)
		u_forces_all.append(ucurr)

		xcurr = xnext


		dist = np.linalg.norm(target - xcurr[0:10])
		distances.append(dist)

		print(dist)

	return x_all, u_impulses_all, u_forces_all, distances


# def xdot_fn(x, u):
# 	# print("xdot")
# 	# IPython.embed()
#
# 	xdot = np.zeros((10))
# 	xdot[6:] = x[:6]
# 	phi = x[-3]
# 	theta = x[-2]
# 	psi = x[-1]
# 	F = u[0]
#
# 	xdot[0] = (-math.cos(phi) * math.sin(theta) * math.cos(psi) - math.sin(phi) * math.sin(psi)) * (F / m)
# 	xdot[1] = (-math.cos(phi) * math.sin(theta) * math.sin(psi) + math.sin(phi) * math.cos(psi)) * (F / m)
# 	xdot[2] = g - (math.cos(phi) * math.cos(theta)) * (F / m)
# 	xdot[3] = (1.0 / Jx) * u[1]
# 	xdot[4] = (1.0 / Jy) * u[2]
# 	xdot[5] = (1.0 / Jz) * u[3]
#
#
#
# 	return xdot

def xdot_fn(x, u):
	xdot = env.x_dot_open_loop(x, u)
	return xdot

if __name__ == "__main__":
	L_p = param_dict["L_p"]
	M = param_dict["m"] + param_dict["m_p"]

	A = np.zeros((10, 10)) # 10 x 10
	A[0:3, 3:6] = np.eye(3)
	A[6:8, 8:10] = np.eye(2)
	A[8, 0] = -3*g/(2*L_p)
	A[9, 1] = 3*g/(2*L_p)

	print(A)

	B = np.zeros((10, 4))
	B[3:6, 1:4] = np.eye(3)
	B[-1, 0] = -3/(2*M*L_p)

	print(B)

	# Check that the system is controllable
	C = control.ctrb(A, B)
	rk = np.linalg.matrix_rank(C)
	assert rk == 10

	# Use LQR to compute feedback portion of controller
	q = 0.5
	r = 1
	Q = q * np.eye(10)
	R = r * np.eye(4)
	K, S, E = control.lqr(A, B, Q, R)

	u_fn = lambda x: np.array([M * g, 0, 0, 0]) - K @ x  # control policy

	# Sanity check the dynamics
	print(xdot_fn(np.zeros(10), np.array([M*g, 0, 0, 0])))

	IPython.embed()

	x_all, u_impulses_all, u_forces_all, distances = simulate_quadrotor(xdot_fn, u_fn, timesteps=2000)

	# Plotting

	# Plot distance from origin over time
	plt.plot(np.arange(len(distances)) * dt, distances, linewidth=0.5)
	plt.title("Distance from origin, for q=%f, r=%f" % (q, r))
	plt.xlabel("Time so far")
	plt.savefig("baselines/backup_set_dist_q_%f_r_%f.png" % (q, r))
	plt.show()
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
	fig.savefig("baselines/backup_set_u_q_%f_r_%f.png" % (q, r))
	plt.clf()

	# IPython.embed()

	# Plot 3D trajectory
	from mpl_toolkits.mplot3d import Axes3D

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x_all = np.array(x_all)  # 1001 x 10
	# IPython.embed()
	plt.title("Trajectory of closed loop system, for q=%f, r=%f" % (q, r))
	ax.plot(x_all[:, 6], x_all[:, 7], x_all[:, 8])
	interval = 50









