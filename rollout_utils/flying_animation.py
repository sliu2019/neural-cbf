import numpy as np
import math
import IPython
import sys, os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

def rotation_matrix(gamma, beta, alpha):
	# roll-pitch-yaw
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
	return R

def create_flying_pend_animation(rollout, param_dict, dt, save_fpth):
	"""
	:param rollout: (N_steps, 16)
	:return:
	"""
	### Init
	fig = plt.figure()
	ax = Axes3D.Axes3D(fig)
	ax.set_xlim3d([-2.0, 2.0])
	ax.set_xlabel('X')
	ax.set_ylim3d([-2.0, 2.0])
	ax.set_ylabel('Y')
	ax.set_zlim3d([-2.0, 2.0])
	ax.set_zlabel('Z')
	ax.set_title('Quadcopter Simulation')

	quad = {}
	quad['l1'], = ax.plot([], [], [], color='blue', linewidth=3, antialiased=False)
	quad['l2'], = ax.plot([], [], [], color='red', linewidth=3, antialiased=False)
	quad['hub'], = ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)
	quad['pend'], = ax.plot([], [], [], color='orange', linewidth=3, antialiased=False)

	time_display = ax.text2D(0., 0.9, "red", color='red', transform=ax.transAxes)
	state_display = ax.text2D(0.6, 0.9, "green", color='green', transform=ax.transAxes)

	time_elapsed = 0

	#########################
	L = param_dict["l"]
	i = param_dict["state_index_dict"]
	Lp = param_dict["L_p"]
	for j in range(rollout.shape[0]):
		x = rollout[j]

		R = rotation_matrix(x[i["gamma"]], x[i["beta"]], x[i["alpha"]])
		# Rotate frame
		points = np.array([[-L, 0, 0], [L, 0, 0], [0, -L, 0], [0, L, 0], [0, 0, 0], [0, 0, 0]]).T
		points = np.dot(R, points)
		points[0, :] += x[i["x"]]
		points[1, :] += x[i["y"]]
		points[2, :] += x[i["z"]]
		quad['l1'].set_data(points[0, 0:2], points[1, 0:2])
		quad['l1'].set_3d_properties(points[2, 0:2])
		quad['l2'].set_data(points[0, 2:4], points[1, 2:4])
		quad['l2'].set_3d_properties(points[2, 2:4])
		quad['hub'].set_data(points[0, 4:6], points[1, 4:6])
		quad['hub'].set_3d_properties(points[2, 4:6])

		# print(x[i["phi"]])
		# print(x[i["theta"]])
		R_pend = rotation_matrix(x[i["phi"]], x[i["theta"]], 0.)
		# Lp = L_p
		pend_points = np.array([[0, 0, 0], [0, 0, Lp]]).T
		pend_points = np.dot(R_pend, pend_points)
		pend_points[0, :] += x[i["x"]]
		pend_points[1, :] += x[i["y"]]
		pend_points[2, :] += x[i["z"]]

		quad['pend'].set_data(pend_points[0, 0:2], pend_points[1, 0:2])
		quad['pend'].set_3d_properties(pend_points[2, 0:2])

		time_elapsed += dt
		time_display.set_text('Simulation time = %.1fs' % (time_elapsed))
		state_display.set_text(
			'Position of the quad: \n x = %.1fm y = %.1fm z = %.1fm' % (x[i["x"]], x[i["y"]], x[i["z"]]))

		plt.pause(0.000000000000001)

if __name__ == "__main__":
    # pass
    env = FlyingInvertedPendulumEnv()
    x = np.zeros(16)

    for i in range(10000):
        u = np.zeros(4)
        u += (np.random.rand(4) - 0.5) * 1e-6
        # u[]

        x = env.step(x, u)
        env.update_visualization(x)
