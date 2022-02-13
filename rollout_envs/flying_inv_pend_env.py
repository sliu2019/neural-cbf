import numpy as np
import math
import IPython
import sys, os

g = 9.81
class FlyingInvertedPendulumEnv():
	def __init__(self, param_dict=None):
		if param_dict is None:
			# Form a default param dict
			sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
			from main import create_flying_param_dict
			self.param_dict = create_flying_param_dict() # default
		else:
			self.param_dict = param_dict
		self.__dict__.update(self.param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict
		self.dt = 0.005 # same as cartpole

		# IPython.embed()

	def x_dot_open_loop(self, x, u):
		# IPython.embed()

		gamma = x[self.i["gamma"]]
		beta = x[self.i["beta"]]
		alpha = x[self.i["alpha"]]

		phi = x[self.i["phi"]]
		theta = x[self.i["theta"]]
		dphi = x[self.i["dphi"]]
		dtheta = x[self.i["dtheta"]]

		R = np.zeros((3, 3))
		R[0, 0] = np.cos(alpha)*np.cos(beta)
		R[0, 1] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
		R[0, 2] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
		R[1, 0] = np.sin(alpha)*np.cos(beta)
		R[1, 1] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
		R[1, 2] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
		R[2, 0] = -np.sin(beta)
		R[2, 1] = np.cos(beta)*np.sin(gamma)
		R[2, 2] = np.cos(beta)*np.cos(gamma)

		k_x = R[0, 2]
		k_y = R[1, 2]
		k_z = R[2, 2]

		F = (u[0] + self.M*g)

		###### Computing state derivatives
		ddquad_angles = R@u[1:]
		ddgamma = (1.0/self.J_x)*ddquad_angles[0]
		ddbeta = (1.0/self.J_y)*ddquad_angles[1]
		ddalpha = (1.0/self.J_z)*ddquad_angles[2]

		ddphi = (3.0)*(k_y*np.cos(phi) + k_z*np.sin(phi))/(2*self.M*self.L_p*np.cos(theta))*F + 2*dtheta*dphi*np.tan(theta)
		ddtheta = (3.0*(-k_x*np.cos(theta)-k_y*np.sin(phi)*np.sin(theta) + k_z*np.cos(phi)*np.sin(theta))/(2.0*self.M*self.L_p))*F - np.square(dphi)*np.sin(theta)*np.cos(theta)

		# Excluding translational motion
		rv = np.array([x[self.i["dgamma"]], x[self.i["dbeta"]], x[self.i["dalpha"]], ddgamma, ddbeta, ddalpha, dphi, dtheta, ddphi, ddtheta])
		return rv

if __name__ == "__main__":
	pass
	# default_env = FlyingInvertedPendulumEnv()
	# x = np.zeros(10)
	# u = np.zeros(4)
	# u[0] = default_env.M + g
	#
	# x_dot = default_env.x_dot_open_loop(x, u)

	# x = np.random.rand(10)
	# u = np.random.rand(4)
	# x_dot = default_env.x_dot_open_loop(x, u)
	# print(x, u)
	# print(x_dot)