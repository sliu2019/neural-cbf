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
		self.__dict__.update(self.param_dict)

		# Form here; do not use the state_index_dict in self.param_dict, which doesn't consider translational motion
		state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
	                     "dtheta", "x", "y", "z", "dx", "dy", "dz"]
		state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
		self.i = state_index_dict
		self.dt = 0.005 # same as cartpole

	def _f(self, x):
		# print("Inside f")
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

		###### Computing state derivatives

		ddphi = (3.0)*(k_y*np.cos(phi) + k_z*np.sin(phi))/(2*self.M*self.L_p*np.cos(theta))*(self.M*g) + 2*dtheta*dphi*np.tan(theta)
		ddtheta = (3.0*(-k_x*np.cos(theta)-k_y*np.sin(phi)*np.sin(theta) + k_z*np.cos(phi)*np.sin(theta))/(2.0*self.M*self.L_p))*(self.M*g) - np.square(dphi)*np.sin(theta)*np.cos(theta)

		ddx = k_x*g
		ddy = k_y*g
		ddz = k_z*g - g

		# Including translational motion
		f = np.array([x[self.i["dgamma"]], x[self.i["dbeta"]], x[self.i["dalpha"]], 0, 0, 0, dphi, dtheta, ddphi, ddtheta, x[self.i["dx"]], x[self.i["dy"]], x[self.i["dz"]], ddx, ddy, ddz])
		return f

	def _g(self, x):
		# print("g: returns matrix")
		# IPython.embed()

		gamma = x[self.i["gamma"]]
		beta = x[self.i["beta"]]
		alpha = x[self.i["alpha"]]

		phi = x[self.i["phi"]]
		theta = x[self.i["theta"]]

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

		###### Computing state derivatives
		dd_drone_angles = np.diag([(1.0/self.J_x), (1.0/self.J_y), (1.0/self.J_z)])@R

		ddphi = (3.0)*(k_y*np.cos(phi) + k_z*np.sin(phi))/(2*self.M*self.L_p*np.cos(theta))
		ddtheta = (3.0*(-k_x*np.cos(theta)-k_y*np.sin(phi)*np.sin(theta) + k_z*np.cos(phi)*np.sin(theta))/(2.0*self.M*self.L_p))

		# Including translational motion
		g = np.zeros((16, 4))
		g[3:6, 1:] = dd_drone_angles
		g[8, 0] = ddphi
		g[9, 0] = ddtheta
		g[13:, 0] = (1.0/self.M)*np.array([k_x, k_y, k_z])

		return g

	def x_dot_open_loop(self, x, u):
		# print("inside x_dot_open_loop")
		# IPython.embed()

		f = self._f(x)
		g = self._g(x)

		rv = f + g@u
		return rv

if __name__ == "__main__":
	pass
	# default_env = FlyingInvertedPendulumEnv()
	#
	# x = np.random.rand(16)
	# u = np.random.rand(4)
	# x_dot = default_env.x_dot_open_loop(x, u)
	#
	# print(x_dot)

