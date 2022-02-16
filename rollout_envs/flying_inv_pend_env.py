import numpy as np
import math
import IPython
import sys, os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

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
        state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
                         "dtheta", "x", "y", "z", "dx", "dy", "dz"]
        state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
        self.i = state_index_dict
        self.dt = 0.005 # same as cartpole

        self.g = 9.81
        # IPython.embed()

        self.init_visualization()
        
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

        R = self.rotation_matrix(gamma, beta, alpha)

        k_x = R[0, 2]
        k_y = R[1, 2]
        k_z = R[2, 2]

        ###### Computing state derivatives

        ddphi = (3.0)*(k_y*np.cos(phi) + k_z*np.sin(phi))/(2*self.M*self.L_p*np.cos(theta))*(self.M*self.g) + 2*dtheta*dphi*np.tan(theta)
        ddtheta = (3.0*(-k_x*np.cos(theta)-k_y*np.sin(phi)*np.sin(theta) + k_z*np.cos(phi)*np.sin(theta))/(2.0*self.M*self.L_p))*(self.M*self.g) - np.square(dphi)*np.sin(theta)*np.cos(theta)

        # ddx = 0
        # ddy = 0
        # ddz = -self.g

        ddx = k_x*self.g
        ddy = k_y*self.g
        ddz = k_z*self.g - self.g

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

        R = self.rotation_matrix(gamma, beta, alpha)

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

    def init_visualization(self):
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-2.0, 2.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-2.0, 2.0])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-2.0, 2.0])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')

        self.quad = {}
        self.quad['l1'], = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
        self.quad['l2'], = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)
        self.quad['hub'], = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)
        self.quad['pend'], = self.ax.plot([],[],[],color='orange',linewidth=3,antialiased=False)

        self.time_display = self.ax.text2D(0., 0.9, "red" ,color='red', transform=self.ax.transAxes)
        self.state_display = self.ax.text2D(0.6, 0.9, "green" ,color='green', transform=self.ax.transAxes)

        self.time_elapsed = 0

    def step(self, x, u):
        x_dot = self.x_dot_open_loop(x, u)
        nx = x + x_dot * self.dt
        self.time_elapsed += self.dt
        return nx

    def rotation_matrix(self, gamma, beta, alpha):
        #roll-pitch-yaw
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
        return R

    def update_visualization(self, x):
        R = self.rotation_matrix(x[self.i["gamma"]], x[self.i["beta"]], x[self.i["alpha"]])
        L = self.l
        points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0]]).T
        points = np.dot(R,points)
        points[0,:] += x[self.i["x"]]
        points[1,:] += x[self.i["y"]]
        points[2,:] += x[self.i["z"]]
        self.quad['l1'].set_data(points[0,0:2],points[1,0:2])
        self.quad['l1'].set_3d_properties(points[2,0:2])
        self.quad['l2'].set_data(points[0,2:4],points[1,2:4])
        self.quad['l2'].set_3d_properties(points[2,2:4])
        self.quad['hub'].set_data(points[0,4:6],points[1,4:6])
        self.quad['hub'].set_3d_properties(points[2,4:6])

        # print(x[self.i["phi"]])
        # print(x[self.i["theta"]])
        R_pend = self.rotation_matrix(x[self.i["phi"]], x[self.i["theta"]], 0.)
        Lp = self.L_p
        pend_points = np.array([[0,0,0], [0, 0, Lp]]).T
        pend_points = np.dot(R_pend, pend_points)
        pend_points[0,:] += x[self.i["x"]]
        pend_points[1,:] += x[self.i["y"]]
        pend_points[2,:] += x[self.i["z"]]

        self.quad['pend'].set_data(pend_points[0,0:2],pend_points[1,0:2])
        self.quad['pend'].set_3d_properties(pend_points[2,0:2])

        self.time_display.set_text('Simulation time = %.1fs' % (self.time_elapsed))
        self.state_display.set_text('Position of the quad: \n x = %.1fm y = %.1fm z = %.1fm' % (x[self.i["x"]], x[self.i["y"]], x[self.i["z"]]))
    
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

    # x = np.random.rand(10)
    # u = np.random.rand(4)
    # x_dot = default_env.x_dot_open_loop(x, u)
    # print(x, u)
    # print(x_dot)
