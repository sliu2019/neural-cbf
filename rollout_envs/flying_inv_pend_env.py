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
            from src.argument import create_parser
            from main import create_flying_param_dict

            parser = create_parser()  # default
            args = parser.parse_known_args()[0]
            self.param_dict = create_flying_param_dict(args) # default
        else:
            self.param_dict = param_dict

        self.__dict__.update(self.param_dict)  # __dict__ holds and object's attributes
        state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
                         "dtheta", "x", "y", "z", "dx", "dy", "dz"]
        state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
        self.i = state_index_dict
        self.dt = 0.00005 # same as cartpole
        # self.dt = 1e-6
        self.g = 9.81

        # self.init_visualization() # TODO: simplifying out viz, for now

        self.control_lim_verts = self.compute_control_lim_vertices()
        
    def compute_control_lim_vertices(self):
        k1 = self.k1
        k2 = self.k2
        l = self.l
        
        M = np.array([[k1, k1, k1, k1], [0, -l*k1, 0, l*k1], [l*k1, 0, -l*k1, 0], [-k2, k2, -k2, k2]]) # mixer matrix

        self.mixer = M
        self.mixer_inv = np.linalg.inv(self.mixer)
        
        r1 = np.concatenate((np.zeros(8), np.ones(8)))
        r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
        r3 = np.concatenate((np.zeros(2), np.ones(2),np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),np.zeros(2), np.ones(2)))
        r4 = np.zeros(16)
        r4[1::2] = 1.0
        impulse_vert = np.concatenate((r1[None], r2[None], r3[None], r4[None]), axis=0) # 16 vertices in the impulse control space

        # print("ulimitsetvertices")
        # IPython.embed()

        force_vert = M@impulse_vert - np.array([[self.M*self.g], [0.0], [0.0], [0.0]]) # Fixed bug: was subtracting self.M*g (not just in the first row)
        force_vert = force_vert.T.astype("float32")
        return force_vert

    def _f(self, x):

        if len(x.shape) == 1:
            x = x[None] # (1, 16)
        # print("Inside f")
        # IPython.embed()
        bs = x.shape[0]

        gamma = x[:, self.i["gamma"]]
        beta = x[:, self.i["beta"]]
        alpha = x[:, self.i["alpha"]]

        phi = x[:, self.i["phi"]]
        theta = x[:, self.i["theta"]]
        dphi = x[:, self.i["dphi"]]
        dtheta = x[:, self.i["dtheta"]]

        R = np.zeros((bs, 3, 3))
        R[:,0, 0] = np.cos(alpha)*np.cos(beta)
        R[:,0, 1] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
        R[:,0, 2] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
        R[:,1, 0] = np.sin(alpha)*np.cos(beta)
        R[:,1, 1] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
        R[:,1, 2] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
        R[:,2, 0] = -np.sin(beta)
        R[:,2, 1] = np.cos(beta)*np.sin(gamma)
        R[:,2, 2] = np.cos(beta)*np.cos(gamma)

        k_x = R[:,0, 2]
        k_y = R[:,1, 2]
        k_z = R[:,2, 2]

        ###### Computing state derivatives
        ddphi = (3.0) * (k_y * np.cos(phi) + k_z * np.sin(phi)) * (self.M * self.g) / (
                    2 * self.M * self.L_p * np.cos(theta)) + 2 * dtheta * dphi * np.tan(theta)

        ddtheta = (3.0*(-k_x*np.cos(theta)-k_y*np.sin(phi)*np.sin(theta) + k_z*np.cos(phi)*np.sin(theta))*(self.M*self.g)/(2.0*self.M*self.L_p)) - np.square(dphi)*np.sin(theta)*np.cos(theta)

        ddx = k_x*self.g
        ddy = k_y*self.g
        ddz = k_z*self.g - self.g

        # Including translational motion
        f = np.vstack([x[:,self.i["dgamma"]], x[:,self.i["dbeta"]], x[:,self.i["dalpha"]], np.zeros(bs), np.zeros(bs), np.zeros(bs), dphi, dtheta, ddphi, ddtheta, x[:,self.i["dx"]], x[:,self.i["dy"]], x[:,self.i["dz"]], ddx, ddy, ddz]).T
        return f

    def _g(self, x):
        if len(x.shape) == 1:
            x = x[None] # (1, 16)
        # print("g: returns matrix")
        # IPython.embed()
        bs = x.shape[0]

        gamma = x[:,self.i["gamma"]]
        beta = x[:,self.i["beta"]]
        alpha = x[:,self.i["alpha"]]

        phi = x[:,self.i["phi"]]
        theta = x[:,self.i["theta"]]

        R = np.zeros((bs, 3, 3))
        R[:,0, 0] = np.cos(alpha)*np.cos(beta)
        R[:,0, 1] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
        R[:,0, 2] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
        R[:,1, 0] = np.sin(alpha)*np.cos(beta)
        R[:,1, 1] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
        R[:,1, 2] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
        R[:,2, 0] = -np.sin(beta)
        R[:,2, 1] = np.cos(beta)*np.sin(gamma)
        R[:,2, 2] = np.cos(beta)*np.cos(gamma)

        k_x = R[:,0, 2]
        k_y = R[:,1, 2]
        k_z = R[:,2, 2]

        ###### Computing state derivatives
        J_inv = np.diag([(1.0/self.J_x), (1.0/self.J_y), (1.0/self.J_z)])
        dd_drone_angles = R@J_inv

        # print(J_inv, R)

        ddphi = (3.0)*(k_y*np.cos(phi) + k_z*np.sin(phi))/(2*self.M*self.L_p*np.cos(theta))
        ddtheta = (3.0*(-k_x*np.cos(theta)-k_y*np.sin(phi)*np.sin(theta) + k_z*np.cos(phi)*np.sin(theta))/(2.0*self.M*self.L_p))

        # Including translational motion
        g = np.zeros((bs, 16, 4))
        g[:, 3:6, 1:] = dd_drone_angles
        g[:, 8, 0] = ddphi
        g[:, 9, 0] = ddtheta
        g[:, 13:, 0] = (1.0/self.M)*np.array([k_x, k_y, k_z]).T

        # print(g)
        return g

    def x_dot_open_loop(self, x, u):
        # Batched
        f = self._f(x)
        g = self._g(x)

        u_clamped, debug_dict = self.clip_u(u)
        # print("in x_dot_open_loop")
        # IPython.embed()
        rv = f + (g@(u_clamped[:, :, None]))[:, :, 0]
        return rv

    def _smooth_clamp(self, motor_impulses):
        # Batched
        # clamps to 0, 1
        # rv = 1.0/(1.0 + np.exp(-8*motor_impulses+4))
        rv = np.clip(motor_impulses, 0, 1)
        return rv

    def clip_u(self, u):
        # Batched: u is bs x u_dim
        # Assumes u is raw
        # print("in clip_u")
        # IPython.embed()
        u_gravity_comp = u + np.array([self.M*self.g, 0, 0, 0]) # u with gravity compensation
        # motor_impulses = np.linalg.solve(self.mixer, u_gravity_comp) # low-level inputs
        motor_impulses = u_gravity_comp@self.mixer_inv.T
        smooth_clamped_motor_impulses = self._smooth_clamp(motor_impulses)

        smooth_clamped_u_gravity_comp = smooth_clamped_motor_impulses@self.mixer.T
        rv = smooth_clamped_u_gravity_comp - np.array([self.M*self.g, 0, 0, 0])

        return rv, {"motor_impulses": motor_impulses, "smooth_clamped_motor_impulses": smooth_clamped_motor_impulses}

if __name__ == "__main__":
    pass
    """param_dict = {
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
    state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
                         "dtheta"]  # excluded x, y, z
    state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
    param_dict["state_index_dict"] = state_index_dict


    np.random.seed(3)
    default_env = FlyingInvertedPendulumEnv(param_dict)

    x = np.random.rand(16)
    u = np.random.rand(4)
    print(x, u)
    x_dot = default_env.x_dot_open_loop(x, u)

    print(x_dot)"""

    # Testing the batch refactoring
    """env = FlyingInvertedPendulumEnv()

    # x = np.random.random((16))
    np.random.seed(0)
    x_batch = np.random.random((10, 16))
    f_vals = env._f(x_batch)
    g_vals = env._g(x_batch)

    print(f_vals)
    print(g_vals)

    print("done")
    IPython.embed() #"""