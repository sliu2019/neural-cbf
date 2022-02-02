import numpy as np
import math

class CartPoleEnv():
    def __init__(self):
        self.dt = 0.005 # TODO
        self.max_theta = math.pi # state space limit
        self.max_angular_velocity = 5.0 # state space limit

        self.theta_safe_lim = math.pi/4.0
        self.max_force = 22.0

        self.g = 9.81
        self.I = 1.2E-3
        self.m = 0.127
        self.M = 1.0731
        self.l = 0.3365 
        
        max_angular_velocity = 5.0 # state space constraint
        self.x_lim = np.array([[-math.pi, math.pi], [-max_angular_velocity, max_angular_velocity]])

    def x_dot_open_loop(self, x, u):

        g = self.g
        I = self.I
        m = self.m
        M = self.M
        l = self.l

        # u is scalar
        x_dot = np.zeros(4)

        x_dot[0] = x[2]
        x_dot[1] = x[3]

        theta = x[1]
        theta_dot = x[3]
        denom = I*(M + m) + m*(l**2)*(M + m*(math.sin(theta)**2))
        x_dot[2] = (I + m*(l**2))*(m*l*theta_dot**2*math.sin(theta)) - g*(m**2)*(l**2)*math.sin(theta)*math.cos(theta) + (I + m*l**2)*u
        x_dot[3] = m*l*(-m*l*theta_dot**2*math.sin(theta)*math.cos(theta) + (M+m)*g*math.sin(theta)) + (-m*l*math.cos(theta))*u

        x_dot[2] = x_dot[2]/denom
        x_dot[3] = x_dot[3]/denom

        return x_dot
