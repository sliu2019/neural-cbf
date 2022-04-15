import numpy as np
import math
from cvxopt import matrix, solvers

class SSA:
    def __init__(self, env):
        self.c1 = 1
        self.c2 = 1 
        self.c3 = 0
        self.theta_max = math.pi/4.0
        self.alpha1 = 1
        self.env = env

    def set_params(self, params):
        self.c1 = params[0]
        self.c2 = params[1]
        self.c3 = params[2]
        
    def phi_0(self, x):
        ''' user defined safety index
        Input: x[1] is theta, x[3] = theta dot
        '''
        theta = x[1]
        phi_0 = theta**2 - self.theta_max**2
        return phi_0

    def phi_1(self, x):
        ''' user defined safety index
        Input: x[1] is theta, x[3] = theta dot
        '''
        theta = x[1]
        dtheta = x[3]
        phi_1 = theta**2 - self.theta_max**2 + 2*self.c2*theta*dtheta
        return phi_1


    def phi_fn(self, x):
        ''' safety index
        Input: 
            x: state, x[1] is theta, x[3] = theta dot
        '''
        theta = x[1]
        dtheta = x[3]
        doth = 2*theta*dtheta
        # phi_eval = theta**(2*self.c1) - theta**(2*self.c1) - self.c2*doth + self.c3

        phi_0 = theta**2 - self.theta_max**2
        phi_1 = theta**2 - self.theta_max**2 + 2*self.c2*theta*dtheta
        beta = theta**(2*self.c1) - theta**(2*self.c1) + self.c3
        phi_star = phi_1 - phi_0 + beta
        return np.array([phi_0, phi_1, phi_star])

    def phi_grad(self, x):
        ''' safety index gradient with respect to x 
        Input: 
            x: state, x[1] is theta, x[3] = theta dot
            u: control 
        '''
        # theta = x[:,1]
        # dtheta = x[:,3]
        theta = x[1]
        dtheta = x[3]
        grad_theta = 2*self.c1*theta**(2*self.c1 - 1) - 2*self.c2*dtheta
        grad_dtheta = -2*self.c2*theta
        grad = np.array([[0],[grad_theta],[0],[grad_theta]])
        return grad 
          

    def safe_control(self, uref, x):
        ''' safe control
        Input:
            uref: reference control 
            x: state 
        '''
        # if the next state is unsafe, then trigger the safety control 
        ###### TODO ######## 

        # solve QP 
        # Compute the control constraints
        # Get f(x), g(x); note it's a hack for scalar u
        f_x = self.env.x_dot_open_loop(x, 0)
        g_x = self.env.x_dot_open_loop(x, 1) - f_x

        # A*u <= b
        A = self.phi_grad(x).T*g_x
        b = -self.phi_grad(x).T*f_x  - self.alpha1

        # compute the QP 
        # objective: 0.5*u*P*u + q*u
        w = 1000.0
        P = 2*np.array([[1.0, 0], [0, 0]])
        q = np.array([[-2.0*uref], [w]])

        # constraint 
        # A u <= b
        # u <= umax
        # -u <= umax
        # c >= 0, c is the slack variable 
        G = np.array([[A, -1.0], [1, 0], [-1, 0], [0, -1]])
        h = np.array([[b], [self.env.max_force], [self.env.max_force], [0.0]])
        sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        sol_var = sol_obj['x']

        return sol_var


