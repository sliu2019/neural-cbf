import numpy as np
import math
from cvxopt import matrix, solvers

class SSA:
    def __init__(self, env):
        self.c1 = 1
        self.c2 = 1 
        self.c3 = 0
        self.x_max = math.pi/4.0
        self.alpha1 = 1
        self.env = env


    def phi(self, x):
        ''' safety index
        Input: x[0] is theta, x[1] = theta dot
        '''
        doth = 2*x[0]*x[1]
        phi_eval = x[0]**(2*self.c1) - x[0]**(2*self.c1) - self.c2*self.doth + self.c3
        return phi_eval 

    def phi_grad(self, x):
        ''' safety index gradient with respect to x 
        Input: 
            x: state 
            u: control 
        '''

        grad_theta = 2*self.c1*x[0]**(2*self.c1 - 1) - 2*self.c2*x[1]
        grad_dtheta = -2*self.c2*x[0]
        grad = np.array([[grad_theta],[grad_theta]])
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
        b = self.phi_grad(x).T*f_x 

        # compute the QP 
        # objective: 0.5*u*P*u + q*u
        P = 2 
        q = -2*uref 

        # constraint 
        # A u <= b
        # u <= umax
        # -u <= umax
        G = np.array([[A], [1], [-1]])
        h = np.array([[self.alpha1], [self.env.max_force], [self.env.max_force]])
        sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        sol_var = sol_obj['x']

        return sol_var

        # Computing control using QP
        # Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
        # w = 1000.0 # TODO: slack weight

        # qp_lhs = lhs.item()
        # qp_rhs = rhs.item()
        # Q = 2*np.array([[1.0, 0], [0, 0]])
        # p = np.array([[-2.0*u_ref], [w]])
        # G = np.array([[lhs, -1.0], [1, 0], [-1, 0], [0, -1]])
        # h = np.array([[rhs], [env.max_force], [env.max_force], [0.0]])

        