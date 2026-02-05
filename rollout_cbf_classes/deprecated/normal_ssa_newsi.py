import numpy as np
import math
from cvxopt import matrix, solvers

class SSA:
    def __init__(self, env):
        # self.c1 = 1.4
        self.c1 = 1
        self.c2 = 0.1
        # self.c2 = 1
        self.c3 = 0
        self.theta_max = math.pi / 4.0
        self.alpha1 = 1
        self.env = env
        print("New safety index")

    def set_params(self, params):
        self.c1 = params[0]
        self.c2 = params[1]
        self.c3 = params[2]

    def phi_0(self, x):
        ''' user defined safety index
        Input: x[1] is theta, x[3] = theta dot
        '''
        theta = x[1]
        phi_0 = abs(theta) - self.theta_max
        return phi_0

    def phi_1(self, x):
        ''' user defined safety index
        Input: x[1] is theta, x[3] = theta dot
        '''
        theta = x[1]
        dtheta = x[3]
        phi_1 = abs(theta) - self.theta_max + self.c2 * np.sign(theta) * dtheta
        return phi_1

    def phi_fn(self, x):
        """ safety index
        Input:
            x: state, x[1] is theta, x[3] = theta dot
        """
        # one dimensional implementation
        # theta = x[1]
        # dtheta = x[3]
        # doth = 2 * theta * dtheta
        # phi_0 = theta ** 2 - self.theta_max ** 2
        # phi_1 = theta ** 2 - self.theta_max ** 2 + 2 * self.c2 * theta * dtheta
        # beta = theta ** (2 * self.c1) - theta ** (2 * self.c1) + self.c3
        # phi_star = phi_1 - phi_0 + beta

        # batch implementation
        try:
            assert len(x.shape) == 2 and x.shape[1] == 4
        except:
            # print(f'the shape of x is {x.shape}')
            # print(f'the x is {x}')
            assert len(x.shape) == 1 and x.shape[0] == 4
            # reshape x into 1,4
            x = np.expand_dims(x, axis=0)

        theta = x[:, 1].reshape((x.shape[0], 1))
        dtheta = x[:, 3].reshape((x.shape[0], 1))
        # doth = 2 * theta * dtheta
        # doth = 2 * np.multiply(theta, dtheta)
        # phi_0 = np.power(theta, 2) - np.power(self.theta_max, 2)
        # phi_1 = np.power(theta, 2) - np.power(self.theta_max, 2) + 2 * self.c2 * np.multiply(theta, dtheta)
        # beta = np.power(theta, 2 * self.c1) - np.power(self.theta_max, 2 * self.c1) + self.c3
        phi_0  = np.absolute(theta) - self.theta_max
        # sigmoid = 2*(1/(1 + np.exp(-theta*100))) - 1
        phi_1 = np.absolute(theta) - self.theta_max + self.c2 * np.multiply(np.sign(theta), dtheta)
        # phi_1 = np.absolute(theta) - self.theta_max + self.c2 * np.multiply(sigmoid, dtheta)
        beta = np.power(np.absolute(theta), self.c1) - np.power(self.theta_max, self.c1) + self.c3
        phi_star = phi_1 - phi_0 + beta
        result = np.hstack((phi_0, phi_1, phi_star))

        return result

    def phi_grad(self, x):
        """ safety index gradient with respect to x
        Input:
            x: state, x[1] is theta, x[3] = theta dot
            u: control
        """
        # batch implementation
        x = np.array(x)
        try:
            assert len(x.shape) == 2 and x.shape[1] == 4
        except:
            # print(f'the shape of x is {x.shape}')
            # print(f'the x is {x}')
            assert len(x.shape) == 1 and x.shape[0] == 4
            # reshape x into 1,4
            x = np.expand_dims(x, axis=0)

        theta = x[:, 1].reshape((x.shape[0], 1))
        dtheta = x[:, 3].reshape((x.shape[0], 1))

        sigmoid = 2 * (1 / (1 + np.exp(-theta*100))) - 1
        # grad_gigmoid = np.multiply(sigmoid,1 - sigmoid)

        grad_theta = np.multiply(self.c1 * np.power(np.absolute(theta), self.c1 - 1), np.sign(theta))
        # grad_theta = np.multiply((self.c1 * np.power(np.absolute(theta), self.c1 - 1)), sigmoid) + np.multiply(self.c2 * dtheta, grad_gigmoid)
        grad_dtheta = self.c2 * np.sign(theta)
        # grad_dtheta = self.c2 * sigmoid

        grad = np.hstack((np.zeros((x.shape[0], 1)), grad_theta, np.zeros((x.shape[0], 1)), grad_dtheta))
        return grad

    def safe_control(self, uref, x):
        ''' safe control
        Input:
            uref: reference control
            x: state
        '''
        # if the next state is unsafe, then trigger the safety control
        # TODO:

        # solve QP
        # Compute the control constraints
        # Get f(x), g(x); note it's a hack for scalar u
        f_x = self.env.x_dot_open_loop(x, 0)
        g_x = self.env.x_dot_open_loop(x, 1) - f_x

        # A*u <= b
        A = self.phi_grad(x).T * g_x
        b = -self.phi_grad(x).T * f_x - self.alpha1

        # compute the QP
        # objective: 0.5*u*P*u + q*u
        w = 1000.0
        P = 2 * np.array([[1.0, 0], [0, 0]])
        q = np.array([[-2.0 * uref], [w]])

        # constraint
        # A u <= b
        # u <= umax
        # -u <= umax
        # c >= 0, c is the slack variable
        G = np.array([[A, -1.0], [1, 0], [-1, 0], [0, -1]])
        rho = np.array([[b], [self.env.max_force], [self.env.max_force], [0.0]])
        sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(rho))
        sol_var = sol_obj['x']

        return sol_var


if __name__ == "__main__":
    ssa = SSA('ssa')
    x = np.array([0, 1, 0, 1])
    y = np.vstack((x, x, x))
    # import ipdb; ipdb.set_trace()
    # result = ssa.phi_fn(y)
    result = ssa.phi_grad(y)
    print(result)