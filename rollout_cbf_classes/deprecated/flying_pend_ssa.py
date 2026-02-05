import numpy as np
import math
from cvxopt import matrix, solvers
from scipy.misc import derivative
import torch
class FlyingPendSSA:
    def __init__(self, env):
        self.w1 = 0
        self.w2 = 1
        self.w3 = 1
        
        self.c1 = 2
        self.c2 = 1
        self.c3 = 0
        self.delta_max = math.pi / 4
        self.alpha1 = 1
        self.env = env
        print("New safety index")

    def set_params(self, params):
        self.c0 = params[0] 
        self.c1 = params[1]
        self.c2 = params[2]
        self.c3 = params[3]

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

    def rho(self,x):
        try:
            assert len(x.shape) == 2 and x.shape[1] >= 10
        except:
            # print(f'the shape of x is {x.shape}')
            # print(f'the x is {x}')
            assert len(x.shape) == 1 and x.shape[0] >= 10
            # reshape x into 1,4
            x = np.expand_dims(x, axis=0)

        gamma = x[:,self.env.i["gamma"]]
        beta = x[:,self.env.i["beta"]]
        alpha = x[:,self.env.i["alpha"]]
        dgamma = x[:,self.env.i["dgamma"]]
        dbeta = x[:,self.env.i["dbeta"]]
        dalpha = x[:,self.env.i["dalpha"]]
        psi = x[:,self.env.i["phi"]]
        theta = x[:,self.env.i["theta"]]
        dpsi = x[:,self.env.i["dphi"]]
        dtheta = x[:,self.env.i["dtheta"]]
        
        # delta = np.power(psi, 2) + np.power(theta, 2)
        delta = np.arccos(np.cos(psi) * np.cos(theta))
        
        phi_0 = np.power(delta, 2) - np.power(self.delta_max, 2) + self.w1 * np.power(alpha, 2) + self.w2 * np.power(beta, 2) + self.w3 * np.power(gamma, 2)
        return phi_0
    
    def phi_fn(self, x, vis=False):
        """ safety index
        Input:
            x: state, x[1] is theta, x[3] = theta dot
        """
        
        # batch implementation
        try:
            assert len(x.shape) == 2 and x.shape[1] >= 10
        except:
            # print(f'the shape of x is {x.shape}')
            # print(f'the x is {x}')
            assert len(x.shape) == 1 and x.shape[0] >= 10
            # reshape x into 1,4
            x = np.expand_dims(x, axis=0)

        gamma = x[:,self.env.i["gamma"]]
        beta = x[:,self.env.i["beta"]]
        alpha = x[:,self.env.i["alpha"]]
        dgamma = x[:,self.env.i["dgamma"]]
        dbeta = x[:,self.env.i["dbeta"]]
        dalpha = x[:,self.env.i["dalpha"]]
        psi = x[:,self.env.i["phi"]]
        theta = x[:,self.env.i["theta"]]
        dpsi = x[:,self.env.i["dphi"]]
        dtheta = x[:,self.env.i["dtheta"]]
        
        # delta = np.power(psi, 2) + np.power(theta, 2)
        delta = np.arccos(np.cos(psi) * np.cos(theta))
        d_delta_d_theta = np.sin(theta) * np.cos(psi) / np.sqrt(1 - np.power(np.cos(theta), 2) * np.power(np.cos(psi), 2))
        d_delta_d_psi = np.sin(psi) * np.cos(theta) / np.sqrt(1 - np.power(np.cos(psi), 2) * np.power(np.cos(theta), 2))
        
        phi_0 = np.power(delta, 2) - np.power(self.delta_max, 2) + self.w1 * np.power(alpha, 2) + self.w2 * np.power(beta, 2) + self.w3 * np.power(gamma, 2)
        dot_phi_0 = 2 * delta * d_delta_d_psi * dpsi + 2 * delta * d_delta_d_theta * dtheta + + 2 * self.w1 * alpha * dalpha + 2 * self.w2 * beta * dbeta + 2 * self.w3 * gamma * dgamma
        phi_1 = phi_0 + self.c2 * dot_phi_0
        # phi_1 = dot_phi_0
        
        phi_star = np.power(phi_0 + np.power(self.delta_max, 2), self.c1) - np.power(self.delta_max, 2*self.c1) + self.c2 * dot_phi_0 + self.c3
        
        result = np.vstack((phi_0, phi_1, phi_star)).T
        # if vis:
            # print(d_delta_d_psi, " ",  d_delta_d_theta)
            # print(2 * delta * d_delta_d_psi * dpsi, " ",  2 * delta * d_delta_d_theta * dtheta)
            # print("psi")
            # print(psi)
            # print("theta")
            # print(theta)
            # print("delta")
            # print(delta)
            
            # print("c1")
            # print(self.c1)
            # print("c2")
            # print(self.c2)
            # print("c3")
            # print(self.c3)
            
            # print("phi_0")
            # print(phi_0)
            
            # print("np.power(phi_0 + np.power(self.delta_max, 2), self.c1)")
            # print(np.power(phi_0 + np.power(self.delta_max, 2), self.c1), " ", self.c2 * dot_phi_0)
            
            # print("d_delta_d_psi")
            # print(d_delta_d_psi)
            
            # print("d_delta_d_theta")
            # print(d_delta_d_theta)
            
            # print("dot_phi_0")
            # print(dot_phi_0)
            
            # print("beta")
            # print(beta)
            # print("np.power(self.delta_max, 2)")
            # print(np.power(self.delta_max, 2))
            # print("np.power(self.delta_max, 2*self.c1)")
            # print(np.power(self.delta_max, 2*self.c1))
            # print("self.c2 * dot_phi_0")
            # print("self.w3 * np.power(gamma, 2)")
            # print(self.w3 * np.power(gamma, 2))
            
            # print(phi_star)
            
        return result

    def torch_phi_grad(self, x):
        try:
            assert len(x.shape) == 2 and x.shape[1] >= 10
        except:
            # print(f'the shape of x is {x.shape}')
            # print(f'the x is {x}')
            assert len(x.shape) == 1 and x.shape[0] >= 10
            # reshape x into 1,4
            x = np.expand_dims(x, axis=0)

        gamma = torch.tensor(x[:,self.env.i["gamma"]], requires_grad=True)
        beta = torch.tensor(x[:,self.env.i["beta"]], requires_grad=True)
        alpha = torch.tensor(x[:,self.env.i["alpha"]], requires_grad=True)
        dgamma = torch.tensor(x[:,self.env.i["dgamma"]], requires_grad=True)
        dbeta = torch.tensor(x[:,self.env.i["dbeta"]], requires_grad=True)
        dalpha = torch.tensor(x[:,self.env.i["dalpha"]], requires_grad=True)
        psi = torch.tensor(x[:,self.env.i["phi"]], requires_grad=True)
        theta = torch.tensor(x[:,self.env.i["theta"]], requires_grad=True)
        dpsi = torch.tensor(x[:,self.env.i["dphi"]], requires_grad=True)
        dtheta = torch.tensor(x[:,self.env.i["dtheta"]], requires_grad=True)
        
        # delta = np.power(psi, 2) + np.power(theta, 2)
        delta = torch.arccos(torch.cos(psi) * torch.cos(theta))
        d_delta_d_theta = torch.sin(theta) * torch.cos(psi) / torch.sqrt(1 - torch.pow(torch.cos(theta), 2) * torch.pow(torch.cos(psi), 2))
        d_delta_d_psi = torch.sin(psi) * torch.cos(theta) / torch.sqrt(1 - torch.pow(torch.cos(psi), 2) * torch.pow(torch.cos(theta), 2))
        
        phi_0 = torch.pow(delta, 2) - np.power(self.delta_max, 2) + self.w1 * torch.pow(alpha, 2) + self.w2 * torch.pow(beta, 2) + self.w3 * torch.pow(gamma, 2)
        dot_phi_0 = 2 * delta * d_delta_d_psi * dpsi + 2 * delta * d_delta_d_theta * dtheta + + 2 * self.w1 * alpha * dalpha + 2 * self.w2 * beta * dbeta + 2 * self.w3 * gamma * dgamma
        phi_1 = phi_0 + self.c2 * dot_phi_0
        
        phi = torch.pow(phi_0 + np.power(self.delta_max, 2), self.c1) - np.power(self.delta_max, 2*self.c1) + self.c2 * dot_phi_0 + self.c3
        
        external_grad = torch.tensor([1.])
        phi.backward(gradient=external_grad)
        grad_phi = np.zeros(len(self.env.i))
        grad_phi[self.env.i["phi"]]     = psi.grad.item()
        grad_phi[self.env.i["theta"]]   = theta.grad.item()
        grad_phi[self.env.i["dphi"]]    = dpsi.grad.item()
        grad_phi[self.env.i["dtheta"]]  = dtheta.grad.item()
        grad_phi[self.env.i["alpha"]]   = alpha.grad.item()
        grad_phi[self.env.i["beta"]]    = beta.grad.item()
        grad_phi[self.env.i["gamma"]]   = gamma.grad.item()
        grad_phi[self.env.i["dalpha"]]  = dalpha.grad.item()
        grad_phi[self.env.i["dbeta"]]   = dbeta.grad.item()
        grad_phi[self.env.i["dgamma"]]  = dgamma.grad.item()


        # u_rand = np.random.rand(4)
        # x = x[0,:]
        # # print(x)
        # x_dot = self.env.x_dot_open_loop(x,u_rand)
        # nx = x + x_dot * self.env.dt
        # phi_now = self.phi_fn(x)[0,-1]
        # phi_next = self.phi_fn(nx)[0,-1]
        # dot_phi = grad_phi @ x_dot
        # print("cmaes: ", dot_phi, " ", (phi_next - phi_now) / self.env.dt)

        return grad_phi

    def phi_grad(self, x):
        return self.torch_phi_grad(x)
        """ safety index gradient with respect to x
        Input:
            x: state, x[1] is theta, x[3] = theta dot
            u: control
        """
        # batch implementation
        x = np.array(x)
        try:
            assert len(x.shape) == 2 and x.shape[1] >= 10
        except:
            # print(f'the shape of x is {x.shape}')
            # print(f'the x is {x}')
            assert len(x.shape) == 1 and x.shape[0] >= 10
            # reshape x into 1,4
            x = np.expand_dims(x, axis=0)

        gamma = x[:,self.env.i["gamma"]]
        beta = x[:,self.env.i["beta"]]
        alpha = x[:,self.env.i["alpha"]]
        dgamma = x[:,self.env.i["dgamma"]]
        dbeta = x[:,self.env.i["dbeta"]]
        dalpha = x[:,self.env.i["dalpha"]]
        psi = x[:,self.env.i["phi"]]
        theta = x[:,self.env.i["theta"]]
        dpsi = x[:,self.env.i["dphi"]]
        dtheta = x[:,self.env.i["dtheta"]]
        
        delta = np.arccos(np.cos(psi) * np.cos(theta))
        d_delta_d_theta = np.sin(theta) * np.cos(psi) / np.sqrt(1 - np.power(np.cos(theta), 2) * np.power(np.cos(psi), 2))
        d_delta_d_psi = np.sin(psi) * np.cos(theta) / np.sqrt(1 - np.power(np.cos(psi), 2) * np.power(np.cos(theta), 2))
        
        def compute_phi(delta_max, psi, theta, dpsi, dtheta, alpha, beta, gamma, dalpha, dbeta, dgamma):
            delta = np.arccos(np.cos(psi) * np.cos(theta))
            d_delta_d_theta = np.sin(theta) * np.cos(psi) / np.sqrt(1 - np.power(np.cos(theta), 2) * np.power(np.cos(psi), 2))
            d_delta_d_psi = np.sin(psi) * np.cos(theta) / np.sqrt(1 - np.power(np.cos(psi), 2) * np.power(np.cos(theta), 2))
            
            phi_0 = np.power(delta, 2) - np.power(delta_max, 2) + self.w1 * np.power(alpha, 2) + self.w2 * np.power(beta, 2) + self.w3 * np.power(gamma, 2)
            dot_phi_0 = 2 * delta * d_delta_d_psi * dpsi + 2 * delta * d_delta_d_theta * dtheta + + 2 * self.w1 * alpha * dalpha + 2 * self.w2 * beta * dbeta + 2 * self.w3 * gamma * dgamma
            phi = np.power(phi_0 + np.power(delta_max, 2), self.c1) - np.power(delta_max, 2*self.c1) + self.c2 * dot_phi_0 + self.c3
            return phi

        def compute_phi_1_part(psi, theta, dpsi, dtheta):
            delta = np.arccos(np.cos(psi) * np.cos(theta))
            d_delta_d_theta = np.sin(theta) * np.cos(psi) / np.sqrt(1 - np.power(np.cos(theta), 2) * np.power(np.cos(psi), 2))
            d_delta_d_psi = np.sin(psi) * np.cos(theta) / np.sqrt(1 - np.power(np.cos(psi), 2) * np.power(np.cos(theta), 2))
            return 2 * delta * d_delta_d_psi * dpsi + 2 * delta * d_delta_d_theta * dtheta
        
        def partial_derivative(func, var=0, point=[]):
            args = point[:]
            def wraps(x):
                args[var] = x
                return func(*args)
            return derivative(wraps, point[var], dx = 1e-8)
        
        values = [self.delta_max, psi, theta, dpsi, dtheta, alpha, beta, gamma, dalpha, dbeta, dgamma]

        grad_phi = np.zeros(len(self.env.i))
        grad_phi[self.env.i["phi"]]     = partial_derivative(compute_phi, 1, values)
        grad_phi[self.env.i["theta"]]   = partial_derivative(compute_phi, 2, values)
        grad_phi[self.env.i["dphi"]]    = partial_derivative(compute_phi, 3, values)
        grad_phi[self.env.i["dtheta"]]  = partial_derivative(compute_phi, 4, values)
        grad_phi[self.env.i["alpha"]]   = partial_derivative(compute_phi, 5, values)
        grad_phi[self.env.i["beta"]]    = partial_derivative(compute_phi, 6, values)
        grad_phi[self.env.i["gamma"]]   = partial_derivative(compute_phi, 7, values)
        grad_phi[self.env.i["dalpha"]]  = partial_derivative(compute_phi, 8, values)
        grad_phi[self.env.i["dbeta"]]   = partial_derivative(compute_phi, 9, values)
        grad_phi[self.env.i["dgamma"]]  = partial_derivative(compute_phi, 10, values)

        # d_phi_1_d_psi   = partial_derivative(compute_phi_1_part, 0, [psi, theta, dpsi, dtheta])
        # d_phi_1_d_theta = partial_derivative(compute_phi_1_part, 1, [psi, theta, dpsi, dtheta])
        # d_phi_1_d_dpsi  = partial_derivative(compute_phi_1_part, 2, [psi, theta, dpsi, dtheta])
        # d_phi_1_d_dtheta= partial_derivative(compute_phi_1_part, 3, [psi, theta, dpsi, dtheta])
        # phi_0 = np.power(delta, 2) - np.power(self.delta_max, 2) + self.w1 * np.power(alpha, 2) + self.w2 * np.power(beta, 2) + self.w3 * np.power(gamma, 2)
        # d_phi_d_phi_0 = self.c1 * np.power(phi_0 + np.power(self.delta_max, 2), self.c1-1)
        # phi_grad = np.zeros(len(self.env.i))
        # phi_grad[self.env.i["gamma"]] = 2 * d_phi_d_phi_0 * self.w3 * gamma + 2 * self.w3 * dgamma * self.c2
        # phi_grad[self.env.i["beta"]] = 2 * d_phi_d_phi_0 *self.w2 * beta + 2 * self.w2 * dbeta * self.c2
        # phi_grad[self.env.i["alpha"]] = 2 * d_phi_d_phi_0 *self.w1 * alpha + 2 * self.w1 * dalpha * self.c2
        # phi_grad[self.env.i["dgamma"]] = 2 * self.w3 * gamma * self.c2
        # phi_grad[self.env.i["dbeta"]] = 2 * self.w2 * beta * self.c2
        # phi_grad[self.env.i["dalpha"]] = 2 * self.w1 * alpha * self.c2
        # phi_grad[self.env.i["phi"]] = d_phi_d_phi_0 * 2 * delta * d_delta_d_psi + self.c2 * d_phi_1_d_psi
        # phi_grad[self.env.i["theta"]] = d_phi_d_phi_0 * 2 * delta * d_delta_d_theta + self.c2 * d_phi_1_d_theta
        # phi_grad[self.env.i["dphi"]] = self.c2 * d_phi_1_d_dpsi
        # phi_grad[self.env.i["dtheta"]] = self.c2 * d_phi_1_d_dtheta
        
        # if np.any(np.absolute(grad_phi - phi_grad) > 1e-7):
        #     print(grad_phi - phi_grad)
        #     input()

        return grad_phi

    def safe_control(self, uref, x):
        ''' safe control
        Input:
            uref: reference control
            x: state
        '''
        # if the next state is unsafe, then trigger the safety control
        
        # solve QP
        # Compute the control constraints
        # Get f(x), g(x); note it's a hack for scalar u
        
        # A*u <= b
        A = self.phi_grad(x).T * self.env._f(x)
        b = -self.phi_grad(x).T * self.env._g(x) - self.c0

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

    def compute_u_ref(self, t, x):
        u_ref = np.zeros(4)
        return u_ref
        
    def compute_control(self, t, x):
        self.eps_bdry = 1
        self.eps_outside = 5

        # print("in CBFcontroller, compute control")
        # print(x)
        # IPython.embed()
        ############ Init log vars
        apply_u_safe = None
        u_ref = self.compute_u_ref(t, x)
        phi_vals = None
        qp_slack = None
        qp_lhs = None
        qp_rhs = None
        impulses = None
        inside_boundary = False
        on_boundary = False
        outside_boundary = False
        ################

        phi_vals = self.phi_fn(x)  # This is an array of (1, r+1), where r is the degree
        phi_grad = self.phi_grad(x)

        # print(x.shape)
        x_next = x + self.env.dt * self.env.x_dot_open_loop(x, self.compute_u_ref(t,
                                                                                  x))  # in the absence of safe control, the next state
        next_phi_val = self.phi_fn(x_next)

        if phi_vals[0, -1] > 0:  # Outside
            # print("STATUS: Outside") # TODO
            eps = self.eps_outside
            apply_u_safe = True
            outside_boundary = True
        elif phi_vals[0, -1] < 0 and next_phi_val[0, -1] >= 0:  # On boundary. Note: cheating way to convert DT to CT
            # print("STATUS: On") # TODO
            eps = self.eps_bdry
            apply_u_safe = True
            on_boundary = True
        else:  # Inside
            # print("STATUS: Inside") # TODO
            apply_u_safe = False
            inside_boundary = True
            debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "qp_slack": qp_slack, "qp_rhs": qp_rhs,
                          "qp_lhs": qp_lhs, "phi_vals": phi_vals.flatten(), "impulses": impulses,
                          "inside_boundary": inside_boundary, "on_boundary": on_boundary, "outside_boundary": outside_boundary}
            return u_ref, debug_dict

        # IPython.embed()
        # Compute the control constraints
        f_x = self.env._f(x)
        f_x = np.reshape(f_x, (16, 1))
        g_x = self.env._g(x)

        phi_grad = np.reshape(phi_grad, (16, 1))
        lhs = phi_grad.T @ g_x  # 1 x 4
        rhs = -phi_grad.T @ f_x - eps
        rhs = rhs.item()  # scalar, not numpy array

        # Computing control using QP
        # Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
        w = 1000.0  # slack weight

        P = np.zeros((5, 5))
        P[:4, :4] = 2 * self.env.mixer.T @ self.env.mixer
        q = np.concatenate([-2 * u_ref.T @ self.env.mixer, np.array([w])])
        q = np.reshape(q, (-1, 1))

        G = np.zeros((10, 5))
        G[0, 0:4] = lhs @ self.env.mixer
        G[0, 4] = -1.0
        G[1:5, 0:4] = -np.eye(4)
        G[5:9, 0:4] = np.eye(4)
        G[-1, -1] = -1.0

        rho = np.concatenate([np.array([rhs]), np.zeros(4), np.ones(4), np.zeros(1)])
        rho = np.reshape(rho, (-1, 1))

        try:
            sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(rho))
        except:
            print("QP solve was unsuccessful, with status: %s " % sol_obj["status"])
            print("Go to line 96 in flying_cbf_controller")
            print("exiting")
            exit(0)

        # print("ln 94 in cbf controller")
        # print("Try to check out the properties on sol_obj")
        # print("So that we can debug exceptions")
        # IPython.embed()
        sol_var = np.array(sol_obj['x'])

        # u_safe = sol_var[0:4]
        sol_impulses = sol_var[0:4]
        u_safe = self.env.mixer @ np.reshape(sol_impulses, (4, 1))

        u_safe = np.reshape(u_safe, (4))
        qp_slack = sol_var[-1]

        # print("Slack: %.6f" % qp_slack) # TODO
        # print(sol_impulses, u_safe, qp_slack)
        impulses = sol_impulses
        debug_dict = {"apply_u_safe": apply_u_safe, "u_ref": u_ref, "phi_vals": phi_vals.flatten(),
                      "qp_slack": qp_slack, "qp_rhs": qp_rhs, "qp_lhs": qp_lhs, "impulses": impulses,
                     "inside_boundary": inside_boundary, "on_boundary": on_boundary, "outside_boundary": outside_boundary}
        return u_safe, debug_dict
