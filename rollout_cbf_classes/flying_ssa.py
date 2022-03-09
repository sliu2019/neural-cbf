from cmath import cos
import numpy as np
import math
from cvxopt import matrix, solvers
import torch
from torch.autograd import grad

class SSA:
    def __init__(self, param_dict):
        self.delta_max = param_dict['delta_safety_limit']
        self.c1 = 1
        self.c2 = 0.02 # safety index hand design rule
        # self.c2 = 1

    def phi_fn(self, x):
        ''' safety index
        Input:
            x:"gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta",
            "dphi", "dtheta", "x", "y", "z", "dx", "dy", "dz.
            "gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
	                     "dtheta", "x", "y", "z", "dx", "dy", "dz"
        '''
        # batch implementation
        try:
            assert len(x.shape) == 2 and x.shape[1] == 16
        except:
            # print(f'the shape of x is {x.shape}')
            # print(f'the x is {x}')
            assert len(x.shape) == 1 and x.shape[0] == 16
            # reshape x into 1,4
            x = np.expand_dims(x, axis=0)

        # get values
        gamma = x[:,0].reshape((x.shape[0], 1))
        beta = x[:,1].reshape((x.shape[0], 1))
        alpha = x[:,2].reshape((x.shape[0], 1))
        dgamma = x[:,3].reshape((x.shape[0], 1))
        dbeta = x[:,4].reshape((x.shape[0], 1))
        dalpha = x[:,5].reshape((x.shape[0], 1))
        phi = x[:,6].reshape((x.shape[0], 1))
        theta = x[:,7].reshape((x.shape[0], 1))
        dphi = x[:,8].reshape((x.shape[0], 1))
        dtheta = x[:,9].reshape((x.shape[0], 1))


        # compute delta values
        delta = np.arccos(np.cos(phi) * np.cos(theta))
        assert min(delta) >= 0.

        # compute dot delta values
        ddelta = (np.cos(theta)*np.sin(phi)*dphi + np.cos(phi)*np.sin(theta)*dtheta) / (1 - np.cos(theta)**2 * np.cos(phi)**2)**0.5

        # phi 0
        phi_0 = delta**2 - self.delta_max**2 + gamma**2 + beta**2
        # dot phi 0
        dot_phi_0 = 2*delta*ddelta + 2*gamma*dgamma + 2*beta*dbeta
        # phi 1
        phi_1 = phi_0 + self.c2*dot_phi_0
        # beta
        phi_beta = delta**2 - self.delta_max**2 + gamma**2 + beta**2
        phi_star = phi_1 - phi_0 + phi_beta
        # final result
        result = np.hstack((phi_0, phi_1, phi_star))

        return result


    def phi_star_torch(self, x):
        ''' safety index
        Input:
            x:"gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta",
            "dphi", "dtheta", "x", "y", "z", "dx", "dy", "dz.
        '''
        # non-batch torch implementation implementation
        assert len(x.shape)==1
        assert len(x)==16

        # get values
        gamma = x[0]
        beta = x[1]
        alpha = x[2]
        dgamma = x[3]
        dbeta = x[4]
        dalpha = x[5]
        phi = x[6]
        theta = x[7]
        dphi = x[8]
        dtheta = x[9]


        # compute delta values
        delta = torch.arccos(torch.cos(phi) * torch.cos(theta))
        assert delta.detach().cpu().numpy() >= 0.

        # compute dot delta values
        ddelta = (torch.cos(theta)*torch.sin(phi)*dphi + torch.cos(phi)*torch.sin(theta)*dtheta) / (1 - torch.cos(theta)**2 * torch.cos(phi)**2)**0.5

        # phi 0
        phi_0 = delta**2 - self.delta_max**2 + gamma**2 + beta**2
        # dot phi 0
        dot_phi_0 = 2*delta*ddelta + 2*gamma*dgamma + 2*beta*dbeta
        # phi 1
        phi_1 = phi_0 + self.c2*dot_phi_0
        # beta
        phi_beta = delta**2 - self.delta_max**2 + gamma**2 + beta**2
        phi_star = phi_1 - phi_0 + phi_beta

        return phi_star


    def phi_grad(self, x):
        ''' safety index gradient with respect to x
        Input:
            x:"gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta",
            "dphi", "dtheta", "x", "y", "z", "dx", "dy", "dz.
        '''
        # batch implementation
        try:
            assert len(x.shape) == 2 and x.shape[1] == 16
        except:
            # print(f'the shape of x is {x.shape}')
            # print(f'the x is {x}')
            assert len(x.shape) == 1 and x.shape[0] == 16
            # reshape x into 1,4
            x = np.expand_dims(x, axis=0)

        # transfer to torch
        x_torch = torch.from_numpy(x.astype("float32"))
        # final results
        gradient = np.zeros((0,x.shape[1]))
        for i in range(x.shape[0]):
            x_tmp_torch = x_torch[i,:]
            x_tmp_torch.requires_grad = True

            # compute gradient
            phi_val = self.phi_star_torch(x_tmp_torch)
            phi_tmp_grad = grad([phi_val], x_tmp_torch)[0]

            # convert to numpy
            x_tmp_torch.requires_grad = False
            phi_tmp_grad = phi_tmp_grad.detach().cpu().numpy().flatten()

            # additional
            phi_tmp_grad = np.expand_dims(phi_tmp_grad,0)
            gradient = np.vstack((gradient, phi_tmp_grad))

        return gradient


if __name__ == "__main__":
    param_dict = {}
    param_dict['delta_safety_limit'] = 0.1
    ssa = SSA(param_dict)
    x = np.ones((5,16))
    x[1,:] *= 2
    x[2,:] *= 3
    phi = ssa.phi_grad(x)
    # import ipdb; ipdb.set_trace()
    print(phi)