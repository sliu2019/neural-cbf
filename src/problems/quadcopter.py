import torch
import numpy as np

from torch import nn
import os, sys
import IPython
import math

g = 9.81
class HMax(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Return value is size (bs, 1)

		print("Inside HMax forward")
		IPython.embed()

		# Try one "wall" first
		py = x[:, [self.i["py"]]]
		gamma = x[:, [self.i["gamma"]]]
		beta = x[:, [self.i["beta"]]]

		rv = torch.maximum(py + 0.5, gamma**2 + beta**2 - (math.pi/3)**2)
		return rv

class XDot(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device
		self.i = self.state_index_dict

	def forward(self, x, u):
		# x: bs x 10, u: bs x 4
		# The way these are implemented should be batch compliant

		# Pre-computations
		# Compute the rotation matrix from quad to global frame
		# Extract the k_{x,y,z}
		gamma = x[:, self.i["gamma"]]
		beta = x[:, self.i["beta"]]
		alpha = x[:, self.i["alpha"]]

		R = torch.zeros((x.shape[0], 3, 3), device=self.device) # is this the correct rotation?
		R[:, 0, 0] = torch.cos(alpha)*torch.cos(beta)
		R[:, 0, 1] = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma)
		R[:, 0, 2] = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)
		R[:, 1, 0] = torch.sin(alpha)*torch.cos(beta)
		R[:, 1, 1] = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma)
		R[:, 1, 2] = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
		R[:, 2, 0] = -torch.sin(beta)
		R[:, 2, 1] = torch.cos(beta)*torch.sin(gamma)
		R[:, 2, 2] = torch.cos(beta)*torch.cos(gamma)

		k_x = R[:, 0, 2]
		k_y = R[:, 1, 2]
		k_z = R[:, 2, 2]

		F = (u[:, 0] + self.M*g)

		###### Computing state derivatives
		J = torch.tensor([self.J_x, self.J_y, self.J_z]).to(self.device)
		norm_torques = u[:, 1:]*(1.0/J)

		ddquad_angles = torch.bmm(R, norm_torques[:, :, None]) # (N, 3, 1)
		ddquad_angles = ddquad_angles[:, :, 0]

		ddgamma = ddquad_angles[:, 0]
		ddbeta = ddquad_angles[:, 1]
		ddalpha = ddquad_angles[:, 2]

		ddx = k_x * (F/self.m + self.g)
		ddy = k_y * (F/self.m + self.g)
		ddz = k_z * (F/self.m + self.g) - self.g

		# Excluding translational motion
		rv = torch.cat([x[:, [self.i["dpx"]]], x[:, [self.i["dpy"]]], x[:, [self.i["dpz"]]], ddx[:, None], ddy[:, None], ddz[:, None], x[:, [self.i["dgamma"]]], x[:, [self.i["dbeta"]]], x[:, [self.i["dalpha"]]], ddgamma[:, None], ddbeta[:, None], ddalpha[:, None]], axis=1)
		return rv

class ULimitSetVertices(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device

		# precompute, just tile it in forward()
		k1 = self.k1
		k2 = self.k2
		l = self.l
		M = np.array([[k1, k1, k1, k1], [0, -l*k1, 0, l*k1], [l*k1, 0, -l*k1, 0], [-k2, k2, -k2, k2]]) # mixer matrix

		r1 = np.concatenate((np.zeros(8), np.ones(8)))
		r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
		r3 = np.concatenate((np.zeros(2), np.ones(2),np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),np.zeros(2), np.ones(2)))
		r4 = np.zeros(16)
		r4[1::2] = 1.0
		impulse_vert = np.concatenate((r1[None], r2[None], r3[None], r4[None]), axis=0) # 16 vertices in the impulse control space

		force_vert = M@impulse_vert - np.array([[self.m*g], [0.0], [0.0], [0.0]]) # Fixed bug: was subtracting self.M*g (not just in the first row)
		force_vert = force_vert.T.astype("float32")
		self.vert = torch.from_numpy(force_vert).to(self.device)

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# (bs, n_vertices, u_dim) or (bs, 16, 4)

		rv = self.vert
		rv = rv.unsqueeze(dim=0)
		rv = rv.expand(x.shape[0], -1, -1)
		return rv

if __name__ == "__main__":
	pass
