import torch
import numpy as np

from torch import nn
import os, sys
import IPython
import math

g = 9.81
class H(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Return value is size (bs, 1)
		theta = x[:, [self.i["theta"]]]
		phi = x[:, [self.i["phi"]]]

		cos_cos = torch.cos(theta)*torch.cos(phi)
		eps = 1e-4 # TODO: prevents nan when cos_cos = +/- 1
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos)*eps
		rv = torch.acos(cos_cos + signed_eps) - self.delta_safety_limit # torch.acos output is in [0, 2pi]? Probably doesn't matter for our purposes

		return rv

class XDot(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		# IPython.embed()
		self.device = device
		self.i = self.state_index_dict

	def forward(self, x, u, same_u=False): # TODO: same_u
		# x: bs x 10, u: bs x 4
		# The way these are implemented should be batch compliant

		# Pre-computations
		# Compute the rotation matrix from quad to global frame
		# Extract the k_{x,y,z}
		gamma = x[:, self.i["gamma"]]
		beta = x[:, self.i["beta"]]
		alpha = x[:, self.i["alpha"]]

		phi = x[:, self.i["phi"]]
		theta = x[:, self.i["theta"]]
		dphi = x[:, self.i["dphi"]]
		dtheta = x[:, self.i["dtheta"]]

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

		F = (u[:, 0] + self.M*g) # TODO: Mg

		###### Computing state derivatives
		ddquad_angles = torch.bmm(R, u[:, 1:, None]) # (N, 3, 1)
		ddquad_angles = ddquad_angles[:, :, 0]
		ddgamma = (1.0/self.J_x)*ddquad_angles[:, 0]
		ddbeta = (1.0/self.J_y)*ddquad_angles[:, 1]
		ddalpha = (1.0/self.J_z)*ddquad_angles[:, 2]

		ddphi = (3.0)*(k_y*torch.cos(phi) + k_z*torch.sin(phi))/(2*self.M*self.L_p*torch.cos(theta))*F + 2*dtheta*dphi*torch.tan(theta)
		ddtheta = (3.0*(-k_x*torch.cos(theta)-k_y*torch.sin(phi)*torch.sin(theta) + k_z*torch.cos(phi)*torch.sin(theta))/(2.0*self.M*self.L_p))*F - torch.square(dphi)*torch.sin(theta)*torch.cos(theta)

		# rv = torch.cat([x[:, [self.i["dx"]]], x[:, [self.i["dy"]]], x[:, [self.i["dz"]]], ddx[:, None], ddy[:, None], ddz[:, None], x[:, [self.i["dgamma"]]], x[:, [self.i["dbeta"]]], x[:, [self.i["dalpha"]]], ddgamma[:, None], ddbeta[:, None], ddalpha[:, None], dphi[:, None], dtheta[:, None], ddphi[:, None], ddtheta[:, None]], axis=1)
		# Excluding translational motion
		rv = torch.cat([x[:, [self.i["dgamma"]]], x[:, [self.i["dbeta"]]], x[:, [self.i["dalpha"]]], ddgamma[:, None], ddbeta[:, None], ddalpha[:, None], dphi[:, None], dtheta[:, None], ddphi[:, None], ddtheta[:, None]], axis=1)
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

		"""
		Biasing the force by Mg makes computing the limit set a little complicated
		Impulses have constraints
		F (thrust) must be positive		
		So after shifting the polyhedron U_lim down by Mg on the F axis, we also have to compute any vertices appearing at the intersection of this polyhedron and the 0-plane
		"""

		"""
		# Deprecated
		force_vert = M@impulse_vert 
		force_vert = force_vert.T.astype("float32")
		self.vert = torch.from_numpy(force_vert).to(self.device)
		"""
		# IPython.embed()

		force_vert = M@impulse_vert + np.array([[-self.M*g], [0],[0], [0]]) # (4, 16)

		"""
		Go through all the edges and check if they intersect with the plane. If so, find the intersection.
		
		Affine transformations preserve edges (as well as vertices)
		So we'll go through the edges in U-space. Each U-vertex has 4 neighbors.
		Naturally, we will be double-computing the plane-poly intersection vertices.
		"""
		# all_cross_points = []
		#
		# for i in range(16):
		# 	v = impulse_vert[:, [i]]
		# 	neighbors = (v + np.eye(4)) % 2
		# 	transf_v = M@v + np.array([[-self.M*g], [0],[0], [0]])
		# 	transf_neighbors = M@neighbors + np.array([[-self.M*g], [0],[0], [0]])
		#
		# 	# Does edge cross F = 0 plane?
		# 	ind = np.argwhere(transf_v[0, 0]*transf_neighbors[0] < 0).flatten()
		# 	if ind.size:
		# 		crossing_neigh = transf_neighbors[:, ind]
		#
		# 		# print(transf_v, crossing_neigh)
		# 		# Find where they intersect
		# 		alphas = -transf_v[0]/(crossing_neigh-transf_v)[0, :]
		# 		cross_point = transf_v + alphas*(crossing_neigh-transf_v)
		# 		all_cross_points.extend(list(cross_point.T))
		#
		# # Set
		# all_cross_points_set = {tuple(cross_point) for cross_point in all_cross_points}
		# all_cross_points = np.array([np.array(x) for x in all_cross_points_set])
		#
		# pos_vert_ind = np.argwhere(force_vert[0, :]>=0).flatten()
		# pos_vert = force_vert[:, pos_vert_ind].T # (4, whatever)
		#
		# all_vert = np.concatenate((all_cross_points, pos_vert), axis=0) # (25, 4)
		# all_vert = all_vert.astype("float32")

		# Processing for Pytorch
		all_vert = np.array([[ 0.0000e+00, -6.0000e+00,  5.6394e+00, -3.0050e-03],
       [ 0.0000e+00,  6.0000e+00,  5.6394e+00, -3.0050e-03],
       [ 0.0000e+00,  6.0000e+00,  5.6394e+00, -3.0050e-03],
       [ 0.0000e+00, -5.6394e+00, -6.0000e+00,  3.0050e-03],
       [ 0.0000e+00, -5.6394e+00, -6.0000e+00,  3.0050e-03],
       [ 0.0000e+00,  3.6060e-01,  0.0000e+00, -9.6995e-02],
       [ 0.0000e+00,  5.6394e+00,  6.0000e+00,  3.0050e-03],
       [ 0.0000e+00,  5.6394e+00,  6.0000e+00,  3.0050e-03],
       [ 0.0000e+00,  6.0000e+00, -5.6394e+00, -3.0050e-03],
       [ 0.0000e+00, -6.0000e+00, -5.6394e+00, -3.0050e-03],
       [ 0.0000e+00,  0.0000e+00,  3.6060e-01,  9.6995e-02],
       [ 0.0000e+00, -6.0000e+00, -5.6394e+00, -3.0050e-03],
       [ 0.0000e+00,  6.0000e+00, -5.6394e+00, -3.0050e-03],
       [ 0.0000e+00, -3.6060e-01,  0.0000e+00, -9.6995e-02],
       [ 0.0000e+00, -5.6394e+00,  6.0000e+00,  3.0050e-03],
       [ 0.0000e+00, -5.6394e+00,  6.0000e+00,  3.0050e-03],
       [ 0.0000e+00,  0.0000e+00, -3.6060e-01,  9.6995e-02],
       [ 0.0000e+00,  5.6394e+00, -6.0000e+00,  3.0050e-03],
       [ 0.0000e+00,  5.6394e+00, -6.0000e+00,  3.0050e-03],
       [ 0.0000e+00, -6.0000e+00,  5.6394e+00, -3.0050e-03],
       [ 3.7596e+00,  0.0000e+00, -6.0000e+00,  5.0000e-02],
       [ 3.7596e+00,  6.0000e+00,  0.0000e+00, -5.0000e-02],
       [ 3.7596e+00,  0.0000e+00,  6.0000e+00,  5.0000e-02],
       [ 3.7596e+00, -6.0000e+00,  0.0000e+00, -5.0000e-02],
       [ 7.7596e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]],
      dtype="float32")
		# print(repr(all_vert))
		self.vert = torch.from_numpy(all_vert).to(self.device)

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# (bs, n_vertices, u_dim) or (bs, 16, 4)

		rv = self.vert
		rv = rv.unsqueeze(dim=0)
		rv = rv.expand(x.shape[0], -1, -1)
		return rv

if __name__ == "__main__":
	param_dict = {
		"m": 0.8,
		"J_x": 0.005,
		"J_y": 0.005,
		"J_z": 0.009,
		"l": 1.5,
		"k1": 4.0,
		"k2": 0.05,
		"m_p": 0.04,
		"L_p": 0.03, # TODO?
		'delta_safety_limit': math.pi/5 # in radians; should be <= math.pi/4
	}
	param_dict["M"] = param_dict["m"] + param_dict["m_p"]

	state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi", "dtheta"] # excluded x, y, z
	state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
	param_dict["state_index_dict"] = state_index_dict
	##############################################

	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'
		dev = "cuda:%i" % (0)
		print("Using GPU device: %s" % dev)
	else:
		dev = "cpu"
	device = torch.device(dev)

	h_fn = H(param_dict)
	xdot_fn = XDot(param_dict, device)
	uvertices_fn = ULimitSetVertices(param_dict, device)

	N = 2
	# x = torch.rand(N, 13).to(device)
	# u = torch.rand(N, 4).to(device)

	# IPython.embed()
	# rv1 = h_fn(x)
	x = torch.zeros(1, 10).to(device)
	u = torch.zeros(1, 4).to(device)
	rv2 = xdot_fn(x, u)
	# IPython.embed()
	# rv3 = uvertices_fn(x)

	# print(rv1.shape)
	# IPython.embed()