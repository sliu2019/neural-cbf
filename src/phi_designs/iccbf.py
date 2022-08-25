import torch
from torch import nn
from torch.autograd import grad
import IPython

class ICCBF(nn.Module):
	# Note: currently, we have a implementation which is generic to any r. May be slow
	def __init__(self, h_fn, xdot_fn, uvertices_fn, class_kappa_fns, x_dim, u_dim, device, args):
		#, nn_input_modifier=None, x_e=None):

		# Later: args specifying how beta is parametrized
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		# assert r>=0

		# turn Namespace into dict
		args_dict = vars(args)

		self.N = len(class_kappa_fns) # the number of "iterations" of the CBF; previously equal to degree

	def forward(self, x, grad_x=False):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		# RV is (bs, r+1)

		IPython.embed()
		bs = x.size()[0]

		hi = self.h_fn(x) # bs x 1, the zeroth derivative

		hi_list = [] # bs x 1
		# f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device)) # bs x x_dim
		u_lim_set_vertices = self.uvertices_fn(x)  # (bs, n_vertices, u_dim), can be a function of x_batch
		n_vertices = u_lim_set_vertices.size()[1]
		U = torch.reshape(u_lim_set_vertices, (-1, self.u_dim))  # (bs x n_vertices, u_dim)

		X = (x.unsqueeze(1)).repeat(1, n_vertices, 1)  # (bs, n_vertices, x_dim)
		X = torch.reshape(X, (-1, self.x_dim))  # (bs x n_vertices, x_dim)

		# Evaluate every X against multiple U
		xdot = self.xdot_fn(X, U)

		orig_req_grad_setting = x.requires_grad
		x.requires_grad = True

		for i in range(self.N): # N+1: just how it's defined in paper
			# phi_value = self.phi_fn(x)
			# grad_phi = grad([torch.sum(phi_value[:, -1])], x, create_graph=True)[0]  # check
			grad_hi = grad([torch.sum(hi)], x, create_graph=True)[0] # TODO: check

			grad_hi = (grad_hi.unsqueeze(1)).repeat(1, n_vertices, 1)
			grad_hi = torch.reshape(grad_hi, (-1, self.x_dim))

			# Dot product
			dot_hi_cand = xdot.unsqueeze(1).bmm(grad_hi.unsqueeze(2))
			dot_hi_cand = torch.reshape(dot_hi_cand, (-1, n_vertices))  # bs x n_vertices

			dot_hi, _ = torch.min(dot_hi_cand, 1)

			hiplus1 = dot_hi + self.class_kappa_fns[i](hi)
			hiplus1 = hiplus1.view(-1, 1)

			hi_list.append(hiplus1)
			hi = hiplus1

		x.requires_grad = orig_req_grad_setting

		result = torch.cat(hi_list, dim=1)
		return result


if __name__ == "__main__":
	pass