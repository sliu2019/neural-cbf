import matplotlib.pyplot as plt
import numpy as np
import IPython

def reg(x_arr):
	x_arr = x_arr + 1.586586586586587
	softplus = np.log(1 + np.exp(x_arr))
	sigmoid = 1 / (1.0 + np.exp(-x_arr))
	rv = -(sigmoid + 0.1*softplus) + 1
	return rv

# print(reg(np.array([0])))

x_vals = np.linspace(-5, 30, 1000)
y_vals = reg(x_vals)

# ind= np.argmin(np.abs(y_vals))
# print(y_vals[ind-1], y_vals[ind], y_vals[ind+1])
# print(ind)
# print(x_vals[ind+1])

# IPython.embed()
plt.plot(x_vals, y_vals)
plt.show()