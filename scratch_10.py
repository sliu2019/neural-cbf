import matplotlib.pyplot as plt
import numpy as np
import IPython

def reg(x_arr):
	x_arr = x_arr - 0.2693 # + root
	softplus = np.log(1 + np.exp(x_arr))
	sigmoid = 1 / (1.0 + np.exp(-x_arr))
	rv = -(softplus + sigmoid) + 1
	return rv


x_vals = np.linspace(-5, 30, 1000)
y_vals = reg(x_vals)

# IPython.embed()
plt.plot(x_vals, y_vals)
plt.show()