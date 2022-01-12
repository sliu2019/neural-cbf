import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.integrate import solve_ivp
from time import time

import deprecated.variables as variables

fig = plt.figure()
ax = fig.add_subplot(111, aspect = 'equal', xlim = (-5, 5), ylim = (-1, 1), title = "Inverted Pendulum Simulation")
ax.grid()

# animation parameters
origin = [0.0, 0.0]
dt = 0.02
frames = 200
t_span = [0.0, frames * dt]

# temporary storage, will restructure variables.py and stateSpace() later
m1 = variables.m1
m2 = variables.m2
g = variables.g
L = variables.L
l = variables.l
I = variables.I
b = variables.b
ss=np.array([0., 0., 0., 0.])

pendulumArm = lines.Line2D(origin, origin, color='r')
cart = patches.Rectangle(origin, 0.5, 0.15, color='b')

def stateSpace(t, i):
    # returns state space model evaluated at initial conditions
    S = np.sin(i[1])
    C = np.cos(i[1])
    denominator = (m2*l*C)**2 - (m1+m2) * (m2*(l**2) + I)
    # nonlinear system
    ss[0] = i[2]
    ss[1] = i[3]
    print(u(i))
    ss[2] = (1/denominator)*(-(m2**2)*g*(l**2)*C*S - m2*l*S*(m2*(l**2)+I)*(i[3]**2) + b*(m2*(l**2) + I)*i[2]) - ((m2*(l**2)+I)/denominator)*u(i)
    ss[3] = (1/denominator)*((m1+m2)*m2*g*l*S + (m2**2)*(l**2)*S*C*(i[3]**2) - b*m2*l*C*i[2]) + ((m2*l*C)/denominator)*u(i)
    return ss

initial=np.array([-1, np.pi-0.5, -5, 0])
final=np.array([1, np.pi, 0, 0])
u = lambda x: np.matmul(-variables.K, (x - final))
ts = np.linspace(t_span[0], t_span[1], frames)
pendulum_state = solve_ivp(stateSpace, t_span, initial, t_eval = ts, rtol=1e-8)

def init():
    ax.add_patch(cart)
    ax.add_line(pendulumArm)
    return pendulumArm, cart

def animate(i):
    xPos = pendulum_state.y[0][i]
    theta = pendulum_state.y[1][i]
    x = [origin[0] + xPos, origin[0] + xPos + variables.L * np.sin(theta)]
    y = [origin[1], origin[1] - variables.L * np.cos(theta)]
    pendulumArm.set_xdata(x)
    pendulumArm.set_ydata(y)
    cartPos = [origin[0] + xPos - cart.get_width()/2, origin[1] - cart.get_height()]
    cart.set_xy(cartPos)
    return pendulumArm, cart

t0 = time()
animate(0)                          # sample time required to evaluate animate() function
t1 = time()
interval = 1000 * dt - (t1 - t0)

anim = animation.FuncAnimation(fig, animate, init_func = init, frames = frames, interval = interval, blit = True)
plt.show()