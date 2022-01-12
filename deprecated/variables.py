import numpy as np
import scipy.signal as signal
import scipy.linalg as linalg
m1 = 0.3
m2 = 0.2
g = 9.8
L = 0.5
l = L/2             # CoM of uniform rod
I = (1./12) * m2 * (L**2)
b = 1           # dampending coefficient

# linear system
denominator = m1*m2*(l**2) + I*(m1 + m2)
A21 = (m2**2)*g*(l**2) / denominator
A31 = m2*g*l*(m1+m2) / denominator
A22 = -b*(m2*(l**2)+I) / denominator
A32 = -m2*l*b / denominator
B2 = (m2*(l**2)+I) / denominator
B3 = m2*l / denominator

A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, A21, A22, 0],
              [0, A31, A32, 0]])

B = np.array([[0], [0], [B2], [B3]])

C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

D = np.array([[0], [0]])

Q = np.array([[1, 0, 0, 0],
              [0, 10, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

R = [[0.0001]]
P = np.matrix(linalg.solve_continuous_are(A, B, Q, R))
K = np.matrix(linalg.inv(R)*(B.T*P))
eigVals, eigVecs = linalg.eig(A-B*K)

print("eigVecs")
print(eigVecs)
print("eigVals")
print(eigVals)
print("K")
print(K)