"""
    5-point method for the Laplace equation using fast-Poisson solver.
    
"""
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

# Functions to define the boundary conditions:
def u_btm(x):
    return np.sin(np.pi*x)

def u_top(x):
    return np.sin(np.pi*x)*np.exp(-np.pi)

def u_left(y):
    return y*0

def u_right(y):
    return y*0

# Bounds of the domain (assumed square):
a = 0
b = 1

# Number of discretization points in each direction:
J = 64

h = (b-a)/float(J)

# Apply the boundary conditions and store in the relevant part of the right-hand side array:
U = np.zeros((J-1, J-1))
x = np.linspace(a+h, b-h, J-1)
U[0,:] -= u_btm(x)
U[-1,:] -= u_top(x)
y = np.linspace(a+h, b-h, J-1)
U[:,0] -= u_left(y)
U[:,-1] -= u_right(y)

# Compute the discrete sine transform of the right-hand side:
U = fft.dstn(U, type=1)

# Solve in Fourier space:
M, N = np.meshgrid(range(1,J), range(1,J))
U = 0.5*U/(np.cos(np.pi*M/J) + np.cos(np.pi*N/J) - 2)

# Invert the discrete sine transform:
U = fft.idstn(U, type=1)/(2*J)**2

# Add boundary values:
U = np.concatenate((np.zeros((1,J-1)), U, np.zeros((1,J-1))), axis=0)
U = np.concatenate((np.zeros((J+1,1)), U, np.zeros((J+1,1))), axis=1)
x = np.linspace(a, b, J+1)
y = np.linspace(a, b, J+1)
U[0,:] = u_btm(x)
U[-1,:] = u_top(x)
U[:,0] = u_left(y)
U[:,-1] = u_right(y)

# Exact solution:
X, Y = np.meshgrid(x, y)
uexact = np.sin(np.pi*X)*np.exp(-np.pi*Y)

# Plot the result against the exact solution:
plt.figure(figsize=(12,3))
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,3,1)
plt.contour(x, y, U, 30)
plt.title('Numerical $U$, J = %g' % J)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar()
plt.subplot(1,3,2)
plt.contour(x, y, uexact, 30)
plt.title('Exact $u$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar()
plt.subplot(1,3,3)
plt.title('Error $U - u$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.contour(x, y, U - uexact, 30)
plt.colorbar()
plt.show()
