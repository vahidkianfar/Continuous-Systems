"""
    Backward-Euler method for the heat equation u_t = u_{xx}.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Exact solution [for testing!]:
def uexact(x, t):
    return np.exp(-x**2/(4*t + 1.))/np.sqrt(4*t+1)

# Bounds of the domain:
a = -5
b = 5
T = 1

# Number of discretization points in x and t:
J = 50
N = 20

# Set up grid:
h = (b-a)/float(J)
k = T/float(N)
mu = k/h**2
x = np.linspace(a, b, J+1)

# Define the initial conditions:
U = uexact(x, 0)

# Set up the plotting:
fig = plt.figure(1,figsize=(6,4))
line, = plt.plot(x, U, color='tab:blue', label=r'$U^n$')
plt.plot(x, uexact(x, 0), alpha=0.25, color='tab:blue', label=r'$u_0$')
plt.ylim(0, 1)
plt.xlabel(r'$x$')
plt.legend()
plt.text(-4.5, 0.9, r'$\mu = $ %g' % mu)

# Array to store the errors:
error = np.zeros(N)

# Formulate matrix (in banded form):
A = np.zeros((3, J-1))
A[0,1:] = -mu  # upper diagonal
A[1,:] = 1 + 2*mu  # main diagonal
A[2,:-1] = -mu  # lower diagonal

# Boundary condition vector:
Ub = np.zeros(J-1)

# Main loop to update solution step-by-step in time:
t = 0
for n in range(N):

    # Advance solution by solving linear system:
    U[1:-1] = la.solve_banded((1,1), A, U[1:-1] + mu*Ub)
    
    # Time:
    t += k

    # Compute global truncation error:
    error[n] = np.max(np.abs(U - uexact(x, t)))
    
    # Update plot:
    line.set_ydata(U)
    plt.title(r'$t =$ %g' % t)
    fig.canvas.draw()
    plt.pause(0.1)
    
plt.plot(x, U, color='tab:blue', marker='o', fillstyle='none')
plt.plot(x, uexact(x, T), 'r--', label=r'$u(x,T)$')
plt.legend()
fig.canvas.draw()
plt.show()

# Plot error versus time:
# plt.close()
# fig = plt.figure(1,figsize=(6,3))
# plt.plot(np.linspace(0, T, N), error)
# plt.xlabel(r'$t$')
# plt.ylabel(r'$T_n$')
# plt.title(r'$\mu = $ %g' % mu)
# plt.show()
