"""
    Lax-Wendroff scheme for the advection equation u_t + cu_x = 0.
    

"""
import numpy as np
import matplotlib.pyplot as plt

def uexact(x, t):
    """Exact solution."""
    return np.exp(-(x-c*t)**2)

# Wave speed (constant):
c = 1

# Bounds of the domain:
a = -5
b = 5
T = 2

# Number of discretization points in x and t:
J = 100
N = 40

# Set up grid:
h = (b-a)/float(J)
k = T/float(N)
lam = c*k/h
x = np.linspace(a, b, J+1)

# Define the initial conditions:
U = uexact(x, 0) 

# Set up the plotting:
fig = plt.figure(1,figsize=(6,4))
line, = plt.plot(x, U, color='tab:blue', label=r'$U^n$')
line2, = plt.plot(x, U, color='tab:red', linestyle='--', label=r'$u(x,T)$')
plt.plot(x, uexact(x,0), alpha=0.25, color='tab:blue', label=r'$u_0$')
plt.ylim(0, 1)
plt.xlabel(r'$x$')
plt.legend()
plt.text(-4.5, 0.9, r'$\lambda = $ %g' % lam)

# Main loop to update solution step-by-step in time:
for n in range(1,N+1):
    # Advance solution:
    U[1:-1] += -0.5*lam*(U[2:] - U[:-2]) + 0.5*lam**2*(U[:-2] - 2*U[1:-1] + U[2:])

    # Compute global truncation error:
    t = n*k
    
    # Update plot:
    line.set_ydata(U)
    line2.set_ydata(uexact(x,t))
    plt.title(r'$t =$ %g' % t)
    fig.canvas.draw()
    plt.pause(0.1)
    
plt.plot(x, U, color='tab:blue', marker='o', fillstyle='none')
plt.plot(x, uexact(x,t), color='tab:red', linestyle='--', fillstyle='none')
plt.legend()
fig.canvas.draw()
plt.show()

