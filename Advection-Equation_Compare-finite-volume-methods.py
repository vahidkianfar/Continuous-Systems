"""
    Compare finite-volume methods for the advection equation u_t + c_x = 0.
    
"""
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def uexact(x, t):
    """Exact solution."""
    return scipy.special.erf(4*(x-t)) 

def minmod(a, b):
    f = a
    f[np.abs(a) > np.abs(b)] = b[np.abs(a) > np.abs(b)]
    f[a*b <= 0] = 0
    return f

# Bounds of the domain:
a = -3
b = 7
T = 5

# Number of discretization points in x and t:
J = 100
N = 100

# Set up grid:
h = (b-a)/float(J)
k = T/float(N)
lam = k/h
x = np.linspace(a+0.5*h, b-0.5*h, J)

# Define the initial conditions:
U_dc = uexact(x, 0) 
U_lw = uexact(x, 0) 
U_fm = uexact(x, 0) 
U_mm = uexact(x, 0)

# Main loop to update solution step-by-step in time:
t = 0
for n in range(N):
    # Donor cell:
    U_dc[1:-1] += -lam*(U_dc[1:-1] - U_dc[:-2])
    
    # Lax-Wendroff:
    U_lw[1:-1] += -0.5*lam*(U_lw[2:] - U_lw[:-2]) + 0.5*lam**2*(U_lw[:-2] - 2*U_lw[1:-1] + U_lw[2:])
    
    # Fromm's method:
    U_fm[2:-1] += -lam*(U_fm[2:-1] - U_fm[1:-2]) - 0.25*lam*(1-lam)*(U_fm[3:] - U_fm[1:-2] - U_fm[2:-1] + U_fm[:-3])

    # Minmod method:
    sig = minmod(U_mm[1:-1] - U_mm[:-2], U_mm[2:] - U_mm[1:-1])
    U_mm[2:-1] += -lam*(U_mm[2:-1] - U_mm[1:-2]) - 0.5*lam*(1-lam)*(sig[1:] - sig[:-1])
    
    t += k

fig = plt.figure(1,figsize=(6,4))
plt.plot(x, U_dc, marker='o', fillstyle='none', label='Donor cell')
plt.plot(x, U_lw, marker='o', fillstyle='none', label='Lax-Wendroff')
plt.plot(x, U_fm, marker='o', fillstyle='none', label='Fromm')
plt.plot(x, U_mm, marker='o', fillstyle='none', label='minmod')
plt.plot(x, uexact(x,T), color='tab:red', linestyle='--')
plt.xlabel(r'$x$')
plt.xlim(2,7)
plt.legend()
fig.canvas.draw()
#plt.savefig('../notes_coreIIb_continuous/pics/fv_minmod.png', bbox_inches='tight')
plt.show()
