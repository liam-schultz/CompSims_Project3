import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

mu_e = 2
R_0 = 7.72e8/mu_e
M_0 = 5.67e33/(mu_e**2)
density_threshold = 1e-5

def drhodr_dmdr(r, state):
    #state = [rho, m]
    rho = state[0]
    m = state[1]
    drho = -(3*m*rho*np.sqrt(1+np.cbrt(rho)**2))/(np.cbrt(rho)**2*r**2)
    dm = r**2*rho
    return [drho, dm]

def zero_density(r, state):
    # density_threshold prevents the solver from getting stuck
    return state[0] - density_threshold
zero_density.terminal = True

def solve(rho):
    solutions = []
    for rho_c in rho_cs:
        solutions.append(sp.integrate.solve_ivp(drhodr_dmdr, (1e-100, 1e300), [rho_c, 0], events=zero_density))

    radii = np.array([solution.t_events[0][0] for solution in solutions])
    masses = np.array([solution.y_events[0][0][1] for solution in solutions])

    radii *= R_0
    masses *= M_0

    return radii, masses

rho_cs = np.logspace(-1, np.log(2.5e6), 10)

radii, masses = solve(rho_cs)

plt.scatter(masses, radii)
plt.title("Radius vs Mass for Values of $\\rho_c$ between $10^{-1}$ and $2.5\\cdot10^6$")
plt.xlabel("Mass (g)")
plt.ylabel("Radius (cm)")
plt.show()
print(f"Chandrasekhar limit: {(5.836/mu_e**2)*1.989e33}g")
print(f"Estimate: {masses[-1]}")
print(f"Percent difference: {100*((5.836/mu_e**2)*1.989e33 - masses[-1])/masses[-1]}%\n")

rho_cs = np.array([rho_cs[0], rho_cs[2], rho_cs[-1]])
