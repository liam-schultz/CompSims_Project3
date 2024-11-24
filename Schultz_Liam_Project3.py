import numpy as np
import scipy as sp

def drhodr_dmdr(r, state):
    #state = [rho, m]
    rho = state[0]
    m = state[1]
    drho = -(3*m*rho*np.sqrt(1+np.cbrt(rho)**2))/(np.cbrt(rho)**2*r**2)
    dm = r**2*rho
    return [drho, dm]

def zero_density(r, state):
    zero_density.terminal = True
    return state[0]

rho_cs = np.linspace(1e-1, 2.5e6, 10)
solutions = []
for rho_c in rho_cs:
    solutions.append(sp.integrate.solve_ivp(drhodr_dmdr, (1e-100, 1e300), [10, 0], events=zero_density))

radii = [solution.t_events[0][0] for solution in solutions]
masses = [solution.y_events[0][0][1] for solution in solutions]
