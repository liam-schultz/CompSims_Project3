import numpy as np

def drhodr_dmdr(r, state):
    #state = [rho, m]
    rho = state[0]
    m = state[1]
    drho = (3*m*rho*np.sqrt(1+np.cbrt(rho)**2))/(np.cbrt(rho)**2*r**2)
    dm = r**2*rho
    return [drho, m]