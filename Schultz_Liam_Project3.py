import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#define constants
mu_e = 2
R_0 = 7.72e8/mu_e
M_0 = 5.67e33/(mu_e**2)
density_threshold = 1e-5

def drhodr_dmdr(r, state):
    """
    Calculates the derivative of the state vector for a white dwarf's mass and density as a function of r
    :param r: the radius of the star
    :param state: the state [rho, m] of the star
    :return: the change in the state at radius r [drho/dr, dm/dr]
    """
    #extract values of rho and m
    rho = state[0]
    m = state[1]

    #compute the difference in rho and m
    drho = -(3*m*rho*np.sqrt(1+np.cbrt(rho)**2))/(np.cbrt(rho)**2*r**2)
    dm = r**2*rho

    #return the derivative of the state vector
    return [drho, dm]

def zero_density(r, state):
    """
    An event to track when the density goes to zero (within a certain threshold)
    :param r: the radius of the star
    :param state: the state [rho, m] of the star
    :return: the density rho of the star minus a density threshold (1e-5)
    """
    # density_threshold prevents the solver from getting stuck
    return state[0] - density_threshold

#sets the terminal attribute of the zero_density function so the ivp solver terminates when the sign of the returned values changes
zero_density.terminal = True

def solve(rho, method="RK45"):
    """
    solves the ivp problem for many values of rho_c
    :param rho: the values of rho_c to use to solve the ivp problem
    :param method: the method to use to solve the ivp problem, see the documentation for scipy.integrate.solve_ivp for valid values
    :return: the radii and masses of the stars for given values of rho_c in cm and g respectively
    """
    #compute the solutions to the ivp problem for each passed value of rho_c
    solutions = []
    for rho_c in rho:
        solutions.append(sp.integrate.solve_ivp(drhodr_dmdr, (1e-100, 1e300), [rho_c, 0], events=zero_density, method=method))

    #extract the radii and masses
    radii = np.array([solution.t_events[0][0] for solution in solutions])
    masses = np.array([solution.y_events[0][0][1] for solution in solutions])

    #convert the radii and masses to cm and g
    radii *= R_0
    masses *= M_0

    return radii, masses

#generate values of rho_c
rho_cs = np.logspace(-1, np.log10(2.5e6), 10)

radii, masses = solve(rho_cs)

#plot the radii vs the masses
plt.scatter(masses, radii)
plt.title("Radius vs Mass for Values of $\\rho_c$ between $10^{-1}$ and $2.5\\cdot10^6$")
plt.xlabel("Mass (g)")
plt.ylabel("Radius (cm)")
plt.show()
print(f"Chandrasekhar limit: {(5.836/mu_e**2)*1.989e33}g")
print(f"Estimate: {masses[-1]}g")
print(f"Percent difference: {100*((5.836/mu_e**2)*1.989e33 - masses[-1])/masses[-1]}%\n")

#pick 3 values of rho_c to try another solving method on
rho_cs = np.array([rho_cs[0], rho_cs[2], rho_cs[-1]])
#get radii and masses to compare against
cmp_radii = np.array([radii[0], radii[2], radii[-1]])
cmp_masses = np.array([masses[0], masses[2], masses[-1]])

#solve for new radii and masses
DOP_radii, DOP_masses = solve(rho_cs, "DOP853")

#calculate percent differences
rad_dif = 100*(cmp_radii-DOP_radii)/DOP_radii
mass_dif = 100*(cmp_masses-DOP_masses)/DOP_masses

#print table of differences
print("Rho_c\t\t\tRadius Percent Difference\tMass Percent Difference")
for i in range(len(rho_cs)):
    print(f"{rho_cs[i]:e}\t{rad_dif[i]:f}%\t\t\t\t\t{mass_dif[i]:f}%")

#load experimental data
wd_data = np.loadtxt("wd_mass_radius.csv", skiprows=1, delimiter=',')

#convert to masses to grams and radii to cm
wd_data[:, 0:2] *= 1.989e33
wd_data[:, 2:] *= 6.957e10

#solve for more values of rho to produce a smooth plot
rho_cs = np.logspace(-1, np.log10(2.5e6), 1000)
radii, masses = solve(rho_cs)

#plot theoretical and experimental values
plt.plot(masses, radii)
plt.errorbar(wd_data[:, 0], wd_data[:, 2], xerr=wd_data[:, 1], yerr=wd_data[:, 3], fmt=".k")
plt.title("Radius vs Mass for Values of $\\rho_c$ between $10^{-1}$ and $2.5\\cdot10^6$")
plt.legend(["Theoretical", "Experimental"])
plt.xlabel("Mass (g)")
plt.ylabel("Radius (cm)")
plt.show()

#Repo hosted at https://github.com/liam-schultz/CompSims_Project3