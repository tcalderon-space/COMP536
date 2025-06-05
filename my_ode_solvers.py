#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import my_star_functions as sf
from my_astro_constants import *
import my_numerical_methods as nm
from pprint import pprint


# In[1]:


class ODE_solver:
    def __init__(self, func, state, dt, arr, tol = 1E-6):
        """
        Initialize the ODE solver.
        Parameters:
            - func: Function that returns the derivatives of the state variables.
            - state: Initial state vector (e.g., [y0, theta0]).
            - dt: Time step size.
            - arr: Array to store results.
            - tol: Tolerance for stopping condition.
        """
        self.dt = dt
        self.state = np.array(state)
        self.func = func
        self.arr = arr
        self.tol = tol

    def integrate(self, t0=1, tstop=100, integrator='euler'):
        """
        Integrate a system of ODEs using either Euler or RK4.
        Parameters:
            - t0: Initial time.
            - tstop: End time.
            - integrator: Integration method ("euler" or "runge-kutta").
        Returns:
            - Filled portion of the results array.
        """
        t = t0
        i = 0  # Initialize index

        while t <= tstop - self.tol:
            if i >= len(self.arr):  # Check if we've run out of space in the array
                raise Exception(f"You ran out of steps, try increasing the size of the array or reducing dt.")

            if integrator == "euler":
                # Euler method
                self.state = self.euler(t)
            elif integrator == "runge-kutta":
                # RK4 method
                self.state = self.runge_kutta_4(t)
            else:
                raise ValueError(f"Unknown integrator: {integrator}")

            # Break the loop if y is close to 0 (pendulum stopped)
            if self.func.__name__ == "f" and np.linalg.norm(self.state[0]) < self.tol:
                print(f"Object stopped at t = {t} s. Stopping integration.")
                break

            # Break the loop if P becomes negative (hydrostatic equilibrium)
            if self.func.__name__ == "hydro_equil" and self.state[0] < 0:
                #print(f"Pressure became negative at Radius = {t} m.")
                break

            t = t + self.dt
            self.arr[i] = self.state  # Store the state vector
            i += 1

        return self.arr[:i]  # Return only the filled portion of the array

    def euler(self, t):
        """
        Euler method for a system of ODEs.
        Returns:
            - Updated state vector.
        """
        derivatives = self.func(self.state, t)  # Compute derivatives
        if derivatives is None:
            raise ValueError("Derivatives cannot be None.")
        state_new = self.state + self.dt * derivatives  # Update state
        return state_new


    def runge_kutta_4(self, t):
        """
        Runge-Kutta 4th order method for a system of ODEs.
        Parameters:
            - t: Current time.
        Returns:
            - Updated state vector.
        """
        k1 = self.dt * self.func(t, self.state)
        k2 = self.dt * self.func(t + 0.5 * self.dt, self.state + 0.5 * k1)
        k3 = self.dt * self.func(t + 0.5 * self.dt, self.state + 0.5 * k2)
        k4 = self.dt * self.func(t + self.dt, self.state + k3)
        
        return self.state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

 
    def leapfrog(self, y0, t0=0, tN=100, N=1000):
        h = (tN - t0) / N  # Step size
        t = np.linspace(t0, tN, N+1)
        
        y = np.zeros((2, N+1, 2))  # 2 for pos/vel, last dim for 2D vector (x, y)
        y[:, 0, :] = y0  # Fill initial state
        
        # Initial half-step for velocity
        v_half = y[1, 0] + 0.5 * h * self.func(t[0], y[:, 0])[1]
        
        for i in range(1, N+1):
            # Update position
            y[0, i] = y[0, i-1] + h * v_half
            
            # Update acceleration
            accel = self.func(t[i], [y[0, i], v_half])[1]
            
            # Update velocity
            v_half += h * accel
            y[1, i] = v_half - 0.5 * h * accel
        
        return t, y


# In[1]:


#class hydrostatic_equilibrium(state):
def hydro_equil(state, r):
    """
    ODE system for hydrostatic equilibrium.
    Parameters:
        - state: State vector [P, M, rho0, n, c]
        - r: Radius (m)
    Returns:
        - Derivatives [dPdr, dMdr, drho0dr, dn_dr, dc_dr]
    """
    if len(state) == 5:
        P, M, rho0, n, c = state  # Unpack all 5 elements
    else:
        P, M, rho0 = state
        n = 0.528
        c = 0.00349
        
    if P < 0:
        P = 0  # Ensure pressure is non-negative

    rho = density_rocky(P, rho0, n, c)  # Use n and c from state
    
    if r < 0:
        raise ValueError("Your initial value can't be 0")
        
    dPdr = -G * M * rho / r**2  # Hydrostatic equilibrium equation
    dMdr = 4 * np.pi * r**2 * rho  # Mass continuity equation
    drho0dr = 0  # rho0 is constant (no change with radius)
    dn_dr = 0  # n is constant (no change with radius)
    dc_dr = 0  # c is constant (no change with radius)
    return np.array([dPdr, dMdr, drho0dr, dn_dr, dc_dr])

def density_rocky(P, rho0, n=0.528, c=0.00349):
    """
    Equation of state for rocky cores.
    Parameters:
        - P: Pressure (Pa)
        - rho0: Initial density (kg/m^3)
        - n: Exponent parameter
        - c: Constant parameter
    Returns:
        - Density (kg/m^3)
    """
    
    if P < 0:
        raise ValueError("Pressure cannot be negative.")
    return rho0 + c * (P**n)


# In[ ]:


def solve_pressure(P0, rho0=8300, dr=1e3, r_final=6371e3, tol=1e-8, max_attempts=10, name_integrator="euler", n=0.528, c=0.00349):
    """
    Solve the hydrostatic equilibrium equations for a given central pressure.
    Parameters:
        - P0: Central pressure (Pa)
        - rho0: Initial density (kg/m^3)
        - r0: Initial radius (m)
        - dr: Initial step size (m)
        - r_final: Final radius (m)
        - tol: Tolerance for stopping condition
        - max_attempts: Maximum number of attempts to adjust dr
        - name_integrator: Integration method ("euler" or "runge-kutta")
        - n: Exponent parameter for density_rocky
        - c: Constant parameter for density_rocky
    Returns:
        - results: Array containing [P, M, rho0, n, c] at each step
        - r_outer: Outer radius (m)
    """
    state = [P0, 0, rho0, n, c]  # [P0, M0, rho0, n, c]
    r0 = 1
    for attempt in range(max_attempts):
        try:
            # Calculate the number of steps
            num_steps = int((r_final - r0) / dr) + 1  # Add 1 to ensure coverage

            # Initialize results array with 5 columns for [P, M, rho0, n, c]
            results = np.zeros((num_steps, 5))

            # Initialize the ODE solver
            solver = ODE_solver(hydro_equil, state, dr, results, tol)

            # Integrate the ODE system
            results = solver.integrate(tstop = r_final, integrator=name_integrator)

            return results

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            dr /= 2  # Reduce the step size and try again
            print(f"Reducing step size to dr = {dr}")

    raise Exception("Failed to converge after maximum attempts.")



def cleaning(final_arr, rho0, radius = 6371e3, P0_arr = np.logspace(11, 15, 20), name_integrator='euler', n=0.528, c=0.00349):
    """
    Compute the normalized total mass for a range of central pressures.

    Parameters:
        - final_arr: Array to store results.
        - rho0: Initial density (kg/m^3).
        - mass: Reference mass for normalization (kg).
        - radius: Reference radius for normalization (m).
        - P0_arr: Array of central pressures (Pa) to solve for.
        - name_integrator: Integration method ("euler" or "runge-kutta"). Default is "euler".
        - n: Exponent for density equation. Default is 0.528.
        - c: Constant for density equation. Default is 0.00349.

    Returns:
        - final_mass: Array of normalized total masses (M_total / mass).
    """
    r_0 = 1
    for i in range(len(P0_arr)):
        res = solve_pressure(P0_arr[i], rho0, n=n, c=c, name_integrator=name_integrator)
        res[:,1] = res[:, 1] / (earth_mass/1000)  # Normalize total mass
        r_arr = np.linspace(r_0, radius, num=len(res))
        r_arr = r_arr/(earth_radius/100)
    return res, r_arr


# In[3]:


def calculate_planet_errors(results, planet_data):
    """
    Calculates relative errors for mass and radius for a single planet.

    Parameters:
        results (np.array): Results from the cleaning function (approximate mass values).
        planet_data (dict): Dictionary containing the planet's real mass and radius.

    Returns:
        dict: A dictionary containing relative errors for mass and radius.
    """
    mass_errors = np.zeros(len(results))
    
    for i in range(len(results)):
        mass_errors[i] = nm.rel_err(planet_data['mass'], results[i, 1])
    
    return mass_errors


# In[ ]:




