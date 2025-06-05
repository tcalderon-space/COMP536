#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import my_numerical_methods as nm
import my_ode_solvers as os
import my_star_functions as sf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from my_astro_constants import *

import numba 
from numba import jit


# In[5]:


def n_body(m_j, r_j, r_i, epsilon=1e-3):
    """
    Calculate acceleration due to gravitational force
    
    Args:
        m_j: Mass of the other body
        r_j: Position of the other body
        r_i: Position of the current body
        epsilon: Softening parameter
        
    Returns:
        Acceleration vector
    """
    r_ij = r_j - r_i
    distance = np.sqrt(np.sum(r_ij**2) + epsilon**2)
    return g * m_j * r_ij / distance**3

def distance(x_j=0, x_i=0, y_j=0, y_i=0, z_j=0, z_i=0):
    """
    Computes the Euclidean distance between two points in 3D space.

    Parameters
    ----------
    x_j, x_i : float, optional
        x-coordinates of the two points.
    y_j, y_i : float, optional
        y-coordinates of the two points.
    z_j, z_i : float, optional
        z-coordinates of the two points.

    Returns
    -------
    float
        The Euclidean distance between the two 3D points.
    """
    return np.sqrt((x_j - x_i)**2 + (y_j - y_i)**2 + (z_j - z_i)**2)


def energy_calc(nrg_arr, pos_arr, velo_arr, m1, m2):
    """
    Calculates and stores the kinetic, potential, and total energy
    for each particle in a system.

    Parameters
    ----------
    nrg_arr : ndarray
        A (3, N) array where N is the number of particles. The function
        fills in the kinetic, potential, and total energy in rows 0, 1, and 2.
    pos_arr : ndarray
        An (N, 3) array of position vectors for each particle.
    velo_arr : ndarray
        An (N, 3) array of velocity vectors for each particle.

    Returns
    -------
    nrg_arr : ndarray
        Updated energy array with kinetic, potential, and total energy
        values for each particle.
    """
    for i in range(len(pos_arr)):
        r_1 = np.linalg.norm(pos_arr[i])
        v = velo_arr[i]

        KE = kinetic_nrg(m1, v)
        PE = potential_nrg(m1, m2, r_1)
        TE = total_nrg(KE, PE)

        nrg_arr[0, i] = KE
        nrg_arr[1, i] = PE
        nrg_arr[2, i] = TE

    return nrg_arr

#plummer density profile
def plummer_density(M, r, a=1.0):
    """
    Calculate density at radius r for a Plummer sphere
    
    Args:
        M: Total mass of the system
        r: Radius from center
        a: Scale radius (default 1 pc)
        
    Returns:
        Density at radius r
    """
    return (3*M/(4*np.pi*(a**3))) * ((1 + (r**2/a**2))**(-5/2))

def kinetic_nrg(m, v):
    """
    Computes the kinetic energy of a particle.

    Parameters
    ----------
    m : float
        Mass of the particle.
    v : ndarray
        Velocity vector of the particle.

    Returns
    -------
    float
        Kinetic energy: (1/2) * m * v^2
    """
    return 0.5 * m * np.dot(v, v)

def potential_nrg(m1, m2, r):
    """
    Computes the gravitational potential energy between two masses.

    Parameters
    ----------
    m1 : float
        First mass (e.g., central or larger mass).
    m2 : float
        Second mass (e.g., orbiting or smaller mass).
    r : float
        Distance between the two masses.

    Returns
    -------
    float
        Gravitational potential energy. Returns 0 if r is too small to avoid singularity.
    """
    if r < 1e-5:
        return 0
    return -g * m1 * m2 / r

def total_nrg(e_k, e_g):
    """
    Computes the total energy of a particle.

    Parameters
    ----------
    e_k : float
        Kinetic energy.
    e_g : float
        Potential energy.

    Returns
    -------
    float
        Total energy: kinetic + potential.
    """
    return e_k + e_g

def velocity_circ(M, r):
    """
    Computes the circular velocity for a particle in orbit.

    Parameters
    ----------
    M : float
        Mass of the central object.
    r : float
        Distance from the center.

    Returns
    -------
    float
        Circular orbital velocity.
    """
    return np.sqrt(g * M / r)

def period(r, v_c):
    """
    Computes the orbital period of a particle in circular motion.

    Parameters
    ----------
    r : float
        Radius of the orbit.
    v_c : float
        Circular velocity.

    Returns
    -------
    float
        Orbital period.
    """
    return (2 * np.pi * r) / v_c
    
def radial_dis(X, a=1.0):
    """
    Inverse transform sampling for the Plummer model radial distribution.

    Parameters
    ----------
    X : float
        Uniform random variable in the interval [0, 1].
    a : float, optional
        Scale radius of the Plummer sphere (default is 1.0).

    Returns
    -------
    float
        Sampled radius r based on the Plummer profile.
    """
    return a * np.sqrt(X**(-2/3) - 1)
    
def polar_angle(U):
    """
    Computes the polar angle θ from a uniform random variable.

    Parameters
    ----------
    U : float
        Uniform random variable in [0, 1].

    Returns
    -------
    float
        Polar angle in radians.
    """
    return np.arccos(1 - 2 * U)

def azimuthal_angle(V):
    """
    Computes the azimuthal angle φ from a uniform random variable.

    Parameters
    ----------
    V : float
        Uniform random variable in [0, 1].

    Returns
    -------
    float
        Azimuthal angle in radians.
    """
    return 2 * np.pi * V


# In[8]:


def velocity_circular(M, r, a=1.0):
    """
    Circular velocity for Plummer potential
    
    Args:
        M: Total mass
        r: Radius
        a: Scale radius
        
    Returns:
        Circular velocity at r
    """
    return np.sqrt(g * M * r**2 / (r**2 + a**2)**(3/2))

def velocity_dispersion(M, r, a=1.0):
    """
    Velocity dispersion for Plummer sphere
    
    Args:
        M: Total mass
        r: Radius
        a: Scale radius
        
    Returns:
        Radial velocity dispersion σ(r)
    """
    return np.sqrt(g * M / (6 * np.sqrt(r**2 + a**2)))
    
#step 4
def convert_spher_to_cart(r, theta, phi):
    """
    Convert spherical to Cartesian coordinates
    
    Args:
        r: Radius
        theta: Polar angle [0,π]
        phi: Azimuthal angle [0,2π]
        
    Returns:
        (x, y, z) Cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def sample_velocities(pos, masses, M_total, a=1.0):
    """
    Sample physically consistent velocities for Plummer distributio
    usingconvert_spher_to_cart() function
    
    Args:
        pos: (n,3) array of positions
        masses: (n,) array of particle masses
        M_total: Total system mass
        a: Scale radius
        
    Returns:
        (n,3) array of velocities
    """
    print(pos)
    #distance(x_j = 0, x_i = 0, y_j = 0, y_i = 0, z_j = 0, z_i = 0)
    #radii = distance(pos[0], pos[1], pos[3])
    radii = np.sqrt(np.sum(pos**2, axis=1))
    velocities = np.zeros_like(pos)
    
    for i in range(len(pos)):
        r = radii[i]
        if r == 0:  # Handle center particle
            continue
            
        # 1. Get velocity dispersion at this radius
        sigma = velocity_dispersion(M_total, r, a)
        
        # 2. Sample velocity components in spherical coordinates
        v_r = np.random.normal(0, sigma)
        v_theta = np.random.normal(0, sigma)
        v_azimuthal = np.random.normal(0, sigma)
        
        # 3. Get angular coordinates from position
        theta = np.arccos(pos[i,2]/r)  # Polar angle
        azimuthal = np.arctan2(pos[i,1], pos[i,0])  # Azimuthal angle
        
        # 4. Convert velocity to Cartesian using your function
        velocities[i] = convert_spher_to_cart(v_r, theta, azimuthal)  # Radial component
        velocities[i] += convert_spher_to_cart(v_theta, theta + np.pi/2, azimuthal)  # Polar component
        velocities[i] += convert_spher_to_cart(v_azimuthal, np.pi/2, azimuthal + np.pi/2)  # Azimuthal component
    
    return velocities


# In[9]:


#implementing everything as a plummer sphere equation
def sample_plummer(n, M_total=1.0, a=1.0):
    """
    Sample positions, velocities, and masses for a Plummer sphere
    using all your dedicated transform functions.
    
    Args:
        n: Number of particles
        M_total: Total mass of the system
        a: Scale radius
        
    Returns:
        positions, velocities, masses
    """
    # 5. Sample masses (0.1-10 M☉ uniform)
    masses = np.random.uniform(0.1, 10.0, n)
    masses = masses * (M_total/np.sum(masses))  # Normalize to exact total mass

    U = np.random.uniform(0, 1, n)
    V = np.random.uniform(0, 1, n)
    
    # 2-4. Sample spherical coordinates using your functions
    r = radial_dis(U, a)
    theta = polar_angle(U)    
    phi = azimuthal_angle(V)  

    # 4. Convert to Cartesian positions
    positions = np.array([convert_spher_to_cart(r[i], theta[i], phi[i]) 
                         for i in range(n)])

    # 6. Velocity sampling (placeholder)
    velocities = sample_velocities(positions, masses, M_total, a)
    
    return positions, velocities, masses


# In[6]:


def plot_plummer_sphere(pos, masses, save_path=None):
    """
    Visualize the sampled Plummer sphere with mass-dependent coloring
    
    Args:
        pos: (n,3) array of positions
        masses: (n,) array of masses
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize masses for coloring [0.1,10] -> [0,1]
    norm_masses = (masses - 0.1) / (10.0 - 0.1)
    colors = plt.cm.viridis(norm_masses)  # Color by mass
    
    # Plot with size proportional to mass^(1/3) (approximate volume scaling)
    sc = ax.scatter(
        pos[:,0], pos[:,1], pos[:,2],
        c=colors,
        s=10 * (masses/0.1)**(1/3),  # Scale marker size
        alpha=0.6,
        edgecolors='none'
    )
    
    # Add colorbar
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=plt.Normalize(0.1, 10), cmap='viridis'),
        ax=ax,
        label='Mass (M☉)'
    )
    cbar.set_alpha(1)
    
    # Labels and title
    ax.set_xlabel('X [pc]')
    ax.set_ylabel('Y [pc]')
    ax.set_zlabel('Z [pc]')
    ax.set_title(f'Plummer Sphere Sampling (N={len(pos)})')
    
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


# In[ ]:




