#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
from my_astro_constants import *

pi = np.pi


# In[30]:


#pt 1
def get_speed_of_light():
    """
        
        No parameters required. Returns the speed of light variable in units of cm/s.
        
        Args:
            N/A
            
        Returns:
            c: var (int) | Units: cm/s | Speed of light constant in units of cm/s.
        
    """
    return c



# In[25]:


# pt 2
def convert_freq_wavelength(value):
    """
         takes "value" based on boolean of return_wavelength will return wavelength or frequency 
                Equation: c = lambda * nu
        Args:
            value: var (int)| Units: hz or cm |input parameter that helps find either frequency or wavelength
            return_wavelength: var (boolean)| Units: None | True returns wavelength, False returns frequency
              c: var (int) | Units: cm | Speed of Light variable 
        
        Returns:
            result: var (int) | Units: Hz/cm | result of either frequency or wavelength
            
    """
    result = 0
    c = get_speed_of_light()
    result = c/value
    if (result.all() >= 0):
        return result
    elif (result >= 0):
        return result
    else:
        raise ValueError("Error Values is out of bounds. Input Parameter must be positive.")   


# In[15]:


#pt 3
def blackbody_function(T_eff, num_points=1000, frequency=False, range=(1e-6, 1e-2)):

    """

        takes T_eff, num_points, frequency, and range to solve spectral radiance of two different blackbody function equations
                Equation 1: 2*h*(c**2)/(wavelength_arr**5) *(1/(np.exp(h*c/(wavelength_arr*k*T_eff)-1)))
                Equation 2: 2*(h*(nu_arr**3))/(c**2) * (1/((np.exp(h*nu_arr/k*T_eff))-1))

        Args:
            T_eff: var (int) | Units: K | Parameter for the blackbody_function it's the effective temperature
            num_points: var (int) | Units: None | Parameter for blackbody_function number of points for the arrays of the result, wavelength, and frequency
            frequency: var (boolean) | Units: None | Parameter for blackbody_function to determine whether given frequency or not (changes the equation type)
            nu_1: var (int) | Units: Hz | Uses range parameter to convert into frequency ranges
            nu_2: var(int) | Units: Hz | Uses range parameter to convert into frequency ranges
            h: var (int) | Units: |
            c: var (int) | Units: |
            range: array (int) | Units: None | Parameter for blackbody_function used to define the wavelength_arr, nu_arr, and nu
            wavelength_arr: array (int) | Units: cm | Array that uses range parameter for wavelength
            nu_arr: array (int)| Units: | Array that uses range parameter for frequency
            spectral_rad/spectral_rad_f: array (int) | Units: cm or Hz| Array that holds the results of blackbody_function
            
        Returns:
           
        
    """
    
    result = np.ones(num_points)
    wavelength_arr = np.linspace(range[0], range[1], num = num_points)
    
    if (not frequency):
        spectral_rad = (2*h*(c**2)/(wavelength_arr**5))*(1/(np.exp(h*c/(wavelength_arr*k*T_eff))-1))
        range_tuple = (wavelength_arr, spectral_rad)
    elif frequency:
        nu_1 = convert_freq_wavelength(range[0], False)
        nu_2 = convert_freq_wavelength(range[1], False)
        nu_arr = np.linspace(convert_freq_wavelength(range[0], False), convert_freq_wavelength(range[1], False), num = num_points)
        
        spectral_rad_f= (2*(h*(nu_arr**3))/(c**2))*(1/(np.exp(h*nu_arr/k*T_eff)-1))
        range_tuple = (nu_arr, spectral_rad_f)

    return range_tuple


# In[16]:


#pt 4
def wiens_law(T_eff, return_type='wavelength'):
    """
    
        Determines the max wavelength or max frequency. 
                return_type = "wavelength" then the case finds the max wavelength.
                return_type = "frequency" then the case finds the max frequency by calling convert_freq_wavelength

        Args:
            T_eff: var (int) | Units: K | Effective temperature, input parameter for weins_law function
            return_type: var (string) | Units: cm or hz | parameter to indicate whether to look for max wavelenght or frequency.
            b: var (int) | Units: cm/k | Wein's law displacement constant
            
        Result:
            wave_max: var (int) | Units: cm | return variable to output max wavelength
            freq_max: var (int) | Units: HZ | return variable to output max frequency
        
    """
    if (return_type.casefold() == 'wavelength'):
            wave_max = b/T_eff
            return wave_max
        
    elif (return_type.casefold() == 'frequency'):
            wave_max = b/T_eff
            freq_max = c/wave_max
            return freq_max
    else:
        raise SyntaxError("Error: Incorrect please type in either 'wavelength' or 'frequency'")

           
    


# In[17]:


def surface_area(R, units = 'solar', return_units = 'solar'):
    """ 
        Determines the surface area given R (radius), units, and the proper units SA = 4*pi*(R**2)

        Args:
            R: var (int) | Units: cm or solar radius | Radius, an input parameter for surface_area
            units: var (string) | Units: None | units, an input parameter for the units of R & T_eff
            return_units: var (string) | Units: None | an input parameter to list the return_units of R&T_eff
            
        Returns:
            result: var (int) | Units: cgs/ solar| SA, output parameter for surface_area
    """
    if R > 0:
        if ((units.casefold() == 'cgs') and (return_units.casefold() == 'cgs')) or ((units.casefold() == 'solar') and (return_units.casefold() == 'solar')):
            SA = 4*pi*(R**2)
        elif (return_units.casefold() == 'cgs'):
            SA = 4*pi*((R*solar_r)**2)
        elif (return_units.casefold() == 'solar'):
            SA = 4*pi*((R/7E10)**2)
        else:
            raise TypeError("CGS/Solar not typed.")
             
    else:
        raise ValueError("Radius cannot be negative.")
    
    return SA

        
    
    


# In[18]:


#pt 2
def calculate_flux(T_eff):
    """
        Determines the flux given the effective temperature. Uses the equation F = stefan-boltzman*(T_eff**4)
        
        Args:
            T_eff: var (int) | Units: K | input parameter (Effective temperature)
            sigma: var (int) | Units: erg/(cm^2*s*K^4) | constant (Stefan-Boltzman constant)
            
        Returns:
              flux: var (int) | Units: erg/cm^2*s | Flux, Return value
    """
    if T_eff >= 0:
        flux = sigma*(T_eff**4)
    else:
        raise ValueError("Temperature cannot be negative.")
    
    return flux


# In[19]:


#pt 3
def calculate_lum(R, T_eff, *, units='solar', return_units='solar'):
    """
        Determines the luminosity given the radius and effective temperature. Uses the equation L = (sigma*(T_eff**4))*4*pi*(R**2)

        Args:
            R: var (int) | Units: cm | Radius, an input parameter for calculate_temp
            T_eff: var (int) | Units: K | T_eff, an input parameter for calculat_lum
            units: var (string) | Units: solar/cgs | an input parameter for the units of R & T_eff
            return_units: var (string) | Units: solar/cgs | an input parameter to list the return_units of R&T_eff
            
        Returns:
            L: var (int) | Units: J/s or W | Luminosity, Output parameter the value we are trying to find. 
        
    """
    if (T_eff >= 0) or (R >= 0):
        if ((units.casefold() == 'cgs') and (return_units.casefold() == 'cgs')) or ((units.casefold() == 'solar') and (return_units.casefold() == 'solar')):
            L = calculate_flux(T_eff)* surface_area(R)
        elif (return_units.casefold() == 'cgs'):
            L = calculate_flux(T_eff)* surface_area(R,return_units = return_units)
        elif (return_units.casefold() == 'solar'):
            L = calculate_flux(T_eff)* surface_area(R,return_units = return_units)
        else:
            raise NameError("CGS/Solar not typed.")
    else:
        raise ValueError(f"Temperature or Radius cannot be negative")
    return L



# In[20]:


def calculate_temp(L, R, units = 'solar', return_units = 'solar'):
    """
        Determines the effective temperature given the luminosity and radius. Uses the equation T = (L/(4*pi*sigma*(R**2)))**(1/4)
        
        Args:
            L: var (int) | Units: J/s or solar | Luminosity, an input parameter for calculate_temp
            R: var (int) | Units: cm or solar | Radius, an input parameter for calculate_temp
            
        Returns:
            T_eff: var (int) | Units: K | Effective Temperature, an output parameter the result of the equation
    
    """

    if (R > 0) and (L > 0):
        if ((units.casefold() == 'cgs') and (return_units.casefold() == 'cgs')) or ((units.casefold() == 'solar') and (return_units.casefold() == 'solar')):
            T_eff = (L/(4*pi*sigma*(R**2)))**(1/4)
        elif (return_units.casefold() == 'cgs'):
            T_eff = (L/(4*pi*sigma*(R**2)))**(1/4)
        elif (return_units.casefold() == 'solar'):
            T_eff = ((L*1E5)/(4*pi*sigma*((R/7E10)**2)))**(1/4)
        else:
            raise NameError("CGS/Solar not typed.")
    else:
        raise ValueError("Radius or Luminosity cannot be negative.")
    return T_eff




# In[21]:


def calculate_radius(L, T_eff, units = 'solar', reutn_units = 'solar'):
    """
        Determines the radius given the luminosity and radius. Uses the equation R = sqrt(L/4*pi*sigma*(T_eff**4))
        
        Args:
            L: var (int) | Units: J/s or solar |  Luminosity, an input parameter for calculate_radius
            T_eff: var (int) | Units: K | T_eff, an input parameter for calculat_radius

        Returns:
            R: var (int) | Units: cm/ solar | Radius, an output parameter for calculate_radius
    """
    if (T_eff >= 0) and (L >= 0):
        R = np.sqrt(L/(4*pi*sigma*(T_eff**4)))
        
    else:
        raise ValueError("Luminosity or Temperature cannot be negative.")
        
    return R
    


# In[22]:


#plotting function
#fig, ax = plt.subplots()
def plot_func(x,y, xmin = 0, xmax = 0, xlabel = None, ymin= 0, ymax = 1e8, ylabel = None, flag_y = 0, flag_x = 0,
              label = "", Title = None, save_fig = False, figname = None):
    
    ax = plt.gca()
    
    ax.ticklabel_format(axis='both', style='scientific')
    ax.plot(x,y, label = label)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    #ax.ticklabel_format(axis='both', style='sci', scilimits=(4,4))
    
    if flag_y == 1:
        plt.yscale("log")
    elif flag_y == 2:
        plt.yscale("symlog")
    else:
        plt.yscale("linear")

    if flag_x == 1:
        plt.xscale("log")
    else:
        plt.yscale("linear")
        
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if Title:
        plt.title(Title)

    ##Show plot in ipynb
    #plt.show()

    if save_fig:
        if figname:
            plt.savefig(figname)
        else:
            print('No figname provided, figure saved as "test_plot.png"')
            plt.savefig('test_plot.png')


# In[ ]:




