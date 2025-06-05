#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from my_astro_constants import *


# In[2]:


#FUNCS - Hubbles Constant, distance_mod, fit_linear_model
def hubbles_constant(b, var_b):
    """
    Computes the Hubble constant H₀ and its uncertainty from the y-intercept (b₀) of a linear fit
    to log(distance) vs. log(redshift) under the assumption that log(z) = log(H₀) + log(d/c).

    Parameters:
    -----------
    b : float
        The fitted intercept (b₀) from the log-log linear regression.

    var_b : float
        The variance (not standard error) of the intercept term b₀, typically from the covariance matrix.

    Returns:
    --------
    H_0 : float
        Estimated value of the Hubble constant in km/s/Mpc.

    se_H0 : float
        Standard error of the estimated Hubble constant, computed via error propagation.

    Notes:
    ------
    Assumes the speed of light `c = 3e5` km/s. The formula used is:
        H₀ = (c / 1e5) * 10^(-b)
        se_H₀ = H₀ * ln(10) * sqrt(var_b)
    """
    H_0 = (c / 1e5) * 10 ** (-b)
    se_H0 = H_0 * np.log(10) * np.sqrt(var_b)
    return H_0, se_H0
    

def distance_modulous(mu, units = 'parsecs'):
    """
    Converts distance modulus to physical distance.

    Parameters:
    -----------
    mu : float or array-like
        Distance modulus values (μ), defined as μ = 5 * log10(d / 10 pc),
        where d is the distance in parsecs.

    units : str, optional (default = 'parsecs')
        Desired output units. Options:
        - 'parsecs': returns distance in parsecs
        - 'megaparsecs': returns distance in megaparsecs

    Returns:
    --------
    d : float or array-like
        Distance corresponding to the input distance modulus, in the specified units.

    Raises:
    -------
    ValueError
        If the units argument is not 'parsecs' or 'megaparsecs'.
    """
    # Convert mu to distance in parsecs
    d_pc = 10 ** ((mu + 5) / 5)

    if units == 'megaparsecs':
        return d_pc / 1e6
    elif units == 'parsecs':
        return d_pc
    else:
        raise ValueError("Unsupported unit: use 'parsecs' or 'megaparsecs'")

def fit_linear_model(X,Y):
    """
    Fits a linear model of the form Y = X * beta + error using ordinary least squares.

    Parameters:
    -----------
    X : numpy.ndarray of shape (N, k)
        The design matrix where each row corresponds to an observation and each column to a model feature.
        For simple linear regression (y = mx + b), X should have two columns:
        - first column: input values (e.g., log(z))
        - second column: ones (to represent the intercept term b₀)

    Y : numpy.ndarray of shape (N,)
        The observed output values corresponding to each row in X (e.g., log(distance)).

    Returns:
    --------
    beta : numpy.ndarray of shape (k,)
        The fitted model parameters. For linear regression, beta[0] is the slope (m) and beta[1] is the intercept (b₀).

    covariance_matrix : numpy.ndarray of shape (k, k)
        The estimated covariance matrix of the fitted parameters. Useful for estimating parameter uncertainties.
    """
    N, k = X.shape  # N = number of observations, k = number of parameters
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y

    #predicted y
    Y_pred = X@beta
    
    #residuals
    R = Y - Y_pred

    # Estimate variance of residuals
    variance = (R.T @ R) / (N - k)

    # Covariance matrix of beta
    covariance_matrix = variance * np.linalg.inv(X.T @ X)
    
    return beta, covariance_matrix

