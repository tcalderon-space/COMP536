#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np


# In[13]:


class Integrators:
    # Define the rule constants as class attributes
    RECTANGLE_RULE = "1 Rectangle"
    TRAPEZOIDAL_RULE = "2 Trapezoidal"
    SIMPSON_RULE = "3 Simpson"

    def __init__(self, a, b, n):
        """
        Initializes the Integrators class with integration limits and the number of steps.
    
        Parameters:
            a (float): The lower limit of integration.
            b (float): The upper limit of integration.
            n (int): The number of steps (subintervals) for integration.
    
        Attributes:
            integ_limits (tuple): The integration limits (a, b).
            n_step (int): The number of steps.
            delta_x (float): The width of each subinterval.
            methods (dict): A mapping of integration methods to their respective functions.
        """
        self.integ_limits = (a, b)
        self.n_step = n
        self.delta_x = (b - a) / n

        # Map aka create a path of assignment to methods
        self.methods = {
            self.RECTANGLE_RULE: self.rectangle_rule,
            self.TRAPEZOIDAL_RULE: self.trapezoidal_rule,
            self.SIMPSON_RULE: self.simpsons_rule
        }

    def rectangle_rule(self, f):
        """
        Approximates the integral of a function using the rectangle rule.
    
        The rectangle rule approximates the integral as:
    
        Parameters:
            f (callable): The function to integrate.
    
        Returns:
            float: The approximate integral value.
        """
        a, b = self.integ_limits
        total = 0

        for i in range(self.n_step):  # Iterate over all steps
            x = a + i * self.delta_x
            total += f(x)

        return total * self.delta_x

    def trapezoidal_rule(self, f):
        """
        Approximates the integral of a function using the trapezoidal rule.
    
    
        Parameters:
            f (callable): The function to integrate.
    
        Returns:
            float: The approximate integral value.
        """
        a, b = self.integ_limits
        total = 0.5 * (f(a) + f(b))  # Add the endpoints
    
        for i in range(1, self.n_step):  # Iterate over intermediate points
            x = a + i * self.delta_x
            total += f(x)
    
        return total * self.delta_x

    def simpsons_rule(self, f):
        """
        Approximates the integral of a function using Simpson's rule.
    
        Parameters:
            f (callable): The function to integrate.
    
        Returns:
            float: The approximate integral value.
    
        Raises:
            ValueError: If the number of steps `n` is not even.
        """
        if self.n_step % 2 != 0:
            raise ValueError("n must be even to use Simpson's rule.")

        a, b = self.integ_limits
        h = self.delta_x
        total = f(a) + f(b)  # Add the endpoints

        for i in range(1, self.n_step):  # Iterate over intermediate points
            x = a + i * h
            if i % 2 == 0:
                total += 2 * f(x)  # Even-indexed points
            else:
                total += 4 * f(x)  # Odd-indexed points

        return total * h / 3  # Return a single scalar value

    def integrate(self, method, f):
        """
        Integrates a function using the specified numerical integration method.
    
        Parameters:
            method (str): The integration method to use. Must be one of:
                - `RECTANGLE_RULE`
                - `TRAPEZOIDAL_RULE`
                - `SIMPSON_RULE`
            f (callable): The function to integrate.
    
        Returns:
            float: The approximate integral value.
    
        Raises:
            ValueError: If the specified method is not recognized.
        """
        if method not in self.methods:
            raise ValueError(f"Unknown integration method: {method}")
        return self.methods[method](f)


# In[14]:


#absolute error
def abs_err(I_real, I_approx):
    """
    Computes the absolute error between a real value and an approximate value.

    Parameters:
        I_real (float): The real (exact) value.
        I_approx (float): The approximate value.

    Returns:
        float: The absolute error between the real and approximate values.
    """
    res = np.absolute(I_real - I_approx)
    return res
    
#relative error
def rel_err(I_real, I_approx):
    """
    Computes the relative error between a real value and an approximate value.

    Parameters:
        I_real (float): The real (exact) value.
        I_approx (float): The approximate value.

    Returns:
         float: The relative error between the real and approximate values.
    """
        
    res = np.absolute((I_real - I_approx)/I_real)
    return res


# In[15]:


#root finders
class RootFinders:
    BISECTION = "1 Bisection"
    NEWTON = "2 Newton"
    SECANT = "3 Secant"

    def __init__(self, func, a, b, tol, max_iter=1000):
        """
        Initialize the RootFinders class.

        Parameters:
            func (callable): The function for which to find the root.
            a (float): Lower bound of the interval (for bisection/secant).
            b (float): Upper bound of the interval (for bisection/secant).
            tol (float): Tolerance for convergence.
            max_iter (int): Maximum number of iterations.
        """
        self.func = func
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter

        # Mapping of methods
        self.method = {
            self.BISECTION: self.bisection,
            self.NEWTON: self.newton,
            self.SECANT: self.secant
        }

    def bisection(self):
        """
        Bisection method to find the root of a function within a given interval [a, b].
    
        The method repeatedly divides the interval in half and selects the subinterval
        that contains the root, based on the Intermediate Value Theorem.
    
        Returns:
            tuple: A tuple containing:
                - The approximate root (float).
                - The number of iterations (int) required to converge.
    
        Raises:
            ValueError: If the function values at `a` and `b` have the same sign,
                       indicating that the interval may not contain a root.
        """
        a, b = self.a, self.b

        iter_count = 0
        if self.func(a) == 0:
            return a, 0
        elif self.func(b) == 0:
            return b, 0
            
        if (self.func(a) * self.func(b) >= 0):
            raise ValueError (f"You have not assumed the right a and b\n")
            
        while (b - a) / 2 > self.tol and iter_count < self.max_iter:
            iter_count += 1
            c = (a + b) / 2
            
            if self.func(c) == 0:
                return c
            elif self.func(a) * self.func(c) < 0:
                b = c
            else:
                a = c
            
        return c, iter_count

    def newton(self):

        """
        Newton's method to find the root of a function using an initial guess.
    
        The method uses the function's derivative to iteratively approximate the root.
        If the derivative is too close to zero, the method falls back to the bisection method.
    
        Returns:
            tuple: A tuple containing:
                - The approximate root (float).
                - The number of iterations (int) required to converge.
    
        Raises:
            ValueError: If the derivative is too close to zero or if the method fails to converge
                       within the maximum number of iterations.
        """
        x = self.a  # Initial guess
        iter_count = 0
        max_iter = 1000  # Increased maximum number of iterations
        for _ in range(max_iter):
            fx = self.func(x)
            if abs(fx) < self.tol:  # Check for convergence
                return x, iter_count
            derivative = self.derivative(x)
            if abs(derivative) < 1e-10:  # Derivative is too close to zero
                # Fall back to bisection
                print("Derivative is too close to zero. Falling back to bisection method.")
                return self.bisection()
            x = x - fx / derivative
            iter_count += 1
        # If the loop completes without converging, raise an error
        raise ValueError("Newton's method did not converge within the maximum number of iterations.")
        
    def secant(self):
        """
        Secant method to find the root of a function using two initial guesses.
    
        The method approximates the root by linearly interpolating between two points
        on the function and iteratively updating the guesses.
    
        Returns:
            tuple: A tuple containing:
                - The approximate root (float).
                - The number of iterations (int) required to converge.
    
        Raises:
            ValueError: If the denominator in the secant formula is too close to zero,
                       indicating potential numerical instability.
        """
        x0, x1 = self.a, self.b
        iter_count = 0
        while abs(self.func(x1)) > self.tol:
            iter_count += 1
            denominator = self.func(x1) - self.func(x0)
            if abs(denominator) < 1e-10:  # Denominator is too close to zero
                # Use a small fixed step size
                step = 1e-5 if self.func(x1) > 0 else -1e-5
            else:
                step = self.func(x1) * (x1 - x0) / denominator
                x2 = x1 - step
                x0, x1 = x1, x2
        
        return x1, iter_count

    def derivative(self, x, h=1e-5):
        """
        Numerical derivative of the function using the central difference method.
    
    
        Parameters:
            x (float): The point at which to compute the derivative.
            h (float, optional): The step size for the central difference method. Default is 1e-5.
    
        Returns:
            float: The numerical derivative of the function at `x`.
    
        Raises:
            ValueError: If `h` is zero, as this would result in division by zero.

        """
        return (self.func(x + h) - self.func(x - h)) / (2 * h)

    def find(self, method):
        """
        Find the root using the specified method.

        Parameters:
            method (str): Method to use (BISECTION, NEWTON, or SECANT).

        Returns:
            float: The approximate root.
        """
        if method in self.method:
            return self.method[method]()
        else:
            raise ValueError(f"Unknown root-finding method: {method}")


# In[ ]:




