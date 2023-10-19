import numpy as np
from kondap_baby_od import get_orbital_elements

def f_func (x, r2, r2_dot, a, n, tau_i):
    """
    Calculate the initial value of the closed value iteration for delta E.

    Args:
    - x (float): Current value of x (used for iteration).
    - r2 (np.ndarray): Position vector of the object at the second observation.
    - r2_dot (np.ndarray): Velocity vector of the object at the second observation.
    - a (float): Semi-major axis.
    - n (float): Mean motion.
    - tau_i (float): Gaussian time difference between observations.

    Returns:
    - float: Value of the equation f(x) for the given parameters.
    """
    term1 = x-(1-np.linalg.norm (r2)/a)*np.sin (x)
    term2 = (np.dot (r2, r2_dot)/(n*a**2) * (1-np.cos (x)))-n*tau_i
    return term1+term2

def f_prime_func (x, r2, r2_dot, a, n):
    """
    Calculate the derivative value to improve the estimate of delta E.

    Args:
    - x (float): Current value of x (used for iteration).
    - r2 (np.ndarray): Position vector of the object at the second observation.
    - r2_dot (np.ndarray): Velocity vector of the object at the second observation.
    - a (float): Semi-major axis.
    - n (float): Mean motion.

    Returns:
    - float: Value of the derivative of the equation f(x) for the given parameters.
    """
    term1 = 1-(1-np.linalg.norm (r2)/a)*np.cos (x)
    term2 = np.dot (r2, r2_dot)/(n*a**2)*np.sin (x)
    return term1+term2

def iteration (initial_val, tolerance, r2, r2_dot, a, n, tau_i):
    """
    Calculate the value of delta E for closed-form functions using iteration.

    Args:
    - initial_val (float): Initial value for delta E.
    - tolerance (float): Tolerance for convergence.
    - r2 (np.ndarray): Position vector of the object at the second observation.
    - r2_dot (np.ndarray): Velocity vector of the object at the second observation.
    - a (float): Semi-major axis.
    - n (float): Mean motion.
    - tau_i (float): Gaussian time difference between observations.

    Returns:
    - float: The calculated value of delta E.
    """
    de_curr = initial_val
    diff = tolerance+1
    while diff>tolerance:
        de_new = de_curr-f_func (de_curr, r2, r2_dot, a, n, tau_i)/f_prime_func (de_curr, r2, r2_dot, a, n)
        diff = np.abs (de_new-de_curr)
        de_curr = de_new
    return de_curr

def closed (r2, r2_dot, tau_1, tau_3, tolerance):
    """
    Determine the closed-form values of f1, g1, f3, and g3.

    Args:
    - r2 (np.ndarray): Position vector of the object at the second observation.
    - r2_dot (np.ndarray): Velocity vector of the object at the second observation.
    - tau_1 (float): Gaussian time for the first observation.
    - tau_3 (float): Gaussian time for the third observation.
    - tolerance (float): Tolerance for convergence.

    Returns:
    - tuple: A tuple containing the following values:
      - (float) f1
      - (float) g1
      - (float) f3
      - (float) g3
    """
    a, e, _, _, _, _, _, n, _, _ = get_orbital_elements (r2, r2_dot)
    val = np.dot (r2, r2_dot)/(n*a**2)
    if (e<0.1):
        x_0_1 = n*tau_1
        x_0_3 = n*tau_3
    else:
        sign_1 = np.sign (val*np.cos (n*tau_1-val)+(1-np.linalg.norm (r2)/a)*np.sin (n*tau_1-val))
        sign_3 = np.sign (val*np.cos (n*tau_3-val)+(1-np.linalg.norm (r2)/a)*np.sin (n*tau_3-val))
        x_0_1 = n*tau_1+sign_1*(0.85*e-val)
        x_0_3 = n*tau_3 + sign_3*(0.85*e-val)

    delta_e_1 = iteration (x_0_1, tolerance, r2, r2_dot, a, n, tau_1)
    delta_e_3 = iteration (x_0_3, tolerance, r2, r2_dot, a, n, tau_3)

    f_1 = 1-a/np.linalg.norm (r2)*(1-np.cos (delta_e_1))
    f_3 = 1-a/np.linalg.norm(r2)*(1-np.cos (delta_e_3))

    g_1 = tau_1+1/n*(np.sin (delta_e_1)-delta_e_1)
    g_3 = tau_3+1/n*(np.sin (delta_e_3)-delta_e_3)
    return f_1, g_1, f_3, g_3

def fg_func (func_type, r2, r2_dot, tau_1, tau_3, mu=1, tolerance = 1e-12):
    """
    Redirect the computation of f1, g1, f3, g3 to the appropriate type of function.

    Args:
    - func_type (str): Type of fg function calculation ('three', 'four', or 'closed').
    - r2 (np.ndarray): Position vector of the object at the second observation.
    - r2_dot (np.ndarray): Velocity vector of the object at the second observation.
    - tau_1 (float): Gaussian time for the first observation.
    - tau_3 (float): Gaussian time for the third observation.

    Kwargs:
    - mu (float): Gravitational parameter (default: 1).
    - tolerance (float): Tolerance for convergence (default: 1e-12).

    Returns:
    - tuple: A tuple containing the values of f1, g1, f3, and g3 based on the specified function type.
    """
    if (func_type == 'closed'):
        f_1, g_1, f_3, g_3 = closed (r2, r2_dot, tau_1, tau_3, tolerance)
    else:
        # determines third and fourth order Taylor series
        u = mu/(np.linalg.norm (r2)**3)
        z = np.dot (r2, r2_dot)/(np.linalg.norm (r2)**2)
        q = np.dot (r2_dot, r2_dot)/(np.linalg.norm (r2)**2)-u
        f_1 = 1-1/2*u*tau_1**2+1/2*u*z*tau_1**3
        g_1 = tau_1-1/6*u*tau_1**3

        f_3 = 1-1/2*u*tau_3**2+1/2*u*z*tau_3**3
        g_3 = tau_3-1/6*u*tau_3**3
        
        if ('four' in func_type):
            f_1+=1/24*(3*u*q-15*u*z**2+u**2)*tau_1**4
            g_1+=1/4*u*z*tau_1**4
            f_3+=1/24*(3*u*q-15*u*z**2+u**2)*tau_3**4
            g_3+=1/4*u*z*tau_3**4
    return f_1, g_1, f_3, g_3