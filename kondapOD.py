# Kavish Kondap, SSP 2023
# Method of Gauss Script

import numpy as np
from numpy import sin, cos, radians
from kondap_baby_od import get_orbital_elements
from kondap_fg_functions import fg_func
import math

def sexagesimal_to_decimal (ra, dec):
    """
    **Convert sexagesimal coordinates to decimal radians.**

    Args:
    - ra (str): Right Ascension in the format 'hh:mm:ss'.
    - dec (str): Declination in the format 'dd:mm:ss'.

    Returns:
    - tuple: A tuple containing the following:
      - (float) Right ascension in radians.
      - (float) Declination in radians.
    """
    ra_hours, ra_minutes, ra_seconds = [float (i) for i in ra.split (':')]
    ra_minutes = np.copysign (ra_minutes, ra_hours)
    ra_seconds = np.copysign (ra_seconds, ra_hours)
    ra_decimal = (ra_hours+ra_minutes/60+ra_seconds/3600)/24*360
    dec_degrees, dec_minutes, dec_seconds = [float (i) for i in dec.split (':')]
    dec_minutes = np.copysign (dec_minutes, dec_degrees)
    dec_seconds = np.copysign (dec_seconds, dec_degrees)
    dec_decimal = dec_degrees+dec_minutes/60+dec_seconds/3600
    return radians (ra_decimal), radians (dec_decimal)

def get_gaussian_time (julian_times, k):
    """
    **Convert Julian times to Gaussian time.**

    Args:
    - julian_times (list): List of Julian times.
    - k (float): Gaussian gravitational constant.

    Returns:
    - tuple: A tuple containing the following:
      - (float) tau_1
      - (float) tau_3
      - (float) tau
    """
    tau_3 = k*(julian_times[2]-julian_times[1])
    tau_1 = k*(julian_times[0]-julian_times[1])
    return tau_1, tau_3, tau_3-tau_1

def get_julian_times (times):
    """
    **Calculate Julian dates for a list of observation times.**

    Args:
    - times (list): List of observation times as tuples (year, month, day, hour, minute, second).

    Returns:
    - list: List of Julian times.
    """
    julian_times = []
    for time in times:
        j_0 = 367*time[0]-int (7/4*(time[0]+int ((time[1]+9)/12))) + int (275/9*time[1])+time[2]+1721013.5
        julian_times.append (j_0+(time[3]+time[4]/60+time[5]/3600)/24)
    return julian_times

def sel (rho_hats, R_vecs, tau_1, tau_3, tau, mu, D_0, D2s):
    """
    **Solve the Scalar Equation of Lagrange to find initial values for r_2.**

    Args:
    - rho_hats (list of np.ndarray): List of unit vectors in the direction of the observations.
    - R_vecs (list of np.ndarray): List of position vectors of the object.
    - tau_1 (float): Gaussian time for the first observation.
    - tau_3 (float): Gaussian time for the third observation.
    - tau (float): Time difference between the first and third observations.
    - mu (float): Gravitational parameter.
    - D_0 (float): Scalar product of rho_hats[0] with the cross product of rho_hats[1] and rho_hats[2].
    - D2s (list of float): List of D2s for each observation.

    Returns:
    - list: List of real and positive roots of the scalar equation.
    """
    F = (np.linalg.norm (R_vecs[1]))**2    
    E = -2*(np.dot(rho_hats[1],R_vecs[1]))
    A_1 = tau_3/tau
    B_1 = A_1/6*(tau**2-tau_3**2)
    A_3 = -tau_1/tau
    B_3 = A_3/6*(tau**2-tau_1**2)
    A = (A_1*D2s[0]-D2s[1]+A_3*D2s[2])/(-D_0)
    B = (B_1*D2s[0]+B_3*D2s[2])/(-D_0)
    a = -(A**2+A*E+F)
    b = -mu*(2*A*B+B*E)
    c = -mu**2*B**2
    # The roots are found using np.roots(coefficients)
    # Only real and positive roots are returned
    roots = [np.real (x) for x in (np.roots ([1, 0, a, 0, 0, b, 0, 0, c])) if (np.isreal(x) and x>0)]
    return roots

def open_file (input_file):
    """
    **Read observation data from a file and convert it into usable format.**

    Args:
    - input_file (str): Path to the input file.

    Returns:
    - tuple: A tuple containing lists of the following:
      - (list) Right ascension in radians.
      - (list) Declination in radians.
      - (list) Position vectors.
      - (list) Observation times.
    """
    print ('Reading from file...')
    file = open (input_file)
    lines = file.read().split ('\n')
    # Ensuring the input file has three data points, otherwise the user is prompted again
    while not (len (lines) == 3):
        print ('This file does not have exactly three data points.\nPlease try again with a different file...')
        input_file = input()
        print ('Reading from file...')
        file = open (input_file)
        lines = file.read().split ('\n')
    print (input_file + ' has exactly three data points\nProceeding with Method of Gauss.')   
    ras = []
    decs = []
    times = []
    R_vecs = []
    # value conversion for ra and dec
    for line in lines:
        values = line.split (' ')
        time = [float (val) for val in values [3].split (':')]
        times.append ((float (values[0]), float (values[1]), float(values [2]), *time))
        ra, dec = sexagesimal_to_decimal (values [4], values[5])
        ras.append (ra)
        decs.append (dec)
        R_vecs.append (np.array ([float (values[i]) for i in [6, 7, 8]]))

    return ras, decs, R_vecs, times

def mog ():
    """
    **Perform the Method of Gauss to calculate orbital elements.**
    """
    print ('Provide a path to your input file')
    input_file = input()
    ra, dec, R_vecs, times = open_file (input_file)
    times = get_julian_times (times)
    print ('Provide a type of fg function calculation (three, four, closed)')
    fg_type = input()
    while (fg_type!='three' and fg_type!='four' and fg_type!='closed'):
        print ('That is not a valid option, please try again')
        fg_type = input()

    #constants
    k = 0.0172020989484
    c = 173.145
    mu = 1
    obliquity = np.radians (23.43829194)

    # The R vectors and rho hat vectors need to be in ecliptic
    coord_conversion = np.array ([[1, 0, 0],
                                  [0, np.cos (obliquity), np.sin (obliquity)],
                                  [0, -np.sin (obliquity), np.cos (obliquity)]])
    for i in range(len (R_vecs)):
        R_vecs[i] = np.matmul (coord_conversion, R_vecs[i])
    tau_1, tau_3, tau = get_gaussian_time (times, k)

    rho_hats = np.array ([[cos(ra[i])*cos(dec[i]), sin(ra[i])*cos(dec[i]), sin(dec[i])] for i in range (len (ra))])
    rho_hats = [coord_conversion@vec for vec in rho_hats]

    D_0 = np.dot (rho_hats[0], np.cross (rho_hats[1], rho_hats[2]))
    D1s = [np.dot (np.cross (R_vecs[i], rho_hats[1]), rho_hats [2]) for i in range (len (R_vecs))]
    D2s = [np.dot (np.cross (rho_hats[0], R_vecs[i]), rho_hats[2]) for i in range (len (R_vecs))]
    D3s = [np.dot (rho_hats[0], np.cross (rho_hats[1], R_vecs[i])) for i in range (len (R_vecs))]
    Ds = np.array ([D1s, D2s, D3s])

    # Finding roots for the scalar equation of lagrange
    roots = sel (rho_hats, R_vecs, tau_1, tau_3, tau, mu, D_0, D2s)
    print (len (roots), 'roots found:', *roots)
    tolerance = 1e-12

    # The script iterates through all of the roots, rather than prompting the user to choose one
    for r2 in roots:
        try:
            print ('Using r2 =', r2, ' au')
            print ()
            u = mu/r2**3
            f_1 = 1-1/2*u*tau_1**2
            f_3 = 1-1/2*u*tau_3**2
            g_1 = tau_1-1/6*u*tau_1**3
            g_3 = tau_3-1/6*u*tau_3**3
            r_diff = tolerance+1
            r_previous = 0
            num_iter = 0
            # Beginning iteration for the current value of r2
            while (r_diff > tolerance and num_iter < 10000):
                c_denominator = f_1*g_3-g_1*f_3
                c_1 = g_3/c_denominator
                c_2 = -1
                c_3 = -g_1/c_denominator
                cs = [c_1, c_2, c_3]
                rhos = np.array ([(c_1*Ds[i][0]+c_2*Ds[i][1]+c_3*Ds[i][2])/(cs[i]*D_0) for i in range (Ds.shape[0])])

                r_vecs = [rhos[i]*rho_hats[i]-R_vecs[i] for i in range (len (rhos))]
                d_denom = f_1*g_3-f_3*g_1
                d_1 = -f_3/d_denom
                d_3 = f_1/d_denom
                r2_dot = np.array (d_1*r_vecs[0] + d_3*r_vecs[2])

                times_updated = [times[i]-rhos[i]/c for i in range (len (times))]
                tau_1, tau_3, tau = get_gaussian_time (times_updated, k)
                f_1, g_1, f_3, g_3 = fg_func (fg_type, r_vecs[1], r2_dot, tau_1, tau_3)
                # if num_iter == 0, then there is no previous value to compare against
                if not (num_iter==0):
                    r_diff = np.abs (r_previous-np.linalg.norm (r_vecs[1]))
                r_previous = np.linalg.norm (r_vecs[1])
                num_iter +=1
            if (num_iter >= 9999):
                print ('The root did not converge')
                continue
            else:
                print ('The root converged in', num_iter, 'iterations')
                a, e, I, O, w, M, E, n_radians, P, M_0 = get_orbital_elements (r_vecs[1], r2_dot, t_m = times[1], sqrt_mu = 0.0172020989484)
                n = math.degrees (n_radians)
                print ()
                print ("ORBITAL ELEMENTS")
                print ('\ta =', a)
                print ('\te =', e)
                print ('\tI =', I%360)
                print ('\tO =', O%360)
                print ('\tw =', w%360)
                print ('\tM (at time of second observation) =', M%360)
                print ('\tM (at epoch) =', M_0%360)
                print ('\tE =', E%360)
                print ('\tn =', n)
                print ('\tP =', P/365.25)
                return a, e, I, O, w, M, E, n, P, M_0
        # The entire iteration is surrounded in a try/except to catch any errors during iteration such as negative a values
        except:
            print ('The root did not converge')
            print ()
mog()