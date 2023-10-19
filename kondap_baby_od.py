import numpy as np
import math

def get_orbital_elements (pos, vel, t_m = 2460147.75, t_0 = 2460147.75, sqrt_mu = 1):
    """
    Calculate orbital elements given a position and velocity vector.

    Args:
    - pos (np.ndarray): Position vector in Cartesian coordinates.
    - vel (np.ndarray): Velocity vector in Cartesian coordinates.

    Kwargs:
    - t_m (float): Time of middle observation (default: 2460147.75).
    - t_0 (float): Epoch time (default: 2460147.75).
    - sqrt_mu (float): Square root of the gravitational parameter (default: 1).
      It may change based on the use of conventional or Gaussian units.

    Returns:
    - tuple: A tuple containing the following orbital elements:
      - (float) Semi-major axis (a).
      - (float) Eccentricity (e).
      - (float) Inclination (I) in degrees.
      - (float) Longitude of the ascending node (O) in degrees.
      - (float) Argument of periapsis (w) in degrees.
      - (float) Mean anomaly (M) in degrees.
      - (float) Eccentric anomaly (E) in degrees.
      - (float) Mean motion (n).
      - (float) Orbital period (P).
      - (float) Mean anomaly at epoch (M_0) in degrees.
    """
    pos_mag = np.linalg.norm (pos)

    #Getting semi-major axis
    a = 1/(2/pos_mag-np.dot (vel, vel))

    #Mean Motion
    n = sqrt_mu/(math.sqrt (a**3))

    # h value
    h = np.cross (pos, vel)

    # eccentricity
    e = np.sqrt (1-(np.linalg.norm (h))**2/a)

    # Inclination
    I = math.acos (h[2]/np.linalg.norm (h))

    # OMEGA
    O = np.arctan2 (h[0]/(np.linalg.norm (h)*np.sin (I)), -h[1]/(np.linalg.norm (h)*np.sin (I)))

    # omega
    sin_wf = pos[-1]/(pos_mag*np.sin (I))
    cos_wf = 1/np.cos (O) * (pos[0]/pos_mag+np.cos (I)*sin_wf*np.sin (O))
    w_plus_f = np.arctan2 (sin_wf, cos_wf)

    cos_f = 1/e* (a*(1-e**2)/pos_mag-1)
    sin_f = np.dot (pos, vel)/(e*pos_mag)*np.sqrt (a*(1-e**2))
    f = np.arctan2 (sin_f, cos_f)
    w = w_plus_f-f

    # M, Mean anomaly
    cos_E = 1/e*(1-pos_mag/a)
    if (f>0):
        E = np.arccos (cos_E)
    else:
        E = np.pi*2 - np.arccos (cos_E)
    M = E-e*np.sin (E)

    # Finding mean anomaly at epoch
    M_0 = (M+(n)*(t_0-t_m))%(np.pi*2)


    P = 2*np.pi/n

    return a, e, math.degrees (I), math.degrees (O), math.degrees (w), math.degrees (M), math.degrees (E), n, P, math.degrees (M_0)