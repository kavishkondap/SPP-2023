import numpy as np
from astropy.io import fits
import math
import re
from astropy.utils.data import get_pkg_data_filename

def get_img_data (path):
    """
    Read a FITS file and return its data as a numpy array.

    Args:
    - path (str): The path to the FITS file.

    Returns:
    - np.ndarray: Numpy array containing the data from the FITS file.
    """
    image_file = get_pkg_data_filename (path)
    image_data = np.array (fits.getdata (image_file), dtype=np.float32)
    return image_data

def extract_image (img, x_0, y_0, radius):
    """
    Extract a region of interest (ROI) from an image.

    Args:
    - img (np.ndarray): The input image as a numpy array.
    - x_0 (float): X-coordinate of the center of the ROI.
    - y_0 (float): Y-coordinate of the center of the ROI.
    - radius (float): Radius of the ROI.

    Returns:
    - np.ndarray: Numpy array containing the extracted ROI.
    """
    # extracts the ROI from a given FITS file
    radius = int (radius)
    right_bound = int (x_0+radius+1)
    left_bound = int (x_0-radius)
    upper_bound = int (y_0-radius)
    lower_bound = int (y_0+radius+1)
    return img [upper_bound:lower_bound, left_bound:right_bound]

def recursion (adu, center_x, center_y, x, y, length, radius, x_weights, y_weights, n_ap):
    """
    Recursive function to divide a pixel into sub-pixels until ADU value is less than 1.

    Args:
    - adu (float): Analog-to-digital unit value.
    - center_x (float): X-coordinate of the center of the region.
    - center_y (float): Y-coordinate of the center of the region.
    - x (float): X-coordinate of the current pixel.
    - y (float): Y-coordinate of the current pixel.
    - length (float): Length of the current pixel.
    - radius (float): Radius of the region.
    - x_weights (np.ndarray): X-coordinate weights.
    - y_weights (np.ndarray): Y-coordinate weights.
    - n_ap (int): Number of sub-pixels in the region.

    Returns:
    - np.ndarray: Updated x_weights.
    - np.ndarray: Updated y_weights.
    - int: Updated number of sub-pixels.
    """
    base_x = int (x)
    base_y = int (y)
    adu/=4
    sub_arr_length = length/2
    increments = [-sub_arr_length/2, sub_arr_length/2]
    sub_arr_xcoords = []
    sub_arr_ycoords = []
    for i in range (len (increments)):
        for j in range (len (increments)):
            sub_arr_xcoords.append (x+increments[i])
            sub_arr_ycoords.append (y+increments[j])
            sub_pix_state = pixel_state (x+increments[i], y+increments[j], center_x, center_y, radius, increments)
            if (sub_pix_state=='in'):
                x_weights [base_x]+=adu
                y_weights [base_y]+=adu
                n_ap += sub_arr_length**2
            elif (sub_pix_state=='partial'):
                if (adu/4>1):
                    x_weights, y_weights, n_ap = recursion (adu, center_x, center_y, x+increments[i], y+increments[j], sub_arr_length, radius, x_weights, y_weights, n_ap)
                elif (circle_function (x+increments[i], y+increments[j], center_x, center_y, radius)):
                    x_weights[base_x]+=adu
                    y_weights [base_y]+=adu
                    n_ap+=sub_arr_length**2
    return x_weights, y_weights, n_ap

def get_avg_background (annulus_extraction, annulus_inner, annulus_outer):
    """
    Calculate the average background value using the center pixel method in the annulus.

    Args:
    - annulus_extraction (np.ndarray): Numpy array of the annulus extraction.
    - annulus_inner (float): Inner radius of the annulus.
    - annulus_outer (float): Outer radius of the annulus.

    Returns:
    - float: Average background value.
    - int: Number of pixels used in the average calculation.
    """
    total = 0
    count = 0
    rows, cols = annulus_extraction.shape
    center_pix_row = int (rows/2)
    center_pix_col = int (cols/2)
    for row in range (rows):
        for col in range (cols):
            dist = math.sqrt ((center_pix_col-col)**2+(center_pix_row-row)**2)
            if (dist > annulus_inner and dist<annulus_outer):
                total+= annulus_extraction[row][col]
                count+=1
    return total/count, count

def circle_function (x, y, center_x, center_y, radius):
    """
    Check if a given (x, y) point is inside or on the specified circle.

    Args:
    - x (float): X-coordinate of the point.
    - y (float): Y-coordinate of the point.
    - center_x (float): X-coordinate of the circle center.
    - center_y (float): Y-coordinate of the circle center.
    - radius (float): Radius of the circle.

    Returns:
    - bool: True if the point is inside or on the circle, False otherwise.
    """
    pix_radius = np.sqrt ((x-center_x)**2+(y-center_y)**2)
    return pix_radius<=radius

def pixel_state (pix_x, pix_y, center_x, center_y, radius, increments):
    """
    Determine the state of a pixel (in, partial, or out) in relation to the specified circle.

    Args:
    - pix_x (float): X-coordinate of the pixel.
    - pix_y (float): Y-coordinate of the pixel.
    - center_x (float): X-coordinate of the circle center.
    - center_y (float): Y-coordinate of the circle center.
    - radius (float): Radius of the circle.
    - increments (list): List of increments for checking corners of the pixel.

    Returns:
    - str: 'in' if the pixel is entirely inside the circle,
           'partial' if the pixel is partially inside the circle,
           'out' if the pixel is entirely outside the circle.
    """
    count_in = 0
    count_out = 0
    # iterates through all the corners of the pixel to determine if they're in or out
    for x_increment in increments:
        for y_increment in increments:
            corner_state = circle_function (x_increment+pix_x, y_increment+pix_y, center_x, center_y, radius)
            if (corner_state):
                count_in+=1
            else:
                count_out+=1
    if (count_out>0 and count_in>0):
        return 'partial'
    elif (count_out>0):
        return 'out'
    else:
        return 'in'
    
def get_deviation (x_centroid, y_centroid, x_weights, y_weights):
    """
    Calculate the standard deviation of the centroid value.

    Args:
    - x_centroid (float): X-coordinate of the centroid.
    - y_centroid (float): Y-coordinate of the centroid.
    - x_weights (np.ndarray): X-coordinate weights.
    - y_weights (np.ndarray): Y-coordinate weights.

    Returns:
    - float: Standard deviation in the X-coordinate.
    - float: Standard deviation in the Y-coordinate.
    """
    N = np.sum (x_weights)
    sigma_x_numerator = np.sum ([x_weights[i]*(x_centroid-i)**2 for i in range (len (x_weights))])
    sigma_x = math.sqrt (sigma_x_numerator/(N*(N-1)))
    sigma_y_numerator = np.sum ([y_weights[i]*(y_centroid-i)**2 for i in range (len (y_weights))])
    sigma_y = math.sqrt (sigma_y_numerator/(N*(N-1)))
    return sigma_x, sigma_y

def centroid (img_path, x_0_image, y_0_image, radius, annulus_inner, annulus_outer):
    """
    Calculate the centroid of an object in an image.

    Args:
    - img_path (str): Path to the input image.
    - x_0_image (float): X-coordinate of the object in the image.
    - y_0_image (float): Y-coordinate of the object in the image.
    - radius (float): Radius of the region of interest.
    - annulus_inner (float): Inner radius of the annulus for background calculation.
    - annulus_outer (float): Outer radius of the annulus for background calculation.

    Returns:
    - float: X-coordinate of the weighted mean of the centroid.
    - float: Y-coordinate of the weighted mean of the centroid.
    - float: Standard deviation in the X-coordinate.
    - float: Standard deviation in the Y-coordinate.
    - int: Number of sub-pixels used in the calculation.
    - float: Analog-to-digital unit value (ADU) of the object.
    - int: Number of pixels used for the average background calculation.
    - float: Average background value.
    """
    x_0_python = x_0_image - 1
    y_0_python = y_0_image - 1

    img = get_img_data (img_path)
    annulus_extraction = extract_image (img, x_0_python, y_0_python, annulus_outer)
    avg_background, n_an = get_avg_background (annulus_extraction, annulus_inner, annulus_outer)
    img_extraction = extract_image (img, x_0_python, y_0_python, radius)
    img_extraction = img_extraction-avg_background
    del (img) #get rid of the image variable itself to save memory

    rows, cols = img_extraction.shape
    center = ((rows-1)/2, (cols-1)/2)

    x_weights = np.zeros ((rows))
    y_weights = np.zeros ((cols))

    n_ap = 0

    # main iteration loop, determines if a pixel can be added, or needs recursion
    for row in range (rows):
        for col in range (cols):
            pix_state = pixel_state (row, col, center[0], center[1], radius, increments=[-0.5, 0.5])
            if (pix_state=='in'):
                x_weights [col]+=img_extraction [row][col]
                y_weights [row]+=img_extraction[row][col]
                n_ap+=1
            elif (pix_state=='partial'):
                adu = img_extraction[row][col]
                x_weights, y_weights, n_ap = recursion (adu, center[0], center[1], row, col, 1, radius, x_weights, y_weights, n_ap)
    
    ADU_ap = np.sum (x_weights)
    x_weights_numerator = 0
    y_weights_numerator = 0

    for i in range (rows):
        x_weights_numerator+=x_weights[i]*i
    for i in range (cols):
        y_weights_numerator+=y_weights [i]*i

    x_weighted_mean = x_weights_numerator/np.sum (x_weights)
    y_weighted_mean = y_weights_numerator/np.sum (y_weights)

    sigma_x, sigma_y = get_deviation (x_weighted_mean, y_weighted_mean, x_weights, y_weights)

    # adjust for ROI cutout:
    x_weighted_mean = x_weighted_mean + (x_0_python-int (cols/2))+1
    y_weighted_mean = y_weighted_mean + (y_0_python-int (rows/2))+1

    return x_weighted_mean, y_weighted_mean, sigma_x, sigma_y, n_ap, ADU_ap, n_an, avg_background

def get_snr (signal, n_ap, n_an, sky_e, d_e, rho_sq, gain):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of an object.

    Args:
    - signal (float): Signal value.
    - n_ap (int): Number of sub-pixels used for aperture photometry.
    - n_an (int): Number of pixels used for average background calculation.
    - sky_e (float): Sky background value.
    - d_e (float): Error value.
    - rho_sq (float): Noise term.
    - gain (float): Gain value.

    Returns:
    - float: Signal-to-Noise Ratio (SNR).
    """
    signal*=gain
    sky_e*=gain
    numerator = np.sqrt (signal)
    n_ap = 79
    denom_term1 = n_ap*(1+(n_ap/n_an))
    denom_term2 = (sky_e+d_e+rho_sq)/signal
    denominator = np.sqrt (1+denom_term1*denom_term2)
    return numerator/denominator

def photometry (input_file):
    """
    Calculate photometric values from an input file.

    Args:
    - input_file (str): Path to the input file containing photometry parameters.

    Returns:
    - float: Signal value.
    - float: Signal-to-Noise Ratio (SNR).
    - float: Instrumental magnitude (m_inst).
    - float: Error in instrumental magnitude (sigma_m_inst).
    """
    input_file = open (input_file)
    input_lines = input_file.readlines()
    img_path = input_lines [0][0:-1]
    input_lines = input_lines[1:] #getting rid of the image path, since it was already saved
    input_lines = [float (re.findall ('(\d+(?:\.\d+)?)', val)[0]) for val in input_lines]
    x_0_image = float (input_lines[0])
    y_0_image = float (input_lines[1])
    radius = float (input_lines[2])
    annulus_inner = float (input_lines[3])
    annulus_outer = float (input_lines[4])

    x_weighted_mean_initial, y_weighted_mean_initial, _, _, _, _, _, _ = centroid (img_path, x_0_image, y_0_image, radius, annulus_inner, annulus_outer)
    x_weighted_mean, y_weighted_mean, sigma_x, sigma_y, n_ap, ADU_ap, n_an, avg_background = centroid (img_path, x_weighted_mean_initial, y_weighted_mean_initial, radius, annulus_inner, annulus_outer)

    gain = 0.75
    read_noise = 6.48

    signal = ADU_ap
    sky_e = avg_background
    d_e = 10

    m_inst = -2.5*np.log10 (ADU_ap)

    rho_sq = read_noise**2+(gain**2)/12

    snr = get_snr (signal, n_ap, n_an, sky_e, d_e, rho_sq, gain)

    sigma_m_inst = 1.0857/snr
    print ('values')
    print (signal, signal/snr, snr, m_inst, sigma_m_inst)
    return signal, signal/snr, m_inst, sigma_m_inst

photometry ('input.txt')