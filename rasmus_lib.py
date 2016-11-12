# Additions to PolarX
# This is done in PyCharm

import numpy as np
import skimage
import matplotlib as plt

def create_polar_bins(n_theta, n_pixels=80, r_stop=10, pixel_area=0.124**2):
    """ Create bin edges of radius with constant circle/ring area

    Parameters
    ----------
    n_theta : int
        Number of theta bins (usually 8 for PolarX)
    n_pixels : int
        Target number of pixels for each polar coordinate using base_lib.darray_to_polar
    r_stop : float
        Maximum value that r_bins[-1] can reach (μm)
    pixel_area : float
        Area of each pixel in (μm)^2

    Returns
    -------
    theta_bins : numpy.ndarray
        Array with bin edges for theta from -pi to pi with length n_theta+1
    r_bins : numpy.ndarray
        Array with bin edges for r from 0 to <r_stop creating constant circle/ring areas for each bin
    """

    area = n_theta * n_pixels * pixel_area

#    theta_bins = np.linspace(-np.pi, np.pi, n_theta+1)
#    r_bins = np.sqrt(np.arange(0, np.pi * r_stop ** 2, area) / np.pi)
    return np.linspace(-np.pi, np.pi, n_theta+1), \
    np.sqrt(np.arange(0, np.pi * r_stop ** 2, area) / np.pi)


def estimate_center(image, threshold=50):
    # threshold should be from histogram
    edges2 = skimage.filters.gaussian(image < threshold, sigma=5) > 0.5
    edges2 = skimage.filters.sobel(edges2)

    y1, x1 = edges2.nonzero()
    y1 = y1.mean()
    x1 = x1.mean()
    return y1, x1

# remember to set default hough_radii when known
def estimate_center_h(img,hough_radii, sig_gauss=5,thresh_factor=0.90):
    # Can this be done as a pipeline??
    img_sobel = skimage.filters.sobel(skimage.filters.gaussian(img,sigma=sig_gauss))

    my_sort = np.sort(np.ravel(img_sobel))
    thresh = my_sort[int(my_sort.size*thresh_factor)]
    
    edges = img_sobel>thresh # make sure threshold is valid
    
    # Hough transform - 3D matrix w. one 2D matrix per input radius
    hough_res = skimage.transform.hough_circle(edges, hough_radii)
    

    
    # find index of max point in hough_res
    max_radius_index, y, x = np.unravel_index(hough_res.argmax(), hough_res.shape)
    radius = hough_radii[max_radius_index]
    
    # Be very much aware of order of coordinates, is "y,  x, radius" better?
    return x, y, radius



def plotStuff(img,y,x,radius=100):
    imgGray = skimage.color.gray2rgb(img)
    cx, cy = skimage.draw.circle_perimeter(y, x, radius)
    imgGray[cy, cx] = (220, 20, 20)
    plt.figure()
    plt.imshow(imgGray)
    plt.plot(y, x, '.', markersize=12)
