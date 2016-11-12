# Additional functionality for the xarray.DataArray and xarray.DataSet classes
# (http://xarray.pydata.org/en/stable/)
# mostly for working with the time series of images with 
# radially symmetric intensity distributions

from skimage.feature import register_translation 
from scipy.ndimage import shift
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn import mixture
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.stats as ss
import pylab as plt
import xarray as xr

R_BINS = np.r_[0., 0.55, .8, 1.0, 1.2, 1.4, np.arange(1.6, 5.8, .2)]
THETA_BINS = np.linspace(-np.pi, np.pi, 9)

def make_bins(n_theta_bins, n_pixels=10, r_stop=1.5, r_step=.25, r_max=15, pixel_area=0.015376):
    """ Create bin edges of annuli with constant area.
    
    For r values in [0, r_stop] creates bin edges, so there is approximately constant number of
    pixels in each bin (n_pixels). After r_stop the bin edges are equidistant with an r_step 
    
    Parameters
    ----------
    n_theta_bins : int
        Number of theta bins
    n_pixels : int
        Target number of pixels for each polar coordinate using base_lib.darray_to_polar
    r_stop : float
        Maximum value that r_bins[-1] can reach
    pixel_area : float
        Area of a pixel in (μm)^2
        
    Returns
    -------
    theta_bins : numpy.ndarray
        Array with bin edges for theta from -pi to pi with length n_theta + 1
    r_bins : numpy.ndarray
        Array with bin edges for r from 0 to <r_stop creating constant circle/ring areas for each bin
    """
    r_bin_edges = np.sqrt(np.arange(0, np.pi*r_stop**2, n_theta_bins*n_pixels*pixel_area) / np.pi)
    return np.linspace(-np.pi, np.pi, n_theta_bins+1), \
            np.r_[r_bin_edges, np.arange(r_bin_edges[-1] + r_step, r_max, r_step)]

def strip_image(x, dim='x', start=24, end=38):
    """ Discard the middle part of the image.
    
    Parameters
    ----------
    x : xarray.DataArray or xarray.Dataset
    dim : str
        Cuts an image along specified dimension (x or y)
    start : int or float
        start edge of a discarded region (in um)
    end : int or float
        end edge of a discarded region (in um)
    
    Returns
    -------
    y : xarray.DataArray or xarray.DataSet
        DataArray or Dataset where the specified part of each image was discarded
    """
    assert dim in ('x', 'y'), 'Allowed dim values: x or y'
    return xr.concat([x.sel(x=slice(0, start)), x.sel(x=slice(end, None))], 'x')

def remove_artefact(x, thr1=-4, thr2=-2.5, **kwargs): # 
    """ Remove spatial intensity drops (detector artefact) from each image.
    
    Stripe-like (horizontal) artefacts in images due to the thermal noise in GaAsP detectors in
    two-photon microscope. Artefact removal based on the assumption that the amplitude of the noise
    is constant (k_noise variable below).
    
    Parameters
    ----------
    x : xarray.DataArray 
        x.dims = ('t', 'y', 'x')
    thr1 : float
        -- threshold separating the correct intensity values and those affected by fluctuations 
    thr2 : float
        -- threshold for defining the 'y' - width of the fluctuations
        
    Returns
    -------
    y : xarray.DataArray
        corrected DataArray
    lines : xarray.DataArray
        DataArray where each image is reduced into a line by averaging along x dimension
    k_noise : float
        Estimated noise fluctuation amplitude
    """
    lines = strip_image(x, **kwargs).mean('x') 
    lines -= np.median(lines, axis=1)[:, np.newaxis] # sets background intensity to zero
    k_noise = np.rint(-lines.where(lines < thr1).mean()) # the average magnitude of an intensity drop
    
    if x.max() + k_noise > np.iinfo(x.values.dtype).max: # Prevents overflow of values in uint8 numpy array
        x = x.astype('uint16')
        
    return x + (k_noise * (lines < thr2)).astype('uint8'), lines, k_noise

def register_xarray(x, ref_image, n_aver=100, upsample=10):
    """ Translational stabilization of images using cross-correlation.
    
    Based on: 
    http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature#skimage.feature.register_translation
    Reference:
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 'Efficient subpixel image registration algorithms,'
    Optics Letters 33, 156-158 (2008).
    
    Parameters:
    x : DataArray, 
        DataArray to register. x.dims == ('t', 'y', 'x').
    ref_image : DataArray or numpy.array
        Reference image to register to.
    n_aver : int 
        number of images to average (block average in time) before registering 
    upsample : float 
        http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature#skimage.feature.register_translation
        
    Returns:
    shifts : numpy.array
        2D array of shifts (dy, dx) in pixels to apply to each image
    """
    def interpolate_parameter(average_array):
        return np.interp(x.t, average_array.t, average_array.values)

    def interpolate_shifts(average_shifts):
        return np.array((interpolate_parameter(average_shifts[:, 0]),
                         interpolate_parameter(average_shifts[:, 1]))).T

    def reg_func(image):
        shifts, _err, _phasediff = register_translation(ref_image, image, upsample_factor=upsample)
        return xr.DataArray(shifts)

    shifts = x.pipe(chunk_average_in_time, n_aver).groupby('t').apply(reg_func)
    return interpolate_shifts(shifts)

def shift_xarray(x, shifts):
    """ Shifts each image in DataArray. 
    
    Parameters:
    x : DataArray
        DataArray to register, x.dims = ('t', 'y', 'x').
    shifts : numpy.array
        array of shifts [[dy1, dx1], [dy2, dx2], [dy3, dx3], ...]. 
        
    Returns:
    x : DataArray
        registered DataArray
    """
    x['dx'], x['dy'] = ('t', shifts[:, 1]), ('t', shifts[:, 0])
    def shift_single_xarray(x):
        return shift(x, (x.dy, x.dx), order=1, mode='nearest')
    return x.groupby('t').apply(shift_single_xarray)

def simp_mean(array):
    return .5 * (array[1:] + array[:-1])

def standardize(x):
    return (x - x.mean()) / x.std()

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def normalize_dataset(x):
    """ Normalizes intensity values into [0, 1] range for each "t" and "theta" values
    """
    assert all([i in x.dims for i in ('t', 'theta', 'r')]), 'Dataset dimensions should contain "t", "theta", "r" '
    return x.y.groupby('t').apply(lambda y: y.groupby('theta').apply(normalize))

def set_radius(x):
    """ Calculates radius for a normalized polar dataarray -- FIX IT!!!
    """
    assert (np.nanmax(x) == 1) & (np.nanmin(x)  == 0), 'DataArray should be normalized'
    
    def radius(x, thr=.5):
        x_cut = x.where((x.r > x.r[int(x.argmax())]) & (x.r < x.r[int(x.argmin())]), drop=True)
        r_fine = np.arange(x_cut.r[0] + 1e-5, x_cut.r[-1] - 1e-5, .001)
        x_interp = xr.DataArray(interp1d(x_cut.r, x_cut, kind='cubic')(r_fine), coords={'r' : r_fine, 
                                                                                            't' : x.t, 
                                                                                            'theta' : x.theta}, dims=('r',))
        return x_interp.r[int((abs(x_interp.values - thr).argmin()))]
    
    x.coords['R'] = (('t', 'theta'), x.groupby('t').apply(lambda y: y.groupby('theta').apply(radius)))
    return x

def cartesian_to_polar(x, y, origin):
    """ Calculates polar coordinates grid from Cartesian.
    
    Parameters:
    x : numpy.array
        array of x-values.
    y : numpy.array
        array of y-values.
    origin : tuple
        coordinates of an origin (x_center, y_center)
        
    Returns:
    theta : numpy.array
        array of theta values (2D)
    R : numpy.array
        array of R values (2D)
    """
    x_g, y_g = np.meshgrid(x - origin[0], y - origin[1])
    return np.arctan2(y_g, x_g), np.sqrt(x_g ** 2 + y_g ** 2)

def chunk_average_in_time(x, chunk_size):
    """ Performs chunk averaging of images in time.
    
    Parameters:
    x : DataArray or Dataset
        DataArray or Dataset to chunk average in time, 't' should be in x.dims
    chunk_size : int 
        size of an image chunk (number of images to average)
        
    Returns:
    averaged_array : DataArray or Dataset
        chunked averaged in time DataArray or Dataset
    """
    t_bins = x.t[::chunk_size].values
    return x.groupby_bins('t', t_bins, labels=simp_mean(t_bins)).mean('t').rename({'t_bins': 't'})

def apply_to_chunks_in_time(x, f, chunk_size):
    """ Performs chunk averaging of images in time.
    
    Parameters:
    x : DataArray or Dataset
        DataArray or Dataset to chunk average in time, 't' should be in x.dims
    f : function
        function to apply to each chunk in time
    chunk_size : int 
        size of an image chunk (number of images to average)
        
    Returns:
    reduced_array : DataArray or Dataset
    """
    t_bins = x.t[::chunk_size].values
    return x.groupby_bins('t', t_bins, labels=simp_mean(t_bins)).apply(f).rename({'t_bins': 't'})

def darray_to_polar(x, origin=(0, 0), bins=(THETA_BINS, R_BINS), statistic='mean'):
    """Converts DataArray from Cartesian to polar coordinates.
    
    Parameters:
    x : DataArray, 
        DataArray with x.dims = ('y', 'x').
    origin : tuple
        coordinates of an origin (x_center, y_center)
    bins : See http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html
    statistic : See http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html
    
    Returns:
    y : DataArray
        DataArray in polar coordinates
    """
    theta, r = cartesian_to_polar(x.x, x.y, origin)
    binned_array, theta, r, b = ss.binned_statistic_2d(theta.ravel(), r.ravel(), 
                                                       x.values.ravel(), statistic=statistic, bins=bins)
    return xr.DataArray(binned_array, coords=[simp_mean(theta), simp_mean(r)], dims=['theta', 'r'])

def get_polar_dataset(x, **kwargs):
    """ Transforms DataArray to a Dataset with images in polar coordinates.
    
    Parameters:
    x : DataArray
        DataArray, x.dims = ('t', 'y', 'x').
    **kwargs:
        see darray_to_polar
        
    Returns:
    y : Dataset
        Dataset with intensity corresponding errorbars in polar coordinates
        y.y : intensity values
        y.yerr : corresponding errorbars
        y.n : number of pixels per bin
    """
    dset = xr.Dataset({'y' : x.groupby('t').apply(darray_to_polar, statistic='mean', **kwargs),
                'yerr' : x.groupby('t').apply(darray_to_polar, statistic=ss.sem, **kwargs)})
    dset.coords['n'] = (('theta', 'r'), darray_to_polar(x[0], statistic='count', **kwargs))
    return dset

def optimize_origin(x, guess0, bins=(THETA_BINS, R_BINS), rmin=None, rmax=None, method='Nelder-Mead'):
    """ Calculates optimal origin position corresponding to the most symmetric intensity distribution 
    
    Parameters:
    x : DataArray, 
        DataArray with x.dims = ('t', 'theta', 'r').
    guess0 : tuple
        initial guess for the origin position 
    bins : See http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html
    rmin, rmax : float
        intensity values with 'r' in [rmin, rmax] are only used
    method : see http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    
    Returns:
    O : OptimizeResult 
        see output in http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    def cost_func(p, bins):
        return np.sqrt(x.groupby('t').apply(darray_to_polar, origin=p, bins=bins)
                       .sel(r=slice(rmin, rmax)).var('theta').mean())
    return minimize(cost_func, guess0, args=(bins,), method=method)


def covariance(x):
    """ Calculates covariance between intensity at the capillary center
    and all other r-values.
    
    Parameters:
    x : Dataset,
        Dataset, x.dims = ('t', 'theta', 'r').
    Returns:
    y : DataArray,
        DataArray with covariance values on (theta, r) grid
    """
    def cov(y):
        def inner(y_r):
            return xr.DataArray(np.cov(y.isel(r=0), y_r)[0,1], coords={'r' : y_r.r,
                                                         'theta' : y_r.theta}) 
        return y.groupby('r').apply(inner)
    return x.y.groupby('theta').apply(cov)
    
    
def corrcoef(x):
    """ Calculates correlation coeffcient between intensity at the capillary center
    and all other r-values.
    
    Parameters:
    x : Dataset,
        Dataset, x.dims = ('t', 'theta', 'r').
    Returns:
    y : DataArray,
        DataArray with covariance values on (theta, r) grid    
    """
    def corr(y):
        def inner(y_r):
            return xr.DataArray(np.corrcoef(y.isel(r=0), y_r)[0,1], coords={'r' : y_r.r,
                                                         'theta' : y_r.theta}) 
        return y.groupby('r').apply(inner)
    return x.y.groupby('theta').apply(corr)


def get_nocell_profiles(x, thr=.95):
    """
    
    """
    X_train = x.w1.values[:, newaxis]
    gmm = mixture.GMM(n_components=2, covariance_type='full')
    gmm.fit(X_train)
    x.coords['w1_labels'] = (('t'), gmm.predict_proba(X_train)[:, gmm.means_.ravel().argmax()])
    return x.where(x.w1_labels > thr, drop=True)


def make_xarr_func(func):
    """ Transforms output of a function to a DataArray instead of numpy.array 
    
    Positional arguments:
    func -- numpy function, transforming numpy.array into new numpy.array
    """
    def xfunc(darr, *args, **kwargs):
        return xr.DataArray(func(darr, *args, **kwargs), coords=darr.coords, dims=darr.dims)
    return xfunc


def weighted_mean(x, dim='theta'):
    """ Weighted average along r or theta dimension.
    
    Parameters
    ----------
    x : xarray.Dataset instance
    dim : str
        Dimension, along which the weighted averaging is performed
    
    Returns
    -------
    reduced : xarray.Dataset instance
        Dataset with a reduced dimension
    
    """
    assert dim in ('r', 'theta'), 'Allowed dimensions are: r or theta'
    weights = x.n / x.n.sum(dim)
    variables = {
        'y' : (x.y * weights).sum(dim),
        'yerr' : np.sqrt((x.yerr ** 2 * weights).sum(dim))
    }
    return xr.Dataset(variables)

def make_pca(x, n_comp=2, pca_func=PCA, index=1):
    """ Apply PCA-based dimensionality reduction.
    
    Positional arguments:
    x -- DataArray or a Dataset, containing variable 'y', x.dims = ('t', 'r').
    Keyword arguments:
    n_comp -- number of principle components
    pca_func -- function for doing PCA. Options: PCA or SparsePCA
    index -- index over 'r' dimension, which defines the sign of the principle component.
    All PC are converted to the same sign.
    """
    pca = pca_func(n_components=n_comp) 
    low_dim_data = pca.fit_transform(StandardScaler().fit_transform(x['y'].groupby('t').apply(standardize).values))
    pca_signs = np.sign(np.sum(pca.components_[:, :index], axis=1))[:, np.newaxis]
    for i, (y, pc) in enumerate(zip(low_dim_data.T * pca_signs, pca.components_ * pca_signs), start=1):
        x.coords['pc' + str(i)] = ('r', pc)
        x.coords['w' + str(i)] = ('t', y)
    return x


def gmm_clustering(x, n=2):
    sign2label = lambda x: 'glycocalyx' if x < 0 else 'lumen'
    gmm = mixture.GMM(n_components=n, covariance_type='full')
    x.coords['labels'] = ('t', gmm.fit_predict(np.stack([x.w1, x.w2]).T))
    x.coords[sign2label(gmm.means_[0, 0])] = ('r', x.y.where(x.labels == 0).mean('t'))
    x.coords[sign2label(gmm.means_[1, 0])] = ('r', x.y.where(x.labels == 1).mean('t'))
    return x

def partition_coeff(x, r_lumen=1.):
    x.coords['partition_coeff'] = (('theta'), \
        x.glycocalyx.max('r') / x.lumen.sel(r=slice(None, r_lumen)).mean('r')) 
    return x.partition_coeff.mean().values, ss.sem(x.partition_coeff)

'''
def make_glycocalyx_image(x, w1_thr=-2, w2_thr=-2):
    x.coords['glycocalyx'] = (('theta', 'r'), \
        x.y.where((x.w1 < w1_thr) & (x.w2 < w2_thr)) \
        .mean('t').groupby('theta').apply(standardize))

def make_lumen_image(x, w1_thr=2, w2_thr=-2):
    x.coords['lumen'] = (('theta', 'r'), \
        x.y.where((x.w1 > w1_thr) & (x.w2 < w2_thr)).mean('t'))    
    
def partition_coeff(x, glx_thr = 1.5, r_lumen = 1.):
    x.coords['partition_coeff'] = (('theta'), \
        x.lumen.where(x.glycocalyx > glx_thr).mean('r') \
        / x.lumen.sel(r = slice(None, r_lumen)).mean('r')) 
    return x.partition_coeff.mean().values, ss.sem(x.partition_coeff)
    
def get_corr(x, ref):
    arr = x.y.transpose('theta', 'r', 't').values
    corr = np.array([np.cov(x, ref)[0, 1] for x in arr.reshape((arr.shape[0]*arr.shape[1], arr.shape[2]))])
    x.coords['corrcoeff'] = (('theta', 'r'), np.reshape(corr, (arr.shape[0], arr.shape[1])))
    return x
'''