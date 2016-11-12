from scipy.interpolate import RectBivariateSpline
from scipy.optimize import leastsq
from scipy.ndimage.filters import gaussian_filter1d
from collections import namedtuple
import numpy as np
import xarray as xr

# Setting default fit parameters
Fit_params = namedtuple('Fit_params', ['r_step', 't_step', 'right_boundary', 'fit_boundary', 'left_boundary_smooth', 
                                       'WLS', 'D_init_guess', 'K_init_guess', 'verbose', 'interp_index'])
P = Fit_params(.1, .1, 30., 0., 1., 'yes', 1e-3, '', 'yes', -5)

def fit_data(x, P, **kwargs):
    """ 
    Input: x - xarray.DataSet instance with variables ('y', 'yerr', 'n')
            
    """
    def get_boundary_condition(dataset, sigma=1):
        """ Return the boundary condition function for the solver """
        def boundary(t):
            return np.interp(t, dataset['t'], gaussian_filter1d(dataset['y'][:, 0, 0], sigma))
        return boundary
    
    def run_solver(p):
        return explicit_solver(r_grid, t_grid, init_cond, f_boundary, p)

    def residuals(p):  
        theory_interp = run_solver(p)(x['t'], x['r'])
        residuals = (y[:, :right_edge] - theory_interp[:, :right_edge]) / weights[:, :right_edge]
        residuals_full = (y - theory_interp) / weights
        if P.verbose:
            K = 0 if len(p) == 1 else p[1]
            chi2 = np.sum(residuals_full**2) / n_dof
            print('D = {0:.3E} um^2/s, K_r = {1:.3E} 1/s, chi2 = {2:.7f}'.format(p[0], K, chi2)) 
        return residuals.ravel()
    
    y, yerr = x['y'].isel(theta=0).values, x['yerr'].isel(theta=0).values
    f_boundary = get_boundary_condition(x, sigma=P.left_boundary_smooth)
    r_grid = np.arange(x['r'][0], P.right_boundary, P.r_step)
    t_grid = np.arange(x['t'][0], x['t'][-1], P.t_step)
    init_cond = np.interp(r_grid, x['r'], y[0], right=np.mean(y[0][P.interp_index:]))
    init_guess = [x for x in (P.D_init_guess, P.K_init_guess) if x]
    n_dof = y.size - y[0].size - x['t'].size - len(init_guess)
    weights = yerr if P.WLS else np.ones(yerr.shape)
    right_edge = len(y[0]) - P.fit_boundary
    
    fit_res = leastsq(residuals, init_guess, **kwargs)[0]
    theory = run_solver(fit_res)
    dims = ('t', 'r')

    exp_data = {
        'y' : (dims, y),
        'yerr' : (dims, yerr),
        'yt' : (dims, theory(x['t'], x['r'])),
    }
    dataset_e = xr.Dataset(exp_data, coords = {'t' : x['t'], 'r' : x['r']})
    dataset_e['res'] = dataset_e['y'] - dataset_e['yt']
    dataset_e['RSS'] = (dataset_e['res'] / dataset_e['yerr'])**2 / n_dof
    dataset_e['gradient'] = (dims, np.gradient(dataset_e['y'])[1])
    dataset_e['flux'] = - fit_res[0]*dataset_e['gradient']
    # save relevant fitting parameters as attributes
    dataset_e.attrs = P._asdict() 
    
    yt_full = theory(t_grid, r_grid)
    dict_t = {
        'y' : (dims, yt_full),
        'yerr' : (dims, np.nan*np.ones(yt_full.shape)),
        'n' : (dims, np.ones(yt_full.shape)),
    }
    dataset_t = xr.Dataset(dict_t, coords = {'t' : t_grid, 'r' : r_grid})
    dataset_t.attrs = P._asdict() 
    return dataset_e, dataset_t 

def explicit_solver(r_grid, t_grid, init_cond, f_boundary, p, s_lim=0.48):
    """ 
        Solves the diffusion equation for a hollow cylinder with a time-dependent boundary condition
        on the cylinder surface and no-flux boundary condition at the second boundary.
        Uses central difference method for the spatial derivatives and forward difference for
        the time derivative.
        
        Input: 
            1) r_grid --- 1D numpy array with a distances from a capillary center in um. First value
            equals to the cylinder radius
            2) dt --- integration time step
            3) T --- final time value
            4) f_boundary --- function that takes a time value in ms and returns the intensity at the first
            boundary
            5) p --- list of parameters, including diffusion coefficient ('D') in um^2/s, photobleaching
            rate constant ('K') in 1\s 
            
        Output: 
            scipy.interpolate.RectBivariateSpline instance, containing the space-time grid with corresponding 
            calculated "concentration values".
            Returns original values if called at grid points and linearly interpolated value if the point is 
            not on the grid.
    """
    # Check if the input is correct:
    assert hasattr(r_grid, '__iter__'), 'Space grid should be an array'
    assert hasattr(t_grid, '__iter__'), 'Time grid should be an array'
    assert hasattr(f_boundary, '__call__'), 'Boundary condition should be a function of time'
    assert len(r_grid) == len(init_cond), 'Initial condition array should be the same length as the space grid'
    assert s_lim < .5, 's_lim value should be less than 0.5'

    
    D, k_pb = p if len(p) == 2 else p[0], 0  
    dr, dt = r_grid[1] - r_grid[0], t_grid[1] - t_grid[0]
    s = D*dt / dr**2 # dimensionless parameter, if < 0.5 then the solution is stable
    
    if s  > s_lim:
        'Solution can be unstable, decreasing the time step'
        dt = s_lim*dr**2 / D
        t_grid = np.arange(t_grid[0], t_grid[-1] + dt, dt)
        s = s_lim # new value of "s" is calculated
    
    Lr, Lt = len(r_grid), len(t_grid)
    K_pb = s * k_pb * dr**2 / D # Dimensionless photobleaching rate constant
    k = np.array([.5 / (r_grid[0] / dr + i) for i in range(len(r_grid))])
    
    # Calculate coefficients for the difference scheme
    a1 = s * (1 + k)
    a2 = 1 - 2 * s - K_pb
    a3 = s * (1 - k)
    b1 = s * (1 - k[-1])
    b2 = 1 - K_pb - b1

    u = np.zeros((Lt, Lr)) # preallocate time-space array for the solution 
    u[0] = init_cond # set initial condition
    u[1:, 0] = np.array([f_boundary(t) for t in t_grid[1:]]) # set time-dependent boundary condition
    
    for i in range(1, Lt):
        u[i, 1:-1] = a1[1: -1]*u[i - 1, 2:] + a2 * u[i - 1, 1: -1] + a3[1: -1] * u[i - 1, :-2]
        u[i, -1] = b1 * u[i-1, -2] + b2 * u[i-1, -1]
        
    return RectBivariateSpline(t_grid, r_grid,  u, kx=1, ky=1)
