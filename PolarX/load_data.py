from string import ascii_letters
from glob import glob
from tifffile import TiffSequence
import numpy as np
import xarray as xr
import pickle

def load_metadata(filepath):
    """ Creates a metadata dict from a .txt file in filepath"""
    data = {}
    lines2read = {
        "X Dimension": {"nx" : 0, "xstep" : 3},
        "Y Dimension": {"ny" : 0},
        "T Dimension": {"nt" : 0, "total_time" : 2}, 
        "Sampling Speed": {"sampling_speed": 0},
        "Laser Transmissivity": {"laser_transmissivity": 0},
        "Laser Wavelength": {"laser_wavelength": 0},
    }
    with open(glob(filepath + '*.txt')[0]) as f:
        for line in f:
            array = line.split('"')
            name, info = array[1], array[3]
            if name in lines2read:
                info = info.translate({ord(c): None for c in ascii_letters + ',[]/-%:*'}).split()
                for key in lines2read[name]:
                    data[key] = float(info[lines2read[name][key]])
    md = {
        'total_time' : data['total_time'],
        'xstep' : data['xstep'],
        'tstep' : data['total_time'] / data['nt'],
        'shape' : (int(data['nt']), int(data['ny']), int(data['nx'])),
        'time_per_image' : 1e-6 * data['sampling_speed'] * data['ny'] * data['nx'],
        }
    return md

def coords_from_metadata(md):
    """ Make coordinate arrays (t, y, x) from a metadata dictionary
    """
    t = md['tstep'] * np.arange(md['shape'][0])
    x = md['xstep'] * np.arange(md['shape'][2])
    y = md['xstep'] * np.arange(md['shape'][1])
    return t, y, x

def load_tif_as_xarray(path, channel):
    md = load_metadata(path)
    with TiffSequence(path + '*C00' + str(channel) + '*.tif', pattern=None) as tifs:
        return xr.DataArray(tifs.asarray(), coords=coords_from_metadata(md), dims=('t', 'y', 'x'))

def load_npy_as_xarray(path, filename):
    md = load_metadata(path)
    return xr.DataArray(np.load(path + filename), coords=coords_from_metadata(md), dims=('t', 'y', 'x'))

def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=-1)
        
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)