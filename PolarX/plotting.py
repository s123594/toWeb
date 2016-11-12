import numpy as np
import pylab as plt

COORD_DIMS = {'t' : 's', 'theta' : 'rad', 'r' : '$\mu m$'}

def plot_dataset(x, dims=('t', 'theta', 'r'), size=None):
    for v1, grp1 in x.groupby(dims[0]):
        fig, ax = plt.subplots(figsize=size)
        ax.set_title(dims[0] + ' = {0:.2f} '.format(v1) + COORD_DIMS[dims[0]])
        for v2, grp2 in grp1.groupby(dims[1]):
            ax.errorbar(grp2[dims[2]], grp2['y'], yerr=grp2['yerr'], label='{0:.2f} '.format(v2) + COORD_DIMS[dims[1]])
            ax.set_ylabel('Fluorescence intensity, r.u.')
            ax.set_xlabel(dims[2] + ', ' + COORD_DIMS[dims[2]])
        ax.legend(loc='best')

def compare_profiles(ye, yt, N, dy=0, size=None): 
    fig, ax = plt.subplots(figsize=size)
    ax.set_ylabel('Fluorescence intensity, r.u.')
    ax.set_xlabel('Radial distance, $\mu m$')
    clr = plt.rcParams['axes.color_cycle'] * 10
    times = ye.t[range(0, len(ye.t) - len(ye.t) % N, len(ye.t) // N)]
    for i, T in enumerate(times.values):
        ye_i, yt_i = ye.sel(t=T, method='nearest'), yt.sel(t=T, method='nearest')
        ax.errorbar(ye.r, ye_i['y'] + i * dy, yerr=ye_i['yerr'], linestyle='', color=clr[i], label=None)
        ax.plot(yt.r, yt_i['y'] + i * dy,  marker='', color=clr[i], label='{0:.1f} '.format(T) + COORD_DIMS['t'])
    ax.legend()
    ax.set_xlim(ye.r.min()-.2, ye.r.max())
    
def plot_pc(x, n=2, bins=25):
    fig, ax = plt.subplots(3, n, figsize=(8 * n, 12))
    for i, (name1, name2) in enumerate([('pc' + str(j), 'w' + str(j)) for j in range(1, n + 1)]):
            x[name1].plot(ax = ax[0, i])
            x[name2].plot(ax = ax[1, i])
            x[name2].plot.hist(ax = ax[2, i], bins=bins)
    plt.tight_layout()
