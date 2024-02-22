from cProfile import label
from cmath import tau
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc("text",usetex=True)
import scipy as sp
import scipy.interpolate
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from matplotlib.ticker import MultipleLocator

fs=22


mp=938.2720813e6 # eV
me=0.510998e6 # eV


def plot_profile():

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    t, vsh, Rsh, nrg=np.loadtxt("profile.dat",unpack=True,usecols=[0,1,2,3])

    ax.plot(t,vsh,'g-',linewidth=3.0,label=r'$v_{\rm sh}$')
    ax.plot(t,Rsh,'k-',linewidth=3.0,label=r'$R_{\rm sh}$')
    # ax.plot(t,nrg/(1.0e6*0.76*8.41e-58),'r-',linewidth=3.0,label=r'$n_{\rm rg}$')
    ax.plot(t,nrg,'r-',linewidth=3.0,label=r'$n_{\rm rg}$')

    t, vsh=np.loadtxt("1shock_Diesing_vsh.dat",unpack=True,usecols=[0,1])
    ax.plot(t,vsh,'g--',linewidth=3.0)

    t, Rsh=np.loadtxt("1shock_Diesing_Rsh.dat",unpack=True,usecols=[0,1])
    ax.plot(t,Rsh,'k--',linewidth=3.0)

    t, nrg=np.loadtxt("1shock_Diesing_nrg.dat",unpack=True,usecols=[0,1])
    ax.plot(t,nrg,'r--',linewidth=3.0)

    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(1.0e-2,1.0e8)
    ax.set_ylim(1.0e-1,1.0e4)
    ax.set_xlabel(r'$t\,({\rm day})$',fontsize=fs)
    # ax.set_ylabel(r'$p^4f(p)$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":22})
    ax.grid(linestyle='--')

    plt.savefig("fg_profile.png")


plot_profile()
