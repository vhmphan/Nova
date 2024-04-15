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
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from pack_gato import *
import time

fs=22

# Define a custom tick formatter to display 10^x
def log_tick_formatter(x, pos):
    return r'$10^{%d}$' % int(x)

# Record the starting time
start_time = time.time()

Mcl=5.0e4*2.0e33 # g
mpCGS=1.67e-24 # g
Dcl=2.0e3*3.086e18 # cm

def func_np(Tp):
    TpGeV=Tp*1.0e-9
    mpGeV=mp*1.0e-9
    Ap=3.15e-17
    delta=2.76

    return Ap*(TpGeV+mpGeV)*(TpGeV**2+2.0*TpGeV*mpGeV)**(-0.5*(delta+1.0))

Tp=np.logspace(8.0,15.0,5001)
dTp=Tp[1:-1]-Tp[0:-2]
Tp=Tp[0:-2]

Eg=np.logspace(8.0,13.0,100)

Tp_new, Eg_new=np.meshgrid(Tp, Eg, indexing='ij')
print(Tp_new)
print(func_alpha(Tp_new))


vp=np.sqrt((Tp+mp)**2-mp**2)*3.0e10/(Tp+mp)
Jp=func_np(Tp)*vp
eps_nucl=func_enhancement(Tp)
d_sigma_g=func_d_sigma_g(Tp,Eg)
print(Tp)

phi=np.sum((Mcl/(4.0*np.pi*Dcl**2*1.4*mpCGS))*(dTp*Jp*eps_nucl)[:,np.newaxis]*d_sigma_g, axis=0)
print(Eg**2*phi*1.60218e-12)
print(Eg)

# Gamma-ray plot compared to Diesing 2023
def plot_gamma():

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(np.log10(Eg),np.log10(Eg**2*phi*1.60218e-12),'r--',linewidth=3.0, label=r'{\rm W28}')

    # Read the image for data    
    img = mpimg.imread("W28.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=8.0
    xmax=13.0
    ymin=-13.0
    ymax=-9.0
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_xticks(np.arange(xmin,xmax+1,1))
    ax.set_yticks(np.arange(ymin,ymax,1))

    # ax.set_ylim(-13.0,np.log10(5.0e-9))

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_aspect(0.8)
    ax.set_xlabel(r'$E_\gamma\, {\rm (GeV)}$',fontsize=fs)
    ax.set_ylabel(r'$E_\gamma^2\phi(E_\gamma) \, ({\rm erg\, cm^{-2}\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_gamma.png')

plot_gamma()

# Record the ending time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")