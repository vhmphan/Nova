from cProfile import label
from cmath import tau
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc("text",usetex=True)
import scipy as sp
import scipy.interpolate
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from matplotlib.ticker import MultipleLocator
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
import gato.pack_gato as gt
import time
from multiprocessing import Pool
import os

fs=22

# Heaviside
def func_Heaviside(x):

    return 0.5*(1.0+np.tanh(10.0*x))

# Optical luminosiy function of the nova
def func_LOPT(t):
# t (day)

    mask=(t==0.25)

    LOPT=2.5e36*(1.0-func_Heaviside(t-0.9))+func_Heaviside(t-0.9)*(1.3e39*pow(abs(t-0.25),-0.28)/(abs(t+0.35)))
    LOPT[mask]=2.5e36

    return LOPT;# erg s^-1

# Nova shock radius
def func_Rsh(pars_nova, t):
# vsh0 (km/s), tST (day), Rmin (au), and t(day)

    vsh0=pars_nova[0] # km/s
    tST=pars_nova[1] # day
    alpha=pars_nova[2]
    Rmin=pars_nova[5] # au
    ter=pars_nova[9] # day
    t=np.array(t)

    mask1=(t>=ter) & (t<tST)
    mask2=(t>=tST)

    Rsh=np.zeros_like(t)

    Rsh[mask1]=vsh0*(t[mask1]-ter) 
    Rsh[mask2]=-vsh0*ter+vsh0*tST*(pow(t[mask2]/tST,1.0-alpha)-alpha)/(1.0-alpha) 

    return Rsh*86400.0*6.68e-9 # au

def func_uOPT_rt(pars_nova, r, t):

    LOPT=func_LOPT(t)
    uOPT=LOPT/(4.0*np.pi*(r*1.496e13)**2*3.0e10)
    
    Rsh=func_Rsh(pars_nova,t)
    uOPT*=1.5*(r/Rsh)**3*((Rsh/r)+((Rsh/r)**2-1.0)*np.log(np.sqrt(abs((r+Rsh)/(r-Rsh)))))

    return uOPT # erg/cm^3

def func_uOPT_r2(pars_nova, r, t):

    LOPT=func_LOPT(t)
    uOPT=LOPT/(4.0*np.pi*(r*1.496e13)**2*3.0e10)
    
    return uOPT # erg/cm^3


pars_nova=[4500.0, 2.0, 0.66, 2.0e-7, 20.0, 1.48, 0.1, 4.4, 1.0, 0.0, 1.0, 0.8e4, 10, 2.0e-9, 1.4e3, 'DM23', 50]

fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

r=np.logspace(-1,2,100)
t=np.array([1.6])
Rsh=func_Rsh(pars_nova,t)
print(Rsh)

ax.plot(r,func_uOPT_r2(pars_nova,r,t),'k-',linewidth=3.0,label=r'$1/r^2$')
ax.plot(r,func_uOPT_rt(pars_nova,r,t),'r--',linewidth=3.0,label=r'${\rm Eq.\,A.3}$')

ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(3.0e-1,1.0e2)
ax.set_ylim(1.0e-3,1.0e2)
ax.set_xlabel(r'$r\, {\rm (au)}$',fontsize=fs)
ax.set_ylabel(r'$u_{\rm opt} \, ({\rm erg\,cm^{-3}})$',fontsize=fs)
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)
ax.legend(loc='upper right', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig('fg_uOPT.png')
