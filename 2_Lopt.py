# from cProfile import label
# from cmath import tau
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rc("text",usetex=True)
# import scipy as sp
# import scipy.interpolate
# from scipy.integrate import solve_ivp
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
# from matplotlib.ticker import MultipleLocator
# import matplotlib.image as mpimg
# from matplotlib.ticker import FuncFormatter
# from scipy.optimize import curve_fit
# import gato.pack_gato as gt
# import time
# from multiprocessing import Pool
# import os

# fs=22

# # Heaviside
# def func_Heaviside(x):

#     return 0.5*(1.0+np.tanh(10.0*x))

# # Optical luminosiy function of the nova
# def func_LOPT(t):
# # t (day)

#     mask=(t==0.25)

#     LOPT=2.5e36*(1.0-func_Heaviside(t-0.9))+func_Heaviside(t-0.9)*(1.3e39*pow(abs(t-0.25),-0.28)/(abs(t+0.35)))
#     LOPT[mask]=2.5e36

#     return LOPT;# erg s^-1

# # Nova shock radius
# def func_Rsh(pars_nova, t):
# # vsh0 (km/s), tST (day), Rmin (au), and t(day)

#     vsh0=pars_nova[0] # km/s
#     tST=pars_nova[1] # day
#     alpha=pars_nova[2]
#     Rmin=pars_nova[5] # au
#     ter=pars_nova[9] # day
#     t=np.array(t)

#     mask1=(t>=ter) & (t<tST)
#     mask2=(t>=tST)

#     Rsh=np.zeros_like(t)

#     Rsh[mask1]=vsh0*(t[mask1]-ter) 
#     Rsh[mask2]=-vsh0*ter+vsh0*tST*(pow(t[mask2]/tST,1.0-alpha)-alpha)/(1.0-alpha) 

#     return Rsh*86400.0*6.68e-9 # au

# # Nova shock speed
# def func_vsh(pars_nova, t):
# # vsh0 (km/s), tST (day), and t(day)

#     vsh0=pars_nova[0] # km/s
#     tST=pars_nova[1] # day
#     alpha=pars_nova[2]
#     ter=pars_nova[9] # day
#     t=np.array(t)

#     mask1=(t>=ter) & (t<tST) 
#     mask2=(t>=tST)

#     vsh=np.zeros_like(t)

#     vsh[mask1]=vsh0 
#     vsh[mask2]=vsh0*pow(t[mask2]/tST,-alpha)

#     return vsh # km/s

# # Density profile of the red giant wind
# def func_rho(pars_nova, r):
# # Mdot (Msol/yr), vwind (km/s), and r (au)    

#     Mdot=pars_nova[3]*1.989e33/(365.0*86400.0) # g/s 
#     vwind=pars_nova[4]*1.0e5 # cm/s
#     Rmin=pars_nova[5]*1.496e13 # cm
#     r=np.array(r)

#     rho=Mdot/(4.0*np.pi*vwind*pow(Rmin+r*1.496e13,2)) 

#     return rho # g/cm^3

# def func_Lsh(pars_nova, t):
#     vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
#     Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
#     rho=func_rho(pars_nova,Rsh/1.496e13) # g/cm^3

#     return 4.0*np.pi*Rsh**2*rho*vsh**3 # erg/s

# def func_gopt(x):
#     return (1.0/x)+((1.0/x**2)-1.0)*np.log(np.sqrt(np.abs((x+1.0)/(x-1.0))))

# def func_uOPT_rt(pars_nova, r, eta, t):

#     LOPT=func_LOPT(t)
#     uOPT=LOPT/(4.0*np.pi*(r*1.496e13)**2*3.0e10)
    
#     Rsh=func_Rsh(pars_nova,t)
#     # uOPT*=1.5*(r/Rsh)**3*((Rsh/r)+((Rsh/r)**2-1.0)*np.log(np.sqrt(abs((r+Rsh)/(r-Rsh)))))
#     uOPT*=1.5*(r/Rsh)**3*(func_gopt(r/Rsh)-func_gopt(r/(eta*Rsh)))/(1.0-eta**3)

#     return uOPT # erg/cm^3

# def func_uOPT_r2(pars_nova, r, t):

#     LOPT=func_LOPT(t)
#     uOPT=LOPT/(4.0*np.pi*(r*1.496e13)**2*3.0e10)
    
#     return uOPT # erg/cm^3


# pars_nova=[4500.0, 2.0, 0.66, 5.0e-7, 20.0, 1.48, 0.1, 4.4, 1.0, 0.0, 1.0, 0.8e4, 10, 2.0e-9, 1.4e3, 'DM23', 50]

# fig=plt.figure(figsize=(10, 8))
# ax=plt.subplot(111)

# r=np.logspace(-1,2,100)
# t=np.array([1.6])
# Rsh=func_Rsh(pars_nova,t)
# print(Rsh)

# ax.plot(r,func_uOPT_r2(pars_nova,r,np.array([1.6])),'r-',linewidth=3.0)
# ax.plot(r,func_uOPT_r2(pars_nova,r,np.array([3.6])),'g-',linewidth=3.0)
# ax.plot(r,func_uOPT_r2(pars_nova,r,np.array([5.6])),'-', color='orange', linewidth=3.0)
# # ax.plot(r,func_uOPT_rt(pars_nova,r,0.75,t),'r--',linewidth=3.0,label=r'${\rm Eq.\,14}$')
# ax.plot(r,func_uOPT_rt(pars_nova,r,0.75,t),'r--',linewidth=5.0, label=r'${\rm Day\, 1}$')
# ax.plot(r,func_uOPT_rt(pars_nova,r,0.75,np.array([3.6])),'g--',linewidth=5.0, label=r'${\rm Day\, 3}$')
# ax.plot(r,func_uOPT_rt(pars_nova,r,0.75,np.array([5.6])),'--', color='orange',linewidth=5.0, label=r'${\rm Day\, 5}$')

# ax.legend()
# # ax.set_xscale('log')
# ax.set_yscale('log')
# # ax.set_xlim(3.0e-1,1.0e2)
# # ax.set_ylim(1.0e-3,1.0e2)

# ax.set_xticks([0, 5, 10, 15, 20])


# ax.set_xlim(0.0,20.0)
# ax.set_ylim(1.0e-3,2.0e0)
# ax.set_xlabel(r'$r\, {\rm (au)}$',fontsize=fs)
# ax.set_ylabel(r'$u_{\rm opt} \, ({\rm erg\,cm^{-3}})$',fontsize=fs)
# for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
#     label_ax.set_fontsize(fs)
# ax.legend(loc='upper right', prop={"size":fs})
# ax.grid(linestyle='--')

# plt.savefig('Results_jax/fg_uOPT.png')


# t_data, LOPT_data=np.loadtxt("Data/LOPT.dat",unpack=True,usecols=[0,1])
# # LOPT_data=np.log10(LOPT_data)

# fig=plt.figure(figsize=(10, 8))
# ax=plt.subplot(111)

# t=np.linspace(-2,30,100)

# ax.plot(t_data,LOPT_data,'ks',linewidth=3.0,label=r'${\rm Cheung\, 2022}$')
# # ax.plot(t,func_Lsh(pars_nova,t),'k-',linewidth=3.0,label=r'${\rm Shock}$')
# ax.plot(t,func_LOPT(t),'r--',linewidth=3.0,label=r'${\rm Optical}$')

# ax.legend()
# # ax.set_xscale('log')
# ax.set_yscale('log')
# # ax.set_xlim(3.0e-1,1.0e2)
# # ax.set_ylim(1.0e-3,1.0e2)
# ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
# ax.set_ylabel(r'$L_{\rm opt} \, ({\rm erg\,s^{-1}})$',fontsize=fs)
# for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
#     label_ax.set_fontsize(fs)
# ax.legend(loc='upper right', prop={"size":fs})
# ax.grid(linestyle='--')

# plt.savefig('fg_LOPT.png')

# sigma_SB=5.67e-5 # erg cm^-2 s^-1 K^-4
# Rs=200.0*6.95e10
# Ts=10780.0
# print(4.0*Rs**2*sigma_SB*(Ts)**4)

# import gato.pack_nova as nv



# pars_init=[4500.0, 2.0, 0.66, 2.0e-7, 20.0, 1.48, 0.1, 4.4, 1.0, 0.0, 1.0, 1.0e4, 10, 2.0e-9, 1.4e3, 'DM23', 50]
# tST=np.array([1.8]) 
# Mdot=np.array([5.0e-7])
# ter=np.array([-0.2]) 
# BRG=np.array([1.5]) 

# pars_nova=pars_init.copy()
# pars_nova[1]=tST
# pars_nova[3]=Mdot
# pars_nova[9]=ter
# pars_nova[10]=BRG

# t=np.linspace(0,20,100)

# nv.plot_Emax(pars_nova,t)
# nv.plot_vsh(pars_nova,t)
# nv.plot_Rsh(pars_nova,t)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

# Model function
def model_function(t, A, alpha):
    # Smooth masks using tanh functions
    smoothing_factor=20.0
    mask=0.5*(1.0+jnp.tanh(smoothing_factor*(t-0.9)))

    LOPT=2.5e36+mask*A*(t/2.0)**(-alpha*mask)

    return LOPT #A * (t/2.0)**(-alpha)

def func_LOPT(t):

    # Smooth masks using tanh functions
    smoothing_factor=20.0
    mask1=0.5*(1.0+jnp.tanh(smoothing_factor*(t-1.0)))
    mask2=0.5*(1.0+jnp.tanh(100.0*smoothing_factor*(t-0.7)))

    # Luminosity function fitted with data from Cheung et al. 2022
    LOPT=2.5e36+mask1*3.9e38*(t/2.0)**(-0.967*mask1)

    return LOPT

# Example data (replace these with your actual data)
t_i, y_i=np.loadtxt("Data/LOPT.dat",unpack=True,usecols=[0,1])

t_i+=-0.25
mask=(t_i>0.0) & (t_i<10.0)
t_i=t_i[mask]
y_i=y_i[mask]

# Curve fitting
initial_A, initial_alpha = 3.9e38, 0.99
params, covariance = curve_fit(model_function, t_i, (y_i), p0=[initial_A, initial_alpha])

# Extracting the parameters
A, alpha = params
print(f"Fitted Parameters: A = {A}, alpha = {alpha}")

# Plotting the results
plt.scatter(t_i, y_i, label='Data')
t_fit = np.linspace(min(t_i), max(t_i), 100)
y_fit = model_function(t_fit, A, alpha)
plt.plot(t_fit, y_fit, label='Fit: $A \\cdot t^{-\\alpha}$', color='red')
t=np.linspace(-5,30,3501)
plt.plot(t, func_LOPT(t), label='LOPT', color='blue')
# print(func_LOPT(1.5))
# print(model_function(1.5, 3.808004757578125e+38, 0.9669400087874354))
# print(0.5*(1.0+jnp.tanh(500.0*1.5)))

plt.xlabel('t')
plt.ylabel('y')
plt.xlim(-5,30)
plt.ylim(1.0e36,1.0e40)
plt.yscale('log')
plt.legend()
plt.show()

