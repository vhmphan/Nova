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
import time

fs=22

# Define a custom tick formatter to display 10^x
def log_tick_formatter(x, pos):
    return r'$10^{%d}$' % int(x)


kB=8.6173324e-5 # eV/K
me=0.510998e6 # eV
sigma_sb=3.5394474508e7 # erg cm^-2 s^-1 K^-4
mp=938.272e6 # eV
mpCGS=1.67262192e-24 # g
qe=1.602176634e-19 # SI unit
qeCGS=4.8032e-10 # CGS unit 

# Nova shock speed
def func_vsh(pars_nova, t):
# vsh0 (km/s), tST (day), and t(day)

    vsh0=pars_nova[0] # km/s
    tST=pars_nova[1] # day
    alpha=pars_nova[2]
    ter=pars_nova[9] # day
    t=np.array(t)

    mask1=(t>=ter) & (t<tST) 
    mask2=(t>=tST)

    vsh=np.zeros_like(t)

    vsh[mask1]=vsh0 
    vsh[mask2]=vsh0*pow(t[mask2]/tST,-alpha) 

    return vsh # km/s

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
    Rsh[mask2]=-vsh0*ter+vsh0*tST*(pow((t[mask2]-ter)/tST,1.0-alpha)-alpha)/(1.0-alpha) 

    return Rsh*86400.0*6.68e-9 # au


# Density profile of the red giant wind
def func_rho(pars_nova, r):
# Mdot (Msol/yr), vwind (km/s), and r (au)    

    Mdot=pars_nova[3]*1.989e33/(365.0*86400.0) # g/s 
    vwind=pars_nova[4]*1.0e5 # cm/s
    Rmin=pars_nova[5]*1.496e13 # cm
    r*=1.496e13 # cm
    r=np.array(r)

    rho=Mdot/(4.0*np.pi*vwind*pow(Rmin+r,2)) 

    return rho # g/cm^3

# Swept-up mass of the nova shock
def func_MSU(pars_nova, t):
# Mdot (Msol/yr) and vwind (km/s)

    Rsh=func_Rsh(pars_nova,t)
    Mdot=pars_nova[3] # Msol/yr 
    vwind=pars_nova[4] # km/s 
    Rmin=pars_nova[5] # au 
    ter=pars_nova[9] # day
    t=np.array(t)

    mask=(t>ter)
    MSU=np.zeros_like(t)

    MSU[mask]=Mdot*(1.989e33/(365.0*86400.0))*(Rsh[mask]-Rmin*atan(Rsh[mask]/Rmin))*1.496e13/(vwind*1.0e5)

    return MSU # g

def func_Emax(pars_nova, t):
    tST=pars_nova[1] # day
    Rmin=pars_nova[5]*1.496e13 # cm
    xip=pars_nova[6] 
    BRG=pars_nova[10] # G

    vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    rho=func_rho(pars_nova,Rsh/1.496e13) # g/cm^3

    B2_bkgr=BRG*np.power(np.sqrt(Rmin*Rmin+Rsh*Rsh)/(0.35*1.496e13),-2)  # Background magnetic field in G
    # B2_Bell = np.sqrt(11.0 * np.pi * rho * np.power(vsh * xip, 2))  # Bell magnetic field strength
    B2=B2_bkgr  # + B2_Bell * func_Heaviside(arr_t - tST)  # Assuming func_Heaviside is defined and vectorized

    dt=t[1]-t[0]  
    dEdt=(qeCGS*B2*np.power(vsh,2))*6.242e+11/(2.0*np.pi*3.0e10)
    Emax=np.cumsum(dEdt*dt*86400.0)

    return Emax

# Acceleration rate of protons
def func_dE_acc(t, E, pars_nova):
    tST=pars_nova[1] # day
    Rmin=pars_nova[5]*1.496e13 # cm
    xip=pars_nova[6] 
    BRG=pars_nova[10] # G

    vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    rho=func_rho(pars_nova,Rsh/1.496e13) # g/cm^3

    B2_bkgr=BRG*np.power(np.sqrt(Rmin*Rmin+Rsh*Rsh)/(0.35*1.496e13),-2)  # Background magnetic field in G
    # B2_Bell = np.sqrt(11.0 * np.pi * rho * np.power(vsh * xip, 2))  # Bell magnetic field strength
    B2=B2_bkgr  # + B2_Bell * func_Heaviside(arr_t - tST)  # Assuming func_Heaviside is defined and vectorized

    dEdt=(qeCGS*B2*np.power(vsh,2))*6.242e+11/(2.0*np.pi*3.0e10)

    return dEdt # eV/s

# Cumulative spectrum of accelerated protons
def func_NEp_p(pars_nova, E, t):

    xip=pars_nova[6] 
    delta=pars_nova[7] 
    epsilon=pars_nova[8] 
    ter=pars_nova[9] 

    sol=solve_ivp(lambda tp,Emax:func_dE_acc(tp,Emax,pars_nova),(t[0],t[-1]),[0.0],method='RK45',dense_output=True)
    Emax=((sol.sol(t)[0]).T*86400.0)[np.newaxis,:]

    # Get the nomalization for the accelerated spectrum
    xmin=np.sqrt(pow(E[0]+mp,2)-mp*mp)/mp 
    xmax=np.sqrt(pow(E[-1]+mp,2)-mp*mp)/mp
    x=np.logspace(np.log10(xmin),np.log10(xmax),5000)

    dx=(x[1:-1]-x[0:-2])[:,np.newaxis]
    x=x[0:-2][:,np.newaxis]
    Ialpha_p=np.sum(pow(x,4.0-delta)*np.exp(-pow(x*mp/Emax,epsilon))*dx/np.sqrt(1.0+x*x),axis=0)

    # Get the momentum and speed 
    p=np.sqrt(pow(E+mp,2)-mp*mp)
    vp=(p/(E+mp))
    NEp=np.zeros((len(E),len(t)))

    # Get all the shock dynamics related quantities
    vsh=(func_vsh(pars_nova,t)*1.0e5) # cm/s
    Rsh=(func_Rsh(pars_nova,t)*1.496e13) # cm
    rho=func_rho(pars_nova,Rsh/1.496e13) # g/cm^3

    # Change the dimension to make the integral    
    p=p[:,np.newaxis]
    vp=vp[:,np.newaxis]
    Rsh=Rsh[np.newaxis,:]
    vsh=vsh[np.newaxis,:]
    rho=rho[np.newaxis,:]
    Ialpha_p=Ialpha_p[np.newaxis,:]

    dt=(t[1]-t[0])*86400.0
    NEp=np.nancumsum(3.0*np.pi*xip*rho*pow(Rsh,2)*pow(vsh,3)*6.242e11*pow(p/mp,2.0-delta)*np.exp(-pow(p/Emax,epsilon))/(mp*mp*vp*Ialpha_p),axis=1)*dt 
    print(np.nancumsum(3.0*np.pi*xip*rho*pow(Rsh,2)*pow(vsh,3)*6.242e11*pow(p/mp,2.0-delta)*np.exp(-pow(p/Emax,epsilon))/(mp*mp*vp*Ialpha_p), axis=1))

    return NEp # eV^-1 

# Plot the cosmic-ray distribution
def plot_fEp(NEp, E, t):

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    ax.plot(np.log10(E*1.0e-9),np.log10((E**3*NEp[:,100]))-18,'k:',linewidth=5.0, label=r'{\rm t=1\, day}')
    print(np.log10((E**3*NEp[:,100]))-18)
    filename='fEp.dat'
    t, E, fEp=np.loadtxt(filename,unpack=True,usecols=[0,1,2])
    Nt=151
    NfEp=len(fEp)
    t=np.reshape(t, (Nt, int(NfEp/Nt)))
    E=np.reshape(E, (Nt, int(NfEp/Nt)))
    fEp=np.reshape(fEp, (Nt, int(NfEp/Nt)))

    # ax.plot(np.log10(E[5,:]*1.0e-9),np.log10((E[5,:])**3*fEp[5,:])-18,'m--',linewidth=3.0, label=r'{\rm t=1\, day}')
    ax.plot(np.log10(E[10,:]*1.0e-9),np.log10((E[10,:])**3*fEp[10,:])-18,'r--',linewidth=3.0, label=r'{\rm t=1\, day}')
    print(t[10,0])
    # ax.plot(np.log10(E[50,:]*1.0e-9),np.log10((E[50,:])**3*fEp[50,:])-18,'g--',linewidth=3.0, label=r'{\rm t=5\, day}')

    # print(np.log10((E[10,:])**3*fEp[10,:])+22)

    # # Read the image for data    
    # img = mpimg.imread("Data/data_fEp.png")
    # img_array = np.mean(np.array(img), axis=2)

    # xmin=0.0
    # xmax=4.0
    # ymin=np.log10(4.0e43)
    # ymax=48
    # ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    # ax.set_ylim(40,50)
    # ax.set_xlim(0,5)

    # ax.set_xticks(np.arange(xmin,xmax+1,1))
    # ax.set_yticks(np.arange(int(ymin)+1,ymax+1,1))

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    # ax.set_aspect(0.8)
    ax.set_xlabel(r'$E\, {\rm (GeV)}$',fontsize=fs)
    ax.set_ylabel(r'$E^3 f(E) \, ({\rm eV\, cm^{-3}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_fEp_py.png')

# Parameters for RS Ophiuchi 2021
pars_nova=np.array([4500.0, 3.0, 0.43, 6.0e-7, 20.0, 1.48, 0.1, 4.4, 1.0, 0.0, 1.0, 1.0e4, 10, 2.0e-9, 1.4e3])

# Record the starting time
start_time = time.time()

# Define the time and energy ranges
# sol=solve_ivp(lambda tp,Emax:func_dE_acc(tp,Emax,pars_nova),(t[0],t[-1]),[0.0],method='RK45',dense_output=True)
# Emax_RK4=sol.sol(t).T*86400.0
t=np.linspace(0,15.0,1501)
E=np.logspace(8,14,601)
# t=np.linspace(0,15.0,2)
# E=np.logspace(8,14,3)
Rsh=func_Rsh(pars_nova,t) # cm
Vsh=np.pi*pow(Rsh*1.496e13,3)/3.0 # cm^3
NEp=func_NEp_p(pars_nova,E,t)/Vsh[np.newaxis,:] # eV^-1 cm^-3
plot_fEp(NEp,E,t)

# Record the ending time
end_time=time.time()

# Calculate the elapsed time
elapsed_time=end_time-start_time

print("Elapsed time:", elapsed_time, "seconds")

# fig=plt.figure(figsize=(10, 8))
# ax=plt.subplot(111)
# ax.plot(t,np.log10(func_Emax(pars_nova,t)*1.0e-12),'r--',linewidth=3.0)
# ax.plot(t,np.log10(Emax_RK4*1.0e-12),'k-.',linewidth=3.0)

# # filename='profile.dat'
# # t, Rsh, Emax, B=np.loadtxt(filename,unpack=True,usecols=[0,2,4,5])
# # ax.plot(t,np.log10(Emax),'g:',linewidth=3.0)

# # # Read the image for data    
# # img = mpimg.imread("Data/data_Emax.png")
# # img_array = np.mean(np.array(img), axis=2)

# # xmin=0.0
# # xmax=5.0
# # ymin=-2.0
# # ymax=1.0
# # ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
# # ax.set_yticks(np.arange(ymin,ymax+1,1))

# # ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

# # ax.legend()
# # ax.set_aspect(1.2)
# # ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
# # ax.set_ylabel(r'$E_{\rm max} \, ({\rm TeV})$',fontsize=fs)
# # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
# #     label_ax.set_fontsize(fs)
# # ax.legend(loc='upper right', prop={"size":fs})
# # ax.grid(linestyle='--')

# # plt.savefig('fg_Emax_py.png')