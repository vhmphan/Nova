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

mp=gt.mp
mpCGS=gt.mpCGS
kB=gt.kB

# Prepare data from HESS and FERMI
t_HESS_raw, flux_HESS_raw=np.loadtxt('Data/data_time_gamma_HESS_raw.dat',unpack=True,usecols=[0,1])
t_HESS_raw=t_HESS_raw-0.25 # Data are chosen at different time orgin than model
t_HESS_raw=t_HESS_raw.reshape((int(len(flux_HESS_raw)/5),5))
flux_HESS_raw=flux_HESS_raw.reshape((int(len(flux_HESS_raw)/5),5))
xerr_HESS_raw=t_HESS_raw[:,0]-t_HESS_raw[:,1]
yerr_HESS_raw=flux_HESS_raw[:,0]-flux_HESS_raw[:,3]
t_HESS_raw=t_HESS_raw[:,0]
flux_HESS_raw=flux_HESS_raw[:,0]
# xerr_HESS_raw=np.array([t_HESS_raw[:,0]-t_HESS_raw[:,1],t_HESS_raw[:,2]-t_HESS_raw[:,0]])
# yerr_HESS_raw=np.array([flux_HESS_raw[:,0]-flux_HESS_raw[:,3],flux_HESS_raw[:,4]-flux_HESS_raw[:,0]])

t_FERMI_raw, flux_FERMI_raw=np.loadtxt('Data/data_time_gamma_FERMI_raw.dat',unpack=True,usecols=[0,1])
t_FERMI_raw=t_FERMI_raw-0.25 # Data are chosen at different time orgin than model
t_FERMI_raw=t_FERMI_raw.reshape((int(len(flux_FERMI_raw)/5),5))
flux_FERMI_raw=flux_FERMI_raw.reshape((int(len(flux_FERMI_raw)/5),5))
xerr_FERMI_raw=t_FERMI_raw[:,0]-t_FERMI_raw[:,1]
yerr_FERMI_raw=flux_FERMI_raw[:,0]-flux_FERMI_raw[:,3]
t_FERMI_raw=t_FERMI_raw[:,0]
flux_FERMI_raw=flux_FERMI_raw[:,0]
# xerr_FERMI_raw=np.array([t_FERMI_raw[:,0]-t_FERMI_raw[:,1],t_FERMI_raw[:,2]-t_FERMI_raw[:,0]])
# yerr_FERMI_raw=np.array([flux_FERMI_raw[:,0]-flux_FERMI_raw[:,3],flux_FERMI_raw[:,4]-flux_FERMI_raw[:,0]])


# Define a custom tick formatter to display 10^x
def log_tick_formatter(x, pos):
    return r'$10^{%d}$' % int(x)


# Function for interpolation
def log_interp1d(xx, yy, kind='linear'):

    logx=np.log10(xx)
    logy=np.log10(yy)
    lin_interp=sp.interpolate.interp1d(logx,logy,kind=kind)
    log_interp=lambda zz: np.power(10.0,lin_interp(np.log10(zz)))

    return log_interp

# Smooth Heaviside function
def func_Heaviside(x):
    return 0.5*(1+np.tanh(10*x))


# Optical luminosiy function of the nova
def func_LOPT(t):
# t (day)

    mask=(t==0.25)

    LOPT=2.5e36*(1.0-func_Heaviside(t-0.9))+func_Heaviside(t-0.9)*(1.3e39*pow(abs(t-0.25),-0.28)/(abs(t+0.35)))
    LOPT[mask]=2.5e36

    return LOPT;# erg s^-1


# Optical luminosity function as modelled from Metzger et al. 2014
def func_LOPT_M14(pars_nova, t):

    # In this model by Metzger et al. 2014, the optical emission is from the thermal X-ray in the postshock region
    # being reprocessed by neutral gas ahead of the shock. We model the optical luminosity using Eq. 51 and 53. But for
    # RS Ophiuchi, a large fraction ~50% of the shock power is required to be converted into optical photons for this model to work. 
    vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    rho=func_rho(pars_nova,Rsh/1.496e13) # g/cm^3    
    Lsh=4.0*np.pi*Rsh**2*rho*vsh**3 # erg s^-1
    XZ=0.1
    Tsh=1.0e7
    tau_X=2.5e2*(XZ/0.1)*(kB*Tsh/1.0e3)**-1.5*(rho/(mpCGS*1.0e10))*(Rsh/1.0e14)
    LOPT=2.5e36+0.5*Lsh*(1.0-np.exp(-tau_X))

    return LOPT


# Plot luminosity function of the optical emission
def plot_LOPT(pars_nova):

    t=np.linspace(-20,60,810)
    vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    rho=func_rho(pars_nova,Rsh/1.496e13) # g/cm^3   
    Lsh=4.0*np.pi*Rsh**2*rho*vsh**3 # erg s^-1

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,np.log10(func_LOPT(t)),'r:',linewidth=3.0, label=r'{\rm Fit}')
    ax.plot(t,np.log10(Lsh),'g--',linewidth=3.0, label=r'{\rm Shock Luminosity}')
    # ax.plot(t,np.log10(func_LOPT_M14(pars_nova,t)),'g--',linewidth=3.0, label=r'{\rm Metzger}')

    # Read the image for data
    img = mpimg.imread("Data/data_LOPT.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=-19.75
    xmax=60.25
    ymin=np.log10(2.0e36)
    ymax=39.0
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_yticks([37, 38, 39])

    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_aspect(20)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$L_{\rm opt} \, ({\rm erg\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_LOPT_py.png')


# Nova shock speed and radius from HESS paper
t_HESS_model, vsh_HESS_model=np.loadtxt('vsh_HESS.dat',unpack=True,usecols=[0,1])
func_vsh_HESS=sp.interpolate.interp1d(t_HESS_model,vsh_HESS_model,kind='linear',fill_value='extrapolate')

Rsh_HESS_model=np.zeros_like(t_HESS_model)
for i in range(1,len(t_HESS_model)):
    t_HESS=np.linspace(t_HESS_model[0],t_HESS_model[i],1000)
    Rsh_HESS_model[i]=np.sum(func_vsh_HESS(t_HESS))*(t_HESS[1]-t_HESS[0]) # km s^-1 day

func_Rsh_HESS=sp.interpolate.interp1d(t_HESS_model,Rsh_HESS_model,kind='linear',fill_value='extrapolate')


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

    if(pars_nova[15]=='HESS'):
        vsh=func_vsh_HESS(t) 

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
    Rsh[mask2]=-vsh0*ter+vsh0*tST*(pow(t[mask2]/tST,1.0-alpha)-alpha)/(1.0-alpha) 

    if(pars_nova[15]=='HESS'):
        Rsh=func_Rsh_HESS(t)

    return Rsh*86400.0*6.68e-9 # au


# Density profile of the red giant wind
def func_rho(pars_nova, r):
# Mdot (Msol/yr), vwind (km/s), and r (au)    

    Mdot=pars_nova[3]*1.989e33/(365.0*86400.0) # g/s 
    vwind=pars_nova[4]*1.0e5 # cm/s
    Rmin=pars_nova[5]*1.496e13 # cm
    r=np.array(r)

    rho=Mdot/(4.0*np.pi*vwind*pow(Rmin+r*1.496e13,2)) 

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

    MSU[mask]=Mdot*(1.989e33/(365.0*86400.0))*(Rsh[mask]-Rmin*np.arctan(Rsh[mask]/Rmin))*1.496e13/(vwind*1.0e5)

    return MSU # g


# Acceleration rate of protons
def func_dE_acc(pars_nova, E, t):
    tST=pars_nova[1] # day
    Rmin=pars_nova[5]*1.496e13 # cm
    xip=pars_nova[6] 
    BRG=pars_nova[10] # G

    vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    rho=func_rho(pars_nova,Rsh/1.496e13) # g/cm^3

    B2_bkgr=BRG*np.power(np.sqrt(Rmin*Rmin+Rsh*Rsh)/(0.35*1.496e13),-2)  # Background magnetic field in G
    # B2_Bell=np.sqrt(11.0*np.pi*rho*np.power(vsh*xip,2))  # Bell magnetic field strength
    B2=B2_bkgr  # + B2_Bell * func_Heaviside(arr_t - tST)  # Assuming func_Heaviside is defined and vectorized

    dEdt=(gt.qeCGS*B2*np.power(vsh,2))*6.242e+11/(2.0*np.pi*3.0e10)

    return dEdt # eV/s

# Plot Emax of protons as a function of time
def plot_Emax(Emax, t):
    
    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,np.log10(Emax*1.0e-12),'k-.',linewidth=3.0)

    filename='profile.dat'
    t, Rsh, Emax, B=np.loadtxt(filename,unpack=True,usecols=[0,2,4,5])
    ax.plot(t,np.log10(Emax),'g:',linewidth=3.0)

    # Read the image for data    
    img = mpimg.imread("Data/data_Emax.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=5.0
    ymin=-2.0
    ymax=1.0
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_yticks(np.arange(ymin,ymax+1,1))

    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_aspect(1.2)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$E_{\rm max} \, ({\rm TeV})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_Emax_py.png')

# Cumulative spectrum of accelerated protons
def func_JEp_p(pars_nova, E, t):

    xip=pars_nova[6] 
    delta=pars_nova[7] 
    epsilon=pars_nova[8] 
    ter=pars_nova[9] 

    # Solve for the maximum energy over time
    sol=solve_ivp(lambda tp,Emax:func_dE_acc(pars_nova,Emax,tp),(t[0],t[-1]),[0.0],method='RK45',dense_output=True)
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
    vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
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

    return NEp*vp*3.0e10 # eV^-1 cm s^-1


# Plot the cosmic-ray distribution
def plot_fEp(NEp, E, t):

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    it_plot=np.array([100, 500])

    for i in it_plot:
        ax.plot(np.log10(E*1.0e-9),np.log10((E**3*NEp[:,i]))+24,'k:',linewidth=5.0, label=r'{\rm t=%.1f\, day}' % t[i])

    filename='fEp.dat'
    t, E, fEp=np.loadtxt(filename,unpack=True,usecols=[0,1,2])
    Nt=151
    NfEp=len(fEp)
    t=np.reshape(t, (Nt, int(NfEp/Nt)))
    E=np.reshape(E, (Nt, int(NfEp/Nt)))
    fEp=np.reshape(fEp, (Nt, int(NfEp/Nt)))

    # ax.plot(np.log10(E[5,:]*1.0e-9),np.log10((E[5,:])**3*fEp[5,:])-18,'m--',linewidth=3.0, label=r'{\rm t=1\, day}')
    ax.plot(np.log10(E[10,:]*1.0e-9),np.log10((E[10,:])**3*fEp[10,:])+24,'r--',linewidth=3.0, label=r'{\rm t=1\, day}')
    ax.plot(np.log10(E[50,:]*1.0e-9),np.log10((E[50,:])**3*fEp[50,:])+24,'g--',linewidth=3.0, label=r'{\rm t=5\, day}')

    # Read the image for data    
    img = mpimg.imread("Data/data_fEp.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=4.0
    ymin=np.log10(4.0e43)
    ymax=48
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_ylim(44,48)
    ax.set_xlim(0,4)

    # ax.set_xticks(np.arange(xmin,xmax+1,1))
    # ax.set_yticks(np.arange(int(ymin)+1,ymax+1,1))

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_aspect(0.8)
    ax.set_xlabel(r'$E\, {\rm (GeV)}$',fontsize=fs)
    ax.set_ylabel(r'$E^3 f(E) \, ({\rm eV\, cm^{-3}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_fEp_py.png')


# #  Plot the gamma-ray attenuation
# def plot_abs():

#     filenameb='gamma.dat'
#     t, E, phi, phi_abs, tau_gg=np.loadtxt(filenameb,unpack=True,usecols=[0,1,2,3,4])
#     Nphi=len(phi)
#     t=np.reshape(t, (Nt, int(Nphi/Nt)))
#     E=np.reshape(E, (Nt, int(Nphi/Nt)))
#     phi=np.reshape(phi, (Nt, int(Nphi/Nt)))
#     phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))
#     tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

#     print(E[0,500],'GeV')

#     fig=plt.figure(figsize=(10, 8))
#     ax=plt.subplot(111)
#     # ax.plot(t[:,84],np.log10(tau_gg[:,84]),'r-',linewidth=3.0, label=r'$E=0.3\,{\rm TeV}$')
#     # ax.plot(t[:,84],np.log10(np.exp(-tau_gg[:,84])),'r--',linewidth=3.0)
#     # ax.plot(t[:,91],np.log10(tau_gg[:,91]),'g-',linewidth=3.0, label=r'$E=0.6\,{\rm TeV}$')
#     # ax.plot(t[:,91],np.log10(np.exp(-tau_gg[:,91])),'g--',linewidth=3.0)
#     ax.plot(t[:,500],np.log10(tau_gg[:,500]),'r-',linewidth=3.0, label=r'$E=1\,{\rm TeV}$')
#     ax.plot(t[:,500],np.log10(np.exp(-tau_gg[:,500])),'r--',linewidth=3.0)

#     # Read the image for data
#     img = mpimg.imread("Data/data_abs.png")
#     img_array = np.mean(np.array(img), axis=2)

#     xmin=0.0
#     xmax=10.0
#     ymin=np.log10(0.02)
#     ymax=np.log10(2.0)
#     ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
#     ax.set_yticks([-1, 0])

#     ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
#     ax.yaxis.set_minor_locator(MultipleLocator(5))

#     ax.legend()
#     ax.set_aspect(4)
#     ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
#     ax.set_ylabel(r'$\tau_{\gamma\gamma}$',fontsize=fs)
#     for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
#         label_ax.set_fontsize(fs)
#     ax.legend(loc='lower right', prop={"size":fs})
#     ax.grid(linestyle='--')

#     plt.savefig('fg_abs.png')


# Plot gamma-ray data
def plot_gamma(pars_nova, phi_PPI, tau_gg, Eg, t, t_day):

    it=t_day
    t_day=t[t_day]

    # filename='gamma.dat'
    # t, E, phi, phi_abs, tau_gg=np.loadtxt(filename,unpack=True,usecols=[0,1,2,3,4])
    # Nphi=len(phi)
    # Nt=151
    scale_t=10
    # t=np.reshape(t, (Nt, int(Nphi/Nt)))
    # E=np.reshape(E, (Nt, int(Nphi/Nt)))
    # phi=np.reshape(phi, (Nt, int(Nphi/Nt)))
    # phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))
    # tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    print(phi_PPI.shape)
    print(tau_gg.shape)

    # print("Day",t[int(scale_t*t_day),0])
    # ax.plot(np.log10(E[int(scale_t*t_day),:]),np.log10(phi[int(scale_t*t_day),:]),'r-',linewidth=3.0)
    # ax.plot(np.log10(E[int(scale_t*t_day),:]),np.log10(phi_abs[int(scale_t*t_day),:]),'r--',linewidth=3.0, label=r'{\rm t=%.1f\, day}' % t_day)

    ax.plot(np.log10(Eg*1.0e-9),np.log10(Eg**2*phi_PPI[:,it]*1.6022e-12),'g-',linewidth=5.0)
    ax.plot(np.log10(Eg*1.0e-9),np.log10(Eg**2*phi_PPI[:,it]*np.exp(-tau_gg[:,it])*1.6022e-12),'g:',linewidth=5.0)

    # Read the image for data    
    img = mpimg.imread("Data/data_day%d.png" % t_day)
    img_array = np.mean(np.array(img), axis=2)

    xmin=-1.0
    xmax=4.0
    ymin=-13.0
    ymax=np.log10(5.0e-9)
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_xticks(np.arange(xmin,xmax+1,1))
    ax.set_yticks(np.arange(ymin,ymax,1))

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_aspect(0.8)
    ax.set_xlabel(r'$E_\gamma\, {\rm (GeV)}$',fontsize=fs)
    ax.set_ylabel(r'$E_\gamma^2\phi(E_\gamma) \, ({\rm erg\, cm^{-2}\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_gamma_day%d_%s.png' % (t_day, pars_nova[15]))


# # Plot time profile of gamma-ray integrated flux
# def plot_time_gamma(pars_nova, phi_PPI, Eg, t):

#     fig=plt.figure(figsize=(10, 8))
#     ax=plt.subplot(111)

#     dlogEg=np.log10(Eg[1]/Eg[0])

#     jmin_FLAT=int(np.log10(0.1e9/Eg[0])/dlogEg)
#     jmax_FLAT=int(np.log10(100.0e9/Eg[0])/dlogEg)
#     jmin_HESS=int(np.log10(250.0e9/Eg[0])/dlogEg)
#     jmax_HESS=int(np.log10(2500.0e9/Eg[0])/dlogEg)

#     print("FERMI band: ",Eg[jmin_FLAT]*1.0e-9,"-",Eg[jmax_FLAT]*1.0e-9,"GeV")
#     print("HESS band:  ",Eg[jmin_HESS]*1.0e-9,"-",Eg[jmax_HESS]*1.0e-9,"GeV")

#     phi_PPI*=Eg[:,np.newaxis]*1.60218e-12
#     flux_FLAT_PPI=np.nansum((Eg[jmin_FLAT+1:jmax_FLAT+1,np.newaxis]-Eg[jmin_FLAT:jmax_FLAT,np.newaxis])*phi_PPI[jmin_FLAT:jmax_FLAT,:],axis=0)
#     flux_HESS_PPI=np.nansum((Eg[jmin_HESS+1:jmax_HESS+1,np.newaxis]-Eg[jmin_HESS:jmax_HESS,np.newaxis])*phi_PPI[jmin_HESS:jmax_HESS,:],axis=0)

#     data=np.column_stack((t, flux_FLAT_PPI, flux_HESS_PPI))
#     np.savetxt('flux.txt', data, fmt='%.2e')

#     print(phi_PPI[100,25])
#     print(flux_FLAT_PPI[25])

#     vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
#     Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
#     rho=func_rho(pars_nova,Rsh/1.496e13) # g/cm^3   
#     test=flux_HESS_PPI[-1]*(rho/rho[-1])**2*(vsh/vsh[-1])**2*(Rsh/Rsh[-1])**3

#     ax.plot(np.log10(t+0.25),np.log10(flux_FLAT_PPI*1.0e-3),'g-',linewidth=3.0,label=r'{\rm FERMI}')
#     ax.plot(np.log10(t+0.25),np.log10(flux_HESS_PPI),'r-',linewidth=3.0,label=r'{\rm HESS}')
#     # ax.plot(np.log10(t+0.25),np.log10(test),'g:',linewidth=5.0)

#     t_HESS_raw, flux_HESS_raw=np.loadtxt('Data/data_time_gamma_HESS_raw.dat',unpack=True,usecols=[0,1])
#     t_HESS_raw=t_HESS_raw.reshape((int(len(flux_HESS_raw)/5),5))
#     flux_HESS_raw=flux_HESS_raw.reshape((int(len(flux_HESS_raw)/5),5))
#     xerr_HESS_raw=np.array([t_HESS_raw[:,0]-t_HESS_raw[:,1],t_HESS_raw[:,4]-t_HESS_raw[:,0]])
#     yerr_HESS_raw=np.array([flux_HESS_raw[:,0]-flux_HESS_raw[:,3],flux_HESS_raw[:,4]-flux_HESS_raw[:,0]])

#     # ax.plot(np.log10(t_HESS_raw[:,0]),np.log10(flux_HESS_raw[:,0]),'rs')
#     # ax.errorbar(np.log10(t_HESS_raw[:,0]), np.log10(flux_HESS_raw[:,0]), yerr=np.log10(yerr_HESS_raw), xerr=np.log10(xerr_HESS_raw), fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='red', markeredgecolor='black', markersize=10, label='HESS')


#     # print(flux_HESS_raw[0,0],flux_HESS_raw[0,1],flux_HESS_raw[0,2])
#     # print(t_HESS_raw[:,0])
#     print(t_HESS_raw[:,0]-t_HESS_raw[:,1],t_HESS_raw[:,2]-t_HESS_raw[:,0])
#     print(flux_HESS_raw[:,0]-flux_HESS_raw[:,3],flux_HESS_raw[:,4]-flux_HESS_raw[:,0])
#     # print(flux_HESS_raw[:,0])

#     filename='gamma.dat'
#     t, E, phi, phi_abs, tau_gg=np.loadtxt(filename,unpack=True,usecols=[0,1,2,3,4])
#     Nphi=len(phi)
#     Nt=151
#     t=np.reshape(t, (Nt, int(Nphi/Nt)))+0.25
#     E=np.reshape(E, (Nt, int(Nphi/Nt)))*1.60218e-3 # GeV -> erg
#     phi=np.reshape(phi, (Nt, int(Nphi/Nt)))/E # cm^-2 s^-1
#     phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))/E # cm^-2 s^-1
#     tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

#     # print("FERMI band: ",E[0,jmin_FLAT]/1.60218e-3,"-",E[0,jmax_FLAT]/1.60218e-3,"GeV")
#     # print("HESS band:  ",E[0,jmin_HESS]/1.60218e-3,"-",E[0,jmax_HESS]/1.60218e-3,"GeV")

#     flux_FLAT=np.nansum((E[:,jmin_FLAT+1:jmax_FLAT+1]-E[:,jmin_FLAT:jmax_FLAT])*phi[:,jmin_FLAT:jmax_FLAT],axis=1)
#     flux_HESS=np.nansum((E[:,jmin_HESS+1:jmax_HESS+1]-E[:,jmin_HESS:jmax_HESS])*phi[:,jmin_HESS:jmax_HESS],axis=1)
#     flux_FLAT_abs=np.nansum((E[:,jmin_FLAT+1:jmax_FLAT+1]-E[:,jmin_FLAT:jmax_FLAT])*phi_abs[:,jmin_FLAT:jmax_FLAT],axis=1)
#     flux_HESS_abs=np.nansum((E[:,jmin_HESS+1:jmax_HESS+1]-E[:,jmin_HESS:jmax_HESS])*phi_abs[:,jmin_HESS:jmax_HESS],axis=1)

#     ax.plot(np.log10([1.85,1.85]),[-13.0,-11.0],'r:')
#     ax.plot(np.log10([2.85,2.85]),[-13.0,-11.0],'r:')
#     ax.plot(np.log10([3.85,3.85]),[-13.0,-11.0],'r:')
#     ax.plot(np.log10([4.85,4.85]),[-13.0,-11.0],'r:')
#     ax.plot(np.log10([5.85,5.85]),[-13.0,-11.0],'r:')
#     # ax.plot(np.log10(t[:,0]),np.log10(flux_FLAT*1.0e-3),'g-')
#     # ax.plot(np.log10(t[:,0]),np.log10(flux_FLAT_abs*1.0e-3),'g--')
#     # ax.plot(np.log10(t[:,0]),np.log10(flux_HESS),'r-')
#     # ax.plot(np.log10(t[:,0]),np.log10(flux_HESS_abs),'r--')

#     # print(E[:,jmin_HESS]*phi_abs[:,jmin_HESS])

#     # Read the image for data    
#     # img = mpimg.imread("Data/data_time_gamma.png")
#     # img_array = np.mean(np.array(img), axis=2)

#     # xmin=np.log10(0.4)
#     # xmax=np.log10(30.0)
#     # ymin=-13.0
#     # ymax=np.log10(2.0e-11)
#     # ax.set_xlim(-1,xmax)
#     # ax.set_aspect(0.5)

#     img = mpimg.imread("Data/data_time_gamma_Zheng.png")
#     img_array = np.mean(np.array(img), axis=2)

#     xmin=np.log10(0.1)
#     xmax=np.log10(50.0)
#     ymin=np.log10(4.0e-14)
#     ymax=np.log10(1.0e-10)

#     ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
#     ax.set_xticks(np.arange(int(xmin),int(xmax)+1,1))
#     ax.set_yticks(np.arange(int(ymin),int(ymax),1))

#     ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
#     ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

#     ax.legend()
#     ax.set_aspect(0.65)
#     ax.set_xlabel(r'$t+0.25\, {\rm (day)}$',fontsize=fs)
#     ax.set_ylabel(r'${\rm Integrated\, Flux} \, ({\rm erg\, cm^{-2}\, s^{-1}})$',fontsize=fs)
#     for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
#         label_ax.set_fontsize(fs)
#     ax.legend(loc='lower left', prop={"size":fs})
#     ax.grid(linestyle='--')

#     plt.savefig('fg_time_gamma_%s.png' % pars_nova[15])


# Plot time profile of gamma-ray integrated flux
def plot_time_gamma(pars_nova, phi_PPI, tau_gg, Eg, t):

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    dlogEg=np.log10(Eg[1]/Eg[0])

    jmin_FLAT=int(np.log10(0.1e9/Eg[0])/dlogEg)
    jmax_FLAT=int(np.log10(100.0e9/Eg[0])/dlogEg)
    jmin_HESS=int(np.log10(250.0e9/Eg[0])/dlogEg)
    jmax_HESS=int(np.log10(2500.0e9/Eg[0])/dlogEg)

    # print("FERMI band: ",Eg[jmin_FLAT]*1.0e-9,"-",Eg[jmax_FLAT]*1.0e-9,"GeV")
    # print("HESS band:  ",Eg[jmin_HESS]*1.0e-9,"-",Eg[jmax_HESS]*1.0e-9,"GeV")

    flux_FLAT_PPI=1.0e-3*1.60218e-12*np.nansum((Eg[jmin_FLAT+1:jmax_FLAT+1,np.newaxis]-Eg[jmin_FLAT:jmax_FLAT,np.newaxis])*Eg[jmin_FLAT:jmax_FLAT,np.newaxis]*phi_PPI[jmin_FLAT:jmax_FLAT,:],axis=0)
    flux_HESS_PPI=1.60218e-12*np.nansum((Eg[jmin_HESS+1:jmax_HESS+1,np.newaxis]-Eg[jmin_HESS:jmax_HESS,np.newaxis])*Eg[jmin_HESS:jmax_HESS,np.newaxis]*phi_PPI[jmin_HESS:jmax_HESS,:],axis=0)

    if(pars_nova[16]==1):
        # phi_PPI*=np.exp(tau_gg)
        phi_PPI*=1.0/(0.5*(np.exp(-1.1*tau_gg)+np.exp(-6.3*tau_gg)))
        flux_FLAT_PPI_noabs=1.0e-3*1.60218e-12*np.nansum((Eg[jmin_FLAT+1:jmax_FLAT+1,np.newaxis]-Eg[jmin_FLAT:jmax_FLAT,np.newaxis])*Eg[jmin_FLAT:jmax_FLAT,np.newaxis]*phi_PPI[jmin_FLAT:jmax_FLAT,:],axis=0)
        flux_HESS_PPI_noabs=1.60218e-12*np.nansum((Eg[jmin_HESS+1:jmax_HESS+1,np.newaxis]-Eg[jmin_HESS:jmax_HESS,np.newaxis])*Eg[jmin_HESS:jmax_HESS,np.newaxis]*phi_PPI[jmin_HESS:jmax_HESS,:],axis=0)
        ax.plot(t,flux_HESS_PPI_noabs,'r--',linewidth=3.0)
        ax.plot(t,flux_FLAT_PPI_noabs,'g--',linewidth=3.0)

    # data=np.column_stack((t, flux_FLAT_PPI, flux_HESS_PPI))
    # np.savetxt('flux.txt', data, fmt='%.2e')

    # vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
    # Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    # rho=func_rho(pars_nova,Rsh/1.496e13) # g/cm^3   
    # test=flux_HESS_PPI[-1]*(rho/rho[-1])**2*(vsh/vsh[-1])**2*(Rsh/Rsh[-1])**3

    ax.plot(t,flux_HESS_PPI,'r-',linewidth=3.0,label=r'{\rm Model\, HESS\, band}')
    ax.plot(t,flux_FLAT_PPI,'g-',linewidth=3.0,label=r'{\rm Model\, FERMI\, band}')

    ax.errorbar(t_HESS_raw,flux_HESS_raw,yerr=yerr_HESS_raw,xerr=xerr_HESS_raw,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='red',markeredgecolor='black',markersize=10,label=r'${\rm HESS}$')
    ax.errorbar(t_FERMI_raw,flux_FERMI_raw,yerr=yerr_FERMI_raw,xerr=xerr_FERMI_raw,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='green',markeredgecolor='black',markersize=10,label=r'${\rm FERMI\,(\times 10^{-3})}$')

    # ax.plot([1.6,1.6],[1.0e-13,1.0e-11],'r:')
    # ax.plot([2.6,2.6],[1.0e-13,1.0e-11],'r:')
    # ax.plot([3.6,3.6],[1.0e-13,1.0e-11],'r:')
    # ax.plot([4.6,4.6],[1.0e-13,1.0e-11],'r:')
    # ax.plot([5.6,5.6],[1.0e-13,1.0e-11],'r:')

    ax.set_xlim(1.0e-1,5.0e1)
    ax.set_ylim(1.0e-13,3.0e-11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'${\rm Integrated\, Flux} \, ({\rm erg\, cm^{-2}\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs}, ncols=2)
    ax.grid(linestyle='--')

    plt.savefig('Results/fg_time_gamma_%s_tST-%.2f_Mdot-%.2e_ter-%.1f_BRG-%.1f.png' % (pars_nova[15], pars_nova[1], pars_nova[3], pars_nova[9], pars_nova[10]))
    plt.close()
 

def plot_Rsh(pars_nova, t):

    Rsh=func_Rsh(pars_nova,t)

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,Rsh,'r--',linewidth=3.0)
    # ax.plot(t,np.sqrt(1.48**2+Rsh**2),'r--',linewidth=3.0)

    t, Rsh=np.loadtxt('profile.dat',unpack=True,usecols=[0,2])
    ax.plot(t,Rsh,'k:',linewidth=3.0)

    # Read the image for data    
    img = mpimg.imread("Data/data_Rsh.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=5.0
    ymin=0.0
    ymax=14.0
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 

    ax.legend()
    ax.set_aspect(0.3)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$R_{\rm sh} \, ({\rm au})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_Rsh_%s.png' % pars_nova[15])


def plot_vsh(pars_nova, t):

    vsh=func_vsh(pars_nova,t)

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,vsh,'r--',linewidth=3.0)

    t, vsh=np.loadtxt('profile.dat',unpack=True,usecols=[0,1])
    ax.plot(t,vsh,'k:',linewidth=3.0)

    # ax.set_xlim(0,5)
    # ax.set_ylim(1.0e1,2.0e5)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$v_{\rm sh} \, ({\rm km/s})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_vsh_%s.png' % pars_nova[15])


def plot_rho(pars_nova, t):

    Rsh=func_Rsh(pars_nova,t) # au
    rho=func_rho(pars_nova,Rsh)/mpCGS # cm^-3

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,rho,'r--',linewidth=3.0)

    t, rho=np.loadtxt('profile.dat',unpack=True,usecols=[0,3])
    ax.plot(t,rho,'k:',linewidth=3.0)

    # # Read the image for data    
    # img = mpimg.imread("Data/data_Rsh.png")
    # img_array = np.mean(np.array(img), axis=2)

    # xmin=0.0
    # xmax=5.0
    # ymin=0.0
    # ymax=14.0
    # ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 

    ax.legend()
    # ax.set_aspect(0.3)
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$n_{\rm w} \, ({\rm cm^{-3}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_rho_%s.png' % pars_nova[15])


def plot_Emax(pars_nova, t):

    # Solve for the maximum energy over time
    sol=solve_ivp(lambda tp,Emax:func_dE_acc(pars_nova,Emax,tp),(t[0],t[-1]),[0.0],method='RK45',dense_output=True)
    Emax=((sol.sol(t)[0]).T*86400.0)


    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,Emax*1.0e-12,'r--',linewidth=3.0)

    t, Emax=np.loadtxt('profile.dat',unpack=True,usecols=[0,4])
    ax.plot(t,Emax,'k:',linewidth=3.0)

    # # Read the image for data    
    # img = mpimg.imread("Data/data_Rsh.png")
    # img_array = np.mean(np.array(img), axis=2)

    # xmin=0.0
    # xmax=5.0
    # ymin=0.0
    # ymax=14.0
    # ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 

    ax.legend()
    # ax.set_aspect(0.3)
    ax.set_yscale('log')
    ax.set_ylim(1.0e-3,1.0e1)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$n_{\rm w} \, ({\rm cm^{-3}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_Emax_%s.png' % pars_nova[15])


# Function to calculate the gamma-ray spectrum
def func_phi_PPI(eps_nucl, d_sigma_g, sigma_gg, pars_nova, E, Eg, t):

    # Some basic functions
    it_interp=int(pars_nova[16])
    t_interp=t[np.arange(0,len(t),it_interp)]
    dlogEg=np.log10(Eg[1]/Eg[0])
    dE=np.append(np.diff(E),0.0)[:,np.newaxis,np.newaxis]

    # Distance from the nova to Earth
    Ds=pars_nova[14]*3.086e18 # cm

    # Calculate the proton distribution
    JEp=func_JEp_p(pars_nova,E,t) # eV^-1 cm s^-1
    JEp=JEp[:,np.newaxis,np.arange(0,len(t),it_interp)] # eV^-1 cm s^-1
    Rsh=func_Rsh(pars_nova,t_interp)*1.496e13 # cm
    rho=func_rho(pars_nova,t_interp) # g cm^-3
    rho=rho[np.newaxis,np.newaxis,:] # g cm^-3

    # Opacity of gamma rays
    TOPT=kB*pars_nova[11] # eV
    Ebg=np.logspace(np.log10(TOPT*1.0e-2),np.log10(TOPT*1.0e2),1000)
    dEbg=np.append(np.diff(Ebg),0.0)[np.newaxis,:,np.newaxis]
    UOPT=func_LOPT(t_interp)*6.242e11/(4.0*np.pi*pow(Rsh,2)*3.0e10) # eV cm^⁻3
    fOPT=gt.func_fEtd(UOPT[np.newaxis,:],TOPT,0.0,Ebg[:,np.newaxis]) # eV^-1 cm^-3
    fOPT=fOPT[np.newaxis,:,:]
    tau_gg=np.sum(fOPT*sigma_gg*Rsh[np.newaxis,np.newaxis,:]*dEbg, axis=1)

    # Calculate the gamma-ray spectrum
    phi_PPI=np.nansum((4.0*rho/(4.0*np.pi*Ds**2*mpCGS))*(dE*JEp*eps_nucl)*d_sigma_g, axis=0)
    phi_PPI=phi_PPI*(0.5*(np.exp(-1.1*tau_gg)+np.exp(-6.3*tau_gg)))
    # phi_PPI*=np.exp(-tau_gg)

    if(it_interp==1):
        plot_gamma(pars_nova,phi_PPI,tau_gg,Eg,t,160)
        plot_gamma(pars_nova,phi_PPI,tau_gg,Eg,t,560)
    plot_time_gamma(pars_nova,phi_PPI,tau_gg,Eg,t_interp)

    jmin_FLAT=int(np.log10(0.1e9/Eg[0])/dlogEg)
    jmax_FLAT=int(np.log10(100.0e9/Eg[0])/dlogEg)
    jmin_HESS=int(np.log10(250.0e9/Eg[0])/dlogEg)
    jmax_HESS=int(np.log10(2500.0e9/Eg[0])/dlogEg)

    flux_FLAT_PPI=1.0e-3*1.60218e-12*np.nansum((Eg[jmin_FLAT+1:jmax_FLAT+1,np.newaxis]-Eg[jmin_FLAT:jmax_FLAT,np.newaxis])*Eg[jmin_FLAT:jmax_FLAT,np.newaxis]*phi_PPI[jmin_FLAT:jmax_FLAT,:],axis=0)
    flux_HESS_PPI=1.60218e-12*np.nansum((Eg[jmin_HESS+1:jmax_HESS+1,np.newaxis]-Eg[jmin_HESS:jmax_HESS,np.newaxis])*Eg[jmin_HESS:jmax_HESS,np.newaxis]*phi_PPI[jmin_HESS:jmax_HESS,:],axis=0)

    interp_func_FLAT=sp.interpolate.interp1d(t_interp, flux_FLAT_PPI, kind='cubic')
    interp_func_HESS=sp.interpolate.interp1d(t_interp, flux_HESS_PPI, kind='cubic')

    flux_FLAT_PPI_interp=interp_func_FLAT(t_FERMI_raw)
    flux_HESS_PPI_interp=interp_func_HESS(t_HESS_raw)

    chi2=np.sum(((flux_FERMI_raw-flux_FLAT_PPI_interp)/yerr_FERMI_raw)**2)
    chi2=np.sum(((flux_HESS_raw-flux_HESS_PPI_interp)/yerr_HESS_raw)**2)

    return chi2

if __name__ == "__main__":

    # Parameters for RS Ophiuchi 2021
    # pars_nova[0]=vsh0, pars_nova[1]=tST, pars_nova[2]=alpha;
    # pars_nova[3]=Mdot, pars_nova[4]=vwind, pars_nova[5]=Rmin;
    # pars_nova[6]=xip, pars_nova[7]=delta, pars_nova[8]=epsilon;
    # pars_nova[9]=ter, pars_nova[10]=BRG, pars_nova[11]=TOPT;
    # pars_nova[12]=scale_t, pars_nova[13]=Mej, pars_nova[14]=Ds;
    pars_init=[4500.0, 2.0, 0.66, 2.0e-7, 20.0, 1.48, 0.1, 4.4, 1.0, 0.0, 1.0, 0.8e4, 10, 2.0e-9, 1.4e3, 'DM23', 50]
    tST=np.array([1.8]) #np.linspace(1.0,4.0,16)
    Mdot=np.array([5.0e-7]) #np.linspace(4.0,6.0,11)*1.0e-7
    ter=np.array([-0.2]) #np.linspace(-1.0,1.0,11)
    BRG=np.array([1.5]) #np.linspace(0.5,1.5,11)
    # tST=np.linspace(1.0,4.0,16)
    # Mdot=np.linspace(4.0,6.0,11)*1.0e-7
    # ter=np.linspace(-1.0,1.0,11)
    # BRG=np.linspace(0.5,1.5,11)

    tST, Mdot, ter, BRG=np.meshgrid(tST, Mdot, ter, BRG)
    tST=tST.flatten()
    Mdot=Mdot.flatten()
    ter=ter.flatten()
    BRG=BRG.flatten()

    # Record the starting time
    start_time = time.time()

    # Define the time and energy ranges
    t=np.linspace(0.0,30.0,3001)
    E=np.logspace(8,14,601)
    Eg=E

    # Gamma-ray production cross-section
    eps_nucl=gt.func_enhancement(E)
    d_sigma_g=gt.func_d_sigma_g(E,Eg)
    eps_nucl=eps_nucl[:,np.newaxis,np.newaxis]
    d_sigma_g=d_sigma_g[:,:,np.newaxis] # cm^2 eV^-1

    # Gamma-gamma cross section
    TOPT=kB*pars_init[11] # eV
    Ebg=np.logspace(np.log10(TOPT*1.0e-2),np.log10(TOPT*1.0e2),1000)
    dEbg=np.append(np.diff(Ebg),0.0)[np.newaxis,:,np.newaxis]
    sigma_gg=gt.func_sigma_gg(Eg,Ebg) # cm^2
    sigma_gg=sigma_gg[:,:,np.newaxis]

    # Limit the range for data
    mask=(t_FERMI_raw<t[-1])
    t_FERMI_raw=t_FERMI_raw[mask]
    flux_FERMI_raw=flux_FERMI_raw[mask]
    yerr_FERMI_raw=yerr_FERMI_raw[mask]
    xerr_FERMI_raw=xerr_FERMI_raw[mask]

    mask=(t_HESS_raw<t[-1])
    t_HESS_raw=t_HESS_raw[mask]
    flux_HESS_raw=flux_HESS_raw[mask]
    yerr_HESS_raw=yerr_HESS_raw[mask]
    xerr_HESS_raw=xerr_HESS_raw[mask]

    # Create the lists of parameters for scanning
    pars_scan=[]
    for i in range(len(tST)):
        pars_nova=pars_init.copy()
        pars_nova[1]=tST[i]
        pars_nova[3]=Mdot[i]
        pars_nova[9]=ter[i]
        pars_nova[10]=BRG[i]

        pars_scan.append(pars_nova)

    args=[(eps_nucl, d_sigma_g, sigma_gg, pars_nova, E, Eg, t) for pars_nova in pars_scan]

    # Create a Pool and use starmap to pass arguments to the worker function
    with Pool(processes=10) as pool:
        results=pool.starmap(func_phi_PPI, args)

    results=np.array(results)
    print(tST[np.where(results==np.min(results))],Mdot[np.where(results==np.min(results))],ter[np.where(results==np.min(results))],BRG[np.where(results==np.min(results))])

    # Save chi2 into a txt file
    combined_array=np.column_stack((tST, Mdot, ter, BRG, results))
    np.savetxt('chi2.txt', combined_array, fmt='%.4e', delimiter=' ')

    # Plot the best fit parameters
    pars_nova[16]=1
    results=func_phi_PPI(eps_nucl,d_sigma_g,sigma_gg,pars_nova,E,Eg,t)

    # # Plot HESS model
    # pars_nova[9]=0.0
    # pars_nova[15]='HESS'
    # results=func_phi_PPI(eps_nucl,d_sigma_g,sigma_gg,pars_nova,E,Eg,t)

    # np.save('phi_PPI_%s.npy' % pars_nova[15], phi_PPI*np.exp(-tau_gg))
    # phi_PPI=np.load('phi_PPI_%s.npy' % pars_nova[15])

    # Plot spectra
    # plot_fEp(NEp,E,t)
    # plot_gamma(pars_nova,phi_PPI,tau_gg,Eg,t,160)
    # plot_gamma(pars_nova,phi_PPI,tau_gg,Eg,t,260)
    # plot_gamma(pars_nova,phi_PPI,tau_gg,Eg,t,360)
    # plot_gamma(pars_nova,phi_PPI,tau_gg,Eg,t,460)
    # plot_gamma(pars_nova,phi_PPI,tau_gg,Eg,t,560)
    # plot_time_gamma(pars_nova,phi_PPI,tau_gg,Eg,t)

    # plot_Rsh(pars_nova,t)
    # plot_vsh(pars_nova,t)
    # plot_Emax(pars_nova,t)
    # plot_rho(pars_nova,t)
    # plot_LOPT(pars_nova)

    # Record the ending time
    end_time=time.time()

    # Calculate the elapsed time
    elapsed_time=end_time-start_time

    print("Elapsed time:", elapsed_time, "seconds")
