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

fs=22

# Define a custom tick formatter to display 10^x
def log_tick_formatter(x, pos):
    return r'$10^{%d}$' % int(x)


mp=938.2720813e6 # eV
me=0.510998e6 # eV

pars=np.loadtxt("pars_RSOph21.dat",max_rows=8)
tmin, tmax, dt=pars[0,:]
Emin, Emax, dlogE=pars[2,:]
Egmin, Egmax, dlogEg=pars[3,:]

scale_t=pars[1,1]
dt*=scale_t
Nt=int((tmax-tmin)/dt)+1
NEg=int(np.log10(Egmax/Egmin)/dlogEg)+1

# Profiles of shock speed, radius and density
# def plot_profile():

#     fig=plt.figure(figsize=(10, 8))
#     ax=plt.subplot(111)

#     t, vsh, Rsh, nrg=np.loadtxt("profile.dat",unpack=True,usecols=[0,1,2,3])

#     ax.plot(t,vsh,'g-',linewidth=3.0,label=r'$v_{\rm sh}$')
#     ax.plot(t,Rsh,'k-',linewidth=3.0,label=r'$R_{\rm sh}$')
#     # ax.plot(t,nrg/(1.0e6*0.76*8.41e-58),'r-',linewidth=3.0,label=r'$n_{\rm rg}$')
#     ax.plot(t,nrg,'r-',linewidth=3.0,label=r'$n_{\rm rg}$')

#     t, vsh=np.loadtxt("1shock_Diesing_vsh.dat",unpack=True,usecols=[0,1])
#     ax.plot(t,vsh,'g--',linewidth=3.0)

#     t, Rsh=np.loadtxt("1shock_Diesing_Rsh.dat",unpack=True,usecols=[0,1])
#     ax.plot(t,Rsh,'k--',linewidth=3.0)

#     t, nrg=np.loadtxt("1shock_Diesing_nrg.dat",unpack=True,usecols=[0,1])
#     ax.plot(t,nrg,'r--',linewidth=3.0)

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     # ax.set_xlim(1.0e-2,1.0e8)
#     ax.set_ylim(1.0e-1,1.0e4)
#     ax.set_xlabel(r'$t\,({\rm day})$',fontsize=fs)
#     # ax.set_ylabel(r'$p^4f(p)$',fontsize=fs)
#     for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
#         label_ax.set_fontsize(fs)
#     ax.legend(loc='upper right', prop={"size":22})
#     ax.grid(linestyle='--')

#     plt.savefig("fg_profile.png")


def func_fit_LOPT(t, a, b, c):

    # LOPT= 38.8 +a*np.log10(t+20.0)+ b*(np.log10(t+20.0))**2+ c*(np.log10(t+20.0))**3
    LOPT= 38.8 +a*t+ b*t**2+ c*np.log10(abs(t))

    return LOPT


# Heaviside
def func_Heaviside(x):

    return 0.5*(1.0+np.tanh(10.0*x))


def custom_function_smooth_decay(x):
    # Parameters for the logistic function
    a = 10  # Controls the steepness of the transition
    b = 1   # Controls the position of the transition
    
    # Calculate the function values based on different regions
    y = np.where(x < -1, 37,                   # For x < -1
                 np.where(x < 1,               # For -1 <= x < 1
                          37 + (39 - 37) * ((x + 1) / 2)**2,
                          39 - 1.2 * np.log10(x)))  # For x >= 1
    
    return y

# Luminosity function of the optical emission
def func_LOPT(t):

    # LOPT=(1.0-func_Heaviside(t-0.8))*3.0e37
    # LOPT=2.5e36+func_Heaviside(t-1.0)*(1.3e39*(np.abs(t-0.25))**-0.28*(np.abs(t+0.35))**-1.0)
    # LOPT=2.0e36*(1.0-func_Heaviside(t-0.6))+1.3e39*func_Heaviside(t-0.6)*(np.abs(t)+0.5)**(-1.28)
    # a = 10  # Controls the steepness of the transition
    # b = 1   # Controls the position of the transition
    # LOPT= 38.8 +a*np.log10(t)
    LOPT=2.5e36*(1.0-func_Heaviside(t-0.9))+func_Heaviside(t-0.9)*(1.3e39*pow(abs(t-0.25),-0.28)/(abs(t+0.35)))
 

    return LOPT


# Plot luminosity function of the optical emission
def plot_LOPT():

    t_data, LOPT_data=np.loadtxt("Data/LOPT.dat",unpack=True,usecols=[0,1])
    LOPT_data=np.log10(LOPT_data)
    popt, pcov = curve_fit(func_fit_LOPT, t_data, LOPT_data, bounds=([-5,-5,-5],[0,5,5]))
    print(popt)


    t=np.linspace(-20,60,810)
    print(t[20])

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,np.log10(func_LOPT(t)),'r--',linewidth=3.0, label=r'{\rm Fit}')
    # ax.plot(t,func_fit_LOPT(t,popt[0],popt[1],popt[2]),'g--',linewidth=3.0, label=r'{\rm Fit}')
    # ax.plot(t,custom_function_smooth_decay(t),'g--',linewidth=3.0, label=r'{\rm Fit}')        
    # ax.plot([1.0,1.0],[36,39],'r:')
    # print(t,func_LOPT(t))

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

    plt.savefig('fg_LOPT.png')


# Plot gamma-ray data
def plot_gamma(t_day):

    filename='gamma.dat'
    t, E, phi, phi_abs, tau_gg=np.loadtxt(filename,unpack=True,usecols=[0,1,2,3,4])
    Nphi=len(phi)
    t=np.reshape(t, (Nt, int(Nphi/Nt)))
    E=np.reshape(E, (Nt, int(Nphi/Nt)))
    phi=np.reshape(phi, (Nt, int(Nphi/Nt)))
    phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))
    tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    t_day+=0.6
    print("Day",t[int(scale_t*t_day),0])
    ax.plot(np.log10(E[int(scale_t*t_day),:]),np.log10(phi[int(scale_t*t_day),:]),'r-',linewidth=3.0)
    ax.plot(np.log10(E[int(scale_t*t_day),:]),np.log10(phi_abs[int(scale_t*t_day),:]),'r--',linewidth=3.0, label=r'{\rm t=%.1f\, day}' % t_day)

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

    plt.savefig('fg_gamma_day%d.png' % t_day)


# Gamma-ray plot compared to Diesing 2023
def plot_gamma_Diesing():

    filename='gamma.dat'
    t, E, phi, phi_abs, tau_gg=np.loadtxt(filename,unpack=True,usecols=[0,1,2,3,4])
    Nphi=len(phi)
    t=np.reshape(t, (Nt, int(Nphi/Nt)))
    E=np.reshape(E, (Nt, int(Nphi/Nt)))
    phi=np.reshape(phi, (Nt, int(Nphi/Nt)))
    phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))
    tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(np.log10(E[int(scale_t*1.6),:]),np.log10(phi_abs[int(scale_t*1.6),:]),'r--',linewidth=3.0, label=r'{\rm t=%d\, day}' % 1)
    ax.plot(np.log10(E[int(scale_t*5.6),:]),np.log10(phi_abs[int(scale_t*5.6),:]),'g--',linewidth=3.0, label=r'{\rm t=%d\, day}' % 5)
    ax.plot(np.log10(E[int(scale_t*5.6),:]),np.log10(1.0e-9*(E[int(scale_t*5.6),:])**(-0.2)),'k--',linewidth=3.0, label=r'{\rm t=%d\, day}' % 5)

    # Read the image for data    
    img = mpimg.imread("Data/data_gamma_Diesing.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=-1.0
    xmax=4.0
    ymin=-15.0
    ymax=-8.0
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_xticks(np.arange(xmin,xmax+1,1))
    ax.set_yticks(np.arange(ymin,ymax,1))

    ax.set_ylim(-13.0,np.log10(5.0e-9))

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

    plt.savefig('fg_gamma_Diesing_1.png')

# Plot time profile of gamma-ray integrated flux
def plot_time_gamma():

    filename='gamma.dat'
    t, E, phi, phi_abs, tau_gg=np.loadtxt(filename,unpack=True,usecols=[0,1,2,3,4])
    Nphi=len(phi)
    t=np.reshape(t, (Nt, int(Nphi/Nt)))+0.25
    E=np.reshape(E, (Nt, int(Nphi/Nt)))*1.60218e-3 # GeV -> erg
    phi=np.reshape(phi, (Nt, int(Nphi/Nt)))/E # cm^-2 s^-1
    phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))/E # cm^-2 s^-1
    tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

    jmin_FLAT=int(np.log10(0.1e9/Egmin)/dlogEg)
    jmax_FLAT=int(np.log10(100.0e9/Egmin)/dlogEg)
    jmin_HESS=int(np.log10(250.0e9/Egmin)/dlogEg)
    jmax_HESS=int(np.log10(2500.0e9/Egmin)/dlogEg)

    print("FERMI band: ",E[0,jmin_FLAT]/1.60218e-3,"-",E[0,jmax_FLAT]/1.60218e-3,"GeV")
    print("HESS band:  ",E[0,jmin_HESS]/1.60218e-3,"-",E[0,jmax_HESS]/1.60218e-3,"GeV")

    flux_FLAT=np.nansum((E[:,jmin_FLAT+1:jmax_FLAT+1]-E[:,jmin_FLAT:jmax_FLAT])*phi[:,jmin_FLAT:jmax_FLAT],axis=1)
    flux_HESS=np.nansum((E[:,jmin_HESS+1:jmax_HESS+1]-E[:,jmin_HESS:jmax_HESS])*phi[:,jmin_HESS:jmax_HESS],axis=1)
    flux_FLAT_abs=np.nansum((E[:,jmin_FLAT+1:jmax_FLAT+1]-E[:,jmin_FLAT:jmax_FLAT])*phi_abs[:,jmin_FLAT:jmax_FLAT],axis=1)
    flux_HESS_abs=np.nansum((E[:,jmin_HESS+1:jmax_HESS+1]-E[:,jmin_HESS:jmax_HESS])*phi_abs[:,jmin_HESS:jmax_HESS],axis=1)

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(np.log10([1.85,1.85]),[-13.0,-11.0],'r:')
    ax.plot(np.log10([2.85,2.85]),[-13.0,-11.0],'r:')
    ax.plot(np.log10([3.85,3.85]),[-13.0,-11.0],'r:')
    ax.plot(np.log10([4.85,4.85]),[-13.0,-11.0],'r:')
    ax.plot(np.log10([5.85,5.85]),[-13.0,-11.0],'r:')
    ax.plot(np.log10(t[:,0]),np.log10(flux_FLAT*1.0e-3),'g-')
    ax.plot(np.log10(t[:,0]),np.log10(flux_FLAT_abs*1.0e-3),'g--')
    ax.plot(np.log10(t[:,0]),np.log10(flux_HESS),'r-')
    ax.plot(np.log10(t[:,0]),np.log10(flux_HESS_abs),'r--')

    # print(E[:,jmin_HESS]*phi_abs[:,jmin_HESS])

    # Read the image for data    
    # img = mpimg.imread("Data/data_time_gamma.png")
    # img_array = np.mean(np.array(img), axis=2)

    # xmin=np.log10(0.4)
    # xmax=np.log10(30.0)
    # ymin=-13.0
    # ymax=np.log10(2.0e-11)
    # ax.set_xlim(-1,xmax)
    # ax.set_aspect(0.5)

    img = mpimg.imread("Data/data_time_gamma_Zheng.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=np.log10(0.1)
    xmax=np.log10(50.0)
    ymin=np.log10(4.0e-14)
    ymax=np.log10(1.0e-10)

    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_xticks(np.arange(int(xmin),int(xmax)+1,1))
    ax.set_yticks(np.arange(int(ymin),int(ymax),1))

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_aspect(0.65)
    ax.set_xlabel(r'$t+0.25\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'${\rm Integrated\, Flux} \, ({\rm erg\, cm^{-2}\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_time_gamma.png')

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    jtest1=int(np.log10(1.0e9/Egmin)/dlogEg)
    jtest2=int(np.log10(1.0e10/Egmin)/dlogEg)
    jtest3=int(np.log10(1.0e11/Egmin)/dlogEg)
    jtest4=int(np.log10(1.0e12/Egmin)/dlogEg)

    print(E[0,jtest1], E[0,jtest2], E[0,jtest3], E[0,jtest4])

    ax.plot(t[:,jtest1],phi_abs[:,jtest1],'r--',linewidth=3.0, label=r'${\rm E=10^{%d}\, eV}$' % int(np.log10(E[0,jtest1]/1.60218e-12)))
    ax.plot(t[:,jtest2],phi_abs[:,jtest2],'g--',linewidth=3.0, label=r'${\rm E=10^{%d}\, eV}$' % int(np.log10(E[0,jtest2]/1.60218e-12)))
    ax.plot(t[:,jtest3],phi_abs[:,jtest3],'b--',linewidth=3.0, label=r'${\rm E=10^{%d}\, eV}$' % int(np.log10(E[0,jtest3]/1.60218e-12)))
    ax.plot(t[:,jtest4],phi_abs[:,jtest4],'m--',linewidth=3.0, label=r'${\rm E=10^{%d}\, eV}$' % int(np.log10(E[0,jtest4]/1.60218e-12)))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$E^3 f(E) \, ({\rm eV\, cm^{-3}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_test_gamma_time.png')


#  Plot profiles of shock radius, magnetic field, and maximum energy
def plot_profile():

    filename='profile.dat'
    t, Rsh, Emax, B=np.loadtxt(filename,unpack=True,usecols=[0,2,4,5])

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,4*B,'r--',linewidth=3.0)

    # Read the image for data    
    img = mpimg.imread("Data/data_Bfield.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=5.0
    ymin=0.0
    ymax=2.5
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 

    ax.legend()
    ax.set_aspect(1.2)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$B \, ({\rm G})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_Bfield.png')

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,Rsh,'g--',linewidth=3.0)
    ax.plot(t,np.sqrt(1.48**2+Rsh**2),'r--',linewidth=3.0)

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

    plt.savefig('fg_Rsh.png')

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,np.log10(Emax),'r--',linewidth=3.0)

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

    plt.savefig('fg_Emax.png')

    filename='profile.dat'
    t, Emax, Emax_Bell, Emax_conf, Emax_TH07=np.loadtxt(filename,unpack=True,usecols=[0,7,8,9,10])

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,Emax,'r--',linewidth=3.0,label=r'{\rm Compressed Background}')
    ax.plot(t,Emax_Bell,'g:',linewidth=3.0,label=r'{\rm Bell Instability}')
    ax.plot(t,Emax_conf,'k-',linewidth=3.0,label=r'{\rm Confinement Limit}')
    # ax.plot(t,Emax_TH07,linestyle=':',color='orange',linewidth=3.0,label=r'{\rm TH07}')

    ax.set_yscale('log')
    ax.set_xlim(0,5)
    ax.set_ylim(1.0e1,2.0e5)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$E_{\rm max} \, ({\rm GeV})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_Emax_new.png')

    filename='profile.dat'
    t, MSU=np.loadtxt(filename,unpack=True,usecols=[0,6])

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,MSU/2.0e33,'r--',linewidth=3.0,label=r'{\rm Compressed Background}')
    # ax.plot(t,Emax_Bell,'g:',linewidth=3.0,label=r'{\rm Bell Instability}')
    # ax.plot(t,Emax_conf,'k-',linewidth=3.0,label=r'{\rm Confinement Limit}')

    ax.set_yscale('log')
    # ax.set_xlim(0,5)
    # ax.set_ylim(1.0e2,2.0e5)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$E_{\rm max} \, ({\rm GeV})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_MSU.png')

    filename='profile.dat'
    t, Gkmax, Gin=np.loadtxt(filename,unpack=True,usecols=[0,11,12])

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    # ax.plot(t,Gkmax/(np.sqrt(1.0-fion)),'r--',linewidth=3.0,label=r'{\rm Growth rate}')
    # ax.plot(t,Gin*(1.0-fion),'g-',linewidth=3.0,label=r'{\rm Damping rate}')

    fion=0.9
    ax.plot(t,(Gkmax/(np.sqrt(1.0-fion)))/(Gin*(1.0-fion)),'r--',linewidth=3.0,label=r'$f_{\rm ion}=0.9$')
    fion=0.7
    ax.plot(t,(Gkmax/(np.sqrt(1.0-fion)))/(Gin*(1.0-fion)),'g--',linewidth=3.0,label=r'$f_{\rm ion}=0.7$')
    fion=0.5
    ax.plot(t,(Gkmax/(np.sqrt(1.0-fion)))/(Gin*(1.0-fion)),linestyle='--',color='orange',linewidth=3.0,label=r'$f_{\rm ion}=0.5$')

    ax.set_yscale('log')
    ax.set_xlim(0,6)
    # ax.set_ylim(1.0e2,2.0e5)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$\Gamma_{\rm k_{\rm max}}/\Gamma_{\rm in}$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_rate.png')


# Plot the cosmic-ray distribution
def plot_fEp():

    filename='fEp.dat'
    t, E, fEp=np.loadtxt(filename,unpack=True,usecols=[0,1,2])
    NfEp=len(fEp)
    t=np.reshape(t, (Nt, int(NfEp/Nt)))
    E=np.reshape(E, (Nt, int(NfEp/Nt)))
    fEp=np.reshape(fEp, (Nt, int(NfEp/Nt)))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(np.log10(E[5,:]*1.0e-9),np.log10((E[5,:])**3*fEp[5,:])-18,'m--',linewidth=3.0, label=r'{\rm t=1\, day}')
    ax.plot(np.log10(E[10,:]*1.0e-9),np.log10((E[10,:])**3*fEp[10,:])-18,'r--',linewidth=3.0, label=r'{\rm t=1\, day}')
    ax.plot(np.log10(E[50,:]*1.0e-9),np.log10((E[50,:])**3*fEp[50,:])-18,'g--',linewidth=3.0, label=r'{\rm t=5\, day}')

    # Read the image for data    
    img = mpimg.imread("Data/data_fEp.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=4.0
    ymin=np.log10(4.0e43)
    ymax=48
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_ylim(ymin,50)
    ax.set_xlim(0,5)

    ax.set_xticks(np.arange(xmin,xmax+1,1))
    ax.set_yticks(np.arange(int(ymin)+1,ymax+1,1))

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

    plt.savefig('fg_fEp.png')

    
    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    jtest1=int(np.log10(1.0e9/Emin)/dlogE)
    jtest2=int(np.log10(1.0e10/Emin)/dlogE)
    jtest3=int(np.log10(1.0e11/Emin)/dlogE)
    jtest4=int(np.log10(1.0e12/Emin)/dlogE)

    print(E[0,jtest1], E[0,jtest2], E[0,jtest3], E[0,jtest4])

    ax.plot(t[:,jtest1],fEp[:,jtest1],'r--',linewidth=3.0, label=r'${\rm E=10^{%d}\, eV}$' % int(np.log10(E[0,jtest1])))
    ax.plot(t[:,jtest2],fEp[:,jtest2],'g--',linewidth=3.0, label=r'${\rm E=10^{%d}\, eV}$' % int(np.log10(E[0,jtest2])))
    ax.plot(t[:,jtest3],fEp[:,jtest3],'b--',linewidth=3.0, label=r'${\rm E=10^{%d}\, eV}$' % int(np.log10(E[0,jtest3])))
    ax.plot(t[:,jtest4],fEp[:,jtest4],'m--',linewidth=3.0, label=r'${\rm E=10^{%d}\, eV}$' % int(np.log10(E[0,jtest4])))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'$E\, {\rm (GeV)}$',fontsize=fs)
    ax.set_ylabel(r'$E^3 f(E) \, ({\rm eV\, cm^{-3}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_fEp_time.png')


#  Plot the gamma-ray attenuation
def plot_abs():

    filenameb='gamma.dat'
    t, E, phi, phi_abs, tau_gg=np.loadtxt(filenameb,unpack=True,usecols=[0,1,2,3,4])
    Nphi=len(phi)
    t=np.reshape(t, (Nt, int(Nphi/Nt)))
    E=np.reshape(E, (Nt, int(Nphi/Nt)))
    phi=np.reshape(phi, (Nt, int(Nphi/Nt)))
    phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))
    tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

    print(E[0,84], E[0,91], E[0,97])

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t[:,84],np.log10(tau_gg[:,84]),'r-',linewidth=3.0, label=r'$E=0.3\,{\rm TeV}$')
    ax.plot(t[:,84],np.log10(np.exp(-tau_gg[:,84])),'r--',linewidth=3.0)
    ax.plot(t[:,91],np.log10(tau_gg[:,91]),'g-',linewidth=3.0, label=r'$E=0.6\,{\rm TeV}$')
    ax.plot(t[:,91],np.log10(np.exp(-tau_gg[:,91])),'g--',linewidth=3.0)
    ax.plot(t[:,97],np.log10(tau_gg[:,97]),'y-',linewidth=3.0, label=r'$E=1\,{\rm TeV}$')
    ax.plot(t[:,97],np.log10(np.exp(-tau_gg[:,97])),'y--',linewidth=3.0)

    # Read the image for data
    img = mpimg.imread("Data/data_abs.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=10.0
    ymin=np.log10(0.02)
    ymax=np.log10(2.0)
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_yticks([-1, 0])

    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.legend()
    ax.set_aspect(4)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$\tau_{\gamma\gamma}$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_abs.png')


#  Plot the gamma-ray attenuation from HESS
def plot_abs_HESS():

    filenameb='gamma.dat'
    t, E, phi, phi_abs, tau_gg=np.loadtxt(filenameb,unpack=True,usecols=[0,1,2,3,4])
    Nphi=len(phi)
    t=np.reshape(t, (Nt, int(Nphi/Nt)))
    E=np.reshape(E, (Nt, int(Nphi/Nt)))
    phi=np.reshape(phi, (Nt, int(Nphi/Nt)))
    phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))
    tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(np.log10(E[50,:]),np.exp(-tau_gg[50,:]),'r-',linewidth=3.0, label=r'$E=0.3\,{\rm TeV}$')

    # Read the image for data
    img = mpimg.imread("Data/data_abs_HESS.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=2.0
    xmax=4.0
    ymin=0.5
    ymax=1.05
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_xticks([2, 3, 4])

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_aspect(3)
    ax.set_xlabel(r'$E\, {\rm (GeV)}$',fontsize=fs)
    ax.set_ylabel(r'${\rm exp}\left(-\tau_{\gamma\gamma}\right)$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_abs_HESS.png')


plot_LOPT()
plot_fEp()
plot_gamma(1)
plot_gamma(2)
plot_gamma(3)
plot_gamma(4)
plot_gamma(5)
plot_gamma_Diesing()
plot_profile()
# plot_abs()
# plot_abs_HESS()
plot_time_gamma()
