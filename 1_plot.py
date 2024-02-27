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

fs=22

# Define a custom tick formatter to display 10^x
def log_tick_formatter(x, pos):
    return r'$10^{%d}$' % int(x)


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

def func_LOPT(t):

	LOPT=1.3e39*t**-0.28*(t+0.6)**-1.0
	
	return LOPT

def plot_LOPT():

    t=np.linspace(-20,60,81)

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,np.log10(func_LOPT(t)),'g--',linewidth=3.0, label=r'{\rm Fit}')
    ax.plot([1.0,1.0],[36,39],'r:')
    print(func_LOPT(1.0))

    # Read the image for data
    img = mpimg.imread("data_LOPT.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=-20.0
    xmax=60.0
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
    ax.legend(loc='lower left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_LOPT.png')

def plot_gamma(t_day):

    filename='gamma.dat'
    t, E, phi, phi_abs, tau_gg=np.loadtxt(filename,unpack=True,usecols=[0,1,2,3,4])
    Nt=101
    Nphi=len(phi)
    t=np.reshape(t, (Nt, int(Nphi/Nt)))
    E=np.reshape(E, (Nt, int(Nphi/Nt)))
    phi=np.reshape(phi, (Nt, int(Nphi/Nt)))
    phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))
    tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(np.log10(E[10*t_day,:]),np.log10(phi_abs[10*t_day,:]),'r--',linewidth=3.0, label=r'{\rm t=%d\, day}' % t_day)

    # Read the image for data    
    img = mpimg.imread("data_day%d.png" % t_day)
    img_array = np.mean(np.array(img), axis=2)

    xmin=-1.0
    xmax=4.0
    ymin=-13.0
    ymax=np.log10(5.0e-9)
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_xticks(np.arange(xmin,xmax,1))
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

def plot_Bfield():

    filename='profile.dat'
    t, Rsh, Emax, B=np.loadtxt(filename,unpack=True,usecols=[0,2,4,5])

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,4*B,'r--',linewidth=3.0)

    # Read the image for data    
    img = mpimg.imread("data_Bfield.png")
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
    img = mpimg.imread("data_Rsh.png")
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
    img = mpimg.imread("data_Emax.png")
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
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_Emax.png')

def plot_fEp():

    filename='fEp.dat'
    t, E, fEp=np.loadtxt(filename,unpack=True,usecols=[0,1,2])
    Nt=101
    NfEp=len(fEp)
    t=np.reshape(t, (Nt, int(NfEp/Nt)))
    E=np.reshape(E, (Nt, int(NfEp/Nt)))
    fEp=np.reshape(fEp, (Nt, int(NfEp/Nt)))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(np.log10(E[10,:]*1.0e-9),np.log10((E[10,:])**2.5*fEp[10,:])+30,'r--',linewidth=3.0, label=r'{\rm t=1\, day}')
    ax.plot(np.log10(E[50,:]*1.0e-9),np.log10((E[50,:])**2.5*fEp[50,:])+30,'r--',linewidth=3.0, label=r'{\rm t=5\, day}')

    print("Hj",np.log10((E[50,0])**3*fEp[50,0])+24)

    # Read the image for gamma-ray data    
    img = mpimg.imread("data_fEp.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=4.0
    ymin=np.log10(4.0e43)
    ymax=48
    print(int(ymin))
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
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

def plot_abs():

    filenameb='gamma.dat'
    t, E, phi, phi_abs, tau_gg=np.loadtxt(filenameb,unpack=True,usecols=[0,1,2,3,4])
    Nt=101
    Nphi=len(phi)
    print(Nphi)
    t=np.reshape(t, (Nt, int(Nphi/Nt)))
    E=np.reshape(E, (Nt, int(Nphi/Nt)))
    phi=np.reshape(phi, (Nt, int(Nphi/Nt)))
    phi_abs=np.reshape(phi_abs, (Nt, int(Nphi/Nt)))
    tau_gg=np.reshape(tau_gg, (Nt, int(Nphi/Nt)))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t[:,84],np.log10(tau_gg[:,84]),'r-',linewidth=3.0, label=r'$E=0.3\,{\rm TeV}$')
    ax.plot(t[:,84],np.log10(np.exp(-tau_gg[:,84])),'r--',linewidth=3.0, label=r'$E=0.3\,{\rm TeV}$')
    ax.plot(t[:,97],np.log10(tau_gg[:,97]),'g-',linewidth=3.0, label=r'$E=0.3\,{\rm TeV}$')
    ax.plot(t[:,97],np.log10(np.exp(-tau_gg[:,97])),'g--',linewidth=3.0, label=r'$E=1\,{\rm TeV}$')
    ax.plot(t[:,121],np.log10(tau_gg[:,121]),'y-',linewidth=3.0, label=r'$E=0.3\,{\rm TeV}$')
    ax.plot(t[:,121],np.log10(np.exp(-tau_gg[:,121])),'y--',linewidth=3.0, label=r'$E=1\,{\rm TeV}$')

    # Read the image for absorpion data
    img = mpimg.imread("data_abs.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=10.0
    ymin=np.log10(0.02)
    ymax=np.log10(2.0)
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_yticks([-1, 0])

    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_aspect(4)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$\tau_{\gamma\gamma}$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_abs.png')

# plot_profile()
plot_LOPT()
plot_fEp()
plot_gamma(1)
plot_gamma(2)
plot_gamma(3)
plot_gamma(4)
plot_gamma(5)
plot_Bfield()
