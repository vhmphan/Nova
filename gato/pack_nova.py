import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc("text",usetex=True)
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from matplotlib.ticker import MultipleLocator
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter

fs=22

mp=938.272e6 # eV
mpCGS=1.67262192e-24 # g
qeCGS=4.8032e-10 # CGS unit -> Electric charge of proton
kB=8.617333262145e-5 # eV/K

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
    t=np.array(t,ndmin=1)

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
    t=np.array(t,ndmin=1)

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
    B2=B2_bkgr #+B2_Bell*func_Heaviside(arr_t - tST)  # Assuming func_Heaviside is defined and vectorized

    dEdt=(qeCGS*B2*np.power(vsh,2))*6.242e+11/(2.0*np.pi*3.0e10)

    return dEdt # eV/s

# Adiabatic energy loss rate
def func_dE_adi(pars_nova, E, t):

    tST=pars_nova[1] # day
    alpha=pars_nova[2]
    Rmin=pars_nova[5]*1.496e13 # cm
    vsh=func_vsh(pars_nova,t)*1.0e5 # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    p=np.sqrt((E+mp)**2-mp**2)
    t=np.array(t)

    dEdt=-0.2*(p**2/(E+mp))*(Rsh*vsh/(Rmin**2+Rsh**2))

    if(t>tST):
        dEdt+=-0.2*(p**2/(E+mp))*(2.0*alpha/(t*86400.0))

    return dEdt # eV/s

def plot_Rsh(pars_nova, t):

    Rsh=func_Rsh(pars_nova,t)
    
    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,Rsh,'r--',linewidth=3.0)
    # ax.plot(t,np.sqrt(1.48**2+Rsh**2),'r--',linewidth=3.0)

    # t, Rsh=np.loadtxt('profile.dat',unpack=True,usecols=[0,2])
    # ax.plot(t,Rsh,'k:',linewidth=3.0)

    # # Read the image for data    
    # img = mpimg.imread("Data/data_Rsh.png")
    # img_array = np.mean(np.array(img), axis=2)

    # xmin=0.0
    # xmax=5.0
    # ymin=0.0
    # ymax=14.0
    # ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 

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

    # t, vsh=np.loadtxt('profile.dat',unpack=True,usecols=[0,1])
    # ax.plot(t,vsh,'k:',linewidth=3.0)

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

    ter=pars_nova[9] # day
    print('hj',t[0], t[-1])

    # Solve for the maximum energy over time
    sol=solve_ivp(lambda tp,Emax:func_dE_acc(pars_nova,Emax,tp),(t[0],t[-1]),[0.0],method='RK45',dense_output=True)
    Emax=((sol.sol(t)[0]).T*86400.0)

    sol=solve_ivp(lambda tp,Emax:func_dE_acc(pars_nova,Emax,tp),(ter,t[-1]),[0.0],method='RK45',dense_output=True)
    Emax_adi=((sol.sol(t)[0]).T*86400.0)

    # sol=solve_ivp(lambda tp,Emax:(func_dE_acc(pars_nova,Emax,tp)+func_dE_adi(pars_nova,Emax,tp)),(t[0],t[-1]),[0.0],method='RK45',dense_output=True)
    # Emax_adi=((sol.sol(t)[0]).T*86400.0)

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,Emax*1.0e-12,'r--',linewidth=3.0)
    ax.plot(t,Emax_adi*1.0e-12,'g:',linewidth=3.0)

    # t, Emax=np.loadtxt('profile.dat',unpack=True,usecols=[0,4])
    # ax.plot(t,Emax,'k:',linewidth=3.0)

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
    ax.set_ylabel(r'$E_{\rm max} \, ({\rm TeV})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_Emax_%s.png' % pars_nova[15])
