import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc("text",usetex=True)
import numpy as np
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from jax.experimental.ode import odeint
from scipy.integrate import solve_ivp
import gato.pack_gato as gt
from jax import jit
import matplotlib.ticker as ticker

fs=22


############################################################################################################################################
# Basic functions
############################################################################################################################################

# Smooth Heaviside function
def func_Heaviside(x):
    return 0.5*(1+jnp.tanh(10*x))

# Zeta function
def func_zeta(s, num_terms=100):
    n=jnp.arange(1, num_terms+1)

    return jnp.sum(1.0/jnp.power(n, s), axis=0)

# Gamma function
def func_gamma(z):
    g = 7
    p = jnp.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ])

    z=z-1
    x=p[0]+jnp.sum(p[1:]/(z+jnp.arange(1, len(p))), axis=0)
    t=z+g+0.5

    return jnp.sqrt(2*jnp.pi)*jnp.power(t,z+0.5)*jnp.exp(-t)*x


############################################################################################################################################
# Physical constants
############################################################################################################################################

me=0.5109989461e6    # eV 
mp=938.272e6         # eV
mpCGS=1.67262192e-24 # g
qeCGS=4.8032e-10     # CGS unit -> Electric charge of proton
kB=8.617333262145e-5 # eV/K
sigmaT=6.6524e-25      # cm^-2 -> Thompson cross-section


############################################################################################################################################
# Prepare data from HESS and FERMI
############################################################################################################################################

t_HESS_raw, flux_HESS_raw=np.loadtxt('Data/data_time_gamma_HESS_raw.dat',unpack=True,usecols=[0,1])
t_HESS_raw=t_HESS_raw-0.25 # Data are chosen at different time orgin than model
t_HESS_raw=t_HESS_raw.reshape((int(len(flux_HESS_raw)/5),5))
flux_HESS_raw=flux_HESS_raw.reshape((int(len(flux_HESS_raw)/5),5))
xerr_HESS_raw=t_HESS_raw[:,0]-t_HESS_raw[:,1]
yerr_HESS_raw=flux_HESS_raw[:,0]-flux_HESS_raw[:,3]
t_HESS_raw=t_HESS_raw[:,0]
flux_HESS_raw=flux_HESS_raw[:,0]

t_FERMI_raw, flux_FERMI_raw=np.loadtxt('Data/data_time_gamma_FERMI_raw.dat',unpack=True,usecols=[0,1])
t_FERMI_raw=t_FERMI_raw-0.25 # Data are chosen at different time orgin than model
t_FERMI_raw=t_FERMI_raw.reshape((int(len(flux_FERMI_raw)/5),5))
flux_FERMI_raw=flux_FERMI_raw.reshape((int(len(flux_FERMI_raw)/5),5))
xerr_FERMI_raw=t_FERMI_raw[:,0]-t_FERMI_raw[:,1]
yerr_FERMI_raw=flux_FERMI_raw[:,0]-flux_FERMI_raw[:,3]
t_FERMI_raw=t_FERMI_raw[:,0]
flux_FERMI_raw=flux_FERMI_raw[:,0]


############################################################################################################################################
# Nova shock model
############################################################################################################################################

# Nova shock speed and radius from HESS paper
t_HESS_model, vsh_HESS_model=np.loadtxt('vsh_HESS.dat',unpack=True,usecols=[0,1])
func_vsh_HESS=lambda t: jnp.interp(t, t_HESS_model, vsh_HESS_model, left=0.0, right=0.0)

Rsh_HESS_model=np.zeros_like(t_HESS_model)
for i in range(1,len(t_HESS_model)):
    t_HESS=np.linspace(t_HESS_model[0],t_HESS_model[i],1000)
    Rsh_HESS_model[i]=np.sum(func_vsh_HESS(t_HESS))*(t_HESS[1]-t_HESS[0]) # km s^-1 day

func_Rsh_HESS=lambda t: jnp.interp(t, t_HESS_model, Rsh_HESS_model, left=0.0, right=0.0)

# Nova shock speed
def func_vsh(pars_nova, t):
# t (day)

    vsh0=pars_nova[0]  # km/s
    tST=pars_nova[1]   # day
    alpha=pars_nova[2] # no unit
    ter=pars_nova[9]   # day
    t=jnp.array(t)     # day

    mask1=(t>=ter) & (t<tST)
    mask2=(t>=tST)

    vsh=jnp.zeros_like(t)

    vsh=jnp.where(mask1, vsh0, vsh)
    vsh=jnp.where(mask2, vsh0*jnp.power(t/tST, -alpha), vsh)

    # # HESS model for shock evolution
    # if(pars_nova[13]==1):
    #     vsh=func_vsh_HESS(t)
    #     vsh=jnp.where(t<ter, 0.0, vsh)

    return vsh # km/s

# Nova shock radius
def func_Rsh(pars_nova, t):
# t (day)

    vsh0=pars_nova[0]  # km/s
    tST=pars_nova[1]   # day
    alpha=pars_nova[2] # no unit
    ter=pars_nova[9]   # day
    t=jnp.array(t)     # day

    mask1=(t>=ter) & (t<tST)
    mask2=(t>=tST)

    Rsh=jnp.zeros_like(t)

    Rsh=jnp.where(mask1, vsh0*(t-ter), Rsh)
    Rsh=jnp.where(
        mask2,
        -vsh0*ter+vsh0*tST*(jnp.power(t/tST, 1.0-alpha)-alpha)/(1.0-alpha),
        Rsh,
    )

    # # HESS model for shock evolution
    # if(pars_nova[13]==1):
    #     Rsh=func_Rsh_HESS(t)
    #     Rsh=jnp.where(t<ter, 0.0, Rsh)

    return Rsh*86400.0*6.68e-9 # au

# Density profile of the red giant wind
def func_rho(pars_nova, r):
# Mdot (Msol/yr), vwind (km/s), and r (au)    

    Mdot=pars_nova[3]*1.989e33/(365.0*86400.0) # g/s 
    vwind=pars_nova[4]*1.0e5                   # cm/s
    Rmin=pars_nova[5]*1.496e13                 # cm
    r=jnp.array(r)                             # au

    rho=Mdot/(4.0*jnp.pi*vwind*pow(Rmin+r*1.496e13,2)) 

    return rho # g/cm^3


############################################################################################################################################
# Particle acceleration model
############################################################################################################################################

# Acceleration rate of protons
def func_dE_acc(pars_nova, E, t):
# E (eV) and t(day)

    tST=pars_nova[1]           # day
    Rmin=pars_nova[5]*1.496e13 # cm
    xip=pars_nova[6]           # no unit
    BRG=pars_nova[10]          # G

    vsh=func_vsh(pars_nova, t)*1.0e5      # cm/s
    Rsh=func_Rsh(pars_nova, t)*1.496e13   # cm
    rho=func_rho(pars_nova, Rsh/1.496e13) # g/cm^3

    B2_bkgr=BRG*jnp.power(jnp.sqrt(Rmin*Rmin+Rsh*Rsh)/(0.35*1.496e13), -2)  # -> Model with background B-field
    # B2_Bell=jnp.sqrt(11.0*jnp.pi*rho*np.power(vsh*xip, 2))                               # -> Model with amplified B-field
    B2=B2_bkgr # +B2_Bell*func_Heaviside(arr_t-tST)                                # -> Model with instability switched on

    dEdt_acc=(qeCGS*B2*jnp.power(vsh, 2))*6.242e+11/(10.0*jnp.pi*3.0e10)

    return dEdt_acc # eV s^-1

# Adiabatic energy loss rate
def func_dE_adi(pars_nova, E, t):
# E (eV) and t(day)

    tST=pars_nova[1]           # day
    alpha=pars_nova[2]         # no unit
    Rmin=pars_nova[5]*1.496e13 # cm
    ter = pars_nova[9]         # day

    vsh=func_vsh(pars_nova,t)*1.0e5    # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    p=jnp.sqrt((E+mp)**2-mp**2)        # eV
    t=jnp.array(t)                     # day

    mask1=(t>=ter) & (t<tST)
    mask2=(t>=tST)

    dEdt_adi=jnp.zeros_like(t)

    dEdt_adi=jnp.where(mask1, -0.2*(p**2/(E+mp))*(Rsh*vsh/(Rmin**2+Rsh**2)), dEdt_adi)
    dEdt_adi=jnp.where(mask2, -0.2*(p**2/(E+mp))*(Rsh*vsh/(Rmin**2+Rsh**2))-0.2*(p**2/(E+mp))*(2.0*alpha/(t*86400.0)), dEdt_adi)

    return dEdt_adi # eV s^-1

def func_dE(pars_nova, E, t):
    return func_dE_acc(pars_nova,E,t) + func_dE_adi(pars_nova,E,t)


# Maximum energy of particle accelerated from the shock calculated with jax
def func_Emax(pars_nova, t):
# t(day)

    ter=pars_nova[9] # day
    E0=1.0e2         # eV

    def dE_dt(Emax, tp):
        return func_dE_acc(pars_nova,Emax,tp/86400.0)+func_dE_adi(pars_nova,Emax,tp/86400.0)

    # Note that we initialize particle energy to be around Emax(t=0 day)=100 eV (this value should be computed self-consistently from temperature of 
    # plasma around the shock but numrical value of Emax for t>~ 1 day does not change for Emax(t=0 day)=100 or 1000 eV).
    t_as=jnp.linspace(ter, t[-1], len(t))
    Emax_as=odeint(dE_dt, E0, t_as*86400.0)

    Emax=jnp.interp(t, t_as, Emax_as, left=E0, right=0.0)
    Emax=Emax[jnp.newaxis,:]
    
    return Emax # eV

# Maximum energy of particle accelerated from the shock calculated with scipy
def func_Emax_np(pars_nova, t):
# t(day)

    sol=solve_ivp(lambda tp,Emax:(func_dE_acc(pars_nova,Emax,tp/86400.0) + func_dE_adi(pars_nova,Emax,tp/86400.0)),(t[0]*86400.0,t[-1]*86400.0),[100.0],method='RK45',dense_output=True)
    # sol=solve_ivp(lambda tp,Emax:(func_dE_adi(pars_nova,Emax,tp/86400.0)),(t[0]*86400.0,t[-1]*86400.0),[100.0],method='RK45',dense_output=True)

    Emax=((sol.sol(t*86400.0)[0]).T)[np.newaxis,:]

    return Emax

# Injection spectrum at the shock
def func_fEp_p(pars_nova, E, t):
# E (eV) and t(day)

    xip=pars_nova[6]     # no unit
    delta=pars_nova[7]   # no unit
    epsilon=pars_nova[8] # no unit
    ter=pars_nova[9]     # day

    # Get the maximum energy over time
    Emax=func_Emax(pars_nova, t) # eV

    # Get the nomalization for the accelerated spectrum
    xmin=jnp.sqrt(pow(1.0e8+mp,2)-mp*mp)/mp 
    xmax=jnp.sqrt(pow(1.0e14+mp,2)-mp*mp)/mp
    x=jnp.logspace(jnp.log10(xmin), jnp.log10(xmax), 5000)

    dx=(x[1:-1]-x[0:-2])[:,jnp.newaxis]
    x=x[0:-2][:,jnp.newaxis]
    Ialpha_p=jnp.sum(pow(x, 4.0-delta)*jnp.exp(-pow(x*mp/Emax, epsilon))*dx/jnp.sqrt(1.0+x*x), axis=0)

    # Get the momentum and speed 
    p=jnp.sqrt(pow(E+mp,2)-mp*mp)
    vp=(p/(E+mp))
    # NEp=np.zeros((len(E),len(t)))

    # Get all the shock dynamics related quantities
    vsh=func_vsh(pars_nova, t)*1.0e5      # cm s^-1
    Rsh=func_Rsh(pars_nova, t)*1.496e13   # cm
    rho=func_rho(pars_nova, Rsh/1.496e13) # g cm^-3

    # Change the dimension to make the integral    
    p=p[:, jnp.newaxis]
    vp=vp[:, np.newaxis]
    Rsh=Rsh[jnp.newaxis, :]
    vsh=vsh[jnp.newaxis, :]
    rho=rho[jnp.newaxis, :]
    Ialpha_p=Ialpha_p[jnp.newaxis, :]

    # Note that Ialpha_p is zero for t<=ter so there will be some nan and we replace it by zero.
    fEp=3.0*jnp.pi*xip*rho*pow(Rsh,2)*pow(vsh,3)*6.242e11*pow(p/mp, 2.0-delta)*jnp.exp(-pow(p/Emax, epsilon))/(mp*mp*vp*Ialpha_p)
    fEp=jnp.where(jnp.isnan(fEp), 0.0, fEp)

    return fEp # eV^-1 s^-1

# Cumulative spectrum of accelerated protons
def func_JEp_p(pars_nova, E, t):

    # Get the momentum and speed 
    p=jnp.sqrt(pow(E+mp,2)-mp*mp)
    vp=p/(E+mp)
    NEp=jnp.zeros((len(E), len(t)))

    # Compute NEp by solving the differential equation
    fEp=func_fEp_p(pars_nova, E, t) # eV^-1 s^-1
    dt=(t[1]-t[0])*86400.0          # s
    NEp=jnp.cumsum(fEp, axis=1)*dt  # eV^-1

    return NEp*vp[:, jnp.newaxis]*3.0e10 # eV^-1 cm s^-1


############################################################################################################################################
# Gamma-ray spectrum
############################################################################################################################################

# Optical luminosiy function of the nova for gamma-ray absorption
def func_LOPT(t):
# t (day)

    mask=(t==0.25)
    LOPT=2.5e36*(1.0-func_Heaviside(t-0.9))+func_Heaviside(t-0.9)*(1.3e39*jnp.power(jnp.abs(t-0.25), -0.28)/jnp.abs(t+0.35))
    LOPT=jnp.where(mask, 2.5e36, LOPT)

    return LOPT # erg s^-1

# Gamma-gamma interaction cross-section
def func_sigma_gg(Eg, Ebg):
    Eg, Ebg=jnp.meshgrid(Eg, Ebg, indexing='ij')
    
    s0=Eg*Ebg/(me*me)
    sigma_gg=jnp.zeros_like(Eg)
    
    mask=(s0>=1.0)
    
    sigma_gg=jnp.where(
        mask,
        (s0+0.5*jnp.log(s0)-(1.0/6.0)+(1.0/(2.0*s0)))*jnp.log(jnp.sqrt(s0)+jnp.sqrt(s0-1.0))-(s0+(4.0/9.0)-(1.0/(9.0*s0)))*jnp.sqrt(1.0-(1.0/s0)),
        0.0
    )
    
    sigma_gg=jnp.where(
        mask,
        sigma_gg*1.5*sigmaT/(s0*s0),
        sigma_gg
    )
    
    return sigma_gg # cm^2

# Photon distribution for gamma-ray absorption
def func_fEtd(Urad, Trad, sigma, Ebg):
# Urad (eV), Trad (eV), sigma (no unit), and Ebg (eV)
    
    fEtd=Urad/(jnp.power(Trad, 2)*func_gamma(4.0+sigma)*func_zeta(4.0+sigma)*(jnp.exp(Ebg/Trad)-1.0))
    fEtd*=jnp.power(Ebg/Trad, 2.0+sigma)
    
    return fEtd  # eV^-1 cm^-3

# Function to calculate the predicted integrated flux
@jit
def func_phi_PPI(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t):

    ter=pars_nova[9]                                             # day
    dE=jnp.append(jnp.diff(E), 0.0)[:, jnp.newaxis, jnp.newaxis] # eV

    # Distance from the nova to Earth
    Ds=pars_nova[12]*3.086e18 # cm

    # Calculate the proton distribution
    JEp=func_JEp_p(pars_nova, E, t)[:, jnp.newaxis, :]      # eV^-1 cm s^-1
    Rsh=func_Rsh(pars_nova, t)*1.496e13                     # cm
    rho=func_rho(pars_nova, t)[jnp.newaxis, jnp.newaxis, :] # g cm^-3

    # Opacity of gamma rays
    TOPT=kB*pars_nova[11]                                                                # eV
    Ebg=jnp.logspace(jnp.log10(TOPT*1.0e-2), jnp.log10(TOPT*1.0e2), 1000)                # eV
    dEbg=jnp.append(jnp.diff(Ebg), 0.0)[jnp.newaxis, :, jnp.newaxis]                     # eV
    UOPT=func_LOPT(t)*6.242e11/(4.0*jnp.pi*pow(Rsh,2)*3.0e10)                            # eV cm^⁻3
    fOPT=func_fEtd(UOPT[jnp.newaxis, :],TOPT,0.0,Ebg[:, jnp.newaxis])[jnp.newaxis, :, :] # eV^-1 cm^-3
    tau_gg=jnp.sum(fOPT*sigma_gg*Rsh[jnp.newaxis, jnp.newaxis, :]*dEbg, axis=1)
    tau_gg=jnp.where((t<=ter)[jnp.newaxis, :], 0.0, tau_gg)

    # Calculate the gamma-ray spectrum
    phi_PPI=jnp.sum((4.0*rho/(4.0*np.pi*Ds**2*mpCGS))*(dE*JEp*eps_nucl)*d_sigma_g, axis=0)
    phi_PPI=phi_PPI*(0.5*(jnp.exp(-1.13*tau_gg)+jnp.exp(-4.45*tau_gg)))

    return phi_PPI, tau_gg

# Function to calculate the chi2 of the integrated flux
@jit
def func_loss(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t):

    phi_PPI, _=func_phi_PPI(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t)

    mask_FLAT=(Eg>=0.1e9) & (Eg<=100.0e9)
    mask_HESS=(Eg>=250.0e9) & (Eg<=2500.0e9)

    dEg=jnp.append(jnp.diff(Eg), 0.0)
    flux_FLAT_PPI=1.0e-3*1.60218e-12*jnp.sum(jnp.where(mask_FLAT[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)
    flux_HESS_PPI=1.60218e-12*jnp.sum(jnp.where(mask_HESS[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)
    flux_FLAT_PPI_interp=jnp.interp(t_FERMI_raw, t, flux_FLAT_PPI, left=0, right=0)
    flux_HESS_PPI_interp=jnp.interp(t_HESS_raw, t, flux_HESS_PPI, left=0, right=0)

    chi2=jnp.sum(((flux_FERMI_raw-flux_FLAT_PPI_interp)/yerr_FERMI_raw)**2)+jnp.sum(((flux_HESS_raw-flux_HESS_PPI_interp)/yerr_HESS_raw)**2)

    return chi2


############################################################################################################################################
# Plots for illustration
############################################################################################################################################

# Plot gamma-ray data
def plot_gamma(pars_nova, phi_PPI, tau_gg, Eg, t, t_day):

    it=np.argmin(np.abs(t-t_day))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    E_FERMI, flux_FERMI, flux_FERMI_lo=np.loadtxt('Data/day%d-FERMI.txt' % int(t_day-0.6),unpack=True,usecols=[0,1,2])
    E_HESS, flux_HESS, flux_HESS_lo=np.loadtxt('Data/day%d-HESS.txt' % int(t_day-0.6),unpack=True,usecols=[0,1,2])

    E_FERMI_data=E_FERMI[flux_FERMI_lo!=0.0]
    flux_FERMI_data=flux_FERMI[flux_FERMI_lo!=0.0]
    flux_FERMI_lo_data=flux_FERMI_lo[flux_FERMI_lo!=0.0]
    E_FERMI_upper=E_FERMI[flux_FERMI_lo==0.0]
    flux_FERMI_upper=flux_FERMI[flux_FERMI_lo==0.0]

    E_HESS_data=E_HESS[flux_HESS_lo!=0.0]
    flux_HESS_data=flux_HESS[flux_HESS_lo!=0.0]
    flux_HESS_lo_data=flux_HESS_lo[flux_HESS_lo!=0.0]
    E_HESS_upper=E_HESS[flux_HESS_lo==0.0]
    flux_HESS_upper=flux_HESS[flux_HESS_lo==0.0]

    ax.errorbar(E_HESS_data, flux_HESS_data, yerr=np.abs(flux_HESS_data-flux_HESS_lo_data), xerr=E_HESS_data*0.0, fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='red', markeredgecolor='black', markersize=10, label=r'{\rm HESS}')
    ax.errorbar(E_FERMI_data, flux_FERMI_data, yerr=np.abs(flux_FERMI_data-flux_FERMI_lo_data), xerr=E_FERMI_data*0.0, fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='green', markeredgecolor='black', markersize=10, label=r'{\rm FERMI}')

    ax.errorbar(E_FERMI_upper, flux_FERMI_upper, yerr=flux_FERMI_upper*0.0, xerr=E_FERMI_upper*0.0, fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='green', markeredgecolor='black', markersize=10)
    ax.errorbar(E_HESS_upper, flux_HESS_upper, yerr=flux_HESS_upper*0.0, xerr=E_HESS_upper*0.0, fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='red', markeredgecolor='black', markersize=10)

    for i in range(len(E_FERMI_upper)):
        ax.annotate("", xy=(E_FERMI_upper[i], 0.95*flux_FERMI_upper[i]), 
                xytext=(E_FERMI_upper[i], flux_FERMI_upper[i] * 0.5),  # Move arrow downward further
                arrowprops=dict(arrowstyle="<-", color='black', lw=2))

    for i in range(len(E_HESS_upper)):
        ax.annotate("", xy=(E_HESS_upper[i], 0.95*flux_HESS_upper[i]), 
                xytext=(E_HESS_upper[i], flux_HESS_upper[i] * 0.5),  # Move arrow downward further
                arrowprops=dict(arrowstyle="<-", color='black', lw=2))

    ax.plot(Eg*1.0e-9,Eg**2*phi_PPI[:,it]*1.6022e-12/(0.5*(np.exp(-1.13*tau_gg[:,it])+np.exp(-4.45*tau_gg[:,it]))),'k--',linewidth=5.0)
    ax.plot(Eg*1.0e-9,Eg**2*phi_PPI[:,it]*1.6022e-12,'k-',linewidth=5.0)

    # # Read the image for data    
    # img=mpimg.imread("Data/data_day%d.png" % int(np.floor(t_day)))
    # img_array = np.mean(np.array(img), axis=2)

    # xmin=-1.0
    # xmax=4.0
    # ymin=-13.0
    # ymax=np.log10(5.0e-9)
    # ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    # ax.set_xticks(np.arange(xmin,xmax+1,1))
    # ax.set_yticks(np.arange(ymin,ymax,1))
    # ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    # ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    # ax.set_aspect(0.8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1,1.0e4)
    ax.set_ylim(1.0e-13,5.0e-9)
    ax.set_xlabel(r'$E_\gamma\, {\rm (GeV)}$',fontsize=fs)
    ax.set_ylabel(r'$E_\gamma^2\phi(E_\gamma) \, ({\rm erg\, cm^{-2}\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', title=r'{\rm Day\, %d}' % int(t_day-0.6), prop={"size":fs}, title_fontsize=fs)
    ax.grid(linestyle='--')

    if(pars_nova[13]==1):
        plt.savefig('Results_jax/new_fg_gamma_day%d_HESS.png' % (t_day))
    else:
        plt.savefig('Results_jax/new_fg_gamma_day%d_DM23.png' % (t_day))

# Plot time profile of gamma-ray integrated flux
def plot_time_gamma(pars_nova, phi_PPI, tau_gg, Eg, t):

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    mask_FLAT=(Eg>=0.1e9) & (Eg<=100.0e9)
    mask_HESS=(Eg>=250.0e9) & (Eg<=2500.0e9)

    print("FERMI band: ", Eg[mask_FLAT][0]*1.0e-9, "-", Eg[mask_FLAT][-1]*1.0e-9, "GeV")
    print("HESS band:  ", Eg[mask_HESS][0]*1.0e-9, "-", Eg[mask_HESS][-1]*1.0e-9, "GeV")

    dEg=jnp.append(jnp.diff(Eg), 0.0)
    flux_FLAT_PPI=1.0e-3*1.60218e-12*jnp.sum(jnp.where(mask_FLAT[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)
    flux_HESS_PPI=1.60218e-12*jnp.sum(jnp.where(mask_HESS[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)

    phi_PPI*=1.0/(0.5*(np.exp(-1.13*tau_gg)+np.exp(-4.45*tau_gg)))
    flux_FLAT_PPI_noabs=1.0e-3*1.60218e-12*jnp.sum(jnp.where(mask_FLAT[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)
    flux_HESS_PPI_noabs=1.60218e-12*jnp.sum(jnp.where(mask_HESS[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)
    ax.plot(t,flux_HESS_PPI_noabs,'r--',linewidth=3.0)
    ax.plot(t,flux_FLAT_PPI_noabs,'g--',linewidth=3.0)

    ax.plot(t,flux_HESS_PPI,'r-',linewidth=3.0,label=r'{\rm Model\, HESS\, band}')
    ax.plot(t,flux_FLAT_PPI,'g-',linewidth=3.0,label=r'{\rm Model\, FERMI\, band}')

    ax.errorbar(t_HESS_raw,flux_HESS_raw,yerr=yerr_HESS_raw,xerr=xerr_HESS_raw,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='red',markeredgecolor='black',markersize=10,label=r'${\rm HESS}$')
    ax.errorbar(t_FERMI_raw,flux_FERMI_raw,yerr=yerr_FERMI_raw,xerr=xerr_FERMI_raw,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='green',markeredgecolor='black',markersize=10,label=r'${\rm FERMI\,(\times 10^{-3})}$')

    data = np.load('test.npz')

    phi_test = data['array1']
    phi_test_unabs = data['array2']

    flux_FLAT_PPI_test=1.0e-3*1.60218e-12*jnp.sum(jnp.where(mask_FLAT[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_test, 0.0), axis=0)
    flux_HESS_PPI_test=1.60218e-12*jnp.sum(jnp.where(mask_HESS[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_test, 0.0), axis=0)

    ax.plot(t,flux_HESS_PPI_test,':', color='orange',linewidth=3.0,label=r'{\rm Model\, HESS\, band}')
    ax.plot(t,flux_FLAT_PPI_test,':', color='yellow',linewidth=3.0,label=r'{\rm Model\, FERMI\, band}')

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

    if(pars_nova[13]==1):
        plt.savefig('Results_jax/newnew_fg_time_gamma_HESS_tST-%.2f_Mdot-%.2e_ter-%.1f_BRG-%.1f.png' % (pars_nova[1], pars_nova[3], pars_nova[9], pars_nova[10]))
    else:
        plt.savefig('Results_jax/newnew_fg_time_gamma_DM23_tST-%.2f_Mdot-%.2e_ter-%.1f_BRG-%.1f.png' % (pars_nova[1], pars_nova[3], pars_nova[9], pars_nova[10]))
    plt.close()
 
# Plot the shock radius
def plot_Rsh(pars_nova, t):

    Rsh=func_Rsh(pars_nova,t)
    
    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,Rsh,'r--',linewidth=3.0)
    ax.plot(t,np.sqrt(1.48**2+Rsh**2),'r--',linewidth=3.0)
    
    # Check model with HESS model
    if(pars_nova[13]==1):
        img = mpimg.imread("Data/data_Rsh.png")
        img_array = np.mean(np.array(img), axis=2)

        xmin=0.0
        xmax=5.0
        ymin=0.0
        ymax=14.0
        ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
        ax.set_aspect(0.3)

    ax.legend()
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$R_{\rm sh} \, ({\rm au})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    if(pars_nova[13]==1):
        plt.savefig('Results_jax/fg_jax_Rsh_HESS.png')
    else:
        plt.savefig('Results_jax/fg_jax_Rsh_DM23.png')

    plt.close()

# Plot the shock speed
def plot_vsh(pars_nova, t):

    vsh=func_vsh(pars_nova,t)

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,vsh,'r--',linewidth=3.0)

    # Check model with HESS model
    if(pars_nova[13]==1):
        img=mpimg.imread("Data/data_vsh.png")
        img_array = np.mean(np.array(img), axis=2)

        xmin=0.0
        xmax=6.0
        ymin=2500.0
        ymax=5000.0
        ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
        ax.set_aspect((xmax-xmin)/(ymax-ymin))

    # ax.set_xlim(0,5)
    # ax.set_ylim(1.0e1,2.0e5)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$v_{\rm sh} \, ({\rm km/s})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    if(pars_nova[13]==1):
        plt.savefig('Results_jax/fg_jax_vsh_HESS.png')
    else:
        plt.savefig('Results_jax/fg_jax_vsh_DM23.png')

    plt.close()

# Plot the shock density
def plot_rho(pars_nova, t):

    Rsh=func_Rsh(pars_nova,t) # au
    rho=func_rho(pars_nova,Rsh)/mpCGS # cm^-3

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,rho,'r--',linewidth=3.0)

    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$n_{\rm w} \, ({\rm cm^{-3}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs})
    ax.grid(linestyle='--')

    if(pars_nova[13]==1):
        plt.savefig('Results_jax/fg_jax_rho_HESS.png')
    else:
        plt.savefig('Results_jax/fg_jax_rho_DM23.png')

    plt.close()


if __name__ == "__main__":

    start_time=time.time()

    # Initialized parameters for RS Ophiuchi 2021  
    vsh0=4500.0  # km/s    -> Initial shock speed
    tST=1.8      # day     -> Time where shock transition to Sedov-Taylor phase
    alpha=0.66   # no unit -> Index for time profile of shock speed
    Mdot=6.0e-7  # Msol/yr -> Mass loss rate of red giant
    vwind=20.0   # km/s    -> Wind speed of red giant
    Rmin=1.48    # au      -> Distance between red giant and white dwarf
    xip=0.1      # no unit -> Fraction of shock ram pressure converted into cosmic-ray energy
    delta=4.4    # no unit -> Injection spectrum index
    epsilon=1.0  # no unit -> Index of the exp cut-off for injection spectrum
    BRG=6.5      # G       -> Magnetic field srength at the pole of red giant
    TOPT=1.0e4   # K       -> Temperature of the optical photons 
    ter=-0.3     # day     -> Shock formation time
    Ds=1.4e3     # pc      -> Distance to Earth of the nova
    model_name=0 # no unit -> Model name (1=HESS and 0=DM23)
    pars_nova=[vsh0, tST, alpha, Mdot, vwind, Rmin, xip, delta, epsilon, ter, BRG, TOPT, Ds, model_name]

    # Define the time and energy ranges -> note that it is required that t[0]<=ter 
    t=np.linspace(-1.0,30.0,3101) # day
    E=np.logspace(8,14,601)       # eV
    Eg=np.logspace(8,14,601)      # eV

    # Gamma-ray production cross-section
    eps_nucl=jnp.array(gt.func_enhancement(E))[:, jnp.newaxis, jnp.newaxis] # no unit
    d_sigma_g=jnp.array(gt.func_d_sigma_g(E, Eg))[:, :, jnp.newaxis]        # cm^2/eV

    # Gamma-gamma cross section
    TOPT=kB*pars_nova[11] # eV
    Ebg=jnp.logspace(jnp.log10(TOPT*1.0e-2), jnp.log10(TOPT*1.0e2), 1000) # eV
    dEbg=jnp.append(jnp.diff(Ebg), 0.0)[jnp.newaxis,:,jnp.newaxis]        # eV
    sigma_gg=gt.func_sigma_gg(Eg, Ebg)[:,:,jnp.newaxis]                   # cm^2

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

    vsh=func_vsh(pars_nova, t)
    Rsh=func_Rsh(pars_nova, t)

    # Plot the best fit parameters
    fEp=func_fEp_p(pars_nova, E, t)
    JEp=func_JEp_p(pars_nova, E, t) # eV^-1 cm s^-1

    phi_PPI, tau_gg=func_phi_PPI(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t)

    plot_gamma(pars_nova, phi_PPI, tau_gg, Eg, t, 1.6)
    plot_gamma(pars_nova, phi_PPI, tau_gg, Eg, t, 3.6)
    plot_gamma(pars_nova, phi_PPI, tau_gg, Eg, t, 5.6)
    plot_time_gamma(pars_nova, phi_PPI, tau_gg, Eg, t)
    chi2=func_loss(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t)

    # np.save('phi_PPI_%s.npy' % pars_nova[13], phi_PPI*np.exp(-tau_gg))
    # phi_PPI=np.load('phi_PPI_%s.npy' % pars_nova[13])

    # plot_Rsh(pars_nova,t)
    # plot_vsh(pars_nova,t)
    # plot_rho(pars_nova,t)

    Emax_np=func_Emax_np(pars_nova,t)
    Emax_jnp=func_Emax(pars_nova,t)
    
    data=np.load('test_Emax.npz')

    Emax_test=data['array2']

    # print(Emax_np[0])
    print(ter, Emax_jnp[0])
    # print(func_dE_acc(pars_nova,1.0e2,0.0)+func_dE_adi(pars_nova,1.0e2,0.0))

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,Emax_test[0],'k-',linewidth=3.0, label='Numpy')
    ax.plot(t,Emax_jnp[0],'-',linewidth=8.0, color='darkgreen')
    ax.plot(t,Emax_np[0],'r:',linewidth=3.0, label='Numpy')

    # ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$E_{\rm max} \, ({\rm eV})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.set_ylim(5.0e1,5.0e12)
    ax.set_xlim(-1.0,10.0)
    # ax.legend(loc='lower right', prop={"size":fs})
    ax.grid(linestyle='--')

    if(pars_nova[13]==1):
        plt.savefig('fg_jax_Emax_HESS.png')
    else:
        plt.savefig('Results_jax/new_fg_jax_Emax_DM23.png')

    plt.close()

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    ax.set_xlim(np.log10(1.0),np.log10(50.0))

    t_vsh, vsh_a, err_vsh_a, vsh_b, err_vsh_b=np.loadtxt('vsh_line.txt',unpack=True,usecols=[0,1,2,3,4])
    ax.errorbar(np.log10(t_vsh),np.log10(vsh_a),yerr=err_vsh_a/vsh_a,xerr=t_vsh*0.0,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='red',markeredgecolor='black',markersize=10,label=r'${\rm H\alpha}$')
    ax.errorbar(np.log10(t_vsh),np.log10(vsh_b),yerr=err_vsh_b/vsh_b,xerr=t_vsh*0.0,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='green',markeredgecolor='black',markersize=10,label=r'${\rm H\beta}$')

    img=mpimg.imread("Data/data_vsh_Xray.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=np.log10(1.0)
    xmax=np.log10(50.0)
    ymin=np.log10(700.0)
    ymax=np.log10(5000.0)
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.plot(np.log10(t), np.log10(func_vsh(pars_nova, t)), '-', color='orange', linewidth=8)

    ax.set_aspect((xmax-xmin)/(ymax-ymin))
    ax.set_xticks([np.log10(1), np.log10(10)])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: r'$10^{%d}$' % int(x)))
    ax.set_yticks([np.log10(1000.0), np.log10(5000.1)])
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: r'$%d$' % int(10**x)))

    ax.legend()
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$v_{\rm sh} \, ({\rm km/s})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results/fg_jax_vsh_Xray.png')

    # Record the ending time
    end_time=time.time()

    # Calculate the elapsed time
    elapsed_time=end_time-start_time

    print("Elapsed time:", elapsed_time, "seconds")