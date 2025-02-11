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
from jax import jit
import matplotlib.ticker as ticker
import diffrax
import scipy as sp
from jax import grad
import optax

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
hP=4.1356676966e-15 # eV s


############################################################################################################################################
# Prepare data from HESS and FERMI and optical luminosity
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

t_data, LOPT_data=np.loadtxt("Data/LOPT.dat",unpack=True,usecols=[0,1])


############################################################################################################################################
# Nova shock model
############################################################################################################################################

# Nova shock speed
def func_vsh(pars_nova, t):
# t (day)

    # Parameters for nova shocks
    vsh0=pars_nova[0]  # km/s
    tST=pars_nova[1]   # day
    alpha=pars_nova[2] # no unit
    ter=pars_nova[9]   # day

    # Redefine time with respect to ter (add small value to avoid tau=0 day)
    tau=t-ter+1.0e-3
    tauST=tST-ter

    # Smooth masks using tanh functions
    smoothing_factor=500.0
    mask1=0.5*(1.0+jnp.tanh(smoothing_factor*tau))
    mask2=0.5*(1.0+jnp.tanh(smoothing_factor*(tau-tauST)))

    # Shock speed
    vsh=mask1*vsh0*(tau/tauST)**-(mask2*alpha)
    
    return vsh # km/s

# Nova shock radius
def func_Rsh(pars_nova, t):
# t (day)

    # Parameters for nova shocks
    vsh0=pars_nova[0]  # km/s
    tST=pars_nova[1]   # day
    alpha=pars_nova[2] # no unit
    ter=pars_nova[9]   # day
    t=jnp.array(t)     # day

    # Redefine time with respect to ter (add small value to avoid tau=0 day)
    tau=t-ter+1.0e-3
    tauST=tST-ter

    # Smooth masks using tanh functions
    smoothing_factor=500.0
    mask1=0.5*(1.0+jnp.tanh(smoothing_factor*tau))
    mask2=0.5*(1.0+jnp.tanh(smoothing_factor*(tau-tauST)))

    # Shock radius
    Rsh=mask1*(vsh0*tauST*(jnp.power(tau/tauST, 1.0-mask2*alpha)-mask2*alpha)/(1.0-mask2*alpha))

    return Rsh*86400.0*6.68e-9 # au

# Density profile of the red giant wind
def func_rho(pars_nova, r):
# Mdot (Msol/yr), vwind (km/s), and r (au)    

    # Parameters for nova shocks
    Mdot=pars_nova[3]*1.989e33/(365.0*86400.0) # g/s 
    vwind=pars_nova[4]*1.0e5                   # cm/s
    Rmin=pars_nova[5]*1.496e13                 # cm

    # Density profile upstream?
    rho=Mdot/(4.0*jnp.pi*vwind*(Rmin**2+(r*1.496e13)**2)) 

    return rho # g/cm^3


############################################################################################################################################
# Particle acceleration model
############################################################################################################################################

# Acceleration rate of protons with background field
def func_dE_acc(pars_nova, E, t):
# E (eV) and t(day)

    Rmin=pars_nova[5]*1.496e13 # cm
    BRG=pars_nova[10]          # G

    vsh=func_vsh(pars_nova, t)*1.0e5      # cm/s
    Rsh=func_Rsh(pars_nova, t)*1.496e13   # cm

    B2=BRG*jnp.power(jnp.sqrt(Rmin*Rmin+Rsh*Rsh)/(0.35*1.496e13), -2)  # -> Model with background B-field
    dEdt_acc=(qeCGS*B2*jnp.power(vsh, 2))*6.242e+11/(10.0*jnp.pi*3.0e10)

    return dEdt_acc # eV s^-1

# Acceleration rate of protons with Bell instability
def func_dE_acc_bell(pars_nova, E, t):
# E (eV) and t(day)

    tST=pars_nova[1]           # day
    Rmin=pars_nova[5]*1.496e13 # cm
    xip=pars_nova[6]           # no unit
    BRG=pars_nova[10]          # G

    vsh=func_vsh(pars_nova, t)*1.0e5      # cm/s
    Rsh=func_Rsh(pars_nova, t)*1.496e13   # cm
    rho=func_rho(pars_nova, Rsh/1.496e13) # g/cm^3

    # B2_bkgr=BRG*jnp.power(jnp.sqrt(Rmin*Rmin+Rsh*Rsh)/(0.35*1.496e13), -2)  # -> Model with background B-field
    B2_Bell=jnp.sqrt(11.0*jnp.pi*rho*jnp.power(vsh*xip, 2))                               # -> Model with amplified B-field
    B2=B2_Bell # +B2_Bell*func_Heaviside(arr_t-tST)                                # -> Model with instability switched on

    dEdt_acc=(qeCGS*B2*jnp.power(vsh, 2))*6.242e+11/(10.0*jnp.pi*3.0e10)

    return dEdt_acc # eV s^-1

# Adiabatic energy loss rate
def func_dE_adi(pars_nova, E, t):
# E (eV) and t(day)

    # Parameters for nova shocks
    tST=pars_nova[1]           # day
    alpha=pars_nova[2]         # no unit
    Rmin=pars_nova[5]*1.496e13 # cm
    ter=pars_nova[9]           # day

    vsh=func_vsh(pars_nova,t)*1.0e5    # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    p=jnp.sqrt((E+mp)**2-mp**2)        # eV

    # Redefine time with respect to ter (add small value to avoid tau=0 day)
    tau=t-ter+1.0e-3
    tauST=tST-ter

    # Smooth masks using tanh functions
    smoothing_factor=500.0
    mask1=0.5*(1.0+jnp.tanh(smoothing_factor*(tau)))
    mask2=0.5*(1.0+jnp.tanh(smoothing_factor*(tau-tauST)))
 
    term=2.0*alpha*mask2/(tau*86400.0) 
    dEdt_adi=mask1*(-0.2*(p**2/(E+mp)))*((Rsh*vsh/(Rmin**2+Rsh**2))+term)

    return dEdt_adi # eV s^-1

# Derivative of adiabatic energy loss rate with respect to kinetric energy
def func_dE_adi_dE(pars_nova, E, t):
# E (eV) and t(day)

    # Parameters for nova shocks
    tST=pars_nova[1]           # day
    alpha=pars_nova[2]         # no unit
    Rmin=pars_nova[5]*1.496e13 # cm
    ter=pars_nova[9]           # day

    vsh=func_vsh(pars_nova,t)*1.0e5    # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    p=jnp.sqrt((E+mp)**2-mp**2)        # eV
    t=jnp.array(t)                     # day

    # Redefine time with respect to ter
    tau=t-ter+1.0e-3
    tauST=tST-ter

    # Smooth masks using tanh functions
    smoothing_factor=500.0
    mask1=0.5*(1.0+jnp.tanh(smoothing_factor*(tau)))
    mask2=0.5*(1.0+jnp.tanh(smoothing_factor*(tau-tauST)))
 
    # Small number is added to tau to avoid nan
    term=2.0*alpha*mask2/((tau+1.0e-3)*86400.0) 

    dEdt_adi_dE=mask1*(-0.2*(2.0-(p/(E+mp))**2))*((Rsh*vsh/(Rmin**2+Rsh**2))+term)

    return dEdt_adi_dE # s^-1

# Maximum energy of particle accelerated from the shock calculated with diffrax
def func_Emax(pars_nova, t):
# t(day)

    E0=5.0e6 # eV

    def dE_dt(tp, Emax, args):
        return (func_dE_acc(pars_nova, Emax, tp)+func_dE_adi(pars_nova, Emax, tp))*86400.0 # eV day^-1

    term=diffrax.ODETerm(dE_dt)
    solver=diffrax.Dopri5()
    saveat=diffrax.SaveAt(ts=t)        
    solution=diffrax.diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[1]-t[0], y0=E0, saveat=saveat, args=pars_nova)
    Emax=solution.ys

    return Emax # eV

# Initial energy after adiabatic energy loss
def func_E0(pars_nova, E, t):
# E (eV) and t (day)

    def dE_dt_adi(tp, Ep, args):
        pars_nova=args
        return func_dE_adi(pars_nova, Ep, tp)*86400.0 # eV day^-1

    def solve_single_ode_E(E0_single):
        term=diffrax.ODETerm(dE_dt_adi)
        solver=diffrax.Dopri5()
        saveat=diffrax.SaveAt(ts=t)
        solution=diffrax.diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[1]-t[0], y0=E0_single, saveat=saveat, args=pars_nova)
        return solution.ys

    E0=solve_single_ode_E(E)
    
    return E0 # eV

# Define the function f(t, E, pars) for the right-hand side of the differential equation
def func_fEp_p(t, Ep, pars_nova):

    xip=pars_nova[6]     # no unit
    delta=pars_nova[7]   # no unit
    epsilon=pars_nova[8] # no unit

    Emax=func_Emax(pars_nova, t)

    # Get the nomalization for the accelerated spectrum
    xmin=jnp.sqrt(pow(1.0e8+mp,2)-mp*mp)/mp 
    xmax=jnp.sqrt(pow(1.0e14+mp,2)-mp*mp)/mp
    x=jnp.logspace(jnp.log10(xmin), jnp.log10(xmax), 500)

    dx=(x[1:-1]-x[0:-2])[:,jnp.newaxis]
    x=x[0:-2][:,jnp.newaxis]
    Ialpha_p=jnp.sum(pow(x, 4.0-delta)*jnp.exp(-pow(x*mp/Emax, epsilon))*dx/jnp.sqrt(1.0+x*x), axis=0)

    # Get the momentum and speed 
    p=jnp.sqrt(pow(Ep+mp,2)-mp*mp)
    vp=(p/(Ep+mp))

    # Get all the shock dynamics related quantities
    vsh=func_vsh(pars_nova, t)*1.0e5     # cm s^-1
    Rsh=func_Rsh(pars_nova, t)*1.496e13  # cm
    rho=func_rho(pars_nova, Rsh/1.496e13) # g cm^-3

    # Note that Ialpha_p is zero for t<=ter so there will be some nan and we replace it by zero.
    fEp=3.0*jnp.pi*xip*rho*pow(Rsh,2)*pow(vsh,3)*6.242e11*pow(p/mp, 2.0-delta)*jnp.exp(-pow(p/Emax, epsilon))/(mp*mp*vp*Ialpha_p)

    return fEp # eV^-1 s^-1

# Cummulated particle flux solved with Runge-Kutta 4
def func_JEp_p_rk4(pars_nova, E, t):

    # Define the function f(t, E, pars) for the right-hand side of the differential equation
    def func_fEp_p(t, Ep, pars_nova):

        xip=pars_nova[6]     # no unit
        delta=pars_nova[7]   # no unit
        epsilon=pars_nova[8] # no unit

        Emax=func_Emax(pars_nova, t)

        # Get the nomalization for the accelerated spectrum
        xmin=jnp.sqrt(pow(1.0e8+mp,2)-mp*mp)/mp 
        xmax=jnp.sqrt(pow(1.0e14+mp,2)-mp*mp)/mp
        x=jnp.logspace(jnp.log10(xmin), jnp.log10(xmax), 500)

        dx=(x[1:-1]-x[0:-2])[:,jnp.newaxis]
        x=x[0:-2][:,jnp.newaxis]
        Ialpha_p=jnp.sum(pow(x, 4.0-delta)*jnp.exp(-pow(x*mp/Emax, epsilon))*dx/jnp.sqrt(1.0+x*x), axis=0)

        # Get the momentum and speed 
        p=jnp.sqrt(pow(Ep+mp,2)-mp*mp)
        vp=(p/(Ep+mp))

        # Get all the shock dynamics related quantities
        vsh=func_vsh(pars_nova, t)*1.0e5     # cm s^-1
        Rsh=func_Rsh(pars_nova, t)*1.496e13  # cm
        rho=func_rho(pars_nova, Rsh/1.496e13) # g cm^-3

        # Note that Ialpha_p is zero for t<=ter so there will be some nan and we replace it by zero.
        fEp=3.0*jnp.pi*xip*rho*pow(Rsh,2)*pow(vsh,3)*6.242e11*pow(p/mp, 2.0-delta)*jnp.exp(-pow(p/Emax, epsilon))/(mp*mp*vp*Ialpha_p)
        fEp=jnp.where(jnp.isnan(fEp), 0.0, fEp)

        return fEp # eV^-1 s^-1

    # Solver function to solve dN/dt for a given energy level E and parameters pars
    def solve_for_energy(E_single):
        dt=t[1]-t[0]
        k1=dt*86400.0*func_fEp_p(t, E_single, pars_nova)
        k2=dt*86400.0*func_fEp_p(t+0.5*dt, E_single, pars_nova)
        k3=dt*86400.0*func_fEp_p(t+0.5*dt, E_single, pars_nova)
        k4=dt*86400.0*func_fEp_p(t+dt, E_single, pars_nova)
        
        return jnp.cumsum((k1+2.0*k2+2.0*k3+k4)/6.0, axis=0)  # This will be N(t, E_single) for this energy level

    NEp=jax.vmap(solve_for_energy)(E)

    # Get the momentum and speed 
    E0=E # func_E0(pars_nova, E, t)   # eV
    p0=jnp.sqrt(pow(E0+mp,2)-mp*mp) # eV
    vp0=p0/(E0+mp)

    return NEp*vp0[:, jnp.newaxis]*3.0e10 # eV^-1 cm s^-1

# Cummulated particle flux with adiabatic energy loss solved with Runge-Kutta 4
def func_JEp_p_ark(pars_nova, E, t):
# E (eV) and t (day)

    # Time step for RK4 solver
    dt=t[1]-t[0]                     # day

    # Get the momentum and speed 
    E0=func_E0(pars_nova, E, t).T   # eV
    p0=jnp.sqrt(pow(E0+mp,2)-mp*mp) # eV
    vp0=p0/(E0+mp)                  # no unit

    # Integration factor to solve for NEp
    def func_It(E_index):
        return jnp.exp(jnp.cumsum(func_dE_adi_dE(pars_nova, E0[E_index, :], t)*dt*86400.0)) # no unit
    
    It=jax.vmap(func_It)(jnp.arange(len(E)))

    # Define the function f(t, E, pars) for the right-hand side of the differential equation
    def func_fEp_p(t, Ep, pars_nova):

        xip=pars_nova[6]     # no unit
        delta=pars_nova[7]   # no unit
        epsilon=pars_nova[8] # no unit

        Emax=func_Emax(pars_nova, t)

        # Get the nomalization for the accelerated spectrum
        xmin=jnp.sqrt(pow(1.0e8+mp,2)-mp*mp)/mp 
        xmax=jnp.sqrt(pow(1.0e14+mp,2)-mp*mp)/mp
        x=jnp.logspace(jnp.log10(xmin), jnp.log10(xmax), 500)

        dx=(x[1:-1]-x[0:-2])[:,jnp.newaxis]
        x=x[0:-2][:,jnp.newaxis]
        Ialpha_p=jnp.sum(pow(x, 4.0-delta)*jnp.exp(-pow(x*mp/Emax, epsilon))*dx/jnp.sqrt(1.0+x*x), axis=0)

        # Get the momentum and speed 
        p=jnp.sqrt(pow(Ep+mp,2)-mp*mp)
        vp=(p/(Ep+mp))

        # Get all the shock dynamics related quantities
        vsh=func_vsh(pars_nova, t)*1.0e5     # cm s^-1
        Rsh=func_Rsh(pars_nova, t)*1.496e13  # cm
        rho=func_rho(pars_nova, Rsh/1.496e13) # g cm^-3

        # Note that Ialpha_p is zero for t<=ter so there will be some nan and we replace it by zero.
        fEp=3.0*jnp.pi*xip*rho*pow(Rsh,2)*pow(vsh,3)*6.242e11*pow(p/mp, 2.0-delta)*jnp.exp(-pow(p/Emax, epsilon))/(mp*mp*vp*Ialpha_p)
        fEp=jnp.where(jnp.isnan(fEp), 0.0, fEp)

        return fEp # eV^-1 s^-1

    # Define the derivative function for d[N(t)*I(t)]/dt
    def dN_dt(t, E_index):
        return func_fEp_p(t, E0[E_index, :], pars_nova)*It[E_index, :]  

    # Solver function to solve dN/dt for a given energy level E and parameters pars
    def solve_for_energy(E_index):
        k1=dt*86400.0*dN_dt(t, E_index)
        k2=dt*86400.0*dN_dt(t+0.5*dt, E_index)
        k3=dt*86400.0*dN_dt(t+0.5*dt, E_index)
        k4=dt*86400.0*dN_dt(t+dt, E_index)
        
        return jnp.cumsum((k1+2.0*k2+2.0*k3+k4)/6.0, axis=0)  # This will be N(t, E_single) for this energy level

    NEp=jax.vmap(solve_for_energy)(jnp.arange(len(E)))/It
    JEp=NEp*vp0*3.0e10                                    # eV^-1 cm s^-1

    def interp_JEp(t_index):
        return jnp.interp(E, E0[:,t_index], JEp[:,t_index], left=0.0, right=0.0) 

    JEp_interp=jax.vmap(interp_JEp)(jnp.arange(len(t)))

    return JEp_interp.T # eV^-1 cm s^-1


############################################################################################################################################
# Gamma-ray spectrum
############################################################################################################################################

# Optical luminosiy function of the nova for gamma-ray absorption
def func_LOPT(t):

    # Smooth masks using tanh functions
    smoothing_factor=40.0
    mask=0.5*(1.0+jnp.tanh(smoothing_factor*(t-0.88)))

    # Luminosity function fitted with data from Cheung et al. 2022
    # LOPT=2.5e36+mask*3.9e38*(t/2.0)**(-mask)
    LOPT=2.5e36+mask*7.8e38*(t/1.0)**(-mask)

    return LOPT # erg s^-1

# Optical luminosiy function of the nova for gamma-ray absorption
def func_LOPT_new(t):

    # Smooth masks using tanh functions
    smoothing_factor=40.0
    mask=0.5*(1.0+jnp.tanh(smoothing_factor*(t-0.88)))

    t_data_interp=t_data-0.25
    LOPT_data_interp=LOPT_data[t_data_interp>0.0]
    t_data_interp=t_data_interp[t_data_interp>0.0]
    LOPT=jnp.where(t>=0.0, 2.5e36+mask*jnp.exp(jnp.interp(jnp.log(t), jnp.log(t_data_interp), jnp.log(LOPT_data_interp), left=0.0, right=0.0)), 2.5e36)

    return LOPT # erg s^-1

# # Auxiliary function for energy density of optical photons
# def func_gOPT(x):
# # x=r/Rsh
#     return (1.0/x)-((1.0/x**2)-1.0)*np.log(np.sqrt(np.abs((x-1.0)/(x+1.0))))

# Auxiliary function for energy density of optical photons
def func_gOPT(x):
    return (1.0/x)-((1.0/x**2)-1.0)*jnp.log(jnp.sqrt(jnp.abs((x-1.0)/(x+1.0))))

# Energy density of optical photons assuming optical photons are uniformly emitted with the photosphere of radius Rph
def func_uOPT_rt(pars_nova, r, t):

    LOPT=func_LOPT(t)                            # erg/s
    uOPT=LOPT/(4.0*np.pi*(r*1.496e13)**2*3.0e10) # erg/cm^3
    Rph=pars_nova[13]                            # au
    
    uOPT*=1.5*(r/Rph)**3*func_gOPT(r/Rph)

    return uOPT # erg/cm^3

# Energy density of optical photons for 1/r^2 profile 
def func_uOPT_r2(pars_nova, r, t):

    LOPT=func_LOPT(t)                            # erg/s
    uOPT=LOPT/(4.0*np.pi*(r*1.496e13)**2*3.0e10) # erg/cm^3
    
    return uOPT # erg/cm^3

# Gamma-gamma interaction cross-section from Aharonian 2013
def func_sigma_gg(Eg, Ebg):
# Eg (eV) and Ebg (eV)

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

# Gamma-gamma interaction cross-section from Gould 1967
def func_sigma_gg_G67(Eg, Ebg):
    
    s0=Eg[:, np.newaxis]*Ebg[np.newaxis, :]/me**2
    s0_1D=s0.ravel()
    Phi_bar_arr=np.zeros_like(s0_1D)
    for i in range(len(s0_1D)):
        Phi_bar_arr[i]=Phi_bar(s0_1D[i])
    Phi_bar_arr=Phi_bar_arr.reshape(s0.shape)
    sigma_gg=(3.0*sigma_Thomson_CGS/8.0)*(1.0/s0)**2*Phi_bar_arr

    return sigma_gg # cm^2

def func_sigma_gg_C90(Eg, Ebg):

    s0=Eg[:, np.newaxis]*Ebg[np.newaxis, :]/me**2
    x=s0.ravel()
    mask=x>=1.0
    sigma_gg=jnp.zeros_like(x)
    sigma_gg=jnp.where(
        mask,
        sigma_Thomson_CGS*(x-1.0)**1.5*x**-2.5*(0.5*x**-0.5+0.75*jnp.log(x)),
        sigma_gg
    )

    return sigma_gg.reshape(s0.shape) # cm^2


# Photon distribution for gamma-ray absorption
def func_fEtd(Urad, Trad, sigma, Ebg):
# Urad (eV), Trad (eV), sigma (no unit), and Ebg (eV)
    
    fEtd=Urad/(jnp.power(Trad, 2)*func_gamma(4.0+sigma)*func_zeta(4.0+sigma)*(jnp.exp(Ebg/Trad)-1.0))
    fEtd*=jnp.power(Ebg/Trad, 2.0+sigma)
    
    return fEtd # eV^-1 cm^-3

# Auxiliary function for the opacity tau_1
def func_inner_int1(pars_nova, eps, t):

    Rph=pars_nova[13]          # au
    Rsh=func_Rsh(pars_nova, t) # au

    def func_inner_int_single(eps_single):
        s=jnp.linspace(0.0*Rsh, 100.0*Rsh, 1001, axis=0)                                                              # au
        ds=jnp.diff(s, append=s[-1:, :], axis=0)  
        r=jnp.sqrt(s**2-2.0*s*Rsh[jnp.newaxis, :]*jnp.sqrt(1.0-eps_single**2)+Rsh[jnp.newaxis, :]**2) # au

        return jnp.sum((1.5*r/Rph)*func_gOPT(r/Rph)*ds/Rph, axis=0)
        
    return jax.vmap(func_inner_int_single)(eps) # no unit

# Auxiliary function for the opacity tau_2
def func_inner_int2(pars_nova, eps, t):

    Rph=pars_nova[13]          # au
    Rsh=func_Rsh(pars_nova, t) # au

    def func_inner_int_single(eps_single):
        s=jnp.linspace(2.0*Rsh*jnp.sqrt(1.0-eps_single**2), 100.0*Rsh, 1001, axis=0)                                                              # au
        ds=jnp.diff(s, append=s[-1:, :], axis=0)  
        r=jnp.sqrt(s**2-2.0*s*Rsh[jnp.newaxis, :]*jnp.sqrt(1.0-eps_single**2)+Rsh[jnp.newaxis, :]**2) # au

        return jnp.sum((1.5*r/Rph)*func_gOPT(r/Rph)*ds/Rph, axis=0)
        
    return jax.vmap(func_inner_int_single)(eps) # no unit

# Average opacity tau_1 for gamma rays passing through the shock downstream
@jax.jit
def func_tau_ph1(pars_nova, tau_ph, t):

    tau_ph_sparse=tau_ph[:,::10]
    eps=jnp.linspace(0.001, 0.999999, 1000)
    deps=jnp.append(jnp.diff(eps), 0.0)
    inner_int=jnp.exp(-tau_ph_sparse[jnp.newaxis,:,:]*func_inner_int1(pars_nova, eps, t[::10])[:,jnp.newaxis,:])
    tau_ph1=jnp.sum(inner_int*eps[:, jnp.newaxis, jnp.newaxis]*deps[:, jnp.newaxis, jnp.newaxis]/jnp.sqrt(1.0-(eps[:, jnp.newaxis, jnp.newaxis])**2), axis=0)

    def interp_tau_ph1(Eg_index):
        return jnp.interp(t, t[::10], tau_ph1[Eg_index, :], left=0.0, right=0.0) 

    NEg, _=tau_ph.shape
    tau_ph1_full=jax.vmap(interp_tau_ph1)(jnp.arange(NEg))

    return tau_ph1_full

# Average opacity tau_2 for gamma rays passing directly into the shock upstream
@jax.jit
def func_tau_ph2(pars_nova, tau_ph, t):

    tau_ph_sparse=tau_ph[:,::10]
    eps=jnp.linspace(0.001, 0.999999, 1000)
    deps=jnp.append(jnp.diff(eps), 0.0)
    inner_int=jnp.exp(-tau_ph_sparse[jnp.newaxis,:,:]*func_inner_int2(pars_nova, eps, t[::10])[:,jnp.newaxis,:])
    tau_ph2=jnp.sum(inner_int*eps[:, jnp.newaxis, jnp.newaxis]*deps[:, jnp.newaxis, jnp.newaxis]/jnp.sqrt(1.0-(eps[:, jnp.newaxis, jnp.newaxis])**2), axis=0)

    def interp_tau_ph2(Eg_index):
        return jnp.interp(t, t[::10], tau_ph2[Eg_index, :], left=0.0, right=0.0) 

    NEg, _=tau_ph.shape
    tau_ph2_full=jax.vmap(interp_tau_ph2)(jnp.arange(NEg))

    return tau_ph2_full

me_g=9.1e-28
k_B=1.38e-16
sigma_Thomson_CGS=6.652453e-25

# Gould & Schreder 1967
def Phi_bar(s0):
    # PB when s0 smaller than 1. (can happen with 0.99999999)
    # s0=max(s0,1.)
    beta0=(1.0-1./s0)**0.5
    w0=(1+beta0)/(1-beta0)

    inte=0.
    W0=np.logspace(np.log10(1.), np.log10(w0), 50)
    for i in range (0,len(W0)-1):
        inte=inte+(np.log(W0[i]+1)/W0[i]+np.log(W0[i+1]+1)/W0[i+1])*(W0[i+1]-W0[i])/2.
        
    # temp=(1+beta0**2)/(1-beta0**2.)*np.log(w0)-beta0**2.*np.log(w0)-(np.log(w0))**2.-(4.*beta0)/(1-beta0**2.)+ 2*beta0+4*np.log(w0)*np.log(w0+1)-inte
    temp=np.where(s0>=1.0, (1+beta0**2)/(1-beta0**2.)*np.log(w0)-beta0**2.*np.log(w0)-(np.log(w0))**2.-(4.*beta0)/(1-beta0**2.)+ 2*beta0+4*np.log(w0)*np.log(w0+1)-inte, 0.0)

    return temp

def f_nu(nu_arr):

    inte=np.zeros_like(nu_arr)
    f_nu_arr=np.zeros_like(nu_arr)
    for j in range(len(nu_arr)):        
        EPS=np.logspace(np.log10(nu_arr[j]),np.log10(1000*nu_arr[j]),50)
        for i in range (0,len(EPS)-1):
            inte[j]=inte[j]+ ((np.exp(EPS[i])-1.)**-1*Phi_bar(EPS[i]/nu_arr[j])+(np.exp(EPS[i+1])-1.)**-1*Phi_bar(EPS[i+1]/nu_arr[j]))*(EPS[i+1]-EPS[i])/2.
        
        f_nu_arr[j]=nu_arr[j]**2*inte[j]

    return f_nu_arr

def kappa_gg(pars_nova, Egamma, t):
# t in sec E in erg, T in Kelvin
# we find a diffrence with Tatischeff 2009 (formule 55,56) -> pi**5 instead of pi**4. -> I'm probably wrong and Vincent is probably right

    Tbb=pars_nova[11]                   # K
    Rph=pars_nova[13]*1.496e13          # cm
    Rsh=func_Rsh(pars_nova, t)*1.496e13 # cm

    nu=me_g**2*(3.0e10)**4/(Egamma*k_B*Tbb)
    # UOPT=((pars_nova[12]/1.6e3)**2)*func_LOPT(t)*1.5*Rsh*func_gOPT(Rsh/Rph)/(4.0*np.pi*pow(Rph,3)*3.0e10) # erg cm^⁻3
    UOPT=((pars_nova[12]/1.6e3)**2)*func_LOPT(t)/(4.0*np.pi*pow(Rsh,2)*3.0e10) # erg cm^⁻3
    f_nu_arr=f_nu(nu)
    temp=45.0*sigma_Thomson_CGS*UOPT[np.newaxis, :]/(8.0*np.pi**4.0*k_B*Tbb)*f_nu_arr[:, np.newaxis]
    
    return temp # cm^-1

def func_tau_T09(pars_nova, Eg, t):
    Rsh=func_Rsh(pars_nova, t)*1.496e13 # cm
    tau_T09=Rsh[np.newaxis, :]*kappa_gg(pars_nova, Eg*1.6e-12, t)

    return tau_T09 # no unit
    
# Function to calculate the predicted integrated flux
def func_phi_PPI(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t):

    # Parameters for nova shocks
    ter=pars_nova[9]                                             # day
    Rph=pars_nova[13]*1.496e13                                   # cm
    dE=jnp.append(jnp.diff(E), 0.0)[:, jnp.newaxis, jnp.newaxis] # eV

    # Distance from the nova to Earth
    Ds=pars_nova[12]*3.086e18 # cm

    # Calculate the proton distribution        
    JEp=func_JEp_p_ark(pars_nova, E, t)[:, jnp.newaxis, :]  # eV^-1 cm s^-1
    Rsh=func_Rsh(pars_nova, t)*1.496e13                     # cm
    rho=func_rho(pars_nova, t)[jnp.newaxis, jnp.newaxis, :] # g cm^-3

    # Opacity of gamma rays
    TOPT=kB*pars_nova[11]                                                                # eV
    Ebg=jnp.logspace(jnp.log10(TOPT*1.0e-2), jnp.log10(TOPT*1.0e2), 1000)                # eV
    dEbg=jnp.append(jnp.diff(Ebg), 0.0)[jnp.newaxis, :, jnp.newaxis]                     # eV
    UOPT=((pars_nova[12]/1.6e3)**2)*func_LOPT(t)*6.242e11/(4.0*jnp.pi*pow(Rph,2)*3.0e10) # eV cm^-3                           # eV cm^⁻3
    fOPT=func_fEtd(UOPT[jnp.newaxis, :],TOPT,0.0,Ebg[:, jnp.newaxis])[jnp.newaxis, :, :] # eV^-1 cm^-3
    tau_ph=jnp.sum(fOPT*sigma_gg*Rph*dEbg, axis=1)
    tau_ph=jnp.where((t<=ter)[jnp.newaxis, :], 0.0, tau_ph)
    tau_ph1=-jnp.log(func_tau_ph1(pars_nova, tau_ph, t))
    tau_ph2=-jnp.log(func_tau_ph2(pars_nova, tau_ph, t))

    # Calculate the gamma-ray spectrum
    phi_PPI=jnp.sum((4.0*rho/(4.0*np.pi*Ds**2*mpCGS))*(dE*JEp*eps_nucl)*d_sigma_g, axis=0)
    phi_PPI=phi_PPI*(0.5*(jnp.exp(-tau_ph1)+jnp.exp(-tau_ph2)))

    return phi_PPI, tau_ph, tau_ph1, tau_ph2


############################################################################################################################################
# Function to fit gamma-ray spectrum
############################################################################################################################################

# Function to calculate the chi2 of the integrated flux
def func_loss(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t):

    phi_PPI, _=func_phi_PPI(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t)

    # Preparing data in the same time range as model
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
def plot_gamma(pars_nova, phi_PPI, tau_ph1, tau_ph2, Eg, t, t_day):

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

    ax.plot(Eg*1.0e-9,Eg**2*phi_PPI[:,it]*1.6022e-12/(0.5*(np.exp(-tau_ph1[:,it])+np.exp(-tau_ph2[:,it]))),'k--',linewidth=5.0)
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
    ax.legend(loc='upper right', title=r'{\rm Night\, %d}' % int(t_day-0.6), prop={"size":fs}, title_fontsize=fs)
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_wiad/fg_gamma_day%d_DM23_Ds-%.2f.png' % (t_day, (pars_nova[12]*1.0e-3)))

# Plot time profile of gamma-ray integrated flux
def plot_time_gamma(pars_nova, phi_PPI, tau_ph1, tau_ph2, Eg, t):

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    mask_FLAT=(Eg>=0.1e9) & (Eg<=100.0e9)
    mask_HESS=(Eg>=250.0e9) & (Eg<=2500.0e9)

    print("FERMI band: ", Eg[mask_FLAT][0]*1.0e-9, "-", Eg[mask_FLAT][-1]*1.0e-9, "GeV")
    print("HESS band:  ", Eg[mask_HESS][0]*1.0e-9, "-", Eg[mask_HESS][-1]*1.0e-9, "GeV")

    dEg=jnp.append(jnp.diff(Eg), 0.0)
    flux_FLAT_PPI=1.0e-3*1.60218e-12*jnp.sum(jnp.where(mask_FLAT[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)
    flux_HESS_PPI=1.60218e-12*jnp.sum(jnp.where(mask_HESS[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)

    phi_PPI*=1.0/(0.5*(np.exp(-tau_ph1)+np.exp(-tau_ph2)))
    flux_FLAT_PPI_noabs=1.0e-3*1.60218e-12*jnp.sum(jnp.where(mask_FLAT[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)
    flux_HESS_PPI_noabs=1.60218e-12*jnp.sum(jnp.where(mask_HESS[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_PPI, 0.0), axis=0)
    ax.plot(t,flux_HESS_PPI_noabs,'r--',linewidth=5.0)
    ax.plot(t,flux_FLAT_PPI_noabs,'g--',linewidth=5.0)

    # ax.plot(t,flux_HESS_PPI,'r-',linewidth=3.0,label=r'{\rm Model\, HESS\, band}')
    # ax.plot(t,flux_FLAT_PPI,'g-',linewidth=3.0,label=r'{\rm Model\, FERMI\, band}')

    mask1=t>=1.0
    mask2=t<1.0
    ax.plot(t[mask1],flux_HESS_PPI[mask1],'r-',linewidth=5.0)
    ax.plot(t[mask2],flux_HESS_PPI[mask2],'r-',alpha=0.2,linewidth=5.0)
    ax.plot(t,flux_FLAT_PPI,'g-',linewidth=5.0)

    ax.errorbar(t_HESS_raw,flux_HESS_raw,yerr=yerr_HESS_raw,xerr=xerr_HESS_raw,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='red',markeredgecolor='black',markersize=10,label=r'${\rm HESS}$')
    ax.errorbar(t_FERMI_raw,flux_FERMI_raw,yerr=yerr_FERMI_raw,xerr=xerr_FERMI_raw,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='green',markeredgecolor='black',markersize=10,label=r'${\rm FERMI\,(\times 10^{-3})}$')

    ax.set_xlim(5.0e-2,5.0e1)
    ax.set_ylim(1.0e-13,5.0e-11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs,fontweight='bold')
    ax.set_ylabel(r'${\rm Integrated\, Flux} \, ({\rm erg\, cm^{-2}\, s^{-1}})$',fontsize=fs,fontweight='bold')
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper left', prop={"size":fs}, ncols=1)
    ax.grid(linestyle='--')

    # ax.spines["top"].set_linewidth(2)
    # ax.spines["bottom"].set_linewidth(2)
    # ax.spines["left"].set_linewidth(2)
    # ax.spines["right"].set_linewidth(2)
    # ax.tick_params(axis='both', which='major', width=2, length=8)
    # ax.tick_params(axis='both', which='minor', width=1.5, length=5)

    plt.savefig('Results_jax_wiad/fg_time_gamma_DM23_tST-%.2f_Mdot-%.2e_ter-%.1f_BRG-%.1f_Ds-%.2f.png' % (pars_nova[1], pars_nova[3], pars_nova[9], pars_nova[10], (pars_nova[12]*1.0e-3)))
    plt.close()

# Plot the nova shock speed 
def plot_vsh(pars_nova, t):
    
    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    t_vsh, vsh_a, err_vsh_a, vsh_b, err_vsh_b=np.loadtxt('vsh_line.txt',unpack=True,usecols=[0,1,2,3,4])
    ax.errorbar(t_vsh, vsh_a, yerr=err_vsh_a, xerr=t_vsh*0.0, fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='red', markeredgecolor='black', markersize=8, label=r'${\rm H\alpha}$')
    ax.errorbar(t_vsh, vsh_b, yerr=err_vsh_b, xerr=t_vsh*0.0, fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='green', markeredgecolor='black', markersize=8, label=r'${\rm H\beta}$')

    t_vsh_Xray, T_Xray, T_Xray_upper, t_vsh_Xray_lower=np.loadtxt('vsh_Xray.txt',unpack=True,usecols=[0,1,2,4])
    t_vsh_Xray+=1.0
    t_vsh_Xray_lower+=1.0
    vsh_Xray=np.sqrt(16.0*T_Xray*1.0e3/(3.0*1.0e-24*3.0e10**2*6.242e11))*3.0e5 # km/s
    vsh_Xray_upper=np.sqrt(16.0*T_Xray_upper*1.0e3/(3.0*1.0e-24*3.0e10**2*6.242e11))*3.0e5 # km/s
    err_vsh_Xray=vsh_Xray_upper-vsh_Xray
    err_t_vsh_Xray=t_vsh_Xray-t_vsh_Xray_lower
    ax.errorbar(t_vsh_Xray, vsh_Xray, yerr=err_vsh_Xray, xerr=err_t_vsh_Xray, fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='orange', markeredgecolor='black', markersize=8, label=r'${\rm X-ray}$')

    ax.plot(t, func_vsh(pars_nova, t), '--', color='black', linewidth=5)

    ax.set_yticks([np.log10(1000.0), np.log10(5000.0)])
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: r'$%d$' % int(10**x)))
    # ax.set_yticks([1000, 5000])

    ax.set_xlim(1.0,3.0e1)
    ax.set_ylim(500.0,5000.0)

    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$v_{\rm sh} \, ({\rm km/s})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.savefig('Results_jax_wiad/fg_jax_vsh_Xray_Ds-%.2f.png' % (pars_nova[12]*1.0e-3))
    plt.close()

# Plot the optical luminosity 
def plot_LOPT(pars_nova):
    t_data, LOPT_data=np.loadtxt("Data/LOPT.dat",unpack=True,usecols=[0,1])

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    t_plot_OPT=np.linspace(-10,30,1000)
    Lsh=2.0*np.pi*func_rho(pars_nova, t_plot_OPT)*(func_Rsh(pars_nova, t_plot_OPT)*1.496e13)**2*(func_vsh(pars_nova, t_plot_OPT)*1.0e5)**3

    ax.errorbar(t_data-0.25,LOPT_data,yerr=LOPT_data*0.0,xerr=t_data*0.0,fmt='s',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='orange',markeredgecolor='black',markersize=15,label=r'${\rm Cheung\, 2022}$')
    ax.plot(t_plot_OPT,func_LOPT(t_plot_OPT),'k--',linewidth=3.0)
    # ax.plot(t_plot_OPT,func_LOPT_new(t_plot_OPT),'r:',linewidth=3.0,label=r'${\rm Optical}$')
    # ax.plot(t_plot_OPT,Lsh,'r:')

    print(func_LOPT(1.0)-2.5e36)

    ax.legend()
    ax.set_yscale('log')
    ax.set_xlim(-5.0,10.0)
    ax.set_ylim(1.0e36,1.0e39)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$L_{\rm opt} \, ({\rm erg\,s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_wiad/fg_LOPT.png')

# Plot optical flux compared with observations
def plot_flux_OPT(pars_nova, t):
    
    TOPT=kB*pars_nova[11]     # eV
    Ds=pars_nova[12]*3.086e18 # cm

    # Note that here we convert fOPT to dF/dlambda by dF/dlambda = c*E^2*fOPT/lambda
    Ebg=jnp.logspace(jnp.log10(TOPT*1.0e-2), jnp.log10(TOPT*1.0e2), 1000)                # eV
    UOPT=((pars_nova[12]/1.6e3)**2)*func_LOPT(t)*6.242e11/(4.0*jnp.pi*pow(Ds,2)*3.0e10)  # eV cm^⁻3
    fOPT=func_fEtd(UOPT[jnp.newaxis, :],TOPT,0.0,Ebg[:, jnp.newaxis])                    # eV^-1 cm^-3 
    lambda_OPT=hP*3.0e10/Ebg                                                             # cm 
    flux_lambda_OPT=(Ebg[:, jnp.newaxis])**3*fOPT*1.6e-12/hP                                 # erg cm^-2 s^-1 cm^-1

    lower_lambda_ANS1, lambda_ANS1, upper_lambda_ANS1, flux_lambda_ANS1=np.loadtxt('Data/day1-ANS.txt',unpack=True,usecols=[0,1,2,3])
    lower_lambda_ANS3, lambda_ANS3, upper_lambda_ANS3, flux_lambda_ANS3=np.loadtxt('Data/day3-ANS.txt',unpack=True,usecols=[0,1,2,3])
    lower_lambda_ANS4, lambda_ANS4, upper_lambda_ANS4, flux_lambda_ANS4=np.loadtxt('Data/day4-ANS.txt',unpack=True,usecols=[0,1,2,3])

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    it1=np.argmin(np.abs(t-1.68))
    ax.plot(lambda_OPT,flux_lambda_OPT[:, it1],'r-',linewidth=3.0,label=r'${\rm Night\, 1}$')
    ax.errorbar(lambda_ANS1, flux_lambda_ANS1, yerr=0.0*flux_lambda_ANS1, xerr=(upper_lambda_ANS1-lambda_ANS1), fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='red', markeredgecolor='black', markersize=10, label=r'{\rm ANS}')

    it3=np.argmin(np.abs(t-3.68))
    ax.plot(lambda_OPT,flux_lambda_OPT[:, it3],'g-',linewidth=3.0,label=r'${\rm Night\, 3}$')
    ax.errorbar(lambda_ANS3, flux_lambda_ANS3, yerr=0.0*flux_lambda_ANS3, xerr=(upper_lambda_ANS3-lambda_ANS3), fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='green', markeredgecolor='black', markersize=10, label=r'{\rm ANS}')

    it4=np.argmin(np.abs(t-4.68))
    ax.plot(lambda_OPT,flux_lambda_OPT[:, it4],'b-',linewidth=3.0,label=r'${\rm Night\, 4}$')
    ax.errorbar(lambda_ANS4, flux_lambda_ANS4, yerr=0.0*flux_lambda_ANS4, xerr=(upper_lambda_ANS4-lambda_ANS4), fmt='o', capsize=5, ecolor='black', elinewidth=2, markerfacecolor='blue', markeredgecolor='black', markersize=10, label=r'{\rm ANS}')

    ax.set_xlim(2.0e-5,1.0e-4)
    ax.set_ylim(1.0e-3,1.0e-1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda\, {\rm (cm)}$',fontsize=fs)
    ax.set_ylabel(r'${\rm d}F/{\rm d}\lambda\, {\rm (erg\, cm^{-2}\, s^{-1}\, cm^{-1})}$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_wiad/fg_flux_lambda.png')

# Plot the gamma-ray opacity
def plot_tau_ph(pars_nova, tau_ph1, tau_ph2, Eg, t):

    Rph=pars_nova[13]*1.496e13 # cm

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    iplot1=np.argmin(np.abs(Eg-5.0e11))
    iplot2=np.argmin(np.abs(Eg-1.0e12))

    print(Eg[iplot1])

    tau_T09=func_tau_T09(pars_nova, Eg, t)

    # Ebg=jnp.linspace(kB*pars_nova[11], 2.0*kB*pars_nova[11], 1)

    nu=me_g**2*(3.0e10)**4/(Eg*1.6e-12*k_B*pars_nova[11])
    UOPT=((pars_nova[12]/1.6e3)**2)*func_LOPT(t)/(4.0*np.pi*pow(Rph,2)*3.0e10) # erg cm^⁻3
    f_nu_arr=f_nu(nu)
    tau_T09_test=Rph*(45.0*sigma_Thomson_CGS*UOPT[np.newaxis, :]/(8.0*np.pi**4.0*k_B*pars_nova[11]))*f_nu_arr[:, np.newaxis]
    # tau_T09_test=Rph*(UOPT[np.newaxis, :]/(k_B*pars_nova[11]))*func_sigma_gg(Eg, Ebg)
    tau2_T09_test=-jnp.log(func_tau_ph2(pars_nova, tau_T09_test, t))
    
    print(iplot1, Eg[iplot1])
    print(iplot2, '%.2e' % Eg[iplot2])

    ax.plot(t,tau_ph1[40,:],'r:',linewidth=3.0,label=r'${\rm \tau_1(E_\gamma=1\,TeV,t)}$')
    ax.plot(t,tau_ph2[40,:],'g--',linewidth=3.0,label=r'${\rm \tau_2(E_\gamma=1\,TeV,t)}$')
    # ax.plot(t,tau_T09[40,:],'r:',linewidth=3.0,label=r'${\rm Tatischeff\, 2009}$')
    # ax.plot(t,tau2_T09_test[40,:],'k-.',linewidth=3.0,label=r'${\rm Tatischeff\, 2009\, \tau_2}$')

    ax.set_xlim(0.0,10.0)
    # ax.set_ylim(1.0e-2,2.0e1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$\tau$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_wiad/fg_tau_ph.png')