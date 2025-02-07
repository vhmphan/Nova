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
import diffrax
import scipy as sp
from matplotlib.ticker import FuncFormatter

fs=22

# Define a custom tick formatter to display 10^x
def log_tick_formatter(x, pos):
    return r'$10^{%d}$' % int(x)


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

    vsh=func_vsh_HESS(t)

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

    Rsh=func_Rsh_HESS(t)

    return Rsh*86400.0*6.68e-9 # au

# # Density profile of the red giant wind
# def func_rho(pars_nova, r):
# # Mdot (Msol/yr), vwind (km/s), and r (au)    

#     Mdot=pars_nova[3]*1.989e33/(365.0*86400.0) # g/s 
#     vwind=pars_nova[4]*1.0e5                   # cm/s
#     Rmin=pars_nova[5]*1.496e13                 # cm
#     r=jnp.array(r)                             # au

#     rho=Mdot/(4.0*jnp.pi*vwind*pow(Rmin+r*1.496e13,2)) 

#     return rho # g/cm^3

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

    dEdt_acc=(qeCGS*B2*jnp.power(vsh, 2))*6.242e+11/(30.0*jnp.pi*3.0e10)

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

    tST=pars_nova[1]           # day
    alpha=pars_nova[2]         # no unit
    Rmin=pars_nova[5]*1.496e13 # cm
    ter=pars_nova[9]           # day

    vsh=func_vsh(pars_nova,t)*1.0e5    # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    p=jnp.sqrt((E+mp)**2-mp**2)        # eV
    t=jnp.array(t)                     # day

    mask1=(t>=ter) & (t<tST)
    mask2=(t>=tST)

    dEdt_adi=jnp.where(mask1, -0.2*(p**2/(E+mp))*(Rsh*vsh/(Rmin**2+Rsh**2)), 0.0)
    dEdt_adi=jnp.where(mask2, -0.2*(p**2/(E+mp))*(Rsh*vsh/(Rmin**2+Rsh**2))-0.2*(p**2/(E+mp))*(2.0*alpha/(t*86400.0)), dEdt_adi)

    return dEdt_adi*0.0 # eV s^-1

# Derivative of adiabatic energy loss rate with respect to kinetric energy
def func_dE_adi_dE(pars_nova, E, t):
# E (eV) and t(day)

    tST=pars_nova[1]           # day
    alpha=pars_nova[2]         # no unit
    Rmin=pars_nova[5]*1.496e13 # cm
    ter=pars_nova[9]           # day

    vsh=func_vsh(pars_nova,t)*1.0e5    # cm/s
    Rsh=func_Rsh(pars_nova,t)*1.496e13 # cm
    p=jnp.sqrt((E+mp)**2-mp**2)        # eV
    t=jnp.array(t)                     # day

    mask1=(t>=ter) & (t<tST)
    mask2=(t>=tST)

    dEdt_adi_dE=jnp.where(mask1, -0.2*(2.0-(p/(E+mp))**2)*(Rsh*vsh/(Rmin**2+Rsh**2)), 0.0)
    dEdt_adi_dE=jnp.where(mask2, -0.2*(2.0-(p/(E+mp))**2)*(Rsh*vsh/(Rmin**2+Rsh**2))-0.2*(2.0-(p/(E+mp))**2)*(2.0*alpha/(t*86400.0)), dEdt_adi_dE)

    return dEdt_adi_dE*0.0 # s^-1

# Maximum energy of particle accelerated from the shock calculated with diffrax
def func_Emax(pars_nova, t):
# t(day)

    E0=1.0e2 # eV

    def dE_dt(tp, Emax, args):
        pars_nova=args
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

    # solve_vectorized=jax.vmap(solve_single_ode)
    # E0=solve_vectorized(E)
    
    return E0 # eV

# Cummulated particle flux solved with diffrax
@jit
def func_JEp_p_dif(pars_nova, E, t):
# E (eV) and t (day)

    Emax_arr=func_Emax(pars_nova, t) # eV

    # Define the function f(t, E, pars) for the right-hand side of the differential equation
    def func_fEp_p(tp, Ep, pars_nova):

        xip=pars_nova[6]     # no unit
        delta=pars_nova[7]   # no unit
        epsilon=pars_nova[8] # no unit

        Emax=jnp.interp(tp, t, Emax_arr, left=0.0, right=0.0)

        # Get the nomalization for the accelerated spectrum
        xmin=jnp.sqrt(pow(1.0e8+mp,2)-mp*mp)/mp 
        xmax=jnp.sqrt(pow(1.0e14+mp,2)-mp*mp)/mp
        x=jnp.logspace(jnp.log10(xmin), jnp.log10(xmax), 500)

        dx=(x[1:-1]-x[0:-2])[:,jnp.newaxis]
        x=x[0:-2][:,jnp.newaxis]
        Ialpha_p=jnp.sum(pow(x, 4.0-delta)*jnp.exp(-pow(x*mp/Emax, epsilon))*dx/jnp.sqrt(1.0+x*x))

        # Get the momentum and speed 
        p=jnp.sqrt(pow(Ep+mp,2)-mp*mp)
        vp=(p/(Ep+mp))

        # Get all the shock dynamics related quantities
        vsh=func_vsh(pars_nova, tp)*1.0e5     # cm s^-1
        Rsh=func_Rsh(pars_nova, tp)*1.496e13  # cm
        rho=func_rho(pars_nova, Rsh/1.496e13) # g cm^-3

        # Note that Ialpha_p is zero for t<=ter so there will be some nan and we replace it by zero.
        fEp=3.0*jnp.pi*xip*rho*pow(Rsh,2)*pow(vsh,3)*6.242e11*pow(p/mp, 2.0-delta)*jnp.exp(-pow(p/Emax, epsilon))/(mp*mp*vp*Ialpha_p)
        fEp=jnp.where(jnp.isnan(fEp), 0.0, fEp)

        return fEp # eV^-1 s^-1

    # Define the derivative function for dN/dt
    def dN_dt(tp, N, args):
        Ep, pars_nova=args
        E0_arr=func_E0(pars_nova, Ep, t)  # eV
        E0=jnp.interp(tp, t, E0_arr)

        return (func_fEp_p(tp, E0, pars_nova)-N*func_dE_adi_dE(pars_nova, E0, tp))*86400.0  # Compute the derivative based on f(t, E, pars)

    # Solver function to solve dN/dt for a given energy level E and parameters pars
    def solve_for_energy(E_single):
        # Set up the ODE term and solver for each energy level
        term = diffrax.ODETerm(dN_dt)
        solver = diffrax.Dopri5()
        
        # Initial condition and save points
        y0=0.0
        saveat=diffrax.SaveAt(ts=t)
        
        # Solve the ODE from t[0] to t[-1] with the specific energy E_single and pars
        solution=diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=t[0],
            t1=t[-1],
            dt0=t[1]-t[0],
            y0=y0,
            saveat=saveat,
            args=(E_single, pars_nova)
        )

        return solution.ys

    NEp=jax.vmap(solve_for_energy)(E)

    # Get the momentum and speed 
    E0=func_E0(pars_nova, E, t).T   # eV
    p0=jnp.sqrt(pow(E0+mp,2)-mp*mp) # eV
    vp0=p0/(E0+mp)                  # no unit
    JEp=NEp*vp0*3.0e10              # eV^-1 cm s^-1

    def interp_JEp(t_index):
        return jnp.interp(E, E0[:,t_index], JEp[:,t_index], left=0.0, right=0.0) 

    JEp_interp=jax.vmap(interp_JEp)(jnp.arange(len(t)))

    return JEp_interp.T # eV^-1 cm s^-1

# Cummulated particle flux solved with Runge-Kutta 4
# @jit
def func_JEp_p_rk4(pars_nova, E, t):

    # Define the function f(t, E, pars) for the right-hand side of the differential equation
    def func_fEp_p(t, Ep, pars_nova):

        xip=pars_nova[6]     # no unit
        delta=pars_nova[7]   # no unit
        epsilon=pars_nova[8] # no unit

        Emax=func_Emax(pars_nova, t) # eV

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
    E0=E                            # eV
    p0=jnp.sqrt(pow(E0+mp,2)-mp*mp) # eV
    vp0=p0/(E0+mp)                  # no unit

    return NEp*vp0[:, jnp.newaxis]*3.0e10 # eV^-1 cm s^-1


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
    JEp=func_JEp_p_rk4(pars_nova, E, t)[:, jnp.newaxis, :]      # eV^-1 cm s^-1
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
        plt.savefig('Results_jax_HESS/fg_gamma_day%d_HESS.png' % (t_day))
    else:
        plt.savefig('Results_jax_HESS/fg_gamma_day%d_DM23.png' % (t_day))

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

    # data = np.load('test.npz')

    # phi_test = data['array1']
    # phi_test_unabs = data['array2']

    # flux_FLAT_PPI_test=1.0e-3*1.60218e-12*jnp.sum(jnp.where(mask_FLAT[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_test, 0.0), axis=0)
    # flux_HESS_PPI_test=1.60218e-12*jnp.sum(jnp.where(mask_HESS[:, jnp.newaxis], dEg[:, jnp.newaxis]*Eg[:, jnp.newaxis]*phi_test, 0.0), axis=0)

    # ax.plot(t,flux_HESS_PPI_test,':', color='orange',linewidth=3.0,label=r'{\rm Model\, HESS\, band}')
    # ax.plot(t,flux_FLAT_PPI_test,':', color='yellow',linewidth=3.0,label=r'{\rm Model\, FERMI\, band}')

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
        plt.savefig('Results_jax_HESS/fg_time_gamma_HESS_tST-%.2f_Mdot-%.2e_ter-%.1f_BRG-%.1f.png' % (pars_nova[1], pars_nova[3], pars_nova[9], pars_nova[10]))
    else:
        plt.savefig('Results_jax_HESS/fg_time_gamma_DM23_tST-%.2f_Mdot-%.2e_ter-%.1f_BRG-%.1f.png' % (pars_nova[1], pars_nova[3], pars_nova[9], pars_nova[10]))
    plt.close()
 

if __name__ == "__main__":

    # Record the starting time
    start_time=time.time()

    # # Initialized parameters for RS Ophiuchi 2021  
    # vsh0=4500.0  # km/s    -> Initial shock speed
    # tST=1.8      # day     -> Time where shock transition to Sedov-Taylor phase
    # alpha=0.66   # no unit -> Index for time profile of shock speed
    # Mdot=2.0e-7  # Msol/yr -> Mass loss rate of red giant
    # vwind=20.0   # km/s    -> Wind speed of red giant
    # Rmin=1.48    # au      -> Distance between red giant and white dwarf
    # xip=0.5      # no unit -> Fraction of shock ram pressure converted into cosmic-ray energy
    # delta=4.2    # no unit -> Injection spectrum index
    # epsilon=0.5  # no unit -> Index of the exp cut-off for injection spectrum
    # BRG=1.0      # G       -> Magnetic field srength at the pole of red giant
    # TOPT=1.0e4   # K       -> Temperature of the optical photons 
    # ter=0.0      # day     -> Shock formation time
    # Ds=1.4e3     # pc      -> Distance to Earth of the nova
    # model_name=0 # no unit -> Model name (1=HESS and 0=DM23)
    # pars_nova=[vsh0, tST, alpha, Mdot, vwind, Rmin, xip, delta, epsilon, ter, BRG, TOPT, Ds, model_name]

    # Initialized parameters for RS Ophiuchi 2021  
    vsh0=4500.0  # km/s    -> Initial shock speed
    tST=1.8      # day     -> Time where shock transition to Sedov-Taylor phase
    alpha=0.66   # no unit -> Index for time profile of shock speed
    Mdot=2.0e-7  # Msol/yr -> Mass loss rate of red giant
    vwind=20.0   # km/s    -> Wind speed of red giant
    Rmin=1.48    # au      -> Distance between red giant and white dwarf
    xip=0.5      # no unit -> Fraction of shock ram pressure converted into cosmic-ray energy
    delta=4.2    # no unit -> Injection spectrum index
    epsilon=0.5  # no unit -> Index of the exp cut-off for injection spectrum
    BRG=1.0      # G       -> Magnetic field srength at the pole of red giant
    TOPT=1.0e4   # K       -> Temperature of the optical photons 
    ter=0.0      # day     -> Shock formation time
    Ds=1.4e3     # pc      -> Distance to Earth of the nova
    model_name=1 # no unit -> Model name (1=HESS and 0=DM23)
    pars_nova=[vsh0, tST, alpha, Mdot, vwind, Rmin, xip, delta, epsilon, ter, BRG, TOPT, Ds, model_name]

    # Define the time and energy ranges -> note that it is required that t[0]<=ter 
    t=jnp.linspace(0.0,6.0,601) # day
    E=jnp.logspace(9.3,14.0,61)    # eV
    Eg=jnp.logspace(8.0,14.0,601)  # eV

    # Gamma-ray production cross-section
    eps_nucl=jnp.array(gt.func_enhancement(np.array(E)))[:, jnp.newaxis, jnp.newaxis] # no unit
    d_sigma_g=jnp.array(gt.func_d_sigma_g(np.array(E), np.array(Eg)))[:, :, jnp.newaxis]        # cm^2/eV

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

    NEp_dif=func_JEp_p_dif(pars_nova, E, t)
    NEp_rk4=func_JEp_p_rk4(pars_nova, E, t)

    print(NEp_dif[10])
    print(NEp_rk4[10])

    phi_PPI, tau_gg=func_phi_PPI(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t)

    plot_gamma(pars_nova, phi_PPI, tau_gg, Eg, t, 1.6)
    plot_gamma(pars_nova, phi_PPI, tau_gg, Eg, t, 3.6)
    plot_gamma(pars_nova, phi_PPI, tau_gg, Eg, t, 5.6)
    plot_time_gamma(pars_nova, phi_PPI, tau_gg, Eg, t)

    EnJEp_dif=E[:,jnp.newaxis]**3*NEp_dif
    EnJEp_rk4=E[:,jnp.newaxis]**3*NEp_rk4

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    ax.plot(E, EnJEp_dif[:, t==1.0], '-', color='red', linewidth=3, label='Day 1')
    ax.plot(E, EnJEp_rk4[:, t==1.0], '--', color='black', linewidth=3)

    ax.plot(E, EnJEp_dif[:, t==5.0], '-', color='orange', linewidth=3, label='Day 5')
    ax.plot(E, EnJEp_rk4[:, t==5.0], '--', color='black', linewidth=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.0e8, 1.0e14)
    ax.set_ylim(1.0e70, 1.0e76)
    ax.set_xlabel(r'$E\, {\rm (eV)}$',fontsize=fs)
    ax.set_ylabel(r'$J(E) \, ({\rm eV^{2}\, cm\, s^{-1} })$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_HESS/fg_jax_JEp.png')
    plt.close()

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    E0=func_E0(pars_nova, E, t).T
    for i in range(len(E)):
        if((i%10==0) & (E[i]>1.0e11)):
            ax.plot(t, E0[i,:], '-', linewidth=3, label='%.2e' % E[i])

    # ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(1.0e8, 1.0e14)
    # ax.set_ylim(1.0e70, 1.0e76)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$E_0 \, ({\rm eV})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_HESS/fg_jax_E0.png')
    plt.close()

    vsh=func_vsh(pars_nova, t)*1.0e5      # cm/s
    Rsh=func_Rsh(pars_nova, t)*1.496e13   # cm
    rho=func_rho(pars_nova, Rsh/1.496e13) # g/cm^3
    Lsh=2.0*jnp.pi*Rsh**2*rho*vsh**3      # erg/s

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    # Read the image for data    
    img = mpimg.imread("Data/data_Lsh.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=5.0
    ymin=0.0
    ymax=3.5e38
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.plot(t, Lsh, 'r:', linewidth=3, label='Minh')

    ax.set_aspect(5.0/3.5e38)
    ax.legend()
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$L_{\rm sh} \, ({\rm erg\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower right', prop={"size":fs})
    ax.grid(linestyle='--')
    ax.set_xlim(0,5)

    plt.savefig('Results_jax_HESS/fg_Lsh.png')

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,vsh/1.0e5,'r--',linewidth=3.0)

    # Read the image for data    
    img = mpimg.imread("Data/data_vsh.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=6.0
    ymin=2500.0
    ymax=5000.0
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_aspect((xmax-xmin)/(ymax-ymin))

    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$v_{\rm sh} \, ({\rm km/s})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_HESS/fg_vsh.png')

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,Rsh/1.496e13,'r--',linewidth=3.0,label='Minh')

    # Read the image for data    
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
    ax.legend(loc='lower right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_HESS/fg_Rsh.png')

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,jnp.log10(rho/1.67e-24),'r:',linewidth=3.0, label=r'{\rm Minh}')

    # Read the image for data
    img = mpimg.imread("Data/data_rho.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=0.0
    xmax=5.0
    ymin=5.0
    ymax=9.0
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_yticks([5, 6, 7, 8, 9])

    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$L_{\rm opt} \, ({\rm erg\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_HESS/fg_rho.png')

    print('Note that adiabatic energy loss has been deactivated for checking')
    print('Mdot/vw = %.2e' % (Mdot*2.0e33/(vwind*365.0*86400.0*1.0e5)))

    # Record the ending time
    end_time=time.time()

    # Calculate the elapsed time
    elapsed_time=end_time-start_time

    print("Elapsed time:", elapsed_time, "seconds")
