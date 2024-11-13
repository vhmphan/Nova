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

    return dEdt_adi # eV s^-1


# Maximum energy of particle accelerated from the shock calculated with diffrax
def func_Emax(pars_nova, t):
# t(day)

    E0=1.0e2 # eV

    def dE_dt(tp, Emax, args):
        pars_nova = args
        return (func_dE_acc(pars_nova, Emax, tp)+func_dE_adi(pars_nova, Emax, tp))*86400.0 # eV day^-1

    term=diffrax.ODETerm(dE_dt)
    solver=diffrax.Dopri5()
    saveat=diffrax.SaveAt(ts=t)        
    solution=diffrax.diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[1]-t[0], y0=E0, saveat=saveat, args=pars_nova)
    Emax=solution.ys

    return Emax # eV

@jit
def func_JEp_p_diff(pars_nova, E, t):

    Emax_arr=func_Emax(pars_nova,t)

    # Define the function f(t, E, pars) for the right-hand side of the differential equation
    def func_fEp_p(tp, Ep, pars_nova):

        xip=pars_nova[6]     # no unit
        delta=pars_nova[7]   # no unit
        epsilon=pars_nova[8] # no unit

        Emax=jnp.interp(tp,t,Emax_arr)

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
        Ep, pars_nova = args  # Unpack the arguments: current energy E and parameters pars
        return func_fEp_p(tp, Ep, pars_nova)*86400.0  # Compute the derivative based on f(t, E, pars)

    # Solver function to solve dN/dt for a given energy level E and parameters pars
    def solve_for_energy(E_single):
        # Set up the ODE term and solver for each energy level
        term = diffrax.ODETerm(dN_dt)
        solver = diffrax.Dopri5()
        
        # Initial condition and save points
        y0 = 0.0  # Initial condition N(t=0, E) = 0
        saveat = diffrax.SaveAt(ts=t)
        
        # Solve the ODE from t0 to t[-1] with the specific energy E_single and pars
        solution = diffrax.diffeqsolve(
            terms=term,           # Use terms=term instead of term=term
            solver=solver,
            t0=t[0],
            t1=t[-1],
            dt0=t[1]-t[0],
            y0=y0,
            saveat=saveat,
            args=(E_single, pars_nova)  # Pass E_single and pars as a tuple to dN_dt
        )

        return solution.ys  # This will be N(t, E_single) for this energy level

    NEp=jax.vmap(solve_for_energy)(E)

    # Get the momentum and speed 
    E0=E # func_E0(pars_nova, E, t)   # eV
    p0=jnp.sqrt(pow(E0+mp,2)-mp*mp) # eV
    vp0=p0/(E0+mp)

    return NEp*vp0[:, jnp.newaxis]*3.0e10 # eV^-1 cm s^-1

# @jit
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
        # print(func_fEp_p(pars_nova, E_single, t).shape)
        
        return jnp.cumsum((k1+2.0*k2+2.0*k3+k4)/6.0, axis=0)  # This will be N(t, E_single) for this energy level

    NEp=jax.vmap(solve_for_energy)(E)

    # Get the momentum and speed 
    E0=E # func_E0(pars_nova, E, t)   # eV
    p0=jnp.sqrt(pow(E0+mp,2)-mp*mp) # eV
    vp0=p0/(E0+mp)

    return NEp*vp0[:, jnp.newaxis]*3.0e10 # eV^-1 cm s^-1

# Injection spectrum at the shock
def func_fEp_p_noad(pars_nova, E, t):
# E (eV) and t(day)

    xip=pars_nova[6]     # no unit
    delta=pars_nova[7]   # no unit
    epsilon=pars_nova[8] # no unit

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

# Cumulative spectrum of accelerated protons without adiabatic energy loss
def func_JEp_p(pars_nova, E, t):
# E (eV) and t(day)

    # Get the momentum and speed 
    p=jnp.sqrt(pow(E+mp,2)-mp*mp)
    vp=p/(E+mp)

    # Compute NEp by solving the differential equation 
    fEp=func_fEp_p_noad(pars_nova, E, t) # eV^-1 s^-1
    dt=(t[1]-t[0])*86400.0          # s
    NEp=jnp.cumsum(fEp, axis=1)*dt  # eV^-1

    return NEp*vp[:, jnp.newaxis]*3.0e10 # eV^-1 cm s^-1


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
ter=0.0     # day     -> Shock formation time
Ds=1.4e3     # pc      -> Distance to Earth of the nova
model_name=0 # no unit -> Model name (1=HESS and 0=DM23)
pars_nova=[vsh0, tST, alpha, Mdot, vwind, Rmin, xip, delta, epsilon, ter, BRG, TOPT, Ds, model_name]

# Define the time and energy ranges -> note that it is required that t[0]<=ter 
t=jnp.linspace(-1.0,10.0,1101) # day
E=jnp.logspace(8,14,61)       # eV
Eg=jnp.logspace(8,14,601)      # eV

NEp=func_JEp_p(pars_nova, E, t)
NEp_diff=func_JEp_p_diff(pars_nova, E, t)
NEp_rk4=func_JEp_p_rk4(pars_nova, E, t)

print(NEp.shape)
print(NEp[10])
print(NEp_diff[10])
print(NEp_rk4[10])