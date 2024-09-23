# import gato.pack_nova as nv
# import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt

# pars_nova=[4500.0, 2.0, 0.66, 6.0e-7, 20.0, 1.48, 0.1, 4.4, 1.0, 0.0, 1.0, 1.0e4, 10, 2.0e-9, 1.4e3, 'DM23', 50]

# t=np.linspace(-1.0,10.0,241)*86400.0
# E0=np.logspace(8, 14, 6)

# sol=sp.integrate.solve_ivp(lambda tp,E:(nv.func_dE_adi(pars_nova,E,tp/86400.0)),[t[0],t[-1]],E0,t_eval=t,method='RK45')
# E=sol.y
# print(E.shape)

# # Plot the results
# plt.figure(figsize=(10, 6))
# for i in range(len(E0)):
#     plt.plot(t/86400.0, E[i,:], label=f'E0=%.1e' % E0[i])

# plt.xlabel('Time t')
# plt.ylabel('E(t)')
# plt.yscale('log')
# plt.title('Solutions of dE/dt = b(E, t) for different initial conditions')
# plt.legend(loc='upper right', fontsize='small')
# plt.show()

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
    # B2_Bell=jnp.sqrt(11.0*jnp.pi*rho*np.power(vsh*xip, 2))                # -> Model with amplified B-field
    B2=B2_bkgr # +B2_Bell*func_Heaviside(arr_t-tST)                         # -> Model with instability switched on

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

def step(pars_nova, E, t0, t1):

    dt = t1 - t0
    k1 = dt * 86400.0 * func_dE_adi(pars_nova, E, t0)
    k2 = dt * 86400.0 * func_dE_adi(pars_nova, E + 0.5 * k1, t0 + 0.5 * dt)
    k3 = dt * 86400.0 * func_dE_adi(pars_nova, E + 0.5 * k2, t0 + 0.5 * dt)
    k4 = dt * 86400.0 * func_dE_adi(pars_nova, E + k3, t0 + dt)

    # Update the solution with the Dormand-Prince formula
    E_next = E + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return E_next

# Maximum energy of particle accelerated from the shock calculated with diffrax
def func_Emax_dif(pars_nova, t):
    # t(day)

    ter=pars_nova[9] # day
    E0=1.0e2         # eV

    def dE_dt(tp, Emax, args):
        pars_nova = args
        return (func_dE_acc(pars_nova, Emax, tp)+func_dE_adi(pars_nova, Emax, tp))*86400.0 # eV/day

    term=diffrax.ODETerm(dE_dt)
    solver=diffrax.Dopri5()
    saveat=diffrax.SaveAt(ts=t)        
    solution=diffrax.diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[1]-t[0], y0=E0, saveat=saveat, args=pars_nova)
    Emax=solution.ys[jnp.newaxis, :]  

    return Emax # eV

# Maximum energy of particle accelerated from the shock calculated with jax
def func_Emax(pars_nova, t):
# t(day)

    ter=pars_nova[9] # day
    E0=1.0e2         # eV

    def dE_dt(Emax, tp):
        return func_dE_acc(pars_nova,Emax,tp/86400.0)+func_dE_adi(pars_nova,Emax,tp/86400.0)

    # Note that we initialize particle energy to be around Emax(t=0 day)=100 eV (this value should be computed self-consistently from temperature of 
    # plasma around the shock but numrical value of Emax for t>~ 1 day does not change for Emax(t=0 day)=100 or 1000 eV).
    t_as=jnp.linspace(ter, t[-1], 1000)
    Emax_as=odeint(dE_dt, E0, t_as*86400.0)

    Emax=jnp.interp(t, t_as, Emax_as, left=E0, right=0.0)
    Emax=Emax[jnp.newaxis,:]
    
    return Emax # eV

# Evolution of energy due to adiabatic energy loss
def func_E0_ode(pars_nova, E, t):
# E (eV) and t(day)

    def dE_dt_adi(Ep, tp):
        return func_dE_adi(pars_nova, Ep, tp)*86400.0 # eV/day

    def solve_single_ode(E0_single):
        return odeint(dE_dt_adi, E0_single, t)

    solve_vectorized = jax.vmap(solve_single_ode)

    E0 = solve_vectorized(E)
    print(E0.shape)

    return E0 # eV

@jit
def func_E0_dif(pars_nova, E, t):
# E (eV) and t (day)

    def dE_dt_adi(tp, Ep, args):
        pars_nova=args
        return func_dE_adi(pars_nova, Ep, tp)*86400.0 # eV/day

    def solve_single_ode(E0_single):
        term=diffrax.ODETerm(dE_dt_adi)
        solver=diffrax.Dopri5()
        saveat=diffrax.SaveAt(ts=t)
        solution=diffrax.diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[1]-t[0], y0=E0_single, saveat=saveat, args=pars_nova)
        return solution.ys

    solve_vectorized=jax.vmap(solve_single_ode)
    E0=solve_vectorized(E)
    
    return E0 # eV

# Evolution of energy due to adiabatic energy loss
def func_E0_rk4(pars_nova, E, t):
# E (eV) and t(day)

    def dE_dt_adi(Ep, tp):
        return func_dE_adi(pars_nova, Ep, tp)*86400.0 # eV/day

    def solve_single_ode(E0_single):
        return odeint(dE_dt_adi, E0_single, t)

    solve_vectorized = jax.vmap(solve_single_ode)

    E0=solve_vectorized(E)

    return E0 # eV

@jit
def func_E0(pars_nova, E, t):

    E0=jnp.zeros((len(E), len(t)))
    E0=E0.at[:, 0].add(E)

    dt=t[1]-t[0]
    for i in range(len(t)-1):
        k1=dt*86400.0*func_dE_adi(pars_nova, E0[:, i], t[i])
        k2=dt*86400.0*func_dE_adi(pars_nova, E0[:, i]+0.5*k1, t[i]+0.5*dt)
        k3=dt*86400.0*func_dE_adi(pars_nova, E0[:, i]+0.5*k2, t[i]+0.5*dt)
        k4=dt*86400.0*func_dE_adi(pars_nova, E0[:, i]+k3, t[i]+dt)

        E0=E0.at[:, i+1].add(E0[:, i]+(k1+2.0*k2+2.0*k3+k4)/6.0)

    return E0

if __name__ == "__main__":

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
    ter=-0.5     # day     -> Shock formation time
    Ds=1.4e3     # pc      -> Distance to Earth of the nova
    model_name=0 # no unit -> Model name (1=HESS and 0=DM23)
    pars_nova=jnp.array([vsh0, tST, alpha, Mdot, vwind, Rmin, xip, delta, epsilon, ter, BRG, TOPT, Ds, model_name])

    # Define the time and energy ranges -> note that it is required that t[0]<=ter 
    t=jnp.linspace(-1.0, 6.0, 701) # day
    E=jnp.logspace(8, 14, 601)       # eV

    # Record the starting time
    start_time=time.time()

    print('NE = %d' % len(E))
    print('Nt = %d' % len(t))

    for i in range(10):
        E_dif=func_E0_dif(pars_nova, E, t)
        # E_ode=func_E0_ode(pars_nova, E, t)
        # E_rk4=func_E0(pars_nova, E, t)
    
    # Record the ending time
    end_time=time.time()

    # Calculate the elapsed time
    elapsed_time=end_time-start_time

    print("Elapsed time:", elapsed_time, "seconds")

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    for i in range(0, len(E), 10):
        # ax.plot(t, E_rk4[i,:], '-')
        # ax.plot(t, E_ode[i,:], '--')
        ax.plot(t, E_dif[i,:], ':')

    ax.set_yscale('log')
    # ax.set_ylim(1.0e7, 5.0e8)
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$E_{\rm 0} \, ({\rm eV})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax/fg_jax_E0_test.png')
    plt.close()

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(t,func_Emax(pars_nova, t)[0],'k-',linewidth=3.0, label='Old')
    ax.plot(t,func_Emax_dif(pars_nova, t)[0],':',linewidth=8.0, color='darkgreen')
    
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
        plt.savefig('Results_jax/new_fg_jax_Emax_DM23_dif.png')
    plt.close()