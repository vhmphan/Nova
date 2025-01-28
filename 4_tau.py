import numpy as np
from scipy.integrate import quad
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

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

# Auxiliary function for energy density of optical photons
def func_gOPT(x):
    return (1.0/x)-((1.0/x**2)-1.0)*jnp.log(jnp.sqrt(jnp.abs((x-1.0)/(x+1.0))))

# Auxiliary function for the opacity tau_1
def func_inner_int1(pars_nova, eps, t):

    Rsh=func_Rsh(pars_nova, t)

    def func_inner_int_single(eps_single):
        s=jnp.linspace(0.0*Rsh, 100.0*Rsh, 1001, axis=0)                                                              # au
        ds=jnp.diff(s, append=s[-1:, :], axis=0)  
        r=jnp.sqrt(s**2-2.0*s*Rsh[jnp.newaxis, :]*jnp.sqrt(1.0-eps_single**2)+Rsh[jnp.newaxis, :]**2) # au

        return jnp.sum((1.5*r/Rph)*func_gOPT(r/Rph)*ds/Rph, axis=0)
        
    return jax.vmap(func_inner_int_single)(eps) # no unit

# Auxiliary function for the opacity tau_2
def func_inner_int2(pars_nova, eps, t):

    Rsh=func_Rsh(pars_nova, t)

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

    tau_ph1_full=jax.vmap(interp_tau_ph1)(jnp.arange(len(Eg)))

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

    tau_ph2_full=jax.vmap(interp_tau_ph2)(jnp.arange(len(Eg)))

    return tau_ph2_full


if __name__ == "__main__":

    # Initialized parameters for RS Ophiuchi 2021  
    tST=2.2                          # day     -> Time where shock transition to Sedov-Taylor phase
    alpha=0.43                       # no unit -> Index for time profile of shock speed
    vsh0=3500.0                      # km/s    -> Initial shock speed
    Mdot=5.3e-7                      # Msol/yr -> Mass loss rate of red giant
    vwind=20.0                       # km/s    -> Wind speed of red giant
    Rmin=1.48                        # au      -> Distance between red giant and white dwarf
    xip=0.2                          # no unit -> Fraction of shock ram pressure converted into cosmic-ray energy
    delta=4.2                        # no unit -> Injection spectrum index
    epsilon=1.0                      # no unit -> Index of the exp cut-off for injection spectrum
    BRG=6.8                          # G       -> Magnetic field srength at the pole of red giant
    TOPT=1.0e4                       # K       -> Temperature of the optical photons 
    ter=0.0                          # day     -> Shock formation time
    Ds=2.45e3 #1.4e3                 # pc      -> Distance to Earth of the nova
    Rph=200.0*0.00465                # au      -> Photospheric radius of the nova
    pars_nova=jnp.array([vsh0, tST, alpha, Mdot, vwind, Rmin, xip, delta, epsilon, ter, BRG, TOPT, Ds, Rph])

    # Load the .npz file
    data=np.load('tau_ph.npz')
    Eg=data['Eg']
    t=data['t']
    tau_ph=data['tau_ph']

    Rsh=func_Rsh(pars_nova, t)
    s = jnp.linspace(0.0 * Rsh, 100.0 * Rsh, 3, axis=0)  
    ds = jnp.diff(s, append=s[-1:, :], axis=0)  
    print(ds)
    # print(jnp.tile(func_Rsh(pars_nova, t), (len(Eg), 1)).shape)

    # tau_ph_sparse=tau_ph[:,::10]
    # # print(tau_ph_sparse)
    # eps=jnp.linspace(0.001, 0.99999, 1000)
    # deps=eps[1]-eps[0]
    # inner_int=jnp.exp(-tau_ph_sparse[jnp.newaxis,:,:]*func_inner_int(pars_nova, eps, t[::10])[:,jnp.newaxis,:])
    # # print('hj', inner_int)
    # # print(func_inner_int(pars_nova,eps,t[::10]))
    # print(jnp.sum(inner_int*eps[:, jnp.newaxis, jnp.newaxis]*deps/jnp.sqrt(1.0-eps[:, jnp.newaxis, jnp.newaxis]**2), axis=0))

    tau_ph1=func_tau_ph1(pars_nova, tau_ph, t)
    tau_ph2=func_tau_ph2(pars_nova, tau_ph, t)

    it=110
    print(t[it])
    print(tau_ph[30,it], func_Rsh(pars_nova,t[it]), Rph, tau_ph1[30,it], tau_ph2[30,it])

    # print(tau_ph[30,it], tau_ph1[30,it], tau_ph2[30,it])
    # # print(func_tau_test1(func_Rsh(pars_nova, t)[it], tau_ph[30,it], 0.001, 0.999999))
    # # print(func_tau_test2(func_Rsh(pars_nova, t)[it], tau_ph[30,it], 0.001, 0.999999))

    # fs=22

    # fig=plt.figure(figsize=(10, 8))
    # ax=plt.subplot(111)

    # iplot1=np.where(np.abs(Eg-100.0e9)==np.min(np.abs(Eg-100.0e9)))
    # iplot2=np.where(np.abs(Eg-1000.0e9)==np.min(np.abs(Eg-1000.0e9)))

    # print(iplot1, Eg[iplot1])
    # print(iplot2, Eg[iplot2])

    # ax.plot(t,tau_ph[30,:],'r-',linewidth=3.0,label=r'$E_\gamma=100\,{\rm GeV}$')
    # # ax.plot(t,tau_ph[40,:],'g--',linewidth=3.0,label=r'$E_\gamma=1000\,{\rm GeV}$')
    # ax.plot(t,-np.log(tau_ph1[30,:]),'g--',linewidth=3.0,label=r'$E_\gamma=100\,{\rm GeV}$')
    # ax.plot(t,-np.log(tau_ph2[30,:]),'y:',linewidth=3.0,label=r'$E_\gamma=100\,{\rm GeV}$')


    # ax.set_xlim(0.0,30.0)
    # ax.set_yscale('log')
    # ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    # ax.set_ylabel(r'$\tau_{\gamma\gamma}$',fontsize=fs)
    # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label_ax.set_fontsize(fs)
    # ax.legend(loc='upper right', prop={"size":fs})
    # ax.grid(linestyle='--')

    # plt.savefig('Results_jax_wiad/fg_tau_ph.png')