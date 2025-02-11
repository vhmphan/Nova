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
# import gato.pack_gato as gt
import Functions.pack_nova as pn
from jax import jit
import matplotlib.ticker as ticker
import diffrax
import scipy as sp
from jax import grad
import optax


if __name__ == "__main__":

    # Record the starting time
    start_time=time.time()

    # Initialized parameters for RS Ophiuchi 2021  
    tST=2.2                          # day     -> Time where shock transition to Sedov-Taylor phase
    alpha=0.43                       # no unit -> Index for time profile of shock speed
    vsh0=3500.0                      # km/s    -> Initial shock speed
    Mdot=6.3e-7                      # Msol/yr -> Mass loss rate of red giant
    vwind=20.0                       # km/s    -> Wind speed of red giant
    Rmin=1.48                        # au      -> Distance between red giant and white dwarf
    xip=0.14                          # no unit -> Fraction of shock ram pressure converted into cosmic-ray energy
    delta=4.2                        # no unit -> Injection spectrum index
    epsilon=1.0                      # no unit -> Index of the exp cut-off for injection spectrum
    BRG=6.8                          # G       -> Magnetic field srength at the pole of red giant
    TOPT=1.0e4                       # K       -> Temperature of the optical photons 
    ter=0.0                          # day     -> Shock formation time
    Ds=2.45e3 #1.4e3                 # pc      -> Distance to Earth of the nova
    Rph=200.0*0.00465                # au      -> Photospheric radius of the nova
    pars_nova=jnp.array([vsh0, tST, alpha, Mdot, vwind, Rmin, xip, delta, epsilon, ter, BRG, TOPT, Ds, Rph])

    # Define the time and energy ranges -> note that it is required that t[0]<=ter 
    t=jnp.linspace(ter,40.0,int((40.0-ter)*100.0)+1) # day

    # Load the energy ranges and pre-computed cross-sections
    data_d_sigma_g=np.load('Data/d_sigma_g.npz')
    E=data_d_sigma_g['E']
    Eg=data_d_sigma_g['Eg']
    eps_nucl=data_d_sigma_g['eps_nucl']
    d_sigma_g=data_d_sigma_g['d_sigma_g']

    # Gamma-gamma cross section
    Ebg=jnp.logspace(jnp.log10(pn.kB*pars_nova[11]*1.0e-2), jnp.log10(pn.kB*pars_nova[11]*1.0e2), 1000) # eV
    dEbg=jnp.append(jnp.diff(Ebg), 0.0)[jnp.newaxis,:,jnp.newaxis]        # eV
    sigma_gg=pn.func_sigma_gg(Eg, Ebg)[:,:,jnp.newaxis]                   # cm^2

    mytodo='plot'

    # Scan parameter space if mytodo='scan' and simply plot if mytodo!='scan'
    if(mytodo=='scan'):
        @jit
        def func_loss_fixed(sub_pars): 
            pars_scan=jnp.array([vsh0, sub_pars[0], alpha, sub_pars[1], vwind, Rmin, xip, delta, epsilon, ter, sub_pars[2], TOPT, Ds])

            return pn.func_loss(pars_scan, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t)

        N_epoch=2000
        sub_pars_arr=[]
        chi2_arr=[]

        sub_pars=jnp.array([tST, Mdot, BRG])
        sub_pars_min=jnp.array([2.0, 1.0e-7, 0.1])
        sub_pars_max=jnp.array([6.0, 10.0e-7, 10.0])

        grads_init=jnp.abs(grad(func_loss_fixed)(sub_pars))
        lr=0.1*sub_pars/grads_init# /(grads_init+1.0e-8))
        optimizer=optax.adam(lr)
        opt_state=optimizer.init(sub_pars)

        for i in range(N_epoch):
            grads=grad(func_loss_fixed)(sub_pars)
            updates, opt_state=optimizer.update(grads, opt_state)
            sub_pars=optax.apply_updates(sub_pars, updates)
            sub_pars=jnp.clip(sub_pars, sub_pars_min, sub_pars_max)

            chi2=func_loss_fixed(sub_pars)
            chi2_arr.append(chi2)
            sub_pars_arr.append(sub_pars)

            if(i%int(N_epoch/10)==0):
                print(i, sub_pars, chi2)
        
        sub_pars_array=np.array(sub_pars_arr)
        chi2_array=np.array(chi2_arr)
        np.savez_compressed('Results_jax_wiad/pars_scan_bf.npz', sub_pars=sub_pars_array, chi2=chi2_array)
    else:
        # data=np.load('Results_jax_wiad/pars_scan.npz')
        # sub_pars_array=data['sub_pars']
        # chi2_array=data['chi2']
        # i_best_fit=jnp.where(chi2_array==np.min(chi2_array))
        # sub_best=sub_pars_array[i_best_fit][0]    
    
        pars_best=pars_nova # jnp.array([vsh0, sub_best[0], alpha, sub_best[1], vwind, Rmin, xip, delta, epsilon, ter, sub_best[2], TOPT, Ds])
        print(pars_best[1], pars_nova[3], pars_nova[10])

        phi_PPI, tau_ph, tau_ph1, tau_ph2=pn.func_phi_PPI(pars_best, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t)
        pn.plot_gamma(pars_best, phi_PPI, tau_ph1, tau_ph2, Eg, t, 1.6)
        pn.plot_gamma(pars_best, phi_PPI, tau_ph1, tau_ph2, Eg, t, 3.6)
        pn.plot_gamma(pars_best, phi_PPI, tau_ph1, tau_ph2, Eg, t, 5.6)
        pn.plot_time_gamma(pars_best, phi_PPI, tau_ph1, tau_ph2, Eg, t)
        pn.plot_vsh(pars_nova, t)
        pn.plot_LOPT(pars_nova)
        pn.plot_flux_OPT(pars_nova, t)
        pn.plot_tau_ph(pars_nova, tau_ph1, tau_ph2, Eg, t)

        np.savez('tau_ph.npz', Eg=Eg, t=t, tau_ph=tau_ph)

    fs=22

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    Ebg=jnp.linspace(pn.kB*pars_nova[11], 2.0*pn.kB*pars_nova[11], 1)

    sigma_gg_A13=pn.func_sigma_gg(Eg, Ebg)
    sigma_gg_C90=pn.func_sigma_gg_C90(Eg, Ebg)
    sigma_gg_G67=pn.func_sigma_gg_G67(Eg, Ebg)

    ax.plot(Eg, sigma_gg_A13[:, 0],'g--',linewidth=3.0,label=r'${\rm Aharonian\, 2013}$')
    ax.plot(Eg, sigma_gg_G67[:, 0], 'm-.', label=r'${\rm Gould \, 1967}$')
    ax.plot(Eg, sigma_gg_C90[:, 0], 'y--', label=r'${\rm Coppi \, 1990}$')

    ax.set_xlim(1.0e11,1.0e14)
    ax.set_ylim(1.0e-28,1.0e-24)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E_\gamma\, {\rm (eV)}$',fontsize=fs)
    ax.set_ylabel(r'$\sigma_{\gamma\gamma}(E_\gamma, k_BT_{\rm opt})\,{\rm (cm^{2})}$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('Results_jax_wiad/fg_sigma_gg.png')

    # NEp_ark=func_JEp_p_ark(pars_nova, E, t)
    # NEp_rk4=func_JEp_p_rk4(pars_nova, E, t)

    # EnJEp_ark=E[:,jnp.newaxis]**3*NEp_ark
    # EnJEp_rk4=E[:,jnp.newaxis]**3*NEp_rk4

    # fig=plt.figure(figsize=(10, 8))
    # ax=plt.subplot(111)

    # ax.plot(E, EnJEp_ark[:, t==0.0], '-', color='green', linewidth=3, label='Day 0')
    # ax.plot(E, EnJEp_rk4[:, t==0.0], '--', color='black', linewidth=3)

    # ax.plot(E, EnJEp_ark[:, t==1.0], '-', color='red', linewidth=3, label='Day 1')
    # ax.plot(E, EnJEp_rk4[:, t==1.0], '--', color='black', linewidth=3)

    # ax.plot(E, EnJEp_ark[:, t==5.0], '-', color='orange', linewidth=3, label='Day 5')
    # ax.plot(E, EnJEp_rk4[:, t==5.0], '--', color='black', linewidth=3)

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlim(1.0e8, 1.0e14)
    # ax.set_ylim(1.0e60, 1.0e76)
    # ax.set_xlabel(r'$E\, {\rm (eV)}$',fontsize=fs)
    # ax.set_ylabel(r'$J(E) \, ({\rm eV^{2}\, cm\, s^{-1} })$',fontsize=fs)
    # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label_ax.set_fontsize(fs)
    # ax.legend(loc='upper right', prop={"size":fs})
    # ax.grid(linestyle='--')

    # plt.savefig('Results_jax_wiad/fg_jax_JEp.png')
    # plt.close()

    # fig=plt.figure(figsize=(10, 8))
    # ax=plt.subplot(111)
    # ax.plot(t,func_vsh(pars_nova, t),'r--',linewidth=3.0)
    # ax.plot(t,func_vsh_step(pars_nova, t),'k:',linewidth=3.0)

    # # ax.plot(t,func_vsh_step(jnp.array([vsh0, tST, alpha, Mdot, vwind, Rmin, xip, delta, epsilon, 0.5, BRG, TOPT, Ds]), t),'r--',linewidth=3.0)
    # # ax.plot(t+0.5,func_vsh(jnp.array([vsh0, tST-0.5, alpha, Mdot, vwind, Rmin, xip, delta, epsilon, 0.0, BRG, TOPT, Ds]), t),'k:',linewidth=3.0)

    # # ax.set_xscale('log')
    # # ax.set_yscale('log')
    # ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    # ax.set_ylabel(r'$v_{\rm sh} \, ({\rm km/s})$',fontsize=fs)
    # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label_ax.set_fontsize(fs)
    # ax.legend(loc='upper right', prop={"size":fs})
    # ax.grid(linestyle='--')
    # ax.set_ylim(100.0,10000.0)

    # plt.savefig('Results_jax_wiad/fg_jax_vsh.png')

    # fig=plt.figure(figsize=(10, 8))
    # ax=plt.subplot(111)

    # ax.set_xlim(np.log10(1.0),np.log10(50.0))

    # t_vsh, vsh_a, err_vsh_a, vsh_b, err_vsh_b=np.loadtxt('vsh_line.txt',unpack=True,usecols=[0,1,2,3,4])
    # ax.errorbar(np.log10(t_vsh),np.log10(vsh_a),yerr=(err_vsh_a/vsh_a)/np.log(10.0),xerr=t_vsh*0.0,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='red',markeredgecolor='black',markersize=10,label=r'${\rm H\alpha}$')
    # ax.errorbar(np.log10(t_vsh),np.log10(vsh_b),yerr=(err_vsh_b/vsh_b)/np.log(10.0),xerr=t_vsh*0.0,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='green',markeredgecolor='black',markersize=10,label=r'${\rm H\beta}$')

    # t_vsh_Xray, T_Xray, T_Xray_upper, t_vsh_Xray_lower=np.loadtxt('vsh_Xray.txt',unpack=True,usecols=[0,1,2,4])
    # t_vsh_Xray+=1.0
    # t_vsh_Xray_lower+=1.0
    # vsh_Xray=np.sqrt(16.0*T_Xray*1.0e3/(3.0*1.0e-24*3.0e10**2*6.242e11))*3.0e5 # km/s
    # vsh_Xray_upper=np.sqrt(16.0*T_Xray_upper*1.0e3/(3.0*1.0e-24*3.0e10**2*6.242e11))*3.0e5 # km/s
    # err_vsh_Xray=vsh_Xray_upper-vsh_Xray
    # err_t_vsh_Xray=t_vsh_Xray-t_vsh_Xray_lower
    # ii=np.where(err_vsh_Xray<0.0)
    # print(t_vsh_Xray[ii])
    # ax.errorbar(np.log10(t_vsh_Xray),np.log10(vsh_Xray),yerr=(err_vsh_Xray/vsh_Xray)/np.log(10.0),xerr=(err_t_vsh_Xray/t_vsh_Xray)/np.log(10.0),fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='orange',markeredgecolor='black',markersize=10,label=r'${\rm X-ray}$')

    # img=mpimg.imread("Data/data_vsh_Xray.png")
    # img_array = np.mean(np.array(img), axis=2)

    # xmin=np.log10(1.0)
    # xmax=np.log10(50.0)
    # ymin=np.log10(700.0)
    # ymax=np.log10(5000.0)
    # ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    # ax.plot(np.log10(t), np.log10(func_vsh(pars_nova, t)), '--', color='black', linewidth=8)
    # # ax.plot(np.log10(t), np.log10(func_vsh_step(pars_nova, t)), ':', color='black', linewidth=8)

    # ax.set_aspect((xmax-xmin)/(ymax-ymin))
    # ax.set_xticks([np.log10(1), np.log10(10)])
    # ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: r'$10^{%d}$' % int(x)))
    # ax.set_yticks([np.log10(1000.0), np.log10(5000.1)])
    # ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: r'$%d$' % int(10**x)))

    # ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    # ax.set_ylabel(r'$v_{\rm sh} \, ({\rm km/s})$',fontsize=fs)
    # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label_ax.set_fontsize(fs)
    # ax.legend(loc='upper right', prop={"size":fs})
    # ax.grid(linestyle='--')

    # plt.savefig('Results_jax_wiad/fg_jax_vsh_Xray.png')
    # plt.close()

    # fig=plt.figure(figsize=(10, 8))
    # ax=plt.subplot(111)
    # # ax.plot(t,func_dE_adi_dE(pars_nova, 1.0e9, t),'r--',linewidth=3.0)
    # # ax.plot(t,func_dE_adi_dE_step(pars_nova, 1.0e9, t),'k:',linewidth=3.0)

    # ax.plot(t,jnp.cumsum(func_dE_adi(pars_nova, 1.0e9, t)),'r--',linewidth=3.0)
    # ax.plot(t,jnp.cumsum(func_dE_adi_step(pars_nova, 1.0e9, t)),'k:',linewidth=3.0)
    
    # ax.set_xscale('log')
    # ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    # ax.set_ylabel(r'$v_{\rm sh} \, ({\rm km/s})$',fontsize=fs)
    # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label_ax.set_fontsize(fs)
    # ax.legend(loc='upper right', prop={"size":fs})
    # ax.grid(linestyle='--')
    # # ax.set_ylim(-100.0,0.1)

    # plt.savefig('Results_jax_wiad/fg_jax_dE_adi.png')

    # fig=plt.figure(figsize=(10, 8))
    # ax=plt.subplot(111)
    # # ax.plot(t,func_Rsh(pars_nova, t),'r--',linewidth=3.0)
    # # ax.plot(t,func_Rsh_step(pars_nova, t),'k:',linewidth=3.0)

    # # ax.plot(t,func_Rsh_step(pars_nova, t),'r--',linewidth=3.0)
    # ax.plot(t,func_Rsh(pars_nova, t),'k-',linewidth=3.0)
    # ax.plot(t,jnp.cumsum(func_vsh(pars_nova, t))*(t[1]-t[0])*86400.0*6.68459e-9,'g-.',linewidth=3.0)

    # # ax.set_xscale('log')
    # ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    # ax.set_ylabel(r'$R_{\rm sh} \, ({\rm au})$',fontsize=fs)
    # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label_ax.set_fontsize(fs)
    # ax.legend(loc='upper right', prop={"size":fs})
    # ax.grid(linestyle='--')

    # plt.savefig('Results_jax_wiad/fg_jax_Rsh.png')

    # fig=plt.figure(figsize=(10, 8))
    # ax=plt.subplot(111)

    # E0=func_E0(pars_nova, E, t).T
    # for i in range(len(E)):
    #     if((i%10==0) & (E[i]>1.0e11)):
    #         ax.plot(t, E0[i,:], '-', linewidth=3, label='%.2e' % E[i])

    # # ax.set_xscale('log')
    # ax.set_yscale('log')
    # # ax.set_xlim(1.0e8, 1.0e14)
    # # ax.set_ylim(1.0e70, 1.0e76)
    # ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    # ax.set_ylabel(r'$E_0 \, ({\rm eV})$',fontsize=fs)
    # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label_ax.set_fontsize(fs)
    # ax.legend(loc='upper right', prop={"size":fs})
    # ax.grid(linestyle='--')

    # plt.savefig('Results_jax_wiad/fg_jax_E0.png')
    # plt.close()

    # fig=plt.figure(figsize=(10, 8))
    # ax=plt.subplot(111)

    # r=np.logspace(-3,2,100)

    # ax.plot(r,func_uOPT_r2(pars_nova,r,np.array([1.6])),'r-',linewidth=3.0)
    # ax.plot(r,func_uOPT_r2(pars_nova,r,np.array([3.6])),'g-',linewidth=3.0)
    # ax.plot(r,func_uOPT_r2(pars_nova,r,np.array([5.6])),'-', color='orange', linewidth=3.0)
    # ax.plot(r,func_uOPT_rt(pars_nova,r,np.array([1.6])),'r--',linewidth=5.0, label=r'${\rm Night\, 1}$')
    # ax.plot(r,func_uOPT_rt(pars_nova,r,np.array([3.6])),'g--',linewidth=5.0, label=r'${\rm Night\, 3}$')
    # ax.plot(r,func_uOPT_rt(pars_nova,r,np.array([5.6])),'--', color='orange',linewidth=5.0, label=r'${\rm Night\, 5}$')

    # ax.legend()
    # # ax.set_xscale('log')
    # ax.set_yscale('log')
    # # ax.set_xlim(3.0e-1,1.0e2)
    # # ax.set_ylim(1.0e-3,1.0e2)

    # ax.set_xticks([0, 1, 2, 3, 4, 5])

    # ax.set_xlim(0.0,5.0)
    # ax.set_ylim(1.0e-2,1.0e2)
    # ax.set_xlabel(r'$r\, {\rm (au)}$',fontsize=fs)
    # ax.set_ylabel(r'$u_{\rm opt} \, ({\rm erg\,cm^{-3}})$',fontsize=fs)
    # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label_ax.set_fontsize(fs)
    # ax.legend(loc='upper right', prop={"size":fs})
    # ax.grid(linestyle='--')

    # plt.savefig('Results_jax_wiad/fg_uOPT.png')

    # Record the ending time
    end_time=time.time()

    # Calculate the elapsed time
    elapsed_time=end_time-start_time

    print("Elapsed time:", elapsed_time, "seconds")