import numpy as np
from scipy.integrate import quad
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

# Constants
Rsh = 2.0  # Example value, set appropriately
Rph = 1.0  # Example value, set appropriately
tau_0=1.0

# Define r(ε, s)
def r(eps, s):
    return np.sqrt(s**2 - 2.0*s*Rsh*np.sqrt(1.0-eps**2) + Rsh**2)

# Define g_opt(x)
def g_opt(x):
    return (1.0/x) - ((1.0/x**2) - 1.0)*np.log(np.sqrt(np.abs((x-1.0)/(x+1.0))))

# Inner integral
def inner_integral(s, eps):
    return (3*r(eps, s)/(2*Rph)) * g_opt(r(eps, s)/Rph)

# Outer integral
def outer_integral(eps):
    inner_result, _ = quad(inner_integral, 0.0, 100.0*Rsh, args=(eps,))
    return np.exp(-tau_0*(Rsh/Rph) * inner_result) * eps/ np.sqrt(1.0-eps**2)

def func_inner_int_test(eps):
    tau_ph=tau_0*(Rsh/Rph)

    def func_r(eps, s):
        return jnp.sqrt(s**2 - 2.0*s*Rsh*jnp.sqrt(1.0-eps**2) + Rsh**2)

    def func_gopt(x):
        return (1.0/x) - ((1.0/x**2) - 1.0)*jnp.log(jnp.sqrt(jnp.abs((x-1.0)/(x+1.0))))
    
    s=jnp.linspace(0.0, 100.0*Rsh, 5000)
    ds=jnp.append(jnp.diff(s),0)
    inner_int=jnp.sum((3*func_r(eps, s)/(2*Rph)) * func_gopt(func_r(eps, s)/Rph) * ds/Rph)
    
    return jnp.exp(-tau_ph * inner_int) * eps /jnp.sqrt(1.0-eps**2)

def func_tau1(tau_ph):
    def func_inner_int(eps):

        def func_r(eps, s):
            return jnp.sqrt(s**2 - 2.0*s*Rsh*jnp.sqrt(1.0-eps**2) + Rsh**2)

        def func_gopt(x):
            return (1.0/x) - ((1.0/x**2) - 1.0)*jnp.log(jnp.sqrt(jnp.abs((x-1.0)/(x+1.0))))
        
        s=jnp.linspace(0, 100.0*Rsh, 5000)
        ds=jnp.append(jnp.diff(s),0)
        inner_int=jnp.sum((3*func_r(eps, s)/(2*Rph)) * func_gopt(func_r(eps, s)/Rph) * ds/Rph)
        
        return jnp.exp(-tau_ph * inner_int) * eps /jnp.sqrt(1.0-eps**2)

    eps=jnp.linspace(0.0, 0.9999, 5000)
    deps=jnp.append(jnp.diff(eps), 0.0)
    result=jnp.sum(jax.vmap(func_inner_int)(eps)*deps)

    return result

def func_tau2(tau_ph):
    def func_inner_int(eps):

        def func_r(eps, s):
            return jnp.sqrt(s**2 - 2.0*s*Rsh*jnp.sqrt(1.0-eps**2) + Rsh**2)

        def func_gopt(x):
            return (1.0/x) - ((1.0/x**2) - 1.0)*jnp.log(jnp.sqrt(jnp.abs((x-1.0)/(x+1.0))))
        
        s=jnp.linspace(2.0*Rsh*jnp.sqrt(1.0-eps**2), 100.0*Rsh, 5000)
        ds=jnp.append(jnp.diff(s),0)
        inner_int=jnp.sum((3*func_r(eps, s)/(2*Rph)) * func_gopt(func_r(eps, s)/Rph) * ds/Rph)
        
        return jnp.exp(-tau_ph * inner_int) * eps /jnp.sqrt(1.0-eps**2)

    eps=jnp.linspace(0.0, 0.9999, 5000)
    deps=jnp.append(jnp.diff(eps), 0.0)
    result=jnp.sum(jax.vmap(func_inner_int)(eps)*deps)

    return result

def func_tau_test(tau_ph, eps_min, eps_max):
    def func_inner_int(eps):

        def func_r(eps, s):
            return jnp.sqrt(s**2 - 2.0*s*Rsh*jnp.sqrt(1.0-eps**2) + Rsh**2)

        def func_gopt(x):
            return (1.0/x) - ((1.0/x**2) - 1.0)*jnp.log(jnp.sqrt(jnp.abs((x-1.0)/(x+1.0))))
        
        s=jnp.linspace(0.0, 100.0*Rsh, 5000)
        ds=jnp.append(jnp.diff(s),0)
        inner_int=jnp.sum((3*func_r(eps, s)/(2*Rph)) * func_gopt(func_r(eps, s)/Rph) * ds/Rph)
        
        return jnp.exp(-tau_ph * inner_int) * eps /jnp.sqrt(1.0-eps**2)

    eps=jnp.linspace(eps_min, eps_max, 10000)
    deps=jnp.append(jnp.diff(eps), 0.0)
    inner_int=jax.vmap(func_inner_int)(eps)
    result=jnp.sum(inner_int*deps)

    return result

def func_tau_full(tau_ph, eps_min):
    def func_inner_int(eps):

        def func_r(eps, s):
            return jnp.sqrt(s**2 - 2.0*s*Rsh*jnp.sqrt(1.0-eps**2) + Rsh**2)

        def func_gopt(x):
            return (1.0/x) - ((1.0/x**2) - 1.0)*jnp.log(jnp.sqrt(jnp.abs((x-1.0)/(x+1.0))))
        
        s=jnp.linspace(0.0, 100.0*Rsh, 5000)
        ds=jnp.append(jnp.diff(s),0)
        inner_int=jnp.sum((3*func_r(eps, s)/(2*Rph))*func_gopt(func_r(eps, s)/Rph)*ds/Rph)
        
        return jnp.exp(-tau_ph*inner_int)

    eps=jnp.linspace(eps_min, 0.999, 10000)
    deps=jnp.append(jnp.diff(eps), 0.0)
    inner_int=jax.vmap(func_inner_int)(eps)
    result=jnp.sum(inner_int*eps*deps/jnp.sqrt(1.0-eps**2))+inner_int[-1]*jnp.sqrt(1.0-0.999**2)

    return result

def func_tau_app(tau_ph, eps_min, eps_max):

    def func_gopt(x):
        return (1.0/x)-((1.0/x**2)-1.0)*jnp.log(jnp.sqrt(jnp.abs((x-1.0)/(x+1.0))))

    s=jnp.linspace(0.0, 100.0*Rsh, 5000)
    ds=jnp.append(jnp.diff(s),0)
    inner_int=jnp.sum((3*jnp.sqrt(s**2+Rsh**2)/(2*Rph))*func_gopt(jnp.sqrt(s**2+Rsh**2)/Rph)*ds/Rph)
        
    result=jnp.exp(-tau_ph*inner_int)*(jnp.sqrt(1.0-eps_min**2)-jnp.sqrt(1.0-eps_max**2))

    return result

def func_tau_ph1(tau_ph, t):
    def func_tau_full_single(tau_ph_single):
        def func_inner_int(eps):

            def func_r(eps, s):
                return jnp.sqrt(s**2 - 2.0*s*Rsh*jnp.sqrt(1.0-eps**2) + Rsh**2)

            def func_gopt(x):
                return (1.0/x) - ((1.0/x**2) - 1.0)*jnp.log(jnp.sqrt(jnp.abs((x-1.0)/(x+1.0))))
            
            s=jnp.linspace(0.0, 100.0*Rsh, 5000)
            ds=jnp.append(jnp.diff(s),0)
            inner_int=jnp.sum((3*func_r(eps, s)/(2*Rph))*func_gopt(func_r(eps, s)/Rph)*ds/Rph)
            
            return jnp.exp(-tau_ph_single*inner_int)

        eps=jnp.linspace(0.001, 0.999, 10000)
        deps=jnp.append(jnp.diff(eps), 0.0)
        inner_int=jax.vmap(func_inner_int)(eps)
        result=jnp.sum(inner_int*eps*deps/jnp.sqrt(1.0-eps**2))+inner_int[-1]*jnp.sqrt(1.0-0.999**2)

        return result

    # tau_ph1=jax.vmap(func_tau_full_single)(tau_ph.ravel())
    # tau_ph1=tau_ph1.reshape(tau_ph.shape)

    tau_ph1=jax.vmap(func_tau_full_single)(tau_ph[:,::10].ravel())
    tau_ph1=tau_ph1.reshape(tau_ph[:,::10].shape)

    def interp_tau_ph1(Eg_index):
        return jnp.interp(t, t[::10], tau_ph1[Eg_index, :], left=0.0, right=0.0) 

    tau_ph1_full=jax.vmap(interp_tau_ph1)(jnp.arange(len(Eg)))

    return tau_ph1_full

# print(func_inner_int_test(0.001))
# print(func_inner_int_test(0.99))
# eps=jnp.linspace(0.001, 0.99, 1000)
# print(jnp.where(jnp.isnan(jax.vmap(func_inner_int_test)(eps))))

# print(func_tau1(tau_0*(Rsh/Rph)))
# print(func_tau2(tau_0*(Rsh/Rph)))

A=func_tau_test(tau_0*(Rsh/Rph), 0.999, 0.999999999)
B=func_tau_app(tau_0*(Rsh/Rph), 0.999, 1.0)
C=func_tau_test(tau_0*(Rsh/Rph), 0.001, 0.999)

result1, error = quad(outer_integral, 0.001, 0.999)
result2, error = quad(outer_integral, 0.999, 0.999999999)

print("Integral Result:", jnp.log(result1+result2), jnp.log(C+B))
print(result2, B)

# Load the .npz file
data=np.load('tau_ph.npz')
Eg=data['Eg']
t=data['t']
tau_ph=data['tau_ph']

print(func_tau_ph1(tau_ph, t))
# print(func_tau_test(tau_ph[0,0], 0.001, 0.999999), func_tau_test(tau_ph[0,1], 0.001, 0.999999))
# print(func_tau_test(tau_ph[1,0], 0.001, 0.999999), func_tau_test(tau_ph[1,1], 0.001, 0.999999))

fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

iplot1=np.where(np.abs(Eg-100.0e9)==np.min(np.abs(Eg-100.0e9)))
iplot2=np.where(np.abs(Eg-1000.0e9)==np.min(np.abs(Eg-1000.0e9)))

print(iplot1, Eg[iplot1])
print(iplot2, Eg[iplot2])

ax.plot(t,tau_ph[30,:],'r-',linewidth=3.0,label=r'$E_\gamma=100\,{\rm GeV}$')
ax.plot(t,tau_ph[40,:],'g--',linewidth=3.0,label=r'$E_\gamma=1000\,{\rm GeV}$')

ax.set_xlim(0.0,30.0)
ax.set_yscale('log')
ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
ax.set_ylabel(r'$\tau_{\gamma\gamma}$',fontsize=fs)
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)
ax.legend(loc='upper right', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig('Results_jax_wiad/fg_tau_ph.png')