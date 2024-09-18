import gato.pack_nova as nv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

pars_nova=[4500.0, 2.0, 0.66, 6.0e-7, 20.0, 1.48, 0.1, 4.4, 1.0, 0.0, 1.0, 1.0e4, 10, 2.0e-9, 1.4e3, 'DM23', 50]

t=np.linspace(-1.0,10.0,241)*86400.0
E0=np.logspace(8, 14, 6)

sol=sp.integrate.solve_ivp(lambda tp,E:(nv.func_dE_adi(pars_nova,E,tp/86400.0)),[t[0],t[-1]],E0,t_eval=t,method='RK45')
E=sol.y
print(E.shape)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(len(E0)):
    plt.plot(t/86400.0, E[i,:], label=f'E0=%.1e' % E0[i])

plt.xlabel('Time t')
plt.ylabel('E(t)')
plt.yscale('log')
plt.title('Solutions of dE/dt = b(E, t) for different initial conditions')
plt.legend(loc='upper right', fontsize='small')
plt.show()
