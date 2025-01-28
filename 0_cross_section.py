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
from jax import grad
import optax

fs=22

# Define the energy ranges -> note that it is required that t[0]<=ter 
E=jnp.logspace(8,14,61)  # eV
Eg=jnp.logspace(8,14,61) # eV

# Gamma-ray production cross-section
eps_nucl=jnp.array(gt.func_enhancement(np.array(E)))[:, jnp.newaxis, jnp.newaxis] # no unit
d_sigma_g=jnp.array(gt.func_d_sigma_g(np.array(E), np.array(Eg)))[:, :, jnp.newaxis]        # cm^2/eV

# Save arrays into npz file
np.savez('Data/d_sigma_g.npz', E=E, Eg=Eg, eps_nucl=eps_nucl, d_sigma_g=d_sigma_g)