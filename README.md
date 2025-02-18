## Transient gamma rays from nova shocks

We model lightcurves and spectra of hadronic gamma rays from a single nova shock taken into account the effect of gamma-ray absorption. Three ingredients are requires: i) particle acceleration model from a single nova shock, ii) cross sections for gamma-ray production and absorption, and iii) optical photon distribution of the nova under consideration  

**Particle acceleration model**: We use a model of particle acceleration in a single shock (see e.g. the appendix of [Aharonian et al. 2022](https://ui.adsabs.harvard.edu/abs/2022Sci...376...77H/abstract)). In this model, particles are injected around the hock with a power law in momentum with a maximum energy that changes in time derived from the acceleration rate and energy loss due to adiabatic expansion. The accumulated spectrum of particles are then solved from the transport equation and the solution can be found in <code>func_JEp_p_ark(pars_nova, E, t)</code> in <code>pack_nova.py</code>.     

**Cross sections**: We estimate the gamma-ray production using cross sections from [Kafexhiu et al. 2014](https://ui.adsabs.harvard.edu/abs/2014PhRvD..90l3014K/abstract). We use directly the associated library [LibppGam](https://github.com/ervinkafex/LibppGam/blob/main/Python/LibppGam.py) from which 4 sets of parametrizations for cross-sections as simulated from Geant 4, Pythia, SIBYLL, and QGSJET. We choose the Geant 4 parametrization for this estimate. The gamma-ray production cross-section is computed using <code>0_cross_section.py</code>. For gamma-ray absorption, we use the approximate cross-section from [Aharonian et al. 2013](https://ui.adsabs.harvard.edu/abs/2013SAAS...40.....A/abstract) (see <code>func_sigma_gg(Eg, Ebg)</code>).  

**Optical lightcurve**: We apply this to the case of the 2021 outburst of RS Oph and the optical lightcurve is as in [Cheung et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935...44C/abstract). Optical photons are assumed to follow a blackbody distribution with temperature of about 10<sup>4</sup> K. The gamma-ray flux with absorption is computed in <code>func_phi_PPI(pars_nova, eps_nucl, d_sigma_g, sigma_gg, E, Eg, t)</code> in <code>pack_nova.py</code>.

**Instructions to run the code** 
Clone the repository and create a folder called <code>Results_jax_wiad</code>. Make sure that you have [<code>JAX</code>](https://docs.jax.dev/en/latest/quickstart.html) installed if not run the following.
```sh
pip3 install jax
```

Then simply run
```sh
python3 0_cross_section.py
python3 1_jax_nova_phsp.py
```
You will find all results in this folder including the lightcurve as shown below. 

![Gamma-ray lightcurve for RS Ophiuchi](https://drive.google.com/uc?export=view&id=1KFCYB4k-Ir4q6RyQ78lLirGyJdv-I4u9)

For testing different parameters, simply change the grid of parameters in <code>1_jax_nova_phsp.py</code> (between line 28 and 42). Part of the code to scan the parameter space in an automatic manner with gradient descent is already built but not yet ready to be used. All the main parameters are as listed below.
```sh
    # Initialized parameters for RS Ophiuchi 2021  
    tST=2.2                          # day     -> Time where shock transition to Sedov-Taylor phase
    alpha=0.43                       # no unit -> Index for time profile of shock speed
    vsh0=3500.0                      # km/s    -> Initial shock speed
    Mdot=6.3e-7                      # Msol/yr -> Mass loss rate of red giant
    vwind=20.0                       # km/s    -> Wind speed of red giant
    Rmin=1.48                        # au      -> Distance between red giant and white dwarf
    xip=0.14                         # no unit -> Fraction of shock ram pressure converted into cosmic-ray energy
    delta=4.2                        # no unit -> Injection spectrum index
    epsilon=1.0                      # no unit -> Index of the exp cut-off for injection spectrum
    BRG=6.8                          # G       -> Magnetic field srength at the pole of red giant
    TOPT=1.0e4                       # K       -> Temperature of the optical photons 
    ter=0.0                          # day     -> Shock formation time
    Ds=2.45e3 #1.4e3                 # pc      -> Distance to Earth of the nova
    Rph=200.0*0.00465                # au      -> Photospheric radius of the nova
```

