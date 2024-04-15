import numpy as np

me=0.510998e6 # eV
mp=938.272e6 # eV
kB=8.617333262145e-5 # eV/K
sigmaT=6.6524e-25 # cm^-2 -> Thompson cross-section
alpha_f=1.0/137.035999084 # -> Fine structure constant
meCGS=9.10938356e-28 # g
sigma_sb=3.5394474508e7 # erg cm^-2 s^-1 K^-4
qe=4.80320451e-10 # cgs unit
hP=4.1357e-15 # eV s
Ktomec2=1.6863699549e-10 # -> To convert temperture T from K/kB to eV/me 
mpi=134.9766e6 # eV
Tpth=2.0*mpi+(pow(mpi,2)/(2.0*mp)) # eV


############################################################################################
# Gamma rays from proton-proton interaction -> Kafexhiu et al. 2014
############################################################################################

# Dimensionless Breit-Wigner distribution.
def func_fBW(sqrt_s):
# Eq. 4.

    M_res=1.1883*pow(10.0,9) # eV
    Gamma_res=0.2264*pow(10.0,9) # eV
    gamma=np.sqrt(pow(M_res,2)*(pow(M_res,2)+pow(Gamma_res,2))) 
    K=np.sqrt(8.0)*M_res*Gamma_res*gamma/(np.pi*np.sqrt(pow(M_res,2)+gamma)) 
    
    fBW=mp*K/(pow(pow(sqrt_s-mp,2)-pow(M_res,2),2)+pow(M_res*Gamma_res,2)) 
    
    return fBW

# Cross-section for p+p -> p+p+pi0.
def func_sigma_1pi(Tp):
# Eq. 2,
# Tp (eV) -> kinetic energy of CR proton.

    sigma_0=7.66e-3 
    sqrt_s=np.sqrt(2.0*mp*(Tp+2.0*mp)) 
    eta=np.sqrt(pow(pow(sqrt_s,2)-pow(mpi,2)-4.0*pow(mp,2),2)-16.0*pow(mpi*mp,2))/(2.0*mpi*sqrt_s) 
    
    sigma_1pi=sigma_0*pow(eta,1.95)*(1.0+eta+pow(eta,5))*pow(func_fBW(sqrt_s),1.86) 
    
    return sigma_1pi # m

# Cross-section for p+p -> p+p+2pi0.
def func_sigma_2pi(Tp):
# Eq. 5,
# Tp (eV) -> kinetic energy of CR proton.
       
    mask1=(Tp<0.56e9)
    mask2=(Tp>=0.56e9)

    sigma_2pi=np.zeros_like(Tp)

    sigma_2pi[mask1]=0.0 
    sigma_2pi[mask2]=5.7/(1.0+np.exp(-9.3*(Tp[mask2]*1.0e-9-1.4))) 
    
    return sigma_2pi # m

# Proton-proton total inelastic cross-section. 
def func_sigma_inel(Tp):
# Eq. 1,
# Tp (eV) -> kinetic energy of CR proton.

    sigma_inel=30.7-0.96*np.log(Tp/Tpth)+0.18*pow(np.log(Tp/Tpth),2) 
    sigma_inel*=pow(1.0-pow(Tpth/Tp,1.9),3) 
    sigma_inel[sigma_inel<0.0]=0.0
    
    return sigma_inel # m

# Average pi0 multiplicity.
def func_npi(Tp):
# Eq. 7 and Table 4 (GEANT 4 model),
# Tp (eV) -> kinetic energy of CR proton.

    a1=0.728 
    a2=0.596 
    a3=0.491 
    a4=0.2503 
    a5=0.117 
    
    Qp=(Tp-Tpth)/mp 
    xip=(Tp-3.0e9)/mp 

    mask1=(Tp>=1.0e9) & (Tp<5.0e9)
    mask2=(Tp>=5.0e9)

    npi=np.zeros_like(Tp)

    npi[mask1]=-6.0e-3+0.237*Qp[mask1]-0.023*pow(Qp[mask1],2) 
    npi[mask2]=a1*pow(xip[mask2],a4)*(1.0+np.exp(-a2*pow(xip[mask2],a5)))*(1.0-np.exp(-a3*pow(xip[mask2],0.25))) 
    
    return npi

# Pi0 production cross-section.
def func_sigma_pi(Tp):
# See paragraph above Table 4,
# Tp (eV) -> kinetic energy of CR proton.
    
    mask1=(Tp>=Tpth) & (Tp<2.0e9)
    mask2=(Tp>=2.0e9)

    sigma_pi=np.zeros_like(Tp)

    sigma_pi[mask1]=func_sigma_1pi(Tp[mask1])+func_sigma_2pi(Tp[mask1]) 
    sigma_pi[mask2]=func_sigma_inel(Tp[mask2])*func_npi(Tp[mask2]) 
    
    return sigma_pi # m

# Complementary function for the differential cross-section.
def func_Amax(Tp):
# Eq. 12 and Table 7 (GEANT 4 model),
# Tp (eV) -> kinetic energy of CR proton.

    sqrt_s=np.sqrt(2.0*mp*(Tp+2.0*mp)) 
    gamma_CM=(Tp+2.0*mp)/sqrt_s 
    beta_CM=np.sqrt(1.0-pow(gamma_CM,-2)) 
    Epi_CM=(pow(sqrt_s,2)-4.0*pow(mp,2)+pow(mpi,2))/(2.0*sqrt_s) 
    Ppi_CM=np.sqrt(pow(Epi_CM,2)-pow(mpi,2)) 
    Epi_max=gamma_CM*(Epi_CM+Ppi_CM*beta_CM) 
    Epi_min=gamma_CM*(Epi_CM-Ppi_CM*beta_CM) 
    theta_p=Tp/mp 

    mask1=(Tp<Tpth)
    mask2=(Tp>=Tpth) & (Tp<1.0e9)
    mask3=(Tp>=1.0e9) & (Tp<5.0e9)
    mask4=(Tp>=5.0e9)

    Amax=np.zeros_like(Tp)

    Amax[mask1]=0.0 
    Amax[mask2]=5.9*func_sigma_pi(Tp[mask2])/Epi_max[mask2] 
    Amax[mask3]=9.53*pow(theta_p[mask3],-0.52)*np.exp(0.054*pow(np.log(theta_p[mask3]),2))*func_sigma_pi(Tp[mask3])/mp 
    Amax[mask4]=9.13*pow(theta_p[mask4],-0.35)*np.exp(9.7e-3*pow(np.log(theta_p[mask4]),2))*func_sigma_pi(Tp[mask4])/mp 
    
    return Amax # mb/e

# Complementary function for the differential cross-section.
def func_alpha(Tp):
# Table 5, Eq. 14, and Eq. 15,
# Tp (eV) -> kinetic energy of CR proton.

    mask1=(Tp>=Tpth) & (Tp<=20.0e9)
    mask2=(Tp>20.0e9)

    alpha=np.zeros_like(Tp)

    alpha[mask1]=1.0
    alpha[mask2]=0.5

    return alpha

# Complementary function for the differential cross-section.
def func_beta(Tp):
# Table 5, Eq. 14, and Eq. 15,
# Tp (eV) -> kinetic energy of CR proton.

    q=(Tp-1.0e9)/mp 
    mu=1.25*pow(q,1.25)*np.exp(-1.25*q) 
    theta_p=Tp/mp 
    
    mask1=(Tp>=Tpth) & (Tp<=1.0e9)
    mask2=(Tp>1.0e9) & (Tp<=4.0e9)
    mask3=(Tp>4.0e9) & (Tp<=20.0e9)
    mask4=(Tp>20.0e9) & (Tp<=100.0e9)
    mask5=(Tp>100.0e9)

    beta=np.zeros_like(Tp)

    beta[mask1]=3.29-0.2*pow(theta_p[mask1],-1.5) 
    beta[mask2]=mu[mask2]+2.45 
    beta[mask3]=1.5*mu[mask3]+4.95 
    beta[mask4]=4.2 
    beta[mask5]=4.9 
    
    return beta

# Complementary function for the differential cross-section.
def func_gamma(Tp):
# Table 5, Eq. 14, and Eq. 15,
# Tp (eV) -> kinetic energy of CR proton.
    
    q=(Tp-1.0e9)/mp 
    mu=1.25*pow(q,1.25)*np.exp(-1.25*q) 
    
    mask1=(Tp>=Tpth) & (Tp<1.0e9)
    mask2=(Tp>=1.0e9) & (Tp<=4.0e9)
    mask3=(Tp>4.0e9) & (Tp<=20.0e9)
    mask4=(Tp>20.0e9)

    gamma=np.zeros_like(Tp)

    gamma[mask1]=0.0 
    gamma[mask2]=mu[mask2]+1.45 
    gamma[mask3]=mu[mask3]+1.5 
    gamma[mask4]=1.0 

    return gamma

# Complementary function for the differential cross-section.
def func_F(Tp, Eg):
# Eq. 11 and Table 5,    
# Tp (eV) -> kinetic energy of CR proton,
# Eg (eV) -> energy of gamma ray.

    Tp, Eg=np.meshgrid(Tp, Eg, indexing='ij')

    sqrt_s=np.sqrt(2.0*mp*(Tp+2.0*mp)) 
    gamma_CM=(Tp+2.0*mp)/sqrt_s 
    beta_CM=np.sqrt(1.0-pow(gamma_CM,-2)) 
    Epi_CM=(pow(sqrt_s,2)-4.0*pow(mp,2)+pow(mpi,2))/(2.0*sqrt_s) 
    Ppi_CM=np.sqrt(pow(Epi_CM,2)-pow(mpi,2)) 
    
    Epi_max_LAB=gamma_CM*(Epi_CM+Ppi_CM*beta_CM) 
    gammapi_LAB=Epi_max_LAB/mpi 
    betapi_LAB=np.sqrt(1.0-pow(gammapi_LAB,-2)) 
    Eg_max=mpi*gammapi_LAB*(1.0+betapi_LAB)/2.0 
    Eg_min=mpi*gammapi_LAB*(1.0-betapi_LAB)/2.0 
    
    mask=(Eg>=Eg_min) & (Eg<=Eg_max)
        
    Yg=Eg+pow(mpi,2)/(4.0*Eg) 
    Yg_max=Eg_max+pow(mpi,2)/(4.0*Eg_max) # Yg_max=mpi*gammapi_LAB
    Xg=(Yg-mpi)/(Yg_max-mpi) 
    C=3.0*mpi/Yg_max 

    print(func_alpha(Tp))

    F=np.where(mask, pow(1.0-pow(Xg,func_alpha(Tp)),func_beta(Tp))/pow(1.0+Xg/C,func_gamma(Tp)), 0)
    
    return F

# Complementary function for the nuclear enhancement factor.
def func_GTp(Tp):
# Eq. 19,
# Tp (eV) -> kinetic energy of CR proton.

    GTp=1.0+np.log(np.maximum(1.0,func_sigma_inel(Tp)/func_sigma_inel(1.0e12*Tp**0))) 
    
    return GTp

# Nuclear enhancement factor. 
def func_enhancement(Tp):
# Eq. 24,
# Tp (eV) -> kinetic energy of CR proton.

    eps_nucl=np.zeros_like(Tp)

    mask1=(Tp>=Tpth) & (Tp<1.0e9)
    mask2=(Tp>=1.0e9)

    eps_nucl[mask1]=1.7
    eps_nucl[mask2]=1.37+0.39*10.0*np.pi*func_GTp(Tp[mask2])/func_sigma_inel(Tp[mask2]) 

    return eps_nucl

# Differential cross-section of gamma-ray from pi0 decay.
def func_d_sigma_g(Tp, Eg):
# Eq. 8,  
# Tp (eV) -> kinetic energy of CR proton,
# Eg (eV) -> energy of gamma ray.

    Amax=np.zeros_like(Tp)

    mask=(Tp>=Tpth)

    Amax[mask]=func_Amax(Tp[mask])
    d_sigma_g=Amax[:,np.newaxis]*func_F(Tp,Eg)*1.0e-27

    return d_sigma_g # cm^2/eV