///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gamma rays from proton-proton interaction -> Kafexhiu et al. 2014
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Dimensionless Breit-Wigner distribution.
double func_fBW(double sqrt_s);

// Cross-section for p+p -> p+p+pi0.
double func_sigma_1pi(double Tp);

// Cross-section for p+p -> p+p+2pi0.
double func_sigma_2pi(double Tp);

// Proton-proton total inelastic cross-section 
double func_sigma_inel(double Tp);

// Average pi0 multiplicity.
double func_npi(double Tp);

// Pi0 production cross-section.
double func_sigma_pi(double Tp);

// Complementary function for the differential cross-section.
double func_Amax(double Tp);

// Complementary function for the differential cross-section.
double func_alpha(double Tp);

// Complementary function for the differential cross-section.
double func_beta(double Tp);

// Complementary function for the differential cross-section.
double func_gamma(double Tp);

// Complementary function for the differential cross-section.
double func_F(double Tp, double Eg);

// Complementary function for the nuclear enhancement factor.
double func_GTp(double Tp);

// Nuclear enhancement factor. 
double func_enhancement(double Tp);

// Differential cross-section of gamma-ray from pi0 decay.
double func_d_sigma_g(double Tp, double Eg);

// Volume emissitivity of hadronic gamma rays.
double func_eps_PPI(double nH, double Tp, double Eg);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gamma rays from inverse Compton scattering -> Khangulyan et al. 2014
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Complementary functions for gamma rays from IC.
double func_G34(double x0, double* a);

// Volume emissitivity of upscattered thermal photons by one electron (IC of a black-body or a gray-body photon distribution). 
double func_eps_ICS(double Urad, double Trad, double E, double Eg);

// Complementary functions for gamma rays from IC.
double func_G(double q, double Lambda);

// // Volume emissitivity of upscattered thermal photons by one electron (IC of a black-body or a gray-body photon distribution). 
// double func_eps_ICS(double Urad, double Trad, double E, double Eg);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bremsstrahlung radiation -> Schlickeiser 2002
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Volume emissitivity of bremstrahlung radiation.
double func_eps_BRE(double nH, double E, double Eg);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Radio and X-ray from synchrotron radiation -> Zirakashvili & Aharonian 2007
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Volume emissitivity of synchrotron photons. 
double func_eps_SYN(double B, double E, double Eg);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Photoionization cross-ection of Iron Kalpha -> Fujita et al. 2021
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

double func_sigma_Kalpha_Xray(double Eg);

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // Photoelectric absorption cross-section per H atom -> Morrison & McCammon 1983
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// double func_sigma_abs(double E);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cross-section and opacity due to gamma-gamma interaction 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Photon density of the thermal bremsstrahlung
double *func_fEtB(double ne, double Te, double R, double *arr_Ebg, int NEbg);

// Photon density of the background photons
double *func_fEtd(double Urad, double Trad, double sigma, double *arr_Ebg, int NEbg);

// Cross section for gamma gamma interaction
double func_sigma_gg(double Eg, double Ebg);

// Opacity for gamma gamma interaction
double func_tau_gg(double E);

// Energy loss rate due to inverse Compton scattering
double func_bIC(double Urad, double Trad, double E);