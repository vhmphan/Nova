#include <iostream>
#include <complex>
#include <math.h>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdlib.h>
#include "pack_Xrad.h"

using namespace std;

const double pi=3.14159265359;
const double kB=8.6173324e-5;// eV/K
const double me=0.510998e6;// eV
const double sigma_sb=3.5394474508e7;// erg cm^-2 s^-1 K^-4
const double mp=938.272e6;// eV
const double mpCGS=1.67262192e-24;// g
const double qe=1.602176634e-19;// SI unit
const double mH=1.67e-24;// g
const double xip=0.1;// -> Acceleration efficiency of protons

// Nova shock speed
double func_vsh(double *pars_nova, double t){
// vsh0 (km/s), tST (day), and t(day)

    double vsh0=pars_nova[0], tST=pars_nova[1], alpha=pars_nova[2], Rsh;
    double vsh=vsh0;
    if(t>tST){
        vsh=vsh0*pow(t/tST,-alpha);
    }

    return vsh;// km/s
}

// Nova shock radius
double func_Rsh(double *pars_nova, double t){
// vsh0 (km/s), tST (day), Rmin (au), and t(day)

    double vsh0=pars_nova[0], tST=pars_nova[1], alpha=pars_nova[2], Rmin=pars_nova[5], Rsh;
    // Rsh=Rmin+vsh0*t;
    // if(t>tST){
    //     Rsh=Rmin+vsh0*tST*(pow(t/tST,1.0-alpha)-alpha)/(1.0-alpha);
    // }
    Rsh=vsh0*t;
    if(t>tST){
        Rsh=vsh0*tST*(pow(t/tST,1.0-alpha)-alpha)/(1.0-alpha);
    }

    return Rsh*86400.0*6.68e-9;// au
}

// Density profile of the red giant wind
double func_rho(double *pars_nova, double r){
// Mdot (Msol/yr), vwind (km/s), and r (au)    

    double Mdot=pars_nova[3], vwind=pars_nova[4], Rmin=pars_nova[5];
    Mdot*=1.989e33/(365.0*86400.0);// g/s 
    vwind*=1.0e5;// cm/s
    Rmin*=1.496e13;// cm
    r*=1.496e13;// cm

    double rho=Mdot/(4.0*pi*vwind*pow(Rmin+r,2));

    return rho;// g/cm^3
}

// Swept-up mass of the nova shock
double func_MSU(double *pars_nova, double t){
// Mdot (Msol/yr) and vwind (km/s)

    double MSU=0.0, Rsh=func_Rsh(pars_nova,t), Mdot=pars_nova[3], vwind=pars_nova[4], Rmin=pars_nova[5];
    MSU=Mdot*(1.989e33/(365.0*86400.0))*(Rsh-Rmin*atan(Rsh/Rmin))*1.496e13/(vwind*1.0e5);
    
    return MSU;// g
}

// Magnetic field strength from the RG wind


// Maximum energy of accelerated protons
double func_Emax(double *pars_nova, double t){

    double Rmin=pars_nova[5];
    Rmin*=1.496e13;// cm

    double vsh, Rsh, rho;
    vsh=func_vsh(pars_nova,t)*1.0e5;// cm/s 
    Rsh=func_Rsh(pars_nova,t);// au
    rho=func_rho(pars_nova,Rsh);// g/cm^3
    Rsh*=1.496e13;// cm
   
    // double B2=10.0e6*pow((Rmin+Rsh)/(0.35*1.496e13),-2);
    double B2=sqrt(11.0*pi*rho*pow(vsh*xip,2))*1.0e6;// uG
    double eta=0.05;// -> Fraction of shock radius to determine Emax
    double Emax=1.0e15*eta*Rsh*vsh*B2/3.4e28;// eV

    cout << t << " " << B2*1.0e-6 << " " << Emax << endl;

    double Mdot=pars_nova[3], vwind=pars_nova[4];
    Mdot*=1.989e33/(365.0*86400.0);// g/s 
    vwind*=1.0e5;// cm/s

    Emax=4.8032e-10*3.0e10*0.01*sqrt(Mdot/vwind)*pow(vsh/3.0e10,2)/20.0;
    Emax*=6.243e11;// eV

    return Emax;
}

double func_Ialpha(double delta, double Emin, double Emax){
    
    double x, dx, Ialpha=0.0, xmin=sqrt(pow(Emin+mp,2)-mp*mp)/mp, xmax=sqrt(pow(1.0e2*Emax+mp,2)-mp*mp)/mp;
    x=xmin;    
    while(x<xmax){
        dx=min(0.01*x,xmax-x);
        Ialpha+=pow(x,4.0-delta)*exp(-x*mp/Emax)*dx/sqrt(1.0+x*x);

        x+=dx;
    }

    return Ialpha;
}

// Spectrum of accelerated protons
double **func_fEp(double *pars_nova, double *arr_E, int NE, double *arr_t, int Nt){

    double Emin=1.0e8;// eV -> Assume the minimum proton energy to be~10^8 eV
    double delta=pars_nova[7];
    double vsh, Rsh, rho, Emax, Ialpha;
    double *arr_p=new double[NE];
    double *arr_beta=new double[NE];
    for(int j=0;j<NE;j++){
        arr_p[j]=sqrt(pow(arr_E[j]+mp,2)-mp*mp);
        arr_beta[j]=arr_p[j]/(arr_E[j]+mp);
    }

    double **fEp=new double*[Nt];
    for(int i=0;i<Nt;i++){
        fEp[i]=new double[NE];
        vsh=func_vsh(pars_nova,arr_t[i])*1.0e5;// cm/s
        Rsh=func_Rsh(pars_nova,arr_t[i]);// au
        rho=func_rho(pars_nova,Rsh);// g/cm^3
        Rsh*=1.496e13;// cm
        Emax=func_Emax(pars_nova,arr_t[i]);// eV
        Ialpha=func_Ialpha(delta,Emin,Emax);

        for(int j=0;j<NE;j++){
            fEp[i][j]=3.0*xip*rho*pow(vsh,2)*6.242e11*pow(arr_p[j]/mp,2.0-delta)*exp(-pow(arr_p[j]/Emax,0.5))/(mp*mp*arr_beta[j]*Ialpha);
        }
    }
    
    return fEp;// eV^-1 cm^-3
}

double func_LOPT(double t){

    double LOPT=0.0;
    if(t>=1.0){
        LOPT=1.3e39*pow(t,-0.28)/(t+0.6);
    }
	
    return LOPT;// erg s^-1
}

int main(){

    ofstream output1, output2, output3;
    output1.open("profile.dat");
    output2.open("fEp.dat");
    output3.open("gamma.dat");

    // Parameters for RS Ophiuchi from Diesing 2023
    double Ds=1.4e3*3.086e18;// cm
    double Mej=2.0e-7*1.989e33;// g
    double UOPT, TOPT=kB*1.0e4;// UOPT (eV/cm^3) and TOPT (eV)  
    double vsh0=4500.0, tST=3.0, Mdot=5.0e-7, vwind=30.0, alpha=0.43, Rmin=1.48, xip=0.1, delta=4.2;// v0 (km/s), t0 (day), Mdot (Msol/yr), vwind (km/s), alpha, Rmin (au), xip, delta
    double *pars_nova=new double[8];
    pars_nova[0]=vsh0, pars_nova[1]=tST, pars_nova[2]=alpha, pars_nova[3]=Mdot, pars_nova[4]=vwind, pars_nova[5]=Rmin, pars_nova[6]=xip, pars_nova[7]=delta;
    
    double tmin=0.0, tmax=10.0, dt=0.1;// day
    int Nt=int((tmax-tmin)/dt)+1;
    double Emin=1.0e8, Emax=1.0e14, dlogE=0.01;// eV
    int NE=int(log10(Emax/Emin)/dlogE)+1;
    double Rsh;

    cout << "Number of time steps Nt = " << Nt << endl;

    double *arr_t=new double[Nt];
    double *arr_E=new double[NE];

    for(int i=0;i<Nt;i++){

        arr_t[i]=i*dt;
        Rsh=func_Rsh(pars_nova,arr_t[i]);

        // cout << arr_t[i] << " " << func_rho(pars_nova,func_Rsh(pars_nova,arr_t[i]))*1.0e-6/mpCGS << endl;

        output1 << arr_t[i];// day 
        output1 << " " << func_vsh(pars_nova,arr_t[i]);// km/s
        output1 << " " << func_Rsh(pars_nova,arr_t[i]);// au
        output1 << " " << func_rho(pars_nova,func_Rsh(pars_nova,arr_t[i]))*1.0e-6/mH;// cm^-3
        // output1 << " " << func_rho(pars_nova,func_Rsh(pars_nova,arr_t[i]))*pow(func_vsh(pars_nova,arr_t[i]),2);// cm^-3
        output1 << " " << func_Emax(pars_nova,arr_t[i])*1.0e-12;// TeV
        output1 << " " << 10.0*pow(sqrt(Rmin*Rmin+Rsh*Rsh)/(0.35),-2);
        // output1 << " " << func_Ialpha(delta,Emin,func_Emax(pars_nova,arr_t[i]));
        // output1 << " " << func_MSU(pars_nova,arr_t[i]);
        output1 << endl;
    }

    for(int j=0;j<NE;j++){
        arr_E[j]=Emin*pow(10.0,j*dlogE);
    }

    double **fEp=new double*[Nt];
    for(int i=0;i<Nt;i++){
        fEp[i]=new double[NE];
    }

    fEp=func_fEp(pars_nova,arr_E,NE,arr_t,Nt);// eV^-1 cm^-3

    for(int i=0;i<Nt;i++){
        for(int j=0;j<NE;j++){
            output2 << arr_t[i] << " " << arr_E[j] << " " << fEp[i][j] << endl;
        }
    }

    // Gamma-ray spectrum 
    double dE, vp_p, phi_PPI, tau_gg, abs;
    double Eg, Egmax=1.0e14, dEg;
    double Ebgmin=1.0e-3, Ebgmax=1.0e2, dlogEbg=0.001;
    int NEbg=int(log10(Ebgmax/Ebgmin)/dlogEbg)+1;
    double *arr_Ebg=new double[NEbg], *fOPT=new double[NEbg];

    for(int k=0;k<NEbg;k++){
        arr_Ebg[k]=Ebgmin*pow(10.0,k*dlogEbg);
    }

    for(int i=0;i<Nt;i++){

        cout << "i = " << i+1 << "/" << Nt << " --> t = " << arr_t[i] << " day" << endl;

        // Differential number denisty of background photons 
        UOPT=func_LOPT(arr_t[i])*6.242e11/(4.0*pi*pow(func_Rsh(pars_nova,arr_t[i])*1.496e13,2)*3.0e10);// eV cm ^⁻3
        fOPT=func_fEtd(UOPT,TOPT,0.0,arr_Ebg,NEbg);// eV^-1 cm^-3
    
        Eg=1.0e8;// eV
        while(Eg<Egmax){
            dEg=min(0.1*Eg,Egmax-Eg);

            tau_gg=0.0;
            for(int k=0;k<NEbg-1;k++){
                tau_gg+=fOPT[k]*func_sigma_gg(Eg,arr_Ebg[k])*func_Rsh(pars_nova,arr_t[i])*1.496e13*(arr_Ebg[k+1]-arr_Ebg[k]);
            }

            phi_PPI=0.0;
            for(int j=0;j<NE-1;j++){
                dE=arr_E[j+1]-arr_E[j];

                // Speed of particles 
                vp_p=sqrt(pow(arr_E[j]+mp,2)-mp*mp)*3.0e10/(arr_E[j]+mp);// cm/s
                
                if(arr_E[j]>=2.0e9){
                    // Pi0 decay
                    phi_PPI+=dE*(fEp[i][j]*vp_p)*func_enhancement(arr_E[j])*func_d_sigma_g(arr_E[j],Eg);
                }
            }

            phi_PPI*=(Mej+func_MSU(pars_nova,arr_t[i]))/(4.0*pi*Ds*Ds*mpCGS);// eV^-1 cm^-2 s^-1
            
            if(tau_gg!=0.0){
                abs=(1.0-exp(-tau_gg))/tau_gg;
            }
            else{
                abs=1.0;
            }

            output3 << arr_t[i]; 
            output3 << " " << Eg*1.0e-9; 
            output3 << " " << Eg*Eg*phi_PPI*1.6022e-12;
            output3 << " " << Eg*Eg*phi_PPI*1.6022e-12*exp(-tau_gg);
            output3 << " " << tau_gg;
            output3 << endl;

            Eg+=dEg;
        }
    }

    output1.close();
    output2.close();
    output3.close();

    return 0;
}
