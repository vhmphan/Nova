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
const double me=0.510998e6;// eV
const double sigma_sb=3.5394474508e7;// erg cm^-2 s^-1 K^-4
const double mp=938.272e6;// eV
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
    Rsh=Rmin+vsh0*t;
    if(t>tST){
        Rsh=Rmin+vsh0*tST*(pow(t/tST,1.0-alpha)-alpha)/(1.0-alpha);
    }

    return Rsh*86400.0*6.68e-9;// au
}

// Density profiel of the red giant wind
double func_rho(double *pars_nova, double r){
// Mdot (Msol/yr), vwind (km/s), and r (au)    

    double Mdot=pars_nova[3], vwind=pars_nova[4];
    Mdot*=1.989e33/(365.0*86400.0);// g/s 
    vwind*=1.0e5;// cm/s
    r*=1.496e13;// cm

    double rho=Mdot/(4.0*pi*vwind*pow(r,2));

    return rho;// g/cm^3
}

// Swept-up mass of the nova shock
double func_MSU(double *pars_nova, double t){
// Mdot (Msol/yr) and vwind (km/s)

    double MSU, Mdot=pars_nova[3], vwind=pars_nova[4], Rmin=pars_nova[5];
    MSU=Mdot*(1.989e33/(365.0*86400.0))*(func_Rsh(pars_nova,t)-Rmin)*1.496e13/(vwind*1.0e5);

    return MSU;// g
}

// Maximum energy of accelerated protons
double func_Emax(double *pars_nova, double t){

    double vsh, Rsh, rho;
    vsh=func_vsh(pars_nova,t)*1.0e5;// cm/s 
    Rsh=func_Rsh(pars_nova,t);// au
    rho=func_rho(pars_nova,Rsh);// g/cm^3
    Rsh*=1.496e13;// cm

    double B2=sqrt(11.0*rho*pow(vsh*xip,2))*1.0e6;// uG
    double eta=0.05;// -> FRaction of shock radius to determine Emax
    double Emax=1.0e15*eta*Rsh*vsh/(3.4e28/B2);// eV

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
            fEp[i][j]=3.0*xip*rho*pow(vsh,2)*6.242e11*pow(arr_p[j]/mp,2.0-delta)/(mp*mp*arr_beta[j]*Ialpha);
        }
    }
    
    return fEp;
}

int main(){

    ofstream output1, output2;
    output1.open("profile.dat");
    output2.open("fEp.dat");

    // Parameters for RS Ophiuchi from Diesing 2023
    double vsh0=4500.0, tST=3.0, Mdot=5.0e-7, vwind=30.0, alpha=0.43, Rmin=3.0, xip=0.1, delta=4.3;// v0 (km/s), t0 (day), Mdot (Msol/yr), vwind (km/s), alpha, Rmin (au), xip, delta
    double *pars_nova=new double[8];
    pars_nova[0]=vsh0, pars_nova[1]=tST, pars_nova[2]=alpha, pars_nova[3]=Mdot, pars_nova[4]=vwind, pars_nova[5]=Rmin, pars_nova[6]=xip, pars_nova[7]=delta;
    
    double tmin=0.0, tmax=100.0, dt=0.01;// day
    int Nt=int((tmax-tmin)/dt)+1;
    double Emin=1.0e8, Emax=pow(10.0,int(log10(1.0e2*func_Emax(pars_nova,0.0)))), dlogE=0.01;// eV
    int NE=int(log10(Emax/Emin)/dlogE)+1;

    double *arr_t=new double[Nt];
    double *arr_E=new double[NE];

    for(int i=0;i<Nt;i++){
        arr_t[i]=i*dt;

        output1 << arr_t[i];// day 
        output1 << " " << func_vsh(pars_nova,arr_t[i]);// km/s
        output1 << " " << func_Rsh(pars_nova,arr_t[i]);// au
        output1 << " " << func_rho(pars_nova,func_Rsh(pars_nova,arr_t[i]))*1.0e-6/mH;// cm^-3
        output1 << " " << func_Emax(pars_nova,arr_t[i])*1.0e-9;// GeV
        output1 << " " << func_Ialpha(delta,Emin,func_Emax(pars_nova,arr_t[i]));
        output1 << endl;
    }

    for(int j=0;j<NE;j++){
        arr_E[j]=Emin*pow(10.0,j*dlogE);
    }

    double **fEp=new double*[Nt];
    for(int i=0;i<Nt;i++){
        fEp[i]=new double[NE];
    }

    fEp=func_fEp(pars_nova,arr_E,NE,arr_t,Nt);

    for(int i=0;i<Nt;i++){
        for(int j=0;j<NE;j++){
            output2 << arr_t[i] << " " << arr_E[j] << " " << fEp[i][j] << endl;
        }
    }

    // Gamma-ray spectrum 
    double Ebgmin=1.0e-4, Ebgmax=1.0e1, dlogEbg=0.001;
    int NEbg=int(log10(Ebgmax/Ebgmin)/dlogEbg)+1;
    double *arr_Ebg=new double[NEbg], *fOPT=new double[NEbg];

    for(int k=0;k<NEbg;k++){
        arr_Ebg[k]=Ebgmin*pow(10.0,k*dlogEbg);
    }

    // Differential number denisty of background photons 
    double UOPT, TOPT=1.0e4;// UOPT (eV/cm^3) and TOPT (K) 

    fOPT=func_fEtd(UOPT,TOPT,0.0,arr_Ebg,NEbg);// eV^-1 cm^-3

    output1.close();
    output2.close();

    return 0;
}
