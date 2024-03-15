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
const double qeCGS=4.8032e-10;// CGS unit 

// Smooth Heaviside function
double func_Heaviside(double x){

    double H=0.5*(1+tanh(10*x));

    return H;
}

// Nova shock speed
double func_vsh(double *pars_nova, double t){
// vsh0 (km/s), tST (day), and t(day)

    double vsh0=pars_nova[0], tST=pars_nova[1], alpha=pars_nova[2], ter=pars_nova[9], Rsh;
    double vsh=vsh0;

    if(t>ter){    
        vsh=vsh0;
        if(t>tST){
            vsh=vsh0*pow(t/tST,-alpha);
        }
    }

    return vsh;// km/s
}

// Nova shock radius
double func_Rsh(double *pars_nova, double t){
// vsh0 (km/s), tST (day), Rmin (au), and t(day)

    double vsh0=pars_nova[0], tST=pars_nova[1], alpha=pars_nova[2], Rmin=pars_nova[5], ter=pars_nova[9];
    double Rsh=0.0;

    if(t>ter){
        Rsh=vsh0*(t-ter);
        if(t>tST){
            Rsh=-vsh0*ter;
            Rsh+=vsh0*tST*(pow((t-ter)/tST,1.0-alpha)-alpha)/(1.0-alpha);
        }
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

    double MSU=0.0, Rsh=func_Rsh(pars_nova,t), Mdot=pars_nova[3], vwind=pars_nova[4], Rmin=pars_nova[5], ter=pars_nova[9];
    if(t>ter){
        MSU=Mdot*(1.989e33/(365.0*86400.0))*(Rsh-Rmin*atan(Rsh/Rmin))*1.496e13/(vwind*1.0e5);
    }

    return MSU;// g
}

// Maximum energy of accelerated protons from B2 with Bell instability 
double func_Emax_Bell(double *pars_nova, double t){

    double Rmin=pars_nova[5]*1.496e13;// cm
    double xip=pars_nova[6];

    double vsh, Rsh, rho;
    vsh=func_vsh(pars_nova,t)*1.0e5;// cm/s 
    Rsh=func_Rsh(pars_nova,t)*1.496e13;// cm
    rho=func_rho(pars_nova,Rsh/1.496e13);// g/cm^3
   
    double B2=sqrt(11.0*pi*rho*pow(vsh*xip,2))*1.0e6;// uG
    double eta=0.05;// -> Fraction of shock radius to determine Emax
    double Emax=1.0e15*eta*Rsh*vsh*B2/3.4e28;// eV

    return Emax;
}

// Maximum energy of accelerated protons from B2 estimated as in Tatischeff & Hernanz 2007
double func_Emax_TH07(double *pars_nova, double t){

    double Rmin=pars_nova[5]*1.496e13;// cm
    double xip=pars_nova[6];

    double vsh, Rsh, rho;
    vsh=func_vsh(pars_nova,t)*1.0e5;// cm/s 
    Rsh=func_Rsh(pars_nova,t)*1.496e13;// cm
    rho=func_rho(pars_nova,Rsh/1.496e13)/mpCGS;// cm^-3
   
    double B2=sqrt(8.0*pi*rho*kB*1.0e4*1.6022e-12)*1.0e6;// uG
    double eta=0.05;// -> Fraction of shock radius to determine Emax
    double Emax=1.0e15*eta*Rsh*vsh*B2/3.4e28;// eV

    // cout << t << " " << Emax << endl;

    return Emax;
}

// Maximum energy of accelerated protons from confinement limit
double func_Emax_conf(double *pars_nova, double t){

    double Mdot=pars_nova[3], vwind=pars_nova[4];
    Mdot*=1.989e33/(365.0*86400.0);// g/s 
    vwind*=1.0e5;// cm/s

    double vsh=func_vsh(pars_nova,t)*1.0e5;// cm/s 
    double Emax=6.243e11*4.8032e-10*3.0e10*0.01*sqrt(Mdot/vwind)*pow(vsh/3.0e10,2)/20.0;// eV

    return Emax;
}

// Maximum energy of accelerated protons
double *func_Emax(double *pars_nova, double *arr_t, int Nt){

    double tST=pars_nova[1];
    double Rmin=pars_nova[5]*1.496e13;// cm
    double xip=pars_nova[6];
    double BRG=pars_nova[10];
    double TOPT=pars_nova[11];// eV

    double vsh, Rsh, rho, B2, B2_bkgr, B2_Bell, Emax_conf;
    // double B2=sqrt(8.0*pi*rho*TOPT*1.6022e-12)*1.0e6;// uG
    // double eta=0.05;// -> Fraction of shock radius to determine Emax
    double *arr_Emax=new double[Nt];//1.0e15*eta*Rsh*vsh*B2/3.4e28;// eV

    double dt=arr_t[1]-arr_t[0];
    arr_Emax[0]=0.0;
    for(int i=1;i<Nt;i++){
        vsh=func_vsh(pars_nova,arr_t[i])*1.0e5;// cm/s 
        Rsh=func_Rsh(pars_nova,arr_t[i])*1.496e13;// cm
        rho=func_rho(pars_nova,Rsh/1.496e13);// g/cm^3
        B2_bkgr=BRG*pow(sqrt(Rmin*Rmin+Rsh*Rsh)/(0.35*1.496e13),-2);// G
        B2_Bell=sqrt(11.0*pi*rho*pow(vsh*xip,2));// G
        B2=B2_bkgr;//*func_Heaviside(tST-arr_t[i])+B2_Bell*func_Heaviside(arr_t[i]-tST);

        arr_Emax[i]=arr_Emax[i-1]+(dt*86400.0*qeCGS*B2*pow(vsh,2))*6.242e+11/(2.0*pi*3.0e10);
        Emax_conf=func_Emax_conf(pars_nova,arr_t[i]);
        if(arr_Emax[i]>Emax_conf){
            arr_Emax[i]=Emax_conf;
        }    
    }

    return arr_Emax;
}

// Function for normalization of the acceleration spectrum in momentum
double func_Ialpha_p(double delta, double epsilon, double Emin, double Emax){
    
    double x, dx, Ialpha_p=0.0, xmin=sqrt(pow(Emin+mp,2)-mp*mp)/mp, xmax=sqrt(pow(1.0e2*Emax+mp,2)-mp*mp)/mp;
    x=xmin;    
    while(x<xmax){
        dx=min(0.01*x,xmax-x);
        Ialpha_p+=pow(x,4.0-delta)*exp(-pow(x*mp/Emax,epsilon))*dx/sqrt(1.0+x*x);

        x+=dx;
    }

    return Ialpha_p;
}

// Cumulative spectrum of accelerated protons
double **func_NEp_p(double *pars_nova, double *arr_E, int NE, double *arr_t, int Nt){

    double Emin=arr_E[0];// eV -> Assume the minimum proton energy to be~10^8 eV
    double xip=pars_nova[6];
    double delta=pars_nova[7];
    double epsilon=pars_nova[8];
    double ter=pars_nova[9];
    double vsh, Rsh, rho, Emax, Ialpha_p;
    double *arr_Emax=new double[Nt];
    arr_Emax=func_Emax(pars_nova,arr_t,Nt);
    double *arr_p=new double[NE];
    double *arr_beta=new double[NE];
    for(int j=0;j<NE;j++){
        arr_p[j]=sqrt(pow(arr_E[j]+mp,2)-mp*mp);
        arr_beta[j]=arr_p[j]/(arr_E[j]+mp);
    }

    double **NEp=new double*[Nt];
    for(int i=0;i<Nt;i++){
        NEp[i]=new double[NE];
        vsh=func_vsh(pars_nova,arr_t[i])*1.0e5;// cm/s
        Rsh=func_Rsh(pars_nova,arr_t[i])*1.496e13;// cm
        rho=func_rho(pars_nova,Rsh/1.496e13);// g/cm^3
        Emax=arr_Emax[i];// eV
        Ialpha_p=func_Ialpha_p(delta,epsilon,Emin,Emax);
        
        if(arr_t[i]<=ter){
            for(int j=0;j<NE;j++){
                NEp[i][j]=0.0;
            }
        }
        else{
            for(int j=0;j<NE;j++){
                NEp[i][j]=NEp[i-1][j]+((arr_t[i]-arr_t[i-1])*86400.0)*3.0*pi*xip*rho*pow(Rsh,2)*pow(vsh,3)*6.242e11*pow(arr_p[j]/mp,2.0-delta)*exp(-pow(arr_p[j]/Emax,epsilon))/(mp*mp*arr_beta[j]*Ialpha_p);
            }
        }
    }
    
    return NEp;// eV^-1 
}

// Function for normalization of the acceleration spectrum in energy
double func_Ialpha_E(double delta, double epsilon, double Emin, double Emax){
    
    double x, dx, Ialpha_E=0.0, xmin=Emin/mp, xmax=1.0e2*Emax/mp;
    x=xmin;    
    while(x<xmax){
        dx=min(0.01*x,xmax-x);
        Ialpha_E+=pow(x,3.0-delta)*exp(-pow(x*mp/Emax,epsilon))*dx;

        x+=dx;
    }

    return Ialpha_E;
}

// Cumulative spectrum of accelerated protons
double **func_NEp_E(double *pars_nova, double *arr_E, int NE, double *arr_t, int Nt){

    double Emin=arr_E[0];// eV -> Assume the minimum proton energy to be~10^8 eV
    double xip=pars_nova[6];
    double delta=pars_nova[7];
    double epsilon=pars_nova[8];
    double vsh, Rsh, rho, Emax, Ialpha_E;
    double *arr_Emax=new double[Nt];
    arr_Emax=func_Emax(pars_nova,arr_t,Nt);

    double **NEp=new double*[Nt];
    for(int i=0;i<Nt;i++){
        NEp[i]=new double[NE];
        vsh=func_vsh(pars_nova,arr_t[i])*1.0e5;// cm/s
        Rsh=func_Rsh(pars_nova,arr_t[i])*1.496e13;// cm
        rho=func_rho(pars_nova,Rsh/1.496e13);// g/cm^3
        Emax=arr_Emax[i];// eV
        Ialpha_E=func_Ialpha_E(delta,epsilon,Emin,Emax);
        
        if(i==0){
            for(int j=0;j<NE;j++){
                NEp[i][j]=0.0;
            }
        }
        else{
            for(int j=0;j<NE;j++){
                NEp[i][j]=NEp[i-1][j]+((arr_t[i]-arr_t[i-1])*86400.0)*2.0*pi*xip*rho*pow(Rsh,2)*pow(vsh,3)*6.242e11*pow(arr_E[j]/mp,2.0-delta)*exp(-pow(arr_E[j]/Emax,epsilon))/(mp*mp*Ialpha_E);
            }
        }
    }
    
    return NEp;// eV^-1 
}

// Luminosiy function of the nova
double func_LOPT(double t){

    // double LOPT=2.0e36;
    // LOPT+=func_Heaviside(x-0.5)*1.3e39*pow(t-0.25,-0.28)/(t+0.35);

    double LOPT;
    LOPT=2.5e36*(1.0-func_Heaviside(t-0.9))+func_Heaviside(t-0.9)*(1.3e39*pow(abs(t-0.25),-0.28)/(abs(t+0.35)));

    return LOPT;// erg s^-1
}

int main(){

    ifstream input;
    input.open("pars_RSOph21.dat");

    ofstream output1, output2, output3;
    output1.open("profile.dat");
    output2.open("fEp.dat");
    output3.open("gamma.dat");

    // Parameters for RS Ophiuchi outburst 2021
    double tmin, tmax, dt, ter;// day
    int scale_t;
    double BRG;// G
    double Emin, Emax, dlogE;// eV
    double Egmin, Egmax, dlogEg;// eV
    double Ds, Mej, TOPT;// kpc, solar mass, K
    double vsh0, tST, Mdot;// km/s, day, solar mass/yr
    double vwind, alpha, Rmin;// km/s
    double xip, delta, epsilon; 

    input >> tmin >> tmax >> dt;
    input >> ter >> scale_t >> BRG;
    input >> Emin >> Emax >> dlogE;
    input >> Egmin >> Egmax >> dlogEg;
    input >> Ds >> Mej >> TOPT;
    input >> vsh0 >> tST >> Mdot;
    input >> vwind >> alpha >> Rmin;
    input >> xip >> delta >> epsilon;

    double dt_print=scale_t*dt;// day
    double UOPT;// eV cm^-3
    double rho, vsh, Rsh, MSU, Bbg, nCR, UCR;

    Ds*=3.086e18;// cm
    Mej*=1.989e33;// g
    TOPT*=kB;// eV  

    double *pars_nova=new double[12];
    pars_nova[0]=vsh0, pars_nova[1]=tST, pars_nova[2]=alpha, pars_nova[3]=Mdot, pars_nova[4]=vwind, pars_nova[5]=Rmin, pars_nova[6]=xip, pars_nova[7]=delta, pars_nova[8]=epsilon, pars_nova[9]=ter, pars_nova[10]=BRG, pars_nova[11]=TOPT;
    
    int Nt=int((tmax-tmin)/dt)+1;
    int NE=int(log10(Emax/Emin)/dlogE)+1;
    int NEg=int(log10(Egmax/Egmin)/dlogEg)+1;

    cout << "Number of time steps Nt = " << Nt << endl;

    double *arr_t=new double[Nt];
    double *arr_E=new double[NE];
    double *arr_vp_p=new double[NE];
    double *arr_enhancement=new double[NE];
    double **arr_d_sigma_g=new double*[NE];

    for(int i=0;i<Nt;i++){
        arr_t[i]=i*dt;
    }

    for(int j=0;j<NE;j++){
        arr_E[j]=Emin*pow(10.0,j*dlogE);
        arr_vp_p[j]=sqrt(pow(arr_E[j]+mp,2)-mp*mp)/(arr_E[j]+mp);
        arr_enhancement[j]=func_enhancement(arr_E[j]);
        arr_d_sigma_g[j]=new double[NEg];
        for(int jg=0;jg<NEg;jg++){
            arr_d_sigma_g[j][jg]=func_d_sigma_g(arr_E[j],Egmin*pow(10.0,jg*dlogEg));
        }
    }

    double *arr_Emax=new double[Nt];
    arr_Emax=func_Emax(pars_nova,arr_t,Nt);// eV

    double **NEp=new double*[Nt];
    for(int i=0;i<Nt;i++){
        NEp[i]=new double[NE];
    }

    NEp=func_NEp_p(pars_nova,arr_E,NE,arr_t,Nt);// eV^-1 cm^-3

    for(int i=0;i<Nt;i++){

        vsh=func_vsh(pars_nova,arr_t[i]);// km/s
        Rsh=func_Rsh(pars_nova,arr_t[i]);// au
        rho=func_rho(pars_nova,func_Rsh(pars_nova,arr_t[i]));// g cm^-3
        Bbg=BRG*pow(sqrt(Rmin*Rmin+Rsh*Rsh)/(0.35),-2);
        UCR=xip*rho*pow(vsh*1.0e5,2);// erg/cm^3

        nCR=0.0;
        for(int j=0;j<NE-1;j++){
            nCR+=NEp[i][j]*(arr_E[j+1]-arr_E[j])/(pi*pow(Rsh*1.496e13,3)/3.0);// cm^-3
        }

        // cout << arr_t[i] << " " << nCR << endl;

        if(i%scale_t==0){
            for(int j=0;j<NE;j++){
                output2 << arr_t[i] << " " << arr_E[j] << " " << NEp[i][j]/(pi*pow(Rsh*1.496e13,3)/3.0) << endl;
            }
        }

        output1 << arr_t[i];// day 
        output1 << " " << vsh;// km/s
        output1 << " " << Rsh;// au
        output1 << " " << rho/mpCGS;// cm^-3
        output1 << " " << arr_Emax[i]*1.0e-12;// TeV
        output1 << " " << Bbg;// G
        // output1 << " " << func_Ialpha_p(delta,epsilon,Emin,func_Emax(pars_nova,arr_t[i]));
        output1 << " " << func_MSU(pars_nova,arr_t[i]);
        output1 << " " << arr_Emax[i]*1.0e-9;// GeV
        output1 << " " << func_Emax_Bell(pars_nova,arr_t[i])*1.0e-9;// GeV
        output1 << " " << func_Emax_conf(pars_nova,arr_t[i])*1.0e-9;// GeV
        output1 << " " << func_Emax_TH07(pars_nova,arr_t[i])*1.0e-9;// GeV
        output1 << " " << qeCGS*nCR*(vsh*1.0e5/3.0e10)*sqrt(4.0*pi/rho);// s^-1 
        output1 << " " << 8.9e-9*(rho/mpCGS)*pow(TOPT/kB,0.4);// s^-1
        output1 << endl;
    }

    // Gamma-ray spectrum 
    double dE, vp_p, phi_PPI, tau_gg;
    double Eg;// eV
    double Ebgmin=TOPT*1.0e-2, Ebgmax=TOPT*1.0e2, dlogEbg=0.001;// eV
    int NEbg=int(log10(Ebgmax/Ebgmin)/dlogEbg)+1;
    double *arr_Ebg=new double[NEbg], *fOPT=new double[NEbg];
    double **arr_sigma_gg=new double*[NEg];

    for(int k=0;k<NEbg;k++){
        arr_Ebg[k]=Ebgmin*pow(10.0,k*dlogEbg);
    }

    for(int jg=0;jg<NEg;jg++){
        arr_sigma_gg[jg]=new double[NEbg];
        for(int k=0;k<NEbg;k++){
            arr_sigma_gg[jg][k]=func_sigma_gg(Eg,arr_Ebg[k]);
        }
    }


    for(int i=0;i<Nt;i+=scale_t){

        // Differential number denisty of background photons 
        UOPT=func_LOPT(arr_t[i])*6.242e11/(4.0*pi*pow(func_Rsh(pars_nova,arr_t[i])*1.496e13,2)*3.0e10);// eV cm ^⁻3
        fOPT=func_fEtd(UOPT,TOPT,0.0,arr_Ebg,NEbg);// eV^-1 cm^-3
        Rsh=func_Rsh(pars_nova,arr_t[i])*1.496e13;// cm
        MSU=func_MSU(pars_nova,arr_t[i]);// g

        cout << "i = " << i+1 << "/" << Nt << " --> t = " << arr_t[i] << " day" << endl;

        Eg=Egmin;// eV
        for(int jg=0;jg<NEg;jg++){
            Eg=Egmin*pow(10.0,jg*dlogEg);

            tau_gg=0.0;
            for(int k=0;k<NEbg-1;k++){
                tau_gg+=fOPT[k]*arr_sigma_gg[jg][k]*Rsh*(arr_Ebg[k+1]-arr_Ebg[k]);
            }

            phi_PPI=0.0;
            if(Rsh>0){

                for(int j=0;j<NE-1;j++){
                    dE=arr_E[j+1]-arr_E[j];
                    
                    phi_PPI+=dE*(NEp[i][j]*arr_vp_p[j])*arr_enhancement[j]*arr_d_sigma_g[j][jg];
                }

                phi_PPI*=1.0/(pi*pow(Rsh,3)/3.0);
                phi_PPI*=(Mej+MSU)/(4.0*pi*Ds*Ds*mpCGS);// eV^-1 cm^-2 s^-1
            }

            output3 << arr_t[i];// day
            output3 << " " << Eg*1.0e-9;// GeV 
            output3 << " " << Eg*Eg*phi_PPI*1.6022e-12;// erg cm^-2 s^-1
            output3 << " " << Eg*Eg*phi_PPI*1.6022e-12*exp(-tau_gg);// erg cm^-2 s^-1
            output3 << " " << tau_gg;
            output3 << endl;
        }
    }

    input.close();
    output1.close();
    output2.close();
    output3.close();

    return 0;
}
