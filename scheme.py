import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Set limits and remove axis
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
ax.set_xlim(0.0, 0.8)
ax.set_ylim(-0.4, 0.4)
ax.axis('off')

# Create a grid of points
x = np.linspace(-1, 1, 20000)
y = np.linspace(-1, 1, 20000)
X, Y = np.meshgrid(x, y)

# Position of the Red Giant (RG)
rg_position = (-0.4, 0)

# Compute the radial distance from the RG center
R = np.sqrt((X - rg_position[0])**2 + (Y - rg_position[1])**2)

# Create a density map that scales as 1/R^2 centered on RG
density = 1 / (R**2 + 0.05)  # Added 0.1 to avoid division by zero and excessive intensity near RG

# Normalize the density map to [0, 1] range
density_normalized = density / np.max(density)

# Plot the density background
ax.imshow(density_normalized, extent=(-1, 1, -1, 1), origin='lower', cmap='Reds', alpha=0.7)

Rshock = 0.1

# Define the x and y values for the hourglass shape
y1 = np.sqrt(Rshock - (x-0.4)**2)
y2 = -np.sqrt(Rshock - (x-0.4)**2)

# Plot the hourglass shape
ax.fill_between(x, y1, y2, color='skyblue', alpha=0.5)

# # Add a boundary line to the hourglass shape
ax.plot(x, y1, color='orange', linewidth=4)
ax.plot(x, y2, color='orange', linewidth=4)

# Add the RG and WD circles
rg_circle = plt.Circle(rg_position, 0.1, color='salmon', zorder=5)
wd_circle = plt.Circle((0.4, 0), 0.05, color='dimgray', zorder=5)

ax.add_patch(rg_circle)
ax.add_patch(wd_circle)

# Add the accretion flow arrow
# ax.arrow(-0.25, 0, 0.5, 0, head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=1.5)

# Add the accretion flow label
# ax.text(0, -0.1, 'accretion flow', color='black', fontsize=12, ha='center')

# Add the RG and WD labels
# ax.text(-0.4, -0.2, 'RG', color='black', fontsize=12, ha='center')
# ax.text(0.4, -0.15, 'WD', color='black', fontsize=12, ha='center')
ax.text(0.4, 0.0, 'WD', color='white', fontsize=12, ha='center')

# if(Rshock>0.0):
#     # Add the RG wind and shock labels and arrows
#     ax.text(-0.9, 0.5, 'RG wind', color='black', fontsize=12)
#     ax.text(0.9, 0.5, 'shock', color='black', fontsize=12, ha='right')
#     ax.arrow(-0.75, 0.42, 0.15, -0.2, head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=1.5)
#     ax.arrow(0.75, 0.42, -0.05-(0.1-Rshock), -0.1-0.5*(0.1-Rshock), head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=1.5)

plt.savefig('scheme_%.2f.png' % Rshock)


# Prepare data from HESS and FERMI
t_HESS_raw, flux_HESS_raw=np.loadtxt('Data/data_time_gamma_HESS_raw.dat',unpack=True,usecols=[0,1])
t_HESS_raw=t_HESS_raw-0.25 # Data are chosen at different time orgin than model
t_HESS_raw=t_HESS_raw.reshape((int(len(flux_HESS_raw)/5),5))
flux_HESS_raw=flux_HESS_raw.reshape((int(len(flux_HESS_raw)/5),5))
xerr_HESS_raw=t_HESS_raw[:,0]-t_HESS_raw[:,1]
yerr_HESS_raw=flux_HESS_raw[:,0]-flux_HESS_raw[:,3]
t_HESS_raw=t_HESS_raw[:,0]
flux_HESS_raw=flux_HESS_raw[:,0]
# xerr_HESS_raw=np.array([t_HESS_raw[:,0]-t_HESS_raw[:,1],t_HESS_raw[:,2]-t_HESS_raw[:,0]])
# yerr_HESS_raw=np.array([flux_HESS_raw[:,0]-flux_HESS_raw[:,3],flux_HESS_raw[:,4]-flux_HESS_raw[:,0]])

t_FERMI_raw, flux_FERMI_raw=np.loadtxt('Data/data_time_gamma_FERMI_raw.dat',unpack=True,usecols=[0,1])
t_FERMI_raw=t_FERMI_raw-0.25 # Data are chosen at different time orgin than model
t_FERMI_raw=t_FERMI_raw.reshape((int(len(flux_FERMI_raw)/5),5))
flux_FERMI_raw=flux_FERMI_raw.reshape((int(len(flux_FERMI_raw)/5),5))
xerr_FERMI_raw=t_FERMI_raw[:,0]-t_FERMI_raw[:,1]
yerr_FERMI_raw=flux_FERMI_raw[:,0]-flux_FERMI_raw[:,3]
t_FERMI_raw=t_FERMI_raw[:,0]
flux_FERMI_raw=flux_FERMI_raw[:,0]
# xerr_FERMI_raw=np.array([t_FERMI_raw[:,0]-t_FERMI_raw[:,1],t_FERMI_raw[:,2]-t_FERMI_raw[:,0]])
# yerr_FERMI_raw=np.array([flux_FERMI_raw[:,0]-flux_FERMI_raw[:,3],flux_FERMI_raw[:,4]-flux_FERMI_raw[:,0]])

fs=22
Ds=1.4e3 # pc


# Heaviside
def func_Heaviside(x):

    return 0.5*(1.0+np.tanh(10.0*x))

# Optical luminosiy function of the nova
def func_LOPT(t):
# t (day)

    mask=(t==0.25)

    LOPT=2.5e36*(1.0-func_Heaviside(t-0.9))+func_Heaviside(t-0.9)*(1.3e39*pow(abs(t-0.25),-0.28)/(abs(t+0.35)))
    LOPT[mask]=2.5e36

    return LOPT;# erg s^-1


# Plot time profile of gamma-ray integrated flux
def plot_time_gamma():

    fig=plt.figure(figsize=(12, 8))
    ax=plt.subplot(111)

    # dlogEg=np.log10(Eg[1]/Eg[0])

    # jmin_FLAT=int(np.log10(0.1e9/Eg[0])/dlogEg)
    # jmax_FLAT=int(np.log10(100.0e9/Eg[0])/dlogEg)
    # jmin_HESS=int(np.log10(250.0e9/Eg[0])/dlogEg)
    # jmax_HESS=int(np.log10(2500.0e9/Eg[0])/dlogEg)

    # print("FERMI band: ",Eg[jmin_FLAT]*1.0e-9,"-",Eg[jmax_FLAT]*1.0e-9,"GeV")
    # print("HESS band:  ",Eg[jmin_HESS]*1.0e-9,"-",Eg[jmax_HESS]*1.0e-9,"GeV")

    ax.errorbar(t_HESS_raw,flux_HESS_raw,yerr=yerr_HESS_raw,xerr=xerr_HESS_raw,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='red',markeredgecolor='black',markersize=10,label=r'${\rm HESS}$')
    ax.errorbar(t_FERMI_raw,flux_FERMI_raw,yerr=yerr_FERMI_raw,xerr=xerr_FERMI_raw,fmt='o',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='green',markeredgecolor='black',markersize=10,label=r'${\rm FERMI\,(\times 10^{-3})}$')
    ax.plot(t_HESS_raw,flux_HESS_raw,'r-',linewidth=4)
    ax.plot(t_FERMI_raw,flux_FERMI_raw,'g-',linewidth=4)

    t_data, LOPT_data=np.loadtxt("Data/LOPT.dat",unpack=True,usecols=[0,1])
    ax.errorbar(t_data-0.25,1.0e-4*LOPT_data/(4.0*np.pi*(Ds*3.086e18)**2),yerr=LOPT_data*0.0,xerr=t_data*0.0,fmt='s',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='orange',markeredgecolor='black',markersize=15,label=r'${\rm Optical\,(\times 10^{-4})}$')

    t=np.linspace(-10,30,30000)
    ax.plot(t,1.0e-4*func_LOPT(t)/(4.0*np.pi*(Ds*3.086e18)**2),'-',color='orange',linewidth=4)

    # ax.set_xlim(1.0e-1,5.0e1)
    # ax.set_ylim(5.0e-13,5.0e-10)
    # ax.set_xscale('log')
    ax.set_xlim(-5.0,10.0)
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'${\rm Integrated\, Flux} \, ({\rm erg\, cm^{-2}\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    # ax.legend(loc='upper left', prop={"size":fs}, ncols=2)
    ax.legend(loc='upper right', prop={"size":fs}, ncols=1)
    ax.grid(linestyle='--')

    plt.savefig('Results/new_fg_time_gamma.png')
    plt.close()

def plot_time_OPT():

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    t_data, LOPT_data=np.loadtxt("Data/LOPT.dat",unpack=True,usecols=[0,1])
    ax.errorbar(t_data-0.25,LOPT_data,yerr=LOPT_data*0.0,xerr=t_data*0.0,fmt='s',capsize=5,ecolor='black',elinewidth=2,markerfacecolor='orange',markeredgecolor='black',markersize=15,label=r'${\rm Cheung\, et\, al.\, 2022}$')

    t=np.linspace(-5,10,30000)
    ax.plot(t,func_LOPT(t),'-',color='black',linewidth=4,label=r'${\rm Fit}$')

    # ax.set_xlim(1.0e-1,5.0e1)
    ax.set_ylim(1.0e36,2.0e39)
    # ax.set_xscale('log')
    ax.set_xlim(-4.0,10.0)
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\, {\rm (day)}$',fontsize=fs)
    ax.set_ylabel(r'$L_{\rm OPT} \, ({\rm erg\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    # ax.legend(loc='upper left', prop={"size":fs}, ncols=2)
    ax.legend(loc='upper right', prop={"size":fs}, ncols=1)
    ax.grid(linestyle='--')

    plt.savefig('Results/fg_LOPT.png')
    plt.close()

plot_time_gamma()
plot_time_OPT()